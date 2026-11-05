
# v0 128x128 + ILP（寄存器双缓冲）+ 输入预重排（preshuffle）
#
# v0 按 MFMA 排布直接读显存：lane l 读第 l % 16 行，一条 global_load_dwordx4 落在
# 16 行、每行 64 B，相邻 lane 不在同一行，每个 lane 的 16 B 各算一次 L1(TCP) 访问，
# 一条 load 要 64 次；64 个 lane 读连续 1 KiB 时只要 16 次（每次 64 B）。
#
# preshuffle()：在 host 端把 A[M, K] / B[N, K] 一次性重排，每个 16 行 x 32 K 的 MFMA
# 块按 lane 顺序存成连续 1 KiB（lane l = row + 16 * kg 的 16 B 在第 16 * l 字节）。
# kernel 只换了视图的 layout，MFMA 部分与 v0 相同。
#
# ILP（寄存器双缓冲）：先发下一个 K 步的 16 条 load，再做当前 K 步的 128 条 MFMA。
# 一次迭代做两个 K 步，两组操作数寄存器轮换，循环末尾不需要寄存器拷贝；
# rocdl.sched_* 把 load 排在 MFMA 前段（同 v2 §3.7），否则 LLVM 会把它们挪到使用处附近。
#
# 开关：PRESHUFFLE=False 读原始布局（只加 ILP）；ILP=False 用 v0 的循环（只预重排）。

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec


def preshuffle(X):
    """[R, K]（K 连续）-> [R/16, K/32, 4 kg, 16 row, 8 elem]：
    X[16 * rb + row, 32 * kb + 8 * kg + e] -> Y[rb, kb, kg, row, e]"""
    R, K = X.shape
    return X.view(R // 16, 16, K // 32, 4, 8).permute(0, 2, 3, 1, 4).contiguous()


def build_v0_preshuffle(BM, BN, OCCUPANCY, M, N, K, PRESHUFFLE=True, ILP=True):
    NK = K // 32                # K-steps: one 16 B load per lane covers 32 K
    assert NK % 2 == 0
    NI, NJ = BM // 16, BN // 16
    NA = NI * NJ
    grid_size = (M // BM, N // BN, 1)
    block_size = (64, 1, 1)

    @flyc.kernel(known_block_size=block_size)
    def gemm_v0_preshuffle(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        bid_m, bid_n = fx.block_idx.x, fx.block_idx.y
        row, kg = tid % 16, tid // 16

        if const_expr(PRESHUFFLE):
            # preshuffle(A) = (8 elem, 64 lane, K//32, BM//16, grid_m): a load is 1 KiB contiguous
            A = fx.make_view(fx.get_iter(A), fx.make_ordered_layout((8, 64, NK, NI, grid_size[0]), 0))
            B = fx.make_view(fx.get_iter(B), fx.make_ordered_layout((8, 64, NK, NJ, grid_size[1]), 0))
        else:
            # original row-major layout, same view as v0: a load is 16 rows x 64 B
            A = fx.make_view(fx.get_iter(A), fx.make_ordered_layout((8, K//8, 16, NI, grid_size[0]), 0))
            B = fx.make_view(fx.get_iter(B), fx.make_ordered_layout((8, K//8, 16, NJ, grid_size[1]), 0))
        C = fx.make_view(fx.get_iter(C), fx.make_ordered_layout((16, NJ, grid_size[1],
                                                                16, NI, grid_size[0]), 0))
        A = A[None, None, None, None, bid_m]
        B = B[None, None, None, None, bid_n]
        C = C[None, None, bid_n, None, None, bid_m]

        def load(X, k, i):
            """this lane's 16 B (row, kg) of MFMA block i at K-step k"""
            if const_expr(PRESHUFFLE):
                return X[None, tid, k, i].load()
            return X[None, k * 4 + kg, row, i].load()

        def gr(k):
            """global -> registers: NI A vectors + NJ B vectors (16 B each),
            issued in the order dot() first uses them: a0, b0..b7, a1..a7"""
            a0 = load(A, k, 0)
            b = [load(B, k, j) for j in range_constexpr(NJ)]
            a = [a0] + [load(A, k, i) for i in range_constexpr(1, NI)]
            return a + b

        def dot(acc, ab):
            """NI x NJ blocks x two K16 MFMAs per 16 B chunk (same as v0)"""
            acc = list(acc)
            a = [v.bitcast(fx.Int16) for v in ab[:NI]]
            b = [v.bitcast(fx.Int16) for v in ab[NI:]]
            for i in range_constexpr(NI):
                for j in range_constexpr(NJ):
                    for h in range_constexpr(2):
                        sel = [4 * h + e for e in range(4)]
                        acc[i * NJ + j] = rocdl.mfma_f32_16x16x16bf16_1k(
                            T.f32x4, [a[i].shuffle(a[i], sel), b[j].shuffle(b[j], sel),
                                      acc[i * NJ + j], 0, 0, 0])
            return acc

        def interleave():
            """the 16 prefetch loads go between the first 16 of the 128 MFMAs"""
            for _ in range_constexpr(NI + NJ):
                rocdl.sched_mfma(1)
                rocdl.sched_vmem(1)
            rocdl.sched_mfma(2 * NA - NI - NJ)

        acc0 = [fx.full(4, 0.0, fx.Float32) for _ in range_constexpr(NA)]
        if const_expr(ILP):
            # two K-steps per iteration, operand buffers cur / nxt alternate.
            # The last iteration prefetches tile 0 again (unused) instead of
            # peeling a K-step after the loop: with the peeled MFMAs LLVM's
            # register allocation degrades (accumulator copies / spills).
            for k, st in range(fx.Index(0), fx.Index(NK // 2), fx.Index(1), init=acc0 + gr(fx.Int32(0))):
                st = list(st)
                kt = fx.Int32(k) * 2
                nxt = gr(kt + 1)                 # prefetch K-step kt+1 ...
                acc = dot(st[:NA], st[NA:])      # ... while computing kt
                interleave()
                cur = gr((kt + 2) % NK)          # prefetch kt+2 while computing kt+1
                acc = dot(acc, nxt)
                interleave()
                results = yield acc + cur
            acc = list(results)[:NA]
        else:
            # v0 loop: loads and MFMAs of the same K-step
            for k, st in range(fx.Index(0), fx.Index(NK), fx.Index(1), init=acc0):
                acc = dot(list(st), gr(fx.Int32(k)))
                results = yield acc
            acc = list(results)

        # Output: lane l holds C[m = 4*(l//16) + r, n = l % 16], r = 0..3
        for i in range_constexpr(NI):
            for j in range_constexpr(NJ):
                out = Vec(acc[i * NJ + j], (4,), fx.Float32)
                for r in range_constexpr(4):
                    C[row, j, kg * 4 + r, i] = out[r]

    @flyc.jit
    def launcher(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream):
        gemm_v0_preshuffle(A, B, C,
        value_attrs={
            "rocdl.waves_per_eu": OCCUPANCY,
        }).launch(grid=grid_size, block=block_size, stream=stream)

    return launcher


if __name__ == "__main__":
    from common import test_gemm
    OCCUPANCY = 1
    M = 16*2*256
    N = 19*2*256
    K = 8192
    test_gemm(M, N, K, build_v0_preshuffle(128, 128, OCCUPANCY, M, N, K, PRESHUFFLE=False))
    test_gemm(M, N, K, build_v0_preshuffle(128, 128, OCCUPANCY, M, N, K), prep=preshuffle)
