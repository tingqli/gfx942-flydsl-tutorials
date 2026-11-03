
# LDS swizzle 的两种写法（编译期开关 build_v1(..., SWIZZLE="manual" | "fly")）
#
# LDS 按 [行][8 个 16B chunk] 存放，一行 64 个 bf16 = 128B = 正好 32 个 bank。
# swizzle 规则：物理 chunk = 逻辑 chunk ^ (row % 8)，使 ds_read_b128 一拍的 8 个 lane
# （8 行同一 chunk）和 ds_write_b128 一拍的 8 个 lane（1 行 8 个 chunk）都铺满 32 个 bank。
#
# - "manual"（默认）：手写 c ^ (r % 8)，在普通 LDS 视图上直接 load()/store()。
# - "fly"：FlyDSL 内置 Swizzle：
#       swz = fx.static(fx.SwizzleType.get(3, 3, 3))
#       layout = fx.make_composed_layout(swz, fx.make_ordered_layout((8, 8, BM), 0))
#   元素偏移 = e + 8*chunk + 64*row，Swizzle(B=3, M=3, S=3) 把偏移的 bit[6,9)（row%8）
#   XOR 到 bit[3,6)（chunk），映射与手写完全相同，调用方直接用逻辑 chunk 号。
#   注意：FlyDSL 0.3.2 不支持在 composed layout 视图上做向量 load/store
#   （MemRefLoadVecOp: MemRefType with ComposedLayout is not supported），
#   所以只用它算地址：fx.crd2idx((0, c, r), layout) 得到 swizzle 后的偏移，
#   再在该地址上建一个连续 8 元素的普通视图做 load()/store()。
#
# 实测（8192x9728x8192，空闲 MI325X）：
#                                      manual        fly
#   正确性                              ✓             ✓
#   LDS 指令                            16 ds_write_b128 + 32 ds_read_b128（两者相同）
#   SQ_LDS_BANK_CONFLICT                0             0
#   循环内 v_xor                        0             0（都被提到循环外）
#   带立即数偏移的 ds_read_b128          26            0
#   VGPR                                436           464
#   TFLOPS                              438.0         438.9
#
# 区别在寄存器：手写版的 XOR 只作用在 chunk 号上，同一 lane 各行 row%8 相同，XOR 结果
# 不变，行间差距可写成 ds_read 的立即数 offset，共用一个地址寄存器；内置 swizzle
# 对整个偏移做 XOR，XOR 与加法不可交换，编译器只能为每次读取单独准备地址寄存器
# （+28 VGPR）。v1 寄存器有余量所以无影响；v2 寄存器已用满，保持手写更稳妥。

import torch
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec



def build_v1(BM, BN, OCCUPANCY, M, N, K, SWIZZLE="manual"):
    # CTA computes BMxBN of C with 4 waves in a 2x2 grid, each wave WMxWN.
    # Per K step (BK=64) the CTA stages A[BM, 64] / B[BN, 64] bf16 in LDS.
    # SWIZZLE (compile time): "manual" = XOR the chunk index by hand,
    #                         "fly"    = FlyDSL Swizzle(3,3,3) composed layout.
    assert SWIZZLE in ("manual", "fly")
    BK = 64
    WM, WN = BM // 2, BN // 2
    NI, NJ = WM // 16, WN // 16
    grid_size = (M//BM, N//BN, 1)
    block_size = (256, 1, 1)

    @fx.struct
    class SharedStorage:
        a: fx.Array[fx.BFloat16, BM * BK, 16]
        b: fx.Array[fx.BFloat16, BN * BK, 16]

    @flyc.kernel(known_block_size=block_size)
    def gemm_v1(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        # 4 waves/CTA, each CTA computes a BMxBN tile of C
        tid = fx.thread_idx.x
        bid_m, bid_n = fx.block_idx.x, fx.block_idx.y
        wave, lane = tid // 64, tid % 64
        wm, wn = wave % 2, wave // 2

        # Global view for coalesced staging: a row's 64 K (128 B) is read by 8
        # consecutive threads (16 B each); 256 threads cover 32 rows per pass.
        #   A [M, K] -> (8 elem, 8 chunk, K//64, 32 row, BM//32 pass, grid_m)
        #   C [M, N] -> (n16, NJ, 2 wn, grid_n, m16, NI, 2 wm, grid_m)
        A = fx.make_view(fx.get_iter(A), fx.make_ordered_layout((8, 8, K//BK, 32, BM//32, grid_size[0]), 0))
        B = fx.make_view(fx.get_iter(B), fx.make_ordered_layout((8, 8, K//BK, 32, BN//32, grid_size[1]), 0))
        C = fx.make_view(fx.get_iter(C), fx.make_ordered_layout((16, NJ, 2, grid_size[1],
                                                                16, NI, 2, grid_size[0]), 0))
        A = A[None, None, None, None, None, bid_m]
        B = B[None, None, None, None, None, bid_n]
        C = C[None, None, wn, bid_n, None, None, wm, bid_m]

        # LDS: [rows][8 chunks of 16 B]; swizzle chunk ^= row % 8 so that the 8
        # lanes of a ds_read_b128 phase (8 rows, same chunk) and of a
        # ds_write_b128 phase (1 row, 8 chunks) both hit all 32 banks.
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        if const_expr(SWIZZLE == "fly"):
            # element offset = e + 8*chunk + 64*row; Swizzle(B=3, M=3, S=3) xors
            # offset bits [6,9) (row % 8) into bits [3,6) (chunk): the same mapping,
            # done by the layout. Vector load/store on a composed view is not
            # supported, so the composed layout only produces the (swizzled)
            # offset of each 16 B chunk; the chunk itself is a plain 8-elem view.
            swz = fx.static(fx.SwizzleType.get(3, 3, 3))
            lds_a = (lds.a.ptr, fx.make_composed_layout(swz, fx.make_ordered_layout((8, 8, BM), 0)))
            lds_b = (lds.b.ptr, fx.make_composed_layout(swz, fx.make_ordered_layout((8, 8, BN), 0)))
        else:
            lds_a = lds.a.view(fx.make_ordered_layout((8, 8, BM), 0))
            lds_b = lds.b.view(fx.make_ordered_layout((8, 8, BN), 0))

        def lds_chunk(L, c, r):
            """8-element (16 B) LDS view holding logical chunk c of row r"""
            if const_expr(SWIZZLE == "fly"):
                ptr, layout = L
                return fx.make_view(fx.add_offset(ptr, fx.crd2idx((0, c, r), layout)),
                                    fx.make_layout(8, 1))
            return L[None, c ^ (r % 8), r]

        # staging coordinates (global -> LDS)
        st_c, st_r = tid % 8, tid // 8

        # MFMA 16x16x16 operand layout: lane l feeds row (l % 16) and a 4-element K
        # group selected by (l // 16). Each lane reads 8 contiguous K (16 bytes) and
        # feeds them to two MFMAs (elements 0..3, then 4..7); A and B use the same K
        # permutation, so the reduction is unchanged. One sub-step covers 32 K.
        row = lane % 16
        kg = lane // 16

        acc_init = [fx.full(4, 0.0, fx.Float32) for _ in range_constexpr(NI * NJ)]
        # runtime scf.for (typed bounds => not unrolled); accumulators are loop-carried
        for k, acc in range(fx.Index(0), fx.Index(K//BK), fx.Index(1), init=acc_init):
            acc = list(acc)
            kt = fx.Int32(k)  # loop iv is index-typed, tid is Int32
            # 1. global -> staging registers (coalesced, 16 B per thread per pass)
            a_st = [A[None, st_c, kt, st_r, p].load() for p in range_constexpr(BM//32)]
            b_st = [B[None, st_c, kt, st_r, p].load() for p in range_constexpr(BN//32)]
            # 2. staging registers -> LDS (wait until every wave finished reading
            #    the previous tile)
            gpu.barrier()
            for p in range_constexpr(BM//32):
                lds_chunk(lds_a, st_c, p * 32 + st_r).store(a_st[p])
            for p in range_constexpr(BN//32):
                lds_chunk(lds_b, st_c, p * 32 + st_r).store(b_st[p])
            gpu.barrier()
            # 3. LDS -> registers, then MFMA
            for s in range_constexpr(BK // 32):
                a_tile = [lds_chunk(lds_a, kg + s * 4, wm * WM + i * 16 + row).load().bitcast(fx.Int16)
                          for i in range_constexpr(NI)]
                b_tile = [lds_chunk(lds_b, kg + s * 4, wn * WN + j * 16 + row).load().bitcast(fx.Int16)
                          for j in range_constexpr(NJ)]
                for i in range_constexpr(NI):
                    for j in range_constexpr(NJ):
                        for h in range_constexpr(2):
                            sel = [4 * h + e for e in range(4)]
                            acc[i * NJ + j] = fx.rocdl.mfma_f32_16x16x16bf16_1k(T.f32x4,
                                                    [a_tile[i].shuffle(a_tile[i], sel),
                                                    b_tile[j].shuffle(b_tile[j], sel),
                                                    acc[i * NJ + j], 0, 0, 0])
            results = yield acc

        # Output: lane l holds C[m = 4*(l//16) + r, n = l % 16], r = 0..3
        for i in range_constexpr(NI):
            for j in range_constexpr(NJ):
                out = Vec(results[i * NJ + j], (4,), fx.Float32)
                for r in range_constexpr(4):
                    C[row, j, kg * 4 + r, i] = out[r]

    @flyc.jit
    def launcher(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream):
        gemm_v1(A, B, C,
        value_attrs={
            "rocdl.waves_per_eu": OCCUPANCY,
        }).launch(grid=grid_size, block=block_size, stream=stream)

    return launcher

if __name__ == "__main__":
    from common import test_gemm
    # V_MFMA_F32_16x16x16_BF16
    OCCUPANCY = 1
    M = 16*2*256
    N = 19*2*256
    K = 8192
    test_gemm(M, N, K, build_v1(256, 256, OCCUPANCY, M, N, K))
