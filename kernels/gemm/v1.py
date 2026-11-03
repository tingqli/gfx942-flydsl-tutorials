
import torch
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec



def build_v1(BM, BN, OCCUPANCY, M, N, K):
    # CTA computes BMxBN of C with 4 waves in a 2x2 grid, each wave WMxWN.
    # Per K step (BK=64) the CTA stages A[BM, 64] / B[BN, 64] bf16 in LDS.
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
        lds_a = lds.a.view(fx.make_ordered_layout((8, 8, BM), 0))
        lds_b = lds.b.view(fx.make_ordered_layout((8, 8, BN), 0))

        # staging coordinates (global -> LDS)
        st_c, st_r = tid % 8, tid // 8
        st_c_sw = st_c ^ (st_r % 8)          # (p*32 + st_r) % 8 == st_r % 8

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
                lds_a[None, st_c_sw, p * 32 + st_r].store(a_st[p])
            for p in range_constexpr(BN//32):
                lds_b[None, st_c_sw, p * 32 + st_r].store(b_st[p])
            gpu.barrier()
            # 3. LDS -> registers, then MFMA
            for s in range_constexpr(BK // 32):
                ck = (kg + s * 4) ^ (row % 8)  # row % 8 == (any 16-aligned base + row) % 8
                a_tile = [lds_a[None, ck, wm * WM + i * 16 + row].load().bitcast(fx.Int16)
                          for i in range_constexpr(NI)]
                b_tile = [lds_b[None, ck, wn * WN + j * 16 + row].load().bitcast(fx.Int16)
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
