
import torch
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec



def build_v0(BM, BN, OCCUPANCY, M, N, K, SLICE=False, INNER="N"):
    # Compile-time knobs (used by flydsl/pipeline_101; defaults = the original v0):
    #   SLICE: index A/B with the dynamic coordinates (kk, row) first, then with the
    #          static tile number -> one base pointer + constant offsets per operand
    #          instead of a full i32 index per load.
    #   INNER: which MFMA tile loop is innermost: "N" (for i: for j) or "M" (for j: for i).
    assert INNER in ("N", "M")
    grid_size = (M//BM, N//BN, 1)
    block_size = (64, 1, 1)
    NI, NJ = BM//16, BN//16
    # visiting order of the (i, j) MFMA tiles inside one K step
    TILES = ([(i, j) for i in range(NI) for j in range(NJ)] if INNER == "N"
             else [(i, j) for j in range(NJ) for i in range(NI)])

    @flyc.kernel(known_block_size=block_size)
    def gemm_v0(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        # 1wave/CTA, each CTA computes a BMxBN tile of C
        tid = fx.thread_idx.x
        bid_m, bid_n = fx.block_idx.x, fx.block_idx.y

        # 16Bytes(8-bf16) per-lane
        # Layouts (mode 0 fastest): A [M, K] / B [N, K] row-major, K split as (8, K//8);
        # C [M, N] row-major, split as (n16, BN//16, grid_n, m16, BM//16, grid_m).
        A = fx.make_view(fx.get_iter(A), fx.make_ordered_layout((8, K//8, 16, BM//16, grid_size[0]), 0))
        B = fx.make_view(fx.get_iter(B), fx.make_ordered_layout((8, K//8, 16, BN//16, grid_size[1]), 0))
        C = fx.make_view(fx.get_iter(C), fx.make_ordered_layout((16, BN//16, grid_size[1],
                                                                16, BM//16, grid_size[0]), 0))

        # slice -> A: (8, K//8, 16, BM//16), B: (8, K//8, 16, BN//16), C: (16, BN//16, 16, BM//16)
        A = A[None, None, None, None, bid_m]
        B = B[None, None, None, None, bid_n]
        C = C[None, None, bid_n, None, None, bid_m]

        # MFMA 16x16x16 operand layout: lane l feeds row (l % 16) and a 4-element K
        # group selected by (l // 16). Each lane loads 8 contiguous K (16 bytes) and
        # feeds them to two MFMAs (elements 0..3, then 4..7); A and B use the same K
        # permutation, so the reduction is unchanged. One wave step covers
        # 4 lane groups x 8 = 32 K.
        row = tid % 16
        kg = tid // 16

        acc_init = [fx.full(4, 0.0, fx.Float32) for _ in range_constexpr(NI * NJ)]
        # runtime scf.for (typed bounds => not unrolled); accumulators are loop-carried
        for k, acc in range(fx.Index(0), fx.Index(K//32), fx.Index(1), init=acc_init):
            acc = list(acc)
            kk = fx.Int32(k) * 4 + kg  # loop iv is index-typed, tid is Int32
            # Load A and B tiles for this k iteration (8 bf16 per lane, one 16 B load)
            if const_expr(SLICE):
                Ak, Bk = A[None, kk, row, None], B[None, kk, row, None]
                a_tile = [Ak[None, i].load().bitcast(fx.Int16) for i in range_constexpr(NI)]
                b_tile = [Bk[None, j].load().bitcast(fx.Int16) for j in range_constexpr(NJ)]
            else:
                a_tile = [A[None, kk, row, i].load().bitcast(fx.Int16) for i in range_constexpr(NI)]
                b_tile = [B[None, kk, row, j].load().bitcast(fx.Int16) for j in range_constexpr(NJ)]
            # Compute C tile: two K16 MFMAs per 8-element lane chunk
            for t in range_constexpr(NI * NJ):
                i, j = TILES[t]
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
        gemm_v0(A, B, C,
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
    test_gemm(M, N, K, build_v0(64, 64, OCCUPANCY, M, N, K))
    test_gemm(M, N, K, build_v0(128, 128, OCCUPANCY, M, N, K))
    test_gemm(M, N, K, build_v0(128, 256, OCCUPANCY, M, N, K))
