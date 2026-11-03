
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec


def build_v2(BM, BN, OCCUPANCY, M, N, K, SCHED=True):
    """v1 + software pipeline, following gfx950-gluon-tutorials gfx942/intra_wave.

    CTA tile 256x256x64, 4 waves (1 per SIMD). A is split along M into A_top /
    A_bot and B along N into B_left / B_right (four 16 KB LDS slots = 64 KB);
    each (A half, B half) pair is one 128x128 quadrant of C, and each wave owns a
    64x64 slice of every quadrant (2x2 waves inside a quadrant).

    One K-step = 4 regions, each = 64 MFMAs on one quadrant plus three memory ops:
        region 0: DOT C_tl | LR A_bot(k)    | LW B_left(k+1)  | GR B_left(k+2)
        region 1: DOT C_bl | LR B_right(k)  | LW A_top(k+1)   | GR A_top(k+2)
        region 2: DOT C_tr | LR B_left(k+1) | LW A_bot(k+1)   | GR A_bot(k+2)
        region 3: DOT C_br | LR A_top(k+1)  | LW B_right(k+1) | GR B_right(k+2)
    GR = global -> registers (2 K-steps ahead), LW = registers -> LDS (1 ahead),
    LR = LDS -> registers (one region ahead of its DOT). A slot is rewritten one
    region after its last read and read one region after its rewrite, so a
    single LDS stage suffices; two barriers per K-step (after regions 1 and 3)
    order every cross-wave read/write pair.
    """
    assert (BM, BN) == (256, 256)
    BK = 64
    NK = K // BK
    assert NK >= 3
    H = 128                    # half-tile rows
    WT = 64                    # per-wave slice of a quadrant
    NI = NJ = WT // 16         # 4x4 MFMA blocks per quadrant slice
    grid_size = (M//BM, N//BN, 1)
    block_size = (256, 1, 1)

    @fx.struct
    class SharedStorage:
        a: fx.Array[fx.BFloat16, BM * BK, 16]   # A_top rows 0..127, A_bot 128..255
        b: fx.Array[fx.BFloat16, BN * BK, 16]   # B_left rows 0..127, B_right 128..255

    @flyc.kernel(known_block_size=block_size)
    def gemm_v2(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        bid_m, bid_n = fx.block_idx.x, fx.block_idx.y
        wave, lane = tid // 64, tid % 64
        wm, wn = wave % 2, wave // 2

        # global: (8 elem, 8 chunk, NK, 32 row, 8 pass, grid); passes 0-3 = top/left half
        A = fx.make_view(fx.get_iter(A), fx.make_ordered_layout((8, 8, NK, 32, BM//32, grid_size[0]), 0))
        B = fx.make_view(fx.get_iter(B), fx.make_ordered_layout((8, 8, NK, 32, BN//32, grid_size[1]), 0))
        # C [M, N]: (n16, j, wn, qn, grid_n, m16, i, wm, qm, grid_m)
        C = fx.make_view(fx.get_iter(C), fx.make_ordered_layout((16, NJ, 2, 2, grid_size[1],
                                                                16, NI, 2, 2, grid_size[0]), 0))
        A = A[None, None, None, None, None, bid_m]
        B = B[None, None, None, None, None, bid_n]
        C = C[None, None, wn, None, bid_n, None, None, wm, None, bid_m]

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_a = lds.a.view(fx.make_ordered_layout((8, 8, BM), 0))
        lds_b = lds.b.view(fx.make_ordered_layout((8, 8, BN), 0))

        st_c, st_r = tid % 8, tid // 8
        st_c_sw = st_c ^ (st_r % 8)
        row = lane % 16
        kg = lane // 16

        # ---- the three memory ops, per half-tile (h = 0 top/left, 1 bot/right)
        def gr(G, h, kt):
            """GR: global -> 4 staging vectors (16 B each) of one half-tile"""
            return [G[None, st_c, kt, st_r, h * 4 + p].load() for p in range_constexpr(4)]

        def lw(L, h, st):
            """LW: staging registers -> LDS slot (swizzled, ds_write_b128)"""
            for p in range_constexpr(4):
                L[None, st_c_sw, h * H + p * 32 + st_r].store(st[p])

        def lr(L, h, w):
            """LR: this wave's 64x64 slice of a half-tile -> 8 operand vectors"""
            return [L[None, (kg + s * 4) ^ (row % 8), h * H + w * WT + i * 16 + row].load()
                    for s in range_constexpr(2) for i in range_constexpr(NI)]

        def dot(acc, a_op, b_op):
            """DOT: 64 MFMAs (4x4 blocks x K=64) on one quadrant slice"""
            acc = list(acc)
            for s in range_constexpr(2):
                for i in range_constexpr(NI):
                    a = a_op[s * NI + i].bitcast(fx.Int16)
                    for j in range_constexpr(NJ):
                        b = b_op[s * NJ + j].bitcast(fx.Int16)
                        for hh in range_constexpr(2):
                            sel = [4 * hh + e for e in range(4)]
                            acc[i * NJ + j] = rocdl.mfma_f32_16x16x16bf16_1k(
                                T.f32x4, [a.shuffle(a, sel), b.shuffle(b, sel), acc[i * NJ + j], 0, 0, 0])
            return acc

        def interleave(n_dsrd, n_dswr, n_vmem):
            """Ask the scheduler to spread one region's memory ops through its
            64 MFMAs: 2 MFMAs (32 cycles) per LDS access (CDNA3 LDS = 128 B/clk,
            shared by 4 SIMDs), 1 MFMA per global load."""
            if const_expr(SCHED):
                for _ in range_constexpr(n_dsrd):
                    rocdl.sched_mfma(2)
                    rocdl.sched_dsrd(1)
                for _ in range_constexpr(n_dswr):
                    rocdl.sched_mfma(2)
                    rocdl.sched_dswr(1)
                for _ in range_constexpr(n_vmem):
                    rocdl.sched_mfma(1)
                    rocdl.sched_vmem(1)
                rocdl.sched_mfma(64 - 2 * n_dsrd - 2 * n_dswr - n_vmem)

        # ---- prologue: tile 0 -> LDS, tile 1 -> registers, prime a_top / b_left
        lw(lds_b, 0, gr(B, 0, fx.Int32(0)))
        lw(lds_a, 0, gr(A, 0, fx.Int32(0)))
        lw(lds_a, 1, gr(A, 1, fx.Int32(0)))
        lw(lds_b, 1, gr(B, 1, fx.Int32(0)))
        g_bl = gr(B, 0, fx.Int32(1))
        gpu.barrier()                                    # tile 0 visible
        g_at = gr(A, 0, fx.Int32(1))                     # same issue order as the loop
        g_ab = gr(A, 1, fx.Int32(1))
        g_br = gr(B, 1, fx.Int32(1))
        b_left = lr(lds_b, 0, wn)
        a_top = lr(lds_a, 0, wm)
        gpu.barrier()                                    # prologue reads done before LW B_left

        acc0 = [fx.full(4, 0.0, fx.Float32) for _ in range_constexpr(4 * NI * NJ)]
        state0 = acc0 + a_top + b_left + g_bl + g_at + g_ab + g_br
        NA = 4 * NI * NJ

        def unpack(st):
            st = list(st)
            acc = [st[q * NI * NJ:(q + 1) * NI * NJ] for q in range_constexpr(4)]
            o = NA
            a_top, b_left = st[o:o + 8], st[o + 8:o + 16]
            o += 16
            g = [st[o + 4 * x:o + 4 * x + 4] for x in range_constexpr(4)]
            return acc, a_top, b_left, g

        def kstep(acc, a_top, b_left, g, kt_next2, last):
            """One K-step. last=0: steady state; 1: no GR; 2: no GR/LW, no next LR."""
            g_bl, g_at, g_ab, g_br = g
            # region 0: C_tl
            acc[0] = dot(acc[0], a_top, b_left)
            a_bot = lr(lds_a, 1, wm)
            if const_expr(last < 2):
                lw(lds_b, 0, g_bl)
            if const_expr(last < 1):
                g_bl = gr(B, 0, kt_next2)
            interleave(8, 4 if last < 2 else 0, 4 if last < 1 else 0)
            # region 1: C_bl
            acc[1] = dot(acc[1], a_bot, b_left)
            b_right = lr(lds_b, 1, wn)
            if const_expr(last < 2):
                lw(lds_a, 0, g_at)
            if const_expr(last < 1):
                g_at = gr(A, 0, kt_next2)
            interleave(8, 4 if last < 2 else 0, 4 if last < 1 else 0)
            if const_expr(last < 2):
                gpu.barrier()      # B_left/A_top(k+1) written; A_bot/B_right(k) read
            # region 2: C_tr
            acc[2] = dot(acc[2], a_top, b_right)
            if const_expr(last < 2):
                b_left = lr(lds_b, 0, wn)
                lw(lds_a, 1, g_ab)
            if const_expr(last < 1):
                g_ab = gr(A, 1, kt_next2)
            interleave(8 if last < 2 else 0, 4 if last < 2 else 0, 4 if last < 1 else 0)
            # region 3: C_br
            acc[3] = dot(acc[3], a_bot, b_right)
            if const_expr(last < 2):
                a_top = lr(lds_a, 0, wm)
                lw(lds_b, 1, g_br)
            if const_expr(last < 1):
                g_br = gr(B, 1, kt_next2)
            interleave(8 if last < 2 else 0, 4 if last < 2 else 0, 4 if last < 1 else 0)
            if const_expr(last < 2):
                gpu.barrier()      # A_bot/B_right(k+1) written; B_left/A_top(k+1) read
            return acc, a_top, b_left, [g_bl, g_at, g_ab, g_br]

        # ---- main loop: K-steps 0 .. NK-3
        for k, st in range(fx.Index(0), fx.Index(NK - 2), fx.Index(1), init=state0):
            acc, a_top, b_left, g = unpack(st)
            acc, a_top, b_left, g = kstep(acc, a_top, b_left, g, fx.Int32(k) + 2, 0)
            results = yield sum(acc, []) + a_top + b_left + sum(g, [])

        # ---- peel: K-step NK-2 (LW, no GR) and NK-1 (drain)
        acc, a_top, b_left, g = unpack(results)
        acc, a_top, b_left, g = kstep(acc, a_top, b_left, g, None, 1)
        acc, a_top, b_left, g = kstep(acc, a_top, b_left, g, None, 2)

        # ---- epilogue: quadrant q = (qm, qn): tl, bl, tr, br
        for q in range_constexpr(4):
            qm, qn = q % 2, q // 2   # tl, bl, tr, br
            for i in range_constexpr(NI):
                for j in range_constexpr(NJ):
                    out = Vec(acc[q][i * NJ + j], (4,), fx.Float32)
                    for r in range_constexpr(4):
                        C[row, j, qn, kg * 4 + r, i, qm] = out[r]

    @flyc.jit
    def launcher(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream):
        gemm_v2(A, B, C,
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
    test_gemm(M, N, K, build_v2(256, 256, OCCUPANCY, M, N, K))
