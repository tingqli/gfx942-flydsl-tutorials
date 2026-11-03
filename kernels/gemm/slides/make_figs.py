"""Figures for the CDNA3 GEMM training slides (English labels: no CJK font here).

    python3 make_figs.py            # -> figs/*.png
Reads data/v2_loop_isa.s (v2 main loop ISA) and data/bench.txt (bench.py output).
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)

C_MFMA = "#e4572e"
C_DSR = "#17bebb"
C_DSW = "#2e86ab"
C_GLD = "#76b041"
C_BAR = "#222222"
C_WAIT = "#ffc914"
C_OTH = "#cfcfcf"


def box(ax, x, y, w, h, text, fc, ec="black", fs=10, tc="black", lw=1.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. CDNA3 memory hierarchy / CU
# ---------------------------------------------------------------------------
def fig_arch():
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    # one CU
    ax.add_patch(Rectangle((0.2, 2.2), 5.6, 4.2, fc="#f6f6f6", ec="black", lw=1.5))
    ax.text(0.35, 6.15, "Compute Unit (x304, 38 per XCD)", fontsize=11, fontweight="bold", va="center")
    for s in range(4):
        x = 0.4 + s * 1.35
        box(ax, x, 4.2, 1.2, 1.6, f"SIMD{s}\n16 lanes x4\nMFMA unit", "#ffe0d6", fs=9)
        box(ax, x, 3.55, 1.2, 0.55, "VGPR 256 +\nAGPR 256 /lane", "#fff3c4", fs=7.5)
    box(ax, 0.4, 2.45, 2.6, 0.9, "LDS 64 KB\n32 banks x 4 B = 128 B/clk", "#d6f5f4", fs=9)
    box(ax, 3.15, 2.45, 2.45, 0.9, "L1 (vector) 32 KB\n64 B/clk", "#e2ecf7", fs=9)
    # chip level
    box(ax, 6.4, 4.9, 5.3, 1.3, "L2: 4 MB per XCD (8 XCDs)\nshared by the 38 CUs of one XCD", "#e2ecf7", fs=10)
    box(ax, 6.4, 3.3, 5.3, 1.3, "Infinity Cache (MALL) 256 MB\nshared by all XCDs", "#dfe8d8", fs=10)
    box(ax, 6.4, 1.7, 5.3, 1.3, "HBM3E 256 GB, ~6 TB/s (MI325X)", "#ececec", fs=10)
    for y0, y1 in ((4.9, 4.6), (3.3, 3.0)):
        ax.annotate("", (9.05, y1), (9.05, y0), arrowprops=dict(arrowstyle="<->", lw=1.4))
    ax.annotate("", (6.4, 5.55), (5.8, 2.9), arrowprops=dict(arrowstyle="<->", lw=1.4))
    ax.text(0.3, 1.45, "Peak BF16 MFMA: 304 CU x 4 SIMD x 512 FLOP/clk x 2.1 GHz = 1307 TFLOPS\n"
            "v_mfma_f32_16x16x16_bf16: 8192 FLOP, 16 cycles per SIMD  (wave64, 1 wave/SIMD = 512 regs/lane)",
            fontsize=10.5, va="top", family="monospace")
    save(fig, "arch.png")


# ---------------------------------------------------------------------------
# 2. MFMA 16x16x16 operand layout + 8-element trick
# ---------------------------------------------------------------------------
def fig_mfma():
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.6), gridspec_kw=dict(width_ratios=[1.15, 1.15, 1]))
    cols = ["#e4572e", "#17bebb", "#76b041", "#ffc914"]
    # A tile 16 x 32 (one lane loads 8 contiguous K)
    ax = axs[0]
    for kg in range(4):
        for r in range(16):
            ax.add_patch(Rectangle((kg * 8, 15 - r), 8, 1, fc=cols[kg], ec="white", lw=0.5, alpha=0.85))
        ax.text(kg * 8 + 4, 16.6, f"lanes {kg*16}-{kg*16+15}", ha="center", fontsize=8.5)
    ax.plot([4, 4], [0, 16], "k--", lw=0.8)
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 17.5)
    ax.set_title("A tile 16(M) x 32(K) in registers\nlane l: row l%16, K = 8*(l//16) .. +7 (16 B load)", fontsize=10)
    ax.set_xlabel("K")
    ax.set_ylabel("M row")
    ax.set_yticks([])
    ax = axs[1]
    for h in range(2):
        for kg in range(4):
            for r in range(16):
                ax.add_patch(Rectangle((h * 18 + kg * 4, 15 - r), 4, 1, fc=cols[kg], ec="white", lw=0.5,
                                       alpha=0.85 if h == 0 else 0.45))
        ax.text(h * 18 + 8, 16.6, f"MFMA #{h}: elements {4*h}..{4*h+3}", ha="center", fontsize=9)
    ax.set_xlim(0, 34)
    ax.set_ylim(0, 17.5)
    ax.axis("off")
    ax.set_title("split into two K=16 MFMAs (same split for B)\nany consistent K permutation keeps sum_k A*B", fontsize=10)
    ax = axs[2]
    for kg in range(4):
        for rr in range(4):
            for n in range(16):
                ax.add_patch(Rectangle((n, 15 - (kg * 4 + rr)), 1, 1, fc=cols[kg], ec="white", lw=0.5, alpha=0.85))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 17.5)
    ax.set_title("C 16x16 f32 accumulators\nlane l: rows 4*(l//16)+0..3, col l%16", fontsize=10)
    ax.set_xlabel("N")
    ax.set_yticks([])
    save(fig, "mfma_layout.png")


# ---------------------------------------------------------------------------
# 3. v0 / v1 / v2 tiling
# ---------------------------------------------------------------------------
def fig_tiling():
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # v0
    ax = axs[0]
    ax.set_title("v0: 1 wave / CTA, 128x128 per wave\nHBM/L2 -> VGPR -> MFMA (no sharing)", fontsize=11)
    for i in range(4):
        for j in range(4):
            ax.add_patch(Rectangle((j, 3 - i), 1, 1, fc="#ffe0d6" if (i + j) % 2 else "#fff3c4", ec="black"))
            ax.text(j + .5, 3.5 - i, "wave", ha="center", va="center", fontsize=8)
    ax.set_xlim(-0.1, 4.1)
    ax.set_ylim(-0.1, 4.1)
    ax.set_aspect("equal")
    ax.axis("off")
    # v1
    ax = axs[1]
    ax.set_title("v1: 4 waves / CTA (256x256), LDS 64 KB\nwave w = 2x2 block of 128x128", fontsize=11)
    cols = ["#e4572e", "#17bebb", "#76b041", "#ffc914"]
    for w in range(4):
        wm, wn = w % 2, w // 2
        ax.add_patch(Rectangle((wn * 2, 2 - wm * 2), 2, 2, fc=cols[w], alpha=0.55, ec="black"))
        ax.text(wn * 2 + 1, 3 - wm * 2, f"wave{w}\n(wm={wm}, wn={wn})", ha="center", va="center", fontsize=10)
    ax.set_xlim(-0.1, 4.1)
    ax.set_ylim(-0.1, 4.1)
    ax.set_aspect("equal")
    ax.axis("off")
    # v2
    ax = axs[2]
    ax.set_title("v2: same CTA, 4 quadrants x (2x2 waves)\nwave w owns a 64x64 slice of every quadrant", fontsize=11)
    qn = {(0, 0): "C_tl", (1, 0): "C_bl", (0, 1): "C_tr", (1, 1): "C_br"}
    for (qm, qnn), name in qn.items():
        for w in range(4):
            wm, wn = w % 2, w // 2
            x = qnn * 2 + wn
            y = 3 - (qm * 2 + wm)
            ax.add_patch(Rectangle((x, y), 1, 1, fc=cols[w], alpha=0.55, ec="gray", lw=0.6))
            ax.text(x + .5, y + .5, f"w{w}", ha="center", va="center", fontsize=8)
        ax.add_patch(Rectangle((qnn * 2, 2 - qm * 2), 2, 2, fill=False, ec="black", lw=2))
        ax.text(qnn * 2 + 1, 4.15 - qm * 2.0 if qm == 0 else 1.95, "", fontsize=1)
        ax.text(qnn * 2 + 1.0, 2 - qm * 2 + 1.0, name, ha="center", va="center", fontsize=12, fontweight="bold",
                color="black", alpha=0.35)
    ax.text(-0.35, 3, "A_top", rotation=90, va="center", fontsize=10)
    ax.text(-0.35, 1, "A_bot", rotation=90, va="center", fontsize=10)
    ax.text(1, 4.15, "B_left", ha="center", fontsize=10)
    ax.text(3, 4.15, "B_right", ha="center", fontsize=10)
    ax.set_xlim(-0.5, 4.1)
    ax.set_ylim(-0.1, 4.5)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "tiling.png")


# ---------------------------------------------------------------------------
# 4. LDS swizzle
# ---------------------------------------------------------------------------
def fig_swizzle():
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2))
    cmap = plt.get_cmap("tab10")
    for t, ax in enumerate(axs):
        sw = t == 1
        for r in range(8):
            for c in range(8):
                phys = c ^ r if sw else c
                ax.add_patch(Rectangle((phys, 7 - r), 1, 1, fc=cmap(c), ec="white", alpha=0.85))
                ax.text(phys + .5, 7.5 - r, f"k{c*8}", ha="center", va="center", fontsize=7.5, color="white")
        # highlight: 8 lanes of one ds_read_b128 phase read chunk 0 of rows 0..7
        for r in range(8):
            phys = 0 ^ r if sw else 0
            ax.add_patch(Rectangle((phys, 7 - r), 1, 1, fill=False, ec="black", lw=2.2))
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_aspect("equal")
        ax.set_xticks([i + .5 for i in range(8)])
        ax.set_xticklabels([f"banks\n{4*i}-{4*i+3}" for i in range(8)], fontsize=7)
        ax.set_yticks([i + .5 for i in range(8)])
        ax.set_yticklabels([f"row {7-i}" for i in range(8)], fontsize=8)
        ax.set_title(("XOR swizzle: chunk' = chunk ^ (row % 8)\n8 lanes read 8 rows of chunk k0 -> 8 different bank groups"
                      if sw else
                      "Linear: row stride 128 B = all 32 banks\n8 lanes read 8 rows of chunk k0 -> 8-way bank conflict"),
                     fontsize=10)
    save(fig, "swizzle.png")


# ---------------------------------------------------------------------------
# 5. v1 vs v2 timeline (conceptual)
# ---------------------------------------------------------------------------
def fig_timeline():
    fig, axs = plt.subplots(2, 1, figsize=(14, 5.2), sharex=True)
    ax = axs[0]
    ax.set_title("v1 (no pipeline): every K-step waits for HBM/L2 latency, then barrier, then compute", fontsize=11)
    t = 0
    for k in range(3):
        for name, d, c in (("GR k", 6, C_GLD), ("wait vmcnt", 7, C_WAIT), ("|", 0.4, C_BAR), ("LW", 2, C_DSW),
                           ("|", 0.4, C_BAR), ("LR + DOT (MFMA)", 16, C_MFMA)):
            ax.add_patch(Rectangle((t, 0.2), d, 0.6, fc=c, ec="white"))
            if d > 2:
                ax.text(t + d / 2, 0.5, name.replace("k", str(k)), ha="center", va="center", fontsize=8.5)
            t += d
    ax.set_xlim(0, 95.4)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax = axs[1]
    ax.set_title("v2 (pipeline): GR(k+2) / LW(k+1) / LR run inside the MFMA stream of K-step k", fontsize=11)
    t = 0
    for k in range(6):
        ax.add_patch(Rectangle((t, 0.45), 16, 0.4, fc=C_MFMA, ec="white"))
        ax.text(t + 8, 0.65, f"DOT k={k} (256 MFMA)", ha="center", va="center", fontsize=8.5)
        for r in range(4):
            ax.add_patch(Rectangle((t + r * 4 + 0.2, 0.1), 1.0, 0.28, fc=C_DSR, ec="white"))
            ax.add_patch(Rectangle((t + r * 4 + 1.3, 0.1), 0.9, 0.28, fc=C_DSW, ec="white"))
            ax.add_patch(Rectangle((t + r * 4 + 2.3, 0.1), 0.8, 0.28, fc=C_GLD, ec="white"))
        t += 16
    ax.set_xlim(0, 95.4)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    from matplotlib.patches import Patch
    axs[1].legend(handles=[Patch(fc=C_MFMA, label="MFMA"), Patch(fc=C_DSR, label="ds_read (LR)"),
                           Patch(fc=C_DSW, label="ds_write (LW)"), Patch(fc=C_GLD, label="global_load (GR)"),
                           Patch(fc=C_WAIT, label="s_waitcnt stall"), Patch(fc=C_BAR, label="s_barrier")],
                  ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=9)
    save(fig, "timeline.png")


# ---------------------------------------------------------------------------
# 6. v2 region schedule + LDS slot lifetimes (the key pipeline picture)
# ---------------------------------------------------------------------------
REGIONS = [
    ("C_tl", "A_top x B_left", "A_bot(k)", "B_left(k+1)", "B_left(k+2)"),
    ("C_bl", "A_bot x B_left", "B_right(k)", "A_top(k+1)", "A_top(k+2)"),
    ("C_tr", "A_top x B_right", "B_left(k+1)", "A_bot(k+1)", "A_bot(k+2)"),
    ("C_br", "A_bot x B_right", "A_top(k+1)", "B_right(k+1)", "B_right(k+2)"),
]


def fig_regions():
    fig, ax = plt.subplots(figsize=(15, 7.4))
    ax.set_xlim(-2.6, 12.2)
    ax.set_ylim(-5.6, 3.4)
    ax.axis("off")
    W = 2.9
    # header rows: one K-step = 4 regions
    for r, (q, ops, lr, lw, gr) in enumerate(REGIONS):
        x = r * W
        box(ax, x, 2.2, W - 0.1, 0.8, f"region {r}: DOT {q}\n{ops}  (64 MFMA)", "#ffd6cc", fs=9.5, bold=True)
        box(ax, x, 1.35, W - 0.1, 0.65, f"LR  {lr}", "#cdf3f2", fs=9.5)
        box(ax, x, 0.55, W - 0.1, 0.65, f"LW  {lw}", "#cfe3f3", fs=9.5)
        box(ax, x, -0.25, W - 0.1, 0.65, f"GR  {gr}", "#ddefcf", fs=9.5)
    for lab, y in (("MFMA", 2.6), ("LDS->reg", 1.67), ("reg->LDS", 0.87), ("HBM->reg", 0.07)):
        ax.text(-0.15, y, lab, ha="right", va="center", fontsize=9.5)
    # barriers after region 1 and 3
    for r in (2, 4):
        ax.plot([r * W - 0.05] * 2, [-0.35, 3.1], color=C_BAR, lw=3)
        ax.text(r * W - 0.05, 3.2, "s_barrier", ha="center", fontsize=9, fontweight="bold")
    ax.text(5.8, -0.75, "one K-step k  (~4096 MFMA cycles per SIMD)", ha="center", fontsize=10.5, style="italic")
    # LDS slot lanes
    slots = ["A_top", "A_bot", "B_left", "B_right"]
    ys = {s: -1.6 - i * 0.95 for i, s in enumerate(slots)}
    # events per region: (slot, kind, tile)
    ev = {0: [("A_bot", "R", "k"), ("B_left", "W", "k+1")],
          1: [("B_right", "R", "k"), ("A_top", "W", "k+1")],
          2: [("B_left", "R", "k+1"), ("A_bot", "W", "k+1")],
          3: [("A_top", "R", "k+1"), ("B_right", "W", "k+1")]}
    content = {"A_top": "k", "A_bot": "k", "B_left": "k", "B_right": "k"}
    colk = {"k": "#fde2c8", "k+1": "#c8e6fd"}
    for s in slots:
        ax.text(-0.15, ys[s] + 0.3, f"LDS slot {s}\n(16 KB)", ha="right", va="center", fontsize=9)
    for r in range(4):
        x = r * W
        for s in slots:
            for (ss, kind, tile) in ev[r]:
                if ss == s and kind == "W":
                    content[s] = tile
            ax.add_patch(Rectangle((x, ys[s]), W - 0.1, 0.6, fc=colk[content[s]], ec="gray"))
            ax.text(x + 0.12, ys[s] + 0.3, f"holds {content[s]}", va="center", fontsize=8.5, color="dimgray")
            for (ss, kind, tile) in ev[r]:
                if ss == s:
                    c = C_DSR if kind == "R" else C_DSW
                    ax.add_patch(Rectangle((x + 1.55, ys[s] + 0.08), 1.15, 0.44, fc=c, ec="black"))
                    ax.text(x + 2.12, ys[s] + 0.3, ("read " if kind == "R" else "write ") + tile,
                            ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
    ax.text(5.8, -5.45, "Each slot: last read (k) -> barrier -> rewrite (k+1) -> barrier -> next read (k+1): "
            "one LDS stage is enough;  the staging VGPRs act as the 2nd buffer",
            ha="center", fontsize=10)
    save(fig, "v2_regions.png")


# ---------------------------------------------------------------------------
# 7. v2 ISA strip (from the real compiled loop)
# ---------------------------------------------------------------------------
def fig_isa_strip():
    path = os.path.join(HERE, "data", "v2_loop_isa.s")
    seq = []
    for line in open(path):
        s = line.strip()
        if not s or s.startswith(";") or s.startswith("."):
            continue
        op = s.split()[0]
        if op.startswith("v_mfma"):
            seq.append("M")
        elif op.startswith("ds_read"):
            seq.append("R")
        elif op.startswith("ds_write"):
            seq.append("W")
        elif op.startswith(("global_load", "buffer_load")):
            seq.append("G")
        elif op.startswith("s_barrier"):
            seq.append("|")
        elif op.startswith("s_waitcnt"):
            seq.append("w")
        else:
            seq.append(".")
    cmap = {"M": C_MFMA, "R": C_DSR, "W": C_DSW, "G": C_GLD, "|": C_BAR, "w": C_WAIT, ".": C_OTH}
    per_row = 101
    rows = (len(seq) + per_row - 1) // per_row
    fig, ax = plt.subplots(figsize=(15, 0.55 * rows + 1.2))
    for i, c in enumerate(seq):
        r, x = divmod(i, per_row)
        ax.add_patch(Rectangle((x, rows - 1 - r), 1, 0.8, fc=cmap[c], ec="none"))
    ax.set_xlim(0, per_row)
    ax.set_ylim(0, rows)
    ax.axis("off")
    cnt = {k: seq.count(k) for k in "MRWG|w"}
    ax.set_title(f"v2 main loop, compiled ISA in program order ({len(seq)} instr): "
                 f"{cnt['M']} MFMA, {cnt['R']} ds_read_b128, {cnt['W']} ds_write_b128, "
                 f"{cnt['G']} global_load_dwordx4, {cnt['|']} s_barrier", fontsize=10.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=v, label=k) for k, v in (("MFMA", C_MFMA), ("ds_read", C_DSR), ("ds_write", C_DSW),
                                                          ("global_load", C_GLD), ("s_waitcnt", C_WAIT),
                                                          ("s_barrier", C_BAR), ("other", C_OTH))],
              ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=9)
    save(fig, "v2_isa_strip.png")


# ---------------------------------------------------------------------------
# 8. performance bars (from data/bench.txt)
# ---------------------------------------------------------------------------
def fig_perf():
    path = os.path.join(HERE, "data", "bench.txt")
    if not os.path.exists(path):
        return
    names, tf = [], []
    for line in open(path):
        m = re.match(r"(\S+)\s+([\d.]+) us\s+([\d.]+) TFLOPS", line)
        if m:
            names.append(m.group(1))
            tf.append(float(m.group(3)))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    colors = ["#bbbbbb" if n == "torch" else ("#e4572e" if n == "v2" else "#2e86ab") for n in names]
    bars = ax.bar(names, tf, color=colors)
    for b, v in zip(bars, tf):
        ax.text(b.get_x() + b.get_width() / 2, v + 8, f"{v:.0f}", ha="center", fontsize=10)
    ax.axhline(1307, color="black", ls="--", lw=1)
    ax.text(len(names) - 0.5, 1290, "peak 1307 (2.1 GHz)", ha="right", va="top", fontsize=9)
    ax.set_ylabel("TFLOPS (BF16 in, FP32 acc)")
    ax.set_ylim(0, 1400)
    prov = "PROVISIONAL" in open(path).read()
    ax.set_title("GEMM 8192 x 9728 x 8192 on one MI325X (median of 20)"
                 + ("\nPROVISIONAL: GPU shared with other jobs" if prov else ""), fontsize=11)
    save(fig, "perf.png")


if __name__ == "__main__":
    fig_arch()
    fig_mfma()
    fig_tiling()
    fig_swizzle()
    fig_timeline()
    fig_regions()
    fig_isa_strip()
    fig_perf()
    print("figures ->", OUT, sorted(os.listdir(OUT)))
