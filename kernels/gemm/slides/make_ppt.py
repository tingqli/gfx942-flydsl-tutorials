"""Build the CDNA3 GEMM training deck (python-pptx).

    python3 make_figs.py && python3 make_ppt.py   # -> CDNA3_GEMM_FlyDSL.pptx
Numbers come from data/bench.txt (bench.py) and data/counters.txt (rocprofv3 --pmc).
"""
import os
import re

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")
DATA = os.path.join(HERE, "data")

RED = RGBColor(0xC0, 0x1F, 0x2F)
DARK = RGBColor(0x22, 0x22, 0x22)
GRAY = RGBColor(0x60, 0x60, 0x60)
CODE_BG = RGBColor(0xF4, 0xF4, 0xF4)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = prs.slide_width, prs.slide_height


# --------------------------------------------------------------------------- helpers
def textbox(slide, x, y, w, h, paras, size=16, color=DARK, font=None, bg=None):
    """paras: list of str or (str, dict(level=, bold=, color=, size=))"""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    if bg is not None:
        tb.fill.solid()
        tb.fill.fore_color.rgb = bg
    first = True
    for p in paras:
        opt = {}
        if isinstance(p, tuple):
            p, opt = p
        para = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        lvl = opt.get("level", 0)
        bullet = opt.get("bullet", font is None)
        run = para.add_run()
        run.text = (("• " if lvl == 0 else "– ") if bullet and p else "") + p
        para.level = min(lvl, 4)
        f = run.font
        f.size = Pt(opt.get("size", size - 2 * lvl))
        f.bold = opt.get("bold", False)
        f.color.rgb = opt.get("color", color)
        if font:
            f.name = font
        para.space_after = Pt(opt.get("space", 4))
    return tb


def code(slide, x, y, w, h, src, size=11):
    lines = src.strip("\n").split("\n")
    return textbox(slide, x, y, w, h, [(l, dict(bullet=False, space=0)) for l in lines],
                   size=size, font="Consolas", bg=CODE_BG)


def title(slide, text, sub=None):
    textbox(slide, Inches(0.5), Inches(0.25), Inches(12.3), Inches(0.8),
            [(text, dict(bullet=False, bold=True, size=28, color=RED))])
    if sub:
        textbox(slide, Inches(0.5), Inches(0.95), Inches(12.3), Inches(0.5),
                [(sub, dict(bullet=False, size=15, color=GRAY))])
    line = slide.shapes.add_connector(1, Inches(0.5), Inches(1.45), Inches(12.83), Inches(1.45))
    line.line.color.rgb = RED


def image(slide, name, x, y, w=None, h=None):
    path = os.path.join(FIG, name)
    if os.path.exists(path):
        return slide.shapes.add_picture(path, x, y, width=w, height=h)
    textbox(slide, x, y, w or Inches(4), Inches(0.5), [(f"[missing figure {name}]", dict(bullet=False))])


def table(slide, x, y, w, rows, col_w=None, size=12, h_row=0.38):
    nr, nc = len(rows), len(rows[0])
    shp = slide.shapes.add_table(nr, nc, x, y, w, Inches(h_row * nr))
    t = shp.table
    for c in range(nc):
        if col_w:
            t.columns[c].width = Inches(col_w[c])
    for r in range(nr):
        for c in range(nc):
            cell = t.cell(r, c)
            cell.text = str(rows[r][c])
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(size)
                    run.font.bold = r == 0
    return t


def slide(t, sub=None):
    s = prs.slides.add_slide(BLANK)
    title(s, t, sub)
    return s


def notes(s, text):
    s.notes_slide.notes_text_frame.text = text


# --------------------------------------------------------------------------- data
def read_bench():
    res = {}
    p = os.path.join(DATA, "bench.txt")
    if os.path.exists(p):
        for line in open(p):
            m = re.match(r"(\S+)\s+([\d.]+) us\s+([\d.]+) TFLOPS", line)
            if m:
                res[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return res


def read_counters():
    res = {}
    p = os.path.join(DATA, "counters.txt")
    if os.path.exists(p):
        for line in open(p):
            parts = line.split()
            if len(parts) >= 2 and not line.startswith("#"):
                res[parts[0]] = parts[1:]
    return res


B = read_bench()
CNT = read_counters()
_bp = os.path.join(DATA, "bench.txt")
PROVISIONAL = os.path.exists(_bp) and "PROVISIONAL" in open(_bp).read()
CAVEAT = "注意：当前数据在 GPU 被其他任务占满时测得，只能作相对参考；空闲时运行 slides/measure.sh 后重新生成"


def tf(name):
    return f"{B[name][1]:.0f}" if name in B else "n/a"


def us(name):
    return f"{B[name][0]:.0f}" if name in B else "n/a"


X0, Y0 = Inches(0.5), Inches(1.6)
FULLW = Inches(12.3)

# =========================================================================== slides
# 1 title
s = prs.slides.add_slide(BLANK)
textbox(s, Inches(0.8), Inches(2.2), Inches(11.7), Inches(1.2),
        [("CDNA3 (MI300X / MI325X) GEMM Kernel 优化实战", dict(bullet=False, bold=True, size=40, color=RED))])
textbox(s, Inches(0.8), Inches(3.5), Inches(11.7), Inches(1.5),
        [("用 FlyDSL 从 v0 写到 v2：MFMA → LDS 协作复用 → 单 wave 软件流水", dict(bullet=False, size=24)),
         ("BF16 输入 / FP32 累加，8192 x 9728 x 8192，gfx942", dict(bullet=False, size=18, color=GRAY))])
textbox(s, Inches(0.8), Inches(5.6), Inches(11.7), Inches(0.8),
        [("代码：gfx942-flydsl-tutorials/kernels/gemm/{v0,v1,v2,bench}.py", dict(bullet=False, size=14, color=GRAY))])

# 2 roadmap
s = slide("路线图：每一步只解决一个瓶颈", "思路：先让数据复用最大化，再用流水把剩下的延迟藏起来")
table(s, X0, Y0, FULLW, [
    ["版本", "CTA / wave 组织", "关键手段", "解决的问题", "TFLOPS"],
    ["v0", "1 wave/CTA，wave 算 128x128", "MFMA 16x16x16 + 寄存器复用", "基本结构；访存/计算比 = 64 flop/B", tf("v0_128x128")],
    ["v1", "4 wave/CTA，256x256，2x2", "LDS 64KB 协作加载 + XOR swizzle", "L2 流量减半；LDS 无 bank conflict", tf("v1")],
    ["v2", "同 v1，4 象限 x (2x2 wave)", "4 region 软件流水 + sched_group_barrier", "1 wave/SIMD 下用 ILP 藏延迟", tf("v2")],
    ["参考", "torch.matmul", "hipBLASLt", "-", tf("torch")],
], col_w=[1.0, 3.0, 3.4, 3.4, 1.5], size=14, h_row=0.6)
textbox(s, X0, Inches(5.0), FULLW, Inches(2), [
    "每一版都先算“应该是多少”（强度 / 寄存器 / LDS 预算），再用 ISA + 计数器 + ATT 验证“实际是多少”",
    "occupancy 始终是 1 wave/SIMD：把寄存器和 LDS 全部用于数据复用，延迟只能靠指令级并行（ILP）隐藏",
] + ([(CAVEAT, dict(bullet=False, size=12, color=RED))] if PROVISIONAL else []))

# 3 hardware
s = slide("CDNA3 硬件要点（gfx942）", "写 kernel 前必须记住的几个数字")
image(s, "arch.png", X0, Y0, w=Inches(7.6))
textbox(s, Inches(8.3), Y0, Inches(4.6), Inches(5.6), [
    "8 XCD x 38 CU = 304 CU；每 CU 4 个 SIMD，wave64",
    "1 wave/SIMD 时每 lane 512 个寄存器：256 VGPR + 256 AGPR（MFMA 累加器可放 AGPR）",
    "LDS 64 KB/CU，32 bank x 4B，128 B/clk，4 个 SIMD 共享",
    "L2 4 MB/XCD（XCD 间不共享）；MALL 256 MB；HBM3E ~6 TB/s",
    "MFMA 16x16x16 bf16：16 cycle/SIMD，峰值 1307 TFLOPS（2.1 GHz）",
    "global→LDS 直通只有 4 B/lane（CDNA4 才有 16 B），所以要经寄存器中转",
    "s_waitcnt vmcnt / lgkmcnt：按发射顺序计数的等待，编译器显式插入",
], size=15)

# 4 vs NVIDIA
s = slide("与 NVIDIA GPU 的对照（以本例为准）")
table(s, X0, Y0, FULLW, [
    ["", "CDNA3 (MI300X/MI325X)", "NVIDIA Hopper (H100)", "对本例的影响"],
    ["执行单位", "wave64，CU = 4 SIMD", "warp32，SM = 4 子分区", "MFMA 操作数排布按 64 lane 设计"],
    ["矩阵指令", "MFMA：操作数在 VGPR/AGPR，同步发射", "wgmma：操作数可直接读 smem，异步", "每个 K 步都要 ds_read 到寄存器"],
    ["片上存储", "LDS 64 KB / CU", "smem 最多 228 KB / SM", "只能放 1 级 256x256x64，无法多级 buffer"],
    ["global→共享", "4 B/lane 直通，常规路径经寄存器", "TMA / cp.async 异步拷贝，不占寄存器", "v2 用寄存器当第二级 buffer，ds_write 占发射槽"],
    ["累加器", "独立 AGPR 文件（1 wave 时 256/lane）", "统一 RF，最多 255/thread", "128x128/wave 的累加器放 AGPR 不挤 VGPR"],
    ["同步", "s_barrier + s_waitcnt 计数器", "mbarrier（事务计数）+ named barrier", "barrier 位置要手工推导 WAR/RAW"],
    ["L2", "每 XCD 独立 4 MB + 256 MB MALL", "统一 50 MB L2", "CTA 调度/重映射影响 L2 命中（下一步）"],
], col_w=[1.5, 3.6, 3.6, 3.6], size=12, h_row=0.62)
notes(s, "要点：NVIDIA 上常见的 3~5 级 cp.async/TMA 多级流水，在 CDNA3 上因 LDS 只有 64KB 且没有宽的异步拷贝而不可行；"
         "取而代之的是寄存器预取 + 单级 LDS 原地刷新，这正是 v2 的设计。")

# 5 methodology
s = slide("分析方法：先算平衡点，再测实际", "每一级存储都有自己的 flop/byte 平衡点")
textbox(s, X0, Y0, Inches(6.4), Inches(5.6), [
    ("算术强度 I = FLOPs / Bytes(该层)", dict(bold=True)),
    ("寄存器层：一个 wave 算 BMxBN，每个 K 读 (BM+BN) 个元素 → I = BM·BN/(BM+BN) flop/B", dict(level=1)),
    ("L2 层：CTA 内共享 → I 随 CTA tile 增大", dict(level=1)),
    ("硬件平衡点 = 峰值算力 / 该层带宽", dict(bold=True)),
    ("每 CU：2048 flop/clk vs LDS 128 B/clk → 需 I_LDS ≥ 16", dict(level=1)),
    ("I < 平衡点 → 带宽受限：加大 tile / 共享（v0→v1）", dict(level=1)),
    ("带宽和 MFMA 都没满 → 延迟受限", dict(bold=True)),
    ("Little's law：在途字节 = 带宽 x 延迟；1 wave/SIMD 时只能靠 ILP（v1→v2）", dict(level=1)),
    ("验证工具", dict(bold=True)),
    ("ISA dump：寄存器、spill、指令配比、s_waitcnt", dict(level=1)),
    ("rocprofv3 --pmc：bank conflict、L2 命中、HBM 流量", dict(level=1)),
    ("ATT：逐指令 stall、wave 时间线、occupancy", dict(level=1)),
], size=16)
code(s, Inches(7.1), Y0, Inches(5.7), Inches(3.0), """
# dump IR + final ISA
FLYDSL_DEBUG_DUMP_ASM=1 FLYDSL_DUMP_IR=1 \\
  FLYDSL_DUMP_DIR=./dumps python3 v2.py
# hardware counters
rocprofv3 --pmc SQ_LDS_BANK_CONFLICT SQ_INSTS_LDS \\
  --kernel-include-regex gemm_v -- python3 bench.py v1
# instruction-level thread trace
rocprofv3 -i trace.yaml -- python3 v0.py
""", size=12)

# 6 MFMA layout
s = slide("MFMA 16x16x16 bf16：一条指令，64 个 lane 怎么分工", "v_mfma_f32_16x16x16_bf16：D(16x16) += A(16x16) · B(16x16)")
image(s, "mfma_layout.png", X0, Y0, w=Inches(8.4))
textbox(s, Inches(9.0), Y0, Inches(4.0), Inches(3.4), [
    "每 lane 提供 A 的 4 个 bf16、B 的 4 个 bf16，得到 C 的 4 个 f32",
    "技巧：一次 16 B load 拿 8 个连续 K，拆成两条 MFMA",
    "A、B 用同一种 K 排列，结果不变 → 全部 load 都是 dwordx4 / ds_read_b128",
], size=14)
code(s, Inches(0.5), Inches(5.2), Inches(12.3), Inches(2.0), """
row, kg = lane % 16, lane // 16                       # lane -> (row, K-group)
a = A[None, k * 4 + kg, row, i].load().bitcast(fx.Int16)    # 8 x bf16 = 16 B
for h in range_constexpr(2):                          # two K=16 MFMAs
    sel = [4 * h + e for e in range(4)]
    acc = rocdl.mfma_f32_16x16x16bf16_1k(T.f32x4, [a.shuffle(a, sel), b.shuffle(b, sel), acc, 0, 0, 0])
""", size=13)

# 7 v0 code
s = slide("v0：最直观的实现 —— 每个 wave 直接从显存读到寄存器", "1 wave / CTA，任务均分到 304 个 CU；FlyDSL 的基本代码结构")
code(s, X0, Y0, Inches(7.4), Inches(5.6), """
@flyc.kernel(known_block_size=(64, 1, 1))
def gemm_v0(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    tid = fx.thread_idx.x
    bid_m, bid_n = fx.block_idx.x, fx.block_idx.y
    # layout algebra: A[M,K] -> (8, K//8, 16, BM//16, grid_m)
    A = fx.make_view(fx.get_iter(A), fx.make_ordered_layout(
            (8, K//8, 16, BM//16, grid_m), 0))
    A = A[None, None, None, None, bid_m]       # slice my tile
    row, kg = tid % 16, tid // 16
    for k, acc in range(fx.Index(0), fx.Index(K//32),
                        fx.Index(1), init=acc_init):   # scf.for
        kk = fx.Int32(k) * 4 + kg
        a_tile = [A[None, kk, row, i].load() ...]       # 16 B
        b_tile = [B[None, kk, row, j].load() ...]
        for i, j, h:                                    # NI*NJ*2
            acc[i*NJ+j] = mfma(a_tile[i], b_tile[j], acc[i*NJ+j])
        results = yield acc                             # loop-carried
    C[row, j, kg*4 + r, i] = results[...][r]            # store
""", size=12)
textbox(s, Inches(8.1), Y0, Inches(4.8), Inches(5.6), [
    ("FlyDSL 三件套", dict(bold=True)),
    ("make_ordered_layout + make_view：把指针变成带布局的张量", dict(level=1)),
    ("切片 A[None, ..., bid_m]：None 保留该维", dict(level=1)),
    ("tensor.load()：一次向量 load（这里 16 B）", dict(level=1)),
    ("循环", dict(bold=True)),
    ("range_constexpr：编译期展开", dict(level=1)),
    ("range(fx.Index(..), init=...)：运行时 scf.for，累加器作为 loop-carried 值 yield", dict(level=1)),
    ("复用", dict(bold=True)),
    ("wave tile 越大，每字节数据做的计算越多：I = BM·BN/(BM+BN)", dict(level=1)),
    ("128x128：A/B 各 32 VGPR，C 256 个累加器 → 寄存器用满", dict(level=1)),
], size=14)

# 8 v0 analysis
s = slide("v0 的问题：寄存器复用到顶后，只剩“等内存”")
table(s, X0, Y0, Inches(7.0), [
    ["wave tile", "强度 (flop/B)", "累加器/lane", "TFLOPS"],
    ["64x64", "32", "64", tf("v0_64x64")],
    ["128x128", "64", "256 (AGPR)", tf("v0_128x128")],
    ["> 128x128", "-", "> 256", "spill"],
], col_w=[1.6, 1.8, 1.8, 1.8], size=14, h_row=0.5)
textbox(s, X0, Inches(3.9), Inches(7.0), Inches(3.4), [
    "4 个 wave 落在同一个 CU 的 4 个 SIMD 上，各自读自己的 A、B 行，CU 内没有共享",
    "每个 K 步：发 load → s_waitcnt vmcnt(0) → MFMA：访存和计算串行",
    "ATT（occupancy.json）：每 CU 4 个 wave 同时上线、同时结束，没有第二轮 wave 可以切换",
    "下一步：让同一 CU 的 4 个 wave 共享 A/B（LDS），把复用从 wave 级扩大到 CTA 级",
], size=15)
image(s, "tiling.png", Inches(7.7), Inches(1.7), w=Inches(5.3))

# 9 v1 dataflow
s = slide("v1：引入 LDS —— 4 个 wave 协作加载、共享复用", "CTA 256x256，BK=64；A、B 各 256x64 bf16 = 32 KB，正好占满 64 KB LDS")
image(s, "tiling.png", X0, Y0, w=Inches(8.0))
textbox(s, X0, Inches(4.5), Inches(12.3), Inches(2.9), [
    "每个 K 步：① 256 线程合并读 A/B（每线程每轮 16 B，8 线程覆盖一行 128 B）→ 暂存寄存器",
    "② barrier → ds_write_b128 写 LDS → barrier   ③ 每个 wave 按 MFMA 排布 ds_read_b128 读自己那 128 行",
    "L2 层强度：64 KB 数据支撑 2·256·256·64 flop → 128 flop/B（v0 的 2 倍）",
    "LDS 层：每 K 步读 128 KB + 写 64 KB ≈ 48 B/clk/CU（LDS 128 B/clk 的 37%）",
], size=15)

# 10 v1 code
s = slide("v1 代码：申请 LDS、合并访存、两次 barrier")
code(s, X0, Y0, Inches(12.3), Inches(5.6), """
@fx.struct
class SharedStorage:
    a: fx.Array[fx.BFloat16, BM * BK, 16]          # 32 KB
    b: fx.Array[fx.BFloat16, BN * BK, 16]          # 32 KB
lds   = fx.SharedAllocator().allocate(SharedStorage).peek()
lds_a = lds.a.view(fx.make_ordered_layout((8, 8, BM), 0))      # [row][chunk][8 elem]

st_c, st_r = tid % 8, tid // 8                     # 8 threads per 128 B row -> coalesced
st_c_sw    = st_c ^ (st_r % 8)                     # XOR swizzle
for k, acc in range(fx.Index(0), fx.Index(K // BK), fx.Index(1), init=acc_init):
    a_st = [A[None, st_c, kt, st_r, p].load() for p in range_constexpr(BM // 32)]  # global -> regs
    gpu.barrier()                                  # everyone finished reading tile k-1
    for p in range_constexpr(BM // 32):
        lds_a[None, st_c_sw, p * 32 + st_r].store(a_st[p])                         # regs -> LDS
    gpu.barrier()                                  # tile k visible to all 4 waves
    for s in range_constexpr(2):                   # LDS -> regs -> MFMA
        ck = (kg + s * 4) ^ (row % 8)
        a_tile = [lds_a[None, ck, wm * WM + i * 16 + row].load() for i in range_constexpr(NI)]
        ...
""", size=13)

# 11 swizzle
s = slide("LDS bank conflict 与 XOR swizzle", "一行 64 个 bf16 = 128 B = 刚好 32 个 bank → 不 swizzle 时同列 8 行撞同一组 bank")
image(s, "swizzle.png", X0, Y0, w=Inches(8.2))
textbox(s, Inches(8.9), Y0, Inches(4.1), Inches(5.6), [
    "ds_read_b128：8 个 lane 一拍（8 x 16 B = 128 B）",
    "读：8 lane = 8 行同一 chunk → 需要 chunk ^ row%8 打散",
    "写：8 lane = 1 行 8 个 chunk → XOR 只是行内置换，天然无冲突",
    "实测 SQ_LDS_BANK_CONFLICT = 0（v1、v2）",
    "XOR 不占额外空间（padding 要占），LDS 满 64 KB 时只能 swizzle",
    "NVIDIA 同理：CUTLASS Swizzle<3,4,3> 也是 16 B 粒度的 XOR",
], size=14)

# 12 v1 problem
s = slide("v1 的瓶颈：搬运与计算完全串行", "1 wave/SIMD：没有别的 wave 可以切换，load 的几百个周期全部暴露")
image(s, "timeline.png", X0, Y0, w=Inches(12.3))
textbox(s, X0, Inches(6.3), FULLW, Inches(1), [
    "TLP（多 wave 切换）已被寄存器/LDS 用满 → 只能用 ILP：在同一个指令流里提前发出与当前计算无关的访存",
], size=15)

# 13 v2 idea
s = slide("v2：单 wave 软件流水的思路", "参照 gfx950-gluon-tutorials / gfx942 / intra_wave")
textbox(s, X0, Y0, Inches(6.2), Inches(5.6), [
    ("要藏的延迟", dict(bold=True)),
    ("global load：数百~上千 cycle → 提前 2 个 K 步发出", dict(level=1)),
    ("ds_read：~100 cycle → 提前 1 个 region（~1024 cycle）发出", dict(level=1)),
    ("为什么不用 LDS 双缓冲？", dict(bold=True)),
    ("256x256x64 一级就 64 KB，LDS 只有 64 KB", dict(level=1)),
    ("BK=32 双缓冲：每行 64 B，一次 load 触 16 条 cache line，实测更慢", dict(level=1)),
    ("做法：寄存器当第二级 buffer", dict(bold=True)),
    ("GR：tile k+2 的 global→寄存器（64 VGPR 暂存）", dict(level=1)),
    ("LW：tile k+1 寄存器→LDS，以“半 tile”为粒度原地刷新", dict(level=1)),
    ("LR：下一个 region 的操作数 LDS→寄存器", dict(level=1)),
], size=15)
textbox(s, Inches(7.0), Y0, Inches(5.9), Inches(5.6), [
    ("对照 NVIDIA", dict(bold=True)),
    ("Hopper：TMA 异步拷贝 + 3~5 级 smem 环形缓冲，数据不经寄存器", dict(level=1)),
    ("CDNA4（gfx950）：buffer_load...lds 16 B 直通 + 160 KB LDS 双缓冲", dict(level=1)),
    ("CDNA3：只能“寄存器预取 + 单级 LDS 原地刷新”，代价是每 K 步 16 条 ds_write 进入指令流", dict(level=1)),
    ("流水深度与 gfx950 双缓冲相同", dict(bold=True)),
    ("GR(k+2) → LW(k+1)：一整个 K 步 ≈ 4096 cycle 藏 HBM/L2 延迟", dict(level=1)),
    ("LR(k+1) → DOT(k+1)：一个 region ≈ 1024 cycle 藏 LDS 延迟", dict(level=1)),
], size=15)

# 14 v2 split
s = slide("v2 切分：A_top/A_bot x B_left/B_right → 4 个象限", "每个 wave 在每个象限各算 64x64；一个 region = 一个象限 = 64 条 MFMA")
image(s, "tiling.png", X0, Y0, w=Inches(8.0))
table(s, Inches(8.7), Y0, Inches(4.2), [
    ["LDS slot", "内容", "大小"],
    ["A_top", "A 行 0-127 x 64K", "16 KB"],
    ["A_bot", "A 行 128-255", "16 KB"],
    ["B_left", "B 行(N) 0-127", "16 KB"],
    ["B_right", "B 行(N) 128-255", "16 KB"],
], col_w=[1.2, 2.0, 1.0], size=13, h_row=0.5)
textbox(s, X0, Inches(4.6), FULLW, Inches(2.7), [
    "为什么切成半 tile：LDS 的刷新粒度从“整块 64 KB”变成“16 KB 的 slot”，一个 slot 读完就能立刻重写，不必等整块用完",
    "region 顺序 C_tl → C_bl → C_tr → C_br：相邻 region 共享一个操作数（b_left、a_bot、a_top 各复用两次）",
    "每 K 步：256 MFMA（4096 cycle/SIMD），32 ds_read_b128，16 ds_write_b128，16 global_load_dwordx4",
], size=15)

# 15 v2 regions (key)
s = slide("v2 流水全景：一个 K 步内的 4 个 region 与 LDS slot 生命周期", "★ 最关键的一页")
image(s, "v2_regions.png", Inches(0.75), Inches(1.55), h=Inches(5.85))
notes(s, "读图方法：上半部分是同一个指令流里在每个 region 发出的四类操作；下半部分是四个 LDS slot 在这个 K 步里装的是哪个 tile，"
         "以及什么时候被读/写。每个 slot 的写都夹在它的上一次读和下一次读之间，并且中间各隔一个 barrier。")

# 16 barrier proof
s = slide("为什么每个 K 步只要 2 个 barrier？—— 逐个 slot 检查读写冲突", "LDS 由 4 个 wave 共享：A 写的数据 B 读，必须有 barrier 隔开")
table(s, X0, Y0, Inches(8.2), [
    ["slot", "上次读 (WAR)", "写", "下次读 (RAW)", "隔开它们的 barrier"],
    ["B_left", "R2 (上一 K 步)", "R0", "R2", "Y(上步末) / X"],
    ["A_top", "R3 (上一 K 步)", "R1", "R3", "Y / X"],
    ["A_bot", "R0", "R2", "R0 (下一 K 步)", "X / Y"],
    ["B_right", "R1", "R3", "R1 (下一 K 步)", "X / Y"],
], col_w=[1.2, 1.9, 0.8, 1.9, 2.4], size=13, h_row=0.55)
textbox(s, X0, Inches(4.7), Inches(8.2), Inches(2.6), [
    "X = region1 与 region2 之间的 s_barrier，Y = region3 之后（K 步末尾）的 s_barrier",
    "每个 slot 的“读→写”和“写→读”都恰好跨过 X 或 Y 之一 → 2 个 barrier 足够",
    "prologue 末尾多一个 barrier：保证 prologue 里读 b_left/a_top 完成后，R0 才能重写 B_left",
], size=14)
code(s, Inches(8.9), Y0, Inches(4.0), Inches(5.6), """
# one K-step
R0: DOT C_tl  LR A_bot(k)
    LW B_left(k+1)  GR B_left(k+2)
R1: DOT C_bl  LR B_right(k)
    LW A_top(k+1)   GR A_top(k+2)
---------- barrier X ----------
R2: DOT C_tr  LR B_left(k+1)
    LW A_bot(k+1)   GR A_bot(k+2)
R3: DOT C_br  LR A_top(k+1)
    LW B_right(k+1) GR B_right(k+2)
---------- barrier Y ----------
""", size=12)

# 17 register budget
s = slide("流水的代价：寄存器预算（每 lane，1 wave/SIMD）")
table(s, X0, Y0, Inches(7.6), [
    ["寄存器文件", "存放", "个数"],
    ["AGPR 256", "C 累加器：4 象限 x 64x64 f32 / 64 lane", "256（满）"],
    ["VGPR 256", "DOT 操作数 a_top/a_bot/b_left/b_right 各 64x64 bf16", "4 x 32 = 128"],
    ["", "GR 暂存：4 个半 tile 各 128x64 bf16 / 256 lane", "4 x 16 = 64"],
    ["", "地址、循环变量等", "其余"],
], col_w=[1.6, 4.4, 1.6], size=13, h_row=0.55)
textbox(s, X0, Inches(4.6), Inches(7.6), Inches(2.6), [
    "编译结果：VGPR 512（含 AGPR 256），循环内无 spill；28 个 spill 只在循环外（peel）",
    "LLVM 会把一部分操作数放进 AGPR、累加器放进 VGPR —— gfx90a+ MFMA 可直接读写 AGPR",
    "NVIDIA 对照：wgmma 操作数直接读 smem、TMA 不占寄存器，所以没有这 64 个暂存寄存器的压力",
], size=14)
code(s, Inches(8.4), Y0, Inches(4.5), Inches(3.0), """
.vgpr_count:       512
.agpr_count:       256
.vgpr_spill_count: 28   # outside the loop
.group_segment_fixed_size: 65536
# loop body: 256 v_mfma, 32 ds_read_b128,
#            16 ds_write_b128,
#            16 global_load_dwordx4, 2 s_barrier
""", size=12)

# 18 v2 code
s = slide("v2 代码：一个 K 步 = 4 个 region", "load/store/compute 写成小函数，region 内按“DOT → LR → LW → GR”排列")
code(s, X0, Y0, Inches(7.3), Inches(5.7), """
def kstep(acc, a_top, b_left, g, kt2, last):
    g_bl, g_at, g_ab, g_br = g
    # region 0: C_tl
    acc[0] = dot(acc[0], a_top, b_left)
    a_bot = lr(lds_a, 1, wm)            # LR A_bot(k)
    lw(lds_b, 0, g_bl)                  # LW B_left(k+1)
    g_bl = gr(B, 0, kt2)                # GR B_left(k+2)
    interleave(8, 4, 4)
    # region 1: C_bl
    acc[1] = dot(acc[1], a_bot, b_left)
    b_right = lr(lds_b, 1, wn)
    lw(lds_a, 0, g_at); g_at = gr(A, 0, kt2)
    interleave(8, 4, 4)
    gpu.barrier()                       # X
    # region 2: C_tr / region 3: C_br  (same pattern)
    ...
    gpu.barrier()                       # Y
    return acc, a_top, b_left, [g_bl, g_at, g_ab, g_br]
""", size=12)
code(s, Inches(8.0), Y0, Inches(4.9), Inches(5.7), """
# loop-carried state:
#   64 acc + a_top/b_left (k+1)
#   + 16 staging vectors (k+1)
state0 = acc0 + a_top + b_left + g
for k, st in range(fx.Index(0),
                   fx.Index(NK - 2),
                   fx.Index(1), init=state0):
    acc, a_top, b_left, g = unpack(st)
    acc, a_top, b_left, g = kstep(
        acc, a_top, b_left, g,
        fx.Int32(k) + 2, last=0)
    results = yield flat(acc, a_top,
                         b_left, g)
# peel: NK-2 (LW, no GR), NK-1 (drain)
kstep(..., last=1); kstep(..., last=2)
""", size=12)

# 19 scheduling
s = slide("让编译器“穿插”发射：sched_group_barrier", "一个 region：64 MFMA 里均匀插入 8 ds_read + 4 ds_write + 4 global_load")
code(s, X0, Y0, Inches(6.0), Inches(2.4), """
def interleave(n_dsrd, n_dswr, n_vmem):
    for _ in range_constexpr(n_dsrd):
        rocdl.sched_mfma(2); rocdl.sched_dsrd(1)
    for _ in range_constexpr(n_dswr):
        rocdl.sched_mfma(2); rocdl.sched_dswr(1)
    for _ in range_constexpr(n_vmem):
        rocdl.sched_mfma(1); rocdl.sched_vmem(1)
    rocdl.sched_mfma(64 - 2*n_dsrd - 2*n_dswr - n_vmem)
""", size=12)
textbox(s, Inches(6.8), Y0, Inches(6.1), Inches(2.5), [
    "为什么每个 LDS 访问配 2 条 MFMA：128 B/clk 的 LDS 由 4 个 SIMD 共享，一条 b128（1 KB）每 SIMD 平均占 32 cycle = 2 条 16-cycle MFMA",
    "一条 ds_write_b128 发射约 20 cycle（4 地址 + 16 数据）> 1 条 MFMA → ds_write 是 CDNA3 上藏不住的主要成本",
    "s_waitcnt vmcnt(15..12)：16 条 global load 在途，每次只等最老的那条",
], size=13)
image(s, "v2_isa_strip.png", Inches(0.9), Inches(4.15), h=Inches(3.2))

# 20 prologue / peel
s = slide("Prologue 与 Peel：流水的启动和排空")
table(s, X0, Y0, FULLW, [
    ["阶段", "做什么", "要点"],
    ["prologue", "tile0：GR→LW（4 个半 tile）；GR B_left(1)；barrier；GR A_top/A_bot/B_right(1)；LR b_left(0)、a_top(0)；barrier",
     "GR 发射顺序与循环一致（B_left, A_top, A_bot, B_right），s_waitcnt 计数才能保持 vmcnt(15..12)"],
    ["主循环 k = 0..NK-3", "4 region：DOT + LR + LW(k+1) + GR(k+2)", "2 barrier / K 步，循环不展开"],
    ["peel 1 (k=NK-2)", "仍然 LW 最后一个 tile，不再 GR", "kstep(last=1)"],
    ["peel 2 (k=NK-1)", "只做 DOT；去掉 LW 和下一 tile 的 LR", "kstep(last=2)"],
    ["epilogue", "4 个象限的累加器写回 C", "每 lane 4 个连续 M 行，同一 N 列"],
], col_w=[2.2, 5.6, 4.5], size=13, h_row=0.75)

# 21 results
s = slide("结果", "GEMM 8192 x 9728 x 8192，BF16 → FP32，单卡 MI325X，20 次中位数")
image(s, "perf.png", X0, Y0, w=Inches(7.4))
rows = [["版本", "时间 (us)", "TFLOPS"]]
for n in ("v0_64x64", "v0_128x128", "v1", "v2_nosched", "v2", "torch"):
    rows.append([n, us(n), tf(n)])
table(s, Inches(8.2), Y0, Inches(4.6), rows, col_w=[1.8, 1.4, 1.4], size=13, h_row=0.42)


def cval(case, key):
    for item in CNT.get(case, []):
        if item.startswith(key + "="):
            return item.split("=", 1)[1]
    return "n/a"


ctab = [["计数器 / 次", "L2 命中", "L2 外读取", "bank 冲突"]]
for n in ("v0_128x128", "v1", "v2"):
    ctab.append([n, cval(n, "L2_hit"), cval(n, "EA_read"), cval(n, "bank_conflict")])
table(s, Inches(8.2), Inches(4.75), Inches(4.6), ctab, col_w=[1.45, 1.0, 1.2, 0.95], size=12, h_row=0.4)
textbox(s, X0, Inches(5.15), Inches(7.5), Inches(2.2), [
    "v0 64→128：寄存器层强度翻倍，2.1x",
    "v1：L2 之外的读取 5.6 GB → 1.2 GB（A+B 本身只有 0.29 GB），+30%",
    "v2：流水把访存藏到 MFMA 后面，+47%；sched_group_barrier 单独贡献 +12%",
    "v2 649 TFLOPS，略高于 hipBLASLt（633）；约为 2.1 GHz 峰值的 50%",
], size=13)
if PROVISIONAL:
    textbox(s, X0, Inches(6.9), Inches(7.6), Inches(0.5), [(CAVEAT, dict(bullet=False, size=12, color=RED))])

# 22 next
s = slide("剩余瓶颈与下一步")
textbox(s, X0, Y0, FULLW, Inches(5.6), [
    ("ds_write 发射成本（v2 已知天花板）", dict(bold=True)),
    ("16 条 ds_write_b128/K 步，每条 ~20 cycle，单个指令流里藏不住", dict(level=1)),
    ("→ inter_wave：8 wave/CTA，两组 wave 乒乓，一组做 MFMA 时另一组做访存（参考实现 ~99.7% MFMA 效率）", dict(level=1)),
    ("XCD 感知的 CTA 重映射 + GROUP_SIZE_M", dict(bold=True)),
    ("CTA 轮流分到 8 个 XCD，每个 XCD 独立 L2 → 重排 CTA id，让同一 XCD 的 CTA 共享 A/B 行", dict(level=1)),
    ("epilogue 写回加宽", dict(bold=True)),
    ("当前每 lane 4 个标量 store；转置 MFMA 布局后可用 dwordx4（LDS 已满，不能借 LDS 转置）", dict(level=1)),
    ("buffer_load + 硬件越界检查", dict(bold=True)),
    ("支持任意 M/N/K，不再要求整除 tile", dict(level=1)),
    ("功耗墙", dict(bold=True)),
    ("750 W 下 MFMA 密集时频率下降，TFLOPS 要结合实际频率解读", dict(level=1)),
], size=16)

# 23 cheat sheet
s = slide("工具速查")
code(s, X0, Y0, FULLW, Inches(5.6), """
# correctness + timing (all versions + hipBLASLt)
python3 bench.py
# IR / ISA dump: registers, spills, instruction mix
FLYDSL_RUNTIME_CACHE_DIR=/tmp/fresh FLYDSL_DEBUG_DUMP_ASM=1 FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=./dumps python3 v2.py
grep -E "vgpr_count|agpr_count|spill_count|group_segment_fixed_size" dumps/gemm_v2_0/21_final_isa.s
# hardware counters
rocprofv3 --pmc SQ_LDS_BANK_CONFLICT SQ_INSTS_LDS --kernel-include-regex gemm_v -- python3 bench.py v1 v2
rocprofv3 --pmc TCC_HIT_sum TCC_MISS_sum TCC_EA0_RDREQ_sum TCC_EA0_RDREQ_32B_sum --kernel-include-regex gemm_v -- python3 bench.py v2
# thread trace (ATT): per-instruction stalls, wave timeline, occupancy
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 rocprofv3 -i trace.yaml -- python3 v2.py
# slides
cd slides && python3 make_figs.py && python3 make_ppt.py
""", size=13)

out = os.path.join(HERE, "CDNA3_GEMM_FlyDSL.pptx")
prs.save(out)
print("saved", out, len(prs.slides), "slides")
