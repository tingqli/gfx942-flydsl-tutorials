"""Build a concise presentation of ../README.md without changing the older deck.

Run from any directory: python make_readme_ppt.py
Dependencies: pip install -r requirements-readme-ppt.txt
Output: CDNA3_GEMM_README.pptx alongside this script.

Slide text is deliberately condensed from the README; numbers are read from
data/bench.txt and data/counters.txt, not embedded in pre-existing charts.
SVGs are rendered to PNG in memory because Office does not reliably animate SMIL.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import cairosvg
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Inches, Pt


HERE = Path(__file__).resolve().parent
FIG = HERE / "figs"
OUTPUT = HERE / "CDNA3_GEMM_README.pptx"
README = (HERE / ".." / "README.md").resolve()

BG = "0C1628"
PANEL = "17283E"
PANEL2 = "203650"
WHITE = "ECF3FA"
FAINT = "AEC2D7"
TEAL = "69D9C6"
BLUE = "8BC6FF"
VIOLET = "C7A9F2"
GOLD = "E8BD75"


def rgb(value: str) -> RGBColor:
    return RGBColor.from_string(value)


def inch(value: float):
    return Inches(value)


def box(slide, x, y, w, h, fill=PANEL, outline=None, radius=True):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        inch(x), inch(y), inch(w), inch(h),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.fill.background() if outline is None else None
    if outline:
        shape.line.color.rgb = rgb(outline)
    if radius:
        shape.adjustments[0] = 0.14
    return shape


def text(slide, value, x, y, w, h, size=17, color=WHITE, bold=False, align=None):
    shape = slide.shapes.add_textbox(inch(x), inch(y), inch(w), inch(h))
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = inch(0.03)
    tf.margin_top = tf.margin_bottom = inch(0.02)
    tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    para = tf.paragraphs[0]
    if align is not None:
        para.alignment = align
    run = para.add_run()
    run.text = value
    run.font.name = "Microsoft YaHei"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)
    return shape


def bullets(slide, items, x, y, w, size=18, line_h=.7, color=WHITE):
    for i, item in enumerate(items):
        text(slide, "•", x, y + i * line_h, .3, line_h, size, TEAL, bold=True)
        text(slide, item, x + .32, y + i * line_h, w - .32, line_h, size, color)


def note(slide, content):
    slide.notes_slide.notes_text_frame.text = content


deck = Presentation()
deck.slide_width = inch(13.333)
deck.slide_height = inch(7.5)
BLANK = deck.slide_layouts[6]


def page(title, subtitle="", section="", notes=""):
    slide = deck.slides.add_slide(BLANK)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(BG)
    box(slide, .42, .3, .075, .62, TEAL, radius=False)
    text(slide, title, .67, .27, 11.9, .53, 26, WHITE, bold=True)
    if subtitle:
        text(slide, subtitle, .67, .86, 11.9, .37, 13, FAINT)
    text(slide, "CDNA3 · gfx942 / FlyDSL", .52, 7.17, 7, .18, 9, FAINT)
    text(slide, section, 10.4, 7.17, 2.4, .18, 9, FAINT, align=PP_ALIGN.RIGHT)
    text(slide, f"{len(deck.slides):02d}", 12.58, 7.0, .38, .32, 10, TEAL, bold=True)
    if notes:
        note(slide, notes)
    return slide


_image_cache: dict[tuple[str, tuple[int, int] | None], bytes] = {}


def figure_bytes(name: str, crop: tuple[int, int] | None = None):
    key = (name, crop)
    if key not in _image_cache:
        path = FIG / name
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = cairosvg.svg2png(url=str(path), output_width=2400) if path.suffix == ".svg" else path.read_bytes()
        if crop:
            with Image.open(io.BytesIO(raw)) as im:
                # Cropping is a fraction of original image height in thousandths.
                top, bottom = (round(im.height * n / 1000) for n in crop)
                out = io.BytesIO()
                im.crop((0, top, im.width, bottom)).save(out, format="PNG")
                raw = out.getvalue()
        _image_cache[key] = raw
    return io.BytesIO(_image_cache[key])


def image(slide, name, x, y, w, h, crop=None, dark_backdrop=True):
    with Image.open(figure_bytes(name, crop)) as im:
        iw, ih = im.size
    ratio = min(w / iw, h / ih)
    ww, hh = iw * ratio, ih * ratio
    if dark_backdrop:
        box(slide, x, y, w, h, PANEL)
    slide.shapes.add_picture(figure_bytes(name, crop), inch(x + (w - ww) / 2), inch(y + (h - hh) / 2),
                             width=inch(ww), height=inch(hh))


def card(slide, label, content, x, y, w, h, accent=TEAL, body_size=17):
    box(slide, x, y, w, h, PANEL, outline="3F5C78")
    box(slide, x + .17, y + .17, .08, .43, accent, radius=False)
    text(slide, label, x + .36, y + .13, w - .53, .52, 17, accent, bold=True)
    text(slide, content, x + .25, y + .75, w - .52, h - .95, body_size)


def rows(slide, headers, body, widths, x=.62, y=1.53, row_h=.7, font=15):
    starts = [x]
    for width in widths[:-1]:
        starts.append(starts[-1] + width)
    for i, cells in enumerate([headers, *body]):
        fill = PANEL2 if i == 0 else (PANEL if i % 2 else "1D3047")
        box(slide, x, y + i * row_h, sum(widths), row_h - .055, fill, radius=False)
        for j, value in enumerate(cells):
            text(slide, str(value), starts[j] + .12, y + i * row_h + .055,
                 widths[j] - .2, row_h - .17, font if i else font - 1,
                 TEAL if i == 0 else WHITE, bold=(i == 0))


def bench():
    results = {}
    for line in (HERE / "data" / "bench.txt").read_text(encoding="utf-8").splitlines():
        match = re.match(r"(\S+)\s+([\d.]+) us\s+([\d.]+) TFLOPS", line)
        if match:
            results[match[1]] = (float(match[2]), float(match[3]))
    required = {"v0_64x64", "v0_128x128", "v1", "v2_nosched", "v2", "torch"}
    if not required.issubset(results):
        raise ValueError(f"Missing benchmark cases: {required - results.keys()}")
    return results


def counters():
    result = {}
    for line in (HERE / "data" / "counters.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            continue
        words = line.split()
        if words:
            result[words[0]] = dict(item.split("=", 1) for item in words[1:])
    return result


B = bench()
C = counters()

# 01 – cover
s = deck.slides.add_slide(BLANK)
s.background.fill.solid()
s.background.fill.fore_color.rgb = rgb(BG)
box(s, .72, 1.05, .12, 4.9, TEAL, radius=False)
text(s, "CDNA3 / gfx942", 1.12, 1.08, 11, .55, 24, TEAL, bold=True)
text(s, "BF16 GEMM\n从 v0 写到 v2", 1.12, 2.02, 11.2, 1.78, 42, WHITE, bold=True)
text(s, "用 FlyDSL 理解 wave64、MFMA、LDS 和单 wave 软件流水", 1.12, 4.2, 10.7, .6, 21, FAINT)
text(s, "基于 kernels/gemm/README.md · 8192 × 9728 × 8192 · BF16 输入 / FP32 输出", 1.12, 5.46, 10.7, .47, 14, FAINT)
note(s, "本文是 README 的精简演示版；两份文件中的硬件峰值为参考模型，测量来自仓库记录的特定形状，不保证其它机器可复现。")

# 02 – route
s = page("路线：先提高复用，再藏住延迟", "每个版本只引入刚好需要的 AMD 硬件知识", "路线图")
for idx, (name, label, detail, perf) in enumerate([
    ("v0", "wave 内寄存器复用", "1 wave/CTA · 128×128/wave\n直接 global load + MFMA", B["v0_128x128"][1]),
    ("v1", "CTA 级 LDS 共享", "4 wave/CTA · 256×256/CTA\nA/B 放 LDS，XOR swizzle", B["v1"][1]),
    ("v2", "单 wave 软件流水", "保持大 tile · GR/LW/LR 与 DOT 交错\n`rocdl.sched_*` 帮助排布", B["v2"][1]),
]):
    x = .65 + idx * 4.2
    box(s, x, 1.65, 3.95, 4.63, PANEL, outline="486788")
    text(s, name, x + .25, 1.95, 2, .5, 28, (BLUE, TEAL, VIOLET)[idx], bold=True)
    text(s, label, x + .25, 2.67, 3.5, .61, 19, WHITE, bold=True)
    text(s, detail.replace("`", ""), x + .25, 3.5, 3.46, 1.72, 16, FAINT)
    text(s, f"{perf:.0f} TFLOPS", x + .25, 5.53, 3.38, .48, 23, (BLUE, TEAL, VIOLET)[idx], bold=True)
text(s, "性能源：slides/data/bench.txt；PyTorch 对照输出 BF16，不宜作为同精度基线。", .75, 6.63, 11.9, .35, 12, FAINT)

# 03 – hardware primer
s = page("硬件心智模型：CU ≠ 一个大 warp", "用单个 wave 驱动大 tile，会先用满累加寄存器", "v0 · 硬件")
card(s, "wave64 与 SIMD", "CU = 4 SIMD；每个 wave 有 64 lane。v0 的 CTA 是 1 wave；v1/v2 是 4 wave/CTA。", .6, 1.58, 4.0, 3.3, BLUE)
card(s, "MFMA 与 AGPR", "BF16 矩阵指令每次算 16×16×16；128×128/wave 的 FP32 累加器占 256 AGPR/lane。", 4.68, 1.58, 4.0, 3.3, VIOLET)
card(s, "占用率取舍", "大 tile 增强寄存器复用，但本例仅 1 wave/SIMD，不能再靠切 wave 隐藏长访存延迟。", 8.76, 1.58, 4.0, 3.3, TEAL)
text(s, "按 MI300X 标称 304 CU × 4 SIMD × 512 BF16 FLOP/cycle × 2.1 GHz ≈ 1307 TFLOPS。", .9, 5.48, 11.7, .75, 19, WHITE)
text(s, "峰值是参考模型；不等于已测频率或本次编译的实际 MFMA 利用率。", .9, 6.29, 11.7, .42, 14, FAINT)

# 04 – code and layout
s = page("v0：先把 FlyDSL 的 layout 读懂", "布局决定谁读哪 16 B；运行时循环携带累加器", "v0 · 代码")
box(s, .62, 1.52, 7.65, 4.93, PANEL)
code_lines = [
    "A = fx.make_view(fx.get_iter(A),",
    "    fx.make_ordered_layout((8, K//8, 16, BM//16, grid_m), 0))",
    "A = A[None, None, None, None, bid_m]",
    "row, kg = tid % 16, tid // 16",
    "for k, acc in range(fx.Index(0), fx.Index(K//32),",
    "                    fx.Index(1), init=acc_init):",
    "    a_tile = A[None, kk, row, i].load()",
    "    acc = mfma(a_tile, b_tile, acc)",
    "    results = yield acc",
]
for i, line in enumerate(code_lines):
    text(s, line, .87, 1.83 + i * .46, 7.1, .41, 13, BLUE if i in (0, 4, 7) else WHITE)
card(s, "读法", "mode 0 最连续：8 个 BF16 = 16 B / lane。\nNone 保留维度；bid_m 固定本 CTA。\nrange_constexpr 展开；fx.Index(..., init=...) 是运行时循环。", 8.5, 1.52, 4.2, 4.93, TEAL, body_size=16)
note(s, "代码为精简伪代码，完整实现见 ../v0.py；A/B 使用相同 K 重排，C 在 AGPR 中逐次累计。")

# 05, 06 – key README SVGs
for heading, subtitle, asset, notes, fact in [
    ("MFMA lane 排布：输入和结果不同", "左：A/B 每 lane 给 4 个 K；右：C 每 lane 保留同列 4 个 FP32", "mfma_lane_mapping.svg", "原图位于 slides/figs/mfma_lane_mapping.svg；同一 wave 64 lane，共两条 MFMA 消费连续 8 个 K 的向量加载。", "两条 K16 MFMA\n合计覆盖 K=32"),
    ("v0 / v1：同宽的 load，利用率不同", "16 B/lane 不变；换 lane→地址映射，单次指令从 16 条半满 line 变为 8 条满 line", "mfma_coalescing.svg", "这里是逻辑 cache-line 利用率，不是实测 HBM 流量；cache 命中和跨步复用会改变外部流量。", "v0：16 行 × 半条\nv1：8 行 × 整条"),
]:
    s = page(heading, subtitle, "v0 · MFMA", notes=notes)
    image(s, asset, .55, 1.42, 10.26, 5.43)
    card(s, "记住", fact, 10.94, 2.75, 1.85, 2.0, TEAL, body_size=14)

# 07 – v0 measured performance
s = page("v0：tile 变大，复用翻倍；再大就 spill", "寄存器算术强度 I = BM·BN / (BM+BN) FLOP/B（BF16）", "v0 · 分析")
rows(s, ["wave tile", "强度", "AGPR / lane", "时间", "吞吐"], [
    ["64×64", "32", "64", f"{B['v0_64x64'][0]:.0f} μs", f"{B['v0_64x64'][1]:.0f} T"],
    ["128×128", "64", "256", f"{B['v0_128x128'][0]:.0f} μs", f"{B['v0_128x128'][1]:.0f} T"],
    ["128×256", "85", "512（spill）", "单独试验", "非统一基准"],
], [2.4, 1.85, 2.45, 2.55, 2.5], row_h=.81, font=18)
card(s, "为什么下一步要加 LDS？", "128×128/wave 已吃满 256 个累加寄存器；更大的 tile 会 spill。4 wave/CTA 共享 A/B 才能在不降低 wave tile 的前提下继续提高复用。", .65, 5.22, 12.0, 1.44, VIOLET, body_size=16)

# 08 – TLP versus ILP
s = page("延迟怎么藏：TLP 与 ILP", "本例 1 wave/SIMD，选择在一条指令流里提前发加载，再与 MFMA 交错", "v0 → v2")
image(s, "tlp_ilp.svg", .7, 1.36, 8.46, 5.57)
card(s, "TLP", "多个驻留 wave 轮流填空：需要为每个 wave 保留寄存器。", 9.35, 1.75, 3.32, 2.1, BLUE, 16)
card(s, "ILP", "只驻留 1 个 wave：先发未来 K 的加载，等数据时继续算当前块。", 9.35, 4.01, 3.32, 2.26, TEAL, 16)

# 09 – v1 GEMM tile
s = page("v1：4 个 wave 共用一份 A/B", "CTA C 256×256 · BK=64 · A/B 共占 64 KiB LDS", "v1 · tiling")
image(s, "v1_tiling.svg", .6, 1.32, 9.74, 5.62)
card(s, "CTA 级复用", "A 的上半给 wave 0、2；B 的左半给 wave 0、1。每 wave 仍算 128×128，C 常驻 AGPR。", 10.5, 2.32, 2.28, 2.8, TEAL, 15)

# 10 – v1 stages
s = page("v1：每个 K 步的三段搬运", "LDS 提升的是 CTA 内复用，但还没有软件流水", "v1 · 数据流")
for i, (head, detail, accent) in enumerate([
    ("① Global → VGPR", "8 个相邻 lane 读满一行的 128 B；一次读 8 行。", BLUE),
    ("② VGPR → LDS", "先 barrier 确保旧 tile 不再被读，再写入 swizzled LDS；第二个 barrier 让新数据可见。", GOLD),
    ("③ LDS → VGPR → MFMA", "每个 wave 读自己的 128×128 子块操作数；结果保持在 AGPR。", VIOLET),
]):
    card(s, head, detail, .66 + 4.2*i, 1.84, 3.95, 3.8, accent, 17)
text(s, "两次全 CTA barrier + vmcnt(0) 使搬运和计算基本串行；v1 更快来自复用，并非延迟已被隐藏。", .75, 6.19, 11.95, .53, 17, FAINT)

# 11 – swizzle
s = page("v1：bank conflict 与 XOR swizzle", "同列 8 行同时读：row stride=128 B，会落在相同 bank 组", "v1 · LDS")
image(s, "swizzle.png", .58, 1.46, 9.25, 4.86)
card(s, "核心映射", "chunk_phys = chunk ^ (row % 8)\n\n无额外 LDS padding；本例的 SQ_LDS_BANK_CONFLICT = 0。", 9.99, 2.02, 2.75, 3.2, GOLD, 15)
text(s, "与 Hopper 的 shared-memory XOR 思路相同；CDNA3 本例由地址计算或 FlyDSL layout 表达，不是普通 LDS 访问自动 swizzle。", .75, 6.45, 11.9, .51, 14, FAINT)

# 12 – v1 outcomes
s = page("v1：降低 L2 外部接口流量，但等待仍在", "这些是记录中的接口读取请求字节数；包含可能被 MALL 命中的数据", "v1 · 证据")
for i, (name, perf, bytes_, color) in enumerate([
    ("v0 128×128", B["v0_128x128"][1], C["v0_128x128"]["EA_read"], BLUE),
    ("v1 256×256", B["v1"][1], C["v1"]["EA_read"], TEAL),
]):
    x = .7 + i * 6.15
    card(s, name, f"{perf:.0f} TFLOPS\nL2 外读取：{bytes_}\nLDS bank conflict：0", x, 1.75, 5.84, 3.49, color, 21)
text(s, f"L2 外部读取下降约 {5.59/1.24:.1f}×；v1 对 v0 加速约 {B['v1'][1]/B['v0_128x128'][1]:.2f}×，仍要逐步等待访存。", .9, 5.82, 11.5, .76, 20, WHITE)
note(s, "流量来自 counter_collection 的 TCC_EA0_RDREQ 和 32B 请求；不能解读为 HBM 实际读字节数。详情见 README §2.4。")

# 13/14 Hopper diagram cropped into two legible slides
s = page("对照 Hopper：producer 和 consumer 各干什么", "WG0 发 TMA；WG1/WG2 共享 B、各读 A 一半，累加各自的 C", "Hopper · 分工")
image(s, "hopper_warp_specialized.svg", .55, 1.42, 12.20, 5.57, crop=(0, 505))
s = page("对照 Hopper：full / empty 按 stage 握手", "TMA 填下一 stage 的同时，两个 consumer WGMMA 算当前 stage", "Hopper · 时序")
image(s, "hopper_warp_specialized.svg", .55, 1.42, 12.20, 5.57, crop=(505, 1000))
note(s, "两组 consumer 必须等上一级 WGMMA 不再读 smem 后各 arrive(empty[prev]) 一次，producer 后续才能覆写。图不是逐周期时序。")

# 15 – contrast
s = page("为什么 Hopper 的多级 warp specialization 不能原样搬过来", "CDNA3 的同步、LDS 容量和 operand 路径都不同", "v2 · 约束")
rows(s, ["Hopper", "本例 CDNA3", "对应做法"], [
    ["TMA → smem", "global → VGPR → LDS", "GR 暂存 + LW 写回"],
    ["WGMMA 直接读 smem", "MFMA 先读到 VGPR/AGPR", "LR 提前读 + DOT"],
    ["多级 smem + mbarrier", "64 KiB LDS 只放本例 1 级", "半 tile 原地更新 + 2 barrier"],
    ["setmaxnreg 动态让寄存器", "本例寄存器数编译期固定", "1 wave/SIMD，优先保留大 tile"],
], [3.05, 4.1, 5.05], row_h=.9, font=17)
text(s, "结论：用同一 wave 内的 ILP，而不是直接复制 producer / consumer warpgroup。", .9, 6.38, 11.5, .49, 18, TEAL, bold=True)

# 16 – v2 staging
s = page("v2：64 KiB LDS 放一级，VGPR 做第二级", "一个 CTA 的 A/B tile = 256×64×2B×2 = 64 KiB，无法保持相同 BK 再放一级", "v2 · 设计")
for idx, (name, subtitle, content, accent) in enumerate([
    ("GR", "提前 2 个 K 步", "global → VGPR\n为未来的 LDS 写入预取", BLUE),
    ("LW", "提前 1 个 K 步", "VGPR → LDS\n刷新已读完的 16 KiB 半 tile", GOLD),
    ("LR", "提前一个 region", "LDS → VGPR\n准备下一象限的 MFMA 操作数", TEAL),
    ("DOT", "计算当前 K", "VGPR 操作数 → MFMA\n累加当前 C 象限", VIOLET),
]):
    x = .62 + idx * 3.18
    box(s, x, 1.75, 2.98, 3.79, PANEL, outline=accent)
    text(s, name, x+.2, 1.94, 2.5, .58, 27, accent, True)
    text(s, subtitle, x+.2, 2.72, 2.55, .46, 16, FAINT)
    text(s, content, x+.2, 3.38, 2.57, 1.48, 16)
text(s, "同一 region 的四类指令在 ISA 中交错，不是依次执行四个完整阶段。", .83, 6.2, 11.8, .51, 17, FAINT)

# 17 – one K step table
s = page("v2：一个 K 步 = 4 region + 2 barrier", "同一 CTA，四个 wave 每人负责每个象限 64×64，1 region / wave 64 条 MFMA", "v2 · 调度")
rows(s, ["region", "DOT 当前 k", "LR 下一用", "LW 下一 K", "GR 再下一 K"], [
    ["R0", "C_tl: A_top · B_left", "A_bot(k)", "B_left(k+1)", "B_left(k+2)"],
    ["R1", "C_bl: A_bot · B_left", "B_right(k)", "A_top(k+1)", "A_top(k+2)"],
    ["X", "barrier：保证上述 LDS 写可读", "—", "—", "—"],
    ["R2", "C_tr: A_top · B_right", "B_left(k+1)", "A_bot(k+1)", "A_bot(k+2)"],
    ["R3", "C_br: A_bot · B_right", "A_top(k+1)", "B_right(k+1)", "B_right(k+2)"],
    ["Y", "barrier：保护下一轮读写", "—", "—", "—"],
], [1.15, 3.5, 2.28, 2.55, 2.72], row_h=.63, font=14)
text(s, "单级 LDS 原地刷新：每个半 tile 先读旧值，跨 barrier 才写新值，再跨 barrier 才读新值。", .7, 6.28, 11.9, .47, 16, TEAL)

# 18 – barrier proof, keeping the implementation's actual slot ordering
s = page("v2：为何每个 K 步只需两个全 CTA barrier？", "每个 slot 检查旧值最后一次读、下一次写、以及新值第一次读", "v2 · barrier")
rows(s, ["LDS slot", "上次读（WAR）", "写入 k+1", "下次读（RAW）", "跨越的 barrier"], [
    ["B_left", "上一 K 步 R2", "R0", "R2", "上一轮 Y / 本轮 X"],
    ["A_top", "上一 K 步 R3", "R1", "R3", "上一轮 Y / 本轮 X"],
    ["A_bot", "R0", "R2", "下轮 R0", "本轮 X / 本轮 Y"],
    ["B_right", "R1", "R3", "下轮 R1", "本轮 X / 本轮 Y"],
], [1.65, 2.45, 1.78, 2.45, 3.87], row_h=.79, font=15)
card(s, "X：R1 与 R2 之间", "旧 A_bot / B_right 已经读完；新的 A_top / B_left 在另一个 wave 中可见。", .68, 5.52, 5.88, 1.33, TEAL, 14)
card(s, "Y：R3 之后", "旧 A_top / B_left 已读完；新的 A_bot / B_right 可供下一轮使用。", 6.74, 5.52, 5.88, 1.33, GOLD, 14)
note(s, "README §3.4 列出每个 slot 的 WAR / RAW；prologue 结束前另需 barrier。v2 的动态可视化见 slides/figs/v2_pipeline.svg（SVG 动画不能可靠嵌入 PPT）。")

# 19 – schedule and register pressure
s = page("v2：调度提示让访存穿插进 MFMA 空档", "使用实际的 rocdl.sched_mfma / sched_dsrd / sched_dswr / sched_vmem", "v2 · 指令级")
card(s, "寄存器预算 / lane", "C 累加：256（AGPR）\n四个半 tile DOT 操作数：128\nGR 暂存：64（VGPR）\n其余供地址与控制", .64, 1.53, 5.76, 4.39, VIOLET, 18)
card(s, "每 K 步的指令", "256 MFMA\n32 ds_read_b128\n16 ds_write_b128\n16 global_load_dwordx4\n2 个全 CTA barrier", 6.58, 1.53, 6.08, 4.39, TEAL, 18)
text(s, "等数据的顺序同样重要：等待最老若干访存，不是指定单条指令；调度顺序决定流水能藏多深。", .81, 6.16, 11.7, .58, 16, FAINT)

# 20 – measured results, native editable bars
s = page("性能：v0 → v1 → v2，收益来自不同层次", "实测记录没有设备型号 / 时钟；基准：BF16 输入，v0/v1/v2 输出 FP32", "结果")
case_list=[("v0 64²", "v0_64x64", BLUE), ("v0 128²", "v0_128x128", BLUE), ("v1", "v1", TEAL),
           ("v2 无调度", "v2_nosched", GOLD), ("v2", "v2", VIOLET), ("torch*", "torch", "73859A")]
for i,(label,key,color) in enumerate(case_list):
    x=.82+i*2.08
    height=3.48*B[key][1]/700
    box(s,x,5.48-height,1.53,height,color,radius=False)
    text(s,f"{B[key][1]:.0f}",x,5.13-height,1.53,.31,18,WHITE,True,PP_ALIGN.CENTER)
    text(s,label,x-.19,5.67,1.93,.44,13,FAINT,align=PP_ALIGN.CENTER)
box(s,.68,6.37,12.0,.38,PANEL2)
text(s,"* torch.matmul 返回 BF16；backend 标签未检测。不能当作 v2 的同输出类型对照。",.83,6.42,11.67,.26,12,FAINT)
note(s,"基准源：slides/data/bench.txt；50 次 HIP-event 计时取中位数。v1/v0 L2 外部读取为 1.24/5.59 GB，v2 1.31 GB：MALL 命中也包含其中，非 HBM 物理读取。")

# 21 – next steps and caveats
s = page("结论与下一步", "这套流程的核心不是硬搬 CUDA 写法，而是从硬件限制反推数据流", "总结")
for i,(label,body,accent) in enumerate([
    ("v0 · 局部复用", "寄存器做大 tile；spill 前停止扩 tile。", BLUE),
    ("v1 · 共享复用", "LDS 让 4 wave 共用 A/B；用 XOR 规避 bank 冲突。", TEAL),
    ("v2 · 藏住延迟", "保留单级 LDS，VGPR 预取 + 提前 LR + 调度约束。", VIOLET),
]):
    card(s,label,body,.7+4.18*i,1.6,3.98,2.52,accent,16)
box(s,.7,4.35,12.0,2.22,PANEL)
text(s,"后续可尝试",.96,4.56,11.1,.42,19,TEAL,True)
bullets(s,["inter_wave 乒乓、XCD 感知的 CTA 排布", "加宽 epilogue、支持 M/N/K 尾块与 buffer 边界处理"],.96,5.08,11.35,16,.55)
note(s,"README §5 列出下一步研究方向；这些尚不是本项目已验证功能。本演示按原文提炼，完整细节和示例以 ../README.md 为准。")


def main(output: Path = OUTPUT):
    output.parent.mkdir(parents=True, exist_ok=True)
    deck.save(output)
    print(f"Saved {len(deck.slides)} slides to {output}")


if __name__ == "__main__":
    main()