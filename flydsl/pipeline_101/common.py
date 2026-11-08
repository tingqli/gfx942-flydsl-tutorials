"""pipeline_101 的公共代码：编译 / 统计 ISA / 计时，以及实验用到的几个 GEMM kernel。"""
import contextlib
import glob
import io
import os
import re
import sys
import tempfile
import time

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx

HERE = os.path.dirname(os.path.abspath(__file__))
GEMM_DIR = os.path.abspath(os.path.join(HERE, "../../kernels/gemm"))
sys.path.insert(0, GEMM_DIR)
from v0 import build_v0  # noqa: E402

# 和 kernels/gemm/common.py 相同的编译选项
HINTS = {"opt_level": 2, "llvm_options": {"amdgpu-mfma-vgpr-form": False}}
SCHED = {                                     # LLVM 机器指令调度器的开关
    "默认": {},
    "关 pre-RA": {"enable-misched": False},
    "关 pre-RA + post-RA": {"enable-misched": False, "enable-post-misched": False},
}


def hints(sched="默认"):
    return {**HINTS, "llvm_options": {**HINTS["llvm_options"], **SCHED[sched]}}


def stream():
    return fx.Stream(torch.cuda.current_stream().cuda_stream)


# ---------------------------------------------------------------- 编译与 ISA

def dumps(launcher, *args, arch="gfx942", sched="默认"):
    """只编译不运行，返回各阶段文件 {文件名: 内容}（21_llvm_ir.ll、22_final_isa.s 等）"""
    d = tempfile.mkdtemp(prefix="pipeline101_")
    env = dict(ARCH=arch, COMPILE_ONLY="1", FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=d)
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            flyc.compile[hints(sched)](launcher, *args)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k)
            else:
                os.environ[k] = v
    return {os.path.basename(p): open(p).read() for p in glob.glob(f"{d}/**/*.*", recursive=True)}


def body(isa_text):
    """kernel 的指令行（去掉注释和伪指令，保留 .LBB 标号）"""
    lines, inside = [], False
    for raw in isa_text.splitlines():
        if re.match(r"^\w+:", raw):
            inside = True
            continue
        line = raw.split(";")[0].strip()
        if inside and line and (not line.startswith(".") or line.startswith(".LBB")):
            lines.append(line)
        if line.startswith("s_endpgm"):
            break
    return lines


def main_loop(lines):
    """第一个有回跳的标号到回跳指令之间，就是主循环"""
    for i, l in enumerate(lines):
        if l.endswith(":"):
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("s_cbranch") and lines[j].endswith(l[:-1]):
                    return lines[i:j + 1]
    return []


def loop_isa(build, A, B, C, arch="gfx942", sched="默认"):
    return main_loop(body(dumps(build(), A, B, C, stream(), arch=arch, sched=sched)["22_final_isa.s"]))


# ---------------------------------------------------------------- 主循环的统计

def reg_set(s):
    m = re.match(r"([av])\[(\d+):(\d+)\]", s) or re.match(r"([av])(\d+)()", s)
    lo = int(m.group(2))
    return {(m.group(1), r) for r in range(lo, int(m.group(3) or lo) + 1)}


def events(loop):
    """把主循环变成事件列表：("L", A/B) 发 load，("W", n) 等到只剩 n 条，("M", a, b) 一条 MFMA"""
    loads, kinds, ev = [], [], []
    for l in loop:
        if l.startswith("global_load"):
            loads.append(reg_set(l.split()[1].rstrip(",")))
            kinds.append("?")
            ev.append(["L", len(loads) - 1])
        elif l.startswith("v_mfma"):
            ops = [x.strip() for x in l.split(None, 1)[1].split(",")]
            ra, rb = reg_set(ops[1]), reg_set(ops[2])
            used = []
            for i, r in enumerate(loads):
                if kinds[i] == "?":
                    kinds[i] = "A" if ra & r else "B" if rb & r else "?"
            # 本条 MFMA 读的是哪两条 load 写的寄存器（取最近写入的那条）
            for rr in (ra, rb):
                idx = [i for i, r in enumerate(loads) if rr & r]
                used.append(idx[-1] if idx else None)
            ev.append(["M", *used])
        else:
            m = re.match(r"s_waitcnt vmcnt\((\d+)\)", l)
            if m:
                ev.append(["W", int(m.group(1))])
    for e in ev:
        if e[0] == "L":
            e.append(kinds[e[1]])
    return ev


def load_profile(loop):
    """load 发出顺序、下沉条数（第一条 MFMA 之后才发的 load）、每次等待时被等的 load 已经过的 MFMA 数、
    串行等待次数（等的是一条被下沉的 load，而且它发出后最多只经过了 2 条 MFMA）"""
    ev = events(loop)
    order, issued_at, dist, serial, n_load, n_mfma = "", [], [], 0, 0, 0
    for e in ev:
        if e[0] == "L":
            order += e[2]
            issued_at.append(n_mfma)
            n_load += 1
        elif e[0] == "M":
            n_mfma += 1
        elif n_mfma > 0:                      # 第一条 MFMA 之前的等待不可避免，不统计
            k = n_load - e[1] - 1             # vmcnt(c)：等到只剩最近 c 条，即等第 k 条
            if k >= 0:
                dist.append(n_mfma - issued_at[k])
                serial += issued_at[k] > 0 and dist[-1] <= 2
    return {"load 顺序": order, "下沉": sum(x > 0 for x in issued_at), "被等 load 已经过的 MFMA": dist,
            "串行等待": serial}


def tag(name):
    """结果文件名（只用 ASCII）"""
    for zh, en in [("关 pre-RA + post-RA", "no_sched"), ("关 pre-RA", "no_preRA"), ("默认", "default")]:
        name = name.replace(zh, en)
    return name.replace(" + ", "__").replace(" ", "_")


# ---------------------------------------------------------------- 计时

def measure(build, A, B, C, ref, sched="默认", iters=20):
    C.zero_()
    f = flyc.compile[hints(sched)](build(), A, B, C, stream())   # 编译并运行一次
    torch.cuda.synchronize()
    ok = bool(torch.allclose(C, ref, atol=1e-3, rtol=1e-3))
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        f(A, B, C, stream())
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3)
    us = sorted(ts)[iters // 2]
    M, K = A.shape
    return {"正确": ok, "us": round(us, 1), "TFLOPS": round(2 * M * B.shape[0] * K / us / 1e6, 1)}


# ---------------------------------------------------------------- kernel

def build_tiled(BM, BN, M, N, K, order=None, b_first=False, fence=False):
    """1 个 wave 算 BM x BN，直接从 global 读 A/B（和 tiled_vs_manual.ipynb 第 2 节相同）。
    order：fx.gemm 的遍历顺序；b_first：先发 B 的 load；fence：copy 和 gemm 之间加 sched_barrier(0)"""
    BK = 32

    @flyc.kernel(known_block_size=(64, 1, 1))
    def gemm_tiled(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        bm, bn = fx.block_idx.x, fx.block_idx.y
        gA = fx.flat_divide(fx.make_view(fx.get_iter(A), fx.make_layout((M, K), (K, 1))), (BM, BK))[None, None, bm, None]
        gB = fx.flat_divide(fx.make_view(fx.get_iter(B), fx.make_layout((N, K), (K, 1))), (BN, BK))[None, None, bn, None]
        gC = fx.flat_divide(fx.make_view(fx.get_iter(C), fx.make_layout((M, N), (N, 1))), (BM, BN))[None, None, bm, bn]
        atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
        tiled_mma = fx.make_tiled_mma(atom, fx.make_layout((1, 1, 1), (0, 0, 0)),
                                      fx.make_tile(None, None, fx.make_layout((4, 4, 2), (1, 8, 4))))
        thr_mma = tiled_mma.thr_slice(tid)
        cp = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
        cpC = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        tcA = fx.make_tiled_copy_A(cp, tiled_mma).get_slice(tid)
        tcB = fx.make_tiled_copy_B(cp, tiled_mma).get_slice(tid)
        tcC = fx.make_tiled_copy_C(cpC, tiled_mma).get_slice(tid)
        tAg, tBg = tcA.partition_S(gA), tcB.partition_S(gB)
        fA, fB = thr_mma.make_fragment_A(gA[None, None, 0]), thr_mma.make_fragment_B(gB[None, None, 0])
        fC = thr_mma.make_fragment_C(gC)
        fC.fill(0)
        rA, rB = tcA.retile(fA), tcB.retile(fB)
        for kt in range(fx.Int32(0), fx.Int32(K // BK), fx.Int32(1)):
            if fx.const_expr(b_first):
                fx.copy(cp, tBg[None, None, None, kt], rB)
            fx.copy(cp, tAg[None, None, None, kt], rA)
            if fx.const_expr(not b_first):
                fx.copy(cp, tBg[None, None, None, kt], rB)
            if fx.const_expr(fence):
                fx.rocdl.sched_barrier(0)          # 调度器不能把任何指令移过这里
            fx.gemm(atom, fC, fA, fB, fC, traversal_order=order)
        fx.copy(cpC, tcC.retile(fC), tcC.partition_S(gC))

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream):
        gemm_tiled(A, B, C).launch(grid=(M // BM, N // BN, 1), block=(64, 1, 1), stream=stream)

    return launch


# v0 的两个变体用 build_v0 的编译期参数生成：SLICE=True 先切动态坐标，INNER="M" 把 M 方向的循环放到内层


def variants(M, N, K, BM=128):
    """实验用的 5 个 kernel：名字 -> 构造 launcher 的函数"""
    O = fx.GemmTraversalOrder
    return {
        "v0": lambda: build_v0(BM, BM, 1, M, N, K),
        "v0_slice": lambda: build_v0(BM, BM, 1, M, N, K, SLICE=True),
        "v0_slice_ji": lambda: build_v0(BM, BM, 1, M, N, K, SLICE=True, INNER="M"),
        "tiled 默认": lambda: build_tiled(BM, BM, M, N, K),
        "tiled MNK": lambda: build_tiled(BM, BM, M, N, K, order=O.MNK),
    }


# ---------------------------------------------------------------- LLVM IR 里的 MFMA 顺序

MFMA_CALL = re.compile(r"(%\S+) = (?:tail )?call <4 x float> @llvm\.amdgcn\.mfma\.f32\.16x16x16bf16\.1k\("
                       r"(?:<4 x i16> )?(%\S+), (?:<4 x i16> )?(%\S+), (?:<4 x float> )?(%\S+),")


def mfma_call_order(ll):
    """LLVM IR（调度之前）里 MFMA 调用的顺序：A0·B1 = 第 0 个 A 操作数 x 第 1 个 B 操作数，+ = 累加在前面某条的结果上"""
    ids, results, seq = {}, set(), []
    name = lambda kind, v: ids.setdefault((kind, v), f"{kind}{sum(k == kind for k, _ in ids)}")
    for res, a, b, acc in MFMA_CALL.findall(ll):
        seq.append(f"{name('A', a)}·{name('B', b)}" + ("+" if acc in results else ""))
        results.add(res)
    second = seq[1]
    inner = "K" if second.endswith("+") else "M" if seq[0].split("·")[1] == second.split("·")[1].rstrip("+") else "N"
    return seq, inner


def width(s):
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(s))


def table(rows, file=None):
    """rows: {名字: {列: 值}}，打印成对齐的表格（中文按两个字符宽）"""
    cols = list(next(iter(rows.values())))
    first = max(width(n) for n in rows)
    ws = [max(width(c), *(width(r[c]) for r in rows.values())) for c in cols]
    pad = lambda s, w: " " * (w - width(s)) + str(s)
    out = [" " * first + " | " + " | ".join(pad(c, w) for c, w in zip(cols, ws))]
    out += [str(n) + " " * (first - width(n)) + " | " + " | ".join(pad(r[c], w) for c, w in zip(cols, ws))
            for n, r in rows.items()]
    print("\n".join(out), file=file)
    return "\n".join(out)
