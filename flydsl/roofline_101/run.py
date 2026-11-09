"""roofline_101：每个 CU 每个周期要多少字节、能供多少字节。结果写到 results/：

    results/report.txt    表格（README 里的数字都来自这里）
    results/results.json  同样的数据

用法：python run.py                 采集全部 case（本机 gfx950，需要 rocprofv3）
      python run.py run CASE        只跑一个 case（给 rocprofv3 调用）
"""
import collections
import csv
import glob
import json
import math
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GEMM = os.path.join(HERE, "..", "..", "kernels", "gemm")
M = N = K = 8192
WARMUP, ITERS = 40, 10       # 连续跑约 30 次后频率才稳定，热身要足够长
CU, XCD = 256, 8

# case -> 说明。k16 = v_mfma_f32_16x16x16_bf16（每 SIMD 每周期 512 FLOP，与 MI300X 相同）
#              k32 = v_mfma_f32_16x16x32_bf16（1024 FLOP，gfx950 新增）
CASES = {
    "v0p_k16": "v0_preshuffle，16x16x16（MI300X 的算力/CU/周期）",
    "v0p_k32": "v0_preshuffle，16x16x32（MI350X 满速）",
    "v2_k16": "v2，16x16x16（MI300X 的算力/CU/周期）",
    "v2_k32": "v2，16x16x32（MI350X 满速）",
    "load_v0": "只 load：与 v0_preshuffle 相同的 global load，去掉 MFMA",
    "load_hit": "只 load：同样的指令，但所有 wave 反复读同一块 16 KB（命中 L1）",
    "lds_read": "只读 LDS：ds_read_b128，无 bank conflict",
}
GROUPS = [
    "GRBM_GUI_ACTIVE SQ_VALU_MFMA_BUSY_CYCLES TA_TA_BUSY",
    "SQ_INSTS_VMEM_RD SQ_INSTS_LDS TA_ADDR_STALLED_BY_TC_CYCLES TA_DATA_STALLED_BY_TC_CYCLES",
]
KERNELS = ("gemm_", "load_only", "lds_read")


# ---------------------------------------------------------------- 微基准 kernel

def build_load_only(M, N, K, HIT=False):
    """v0_preshuffle 的 load 部分：每个 wave 每个 K 步读 8 个 A + 8 个 B 向量（各 1 KB 连续），
    下一个 K 步的 load 在用当前数据之前发出（同 ILP）。HIT=True 时所有 wave 只读 bid 0、K 步 0
    的 16 KB，每步换一个顺序，地址随循环变，编译器不能把 load 提到循环外。"""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import const_expr, range_constexpr, rocdl

    NK, NI = K // 32, 8
    grid, block = (M // 128, N // 128, 1), (64, 1, 1)

    @flyc.kernel(known_block_size=block)
    def load_only(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        bid_m, bid_n = fx.block_idx.x, fx.block_idx.y
        A = fx.make_view(fx.get_iter(A), fx.make_ordered_layout((8, 64, NK, NI, grid[0]), 0))
        B = fx.make_view(fx.get_iter(B), fx.make_ordered_layout((8, 64, NK, NI, grid[1]), 0))
        C = fx.make_view(fx.get_iter(C), fx.make_ordered_layout((4, 64, grid[0], grid[1]), 0))
        if const_expr(HIT):
            A, B = A[None, None, None, None, 0], B[None, None, None, None, 0]
        else:
            A, B = A[None, None, None, None, bid_m], B[None, None, None, None, bid_n]

        def gr(k):
            if const_expr(HIT):
                return [X[None, tid, 0, (i + k * 5) % NI].load() for X in (A, B) for i in range_constexpr(NI)]
            return [X[None, tid, k, i].load() for X in (A, B) for i in range_constexpr(NI)]

        def use(acc, vs):
            for v in vs:
                acc = acc + v.bitcast(fx.Float32)
            return acc

        # 和 v0_preshuffle 的 ILP 一样：一次迭代两个 K 步，两组寄存器轮换，循环末尾不用拷贝。
        # sched_barrier 让 load 留在原位，否则 LLVM 会把它们挪到使用处附近。
        acc0 = [fx.full(4, 0.0, fx.Float32), fx.full(4, 0.0, fx.Float32)]
        for k, st in range(fx.Index(0), fx.Index(NK // 2), fx.Index(1), init=acc0 + gr(fx.Int32(0))):
            st = list(st)
            kt = fx.Int32(k) * 2
            nxt = gr(kt + 1)
            rocdl.sched_barrier(0)
            acc = use(st[0], st[2:])
            cur = gr((kt + 2) % NK)
            rocdl.sched_barrier(0)
            acc = use(acc, nxt)
            results = yield [acc, st[1]] + cur
        C[None, tid, bid_m, bid_n].store(list(results)[0])

    @flyc.jit
    def launcher(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream):
        load_only(A, B, C).launch(grid=grid, block=block, stream=stream)

    return launcher


def build_lds_read(NBLK, NITER):
    """每个 block 4 个 wave、16 KB LDS（每个 CU 能放多个 block，有足够的 wave 轮流发）；
    每次迭代每个 lane 读 8 个 16 B，一个 wave 的 64 个 lane 读连续 1 KB（无 bank conflict）。起始行随迭代变化。
    每次迭代读的是同一组地址，LLVM 展开后合并了一部分重复读（每次迭代实际约 5 条），
    带宽按 SQ_INSTS_LDS 实际计到的条数算。"""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import range_constexpr

    ROWS = 1024                      # 1024 x 16 B = 16 KB
    block = (256, 1, 1)

    @fx.struct
    class SharedStorage:
        a: fx.Array[fx.BFloat16, ROWS * 8, 16]

    @flyc.kernel(known_block_size=block)
    def lds_read(C: fx.Tensor):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        C = fx.make_view(fx.get_iter(C), fx.make_ordered_layout((4, 256, NBLK), 0))
        L = fx.SharedAllocator().allocate(SharedStorage).peek().a.view(fx.make_ordered_layout((8, ROWS), 0))
        acc0 = [fx.full(4, 0.0, fx.Float32), fx.full(4, 0.0, fx.Float32)]
        for k, st in range(fx.Index(0), fx.Index(NITER), fx.Index(1), init=acc0):
            acc = st[0]
            base = ((fx.Int32(k) * 37) & 7) * 128
            for j in range_constexpr(8):
                acc = acc + L[None, (base + j * 128 + fx.Int32(tid)) % ROWS].load().bitcast(fx.Float32)
            results = yield [acc, st[1]]
        C[None, tid, bid].store(list(results)[0])

    @flyc.jit
    def launcher(C: fx.Tensor, stream: fx.Stream):
        lds_read(C).launch(grid=(NBLK, 1, 1), block=block, stream=stream)

    return launcher


LDS_BLK, LDS_ITER = CU * 8, 4096


# ---------------------------------------------------------------- worker

def run(case):
    sys.path.insert(0, GEMM)
    import torch
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from v0_preshuffle import build_v0_preshuffle, preshuffle
    from v2 import build_v2

    hints = {"opt_level": 2, "llvm_options": {"amdgpu-mfma-vgpr-form": False}}
    st = fx.Stream(torch.cuda.current_stream().cuda_stream)
    torch.manual_seed(0)
    if case == "lds_read":
        C = torch.zeros(4 * 256 * LDS_BLK, device="cuda")
        f = flyc.compile[hints](build_lds_read(LDS_BLK, LDS_ITER), C, st)
        reps = [() for _ in range(WARMUP + ITERS)]
        fn = lambda: f(C, st)
    else:
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / math.sqrt(K)
        B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / math.sqrt(K)
        C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
        k32 = case.endswith("k32")
        if case.startswith("v2"):
            launcher = build_v2(256, 256, 1, M, N, K, K32=k32)
        else:
            A, B = preshuffle(A), preshuffle(B)
            if case.startswith("v0p"):
                launcher = build_v0_preshuffle(128, 128, 1, M, N, K, K32=k32)
            else:
                launcher = build_load_only(M, N, K, HIT=case == "load_hit")
        f = flyc.compile[hints](launcher, A, B, C, st)
        reps = [(A.clone(), B.clone()) for _ in range(WARMUP + ITERS)]
        fn = lambda a, b: f(a, b, C, st)
    torch.cuda.synchronize()
    for r in reps:
        fn(*r)
        torch.cuda.synchronize()


# ---------------------------------------------------------------- driver

def rocprof(case, args):
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["rocprofv3", *args, "-d", d, "-o", "out", "-f", "csv", "--",
                        sys.executable, os.path.abspath(__file__), "run", case],
                       check=True, capture_output=True, cwd=d)
        out = {}
        for name in ["kernel_trace", "counter_collection"]:
            paths = glob.glob(f"{d}/**/*{name}.csv", recursive=True)
            if paths:
                out[name] = [r for r in csv.DictReader(open(paths[0])) if r["Kernel_Name"].startswith(KERNELS)]
        return out


def counters(case):
    vals = {}
    for i, g in enumerate(GROUPS):
        res = rocprof(case, (["--kernel-trace"] if i == 0 else []) + ["--pmc", *g.split()])
        if i == 0:
            us = sorted((int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3 for r in res["kernel_trace"][WARMUP:])
            vals["us"] = us[len(us) // 2]
        per = collections.defaultdict(lambda: collections.defaultdict(float))
        for r in res["counter_collection"]:
            per[r["Dispatch_Id"]][r["Counter_Name"]] += float(r["Counter_Value"])
        disp = list(per.values())[WARMUP:]
        for name in g.split():
            vals[name] = sum(d[name] for d in disp) / len(disp)
    return vals


def summarize(case, c):
    cyc = c["GRBM_GUI_ACTIVE"] / XCD                    # GRBM_GUI_ACTIVE 是 8 个 XCD 之和
    per_cu = lambda x: x / CU / cyc
    row = {
        "时间 (us)": round(c["us"], 1),
        "主频 (MHz)": round(cyc / c["us"]),
        "周期 (M)": round(cyc / 1e6, 3),
    }
    if case[:2] in ("v0", "v2"):
        row["TFLOPS"] = round(2 * M * N * K / c["us"] / 1e6, 1)
        row["MFMA 忙碌"] = f'{per_cu(c["SQ_VALU_MFMA_BUSY_CYCLES"]) / 4:.1%}'
    row["TA 忙碌"] = f'{per_cu(c["TA_TA_BUSY"]):.1%}'
    row["TA 被 L1 反压"] = f'{per_cu(c["TA_ADDR_STALLED_BY_TC_CYCLES"] + c["TA_DATA_STALLED_BY_TC_CYCLES"]):.1%}'
    row["global load B/clk/CU"] = round(per_cu(c["SQ_INSTS_VMEM_RD"] * 1024), 1)    # 每条 1 KB
    row["LDS B/clk/CU"] = round(per_cu(c["SQ_INSTS_LDS"] * 1024), 1)                # 每条 ds_*_b128 1 KB
    return row


def demand():
    """MFMA 100% 忙时，每个 CU 每周期要搬多少字节（由 tile 形状算出，与硬件无关）"""
    flop_v0 = 2 * 128 * 128 * 32                       # 一个 wave 一个 K 步（K=32）
    bytes_v0 = 16 * 1024                               # 8 个 A + 8 个 B 向量
    flop_v2 = 2 * 256 * 256 * 64                       # 一个 CTA 一个 K 步（K=64）
    gl_v2 = (256 + 256) * 64 * 2                       # global：A、B 各 256 行 x 64 K
    lds_v2 = gl_v2 + 4 * 4 * 8 * 1024                  # LDS：写一遍 + 4 个 wave x 4 个 region x 8 条 1 KB 读
    rows = {}
    for name, fpc in [("k16", 4 * 512), ("k32", 4 * 1024)]:
        rows[f"v0p_{name}"] = {"FLOP/clk/CU": fpc, "global B/clk/CU": fpc * bytes_v0 / flop_v0, "LDS B/clk/CU": 0}
        rows[f"v2_{name}"] = {"FLOP/clk/CU": fpc, "global B/clk/CU": fpc * gl_v2 / flop_v2,
                             "LDS B/clk/CU": fpc * lds_v2 / flop_v2}
    return rows


def table(rows, file):
    keys = list(dict.fromkeys(k for r in rows.values() for k in r))
    w0 = max(len(k) for k in rows) + 2
    print(f"{'':{w0}}" + "".join(f"{k:>24}" for k in keys), file=file)
    for name, r in rows.items():
        print(f"{name:{w0}}" + "".join(f"{str(r.get(k, '')):>24}" for k in keys), file=file)


def main():
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    raw, rows = {}, {}
    for case in CASES:
        print("采集", case, flush=True)
        raw[case] = counters(case)
        rows[case] = summarize(case, raw[case])
    need = demand()
    with open(os.path.join(HERE, "results", "report.txt"), "w") as f:
        for out in (sys.stdout, f):
            print(f"# gfx950 MI350X，{CU} CU；GEMM {M}x{N}x{K} bf16 -> fp32；热身 {WARMUP} 次后取 {ITERS} 次 dispatch", file=out)
            for case, text in CASES.items():
                print(f"#   {case}: {text}", file=out)
            print("\n== 1. 需求：MFMA 满速时每个 CU 每周期要搬的字节（由 tile 形状算出）", file=out)
            table(need, out)
            print("\n== 2. 实测", file=out)
            table(rows, out)
    with open(os.path.join(HERE, "results", "results.json"), "w") as f:
        json.dump({"demand": need, "measured": rows, "raw": raw}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        main()
