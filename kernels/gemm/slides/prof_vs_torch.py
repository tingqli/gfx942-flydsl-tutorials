"""在 gfx950（MI350X）上用 rocprofv3 对比 v0_preshuffle / v2 和 torch.matmul（hipBLASLt）。

    python3 slides/prof_vs_torch.py              # 采集并汇总，写 slides/data/gfx950_vs_torch.txt
    python3 slides/prof_vs_torch.py run CASE     # 只跑一个 case（给 rocprofv3 调用）

每个 case 跑 WARMUP + ITERS 次，每次用一份新拷贝的输入（和 bench.py 一样，避免 L2/MALL 里留着上一次的数据）。
计数器按组分别采集（一组一次运行），每个值取各次 dispatch 的平均。
"""
import collections
import csv
import glob
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GEMM = os.path.dirname(HERE)
M = N = K = 8192
WARMUP, ITERS = 40, 10      # 连续跑约 30 次后时间才稳定（频率爬升），热身要足够长
CASES = ["torch", "v0_preshuffle", "v0_preshuffle_k32", "v2", "v2_k32"]
GROUPS = [
    "SQ_INSTS_MFMA SQ_INSTS_VALU_MFMA_MOPS_BF16 SQ_VALU_MFMA_BUSY_CYCLES GRBM_GUI_ACTIVE",
    "SQ_WAVES SQ_WAVE_CYCLES SQ_BUSY_CYCLES SQ_WAIT_INST_ANY",
    "SQ_INSTS_VMEM_RD SQ_INSTS_LDS SQ_LDS_BANK_CONFLICT SQ_INSTS_VALU",
    "TCC_HIT_sum TCC_MISS_sum TCC_REQ_sum",
    "TCC_EA0_RDREQ_sum TCC_EA0_RDREQ_32B_sum TCC_EA0_RDREQ_64B_sum",
    "TA_BUSY_avr TCP_TOTAL_CACHE_ACCESSES_sum TCP_TCC_READ_REQ_sum",
    "SQ_INSTS_LDS_LOAD SQ_INSTS_LDS_STORE TA_BUFFER_READ_LDS_WAVEFRONTS_sum TA_FLAT_READ_LDS_WAVEFRONTS_sum",
]


# ---------------------------------------------------------------- worker

def run(case):
    sys.path.insert(0, GEMM)
    import torch
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from v0_preshuffle import build_v0_preshuffle, preshuffle
    from v2 import build_v2

    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / math.sqrt(K)
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / math.sqrt(K)
    if case == "torch":
        fn, args = (lambda a, b: torch.matmul(a, b.T)), (A, B)
    else:
        hints = {"opt_level": 2, "llvm_options": {"amdgpu-mfma-vgpr-form": False}}
        st = fx.Stream(torch.cuda.current_stream().cuda_stream)
        C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
        if case.startswith("v2"):
            launcher, a, b = build_v2(256, 256, 1, M, N, K, K32=case.endswith("k32")), A, B
        else:
            launcher = build_v0_preshuffle(128, 128, 1, M, N, K, K32=case.endswith("k32"))
            a, b = preshuffle(A), preshuffle(B)
        f = flyc.compile[hints](launcher, a, b, C, st)
        fn, args = (lambda a, b, c: f(a, b, c, st)), (a, b, C)
    reps = [[x.clone() for x in args] for _ in range(WARMUP + ITERS)]
    torch.cuda.synchronize()
    for r in reps:
        fn(*r)
        torch.cuda.synchronize()


# ---------------------------------------------------------------- driver

ROCPROF = shutil.which("rocprofv3") or "/opt/rocm/bin/rocprofv3"
GEMM_KERNEL = re.compile(r"^(gemm_v|Custom_Cijk|Cijk)")      # 我们的 kernel / hipBLASLt 的 kernel


def rocprof(case, extra):
    """返回 {"kernel_trace": rows, "counter_collection": rows}（只保留 GEMM kernel）"""
    out = {}
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([ROCPROF, *extra, "-d", d, "-o", "out", "-f", "csv", "--",
                        sys.executable, os.path.abspath(__file__), "run", case],
                       check=True, capture_output=True, cwd=d)
        for name in ["kernel_trace", "counter_collection"]:
            paths = glob.glob(f"{d}/**/*{name}.csv", recursive=True)
            if paths:
                out[name] = [r for r in csv.DictReader(open(paths[0])) if GEMM_KERNEL.match(r["Kernel_Name"])]
    return out


def median_us(rows):
    us = sorted((int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3 for r in rows[WARMUP:])
    return us[len(us) // 2]


def trace(case):
    rows = rocprof(case, ["--kernel-trace"])["kernel_trace"]
    r = rows[0]
    return {"kernel": r["Kernel_Name"], "us": median_us(rows),
            "LDS": int(r["LDS_Block_Size"]), "scratch": int(r["Scratch_Size"]),
            "block": int(r["Workgroup_Size_X"]) * int(r["Workgroup_Size_Y"]) * int(r["Workgroup_Size_Z"]),
            "grid": int(r["Grid_Size_X"]) * int(r["Grid_Size_Y"]) * int(r["Grid_Size_Z"])}


def counters(case):
    vals = {}
    for i, g in enumerate(GROUPS):
        res = rocprof(case, (["--kernel-trace"] if i == 0 else []) + ["--pmc", *g.split()])
        if i == 0:                                  # 和 GRBM_GUI_ACTIVE 同一次运行的时间，用来算主频
            vals["us_pmc"] = median_us(res["kernel_trace"])
        per = collections.defaultdict(dict)
        for r in res["counter_collection"]:
            per[r["Dispatch_Id"]][r["Counter_Name"]] = per[r["Dispatch_Id"]].get(r["Counter_Name"], 0) + float(r["Counter_Value"])
        disp = list(per.values())[WARMUP:]
        for name in g.split():
            vals[name] = sum(d[name] for d in disp) / len(disp)
    return vals


def summarize(t, c, cu=256, xcd=8):
    flop = 2 * M * N * K
    cycles = c["GRBM_GUI_ACTIVE"] / xcd                     # GRBM_GUI_ACTIVE 是各 XCD 之和
    ea = c["TCC_EA0_RDREQ_32B_sum"] * 32 + c["TCC_EA0_RDREQ_64B_sum"] * 64 + \
        (c["TCC_EA0_RDREQ_sum"] - c["TCC_EA0_RDREQ_32B_sum"] - c["TCC_EA0_RDREQ_64B_sum"]) * 128
    return {
        "时间 (us)": round(t["us"], 1),
        "TFLOPS": round(flop / t["us"] / 1e6, 1),
        "block x 线程": f'{t["grid"] // t["block"]} x {t["block"]}',
        "LDS (KB)": round(t["LDS"] / 1024, 1),
        "每条 MFMA 的 FLOP": round(flop / c["SQ_INSTS_MFMA"] / 64) * 64,
        "FLOP / MFMA 忙碌周期": round(flop / c["SQ_VALU_MFMA_BUSY_CYCLES"]),
        "MFMA 忙碌": f'{c["SQ_VALU_MFMA_BUSY_CYCLES"] / (cycles * cu * 4) * 100:.1f}%',
        "计数时主频 (MHz)": round(cycles / c["us_pmc"]),
        "wave 等待指令": f'{c["SQ_WAIT_INST_ANY"] / c["SQ_WAVE_CYCLES"] * 100:.1f}%',
        "global/buffer load 条数 (M)": round(c["SQ_INSTS_VMEM_RD"] / 1e6, 2),
        "LDS 指令 (M)": round(c["SQ_INSTS_LDS"] / 1e6, 2),
        "  其中读 / 写 (M)": f'{c["SQ_INSTS_LDS_LOAD"] / 1e6:.2f} / {c["SQ_INSTS_LDS_STORE"] / 1e6:.2f}',
        "load 直写 LDS 的 wave (M)": round((c["TA_BUFFER_READ_LDS_WAVEFRONTS_sum"] +
                                       c["TA_FLAT_READ_LDS_WAVEFRONTS_sum"]) / 1e6, 2),
        "LDS bank 冲突 (M 周期)": round(c["SQ_LDS_BANK_CONFLICT"] / 1e6, 2),
        "L1 访问 (M)": round(c["TCP_TOTAL_CACHE_ACCESSES_sum"] / 1e6, 1),
        "L1→L2 读请求 (M)": round(c["TCP_TCC_READ_REQ_sum"] / 1e6, 1),
        "TA 忙碌": f'{c["TA_BUSY_avr"] / cycles * 100:.1f}%',
        "L2 命中": f'{c["TCC_HIT_sum"] / max(1, c["TCC_HIT_sum"] + c["TCC_MISS_sum"]) * 100:.1f}%',
        "L2 之外读取 (GB)": round(ea / 1e9, 2),
    }


def main():
    out = {}
    for case in CASES:
        print("采集", case, flush=True)
        t = trace(case)
        c = counters(case)
        out[case] = (t, c)
    lines = [f"# gfx950 MI350X，GEMM {M}x{N}x{K} bf16（v0_preshuffle / v2 输出 fp32，torch 输出 bf16），"
             f"每个 case {ITERS} 次 dispatch 的中位数（时间）/ 平均（计数器）"]
    for case, (t, c) in out.items():
        lines.append(f"{case}: {t['kernel']}")
    keys = list(summarize(*next(iter(out.values()))).keys())
    rows = {case: summarize(t, c) for case, (t, c) in out.items()}
    w = max(len(k) for k in keys) + 4
    lines.append("")
    lines.append(f"{'':{w}}" + "".join(f"{case:>22}" for case in rows))
    for k in keys:
        lines.append(f"{k:{w}}" + "".join(f"{str(rows[case][k]):>22}" for case in rows))
    lines.append("")
    lines.append("# 原始计数器（每次 dispatch 平均）")
    for case, (t, c) in out.items():
        lines.append(case + " " + " ".join(f"{k}={v:.6g}" for k, v in c.items()))
    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with open(os.path.join(HERE, "data", "gfx950_vs_torch.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        main()
