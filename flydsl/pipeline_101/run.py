"""跑 pipeline_101 的全部实验，结果写到 results/：

    results/report.txt   各实验的表格（README 里的数字都来自这里）
    results/results.json 同样的数据，供 make_figs.py 画图
    results/isa/*.s      各版本、各调度设置下的主循环 ISA

用法：python run.py        （本机 GPU 上计时；ISA 统计按 gfx942 编译，并检查 gfx950 是否相同）
"""
import json
import os
import sys

from common import *  # noqa: F401,F403
from common import HERE, SCHED, build_tiled, dumps, load_profile, loop_isa, measure, mfma_call_order, stream, table, tag, variants

OUT = os.path.join(HERE, "results")
os.makedirs(os.path.join(OUT, "isa"), exist_ok=True)
report = open(os.path.join(OUT, "report.txt"), "w")
data = {}


def section(title):
    line = f"\n==== {title}"
    print(line)
    print(line, file=report)


def show(rows):
    table(rows)
    table(rows, file=report)
    report.flush()


def short(d):
    """被等 load 已经过的 MFMA 数：列表太长时只显示前 8 个"""
    return str(d[:8])[:-1] + (", …]" if len(d) > 8 else "]")


M = N = K = 4096
torch.manual_seed(0)
A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / 64
B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / 64
C = torch.zeros(M, N, device="cuda")
ref = A.float() @ B.float().T
O = fx.GemmTraversalOrder
local = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
print(f"本机 GPU {local}；GEMM {M}x{N}x{K} bf16 -> fp32，BM = BN = 128，1 个 wave / block", file=report)

# ---- 1. 13 种遍历顺序
section("1. fx.gemm 的 13 种遍历顺序（tiled，先发 A 后发 B）")
rows = {}
for order in [None] + list(O):
    build = lambda: build_tiled(128, 128, M, N, K, order=order)
    p = load_profile(loop_isa(build, A, B, C))
    rows[str(order)] = {**measure(build, A, B, C, ref), "下沉": p["下沉"], "串行等待": p["串行等待"],
                        "被等 load 已经过的 MFMA": short(p["被等 load 已经过的 MFMA"])}
show(rows)
data["orders"] = rows

# ---- 2. 名字的含义：LLVM IR 里 MFMA 调用的顺序（调度之前）
section("2. 遍历顺序名字的含义（BM = BN = 32，LLVM IR 里前 8 条 MFMA）")
rows = {}
for order in [None, O.MNK, O.NMK, O.KMN, O.MKN, O.KNM]:
    ll = dumps(build_tiled(32, 32, M, N, K, order=order), A, B, C, stream())["21_llvm_ir.ll"]
    seq, inner = mfma_call_order(ll)
    rows[str(order)] = {"最内层": inner, "前 8 条 MFMA": " ".join(seq[:8])}
show(rows)
data["names"] = rows

# ---- 3. 五个版本的主循环
section("3. 五个版本（默认调度）：load 顺序、下沉条数、被等 load 已经过的 MFMA 数")
rows = {}
for name, build in variants(M, N, K).items():
    p942 = load_profile(loop_isa(build, A, B, C, arch="gfx942"))
    p950 = load_profile(loop_isa(build, A, B, C, arch="gfx950"))
    rows[name] = {**measure(build, A, B, C, ref), "load 顺序": p942["load 顺序"], "下沉": p942["下沉"], "串行等待": p942["串行等待"],
                  "被等 load 已经过的 MFMA": short(p942["被等 load 已经过的 MFMA"]), "gfx950 同": p942 == p950}
show(rows)
data["variants"] = rows

# ---- 4. 对调 A、B 的 copy 顺序
section("4. 对调 copy 顺序（TFLOPS / 串行等待）")
rows = {}
for order in [None, O.MNK, O.NMK, O.KMN, O.KNM]:
    rows[str(order)] = {}
    for first in "AB":
        build = lambda: build_tiled(128, 128, M, N, K, order=order, b_first=first == "B")
        p = load_profile(loop_isa(build, A, B, C))
        rows[str(order)][f"先发 {first}"] = f"{measure(build, A, B, C, ref)['TFLOPS']} / {p['串行等待']}"
show(rows)
data["swap"] = rows

# ---- 5. 关掉调度器
section("5. 关掉 LLVM 的机器指令调度器（TFLOPS / 下沉 / 串行等待）")
rows = {}
for name, build in variants(M, N, K).items():
    rows[name] = {}
    for sched in SCHED:
        loop = loop_isa(build, A, B, C, sched=sched)
        with open(os.path.join(OUT, "isa", f"{tag(name)}__{tag(sched)}.s"), "w") as f:
            f.write("\n".join(loop) + "\n")
        p = load_profile(loop)
        rows[name][sched] = f"{measure(build, A, B, C, ref, sched=sched)['TFLOPS']} / {p['下沉']} / {p['串行等待']}"
show(rows)
data["sched"] = rows

# ---- 6. 在源码里阻止下沉：sched_barrier
section("6. copy 和 gemm 之间加 rocdl.sched_barrier(0)（默认调度）")
rows = {}
for name, build in [("tiled 默认", lambda: build_tiled(128, 128, M, N, K)),
                    ("tiled 默认 + sched_barrier", lambda: build_tiled(128, 128, M, N, K, fence=True)),
                    ("tiled MNK", lambda: build_tiled(128, 128, M, N, K, order=O.MNK)),
                    ("tiled MNK + sched_barrier", lambda: build_tiled(128, 128, M, N, K, order=O.MNK, fence=True))]:
    loop = loop_isa(build, A, B, C)
    p = load_profile(loop)
    rows[name] = {**measure(build, A, B, C, ref), "下沉": p["下沉"], "串行等待": p["串行等待"],
                  "被等 load 已经过的 MFMA": short(p["被等 load 已经过的 MFMA"])}
    with open(os.path.join(OUT, "isa", tag(name) + ".s"), "w") as f:
        f.write("\n".join(loop) + "\n")
show(rows)
data["fence"] = rows

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(data, f, ensure_ascii=False, indent=1)
report.close()
print("\n写入", OUT)
