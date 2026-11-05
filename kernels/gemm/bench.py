"""Benchmark v0 / v1 / v2 (and torch.matmul = hipBLASLt) on the same shape.

    python3 bench.py                 # all
    python3 bench.py v2 v2_nosched   # selected
Timing: median of N back-to-back launches measured with HIP events (after warmup).
"""
import math
import sys

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx

from v0 import build_v0
from v0_preshuffle import build_v0_preshuffle, preshuffle
from v1 import build_v1
from v2 import build_v2

props = torch.cuda.get_device_properties(0) 
sm_count = props.multi_processor_count

if sm_count == 304:
    M, N, K = 16 * 2 * 256, 19 * 2 * 256, 8192
else:
    assert sm_count == 80
    M, N, K = 8 * 2 * 256, 10 * 2 * 256, 8192

HINTS = {"opt_level": 2, "llvm_options": {"amdgpu-mfma-vgpr-form": False}}

CASES = {
    "v0_64x64": lambda: build_v0(64, 64, 1, M, N, K),
    "v0_128x128": lambda: build_v0(128, 128, 1, M, N, K),
    "v0_ilp": lambda: build_v0_preshuffle(128, 128, 1, M, N, K, PRESHUFFLE=False),
    "v0_preshuffle_noilp": lambda: build_v0_preshuffle(128, 128, 1, M, N, K, ILP=False),
    "v0_preshuffle": lambda: build_v0_preshuffle(128, 128, 1, M, N, K),
    "v1": lambda: build_v1(256, 256, 1, M, N, K),
    "v1_flyswz": lambda: build_v1(256, 256, 1, M, N, K, SWIZZLE="fly"),
    "v2_nosched": lambda: build_v2(256, 256, 1, M, N, K, SCHED=False),
    "v2": lambda: build_v2(256, 256, 1, M, N, K),
}
PRESHUFFLED = {"v0_preshuffle", "v0_preshuffle_noilp"}   # inputs rearranged once on the host, not timed


def timeit(fn, *args, iters=50, warmup=5):
    for _ in range(warmup):
        fn(*args)
    
    replicate_args = []
    for i in range(iters):
        replicate_args.append([a.clone() if isinstance(a, torch.Tensor) else a for a in args])

    torch.cuda.synchronize()
    ts = []
    for i in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn(*replicate_args[i])
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3)
    return sorted(ts)[len(ts) // 2]


def main(names):
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / math.sqrt(K)
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / math.sqrt(K)
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    ref = A.float() @ B.float().T
    stream = torch.cuda.current_stream().cuda_stream
    flop = 2 * M * N * K
    print(f"gemm {M}x{N}x{K} bf16 -> fp32")
    for name in names:
        if name == "torch":
            us = timeit(lambda: torch.matmul(A, B.T))
            print(f"{name:<12} {us:9.1f} us {flop / us / 1e6:7.1f} TFLOPS  (hipBLASLt, bf16 out)")
            continue
        launcher = CASES[name]()
        a, b = (preshuffle(A), preshuffle(B)) if name in PRESHUFFLED else (A, B)
        compiled = flyc.compile[HINTS](launcher, a, b, C, fx.Stream(stream))
        C.zero_()
        compiled(a, b, C, fx.Stream(stream))
        ok = torch.allclose(C, ref, atol=1e-5, rtol=1e-5)
        us = timeit(lambda: compiled(a, b, C, fx.Stream(stream)))
        print(f"{name:<12} {us:9.1f} us {flop / us / 1e6:7.1f} TFLOPS  correct={ok}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(CASES) + ["torch"])
