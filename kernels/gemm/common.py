
import torch
import math
import flydsl.compiler as flyc
import flydsl.expr as fx

def test_gemm(M, N, K, launcher, prep=None):
    """prep: optional input transform (e.g. v0_preshuffle.preshuffle); the
    reference still uses the original A, B."""
    stream = torch.cuda.current_stream().cuda_stream
    in_dtype = torch.bfloat16
    out_dtype = torch.float32
    A = torch.randn(M, K, dtype=in_dtype).cuda() / math.sqrt(K)
    B = torch.randn(N, K, dtype=in_dtype).cuda() / math.sqrt(K)
    C = torch.zeros(M, N, dtype=out_dtype).cuda()
    a, b = (prep(A), prep(B)) if prep else (A, B)

    hints = {
        "opt_level": 2,
        "llvm_options": {"amdgpu-mfma-vgpr-form": False}
    }
    compiled = flyc.compile[hints](launcher, a, b, C, fx.Stream(stream))
    
    compiled(a, b, C, fx.Stream(stream))

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)

    tflops = 2 * M * N * K * 1e-12
    for i in range(10):
        torch.cuda._sleep(1_000_000) # avoid CPU side overhead affecting timing
        ev_start.record()
        compiled(a, b, C, fx.Stream(stream))
        ev_end.record()
        torch.cuda.synchronize()
        dt_us = ev_start.elapsed_time(ev_end) * 1e3
        print(f"[{i}]: {dt_us:.2f} us  ({tflops / (dt_us * 1e-6):.2f} TFLOPS)")

    expected = A.to(out_dtype) @ B.to(out_dtype).T
    is_correct = torch.allclose(C, expected, atol=1e-5, rtol=1e-5)
    print(f" gemm {M}x{N}x{K} Correctness: {is_correct}")
