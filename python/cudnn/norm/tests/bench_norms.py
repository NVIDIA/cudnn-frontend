"""Quick bandwidth benchmark: sm_100 CUTLASS-primitive LayerNorm vs PyTorch.

    python cudnn/norm/tests/bench_norms.py

Norm forward is memory-bound; the figure of merit is achieved global bandwidth
(GB/s) reading X once (cp.async-staged) + writing Y.
"""

import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_CUDNN_DIR = os.path.abspath(os.path.join(_HERE, "..", ".."))
if "cudnn" not in sys.modules:
    stub = types.ModuleType("cudnn")
    stub.__path__ = [_CUDNN_DIR]
    stub.pygraph = type("pygraph", (), {})
    sys.modules["cudnn"] = stub

import torch
import torch.nn.functional as F

from cudnn.norm import NormVariant, norm_fprop


def _time(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def bench_layernorm(N, D, dtype):
    x = torch.randn(N, D, device="cuda", dtype=dtype)
    g = torch.randn(D, device="cuda", dtype=dtype)
    b = torch.randn(D, device="cuda", dtype=dtype)
    bytes_moved = 2 * N * D * x.element_size()  # read X + write Y (dominant)

    def frost():
        norm_fprop(NormVariant.LAYER_NORM, x, g, b, normalized_shape=[D], eps=1e-5)

    def torch_ln():
        F.layer_norm(x, (D,), g, b, eps=1e-5)

    frost(); torch_ln()  # prime compiles
    tf, tt = _time(frost), _time(torch_ln)
    gbps = lambda ms: bytes_moved / (ms * 1e-3) / 1e9
    print(f"LN N={N} D={D} {str(dtype).split('.')[-1]:>8}: "
          f"frost {gbps(tf):.0f} | torch {gbps(tt):.0f} GB/s  ({tt / tf:.2f}x torch)")


def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)} {torch.cuda.get_device_capability(0)}")
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for (N, D) in [(4096, 1024), (8192, 4096), (16384, 8192)]:
            bench_layernorm(N, D, dtype)


if __name__ == "__main__":
    main()
