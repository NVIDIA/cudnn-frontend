"""Quick bandwidth benchmark: scalar vs vectorized FROST norm vs PyTorch.

    python cudnn/norm/tests/bench_norms.py
"""

import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_CUDNN_DIR = os.path.abspath(os.path.join(_HERE, "..", ".."))
if "cudnn" not in sys.modules:
    stub = types.ModuleType("cudnn")
    stub.__path__ = [_CUDNN_DIR]
    sys.modules["cudnn"] = stub

import torch
import torch.nn.functional as F

from cudnn.norm.frost import NormVariant, norm_forward


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
    # forward reads x + writes y + reads gamma/beta ~= 2*N*D elems (dominant)
    bytes_moved = 2 * N * D * x.element_size()

    def scalar():
        norm_forward(NormVariant.LAYER_NORM, x, g, b, normalized_shape=[D], eps=1e-5, impl="scalar")

    def vec():
        norm_forward(NormVariant.LAYER_NORM, x, g, b, normalized_shape=[D], eps=1e-5, impl="vec")

    def cpa():
        norm_forward(NormVariant.LAYER_NORM, x, g, b, normalized_shape=[D], eps=1e-5, impl="cpasync")

    def torch_ln():
        F.layer_norm(x, (D,), g, b, eps=1e-5)

    # prime compiles
    scalar(); vec(); cpa(); torch_ln()
    ts, tv, tc, tt = _time(scalar), _time(vec), _time(cpa), _time(torch_ln)
    gbps = lambda ms: bytes_moved / (ms * 1e-3) / 1e9
    print(f"LN N={N} D={D} {str(dtype).split('.')[-1]:>8}: "
          f"scalar {gbps(ts):.0f} | vec {gbps(tv):.0f} | cpasync {gbps(tc):.0f} | "
          f"torch {gbps(tt):.0f} GB/s  (cpasync {ts/tc:.2f}x scalar, {tt/tc:.2f}x torch)")


def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for (N, D) in [(4096, 1024), (8192, 4096), (16384, 8192)]:
            bench_layernorm(N, D, dtype)


if __name__ == "__main__":
    main()
