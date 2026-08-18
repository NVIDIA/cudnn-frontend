"""Flow 1 — imperative, experimental. What ships today.

No graph, no artifact, no ABI: compile the kernel in this process and call it.
That is what the shipped per-family entry points do underneath — JIT on first
use, cache in a module-level dict — and that cache dies with the process, so the
JIT cost is paid again in every process. Which is what flows 2 and 3 exist to
remove.

Deliberately the same kernel as flows 2 and 3, so the three flows differ in how
the kernel is reached and in nothing else.

This layer is EXPERIMENTAL: the API is not fixed and is free to change shape as
the kernels evolve.

    python flow1_imperative.py
"""

import time

import torch

from common import CONTAINER_KERNELS

NAME = "tile_add_large_f32"
SHAPE = CONTAINER_KERNELS[NAME]

# A module-level cache, exactly like the shipped imperative APIs keep. It is the
# thing that does not survive to the next process.
_CACHE = {}


def tile_add(a, b, c):
    """Compile on first use for this shape, then replay."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from cudnn.engines.cutedsl_tma_add_engine import build_kernel

    key = tuple(a.shape)
    if key not in _CACHE:
        fn = build_kernel()
        fakes = [cute.runtime.make_fake_compact_tensor(cutlass.Float32, key, stride_order=(1, 0)) for _ in range(3)]
        _CACHE[key] = cute.compile(fn, *fakes, cuda.CUstream(0), options="--enable-tvm-ffi")
    _CACHE[key](a, b, c, cuda.CUstream(torch.cuda.current_stream().cuda_stream))


def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")

    a = torch.randn(*SHAPE, device="cuda", dtype=torch.float32)
    b = torch.randn(*SHAPE, device="cuda", dtype=torch.float32)
    c = torch.zeros(*SHAPE, device="cuda", dtype=torch.float32)

    # First call JITs: seconds, not microseconds.
    t0 = time.perf_counter()
    tile_add(a, b, c)
    torch.cuda.synchronize()
    cold = time.perf_counter() - t0

    # Every later call in THIS process hits the cache above.
    t0 = time.perf_counter()
    tile_add(a, b, c)
    torch.cuda.synchronize()
    warm = time.perf_counter() - t0

    print(f"first call (JIT): {cold * 1e3:9.1f} ms")
    print(f"second call:      {warm * 1e3:9.3f} ms   <- the cache flows 2 and 3 make durable")
    err = (c - (a + b)).abs().max().item()
    print(f"max |err| = {err}")
    return 0 if err == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
