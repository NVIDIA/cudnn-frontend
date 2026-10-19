# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The programmatic-dependent-launch (PDL) probes of the gated-block suite -- ONE copy, shared by the PDL test modules.

``delayed_write``: a PRIMARY kernel that releases its dependent grid at once (``griddepcontrol.launch_dependents`` as its first
instruction), spins ``spin_us`` on ``%globaltimer`` and only THEN copies ``src`` over ``dst``.  A kernel launched right after it
with the PDL attribute may start during the spin; one that waits on its dependent grid (``griddepcontrol.wait``) before its first
read of ``dst`` sees the copy, one that reads ``dst`` earlier sees whatever the caller left there (the tests leave NaN).  That is
the detector of a late or missing wait: GREEN with the shipped kernel, RED with its wait removed (``retrace``).

``retrace``: forget the in-process compiled objects (the compiled-plan memo and the kernel modules' ``lru_cache``d ``compile``
entries) so the next ``compile()`` traces the kernel body again -- the way a test swaps a module-level primitive (the wait) for a
no-op under ``monkeypatch`` and gets a kernel WITHOUT it; the caller disables the on-disk compiled-plan cache for the same reason
(``CUDNN_FRONTEND_DISABLE_COMPILED_CACHE=1``: the cache key hashes the template FILE and the compile arguments, not the live module
globals, so a swapped primitive would otherwise reload the shipped artifact).  Call it again after the swap is undone: the memo
would otherwise hand the no-wait object to the next plan with the same key.

``cuda_versions``: the driver / runtime / torch / DSL versions a capture test records (programmatic edges in a captured graph are a
driver feature; the record names what served the run).
"""

import importlib.metadata

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from cudnn.frost.tile_dsl.barrier import launch_dependent_grids

THREADS = 256
_compiled = {}
_FAKE_STREAM = None


@cute.kernel
def frost_test_delayed_write(mSrc: cute.Tensor, mDst: cute.Tensor, n: cutlass.Int32, spin_ns: cutlass.Int64) -> None:
    # Release the dependent grid FIRST: a dependent launched with the attribute may now start.  A correct dependent still waits
    # on this grid's completion before its first dependent read; an incorrect one reads dst before the copy below lands.
    launch_dependent_grids()
    t0 = cute.arch.globaltimer()
    while cute.arch.globaltimer() - t0 < spin_ns:
        pass
    src = cutlass.make_array_view(mSrc)
    dst = cutlass.make_array_view(mDst)
    tid = cutlass.Int32(cute.arch.block_idx()[0]) * cutlass.Int32(THREADS) + cutlass.Int32(cute.arch.thread_idx()[0])
    stride = cutlass.Int32(cute.arch.grid_dim()[0]) * cutlass.Int32(THREADS)
    for i in cutlass.range(tid, n, stride, unroll=1):
        dst[i] = src[i]


@cute.jit
def delayed_write_launch(src: cute.Tensor, dst: cute.Tensor, n: cutlass.Int32, spin_ns: cutlass.Int64, n_blocks: cutlass.Int32, stream: cuda.CUstream):
    # An ORDINARY launch (no PDL attribute): the primary of the probe is never itself a dependent of what ran before it.
    frost_test_delayed_write(src, dst, n, spin_ns).launch(grid=(n_blocks, 1, 1), block=(THREADS, 1, 1), stream=stream)


frost_test_delayed_write.set_name_prefix("cudnn", remove_cutlass_symbol=True)


def _compiled_delayed_write():
    global _FAKE_STREAM
    if "delayed_write" not in _compiled:
        if _FAKE_STREAM is None:
            from cutlass.cute.runtime import make_fake_stream

            _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)
        fake = lambda: cute.runtime.make_fake_tensor(dtype=cutlass.Int32, shape=(cute.sym_int(),), stride=(1,), assumed_align=16)  # noqa: E731
        _compiled["delayed_write"] = cute.compile(
            delayed_write_launch, fake(), fake(), cutlass.Int32(0), cutlass.Int64(0), cutlass.Int32(1), _FAKE_STREAM, options="--enable-tvm-ffi"
        )
    return _compiled["delayed_write"]


def _as_i32(t: torch.Tensor) -> torch.Tensor:
    if not t.is_contiguous():
        raise ValueError("delayed_write takes contiguous buffers (the whole slab, a whole pool), not a strided band")
    if t.dtype == torch.int32:
        return t
    flat = t.view(-1)
    if (flat.numel() * flat.element_size()) % 4:
        raise ValueError("delayed_write copies 4-byte words; the buffer's byte count must be a multiple of 4")
    return flat.view(torch.int32)


def delayed_write(src: torch.Tensor, dst: torch.Tensor, *, spin_us: float, stream=None) -> None:
    """Enqueue the primary: ``dst[:] = src`` after ``spin_us`` microseconds, the dependent grid released at once (see the module docstring).
    ``src`` / ``dst`` are CUDA buffers of one shape and dtype (any dtype whose byte count is a multiple of 4), contiguous."""
    if src.shape != dst.shape or src.dtype != dst.dtype or not (src.is_cuda and dst.is_cuda):
        raise ValueError("delayed_write: src and dst must be CUDA buffers of one shape and dtype")
    s, d = _as_i32(src), _as_i32(dst)
    n = int(s.numel())
    n_blocks = max(1, min(64, (n + THREADS - 1) // THREADS))  # a few CTAs: the primary must never crowd the dependent off the SMs
    stream = torch.cuda.current_stream(src.device).cuda_stream if stream is None else stream
    _compiled_delayed_write()(s, d, cutlass.Int32(n), cutlass.Int64(int(spin_us * 1000)), cutlass.Int32(n_blocks), cuda.CUstream(int(stream)))


def retrace(*lru_compiles) -> None:
    """Forget every in-process compiled object: the compiled-plan memo and the given kernel modules' ``lru_cache``d ``compile`` entries."""
    from cudnn.frost import compiled_cache

    compiled_cache.clear_memo()
    for fn in lru_compiles:
        fn.cache_clear()


def cuda_versions() -> dict:
    """The driver, runtime, torch and DSL versions of this process (what a capture test records beside its verdict)."""
    from cuda.bindings import runtime as rt

    err, drv = cuda.cuDriverGetVersion()
    err2, run = rt.cudaRuntimeGetVersion()
    try:
        dsl = importlib.metadata.version("nvidia-cutlass-dsl")
    except Exception:  # noqa: BLE001
        dsl = "n/a"
    return dict(driver=int(drv), runtime=int(run), torch=torch.__version__, torch_cuda=torch.version.cuda, dsl=dsl)
