# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A predecessor that lets its programmatic dependents start before it writes, for the PDL cells of the general grouped suite.

``late_write(src, dst, spin_us=..., stream=...)`` enqueues one thread block that issues ``griddepcontrol.launch_dependents``
as its first instruction, spins ``spin_us`` microseconds on ``%globaltimer`` and only then copies ``src`` over ``dst``
(int32 words): the order of a routing kernel that releases the grouped GEMM behind it before it has written the routing.
A kernel launched after it with programmatic stream serialization may start during the spin. One that waits on its
predecessor (``griddepcontrol.wait``) before it reads ``dst`` reads the copy; one that reads ``dst`` earlier reads what
was there before. The launch itself is an ordinary one, and it is captured into a CUDA graph like any other.

Not a test module (no ``test_`` prefix): imported by the suite beside it, after its DSL gate."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from cudnn.frost.tile_dsl.barrier import launch_dependent_grids

THREADS = 32
_compiled = {}


@cute.kernel
def frost_test_late_write(mSrc: cute.Tensor, mDst: cute.Tensor, n: cutlass.Int32, spin_ns: cutlass.Int64) -> None:
    # The dependents may start now, before a word of dst is written.
    launch_dependent_grids()
    t0 = cute.arch.globaltimer()
    while cute.arch.globaltimer() - t0 < spin_ns:
        pass
    src = cutlass.make_array_view(mSrc)
    dst = cutlass.make_array_view(mDst)
    for i in cutlass.range(cutlass.Int32(cute.arch.thread_idx()[0]), n, THREADS, unroll=1):
        dst[i] = src[i]


@cute.jit
def late_write_launch(src: cute.Tensor, dst: cute.Tensor, n: cutlass.Int32, spin_ns: cutlass.Int64, stream: cuda.CUstream):
    # An ordinary launch: the writer is not itself a dependent of what runs before it.
    frost_test_late_write(src, dst, n, spin_ns).launch(grid=(1, 1, 1), block=(THREADS, 1, 1), stream=stream)


frost_test_late_write.set_name_prefix("cudnn", remove_cutlass_symbol=True)


def _late_write():
    if "late_write" not in _compiled:
        from cutlass.cute.runtime import make_fake_stream

        def fake():
            return cute.runtime.make_fake_tensor(dtype=cutlass.Int32, shape=(cute.sym_int(),), stride=(1,), assumed_align=4)

        _compiled["late_write"] = cute.compile(
            late_write_launch,
            fake(),
            fake(),
            cutlass.Int32(0),
            cutlass.Int64(0),
            make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )
    return _compiled["late_write"]


def late_write(src: torch.Tensor, dst: torch.Tensor, *, spin_us: float, stream: int) -> None:
    """Enqueue on ``stream`` (a raw handle): ``dst[:] = src`` after ``spin_us`` microseconds, the dependents released at once.
    ``src`` and ``dst`` are contiguous int32 CUDA tensors of one element count."""
    if src.dtype != torch.int32 or dst.dtype != torch.int32 or src.numel() != dst.numel() or not (src.is_contiguous() and dst.is_contiguous()):
        raise ValueError("late_write copies one contiguous int32 tensor over another of the same size")
    n = int(src.numel())
    _late_write()(src.view(-1), dst.view(-1), cutlass.Int32(n), cutlass.Int64(int(spin_us * 1000)), cuda.CUstream(int(stream)))
