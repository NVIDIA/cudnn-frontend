# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-local stream NEIGHBOURS of the add + RMSNorm + quantize decode step, for its dependent-launch cells: three byte-copy launches.

* ``DependentCopy(device)`` -- a DEPENDENT of the step.  Launched with the programmatic-stream-serialization attribute, it issues
  ``griddepcontrol.wait`` as its first statement and then copies ``n`` bytes from ``src`` to ``dst``: the minimal consumer that honours
  the release's obligation (wait before reading the predecessor's outputs).  Behind an entry built with ``pdl=True`` (wait + release)
  this grid may START while the step's kernel still runs; its wait holds the copy until the step has completed, so the copied bytes
  are the step's.
* ``DependentCopy(device, wait=False)`` -- the same launch WITHOUT the wait: a positive control.  Behind a predecessor that released
  early it copies whatever its source holds when it starts.
* ``EarlyReleaseWriter(device, passes)`` -- a PREDECESSOR of the step.  An ordinary launch of ONE thread block that issues
  ``griddepcontrol.launch_dependents`` as its first statement and only then writes ``dst``, in ``passes`` passes over the whole
  buffer: every pass but the last writes the COMPLEMENT of ``src``'s bytes, the last one ``src``'s bytes.  Its dependents may start
  at once, while the destination holds ``src``'s bytes only when the grid has completed -- which takes far longer than a launch
  needs to start, on purpose.  A dependent that reads this grid's destination in front of its own wait reads other bytes.

Not a test module (no ``test_`` prefix) and not part of the package: imported by name from the suite beside it, after the DSL gate."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from cudnn.frost.tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids

THREADS = 256
#: the writer's default pass count: every pass walks the whole buffer with one thread block, long against the start of a dependent
WRITER_PASSES = 8


@cute.kernel
def pdl_dependent_copy_kernel(src: cute.Tensor, dst: cute.Tensor, n: cutlass.Int32):
    wait_on_dependent_grids()  # the dependent's obligation: nothing of the predecessor is read before this wait
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    i = bid * THREADS + tid
    if i < n:
        dst[i] = src[i]


pdl_dependent_copy_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def pdl_unwaited_copy_kernel(src: cute.Tensor, dst: cute.Tensor, n: cutlass.Int32):
    # the control: NO wait -- behind a predecessor that released early this copy reads what that grid has not written yet
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    i = bid * THREADS + tid
    if i < n:
        dst[i] = src[i]


pdl_unwaited_copy_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def pdl_early_release_writer_kernel(src: cute.Tensor, dst: cute.Tensor, n: cutlass.Int32, passes: cutlass.Int32):
    launch_dependent_grids()  # the FIRST statement: this grid's dependents may start now, in front of every byte written below
    tid, _, _ = cute.arch.thread_idx()
    for p in cutlass.range(passes, unroll=1):  # ONE thread block walks the whole buffer `passes` times: slow on purpose
        flip = cutlass.Int32(255)  # every pass but the last writes the complement of src's bytes
        if p == passes - 1:
            flip = cutlass.Int32(0)
        for i in cutlass.range(tid, n, THREADS, unroll=1):
            dst[i] = (src[i].to(cutlass.Int32) ^ flip).to(cutlass.Uint8)


pdl_early_release_writer_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def pdl_dependent_copy_launch(src: cute.Tensor, dst: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
    pdl_dependent_copy_kernel(src, dst, n).launch(grid=((n + THREADS - 1) // THREADS, 1, 1), block=(THREADS, 1, 1), stream=stream, use_pdl=True)


@cute.jit
def pdl_unwaited_copy_launch(src: cute.Tensor, dst: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
    pdl_unwaited_copy_kernel(src, dst, n).launch(grid=((n + THREADS - 1) // THREADS, 1, 1), block=(THREADS, 1, 1), stream=stream, use_pdl=True)


@cute.jit
def pdl_early_release_writer_launch(src: cute.Tensor, dst: cute.Tensor, n: cutlass.Int32, passes: cutlass.Int32, stream: cuda.CUstream):
    pdl_early_release_writer_kernel(src, dst, n, passes).launch(grid=(1, 1, 1), block=(THREADS, 1, 1), stream=stream, use_pdl=False)


class _ByteCopy:
    """``copy(src, dst)``: the bytes of ``src`` into ``dst`` (two contiguous tensors of one byte size) with ONE launch of ``launch``
    on torch's current stream of ``device``.  Compiled once: the byte count and the ``extra`` integers are runtime arguments."""

    def __init__(self, device, launch, *extra):
        self.device = device
        self._launch = launch
        self._extra = tuple(int(v) for v in extra)
        self._fn = None

    @staticmethod
    def _view(t):
        import torch
        from cutlass.cute.runtime import from_dlpack

        flat = t.detach().view(torch.uint8).reshape(-1)
        return from_dlpack(flat, assumed_align=16).mark_layout_dynamic()

    def __call__(self, src, dst):
        import torch
        from cutlass.cute.runtime import make_fake_stream

        n = src.numel() * src.element_size()
        assert n == dst.numel() * dst.element_size() and n > 1
        args = [self._view(src), self._view(dst), cutlass.Int32(n), *(cutlass.Int32(v) for v in self._extra)]
        if self._fn is None:
            self._fn = cute.compile(self._launch, *args, make_fake_stream())
        self._fn(*args, cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream))


class DependentCopy(_ByteCopy):
    """A dependent launch that copies ``src`` into ``dst``: behind its own wait (the default), or with ``wait=False`` without one
    (the positive control of the early start)."""

    def __init__(self, device, wait=True):
        super().__init__(device, pdl_dependent_copy_launch if wait else pdl_unwaited_copy_launch)


class EarlyReleaseWriter(_ByteCopy):
    """A predecessor that releases its dependents first and then writes ``src``'s bytes into ``dst`` with one thread block, in
    ``passes`` passes of which only the last leaves ``src``'s bytes."""

    def __init__(self, device, passes=WRITER_PASSES):
        assert int(passes) >= 1
        super().__init__(device, pdl_early_release_writer_launch, passes)
