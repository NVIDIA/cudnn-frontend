# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A test-local DEPENDENT launch for the SwiGLU FP4 decode entry's chain cell: one byte-copy kernel launched with the
programmatic-stream-serialization attribute.  It issues ``griddepcontrol.wait`` as its first statement and then copies ``n`` bytes
from ``src`` to ``dst`` -- the minimal consumer that honours the entry's obligation for a dependent (wait before reading ``out`` or
``out_scale``).  A thread block of the entry signals its dependents when its loader warp has finished, before the block's last output
tile is written, so behind an entry built with ``pdl=True`` this grid may START while the entry's kernel still writes; its wait holds
the copy until that kernel has completed, so the copied bytes are the entry's.

Not a test module (no ``test_`` prefix) and not part of the package: imported by name from the suite beside it, after the DSL gate."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from cudnn.frost.tile_dsl.barrier import wait_on_dependent_grids

THREADS = 256


@cute.kernel
def swiglu_dependent_copy_kernel(src: cute.Tensor, dst: cute.Tensor, n: cutlass.Int32):
    wait_on_dependent_grids()  # the dependent's obligation: nothing the predecessor writes is read before this wait
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    i = bid * THREADS + tid
    if i < n:
        dst[i] = src[i]


swiglu_dependent_copy_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def swiglu_dependent_copy_launch(src: cute.Tensor, dst: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
    swiglu_dependent_copy_kernel(src, dst, n).launch(grid=((n + THREADS - 1) // THREADS, 1, 1), block=(THREADS, 1, 1), stream=stream, use_pdl=True)


class DependentCopy:
    """``DependentCopy(device)(src, dst)``: copy the bytes of ``src`` into ``dst`` (two contiguous tensors of one byte size) with ONE
    dependent launch on torch's current stream of ``device``.  Compiled once (the byte count is a runtime argument)."""

    def __init__(self, device):
        self.device = device
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
        assert n == dst.numel() * dst.element_size() and n > 1, "the two tensors must hold the same number of bytes"
        a, b = self._view(src), self._view(dst)
        if self._fn is None:
            self._fn = cute.compile(swiglu_dependent_copy_launch, a, b, cutlass.Int32(n), make_fake_stream())
        self._fn(a, b, cutlass.Int32(n), cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream))
