# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic reduction of the BF16 dSReLU kernel's per-N-tile dprob slots (private).

``dprob[m] += (((slots[m, 0] + slots[m, 1]) + slots[m, 2]) + ...)``: a strict left fold over the
N tiles in increasing tile order, FP32, one thread per row. Every slot is zeroed after it is read,
so a workspace the caller zero-initialized once stays ready for the next launch, and a second
reduction of the same workspace adds nothing.

Why not ``slots.sum(dim=1)``: torch's CUDA sum is deterministic run to run for a fixed shape and
device, but its order is a vectorized tree rather than the increasing-tile order, so the result
would depend on how many slots a row has in a way that a plain left fold does not.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass.cute.runtime import make_fake_stream
from cutlass.cute.typing import Int32

from cudnn._torch_stream import device_context
from cudnn.frost.compiled_cache import dsl_compile_lock

_THREADS = 256
_INT32_LIMIT = 2**31


class _DprobFoldKernel:
    """``out[m] += left_fold_j(slots[m * S + j])``, then zero the slots, over flat 1-D views.

    One thread per row sums its ``S`` slots in increasing ``j``. The slots are zeroed only after
    the block's barrier, by the whole block over its contiguous ``THREADS * S`` range, so the
    stores are coalesced (per-row stores interleaved with the loads cost ~20x the fold itself).
    """

    @cute.jit
    def __call__(self, mSlots: cute.Tensor, mOut: cute.Tensor, num_rows: Int32, num_slots: Int32, stream: cuda.CUstream):
        grid_x = (num_rows + (_THREADS - 1)) // _THREADS
        self.kernel(mSlots, mOut, num_rows, num_slots).launch(grid=(grid_x, 1, 1), block=(_THREADS, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, mSlots: cute.Tensor, mOut: cute.Tensor, num_rows: Int32, num_slots: Int32):
        tidx = cute.arch.thread_idx()[0]
        bidx = cute.arch.block_idx()[0]
        row = bidx * _THREADS + tidx
        if row < num_rows:
            base = row * num_slots
            acc = mSlots[base]
            for j in cutlass.range(1, num_slots, unroll=1):
                acc = acc + mSlots[base + j]
            mOut[row] = mOut[row] + acc
        # Every row of this block has been read: zero its slots, consecutive threads on
        # consecutive addresses.
        cute.arch.sync_threads()
        total = num_rows * num_slots
        block_base = bidx * _THREADS * num_slots
        for j in cutlass.range(0, num_slots, unroll=1):
            index = block_base + j * _THREADS + tidx
            if index < total:
                mSlots[index] = cutlass.Float32(0.0)

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


# One compiled fold per device index: its 1-D layouts are dynamic, so it serves every shape.
_compiled: dict = {}


def _fake_1d():
    """A contiguous FP32 vector of symbolic length: the compiled fold binds every rows * slots."""
    return cute.runtime.make_fake_tensor(dtype=cutlass.Float32, shape=(cute.sym_int(),), stride=(1,), assumed_align=4)


def compile_dprob_fold(device) -> object:
    """Compile the fold for ``device`` once and return it; later calls are cache hits.

    The BF16 dSReLU ``compile()`` calls this at plan time, so ``execute()`` only launches. It
    compiles from fake tensors and allocates nothing.
    """
    import torch

    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    fn = _compiled.get(index)
    if fn is None:
        with dsl_compile_lock():
            fn = _compiled.get(index)
            if fn is None:
                with device_context(index):
                    fn = cute.compile(
                        _DprobFoldKernel(),
                        _fake_1d(),
                        _fake_1d(),
                        Int32(1),
                        Int32(1),
                        make_fake_stream(use_tvm_ffi_env_stream=False),
                        options="--enable-tvm-ffi",
                    )
                _compiled[index] = fn
    return fn


def fold_dprob_slots(slots: torch.Tensor, dprob: torch.Tensor, current_stream: cuda.CUstream) -> None:
    """Add the left fold of each row of ``slots`` to ``dprob`` and zero ``slots``, on ``current_stream``.

    :param slots: ``(M, S, 1)`` (or ``(M, S)``) FP32 workspace with row-major contiguous rows.
    :param dprob: ``(M, 1, 1)`` (or ``(M,)``) FP32 output with a unit row stride.
    """
    num_rows, num_slots = int(slots.shape[0]), int(slots.shape[1])
    if num_rows == 0 or num_slots == 0:
        return
    if num_rows * num_slots >= _INT32_LIMIT:
        raise ValueError(f"dprob slots too large for int32 indexing: {num_rows} x {num_slots}")
    fn = compile_dprob_fold(slots.device)
    # Flat views of the caller's storage (as_strided never copies); detached so the launch never
    # exports a tensor that is on an autograd tape.
    args = (
        slots.detach().as_strided((num_rows * num_slots,), (1,)),
        dprob.detach().as_strided((num_rows,), (1,)),
        Int32(num_rows),
        Int32(num_slots),
        current_stream,
    )
    # Launch with the slots' device current, as the compile ran (Rule 5: restored on exit).
    with device_context(slots.device):
        fn(*args)
