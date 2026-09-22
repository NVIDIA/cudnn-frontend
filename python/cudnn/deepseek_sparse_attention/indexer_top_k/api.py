# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""APIBase wrapper for the DSA indexer top-K CuTe DSL kernel."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import cuda.bindings.driver as cuda

from cudnn._torch_stream import as_torch_stream, contiguous_on_stream
from cudnn.api_base import APIBase, TupleDict, WorkspaceCarver, ws_align
from cudnn.deepseek_sparse_attention.utils.runtime import (
    device_capability,
    torch_stream_context as _torch_stream_context,
)

from .local_to_global_dsl import local_to_global as _local_to_global
from .compactify import compactify as _compactify

_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _kernel_module():
    from . import indexer_top_k_decode_varlen

    return indexer_top_k_decode_varlen


def _check_execute_tensor(tensor, name: str, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device) -> None:
    if tensor is None:
        raise ValueError(f"{name} is required")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} tensor shape mismatch: expected {tuple(shape)}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} dtype mismatch: expected {dtype}, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got strides {tuple(tensor.stride())}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")


class IndexerTopK(APIBase):
    """Top-K filter using the SM90+ CuTe-DSL radix kernel.

    Selects the ``top_k`` largest entries from each row of ``input_values``.

    Parameter conventions (important — easy to misuse)
    --------------------------------------------------
    ``input_values`` has shape ``(n_rows, num_cols)``. The kernel treats the
    rows as belonging to ``batch_size = seq_lens.numel()`` groups, with
    ``n_rows == batch_size * next_n`` exactly. Within each group of
    ``next_n`` rows the kernel applies a **speculative-decoding stagger**:
    the effective valid length for row ``task_id`` inside batch ``b`` is
    ``seq_lens[b] - next_n + (task_id % next_n) + 1`` (i.e. the first row in
    a group sees the shortest prefix, the last row sees the full
    ``seq_lens[b]``).

    Use cases:

    * **Independent per-row top-K over equal-length rows** (the common
      case): set ``next_n = 1``, ``batch_size = n_rows``, ``seq_lens`` a
      length-``n_rows`` tensor. Each row then sees its own ``seq_lens[i]``
      columns.
    * **Speculative decoding / medusa-style drafts**: set ``next_n`` to the
      number of draft tokens per batch; ``seq_lens`` describes the cache
      length per batch. The kernel produces the staggered behaviour
      automatically.

    Setting ``next_n > 1`` with ``batch_size < n_rows / next_n`` causes the
    stagger formula to produce non-positive lengths for early rows; those
    rows receive no kernel writes and ``out_indices`` keeps whatever the
    caller stored there.

    Memory contract (Rule 8)
    ------------------------
    The class owns no device memory. ``execute()`` writes into the caller's
    ``out_indices`` (``(n_rows, top_k)`` int32) and, when ``return_val``,
    ``out_values`` (``(n_rows, top_k)`` in the input dtype), and carves the
    radix scratch from the caller's ``workspace`` of
    :meth:`scratch_workspace_bytes` bytes. ``compile()`` builds the kernel
    from fake tensors; the module-level cache is keyed on the
    next-power-of-two of ``num_cols`` and the plan-time flags.
    """

    def __init__(
        self,
        sample_input_values: torch.Tensor,
        sample_seq_lens: torch.Tensor,
        top_k: int,
        next_n: int = 1,
        return_val: bool = True,
        num_copy_bits: int = 256,
    ):
        super().__init__()
        self.input_desc = self._make_tensor_desc(sample_input_values, name="input_values")
        self.seq_lens_desc = self._make_tensor_desc(sample_seq_lens, name="seq_lens")
        self.top_k = int(top_k)
        self.next_n = int(next_n)
        self.return_val = bool(return_val)
        self.num_copy_bits = int(num_copy_bits)

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")
        self._check_dtype(self.input_desc, list(_SUPPORTED_DTYPES), name="input_values")
        self._check_dtype(self.seq_lens_desc, torch.int32, name="seq_lens")
        self._value_error_if(
            self.input_desc.ndim != 2,
            f"input_values must be 2-D (n_rows, num_cols), got {self.input_desc.shape}",
        )
        self._value_error_if(
            self.seq_lens_desc.ndim != 1,
            f"seq_lens must be 1-D, got {self.seq_lens_desc.shape}",
        )
        self._value_error_if(
            self.top_k <= 0 or self.top_k > 2048,
            f"top_k must be in (0, 2048], got {self.top_k}",
        )

        # Enforce the kernel's n_rows == batch_size * next_n invariant
        # up-front so misuse surfaces here rather than as silently-empty
        # rows (the stagger formula produces non-positive lengths when
        # next_n exceeds n_rows per batch).
        n_rows = self.input_desc.shape[0]
        batch_size = self.seq_lens_desc.shape[0]
        self._value_error_if(
            n_rows != batch_size * self.next_n,
            f"n_rows ({n_rows}) must equal seq_lens.numel() * next_n "
            f"({batch_size} * {self.next_n} = {batch_size * self.next_n}). "
            f"For independent top-K over equal-length rows use "
            f"next_n=1 and seq_lens of shape (n_rows,).",
        )

        self._check_tensor_stride(self.input_desc, stride=(self.input_desc.shape[1], 1), name="input_values")
        self._check_tensor_stride(self.seq_lens_desc, stride=(1,), name="seq_lens")

        major, _ = device_capability(self.input_desc.device)
        self._runtime_error_if(
            major < 9,
            f"IndexerTopK requires SM90+ compute capability, found SM{major}",
        )
        self._is_supported = True
        return True

    def _buffer_shape(self) -> Tuple[int, int, int]:
        n_rows, num_cols = self.input_desc.shape
        return n_rows, _kernel_module().buffer_numbers(self.input_desc.dtype), num_cols

    def scratch_workspace_bytes(self) -> int:
        """Bytes of ``workspace`` ``execute()`` carves its int32 radix scratch from (R2)."""
        n_rows, buffer_numbers, num_cols = self._buffer_shape()
        return ws_align(n_rows * buffer_numbers * num_cols * torch.int32.itemsize)

    def compile(self) -> None:
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        n_rows, num_cols = self.input_desc.shape
        # Fake operands only (R11); the kernel is compiled for the current device.
        with torch.cuda.device(self.input_desc.device):
            self._compiled_kernel = _kernel_module().compile_topk_kernel(
                self.input_desc.dtype,
                n_rows,
                num_cols,
                self.top_k,
                self.next_n,
                return_val=self.return_val,
                num_copy_bits=self.num_copy_bits,
            )

    def execute(
        self,
        input_values: torch.Tensor,
        seq_lens: torch.Tensor,
        out_indices: torch.Tensor,
        out_values: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
        workspace: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Launch into the caller's ``out_indices`` / ``out_values``; returns them.

        ``out_values`` is required when the object was built with ``return_val=True``
        and must be ``None`` otherwise. ``workspace`` is a CUDA uint8 tensor of at
        least :meth:`scratch_workspace_bytes` bytes on ``input_values``' device.
        """
        self._logger.debug("Entering execute")
        self._ensure_support_checked()
        if self._compiled_kernel is None:
            self.compile()

        device = input_values.device
        n_rows, buffer_numbers, num_cols = self._buffer_shape()
        _check_execute_tensor(input_values, "input_values", self.input_desc.shape, self.input_desc.dtype, device)
        _check_execute_tensor(seq_lens, "seq_lens", self.seq_lens_desc.shape, torch.int32, device)
        _check_execute_tensor(out_indices, "out_indices", (n_rows, self.top_k), torch.int32, device)
        if self.return_val:
            self._value_error_if(out_values is None, "out_values is required: this IndexerTopK was built with return_val=True")
            _check_execute_tensor(out_values, "out_values", (n_rows, self.top_k), self.input_desc.dtype, device)
        else:
            self._value_error_if(out_values is not None, "out_values must be None: this IndexerTopK was built with return_val=False")

        kernel_module = _kernel_module()
        carver = WorkspaceCarver(workspace, self.scratch_workspace_bytes(), "IndexerTopK")
        buffer = carver.take(n_rows * buffer_numbers * num_cols, torch.int32).view(n_rows, buffer_numbers, num_cols)
        self._value_error_if(
            buffer.data_ptr() % kernel_module.BUFFER_ALIGN != 0,
            f"IndexerTopK workspace must be {kernel_module.BUFFER_ALIGN}-byte aligned; got data_ptr=0x{buffer.data_ptr():x}",
        )

        # The compiled kernel launches on the TVM-FFI environment stream, i.e.
        # torch's current stream: enter the caller's stream so it is honored.
        with _torch_stream_context(current_stream):
            kernel_module.launch_topk_kernel(self._compiled_kernel, input_values, seq_lens, out_indices, out_values, buffer, self.next_n)

        # TVM-FFI launches are invisible to the caching allocator: record every
        # tensor the kernel touches on the launch stream (R1).
        launch_stream = as_torch_stream(current_stream, device) if current_stream is not None else torch.cuda.current_stream(device)
        for tensor in (input_values, seq_lens, out_indices, out_values, workspace):
            if tensor is not None:
                tensor.record_stream(launch_stream)
        return out_indices, out_values


_cache_of_IndexerTopKObjects: dict = {}


def indexer_top_k_wrapper(
    input_values: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k: int,
    next_n: int = 1,
    return_val: bool = True,
    num_copy_bits: int = 256,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """High-level wrapper returning ``{'indices', 'values'}``.

    ``input_values`` is ``(n_rows, num_cols)``; the kernel requires
    ``n_rows == seq_lens.numel() * next_n`` and applies a
    speculative-decoding length stagger within each ``next_n``-row group
    (first row sees the shortest prefix, last sees the full ``seq_lens``).
    For "independent top-K over every row" set ``next_n=1`` and make
    ``seq_lens`` a length-``n_rows`` tensor. See :class:`IndexerTopK` for
    full details.

    ``values`` is ``None`` when ``return_val=False``. The outputs and the
    :meth:`IndexerTopK.scratch_workspace_bytes` workspace are allocated here
    per call, on ``stream``; use the class API with caller-owned buffers to
    avoid the allocations. Strided ``input_values`` / ``seq_lens`` are copied
    contiguous here (the class declines them in ``check_support()``).
    """
    # R1 staging: the originals are record_stream'ed on `stream` before they are rebound.
    input_values = contiguous_on_stream(input_values, stream, input_values.device)
    seq_lens = contiguous_on_stream(seq_lens, stream, seq_lens.device)
    cache_key = (
        input_values.dtype,
        int(input_values.shape[0]),  # n_rows affects wrapper validation and output shape
        input_values.shape[-1],  # num_cols buckets internally
        int(seq_lens.shape[0]),  # batch_size / seq_lens.numel()
        int(top_k),
        int(next_n),
        bool(return_val),
        int(num_copy_bits),
    )
    obj = _cache_of_IndexerTopKObjects.get(cache_key)
    if obj is None:
        obj = IndexerTopK(
            sample_input_values=input_values,
            sample_seq_lens=seq_lens,
            top_k=top_k,
            next_n=next_n,
            return_val=return_val,
            num_copy_bits=num_copy_bits,
        )
        assert obj.check_support()
        obj.compile()
        _cache_of_IndexerTopKObjects[cache_key] = obj

    n_rows = int(input_values.shape[0])
    device = input_values.device
    with torch.cuda.device(device), _torch_stream_context(stream):
        indices = torch.empty(n_rows, top_k, dtype=torch.int32, device=device)
        values = torch.empty(n_rows, top_k, dtype=input_values.dtype, device=device) if return_val else None
        workspace = torch.empty(obj.scratch_workspace_bytes(), dtype=torch.uint8, device=device)
    obj.execute(input_values, seq_lens, indices, values, current_stream=stream, workspace=workspace)
    return TupleDict(indices=indices, values=values)


def local_to_global_wrapper(
    local_indices: torch.Tensor,
    seqlen_k: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Convert local top-K indices to the global index space."""
    indices = _local_to_global(
        local_indices,
        seqlen_k,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        stream=stream,
    )
    return TupleDict(indices=indices)


def compactify_wrapper(
    indices: torch.Tensor,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Pack valid indices row-wise and return ``indices`` plus ``topk_length``."""
    compact_indices, topk_length = _compactify(indices, stream=stream)
    return TupleDict(indices=compact_indices, topk_length=topk_length)
