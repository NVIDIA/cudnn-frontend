# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
from typing import Iterator, Optional

from cuda.bindings import driver as cuda


class GroupedGemmBackend(str, Enum):
    BF16 = "bf16"
    BLOCK_SCALED = "block_scaled"


def _resolve_torch_stream(handle: int, device: torch.device):
    """The torch stream object for a raw CUDA stream handle.

    Prefers torch's own current/default stream objects when the handle matches one of them, so
    the common case does not mint an ExternalStream per call.
    """
    import torch

    torch_current = torch.cuda.current_stream(device)
    if handle == torch_current.cuda_stream:
        return torch_current
    torch_default = torch.cuda.default_stream(device)
    if handle == torch_default.cuda_stream:
        return torch_default
    return torch.cuda.ExternalStream(handle, device=device)


@contextmanager
def _torch_stream_context(current_stream: Optional[cuda.CUstream], device: torch.device) -> Iterator[None]:
    """Run PyTorch work on the CUDA stream used for the kernel launch.

    torch-only: callers must guard this context so non-torch (e.g. JAX) code paths
    never enter it -- it imports torch and interprets ``device`` as a torch device.
    """
    import torch

    if current_stream is None:
        yield
        return
    with torch.cuda.stream(_resolve_torch_stream(int(current_stream), device)):
        yield


def _record_streams(current_stream: Optional[cuda.CUstream], device: torch.device, *tensors) -> None:
    """Keep torch's allocator from recycling buffers the kernel is still writing.

    A wrapper that allocates its outputs on torch's stream but launches on the caller's has torch
    tag each block to the allocation stream, so it can hand the block to the next allocation there
    as soon as the tensor is freed -- without waiting for the kernel. Recording the consumer stream
    is what defers that reuse. No-op when the two are the same stream, which is the common case and
    where record_stream would only add the block to the allocator's event-polled path for nothing.

    torch-only, like _torch_stream_context: guard the call on the framework.
    """
    if current_stream is None:
        return
    import torch

    handle = int(current_stream)
    if handle == torch.cuda.current_stream(device).cuda_stream:
        return
    consumer = _resolve_torch_stream(handle, device)
    for tensor in tensors:
        if tensor is not None:
            tensor.record_stream(consumer)


def select_grouped_gemm_backend(
    *,
    operation,
    a_dtype,
    b_dtype,
    scale_controls,
    block_scaled_dtype_pairs,
):
    # Compare in canonical (cutlass) dtype space so torch/jax/numpy/str dtypes all
    # resolve; dtypes with no cutlass mapping fall through to the unsupported-pair error.
    import cutlass

    from cudnn.datatypes import _convert_to_cutlass_data_type_or_none

    a_dtype_canonical = _convert_to_cutlass_data_type_or_none(a_dtype)
    b_dtype_canonical = _convert_to_cutlass_data_type_or_none(b_dtype)
    bf16_operands = (a_dtype_canonical is cutlass.BFloat16, b_dtype_canonical is cutlass.BFloat16)
    if any(bf16_operands):
        if not all(bf16_operands):
            raise ValueError(f"{operation}: mixed dtype families: a_dtype={a_dtype}, " f"b_dtype={b_dtype}")
        forbidden = [name for name, value in scale_controls if value is not None]
        if forbidden:
            raise ValueError(f"{operation}: BF16 forbids scale control {forbidden[0]}")
        return GroupedGemmBackend.BF16
    canonical_pairs = {
        (_convert_to_cutlass_data_type_or_none(pair_a), _convert_to_cutlass_data_type_or_none(pair_b)) for pair_a, pair_b in block_scaled_dtype_pairs
    }
    if (a_dtype_canonical, b_dtype_canonical) in canonical_pairs:
        return GroupedGemmBackend.BLOCK_SCALED
    raise ValueError(f"{operation}: unsupported dtype pair a_dtype={a_dtype}, " f"b_dtype={b_dtype}")


def backend_cache_key(backend, *components):
    return (backend.value, *components)


def rubin_single_group_offsets_kwarg(is_rubin_kernel, use_single_group_runtime_offsets):
    """Return the ``use_single_group_runtime_offsets`` kwarg for a kernel constructor.

    All grouped GEMM kernels accepting this helper implement
    ``use_single_group_runtime_offsets``. Keep the helper so the call sites share a
    single constructor-argument policy.
    """
    return {"use_single_group_runtime_offsets": use_single_group_runtime_offsets}
