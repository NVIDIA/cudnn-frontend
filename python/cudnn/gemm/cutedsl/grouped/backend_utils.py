# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from enum import Enum
from typing import Iterator, Optional

import torch
from cuda.bindings import driver as cuda


class GroupedGemmBackend(str, Enum):
    BF16 = "bf16"
    BLOCK_SCALED = "block_scaled"


@contextmanager
def _torch_stream_context(current_stream: Optional[cuda.CUstream], device: torch.device) -> Iterator[None]:
    """Run PyTorch work on the CUDA stream used for the kernel launch."""
    if current_stream is None:
        yield
        return
    handle = int(current_stream)
    torch_current = torch.cuda.current_stream(device)
    torch_default = torch.cuda.default_stream(device)
    if handle == torch_current.cuda_stream:
        launch_stream = torch_current
    elif handle == torch_default.cuda_stream:
        launch_stream = torch_default
    else:
        launch_stream = torch.cuda.ExternalStream(handle, device=device)
    with torch.cuda.stream(launch_stream):
        yield


def select_grouped_gemm_backend(
    *,
    operation,
    a_dtype,
    b_dtype,
    scale_controls,
    block_scaled_dtype_pairs,
):
    bf16_operands = (a_dtype == torch.bfloat16, b_dtype == torch.bfloat16)
    if any(bf16_operands):
        if not all(bf16_operands):
            raise ValueError(f"{operation}: mixed dtype families: a_dtype={a_dtype}, " f"b_dtype={b_dtype}")
        forbidden = [name for name, value in scale_controls if value is not None]
        if forbidden:
            raise ValueError(f"{operation}: BF16 forbids scale control {forbidden[0]}")
        return GroupedGemmBackend.BF16
    if (a_dtype, b_dtype) in block_scaled_dtype_pairs:
        return GroupedGemmBackend.BLOCK_SCALED
    raise ValueError(f"{operation}: unsupported dtype pair a_dtype={a_dtype}, " f"b_dtype={b_dtype}")


def backend_cache_key(backend, *components):
    return (backend.value, *components)


def rubin_single_group_offsets_kwarg(is_rubin_kernel, use_single_group_runtime_offsets):
    """Return the ``use_single_group_runtime_offsets`` kwarg for a kernel constructor.

    The Rubin (sm107) grouped GEMM kernels predate ``use_single_group_runtime_offsets``
    and do not accept it, so forwarding it unconditionally is a ``TypeError`` even when
    it is ``False``. Send it only to kernels that implement it, and reject an explicit
    request on Rubin rather than silently ignoring it and running a different schedule
    than the caller asked for.
    """
    if not is_rubin_kernel:
        return {"use_single_group_runtime_offsets": use_single_group_runtime_offsets}
    if use_single_group_runtime_offsets:
        raise NotImplementedError("The Rubin grouped GEMM kernels do not support use_single_group_runtime_offsets")
    return {}
