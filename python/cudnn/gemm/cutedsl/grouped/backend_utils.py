# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from enum import Enum
from typing import Iterator, Optional, Sequence, Tuple

from cuda.bindings import driver as cuda

from cudnn.api_base import ceil_div
from cudnn.tensor_adapter import get_device, get_shape, get_strides, is_torch_tensor


class GroupedGemmBackend(str, Enum):
    BF16 = "bf16"
    BLOCK_SCALED = "block_scaled"


# Offset / pointer-table VALUES are a device-data contract: the kernels read them on
# device, and checking them on the host needs a D2H read that blocks the launch stream
# and is illegal under CUDA-graph capture (python/cudnn/AGENTS.md Rule 3, recipe R6).
# This switch restores the blocking checks for debugging. Read once at import; never
# on by default; never memoized.
DEBUG_VALIDATE_DEVICE_VALUES_ENV = "CUDNN_FE_GROUPED_GEMM_VALIDATE_DEVICE_VALUES"
DEBUG_VALIDATE_DEVICE_VALUES = os.getenv(DEBUG_VALIDATE_DEVICE_VALUES_ENV, "0") == "1"


def _check_not_capturing(stream, what: str) -> None:
    err, status = cuda.cuStreamIsCapturing(cuda.CUstream(int(stream)))
    if int(err) != 0:
        raise RuntimeError(f"cuStreamIsCapturing failed: {err}")
    if status != cuda.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE:
        raise RuntimeError(
            f"{what}: {DEBUG_VALIDATE_DEVICE_VALUES_ENV}=1 reads device values back to the host "
            "and cannot run under CUDA graph capture; unset it for captured graphs"
        )


def _host_int_values(tensor) -> Tuple[int, ...]:
    """Blocking D2H read of a small integer tensor (debug path only)."""
    if is_torch_tensor(tensor):
        return tuple(int(v) for v in tensor.detach().cpu().flatten().tolist())
    import numpy as np

    return tuple(int(v) for v in np.asarray(tensor).flatten().tolist())


def _host_pointer_values(ptrs) -> Tuple[int, ...]:
    """Pointer-table entries as ints, decoding the packed little-endian uint8 JAX form."""
    import cutlass

    from cudnn.datatypes import _convert_to_cutlass_data_type

    if not is_torch_tensor(ptrs) and _convert_to_cutlass_data_type(ptrs.dtype) is cutlass.Uint8:
        import numpy as np

        return tuple(int(v) for v in np.asarray(ptrs).view(np.int64))
    return _host_int_values(ptrs)


def check_offsets_sequence(values: Sequence[int], *, expert_cnt: int, limit: int, mode: str, alignment: int = 256, name: str = "padded_offsets") -> None:
    """Host rules on offset values; pure Python, no device access.

    ``mode="padded"`` (unfused / GLU / dGLU): one cumulative end per expert, non-decreasing,
    every end ``alignment``-aligned, last in ``(0, limit]``. ``mode="wgrad"``: one cumulative
    end per expert, non-decreasing, every group size ``alignment``-aligned, last ``== limit``.
    """
    if mode not in ("padded", "wgrad"):
        raise ValueError(f"unknown offsets mode {mode!r}")
    if len(values) != expert_cnt:
        raise ValueError(f"{name} length mismatch: expected {expert_cnt}, got {len(values)}")
    previous = 0
    for index, value in enumerate(values):
        if value < previous:
            raise ValueError(f"{name} must be a non-decreasing cumulative sum; index {index} is {value} after {previous}")
        if mode == "padded" and value % alignment != 0:
            raise ValueError(f"{name}[{index}] must be {alignment}-aligned, got {value}")
        if mode == "wgrad" and (value - previous) % alignment != 0:
            raise ValueError(f"{name} group {index} must be {alignment}-aligned, got {value - previous}")
        previous = value
    last = values[-1] if values else None
    if mode == "padded" and (last is None or last <= 0 or last > limit):
        raise ValueError(f"{name} last value must be in [1, {limit}], got {last}")
    if mode == "wgrad" and last != limit:
        raise ValueError(f"{name} last value must equal total tokens {limit}, got {last}")


def debug_validate_offsets(offsets, *, expert_cnt: int, limit: int, mode: str, stream, alignment: int = 256, name: str = "padded_offsets") -> None:
    """No-op unless ``CUDNN_FE_GROUPED_GEMM_VALIDATE_DEVICE_VALUES=1``; then a blocking
    host check of the offset values that refuses to run under stream capture (R6)."""
    if not DEBUG_VALIDATE_DEVICE_VALUES:
        return
    _check_not_capturing(stream, f"validating {name} values")
    if is_torch_tensor(offsets):
        with _torch_stream_context(stream, offsets.device):
            values = _host_int_values(offsets)
    else:
        values = _host_int_values(offsets)
    check_offsets_sequence(values, expert_cnt=expert_cnt, limit=limit, mode=mode, alignment=alignment, name=name)


def debug_validate_pointer_values(ptrs, name: str, *, stream) -> None:
    """No-op unless ``CUDNN_FE_GROUPED_GEMM_VALIDATE_DEVICE_VALUES=1``; then a blocking
    host check that every table entry is non-null and 16-byte aligned (R6 under capture)."""
    if not DEBUG_VALIDATE_DEVICE_VALUES:
        return
    _check_not_capturing(stream, f"validating {name} entries")
    if is_torch_tensor(ptrs):
        with _torch_stream_context(stream, ptrs.device):
            values = _host_pointer_values(ptrs)
    else:
        values = _host_pointer_values(ptrs)
    if any(value == 0 or value % 16 != 0 for value in values):
        raise ValueError(f"{name} entries must be non-null and 16-byte aligned")


@contextmanager
def _torch_stream_context(current_stream: Optional[cuda.CUstream], device: torch.device) -> Iterator[None]:
    """Run PyTorch work on the CUDA stream used for the kernel launch.

    torch-only: callers must guard this context so non-torch (e.g. JAX) code paths
    never enter it -- it imports torch and interprets ``device`` as a torch device.
    """
    from cudnn._torch_stream import stream_context

    with stream_context(current_stream, device):
        yield


def allocate_wrapper_workspace(framework: str, nbytes: int, device, current_stream: Optional[cuda.CUstream]):
    """Caller-layer allocation of an APIBase's ``scratch_workspace_bytes()`` (recipe R2).

    The ``*_wrapper_sm100`` functions allocate here, on the launch stream (R1), and pass
    ``workspace=``; an APIBase never allocates one. Returns None when ``nbytes`` is 0.
    """
    if nbytes <= 0:
        return None
    if framework == "torch":
        import torch

        with _torch_stream_context(current_stream, device):
            return torch.empty(nbytes, dtype=torch.uint8, device=device)
    import jax
    import jax.numpy as jnp

    return jax.block_until_ready(jnp.empty((nbytes,), dtype=jnp.uint8, device=device))


class _PendingWorkspace:
    """One eager-framework workspace whose consuming launch may still be in flight."""

    __slots__ = ("buffer", "stream", "event")

    def __init__(self, buffer, stream):
        self.buffer, self.stream, self.event = buffer, stream, None


def _event_after(stream: int):
    """A disabled-timing event recorded on ``stream`` now; None if the driver declines."""
    err, event = cuda.cuEventCreate(cuda.CUevent_flags.CU_EVENT_DISABLE_TIMING)
    if err != cuda.CUresult.CUDA_SUCCESS:
        return None
    (err,) = cuda.cuEventRecord(event, cuda.CUstream(stream))
    if err != cuda.CUresult.CUDA_SUCCESS:
        cuda.cuEventDestroy(event)
        return None
    return event


def _event_done(event) -> bool:
    (err,) = cuda.cuEventQuery(event)
    if err == cuda.CUresult.CUDA_SUCCESS:
        cuda.cuEventDestroy(event)
        return True
    return False


def retain_workspace(api, workspace, current_stream: Optional[cuda.CUstream]) -> None:
    """Guard and keep alive the caller's workspace across the asynchronous launch that reads it (R2 lifetime).

    Call it right after the carve, before the launch. Metadata guard first: the buffer must
    be CUDA memory on the plan's device (an aligned CPU or other-GPU buffer would otherwise
    hand a foreign address to the descriptor workspace). torch: ``record_stream`` on the
    launch stream, so a buffer allocated or last used on another stream is not recycled by
    the caching allocator before the kernel is done. Immutable frameworks (JAX) have no
    record_stream: the API keeps a reference to every such workspace until an event recorded
    on its launch stream -- at the NEXT call, once its launch is enqueued -- has completed.
    Overlapping calls therefore never free each other's scratch; holding only the most recent
    buffer would, and no cap ever drops an entry whose fence has not completed. The fence for a
    call is recorded by the NEXT call on the same API instance, which assumes the earlier launch
    is enqueued by then: an API instance (and the wrappers' memoised instances) serves one host
    thread at a time, like the compile cache it fronts.
    """
    if workspace is None:
        return
    plan_device = getattr(getattr(api, "a_desc", None), "device", None)
    from cudnn.frost.buffers import DeviceView

    if isinstance(workspace, DeviceView):
        # Owns no memory (a Workspace carve, or wrapper_workspace's stream-ordered allocation that is
        # freed on the launch stream after the launch): check the device, retain nothing.
        if plan_device is not None and plan_device.index is not None and workspace._device_id != plan_device.index:
            raise ValueError(f"{type(api).__name__}: workspace must be on the plan's device {plan_device}, got cuda:{workspace._device_id}")
        return
    device = get_device(workspace)
    if getattr(device, "type", None) != "cuda":
        raise ValueError(f"{type(api).__name__}: workspace must be a CUDA buffer, got {device}")
    if plan_device is not None and plan_device.index is not None and device.index is not None and device.index != plan_device.index:
        raise ValueError(f"{type(api).__name__}: workspace must be on the plan's device {plan_device}, got {device}")
    if is_torch_tensor(workspace):
        import torch

        from cudnn._torch_stream import as_torch_stream

        stream = as_torch_stream(int(current_stream), workspace.device) if current_stream is not None else torch.cuda.current_stream(workspace.device)
        workspace.record_stream(stream)
        return
    from cudnn._device import ensure_current_context

    stream = 0 if current_stream is None else int(current_stream)
    ensure_current_context(stream, device.index)
    pending = api.__dict__.setdefault("_live_workspaces", [])
    for entry in pending:  # every earlier entry's launch is enqueued by now: fence it
        if entry.event is None:
            entry.event = _event_after(entry.stream)
    # An entry is released only once its own fence has completed. An entry whose fence could not
    # be recorded (driver declined) is released when a LATER entry on the same stream has completed:
    # stream order puts its launch before that fence. Nothing here blocks the host, and nothing is
    # ever dropped while it may still be in flight -- the list is bounded by what is genuinely pending.
    done_streams = set()
    kept = []
    for entry in reversed(pending):
        if entry.event is not None:
            if _event_done(entry.event):
                done_streams.add(entry.stream)
                continue
        elif entry.stream in done_streams:
            continue
        kept.append(entry)
    kept.reverse()
    pending[:] = kept
    pending.append(_PendingWorkspace(workspace, stream))


def wrapper_workspace(framework: str, nbytes: int, device, current_stream: Optional[cuda.CUstream]):
    """Wrapper-owned scratch whose release follows the consumer on its launch stream.

    JAX allocation readiness does not cover a foreign CUDA consumer. Use a driver
    stream-ordered allocation for that caller layer, paired with a free after launch,
    so overlapping calls have independent lifetimes without plan-owned storage.
    The APIBase execute path still only receives and carves the supplied buffer.
    """
    if framework == "torch" or nbytes <= 0:
        return nullcontext(allocate_wrapper_workspace(framework, nbytes, device, current_stream))
    return _jax_wrapper_workspace(nbytes, device, current_stream)


@contextmanager
def _jax_wrapper_workspace(nbytes: int, device, current_stream: cuda.CUstream):
    from cudnn._device import _ck, ensure_current_context
    from cudnn.frost.buffers import DeviceView

    device_id = int(device.local_hardware_id)
    ensure_current_context(current_stream, device_id)
    ptr = _ck(*cuda.cuMemAllocAsync(nbytes, current_stream))
    try:
        yield DeviceView(int(ptr), (nbytes,), "uint8", device_id)
    finally:
        # Free is enqueued even when validation/launch raises; no host wait or GC
        # finalizer, and no mutable "latest workspace" on the cached plan.
        _ck(*cuda.cuMemFreeAsync(ptr, current_stream))


def wrapper_operand_meta(tensor):
    """Everything a wrapper's derivation reads off an operand, and nothing else.

    Deliberately not the object's identity: CPython recycles a freed tensor's address,
    so an id-keyed memo answers for tensors it never saw. Data pointers are excluded --
    they vary per call and nothing derived depends on them; execute() re-checks them.
    """
    if tensor is None or not hasattr(tensor, "shape"):
        return tensor
    device = get_device(tensor)
    return (get_shape(tensor), get_strides(tensor), tensor.dtype, device.type, device.index)


def block_scaled_sfd_tensors(valid_m, n_out, sf_dtype, sf_vec_size, device):
    """MMA-interleaved (sfd_row, sfd_col) output scale-factor buffers for a (valid_m, n_out) result."""
    import torch

    mma_permute_order = (3, 4, 1, 5, 2, 0)
    mma_shape_row = (1, ceil_div(valid_m, 128), ceil_div(ceil_div(n_out, sf_vec_size), 4), 32, 4, 4)
    mma_shape_col = (1, ceil_div(n_out, 128), ceil_div(ceil_div(valid_m, sf_vec_size), 4), 32, 4, 4)
    sfd_row_tensor = torch.empty(mma_shape_row, dtype=sf_dtype, device=device).permute(mma_permute_order)
    sfd_col_tensor = torch.empty(mma_shape_col, dtype=sf_dtype, device=device).permute(mma_permute_order)
    return sfd_row_tensor, sfd_col_tensor


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
