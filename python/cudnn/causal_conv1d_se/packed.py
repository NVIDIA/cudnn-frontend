# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private packed-convolution compilation and autograd adapter.

Compiled entries own no tensors. Packed extents remain runtime arguments;
known contiguous strides are specialized, and other nonunit strides stay
dynamic. Device/context ownership is handled by compiled_cache.
"""

from functools import cache
from hashlib import sha256
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import Tensor


@cache
def _source_digest():
    here = Path(__file__).resolve().parent
    return sha256(
        b"".join((here / name).read_bytes() for name in ("packed.py", "packed_kernel.py", "persistent_backward.py", "schedule.py", "kernel.py"))
    ).hexdigest()


def _tensor_byte_span(tensor: Tensor) -> Tuple[int, int]:
    """Half-open device byte range addressed by a possibly strided view."""

    begin = tensor.data_ptr()
    if tensor.numel() == 0:
        return begin, begin
    last_offset = sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
    return begin, begin + (last_offset + 1) * tensor.element_size()


def _tensors_share_memory(lhs: Tensor, rhs: Tensor) -> bool:
    if lhs.device != rhs.device:
        return False
    lhs_begin, lhs_end = _tensor_byte_span(lhs)
    rhs_begin, rhs_end = _tensor_byte_span(rhs)
    return lhs_begin < rhs_end and rhs_begin < lhs_end


def _require_dsl():
    from cudnn.frost.buffers import cutedsl_requirement_error, cutedsl_state, cutedsl_too_old

    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        raise ImportError(cutedsl_requirement_error("packed causal_conv1d"))


def _validate(x, weight, cu, bias, initial, silu, return_final):
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("packed input and filter must have shapes [T, H] and [H, K]")
    if x.shape[1] <= 0 or x.shape[1] % 8:
        raise ValueError("packed causal_conv1d requires a positive channel count divisible by eight")
    if weight.shape[0] != x.shape[1] or not 2 <= weight.shape[1] <= 8:
        raise NotImplementedError("packed causal_conv1d requires a [H, K] filter with K=2..8")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError("packed causal_conv1d supports FP16, BF16, and FP32")
    if not x.is_cuda:
        raise ValueError("packed causal_conv1d requires CUDA tensors")
    if not isinstance(silu, bool) or not isinstance(return_final, bool):
        raise TypeError("silu and return_final must be bool")
    if cu.ndim != 1 or cu.numel() < 2 or cu.dtype != torch.int32 or not cu.is_contiguous():
        raise ValueError("cu_seqlens must be contiguous int32 [S+1], with S >= 1")
    if cu.device != x.device:
        raise ValueError("cu_seqlens must be on the input device")
    if x.shape[0] > torch.iinfo(torch.int32).max:
        raise ValueError("packed token count must fit int32 sequence offsets")
    if bias is not None and bias.shape != (x.shape[1],):
        raise ValueError("bias must have shape [H]")
    if initial is not None and initial.shape != (cu.numel() - 1, weight.shape[1] - 1, x.shape[1]):
        raise ValueError("initial history must have shape [S, K-1, H]")
    for name, tensor in (("weight", weight), ("bias", bias), ("initial", initial)):
        if tensor is not None and (tensor.dtype != x.dtype or tensor.device != x.device):
            raise ValueError(f"{name} must match the input dtype and device")
    _require_dsl()
    if torch.cuda.get_device_capability(x.device) not in {(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0)}:
        raise NotImplementedError("packed causal_conv1d requires SM80/86/89/90/100/103/120")


def _backward_schedule(x, weight, bias, initial, dy, dfinal, silu):
    from .schedule import backward_schedule

    with torch.cuda.device(x.device):
        capability = torch.cuda.get_device_capability(x.device)
        multiprocessors = torch.cuda.get_device_properties(x.device).multi_processor_count
    native = (
        all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in (x, dy))
        and (bias is None or (bias.is_contiguous() and bias.data_ptr() % 16 == 0))
        and weight.data_ptr() % 16 == 0
        and x.dtype == weight.dtype == dy.dtype
        and initial is None
        and dfinal is None
    )
    return backward_schedule(x.shape[0], x.shape[1], weight.shape[1], str(x.dtype), silu, native, capability, multiprocessors)


def _compiled(direction, tensors, width, silu):
    """Compile symbolic sequence extents; key only on the traced tensor contract."""
    _require_dsl()
    import cutlass
    from cutlass import cute
    from cutlass.cute.runtime import make_fake_tensor
    from cuda.bindings import driver as cuda
    from cudnn.frost.compiled_cache import compile_cached
    from .packed_kernel import launch_packed_forward, launch_packed_backward, launch_packed_state_backward

    dtype_map = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
        torch.float64: cutlass.Float64,
        torch.int32: cutlass.Int32,
    }
    roles = {
        "forward": ("x", "w", "b", "cu", "initial", "y", "final"),
        "backward": ("x", "w", "b", "cu", "initial", "dy", "dfinal", "dz", "dx", "dw", "db", "dinitial", "partials"),
        "state_backward": ("cu", "dfinal", "dx", "dinitial"),
    }[direction]
    dynamic_leading = {"x", "cu", "initial", "y", "final", "dy", "dfinal", "dz", "dx", "dinitial"}
    fast = False
    weight_vector = False
    if direction == "forward":
        x, w, b, cu, initial, y, final = tensors
        hidden = x.shape[1]
        weight_vector = w.stride(0) == 1 and (w.stride(1) * w.element_size()) % 16 == 0
        fast = (
            x.stride() == (hidden, 1)
            and y.stride() == (hidden, 1)
            and (b is None or b.stride() == (1,))
            and all(t is None or t.data_ptr() % 16 == 0 for t in (x, w, b, initial, y, final))
            and (final is None or final.stride() in (((width - 1) * hidden, hidden, 1), ((width - 1) * hidden, 1, width - 1)))
        )
    elif direction == "backward":
        x, w, b, cu, initial, dy, dfinal, dz, dx, dw, db, di, partials = tensors
        hidden = x.shape[1]
        fast = (
            all(t.stride() == (hidden, 1) for t in (x, dy, dx))
            and (b is None or b.stride() == (1,))
            and all(t is None or t.data_ptr() % 16 == 0 for t in (x, w, b, dy, dx))
        )
    else:
        cu, dfinal, x, di = tensors
    # A complete tile plus either halo must fit before any Int32 arithmetic.
    # Pointer products are widened independently. Extreme extents use Int64;
    # this distinction is part of the compiled signature, not a token count.
    index32 = (
        x.shape[0] <= torch.iinfo(torch.int32).max - 256 and x.shape[1] <= torch.iinfo(torch.int32).max - 1024 and cu.numel() <= torch.iinfo(torch.int32).max
    )
    args, signatures = [], []
    for role, tensor in zip(roles, tensors):
        if tensor is None:
            args.append(None)
            signatures.append(None)
            continue
        shape = list(tensor.shape)
        if role in dynamic_leading:
            shape[0] = cute.sym_int64()
        # Contiguous row pitches depend only on the static trailing dimensions,
        # so specializing them preserves reuse across token/sequence counts.
        # Other layouts retain runtime strides, including column-major dY whose
        # channel stride depends on the packed token extent.
        compact = fast and (
            tensor.is_contiguous() or (role in {"w", "dw"} and tensor.T.is_contiguous()) or (role == "final" and tensor.transpose(1, 2).is_contiguous())
        )
        stride_signature = tuple(tensor.stride()) if compact else tuple(1 if value == 1 else None for value in tensor.stride())
        strides = tuple(cute.sym_int64() if value is None else value for value in stride_signature)
        args.append(make_fake_tensor(dtype_map[tensor.dtype], tuple(shape), strides))
        signatures.append(
            (role, str(tensor.dtype), tuple(None if i == 0 and role in dynamic_leading else s for i, s in enumerate(tensor.shape)), stride_signature)
        )
    key = repr((_source_digest(), direction, width, silu, fast, weight_vector, index32, signatures))
    fn = {"forward": launch_packed_forward, "backward": launch_packed_backward, "state_backward": launch_packed_state_backward}[direction]
    if direction == "backward" and fast and index32 and _backward_schedule(x, w, b, initial, dy, dfinal, silu)[0] == "persistent":
        from .persistent_backward import launch_supported_backward

        fn = launch_supported_backward
    key = repr((key, fn.__name__))
    with torch.cuda.device(tensors[0].device):
        extra = (
            (width,)
            if direction == "state_backward"
            else (width, silu, fast, weight_vector, index32) if direction == "forward" else (width, silu, fast, index32)
        )
        return compile_cached(fn, *args, cuda.CUstream(0), *extra, cache_key=key, symbol="causal_conv1d_packed_" + direction, options="--enable-tvm-ffi")


def _launch(direction, tensors, width, silu):
    from cuda.bindings import driver as cuda

    with torch.cuda.device(tensors[0].device):
        compiled = _compiled(direction, tensors, width, silu)
        stream = cuda.CUstream(torch.cuda.current_stream(tensors[0].device).cuda_stream)
        compiled(*tensors, stream)


@torch.library.custom_op(
    "cudnn::causal_conv1d_packed_se_fwd",
    mutates_args=(),
    device_types="cuda",
    schema="(Tensor x, Tensor weight, Tensor cu, Tensor? bias, Tensor? initial, bool silu, bool return_final, Tensor? final_out_guard=None) -> (Tensor, Tensor?)",
)
def _forward(
    x: Tensor,
    weight: Tensor,
    cu: Tensor,
    bias: Optional[Tensor],
    initial: Optional[Tensor],
    silu: bool,
    return_final: bool,
    final_out_guard: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    _validate(x, weight, cu, bias, initial, silu, return_final)
    if final_out_guard is not None:
        if not return_final:
            raise ValueError("final_states_out requires return_final_states=True")
        for name, tensor in (("x", x), ("weight", weight), ("bias", bias), ("cu_seqlens", cu), ("initial_states", initial)):
            if tensor is not None and _tensors_share_memory(final_out_guard, tensor):
                raise ValueError(f"final_states_out must not share memory with {name}")
    y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    final = None
    if return_final:
        if final_out_guard is not None and final_out_guard.is_contiguous():
            # Match the public caller's contiguous SHK history through a SKH view.
            # The public copy retains its existing autograd and alias contract.
            final = torch.empty((cu.numel() - 1, x.shape[1], weight.shape[1] - 1), device=x.device, dtype=x.dtype).transpose(1, 2)
        else:
            final = torch.empty((cu.numel() - 1, weight.shape[1] - 1, x.shape[1]), device=x.device, dtype=x.dtype)
    _launch("forward", (x, weight, bias, cu, initial, y, final), weight.shape[1], silu)
    return y, final


@_forward.register_fake
def _forward_fake(x, weight, cu, bias, initial, silu, return_final, final_out_guard=None):
    y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    final = None
    if return_final:
        if final_out_guard is not None and final_out_guard.is_contiguous():
            # Match the public caller's contiguous SHK history through a SKH view.
            # The public copy retains its existing autograd and alias contract.
            final = torch.empty((cu.numel() - 1, x.shape[1], weight.shape[1] - 1), device=x.device, dtype=x.dtype).transpose(1, 2)
        else:
            final = torch.empty((cu.numel() - 1, weight.shape[1] - 1, x.shape[1]), device=x.device, dtype=x.dtype)
    return y, final


@torch.library.custom_op(
    "cudnn::causal_conv1d_packed_se_bwd",
    mutates_args=(),
    device_types="cuda",
    schema="(Tensor x, Tensor weight, Tensor cu, Tensor? bias, Tensor? initial, Tensor? dy, Tensor? dfinal, bool silu) -> (Tensor, Tensor, Tensor?, Tensor?)",
)
def _backward(
    x: Tensor, weight: Tensor, cu: Tensor, bias: Optional[Tensor], initial: Optional[Tensor], dy: Optional[Tensor], dfinal: Optional[Tensor], silu: bool
) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    _validate(x, weight, cu, bias, initial, silu, dfinal is not None)
    for name, grad, shape in (("dy", dy, x.shape), ("dfinal", dfinal, (cu.numel() - 1, weight.shape[1] - 1, x.shape[1]))):
        if grad is not None and (grad.shape != shape or grad.dtype != x.dtype or grad.device != x.device):
            raise ValueError(f"{name} must match the corresponding output shape, dtype, and device")
    if dy is None:
        dx = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
        dw = torch.zeros_like(weight)
        db = torch.zeros_like(bias) if bias is not None else None
        di = (torch.empty_like(initial) if dfinal is not None else torch.zeros_like(initial)) if initial is not None else None
        if dfinal is not None:
            _launch("state_backward", (cu, dfinal, dx, di), weight.shape[1], silu)
        return dx, dw, db, di
    dx = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    dw = torch.empty_like(weight)
    db = torch.empty_like(bias) if bias is not None else None
    di = torch.empty_like(initial) if initial is not None else None
    dz = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    partial_dtype = torch.float64 if x.dtype == torch.float32 else torch.float32
    _, splits = _backward_schedule(x, weight, bias, initial, dy, dfinal, silu)
    partials = torch.empty((splits, weight.shape[1] + 1, x.shape[1]), device=x.device, dtype=partial_dtype)
    _launch("backward", (x, weight, bias, cu, initial, dy, dfinal, dz, dx, dw, db, di, partials), weight.shape[1], silu)
    return dx, dw, db, di


@_backward.register_fake
def _backward_fake(x, weight, cu, bias, initial, dy, dfinal, silu):
    return (
        torch.empty(x.shape, device=x.device, dtype=x.dtype),
        torch.empty_like(weight),
        None if bias is None else torch.empty_like(bias),
        None if initial is None else torch.empty_like(initial),
    )


def _setup_context(ctx, inputs, output):
    x, weight, cu, bias, initial, silu, return_final, final_out_guard = inputs
    ctx.save_for_backward(x, weight, cu, bias, initial)
    ctx.silu = silu
    ctx.set_materialize_grads(False)


def _autograd(ctx, dy, dfinal):
    x, weight, cu, bias, initial = ctx.saved_tensors
    dx, dw, db, di = _backward(x, weight, cu, bias, initial, dy, dfinal, ctx.silu)
    return dx, dw, None, db, di, None, None, None


_forward.register_autograd(_autograd, setup_context=_setup_context)


def packed_causal_conv1d(x, weight, cu_seqlens, bias=None, initial_states=None, *, silu=False, return_final_states=False, _final_states_out_guard=None):
    """Run the private token-major adapter; accept either upstream filter orientation."""
    if weight.ndim == 2 and weight.shape[0] != x.shape[1] and weight.shape[1] == x.shape[1]:
        weight = weight.T
    return _forward(x, weight, cu_seqlens, bias, initial_states, silu, return_final_states, _final_states_out_guard)
