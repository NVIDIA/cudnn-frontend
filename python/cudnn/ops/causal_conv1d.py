# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from collections import OrderedDict
from contextvars import ContextVar
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

_TORCH_DTYPE_TO_CUDNN = {
    torch.float32: 0,  # CUDNN_DATA_FLOAT
    torch.float64: 1,  # CUDNN_DATA_DOUBLE
    torch.float16: 2,  # CUDNN_DATA_HALF
    torch.bfloat16: 9,  # CUDNN_DATA_BFLOAT16
}

_ACTIVATION_TO_INT = {
    "identity": 0,  # CUDNN_CAUSAL_CONV1D_ACTIVATION_IDENTITY
    "silu": 1,  # CUDNN_CAUSAL_CONV1D_ACTIVATION_SILU
}


def _dtype_to_int(dtype: torch.dtype) -> int:
    if dtype not in _TORCH_DTYPE_TO_CUDNN:
        raise ValueError(f"Unsupported dtype {dtype}. Supported: float64, float32, float16, bfloat16.")
    return _TORCH_DTYPE_TO_CUDNN[dtype]


def _gradient_dtype(dtype: torch.dtype) -> torch.dtype:
    # Match cuhyena: FP16/BF16 parameter gradients accumulate in FP32,
    # while FP32 and FP64 parameter gradients accumulate in their input type.
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def _activation_to_int(activation: str) -> int:
    if activation not in _ACTIVATION_TO_INT:
        raise ValueError(f"Unsupported activation '{activation}'. Supported: 'identity', 'silu'.")
    return _ACTIVATION_TO_INT[activation]


def _match_causal_conv1d_output_layout(output: Tensor, public_x: Tensor) -> Tensor:
    """Preserve the input's dense memory format across backend selection."""

    if output.stride() == public_x.stride():
        return output
    result = torch.empty_like(public_x)
    result.copy_(output)
    return result


# ---------------------------------------------------------------------------
# Forward primitive
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "cudnn::causal_conv1d_fwd_primitive",
    mutates_args=(),
    device_types="cuda",
)
def _fwd_primitive(x: Tensor, weight: Tensor, bias: Tensor, activation: str) -> Tensor:
    if x.dim() != 3 or weight.dim() != 2 or bias.dim() != 1:
        raise ValueError(f"Expected x(3D), weight(2D), bias(1D); got {x.shape}, {weight.shape}, {bias.shape}")

    if not (x.is_cuda and weight.is_cuda and bias.is_cuda):
        raise ValueError(f"All tensors must be on CUDA: x.device={x.device}, " f"weight.device={weight.device}, bias.device={bias.device}")
    if not (x.device == weight.device == bias.device):
        raise ValueError(f"All tensors must be on the same device: x.device={x.device}, " f"weight.device={weight.device}, bias.device={bias.device}")

    if not (x.dtype == weight.dtype == bias.dtype):
        raise TypeError(f"Dtype mismatch: x.dtype={x.dtype}, weight.dtype={weight.dtype}, " f"bias.dtype={bias.dtype} (all must match)")

    # The backend ABI is contiguous BDT, but the public operation preserves
    # the input's dense memory format.  In particular, model code commonly
    # passes a BDT transpose view backed by contiguous BTD storage.  Keep that
    # observable layout stable whether this generic primitive or the native
    # channel-last route is selected.
    public_x = x
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    batch, dim, seq_len = x.shape
    kernel_size = weight.shape[1]

    if weight.shape[0] != dim:
        raise ValueError(f"Channel mismatch: x has dim={dim} but weight has shape {weight.shape} " f"(expected weight.shape[0]={dim})")

    if bias.shape[0] != dim:
        raise ValueError(f"Bias mismatch: x has dim={dim} but bias has shape {bias.shape} " f"(expected bias.shape[0]={dim})")

    y_contiguous = torch.empty_like(x)

    import cudnn

    cudnn.causal_conv1d_forward(
        torch.cuda.current_stream().cuda_stream,
        x.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        y_contiguous.data_ptr(),
        batch,
        dim,
        seq_len,
        kernel_size,
        _dtype_to_int(x.dtype),
        _activation_to_int(activation),
    )
    return _match_causal_conv1d_output_layout(y_contiguous, public_x)


@torch.library.register_fake("cudnn::causal_conv1d_fwd_primitive")
def _fwd_fake(x: Tensor, weight: Tensor, bias: Tensor, activation: str) -> Tensor:
    return torch.empty_like(x)


# ---------------------------------------------------------------------------
# Backward primitive
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "cudnn::causal_conv1d_bwd_primitive",
    mutates_args=(),
    device_types="cuda",
)
def _bwd_primitive(grad_out: Tensor, x: Tensor, weight: Tensor, bias: Tensor, activation: str) -> List[Tensor]:
    if x.dim() != 3 or weight.dim() != 2 or bias.dim() != 1:
        raise ValueError(f"Expected x(3D), weight(2D), bias(1D); got {x.shape}, {weight.shape}, {bias.shape}")
    if grad_out.shape != x.shape:
        raise ValueError(f"Shape mismatch: dy has shape {grad_out.shape} but x has shape {x.shape} " f"(expected dy.shape == x.shape)")
    if not grad_out.is_cuda:
        raise ValueError(f"grad_out must be on CUDA: grad_out.device={grad_out.device}")
    if grad_out.device != x.device:
        raise ValueError(f"Device mismatch: grad_out.device={grad_out.device}, x.device={x.device}")
    if grad_out.dtype != x.dtype:
        raise ValueError(f"Dtype mismatch: grad_out.dtype={grad_out.dtype}, x.dtype={x.dtype}")

    if not (x.is_cuda and weight.is_cuda and bias.is_cuda):
        raise ValueError(f"All tensors must be on CUDA: x.device={x.device}, " f"weight.device={weight.device}, bias.device={bias.device}")
    if not (x.device == weight.device == bias.device):
        raise ValueError(f"All tensors must be on the same device: x.device={x.device}, " f"weight.device={weight.device}, bias.device={bias.device}")

    if not (x.dtype == weight.dtype == bias.dtype):
        raise TypeError(f"Dtype mismatch: x.dtype={x.dtype}, weight.dtype={weight.dtype}, " f"bias.dtype={bias.dtype} (all must match)")

    batch, dim, seq_len = x.shape

    if weight.shape[0] != dim:
        raise ValueError(f"Channel mismatch: x has dim={dim} but weight has shape {weight.shape} " f"(expected weight.shape[0]={dim})")

    if bias.shape[0] != dim:
        raise ValueError(f"Bias mismatch: x has dim={dim} but bias has shape {bias.shape} " f"(expected bias.shape[0]={dim})")

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    grad_out = grad_out.contiguous()

    kernel_size = weight.shape[1]

    dx = torch.empty_like(x)
    grad_dtype = _gradient_dtype(x.dtype)
    dweight = torch.zeros(weight.shape, device=x.device, dtype=grad_dtype)
    dbias = torch.zeros(bias.shape, device=x.device, dtype=grad_dtype)

    import cudnn

    cudnn.causal_conv1d_backward(
        torch.cuda.current_stream().cuda_stream,
        x.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        grad_out.data_ptr(),
        dx.data_ptr(),
        dweight.data_ptr(),
        dbias.data_ptr(),
        batch,
        dim,
        seq_len,
        kernel_size,
        _dtype_to_int(x.dtype),
        _dtype_to_int(grad_dtype),
        _activation_to_int(activation),
    )
    return [dx, dweight.to(x.dtype), dbias.to(x.dtype)]


@torch.library.register_fake("cudnn::causal_conv1d_bwd_primitive")
def _bwd_fake(grad_out: Tensor, x: Tensor, weight: Tensor, bias: Tensor, activation: str) -> List[Tensor]:
    return [torch.empty_like(x), torch.empty_like(weight), torch.empty_like(bias)]


# ---------------------------------------------------------------------------
# Autograd glue
# ---------------------------------------------------------------------------


def _setup_context(ctx, inputs, output):
    x, weight, bias, activation = inputs
    ctx.save_for_backward(x, weight, bias)
    ctx.activation = activation


@torch.compiler.allow_in_graph
def _autograd_bwd(ctx, grad_out):
    x, weight, bias = ctx.saved_tensors
    dx, dw, db = torch.ops.cudnn.causal_conv1d_bwd_primitive(grad_out, x, weight, bias, ctx.activation)
    return dx, dw, db, None


torch.library.register_autograd(
    "cudnn::causal_conv1d_fwd_primitive",
    _autograd_bwd,
    setup_context=_setup_context,
)


# ---------------------------------------------------------------------------
# Public semantic API and private bulk-backend adapter
# ---------------------------------------------------------------------------


_CAUSAL_CONV1D_TRAINING_CACHE_CAPACITY = 64
_CAUSAL_CONV1D_TRAINING_CACHE = OrderedDict()
_CAUSAL_CONV1D_TRAINING_CACHE_LOCK = threading.Lock()
_CAUSAL_CONV1D_LAST_ROUTE: ContextVar[Optional[str]] = ContextVar(
    "_CAUSAL_CONV1D_LAST_ROUTE",
    default=None,
)


def _get_causal_conv1d_last_route() -> Optional[str]:
    """Return this context's most recent successful eager public route.

    Compiled calls do not touch this diagnostic, so they leave the previous
    eager value unchanged instead of adding ContextVar operations to the graph.
    """

    return _CAUSAL_CONV1D_LAST_ROUTE.get()


def _reset_causal_conv1d_last_route() -> None:
    if not torch.compiler.is_compiling():
        _CAUSAL_CONV1D_LAST_ROUTE.set(None)


def _record_causal_conv1d_route(route: str) -> None:
    if not torch.compiler.is_compiling():
        _CAUSAL_CONV1D_LAST_ROUTE.set(route)


def _causal_conv1d_native_route(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    initial_state: Optional[Tensor] = None,
) -> str:
    differentiable = (x, weight, bias, initial_state)
    if torch.is_grad_enabled() and any(tensor is not None and tensor.requires_grad for tensor in differentiable):
        return "native-autograd"
    return "native-inference"


def _tensor_plan_signature(tensor: Optional[Tensor]):
    if tensor is None:
        return None
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def _causal_conv1d_training_key(
    x_btd: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    cu_seqlens: Optional[Tensor],
    initial_state: Optional[Tensor] = None,
    output_final_state: bool = False,
    deterministic: bool = False,
):
    """Key every plan-time field consumed by the exact-shape training backend.

    The routed backend is fixed to BF16 width-four SiLU. Tensor signatures key
    its remaining specializations: dense versus packed presence, packed N,
    bias and state presence, final-state output, shapes, strides, dtypes, and
    device. Device properties key the architecture-dependent schedule, and the
    torch deterministic-algorithms mode keys the dweight schedule. Runtime
    ``cu_seqlens`` values are intentionally absent because the kernel consumes
    them on device.
    """

    properties = torch.cuda.get_device_properties(x_btd.device)
    return (
        "bf16-width4-silu",
        "packed" if cu_seqlens is not None else "dense",
        _tensor_plan_signature(x_btd),
        _tensor_plan_signature(weight),
        _tensor_plan_signature(bias),
        _tensor_plan_signature(cu_seqlens),
        _tensor_plan_signature(initial_state),
        bool(output_final_state),
        bool(deterministic),
        (properties.major, properties.minor),
        properties.multi_processor_count,
    )


def _compile_causal_conv1d_training_backend(
    x_btd: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    cu_seqlens: Optional[Tensor],
    initial_state: Optional[Tensor] = None,
    output_final_state: bool = False,
    deterministic: bool = False,
):
    from cudnn.causal_conv1d_bulk_sm100.autograd import (
        CausalConv1dBulkAutogradPrototype as _TrainingBackend,
    )

    return _TrainingBackend(
        x_btd,
        weight,
        cu_seqlens,
        sample_bias=bias,
        sample_initial_state=initial_state,
        output_final_state=output_final_state,
        deterministic=deterministic,
    )


def _get_causal_conv1d_training_backend(
    x_btd: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    cu_seqlens: Optional[Tensor],
    initial_state: Optional[Tensor] = None,
    output_final_state: bool = False,
    deterministic: bool = False,
):
    key = _causal_conv1d_training_key(
        x_btd,
        weight,
        bias,
        cu_seqlens,
        initial_state,
        output_final_state,
        deterministic,
    )
    with _CAUSAL_CONV1D_TRAINING_CACHE_LOCK:
        backend = _CAUSAL_CONV1D_TRAINING_CACHE.get(key)
        if backend is not None:
            _CAUSAL_CONV1D_TRAINING_CACHE.move_to_end(key)
            return backend

        backend = _compile_causal_conv1d_training_backend(
            x_btd,
            weight,
            bias,
            cu_seqlens,
            initial_state,
            output_final_state,
            deterministic,
        )
        _CAUSAL_CONV1D_TRAINING_CACHE[key] = backend
        if len(_CAUSAL_CONV1D_TRAINING_CACHE) > _CAUSAL_CONV1D_TRAINING_CACHE_CAPACITY:
            _CAUSAL_CONV1D_TRAINING_CACHE.popitem(last=False)
        return backend


def _run_causal_conv1d_bulk_backend(
    x_btd: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    cu_seqlens: Optional[Tensor],
    *,
    initial_state: Optional[Tensor] = None,
    output_final_state: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Run the current backend without exporting its lifecycle or result type."""

    if _causal_conv1d_native_route(x_btd, weight, bias, initial_state) == "native-autograd":
        backend = _get_causal_conv1d_training_backend(
            x_btd,
            weight,
            bias,
            cu_seqlens,
            initial_state,
            output_final_state,
            torch.are_deterministic_algorithms_enabled(),
        )
        result = backend(
            x_btd,
            weight,
            cu_seqlens,
            bias=bias,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
        if output_final_state:
            return result["output_tensor"], result["final_state_tensor"]
        return result

    from cudnn.causal_conv1d_bulk_sm100.api import (
        causal_conv1d_bulk_fwd_wrapper_sm100 as _forward_backend,
    )

    result = _forward_backend(
        x_btd,
        weight,
        cu_seqlens_tensor=cu_seqlens,
        initial_state_tensor=initial_state,
        output_final_state=output_final_state,
        bias_tensor=bias,
    )
    if output_final_state:
        return result["output_tensor"], result["final_state_tensor"]
    return result["output_tensor"]


def _can_route_causal_conv1d_bulk(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    cu_seqlens: Optional[Tensor],
    activation: str,
) -> bool:
    """Whether the current BF16 width-four backend is a transparent route."""

    if torch.compiler.is_compiling() or activation != "silu":
        return False
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return False
    if bias is not None and not isinstance(bias, torch.Tensor):
        return False
    if cu_seqlens is not None and not isinstance(cu_seqlens, torch.Tensor):
        return False
    if x.ndim != 3 or weight.ndim != 2:
        return False

    batch, channels, tokens = x.shape
    if batch <= 0 or channels <= 0 or tokens <= 0 or tuple(weight.shape) != (channels, 4):
        return False
    if bias is not None and tuple(bias.shape) != (channels,):
        return False
    if cu_seqlens is not None:
        if batch != 1 or cu_seqlens.ndim != 1 or cu_seqlens.shape[0] < 2:
            return False
        if cu_seqlens.dtype != torch.int32 or not cu_seqlens.is_contiguous():
            return False
    if x.dtype != torch.bfloat16 or weight.dtype not in (
        torch.bfloat16,
        torch.float32,
    ):
        return False
    # The FP32-weight epilogue is only tested without bias; other mixed-dtype
    # combinations stay off the route until their epilogue contract is tested.
    if weight.dtype == torch.float32 and bias is not None:
        return False
    if bias is not None and bias.dtype != torch.bfloat16:
        return False
    tensors = (x, weight) if bias is None else (x, weight, bias)
    if not x.is_cuda or any(tensor.device != x.device for tensor in tensors[1:]):
        return False
    if cu_seqlens is not None and cu_seqlens.device != x.device:
        return False

    x_btd = x.transpose(1, 2)
    if not x_btd.is_contiguous() or x_btd.data_ptr() != x.data_ptr():
        return False
    if not weight.is_contiguous() or (bias is not None and not bias.is_contiguous()):
        return False
    if any(tensor.data_ptr() % 16 for tensor in tensors):
        return False

    try:
        from cudnn._causal_conv1d_arch import is_functional_arch
        from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

        installed, version = cutedsl_state()
        capability = torch.cuda.get_device_capability(x.device)
    except (AttributeError, ImportError, OSError, RuntimeError):
        return False
    return installed and not cutedsl_too_old(version) and is_functional_arch(capability)


def _normalize_causal_conv1d_activation(activation: Optional[str]) -> str:
    if activation is None or activation == "identity":
        return "identity"
    if activation in ("silu", "swish"):
        return "silu"
    raise NotImplementedError("activation must be None, 'identity', 'silu', or 'swish'")


def _validate_causal_conv1d_sequence_contract(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    seq_idx: Optional[Tensor],
    cu_seqlens: Optional[Tensor],
    initial_states: Optional[Tensor],
    return_final_states: bool,
    final_states_out: Optional[Tensor],
) -> None:
    """Validate sequence and mathematical-state semantics only.

    Physical layout requirements belong to a private backend route.  Keeping
    them out of this validator prevents the current kernel schedule from
    becoming part of the public operation contract.
    """

    if not isinstance(return_final_states, bool):
        raise TypeError(f"return_final_states must be bool, got {type(return_final_states).__name__}")
    if final_states_out is not None and not return_final_states:
        raise ValueError("final_states_out requires return_final_states=True")
    if seq_idx is not None and cu_seqlens is not None:
        raise ValueError("seq_idx and cu_seqlens are mutually exclusive")
    if seq_idx is not None and return_final_states:
        raise ValueError("seq_idx and return_final_states are mutually exclusive")
    for name, tensor in (("x", x), ("weight", weight)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    for name, tensor in (
        ("bias", bias),
        ("seq_idx", seq_idx),
        ("cu_seqlens", cu_seqlens),
        ("initial_states", initial_states),
        ("final_states_out", final_states_out),
    ):
        if tensor is not None and not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor or None, got {type(tensor).__name__}")

    if x.ndim != 3:
        raise ValueError(f"x must have shape [B, D, T], got {tuple(x.shape)}")
    if weight.ndim != 2:
        raise ValueError(f"weight must have shape [D, W], got {tuple(weight.shape)}")
    batch, channels, tokens = x.shape
    if weight.shape[0] != channels:
        raise ValueError(f"weight must have shape [D, W] with D={channels}, got {tuple(weight.shape)}")
    width = weight.shape[1]
    if width < 2:
        raise ValueError(f"weight width must be at least two, got W={width}")
    if bias is not None and tuple(bias.shape) != (channels,):
        raise ValueError(f"bias must have shape {(channels,)}, got {tuple(bias.shape)}")
    if seq_idx is not None and tuple(seq_idx.shape) != (batch, tokens):
        raise ValueError(f"seq_idx must have shape {(batch, tokens)}, got {tuple(seq_idx.shape)}")
    if cu_seqlens is not None:
        if batch != 1:
            raise ValueError(f"packed x must have B=1, got B={batch}")
        if cu_seqlens.ndim != 1 or cu_seqlens.shape[0] < 2:
            raise ValueError("cu_seqlens must have shape [N + 1] with N >= 1")
        state_batch = cu_seqlens.shape[0] - 1
    else:
        state_batch = batch

    state_shape = (state_batch, channels, width - 1)
    for name, state in (("initial_states", initial_states), ("final_states_out", final_states_out)):
        if state is not None and tuple(state.shape) != state_shape:
            raise ValueError(f"{name} must have shape {state_shape}, got {tuple(state.shape)}")
        if state is not None and state.dtype != x.dtype:
            raise TypeError(f"{name} dtype must match x dtype {x.dtype}, got {state.dtype}")
        if state is not None and state.device != x.device:
            raise ValueError(f"{name} device must match x device {x.device}, got {state.device}")
    if final_states_out is not None:
        # The final state is written after the forward, and autograd saves the
        # inputs for backward, so an aliased output would corrupt them.
        for name, tensor in (
            ("x", x),
            ("weight", weight),
            ("bias", bias),
            ("cu_seqlens", cu_seqlens),
            ("initial_states", initial_states),
        ):
            if tensor is not None and _tensors_share_memory(final_states_out, tensor):
                raise ValueError(f"final_states_out must not share memory with {name}")


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


def _to_causal_conv1d_full_width_state(initial_states: Optional[Tensor]) -> Optional[Tensor]:
    """Translate public ``W - 1`` history to the backend's private ``W`` cache.

    The backend shifts before filtering the current token.  Prepending one
    unobservable lane therefore makes ``[0, h0, h1, h2]`` become
    ``[h0, h1, h2, x0]`` for the first width-four convolution window.
    """

    if initial_states is None:
        return None
    padding = torch.zeros_like(initial_states[..., :1])
    return torch.cat((padding, initial_states), dim=-1)


def _from_causal_conv1d_full_width_state(
    final_state: Tensor,
    final_states_out: Optional[Tensor],
) -> Tensor:
    """Copy the private ``W`` cache into independent public ``W - 1`` storage."""

    mathematical_state = final_state[..., 1:]
    if final_states_out is None:
        # Match the channel-last state allocation used by the consumed API and
        # avoid returning a view whose stride/storage exposes the private lane.
        batch, channels, history = mathematical_state.shape
        final_states_out = torch.empty(
            (batch, history, channels),
            dtype=mathematical_state.dtype,
            device=mathematical_state.device,
        ).transpose(1, 2)
    final_states_out.copy_(mathematical_state)
    return final_states_out


def _run_causal_conv1d_sequence_backend(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    activation: str,
    seq_idx: Optional[Tensor],
    cu_seqlens: Optional[Tensor],
    initial_states: Optional[Tensor],
    return_final_states: bool,
    final_states_out: Optional[Tensor],
) -> Tuple[Tensor, Optional[Tensor]]:
    """Adapt mathematical state to the private width-four backend ABI."""

    if seq_idx is not None:
        raise NotImplementedError("the current backend does not yet implement seq_idx")
    if not _can_route_causal_conv1d_bulk(x, weight, bias, cu_seqlens, activation):
        raise NotImplementedError("the current backend cannot execute this stateful causal_conv1d call")

    full_width_initial_state = _to_causal_conv1d_full_width_state(initial_states)
    result = _run_causal_conv1d_bulk_backend(
        x.transpose(1, 2),
        weight,
        bias,
        cu_seqlens,
        initial_state=full_width_initial_state,
        output_final_state=return_final_states,
    )
    if return_final_states:
        output_btd, full_width_final_state = result
        public_final_state = _from_causal_conv1d_full_width_state(
            full_width_final_state,
            final_states_out,
        )
    else:
        output_btd = result
        public_final_state = None
    output = _match_causal_conv1d_output_layout(output_btd.transpose(1, 2), x)
    return output, public_final_state


def causal_conv1d(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    activation: Optional[str] = None,
    *,
    seq_idx: Optional[Tensor] = None,
    cu_seqlens: Optional[Tensor] = None,
    initial_states: Optional[Tensor] = None,
    return_final_states: bool = False,
    final_states_out: Optional[Tensor] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    r"""Apply depthwise causal convolution to ``x[B, D, T]``.

    The semantic contract follows the commonly consumed ``causal-conv1d``
    interface: packed offsets prevent cross-sequence filtering, and optional
    initial/final histories contain exactly ``W - 1`` samples. A channel-last
    physical allocation can be passed as its zero-copy ``[B, D, T]`` transpose
    view. ``seq_idx`` is a reserved compatibility keyword and currently must be
    ``None``; it and ``cu_seqlens`` are mutually exclusive.

    Kernel compilation, schedule selection, workspace allocation, result
    wrappers, architecture class names, and CUDA stream handles are private.
    The result is a Tensor, or ``(output, final_states)`` when requested.

    The current optimized route implements dense and ``cu_seqlens``-packed
    BF16 width-four SiLU with forward and backward. Dense bias-free calls also
    accept an FP32 depthwise filter with BF16 activations. Unsupported optional
    modes fail explicitly without changing the public state shape to match a
    backend cache convention.

    ``final_states_out`` must not share memory with any input. The backward
    honors ``torch.use_deterministic_algorithms``: bias-free calls switch to a
    deterministic dweight schedule, while calls with ``bias`` raise (or warn
    under ``warn_only=True``) because dbias accumulates with FP32 atomics.
    """

    _reset_causal_conv1d_last_route()
    normalized_activation = _normalize_causal_conv1d_activation(activation)
    if seq_idx is not None and cu_seqlens is not None:
        raise ValueError("seq_idx and cu_seqlens are mutually exclusive")
    state_or_seq_idx_call = seq_idx is not None or initial_states is not None or return_final_states or final_states_out is not None
    if state_or_seq_idx_call:
        _validate_causal_conv1d_sequence_contract(
            x,
            weight,
            bias,
            seq_idx,
            cu_seqlens,
            initial_states,
            return_final_states,
            final_states_out,
        )
        output, final_states = _run_causal_conv1d_sequence_backend(
            x,
            weight,
            bias,
            normalized_activation,
            seq_idx,
            cu_seqlens,
            initial_states,
            return_final_states,
            final_states_out,
        )
        if return_final_states:
            assert final_states is not None
            _record_causal_conv1d_route(_causal_conv1d_native_route(x, weight, bias, initial_states))
            return output, final_states
        _record_causal_conv1d_route(_causal_conv1d_native_route(x, weight, bias, initial_states))
        return output

    if cu_seqlens is not None:
        _validate_causal_conv1d_sequence_contract(
            x,
            weight,
            bias,
            None,
            cu_seqlens,
            None,
            False,
            None,
        )
        if not _can_route_causal_conv1d_bulk(x, weight, bias, cu_seqlens, normalized_activation):
            raise NotImplementedError("the current backend cannot execute this cu_seqlens-packed causal_conv1d call")
        x_btd = x.transpose(1, 2)
        output = _run_causal_conv1d_bulk_backend(x_btd, weight, bias, cu_seqlens).transpose(1, 2)
        _record_causal_conv1d_route(_causal_conv1d_native_route(x, weight, bias))
        return output

    if _can_route_causal_conv1d_bulk(x, weight, bias, None, normalized_activation):
        output_btd = _run_causal_conv1d_bulk_backend(x.transpose(1, 2), weight, bias, None)
        output = output_btd.transpose(1, 2)
        _record_causal_conv1d_route(_causal_conv1d_native_route(x, weight, bias))
        return output

    if x.dtype == torch.bfloat16 and weight.dtype == torch.float32:
        raise NotImplementedError("BF16 activation with FP32 weight requires the native channel-last " "width-four SiLU route")
    if bias is None:
        bias = torch.zeros(weight.shape[0], device=x.device, dtype=x.dtype)
    output = torch.ops.cudnn.causal_conv1d_fwd_primitive(x, weight, bias, normalized_activation)
    _record_causal_conv1d_route("generic-cudnn")
    return output


# ===========================================================================
# NWH variant — x is (batch, seq_len, dim)
# ===========================================================================


# ---------------------------------------------------------------------------
# NWH Forward primitive
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "cudnn::causal_conv1d_nwh_fwd_primitive",
    mutates_args=(),
    device_types="cuda",
)
def _nwh_fwd_primitive(x: Tensor, weight: Tensor, bias: Tensor, activation: str) -> Tensor:
    if x.dim() != 3 or weight.dim() != 2 or bias.dim() != 1:
        raise ValueError(f"Expected x(3D), weight(2D), bias(1D); got {x.shape}, {weight.shape}, {bias.shape}")

    if not (x.is_cuda and weight.is_cuda and bias.is_cuda):
        raise ValueError(f"All tensors must be on CUDA: x.device={x.device}, " f"weight.device={weight.device}, bias.device={bias.device}")
    if not (x.device == weight.device == bias.device):
        raise ValueError(f"All tensors must be on the same device: x.device={x.device}, " f"weight.device={weight.device}, bias.device={bias.device}")

    if not (x.dtype == weight.dtype == bias.dtype):
        raise TypeError(f"Dtype mismatch: x.dtype={x.dtype}, weight.dtype={weight.dtype}, " f"bias.dtype={bias.dtype} (all must match)")

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    batch, seq_len, dim = x.shape
    kernel_size = weight.shape[0]

    if weight.shape[1] != dim:
        raise ValueError(f"Channel mismatch: x has dim={dim} but weight has shape {weight.shape} " f"(expected weight.shape[1]={dim})")

    if bias.shape[0] != dim:
        raise ValueError(f"Bias mismatch: x has dim={dim} but bias has shape {bias.shape} " f"(expected bias.shape[0]={dim})")

    y = torch.empty_like(x)

    import cudnn

    cudnn.causal_conv1d_nwh_forward(
        torch.cuda.current_stream().cuda_stream,
        x.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        y.data_ptr(),
        batch,
        dim,
        seq_len,
        kernel_size,
        _dtype_to_int(x.dtype),
        _activation_to_int(activation),
    )
    return y


@torch.library.register_fake("cudnn::causal_conv1d_nwh_fwd_primitive")
def _nwh_fwd_fake(x: Tensor, weight: Tensor, bias: Tensor, activation: str) -> Tensor:
    return torch.empty_like(x)


# ---------------------------------------------------------------------------
# NWH Backward primitive
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "cudnn::causal_conv1d_nwh_bwd_primitive",
    mutates_args=(),
    device_types="cuda",
)
def _nwh_bwd_primitive(grad_out: Tensor, x: Tensor, weight: Tensor, bias: Tensor, activation: str) -> List[Tensor]:
    if x.dim() != 3 or weight.dim() != 2 or bias.dim() != 1:
        raise ValueError(f"Expected x(3D), weight(2D), bias(1D); got {x.shape}, {weight.shape}, {bias.shape}")
    if grad_out.shape != x.shape:
        raise ValueError(f"Shape mismatch: dy has shape {grad_out.shape} but x has shape {x.shape} " f"(expected dy.shape == x.shape)")
    if not grad_out.is_cuda:
        raise ValueError(f"grad_out must be on CUDA: grad_out.device={grad_out.device}")
    if grad_out.device != x.device:
        raise ValueError(f"Device mismatch: grad_out.device={grad_out.device}, x.device={x.device}")
    if grad_out.dtype != x.dtype:
        raise ValueError(f"Dtype mismatch: grad_out.dtype={grad_out.dtype}, x.dtype={x.dtype}")

    if not (x.is_cuda and weight.is_cuda and bias.is_cuda):
        raise ValueError(f"All tensors must be on CUDA: x.device={x.device}, " f"weight.device={weight.device}, bias.device={bias.device}")
    if not (x.device == weight.device == bias.device):
        raise ValueError(f"All tensors must be on the same device: x.device={x.device}, " f"weight.device={weight.device}, bias.device={bias.device}")

    if not (x.dtype == weight.dtype == bias.dtype):
        raise TypeError(f"Dtype mismatch: x.dtype={x.dtype}, weight.dtype={weight.dtype}, " f"bias.dtype={bias.dtype} (all must match)")

    batch, seq_len, dim = x.shape

    if weight.shape[1] != dim:
        raise ValueError(f"Channel mismatch: x has dim={dim} but weight has shape {weight.shape} " f"(expected weight.shape[1]={dim})")

    if bias.shape[0] != dim:
        raise ValueError(f"Bias mismatch: x has dim={dim} but bias has shape {bias.shape} " f"(expected bias.shape[0]={dim})")

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    grad_out = grad_out.contiguous()

    kernel_size = weight.shape[0]

    dx = torch.empty_like(x)
    grad_dtype = _gradient_dtype(x.dtype)
    dweight = torch.zeros(weight.shape, device=x.device, dtype=grad_dtype)
    dbias = torch.zeros(bias.shape, device=x.device, dtype=grad_dtype)

    import cudnn

    cudnn.causal_conv1d_nwh_backward(
        torch.cuda.current_stream().cuda_stream,
        x.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        grad_out.data_ptr(),
        dx.data_ptr(),
        dweight.data_ptr(),
        dbias.data_ptr(),
        batch,
        dim,
        seq_len,
        kernel_size,
        _dtype_to_int(x.dtype),
        _dtype_to_int(grad_dtype),
        _activation_to_int(activation),
    )
    return [dx, dweight.to(x.dtype), dbias.to(x.dtype)]


@torch.library.register_fake("cudnn::causal_conv1d_nwh_bwd_primitive")
def _nwh_bwd_fake(grad_out: Tensor, x: Tensor, weight: Tensor, bias: Tensor, activation: str) -> List[Tensor]:
    return [torch.empty_like(x), torch.empty_like(weight), torch.empty_like(bias)]


# ---------------------------------------------------------------------------
# NWH Autograd glue
# ---------------------------------------------------------------------------


def _nwh_setup_context(ctx, inputs, output):
    x, weight, bias, activation = inputs
    ctx.save_for_backward(x, weight, bias)
    ctx.activation = activation


@torch.compiler.allow_in_graph
def _nwh_autograd_bwd(ctx, grad_out):
    x, weight, bias = ctx.saved_tensors
    dx, dw, db = torch.ops.cudnn.causal_conv1d_nwh_bwd_primitive(grad_out, x, weight, bias, ctx.activation)
    return dx, dw, db, None


torch.library.register_autograd(
    "cudnn::causal_conv1d_nwh_fwd_primitive",
    _nwh_autograd_bwd,
    setup_context=_nwh_setup_context,
)


# ---------------------------------------------------------------------------
# NWH Public API
# ---------------------------------------------------------------------------


def causal_conv1d_nwh(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    activation: str = "identity",
) -> Tensor:
    r"""Depthwise causal 1D convolution (NWH layout).

    Same operation as :func:`causal_conv1d` but with NWH tensor layout::

        y = activation(conv1d_causal(x, weight) + bias)

    Supports ``torch.compile`` and ``torch.autograd`` — backward is handled
    automatically when inputs require gradients.

    Args:
        x (torch.Tensor): Input tensor of shape ``(batch, seq_len, dim)``.
            Must be BF16, FP16, FP32, or FP64.
        weight (torch.Tensor): Filter tensor of shape ``(kernel_size, dim)``.
            ``kernel_size`` must be between 2 and 128, inclusive.
        bias (torch.Tensor | None): Optional bias of shape ``(dim,)``.
        activation (str): ``"identity"`` (default) or ``"silu"``.

    Returns:
        torch.Tensor: Output of shape ``(batch, seq_len, dim)``.
    """
    if activation not in _ACTIVATION_TO_INT:
        raise ValueError(f"Unsupported activation '{activation}'. Supported: 'identity', 'silu'.")
    if bias is None:
        bias = torch.zeros(weight.shape[1], device=x.device, dtype=x.dtype)
    return torch.ops.cudnn.causal_conv1d_nwh_fwd_primitive(x, weight, bias, activation)


# ===========================================================================
# Back-to-back (B2B) causal conv1d — fused projection + gating + mixer
# ===========================================================================


# ---------------------------------------------------------------------------
# B2B Forward primitive
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "cudnn::b2b_causal_conv1d_fwd_primitive",
    mutates_args=(),
    device_types="cuda",
)
def _b2b_fwd_primitive(x: Tensor, weights_proj: Tensor, weights_mixer: Tensor, skip_bias: Tensor) -> Tuple[Tensor, Tensor]:
    if x.dim() != 3:
        raise ValueError(f"Expected x(3D) with shape (batch, 3*dim, seq_len); got {x.shape}")
    batch = x.shape[0]
    if x.shape[1] % 3 != 0:
        raise ValueError(f"Expected x.shape[1] divisible by 3; got {x.shape[1]}")
    dim = x.shape[1] // 3
    seq_len = x.shape[2]

    if weights_proj.dim() != 2 or weights_mixer.dim() != 2 or skip_bias.dim() != 1:
        raise ValueError(
            f"Expected weights_proj(2D: 3*dim,K_proj), weights_mixer(2D: dim,K_mixer), skip_bias(1D: dim); "
            f"got {weights_proj.shape}, {weights_mixer.shape}, {skip_bias.shape}"
        )

    if not (x.is_cuda and weights_proj.is_cuda and weights_mixer.is_cuda and skip_bias.is_cuda):
        raise ValueError("All tensors must be on CUDA")
    if not (x.device == weights_proj.device == weights_mixer.device == skip_bias.device):
        raise ValueError("All tensors must be on the same device")

    if not (x.dtype == weights_proj.dtype == weights_mixer.dtype == skip_bias.dtype):
        raise TypeError(
            f"Dtype mismatch: x.dtype={x.dtype}, weights_proj.dtype={weights_proj.dtype}, "
            f"weights_mixer.dtype={weights_mixer.dtype}, skip_bias.dtype={skip_bias.dtype}"
        )

    if weights_proj.shape[0] != 3 * dim:
        raise ValueError(f"Channel mismatch: x has 3*dim={3*dim} but weights_proj has shape {weights_proj.shape}")
    if weights_mixer.shape[0] != dim:
        raise ValueError(f"Channel mismatch: x has dim={dim} but weights_mixer has shape {weights_mixer.shape}")
    if skip_bias.shape[0] != dim:
        raise ValueError(f"Channel mismatch: x has dim={dim} but skip_bias has shape {skip_bias.shape}")

    x = x.contiguous()
    weights_proj = weights_proj.contiguous()
    weights_mixer = weights_mixer.contiguous()
    skip_bias = skip_bias.contiguous()

    kernel_size_proj = weights_proj.shape[1]
    kernel_size_mixer = weights_mixer.shape[1]

    y = torch.empty(batch, dim, seq_len, device=x.device, dtype=x.dtype)
    y_gated = torch.empty(batch, dim, seq_len, device=x.device, dtype=x.dtype)

    import cudnn

    cudnn.b2b_causal_conv1d_forward(
        torch.cuda.current_stream().cuda_stream,
        x.data_ptr(),
        weights_proj.data_ptr(),
        weights_mixer.data_ptr(),
        skip_bias.data_ptr(),
        y.data_ptr(),
        y_gated.data_ptr(),
        batch,
        dim,
        seq_len,
        kernel_size_proj,
        kernel_size_mixer,
        _dtype_to_int(x.dtype),
    )
    return y, y_gated


@torch.library.register_fake("cudnn::b2b_causal_conv1d_fwd_primitive")
def _b2b_fwd_fake(x: Tensor, weights_proj: Tensor, weights_mixer: Tensor, skip_bias: Tensor) -> Tuple[Tensor, Tensor]:
    batch = x.shape[0]
    dim = x.shape[1] // 3
    seq_len = x.shape[2]
    y = torch.empty(batch, dim, seq_len, device=x.device, dtype=x.dtype)
    y_gated = torch.empty(batch, dim, seq_len, device=x.device, dtype=x.dtype)
    return y, y_gated


# ---------------------------------------------------------------------------
# B2B Backward primitive
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "cudnn::b2b_causal_conv1d_bwd_primitive",
    mutates_args=(),
    device_types="cuda",
)
def _b2b_bwd_primitive(
    grad_y: Tensor,
    x: Tensor,
    weights_proj: Tensor,
    weights_mixer: Tensor,
    skip_bias: Tensor,
    y: Tensor,
) -> List[Tensor]:
    if x.dim() != 3:
        raise ValueError(f"Expected x(3D) with shape (batch, 3*dim, seq_len); got {x.shape}")
    if x.shape[1] % 3 != 0:
        raise ValueError(f"Expected x.shape[1] divisible by 3; got {x.shape[1]}")
    batch = x.shape[0]
    dim = x.shape[1] // 3
    seq_len = x.shape[2]

    if weights_proj.dim() != 2 or weights_mixer.dim() != 2 or skip_bias.dim() != 1:
        raise ValueError(
            f"Expected weights_proj(2D: 3*dim,K_proj), weights_mixer(2D: dim,K_mixer), skip_bias(1D: dim); "
            f"got {weights_proj.shape}, {weights_mixer.shape}, {skip_bias.shape}"
        )
    if y.shape != (batch, dim, seq_len):
        raise ValueError(f"Shape mismatch: expected y shape {(batch, dim, seq_len)}; got {y.shape}")
    if grad_y.shape != y.shape:
        raise ValueError(f"Shape mismatch: grad_y {grad_y.shape} vs y {y.shape}")
    if not (x.is_cuda and weights_proj.is_cuda and weights_mixer.is_cuda and skip_bias.is_cuda and y.is_cuda and grad_y.is_cuda):
        raise ValueError("All tensors must be on CUDA")
    if not (x.device == weights_proj.device == weights_mixer.device == skip_bias.device == y.device == grad_y.device):
        raise ValueError("All tensors must be on the same device")

    if not (x.dtype == weights_proj.dtype == weights_mixer.dtype == skip_bias.dtype == y.dtype == grad_y.dtype):
        raise TypeError(
            f"Dtype mismatch: x.dtype={x.dtype}, weights_proj.dtype={weights_proj.dtype}, "
            f"weights_mixer.dtype={weights_mixer.dtype}, skip_bias.dtype={skip_bias.dtype}, "
            f"y.dtype={y.dtype}, grad_y.dtype={grad_y.dtype}"
        )

    if weights_proj.shape[0] != 3 * dim:
        raise ValueError(f"Channel mismatch: x has 3*dim={3*dim} but weights_proj has shape {weights_proj.shape}")
    if weights_mixer.shape[0] != dim:
        raise ValueError(f"Channel mismatch: x has dim={dim} but weights_mixer has shape {weights_mixer.shape}")
    if skip_bias.shape[0] != dim:
        raise ValueError(f"Channel mismatch: x has dim={dim} but skip_bias has shape {skip_bias.shape}")

    x = x.contiguous()
    weights_proj = weights_proj.contiguous()
    weights_mixer = weights_mixer.contiguous()
    skip_bias = skip_bias.contiguous()
    y = y.contiguous()
    grad_y = grad_y.contiguous()

    kernel_size_proj = weights_proj.shape[1]
    kernel_size_mixer = weights_mixer.shape[1]

    dx = torch.empty_like(x)
    grad_dtype = _gradient_dtype(x.dtype)
    dweights_proj = torch.zeros(weights_proj.shape, device=x.device, dtype=grad_dtype)
    dweights_mixer = torch.zeros(weights_mixer.shape, device=x.device, dtype=grad_dtype)
    dskip_bias = torch.zeros(skip_bias.shape, device=x.device, dtype=grad_dtype)

    import cudnn

    cudnn.b2b_causal_conv1d_backward(
        torch.cuda.current_stream().cuda_stream,
        x.data_ptr(),
        weights_proj.data_ptr(),
        weights_mixer.data_ptr(),
        skip_bias.data_ptr(),
        y.data_ptr(),
        grad_y.data_ptr(),
        dx.data_ptr(),
        dweights_proj.data_ptr(),
        dweights_mixer.data_ptr(),
        dskip_bias.data_ptr(),
        batch,
        dim,
        seq_len,
        kernel_size_proj,
        kernel_size_mixer,
        _dtype_to_int(x.dtype),
        _dtype_to_int(grad_dtype),
    )
    return [
        dx,
        dweights_proj.to(x.dtype),
        dweights_mixer.to(x.dtype),
        dskip_bias.to(x.dtype),
    ]


@torch.library.register_fake("cudnn::b2b_causal_conv1d_bwd_primitive")
def _b2b_bwd_fake(
    grad_y: Tensor,
    x: Tensor,
    weights_proj: Tensor,
    weights_mixer: Tensor,
    skip_bias: Tensor,
    y: Tensor,
) -> List[Tensor]:
    return [
        torch.empty_like(x),
        torch.empty_like(weights_proj),
        torch.empty_like(weights_mixer),
        torch.empty_like(skip_bias),
    ]


# ---------------------------------------------------------------------------
# B2B Autograd glue
# ---------------------------------------------------------------------------


def _b2b_setup_context(ctx, inputs, output):
    x, weights_proj, weights_mixer, skip_bias = inputs
    y = output[0]
    ctx.save_for_backward(x, weights_proj, weights_mixer, skip_bias, y)


@torch.compiler.allow_in_graph
def _b2b_autograd_bwd(ctx, grad_y, grad_y_gated):
    # PyTorch may pass a ZeroTensor for the discarded intermediate output.
    if grad_y is not None and not torch._is_zerotensor(grad_y) and torch.count_nonzero(grad_y).item() != 0:
        raise RuntimeError("Gradient for the intermediate B2B output y is not supported; use cudnn.ops.b2b_causal_conv1d")
    x, weights_proj, weights_mixer, skip_bias, y = ctx.saved_tensors
    dx, dwp, dwm, dsb = torch.ops.cudnn.b2b_causal_conv1d_bwd_primitive(grad_y_gated, x, weights_proj, weights_mixer, skip_bias, y)
    return dx, dwp, dwm, dsb


torch.library.register_autograd(
    "cudnn::b2b_causal_conv1d_fwd_primitive",
    _b2b_autograd_bwd,
    setup_context=_b2b_setup_context,
)


# ---------------------------------------------------------------------------
# B2B Public API
# ---------------------------------------------------------------------------


def b2b_causal_conv1d(
    x: Tensor,
    weights_proj: Tensor,
    weights_mixer: Tensor,
    skip_bias: Tensor,
) -> Tensor:
    r"""Fused back-to-back causal conv1d: projection conv, gating, mixer conv, and post-gating.

    Computes the fused Hyena-SE block::

        proj       = causal_conv1d(x, weights_proj)            # (batch, 3*dim, seq_len)
        gated      = proj[:, 1::3, :] * proj[:, 2::3, :]       # Q * K gate
        y          = causal_conv1d(gated, weights_mixer) + skip_bias[:, None] * gated
        y_gated    = y * proj[:, 0::3, :]                      # final * V

    Supports ``torch.compile`` and ``torch.autograd`` — backward is handled
    automatically when inputs require gradients.

    Args:
        x (torch.Tensor): Input tensor of shape ``(batch, 3*dim, seq_len)``.
            Must be BF16, FP16, FP32, or FP64.
        weights_proj (torch.Tensor): Projection filter ``(3*dim, kernel_size_proj)``.
            ``kernel_size_proj`` must be between 2 and 32, inclusive.
        weights_mixer (torch.Tensor): Mixer filter ``(dim, kernel_size_mixer)``.
            ``kernel_size_mixer`` must be between 2 and 256, inclusive.
        skip_bias (torch.Tensor): Skip-connection bias ``(dim,)``.

    Returns:
        torch.Tensor: ``y_gated`` of shape ``(batch, dim, seq_len)`` — the
        post-gated final output of the fused Hyena-SE block.
    """
    _, y_gated = torch.ops.cudnn.b2b_causal_conv1d_fwd_primitive(x, weights_proj, weights_mixer, skip_bias)
    return y_gated
