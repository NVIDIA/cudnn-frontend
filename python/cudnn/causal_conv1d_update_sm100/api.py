# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private prepared implementation for causal-convolution decode update.

The model-facing API lives at :func:`cudnn.ops.causal_conv1d_update`. This
module owns compilation, preallocated output execution, and explicit-stream
support used by focused benchmarks; none of those mechanics are public API.
"""

import threading
from contextlib import contextmanager
from typing import Iterator, Optional, Union

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda
from cutlass.cute.runtime import make_fake_stream

from cudnn._causal_conv1d_arch import (
    is_supported_causal_conv1d_update_compute_capability,
    supported_causal_conv1d_update_compute_capabilities_text,
)
from cudnn.api_base import APIBase, TensorDesc

from .kernel import _CausalConv1dUpdateKernel

_API_CACHE = {}
_API_CACHE_LOCK = threading.Lock()
_API_CACHE_CAPACITY = 128


@contextmanager
def _torch_stream_context(
    current_stream: Optional[cuda.CUstream],
    device: torch.device,
) -> Iterator[None]:
    """Run PyTorch allocations on the CUDA stream used by the kernel."""

    if current_stream is None:
        yield
        return

    launch_stream = _as_torch_stream(current_stream, device)
    with torch.cuda.stream(launch_stream):
        yield


def _as_torch_stream(current_stream: cuda.CUstream, device: torch.device) -> torch.cuda.Stream:
    """Resolve a concrete driver handle to the matching PyTorch stream."""

    handle = int(current_stream)
    torch_current = torch.cuda.current_stream(device)
    torch_default = torch.cuda.default_stream(device)
    if handle in (0, 1, torch_default.cuda_stream):
        return torch_default
    if handle == 2:
        # CU_STREAM_PER_THREAD cannot be represented safely as a PyTorch
        # ExternalStream on every supported build.
        raise ValueError("causal_conv1d_update helpers do not support the " "CU_STREAM_PER_THREAD sentinel; pass a concrete stream handle")
    if handle == torch_current.cuda_stream:
        return torch_current
    return torch.cuda.ExternalStream(handle, device=device)


def _record_streams(
    tensors: tuple[Optional[torch.Tensor], ...],
    consumer: Optional[torch.cuda.Stream],
) -> None:
    """Keep raw-pointer operands alive through an explicit-stream launch."""

    if consumer is None:
        return
    for tensor in tensors:
        if tensor is not None:
            tensor.record_stream(consumer)


def _require_storage_alignment(tensor: torch.Tensor, alignment: int, name: str) -> None:
    """Enforce the runtime alignment promised to CuTe during compilation."""

    remainder = tensor.data_ptr() % alignment
    if remainder:
        raise ValueError(f"{name} data pointer must be {alignment}-byte aligned, " f"got address modulo {alignment} = {remainder}")


def _tensor_byte_span(tensor: torch.Tensor) -> tuple[int, int]:
    """Return the conservative byte span of one positive-stride tensor."""

    begin = tensor.data_ptr()
    if tensor.numel() == 0:
        return begin, begin

    # Every operand except row-strided X is contiguous.  Keep that hot path
    # constant-time; execute() reuses the resulting span across all alias
    # pairs instead of rebuilding it for each comparison.
    is_contiguous = getattr(tensor, "is_contiguous", None)
    if is_contiguous is not None and is_contiguous():
        return begin, begin + tensor.numel() * tensor.element_size()

    shape = tuple(tensor.shape)
    stride = tuple(tensor.stride())
    last_element = 0
    for dim, axis_stride in zip(shape, stride):
        if dim > 1 and axis_stride <= 0:
            raise ValueError("alias validation requires positive strides, " f"got shape {shape}, strides {stride}")
        last_element += (dim - 1) * axis_stride
    return begin, begin + (last_element + 1) * tensor.element_size()


def _byte_spans_overlap(lhs: tuple[int, int], rhs: tuple[int, int]) -> bool:
    lhs_begin, lhs_end = lhs
    rhs_begin, rhs_end = rhs
    if lhs_begin == lhs_end or rhs_begin == rhs_end:
        return False
    return lhs_begin < rhs_end and rhs_begin < lhs_end


def _tensors_overlap(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    """Return whether two positive-stride tensor spans share device bytes.

    ``torch._C._overlaps`` only compares PyTorch Storage identity. DLPack can
    wrap the same device address in a distinct Storage, so compare byte spans
    derived from shape and stride.  The span conservatively includes padding
    between rows of a strided tensor.
    """

    return _byte_spans_overlap(_tensor_byte_span(lhs), _tensor_byte_span(rhs))


class _CausalConv1dUpdatePlan(APIBase):
    """Compile and execute BF16 K=4 causal-convolution decode.

    Every compute capability admitted by the decode architecture policy uses
    the same functional schedule.

    This inference-only API advances ``state`` in place and writes ``output``.
    It deliberately supports one narrow contract:

    * ``x``: BF16 ``[N, D]`` with strides ``(ld, 1)``; compact rows admit any
      ``D``, while padded rows require ``ld > D`` and ``ld % 8 == 0``
    * ``output``: contiguous ``[N, D]`` BF16
    * ``weight``: contiguous ``[D, 4]`` BF16
    * ``state``: contiguous ``[S, D, L]`` BF16 with ``L`` in ``{3, 4}``,
      mutated in place
    * optional ``state_indices``: contiguous CUDA ``int32[N]``
    * optional ``bias``: contiguous ``[D]`` BF16
    * ``activation``: compile-time ``"identity"`` or ``"silu"``

    Indexed slots must be ``-1`` or in ``[0, S)``.  ``-1`` marks a padding row
    whose output is zero and whose state is untouched.  Non-padding slots must
    be unique.  The kernel checks these properties on device and executes a
    PTX trap on violation, so invalid mutable routing never silently races.
    The CUDA error is asynchronous and the failed launch is not transactional.
    """

    def __init__(
        self,
        sample_x: Union[torch.Tensor, TensorDesc],
        sample_weight: Union[torch.Tensor, TensorDesc],
        sample_state: Union[torch.Tensor, TensorDesc],
        sample_output: Union[torch.Tensor, TensorDesc],
        sample_state_indices: Optional[Union[torch.Tensor, TensorDesc]] = None,
        sample_bias: Optional[Union[torch.Tensor, TensorDesc]] = None,
        *,
        activation: str = "identity",
    ):
        super().__init__()
        self._warn_experimental_api()

        self.x_desc = self._make_tensor_desc(sample_x, name="sample_x")
        self.weight_desc = self._make_tensor_desc(sample_weight, name="sample_weight")
        self.state_desc = self._make_tensor_desc(sample_state, name="sample_state")
        self.output_desc = self._make_tensor_desc(sample_output, name="sample_output")
        self.state_indices_desc = self._make_tensor_desc(sample_state_indices, name="sample_state_indices")
        self.bias_desc = self._make_tensor_desc(sample_bias, name="sample_bias")
        if activation not in ("identity", "silu"):
            raise ValueError(f"activation must be 'identity' or 'silu', got {activation!r}")
        self.activation = activation

        # TensorDesc deliberately does not retain sample tensors.  Preserve
        # only the pointer remainders needed to validate the assumed alignment
        # passed to make_fake_cute_tensor_from_desc().  Metadata-only samples
        # defer this check to the live tensors validated by execute().
        self._sample_alignment_remainders = {}
        for name, sample, alignment in (
            ("X", sample_x, 16),
            ("Weight", sample_weight, 16),
            ("State", sample_state, 16),
            ("Output", sample_output, 16),
            ("State indices", sample_state_indices, 4),
            ("Bias", sample_bias, 16),
        ):
            data_ptr = getattr(sample, "data_ptr", None)
            if callable(data_ptr):
                self._sample_alignment_remainders[name] = data_ptr() % alignment

        self.n_rows = None
        self.n_channels = None
        self.n_slots = None
        self.state_len = None

    @staticmethod
    def _require_rank(desc: TensorDesc, rank: int, name: str) -> None:
        if desc.ndim != rank:
            raise ValueError(f"{name} must be {rank}D, got shape {desc.shape}")

    @staticmethod
    def _require_cuda(desc: TensorDesc, name: str) -> None:
        if desc.device.type != "cuda":
            raise ValueError(f"{name} must be a CUDA tensor, got device {desc.device}")

    def check_support(self) -> bool:
        self._require_rank(self.x_desc, 2, "X")
        self._require_rank(self.weight_desc, 2, "Weight")
        self._require_rank(self.state_desc, 3, "State")
        self._require_rank(self.output_desc, 2, "Output")
        if self.bias_desc is not None:
            self._require_rank(self.bias_desc, 1, "Bias")
        if self.state_indices_desc is not None:
            self._require_rank(self.state_indices_desc, 1, "State indices")

        n_rows, n_channels = self.x_desc.shape
        n_slots = self.state_desc.shape[0]
        state_len = self.state_desc.shape[2]
        self._value_error_if(n_rows <= 0, f"N must be positive, got {n_rows}")
        self._value_error_if(n_channels <= 0, f"D must be positive, got {n_channels}")
        self._value_error_if(n_slots <= 0, f"S must be positive, got {n_slots}")
        self._value_error_if(n_rows > 2**31 - 1, f"N exceeds the Int32 launch limit: {n_rows}")
        self._value_error_if(n_slots > 2**31 - 1, f"S exceeds the Int32 indexing limit: {n_slots}")
        self._value_error_if(
            n_channels > 256 * 65535,
            f"D exceeds the CUDA grid-y limit for 256-channel tiles: {n_channels}",
        )
        self._value_error_if(
            state_len not in (3, 4),
            f"State length must be 3 or 4 for width-four decode, got L={state_len}",
        )

        self._check_tensor_shape(self.weight_desc, (n_channels, 4), "Weight")
        self._check_tensor_shape(self.state_desc, (n_slots, n_channels, state_len), "State")
        self._check_tensor_shape(self.output_desc, (n_rows, n_channels), "Output")
        if self.bias_desc is not None:
            self._check_tensor_shape(self.bias_desc, (n_channels,), "Bias")
        if self.state_indices_desc is None:
            self._value_error_if(
                n_slots < n_rows,
                f"State needs at least N slots when state_indices is omitted, got S={n_slots}, N={n_rows}",
            )
        else:
            self._check_tensor_shape(self.state_indices_desc, (n_rows,), "State indices")

        x_row_stride, x_channel_stride = self.x_desc.stride
        self._value_error_if(
            x_channel_stride != 1,
            f"X must have row-major strides (ld, 1), got {self.x_desc.stride}",
        )
        self._value_error_if(
            x_row_stride < n_channels,
            f"X row stride ld must be at least D={n_channels}, got ld={x_row_stride}",
        )
        self._value_error_if(
            x_row_stride > n_channels and x_row_stride % 8 != 0,
            "Padded X rows must start at 16-byte-aligned BF16 addresses " f"(ld % 8 == 0), got D={n_channels}, ld={x_row_stride}",
        )
        self._check_tensor_stride(
            self.weight_desc,
            stride=(4, 1),
            name="Weight",
            extra_error_msg="Weight must be row-major contiguous",
        )
        self._check_tensor_stride(
            self.state_desc,
            stride=(n_channels * state_len, state_len, 1),
            name="State",
            extra_error_msg="State must be row-major contiguous",
        )
        self._check_tensor_stride(
            self.output_desc,
            stride=(n_channels, 1),
            name="Output",
            extra_error_msg="Output must be row-major contiguous",
        )
        if self.bias_desc is not None:
            self._check_tensor_stride(
                self.bias_desc,
                stride=(1,),
                name="Bias",
                extra_error_msg="Bias must be contiguous",
            )
        if self.state_indices_desc is not None:
            self._check_tensor_stride(
                self.state_indices_desc,
                stride=(1,),
                name="State indices",
                extra_error_msg="State indices must be contiguous",
            )

        self._check_dtype(self.x_desc, dtype=torch.bfloat16, name="X")
        self._check_dtype(self.weight_desc, dtype=torch.bfloat16, name="Weight")
        self._check_dtype(self.state_desc, dtype=torch.bfloat16, name="State")
        self._check_dtype(self.output_desc, dtype=torch.bfloat16, name="Output")
        if self.bias_desc is not None:
            self._check_dtype(self.bias_desc, dtype=torch.bfloat16, name="Bias")
        if self.state_indices_desc is not None:
            self._check_dtype(self.state_indices_desc, dtype=torch.int32, name="State indices")

        for name, remainder in self._sample_alignment_remainders.items():
            alignment = 4 if name == "State indices" else 16
            self._value_error_if(
                remainder != 0,
                f"{name} data pointer must be {alignment}-byte aligned, " f"got address modulo {alignment} = {remainder}",
            )

        descs = [
            ("X", self.x_desc),
            ("Weight", self.weight_desc),
            ("State", self.state_desc),
            ("Output", self.output_desc),
        ]
        if self.bias_desc is not None:
            descs.append(("Bias", self.bias_desc))
        if self.state_indices_desc is not None:
            descs.append(("State indices", self.state_indices_desc))
        for name, desc in descs:
            self._require_cuda(desc, name)
            self._value_error_if(
                desc.device != self.x_desc.device,
                f"{name} must be on {self.x_desc.device}, got {desc.device}",
            )

        self._runtime_error_if(not torch.cuda.is_available(), "CUDA is not available")
        compute_capability = torch.cuda.get_device_capability(self.x_desc.device)
        self._runtime_error_if(
            not is_supported_causal_conv1d_update_compute_capability(compute_capability),
            "causal_conv1d_update supports compute capabilities "
            f"{supported_causal_conv1d_update_compute_capabilities_text()}, "
            f"found {compute_capability[0]}.{compute_capability[1]}",
        )

        self.n_rows = n_rows
        self.n_channels = n_channels
        self.n_slots = n_slots
        self.state_len = state_len
        self._is_supported = True
        return True

    def compile(self) -> None:
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return

        fake_x = self._make_fake_cute_tensor_from_desc(self.x_desc, assumed_align=16)
        fake_weight = self._make_fake_cute_tensor_from_desc(self.weight_desc, assumed_align=16)
        fake_bias = self._make_fake_cute_tensor_from_desc(self.bias_desc, assumed_align=16)
        fake_state = self._make_fake_cute_tensor_from_desc(self.state_desc, assumed_align=16)
        fake_output = self._make_fake_cute_tensor_from_desc(self.output_desc, assumed_align=16)
        fake_state_indices = self._make_fake_cute_tensor_from_desc(self.state_indices_desc, assumed_align=4)
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)

        kernel = _CausalConv1dUpdateKernel(
            apply_silu=self.activation == "silu",
            state_len=self.state_len,
        )
        # CuTe DSL targets the current CUDA device.  Honor the sample tensor's
        # device even when a multi-GPU caller has another device current.
        with torch.cuda.device(self.x_desc.device):
            self._compiled_kernel = cute.compile(
                kernel,
                fake_x,
                fake_weight,
                fake_bias,
                fake_state,
                fake_output,
                fake_state_indices,
                cutlass.Int32(self.n_slots),
                cutlass.Int32(self.n_channels),
                fake_stream,
                options="--enable-tvm-ffi --generate-line-info",
            )

    @staticmethod
    def _validate_runtime_tensor(tensor: torch.Tensor, desc: TensorDesc, name: str) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
        if tuple(tensor.shape) != desc.shape:
            raise ValueError(f"{name} shape mismatch: expected {desc.shape}, got {tuple(tensor.shape)}")
        if tuple(tensor.stride()) != desc.stride:
            raise ValueError(f"{name} stride mismatch: expected {desc.stride}, got {tuple(tensor.stride())}")
        if tensor.dtype != desc.dtype:
            raise TypeError(f"{name} dtype mismatch: expected {desc.dtype}, got {tensor.dtype}")
        if tensor.device != desc.device:
            raise ValueError(f"{name} device mismatch: expected {desc.device}, got {tensor.device}")

    def execute(
        self,
        x_tensor: torch.Tensor,
        weight_tensor: torch.Tensor,
        state_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        state_indices_tensor: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
        bias_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._runtime_error_if(
            self._compiled_kernel is None,
            "causal_conv1d_update plan not compiled; call compile() first",
        )

        self._validate_runtime_tensor(x_tensor, self.x_desc, "X")
        self._validate_runtime_tensor(weight_tensor, self.weight_desc, "Weight")
        self._validate_runtime_tensor(state_tensor, self.state_desc, "State")
        self._validate_runtime_tensor(output_tensor, self.output_desc, "Output")
        if (bias_tensor is None) != (self.bias_desc is None):
            raise ValueError("bias presence must match the compiled signature")
        if bias_tensor is not None:
            self._validate_runtime_tensor(bias_tensor, self.bias_desc, "Bias")
        if (state_indices_tensor is None) != (self.state_indices_desc is None):
            raise ValueError("state_indices presence must match the compiled signature")
        if state_indices_tensor is not None:
            self._validate_runtime_tensor(
                state_indices_tensor,
                self.state_indices_desc,
                "State indices",
            )

        operands = (
            ("X", x_tensor, 16),
            ("Weight", weight_tensor, 16),
            ("State", state_tensor, 16),
            ("Output", output_tensor, 16),
            ("Bias", bias_tensor, 16),
            ("State indices", state_indices_tensor, 4),
        )
        spans = {}
        for name, tensor, alignment in operands:
            if tensor is not None:
                _require_storage_alignment(tensor, alignment, name)
                spans[name] = _tensor_byte_span(tensor)

        grad_tensors = (
            x_tensor,
            weight_tensor,
            state_tensor,
            output_tensor,
            bias_tensor,
        )
        if torch.is_grad_enabled() and any(tensor is not None and tensor.requires_grad for tensor in grad_tensors):
            raise RuntimeError("causal_conv1d_update is inference-only; call it under torch.no_grad()")
        for owner, other in (
            ("State", "X"),
            ("State", "Weight"),
            ("State", "Output"),
            ("State", "Bias"),
            ("State", "State indices"),
            ("Output", "X"),
            ("Output", "Weight"),
            ("Output", "Bias"),
            ("Output", "State indices"),
        ):
            if other in spans and _byte_spans_overlap(spans[owner], spans[other]):
                raise ValueError(f"{owner} must not overlap {other}")

        if current_stream is None:
            consumer_stream = torch.cuda.current_stream(x_tensor.device)
            launch_stream = cuda.CUstream(consumer_stream.cuda_stream)
        else:
            consumer_stream = _as_torch_stream(current_stream, x_tensor.device)
            launch_stream = current_stream

        self._compiled_kernel(
            x_tensor,
            weight_tensor,
            bias_tensor,
            state_tensor,
            output_tensor,
            state_indices_tensor,
            cutlass.Int32(self.n_slots),
            cutlass.Int32(self.n_channels),
            launch_stream,
        )
        _record_streams(
            (
                x_tensor,
                weight_tensor,
                bias_tensor,
                state_tensor,
                output_tensor,
                state_indices_tensor,
            ),
            consumer_stream,
        )
        return output_tensor


def _cache_key(
    x_tensor: torch.Tensor,
    state_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    state_indices_tensor: Optional[torch.Tensor],
    bias_tensor: Optional[torch.Tensor],
    activation: str,
):
    return (
        x_tensor.device.type,
        x_tensor.device.index,
        tuple(x_tensor.shape),
        tuple(x_tensor.stride()),
        tuple(weight_tensor.shape),
        tuple(state_tensor.shape),
        state_indices_tensor is not None,
        bias_tensor is not None,
        activation,
    )


def _causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: str = "identity",
    *,
    conv_state_indices: Optional[torch.Tensor] = None,
    current_stream: Optional[cuda.CUstream] = None,
) -> torch.Tensor:
    """Private allocating route used by the public custom op and FLA adapter.

    ``conv_state`` is updated in place. ``activation`` is already normalized
    to ``"identity"`` or ``"silu"`` by the public semantic API. Explicit
    streams remain private because the public PyTorch operation follows the
    current-stream convention.
    """

    for name, tensor in (("X", x), ("Weight", weight), ("State", conv_state)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if conv_state_indices is not None and not isinstance(conv_state_indices, torch.Tensor):
        raise TypeError("State indices must be a torch.Tensor or None, " f"got {type(conv_state_indices).__name__}")
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise TypeError("Bias must be a torch.Tensor or None, " f"got {type(bias).__name__}")
    if activation not in ("identity", "silu"):
        raise ValueError(f"activation must be 'identity' or 'silu', got {activation!r}")
    if not x.is_cuda:
        raise ValueError(f"X must be a CUDA tensor, got device {x.device}")
    key = _cache_key(x, conv_state, weight, conv_state_indices, bias, activation)

    with torch.cuda.device(x.device), _torch_stream_context(current_stream, x.device):
        output = torch.empty_like(x, memory_format=torch.contiguous_format)
        api = _API_CACHE.get(key)
        if api is None:
            with _API_CACHE_LOCK:
                api = _API_CACHE.get(key)
                if api is None:
                    api = _CausalConv1dUpdatePlan(
                        sample_x=x,
                        sample_weight=weight,
                        sample_state=conv_state,
                        sample_output=output,
                        sample_state_indices=conv_state_indices,
                        sample_bias=bias,
                        activation=activation,
                    )
                    api.check_support()
                    api.compile()
                    if len(_API_CACHE) >= _API_CACHE_CAPACITY:
                        _API_CACHE.pop(next(iter(_API_CACHE)))
                    _API_CACHE[key] = api

        return api.execute(
            x_tensor=x,
            weight_tensor=weight,
            state_tensor=conv_state,
            output_tensor=output,
            state_indices_tensor=conv_state_indices,
            current_stream=current_stream,
            bias_tensor=bias,
        )
