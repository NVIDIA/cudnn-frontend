# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental FE-OSS API for SM100 causal-convolution decode update."""

from contextlib import contextmanager
import threading
from typing import Iterator, Optional

from cuda.bindings import driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_stream
import torch

from cudnn.api_base import APIBase, TensorDesc, TupleDict

from .kernel import (
    CausalConv1dUpdateKernel,
    CausalConv1dUpdateRowBatchKernel,
    select_rows_per_cta,
)

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

    handle = int(current_stream)
    torch_current = torch.cuda.current_stream(device)
    torch_default = torch.cuda.default_stream(device)
    if handle in (0, 1, torch_default.cuda_stream):
        launch_stream = torch_default
    elif handle == 2:
        # CU_STREAM_PER_THREAD cannot be represented safely as a PyTorch
        # ExternalStream on every supported build.  The class API remains
        # available to callers that own a preallocated output on that stream.
        raise ValueError("causal_conv1d_update helpers do not support the " "CU_STREAM_PER_THREAD sentinel; pass a concrete stream handle")
    elif handle == torch_current.cuda_stream:
        launch_stream = torch_current
    else:
        launch_stream = torch.cuda.ExternalStream(handle, device=device)
    with torch.cuda.stream(launch_stream):
        yield


def _require_storage_alignment(tensor: torch.Tensor, alignment: int, name: str) -> None:
    """Enforce the runtime alignment promised to CuTe during compilation."""

    remainder = tensor.data_ptr() % alignment
    if remainder:
        raise ValueError(f"{name} data pointer must be {alignment}-byte aligned, " f"got address modulo {alignment} = {remainder}")


def _tensors_overlap(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    """Conservative storage-overlap check used by the mutating public API."""

    overlaps = getattr(torch._C, "_overlaps", None)
    if overlaps is not None:
        return bool(overlaps(lhs, rhs))
    return lhs.untyped_storage().data_ptr() == rhs.untyped_storage().data_ptr()


class CausalConv1dUpdateSm100(APIBase):
    """Compile and execute BF16 K=4 causal-convolution decode on SM100.

    This inference-only API advances ``state`` in place and writes ``output``.
    It deliberately supports one narrow contract:

    * ``x`` and ``output``: contiguous ``[N, D]`` BF16
    * ``weight``: contiguous ``[D, 4]`` BF16
    * ``state``: contiguous ``[S, D, 4]`` BF16, mutated in place
    * optional ``state_indices``: contiguous CUDA ``int32[N]``
    * no bias; SiLU is always fused

    Indexed slots must be in ``[0, S)`` and unique.  The kernel checks both
    properties on device and executes a PTX trap on violation, so invalid
    mutable routing never silently races.  The CUDA error is asynchronous and
    the failed launch is not transactional.
    """

    def __init__(
        self,
        sample_x: torch.Tensor,
        sample_weight: torch.Tensor,
        sample_state: torch.Tensor,
        sample_output: torch.Tensor,
        sample_state_indices: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self._warn_experimental_api()

        self.x_desc = self._make_tensor_desc(sample_x, name="sample_x")
        self.weight_desc = self._make_tensor_desc(sample_weight, name="sample_weight")
        self.state_desc = self._make_tensor_desc(sample_state, name="sample_state")
        self.output_desc = self._make_tensor_desc(sample_output, name="sample_output")
        self.state_indices_desc = self._make_tensor_desc(sample_state_indices, name="sample_state_indices")

        # TensorDesc deliberately does not retain sample tensors.  Preserve
        # only the pointer remainders needed to validate the assumed alignment
        # passed to make_fake_cute_tensor_from_desc().
        self._sample_alignment_remainders = {
            "X": sample_x.data_ptr() % 16,
            "Weight": sample_weight.data_ptr() % 16,
            "State": sample_state.data_ptr() % 16,
            "Output": sample_output.data_ptr() % 16,
        }
        if sample_state_indices is not None:
            self._sample_alignment_remainders["State indices"] = sample_state_indices.data_ptr() % 4

        self.n_rows = None
        self.n_channels = None
        self.n_slots = None
        self.rows_per_cta = None

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
        if self.state_indices_desc is not None:
            self._require_rank(self.state_indices_desc, 1, "State indices")

        n_rows, n_channels = self.x_desc.shape
        n_slots = self.state_desc.shape[0]
        self._value_error_if(n_rows <= 0, f"N must be positive, got {n_rows}")
        self._value_error_if(n_channels <= 0, f"D must be positive, got {n_channels}")
        self._value_error_if(n_slots <= 0, f"S must be positive, got {n_slots}")
        self._value_error_if(n_rows > 2**31 - 1, f"N exceeds the Int32 launch limit: {n_rows}")
        self._value_error_if(n_slots > 2**31 - 1, f"S exceeds the Int32 indexing limit: {n_slots}")
        self._value_error_if(
            n_channels > 256 * 65535,
            f"D exceeds the CUDA grid-y limit for 256-channel tiles: {n_channels}",
        )

        self._check_tensor_shape(self.weight_desc, (n_channels, 4), "Weight")
        self._check_tensor_shape(self.state_desc, (n_slots, n_channels, 4), "State")
        self._check_tensor_shape(self.output_desc, (n_rows, n_channels), "Output")
        if self.state_indices_desc is None:
            self._value_error_if(
                n_slots < n_rows,
                f"State needs at least N slots when state_indices is omitted, got S={n_slots}, N={n_rows}",
            )
        else:
            self._check_tensor_shape(self.state_indices_desc, (n_rows,), "State indices")

        self._check_tensor_stride(
            self.x_desc,
            stride=(n_channels, 1),
            name="X",
            extra_error_msg="X must be row-major contiguous",
        )
        self._check_tensor_stride(
            self.weight_desc,
            stride=(4, 1),
            name="Weight",
            extra_error_msg="Weight must be row-major contiguous",
        )
        self._check_tensor_stride(
            self.state_desc,
            stride=(n_channels * 4, 4, 1),
            name="State",
            extra_error_msg="State must be row-major contiguous",
        )
        self._check_tensor_stride(
            self.output_desc,
            stride=(n_channels, 1),
            name="Output",
            extra_error_msg="Output must be row-major contiguous",
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
            compute_capability != (10, 0),
            "CausalConv1dUpdateSm100 requires exactly SM100 " f"(compute capability 10.0), found {compute_capability[0]}.{compute_capability[1]}",
        )

        self.n_rows = n_rows
        self.n_channels = n_channels
        self.n_slots = n_slots
        self.rows_per_cta = select_rows_per_cta(
            n_rows,
            n_channels,
            self.state_indices_desc is not None,
        )
        self._is_supported = True
        return True

    def compile(self) -> None:
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return

        fake_x = self._make_fake_cute_tensor_from_desc(self.x_desc, assumed_align=16)
        fake_weight = self._make_fake_cute_tensor_from_desc(self.weight_desc, assumed_align=16)
        fake_state = self._make_fake_cute_tensor_from_desc(self.state_desc, assumed_align=16)
        fake_output = self._make_fake_cute_tensor_from_desc(self.output_desc, assumed_align=16)
        fake_state_indices = self._make_fake_cute_tensor_from_desc(self.state_indices_desc, assumed_align=4)
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)

        kernel = CausalConv1dUpdateRowBatchKernel() if self.rows_per_cta == 2 else CausalConv1dUpdateKernel()
        # CuTe DSL targets the current CUDA device.  Honor the sample tensor's
        # device even when a multi-GPU caller has another device current.
        with torch.cuda.device(self.x_desc.device):
            self._compiled_kernel = cute.compile(
                kernel,
                fake_x,
                fake_weight,
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
    ) -> torch.Tensor:
        self._runtime_error_if(
            self._compiled_kernel is None,
            "CausalConv1dUpdateSm100 kernel not compiled; call compile() first",
        )

        self._validate_runtime_tensor(x_tensor, self.x_desc, "X")
        self._validate_runtime_tensor(weight_tensor, self.weight_desc, "Weight")
        self._validate_runtime_tensor(state_tensor, self.state_desc, "State")
        self._validate_runtime_tensor(output_tensor, self.output_desc, "Output")
        if (state_indices_tensor is None) != (self.state_indices_desc is None):
            raise ValueError("state_indices presence must match the compiled signature")
        if state_indices_tensor is not None:
            self._validate_runtime_tensor(
                state_indices_tensor,
                self.state_indices_desc,
                "State indices",
            )

        for name, tensor in (
            ("X", x_tensor),
            ("Weight", weight_tensor),
            ("State", state_tensor),
            ("Output", output_tensor),
        ):
            _require_storage_alignment(tensor, 16, name)
        if state_indices_tensor is not None:
            _require_storage_alignment(state_indices_tensor, 4, "State indices")

        if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (x_tensor, weight_tensor, state_tensor, output_tensor)):
            raise RuntimeError("causal_conv1d_update is inference-only; call it under torch.no_grad()")
        for name, tensor in (
            ("X", x_tensor),
            ("Weight", weight_tensor),
            ("Output", output_tensor),
        ):
            if _tensors_overlap(state_tensor, tensor):
                raise ValueError(f"State must not overlap {name}")
        if state_indices_tensor is not None and _tensors_overlap(state_tensor, state_indices_tensor):
            raise ValueError("State must not overlap State indices")
        for name, tensor in (("X", x_tensor), ("Weight", weight_tensor)):
            if _tensors_overlap(output_tensor, tensor):
                raise ValueError(f"Output must not overlap {name}")
        if state_indices_tensor is not None and _tensors_overlap(output_tensor, state_indices_tensor):
            raise ValueError("Output must not overlap State indices")

        if current_stream is None:
            current_stream = cuda.CUstream(torch.cuda.current_stream(x_tensor.device).cuda_stream)

        self._compiled_kernel(
            x_tensor,
            weight_tensor,
            state_tensor,
            output_tensor,
            state_indices_tensor,
            cutlass.Int32(self.n_slots),
            cutlass.Int32(self.n_channels),
            current_stream,
        )
        return output_tensor


def _cache_key(
    x_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    state_tensor: torch.Tensor,
    state_indices_tensor: Optional[torch.Tensor],
):
    return (
        x_tensor.device.type,
        x_tensor.device.index,
        tuple(x_tensor.shape),
        tuple(weight_tensor.shape),
        tuple(state_tensor.shape),
        state_indices_tensor is not None,
    )


def causal_conv1d_update(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    state_indices: Optional[torch.Tensor] = None,
    *,
    current_stream: Optional[cuda.CUstream] = None,
) -> torch.Tensor:
    """Advance a BF16 K=4 causal-convolution cache and return fused-SiLU output.

    ``state`` is updated in place.  This experimental operation is SM100-only,
    inference-only, no-bias, and always applies SiLU.  See
    :class:`CausalConv1dUpdateSm100` for the exact tensor contract.
    """

    for name, tensor in (("X", x), ("Weight", weight), ("State", state)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if state_indices is not None and not isinstance(state_indices, torch.Tensor):
        raise TypeError("State indices must be a torch.Tensor or None, " f"got {type(state_indices).__name__}")
    if not x.is_cuda:
        raise ValueError(f"X must be a CUDA tensor, got device {x.device}")
    key = _cache_key(x, weight, state, state_indices)

    with torch.cuda.device(x.device), _torch_stream_context(current_stream, x.device):
        output = torch.empty_like(x, memory_format=torch.contiguous_format)
        api = _API_CACHE.get(key)
        if api is None:
            with _API_CACHE_LOCK:
                api = _API_CACHE.get(key)
                if api is None:
                    api = CausalConv1dUpdateSm100(
                        sample_x=x,
                        sample_weight=weight,
                        sample_state=state,
                        sample_output=output,
                        sample_state_indices=state_indices,
                    )
                    api.check_support()
                    api.compile()
                    if len(_API_CACHE) >= _API_CACHE_CAPACITY:
                        _API_CACHE.pop(next(iter(_API_CACHE)))
                    _API_CACHE[key] = api

        return api.execute(
            x_tensor=x,
            weight_tensor=weight,
            state_tensor=state,
            output_tensor=output,
            state_indices_tensor=state_indices,
            current_stream=current_stream,
        )


def causal_conv1d_update_wrapper_sm100(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    state_indices: Optional[torch.Tensor] = None,
    *,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Run the SM100 decode update and return ``TupleDict(output_tensor=...)``.

    ``state`` is updated in place.  Use :func:`causal_conv1d_update` when a
    direct Tensor return is more convenient for model-integration shims.
    """

    output = causal_conv1d_update(
        x,
        weight,
        state,
        state_indices,
        current_stream=current_stream,
    )
    return TupleDict(output_tensor=output)
