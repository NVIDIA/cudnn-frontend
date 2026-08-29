# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental FE-OSS API for portable bulk causal convolution.

The public class retains its original ``Sm100`` suffix while this experimental
API evolves. SM100 uses the measured vector schedule, other listed targets at
SM100 or newer use the same instruction-compatible path, and supported
pre-Blackwell targets use the correctness-first scalar schedule.
"""

from contextlib import contextmanager
import threading
from typing import Iterator, Optional

from cuda.bindings import driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_stream
import torch

from cudnn.api_base import APIBase, TensorDesc, TupleDict
from cudnn._causal_conv1d_bulk_arch import (
    FUNCTIONAL_COMPUTE_CAPABILITIES,
    is_functional_arch,
    uses_vec8_schedule,
)
from cudnn.frost.buffers import CUTEDSL_MIN_VERSION, cutedsl_state, cutedsl_too_old

_API_CACHE = {}
_API_CACHE_LOCK = threading.Lock()
_API_CACHE_CAPACITY = 128
_INT32_MAX = 2**31 - 1
# Kernels form ``tile_end = first_token + 16`` in Int32.  Reserve the
# largest possible tile overrun so that every accepted launch stays defined.
_MAX_TOTAL_TOKENS = _INT32_MAX - 15
# Packed kernels use ``(lower + upper) // 2`` during their device-side search.
_MAX_PACKED_SEQUENCES = _INT32_MAX // 2
_MAX_SCALAR_GRID_CHANNELS = 256 * 65535


@contextmanager
def _torch_stream_context(
    current_stream: Optional[cuda.CUstream],
    device: torch.device,
) -> Iterator[None]:
    """Run wrapper-owned PyTorch allocations on the launch stream."""

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
        raise ValueError("causal_conv1d_bulk_sm100 does not support the " "CU_STREAM_PER_THREAD sentinel; pass a concrete stream handle")
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
    remainder = tensor.data_ptr() % alignment
    if remainder:
        raise ValueError(f"{name} data pointer must be {alignment}-byte aligned, " f"got address modulo {alignment} = {remainder}")


def _tensors_overlap(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    """Return whether two exact-contiguous tensors share any device bytes.

    ``torch._C._overlaps`` only compares PyTorch Storage identity.  DLPack can
    wrap the same device address in a distinct Storage, so use the byte spans
    guaranteed by this API's exact-contiguous tensor contract instead.
    """

    lhs_begin = lhs.data_ptr()
    rhs_begin = rhs.data_ptr()
    lhs_end = lhs_begin + lhs.numel() * lhs.element_size()
    rhs_end = rhs_begin + rhs.numel() * rhs.element_size()
    return lhs_begin < rhs_end and rhs_begin < lhs_end


class CausalConv1dBulkFwdSm100(APIBase):
    """Compile and execute BF16 width-four bulk causal convolution.

    The native layout is contiguous ``x[B, T, D]`` and ``weight[D, 4]``.
    Dense mode treats each batch row as a sequence.  Packed mode requires
    ``B == 1`` and a CUDA int32 ``cu_seqlens[N + 1]`` tensor.  Optional initial
    and final states use the full-width ``[N, D, 4]`` decode-cache layout.

    This first slice has no bias and always applies SiLU.  It is inference-only
    until the matching backward primitive is available.
    """

    def __init__(
        self,
        sample_x: torch.Tensor,
        sample_weight: torch.Tensor,
        sample_output: torch.Tensor,
        sample_cu_seqlens: Optional[torch.Tensor] = None,
        sample_initial_state: Optional[torch.Tensor] = None,
        sample_final_state: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self._warn_experimental_api()

        for name, sample in (
            ("sample_x", sample_x),
            ("sample_weight", sample_weight),
            ("sample_output", sample_output),
        ):
            if not isinstance(sample, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(sample).__name__}")
        for name, sample in (
            ("sample_cu_seqlens", sample_cu_seqlens),
            ("sample_initial_state", sample_initial_state),
            ("sample_final_state", sample_final_state),
        ):
            if sample is not None and not isinstance(sample, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(sample).__name__}")

        self.x_desc = self._make_tensor_desc(sample_x, name="sample_x")
        self.weight_desc = self._make_tensor_desc(sample_weight, name="sample_weight")
        self.output_desc = self._make_tensor_desc(sample_output, name="sample_output")
        self.cu_seqlens_desc = self._make_tensor_desc(sample_cu_seqlens, name="sample_cu_seqlens")
        self.initial_state_desc = self._make_tensor_desc(sample_initial_state, name="sample_initial_state")
        self.final_state_desc = self._make_tensor_desc(sample_final_state, name="sample_final_state")

        # TensorDesc intentionally does not retain sample storage.  Preserve
        # only the pointer remainders backing the alignment promises made to
        # CuTe during compilation.
        self._sample_alignment_remainders = {
            "X": sample_x.data_ptr() % 16,
            "Weight": sample_weight.data_ptr() % 16,
            "Output": sample_output.data_ptr() % 16,
        }
        if sample_cu_seqlens is not None:
            self._sample_alignment_remainders["cu_seqlens"] = sample_cu_seqlens.data_ptr() % 4
        if sample_initial_state is not None:
            self._sample_alignment_remainders["Initial state"] = sample_initial_state.data_ptr() % 16
        if sample_final_state is not None:
            self._sample_alignment_remainders["Final state"] = sample_final_state.data_ptr() % 16

        self.batch_size = None
        self.sample_sequence_length = None
        self.n_channels = None
        self.num_sequences = None
        self.compute_capability = None
        self.use_vec8_schedule = None
        self.is_packed = sample_cu_seqlens is not None

    @staticmethod
    def _require_rank(desc: TensorDesc, rank: int, name: str) -> None:
        if desc.ndim != rank:
            raise ValueError(f"{name} must be {rank}D, got shape {desc.shape}")

    @staticmethod
    def _require_cuda(desc: TensorDesc, name: str) -> None:
        if desc.device.type != "cuda":
            raise ValueError(f"{name} must be a CUDA tensor, got device {desc.device}")

    def check_support(self) -> bool:
        cutedsl_installed, cutedsl_version = cutedsl_state()
        cutedsl_floor = ".".join(str(component) for component in CUTEDSL_MIN_VERSION)
        self._runtime_error_if(
            not cutedsl_installed,
            "causal_conv1d_bulk_sm100 requires the cutedsl extra " f"(nvidia-cutlass-dsl>={cutedsl_floor})",
        )
        if cutedsl_too_old(cutedsl_version):
            self._runtime_error_if(
                True,
                f"causal_conv1d_bulk_sm100 requires nvidia-cutlass-dsl>={cutedsl_floor}; " f"found {cutedsl_version[1]}",
            )

        self._require_rank(self.x_desc, 3, "X")
        self._require_rank(self.weight_desc, 2, "Weight")
        self._require_rank(self.output_desc, 3, "Output")
        if self.cu_seqlens_desc is not None:
            self._require_rank(self.cu_seqlens_desc, 1, "cu_seqlens")
        if self.initial_state_desc is not None:
            self._require_rank(self.initial_state_desc, 3, "Initial state")
        if self.final_state_desc is not None:
            self._require_rank(self.final_state_desc, 3, "Final state")

        batch_size, sequence_length, n_channels = self.x_desc.shape
        self._value_error_if(batch_size <= 0, f"B must be positive, got {batch_size}")
        self._value_error_if(sequence_length <= 0, f"T must be positive, got {sequence_length}")
        self._value_error_if(n_channels <= 0, f"D must be positive, got {n_channels}")
        self._value_error_if(batch_size > _INT32_MAX, f"B exceeds the Int32 limit: {batch_size}")
        self._value_error_if(sequence_length > _INT32_MAX, f"T exceeds the Int32 limit: {sequence_length}")
        self._value_error_if(
            batch_size * sequence_length > _MAX_TOTAL_TOKENS,
            f"B*T exceeds the safe Int32 tiled-indexing limit: {batch_size * sequence_length}",
        )
        self._value_error_if(n_channels > _MAX_SCALAR_GRID_CHANNELS, f"D exceeds the supported CUDA grid limit: {n_channels}")

        if self.cu_seqlens_desc is None:
            num_sequences = batch_size
        else:
            self._value_error_if(batch_size != 1, f"Packed X must have B=1, got B={batch_size}")
            self._value_error_if(
                self.cu_seqlens_desc.shape[0] < 2,
                "Packed cu_seqlens must contain at least a start and end offset",
            )
            num_sequences = self.cu_seqlens_desc.shape[0] - 1
            self._value_error_if(
                num_sequences > _MAX_PACKED_SEQUENCES,
                f"Packed N exceeds the safe Int32 search limit: {num_sequences}",
            )
            self._value_error_if(
                num_sequences > batch_size * sequence_length,
                f"Packed N={num_sequences} cannot exceed total_T={batch_size * sequence_length} " "when every sequence must be non-empty",
            )

        self._check_tensor_shape(self.weight_desc, (n_channels, 4), "Weight")
        self._check_tensor_shape(self.output_desc, (batch_size, sequence_length, n_channels), "Output")
        if self.cu_seqlens_desc is not None:
            self._check_tensor_shape(self.cu_seqlens_desc, (num_sequences + 1,), "cu_seqlens")
        if self.initial_state_desc is not None:
            self._check_tensor_shape(self.initial_state_desc, (num_sequences, n_channels, 4), "Initial state")
        if self.final_state_desc is not None:
            self._check_tensor_shape(self.final_state_desc, (num_sequences, n_channels, 4), "Final state")

        self._check_tensor_stride(
            self.x_desc,
            stride=(sequence_length * n_channels, n_channels, 1),
            name="X",
            extra_error_msg="X must be [B, T, D] contiguous",
        )
        self._check_tensor_stride(
            self.weight_desc,
            stride=(4, 1),
            name="Weight",
            extra_error_msg="Weight must be [D, 4] contiguous",
        )
        self._check_tensor_stride(
            self.output_desc,
            stride=(sequence_length * n_channels, n_channels, 1),
            name="Output",
            extra_error_msg="Output must be [B, T, D] contiguous",
        )
        if self.cu_seqlens_desc is not None:
            self._check_tensor_stride(self.cu_seqlens_desc, stride=(1,), name="cu_seqlens")
        for name, desc in (("Initial state", self.initial_state_desc), ("Final state", self.final_state_desc)):
            if desc is not None:
                self._check_tensor_stride(
                    desc,
                    stride=(n_channels * 4, 4, 1),
                    name=name,
                    extra_error_msg=f"{name} must be [N, D, 4] contiguous",
                )

        for name, desc in (
            ("X", self.x_desc),
            ("Weight", self.weight_desc),
            ("Output", self.output_desc),
            ("Initial state", self.initial_state_desc),
            ("Final state", self.final_state_desc),
        ):
            if desc is not None:
                self._check_dtype(desc, dtype=torch.bfloat16, name=name)
        if self.cu_seqlens_desc is not None:
            self._check_dtype(self.cu_seqlens_desc, dtype=torch.int32, name="cu_seqlens")

        for name, remainder in self._sample_alignment_remainders.items():
            alignment = 4 if name == "cu_seqlens" else 16
            self._value_error_if(
                remainder != 0,
                f"{name} data pointer must be {alignment}-byte aligned, " f"got address modulo {alignment} = {remainder}",
            )

        descs = [
            ("X", self.x_desc),
            ("Weight", self.weight_desc),
            ("Output", self.output_desc),
            ("cu_seqlens", self.cu_seqlens_desc),
            ("Initial state", self.initial_state_desc),
            ("Final state", self.final_state_desc),
        ]
        for name, desc in descs:
            if desc is None:
                continue
            self._require_cuda(desc, name)
            self._value_error_if(desc.device != self.x_desc.device, f"{name} must be on {self.x_desc.device}, got {desc.device}")

        self._runtime_error_if(not torch.cuda.is_available(), "CUDA is not available")
        compute_capability = torch.cuda.get_device_capability(self.x_desc.device)
        self._runtime_error_if(
            not is_functional_arch(compute_capability),
            "CausalConv1dBulkFwdSm100 does not support compute capability "
            f"{compute_capability[0]}.{compute_capability[1]}; supported capabilities are "
            f"{sorted(FUNCTIONAL_COMPUTE_CAPABILITIES)}",
        )
        use_vec8_schedule = uses_vec8_schedule(compute_capability, n_channels)
        if not use_vec8_schedule:
            self._value_error_if(
                batch_size * sequence_length * n_channels > _INT32_MAX,
                "The scalar schedule requires X and Output to contain at most " f"{_INT32_MAX} elements",
            )
            if self.initial_state_desc is not None or self.final_state_desc is not None:
                self._value_error_if(
                    num_sequences * n_channels * 4 > _INT32_MAX,
                    "The scalar schedule requires each state tensor to contain at most " f"{_INT32_MAX} elements",
                )

        self.batch_size = batch_size
        self.sample_sequence_length = sequence_length
        self.n_channels = n_channels
        self.num_sequences = num_sequences
        self.compute_capability = compute_capability
        self.use_vec8_schedule = use_vec8_schedule
        self._is_supported = True
        return True

    def compile(self) -> None:
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return

        # Import only after check_support's explicit 4.7 gate. The kernel's
        # packed math helpers do not exist in the package-wide 4.5 DSL floor.
        from .kernel import CausalConv1dBulkForwardKernel

        valid_tokens = cute.sym_int(divisibility=1)
        fake_x = self._make_fake_cute_tensor(
            dtype=self.x_desc.dtype,
            shape=(valid_tokens, self.n_channels),
            stride=(self.n_channels, 1),
            assumed_align=16,
        )
        fake_weight = self._make_fake_cute_tensor_from_desc(self.weight_desc, assumed_align=16)
        fake_initial_state = self._make_fake_cute_tensor_from_desc(self.initial_state_desc, assumed_align=16)
        fake_cu_seqlens = self._make_fake_cute_tensor_from_desc(self.cu_seqlens_desc, assumed_align=4)
        fake_output = self._make_fake_cute_tensor(
            dtype=self.output_desc.dtype,
            shape=(valid_tokens, self.n_channels),
            stride=(self.n_channels, 1),
            assumed_align=16,
        )
        fake_final_state = self._make_fake_cute_tensor_from_desc(self.final_state_desc, assumed_align=16)
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)

        kernel = CausalConv1dBulkForwardKernel(use_vec8=self.use_vec8_schedule)
        dense_tokens_per_sequence = self.sample_sequence_length if not self.is_packed else 0
        with torch.cuda.device(self.x_desc.device):
            self._compiled_kernel = cute.compile(
                kernel,
                fake_x,
                fake_weight,
                fake_initial_state,
                fake_cu_seqlens,
                fake_output,
                fake_final_state,
                cutlass.Int32(self.num_sequences),
                cutlass.Int32(dense_tokens_per_sequence),
                cutlass.Int32(self.n_channels),
                fake_stream,
                options="--enable-tvm-ffi --generate-line-info",
            )

    @staticmethod
    def _validate_runtime_static_tensor(tensor: torch.Tensor, desc: TensorDesc, name: str) -> None:
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

    def _validate_runtime_x_or_output(self, tensor: torch.Tensor, name: str) -> int:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
        if tensor.ndim != 3:
            raise ValueError(f"{name} must be 3D, got shape {tuple(tensor.shape)}")
        batch_size, sequence_length, n_channels = tensor.shape
        if batch_size != self.batch_size or n_channels != self.n_channels:
            raise ValueError(f"{name} shape mismatch: expected B={self.batch_size}, D={self.n_channels}, " f"got {tuple(tensor.shape)}")
        if sequence_length <= 0:
            raise ValueError(f"{name} T must be positive, got {sequence_length}")
        if batch_size * sequence_length > _MAX_TOTAL_TOKENS:
            raise ValueError(f"{name} B*T exceeds the safe Int32 tiled-indexing limit")
        if not self.use_vec8_schedule and batch_size * sequence_length * n_channels > _INT32_MAX:
            raise ValueError(f"The scalar schedule requires {name} to contain at most {_INT32_MAX} elements")
        expected_stride = (sequence_length * n_channels, n_channels, 1)
        if tuple(tensor.stride()) != expected_stride:
            raise ValueError(f"{name} must be [B, T, D] contiguous, got stride {tuple(tensor.stride())}")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} dtype mismatch: expected torch.bfloat16, got {tensor.dtype}")
        if tensor.device != self.x_desc.device:
            raise ValueError(f"{name} device mismatch: expected {self.x_desc.device}, got {tensor.device}")
        return sequence_length

    def execute(
        self,
        x_tensor: torch.Tensor,
        weight_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        cu_seqlens_tensor: Optional[torch.Tensor] = None,
        initial_state_tensor: Optional[torch.Tensor] = None,
        final_state_tensor: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> TupleDict:
        self._runtime_error_if(
            self._compiled_kernel is None,
            "CausalConv1dBulkFwdSm100 kernel not compiled; call compile() first",
        )

        sequence_length = self._validate_runtime_x_or_output(x_tensor, "X")
        output_sequence_length = self._validate_runtime_x_or_output(output_tensor, "Output")
        if output_sequence_length != sequence_length:
            raise ValueError(f"Output T must match X T={sequence_length}, got {output_sequence_length}")
        if self.is_packed and self.num_sequences > self.batch_size * sequence_length:
            raise ValueError(
                f"Packed N={self.num_sequences} cannot exceed runtime total_T={self.batch_size * sequence_length} " "when every sequence must be non-empty"
            )
        self._validate_runtime_static_tensor(weight_tensor, self.weight_desc, "Weight")

        optional_tensors = (
            ("cu_seqlens", cu_seqlens_tensor, self.cu_seqlens_desc),
            ("Initial state", initial_state_tensor, self.initial_state_desc),
            ("Final state", final_state_tensor, self.final_state_desc),
        )
        for name, tensor, desc in optional_tensors:
            if (tensor is None) != (desc is None):
                raise ValueError(f"{name} presence must match the compiled signature")
            if tensor is not None:
                self._validate_runtime_static_tensor(tensor, desc, name)

        for name, tensor, alignment in (
            ("X", x_tensor, 16),
            ("Weight", weight_tensor, 16),
            ("Output", output_tensor, 16),
            ("cu_seqlens", cu_seqlens_tensor, 4),
            ("Initial state", initial_state_tensor, 16),
            ("Final state", final_state_tensor, 16),
        ):
            if tensor is not None:
                _require_storage_alignment(tensor, alignment, name)

        grad_inputs = [x_tensor, weight_tensor]
        if initial_state_tensor is not None:
            grad_inputs.append(initial_state_tensor)
        if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in grad_inputs):
            raise RuntimeError("causal_conv1d_bulk_fwd is inference-only; call it under torch.no_grad()")

        read_tensors = [x_tensor, weight_tensor]
        if cu_seqlens_tensor is not None:
            read_tensors.append(cu_seqlens_tensor)
        if initial_state_tensor is not None:
            read_tensors.append(initial_state_tensor)
        for tensor in read_tensors:
            if _tensors_overlap(output_tensor, tensor):
                raise ValueError("Output must not overlap an input tensor")
        if final_state_tensor is not None:
            for tensor in [*read_tensors, output_tensor]:
                if _tensors_overlap(final_state_tensor, tensor):
                    raise ValueError("Final state must not overlap another input or output tensor")

        if current_stream is None:
            consumer_stream = torch.cuda.current_stream(x_tensor.device)
            launch_stream = cuda.CUstream(consumer_stream.cuda_stream)
        else:
            consumer_stream = _as_torch_stream(current_stream, x_tensor.device)
            launch_stream = current_stream

        flat_x = x_tensor.view(-1, self.n_channels)
        flat_output = output_tensor.view(-1, self.n_channels)
        dense_tokens_per_sequence = sequence_length if not self.is_packed else 0
        self._compiled_kernel(
            flat_x,
            weight_tensor,
            initial_state_tensor,
            cu_seqlens_tensor,
            flat_output,
            final_state_tensor,
            cutlass.Int32(self.num_sequences),
            cutlass.Int32(dense_tokens_per_sequence),
            cutlass.Int32(self.n_channels),
            launch_stream,
        )
        _record_streams(
            (
                x_tensor,
                weight_tensor,
                output_tensor,
                cu_seqlens_tensor,
                initial_state_tensor,
                final_state_tensor,
            ),
            consumer_stream,
        )
        return TupleDict(output_tensor=output_tensor, final_state_tensor=final_state_tensor)


def _cache_key(
    x_tensor: torch.Tensor,
    cu_seqlens_tensor: Optional[torch.Tensor],
    initial_state_tensor: Optional[torch.Tensor],
    output_final_state: bool,
):
    if not isinstance(x_tensor, torch.Tensor):
        raise TypeError(f"X must be a torch.Tensor, got {type(x_tensor).__name__}")
    if x_tensor.ndim != 3:
        raise ValueError(f"X must be 3D, got shape {tuple(x_tensor.shape)}")
    batch_size, sequence_length, n_channels = x_tensor.shape
    if batch_size <= 0 or sequence_length <= 0 or n_channels <= 0:
        raise ValueError(f"X dimensions must be positive, got shape {tuple(x_tensor.shape)}")
    total_tokens = batch_size * sequence_length
    if total_tokens > _MAX_TOTAL_TOKENS:
        raise ValueError(f"X B*T exceeds the safe Int32 tiled-indexing limit: {total_tokens}")
    if n_channels > _MAX_SCALAR_GRID_CHANNELS:
        raise ValueError(f"X D exceeds the supported CUDA grid limit: {n_channels}")
    if cu_seqlens_tensor is None:
        num_sequences = batch_size
    else:
        if not isinstance(cu_seqlens_tensor, torch.Tensor):
            raise TypeError("cu_seqlens must be a torch.Tensor or None, " f"got {type(cu_seqlens_tensor).__name__}")
        if cu_seqlens_tensor.ndim != 1:
            raise ValueError(f"cu_seqlens must be 1D, got shape {tuple(cu_seqlens_tensor.shape)}")
        if batch_size != 1:
            raise ValueError(f"Packed X must have B=1, got B={batch_size}")
        if cu_seqlens_tensor.shape[0] < 2:
            raise ValueError("Packed cu_seqlens must contain at least a start and end offset")
        num_sequences = cu_seqlens_tensor.shape[0] - 1
        if num_sequences > _MAX_PACKED_SEQUENCES:
            raise ValueError(f"Packed N exceeds the safe Int32 search limit: {num_sequences}")
        if num_sequences > total_tokens:
            raise ValueError(f"Packed N={num_sequences} cannot exceed total_T={total_tokens} " "when every sequence must be non-empty")
    compute_capability = torch.cuda.get_device_capability(x_tensor.device)
    use_vec8_schedule = uses_vec8_schedule(compute_capability, n_channels)
    if not use_vec8_schedule:
        if x_tensor.numel() > _INT32_MAX:
            raise ValueError(f"The scalar schedule requires X to contain at most {_INT32_MAX} elements")
        if (initial_state_tensor is not None or output_final_state) and num_sequences * n_channels * 4 > _INT32_MAX:
            raise ValueError(f"The scalar schedule requires each state tensor to contain at most {_INT32_MAX} elements")
    return (
        x_tensor.device.type,
        x_tensor.device.index,
        compute_capability,
        use_vec8_schedule,
        batch_size,
        num_sequences,
        n_channels,
        cu_seqlens_tensor is not None,
        initial_state_tensor is not None,
        bool(output_final_state),
    )


def causal_conv1d_bulk_fwd_wrapper_sm100(
    x_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    cu_seqlens_tensor: Optional[torch.Tensor] = None,
    initial_state_tensor: Optional[torch.Tensor] = None,
    *,
    output_final_state: bool = False,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Run the BF16 width-four bulk causal-convolution forward.

    The result always contains ``output_tensor`` and ``final_state_tensor`` in
    that order.  When ``output_final_state`` is false, the latter is a BF16
    CUDA sentinel with shape ``(0,)``.  Packed metadata is consumed on device;
    it must start at zero, end at ``total_T``, and be strictly increasing.
    """

    for name, tensor in (("X", x_tensor), ("Weight", weight_tensor)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if initial_state_tensor is not None and not isinstance(initial_state_tensor, torch.Tensor):
        raise TypeError("Initial state must be a torch.Tensor or None, " f"got {type(initial_state_tensor).__name__}")
    if not isinstance(output_final_state, bool):
        raise TypeError(f"output_final_state must be bool, got {type(output_final_state).__name__}")
    if not x_tensor.is_cuda:
        raise ValueError(f"X must be a CUDA tensor, got device {x_tensor.device}")

    grad_inputs = [x_tensor, weight_tensor]
    if initial_state_tensor is not None:
        grad_inputs.append(initial_state_tensor)
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in grad_inputs):
        raise RuntimeError("causal_conv1d_bulk_fwd is inference-only; call it under torch.no_grad()")

    key = _cache_key(
        x_tensor,
        cu_seqlens_tensor,
        initial_state_tensor,
        output_final_state,
    )
    if cu_seqlens_tensor is None:
        num_sequences = x_tensor.shape[0]
    else:
        num_sequences = cu_seqlens_tensor.shape[0] - 1

    with torch.cuda.device(x_tensor.device), _torch_stream_context(current_stream, x_tensor.device):
        output_tensor = torch.empty_like(x_tensor, memory_format=torch.contiguous_format)
        if output_final_state:
            n_channels = x_tensor.shape[2]
            final_state_tensor = torch.empty(
                (num_sequences, n_channels, 4),
                dtype=torch.bfloat16,
                device=x_tensor.device,
            )
        else:
            final_state_tensor = torch.empty((0,), dtype=torch.bfloat16, device=x_tensor.device)

        api = _API_CACHE.get(key)
        if api is None:
            with _API_CACHE_LOCK:
                api = _API_CACHE.get(key)
                if api is None:
                    api = CausalConv1dBulkFwdSm100(
                        sample_x=x_tensor,
                        sample_weight=weight_tensor,
                        sample_output=output_tensor,
                        sample_cu_seqlens=cu_seqlens_tensor,
                        sample_initial_state=initial_state_tensor,
                        sample_final_state=final_state_tensor if output_final_state else None,
                    )
                    api.check_support()
                    api.compile()
                    if len(_API_CACHE) >= _API_CACHE_CAPACITY:
                        _API_CACHE.pop(next(iter(_API_CACHE)))
                    _API_CACHE[key] = api

        result = api.execute(
            x_tensor=x_tensor,
            weight_tensor=weight_tensor,
            output_tensor=output_tensor,
            cu_seqlens_tensor=cu_seqlens_tensor,
            initial_state_tensor=initial_state_tensor,
            final_state_tensor=final_state_tensor if output_final_state else None,
            current_stream=current_stream,
        )
        if not output_final_state:
            result["final_state_tensor"] = final_state_tensor
        return result
