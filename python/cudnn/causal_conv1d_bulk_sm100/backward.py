# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental bulk causal-convolution backward API.

The prototype covers the same contiguous BF16, width-four, fused-SiLU data
slice as the forward API. It supports dense [B, T, D] input and packed
[1, total_T, D] input with device cu_seqlens. Optional BF16 bias [D] produces
an FP32 dbias accumulator. Optional BF16 initial state [N, D, 4] produces a
BF16 d_initial_state, and an optional BF16 upstream d_final_state contributes
to both dX and d_initial_state. Residual and other widths are intentionally
outside this surface.

Compilation is exact-shape. Packed metadata values stay on device: a small
pre-kernel validates the prefix sums and builds a bounded per-sequence tile
map, after which the dense backward math is reused without cross-sequence
reads. The default schedules accumulate dweight with FP32 atomics; partial
schedules use caller-owned FP32 partials and a second reduction. The atomic
schedules do not produce bitwise-reproducible dweight, even when PyTorch
deterministic algorithms are enabled; request ``t64-partial`` when reproducible
dweight is required. ``v4-stream`` is a dense, bias-free FE-native schedule
derived from the operator math: each thread streams four adjacent channels,
and a single occupancy formula chooses its token tile instead of maintaining a
config zoo.
``v2-cpasync`` keeps the same workspace/reducer ABI while using four G8 shared-
memory stages, packed-FP32x2 math, and a fast-tanh SiLU derivative on capable
architectures. Its token grid is derived from the live SM count.
Optional dbias uses one FP32 atomic per token-tile and channel in the scalar
schedules.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from cuda.bindings import driver as cuda
from cutlass import cute
from cutlass.cute.runtime import make_fake_stream

from cudnn._causal_conv1d_arch import (
    F32X2_COMPUTE_CAPABILITIES,
    FUNCTIONAL_COMPUTE_CAPABILITIES,
    is_functional_arch,
)
from cudnn.api_base import APIBase, TensorDesc
from cudnn.frost.buffers import CUTEDSL_MIN_VERSION, cutedsl_state, cutedsl_too_old

from .api import (
    _INT32_MAX,
    _MAX_PACKED_SEQUENCES,
    _MAX_SCALAR_GRID_CHANNELS,
    _as_torch_stream,
    _record_streams,
    _require_storage_alignment,
    _tensors_overlap,
)

_WIDTH = 4
_DWEIGHT_ATOMIC = "atomic"
_DWEIGHT_PARTIAL = "partial"
_AUTO_SCHEDULE = "auto"
_SCALAR_KERNEL = "scalar"
_VEC4_STREAM_KERNEL = "vec4-stream"
_VEC2_CPASYNC_KERNEL = "vec2-cpasync"
_VEC4_CHANNELS_PER_CTA = 128 * 4
_VEC4_TARGET_CTAS_PER_SM = 5
_VEC2_CPASYNC_CHANNELS_PER_CTA = 128 * 2
_VEC2_CPASYNC_TOKENS_PER_STAGE = 8
_VEC2_CPASYNC_TARGET_CTAS_PER_SM = 4
_VEC2_CPASYNC_MIN_TOKENS = 64


class _BulkBwdSchedule(NamedTuple):
    threads: int
    tokens_per_cta: int
    dweight_mode: str
    reduction_threads: int = 256
    kernel_variant: str = _SCALAR_KERNEL


_SCHEDULES = {
    "t32": _BulkBwdSchedule(256, 32, _DWEIGHT_ATOMIC),
    "t64": _BulkBwdSchedule(256, 64, _DWEIGHT_ATOMIC),
    "t128": _BulkBwdSchedule(256, 128, _DWEIGHT_ATOMIC),
    "t64-partial": _BulkBwdSchedule(256, 64, _DWEIGHT_PARTIAL),
    # tokens_per_cta is shape/device-derived during check_support().
    "v4-stream": _BulkBwdSchedule(128, 0, _DWEIGHT_PARTIAL, 128, _VEC4_STREAM_KERNEL),
    "v2-cpasync": _BulkBwdSchedule(128, 0, _DWEIGHT_PARTIAL, 128, _VEC2_CPASYNC_KERNEL),
}


def select_bulk_bwd_schedule(total_tokens: int) -> str:
    """Choose the measured atomic schedule for the dense prototype.

    Both returned schedules use FP32 atomics, so dweight is not bitwise
    reproducible across launches, including when PyTorch deterministic
    algorithms are enabled. Request ``t64-partial`` explicitly when
    reproducible dweight is required.
    """

    return "t128" if total_tokens >= 16384 else "t64"


def _align_vec2_cpasync_tile(tokens_per_cta: int) -> int:
    """Round K up so every staged token after the three-token prime is G8."""

    staged_tokens = max(0, tokens_per_cta - 3)
    return 3 + (staged_tokens + _VEC2_CPASYNC_TOKENS_PER_STAGE - 1) // _VEC2_CPASYNC_TOKENS_PER_STAGE * _VEC2_CPASYNC_TOKENS_PER_STAGE


def _plan_vec2_cpasync(sequence_length: int, n_channels: int, sm_count: int) -> tuple[int, int]:
    """Derive the cp.async token grid from live SM count and exact shape."""

    channel_ctas = n_channels // _VEC2_CPASYNC_CHANNELS_PER_CTA
    resident_cta_budget = sm_count * _VEC2_CPASYNC_TARGET_CTAS_PER_SM
    target_token_ctas = max(1, min(sequence_length // 32, resident_cta_budget // channel_ctas))
    tokens_per_cta = _align_vec2_cpasync_tile((sequence_length + target_token_ctas - 1) // target_token_ctas)
    token_ctas = (sequence_length + tokens_per_cta - 1) // tokens_per_cta
    while channel_ctas * token_ctas > resident_cta_budget and token_ctas > 1:
        tokens_per_cta = _align_vec2_cpasync_tile(tokens_per_cta + 1)
        token_ctas = (sequence_length + tokens_per_cta - 1) // tokens_per_cta
    # Every CTA primes exactly three tokens before entering the staged loop.
    # Grow K in whole G8 stages if an arbitrary T would otherwise leave a
    # one- or two-token final CTA. This can only reduce the grid, so the
    # resident-wave budget established above remains valid.
    while token_ctas > 1 and sequence_length - (token_ctas - 1) * tokens_per_cta < 3:
        tokens_per_cta += _VEC2_CPASYNC_TOKENS_PER_STAGE
        token_ctas = (sequence_length + tokens_per_cta - 1) // tokens_per_cta
    return token_ctas, tokens_per_cta


class CausalConv1dBulkBwdPrototype(APIBase):
    """Compile one exact-shape dense or packed BF16+SiLU backward."""

    def __init__(
        self,
        sample_x: torch.Tensor,
        sample_weight: torch.Tensor,
        sample_dy: torch.Tensor,
        sample_cu_seqlens: torch.Tensor | None = None,
        *,
        schedule: str = _AUTO_SCHEDULE,
        sample_bias: torch.Tensor | None = None,
        sample_initial_state: torch.Tensor | None = None,
        sample_d_final_state: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self._warn_experimental_api()
        for name, sample in (
            ("sample_x", sample_x),
            ("sample_weight", sample_weight),
            ("sample_dy", sample_dy),
        ):
            if not isinstance(sample, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(sample).__name__}")
        if sample_cu_seqlens is not None and not isinstance(sample_cu_seqlens, torch.Tensor):
            raise TypeError("sample_cu_seqlens must be a torch.Tensor or None, " f"got {type(sample_cu_seqlens).__name__}")
        if sample_initial_state is not None and not isinstance(sample_initial_state, torch.Tensor):
            raise TypeError("sample_initial_state must be a torch.Tensor or None, " f"got {type(sample_initial_state).__name__}")
        if sample_d_final_state is not None and not isinstance(sample_d_final_state, torch.Tensor):
            raise TypeError("sample_d_final_state must be a torch.Tensor or None, " f"got {type(sample_d_final_state).__name__}")
        if sample_bias is not None and not isinstance(sample_bias, torch.Tensor):
            raise TypeError(f"sample_bias must be a torch.Tensor or None, got {type(sample_bias).__name__}")
        choices = (_AUTO_SCHEDULE, *_SCHEDULES)
        if schedule not in choices:
            raise ValueError(f"unknown schedule {schedule!r}; choose one of {choices}")
        self.requested_schedule = schedule
        self.schedule: str | None = None

        self.x_desc = self._make_tensor_desc(sample_x, name="sample_x")
        self.weight_desc = self._make_tensor_desc(sample_weight, name="sample_weight")
        self.dy_desc = self._make_tensor_desc(sample_dy, name="sample_dy")
        self.cu_seqlens_desc = self._make_tensor_desc(sample_cu_seqlens, name="sample_cu_seqlens")
        self.initial_state_desc = self._make_tensor_desc(sample_initial_state, name="sample_initial_state")
        self.d_final_state_desc = self._make_tensor_desc(sample_d_final_state, name="sample_d_final_state")
        self.bias_desc = self._make_tensor_desc(sample_bias, name="sample_bias")
        self._sample_alignment_remainders = {
            "X": sample_x.data_ptr() % 16,
            "Weight": sample_weight.data_ptr() % 16,
            "dY": sample_dy.data_ptr() % 16,
        }
        if sample_cu_seqlens is not None:
            self._sample_alignment_remainders["cu_seqlens"] = sample_cu_seqlens.data_ptr() % 4
        if sample_initial_state is not None:
            self._sample_alignment_remainders["Initial state"] = sample_initial_state.data_ptr() % 16
        if sample_d_final_state is not None:
            self._sample_alignment_remainders["dFinal state"] = sample_d_final_state.data_ptr() % 16
        if sample_bias is not None:
            self._sample_alignment_remainders["Bias"] = sample_bias.data_ptr() % 16

        self.is_packed = sample_cu_seqlens is not None
        self.batch_size: int | None = None
        self.sequence_length: int | None = None
        self.n_channels: int | None = None
        self.num_sequences: int | None = None
        self.total_tokens: int | None = None
        self.threads: int | None = None
        self.tokens_per_cta: int | None = None
        self.dweight_mode: str | None = None
        self.reduction_threads: int | None = None
        self.kernel_variant: str | None = None
        self.tiles_per_sequence: int | None = None
        self.packed_tile_capacity = 0
        self.num_dweight_partials: int | None = None
        self.compute_capability: tuple[int, int] | None = None
        self.sm_count: int | None = None

    @staticmethod
    def _require_rank(desc: TensorDesc, rank: int, name: str) -> None:
        if desc.ndim != rank:
            raise ValueError(f"{name} must be {rank}D, got shape {desc.shape}")

    @staticmethod
    def _require_cuda(desc: TensorDesc, name: str) -> None:
        if desc.device.type != "cuda":
            raise ValueError(f"{name} must be a CUDA tensor, got device {desc.device}")

    @property
    def packed_tile_map_numel(self) -> int:
        """Number of int32 elements in the device-built packed tile map."""

        return self.packed_tile_capacity * _WIDTH

    @property
    def packed_tile_map_bytes(self) -> int:
        return self.packed_tile_map_numel * 4

    @property
    def dweight_workspace_numel(self) -> int:
        """Number of FP32 elements required by the selected dweight mode."""

        if self.dweight_mode != _DWEIGHT_PARTIAL:
            return 0
        assert self.num_dweight_partials is not None
        assert self.n_channels is not None
        return self.num_dweight_partials * self.n_channels * _WIDTH

    @property
    def dweight_workspace_bytes(self) -> int:
        return self.dweight_workspace_numel * 4

    @property
    def total_workspace_bytes(self) -> int:
        """Total caller-owned tile-map and dweight workspace in bytes."""

        return self.packed_tile_map_bytes + self.dweight_workspace_bytes

    def check_support(self) -> bool:
        cutedsl_installed, cutedsl_version = cutedsl_state()
        cutedsl_floor = ".".join(str(component) for component in CUTEDSL_MIN_VERSION)
        self._runtime_error_if(
            not cutedsl_installed,
            "causal_conv1d_bulk backward requires the cutedsl extra " f"(nvidia-cutlass-dsl>={cutedsl_floor})",
        )
        if cutedsl_too_old(cutedsl_version):
            self._runtime_error_if(
                True,
                "causal_conv1d_bulk backward requires " f"nvidia-cutlass-dsl>={cutedsl_floor}; found {cutedsl_version[1]}",
            )

        self._require_rank(self.x_desc, 3, "X")
        self._require_rank(self.weight_desc, 2, "Weight")
        self._require_rank(self.dy_desc, 3, "dY")
        if self.bias_desc is not None:
            self._require_rank(self.bias_desc, 1, "Bias")
        if self.cu_seqlens_desc is not None:
            self._require_rank(self.cu_seqlens_desc, 1, "cu_seqlens")
        if self.initial_state_desc is not None:
            self._require_rank(self.initial_state_desc, 3, "Initial state")
        if self.d_final_state_desc is not None:
            self._require_rank(self.d_final_state_desc, 3, "dFinal state")

        batch_size, sequence_length, n_channels = self.x_desc.shape
        self._value_error_if(batch_size <= 0, f"B must be positive, got {batch_size}")
        self._value_error_if(sequence_length <= 0, f"T must be positive, got {sequence_length}")
        self._value_error_if(n_channels <= 0, f"D must be positive, got {n_channels}")
        total_tokens = batch_size * sequence_length
        self._value_error_if(batch_size > _INT32_MAX, f"B exceeds the Int32 limit: {batch_size}")
        self._value_error_if(
            sequence_length > _INT32_MAX,
            f"T exceeds the Int32 limit: {sequence_length}",
        )
        self._value_error_if(
            total_tokens > _INT32_MAX,
            f"B*T exceeds the Int32 limit: {total_tokens}",
        )
        self._value_error_if(
            n_channels > _MAX_SCALAR_GRID_CHANNELS,
            f"D exceeds the supported CUDA grid limit: {n_channels}",
        )

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
                f"Packed N exceeds the safe Int32 limit: {num_sequences}",
            )
            self._value_error_if(
                num_sequences > total_tokens,
                f"Packed N={num_sequences} cannot exceed total_T={total_tokens} " "when every sequence must be non-empty",
            )

        self._check_tensor_shape(self.weight_desc, (n_channels, _WIDTH), "Weight")
        self._check_tensor_shape(self.dy_desc, (batch_size, sequence_length, n_channels), "dY")
        if self.bias_desc is not None:
            self._check_tensor_shape(self.bias_desc, (n_channels,), "Bias")
        if self.cu_seqlens_desc is not None:
            self._check_tensor_shape(self.cu_seqlens_desc, (num_sequences + 1,), "cu_seqlens")
        state_shape = (num_sequences, n_channels, _WIDTH)
        if self.initial_state_desc is not None:
            self._check_tensor_shape(self.initial_state_desc, state_shape, "Initial state")
        if self.d_final_state_desc is not None:
            self._check_tensor_shape(self.d_final_state_desc, state_shape, "dFinal state")

        expected_data_stride = (sequence_length * n_channels, n_channels, 1)
        self._check_tensor_stride(
            self.x_desc,
            stride=expected_data_stride,
            name="X",
            extra_error_msg="X must be [B, T, D] contiguous",
        )
        self._check_tensor_stride(
            self.dy_desc,
            stride=expected_data_stride,
            name="dY",
            extra_error_msg="dY must be [B, T, D] contiguous",
        )
        self._check_tensor_stride(
            self.weight_desc,
            stride=(_WIDTH, 1),
            name="Weight",
            extra_error_msg="Weight must be [D, 4] contiguous",
        )
        if self.bias_desc is not None:
            self._check_tensor_stride(
                self.bias_desc,
                stride=(1,),
                name="Bias",
                extra_error_msg="Bias must be [D] contiguous",
            )
        if self.cu_seqlens_desc is not None:
            self._check_tensor_stride(self.cu_seqlens_desc, stride=(1,), name="cu_seqlens")
        for name, desc in (
            ("Initial state", self.initial_state_desc),
            ("dFinal state", self.d_final_state_desc),
        ):
            if desc is not None:
                self._check_tensor_stride(
                    desc,
                    stride=(n_channels * _WIDTH, _WIDTH, 1),
                    name=name,
                    extra_error_msg=f"{name} must be [N, D, 4] contiguous",
                )

        for name, desc in (
            ("X", self.x_desc),
            ("dY", self.dy_desc),
            ("Initial state", self.initial_state_desc),
            ("dFinal state", self.d_final_state_desc),
        ):
            if desc is not None:
                self._check_dtype(desc, dtype=torch.bfloat16, name=name)
        self._value_error_if(
            self.weight_desc.dtype not in (torch.bfloat16, torch.float32),
            "Weight must have dtype torch.bfloat16 or torch.float32, " f"got {self.weight_desc.dtype}",
        )
        if self.bias_desc is not None:
            self._value_error_if(
                self.weight_desc.dtype != torch.bfloat16,
                "FP32 Weight is currently supported only when Bias is omitted",
            )
            self._check_dtype(self.bias_desc, dtype=torch.bfloat16, name="Bias")
        if self.cu_seqlens_desc is not None:
            self._check_dtype(self.cu_seqlens_desc, dtype=torch.int32, name="cu_seqlens")

        for name, remainder in self._sample_alignment_remainders.items():
            alignment = 4 if name == "cu_seqlens" else 16
            self._value_error_if(
                remainder != 0,
                f"{name} data pointer must be {alignment}-byte aligned, " f"got address modulo {alignment} = {remainder}",
            )

        for name, desc in (
            ("X", self.x_desc),
            ("Weight", self.weight_desc),
            ("Bias", self.bias_desc),
            ("dY", self.dy_desc),
            ("cu_seqlens", self.cu_seqlens_desc),
            ("Initial state", self.initial_state_desc),
            ("dFinal state", self.d_final_state_desc),
        ):
            if desc is None:
                continue
            self._require_cuda(desc, name)
            self._value_error_if(
                desc.device != self.x_desc.device,
                f"{name} must be on {self.x_desc.device}, got {desc.device}",
            )

        self._runtime_error_if(not torch.cuda.is_available(), "CUDA is not available")
        compute_capability = torch.cuda.get_device_capability(self.x_desc.device)
        self._runtime_error_if(
            not is_functional_arch(compute_capability),
            "CausalConv1dBulkBwdPrototype does not support compute capability "
            f"{compute_capability[0]}.{compute_capability[1]}; supported capabilities are "
            f"{sorted(FUNCTIONAL_COMPUTE_CAPABILITIES)}",
        )
        self._value_error_if(
            total_tokens * n_channels > _INT32_MAX,
            "The scalar backward schedule requires X, dY, and dX to contain at most " f"{_INT32_MAX} elements",
        )
        if self.initial_state_desc is not None or self.d_final_state_desc is not None:
            self._value_error_if(
                num_sequences * n_channels * _WIDTH > _INT32_MAX,
                "The scalar backward schedule requires each state tensor to contain at most " f"{_INT32_MAX} elements",
            )

        schedule_extent = total_tokens if self.is_packed else sequence_length
        auto_streaming = (
            self.requested_schedule == _AUTO_SCHEDULE
            and not self.is_packed
            and batch_size == 1
            and self.bias_desc is None
            and self.initial_state_desc is None
            and self.d_final_state_desc is None
            and n_channels % _VEC4_CHANNELS_PER_CTA == 0
            and sequence_length >= 8192
        )
        if auto_streaming:
            schedule = "v2-cpasync" if compute_capability in F32X2_COMPUTE_CAPABILITIES else "v4-stream"
        else:
            schedule = select_bulk_bwd_schedule(schedule_extent) if self.requested_schedule == _AUTO_SCHEDULE else self.requested_schedule
        config = _SCHEDULES[schedule]
        kernel_variant = config.kernel_variant
        sm_count = None
        if config.kernel_variant in (_VEC4_STREAM_KERNEL, _VEC2_CPASYNC_KERNEL):
            self._value_error_if(self.is_packed, f"{schedule} currently supports dense input only")
            self._value_error_if(batch_size != 1, f"{schedule} requires B=1, got B={batch_size}")
            self._value_error_if(self.bias_desc is not None, f"{schedule} does not yet support bias/dbias")
            self._value_error_if(
                self.initial_state_desc is not None or self.d_final_state_desc is not None,
                f"{schedule} does not yet support initial-state or final-state gradients",
            )
            if config.kernel_variant == _VEC2_CPASYNC_KERNEL:
                self._value_error_if(
                    compute_capability not in F32X2_COMPUTE_CAPABILITIES,
                    f"v2-cpasync requires packed-f32x2 support, got compute capability {compute_capability}",
                )
                self._value_error_if(
                    n_channels % _VEC2_CPASYNC_CHANNELS_PER_CTA != 0,
                    f"v2-cpasync requires D divisible by {_VEC2_CPASYNC_CHANNELS_PER_CTA}, got D={n_channels}",
                )
                self._value_error_if(
                    sequence_length < _VEC2_CPASYNC_MIN_TOKENS,
                    f"v2-cpasync requires T>={_VEC2_CPASYNC_MIN_TOKENS}, got T={sequence_length}",
                )
                sm_count = torch.cuda.get_device_properties(self.x_desc.device).multi_processor_count
                tiles_per_sequence, tokens_per_cta = _plan_vec2_cpasync(sequence_length, n_channels, sm_count)
                last_tile_tokens = sequence_length - (tiles_per_sequence - 1) * tokens_per_cta
                self._value_error_if(last_tile_tokens < 3, f"v2-cpasync planner produced an unsafe final tile of {last_tile_tokens} tokens")
            else:
                self._value_error_if(
                    n_channels % _VEC4_CHANNELS_PER_CTA != 0,
                    f"v4-stream requires D divisible by {_VEC4_CHANNELS_PER_CTA}, got D={n_channels}",
                )
                self._value_error_if(sequence_length < 16, f"v4-stream requires T>=16, got T={sequence_length}")
                sm_count = torch.cuda.get_device_properties(self.x_desc.device).multi_processor_count
                channel_ctas = n_channels // _VEC4_CHANNELS_PER_CTA
                target_token_ctas = max(1, round(sm_count * _VEC4_TARGET_CTAS_PER_SM / channel_ctas))
                target_token_ctas = min(sequence_length // 16, target_token_ctas)
                tokens_per_cta = ((sequence_length + target_token_ctas - 1) // target_token_ctas + 7) // 8 * 8
                # The streaming kernel primes three dz values per tile. Fold a
                # one- or two-token remainder into the preceding tile instead
                # of admitting an undersized final tile.
                while sequence_length - ((sequence_length - 1) // tokens_per_cta) * tokens_per_cta < 3:
                    tokens_per_cta += 1
                tiles_per_sequence = (sequence_length + tokens_per_cta - 1) // tokens_per_cta
            packed_tile_capacity = 0
            num_dweight_partials = tiles_per_sequence
        else:
            tokens_per_cta = config.tokens_per_cta
            tiles_per_sequence = (sequence_length - 1) // tokens_per_cta + 1
            if self.is_packed:
                # Tight shape-only upper bound for sum_i ceil(length_i / K) when
                # every sequence is non-empty. It avoids a host read of offsets.
                packed_tile_capacity = num_sequences + (total_tokens - num_sequences) // tokens_per_cta
                num_dweight_partials = packed_tile_capacity
            else:
                packed_tile_capacity = 0
                num_dweight_partials = batch_size * tiles_per_sequence

        self.schedule = schedule
        self.threads = config.threads
        self.tokens_per_cta = tokens_per_cta
        self.dweight_mode = config.dweight_mode
        self.reduction_threads = config.reduction_threads
        self.kernel_variant = kernel_variant
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.n_channels = n_channels
        self.num_sequences = num_sequences
        self.total_tokens = total_tokens
        self.tiles_per_sequence = tiles_per_sequence
        self.packed_tile_capacity = packed_tile_capacity
        self.num_dweight_partials = num_dweight_partials
        self.compute_capability = compute_capability
        self.sm_count = sm_count
        self._is_supported = True
        return True

    def compile(self) -> None:
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        assert self.total_tokens is not None
        assert self.n_channels is not None
        assert self.batch_size is not None
        assert self.sequence_length is not None
        assert self.num_sequences is not None
        assert self.threads is not None
        assert self.tokens_per_cta is not None
        assert self.reduction_threads is not None

        fake_x = self._make_fake_cute_tensor(
            dtype=torch.bfloat16,
            shape=(self.total_tokens, self.n_channels),
            stride=(self.n_channels, 1),
            assumed_align=16,
        )
        fake_weight = self._make_fake_cute_tensor_from_desc(self.weight_desc, assumed_align=16)
        fake_bias = self._make_fake_cute_tensor_from_desc(self.bias_desc, assumed_align=16)
        fake_dy = self._make_fake_cute_tensor(
            dtype=torch.bfloat16,
            shape=(self.total_tokens, self.n_channels),
            stride=(self.n_channels, 1),
            assumed_align=16,
        )
        fake_dx = self._make_fake_cute_tensor(
            dtype=torch.bfloat16,
            shape=(self.total_tokens, self.n_channels),
            stride=(self.n_channels, 1),
            assumed_align=16,
        )
        fake_dw = self._make_fake_cute_tensor(
            dtype=torch.float32,
            shape=(self.n_channels, _WIDTH),
            stride=(_WIDTH, 1),
            assumed_align=16,
        )
        fake_db = None
        if self.bias_desc is not None:
            fake_db = self._make_fake_cute_tensor(
                dtype=torch.float32,
                shape=(self.n_channels,),
                stride=(1,),
                assumed_align=16,
            )
        fake_dw_partials = None
        if self.dweight_mode == _DWEIGHT_PARTIAL:
            fake_dw_partials = self._make_fake_cute_tensor(
                dtype=torch.float32,
                shape=(self.dweight_workspace_numel,),
                stride=(1,),
                assumed_align=16,
            )
        fake_cu_seqlens = self._make_fake_cute_tensor_from_desc(self.cu_seqlens_desc, assumed_align=4)
        fake_initial_state = self._make_fake_cute_tensor_from_desc(self.initial_state_desc, assumed_align=16)
        fake_d_final_state = self._make_fake_cute_tensor_from_desc(self.d_final_state_desc, assumed_align=16)
        fake_d_initial_state = None
        if self.initial_state_desc is not None:
            fake_d_initial_state = self._make_fake_cute_tensor_from_desc(self.initial_state_desc, assumed_align=16)
        fake_packed_tile_map = None
        if self.is_packed:
            fake_packed_tile_map = self._make_fake_cute_tensor(
                dtype=torch.int32,
                shape=(self.packed_tile_map_numel,),
                stride=(1,),
                assumed_align=16,
            )
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
        if self.kernel_variant == _VEC2_CPASYNC_KERNEL:
            from .backward_kernel_vec2_cpasync import CausalConv1dBulkBackwardVec2CpAsyncKernel

            kernel = CausalConv1dBulkBackwardVec2CpAsyncKernel(
                sequence_length=self.sequence_length,
                n_channels=self.n_channels,
                tokens_per_cta=self.tokens_per_cta,
                n_token_tiles=self.num_dweight_partials,
                reduction_threads=self.reduction_threads,
            )
        elif self.kernel_variant == _VEC4_STREAM_KERNEL:
            from .backward_kernel_vec4 import CausalConv1dBulkBackwardVec4Kernel

            kernel = CausalConv1dBulkBackwardVec4Kernel(
                sequence_length=self.sequence_length,
                n_channels=self.n_channels,
                tokens_per_cta=self.tokens_per_cta,
                n_token_tiles=self.num_dweight_partials,
                reduction_threads=self.reduction_threads,
            )
        else:
            from .backward_kernel import CausalConv1dBulkBackwardKernel

            kernel = CausalConv1dBulkBackwardKernel(
                batch_size=self.batch_size,
                sequence_length=self.sequence_length,
                num_sequences=self.num_sequences,
                packed_tile_capacity=self.packed_tile_capacity,
                threads=self.threads,
                tokens_per_cta=self.tokens_per_cta,
                use_dweight_partials=self.dweight_mode == _DWEIGHT_PARTIAL,
                reduction_threads=self.reduction_threads,
            )
        with torch.cuda.device(self.x_desc.device):
            self._compiled_kernel = cute.compile(
                kernel,
                fake_x,
                fake_weight,
                fake_bias,
                fake_dy,
                fake_dx,
                fake_dw,
                fake_db,
                fake_dw_partials,
                fake_cu_seqlens,
                fake_initial_state,
                fake_d_final_state,
                fake_d_initial_state,
                fake_packed_tile_map,
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
        x: torch.Tensor,
        weight: torch.Tensor,
        dy: torch.Tensor,
        dx: torch.Tensor,
        dw_accum: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        packed_tile_map: torch.Tensor | None = None,
        dweight_workspace: torch.Tensor | None = None,
        current_stream: cuda.CUstream | None = None,
        bias: torch.Tensor | None = None,
        db_accum: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        d_final_state: torch.Tensor | None = None,
        d_initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._compiled_kernel is None:
            raise RuntimeError("compile() must be called before execute()")
        self._validate_runtime_tensor(x, self.x_desc, "X")
        self._validate_runtime_tensor(weight, self.weight_desc, "Weight")
        self._validate_runtime_tensor(dy, self.dy_desc, "dY")
        self._validate_runtime_tensor(dx, self.x_desc, "dX")
        if (bias is None) != (self.bias_desc is None):
            raise ValueError("Bias presence must match the compiled signature")
        if bias is not None:
            assert self.bias_desc is not None
            self._validate_runtime_tensor(bias, self.bias_desc, "Bias")
        if (db_accum is None) != (self.bias_desc is None):
            raise ValueError("dBias accumulator presence must match the compiled signature")
        assert self.n_channels is not None
        expected_dw = (self.n_channels, _WIDTH)
        if not isinstance(dw_accum, torch.Tensor):
            raise TypeError("dW accumulator must be a torch.Tensor, " f"got {type(dw_accum).__name__}")
        if tuple(dw_accum.shape) != expected_dw or tuple(dw_accum.stride()) != (_WIDTH, 1) or dw_accum.dtype != torch.float32 or dw_accum.device != x.device:
            raise ValueError(f"dW accumulator must be contiguous FP32 {expected_dw} on {x.device}")
        if db_accum is not None:
            if not isinstance(db_accum, torch.Tensor):
                raise TypeError("dBias accumulator must be a torch.Tensor, " f"got {type(db_accum).__name__}")
            if (
                tuple(db_accum.shape) != (self.n_channels,)
                or tuple(db_accum.stride()) != (1,)
                or db_accum.dtype != torch.float32
                or db_accum.device != x.device
            ):
                raise ValueError(f"dBias accumulator must be contiguous FP32 ({self.n_channels},) on {x.device}")

        for name, tensor, desc in (
            ("Initial state", initial_state, self.initial_state_desc),
            ("dFinal state", d_final_state, self.d_final_state_desc),
        ):
            if (tensor is None) != (desc is None):
                raise ValueError(f"{name} presence must match the compiled signature")
            if tensor is not None:
                assert desc is not None
                self._validate_runtime_tensor(tensor, desc, name)
        if (d_initial_state is None) != (self.initial_state_desc is None):
            raise ValueError("dInitial state presence must match the compiled initial-state signature")
        if d_initial_state is not None:
            assert self.initial_state_desc is not None
            self._validate_runtime_tensor(d_initial_state, self.initial_state_desc, "dInitial state")

        if (cu_seqlens is None) != (self.cu_seqlens_desc is None):
            raise ValueError("cu_seqlens presence must match the compiled signature")
        if cu_seqlens is not None:
            assert self.cu_seqlens_desc is not None
            self._validate_runtime_tensor(cu_seqlens, self.cu_seqlens_desc, "cu_seqlens")

        if self.is_packed:
            if packed_tile_map is None:
                raise ValueError(f"packed backward requires {self.packed_tile_map_bytes} bytes " "of int32 tile-map workspace")
            if (
                packed_tile_map.dtype != torch.int32
                or not packed_tile_map.is_contiguous()
                or packed_tile_map.numel() != self.packed_tile_map_numel
                or packed_tile_map.device != x.device
            ):
                raise ValueError("packed tile map must be contiguous int32 " f"[{self.packed_tile_map_numel}] on {x.device}")
        elif packed_tile_map is not None:
            raise ValueError("dense backward does not consume a packed tile map")

        if self.dweight_mode == _DWEIGHT_PARTIAL:
            if dweight_workspace is None:
                raise ValueError(f"schedule {self.schedule!r} requires " f"{self.dweight_workspace_bytes} bytes of FP32 dweight workspace")
            if (
                dweight_workspace.dtype != torch.float32
                or not dweight_workspace.is_contiguous()
                or dweight_workspace.numel() != self.dweight_workspace_numel
                or dweight_workspace.device != x.device
            ):
                raise ValueError("dweight workspace must be contiguous FP32 " f"[{self.dweight_workspace_numel}] on {x.device}")
        elif dweight_workspace is not None:
            raise ValueError(f"schedule {self.schedule!r} does not consume dweight workspace")

        named_tensors = (
            ("X", x, False),
            ("Weight", weight, False),
            ("Bias", bias, False),
            ("dY", dy, False),
            ("Initial state", initial_state, False),
            ("dFinal state", d_final_state, False),
            ("cu_seqlens", cu_seqlens, False),
            ("dX", dx, True),
            ("dW accumulator", dw_accum, True),
            ("dBias accumulator", db_accum, True),
            ("dInitial state", d_initial_state, True),
            ("Packed tile map", packed_tile_map, True),
            ("dW workspace", dweight_workspace, True),
        )
        for index, (left_name, left, left_writable) in enumerate(named_tensors):
            if left is None:
                continue
            for right_name, right, right_writable in named_tensors[index + 1 :]:
                if right is None or not (left_writable or right_writable):
                    continue
                if _tensors_overlap(left, right):
                    raise ValueError(f"{left_name} and {right_name} must not overlap")

        for name, tensor, alignment in (
            ("X", x, 16),
            ("Weight", weight, 16),
            ("Bias", bias, 16),
            ("dY", dy, 16),
            ("Initial state", initial_state, 16),
            ("dFinal state", d_final_state, 16),
            ("dX", dx, 16),
            ("dW accumulator", dw_accum, 16),
            ("dBias accumulator", db_accum, 16),
            ("dInitial state", d_initial_state, 16),
            ("cu_seqlens", cu_seqlens, 4),
            ("Packed tile map", packed_tile_map, 16),
            ("dW workspace", dweight_workspace, 16),
        ):
            if tensor is not None:
                _require_storage_alignment(tensor, alignment, name)

        if current_stream is None:
            torch_stream = torch.cuda.current_stream(x.device)
            launch_stream = cuda.CUstream(torch_stream.cuda_stream)
        else:
            launch_stream = current_stream
            torch_stream = _as_torch_stream(current_stream, x.device)
        if self.dweight_mode == _DWEIGHT_ATOMIC or db_accum is not None:
            with torch.cuda.stream(torch_stream):
                if self.dweight_mode == _DWEIGHT_ATOMIC:
                    dw_accum.zero_()
                if db_accum is not None:
                    db_accum.zero_()
        self._compiled_kernel(
            x.view(-1, self.n_channels),
            weight,
            bias,
            dy.view(-1, self.n_channels),
            dx.view(-1, self.n_channels),
            dw_accum,
            db_accum,
            dweight_workspace,
            cu_seqlens,
            initial_state,
            d_final_state,
            d_initial_state,
            packed_tile_map,
            launch_stream,
        )
        _record_streams(
            (
                x,
                weight,
                bias,
                dy,
                initial_state,
                d_final_state,
                dx,
                dw_accum,
                db_accum,
                d_initial_state,
                cu_seqlens,
                packed_tile_map,
                dweight_workspace,
            ),
            torch_stream,
        )
        return dx, dw_accum


def compile_causal_conv1d_bulk_bwd_prototype(
    x: torch.Tensor,
    weight: torch.Tensor,
    dy: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    *,
    schedule: str = _AUTO_SCHEDULE,
    bias: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    d_final_state: torch.Tensor | None = None,
) -> CausalConv1dBulkBwdPrototype:
    """Construct, support-check, and compile an exact-shape prototype."""

    api = CausalConv1dBulkBwdPrototype(
        x,
        weight,
        dy,
        sample_cu_seqlens=cu_seqlens,
        sample_initial_state=initial_state,
        sample_d_final_state=d_final_state,
        schedule=schedule,
        sample_bias=bias,
    )
    api.check_support()
    api.compile()
    return api


__all__ = [
    "CausalConv1dBulkBwdPrototype",
    "compile_causal_conv1d_bulk_bwd_prototype",
    "select_bulk_bwd_schedule",
]
