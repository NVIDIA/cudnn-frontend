# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Python API surface for fused SwiGLU MoE with expert parallelism.

The public contract is present before the device implementation so callers and
the PyTorch reference can be wired into tests.  ``MoeEp.__call__`` currently
allocates output storage but does not launch a backend kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist


class MoeFormat(str, Enum):
    """Data formats supported by the MoE+EP interface."""

    BF16 = "bf16"
    MXFP8 = "mxfp8"
    NVFP4 = "nvfp4"


def _parse_format(value: Union[MoeFormat, str]) -> MoeFormat:
    if isinstance(value, MoeFormat):
        return value
    try:
        return MoeFormat(value.lower())
    except (AttributeError, ValueError) as exc:
        choices = ", ".join(item.value for item in MoeFormat)
        raise ValueError(f"unsupported format {value!r}; expected one of: {choices}") from exc


def _require_torch_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if dtype is None:
        raise RuntimeError(f"this PyTorch build does not provide torch.{name}")
    return dtype


def _normalize_axis(axis: int, ndim: int) -> int:
    normalized = axis + ndim if axis < 0 else axis
    if normalized < 0 or normalized >= ndim:
        raise IndexError(f"axis {axis} is out of range for a {ndim}-D tensor")
    return normalized


@dataclass(frozen=True)
class BlockScaledTensor:
    """Data-plus-scale result returned for MXFP8 and NVFP4 outputs."""

    data: torch.Tensor
    scale: torch.Tensor
    format: Union[MoeFormat, str]
    logical_shape: Tuple[int, ...]
    axis: int = -1

    def __post_init__(self) -> None:
        fmt = _parse_format(self.format)
        if fmt is MoeFormat.BF16:
            raise ValueError("BlockScaledTensor only represents mxfp8 or nvfp4")
        logical_shape = tuple(int(dim) for dim in self.logical_shape)
        axis = _normalize_axis(self.axis, len(logical_shape))
        object.__setattr__(self, "format", fmt)
        object.__setattr__(self, "logical_shape", logical_shape)
        object.__setattr__(self, "axis", axis)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.logical_shape

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def block_size(self) -> int:
        return 32 if self.format is MoeFormat.MXFP8 else 16

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Decode the logical, unswizzled block-scaled representation."""

        logical_extent = self.logical_shape[self.axis]
        scale = self.scale.movedim(self.axis, -1).float()
        expanded_scale = scale.repeat_interleave(self.block_size, dim=-1)[..., :logical_extent]

        if self.format is MoeFormat.MXFP8:
            values = self.data.movedim(self.axis, -1).float()
        else:
            packed = self.data.movedim(self.axis, -1)
            low = packed & 0x0F
            high = packed >> 4
            codes = torch.stack((low, high), dim=-1).flatten(-2)[..., :logical_extent]
            table = torch.tensor(
                [
                    0.0,
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                    -0.0,
                    -0.5,
                    -1.0,
                    -1.5,
                    -2.0,
                    -3.0,
                    -4.0,
                    -6.0,
                ],
                dtype=torch.float32,
                device=packed.device,
            )
            values = table[codes.long()]

        return (values * expanded_scale).movedim(-1, self.axis).to(dtype)


MoeTensor = Union[torch.Tensor, BlockScaledTensor]


def _logical_shape(tensor: MoeTensor) -> Tuple[int, ...]:
    if isinstance(tensor, BlockScaledTensor):
        return tensor.logical_shape
    return tuple(tensor.shape)


def _tensor_device(tensor: MoeTensor) -> torch.device:
    return tensor.device


class MoeEp:
    """Fused SwiGLU MoE operator with contiguous expert parallel sharding.

    Global expert ``e`` belongs to group-relative EP rank
    ``e // experts_per_rank``.  The constructor captures static configuration;
    calling the instance accepts runtime tensors for this rank.

    With ``generate_c=True`` (training integration), ``__call__`` additionally
    returns ``fc1_c`` and ``route_metadata``.  ``fc1_c`` is the raw pre-SwiGLU
    FC1 accumulator for every route this rank's experts processed, BF16, shape
    ``(local_routes, 2 * intermediate)``.  Rows are grouped by local expert
    (ascending) and ordered within each expert by source rank, then the source
    rank's token-major route order.  The rows are captured before the gate/up
    clamp and carry no router weight.  ``route_metadata`` is Int32
    ``(local_routes, 4)`` with columns
    ``(local_expert, src_rank, src_token, src_slot)``, row-aligned with
    ``fc1_c``, identifying each route for the backward gradient re-dispatch.

    The backend launch is intentionally a TODO.  Until it is implemented,
    ``__call__`` returns newly allocated, uninitialized output storage with the
    correct public representation.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int,
        ep_group: Optional[dist.ProcessGroup] = None,
        max_tokens_per_rank: Optional[int] = None,
        output_format: Union[MoeFormat, str] = MoeFormat.BF16,
        combine_format: Union[MoeFormat, str] = MoeFormat.BF16,
        apply_topk_in_fc1: bool = True,
        gate_up_clamp: Optional[float] = None,
        generate_c: bool = False,
    ) -> None:
        for name, value in (
            ("num_experts", num_experts),
            ("hidden_size", hidden_size),
            ("intermediate_size", intermediate_size),
            ("top_k", top_k),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) cannot exceed num_experts ({num_experts})")
        if max_tokens_per_rank is not None and max_tokens_per_rank < 0:
            raise ValueError("max_tokens_per_rank must be non-negative")

        if ep_group is None:
            ep_size, ep_rank = 1, 0
        else:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("ep_group requires an initialized torch.distributed process group")
            ep_size = dist.get_world_size(ep_group)
            ep_rank = dist.get_rank(ep_group)
        if num_experts % ep_size != 0:
            raise ValueError(f"num_experts ({num_experts}) must be divisible by EP size ({ep_size})")

        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.ep_group = ep_group
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.experts_per_rank = num_experts // ep_size
        self.max_tokens_per_rank = max_tokens_per_rank
        self.output_format = _parse_format(output_format)
        self.combine_format = _parse_format(combine_format)
        self.apply_topk_in_fc1 = bool(apply_topk_in_fc1)
        self.gate_up_clamp = None if gate_up_clamp is None else abs(float(gate_up_clamp))
        self.generate_c = bool(generate_c)

        for name, fmt in (
            ("output_format", self.output_format),
            ("combine_format", self.combine_format),
        ):
            required_multiple = 32 if fmt is MoeFormat.MXFP8 else 16 if fmt is MoeFormat.NVFP4 else 1
            if hidden_size % required_multiple != 0:
                raise ValueError(f"hidden_size ({hidden_size}) must be divisible by " f"{required_multiple} for {name}={fmt.value}")

    def _allocate_output(self, token_count: int, device: torch.device) -> MoeTensor:
        logical_shape = (token_count, self.hidden_size)
        if self.output_format is MoeFormat.BF16:
            return torch.empty(logical_shape, dtype=torch.bfloat16, device=device)
        if self.output_format is MoeFormat.MXFP8:
            data = torch.empty(
                logical_shape,
                dtype=_require_torch_dtype("float8_e4m3fn"),
                device=device,
            )
            scale = torch.empty(
                (token_count, self.hidden_size // 32),
                dtype=_require_torch_dtype("float8_e8m0fnu"),
                device=device,
            )
        else:
            data = torch.empty(
                (token_count, self.hidden_size // 2),
                dtype=torch.uint8,
                device=device,
            )
            scale = torch.empty(
                (token_count, self.hidden_size // 16),
                dtype=_require_torch_dtype("float8_e4m3fn"),
                device=device,
            )
        return BlockScaledTensor(
            data=data,
            scale=scale,
            format=self.output_format,
            logical_shape=logical_shape,
            axis=-1,
        )

    def _count_local_routes(self, topk_idx: torch.Tensor) -> int:
        """Number of valid routes this rank's experts receive.

        Data-dependent: single-rank counts locally, EP exchanges per-rank
        route counts (the same exchange the device dispatch performs).
        """

        flat = topk_idx.reshape(-1).to(torch.int64)
        expert = flat[flat != -1]
        if expert.numel() > 0 and bool(((expert < 0) | (expert >= self.num_experts)).any().item()):
            raise ValueError("topk_idx contains out-of-range expert ids")
        if self.ep_size == 1:
            return int(expert.numel())
        destination = torch.div(expert, self.experts_per_rank, rounding_mode="floor")
        send_counts = torch.bincount(destination, minlength=self.ep_size)
        if send_counts.device.type != "cpu" and dist.get_backend(self.ep_group) == "gloo":
            send_counts = send_counts.cpu()
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)
        return int(recv_counts.sum().item())

    def __call__(
        self,
        activation: MoeTensor,
        fc1_weight: MoeTensor,
        fc2_weight: MoeTensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Union[MoeTensor, Tuple[MoeTensor, torch.Tensor, torch.Tensor]]:
        """Allocate the result for a future backend MoE+EP launch.

        Expected logical shapes are ``activation=(T,H)``,
        ``fc1_weight=(E_local,H,2I)``, ``fc2_weight=(E_local,I,H)``, and
        ``topk_idx=topk_weights=(T,K)``.

        Returns the ``(T, H)`` result, or ``(result, fc1_c, route_metadata)``
        when the operator was constructed with ``generate_c=True``.
        """

        activation_shape = _logical_shape(activation)
        if len(activation_shape) != 2 or activation_shape[1] != self.hidden_size:
            raise ValueError(f"activation logical shape must be (T, {self.hidden_size}), got {activation_shape}")
        token_count = activation_shape[0]
        expected_fc1 = (self.experts_per_rank, self.hidden_size, 2 * self.intermediate_size)
        expected_fc2 = (self.experts_per_rank, self.intermediate_size, self.hidden_size)
        if _logical_shape(fc1_weight) != expected_fc1:
            raise ValueError(f"fc1_weight logical shape must be {expected_fc1}")
        if _logical_shape(fc2_weight) != expected_fc2:
            raise ValueError(f"fc2_weight logical shape must be {expected_fc2}")
        route_shape = (token_count, self.top_k)
        if tuple(topk_idx.shape) != route_shape:
            raise ValueError(f"topk_idx shape must be {route_shape}, got {tuple(topk_idx.shape)}")
        if tuple(topk_weights.shape) != route_shape:
            raise ValueError(f"topk_weights shape must be {route_shape}, got {tuple(topk_weights.shape)}")
        if self.max_tokens_per_rank is not None and token_count > self.max_tokens_per_rank:
            raise ValueError(f"token count {token_count} exceeds max_tokens_per_rank={self.max_tokens_per_rank}")

        device = _tensor_device(activation)
        for name, tensor in (
            ("fc1_weight", fc1_weight),
            ("fc2_weight", fc2_weight),
            ("topk_idx", topk_idx),
            ("topk_weights", topk_weights),
        ):
            if _tensor_device(tensor) != device:
                raise ValueError(f"{name} must be on {device}, got {_tensor_device(tensor)}")

        # TODO: dispatch routes, launch local expert FC1/SwiGLU/FC2, return
        # contributions, reduce top-k, and write this allocation.
        output = self._allocate_output(token_count, device)
        if not self.generate_c:
            return output
        local_routes = self._count_local_routes(topk_idx)
        fc1_c = torch.empty(
            (local_routes, 2 * self.intermediate_size),
            dtype=torch.bfloat16,
            device=device,
        )
        route_metadata = torch.empty((local_routes, 4), dtype=torch.int32, device=device)
        return output, fc1_c, route_metadata


__all__ = ["BlockScaledTensor", "MoeEp", "MoeFormat", "MoeTensor"]
