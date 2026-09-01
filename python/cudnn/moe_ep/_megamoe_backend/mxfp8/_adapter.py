# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Logical MXFP8 to Rubin SM107 MegaMoE tensor staging."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..._contracts import Fc1WeightLayout, ValidatedForwardRequest
from ..._types import BlockScaledTensor, MoeFormat
from .._plan import PreparedResources
from .._workspace import padded_mxfp8_scale_columns
from ._config import Mxfp8KernelConfig

_MXFP8_DATA_DTYPE = torch.float8_e4m3fn
_MXFP8_SCALE_DTYPE = torch.float8_e8m0fnu
_GATE_UP_INTERLEAVE = 32
_WORKSPACE_GUARD_BYTE = 0xA5


def _decode_moe_tensor(
    tensor: torch.Tensor | BlockScaledTensor,
) -> torch.Tensor:
    """Decode a public MoE tensor to float32 for host-side staging math."""

    if isinstance(tensor, BlockScaledTensor):
        return tensor.dequantize(torch.float32)
    return tensor.float()


def _quantize_plain_mxfp8(
    tensor: torch.Tensor,
    *,
    axis: int = 1,
) -> BlockScaledTensor:
    """Stage a plain floating tensor through the backend's MXFP8 family."""

    moved = tensor.float().movedim(axis, -1)
    logical_extent = moved.shape[-1]
    block_count = (logical_extent + 31) // 32
    padded_extent = block_count * 32
    if padded_extent != logical_extent:
        moved = torch.nn.functional.pad(
            moved,
            (0, padded_extent - logical_extent),
        )
    blocks = moved.reshape(*moved.shape[:-1], block_count, 32)
    raw_scale = blocks.abs().amax(dim=-1) / 448.0
    safe_scale = torch.where(raw_scale > 0, raw_scale, 1.0)
    scale_float = torch.where(
        raw_scale > 0,
        torch.pow(2.0, torch.ceil(torch.log2(safe_scale))),
        torch.zeros_like(raw_scale),
    )
    scale = scale_float.to(_MXFP8_SCALE_DTYPE)
    scale_for_math = scale.float()
    reciprocal = torch.where(
        scale_for_math > 0,
        scale_for_math.reciprocal(),
        0.0,
    )
    normalized = (blocks * reciprocal.unsqueeze(-1)).clamp(-448.0, 448.0)
    data = normalized.to(_MXFP8_DATA_DTYPE).reshape(*moved.shape)[..., :logical_extent].movedim(-1, axis).contiguous()
    return BlockScaledTensor(
        data=data,
        scale=scale.movedim(-1, axis).contiguous(),
        format=MoeFormat.MXFP8,
        logical_shape=tuple(tensor.shape),
        axis=axis,
    )


def _as_mxfp8(tensor: torch.Tensor | BlockScaledTensor) -> BlockScaledTensor:
    if isinstance(tensor, BlockScaledTensor):
        if tensor.format is not MoeFormat.MXFP8:
            raise NotImplementedError("MXFP8 staging cannot convert " f"{tensor.format.value!r} block-scaled input")
        return tensor
    return _quantize_plain_mxfp8(tensor)


def _typed_view(
    byte_tensor: torch.Tensor,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    expected_bytes = 1
    for extent in shape:
        expected_bytes *= extent
    expected_bytes *= dtype.itemsize
    if byte_tensor.numel() != expected_bytes:
        raise ValueError(f"byte region has {byte_tensor.numel()} bytes, " f"expected {expected_bytes} for shape={shape}, dtype={dtype}")
    return byte_tensor.view(dtype).reshape(shape)


def _typed_k_major_view(
    byte_tensor: torch.Tensor,
    dtype: torch.dtype,
    shape: tuple[int, int],
) -> torch.Tensor:
    """Return a rank-2 ``(K,N)`` view with K as the unit-stride mode."""

    if len(shape) != 2:
        raise ValueError(f"K-major view requires rank-2 shape, got {shape}")
    rows, columns = shape
    return _typed_view(
        byte_tensor,
        dtype,
        (columns, rows),
    ).transpose(0, 1)


def _as_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.uint8)


def _validate_int32_downcast(tensor: torch.Tensor) -> None:
    """Validate the only lossy public-to-kernel dtype conversion."""

    if tensor.dtype is torch.int32:
        return
    if tensor.dtype is not torch.int64:
        raise TypeError("topk_idx staging requires torch.int32 or torch.int64, " f"got {tensor.dtype}")
    capturing = tensor.device.type == "cuda" and torch.cuda.is_current_stream_capturing()
    if capturing or tensor.numel() == 0:
        # The public validator checked the same tensor before capture. During
        # replay callers must preserve its documented expert-id invariant.
        return
    limits = torch.iinfo(torch.int32)
    outside_int32 = (tensor < limits.min) | (tensor > limits.max)
    if bool(outside_int32.any().item()):
        raise OverflowError("topk_idx contains a value outside the int32 range")


def _zero_workspace_prefix(
    workspace: torch.Tensor,
    nbytes: int,
    *,
    name: str,
) -> None:
    if nbytes < 0 or nbytes > workspace.numel():
        raise ValueError(f"{name} zero prefix {nbytes} exceeds {workspace.numel()} bytes")
    workspace[:nbytes].zero_()


def _zero_workspace_range(
    workspace: torch.Tensor,
    offset: int,
    nbytes: int,
    *,
    name: str,
) -> None:
    if offset < 0 or nbytes < 0 or offset + nbytes > workspace.numel():
        raise ValueError(f"{name} byte range [{offset}, {offset + nbytes}) exceeds " f"{workspace.numel()} bytes")
    workspace.narrow(0, offset, nbytes).zero_()


def _interleave_gate_up_rows(
    tensor: torch.Tensor,
    intermediate: int,
) -> torch.Tensor:
    """Convert gate-half/up-half rows to 32-row gate/up pairs."""

    if intermediate % _GATE_UP_INTERLEAVE:
        raise ValueError("MXFP8 gate/up interleave requires intermediate_size to be " f"divisible by {_GATE_UP_INTERLEAVE}, got {intermediate}")
    if tensor.ndim != 3 or tensor.shape[1] != 2 * intermediate:
        raise ValueError(f"expected (experts, {2 * intermediate}, K) tensor, " f"got {tuple(tensor.shape)}")

    experts, _gate_up, reduction = tensor.shape
    pairs = intermediate // _GATE_UP_INTERLEAVE
    gate = tensor[:, :intermediate].reshape(
        experts,
        pairs,
        _GATE_UP_INTERLEAVE,
        reduction,
    )
    up = tensor[:, intermediate:].reshape(
        experts,
        pairs,
        _GATE_UP_INTERLEAVE,
        reduction,
    )
    return torch.stack((gate, up), dim=2).reshape(experts, 2 * intermediate, reduction).contiguous()


def _to_blocked_bytes(scale_2d: torch.Tensor) -> torch.Tensor:
    """Apply the kernel's 32x4x4 scale-factor atom swizzle."""

    if scale_2d.ndim != 2:
        raise ValueError(f"expected 2D scale tensor, got {scale_2d.ndim}D")
    rows, columns = scale_2d.shape
    if rows == 0 or columns == 0:
        return scale_2d.new_empty((0,), dtype=torch.uint8)

    row_blocks = (rows + 127) // 128
    column_blocks = (columns + 3) // 4
    padded_rows = row_blocks * 128
    padded_columns = column_blocks * 4
    padded = torch.zeros(
        padded_rows,
        padded_columns,
        dtype=torch.uint8,
        device=scale_2d.device,
    )
    padded[:rows, :columns].copy_(_as_bytes(scale_2d))
    blocks = padded.view(row_blocks, 128, column_blocks, 4).permute(
        0,
        2,
        1,
        3,
    )
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def _stack_blocked_scales(raw_scales: torch.Tensor) -> torch.Tensor:
    experts = raw_scales.shape[0]
    blocked = [_to_blocked_bytes(raw_scales[e]) for e in range(experts)]
    if not blocked:
        return torch.empty(
            (0, 0),
            dtype=torch.uint8,
            device=raw_scales.device,
        ).view(raw_scales.dtype)
    flat_size = blocked[0].numel()
    output = torch.empty(
        experts,
        flat_size,
        dtype=torch.uint8,
        device=raw_scales.device,
    )
    for expert, values in enumerate(blocked):
        output[expert].copy_(values)
    return output.view(raw_scales.dtype)


def _prepare_fc1(
    tensor: BlockScaledTensor,
    intermediate: int,
    *,
    already_interleaved: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build kernel-native K-major FC1 tensors."""

    payload_nkh = _as_bytes(tensor.data).permute(0, 2, 1).contiguous()
    payload_interleaved = (
        payload_nkh
        if already_interleaved
        else _interleave_gate_up_rows(payload_nkh, intermediate)
    )
    payload = payload_interleaved.view(_MXFP8_DATA_DTYPE).permute(0, 2, 1)

    scales_nk = _as_bytes(tensor.scale).permute(0, 2, 1).contiguous()
    scales_interleaved = (
        scales_nk
        if already_interleaved
        else _interleave_gate_up_rows(scales_nk, intermediate)
    )
    scale = _stack_blocked_scales(scales_interleaved)
    return payload, scale


def _prepare_fc2(
    tensor: BlockScaledTensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve logical bytes while building K-major FC2 tensors."""

    payload_nk = _as_bytes(tensor.data).permute(0, 2, 1).contiguous()
    payload = payload_nk.view(_MXFP8_DATA_DTYPE).permute(0, 2, 1)
    scales_nk = _as_bytes(tensor.scale).permute(0, 2, 1).contiguous()
    scale = _stack_blocked_scales(scales_nk)
    return payload, scale


def _tensor_fingerprint(tensor: torch.Tensor) -> tuple | None:
    try:
        version = tensor._version
    except RuntimeError:
        return None
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        version,
    )


def _block_scaled_fingerprint(tensor: BlockScaledTensor) -> tuple | None:
    data = _tensor_fingerprint(tensor.data)
    scale = _tensor_fingerprint(tensor.scale)
    if data is None or scale is None:
        return None
    return data, scale


@dataclass(frozen=True)
class Mxfp8Weights:
    fc1_weight: torch.Tensor
    fc1_weight_sf: torch.Tensor
    fc2_weight: torch.Tensor
    fc2_weight_sf: torch.Tensor


@dataclass(frozen=True)
class Mxfp8LaunchInputs:
    activation: torch.Tensor
    activation_sf: torch.Tensor
    topk_indices: torch.Tensor
    topk_scores: torch.Tensor
    weights: Mxfp8Weights
    fc1_c: torch.Tensor | None
    output_data: torch.Tensor
    col_quant_data: torch.Tensor | None
    col_quant_sf: torch.Tensor | None
    overflow_flag: torch.Tensor
    local_workspace: torch.Tensor
    shared_workspace: torch.Tensor
    token_count: int


class Mxfp8InputAdapter:
    """Stateful staging adapter with mutation-aware weight transforms."""

    def __init__(self) -> None:
        self._weight_key: tuple | None = None
        self._weights: Mxfp8Weights | None = None
        self._weight_sources: tuple[torch.Tensor, ...] | None = None
        self._weight_refresh_count = 0
        self._initialized_workspace_key: tuple[int, int] | None = None

    @property
    def weight_refresh_count(self) -> int:
        return self._weight_refresh_count

    def has_cached_weights(self, request: ValidatedForwardRequest) -> bool:
        key = self._request_weight_key(request)
        return key is not None and key == self._weight_key and self._weights is not None

    def weights_have_version_counters(
        self,
        request: ValidatedForwardRequest,
    ) -> bool:
        return self._request_weight_key(request) is not None

    @staticmethod
    def _request_weight_key(
        request: ValidatedForwardRequest,
    ) -> tuple | None:
        fc1 = _block_scaled_fingerprint(request.fc1_weight) if isinstance(request.fc1_weight, BlockScaledTensor) else _tensor_fingerprint(request.fc1_weight)
        fc2 = _block_scaled_fingerprint(request.fc2_weight) if isinstance(request.fc2_weight, BlockScaledTensor) else _tensor_fingerprint(request.fc2_weight)
        if fc1 is None or fc2 is None:
            return None
        return fc1, fc2

    def _prepare_weights(
        self,
        request: ValidatedForwardRequest,
        config: Mxfp8KernelConfig,
    ) -> Mxfp8Weights:
        key = self._request_weight_key(request)
        if key is not None and key == self._weight_key and self._weights is not None:
            return self._weights

        fc1_source = _as_mxfp8(request.fc1_weight)
        fc2_source = _as_mxfp8(request.fc2_weight)
        fc1_weight, fc1_weight_sf = _prepare_fc1(
            fc1_source,
            config.intermediate,
            already_interleaved=(
                config.fc1_weight_layout is Fc1WeightLayout.GATE_UP_INTERLEAVED_32
            ),
        )
        fc2_weight, fc2_weight_sf = _prepare_fc2(fc2_source)
        weights = Mxfp8Weights(
            fc1_weight=fc1_weight,
            fc1_weight_sf=fc1_weight_sf,
            fc2_weight=fc2_weight,
            fc2_weight_sf=fc2_weight_sf,
        )
        self._weight_key = key
        self._weights = weights
        # Retain the source storages while this entry is cached so allocator
        # pointer reuse cannot produce a false cache hit.
        self._weight_sources = (
            *((request.fc1_weight.data, request.fc1_weight.scale) if isinstance(request.fc1_weight, BlockScaledTensor) else (request.fc1_weight,)),
            *((request.fc2_weight.data, request.fc2_weight.scale) if isinstance(request.fc2_weight, BlockScaledTensor) else (request.fc2_weight,)),
        )
        self._weight_refresh_count += 1
        return weights

    def stage(
        self,
        request: ValidatedForwardRequest,
        resources: PreparedResources,
        config: Mxfp8KernelConfig,
        *,
        local_workspace_zero_bytes: int,
        shared_workspace_zero_bytes: int,
        pre_reduced_activation_offset: int | None,
        pre_reduced_activation_bytes_per_token: int,
        pre_reduced_activation_sf_offset: int | None,
        pre_reduced_activation_sf_bytes_per_token: int,
        col_quant_data_rows: int,
        col_quant_sf_elements: int,
        fc1_c: torch.Tensor | None = None,
    ) -> Mxfp8LaunchInputs:
        capacity = config.max_tokens_per_rank
        token_count = request.token_count
        if config.generate_c:
            if fc1_c is None:
                raise ValueError("generate_c=True requires an fc1_c buffer")
            if (
                fc1_c.dtype is not torch.bfloat16
                or fc1_c.device != request.device
                or fc1_c.ndim != 2
                or fc1_c.shape[0] <= 0
                or fc1_c.shape[1] != config.fc1_out
                or not fc1_c.is_contiguous()
            ):
                raise ValueError("fc1_c buffer must be contiguous BF16 on the request " f"device with shape (capacity, {config.fc1_out})")
        elif fc1_c is not None:
            raise ValueError("generate_c=False must not receive an fc1_c buffer")
        hidden_sf_columns = (config.hidden + 31) // 32
        padded_sf_columns = padded_mxfp8_scale_columns(config.hidden)
        symmetric = resources.workspace.symmetric
        local = resources.workspace.local
        symmetric_guard = symmetric.get("symmetric_guard")
        if symmetric_guard is not None:
            symmetric_guard.fill_(_WORKSPACE_GUARD_BYTE)
        local_guard = local.get("local_guard")
        if local_guard is not None:
            local_guard.fill_(_WORKSPACE_GUARD_BYTE)

        activation = _typed_view(
            symmetric["activation_data"],
            _MXFP8_DATA_DTYPE,
            (capacity, config.hidden),
        )
        activation_sf = _typed_view(
            symmetric["activation_scale"],
            _MXFP8_SCALE_DTYPE,
            (capacity, padded_sf_columns),
        )
        topk_weights = _typed_view(
            symmetric["topk_weights"],
            torch.float32,
            (capacity, config.top_k),
        )
        output_data = _typed_view(
            symmetric["output_data"],
            torch.bfloat16,
            (capacity, config.hidden),
        )
        topk_indices = _typed_view(
            local["topk_idx"],
            torch.int32,
            (capacity, config.top_k),
        )
        overflow_flag = _typed_view(
            local["overflow_flag"],
            torch.int32,
            (1,),
        )
        if config.enable_col_quant:
            if col_quant_data_rows <= 0 or col_quant_sf_elements <= 0:
                raise ValueError("enabled column requant requires positive output capacities")
            col_quant_data = _typed_k_major_view(
                local["col_quant_data"],
                _MXFP8_DATA_DTYPE,
                (col_quant_data_rows, config.hidden),
            )
            col_quant_sf = _typed_view(
                local["col_quant_sf"],
                torch.uint8,
                (col_quant_sf_elements,),
            )
        else:
            if col_quant_data_rows != 0 or col_quant_sf_elements != 0:
                raise ValueError("disabled column requant must not reserve output capacity")
            col_quant_data = None
            col_quant_sf = None
        local_workspace = local["kernel_local_workspace"]
        shared_workspace = symmetric["kernel_shared_workspace"]

        staged_activation = _as_mxfp8(request.activation)
        _as_bytes(activation).zero_()
        _as_bytes(activation[:token_count]).copy_(_as_bytes(staged_activation.data))
        _as_bytes(activation_sf).zero_()
        _as_bytes(activation_sf[:token_count, :hidden_sf_columns]).copy_(_as_bytes(staged_activation.scale))
        _validate_int32_downcast(request.topk_idx)
        topk_indices.fill_(-1)
        topk_indices[:token_count].copy_(request.topk_idx)
        topk_weights.zero_()
        topk_weights[:token_count].copy_(request.topk_weights)
        _as_bytes(output_data).zero_()
        if col_quant_data is not None:
            local["col_quant_data"].zero_()
        if col_quant_sf is not None:
            col_quant_sf.zero_()
        overflow_flag.zero_()
        workspace_key = (
            local_workspace.data_ptr(),
            shared_workspace.data_ptr(),
        )
        if workspace_key != self._initialized_workspace_key:
            # This prefix contains both tail-reset regions and persistent
            # sense-reversing NVLink barrier counters marked
            # zero_on_first_allocate. Re-zeroing it on a later rank-skewed
            # launch can erase a peer's signal and deadlock both kernels.
            _zero_workspace_prefix(
                local_workspace,
                local_workspace_zero_bytes,
                name="local workspace",
            )
            _zero_workspace_prefix(
                shared_workspace,
                shared_workspace_zero_bytes,
                name="shared workspace",
            )
            self._initialized_workspace_key = workspace_key
        if config.fc2_in_kernel_topk_reduce:
            if (
                pre_reduced_activation_offset is not None
                or pre_reduced_activation_bytes_per_token != 0
                or pre_reduced_activation_sf_offset is not None
                or pre_reduced_activation_sf_bytes_per_token != 0
            ):
                raise ValueError("in-kernel top-k reduction must not receive a " "standalone pre-reduced activation workspace")
            # output_data is the in-kernel REDG accumulation base and was
            # cleared above.
        else:
            if pre_reduced_activation_offset is None or pre_reduced_activation_bytes_per_token <= 0:
                raise ValueError("standalone top-k reduction requires a pre-reduced " "activation workspace")
            # The kernel writes only valid routes into this persistent combine
            # plane. Clear the active token rows so dropped routes cannot reuse
            # contributions from a previous launch.
            _zero_workspace_range(
                shared_workspace,
                pre_reduced_activation_offset,
                token_count * pre_reduced_activation_bytes_per_token,
                name="pre-reduced activation workspace",
            )
            quantized_combine = config.combine_format != "bf16"
            if quantized_combine:
                if pre_reduced_activation_sf_offset is None or pre_reduced_activation_sf_bytes_per_token <= 0:
                    raise ValueError("quantized standalone top-k reduction requires a " "pre-reduced scale workspace")
                _zero_workspace_range(
                    shared_workspace,
                    pre_reduced_activation_sf_offset,
                    token_count * pre_reduced_activation_sf_bytes_per_token,
                    name="pre-reduced activation scale workspace",
                )
            elif pre_reduced_activation_sf_offset is not None or pre_reduced_activation_sf_bytes_per_token != 0:
                raise ValueError("BF16 standalone top-k reduction must not receive a " "pre-reduced scale workspace")

        weights = self._prepare_weights(request, config)
        return Mxfp8LaunchInputs(
            activation=activation,
            activation_sf=activation_sf,
            topk_indices=topk_indices,
            topk_scores=topk_weights,
            weights=weights,
            fc1_c=fc1_c,
            output_data=output_data,
            col_quant_data=col_quant_data,
            col_quant_sf=col_quant_sf,
            overflow_flag=overflow_flag,
            local_workspace=local_workspace,
            shared_workspace=shared_workspace,
            token_count=token_count,
        )

    def close(self) -> None:
        self._weights = None
        self._weight_key = None
        self._weight_sources = None
        self._initialized_workspace_key = None


__all__ = [
    "Mxfp8InputAdapter",
    "Mxfp8LaunchInputs",
    "Mxfp8Weights",
]
