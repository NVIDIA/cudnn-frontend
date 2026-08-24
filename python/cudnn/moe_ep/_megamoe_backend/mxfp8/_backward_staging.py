# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Tensor staging for the explicit Rubin MXFP8 backward invocation."""

from __future__ import annotations

import math

import torch

from ..._contracts import ValidatedBackwardRequest
from .._plan import PreparedResources
from .._workspace import padded_mxfp8_scale_columns
from ._adapter import (
    _GATE_UP_INTERLEAVE,
    _decode_moe_tensor,
    _interleave_gate_up_rows,
    _quantize_plain_mxfp8,
    _stack_blocked_scales,
    _typed_view,
    _zero_workspace_prefix,
    _zero_workspace_range,
)
from ._backward_compile import (
    Mxfp8BackwardLaunchInputs,
    PreparedMxfp8BackwardKernel,
)
from ._backward_layout import Mxfp8BackwardLayout

_DATA_DTYPE = torch.float8_e4m3fn
_SCALE_DTYPE = torch.float8_e8m0fnu


def _typed_prefix_view(
    byte_tensor: torch.Tensor,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Return a compact typed view of a prefix of a reusable byte region."""

    nbytes = math.prod(shape) * dtype.itemsize
    if nbytes > byte_tensor.numel():
        raise ValueError(
            f"byte region has {byte_tensor.numel()} bytes, "
            f"cannot provide {nbytes} bytes for shape={shape}, dtype={dtype}"
        )
    return _typed_view(byte_tensor.narrow(0, 0, nbytes), dtype, shape)


def _k_major(tensor: torch.Tensor) -> torch.Tensor:
    """Return the same logical ``(E,K,N)`` tensor with K stride one."""

    return tensor.permute(0, 2, 1).contiguous().permute(0, 2, 1)


def _prepare_backward_weights(
    request: ValidatedBackwardRequest,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize W2^T and W1^T along their backward reduction axes."""

    config = request.config
    intermediate = config.intermediate_size

    w2_t = _decode_moe_tensor(request.fc2_weight).transpose(1, 2)
    q_w2_t = _quantize_plain_mxfp8(w2_t, axis=1)
    fc1_weight = _k_major(q_w2_t.data)
    fc1_weight_sf = _stack_blocked_scales(
        q_w2_t.scale.permute(0, 2, 1).contiguous()
    )

    w1_t = _decode_moe_tensor(request.fc1_weight).transpose(1, 2)
    q_w1_t = _quantize_plain_mxfp8(w1_t, axis=1)
    interleaved_w1 = _interleave_gate_up_rows(
        q_w1_t.data,
        intermediate,
    )
    fc2_weight = _k_major(interleaved_w1)

    scale_blocks = intermediate // 32
    gate_sf = q_w1_t.scale[:, :scale_blocks, :]
    up_sf = q_w1_t.scale[:, scale_blocks:, :]
    interleaved_sf = (
        torch.stack(
            (
                gate_sf.view(torch.uint8),
                up_sf.view(torch.uint8),
            ),
            dim=2,
        )
        .reshape(
            q_w1_t.scale.shape[0],
            2 * scale_blocks,
            q_w1_t.scale.shape[2],
        )
        .view(_SCALE_DTYPE)
    )
    fc2_weight_sf = _stack_blocked_scales(
        interleaved_sf.permute(0, 2, 1).contiguous()
    )
    return (
        fc1_weight,
        fc1_weight_sf,
        fc2_weight,
        fc2_weight_sf,
    )


def _router_order_key(
    source_token: torch.Tensor,
    source_slot: torch.Tensor,
    request: ValidatedBackwardRequest,
    prepared: PreparedMxfp8BackwardKernel,
) -> tuple[torch.Tensor, int]:
    """Return each route's position in the deterministic router's source run."""

    elements_per_vector = 4  # The staged top-k index tensor is Int32.
    token_comm = prepared.kernel.token_comm
    router_ctas = int(token_comm.router_data_cta_count)
    router_warps = int(token_comm.router_warps_per_cta)
    threads_per_cta = router_warps * 32
    grid_threads = router_ctas * threads_per_cta
    tile_span = elements_per_vector * grid_threads
    maximum_elements = (
        int(request.config.max_tokens_per_rank) * request.config.top_k
    )
    load_rounds = (
        maximum_elements + tile_span - 1
    ) // tile_span
    elements_per_thread = load_rounds * elements_per_vector

    flat_index = source_token * request.config.top_k + source_slot
    load_round = torch.div(
        flat_index,
        tile_span,
        rounding_mode="floor",
    )
    in_tile = flat_index.remainder(tile_span)
    grid_thread = torch.div(
        in_tile,
        elements_per_vector,
        rounding_mode="floor",
    )
    register_index = (
        load_round * elements_per_vector
        + in_tile.remainder(elements_per_vector)
    )
    cta = torch.div(
        grid_thread,
        threads_per_cta,
        rounding_mode="floor",
    )
    thread_in_cta = grid_thread.remainder(threads_per_cta)
    warp = torch.div(thread_in_cta, 32, rounding_mode="floor")
    lane = thread_in_cta.remainder(32)
    order_key = (
        (
            (cta * router_warps + warp) * elements_per_thread
            + register_index
        )
        * 32
        + lane
    )
    order_span = (
        router_ctas * router_warps * elements_per_thread * 32
    )
    return order_key, order_span


def _stage_fc1_preact(
    request: ValidatedBackwardRequest,
    layout: Mxfp8BackwardLayout,
    prepared: PreparedMxfp8BackwardKernel,
    fc1_preact: torch.Tensor,
) -> None:
    """Lower compact public stash rows into the upstream dGLU pool layout."""

    config = prepared.config
    expected_shape = (
        prepared.pool_token_capacity,
        2 * config.intermediate,
    )
    if (
        fc1_preact.dtype is not torch.bfloat16
        or tuple(fc1_preact.shape) != expected_shape
        or not fc1_preact.is_contiguous()
    ):
        raise ValueError(
            "backward fc1_preact must be contiguous BF16 with shape "
            f"{expected_shape}, got shape={tuple(fc1_preact.shape)}, "
            f"dtype={fc1_preact.dtype}"
        )
    fc1_preact.zero_()

    metadata = request.route_metadata.to(torch.int64)
    compact_rows = layout.preact_row_lut[
        metadata[:, 1],
        metadata[:, 2],
        metadata[:, 3],
    ].to(torch.int64)
    if compact_rows.numel() and bool((compact_rows < 0).any().item()):
        raise RuntimeError("backward preactivation LUT is incomplete")

    if config.intermediate % _GATE_UP_INTERLEAVE:
        raise RuntimeError(
            "backward preactivation requires intermediate_size divisible by "
            f"{_GATE_UP_INTERLEAVE}"
        )
    gate, up = request.fc1_c.split(config.intermediate, dim=1)
    pairs = config.intermediate // _GATE_UP_INTERLEAVE
    interleaved_preact = torch.stack(
        (
            gate.reshape(-1, pairs, _GATE_UP_INTERLEAVE),
            up.reshape(-1, pairs, _GATE_UP_INTERLEAVE),
        ),
        dim=2,
    ).reshape(-1, 2 * config.intermediate)

    physical_offset = 0
    for expert in range(config.num_experts):
        positions = torch.nonzero(
            metadata[:, 0] == expert,
            as_tuple=False,
        ).flatten()
        count = int(positions.numel())
        if count:
            # Receiver pools concatenate source ranks in a destination-relative
            # ring: local rank first, then increasing ranks with wraparound.
            # The public stash is source-rank sorted, so restore pool order.
            source_rank = metadata.index_select(0, positions)[:, 1]
            source_token = metadata.index_select(0, positions)[:, 2]
            source_slot = metadata.index_select(0, positions)[:, 3]
            source_order, source_order_span = _router_order_key(
                source_token,
                source_slot,
                request,
                prepared,
            )
            ring_position = (
                source_rank - request.config.ep_rank
            ) % request.config.ep_size
            route_key = ring_position * source_order_span + source_order
            positions = positions.index_select(
                0,
                torch.argsort(route_key, stable=True),
            )
            if physical_offset + count > prepared.pool_token_capacity:
                raise RuntimeError(
                    "backward preactivation rows exceed Rubin pool capacity"
                )
            destination = fc1_preact.narrow(0, physical_offset, count)
            destination.copy_(
                interleaved_preact.index_select(
                    0,
                    compact_rows.index_select(0, positions),
                )
            )
        physical_offset += (
            count + config.token_padding_block - 1
        ) // config.token_padding_block * config.token_padding_block

    if physical_offset > prepared.pool_token_capacity:
        raise RuntimeError(
            "backward preactivation rows exceed Rubin pool capacity"
        )


def stage_backward(
    request: ValidatedBackwardRequest,
    layout: Mxfp8BackwardLayout,
    prepared: PreparedMxfp8BackwardKernel,
    resources: PreparedResources,
) -> Mxfp8BackwardLaunchInputs:
    config = prepared.config
    capacity = config.max_tokens_per_rank
    token_count = request.token_count
    hidden = config.hidden
    top_k = config.top_k

    symmetric = resources.workspace.symmetric
    local = resources.workspace.local
    grad_out = _typed_view(
        symmetric["activation_data"],
        _DATA_DTYPE,
        (capacity, hidden),
    )
    grad_out_sf = _typed_view(
        symmetric["activation_scale"],
        _SCALE_DTYPE,
        (capacity, padded_mxfp8_scale_columns(hidden)),
    )
    topk_weights = _typed_view(
        symmetric["topk_weights"],
        torch.float32,
        (capacity, top_k),
    )
    topk_idx = _typed_view(
        local["topk_idx"],
        torch.int32,
        (capacity, top_k),
    )
    output_activation = _typed_view(
        symmetric["output_data"],
        torch.bfloat16,
        (capacity, hidden),
    )
    overflow_flag = _typed_view(
        local["overflow_flag"],
        torch.int32,
        (1,),
    )
    fc1_preact_shape = tuple(
        int(extent) for extent in prepared.kernel.get_fc1_preact_shape()
    )
    fc1_preact = _typed_view(
        local["backward_fc1_preact"],
        torch.bfloat16,
        fc1_preact_shape,
    )
    local_workspace = local["kernel_local_workspace"]
    shared_workspace = symmetric["kernel_shared_workspace"]

    _zero_workspace_prefix(
        local_workspace,
        prepared.local_workspace_zero_bytes,
        name="MXFP8 backward local workspace",
    )
    _zero_workspace_prefix(
        shared_workspace,
        prepared.shared_workspace_zero_bytes,
        name="MXFP8 backward shared workspace",
    )
    _stage_fc1_preact(
        request,
        layout,
        prepared,
        fc1_preact,
    )
    _zero_workspace_range(
        shared_workspace,
        prepared.pre_reduced_activation_offset,
        token_count * prepared.pre_reduced_activation_bytes_per_token,
        name="MXFP8 backward pre-reduced activation workspace",
    )
    quantized_combine = config.combine_format != "bf16"
    if quantized_combine:
        if (
            prepared.pre_reduced_activation_sf_offset is None
            or prepared.pre_reduced_activation_sf_bytes_per_token <= 0
        ):
            raise RuntimeError(
                "MXFP8 backward quantized combine requires scale workspace"
            )
        _zero_workspace_range(
            shared_workspace,
            prepared.pre_reduced_activation_sf_offset,
            token_count
            * prepared.pre_reduced_activation_sf_bytes_per_token,
            name="MXFP8 backward pre-reduced activation scale workspace",
        )
    elif (
        prepared.pre_reduced_activation_sf_offset is not None
        or prepared.pre_reduced_activation_sf_bytes_per_token != 0
    ):
        raise RuntimeError(
            "MXFP8 backward BF16 combine must not expose scale workspace"
        )
    quantized_grad = _quantize_plain_mxfp8(
        request.grad_output,
        axis=1,
    )
    grad_out.zero_()
    grad_out[:token_count].copy_(quantized_grad.data)
    grad_out_sf.zero_()
    grad_out_sf[
        :token_count,
        : quantized_grad.scale.shape[1],
    ].copy_(quantized_grad.scale)
    topk_idx.fill_(-1)
    topk_idx[:token_count].copy_(
        request.topk_idx.to(torch.int32)
    )
    topk_weights.zero_()
    topk_weights[:token_count].copy_(
        request.topk_weights.float()
    )
    output_activation.zero_()
    overflow_flag.zero_()

    aux_shapes = {
        name: tuple(int(extent) for extent in shape)
        for name, shape in prepared.kernel.get_aux_output_shapes().items()
    }
    dprob = _typed_view(
        symmetric["backward_dprob"],
        torch.float32,
        aux_shapes["dprob"],
    )
    dprob.zero_()
    operands_mode = request.config.backward_wgrad_mode == "operands"
    expected_flags = (
        prepared.dfc2_recompute,
        prepared.dfc2_col_output,
        prepared.enable_grad_y2_col_quant,
    )
    if operands_mode:
        if expected_flags != (True, True, True):
            raise RuntimeError(
                "wgrad operands require every backward auxiliary output"
            )
        # These allocations are intentionally not execution-plan workspace:
        # the returned operand bundle must remain valid after later calls.
        fc1_recompute = torch.zeros(
            aux_shapes["fc1_recompute"],
            dtype=_DATA_DTYPE,
            device=request.device,
        )
        fc1_recompute_sf = torch.full(
            aux_shapes["fc1_recompute_sf"],
            127,
            dtype=torch.uint8,
            device=request.device,
        ).view(_SCALE_DTYPE)
        fc1_col_output = torch.zeros(
            aux_shapes["fc1_col_output"],
            dtype=_DATA_DTYPE,
            device=request.device,
        )
        fc1_col_output_sf = torch.full(
            aux_shapes["fc1_col_output_sf"],
            127,
            dtype=torch.uint8,
            device=request.device,
        ).view(_SCALE_DTYPE)
        grad_y2 = torch.zeros(
            aux_shapes["grad_y2"],
            dtype=_DATA_DTYPE,
            device=request.device,
        )
        grad_y2_sf = torch.full(
            aux_shapes["grad_y2_sf"],
            127,
            dtype=torch.uint8,
            device=request.device,
        )
    else:
        if expected_flags != (False, False, False):
            raise RuntimeError(
                "default backward must disable wgrad auxiliary outputs"
            )
        # Preserve the existing mode-none allocation/performance behavior:
        # disabled fixed-ABI arguments alias reusable plan scratch.
        fc1_recompute = _typed_prefix_view(
            local["backward_aux_data"],
            _DATA_DTYPE,
            aux_shapes["fc1_recompute"],
        )
        fc1_col_output = _typed_prefix_view(
            local["backward_aux_data"],
            _DATA_DTYPE,
            aux_shapes["fc1_col_output"],
        )
        fc1_recompute_sf = _typed_prefix_view(
            local["backward_aux_scale"],
            _SCALE_DTYPE,
            aux_shapes["fc1_recompute_sf"],
        )
        fc1_col_output_sf = _typed_prefix_view(
            local["backward_aux_scale"],
            _SCALE_DTYPE,
            aux_shapes["fc1_col_output_sf"],
        )
        grad_y2 = _typed_prefix_view(
            local["backward_aux_data"],
            _DATA_DTYPE,
            aux_shapes["grad_y2"],
        )
        grad_y2_sf = _typed_prefix_view(
            local["backward_aux_scale"],
            torch.uint8,
            aux_shapes["grad_y2_sf"],
        )
    fc1_weight, fc1_weight_sf, fc2_weight, fc2_weight_sf = (
        _prepare_backward_weights(request)
    )
    return Mxfp8BackwardLaunchInputs(
        grad_out=grad_out,
        grad_out_sf=grad_out_sf,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        fc1_weight=fc1_weight,
        fc1_weight_sf=fc1_weight_sf,
        fc2_weight=fc2_weight,
        fc2_weight_sf=fc2_weight_sf,
        beta=torch.ones(
            (config.num_experts,),
            dtype=torch.float32,
            device=request.device,
        ),
        fc1_preact=fc1_preact,
        output_activation=output_activation,
        overflow_flag=overflow_flag,
        dprob=dprob,
        fc1_recompute=fc1_recompute,
        fc1_recompute_sf=fc1_recompute_sf,
        fc1_col_output=fc1_col_output,
        fc1_col_output_sf=fc1_col_output_sf,
        grad_y2=grad_y2,
        grad_y2_sf=grad_y2_sf,
        local_workspace=local_workspace,
        shared_workspace=shared_workspace,
        token_count=token_count,
    )


__all__ = ["stage_backward"]
