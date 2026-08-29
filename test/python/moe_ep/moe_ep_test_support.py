# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared data, forward, and backward support for MoE EP tests."""

from __future__ import annotations

# Common

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from moe_ep.moe_ep_reference import (
    BlockScaledTensor as ReferenceBlockScaledTensor,
    MoeEpReference,
    MoeFormat,
    forward_combine_round_trip,
    quantize_blockwise,
)

__all__ = [
    "_allocate_dense_grouped_wgrad_outputs",
    "_assert_backward_matches",
    "_assert_fixed_training_drop_overflow_result",
    "_assert_fixed_training_matches_reference",
    "_assert_grouped_wgrads_match_reference",
    "_assert_matches_reference",
    "_assert_training_graph_tails_are_reset",
    "_assert_training_weight_sources_changed",
    "_assert_wgrads_match_reference",
    "_capture_fixed_training_batch",
    "_copy_training_weight_sources_",
    "_dense_wgrads_from_operands",
    "_dense_wgrads_from_grouped_kernel",
    "_expected_backward",
    "_fixed_training_case",
    "_fixed_training_drop_overflow_case",
    "_fixed_training_drop_overflow_reference",
    "_fixed_training_reference",
    "_fixed_training_weights",
    "_forward_config",
    "_grad_output",
    "_make_forward_case",
    "_naive_reference",
    "_output_as_float",
    "_prefill_training_graph_sentinels",
    "_reference_backward",
    "_reference_forward",
    "_replay_cuda_graph",
    "_require_distributed_sm107",
    "_run_fixed_training_batch",
    "_run_grouped_wgrad_kernel",
    "_sm107_device",
    "_stress_backend_reuse",
    "_training_public_pointers",
    "_training_source_pointers",
    "_training_weight_source_pointers",
    "_training_weight_source_values",
    "_TrainingResourceContractOwner",
    "_training_abi_prepared",
    "_training_config",
    "_training_contract_resources",
    "_training_inputs",
    "_training_prepared_pair",
    "_training_staging_tensors",
    "_training_weight_defect",
    "_training_weights",
    "make_distributed_forward_inputs",
    "make_forward_inputs",
    "quantize_mxfp8",
]


def _allocate_dense_grouped_wgrad_outputs(
    operands,
    *,
    fill_value=None,
):
    """Allocate fixed-address dense BF16 outputs for FC1 and FC2 WGrad."""

    expert_count = operands.expert_offsets.numel()
    outputs = tuple(
        torch.empty(
            (
                expert_count,
                getattr(operands, f"{prefix}_a").shape[0],
                getattr(operands, f"{prefix}_b").shape[1],
            ),
            dtype=torch.bfloat16,
            device=operands.expert_offsets.device,
        )
        for prefix in ("fc1", "fc2")
    )
    if fill_value is not None:
        for output in outputs:
            output.fill_(fill_value)
    return outputs


# Data


def make_forward_inputs(device: torch.device):
    """Build one deterministic MXFP8 forward case."""

    generator = torch.Generator(device=device).manual_seed(20260811)
    experts, tokens, hidden, intermediate = 2, 5, 128, 256
    activation = quantize_mxfp8(
        torch.randn(tokens, hidden, generator=generator, device=device),
        axis=1,
    )
    fc1_weight = quantize_mxfp8(
        torch.randn(
            experts,
            hidden,
            2 * intermediate,
            generator=generator,
            device=device,
        )
        / 8,
        axis=1,
    )
    fc2_weight = quantize_mxfp8(
        torch.randn(
            experts,
            intermediate,
            hidden,
            generator=generator,
            device=device,
        )
        / 8,
        axis=1,
    )
    topk_idx = torch.tensor(
        [[0, 1], [1, 0], [0, -1], [1, 0], [0, 1]],
        dtype=torch.int32,
        device=device,
    )
    topk_weights = torch.tensor(
        [
            [0.75, 0.25],
            [0.625, 0.375],
            [1.0, 0.0],
            [0.5, 0.5],
            [0.875, 0.125],
        ],
        dtype=torch.bfloat16,
        device=device,
    )
    return activation, fc1_weight, fc2_weight, topk_idx, topk_weights


def make_distributed_forward_inputs(
    rank: int,
    world_size: int,
    device: torch.device,
):
    """Build rank-local inputs with one local and one remote route per token."""

    generator = torch.Generator(device=device).manual_seed(20260811 + rank)
    # Vary local shapes without exceeding the distributed tests'
    # max_tokens_per_rank=8 contract at EP sizes above seven.
    local_experts, tokens, hidden, intermediate = (
        2,
        rank % 7 + 2,
        128,
        256,
    )
    activation = quantize_mxfp8(
        torch.randn(tokens, hidden, generator=generator, device=device),
        axis=1,
    )
    fc1_weight = quantize_mxfp8(
        torch.randn(
            local_experts,
            hidden,
            2 * intermediate,
            generator=generator,
            device=device,
        )
        / 8,
        axis=1,
    )
    fc2_weight = quantize_mxfp8(
        torch.randn(
            local_experts,
            intermediate,
            hidden,
            generator=generator,
            device=device,
        )
        / 8,
        axis=1,
    )
    remote_rank = (rank + 1) % world_size
    topk_idx = torch.tensor(
        [
            [
                rank * local_experts + token % local_experts,
                remote_rank * local_experts + (token + 1) % local_experts,
            ]
            for token in range(tokens)
        ],
        dtype=torch.int32,
        device=device,
    )
    topk_weights = (
        torch.tensor(
            [[0.625, 0.375]],
            dtype=torch.bfloat16,
            device=device,
        )
        .expand(tokens, -1)
        .contiguous()
    )
    return activation, fc1_weight, fc2_weight, topk_idx, topk_weights


def quantize_mxfp8(tensor: torch.Tensor, *, axis: int = -1):
    """Return a public logical MXFP8 tensor (E4M3 payload + E8M0 scales)."""

    from cudnn import BlockScaledTensor

    axis = axis % tensor.ndim
    logical_shape = tuple(tensor.shape)
    logical_extent = logical_shape[axis]
    moved = tensor.float().movedim(axis, -1)
    block_count = (logical_extent + 31) // 32
    padded_extent = block_count * 32
    if padded_extent != logical_extent:
        moved = F.pad(moved, (0, padded_extent - logical_extent))

    blocks = moved.reshape(*moved.shape[:-1], block_count, 32)
    raw_scale = blocks.abs().amax(dim=-1) / 448.0
    safe_scale = torch.where(raw_scale > 0, raw_scale, 1.0)
    power_of_two_scale = torch.where(
        raw_scale > 0,
        torch.pow(2.0, torch.ceil(torch.log2(safe_scale))),
        torch.zeros_like(raw_scale),
    )
    scale = power_of_two_scale.to(torch.float8_e8m0fnu)
    reciprocal = torch.where(scale.float() > 0, scale.float().reciprocal(), 0.0)
    payload = (blocks * reciprocal.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).reshape(*moved.shape)[..., :logical_extent]

    return BlockScaledTensor(
        data=payload.movedim(-1, axis).contiguous(),
        scale=scale.movedim(-1, axis).contiguous(),
        format="mxfp8",
        logical_shape=logical_shape,
        axis=axis,
    )


# Training setup


def _training_config(**overrides):
    from cudnn.moe_ep._contracts import ForwardConfig
    from cudnn.moe_ep._tuning import MoeEpTuningConfig

    values = {
        "num_experts": 2,
        "hidden_size": 128,
        "intermediate_size": 256,
        "top_k": 2,
        "experts_per_rank": 2,
        "ep_size": 1,
        "ep_rank": 0,
        "ep_group": None,
        "ep_global_ranks": (),
        "max_tokens_per_rank": 4,
        "max_recv_size_per_rank": 4,
        "drop_on_overflow": True,
        "output_format": "bf16",
        "combine_format": "bf16",
        "apply_topk_in_fc1": True,
        "gate_up_clamp": None,
        "generate_c": True,
        "token_padding_size": 128,
        "sf_padding_size": 128,
        "tuning": MoeEpTuningConfig(),
        "backward_wgrad_mode": "operands",
    }
    values.update(overrides)
    return ForwardConfig(**values)


def _training_inputs():
    return (
        torch.randn(2, 128, dtype=torch.bfloat16),
        torch.randn(2, 128, 512, dtype=torch.bfloat16),
        torch.randn(2, 256, 128, dtype=torch.bfloat16),
        torch.tensor([[0, -1], [1, 0]], dtype=torch.int32),
        torch.randn(2, 2, dtype=torch.float32),
    )


def _training_prepared_pair(config, pool_rows: int = 512):
    from cudnn.moe_ep._megamoe_backend._workspace import WorkspaceRequirements

    forward_shapes = {
        "fc1_c": (pool_rows, 512),
        "col_quant_data": (pool_rows, 128),
        "col_quant_sf": (2048,),
    }
    backward_shapes = {
        "dprob": (4, 2),
        "fc1_recompute": (pool_rows, 256),
        "fc1_recompute_sf": (256, 8),
        "fc1_col_output": (pool_rows, 512),
        "fc1_col_output_sf": (512, 8),
        "grad_y2": (pool_rows, 128),
        "grad_y2_sf": (2048,),
    }
    forward = SimpleNamespace(
        pool_token_capacity=pool_rows,
        workspace_requirements=WorkspaceRequirements.for_mxfp8(
            config,
            kernel_local_workspace_bytes=1024,
            kernel_shared_workspace_bytes=2048,
            col_quant_data_bytes=pool_rows * 128,
            col_quant_sf_bytes=2048,
        ),
        kernel=SimpleNamespace(get_aux_output_shapes=lambda: forward_shapes),
        col_quant_sizes_offset=0,
        col_quant_sizes_bytes=8,
    )
    backward = SimpleNamespace(
        pool_token_capacity=pool_rows,
        config=SimpleNamespace(sf_padding_block=128),
        workspace_requirements=WorkspaceRequirements.for_mxfp8(
            config,
            kernel_local_workspace_bytes=3072,
            kernel_shared_workspace_bytes=4096,
            backward_fc1_preact_bytes=pool_rows * 512 * 2,
            backward_dprob_bytes=4 * 2 * 4,
            backward_aux_data_bytes=pool_rows * 512,
            backward_aux_scale_bytes=512 * 8,
        ),
        kernel=SimpleNamespace(get_aux_output_shapes=lambda: backward_shapes),
    )
    return forward, backward


def _training_abi_prepared(name: str, max_recv_size: int = 4):
    from cudnn.moe_ep._megamoe_backend._workspace import (
        BufferRegion,
        WorkspaceRequirements,
    )

    workspace = WorkspaceRequirements(
        max_tokens_per_rank=4,
        symmetric_regions=(BufferRegion("symmetric", 256),),
        local_regions=(BufferRegion("local", 128),),
    )
    kernel_config = SimpleNamespace(
        max_recv_size_per_rank=max_recv_size,
        effective_config=lambda cluster_count: {
            "name": name,
            "max_recv_size_per_rank": max_recv_size,
            "launch_cluster_count": cluster_count,
        },
    )
    return SimpleNamespace(
        kernel=SimpleNamespace(
            name=lambda: name,
            threads_per_cta=128,
            occupancy=1,
            smem_capacity=1024,
        ),
        architecture=(10, 7),
        config=kernel_config,
        launch_cluster_count=16,
        workspace_requirements=workspace,
        pool_token_capacity=512,
    )


def _training_weights(args=None):
    from cudnn.moe_ep import MoeEpTrainingWeights
    from cudnn.moe_ep._megamoe_backend.mxfp8._adapter import (
        _quantize_plain_mxfp8,
    )

    if args is None:
        args = _training_inputs()
    return MoeEpTrainingWeights(
        forward_fc1=_quantize_plain_mxfp8(args[1], axis=1),
        forward_fc2=_quantize_plain_mxfp8(args[2], axis=1),
        backward_w2_transpose=_quantize_plain_mxfp8(
            args[2].transpose(1, 2).contiguous(),
            axis=1,
        ),
        backward_w1_transpose=_quantize_plain_mxfp8(
            args[1].transpose(1, 2).contiguous(),
            axis=1,
        ),
    )


def _training_empty_block_scaled_like(tensor, *, axis: int, format: str):
    import cudnn

    logical_shape = tensor.logical_shape
    data_shape = list(logical_shape)
    scale_shape = list(logical_shape)
    if format == "mxfp8":
        data_dtype = tensor.data.dtype
        scale_dtype = tensor.scale.dtype
        scale_shape[axis] = (logical_shape[axis] + 31) // 32
    else:
        data_dtype = torch.uint8
        scale_dtype = tensor.data.dtype
        data_shape[axis] = (logical_shape[axis] + 1) // 2
        scale_shape[axis] = (logical_shape[axis] + 15) // 16
    return cudnn.BlockScaledTensor(
        data=torch.empty(tuple(data_shape), dtype=data_dtype, device=tensor.device),
        scale=torch.empty(
            tuple(scale_shape),
            dtype=scale_dtype,
            device=tensor.device,
        ),
        format=format,
        logical_shape=logical_shape,
        axis=axis,
    )


def _training_same_shape_noncontiguous(tensor: torch.Tensor) -> torch.Tensor:
    result = tensor.transpose(-2, -1).contiguous().transpose(-2, -1)
    assert tuple(result.shape) == tuple(tensor.shape)
    assert not result.is_contiguous()
    return result


def _training_weight_defect(weights, field: str, defect: str):
    tensor = getattr(weights, field)
    expected_shape = tensor.logical_shape
    if defect == "plain_tensor":
        invalid = torch.empty(
            expected_shape,
            dtype=torch.bfloat16,
            device=tensor.device,
        )
        error_type = TypeError
        message = f"weights.{field} must be an MXFP8 BlockScaledTensor for " "fixed training resources"
    elif defect == "logical_shape":
        wrong_shape = (expected_shape[0] - 1, *expected_shape[1:])
        invalid = replace(
            tensor,
            data=tensor.data[: wrong_shape[0]].contiguous(),
            scale=tensor.scale[: wrong_shape[0]].contiguous(),
            logical_shape=wrong_shape,
        )
        error_type = ValueError
        message = f"weights.{field} logical shape must be {expected_shape}, " f"got {wrong_shape}"
    elif defect == "axis":
        invalid = _training_empty_block_scaled_like(
            tensor,
            axis=2,
            format="mxfp8",
        )
        error_type = ValueError
        message = f"weights.{field} block-scaled axis must be 1, got 2"
    elif defect == "format":
        invalid = _training_empty_block_scaled_like(
            tensor,
            axis=1,
            format="nvfp4",
        )
        error_type = NotImplementedError
        message = f"weights.{field} must use format='mxfp8', got 'nvfp4'"
    elif defect == "device":
        invalid = replace(
            tensor,
            data=torch.empty_like(tensor.data, device="meta"),
            scale=torch.empty_like(tensor.scale, device="meta"),
        )
        error_type = ValueError
        message = f"weights.{field} must be on cpu, got meta"
    else:
        part = "data" if defect == "data_noncontiguous" else "scale"
        invalid = replace(
            tensor,
            **{part: _training_same_shape_noncontiguous(getattr(tensor, part))},
        )
        error_type = ValueError
        message = f"weights.{field} data and scale must be contiguous for fixed " "training weight binding"
    return replace(weights, **{field: invalid}), error_type, message


class _TrainingResourceContractOwner:
    def __init__(self, *, slot_count: int = 2, lane_count: int = 1) -> None:
        self.slot_count = slot_count
        self.lane_count = lane_count
        self.close_calls = 0
        self.refresh_calls = 0
        self.views_calls = 0

    def refresh_weights(self) -> None:
        self.refresh_calls += 1

    def views(self, **kwargs):
        del kwargs
        self.views_calls += 1
        raise AssertionError("binding rejection must happen before owner views")

    def _flat_views(self, token_count: int):
        del token_count
        raise AssertionError("invalid finalization must fail before workspace access")

    def finalize_overflow(self, slots, *, lane):
        from cudnn.moe_ep._megamoe_backend.mxfp8._training_resources import (
            Mxfp8TrainingResourceOwner,
        )

        return Mxfp8TrainingResourceOwner.finalize_overflow(
            self,
            slots,
            lane=lane,
        )

    def close(self) -> None:
        self.close_calls += 1


def _training_contract_resources(
    *,
    owner=None,
    slot_count: int = 2,
    lane_count: int = 1,
):
    from cudnn.moe_ep import MoeEpTrainingResources

    if owner is None:
        owner = _TrainingResourceContractOwner(
            slot_count=slot_count,
            lane_count=lane_count,
        )
    resources = MoeEpTrainingResources(
        owner=owner,
        operator_token=object(),
        weights=SimpleNamespace(mock_training_weights=True),
        slot_count=slot_count,
        lane_count=lane_count,
        device=torch.device("cpu"),
    )
    return resources, owner


def _training_staging_tensors(*, capacity: int | None = None):
    activation, _, _, topk_idx, topk_weights = make_forward_inputs(torch.device("cpu"))
    source = activation.dequantize(torch.bfloat16).contiguous()
    token_count, hidden = source.shape
    top_k = topk_idx.shape[1]
    if capacity is None:
        capacity = token_count
    return {
        "source": source,
        "topk_idx": topk_idx,
        "topk_weights": topk_weights.float().contiguous(),
        "output": torch.empty(
            (capacity, hidden),
            dtype=torch.float8_e4m3fn,
        ),
        "output_sf": torch.empty(
            (capacity, hidden // 32),
            dtype=torch.float8_e8m0fnu,
        ),
        "output_topk_idx": torch.empty(
            (capacity, top_k),
            dtype=torch.int32,
        ),
        "output_topk_weights": torch.empty(
            (capacity, top_k),
            dtype=torch.float32,
        ),
    }


# Forward


_DEFAULT_FORWARD_CONFIG = {
    "num_experts": 2,
    "hidden_size": 128,
    "intermediate_size": 256,
    "top_k": 2,
    "max_tokens_per_rank": 5,
    "apply_topk_in_fc1": True,
    "combine_format": "bf16",
    "output_format": "bf16",
}
_REFERENCE_CLOSE_KWARGS = {"rtol": 0.05, "atol": 0.0625}


def _forward_config(**overrides):
    return {**_DEFAULT_FORWARD_CONFIG, **overrides}


def _output_as_float(output):
    if isinstance(output, torch.Tensor):
        return output.float()
    return output.dequantize()


def _assert_matches_reference(actual, expected):
    torch.testing.assert_close(
        _output_as_float(actual),
        _output_as_float(expected),
        **_REFERENCE_CLOSE_KWARGS,
    )


def _naive_reference(
    activation,
    fc1_weight,
    fc2_weight,
    topk_idx,
    topk_weights,
    *,
    apply_topk_in_fc1,
    clamp=None,
    combine_format=MoeFormat.BF16,
    intermediate_format=None,
    apply_topk_after_combine=False,
):
    token_count, top_k = topk_idx.shape
    hidden_size = activation.shape[1]
    intermediate_size = fc2_weight.shape[1]
    combine = torch.zeros(
        token_count,
        top_k,
        hidden_size,
        dtype=torch.float32,
        device=activation.device,
    )
    for token in range(token_count):
        for slot in range(top_k):
            expert = int(topk_idx[token, slot])
            if expert == -1:
                continue
            gate_up = activation[token].float() @ fc1_weight[expert].float()
            gate, up = gate_up.split(intermediate_size)
            if clamp is not None:
                gate = gate.clamp(max=clamp)
                up = up.clamp(-clamp, clamp)
            intermediate = F.silu(gate) * up
            route_weight = topk_weights[token, slot].float()
            if apply_topk_in_fc1:
                intermediate = intermediate * route_weight
            if intermediate_format is not None:
                intermediate = quantize_blockwise(
                    intermediate,
                    intermediate_format,
                ).dequantize()
            result = intermediate @ fc2_weight[expert].float()
            if not apply_topk_in_fc1 and not apply_topk_after_combine:
                result = result * route_weight
            result = forward_combine_round_trip(result, combine_format)
            if not apply_topk_in_fc1 and apply_topk_after_combine:
                result = result * route_weight
            combine[token, slot] = result
    return combine.sum(dim=1).to(torch.bfloat16)


def _as_reference_tensor(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor
    return ReferenceBlockScaledTensor(
        data=tensor.data,
        scale=tensor.scale,
        format=tensor.format.value,
        logical_shape=tensor.logical_shape,
        axis=tensor.axis,
    )


def _reference_args(args):
    return (
        _as_reference_tensor(args[0]),
        _as_reference_tensor(args[1]),
        _as_reference_tensor(args[2]),
        args[3],
        args[4],
    )


def _reference_forward(args, **overrides):
    # Rubin's fused FC1 epilogue stores the post-SwiGLU intermediate as MXFP8
    # before FC2 consumes it. Keep MoeEpReference's default raw semantics for
    # its standalone tests, but model the device precision for API comparisons.
    config = _forward_config(**overrides)
    config.pop("tuning", None)
    config.setdefault("intermediate_format", "mxfp8")
    return MoeEpReference(**config)(*_reference_args(args))


def _sm107_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("Rubin MXFP8 forward requires CUDA")
    device = torch.device("cuda", 0)
    if torch.cuda.get_device_capability(device) != (10, 7):
        pytest.skip("Rubin MXFP8 forward requires exactly SM107 (compute capability 10.7)")
    return device


def _require_distributed_sm107(world_size: int) -> None:
    if not dist.is_available() or not dist.is_nccl_available():
        pytest.skip("multi-GPU Rubin MXFP8 forward requires NCCL")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"multi-GPU Rubin MXFP8 forward requires {world_size} GPUs")
    if any(torch.cuda.get_device_capability(index) != (10, 7) for index in range(world_size)):
        pytest.skip("multi-GPU Rubin MXFP8 forward requires exactly SM107 " "(compute capability 10.7) on every rank")
    try:
        import nvshmem.core  # noqa: F401
    except (ImportError, OSError):
        pytest.skip("multi-GPU Rubin MXFP8 forward requires NVSHMEM")


def _make_forward_case(
    device: torch.device,
    *,
    experts: int,
    tokens: int,
    hidden: int,
    intermediate: int,
    top_k: int,
    index_dtype: torch.dtype,
    weight_dtype: torch.dtype,
):
    """Build a deterministic supported case for the shape/format matrix."""

    seed = 20260811 + experts * 1009 + tokens * 101 + hidden * 11 + intermediate + top_k
    generator = torch.Generator(device=device).manual_seed(seed)
    activation = quantize_mxfp8(
        torch.randn(tokens, hidden, generator=generator, device=device),
        axis=1,
    )
    fc1_weight = quantize_mxfp8(
        torch.randn(
            experts,
            hidden,
            2 * intermediate,
            generator=generator,
            device=device,
        )
        / 8,
        axis=1,
    )
    fc2_weight = quantize_mxfp8(
        torch.randn(
            experts,
            intermediate,
            hidden,
            generator=generator,
            device=device,
        )
        / 8,
        axis=1,
    )
    topk_idx = torch.arange(tokens * top_k, device=device).reshape(tokens, top_k).remainder(experts).to(index_dtype)
    topk_weights = torch.arange(
        1,
        tokens * top_k + 1,
        dtype=torch.float32,
        device=device,
    ).reshape(tokens, top_k)
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)
    return (
        activation,
        fc1_weight,
        fc2_weight,
        topk_idx,
        topk_weights.to(weight_dtype),
    )


def _stress_backend_reuse(
    op,
    args,
    original_topk_idx,
    original_topk_weights,
    device,
    *,
    check_weight_refresh,
):
    backend = op._forward_backend
    assert backend is not None
    compiled = backend._compiled
    plan_workspace = backend._plan._workspace
    weight_refresh_count = backend._adapter.weight_refresh_count if check_weight_refresh else None
    alternate_stream = torch.cuda.Stream(device=device)

    for iteration in range(100):
        args[3].copy_(original_topk_idx)
        args[4].copy_(original_topk_weights * float((iteration % 7) + 1) / 7.0)
        if iteration % 10 == 0:
            args[3].fill_(-1)
        stream = torch.cuda.current_stream(device) if iteration % 2 == 0 else alternate_stream
        with torch.cuda.stream(stream):
            stressed = op(*args)
        stream.synchronize()
        if iteration % 10 == 0:
            assert _output_as_float(stressed).eq(0).all()
        else:
            assert torch.isfinite(_output_as_float(stressed)).all()
        assert backend._compiled is compiled
        assert backend._plan._workspace is plan_workspace
        if weight_refresh_count is not None:
            assert backend._adapter.weight_refresh_count == weight_refresh_count


def _replay_cuda_graph(
    op,
    args,
    original_topk_idx,
    expected,
    device,
    *,
    synchronize_ranks=None,
):
    synchronize_ranks = synchronize_ranks or (lambda: None)
    op.warmup(*args)
    synchronize_ranks()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = op(*args)
    synchronize_ranks()

    for replay in range(20):
        if replay % 2:
            args[3].fill_(-1)
        else:
            args[3].copy_(original_topk_idx)
        synchronize_ranks()
        graph.replay()
        torch.cuda.synchronize(device)
        if replay % 2:
            assert _output_as_float(graph_output).eq(0).all()
        else:
            _assert_matches_reference(graph_output, expected)


# Backward


_BACKWARD_CLOSE_KWARGS = (
    {"rtol": 0.15, "atol": 0.125},  # grad_activation is BF16-rounded.
    {"rtol": 0.15, "atol": 0.125},  # router-weight gradient.
)
_WGRAD_CLOSE_KWARGS = {"rtol": 0.2, "atol": 0.25}
_GROUPED_WGRAD_CLOSE_KWARGS = {"rtol": 0.1, "atol": 0.1}


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _unpack_wgrad_scale_part(
    packed: torch.Tensor,
    rows: int,
    columns: int,
) -> torch.Tensor:
    """Invert grouped-wgrad's 128x4 scale-atom swizzle."""

    padded_rows = _round_up(rows, 128)
    padded_columns = _round_up(columns, 4)
    row_atoms = padded_rows // 128
    column_atoms = padded_columns // 4
    atom_count = row_atoms * column_atoms
    expected = padded_rows * padded_columns
    if packed.numel() != expected:
        raise ValueError(f"packed scale part has {packed.numel()} bytes, expected {expected}")
    blocked = (
        packed.reshape(atom_count, 32, 4, 4).transpose(1, 2).reshape(row_atoms, column_atoms, 128, 4).permute(0, 2, 1, 3).reshape(padded_rows, padded_columns)
    )
    return blocked[:rows, :columns].view(torch.float8_e8m0fnu).float()


def _dequantize_wgrad_operand(
    data: torch.Tensor,
    scales: torch.Tensor,
    expert_offsets: torch.Tensor,
    *,
    k_dim: int,
) -> torch.Tensor:
    """Decode one public grouped-wgrad operand without launching a GEMM."""

    if data.ndim != 2 or k_dim not in (0, 1):
        raise ValueError("wgrad operand must be rank 2 with k_dim 0 or 1")
    non_k = int(data.shape[1 - k_dim])
    padded_non_k = _round_up(non_k, 128)
    flat_scales = scales.view(torch.uint8).reshape(-1)
    output = torch.zeros(data.shape, dtype=torch.float32, device=data.device)
    ends = [int(value) for value in expert_offsets.detach().cpu().tolist()]
    k_capacity = int(data.shape[k_dim])
    previous = 0
    scale_byte_offset = 0
    for end in ends:
        if end < previous or end > k_capacity:
            raise ValueError("expert offsets must be nondecreasing and fit the operand " f"K capacity ({k_capacity})")
        extent = end - previous
        if extent % 32:
            raise ValueError("each padded expert K extent must be divisible by 32")
        if extent == 0:
            continue
        scale_columns = _round_up(extent // 32, 4)
        scale_byte_count = padded_non_k * scale_columns
        if scale_byte_offset + scale_byte_count > flat_scales.numel():
            raise ValueError("expert offsets exceed the scale tensor")
        part = flat_scales.narrow(
            0,
            scale_byte_offset,
            scale_byte_count,
        )
        logical_scale = _unpack_wgrad_scale_part(
            part,
            non_k,
            extent // 32,
        )
        if k_dim == 1:
            expanded_scale = logical_scale.repeat_interleave(32, dim=1)
            output[:, previous:end] = data[:, previous:end].float() * expanded_scale
        else:
            expanded_scale = logical_scale.repeat_interleave(
                32,
                dim=1,
            ).transpose(0, 1)
            output[previous:end, :] = data[previous:end, :].float() * expanded_scale
        previous = end
        scale_byte_offset += scale_byte_count

    if previous < k_capacity:
        capacity_tail = data.narrow(k_dim, previous, k_capacity - previous)
        if bool(capacity_tail.float().ne(0).any().item()):
            raise ValueError("unused WGrad operand capacity tail must contain zero data")
    scale_tail = flat_scales[scale_byte_offset:]
    if scale_tail.numel() and bool(scale_tail.ne(127).any().item()):
        raise ValueError("unused WGrad operand capacity tail must contain neutral E8M0 scales")
    return output


def _dense_wgrads_from_operands(operands):
    """Reference grouped matmuls over the exported operand ABI."""

    fc1_a = _dequantize_wgrad_operand(
        operands.fc1_a,
        operands.fc1_sfa,
        operands.expert_offsets,
        k_dim=1,
    )
    fc1_b = _dequantize_wgrad_operand(
        operands.fc1_b,
        operands.fc1_sfb,
        operands.expert_offsets,
        k_dim=0,
    )
    fc2_a = _dequantize_wgrad_operand(
        operands.fc2_a,
        operands.fc2_sfa,
        operands.expert_offsets,
        k_dim=1,
    )
    fc2_b = _dequantize_wgrad_operand(
        operands.fc2_b,
        operands.fc2_sfb,
        operands.expert_offsets,
        k_dim=0,
    )
    fc1_parts = []
    fc2_parts = []
    ends = [int(value) for value in operands.expert_offsets.detach().cpu().tolist()]
    valid_counts = [int(value) for value in operands.valid_route_counts.detach().cpu().tolist()]
    if len(ends) != len(valid_counts):
        raise ValueError("expert offsets and valid route counts must have equal size")

    previous = 0
    for expert, (end, valid_count) in enumerate(zip(ends, valid_counts)):
        extent = end - previous
        if valid_count < 0 or valid_count > extent:
            raise ValueError(f"expert {expert} valid route count {valid_count} exceeds " f"its padded extent {extent}")
        valid_end = previous + valid_count
        for name, tensor, k_dim in (
            ("fc1_a", fc1_a, 1),
            ("fc1_b", fc1_b, 0),
            ("fc2_a", fc2_a, 1),
            ("fc2_b", fc2_b, 0),
        ):
            padding = tensor.narrow(k_dim, valid_end, end - valid_end)
            if bool(padding.ne(0).any().item()):
                raise ValueError(f"{name} expert {expert} padded rows must decode to zero")
        fc1_parts.append(fc1_a[:, previous:valid_end] @ fc1_b[previous:valid_end, :])
        fc2_parts.append(fc2_a[:, previous:valid_end] @ fc2_b[previous:valid_end, :])
        previous = end
    return torch.stack(fc1_parts), torch.stack(fc2_parts)


def _run_grouped_wgrad_kernel(
    operands,
    prefix: str,
    *,
    wgrad_tensor=None,
    accumulate_on_output: bool = False,
    current_stream=None,
):
    """Run one fixed-capacity operand bundle through production WGrad."""

    import cudnn

    if prefix not in ("fc1", "fc2"):
        raise ValueError(f"prefix must be 'fc1' or 'fc2', got {prefix!r}")
    # Graph callers provide one persistent output per training slot. This is
    # currently also the isolation key for a temporary production-WGrad
    # workaround: an EP2 graph with two same-signature calls produced correct
    # operands but corrupted the second WGrad when both calls shared the
    # cached API object's mutable TMA descriptor workspace. Distinct fixed
    # outputs make the calls use distinct workspaces. The production fix
    # should instead share the compiled kernel while owning descriptor
    # workspace per graph call site, after which output identity must no
    # longer participate in the compile cache key.
    return cudnn.grouped_gemm_wgrad_wrapper_sm100(
        a_tensor=getattr(operands, f"{prefix}_a"),
        b_tensor=getattr(operands, f"{prefix}_b"),
        sfa_tensor=getattr(operands, f"{prefix}_sfa"),
        sfb_tensor=getattr(operands, f"{prefix}_sfb"),
        offsets_tensor=operands.expert_offsets,
        output_mode="dense",
        wgrad_tensor=wgrad_tensor,
        wgrad_dtype=torch.bfloat16,
        acc_dtype=torch.float32,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        sf_vec_size=32,
        accumulate_on_output=accumulate_on_output,
        input_order="tensor2d",
        current_stream=current_stream,
    )["wgrad_tensor"]


def _dense_wgrads_from_grouped_kernel(
    operands,
    *,
    wgrad_tensors=None,
    accumulate_on_output: bool = False,
    current_stream=None,
):
    """Run both fixed-capacity operand bundles through production WGrad."""

    if wgrad_tensors is None:
        wgrad_tensors = (None, None)
    if len(wgrad_tensors) != 2:
        raise ValueError("wgrad_tensors must contain FC1 and FC2 outputs")
    return tuple(
        _run_grouped_wgrad_kernel(
            operands,
            prefix,
            wgrad_tensor=output,
            accumulate_on_output=accumulate_on_output,
            current_stream=current_stream,
        )
        for prefix, output in zip(("fc1", "fc2"), wgrad_tensors)
    )


def _assert_grouped_wgrads_match_reference(
    actual,
    expected,
    *,
    reference_name: str,
    close_kwargs=None,
) -> None:
    """Compare grouped-kernel FC1/FC2 outputs and report useful error maxima."""

    if close_kwargs is None:
        close_kwargs = _GROUPED_WGRAD_CLOSE_KWARGS
    for name, actual_dw, expected_dw in zip(
        ("grad_fc1_weight", "grad_fc2_weight"),
        actual,
        expected,
    ):
        actual_fp32 = actual_dw.float()
        expected_fp32 = expected_dw.float()
        absolute_error = (actual_fp32 - expected_fp32).abs()
        max_absolute_error = absolute_error.max().item()
        max_relative_error = (
            (absolute_error / expected_fp32.abs().clamp_min(1.0e-6)).max().item()
        )
        torch.testing.assert_close(
            actual_fp32,
            expected_fp32,
            msg=lambda default, name=name: (
                f"{name} does not match {reference_name}; "
                f"max_abs_error={max_absolute_error:.6g}, "
                f"max_rel_error={max_relative_error:.6g}\n{default}"
            ),
            **close_kwargs,
        )


def _reference_backward(config) -> MoeEpReference:
    options = dict(config)
    for production_only in (
        "drop_on_overflow",
        "ep_global_ranks",
        "ep_rank",
        "ep_size",
        "experts_per_rank",
        "max_recv_size_per_rank",
        "sf_padding_size",
        "tuning",
    ):
        options.pop(production_only, None)
    options["intermediate_format"] = "mxfp8"
    options["backward_operand_format"] = "mxfp8"
    return MoeEpReference(**options)


def _fixed_training_weights(args):
    """Build the four stable MXFP8 source packs required by training."""

    from cudnn.moe_ep import MoeEpTrainingWeights
    from cudnn.moe_ep._megamoe_backend.mxfp8._adapter import (
        _quantize_plain_mxfp8,
    )

    fc1_weight = args[1]
    fc2_weight = args[2]
    dense_fc1 = fc1_weight if isinstance(fc1_weight, torch.Tensor) else fc1_weight.dequantize()
    dense_fc2 = fc2_weight if isinstance(fc2_weight, torch.Tensor) else fc2_weight.dequantize()
    return MoeEpTrainingWeights(
        forward_fc1=(_quantize_plain_mxfp8(dense_fc1, axis=1) if isinstance(fc1_weight, torch.Tensor) else fc1_weight),
        forward_fc2=(_quantize_plain_mxfp8(dense_fc2, axis=1) if isinstance(fc2_weight, torch.Tensor) else fc2_weight),
        backward_w2_transpose=_quantize_plain_mxfp8(
            dense_fc2.transpose(1, 2).contiguous(),
            axis=1,
        ),
        backward_w1_transpose=_quantize_plain_mxfp8(
            dense_fc1.transpose(1, 2).contiguous(),
            axis=1,
        ),
    )


def _fixed_training_reference(
    args,
    grad_output,
    *,
    combine_format,
    gate_up_clamp,
    ep_group=None,
    num_experts=None,
    **config_overrides,
):
    """Run the standalone oracle for EP1 or a distributed EP group."""

    ep_size = 1 if ep_group is None else dist.get_world_size(ep_group)
    local_experts = int(args[1].shape[0])
    if num_experts is None:
        num_experts = local_experts * ep_size
    reference_config = _forward_config(**config_overrides)
    reference_config.update(
        num_experts=num_experts,
        hidden_size=int(args[0].shape[1]),
        intermediate_size=int(args[2].shape[1]),
        top_k=int(args[3].shape[1]),
        max_tokens_per_rank=config_overrides.get(
            "max_tokens_per_rank",
            int(args[0].shape[0]),
        ),
        ep_group=ep_group,
        combine_format=combine_format,
        gate_up_clamp=gate_up_clamp,
        generate_c=True,
        backward_wgrad_mode="operands",
        # The standalone operand oracle's legacy ABI uses 256-row
        # segments. Production fixed resources use 128-row segments;
        # their represented dense gradients are compared below.
        token_padding_size=256,
    )
    reference = _reference_backward(reference_config)
    reference_args = _reference_args(args)
    output, fc1_c, route_metadata, forward_stash = reference(*reference_args)
    grad_activation, grad_topk_weights, wgrad_operands = reference.backward(
        grad_output,
        *reference_args[1:],
        fc1_c,
        route_metadata,
        wgrad_forward_stash=forward_stash,
    )
    return (
        output,
        grad_activation,
        grad_topk_weights,
        wgrad_operands,
    )


def _grad_output(
    device: torch.device,
    token_count: int,
    *,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return (
        torch.randn(
            token_count,
            128,
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        / 8
    )


def _expected_backward(reference, grad_output, args, stash):
    return reference.backward(
        grad_output,
        *_reference_args(args)[1:],
        *stash,
    )


def _assert_backward_matches(actual, expected, topk_idx) -> None:
    assert len(actual) == len(expected) == 2
    for name, gradient, reference, close_kwargs in zip(
        ("grad_activation", "grad_topk_weights"),
        actual,
        expected,
        _BACKWARD_CLOSE_KWARGS,
    ):
        assert gradient.shape == reference.shape
        assert gradient.dtype == torch.float32
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(
            gradient,
            reference,
            msg=lambda default, name=name: (f"{name} does not match the backward reference\n{default}"),
            **close_kwargs,
        )

    dropped = topk_idx == -1
    assert actual[1][dropped].eq(0).all()


def _assert_wgrads_match_reference(
    actual,
    expected,
    *,
    expected_dense=None,
) -> None:
    """Compare fixed-capacity production operands with standalone dense dW."""

    torch.testing.assert_close(
        actual.valid_route_counts,
        expected.valid_route_counts,
        rtol=0,
        atol=0,
        msg="valid route counts differ from the independent reference",
    )
    actual_dense = _dense_wgrads_from_operands(actual)
    if expected_dense is None:
        expected_dense = expected.dense_wgrads()
    for name, actual_dw, expected_dw in zip(
        ("grad_fc1_weight", "grad_fc2_weight"),
        actual_dense,
        expected_dense,
    ):
        torch.testing.assert_close(
            actual_dw,
            expected_dw,
            msg=lambda default, name=name: (f"{name} does not match the independent reference\n{default}"),
            **_WGRAD_CLOSE_KWARGS,
        )


_TRAINING_WGRAD_DATA_FIELDS = ("fc1_a", "fc1_b", "fc2_a", "fc2_b")
_TRAINING_WGRAD_SF_FIELDS = ("fc1_sfa", "fc1_sfb", "fc2_sfa", "fc2_sfb")
_TRAINING_WEIGHT_FIELDS = (
    "forward_fc1",
    "forward_fc2",
    "backward_w2_transpose",
    "backward_w1_transpose",
)


def _fixed_training_case(device):
    args = list(make_forward_inputs(device))
    args[0] = args[0].dequantize(torch.bfloat16)
    args[4] = args[4].float()
    args[3].fill_(-1)
    args[4].zero_()
    args[3][0, 0] = 0
    args[4][0, 0] = 1
    grad_output = _grad_output(
        device,
        args[0].shape[0],
        seed=20260828,
    )
    return args, grad_output


def _assert_fixed_training_matches_reference(
    actual,
    expected,
    topk_idx,
) -> None:
    actual_y, actual_dx, actual_dprob, actual_wgrads = actual
    expected_y, expected_dx, expected_dprob, expected_wgrads = expected
    _assert_matches_reference(actual_y, expected_y)
    _assert_backward_matches(
        (actual_dx, actual_dprob),
        (expected_dx, expected_dprob),
        topk_idx,
    )
    _assert_wgrads_match_reference(actual_wgrads, expected_wgrads)


def _run_fixed_training_batch(resources, lane, cases):
    """Run refresh, ordered forwards/backwards, and one overflow finalization."""

    resources.refresh_weights()
    outputs = [resources.forward(slot, lane, args[0], args[3], args[4]) for slot, args, _ in cases]
    backwards = [resources.backward(slot, lane, grad_output) for slot, _, grad_output in cases]
    overflow = resources.finalize_overflow(
        tuple(slot for slot, _, _ in cases),
        lane,
    )
    return tuple(
        SimpleNamespace(
            y=output,
            dx=backward[0],
            dprob=backward[1],
            wgrads=backward[2],
            overflow=overflow,
        )
        for output, backward in zip(outputs, backwards)
    )


def _capture_fixed_training_batch(
    resources,
    lane,
    cases,
    capture_stream,
    *,
    grouped_wgrad_outputs=None,
):
    """Capture the shared fixed-training sequence for one or more slots."""

    if grouped_wgrad_outputs is not None and len(grouped_wgrad_outputs) != len(cases):
        raise ValueError("grouped_wgrad_outputs must match the captured case count")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        actuals = _run_fixed_training_batch(resources, lane, cases)
        grouped_wgrads = (
            None
            if grouped_wgrad_outputs is None
            else tuple(
                _dense_wgrads_from_grouped_kernel(
                    actual.wgrads,
                    wgrad_tensors=outputs,
                )
                for actual, outputs in zip(actuals, grouped_wgrad_outputs)
            )
        )
    capture_stream.synchronize()
    return SimpleNamespace(
        graph=graph,
        actuals=actuals,
        grouped_wgrads=grouped_wgrads,
        public_pointers=tuple(_training_public_pointers(actual) for actual in actuals),
    )


def _training_source_pointers(case) -> dict[str, int]:
    return {
        name: getattr(case, name).data_ptr()
        for name in (
            "activation",
            "topk_idx",
            "topk_weights",
            "grad_output",
        )
    }


def _training_weight_source_pointers(weights) -> dict[str, int]:
    return {f"{name}.{part}": getattr(getattr(weights, name), part).data_ptr() for name in _TRAINING_WEIGHT_FIELDS for part in ("data", "scale")}


def _training_weight_source_values(weights) -> dict[str, torch.Tensor]:
    return {f"{name}.{part}": getattr(getattr(weights, name), part).clone() for name in _TRAINING_WEIGHT_FIELDS for part in ("data", "scale")}


def _assert_training_weight_sources_changed(weights, previous) -> None:
    for name in _TRAINING_WEIGHT_FIELDS:
        for part in ("data", "scale"):
            assert not torch.equal(
                getattr(getattr(weights, name), part),
                previous[f"{name}.{part}"],
            )


def _training_public_pointers(actual) -> dict[str, int]:
    pointers = {
        "y": actual.y.data_ptr(),
        "dx": actual.dx.data_ptr(),
        "dprob": actual.dprob.data_ptr(),
        "overflow": actual.overflow.data_ptr(),
    }
    pointers.update(
        {
            f"wgrads.{name}": getattr(actual.wgrads, name).data_ptr()
            for name in (
                *_TRAINING_WGRAD_DATA_FIELDS,
                *_TRAINING_WGRAD_SF_FIELDS,
                "expert_offsets",
                "valid_route_counts",
            )
        }
    )
    return pointers


def _prefill_training_graph_sentinels(slot_views, actual) -> None:
    """Poison every history-sensitive full-capacity destination."""

    slot_views.routing_topk_idx.fill_(0x1A2B3C)
    slot_views.routing_topk_weights.fill_(31.25)
    slot_views.forward_output.fill_(29.0)
    slot_views.backward_output.fill_(-27.0)
    slot_views.grad_activation.fill_(23.0)
    slot_views.dprob.fill_(-19.0)
    slot_views.expert_offsets.fill_(-17)
    slot_views.valid_route_counts.fill_(-13)
    for name in _TRAINING_WGRAD_DATA_FIELDS:
        getattr(actual.wgrads, name).fill_(1.0)
    for name in _TRAINING_WGRAD_SF_FIELDS:
        getattr(actual.wgrads, name).view(torch.uint8).fill_(0)


def _assert_training_graph_tails_are_reset(
    slot_views,
    actual,
    *,
    token_count: int,
    capacity: int,
) -> None:
    if token_count < capacity:
        assert slot_views.routing_topk_idx[token_count:].eq(-1).all()
        assert slot_views.routing_topk_weights[token_count:].eq(0).all()
        assert slot_views.forward_output[token_count:].eq(0).all()
        assert slot_views.backward_output[token_count:].eq(0).all()
        assert slot_views.grad_activation[token_count:].eq(0).all()
        assert slot_views.dprob[token_count:].eq(0).all()

    counts = actual.wgrads.valid_route_counts.detach().cpu().tolist()
    expected_offsets = []
    offset = 0
    for count in counts:
        offset += (int(count) + 127) // 128 * 128
        expected_offsets.append(offset)
    assert actual.wgrads.expert_offsets.detach().cpu().tolist() == expected_offsets


def _copy_training_weight_sources_(destination, source) -> None:
    for name in _TRAINING_WEIGHT_FIELDS:
        destination_pack = getattr(destination, name)
        source_pack = getattr(source, name)
        destination_pack.data.copy_(source_pack.data)
        destination_pack.scale.copy_(source_pack.scale)


def _fixed_training_drop_overflow_case(device):
    base_args, base_grad_output = _fixed_training_case(device)
    topk_idx = torch.tensor(
        [[0, 1]],
        dtype=torch.int32,
        device=device,
    )
    topk_weights = torch.tensor(
        [[0.75, 0.25]],
        dtype=torch.float32,
        device=device,
    )
    args = (
        base_args[0][:1].clone(),
        base_args[1],
        base_args[2],
        topk_idx,
        topk_weights,
    )
    return args, base_grad_output[:1].clone()


def _fixed_training_drop_overflow_reference(
    args,
    grad_output,
    *,
    drop_expert1,
):
    reference_topk_idx = args[3].clone()
    if drop_expert1:
        assert reference_topk_idx.shape == (1, 2)
        assert reference_topk_idx.detach().cpu().tolist() == [[0, 1]]
        reference_topk_idx[0, 1] = -1
    reference_args = (
        args[0],
        args[1],
        args[2],
        reference_topk_idx,
        args[4].clone(),
    )
    return (
        _fixed_training_reference(
            reference_args,
            grad_output,
            combine_format="bf16",
            gate_up_clamp=None,
        ),
        reference_topk_idx,
    )


def _assert_fixed_training_drop_overflow_result(
    actual,
    expected,
    reference_topk_idx,
    *,
    expected_overflow,
):
    assert actual.overflow.eq(expected_overflow).all()
    _assert_fixed_training_matches_reference(
        (actual.y, actual.dx, actual.dprob, actual.wgrads),
        expected,
        reference_topk_idx,
    )

    if expected_overflow:
        assert reference_topk_idx[0, 1].eq(-1)
        assert actual.dprob[0, 1].eq(0)
        assert actual.wgrads.valid_route_counts.detach().cpu().tolist() == [1, 0]
        # Expert 0 owns the first 128-row padded segment. Expert 1 starts at
        # pool capacity and therefore has no retained segment or dense dW.
        assert actual.wgrads.expert_offsets.detach().cpu().tolist() == [128, 128]
        dense_dw1, dense_dw2 = _dense_wgrads_from_operands(actual.wgrads)
        assert dense_dw1[1].eq(0).all()
        assert dense_dw2[1].eq(0).all()
