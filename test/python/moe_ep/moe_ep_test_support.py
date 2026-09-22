# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared data, forward, and backward support for MoE EP tests."""

from __future__ import annotations

# Common

from dataclasses import dataclass
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
    "_assert_backward_matches",
    "_assert_grouped_wgrads_match_reference",
    "_assert_matches_reference",
    "_assert_wgrads_match_reference",
    "_dense_wgrads_from_operands",
    "_dense_wgrads_from_grouped_kernel",
    "_fixed_training_reference",
    "_fixed_training_weights",
    "_allocate_stateless_training_outputs",
    "_allocate_training_weight_staging",
    "_make_discrete_training_weights",
    "_forward_config",
    "_moe_ep_config",
    "_grad_output",
    "_make_forward_case",
    "_naive_reference",
    "_output_as_float",
    "_poison_pre_reduced_for_test",
    "_poison_training_outputs_for_test",
    "_reference_backward",
    "_reference_forward",
    "_replay_cuda_graph",
    "_require_distributed_sm107",
    "_run_grouped_wgrad_kernel",
    "_sm107_device",
    "_stress_backend_reuse",
    "_training_abi_prepared",
    "_training_config",
    "_training_graph_pattern",
    "_training_prepared_pair",
    "make_distributed_forward_inputs",
    "make_forward_inputs",
    "make_training_graph_pattern_inputs",
    "make_training_graph_pattern_weights",
    "quantize_mxfp8",
]


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
        [[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]],
        dtype=torch.int32,
        device=device,
    )
    topk_weights = torch.tensor(
        [
            [0.75, 0.25],
            [0.625, 0.375],
            [0.8, 0.2],
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


@dataclass(frozen=True)
class _TrainingGraphPattern:
    name: str
    num_experts: int
    hidden_size: int
    intermediate_size: int
    top_k: int
    max_tokens_per_rank: int
    physical_recv_pool_size: int
    combine_format: str


def _training_graph_pattern(
    name: str,
    world_size: int,
) -> _TrainingGraphPattern:
    """Resolve one reusable public-MoeEP graph workload pattern."""

    if name == "smoke":
        return _TrainingGraphPattern(
            name=name,
            num_experts=2 * world_size,
            hidden_size=128,
            intermediate_size=256,
            top_k=2,
            max_tokens_per_rank=8,
            physical_recv_pool_size=128,
            combine_format="bf16",
        )
    if name == "ds3_ep4_v1":
        if world_size != 4:
            raise ValueError("ds3_ep4_v1 training graph pattern requires world_size=4, " f"got {world_size}")
        return _TrainingGraphPattern(
            name=name,
            num_experts=32,
            hidden_size=7168,
            intermediate_size=2048,
            top_k=8,
            max_tokens_per_rank=4096,
            physical_recv_pool_size=131968,
            combine_format="mxfp8",
        )
    raise ValueError("training graph pattern must be 'smoke' or 'ds3_ep4_v1', " f"got {name!r}")


def _constant_mxfp8(
    shape: tuple[int, ...],
    *,
    axis: int,
    value: float,
    device: torch.device,
):
    """Allocate deterministic MXFP8 data without a temporary FP32 tensor."""

    from cudnn import BlockScaledTensor

    canonical_axis = axis % len(shape)
    scale_shape = list(shape)
    scale_shape[canonical_axis] = (scale_shape[canonical_axis] + 31) // 32
    return BlockScaledTensor(
        data=torch.full(
            shape,
            value,
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        scale=torch.full(
            tuple(scale_shape),
            1.0,
            dtype=torch.float8_e8m0fnu,
            device=device,
        ),
        format="mxfp8",
        logical_shape=shape,
        axis=canonical_axis,
    )


def make_training_graph_pattern_inputs(
    pattern: _TrainingGraphPattern,
    rank: int,
    world_size: int,
    device: torch.device,
):
    """Build rank-local inputs for a reusable training Graph pattern."""

    if pattern.name == "smoke":
        args = make_distributed_forward_inputs(
            rank,
            world_size,
            device,
        )
        return (*args[:4], args[4].float().contiguous())

    experts_per_rank = pattern.num_experts // world_size
    # Keep the optimized preset on its only qualified runtime workload.
    # max_tokens_per_rank is not merely spare capacity for ds3_ep4_v1:
    # upstream qualified every rank with exactly T4096 inputs.
    token_count = pattern.max_tokens_per_rank
    activation = _constant_mxfp8(
        (token_count, pattern.hidden_size),
        axis=1,
        value=0.125,
        device=device,
    )
    fc1_weight = _constant_mxfp8(
        (
            experts_per_rank,
            pattern.hidden_size,
            2 * pattern.intermediate_size,
        ),
        axis=1,
        value=0.03125,
        device=device,
    )
    fc2_weight = _constant_mxfp8(
        (
            experts_per_rank,
            pattern.intermediate_size,
            pattern.hidden_size,
        ),
        axis=1,
        value=0.03125,
        device=device,
    )
    tokens = torch.arange(
        token_count,
        dtype=torch.int64,
        device=device,
    ).unsqueeze(1)
    slots = torch.arange(
        pattern.top_k,
        dtype=torch.int64,
        device=device,
    ).unsqueeze(0)
    destination_rank = (rank + slots) % world_size
    local_expert = (tokens + torch.div(slots, world_size, rounding_mode="floor")) % experts_per_rank
    topk_idx = (destination_rank * experts_per_rank + local_expert).to(torch.int32)
    topk_weights = torch.arange(
        1,
        pattern.top_k + 1,
        dtype=torch.float32,
        device=device,
    ).expand(token_count, -1)
    topk_weights = (topk_weights / topk_weights.sum(dim=1, keepdim=True)).contiguous()
    return (
        activation,
        fc1_weight,
        fc2_weight,
        topk_idx,
        topk_weights,
    )


def make_training_graph_pattern_weights(
    pattern: _TrainingGraphPattern,
    inputs,
):
    """Build matching forward/backward source weights for one pattern."""

    if pattern.name == "smoke":
        return _fixed_training_weights(inputs)

    from cudnn import MoeEpBackwardWeights, MoeEpForwardWeights

    fc1_weight = inputs[1]
    fc2_weight = inputs[2]
    device = fc1_weight.device
    experts_per_rank = int(fc1_weight.logical_shape[0])
    forward = MoeEpForwardWeights(
        fc1=fc1_weight,
        fc2=fc2_weight,
    )
    backward = MoeEpBackwardWeights(
        w2_transpose=_constant_mxfp8(
            (
                experts_per_rank,
                pattern.hidden_size,
                pattern.intermediate_size,
            ),
            axis=1,
            value=0.03125,
            device=device,
        ),
        w1_transpose=_constant_mxfp8(
            (
                experts_per_rank,
                2 * pattern.intermediate_size,
                pattern.hidden_size,
            ),
            axis=1,
            value=0.03125,
            device=device,
        ),
    )
    return forward, backward


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
    from cudnn import MoeEpFc1WeightLayout
    from cudnn.moe_ep._config import (
        ResolvedMoeEpConfig,
        _ResolvedMoeEpTopology,
    )

    ep_size = overrides.pop("ep_size", 1)
    ep_rank = overrides.pop("ep_rank", 0)
    ep_global_ranks = overrides.pop("ep_global_ranks", ())
    experts_per_rank = overrides.pop("experts_per_rank", None)
    values = {
        "num_experts": 2,
        "hidden_size": 128,
        "intermediate_size": 256,
        "top_k": 2,
        "max_tokens_per_rank": 4,
        "physical_recv_pool_rows": 128,
        "drop_on_overflow": True,
        "fc1_weight_layout": (MoeEpFc1WeightLayout.GATE_UP_INTERLEAVED_32),
        **overrides,
    }
    config = _moe_ep_config(**values)
    if experts_per_rank is None:
        experts_per_rank = config.model.num_experts // ep_size
    return ResolvedMoeEpConfig(
        public_config=config,
        topology=_ResolvedMoeEpTopology(
            ep_size=ep_size,
            ep_rank=ep_rank,
            ep_global_ranks=ep_global_ranks,
            experts_per_rank=experts_per_rank,
        ),
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
        config=SimpleNamespace(
            generate_c=True,
            token_padding_block=128,
            sf_padding_block=128,
            kernel_drop_on_overflow=True,
            enable_col_quant=True,
            dfc2_recompute=False,
            dfc2_col_output=False,
            enable_grad_y2_col_quant=False,
            max_tokens_per_rank=(config.public_config.parallel.max_tokens_per_rank),
            top_k=config.public_config.model.top_k,
            weight_storage_mode="contiguous",
        ),
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
        config=SimpleNamespace(
            generate_c=True,
            token_padding_block=128,
            sf_padding_block=128,
            kernel_drop_on_overflow=True,
            enable_col_quant=True,
            dfc2_recompute=True,
            dfc2_col_output=True,
            enable_grad_y2_col_quant=True,
            max_tokens_per_rank=(config.public_config.parallel.max_tokens_per_rank),
            top_k=config.public_config.model.top_k,
            weight_storage_mode="contiguous",
        ),
        workspace_requirements=WorkspaceRequirements.for_mxfp8(
            config,
            kernel_local_workspace_bytes=3072,
            kernel_shared_workspace_bytes=4096,
            backward_dprob_bytes=4 * 2 * 4,
            backward_aux_data_bytes=pool_rows * 512,
            backward_aux_scale_bytes=512 * 8,
        ),
        kernel=SimpleNamespace(
            get_aux_output_shapes=lambda: backward_shapes,
            get_fc1_preact_shape=lambda: forward_shapes["fc1_c"],
        ),
    )
    return forward, backward


def _training_abi_prepared(
    name: str,
    max_recv_size: int = 4,
    *,
    weight_storage_mode: str = "contiguous",
):
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
        physical_recv_pool_size=max_recv_size,
        weight_storage_mode=weight_storage_mode,
        effective_config=lambda: {
            "name": name,
            "max_recv_size_per_rank": max_recv_size,
            "weight_storage_mode": weight_storage_mode,
            "launch_cluster_count": 16,
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


def _moe_ep_config(**values):
    """Build the public nested config used by API-facing tests."""

    from cudnn import (
        MoeEpConfig,
        MoeEpDataPathConfig,
        MoeEpFc1WeightLayout,
        MoeEpModelConfig,
        MoeEpNativeWeightStorageMode,
        MoeEpParallelConfig,
        MoeEpTuningConfig,
    )
    from cudnn import MoeFormat as PublicMoeFormat

    values = dict(values)
    model = MoeEpModelConfig(
        num_experts=values.pop("num_experts"),
        hidden_size=values.pop("hidden_size"),
        intermediate_size=values.pop("intermediate_size"),
        top_k=values.pop("top_k"),
    )
    parallel_names = (
        "ep_group",
        "max_tokens_per_rank",
        "physical_recv_pool_rows",
        "drop_on_overflow",
        "token_padding_size",
        "sf_padding_size",
    )
    parallel_values = {name: values.pop(name) for name in parallel_names if name in values}
    for name in ("output_format", "combine_format"):
        if name in values and not isinstance(values[name], PublicMoeFormat):
            values[name] = PublicMoeFormat(values[name])
    if "fc1_weight_layout" in values and not isinstance(values["fc1_weight_layout"], MoeEpFc1WeightLayout):
        raise TypeError("test configs must use MoeEpFc1WeightLayout directly")
    data_path_names = (
        "output_format",
        "combine_format",
        "apply_topk_in_fc1",
        "fc1_weight_layout",
        "gate_up_clamp",
    )
    data_path_values = {name: values.pop(name) for name in data_path_names if name in values}
    tuning_values = {}
    for name in (
        "inference_tuning",
        "training_forward_tuning",
        "training_backward_tuning",
    ):
        if name in values:
            tuning = values.pop(name)
            if not isinstance(tuning, MoeEpTuningConfig):
                raise TypeError(f"{name} must be a MoeEpTuningConfig")
            tuning_values[name] = tuning
    validation_mode = values.pop("validation_mode", "strict")
    training_weight_storage_mode = values.pop(
        "training_weight_storage_mode",
        MoeEpNativeWeightStorageMode.CONTIGUOUS,
    )
    if not isinstance(
        training_weight_storage_mode,
        MoeEpNativeWeightStorageMode,
    ):
        raise TypeError("training_weight_storage_mode must be a " "MoeEpNativeWeightStorageMode")
    if values:
        raise TypeError(f"unsupported nested MoeEp test config fields: {sorted(values)}")
    return MoeEpConfig(
        model=model,
        parallel=MoeEpParallelConfig(**parallel_values),
        data_path=MoeEpDataPathConfig(**data_path_values),
        training_weight_storage_mode=training_weight_storage_mode,
        validation_mode=validation_mode,
        **tuning_values,
    )


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
            if expert < 0 or expert >= fc1_weight.shape[0]:
                raise ValueError(f"topk_idx contains invalid expert id {expert}")
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
    backend = op._execution_state.backend
    assert backend is not None
    compiled = backend._compiled
    inference_workspace = backend._inference_resources._workspace
    weight_refresh_count = backend._adapter.weight_refresh_count if check_weight_refresh else None
    alternate_stream = torch.cuda.Stream(device=device)

    for iteration in range(100):
        args[3].copy_(original_topk_idx)
        args[4].copy_(original_topk_weights * float((iteration % 7) + 1) / 7.0)
        if iteration % 10 == 0:
            args[3].copy_(original_topk_idx.flip(1))
        stream = torch.cuda.current_stream(device) if iteration % 2 == 0 else alternate_stream
        with torch.cuda.stream(stream):
            stressed = op(*args)
        stream.synchronize()
        assert torch.isfinite(_output_as_float(stressed)).all()
        assert backend._compiled is compiled
        assert backend._inference_resources._workspace is inference_workspace
        if weight_refresh_count is not None:
            assert backend._adapter.weight_refresh_count == weight_refresh_count


def _replay_cuda_graph(
    op,
    args,
    original_topk_idx,
    expected,
    device,
    *,
    alternate_topk_idx,
    alternate_expected,
    poison_before_replay=None,
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
            args[3].copy_(alternate_topk_idx)
        else:
            args[3].copy_(original_topk_idx)
        if poison_before_replay is not None:
            poison_before_replay()
        synchronize_ranks()
        graph.replay()
        torch.cuda.synchronize(device)
        if replay % 2:
            _assert_matches_reference(graph_output, alternate_expected)
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


def _unpack_wgrad_scale_part_bytes(
    packed: torch.Tensor,
    rows: int,
    columns: int,
) -> torch.Tensor:
    """Invert grouped-wgrad's 128x4 scale-atom swizzle as raw bytes."""

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
    return blocked[:rows, :columns]


def _unpack_wgrad_scale_part(
    packed: torch.Tensor,
    rows: int,
    columns: int,
) -> torch.Tensor:
    """Decode one grouped-wgrad scale part as logical E8M0 values."""

    return _unpack_wgrad_scale_part_bytes(packed, rows, columns).view(torch.float8_e8m0fnu).float()


def _poison_training_outputs_for_test(forward_out, backward_out) -> None:
    """Poison caller-owned destinations without production-side fill ops.

    ``backward_out.dprob`` is only the final copy destination here; the private
    kernel accumulation buffer remains zero-initialized in production.
    """

    for tensor in (
        forward_out.output,
        forward_out.fc1_preact,
        forward_out.fc1_a,
    ):
        if tensor is not None:
            tensor.fill_(float("nan"))
    if forward_out.fc1_sfa is not None:
        forward_out.fc1_sfa.view(torch.uint8).fill_(0xFF)
    for tensor in (forward_out.valid_route_counts, forward_out.expert_offsets):
        if tensor is not None:
            tensor.fill_(-1)

    for name in (
        "grad_activation",
        "dprob",
        "fc1_b",
        "fc2_a",
        "fc2_b",
    ):
        tensor = getattr(backward_out, name)
        if tensor is not None:
            tensor.fill_(float("nan"))
    for name in ("fc1_sfb", "fc2_sfa", "fc2_sfb"):
        tensor = getattr(backward_out, name)
        if tensor is not None:
            tensor.view(torch.uint8).fill_(0xFF)


def _poison_pre_reduced_for_test(prepared, shared_workspace) -> None:
    """Poison every persistent standalone-reduce data and scale byte."""

    capacity = prepared.config.max_tokens_per_rank
    poisoned = False
    for offset, bytes_per_token in (
        (
            prepared.pre_reduced_activation_offset,
            prepared.pre_reduced_activation_bytes_per_token,
        ),
        (
            prepared.pre_reduced_activation_sf_offset,
            prepared.pre_reduced_activation_sf_bytes_per_token,
        ),
    ):
        if offset is not None and bytes_per_token:
            shared_workspace.narrow(
                0,
                offset,
                capacity * bytes_per_token,
            ).fill_(0xFF)
            poisoned = True
    assert poisoned, "test requires a standalone top-k reduction workspace"


def _dequantize_wgrad_operand(
    data: torch.Tensor,
    scales: torch.Tensor,
    expert_offsets: torch.Tensor,
    valid_route_counts: torch.Tensor,
    *,
    k_dim: int,
) -> torch.Tensor:
    """Decode only the valid expert rows of one fixed-capacity operand."""

    if data.ndim != 2 or k_dim not in (0, 1):
        raise ValueError("wgrad operand must be rank 2 with k_dim 0 or 1")
    non_k = int(data.shape[1 - k_dim])
    padded_non_k = _round_up(non_k, 128)
    flat_scales = scales.view(torch.uint8).reshape(-1)
    output = torch.zeros(data.shape, dtype=torch.float32, device=data.device)
    ends = [int(value) for value in expert_offsets.detach().cpu().tolist()]
    valid_counts = [int(value) for value in valid_route_counts.detach().cpu().tolist()]
    if len(ends) != len(valid_counts):
        raise ValueError("expert offsets and valid route counts must have equal size")
    k_capacity = int(data.shape[k_dim])
    previous = 0
    scale_byte_offset = 0
    for expert, (end, valid_count) in enumerate(zip(ends, valid_counts)):
        if end < previous or end > k_capacity:
            raise ValueError("expert offsets must be nondecreasing and fit the operand " f"K capacity ({k_capacity})")
        extent = end - previous
        if valid_count < 0 or valid_count > extent:
            raise ValueError(f"expert {expert} valid route count {valid_count} exceeds " f"its padded extent {extent}")
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
        if valid_count:
            valid_end = previous + valid_count
            valid_scale_columns = (valid_count + 31) // 32
            logical_scale = logical_scale[:, :valid_scale_columns]
            if k_dim == 1:
                expanded_scale = logical_scale.repeat_interleave(32, dim=1)[:, :valid_count]
                output[:, previous:valid_end] = data[:, previous:valid_end].float() * expanded_scale
            else:
                expanded_scale = logical_scale.repeat_interleave(
                    32,
                    dim=1,
                )[
                    :, :valid_count
                ].transpose(0, 1)
                output[previous:valid_end, :] = data[previous:valid_end, :].float() * expanded_scale
        previous = end
        scale_byte_offset += scale_byte_count

    return output


def _dense_wgrads_from_operands(operands):
    """Reference grouped matmuls over the producer-native operand ABI."""

    fc1_a = _dequantize_wgrad_operand(
        operands.fc1_a,
        operands.fc1_sfa,
        operands.expert_offsets,
        operands.valid_route_counts,
        k_dim=0,
    )
    fc1_b = _dequantize_wgrad_operand(
        operands.fc1_b,
        operands.fc1_sfb,
        operands.expert_offsets,
        operands.valid_route_counts,
        k_dim=0,
    )
    fc2_a = _dequantize_wgrad_operand(
        operands.fc2_a,
        operands.fc2_sfa,
        operands.expert_offsets,
        operands.valid_route_counts,
        k_dim=0,
    )
    fc2_b = _dequantize_wgrad_operand(
        operands.fc2_b,
        operands.fc2_sfb,
        operands.expert_offsets,
        operands.valid_route_counts,
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
        fc1_parts.append(fc1_a[previous:valid_end, :].transpose(0, 1) @ fc1_b[previous:valid_end, :])
        fc2_parts.append(fc2_a[previous:valid_end, :].transpose(0, 1) @ fc2_b[previous:valid_end, :])
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
    """Run one explicitly padded operand bundle through production WGrad."""

    import cudnn

    if prefix not in ("fc1", "fc2"):
        raise ValueError(f"prefix must be 'fc1' or 'fc2', got {prefix!r}")
    # The wrapper allocates the TMA-descriptor scratch per call on the launch
    # stream (the plan owns none); pass descriptor_workspace= to replay a
    # captured call site over a caller-owned buffer.
    return cudnn.grouped_gemm_wgrad_wrapper_sm100(
        a_tensor=getattr(operands, f"{prefix}_a").transpose(0, 1),
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
    """Run both explicitly padded operand bundles through production WGrad."""

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
        max_relative_error = (absolute_error / expected_fp32.abs().clamp_min(1.0e-6)).max().item()
        torch.testing.assert_close(
            actual_fp32,
            expected_fp32,
            msg=lambda default, name=name: (
                f"{name} does not match {reference_name}; " f"max_abs_error={max_absolute_error:.6g}, " f"max_rel_error={max_relative_error:.6g}\n{default}"
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
        "physical_recv_pool_rows",
        "sf_padding_size",
        "tuning",
    ):
        options.pop(production_only, None)
    options["intermediate_format"] = "mxfp8"
    options["backward_operand_format"] = "mxfp8"
    return MoeEpReference(**options)


def _fixed_training_weights(args):
    """Build independent source packs for allocation-free native packing."""

    from cudnn.moe_ep import MoeEpBackwardWeights, MoeEpForwardWeights
    from cudnn.moe_ep._megamoe_backend.mxfp8._adapter import (
        _quantize_plain_mxfp8,
    )

    fc1_weight = args[1]
    fc2_weight = args[2]
    dense_fc1 = fc1_weight if isinstance(fc1_weight, torch.Tensor) else fc1_weight.dequantize()
    dense_fc2 = fc2_weight if isinstance(fc2_weight, torch.Tensor) else fc2_weight.dequantize()
    forward = MoeEpForwardWeights(
        fc1=(_quantize_plain_mxfp8(dense_fc1, axis=1) if isinstance(fc1_weight, torch.Tensor) else fc1_weight),
        fc2=(_quantize_plain_mxfp8(dense_fc2, axis=1) if isinstance(fc2_weight, torch.Tensor) else fc2_weight),
    )
    backward = MoeEpBackwardWeights(
        w2_transpose=_quantize_plain_mxfp8(
            dense_fc2.transpose(1, 2).contiguous(),
            axis=1,
        ),
        w1_transpose=_quantize_plain_mxfp8(
            dense_fc1.transpose(1, 2).contiguous(),
            axis=1,
        ),
    )
    return forward, backward


def _allocate_training_weight_staging(weights):
    """Allocate caller-owned native pack destinations for one source pair."""

    from cudnn.moe_ep import (
        MoeEpBackwardWeightStaging,
        MoeEpForwardWeightStaging,
    )

    forward, backward = weights
    fc1 = forward.fc1
    fc2 = forward.fc2
    experts, hidden, gate_up = fc1.data.shape
    intermediate = fc2.data.shape[1]

    def scale(elements):
        return torch.empty(
            (experts, elements),
            dtype=torch.float8_e8m0fnu,
            device=fc1.device,
        )

    def blocked_elements(rows, columns):
        return ((rows + 127) // 128 * 128) * ((columns + 3) // 4 * 4)

    forward_out = MoeEpForwardWeightStaging(
        fc1_payload=torch.empty_strided(
            fc1.data.shape,
            (hidden * gate_up, 1, hidden),
            dtype=fc1.data.dtype,
            device=fc1.device,
        ),
        fc1_scale=scale(blocked_elements(gate_up, hidden // 32)),
        fc2_payload=torch.empty_strided(
            fc2.data.shape,
            (intermediate * hidden, 1, intermediate),
            dtype=fc2.data.dtype,
            device=fc2.device,
        ),
        fc2_scale=scale(blocked_elements(hidden, intermediate // 32)),
    )
    w2t = backward.w2_transpose
    w1t = backward.w1_transpose
    backward_out = MoeEpBackwardWeightStaging(
        # Quantization and empty_like may collapse a singleton expert
        # dimension's leading stride. Construct the native ABI's canonical
        # expert-major layouts from logical dimensions instead.
        w2_transpose_payload=torch.empty_strided(
            w2t.data.shape,
            (hidden * intermediate, intermediate, 1),
            dtype=w2t.data.dtype,
            device=w2t.data.device,
        ),
        w2_transpose_scale=scale(blocked_elements(intermediate, hidden // 32)),
        w1_transpose_payload=torch.empty_strided(
            w1t.data.shape,
            (gate_up * hidden, hidden, 1),
            dtype=w1t.data.dtype,
            device=w1t.data.device,
        ),
        w1_transpose_scale=scale(blocked_elements(hidden, gate_up // 32)),
    )
    return forward_out, backward_out


def _make_discrete_training_weights(forward, backward):
    """Clone packed experts into independent allocations and build pointer tables."""

    from cudnn.moe_ep import (
        MoeEpNativeDiscreteBackwardWeights,
        MoeEpNativeDiscreteForwardWeights,
        MoeEpNativeDiscreteWeight,
        MoeEpNativeWeightLayout,
    )

    def separate(packed):
        alignment_elements = 256 // packed.element_size()
        views = []
        backings = []
        for expert_index, source in enumerate(packed):
            footprint = 1 + sum((size - 1) * stride for size, stride in zip(source.shape, source.stride()))
            storage_offset = expert_index * (expert_index + 1) // 2 * alignment_elements
            backing = torch.empty(
                storage_offset + footprint + alignment_elements,
                dtype=source.dtype,
                device=source.device,
            )
            view = torch.as_strided(
                backing,
                source.shape,
                source.stride(),
                storage_offset,
            )
            view.copy_(source)
            if view.data_ptr() % 256:
                raise RuntimeError("discrete expert weight base must be 256-byte aligned")
            views.append(view)
            backings.append(backing)
        return tuple(views), tuple(backings)

    def weight(native, *, payload=None, layout_id=None):
        if payload is None:
            payload = native.payload
        if layout_id is None:
            layout_id = native.layout_id
        payloads, payload_owners = separate(payload)
        scales, scale_owners = separate(native.scale)
        return (
            MoeEpNativeDiscreteWeight(
                torch.tensor(
                    [value.data_ptr() for value in payloads],
                    dtype=torch.int64,
                    device=native.device,
                ),
                torch.tensor(
                    [value.data_ptr() for value in scales],
                    dtype=torch.int64,
                    device=native.device,
                ),
                layout_id,
            ),
            (payloads, payload_owners, scales, scale_owners),
        )

    fc1, fc1_owners = weight(forward.fc1)
    fc2, fc2_owners = weight(forward.fc2)
    w2, w2_owners = weight(
        backward.w2_transpose,
        payload=backward.w2_transpose.payload.transpose(1, 2).contiguous(),
        layout_id=MoeEpNativeWeightLayout.BACKWARD_W2_DGRAD_NK_ROW_MAJOR_V1,
    )
    w1, w1_owners = weight(
        backward.w1_transpose,
        payload=backward.w1_transpose.payload.transpose(1, 2).contiguous(),
        layout_id=(MoeEpNativeWeightLayout.BACKWARD_W1_DGRAD_GATE_UP_INTERLEAVED_32_NK_ROW_MAJOR_V1),
    )
    return (
        MoeEpNativeDiscreteForwardWeights(fc1=fc1, fc2=fc2),
        MoeEpNativeDiscreteBackwardWeights(
            w2_transpose=w2,
            w1_transpose=w1,
        ),
        (fc1_owners, fc2_owners, w2_owners, w1_owners),
    )


def _allocate_stateless_training_outputs(requirements, device, symmetric_buffers):
    """Bind symmetric final outputs and allocate the remaining contracts."""

    from cudnn.moe_ep import (
        MoeEpTrainingBackwardOutputs,
        MoeEpTrainingForwardOutputs,
    )

    def allocate(name):
        shape, stride, dtype, _alignment = requirements[name]
        return torch.empty_strided(
            shape,
            stride,
            dtype=dtype,
            device=device,
        )

    forward = MoeEpTrainingForwardOutputs(
        output=symmetric_buffers["output"],
        fc1_preact=allocate("fc1_preact"),
        fc1_a=allocate("fc1_a"),
        fc1_sfa=allocate("fc1_sfa"),
        valid_route_counts=allocate("valid_route_counts"),
        expert_offsets=allocate("expert_offsets"),
    )
    backward = MoeEpTrainingBackwardOutputs(
        grad_activation=symmetric_buffers["grad_activation"],
        dprob=symmetric_buffers["dprob"],
        fc1_b=allocate("fc1_b"),
        fc1_sfb=allocate("fc1_sfb"),
        fc2_a=allocate("fc2_a"),
        fc2_sfa=allocate("fc2_sfa"),
        fc2_b=allocate("fc2_b"),
        fc2_sfb=allocate("fc2_sfb"),
    )
    return forward, backward


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
        # segments. The stateless producer ABI uses 128-row segments;
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
    hidden_size: int = 128,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return (
        torch.randn(
            token_count,
            hidden_size,
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        / 8
    )


def _assert_backward_matches(actual, expected, topk_idx) -> None:
    assert len(actual) == len(expected) == 2
    for name, gradient, reference, expected_dtype, close_kwargs in zip(
        ("grad_activation", "grad_topk_weights"),
        actual,
        expected,
        (torch.bfloat16, torch.float32),
        _BACKWARD_CLOSE_KWARGS,
    ):
        assert gradient.shape == reference.shape
        assert gradient.dtype == expected_dtype
        assert reference.dtype == torch.float32
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(
            gradient.float(),
            reference,
            msg=lambda default, name=name: (f"{name} does not match the backward reference\n{default}"),
            **close_kwargs,
        )


def _interleave_fc1_wgrad(
    tensor: torch.Tensor,
    interleave_size: int = 32,
) -> torch.Tensor:
    """Convert logical gate-then-up columns to producer-native strip order."""

    out_features = tensor.shape[-1]
    return (
        tensor.view(
            *tensor.shape[:-1],
            2,
            out_features // (2 * interleave_size),
            interleave_size,
        )
        .transpose(-3, -2)
        .reshape(tensor.shape)
    )


def _assert_wgrads_match_reference(
    actual,
    expected,
    *,
    expected_dense=None,
    weight_interleave_size=None,
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
    if weight_interleave_size is not None:
        expected_fc1, expected_fc2 = expected_dense
        expected_dense = (
            _interleave_fc1_wgrad(expected_fc1, weight_interleave_size),
            expected_fc2,
        )
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
