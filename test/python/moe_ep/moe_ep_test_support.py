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
    "_forward_config",
    "_grad_output",
    "_make_forward_case",
    "_naive_reference",
    "_output_as_float",
    "_reference_backward",
    "_reference_forward",
    "_replay_cuda_graph",
    "_require_distributed_sm107",
    "_run_grouped_wgrad_kernel",
    "_sm107_device",
    "_stress_backend_reuse",
    "_training_abi_prepared",
    "_training_config",
    "_training_prepared_pair",
    "make_distributed_forward_inputs",
    "make_forward_inputs",
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
    from cudnn.moe_ep._contracts import ForwardConfig, normalize_fc1_weight_layout
    from cudnn.moe_ep._tuning import MoeEpTuningConfig

    weight_interleave_size = overrides.pop("weight_interleave_size", None)
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
        "fc1_weight_layout": normalize_fc1_weight_layout(weight_interleave_size),
    }
    values.update(overrides)
    return ForwardConfig(**values)


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
    """Reference grouped matmuls over the producer-native operand ABI."""

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
    # Graph callers provide one persistent output per training lane. This is
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
        "max_recv_size_per_rank",
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
        w2_transpose_payload=torch.empty_like(w2t.data),
        w2_transpose_scale=scale(blocked_elements(intermediate, hidden // 32)),
        w1_transpose_payload=torch.empty_like(w1t.data),
        w1_transpose_scale=scale(blocked_elements(hidden, gate_up // 32)),
    )
    return forward_out, backward_out


def _allocate_stateless_training_outputs(requirements, device):
    """Allocate every advertised caller-owned output contract."""

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
        output=allocate("output"),
        fc1_preact=allocate("fc1_preact"),
        fc1_a=allocate("fc1_a"),
        fc1_sfa=allocate("fc1_sfa"),
        valid_route_counts=allocate("valid_route_counts"),
        expert_offsets=allocate("expert_offsets"),
    )
    backward = MoeEpTrainingBackwardOutputs(
        grad_activation=allocate("grad_activation"),
        dprob=allocate("dprob"),
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
