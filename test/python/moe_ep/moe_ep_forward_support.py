# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared support for MoE EP forward tests."""

from __future__ import annotations

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
from moe_ep.moe_ep_test_data import quantize_mxfp8

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

__all__ = [
    "_assert_matches_reference",
    "_forward_config",
    "_make_forward_case",
    "_naive_reference",
    "_output_as_float",
    "_reference_forward",
    "_replay_cuda_graph",
    "_require_distributed_sm107",
    "_sm107_device",
    "_stress_backend_reuse",
]


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
    if any(
        torch.cuda.get_device_capability(index) != (10, 7)
        for index in range(world_size)
    ):
        pytest.skip(
            "multi-GPU Rubin MXFP8 forward requires exactly SM107 "
            "(compute capability 10.7) on every rank"
        )
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

    seed = (
        20260811
        + experts * 1009
        + tokens * 101
        + hidden * 11
        + intermediate
        + top_k
    )
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
    topk_idx = (
        torch.arange(tokens * top_k, device=device)
        .reshape(tokens, top_k)
        .remainder(experts)
        .to(index_dtype)
    )
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
    weight_refresh_count = (
        backend._adapter.weight_refresh_count if check_weight_refresh else None
    )
    alternate_stream = torch.cuda.Stream(device=device)

    for iteration in range(100):
        args[3].copy_(original_topk_idx)
        args[4].copy_(
            original_topk_weights * float((iteration % 7) + 1) / 7.0
        )
        if iteration % 10 == 0:
            args[3].fill_(-1)
        stream = (
            torch.cuda.current_stream(device)
            if iteration % 2 == 0
            else alternate_stream
        )
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
