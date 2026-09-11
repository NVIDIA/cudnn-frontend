# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core MoE EP forward contract, parity, runtime, and distributed tests."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from moe_ep.moe_ep_distributed_workers import (
    _distributed_output_worker,
    _distributed_subgroup_output_worker,
)
from moe_ep.moe_ep_test_support import (
    _assert_matches_reference,
    _forward_config,
    _make_forward_case,
    _naive_reference,
    _poison_pre_reduced_for_test,
    _reference_forward,
    _replay_cuda_graph,
    _require_distributed_sm107,
    _sm107_device,
    _stress_backend_reuse,
    make_forward_inputs,
    quantize_mxfp8,
)
from moe_ep.moe_ep_reference import (
    MoeEpReference,
    MoeFormat,
    forward_combine_round_trip,
    quantize_blockwise,
)

# Public API, capability, layout, and workspace contracts.


@pytest.mark.L0
@pytest.mark.parametrize(
    ("physical_capacity", "logical_limit"),
    ((128, 1), (256, 129), (384, 257)),
)
def test_reverse_capacity_preserves_the_prescribed_physical_pool(
    physical_capacity,
    logical_limit,
):
    from cudnn import MoeEp
    from cudnn.moe_ep._megamoe_backend.mxfp8._config import (
        Mxfp8KernelConfig,
    )

    with MoeEp(
        **_forward_config(
            num_experts=2,
            top_k=2,
            max_tokens_per_rank=256,
        ),
        max_recv_size_per_rank=physical_capacity,
    ) as op:
        config = Mxfp8KernelConfig.from_operator_config(op._forward_config)

    assert config.physical_recv_pool_size == physical_capacity
    assert config.max_recv_size_per_rank == logical_limit


@pytest.mark.L0
def test_reverse_capacity_preserves_overprovisioned_physical_pool():
    from cudnn import MoeEp
    from cudnn.moe_ep._megamoe_backend.mxfp8._config import (
        Mxfp8KernelConfig,
    )

    with MoeEp(
        **_forward_config(),
        max_recv_size_per_rank=384,
    ) as op:
        config = Mxfp8KernelConfig.from_operator_config(op._forward_config)

    assert config.physical_recv_pool_size == 384
    assert config.max_recv_size_per_rank == 257


@pytest.mark.L0
@pytest.mark.parametrize("physical_capacity", (127, 129))
def test_reverse_capacity_rejects_unrepresentable_physical_pool(physical_capacity):
    from cudnn import MoeEp

    with pytest.raises(ValueError, match=r"P % 128 == 0"):
        MoeEp(
            **_forward_config(),
            max_recv_size_per_rank=physical_capacity,
        )


@pytest.mark.L0
def test_upstream_receive_capacity_applies_per_expert_padding():
    from cudnn.moe_ep._megamoe_backend.cutedsl_src.communication.nvlink_domain.token_comm_deterministic import (
        _compute_receive_capacity,
    )

    capacity = _compute_receive_capacity(
        world_size=4,
        max_tokens_per_rank=64,
        topk=2,
        experts_per_rank=2,
        max_recv_size_per_rank=129,
        padding_block=128,
    )

    assert capacity.raw_route_count == 512
    assert capacity.logical_route_count == 129
    assert capacity.padded_route_count == 256


@pytest.mark.L0
@pytest.mark.parametrize("validation_mode", ["strict", "trusted"])
def test_moe_ep_accepts_validation_modes(validation_mode):
    from cudnn import MoeEp

    with MoeEp(
        **_forward_config(),
        validation_mode=validation_mode,
    ) as op:
        assert op.validation_mode == validation_mode


@pytest.mark.L0
def test_strict_expert_id_validation_requires_dense_routes():
    from cudnn import MoeEp
    from cudnn.moe_ep._validation import _validate_expert_ids

    with MoeEp(**_forward_config()) as op:
        config = op._forward_config
        _validate_expert_ids(config, torch.tensor([[0, 1]], dtype=torch.int32))
        with pytest.raises(ValueError, match="dropped-route sentinel"):
            _validate_expert_ids(config, torch.tensor([[0, -1]], dtype=torch.int32))
        with pytest.raises(ValueError, match="valid global expert id"):
            _validate_expert_ids(
                config,
                torch.tensor([[0, config.num_experts]], dtype=torch.int32),
            )


@pytest.mark.L0
@pytest.mark.parametrize("validation_mode", [None, True, "fast"])
def test_moe_ep_rejects_invalid_validation_mode(validation_mode):
    from cudnn import MoeEp

    with pytest.raises(ValueError, match="validation_mode"):
        MoeEp(
            **_forward_config(),
            validation_mode=validation_mode,
        )


@pytest.mark.L0
def test_moe_ep_rejects_conflicting_forward_tuning_aliases():
    from cudnn import MoeEp, MoeEpTuningConfig

    with pytest.raises(ValueError, match="aliases"):
        MoeEp(
            **_forward_config(),
            tuning=MoeEpTuningConfig(),
            forward_tuning=MoeEpTuningConfig(),
        )


@pytest.mark.L0
def test_moe_ep_backward_tuning_default_is_independent():
    from cudnn import MoeEp, MoeEpTuningConfig

    forward_tuning = MoeEpTuningConfig(
        token_back_mode="standalone_warps",
        epi_flag_batch=(4, 2),
        token_in_flag_batch=8,
        group_hint=768,
    )
    with MoeEp(**_forward_config(), forward_tuning=forward_tuning) as op:
        assert op.forward_tuning is forward_tuning
        assert op.backward_tuning == MoeEpTuningConfig()


# Single-rank and distributed forward numerical parity.


@pytest.mark.L0
def test_bf16_forward_matches_reference_and_returns_fresh_outputs():
    from cudnn import MoeEp

    device = _sm107_device()
    args = make_forward_inputs(device)
    activation, fc1_weight, fc2_weight = args[:3]
    expected = _reference_forward(args)

    assert activation.logical_shape == (5, 128)
    assert fc1_weight.logical_shape == (2, 128, 512)
    assert fc2_weight.logical_shape == (2, 256, 128)

    with MoeEp(**_forward_config()) as op:
        first = op(*args)
        snapshot = first.clone()
        second = op(*args)
        torch.cuda.synchronize(device)

    assert isinstance(first, torch.Tensor)
    assert isinstance(second, torch.Tensor)
    assert first.shape == second.shape == (5, 128)
    assert first.dtype == second.dtype == torch.bfloat16
    assert first.device == second.device == device
    assert first is not second
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, snapshot, rtol=0, atol=0)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    _assert_matches_reference(first, expected)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
def test_mxfp8_combine_matches_direct_fp32_training_reference():
    from cudnn import MoeEp

    device = _sm107_device()
    args = make_forward_inputs(device)
    config = _forward_config(
        combine_format="mxfp8",
    )
    expected = _reference_forward(args, **config)

    with MoeEp(**config) as op:
        actual = op(*args)
        torch.cuda.synchronize(device)

    _assert_matches_reference(actual, expected)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize(
    ("plain_mask", "plain_dtype"),
    [
        pytest.param(
            (True, False, False),
            torch.bfloat16,
            id="activation-bf16",
        ),
        pytest.param(
            (False, True, False),
            torch.float16,
            id="fc1-fp16",
        ),
        pytest.param(
            (False, False, True),
            torch.float32,
            id="fc2-fp32",
        ),
    ],
)
def test_plain_and_mixed_inputs_match_staged_reference(
    plain_mask,
    plain_dtype,
):
    from cudnn import MoeEp

    device = _sm107_device()
    args = list(make_forward_inputs(device))
    for index, make_plain in enumerate(plain_mask):
        if make_plain:
            args[index] = args[index].dequantize(dtype=plain_dtype)
    args = tuple(args)
    expected = _reference_forward(args)

    with MoeEp(**_forward_config()) as op:
        actual = op(*args)
        torch.cuda.synchronize(device)

    _assert_matches_reference(actual, expected)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
def test_one_operator_switches_mxfp8_and_plain_weight_families():
    from cudnn import MoeEp

    device = _sm107_device()
    quantized_args = make_forward_inputs(device)
    plain_args = (
        quantized_args[0].dequantize(dtype=torch.bfloat16),
        quantized_args[1].dequantize(dtype=torch.bfloat16),
        quantized_args[2].dequantize(dtype=torch.bfloat16),
        *quantized_args[3:],
    )
    expected_quantized = _reference_forward(quantized_args)
    expected_plain = _reference_forward(plain_args)

    with MoeEp(**_forward_config()) as op:
        quantized = op(*quantized_args)
        backend = op._forward_backend
        refresh_before = backend._adapter.weight_refresh_count
        plain = op(*plain_args)
        refresh_after = backend._adapter.weight_refresh_count
        torch.cuda.synchronize(device)

    assert op._forward_backend is None
    assert refresh_after == refresh_before + 1
    _assert_matches_reference(quantized, expected_quantized)
    _assert_matches_reference(plain, expected_plain)


@pytest.mark.L0
def test_nondefault_moe_ep_tuning_matches_reference_and_reuses_plan():
    from cudnn import MoeEp, MoeEpTuningConfig

    device = _sm107_device()
    args = make_forward_inputs(device)
    expected = _reference_forward(args)
    tuning = MoeEpTuningConfig(
        token_back_mode="standalone_warps",
        epi_flag_batch=(4, 2),
        token_in_flag_batch=4,
        group_hint=64,
    )

    with MoeEp(**_forward_config(), tuning=tuning) as op:
        first = op(*args)
        backend = op._forward_backend
        assert backend is not None
        compiled = backend._compiled
        workspace = backend._plan._workspace
        second = op(*args)
        torch.cuda.synchronize(device)

        assert backend._compiled is compiled
        assert backend._plan._workspace is workspace
        assert backend.kernel_config.tuning_signature(backend._prepared_kernel.launch_cluster_count) == ("standalone_warps", (4, 2), 4, 64, False)

    _assert_matches_reference(first, expected)
    _assert_matches_reference(second, expected)


@pytest.mark.L0
def test_gate_up_clamp_matches_moe_ep_reference():
    from cudnn import MoeEp

    device = _sm107_device()
    args = make_forward_inputs(device)
    clamp = 0.5
    expected = _reference_forward(args, gate_up_clamp=clamp)
    unclamped = _reference_forward(args)

    with MoeEp(**_forward_config(gate_up_clamp=clamp)) as op:
        actual = op(*args)
        torch.cuda.synchronize(device)

    assert not torch.equal(expected, unclamped)
    _assert_matches_reference(actual, expected)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize(
    (
        "experts",
        "tokens",
        "hidden",
        "intermediate",
        "top_k",
        "index_dtype",
        "weight_dtype",
    ),
    [
        pytest.param(
            2,
            3,
            128,
            256,
            1,
            torch.int32,
            torch.bfloat16,
            id="topk1-h128-i256-int32-bf16",
        ),
        pytest.param(
            2,
            5,
            128,
            256,
            2,
            torch.int64,
            torch.float32,
            id="topk2-h128-i256-int64-fp32",
        ),
        pytest.param(
            4,
            3,
            256,
            256,
            4,
            torch.int32,
            torch.float16,
            id="topk4-h256-i256-int32-fp16",
        ),
        pytest.param(
            32,
            1,
            128,
            256,
            8,
            torch.int64,
            torch.float32,
            id="topk8-boundary-int64-fp32",
        ),
    ],
)
def test_supported_topk_shape_and_routing_format_matrix(
    experts,
    tokens,
    hidden,
    intermediate,
    top_k,
    index_dtype,
    weight_dtype,
):
    from cudnn import MoeEp

    device = _sm107_device()
    args = _make_forward_case(
        device,
        experts=experts,
        tokens=tokens,
        hidden=hidden,
        intermediate=intermediate,
        top_k=top_k,
        index_dtype=index_dtype,
        weight_dtype=weight_dtype,
    )
    config = _forward_config(
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        top_k=top_k,
        max_tokens_per_rank=tokens,
    )
    expected = _reference_forward(args, **config)

    with MoeEp(**config) as op:
        actual = op(*args)
        torch.cuda.synchronize(device)

    assert actual.shape == (tokens, hidden)
    assert actual.dtype == torch.bfloat16
    _assert_matches_reference(actual, expected)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("combine_format", ["bf16", "mxfp8"])
@pytest.mark.parametrize("capacity", [5, 129])
def test_single_gpu_stress_and_cuda_graph_replay(combine_format, capacity):
    from cudnn import MoeEp

    device = _sm107_device()
    args = make_forward_inputs(device)
    original_topk_idx = args[3].clone()
    original_topk_weights = args[4].clone()
    config = _forward_config(
        combine_format=combine_format,
        max_tokens_per_rank=capacity,
    )
    expected = _reference_forward(args, **config)
    alternate_topk_idx = original_topk_idx.flip(1).contiguous()
    alternate_args = (*args[:3], alternate_topk_idx, args[4])
    alternate_expected = _reference_forward(alternate_args, **config)

    with MoeEp(**config) as op:
        op.warmup(*args)
        _stress_backend_reuse(
            op,
            args,
            original_topk_idx,
            original_topk_weights,
            device,
            check_weight_refresh=True,
        )

        args[3].copy_(original_topk_idx)
        args[4].copy_(original_topk_weights)
        backend = op._forward_backend
        assert backend is not None
        assert backend._prepared_kernel is not None
        assert backend._plan is not None
        assert backend._plan._workspace is not None

        def poison_pre_reduced():
            workspace = backend._plan._workspace.views(args[0].logical_shape[0])
            _poison_pre_reduced_for_test(
                backend._prepared_kernel,
                workspace.symmetric["kernel_shared_workspace"],
            )

        poison_pre_reduced()
        eager = op(*args)
        torch.cuda.synchronize(device)
        _assert_matches_reference(eager, expected)
        _replay_cuda_graph(
            op,
            args,
            original_topk_idx,
            expected,
            device,
            alternate_topk_idx=alternate_topk_idx,
            alternate_expected=alternate_expected,
            poison_before_replay=poison_pre_reduced,
        )


@pytest.mark.L0
def test_forward_mxfp8_combine_is_direct_fp32():
    generator = torch.Generator().manual_seed(20260819)
    accumulator = torch.randn(4, 128, generator=generator) * 3.25

    forward = forward_combine_round_trip(accumulator, MoeFormat.MXFP8)
    direct_fp32 = quantize_blockwise(
        accumulator,
        MoeFormat.MXFP8,
    ).dequantize()
    bf16_preround = quantize_blockwise(
        accumulator.to(torch.bfloat16).float(),
        MoeFormat.MXFP8,
    ).dequantize()

    torch.testing.assert_close(
        forward,
        direct_fp32,
        rtol=0,
        atol=0,
    )
    assert not torch.equal(forward, bf16_preround)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize(
    ("world_size", "combine_format"),
    [
        pytest.param(2, "bf16", id="ep2-bf16"),
        pytest.param(2, "mxfp8", id="ep2-mxfp8"),
        pytest.param(3, "mxfp8", id="ep3-mxfp8"),
        pytest.param(4, "bf16", id="ep4-bf16"),
    ],
)
def test_mxfp8_forward_multi_gpu_matches_reference(
    world_size,
    combine_format,
    tmp_path,
):
    _require_distributed_sm107(world_size)
    os.environ.setdefault("NVIDIA_IMEX_CHANNELS", "0")
    init_file = tmp_path / f"{combine_format}_combine_ep{world_size}.init"
    mp.spawn(
        _distributed_output_worker,
        args=(world_size, str(init_file), combine_format),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.L1
@pytest.mark.gpu_exclusive
def test_mxfp8_forward_noncontiguous_ep_subgroups(tmp_path):
    global_world_size = 4
    _require_distributed_sm107(global_world_size)
    os.environ.setdefault("NVIDIA_IMEX_CHANNELS", "0")
    init_file = tmp_path / "two_noncontiguous_ep2.init"
    mp.spawn(
        _distributed_subgroup_output_worker,
        args=(global_world_size, str(init_file)),
        nprocs=global_world_size,
        join=True,
    )


# Input staging and workspace layout.


@pytest.mark.L0
def test_plain_tensor_staging_matches_logical_mxfp8_quantization():
    if not torch.cuda.is_available():
        pytest.skip("MXFP8 staging test requires CUDA")
    from cudnn.moe_ep._megamoe_backend.mxfp8._adapter import (
        _quantize_plain_mxfp8,
    )

    device = torch.device("cuda", 0)
    plain = torch.randn(2, 128, 3, device=device).to(torch.bfloat16)
    actual = _quantize_plain_mxfp8(plain, axis=1)
    expected = quantize_mxfp8(plain, axis=1)

    torch.testing.assert_close(
        actual.data.view(torch.uint8),
        expected.data.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual.scale.view(torch.uint8),
        expected.scale.view(torch.uint8),
        rtol=0,
        atol=0,
    )


@pytest.mark.L0
def test_inference_activation_scale_uses_unpadded_prefix(monkeypatch):
    import cudnn.moe_ep._megamoe_backend.mxfp8._adapter as adapter_module
    from cudnn import MoeEp
    from cudnn.moe_ep._megamoe_backend._workspace import (
        WorkspaceRequirements,
        padded_mxfp8_scale_columns,
    )
    from cudnn.moe_ep._megamoe_backend.mxfp8._adapter import (
        Mxfp8InputAdapter,
        Mxfp8Weights,
    )

    assert padded_mxfp8_scale_columns(128) == 16
    assert padded_mxfp8_scale_columns(512) == 16
    assert padded_mxfp8_scale_columns(640) == 32

    with MoeEp(**_forward_config()) as op:
        requirements = WorkspaceRequirements.for_mxfp8(
            op._forward_config,
            kernel_local_workspace_bytes=128,
            kernel_shared_workspace_bytes=128,
        )
    activation_scale = next(region for region in requirements.symmetric_regions if region.name == "activation_scale")
    assert activation_scale.nbytes == 128 * 16

    capacity = 5
    hidden = 128
    top_k = 2
    symmetric = {
        "activation_data": torch.empty(capacity * hidden, dtype=torch.uint8),
        "activation_scale": torch.empty(activation_scale.nbytes, dtype=torch.uint8),
        "topk_weights": torch.empty(capacity * top_k * 4, dtype=torch.uint8),
        "output_data": torch.empty(capacity * hidden * 2, dtype=torch.uint8),
        "kernel_shared_workspace": torch.empty(128, dtype=torch.uint8),
    }
    local = {
        "topk_idx": torch.empty(capacity * top_k * 4, dtype=torch.uint8),
        "overflow_flag": torch.empty(4, dtype=torch.uint8),
        "kernel_local_workspace": torch.empty(128, dtype=torch.uint8),
    }
    request = SimpleNamespace(
        token_count=1,
        activation=object(),
        topk_idx=torch.zeros((1, top_k), dtype=torch.int32),
        topk_weights=torch.ones((1, top_k), dtype=torch.float32),
    )
    staged_activation = SimpleNamespace(
        data=torch.zeros((1, hidden), dtype=torch.float8_e4m3fn),
        scale=torch.zeros((1, hidden // 32), dtype=torch.float8_e8m0fnu),
    )
    config = SimpleNamespace(
        max_tokens_per_rank=capacity,
        hidden=hidden,
        top_k=top_k,
        generate_c=False,
        enable_col_quant=False,
        fc2_in_kernel_topk_reduce=True,
        combine_format="bf16",
    )
    resources = SimpleNamespace(
        workspace=SimpleNamespace(symmetric=symmetric, local=local),
    )
    weights = Mxfp8Weights(*(torch.empty(0) for _ in range(4)))
    adapter = Mxfp8InputAdapter()
    monkeypatch.setattr(adapter_module, "_as_mxfp8", lambda _: staged_activation)
    monkeypatch.setattr(adapter, "_prepare_weights", lambda *_: weights)

    launch = adapter.stage(
        request,
        resources,
        config,
        local_workspace_zero_bytes=0,
        shared_workspace_zero_bytes=0,
        pre_reduced_activation_offset=None,
        pre_reduced_activation_bytes_per_token=0,
        pre_reduced_activation_sf_offset=None,
        pre_reduced_activation_sf_bytes_per_token=0,
        col_quant_data_rows=0,
        col_quant_sf_elements=0,
    )

    assert launch.activation_sf.shape == (capacity, 16)
    assert launch.activation_sf.data_ptr() == symmetric["activation_scale"].data_ptr()


@pytest.mark.L0
def test_column_requant_workspace_is_allocated_only_when_enabled():
    from cudnn import MoeEp
    from cudnn.moe_ep._megamoe_backend._workspace import WorkspaceRequirements

    with MoeEp(**_forward_config()) as op:
        disabled = WorkspaceRequirements.for_mxfp8(
            op._forward_config,
            kernel_local_workspace_bytes=128,
            kernel_shared_workspace_bytes=128,
        )
        enabled = WorkspaceRequirements.for_mxfp8(
            op._forward_config,
            kernel_local_workspace_bytes=128,
            kernel_shared_workspace_bytes=128,
            col_quant_data_bytes=640,
            col_quant_sf_bytes=80,
        )

    disabled_names = {region.name for region in disabled.local_regions}
    assert "col_quant_data" not in disabled_names
    assert "col_quant_sf" not in disabled_names
    enabled_sizes = {region.name: region.nbytes for region in enabled.local_regions}
    assert enabled_sizes["col_quant_data"] == 640
    assert enabled_sizes["col_quant_sf"] == 80

    with pytest.raises(ValueError, match="must be enabled together"):
        WorkspaceRequirements.for_mxfp8(
            op._forward_config,
            kernel_local_workspace_bytes=128,
            kernel_shared_workspace_bytes=128,
            col_quant_data_bytes=640,
        )


# Reference and quantization self-checks.


@pytest.mark.L0
@pytest.mark.parametrize(
    "intermediate_format",
    [None, MoeFormat.MXFP8],
    ids=["fp32-intermediate", "mxfp8-intermediate"],
)
def test_reference_mxfp8_inputs_bf16_combine_matches_naive(
    intermediate_format,
):
    torch.manual_seed(19)
    experts, tokens, hidden, intermediate = 2, 3, 128, 128
    activation = torch.randn(tokens, hidden)
    fc1_weight = torch.randn(experts, hidden, 2 * intermediate) / 8
    fc2_weight = torch.randn(experts, intermediate, hidden) / 8
    q_activation = quantize_blockwise(activation, MoeFormat.MXFP8, axis=1)
    q_fc1 = quantize_blockwise(fc1_weight, MoeFormat.MXFP8, axis=1)
    q_fc2 = quantize_blockwise(fc2_weight, MoeFormat.MXFP8, axis=1)
    topk_idx = torch.tensor([[0], [1], [0]], dtype=torch.int64)
    topk_weights = torch.ones(tokens, 1)
    op = MoeEpReference(
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        top_k=1,
        combine_format="bf16",
        output_format="bf16",
        intermediate_format=intermediate_format,
    )

    actual = op(q_activation, q_fc1, q_fc2, topk_idx, topk_weights)
    expected = _naive_reference(
        q_activation.dequantize(),
        q_fc1.dequantize(),
        q_fc2.dequantize(),
        topk_idx,
        topk_weights,
        apply_topk_in_fc1=True,
        combine_format=MoeFormat.BF16,
        intermediate_format=intermediate_format,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


# Host-side EP topology and runtime bootstrap.


@pytest.mark.L0
def test_megamoe_capability_and_kernel_config_accept_ep_above_16():
    from cudnn import MoeEp
    from cudnn.moe_ep._megamoe_backend._capability import validate_config
    from cudnn.moe_ep._megamoe_backend.mxfp8._config import (
        Mxfp8KernelConfig,
    )

    with MoeEp(**_forward_config()) as op:
        config = replace(
            op._forward_config,
            num_experts=32,
            experts_per_rank=1,
            ep_size=32,
            ep_rank=31,
            ep_group=object(),
            ep_global_ranks=tuple(range(32)),
        )

    validate_config(config)
    kernel_config = Mxfp8KernelConfig.from_operator_config(config)
    assert kernel_config.world_size == 32
    assert kernel_config.local_rank == 31


@pytest.mark.L0
def test_ep32_peer_mapping_selects_version_compatible_payload():
    import cutlass
    from cutlass._mlir import ir
    from packaging.version import Version

    from cudnn.moe_ep._megamoe_backend._comm import PeerMapping
    from cudnn.moe_ep._megamoe_backend.cutedsl_src.communication.nvlink_domain.symmetric_buffer import (
        SymmetricBufferDevice,
    )

    offsets = tuple(index * 4096 for index in range(32))
    mapping = PeerMapping(
        base_address=0x1000,
        offsets=offsets,
        rank=0,
    )
    host = mapping.to_sym_buffer_host()
    with ir.Context():
        device_type = SymmetricBufferDevice(
            None,
            host.max_ranks,
        ).__get_mlir_types__()[0]
        device_type_text = str(device_type)

    assert host.offsets == offsets
    assert int(host.max_ranks) == 32
    dsl_release = Version(Version(cutlass.__version__).base_version)
    grid_constant_width_is_free = dsl_release < Version("4.0.0") or dsl_release >= Version("4.7.0")
    expected_type = "!llvm.ptr" if grid_constant_width_is_free else "vector<32xi64>"
    assert device_type_text == expected_type


@pytest.fixture
def runtime_module():
    from cudnn.moe_ep._megamoe_backend import _runtime

    with _runtime._PROCESS_RUNTIME_REGISTRY.lock:
        _runtime._PROCESS_RUNTIME_REGISTRY.active = None
    yield _runtime
    with _runtime._PROCESS_RUNTIME_REGISTRY.lock:
        _runtime._PROCESS_RUNTIME_REGISTRY.active = None


class _FakeRuntimeProvider:
    def __init__(self, runtime_module, state=None):
        self._runtime_module = runtime_module
        self._state = state or runtime_module.RuntimeInitState.NOT_INITIALIZED
        self._world = None
        self.initialize_count = 0
        self.finalize_count = 0

    def initialization_state(self):
        return self._state

    def initialize(self, device, world):
        del device
        self.initialize_count += 1
        self._world = world
        self._state = self._runtime_module.RuntimeInitState.INITIALIZED

    def rank(self):
        return self._world.rank

    def world_size(self):
        return self._world.size

    def device(self):
        return torch.device("cuda", 0)

    def finalize(self):
        self.finalize_count += 1
        self._state = self._runtime_module.RuntimeInitState.NOT_INITIALIZED


@pytest.mark.L0
def test_runtime_manager_shares_only_identical_subgroup(runtime_module):
    world = runtime_module.RuntimeWorld(
        rank=1,
        size=2,
        group=object(),
        global_ranks=(1, 3),
    )
    provider = _FakeRuntimeProvider(runtime_module)
    manager = runtime_module.RuntimeManager(
        provider_factory=lambda: provider,
        world_resolver=lambda config: world,
    )

    first = manager.acquire(object(), torch.device("cuda", 0))
    second = manager.acquire(object(), torch.device("cuda", 0))
    assert manager.ref_count == 2
    assert second.global_ranks == (1, 3)

    second.close()
    assert manager.ref_count == 1
    first.close()
    assert manager.ref_count == 0
    assert provider.finalize_count == 1


@pytest.mark.L0
def test_runtime_manager_keep_alive_reuses_until_explicit_shutdown(runtime_module):
    world = runtime_module.RuntimeWorld(
        rank=0,
        size=2,
        group=object(),
        global_ranks=(0, 1),
    )
    provider = _FakeRuntimeProvider(runtime_module)
    manager = runtime_module.RuntimeManager(
        provider_factory=lambda: provider,
        world_resolver=lambda config: world,
        keep_alive=True,
    )

    first = manager.acquire(object(), torch.device("cuda", 0))
    first.close()
    assert manager.ref_count == 0
    assert provider.initialize_count == 1
    assert provider.finalize_count == 0

    second = manager.acquire(object(), torch.device("cuda", 0))
    assert manager.ref_count == 1
    assert provider.initialize_count == 1
    second.close()

    manager.shutdown()
    assert manager.ref_count == 0
    assert provider.finalize_count == 1


@pytest.mark.L0
def test_runtime_manager_rejects_different_same_geometry_subgroup(
    runtime_module,
):
    first_world = runtime_module.RuntimeWorld(
        rank=0,
        size=2,
        group=object(),
        global_ranks=(0, 2),
    )
    second_world = runtime_module.RuntimeWorld(
        rank=0,
        size=2,
        group=object(),
        global_ranks=(0, 3),
    )
    provider = _FakeRuntimeProvider(runtime_module)
    first_manager = runtime_module.RuntimeManager(
        provider_factory=lambda: provider,
        world_resolver=lambda config: first_world,
    )
    second_manager = runtime_module.RuntimeManager(
        provider_factory=lambda: provider,
        world_resolver=lambda config: second_world,
    )

    handle = first_manager.acquire(object(), torch.device("cuda", 0))
    with pytest.raises(RuntimeError, match="different EP subgroup"):
        second_manager.acquire(object(), torch.device("cuda", 0))
    handle.close()


@pytest.mark.L0
def test_runtime_manager_rejects_unverifiable_external_subgroup(
    runtime_module,
    monkeypatch,
):
    world = runtime_module.RuntimeWorld(
        rank=0,
        size=2,
        group=object(),
        global_ranks=(1, 3),
    )
    provider = _FakeRuntimeProvider(
        runtime_module,
        runtime_module.RuntimeInitState.INITIALIZED,
    )
    provider._world = world
    manager = runtime_module.RuntimeManager(
        provider_factory=lambda: provider,
        world_resolver=lambda config: world,
    )
    monkeypatch.setattr(
        runtime_module,
        "_spans_default_distributed_world",
        lambda selected: False,
    )

    with pytest.raises(RuntimeError, match="cannot safely attach"):
        manager.acquire(object(), torch.device("cuda", 0))


@pytest.mark.L0
def test_nvshmem_uid_broadcast_uses_subgroup_root_global_rank(
    runtime_module,
    monkeypatch,
):
    class _FakeDevice:
        def __init__(self, index):
            self.index = index

        def set_current(self):
            return None

    cuda_module = ModuleType("cuda")
    cuda_core_module = ModuleType("cuda.core")
    cuda_experimental_module = ModuleType("cuda.core.experimental")
    cuda_experimental_module.Device = _FakeDevice
    cuda_core_module.experimental = cuda_experimental_module
    cuda_module.core = cuda_core_module
    monkeypatch.setitem(sys.modules, "cuda", cuda_module)
    monkeypatch.setitem(sys.modules, "cuda.core", cuda_core_module)
    monkeypatch.setitem(
        sys.modules,
        "cuda.core.experimental",
        cuda_experimental_module,
    )

    init_args = {}

    class _FakeUid:
        def __init__(self):
            self._data = np.arange(16, dtype=np.uint8)

    core = SimpleNamespace(
        get_unique_id=lambda empty: _FakeUid(),
        init=lambda **kwargs: init_args.update(kwargs),
    )
    monkeypatch.setattr(runtime_module, "_load_nvshmem_core", lambda: core)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)

    group = object()
    broadcast_args = {}
    monkeypatch.setattr(dist, "get_backend", lambda selected: "gloo")
    monkeypatch.setattr(
        dist,
        "get_global_rank",
        lambda selected, group_rank: (1, 3)[group_rank],
    )

    def _broadcast(tensor, *, src, group):
        broadcast_args.update(tensor=tensor, src=src, group=group)

    monkeypatch.setattr(dist, "broadcast", _broadcast)
    monkeypatch.setattr(dist, "barrier", lambda *, group: None)

    world = runtime_module.RuntimeWorld(
        rank=0,
        size=2,
        group=group,
        global_ranks=(1, 3),
    )
    runtime_module._DefaultNvshmemRuntimeProvider().initialize(
        torch.device("cuda", 0),
        world,
    )

    assert broadcast_args["src"] == 1
    assert broadcast_args["group"] is group
    assert broadcast_args["tensor"].device.type == "cpu"
    assert init_args["rank"] == 0
    assert init_args["nranks"] == 2
