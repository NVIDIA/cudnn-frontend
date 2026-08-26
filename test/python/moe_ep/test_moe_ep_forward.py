# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

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
from moe_ep.moe_ep_forward_support import (
    _assert_matches_reference,
    _forward_config,
    _make_forward_case,
    _naive_reference,
    _output_as_float,
    _reference_forward,
    _replay_cuda_graph,
    _require_distributed_sm107,
    _sm107_device,
    _stress_backend_reuse,
)
from moe_ep.moe_ep_reference import (
    BlockScaledTensor as ReferenceBlockScaledTensor,
    MoeEpReference,
    MoeFormat,
    forward_combine_round_trip,
    quantize_blockwise,
)
from moe_ep.moe_ep_test_data import (
    make_forward_inputs,
    quantize_mxfp8,
)



def _public_nvfp4(data, scale, logical_shape):
    from cudnn import BlockScaledTensor

    return BlockScaledTensor(
        data=data,
        scale=scale,
        format="nvfp4",
        logical_shape=logical_shape,
        axis=1,
    )


def _request(activation, fc1, fc2):
    return SimpleNamespace(
        activation=activation,
        fc1_weight=fc1,
        fc2_weight=fc2,
    )


# Public API, capability, layout, and workspace contracts.


@pytest.mark.L0
def test_moe_ep_finalizer_warns_without_retaining_failed_backend():
    import cudnn.moe_ep.api as api_module
    from cudnn import MoeEp

    class Backend:
        close_calls = 0

        def close(self):
            self.close_calls += 1
            raise RuntimeError("cleanup failed")

    operator = MoeEp(**_forward_config())
    backend = Backend()
    operator._forward_backend = backend

    with pytest.warns(ResourceWarning, match="cleanup failed"):
        operator.__del__()

    assert backend.close_calls == 1
    assert not hasattr(api_module, "_FAILED_FINALIZER_BACKENDS")
    operator._forward_backend = None
    operator._closed = True


@pytest.mark.L0
@pytest.mark.parametrize(
    ("field", "value"),
    [
        *(
            ("token_back_mode", value)
            for value in (
                "epi_warps",
                "standalone_warps",
                "reuse_dispatch_warps",
            )
        ),
        *(
            ("epi_flag_batch", value)
            for value in (
                (4, 2),
                (1, 1),
                (1, 2),
                (1, 4),
                (2, 1),
                (2, 2),
                (2, 4),
                (4, 4),
            )
        ),
        *(
            ("token_in_flag_batch", value)
            for value in (1, 2, 4, 8, 16)
        ),
        *(
            ("group_hint", value)
            for value in (None, 64, 128, 256, 512, 768, 1024)
        ),
        ("reduce_topk_in_kernel", False),
        ("reduce_topk_in_kernel", True),
    ],
)
def test_moe_ep_tuning_accepts_candidate_values(field, value):
    from cudnn import MoeEpTuningConfig

    tuning = MoeEpTuningConfig(**{field: value})
    assert getattr(tuning, field) == value


@pytest.mark.L0
@pytest.mark.parametrize(
    "kwargs",
    [
        {"token_back_mode": "unknown"},
        {"token_back_mode": []},
        {"epi_flag_batch": [1, 1]},
        {"epi_flag_batch": (3, 3)},
        {"token_in_flag_batch": 3},
        {"token_in_flag_batch": True},
        {"group_hint": 0},
        {"group_hint": True},
        {"reduce_topk_in_kernel": 1},
        {
            "token_back_mode": "standalone_warps",
            "reduce_topk_in_kernel": True,
        },
    ],
)
def test_moe_ep_tuning_rejects_unvalidated_values(kwargs):
    from cudnn import MoeEpTuningConfig

    with pytest.raises(ValueError):
        MoeEpTuningConfig(**kwargs)


@pytest.mark.L0
def test_moe_ep_tuning_public_contract_mapping_and_cache_key():
    from cudnn import MoeEp, MoeEpTuningConfig
    from cudnn.moe_ep import (
        MoeEpTuningConfig as PackageMoeEpTuningConfig,
    )
    from cudnn.moe_ep._megamoe_backend.mxfp8._config import (
        Mxfp8KernelConfig,
    )

    assert PackageMoeEpTuningConfig is MoeEpTuningConfig
    tuning = MoeEpTuningConfig(
        token_back_mode="epi_warps",
        epi_flag_batch=(4, 2),
        token_in_flag_batch=4,
        group_hint=768,
        reduce_topk_in_kernel=True,
    )
    with MoeEp(
        **_forward_config(),
        token_padding_size=64,
        sf_padding_size=256,
        tuning=tuning,
    ) as op:
        assert op.tuning is tuning
        kernel_config = Mxfp8KernelConfig.from_forward_config(
            op._forward_config
        )

    assert kernel_config.token_back_mode == "epi_warps"
    assert kernel_config.epi_flag_batch == (4, 2)
    assert kernel_config.flag_batch == 4
    assert kernel_config.group_hint == 768
    assert kernel_config.token_padding_block == 64
    assert kernel_config.sf_padding_block == 256
    assert kernel_config.tuning_signature(123) == (
        "epi_warps",
        (4, 2),
        4,
        768,
        True,
    )
    effective = kernel_config.effective_config(123)
    assert effective["token_padding_block"] == 64
    assert effective["sf_padding_block"] == 256
    assert effective["effective_group_hint"] == 768
    assert effective["fc2_in_kernel_topk_reduce"] is True
    assert effective["launch_cluster_count"] == 123
    assert effective["drop_on_overflow"] is True
    assert effective["enable_col_quant"] is False
    assert "output_format" not in effective

    with MoeEp(**_forward_config()) as default_op:
        default_config = Mxfp8KernelConfig.from_forward_config(
            default_op._forward_config
        )
    assert default_config.tuning_signature(123) == (
        "epi_warps",
        (1, 1),
        1,
        123,
        False,
    )
    key_args = (
        torch.device("cuda", 0),
        (10, 7),
        123,
        (),
    )
    assert kernel_config.compile_key(*key_args) != default_config.compile_key(
        *key_args
    )


@pytest.mark.L0
def test_internal_column_requant_config_is_disabled_by_default_and_cache_distinct():
    from cudnn import MoeEp
    from cudnn.moe_ep._megamoe_backend.mxfp8._config import (
        Mxfp8KernelConfig,
    )

    with MoeEp(**_forward_config()) as op:
        default_forward = op._forward_config
        default_config = Mxfp8KernelConfig.from_forward_config(default_forward)
        enabled_config = replace(
            default_config,
            enable_col_quant=True,
            col_quant_num_ctas=512,
        )

    assert default_config.enable_col_quant is False
    assert default_config.max_recv_size_per_rank == (
        default_forward.ep_size
        * default_forward.max_tokens_per_rank
        * default_forward.top_k
    )
    assert enabled_config.enable_col_quant is True
    assert enabled_config.col_quant_num_ctas == 512
    with pytest.raises(ValueError, match="max_recv_size_per_rank"):
        replace(default_config, max_recv_size_per_rank=0)
    with pytest.raises(ValueError, match="col_quant_num_ctas"):
        replace(default_config, col_quant_num_ctas=0)
    key_args = (torch.device("cuda", 0), (10, 7), 123, ())
    assert default_config.compile_key(*key_args) != enabled_config.compile_key(
        *key_args
    )


@pytest.mark.L0
@pytest.mark.parametrize(
    ("public_format", "wire_format"),
    [
        ("bf16", "bf16"),
        ("mxfp8", "32e4m3xe8m0"),
    ],
)
def test_combine_format_maps_to_contract_wire(public_format, wire_format):
    from cudnn import MoeEp
    from cudnn.moe_ep._megamoe_backend.mxfp8._config import (
        Mxfp8KernelConfig,
    )

    with MoeEp(
        **_forward_config(combine_format=public_format)
    ) as op:
        kernel_config = Mxfp8KernelConfig.from_forward_config(
            op._forward_config
        )

    assert kernel_config.combine_format == wire_format


@pytest.mark.L0
@pytest.mark.parametrize("combine_format", ["bf16", "mxfp8"])
def test_megamoe_capability_enables_combine_formats(combine_format):
    from cudnn import MoeEp
    from cudnn.moe_ep._megamoe_backend._capability import validate_config

    with MoeEp(**_forward_config(combine_format=combine_format)) as op:
        validate_config(op._forward_config)


@pytest.mark.L0
def test_distributed_topk_can_exceed_local_expert_count():
    from cudnn import MoeEp
    from cudnn.moe_ep._megamoe_backend._capability import validate_config

    with MoeEp(
        **_forward_config(
            num_experts=4,
            top_k=3,
        )
    ) as op:
        distributed = replace(
            op._forward_config,
            experts_per_rank=2,
            ep_size=2,
            ep_global_ranks=(0, 1),
        )

    validate_config(distributed)


@pytest.mark.L0
def test_overflow_check_prefers_device_assert(monkeypatch):
    from cudnn.moe_ep._megamoe_backend.mxfp8._launch import (
        _check_overflow,
    )

    calls = []

    def record_assert(condition, message):
        calls.append((condition.clone(), message))

    monkeypatch.setattr(torch, "_assert_async", record_assert)
    flag = torch.zeros(1, dtype=torch.int32)
    _check_overflow(flag)

    assert len(calls) == 1
    assert bool(calls[0][0])
    assert "route-pool overflow" in calls[0][1]


@pytest.mark.L0
def test_overflow_check_fallback_rejects_nonzero_flag(monkeypatch):
    from cudnn.moe_ep._megamoe_backend.mxfp8._launch import (
        _check_overflow,
    )

    monkeypatch.setattr(torch, "_assert_async", None)
    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        lambda: False,
    )
    with pytest.raises(RuntimeError, match="receive route-pool overflow"):
        _check_overflow(torch.ones(1, dtype=torch.int32))


@pytest.mark.L0
def test_in_kernel_topk_reduce_omits_standalone_combine_workspace():
    from cudnn.moe_ep._megamoe_backend.mxfp8._compile import (
        _pre_reduced_workspace_metadata,
    )

    class NoStandaloneWorkspace:
        def region(self, _name):
            raise AssertionError("in-kernel reduction must not query region")

        def offset(self, _name):
            raise AssertionError("in-kernel reduction must not query offset")

        def nbytes(self, _name):
            raise AssertionError("in-kernel reduction must not query size")

    config = SimpleNamespace(
        fc2_in_kernel_topk_reduce=True,
        top_k=6,
        hidden=7168,
        max_tokens_per_rank=4096,
        combine_format="bf16",
    )

    assert _pre_reduced_workspace_metadata(
        NoStandaloneWorkspace(),
        config,
        shared_bytes=0,
    ) == (None, 0)

    bytes_per_token = config.top_k * config.hidden * 2
    total_bytes = config.max_tokens_per_rank * bytes_per_token

    class StandaloneWorkspace:
        def region(self, _name):
            return SimpleNamespace(buffer_space="shared")

        def offset(self, _name):
            return 256

        def nbytes(self, _name):
            return total_bytes

    config.fc2_in_kernel_topk_reduce = False
    assert _pre_reduced_workspace_metadata(
        StandaloneWorkspace(),
        config,
        shared_bytes=256 + total_bytes,
    ) == (256, bytes_per_token)


@pytest.mark.L0
@pytest.mark.parametrize(
    ("combine_format", "bits_per_element"),
    [
        ("bf16", 16),
        ("32e4m3xe8m0", 8),
    ],
)
def test_standalone_combine_workspace_tracks_wire_width(
    combine_format,
    bits_per_element,
):
    from cudnn.moe_ep._megamoe_backend.mxfp8._compile import (
        _pre_reduced_workspace_metadata,
    )

    config = SimpleNamespace(
        fc2_in_kernel_topk_reduce=False,
        top_k=2,
        hidden=128,
        max_tokens_per_rank=5,
        combine_format=combine_format,
    )
    bytes_per_token = config.top_k * config.hidden * bits_per_element // 8
    total_bytes = config.max_tokens_per_rank * bytes_per_token

    class StandaloneWorkspace:
        def region(self, _name):
            return SimpleNamespace(buffer_space="shared")

        def offset(self, _name):
            return 128

        def nbytes(self, _name):
            return total_bytes

    assert _pre_reduced_workspace_metadata(
        StandaloneWorkspace(),
        config,
        shared_bytes=128 + total_bytes,
    ) == (128, bytes_per_token)


@pytest.mark.L0
@pytest.mark.parametrize(
    ("combine_format", "expected"),
    [
        ("bf16", (None, 0)),
        ("32e4m3xe8m0", (128, 64)),
    ],
)
def test_standalone_combine_scale_workspace_metadata(
    combine_format,
    expected,
):
    from cudnn.moe_ep._megamoe_backend.mxfp8._compile import (
        _pre_reduced_sf_workspace_metadata,
    )

    config = SimpleNamespace(
        fc2_in_kernel_topk_reduce=False,
        max_tokens_per_rank=5,
        combine_format=combine_format,
    )

    class StandaloneWorkspace:
        def region(self, _name):
            return SimpleNamespace(buffer_space="shared")

        def offset(self, _name):
            return 128

        def nbytes(self, _name):
            return config.max_tokens_per_rank * 64

    workspace = StandaloneWorkspace()
    if combine_format == "bf16":
        class NoScaleWorkspace:
            def region(self, _name):
                raise AssertionError("BF16 combine must not query scale region")

        workspace = NoScaleWorkspace()

    assert _pre_reduced_sf_workspace_metadata(
        workspace,
        config,
        shared_bytes=128 + config.max_tokens_per_rank * 64,
    ) == expected


@pytest.mark.L0
@pytest.mark.parametrize(
    "kwargs",
    [
        {"token_padding_size": True},
        {"token_padding_size": 0},
        {"token_padding_size": -1},
        {"token_padding_size": 64.0},
        {"sf_padding_size": True},
        {"sf_padding_size": 0},
        {"sf_padding_size": 64},
        {"sf_padding_size": 128.0},
    ],
)
def test_moe_ep_rejects_invalid_padding(kwargs):
    from cudnn import MoeEp

    with pytest.raises(ValueError):
        MoeEp(**_forward_config(), **kwargs)


@pytest.mark.L0
def test_moe_ep_rejects_untyped_tuning():
    from cudnn import MoeEp

    with pytest.raises(TypeError, match="MoeEpTuningConfig"):
        MoeEp(**_forward_config(), tuning={"group_hint": 768})


@pytest.mark.L0
@pytest.mark.parametrize(
    "kwargs",
    [
        {"combine_format": "mxfp8"},
        {"output_format": "mxfp8"},
        {"apply_topk_in_fc1": False},
    ],
)
def test_moe_ep_rejects_incompatible_in_kernel_topk_reduce(kwargs):
    from cudnn import MoeEp, MoeEpTuningConfig

    config = _forward_config()
    config.update(kwargs)
    with pytest.raises(ValueError, match="reduce_topk_in_kernel requires"):
        MoeEp(
            **config,
            tuning=MoeEpTuningConfig(reduce_topk_in_kernel=True),
        )


@pytest.mark.L0
def test_distributed_launch_rejects_mismatched_tuning_before_barrier(
    monkeypatch,
):
    from cudnn import MoeEp
    from cudnn.moe_ep._megamoe_backend.mxfp8._backend import (
        Mxfp8Backend,
    )

    with MoeEp(**_forward_config()) as op:
        backend = Mxfp8Backend(
            op._forward_config,
            torch.device("cuda", 0),
        )
    backend._ep_launch_ready = False

    stream = SimpleNamespace(synchronize=lambda: None)
    resources = SimpleNamespace(
        runtime=SimpleNamespace(group=object(), world_size=2)
    )
    prepared = SimpleNamespace(launch_cluster_count=123)
    monkeypatch.setattr(
        backend,
        "_ensure_prepared_kernel",
        lambda: prepared,
    )

    def mismatched_all_gather(output, signature, *, group):
        assert group is resources.runtime.group
        output[:] = [
            signature,
            ("standalone_warps", (1, 1), 1, 123),
        ]

    barrier_called = False

    def unexpected_barrier(*, group):
        nonlocal barrier_called
        barrier_called = True

    monkeypatch.setattr(dist, "all_gather_object", mismatched_all_gather)
    monkeypatch.setattr(dist, "barrier", unexpected_barrier)

    with pytest.raises(RuntimeError, match="MoeEp tuning must match"):
        backend._ensure_ep_launch_ready(resources, stream)
    assert not barrier_called
    assert not backend._ep_launch_ready


@pytest.mark.L0
def test_api_allocates_fresh_bf16_outputs_with_logical_shape():
    from cudnn import MoeEp

    device = _sm107_device()
    args = make_forward_inputs(device)
    activation, fc1_weight, fc2_weight = args[:3]

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


@pytest.mark.L0
@pytest.mark.parametrize(
    "kwargs",
    [
        {"combine_format": "nvfp4"},
        {"output_format": "mxfp8"},
        {"output_format": "nvfp4"},
        {"apply_topk_in_fc1": False},
    ],
)
def test_training_megamoe_rejects_unsupported_config_before_backend(kwargs):
    from cudnn import MoeEp
    from cudnn.moe_ep._megamoe_backend._capability import validate_config

    with MoeEp(**_forward_config(**kwargs)) as op:
        with pytest.raises(NotImplementedError, match="training MegaMoE"):
            validate_config(op._forward_config)


@pytest.mark.L0
def test_training_megamoe_rejects_nvfp4_operand_before_cuda_query(monkeypatch):
    from cudnn.moe_ep._megamoe_backend._capability import validate_request

    operand = _public_nvfp4(
        torch.zeros(2, 64, dtype=torch.uint8),
        torch.ones(2, 8).to(torch.float8_e4m3fn),
        (2, 128),
    )
    request = _request(operand, operand, operand)
    request.device = torch.device("cuda", 0)

    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: pytest.fail("CUDA capability queried too early"),
    )
    with pytest.raises(NotImplementedError, match="only MXFP8"):
        validate_request(request)


# Single-rank and distributed forward numerical parity.


@pytest.mark.L0
def test_fp8_activation_bf16_combine_forward_single_gpu():
    from cudnn import MoeEp

    device = _sm107_device()
    args = make_forward_inputs(device)
    expected = _reference_forward(args)

    with MoeEp(**_forward_config()) as op:
        actual = op(*args)
        torch.cuda.synchronize(device)

        args[3].fill_(-1)
        dropped = op(*args)
        torch.cuda.synchronize(device)

    assert actual.shape == (5, 128)
    assert actual.dtype == torch.bfloat16
    _assert_matches_reference(actual, expected)
    assert dropped.eq(0).all()


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
    "plain_mask",
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ],
)
@pytest.mark.parametrize(
    "plain_dtype",
    [torch.bfloat16, torch.float16, torch.float32],
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
        assert backend.kernel_config.tuning_signature(
            backend._prepared_kernel.launch_cluster_count
        ) == ("standalone_warps", (4, 2), 4, 64, False)

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
def test_generate_c_outputs_fc1_c_and_route_metadata():
    from cudnn import MoeEp

    device = _sm107_device()
    args = make_forward_inputs(device)
    config = _forward_config(
        gate_up_clamp=1.25,
        generate_c=True,
    )
    expected_output, expected_fc1_c, expected_metadata = _reference_forward(
        args,
        **config,
    )

    with MoeEp(**config) as op:
        first = op(*args)
        output, fc1_c, route_metadata = first
        fc1_c_snapshot = fc1_c.clone()
        metadata_snapshot = route_metadata.clone()

        scaled_args = (*args[:4], args[4] * 0.25)
        _, scaled_fc1_c, scaled_metadata = op(*scaled_args)
        torch.cuda.synchronize(device)

    assert isinstance(first, tuple)
    assert len(first) == 3
    assert output.shape == (5, 128)
    assert output.dtype == torch.bfloat16
    assert fc1_c.shape == (9, 512)
    assert fc1_c.dtype == torch.bfloat16
    assert route_metadata.shape == (9, 4)
    assert route_metadata.dtype == torch.int32
    _assert_matches_reference(output, expected_output)
    torch.testing.assert_close(
        _output_as_float(fc1_c),
        _output_as_float(expected_fc1_c),
        rtol=0.01,
        atol=0.01,
    )
    torch.testing.assert_close(
        route_metadata,
        expected_metadata,
        rtol=0,
        atol=0,
    )

    # FC1 C is captured before clamp/SwiGLU and does not include router weights.
    torch.testing.assert_close(scaled_fc1_c, fc1_c_snapshot, rtol=0, atol=0)
    torch.testing.assert_close(scaled_metadata, metadata_snapshot, rtol=0, atol=0)
    torch.testing.assert_close(fc1_c, fc1_c_snapshot, rtol=0, atol=0)
    torch.testing.assert_close(route_metadata, metadata_snapshot, rtol=0, atol=0)
    assert scaled_fc1_c is not fc1_c
    assert scaled_metadata is not route_metadata
    assert scaled_fc1_c.data_ptr() != fc1_c.data_ptr()
    assert scaled_metadata.data_ptr() != route_metadata.data_ptr()


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
            32,
            torch.int64,
            torch.float32,
            id="topk32-boundary-int64-fp32",
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
def test_single_gpu_stress_and_cuda_graph_replay():
    from cudnn import MoeEp

    device = _sm107_device()
    args = make_forward_inputs(device)
    original_topk_idx = args[3].clone()
    original_topk_weights = args[4].clone()
    config = _forward_config()
    expected = _reference_forward(args, **config)

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
        eager = op(*args)
        torch.cuda.synchronize(device)
        _assert_matches_reference(eager, expected)
        _replay_cuda_graph(op, args, original_topk_idx, expected, device)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
def test_nondefault_tuning_warmup_and_cuda_graph_replay():
    from cudnn import MoeEp, MoeEpTuningConfig

    device = _sm107_device()
    args = make_forward_inputs(device)
    original_topk_idx = args[3].clone()
    expected = _reference_forward(args)
    tuning = MoeEpTuningConfig(
        token_back_mode="reuse_dispatch_warps",
        epi_flag_batch=(2, 2),
        token_in_flag_batch=2,
        group_hint=128,
    )

    with MoeEp(**_forward_config(), tuning=tuning) as op:
        _replay_cuda_graph(
            op,
            args,
            original_topk_idx,
            expected,
            device,
        )


@pytest.mark.L0
def test_reference_apply_topk_after_fc2_weights_after_combine_rounding():
    """Keep post-combine router weighting in reference-only semantics."""

    device = torch.device("cpu")
    args = make_forward_inputs(device)
    # Duplicate routes make pre/post-combine weighting observably different.
    args[3].copy_(
        torch.tensor(
            [[0, 0], [1, 1], [0, 0], [1, 1], [0, 0]],
            dtype=torch.int32,
            device=device,
        )
    )
    args[4].copy_(
        torch.tensor(
            [[256.0, -255.0]] * 5,
            dtype=torch.bfloat16,
            device=device,
        )
    )
    decoded_args = (
        args[0].dequantize(),
        args[1].dequantize(),
        args[2].dequantize(),
        args[3],
        args[4],
    )
    expected = _naive_reference(
        *decoded_args,
        apply_topk_in_fc1=False,
        intermediate_format=MoeFormat.MXFP8,
        apply_topk_after_combine=True,
    )
    pre_combine_weighting = _naive_reference(
        *decoded_args,
        apply_topk_in_fc1=False,
        intermediate_format=MoeFormat.MXFP8,
    )
    assert not torch.equal(expected, pre_combine_weighting)

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
    "world_size",
    [2, 3, 4],
    ids=["ep2", "ep3", "ep4"],
)
@pytest.mark.parametrize("combine_format", ["bf16", "mxfp8"])
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
def test_intermediate_requires_full_mma_n_tile():
    from cudnn import MoeEp

    args = _make_forward_case(
        torch.device("cpu"),
        experts=2,
        tokens=3,
        hidden=128,
        intermediate=128,
        top_k=2,
        index_dtype=torch.int32,
        weight_dtype=torch.bfloat16,
    )
    with MoeEp(
        **_forward_config(intermediate_size=128, max_tokens_per_rank=3)
    ) as op:
        with pytest.raises(
            NotImplementedError,
            match=r"intermediate_size .*divisible by 256",
        ):
            op(*args)
        assert op._forward_backend is None


@pytest.mark.L0
def test_activation_scale_rows_are_padded_to_16_bytes():
    from cudnn import MoeEp
    from cudnn.moe_ep._megamoe_backend._workspace import (
        WorkspaceRequirements,
        padded_mxfp8_scale_columns,
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
    activation_scale = next(
        region
        for region in requirements.symmetric_regions
        if region.name == "activation_scale"
    )
    assert activation_scale.nbytes == 5 * 16


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
    enabled_sizes = {
        region.name: region.nbytes for region in enabled.local_regions
    }
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
def test_mxfp8_activation_representation():
    activation = quantize_mxfp8(torch.randn(3, 128), axis=1)

    assert activation.format.value == "mxfp8"
    assert activation.logical_shape == (3, 128)
    assert activation.axis == 1
    assert activation.data.shape == (3, 128)
    assert activation.data.dtype == torch.float8_e4m3fn
    assert activation.scale.shape == (3, 4)
    assert activation.scale.dtype == torch.float8_e8m0fnu
    assert torch.isfinite(activation.dequantize()).all()


@pytest.mark.L0
def test_reference_mxfp8_block_scaled_round_trip():
    values = torch.linspace(-4.0, 4.0, 3 * 64).reshape(3, 64)
    quantized = quantize_blockwise(values, MoeFormat.MXFP8)

    assert isinstance(quantized, ReferenceBlockScaledTensor)
    assert quantized.format is MoeFormat.MXFP8
    assert quantized.logical_shape == (3, 64)
    assert tuple(quantized.data.shape) == (3, 64)
    assert tuple(quantized.scale.shape) == (3, 2)
    assert quantized.scale.dtype == torch.float8_e8m0fnu
    assert quantized.dequantize().shape == values.shape
    assert torch.isfinite(quantized.dequantize()).all()


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
def test_resolve_ep_topology_preserves_group_rank_order(monkeypatch):
    from cudnn.moe_ep.api import _resolve_ep_topology

    group = object()
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda selected=None: 2)
    monkeypatch.setattr(
        dist,
        "get_rank",
        lambda selected=None: 1,
    )
    monkeypatch.setattr(
        dist,
        "get_global_rank",
        lambda selected, group_rank: (3, 1)[group_rank],
    )

    assert _resolve_ep_topology(group) == (2, 1, (3, 1))


@pytest.mark.L0
def test_resolve_ep_topology_rejects_nonmember(monkeypatch):
    from cudnn.moe_ep.api import _resolve_ep_topology

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(dist, "get_rank", lambda group: -1)

    with pytest.raises(ValueError, match="must be a member"):
        _resolve_ep_topology(object())


@pytest.mark.L0
def test_resolve_runtime_world_revalidates_ordered_membership(monkeypatch):
    from cudnn.moe_ep._megamoe_backend._runtime import _resolve_world

    group = object()
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda selected: 2)
    monkeypatch.setattr(dist, "get_rank", lambda selected: 1)
    monkeypatch.setattr(
        dist,
        "get_global_rank",
        lambda selected, group_rank: (3, 1)[group_rank],
    )
    config = SimpleNamespace(
        ep_group=group,
        ep_size=2,
        ep_rank=1,
        ep_global_ranks=(3, 1),
    )

    world = _resolve_world(config)
    assert world.identity == (1, 2, (3, 1))

    config.ep_global_ranks = (1, 3)
    with pytest.raises(RuntimeError, match="membership does not match"):
        _resolve_world(config)


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
    kernel_config = Mxfp8KernelConfig.from_forward_config(config)
    assert kernel_config.world_size == 32
    assert kernel_config.local_rank == 31


@pytest.mark.L0
def test_megamoe_capability_accepts_nonworld_subgroup_config():
    from cudnn.moe_ep._contracts import ForwardConfig
    from cudnn.moe_ep._megamoe_backend._capability import validate_config
    from cudnn.moe_ep._tuning import MoeEpTuningConfig

    config = ForwardConfig(
        num_experts=4,
        hidden_size=128,
        intermediate_size=256,
        top_k=2,
        experts_per_rank=2,
        ep_size=2,
        ep_rank=0,
        ep_group=object(),
        ep_global_ranks=(1, 3),
        max_tokens_per_rank=8,
        output_format="bf16",
        combine_format="bf16",
        apply_topk_in_fc1=True,
        gate_up_clamp=None,
        generate_c=False,
        token_padding_size=128,
        sf_padding_size=128,
        tuning=MoeEpTuningConfig(),
    )

    validate_config(config)


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
        self.finalize_count = 0

    def initialization_state(self):
        return self._state

    def initialize(self, device, world):
        del device
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
