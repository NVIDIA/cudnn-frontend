# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Focused L0 tests for the public MoeEP wgrad operand contract."""

from __future__ import annotations

import os
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

from cudnn.moe_ep import (
    MoeEp,
    MoeEpWgradForwardStash,
    MoeEpWgradOperands,
)
from cudnn.moe_ep._megamoe_backend import _capability
from cudnn.moe_ep._megamoe_backend.mxfp8._backward_launch import (
    Mxfp8DgluResult,
)
from cudnn.moe_ep._megamoe_backend.mxfp8._backward_wgrad_export import (
    export_wgrad_operands,
)
from cudnn.moe_ep._megamoe_backend.mxfp8._config import Mxfp8KernelConfig
from cudnn.moe_ep._megamoe_backend.mxfp8._stash import (
    Mxfp8ForwardStash as Mxfp8ForwardStashOwner,
)
from cudnn.moe_ep._megamoe_backend.mxfp8._wgrad_layout import (
    assemble_dfc2_atom_scales,
    assemble_discrete_col_requant_scales,
    assemble_plain_col_scales,
)
from cudnn.moe_ep._validation import validate_backward
from moe_ep.moe_ep_backward_support import _dense_wgrads_from_operands
from moe_ep.moe_ep_distributed_workers import (
    _distributed_wgrad_worker,
    _run_wgrad_operand_case,
)
from moe_ep.moe_ep_forward_support import (
    _require_distributed_sm107,
    _sm107_device,
)
from moe_ep.moe_ep_reference import (
    MoeEpReference,
    MoeFormat,
    WgradOperandsReference,
    quantize_blockwise,
)


def _inputs():
    activation = torch.randn(2, 128, dtype=torch.bfloat16)
    fc1_weight = torch.randn(2, 128, 512, dtype=torch.bfloat16)
    fc2_weight = torch.randn(2, 256, 128, dtype=torch.bfloat16)
    topk_idx = torch.tensor([[0, -1], [1, 0]], dtype=torch.int32)
    topk_weights = torch.randn(2, 2, dtype=torch.float32)
    return activation, fc1_weight, fc2_weight, topk_idx, topk_weights


def _route_metadata():
    return torch.tensor(
        [[0, 0, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]],
        dtype=torch.int32,
    )


def _forward_stash(route_metadata=None):
    if route_metadata is None:
        route_metadata = _route_metadata()
    return MoeEpWgradForwardStash(
        fc1_a=torch.empty(128, 512, dtype=torch.float8_e4m3fn),
        fc1_sfa=torch.empty(128, 16, dtype=torch.float8_e8m0fnu),
        expert_offsets=torch.tensor([256, 512], dtype=torch.int32),
        valid_route_counts=torch.tensor([2, 1], dtype=torch.int32),
        route_metadata=route_metadata.clone(),
    )


def _operator(**overrides):
    kwargs = {
        "num_experts": 2,
        "hidden_size": 128,
        "intermediate_size": 256,
        "top_k": 2,
        "max_tokens_per_rank": 4,
        "generate_c": True,
    }
    kwargs.update(overrides)
    if (
        kwargs.get("backward_wgrad_mode") == "operands"
        and "token_padding_size" not in overrides
    ):
        kwargs["token_padding_size"] = 256
    return MoeEp(**kwargs)


@pytest.mark.L0
def test_wgrad_types_are_public_and_have_stable_fields():
    from cudnn import (
        MoeEpWgradForwardStash as TopLevelForwardStash,
        MoeEpWgradOperands as TopLevelOperands,
    )

    assert TopLevelForwardStash is MoeEpWgradForwardStash
    assert TopLevelOperands is MoeEpWgradOperands
    assert [field.name for field in fields(MoeEpWgradForwardStash)] == [
        "fc1_a",
        "fc1_sfa",
        "expert_offsets",
        "valid_route_counts",
        "route_metadata",
    ]
    assert [field.name for field in fields(MoeEpWgradOperands)] == [
        "fc1_a",
        "fc1_sfa",
        "fc1_b",
        "fc1_sfb",
        "fc2_a",
        "fc2_sfa",
        "fc2_b",
        "fc2_sfb",
        "expert_offsets",
        "valid_route_counts",
        "route_metadata",
    ]


@pytest.mark.L0
def test_wgrad_mode_is_opt_in_and_requires_generate_c():
    with _operator(generate_c=False) as operator:
        assert operator.backward_wgrad_mode == "none"
        assert operator._forward_config.backward_wgrad_mode == "none"

    with pytest.raises(ValueError, match="must be 'none' or 'operands'"):
        _operator(backward_wgrad_mode="weights")
    with pytest.raises(ValueError, match="requires generate_c=True"):
        _operator(
            generate_c=False,
            backward_wgrad_mode="operands",
        )
    with pytest.raises(ValueError, match="requires token_padding_size=256"):
        _operator(
            backward_wgrad_mode="operands",
            token_padding_size=128,
        )
    with pytest.raises(ValueError, match="requires sf_padding_size=128"):
        _operator(
            backward_wgrad_mode="operands",
            sf_padding_size=256,
        )

    with _operator(backward_wgrad_mode="operands") as operator:
        assert operator.backward_wgrad_mode == "operands"
        assert operator._forward_config.backward_wgrad_mode == "operands"


@pytest.mark.L0
def test_validate_backward_checks_wgrad_stash_layout_and_route_identity():
    args = _inputs()
    route_metadata = _route_metadata()
    stash = _forward_stash(route_metadata)
    with _operator(backward_wgrad_mode="operands") as operator:
        with pytest.raises(TypeError, match="must be a MoeEpWgradForwardStash"):
            validate_backward(
                operator._forward_config,
                torch.randn(2, 128),
                *args[1:],
                torch.randn(3, 512, dtype=torch.bfloat16),
                route_metadata,
            )

        request = validate_backward(
            operator._forward_config,
            torch.randn(2, 128),
            *args[1:],
            torch.randn(3, 512, dtype=torch.bfloat16),
            route_metadata,
            wgrad_forward_stash=stash,
        )
        assert request.wgrad_forward_stash is stash

        wrong_scale_shape = replace(
            stash,
            fc1_sfa=torch.empty(128, 15, dtype=torch.float8_e8m0fnu),
        )
        with pytest.raises(ValueError, match="fc1_sfa shape must be"):
            validate_backward(
                operator._forward_config,
                torch.randn(2, 128),
                *args[1:],
                torch.randn(3, 512, dtype=torch.bfloat16),
                route_metadata,
                wgrad_forward_stash=wrong_scale_shape,
            )

        wrong_a_stride = replace(
            stash,
            fc1_a=torch.empty(
                512,
                128,
                dtype=torch.float8_e4m3fn,
            ).transpose(0, 1),
        )
        with pytest.raises(ValueError, match=r"compact \(K, 1\) strides"):
            validate_backward(
                operator._forward_config,
                torch.randn(2, 128),
                *args[1:],
                torch.randn(3, 512, dtype=torch.bfloat16),
                route_metadata,
                wgrad_forward_stash=wrong_a_stride,
            )

        wrong_scale_stride = replace(
            stash,
            fc1_sfa=torch.empty(
                16,
                128,
                dtype=torch.float8_e8m0fnu,
            ).transpose(0, 1),
        )
        with pytest.raises(ValueError, match="fc1_sfa must be contiguous"):
            validate_backward(
                operator._forward_config,
                torch.randn(2, 128),
                *args[1:],
                torch.randn(3, 512, dtype=torch.bfloat16),
                route_metadata,
                wgrad_forward_stash=wrong_scale_stride,
            )

        noncanonical_offsets = replace(
            stash,
            expert_offsets=torch.tensor([512, 768], dtype=torch.int32),
        )
        with pytest.raises(ValueError, match="canonical 256-row padding"):
            validate_backward(
                operator._forward_config,
                torch.randn(2, 128),
                *args[1:],
                torch.randn(3, 512, dtype=torch.bfloat16),
                route_metadata,
                wgrad_forward_stash=noncanonical_offsets,
            )

        wrong_identity = _forward_stash(route_metadata.flip(0))
        with pytest.raises(ValueError, match="route identity"):
            validate_backward(
                operator._forward_config,
                torch.randn(2, 128),
                *args[1:],
                torch.randn(3, 512, dtype=torch.bfloat16),
                route_metadata,
                wgrad_forward_stash=wrong_identity,
            )

        wrong_counts = MoeEpWgradForwardStash(
            stash.fc1_a,
            stash.fc1_sfa,
            stash.expert_offsets,
            torch.tensor([1, 2], dtype=torch.int32),
            stash.route_metadata,
        )
        with pytest.raises(ValueError, match="do not match route_metadata"):
            validate_backward(
                operator._forward_config,
                torch.randn(2, 128),
                *args[1:],
                torch.randn(3, 512, dtype=torch.bfloat16),
                route_metadata,
                wgrad_forward_stash=wrong_counts,
            )


@pytest.mark.L0
def test_default_mode_rejects_wgrad_stash_without_changing_default_contract():
    args = _inputs()
    route_metadata = _route_metadata()
    with _operator() as operator:
        with pytest.raises(ValueError, match="only accepted"):
            validate_backward(
                operator._forward_config,
                torch.randn(2, 128),
                *args[1:],
                torch.randn(3, 512, dtype=torch.bfloat16),
                route_metadata,
                wgrad_forward_stash=_forward_stash(route_metadata),
            )


@pytest.mark.L0
def test_wgrad_mode_is_backend_capable_and_enables_forward_col_quant(
    monkeypatch,
):
    with _operator(backward_wgrad_mode="operands") as operator:
        monkeypatch.setattr(
            _capability,
            "_validate_device",
            lambda device: pytest.fail("device capability queried"),
        )
        _capability.validate_config(operator._forward_config)
        kernel_config = Mxfp8KernelConfig.from_forward_config(
            operator._forward_config
        )

    assert kernel_config.enable_col_quant is True
    assert kernel_config.token_padding_block == 256


@pytest.mark.L0
def test_forward_col_quant_runtime_uses_static_cute_layout(monkeypatch):
    from cudnn.moe_ep._megamoe_backend.mxfp8 import _launch

    tensors = {
        name: object()
        for name in (
            "activation",
            "activation_sf",
            "topk_indices",
            "topk_scores",
            "fc1_weight",
            "fc1_weight_sf",
            "fc2_weight",
            "fc2_weight_sf",
            "fc1_c",
            "output_data",
            "col_quant_data",
            "col_quant_sf",
            "overflow_flag",
            "local_workspace",
            "shared_workspace",
        )
    }
    calls = {}

    def fake_to_cute(
        tensor,
        assumed_align=16,
        *,
        dynamic_layout=True,
    ):
        calls[id(tensor)] = (assumed_align, dynamic_layout)
        return tensor

    monkeypatch.setattr(_launch, "_to_cute", fake_to_cute)
    monkeypatch.setattr(_launch, "_to_cute_ptr", lambda tensor: tensor)
    inputs = SimpleNamespace(
        activation=tensors["activation"],
        activation_sf=tensors["activation_sf"],
        topk_indices=tensors["topk_indices"],
        topk_scores=tensors["topk_scores"],
        weights=SimpleNamespace(
            fc1_weight=tensors["fc1_weight"],
            fc1_weight_sf=tensors["fc1_weight_sf"],
            fc2_weight=tensors["fc2_weight"],
            fc2_weight_sf=tensors["fc2_weight_sf"],
        ),
        fc1_c=tensors["fc1_c"],
        output_data=tensors["output_data"],
        col_quant_data=tensors["col_quant_data"],
        col_quant_sf=tensors["col_quant_sf"],
        overflow_flag=tensors["overflow_flag"],
        local_workspace=tensors["local_workspace"],
        shared_workspace=tensors["shared_workspace"],
    )
    resources = SimpleNamespace(
        runtime=SimpleNamespace(
            current_stream=lambda: SimpleNamespace(cuda_stream=0)
        ),
        workspace=SimpleNamespace(
            peer_mapping=SimpleNamespace(
                to_sym_buffer_host=lambda: object()
            )
        ),
    )

    _launch.build_runtime_kwargs(inputs, resources)

    assert calls[id(tensors["col_quant_data"])] == (128, False)
    assert calls[id(tensors["col_quant_sf"])] == (16, False)
    assert calls[id(tensors["overflow_flag"])] == (4, False)


@pytest.mark.L0
def test_opt_in_forward_and_backward_results_are_backend_representable(
    monkeypatch,
):
    import cudnn.moe_ep._backend as backend_seam

    args = _inputs()
    route_metadata = _route_metadata()
    fc1_c = torch.randn(3, 512, dtype=torch.bfloat16)
    stash = _forward_stash(route_metadata)
    forward_result = (torch.randn(2, 128), fc1_c, route_metadata, stash)
    operands = MoeEpWgradOperands(
        fc1_a=stash.fc1_a,
        fc1_sfa=stash.fc1_sfa,
        fc1_b=torch.empty(512, 512, dtype=torch.float8_e4m3fn),
        fc1_sfb=torch.empty(512, 16, dtype=torch.float8_e8m0fnu),
        fc2_a=torch.empty(256, 512, dtype=torch.float8_e4m3fn),
        fc2_sfa=torch.empty(256, 16, dtype=torch.float8_e8m0fnu),
        fc2_b=torch.empty(512, 128, dtype=torch.float8_e4m3fn),
        fc2_sfb=torch.empty(128, 16, dtype=torch.float8_e8m0fnu),
        expert_offsets=stash.expert_offsets,
        valid_route_counts=stash.valid_route_counts,
        route_metadata=stash.route_metadata,
    )
    backward_result = (
        torch.randn(2, 128),
        torch.randn(2, 2),
        operands,
    )

    class Backend:
        backward_request = None

        def forward(self, request):
            return forward_result

        def backward(self, request):
            self.backward_request = request
            return backward_result

        def close(self):
            pass

    backend = Backend()
    monkeypatch.setattr(backend_seam, "validate_config", lambda config: None)
    monkeypatch.setattr(backend_seam, "validate_request", lambda request: None)
    monkeypatch.setattr(
        backend_seam,
        "validate_backward_request",
        lambda request: None,
    )
    monkeypatch.setattr(
        backend_seam,
        "create_backend",
        lambda config, device: backend,
    )

    with _operator(backward_wgrad_mode="operands") as operator:
        assert operator(*args) is forward_result
        actual = operator.backward(
            torch.randn(2, 128),
            *args[1:],
            fc1_c,
            route_metadata,
            wgrad_forward_stash=stash,
        )

    assert actual is backward_result
    assert backend.backward_request.wgrad_forward_stash is stash


def _blocked_reference(raw: torch.Tensor) -> torch.Tensor:
    rows, columns = raw.shape
    if columns == 0:
        return raw.new_empty((0,))
    padded_rows = (rows + 127) // 128 * 128
    padded_columns = (columns + 3) // 4 * 4
    padded = torch.full(
        (padded_rows, padded_columns),
        127,
        dtype=torch.uint8,
        device=raw.device,
    )
    padded[:rows, :columns] = raw
    return (
        padded.view(padded_rows // 128, 128, padded_columns // 4, 4)
        .permute(0, 2, 1, 3)
        .reshape(-1, 4, 32, 4)
        .transpose(1, 2)
        .reshape(-1)
    )


@pytest.mark.L0
def test_col_requant_scales_accept_upstream_hidden_atom_major_order():
    non_k = 256
    valid_counts = (33, 0, 129)
    padded_ends = (256, 256, 512)
    source0 = torch.arange(non_k * 4, dtype=torch.int32).to(torch.uint8).reshape(
        non_k,
        4,
    )
    source2 = (
        torch.arange(non_k * 8, dtype=torch.int32) + 37
    ).to(torch.uint8).reshape(non_k, 8)
    target0 = torch.full((non_k, 8), 127, dtype=torch.uint8)
    target0[:, :4] = source0
    blocked0 = _blocked_reference(target0)
    blocked2 = _blocked_reference(source2)

    # Upstream col requant stores hidden atoms before token atoms inside each
    # expert. Its 128-row SF padding is expanded to the 256-row data padding.
    packed = torch.cat(
        (_blocked_reference(source0), _blocked_reference(source2))
    )
    actual = assemble_discrete_col_requant_scales(
        packed,
        valid_counts,
        padded_ends,
        non_k,
        128,
    )
    expected = torch.cat((blocked0, blocked2)).reshape(non_k, 16)

    assert actual.shape == (non_k, 16)
    assert torch.equal(actual.view(torch.uint8), expected)


@pytest.mark.L0
def test_dfc2_atom_scales_reorder_token_major_atoms_per_expert():
    non_k = 256
    valid_counts = (33, 0, 129)
    padded_ends = (256, 256, 512)
    source0 = torch.arange(non_k * 4, dtype=torch.int32).to(torch.uint8).reshape(
        non_k,
        4,
    )
    source2 = (
        torch.arange(non_k * 8, dtype=torch.int32) + 37
    ).to(torch.uint8).reshape(non_k, 8)
    target0 = torch.full((non_k, 8), 127, dtype=torch.uint8)
    target0[:, :4] = source0
    blocked0 = _blocked_reference(target0)
    blocked2 = _blocked_reference(source2)
    source_blocked0 = _blocked_reference(source0)
    source_blocked2 = _blocked_reference(source2)

    # The dFC2 epilogue writes token atoms before hidden atoms.
    physical0 = source_blocked0.reshape(2, 1, 512).permute(1, 0, 2).reshape(-1)
    physical2 = source_blocked2.reshape(2, 2, 512).permute(1, 0, 2).reshape(-1)
    actual = assemble_dfc2_atom_scales(
        torch.cat((physical0, physical2)),
        valid_counts,
        padded_ends,
        non_k,
        128,
    )
    expected = torch.cat((blocked0, blocked2)).reshape(non_k, 16)

    assert actual.shape == (non_k, 16)
    assert torch.equal(actual.view(torch.uint8), expected)


@pytest.mark.L0
def test_dfc2_atom_scales_deinterleave_gate_up_before_repacking():
    intermediate = 256
    non_k = 2 * intermediate
    logical_source = torch.arange(
        non_k * 4,
        dtype=torch.int32,
    ).to(torch.uint8).reshape(non_k, 4)
    gate = logical_source[:intermediate].reshape(8, 32, 4)
    up = logical_source[intermediate:].reshape(8, 32, 4)
    interleaved_source = torch.stack((gate, up), dim=1).reshape(non_k, 4)
    target = torch.full((non_k, 8), 127, dtype=torch.uint8)
    target[:, :4] = logical_source

    actual = assemble_dfc2_atom_scales(
        _blocked_reference(interleaved_source),
        (33,),
        (256,),
        non_k,
        128,
        deinterleave_gate_up=intermediate,
    )

    assert torch.equal(
        actual.view(torch.uint8),
        _blocked_reference(target).reshape(non_k, 8),
    )


@pytest.mark.L0
def test_plain_col_scales_assemble_256_data_padding_with_empty_expert():
    non_k = 256
    counts = (33, 0, 129)
    padded_ends = (256, 256, 512)
    source = torch.full((12, non_k), 127, dtype=torch.uint8)
    source[:2] = torch.arange(2 * non_k, dtype=torch.int32).to(
        torch.uint8
    ).reshape(2, non_k)
    source[4:9] = (
        torch.arange(5 * non_k, dtype=torch.int32) + 19
    ).to(torch.uint8).reshape(5, non_k)

    raw0 = torch.full((non_k, 8), 127, dtype=torch.uint8)
    raw0[:, :2] = source[:2].transpose(0, 1)
    raw1 = torch.empty((non_k, 0), dtype=torch.uint8)
    raw2 = torch.full((non_k, 8), 127, dtype=torch.uint8)
    raw2[:, :5] = source[4:9].transpose(0, 1)
    expected = torch.cat(
        tuple(
            _blocked_reference(raw)
            for raw in (raw0, raw1, raw2)
        )
    ).reshape(non_k, 16)

    actual = assemble_plain_col_scales(
        source.view(torch.float8_e8m0fnu),
        counts,
        padded_ends,
        non_k,
        128,
    )

    assert torch.equal(actual.view(torch.uint8), expected)
    empty_plain = assemble_plain_col_scales(
        torch.empty(0, non_k, dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        ),
        (0, 0),
        (0, 0),
        non_k,
        128,
    )
    empty_discrete = assemble_discrete_col_requant_scales(
        torch.empty(0, dtype=torch.uint8),
        (0, 0),
        (0, 0),
        non_k,
        128,
    )
    empty_dfc2 = assemble_dfc2_atom_scales(
        torch.empty(0, dtype=torch.uint8),
        (0, 0),
        (0, 0),
        non_k,
        128,
    )
    assert empty_plain.shape == (non_k, 0)
    assert empty_discrete.shape == (non_k, 0)
    assert empty_dfc2.shape == (non_k, 0)


@pytest.mark.L0
def test_forward_materializes_caller_owned_256_padded_operand_stash():
    operator = _operator(backward_wgrad_mode="operands")
    config = operator._forward_config
    owner = Mxfp8ForwardStashOwner(config, torch.device("cpu"))
    request = SimpleNamespace(
        topk_idx=torch.tensor([[0, -1], [1, 0]], dtype=torch.int32),
        device=torch.device("cpu"),
    )
    plan = owner.prepare(request, pool_token_capacity=512)
    plan.buffer.zero_()
    plan.buffer[0, 0] = 10
    plan.buffer[1, 0] = 20
    plan.buffer[256, 0] = 30

    def pack(token, slot):
        return token | (slot << 32)

    packed_metadata = torch.zeros(512, dtype=torch.int64)
    packed_metadata[0] = pack(1, 1)
    packed_metadata[1] = pack(0, 0)
    packed_metadata[256] = pack(1, 0)
    col_data = torch.zeros(
        512,
        128,
        dtype=torch.float8_e4m3fn,
    )
    col_sf = torch.full(
        (512 // 32 * 128,),
        127,
        dtype=torch.uint8,
    )
    inputs = SimpleNamespace(
        fc1_c=plan.buffer,
        shared_workspace=packed_metadata.view(torch.uint8),
        col_quant_data=col_data,
        col_quant_sf=col_sf,
    )
    prepared = SimpleNamespace(
        token_src_metadata_offset=0,
        token_src_metadata_bytes=512 * 8,
        pool_token_capacity=512,
    )

    fc1_c, route_metadata, stash = owner.materialize(
        plan,
        inputs,
        prepared,
    )

    assert stash is not None
    assert stash.fc1_a.shape == (128, 512)
    assert stash.fc1_a.stride(1) == 1
    assert stash.fc1_sfa.shape == (128, 16)
    assert stash.expert_offsets.tolist() == [256, 512]
    assert stash.valid_route_counts.tolist() == [2, 1]
    assert stash.route_metadata is route_metadata
    assert route_metadata.tolist() == [
        [0, 0, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 1, 0],
    ]
    assert fc1_c[:, 0].tolist() == [20, 10, 30]
    preserved = stash.fc1_a.clone()
    col_data.fill_(1)
    assert torch.equal(stash.fc1_a, preserved)
    owner.close()
    operator.close()


@pytest.mark.L0
def test_backward_export_owns_outputs_and_uses_grouped_wgrad_strides():
    config = _operator(
        backward_wgrad_mode="operands"
    )._forward_config
    stash = _forward_stash()
    request = SimpleNamespace(
        config=config,
        wgrad_forward_stash=stash,
    )
    pool_rows = 512
    sf_rows = 8
    aux = Mxfp8DgluResult(
        grad_activation=torch.empty(2, 128),
        fc1_recompute=torch.zeros(
            pool_rows,
            256,
            dtype=torch.float8_e4m3fn,
        ),
        fc1_recompute_sf=torch.full(
            (sf_rows, 256),
            127,
            dtype=torch.uint8,
        ).view(torch.float8_e8m0fnu),
        fc1_col_output=torch.zeros(
            pool_rows,
            512,
            dtype=torch.float8_e4m3fn,
        ),
        fc1_col_output_sf=torch.full(
            (sf_rows, 512),
            127,
            dtype=torch.uint8,
        ).view(torch.float8_e8m0fnu),
        grad_y2=torch.zeros(
            pool_rows,
            128,
            dtype=torch.float8_e4m3fn,
        ),
        grad_y2_sf=torch.full(
            (pool_rows // 32 * 128,),
            127,
            dtype=torch.uint8,
        ),
    )

    operands = export_wgrad_operands(request, aux)

    assert operands.fc1_b.shape == (512, 512)
    assert operands.fc1_b.stride(0) == 1
    assert operands.fc2_a.shape == (256, 512)
    assert operands.fc2_a.stride(1) == 1
    assert operands.fc2_b.shape == (512, 128)
    assert operands.fc2_b.stride(0) == 1
    assert operands.fc1_sfb.shape == (512, 16)
    assert operands.fc2_sfa.shape == (256, 16)
    assert operands.fc2_sfb.shape == (128, 16)

    preserved_fc1_b = operands.fc1_b.clone()
    preserved_fc2_a = operands.fc2_a.clone()
    preserved_fc2_b = operands.fc2_b.clone()
    aux.fc1_col_output.fill_(1)
    aux.fc1_recompute.fill_(1)
    aux.grad_y2.fill_(1)
    assert torch.equal(operands.fc1_b, preserved_fc1_b)
    assert torch.equal(operands.fc2_a, preserved_fc2_a)
    assert torch.equal(operands.fc2_b, preserved_fc2_b)


def _reference_wgrad_case(topk_idx, topk_weights):
    torch.manual_seed(20260821)
    token_count = topk_idx.shape[0]
    hidden = intermediate = 32
    activation = torch.randn(token_count, hidden) / 4
    fc1_weight = torch.randn(3, hidden, 2 * intermediate) / 8
    fc2_weight = torch.randn(3, intermediate, hidden) / 8
    grad_output = torch.randn(token_count, hidden) / 8
    reference = MoeEpReference(
        num_experts=3,
        hidden_size=hidden,
        intermediate_size=intermediate,
        top_k=topk_idx.shape[1],
        max_tokens_per_rank=token_count,
        generate_c=True,
        backward_wgrad_mode="operands",
        token_padding_size=256,
    )
    _, fc1_c, route_metadata, forward_stash = reference(
        activation,
        fc1_weight,
        fc2_weight,
        topk_idx,
        topk_weights,
    )
    _, _, operands = reference.backward(
        grad_output,
        fc1_weight,
        fc2_weight,
        topk_idx,
        topk_weights,
        fc1_c,
        route_metadata,
        wgrad_forward_stash=forward_stash,
    )
    return (
        activation,
        fc2_weight,
        grad_output,
        fc1_c,
        route_metadata,
        operands,
    )


@pytest.mark.L0
def test_reference_wgrad_operands_follow_route_weights_and_dense_formulas():
    topk_idx = torch.tensor(
        [[0, 2], [2, 0], [0, 2]],
        dtype=torch.int32,
    )
    topk_weights = torch.tensor(
        [[0.5, 0.25], [0.0, 0.75], [1.0, 0.125]],
        dtype=torch.float32,
    )
    (
        activation,
        fc2_weight,
        grad_output,
        fc1_c,
        route_metadata,
        operands,
    ) = _reference_wgrad_case(topk_idx, topk_weights)

    assert isinstance(operands, WgradOperandsReference)
    assert operands.expert_offsets.tolist() == [256, 256, 512]
    assert operands.valid_route_counts.tolist() == [3, 0, 3]
    assert operands.fc1_a.shape == (32, 512)
    assert operands.fc1_b.shape == (512, 64)
    assert operands.fc2_a.shape == (32, 512)
    assert operands.fc2_b.shape == (512, 32)

    # Rebuild the staged values independently in compact metadata order.
    staged_x = quantize_blockwise(
        activation,
        MoeFormat.MXFP8,
        axis=1,
    ).dequantize()
    staged_dy = quantize_blockwise(
        grad_output,
        MoeFormat.MXFP8,
        axis=1,
    ).dequantize()
    staged_w2 = quantize_blockwise(
        fc2_weight.transpose(1, 2),
        MoeFormat.MXFP8,
        axis=1,
    ).dequantize().transpose(1, 2)
    compact_x = []
    compact_weighted_h = []
    compact_dy = []
    compact_dc = []
    for row, (expert, _, token, slot) in enumerate(route_metadata.tolist()):
        c_gate, c_up = fc1_c[row].float().split(32)
        sigmoid = torch.sigmoid(c_gate)
        silu = c_gate * sigmoid
        h = silu * c_up
        p = topk_weights[token, slot]
        dy = staged_dy[token]
        dh = (dy @ staged_w2[expert].transpose(0, 1)) * p
        dc = torch.cat(
            (
                dh * c_up * sigmoid * (1 + c_gate * (1 - sigmoid)),
                dh * silu,
            )
        )
        compact_x.append(staged_x[token])
        compact_weighted_h.append(p * h)
        compact_dy.append(dy)
        compact_dc.append(dc)

    def padded(rows):
        result = torch.zeros(512, rows[0].numel())
        result[:3] = torch.stack(rows[:3])
        result[256:259] = torch.stack(rows[3:])
        return result

    expected_x = quantize_blockwise(
        padded(compact_x).transpose(0, 1),
        MoeFormat.MXFP8,
        axis=1,
    )
    expected_h = quantize_blockwise(
        padded(compact_weighted_h).transpose(0, 1),
        MoeFormat.MXFP8,
        axis=1,
    )
    expected_pdy = quantize_blockwise(
        padded(compact_dy),
        MoeFormat.MXFP8,
        axis=0,
    )
    expected_dc = quantize_blockwise(
        padded(compact_dc),
        MoeFormat.MXFP8,
        axis=0,
    )
    for actual, expected in (
        (operands.fc1_a, expected_x),
        (operands.fc1_b, expected_dc),
        (operands.fc2_a, expected_h),
        (operands.fc2_b, expected_pdy),
    ):
        assert torch.equal(actual.data, expected.data)
        assert torch.equal(actual.scale, expected.scale)

    # The valid route with p=0 keeps x/dY but contributes zero to both wgrads.
    zero_weight_row = 257
    assert operands.fc1_a.dequantize()[:, zero_weight_row].abs().sum() > 0
    assert operands.fc2_a.dequantize()[:, zero_weight_row].eq(0).all()
    assert operands.fc1_b.dequantize()[zero_weight_row].eq(0).all()
    assert operands.fc2_b.dequantize()[zero_weight_row].abs().sum() > 0

    dw1, dw2 = operands.dense_wgrads()
    a1, b1 = operands.fc1_a.dequantize(), operands.fc1_b.dequantize()
    a2, b2 = operands.fc2_a.dequantize(), operands.fc2_b.dequantize()
    torch.testing.assert_close(dw1[0], a1[:, :256] @ b1[:256])
    torch.testing.assert_close(dw1[1], torch.zeros_like(dw1[1]))
    torch.testing.assert_close(dw1[2], a1[:, 256:] @ b1[256:])
    torch.testing.assert_close(dw2[0], a2[:, :256] @ b2[:256])
    torch.testing.assert_close(dw2[1], torch.zeros_like(dw2[1]))
    torch.testing.assert_close(dw2[2], a2[:, 256:] @ b2[256:])
    decoded_dw1, decoded_dw2 = _dense_wgrads_from_operands(
        _as_production_operands(operands)
    )
    torch.testing.assert_close(decoded_dw1, dw1)
    torch.testing.assert_close(decoded_dw2, dw2)


@pytest.mark.L0
def test_reference_wgrad_empty_routes_padding_offsets_and_invalid_routes():
    topk_idx = torch.full((2, 2), -1, dtype=torch.int32)
    topk_weights = torch.randn(2, 2)
    *_, fc1_c, route_metadata, operands = _reference_wgrad_case(
        topk_idx,
        topk_weights,
    )

    assert fc1_c.shape == (0, 64)
    assert route_metadata.shape == (0, 4)
    assert operands.expert_offsets.tolist() == [0, 0, 0]
    assert operands.valid_route_counts.tolist() == [0, 0, 0]
    assert operands.fc1_a.shape == (32, 0)
    assert operands.fc1_b.shape == (0, 64)
    assert operands.fc2_a.shape == (32, 0)
    assert operands.fc2_b.shape == (0, 32)
    for dense in operands.dense_wgrads():
        assert dense.eq(0).all()

    invalid = torch.tensor([[0, -2], [3, -1]], dtype=torch.int32)
    with pytest.raises(ValueError, match="out-of-range expert id"):
        _reference_wgrad_case(invalid, topk_weights)


def _assemble_reference_scale(
    tensor,
    expert_offsets: torch.Tensor,
) -> torch.Tensor:
    k_axis = tensor.axis
    scale = tensor.scale
    parts = []
    begin = 0
    for end_tensor in expert_offsets:
        end = int(end_tensor.item())
        if k_axis == 1:
            raw = scale[:, begin // 32 : end // 32]
        else:
            raw = scale[begin // 32 : end // 32].transpose(0, 1)
        parts.append(_blocked_reference(raw.view(torch.uint8)))
        begin = end
    non_k = tensor.shape[1 - k_axis]
    rounded_non_k = (non_k + 127) // 128 * 128
    scale_columns = (tensor.shape[k_axis] // 32 + 3) // 4 * 4
    if parts:
        packed = torch.cat(parts)
    else:
        packed = torch.empty(0, dtype=torch.uint8, device=tensor.device)
    return packed.reshape(rounded_non_k, scale_columns).view(
        torch.float8_e8m0fnu
    )


def _as_production_operands(reference: WgradOperandsReference):
    return MoeEpWgradOperands(
        fc1_a=reference.fc1_a.data,
        fc1_sfa=_assemble_reference_scale(
            reference.fc1_a,
            reference.expert_offsets,
        ),
        fc1_b=reference.fc1_b.data.transpose(0, 1).contiguous().transpose(0, 1),
        fc1_sfb=_assemble_reference_scale(
            reference.fc1_b,
            reference.expert_offsets,
        ),
        fc2_a=reference.fc2_a.data,
        fc2_sfa=_assemble_reference_scale(
            reference.fc2_a,
            reference.expert_offsets,
        ),
        fc2_b=reference.fc2_b.data.transpose(0, 1).contiguous().transpose(0, 1),
        fc2_sfb=_assemble_reference_scale(
            reference.fc2_b,
            reference.expert_offsets,
        ),
        expert_offsets=reference.expert_offsets,
        valid_route_counts=reference.valid_route_counts,
        route_metadata=reference.route_metadata,
    )


@pytest.mark.L1
@pytest.mark.gpu_exclusive
def test_returned_operand_abi_runs_direct_grouped_wgrad():
    if not torch.cuda.is_available():
        pytest.skip("grouped wgrad integration requires CUDA")
    device = torch.device("cuda", 0)
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) != (10, 0):
        pytest.skip("the in-tree grouped wgrad integration kernel targets SM100")

    import cudnn

    generator = torch.Generator(device=device).manual_seed(20260821)
    hidden, intermediate = 128, 256
    activation = torch.randn(
        3,
        hidden,
        generator=generator,
        device=device,
    ) / 8
    fc1_weight = torch.randn(
        2,
        hidden,
        2 * intermediate,
        generator=generator,
        device=device,
    ) / 8
    fc2_weight = torch.randn(
        2,
        intermediate,
        hidden,
        generator=generator,
        device=device,
    ) / 8
    topk_idx = torch.tensor(
        [[0, 1], [1, 0], [0, 1]],
        dtype=torch.int32,
        device=device,
    )
    topk_weights = torch.tensor(
        [[0.5, 0.25], [0.75, 0.5], [1.0, 0.125]],
        device=device,
    )
    grad_output = torch.randn(
        3,
        hidden,
        generator=generator,
        device=device,
    ) / 8
    reference = MoeEpReference(
        num_experts=2,
        hidden_size=hidden,
        intermediate_size=intermediate,
        top_k=2,
        max_tokens_per_rank=3,
        generate_c=True,
        backward_wgrad_mode="operands",
        token_padding_size=256,
    )
    _, fc1_c, metadata, forward_stash = reference(
        activation,
        fc1_weight,
        fc2_weight,
        topk_idx,
        topk_weights,
    )
    _, _, logical = reference.backward(
        grad_output,
        fc1_weight,
        fc2_weight,
        topk_idx,
        topk_weights,
        fc1_c,
        metadata,
        wgrad_forward_stash=forward_stash,
    )
    operands = _as_production_operands(logical)

    common = {
        "offsets_tensor": operands.expert_offsets,
        "output_mode": "dense",
        "wgrad_dtype": torch.bfloat16,
        "acc_dtype": torch.float32,
        "mma_tiler_mn": (128, 128),
        "cluster_shape_mn": (1, 1),
        "sf_vec_size": 32,
    }
    fc1 = cudnn.grouped_gemm_wgrad_wrapper_sm100(
        a_tensor=operands.fc1_a,
        b_tensor=operands.fc1_b,
        sfa_tensor=operands.fc1_sfa,
        sfb_tensor=operands.fc1_sfb,
        **common,
    )["wgrad_tensor"]
    fc2 = cudnn.grouped_gemm_wgrad_wrapper_sm100(
        a_tensor=operands.fc2_a,
        b_tensor=operands.fc2_b,
        sfa_tensor=operands.fc2_sfa,
        sfb_tensor=operands.fc2_sfb,
        **common,
    )["wgrad_tensor"]
    torch.cuda.synchronize(device)

    expected_fc1, expected_fc2 = logical.dense_wgrads()
    torch.testing.assert_close(
        fc1.float(),
        expected_fc1,
        rtol=0.15,
        atol=0.125,
    )
    torch.testing.assert_close(
        fc2.float(),
        expected_fc2,
        rtol=0.15,
        atol=0.125,
    )


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("world_size", [1, 2, 4], ids=["ep1", "ep2", "ep4"])
def test_production_wgrad_operands_run_end_to_end(world_size, tmp_path):
    if world_size == 1:
        _run_wgrad_operand_case(
            device=_sm107_device(),
            ep_group=None,
            rank=0,
            world_size=1,
        )
        return

    _require_distributed_sm107(world_size)
    os.environ.setdefault("NVIDIA_IMEX_CHANNELS", "0")
    init_file = tmp_path / f"mxfp8_wgrad_operands_ep{world_size}.init"
    mp.spawn(
        _distributed_wgrad_worker,
        args=(world_size, str(init_file)),
        nprocs=world_size,
        join=True,
    )
