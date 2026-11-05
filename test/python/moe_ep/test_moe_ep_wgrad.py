# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoeEp.training_wgrad: the Rubin WGrad kernel over the producer-native operand ABI.

Host-only cases pin the contract (tile/input-order selection, zero-copy views,
validation, the DSL gate). GPU cases build synthetic operands directly in the
layouts ``prepare_training`` returns, so they compile only the WGrad kernel.
The end-to-end path through real producers is covered by the training tests in
``test_moe_ep_backward.py``.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from determinism_utils import assert_bitwise, assert_replays, snapshot

from cudnn.moe_ep import (
    MoeEp,
    MoeEpTrainingDiscreteWgradOutputs,
    MoeEpTrainingWgradOperands,
    MoeEpTrainingWgradOutputs,
    MoeEpWgradLayout,
)
from cudnn.moe_ep._megamoe_backend._workspace import BufferRegion, WorkspaceRequirements, WorkspaceViews
from cudnn.moe_ep._megamoe_backend.mxfp8 import _cutedsl
from cudnn.moe_ep._megamoe_backend.mxfp8 import _wgrad_compile
from cudnn.moe_ep._megamoe_backend.mxfp8._training_resources import (
    _build_training_abi_facts,
    _Mxfp8TrainingState,
)
from cudnn.moe_ep._megamoe_backend.mxfp8._wgrad_compile import (
    Mxfp8WgradGemm,
    PreparedMxfp8Wgrad,
    discrete_output_view,
    scale_view,
    select_wgrad_tile,
    wgrad_input_order,
)
from cudnn.moe_ep._validation import (
    validate_training_discrete_wgrad_outputs,
    validate_training_wgrad_operands,
    validate_training_wgrad_outputs,
)
from moe_ep.moe_ep_test_support import (
    _GROUPED_WGRAD_CLOSE_KWARGS,
    _dense_wgrads_from_operands,
    _require_training_wgrad,
    _round_up,
    _run_grouped_wgrad_kernel,
    _sm107_device,
    _training_abi_prepared,
    _training_config,
    _training_device,
    _training_wgrad_supported,
    _training_prepared_pair,
    _unpack_wgrad_scale_part_bytes,
)

_OPERAND_NAMES = (
    "fc1_a",
    "fc1_sfa",
    "fc1_b",
    "fc1_sfb",
    "fc2_a",
    "fc2_sfa",
    "fc2_b",
    "fc2_sfb",
    "valid_route_counts",
    "expert_offsets",
)


# ---------------------------------------------------------------------------
# Host-only contract


@pytest.mark.L0
@pytest.mark.parametrize(
    ("m", "n", "tile", "cluster"),
    (
        (7168, 4096, (512, 256, 128), (2, 1)),
        (2048, 7168, (512, 256, 128), (2, 1)),
        (256, 7168, (256, 256, 128), (2, 1)),
        (768, 512, (256, 256, 128), (2, 1)),
        (256, 640, (256, 128, 128), (2, 1)),
        (640, 512, (128, 128, 128), (1, 1)),
        (128, 512, (128, 128, 128), (1, 1)),
    ),
)
def test_wgrad_tile_divides_problem_and_pins_k_to_padding(m, n, tile, cluster):
    assert select_wgrad_tile(m, n) == (tile, cluster)
    assert tile[2] == _wgrad_compile.WGRAD_TILE_K == 128


@pytest.mark.L0
def test_wgrad_tile_selection_declines_partial_tiles():
    assert select_wgrad_tile(64, 512) is None
    assert select_wgrad_tile(128, 64) is None


@pytest.mark.L0
def test_wgrad_input_order_routes_global_pitch_k_major_to_tensor2d():
    assert wgrad_input_order(False, False) == "tensor_ragged"
    assert wgrad_input_order(False, True) == "tensor2d"
    assert wgrad_input_order(True, False) == "tensor2d"


@pytest.mark.L0
def test_wgrad_scale_view_is_zero_copy():
    exact = torch.zeros((256, 8), dtype=torch.uint8).view(torch.float8_e8m0fnu)
    assert scale_view(exact, 200, 256) is exact

    carrier = torch.zeros((256, 12), dtype=torch.uint8).view(torch.float8_e8m0fnu)
    view = scale_view(carrier, 256, 256)
    assert tuple(view.shape) == (256, 8)
    assert view.is_contiguous()
    assert view.data_ptr() == carrier.data_ptr()
    assert view.untyped_storage().data_ptr() == carrier.untyped_storage().data_ptr()

    with pytest.raises(ValueError, match="elements"):
        scale_view(torch.zeros((128, 4), dtype=torch.uint8).view(torch.float8_e8m0fnu), 256, 256)
    with pytest.raises(ValueError, match="contiguous"):
        scale_view(torch.zeros((16, 256), dtype=torch.uint8).view(torch.float8_e8m0fnu).T, 256, 256)


@pytest.mark.L0
def test_wgrad_discrete_output_view_aliases_pointer_table():
    table = torch.arange(4, dtype=torch.int64)
    for dtype in (torch.bfloat16, torch.float32):
        view = discrete_output_view(table, dtype, (4, 512, 256))
        assert view.dtype is dtype
        assert tuple(view.shape) == (4, 512, 256)
        assert tuple(view.stride()) == (0, 0, 0)
        assert view.data_ptr() == table.data_ptr()


@pytest.mark.L0
def test_dense_wgrad_outputs_resolve_both_layouts():
    config = _training_config()
    experts = config.topology.experts_per_rank
    gemm_fc1 = torch.empty((experts, 128, 512), dtype=torch.bfloat16)
    weight_fc2 = torch.empty((experts, 128, 256), dtype=torch.float32).transpose(1, 2)
    resolved = validate_training_wgrad_outputs(
        config,
        MoeEpTrainingWgradOutputs(fc1=gemm_fc1, fc2=weight_fc2),
        device=torch.device("cpu"),
    )
    assert resolved == {
        "fc1": (torch.bfloat16, MoeEpWgradLayout.GEMM_ROW_MAJOR),
        "fc2": (torch.float32, MoeEpWgradLayout.WEIGHT_ROW_MAJOR),
    }


@pytest.mark.L0
@pytest.mark.parametrize(
    ("fc1", "message"),
    (
        (lambda e: torch.empty((e, 512, 128), dtype=torch.bfloat16), r"out\.fc1 shape"),
        (lambda e: torch.empty((e, 128, 512), dtype=torch.float16), r"out\.fc1 dtype"),
        (lambda e: torch.empty((e, 128, 1024), dtype=torch.bfloat16)[:, :, ::2], r"out\.fc1 stride"),
    ),
)
def test_dense_wgrad_outputs_reject_unbindable_tensors(fc1, message):
    config = _training_config()
    experts = config.topology.experts_per_rank
    out = MoeEpTrainingWgradOutputs(
        fc1=fc1(experts),
        fc2=torch.empty((experts, 256, 128), dtype=torch.bfloat16),
    )
    with pytest.raises(ValueError, match=message):
        validate_training_wgrad_outputs(config, out, device=torch.device("cpu"))


@pytest.mark.L0
def test_discrete_wgrad_outputs_validate_tables_and_layout():
    config = _training_config()
    experts = config.topology.experts_per_rank
    table = torch.zeros((experts,), dtype=torch.int64)
    out = MoeEpTrainingDiscreteWgradOutputs(fc1_ptrs=table, fc2_ptrs=table.clone(), dtype=torch.float32)
    assert out.layout is MoeEpWgradLayout.WEIGHT_ROW_MAJOR
    assert MoeEpTrainingDiscreteWgradOutputs(table, table, torch.bfloat16, "gemm_row_major").layout is MoeEpWgradLayout.GEMM_ROW_MAJOR
    with pytest.raises(ValueError, match="unsupported WGrad layout"):
        MoeEpTrainingDiscreteWgradOutputs(table, table, torch.bfloat16, "column")
    with pytest.raises(ValueError, match="must be a CUDA tensor"):
        validate_training_discrete_wgrad_outputs(config, out, validate_pointees=False)
    with pytest.raises(ValueError, match=r"out\.dtype"):
        validate_training_discrete_wgrad_outputs(config, replace(out, dtype=torch.float16), validate_pointees=False)
    with pytest.raises(ValueError, match=r"out\.fc1_ptrs shape"):
        validate_training_discrete_wgrad_outputs(
            config,
            replace(out, fc1_ptrs=torch.zeros((experts + 1,), dtype=torch.int64)),
            validate_pointees=False,
        )


@pytest.mark.L0
def test_wgrad_operand_validation_names_the_operand():
    requirement = ((4,), (1,), torch.int32, 1)
    requirements = {name: requirement for name in _OPERAND_NAMES}
    operands = MoeEpTrainingWgradOperands(**{name: torch.empty(4, dtype=torch.int32) for name in _OPERAND_NAMES})
    validate_training_wgrad_operands(operands, requirements, device=torch.device("cpu"))
    bad = replace(operands, fc2_b=torch.empty(8, dtype=torch.int32)[::2])
    with pytest.raises(ValueError, match=r"operands\.fc2_b stride"):
        validate_training_wgrad_operands(bad, requirements, device=torch.device("cpu"))
    with pytest.raises(TypeError, match="MoeEpTrainingWgradOperands"):
        validate_training_wgrad_operands(object(), requirements, device=torch.device("cpu"))


@pytest.mark.L0
def test_wgrad_declines_architectures_without_a_wgrad_kernel():
    from cudnn.moe_ep._megamoe_backend.mxfp8._wgrad_compile import wgrad_unavailable_reason

    # Decided before any DSL probe, so the reason holds on every installation.
    reason = wgrad_unavailable_reason((10, 0))
    assert reason is not None and "not supported on Blackwell SM100" in reason
    with pytest.raises(NotImplementedError, match="compute capability 10.3"):
        wgrad_unavailable_reason((10, 3))


@pytest.mark.L0
def test_training_wgrad_declines_on_device_without_a_wgrad_kernel():
    device = _training_device()
    if _training_wgrad_supported(device):
        pytest.skip("this training architecture has a WGrad kernel")
    capacity = 256
    with MoeEp(
        _training_config(
            num_experts=_EXPERTS,
            hidden_size=_HIDDEN,
            intermediate_size=_INTERMEDIATE,
            top_k=2,
            max_tokens_per_rank=capacity,
            physical_recv_pool_rows=_round_up(capacity * 2 + 128 * _EXPERTS, 128),
            combine_format="bf16",
        ).public_config
    ) as op:
        # Forward and backward still prepare; only WGrad is declined, before any operand check.
        op.prepare_training(device=device)
        with pytest.raises(RuntimeError, match="WGrad is not supported on Blackwell SM100"):
            op.training_wgrad(None, out=None)


@pytest.mark.L0
def test_wgrad_gate_probes_the_tma_residue_op(monkeypatch):
    pytest.importorskip("cutlass")
    try:
        _cutedsl.require_megamoe_cutedsl()
    except RuntimeError as error:
        pytest.skip(str(error))
    import cutlass._mlir.dialects.cute_nvgpu as cute_nvgpu_ir

    monkeypatch.delattr(cute_nvgpu_ir, "atom_make_tma_residue_tensor", raising=False)
    reason = _cutedsl.rubin_wgrad_cutedsl_error()
    assert reason is not None and "atom_make_tma_residue_tensor" in reason and "nvidia-cutlass-dsl>=4.8.0" in reason
    with pytest.raises(RuntimeError, match="atom_make_tma_residue_tensor"):
        _cutedsl.require_rubin_wgrad_cutedsl()

    monkeypatch.setattr(cute_nvgpu_ir, "atom_make_tma_residue_tensor", object(), raising=False)
    assert _cutedsl.rubin_wgrad_cutedsl_error() is None


@pytest.mark.L0
def test_wgrad_gate_names_the_required_dsl_when_none_is_installed(monkeypatch):
    monkeypatch.setattr(_cutedsl, "_public_cutedsl_version", lambda: None)
    monkeypatch.setattr(_cutedsl, "_cutlass_installed", lambda: False)
    reason = _cutedsl.rubin_wgrad_cutedsl_error()
    assert reason is not None and "nvidia-cutlass-dsl>=4.8.0" in reason and "no CUTLASS DSL is installed" in reason


@pytest.mark.L0
def test_wgrad_gate_reports_an_old_public_dsl(monkeypatch):
    monkeypatch.setattr(_cutedsl, "_public_cutedsl_version", lambda: "4.7.1")
    reason = _cutedsl.rubin_wgrad_cutedsl_error()
    assert reason is not None and "4.7.1" in reason


@pytest.mark.L0
def test_wgrad_preparation_requires_128_padding(monkeypatch):
    monkeypatch.setattr(_wgrad_compile, "require_rubin_wgrad_cutedsl", lambda: None)
    config = _training_config()
    with pytest.raises(ValueError, match="token and SF padding 128"):
        _wgrad_compile.prepare_wgrad(
            config,
            pool_rows=512,
            token_padding=256,
            sf_padding=128,
            device=torch.device("cpu"),
            architecture=(10, 7),
        )
    with pytest.raises(ValueError, match="multiple of the token padding"):
        _wgrad_compile.prepare_wgrad(
            config,
            pool_rows=500,
            token_padding=128,
            sf_padding=128,
            device=torch.device("cpu"),
            architecture=(10, 7),
        )


@pytest.mark.L0
def test_training_wgrad_reports_why_it_is_unavailable():
    op = MoeEp(_training_config().public_config)
    op._training_state = SimpleNamespace(wgrad_prepared=None, wgrad_unavailable_reason="no Rubin TMA overrides here")
    op._training_requirements = {}
    state = op._execution_state
    op._execution_state = replace(
        state,
        backend=SimpleNamespace(resolved_config=state.resolved_config, device=torch.device("cuda", 0)),
    )
    with pytest.raises(RuntimeError, match="no Rubin TMA overrides here"):
        op.training_wgrad(None, out=None)
    with pytest.raises(TypeError, match="accumulate must be a bool"):
        op.training_wgrad(None, out=None, accumulate=1)


def _fake_wgrad(pool_rows: int) -> PreparedMxfp8Wgrad:
    gemm = Mxfp8WgradGemm("fc1", 128, 512, False, False, "tensor_ragged", (128, 128, 128), (1, 1))
    return PreparedMxfp8Wgrad(
        device=torch.device("cpu"),
        architecture=(10, 7),
        experts=2,
        pool_rows=pool_rows,
        token_padding=128,
        fc1=gemm,
        fc2=replace(gemm, name="fc2", m=256, n=128, b_k_major=True, input_order="tensor2d"),
        local_workspace_bytes=256,
        local_workspace_zero_bytes=0,
    )


@pytest.mark.L0
def test_training_workspace_reserves_the_wgrad_region_only_when_prepared():
    config = _training_config()
    forward, backward = _training_prepared_pair(config)
    without = _Mxfp8TrainingState(config, torch.device("cpu"), forward, backward)
    assert without.wgrad_prepared is None
    assert "not prepared" in without.wgrad_unavailable_reason
    assert not any(region.name.startswith("wgrad.") for region in without.requirements.local_regions)

    state = _Mxfp8TrainingState(
        config,
        torch.device("cpu"),
        forward,
        backward,
        wgrad=_fake_wgrad(backward.pool_token_capacity),
    )
    regions = {region.name: region for region in state.requirements.local_regions}
    region = regions["wgrad.local.kernel_local_workspace"]
    assert (region.nbytes, region.alignment) == (256, 128)
    flat = WorkspaceViews(
        token_count=0,
        symmetric={region.name: torch.empty(region.nbytes, dtype=torch.uint8) for region in state.requirements.symmetric_regions},
        local={region.name: torch.empty(region.nbytes, dtype=torch.uint8) for region in state.requirements.local_regions},
        peer_mapping=object(),
    )
    assert flat.local["wgrad.local.kernel_local_workspace"].numel() == 256

    with pytest.raises(ValueError, match="pool rows must match"):
        _Mxfp8TrainingState(
            config,
            torch.device("cpu"),
            forward,
            backward,
            wgrad=_fake_wgrad(backward.pool_token_capacity + 128),
        )


@pytest.mark.L0
def test_training_abi_fingerprint_covers_the_wgrad_plan():
    config = _training_config(ep_size=2, ep_global_ranks=(0, 1))
    forward = _training_abi_prepared("forward")
    backward = _training_abi_prepared("backward")
    requirements = WorkspaceRequirements(
        max_tokens_per_rank=4,
        symmetric_regions=(BufferRegion("symmetric", 256),),
        local_regions=(BufferRegion("local", 128),),
    )
    facts = {
        name: _build_training_abi_facts(config, forward, backward, requirements, source_tree_digest="source", wgrad=wgrad)
        for name, wgrad in (("none", None), ("planned", _fake_wgrad(128)))
    }
    assert facts["none"]["schema_version"] == 7
    assert facts["none"]["wgrad_kernel"] is None
    planned = facts["planned"]["wgrad_kernel"]
    assert planned["fc2"]["input_order"] == "tensor2d"
    assert planned["fc1"]["mma_tiler_mnk"] == [128, 128, 128]
    assert facts["none"] != facts["planned"]


# ---------------------------------------------------------------------------
# GPU: synthetic operands in the producer-native layouts


_E8M0_ONE = 127
_E4M3_NAN = 0x7F
_E8M0_NAN = 0xFF


def _pack_wgrad_scale_part(raw: torch.Tensor) -> torch.Tensor:
    """Inverse of ``_unpack_wgrad_scale_part_bytes`` for a padded ``(R, C)`` byte block."""

    rows, columns = raw.shape
    assert rows % 128 == 0 and columns % 4 == 0
    row_atoms, column_atoms = rows // 128, columns // 4
    return raw.reshape(row_atoms, 128, column_atoms, 4).permute(0, 2, 1, 3).reshape(row_atoms * column_atoms, 4, 32, 4).transpose(1, 2).reshape(-1).contiguous()


def _expert_payloads(counts, dims, *, seed, device):
    """Per-expert valid rows and raw scales, independent of placement."""

    generator = torch.Generator(device=device).manual_seed(seed)
    payloads = []
    for count in counts:
        payload = {}
        for name, columns in dims.items():
            values = torch.randn((count, columns), generator=generator, device=device).clamp_(-6, 6)
            payload[name] = values.to(torch.float8_e4m3fn).view(torch.uint8)
            blocks = (count + 31) // 32
            payload[f"{name}_scale"] = torch.randint(
                124,
                131,
                (columns, blocks),
                generator=generator,
                device=device,
                dtype=torch.int32,
            ).to(torch.uint8)
        payloads.append(payload)
    return payloads


def _wgrad_dims(requirements):
    return {
        "fc1_a": requirements["fc1_a"][0][1],
        "fc1_b": requirements["fc1_b"][0][1],
        "fc2_a": requirements["fc2_a"][0][1],
        "fc2_b": requirements["fc2_b"][0][1],
    }


def _place_operands(requirements, payloads, device) -> MoeEpTrainingWgradOperands:
    """Lay payloads out in the producer ABI, poisoning padding scales' neighbors and the tail."""

    tensors = {}
    for name in _OPERAND_NAMES:
        shape, stride, dtype, _ = requirements[name]
        tensors[name] = torch.empty_strided(shape, stride, dtype=dtype, device=device)
    counts = [int(payload["fc1_a"].shape[0]) for payload in payloads]
    extents = [_round_up(count, 128) for count in counts]
    ends = torch.tensor(extents, dtype=torch.int64).cumsum(0)
    pool_rows = int(requirements["fc1_a"][0][0])
    assert int(ends[-1]) <= pool_rows
    tensors["valid_route_counts"].copy_(torch.tensor(counts, dtype=torch.int32))
    tensors["expert_offsets"].copy_(ends.to(torch.int32))
    for name, columns in _wgrad_dims(requirements).items():
        data = tensors[name].view(torch.uint8)
        data.fill_(_E4M3_NAN)
        scales = tensors[f"{name[:3]}_sf{name[-1]}"].view(torch.uint8).reshape(-1)
        scales.fill_(_E8M0_NAN)
        padded_columns = _round_up(columns, 128)
        begin = 0
        scale_offset = 0
        for payload, count, extent in zip(payloads, counts, extents):
            data[begin : begin + extent].zero_()
            data[begin : begin + count].copy_(payload[name])
            raw = torch.full((padded_columns, extent // 32), _E8M0_ONE, dtype=torch.uint8, device=device)
            raw[:columns, : payload[f"{name}_scale"].shape[1]] = payload[f"{name}_scale"]
            packed = _pack_wgrad_scale_part(raw)
            assert torch.equal(_unpack_wgrad_scale_part_bytes(packed, padded_columns, extent // 32), raw)
            scales[scale_offset : scale_offset + packed.numel()].copy_(packed)
            scale_offset += packed.numel()
            begin += extent
    return MoeEpTrainingWgradOperands(**tensors)


_HIDDEN = 512
_INTERMEDIATE = 256
_EXPERTS = 4


@pytest.fixture(scope="module")
def wgrad_op():
    device = _sm107_device()
    _require_training_wgrad(device)
    capacity = 512
    top_k = 2
    with MoeEp(
        _training_config(
            num_experts=_EXPERTS,
            hidden_size=_HIDDEN,
            intermediate_size=_INTERMEDIATE,
            top_k=top_k,
            max_tokens_per_rank=capacity,
            physical_recv_pool_rows=_round_up(capacity * top_k + 128 * _EXPERTS, 128),
            combine_format="bf16",
        ).public_config
    ) as op:
        requirements = op.prepare_training(device=device)
        yield op, requirements, device


def _reference(operands):
    return tuple(value.float() for value in _dense_wgrads_from_operands(operands))


def _assert_close(actual, expected, name):
    torch.testing.assert_close(actual.float(), expected, msg=lambda m: f"{name}: {m}", **_GROUPED_WGRAD_CLOSE_KWARGS)


def _dense_destination(shape, dtype, layout, device, fill):
    experts, m, n = shape
    if layout is MoeEpWgradLayout.GEMM_ROW_MAJOR:
        storage = torch.empty((experts, m, n), dtype=dtype, device=device)
        view = storage
    else:
        storage = torch.empty((experts, n, m), dtype=dtype, device=device)
        view = storage.transpose(1, 2)
    view.copy_(fill)
    return view


def _discrete_destination(shape, dtype, layout, device, fill):
    experts, m, n = shape
    buffers = []
    for expert in range(experts):
        if layout is MoeEpWgradLayout.GEMM_ROW_MAJOR:
            buffer = fill[expert].to(dtype).contiguous()
            buffers.append((buffer, buffer))
        else:
            buffer = fill[expert].to(dtype).transpose(0, 1).contiguous()
            buffers.append((buffer, buffer.transpose(0, 1)))
    table = torch.tensor([owner.data_ptr() for owner, _ in buffers], dtype=torch.int64, device=device)
    return table, buffers


_VARIANTS = (
    ("dense", torch.bfloat16, MoeEpWgradLayout.GEMM_ROW_MAJOR, False),
    ("dense", torch.float32, MoeEpWgradLayout.WEIGHT_ROW_MAJOR, False),
    ("dense", torch.float32, MoeEpWgradLayout.GEMM_ROW_MAJOR, True),
    ("discrete", torch.bfloat16, MoeEpWgradLayout.WEIGHT_ROW_MAJOR, False),
    ("discrete", torch.float32, MoeEpWgradLayout.WEIGHT_ROW_MAJOR, True),
    ("discrete", torch.bfloat16, MoeEpWgradLayout.GEMM_ROW_MAJOR, True),
)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize(
    ("storage", "dtype", "layout", "accumulate"),
    _VARIANTS,
    ids=[f"{s}-{str(d).split('.')[-1]}-{l.value}-{'beta1' if a else 'beta0'}" for s, d, l, a in _VARIANTS],
)
def test_training_wgrad_matches_reference_for_every_output_form(wgrad_op, storage, dtype, layout, accumulate):
    op, requirements, device = wgrad_op
    counts = (300, 0, 128, 1)
    operands = _place_operands(requirements, _expert_payloads(counts, _wgrad_dims(requirements), seed=11, device=device), device)
    expected = _reference(operands)
    generator = torch.Generator(device=device).manual_seed(5)
    initial = tuple(torch.randn(value.shape, generator=generator, device=device).to(dtype).float() for value in expected)
    if storage == "dense":
        fc1 = _dense_destination(expected[0].shape, dtype, layout, device, initial[0])
        fc2 = _dense_destination(expected[1].shape, dtype, layout, device, initial[1])
        out = MoeEpTrainingWgradOutputs(fc1=fc1, fc2=fc2)
        result = op.training_wgrad(operands, out=out, accumulate=accumulate)
        assert result is out
        actual = (fc1, fc2)
    else:
        fc1_ptrs, fc1_buffers = _discrete_destination(expected[0].shape, dtype, layout, device, initial[0])
        fc2_ptrs, fc2_buffers = _discrete_destination(expected[1].shape, dtype, layout, device, initial[1])
        out = MoeEpTrainingDiscreteWgradOutputs(fc1_ptrs=fc1_ptrs, fc2_ptrs=fc2_ptrs, dtype=dtype, layout=layout)
        op.training_wgrad(operands, out=out, accumulate=accumulate)
        actual = tuple(torch.stack([view for _, view in buffers]) for buffers in (fc1_buffers, fc2_buffers))
    torch.cuda.synchronize(device)
    for name, got, want, init in zip(("fc1", "fc2"), actual, expected, initial):
        target = want + init if accumulate else want
        assert torch.isfinite(got).all(), f"{name}: non-finite output"
        _assert_close(got, target, name)
        # Expert 1 has no tokens: beta = 0 writes zeros, beta = 1 leaves the destination.
        empty = init[1] if accumulate else torch.zeros_like(init[1])
        assert torch.equal(got[1].float(), empty), f"{name}: empty expert"


@pytest.mark.L1
@pytest.mark.gpu_exclusive
def test_training_wgrad_reads_operands_in_place_and_replays_in_a_graph(wgrad_op):
    from cuda.bindings import driver as cuda_driver

    op, requirements, device = wgrad_op
    dims = _wgrad_dims(requirements)
    operands = _place_operands(requirements, _expert_payloads((257, 64, 0, 129), dims, seed=21, device=device), device)
    out = MoeEpTrainingWgradOutputs(
        fc1=torch.empty((_EXPERTS, _HIDDEN, 2 * _INTERMEDIATE), dtype=torch.bfloat16, device=device),
        fc2=torch.empty((_EXPERTS, _INTERMEDIATE, _HIDDEN), dtype=torch.bfloat16, device=device),
    )
    op.training_wgrad(operands, out=out)  # compile outside the measured region
    torch.cuda.synchronize(device)

    allocated = torch.cuda.memory_allocated(device)
    torch.cuda.set_sync_debug_mode("error")
    try:
        op.training_wgrad(operands, out=out)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize(device)
    assert torch.cuda.memory_allocated(device) == allocated
    for name, got, want in zip(("fc1", "fc2"), (out.fc1, out.fc2), _reference(operands)):
        _assert_close(got, want, name)

    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=stream):
        op.training_wgrad(operands, out=out)
    raw = cuda_driver.CUgraph(graph.raw_cuda_graph())
    error, _, count = cuda_driver.cuGraphGetNodes(raw, 0)
    assert error == cuda_driver.CUresult.CUDA_SUCCESS
    error, nodes, count = cuda_driver.cuGraphGetNodes(raw, count)
    kinds = [cuda_driver.cuGraphNodeGetType(node)[1].name for node in nodes[:count]]
    assert kinds == ["CU_GRAPH_NODE_TYPE_KERNEL"] * 2, kinds
    graph.instantiate()

    # Same buffers, new routing: the kernel reads counts and offsets on device.
    replacement = _place_operands(requirements, _expert_payloads((0, 384, 1, 200), dims, seed=22, device=device), device)
    for name in _OPERAND_NAMES:
        getattr(operands, name).copy_(getattr(replacement, name))
    graph.replay()
    torch.cuda.synchronize(device)
    for name, got, want in zip(("fc1", "fc2"), (out.fc1, out.fc2), _reference(operands)):
        _assert_close(got, want, name)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
def test_training_wgrad_rejects_mismatched_operands_and_aliasing(wgrad_op):
    op, requirements, device = wgrad_op
    operands = _place_operands(requirements, _expert_payloads((1, 1, 1, 1), _wgrad_dims(requirements), seed=3, device=device), device)
    good = MoeEpTrainingWgradOutputs(
        fc1=torch.empty((_EXPERTS, _HIDDEN, 2 * _INTERMEDIATE), dtype=torch.bfloat16, device=device),
        fc2=torch.empty((_EXPERTS, _INTERMEDIATE, _HIDDEN), dtype=torch.bfloat16, device=device),
    )
    contiguous_fc2_b = replace(operands, fc2_b=operands.fc2_b.contiguous())
    with pytest.raises(ValueError, match=r"operands\.fc2_b stride"):
        op.training_wgrad(contiguous_fc2_b, out=good)
    aliased = MoeEpTrainingWgradOutputs(
        fc1=good.fc1,
        fc2=good.fc1.view(-1)[: _EXPERTS * _INTERMEDIATE * _HIDDEN].view(_EXPERTS, _INTERMEDIATE, _HIDDEN),
    )
    with pytest.raises(ValueError, match="must not alias"):
        op.training_wgrad(operands, out=aliased)


# ---------------------------------------------------------------------------
# Determinism: replay and expert (batch) invariance


def _output_shapes(requirements, experts):
    """GEMM-oriented dW1 and dW2 shapes: ``(E, H, 2I)`` and ``(E, I, H)``."""

    dims = _wgrad_dims(requirements)
    return (experts, dims["fc1_a"], dims["fc1_b"]), (experts, dims["fc2_a"], dims["fc2_b"])


def _run_dense(op, operands, dtype, device, shapes=None):
    fc1_shape, fc2_shape = shapes or ((_EXPERTS, _HIDDEN, 2 * _INTERMEDIATE), (_EXPERTS, _INTERMEDIATE, _HIDDEN))
    out = MoeEpTrainingWgradOutputs(
        fc1=torch.empty(fc1_shape, dtype=dtype, device=device),
        fc2=torch.empty(fc2_shape, dtype=dtype, device=device),
    )
    op.training_wgrad(operands, out=out)
    return {"dw1": out.fc1, "dw2": out.fc2}


@pytest.mark.L0
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32), ids=("bf16", "fp32"))
def test_training_wgrad_replays_bitwise(wgrad_op, dtype):
    op, requirements, device = wgrad_op
    operands = _place_operands(requirements, _expert_payloads((300, 0, 128, 129), _wgrad_dims(requirements), seed=31, device=device), device)
    expected = assert_replays(lambda: _run_dense(op, operands, dtype, device))
    for name, want in zip(("dw1", "dw2"), _reference(operands)):
        _assert_close(expected[name], want, name)


@pytest.mark.L0
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32), ids=("bf16", "fp32"))
def test_training_wgrad_expert_results_do_not_depend_on_neighbors(wgrad_op, dtype):
    """An expert's bytes survive moving it, changing neighbors and duplicating it."""

    op, requirements, device = wgrad_op
    dims = _wgrad_dims(requirements)
    base = _expert_payloads((300, 45, 128, 129), dims, seed=41, device=device)
    others = _expert_payloads((7, 256, 1, 384), dims, seed=42, device=device)

    reference = snapshot(_run_dense(op, _place_operands(requirements, base, device), dtype, device))
    # Expert 1 alone at slot 1 after a different, longer expert 0; neighbors 2 and 3 replaced.
    neighbors = snapshot(_run_dense(op, _place_operands(requirements, [others[3], base[1], others[2], others[0]], device), dtype, device))
    # Reordered: base expert 1 moves to slot 3 and base expert 0 to slot 2.
    reordered = snapshot(_run_dense(op, _place_operands(requirements, [others[1], base[2], base[0], base[1]], device), dtype, device))
    # Duplicated: base expert 3 in every slot.
    duplicated = snapshot(_run_dense(op, _place_operands(requirements, [base[3]] * _EXPERTS, device), dtype, device))

    def pick(result, expert):
        return {name: value[expert] for name, value in result.items()}

    assert_bitwise(pick(neighbors, 1), pick(reference, 1), "expert 1 with new neighbors")
    assert_bitwise(pick(reordered, 3), pick(reference, 1), "expert 1 moved to slot 3")
    assert_bitwise(pick(reordered, 2), pick(reference, 0), "expert 0 moved to slot 2")
    assert_bitwise(pick(reordered, 1), pick(reference, 2), "expert 2 moved to slot 1")
    for slot in range(_EXPERTS):
        assert_bitwise(pick(duplicated, slot), pick(reference, 3), f"expert 3 duplicated into slot {slot}")


@pytest.mark.L0
@pytest.mark.gpu_exclusive
def test_training_wgrad_layouts_store_the_same_accumulator(wgrad_op):
    """GEMM- and weight-oriented FP32 outputs, dense and discrete, hold identical bytes."""

    op, requirements, device = wgrad_op
    operands = _place_operands(requirements, _expert_payloads((300, 45, 128, 129), _wgrad_dims(requirements), seed=51, device=device), device)
    shapes = ((_EXPERTS, _HIDDEN, 2 * _INTERMEDIATE), (_EXPERTS, _INTERMEDIATE, _HIDDEN))
    results = {}
    for layout in MoeEpWgradLayout:
        dense = tuple(_dense_destination(shape, torch.float32, layout, device, torch.zeros(shape, device=device)) for shape in shapes)
        op.training_wgrad(operands, out=MoeEpTrainingWgradOutputs(fc1=dense[0], fc2=dense[1]))
        results[f"dense-{layout.value}"] = dense
        tables = [_discrete_destination(shape, torch.float32, layout, device, torch.zeros(shape, device=device)) for shape in shapes]
        op.training_wgrad(
            operands,
            out=MoeEpTrainingDiscreteWgradOutputs(fc1_ptrs=tables[0][0], fc2_ptrs=tables[1][0], dtype=torch.float32, layout=layout),
        )
        results[f"discrete-{layout.value}"] = tuple(torch.stack([view for _, view in buffers]) for _, buffers in tables)
    torch.cuda.synchronize(device)
    first = next(iter(results.values()))
    for name, value in results.items():
        for gemm, got, want in zip(("dw1", "dw2"), value, first):
            assert torch.equal(got.contiguous().view(torch.int32), want.contiguous().view(torch.int32)), f"{name} {gemm}"


@pytest.mark.L0
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize(
    ("storage", "dtype", "layout", "accumulate"),
    (
        ("discrete", torch.bfloat16, MoeEpWgradLayout.WEIGHT_ROW_MAJOR, False),
        ("dense", torch.float32, MoeEpWgradLayout.GEMM_ROW_MAJOR, True),
        ("discrete", torch.float32, MoeEpWgradLayout.WEIGHT_ROW_MAJOR, True),
    ),
    ids=("discrete-bf16-beta0", "dense-fp32-beta1", "discrete-fp32-beta1"),
)
def test_training_wgrad_discrete_and_accumulate_replay_bitwise(wgrad_op, storage, dtype, layout, accumulate):
    """Discrete destinations, and accumulate=True from one restored starting value, replay to the same bytes."""

    op, requirements, device = wgrad_op
    operands = _place_operands(requirements, _expert_payloads((300, 0, 128, 129), _wgrad_dims(requirements), seed=61, device=device), device)
    shapes = _output_shapes(requirements, _EXPERTS)
    generator = torch.Generator(device=device).manual_seed(62)
    initial = tuple(torch.randn(shape, generator=generator, device=device).to(dtype).float() for shape in shapes)

    def run():
        if storage == "dense":
            fc1, fc2 = (_dense_destination(shape, dtype, layout, device, start) for shape, start in zip(shapes, initial))
            op.training_wgrad(operands, out=MoeEpTrainingWgradOutputs(fc1=fc1, fc2=fc2), accumulate=accumulate)
            return {"dw1": fc1, "dw2": fc2}
        tables = [_discrete_destination(shape, dtype, layout, device, start) for shape, start in zip(shapes, initial)]
        out = MoeEpTrainingDiscreteWgradOutputs(fc1_ptrs=tables[0][0], fc2_ptrs=tables[1][0], dtype=dtype, layout=layout)
        op.training_wgrad(operands, out=out, accumulate=accumulate)
        return {name: torch.stack([view for _, view in buffers]) for name, (_, buffers) in zip(("dw1", "dw2"), tables)}

    expected = assert_replays(run)
    for name, want, start in zip(("dw1", "dw2"), _reference(operands), initial):
        _assert_close(expected[name], want + start if accumulate else want, name)


# Multi-wave shape: dW1 is 2048 x 2048 and dW2 is 1024 x 2048 per expert, so each GEMM launches more 2-CTA clusters than
# one wave holds, and an expert's K loop runs up to 16 tiles of 128 rows, refilling the A/B pipeline stages inside a tile.
_LARGE_HIDDEN = 2048
_LARGE_INTERMEDIATE = 1024
_LARGE_EXPERTS = 8
# Full, partial, empty and single-row segments.
_LARGE_COUNTS = (2048, 1500, 1024, 0, 1, 1793, 640, 2047)


@pytest.fixture(scope="module")
def wgrad_op_large():
    device = _sm107_device()
    _require_training_wgrad()
    capacity = 8192
    top_k = 2
    with MoeEp(
        _training_config(
            num_experts=_LARGE_EXPERTS,
            hidden_size=_LARGE_HIDDEN,
            intermediate_size=_LARGE_INTERMEDIATE,
            top_k=top_k,
            max_tokens_per_rank=capacity,
            physical_recv_pool_rows=_round_up(capacity * top_k + 128 * _LARGE_EXPERTS, 128),
            combine_format="bf16",
        ).public_config
    ) as op:
        requirements = op.prepare_training(device=device)
        yield op, requirements, device


def _operand_bytes(operands):
    return {name: getattr(operands, name).clone(memory_format=torch.contiguous_format).view(torch.uint8) for name in _OPERAND_NAMES}


@pytest.mark.L0
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32), ids=("bf16", "fp32"))
def test_training_wgrad_replays_bitwise_across_waves(wgrad_op_large, dtype):
    op, requirements, device = wgrad_op_large
    shapes = _output_shapes(requirements, _LARGE_EXPERTS)
    operands = _place_operands(requirements, _expert_payloads(_LARGE_COUNTS, _wgrad_dims(requirements), seed=71, device=device), device)
    before = _operand_bytes(operands)
    expected = assert_replays(lambda: _run_dense(op, operands, dtype, device, shapes))
    after = _operand_bytes(operands)
    for name in _OPERAND_NAMES:
        assert torch.equal(after[name], before[name]), f"operands.{name} changed"
    for name, want in zip(("dw1", "dw2"), _reference(operands)):
        _assert_close(expected[name], want, name)


@pytest.mark.L0
@pytest.mark.gpu_exclusive
def test_training_wgrad_expert_results_do_not_depend_on_neighbors_across_waves(wgrad_op_large):
    op, requirements, device = wgrad_op_large
    dims = _wgrad_dims(requirements)
    shapes = _output_shapes(requirements, _LARGE_EXPERTS)
    base = _expert_payloads(_LARGE_COUNTS, dims, seed=81, device=device)
    others = _expert_payloads((129, 2000, 7, 1024, 640, 0, 1900, 300), dims, seed=82, device=device)

    def run(payloads):
        return snapshot(_run_dense(op, _place_operands(requirements, payloads, device), torch.bfloat16, device, shapes))

    def pick(result, expert):
        return {name: value[expert] for name, value in result.items()}

    reference = run(base)
    # Expert 1 after a shorter expert 0 (its segment starts 1792 rows earlier); every other neighbor replaced.
    neighbors = run([others[0], base[1], *others[2:]])
    # Reversed: every expert moves and every segment starts at another row.
    reversed_order = run(base[::-1])
    # Duplicated: base expert 5 in every slot.
    duplicated = run([base[5]] * _LARGE_EXPERTS)

    assert_bitwise(pick(neighbors, 1), pick(reference, 1), "expert 1 with new neighbors")
    for slot in range(_LARGE_EXPERTS):
        assert_bitwise(pick(reversed_order, _LARGE_EXPERTS - 1 - slot), pick(reference, slot), f"expert {slot} reversed")
        assert_bitwise(pick(duplicated, slot), pick(reference, 5), f"expert 5 duplicated into slot {slot}")


@pytest.mark.L0
@pytest.mark.gpu_exclusive
def test_training_wgrad_warm_stress(wgrad_op_large):
    """Rule 11: warm and poisoned launches of the multi-wave shape store the bytes of FE grouped WGrad, an independent kernel."""

    from warm_stress import StressCase, assert_warm_stress, stress_validation

    op, requirements, device = wgrad_op_large
    shapes = _output_shapes(requirements, _LARGE_EXPERTS)
    operands = _place_operands(requirements, _expert_payloads(_LARGE_COUNTS, _wgrad_dims(requirements), seed=91, device=device), device)
    out = MoeEpTrainingWgradOutputs(
        fc1=torch.empty(shapes[0], dtype=torch.bfloat16, device=device),
        fc2=torch.empty(shapes[1], dtype=torch.bfloat16, device=device),
    )
    op.training_wgrad(operands, out=out)  # compile before the stress
    reference = {"dw1": _run_grouped_wgrad_kernel(operands, "fc1"), "dw2": _run_grouped_wgrad_kernel(operands, "fc2")}
    torch.cuda.synchronize(device)
    assert_bitwise({"dw1": out.fc1, "dw2": out.fc2}, reference, "training_wgrad vs FE grouped WGrad")

    prepared = op._training_state.wgrad_prepared
    k_tiles = [_round_up(count, 128) // 128 for count in _LARGE_COUNTS]
    thread_blocks = 0
    refills = 0
    for gemm in (prepared.fc1, prepared.fc2):
        tile_m, tile_n, _ = gemm.mma_tiler_mnk
        tiles = (gemm.m // tile_m) * (gemm.n // tile_n)
        thread_blocks = max(thread_blocks, _LARGE_EXPERTS * tiles * gemm.cluster_shape_mn[0] * gemm.cluster_shape_mn[1])
        # Lower bound: fewer than 8 A/B stages fit in shared memory, so a tile with k K-tiles refills more than k - 8.
        refills += tiles * sum(max(0, k - 8) for k in k_tiles)
    assert refills > 0
    assert_warm_stress(
        StressCase(
            f"MoeEp.training_wgrad dense BF16, E{_LARGE_EXPERTS} H{_LARGE_HIDDEN} I{_LARGE_INTERMEDIATE}",
            lambda: op.training_wgrad(operands, out=out),
            {"dw1": out.fc1, "dw2": out.fc2},
            thread_blocks=thread_blocks,
            refills_per_launch=refills,
            reference={name: value.clone() for name, value in reference.items()},
        ),
        validation=stress_validation(device, cluster=2),
        warm=16384,
        poisoned=16384,
    )
