# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Legacy SM90 ragged-backward exclusion, replay protection, and version detector."""

from dataclasses import replace

import cudnn
import pytest
from cudnn.engines import manifest
from cudnn.sdpa import graph_analyzer
from cudnn.sdpa.bwd import backend_guard as guard
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0


def _facts(**overrides):
    facts = SdpaGraphFacts(is_backward=True, device_cc=(9, 0), thd=True, d_qk=200, d_v=72, dtype=cudnn.data_type.BFLOAT16)
    return replace(facts, **overrides)


@pytest.mark.parametrize(
    "version,excluded",
    [
        (90000, True),
        (91002, True),
        (91600, True),
        (91801, True),
        (91901, True),
        (92000, True),
        (92101, True),
        (92200, True),
        (92301, True),
        (92302, True),
        (92400, True),
        (92401, True),
        (92500, True),
        (92501, False),
        (92700, False),
    ],
)
@pytest.mark.parametrize("dtype", [cudnn.data_type.HALF, cudnn.data_type.BFLOAT16])
def test_ragged_backward_guard_version_bounds(monkeypatch, version, excluded, dtype):
    monkeypatch.setattr(cudnn, "backend_version", lambda: version)
    result = guard.backend_engine_guard(None, _facts(dtype=dtype))
    assert set(result) == ({6} if excluded else set())
    if excluded:
        assert "fixed in cuDNN 9.25.1" in result[6]


@pytest.mark.parametrize(
    "overrides",
    [
        dict(d_qk=192),
        dict(d_qk=128),
        dict(d_qk=512),
        dict(d_v=64),
        dict(d_v=192),
        dict(thd=False),
        dict(device_cc=(8, 0)),
        dict(device_cc=(10, 0)),
        dict(device_cc=(10, 7)),
        dict(device_cc=(12, 0)),
        dict(is_backward=False),
        dict(is_fp8=True),
        dict(is_mxfp8=True),
        dict(dtype=cudnn.data_type.FLOAT),
        dict(invalid="bad graph"),
    ],
)
@pytest.mark.parametrize("version", [91801, 92400])
def test_ragged_backward_guard_preserves_other_domains(monkeypatch, overrides, version):
    monkeypatch.setattr(cudnn, "backend_version", lambda: version)
    assert guard.backend_engine_guard(None, _facts(**overrides)) == {}


def test_ragged_backward_guard_manifest_and_missing_facts():
    family = next(f for f in manifest.MANIFEST if f.name == "frost_sdpa_bwd")
    assert manifest.resolve_backend_engine_guard(family) is guard.backend_engine_guard
    assert guard.backend_engine_guard(None, None) == {}


def _graph(dtype=cudnn.data_type.BFLOAT16, *, ragged_q=True, ragged_kv=True):
    g = cudnn.pygraph(io_data_type=dtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    b, h, s, dqk, dv = 2, 1, 129, 200, 72

    def tensor(name, width, dt=dtype, ragged=ragged_q):
        result = g.tensor(name=name, dim=(b, h, s, width), stride=(h * s * width, width, h * width, 1), data_type=dt)
        if ragged:
            offset = g.tensor(name=name + "_offset", dim=(b + 1,), stride=(1,), data_type=cudnn.data_type.INT64)
            result.set_ragged_offset(offset)
        return result

    q, k, v, o, do, stats = (
        tensor("q", dqk),
        tensor("k", dqk, ragged=ragged_kv),
        tensor("v", dv, ragged=ragged_kv),
        tensor("o", dv),
        tensor("do", dv),
        tensor("stats", 1, cudnn.data_type.FLOAT),
    )
    seq_q = g.tensor(name="seq_q", dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)
    seq_kv = g.tensor(name="seq_kv", dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)
    grads = g.sdpa_backward(q=q, k=k, v=v, o=o, dO=do, stats=stats, attn_scale=dqk**-0.5, use_padding_mask=True, seq_len_q=seq_q, seq_len_kv=seq_kv)
    for grad, source in zip(grads, (q, k, v)):
        grad.set_output(True).set_dim(source.get_dim()).set_stride(source.get_stride()).set_ragged_offset(source.ragged_offset)
    g.validate()
    g.build_operation_graph()
    return g


@pytest.mark.parametrize("ragged_q,ragged_kv", [(True, True), (False, False), (True, False), (False, True)])
def test_mixed_ragged_layout_is_provider_support_not_invalid(monkeypatch, ragged_q, ragged_kv):
    from cudnn.sdpa.fwd import engines as fwd
    from cudnn.sdpa.bwd import engines as bwd

    monkeypatch.setattr(graph_analyzer, "_device_cc", lambda: (9, 0))
    g = _graph(ragged_q=ragged_q, ragged_kv=ragged_kv)
    facts = graph_analyzer.analyze(g)
    assert facts.invalid is None
    assert facts.is_backward and facts.device_cc == (9, 0)
    assert (facts.d_qk, facts.d_v, facts.dtype) == (200, 72, cudnn.data_type.BFLOAT16)
    assert facts.thd == ragged_q
    assert facts.mixed_qkv_ragged == (ragged_q != ragged_kv)
    if facts.mixed_qkv_ragged:
        # Neither direction may treat a legal backend layout as a FROST layout.
        # Check every registered row, including explicit public engine replay.
        for spec in bwd.ENGINE_SPECS:
            assert "all dense or all ragged" in bwd.mismatch(spec.capabilities, facts)
        for spec in fwd.ENGINE_SPECS:
            assert "all dense or all ragged" in fwd.mismatch(spec.capabilities, replace(facts, is_backward=False))


@pytest.mark.parametrize("ragged_kv", [True, False], ids=["ragged-kv", "dense-kv"])
@pytest.mark.parametrize("knobs", [{}, {cudnn.knob_type.STREAM_K: 1}])
@pytest.mark.parametrize("version", [91600, 91801, 92302, 92400, 92500])
def test_ragged_backward_explicit_replay_declines_before_lowering(monkeypatch, knobs, version, ragged_kv):
    monkeypatch.setattr(cudnn, "backend_version", lambda: version)
    monkeypatch.setattr(graph_analyzer, "_device_cc", lambda: (9, 0))
    g = _graph(ragged_kv=ragged_kv)
    facts = graph_analyzer.analyze(g)
    assert facts.invalid is None and facts.thd and facts.is_backward
    monkeypatch.setattr(g, "_lower_backend_graph", lambda: pytest.fail("an excluded engine must not be lowered"))
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="ragged tile mapping is fixed"):
        g.create_execution_plan(6, knobs)


@pytest.mark.parametrize("ragged_kv", [True, False], ids=["ragged-kv", "dense-kv"])
@pytest.mark.parametrize("mode", [cudnn.heur_mode.A, cudnn.heur_mode.B, cudnn.heur_mode.OPENSOURCE])
def test_ragged_backward_default_has_no_unsafe_plan(mode, ragged_kv):
    import torch

    if torch.cuda.get_device_capability() != (9, 0) or cudnn.backend_version() >= guard.RAGGED_TILE_MAPPING_FIXED_IN:
        pytest.skip("requires native SM90 with an affected backend before cuDNN 9.25.1")
    g = _graph(ragged_kv=ragged_kv)
    try:
        g.create_execution_plans([mode, cudnn.heur_mode.FALLBACK])
    except cudnn.cudnnGraphNotSupportedError:
        return
    from cudnn.engines.engine_ids import BACKEND_HEURISTIC_ENGINE_ID

    assert all(plan.engine_id != 6 for plan in g.plans)
    if any(plan.engine_id == BACKEND_HEURISTIC_ENGINE_ID for plan in g.plans):
        # OPENSOURCE can leave an opaque candidate even with no concrete
        # backend configs. On this old SM90 stack that candidate must decline;
        # merely appearing in the unbuilt list is not successful execution.
        assert g._lowered_graph.get_execution_plan_count() == 0
        with pytest.raises(cudnn.cudnnGraphNotSupportedError):
            g.check_support()
            g.build_plans()


@pytest.mark.parametrize("ragged_kv", [True, False], ids=["ragged-kv", "dense-kv"])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("deterministic", [False, True])
def test_ragged_backward_fix_version_detector(monkeypatch, request, cudnn_handle, dtype_name, deterministic, ragged_kv):
    """Known-old failure is a detector result, not a qualified numerical case."""
    import torch
    from sdpa.fp16 import TensorUid, exec_sdpa
    from sdpa.random_config import ExecConfig

    if torch.cuda.get_device_capability() != (9, 0) or cudnn.backend_version() < 91000:
        pytest.skip("requires native SM90 and the numerical harness's cuDNN 9.10+ baseline")
    affected = cudnn.backend_version() < guard.RAGGED_TILE_MAPPING_FIXED_IN
    monkeypatch.setattr(guard, "RAGGED_TILE_MAPPING_FIXED_IN", 0)
    original_plans = cudnn.pygraph.create_execution_plans

    def pin_backend_backward(graph, heuristics=None):
        original_plans(graph, heuristics)
        if graph_analyzer.analyze(graph).is_backward:
            index = next((i for i, plan in enumerate(graph.plans) if plan.engine_id == 6), None)
            if index is None:
                pytest.skip(f"cuDNN {cudnn.backend_version()} no longer offers engine 6 for the detector graph")
            graph.select_plan(index)

    monkeypatch.setattr(cudnn.pygraph, "create_execution_plans", pin_backend_backward)
    if not ragged_kv:
        from cudnn.graph_types import Tensor

        set_ragged_offset = Tensor.set_ragged_offset
        kv_uids = {int(uid) for uid in (TensorUid.k, TensorUid.v, TensorUid.dK, TensorUid.dV)}

        def dense_kv_offset(tensor, offset):
            if tensor.get_uid() in kv_uids:
                # Full equal lengths make the harness's dense and packed K/V
                # byte offsets identical. Only the graph declaration changes.
                assert tensor.get_stride()[0] == tensor.get_dim()[2] * tensor.get_stride()[2]
                offset = None
            return set_ragged_offset(tensor, offset)

        monkeypatch.setattr(Tensor, "set_ragged_offset", dense_kv_offset)
    cfg = ExecConfig(
        data_type=getattr(torch, dtype_name),
        rng_geom_seed=829,
        rng_data_seed=857,
        is_infer=False,
        is_padding=True,
        is_ragged=True,
        is_determin=deterministic,
        batches=2,
        d_qk=200,
        d_v=72,
        s_q=129,
        s_kv=129,
        h_q=1,
        h_k=1,
        h_v=1,
        seq_len_q=[129, 103] if ragged_kv else [129, 129],
        seq_len_kv=[129, 112] if ragged_kv else [129, 129],
        diag_align=cudnn.diagonal_alignment.TOP_LEFT,
        right_bound=None,
        with_ragged_token_gap=False,
        with_ragged_head_gap=False,
    )
    cfg.fill_derived_fields()
    if affected:
        with pytest.raises(pytest.fail.Exception, match="disallowed mismatches"):
            exec_sdpa(cfg, request, cudnn_handle)
    else:
        exec_sdpa(cfg, request, cudnn_handle)
