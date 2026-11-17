# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bias candidate coverage uses real graph declarations without allocating CUDA tensors."""

import cudnn
import pytest

from cudnn.engines.manifest import MANIFEST
from cudnn.frost import buffers
from cudnn.sdpa import graph_analyzer as ga
from cudnn.sdpa.fwd import engines, heuristics

pytestmark = pytest.mark.L0


@pytest.fixture
def offered(monkeypatch):
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    return next(family for family in MANIFEST if family.name == "frost_sdpa_fwd").offered_ids()


def _facts(
    monkeypatch,
    *,
    cc=(12, 0),
    sm=188,
    dtype=cudnn.data_type.BFLOAT16,
    sq=67,
    sk=139,
    d=128,
    bias=True,
    bias_fp32=True,
    bias_heads=1,
    causal=False,
    bottom_right=False,
):
    monkeypatch.setattr(ga, "_device_cc", lambda: cc)
    monkeypatch.setattr(ga, "_device_sm_count", lambda: sm)
    graph = cudnn.pygraph(io_data_type=dtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)

    def tensor(name, heads, seq):
        return graph.tensor(name=name, dim=[2, heads, seq, d], stride=[seq * heads * d, d, heads * d, 1], data_type=dtype)

    q, k, v = tensor("q", 4, sq), tensor("k", 2, sk), tensor("v", 2, sk)
    # A key tail needs aligned storage even though its live extent is odd.
    pitch = (sk + 7) // 8 * 8
    bias_t = (
        graph.tensor(
            name="bias",
            dim=[1, bias_heads, sq, sk],
            stride=[bias_heads * sq * pitch, sq * pitch, pitch, 1],
            data_type=cudnn.data_type.FLOAT if bias_fp32 else dtype,
        )
        if bias
        else None
    )
    o, stats = graph.sdpa(
        q=q, k=k, v=v, bias=bias_t, generate_stats=True, attn_scale=d**-0.5, use_causal_mask=causal, use_causal_mask_bottom_right=bottom_right
    )
    o.set_output(True).set_data_type(dtype).set_dim([2, 4, sq, d]).set_stride([sq * 4 * d, d, 4 * d, 1])
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim([2, 4, sq, 1]).set_stride([4 * sq, sq, 1, 1])
    facts = ga.analyze(graph)
    assert facts is not None and facts.invalid is None
    return facts


def _assert_candidates(facts, offered, kind="A"):
    by_id = {offered[spec.name]: spec for spec in engines.ENGINE_SPECS if spec.name in offered}
    plans = heuristics.recommend(kind, facts, offered)
    assert plans, "a graph with a legal explicit plan lost every heuristic candidate"
    for plan in plans:
        assert engines.mismatch(by_id[plan.engine_id].capabilities, facts, plan.knobs) is None
    return plans


@pytest.mark.parametrize("dtype", [cudnn.data_type.HALF, cudnn.data_type.BFLOAT16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "cc,sm",
    [((12, 0), 0), ((12, 0), 84), ((12, 0), 142), ((12, 0), 188), ((12, 1), 48)],
    ids=["sm120-unknown", "sm120-84", "sm120-142", "sm120-188", "sm121-48"],
)
@pytest.mark.parametrize("sq", [1, 16, 17, 67, 129], ids=["decode", "decode-edge", "short", "tail", "prefill"])
@pytest.mark.parametrize("causal", [False, True], ids=["unmasked", "causal"])
@pytest.mark.parametrize("bias_fp32", [False, True], ids=["io-bias", "fp32-bias"])
def test_sm120_bias_keeps_an_admissible_candidate(monkeypatch, offered, dtype, cc, sm, sq, causal, bias_fp32):
    facts = _facts(monkeypatch, cc=cc, sm=sm, dtype=dtype, sq=sq, causal=causal, bias_fp32=bias_fp32)
    owner = next(spec for spec in engines.ENGINE_SPECS if spec.name == "sdpa_fwd_sm120_f16_general")
    explicit = engines.SdpaFwdKnobs(tile_m=64, tile_n=128, cga=1, sched_policy=0, pack_gqa=False, split_kv=1)
    assert owner.name in offered
    assert engines.mismatch(owner.capabilities, facts, explicit) is None, "the fixture must have a legal explicit witness"
    _assert_candidates(facts, offered, "FALLBACK")
    _assert_candidates(facts, offered)


@pytest.mark.parametrize("dtype", [cudnn.data_type.HALF, cudnn.data_type.BFLOAT16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("bias", [False, True], ids=["plain", "bias"])
@pytest.mark.parametrize("cc,sm,d", [((8, 0), 108, 64), ((10, 0), 148, 64), ((10, 3), 152, 64), ((10, 7), 216, 64)], ids=["sm80", "sm100", "sm103", "sm107"])
def test_other_arch_bias_candidates_remain_available(monkeypatch, offered, dtype, bias, cc, sm, d):
    # SM80 needs a compact per-head plane; SM100/107 require the I/O dtype.
    facts = _facts(monkeypatch, cc=cc, sm=sm, dtype=dtype, d=d, sq=128, sk=256, bias=bias, bias_heads=4, bias_fp32=False)
    witnesses = [engines.SdpaFwdKnobs(tile_m=128, tile_n=tile_n, cga=cga, sched_policy=0, pack_gqa=False, split_kv=1) for tile_n in (64, 128) for cga in (1, 2)]
    assert any(engines.mismatch(spec.capabilities, facts, knobs) is None for spec in engines.ENGINE_SPECS if spec.name in offered for knobs in witnesses)
    _assert_candidates(facts, offered, "FALLBACK")
    _assert_candidates(facts, offered)


@pytest.mark.parametrize("d,bottom_right", [(64, False), (256, False), (128, True)], ids=["d64", "d256", "bottom-right"])
def test_sm120_unsupported_bias_graphs_remain_declined(monkeypatch, offered, d, bottom_right):
    facts = _facts(monkeypatch, d=d, bottom_right=bottom_right)
    assert not heuristics.recommend("A", facts, offered)


@pytest.mark.parametrize("dtype", [cudnn.data_type.HALF, cudnn.data_type.BFLOAT16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("cc,sm", [((8, 0), 108), ((8, 9), 128), ((12, 0), 188), ((12, 1), 48)], ids=["sm80", "sm89", "sm120", "sm121"])
@pytest.mark.parametrize("bias_fp32", [False, True], ids=["io-bias", "fp32-bias"])
def test_backward_bias_candidates_remain_available(monkeypatch, dtype, cc, sm, bias_fp32):
    from cudnn.sdpa.bwd import engines as bwd_engines
    from cudnn.sdpa.bwd import heuristics as bwd_heuristics

    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    monkeypatch.setattr(ga, "_device_cc", lambda: cc)
    monkeypatch.setattr(ga, "_device_sm_count", lambda: sm)
    graph = cudnn.pygraph(io_data_type=dtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    b, h, hk, sq, sk, d = 2, 4, 2, 128, 256, 64

    def tensor(name, heads, seq):
        return graph.tensor(name=name, dim=[b, heads, seq, d], stride=[seq * heads * d, d, heads * d, 1], data_type=dtype)

    q, k, v = tensor("q", h, sq), tensor("k", hk, sk), tensor("v", hk, sk)
    o, do = tensor("o", h, sq), tensor("do", h, sq)
    stats = graph.tensor(name="stats", dim=[b, h, sq, 1], stride=[h * sq, sq, 1, 1], data_type=cudnn.data_type.FLOAT)
    bias_dtype = cudnn.data_type.FLOAT if bias_fp32 else dtype
    bias = graph.tensor(name="bias", dim=[1, h, sq, sk], stride=[h * sq * sk, sq * sk, sk, 1], data_type=bias_dtype)
    dbias = graph.tensor(name="dbias", dim=[1, h, sq, sk], stride=[h * sq * sk, sq * sk, sk, 1], data_type=bias_dtype)
    dbias.set_output(True)
    # SM80/SM89's native contract requires bias and dBias together.
    dq, dk, dv = graph.sdpa_backward(q=q, k=k, v=v, o=o, dO=do, stats=stats, bias=bias, dBias=dbias, attn_scale=d**-0.5)
    for output, heads, seq in ((dq, h, sq), (dk, hk, sk), (dv, hk, sk)):
        output.set_output(True).set_data_type(dtype).set_dim([b, heads, seq, d]).set_stride([seq * heads * d, d, heads * d, 1])
    facts = ga.analyze(graph)
    assert facts is not None and facts.invalid is None
    offered = next(family for family in MANIFEST if family.name == "frost_sdpa_bwd").offered_ids()
    by_id = {offered[spec.name]: spec for spec in bwd_engines.ENGINE_SPECS if spec.name in offered}
    for kind in ("A", "FALLBACK"):
        plans = bwd_heuristics.recommend(kind, facts, offered)
        assert plans
        assert all(bwd_engines.mismatch(by_id[plan.engine_id].capabilities, facts, plan.knobs) is None for plan in plans)


@pytest.mark.parametrize("dtype", [cudnn.data_type.HALF, cudnn.data_type.BFLOAT16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("opt_in", [False, True], ids=["default", "opt-in"])
@pytest.mark.parametrize(
    "over",
    [
        dict(sq=17),
        dict(sq=67),
        dict(cc=(10, 0), sm=148, d=64, sq=128, sk=256, bias_heads=4, bias_fp32=False),
        dict(cc=(10, 3), sm=152, d=64, sq=128, sk=256, bias_heads=4, bias_fp32=False),
    ],
    ids=["sm120-short", "sm120-tail", "sm100", "sm103"],
)
def test_recovered_bias_candidates_preserve_backend_priority(monkeypatch, offered, dtype, opt_in, over):
    from cudnn.engines.base import PlanConfig
    from cudnn.engines.heuristics import BACKEND, LAST_RESORT, _assemble

    facts = _facts(monkeypatch, dtype=dtype, **over)
    if not opt_in:
        monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES")
    plans = _assert_candidates(facts, offered)
    assert plans == _assert_candidates(facts, offered, "FALLBACK")
    proposal = heuristics.propose("A", facts, offered)
    assert proposal == (plans + [BACKEND] if opt_in else [BACKEND, LAST_RESORT] + plans)
    if not opt_in:
        # Recovery must also trail backend plans found only by FALLBACK mode.
        backend = PlanConfig(8, mode=cudnn.heur_mode.FALLBACK)
        assembled = _assemble([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK], lambda kind: proposal if kind == "A" else [], [backend])
        assert assembled[0].engine_id == backend.engine_id
        assert [p.engine_id for p in assembled[1:]] == [p.engine_id for p in plans]
        assert [p.engine_id for p in _assemble([cudnn.heur_mode.OPENSOURCE], lambda kind: proposal, [])] == [p.engine_id for p in plans]
