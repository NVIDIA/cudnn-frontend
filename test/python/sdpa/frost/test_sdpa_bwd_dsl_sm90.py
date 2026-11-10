# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sdpa_bwd_sm90``: the SM80 prepared backward chain around a warpgroup-MMA
main kernel, through the graph API against the canonical references
(``sdpa/fp16_ref.py``), plus the row's registration and its declines.

THD / varlen runs the SM80 row's prologue and chain; the packed graph cases
use the shared independent-origin fixtures (``d128_thd_test_utils``) with a
CPU FP64 oracle.
"""

from __future__ import annotations

import math

import pytest
import torch

import cudnn
from frost_test_utils import offers_engine, requires_bwd_engine_row, requires_dsl, select_engine
from test_utils import torch_fork_set_rng

ENGINE = "sdpa_bwd_sm90"
_ENGINE_ID = 20618  # FROST_SDPA_BWD_ID_BASE + manifest slot 18


@pytest.fixture(autouse=True)
def _enable_frost(monkeypatch):
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")


# ---------------------------------------------------------------------------
# Registration (no GPU)
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_sm90_row_registration():
    from cudnn.engines.manifest import MANIFEST
    from cudnn.sdpa.bwd import heuristics
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == ENGINE)
    caps = spec.capabilities
    assert (caps.sm_lo, caps.sm_hi) == (90, 90)
    assert not caps.bias and not caps.dbias and caps.dsink and caps.thd and caps.deterministic
    family = next(f for f in MANIFEST if f.name == "frost_sdpa_bwd")
    # New rows start behind CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1.
    assert family.slots[ENGINE].slot == 18 and family.slots[ENGINE].opt_in
    from cudnn.engines.engine_ids import FROST_SDPA_BWD_ID_BASE

    assert FROST_SDPA_BWD_ID_BASE + 18 == _ENGINE_ID
    # Opt-in only: TRAIL_ONLY_ROWS ranks default-offered rows, so the row joins it only if it is promoted to default.
    assert ENGINE not in heuristics.TRAIL_ONLY_ROWS


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def _bhsd(b, h, s, d, dtype, empty=False):
    """Logical BHSD over compact BSHD storage."""
    return (torch.empty if empty else torch.randn)(b, s, h, d, dtype=dtype, device="cuda").transpose(1, 2)


def _ref(q, k, v, do, *, scale, causal, bottom_right, window_left, window_right, padding, sink):
    from sdpa.fp16_ref import compute_ref, compute_ref_backward

    kw = dict(
        attn_scale=scale,
        diag_align=cudnn.diagonal_alignment.BOTTOM_RIGHT if bottom_right else cudnn.diagonal_alignment.TOP_LEFT,
        right_bound=window_right if window_right is not None else (0 if causal else None),
        left_bound=None if window_left is None else window_left + 1,
        padding=padding,
        sink_token=sink,
        torch_type=q.dtype,
    )
    o, stats, _, _ = compute_ref(q, k, v, **kw)
    dq, dk, dv, _, dsink = compute_ref_backward(q, k, v, o, do, **kw)
    return o.to(q.dtype), stats.contiguous(), dq, dk, dv, dsink


def _build(tensors, *, scale, causal, bottom_right, window_left, window_right, deterministic, lens, sink, bias=None):
    q_gpu, k_gpu, v_gpu, o_gpu, do_gpu, stats_gpu, dq_gpu, dk_gpu, dv_gpu = tensors
    io = cudnn.data_type.HALF if q_gpu.dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    ports = {name: g.tensor_like(t, name=name) for name, t in zip(("q", "k", "v", "o", "dO", "stats"), tensors[:6])}
    kw = dict(name="sdpa_backward", attn_scale=scale, **ports)
    if window_right is not None:
        kw["diagonal_band_right_bound"] = window_right
        kw["diagonal_alignment"] = cudnn.diagonal_alignment.BOTTOM_RIGHT if bottom_right else cudnn.diagonal_alignment.TOP_LEFT
        if window_left is not None:
            kw["diagonal_band_left_bound"] = window_left + 1
    else:
        if bottom_right:
            kw["use_causal_mask_bottom_right"] = True
        elif causal:
            kw["use_causal_mask"] = True
        if window_left is not None:
            kw["sliding_window_length"] = window_left + 1
    if deterministic:
        kw["use_deterministic_algorithm"] = True
    pack = {}
    if lens is not None:
        sq_t, skv_t = g.tensor_like(lens[0], name="seq_q"), g.tensor_like(lens[1], name="seq_kv")
        kw.update(use_padding_mask=True, seq_len_q=sq_t, seq_len_kv=skv_t)
        pack.update({sq_t: lens[0], skv_t: lens[1]})
    dsink_gpu = None
    if sink is not None:
        dsink_gpu = torch.full_like(sink, float("nan"))
        sink_t, dsink_t = g.tensor_like(sink, name="sink"), g.tensor_like(dsink_gpu, name="dSink")
        kw.update(sink_token=sink_t, dSink_token=dsink_t)
        pack.update({sink_t: sink, dsink_t: dsink_gpu})
    if bias is not None:
        kw["bias"] = g.tensor_like(bias, name="bias")
    dq, dk, dv = g.sdpa_backward(**kw)
    for port, t in zip((dq, dk, dv), (dq_gpu, dk_gpu, dv_gpu)):
        port.set_output(True).set_dim(t.shape).set_stride(t.stride())
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    pack.update({ports[n]: t for n, t in zip(("q", "k", "v", "o", "dO", "stats"), tensors[:6])})
    pack.update({dq: dq_gpu, dk: dk_gpu, dv: dv_gpu})
    return g, pack, dsink_gpu


def _run_case(
    *,
    b=2,
    h=4,
    hk=None,
    sq=256,
    skv=256,
    d=128,
    dv=None,
    dtype=torch.bfloat16,
    causal=False,
    bottom_right=False,
    window_left=None,
    window_right=None,
    deterministic=False,
    padding=None,
    sink=False,
    scale=None,
    repeats=1,
):
    hk = h if hk is None else hk
    dv = d if dv is None else dv
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    q, k, v, do = _bhsd(b, h, sq, d, dtype), _bhsd(b, hk, skv, d, dtype), _bhsd(b, hk, skv, dv, dtype), _bhsd(b, h, sq, dv, dtype)
    sink_gpu = torch.randn(1, h, 1, 1, dtype=torch.float32, device="cuda") if sink else None
    mask = dict(causal=causal, bottom_right=bottom_right, window_left=window_left, window_right=window_right)
    o_ref, stats, dq_ref, dk_ref, dv_ref, dsink_ref = _ref(q, k, v, do, scale=scale, padding=padding, sink=sink_gpu, **mask)
    o = _bhsd(b, h, sq, dv, dtype, empty=True).copy_(o_ref)
    grads = [_bhsd(b, n, s, w, dtype, empty=True) for n, s, w in ((h, sq, d), (hk, skv, d), (hk, skv, dv))]
    lens = None
    if padding is not None:
        lens = tuple(torch.tensor(p, dtype=torch.int32, device="cuda").view(b, 1, 1, 1) for p in padding)
    g, pack, dsink = _build((q, k, v, o, do, stats, *grads), scale=scale, deterministic=deterministic, lens=lens, sink=sink_gpu, **mask)
    select_engine(g, ENGINE)
    g.check_support()
    g.build_plans()
    assert g.selected_engine is not None and g.selected_engine.name == ENGINE
    # The shared SM80 native binder (and its output-alias guards) serves the row.
    assert g._compiled_plans[g._plan_index]._prepared.spec.name == "sdpa_bwd_sm80"
    workspace = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    runs = []
    for _ in range(repeats):
        for t in grads:
            t.fill_(float("nan"))
        g.execute(pack, workspace)
        torch.cuda.synchronize()
        runs.append([t.clone() for t in grads])
    tol = {"atol": 2e-2 if dtype == torch.float16 else 5e-2, "rtol": 5e-2}
    for got, ref in zip(grads, (dq_ref, dk_ref, dv_ref)):
        torch.testing.assert_close(got.float(), ref.float(), **tol)
    if sink:
        torch.testing.assert_close(dsink.float(), dsink_ref.float(), **tol)
    if padding is not None:
        # Padded rows are written as exact zeros.
        for i, (lq, lkv) in enumerate(zip(*padding)):
            for t, live in zip(grads, (lq, lkv, lkv)):
                assert not t[i, :, live:].any()
    return runs


# ---------------------------------------------------------------------------
# Numerics through the graph
# ---------------------------------------------------------------------------

_MASKS = {
    "dense": dict(),
    "causal": dict(causal=True),
    "causal_br_cross": dict(causal=True, bottom_right=True, sq=96, skv=200),
    "swa": dict(causal=True, window_left=37),
    "swa_br": dict(causal=True, bottom_right=True, window_left=50, sq=70, skv=130),
    "right_band": dict(window_left=21, window_right=5),
    "tails": dict(sq=77, skv=93),
    "gqa": dict(h=4, hk=2),
    "mqa": dict(h=6, hk=1),
    "padding": dict(padding=([100, 1], [150, 37]), sq=100, skv=150),
    "padding_causal_br": dict(causal=True, bottom_right=True, padding=([256, 63], [255, 129])),
    "sink": dict(sink=True),
    "sink_causal_gqa": dict(sink=True, causal=True, hk=2),
    "zero_scale": dict(scale=0.0),
    "zero_scale_causal_padding": dict(scale=0.0, causal=True, padding=([256, 100], [256, 200])),
}


@pytest.mark.L0
@requires_dsl
@requires_bwd_engine_row(ENGINE)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("mask", list(_MASKS))
@torch_fork_set_rng(seed=0)
def test_sdpa_bwd_dsl_sm90_graph(mask, d, dtype):
    _run_case(d=d, dtype=dtype, **_MASKS[mask])


@pytest.mark.L0
@requires_dsl
@requires_bwd_engine_row(ENGINE)
@pytest.mark.parametrize("d,dv", [(40, 40), (96, 96), (64, 128), (128, 64), (72, 120)])
@torch_fork_set_rng(seed=1)
def test_sdpa_bwd_dsl_sm90_head_dim_envelopes(d, dv):
    """Logical head dims inside the D64 / D128 envelopes (masked loads, gated stores)."""
    _run_case(d=d, dv=dv, causal=True, sq=130, skv=130)


@pytest.mark.L0
@requires_dsl
@requires_bwd_engine_row(ENGINE)
@torch_fork_set_rng(seed=2)
def test_sdpa_bwd_dsl_sm90_deterministic_is_bitwise_repeatable():
    runs = _run_case(deterministic=True, causal=True, sq=512, skv=512, hk=2, repeats=3)
    for again in runs[1:]:
        for a, b in zip(runs[0], again):
            assert torch.equal(a, b)


@pytest.mark.L0
@requires_dsl
@requires_bwd_engine_row(ENGINE)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
def test_sdpa_bwd_dsl_sm90_padding_is_unread(d, dtype, causal):
    """Rows past the runtime lengths hold NaN, and one batch member is empty
    (S_q = 0): its Q / O / dO / Stats are all NaN, so masked P = 0 must give
    dS = 0. Valid rows match a reference on clean data; the empty member's
    dK / dV are exact zeros."""
    torch.manual_seed(90 + d)
    b, h, hk, sq, skv = 3, 4, 2, 128, 192
    lens_q, lens_kv = [sq - 7, 0, sq // 2], [skv - 3, skv // 2, 1]
    scale = d**-0.5
    q, k, v, do = _bhsd(b, h, sq, d, dtype), _bhsd(b, hk, skv, d, dtype), _bhsd(b, hk, skv, d, dtype), _bhsd(b, h, sq, d, dtype)
    mask = dict(causal=causal, bottom_right=False, window_left=None, window_right=None)
    o_ref, stats, dq_ref, dk_ref, dv_ref, _ = _ref(q, k, v, do, scale=scale, padding=(lens_q, lens_kv), sink=None, **mask)
    o = _bhsd(b, h, sq, d, dtype, empty=True).copy_(o_ref)
    for t, lens in ((q, lens_q), (do, lens_q), (o, lens_q), (k, lens_kv), (v, lens_kv)):
        for i, n in enumerate(lens):
            t[i, :, n:] = float("nan")
    stats[1] = float("nan")
    grads = [_bhsd(b, n, s, d, dtype, empty=True).fill_(float("nan")) for n, s in ((h, sq), (hk, skv), (hk, skv))]
    lens = tuple(torch.tensor(x, dtype=torch.int32, device="cuda").view(b, 1, 1, 1) for x in (lens_q, lens_kv))
    g, pack, _ = _build((q, k, v, o, do, stats, *grads), scale=scale, deterministic=False, lens=lens, sink=None, **mask)
    select_engine(g, ENGINE)
    g.check_support()
    g.build_plans()
    assert g.selected_engine.name == ENGINE
    workspace = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    g.execute(pack, workspace)
    torch.cuda.synchronize()
    for got, ref, lens_x in zip(grads, (dq_ref, dk_ref, dv_ref), (lens_q, lens_kv, lens_kv)):
        for i, n in enumerate(lens_x):
            assert not torch.isnan(got[i, :, :n]).any(), f"batch {i}: NaN in valid rows"
            torch.testing.assert_close(got[i, :, :n].float(), ref[i, :, :n].float(), atol=3e-2, rtol=3e-2)
    assert not grads[1][1, :, : lens_kv[1]].any() and not grads[2][1, :, : lens_kv[1]].any(), "empty member: dK / dV must be zero"


# ---------------------------------------------------------------------------
# THD: packed ports with independent origins, CPU FP64 oracle
# ---------------------------------------------------------------------------


@pytest.mark.L0
@requires_dsl
@requires_bwd_engine_row(ENGINE)
@pytest.mark.parametrize(
    "lens_q,lens_kv,dtype,causal,d",
    [
        ((1, 63, 64, 65, 127, 128, 129), (129, 128, 127, 65, 64, 63, 1), torch.bfloat16, False, 128),
        ((129, 0, 65, 1), (3, 257, 0, 129), torch.float16, True, 128),
        ((77, 130, 5), (200, 64, 9), torch.bfloat16, True, 64),
        ((0, 0, 5), (1, 129, 9), torch.float16, False, 128),
        ((3, 0), (0, 70), torch.bfloat16, True, 64),
    ],
)
@pytest.mark.parametrize("stats_layout", ["TH1", "HN"])
def test_sdpa_bwd_dsl_sm90_thd_graph(lens_q, lens_kv, dtype, causal, d, stats_layout):
    from d128_thd_test_utils import check_case, make_case, make_graph, make_workspace, poison, variant_pack

    case = make_case(lens_q, lens_kv, dtype=dtype, causal=causal, stats_layout=stats_layout, d_qk=d, d_v=d)
    graph, refs = make_graph(case, engine_name=ENGINE, engine_id=_ENGINE_ID)
    pack, workspace = variant_pack(case, refs), make_workspace(graph.get_workspace_size())
    for _ in range(2):
        poison(case, workspace)
        graph.execute(pack, workspace.tensor)
        check_case(case, workspace)


@pytest.mark.L0
@requires_dsl
@requires_bwd_engine_row(ENGINE)
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("bottom_right", [False, True], ids=["causal_tl", "causal_br"])
def test_sdpa_bwd_dsl_sm90_thd_gqa(d, bottom_right):
    """Packed GQA (6 query heads over 2 KV heads): the SM80 per-query-head dK / dV partials and their reduction."""
    from d128_thd_test_utils import check_case, make_case, make_graph, make_workspace, poison, variant_pack

    case = make_case((65, 0, 3), (17, 129, 97), heads=6, kv_heads=2, d_qk=d, d_v=d, causal=True, bottom_right=bottom_right)
    graph, refs = make_graph(case, engine_name=ENGINE, engine_id=_ENGINE_ID)
    pack, workspace = variant_pack(case, refs), make_workspace(graph.get_workspace_size())
    poison(case, workspace)
    graph.execute(pack, workspace.tensor)
    check_case(case, workspace)


# ---------------------------------------------------------------------------
# Declines: the row is not offered, the backend keeps the graph
# ---------------------------------------------------------------------------


def _offered(*, d=128, bias=False):
    b, h, s = 1, 2, 128
    q = _bhsd(b, h, s, d, torch.bfloat16)
    stats = torch.empty(b, h, s, 1, dtype=torch.float32, device="cuda")
    grads = [_bhsd(b, h, s, d, torch.bfloat16, empty=True) for _ in range(3)]
    bias_gpu = torch.randn(1, h, s, s, dtype=torch.bfloat16, device="cuda") if bias else None
    g, _, _ = _build(
        (q, q, q, q, q, stats, *grads),
        scale=d**-0.5,
        causal=False,
        bottom_right=False,
        window_left=None,
        window_right=None,
        deterministic=False,
        lens=None,
        sink=None,
        bias=bias_gpu,
    )
    return offers_engine(g, ENGINE)


@pytest.mark.L0
@requires_dsl
@requires_bwd_engine_row(ENGINE)
def test_sdpa_bwd_dsl_sm90_declines():
    assert _offered()
    assert not _offered(bias=True)
    assert not _offered(d=192)
    assert not _offered(d=256)


@pytest.mark.L0
@requires_dsl
@requires_bwd_engine_row(ENGINE)
def test_sdpa_bwd_dsl_sm90_is_opt_in(monkeypatch):
    """Without the FROST opt-in the row is not offered, even for a graph it serves."""
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    assert not _offered()


@pytest.mark.L0
@requires_dsl
@requires_bwd_engine_row(ENGINE)
def test_sdpa_bwd_dsl_sm90_rejects_other_tiles():
    from cudnn.sdpa.bwd.api_dsl_sm90 import SdpaBwdDslSm90

    e = _bhsd(1, 2, 128, 128, torch.bfloat16, empty=True)
    stats = torch.empty(1, 2, 128, 1, dtype=torch.float32, device="cuda")
    kw = dict(sample_q=e, sample_k=e, sample_v=e, sample_o=e, sample_do=e, sample_stats=stats, sample_dq=e, sample_dk=e, sample_dv=e, scale_softmax=128**-0.5)
    assert SdpaBwdDslSm90(**kw).check_support()
    with pytest.raises(NotImplementedError, match="64 x 64"):
        SdpaBwdDslSm90(**kw, tile_m=128).check_support()
    with pytest.raises(NotImplementedError, match="bias"):
        SdpaBwdDslSm90(**kw, has_bias=True).check_support()
