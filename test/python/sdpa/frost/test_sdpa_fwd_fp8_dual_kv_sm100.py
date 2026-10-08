# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-tensor FP8 d128 decode with two KV streams per CTA (TemplateParams.dual_kv_stream).

sm100/prefill_d128_fp8.py runs its second Q sub-tile as a second KV stream over
the first sub-tile's 128 rows: stream 0 takes the even KV tiles, stream 1 the
odd ones, each with its own running max / sum and O accumulator, merged once in
the correction epilogue.  The routing is the adapter's (``_fp8_dual_kv_stream``),
on by default behind ``config_sm100.FP8_DUAL_KV_STREAM_ROUTED``; these tests pin
it on and check every shape family the merge has to get right: an even and an odd tile
count (stream 1's phantom step), one and two tiles, a KV split whose chunks are
odd, bottom-right causal multi-query, an unpacked 64-row Q tile, fp8 and bf16 O,
and paged K/V with unaligned lengths.
"""

import math

import pytest
import torch

import cudnn
import cudnn.sdpa  # noqa: F401  -- registers the FROST engines
from cudnn.sdpa.fwd import config_sm100
from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100
from cudnn.sdpa.fwd.engines import SdpaFwdKnobs
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell, select_engine

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]

ENGINE = "sdpa_fwd_prefill_sm100_fp8"
ENGINE_ID = 20513  # FROST_SDPA_FWD_ID_BASE + manifest slot 13 (sdpa_fwd_prefill_sm100_fp8)
D = 128
F8 = torch.float8_e4m3fn


@pytest.fixture
def routed(monkeypatch):
    monkeypatch.setattr(config_sm100, "FP8_DUAL_KV_STREAM_ROUTED", True)


def _rand(shape, scale=0.5, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(shape, device="cuda", dtype=torch.float32, generator=gen) * scale).to(F8)


@pytest.mark.parametrize(
    ("b", "h", "kh", "s_q", "s_kv", "cga", "expect"),
    [
        (1, 64, 8, 1, 4096, 1, True),  # the decode step
        (8, 64, 8, 4, 4096, 1, True),  # verify width, 4 x 8 packed rows
        (1, 8, 8, 128, 2048, 1, True),  # unpacked, one full Q tile
        (1, 8, 8, 129, 2048, 1, False),  # one row too many for a single tile
        (1, 64, 8, 1, 4096, 2, False),  # an explicit cga2 pair is the classic pipeline
        (1, 64, 8, 1, 4096, None, False),  # an unnamed width resolves to the flavor's cga2 default
    ],
    ids=["decode", "verify4", "unpacked128", "rows129", "cga2", "cga-unset"],
)
def test_adapter_routes_one_tile_decode_shapes_onto_two_kv_streams(routed, b, h, kh, s_q, s_kv, cga, expect):
    """The lowering flag follows the adapter's predicate: per-tensor fp8, d128,
    a NAMED cga1 (the heuristics' choice for every dense / paged fp8 d128 plan;
    an unset knob is this flavor's cga2 default), dense, S_q x packed heads
    within one 128-row tile."""
    q = _rand((b, h, s_q, D))
    k = _rand((b, kh, s_kv, D))
    v = _rand((b, kh, s_kv, D))
    out = torch.empty(b, h, s_q, D, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(b, h, s_q, device="cuda", dtype=torch.float32)
    kw = {} if cga is None else {"cga": cga}
    api = SdpaFwdDslSm100(
        sample_q=q, sample_k=k, sample_v=v, sample_o=out, sample_lse=lse, scale_softmax=1.0 / math.sqrt(D), pertensor_fp8=True, pack_gqa=(h != kh), **kw
    )
    assert api.check_support()
    assert api.template_params().dual_kv_stream is expect


def test_routing_is_on_by_default_and_the_switch_turns_it_off(monkeypatch):
    """The decode step takes the two-stream specialization out of the box;
    config_sm100.FP8_DUAL_KV_STREAM_ROUTED is the one switch that keeps it off."""
    q = _rand((1, 64, 1, D))
    k = _rand((1, 8, 4096, D))
    v = _rand((1, 8, 4096, D))
    out = torch.empty(1, 64, 1, D, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(1, 64, 1, device="cuda", dtype=torch.float32)
    kw = dict(sample_q=q, sample_k=k, sample_v=v, sample_o=out, sample_lse=lse, scale_softmax=1.0 / math.sqrt(D), pertensor_fp8=True, pack_gqa=True, cga=1)
    api = SdpaFwdDslSm100(**kw)
    assert api.check_support()
    assert api.template_params().dual_kv_stream is True
    monkeypatch.setattr(config_sm100, "FP8_DUAL_KV_STREAM_ROUTED", False)
    api = SdpaFwdDslSm100(**kw)
    assert api.check_support()
    assert api.template_params().dual_kv_stream is False


@pytest.mark.parametrize("make", ["make_cfg_d128_decode", "make_cfg_d64_decode", "make_cfg_d128", "make_cfg_d64", "make_cfg_d192"])
def test_only_the_two_stream_fp8_geometry_names_one_live_q_tile(make):
    """The decode tiles subclass CfgD128 and must not inherit its two-stream fields:
    a decode tile's Q span is its own TILES_Q (one tile), never CfgD128's default
    Q_TILES_LIVE of two -- doubling it corrupts the paged THD split's mask bounds."""
    import math as _math

    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    kw = dict(dtype_qkv=2, cta_mma=1)
    if make == "make_cfg_d64_decode":
        kw.update(d_flavor=64, decode_tile=True)
    elif make == "make_cfg_d64":
        kw.update(d_flavor=64)
    elif make == "make_cfg_d192":
        kw.update(cta_mma=2)
    cfg, _ = getattr(config_sm100, make)(TemplateParams(**kw))
    assert cfg.DUAL_KV_STREAM == 0
    assert cfg.Q_TILES_LIVE == cfg.TILES_Q, (make, cfg.TILES_Q, cfg.Q_TILES_LIVE)
    if make == "make_cfg_d128":
        dual, _ = config_sm100.make_cfg_d128(TemplateParams(dtype_qkv=0, dtype_o=2, cta_mma=1, dual_kv_stream=True))
        assert (dual.DUAL_KV_STREAM, dual.Q_TILES_LIVE, dual.TILES_Q) == (1, 1, 2)


def _reference(q, k, v, lens, s_q, causal_br):
    """fp32 reference from the fp8 values; K/V as [B, KH, S_kv, D] with per-batch live lengths."""
    b, h = q.shape[0], q.shape[1]
    g = h // k.shape[1]
    ref = torch.zeros(b, h, s_q, D, device="cuda", dtype=torch.float32)
    for bi in range(b):
        kb = k[bi, :, : lens[bi]].float().repeat_interleave(g, dim=0)
        vb = v[bi, :, : lens[bi]].float().repeat_interleave(g, dim=0)
        s = (q[bi].float() @ kb.transpose(-1, -2)) * D**-0.5
        if causal_br:
            i = torch.arange(s_q, device="cuda")[:, None]
            j = torch.arange(lens[bi], device="cuda")[None, :]
            s = s.masked_fill(j > i + (lens[bi] - s_q), float("-inf"))
        ref[bi] = torch.softmax(s, dim=-1) @ vb
    return ref


def _run_dense(b, h, kh, s_q, s_kv, *, split, causal_br, o_dtype):
    q = _rand((b, s_q, h, D), seed=1).transpose(1, 2)
    k = _rand((b, s_kv, kh, D), seed=2).transpose(1, 2)
    v = _rand((b, s_kv, kh, D), seed=3).transpose(1, 2)
    o = torch.empty(b, s_q, h, D, device="cuda", dtype=o_dtype).transpose(1, 2)
    amax_o = torch.zeros(1, 1, 1, 1, device="cuda", dtype=torch.float32)
    ones = {n: torch.ones(1, 1, 1, 1, device="cuda", dtype=torch.float32) for n in ("dq", "dk", "dv", "ds", "ss", "so")}
    g = cudnn.pygraph(io_data_type=cudnn.data_type.FP8_E4M3, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tq, tk, tv = g.tensor_like(q), g.tensor_like(k), g.tensor_like(v)
    sc = {n: g.tensor_like(t) for n, t in ones.items()}
    kw = dict(use_causal_mask_bottom_right=True) if causal_br else {}
    ret = g.sdpa_fp8(
        q=tq,
        k=tk,
        v=tv,
        descale_q=sc["dq"],
        descale_k=sc["dk"],
        descale_v=sc["dv"],
        descale_s=sc["ds"],
        scale_s=sc["ss"],
        scale_o=sc["so"],
        is_inference=True,
        attn_scale=D**-0.5,
        name="sdpa_fp8",
        **kw,
    )
    to, ta = ret[0], ret[3]
    to.set_output(True).set_dim(o.size()).set_stride(o.stride())
    if o_dtype == torch.bfloat16:
        to.set_data_type(cudnn.data_type.BFLOAT16)
    ta.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.OPENSOURCE])
    lead = select_engine(g, ENGINE)
    if split:
        kn = lead.knobs
        g.create_execution_plan(ENGINE_ID, SdpaFwdKnobs(sched_policy=0, tile_m=kn.tile_m, tile_n=kn.tile_n, cga=1, pack_gqa=kn.pack_gqa, split_kv=split))
        g.select_plan(len(g.plans) - 1)
    plan = g.plans[g._plan_index]
    assert plan.knobs.cga in (None, 1), plan.knobs
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    vp = {tq: q, tk: k, tv: v, to: o, ta: amax_o}
    vp.update({sc[n]: ones[n] for n in ones})
    g.execute(vp, ws, handle=cudnn.create_handle())
    torch.cuda.synchronize()
    ref = _reference(q, k, v, [s_kv] * b, s_q, causal_br)
    return o.float(), ref, amax_o.item()


_DENSE_CASES = [
    # name, b, h, kh, s_q, s_kv (tiles), split, causal, o dtype
    ("even32", 2, 64, 8, 1, 4096, 0, False, torch.bfloat16),
    ("odd33", 2, 64, 8, 1, 4224, 0, False, torch.bfloat16),
    ("one", 2, 64, 8, 1, 128, 0, False, torch.bfloat16),
    ("two", 2, 64, 8, 1, 256, 0, False, torch.bfloat16),
    ("split4-odd-chunks", 1, 64, 8, 1, 4224, 4, False, torch.bfloat16),
    ("causal-q4-odd", 2, 64, 8, 4, 4224, 0, True, torch.bfloat16),
    ("causal-q16-128rows", 1, 64, 8, 16, 2048, 0, True, torch.bfloat16),
    ("unpacked-q64", 1, 8, 8, 64, 2048, 0, False, torch.bfloat16),
    ("fp8-o-odd", 2, 64, 8, 1, 4224, 0, False, F8),
]


@pytest.mark.parametrize(("name", "b", "h", "kh", "s_q", "s_kv", "split", "causal", "o_dtype"), _DENSE_CASES, ids=[c[0] for c in _DENSE_CASES])
def test_dense_two_kv_streams_match_the_reference(routed, name, b, h, kh, s_q, s_kv, split, causal, o_dtype):
    o, ref, amax = _run_dense(b, h, kh, s_q, s_kv, split=split, causal_br=causal, o_dtype=o_dtype)
    assert not torch.isnan(o).any()
    tol = 0.08 if o_dtype == F8 else 0.04  # fp8 P (and O) quantization; the classic path sits at ~0.03
    err = (o - ref).abs().max().item() / ref.abs().max().item()
    assert err < tol, (name, err)
    assert abs(amax - ref.abs().max().item()) <= 0.08 * ref.abs().max().item() + 1e-3, (name, amax, ref.abs().max().item())


def _run_paged(b, h, kh, s_q, page, max_pages, lens, *, causal_br):
    npages = b * max_pages
    k_pool = _rand((npages, kh, page, D), seed=4)
    v_pool = _rand((npages, kh, page, D), seed=5)
    bt = torch.randperm(npages, device="cuda", dtype=torch.int32).view(b, max_pages)
    q = _rand((b, s_q, h, D), seed=6).transpose(1, 2)
    o = torch.empty(b, s_q, h, D, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    amax_o = torch.zeros(1, 1, 1, 1, device="cuda", dtype=torch.float32)
    ones = {n: torch.ones(1, 1, 1, 1, device="cuda", dtype=torch.float32) for n in ("dq", "dk", "dv", "ds", "ss", "so")}
    bt4 = bt.view(b, 1, max_pages, 1)
    slk = torch.tensor(lens, dtype=torch.int32, device="cuda").view(b, 1, 1, 1)
    slq = torch.full((b, 1, 1, 1), s_q, dtype=torch.int32, device="cuda")
    g = cudnn.pygraph(io_data_type=cudnn.data_type.FP8_E4M3, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tq, tk, tv = g.tensor_like(q), g.tensor_like(k_pool), g.tensor_like(v_pool)
    tbk, tbv = g.tensor_like(bt4), g.tensor_like(bt4)
    tsq, tsk = g.tensor_like(slq), g.tensor_like(slk)
    sc = {n: g.tensor_like(t) for n, t in ones.items()}
    kw = dict(use_causal_mask_bottom_right=True) if causal_br else {}
    ret = g.sdpa_fp8(
        q=tq,
        k=tk,
        v=tv,
        descale_q=sc["dq"],
        descale_k=sc["dk"],
        descale_v=sc["dv"],
        descale_s=sc["ds"],
        scale_s=sc["ss"],
        scale_o=sc["so"],
        is_inference=True,
        attn_scale=D**-0.5,
        name="sdpa_fp8",
        use_padding_mask=True,
        seq_len_q=tsq,
        seq_len_kv=tsk,
        paged_attention_k_table=tbk,
        paged_attention_v_table=tbv,
        paged_attention_max_seq_len_kv=max_pages * page,
        **kw,
    )
    to, ta = ret[0], ret[3]
    to.set_output(True).set_dim(o.size()).set_stride(o.stride()).set_data_type(cudnn.data_type.BFLOAT16)
    ta.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.OPENSOURCE])
    plan = select_engine(g, ENGINE)
    assert plan.knobs.cga in (None, 1), plan.knobs
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    vp = {tq: q, tk: k_pool, tv: v_pool, tbk: bt4, tbv: bt4, tsq: slq, tsk: slk, to: o, ta: amax_o}
    vp.update({sc[n]: ones[n] for n in ones})
    g.execute(vp, ws, handle=cudnn.create_handle())
    torch.cuda.synchronize()
    # gather the pages into dense [B, KH, S, D] for the reference
    k_d = k_pool[bt.long()].permute(0, 2, 1, 3, 4).reshape(b, kh, max_pages * page, D)
    v_d = v_pool[bt.long()].permute(0, 2, 1, 3, 4).reshape(b, kh, max_pages * page, D)
    ref = _reference(q, k_d, v_d, lens, s_q, causal_br)
    return o.float(), ref


_PAGED_CASES = [
    # name, b, s_q, page, max_pages, lens, causal
    ("page16-unaligned", 2, 1, 16, 64, [1024, 777], False),
    ("page128-odd-tiles", 2, 1, 128, 9, [1152, 1000], False),
    ("page16-causal-q4", 2, 4, 16, 64, [1024, 530], True),
    ("page16-one-tile", 2, 1, 16, 8, [128, 40], False),
]


@pytest.mark.parametrize(("name", "b", "s_q", "page", "max_pages", "lens", "causal"), _PAGED_CASES, ids=[c[0] for c in _PAGED_CASES])
def test_paged_two_kv_streams_match_the_reference(routed, name, b, s_q, page, max_pages, lens, causal):
    o, ref = _run_paged(b, 64, 8, s_q, page, max_pages, lens, causal_br=causal)
    assert not torch.isnan(o).any()
    err = (o - ref).abs().max().item() / ref.abs().max().item()
    assert err < 0.04, (name, err)
