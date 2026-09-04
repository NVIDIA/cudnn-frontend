# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sdpa_bwd_sm100_mxfp8``: the SM100 d=256 block-scale MXFP8 backward.

Every capability the row claims gets an ACCEPT test that runs the engine
end-to-end through the graph API (``sdpa_mxfp8_backward``) against the repo's
MXFP8 backward reference (``sdpa.mxfp8_ref.compute_ref_backward``, the same
oracle the cuDNN-backend MXFP8 tests use), and every capability it declines
gets a REJECT test that asserts the decline through the row's own
``mismatch()``.

The engine is a chain (SF repack -> dQ kernel -> fused dK/dV kernel), so the
accept tests are graph-level: the seams (scale-factor layouts, GQA head order,
LSE convention, workspace hand-off between the two kernels) are exactly where
a unit test of one launch would not look.
"""

from __future__ import annotations

import math

import pytest
import torch

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

import cudnn

# Levels are per test (pytest accumulates function and module markers, so a
# module-wide L0 would keep an L1 test inside `-m L0`): every test is L0 except
# the long-sequence case.
pytestmark = [requires_pre_rubin_blackwell, requires_dsl]

_ENGINE = "sdpa_bwd_sm100_mxfp8"
_D = 256
_BLOCK = 32
# dQ/dK carry the online-quantized dS (one E4M3 rounding the reference models
# too); dV rides a fixed 2^-8 P scale that matches the reference bit-for-bit
# in practice. The gate is on the whole tensor, so it is deliberately looser
# than the half-precision d512 row's.
_TOL_COS = 0.9995
_OUT = {"bf16": torch.bfloat16, "fp16": torch.float16}
_OUT_CUDNN = {torch.bfloat16: cudnn.data_type.BFLOAT16, torch.float16: cudnn.data_type.HALF}


def _cdiv(a, b):
    """Ceiling division."""
    return (a + b - 1) // b


def _bshd_stride(shape):
    """cuDNN declares logical BHSD; the engine needs BSHD-physical storage."""
    b, h, s, d = shape
    return [s * h * d, d, h * d, 1]


def _to_bshd(t_bhsd):
    """[B,H,S,D] (any storage) -> a [B,H,S,D] view over BSHD-physical memory."""
    return t_bhsd.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)


def _quantize(t, b, h, s, d):
    """Rowwise + columnwise MXFP8 of a [b,h,s,d] fp32 tensor, as the producer
    (TE-equivalent torch quantizer) hands them to cuDNN: fp8 payloads in the
    natural orientation, per-element dequant scales for the reference, and the
    F8_128x4-swizzled E8M0 bytes for the graph."""
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    d8, sf_d_ref, sf_d_swz, s8, sf_s_ref, sf_s_swz = quantize_to_mxfp8(t, b, h, s, d, _BLOCK, torch.float8_e4m3fn, with_ref=True)
    return dict(
        row=_to_bshd(d8),
        col=_to_bshd(s8),
        sf_row_ref=sf_d_ref,
        sf_col_ref=sf_s_ref,
        sf_row=sf_d_swz.contiguous(),
        sf_col=sf_s_swz.contiguous(),
    )


def _sf_dims(b, h, s, d):
    """Graph dims of the rowwise (scales along D) and columnwise (along S)
    F8_128x4 scale tensors, cuDNN's padding rules (rows to 128, blocks to 4)."""
    row = (b, h, _cdiv(s, 128) * 128, _cdiv(_cdiv(d, _BLOCK), 4) * 4)
    col = (b, h, _cdiv(_cdiv(s, _BLOCK), 4) * 4, _cdiv(d, 128) * 128)
    return row, col


def _build_graph(
    b,
    hq,
    hkv,
    sq,
    skv,
    d=_D,
    out_dt=torch.bfloat16,
    scale=None,
    fp8=cudnn.data_type.FP8_E4M3,
    grad_dt=None,
    stride_fn=_bshd_stride,
    declare_amax=False,
    with_sink=False,
    seq_len_dims=None,
    **sdpa_kwargs,
):
    """An ``sdpa_mxfp8_backward`` graph over BSHD-physical tensors. Returns the
    graph, its named tensors, and the (dQ, dK, dV, amax...) outputs."""
    io_half = _OUT_CUDNN[out_dt]
    grad_half = _OUT_CUDNN[grad_dt] if grad_dt is not None else io_half
    g = cudnn.pygraph(io_data_type=fp8, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    shq, shk = (b, hq, sq, d), (b, hkv, skv, d)
    t = {}
    for name, sh in (("q", shq), ("q_T", shq), ("k", shk), ("k_T", shk), ("v", shk), ("dO", shq), ("dO_T", shq)):
        t[name] = g.tensor(name=name, dim=list(sh), stride=stride_fn(sh), data_type=fp8)
    for name in ("o_f16", "dO_f16"):
        t[name] = g.tensor(name=name, dim=list(shq), stride=stride_fn(shq), data_type=io_half)
    t["stats"] = g.tensor(name="stats", dim=[b, hq, sq, 1], stride=[hq * sq, sq, 1, 1], data_type=cudnn.data_type.FLOAT)
    q_row, q_col = _sf_dims(b, hq, sq, d)
    k_row, k_col = _sf_dims(b, hkv, skv, d)
    for name, dims in (
        ("sf_q", q_row),
        ("sf_q_T", q_col),
        ("sf_k", k_row),
        ("sf_k_T", k_col),
        ("sf_v", k_row),
        ("sf_dO", q_row),
        ("sf_dO_T", q_col),
    ):
        t[name] = g.tensor(
            name=name,
            dim=list(dims),
            stride=[dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1],
            data_type=cudnn.data_type.FP8_E8M0,
            reordering_type=cudnn.tensor_reordering.F8_128x4,
        )
    if with_sink:
        t["sink"] = g.tensor(name="sink", dim=[1, hq, 1, 1], stride=[hq, 1, 1, 1], data_type=cudnn.data_type.FLOAT)
        t["dsink"] = g.tensor(name="dsink", dim=[1, hq, 1, 1], stride=[hq, 1, 1, 1], data_type=cudnn.data_type.FLOAT)
        sdpa_kwargs["sink_token"] = t["sink"]
        sdpa_kwargs["dSink_token"] = t["dsink"]
    if seq_len_dims is not None:
        t["seq_len_q"] = g.tensor(name="seq_len_q", dim=list(seq_len_dims), stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
        t["seq_len_kv"] = g.tensor(name="seq_len_kv", dim=list(seq_len_dims), stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
        sdpa_kwargs["seq_len_q"] = t["seq_len_q"]
        sdpa_kwargs["seq_len_kv"] = t["seq_len_kv"]
    if scale is not None:
        sdpa_kwargs["attn_scale"] = scale
    outs = g.sdpa_mxfp8_backward(
        name="bwd",
        q=t["q"],
        q_T=t["q_T"],
        k=t["k"],
        k_T=t["k_T"],
        v=t["v"],
        o_f16=t["o_f16"],
        dO_f16=t["dO_f16"],
        dO=t["dO"],
        dO_T=t["dO_T"],
        stats=t["stats"],
        descale_q=t["sf_q"],
        descale_q_T=t["sf_q_T"],
        descale_k=t["sf_k"],
        descale_k_T=t["sf_k_T"],
        descale_v=t["sf_v"],
        descale_dO=t["sf_dO"],
        descale_dO_T=t["sf_dO_T"],
        **sdpa_kwargs,
    )
    dq, dk, dv, amax_dq, amax_dk, amax_dv = outs
    for out, sh in ((dq, shq), (dk, shk), (dv, shk)):
        out.set_output(True).set_data_type(grad_half).set_dim(list(sh)).set_stride(stride_fn(sh))
    # The op returns the amax ports unconditionally and validate() wants their
    # dims; they stay virtual (not requested) unless declare_amax.
    for a in (amax_dq, amax_dk, amax_dv):
        a.set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
        if declare_amax:
            a.set_output(True)
    if with_sink:
        t["dsink"].set_output(True)
    g.validate()
    g.build_operation_graph()
    return g, t, (dq, dk, dv, amax_dq, amax_dk, amax_dv)


def _plan_index(g, name=_ENGINE):
    """Index of the FROST MXFP8 bwd plan in the graph's plan list."""
    for i in range(g.get_execution_plan_count()):
        pn = g.get_plan_name_at_index(i)
        if pn == name or pn.startswith(name + "["):
            return i
    return None


def _run(b=1, hq=2, hkv=None, sq=256, skv=256, out_dt=torch.bfloat16, causal=False, omit_scale=False, tol_cos=_TOL_COS, seed=0, **sdpa_kwargs):
    """Quantize random operands, build the graph, pin the engine, execute, and
    compare dQ/dK/dV against the MXFP8 backward reference."""
    from sdpa.mxfp8_ref import compute_ref, compute_ref_backward

    torch.manual_seed(seed)
    hkv = hq if hkv is None else hkv
    d, dev = _D, "cuda"
    scale = 1.0 / math.sqrt(d)
    q, dO = torch.randn(b, hq, sq, d, device=dev), torch.randn(b, hq, sq, d, device=dev)
    k, v = torch.randn(b, hkv, skv, d, device=dev), torch.randn(b, hkv, skv, d, device=dev)
    Q, K, V, DO = (_quantize(x, b, h, s, d) for x, h, s in ((q, hq, sq), (k, hkv, skv), (v, hkv, skv), (dO, hq, sq)))

    right = 0 if causal else None
    align = cudnn.diagonal_alignment.TOP_LEFT if causal else None
    o_ref, stats = compute_ref(
        Q["row"], K["row"], V["col"], Q["sf_row_ref"], K["sf_row_ref"], V["sf_col_ref"], scale, output_type=out_dt, right_bound=right, diag_align=align
    )
    o_f16 = _to_bshd(o_ref.to(out_dt))
    dO_f16 = _to_bshd(dO.to(out_dt))
    dq_r, dk_r, dv_r = compute_ref_backward(
        Q["row"],
        Q["col"],
        K["row"],
        K["col"],
        V["row"],
        o_f16,
        dO_f16,
        DO["row"],
        DO["col"],
        scale,
        Q["sf_row_ref"],
        Q["sf_col_ref"],
        K["sf_row_ref"],
        K["sf_col_ref"],
        V["sf_row_ref"],
        DO["sf_row_ref"],
        DO["sf_col_ref"],
        torch_otype=out_dt,
        right_bound=right,
        diag_align=align,
        stats=stats,
    )[:3]

    if causal:
        sdpa_kwargs["use_causal_mask"] = True
    g, t, (dq_t, dk_t, dv_t, *_amax) = _build_graph(b, hq, hkv, sq, skv, d, out_dt=out_dt, scale=None if omit_scale else scale, **sdpa_kwargs)
    g.create_execution_plans([cudnn.heur_mode.A])
    idx = _plan_index(g)
    assert idx is not None, f"{_ENGINE} not offered; plans = {[g.get_plan_name_at_index(i) for i in range(g.get_execution_plan_count())]}"
    g.select_plan(idx)
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    # NaN-poisoned outputs: a tile the kernels skip (e.g. the causal tail past
    # S_q) must still be WRITTEN, not inherited from the buffer.
    dq = _to_bshd(torch.full((b, hq, sq, d), float("nan"), device=dev, dtype=out_dt))
    dk = _to_bshd(torch.full((b, hkv, skv, d), float("nan"), device=dev, dtype=out_dt))
    dv = _to_bshd(torch.full((b, hkv, skv, d), float("nan"), device=dev, dtype=out_dt))
    pack = {
        t["q"]: Q["row"],
        t["q_T"]: Q["col"],
        t["k"]: K["row"],
        t["k_T"]: K["col"],
        t["v"]: V["row"],
        t["o_f16"]: o_f16,
        t["dO_f16"]: dO_f16,
        t["dO"]: DO["row"],
        t["dO_T"]: DO["col"],
        t["stats"]: stats.contiguous(),
        t["sf_q"]: Q["sf_row"],
        t["sf_q_T"]: Q["sf_col"],
        t["sf_k"]: K["sf_row"],
        t["sf_k_T"]: K["sf_col"],
        t["sf_v"]: V["sf_row"],
        t["sf_dO"]: DO["sf_row"],
        t["sf_dO_T"]: DO["sf_col"],
        dq_t: dq,
        dk_t: dk,
        dv_t: dv,
    }
    g.execute(pack, ws)
    torch.cuda.synchronize()
    for name, got, ref in (("dQ", dq, dq_r), ("dK", dk, dk_r), ("dV", dv, dv_r)):
        assert not torch.isnan(got).any(), f"{name} has NaN"
        cos = torch.nn.functional.cosine_similarity(got.float().flatten(), ref.flatten(), dim=0).item()
        assert cos > tol_cos, f"{name}: cos={cos:.6f}"


def _adapter_inputs(b, hq, hkv, sq, skv, out_dt=torch.bfloat16, causal=False, seed=0):
    """Quantized operands + reference gradients for driving the adapter directly
    (bypassing the graph), as a dict of BSHD-physical torch tensors."""
    from sdpa.mxfp8_ref import compute_ref, compute_ref_backward

    torch.manual_seed(seed)
    d, dev = _D, "cuda"
    scale = 1.0 / math.sqrt(d)
    q, dO = torch.randn(b, hq, sq, d, device=dev), torch.randn(b, hq, sq, d, device=dev)
    k, v = torch.randn(b, hkv, skv, d, device=dev), torch.randn(b, hkv, skv, d, device=dev)
    Q, K, V, DO = (_quantize(x, b, h, s, d) for x, h, s in ((q, hq, sq), (k, hkv, skv), (v, hkv, skv), (dO, hq, sq)))
    right = 0 if causal else None
    align = cudnn.diagonal_alignment.TOP_LEFT if causal else None
    o_ref, stats = compute_ref(
        Q["row"], K["row"], V["col"], Q["sf_row_ref"], K["sf_row_ref"], V["sf_col_ref"], scale, output_type=out_dt, right_bound=right, diag_align=align
    )
    o_f16 = _to_bshd(o_ref.to(out_dt))
    dO_f16 = _to_bshd(dO.to(out_dt))
    dq_r, dk_r, dv_r = compute_ref_backward(
        Q["row"],
        Q["col"],
        K["row"],
        K["col"],
        V["row"],
        o_f16,
        dO_f16,
        DO["row"],
        DO["col"],
        scale,
        Q["sf_row_ref"],
        Q["sf_col_ref"],
        K["sf_row_ref"],
        K["sf_col_ref"],
        V["sf_row_ref"],
        DO["sf_row_ref"],
        DO["sf_col_ref"],
        torch_otype=out_dt,
        right_bound=right,
        diag_align=align,
        stats=stats,
    )[:3]
    sf_i8 = lambda t: t.view(torch.int8)  # noqa: E731
    return dict(
        q=Q["row"],
        q_T=Q["col"],
        k=K["row"],
        k_T=K["col"],
        v=V["row"],
        o=o_f16,
        dO_f16=dO_f16,
        dO=DO["row"],
        dO_T=DO["col"],
        stats=stats.contiguous(),
        sf_q=sf_i8(Q["sf_row"]),
        sf_q_T=sf_i8(Q["sf_col"]),
        sf_k=sf_i8(K["sf_row"]),
        sf_k_T=sf_i8(K["sf_col"]),
        sf_v=sf_i8(V["sf_row"]),
        sf_dO=sf_i8(DO["sf_row"]),
        sf_dO_T=sf_i8(DO["sf_col"]),
        dq=_to_bshd(torch.zeros(b, hq, sq, d, device=dev, dtype=out_dt)),
        dk=_to_bshd(torch.zeros(b, hkv, skv, d, device=dev, dtype=out_dt)),
        dv=_to_bshd(torch.zeros(b, hkv, skv, d, device=dev, dtype=out_dt)),
        ref=(dq_r, dk_r, dv_r),
        scale=scale,
        causal=causal,
    )


def _adapter_shapes(b, hq, hkv, sq, skv, out_dt=torch.bfloat16, causal=False):
    """Shape-only operands (uninitialized, no reference) for probing the adapter's
    plan-time decisions without paying for a reference computation."""
    d, dev = _D, "cuda"
    fp8 = lambda h, s: _to_bshd(torch.empty(b, h, s, d, device=dev, dtype=torch.float8_e4m3fn))  # noqa: E731
    half = lambda h, s: _to_bshd(torch.empty(b, h, s, d, device=dev, dtype=out_dt))  # noqa: E731
    q_row, q_col = _sf_dims(b, hq, sq, d)
    k_row, k_col = _sf_dims(b, hkv, skv, d)
    sf = lambda dims: torch.empty(dims, device=dev, dtype=torch.int8)  # noqa: E731
    return dict(
        q=fp8(hq, sq),
        q_T=fp8(hq, sq),
        k=fp8(hkv, skv),
        k_T=fp8(hkv, skv),
        v=fp8(hkv, skv),
        o=half(hq, sq),
        dO_f16=half(hq, sq),
        dO=fp8(hq, sq),
        dO_T=fp8(hq, sq),
        stats=torch.empty(b, hq, sq, 1, device=dev),
        sf_q=sf(q_row),
        sf_q_T=sf(q_col),
        sf_k=sf(k_row),
        sf_k_T=sf(k_col),
        sf_v=sf(k_row),
        sf_dO=sf(q_row),
        sf_dO_T=sf(q_col),
        dq=half(hq, sq),
        dk=half(hkv, skv),
        dv=half(hkv, skv),
        ref=None,
        scale=1.0 / math.sqrt(d),
        causal=causal,
    )


def _make_adapter(inp, **kw):
    """Construct the SdpaBwdDslSm100Mxfp8 adapter for prepared inputs, forwarding split overrides."""
    from cudnn.sdpa.bwd.api_dsl_mxfp8_sm100 import SdpaBwdDslSm100Mxfp8

    return SdpaBwdDslSm100Mxfp8(
        inp["q"],
        inp["k"],
        inp["v"],
        inp["o"],
        inp["dO"],
        inp["stats"],
        inp["dq"],
        inp["dk"],
        inp["dv"],
        sample_q_T=inp["q_T"],
        sample_k_T=inp["k_T"],
        sample_do_T=inp["dO_T"],
        sample_do_f16=inp["dO_f16"],
        sample_sf_q=inp["sf_q"],
        sample_sf_q_T=inp["sf_q_T"],
        sample_sf_k=inp["sf_k"],
        sample_sf_k_T=inp["sf_k_T"],
        sample_sf_v=inp["sf_v"],
        sample_sf_do=inp["sf_dO"],
        sample_sf_do_T=inp["sf_dO_T"],
        is_causal=inp["causal"],
        scale_softmax=inp["scale"],
        **kw,
    )


def _run_adapter(api, inp):
    """Run the adapter directly (no graph) and return dQ/dK/dV."""
    api.check_support()
    api.compile()
    ws = torch.empty(max(api.scratch_workspace_bytes(), 1), device="cuda", dtype=torch.uint8)
    for t in (inp["dq"], inp["dk"], inp["dv"]):
        t.fill_(float("nan"))
    api.execute(
        q_tensor=inp["q"],
        k_tensor=inp["k"],
        v_tensor=inp["v"],
        o_tensor=inp["o"],
        do_tensor=inp["dO"],
        stats_tensor=inp["stats"],
        dq_tensor=inp["dq"],
        dk_tensor=inp["dk"],
        dv_tensor=inp["dv"],
        workspace=ws,
        q_T_tensor=inp["q_T"],
        k_T_tensor=inp["k_T"],
        do_T_tensor=inp["dO_T"],
        do_f16_tensor=inp["dO_f16"],
        sf_q=inp["sf_q"],
        sf_q_T=inp["sf_q_T"],
        sf_k=inp["sf_k"],
        sf_k_T=inp["sf_k_T"],
        sf_v=inp["sf_v"],
        sf_do=inp["sf_dO"],
        sf_do_T=inp["sf_dO_T"],
    )
    torch.cuda.synchronize()
    for name, got, ref in zip(("dQ", "dK", "dV"), (inp["dq"], inp["dk"], inp["dv"]), inp["ref"]):
        assert not torch.isnan(got).any(), f"{name} has NaN"
        cos = torch.nn.functional.cosine_similarity(got.float().flatten(), ref.flatten(), dim=0).item()
        assert cos > _TOL_COS, f"{name}: cos={cos:.6f}"
    return tuple(t.clone() for t in (inp["dq"], inp["dk"], inp["dv"]))


# --------------------------------------------------------------------------- #
# ACCEPT -- every capability the row claims                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.L0
@pytest.mark.parametrize("out", list(_OUT), ids=list(_OUT))
def test_dense(out):
    """Dense (no mask) shapes across batch/head/seqlen combinations."""
    _run(out_dt=_OUT[out])


@pytest.mark.L0
def test_default_attn_scale():
    """attn_scale is optional on the graph; the engine must default to 1/sqrt(d)."""
    _run(omit_scale=True)


@pytest.mark.L0
def test_causal():
    """Top-left causal mask, S_q == S_kv."""
    _run(causal=True)


@pytest.mark.L0
def test_gqa():
    """Grouped-query attention with a small KV-head count."""
    _run(hq=4, hkv=2)


@pytest.mark.L0
def test_mqa():
    """Multi-query attention (one KV head)."""
    _run(hq=4, hkv=1)


@pytest.mark.L0
def test_batch_and_heads():
    """Multiple (batch, head) planes: the columnwise scale factors interleave
    head planes inside each 128-row D tile at d=256 (see the repack module);
    a single plane cannot catch a wrong plane stride."""
    _run(b=2, hq=2, hkv=2, sq=256, skv=384, out_dt=torch.float16)


@pytest.mark.L0
def test_ragged_tails():
    """S_q and S_kv that are not tile multiples: the kernels mask the tail."""
    _run(sq=129, skv=160)


@pytest.mark.L0
@pytest.mark.L0
@pytest.mark.parametrize("skv", [160, 96])
def test_ragged_kv_only(skv):
    """Aligned S_q with a ragged S_kv (including a sub-tile one) runs on the
    window-mask path: the KV tail is covered by the kv trip count and masked
    there, so the residual mask is only needed for a Q-side tail."""
    _run(hq=2, hkv=1, sq=256, skv=skv)


@pytest.mark.L0
def test_ragged_tails_causal_gqa():
    """Causal GQA with S_q and S_kv not multiples of the tile size."""
    _run(hq=4, hkv=2, sq=129, skv=160, causal=True)


@pytest.mark.L0
def test_causal_kv_tail():
    """Causal with S_kv > S_q: the KV tiles past S_q are attended by no query row,
    their gradients are zero, and the kernel must WRITE that zero (it skips the
    mainloop for them). Outputs start NaN-poisoned, so an unwritten tile fails."""
    _run(hq=4, hkv=2, sq=256, skv=640, causal=True)


@pytest.mark.L0
def test_q_tile_boundary_causal():
    """Causal with S_q exactly on / one past a Q-tile boundary."""
    _run(hq=8, hkv=2, sq=257, skv=256, causal=True)


@pytest.mark.L0
def test_decode_shaped():
    """S_q == 1: the dQ kernel narrows its store to one element."""
    _run(sq=1, skv=128)


@pytest.mark.L0
def test_deterministic_flag():
    """Both kernels own their output tiles (no atomics), so the row honors
    use_deterministic_algorithm; the result must be bitwise stable."""
    _run(use_deterministic_algorithm=True)


@pytest.mark.L1
def test_long_causal():
    """Long causal sequence (L1) to exercise many KV iterations."""
    _run(b=1, hq=2, hkv=2, sq=2048, skv=2048, causal=True)


@pytest.mark.L0
def test_dkdv_head_split_heuristic():
    """The fused dK/dV kernel's grid is (KV tiles, KV heads, B); with few KV
    heads the adapter splits the Q-head group (a divisor of the group, the
    smallest reaching ~1 cluster per SM). The sequence split is opt-in only, and
    a large MHA shape needs neither -- no partials, no fold. Shape-only inputs:
    nothing is computed."""
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    api = _make_adapter(_adapter_shapes(1, 32, 2, 2048, 2048))
    kv_clusters = 16 * 2  # 2048/128 tiles x 2 KV heads x B=1
    assert 16 % api._dkdv_head == 0 and api._dkdv_head > 1
    assert kv_clusters * api._dkdv_head >= sms or api._dkdv_head == 16
    assert kv_clusters * (api._dkdv_head // 2) < sms  # the smallest such divisor
    assert api._dkdv_seq == 1
    # MHA, small batch, short sequence: no group to split; the sequence split is
    # never chosen automatically.
    api = _make_adapter(_adapter_shapes(1, 2, 2, 2048, 2048))
    assert (api._dkdv_head, api._dkdv_seq) == (1, 1)
    assert _make_adapter(_adapter_shapes(1, 2, 2, 2048, 2048), dkdv_seq_split=4)._dkdv_seq == 4
    # Large MHA: the grid alone fills the machine.
    api = _make_adapter(_adapter_shapes(4, 16, 16, 8192, 8192))
    assert (api._dkdv_head, api._dkdv_seq) == (1, 1)
    with pytest.raises(ValueError):
        _make_adapter(_adapter_shapes(1, 32, 2, 2048, 2048), dkdv_head_split=3)  # not a divisor of 16


@pytest.mark.L0
@pytest.mark.parametrize("split", [1, 2, 8])
def test_dkdv_head_split_forced(split):
    """Every split writes per-slice partials that a fixed-order fold sums; the
    result must match the reference (and the unsplit kernel) for each split."""
    inp = _adapter_inputs(1, 8, 1, 256, 384, causal=True)
    got = _run_adapter(_make_adapter(inp, dkdv_head_split=split), inp)
    if split > 1:
        base = _run_adapter(_make_adapter(inp, dkdv_head_split=1), inp)
        for name, a, b_ in zip(("dQ", "dK", "dV"), got, base):
            cos = torch.nn.functional.cosine_similarity(a.float().flatten(), b_.float().flatten(), dim=0).item()
            assert cos > 0.99999, f"{name} split={split} vs unsplit: cos={cos:.7f}"


@pytest.mark.L0
@pytest.mark.parametrize("seq_split", [2, 3, 8])
def test_dkdv_seq_split_forced(seq_split):
    """Q-sequence chunks per KV tile (3 does not divide the 4 Q tiles: uneven
    chunks; 8 exceeds them: empty chunks must still write zeros), causal so
    the per-tile ranges differ, MHA so only the sequence split is at work."""
    inp = _adapter_inputs(1, 2, 2, 512, 384, causal=True)
    got = _run_adapter(_make_adapter(inp, dkdv_seq_split=seq_split), inp)
    base = _run_adapter(_make_adapter(inp, dkdv_seq_split=1), inp)
    for name, a, b_ in zip(("dQ", "dK", "dV"), got, base):
        cos = torch.nn.functional.cosine_similarity(a.float().flatten(), b_.float().flatten(), dim=0).item()
        assert cos > 0.99999, f"{name} seq_split={seq_split} vs unsplit: cos={cos:.7f}"


@pytest.mark.L0
def test_workspace_accounts_for_repack_buffers():
    """The SF repack buffers are carved from the caller's workspace; the plan
    must ask for them (a too-small request would corrupt memory, not fail)."""
    b, hq, sq = 1, 2, 256
    g, _t, _outs = _build_graph(b, hq, hq, sq, sq, scale=1.0 / math.sqrt(_D))
    g.create_execution_plans([cudnn.heur_mode.A])
    idx = _plan_index(g)
    assert idx is not None
    g.select_plan(idx)
    g.check_support()
    g.build_plans()
    from cudnn.sdpa.bwd.kernels.bprop_sf_repack_mxfp8_sm100 import SF_LAYOUT_SFA, SF_LAYOUT_SFB, repack_geometry

    l = b * hq
    # 4 rowwise-A + 5 rowwise/columnwise-B + 2 columnwise-B buffers (see _sf_plan)
    sfa = repack_geometry(sq, _D // 32, l, SF_LAYOUT_SFA)[3]
    sfb = repack_geometry(sq, _D // 32, l, SF_LAYOUT_SFB)[3]
    sfb_t = repack_geometry(_D, sq // 32, l, SF_LAYOUT_SFB)[3]
    assert g.get_workspace_size() >= 4 * sfa + 4 * sfb + 3 * sfb_t


# --------------------------------------------------------------------------- #
# REJECT -- every capability the row declines                                  #
# --------------------------------------------------------------------------- #


def _decline_reason(**kw):
    """Why this engine declines the graph, or None if it would serve it. Asks the
    row's own ``mismatch()`` (the plan list also holds backend plans, which
    confound "my engine is absent")."""
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS, mismatch

    knobs = kw.pop("knobs", None)
    b, hq, hkv = kw.pop("b", 1), kw.pop("hq", 2), kw.pop("hkv", 2)
    sq, skv, d = kw.pop("sq", 256), kw.pop("skv", 256), kw.pop("d", _D)
    spec = next(s for s in ENGINE_SPECS if s.name == _ENGINE)
    try:
        g, _t, _outs = _build_graph(b, hq, hkv, sq, skv, d, scale=1.0 / math.sqrt(d), **kw)
    except (cudnn.cudnnGraphNotSupportedError, RuntimeError) as e:
        return f"frontend refused the graph: {e}"
    facts = ga.analyze(g)
    if facts is None:
        return "analyzer did not recognise the graph"
    return mismatch(spec.capabilities, facts, knobs)


@pytest.mark.L0
def test_accepts_the_claimed_dense_graph():
    """Sanity for the reject tests: the plain graph IS served."""
    assert _decline_reason() is None


@pytest.mark.L0
@pytest.mark.parametrize("d", [128, 192, 512])
def test_reject_other_head_dims(d):
    """Exact d=256 only: the SF plumbing has no envelope story."""
    assert _decline_reason(d=d) is not None


@pytest.mark.L0
def test_reject_e5m2_payloads():
    """e5m2 fp8 payloads are declined."""
    assert _decline_reason(fp8=cudnn.data_type.FP8_E5M2) is not None


@pytest.mark.L0
def test_reject_gradient_dtype_mismatch():
    """o_f16/dO_f16/dQ/dK/dV share one half dtype; the kernels have one element type."""
    assert _decline_reason(out_dt=torch.bfloat16, grad_dt=torch.float16) is not None


@pytest.mark.L0
def test_reject_bottom_right_causal():
    """Bottom-right causal is declined."""
    assert _decline_reason(use_causal_mask_bottom_right=True) is not None


@pytest.mark.L0
def test_reject_right_band_widening():
    """Right-band widening masks are declined."""
    assert _decline_reason(right_bound=16) is not None


@pytest.mark.L0
def test_reject_sliding_window():
    """Sliding-window masks are declined."""
    assert _decline_reason(use_causal_mask=True, left_bound=64) is not None


@pytest.mark.L0
def test_reject_padding_mask():
    """Padding masks are declined."""
    assert _decline_reason(use_padding_mask=True, seq_len_dims=(1, 1, 1, 1)) is not None


@pytest.mark.L0
def test_reject_amax_outputs():
    """The kernels write half gradients and never produce amax_dQ/dK/dV; a graph
    asking for one would otherwise get garbage in that buffer."""
    assert _decline_reason(declare_amax=True) is not None


@pytest.mark.L0
def test_reject_sink():
    """Attention sinks are declined."""
    assert _decline_reason(with_sink=True) is not None


@pytest.mark.L0
def test_reject_non_bshd_layout():
    """The kernels derive their head/batch strides from BSHD-compact storage; a
    BHSD-contiguous graph is declined, not staged (Hard Rule 2)."""

    def bhsd(sh):
        """BHSD-strided tensor helper for the sink test."""
        b, h, s, d = sh
        return [h * s * d, s * d, d, 1]

    assert _decline_reason(stride_fn=bhsd) is not None


@pytest.mark.L0
def test_reject_gqa_ratio_not_integer():
    """Non-integer H_q / H_kv ratios are declined."""
    assert _decline_reason(hq=6, hkv=4) is not None


@pytest.mark.L0
def test_reject_foreign_tiles():
    """Scale-factor tiles of a foreign layout are declined."""
    from cudnn.sdpa.bwd.engines import SdpaBwdKnobs

    assert _decline_reason(knobs=SdpaBwdKnobs(tile_m=64)) is not None


@pytest.mark.L0
def test_half_row_declines_mxfp8_graph():
    """The half-precision d512 row must not claim an MXFP8 backward (it used to
    hard-decline every quantized graph; now the family match does it)."""
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS, mismatch

    g, _t, _outs = _build_graph(1, 2, 2, 256, 256, scale=1.0 / math.sqrt(_D))
    facts = ga.analyze(g)
    assert facts is not None and facts.is_mxfp8 and facts.is_backward
    for spec in ENGINE_SPECS:
        if spec.name != _ENGINE:
            assert mismatch(spec.capabilities, facts) is not None, spec.name


@pytest.mark.L0
def test_capabilities_match_what_is_implemented():
    """The row must not claim anything the adapter would refuse at build."""
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == _ENGINE)
    c = spec.capabilities
    assert c.is_mxfp8 and not c.is_fp8
    assert c.d == frozenset({256}) and not c.d_envelope
    assert c.dtypes == frozenset({cudnn.data_type.FP8_E4M3})
    assert c.out_dtypes == frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16})
    assert c.causal and c.gqa and c.deterministic
    assert not (c.bottom_right or c.swa or c.right_band_widening or c.padded or c.thd or c.cu_seq_len)
    assert not (c.bias or c.dbias or c.sink or c.dsink or c.amax_dgrad)
    assert c.layouts == frozenset({"bshd"})
    assert c.tile_ms == frozenset({128}) and c.tile_ns == frozenset({128})
    assert c.sm_lo == 100 and c.sm_hi == 106


@pytest.mark.L0
def test_engine_is_registered_and_opt_in():
    """The engine is in the manifest and only used with the FROST opt-in env."""
    from cudnn.engines.manifest import MANIFEST

    fam = next(f for f in MANIFEST if f.name == "frost_sdpa_bwd")
    assert _ENGINE in fam.slots
    assert fam.slots[_ENGINE].opt_in, "new engines stay opt-in until they earn arch coverage + benchmarks"
