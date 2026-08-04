# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""End-to-end tests for the FROST SM100 DSL block-scale MXFP8 SDPA-forward engine.

Drives ``graph.sdpa_mxfp8`` (FP8 E4M3/E5M2 Q/K/V + per-32-block E8M0 scale factors)
routed to the ``sdpa_fwd_prefill_sm100_d128_mxfp8`` engine, and validates the output
against an fp32-dequant reference. Inputs are quantized with the torch-only
MXFP8 quantizer in ``test/python/sdpa/mxfp8_quant.py`` (TE-equivalent semantics
— the same layout cuDNN's own mxfp8 path uses), so the scale-factor tensors
reach the engine in cuDNN's F8_128x4 reordering. TransformerEngine is NOT
required.

cuDNN's ``sdpa_mxfp8`` op exposes causal / bottom-right / sliding-window masks +
attention sink, but NOT a padding mask or THD/ragged inputs — so those are out of
scope here (and the engine declares thd=False to match).

Requires: SM100 (Blackwell), cutlass-dsl, cuDNN >= 9.21 (mxfp8 support).
Skips cleanly otherwise.
"""

import math

import pytest
import torch

from test_utils import torch_fork_set_rng

from cudnn.sdpa.fwd.engines import engine_name


def _is_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    # The MXFP8 prefill engine supports both Blackwell arches (cc10.0 and cc10.3);
    # on cc10.3 the fused-LDTM statistics path (tcgen05.ld.red.f32.max) auto-engages.
    return torch.cuda.get_device_capability(torch.cuda.current_device()) in ((10, 0), (10, 3))


def _deps_available() -> bool:
    try:
        import cutlass  # noqa: F401
    except ImportError:
        return False
    return True


def _select_engine(graph, name):
    """Pin the ranked entry named ``name`` (graph.plans holds the backend's
    plans and the python engines' in one list). A pin is strict: check_support /
    build_plans raise if that engine declines the graph."""
    names = [graph.get_plan_name_at_index(i) for i in range(len(graph.plans))]
    assert name in names, f"engine {name!r} did not claim this graph; plans={names}"
    graph.select_plan(names.index(name))
    return graph


pytestmark = pytest.mark.skipif(
    not (_is_sm100() and _deps_available()),
    reason="MXFP8 SDPA engine requires SM100 + cutlass-dsl.",
)


_FP8 = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
_OUT = {"fp16": torch.float16, "bf16": torch.bfloat16, "e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
_CUDNN_ITYPE = {"e4m3": "FP8_E4M3", "e5m2": "FP8_E5M2"}
_CUDNN_OTYPE = {torch.float16: "HALF", torch.bfloat16: "BFLOAT16", torch.float8_e4m3fn: "FP8_E4M3", torch.float8_e5m2: "FP8_E5M2"}
_BLOCK = 32


def _cdiv(a, b):
    return (a + b - 1) // b


def _quantize(t, b, h, s, d, fp8, *, columnwise):
    """MXFP8-quantize [b,h,s,d] fp32 → (fp8_data[b,h,s,d], swizzled SF, per-elem dequant
    scale[b,h,s,d], (scale_padded, dblock_padded)). columnwise=True scales along S (V),
    else along D (Q/K). Torch-only (sdpa.mxfp8_quant), TE-equivalent semantics."""
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    d_scale_pad = _cdiv(_cdiv(d, _BLOCK), 4) * 4
    d_pad = d_scale_pad * _BLOCK
    s_scale_pad = _cdiv(_cdiv(s, _BLOCK), 4) * 4
    s_pad = s_scale_pad * _BLOCK
    data_d, dq_d, swz_d, data_s, dq_s, swz_s = quantize_to_mxfp8(t, b, h, s, d, _BLOCK, fp8, with_ref=True)
    if columnwise:
        dq = dq_s.reshape(b, h, s, d)
        return data_s, swz_s, dq, (s_scale_pad, d_pad)
    dq = dq_d.reshape(b, h, s, d)
    return data_d, swz_d, dq, (s_pad, d_scale_pad)


def _ref(qd, kd, vd, *, scale, is_causal=False, bottom_right=False, swa_window=None, sinks=None):
    """fp32 reference matching the kernel's mask + sink semantics (BHSD; GQA-aware)."""
    b, h_q, s_q, _ = qd.shape
    _, h_kv, s_kv, _ = vd.shape
    dev = qd.device
    g = h_q // h_kv
    k_e = kd.repeat_interleave(g, dim=1)
    v_e = vd.repeat_interleave(g, dim=1)
    scores = torch.matmul(qd, k_e.transpose(-1, -2)) * scale
    i = torch.arange(s_q, device=dev).view(1, 1, s_q, 1)
    j = torch.arange(s_kv, device=dev).view(1, 1, 1, s_kv)
    masked = torch.zeros(1, 1, s_q, s_kv, dtype=torch.bool, device=dev)
    if is_causal:
        lim = i + (s_kv - s_q) if bottom_right else i
        masked = masked | (j > lim)
    if swa_window is not None:
        masked = masked | (j < i - swa_window)
    scores = scores.masked_fill(masked, float("-inf"))
    if sinks is not None:
        col = sinks.view(1, h_q, 1, 1).float().expand(b, h_q, s_q, 1).to(dev)
        probs = torch.softmax(torch.cat([scores, col], dim=-1), dim=-1)
        return torch.matmul(probs[..., :s_kv], v_e)
    return torch.matmul(torch.softmax(scores, dim=-1), v_e)


def _run(B, H_q, H_kv, S, in_key, out_dt, *, scale, sdpa_kwargs, sink=None):
    """Quantize, build the sdpa_mxfp8 graph, route to the frost engine, execute.
    Returns (O_frost [B,H,S,D] on device, O_ref fp32 [B,H,S,D])."""
    import cudnn

    dev = "cuda"
    D = 128
    fp8 = _FP8[in_key]
    Qf = torch.randn(B, H_q, S, D, device=dev) * 0.5
    Kf = torch.randn(B, H_kv, S, D, device=dev) * 0.5
    Vf = torch.randn(B, H_kv, S, D, device=dev) * 0.5
    Q8, sfq, dqq, (sqp, dsc) = _quantize(Qf, B, H_q, S, D, fp8, columnwise=False)
    K8, sfk, dqk, (skp, _) = _quantize(Kf, B, H_kv, S, D, fp8, columnwise=False)
    V8, sfv, dqv, (ssc, dvp) = _quantize(Vf, B, H_kv, S, D, fp8, columnwise=True)

    def bshd(x8):
        return x8.permute(0, 2, 1, 3).contiguous().transpose(1, 2)

    Qb, Kb, Vb = bshd(Q8), bshd(K8), bshd(V8)
    Ob = torch.empty(B, S, H_q, D, device=dev, dtype=out_dt).transpose(1, 2)
    lse = torch.empty(B, H_q, S, 1, device=dev, dtype=torch.float32)
    amax = torch.zeros(1, 1, 1, 1, device=dev, dtype=torch.float32)
    sfq_g = sfq.view(torch.uint8).reshape(B, H_q, sqp, dsc)
    sfk_g = sfk.view(torch.uint8).reshape(B, H_kv, skp, dsc)
    sfv_g = sfv.view(torch.uint8).reshape(B, H_kv, ssc, dvp)

    itype = getattr(cudnn.data_type, _CUDNN_ITYPE[in_key])
    otype = getattr(cudnn.data_type, _CUDNN_OTYPE[out_dt])
    g = cudnn.pygraph(io_data_type=itype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q = g.tensor_like(Qb)
    k = g.tensor_like(Kb)
    v = g.tensor_like(Vb)

    def _sf(dims):
        return g.tensor(
            dim=list(dims),
            stride=[dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1],
            data_type=cudnn.data_type.FP8_E8M0,
            reordering_type=cudnn.tensor_reordering.F8_128x4,
        )

    dq = _sf((B, H_q, sqp, dsc))
    dk = _sf((B, H_kv, skp, dsc))
    dv = _sf((B, H_kv, ssc, dvp))
    kw = dict(q=q, k=k, v=v, descale_q=dq, descale_k=dk, descale_v=dv, attn_scale=scale, generate_stats=True)
    vp = {q: Qb, k: Kb, v: Vb, dq: sfq_g, dk: sfk_g, dv: sfv_g}
    if sink is not None:
        st = g.tensor_like(sink)
        kw["sink_token"] = st
        vp[st] = sink
    kw.update(sdpa_kwargs)
    o, stats, amax_o = g.sdpa_mxfp8(**kw)
    o.set_output(True).set_dim(list(Ob.shape)).set_stride(list(Ob.stride())).set_data_type(otype)
    stats.set_output(True).set_dim([B, H_q, S, 1]).set_stride([H_q * S, S, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    amax_o.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    _select_engine(g, engine_name(128, mxfp8=True))
    g.check_support()
    g.build_plans()
    vp.update({o: Ob, stats: lse, amax_o: amax})
    g.execute(vp, torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8))
    torch.cuda.synchronize()

    ref_kwargs = {k2: v2 for k2, v2 in _ref_from_sdpa(sdpa_kwargs).items()}
    if sink is not None:
        ref_kwargs["sinks"] = sink.flatten()
    o_ref = _ref(Q8.float() * dqq, K8.float() * dqk, V8.float() * dqv, scale=scale, **ref_kwargs)
    return Ob, o_ref, amax


def _ref_from_sdpa(sdpa_kwargs):
    """Translate the sdpa_mxfp8 mask kwargs into _ref() kwargs."""
    out = {}
    if sdpa_kwargs.get("use_causal_mask"):
        out["is_causal"] = True
    if sdpa_kwargs.get("use_causal_mask_bottom_right"):
        out["is_causal"] = True
        out["bottom_right"] = True
    lb = sdpa_kwargs.get("diagonal_band_left_bound")
    if lb is not None:
        out["swa_window"] = lb - 1  # cuDNN length - 1
    return out


def _check(O, O_ref, out_dt, in_key=None):
    """Compare frost output to fp32 ref; for FP8 output, gauge against the fp8-quant floor.

    E5M2 inputs (2-bit mantissa) are noisier than E4M3 (3-bit), so the half-output
    tolerance is widened for them.
    """
    atol_half = 7e-2 if in_key == "e5m2" else 5e-2
    diff = (O.float() - O_ref).abs().max().item()
    if out_dt in (torch.float8_e4m3fn, torch.float8_e5m2):
        floor = (O_ref - O_ref.to(out_dt).float()).abs().max().item()
        tol = max(atol_half, 3.0 * floor)
    else:
        tol = atol_half
    assert diff <= tol, f"max|O-ref|={diff:.4f} exceeds tol={tol:.4f}"


_INS = ["e4m3", "e5m2"]
_MASKS = {
    "none": {},
    "causal": dict(use_causal_mask=True),
    "causal_br": dict(use_causal_mask_bottom_right=True),
    "swa": dict(use_causal_mask=True, diagonal_band_left_bound=65),  # window = 64
}


@pytest.mark.L0
@pytest.mark.parametrize("in_key", _INS)
@pytest.mark.parametrize("mask", list(_MASKS))
@torch_fork_set_rng(seed=0)
def test_mxfp8_masks(in_key, mask):
    """none / causal / bottom-right / SWA masks (half output)."""
    B, H, S = 2, 8, 256
    if mask == "causal_br":
        # bottom-right needs S_q != S_kv to be distinct from top-left; keep square here
        pass
    scale = 1.0 / math.sqrt(128)
    O, O_ref, _ = _run(B, H, H, S, in_key, torch.float16, scale=scale, sdpa_kwargs=_MASKS[mask])
    _check(O, O_ref, torch.float16, in_key)


@pytest.mark.L0
@pytest.mark.parametrize("out_key", ["fp16", "bf16", "e4m3", "e5m2"])
@pytest.mark.parametrize("in_key", _INS)
@torch_fork_set_rng(seed=0)
def test_mxfp8_output_dtypes(in_key, out_key):
    """FP8 in (E4M3/E5M2) → {FP16, BF16, E4M3, E5M2} out."""
    scale = 1.0 / math.sqrt(128)
    O, O_ref, amax = _run(1, 8, 8, 512, in_key, _OUT[out_key], scale=scale, sdpa_kwargs=dict(use_causal_mask=True))
    _check(O, O_ref, _OUT[out_key], in_key)
    assert amax.item() > 0.0  # Amax_O produced


@pytest.mark.L0
@pytest.mark.parametrize("in_key", _INS)
@torch_fork_set_rng(seed=0)
def test_mxfp8_sink(in_key):
    """Causal + attention sink (per-Q-head logit in the softmax denominator)."""
    B, H, S = 2, 8, 256
    scale = 1.0 / math.sqrt(128)
    sink = torch.randn(1, H, 1, 1, dtype=torch.float32, device="cuda")
    O, O_ref, _ = _run(B, H, H, S, in_key, torch.float16, scale=scale, sdpa_kwargs=dict(use_causal_mask=True), sink=sink)
    _check(O, O_ref, torch.float16, in_key)


@pytest.mark.L0
@pytest.mark.parametrize("in_key", _INS)
@torch_fork_set_rng(seed=0)
def test_mxfp8_gqa(in_key):
    """GQA: H_q=8, H_kv=2 (K/V shared across groups), causal."""
    B, S = 2, 256
    scale = 1.0 / math.sqrt(128)
    O, O_ref, _ = _run(B, 8, 2, S, in_key, torch.float16, scale=scale, sdpa_kwargs=dict(use_causal_mask=True))
    _check(O, O_ref, torch.float16, in_key)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_mxfp8_bottom_right_rectangular():
    """Bottom-right causal with S_q != S_kv (the case where it differs from top-left)."""
    scale = 1.0 / math.sqrt(128)
    # S must be a multiple of TILE_N (128) for the non-padded path; use S_q=128, S_kv=256.
    import cudnn  # noqa: F401

    O, O_ref, _ = _run_rect(2, 8, 128, 256, "e4m3", torch.float16, scale=scale, sdpa_kwargs=dict(use_causal_mask_bottom_right=True))
    _check(O, O_ref, torch.float16, "e4m3")


def _run_rect(B, H, S_q, S_kv, in_key, out_dt, *, scale, sdpa_kwargs):
    """_run variant allowing S_q != S_kv (bottom-right causal)."""
    import cudnn

    dev = "cuda"
    D = 128
    fp8 = _FP8[in_key]
    Qf = torch.randn(B, H, S_q, D, device=dev) * 0.5
    Kf = torch.randn(B, H, S_kv, D, device=dev) * 0.5
    Vf = torch.randn(B, H, S_kv, D, device=dev) * 0.5
    Q8, sfq, dqq, (sqp, dsc) = _quantize(Qf, B, H, S_q, D, fp8, columnwise=False)
    K8, sfk, dqk, (skp, _) = _quantize(Kf, B, H, S_kv, D, fp8, columnwise=False)
    V8, sfv, dqv, (ssc, dvp) = _quantize(Vf, B, H, S_kv, D, fp8, columnwise=True)

    def bshd(x8):
        return x8.permute(0, 2, 1, 3).contiguous().transpose(1, 2)

    Qb, Kb, Vb = bshd(Q8), bshd(K8), bshd(V8)
    Ob = torch.empty(B, S_q, H, D, device=dev, dtype=out_dt).transpose(1, 2)
    lse = torch.empty(B, H, S_q, 1, device=dev, dtype=torch.float32)
    amax = torch.zeros(1, 1, 1, 1, device=dev, dtype=torch.float32)
    g = cudnn.pygraph(
        io_data_type=getattr(cudnn.data_type, _CUDNN_ITYPE[in_key]), intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT
    )
    q = g.tensor_like(Qb)
    k = g.tensor_like(Kb)
    v = g.tensor_like(Vb)

    def _sf(dims):
        return g.tensor(
            dim=list(dims),
            stride=[dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1],
            data_type=cudnn.data_type.FP8_E8M0,
            reordering_type=cudnn.tensor_reordering.F8_128x4,
        )

    dq = _sf((B, H, sqp, dsc))
    dk = _sf((B, H, skp, dsc))
    dv = _sf((B, H, ssc, dvp))
    o, stats, amax_o = g.sdpa_mxfp8(q=q, k=k, v=v, descale_q=dq, descale_k=dk, descale_v=dv, attn_scale=scale, generate_stats=True, **sdpa_kwargs)
    o.set_output(True).set_dim(list(Ob.shape)).set_stride(list(Ob.stride())).set_data_type(getattr(cudnn.data_type, _CUDNN_OTYPE[out_dt]))
    stats.set_output(True).set_dim([B, H, S_q, 1]).set_stride([H * S_q, S_q, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    amax_o.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    _select_engine(g, engine_name(128, mxfp8=True))
    g.check_support()
    g.build_plans()
    g.execute(
        {
            q: Qb,
            k: Kb,
            v: Vb,
            dq: sfq.view(torch.uint8).reshape(B, H, sqp, dsc),
            dk: sfk.view(torch.uint8).reshape(B, H, skp, dsc),
            dv: sfv.view(torch.uint8).reshape(B, H, ssc, dvp),
            o: Ob,
            stats: lse,
            amax_o: amax,
        },
        torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8),
    )
    torch.cuda.synchronize()
    o_ref = _ref(Q8.float() * dqq, K8.float() * dqk, V8.float() * dqv, scale=scale, is_causal=True, bottom_right=True)
    return Ob, o_ref, amax
