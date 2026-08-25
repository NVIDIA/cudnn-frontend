# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""End-to-end tests for the FROST SM100 DSL block-scale MXFP8 SDPA-forward engine.

Drives ``graph.sdpa_mxfp8`` (FP8 E4M3/E5M2 Q/K/V + per-32-block E8M0 scale factors)
routed to the native D128/D128 or D192/D128 MXFP8 engine, and validates the output
against an fp32-dequant reference. Inputs are quantized with the torch-only
MXFP8 quantizer in ``test/python/sdpa/mxfp8_quant.py`` (TE-equivalent semantics
— the same layout cuDNN's own mxfp8 path uses), so the scale-factor tensors
reach the engine in cuDNN's F8_128x4 reordering. TransformerEngine is NOT
required.

cuDNN's ``sdpa_mxfp8`` op exposes causal / bottom-right / sliding-window masks +
attention sink, and (frontend extension) ``use_padding_mask``/``seq_len_q``/
``seq_len_kv`` — which also carry THD/ragged inputs (packed Q/K/V/O + per-operand
ragged_offset, tested here, incl. the cu_seq_len prefix-sum form). THD scale
factors travel PACKED: per head, each sequence's TILE(128-row)-padded F8_128x4
SF tiles concatenated in cu_seqlens order (the F8_128x4 atom padding rounds each
sequence's SF to whole 128-row tiles, so the dense per-sequence quantizer output
concatenates directly). Ragged Stats use the packed token-major TH1 layout.

Requires: SM100 (Blackwell), cutlass-dsl, cuDNN >= 9.21 (mxfp8 support).
Skips cleanly otherwise.
"""

import math
from typing import NamedTuple

import pytest
import torch

from test_utils import torch_fork_set_rng

from cudnn.sdpa.fwd.engines import engine_name
from frost_test_utils import make_dense_stats, requires_pre_rubin_blackwell, requires_dsl


from frost_test_utils import select_engine as _select_engine  # noqa: F401

pytestmark = [requires_pre_rubin_blackwell, requires_dsl]


_FP8 = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
_OUT = {"fp16": torch.float16, "bf16": torch.bfloat16, "e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
_CUDNN_ITYPE = {"e4m3": "FP8_E4M3", "e5m2": "FP8_E5M2"}
_CUDNN_OTYPE = {torch.float16: "HALF", torch.bfloat16: "BFLOAT16", torch.float8_e4m3fn: "FP8_E4M3", torch.float8_e5m2: "FP8_E5M2"}
_BLOCK = 32


class _ReferenceWithStats(NamedTuple):
    output: torch.Tensor
    stats: torch.Tensor


class _RunResult(NamedTuple):
    output: torch.Tensor
    reference: torch.Tensor
    amax: torch.Tensor


class _RunWithStatsResult(NamedTuple):
    output: torch.Tensor
    reference: torch.Tensor
    amax: torch.Tensor
    stats: torch.Tensor
    reference_stats: torch.Tensor


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


def _ref(qd, kd, vd, *, scale, is_causal=False, bottom_right=False, swa_window=None, sinks=None, seq_lens_kv=None, return_stats=False):
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
    if seq_lens_kv is not None:
        # Per-batch KV padding: columns j >= seq_len_kv[b] are padding -> masked.
        slk = torch.as_tensor(seq_lens_kv, device=dev, dtype=torch.long).view(b, 1, 1, 1)
        masked = masked | (j >= slk)
    scores = scores.masked_fill(masked, float("-inf"))
    if sinks is not None:
        col = sinks.view(1, h_q, 1, 1).float().expand(b, h_q, s_q, 1).to(dev)
        ext = torch.cat([scores, col], dim=-1)
        probs = torch.softmax(ext, dim=-1)
        o = torch.matmul(probs[..., :s_kv], v_e)
        if return_stats:
            return _ReferenceWithStats(o, torch.logsumexp(ext, dim=-1))
        return o
    o = torch.matmul(torch.softmax(scores, dim=-1), v_e)
    if return_stats:
        return _ReferenceWithStats(o, torch.logsumexp(scores, dim=-1))
    return o


def _run(
    B,
    H_q,
    H_kv,
    S,
    in_key,
    out_dt,
    *,
    scale,
    sdpa_kwargs,
    sink=None,
    stats=True,
    seq_lens_kv=None,
    d_qk=128,
    d_v=128,
    stats_layout="contiguous",
    return_lse=False,
):
    """Quantize, build the sdpa_mxfp8 graph, route to the frost engine, execute.

    Returns ``_RunResult`` or, when ``return_lse`` is set,
    ``_RunWithStatsResult``.
    """
    import cudnn

    dev = "cuda"
    fp8 = _FP8[in_key]
    Qf = torch.randn(B, H_q, S, d_qk, device=dev) * 0.5
    Kf = torch.randn(B, H_kv, S, d_qk, device=dev) * 0.5
    Vf = torch.randn(B, H_kv, S, d_v, device=dev) * 0.5
    Q8, sfq, dqq, (sqp, dsc) = _quantize(Qf, B, H_q, S, d_qk, fp8, columnwise=False)
    K8, sfk, dqk, (skp, _) = _quantize(Kf, B, H_kv, S, d_qk, fp8, columnwise=False)
    V8, sfv, dqv, (ssc, dvp) = _quantize(Vf, B, H_kv, S, d_v, fp8, columnwise=True)

    def bshd(x8):
        return x8.permute(0, 2, 1, 3).contiguous().transpose(1, 2)

    Qb, Kb, Vb = bshd(Q8), bshd(K8), bshd(V8)
    Ob = torch.empty(B, S, H_q, d_v, device=dev, dtype=out_dt).transpose(1, 2)
    lse = make_dense_stats(B, H_q, S, stats_layout)
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
    kw = dict(q=q, k=k, v=v, descale_q=dq, descale_k=dk, descale_v=dv, attn_scale=scale, generate_stats=stats)
    vp = {q: Qb, k: Kb, v: Vb, dq: sfq_g, dk: sfk_g, dv: sfv_g}
    if sink is not None:
        st = g.tensor_like(sink)
        kw["sink_token"] = st
        vp[st] = sink
    if seq_lens_kv is not None:
        # KV padding (frontend extension on sdpa_mxfp8): full (unpadded) query
        # lengths + per-batch valid KV lengths.
        slq = torch.full((B, 1, 1, 1), S, dtype=torch.int32, device=dev)
        slk = torch.tensor(seq_lens_kv, dtype=torch.int32, device=dev).reshape(B, 1, 1, 1)
        sq_h = g.tensor(dim=[B, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
        skv_h = g.tensor(dim=[B, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
        kw.update(use_padding_mask=True, seq_len_q=sq_h, seq_len_kv=skv_h)
        vp[sq_h] = slq
        vp[skv_h] = slk
    kw.update(sdpa_kwargs)
    o, stats_t, amax_o = g.sdpa_mxfp8(**kw)
    o.set_output(True).set_dim(list(Ob.shape)).set_stride(list(Ob.stride())).set_data_type(otype)
    if stats:
        stats_t.set_output(True).set_dim([B, H_q, S, 1]).set_stride(list(lse.stride())).set_data_type(cudnn.data_type.FLOAT)
    amax_o.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    _select_engine(g, engine_name(mxfp8=True))
    g.check_support()
    g.build_plans()
    if not stats:
        # No Stats output: the kernel compiles the LSE store out (has_lse=False)
        # — no dummy buffer exists at any level, so the dense workspace is 0.
        assert g.get_workspace_size() == 0
    vp.update({o: Ob, amax_o: amax})
    if stats:
        vp[stats_t] = lse
    g.execute(vp, torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8))
    torch.cuda.synchronize()

    ref_kwargs = {k2: v2 for k2, v2 in _ref_from_sdpa(sdpa_kwargs).items()}
    if sink is not None:
        ref_kwargs["sinks"] = sink.flatten()
    if return_lse:
        assert stats, "return_lse requires stats=True"
        o_ref, lse_ref = _ref(
            Q8.float() * dqq,
            K8.float() * dqk,
            V8.float() * dqv,
            scale=scale,
            seq_lens_kv=seq_lens_kv,
            return_stats=True,
            **ref_kwargs,
        )
        return _RunWithStatsResult(Ob, o_ref, amax, lse.squeeze(-1), lse_ref)
    o_ref = _ref(Q8.float() * dqq, K8.float() * dqk, V8.float() * dqv, scale=scale, seq_lens_kv=seq_lens_kv, **ref_kwargs)
    return _RunResult(Ob, o_ref, amax)


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


def _half_atol(in_key, d_qk):
    return 8e-2 if in_key == "e5m2" and d_qk > 128 else 7e-2 if in_key == "e5m2" else 5e-2


def _check(O, O_ref, out_dt, in_key=None, d_qk=128):
    """Compare frost output to fp32 ref; for FP8 output, gauge against the fp8-quant floor.

    E5M2 inputs (2-bit mantissa) are noisier than E4M3 (3-bit), so the half-output
    tolerance is widened for them.
    """
    # D192 accumulates 50% more QK products than D128. Keep the existing D128
    # threshold unchanged while allowing the measured E5M2 accumulation floor.
    atol_half = _half_atol(in_key, d_qk)
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


def _check_mxfp8_strided_stats(d_qk, d_v, in_key):
    if torch.cuda.get_device_capability() == (10, 7):
        pytest.skip("SM107 serves per-tensor FP8 d128, not block-scaled MXFP8")
    kwargs = dict(
        B=2,
        H_q=4,
        H_kv=2,
        S=128,
        in_key=in_key,
        out_dt=torch.float16,
        scale=1.0 / math.sqrt(d_qk),
        sdpa_kwargs=dict(use_causal_mask=True),
        d_qk=d_qk,
        d_v=d_v,
        return_lse=True,
    )
    torch.manual_seed(59)
    contiguous = _run(**kwargs, stats_layout="contiguous")
    torch.manual_seed(59)
    strided = _run(**kwargs, stats_layout="strided")
    _check(strided.output, strided.reference, torch.float16, in_key, d_qk=d_qk)
    torch.testing.assert_close(strided.stats, contiguous.stats, rtol=0, atol=0)
    torch.testing.assert_close(strided.stats, strided.reference_stats, rtol=3e-2, atol=_half_atol(in_key, d_qk))


@pytest.mark.L0
@torch_fork_set_rng(seed=59)
def test_mxfp8_strided_stats():
    """The block-scaled FP8 L0 flavor preserves dense Stats strides."""
    _check_mxfp8_strided_stats(128, 128, "e4m3")


@pytest.mark.L1
@pytest.mark.parametrize(
    ("d_qk", "d_v", "in_key"),
    [(128, 128, "e5m2"), (192, 128, "e4m3"), (192, 128, "e5m2")],
    ids=["d128_e5m2", "d192_d128_e4m3", "d192_d128_e5m2"],
)
@torch_fork_set_rng(seed=59)
def test_mxfp8_strided_stats_other_flavors(d_qk, d_v, in_key):
    """The remaining block-scaled FP8 flavors preserve dense Stats strides."""
    _check_mxfp8_strided_stats(d_qk, d_v, in_key)


@pytest.mark.L0
@pytest.mark.parametrize("in_key", _INS)
@torch_fork_set_rng(seed=0)
def test_mxfp8_stats_less_zero_workspace(in_key):
    """No Stats output: the kernel compiles the LSE store out (has_lse=False),
    no dummy buffer exists at any level, and the dense graph reports
    ``get_workspace_size() == 0`` (asserted inside ``_run``). The Amax_O
    atomicMax write is independent of the LSE and still produced."""
    scale = 1.0 / math.sqrt(128)
    O, O_ref, _ = _run(2, 8, 8, 256, in_key, torch.float16, scale=scale, sdpa_kwargs=dict(use_causal_mask=True), stats=False)
    _check(O, O_ref, torch.float16, in_key)


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
@pytest.mark.parametrize("in_key", _INS)
@pytest.mark.parametrize("mask", list(_MASKS))
@torch_fork_set_rng(seed=0)
def test_mxfp8_d192_d128(in_key, mask):
    """Native D192/D128 path, including the grouped-LPT scheduler geometry."""
    d_qk, d_v = 192, 128
    O, O_ref, _ = _run(
        2,
        16,
        16,
        256,
        in_key,
        torch.bfloat16,
        scale=1.0 / math.sqrt(d_qk),
        sdpa_kwargs=_MASKS[mask],
        d_qk=d_qk,
        d_v=d_v,
    )
    _check(O, O_ref, torch.bfloat16, in_key, d_qk=d_qk)


@pytest.mark.L0
@pytest.mark.parametrize("out_key", ["fp16", "bf16", "e4m3", "e5m2"])
@torch_fork_set_rng(seed=0)
def test_mxfp8_d192_d128_output_dtypes(out_key):
    """D192/D128 E4M3 input to each supported output dtype."""
    d_qk, d_v = 192, 128
    O, O_ref, amax = _run(
        1,
        8,
        8,
        256,
        "e4m3",
        _OUT[out_key],
        scale=1.0 / math.sqrt(d_qk),
        sdpa_kwargs=dict(use_causal_mask=True),
        d_qk=d_qk,
        d_v=d_v,
    )
    _check(O, O_ref, _OUT[out_key], "e4m3", d_qk=d_qk)
    amax_value = amax.item()
    amax_ref = O_ref.abs().max().item()
    assert abs(amax_value - amax_ref) <= 0.03, f"amax {amax_value:.4f} vs ref {amax_ref:.4f}"


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_mxfp8_d192_d128_gqa_sink():
    """D192/D128 GQA with an attention sink exercises head sharing."""
    d_qk, d_v = 192, 128
    sink = torch.randn(1, 8, 1, 1, dtype=torch.float32, device="cuda")
    O, O_ref, _ = _run(
        1,
        8,
        2,
        256,
        "e5m2",
        torch.float16,
        scale=1.0 / math.sqrt(d_qk),
        sdpa_kwargs=dict(use_causal_mask=True),
        sink=sink,
        d_qk=d_qk,
        d_v=d_v,
    )
    _check(O, O_ref, torch.float16, "e5m2", d_qk=d_qk)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_mxfp8_d192_d128_stats_less():
    """D192/D128 supports compiling out the optional stats output."""
    d_qk, d_v = 192, 128
    O, O_ref, _ = _run(
        1,
        8,
        8,
        256,
        "e4m3",
        torch.bfloat16,
        scale=1.0 / math.sqrt(d_qk),
        sdpa_kwargs=dict(use_causal_mask=True),
        stats=False,
        d_qk=d_qk,
        d_v=d_v,
    )
    _check(O, O_ref, torch.bfloat16, "e4m3", d_qk=d_qk)


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
@pytest.mark.parametrize("d_qk,d_v", [(128, 128), (192, 128)])
@torch_fork_set_rng(seed=0)
def test_mxfp8_bottom_right_rectangular(d_qk, d_v):
    """Bottom-right causal with S_q != S_kv (the case where it differs from top-left)."""
    scale = 1.0 / math.sqrt(d_qk)
    # S must be a multiple of TILE_N (128) for the non-padded path; use S_q=128, S_kv=256.
    import cudnn  # noqa: F401

    O, O_ref, _ = _run_rect(
        2,
        8,
        128,
        256,
        "e4m3",
        torch.float16,
        scale=scale,
        sdpa_kwargs=dict(use_causal_mask_bottom_right=True),
        d_qk=d_qk,
        d_v=d_v,
    )
    _check(O, O_ref, torch.float16, "e4m3", d_qk=d_qk)


def _run_rect(B, H, S_q, S_kv, in_key, out_dt, *, scale, sdpa_kwargs, d_qk=128, d_v=128):
    """_run variant allowing S_q != S_kv (bottom-right causal)."""
    import cudnn

    dev = "cuda"
    fp8 = _FP8[in_key]
    Qf = torch.randn(B, H, S_q, d_qk, device=dev) * 0.5
    Kf = torch.randn(B, H, S_kv, d_qk, device=dev) * 0.5
    Vf = torch.randn(B, H, S_kv, d_v, device=dev) * 0.5
    Q8, sfq, dqq, (sqp, dsc) = _quantize(Qf, B, H, S_q, d_qk, fp8, columnwise=False)
    K8, sfk, dqk, (skp, _) = _quantize(Kf, B, H, S_kv, d_qk, fp8, columnwise=False)
    V8, sfv, dqv, (ssc, dvp) = _quantize(Vf, B, H, S_kv, d_v, fp8, columnwise=True)

    def bshd(x8):
        return x8.permute(0, 2, 1, 3).contiguous().transpose(1, 2)

    Qb, Kb, Vb = bshd(Q8), bshd(K8), bshd(V8)
    Ob = torch.empty(B, S_q, H, d_v, device=dev, dtype=out_dt).transpose(1, 2)
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
    _select_engine(g, engine_name(mxfp8=True))
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


@pytest.mark.L0
@pytest.mark.parametrize("in_key", _INS)
@pytest.mark.parametrize("causal", [False, True])
@torch_fork_set_rng(seed=0)
def test_mxfp8_dense_padding(in_key, causal):
    """Dense KV padding mask (frontend extension on sdpa_mxfp8): batch 0 uses all
    256 KV cols, batch 1 only 192 (a partial KV tile). Padded columns must not
    leak into the softmax or the in-kernel Amax_O. Stats stay off (padded+stats
    needs the per-batch LSE trim, which this cell does not declare)."""
    scale = 1.0 / math.sqrt(128)
    sk = dict(use_causal_mask=True) if causal else {}
    o_out, o_ref, amax = _run(2, 8, 8, 256, in_key, torch.float16, scale=scale, sdpa_kwargs=sk, seq_lens_kv=[256, 192], stats=False)
    _check(o_out, o_ref, torch.float16, in_key)
    assert abs(amax.item() - o_ref.abs().max().item()) <= 0.03


def _quantize_seq(t_1hsd, h, s, d, fp8, *, columnwise):
    """Per-sequence MXFP8 quantization for the THD packing.

    Returns (fp8_data [1,h,s,d], per-elem dequant scale [1,h,s,d], SF tiles
    [h, n_tiles, SF_SMEM] uint8) where n_tiles = ceil(s/128): the quantizer's
    F8_128x4 atom padding rounds the S extent up to a multiple of 128, i.e.
    exactly the kernel's per-sequence-TILE-padded SF layout."""
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    if s == 0:
        # Zero-length sequence: no tokens, no SF tiles.
        empty = t_1hsd.new_zeros((1, h, 0, d))
        return empty.to(fp8), empty.float(), torch.zeros((h, 0, 128 * d // _BLOCK), dtype=torch.uint8, device=t_1hsd.device)
    data_d, dq_d, swz_d, data_s, dq_s, swz_s = quantize_to_mxfp8(t_1hsd, 1, h, s, d, _BLOCK, fp8, with_ref=True)
    n_tiles = _cdiv(s, 128)
    if columnwise:
        return data_s, dq_s.reshape(1, h, s, d), swz_s.view(torch.uint8).reshape(h, n_tiles, -1)
    return data_d, dq_d.reshape(1, h, s, d), swz_d.view(torch.uint8).reshape(h, n_tiles, -1)


def _run_thd(seq_lens_q, seq_lens_kv, H_q, H_kv, in_key, out_dt, *, scale, causal=False, sink=None, stats=False, cu_lens=False):
    """THD/varlen: packed [T,H,D] Q/K/V/O + ragged offsets + per-batch lengths
    (or their cu prefix-sum form) + PACKED per-sequence-TILE-padded SF."""
    import cudnn

    dev = "cuda"
    D = 128
    fp8 = _FP8[in_key]
    B = len(seq_lens_q)
    S_max_q, S_max_kv = max(seq_lens_q), max(seq_lens_kv)
    T_q = sum(seq_lens_q)

    def _cu(sl):
        c = [0]
        for s in sl:
            c.append(c[-1] + s)
        return c

    cu_q, cu_k = _cu(seq_lens_q), _cu(seq_lens_kv)

    # Per-sequence quantization; pack the fp8 tokens and, per head, the
    # TILE-padded SF tiles in cu_seqlens order.
    q8_seqs, k8_seqs, v8_seqs, dq_seqs, dk_seqs, dv_seqs = [], [], [], [], [], []
    sfq_seqs, sfk_seqs, sfv_seqs = [], [], []
    for b in range(B):
        s_q, s_kv = seq_lens_q[b], seq_lens_kv[b]
        Qf = torch.randn(1, H_q, s_q, D, device=dev) * 0.5
        Kf = torch.randn(1, H_kv, s_kv, D, device=dev) * 0.5
        Vf = torch.randn(1, H_kv, s_kv, D, device=dev) * 0.5
        q8, dqq, sfq = _quantize_seq(Qf, H_q, s_q, D, fp8, columnwise=False)
        k8, dqk, sfk = _quantize_seq(Kf, H_kv, s_kv, D, fp8, columnwise=False)
        v8, dqv, sfv = _quantize_seq(Vf, H_kv, s_kv, D, fp8, columnwise=True)
        q8_seqs.append(q8)
        k8_seqs.append(k8)
        v8_seqs.append(v8)
        dq_seqs.append(dqq)
        dk_seqs.append(dqk)
        dv_seqs.append(dqv)
        sfq_seqs.append(sfq)
        sfk_seqs.append(sfk)
        sfv_seqs.append(sfv)

    def _pack_tokens(x8_seqs):
        # [1,h,s,d] per sequence -> packed [T,h,D] tokens.
        return torch.cat([x.squeeze(0).permute(1, 0, 2) for x in x8_seqs], dim=0)

    q_pk = _pack_tokens(q8_seqs)
    k_pk = _pack_tokens(k8_seqs)
    v_pk = _pack_tokens(v8_seqs)
    # Packed SF: [h, total_tiles, SF_SMEM] — per head, sequences' tiles in
    # cu_seqlens order. The buffer is EXACTLY the packed layout (the engine
    # derives the packed tile extent from its byte size).
    sfq_pk = torch.cat(sfq_seqs, dim=1).contiguous()
    sfk_pk = torch.cat(sfk_seqs, dim=1).contiguous()
    sfv_pk = torch.cat(sfv_seqs, dim=1).contiguous()

    def _dense_buf(packed, s_max, h, dt):
        # Dense-capacity storage; packed tokens in the leading elements (THD
        # contract). The capacity tail is NaN-POISONED (test_mhas_v2 parity):
        # the last sequence's KV tile steps past the packed total, and those
        # tail loads must land as zeros through the setup kernel's
        # packed-total-clamped K/V descriptors — a leaked NaN would wipe the
        # tile via BMM2's P·V (0 · NaN == NaN).
        stride = (s_max * h * D, D, h * D, 1)
        stor = torch.full((B * s_max * h * D,), float("nan"), device=dev, dtype=torch.float32).to(dt)
        stor[: packed.numel()] = packed.reshape(-1)
        return stor, stor.as_strided((B, h, s_max, D), stride), stride

    _, q_gpu, stride_q = _dense_buf(q_pk, S_max_q, H_q, q_pk.dtype)
    _, k_gpu, stride_kv = _dense_buf(k_pk, S_max_kv, H_kv, k_pk.dtype)
    _, v_gpu, _ = _dense_buf(v_pk, S_max_kv, H_kv, v_pk.dtype)
    o_stor = torch.zeros(B * S_max_q * H_q * D, device=dev, dtype=out_dt)
    o_gpu = o_stor.as_strided((B, H_q, S_max_q, D), stride_q)
    amax = torch.zeros(1, 1, 1, 1, device=dev, dtype=torch.float32)

    slq = torch.tensor(seq_lens_q, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    slk = torch.tensor(seq_lens_kv, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    cuq_t = torch.tensor(cu_q, dtype=torch.int32, device=dev).view(B + 1, 1, 1, 1)
    cuk_t = torch.tensor(cu_k, dtype=torch.int32, device=dev).view(B + 1, 1, 1, 1)
    ro_q = (torch.tensor(cu_q, dtype=torch.int64, device=dev) * H_q * D).view(B + 1, 1, 1, 1)
    ro_k = (torch.tensor(cu_k, dtype=torch.int64, device=dev) * H_kv * D).view(B + 1, 1, 1, 1)

    io = getattr(cudnn.data_type, _CUDNN_ITYPE[in_key])
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tq = g.tensor(dim=[B, H_q, S_max_q, D], stride=list(stride_q), data_type=io, name="q")
    tk = g.tensor(dim=[B, H_kv, S_max_kv, D], stride=list(stride_kv), data_type=io, name="k")
    tv = g.tensor(dim=[B, H_kv, S_max_kv, D], stride=list(stride_kv), data_type=io, name="v")
    sq_h = g.tensor_like(cuq_t if cu_lens else slq)
    skv_h = g.tensor_like(cuk_t if cu_lens else slk)
    qro, kro, vro, oro = (g.tensor_like(ro_q) for _ in range(4))
    tq.set_ragged_offset(qro)
    tk.set_ragged_offset(kro)
    tv.set_ragged_offset(vro)

    def _sf(dims):
        # Dense-capacity declaration; the bound buffer holds the PACKED layout
        # (same convention as the ragged Q/K/V storage).
        return g.tensor(
            dim=list(dims),
            stride=[dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1],
            data_type=cudnn.data_type.FP8_E8M0,
            reordering_type=cudnn.tensor_reordering.F8_128x4,
        )

    dsc = _cdiv(_cdiv(D, _BLOCK), 4) * 4
    dvp = dsc * _BLOCK
    dq = _sf((B, H_q, _cdiv(S_max_q, 128) * 128, dsc))
    dk = _sf((B, H_kv, _cdiv(S_max_kv, 128) * 128, dsc))
    dv = _sf((B, H_kv, _cdiv(S_max_kv, 128) * 4, dvp))
    kw = dict(
        q=tq,
        k=tk,
        v=tv,
        descale_q=dq,
        descale_k=dk,
        descale_v=dv,
        attn_scale=scale,
        generate_stats=stats,
        use_padding_mask=True,
    )
    if cu_lens:
        kw.update(cu_seq_len_q=sq_h, cu_seq_len_kv=skv_h)
    else:
        kw.update(seq_len_q=sq_h, seq_len_kv=skv_h)
    if causal:
        kw["use_causal_mask"] = True
    vp = {
        tq: q_gpu,
        tk: k_gpu,
        tv: v_gpu,
        dq: sfq_pk,
        dk: sfk_pk,
        dv: sfv_pk,
        sq_h: (cuq_t if cu_lens else slq),
        skv_h: (cuk_t if cu_lens else slk),
        qro: ro_q,
        kro: ro_k,
        vro: ro_k,
        oro: ro_q,
    }
    if sink is not None:
        st = g.tensor_like(sink)
        kw["sink_token"] = st
        vp[st] = sink
    o, stats_t, amax_o = g.sdpa_mxfp8(**kw)
    o.set_output(True).set_dim([B, H_q, S_max_q, D]).set_stride(list(stride_q)).set_data_type(getattr(cudnn.data_type, _CUDNN_OTYPE[out_dt]))
    o.set_ragged_offset(oro)
    amax_o.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    stats_stor = None
    if stats:
        # Ragged Stats, packed token-major TH1 ([t, h]; offsets = cu_q * h_q).
        stats_stor = torch.zeros(B * S_max_q * H_q, dtype=torch.float32, device=dev)
        stats_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
        stats_t.set_dim((B, H_q, S_max_q, 1)).set_stride((S_max_q * H_q, 1, H_q, 1))
        stats_ro_t = (ro_q.flatten() // D).view(B + 1, 1, 1, 1).contiguous()
        stats_ro = g.tensor_like(stats_ro_t, name="stats_ro")
        stats_t.set_ragged_offset(stats_ro)
        vp[stats_ro] = stats_ro_t
        vp[stats_t] = stats_stor

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    _select_engine(g, engine_name(mxfp8=True))
    g.check_support()
    g.build_plans()
    vp.update({o: o_gpu, amax_o: amax})
    g.execute(vp, torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8))
    torch.cuda.synchronize()

    o_ref = torch.zeros(T_q, H_q, D, device=dev, dtype=torch.float32)
    for b in range(B):
        if cu_q[b + 1] == cu_q[b] or cu_k[b + 1] == cu_k[b]:
            # Zero-length Q contributes no rows; zero-length KV leaves every
            # row of the sequence dead — O := 0 (o_ref is pre-zeroed).
            continue
        qd = q8_seqs[b].float() * dq_seqs[b]
        kd = k8_seqs[b].float() * dk_seqs[b]
        vd = v8_seqs[b].float() * dv_seqs[b]
        ref_kw = dict(is_causal=True) if causal else {}
        if sink is not None:
            ref_kw["sinks"] = sink.flatten()
        ob = _ref(qd, kd, vd, scale=scale, **ref_kw)
        o_ref[cu_q[b] : cu_q[b + 1]] = ob.squeeze(0).permute(1, 0, 2)

    o_out = o_stor[: T_q * H_q * D].reshape(T_q, H_q, D)
    lse_out = stats_stor[: T_q * H_q].reshape(T_q, H_q) if stats else None
    return o_out, o_ref, amax, lse_out


@pytest.mark.L0
@pytest.mark.parametrize("in_key", _INS)
@pytest.mark.parametrize("causal", [False, True])
@torch_fork_set_rng(seed=0)
def test_mxfp8_thd(in_key, causal):
    """THD/varlen self-attention: two packed sequences of unequal, tile-ragged length."""
    scale = 1.0 / math.sqrt(128)
    o_out, o_ref, amax, _ = _run_thd([200, 150], [200, 150], 8, 8, in_key, torch.float16, scale=scale, causal=causal)
    _check(o_out, o_ref, torch.float16, in_key)
    assert abs(amax.item() - o_ref.abs().max().item()) <= 0.03


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_mxfp8_thd_cross_gqa():
    """THD cross-attention (unequal packed Q and K/V totals) with GQA heads."""
    scale = 1.0 / math.sqrt(128)
    o_out, o_ref, _, _ = _run_thd([64, 200], [256, 128], 8, 2, "e4m3", torch.float16, scale=scale)
    _check(o_out, o_ref, torch.float16, "e4m3")


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_mxfp8_thd_sink():
    """THD causal + attention sink."""
    scale = 1.0 / math.sqrt(128)
    sink = torch.randn(1, 8, 1, 1, dtype=torch.float32, device="cuda")
    o_out, o_ref, _, _ = _run_thd([200, 150], [200, 150], 8, 8, "e4m3", torch.float16, scale=scale, causal=True, sink=sink)
    _check(o_out, o_ref, torch.float16, "e4m3")


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_mxfp8_thd_stats():
    """THD + generate_stats: the ragged token-major TH1 LSE is written next to O."""
    scale = 1.0 / math.sqrt(128)
    o_out, o_ref, _, lse = _run_thd([200, 150], [200, 150], 8, 8, "e4m3", torch.float16, scale=scale, causal=True, stats=True)
    _check(o_out, o_ref, torch.float16, "e4m3")
    assert lse is not None and torch.isfinite(lse).all()


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_mxfp8_thd_zero_len_kv():
    """Zero-length Q and KV sequences (test_mhas_v2 ragged parity): the
    zero-KV sequence's rows are dead — the epilogue must come back O := 0,
    not the unwritten O TMEM (garbage survives `* inv_sum(=0)` when NaN)."""
    scale = 1.0 / math.sqrt(128)
    o_out, o_ref, _, _ = _run_thd([126, 0, 60], [0, 83, 77], 8, 8, "e4m3", torch.float16, scale=scale)
    _check(o_out, o_ref, torch.float16, "e4m3")


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_mxfp8_thd_cu_seq_len():
    """THD via the (B+1,) cu_seq_len prefix-sum length form."""
    scale = 1.0 / math.sqrt(128)
    o_out, o_ref, _, _ = _run_thd([200, 150], [180, 120], 8, 8, "e4m3", torch.float16, scale=scale, cu_lens=True)
    _check(o_out, o_ref, torch.float16, "e4m3")
