# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""End-to-end tests for the FROST SM100 DSL block-scale MXFP8 SDPA-forward engine.

Drives ``graph.sdpa_mxfp8`` (FP8 E4M3/E5M2 Q/K/V + per-32-block E8M0 scale factors)
routed to the exact D128/D128, D192/D128, D256/D256, or D512/D512 MXFP8 engine, and
validates the output against an fp32-dequant reference. Inputs are quantized
with the torch-only MXFP8 quantizer in ``test/python/sdpa/mxfp8_quant.py``
(TE-equivalent semantics
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

import glob
import math
import os
import shutil
import subprocess
import sys
import textwrap
from typing import NamedTuple, Optional

import pytest
import torch

from test_utils import torch_fork_set_rng

from cudnn.sdpa.fwd.engines import engine_name
from frost_test_utils import _SM, make_dense_stats, requires_blackwell, requires_dsl


from frost_test_utils import select_engine as _select_engine  # noqa: F401

# Rubin (cc10.7) now has MXFP8 lowerings of its own, so this suite is no longer
# pre-Rubin only.  ONE engine per arch line: pin the row that serves the device
# under test, exactly as test_sdpa_fwd_fp8_sm100.py does -- engine_name()
# defaults to arch="sm100", whose row stops at cc10.6, so an unqualified pin
# fails on Rubin with "no plan for engine" for routing reasons that have
# nothing to do with the kernel.
_ARCH = "sm107" if _SM == 107 else "sm100"

# Rubin gaps the SM100 line does not have.  These are CAPABILITY declines the
# engine row states honestly, not kernel bugs -- the graph is never served, so
# the test cannot run.  Flip the condition to False when the gap closes.
#
# INVERTED 2026-09-09: the d192xd128 MXFP8 gap CLOSED
# (sm107/prefill_d192_d128_mxfp8.py), so every DENSE d192 case below now runs on
# Rubin.  It runs at cga2 there and only cga2 -- at cga1 that flavor's four
# scale-factor tiles start past the 256 KiB version-0 tcgen05 descriptor window
# -- which the row expresses by leaving (192, 128) on its default cgas={2}, so
# nothing here needs to say so.  THD is still declined row-wide.
# THD/varlen is not ported to the Rubin MXFP8 kernels: the setup-kernel call
# site still speaks the pre-upstream 7-arg contract against a 14-arg helper and
# the metadata layout differs (3B+2 vs 4B+4), so compile() raises and the row
# declares thd=False.  Flip to False when the THD port lands.
_skip_thd_mxfp8_on_rubin = pytest.mark.skipif(
    _SM == 107,
    reason="THD/varlen not ported to the Rubin MXFP8 kernels (row sets thd=False)",
)
pytestmark = [requires_blackwell, requires_dsl]


_FP8 = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
_OUT = {"fp16": torch.float16, "bf16": torch.bfloat16, "e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
_CUDNN_ITYPE = {"e4m3": "FP8_E4M3", "e5m2": "FP8_E5M2"}
_CUDNN_OTYPE = {torch.float16: "HALF", torch.bfloat16: "BFLOAT16", torch.float8_e4m3fn: "FP8_E4M3", torch.float8_e5m2: "FP8_E5M2"}
_BLOCK = 32
# Block-scaled O (sdpa_mxfp8 + sf_o): "nvfp4" = FP4_E2M1 O + E4M3 scale per 16 d (needs
# scale_o, the FP4 global scale), "mxfp8" = FP8_E4M3 O + UE8M0 scale per 32 d.
_BLOCK_SCALED_O = {"nvfp4": 16, "mxfp8": 32}


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


def _ref(
    qd,
    kd,
    vd,
    *,
    scale,
    is_causal=False,
    bottom_right=False,
    swa_window=None,
    right_bound=0,
    sinks=None,
    seq_lens_q=None,
    seq_lens_kv=None,
    return_stats=False,
):
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
    # Bottom-right anchors the diagonal at each batch's corner (seq_len_q[b], seq_len_kv[b]),
    # not at the padded (S_q, S_kv): a shorter Q keeps its rows aligned with the END of its KV.
    diag = 0
    if bottom_right:
        slq_d = torch.as_tensor(seq_lens_q if seq_lens_q is not None else [s_q] * b, device=dev, dtype=torch.long).view(b, 1, 1, 1)
        slk_d = torch.as_tensor(seq_lens_kv if seq_lens_kv is not None else [s_kv] * b, device=dev, dtype=torch.long).view(b, 1, 1, 1)
        diag = slk_d - slq_d
    if is_causal:
        lim = i + diag
        masked = masked | (j > lim + right_bound)
    if swa_window is not None:
        swa_base = i + diag
        masked = masked | (j < swa_base - swa_window)
    if seq_lens_kv is not None:
        # Per-batch KV padding: columns j >= seq_len_kv[b] are padding -> masked.
        slk = torch.as_tensor(seq_lens_kv, device=dev, dtype=torch.long).view(b, 1, 1, 1)
        masked = masked | (j >= slk)
    scores = scores.masked_fill(masked, float("-inf"))

    def finalize(o, lse=None):
        if seq_lens_q is not None:
            slq = torch.as_tensor(seq_lens_q, device=dev, dtype=torch.long).view(b, 1, 1, 1)
            valid_q = i < slq
            o = torch.where(valid_q, o, torch.zeros_like(o))
            if lse is not None:
                lse = torch.where(valid_q.squeeze(-1), lse, torch.full_like(lse, float("-inf")))
        return _ReferenceWithStats(o, lse) if lse is not None else o

    if sinks is not None:
        col = sinks.view(1, h_q, 1, 1).float().expand(b, h_q, s_q, 1).to(dev)
        ext = torch.cat([scores, col], dim=-1)
        probs = torch.softmax(ext, dim=-1)
        o = torch.matmul(probs[..., :s_kv], v_e)
        return finalize(o, torch.logsumexp(ext, dim=-1) if return_stats else None)
    row_has_kv = torch.isfinite(scores).any(dim=-1, keepdim=True)
    probs = torch.softmax(scores, dim=-1)
    probs = torch.where(row_has_kv, probs, torch.zeros_like(probs))
    o = torch.matmul(probs, v_e)
    return finalize(o, torch.logsumexp(scores, dim=-1) if return_stats else None)


class _BlockScaledRunResult(NamedTuple):
    output: torch.Tensor  # dequantized O (B, H, S, d), fp32, in scale_o units
    reference: torch.Tensor  # fp32 reference O * scale_o
    amax: float
    reference_amax: float
    sf_pad_ok: bool
    scale_o: float


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
    seq_lens_q=None,
    seq_lens_kv=None,
    d_qk=128,
    d_v=128,
    stats_layout="contiguous",
    return_lse=False,
    poison_tmem_before_execute: bool = False,
    k_tile_growth: float = 1.0,
    gate: Optional[torch.Tensor] = None,
    amax: bool = True,
    block_scaled_o=None,
    sf_o_layout="planes",
    scale_o=None,
    capture_first: bool = False,
):
    """Quantize, build the sdpa_mxfp8 graph, route to the frost engine, execute.

    ``k_tile_growth`` > 1 scales K by ``growth ** (key // 128)`` so every KV
    tile raises the row max past the kernel's RESCALE_THRESHOLD and the
    correction path rescales O on every iteration (see
    ``test_mxfp8_d256_causal_rescale_every_tile``).

    Append-only knobs (PR-B): ``gate`` (a bf16 BHSD-logical tensor of O's shape)
    adds the epilogue-gate tail ``sdpa(virtual O_v) -> sigmoid(G) -> mul`` and
    folds sigmoid(G) into the returned ``reference`` (what O must match); the
    returned ``amax`` is the graph's ``Amax_O``, an output of the sdpa NODE that
    precedes the tail, so its reference is the UNGATED O's amax -- take it from
    a no-gate run's ``reference`` on the same problem.  ``amax=False`` leaves
    ``Amax_O`` a virtual (unrequested) port, which the engine folds out
    (has_amax_o=False).

    Returns ``_RunResult`` or, when ``return_lse`` is set,
    ``_RunWithStatsResult``.
    """
    import cudnn

    dev = "cuda"
    fp8 = _FP8[in_key]
    Qf = torch.randn(B, H_q, S, d_qk, device=dev) * 0.5
    Kf = torch.randn(B, H_kv, S, d_qk, device=dev) * 0.5
    if k_tile_growth != 1.0:
        Kf = Kf * (k_tile_growth ** (torch.arange(S, device=dev) // 128).float()).view(1, 1, S, 1)
    Vf = torch.randn(B, H_kv, S, d_v, device=dev) * 0.5
    Q8, sfq, dqq, (sqp, dsc) = _quantize(Qf, B, H_q, S, d_qk, fp8, columnwise=False)
    K8, sfk, dqk, (skp, _) = _quantize(Kf, B, H_kv, S, d_qk, fp8, columnwise=False)
    V8, sfv, dqv, (ssc, dvp) = _quantize(Vf, B, H_kv, S, d_v, fp8, columnwise=True)

    def bshd(x8):
        return x8.permute(0, 2, 1, 3).contiguous().transpose(1, 2)

    Qb, Kb, Vb = bshd(Q8), bshd(K8), bshd(V8)
    blk = _BLOCK_SCALED_O.get(block_scaled_o, 0)
    if blk == 16 and not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("the packed FP4 dtype (torch.float4_e2m1fn_x2) needs torch >= 2.8")
    if blk == 16:
        # FP4 O: the byte container (two E2M1 per byte) in torch's packed dtype.
        Ob = torch.full((B, S, H_q, d_v // 2), 0x7F, device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2).transpose(1, 2)
    else:
        Ob = torch.empty(B, S, H_q, d_v, device=dev, dtype=out_dt).transpose(1, 2)
    lse = make_dense_stats(B, H_q, S, stats_layout)
    amax_buf = torch.zeros(1, 1, 1, 1, device=dev, dtype=torch.float32)
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
    if seq_lens_q is not None or seq_lens_kv is not None:
        slq = torch.tensor(seq_lens_q if seq_lens_q is not None else [S] * B, dtype=torch.int32, device=dev).reshape(B, 1, 1, 1)
        slk = torch.tensor(seq_lens_kv if seq_lens_kv is not None else [S] * B, dtype=torch.int32, device=dev).reshape(B, 1, 1, 1)
        sq_h = g.tensor(dim=[B, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
        skv_h = g.tensor(dim=[B, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
        kw.update(use_padding_mask=True, seq_len_q=sq_h, seq_len_kv=skv_h)
        vp[sq_h] = slq
        vp[skv_h] = slk
    kw.update(sdpa_kwargs)
    sf_o_buf = None
    if blk:
        from sdpa.block_scale_o_ref import round_up

        C = max(4, round_up(d_v // blk, 4))
        if sf_o_layout == "planes":
            R = round_up(S, 128)
            sf_o_buf = torch.full((B, H_q, R, C), 0xAA, device=dev, dtype=torch.uint8)
            sf_o_stride = [H_q * R * C, R * C, C, 1]
        else:
            # token-major: one [B*S (padded to 128), H_q*C] matrix, declared BSHC;
            # the rows past B*S are caller-owned (pre-zeroed here).
            R = S
            sf_o_buf = torch.full((round_up(B * S, 128), H_q * C), 0xAA, device=dev, dtype=torch.uint8)
            sf_o_buf[B * S :] = 0
            sf_o_stride = [R * H_q * C, C, H_q * C, 1]
        sf_dtype = cudnn.data_type.FP8_E4M3 if blk == 16 else cudnn.data_type.FP8_E8M0
        sf_o_t = g.tensor(dim=[B, H_q, R, C], stride=sf_o_stride, data_type=sf_dtype)
        kw["sf_o"] = sf_o_t
        if scale_o is not None:
            # sdpa_mxfp8 has no per-tensor O scale otherwise: scale_o is the FP4
            # global scale the block-scaled epilogue folds into O (python-only input).
            so_g = g.tensor(dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.FLOAT)
            kw["scale_o"] = so_g
            vp[so_g] = torch.tensor([[[[float(scale_o)]]]], dtype=torch.float32, device=dev)
    o, stats_t, amax_o = g.sdpa_mxfp8(**kw)
    if gate is not None:
        # Epilogue-gate tail: O_v stays VIRTUAL but DECLARED (dim + stride), the mul output is the real O.
        o.set_dim(list(Ob.shape)).set_stride(list(Ob.stride()))
        gate_t = g.tensor_like(gate)
        vp[gate_t] = gate
        o = g.mul(a=o, b=g.sigmoid(input=gate_t, name="sig"), name="gated")
    if blk == 16:
        assert gate is None, "the epilogue gate and a block-scaled O are separate epilogues"
        # The FP4 O is declared with its LOGICAL element extent; the binding is the byte container.
        o.set_output(True).set_dim([B, H_q, S, d_v]).set_stride([S * H_q * d_v, d_v, H_q * d_v, 1]).set_data_type(cudnn.data_type.FP4_E2M1)
    else:
        o.set_output(True).set_dim(list(Ob.shape)).set_stride(list(Ob.stride())).set_data_type(otype)
    if stats:
        stats_t.set_output(True).set_dim([B, H_q, S, 1]).set_stride(list(lse.stride())).set_data_type(cudnn.data_type.FLOAT)
    if amax:
        amax_o.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    # amax=False: Amax_O is UNREQUESTED and left exactly as the block leaves it -- virtual, undeclared,
    # never bound.  sdpa_mxfp8 infers its [1, 1, 1, 1] dims the way sdpa_fp8 does, and a virtual
    # Amax_O is what makes the engine fold the atomic out (has_amax_o=False).

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    _select_engine(g, engine_name(arch=_ARCH, mxfp8=True))
    g.check_support()
    g.build_plans()
    if not stats:
        # No Stats output: the kernel compiles the LSE store out (has_lse=False)
        # — no dummy buffer exists at any level, so the dense workspace is 0.
        assert g.get_workspace_size() == 0
    vp[o] = Ob
    if amax:
        vp[amax_o] = amax_buf
    if blk:
        vp[sf_o_t] = sf_o_buf
    if stats:
        vp[stats_t] = lse
    workspace = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    if poison_tmem_before_execute:
        assert seq_lens_kv is not None
        poison_vp = dict(vp)
        poison_vp[v] = torch.full_like(Vb, float("nan"))
        poison_vp[skv_h] = torch.full_like(slk, S)
        g.execute(poison_vp, workspace)
        torch.cuda.synchronize()
        amax_buf.zero_()
    if capture_first:
        # The plan's FIRST execute is CAPTURED and never replayed here: anything
        # a plan lazily creates and fills on its first execute (a cached identity
        # scale, say) is allocated but its fill only captured, so the eager
        # execute below would read it half-initialized (Rule 8; review on #1180).
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            g.execute(vp, workspace)
        torch.cuda.synchronize()
    g.execute(vp, workspace)
    torch.cuda.synchronize()

    ref_kwargs = {k2: v2 for k2, v2 in _ref_from_sdpa(sdpa_kwargs).items()}
    if sink is not None:
        ref_kwargs["sinks"] = sink.flatten()

    def _gated(o_ref):
        return o_ref * torch.sigmoid(gate.float()).to(o_ref.dtype) if gate is not None else o_ref

    if blk:
        # Dequantize O from (bytes, sf_o) in the declared layout; the reference is
        # the fp32 O times scale_o (what the epilogue quantizes).
        from sdpa.block_scale_o_ref import dequant_block_scaled_o

        so = float(scale_o) if scale_o is not None else 1.0
        o_ref = _ref(Q8.float() * dqq, K8.float() * dqk, V8.float() * dqv, scale=scale, seq_lens_q=seq_lens_q, seq_lens_kv=seq_lens_kv, **ref_kwargs)
        o_deq, sf_pad_ok = dequant_block_scaled_o(Ob, sf_o_buf, blk, sf_o_layout, B, H_q, S, d_v, C)
        return _BlockScaledRunResult(o_deq, o_ref * so, amax_buf.item(), o_ref.abs().max().item(), sf_pad_ok, so)
    if return_lse:
        assert stats, "return_lse requires stats=True"
        o_ref, lse_ref = _ref(
            Q8.float() * dqq,
            K8.float() * dqk,
            V8.float() * dqv,
            scale=scale,
            seq_lens_q=seq_lens_q,
            seq_lens_kv=seq_lens_kv,
            return_stats=True,
            **ref_kwargs,
        )
        return _RunWithStatsResult(Ob, _gated(o_ref), amax_buf, lse.squeeze(-1), lse_ref)
    o_ref = _gated(_ref(Q8.float() * dqq, K8.float() * dqk, V8.float() * dqv, scale=scale, seq_lens_q=seq_lens_q, seq_lens_kv=seq_lens_kv, **ref_kwargs))
    return _RunResult(Ob, o_ref, amax_buf)


def _ref_from_sdpa(sdpa_kwargs):
    """Translate the sdpa_mxfp8 mask kwargs into _ref() kwargs."""
    out = {}
    if sdpa_kwargs.get("use_causal_mask"):
        out["is_causal"] = True
    if sdpa_kwargs.get("use_causal_mask_bottom_right"):
        out["is_causal"] = True
        out["bottom_right"] = True
    right_bound = sdpa_kwargs.get("diagonal_band_right_bound")
    if right_bound is not None:
        out["is_causal"] = True
        out["right_bound"] = right_bound
    diagonal_alignment = sdpa_kwargs.get("diagonal_alignment")
    if diagonal_alignment is not None:
        import cudnn

        out["bottom_right"] = diagonal_alignment == cudnn.diagonal_alignment.BOTTOM_RIGHT
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
@pytest.mark.parametrize("h_q,h_kv", [(4, 4), (4, 2)], ids=["mha", "gqa"])
@pytest.mark.parametrize("d_qk,d_v", [(128, 128), (192, 128)], ids=["d128", "d192_d128"])
@torch_fork_set_rng(seed=61)
def test_mxfp8_qk_bf16_pv_direct_experiment(h_q, h_kv, d_qk, d_v):
    """Validate the direct-only hybrid BF16-output contract.

    This deliberately exercises the direct-only adapter switch rather than a
    graph route: graph capability selection remains unchanged until benchmark
    evidence establishes that this tradeoff is worth exposing publicly.
    """
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    if torch.cuda.get_device_capability() == (10, 7):
        pytest.skip("SM107 serves per-tensor FP8 d128, not block-scaled MXFP8")

    b, s = 1, 256
    dev = "cuda"
    scale = d_qk**-0.5
    qf = torch.randn(b, h_q, s, d_qk, device=dev) * 0.5
    kf = torch.randn(b, h_kv, s, d_qk, device=dev) * 0.5
    v = (torch.randn(b, h_kv, s, d_v, device=dev) * 0.5).to(torch.bfloat16)
    q, sf_q, dq, _ = _quantize(qf, b, h_q, s, d_qk, torch.float8_e4m3fn, columnwise=False)
    k, sf_k, dk, _ = _quantize(kf, b, h_kv, s, d_qk, torch.float8_e4m3fn, columnwise=False)
    o = torch.empty(b, h_q, s, d_v, device=dev, dtype=torch.bfloat16)

    api = SdpaFwdDslSm100(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=o,
        is_causal=True,
        scale_softmax=scale,
        dtype_o=torch.bfloat16,
        split_kv=1,
        pv_bf16=True,
    )
    assert api.check_support()
    api.compile()
    with pytest.raises(ValueError, match="without Amax_O"):
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, sf_q=sf_q, sf_k=sf_k, amax_o=torch.empty(1, device=dev, dtype=torch.float32))
    api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, sf_q=sf_q, sf_k=sf_k)
    torch.cuda.synchronize()

    o_ref = _ref(q.float() * dq, k.float() * dk, v.float(), scale=scale, is_causal=True)
    _check(o, o_ref, torch.bfloat16, "e4m3", d_qk=d_qk)

    amax_o = torch.empty(1, device=dev, dtype=torch.float32)
    api_amax = SdpaFwdDslSm100(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=o,
        sample_amax_o=amax_o,
        is_causal=True,
        scale_softmax=scale,
        dtype_o=torch.bfloat16,
        split_kv=1,
        pv_bf16=True,
    )
    assert api_amax.check_support()
    api_amax.compile()
    api_amax.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, sf_q=sf_q, sf_k=sf_k, amax_o=amax_o)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        amax_o,
        o_ref.abs().max().reshape_as(amax_o),
        rtol=0.0,
        atol=_half_atol("e4m3", d_qk),
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=67)
def test_mxfp8_d192_split_kv_publishes_amax_from_combined_output():
    """D192 split-KV must leave Amax publication to the combine kernel."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    b, h_q, h_kv, s, d_qk, d_v = 1, 4, 2, 256, 192, 128
    dev = "cuda"
    scale = d_qk**-0.5
    qf = torch.randn(b, h_q, s, d_qk, device=dev) * 0.5
    kf = torch.randn(b, h_kv, s, d_qk, device=dev) * 0.5
    vf = torch.randn(b, h_kv, s, d_v, device=dev) * 0.5
    q, sf_q, dq, _ = _quantize(qf, b, h_q, s, d_qk, torch.float8_e4m3fn, columnwise=False)
    k, sf_k, dk, _ = _quantize(kf, b, h_kv, s, d_qk, torch.float8_e4m3fn, columnwise=False)
    v, sf_v, dv, _ = _quantize(vf, b, h_kv, s, d_v, torch.float8_e4m3fn, columnwise=True)
    o = torch.empty((b, h_q, s, d_v), device=dev, dtype=torch.bfloat16)
    amax_o = torch.empty(1, device=dev, dtype=torch.float32)
    api = SdpaFwdDslSm100(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=o,
        sample_amax_o=amax_o,
        is_causal=True,
        scale_softmax=scale,
        dtype_o=torch.bfloat16,
        split_kv=2,
    )

    assert api.check_support()
    api.compile()
    workspace = torch.empty(api.scratch_workspace_bytes(), device=dev, dtype=torch.uint8)
    api.execute(
        q_tensor=q,
        k_tensor=k,
        v_tensor=v,
        o_tensor=o,
        sf_q=sf_q,
        sf_k=sf_k,
        sf_v=sf_v,
        amax_o=amax_o,
        workspace=workspace,
    )
    torch.cuda.synchronize()

    o_ref = _ref(q.float() * dq, k.float() * dk, v.float() * dv, scale=scale, is_causal=True)
    _check(o, o_ref, torch.bfloat16, "e4m3", d_qk=d_qk)
    torch.testing.assert_close(
        amax_o,
        o_ref.abs().max().reshape_as(amax_o),
        rtol=0.0,
        atol=_half_atol("e4m3", d_qk),
    )


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
@pytest.mark.parametrize("mask", list(_MASKS))
@torch_fork_set_rng(seed=0)
def test_mxfp8_d256_masks(in_key, mask):
    O, O_ref, amax = _run(
        1,
        8,
        8,
        256,
        in_key,
        torch.float16,
        scale=1.0 / math.sqrt(256),
        sdpa_kwargs=_MASKS[mask],
        d_qk=256,
        d_v=256,
    )
    _check(O, O_ref, torch.float16, in_key, d_qk=256)
    assert amax.item() > 0.0


@pytest.mark.L0
@pytest.mark.skipif(_SM != 107, reason="the fused epilogue gate (O * sigmoid(G)) is served by the Rubin (sm107) d256 rows only")
@pytest.mark.parametrize("out_key", ["bf16", "e4m3"])
@torch_fork_set_rng(seed=0)
def test_mxfp8_gate_tail_graph_api(out_key):
    """Rubin graph-path e2e for the MXFP8 row's gate claim (PR-B): the 3-node tail
    on ``sdpa_mxfp8`` with a bf16 G (the only gate dtype the row lists), bf16 and
    e4m3 O -- the e4m3 O is UNSCALED (the block-scale kernel has no per-tensor
    scale_o).  ``Amax_O`` is an output of the sdpa NODE, which precedes the
    sigmoid/mul tail, so it is the amax of the UNGATED normalised O (pre-gate,
    pre-quant, in the O's own units): identical for G = +1e4 (sigmoid 1) and
    G = -1e4 (sigmoid 0, the returned O ~0), and identical to the no-gate
    graph's -- the fused kernel folds |h| = |u/2| and doubles once per tile,
    bit-exact in fp32.  It is also a COMPILE-TIME fact: a graph that does not
    request it (``amax=False`` -> has_amax_o=False, the atomic folded out)
    produces a BITWISE identical O."""
    scale = 1.0 / math.sqrt(256)
    gate = (torch.randn(1, 512, 8, 256, device="cuda") * 2.0).to(torch.bfloat16).transpose(1, 2)

    def _graph(gate_t, amax):
        torch.manual_seed(0)  # _run draws Q/K/V from the global RNG: every specialization must see the same problem
        return _run(1, 8, 8, 512, "e4m3", _OUT[out_key], scale=scale, sdpa_kwargs=dict(use_causal_mask=True), d_qk=256, d_v=256, gate=gate_t, amax=amax)

    runs = {amax: _graph(gate, amax) for amax in (True, False)}
    out, o_ref, a_o = runs[True]
    assert torch.isfinite(out.float()).all(), "unwritten / non-finite O cells"
    _check(out, o_ref, _OUT[out_key], "e4m3", d_qk=256)  # the reference O carries sigmoid(G); the reference amax is the UNGATED O's (below)
    assert torch.equal(runs[True].output, runs[False].output), "has_amax_o=False must fold only the Amax_O write out"
    assert runs[False].amax.item() == 0.0, "the unrequested Amax_O buffer was never bound, so never written"
    # Amax_O belongs to the sdpa node, so G must not move it: sigmoid(-1e4) == 0 zeroes the O the graph
    # returns while its Amax_O stays the ungated amax; sigmoid(+1e4) == 1 returns the ungated O; both -- and
    # the random-G run above -- equal the no-gate graph's Amax_O bit-for-bit (2 * max|u/2| == max|u| in
    # fp32), not merely within the 0.03 bound.  The no-gate run's ``reference`` IS the ungated O, so its
    # amax is the reference for every gated run (same seed, same problem).
    no_gate = _graph(None, True)
    pos = _graph(torch.full_like(gate, 1e4), True)
    neg = _graph(torch.full_like(gate, -1e4), True)
    ungated_ref_amax = no_gate.reference.abs().max().item()
    assert neg.output.float().abs().max().item() <= 1e-3, "sigmoid(G) == 0 must zero the O the graph returns"
    assert ungated_ref_amax > 0.06, f"the probe's ungated amax {ungated_ref_amax:.4f} is too small to tell from zero -- reshape it"
    assert abs(no_gate.amax.item() - ungated_ref_amax) <= 0.03, f"no-gate Amax_O {no_gate.amax.item():.4f} vs ref {ungated_ref_amax:.4f}"
    assert (
        abs(a_o.item() - ungated_ref_amax) <= 0.03
    ), f"gated Amax_O {a_o.item():.4f} vs the UNGATED reference {ungated_ref_amax:.4f}: the kernel reduced the GATED O"
    assert pos.amax.item() == neg.amax.item() == a_o.item() == no_gate.amax.item(), (
        f"Amax_O must be G-independent and equal the no-gate graph's: +1e4 {pos.amax.item():.6f}, -1e4 {neg.amax.item():.6f}, "
        f"random {a_o.item():.6f}, none {no_gate.amax.item():.6f}"
    )


@pytest.mark.L0
@pytest.mark.parametrize("in_key", _INS)
@torch_fork_set_rng(seed=981)
def test_mxfp8_d256_causal_rescale_every_tile(in_key):
    """Strict top-left causal at d256 runs the FUSED_CORR_SPLIT_P schedule: the
    correction warpgroup owns keys 64-127 while half 0 stores its P into S cols
    96-111 (P is aliased into the score slot).  That warpgroup only lags when
    it rescales O, so scale K by 8 per KV tile to force a rescale every
    iteration; the unordered kernel then reads clobbered scores and emits
    inf/NaN or O(1) errors on every run (GitHub #981 class, Rule S3).  Random
    data almost never rescales (RESCALE_THRESHOLD) and cannot catch this.

    The bound is the block-scale quantization floor of this deliberately
    extreme input, measured on the (named-barrier ordered) dense schedule:
    ~0.13-0.16 for e4m3; the racing kernel produced >= 1.75.
    """
    O, O_ref, amax = _run(1, 8, 8, 1024, in_key, torch.float16, scale=1.0 / math.sqrt(256), sdpa_kwargs=_MASKS["causal"], d_qk=256, d_v=256, k_tile_growth=8.0)
    assert torch.isfinite(O.float()).all(), "non-finite O: fused warpgroup read P-clobbered scores"
    diff = (O.float() - O_ref).abs().max().item()
    assert diff <= 0.3, f"max|O-ref|={diff:.4f} (quantization floor ~0.15; the unordered kernel gives >= 1.75)"
    assert amax.item() > 0.0


@pytest.mark.L1
@pytest.mark.parametrize("out_key", ["fp16", "bf16", "e4m3", "e5m2"])
@pytest.mark.parametrize("in_key", _INS)
@torch_fork_set_rng(seed=0)
def test_mxfp8_d256_output_dtypes(in_key, out_key):
    """D256 supports both MXFP8 encodings and every declared output dtype."""
    d = 256
    O, O_ref, amax = _run(
        1,
        8,
        8,
        256,
        in_key,
        _OUT[out_key],
        scale=1.0 / math.sqrt(d),
        sdpa_kwargs=dict(use_causal_mask=True),
        d_qk=d,
        d_v=d,
    )
    _check(O, O_ref, _OUT[out_key], in_key, d_qk=d)
    assert abs(amax.item() - O_ref.abs().max().item()) <= 0.03


@pytest.mark.L1
@pytest.mark.parametrize("in_key", _INS)
@torch_fork_set_rng(seed=59)
def test_mxfp8_d256_strided_stats(in_key):
    """D256 preserves the caller's non-contiguous Stats layout."""
    _check_mxfp8_strided_stats(256, 256, in_key)


@pytest.mark.L1
@pytest.mark.parametrize("in_key", _INS)
@torch_fork_set_rng(seed=0)
def test_mxfp8_d256_bottom_right_rectangular(in_key):
    """D256 bottom-right causal differs from top-left when S_q != S_kv."""
    d = 256
    O, O_ref, _ = _run_rect(
        2,
        8,
        128,
        256,
        in_key,
        torch.float16,
        scale=1.0 / math.sqrt(d),
        sdpa_kwargs=dict(use_causal_mask_bottom_right=True),
        d_qk=d,
        d_v=d,
    )
    _check(O, O_ref, torch.float16, in_key, d_qk=d)


@pytest.mark.L1
@pytest.mark.parametrize("d", [256, 512], ids=["d256", "d512"])
@torch_fork_set_rng(seed=0)
def test_mxfp8_wide_right_band(d):
    """Pin the wide MX-specific right-band lowerings."""
    out, out_ref, _ = _run(
        1,
        4,
        4,
        256,
        "e4m3",
        torch.float16,
        scale=1.0 / math.sqrt(d),
        sdpa_kwargs=dict(diagonal_band_right_bound=40),
        d_qk=d,
        d_v=d,
    )
    _check(out, out_ref, torch.float16, "e4m3", d_qk=d)


@pytest.mark.L1
@pytest.mark.parametrize("d", [256, 512], ids=["d256", "d512"])
@pytest.mark.parametrize("in_key", _INS)
@pytest.mark.parametrize("causal", [False, True])
@torch_fork_set_rng(seed=0)
def test_mxfp8_wide_dense_padding(d, in_key, causal):
    """Wide MXFP8 kernels mask a per-batch partial KV tile in dense storage."""
    sdpa_kwargs = dict(use_causal_mask=True) if causal else {}
    O, O_ref, amax = _run(
        2,
        8,
        8,
        256,
        in_key,
        torch.float16,
        scale=1.0 / math.sqrt(d),
        sdpa_kwargs=sdpa_kwargs,
        seq_lens_kv=[256, 192],
        stats=False,
        d_qk=d,
        d_v=d,
    )
    _check(O, O_ref, torch.float16, in_key, d_qk=d)
    assert abs(amax.item() - O_ref.abs().max().item()) <= 0.03


@pytest.mark.L0
@pytest.mark.parametrize("d", [256, 512], ids=["d256", "d512"])
@torch_fork_set_rng(seed=0)
def test_mxfp8_dense_q_trim_bottom_right_multiwave(d):
    """More Q tiles than SMs (3 x 8 heads x 8 tiles of 128 rows), batches of
    different lengths, bottom-right: a persistent worker crosses batches, and
    the second tile's diagonal and keyless-row test must use ITS batch's
    lengths (the wide templates take the next tile's bounds from the scheduler
    payload and used to keep the previous batch's lengths)."""
    result = _run(
        3,
        8,
        8,
        1024,
        "e4m3",
        torch.float16,
        scale=1.0 / math.sqrt(d),
        sdpa_kwargs=dict(use_causal_mask_bottom_right=True),
        seq_lens_q=[1024, 640, 0],
        seq_lens_kv=[800, 1024, 512],
        d_qk=d,
        d_v=d,
        return_lse=True,
    )
    _check(result.output, result.reference, torch.float16, "e4m3", d_qk=d)
    # batch 0: diagonal 800 - 1024 = -224 -> rows 0..223 keyless; batch 1: diagonal +384, no keyless row; batch 2 empty
    assert (result.output[0, :, :224] == 0).all() and torch.isneginf(result.stats[0, :, :224]).all()
    assert (result.output[1, :, 640:] == 0).all() and (result.output[2] == 0).all()
    torch.testing.assert_close(result.stats[0, :, 224:], result.reference_stats[0, :, 224:], atol=5e-2, rtol=3e-2)
    torch.testing.assert_close(result.stats[1, :, :640], result.reference_stats[1, :, :640], atol=5e-2, rtol=3e-2)


@pytest.mark.L0
@pytest.mark.parametrize("d", [128, 256], ids=["d128", "d256"])
@torch_fork_set_rng(seed=0)
def test_mxfp8_dense_q_trim_bottom_right_sink(d):
    """Keyless rows (bottom-right, S_kv < S_q) WITH a sink: the row's mass is
    the sink alone, so LSE = sink_logit and O = 0 -- a kernel that lets the
    masked scores through its softmax returns NaN there. scale * log2(e) > 1 makes the
    scaled sentinel overflow, the case that trips a finite-sentinel mask."""
    sink = torch.randn(1, 8, 1, 1, dtype=torch.float32, device="cuda")
    result = _run(
        3,
        8,
        8,
        256,
        "e4m3",
        torch.float16,
        scale=0.75,  # scale * log2(e) > 1: the scaled finite sentinel overflows, the case a sum-based dead-row test misses
        sdpa_kwargs=dict(use_causal_mask_bottom_right=True),
        sink=sink,
        seq_lens_q=[256, 129, 0],
        seq_lens_kv=[200, 256, 128],
        d_qk=d,
        d_v=d,
        return_lse=True,
    )
    # scale 0.75 sharpens the softmax (scores ~12x the 1/sqrt(d) norm), which widens the e4m3 P-quantization
    # noise on the WIDE flavor: develop measures the same 0.06 on d256 with this graph, so it gets 0.08 here.
    diff = (result.output.float() - result.reference).abs().max().item()
    assert diff <= (0.05 if d == 128 else 0.08), f"max|O-ref|={diff:.4f}"
    assert (result.output[0, :, :56] == 0).all() and (result.output[1, :, 129:] == 0).all() and (result.output[2] == 0).all()
    assert torch.isfinite(result.stats[0]).all(), "keyless rows with a sink must carry the sink logit, not NaN"
    torch.testing.assert_close(result.stats[0, :, :56], sink.view(8, 1).expand(8, 56), atol=1e-4, rtol=0)
    torch.testing.assert_close(result.stats[0], result.reference_stats[0], atol=5e-2, rtol=3e-2)
    assert torch.isneginf(result.stats[1, :, 129:]).all() and torch.isneginf(result.stats[2]).all()


@pytest.mark.L0
@pytest.mark.parametrize("d, d_v", [(128, 128), (192, 128), (256, 256), (512, 512)], ids=["d128", "d192_128", "d256", "d512"])
@pytest.mark.parametrize("band", [False, True], ids=["br", "br_band"])
@torch_fork_set_rng(seed=0)
def test_mxfp8_dense_q_trim_bottom_right(d, d_v, band):
    """Short per-batch Q under bottom-right causal (and with a left band): the
    diagonal anchors at (seq_len_q[b], seq_len_kv[b]) for the VALID rows, so the
    tile bounds and the per-element mask must use the same per-batch Q length
    -- a mask still anchored at the padded S_q drops keys the valid rows must
    see. Batches: full Q, mid-tile Q (129 of 256), empty Q."""
    kw = dict(use_causal_mask_bottom_right=True)
    if band:
        kw["diagonal_band_left_bound"] = 65  # window = 64
    result = _run(
        3,
        8,
        8,
        256,
        "e4m3",
        torch.float16,
        scale=1.0 / math.sqrt(d),
        sdpa_kwargs=kw,
        seq_lens_q=[256, 129, 0],
        seq_lens_kv=[200, 256, 128],
        d_qk=d,
        d_v=d_v,
        return_lse=True,
    )
    _check(result.output, result.reference, torch.float16, "e4m3", d_qk=d)
    assert (result.output[1, :, 129:] == 0).all() and (result.output[2] == 0).all()
    # batch 0: S_kv 200 < S_q 256 under bottom-right puts rows 0..55 above the diagonal -- keyless inside a
    # live tile, so exactly O = 0 / LSE = -inf (a finite mask sentinel would leave them a uniform average).
    assert (result.output[0, :, :56] == 0).all() and torch.isneginf(result.stats[0, :, :56]).all()
    assert torch.isneginf(result.stats[1, :, 129:]).all() and torch.isneginf(result.stats[2]).all()
    torch.testing.assert_close(result.stats[0], result.reference_stats[0], atol=5e-2, rtol=3e-2)
    torch.testing.assert_close(result.stats[1, :, :129], result.reference_stats[1, :, :129], atol=5e-2, rtol=3e-2)


@pytest.mark.L0
@pytest.mark.parametrize("d, d_v", [(128, 128), (192, 128), (256, 256), (512, 512)], ids=["d128", "d192_128", "d256", "d512"])
@torch_fork_set_rng(seed=0)
def test_mxfp8_dense_q_trim_stats_sink(d, d_v):
    """Short dense Q rows trim O/LSE even when a sink makes softmax finite --
    on every SM100 MXFP8 flavor."""
    sink = torch.randn(1, 8, 1, 1, dtype=torch.float32, device="cuda")
    result = _run(
        2,
        8,
        8,
        256,
        "e4m3",
        torch.float16,
        scale=1.0 / math.sqrt(d),
        sdpa_kwargs={},
        sink=sink,
        seq_lens_q=[129, 0],
        seq_lens_kv=[200, 256],
        d_qk=d,
        d_v=d_v,
        return_lse=True,
    )
    _check(result.output, result.reference, torch.float16, "e4m3", d_qk=d)
    assert (result.output[0, :, 129:] == 0).all() and (result.output[1] == 0).all()
    assert torch.isneginf(result.stats[0, :, 129:]).all() and torch.isneginf(result.stats[1]).all()
    torch.testing.assert_close(result.stats, result.reference_stats, atol=5e-2, rtol=3e-2)
    assert abs(result.amax.item() - result.reference.abs().max().item()) <= 0.03


@pytest.mark.L1
@pytest.mark.parametrize("in_key", _INS)
@torch_fork_set_rng(seed=0)
def test_mxfp8_d256_gqa_sink(in_key):
    """D256 GQA composes with a per-query-head attention sink."""
    d = 256
    sink = torch.randn(1, 8, 1, 1, dtype=torch.float32, device="cuda")
    O, O_ref, _ = _run(
        2,
        8,
        2,
        256,
        in_key,
        torch.float16,
        scale=1.0 / math.sqrt(d),
        sdpa_kwargs=dict(use_causal_mask=True),
        sink=sink,
        d_qk=d,
        d_v=d,
    )
    _check(O, O_ref, torch.float16, in_key, d_qk=d)


@pytest.mark.L0
@pytest.mark.parametrize(
    ("out_key", "with_sink"),
    [("fp16", False), ("e4m3", True)],
    ids=["fp16-nosink", "fp8-sink"],
)
@torch_fork_set_rng(seed=0)
def test_mxfp8_d256_leading_zero_length_kv(out_key: str, with_sink: bool):
    """A leading zero-length KV batch must produce zero O and a finite amax."""
    d = 256
    sink = torch.randn(1, 8, 1, 1, dtype=torch.float32, device="cuda") if with_sink else None
    result = _run(
        2,
        8,
        8,
        256,
        "e4m3",
        _OUT[out_key],
        scale=1.0 / math.sqrt(d),
        sdpa_kwargs={},
        sink=sink,
        seq_lens_kv=[0, 256],
        stats=False,
        d_qk=d,
        d_v=d,
        poison_tmem_before_execute=True,
    )
    _check(result.output, result.reference, _OUT[out_key], "e4m3", d_qk=d)
    assert (result.output[0] == 0).all()
    assert abs(result.amax.item() - result.reference.abs().max().item()) <= 0.03


@pytest.mark.L0
@pytest.mark.parametrize("in_key", _INS)
@pytest.mark.parametrize("mask", list(_MASKS))
@torch_fork_set_rng(seed=0)
def test_mxfp8_d512_masks(in_key, mask):
    """D512 covers both V/O slices for every declared mask."""
    d = 512
    O, O_ref, amax = _run(
        1,
        4,
        4,
        256,
        in_key,
        torch.float16,
        scale=1.0 / math.sqrt(d),
        sdpa_kwargs=_MASKS[mask],
        d_qk=d,
        d_v=d,
    )
    _check(O, O_ref, torch.float16, in_key, d_qk=d)
    assert abs(amax.item() - O_ref.abs().max().item()) <= 0.03


@pytest.mark.L1
@pytest.mark.parametrize("out_key", ["fp16", "bf16", "e4m3", "e5m2"])
@torch_fork_set_rng(seed=0)
def test_mxfp8_d512_output_dtypes(out_key):
    """D512 writes each declared output type across the slice boundary."""
    d = 512
    O, O_ref, amax = _run(
        1,
        4,
        4,
        256,
        "e4m3",
        _OUT[out_key],
        scale=1.0 / math.sqrt(d),
        sdpa_kwargs=dict(use_causal_mask=True),
        d_qk=d,
        d_v=d,
    )
    _check(O, O_ref, _OUT[out_key], "e4m3", d_qk=d)
    assert abs(amax.item() - O_ref.abs().max().item()) <= 0.03


@pytest.mark.L1
@torch_fork_set_rng(seed=0)
def test_mxfp8_d512_bottom_right_rectangular():
    """D512 bottom-right causal uses the shifted diagonal for rectangular attention."""
    d = 512
    O, O_ref, _ = _run_rect(
        1,
        4,
        128,
        256,
        "e4m3",
        torch.float16,
        scale=1.0 / math.sqrt(d),
        sdpa_kwargs=dict(use_causal_mask_bottom_right=True),
        d_qk=d,
        d_v=d,
    )
    _check(O, O_ref, torch.float16, "e4m3", d_qk=d)


@pytest.mark.L1
@torch_fork_set_rng(seed=0)
def test_mxfp8_d512_gqa_sink():
    """D512 composes unpacked GQA with a per-query-head sink."""
    d = 512
    sink = torch.randn(1, 4, 1, 1, dtype=torch.float32, device="cuda")
    result = _run(
        2,
        4,
        2,
        256,
        "e5m2",
        torch.float16,
        scale=1.0 / math.sqrt(d),
        sdpa_kwargs=dict(use_causal_mask=True),
        sink=sink,
        d_qk=d,
        d_v=d,
    )
    _check(result.output, result.reference, torch.float16, "e5m2", d_qk=d)
    assert abs(result.amax.item() - result.reference.abs().max().item()) <= 0.03


@pytest.mark.L1
@pytest.mark.parametrize("left_bound", [128, 129], ids=["reuse-p", "two-pass"])
@torch_fork_set_rng(seed=0)
def test_mxfp8_d512_dense_multi_wave_alias(left_bound):
    """Persistent dense GQA preserves Q/O ownership in both Dv pipelines."""
    d = 512
    result = _run(
        1,
        64,
        1,
        512,
        "e4m3",
        torch.float16,
        scale=1.0 / math.sqrt(d),
        sdpa_kwargs=dict(use_causal_mask=True, diagonal_band_left_bound=left_bound),
        d_qk=d,
        d_v=d,
    )
    assert torch.isfinite(result.output).all()
    _check(result.output, result.reference, torch.float16, "e4m3", d_qk=d)
    assert abs(result.amax.item() - result.reference.abs().max().item()) <= 0.03


@pytest.mark.L1
@torch_fork_set_rng(seed=0)
def test_mxfp8_d512_leading_zero_length_kv():
    """Both D512 output slices must zero an empty-KV batch."""
    d = 512
    result = _run(
        2,
        4,
        4,
        256,
        "e4m3",
        torch.float16,
        scale=1.0 / math.sqrt(d),
        sdpa_kwargs={},
        seq_lens_kv=[0, 256],
        stats=False,
        d_qk=d,
        d_v=d,
        poison_tmem_before_execute=True,
    )
    _check(result.output, result.reference, torch.float16, "e4m3", d_qk=d)
    assert (result.output[0] == 0).all()
    assert abs(result.amax.item() - result.reference.abs().max().item()) <= 0.03


@pytest.mark.L1
@torch_fork_set_rng(seed=59)
def test_mxfp8_d512_strided_stats():
    """The two D512 passes preserve the caller's Stats layout."""
    _check_mxfp8_strided_stats(512, 512, "e4m3")


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
@pytest.mark.parametrize(
    ("out_key", "with_sink"),
    [("fp16", False), ("e4m3", True)],
    ids=["fp16-nosink", "fp8-sink"],
)
@torch_fork_set_rng(seed=0)
def test_mxfp8_d192_d128_leading_zero_length_kv(out_key: str, with_sink: bool):
    """A leading zero-length KV batch must produce zero O and a finite amax."""
    d_qk, d_v = 192, 128
    sink = torch.randn(1, 8, 1, 1, dtype=torch.float32, device="cuda") if with_sink else None
    result = _run(
        2,
        8,
        8,
        256,
        "e4m3",
        _OUT[out_key],
        scale=1.0 / math.sqrt(d_qk),
        sdpa_kwargs={},
        sink=sink,
        seq_lens_kv=[0, 256],
        # The MXFP8 engine intentionally declines padding plus Stats; O and
        # amax exercise the affected epilogue independently of that contract.
        stats=False,
        d_qk=d_qk,
        d_v=d_v,
        poison_tmem_before_execute=True,
    )
    _check(result.output, result.reference, _OUT[out_key], "e4m3", d_qk=d_qk)
    assert (result.output[0] == 0).all()
    assert abs(result.amax.item() - result.reference.abs().max().item()) <= 0.03


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
@pytest.mark.parametrize("d_qk,d_v", [(128, 128), (192, 128), (256, 256)], ids=["d128", "d192_d128", "d256"])
@pytest.mark.parametrize("h_q,h_kv", [(8, 4), (8, 2), (8, 1)], ids=["g2", "g4", "mqa"])
@torch_fork_set_rng(seed=0)
def test_mxfp8_dense_gqa_ratios(d_qk, d_v, h_q, h_kv):
    """DENSE GQA/MQA across every ratio and shape -- see the FP8 twin.

    Block-scale adds a reason to care: K and V carry PER-HEAD scale-factor
    planes, so head sharing has to index the SF tensors by the KV head while
    indexing Q's SF by the query head.  MQA (h_kv=1) collapses every KV-side SF
    lookup onto plane 0, which is precisely where a stride that is really
    `h_q`-based instead of `h_kv`-based still reads in-bounds and returns the
    wrong exponents."""
    scale = 1.0 / math.sqrt(d_qk)
    O, O_ref, _ = _run(2, h_q, h_kv, 256, "e4m3", torch.float16, scale=scale, sdpa_kwargs=dict(use_causal_mask=True), d_qk=d_qk, d_v=d_v)
    _check(O, O_ref, torch.float16, "e4m3")


@pytest.mark.L0
@pytest.mark.parametrize(
    "d_qk,d_v",
    [(128, 128), (192, 128)],
)
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
    _select_engine(g, engine_name(arch=_ARCH, mxfp8=True))
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
    o_ref = _ref(Q8.float() * dqq, K8.float() * dqk, V8.float() * dqv, scale=scale, **_ref_from_sdpa(sdpa_kwargs))
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
        d_scale = _cdiv(_cdiv(d, _BLOCK), 4) * 4
        return empty.to(fp8), empty.float(), torch.zeros((h, 0, 128 * d_scale), dtype=torch.uint8, device=t_1hsd.device)
    data_d, dq_d, swz_d, data_s, dq_s, swz_s = quantize_to_mxfp8(t_1hsd, 1, h, s, d, _BLOCK, fp8, with_ref=True)
    n_tiles = _cdiv(s, 128)
    if columnwise:
        # Columnwise F8_128x4 is plane-major for D > 128. THD's packed
        # contract keeps every sequence tile contiguous, so transpose
        # [D/128, H, tile, 512] into [H, tile, D/128, 512] before sequences
        # are concatenated in cu_seqlens order.
        planes = d // 128
        sf = swz_s.view(torch.uint8).reshape(planes, h, n_tiles, 512).permute(1, 2, 0, 3).contiguous().reshape(h, n_tiles, -1)
        return data_s, dq_s.reshape(1, h, s, d), sf
    return data_d, dq_d.reshape(1, h, s, d), swz_d.view(torch.uint8).reshape(h, n_tiles, -1)


def _run_thd(
    seq_lens_q,
    seq_lens_kv,
    H_q,
    H_kv,
    in_key,
    out_dt,
    *,
    scale,
    causal=False,
    bottom_right=False,
    swa_window=None,
    sink=None,
    stats=False,
    cu_lens=False,
    declare_totals=False,
    d_qk=128,
    d_v=128,
):
    """THD/varlen: packed [T,H,D] Q/K/V/O + ragged offsets + per-batch lengths
    (or their cu prefix-sum form) + PACKED per-sequence-TILE-padded SF."""
    import cudnn

    dev = "cuda"
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
        Qf = torch.randn(1, H_q, s_q, d_qk, device=dev) * 0.5
        Kf = torch.randn(1, H_kv, s_kv, d_qk, device=dev) * 0.5
        Vf = torch.randn(1, H_kv, s_kv, d_v, device=dev) * 0.5
        q8, dqq, sfq = _quantize_seq(Qf, H_q, s_q, d_qk, fp8, columnwise=False)
        k8, dqk, sfk = _quantize_seq(Kf, H_kv, s_kv, d_qk, fp8, columnwise=False)
        v8, dqv, sfv = _quantize_seq(Vf, H_kv, s_kv, d_v, fp8, columnwise=True)
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

    def _dense_buf(packed, s_max, h, d, dt):
        # Dense-capacity storage; packed tokens in the leading elements (THD
        # contract). The capacity tail is NaN-POISONED (test_mhas_v2 parity):
        # the last sequence's KV tile steps past the packed total, and those
        # tail loads must land as zeros through the setup kernel's
        # packed-total-clamped K/V descriptors — a leaked NaN would wipe the
        # tile via BMM2's P·V (0 · NaN == NaN).
        stride = (s_max * h * d, d, h * d, 1)
        stor = torch.full((B * s_max * h * d,), float("nan"), device=dev, dtype=torch.float32).to(dt)
        stor[: packed.numel()] = packed.reshape(-1)
        return stor, stor.as_strided((B, h, s_max, d), stride), stride

    _, q_gpu, stride_q = _dense_buf(q_pk, S_max_q, H_q, d_qk, q_pk.dtype)
    _, k_gpu, stride_k = _dense_buf(k_pk, S_max_kv, H_kv, d_qk, k_pk.dtype)
    _, v_gpu, stride_v = _dense_buf(v_pk, S_max_kv, H_kv, d_v, v_pk.dtype)
    stride_o = (S_max_q * H_q * d_v, d_v, H_q * d_v, 1)
    o_stor = torch.zeros(B * S_max_q * H_q * d_v, device=dev, dtype=out_dt)
    o_gpu = o_stor.as_strided((B, H_q, S_max_q, d_v), stride_o)
    amax = torch.zeros(1, 1, 1, 1, device=dev, dtype=torch.float32)

    slq = torch.tensor(seq_lens_q, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    slk = torch.tensor(seq_lens_kv, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    cuq_t = torch.tensor(cu_q, dtype=torch.int32, device=dev).view(B + 1, 1, 1, 1)
    cuk_t = torch.tensor(cu_k, dtype=torch.int32, device=dev).view(B + 1, 1, 1, 1)
    ro_q = (torch.tensor(cu_q, dtype=torch.int64, device=dev) * H_q * d_qk).view(B + 1, 1, 1, 1)
    ro_k = (torch.tensor(cu_k, dtype=torch.int64, device=dev) * H_kv * d_qk).view(B + 1, 1, 1, 1)
    ro_v = (torch.tensor(cu_k, dtype=torch.int64, device=dev) * H_kv * d_v).view(B + 1, 1, 1, 1)
    ro_o = (torch.tensor(cu_q, dtype=torch.int64, device=dev) * H_q * d_v).view(B + 1, 1, 1, 1)

    io = getattr(cudnn.data_type, _CUDNN_ITYPE[in_key])
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tq = g.tensor(dim=[B, H_q, S_max_q, d_qk], stride=list(stride_q), data_type=io, name="q")
    tk = g.tensor(dim=[B, H_kv, S_max_kv, d_qk], stride=list(stride_k), data_type=io, name="k")
    tv = g.tensor(dim=[B, H_kv, S_max_kv, d_v], stride=list(stride_v), data_type=io, name="v")
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

    dsc_qk = _cdiv(_cdiv(d_qk, _BLOCK), 4) * 4
    dsc_v = _cdiv(_cdiv(d_v, _BLOCK), 4) * 4
    dvp = dsc_v * _BLOCK
    dq = _sf((B, H_q, _cdiv(S_max_q, 128) * 128, dsc_qk))
    dk = _sf((B, H_kv, _cdiv(S_max_kv, 128) * 128, dsc_qk))
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
    assert not (bottom_right and causal)
    if bottom_right:
        kw["use_causal_mask_bottom_right"] = True
    elif causal or swa_window is not None:
        kw["use_causal_mask"] = True
    if swa_window is not None:
        kw["diagonal_band_left_bound"] = swa_window + 1
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
        vro: ro_v,
        oro: ro_o,
    }
    if sink is not None:
        st = g.tensor_like(sink)
        kw["sink_token"] = st
        vp[st] = sink
    if declare_totals:
        # Packed token totals: exact THD extents instead of buffer-inferred.
        kw.update(max_total_seq_len_q=sum(seq_lens_q), max_total_seq_len_kv=sum(seq_lens_kv))
    o, stats_t, amax_o = g.sdpa_mxfp8(**kw)
    o.set_output(True).set_dim([B, H_q, S_max_q, d_v]).set_stride(list(stride_o)).set_data_type(getattr(cudnn.data_type, _CUDNN_OTYPE[out_dt]))
    o.set_ragged_offset(oro)
    amax_o.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    stats_stor = None
    if stats:
        # Ragged Stats, packed token-major TH1 ([t, h]; offsets = cu_q * h_q).
        stats_stor = torch.zeros(B * S_max_q * H_q, dtype=torch.float32, device=dev)
        stats_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
        stats_t.set_dim((B, H_q, S_max_q, 1)).set_stride((S_max_q * H_q, 1, H_q, 1))
        stats_ro_t = (ro_q.flatten() // d_qk).view(B + 1, 1, 1, 1).contiguous()
        stats_ro = g.tensor_like(stats_ro_t, name="stats_ro")
        stats_t.set_ragged_offset(stats_ro)
        vp[stats_ro] = stats_ro_t
        vp[stats_t] = stats_stor

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    _select_engine(g, engine_name(arch=_ARCH, mxfp8=True))
    g.check_support()
    g.build_plans()
    vp.update({o: o_gpu, amax_o: amax})
    g.execute(vp, torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8))
    torch.cuda.synchronize()

    o_ref = torch.zeros(T_q, H_q, d_v, device=dev, dtype=torch.float32)
    for b in range(B):
        if cu_q[b + 1] == cu_q[b] or cu_k[b + 1] == cu_k[b]:
            # Zero-length Q contributes no rows; zero-length KV leaves every
            # row of the sequence dead — O := 0 (o_ref is pre-zeroed).
            continue
        qd = q8_seqs[b].float() * dq_seqs[b]
        kd = k8_seqs[b].float() * dk_seqs[b]
        vd = v8_seqs[b].float() * dv_seqs[b]
        ref_kw = dict(
            is_causal=causal or bottom_right or swa_window is not None,
            bottom_right=bottom_right,
            swa_window=swa_window,
        )
        if sink is not None:
            ref_kw["sinks"] = sink.flatten()
        ob = _ref(qd, kd, vd, scale=scale, **ref_kw)
        o_ref[cu_q[b] : cu_q[b + 1]] = ob.squeeze(0).permute(1, 0, 2)

    o_out = o_stor[: T_q * H_q * d_v].reshape(T_q, H_q, d_v)
    lse_out = stats_stor[: T_q * H_q].reshape(T_q, H_q) if stats else None
    return o_out, o_ref, amax, lse_out


@_skip_thd_mxfp8_on_rubin
@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_mxfp8_thd_declared_totals():
    """The MXFP8 THD path accepts ``sdpa_mxfp8(max_total_seq_len_q/kv=...)``.

    A ragged graph cannot express its packed token total (dims stay
    ``(B, H, S_max, D)`` with the per-sequence starts in a device offset
    tensor), so the execute path otherwise infers an upper bound from the
    bound buffers. Declaring the totals binds exact extents; results must be
    unchanged."""
    scale = 1.0 / math.sqrt(128)
    seq_q, seq_kv = [200, 150], [200, 150]

    def _run(declare):
        torch.manual_seed(0)  # each call draws its own inputs -- pin them so the two runs are comparable
        return _run_thd(seq_q, seq_kv, 8, 8, _INS[0], torch.float16, scale=scale, declare_totals=declare)

    o_dec, ref_dec, amax_dec, _ = _run(True)
    o_inf, _, _, _ = _run(False)
    _check(o_dec, ref_dec, torch.float16, _INS[0])
    assert abs(amax_dec.item() - ref_dec.abs().max().item()) <= 0.03
    assert torch.equal(o_dec, o_inf), "declaring the packed totals must not change O"


@_skip_thd_mxfp8_on_rubin
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


@_skip_thd_mxfp8_on_rubin
@pytest.mark.L0
@pytest.mark.parametrize("d_qk,d_v", [(128, 128), (192, 128), (256, 256), (512, 512)], ids=["d128", "d192_d128", "d256", "d512"])
@pytest.mark.parametrize("in_key", _INS)
@torch_fork_set_rng(seed=0)
def test_mxfp8_thd_multi_unit_per_cta(monkeypatch, in_key, d_qk, d_v):
    """THD where a cluster claims more than one unit (issue #618).

    The persistent grid is machine-sized, so a cluster pulls units repeatedly
    off the device-bounded counter; every other MXFP8 THD case fits one unit
    per cluster and never re-enters the K/V pipeline (nor the per-sequence SF
    tile bases for a second range). FROST_THD_CLUSTERS pins the grid to 4
    clusters so the claim loop runs deep on any device."""
    monkeypatch.setenv("FROST_THD_CLUSTERS", "4")
    scale = 1.0 / math.sqrt(d_qk)
    lens = [1024, 768, 512, 256]
    o_out, o_ref, amax, _ = _run_thd(lens, lens, 8, 8, in_key, torch.float16, scale=scale, causal=True, d_qk=d_qk, d_v=d_v)
    _check(o_out, o_ref, torch.float16, in_key)
    assert abs(amax.item() - o_ref.abs().max().item()) <= 0.03


@_skip_thd_mxfp8_on_rubin
@pytest.mark.L0
@pytest.mark.parametrize("in_key", _INS)
@pytest.mark.parametrize("causal", [False, True])
@torch_fork_set_rng(seed=0)
def test_mxfp8_d256_thd(in_key, causal):
    """D256 MXFP8 THD uses packed per-sequence SF tiles."""
    scale = 1.0 / math.sqrt(256)
    o_out, o_ref, amax, _ = _run_thd(
        [160, 96],
        [160, 96],
        8,
        8,
        in_key,
        torch.float16,
        scale=scale,
        causal=causal,
        d_qk=256,
        d_v=256,
    )
    _check(o_out, o_ref, torch.float16, in_key, d_qk=256)
    assert abs(amax.item() - o_ref.abs().max().item()) <= 0.03


@_skip_thd_mxfp8_on_rubin
@pytest.mark.L0
@pytest.mark.parametrize("d", [128, 256, 512], ids=["d128", "d256", "d512"])
@pytest.mark.parametrize("in_key", _INS)
@pytest.mark.parametrize("bottom_right", [False, True])
@torch_fork_set_rng(seed=0)
def test_mxfp8_thd_sliding_window(d, in_key, bottom_right):
    """MXFP8 THD sliding window uses each sequence's local causal diagonal."""
    q_lens = [173, 97] if bottom_right else [257, 193]
    kv_lens = [257, 193] if bottom_right else q_lens
    scale = 1.0 / math.sqrt(d)
    o_out, o_ref, amax, lse = _run_thd(
        q_lens,
        kv_lens,
        8,
        2,
        in_key,
        torch.float16,
        scale=scale,
        causal=not bottom_right,
        bottom_right=bottom_right,
        swa_window=73,
        stats=True,
        d_qk=d,
        d_v=d,
    )
    _check(o_out, o_ref, torch.float16, in_key, d_qk=d)
    assert abs(amax.item() - o_ref.abs().max().item()) <= 0.03
    assert lse is not None and torch.isfinite(lse).all()


@_skip_thd_mxfp8_on_rubin
@pytest.mark.L0
@pytest.mark.parametrize("d", [128, 256, 512], ids=["d128", "d256", "d512"])
@torch_fork_set_rng(seed=0)
def test_mxfp8_thd_cross_gqa(d):
    """THD cross-attention (unequal packed Q and K/V totals) with GQA heads."""
    scale = 1.0 / math.sqrt(d)
    o_out, o_ref, _, _ = _run_thd([64, 200], [256, 128], 8, 2, "e4m3", torch.float16, scale=scale, d_qk=d, d_v=d)
    _check(o_out, o_ref, torch.float16, "e4m3", d_qk=d)


@_skip_thd_mxfp8_on_rubin
@pytest.mark.L0
@pytest.mark.parametrize("d", [128, 256, 512], ids=["d128", "d256", "d512"])
@torch_fork_set_rng(seed=0)
def test_mxfp8_thd_sink(d):
    """THD causal + attention sink."""
    scale = 1.0 / math.sqrt(d)
    sink = torch.randn(1, 8, 1, 1, dtype=torch.float32, device="cuda")
    o_out, o_ref, _, _ = _run_thd([200, 150], [200, 150], 8, 8, "e4m3", torch.float16, scale=scale, causal=True, sink=sink, d_qk=d, d_v=d)
    _check(o_out, o_ref, torch.float16, "e4m3", d_qk=d)


@_skip_thd_mxfp8_on_rubin
@pytest.mark.L0
@pytest.mark.parametrize("d", [128, 256, 512], ids=["d128", "d256", "d512"])
@torch_fork_set_rng(seed=0)
def test_mxfp8_thd_stats(d):
    """THD + generate_stats: the ragged token-major TH1 LSE is written next to O."""
    scale = 1.0 / math.sqrt(d)
    o_out, o_ref, _, lse = _run_thd([200, 150], [200, 150], 8, 8, "e4m3", torch.float16, scale=scale, causal=True, stats=True, d_qk=d, d_v=d)
    _check(o_out, o_ref, torch.float16, "e4m3", d_qk=d)
    assert lse is not None and torch.isfinite(lse).all()


@_skip_thd_mxfp8_on_rubin
@pytest.mark.L0
@pytest.mark.parametrize("d", [128, 256, 512], ids=["d128", "d256", "d512"])
@torch_fork_set_rng(seed=0)
def test_mxfp8_thd_zero_len_kv(d):
    """Zero-length Q and KV sequences (test_mhas_v2 ragged parity): the
    zero-KV sequence's rows are dead — the epilogue must come back O := 0,
    not the unwritten O TMEM (garbage survives `* inv_sum(=0)` when NaN)."""
    scale = 1.0 / math.sqrt(d)
    o_out, o_ref, _, _ = _run_thd([126, 0, 60], [0, 83, 77], 8, 8, "e4m3", torch.float16, scale=scale, d_qk=d, d_v=d)
    _check(o_out, o_ref, torch.float16, "e4m3", d_qk=d)


@_skip_thd_mxfp8_on_rubin
@pytest.mark.L0
@pytest.mark.parametrize("d", [128, 256, 512], ids=["d128", "d256", "d512"])
@torch_fork_set_rng(seed=0)
def test_mxfp8_thd_cu_seq_len(d):
    """THD via the (B+1,) cu_seq_len prefix-sum length form."""
    scale = 1.0 / math.sqrt(d)
    o_out, o_ref, _, _ = _run_thd(
        [200, 150],
        [180, 120],
        8,
        8,
        "e4m3",
        torch.float16,
        scale=scale,
        cu_lens=True,
        d_qk=d,
        d_v=d,
    )
    _check(o_out, o_ref, torch.float16, "e4m3", d_qk=d)


@_skip_thd_mxfp8_on_rubin
@pytest.mark.L0
@pytest.mark.parametrize("in_key", _INS)
@torch_fork_set_rng(seed=0)
def test_mxfp8_d192_d128_thd_cross_gqa_stats(in_key):
    """D192 THD covers rank-5 K descriptors, packed SF, GQA, and ragged LSE."""
    d_qk, d_v = 192, 128
    scale = 1.0 / math.sqrt(d_qk)
    o_out, o_ref, amax, lse = _run_thd(
        [129, 257],
        [385, 193],
        8,
        2,
        in_key,
        torch.float16,
        scale=scale,
        causal=True,
        stats=True,
        d_qk=d_qk,
        d_v=d_v,
    )
    _check(o_out, o_ref, torch.float16, in_key, d_qk=d_qk)
    assert abs(amax.item() - o_ref.abs().max().item()) <= 0.03
    assert lse is not None and torch.isfinite(lse).all()


@_skip_thd_mxfp8_on_rubin
@pytest.mark.L0
@pytest.mark.parametrize("in_key", _INS)
@pytest.mark.parametrize("mask", ["causal_br", "swa"])
@torch_fork_set_rng(seed=0)
def test_mxfp8_d192_d128_thd_mask_variants(in_key, mask):
    """D192 THD supports bottom-right causal and sliding-window masks."""
    d_qk, d_v = 192, 128
    scale = 1.0 / math.sqrt(d_qk)
    seq_q = [129, 257]
    seq_kv = [385, 193] if mask == "causal_br" else seq_q
    mask_kw = {"bottom_right": True} if mask == "causal_br" else {"swa_window": 64}
    o_out, o_ref, amax, _ = _run_thd(
        seq_q,
        seq_kv,
        8,
        2,
        in_key,
        torch.float16,
        scale=scale,
        d_qk=d_qk,
        d_v=d_v,
        **mask_kw,
    )
    _check(o_out, o_ref, torch.float16, in_key, d_qk=d_qk)
    assert abs(amax.item() - o_ref.abs().max().item()) <= 0.03


@_skip_thd_mxfp8_on_rubin
@pytest.mark.L0
@pytest.mark.parametrize("d_qk", [128, 192])
@pytest.mark.parametrize("in_key", _INS)
@torch_fork_set_rng(seed=0)
def test_mxfp8_thd_nonfinite_v_sf_padding(d_qk, in_key, monkeypatch):
    """Fully padded V scale blocks must not poison zero-filled V rows."""
    original = _quantize_seq

    def quantize(*args, **kwargs):
        data, dequant, sf = original(*args, **kwargs)
        if kwargs["columnwise"]:
            # S=129: block 0 of the second F8_128x4 tile is partially valid;
            # blocks 1..3 are physical padding. Each lane-group is 16 bytes.
            offsets = [(row % 32) * 16 + (row // 32) * 4 + col for row in range(128) for col in (1, 2, 3)]
            sf[:, 1, torch.tensor(offsets, device=sf.device)] = 0xFF
        return data, dequant, sf

    monkeypatch.setitem(globals(), "_quantize_seq", quantize)
    out, ref, _, _ = _run_thd(
        [129],
        [129],
        2,
        2,
        in_key,
        torch.float16,
        scale=1.0 / math.sqrt(d_qk),
        d_qk=d_qk,
        d_v=128,
    )
    assert torch.isfinite(out).all()
    _check(out, ref, torch.float16, in_key, d_qk=d_qk)


@pytest.mark.L1
@pytest.mark.parametrize("d_qk,d_v", [(128, 128), (192, 128), (256, 256), (512, 512)])
@torch_fork_set_rng(seed=931)
def test_mxfp8_stats_log2_every_flavor(d_qk, d_v):
    """Both bases agree with analytic constant logits on every kernel flavor.

    Nonzero logits catch scaling only log(sum_exp) and forgetting the maximum.
    The same inputs exercise both cache specializations and leave O unchanged.
    """
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    b, h, s = 1, 2, 256
    qf = torch.full((b, h, s, d_qk), 0.125, device="cuda")
    kf = torch.full_like(qf, 0.25)
    vf = torch.full((b, h, s, d_v), 0.5, device="cuda")
    q, sfq, dq, _ = _quantize(qf, b, h, s, d_qk, torch.float8_e4m3fn, columnwise=False)
    k, sfk, dk, _ = _quantize(kf, b, h, s, d_qk, torch.float8_e4m3fn, columnwise=False)
    v, sfv, dv, _ = _quantize(vf, b, h, s, d_v, torch.float8_e4m3fn, columnwise=True)
    expected = (q.float() * dq)[0, 0, 0].double().dot((k.float() * dk)[0, 0, 0].double()) * d_qk**-0.5 + math.log(s)
    q, k, v = [x.transpose(1, 2).contiguous().transpose(1, 2) for x in (q, k, v)]
    outputs = []
    for log2 in (False, True):
        o = torch.empty(b, s, h, d_v, device="cuda", dtype=torch.float16).transpose(1, 2)
        lse = torch.full((b, h, s), float("nan"), device="cuda")
        api = SdpaFwdDslSm100(sample_q=q, sample_k=k, sample_v=v, sample_o=o, sample_lse=lse, scale_softmax=d_qk**-0.5, stats_log2=log2, split_kv=1)
        assert api.check_support()
        api.compile()
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, lse_tensor=lse, sf_q=sfq, sf_k=sfk, sf_v=sfv)
        torch.cuda.synchronize()
        want = expected * (math.log2(math.e) if log2 else 1.0)
        torch.testing.assert_close(lse, torch.full_like(lse, want.item()), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(o, torch.full_like(o, 0.5), atol=1e-2, rtol=1e-2)
        outputs.append(o)
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)


# --- Block-scaled O (sf_o): NVFP4 / MXFP8 output on the MXFP8-input kernel -------------
def _check_block_scaled(res: _BlockScaledRunResult, blk: int, in_key: str):
    """Kernel dequant vs fp32 reference: within the MXFP8 pipeline tolerance (in
    scale_o units) plus three times the pure block-quantization floor of the
    reference itself; per-plane pad rows zero; Amax_O the pre-scale amax."""
    from sdpa.block_scale_o_ref import quantize_o_mxfp8, quantize_o_nvfp4

    ref = res.reference
    _, _, ref_q = (quantize_o_nvfp4 if blk == 16 else quantize_o_mxfp8)(ref)
    floor = (ref_q - ref).abs().max().item()
    diff = (res.output - ref).abs().max().item()
    atol = max(_half_atol(in_key, 128) * res.scale_o, 3.0 * floor)
    assert not torch.isnan(res.output).any(), "NaN in dequantized O"
    assert diff <= atol, f"max|O-ref|={diff:.4f} > {atol:.4f} (quantization floor {floor:.4f})"
    assert res.sf_pad_ok, "sf_o pad rows past S must be zero"
    assert abs(res.amax - res.reference_amax) <= 0.03, f"amax_o {res.amax:.4f} vs ref {res.reference_amax:.4f}"


@pytest.mark.L0
@pytest.mark.parametrize("sf_o_layout", ["planes", "token_major"])
@pytest.mark.parametrize("mask", ["none", "causal"])
@pytest.mark.parametrize("mode", list(_BLOCK_SCALED_O))
@torch_fork_set_rng(seed=71)
def test_mxfp8_block_scaled_output(mode, mask, sf_o_layout):
    """Block-scaled O epilogue (sf_o) on the d128 MXFP8-input kernel: FP4 O + E4M3/16
    scales with scale_o as the FP4 global scale, or E4M3 O + UE8M0/32 scales without
    a scale. The causal case runs S = 300 -- a partially valid tail tile whose
    per-plane pad rows must come back zero (the MXFP8 kernel takes a dense KV
    tail only under a mask that covers it, so the unmasked case runs S = 256) --
    B > 1 the plane / token-major row bookkeeping, GQA the head -> plane mapping.
    Runs on the whole SM100 line (_ARCH picks the Rubin kernel there)."""
    blk = _BLOCK_SCALED_O[mode]
    res = _run(
        2,
        4,
        2,
        300 if mask == "causal" else 256,
        "e4m3",
        torch.float8_e4m3fn,
        scale=1.0 / math.sqrt(128),
        sdpa_kwargs=_MASKS[mask],
        block_scaled_o=mode,
        sf_o_layout=sf_o_layout,
        scale_o=3.0 if mode == "nvfp4" else None,
    )
    _check_block_scaled(res, blk, "e4m3")


@pytest.mark.L0
@torch_fork_set_rng(seed=72)
def test_mxfp8_block_scaled_mxfp8_out_with_scale_o():
    """The UE8M0 mode accepts an optional scale_o too (folded into O, divided back
    out of Amax_O), so both modes share one contract with sdpa_fp8."""
    res = _run(1, 2, 2, 256, "e4m3", torch.float8_e4m3fn, scale=1.0 / math.sqrt(128), sdpa_kwargs={}, block_scaled_o="mxfp8", scale_o=0.5)
    _check_block_scaled(res, 32, "e4m3")


@pytest.mark.L0
@torch_fork_set_rng(seed=75)
def test_mxfp8_block_scaled_omitted_scale_o_first_execute_captured_then_eager():
    """UE8M0 mode without scale_o: the identity scale is a compile-time fold of
    the kernel, not a device constant the plan creates on its first execute.
    First execute CAPTURED (never replayed), then an eager execute -- O and the
    pre-scale Amax_O must be right (review on #1180: the cached-dummy version
    returned all-zero O and a NaN amax here)."""
    res = _run(2, 4, 2, 256, "e4m3", torch.float8_e4m3fn, scale=1.0 / math.sqrt(128), sdpa_kwargs={}, block_scaled_o="mxfp8", scale_o=None, capture_first=True)
    _check_block_scaled(res, 32, "e4m3")


@pytest.mark.L0
@torch_fork_set_rng(seed=73)
def test_mxfp8_block_scaled_fp4_requires_scale_o():
    """An FP4 O without scale_o is rejected at validate: the E4M3 block scale alone
    cannot span O's range, and sdpa_mxfp8 has no other per-tensor O scale."""
    import cudnn

    with pytest.raises((cudnn.cudnnGraphNotSupportedError, RuntimeError, ValueError)):
        _run(1, 2, 2, 256, "e4m3", torch.float8_e4m3fn, scale=1.0 / math.sqrt(128), sdpa_kwargs={}, block_scaled_o="nvfp4", scale_o=None)


@pytest.mark.L0
@torch_fork_set_rng(seed=74)
def test_mxfp8_block_scaled_output_declines_wide_flavor():
    """sf_o is a d128-only epilogue: a d256 MXFP8 graph requesting it has no engine."""
    import cudnn

    with pytest.raises((cudnn.cudnnGraphNotSupportedError, RuntimeError, ValueError)):
        _run(1, 2, 2, 256, "e4m3", torch.float8_e4m3fn, scale=1.0 / 16, sdpa_kwargs={}, d_qk=256, d_v=256, block_scaled_o="mxfp8")


# ============================================================================ sm100 d128 MXFP8: the exp2 MUFU / FMA split and the Amax_O fold, pinned on the sm_100a SASS
# Both are invisible to every numerics test (O to the bf16 rounding, LSE within 6e-6, Amax_O exact before and after) and
# worth +10.9 % at S=16K on B200 (`sm100/prefill_d128_mxfp8.py`, the `_E2E_*` block).  The only tripwire is the SASS, so
# this pin compiles the kernel for sm_100a here (`CUTE_DSL_ARCH=sm_100a` needs no matching device -- the module-level
# `requires_blackwell` is what confines it to the Blackwell-line lanes) and counts the instructions.  SKIPS when no
# nvdisasm on $CUDA_PATH/bin or $PATH decodes the cubin; a compile failure is a FAIL.
_SM100_D128_MXFP8_SASS_PROBE = textwrap.dedent("""
    import glob, os, subprocess, sys
    dump, cands = sys.argv[1], sys.argv[2:]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump          # read once, at the first cutlass import
    os.environ["CUTE_DSL_KEEP"] = "cubin"            # keep the cubin, disassemble it ourselves
    os.environ["CUTE_DSL_ARCH"] = "sm_100a"       # unconditional: an inherited value would pin the wrong target's SASS
    os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"  # a compiled-plan cache HIT skips ptxas and dumps no cubin
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module, supported_cgas_for
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    # The PRODUCTION geometry from the adapter itself (cga2), the cc 10.0 specialization (fused_ldtm_stat=False: the
    # manual tcgen05_ld + software row-max, i.e. B200), E4M3 in, BF16 out, GQA 24/8, Stats + Amax_O -- the shape the
    # split was tuned and measured on (B=1 H=24/8 S=16K dense).
    (cta_mma,) = supported_cgas_for((128, 128), fp8=True, device_cc=(10, 0), pertensor=False)
    params = TemplateParams(dtype_qkv=0, dtype_o=2, cta_mma=cta_mma, qh_per_kh=3, sched_policy=0, fused_ldtm_stat=False, emit_amax_o=True)
    mod = _load_sm100_kernel_module((128, 128), params, fp8=True, pertensor=False, rubin=False)
    # MUFU.EX2 the kernel must carry: per softmax body one alpha exp2 plus the non-emulated columns, traced once per
    # softmax warpgroup (sub-tile) -- derived from the module, so the pin follows the pattern rather than a literal.
    n_bodies = 2 if getattr(mod.CFG, "SOFTMAX_WARPGROUPS", 2) == 2 else 1
    print("EXPECT_MUFU_EX2", n_bodies * (mod.CFG.TILE_N - mod._E2E_EMULATED_COLS + 1))
    print("EMULATED_COLS", mod._E2E_EMULATED_COLS)
    mod.compile(b=1, qh=24, kh=8, sq=16384, skv=16384, has_lse=True)
    cubins = sorted(glob.glob(os.path.join(dump, "*.cubin")), key=os.path.getmtime)
    if not cubins:
        print("FAIL no cubin dumped into", dump, os.listdir(dump)); sys.exit(3)
    nvd = None
    for c in cands:
        try:
            proc = subprocess.run([c, "-c", cubins[-1]], capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.SubprocessError) as exc:
            print("REJECT", c, "->", repr(exc)); continue
        if proc.returncode == 0 and proc.stdout.strip():
            nvd = c; print("NVDISASM", c); break
        print("REJECT", c, "->", (proc.stderr.strip().splitlines() or [str(proc.returncode)])[-1])
    if nvd is None:
        print("SKIP no nvdisasm candidate decodes the cubin"); sys.exit(0)
    sass = subprocess.run([nvd, "-c", cubins[-1]], capture_output=True, text=True, check=True).stdout.splitlines()
    def cnt(*subs):
        return sum(1 for ln in sass if all(sb in ln for sb in subs))
    print("SASS MUFU_EX2", cnt("MUFU.EX2"))
    print("SASS FFMA2", cnt(" FFMA2"))
    print("SASS FADD2", cnt(" FADD2"))
    print("SASS FSETP", cnt("FSETP"))
    print("SASS FSEL", cnt("FSEL"))
    print("SASS FMNMX3", cnt("FMNMX3"))
    print("SASS STL", cnt("STL"))
    print("SASS LDL", cnt("LDL"))
    print("SASS LINES", len(sass))
    """)

# sm_100a counts of the shipped kernel (2026-09-22, PRODUCTION geometry, this probe's shape), all MEASURED on the
# trace-compiled cubin that is md5-identical to the one A/B'd on B200:  MUFU.EX2 258 -> 194 (2 x 97), FFMA2 130 -> 226
# and FADD2 126 -> 222 (+3 each per emulated pair), FSETP 782 -> 19, FSEL 780 -> 270 (the 256 row_dead selects stay),
# FMNMX3 124 -> 252, STL 3 / LDL 3 unchanged (none in the softmax loops), REG 128.  Slack 16 tolerates unrelated ptxas
# drift and still catches one site: a fold site regressing to compare+select adds >= 384 FSETP, one emulated pair
# falling back to MUFU adds 2 MUFU.EX2 and drops 3 FFMA2 (the MUFU count is pinned EXACTLY, derived from the module).
_SM100_SASS_SLACK = 16
_SM100_D128_MXFP8_SASS_PINS = {"FFMA2": 226, "FADD2": 222, "FSETP": 19, "FMNMX3": 252, "STL": 3, "LDL": 3}


def _nvdisasm_candidates():
    cands = []
    if os.environ.get("CUDA_PATH"):
        cands.append(os.path.join(os.environ["CUDA_PATH"], "bin", "nvdisasm"))
    on_path = shutil.which("nvdisasm")
    if on_path:
        cands.append(on_path)
    return [c for c in dict.fromkeys(cands) if os.path.isfile(c) and os.access(c, os.X_OK)]


@pytest.mark.L0
def test_sm100_d128_mxfp8_exp2_split_and_amax_fold_sass_pins(tmp_path):
    """32 of the 128 softmax columns are evaluated on the FMA pipe (MUFU.EX2 == 2 x 97 exactly, derived from the
    module's `_E2E_EMULATED_COLS`; FFMA2 / FADD2 at or above the measured 226 / 222 minus slack) and the Amax_O fold
    is FMNMX3, not a compare+select chain (FSETP <= 19 + 16, FMNMX3 >= 252 - 16); no new spill (STL / LDL <= 3, the
    pre-existing count).  Compiled for sm_100a at the production geometry -- no Blackwell device needed for the
    compile, only for this module's gate."""
    cands = _nvdisasm_candidates()
    if not cands:
        pytest.skip("no nvdisasm executable to try (CUDA_PATH unset and none on PATH)")
    dump = tmp_path / "sm100a_mxfp8_d128"
    dump.mkdir()
    argv = [sys.executable, "-c", _SM100_D128_MXFP8_SASS_PROBE, str(dump), *cands]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=1500)
    assert proc.returncode == 0, f"sm_100a trace-compile of the d128 mxfp8 kernel failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    out = proc.stdout.splitlines()
    if any(ln.startswith("SKIP") for ln in out):
        pytest.skip(str([ln for ln in out if ln.startswith(("SKIP", "REJECT"))]))
    stats = {ln.split()[1]: int(ln.split()[2]) for ln in out if ln.startswith("SASS ") and len(ln.split()) == 3 and ln.split()[2].isdigit()}
    expect = {ln.split()[0]: int(ln.split()[1]) for ln in out if ln.startswith(("EXPECT_MUFU_EX2 ", "EMULATED_COLS "))}
    print(f"\nsm100 d128 mxfp8 sm_100a SASS: {stats}; expected MUFU.EX2 {expect}")
    assert expect["EMULATED_COLS"] == 32, f"the shipped pattern emulates 32 of 128 columns, the module says {expect['EMULATED_COLS']}"
    assert (
        stats["MUFU_EX2"] == expect["EXPECT_MUFU_EX2"] == 194
    ), f"MUFU.EX2 {stats['MUFU_EX2']} != {expect['EXPECT_MUFU_EX2']}: an emulated pair fell back to MUFU (or the split leaked)"
    for key in ("FFMA2", "FADD2", "FMNMX3"):
        assert (
            stats[key] >= _SM100_D128_MXFP8_SASS_PINS[key] - _SM100_SASS_SLACK
        ), f"{key} {stats[key]} < {_SM100_D128_MXFP8_SASS_PINS[key]} - {_SM100_SASS_SLACK}: {stats}"
    assert (
        stats["FSETP"] <= _SM100_D128_MXFP8_SASS_PINS["FSETP"] + _SM100_SASS_SLACK
    ), f"{stats['FSETP']} FSETP: the Amax_O fold is lowering to compare+select again"
    assert stats["STL"] <= _SM100_D128_MXFP8_SASS_PINS["STL"] and stats["LDL"] <= _SM100_D128_MXFP8_SASS_PINS["LDL"], f"new spills: {stats}"


# ============================================================================ sm100 d128 MXFP8: Stats is the exact fp32 log-sum-exp (fp64 reference), with the exp2 split in place
@pytest.mark.L0
@pytest.mark.skipif(
    _SM is None or not (100 <= _SM <= 106),
    reason="the sm100 MXFP8 d128 kernel serves cc 10.0-10.6 (the Rubin sibling has its own pin in test_sdpa_fwd_dsl_sm107.py)",
)
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
def test_mxfp8_d128_stats_is_the_exact_softmax_lse_sm100(causal):
    """B=1 H=24/8 S=2048 (the split's tuning shape at a reference-friendly length): (1) the PUBLISHED Stats is within
    1e-4 of the fp64 log-sum-exp of the DEQUANTIZED inputs -- the exp2 emulation's 8.8e-5 per-element error averages
    to ~6e-6 on a dense row and ~2e-5 on a causal one (MEASURED on B200: 6.06e-6 / 1.90e-5; the all-MUFU kernel
    1.6e-6, the cuDNN backend kernel 6.05e-6 / 3.07e-5); (2) O is finite, sentinel-free and within the bf16 output
    rounding of the reference; (3) O is bit-identical with and without Stats (sdpa-invariants s4)."""
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100, supported_cgas_for

    torch.manual_seed(0)
    b, hq, hkv, s, d = 1, 24, 8, 2048, 128
    dev = "cuda"
    qf = torch.randn(b, hq, s, d, device=dev) * 0.5
    kf = torch.randn(b, hkv, s, d, device=dev) * 0.5
    vf = torch.randn(b, hkv, s, d, device=dev) * 0.5

    def mx(x, h, columnwise):
        data_d, sf_d, swz_d, data_s, sf_s, swz_s = quantize_to_mxfp8(x.contiguous(), b, h, s, d, 32, torch.float8_e4m3fn, with_ref=True)
        data, sf, swz = (data_s, sf_s, swz_s) if columnwise else (data_d, sf_d, swz_d)
        # sf_*_ref are the per-element fp32 DEQUANT SCALES [b, h, s, d]; fp64 so the reference logits are exact
        # (fp32 matmul runs in TF32 in the DLFW containers, a 1e-4-class LSE error on a causal row with few columns).
        deq = data.double().reshape(b, h, s, d) * sf.double().reshape(b, h, s, d)
        return data.permute(0, 2, 1, 3).contiguous().transpose(1, 2), deq, swz.contiguous()

    q8, q_deq, sfq = mx(qf, hq, False)
    k8, k_deq, sfk = mx(kf, hkv, False)
    v8, v_deq, sfv = mx(vf, hkv, True)
    (cga,) = supported_cgas_for((d, d), fp8=True, device_cc=torch.cuda.get_device_capability(), pertensor=False)
    outs = {}
    lse = torch.full((b, hq, s), float("nan"), device=dev, dtype=torch.float32)
    for with_stats in (True, False):
        out = torch.full((b, s, hq, d), 1.5e30, device=dev, dtype=torch.bfloat16).transpose(1, 2)
        api = SdpaFwdDslSm100(
            q8, k8, v8, out, lse if with_stats else None, scale_softmax=d**-0.5, is_causal=causal, pertensor_fp8=False, dtype_o=torch.bfloat16, cga=cga
        )
        assert api.check_support()
        api.compile()
        api.execute(q8, k8, v8, out, lse_tensor=lse if with_stats else None, sf_q=sfq, sf_k=sfk, sf_v=sfv)
        torch.cuda.synchronize()
        outs[with_stats] = out.clone()
    assert torch.equal(outs[True], outs[False]), "O must not depend on whether Stats is requested"
    rep = hq // hkv
    logits = q_deq @ k_deq.repeat_interleave(rep, 1).transpose(-1, -2) * d**-0.5
    if causal:
        logits = logits.masked_fill(~torch.tril(torch.ones(s, s, dtype=torch.bool, device=dev)), float("-inf"))
    lse_ref = torch.logsumexp(logits, dim=-1)
    o_ref = torch.softmax(logits, dim=-1) @ v_deq.repeat_interleave(rep, 1)
    o = outs[True]
    assert not (o.float() == 1.5e30).any(), "sentinel survived: O rows never written"
    assert torch.isfinite(o.float()).all() and torch.isfinite(lse).all(), "non-finite O / unwritten LSE rows"
    bf16_floor = (o_ref - o_ref.to(torch.bfloat16).double()).abs().max().item()
    d_o = (o.double() - o_ref).abs().max().item()
    assert d_o <= 4 * bf16_floor + 1e-3, f"max |dO| {d_o:.3e} vs the bf16 rounding floor {bf16_floor:.3e} of the fp64 reference"
    err = (lse.double() - lse_ref).abs()
    print(
        f"\nsm100 d128 mxfp8 {'causal' if causal else 'dense'}: max|dLSE| {err.max().item():.3e} rms {err.pow(2).mean().sqrt().item():.3e}, max|dO| {d_o:.3e} (bf16 floor {bf16_floor:.3e})"
    )
    assert (
        err.max().item() <= 1e-4
    ), f"Stats is not the exact log-sum-exp: max |dLSE| {err.max().item():.3e}, rms {err.pow(2).mean().sqrt().item():.3e} (the exp2 emulation reads ~6e-6 dense / ~2e-5 causal; a quantized-sum LSE ~1e-3..1e-2)"
