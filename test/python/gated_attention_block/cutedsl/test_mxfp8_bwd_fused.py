# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The MXFP8 block backward's two FUSED small-kernel launches (``kernels/mxfp8_bwd_fused.py``): the PROLOGUE (scalar init without the
dP reciprocal + dY amax partials + the Q / K rebuild with its MX epilogue -- q8 / sf_q / q_T8 / sf_q_T / k8 / sf_k / k_T8 / sf_k_T from
one 32-token tile -- + the MX rowwise v8) and the EPILOGUE (dW_norm reduce at two columns per block + the dual-axis canonical dqkvg
cast), each one ``@cute.kernel`` dispatching its jobs by block range.

The contract is BITWISE: every job of a fused launch is the ``@cute.jit`` body its standalone kernel runs, so every output of the
fused launch must equal the standalone launches' byte for byte -- the e4m3 payloads, the E8M0 scale-factor blobs (every byte of
every 128-row unit written, pad blocks ``0x00``), the partials, the slots, the fp32 ``dW_norm`` (the reduce at two columns per block
is the SAME fixed-order chain as the standalone ``(8, 128)`` reduce).  Every destination is 0xFF-poisoned before the fused launch, so
an unwritten byte cannot pass as a stale zero.  Needs the TMA kernel (sm_90+), the fp8 cvt (sm_89+) and ``cvt.rp.satfinite.ue8m0x2``
(sm_100+); the typed contracts run on any CUDA device.
"""

import os
import shutil
import subprocess
import sys
import textwrap

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

from cudnn.gated_attention_block.kernels import mxfp8_bwd_fused as F  # noqa: E402
from cudnn.gated_attention_block.kernels import qk_norm_rope_bwd as NB  # noqa: E402
from cudnn.gated_attention_block.kernels import qk_norm_rope_tma as TMA  # noqa: E402
from cudnn.gated_attention_block.kernels import quantize as Q  # noqa: E402
from cudnn.gated_attention_block.kernels import quantize_mxfp8 as MX  # noqa: E402
from cudnn.gated_attention_block.kernels.proj_gemm import sf_blob_bytes  # noqa: E402

E4 = torch.float8_e4m3fn
U8 = torch.uint8
_EPS = 1e-6


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
requires_mx_tma = pytest.mark.skipif(
    _cc() is None or _cc() < (10, 0), reason="the fused MXFP8 launches need TMA (sm_90+), the fp8 cvt (sm_89+) and cvt.rp.satfinite.ue8m0x2 (sm_100+)"
)
_QK_NORM = pytest.mark.parametrize("qk_norm", [True, False], ids=["norm", "rope_only"])


def _stream():
    return torch.cuda.current_stream().cuda_stream


def _make(t, h_q, h_kv, d, rope_dim, d_model, seed=0):
    """A random bf16 slab ``[T, N]`` (its Q / GATE / K / V bands as the block hands them), the norm weights, cos / sin, dY."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*shape, std=1.0):
        return (torch.randn(*shape, generator=g, device="cuda", dtype=torch.float32) * std).to(torch.bfloat16)

    n = 2 * h_q * d + 2 * h_kv * d
    slab = rnd(t, n, std=2.0)
    offs = dict(q=0, g=h_q * d, k=2 * h_q * d, v=2 * h_q * d + h_kv * d)
    heads = dict(q=h_q, g=h_q, k=h_kv, v=h_kv)
    bands = {k: torch.as_strided(slab, (t, heads[k], d), (n, d, 1), storage_offset=offs[k]) for k in offs}
    ang = torch.randn(t, rope_dim // 2, generator=g, device="cuda")
    emb = torch.cat((ang, ang), -1)
    return dict(
        slab=slab, n=n, bands=bands, w_q=rnd(d), w_k=rnd(d), cos=emb.cos().to(torch.bfloat16), sin=emb.sin().to(torch.bfloat16), dy=rnd(t, d_model, std=4.0)
    )


def _poisoned(b, s, h, d):
    """A 0xFF-poisoned compact e4m3 payload and SDPA-layout SF blob for ``(b, s, h)``."""
    pay = torch.full((b * s, h, d), 0xFF, dtype=U8, device="cuda").view(E4)
    sf = torch.full((MX.sf_bytes(b, h, s, d),), 0xFF, dtype=U8, device="cuda")
    return pay, sf


def _standalone_mx(src, *, batch, seq_len, h, d):
    """The standalone rowwise and columnwise SDPA-layout quantizes of a compact ``[T, H, D]`` buffer (0xFF-poisoned destinations)."""
    st = _stream()
    out = []
    for axis in ("row", "col"):
        r = MX.compile_quantize_mxfp8(dtype_in=src.dtype, h=h, d=d, axis=axis)
        dst, sf = _poisoned(batch, seq_len, h, d)
        MX.run_quantize_mxfp8(r, src, dst, sf, batch=batch, seq_len=seq_len, stream=st)
        out += [dst, sf]
    torch.cuda.synchronize()
    return tuple(out)


def _eq(a, b):
    return torch.equal(a.view(U8) if a.dtype == E4 else a, b.view(U8) if b.dtype == E4 else b)


@requires_mx_tma
@_QK_NORM
@pytest.mark.parametrize("b, s", [(1, 1000), (2, 1008), (1, "cap+1")], ids=["s1000", "s1008-b2", "capped-amax"])
def test_prologue_is_bitwise_the_standalone_launches(qk_norm, b, s):
    """One prologue launch == {init_scalars (the zeroing and the plan-time constants out of kernel arguments; no dP reciprocal), the
    amax partials pass, the bf16 TMA rebuild followed by the FOUR standalone MXFP8 quantizes of q16 / k16 (rowwise + columnwise),
    the standalone rowwise v8 quantize of the slab's V band} run standalone: the slots, every partial (the words past ``n_amax``
    untouched), the eight Q / K payload + SF outputs and v8 + sf_v -- byte for byte, every destination 0xFF-poisoned (every SF byte
    of every ceil128(S) unit written, pad blocks ``0x00``: ``s1000`` / ``s1008`` are ragged, ``s1008`` at B = 2 the cell where an
    unpredicated pad-row store would land in batch 1).  ``prologue_grid`` is the three job widths.  The ``capped-amax`` cell sizes its
    token count from the compiled recipe so the dY amax job has one row group more than its persistent cap on ANY part."""
    h_q, h_kv, d, rope_dim, d_model = 8, 2, 256, 64, 512
    st = _stream()
    n_slots, c0, n_c = 29, 15, 14  # the block's slot layout: fifteen pre-existing slots, then the fourteen plan-time constants
    r = F.compile_mxfp8_bwd_prologue(
        dtype=torch.bfloat16,
        h_q=h_q,
        h_kv=h_kv,
        d_model=d_model,
        d=d,
        rope_dim=rope_dim,
        eps=_EPS,
        apply_norm=qk_norm,
        n_slots=n_slots,
        const_slot0=c0,
        n_consts=n_c,
    )
    assert (r.n_slots, r.const_slot0, r.n_consts, r.threads) == (n_slots, c0, n_c, F.PROLOGUE_THREADS)
    capped = s == "cap+1"
    if capped:
        s = ((r.n_ctas_cap + 1) * r.rows_per_cta + r.h_dy - 1) // r.h_dy  # EXACTLY cap + 1 dY row groups on any part
    t = b * s
    m = _make(t, h_q, h_kv, d, rope_dim, d_model, seed=t)
    w_q, w_k = (m["w_q"], m["w_k"]) if qk_norm else (None, None)
    n_amax, n_rec, n_v8 = F.prologue_grid(r, b, s)
    n_blk, n_sft, n_tiles = TMA.mx_tile_counts(b, s, h_q, h_kv)
    assert n_amax == Q.n_partials_for(Q.compile_amax_partials(dtype_in=torch.bfloat16, h=d_model // d, d=d), t)
    # the rebuild job's persistent cap is the MX arm's RESIDENCY (SMs x 5 norm / x 6 RoPE-only), never the amax job's SMs x 8
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    assert r.n_rec_cap == sms * TMA.mx_rebuild_ctas_per_sm(qk_norm) and r.n_rec_cap < r.n_ctas_cap
    assert n_rec == max(1, min(n_tiles, r.n_rec_cap)) and n_v8 == b * h_kv * n_sft
    dy_groups = (t * (d_model // d) + r.rows_per_cta - 1) // r.rows_per_cta
    assert n_amax == max(1, min(dy_groups, r.n_ctas_cap))
    if capped:
        assert dy_groups == r.n_ctas_cap + 1 and n_amax == r.n_ctas_cap, "the capped cell must stride the amax job (one row group more than CTAs)"
    slots = torch.full((n_slots,), float("nan"), device="cuda")
    consts = tuple(0.5 + 0.25 * i for i in range(n_c))  # exact in fp32: the bitwise check asks no rounding question
    partials = torch.full((r.n_ctas_cap + 3,), float("nan"), device="cuda")
    q8, sf_q = _poisoned(b, s, h_q, d)
    q_T8, sf_q_T = _poisoned(b, s, h_q, d)
    k8, sf_k = _poisoned(b, s, h_kv, d)
    k_T8, sf_k_T = _poisoned(b, s, h_kv, d)
    v8, sf_v = _poisoned(b, s, h_kv, d)
    n = F.run_mxfp8_bwd_prologue(
        r,
        slots=slots,
        dy=m["dy"].view(t, d_model // d, d),
        partials=partials,
        q=m["bands"]["q"],
        k=m["bands"]["k"],
        w_q=w_q,
        w_k=w_k,
        cos=m["cos"],
        sin=m["sin"],
        q8=q8,
        sf_q=sf_q,
        q_T8=q_T8,
        sf_q_T=sf_q_T,
        k8=k8,
        sf_k=sf_k,
        k_T8=k_T8,
        sf_k_T=sf_k_T,
        v=m["bands"]["v"],
        v8=v8,
        sf_v=sf_v,
        batch=b,
        seq_len=s,
        stream=st,
        consts=consts,
    )
    torch.cuda.synchronize()
    assert n == n_amax
    # init (the descale_dp-less artifact)
    slots_ref = torch.full((n_slots,), float("nan"), device="cuda")
    Q.run_init_scalars(Q.compile_init_scalars(n_slots, c0, n_c, descale_dp=False), slots_ref, consts=consts, stream=st)
    # partials
    part_ref = torch.full_like(partials, float("nan"))
    Q.run_amax_partials(Q.compile_amax_partials(dtype_in=torch.bfloat16, h=d_model // d, d=d), m["dy"].view(t, d_model // d, d), part_ref, stream=st)
    # the rebuild: the bf16 TMA artifact, then the four standalone MXFP8 quantizes
    kw = dict(dtype=torch.bfloat16, h_q=h_q, h_kv=h_kv, d=d, rope_dim=rope_dim, eps=_EPS, want_rstd=False, tile_rows=8, apply_norm=qk_norm)
    q16, k16 = torch.empty(t, h_q, d, dtype=torch.bfloat16, device="cuda"), torch.empty(t, h_kv, d, dtype=torch.bfloat16, device="cuda")
    TMA.run_qk_norm_rope_tma(TMA.compile_qk_norm_rope_tma(**kw), m["bands"]["q"], m["bands"]["k"], q16, k16, w_q, w_k, m["cos"], m["sin"], stream=st)
    torch.cuda.synchronize()
    rq_d, rq_sf, cq_d, cq_sf = _standalone_mx(q16, batch=b, seq_len=s, h=h_q, d=d)
    rk_d, rk_sf, ck_d, ck_sf = _standalone_mx(k16, batch=b, seq_len=s, h=h_kv, d=d)
    rv_d, rv_sf, _, _ = _standalone_mx(m["bands"]["v"], batch=b, seq_len=s, h=h_kv, d=d)
    assert torch.equal(slots, slots_ref) and torch.equal(slots[:c0], torch.zeros(c0, device="cuda"))
    assert torch.equal(slots[c0 : c0 + n_c], torch.tensor(consts, dtype=torch.float32, device="cuda"))
    assert torch.equal(partials[:n], part_ref[:n]) and torch.isnan(partials[n:]).all() and torch.equal(partials[:n].max(), m["dy"].float().abs().amax())
    for name, got, want in (
        ("q8", q8, rq_d),
        ("sf_q", sf_q, rq_sf),
        ("q_T8", q_T8, cq_d),
        ("sf_q_T", sf_q_T, cq_sf),
        ("k8", k8, rk_d),
        ("sf_k", sf_k, rk_sf),
        ("k_T8", k_T8, ck_d),
        ("sf_k_T", sf_k_T, ck_sf),
        ("v8", v8, rv_d),
        ("sf_v", sf_v, rv_sf),
    ):
        assert _eq(got, want), f"the fused prologue's {name} differs from the standalone launches'"
    for name, blob in (("sf_q", sf_q), ("sf_q_T", sf_q_T), ("sf_k", sf_k), ("sf_k_T", sf_k_T), ("sf_v", sf_v)):
        assert not (blob == 0xFF).any(), f"{name}: an SF byte was never written"


@requires_mx_tma
@pytest.mark.parametrize("want_dw", [True, False], ids=["reduce", "no-reduce"])
@pytest.mark.parametrize("halves", ["both", "row", "col"])
@pytest.mark.parametrize("t", [992, 2016])
def test_epilogue_is_bitwise_the_standalone_launches(want_dw, halves, t):
    """One epilogue launch == {the standalone ``(8, 128)`` dW_norm reduce, the standalone rowwise canonical quantize of dqkvg (the
    ``(T, N)`` blob), the standalone TRANSPOSED canonical quantize (the ``[N, T]`` matrix + the ``(N, T)`` blob)} run standalone:
    ``dW_q_norm`` / ``dW_k_norm`` (the per-column fixed-order chain does not depend on how many columns share a block), the e4m3
    payloads and the two blobs byte for byte, on 0xFF-poisoned destinations (every blob byte written: the ceil128 pad blocks are
    ``0x00``); a half that is not traced is refused at execute.  ``epilogue_grid`` is the two job widths."""
    want_row, want_col = halves in ("both", "row"), halves in ("both", "col")
    if not (want_dw or want_row or want_col):
        pytest.skip("nothing to launch")
    h_q, h_kv, d, rope_dim = 8, 2, 256, 64
    m = _make(t, h_q, h_kv, d, rope_dim, 512, seed=9 + t)
    st = _stream()
    n = m["n"]
    g = torch.Generator(device="cuda").manual_seed(10)
    dq, dk, dv = (torch.randn(t, h, d, generator=g, device="cuda").to(torch.bfloat16) for h in (h_q, h_kv, h_kv))
    dqkvg = torch.empty(t, n, dtype=torch.bfloat16, device="cuda")
    bands = m["bands"]
    outs = tuple(
        torch.as_strided(dqkvg, (t, h, d), (n, d, 1), storage_offset=off) for off, h in ((0, h_q), (2 * h_q * d, h_kv), (2 * h_q * d + h_kv * d, h_kv))
    )
    rstd_q = torch.rsqrt(bands["q"].float().pow(2).mean(-1) + _EPS).contiguous()
    rstd_k = torch.rsqrt(bands["k"].float().pow(2).mean(-1) + _EPS).contiguous()
    rn = NB.compile_qk_norm_rope_bwd(
        dtype=torch.bfloat16, h_q=h_q, h_kv=h_kv, d=d, rope_dim=rope_dim, eps=_EPS, apply_norm=True, want_dw=want_dw, has_seq_lens=False, want_amax=False
    )
    nq, nk, _nv = NB.n_ctas_for(rn, t)
    pq = torch.full((nq, d), float("nan"), device="cuda") if want_dw else None
    pk = torch.full((nk, d), float("nan"), device="cuda") if want_dw else None
    NB.run_qk_norm_rope_bwd(rn, dq, dk, dv, bands["q"], bands["k"], rstd_q, rstd_k, m["w_q"], m["w_k"], m["cos"], m["sin"], *outs, pq, pk, stream=st)
    gate_band = torch.as_strided(dqkvg, (t, h_q, d), (n, d, 1), storage_offset=h_q * d)
    gate_band.copy_(torch.randn(t, h_q, d, generator=g, device="cuda").to(torch.bfloat16))
    torch.cuda.synchronize()
    # fused
    re = F.compile_mxfp8_bwd_epilogue(dtype=torch.bfloat16, n_cols=n, d=d, want_dw=want_dw, want_row=want_row, want_col=want_col)
    assert (re.want_dw, re.want_row, re.want_col, re.threads) == (want_dw, want_row, want_col, F.EPILOGUE_THREADS)
    n_red, n_cast = F.epilogue_grid(re, t)
    assert n_red == (d if want_dw else 0) and n_cast == (n // d) * MX.n_sf_tiles(t)
    h = n // d
    dst_f = torch.full((t, h, d), 0xFF, dtype=U8, device="cuda").view(E4) if want_row else None
    sf_f = torch.full((sf_blob_bytes(t, n),), 0xFF, dtype=U8, device="cuda") if want_row else None
    dstT_f = torch.full((n, t), 0xFF, dtype=U8, device="cuda").view(E4) if want_col else None
    sfT_f = torch.full((sf_blob_bytes(n, t),), 0xFF, dtype=U8, device="cuda") if want_col else None
    dw_q_f = torch.full((d,), float("nan"), device="cuda") if want_dw else None
    dw_k_f = torch.full((d,), float("nan"), device="cuda") if want_dw else None
    F.run_mxfp8_bwd_epilogue(
        re, plane_q=pq, plane_k=pk, dw_q=dw_q_f, dw_k=dw_k_f, src=dqkvg.view(t, h, d), dst=dst_f, sf=sf_f, dst_t=dstT_f, sf_t=sfT_f, stream=st
    )
    torch.cuda.synchronize()
    # standalone
    if want_row:
        rr = MX.compile_quantize_mxfp8(dtype_in=torch.bfloat16, h=h, d=d, axis="row", sf_layout="gemm")
        dst_s = torch.full((t, h, d), 0xFF, dtype=U8, device="cuda").view(E4)
        sf_s = torch.full((sf_blob_bytes(t, n),), 0xFF, dtype=U8, device="cuda")
        MX.run_quantize_mxfp8(rr, dqkvg.view(t, h, d), dst_s, sf_s, batch=1, seq_len=t, stream=st)
    if want_col:
        rc = MX.compile_quantize_mxfp8(dtype_in=torch.bfloat16, h=h, d=d, axis="col", sf_layout="gemm", transposed=True)
        dstT_s = torch.full((n, t), 0xFF, dtype=U8, device="cuda").view(E4)
        sfT_s = torch.full((sf_blob_bytes(n, t),), 0xFF, dtype=U8, device="cuda")
        MX.run_quantize_mxfp8(rc, dqkvg.view(t, h, d), dstT_s, sfT_s, batch=1, seq_len=t, stream=st)
    if want_dw:
        dw_q_s, dw_k_s = torch.empty(d, device="cuda"), torch.empty(d, device="cuda")
        NB.run_dw_norm_reduce(rn, pq, pk, dw_q_s, dw_k_s, stream=st, t=t)
    torch.cuda.synchronize()
    if want_row:
        assert _eq(dst_f, dst_s) and torch.equal(sf_f, sf_s), "the fused rowwise canonical cast differs from the standalone launch"
        assert not (sf_f == 0xFF).any()
    if want_col:
        assert _eq(dstT_f, dstT_s) and torch.equal(sfT_f, sfT_s), "the fused transposed canonical cast differs from the standalone launch"
        assert not (sfT_f == 0xFF).any()
    if want_dw:
        assert torch.equal(dw_q_f, dw_q_s) and torch.equal(dw_k_f, dw_k_s), "the two-columns-per-block reduce is not the (8, 128) reduce's fixed-order sum"


@requires_mx_tma
def test_epilogue_reduce_grid_is_derived_from_the_block_size():
    """The reduce arm sums ``cols = threads_per_cta // REDUCE_LANES`` columns per block, so its grid is ``ceil(2 d / cols)`` blocks over
    the ``2 d`` columns (d Q, then d K): ``2 d`` at 128 threads, ``d`` at the default 256, ``d / 2`` at 512 -- ``epilogue_grid`` DERIVES
    it from the recipe's block size.  Pinned on the launch that exposed a grid of ``d`` blocks at EVERY block size: at 128 threads (one
    column per block) it covered the Q half only and ``dW_k_norm`` kept its initial bytes.  Q partials all ones over 3 rows and K
    partials all 2 over 5 rows make the sums 3 and 10 exactly, on outputs initialised to -777; the default 256-thread artifact on the
    same planes is bitwise the 128-thread one (the per-column chain never depends on ``cols``)."""
    d, t = 128, 128
    n_cols = 2 * d
    pq = torch.ones(3, d, device="cuda")
    pk = torch.full((5, d), 2.0, device="cuda")
    src = torch.zeros(t, n_cols // d, d, dtype=torch.bfloat16, device="cuda")  # the cast arm is folded out: src pins T only
    st = _stream()
    got = {}
    for threads in (128, F.EPILOGUE_THREADS):
        re = F.compile_mxfp8_bwd_epilogue(dtype=torch.bfloat16, n_cols=n_cols, d=d, want_dw=True, want_row=False, want_col=False, threads_per_cta=threads)
        dw_q = torch.full((d,), -777.0, device="cuda")
        dw_k = torch.full((d,), -777.0, device="cuda")
        F.run_mxfp8_bwd_epilogue(re, plane_q=pq, plane_k=pk, dw_q=dw_q, dw_k=dw_k, src=src, dst=None, sf=None, dst_t=None, sf_t=None, stream=st)
        torch.cuda.synchronize()
        assert torch.equal(dw_q, torch.full((d,), 3.0, device="cuda")), f"{threads} threads: dW_q_norm = {dw_q[:4].tolist()} ..., expected 3.0"
        assert torch.equal(
            dw_k, torch.full((d,), 10.0, device="cuda")
        ), f"{threads} threads: dW_k_norm = {dw_k[:4].tolist()} ..., expected 10.0 (the K half's reduce blocks were never launched)"
        cols = threads // NB.REDUCE_LANES
        assert F.epilogue_grid(re, t) == (-(-2 * d // cols), 0), f"{threads} threads: {cols} column(s) per block need {-(-2 * d // cols)} reduce blocks"
        got[threads] = (dw_q, dw_k)
    assert all(torch.equal(a, b) for a, b in zip(got[128], got[F.EPILOGUE_THREADS]))


@requires_cuda
def test_fused_launch_contracts_are_typed():
    """Compile-time refusals (the dY view's head count, the init job's constant range, the MX geometry, an epilogue with nothing to
    launch, a block size the reduce lanes do not divide) and the execute-time both-ways checks (the halves against ``want_row`` /
    ``want_col``, planes against ``want_dw``, ``T == batch * seq_len``, the constants' count, an SF blob of the wrong size) -- every one a
    typed error before any launch, on any CUDA device (the kernels are not traced)."""
    if _cc() is not None and _cc() >= (10, 0):
        with pytest.raises(ValueError, match="d_model=500 must be a multiple of d_head=256"):
            F.compile_mxfp8_bwd_prologue(dtype=torch.bfloat16, h_q=8, h_kv=2, d_model=500, d=256, rope_dim=64, eps=_EPS, apply_norm=True, n_slots=15)
        with pytest.raises(ValueError, match="do not fit the 15-slot block"):
            F.compile_mxfp8_bwd_prologue(
                dtype=torch.bfloat16, h_q=8, h_kv=2, d_model=512, d=256, rope_dim=64, eps=_EPS, apply_norm=True, n_slots=15, const_slot0=14, n_consts=2
            )
        with pytest.raises(ValueError, match="warp-per-row"):
            F.compile_mxfp8_bwd_prologue(dtype=torch.bfloat16, h_q=8, h_kv=2, d_model=512, d=128, rope_dim=64, eps=_EPS, apply_norm=True, n_slots=15)
        with pytest.raises(ValueError, match="nothing to launch"):
            F.compile_mxfp8_bwd_epilogue(dtype=torch.bfloat16, n_cols=5120, d=256, want_dw=False, want_row=False, want_col=False)
        with pytest.raises(ValueError, match="n_cols=500 must be a multiple of d_head=256"):
            F.compile_mxfp8_bwd_epilogue(dtype=torch.bfloat16, n_cols=500, d=256, want_dw=True, want_row=True, want_col=True)
        with pytest.raises(ValueError, match="one thread per d"):
            F.compile_mxfp8_bwd_epilogue(dtype=torch.bfloat16, n_cols=5120, d=256, want_dw=True, want_row=True, want_col=True, threads_per_cta=128)
    # the execute-time checks need no compiled artifact: recipes with compiled=None
    t, h, d = 64, 20, 256
    re = F.Mxfp8BwdEpilogueRecipe(compiled=None, dtype=torch.bfloat16, h=h, d=d, want_dw=True, want_row=True, want_col=False, threads=256)
    src = torch.empty(t, h, d, dtype=torch.bfloat16, device="cuda")
    dst = torch.empty(t, h, d, dtype=E4, device="cuda")
    sf = torch.empty(sf_blob_bytes(t, h * d), dtype=U8, device="cuda")
    dst_t = torch.empty(h * d, t, dtype=E4, device="cuda")
    sf_t = torch.empty(sf_blob_bytes(h * d, t), dtype=U8, device="cuda")
    plane = torch.empty(3, d, device="cuda")
    dw = torch.empty(d, device="cuda")
    with pytest.raises(ValueError, match="WITH the dW_norm reduce"):
        F.run_mxfp8_bwd_epilogue(re, plane_q=None, plane_k=None, dw_q=None, dw_k=None, src=src, dst=dst, sf=sf, dst_t=None, sf_t=None, stream=0)
    with pytest.raises(ValueError, match="WITHOUT the transposed half"):
        F.run_mxfp8_bwd_epilogue(re, plane_q=plane, plane_k=plane, dw_q=dw, dw_k=dw, src=src, dst=dst, sf=sf, dst_t=dst_t, sf_t=sf_t, stream=0)
    with pytest.raises(ValueError, match="WITH the rowwise half"):
        F.run_mxfp8_bwd_epilogue(re, plane_q=plane, plane_k=plane, dw_q=dw, dw_k=dw, src=src, dst=None, sf=None, dst_t=None, sf_t=None, stream=0)
    with pytest.raises(ValueError, match="sf must be the padded F8_128x4 blob"):
        F.run_mxfp8_bwd_epilogue(re, plane_q=plane, plane_k=plane, dw_q=dw, dw_k=dw, src=src, dst=dst, sf=sf[:-16], dst_t=None, sf_t=None, stream=0)
    re_c = re._replace(want_dw=False, want_row=False, want_col=True)
    with pytest.raises(ValueError, match="WITHOUT the dW_norm reduce"):
        F.run_mxfp8_bwd_epilogue(re_c, plane_q=plane, plane_k=plane, dw_q=dw, dw_k=dw, src=src, dst=None, sf=None, dst_t=dst_t, sf_t=sf_t, stream=0)
    with pytest.raises(ValueError, match="WITH the transposed half"):
        F.run_mxfp8_bwd_epilogue(re_c, plane_q=None, plane_k=None, dw_q=None, dw_k=None, src=src, dst=None, sf=None, dst_t=None, sf_t=None, stream=0)
    with pytest.raises(ValueError, match="multiple of 32"):
        F.run_mxfp8_bwd_epilogue(
            re_c, plane_q=None, plane_k=None, dw_q=None, dw_k=None, src=src[:40], dst=None, sf=None, dst_t=dst_t[:, :40].contiguous(), sf_t=sf_t, stream=0
        )
    assert (
        F.epilogue_grid(re, 1000) == (d, h * 8)
        and F.epilogue_grid(re._replace(want_dw=False), 128) == (0, h)
        and F.epilogue_grid(re_c._replace(want_col=False), 128) == (0, 0)
    )
    # the reduce grid follows the block size: cols = threads // REDUCE_LANES columns per block over the 2 d columns
    assert F.epilogue_grid(re._replace(threads=512), 1000) == (d // 2, h * 8)
    assert F.epilogue_grid(re._replace(d=128, h=8, threads=128), 128) == (256, 8) and F.epilogue_grid(re._replace(d=128, h=8), 128) == (128, 8)
    rp = F.Mxfp8BwdPrologueRecipe(
        compiled=None,
        dtype=torch.bfloat16,
        h_q=8,
        h_kv=2,
        h_dy=2,
        d=256,
        rope_dim=64,
        eps=_EPS,
        stages=2,
        apply_norm=True,
        n_slots=15,
        rows_per_cta=16,
        n_ctas_cap=64,
        threads=128,
    )
    b, s = 2, 32
    t = b * s
    slots = torch.zeros(15, device="cuda")
    q = torch.empty(t, 8, 256, dtype=torch.bfloat16, device="cuda")
    k = torch.empty(t, 2, 256, dtype=torch.bfloat16, device="cuda")
    w = torch.ones(256, dtype=torch.bfloat16, device="cuda")
    tab = torch.empty(t, 64, dtype=torch.bfloat16, device="cuda")
    q8, sf_q = _poisoned(b, s, 8, 256)
    q_T8, sf_q_T = _poisoned(b, s, 8, 256)
    k8, sf_k = _poisoned(b, s, 2, 256)
    k_T8, sf_k_T = _poisoned(b, s, 2, 256)
    v8, sf_v = _poisoned(b, s, 2, 256)
    parts = torch.zeros(64, device="cuda")
    kw = dict(
        slots=slots,
        dy=torch.empty(t, 2, 256, dtype=torch.bfloat16, device="cuda"),
        partials=parts,
        q=q,
        k=k,
        w_q=w,
        w_k=w,
        cos=tab,
        sin=tab,
        q8=q8,
        sf_q=sf_q,
        q_T8=q_T8,
        sf_q_T=sf_q_T,
        k8=k8,
        sf_k=sf_k,
        k_T8=k_T8,
        sf_k_T=sf_k_T,
        v=k,
        v8=v8,
        sf_v=sf_v,
        stream=0,
    )
    with pytest.raises(ValueError, match="T must equal batch\\*seq_len"):
        F.run_mxfp8_bwd_prologue(rp, batch=b, seq_len=s + 1, **kw)
    with pytest.raises(ValueError, match="n_consts=0 plan-time constants; got 2 consts"):
        F.run_mxfp8_bwd_prologue(rp, batch=b, seq_len=s, consts=(1.0, 2.0), **kw)
    with pytest.raises(ValueError, match="sf_k_T must hold exactly"):
        F.run_mxfp8_bwd_prologue(rp, batch=b, seq_len=s, **{**kw, "sf_k_T": sf_k_T[:-16]})
    with pytest.raises(ValueError, match="WITH the RMSNorm"):
        F.run_mxfp8_bwd_prologue(rp, batch=b, seq_len=s, **{**kw, "w_q": None, "w_k": None})
    with pytest.raises(ValueError, match="dy must be \\[T=64, H=2, D=256\\]"):
        F.run_mxfp8_bwd_prologue(rp, batch=b, seq_len=s, **{**kw, "dy": q})
    # b = 2, s = 32: 64 x 2 dY rows = 8 row groups; 2 x (ceil128(32)/32 = 4) x (8 + 2) = 80 token tiles -> capped at 64 CTAs; 2 x 2 x 1 v8 units
    assert F.prologue_grid(rp, b, s) == (8, 64, 4) and F.prologue_grid(rp, 1, 100000) == (64, 64, 2 * 782)
    # the rebuild job's own cap (n_rec_cap, appended): 0 = the amax job's cap above; set = SMs x the MX arm's residency, the amax cap untouched
    assert (
        rp.n_rec_cap == 0
        and F.prologue_grid(rp._replace(n_rec_cap=40), b, s) == (8, 40, 4)
        and F.prologue_grid(rp._replace(n_rec_cap=40), 1, 100000) == (64, 40, 2 * 782)
    )


_PROLOGUE_REG_PROBE = textwrap.dedent("""
    import glob, os, subprocess, sys
    norm, dump, cuobjdump = sys.argv[1] == "1", sys.argv[2], sys.argv[3]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump
    import torch
    import cutlass.cute as cute
    import cudnn.gated_attention_block.kernels.mxfp8_bwd_fused as F
    _orig = cute.compile
    def _compile(*a, **k):
        k["options"] = k.get("options", "") + " --gpu-arch sm_107a --keep-cubin"
        return _orig(*a, **k)
    cute.compile = _compile
    F.require_fp8_cvt = lambda who: None  # the trace-compile targets sm_107a whatever this box is
    F.compile_mxfp8_bwd_prologue(dtype=torch.bfloat16, h_q=8, h_kv=2, d_model=512, d=256, rope_dim=64, eps=1e-6, apply_norm=norm, n_slots=29, const_slot0=15, n_consts=14)
    cubins = glob.glob(os.path.join(dump, "*.sm_107a.cubin"))
    if not cubins:
        print("FAIL no .sm_107a.cubin landed in", dump)
        sys.exit(3)
    res = subprocess.run([cuobjdump, "--dump-resource-usage", cubins[0]], capture_output=True, text=True, timeout=120).stdout
    regs = [int(tok.split(":")[1]) for ln in res.splitlines() for tok in ln.split() if tok.startswith("REG:")]
    if not regs:
        print("SKIP cuobjdump printed no REG line:", res[-300:])
        sys.exit(0)
    print("REG", max(regs))
    """)


def _sm107a_known_to_the_dsl() -> bool:
    try:
        from cutlass.base_dsl.enums import Arch

        Arch.from_string("sm_107a")
        return True
    except Exception:  # noqa: BLE001
        return False


def _cuobjdump():
    cands = []
    if os.environ.get("CUDA_PATH"):
        cands.append(os.path.join(os.environ["CUDA_PATH"], "bin", "cuobjdump"))
    on_path = shutil.which("cuobjdump")
    if on_path:
        cands.append(on_path)
    return next((c for c in cands if os.path.isfile(c) and os.access(c, os.X_OK)), None)


@requires_cuda
@_QK_NORM
def test_prologue_persistent_cap_is_the_sm107a_residency(qk_norm, tmp_path):
    """The rebuild job's persistent cap is the MX arm's RESIDENCY per SM, ``min(register cap, SMEM cap)`` = 5 (norm) / 6 (RoPE-only):
    the arithmetic (``register_cap_ctas_per_sm``: 65536 / (128 x ceil8(REG)); REG 90 and 96 -> 5, 80 -> 6, 72 -> 7, the SMEM-bound 6)
    runs anywhere; then the prologue is trace-compiled for sm_107a here (no device match needed) and REG read off the cubin with
    ``cuobjdump --dump-resource-usage`` -- the register cap it yields, clipped by the SMEM cap, must still be the table's value, so a
    kernel edit that moves REG across an allocation boundary fails HERE and not in a perf table.  SKIPS where the DSL predates sm_107a
    or no cuobjdump is found."""
    assert (
        TMA.register_cap_ctas_per_sm(90) == 5
        and TMA.register_cap_ctas_per_sm(96) == 5
        and TMA.register_cap_ctas_per_sm(80) == 6
        and TMA.register_cap_ctas_per_sm(72) == 7
    )
    assert TMA.register_cap_ctas_per_sm(97) == 4 and TMA.register_cap_ctas_per_sm(64) == 8
    assert TMA.MX_REBUILD_CTAS_PER_SM == {True: 5, False: 6} and TMA.mx_rebuild_ctas_per_sm(qk_norm) == (5 if qk_norm else 6)
    if not _sm107a_known_to_the_dsl():
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0.dev0, --pre)")
    cuobjdump = _cuobjdump()
    if cuobjdump is None:
        pytest.skip("no cuobjdump executable (CUDA_PATH unset and none on PATH)")
    dump = tmp_path / f"mxfp8_bwd_prologue_{int(qk_norm)}"
    dump.mkdir()
    proc = subprocess.run(
        [sys.executable, "-c", _PROLOGUE_REG_PROBE, "1" if qk_norm else "0", str(dump), cuobjdump], capture_output=True, text=True, timeout=900
    )
    assert proc.returncode == 0, f"trace-compile failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    if any(ln.startswith("SKIP") for ln in proc.stdout.splitlines()):
        pytest.skip([ln for ln in proc.stdout.splitlines() if ln.startswith("SKIP")][0])
    reg = int(next(ln.split()[1] for ln in proc.stdout.splitlines() if ln.startswith("REG ")))
    cap = min(TMA.register_cap_ctas_per_sm(reg), TMA.MX_REBUILD_SMEM_CTAS_PER_SM)
    print(
        f"\n[prologue norm={qk_norm}] sm_107a REG {reg} -> register cap {TMA.register_cap_ctas_per_sm(reg)} CTAs/SM, SMEM cap {TMA.MX_REBUILD_SMEM_CTAS_PER_SM} -> {cap}"
    )
    assert cap == TMA.mx_rebuild_ctas_per_sm(qk_norm), (
        f"the sm_107a cubin's REG {reg} gives a residency of {cap} CTAs/SM but MX_REBUILD_CTAS_PER_SM says {TMA.mx_rebuild_ctas_per_sm(qk_norm)}: "
        "re-derive the table (qk_norm_rope_tma.py) -- a stale cap runs a grid-stride tail or idles SMs"
    )


@requires_cuda
def test_fused_launches_move_their_jobs_bytes_minus_the_round_trip():
    """The shared-geometry pin, host-only: a fused launch moves EXACTLY the bytes of the standalone launches it replaces minus the
    round trip it deletes -- the prologue the bf16 ``recompute`` / ``recompute_k`` write + the four quantizes' two reads of every Q / K
    element (``T x (HD + HKD) x 6``), the dual-axis dO launch one bf16 read of dO (``T x HD x 2``), the epilogue one bf16 read of the
    dqkvg slab (``T x N x 2``, both halves traced; nothing with one half) -- and each job's bytes are the standalone kernel's own model
    (``quantize.amax_moved_bytes``, ``quantize_mxfp8.moved_bytes`` / ``moved_bytes_dual``).  Pinned at the 397B geometry (S = 8K, B = 1) to
    the MiB the module docstrings quote (354.4 / 762.4, 260.0 / 388.0, 555.8 / 827.8) and to the cos / sin L1-miss upper bound
    (``+ T x rope_dim x 4 x (h_q + h_kv - 1)``).  The GB/s a launch reaches on these bytes is measured by the perf tooling on an exclusive
    GPU of a perf node and never asserted here: a rate on a shared development node is not a pin.  No device, no kernel."""
    h_q, h_kv, d, rope_dim, d_model, n_cols, n_slots, sms = 32, 2, 256, 64, 4096, 17408, 29, 212
    b, s = 1, 8192
    t = b * s
    hd, hkd = h_q * d, h_kv * d
    rp = F.Mxfp8BwdPrologueRecipe(
        compiled=None,
        dtype=torch.bfloat16,
        h_q=h_q,
        h_kv=h_kv,
        h_dy=d_model // d,
        d=d,
        rope_dim=rope_dim,
        eps=_EPS,
        stages=2,
        apply_norm=True,
        n_slots=n_slots,
        rows_per_cta=16,
        n_ctas_cap=sms * 8,
        threads=128,
    )
    fused, standalone = F.prologue_moved_bytes(rp, b, s), F.prologue_standalone_set_bytes(rp, b, s)
    # every job at its standalone kernel's own byte model: init | dY amax | the rebuild's reads + its four quantized outputs | v8
    rebuild_reads = t * (hd + hkd) * 2 + t * rope_dim * 4
    four_outputs = 2 * (t * (hd + hkd) + t * (hd + hkd) // MX.SF_BLOCK)
    assert fused == n_slots * 4 + Q.amax_moved_bytes(t, d_model // d, d) + rebuild_reads + four_outputs + MX.moved_bytes(t, h_kv, d)
    assert standalone - fused == t * (hd + hkd) * 6, "the fusion deletes exactly the bf16 Q / K round trip (one write, two reads per element)"
    assert F.prologue_moved_bytes(rp, b, s, rope_l1_miss=True) - fused == t * rope_dim * 4 * (h_q + h_kv - 1)
    assert (round(fused / 2**20, 1), round(standalone / 2**20, 1)) == (354.4, 762.4)
    # the dual-axis dO launch against the two standalone launches it replaces (the kernel module's own pair of models)
    dual, pair = MX.moved_bytes_dual(t, h_q, d), 2 * MX.moved_bytes(t, h_q, d)
    assert pair - dual == t * hd * 2 and (round(dual / 2**20, 1), round(pair / 2**20, 1)) == (260.0, 388.0)
    # the epilogue: the reduce's two fp32 partial planes (one row per norm-backward CTA of each class, capped at SMs x 8) + the cast
    re = F.Mxfp8BwdEpilogueRecipe(compiled=None, dtype=torch.bfloat16, h=n_cols // d, d=d, want_dw=True, want_row=True, want_col=True, threads=256)
    n_plane_rows = 2 * sms * Q.AMAX_CTAS_PER_SM
    fe, se = F.epilogue_moved_bytes(re, t, n_plane_rows), F.epilogue_standalone_set_bytes(re, t, n_plane_rows)
    assert fe == n_plane_rows * d * 4 + MX.moved_bytes_dual(t, n_cols // d, d) and se - fe == t * n_cols * 2
    assert (round(fe / 2**20, 1), round(se / 2**20, 1)) == (555.8, 827.8)
    # a folded-out half moves nothing of its own: one half = one standalone quantize's bytes, no round trip to delete; no reduce = no planes
    for want_row, want_col in ((True, False), (False, True)):
        r1 = re._replace(want_dw=False, want_row=want_row, want_col=want_col)
        assert F.epilogue_moved_bytes(r1, t, n_plane_rows) == MX.moved_bytes(t, n_cols // d, d) == F.epilogue_standalone_set_bytes(r1, t, n_plane_rows)
    assert F.epilogue_moved_bytes(re._replace(want_row=False, want_col=False), t, n_plane_rows) == n_plane_rows * d * 4


def test_epilogue_execute_path_allocates_nothing_for_a_folded_out_half():
    """An epilogue with one half folded out (``want_row=False`` or ``want_col=False``) binds ``None`` for that half on the execute
    path and allocates NO stand-in tensor for it: ``torch.cuda.memory_allocated`` is unchanged across the call (a stub stands in for
    the compiled artifact and records the bound operands -- the folded-out pair arrives as ``None``, the traced pair as the caller's
    tensors), on any CUDA device.  A full-size stand-in was ~143 MiB per folded half at the 397B geometry, S = 8K, per execute."""
    t, h, d = 256, 20, 256
    src = torch.empty(t, h, d, dtype=torch.bfloat16, device="cuda")
    dst = torch.empty(t, h, d, dtype=E4, device="cuda")
    sf = torch.empty(sf_blob_bytes(t, h * d), dtype=U8, device="cuda")
    dst_t = torch.empty(h * d, t, dtype=E4, device="cuda")
    sf_t = torch.empty(sf_blob_bytes(h * d, t), dtype=U8, device="cuda")
    calls = []

    def stub(*args):
        calls.append(args)

    for want_row, want_col in ((True, False), (False, True)):
        re = F.Mxfp8BwdEpilogueRecipe(compiled=stub, dtype=torch.bfloat16, h=h, d=d, want_dw=False, want_row=want_row, want_col=want_col, threads=256)
        before = torch.cuda.memory_allocated()
        F.run_mxfp8_bwd_epilogue(
            re,
            plane_q=None,
            plane_k=None,
            dw_q=None,
            dw_k=None,
            src=src,
            dst=dst if want_row else None,
            sf=sf if want_row else None,
            dst_t=dst_t if want_col else None,
            sf_t=sf_t if want_col else None,
            stream=0,
        )
        assert torch.cuda.memory_allocated() == before, f"the folded-out half allocated a stand-in tensor on the execute path (want_row={want_row})"
    args_row, args_col = calls
    # the compiled signature: plane_q, plane_k, dw_q, dw_k, src, dst, sf, dst_t, sf_t, ... -- the folded-out pair is None, the traced one bound
    assert args_row[5] is dst and args_row[6] is not None and args_row[7] is None and args_row[8] is None
    assert args_col[5] is None and args_col[6] is None and args_col[7] is dst_t and args_col[8] is not None
