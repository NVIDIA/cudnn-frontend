# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The quantized block backward's two FUSED small-kernel launches (``kernels/fp8_bwd_fused.py``): the PROLOGUE (scalar init -- the
zeroing, ``descale_dp`` and the plan-time constants out of the launch's kernel arguments -- + dY amax partials + the Q / K rebuild
with its e4m3 epilogue + v8, the rebuild's and v8's static scales kernel arguments too) and the EPILOGUE (dW_norm reduce + dqkvg
quantize), each one ``@cute.kernel`` dispatching its jobs by block range.

The contract is BITWISE: every job of a fused launch is the ``@cute.jit`` body its standalone kernel runs, so every output of the
fused launch must equal the standalone launches' byte for byte -- the e4m3 bytes, the partials, the published slots, the fp32
``dW_norm`` (the reduce at one column per block is the SAME fixed-order chain as the standalone ``(8, 128)`` reduce).  Needs the
TMA kernel (sm_90+) and the fp8 cvt (sm_89+); the typed contracts run on any CUDA device.
"""

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

from cudnn.gated_attention_block.kernels import fp8_bwd_fused as F  # noqa: E402
from cudnn.gated_attention_block.kernels import qk_norm_rope_bwd as NB  # noqa: E402
from cudnn.gated_attention_block.kernels import qk_norm_rope_tma as TMA  # noqa: E402
from cudnn.gated_attention_block.kernels import quantize as Q  # noqa: E402

E4 = torch.float8_e4m3fn
_EPS = 1e-6


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
requires_fp8_tma = pytest.mark.skipif(_cc() is None or _cc() < (9, 0), reason="the fused launches need the TMA kernel (sm_90+) and the fp8 cvt (sm_89+)")
_QK_NORM = pytest.mark.parametrize("qk_norm", [True, False], ids=["norm", "rope_only"])


def _stream():
    return torch.cuda.current_stream().cuda_stream


def _quant_ref(x, scale):
    return torch.clamp(x.float() * scale, -Q.FP8_E4M3_MAX, Q.FP8_E4M3_MAX).to(E4)


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


@requires_fp8_tma
@_QK_NORM
@pytest.mark.parametrize("t", [1000, 4003, "cap+1"], ids=["1000", "4003", "capped-amax"])
def test_prologue_is_bitwise_the_standalone_launches(qk_norm, t):
    """One prologue launch == {init_scalars (the zeroing, ``descale_dp`` and the plan-time constants out of kernel arguments), the
    amax partials pass, the TMA rebuild's e4m3 epilogue, the v8 quantize} run standalone: the slots (zeroed, ``descale_dp``, the
    fourteen constants in their tail), every partial (the words past ``n_amax`` untouched), ``q8`` / ``k8`` (also == the quantize
    pass over the bf16 rebuild; the fused launch takes the static scales as kernel arguments, the standalone e4m3 epilogue reads
    them from slots) and ``v8`` -- byte for byte; ``prologue_grid`` is the three job widths.  The ``capped-amax`` cell sizes its
    token count from the compiled recipe so that the dY amax job has one row group more than its persistent cap on ANY part (its
    CTAs stride)."""
    h_q, h_kv, d, rope_dim, d_model = 8, 2, 256, 64, 512
    st = _stream()
    tile_rows = 8
    n_slots, c0, n_c = 29, 15, 14  # the block's slot layout: fifteen pre-existing slots, then the fourteen plan-time constants
    r = F.compile_fp8_bwd_prologue(
        dtype=torch.bfloat16,
        h_q=h_q,
        h_kv=h_kv,
        d_model=d_model,
        d=d,
        rope_dim=rope_dim,
        eps=_EPS,
        apply_norm=qk_norm,
        n_slots=n_slots,
        tile_rows=tile_rows,
        const_slot0=c0,
        n_consts=n_c,
    )
    assert (r.n_slots, r.const_slot0, r.n_consts) == (n_slots, c0, n_c)
    capped = t == "cap+1"
    if capped:
        # ceil((cap + 1) * rows_per_cta / h_dy) tokens = EXACTLY cap + 1 dY row groups, so the amax job strides on ANY part (13064
        # tokens on a 204-SM part, cap 1632); a token LITERAL exceeds the SMs x 8 cap only up to some SM count (one that gives 1633
        # groups does so for <= 204 SMs and sits below a 208-SM part's 1664), so the cell derives it from the recipe
        t = ((r.n_ctas_cap + 1) * r.rows_per_cta + r.h_dy - 1) // r.h_dy
    m = _make(t, h_q, h_kv, d, rope_dim, d_model, seed=t)
    w_q, w_k = (m["w_q"], m["w_k"]) if qk_norm else (None, None)
    n_amax, n_rec, n_v8 = F.prologue_grid(r, t)
    assert n_amax == Q.n_partials_for(Q.compile_amax_partials(dtype_in=torch.bfloat16, h=d_model // d, d=d), t) and n_rec >= 1 and n_v8 == (t * h_kv + 15) // 16
    # the recipe's own formula: min(the dY row groups, the SMs x 8 cap), at least 1 -- never "the cap" (501 groups at 4003 tokens
    # sit below a 204-SM part's 1632)
    dy_groups = (t * (d_model // d) + r.rows_per_cta - 1) // r.rows_per_cta
    assert n_amax == max(1, min(dy_groups, r.n_ctas_cap))
    if capped:
        assert dy_groups == r.n_ctas_cap + 1 and n_amax == r.n_ctas_cap, "the capped cell must stride the amax job (one row group more than CTAs)"
    slots = torch.full((n_slots,), float("nan"), device="cuda")
    scale_dp = torch.tensor([3.0], device="cuda")
    consts = tuple(0.5 + 0.25 * i for i in range(n_c))  # exact in fp32: the bitwise check asks no rounding question
    partials = torch.full((r.n_ctas_cap + 3,), float("nan"), device="cuda")
    q8 = torch.full((t, h_q, d), 0x7F, dtype=torch.uint8, device="cuda").view(E4)
    k8 = torch.full((t, h_kv, d), 0x7F, dtype=torch.uint8, device="cuda").view(E4)
    v8 = torch.full((t, h_kv, d), 0x7F, dtype=torch.uint8, device="cuda").view(E4)
    sq, sk, sv = 7.0, 0.25, 2.0  # the static scales: kernel ARGUMENTS of the fused launch (Python floats)
    sq_t, sk_t = (torch.tensor([v], device="cuda") for v in (sq, sk))  # ... and slots for the standalone e4m3 epilogue
    n = F.run_fp8_bwd_prologue(
        r,
        slots=slots,
        scale_dp=scale_dp,
        descale_dp_out=slots[14:15],
        dy=m["dy"].view(t, d_model // d, d),
        partials=partials,
        q=m["bands"]["q"],
        k=m["bands"]["k"],
        w_q=w_q,
        w_k=w_k,
        cos=m["cos"],
        sin=m["sin"],
        q8=q8,
        k8=k8,
        scale_q=sq,
        scale_k=sk,
        v=m["bands"]["v"],
        v8=v8,
        scale_v=sv,
        stream=st,
        consts=consts,
    )
    torch.cuda.synchronize()
    assert n == n_amax
    # init
    slots_ref = torch.full((n_slots,), float("nan"), device="cuda")
    Q.run_init_scalars(Q.compile_init_scalars(n_slots, c0, n_c), slots_ref, scale_dp, slots_ref[14:15], consts, stream=st)
    # partials
    part_ref = torch.full_like(partials, float("nan"))
    Q.run_amax_partials(Q.compile_amax_partials(dtype_in=torch.bfloat16, h=d_model // d, d=d), m["dy"].view(t, d_model // d, d), part_ref, stream=st)
    # the rebuild: the standalone e4m3 epilogue AND the quantize pass over the bf16 rebuild
    kw = dict(dtype=torch.bfloat16, h_q=h_q, h_kv=h_kv, d=d, rope_dim=rope_dim, eps=_EPS, want_rstd=False, tile_rows=tile_rows, apply_norm=qk_norm)
    q8_ref, k8_ref = torch.empty_like(q8), torch.empty_like(k8)
    TMA.run_qk_norm_rope_tma(
        TMA.compile_qk_norm_rope_tma(**kw, fp8_out=True),
        m["bands"]["q"],
        m["bands"]["k"],
        None,
        None,
        w_q,
        w_k,
        m["cos"],
        m["sin"],
        stream=st,
        q8=q8_ref,
        k8=k8_ref,
        scale_q=sq_t,
        scale_k=sk_t,
    )
    q16, k16 = torch.empty(t, h_q, d, dtype=torch.bfloat16, device="cuda"), torch.empty(t, h_kv, d, dtype=torch.bfloat16, device="cuda")
    TMA.run_qk_norm_rope_tma(TMA.compile_qk_norm_rope_tma(**kw), m["bands"]["q"], m["bands"]["k"], q16, k16, w_q, w_k, m["cos"], m["sin"], stream=st)
    torch.cuda.synchronize()
    assert (
        torch.equal(slots, slots_ref)
        and torch.equal(slots[:14], torch.zeros(14, device="cuda"))
        and slots[14].item() == (torch.tensor(1.0) / torch.tensor(3.0)).item()
        and torch.equal(slots[c0 : c0 + n_c], torch.tensor(consts, dtype=torch.float32, device="cuda"))
    )
    assert torch.equal(partials[:n], part_ref[:n]) and torch.isnan(partials[n:]).all() and torch.equal(partials[:n].max(), m["dy"].float().abs().amax())
    assert torch.equal(q8.view(torch.uint8), q8_ref.view(torch.uint8)) and torch.equal(k8.view(torch.uint8), k8_ref.view(torch.uint8))
    assert torch.equal(q8.view(torch.uint8), _quant_ref(q16, sq).view(torch.uint8)) and torch.equal(k8.view(torch.uint8), _quant_ref(k16, sk).view(torch.uint8))
    assert torch.equal(v8.view(torch.uint8), _quant_ref(m["bands"]["v"], sv).view(torch.uint8))


@requires_fp8_tma
@pytest.mark.parametrize("want_dw", [True, False], ids=["reduce+cast", "cast-only"])
@pytest.mark.parametrize("scale_src", ["amax", "given"])
def test_epilogue_is_bitwise_the_standalone_launches(want_dw, scale_src):
    """One epilogue launch == {the standalone ``(8, 128)`` dW_norm reduce, the quantize launch} run standalone: ``dW_q_norm`` /
    ``dW_k_norm`` (the per-column fixed-order chain does not depend on how many columns share a block), the e4m3 bytes, the
    published amax / scale / descale / alphas -- byte for byte; the amax reduced in every cast block from TWO partials arrays
    (the norm backward's band partials, written by the real kernel here, and a gate-partials array standing in for the gate
    backward's) equals the slot pass's fold over the whole slab, under "current" (the scale derived from it) and "delayed"
    (the caller's scale, the amax still published); the cast blocks stride persistently (``epilogue_grid``)."""
    t, h_q, h_kv, d, rope_dim = 1000, 8, 2, 256, 64
    m = _make(t, h_q, h_kv, d, rope_dim, 512, seed=9)
    st = _stream()
    n = m["n"]
    # the dqkvg slab and (under want_dw) partial planes written by the norm backward over random dq / dk / dv
    g = torch.Generator(device="cuda").manual_seed(10)
    dq, dk, dv = (torch.randn(t, h, d, generator=g, device="cuda").to(torch.bfloat16) for h in (h_q, h_kv, h_kv))
    dqkvg = torch.empty(t, n, dtype=torch.bfloat16, device="cuda")
    b = m["bands"]
    outs = tuple(
        torch.as_strided(dqkvg, (t, h, d), (n, d, 1), storage_offset=off) for off, h in ((0, h_q), (2 * h_q * d, h_kv), (2 * h_q * d + h_kv * d, h_kv))
    )
    rstd_q = torch.rsqrt(b["q"].float().pow(2).mean(-1) + _EPS).contiguous()
    rstd_k = torch.rsqrt(b["k"].float().pow(2).mean(-1) + _EPS).contiguous()
    rn = NB.compile_qk_norm_rope_bwd(
        dtype=torch.bfloat16, h_q=h_q, h_kv=h_kv, d=d, rope_dim=rope_dim, eps=_EPS, apply_norm=True, want_dw=want_dw, has_seq_lens=False, want_amax=True
    )
    nq, nk, nv = NB.n_ctas_for(rn, t)
    pq = torch.full((nq, d), float("nan"), device="cuda") if want_dw else None
    pk = torch.full((nk, d), float("nan"), device="cuda") if want_dw else None
    n_bands = NB.n_amax_partials_for(rn, t)
    assert n_bands == nq + nk + nv
    bands = torch.full((n_bands,), float("nan"), device="cuda")
    n_w = NB.run_qk_norm_rope_bwd(
        rn, dq, dk, dv, b["q"], b["k"], rstd_q, rstd_k, m["w_q"], m["w_k"], m["cos"], m["sin"], *outs, pq, pk, stream=st, amax_out=bands
    )
    assert n_w == n_bands
    # the GATE band: random bf16 (the gate backward's output); its per-CTA partials stand in as the max over 7 random windows
    gate_band = torch.as_strided(dqkvg, (t, h_q, d), (n, d, 1), storage_offset=h_q * d)
    gate_band.copy_(torch.randn(t, h_q, d, generator=g, device="cuda").to(torch.bfloat16))
    torch.cuda.synchronize()
    gate_parts = torch.stack([w.float().abs().amax() for w in torch.tensor_split(gate_band.reshape(-1), 7)]).contiguous()
    assert torch.equal(torch.maximum(gate_parts.max(), bands.max()), dqkvg.float().abs().amax())
    amax = dqkvg.float().abs().amax().reshape(1)
    consts = (torch.tensor([0.125], device="cuda"), torch.tensor([2.0 / 3.0], device="cuda"))
    given = torch.tensor([Q.grad_scale_from_amax(amax.item())], device="cuda") if scale_src == "given" else None
    # fused
    re = F.compile_fp8_bwd_epilogue(dtype=torch.bfloat16, n_cols=n, d=d, want_dw=want_dw, scale_src=scale_src, n_alpha=2)
    assert re.want_dw is want_dw and re.scale_src == scale_src and re.n_ctas_cap == 8 * torch.cuda.get_device_properties(0).multi_processor_count
    n_red, n_cast = F.epilogue_grid(re, t)
    groups = (t * (n // d) + re.rows_per_cta - 1) // re.rows_per_cta
    assert re.cast_persistent is bool(F.EPILOGUE_CAST_PERSISTENT)
    assert n_red == (2 * d if want_dw else 0) and n_cast == (max(1, min(groups, re.n_ctas_cap)) if re.cast_persistent else groups)
    dst_f = torch.full((t, n // d, d), 0x7F, dtype=torch.uint8, device="cuda").view(E4)
    blk_f = torch.full((16,), float("nan"), device="cuda")
    dw_q_f = torch.full((d,), float("nan"), device="cuda") if want_dw else None
    dw_k_f = torch.full((d,), float("nan"), device="cuda") if want_dw else None
    F.run_fp8_bwd_epilogue(
        re,
        plane_q=pq,
        plane_k=pk,
        dw_q=dw_q_f,
        dw_k=dw_k_f,
        src=dqkvg.view(t, n // d, d),
        dst=dst_f,
        scale=given,
        partials_gate=gate_parts,
        n_gate=7,
        partials_bands=bands,
        n_bands=n_bands,
        amax_out=blk_f[2:3],
        scale_out=blk_f[8:9],
        descale=blk_f[9:10],
        alpha_consts=consts,
        alpha_outs=(blk_f[12:13], blk_f[13:14]),
        stream=st,
    )
    # standalone
    rq = Q.compile_quantize(dtype_in=torch.bfloat16, h=n // d, d=d, scale_src=scale_src, n_alpha=2)
    dst_s = torch.empty_like(dst_f)
    blk_s = torch.full((16,), float("nan"), device="cuda")
    Q.run_quantize(
        rq,
        dqkvg.view(t, n // d, d),
        dst_s,
        given,
        stream=st,
        amax=None if scale_src == "given" else amax,
        scale_out=blk_s[8:9],
        descale=blk_s[9:10],
        alpha_consts=consts,
        alpha_outs=(blk_s[12:13], blk_s[13:14]),
    )
    if want_dw:
        dw_q_s, dw_k_s = torch.empty(d, device="cuda"), torch.empty(d, device="cuda")
        NB.run_dw_norm_reduce(rn, pq, pk, dw_q_s, dw_k_s, stream=st, t=t)
    torch.cuda.synchronize()
    assert torch.equal(dst_f.view(torch.uint8), dst_s.view(torch.uint8)), "the fused cast's bytes differ from the standalone quantize's"
    assert torch.equal(blk_f[2:3], amax), "the published amax_dqkvg is not the max over the two partials arrays"
    assert (
        torch.equal(blk_f[8:10], blk_s[8:10])
        and torch.equal(blk_f[12:14], blk_s[12:14])
        and torch.isnan(blk_f[:2]).all()
        and torch.isnan(blk_f[3:8]).all()
        and torch.isnan(blk_f[10:12]).all()
    )
    if want_dw:
        assert torch.equal(dw_q_f, dw_q_s) and torch.equal(dw_k_f, dw_k_s), "the one-column-per-block reduce is not the (8, 128) reduce's fixed-order sum"


@requires_cuda
def test_fused_launch_contracts_are_typed():
    """Compile-time refusals (the dY view's head count, a geometry the TMA tile cannot cover, the init job's constant range, the
    reduce's lane count) and the execute-time both-ways checks (planes / dw outputs against ``want_dw``, scale vs amax against
    ``scale_src``, the static scales and the plan-time constants as finite Python numbers -- kernel arguments, never a slot of the
    block the launch writes --, the constants' count against the artifact, ``descale_dp_out`` outside the constants' slots) -- every
    one a typed error before any launch, on any CUDA device (the kernels are not traced)."""
    if _cc() is not None and _cc() >= (9, 0):
        with pytest.raises(ValueError, match="d_model=500 must be a multiple of d_head=256"):
            F.compile_fp8_bwd_prologue(dtype=torch.bfloat16, h_q=8, h_kv=2, d_model=500, d=256, rope_dim=64, eps=_EPS, apply_norm=True, n_slots=15, tile_rows=8)
        with pytest.raises(ValueError, match="do not fit the 15-slot block"):
            F.compile_fp8_bwd_prologue(
                dtype=torch.bfloat16,
                h_q=8,
                h_kv=2,
                d_model=512,
                d=256,
                rope_dim=64,
                eps=_EPS,
                apply_norm=True,
                n_slots=15,
                tile_rows=8,
                const_slot0=14,
                n_consts=2,
            )
        with pytest.raises(ValueError, match="tile_rows=6 must divide evenly"):
            F.compile_fp8_bwd_prologue(dtype=torch.bfloat16, h_q=6, h_kv=2, d_model=512, d=256, rope_dim=64, eps=_EPS, apply_norm=True, n_slots=15, tile_rows=6)
        with pytest.raises(ValueError, match="n_cols=500 must be a multiple of d_head=256"):
            F.compile_fp8_bwd_epilogue(dtype=torch.bfloat16, n_cols=500, d=256, want_dw=True, scale_src="amax", n_alpha=2)
        with pytest.raises(ValueError, match="REDUCE_LANES"):
            F.compile_fp8_bwd_epilogue(dtype=torch.bfloat16, n_cols=5120, d=256, want_dw=True, scale_src="amax", n_alpha=2, threads_per_cta=256)
    # the execute-time checks need no compiled artifact: a recipe with compiled=None
    re = F.Fp8BwdEpilogueRecipe(
        compiled=None, dtype=torch.bfloat16, h=20, d=256, want_dw=True, scale_src="amax", n_alpha=0, margin_log2=0, rows_per_cta=16, threads=128
    )
    t = 4
    src = torch.empty(t, 20, 256, dtype=torch.bfloat16, device="cuda")
    dst = torch.empty(t, 20, 256, dtype=E4, device="cuda")
    plane = torch.empty(3, 256, device="cuda")
    dw = torch.empty(256, device="cuda")
    s0, s1, s2, s3 = (torch.zeros(1, device="cuda") for _ in range(4))
    parts = torch.zeros(8, device="cuda")
    pkw = dict(partials_gate=parts, n_gate=3, partials_bands=parts, n_bands=8, amax_out=s3)
    kw = dict(src=src, dst=dst, scale_out=s1, descale=s2, alpha_consts=(), alpha_outs=(), stream=0)
    with pytest.raises(ValueError, match="WITH the dW_norm reduce"):
        F.run_fp8_bwd_epilogue(re, plane_q=None, plane_k=None, dw_q=None, dw_k=None, scale=None, **pkw, **kw)
    with pytest.raises(ValueError, match="passing a caller scale would silently ignore it"):
        F.run_fp8_bwd_epilogue(re, plane_q=plane, plane_k=plane, dw_q=dw, dw_k=dw, scale=s0, **pkw, **kw)
    # the partials contract: a too-short array, a CPU one, a published slot inside the partials
    with pytest.raises(ValueError, match="n_partials must be an int in \\[1, 8\\]"):
        F.run_fp8_bwd_epilogue(re, plane_q=plane, plane_k=plane, dw_q=dw, dw_k=dw, scale=None, **{**pkw, "n_bands": 9}, **kw)
    with pytest.raises(ValueError, match="partials_gate must be a contiguous fp32"):
        F.run_fp8_bwd_epilogue(re, plane_q=plane, plane_k=plane, dw_q=dw, dw_k=dw, scale=None, **{**pkw, "partials_gate": torch.zeros(8)}, **kw)
    with pytest.raises(ValueError, match="amax_out lies inside partials_bands"):
        F.run_fp8_bwd_epilogue(re, plane_q=plane, plane_k=plane, dw_q=dw, dw_k=dw, scale=None, **{**pkw, "amax_out": parts[5:6]}, **kw)
    re_g = re._replace(scale_src="given", want_dw=False)
    with pytest.raises(ValueError, match="WITHOUT the dW_norm reduce"):
        F.run_fp8_bwd_epilogue(re_g, plane_q=plane, plane_k=plane, dw_q=dw, dw_k=dw, scale=s0, **pkw, **kw)
    with pytest.raises(ValueError, match="reads the caller's scale .*scale must be bound"):
        F.run_fp8_bwd_epilogue(re_g, plane_q=None, plane_k=None, dw_q=None, dw_k=None, scale=None, **pkw, **kw)
    with pytest.raises(ValueError, match="scale_out aliases scale"):
        F.run_fp8_bwd_epilogue(re_g, plane_q=None, plane_k=None, dw_q=None, dw_k=None, scale=s1, **pkw, **kw)
    with pytest.raises(ValueError, match="amax_out and scale_out are the same slot"):
        F.run_fp8_bwd_epilogue(re_g, plane_q=None, plane_k=None, dw_q=None, dw_k=None, scale=s0, **{**pkw, "amax_out": s1}, **kw)
    # the cast's grid: persistent -> a huge T caps it at the recipe's cap, a tiny one is its row groups; one row group per block
    # otherwise (a recipe built without the knob: today's default); the reduce blocks are 2 d / 0
    rp = re._replace(n_ctas_cap=64, cast_persistent=True)
    assert F.epilogue_grid(rp, 4) == (512, 5) and F.epilogue_grid(rp, 100000) == (512, 64)
    assert F.epilogue_grid(re_g._replace(n_ctas_cap=64, cast_persistent=True), 4) == (0, 5)
    assert re.cast_persistent is False and F.epilogue_grid(re._replace(n_ctas_cap=64), 100000) == (512, 125000)
    rp = F.Fp8BwdPrologueRecipe(
        compiled=None,
        dtype=torch.bfloat16,
        h_q=8,
        h_kv=2,
        h_dy=2,
        d=256,
        rope_dim=64,
        eps=_EPS,
        tile_rows=8,
        stages=2,
        apply_norm=True,
        n_slots=15,
        rows_per_cta=16,
        n_ctas_cap=64,
        threads=128,
    )
    slots = torch.zeros(15, device="cuda")
    q = torch.empty(t, 8, 256, dtype=torch.bfloat16, device="cuda")
    k = torch.empty(t, 2, 256, dtype=torch.bfloat16, device="cuda")
    w = torch.ones(256, dtype=torch.bfloat16, device="cuda")
    tab = torch.empty(t, 64, dtype=torch.bfloat16, device="cuda")
    q8 = torch.empty(t, 8, 256, dtype=E4, device="cuda")
    k8 = torch.empty(t, 2, 256, dtype=E4, device="cuda")
    parts = torch.zeros(64, device="cuda")
    pkw = dict(
        dy=torch.empty(t, 2, 256, dtype=torch.bfloat16, device="cuda"),
        partials=parts,
        q=q,
        k=k,
        w_q=w,
        w_k=w,
        cos=tab,
        sin=tab,
        q8=q8,
        k8=k8,
        v=k,
        v8=k8,
        stream=0,
    )
    one = dict(scale_q=1.0, scale_k=1.0, scale_v=1.0)
    with pytest.raises(ValueError, match="scale_dp lies inside the slot block"):
        F.run_fp8_bwd_prologue(rp, slots=slots, scale_dp=slots[3:4], descale_dp_out=slots[14:15], **one, **pkw)
    # the static scales are kernel ARGUMENTS: a slot (or any tensor) is refused by name -- the init job writes the block in this launch
    with pytest.raises(ValueError, match="scale_q must be a finite Python number"):
        F.run_fp8_bwd_prologue(rp, slots=slots, scale_dp=s0, descale_dp_out=slots[14:15], **{**one, "scale_q": slots[15:16]}, **pkw)
    with pytest.raises(ValueError, match="scale_v must be a finite Python number"):
        F.run_fp8_bwd_prologue(rp, slots=slots, scale_dp=s0, descale_dp_out=slots[14:15], **{**one, "scale_v": float("nan")}, **pkw)
    # the constants: their count is the artifact's ABI, each a finite number, and descale_dp_out may not sit in their slot range
    with pytest.raises(ValueError, match="n_consts=0 plan-time constants; got 2 consts"):
        F.run_fp8_bwd_prologue(rp, slots=slots, scale_dp=s0, descale_dp_out=slots[14:15], consts=(1.0, 2.0), **one, **pkw)
    rc = rp._replace(n_slots=29, const_slot0=15, n_consts=14)
    slots29 = torch.zeros(29, device="cuda")
    with pytest.raises(ValueError, match="consts\\[3\\] must be a finite Python number"):
        F.run_fp8_bwd_prologue(rc, slots=slots29, scale_dp=s0, descale_dp_out=slots29[14:15], consts=(1.0,) * 3 + (s0,) + (1.0,) * 10, **one, **pkw)
    with pytest.raises(ValueError, match="descale_dp_out lies inside slots \\[15, 29\\)"):
        F.run_fp8_bwd_prologue(rc, slots=slots29, scale_dp=s0, descale_dp_out=slots29[20:21], consts=(1.0,) * 14, **one, **pkw)
    with pytest.raises(ValueError, match="WITH the RMSNorm"):
        F.run_fp8_bwd_prologue(rp, slots=slots, scale_dp=s0, descale_dp_out=slots[14:15], **one, **{**pkw, "w_q": None, "w_k": None})
    with pytest.raises(ValueError, match="dy must be \\[T=4, H=2, D=256\\]"):
        F.run_fp8_bwd_prologue(rp, slots=slots, scale_dp=s0, descale_dp_out=slots[14:15], **one, **{**pkw, "dy": q})
    # t = 4: one dY row group, 4 Q tiles + 1 K tile (tile_rows 8 over h_q 8 / h_kv 2), one v8 block; a huge T caps the two persistent jobs
    assert F.prologue_grid(rp, 4) == (1, 5, 1) and F.prologue_grid(rp, 100000) == (64, 64, (100000 * 2 + 15) // 16)
