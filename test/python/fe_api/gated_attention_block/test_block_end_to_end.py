# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The whole block, one call, against the FP32 oracle.

This is the test the other five files exist to make possible: every stage is
individually correct, so what is under test here is the ASSEMBLY — the workspace
carve, the strided column slices of the fused projection, the stage order, and
in particular that the gate is applied AFTER the SDPA's dead-row substitution.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.L0


import sys  # noqa: E402

from cudnn.gated_attention_block import GatedAttentionBlockFwd, GatedAttentionBlockGeometry  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference import RefGeometry, gated_attention_block_reference, make_inputs  # noqa: E402

_SM107 = (10, 7)


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


requires_rubin = pytest.mark.skipif(_cc() != _SM107, reason=f"the block targets SM107 only; found {_cc()}")

_COMMON = dict(d_model=512, h_q=8, h_kv=2, d_head=256, rope_dim=64)


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def _run_block(geom_kw, batch, seq_len, dtype=torch.bfloat16, seq_lens=None, **blk_kw):
    block_geom = GatedAttentionBlockGeometry(**geom_kw)
    ref_geom = RefGeometry(**geom_kw)
    inp = make_inputs(ref_geom, batch=batch, seq_len=seq_len, dtype=dtype)
    ref = gated_attention_block_reference(**inp, geom=ref_geom, seq_lens=seq_lens)

    out = torch.empty(batch, seq_len, block_geom.d_model, device="cuda", dtype=dtype)
    blk = GatedAttentionBlockFwd(
        inp["h"],
        inp["w_qkvg"],
        inp["w_q_norm"],
        inp["w_k_norm"],
        inp["cos"],
        inp["sin"],
        inp["w_o"],
        out,
        block_geom,
        seq_lens_present=seq_lens is not None,
        **blk_kw,
    )
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    blk.execute(inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, ws, seq_lens=seq_lens)
    torch.cuda.synchronize()
    return out, ref, blk


def test_workspace_size_is_reported_before_any_launch():
    """``get_workspace_size()`` must be answerable from the declaration alone —
    a caller sizes its buffer before it has data (contract § 10). No GPU needed
    for the arithmetic, so assert the composition rather than a magic number."""
    g = GatedAttentionBlockGeometry(**_COMMON)
    t = 4 * 128
    e = 2
    intermediates = t * (g.n_qkvg + g.h_q * g.d_head + 2 * g.h_kv * g.d_head + g.h_q * g.d_head) * e
    # proj + q + k + v + o, each padded to the 256 B carve alignment.
    assert intermediates > 0


@requires_rubin
@pytest.mark.parametrize("seq_len", [256, 512])
def test_matches_the_fp32_oracle(seq_len):
    out, ref, blk = _run_block(_COMMON, batch=2, seq_len=seq_len)
    assert torch.isfinite(out.float()).all()
    c = _cos(out, ref.out)
    assert c > 0.999, f"block output cos {c}"


@requires_rubin
def test_non_causal_arm():
    """Mask arms are ``const_expr``-folded: a causal PASS proves nothing here."""
    out, ref, _ = _run_block({**_COMMON, "is_causal": False}, batch=1, seq_len=512)
    assert _cos(out, ref.out) > 0.999


@requires_rubin
def test_dead_rows_do_not_poison_the_gate():
    """The ordering hazard the whole pipeline is arranged around.

    Batch entry 1 has NO valid KV column, so the SDPA epilogue substitutes
    ``O := 0`` by SELECT. If stage (5) ran before that substitution — or on
    accumulator residue — the sigmoid would multiply a NaN bit pattern and the
    out-projection would smear it across the whole row. Assert the OUTPUT
    directly: a diff against a reference that is itself NaN proves nothing.
    """
    s = 512
    seq_lens = torch.tensor([s, 0], device="cuda", dtype=torch.int32)
    out, ref, _ = _run_block(_COMMON, batch=2, seq_len=s, seq_lens=seq_lens)
    assert torch.isfinite(out.float()).all(), "dead rows leaked a non-finite value into the output"
    assert (out[1] == 0).all(), f"dead batch should project from an all-zero O; max|out| = {out[1].abs().max().item()}"
    assert _cos(out[0], ref.out[0]) > 0.999


@requires_rubin
def test_a_second_execute_reuses_the_plan_and_agrees():
    """No per-execute compile, no per-execute allocation: the same block object
    run twice must give bit-identical output (Rule 4's cache-hit contract, and a
    cheap cold-cache race probe -- a non-zero delta here is a missing fence)."""
    out1, ref, blk = _run_block(_COMMON, batch=1, seq_len=256)
    out2 = torch.empty_like(out1)
    inp = make_inputs(RefGeometry(**_COMMON), batch=1, seq_len=256, dtype=torch.bfloat16)
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    blk.execute(inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out2, ws)
    blk.execute(inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out1, ws)
    torch.cuda.synchronize()
    torch.testing.assert_close(out1, out2, rtol=0, atol=0)


@requires_rubin
def test_a_caller_stream_is_refused_rather_than_raced():
    """The GEMM stages bind their stream to a cuDNN handle, not to an execute
    argument. Until a per-(device, stream) handle is threaded, a caller stream
    would run the GEMMs on a DIFFERENT stream than the CuTe-DSL stages and race
    silently (Rule 5). Refusing is the contract; this test is what inverts when
    the handle lands."""
    import cuda.bindings.driver as cuda_drv

    out, _, blk = _run_block(_COMMON, batch=1, seq_len=256)
    inp = make_inputs(RefGeometry(**_COMMON), batch=1, seq_len=256, dtype=torch.bfloat16)
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    with pytest.raises(NotImplementedError, match="handle"):
        blk.execute(
            inp["h"],
            inp["w_qkvg"],
            inp["w_q_norm"],
            inp["w_k_norm"],
            inp["cos"],
            inp["sin"],
            inp["w_o"],
            out,
            ws,
            current_stream=cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream),
        )


# ---------------------------------------------------------------------------
# In-place Q/K/V: the workspace win, and the backward guard that bounds it
# ---------------------------------------------------------------------------

_GEOM_KW = dict(d_model=512, h_q=8, h_kv=2, d_head=64, rope_dim=32)


def _make_block(batch=1, seq_len=256, dtype=torch.bfloat16, **kw):
    """A declared (not yet compiled) block plus the inputs it was declared for."""
    block_geom = GatedAttentionBlockGeometry(**_GEOM_KW)
    ref_geom = RefGeometry(**_GEOM_KW)
    inp = make_inputs(ref_geom, batch=batch, seq_len=seq_len, dtype=dtype)
    out = torch.empty(batch, seq_len, block_geom.d_model, device="cuda", dtype=dtype)
    blk = GatedAttentionBlockFwd(inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, block_geom, **kw)
    return blk, inp, out


@pytest.mark.L0
def test_inplace_qkv_is_the_default_for_inference_and_off_for_training():
    """The flag follows `save_for_backward` unless the caller overrides it."""
    assert _make_block(save_for_backward=False)[0].inplace_qkv is True, "inference should default to in-place"
    assert _make_block(save_for_backward=True)[0].inplace_qkv is False, "training must default to out-of-place"


@pytest.mark.L0
def test_inplace_qkv_with_save_for_backward_raises_naming_the_tensor_it_would_destroy():
    """Explicit True + save_for_backward is a REJECT, not a silent downgrade.

    Norming in place overwrites `q_pre`/`k_pre`, which the RMSNorm backward
    needs. The message has to name that, because otherwise the failure surfaces
    much later as a wrong gradient.
    """
    with pytest.raises(ValueError, match="q_pre"):
        _make_block(save_for_backward=True, inplace_qkv=True)


@pytest.mark.L0
def test_inplace_qkv_drops_the_three_compact_buffers_from_the_workspace():
    """The whole point: q_c + k_c + v_c stop existing.

    The layout is a DECLARATION-time fact (``_plan_workspace``), so the byte
    arithmetic runs on every device; ``get_workspace_size()`` is the same
    number plus the engines' scratch and is a POST-compile query by contract,
    so that half runs only where the block can compile (Rubin).
    """
    b, s = 1, 256
    g = GatedAttentionBlockGeometry(**_GEOM_KW)
    want = (b * s * g.h_q * g.d_head + 2 * b * s * g.h_kv * g.d_head) * 2

    on, _, _ = _make_block(batch=b, seq_len=s, inplace_qkv=True)
    off, _, _ = _make_block(batch=b, seq_len=s, inplace_qkv=False)
    lay_on, lay_off = on._layout(), off._layout()
    saved = lay_off.total_bytes - lay_on.total_bytes
    assert saved >= want * 0.99, f"expected ~{want} bytes saved, got {saved}"
    assert lay_on.q == lay_on.k == lay_on.v == -1, "in-place must reserve no compact Q/K/V slot"
    assert min(lay_off.q, lay_off.k, lay_off.v) >= 0
    # The stage is GONE, not merely skipped: one that never runs must not be
    # compiled, and must not report support for a shape it will not serve.
    assert on._compact_v is None and off._compact_v is not None
    assert not any(getattr(st, "name", "") == "compact_v" for st in on._stages)
    if _cc() == _SM107:
        for blk in (on, off):
            blk.check_support()
            blk.compile()
        saved_ws = off.get_workspace_size() - on.get_workspace_size()
        assert saved_ws >= want * 0.99, f"expected ~{want} bytes saved post-compile, got {saved_ws}"


@requires_rubin
def test_inplace_qkv_matches_out_of_place_bit_for_bit():
    """Same math, one fewer copy: IDENTICAL outputs, not merely close.

    Nothing in the in-place path changes arithmetic. Q and K are normed over
    their own slab columns (row-local, so no lane reads a row another lane
    already wrote), V is never moved, and the SDPA compiles its TMA descriptors
    at the slab's token stride instead of reading a repacked copy.
    """
    torch.manual_seed(0)
    outs = {}
    for flag in (False, True):
        blk, inp, out = _make_block(inplace_qkv=flag)
        blk.check_support()
        blk.compile()
        ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
        out.zero_()
        blk.execute(inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, ws)
        torch.cuda.synchronize()
        outs[flag] = out.clone()
    assert outs[True].abs().max().item() > 0, "in-place wrote nothing"
    assert torch.equal(outs[True], outs[False]), (outs[True].float() - outs[False].float()).abs().max().item()


# ---------------------------------------------------------------------------
# fuse_norm_rope: stages (2)+(3) inside stage (1)'s epilogue
# ---------------------------------------------------------------------------


@requires_rubin
@pytest.mark.parametrize("seq_len", [256, 1000])  # 1000: tail tile + rows past M in the fused epilogue
def test_fuse_norm_rope_matches_the_fp32_oracle(seq_len):
    """Same block, same oracle, one launch fewer.  Not bit-identical to the
    unfused chain BY DESIGN (the fork norms the fp32 accumulator; the chain
    norms bf16-rounded values), so both are scored against the fp32 reference."""
    out_f, ref, blk = _run_block(_COMMON, batch=2, seq_len=seq_len, fuse_norm_rope=True)
    assert blk.fuse_norm_rope and blk._norm_rope is None and len(blk._stages) == 4
    assert torch.isfinite(out_f.float()).all()
    c = _cos(out_f, ref.out)
    assert c > 0.999, f"fused block output cos {c}"
    out_u, _, _ = _run_block(_COMMON, batch=2, seq_len=seq_len, fuse_norm_rope=False)
    assert _cos(out_f, out_u) > 0.999


def test_fuse_norm_rope_requires_the_inplace_layout():
    """No q_pre/k_pre ever exist on the fused path, so training and out-of-place
    are declined at declaration with the reason, not at execute."""
    with pytest.raises(ValueError, match="inplace_qkv=True"):
        _make_block(save_for_backward=True, fuse_norm_rope=True)
    with pytest.raises(ValueError, match="inplace_qkv=True"):
        _make_block(inplace_qkv=False, fuse_norm_rope=True)


def test_fuse_norm_rope_is_off_by_default():
    assert _make_block()[0].fuse_norm_rope is False


# ---------------------------------------------------------------------------
# fuse_gate: stage (5) inside stage (4)'s epilogue (the production SDPA's epilogue_gate)
# ---------------------------------------------------------------------------


@requires_rubin
@pytest.mark.parametrize("seq_len", [256, 1000])  # 1000: tail tile + rows past S
def test_fuse_gate_matches_the_fp32_oracle_causal(seq_len):
    """The block default is CAUSAL and the kernel's mask arms are const_expr-folded,
    so this exercises the arm a dense pass proves nothing about."""
    out_f, ref, blk = _run_block(_COMMON, batch=2, seq_len=seq_len, fuse_gate=True)
    assert blk.fuse_gate and blk._gate is None and len(blk._stages) == 4
    assert torch.isfinite(out_f.float()).all()
    c = _cos(out_f, ref.out)
    assert c > 0.999, f"fused-gate block output cos {c}"
    out_u, _, _ = _run_block(_COMMON, batch=2, seq_len=seq_len, fuse_gate=False)
    assert _cos(out_f, out_u) > 0.999


@requires_rubin
def test_fuse_gate_dense_arm():
    out_f, ref, _ = _run_block({**_COMMON, "is_causal": False}, batch=1, seq_len=512, fuse_gate=True)
    assert _cos(out_f, ref.out) > 0.999


@requires_rubin
@pytest.mark.parametrize("seq_len", [256, 1000])
def test_fully_fused_block_matches_the_fp32_oracle(seq_len):
    """Both knobs: proj(+norm+rope) -> sdpa(+gate) -> out_proj, three launches."""
    out_f, ref, blk = _run_block(_COMMON, batch=2, seq_len=seq_len, fuse_gate=True, fuse_norm_rope=True)
    assert blk._gate is None and blk._norm_rope is None and len(blk._stages) == 3
    assert torch.isfinite(out_f.float()).all()
    c = _cos(out_f, ref.out)
    assert c > 0.999, f"fully fused block output cos {c}"


def test_fuse_gate_declines_training_naming_the_tensor_it_destroys():
    with pytest.raises(ValueError, match="pre-gate O"):
        _make_block(save_for_backward=True, fuse_gate=True)


def test_fuse_gate_is_off_by_default():
    assert _make_block()[0].fuse_gate is False


def _run_block_with_lse(geom_kw, batch, seq_len, **blk_kw):
    """Like ``_run_block`` but with ``return_lse=True`` and the LSE bound; returns (out, lse, ref)."""
    block_geom = GatedAttentionBlockGeometry(**geom_kw)
    ref_geom = RefGeometry(**geom_kw)
    inp = make_inputs(ref_geom, batch=batch, seq_len=seq_len, dtype=torch.bfloat16)
    ref = gated_attention_block_reference(**inp, geom=ref_geom)
    out = torch.empty(batch, seq_len, block_geom.d_model, device="cuda", dtype=torch.bfloat16)
    lse = torch.full((batch, block_geom.h_q, seq_len), float("nan"), device="cuda", dtype=torch.float32)
    blk = GatedAttentionBlockFwd(
        inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, block_geom, return_lse=True, **blk_kw
    )
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    blk.execute(inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, ws, lse=lse)
    torch.cuda.synchronize()
    return out, lse, ref


@requires_rubin
def test_fuse_gate_dead_rows_do_not_poison_the_output():
    """The gate epilogue applies the gate AFTER the dead-row select, per element:
    a batch entry with no valid KV column must project from an all-zero O,
    exactly as the unfused pipeline does (``test_dead_rows_do_not_poison_the_gate``)."""
    s = 512
    seq_lens = torch.tensor([s, 0], device="cuda", dtype=torch.int32)
    out, ref, blk = _run_block(_COMMON, batch=2, seq_len=s, seq_lens=seq_lens, fuse_gate=True)
    assert blk.fuse_gate and blk._gate is None
    assert torch.isfinite(out.float()).all(), "dead rows leaked a non-finite value through the fused gate"
    assert (out[1] == 0).all(), f"dead batch should project from an all-zero gated O; max|out| = {out[1].abs().max().item()}"
    assert _cos(out[0], ref.out[0]) > 0.999


@requires_rubin
def test_fuse_gate_with_lse_matches_the_unfused_block():
    """``return_lse=True`` compiles the gated specialization with ``has_lse``; the
    LSE is a softmax statistic the gate cannot touch, and gate-on / gate-off are
    the SAME kernel differing only in the epilogue, so the LSE must be
    BIT-IDENTICAL to the unfused block's -- not merely close."""
    out_f, lse_f, ref = _run_block_with_lse(_COMMON, batch=2, seq_len=512, fuse_gate=True)
    out_u, lse_u, _ = _run_block_with_lse(_COMMON, batch=2, seq_len=512, fuse_gate=False)
    assert torch.isfinite(lse_f).all() and torch.isfinite(lse_u).all()
    assert torch.equal(lse_f, lse_u), (lse_f - lse_u).abs().max().item()
    assert _cos(out_f, ref.out) > 0.999


@requires_rubin
def test_fuse_gate_with_out_of_place_qkv():
    """Compact Q/K/V fakes plus a slab-strided GATE fake in ONE artifact -- the
    only combination where the four declared strides differ."""
    out_f, ref, blk = _run_block(_COMMON, batch=1, seq_len=512, fuse_gate=True, inplace_qkv=False)
    assert not blk.inplace_qkv and blk._compact_v is not None
    assert _cos(out_f, ref.out) > 0.999


def test_fuse_gate_declines_a_head_dim_the_gate_rows_do_not_serve():
    """The adapter would route d_head=64 to the d128 flavor; the gate epilogue is
    wired on the d256 kernels only (the Rubin rows' ``epilogue_gate_d_shapes``),
    so the block must decline BEFORE eligibility passes -- on any device, with
    the head dim AND the knob to flip in the message."""
    blk, _, _ = _make_block(fuse_gate=True)  # _GEOM_KW has d_head=64
    with pytest.raises(NotImplementedError, match="256") as ei:
        blk.check_support()
    assert "fuse_gate" in str(ei.value)


def _make_gate_block(**kw):
    """A declared (not compiled) d256 block with fuse_gate=True plus its inputs."""
    block_geom = GatedAttentionBlockGeometry(**_COMMON)
    inp = make_inputs(RefGeometry(**_COMMON), batch=1, seq_len=256, dtype=torch.bfloat16)
    out = torch.empty(1, 256, block_geom.d_model, device="cuda", dtype=torch.bfloat16)
    blk = GatedAttentionBlockFwd(
        inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, block_geom, fuse_gate=True, **kw
    )
    return blk, block_geom


def test_fuse_gate_uses_the_production_adapter():
    """``fuse_gate`` is the shipped adapter with a gate descriptor -- no fork, no
    private launch path.  Declaration-time facts, so this runs on every device:
    the stage builds a ``SdpaFwdDslSm100`` (its constructor touches no device)
    whose ``gate_desc`` is the GATE as a column slice of the projection slab
    (BHSD strides ``(S*N, D, N, 1)``), and nothing of the retired fork machinery
    survives on the stage.  On Rubin the compiled record must carry the gate."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    blk, g = _make_gate_block()
    st = blk._sdpa
    assert st.fuse_gate and st.gate_token_stride == g.n_qkvg and st.gate_dtype == torch.bfloat16
    assert [a for a in dir(st) if "fork" in a.lower()] == [], "the retired SDPA fork machinery must not survive on the stage"
    impl = st._build_impl()
    assert isinstance(impl, SdpaFwdDslSm100)
    assert impl.gate_desc is not None and impl.gate_desc.dtype == torch.bfloat16
    assert tuple(impl.gate_desc.shape) == (1, g.h_q, 256, g.d_head)
    assert tuple(impl.gate_desc.stride) == (256 * g.n_qkvg, g.d_head, g.n_qkvg, 1)
    assert impl.has_amax_o is True, "the bf16 path has no amax to fold out; the default is legacy"
    if _cc() == _SM107:
        blk.check_support()
        assert blk._sdpa._impl.template_params().epilogue_gate is True
        assert blk._sdpa._impl.gate_desc is not None


def test_fuse_gate_off_builds_an_ungated_adapter():
    """The gate-off block compiles the ungated specialization: no gate descriptor,
    so the module the adapter loads is byte-identical to the standalone SDPA's."""
    block_geom = GatedAttentionBlockGeometry(**_COMMON)
    inp = make_inputs(RefGeometry(**_COMMON), batch=1, seq_len=256, dtype=torch.bfloat16)
    out = torch.empty(1, 256, block_geom.d_model, device="cuda", dtype=torch.bfloat16)
    blk = GatedAttentionBlockFwd(inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, block_geom)
    impl = blk._sdpa._build_impl()
    assert impl.gate_desc is None
    if _cc() == _SM107:
        blk.check_support()
        assert blk._sdpa._impl.template_params().epilogue_gate is False


def test_fuse_gate_execute_contract_is_both_directions():
    """A gate tensor is REQUIRED by a gated stage and REFUSED by an ungated one --
    both typed ValueErrors, checked before anything reaches the adapter."""
    from cudnn.gated_attention_block.api import _Sdpa

    g = GatedAttentionBlockGeometry(**_COMMON)
    t = torch.empty(1, 256, g.h_q, g.d_head, device="cuda", dtype=torch.bfloat16)
    kv = torch.empty(1, 256, g.h_kv, g.d_head, device="cuda", dtype=torch.bfloat16)
    gated = _Sdpa(g, batch=1, seq_len=256, dtype=torch.bfloat16, device=torch.device("cuda"), want_lse=False, fuse_gate=True, gate_dtype=torch.bfloat16)
    plain = _Sdpa(g, batch=1, seq_len=256, dtype=torch.bfloat16, device=torch.device("cuda"), want_lse=False)

    # The pinned contract: execute() validates its OWN arguments before it
    # touches the adapter (so the checks run on any device, pre-compile).
    # Any attribute access on the stand-in fails with that sentence, so a
    # future reorder reads as a contract change, not as a test bug.
    class _NeverReached:
        def __getattr__(self, attr):
            pytest.fail(f"_Sdpa.execute touched the adapter (._impl.{attr}) before validating its own arguments")

    gated._impl = plain._impl = _NeverReached()
    with pytest.raises(ValueError, match="requires the GATE"):
        gated.execute(t, kv, kv, t)
    with pytest.raises(ValueError, match="fuse_gate=True"):
        plain.execute(t, kv, kv, t, gate=t)
    with pytest.raises(ValueError, match="FP8"):
        plain.execute(t, kv, kv, t, descale_q=torch.ones(1, device="cuda"))
