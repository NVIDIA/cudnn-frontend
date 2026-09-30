# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The whole block, one call, against the FP32 oracle.

This is the test the other five files exist to make possible: every stage is
individually correct, so what is under test here is the ASSEMBLY — the workspace
carve, the strided column slices of the fused projection, the stage order, and
in particular that the gate is applied AFTER the SDPA's dead-row substitution.

The oracle tests run in two arms, ``norm`` and ``rope_only``
(``GatedAttentionBlockGeometry.qk_norm``): RoPE-only Q/K takes ``None`` for both
norm weights and writes no rstd, and the two are the SAME assembly with one
kernel traced differently -- so both must pass the same oracle bar.
"""

import re
import os

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0


import sys  # noqa: E402

from cudnn.gated_attention_block import GatedAttentionBlockFwd, GatedAttentionBlockGeometry, SavedForBackward  # noqa: E402
from cudnn.gated_attention_block.api import _FusedQkvProjection  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_reference import RefGeometry, gated_attention_block_reference, make_inputs  # noqa: E402

_SM107 = (10, 7)


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


requires_rubin = pytest.mark.skipif(_cc() != _SM107, reason=f"the block targets SM107 only; found {_cc()}")

_COMMON = dict(d_model=512, h_q=8, h_kv=2, d_head=256, rope_dim=64)
_QK_NORM = pytest.mark.parametrize("qk_norm", [True, False], ids=["norm", "rope_only"])


def _fork_supports_qk_norm() -> bool:
    """True once the GEMM forks carry ``NormRopeFusionParams.qk_norm`` (PR-B slice
    S2). Until then ``qk_norm=False + fuse_norm_rope=True`` is a typed decline;
    the tests below INVERT on that fact rather than skipping, so either landing
    order is green."""
    return _FusedQkvProjection._fork_supports_qk_norm()


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


def test_workspace_layout_is_the_declared_composition():
    """The intermediates are a DECLARATION-time fact (``_plan_workspace``), so the
    carve is asserted against its composition on any device: the in-place default
    reserves PROJ and O only, each padded to the 256 B carve alignment, with the
    sub-engines' scratch appended AFTER them.  ``get_workspace_size()`` is a
    POST-compile query by contract (the scratch it adds exists only once the plans
    do -- ``_Projection.workspace_bytes`` raises before ``compile()``), so that
    half runs where the block compiles (Rubin): the carve must be its prefix, the
    remainder at least what every sub-engine asked for, the whole a multiple of
    the alignment (contract § 10: honest, never exceeded)."""
    from cudnn.gated_attention_block.api import _WS_ALIGN, _align_up, _itemsize

    b, s = 4, 128
    g = GatedAttentionBlockGeometry(**_COMMON)
    inp = make_inputs(RefGeometry(**_COMMON), batch=b, seq_len=s, dtype=torch.bfloat16)
    out = torch.empty(b, s, g.d_model, device="cuda", dtype=torch.bfloat16)
    blk = GatedAttentionBlockFwd(inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, g)
    assert blk.inplace_qkv, "the inference default is in-place: PROJ + O only"
    t, e = b * s, _itemsize(torch.bfloat16)
    proj_bytes = _align_up(t * g.n_qkvg * e)
    o_bytes = _align_up(t * g.h_q * g.d_head * e)
    lay = blk._layout()
    assert (lay.proj, lay.o) == (0, proj_bytes), (lay.proj, lay.o, proj_bytes)
    assert lay.q == lay.k == lay.v == -1, "in-place reserves no compact Q/K/V slot"
    assert lay.engine_scratch == lay.total_bytes == proj_bytes + o_bytes
    assert lay.total_bytes % _WS_ALIGN == 0
    if _cc() == _SM107:
        blk.check_support()
        blk.compile()
        ws = blk.get_workspace_size()
        asked = max(blk._proj.workspace_bytes(), blk._out_proj.workspace_bytes(), blk._sdpa.scratch_workspace_bytes(), 1)
        assert ws >= lay.total_bytes + asked, f"workspace {ws} does not cover the carve {lay.total_bytes} plus the engines' scratch {asked}"
        assert (ws - lay.total_bytes) % _WS_ALIGN == 0 and ws % _WS_ALIGN == 0, "the engine region and the total must keep the carve alignment"


@requires_rubin
@_QK_NORM
@pytest.mark.parametrize("seq_len", [256, 512])
def test_matches_the_fp32_oracle(seq_len, qk_norm):
    out, ref, blk = _run_block({**_COMMON, "qk_norm": qk_norm}, batch=2, seq_len=seq_len)
    assert torch.isfinite(out.float()).all()
    c = _cos(out, ref.out)
    assert c > 0.999, f"block output cos {c}"
    assert blk._norm_rope.want_rstd is False and blk._norm_rope._recipe.apply_norm is qk_norm
    assert (ref.rstd_q is None) is (not qk_norm)


@requires_rubin
def test_rope_only_passthrough_dims_are_bit_exact_through_the_block():
    """Out-of-place (``inplace_qkv=False``) keeps the pre-RoPE slab AND the compact
    rotated Q/K in the workspace, so the block-level proof of the RoPE-only
    contract is direct: the dims ``[rope_dim, D)`` of compact Q/K equal the slab's
    columns bit for bit (no fp32 op touched them), and the rope band differs."""
    from cudnn.gated_attention_block.api import _cols, _view

    geom_kw = {**_COMMON, "qk_norm": False}
    block_geom = GatedAttentionBlockGeometry(**geom_kw)
    inp = make_inputs(RefGeometry(**geom_kw), batch=1, seq_len=256, dtype=torch.bfloat16)
    assert inp["w_q_norm"] is None and inp["w_k_norm"] is None
    out = torch.empty(1, 256, block_geom.d_model, device="cuda", dtype=torch.bfloat16)
    blk = GatedAttentionBlockFwd(inp["h"], inp["w_qkvg"], None, None, inp["cos"], inp["sin"], inp["w_o"], out, block_geom, inplace_qkv=False)
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    blk.execute(inp["h"], inp["w_qkvg"], None, None, inp["cos"], inp["sin"], inp["w_o"], out, ws)
    torch.cuda.synchronize()
    g, lay, t, r = block_geom, blk._ws, 256, block_geom.rope_dim
    proj = _view(ws, lay.proj, (t, g.n_qkvg), torch.bfloat16)
    o_q, _, o_k, _ = g.qkvg_offsets
    q_pre, k_pre = _cols(proj, o_q, g.h_q, g.d_head), _cols(proj, o_k, g.h_kv, g.d_head)
    q_c = _view(ws, lay.q, (t, g.h_q, g.d_head), torch.bfloat16)
    k_c = _view(ws, lay.k, (t, g.h_kv, g.d_head), torch.bfloat16)
    assert torch.equal(q_c[..., r:], q_pre[..., r:]) and torch.equal(k_c[..., r:], k_pre[..., r:]), "RoPE-only passthrough dims are not a bit-exact copy"
    assert not torch.equal(q_c[..., :r], q_pre[..., :r]), "the rope band did not rotate"
    assert torch.isfinite(out.float()).all()


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


def _park_the_default_stream(seconds: float = 0.5) -> None:
    """Enqueue a long spin on torch's CURRENT (default) stream so that anything a
    stage wrongly launches there runs LATE -- after a side stream is long done."""
    if hasattr(torch.cuda, "_sleep"):
        torch.cuda._sleep(int(seconds * 2.0e9))  # cycles at ~2 GHz
        return
    x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    for _ in range(16):
        x = x @ x


@requires_rubin
@pytest.mark.parametrize("how", ["ambient", "explicit"])
def test_a_caller_stream_orders_every_stage(how):
    """Every stage -- the CuTe-DSL kernels AND the two FROST GEMMs -- launches on
    ONE stream, the caller's: ambient (``with torch.cuda.stream(s):``) or explicit
    (``current_stream=``).  The GEMMs take it through ``run_proj_gemm(stream=)``
    (the JIT plan's own ``stream=``; the per-device cuDNN handle re-streamed to it
    on the graph route).  Before that they launched on the DEFAULT stream regardless,
    so under a caller stream the SDPA consumed a slab the projection had not
    written yet -- all-zero output, reproduced on SM107 (Rule 5).  This is the
    inverse of the refusal the v1 block shipped.

    The probe makes the race DETERMINISTIC rather than lucky.  The workspace is
    zero-filled and the default stream is parked behind a long spin, so a stage
    enqueued there runs late: a late PRODUCER leaves its consumers reading zeros,
    and a late CONSUMER (out_proj) reads the zeros the side stream writes over the
    workspace right after the block.  Correct threading gives an output
    BIT-IDENTICAL to the default-stream run (the block is deterministic --
    ``test_a_second_execute_reuses_the_plan_and_agrees``)."""
    import cuda.bindings.driver as cuda_drv

    out_ref, _, blk = _run_block(_COMMON, batch=1, seq_len=256)  # default stream, synchronized
    assert out_ref.abs().max().item() > 0
    inp = make_inputs(RefGeometry(**_COMMON), batch=1, seq_len=256, dtype=torch.bfloat16)
    ws = torch.zeros(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    out = torch.zeros_like(out_ref)
    side = torch.cuda.Stream()
    torch.cuda.synchronize()
    args = (inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, ws)
    _park_the_default_stream()
    if how == "ambient":
        with torch.cuda.stream(side):
            blk.execute(*args)
    else:
        blk.execute(*args, current_stream=cuda_drv.CUstream(side.cuda_stream))
    with torch.cuda.stream(side):
        ws.zero_()  # ordered AFTER the block on the side stream; a consumer parked on the default stream would read this instead
    torch.cuda.synchronize()
    assert torch.equal(out, out_ref), (
        f"a stage ran off the caller's stream ({how}): max|diff| = {(out.float() - out_ref.float()).abs().max().item()}, "
        f"zeros = {(out == 0).float().mean().item():.0%}"
    )


def test_run_proj_gemm_refuses_a_handle_bound_to_another_stream():
    """Given BOTH a handle and a stream, they must agree: a handle bound elsewhere
    would run the GEMM off the launch stream the caller ordered everything else on
    -- the race the stream threading closes -- so it is a typed ``ValueError``
    before any route is taken (no graph, no JIT, no launch).  Runs on any device:
    the check needs only a cuDNN handle."""
    import cudnn
    from cudnn.gated_attention_block.kernels.proj_gemm import ProjGemmPlan, run_proj_gemm

    side = torch.cuda.Stream()
    h = cudnn.create_handle()
    cudnn.set_stream(handle=h, stream=side.cuda_stream)
    plan = ProjGemmPlan(graph=None, a=None, b=None, c=None, m=1, k=1, n=1, label="probe")  # never launched: the check precedes every route
    with pytest.raises(ValueError, match="bound to stream"):
        run_proj_gemm(plan, None, None, None, None, h, stream=torch.cuda.default_stream().cuda_stream)


def test_graph_route_handle_is_per_device_and_restreamed(monkeypatch):
    """R7 (torch-op form): the graph route owns ONE process-lifetime cuDNN handle per
    device and re-streams it with ``cudnn.set_stream`` before every launch -- so a
    second execute, on any stream, never calls ``cudnnCreate`` again (a captured
    execute records no resource creation).  Runs on any device: a stand-in graph
    records the handle it is executed with; no kernel launches."""
    import cudnn
    from cudnn.gated_attention_block.kernels.proj_gemm import ProjGemmPlan, _graph_handle_lock, graph_handle, run_proj_gemm

    dev = torch.device("cuda", torch.cuda.current_device())
    h = graph_handle(dev)
    assert graph_handle(torch.device("cuda")) is h and graph_handle(dev) is h, "one handle per device, whichever spelling names it"
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    own = cudnn.create_handle()  # a caller-owned handle, made before cudnnCreate is forbidden below
    cudnn.set_stream(handle=own, stream=s2.cuda_stream)

    seen = []

    class _Graph:  # the plan's graph, standing in for the backend: records (handle, its stream, lock held?) per execute
        def execute(self, vp, workspace, handle):
            seen.append((handle, int(cudnn.get_stream(handle)), _graph_handle_lock(dev).locked()))

    plan = ProjGemmPlan(graph=_Graph(), a="a", b="b", c="c", m=1, k=1, n=1, label="probe")  # route None -> the graph route, no JIT
    x = torch.zeros(1, 1, device="cuda")
    monkeypatch.setattr(cudnn, "create_handle", lambda: pytest.fail("run_proj_gemm called cudnnCreate: the per-device handle must already exist"))
    run_proj_gemm(plan, x, x, x, x, stream=s1.cuda_stream)
    assert seen[-1] == (h, s1.cuda_stream, True) and cudnn.get_stream(h) == s1.cuda_stream
    run_proj_gemm(plan, x, x, x, x, stream=s2.cuda_stream)
    assert seen[-1] == (h, s2.cuda_stream, True) and cudnn.get_stream(h) == s2.cuda_stream, "the same handle, re-streamed"
    run_proj_gemm(plan, x, x, x, x)  # no stream: torch's current stream on out.device
    assert seen[-1] == (h, torch.cuda.current_stream().cuda_stream, True)
    run_proj_gemm(plan, x, x, x, x, h)  # the shared handle handed back in: same lock, the stream it is bound to
    assert seen[-1] == (h, torch.cuda.current_stream().cuda_stream, True), "the shared handle passed explicitly is serialised like the implicit one"
    run_proj_gemm(plan, x, x, x, x, own)  # a caller-owned handle: its own stream, no lock
    assert seen[-1] == (own, s2.cuda_stream, False)
    assert len(seen) == 5


# ---------------------------------------------------------------------------
# In-place Q/K/V: the workspace win, and the backward guard that bounds it
# ---------------------------------------------------------------------------

_GEOM_KW = dict(d_model=512, h_q=8, h_kv=2, d_head=64, rope_dim=32)


def _make_block(batch=1, seq_len=256, dtype=torch.bfloat16, geom_kw=None, **kw):
    """A declared (not yet compiled) block plus the inputs it was declared for.
    ``geom_kw`` overrides ``_GEOM_KW`` (e.g. ``qk_norm=False`` -> None norm weights)."""
    geom_kw = _GEOM_KW if geom_kw is None else geom_kw
    block_geom = GatedAttentionBlockGeometry(**geom_kw)
    ref_geom = RefGeometry(**geom_kw)
    inp = make_inputs(ref_geom, batch=batch, seq_len=seq_len, dtype=dtype)
    out = torch.empty(batch, seq_len, block_geom.d_model, device="cuda", dtype=dtype)
    blk = GatedAttentionBlockFwd(inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, block_geom, **kw)
    return blk, inp, out


# ---------------------------------------------------------------------------
# qk_norm=False (RoPE-only Q/K): the knob and its weight / rstd contract
# ---------------------------------------------------------------------------


def test_qk_norm_is_on_by_default_and_declares_no_rstd_when_off():
    assert GatedAttentionBlockGeometry(**_GEOM_KW).qk_norm is True
    on, _, _ = _make_block(save_for_backward=True)
    off, inp, _ = _make_block(save_for_backward=True, geom_kw={**_GEOM_KW, "qk_norm": False})
    assert inp["w_q_norm"] is None and inp["w_k_norm"] is None
    # Same stage list, same launch count: the norm is folded out of ONE kernel, not a stage removed.
    assert [s.name for s in on._stages] == [s.name for s in off._stages]
    assert on._norm_rope.want_rstd is True and off._norm_rope.want_rstd is False
    assert off._descs["w_q_norm"] is None and off._descs["w_k_norm"] is None
    # SavedForBackward carries None rstd for a RoPE-only forward (positional slots kept).
    z = torch.empty(0)
    assert SavedForBackward(h=z, gate=z, o=z, lse=z, rstd_q=None, rstd_k=None).rstd_q is None


def test_qk_norm_flag_and_weights_must_agree():
    """Both directions, typed, naming ``geometry.qk_norm`` -- at DECLARATION.

    Load-bearing: ``_make_tensor_desc(None)`` returns None silently, so without
    this check a norm-on block declared with None weights would only die in
    ``check_support``'s dtype loop with an untyped AttributeError."""
    geom_on = GatedAttentionBlockGeometry(**_GEOM_KW)
    geom_off = GatedAttentionBlockGeometry(**{**_GEOM_KW, "qk_norm": False})
    inp = make_inputs(RefGeometry(**_GEOM_KW), batch=1, seq_len=256, dtype=torch.bfloat16)
    out = torch.empty(1, 256, geom_on.d_model, device="cuda", dtype=torch.bfloat16)
    w = inp["w_q_norm"]
    decl = lambda geom, wq, wk: GatedAttentionBlockFwd(inp["h"], inp["w_qkvg"], wq, wk, inp["cos"], inp["sin"], inp["w_o"], out, geom)  # noqa: E731
    with pytest.raises(ValueError, match="qk_norm=True") as ei:
        decl(geom_on, None, None)  # norm ON, no weights
    assert "qk_norm=False" in str(ei.value), "the message must name the knob to flip"
    with pytest.raises(ValueError, match="qk_norm=False"):
        decl(geom_off, w, w)  # norm OFF, weights given
    for wq, wk in ((w, None), (None, w)):  # mixed, either geometry
        with pytest.raises(ValueError, match="together"):
            decl(geom_on, wq, wk)
        with pytest.raises(ValueError, match="together"):
            decl(geom_off, wq, wk)
    # The consistent pairs declare fine.
    decl(geom_on, w, w)
    decl(geom_off, None, None)


def test_qk_norm_off_with_rope_dim_zero_is_refused_at_the_geometry():
    """No norm and no RoPE would make stage (2)+(3) an identity copy: refuse, do not launch."""
    with pytest.raises(ValueError, match="identity copy"):
        GatedAttentionBlockGeometry(**{**_GEOM_KW, "rope_dim": 0, "qk_norm": False}).validate()
    # qk_norm_eps stays validated regardless of the knob (D10: the fused fork re-checks it after eligibility).
    with pytest.raises(ValueError, match="qk_norm_eps"):
        GatedAttentionBlockGeometry(**{**_GEOM_KW, "qk_norm": False, "qk_norm_eps": 0.0}).validate()


def test_rope_only_with_fuse_norm_rope_is_a_typed_decline_until_the_fork_knows_the_knob():
    """``qk_norm=False + fuse_norm_rope=True`` needs the GEMM forks' RoPE-only
    epilogue (``NormRopeFusionParams.qk_norm``, PR-B slice S2). Before it lands
    the stage declines with a NotImplementedError naming the knob -- never a
    silently norm-ON artifact; once it lands, ``params()`` carries the field."""
    blk, _, _ = _make_block(fuse_norm_rope=True, geom_kw={**_COMMON, "qk_norm": False})
    assert isinstance(blk._proj, _FusedQkvProjection)
    if _fork_supports_qk_norm():
        assert blk._proj.params().qk_norm is False
    else:
        with pytest.raises(NotImplementedError, match="qk_norm"):
            blk._proj.check_support()
        with pytest.raises(NotImplementedError, match="qk_norm"):
            blk._proj.params()
    # The unfused chain never needs the fork: qk_norm=False is served there on any checkout.
    unfused, _, _ = _make_block(geom_kw={**_COMMON, "qk_norm": False})
    assert unfused._norm_rope is not None and unfused._norm_rope.want_rstd is False


@requires_rubin
def test_qk_norm_execute_checks_weights_against_the_geometry_both_ways():
    """The same contract at EXECUTE, on a compiled block, before any launch."""
    for qk_norm in (True, False):
        blk, inp, out = _make_block(geom_kw={**_GEOM_KW, "qk_norm": qk_norm})
        blk.check_support()
        blk.compile()
        ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
        w = torch.ones(blk.geom.d_head, device="cuda", dtype=torch.bfloat16)
        wrong = (None, None) if qk_norm else (w, w)
        with pytest.raises(ValueError, match=f"qk_norm={qk_norm}"):
            blk.execute(inp["h"], inp["w_qkvg"], *wrong, inp["cos"], inp["sin"], inp["w_o"], out, ws)
        with pytest.raises(ValueError, match="together"):
            blk.execute(inp["h"], inp["w_qkvg"], w, None, inp["cos"], inp["sin"], inp["w_o"], out, ws)


def test_bwd_declaration_contracts_fire_before_the_stub_decline():
    """The (still stubbed) backward's four declaration-time contracts are typed
    ValueErrors that must fire BEFORE the ``NotImplementedError`` stub at the end
    of ``__init__`` -- pinned so a later edit that hoists the stub cannot silently
    drop them. Nothing here touches a GPU: the checks read None-ness and the
    geometry only, so placeholders stand in for every tensor."""
    from cudnn.gated_attention_block import GatedAttentionBlockBwd

    z = torch.empty(0)
    w = torch.ones(_GEOM_KW["d_head"])
    geom_on = GatedAttentionBlockGeometry(**_GEOM_KW)
    geom_off = GatedAttentionBlockGeometry(**{**_GEOM_KW, "qk_norm": False})
    saved = lambda rstd: SavedForBackward(h=z, gate=z, o=z, lse=z, rstd_q=rstd, rstd_k=rstd)  # noqa: E731
    decl = lambda geom, wq, wk, rstd, **kw: GatedAttentionBlockBwd(z, saved(rstd), z, wq, wk, z, z, z, geom, **kw)  # noqa: E731
    # 1. sample weights agree with geometry.qk_norm, both ways -- the forward's helper, same messages.
    with pytest.raises(ValueError, match="qk_norm=True"):
        decl(geom_on, None, None, z)
    with pytest.raises(ValueError, match="qk_norm=False"):
        decl(geom_off, w, w, None)
    with pytest.raises(ValueError, match="together"):
        decl(geom_on, w, None, z)
    # 2. an explicit need_dw_norms=True under norm-off is a typed decline, never a silent False.
    with pytest.raises(ValueError, match="need_dw_norms=True"):
        decl(geom_off, None, None, None, need_dw_norms=True)
    # 3. the saved rstd must be None under norm-off (the forward wrote none).
    with pytest.raises(ValueError, match="rstd_q / rstd_k must be None"):
        decl(geom_off, None, None, z)
    # 4. consistent calls pass every contract and reach the stub; need_dw_norms=None resolves
    #    to the knob. __init__ raises before returning, so build the instance by hand to read it.
    for geom, wq, rstd, kw, want in (
        (geom_on, w, z, {}, True),
        (geom_on, w, z, {"need_dw_norms": False}, False),
        (geom_off, None, None, {}, False),
    ):
        obj = GatedAttentionBlockBwd.__new__(GatedAttentionBlockBwd)
        with pytest.raises(NotImplementedError, match=re.escape("GatedAttentionBlockBwd.__init__")):
            obj.__init__(z, saved(rstd), z, wq, wq, z, z, z, geom, **kw)
        assert obj.need_dw_norms is want


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
@_QK_NORM
@pytest.mark.parametrize("seq_len", [256, 1000])  # 1000: tail tile + rows past M in the fused epilogue
def test_fuse_norm_rope_matches_the_fp32_oracle(seq_len, qk_norm):
    """Same block, same oracle, one launch fewer.  Not bit-identical to the
    unfused chain BY DESIGN (the fork norms the fp32 accumulator; the chain
    norms bf16-rounded values), so both are scored against the fp32 reference.
    ``rope_only`` INVERTS while the fork lacks the knob: a typed decline, not a
    silently norm-ON artifact (see ``_fork_supports_qk_norm``)."""
    geom_kw = {**_COMMON, "qk_norm": qk_norm}
    if not qk_norm and not _fork_supports_qk_norm():
        with pytest.raises(NotImplementedError, match="qk_norm"):
            _run_block(geom_kw, batch=2, seq_len=seq_len, fuse_norm_rope=True)
        return
    out_f, ref, blk = _run_block(geom_kw, batch=2, seq_len=seq_len, fuse_norm_rope=True)
    assert blk.fuse_norm_rope and blk._norm_rope is None and len(blk._stages) == 4
    assert torch.isfinite(out_f.float()).all()
    c = _cos(out_f, ref.out)
    assert c > 0.999, f"fused block output cos {c}"
    out_u, _, _ = _run_block(geom_kw, batch=2, seq_len=seq_len, fuse_norm_rope=False)
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
@_QK_NORM
@pytest.mark.parametrize("seq_len", [256, 1000])  # 1000: tail tile + rows past S
def test_fuse_gate_matches_the_fp32_oracle_causal(seq_len, qk_norm):
    """The block default is CAUSAL and the kernel's mask arms are const_expr-folded,
    so this exercises the arm a dense pass proves nothing about."""
    geom_kw = {**_COMMON, "qk_norm": qk_norm}
    out_f, ref, blk = _run_block(geom_kw, batch=2, seq_len=seq_len, fuse_gate=True)
    assert blk.fuse_gate and blk._gate is None and len(blk._stages) == 4
    assert torch.isfinite(out_f.float()).all()
    c = _cos(out_f, ref.out)
    assert c > 0.999, f"fused-gate block output cos {c}"
    out_u, _, _ = _run_block(geom_kw, batch=2, seq_len=seq_len, fuse_gate=False)
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
