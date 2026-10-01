# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The TRAINING forward of the gated attention block (``save_for_backward=True``).

What is under test is the forward/backward BOUNDARY, not the math: every tensor
the ``SavedForBackward`` record receives is compared to the fp32 oracle's own
copy of it (``RefOutputs`` exposes every saved intermediate), the block's ``out``
is pinned bitwise to the inference block's (same kernels, different buffers),
and the contracts the backward's declaration-time guards rest on -- the saved
views alias ``proj_slab``, ``saved.h`` IS ``h``, ``saved.seq_lens`` IS the
``seq_lens`` the forward ran with, ``lse`` defaults to ``saved.lse`` -- are typed
``ValueError`` exceptions that fire before any launch.

Two SAVE MODES ship:

* ``proj_slab`` (the declaration default): the stage-(1) GEMM writes the
  caller-owned ``saved.proj_slab`` and ``gate`` / ``q_pre`` / ``k_pre`` / ``v``
  are its column bands (``saved_slab_views``) -- zero copies, 34 KiB/token saved
  at the 397B geometry;
* ``gate_copy`` (``saved_gate_copy=True``): the slab stays in the workspace and
  ONE elementwise launch copies the GATE band into a compact ``saved.gate``
  (``q_pre`` / ``k_pre`` only if the caller passed buffers) -- 16 KiB/token
  saved, the backward recomputes the rest.

Accept tests are ``requires_rubin`` (the block targets SM107 only); the reject
tests build CUDA tensors for a DECLARED block (``requires_cuda``: any CUDA
device, no compile); the pure-layout tests (``saved_slab_views``,
``_plan_workspace``) run anywhere, CPU included.
"""

import dataclasses
import os
import sys

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

from cudnn.gated_attention_block import GatedAttentionBlockFwd, GatedAttentionBlockGeometry, SavedForBackward, saved_slab_views  # noqa: E402
from cudnn.gated_attention_block.api import _plan_workspace  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_reference import RefGeometry, gated_attention_block_reference, make_inputs, qk_norm_rope_reference  # noqa: E402

_SM107 = (10, 7)


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


# Hoisted into cutedsl/conftest.py (registered there; the skip is applied at collection) -- scoped to the modules under
# that directory, which is where every gated-block CuTeDSL module lives.
requires_rubin = pytest.mark.requires_rubin
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

_COMMON = dict(d_model=512, h_q=8, h_kv=2, d_head=256, rope_dim=64)
_QK_NORM = pytest.mark.parametrize("qk_norm", [True, False], ids=["norm", "rope_only"])
_SAVE_MODE = pytest.mark.parametrize("save_mode", ["proj_slab", "gate_copy"])


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def _alloc_saved(geom, inp, batch, seq_len, *, save_mode, with_pre=False, seq_lens=None, sentinel=None):
    """A caller-owned ``SavedForBackward`` for ``save_mode`` (``inp["h"]`` IS ``saved.h``).

    ``with_pre`` (gate-copy mode only): also pass compact ``q_pre`` / ``k_pre`` buffers, so the block copies them.
    ``sentinel``: fill every caller buffer with it, so an untouched cell is recognisable."""
    b, s, g = batch, seq_len, geom
    dtype, dev = inp["h"].dtype, inp["h"].device

    def buf(*shape, dt=dtype):
        t = torch.empty(*shape, dtype=dt, device=dev)
        if sentinel is not None:
            t.fill_(sentinel)
        return t

    rstd_q = buf(b, s, g.h_q, dt=torch.float32) if g.qk_norm else None
    rstd_k = buf(b, s, g.h_kv, dt=torch.float32) if g.qk_norm else None
    common = dict(h=inp["h"], o=buf(b, s, g.h_q, g.d_head), lse=buf(b, g.h_q, s, dt=torch.float32), rstd_q=rstd_q, rstd_k=rstd_k, seq_lens=seq_lens)
    if save_mode == "proj_slab":
        proj_slab = buf(b * s, g.n_qkvg)
        q_pre, gate, k_pre, _v = saved_slab_views(proj_slab, g, b, s)
        return SavedForBackward(gate=gate, q_pre=q_pre, k_pre=k_pre, proj_slab=proj_slab, **common)
    assert save_mode == "gate_copy"
    q_pre = buf(b, s, g.h_q, g.d_head) if with_pre else None
    k_pre = buf(b, s, g.h_kv, g.d_head) if with_pre else None
    return SavedForBackward(gate=buf(b, s, g.h_q, g.d_head), q_pre=q_pre, k_pre=k_pre, proj_slab=None, **common)


def _declare(geom_kw, batch, seq_len, *, save_mode="proj_slab", dtype=torch.bfloat16, seq_lens_present=False, **blk_kw):
    """A DECLARED (not compiled) training block plus its inputs and output buffer."""
    block_geom = GatedAttentionBlockGeometry(**geom_kw)
    inp = make_inputs(RefGeometry(**geom_kw), batch=batch, seq_len=seq_len, dtype=dtype)
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
        save_for_backward=True,
        seq_lens_present=seq_lens_present,
        saved_gate_copy=(save_mode == "gate_copy"),
        **blk_kw,
    )
    return blk, inp, out


def _execute(blk, inp, out, ws, *, saved, seq_lens=None, lse=None):
    blk.execute(inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, ws, seq_lens=seq_lens, lse=lse, saved=saved)


def _run_training(geom_kw, batch, seq_len, *, save_mode, seq_lens=None, with_pre=False, dtype=torch.bfloat16, sentinel=None):
    """Declare, compile and run the training forward; returns ``(out, ref, blk, saved, inp)``."""
    blk, inp, out = _declare(geom_kw, batch, seq_len, save_mode=save_mode, dtype=dtype, seq_lens_present=seq_lens is not None)
    ref = gated_attention_block_reference(**inp, geom=RefGeometry(**geom_kw), seq_lens=seq_lens)
    saved = _alloc_saved(blk.geom, inp, batch, seq_len, save_mode=save_mode, with_pre=with_pre, seq_lens=seq_lens, sentinel=sentinel)
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    _execute(blk, inp, out, ws, saved=saved, seq_lens=seq_lens)
    torch.cuda.synchronize()
    return out, ref, blk, saved, inp


# ---------------------------------------------------------------------------
# saved_slab_views: the zero-copy spelling of the four bands (any device, CPU is fine)
# ---------------------------------------------------------------------------


def test_saved_slab_views_are_strided_views_of_the_slab():
    """``(q_pre, gate, k_pre, v)`` as ``[B, S, heads, D]`` VIEWS: same storage, token stride ``n_qkvg``, head stride
    ``d_head``, storage offsets at ``qkvg_offsets`` -- and each equals the plain column slice of the ``[B, S, N]`` slab.
    A ``[B, S, N]`` or ``[B*S, N]`` slab both spell the same views; a wrong element count, a non-contiguous slab or a
    slab whose base is not 16-B aligned (the GEMM TMA-stores it) is a typed ``ValueError``."""
    g = GatedAttentionBlockGeometry(**_COMMON)
    b, s, n = 2, 8, g.n_qkvg
    slab = torch.randn(b * s, n, dtype=torch.bfloat16)
    views = saved_slab_views(slab, g, b, s)
    slab3 = slab.view(b, s, n)
    o_q, o_g, o_k, o_v = g.qkvg_offsets
    bands = ((o_q, g.h_q), (o_g, g.h_q), (o_k, g.h_kv), (o_v, g.h_kv))
    assert len(views) == 4
    for v, (off, h) in zip(views, bands):
        assert v.data_ptr() == slab.data_ptr() + off * slab.element_size(), "not a view at the band's column offset"
        assert tuple(v.shape) == (b, s, h, g.d_head)
        assert v.stride() == (s * n, n, g.d_head, 1), v.stride()
        assert torch.equal(v, slab3[..., off : off + h * g.d_head].view(b, s, h, g.d_head))
    for v3, v2 in zip(saved_slab_views(slab3, g, b, s), views):
        assert v3.data_ptr() == v2.data_ptr() and v3.stride() == v2.stride() and v3.shape == v2.shape
    with pytest.raises(ValueError, match="n_qkvg"):
        saved_slab_views(torch.empty(b * s, n - 1, dtype=torch.bfloat16), g, b, s)
    with pytest.raises(ValueError, match="contiguous"):
        saved_slab_views(torch.empty(b * s, 2 * n, dtype=torch.bfloat16)[:, ::2], g, b, s)
    with pytest.raises(ValueError, match="aligned"):  # right count, contiguous, base at an odd element offset (data_ptr % 16 == 2)
        saved_slab_views(_odd_offset((b * s, n), dtype=torch.bfloat16, device="cpu"), g, b, s)


def _odd_offset(shape, dtype, device="cuda"):
    """A contiguous tensor of exactly ``shape`` / ``dtype`` whose base sits ONE element past an allocation: right count,
    dtype, device and strides, but ``data_ptr() % 16 == itemsize`` -- the misaligned caller buffer the record checks
    must refuse before a TMA store or a 16-B vector store hits it."""
    n = 1
    for x in shape:
        n *= int(x)
    return torch.empty(n + 1, device=device, dtype=dtype)[1:].view(*shape)


# ---------------------------------------------------------------------------
# Workspace layout: byte-identical without want_saved; the training carve
# ---------------------------------------------------------------------------

# The frozen offsets of EVERY existing pipeline at the suite geometry (B=2, S=1000) and the 397B geometry (B=1, S=4096),
# computed on develop dd3235c3 (before `want_saved` existed) and pasted as literals on purpose: a change to any number here
# is a workspace-layout change for a caller that never asked for a training forward.
_GEOM_397B = dict(d_model=4096, h_q=32, h_kv=2, d_head=256, rope_dim=64)
_SNAPSHOT_SHAPES = {"test": (_COMMON, 2, 1000), "397b": (_GEOM_397B, 1, 4096)}
_SNAPSHOT_ARMS = {
    "bf16_inplace": dict(inplace_qkv=True),
    "bf16_compact": dict(inplace_qkv=False),
    "fp8_unfused": dict(inplace_qkv=True, fp8=True),
    "fp8_fused": dict(inplace_qkv=True, fp8=True, fp8_fused=True),
    "mxfp8_unfused": dict(inplace_qkv=True, fp8=True, mxfp8=True),
    "mxfp8_fused": dict(inplace_qkv=True, fp8=True, fp8_fused=True, mxfp8=True),
}
_ABSENT = dict(gate=-1, o_gated=-1, base_align=256, o4=-1, sf_o=-1)
_NO_Q8 = dict(q8=-1, k8=-1, v8=-1, o8=-1, gate16=-1)
_NO_SF = dict(sf_q=-1, sf_k=-1, sf_v=-1)
_SNAPSHOT = {
    ("test", "bf16_inplace"): dict(proj=0, q=-1, k=-1, v=-1, o=20480000, engine_scratch=28672000, total_bytes=28672000, **_NO_Q8, **_NO_SF),
    ("test", "bf16_compact"): dict(proj=0, q=20480000, k=28672000, v=30720000, o=32768000, engine_scratch=40960000, total_bytes=40960000, **_NO_Q8, **_NO_SF),
    ("test", "fp8_unfused"): dict(
        proj=0,
        q=-1,
        k=-1,
        v=-1,
        o=26624000,
        engine_scratch=38912000,
        total_bytes=38912000,
        q8=20480000,
        k8=24576000,
        v8=25600000,
        o8=34816000,
        gate16=-1,
        **_NO_SF,
    ),
    ("test", "fp8_fused"): dict(
        proj=-1,
        q=-1,
        k=-1,
        v=-1,
        o=-1,
        engine_scratch=18432000,
        total_bytes=18432000,
        q8=0,
        k8=4096000,
        v8=5120000,
        o8=14336000,
        gate16=6144000,
        **_NO_SF,
    ),
    ("test", "mxfp8_unfused"): dict(
        proj=0,
        q=-1,
        k=-1,
        v=-1,
        o=26624000,
        engine_scratch=39108608,
        total_bytes=39108608,
        q8=20480000,
        k8=24576000,
        v8=25600000,
        o8=34816000,
        gate16=-1,
        sf_q=38912000,
        sf_k=39043072,
        sf_v=39075840,
    ),
    ("test", "mxfp8_fused"): dict(
        proj=-1,
        q=-1,
        k=-1,
        v=-1,
        o=-1,
        engine_scratch=18628608,
        total_bytes=18628608,
        q8=0,
        k8=4096000,
        v8=5120000,
        o8=14336000,
        gate16=6144000,
        sf_q=18432000,
        sf_k=18563072,
        sf_v=18595840,
    ),
    ("397b", "bf16_inplace"): dict(proj=0, q=-1, k=-1, v=-1, o=142606336, engine_scratch=209715200, total_bytes=209715200, **_NO_Q8, **_NO_SF),
    ("397b", "bf16_compact"): dict(
        proj=0, q=142606336, k=209715200, v=213909504, o=218103808, engine_scratch=285212672, total_bytes=285212672, **_NO_Q8, **_NO_SF
    ),
    ("397b", "fp8_unfused"): dict(
        proj=0,
        q=-1,
        k=-1,
        v=-1,
        o=180355072,
        engine_scratch=281018368,
        total_bytes=281018368,
        q8=142606336,
        k8=176160768,
        v8=178257920,
        o8=247463936,
        gate16=-1,
        **_NO_SF,
    ),
    ("397b", "fp8_fused"): dict(
        proj=-1,
        q=-1,
        k=-1,
        v=-1,
        o=-1,
        engine_scratch=138412032,
        total_bytes=138412032,
        q8=0,
        k8=33554432,
        v8=35651584,
        o8=104857600,
        gate16=37748736,
        **_NO_SF,
    ),
    ("397b", "mxfp8_unfused"): dict(
        proj=0,
        q=-1,
        k=-1,
        v=-1,
        o=180355072,
        engine_scratch=282198016,
        total_bytes=282198016,
        q8=142606336,
        k8=176160768,
        v8=178257920,
        o8=247463936,
        gate16=-1,
        sf_q=281018368,
        sf_k=282066944,
        sf_v=282132480,
    ),
    ("397b", "mxfp8_fused"): dict(
        proj=-1,
        q=-1,
        k=-1,
        v=-1,
        o=-1,
        engine_scratch=139591680,
        total_bytes=139591680,
        q8=0,
        k8=33554432,
        v8=35651584,
        o8=104857600,
        gate16=37748736,
        sf_q=138412032,
        sf_k=139460608,
        sf_v=139526144,
    ),
}


@pytest.mark.parametrize("arm", list(_SNAPSHOT_ARMS))
@pytest.mark.parametrize("shape", list(_SNAPSHOT_SHAPES))
def test_workspace_layout_is_byte_identical_without_want_saved(shape, arm):
    """Every field of ``_plan_workspace`` for every pre-existing pipeline equals the frozen snapshot, and the appended
    ``want_saved=False, saved_gate_copy=False`` spells the same layout as not passing them.  No GPU."""
    geom_kw, b, s = _SNAPSHOT_SHAPES[shape]
    kw = dict(_SNAPSHOT_ARMS[arm])
    inplace = kw.pop("inplace_qkv")
    geom = GatedAttentionBlockGeometry(**geom_kw)
    lay = _plan_workspace(geom, b, s, torch.bfloat16, False, False, inplace, **kw)
    assert dataclasses.asdict(lay) == {**_ABSENT, **_SNAPSHOT[(shape, arm)]}
    assert lay == _plan_workspace(geom, b, s, torch.bfloat16, False, False, inplace, want_saved=False, saved_gate_copy=False, **kw)


@pytest.mark.parametrize("shape", list(_SNAPSHOT_SHAPES))
def test_workspace_layout_under_want_saved(shape):
    """``want_saved`` on the bf16 out-of-place layout: ``o`` is gone (the SDPA writes ``saved.o``), ``o_gated`` is
    reserved (stage (5) gates OUT of place, stage (6) reads it), q / k / v stay; ``proj`` is gone in the proj_slab
    mode (the GEMM writes ``saved.proj_slab``) and reserved in the gate-copy mode.  Slots follow stage order and stay
    256-B aligned; the training layout that disagrees with its body (in place, or a quantized pipeline) is a typed
    ``ValueError`` rather than a silent carve.  No GPU."""
    from cudnn.gated_attention_block.api import _WS_ALIGN, _align_up

    geom_kw, b, s = _SNAPSHOT_SHAPES[shape]
    g = GatedAttentionBlockGeometry(**geom_kw)
    t, e = b * s, 2
    base = _plan_workspace(g, b, s, torch.bfloat16, True, True, False)
    slab = _plan_workspace(g, b, s, torch.bfloat16, True, True, False, want_saved=True)
    copy = _plan_workspace(g, b, s, torch.bfloat16, True, True, False, want_saved=True, saved_gate_copy=True)
    q_b, kv_b, o_b = _align_up(t * g.h_q * g.d_head * e), _align_up(t * g.h_kv * g.d_head * e), _align_up(t * g.h_q * g.d_head * e)
    proj_b = _align_up(t * g.n_qkvg * e)
    # proj_slab mode: q, k, v, o_gated -- nothing else.
    assert slab.proj == -1 and slab.o == -1 and slab.gate == -1
    assert (slab.q, slab.k, slab.v, slab.o_gated) == (0, q_b, q_b + kv_b, q_b + 2 * kv_b)
    assert slab.engine_scratch == slab.total_bytes == q_b + 2 * kv_b + o_b
    # gate-copy mode: the slab comes first, then the same four.
    assert copy.o == -1 and copy.gate == -1
    assert (copy.proj, copy.q, copy.k, copy.v, copy.o_gated) == (0, proj_b, proj_b + q_b, proj_b + q_b + kv_b, proj_b + q_b + 2 * kv_b)
    assert copy.engine_scratch == copy.total_bytes == proj_b + q_b + 2 * kv_b + o_b
    # Against the inference out-of-place layout: same slots minus `o`, plus `o_gated` of the same size.
    assert copy.total_bytes == base.total_bytes and slab.total_bytes == base.total_bytes - proj_b
    for lay in (slab, copy):
        assert lay.total_bytes % _WS_ALIGN == 0
        assert (lay.q8, lay.k8, lay.v8, lay.o8, lay.gate16, lay.sf_q, lay.sf_k, lay.sf_v, lay.o4, lay.sf_o) == (-1,) * 10
    with pytest.raises(ValueError, match="inplace_qkv"):
        _plan_workspace(g, b, s, torch.bfloat16, True, True, True, want_saved=True)
    with pytest.raises(ValueError, match="bf16"):
        _plan_workspace(g, b, s, torch.bfloat16, True, True, False, fp8=True, want_saved=True)
    with pytest.raises(ValueError, match="want_saved"):
        _plan_workspace(g, b, s, torch.bfloat16, True, True, False, saved_gate_copy=True)


@requires_cuda
def test_training_block_declares_the_training_carve_and_the_gate_copy_stage():
    """The declared block's layout is the training carve of its save mode, and the gate-copy block -- and only it --
    builds the ``gate_compaction`` stage (a stage that never runs is not in ``_stages``).  ``saved_gate_copy=True``
    without ``save_for_backward`` names the knob.  Declaration only (CUDA inputs, no compile)."""
    blk_slab, _, _ = _declare(_COMMON, 1, 256, save_mode="proj_slab")
    blk_copy, _, _ = _declare(_COMMON, 1, 256, save_mode="gate_copy")
    assert blk_slab.save_for_backward and not blk_slab.inplace_qkv and blk_slab.saved_gate_copy is False
    assert blk_copy.saved_gate_copy is True
    lay_slab, lay_copy = blk_slab._layout(), blk_copy._layout()
    assert lay_slab.proj == -1 and lay_slab.o == -1 and lay_slab.o_gated >= 0 and lay_slab.q >= 0
    assert lay_copy.proj == 0 and lay_copy.o == -1 and lay_copy.o_gated >= 0
    names_slab = [st.name for st in blk_slab._stages]
    names_copy = [st.name for st in blk_copy._stages]
    assert names_slab == ["qkv_gate_proj", "qk_norm_rope", "compact_v", "sdpa", "sigmoid_gate", "out_proj"], names_slab
    assert names_copy == ["qkv_gate_proj", "qk_norm_rope", "gate_compaction", "compact_v", "sdpa", "sigmoid_gate", "out_proj"], names_copy
    assert blk_copy._gate_copy is not None and blk_copy._gate_copy.heads == blk_copy.geom.h_q and blk_copy._gate_copy.has_gate is False
    assert blk_slab._gate_copy is None
    # Only the constructor is under the `raises`: an error from the input builders must not pass as the pin.
    geom = GatedAttentionBlockGeometry(**_COMMON)
    inp = make_inputs(RefGeometry(**_COMMON), batch=1, seq_len=256, dtype=torch.bfloat16)
    out = torch.empty(1, 256, geom.d_model, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="saved_gate_copy"):
        GatedAttentionBlockFwd(inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, geom, saved_gate_copy=True)


# ---------------------------------------------------------------------------
# The record contracts -- typed, on a declared block, before any launch (any CUDA device)
# ---------------------------------------------------------------------------


@requires_cuda
def test_saved_views_must_alias_proj_slab():
    """In the proj_slab mode ``saved.gate`` / ``q_pre`` / ``k_pre`` are optional, but when given each must be EXACTLY the
    view ``saved_slab_views`` spells (data_ptr, shape AND strides): a fresh tensor, a view of another slab, or the
    ``[T, h, D]`` spelling is a ``ValueError`` naming the alias -- before any launch."""
    b, s = 1, 256
    blk, inp, out = _declare(_COMMON, b, s, save_mode="proj_slab")
    g = blk.geom
    good = _alloc_saved(g, inp, b, s, save_mode="proj_slab")
    bound = blk._check_saved_set(inp["h"], None, None, good)  # the healthy spelling passes
    assert bound.proj.data_ptr() == good.proj_slab.data_ptr() and bound.o.data_ptr() == good.o.data_ptr() and bound.lse is good.lse
    # Every band may be left None (the backward derives it from the slab).
    blk._check_saved_set(inp["h"], None, None, dataclasses.replace(good, gate=None, q_pre=None, k_pre=None))
    fresh = torch.empty(b, s, g.h_q, g.d_head, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="alias"):
        blk._check_saved_set(inp["h"], None, None, dataclasses.replace(good, gate=fresh))
    with pytest.raises(ValueError, match="alias"):
        blk._check_saved_set(inp["h"], None, None, dataclasses.replace(good, q_pre=fresh))
    other_slab = torch.empty_like(good.proj_slab)
    with pytest.raises(ValueError, match="alias"):
        blk._check_saved_set(inp["h"], None, None, dataclasses.replace(good, k_pre=saved_slab_views(other_slab, g, b, s)[2]))
    with pytest.raises(ValueError, match="alias"):  # right bytes, wrong spelling ([T, h, D] instead of [B, S, h, D])
        blk._check_saved_set(inp["h"], None, None, dataclasses.replace(good, gate=good.gate.view(b * s, g.h_q, g.d_head)))
    # The slab itself: element count, dtype, contiguity, device -- each typed.
    with pytest.raises(ValueError, match="proj_slab"):
        blk._check_saved_set(inp["h"], None, None, dataclasses.replace(good, proj_slab=None, gate=None, q_pre=None, k_pre=None))
    with pytest.raises(ValueError, match="proj_slab"):
        blk._check_saved_set(inp["h"], None, None, dataclasses.replace(good, proj_slab=torch.empty(b * s, g.n_qkvg - 64, device="cuda", dtype=torch.bfloat16)))
    with pytest.raises(ValueError, match="proj_slab"):
        blk._check_saved_set(inp["h"], None, None, dataclasses.replace(good, proj_slab=torch.empty(b * s, g.n_qkvg, device="cuda", dtype=torch.float16)))
    with pytest.raises(ValueError, match="proj_slab"):
        blk._check_saved_set(
            inp["h"], None, None, dataclasses.replace(good, proj_slab=torch.empty(b * s, 2 * g.n_qkvg, device="cuda", dtype=torch.bfloat16)[:, ::2])
        )
    if _cc() == _SM107:
        # Through execute() on a compiled block: the refusal lands BEFORE any launch -- the sentinel-filled output,
        # slab and O are untouched.
        blk.check_support()
        blk.compile()
        ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
        bad = dataclasses.replace(_alloc_saved(g, inp, b, s, save_mode="proj_slab", sentinel=1.5e30), gate=fresh)
        out.fill_(1.5e30)
        with pytest.raises(ValueError, match="alias"):
            _execute(blk, inp, out, ws, saved=bad)
        torch.cuda.synchronize()
        assert (out == 1.5e30).all() and (bad.proj_slab == 1.5e30).all() and (bad.o == 1.5e30).all()


@requires_cuda
@_SAVE_MODE
def test_saved_record_shape_and_mode_contracts_are_typed(save_mode):
    """The declared save mode and the record must agree (``proj_slab`` given iff the block was declared
    ``saved_gate_copy=False``), ``saved.o`` is compact ``[B, S, H_q, D]`` in the activation dtype on ``h``'s device,
    ``saved.lse`` is ``[B, H_q, S]`` fp32, ``saved.rstd_q`` / ``rstd_k`` are ``[B, S, H]`` fp32 compact (the norm kernel
    writes them through raw pointer arithmetic), every caller buffer a kernel writes is 16-B aligned (a TMA store or a
    16-B vector store lands on it), and in the gate-copy mode ``saved.gate`` is a compact caller buffer while ``q_pre`` /
    ``k_pre`` are optional -- every miss is a ``ValueError`` naming the field."""
    b, s = 1, 256
    blk, inp, out = _declare(_COMMON, b, s, save_mode=save_mode)
    g = blk.geom
    good = _alloc_saved(g, inp, b, s, save_mode=save_mode, with_pre=True)
    blk._check_saved_set(inp["h"], None, None, good)
    other_mode = "gate_copy" if save_mode == "proj_slab" else "proj_slab"
    with pytest.raises(ValueError, match="saved_gate_copy"):
        blk._check_saved_set(inp["h"], None, None, _alloc_saved(g, inp, b, s, save_mode=other_mode, with_pre=True))
    bad_o = torch.empty(b, s, g.h_q * g.d_head, device="cuda", dtype=torch.bfloat16)  # right bytes, wrong rank
    with pytest.raises(ValueError, match="saved.o"):
        blk._check_saved_set(inp["h"], None, None, dataclasses.replace(good, o=bad_o))
    with pytest.raises(ValueError, match="saved.o"):
        blk._check_saved_set(inp["h"], None, None, dataclasses.replace(good, o=torch.empty(b, s, g.h_q, g.d_head, device="cuda", dtype=torch.float32)))
    with pytest.raises(ValueError, match="saved.o"):
        blk._check_saved_set(inp["h"], None, None, dataclasses.replace(good, o=good.o.transpose(1, 2).contiguous().transpose(1, 2)))
    with pytest.raises(ValueError, match="saved.lse"):
        blk._check_saved_set(inp["h"], None, None, dataclasses.replace(good, lse=torch.empty(b, s, g.h_q, device="cuda", dtype=torch.float32)))
    with pytest.raises(ValueError, match="saved.lse"):
        blk._check_saved_set(inp["h"], None, None, dataclasses.replace(good, lse=good.lse.to(torch.bfloat16)))
    # 16-B alignment of every caller buffer a kernel writes: right count, dtype, device and contiguity, but the base one
    # element into an allocation -> refused, naming the field, before the TMA store / vector store would hit it.
    with pytest.raises(ValueError, match=r"saved\.o.*aligned"):
        blk._check_saved_set(inp["h"], None, None, dataclasses.replace(good, o=_odd_offset((b, s, g.h_q, g.d_head), torch.bfloat16)))
    with pytest.raises(ValueError, match=r"saved\.lse.*aligned"):
        blk._check_saved_set(inp["h"], None, None, dataclasses.replace(good, lse=_odd_offset((b, g.h_q, s), torch.float32)))
    if g.qk_norm:
        with pytest.raises(ValueError, match=r"saved\.rstd_q"):  # right bytes, wrong rank
            blk._check_saved_set(inp["h"], None, None, dataclasses.replace(good, rstd_q=good.rstd_q.view(b * s, g.h_q)))
        with pytest.raises(ValueError, match=r"saved\.rstd_k"):  # wrong dtype: the kernel writes fp32 through a raw pointer
            blk._check_saved_set(inp["h"], None, None, dataclasses.replace(good, rstd_k=good.rstd_k.to(torch.bfloat16)))
        with pytest.raises(ValueError, match=r"saved\.rstd_q.*aligned"):
            blk._check_saved_set(inp["h"], None, None, dataclasses.replace(good, rstd_q=_odd_offset((b, s, g.h_q), torch.float32)))
    if save_mode == "proj_slab":
        bad_slab = _odd_offset((b * s, g.n_qkvg), torch.bfloat16)
        with pytest.raises(ValueError, match=r"saved\.proj_slab.*aligned"):
            blk._check_saved_set(inp["h"], None, None, dataclasses.replace(good, proj_slab=bad_slab, gate=None, q_pre=None, k_pre=None))
    else:
        with pytest.raises(ValueError, match=r"saved\.gate.*aligned"):
            blk._check_saved_set(inp["h"], None, None, dataclasses.replace(good, gate=_odd_offset((b, s, g.h_q, g.d_head), torch.bfloat16)))
        with pytest.raises(ValueError, match=r"saved\.q_pre.*aligned"):
            blk._check_saved_set(inp["h"], None, None, dataclasses.replace(good, q_pre=_odd_offset((b, s, g.h_q, g.d_head), torch.bfloat16)))
    if save_mode == "gate_copy":
        with pytest.raises(ValueError, match="saved.gate"):
            blk._check_saved_set(inp["h"], None, None, dataclasses.replace(good, gate=None))
        with pytest.raises(ValueError, match="saved.gate"):
            blk._check_saved_set(inp["h"], None, None, dataclasses.replace(good, gate=torch.empty(b * s, g.h_q, g.d_head, device="cuda", dtype=torch.bfloat16)))
        with pytest.raises(ValueError, match="saved.k_pre"):
            blk._check_saved_set(inp["h"], None, None, dataclasses.replace(good, k_pre=torch.empty(b, s, g.h_q, g.d_head, device="cuda", dtype=torch.bfloat16)))
        # q_pre / k_pre may be left None: nothing is copied for them.
        bound = blk._check_saved_set(inp["h"], None, None, dataclasses.replace(good, q_pre=None, k_pre=None))
        assert bound.proj is None and bound.q_pre_dst is None and bound.k_pre_dst is None and bound.gate_dst.data_ptr() == good.gate.data_ptr()


@requires_cuda
def test_h_and_lse_identity_are_typed():
    """``saved.h`` must be ``h``'s storage (the backward reads it for the wgrad and the recompute); ``lse`` defaults to
    ``saved.lse`` and, when given, must be that same storage -- both typed, before any launch."""
    b, s = 1, 256
    blk, inp, out = _declare(_COMMON, b, s)
    good = _alloc_saved(blk.geom, inp, b, s, save_mode="proj_slab")
    with pytest.raises(ValueError, match=r"saved\.h"):
        blk._check_saved_set(inp["h"], None, None, dataclasses.replace(good, h=inp["h"].clone()))
    with pytest.raises(ValueError, match="lse"):
        blk._check_saved_set(inp["h"], None, torch.empty_like(good.lse), good)
    assert blk._check_saved_set(inp["h"], None, None, good).lse is good.lse, "lse=None defaults to saved.lse"
    assert blk._check_saved_set(inp["h"], None, good.lse, good).lse is good.lse
    same_storage = good.lse.view(b * blk.geom.h_q, s)  # accepted: same storage; the SDPA still writes saved.lse
    assert blk._check_saved_set(inp["h"], None, same_storage, good).lse is good.lse
    with pytest.raises(ValueError, match=r"saved\.lse"):
        blk._check_saved_set(inp["h"], None, None, dataclasses.replace(good, lse=good.lse.transpose(1, 2).contiguous().transpose(1, 2)))


@requires_cuda
def test_saved_seq_lens_identity_is_typed():
    """``saved.seq_lens`` must be the very tensor ``execute`` runs with (``is``, or both None): the backward's
    declaration-time padding decline rests on this identity, so a record that disagrees is a
    ``ValueError`` naming ``seq_lens`` -- before any launch, and without a device read."""
    b, s = 2, 256
    blk, inp, out = _declare(_COMMON, b, s, seq_lens_present=True)
    lens = torch.tensor([s, 0], device="cuda", dtype=torch.int32)
    g = blk.geom
    with pytest.raises(ValueError, match="seq_lens"):
        blk._check_saved_set(inp["h"], lens, None, _alloc_saved(g, inp, b, s, save_mode="proj_slab", seq_lens=None))
    with pytest.raises(ValueError, match="seq_lens"):
        blk._check_saved_set(inp["h"], lens, None, _alloc_saved(g, inp, b, s, save_mode="proj_slab", seq_lens=lens.clone()))
    with pytest.raises(ValueError, match="seq_lens"):
        blk._check_saved_set(inp["h"], None, None, _alloc_saved(g, inp, b, s, save_mode="proj_slab", seq_lens=lens))
    blk._check_saved_set(inp["h"], None, None, _alloc_saved(g, inp, b, s, save_mode="proj_slab", seq_lens=None))  # both None
    blk._check_saved_set(inp["h"], lens, None, _alloc_saved(g, inp, b, s, save_mode="proj_slab", seq_lens=lens))  # the same object
    if _cc() == _SM107:
        blk.check_support()
        blk.compile()
        ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
        bad = _alloc_saved(g, inp, b, s, save_mode="proj_slab", seq_lens=None, sentinel=1.5e30)
        out.fill_(1.5e30)
        with pytest.raises(ValueError, match="seq_lens"):
            _execute(blk, inp, out, ws, saved=bad, seq_lens=lens)
        torch.cuda.synchronize()
        assert (out == 1.5e30).all() and (bad.proj_slab == 1.5e30).all() and (bad.o == 1.5e30).all()


@requires_cuda
def test_inference_block_refuses_a_saved_record():
    """``saved=`` on a block declared WITHOUT ``save_for_backward`` is a ``ValueError`` naming the argument and the
    declaration knob, before any launch: an inference forward writes none of the record's tensors (no stage targets
    ``saved.o`` / ``lse`` / ``rstd_*`` / ``proj_slab`` / ``gate``), so accepting it would hand the caller an uninitialised
    save set that the backward then consumes -- the same rule as every other ignored argument (``h_sf`` outside MXFP8,
    ``w_o_sf`` outside fp4 O, ``gate`` without ``fuse_gate``).  ``saved=None`` stays the inference default, and the
    training block keeps REQUIRING its record.  Declared block on any CUDA device (``_ws`` set by hand: the argument
    contracts run before the workspace is read); through ``execute()`` on a COMPILED block on Rubin, sentinel-checked."""
    b, s = 1, 256
    geom = GatedAttentionBlockGeometry(**_COMMON)
    inp = make_inputs(RefGeometry(**_COMMON), batch=b, seq_len=s, dtype=torch.bfloat16)
    out = torch.empty(b, s, geom.d_model, device="cuda", dtype=torch.bfloat16)
    infer = GatedAttentionBlockFwd(inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, geom)
    assert not infer.save_for_backward
    record = _alloc_saved(infer.geom, inp, b, s, save_mode="proj_slab", sentinel=1.5e30)
    infer._ws = infer._layout()  # bypass compile: the argument contracts run before the workspace is read (any CUDA device)
    ws16 = torch.empty(16, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="save_for_backward") as ei:
        _execute(infer, inp, out, ws16, saved=record)
    assert "saved=" in str(ei.value)
    torch.cuda.synchronize()
    assert (record.proj_slab == 1.5e30).all() and (record.o == 1.5e30).all() and (record.lse == 1.5e30).all()  # nothing wrote the record
    # The other direction is unchanged: a training block without its record is the existing typed decline.
    train, inp_t, _ = _declare(_COMMON, b, s)
    with pytest.raises(ValueError, match="requires a SavedForBackward"):
        train._check_saved_set(inp_t["h"], None, None, None)
    if _cc() == _SM107:
        # Through execute() on a COMPILED inference block: refused before any launch -- out and the record untouched --
        # and the inference default (saved=None) still runs.
        infer._ws = None
        infer.check_support()
        infer.compile()
        ws = torch.empty(infer.get_workspace_size(), dtype=torch.uint8, device="cuda")
        out.fill_(1.5e30)
        with pytest.raises(ValueError, match="save_for_backward"):
            _execute(infer, inp, out, ws, saved=record)
        torch.cuda.synchronize()
        assert (out == 1.5e30).all() and (record.proj_slab == 1.5e30).all() and (record.o == 1.5e30).all()
        _execute(infer, inp, out, ws, saved=None)
        torch.cuda.synchronize()
        assert torch.isfinite(out.float()).all() and not (out == 1.5e30).any()


# ---------------------------------------------------------------------------
# The accept half -- Rubin only
# ---------------------------------------------------------------------------


def _assert_saved_set_matches_the_oracle(out, ref, blk, saved, inp, *, qk_norm, save_mode, batch, seq_len):
    """The oracle comparison shared by the bf16 matrix and the fp16 arm: every saved tensor against the fp32 oracle's
    copy of it, at the bounds ``test_saved_set_matches_the_oracle`` states."""
    g = blk.geom
    assert torch.isfinite(out.float()).all() and _cos(out, ref.out) > 0.999
    tol = dict(rtol=2**-7, atol=1e-3)
    torch.testing.assert_close(saved.q_pre, ref.q_pre, **tol)
    torch.testing.assert_close(saved.k_pre, ref.k_pre, **tol)
    torch.testing.assert_close(saved.gate, ref.gate, **tol)
    if save_mode == "proj_slab":
        v = saved_slab_views(saved.proj_slab, g, batch, seq_len)[3]
        torch.testing.assert_close(v, ref.v, **tol)
        assert saved.gate.data_ptr() == saved.proj_slab.data_ptr() + g.qkvg_offsets[1] * saved.proj_slab.element_size(), "gate is a view of the slab"
    else:
        assert saved.proj_slab is None and saved.gate.is_contiguous()
    assert torch.isfinite(saved.o.float()).all()
    c = _cos(saved.o, ref.o)
    assert c > 0.999, f"pre-gate O cos {c}"
    torch.testing.assert_close(saved.o.float(), ref.o.float(), rtol=0, atol=2e-2)
    torch.testing.assert_close(saved.lse, ref.lse, rtol=0, atol=2e-2)
    assert blk._norm_rope.want_rstd is qk_norm and (ref.rstd_q is None) is (not qk_norm)
    if qk_norm:
        _, rstd_q_ref = qk_norm_rope_reference(saved.q_pre, inp["w_q_norm"], inp["cos"], inp["sin"], g.rope_dim, g.qk_norm_eps, qk_norm=True)
        _, rstd_k_ref = qk_norm_rope_reference(saved.k_pre, inp["w_k_norm"], inp["cos"], inp["sin"], g.rope_dim, g.qk_norm_eps, qk_norm=True)
        torch.testing.assert_close(saved.rstd_q, rstd_q_ref, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(saved.rstd_k, rstd_k_ref, rtol=1e-5, atol=1e-6)
        # Cross-check against the oracle's OWN rstd, whose q_pre / k_pre come from torch's GEMM (a different fp32
        # accumulation order rounds a few of the 256 elements to the other bf16 neighbour).  Measured on Rubin (cc 10.7,
        # 204 SMs; B=2 S=512, this geometry, rstd in [1.79, 2.78]): max rel 1.88e-4 bf16 / 3.2e-5 fp16, ~1 % of rows
        # above 1e-5 -- so rtol 1e-3 / atol 1e-5 is a 5x envelope of the effect, not a guess; the equal-input bound above is
        # the strict one (max rel 1.2e-7 measured).  Row-to-row rstd varies ~4 %, so this still catches a row-indexing slip.
        torch.testing.assert_close(saved.rstd_q, ref.rstd_q, rtol=1e-3, atol=1e-5)
        torch.testing.assert_close(saved.rstd_k, ref.rstd_k, rtol=1e-3, atol=1e-5)
    else:
        assert saved.rstd_q is None and saved.rstd_k is None


@requires_rubin
@_QK_NORM
@_SAVE_MODE
def test_saved_set_matches_the_oracle(qk_norm, save_mode):
    """Every saved tensor against the fp32 oracle's copy of it: the four stage-(1) bands at one bf16 rounding of a
    fp32-accumulated GEMM (``rtol 2**-7, atol 1e-3``), pre-gate ``O`` and ``LSE`` at the SDPA stage test's bounds
    (``cos >= 0.999`` / ``atol 2e-2``), ``rstd`` at the norm test's bound (``rtol 1e-5, atol 1e-6`` against the oracle
    norm of the block's OWN ``q_pre`` -- the kernel and the oracle round the GEMM differently, so the strict bound is
    stated on equal inputs -- plus a MEASURED-envelope cross-check against the oracle's rstd).  Gate-copy mode passes
    ``q_pre`` / ``k_pre`` buffers so the copies are checked too."""
    geom_kw = {**_COMMON, "qk_norm": qk_norm}
    out, ref, blk, saved, inp = _run_training(geom_kw, batch=2, seq_len=512, save_mode=save_mode, with_pre=True)
    _assert_saved_set_matches_the_oracle(out, ref, blk, saved, inp, qk_norm=qk_norm, save_mode=save_mode, batch=2, seq_len=512)


@requires_rubin
@_SAVE_MODE
def test_saved_set_matches_the_oracle_in_fp16(save_mode):
    """The fp16 activation arm (``act_dtype`` keys every dtype check of the record contract and every kernel artifact):
    the same oracle comparison at the same bounds, both save modes, ``qk_norm`` on.  Measured on Rubin (cc 10.7, 204 SMs)
    before the bounds were adopted for fp16: bands max|diff| 9.8e-4 (one fp16 ulp at this magnitude), O 9.8e-4, LSE 1.9e-4."""
    out, ref, blk, saved, inp = _run_training({**_COMMON, "qk_norm": True}, batch=2, seq_len=512, save_mode=save_mode, with_pre=True, dtype=torch.float16)
    assert out.dtype == saved.o.dtype == saved.gate.dtype == saved.q_pre.dtype == saved.k_pre.dtype == torch.float16
    _assert_saved_set_matches_the_oracle(out, ref, blk, saved, inp, qk_norm=True, save_mode=save_mode, batch=2, seq_len=512)


@requires_rubin
@_SAVE_MODE
def test_out_is_bitwise_the_inference_block(save_mode):
    """Same kernels, different buffers: the training forward's ``out`` equals the out-of-place inference block's
    (``inplace_qkv=False``, no LSE) bit for bit -- the GEMM writes the caller's slab instead of the workspace one, the
    SDPA writes ``saved.o`` (and its LSE), the gate lands in the workspace ``o_gated`` instead of in place."""
    b, s = 2, 512
    out_train, _, _, saved, inp = _run_training(_COMMON, b, s, save_mode=save_mode)
    geom = GatedAttentionBlockGeometry(**_COMMON)
    out_infer = torch.zeros_like(out_train)
    infer = GatedAttentionBlockFwd(
        inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out_infer, geom, inplace_qkv=False
    )
    assert not infer.save_for_backward and not infer.return_lse and not infer.inplace_qkv
    infer.check_support()
    infer.compile()
    ws = torch.empty(infer.get_workspace_size(), dtype=torch.uint8, device="cuda")
    infer.execute(inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out_infer, ws)
    torch.cuda.synchronize()
    assert out_infer.abs().max().item() > 0
    assert torch.equal(out_train, out_infer), (out_train.float() - out_infer.float()).abs().max().item()


@requires_rubin
def test_gate_copy_mode_populates_gate_only(monkeypatch):
    """``saved.proj_slab is None`` (``saved_gate_copy=True``): the compact ``saved.gate`` equals the proj_slab run's GATE
    band BIT for bit (a copy, not a recompute); ``q_pre`` / ``k_pre`` are copied ONLY when the caller passed buffers --
    pinned by COUNTING the elementwise launches (proj_slab mode: V compaction + sigmoid gate; gate-copy: + the GATE
    copy into ``saved.gate`` and nothing else; with ``q_pre`` / ``k_pre`` buffers: + one copy each, landing in those
    buffers) -- and are the slab bands bitwise when they are; the block's stage list carries exactly one extra stage."""
    from cudnn.gated_attention_block.kernels import elementwise as ew

    launches = []  # (heads, has_gate, dst data_ptr) per elementwise launch; `_ElementwiseStage._run` imports the runner per call
    real_run = ew.run_elementwise_gate

    def counting_run(r, src, gate, dst, *, stream):
        launches.append((int(r.h), bool(r.has_gate), dst.data_ptr()))
        return real_run(r, src, gate, dst, stream=stream)

    monkeypatch.setattr(ew, "run_elementwise_gate", counting_run)
    b, s = 2, 512
    _, _, blk_slab, saved_slab, _ = _run_training(_COMMON, b, s, save_mode="proj_slab")
    slab_launches, launches[:] = list(launches), []
    out_gc, _, blk_gc, saved_gc, _ = _run_training(_COMMON, b, s, save_mode="gate_copy", with_pre=False)
    gc_launches, launches[:] = list(launches), []
    g = blk_gc.geom
    assert [(h, hg) for h, hg, _ in slab_launches] == [(g.h_kv, False), (g.h_q, True)], slab_launches  # (3b) V compaction, (5) gate
    assert [(h, hg) for h, hg, _ in gc_launches] == [(g.h_q, False), (g.h_kv, False), (g.h_q, True)], gc_launches  # + (3g) GATE copy only
    assert gc_launches[0][2] == saved_gc.gate.data_ptr(), "the extra launch is not the GATE copy into saved.gate"
    assert saved_gc.proj_slab is None
    assert saved_gc.gate.is_contiguous() and torch.equal(saved_gc.gate, saved_slab.gate), "the GATE copy is not the slab band"
    assert torch.equal(saved_gc.o, saved_slab.o) and torch.equal(saved_gc.lse, saved_slab.lse)
    assert len(blk_gc._stages) == len(blk_slab._stages) + 1 and "gate_compaction" in [st.name for st in blk_gc._stages]
    out_pre, _, _, saved_pre, _ = _run_training(_COMMON, b, s, save_mode="gate_copy", with_pre=True, sentinel=1.5e30)
    pre_launches = list(launches)
    # (3g) GATE copy, q_pre copy (h_q recipe), k_pre copy (the V-compaction h_kv recipe), (3b) V compaction, (5) gate.
    assert [(h, hg) for h, hg, _ in pre_launches] == [(g.h_q, False), (g.h_q, False), (g.h_kv, False), (g.h_kv, False), (g.h_q, True)], pre_launches
    assert [d for _, _, d in pre_launches[:3]] == [saved_pre.gate.data_ptr(), saved_pre.q_pre.data_ptr(), saved_pre.k_pre.data_ptr()]
    assert torch.equal(saved_pre.gate, saved_slab.gate)
    assert torch.equal(saved_pre.q_pre, saved_slab.q_pre) and torch.equal(saved_pre.k_pre, saved_slab.k_pre), "q_pre / k_pre copies are not the slab bands"
    assert not (saved_pre.q_pre == 1.5e30).any() and not (saved_pre.k_pre == 1.5e30).any()
    assert torch.equal(out_pre, out_gc)


@requires_rubin
def test_training_forward_dead_entry_o_is_exactly_zero():
    """``seq_lens = [s, 0]``: the dead entry's saved ``O`` is EXACTLY zero (a SELECT in the SDPA epilogue, not residue),
    its ``LSE`` is ``-inf`` (no floored ``-69.08``), and ``out[1] == 0`` -- the SDPA's empty-range contract (O = 0 by SELECT,
    LSE = -inf, no denominator floor reaching either) on the SAVED set, asserted on the outputs directly (a diff against a
    NaN reference proves nothing).  ``saved.seq_lens`` is the very tensor the forward ran with."""
    b, s = 2, 512
    lens = torch.tensor([s, 0], device="cuda", dtype=torch.int32)
    out, ref, blk, saved, _ = _run_training(_COMMON, b, s, save_mode="proj_slab", seq_lens=lens, sentinel=1.5e30)
    assert saved.seq_lens is lens
    assert torch.isfinite(saved.o.float()).all() and torch.isfinite(out.float()).all()
    assert (saved.o[1] == 0).all(), f"dead entry O max|.| = {saved.o[1].abs().max().item()}"
    assert torch.isneginf(saved.lse[1]).all(), f"dead entry LSE {saved.lse[1].flatten()[:4].tolist()} (expected -inf everywhere)"
    assert (out[1] == 0).all(), f"dead entry out max|.| = {out[1].abs().max().item()}"
    assert not (saved.o[0] == 1.5e30).any() and torch.isfinite(saved.lse[0]).all(), "the live entry was not written"
    assert _cos(saved.o[0], ref.o[0]) > 0.999 and _cos(out[0], ref.out[0]) > 0.999
    torch.testing.assert_close(saved.lse[0], ref.lse[0], rtol=0, atol=2e-2)
