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

The QUANTIZED training forward (per-tensor FP8 / MXFP8, unfused; the last section) writes the SAME record with the
same kernels as quantized inference, one routing apart: norm+RoPE lands in compact bf16 workspace slots instead of in
place, so the slab keeps the PRE-norm Q/K bands the backward differentiates.  Pinned host-side (the routing, RED
without it) and on Rubin (``out`` / ``O`` / ``LSE`` / the GATE and V bands bitwise the inference block; the bands
against the dequantized GEMM; the forward's own norm + quantize replayed over the record's bands reproduce the
workspace ``q8`` / ``k8`` bitwise; the LSE within 1e-4 of the fp64 log-sum-exp over the operands the SDPA read).

Accept tests are ``requires_rubin`` (the block targets SM107 only); the reject
tests build CUDA tensors for a DECLARED block (``requires_cuda``: any CUDA
device, no compile); the pure-layout tests (``saved_slab_views``,
``_plan_workspace``) run anywhere, CPU included.
"""

import dataclasses
import gc
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

from cudnn.gated_attention_block import GatedAttentionBlockFwd, GatedAttentionBlockGeometry, MxQuantSpec, SavedForBackward, saved_slab_views  # noqa: E402
from cudnn.gated_attention_block.api import _plan_workspace, _sf_slot_bytes, _view  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_reference import (  # noqa: E402
    RefGeometry,
    dequant_e4m3,
    gated_attention_block_fp8_reference,
    gated_attention_block_mxfp8_reference,
    gated_attention_block_reference,
    make_inputs,
    mx_dequant_rowwise_2d,
    mx_fake_quant_rowwise,
    mx_fake_quant_v_columnwise,
    mxfp8_calibrated_scale_o,
    qk_norm_rope_reference,
    quantize_block_inputs,
    quantize_block_inputs_mxfp8,
)
from test_block_fp8 import _calibrated_spec as _fp8_calibrated_spec  # noqa: E402

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


def _alloc_saved(geom, inp, batch, seq_len, *, save_mode, with_pre=False, seq_lens=None, sentinel=None, act_dtype=None):
    """A caller-owned ``SavedForBackward`` for ``save_mode`` (``inp["h"]`` IS ``saved.h``).

    ``with_pre`` (gate-copy mode only): also pass compact ``q_pre`` / ``k_pre`` buffers, so the block copies them.
    ``sentinel``: fill every caller buffer with it, so an untouched cell is recognisable.
    ``act_dtype`` (appended): the ACTIVATION dtype of the record's slab / O / gate buffers -- ``inp["h"].dtype`` by default
    (bf16 / fp16); ``torch.bfloat16`` for a QUANTIZED forward, whose ``h`` is e4m3 codes while every activation is bf16."""
    b, s, g = batch, seq_len, geom
    dtype, dev = (inp["h"].dtype if act_dtype is None else act_dtype), inp["h"].device

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
    256-B aligned; the training layout that disagrees with its body (in place, or a FULLY FUSED quantized pipeline) is a
    typed ``ValueError`` rather than a silent carve.  No GPU."""
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
    with pytest.raises(ValueError, match="UNFUSED"):
        _plan_workspace(g, b, s, torch.bfloat16, True, True, False, fp8=True, fp8_fused=True, want_saved=True)
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


# ---------------------------------------------------------------------------
# The QUANTIZED training forward (per-tensor FP8 / MXFP8, unfused): the bf16 record, one routing apart
# ---------------------------------------------------------------------------
#
# The quantized INFERENCE pipelines norm Q/K IN PLACE on the bf16 slab.  A training record needs the slab's Q/K bands
# PRE-norm (the backward recomputes Q/K from them with the forward's own kernel and differentiates the norm over them), so
# the training forward routes norm+RoPE OUT of place into compact bf16 workspace slots and the quantize stages read those.
# Everything else is the inference chain: `out`, O, LSE and the untouched GATE / V bands are bitwise the inference block's.
# Lifting the training decline WITHOUT that routing hands the backward POST-norm bands -- wrong dQ-side and norm-weight
# gradients, no crash, `out` still bitwise -- which is why the routing has a host-level pin of its own (RED without it).

_FAMILY = pytest.mark.parametrize("family", ["fp8", "mxfp8"])
_E4M3 = torch.float8_e4m3fn
_FP8_STAGES = ["qkv_gate_proj", "qk_norm_rope", "quantize_q", "quantize_kv", "sdpa", "sigmoid_gate", "out_proj"]
_MX_STAGES = ["qkv_gate_proj", "qk_norm_rope", "quantize_mxfp8_q", "quantize_mxfp8_k", "quantize_mxfp8_v", "sdpa", "sigmoid_gate", "quantize_o", "out_proj"]
# The quantized training forward's accept geometry, crossed with ``_QK_NORM`` (the norm kernel's ``apply_norm`` trace and
# whether rstd is written are the axis that changes what the out-of-place norm writes): S in {256, 512, 992, 1024} -- 992 is
# ``S % 128 != 0`` (the causal tail tile and, under MXFP8, the SF-pad arm); a DENSE 992 is the quantized SDPA rows' typed
# ``S % 128`` decline, pinned in the cell -- plus S = 1000 causal (``S % 32 != 0``), B in {1, 2}, GQA 8/2 and MHA.
_QUANT_GEOMS = pytest.mark.parametrize(
    "seq_len, causal, batch, h_kv",
    [(256, True, 1, 2), (512, True, 2, 2), (992, True, 1, 2), (992, False, 2, 2), (1000, True, 2, 2), (1024, False, 1, 8)],
    ids=["s256_causal_b1", "s512_causal_b2", "s992_causal_b1", "s992_dense_b2", "s1000_causal_b2", "s1024_dense_b1_mha"],
)


def _quant_geom(qk_norm, causal, h_kv):
    """The geometry kwargs of one quantized training cell: ``_COMMON`` with the three axes the cells cross -- ``h_kv``, ``qk_norm``,
    ``is_causal`` -- overridden."""
    return {**_COMMON, "h_kv": h_kv, "qk_norm": qk_norm, "is_causal": causal}


def _dense_tail_declined(geom_kw, batch, seq_len, family):
    """A DENSE ``S % 128 != 0`` is the quantized SDPA rows' typed decline (no padding mask and no causal mask covering the KV
    tail), on the training forward exactly as on inference (``test_fp8_dense_kv_tail_is_declined_not_computed_wrong`` and
    its MXFP8 twin): pinned at ``check_support``, before any launch."""
    with pytest.raises((ValueError, NotImplementedError), match="multiple of 128"):
        _run_training_quant(geom_kw, batch, seq_len, family)


def _quant_inputs(geom_kw, batch, seq_len, family):
    """bf16 inputs, their quantized twin (e4m3 codes, plus the F8_128x4 blobs under MXFP8), the QuantSpec / MxQuantSpec calibrated the way the
    fp8 / mxfp8 forward suites calibrate it, and the block's extra declaration kwargs."""
    rg = RefGeometry(**geom_kw)
    geom = GatedAttentionBlockGeometry(**geom_kw)
    inp16 = make_inputs(rg, batch=batch, seq_len=seq_len, dtype=torch.bfloat16)
    if family == "fp8":
        inp_q, desc = quantize_block_inputs(inp16)
        return inp16, inp_q, _fp8_calibrated_spec(inp_q, desc, geom, batch, seq_len), {}
    assert family == "mxfp8"
    inp_q, desc = quantize_block_inputs_mxfp8(inp16)
    spec = MxQuantSpec(**desc, scale_o=mxfp8_calibrated_scale_o(inp_q, rg))
    return inp16, inp_q, spec, dict(sample_h_sf=inp_q["h_sf"], sample_w_qkvg_sf=inp_q["w_qkvg_sf"])


def _dequantized_bf16_inputs(inp_q, spec, family):
    """The bf16 operands a quantized record REPRESENTS -- ``h`` / ``W_qkvg`` / ``W_o`` dequantized (per tensor, or through
    the MXFP8 blobs, where e4m3 codes times a power of two are exact in bf16) -- what the bf16 backward is handed with it."""
    b, s, dm = inp_q["h"].shape
    out = {k: v for k, v in inp_q.items() if k not in ("h_sf", "w_qkvg_sf")}
    if family == "fp8":
        out["h"] = dequant_e4m3(inp_q["h"], spec.descale_h).to(torch.bfloat16)
        out["w_qkvg"] = dequant_e4m3(inp_q["w_qkvg"], spec.descale_w_qkvg).to(torch.bfloat16)
    else:
        out["h"] = mx_dequant_rowwise_2d(inp_q["h"].reshape(b * s, dm), inp_q["h_sf"]).reshape(b, s, dm).to(torch.bfloat16)
        out["w_qkvg"] = mx_dequant_rowwise_2d(inp_q["w_qkvg"], inp_q["w_qkvg_sf"]).to(torch.bfloat16)
    out["w_o"] = dequant_e4m3(inp_q["w_o"], spec.descale_w_o).to(torch.bfloat16)
    return out


def _dequantized_fp64_proj(inp_q, spec, family):
    """Stage (1) in fp64 from the EXACT dequantized operands (no bf16 rounding of h / W): the ``[T, N]`` product the
    forward's GEMM rounds to the bf16 slab once."""
    b, s, dm = inp_q["h"].shape
    if family == "fp8":
        h64 = dequant_e4m3(inp_q["h"], spec.descale_h).double().reshape(b * s, dm)
        w64 = dequant_e4m3(inp_q["w_qkvg"], spec.descale_w_qkvg).double()
    else:
        h64 = mx_dequant_rowwise_2d(inp_q["h"].reshape(b * s, dm), inp_q["h_sf"]).double()
        w64 = mx_dequant_rowwise_2d(inp_q["w_qkvg"], inp_q["w_qkvg_sf"]).double()
    return h64 @ w64.t()


def _declare_quant(geom_kw, batch, seq_len, family, *, save_mode="proj_slab", training=True, seq_lens_present=False, scale_o=None, **blk_kw):
    """A DECLARED (not compiled) quantized block -- a TRAINING one by default -- with its inputs, spec and output buffer.
    ``scale_o`` overrides the calibrated spec's (the fully fused MXFP8 contract pins ``scale_o == 1.0``, checked before any training guard)."""
    inp16, inp_q, spec, extra = _quant_inputs(geom_kw, batch, seq_len, family)
    if scale_o is not None:
        spec = dataclasses.replace(spec, scale_o=float(scale_o))
    geom = GatedAttentionBlockGeometry(**geom_kw)
    out = torch.empty(batch, seq_len, geom.d_model, device="cuda", dtype=torch.bfloat16)
    kw = dict(quant=spec, seq_lens_present=seq_lens_present, **extra, **blk_kw)
    if training:
        kw.update(save_for_backward=True, saved_gate_copy=(save_mode == "gate_copy"))
    blk = GatedAttentionBlockFwd(inp_q["h"], inp_q["w_qkvg"], inp_q["w_q_norm"], inp_q["w_k_norm"], inp_q["cos"], inp_q["sin"], inp_q["w_o"], out, geom, **kw)
    return SimpleNamespace(blk=blk, inp16=inp16, inp=inp_q, spec=spec, out=out, family=family, geom=geom, geom_kw=geom_kw, batch=batch, seq_len=seq_len)


def _execute_quant(r, ws, *, saved=None, seq_lens=None, lse=None, out=None, blk=None):
    """One ``execute`` of ``r``'s quantized block (or ``blk``) on ``r``'s inputs and the workspace ``ws``, with the MXFP8
    scale-factor blobs when the family needs them; ``out`` defaults to ``r.out``, ``saved`` / ``seq_lens`` / ``lse`` pass through."""
    inp = r.inp
    sf = dict(h_sf=inp["h_sf"], w_qkvg_sf=inp["w_qkvg_sf"]) if r.family == "mxfp8" else {}
    (r.blk if blk is None else blk).execute(
        inp["h"],
        inp["w_qkvg"],
        inp["w_q_norm"],
        inp["w_k_norm"],
        inp["cos"],
        inp["sin"],
        inp["w_o"],
        r.out if out is None else out,
        ws,
        seq_lens=seq_lens,
        lse=lse,
        saved=saved,
        **sf,
    )


def _run_training_quant(geom_kw, batch, seq_len, family, *, save_mode="proj_slab", with_pre=False, sentinel=None):
    """Declare, compile and run the quantized TRAINING forward; the record and the workspace ride on the returned namespace."""
    r = _declare_quant(geom_kw, batch, seq_len, family, save_mode=save_mode)
    r.saved = _alloc_saved(r.geom, r.inp, batch, seq_len, save_mode=save_mode, with_pre=with_pre, sentinel=sentinel, act_dtype=torch.bfloat16)
    r.blk.check_support()
    r.blk.compile()
    r.ws = torch.empty(r.blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    _execute_quant(r, r.ws, saved=r.saved)
    torch.cuda.synchronize()
    return r


def _run_inference_quant(r):
    """The quantized INFERENCE block on the same inputs and spec (in place, the inference default), its LSE requested and
    its PRE-gate O caught before stage (5) gates it in place; returns the output, that O, the LSE and the workspace slab."""
    inp, geom, b, s = r.inp, r.geom, r.batch, r.seq_len
    out = torch.zeros_like(r.out)
    extra = dict(sample_h_sf=inp["h_sf"], sample_w_qkvg_sf=inp["w_qkvg_sf"]) if r.family == "mxfp8" else {}
    infer = GatedAttentionBlockFwd(
        inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, geom, quant=r.spec, return_lse=True, **extra
    )
    assert not infer.save_for_backward and infer.inplace_qkv and infer.return_lse
    infer.check_support()
    infer.compile()
    ws = torch.empty(infer.get_workspace_size(), dtype=torch.uint8, device="cuda")
    lse = torch.empty(b, geom.h_q, s, dtype=torch.float32, device="cuda")
    caught = {}
    real_gate = infer._gate.execute

    def gate_catching_o(o, gate, dst, current_stream=None):
        """Stands in for stage (5)'s ``execute``: clones the PRE-gate ``O`` into ``caught`` before the real gate overwrites it in place."""
        caught["o_pre"] = o.clone()  # the block launches on torch's current stream here, so the clone is ordered after the SDPA
        return real_gate(o, gate, dst, current_stream=current_stream)

    infer._gate.execute = gate_catching_o
    _execute_quant(r, ws, lse=lse, out=out, blk=infer)
    torch.cuda.synchronize()
    slab = _view(ws, infer._layout().proj, (b * s, geom.n_qkvg), torch.bfloat16)
    return SimpleNamespace(out=out, o_pre=caught["o_pre"].view(b, s, geom.h_q, geom.d_head), lse=lse, slab=slab, blk=infer, ws=ws)


def _replay_norm(r, q_band, k_band):
    """The forward's OWN norm+RoPE stage over two PRE-norm bands, out of place into fresh compact buffers (+ fresh rstd):
    bitwise what the forward wrote, so the record's bands can be mapped onto the normed values exactly."""
    g, t = r.geom, r.batch * r.seq_len
    nq = torch.empty(t, g.h_q, g.d_head, dtype=torch.bfloat16, device="cuda")
    nk = torch.empty(t, g.h_kv, g.d_head, dtype=torch.bfloat16, device="cuda")
    rq = torch.empty(t, g.h_q, dtype=torch.float32, device="cuda") if g.qk_norm else None
    rk = torch.empty(t, g.h_kv, dtype=torch.float32, device="cuda") if g.qk_norm else None
    r.blk._norm_rope.execute(q_band, k_band, r.inp["w_q_norm"], r.inp["w_k_norm"], r.inp["cos"], r.inp["sin"], q_out=nq, k_out=nk, rstd_q=rq, rstd_k=rk)
    torch.cuda.synchronize()
    return nq, nk, rq, rk


def _sdpa_operands_fp64(r):
    """The operands the quantized SDPA READ, dequantized in fp64: per-tensor FP8 -> the workspace ``q8`` / ``k8`` / ``v8``
    (the codes bitwise) over the descales; MXFP8 -> the torch MX quantizer (bit-exact vs the kernel's,
    ``test_quantize_mxfp8``) over the forward's own normed Q/K (kernel replay over the record's bands) and the slab's V."""
    g, b, s = r.geom, r.batch, r.seq_len
    t, d = b * s, g.d_head
    lay = r.blk._layout()
    if r.family == "fp8":
        q = _view(r.ws, lay.q8, (t, g.h_q, d), _E4M3).float() / r.spec.scale_q
        k = _view(r.ws, lay.k8, (t, g.h_kv, d), _E4M3).float() / r.spec.scale_k
        v = _view(r.ws, lay.v8, (t, g.h_kv, d), _E4M3).float() / r.spec.scale_v
        return q.double().view(b, s, g.h_q, d), k.double().view(b, s, g.h_kv, d), v.double().view(b, s, g.h_kv, d)
    q_pre, _gate, k_pre, v = saved_slab_views(r.saved.proj_slab, g, b, s)
    nq, nk, _, _ = _replay_norm(r, q_pre, k_pre)
    return (
        mx_fake_quant_rowwise(nq.view(b, s, g.h_q, d)).double(),
        mx_fake_quant_rowwise(nk.view(b, s, g.h_kv, d)).double(),
        mx_fake_quant_v_columnwise(v).double(),
    )


def _attention_fp64(q, k, v, geom):
    """fp64 attention over ``[B, S, H, D]`` operands with the GQA broadcast and the geometry's causal mask:
    ``(O [B, S, H_q, D], LSE [B, H_q, S])`` -- the exact log-sum-exp the SDPA publishes (``has_lse``)."""
    b, s, hq, _ = q.shape
    rep = hq // k.shape[2]
    qb = q.transpose(1, 2)
    kb = k.transpose(1, 2).repeat_interleave(rep, 1)
    vb = v.transpose(1, 2).repeat_interleave(rep, 1)
    scores = torch.matmul(qb, kb.transpose(-1, -2)) * geom.scale
    if geom.is_causal:
        scores = scores.masked_fill(~torch.tril(torch.ones(s, s, dtype=torch.bool, device=q.device)), float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    p = torch.exp(scores - lse[..., None])
    return torch.matmul(p, vb).transpose(1, 2), lse


def _quant_oracle_out(r):
    """The family's fake-quant fp32 oracle of ``out`` (the fp8 / mxfp8 forward suites' oracle)."""
    rg = RefGeometry(**r.geom_kw)
    if r.family == "fp8":
        sp = r.spec
        return gated_attention_block_fp8_reference(
            r.inp,
            rg,
            descale_h=sp.descale_h,
            descale_w_qkvg=sp.descale_w_qkvg,
            descale_w_o=sp.descale_w_o,
            scale_q=sp.scale_q,
            scale_k=sp.scale_k,
            scale_v=sp.scale_v,
            scale_o=sp.scale_o,
        )
    return gated_attention_block_mxfp8_reference(r.inp, rg, descale_w_o=r.spec.descale_w_o, scale_o=r.spec.scale_o)


@pytest.mark.parametrize("shape", list(_SNAPSHOT_SHAPES))
@_FAMILY
def test_workspace_layout_under_want_saved_quantized(shape, family):
    """``want_saved`` on the quantized layouts: the bf16 compact ``q`` / ``k`` are reserved (the out-of-place norm writes
    them, the quantize stages read them; no ``v`` -- never normed), ``o`` is gone (the SDPA writes ``saved.o``), ``o_gated``
    is reserved, the e4m3 slots and the MXFP8 SF blobs stay; ``proj`` is gone in the proj_slab mode and first in the
    gate-copy mode.  Slots follow stage order and stay 256-B aligned; against the INFERENCE carve (byte-identical to the
    frozen snapshot, which never asked for a training forward) the only growth is the compact normed Q/K -- 17408 B/token at
    the 397B geometry.  The fully fused quantized and the fp4-O training carves are typed ``ValueError``s.  No GPU."""
    from cudnn.gated_attention_block.api import _WS_ALIGN, _align_up

    geom_kw, b, s = _SNAPSHOT_SHAPES[shape]
    g = GatedAttentionBlockGeometry(**geom_kw)
    t, e, d = b * s, 2, g.d_head
    mx = family == "mxfp8"
    kw = dict(fp8=True, mxfp8=mx)
    infer = _plan_workspace(g, b, s, torch.bfloat16, False, False, True, **kw)
    slab = _plan_workspace(g, b, s, torch.bfloat16, True, True, False, want_saved=True, **kw)
    copy = _plan_workspace(g, b, s, torch.bfloat16, True, True, False, want_saved=True, saved_gate_copy=True, **kw)
    q_b, kv_b, o_b = _align_up(t * g.h_q * d * e), _align_up(t * g.h_kv * d * e), _align_up(t * g.h_q * d * e)
    q8_b, kv8_b, o8_b, proj_b = _align_up(t * g.h_q * d), _align_up(t * g.h_kv * d), _align_up(t * g.h_q * d), _align_up(t * g.n_qkvg * e)
    # Inference: no compact bf16 Q/K (the norm is in place), the slab and O in the workspace -- as before.
    assert infer.q == infer.k == infer.v == -1 and infer.proj == 0 and infer.o >= 0 and infer.o_gated == -1
    # proj_slab mode, stage order: q, k (bf16 normed), q8, k8, v8, o_gated, o8 [, sf_q, sf_k, sf_v]; no slab, no o, no v.
    assert slab.proj == -1 and slab.o == -1 and slab.v == -1 and slab.gate == -1 and slab.gate16 == -1
    assert (slab.q, slab.k) == (0, q_b)
    assert (slab.q8, slab.k8, slab.v8) == (q_b + kv_b, q_b + kv_b + q8_b, q_b + kv_b + q8_b + kv8_b)
    assert slab.o_gated == slab.v8 + kv8_b and slab.o8 == slab.o_gated + o_b
    end = slab.o8 + o8_b
    if mx:
        sfq, sfkv = _align_up(_sf_slot_bytes(b, g.h_q, s, d)), _align_up(_sf_slot_bytes(b, g.h_kv, s, d))
        assert (slab.sf_q, slab.sf_k, slab.sf_v) == (end, end + sfq, end + sfq + sfkv)
        end += sfq + 2 * sfkv
    else:
        assert (slab.sf_q, slab.sf_k, slab.sf_v) == (-1, -1, -1)
    assert slab.engine_scratch == slab.total_bytes == end and end % _WS_ALIGN == 0
    # gate-copy mode: the slab first, then the same slots shifted by it.
    assert copy.proj == 0 and copy.o == -1 and copy.v == -1
    assert (copy.q, copy.k, copy.q8, copy.k8, copy.v8, copy.o_gated, copy.o8) == tuple(
        x + proj_b for x in (slab.q, slab.k, slab.q8, slab.k8, slab.v8, slab.o_gated, slab.o8)
    )
    assert copy.total_bytes == slab.total_bytes + proj_b
    # Against the inference carve: `o` -> `o_gated` is the same size, the slab moves to the record (proj_slab mode), and the
    # compact normed Q/K are the ONLY growth -- 17 KiB/token at the 397B geometry.
    assert slab.total_bytes == infer.total_bytes - proj_b + q_b + kv_b
    assert copy.total_bytes == infer.total_bytes + q_b + kv_b
    assert (q_b + kv_b) // t == (g.h_q + g.h_kv) * d * e == (17408 if shape == "397b" else 5120)
    with pytest.raises(ValueError, match="UNFUSED"):
        _plan_workspace(g, b, s, torch.bfloat16, True, True, False, fp8=True, fp8_fused=True, mxfp8=mx, want_saved=True)
    with pytest.raises(ValueError, match="inplace_qkv"):
        _plan_workspace(g, b, s, torch.bfloat16, True, True, True, want_saved=True, **kw)
    if mx:
        from cudnn.gated_attention_block import Fp4Format

        with pytest.raises(ValueError, match="fp4"):
            _plan_workspace(g, b, s, torch.bfloat16, True, True, False, fp8=True, mxfp8=True, o_fp4=Fp4Format.MXFP4, want_saved=True)


@requires_cuda
@_FAMILY
def test_quantized_training_block_routes_the_norm_out_of_place_into_the_compact_slots(family, monkeypatch):
    """The pre-norm-band hazard, pinned where it lives -- the execute body -- with every stage's launch replaced by a
    recorder (no kernel runs, any CUDA device).  The quantized training block keeps the inference stage list and carves the
    compact bf16 Q/K; its execute hands the record's slab to the GEMM, the slab's Q/K bands to norm+RoPE with ``q_out`` /
    ``k_out`` = the compact slots (RED on the in-place routing: ``q_out=None``), the compact NORMED Q/K to the Q / K quantize
    stages and the slab's V band to the V one, ``saved.o`` / ``saved.lse`` to the SDPA, and gates ``saved.o`` OUT of place
    into the workspace ``o_gated`` the O quantize then reads -- nine stage launches, the inference count."""
    b, s = 1, 256
    r = _declare_quant(_COMMON, b, s, family)
    blk, g = r.blk, r.geom
    infer = _declare_quant(_COMMON, b, s, family, training=False).blk
    assert blk.save_for_backward and blk.return_lse and not blk.inplace_qkv and not blk.quant_fused
    assert [st.name for st in blk._stages] == [st.name for st in infer._stages] == (_FP8_STAGES if family == "fp8" else _MX_STAGES)
    lay = blk._layout()
    assert lay.q >= 0 and lay.k >= 0 and lay.v == -1 and lay.proj == -1 and lay.o == -1 and lay.o_gated >= 0
    assert lay.q8 >= 0 and lay.k8 >= 0 and lay.v8 >= 0 and lay.o8 >= 0 and (lay.sf_q >= 0) == (family == "mxfp8")
    assert infer._layout().q == -1 and infer._layout().k == -1, "the inference carve must not grow"
    saved = _alloc_saved(g, r.inp, b, s, save_mode="proj_slab", act_dtype=torch.bfloat16)
    blk._ws = lay
    blk._is_supported = True
    monkeypatch.setattr(blk, "get_workspace_size", lambda: lay.total_bytes)
    blk._quant_dev = blk._make_quant_dev()
    calls = []

    def recorder(name):
        """A stand-in for stage ``name``'s ``execute`` that appends ``(name, args, kwargs)`` to ``calls`` instead of launching anything."""
        return lambda *a, **k: calls.append((name, a, k))

    for st in blk._stages:
        monkeypatch.setattr(st, "execute", recorder(st.name))
    ws = torch.zeros(lay.total_bytes, dtype=torch.uint8, device="cuda")
    _execute_quant(r, ws, saved=saved)
    names = [n for n, _, _ in calls]
    # 9 stage launches either way: under FP8 `quantize_q` serves Q and (as quantize_o) the gated O, `quantize_kv` K and V.
    if family == "fp8":
        assert names == ["qkv_gate_proj", "qk_norm_rope", "quantize_q", "quantize_kv", "quantize_kv", "sdpa", "sigmoid_gate", "quantize_q", "out_proj"], names
    else:
        assert names == _MX_STAGES, names
    base, e = ws.data_ptr(), 2
    q_slot, k_slot = base + lay.q, base + lay.k
    o_q, _o_g, o_k, o_v = g.qkvg_offsets
    slab_ptr = saved.proj_slab.data_ptr()
    a, k = calls[0][1], calls[0][2]  # (1) the GEMM writes the record's slab
    assert a[2].data_ptr() == slab_ptr
    a, k = calls[1][1], calls[1][2]  # (2)+(3) reads the slab's Q/K bands, writes OUT of place into the compact slots
    assert a[0].data_ptr() == slab_ptr + o_q * e and a[1].data_ptr() == slab_ptr + o_k * e
    assert k.get("q_out") is not None and k.get("k_out") is not None, "norm+RoPE ran IN PLACE on the record's slab: POST-norm q_pre / k_pre bands"
    assert k["q_out"].data_ptr() == q_slot and k["k_out"].data_ptr() == k_slot
    assert k["rstd_q"] is saved.rstd_q and k["rstd_k"] is saved.rstd_k
    # (3q) the Q / K quantizes read the compact NORMED Q/K, the V quantize the slab's V band; (5q) the O quantize the gated O.
    quant = [(n, a) for n, a, _ in calls if n.startswith("quantize")]
    assert [(a[0].data_ptr(), a[1].data_ptr()) for _, a in quant[:3]] == [(q_slot, base + lay.q8), (k_slot, base + lay.k8), (slab_ptr + o_v * e, base + lay.v8)]
    assert (quant[3][1][0].data_ptr(), quant[3][1][1].data_ptr()) == (base + lay.o_gated, base + lay.o8)
    sdpa = next((a, k) for n, a, k in calls if n == "sdpa")  # (4) writes the record's pre-gate O and LSE
    assert sdpa[0][3].data_ptr() == saved.o.data_ptr() and sdpa[1]["lse"] is saved.lse
    gate = next(a for n, a, _ in calls if n == "sigmoid_gate")  # (5) gates saved.o OUT of place into the workspace o_gated
    assert gate[0].data_ptr() == saved.o.data_ptr() and gate[2].data_ptr() == base + lay.o_gated
    out_proj = next(a for n, a, _ in calls if n == "out_proj")  # (6) reads the quantized gated O
    assert out_proj[0].data_ptr() == base + lay.o8


@requires_cuda
@_FAMILY
def test_quantized_fused_forks_stay_declined_for_training(family):
    """The FULLY FUSED quantized forward writes no bf16 slab and no pre-gate O, so it cannot write the record: with
    ``save_for_backward`` its knobs' own training guards fire (typed, naming the knob), a single fusion knob stays the
    both-or-neither decline, and in-place Q/K stays refused.  The fp4 modes' decline is pinned in ``test_block_fp4.py``."""
    b, s = 1, 256
    with pytest.raises(ValueError, match="incompatible with save_for_backward"):
        _declare_quant(_COMMON, b, s, family, fuse_norm_rope=True, fuse_gate=True, scale_o=1.0)
    with pytest.raises(NotImplementedError, match="fuse_norm_rope"):
        _declare_quant(_COMMON, b, s, family, fuse_gate=True, scale_o=1.0)
    with pytest.raises(ValueError, match="inplace_qkv"):
        _declare_quant(_COMMON, b, s, family, inplace_qkv=True)
    r = _declare_quant(_COMMON, b, s, family, save_mode="gate_copy")  # the gate-copy save mode is served too
    assert r.blk.saved_gate_copy and r.blk._gate_copy is not None and r.blk._layout().proj == 0 and r.blk._layout().q >= 0
    # the k_pre copy cannot ride `_compact_v` here (the quantize stages compact V, no such stage): an h_kv band copy of its own
    assert r.blk._compact_v is None and r.blk._kpre_copy is not None and r.blk._kpre_copy.heads == r.geom.h_kv and r.blk._kpre_copy.name == "k_pre_compaction"
    assert [st.name for st in r.blk._stages][:4] == ["qkv_gate_proj", "qk_norm_rope", "gate_compaction", "k_pre_compaction"]
    r16, _, _ = _declare(_COMMON, b, s, save_mode="gate_copy")  # the bf16 gate-copy block is unchanged: k_pre rides compact_v
    assert r16._kpre_copy is None and r16._compact_v is not None


@requires_rubin
@_FAMILY
@_QK_NORM
@_QUANT_GEOMS
def test_quantized_training_forward_is_bitwise_the_inference_block(family, qk_norm, seq_len, causal, batch, h_kv):
    """Same kernels, different buffers: ``out``, the pre-gate ``O``, the LSE, the e4m3 ``q8`` / ``k8`` / ``v8`` and the slab's
    GATE / V bands of the quantized TRAINING forward equal the quantized INFERENCE block's bit for bit.  The slab's Q/K bands
    do NOT: the inference block normed (or, ``rope_only``, rotated) them IN PLACE, the record keeps them PRE-norm -- and the
    forward's own norm+RoPE over the record's bands reproduces the inference slab's normed bands (and the saved rstd, when the
    geometry norms) bitwise.  Over ``_QUANT_GEOMS`` x ``_QK_NORM``; the dense ``S % 128 != 0`` cell pins the rows' decline."""
    geom_kw = _quant_geom(qk_norm, causal, h_kv)
    if not causal and seq_len % 128:
        _dense_tail_declined(geom_kw, batch, seq_len, family)
        return
    b, s = batch, seq_len
    r = _run_training_quant(geom_kw, b, s, family)
    inf = _run_inference_quant(r)
    g, t, d = r.geom, b * s, r.geom.d_head
    assert inf.out.abs().max().item() > 0 and torch.isfinite(inf.out.float()).all()
    assert torch.equal(r.out, inf.out), (r.out.float() - inf.out.float()).abs().max().item()
    assert torch.equal(r.saved.o, inf.o_pre), (r.saved.o.float() - inf.o_pre.float()).abs().max().item()
    assert torch.equal(r.saved.lse, inf.lse), (r.saved.lse - inf.lse).abs().max().item()
    lay_t, lay_i = r.blk._layout(), inf.blk._layout()
    for nm, h in (("q8", g.h_q), ("k8", g.h_kv), ("v8", g.h_kv)):
        a = _view(r.ws, getattr(lay_t, nm), (t, h, d), _E4M3)
        bb = _view(inf.ws, getattr(lay_i, nm), (t, h, d), _E4M3)
        assert torch.equal(a.view(torch.uint8), bb.view(torch.uint8)), f"{nm}: the quantize stages read different values"
    tq, tgate, tk, tv = saved_slab_views(r.saved.proj_slab, g, b, s)
    iq, igate, ik, iv = saved_slab_views(inf.slab, g, b, s)
    assert torch.equal(tgate, igate) and torch.equal(tv, iv)
    assert not torch.equal(tq, iq) and not torch.equal(tk, ik), "the record's Q/K bands are the inference slab's POST-norm bands"
    nq, nk, rq, rk = _replay_norm(r, tq, tk)
    assert torch.equal(nq.view(b, s, g.h_q, d), iq) and torch.equal(nk.view(b, s, g.h_kv, d), ik), "norm+RoPE over the record's bands is not the inference slab"
    if g.qk_norm:
        assert torch.equal(rq.view(b, s, g.h_q), r.saved.rstd_q) and torch.equal(rk.view(b, s, g.h_kv), r.saved.rstd_k)
    else:
        assert rq is None and rk is None and r.saved.rstd_q is None and r.saved.rstd_k is None, "rope_only: no rstd anywhere"


@requires_rubin
@_FAMILY
@_QK_NORM
@_QUANT_GEOMS
def test_quantized_training_record_keeps_pre_norm_bands(family, qk_norm, seq_len, causal, batch, h_kv):
    """The record's Q/K bands ARE the stage-(1) output -- the dequantized product in fp64 rounded once to bf16, at the bf16
    training forward's band bound (``rtol 2**-7, atol 1e-3``; GATE and V too) -- and NOT the normed values (the torch norm
    -- or, ``rope_only``, the rotation -- of a band is far from the band).  The forward's OWN norm+RoPE then quantize,
    replayed over the record's bands into fresh buffers, reproduce the workspace ``q8`` / ``k8`` (and the MXFP8 scale-factor
    blobs) bitwise: the quantizers consumed exactly the normed form of what the record keeps.  Over ``_QUANT_GEOMS`` x
    ``_QK_NORM``; the dense ``S % 128 != 0`` cell pins the rows' decline."""
    geom_kw = _quant_geom(qk_norm, causal, h_kv)
    if not causal and seq_len % 128:
        _dense_tail_declined(geom_kw, batch, seq_len, family)
        return
    b, s = batch, seq_len
    r = _run_training_quant(geom_kw, b, s, family)
    g, t, d = r.geom, b * s, r.geom.d_head
    o_q, o_g, o_k, o_v = g.qkvg_offsets
    proj = _dequantized_fp64_proj(r.inp, r.spec, family).to(torch.bfloat16)
    ref_q = proj[:, o_q : o_q + g.h_q * d].view(b, s, g.h_q, d)
    ref_gate = proj[:, o_g : o_g + g.h_q * d].view(b, s, g.h_q, d)
    ref_k = proj[:, o_k : o_k + g.h_kv * d].view(b, s, g.h_kv, d)
    ref_v = proj[:, o_v : o_v + g.h_kv * d].view(b, s, g.h_kv, d)
    tq, tgate, tk, tv = saved_slab_views(r.saved.proj_slab, g, b, s)
    tol = dict(rtol=2**-7, atol=1e-3)
    for nm, got, want in (("q_pre", tq, ref_q), ("gate", tgate, ref_gate), ("k_pre", tk, ref_k), ("v", tv, ref_v)):
        rel = ((got.float() - want.float()).abs().max() / want.float().abs().max()).item()
        print(f"\n{family} S={s} B={b} causal={causal} qk_norm={qk_norm} record {nm} vs the dequantized GEMM: max_rel={rel:.3e}")
        torch.testing.assert_close(got, want, **tol, msg=nm)
    qn_ref, _ = qk_norm_rope_reference(tq, r.inp["w_q_norm"], r.inp["cos"], r.inp["sin"], g.rope_dim, g.qk_norm_eps, qk_norm=g.qk_norm)
    assert not torch.allclose(qn_ref.float(), tq.float(), **tol), "the saved Q band already IS the normed / rotated Q: the record is POST-norm"
    nq, nk, _, _ = _replay_norm(r, tq, tk)
    lay = r.blk._layout()
    q8, k8 = _view(r.ws, lay.q8, (t, g.h_q, d), _E4M3), _view(r.ws, lay.k8, (t, g.h_kv, d), _E4M3)
    q8r, k8r = torch.empty_like(q8), torch.empty_like(k8)
    if family == "fp8":
        r.blk._quant_q.execute(nq, q8r, r.blk._quant_dev["scale_q"])
        r.blk._quant_kv.execute(nk, k8r, r.blk._quant_dev["scale_k"])
        torch.cuda.synchronize()
    else:
        sfq = _view(r.ws, lay.sf_q, (_sf_slot_bytes(b, g.h_q, s, d),), torch.uint8)
        sfk = _view(r.ws, lay.sf_k, (_sf_slot_bytes(b, g.h_kv, s, d),), torch.uint8)
        sfq_r, sfk_r = torch.empty_like(sfq), torch.empty_like(sfk)
        r.blk._quant_q.execute(nq, q8r, sfq_r, batch=b, seq_len=s)
        r.blk._quant_k.execute(nk, k8r, sfk_r, batch=b, seq_len=s)
        torch.cuda.synchronize()
        assert torch.equal(sfq_r, sfq) and torch.equal(sfk_r, sfk), "the replayed scale factors differ from the forward's"
    assert torch.equal(q8r.view(torch.uint8), q8.view(torch.uint8)) and torch.equal(
        k8r.view(torch.uint8), k8.view(torch.uint8)
    ), "norm + quantize over the record's bands is not the forward's q8 / k8"


@requires_rubin
@_FAMILY
def test_quantized_saved_set_matches_the_oracle(family):
    """Every saved tensor against its oracle: the bands at the GEMM bound (the previous test), ``rstd`` against the oracle
    norm of the block's OWN ``q_pre`` / ``k_pre`` (the bf16 training forward's strict bound), the LSE within 1e-4 of the fp64
    log-sum-exp over the operands the SDPA READ (the exact-Stats contract of the quantized rows), the pre-gate ``O`` and
    ``out`` against the family's fake-quant oracle at the fp8 / mxfp8 forward suites' cosine bar (the kernels' e4m3 P is not
    modelled by any oracle, which is why O is a cosine and the LSE -- independent of P -- is a tight absolute bound)."""
    b, s = 2, 512
    r = _run_training_quant(_COMMON, b, s, family)
    g = r.geom
    saved = r.saved
    assert torch.isfinite(saved.o.float()).all() and torch.isfinite(saved.lse).all() and torch.isfinite(r.out.float()).all()
    tq, _tgate, tk, _tv = saved_slab_views(saved.proj_slab, g, b, s)
    _, rstd_q_ref = qk_norm_rope_reference(tq, r.inp["w_q_norm"], r.inp["cos"], r.inp["sin"], g.rope_dim, g.qk_norm_eps, qk_norm=True)
    _, rstd_k_ref = qk_norm_rope_reference(tk, r.inp["w_k_norm"], r.inp["cos"], r.inp["sin"], g.rope_dim, g.qk_norm_eps, qk_norm=True)
    torch.testing.assert_close(saved.rstd_q, rstd_q_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(saved.rstd_k, rstd_k_ref, rtol=1e-5, atol=1e-6)
    q64, k64, v64 = _sdpa_operands_fp64(r)
    o64, lse64 = _attention_fp64(q64, k64, v64, g)
    d_lse = (saved.lse.double() - lse64).abs().max().item()
    c_o = _cos(saved.o, o64)
    c_out = _cos(r.out, _quant_oracle_out(r))
    print(f"\n{family} training record: max|dLSE|={d_lse:.3e} vs the fp64 log-sum-exp, O cos={c_o:.6f}, out cos={c_out:.6f}")
    assert d_lse <= 1e-4, f"the saved LSE is not the exact log-sum-exp of the operands the SDPA read (max |dLSE| {d_lse:.3e})"
    assert c_o > 0.99, f"pre-gate O cos {c_o}"
    assert c_out > 0.99, f"out cos {c_out}"


@requires_rubin
@_FAMILY
def test_quantized_gate_copy_mode_matches_the_proj_slab_record(family):
    """The gate-copy save mode under a quant spec: the slab stays in the workspace, the compact ``saved.gate`` / ``q_pre`` /
    ``k_pre`` copies are the proj_slab run's bands bitwise (pre-norm: the copies are taken from the slab the out-of-place
    norm left untouched), and ``O`` / ``LSE`` / ``out`` are bitwise the proj_slab run's."""
    b, s = 2, 512
    ps = _run_training_quant(_COMMON, b, s, family)
    gc = _run_training_quant(_COMMON, b, s, family, save_mode="gate_copy", with_pre=True, sentinel=1.5e30)
    assert gc.saved.proj_slab is None and gc.saved.gate.is_contiguous() and gc.blk._layout().proj == 0
    assert torch.equal(gc.saved.gate, ps.saved.gate) and torch.equal(gc.saved.q_pre, ps.saved.q_pre) and torch.equal(gc.saved.k_pre, ps.saved.k_pre)
    assert not (gc.saved.q_pre == 1.5e30).any() and not (gc.saved.k_pre == 1.5e30).any()
    assert torch.equal(gc.saved.o, ps.saved.o) and torch.equal(gc.saved.lse, ps.saved.lse) and torch.equal(gc.out, ps.out)


@requires_rubin
@_FAMILY
def test_quantized_training_launch_count_and_workspace_are_honest(family):
    """CUPTI on the SECOND execute (the first materialises the adapter's cached operands), training block AND the quantized
    inference block on the same inputs: NINE kernels each -- the out-of-place norm and the compact-slot quantizes are the same
    launches writing / reading elsewhere -- and the training execute's memset / memcpy events are EXACTLY the inference
    execute's.  (Zero is not the pin: today the per-tensor FP8 chain carries ONE device memset on BOTH paths -- the prepared
    SDPA host fills its scratch identity-scale word per execute because the block omits ``scale_o`` for a bf16 O,
    ``cudnn.sdpa.fwd.prepared.execute_quantized`` ``needs_identity``; the MXFP8 chain has none.  The training forward adds
    nothing to it; the absolute count is BOUNDED at one, not pinned, so the adapter may drop its fill without touching this
    cell.)  The caching allocator's cumulative allocation COUNT is unchanged across ONE plain, warm training execute and its
    peak does not rise (nothing allocates on the hot path -- not even a temporary freed before ``execute`` returns), and
    ``get_workspace_size()`` is the carve plus the aligned engine scratch, exactly."""
    from torch.profiler import ProfilerActivity, profile

    from cudnn.gated_attention_block.api import _align_up

    def events(fn):
        """Run ``fn`` once warm, then once under the CUDA profiler; returns the CUDA event names and their split into kernels,
        memsets and memcpys."""
        fn()  # warm: the adapter's cached operands
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        memsets = sorted(n for n in names if "memset" in n.lower())
        memcpys = sorted(n for n in names if "memcpy" in n.lower())
        kernels = [n for n in names if n not in memsets and n not in memcpys]
        return names, kernels, memsets, memcpys

    r = _run_training_quant(_COMMON, 2, 512, family)
    lay = r.blk._layout()
    engine = max(r.blk._proj.workspace_bytes(), r.blk._out_proj.workspace_bytes(), r.blk._sdpa.scratch_workspace_bytes(), 1)
    assert r.blk.get_workspace_size() == lay.total_bytes + _align_up(engine)
    inf = _run_inference_quant(r)
    inf.blk._gate.execute = type(inf.blk._gate).execute.__get__(inf.blk._gate)  # drop the O-catching wrapper: plain launches only
    # The allocation pin is measured on ONE plain, warm training execute -- never around ``events()``: its warm-up run and the
    # profiler it creates and destroys acquire and release memory of their own, so the process-wide live bytes can move either
    # way across it (a release by unrelated object cleanup once read as "execute allocated").  The caching allocator's
    # cumulative allocation COUNT is the measure -- a release elsewhere cannot lower it, and a temporary the execute frees
    # before returning still raises it -- with the allocator peak as the second witness for such a temporary's bytes.  Every
    # object the execute reads (``r``: inputs, workspace, record) stays alive across the window.
    _execute_quant(r, r.ws, saved=r.saved)  # settle: the adapter's cached operands are materialised, the launch path is warm
    torch.cuda.synchronize()
    gc.collect()
    live = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    n0 = torch.cuda.memory_stats()["allocation.all.allocated"]
    _execute_quant(r, r.ws, saved=r.saved)
    torch.cuda.synchronize()
    n1 = torch.cuda.memory_stats()["allocation.all.allocated"]
    peak = torch.cuda.max_memory_allocated()
    assert n1 == n0, f"the training forward made {n1 - n0} CUDA allocation(s) on the execute path (allocation.all.allocated {n0} -> {n1})"
    assert peak <= live, f"a temporary on the training forward's execute path: the allocator peak rose from {live} to {peak} bytes"
    names_t, kernels_t, memsets_t, memcpys_t = events(lambda: _execute_quant(r, r.ws, saved=r.saved))
    if not names_t:
        pytest.skip("torch.profiler recorded no CUDA events (CUPTI unavailable on this node); the launch count is unverified here")
    names_i, kernels_i, memsets_i, memcpys_i = events(lambda: _execute_quant(r, inf.ws, lse=inf.lse, out=inf.out, blk=inf.blk))
    print(f"\n{family} TRAINING forward CUDA events:\n  " + "\n  ".join(names_t))
    print(f"{family} INFERENCE forward CUDA events:\n  " + "\n  ".join(names_i))
    assert len(kernels_t) == len(kernels_i) == 9, (len(kernels_t), len(kernels_i))
    assert [n.split("_tensorptr")[0] for n in kernels_t] == [n.split("_tensorptr")[0] for n in kernels_i], "a different kernel ran under training"
    assert (
        memsets_t == memsets_i and memcpys_t == memcpys_i
    ), f"training adds hidden copies / memsets: {memsets_t + memcpys_t} vs inference {memsets_i + memcpys_i}"
    assert not memcpys_t, f"a hidden copy on the execute path: {memcpys_t}"
    # At most ONE memset, and only ever the adapter's fill: the EQUALITY above is the pin; the bound keeps a new memset from
    # hiding behind it, and a plan-time identity word landing in the adapter (0 memsets on both paths) cannot turn this red.
    assert len(memsets_t) <= 1, f"unexpected memsets (at most the per-tensor FP8 adapter's identity-scale fill): {memsets_t}"
