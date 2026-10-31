# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Programmatic dependent launch (PDL) over the gated attention block's DECODE chain: the ``use_pdl`` declaration switch.

With ``use_pdl=True`` every kernel of the block's own chain launches with the programmatic-stream-serialization attribute and
carries the ``griddepcontrol`` pair -- a wait before its first global read of a predecessor's output, a release after its
prologue: the norm + RoPE pass (TMA), the paged write-through, the sigmoid gate, the Rubin d256 decode tile and its split
combine (the FROST GEMM templates already launch that way).  The same kernels run the same math, so the switch is a KNOB:
``out`` is BITWISE the ``use_pdl=False`` block's.

Host cells (any device): the switch is recorded on the block and on every stage it reaches; its default is AUTO -- resolved True on
the DECODE form's decode-tile routes for the bf16 / f16 block on cc 10.7 (``DECODE_PDL_DEFAULT_CCS``, the measured part; the device read
patched here to pin both readings on any host), False on every other part, form (EXTEND, ``kv_source="self"``) and route (the prefill
route), an explicit value winning everywhere -- and a declaration of a pipeline whose stages carry no sites (the e4m3 KV cache here) is a
typed decline that names the switch; the switch is PER
STAGE -- a stage whose kernel carries no sites (the paged d256 prefill tile of the DECODE form's prefill route) launches plain and
the link serializes, which PDL's semantics keep correct -- read back by ``pdl_stages`` / ``use_pdl_effective`` on the block,
``DecodeRoute.pdl_capable`` per route and ``pdl_stages`` on each form's envelope.

Rubin cells: the decode step BITWISE with and without the switch (``out``, the pools) at both declarations (the fused norm +
RoPE projection with the write-through folded in or as its own launch; the unfused projection + the TMA norm + RoPE pass +
the write-through), split AUTO (the gate in the combine) and unsplit (the sigmoid gate as its own launch), the route pinned
equal; CUDA-graph capture of the FIRST execute of a fresh ``use_pdl=True`` declaration on a new stream (the programmatic
edges captured without error -- the driver / runtime versions printed) replaying BITWISE the eager plain chain and the eager
PDL chain, stable over repeated replays; and the DETECTOR of a late wait -- a primary that releases its dependent at once
and writes the operand late (``gated_block_pdl_probe.delayed_write``) ahead of the decode tile, and the tile's own early
release ahead of the combine over NaN-filled partials: GREEN with the shipped kernels, and the RED twins (the wait swapped
for a no-op and the kernel retraced) prove each detector's sensitivity: a kernel that launches with the attribute and reads
before its wait is silent corruption, and these cells are what turns it red.  The prefill route (32 / 2 at ``S_q = 5``) with the
switch on: compiles, bitwise the plain chain, the readback naming the two stages that took the attribute, the first execute
captured and replayed bitwise, and a stage FORCED to carry the switch onto the prefill tile declined by the adapter, typed.
"""

import os
import sys

import pytest
import torch

from cudnn.gated_attention_block import api as _api
from cudnn.frost import buffers as _buffers
from cudnn.frost.buffers import cutedsl_requirement_error
from cudnn.gated_attention_block import GatedAttentionBlockFwd, GatedAttentionBlockGeometry, QsaSpec, QuantSpec, decode_route
from cudnn.gated_attention_block.api import (
    DECODE_PDL_DEFAULT_CCS,
    DECODE_PDL_STAGES,
    DECODE_TILE,
    EXTEND_PDL_STAGES,
    PREFILL_TILE,
    _CacheWrite,
    _QkNormRope,
    _Sdpa,
    _SigmoidGate,
    _view,
)

requirement_error = cutedsl_requirement_error("the gated attention block's PDL tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

requires_rubin = pytest.mark.requires_rubin  # the suite's registered marker (conftest.py): skipped off SM107
requires_blackwell = pytest.mark.requires_blackwell  # the registered marker: an admitted Blackwell-line part of DENSE_SERVING_CCS (cc 10.0; 2026-10-29)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_pdl_probe import cuda_versions, delayed_write, retrace  # noqa: E402
from test_block_serving_decode import _D, _D_MODEL, _DEV, _ROPE, _Case, _bits, _geom, _make_block, _pools, _written  # noqa: E402

_LENS = (100, 384, 500, 0)  # the decode module's standard cell: a tail tile, ring depth 3 and 4, a PADDED entry
_CASES = {}


def _case(group, s_q, lens=_LENS, *, pdl, seed=0, **blk_kw):
    """One compiled DECODE block per (cell, switch), cached: the two arms of a cell share the seed, so every input is identical."""
    key = (group, s_q, tuple(lens), bool(pdl), seed, tuple(sorted(blk_kw.items())))
    if key not in _CASES:
        _CASES[key] = _Case(group, s_q, lens, seed=seed, blk_kw=dict(use_pdl=bool(pdl), **blk_kw)).compile()
    return _CASES[key]


def _stage_names(case):
    return [st.name for st in case.blk._stages]


def _slab(case):
    """The [T, n_qkvg] projection slab the last execute left in the workspace (the Q / GATE / K / V bands), contiguous."""
    g, lay = case.geom, case.blk._ws
    return _view(case.ws, lay.proj, (case.blk.batch * case.blk.seq_len, g.n_qkvg), case.dtype)


# -- the host cell ------------------------------------------------------------------------------------------------------------


def test_the_switch_is_recorded_on_every_stage_resolves_by_part_and_declines_the_unserved_pipelines(monkeypatch):
    off = _Case(16, 1, (100,), seed=1, blk_kw=dict(use_pdl=False))
    on = _Case(16, 1, (100,), seed=1, blk_kw=dict(use_pdl=True))
    assert off.blk.use_pdl is False and on.blk.use_pdl is True and off.blk.use_pdl_requested is False and on.blk.use_pdl_requested is True
    assert off.blk._sdpa.use_pdl is False and on.blk._sdpa.use_pdl is True and on.blk.decode_route.pdl_capable is True
    assert off.blk.pdl_stages == () and off.blk.use_pdl_effective is False
    # the default declaration (fuse_norm_rope=True, lowered to the unfused chain at decode row counts, the paged write-through folded into
    # the norm + RoPE launch at one row): the norm + RoPE pass carries the switch and there is no separate cache write to carry it; the
    # fused fork (its template launches with PDL unconditionally) under the control knob carries none
    assert isinstance(on.blk._norm_rope, _QkNormRope) and on.blk._norm_rope.use_pdl is True and on.blk._cache_write is None and on.blk._gate is None
    assert on.blk.cache_write_in_norm_rope and on.blk.pdl_stages == ("qk_norm_rope", "sdpa") and on.blk.use_pdl_effective is True
    sep = _Case(16, 1, (100,), seed=1, blk_kw=dict(use_pdl=True, fuse_cache_write_small_m=False))
    assert isinstance(sep.blk._cache_write, _CacheWrite) and sep.blk._cache_write.use_pdl is True
    assert sep.blk.pdl_stages == ("qk_norm_rope", "cache_write", "sdpa") and sep.blk.use_pdl_effective is True
    fork = _Case(16, 1, (100,), seed=1, blk_kw=dict(use_pdl=True, fuse_norm_rope_small_m=True))
    assert fork.blk._norm_rope is None and fork.blk._cache_write is None and fork.blk._gate is None and fork.blk.use_pdl is True
    assert fork.blk.pdl_stages == ("sdpa",) and fork.blk.use_pdl_effective is True
    # the unfused, unsplit declaration with the separate cache write reaches every stage with its own kernel
    on1 = _Case(16, 1, (100,), seed=1, blk_kw=dict(use_pdl=True, fuse_norm_rope=False, decode_split_kv=1, fuse_cache_write_small_m=False))
    assert isinstance(on1.blk._norm_rope, _QkNormRope) and on1.blk._norm_rope.use_pdl is True
    assert isinstance(on1.blk._cache_write, _CacheWrite) and on1.blk._cache_write.use_pdl is True
    assert isinstance(on1.blk._gate, _SigmoidGate) and on1.blk._gate.use_pdl is True
    assert _stage_names(on1) == ["qkv_gate_proj", "qk_norm_rope", "cache_write", "sdpa", "sigmoid_gate", "out_proj"]
    assert on1.blk.pdl_stages == ("qk_norm_rope", "cache_write", "sdpa", "sigmoid_gate") and on1.blk.use_pdl_effective is True
    off1 = _Case(16, 1, (100,), seed=1, blk_kw=dict(use_pdl=False, fuse_norm_rope=False, decode_split_kv=1, fuse_cache_write_small_m=False))
    assert off1.blk._norm_rope.use_pdl is False and off1.blk._cache_write.use_pdl is False and off1.blk._gate.use_pdl is False
    # the same stage list either way: the switch adds no launch and removes none
    assert _stage_names(on) == _stage_names(off) and _stage_names(on1) == _stage_names(off1)
    # AUTO (the default, None): the PART decides -- True on the measured part (cc 10.7) for this decode-tile route, False on every other
    # part; this host's own reading is one of the two, and the declaration is kept beside the resolved switch
    assert DECODE_PDL_DEFAULT_CCS == ((10, 7),)
    auto = _Case(16, 1, (100,), seed=1)
    assert auto.blk.use_pdl_requested is None and auto.blk.use_pdl is (_api._device_cc_of(_DEV) in DECODE_PDL_DEFAULT_CCS)
    for cc, want in (((10, 7), True), ((10, 0), False), ((10, 3), False), ((9, 0), False), (None, False)):
        monkeypatch.setattr(_api, "_device_cc_of", lambda device, _cc=cc: _cc)
        blk = _Case(16, 1, (100,), seed=1).blk
        assert blk.use_pdl is want and blk.use_pdl_requested is None and blk._norm_rope.use_pdl is want and blk._sdpa.use_pdl is want, cc
        assert blk.pdl_stages == (("qk_norm_rope", "sdpa") if want else ()) and blk.use_pdl_effective is want, cc
        # an explicit value wins on every part
        assert _Case(16, 1, (100,), seed=1, blk_kw=dict(use_pdl=False)).blk.use_pdl is False, cc
        assert _Case(16, 1, (100,), seed=1, blk_kw=dict(use_pdl=True)).blk.use_pdl is True, cc
    monkeypatch.undo()
    # the knob is None / True / False
    with pytest.raises(TypeError, match="use_pdl must be None"):
        _Case(16, 1, (100,), seed=1, blk_kw=dict(use_pdl=1))
    # an unserved pipeline declines, typed, naming the switch -- before any other check of that pipeline runs
    with pytest.raises(NotImplementedError, match="use_pdl=True is served for the bf16 / f16 block only"):
        _Case(16, 1, (100,), seed=1, blk_kw=dict(use_pdl=True, kv_cache_dtype=torch.float8_e4m3fn))


def test_auto_and_the_explicit_switch_answer_for_one_domain_on_the_measured_part(monkeypatch):
    """With the device read patched to the measured part, AUTO resolves True ONLY where an explicit True is served: the two readings of
    the knob answer for one domain.  Pools declared in the activation dtype (``kv_cache_dtype=torch.bfloat16``, the ``None`` spelling)
    take the default and the explicit True alike -- a serving stack that spells the dtype can form its own A/B --, a 16-bit pool of the
    other width hears the attribute's own ``ValueError``, not the switch's decline.  On every pipeline the default excludes -- the EXTEND
    form, ``kv_source="self"``, a QuantSpec pipeline, a QsaSpec geometry under the in-block indexer (the decode indexer over the pools on
    both readings of ``fuse_prepare``), the e4m3 cache -- AUTO
    reads False with no stage switched; the explicit True is served per stage on the two bf16 forms without a decode route (their
    prefill tiles carry no sites), SERVED on the block-sparse decode mode (the sparse core and its combine carry the sites) -- where AUTO
    resolves True too on the caller-list block, with and without the indexer band, since the measured part entered
    ``SPARSE_DECODE_PDL_DEFAULT_CCS`` (2026-10-27) -- and is the typed decline on the QuantSpec pipeline, the e4m3 cache, the sparse core's
    packed form and the in-block indexer (its kernels would take the switch unmeasured)."""
    import test_block_qsa_decode as _qsa
    import test_block_qsa_fp8_kv as _qkv
    import test_block_serving_extend as _ext

    monkeypatch.setattr(_api, "_device_cc_of", lambda device: (10, 7))
    off = lambda blk: blk.use_pdl is False and blk.use_pdl_requested is None and blk.pdl_stages == () and blk.use_pdl_effective is False  # noqa: E731
    decline = "use_pdl=True is served for the bf16 / f16 block only"
    # pools spelled in the activation dtype: the same block as kv_cache_dtype=None, under both readings
    auto = _make_block(12, 1, 1)[0]
    spelled = _make_block(12, 1, 1, kv_cache_dtype=torch.bfloat16)[0]
    assert auto.use_pdl is True and spelled.use_pdl is True and spelled.use_pdl_requested is None and not spelled.kv_cache_e4m3
    assert spelled.pdl_stages == auto.pdl_stages == ("qk_norm_rope", "sdpa", "sigmoid_gate") and spelled.use_pdl_effective is True
    on = _make_block(12, 1, 1, kv_cache_dtype=torch.bfloat16, use_pdl=True)[0]
    assert on.use_pdl is True and on.use_pdl_requested is True and on.pdl_stages == auto.pdl_stages
    assert _make_block(12, 1, 1, kv_cache_dtype=torch.bfloat16, use_pdl=False)[0].pdl_stages == ()
    assert _make_block(12, 1, 1, dtype=torch.float16, kv_cache_dtype=torch.float16, use_pdl=True)[0].use_pdl is True
    assert _make_block(12, 1, 1, dtype=torch.float16, kv_cache_dtype=torch.float16)[0].use_pdl is True
    with pytest.raises(ValueError, match="differs from the activation dtype"):
        _make_block(12, 1, 1, kv_cache_dtype=torch.float16, use_pdl=True)
    with pytest.raises(NotImplementedError, match="no e4m3 KV cache"):
        _make_block(12, 1, 1, kv_cache_dtype=torch.float8_e4m3fn, use_pdl=True)
    # the EXTEND form and the block's own K / V: no decode route, AUTO False; the explicit True threads the stages with sites only
    ext_auto, *_ = _ext._declare_extend(2, 128, max_kv_len=128)
    assert off(ext_auto) and ext_auto.decode_route is None and isinstance(ext_auto._cache_write, _CacheWrite)
    ext_on, *_ = _ext._declare_extend(2, 128, max_kv_len=128, use_pdl=True)
    assert ext_on.use_pdl is True and ext_on._sdpa.use_pdl is False and "sdpa" not in ext_on.pdl_stages and ext_on.use_pdl_effective is False
    assert ext_on.pdl_stages == ("qk_norm_rope", "cache_write", "sigmoid_gate")
    geom, _ = _geom(12)
    mk = lambda *shape: torch.empty(*shape, dtype=torch.bfloat16, device=_DEV)  # noqa: E731
    args = (mk(1, 1, _D_MODEL), mk(geom.n_qkvg, _D_MODEL), mk(_D), mk(_D), mk(1, 1, _ROPE), mk(1, 1, _ROPE), mk(_D_MODEL, 24 * _D), mk(1, 1, _D_MODEL), geom)
    own = GatedAttentionBlockFwd(*args)
    assert off(own) and own.decode_route is None and own.kv_source == "self"
    own_on = GatedAttentionBlockFwd(*args, use_pdl=True)
    assert own_on.use_pdl is True and own_on._sdpa.use_pdl is False and "sdpa" not in own_on.pdl_stages and own_on.use_pdl_effective is False
    # a QuantSpec pipeline: AUTO False, the explicit True declined
    spec = QuantSpec(descale_h=0.01, descale_w_qkvg=0.02, descale_w_o=0.03, scale_q=1.0, scale_k=2.0, scale_v=3.0, scale_o=4.0)
    g8 = GatedAttentionBlockGeometry(d_model=_D_MODEL, h_q=8, h_kv=2, d_head=_D, rope_dim=_ROPE)
    e4 = lambda *shape: torch.zeros(*shape, dtype=torch.float8_e4m3fn, device=_DEV)  # noqa: E731
    qargs = (
        e4(1, 256, _D_MODEL),
        e4(g8.n_qkvg, _D_MODEL),
        mk(_D),
        mk(_D),
        mk(1, 256, _ROPE),
        mk(1, 256, _ROPE),
        e4(_D_MODEL, 8 * _D),
        mk(1, 256, _D_MODEL),
        g8,
    )
    assert off(GatedAttentionBlockFwd(*qargs, quant=spec))
    with pytest.raises(NotImplementedError, match=decline):
        GatedAttentionBlockFwd(*qargs, quant=spec, use_pdl=True)
    # the FP8-projection + bf16-attention mode (QuantSpec(attention_dtype=torch.bfloat16)) on the DECODE form: AUTO False (the quantize
    # launch carries no griddepcontrol sites; the switched chain is unmeasured there), the explicit True the same decline, naming the mode
    mode = QuantSpec.bf16_attention(0.01, 0.02, 0.03, 4.0)
    kc, vc = _pools(8, 2, 16, _D, "hnd", torch.bfloat16)
    geom_br = GatedAttentionBlockGeometry(d_model=_D_MODEL, h_q=8, h_kv=2, d_head=_D, rope_dim=_ROPE, causal_bottom_right=True)  # the paged forms' diagonal
    mode_args = (
        e4(4, 1, _D_MODEL),
        e4(geom_br.n_qkvg, _D_MODEL),
        mk(_D),
        mk(_D),
        mk(4, 1, _ROPE),
        mk(4, 1, _ROPE),
        e4(_D_MODEL, 8 * _D),
        mk(4, 1, _D_MODEL),
        geom_br,
    )
    mode_kw = dict(
        paged_kv_page_size=16, kv_source="paged", sample_k_cache=kc, sample_v_cache=vc, sample_block_table=torch.zeros(4, 8, dtype=torch.int32, device=_DEV)
    )
    mode_blk = GatedAttentionBlockFwd(*mode_args, quant=mode, **mode_kw)
    assert mode_blk.attention_bf16 and mode_blk.decode_route is not None and mode_blk.decode_route.pdl_capable and off(mode_blk)
    assert mode_blk._sdpa.use_pdl is False and mode_blk._norm_rope.use_pdl is False
    with pytest.raises(NotImplementedError, match=decline) as e:
        GatedAttentionBlockFwd(*mode_args, quant=mode, use_pdl=True, **mode_kw)
    assert "FP8-projection + bf16-attention mode" in str(e.value) and "None = auto reads False" in str(e.value)
    env_mode = GatedAttentionBlockFwd.serving_envelope(geom_br, torch.float8_e4m3fn, _DEV, device_cc=(10, 7), quant=mode)
    assert env_mode.decode.pdl_default is False
    # the block-sparse decode mode: a QsaSpec geometry under CALLER lists (with and without the indexer band) -- AUTO True on the measured
    # part (the sparse chain's own measured-parts tuple holds cc 10.7 since 2026-10-27: the switched block measured bitwise the plain one and
    # never slower over B 1 / 4 / 32), the explicit True SERVED the same way: the sparse core and its split combine carry the sites, so the
    # sparse stage takes the switch beside the chain's other switched stages; the explicit False pins the plain chain on the measured part
    assert _api.SPARSE_DECODE_PDL_DEFAULT_CCS == ((10, 7),)
    sparse = _qsa._declare(_qsa.GEOM_SMALL, 2, 1, n_pages=4)[0]
    assert sparse.use_pdl is True and sparse.use_pdl_requested is None and sparse._sdpa.use_pdl is True and sparse.decode_route is None and sparse.sparse_decode
    assert "sdpa_sparse" in sparse.pdl_stages and sparse.use_pdl_effective is True
    sparse_on = _qsa._declare(_qsa.GEOM_SMALL, 2, 1, n_pages=4, use_pdl=True)[0]
    assert sparse_on.use_pdl is True and sparse_on.use_pdl_requested is True and sparse_on._sdpa.use_pdl is True and sparse_on.sparse_decode
    assert "sdpa_sparse" in sparse_on.pdl_stages and sparse_on.use_pdl_effective is True and sparse_on.pdl_stages == sparse.pdl_stages
    assert sparse_on.pdl_stages == tuple(st.name for st in sparse_on._stages if getattr(st, "use_pdl", False))
    sparse_off = _qsa._declare(_qsa.GEOM_SMALL, 2, 1, n_pages=4, use_pdl=False)[0]
    assert sparse_off.use_pdl is False and sparse_off.use_pdl_requested is False and sparse_off.pdl_stages == () and sparse_off._sdpa.use_pdl is False
    band = _qsa._declare(_qsa.GEOM_SMALL, 4, 1, n_pages=6, index_band=True, fuse_norm_rope=False)[0]
    assert band.use_pdl is True and band.use_pdl_requested is None and band.geom.index_band, "AUTO True with the indexer band under caller lists"
    assert "sdpa_sparse" in band.pdl_stages and "cache_write" in band.pdl_stages and band.use_pdl_effective is True
    # the in-block indexer's DECODE step over the pools (2026-10-21): AUTO False on both readings of fuse_prepare (its stage threads the
    # switch it is handed -- `pdl_names` spells the kernels that would take it -- and is handed False), the explicit True the same decline
    for fused in (False, True):
        ix = _qsa._declare_indexer(_qsa.GEOM_SMALL, 2, 1, fuse_prepare=fused)
        assert off(ix) and ix.indexer_decode and ix._indexer_decode.use_pdl is False and ix.fuse_prepare is fused
        assert ix._indexer_decode.pdl_names == ((("qsa_compress_step",) if not fused else ()) + ("qsa_score", "qsa_topk"))
        with pytest.raises(NotImplementedError, match=decline):
            _qsa._declare_indexer(_qsa.GEOM_SMALL, 2, 1, fuse_prepare=fused, use_pdl=True)
    band_on = _qsa._declare(_qsa.GEOM_SMALL, 4, 1, n_pages=6, index_band=True, fuse_norm_rope=False, use_pdl=True)[0]
    assert band_on.use_pdl is True and "sdpa_sparse" in band_on.pdl_stages and "cache_write" in band_on.pdl_stages and band_on.use_pdl_effective is True
    # the sparse core's PACKED form with the switch: declined by name at declaration (the setup launch's metadata is read above every role wait)
    with pytest.raises(NotImplementedError, match="thd=True with use_pdl=True is not served") as e_thd:
        _declare_sparse_thd(use_pdl=True)
    assert "on a QsaSpec geometry" in str(e_thd.value) and "setup launch" in str(e_thd.value)  # the adapter's one statement, carried by the block
    e4m3 = _qkv._declare(_qkv.GEOM_SMALL, 2, 1, fuse_norm_rope=False)[0]
    assert off(e4m3) and e4m3.kv_cache_e4m3 and e4m3.decode_route is None
    with pytest.raises(NotImplementedError, match="no e4m3 KV cache"):
        _qkv._declare(_qkv.GEOM_SMALL, 2, 1, fuse_norm_rope=False, use_pdl=True)


def _declare_sparse_thd(**blk_kw):
    """A PACKED (thd=True) QsaSpec declaration on the block's own K / V (the sparse core's THD arm; top-left) -- the one sparse form the switch
    never reaches."""
    import test_block_qsa_decode as _qsa

    geom = GatedAttentionBlockGeometry(**_qsa.GEOM_SMALL, qsa=QsaSpec())
    mk = lambda *shape: torch.empty(*shape, dtype=torch.bfloat16, device=_DEV)  # noqa: E731
    return GatedAttentionBlockFwd(
        mk(32, geom.d_model),
        mk(geom.n_qkvg, geom.d_model),
        mk(_D),
        mk(_D),
        mk(32, _ROPE),
        mk(32, _ROPE),
        mk(geom.d_model, geom.h_q * _D),
        mk(32, geom.d_model),
        geom,
        thd=True,
        num_sequences=2,
        max_seq_len=32,
        token_capacity=32,
        **blk_kw,
    )


def test_the_prefill_route_keeps_the_switch_off_stage_four_and_reads_back_which_stages_took_it(monkeypatch):
    """The switch is PER STAGE: at 32 / 2, ``S_q = 5`` the DECODE form rides the paged d256 PREFILL tile (80 packed rows), whose
    template carries no griddepcontrol sites -- so the block declares fine, hands the switch to the norm + RoPE pass and the
    write-through, never to stage (4) (it launches plain and the link serializes: correct by PDL's semantics), and reads it back:
    ``pdl_stages`` names the two, ``use_pdl_effective`` is False, ``DecodeRoute.pdl_capable`` is False on that route and True on a
    decode-tile route, and the envelopes name the switched stage kinds per form before anything is declared."""
    # 32 sequences x 2 KV heads = 64 CTAs: the prefill route's fill; 32 x 5 = 160 rows is past the small-M lowering's band (the fused
    # declaration keeps its fork there), so the unfused chain -- the stages the switch reaches -- is the fuse_norm_rope=False declaration
    # (160 rows with S_q > 1: the paged write-through stays its own launch, the measured rule of fuse_cache_write_small_m)
    case = _Case(16, 5, (100,) * 32, seed=1, blk_kw=dict(use_pdl=True, fuse_norm_rope=False))
    blk = case.blk
    assert blk.use_pdl is True and blk.decode_route.prefill_tile and blk.decode_route.pdl_capable is False and blk._sdpa.thd and blk.decode_route.batch == 32
    assert blk._sdpa.use_pdl is False, "the switch never reaches a kernel without the sites"
    assert isinstance(blk._norm_rope, _QkNormRope) and blk._norm_rope.use_pdl is True and blk._cache_write.use_pdl is True and blk._gate is None
    assert _stage_names(case) == ["qkv_gate_proj", "qk_norm_rope", "cache_write", "sdpa", "out_proj"]
    assert blk.pdl_stages == ("qk_norm_rope", "cache_write") and blk.use_pdl_effective is False
    plain = _Case(16, 5, (100,) * 32, seed=1, blk_kw=dict(use_pdl=False, fuse_norm_rope=False))
    assert plain.blk.pdl_stages == () and plain.blk.use_pdl_effective is False and _stage_names(plain) == _stage_names(case)
    # AUTO on the prefill route: the default is scoped to decode-tile routes (where stage (4) takes the switch), so even on the measured
    # part this declaration launches the plain chain -- an explicit True still threads the two stages (above)
    monkeypatch.setattr(_api, "_device_cc_of", lambda device: (10, 7))
    auto = _Case(16, 5, (100,) * 32, seed=1, blk_kw=dict(fuse_norm_rope=False))
    assert auto.blk.decode_route.prefill_tile and auto.blk.use_pdl is False and auto.blk.use_pdl_requested is None and auto.blk.pdl_stages == ()
    assert _Case(16, 5, (100,), seed=1).blk.use_pdl is True, "below the fill the same step rides the decode tile: the default is on"
    monkeypatch.undo()
    # the stage's own rule and the record's fact agree: a decode-tile route takes the switch, the prefill route and no route do not
    geom = blk.geom
    assert _Sdpa.pdl_capable(None) is False and _Sdpa.pdl_capable(decode_route(geom, 4)) is True and _Sdpa.pdl_capable(decode_route(geom, 5, batch=32)) is False
    assert decode_route(geom, 4).pdl_capable is True and decode_route(geom, 5, batch=32).pdl_capable is False and decode_route(geom, 33).pdl_capable is False
    assert (
        decode_route(geom, 5).pdl_capable is True and decode_route(geom, 5, batch=4).decode_tile
    ), "below the fill the step rides the decode tile: the switch reaches stage (4)"
    # the envelopes: the switched stage KINDS per form (stage (4) on a decode-tile route only), the route's own flag per S_q
    env = GatedAttentionBlockFwd.serving_envelope(geom, torch.bfloat16, _DEV)
    assert env.decode.pdl_stages == DECODE_PDL_STAGES == ("qk_norm_rope", "cache_write", "sigmoid_gate", "sdpa")
    assert env.extend.pdl_stages == EXTEND_PDL_STAGES == ("qk_norm_rope", "cache_write", "sigmoid_gate")
    # the part's AUTO reading of the switch, before anything is declared: this host's, and the measured part's against another part's --
    # a FORM-level reading of a SERVED form (the measured part's reading needs the DSL's Rubin target, patched present as the decode module
    # does; an unserved dtype reads False there); the per-step reading is `pdl_default and route(s_q, batch).pdl_capable`
    assert env.decode.pdl_default is (env.decode.served and env.device_cc in DECODE_PDL_DEFAULT_CCS)
    monkeypatch.setattr(_buffers, "_cutedsl_has_sm107", lambda: True)
    rubin = GatedAttentionBlockFwd.serving_envelope(geom, torch.bfloat16, _DEV, device_cc=(10, 7)).decode
    assert rubin.served and rubin.pdl_default is True and rubin.route(4).pdl_capable and not rubin.route(5, batch=32).pdl_capable
    assert GatedAttentionBlockFwd.serving_envelope(geom, torch.bfloat16, _DEV, device_cc=(10, 0)).decode.pdl_default is False
    fp32 = GatedAttentionBlockFwd.serving_envelope(geom, torch.float32, _DEV, device_cc=(10, 7)).decode
    assert not fp32.served and fp32.pdl_default is False, "the default is a reading of a served form"
    monkeypatch.undo()
    # the Rubin table's route facts on a Rubin reading (a cc 10.0 host's own reading records the SM100 line: group 16 at S_q 4 is the
    # prefill tile there, pinned in test_block_serving_decode.py)
    rubin_routes = GatedAttentionBlockFwd.serving_envelope(geom, torch.bfloat16, _DEV, device_cc=(10, 7)).decode
    assert (
        all(r.pdl_capable is r.decode_tile for r in env.decode.routes)
        and rubin_routes.route(5, batch=32).pdl_capable is False
        and rubin_routes.route(4).pdl_capable
    )
    assert rubin_routes.route(5).pdl_capable is True
    # the token-unit policy takes the same step on the decode tile at any batch, where the switch reaches stage (4); so does the default
    # policy below the fill (the Rubin reading on any host)
    monkeypatch.setattr(_api, "_route_device_cc", lambda device: None)
    for extra in (dict(decode_route_policy="token_units"), dict()):
        tu = _Case(16, 5, (100,), seed=1, blk_kw=dict(use_pdl=True, **extra))
        assert tu.blk.decode_route.decode_tile and tu.blk.decode_route.pdl_capable and tu.blk._sdpa.use_pdl is True and tu.blk.use_pdl_effective is True
        assert tu.blk.pdl_stages == ("qk_norm_rope", "sdpa"), extra  # 5 rows: the write-through rides the norm + RoPE launch


# -- the Rubin cells: bitwise with and without the switch ----------------------------------------------------------------------

# The declarations of the DECODE form the switch reaches, each a different chain:
#   auto:            the production default -- fuse_norm_rope=True LOWERED at decode row counts: slab GEMM | norm + RoPE (TMA) | write-through
#                    | decode tile | gated combine | out proj (+ the split-K reducer)
#   fused-auto:      fused projection + norm/RoPE + the write-through folded in | decode tile | gated combine | out proj (the control knob)
#   fused-cachewrite: the same with the write-through as its own launch between the projection and the tile
#   fused-unsplit:   fused projection | unsplit tile | sigmoid gate | out proj
#   unfused-auto:    slab GEMM | norm + RoPE (TMA) | write-through | decode tile | gated combine | out proj (declared unfused)
#   unfused-unsplit: slab GEMM | norm + RoPE (TMA) | write-through | unsplit tile | sigmoid gate | out proj
_DECLARATIONS = {
    "auto": dict(),
    "fused-auto": dict(fuse_norm_rope_small_m=True),
    "fused-cachewrite": dict(fuse_norm_rope_small_m=True, fuse_cache_write=False),
    "fused-unsplit": dict(fuse_norm_rope_small_m=True, decode_split_kv=1),
    "unfused-auto": dict(fuse_norm_rope=False),
    "unfused-unsplit": dict(fuse_norm_rope=False, decode_split_kv=1),
}
_CELLS = [(name, s_q) for name in _DECLARATIONS for s_q in ((1, 4) if name in ("auto", "fused-auto", "unfused-unsplit") else (1,))]


@requires_rubin
@pytest.mark.parametrize("decl, s_q", _CELLS, ids=[f"{n}-sq{s}" for n, s in _CELLS])
def test_the_decode_step_is_bitwise_with_and_without_pdl(decl, s_q):
    kw = _DECLARATIONS[decl]
    off = _case(16, s_q, pdl=False, **kw)
    on = _case(16, s_q, pdl=True, **kw)
    # the SAME kernels: the route the adapter compiled (template, packing, split, where the gate runs) and the stage list
    assert on.route_readback() == off.route_readback(), (on.route_readback(), off.route_readback())
    assert _stage_names(on) == _stage_names(off)
    assert on.route_readback()["template"] == DECODE_TILE
    a = off.run().clone()
    b = on.run().clone()
    on.check_against_oracle(b, what=f"{decl} s_q={s_q} use_pdl=True")
    assert torch.equal(a, b), f"{decl}: out differs with use_pdl (max|diff| {(a.float() - b.float()).abs().max().item():.3e})"
    assert torch.equal(_bits(on.k_cache), _bits(off.k_cache)) and torch.equal(_bits(on.v_cache), _bits(off.v_cache)), "the write-through differs"
    # a second PDL step is bitwise the first: nothing in the chain raced
    c = on.run().clone()
    assert torch.equal(b, c)


@requires_blackwell
@pytest.mark.parametrize(
    "decl, group, s_q", [("auto", 16, 1), ("unfused-auto", 6, 1), ("unfused-unsplit", 6, 4)], ids=["auto-16-sq1", "unfused-auto-6-sq1", "unfused-unsplit-6-sq4"]
)
def test_the_decode_step_is_bitwise_with_and_without_pdl_on_cc_10_0(decl, group, s_q):
    """The explicit switch on cc 10.0 (2026-10-29; AUTO reads False there -- the default is the measured part's): the SM100 decode tile
    and its combine carry the griddepcontrol pair, so the switched chain is BITWISE the plain one on ``out`` and on the pools, the same
    kernels (``fuse_gate=False``: the gate its own launch on this part; the fused-fork declarations are the sm_107a decline there)."""
    kw = dict(_DECLARATIONS[decl], fuse_gate=False)
    off = _case(group, s_q, pdl=False, **kw)
    on = _case(group, s_q, pdl=True, **kw)
    assert on.route_readback() == off.route_readback() and _stage_names(on) == _stage_names(off)
    assert on.route_readback()["template"] == DECODE_TILE and on.route_readback()["file"].endswith("/sm100/decode_d256_f16.py")
    assert on.blk.use_pdl is True and on.blk._sdpa.use_pdl is True and "sdpa" in on.blk.pdl_stages and off.blk.use_pdl is False
    a = off.run().clone()
    b = on.run().clone()
    on.check_against_oracle(b, what=f"cc 10.0 {decl} group {group} s_q={s_q} use_pdl=True")
    assert torch.equal(a, b), f"{decl}: out differs with use_pdl on cc 10.0 (max|diff| {(a.float() - b.float()).abs().max().item():.3e})"
    assert torch.equal(_bits(on.k_cache), _bits(off.k_cache)) and torch.equal(_bits(on.v_cache), _bits(off.v_cache)), "the write-through differs"
    c = on.run().clone()
    assert torch.equal(b, c)
    auto = _Case(group, s_q, _LENS, seed=0, blk_kw=dict(fuse_gate=False, **_DECLARATIONS[decl]))
    assert auto.blk.use_pdl is False and auto.blk.use_pdl_requested is None and auto.blk.pdl_stages == ()


@requires_rubin
def test_the_default_declaration_resolves_the_switch_on_and_is_bitwise_the_plain_chain():
    """On the measured part the DEFAULT declaration (``use_pdl`` left None) is the PDL chain: the readback says so (``use_pdl`` True from
    ``use_pdl_requested`` None, the stages that took the attribute), the route and the stage list are the explicit chains', and ``out`` and
    the pools are BITWISE the explicit plain chain's (``use_pdl=False``) and the explicit PDL chain's."""
    auto = _Case(16, 1, _LENS, seed=0).compile()
    off = _case(16, 1, pdl=False)
    on = _case(16, 1, pdl=True)
    assert auto.blk.use_pdl_requested is None and auto.blk.use_pdl is True and auto.blk.use_pdl_effective is True
    assert auto.blk.pdl_stages == on.blk.pdl_stages == ("qk_norm_rope", "sdpa") and off.blk.pdl_stages == ()
    assert auto.route_readback() == off.route_readback() and _stage_names(auto) == _stage_names(off) == _stage_names(on)
    a = off.run().clone()
    b = auto.run().clone()
    auto.check_against_oracle(b, what="the default declaration")
    assert torch.equal(a, b), f"the default chain differs from the plain chain (max|diff| {(a.float() - b.float()).abs().max().item():.3e})"
    assert torch.equal(b, on.run().clone())
    assert torch.equal(_bits(auto.k_cache), _bits(off.k_cache)) and torch.equal(_bits(auto.v_cache), _bits(off.v_cache)), "the write-through differs"


# -- the Rubin cell: capture + replay ------------------------------------------------------------------------------------------


@requires_rubin
def test_capture_of_the_first_pdl_execute_replays_bitwise_the_eager_plain_and_pdl_chains(tmp_path):
    """The serving stack's defaults on the PDL declaration: the FIRST execute of a fresh ``use_pdl=True`` block captured on a NEW
    stream, no warm-up, under ``set_sync_debug_mode("error")``; the programmatic edges are captured without error, the capture
    launches nothing, the replay is BITWISE the eager plain (``use_pdl=False``) chain and the eager PDL chain, and stays bitwise
    over repeated replays.  The driver / runtime / torch / DSL versions are printed (the record of what served the capture)."""
    v = cuda_versions()
    print(f"cuda versions: {v}")
    lens = (300, 0, 1000, 77)
    plain = _Case(16, 2, lens, seed=23, blk_kw=dict(use_pdl=False)).compile()
    ref = plain.run().clone()
    plain.check_against_oracle(ref, what="plain eager")
    case = _Case(16, 2, lens, seed=23, blk_kw=dict(use_pdl=True))  # fresh: never executed
    case.blk.check_support()
    case.blk.compile()
    case.ws = torch.empty(case.blk.get_workspace_size(), dtype=torch.uint8, device=_DEV)
    case.out.fill_(float("nan"))
    k_before = _written(case.k_cache)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    dumped = None
    try:
        if hasattr(graph, "enable_debug_mode"):
            graph.enable_debug_mode()
        with torch.cuda.graph(graph, stream=stream):
            prev = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                case.execute()  # the FIRST execute of this declaration, captured: every launch carries the PDL attribute
            finally:
                torch.cuda.set_sync_debug_mode(prev)
        torch.cuda.synchronize()
        assert torch.isnan(case.out).all() and _written(case.k_cache) == k_before, "the capture launched work"
        if hasattr(graph, "debug_dump"):
            # Best effort: the graph's DOT (cudaGraphDebugDotPrint) -- the kernel-node census beside the verdict.
            try:
                path = str(tmp_path / "pdl_decode_graph.dot")
                graph.debug_dump(path)
                dot = open(path).read()
                dumped = dict(kernel_nodes=dot.count("KERNEL"), programmatic=sum(dot.lower().count(w) for w in ("programmatic", "port")))
            except Exception as exc:  # noqa: BLE001 -- the dump is a record, never the verdict
                dumped = f"debug_dump unavailable ({type(exc).__name__})"
        graph.replay()
        torch.cuda.synchronize()
        replayed = case.out.clone()
        case.check_against_oracle(replayed, what="replay")
        assert torch.equal(replayed, ref), "the replay of the captured PDL chain differs from the eager plain chain"
        case.out.fill_(float("nan"))
        case.execute()
        torch.cuda.synchronize()
        assert torch.equal(case.out, replayed), "the eager PDL chain differs from its replay"
        for _ in range(50):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(case.out, replayed), "repeated replays of the PDL chain are not bitwise (a race across the programmatic edges)"
    finally:
        graph.reset()
    print(f"graph census: {dumped}")


# -- the Rubin cell: the prefill route with the switch on ----------------------------------------------------------------------


@requires_rubin
def test_the_prefill_route_step_is_bitwise_with_the_switch_and_its_first_execute_captures_and_replays_bitwise():
    """32 / 2 at ``S_q = 5`` (the prefill route) with ``use_pdl=True``: the block COMPILES (the prefill tile launches plain, the norm +
    RoPE pass and the write-through with the attribute), ``out`` and the pools are BITWISE the plain chain's, the readback names the two
    stages; a stage FORCED to carry the switch onto the prefill tile is the adapter's typed decline at compile (the block never does
    it); and the FIRST execute of a fresh switched declaration captured on a new stream under ``set_sync_debug_mode("error")`` replays
    BITWISE the eager plain chain, stable over repeated replays."""
    # 32 sequences x 2 KV heads = 64 CTAs: the prefill route's fill; 160 rows is past the small-M lowering's band (the fused declaration
    # keeps its fork there), so the unfused chain -- the two switched stages before stage (4) -- is the fuse_norm_rope=False declaration
    off = _case(16, 5, _LENS * 8, pdl=False, fuse_norm_rope=False)
    on = _case(16, 5, _LENS * 8, pdl=True, fuse_norm_rope=False)
    assert on.route_readback() == off.route_readback() and on.route_readback()["template"] == PREFILL_TILE, on.route_readback()
    assert on.blk.decode_route.prefill_tile and on.blk.decode_route.batch == 32
    assert on.blk.pdl_stages == ("qk_norm_rope", "cache_write") and on.blk.use_pdl_effective is False and off.blk.pdl_stages == ()
    assert on.blk._sdpa.use_pdl is False and _stage_names(on) == _stage_names(off)
    a = off.run().clone()
    b = on.run().clone()
    on.check_against_oracle(b, what="prefill route use_pdl=True")
    assert torch.equal(a, b), f"out differs with use_pdl on the prefill route (max|diff| {(a.float() - b.float()).abs().max().item():.3e})"
    assert torch.equal(_bits(on.k_cache), _bits(off.k_cache)) and torch.equal(_bits(on.v_cache), _bits(off.v_cache)), "the write-through differs"
    assert torch.equal(b, on.run().clone()), "a second switched step is not bitwise the first"
    # a stage forced to carry the switch onto a kernel without the sites: the adapter's typed decline, never a silent plain launch (at the
    # fill: below it the same step rides the decode tile, which carries the sites, and the forced switch is simply the switch)
    forced = _Case(16, 5, _LENS * 8, seed=0, blk_kw=dict(use_pdl=True, fuse_norm_rope=False))
    assert forced.blk.decode_route.prefill_tile
    forced.blk._sdpa.use_pdl = True
    forced.blk._sdpa.check_support()
    with pytest.raises(NotImplementedError, match="programmatic-dependent-launch sites"):
        forced.blk._sdpa.compile()
    # capture + replay of the FIRST execute of a fresh switched declaration (the same seed, lengths and declaration as `off`: identical
    # inputs, pages, slots)
    case = _Case(16, 5, _LENS * 8, seed=0, blk_kw=dict(use_pdl=True, fuse_norm_rope=False))
    assert case.blk.decode_route.prefill_tile
    case.blk.check_support()
    case.blk.compile()
    case.ws = torch.empty(case.blk.get_workspace_size(), dtype=torch.uint8, device=_DEV)
    case.out.fill_(float("nan"))
    k_before = _written(case.k_cache)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            prev = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                case.execute()
            finally:
                torch.cuda.set_sync_debug_mode(prev)
        torch.cuda.synchronize()
        assert torch.isnan(case.out).all() and _written(case.k_cache) == k_before, "the capture launched work"
        graph.replay()
        torch.cuda.synchronize()
        replayed = case.out.clone()
        case.check_against_oracle(replayed, what="replay (prefill route, use_pdl=True)")
        assert torch.equal(replayed, a), "the replay of the captured switched chain differs from the eager plain chain"
        for _ in range(20):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(case.out, replayed), "repeated replays are not bitwise"
    finally:
        graph.reset()


# -- the Rubin cells: the detector of a late wait ------------------------------------------------------------------------------


def _standalone_tile(case, *, pdl):
    """The block's stage (4) as a standalone adapter on the block's OWN views (the decode module's ``_adapter_like``), with the switch."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    blk, g = case.blk, case.geom
    q, gate_v, o, _ = case.slab_views()
    o_buf = torch.empty_like(o)
    bts = tuple(int(x) for x in blk._descs["block_table"].stride)
    api = SdpaFwdDslSm100(
        sample_q=q.transpose(1, 2),
        sample_k=case.k_cache,
        sample_v=case.v_cache,
        sample_o=o_buf.transpose(1, 2),
        sample_lse=None,
        is_causal=g.is_causal,
        causal_bottom_right=g.causal_bottom_right,
        scale_softmax=g.scale,
        seq_kv_lens_present=True,
        sched_policy=blk._sdpa._impl.sched_policy,
        split_kv=blk.decode_split_kv,
        pack_gqa=True if blk.decode_route.pack_gqa else None,
        paged_page_size=case.page,
        paged_max_seq_len_kv=case.max_kv_len,
        paged_table_stride=bts,
        paged_table_v_stride=bts,
        sample_gate=gate_v.transpose(1, 2),
        use_pdl=pdl,
    )
    api.check_support()
    api.compile()
    assert api.kernel_template == DECODE_TILE and api.use_pdl is pdl
    ws = torch.empty(max(api.scratch_workspace_bytes(), 1), dtype=torch.uint8, device=_DEV)

    def run():
        api.execute(
            q.transpose(1, 2),
            case.k_cache,
            case.v_cache,
            o_buf.transpose(1, 2),
            seq_kv_lens=case.kv_lens_t,
            block_table=case.block_table,
            workspace=ws,
            gate=gate_v.transpose(1, 2),
        )
        return o_buf

    return api, run, o_buf


_SPIN_US = (300.0, 3000.0)  # the primary's late write: well past any launch latency, so a tile that does not wait reads the NaN


def _race_the_tile(case, run, o_buf, ref):
    """Ahead of each tile launch: NaN over the operand, then the primary that releases its dependent at once and restores the
    operand late.  Returns, per (operand, spin), whether the tile's output equals ``ref``."""
    verdicts = {}
    slab = _slab(case)
    slab_saved = slab.clone()
    k_saved = case.k_cache.clone()
    for spin in _SPIN_US:
        slab.fill_(float("nan"))
        delayed_write(slab_saved, slab, spin_us=spin)
        run()
        torch.cuda.synchronize()
        verdicts[("slab", spin)] = torch.equal(o_buf, ref)
        slab.copy_(slab_saved)
        case.k_cache.fill_(float("nan"))
        delayed_write(k_saved, case.k_cache, spin_us=spin)
        run()
        torch.cuda.synchronize()
        verdicts[("k_pool", spin)] = torch.equal(o_buf, ref)
        case.k_cache.copy_(k_saved)
    return verdicts


@requires_rubin
def test_detector_the_decode_tile_reads_the_slab_and_the_pools_only_after_its_dependent_wait():
    """GREEN: the shipped tile, launched with the attribute right after a primary that released it at once and wrote the slab
    (the Q box) or the K pool late, produces the undisturbed output -- its first read of either sits below its wait."""
    case = _case(16, 2, pdl=True, seed=5)
    case.run()
    api, run, o_buf = _standalone_tile(case, pdl=True)
    ref = run().clone()
    torch.cuda.synchronize()
    _, _, o_blk, _ = case.slab_views()
    assert torch.equal(ref, o_blk), "the standalone PDL tile differs from the block's own stage (4)"
    verdicts = _race_the_tile(case, run, o_buf, ref)
    print(f"detector (shipped tile): {verdicts}")
    assert all(verdicts.values()), f"the tile read an operand before its dependent wait: {verdicts}"


@requires_rubin
def test_detector_turns_red_when_the_tile_launches_with_the_attribute_but_does_not_wait(monkeypatch):
    """The detector's sensitivity: the same cell with the tile's wait swapped for a no-op (the launch attribute kept, the kernel
    retraced) reads the slab / the pool during the primary's spin -- a deliberately late (here: absent) wait turns the cell RED.
    Runs after the GREEN cell; the retrace is undone before the next test compiles."""
    import cudnn.frost.compiled_cache as compiled_cache

    case = _case(16, 2, pdl=True, seed=5)
    case.run()
    mod = case.blk._sdpa._impl._k_mod  # the loaded decode module (one object per template params)
    assert mod.__name__.startswith("cudnn.frost._templates.") and hasattr(mod, "wait_on_dependent_grids")
    monkeypatch.setenv("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", "1")  # the on-disk artifact would be the SHIPPED kernel
    monkeypatch.setattr(mod, "wait_on_dependent_grids", lambda: None)
    retrace(mod.compile)
    try:
        api, run, o_buf = _standalone_tile(case, pdl=True)  # a fresh trace: the attribute on, the wait gone
        ref = run().clone()  # unraced, the no-wait tile is still correct: nothing is in flight ahead of it
        torch.cuda.synchronize()
        verdicts = _race_the_tile(case, run, o_buf, ref)
        print(f"detector (tile WITHOUT its wait): {verdicts}")
        assert not all(verdicts.values()), f"the detector did not turn red for a tile that never waits: {verdicts}"
        assert not any(verdicts[k] for k in verdicts if k[1] == max(_SPIN_US)), f"the longest spin must turn every operand red: {verdicts}"
        del api, run
    finally:
        monkeypatch.undo()
        retrace(mod.compile)  # the memo held the no-wait object under the shipped key
        compiled_cache.clear_memo()


_COMBINE_LENS = (8192,)  # 64 KV tiles: the tile runs long enough past its early release that a combine reading the partials at once reads garbage


def _combine_case(*, pdl, seed=7, fresh=False):
    kw = dict(decode_split_kv=2)  # two long halves (a split AUTO over 8K keys would be many short ones)
    if fresh:
        return _Case(16, 1, _COMBINE_LENS, seed=seed, blk_kw=dict(use_pdl=pdl, **kw)).compile()
    return _case(16, 1, _COMBINE_LENS, pdl=pdl, seed=seed, **kw)


def _run_over_poisoned_workspace(case):
    case.ws.fill_(0xFF)  # every slot NaN-ish before the step: the slab, the partials, O -- a correct chain writes each before it reads it
    case.out.fill_(float("nan"))
    case.execute()
    torch.cuda.synchronize()
    return case.out.clone()


@requires_rubin
def test_detector_the_combine_reads_the_partials_only_after_its_dependent_wait():
    """GREEN: over a workspace poisoned before the step, the PDL chain (the tile releases the combine right after its prologue;
    the combine waits before its first read of the partial slabs) is bitwise the plain chain."""
    plain = _combine_case(pdl=False)
    ref = _run_over_poisoned_workspace(plain)
    plain.check_against_oracle(ref, what="plain, poisoned workspace")
    case = _combine_case(pdl=True)
    assert case.blk.decode_split_kv == 2 and case.blk.gate_fused
    for _ in range(3):
        out = _run_over_poisoned_workspace(case)
        assert torch.equal(out, ref), "the PDL chain over a poisoned workspace differs from the plain chain"


@requires_rubin
def test_detector_turns_red_when_the_combine_launches_with_the_attribute_but_does_not_wait(monkeypatch):
    """The combine's wait swapped for a no-op (attribute kept, the entry retraced in a FRESH plan): it reads the partial slabs
    while the released tile is still writing them -- the poisoned workspace shows through and the cell turns RED."""
    import cudnn.frost.compiled_cache as compiled_cache
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine

    plain = _combine_case(pdl=False)
    ref = _run_over_poisoned_workspace(plain)
    monkeypatch.setenv("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", "1")
    monkeypatch.setattr(split_combine, "wait_on_dependent_grids", lambda: None)
    retrace(split_combine.compile_ptr)
    try:
        case = _combine_case(pdl=True, fresh=True)  # a fresh plan: its combine entry traced WITHOUT the wait
        reds = 0
        for _ in range(3):
            out = _run_over_poisoned_workspace(case)
            reds += int(not torch.equal(out, ref))
        print(f"detector (combine WITHOUT its wait): {reds} / 3 steps differ from the plain chain")
        assert reds > 0, "the detector did not turn red for a combine that never waits"
        del case
    finally:
        monkeypatch.undo()
        retrace(split_combine.compile_ptr)
        compiled_cache.clear_memo()
