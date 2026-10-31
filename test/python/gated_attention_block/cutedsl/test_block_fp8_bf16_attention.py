# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The FP8-PROJECTION + BF16-ATTENTION mode of the gated attention block's paged serving forms -- ``QuantSpec(attention_dtype=torch.bfloat16)``,
the spelling of a serving stack whose attention projections are static per-tensor FP8 W8A8 while its KV cache and attention math are bf16:
the two FP8 projection GEMMs with the descale folded into their bf16 epilogue, QK-norm + RoPE and the pool write-through in bf16, the bf16
decode / prefill tiles over bf16 pools (the bf16 serving forms' own kernels and routes), the gated O quantized to e4m3 at ``scale_o`` for the
FP8 out projection.

Host cells (any device, the declaration cells even on the CPU): the appended ``QuantSpec`` attribute -- its default, the
:meth:`QuantSpec.bf16_attention` spelling, ``validate()``'s rows (f16 refused, a Q / K / V scale set under the mode refused by name, a None
scale under ``attention_dtype=None`` a typed ``ValueError``) --, the exports and the API index rows, the part statement
(:data:`FP8_PROJ_BF16_ATTENTION_CCS`), the workspace carve (no compact e4m3 Q / K / V under the mode, ``o8`` alone; every carve without the
flag byte-identical), the declared DECODE and EXTEND blocks' facts (the stage list, the bf16 SDPA stage, the distinct ``quantize_o``, the
three device scalars, the ``update_quant_scales`` plan fact), every refused combination by its exact text, and the envelope's prediction
of the mode (``serving_envelope(quant=)``, ``quant_modes``) on the Rubin and the cc 10.0 readings.

Both-parts cell (``requires_dense_part``): the FP8 forced-tile projection GEMMs at the mode's shapes (decode row counts and 128 rows; the
8 / 2 test geometry and the 24 / 4 geometry at ``d_model`` 5120), the ``jit-only`` capture-safe route, the alpha epilogue, against the fp32
product of the dequantized codes rounded once to bf16 -- the cell whose cc 10.0 verdict admits the part into
:data:`FP8_PROJ_BF16_ATTENTION_CCS`.  The whole-block DECODE / EXTEND cells of the mode live beside the bf16 forms' cells
(``test_block_serving_decode.py`` / ``test_block_serving_extend.py``: the ``quant=`` arms of their harnesses).
"""

import dataclasses
import os
import sys
from pathlib import Path

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error
from cudnn.gated_attention_block import (
    DENSE_SERVING_CCS,
    FP8_PROJ_BF16_ATTENTION_CCS,
    QUANT_MODE_FP8_PROJ_BF16_ATTENTION,
    GatedAttentionBlockFwd,
    GatedAttentionBlockGeometry,
    MxQuantSpec,
    QsaSpec,
    QuantSpec,
)
from cudnn.gated_attention_block import api as _api
from cudnn.gated_attention_block.api import _CacheWrite, _fp8_proj_bf16_attention_part_decline, _plan_workspace, _Quantize, _Sdpa, _SigmoidGate
from cudnn.gated_attention_block.kernels.proj_gemm import build_proj_gemm, run_proj_gemm

requirement_error = cutedsl_requirement_error("the gated attention block's FP8-projection + bf16-attention tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

requires_dense_part = pytest.mark.requires_dense_part  # the registered marker: EVERY served part of DENSE_SERVING_CCS (cc 10.7 and cc 10.0)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_reference import amax_scale, quant_e4m3  # noqa: E402

_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_SM107, _SM100 = (10, 7), (10, 0)
E4 = torch.float8_e4m3fn
BF16 = torch.bfloat16
_D, _ROPE, _D_MODEL = 256, 64, 512
_GEOM = GatedAttentionBlockGeometry(d_model=_D_MODEL, h_q=8, h_kv=2, d_head=_D, rope_dim=_ROPE, causal_bottom_right=True)
_FP8_SDPA = QuantSpec(descale_h=0.01, descale_w_qkvg=0.02, descale_w_o=0.03, scale_q=1.0, scale_k=2.0, scale_v=3.0, scale_o=4.0)  # the fp8 SDPA pipeline
_MODE = QuantSpec.bf16_attention(0.01, 0.02, 0.03, 4.0)  # the mode


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


def _pools(n_pages, h, page, dtype=BF16):
    return tuple(torch.zeros(n_pages, h, page, _D, dtype=dtype, device=_DEV) for _ in range(2))


def _decode_args(b, s_q, *, geom=_GEOM, h_dtype=E4, act=BF16):
    """The DECODE declaration's nine positional samples: e4m3 ``h`` / weights (``h_dtype``) under the mode, bf16 activations."""
    mk = lambda *shape, dt=act: torch.zeros(*shape, dtype=dt, device=_DEV)  # noqa: E731
    return (
        mk(b, s_q, geom.d_model, dt=h_dtype),
        mk(geom.n_qkvg, geom.d_model, dt=h_dtype),
        torch.ones(_D, dtype=act, device=_DEV),
        torch.ones(_D, dtype=act, device=_DEV),
        mk(b, s_q, _ROPE),
        mk(b, s_q, _ROPE),
        mk(geom.d_model, geom.h_q * _D, dt=h_dtype),
        mk(b, s_q, geom.d_model),
        geom,
    )


def _decode_kw(b, *, page=16, n_pages=64, h_kv=2, **over):
    kc, vc = _pools(n_pages, h_kv, page)
    kw = dict(
        paged_kv_page_size=page,
        kv_source="paged",
        sample_k_cache=kc,
        sample_v_cache=vc,
        sample_block_table=torch.zeros(b, n_pages, dtype=torch.int32, device=_DEV),
    )
    kw.update(over)
    return kw


def _extend_args_kw(t, b_max, *, geom=_GEOM, act=BF16):
    """The EXTEND declaration under the mode: e4m3 token matrix and weights, bf16 pools, the packed envelope."""
    mk = lambda *shape, dt=act: torch.zeros(*shape, dtype=dt, device=_DEV)  # noqa: E731
    kc, vc = _pools(9, geom.h_kv, 16)
    args = (
        mk(t, geom.d_model, dt=E4),
        mk(geom.n_qkvg, geom.d_model, dt=E4),
        torch.ones(_D, dtype=act, device=_DEV),
        torch.ones(_D, dtype=act, device=_DEV),
        mk(t, _ROPE),
        mk(t, _ROPE),
        mk(geom.d_model, geom.h_q * _D, dt=E4),
        mk(t, geom.d_model),
        geom,
    )
    kw = dict(
        thd=True,
        num_sequences=b_max,
        max_seq_len=t,
        cu_seqlens=True,
        token_capacity=t,
        paged_kv_page_size=16,
        kv_source="paged",
        sample_k_cache=kc,
        sample_v_cache=vc,
        sample_block_table=torch.zeros(b_max, 8, dtype=torch.int32, device=_DEV),
    )
    return args, kw


# ---------------------------------------------------------------------------
# The attribute
# ---------------------------------------------------------------------------


def test_quantspec_attention_dtype_is_appended_and_validate_names_every_row():
    """``attention_dtype`` is appended after ``dtype`` with the default None (every existing positional call unchanged); the
    :meth:`bf16_attention` spelling; the four ``validate`` rows: the dtype row, ``attention_dtype`` None / bf16 only (f16 by name), a
    None scale under ``attention_dtype=None`` a TYPED ``ValueError`` naming the scale and the way in (the untyped ``float(None)`` before),
    and under the mode the three Q / K / V scales MUST be None (each refused by name) while the three descales and ``scale_o`` stay
    finite positive floats.  The alpha properties are the same arithmetic on both spellings."""
    assert [f.name for f in dataclasses.fields(QuantSpec)] == [
        "descale_h",
        "descale_w_qkvg",
        "descale_w_o",
        "scale_q",
        "scale_k",
        "scale_v",
        "scale_o",
        "dtype",
        "attention_dtype",
    ]
    s = QuantSpec(0.01, 0.02, 0.03, 1.0, 2.0, 3.0, 4.0)
    assert s.attention_dtype is None and s.attention_bf16 is False and s.dtype == E4
    s.validate()
    mode = QuantSpec.bf16_attention(0.01, 0.02, 0.03, 4.0)
    assert mode == QuantSpec(0.01, 0.02, 0.03, None, None, None, 4.0, E4, BF16) and mode.attention_bf16 and mode.attention_dtype == BF16
    mode.validate()
    assert mode.alpha_qkvg == s.alpha_qkvg == 0.01 * 0.02 and mode.alpha_o == s.alpha_o == (1.0 / 4.0) * 0.03
    assert _MODE.validate() is None and QuantSpec.bf16_attention(1.0, 1.0, 1.0, 1.0, dtype=E4).dtype == E4
    with pytest.raises(NotImplementedError, match="only torch.float8_e4m3fn is served"):
        QuantSpec.bf16_attention(0.01, 0.02, 0.03, 4.0, dtype=torch.float8_e5m2).validate()
    with pytest.raises(
        NotImplementedError, match=r"QuantSpec.attention_dtype must be None .* or torch.bfloat16 .*torch.float16 has no such mode.*got torch.float16"
    ):
        QuantSpec(0.01, 0.02, 0.03, None, None, None, 4.0, E4, torch.float16).validate()
    with pytest.raises(NotImplementedError, match="QuantSpec.attention_dtype must be None"):
        QuantSpec(0.01, 0.02, 0.03, None, None, None, 4.0, E4, E4).validate()
    for name in ("scale_q", "scale_k", "scale_v"):
        kw = dict(descale_h=0.01, descale_w_qkvg=0.02, descale_w_o=0.03, scale_q=1.0, scale_k=2.0, scale_v=3.0, scale_o=4.0)
        kw[name] = None
        with pytest.raises(
            ValueError, match=rf"QuantSpec.{name} is None: the fp8 d256 SDPA \(attention_dtype=None\) .* declare attention_dtype=torch.bfloat16"
        ):
            QuantSpec(**kw).validate()
        kw = dict(descale_h=0.01, descale_w_qkvg=0.02, descale_w_o=0.03, scale_q=None, scale_k=None, scale_v=None, scale_o=4.0, attention_dtype=BF16)
        kw[name] = 1.5
        with pytest.raises(ValueError, match=rf"QuantSpec.{name}=1.5 is unused under attention_dtype=torch.bfloat16.*pass None"):
            QuantSpec(**kw).validate()
    for name in ("descale_h", "descale_w_qkvg", "descale_w_o", "scale_o"):
        kw = dict(descale_h=0.01, descale_w_qkvg=0.02, descale_w_o=0.03, scale_q=None, scale_k=None, scale_v=None, scale_o=4.0, attention_dtype=BF16)
        kw[name] = None
        with pytest.raises(ValueError, match=rf"QuantSpec.{name} is None: a finite positive float is required"):
            QuantSpec(**kw).validate()
        kw[name] = 0.0
        with pytest.raises(ValueError, match=rf"QuantSpec.{name} must be a finite positive float"):
            QuantSpec(**kw).validate()
        kw[name] = float("inf")
        with pytest.raises(ValueError, match=rf"QuantSpec.{name} must be a finite positive float"):
            QuantSpec(**kw).validate()
    # MxQuantSpec carries no such attribute: the mode is QuantSpec's
    assert not hasattr(MxQuantSpec(descale_w_o=1.0), "attention_dtype") and not hasattr(MxQuantSpec, "bf16_attention")


def test_exports_the_part_tuple_and_the_api_index_rows():
    """The two constants are exported by the package and indexed; the appended field, the two readbacks and the envelopes' ``quant_modes``
    are in the API index; the tuple holds cc 10.7 and only served dense parts."""
    pkg = sys.modules["cudnn.gated_attention_block"]  # the imported package object (ONE import style in this module: `from ... import`)

    assert "FP8_PROJ_BF16_ATTENTION_CCS" in pkg.__all__ and "QUANT_MODE_FP8_PROJ_BF16_ATTENTION" in pkg.__all__
    assert pkg.FP8_PROJ_BF16_ATTENTION_CCS is FP8_PROJ_BF16_ATTENTION_CCS and QUANT_MODE_FP8_PROJ_BF16_ATTENTION == "QuantSpec(attention_dtype=torch.bfloat16)"
    assert _SM107 in FP8_PROJ_BF16_ATTENTION_CCS and set(FP8_PROJ_BF16_ATTENTION_CCS) <= set(DENSE_SERVING_CCS), (
        FP8_PROJ_BF16_ATTENTION_CCS,
        DENSE_SERVING_CCS,
    )
    rows = (Path(__file__).resolve().parents[2] / "api_index" / "api_index.txt").read_text(encoding="utf-8").splitlines()
    for name in (
        "cudnn.gated_attention_block.FP8_PROJ_BF16_ATTENTION_CCS",
        "cudnn.gated_attention_block.QUANT_MODE_FP8_PROJ_BF16_ATTENTION",
        "cudnn.gated_attention_block.QuantSpec.attention_dtype",
        "cudnn.gated_attention_block.QuantSpec.attention_bf16",
        "cudnn.gated_attention_block.QuantSpec.bf16_attention",
        "cudnn.gated_attention_block.DecodeEnvelope.quant_modes",
        "cudnn.gated_attention_block.ExtendEnvelope.quant_modes",
    ):
        assert name in rows, name


def test_the_part_statement_names_the_tuple_and_the_lift():
    """``_fp8_proj_bf16_attention_part_decline``: None on every part of the tuple, the one statement (the mode's spelling, the tuple, the
    lift) off it -- cc 10.0 while the tuple lacks it, no CUDA device."""
    for cc in FP8_PROJ_BF16_ATTENTION_CCS:
        assert _fp8_proj_bf16_attention_part_decline(cc) is None
    off = _fp8_proj_bf16_attention_part_decline((9, 0))
    assert off.startswith("QuantSpec(attention_dtype=torch.bfloat16) on cc 9.0") and "FP8_PROJ_BF16_ATTENTION_CCS" in off and "the lift" in off
    assert "no CUDA device" in _fp8_proj_bf16_attention_part_decline(None)
    if _SM100 not in FP8_PROJ_BF16_ATTENTION_CCS:
        assert "on cc 10.0" in _fp8_proj_bf16_attention_part_decline(_SM100)


def test_the_carve_skips_the_compact_e4m3_qkv_slots_and_keeps_o8():
    """``_plan_workspace(fp8=True, fp8_attention_bf16=True)``: ``q8 = k8 = v8 = -1`` (stage (4) reads the bf16 slab), ``o8`` reserved, every
    other offset the bf16 in-place twin's, ``total_bytes`` = the bf16 twin's + the aligned ``o8`` slot; the default spells today's fp8 carve."""
    for b, s in ((4, 1), (2, 256), (1, 128)):
        fp8 = _plan_workspace(_GEOM, b, s, BF16, False, False, True, fp8=True)
        assert fp8 == _plan_workspace(_GEOM, b, s, BF16, False, False, True, fp8=True, fp8_attention_bf16=False) and fp8.q8 >= 0 and fp8.o8 >= 0
        mode = _plan_workspace(_GEOM, b, s, BF16, False, False, True, fp8=True, fp8_attention_bf16=True)
        bf = _plan_workspace(_GEOM, b, s, BF16, False, False, True)
        assert mode.q8 == mode.k8 == mode.v8 == -1 and mode.o8 >= 0 and bf.o8 == -1
        assert (mode.proj, mode.o, mode.q, mode.k, mode.v) == (bf.proj, bf.o, bf.q, bf.k, bf.v) == (bf.proj, bf.o, -1, -1, -1)
        assert mode.total_bytes == bf.total_bytes + _api._align_up(b * s * _GEOM.h_q * _D) and mode.total_bytes < fp8.total_bytes
        assert mode.o8 == bf.total_bytes, "o8 sits where the bf16 twin's carve ends"


# ---------------------------------------------------------------------------
# The declared blocks (host)
# ---------------------------------------------------------------------------


def test_declared_decode_block_under_the_mode_records_the_bf16_attention_and_the_one_quantize_launch(monkeypatch):
    """A DECODE declaration under the mode (the Rubin reading of the route on any host): ``attention_bf16``, the stage list
    ``qkv_gate_proj, qk_norm_rope, sdpa, quantize_o, out_proj`` with the gate in the combine (split >= 2; ``sigmoid_gate`` under
    ``fuse_gate=False``, ``cache_write`` under ``fuse_cache_write_small_m=False``), a bf16 SDPA stage reading the slab in place (``fp8`` False
    on the stage, the slab's token stride), no ``quantize_q`` / ``quantize_kv``, a DISTINCT ``quantize_o`` over ``h_q`` heads, both FP8
    projections with the alpha epilogue and capture-safe, the write-through folded into the norm + RoPE launch (the AUTO fold admitted),
    the three device scalars, the ``update_quant_scales`` plan fact, PDL AUTO off."""
    monkeypatch.setattr(_api, "_route_device_cc", lambda device: None)
    blk = GatedAttentionBlockFwd(*_decode_args(4, 1), quant=_MODE, fuse_gate=True, decode_split_kv=2, **_decode_kw(4))
    assert blk.attention_bf16 and blk.quant is _MODE and blk.dtype == E4 and blk.act_dtype == BF16 and blk.kv_cache_dtype == BF16 and blk.inplace_qkv
    assert [st.name for st in blk._stages] == ["qkv_gate_proj", "qk_norm_rope", "sdpa", "quantize_o", "out_proj"]
    assert blk.gate_fused and blk._gate is None and blk.decode_route.decode_tile and blk.decode_split_kv == 2
    sd = blk._sdpa
    assert (
        isinstance(sd, _Sdpa)
        and sd.dtype == BF16
        and sd.fp8 is False
        and sd.o_dtype == BF16
        and sd.token_stride == _GEOM.n_qkvg
        and sd.gate_token_stride == _GEOM.n_qkvg
    )
    assert sd.kv_source == "paged" and sd.paged_dense and sd.fuse_gate and not sd.mxfp8
    assert blk._quant_q is None and blk._quant_kv is None and blk._quant_k is None and blk._quant_v is None
    assert isinstance(blk._quant_o, _Quantize) and blk._quant_o.name == "quantize_o" and blk._quant_o.heads == _GEOM.h_q and blk._quant_o.dtype_in == BF16
    assert blk._proj.dtype == E4 and blk._proj.out_dtype == BF16 and blk._proj.alpha and blk._proj.capture_safe
    assert blk._out_proj.dtype == E4 and blk._out_proj.out_dtype == BF16 and blk._out_proj.alpha and blk._out_proj.capture_safe
    assert blk.cache_write_in_norm_rope and blk._cache_write is None and blk._norm_rope.page_size == 16 and blk._norm_rope.dtype == BF16
    assert blk.use_pdl is False and blk.use_pdl_requested is None and blk.pdl_stages == () and blk.fuse_norm_rope is False and not blk.fuse_norm_rope_effective
    lay = blk._layout()
    assert lay.q8 == lay.k8 == lay.v8 == -1 and lay.o8 >= 0 and lay.q == lay.k == lay.v == -1
    assert blk._quant_dev_values(_MODE) == dict(alpha_qkvg=_MODE.alpha_qkvg, alpha_o=_MODE.alpha_o, scale_o=4.0)
    blk._check_declaration()
    with pytest.raises(ValueError, match="QuantSpec.attention_dtype is a plan fact"):
        blk._check_quant_update(_FP8_SDPA)
    with pytest.raises(RuntimeError, match=r"call compile\(\) before update_quant_scales"):
        blk._check_quant_update(QuantSpec.bf16_attention(0.02, 0.02, 0.03, 5.0))
    with pytest.raises(TypeError, match="spec must be a QuantSpec"):
        blk._check_quant_update(MxQuantSpec(descale_w_o=1.0))
    # the elementwise gate and the separate cache write
    sep = GatedAttentionBlockFwd(*_decode_args(4, 1), quant=_MODE, fuse_gate=False, fuse_cache_write_small_m=False, **_decode_kw(4))
    assert [st.name for st in sep._stages] == ["qkv_gate_proj", "qk_norm_rope", "cache_write", "sdpa", "sigmoid_gate", "quantize_o", "out_proj"]
    assert isinstance(sep._cache_write, _CacheWrite) and isinstance(sep._gate, _SigmoidGate) and sep._gate.dtype == BF16 and sep._quant_o is not sep._quant_q
    # an unsplit decode tile: the gate is its own launch, as on the bf16 form
    one = GatedAttentionBlockFwd(*_decode_args(4, 1), quant=_MODE, fuse_gate=True, decode_split_kv=1, **_decode_kw(4))
    assert not one.gate_fused and isinstance(one._gate, _SigmoidGate) and [st.name for st in one._stages][3:5] == ["sigmoid_gate", "quantize_o"]
    # the fp8 SDPA pipeline over the block's own K / V is untouched by the attribute (the stage list the fp8 suite pins)
    own = GatedAttentionBlockFwd(*_decode_args(1, 64), quant=_FP8_SDPA)
    assert not own.attention_bf16 and own._sdpa.dtype == E4 and own._sdpa.fp8 and own._quant_q is not None and own._quant_o is own._quant_q
    assert [st.name for st in own._stages] == ["qkv_gate_proj", "qk_norm_rope", "quantize_q", "quantize_kv", "sdpa", "sigmoid_gate", "out_proj"]
    assert set(own._quant_dev_values(_FP8_SDPA)) == {"alpha_qkvg", "alpha_o", "scale_q", "scale_k", "scale_v", "scale_o", "descale_q", "descale_k", "descale_v"}
    assert own._layout().q8 >= 0 and own._layout().o8 >= 0


def test_declared_extend_block_under_the_mode_records_the_paged_thd_stage():
    """The EXTEND form under the mode: the packed declaration with e4m3 token matrices builds the bf16 paged THD SDPA stage, the
    separate cache write, the elementwise gate (or the fused one under ``fuse_gate=True``: the bf16 body's epilogue), the distinct
    ``quantize_o`` and the FP8 out projection; ``num_tokens_live`` is refused by name at execute."""
    args, kw = _extend_args_kw(128, 2)
    blk = GatedAttentionBlockFwd(*args, quant=_MODE, **kw)
    assert blk.attention_bf16 and blk.thd and blk.token_capacity == 128 and blk.kv_source == "paged"
    assert [st.name for st in blk._stages] == ["qkv_gate_proj", "qk_norm_rope", "cache_write", "sdpa", "sigmoid_gate", "quantize_o", "out_proj"]
    sd = blk._sdpa
    assert sd.thd and sd.kv_source == "paged" and sd.dtype == BF16 and sd.fp8 is False and sd.token_stride == _GEOM.n_qkvg and not sd.fuse_gate
    assert blk._proj.capture_safe and blk._out_proj.capture_safe and blk._quant_q is None and isinstance(blk._quant_o, _Quantize)
    gated = GatedAttentionBlockFwd(*args, quant=_MODE, fuse_gate=True, **kw)
    assert gated.gate_fused and gated._gate is None and gated._sdpa.fuse_gate
    assert [st.name for st in gated._stages] == ["qkv_gate_proj", "qk_norm_rope", "cache_write", "sdpa", "quantize_o", "out_proj"]
    blk._check_declaration()
    for n in (64, 128):  # the capacity count too: a quantized pipeline takes the capacity launch (None), as before the mode
        with pytest.raises(
            NotImplementedError, match=r"num_tokens_live=\d+ on a quantized pipeline \(QuantSpec\).*the FP8 projections' live-row launch is a follow-up"
        ):
            blk._check_num_tokens_live(n)
    assert blk._check_num_tokens_live(None) is None


def test_every_refused_combination_is_typed_and_names_the_mode():
    """Each combination the mode refuses, by its exact text, reachable from the declaration (accept cells above): the fp8 SDPA spelling
    and an MxQuantSpec on the paged forms (the write-through's quant row, which names the way in), ``fuse_norm_rope=True``,
    ``kv_source="self"`` (dense and packed, with or without the write-through), ``inplace_qkv=False``, training (``save_for_backward``:
    the write-through's row on the paged forms, the mode's inference-only row on the self forms), the e4m3 KV cache, ``use_pdl=True``
    (AUTO False), a QsaSpec geometry, an explicit ``fuse_cache_write=True`` / ``fuse_norm_rope_small_m``, and the mode with 16-bit
    ``h`` (FP8 needs both halves)."""
    a = _decode_args(4, 1)
    with pytest.raises(NotImplementedError, match=r"paged_kv_page_size with quant=QuantSpec: .* or on the FP8-projection \+ bf16-attention mode") as e:
        GatedAttentionBlockFwd(*a, quant=_FP8_SDPA, **_decode_kw(4))
    assert "the fp8 d256 SDPA pipeline (attention_dtype=None), MXFP8, MXFP4 and the fp4 O modes have no paged leg" in str(e.value)
    u8 = torch.zeros(1, dtype=torch.uint8, device=_DEV)
    with pytest.raises(NotImplementedError, match="paged_kv_page_size with quant=MxQuantSpec"):
        GatedAttentionBlockFwd(*a, quant=MxQuantSpec(descale_w_o=1.0), sample_h_sf=u8, sample_w_qkvg_sf=u8, **_decode_kw(4))
    with pytest.raises(NotImplementedError, match=r"fuse_norm_rope=True with QuantSpec\(attention_dtype=torch.bfloat16\): the fused FP8 projection fork"):
        GatedAttentionBlockFwd(*a, quant=_MODE, fuse_norm_rope=True, **_decode_kw(4))
    with pytest.raises(NotImplementedError, match="fuse_norm_rope=True with QuantSpec"):
        GatedAttentionBlockFwd(*a, quant=_MODE, fuse_norm_rope=True, fuse_norm_rope_small_m=True, **_decode_kw(4))
    with pytest.raises(NotImplementedError, match="fuse_norm_rope=True with QuantSpec"):
        GatedAttentionBlockFwd(*a, quant=_MODE, fuse_norm_rope=True, fuse_cache_write=True, **_decode_kw(4))
    with pytest.raises(NotImplementedError, match="fuse_cache_write=True"):
        GatedAttentionBlockFwd(*a, quant=_MODE, fuse_cache_write=True, **_decode_kw(4))
    with pytest.raises(ValueError, match="fuse_norm_rope_small_m=True: the knob decides"):
        GatedAttentionBlockFwd(*a, quant=_MODE, fuse_norm_rope_small_m=True, **_decode_kw(4))
    self_text = r"QuantSpec\(attention_dtype=torch.bfloat16\) with kv_source='self': the FP8-projection \+ bf16-attention mode serves the paged serving forms"
    with pytest.raises(NotImplementedError, match=self_text):
        GatedAttentionBlockFwd(*_decode_args(1, 64), quant=_MODE)
    with pytest.raises(NotImplementedError, match=self_text):
        GatedAttentionBlockFwd(*_decode_args(1, 64), quant=_MODE, paged_kv_page_size=16)
    t = 128
    mk = lambda *shape, dt=BF16: torch.zeros(*shape, dtype=dt, device=_DEV)  # noqa: E731
    packed = (
        mk(t, _D_MODEL, dt=E4),
        mk(_GEOM.n_qkvg, _D_MODEL, dt=E4),
        mk(_D),
        mk(_D),
        mk(t, _ROPE),
        mk(t, _ROPE),
        mk(_D_MODEL, 8 * _D, dt=E4),
        mk(t, _D_MODEL),
        _GEOM,
    )
    with pytest.raises(NotImplementedError, match=self_text):
        GatedAttentionBlockFwd(*packed, quant=_MODE, thd=True, num_sequences=2, max_seq_len=64)
    with pytest.raises(NotImplementedError, match=r"QuantSpec\(attention_dtype=torch.bfloat16\) with inplace_qkv=False: .* norms Q / K in place"):
        GatedAttentionBlockFwd(*a, quant=_MODE, inplace_qkv=False, **_decode_kw(4))
    with pytest.raises(NotImplementedError, match="paged_kv_page_size with save_for_backward=True"):
        GatedAttentionBlockFwd(*a, quant=_MODE, save_for_backward=True, inplace_qkv=False, **_decode_kw(4))
    with pytest.raises(NotImplementedError, match=r"save_for_backward=True / return_lse=True: the FP8-projection \+ bf16-attention mode is an INFERENCE form"):
        GatedAttentionBlockFwd(*_decode_args(1, 64), quant=_MODE, save_for_backward=True, return_lse=True, inplace_qkv=False)
    with pytest.raises(NotImplementedError, match="kv_source='paged' writes no LSE"):
        GatedAttentionBlockFwd(*a, quant=_MODE, return_lse=True, **_decode_kw(4))
    with pytest.raises(
        NotImplementedError, match=r"kv_cache_dtype=torch.float8_e4m3fn with quant=QuantSpec: .* the pools STAY bf16 .* declare kv_cache_dtype=None"
    ):
        GatedAttentionBlockFwd(*a, quant=_MODE, kv_cache_dtype=E4, **_decode_kw(4))
    with pytest.raises(NotImplementedError, match=r"use_pdl=True is served for the bf16 / f16 block only .* is declined too: its quantize launch carries no"):
        GatedAttentionBlockFwd(*a, quant=_MODE, use_pdl=True, **_decode_kw(4))
    assert GatedAttentionBlockFwd(*a, quant=_MODE, **_decode_kw(4)).use_pdl is False
    sg = GatedAttentionBlockGeometry(d_model=_D_MODEL, h_q=8, h_kv=2, d_head=_D, rope_dim=_ROPE, causal_bottom_right=True, qsa=QsaSpec())
    sa = list(_decode_args(4, 1))
    sa[1], sa[-1] = torch.zeros(sg.n_qkvg, _D_MODEL, dtype=E4, device=_DEV), sg
    with pytest.raises(
        NotImplementedError, match=r"QsaSpec with quant=QuantSpec: .* the FP8-projection \+ bf16-attention mode .* on the sparse forms is a follow-up"
    ):
        GatedAttentionBlockFwd(*sa, quant=_MODE, **_decode_kw(4))
    with pytest.raises(ValueError, match="FP8 needs both halves"):
        GatedAttentionBlockFwd(*_decode_args(4, 1, h_dtype=BF16), quant=_MODE, **_decode_kw(4))
    with pytest.raises(ValueError, match="FP8 needs both halves"):
        GatedAttentionBlockFwd(*_decode_args(4, 1), **_decode_kw(4))
    with pytest.raises(ValueError, match="is unused under attention_dtype=torch.bfloat16"):
        GatedAttentionBlockFwd(*a, quant=QuantSpec(0.01, 0.02, 0.03, 1.0, None, None, 4.0, E4, BF16), **_decode_kw(4))
    # f16 activations cannot be spelled under the mode: the activation dtype under quant is bf16, the attribute admits bf16 alone
    with pytest.raises(NotImplementedError, match="QuantSpec.attention_dtype must be None"):
        GatedAttentionBlockFwd(*a, quant=QuantSpec(0.01, 0.02, 0.03, None, None, None, 4.0, E4, torch.float16), **_decode_kw(4))
    # the resolver's own guard keeps the None spelling's text; the mode's spelling passes to the tiling rule
    blk = GatedAttentionBlockFwd(*a, quant=_MODE, fuse_cache_write_small_m=True, **_decode_kw(4))
    assert blk.cache_write_in_norm_rope
    with pytest.raises(NotImplementedError, match="serves the bf16 / f16 pipeline only, got quant=QuantSpec"):
        blk._resolve_cache_write_small_m(True, _FP8_SDPA)
    assert blk._resolve_cache_write_small_m(None, _FP8_SDPA) is False and blk._resolve_cache_write_small_m(None, _MODE) is True


def test_the_envelope_predicts_the_mode_on_both_readings():
    """``serving_envelope(..., quant=)``: the mode with e4m3 ``h`` on the Rubin reading serves both forms and both ``quant_modes`` name the
    spelling (so does ``quant=None`` with bf16 activations); ``QuantSpec(attention_dtype=None)`` and an ``MxQuantSpec`` read "no paged leg";
    the mode with 16-bit ``h`` reads the both-halves rule; e4m3 ``h`` without ``quant`` the dtype row, naming the way in; a QsaSpec geometry the
    sparse x quant row; the cc 10.0 reading follows ``FP8_PROJ_BF16_ATTENTION_CCS`` (the part statement and empty ``quant_modes`` while the
    tuple lacks the part; served once it holds it); ``pdl_default`` False under every quant spec."""
    from cudnn.frost import buffers

    env = GatedAttentionBlockFwd.serving_envelope
    e = env(_GEOM, E4, _DEV, device_cc=_SM107, quant=_MODE)
    if not e.cutedsl_ok:
        assert e.forms == () and e.decode.reason == e.cutedsl_reason
    import unittest.mock as um

    with um.patch.object(buffers, "_cutedsl_has_sm107", lambda: True):
        e = env(_GEOM, E4, _DEV, device_cc=_SM107, quant=_MODE)
        assert e.forms == ("extend", "decode") and e.decode.reason is None and e.extend.reason is None
        assert e.decode.quant_modes == e.extend.quant_modes == (QUANT_MODE_FP8_PROJ_BF16_ATTENTION,) and e.decode.pdl_default is False
        assert e.decode.dtypes == (E4,) == e.extend.dtypes and e.decode.kv_cache_dtypes == (BF16,), "under the mode: the e4m3 h in, bf16 pools"
        assert "FP8_PROJ_BF16_ATTENTION_CCS" in " ".join(e.decode.requires) + " ".join(e.extend.requires)
        plain = env(_GEOM, BF16, _DEV, device_cc=_SM107)
        assert plain.forms == ("extend", "decode") and plain.decode.quant_modes == (QUANT_MODE_FP8_PROJ_BF16_ATTENTION,) == plain.extend.quant_modes
        assert plain.decode.pdl_default is True, "the bf16 form's AUTO default on the measured part is untouched"
        none_spelling = env(_GEOM, E4, _DEV, device_cc=_SM107, quant=_FP8_SDPA)
        assert (
            none_spelling.forms == ()
            and "got quant=QuantSpec(attention_dtype=None) -- the fp8 d256 SDPA pipeline has no paged leg" in none_spelling.decode.reason
        )
        assert none_spelling.decode.quant_modes == (
            QUANT_MODE_FP8_PROJ_BF16_ATTENTION,
        ), "the served spellings are facts of the part, not of the QuantSpec asked about"
        mx = env(_GEOM, E4, _DEV, device_cc=_SM107, quant=MxQuantSpec(descale_w_o=1.0))
        assert mx.forms == () and "got quant=MxQuantSpec -- MXFP8, MXFP4 and the fp4 O modes have no paged leg" in mx.extend.reason
        half = env(_GEOM, BF16, _DEV, device_cc=_SM107, quant=_MODE)
        assert half.forms == () and "needs an e4m3 h (FP8 needs both halves" in half.decode.reason
        bare = env(_GEOM, E4, _DEV, device_cc=_SM107)
        assert bare.forms == () and "bf16 / f16" in bare.decode.reason and "serving_envelope(quant=)" in bare.decode.reason
        f32 = env(_GEOM, torch.float32, _DEV, device_cc=_SM107)
        assert f32.forms == () and "bf16 / f16" in f32.decode.reason
        sg = GatedAttentionBlockGeometry(d_model=_D_MODEL, h_q=8, h_kv=2, d_head=_D, rope_dim=_ROPE, causal_bottom_right=True, qsa=QsaSpec())
        sparse = env(sg, E4, _DEV, device_cc=_SM107, quant=_MODE)
        assert sparse.forms == () and "QsaSpec with quant=QuantSpec" in sparse.decode.reason and sparse.decode.quant_modes == () == sparse.extend.quant_modes
        assert env(sg, BF16, _DEV, device_cc=_SM107).decode.quant_modes == ()
        d128 = GatedAttentionBlockGeometry(d_model=_D_MODEL, h_q=8, h_kv=2, d_head=128, rope_dim=_ROPE, causal_bottom_right=True)
        assert env(d128, E4, _DEV, device_cc=_SM107, quant=_MODE).decode.quant_modes == (), "a geometry row the form declines: no spelling served"
        # the cc 10.0 reading
        sm100 = env(_GEOM, E4, _DEV, device_cc=_SM100, quant=_MODE)
        bf_sm100 = env(_GEOM, BF16, _DEV, device_cc=_SM100)
        assert bf_sm100.forms == ("extend", "decode")
        if _SM100 in FP8_PROJ_BF16_ATTENTION_CCS:
            assert sm100.forms == ("extend", "decode") and sm100.decode.quant_modes == (QUANT_MODE_FP8_PROJ_BF16_ATTENTION,) == bf_sm100.decode.quant_modes
        else:
            assert sm100.forms == () and sm100.decode.reason == _fp8_proj_bf16_attention_part_decline(_SM100) == sm100.extend.reason
            assert sm100.decode.quant_modes == () == sm100.extend.quant_modes == bf_sm100.decode.quant_modes == bf_sm100.extend.quant_modes
        # an unserved part: the dense part statement first, whatever the quant argument
        other = env(_GEOM, E4, _DEV, device_cc=(9, 0), quant=_MODE)
        assert other.forms == () and "SM90" in other.decode.reason and other.decode.quant_modes == ()


def test_the_envelope_validates_the_quant_spec_first_and_states_the_modes_dtypes():
    """``serving_envelope(quant=)`` runs ``QuantSpec.validate()`` FIRST, as the constructor does at declaration: an invalid spec -- a
    non-positive ``scale_o``, ``attention_dtype=torch.float16``, a ``scale_q`` set under the mode -- serves NEITHER form, each form's ``reason``
    carries the constructor's text (the very error the declaration raises), and ``quant_modes`` stays the part's fact; under a valid mode spec
    the records state the mode's input dtype (the e4m3 ``h``) and the DECODE pools' dtype (bf16), where ``quant=None`` keeps bf16 / f16 and the
    activation dtype as the pool dtype."""
    import unittest.mock as um

    from cudnn.frost import buffers

    env = GatedAttentionBlockFwd.serving_envelope
    zero = QuantSpec.bf16_attention(1.0, 1.0, 1.0, 0.0)
    f16 = QuantSpec(0.01, 0.02, 0.03, None, None, None, 4.0, E4, torch.float16)
    q_set = QuantSpec(0.01, 0.02, 0.03, 1.0, None, None, 4.0, E4, BF16)
    with um.patch.object(buffers, "_cutedsl_has_sm107", lambda: True):
        for cc in (_SM107, _SM100):
            bad = env(_GEOM, E4, _DEV, device_cc=cc, quant=zero)
            assert bad.forms == () and bad.decode.served is False and bad.extend.served is False
            assert bad.decode.reason == bad.extend.reason and bad.decode.reason.startswith("QuantSpec.validate(): ")
            assert "QuantSpec.scale_o must be a finite positive float, got 0.0" in bad.decode.reason
            assert (
                env(_GEOM, E4, _DEV, device_cc=cc, quant=f16).forms == ()
                and "QuantSpec.attention_dtype must be None" in env(_GEOM, E4, _DEV, device_cc=cc, quant=f16).decode.reason
            )
            assert "is unused under attention_dtype=torch.bfloat16" in env(_GEOM, E4, _DEV, device_cc=cc, quant=q_set).extend.reason
            if cc in FP8_PROJ_BF16_ATTENTION_CCS:
                assert bad.decode.quant_modes == (QUANT_MODE_FP8_PROJ_BF16_ATTENTION,) == bad.extend.quant_modes, "the served spellings are the part's fact"
            # the dtype rows
            good = env(_GEOM, E4, _DEV, device_cc=cc, quant=_MODE)
            assert good.decode.dtypes == (E4,) == good.extend.dtypes and good.decode.kv_cache_dtypes == (BF16,)
            plain = env(_GEOM, BF16, _DEV, device_cc=cc)
            assert plain.decode.dtypes == (BF16, torch.float16) == plain.extend.dtypes and plain.decode.kv_cache_dtypes == (BF16,)
            assert env(_GEOM, torch.float16, _DEV, device_cc=cc).decode.kv_cache_dtypes == (torch.float16,)
    # the constructor raises the same texts at declaration (the envelope's reason is the constructor's row, word for word)
    with pytest.raises(ValueError, match="QuantSpec.scale_o must be a finite positive float, got 0.0"):
        GatedAttentionBlockFwd(*_decode_args(4, 1), quant=zero, **_decode_kw(4))


def test_on_a_cc_10_0_reading_the_part_gate_reads_the_tuple(monkeypatch):
    """``check_support``'s part rows on a cc 10.0 reading: the mode is declined by the part statement while the tuple lacks the part (and
    passes the rows once it holds it); the fp8 SDPA spelling keeps its decline, which now names the mode's tuple."""
    monkeypatch.setattr(_api, "_device_cc_of", lambda device: _SM100)
    monkeypatch.setattr(_api, "_route_device_cc", lambda device: _SM100)
    blk = GatedAttentionBlockFwd(*_decode_args(4, 1), quant=_MODE, fuse_gate=False, **_decode_kw(4))
    blk._check_declaration()
    if _SM100 in FP8_PROJ_BF16_ATTENTION_CCS:
        blk._check_part_support()
    else:
        with pytest.raises(NotImplementedError, match=r"QuantSpec\(attention_dtype=torch.bfloat16\) on cc 10.0: .*FP8_PROJ_BF16_ATTENTION_CCS") as e:
            blk._check_part_support()
        assert str(e.value) == _fp8_proj_bf16_attention_part_decline(_SM100) and "the lift" in str(e.value)
    with pytest.raises(NotImplementedError, match="quant=QuantSpec on cc 10.0") as e:
        GatedAttentionBlockFwd(*_decode_args(1, 64), quant=_FP8_SDPA)._check_part_support()
    assert "FP8_PROJ_BF16_ATTENTION_CCS" in str(e.value) and "FP8-projection + bf16-attention" in str(e.value)
    with monkeypatch.context() as m:  # the one-tuple lift
        m.setattr(_api, "FP8_PROJ_BF16_ATTENTION_CCS", (_SM107, _SM100))
        blk._check_part_support()
        assert GatedAttentionBlockFwd.serving_envelope(_GEOM, E4, _DEV, device_cc=_SM100, quant=_MODE).decode.quant_modes == (
            QUANT_MODE_FP8_PROJ_BF16_ATTENTION,
        )


# ---------------------------------------------------------------------------
# Both served parts: the FP8 forced-tile projection GEMMs at the mode's shapes (the cell that admits a part into the tuple)
# ---------------------------------------------------------------------------

# (label, m, k, n): the two GEMMs of the mode at decode row counts and at 128 rows -- the 8 / 2 test geometry (n_qkvg 5120, h_q x D 2048 at
# d_model 512) and the 24 / 4 geometry at d_model 5120 (n_qkvg 14336, h_q x D 6144); every N a multiple of 256 (the forced tile's grid).
_GEMM_SHAPES = [
    pytest.param("qkv_gate_proj", 4, 512, 5120, id="qkv_8_2_m4"),
    pytest.param("out_proj", 4, 2048, 512, id="out_8_2_m4"),
    pytest.param("qkv_gate_proj", 128, 512, 5120, id="qkv_8_2_m128"),
    pytest.param("out_proj", 128, 2048, 512, id="out_8_2_m128"),
    pytest.param("qkv_gate_proj", 4, 5120, 14336, id="qkv_24_4_m4"),
    pytest.param("out_proj", 4, 6144, 5120, id="out_24_4_m4"),
]


@requires_dense_part
@pytest.mark.parametrize("label, m, k, n", _GEMM_SHAPES)
def test_fp8_forced_tile_projection_gemm_with_the_alpha_epilogue_runs_on_the_capture_safe_route(monkeypatch, label, m, k, n):
    """The mode's projection GEMM as the block builds it (e4m3 x e4m3 -> fp32 -> * alpha -> bf16, ``capture_safe=True``: the ``jit-only``
    route at the forced 256-wide tile, no ranked list, no cuDNN handle) with the opt-in flag UNSET, against the fp32 product of the
    DEQUANTIZED codes rounded once to bf16 (one bf16 rounding apart: ``rtol 2^-7``, ``atol 1e-3``; cos > 0.9999); the alpha scalar read
    in-kernel from a 1-element fp32 device tensor.  On cc 10.0 THIS cell's verdict is what admits the part into
    :data:`FP8_PROJ_BF16_ATTENTION_CCS`."""
    from cudnn.gated_attention_block.kernels import proj_gemm

    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    handles_before = dict(proj_gemm._GRAPH_HANDLES)
    plan = build_proj_gemm(m=m, k=k, n=n, dtype=E4, out_dtype=BF16, alpha=True, label=f"mode_{label}_{m}x{k}x{n}", capture_safe=True)
    assert plan.route == "jit-only" and plan.jit is not None and plan.capture_safe, (plan.route, plan.jit)
    assert plan.graph.plans == [], "the capture-safe plan built a ranked list"
    g = torch.Generator(device="cuda").manual_seed(m * 7 + n)
    a32 = torch.randn(m, k, generator=g, device="cuda", dtype=torch.float32)
    w32 = torch.randn(n, k, generator=g, device="cuda", dtype=torch.float32) * 0.02
    s_a, s_w = amax_scale(a32), amax_scale(w32)
    a8, w8 = quant_e4m3(a32, s_a), quant_e4m3(w32, s_w)
    alpha = (1.0 / s_a) * (1.0 / s_w)
    alpha_t = torch.full((1,), alpha, dtype=torch.float32, device="cuda")
    out = torch.full((m, n), float("nan"), device="cuda", dtype=BF16)
    ws = torch.empty(max(int(plan.workspace_bytes), 1), dtype=torch.uint8, device="cuda")
    run_proj_gemm(plan, a8, w8, out, ws, alpha=alpha_t)
    torch.cuda.synchronize()
    ref = ((a8.float() @ w8.float().t()) * alpha).to(BF16)
    assert torch.isfinite(out.float()).all(), label
    cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item()
    md = (out.float() - ref.float()).abs().max().item()
    print(
        f"[{label} {m}x{k}x{n} cc {_cc()}] fp8 forced-tile GEMM vs the dequantized fp32 product: cos {cos:.7f} max|diff| {md:.3e} tile {plan.tile_config_name}"
    )
    assert cos > 0.9999, (label, cos)
    torch.testing.assert_close(out.float(), ref.float(), rtol=2.0**-7, atol=1e-3)
    assert dict(proj_gemm._GRAPH_HANDLES) == handles_before, "the jit-only route created a cuDNN handle"
    # the alpha is read on device: a second scalar value through the same plan moves the output by that factor (one more rounding)
    alpha_t.fill_(alpha * 2.0)
    out2 = torch.full((m, n), float("nan"), device="cuda", dtype=BF16)
    run_proj_gemm(plan, a8, w8, out2, ws, alpha=alpha_t)
    torch.cuda.synchronize()
    torch.testing.assert_close(out2.float(), ((a8.float() @ w8.float().t()) * (alpha * 2.0)).to(BF16).float(), rtol=2.0**-7, atol=2e-3)
