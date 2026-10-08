# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fp4 WEIGHT MODES' BACKWARD of the gated attention block -- ``GatedAttentionBlockBwd(quant=MxQuantSpec(w_qkvg_dtype=e2m1 | o_fp4=...))``
over the MXFP8 training forward's record (byte for byte the MXFP8 record) and the caller's transposed e2m1 artifacts -- accept (Rubin) and
reject (any CUDA device) suite, the sibling of ``test_block_backward_mxfp8.py`` (whose helpers it imports by module name; nothing here
re-derives a bound, a shape or a stage list).

The five CONFIGURATIONS (one axis of the matrix) x the five CELLS of the MXFP8 matrix it reuses (ids ``<config>-<cell>``)::

    w4            MXFP4 W_qkvg (e2m1 W_qkvg [N, dm/2], e4m3 W_o): B8 on the MIXED e4m3 x e2m1 row over the packed e2m1 w_qkvg_t; the MXFP8 census
    o_nvfp4       NVFP4 W_o:  B2 on the NVFP4 x NVFP4 row over dy4 (the TWO-LEVEL cast of scale_dy x dY, e4m3 scales per 16) and w_o_t; +1 launch
    o_mxfp4       MXFP4 W_o:  B2 on the MIXED row over dy_mx8 (the MX-rowwise e4m3 dY, canonical E8M0 blob) and w_o_t; +1 launch
    w4_o_nvfp4    both (NVFP4 W_o)
    w4_o_mxfp4    both (MXFP4 W_o)

    s512_causal_b2-norm        THE bitwise / graph / CUPTI cell (GQA 8/2)
    s256_dense_b1-norm
    s992_causal_b1-norm        q- and kv-padded (the row's SF re-stagings)
    s1024_dense_b1_mha-norm    MHA
    s256_causal_b2_rope-rope_only

What is pinned, in tiers.  The RECORD: the fp4 training forward's ``out`` is bitwise the inference fp4 block's, its record under an fp4 W_o
alone is bitwise the MXFP8 training record over the same inputs (the record is written before the out projection), the record of ``both``
is bitwise the ``w4`` record, and every record keeps PRE-norm Q / K bands (the training-forward suite's two detectors, imported by name: the
saved rstd against the oracle norm of the slab's own Q band at the equal-input rstd bound, and the normed / rotated band NOT within the band
bound of the band itself).  The
BITWISE layer: ``dy8`` and ``og8`` (torch's casts; under an fp4 W_o ``og8`` against the kernels' bf16 O_gated, and the forward's ``o4`` /
``sf_o`` against the fp4 cast of the same bf16 words), the seven SDPA-layout payloads and blobs, ``q8 / sf_q`` and ``k8 / sf_k`` the forward's
bytes, the two canonical dQKVG casts, and the dY BLOCK point of the fp4 W_o arms -- ``dy_mx8 / sf_dy_mx`` bitwise the canonical rowwise
quantization of the block's bf16 dY (MXFP4), ``dy4 / sf_dy4`` bitwise ``fp4_quantize_rowwise_2d(dY, "nvfp4", global_scale=scale_dy)`` --
codes as uint8, the blob byte for byte -- with ``scale_dy`` READ BACK from the scalar block (NVFP4); the scalars (eight live, 21 dead at 0.0;
``scale_o = descale_o = descale_w_o = 1`` under ``o_fp4``) and ``delta``.  STAGE-LOCALISED: the dO composite (B2 + B3) under the bf16 bound on
fp64 of the operands dequantized THROUGH THE BLOBS (``fp4_dequant_rowwise_2d`` for every e2m1 side; under NVFP4 the bf16-rounded scaled
product descaled as B3 does), B1 / B7 / B8 under the GEMM bound through the blobs (B8 through the e2m1 ``w_qkvg_t`` under ``w4``), the SDPA
stage bitwise the row's own pre-pass and under the row's recipe against its once-rounded / fold-modelled references with the per-head
partials (the MXFP8 module's layer, unchanged: the SDPA stage's contract), ``dh / dW_o / dW_*_norm`` against the oracle SEEDED with the block's own dQ / dK / dV
under the bf16 block's bound and ``dW_qkvg`` in the row-budgeted form with its flip attribution (HYPOTHESIS on the MXFP8 precedent: switched
on from the first run; a miss is reported with its magnitude, never widened).  END TO END against the fold-modelled (M), the once-rounded (M)
and the (U) oracles -- the fp4 oracle arm dequantizes the e2m1 artifacts through their blobs and takes the two-level NVFP4 dY point --,
the (M) row budget asserted.  The EQUIVARIANCE pin on EVERY configuration: ``dy * 2^-13`` gives bitwise ``dy4 / sf_dy4`` (the two-level
cast absorbs the power of two; the single-level cast of the same tensor keeps about 0.004-0.006 % of the codes -- RED, shown on the
oracle in the same test), bitwise ``dy_mx8`` codes with the E8M0 bytes shifted by 13, every gradient bitwise ``2^-13 x`` the unit run's.
The ORIENTATION guard on every cell (the fp4 twin of the MXFP8 pin): the forward's ``w_qkvg_sf`` / ``w_o_sf`` handed as the transposed
artifact's blob passes every host check (the byte count is symmetric) and FAILS the B8 / dO bound, the right blob passes it.  The dy4 FLOOR
SHARES (all-zero 16-blocks, floor-hit scale bytes ``0x01``, saturated codes) printed per cell next to the single-level oracle's.  The
CUPTI census (``+ 1`` under an fp4 W_o), two executes bitwise under every knob set, CUDA-graph replay, a caller stream, the workspace
honest.  REJECTS (any CUDA device): every typed decline of the fp4 surface, matched by attribute name.

Margins -- FUNCTIONAL verdicts of the whole module (218 cells) on two Rubin parts, cc 10.7, SM clock locked at 2376 MHz: a 204-SM part
and a 212-SM part, two DATASETS of one tree (torch's Philox draws follow the SM count, so every cell's number moves between them; a third
part's miss at this layer is the dataset moving a cell before it is a defect -- re-measure before touching a form).  The seeded dh / dW_o
bound form and the (M) row budget were asserted from the first run (``_SEEDED_BF16_FORM_ASSERTED`` / ``_M_ROW_BUDGET_ASSERTED``, on the
MXFP8 precedent's 0.774 / 0.532) and held on both datasets, no bound touched.  Worst cell of each configuration as a fraction of the bound
named for it, 204-SM / 212-SM::

    configuration   seeded dh      seeded dW_o    seeded dW_qkvg   dW_q_norm      dW_k_norm      (M) row-budget use dh / dW_qkvg / dW_o
                    (asserted)     (asserted)     (row-budgeted)   (printed)      (printed)      (asserted; 204-SM | 212-SM)
    w4              0.744 / 0.851  0.509 / 0.223  2.230 / 1.106    0.157 / 0.167  0.135 / 0.165  0.00 / 0.06 / 0.00 | 0.00 / 0.08 / 0.00
    o_nvfp4         0.614 / 0.710  0.243 / 0.270  1.886 / 1.825    0.154 / 0.178  0.145 / 0.139  0.00 / 0.12 / 0.00 | 0.00 / 0.23 / 0.00
    o_mxfp4         0.644 / 0.951  0.243 / 0.270  1.778 / 1.587    0.147 / 0.179  0.153 / 0.151  0.00 / 0.15 / 0.00 | 0.00 / 0.15 / 0.00
    w4_o_nvfp4      0.630 / 0.893  0.238 / 0.302  2.014 / 1.676    0.161 / 0.165  0.130 / 0.174  0.00 / 0.06 / 0.00 | 0.00 / 0.15 / 0.00
    w4_o_mxfp4      0.756 / 0.809  0.238 / 0.302  1.743 / 1.260    0.163 / 0.158  0.134 / 0.162  0.00 / 0.15 / 0.00 | 0.00 / 0.11 / 0.00

The thin spot is the seeded ``dh`` on the 212-SM dataset: 0.951 of the bound at ``o_mxfp4-s256_dense_b1-norm`` (0.893 at
``w4_o_nvfp4-s1024_dense_b1_mha-norm``, 0.851 at ``w4-s1024_dense_b1_mha-norm``) against 0.756 at most on the 204-SM one
(``w4_o_mxfp4-s256_dense_b1-norm``) -- inside, not widened.  The seeded ``dW_qkvg`` above 1.0 is the row-budgeted output, every row outside
a near-amax ``dqkvg_t8`` code flip as on the MXFP8 row (``_assert_seeded_dw_qkvg_row_budgeted``).  The (M) fold-modelled oracle is inside
its row budget on every output of every cell -- at most 0.23 of the budget used (``dW_qkvg`` at ``o_nvfp4-s256_dense_b1-norm`` on the
212-SM dataset: 3 of 5120 rows against 13.1; 0.15 on the 204-SM one), cos >= 0.999991 (``dh`` at ``w4-s1024_dense_b1_mha-norm`` on both).
The dy4 floor shares at the test geometry (``scale_dy = 64``): 0 all-zero 16-blocks and 0 floor-hit scale bytes under either cast on
every NVFP4 cell, saturated codes 0.111-0.112 of the codes (identical under both casts: a power of two only shifts the exponent), while
the single-level cast of ``dy x 2^-13`` keeps 0.0038-0.0059 % of its codes on both datasets (0.0038 % on the oracle alone) -- the RED half
of the equivariance pin.  The record cells: ``saved.rstd_q`` against the oracle norm's rstd of the slab's own Q band at most 1.19e-7
relative (``_RSTD_EQUAL_INPUT_TOL``) on every qk_norm cell, the 20 record-bitwise cells (an fp4 W_o alone: bitwise the MXFP8 training
record; both modes: bitwise the ``w4`` record) bitwise on both datasets.  The 204-SM dataset's margins are bitwise reproducible: the
module's second run on that part (the convenience-wrapper fix in between) printed every figure above unchanged.
"""

import dataclasses
import gc
import inspect
import math
import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block fp4 backward tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

from cudnn.gated_attention_block import (  # noqa: E402
    Fp4Format,
    GatedAttentionBlockBwd,
    GatedAttentionBlockFwd,
    GatedAttentionBlockGeometry,
    MxQuantSpec,
    QuantSpec,
    gated_attention_block_backward,
)
from cudnn.gated_attention_block import api_bwd as _api_bwd  # noqa: E402
from cudnn.gated_attention_block.api import _view  # noqa: E402
from cudnn.gated_attention_block.kernels.proj_gemm import sf_blob_bytes  # noqa: E402

_E4M3 = torch.float8_e4m3fn
_FP4 = getattr(torch, "float4_e2m1fn_x2", None)
_E8M0 = getattr(torch, "float8_e8m0fnu", None)
if _FP4 is None or _E8M0 is None:
    pytest.skip(f"torch {torch.__version__} has no float4_e2m1fn_x2 / float8_e8m0fnu", allow_module_level=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_block_backward_mxfp8 as mx_bwd  # noqa: E402  -- the MXFP8 suite's helpers, by module name
from gated_block_reference import (  # noqa: E402
    RefGeometry,
    fp4_dequant_rowwise_2d,
    fp4_quantize_rowwise_2d,
    gated_attention_block_mxfp8_bwd_reference,
    make_inputs,
    mx_dequant_rowwise_2d,
    mx_swizzle_sf_rowwise_padded,
    mx_unswizzle_sf_rowwise,
    mxfp8_calibrated_scale_o,
    qk_norm_rope_reference,
    quantize_block_inputs_mxfp8,
    unpack_e2m1,
)
from gated_block_stream_probe import park_the_default_stream  # noqa: E402
from test_block_backward import _COMMON, _KNOBS, _alloc_grads, _assert_dw_norm_close, _assert_grad_close, _make_dy  # noqa: E402
from test_block_backward_fp8 import (  # noqa: E402
    _assert_e4m3_bitwise,
    _dev_scalar,
    _f32,
    _f32_mul,
    _gemm_bound,
    _grad_scale,
    _print_end_to_end,
    _report_stage_difference,
)
from test_block_training_forward import _BAND_TOL, _RSTD_EQUAL_INPUT_TOL, _alloc_saved  # noqa: E402

_SM107 = (10, 7)


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


requires_rubin = pytest.mark.requires_rubin
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

# The appended keyword-only tail of the quantized backward: the fp8 scalars, the MXFP8 artifacts, then the fp4 W_o artifacts (append-only).
_QUANT_INIT_KWARGS = ("quant", "grad_scaling")
_QUANT_EXECUTE_KWARGS = ("scale_dp", "scale_dy", "scale_do", "scale_dqkvg")
_MX_EXECUTE_KWARGS = ("h_t", "h_t_sf", "w_qkvg_t", "w_qkvg_t_sf")
_FP4_EXECUTE_KWARGS = ("w_o_t", "w_o_t_sf")
_FP4_ARTIFACTS = _MX_EXECUTE_KWARGS + _FP4_EXECUTE_KWARGS


def test_the_fp4_surface_is_an_appended_keyword_only_tail():
    """Host, no GPU: ``w_o_t`` / ``w_o_t_sf`` are the LAST two parameters of ``execute`` and of the convenience wrapper, after the MXFP8
    artifacts, keyword-only and defaulted (public signatures evolve append-only); ``__init__``'s tail is unchanged (the fp4 modes are fields
    of ``MxQuantSpec``, nothing appended)."""
    for fn, names in (
        (GatedAttentionBlockBwd.__init__, _QUANT_INIT_KWARGS),
        (GatedAttentionBlockBwd.execute, _QUANT_EXECUTE_KWARGS + _MX_EXECUTE_KWARGS + _FP4_EXECUTE_KWARGS),
        (gated_attention_block_backward, _QUANT_INIT_KWARGS + _QUANT_EXECUTE_KWARGS + _MX_EXECUTE_KWARGS + _FP4_EXECUTE_KWARGS),
    ):
        tail = list(inspect.signature(fn).parameters.values())[-len(names) :]
        assert [p.name for p in tail] == list(names), (fn.__qualname__, [p.name for p in tail])
        assert all(p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is not inspect.Parameter.empty for p in tail), fn.__qualname__
    sig = inspect.signature(_api_bwd._SigmoidGateBwd.__init__).parameters
    assert list(sig)[-1] == "want_dy_descale" and sig["want_dy_descale"].default is False
    assert list(inspect.signature(_api_bwd._SigmoidGateBwd.execute).parameters)[-1] == "descale_dy"
    ref_sig = inspect.signature(gated_attention_block_mxfp8_bwd_reference).parameters
    assert list(ref_sig)[-3:] == ["w_o_t", "w_o_t_sf", "o_fp4"]
    assert list(inspect.signature(fp4_quantize_rowwise_2d).parameters)[-1] == "global_scale"


# ---------------------------------------------------------------------------
# The matrix -- five configurations x five of the MXFP8 cells
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Cfg:
    name: str
    w_qkvg_fp4: bool
    o_fp4: Optional[Fp4Format]

    @property
    def w_dtype(self):
        return _FP4 if self.w_qkvg_fp4 else _E4M3

    @property
    def extra_launches(self) -> int:
        """The dY block quantize under an fp4 W_o (the one launch the fp4 arms add); an MXFP4 W_qkvg alone adds none."""
        return 1 if self.o_fp4 is not None else 0


_CFGS = (
    _Cfg("w4", True, None),
    _Cfg("o_nvfp4", False, Fp4Format.NVFP4),
    _Cfg("o_mxfp4", False, Fp4Format.MXFP4),
    _Cfg("w4_o_nvfp4", True, Fp4Format.NVFP4),
    _Cfg("w4_o_mxfp4", True, Fp4Format.MXFP4),
)
_CFG_BY_NAME = {c.name: c for c in _CFGS}
_NVFP4_CFGS = [c for c in _CFGS if c.o_fp4 is Fp4Format.NVFP4]
_O_CFGS = [c for c in _CFGS if c.o_fp4 is not None]
_CELL_IDS = ("s512_causal_b2-norm", "s256_dense_b1-norm", "s992_causal_b1-norm", "s1024_dense_b1_mha-norm", "s256_causal_b2_rope-rope_only")
_CELLS = [mx_bwd._BY_ID[i] for i in _CELL_IDS]
_THE_CELL = mx_bwd._BY_ID["s512_causal_b2-norm"]
_GRAPH_CELL = mx_bwd._BY_ID["s256_dense_b1-norm"]
_MATRIX = pytest.mark.parametrize("cfg,cell", [(c, k) for c in _CFGS for k in _CELLS], ids=[f"{c.name}-{k.id}" for c in _CFGS for k in _CELLS])
_CFG_AXIS = pytest.mark.parametrize("cfg", list(_CFGS), ids=[c.name for c in _CFGS])
_O_AXIS = pytest.mark.parametrize("cfg", _O_CFGS, ids=[c.name for c in _O_CFGS])
_NVFP4_MATRIX = pytest.mark.parametrize(
    "cfg,cell", [(c, k) for c in _NVFP4_CFGS for k in _CELLS], ids=[f"{c.name}-{k.id}" for c in _NVFP4_CFGS for k in _CELLS]
)
_KNOB_SETS = pytest.mark.parametrize("knobs", list(_KNOBS.values()), ids=list(_KNOBS))
# The seeded dh / dW_o bound and the (M) row budget: asserted from the first run on the MXFP8 precedent (0.774 / 0.532 of the bound; the
# (M) oracle inside the budget on every cell).  HYPOTHESIS for the fp4 arms -- a miss is reported with its magnitude, never widened.
_SEEDED_BF16_FORM_ASSERTED = True
_M_ROW_BUDGET_ASSERTED = True
_M_OUTPUTS = ("dh", "dw_qkvg", "dw_o")
_DY_SHIFT = mx_bwd._DY_SHIFT


def fp4_expected_launches(cfg: _Cfg, cell) -> int:
    """CUPTI kernel records of ONE execute under ``cfg`` at ``cell``: the MXFP8 expectation (COMPUTED by the MXFP8 module from the block's
    table and the row's terms) + the dY block quantize under an fp4 W_o; an MXFP4 W_qkvg alone changes nothing (B8 swaps its B operand)."""
    return mx_bwd.mxfp8_expected_launches(cell) + cfg.extra_launches


def fp4_launch_formula_from_facts(blk) -> int:
    """The MXFP8 formula from the adapter's facts + 1 under the block's own ``o_fp4`` (the stage list carries the dY block quantize)."""
    extra = 1 if blk.o_fp4 is not None else 0
    assert (blk._quant_dy_block is not None) == (extra == 1), "the dY block quantize exists exactly under an fp4 W_o"
    return mx_bwd.mxfp8_launch_formula_from_facts(blk) + extra


def test_fp4_launch_formula_reproduces_the_derivations():
    """Host, no GPU: 15 / 16 at the bitwise cell (w4 / an fp4 W_o), 15 / 16 RoPE-only, 14 / 15 MHA, 30 / 31 at the padded GQA cell --
    the MXFP8 chain's fused counts (its PROLOGUE / dual-axis dO / EPILOGUE launches) over the row's single-launch block-scale dQ, plus
    the dY block quantize under an fp4 W_o; the unfused chain over the per-member dQ read 28 / 29, 27 / 28, 24 / 25, 43 / 44, the fused
    chain over the per-member dQ 18 / 19, 18 / 19, 14 / 15, 33 / 34."""
    want = {
        ("w4", "s512_causal_b2-norm"): 15,
        ("o_nvfp4", "s512_causal_b2-norm"): 16,
        ("w4_o_mxfp4", "s512_causal_b2-norm"): 16,
        ("w4", "s256_causal_b2_rope-rope_only"): 15,
        ("o_mxfp4", "s256_causal_b2_rope-rope_only"): 16,
        ("w4", "s1024_dense_b1_mha-norm"): 14,
        ("w4_o_nvfp4", "s1024_dense_b1_mha-norm"): 15,
        ("w4", "s992_causal_b1-norm"): 30,
        ("o_nvfp4", "s992_causal_b1-norm"): 31,
    }
    for (cfg_name, cell_id), n in want.items():
        got = fp4_expected_launches(_CFG_BY_NAME[cfg_name], mx_bwd._BY_ID[cell_id])
        assert got == n, (cfg_name, cell_id, got, n)


# ---------------------------------------------------------------------------
# Building an fp4 backward: the fp4 training forward supplies the record exactly as a user would
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_tf32():
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def _e4m3_shadow(mx: dict) -> dict:
    """The forward suite's shadow: an e2m1 ``w_qkvg``'s VALUES as e4m3 codes (every E2M1 value is e4m3-exact) with the same blob, so the
    forward oracle's calibration (``mxfp8_calibrated_scale_o``) dequantizes exactly the bytes the mixed GEMM reads."""
    if mx["w_qkvg"].dtype != _FP4:
        return mx
    return dict(mx, w_qkvg=unpack_e2m1(mx["w_qkvg"].view(torch.uint8)).to(_E4M3))


def _fp4_inputs(cfg: _Cfg, cell, inp16: Optional[dict] = None):
    """``(inp16, mx, spec)``: the bf16 inputs, their quantized twin with the fp4 weights of ``cfg`` and the backward's six artifacts
    (``quantize_block_inputs_mxfp8(o_fp4=, w_qkvg_fp4=, backward=True)``, the convergence harness's contract), and the ``MxQuantSpec``
    (``scale_o`` calibrated as the MXFP8 suite does under an e4m3 W_o, 1.0 under ``o_fp4``)."""
    rg = RefGeometry(**cell.geom_kw)
    if inp16 is None:
        inp16 = dict(make_inputs(rg, batch=cell.b, seq_len=cell.s, dtype=torch.bfloat16))
    mx, desc = quantize_block_inputs_mxfp8(inp16, o_fp4=cfg.o_fp4, backward=True, w_qkvg_fp4=cfg.w_qkvg_fp4)
    scale_o = 1.0 if cfg.o_fp4 is not None else mxfp8_calibrated_scale_o(_e4m3_shadow(mx), rg)
    spec = MxQuantSpec(**desc, scale_o=scale_o, w_qkvg_dtype=cfg.w_dtype, o_fp4=cfg.o_fp4)
    return inp16, mx, spec


def _fwd_block(cfg: _Cfg, cell, mx: dict, spec: MxQuantSpec, out: torch.Tensor, *, save: bool) -> GatedAttentionBlockFwd:
    geom = GatedAttentionBlockGeometry(**cell.geom_kw)
    kw = dict(quant=spec, sample_h_sf=mx["h_sf"], sample_w_qkvg_sf=mx["w_qkvg_sf"])
    if cfg.o_fp4 is not None:
        kw["sample_w_o_sf"] = mx["w_o_sf"]
    if save:
        kw["save_for_backward"] = True
    return GatedAttentionBlockFwd(mx["h"], mx["w_qkvg"], mx["w_q_norm"], mx["w_k_norm"], mx["cos"], mx["sin"], mx["w_o"], out, geom, **kw)


def _fwd_execute(cfg: _Cfg, blk, mx: dict, out, ws, *, saved=None) -> None:
    kw = dict(h_sf=mx["h_sf"], w_qkvg_sf=mx["w_qkvg_sf"])
    if cfg.o_fp4 is not None:
        kw["w_o_sf"] = mx["w_o_sf"]
    blk.execute(mx["h"], mx["w_qkvg"], mx["w_q_norm"], mx["w_k_norm"], mx["cos"], mx["sin"], mx["w_o"], out, ws, saved=saved, **kw)


_FWD_MEMO: dict = {}


def _run_training_fp4(cfg: _Cfg, cell, *, inp16: Optional[dict] = None, memo: bool = True):
    """Declare, compile and run the fp4 TRAINING forward (``save_for_backward=True``); the record, the inputs and the workspace ride on the
    returned namespace (``_run_training_quant``'s shape).  Memoised per (configuration, cell) on the default dataset."""
    key = (cfg.name, cell.id)
    if memo and inp16 is None and key in _FWD_MEMO:
        return _FWD_MEMO[key]
    own_dataset = inp16 is None
    inp16, mx, spec = _fp4_inputs(cfg, cell, inp16)
    geom = GatedAttentionBlockGeometry(**cell.geom_kw)
    b, s = cell.b, cell.s
    out = torch.empty(b, s, geom.d_model, device="cuda", dtype=torch.bfloat16)
    blk = _fwd_block(cfg, cell, mx, spec, out, save=True)
    saved = _alloc_saved(geom, mx, b, s, save_mode="proj_slab", act_dtype=torch.bfloat16)
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    _fwd_execute(cfg, blk, mx, out, ws, saved=saved)
    torch.cuda.synchronize()
    r = SimpleNamespace(
        blk=blk, inp16=inp16, inp=mx, spec=spec, out=out, family="mxfp8", geom=geom, geom_kw=cell.geom_kw, batch=b, seq_len=s, saved=saved, ws=ws, cfg=cfg
    )
    if memo and own_dataset and key not in _FWD_MEMO:
        _FWD_MEMO[key] = r
    return r


def _execute_fp4(blk, inp, saved, dy, grads, ws, art, *, scale_dy=None, current_stream=None, **over):
    """One ``execute`` of an fp4-mode block: the record, the weights, the gradients, the workspace and the artifacts the block declared
    (the four MXFP8 ones; ``w_o_t`` / ``w_o_t_sf`` under ``o_fp4``), ``over`` replacing one (the wrong-orientation pin)."""
    kw = {k: v for k, v in dict(art, **over).items() if k in _FP4_ARTIFACTS and v is not None}
    if scale_dy is not None:
        kw["scale_dy"] = scale_dy
    blk.execute(
        dy,
        saved,
        inp["w_qkvg"],
        inp["w_q_norm"],
        inp["w_k_norm"],
        inp["cos"],
        inp["sin"],
        inp["w_o"],
        workspace=ws,
        current_stream=current_stream,
        **grads,
        **kw,
    )


def _artifacts_of(cfg: _Cfg, mx: dict) -> dict:
    """The six caller artifacts the builder produced (``h_t`` absent at a ragged T, ``w_o_t`` absent without ``o_fp4``)."""
    art = {k: mx.get(k) for k in _FP4_ARTIFACTS}
    assert (art["w_o_t"] is not None) == (cfg.o_fp4 is not None) and art["w_qkvg_t"].dtype == cfg.w_dtype
    return art


_MEMO: dict = {}


def _backward_fp4(cfg: _Cfg, cell, *, fwd=None, memo: bool = True, **bwd_kw):
    """Run the fp4 training forward (memoised), build the artifacts, declare / compile / run the backward, read the scalar block back."""
    key = (cfg.name, cell.id, tuple(sorted(bwd_kw.items())), id(fwd) if fwd is not None else None)
    if memo and key in _MEMO:
        return _MEMO[key]
    r = _run_training_fp4(cfg, cell) if fwd is None else fwd
    art = _artifacts_of(cfg, r.inp)
    dy = _make_dy(r.out)
    blk = mx_bwd._declare_mx_bwd(dy, r.saved, r.inp, r.geom, quant=r.spec, **bwd_kw)
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    grads = _alloc_grads(blk)
    _execute_fp4(blk, r.inp, r.saved, dy, grads, ws, art)
    torch.cuda.synchronize()
    res = SimpleNamespace(
        blk=blk,
        fwd=r,
        inp=r.inp,
        inp16=r.inp16,
        spec=r.spec,
        saved=r.saved,
        dy=dy,
        out=r.out,
        ws=ws,
        grads=grads,
        art=art,
        scale_dy_t=None,
        scalars={k: float(v.item()) for k, v in blk.quant_scalars(ws).items()},
        geom=r.geom,
        geom_kw=cell.geom_kw,
        batch=cell.b,
        seq_len=cell.s,
        grad_scaling="current",
        cfg=cfg,
        cell=cell,
    )
    if memo:
        _MEMO[key] = res
    return res


def _twin_fp4(res, *, poison=0xFF, art=None, dy=None, **bwd_kw):
    """A second block over the SAME record / inputs as ``res`` (other knobs, or another artifact set / dy on purpose), compiled and run once
    into a poisoned workspace and NaN-filled gradients; returns ``(blk, ws, grads)``."""
    need = dict(need_dh=res.blk.need_dh, need_dw_qkvg=res.blk.need_dw_qkvg, need_dw_o=res.blk.need_dw_o)
    need.update(bwd_kw)
    dy = res.dy if dy is None else dy
    blk = mx_bwd._declare_mx_bwd(dy, res.saved, res.inp, res.geom, quant=res.spec, **need)
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda").fill_(poison)
    grads = _alloc_grads(blk, fill=float("nan"))
    _execute_fp4(blk, res.inp, res.saved, dy, grads, ws, res.art if art is None else art)
    torch.cuda.synchronize()
    return blk, ws, grads


def _res_of(blk, ws, res, dy=None):
    """A result namespace over another block's workspace (the equivariance and orientation twins read it through ``_fp4_slots``)."""
    return SimpleNamespace(
        blk=blk,
        ws=ws,
        geom=res.geom,
        batch=res.batch,
        seq_len=res.seq_len,
        saved=res.saved,
        dy=res.dy if dy is None else dy,
        scalars={k: float(x.item()) for k, x in blk.quant_scalars(ws).items()},
        cfg=res.cfg,
        fwd=res.fwd,
    )


# ---------------------------------------------------------------------------
# Reading the block's intermediates back: the MXFP8 slots + the fp4 W_o arms' dY block point
# ---------------------------------------------------------------------------


def _fp4_slots(res) -> dict:
    """``_slots`` of the MXFP8 module plus the dY BLOCK point: ``dy_mx8`` / ``sf_dy_mx`` (MXFP4 W_o) or ``dy4`` / ``sf_dy4`` (NVFP4 W_o; the
    packed e2m1 codes as uint8), each a view of the region the carve appended last; None without ``o_fp4``."""
    v = mx_bwd._slots(res)
    blk, g = res.blk, res.geom
    lay = blk._layout()
    t, dm = blk.batch * blk.seq_len, g.d_model
    v["dy_mx8"] = _view(res.ws, lay.dy_mx8, (t, dm), _E4M3) if lay.dy_mx8 >= 0 else None
    v["sf_dy_mx"] = _view(res.ws, lay.sf_dy_mx, (sf_blob_bytes(t, dm),), torch.uint8) if lay.sf_dy_mx >= 0 else None
    v["dy4"] = _view(res.ws, lay.dy4, (t, dm // 2), torch.uint8) if lay.dy4 >= 0 else None
    v["sf_dy4"] = _view(res.ws, lay.sf_dy4, (sf_blob_bytes(t, dm, Fp4Format.NVFP4.block_size),), torch.uint8) if lay.sf_dy4 >= 0 else None
    o_fp4 = blk.o_fp4
    assert (v["dy_mx8"] is not None) == (o_fp4 is Fp4Format.MXFP4) and (v["dy4"] is not None) == (o_fp4 is Fp4Format.NVFP4)
    return v


def _forward_og16(res, v) -> torch.Tensor:
    """The kernels' bf16 ``O_gated`` over the record's O and GATE band (the forward's gate stage re-run): what both the per-tensor ``og8``
    and the forward's fp4 ``o4`` are cast from."""
    from test_sigmoid_gate_bwd import _forward_og

    g, t, d = res.geom, res.batch * res.seq_len, res.geom.d_head
    return _forward_og(res.saved.o.view(t, g.h_q, d), v["gate"], g.h_q, d)


def _assert_fp4_bitwise_layer(res) -> dict:
    """The BITWISE layer of a cell (module docstring): ``dy8``; ``og8`` against the kernels' bf16 O_gated (and against the forward's own ``o8``
    under an e4m3 W_o, the forward's ``o4`` / ``sf_o`` against the fp4 cast of the same bf16 words under an fp4 W_o); the seven SDPA-layout
    payloads and blobs; ``q8 / sf_q`` and ``k8 / sf_k`` the forward's bytes; the canonical dQKVG casts; the dY BLOCK point of the fp4 W_o
    arms; the scalars; ``delta``.  Returns the intermediates (``_fp4_slots``)."""
    blk, g, sc, sp, saved, cfg = res.blk, res.geom, res.scalars, res.spec, res.saved, res.cfg
    b, s, d = res.batch, res.seq_len, g.d_head
    t, hq, dm = b * s, g.h_q, g.d_model
    v = _fp4_slots(res)
    _assert_e4m3_bitwise(v["dy8"], res.dy, sc["scale_dy"], "dy8")
    flay = res.fwd.blk._layout()
    og16 = _forward_og16(res, v)
    if blk.need_dw_o:
        assert v["og8"] is not None
        _assert_e4m3_bitwise(v["og8"], og16, sp.scale_o, "og8 (vs the kernels' bf16 O_gated)")
        if flay.o8 >= 0:
            assert torch.equal(v["og8"].view(torch.uint8), _view(res.fwd.ws, flay.o8, (t, hq, d), _E4M3).view(torch.uint8)), "og8 vs the forward's o8"
    else:
        assert v["og8"] is None and v["lay"].og8 == -1
    if cfg.o_fp4 is not None:
        # the forward's fp4 tail: o4 / sf_o are the fp4 cast of the SAME bf16 O_gated the backward's og8 is cast from
        assert flay.o8 == -1 and flay.o4 >= 0 and flay.sf_o >= 0
        blk_sz = cfg.o_fp4.block_size
        packed, e = fp4_quantize_rowwise_2d(og16.reshape(t, hq * d).float(), cfg.o_fp4)
        o4 = _view(res.fwd.ws, flay.o4, (t, hq * d // 2), torch.uint8)
        sfo = _view(res.fwd.ws, flay.sf_o, (sf_blob_bytes(t, hq * d, blk_sz),), torch.uint8)
        n_bad = int((o4 != packed).sum())
        print(f"forward o4 vs the fp4 cast of the kernels' bf16 O_gated: {n_bad} of {o4.numel()} code bytes differ")
        assert n_bad == 0 and torch.equal(sfo, mx_swizzle_sf_rowwise_padded(e, blk_sz)), "the forward's o4 / sf_o are not the fp4 cast of the kernels' O_gated"
    # the SDPA-layout block quantizations, bitwise the torch quantization of their own sources
    mx_bwd._assert_mx_bitwise(v["do"], v["do8"], v["sf"]["sf_do"], "row", b=b, s=s, h=hq, what="do8 / sf_do (the gate backward's bf16 dO)")
    mx_bwd._assert_mx_bitwise(v["do"], v["do_T8"], v["sf"]["sf_do_T"], "col", b=b, s=s, h=hq, what="do_T8 / sf_do_T")
    mx_bwd._assert_mx_bitwise(v["rq"], v["q8"], v["sf"]["sf_q"], "row", b=b, s=s, h=hq, what="q8 / sf_q (the bf16 rebuild)")
    mx_bwd._assert_mx_bitwise(v["rq"], v["q_T8"], v["sf"]["sf_q_T"], "col", b=b, s=s, h=hq, what="q_T8 / sf_q_T")
    mx_bwd._assert_mx_bitwise(v["rk"], v["k8"], v["sf"]["sf_k"], "row", b=b, s=s, h=g.h_kv, what="k8 / sf_k")
    mx_bwd._assert_mx_bitwise(v["rk"], v["k_T8"], v["sf"]["sf_k_T"], "col", b=b, s=s, h=g.h_kv, what="k_T8 / sf_k_T")
    mx_bwd._assert_mx_bitwise(v["v_band"], v["v8"], v["sf"]["sf_v"], "row", b=b, s=s, h=g.h_kv, what="v8 / sf_v (the slab's V band, ROWWISE)")
    from cudnn.gated_attention_block.api import _sf_slot_bytes

    for name, sf_name, h in (("q8", "sf_q", hq), ("k8", "sf_k", g.h_kv)):
        fwd_bytes = _view(res.fwd.ws, getattr(flay, name), (t, h, d), _E4M3).view(torch.uint8)
        assert torch.equal(v[name].view(torch.uint8), fwd_bytes), f"{name}: the recomputed payload is not bitwise the forward's"
        assert torch.equal(v["sf"][sf_name], _view(res.fwd.ws, getattr(flay, sf_name), (_sf_slot_bytes(b, h, s, d),), torch.uint8)), sf_name
    # the GEMM-canonical block quantizations of dQKVG
    if blk.need_dh:
        mx_bwd._assert_canonical_bitwise(v["dqkvg"], v["dqkvg8"], v["sf_dqkvg"], "dqkvg8 / sf_dqkvg (rowwise over (T, N))")
    if blk.need_dw_qkvg:
        mx_bwd._assert_canonical_bitwise(v["dqkvg"].t(), v["dqkvg_t8"], v["sf_dqkvg_t"], "dqkvg_t8 / sf_dqkvg_t (transposed over (N, T))")
    # the dY BLOCK point of the fp4 W_o arms
    dy2d = res.dy.view(t, dm)
    if cfg.o_fp4 is Fp4Format.MXFP4:
        mx_bwd._assert_canonical_bitwise(dy2d, v["dy_mx8"], v["sf_dy_mx"], "dy_mx8 / sf_dy_mx (MX rowwise over (T, d_model))")
    elif cfg.o_fp4 is Fp4Format.NVFP4:
        packed, e = fp4_quantize_rowwise_2d(dy2d.float(), "nvfp4", global_scale=sc["scale_dy"])
        blob = mx_swizzle_sf_rowwise_padded(e, Fp4Format.NVFP4.block_size)
        n_bad_d, n_bad_sf = int((v["dy4"] != packed).sum()), int((v["sf_dy4"] != blob).sum())
        print(
            f"dy4 / sf_dy4 (the two-level NVFP4 cast at scale_dy={sc['scale_dy']:g}): {n_bad_d} of {packed.numel()} code bytes and {n_bad_sf} of {blob.numel()} scale bytes differ"
        )
        assert n_bad_d == 0, "dy4: the kernel's two-level NVFP4 codes differ from fp4_quantize_rowwise_2d(dY, nvfp4, global_scale=scale_dy)"
        assert n_bad_sf == 0, "sf_dy4: the kernel's canonical e4m3 scale bytes differ from the oracle's padded blob"
    # the scalars
    assert sc["amax_dy"] == res.dy.float().abs().max().item()
    assert sc["scale_dy"] == _grad_scale(sc["amax_dy"]) and sc["descale_dy"] == _f32(1.0 / sc["scale_dy"])
    assert sc["scale_o"] == _f32(sp.scale_o) and sc["descale_o"] == _f32(1.0 / sp.scale_o) and sc["descale_w_o"] == _f32(sp.descale_w_o)
    if cfg.o_fp4 is not None:
        assert sc["scale_o"] == sc["descale_o"] == sc["descale_w_o"] == 1.0, "no per-tensor scale on either side of an fp4 out projection"
    assert sc["alpha_b1"] == _f32_mul(sc["descale_dy"], sc["descale_o"]) and sc["alpha_b2"] == _f32_mul(sc["descale_dy"], sc["descale_w_o"])
    for name in mx_bwd._dead_slots():
        assert sc[name] == 0.0, f"{name}: a dead slot reads {sc[name]!r}, not 0.0"
    assert all(np.isfinite(sc[n]) for n in mx_bwd._MXFP8_LIVE_SLOTS)
    # delta
    from test_sigmoid_gate_bwd import _chain_dot_do_o

    delta = mx_bwd._delta(res)
    want = _chain_dot_do_o(saved.o, v["do"].view(b, s, hq, d))
    assert torch.equal(delta[..., :s], want[..., :s]) and torch.equal(
        delta[..., s:], torch.zeros_like(delta[..., s:])
    ), "delta is not bitwise the chain's dot_do_o"
    return v


def _w_o_t_deq64(res) -> torch.Tensor:
    """The caller's ``w_o_t`` dequantized THROUGH its blob in fp64 (``[H_q*D, d_model]``)."""
    return fp4_dequant_rowwise_2d(res.art["w_o_t"].view(torch.uint8), res.art["w_o_t_sf"], res.cfg.o_fp4, out_dtype=torch.float64)


def _do_composite_ref64(res, v: dict, *, w_o_t_deq=None) -> torch.Tensor:
    """fp64 of the dO the gate backward wrote, composed as the kernels do: B2 on the operands dequantized THROUGH THE BLOBS (the per-tensor
    ``dy8 . W_o8`` under an e4m3 W_o; ``dy_mx8 . w_o_t^T`` on the mixed row; ``dy4 . w_o_t^T`` on the NVFP4 row = ``scale_dy x`` the true
    value), rounded to bf16 as B2 stores it, the NVFP4 arm's descale as B3 applies it, then ``x sigmoid(gate)``."""
    g, sc, sp, cfg = res.geom, res.scalars, res.spec, res.cfg
    t, hq, d = res.batch * res.seq_len, g.h_q, g.d_head
    if cfg.o_fp4 is None:
        dog64 = (v["dy8"].double() * (1.0 / sc["scale_dy"])) @ (res.inp["w_o"].double() * sp.descale_w_o)
        dog16 = dog64.to(torch.bfloat16).double()
    else:
        w64 = _w_o_t_deq64(res) if w_o_t_deq is None else w_o_t_deq
        if cfg.o_fp4 is Fp4Format.MXFP4:
            a64 = mx_dequant_rowwise_2d(v["dy_mx8"], v["sf_dy_mx"]).double()
            dog16 = (a64 @ w64.t()).to(torch.bfloat16).double()
        else:
            a64 = fp4_dequant_rowwise_2d(v["dy4"], v["sf_dy4"], "nvfp4", out_dtype=torch.float64)  # = the dequantized scale_dy x dY
            dog16 = (a64 @ w64.t()).to(torch.bfloat16).double() * (1.0 / sc["scale_dy"])  # B2 stores bf16, B3 descales in fp32
    return dog16.view(t, hq, d) * torch.sigmoid(v["gate"].double())


# ---------------------------------------------------------------------------
# The oracles: the fp4 arm (the e2m1 artifacts through their blobs, the two-level NVFP4 dY point)
# ---------------------------------------------------------------------------


def _oracle(res, *, modelled: bool, seeded: Optional[dict] = None, fold: str = "kernel") -> dict:
    sc, cfg = res.scalars, res.cfg
    record = {}
    if modelled:
        v = mx_bwd._slots(res)
        record = dict(
            lse=res.saved.lse, o=res.saved.o, gate=v["gate"], do=v["do"],
            q8=v["q8"], sf_q=v["sf"]["sf_q"], q_T8=v["q_T8"], sf_q_T=v["sf"]["sf_q_T"],
            k8=v["k8"], sf_k=v["sf"]["sf_k"], k_T8=v["k_T8"], sf_k_T=v["sf"]["sf_k_T"],
            v8=v["v8"], sf_v=v["sf"]["sf_v"], do8=v["do8"], sf_do=v["sf"]["sf_do"], do_T8=v["do_T8"], sf_do_T=v["sf"]["sf_do_T"],
        )  # fmt: skip
    return gated_attention_block_mxfp8_bwd_reference(
        res.inp,
        RefGeometry(**res.geom_kw),
        res.spec,
        res.dy,
        scale_dy=sc["scale_dy"],
        delta=mx_bwd._delta(res)[..., : res.seq_len],
        modelled=modelled,
        seeded=seeded,
        fold=fold,
        **{k: res.art.get(k) for k in _MX_EXECUTE_KWARGS},
        w_o_t=res.art.get("w_o_t"),
        w_o_t_sf=res.art.get("w_o_t_sf"),
        o_fp4=cfg.o_fp4,
        **record,
    )


def _oracle_m(res, fold: str = "kernel") -> dict:
    cache = getattr(res, "oracle_m", None) or {}
    if fold not in cache:
        cache[fold] = _oracle(res, modelled=True, fold=fold)
        res.oracle_m = cache
    return cache[fold]


def _oracle_u(res) -> dict:
    return _oracle(res, modelled=False)


def _oracle_seeded(res) -> dict:
    v = mx_bwd._slots(res)
    b, s, g = res.batch, res.seq_len, res.geom
    seed = dict(dq=v["dq"].view(b, s, g.h_q, g.d_head), dk=v["dk"].view(b, s, g.h_kv, g.d_head), dv=v["dv"].view(b, s, g.h_kv, g.d_head))
    return _oracle(res, modelled=True, seeded=seed)


# ---------------------------------------------------------------------------
# ACCEPT (Rubin): the record
# ---------------------------------------------------------------------------


@requires_rubin
@_MATRIX
def test_fp4_training_forward_out_is_bitwise_the_inference_block_and_the_record_is_the_mxfp8_record(cfg, cell):
    """The fp4 TRAINING forward (``save_for_backward=True``): ``out`` bitwise the INFERENCE fp4 block's over the same inputs; the record
    fields finite and of the MXFP8 record's shapes; under an fp4 W_o alone the WHOLE record ``torch.equal`` the MXFP8 training record over
    the same bf16 inputs (the record is written before the out projection, the only stage the fp4 O mode changes), and under ``both`` it
    ``torch.equal`` the ``w4`` record; the slab's Q band is PRE-norm by the training-forward suite's two detectors (imported by name: the saved
    rstd against the oracle norm of the slab's own Q band at ``_RSTD_EQUAL_INPUT_TOL``, and the normed / rotated band not within ``_BAND_TOL``
    of the band -- a post-norm record would pass the first and fail the second)."""
    r = _run_training_fp4(cfg, cell)
    geom, b, s = r.geom, cell.b, cell.s
    out_inf = torch.empty_like(r.out)
    blk_inf = _fwd_block(cfg, cell, r.inp, r.spec, out_inf, save=False)
    blk_inf.check_support()
    blk_inf.compile()
    ws_inf = torch.empty(blk_inf.get_workspace_size(), dtype=torch.uint8, device="cuda")
    _fwd_execute(cfg, blk_inf, r.inp, out_inf, ws_inf)
    torch.cuda.synchronize()
    assert torch.equal(out_inf, r.out), f"{cfg.name}-{cell.id}: the training forward's out differs from the inference fp4 block's"
    sv = r.saved
    t, d = b * s, geom.d_head
    for name, ten in (("proj_slab", sv.proj_slab), ("o", sv.o), ("lse", sv.lse), ("rstd_q", sv.rstd_q), ("rstd_k", sv.rstd_k)):
        if ten is not None:
            assert torch.isfinite(ten.float()).all(), f"{name}: non-finite record cells"
    assert tuple(sv.proj_slab.shape) == (t, geom.n_qkvg) and tuple(sv.o.shape) == (b, s, geom.h_q, d) and tuple(sv.lse.shape) == (b, geom.h_q, s)
    # the record vs its MXFP8 / w4 sibling over the SAME bf16 inputs (built from r.inp16, the memoised dataset)
    sibling = _CFG_BY_NAME["w4"] if cfg.w_qkvg_fp4 else None
    if cfg.o_fp4 is not None:
        if sibling is None:
            mx8, desc8 = quantize_block_inputs_mxfp8(r.inp16)
            spec8 = MxQuantSpec(**desc8, scale_o=mxfp8_calibrated_scale_o(mx8, RefGeometry(**cell.geom_kw)))
            out8 = torch.empty_like(r.out)
            blk8 = GatedAttentionBlockFwd(
                mx8["h"], mx8["w_qkvg"], mx8["w_q_norm"], mx8["w_k_norm"], mx8["cos"], mx8["sin"], mx8["w_o"], out8, geom,
                quant=spec8, sample_h_sf=mx8["h_sf"], sample_w_qkvg_sf=mx8["w_qkvg_sf"], save_for_backward=True,
            )  # fmt: skip
            saved8 = _alloc_saved(geom, mx8, b, s, save_mode="proj_slab", act_dtype=torch.bfloat16)
            blk8.check_support()
            blk8.compile()
            ws8 = torch.empty(blk8.get_workspace_size(), dtype=torch.uint8, device="cuda")
            blk8.execute(
                mx8["h"],
                mx8["w_qkvg"],
                mx8["w_q_norm"],
                mx8["w_k_norm"],
                mx8["cos"],
                mx8["sin"],
                mx8["w_o"],
                out8,
                ws8,
                saved=saved8,
                h_sf=mx8["h_sf"],
                w_qkvg_sf=mx8["w_qkvg_sf"],
            )
            torch.cuda.synchronize()
            other, what = saved8, "the MXFP8 training record"
        else:
            other, what = _run_training_fp4(sibling, cell, inp16=r.inp16, memo=False).saved, "the w4 record"
        for name in ("proj_slab", "o", "lse", "rstd_q", "rstd_k"):
            a, c = getattr(sv, name), getattr(other, name)
            if a is not None:
                assert torch.equal(a, c), f"{cfg.name}-{cell.id}: saved.{name} is not bitwise {what}'s (the fp4 tail must not touch the record)"
        print(f"{cfg.name}-{cell.id}: the record is bitwise {what}")
    # pre-norm bands -- the training-forward suite's two detectors, by name: the normed / rotated Q band is NOT the band (a post-norm record
    # would already hold it), and the saved rstd IS the oracle norm's rstd of the slab's own Q band at the equal-input bound
    from cudnn.gated_attention_block.api import saved_slab_views

    q_pre, _gate, _k_pre, _v = saved_slab_views(sv.proj_slab, geom, b, s)
    qn_ref, rstd_ref = qk_norm_rope_reference(q_pre, r.inp["w_q_norm"], r.inp["cos"], r.inp["sin"], geom.rope_dim, geom.qk_norm_eps, qk_norm=geom.qk_norm)
    assert not torch.allclose(qn_ref.float(), q_pre.float(), **_BAND_TOL), "the saved Q band already IS the normed / rotated Q: the record is POST-norm"
    if geom.qk_norm:
        rel = ((rstd_ref - sv.rstd_q).abs() / rstd_ref).max().item()
        print(f"{cfg.name}-{cell.id}: saved.rstd_q vs the oracle norm's rstd of the slab's Q band: max rel {rel:.3g} (pre-norm bands)")
        torch.testing.assert_close(sv.rstd_q, rstd_ref, **_RSTD_EQUAL_INPUT_TOL)
    else:
        assert rstd_ref is None and sv.rstd_q is None and sv.rstd_k is None, "rope_only: no rstd anywhere"


# ---------------------------------------------------------------------------
# ACCEPT (Rubin): stage-localised bounds, end to end
# ---------------------------------------------------------------------------


@requires_rubin
@_MATRIX
def test_fp4_stage_localised_bounds(cfg, cell):
    """Every stage against the bound calibrated FOR IT (module docstring): the bitwise layer; the dO composite (B2 + B3) under the bf16
    bound on fp64 of the operands dequantized through the blobs; B1 under the GEMM bound; B7 / B8 through the blobs (B8 through the e2m1
    ``w_qkvg_t`` under ``w4``); the SDPA stage bitwise the row's own pre-pass and under the row's recipe (the MXFP8 layer); ``dh / dW_o /
    dW_*_norm`` against the seeded oracle under the bf16 bound, ``dW_qkvg`` row-budgeted with its flip attribution."""
    res = _backward_fp4(cfg, cell)
    blk, g, sc, sp = res.blk, res.geom, res.scalars, res.spec
    b, s, d = res.batch, res.seq_len, g.d_head
    t, grp = b * s, g.h_q // g.h_kv
    v = _assert_fp4_bitwise_layer(res)
    # --- the GEMMs on the block's own operands, through the blobs ---------------------------------------------------------------
    _assert_grad_close(
        v["do"], _do_composite_ref64(res, v), f"dO (B2 + B3 composite; B2 {'per-tensor e4m3' if cfg.o_fp4 is None else cfg.o_fp4.name + ' x e2m1 w_o_t'})"
    )
    if blk.need_dw_o:
        dy8_64 = v["dy8"].double() * (1.0 / sc["scale_dy"])
        og8_64 = v["og8"].double() * (1.0 / sp.scale_o)
        _gemm_bound(res.grads["dw_o"], dy8_64.t() @ og8_64.view(t, g.h_q * d), "B1 dW_o = dy8^T . og8 (alpha_b1)")
    if blk.need_dw_qkvg:
        a64 = mx_dequant_rowwise_2d(v["dqkvg_t8"], v["sf_dqkvg_t"]).double()
        b64 = mx_dequant_rowwise_2d(res.art["h_t"], res.art["h_t_sf"]).double()
        _gemm_bound(res.grads["dw_qkvg"], a64 @ b64.t(), "B7 dW_qkvg = dqkvg_t8 . h_t^T (block scales through the blobs)")
    if blk.need_dh:
        a64 = mx_dequant_rowwise_2d(v["dqkvg8"], v["sf_dqkvg"]).double()
        if cfg.w_qkvg_fp4:
            b64 = fp4_dequant_rowwise_2d(res.art["w_qkvg_t"].view(torch.uint8), res.art["w_qkvg_t_sf"], "mxfp4", out_dtype=torch.float64)
        else:
            b64 = mx_dequant_rowwise_2d(res.art["w_qkvg_t"], res.art["w_qkvg_t_sf"]).double()
        _gemm_bound(
            res.grads["dh"].view(t, g.d_model),
            a64 @ b64.t(),
            f"B8 dh = dqkvg8 . w_qkvg_t^T ({'e2m1' if cfg.w_qkvg_fp4 else 'e4m3'} w_qkvg_t through the blobs)",
        )
    # --- the SDPA stage: the MXFP8 layer, unchanged (its contract is the MXFP8 module's) ----------------------------------------------------------------------
    mx_bwd._assert_sdpa_stage_bitwise_the_rows_own_pre_pass(res, v)
    grad_tol, assert_close_fp8_grad = mx_bwd._row_tol()
    refs = mx_bwd._row_reference(res, v)
    for name, h, tag in (("dq", g.h_q, "dQ"), ("dk", g.h_kv, "dK"), ("dv", g.h_kv, "dV")):
        got = v[name].view(b, s, h, d).float()
        assert_close_fp8_grad(got, refs[name], grad_tol["atol"], grad_tol["rtol"], tag, keys=s, budget=1e-5)
        _report_stage_difference(v[name].view(b, s, h, d), refs[name], f"{tag} (once-rounded reference)")
    _assert_grad_close(v["dq"].view(b, s, g.h_q, d), refs["dq"].double(), "dQ vs the row's reference (the bf16 bound form)")
    _assert_grad_close(v["dk"].view(b, s, g.h_kv, d), refs["dk"].double(), "dK vs the once-rounded reference (the bf16 bound form)")
    _assert_grad_close(v["dv"].view(b, s, g.h_kv, d), refs["dv_fold"].double(), "dV vs the fold-modelled reference (the bf16 bound form)")
    if grp > 1:
        dk_part = mx_bwd._live_rows(mx_bwd._adapter_tensor(res, "dk_part"), s)
        dv_part = mx_bwd._live_rows(mx_bwd._adapter_tensor(res, "dv_part"), s)
        assert_close_fp8_grad(dk_part.float(), refs["dk_parts"], grad_tol["atol"], grad_tol["rtol"], "dk_part (per Q head)", keys=s, budget=1e-5)
        assert_close_fp8_grad(
            dv_part.float(), refs["dv_parts"].to(torch.bfloat16).float(), grad_tol["atol"], grad_tol["rtol"], "dv_part (per Q head, bf16)", keys=s, budget=1e-5
        )
    # --- downstream of B4: the SEEDED oracle (the fp4 arm) under the bf16 block's bound ------------------------------------------
    ref = _oracle_seeded(res)
    flip_ev = mx_bwd._report_seeded_intermediates(res, v, ref)
    worst = {}
    for name in ("dh", "dw_o"):
        if res.grads[name] is not None:
            if _SEEDED_BF16_FORM_ASSERTED:
                worst[name] = _assert_grad_close(res.grads[name], ref[name], f"{name} vs the seeded oracle")
            else:
                worst[name] = mx_bwd._report_close(res.grads[name], ref[name], f"{name} vs the seeded oracle (printed until the first run records it)")
    if res.grads["dw_qkvg"] is not None:
        worst["dw_qkvg"] = mx_bwd._assert_seeded_dw_qkvg_row_budgeted(res, v, ref, flip_ev, "dw_qkvg vs the seeded oracle")
    for name in ("dw_q_norm", "dw_k_norm"):
        if res.grads[name] is not None:
            worst[name] = _assert_dw_norm_close(res.grads[name], ref[name], ref[name + "_mass"], f"{name} vs the seeded oracle")
    print(f"{cfg.name}-{cell.id}: worst cells (fraction of the bound) {worst}")


@requires_rubin
@_MATRIX
def test_fp4_end_to_end_vs_the_oracles(cfg, cell):
    """End to end against the fold-modelled (M), the once-rounded (M) and the (U) oracles of the fp4 arm (the e2m1 artifacts dequantized
    through their blobs, the two-level NVFP4 dY point): ``cos``, ``max|diff| / max|ref|`` and the rows outside the bf16 bound PRINTED per
    gradient; finiteness pinned.  The (M) fold-modelled one is asserted by the row-budget test."""
    res = _backward_fp4(cfg, cell)
    for name, ten in res.grads.items():
        if ten is not None:
            assert torch.isfinite(ten).all(), f"{name}: non-finite cells"
    keys = mx_bwd._row_keys(res)
    m = _print_end_to_end(f"{cfg.name}-{cell.id} (M fold-modelled)", res.grads, _oracle_m(res, "kernel"), keys=keys)
    m1 = _print_end_to_end(f"{cfg.name}-{cell.id} (M once-rounded)", res.grads, _oracle_m(res, "once"), keys=keys)
    u = _print_end_to_end(f"{cfg.name}-{cell.id} (U)", res.grads, _oracle_u(res), keys=keys)
    assert m and m1 and u


@requires_rubin
@_MATRIX
def test_fp4_end_to_end_modelled_is_row_budgeted(cfg, cell):
    """The (M) end-to-end in the row-budgeted form (``1e-5 x rows x keys``, at least 1) against the FOLD-MODELLED fp4 oracle on every produced
    bf16 output -- asserted (``_M_ROW_BUDGET_ASSERTED``), never widened."""
    res = _backward_fp4(cfg, cell)
    m = _print_end_to_end(f"{cfg.name}-{cell.id} (M fold-modelled)", res.grads, _oracle_m(res, "kernel"), keys=mx_bwd._row_keys(res))
    assert m
    over = {n: (m[n]["rows_outside"], m[n]["rows"], m[n]["row_budget"]) for n in _M_OUTPUTS if n in m and m[n]["rows_outside"] > m[n]["row_budget"]}
    print(f"{cfg.name}-{cell.id}: (M) outputs over the row budget: {over or 'none'} (asserted: {_M_ROW_BUDGET_ASSERTED})")
    if _M_ROW_BUDGET_ASSERTED:
        assert not over, f"{cfg.name}-{cell.id}: (M) rows outside the bf16 bound exceed the row budget (rows outside, rows, budget): {over}"


# ---------------------------------------------------------------------------
# ACCEPT (Rubin): the equivariance pin (RED on the single-level cast), the floor shares, the orientation guard
# ---------------------------------------------------------------------------


@requires_rubin
@_MATRIX
def test_fp4_backward_is_bitwise_equivariant_under_a_power_of_two_dy_scaling(cfg, cell):
    """``dy * 2^-13``: ``dy8`` bitwise (``scale_dy`` absorbs the power of two), every dY-independent payload / blob bitwise, the dY-dependent
    MX casts keep their e4m3 codes with the E8M0 bytes shifted by 13 (``do8 / do_T8``, ``dqkvg8 / dqkvg_t8``, and ``dy_mx8`` under an MXFP4
    W_o), the two-level NVFP4 cast ``dy4 / sf_dy4`` BITWISE the unit run's (the pre-scale absorbs the power of two -- while the single-level
    cast of the same ``dy * 2^-13`` keeps about 0.004-0.006 % of its codes, RED: shown here on the oracle), every gradient bitwise
    ``2^-13 x`` the unit run's, the live scalars scaled."""
    res = _backward_fp4(cfg, cell)
    f = 2.0**-_DY_SHIFT
    dy2 = (res.dy.float() * f).to(torch.bfloat16)
    assert torch.equal(dy2.float(), res.dy.float() * f)
    blk2, ws2, grads2 = _twin_fp4(res, dy=dy2)
    res2 = _res_of(blk2, ws2, res, dy=dy2)
    v1, v2 = _fp4_slots(res), _fp4_slots(res2)
    assert torch.equal(v2["dy8"].view(torch.uint8), v1["dy8"].view(torch.uint8)), "dy8: scale_dy did not absorb the power of two"
    for name in ("q8", "q_T8", "k8", "k_T8", "v8"):
        assert torch.equal(v2[name].view(torch.uint8), v1[name].view(torch.uint8)), f"{name}: a dY-independent payload changed"
    for name in ("sf_q", "sf_q_T", "sf_k", "sf_k_T", "sf_v"):
        assert torch.equal(v2["sf"][name], v1["sf"][name]), f"{name}: a dY-independent blob changed"

    def shifted(unit, scaled, what):
        live = unit != 0
        assert torch.equal(scaled[~live], torch.zeros_like(scaled[~live])), f"{what}: a zero / pad scale byte became non-zero"
        diff = unit[live].int() - scaled[live].int()
        assert bool((diff == _DY_SHIFT).all()), f"{what}: E8M0 bytes did not shift by exactly {_DY_SHIFT} ({int((diff != _DY_SHIFT).sum())} bytes off)"

    for pay, sf_name in (("do8", "sf_do"), ("do_T8", "sf_do_T")):
        assert torch.equal(v2[pay].view(torch.uint8), v1[pay].view(torch.uint8)), f"{pay}: the e4m3 codes changed under a power-of-two dY"
        shifted(v1["sf"][sf_name], v2["sf"][sf_name], sf_name)
    if blk2.need_dh:
        assert torch.equal(v2["dqkvg8"].view(torch.uint8), v1["dqkvg8"].view(torch.uint8))
        shifted(v1["sf_dqkvg"], v2["sf_dqkvg"], "sf_dqkvg")
    if blk2.need_dw_qkvg:
        assert torch.equal(v2["dqkvg_t8"].view(torch.uint8), v1["dqkvg_t8"].view(torch.uint8))
        shifted(v1["sf_dqkvg_t"], v2["sf_dqkvg_t"], "sf_dqkvg_t")
    if cfg.o_fp4 is Fp4Format.MXFP4:
        assert torch.equal(v2["dy_mx8"].view(torch.uint8), v1["dy_mx8"].view(torch.uint8)), "dy_mx8: the codes changed"
        shifted(v1["sf_dy_mx"], v2["sf_dy_mx"], "sf_dy_mx")
    elif cfg.o_fp4 is Fp4Format.NVFP4:
        assert torch.equal(v2["dy4"], v1["dy4"]) and torch.equal(v2["sf_dy4"], v1["sf_dy4"]), "dy4 / sf_dy4: the two-level cast did not absorb the power of two"
        t, dm = res.batch * res.seq_len, res.geom.d_model
        single_unit = fp4_quantize_rowwise_2d(res.dy.view(t, dm).float(), "nvfp4")[0]
        single_small = fp4_quantize_rowwise_2d(dy2.view(t, dm).float(), "nvfp4")[0]
        live_small = (unpack_e2m1(single_small) != 0).float().mean().item()
        assert not torch.equal(single_small, single_unit), "the single-level cast must differ at 2^-13 (the floor) for this pin to have teeth"
        print(
            f"{cfg.name}-{cell.id}: the two-level kernel cast is bitwise under 2^-13; the SINGLE-level cast of the same dY keeps {live_small:.4%} live codes (RED)"
        )
    if v1["og8"] is not None:
        assert torch.equal(v2["og8"].view(torch.uint8), v1["og8"].view(torch.uint8))
    for name, ten in grads2.items():
        if ten is not None:
            want = (res.grads[name].float() * f).to(ten.dtype)
            assert torch.equal(ten, want), f"{name}: not bitwise 2^-{_DY_SHIFT} x the unit run's"
    sc1, sc2 = res.scalars, res2.scalars
    assert sc2["amax_dy"] == sc1["amax_dy"] * f and sc2["scale_dy"] == sc1["scale_dy"] * 2.0 ** _DY_SHIFT and sc2["descale_dy"] == sc1["descale_dy"] * f
    assert sc2["alpha_b1"] == sc1["alpha_b1"] * f and sc2["alpha_b2"] == sc1["alpha_b2"] * f
    for name in mx_bwd._dead_slots():
        assert sc2[name] == 0.0, name


def _nvfp4_shares(packed_u8: torch.Tensor, blob: torch.Tensor, rows: int, k: int) -> dict:
    """The dy4 floor detectors (the training harness's per-step log fields): the share of 16-blocks whose codes are ALL zero (by VALUE --
    a negative-zero nibble is dead), the share of FLOOR-HIT blocks (scale byte ``0x01`` = the e4m3 min-subnormal floor) and the share of
    saturated codes (``|value| == 6``)."""
    blk = Fp4Format.NVFP4.block_size
    sf = mx_unswizzle_sf_rowwise(blob, rows, k, blk)
    vals = unpack_e2m1(packed_u8).reshape(rows, k // blk, blk)
    return dict(
        all_zero_blocks=(vals == 0).all(-1).float().mean().item(),
        floor_hit_blocks=(sf == 0x01).float().mean().item(),
        saturated_codes=(vals.abs() == 6.0).float().mean().item(),
    )


@requires_rubin
@_NVFP4_MATRIX
def test_fp4_dy4_floor_shares_are_reported(cfg, cell):
    """The floor's detectors on the kernel's two-level ``dy4`` -- all-zero 16-blocks, floor-hit scale bytes, saturated codes -- printed per cell
    next to the SINGLE-level oracle cast's shares of the same dY (the training harness's log fields, exercised here).  Asserted: every share is
    a fraction, the two-level cast has no MORE all-zero blocks than the single-level one, and the two-level ``dy4`` IS the oracle's."""
    res = _backward_fp4(cfg, cell)
    v = _fp4_slots(res)
    t, dm = res.batch * res.seq_len, res.geom.d_model
    two = _nvfp4_shares(v["dy4"], v["sf_dy4"], t, dm)
    p1, e1 = fp4_quantize_rowwise_2d(res.dy.view(t, dm).float(), "nvfp4")
    one = _nvfp4_shares(p1, mx_swizzle_sf_rowwise_padded(e1, Fp4Format.NVFP4.block_size), t, dm)
    print(f"{cfg.name}-{cell.id}: dy4 shares -- two-level (kernel, scale_dy={res.scalars['scale_dy']:g}): {two}; single-level (oracle, no pre-scale): {one}")
    for d_ in (two, one):
        assert all(0.0 <= x <= 1.0 for x in d_.values()), d_
    assert two["all_zero_blocks"] <= one["all_zero_blocks"] + 1e-12, "the two-level cast zeroes MORE blocks than the single-level one"


def _spread_inputs(cell) -> dict:
    """The orientation pin's OWN dataset: a per-token / per-row POWER-OF-TWO spread (``h[t, :] *= 2^((t % 8) - 4)``, ``W_qkvg[n, :] *= 2^((n % 8)
    - 4)``, ``W_o[r, :] *= 2^((r % 8) - 4)``) on the bf16 inputs, so the scale bytes of a blob over a matrix and over its transpose differ at
    most positions BY CONSTRUCTION (the MXFP8 pin's device, extended to W_o's rows)."""
    rg, geom = RefGeometry(**cell.geom_kw), GatedAttentionBlockGeometry(**cell.geom_kw)
    inp16 = dict(make_inputs(rg, batch=cell.b, seq_len=cell.s, dtype=torch.bfloat16))
    t, dm, n = cell.b * cell.s, geom.d_model, geom.n_qkvg
    spread = lambda m: (2.0 ** ((torch.arange(m, device="cuda") % 8) - 4)).to(torch.float32)  # noqa: E731
    inp16["h"] = (inp16["h"].reshape(t, dm).float() * spread(t)[:, None]).to(torch.bfloat16).reshape(cell.b, cell.s, dm).contiguous()
    inp16["w_qkvg"] = (inp16["w_qkvg"].float() * spread(n)[:, None]).to(torch.bfloat16).contiguous()
    inp16["w_o"] = (inp16["w_o"].float() * spread(dm)[:, None]).to(torch.bfloat16).contiguous()
    return inp16


@requires_rubin
@_MATRIX
def test_fp4_wrong_orientation_blob_fails_the_gemm_bound(cfg, cell):
    """The orientation guard over the fp4 artifacts, on every cell and on the pin's own spread dataset: the byte count of a scale-factor blob
    is symmetric in ``(rows, K)``, so the forward's ``w_qkvg_sf`` handed as ``w_qkvg_t_sf`` (under ``w4``) and the forward's ``w_o_sf`` handed
    as ``w_o_t_sf`` (under an fp4 W_o) pass every host check -- and the B8 bound / the dO composite bound, whose REFERENCE dequantizes the
    artifact through the CORRECT blob while only the block was handed the wrong one, must FAIL; the control (the right blobs) passes first."""
    fwd = _run_training_fp4(cfg, cell, inp16=_spread_inputs(cell), memo=False)
    res = _backward_fp4(cfg, cell, fwd=fwd, memo=False)
    blk, g, art = res.blk, res.geom, res.art
    t = res.batch * res.seq_len
    v = _fp4_slots(res)
    checked = False
    if cfg.w_qkvg_fp4 and blk.need_dh:
        wrong = fwd.inp["w_qkvg_sf"]
        assert wrong.numel() == art["w_qkvg_t_sf"].numel() and not torch.equal(
            wrong, art["w_qkvg_t_sf"]
        ), "the un-transposed blob must have the same byte count and different bytes"
        a64 = mx_dequant_rowwise_2d(v["dqkvg8"], v["sf_dqkvg"]).double()
        b64 = fp4_dequant_rowwise_2d(art["w_qkvg_t"].view(torch.uint8), art["w_qkvg_t_sf"], "mxfp4", out_dtype=torch.float64)  # the CORRECT blob
        _gemm_bound(res.grads["dh"].view(t, g.d_model), a64 @ b64.t(), f"{cfg.name}-{cell.id}: B8 with the right w_qkvg_t_sf (control)")
        _blk, _ws, grads = _twin_fp4(res, art={**art, "w_qkvg_t_sf": wrong})
        with pytest.raises(AssertionError):
            _gemm_bound(grads["dh"].view(t, g.d_model), a64 @ b64.t(), f"{cfg.name}-{cell.id}: B8 with the forward's w_qkvg_sf as w_qkvg_t_sf (must FAIL)")
        checked = True
    if cfg.o_fp4 is not None:
        wrong = fwd.inp["w_o_sf"]
        assert wrong.numel() == art["w_o_t_sf"].numel() and not torch.equal(
            wrong, art["w_o_t_sf"]
        ), "the un-transposed W_o blob must have the same byte count and different bytes"
        ref64 = _do_composite_ref64(res, v)  # the CORRECT blob, in the reference
        _assert_grad_close(v["do"], ref64, f"{cfg.name}-{cell.id}: dO with the right w_o_t_sf (control)")
        blk2, ws2, _grads = _twin_fp4(res, art={**art, "w_o_t_sf": wrong})
        v2 = _fp4_slots(_res_of(blk2, ws2, res))
        with pytest.raises(AssertionError):
            _assert_grad_close(v2["do"], ref64, f"{cfg.name}-{cell.id}: dO with the forward's w_o_sf as w_o_t_sf (must FAIL)")
        checked = True
    assert checked


# ---------------------------------------------------------------------------
# ACCEPT (Rubin): census, determinism, graph, stream, workspace, wrapper
# ---------------------------------------------------------------------------


_CENSUS = [(c, _THE_CELL) for c in _CFGS] + [
    (_CFG_BY_NAME["w4_o_nvfp4"], mx_bwd._BY_ID["s1024_dense_b1_mha-norm"]),
    (_CFG_BY_NAME["o_mxfp4"], mx_bwd._BY_ID["s992_causal_b1-norm"]),
]


@requires_rubin
@pytest.mark.parametrize("cfg,cell", _CENSUS, ids=[f"{c.name}-{k.id}" for c, k in _CENSUS])
def test_fp4_launch_count_is_honest(cfg, cell):
    """CUPTI kernel records of one execute == the COMPUTED expectation (the MXFP8 one + 1 under an fp4 W_o) == the formula from the adapter's
    facts; no memcpy, no memset."""
    from torch.profiler import ProfilerActivity, profile

    res = _backward_fp4(cfg, cell)
    expected, formula = fp4_expected_launches(cfg, cell), fp4_launch_formula_from_facts(res.blk)
    grads = _alloc_grads(res.blk)
    _execute_fp4(res.blk, res.inp, res.saved, res.dy, grads, res.ws, res.art)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        _execute_fp4(res.blk, res.inp, res.saved, res.dy, grads, res.ws, res.art)
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    if not names:
        pytest.skip("torch.profiler recorded no CUDA events (CUPTI unavailable on this node); the launch count is unverified here")
    memsets = [n for n in names if "memset" in n.lower()]
    memcpys = [n for n in names if "memcpy" in n.lower()]
    kernels = [n for n in names if n not in memsets and n not in memcpys]
    print(
        f"\n{cfg.name}-{cell.id}: {len(kernels)} kernels (formula {formula}, expected {expected}), {len(memsets)} memsets, {len(memcpys)} memcpys:\n  "
        + "\n  ".join(names)
    )
    assert not memcpys and not memsets, (memcpys, memsets)
    assert formula == expected, (formula, expected)
    assert len(kernels) == expected, (len(kernels), expected, kernels)


@requires_rubin
@_CFG_AXIS
@_KNOB_SETS
def test_fp4_two_runs_are_bitwise(cfg, knobs):
    """Two executes of one block (the second into a poisoned workspace and NaN-filled gradients) and a FRESH block over the same record are
    ``torch.equal`` on every gradient and on the scalar block, under every knob set: no atomic anywhere (the fp4 quantize and the block-scale
    GEMMs are per element / deterministic)."""
    res = _backward_fp4(cfg, _THE_CELL, **knobs)
    sb1 = mx_bwd._scalar_block(res).clone()
    ws2 = torch.empty_like(res.ws).fill_(0xFF)
    grads2 = _alloc_grads(res.blk, fill=float("nan"))
    _execute_fp4(res.blk, res.inp, res.saved, res.dy, grads2, ws2, res.art)
    torch.cuda.synchronize()
    for name, ten in grads2.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), f"{name}: a second execute of the same block differs (knobs={knobs})"
    assert torch.equal(_view(ws2, res.blk._layout().quant_scalars, sb1.shape, torch.float32), sb1)
    blk, ws, grads = _twin_fp4(res, **knobs)
    for name, ten in grads.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), f"{name}: two runs differ (knobs={knobs})"
    assert torch.equal(_view(ws, blk._layout().quant_scalars, sb1.shape, torch.float32), sb1)


@requires_rubin
@_O_AXIS
def test_fp4_cuda_graph_capture_replays_bitwise(cfg):
    """One ``execute`` captured into a CUDA graph on a side stream replays bitwise the eager run and recomputes over a NEW ``dy`` and NEW
    artifact bytes (the e2m1 ``w_o_t`` and its blob included) written IN PLACE through the captured pointers; the capture launches nothing."""
    res = _backward_fp4(cfg, _GRAPH_CELL)
    blk = res.blk
    dy2 = res.dy.clone()
    art2 = {k: v.clone() for k, v in res.art.items() if v is not None}
    ws = torch.empty_like(res.ws)
    grads = _alloc_grads(blk, fill=float("nan"))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _execute_fp4(blk, res.inp, res.saved, dy2, grads, ws, art2)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    for ten in grads.values():
        if ten is not None:
            ten.fill_(float("nan"))
    ws.fill_(0xFF)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            _execute_fp4(blk, res.inp, res.saved, dy2, grads, ws, art2)
        torch.cuda.synchronize()
        for name, ten in grads.items():
            if ten is not None:
                assert torch.isnan(ten).all(), f"{name}: the capture launched work"
        graph.replay()
        torch.cuda.synchronize()
        for name, ten in grads.items():
            if ten is not None:
                assert torch.equal(ten, res.grads[name]), f"{name}: the replay differs from the eager run"
        # new inputs through the captured pointers: a new dy and the artifacts of a re-spread master weight (new e2m1 codes AND blob bytes)
        dy3 = _make_dy(res.out, seed=7)
        inp16b = dict(res.inp16)
        inp16b["w_o"] = (inp16b["w_o"].float() * 0.5).to(torch.bfloat16)
        inp16b["w_qkvg"] = (inp16b["w_qkvg"].float() * 0.5).to(torch.bfloat16)
        mx3, _ = quantize_block_inputs_mxfp8(inp16b, o_fp4=cfg.o_fp4, backward=True, w_qkvg_fp4=cfg.w_qkvg_fp4)
        art3 = {k: mx3[k] for k in art2}
        dy2.copy_(dy3)
        for k in art2:
            art2[k].view(torch.uint8).copy_(art3[k].view(torch.uint8))
        ws.fill_(0xFF)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        ref = _alloc_grads(blk, fill=float("nan"))
        ws_ref = torch.empty_like(ws).fill_(0xFF)
        _execute_fp4(blk, res.inp, res.saved, dy3, ref, ws_ref, art3)
        torch.cuda.synchronize()
        for name, ten in grads.items():
            if ten is not None:
                assert torch.isfinite(ten).all() and torch.equal(ten, ref[name]), f"{name}: the replay over new inputs differs from eager"
    finally:
        graph.reset()


@requires_rubin
@_O_AXIS
@pytest.mark.parametrize("how", ["ambient", "explicit"])
def test_fp4_a_caller_stream_orders_every_stage(cfg, how):
    """Every stage -- the dY block quantize and the block-scale out-projection dgrad included -- launches on ONE stream, the caller's
    (ambient or explicit): with the default stream parked and the workspace zeroed on the side stream right after the block, the gradients
    are bitwise the default-stream run's."""
    import cuda.bindings.driver as cuda_drv

    res = _backward_fp4(cfg, _GRAPH_CELL)
    side = torch.cuda.Stream()
    ws2 = torch.zeros_like(res.ws)
    grads2 = _alloc_grads(res.blk, fill=0)
    torch.cuda.synchronize()
    park_the_default_stream()
    with torch.cuda.stream(side):
        dy2 = res.dy.clone()
        art2 = {k: v.clone() for k, v in res.art.items() if v is not None}
        cs = None if how == "ambient" else cuda_drv.CUstream(side.cuda_stream)
        _execute_fp4(res.blk, res.inp, res.saved, dy2, grads2, ws2, art2, current_stream=cs)
        ws2.zero_()
    torch.cuda.synchronize()
    for name, ten in grads2.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), f"{name}: a stage escaped the caller's stream ({how})"


@requires_rubin
@_CFG_AXIS
def test_fp4_workspace_size_is_honest(cfg):
    """``get_workspace_size()`` is exact and never exceeded: the MXFP8 carve plus -- under an fp4 W_o -- the dY block pair appended LAST at
    256-B alignment (every MXFP8 offset unchanged against a block without ``o_fp4`` over the same record geometry); a buffer 4096 B larger
    keeps its tail untouched; two executes allocate nothing; every region is written in full (the fp4 ones by their bitwise equality with
    the oracle cast after a 0xFF poison)."""
    from cudnn.gated_attention_block.api import _WS_ALIGN

    res = _backward_fp4(cfg, _THE_CELL)
    blk, lay = res.blk, res.blk._layout()
    size = blk.get_workspace_size()
    assert size == lay.total_bytes and size % _WS_ALIGN == 0
    plans = blk.gemm_plans
    assert len(plans) == 4 and all(p.mma_tile_k_bytes == 64 for p in plans.values())
    assert plans["out_proj_wgrad"].has_alpha and plans["qkv_gate_wgrad"].block_scale and plans["qkv_gate_dgrad"].block_scale
    assert plans["out_proj_dgrad"].block_scale == (cfg.o_fp4 is not None) and plans["out_proj_dgrad"].has_alpha == (cfg.o_fp4 is None)
    if cfg.w_qkvg_fp4:
        assert plans["qkv_gate_dgrad"].w_dtype == _FP4
    if cfg.o_fp4 is not None:
        p = plans["out_proj_dgrad"]
        assert (p.dtype, p.w_dtype, p.block_size) == ((_FP4 if cfg.o_fp4 is Fp4Format.NVFP4 else _E4M3), _FP4, cfg.o_fp4.block_size)
    t, dm = blk.batch * blk.seq_len, blk.geom.d_model
    # the appended pair sits LAST, 256-B aligned, right after the MXFP8 carve's last region
    mx_last = max(lay.amax_partials + lay.amax_partials_n * 4, lay.quant_scalars + _api_bwd.QUANT_SCALARS_BYTES)
    if cfg.o_fp4 is Fp4Format.MXFP4:
        assert lay.dy_mx8 >= mx_last and lay.dy_mx8 % _WS_ALIGN == 0 and lay.sf_dy_mx > lay.dy_mx8 and lay.total_bytes >= lay.sf_dy_mx + sf_blob_bytes(t, dm)
        assert (lay.dy4, lay.sf_dy4) == (-1, -1)
    elif cfg.o_fp4 is Fp4Format.NVFP4:
        assert lay.dy4 >= mx_last and lay.dy4 % _WS_ALIGN == 0 and lay.sf_dy4 > lay.dy4 and lay.total_bytes >= lay.sf_dy4 + sf_blob_bytes(t, dm, 16)
        assert (lay.dy_mx8, lay.sf_dy_mx) == (-1, -1)
    else:
        assert (lay.dy_mx8, lay.sf_dy_mx, lay.dy4, lay.sf_dy4) == (-1, -1, -1, -1)
    # the MXFP8 offsets are untouched by the fp4 pair: the SAME declaration without o_fp4 carves them identically
    spec8 = dataclasses.replace(res.spec, o_fp4=None, descale_w_o=0.125) if cfg.o_fp4 is not None else res.spec
    lay8 = _api_bwd._plan_bwd_workspace(
        blk.geom, blk.batch, blk.seq_len, blk.act_dtype, blk.recompute,
        need=dict(dw_o=blk.need_dw_o, dw_norms=blk.need_dw_norms, dw_qkvg=blk.need_dw_qkvg),
        sdpa_bwd_bytes=lay.sdpa_bwd_bytes, gemm_scratch_bytes=lay.gemm_scratch_bytes, n_ctas_q=lay.n_ctas_q, n_ctas_k=lay.n_ctas_k,
        delta_shape=lay.delta_shape, quant=spec8, amax_partials_n=lay.amax_partials_n, mx_prologue_arm=blk.mx_prologue_arm,
    )  # fmt: skip
    for f_ in dataclasses.fields(lay):
        if f_.name not in ("dy_mx8", "sf_dy_mx", "dy4", "sf_dy4", "total_bytes"):
            assert getattr(lay, f_.name) == getattr(lay8, f_.name), f"{f_.name}: an MXFP8 offset moved under o_fp4"
    ws = torch.full((size + 4096,), 0xFF, dtype=torch.uint8, device="cuda")
    grads = _alloc_grads(blk)
    _execute_fp4(blk, res.inp, res.saved, res.dy, grads, ws[:size], res.art)
    torch.cuda.synchronize()
    gc.collect()
    live = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    n0 = torch.cuda.memory_stats()["allocation.all.allocated"]
    _execute_fp4(blk, res.inp, res.saved, res.dy, grads, ws[:size], res.art)
    _execute_fp4(blk, res.inp, res.saved, res.dy, grads, ws[:size], res.art)
    torch.cuda.synchronize()
    n1 = torch.cuda.memory_stats()["allocation.all.allocated"]
    peak = torch.cuda.max_memory_allocated()
    assert n1 == n0, f"the fp4 backward made {n1 - n0} CUDA allocation(s) on the execute path"
    assert peak <= live, f"a temporary on the execute path: the allocator peak rose from {live} to {peak} bytes"
    assert torch.equal(ws[size:], torch.full((4096,), 0xFF, dtype=torch.uint8, device="cuda")), "bytes past get_workspace_size() were written"
    for name, ten in grads.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), name
    # the fp4 regions written in full: bitwise the oracle cast after the poison (the whole bitwise layer, over this workspace)
    res_p = SimpleNamespace(**{**vars(res), "ws": ws[:size], "grads": grads})
    _assert_fp4_bitwise_layer(res_p)


@requires_rubin
@_CFG_AXIS
def test_fp4_convenience_wrapper_matches_the_class(cfg):
    """``gated_attention_block_backward(..., quant=<fp4 spec>, <the artifacts the derived needs read>)`` allocates, caches the compiled block
    (the artifacts' PRESENCE in the key) and delegates: every gradient it produces ``torch.equal`` the class path's (each gradient's chain is
    independent of the other needs, so a block declared over fewer needs computes the same bits).  The wrapper derives the needs from
    ``requires_grad``, and a packed e2m1 weight CARRIES it like any other tensor (only the fill-style factories are unimplemented for the
    dtype: ``torch.zeros(dtype=float4_e2m1fn_x2)`` is what raises, not ``requires_grad_`` on a ``.view``-created one), so every configuration
    asks the wrapper for an fp4 weight's gradient -- which the wrapper must size from the GEOMETRY (``(n_qkvg, d_model)`` / ``(d_model, H_q*D)``
    in ``dy``'s dtype): a packed weight's ``.shape`` is its storage ``[rows, K // 2]``, and an ``empty_like`` gradient would be half-width (the
    class's typed refusal).  The artifacts handed over are the six the needs read."""
    res = _backward_fp4(cfg, _GRAPH_CELL)
    inp, saved, g = res.inp, res.saved, res.geom
    flagged = (saved.h, inp["w_qkvg"], inp["w_o"], inp["w_q_norm"], inp["w_k_norm"])
    for t_ in flagged:
        t_.requires_grad_(True)  # the packed e2m1 weight(s) included: torch accepts it on a view-created tensor
    try:
        assert all(t_.requires_grad for t_ in flagged), [t_.dtype for t_ in flagged if not t_.requires_grad]
        fp4_weights = [n for n, t_ in (("w_qkvg", inp["w_qkvg"]), ("w_o", inp["w_o"])) if t_.dtype == _FP4]
        assert fp4_weights, cfg.name  # every configuration has a packed e2m1 weight carrying requires_grad
        art = dict(h_t=res.art["h_t"], h_t_sf=res.art["h_t_sf"], w_qkvg_t=res.art["w_qkvg_t"], w_qkvg_t_sf=res.art["w_qkvg_t_sf"])
        if cfg.o_fp4 is not None:
            art.update(w_o_t=res.art["w_o_t"], w_o_t_sf=res.art["w_o_t_sf"])
        out = gated_attention_block_backward(
            res.dy, saved, inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], g, quant=res.spec, **art
        )  # fmt: skip
        torch.cuda.synchronize()
        # the weight gradients at the LOGICAL shapes in dy's dtype -- never a packed weight's storage shape
        assert tuple(out["dw_qkvg"].shape) == (g.n_qkvg, g.d_model) and tuple(out["dw_o"].shape) == (g.d_model, g.h_q * g.d_head), (
            tuple(out["dw_qkvg"].shape),
            tuple(out["dw_o"].shape),
        )
        assert out["dh"].dtype == out["dw_qkvg"].dtype == out["dw_o"].dtype == res.dy.dtype
        for name in ("dh", "dw_qkvg", "dw_o", "dw_q_norm", "dw_k_norm"):
            assert out[name] is not None and torch.equal(out[name], res.grads[name]), name
        print(f"{cfg.name}: the wrapper produced every gradient from the requires_grad-derived needs; packed e2m1 weights carrying it: {fp4_weights}")
    finally:
        for t_ in flagged:
            t_.requires_grad_(False)


# ---------------------------------------------------------------------------
# REJECT (any CUDA device): a DECLARED block, the artifact checker, the forward's own declines -- attribute names only
# ---------------------------------------------------------------------------


def _placeholder_block(cfg: _Cfg, geom_kw=None, b: int = 2, s: int = 512, **over) -> GatedAttentionBlockBwd:
    """A DECLARED (not compiled) fp4-mode backward over placeholder tensors of the right dtypes and shapes -- no launch, any CUDA device."""
    from cudnn.gated_attention_block.api import SavedForBackward, saved_slab_views

    g = GatedAttentionBlockGeometry(**(geom_kw or {**_COMMON, "qk_norm": True, "is_causal": True}))
    t, dm, n, hd, d = b * s, g.d_model, g.n_qkvg, g.h_q * g.d_head, g.d_head
    dev = "cuda"
    spec = MxQuantSpec(descale_w_o=1.0 if cfg.o_fp4 else 0.125, scale_o=1.0, w_qkvg_dtype=cfg.w_dtype, o_fp4=cfg.o_fp4)
    h = torch.zeros(b, s, dm, dtype=_E4M3, device=dev)
    proj = torch.zeros(t, n, dtype=torch.bfloat16, device=dev)
    q_pre, gate, k_pre, _v = saved_slab_views(proj, g, b, s)
    saved = SavedForBackward(
        h=over.get("h", h), gate=gate, q_pre=q_pre, k_pre=k_pre, proj_slab=proj, o=torch.zeros(b, s, g.h_q, d, dtype=torch.bfloat16, device=dev),
        lse=torch.zeros(b, g.h_q, s, dtype=torch.float32, device=dev), rstd_q=torch.zeros(b, s, g.h_q, device=dev), rstd_k=torch.zeros(b, s, g.h_kv, device=dev),
    )  # fmt: skip
    w_qkvg = over.get(
        "w_qkvg", torch.zeros(n, dm // 2, dtype=torch.uint8, device=dev).view(_FP4) if cfg.w_qkvg_fp4 else torch.zeros(n, dm, dtype=_E4M3, device=dev)
    )
    w_o = over.get("w_o", torch.zeros(dm, hd // 2, dtype=torch.uint8, device=dev).view(_FP4) if cfg.o_fp4 else torch.zeros(dm, hd, dtype=_E4M3, device=dev))
    dy = torch.zeros(b, s, dm, dtype=torch.bfloat16, device=dev)
    wn = torch.ones(d, dtype=torch.bfloat16, device=dev)
    cos = torch.zeros(b, s, g.rope_dim, dtype=torch.bfloat16, device=dev)
    kw = {k: v for k, v in over.items() if k not in ("h", "w_qkvg", "w_o")}
    kw.setdefault("quant", spec)
    return GatedAttentionBlockBwd(dy, saved, w_qkvg, wn, wn.clone(), cos, cos.clone(), w_o, g, **kw)


def _good_artifacts(cfg: _Cfg, blk) -> dict:
    g, t = blk.geom, blk.batch * blk.seq_len
    dm, n, hd = g.d_model, g.n_qkvg, g.h_q * g.d_head
    dev = "cuda"
    art = dict(
        h_t=torch.zeros(dm, t, dtype=_E4M3, device=dev),
        h_t_sf=torch.zeros(sf_blob_bytes(dm, t), dtype=torch.uint8, device=dev),
        w_qkvg_t=torch.zeros(dm, n // 2, dtype=torch.uint8, device=dev).view(_FP4) if cfg.w_qkvg_fp4 else torch.zeros(dm, n, dtype=_E4M3, device=dev),
        w_qkvg_t_sf=torch.zeros(sf_blob_bytes(dm, n), dtype=torch.uint8, device=dev),
    )
    if cfg.o_fp4 is not None:
        art["w_o_t"] = torch.zeros(hd, dm // 2, dtype=torch.uint8, device=dev).view(_FP4)
        art["w_o_t_sf"] = torch.zeros(sf_blob_bytes(hd, dm, cfg.o_fp4.block_size), dtype=torch.uint8, device=dev)
    return art


def _reaches_the_rubin_gate(blk) -> None:
    """``check_support`` on a declared fp4 block passes every check before the device gate (Rubin: passes outright)."""
    try:
        blk.check_support()
    except NotImplementedError as e:
        assert _cc() != _SM107 and "Rubin" in str(e), str(e)


@requires_cuda
@_CFG_AXIS
def test_fp4_declaration_constructs_and_the_artifact_rejects_match_the_attribute_names(cfg):
    """Any CUDA device, no compile and no launch.  The five configurations DECLARE (no ``NotImplementedError`` on ``w_qkvg_dtype`` / ``o_fp4``:
    their backward is served) with the stage list of the MXFP8 backward plus the dY block quantize under an fp4 W_o, B2 / B8 keyed on the
    spec, the gate backward's dY descale arm on under NVFP4 only; ``check_support`` reaches the device gate.  Then the artifact checker, by
    attribute name: a ``w_qkvg_t`` of the other dtype (e4m3 under an e2m1 spec and the reverse), uint8 packed codes (the ``.view`` hint), a
    LOGICAL ``[dm, N]`` fp4 ``w_qkvg_t``, a ``.t()`` view, ``w_o_t`` / ``w_o_t_sf`` missing under ``o_fp4`` and given without, the OTHER format's
    ``w_o_t_sf`` (the byte-count decline), a right-sized blob of the wrong dtype, an e4m3 ``w_o_t``; and the weight gates both ways at
    ``check_support`` (an e4m3 ``w_qkvg`` under an e2m1 spec, an e2m1 one under an e4m3 spec, a uint8 / logical ``w_o`` under ``o_fp4``)."""
    blk = _placeholder_block(cfg)
    names = [type(st).__name__ for st in blk._stages]
    assert names.count("_QuantizeFp4") == (1 if cfg.o_fp4 is Fp4Format.NVFP4 else 0)
    # the fused MXFP8 chain: ONE standalone quantize stage (the dual-axis dO) + the dY block quantize under an MXFP4 W_o; 11 stages
    assert names.count("_QuantizeMxfp8") == 1 + (1 if cfg.o_fp4 is Fp4Format.MXFP4 else 0)
    assert len(names) == 11 + cfg.extra_launches and names[0] == "_MxQuantPrologue" and "_MxQuantEpilogue" in names
    if cfg.o_fp4 is not None:
        assert names[2] in ("_QuantizeFp4", "_QuantizeMxfp8") and names[3] == "_OutProjDgrad", names[:4]
    b2, b8 = blk._out_proj_dgrad, blk._qkv_gate_dgrad
    assert b2.block_scale == (cfg.o_fp4 is not None) and b8.block_scale and (b8.w_dtype == _FP4) == cfg.w_qkvg_fp4
    assert blk._gate_bwd.want_dy_descale == (cfg.o_fp4 is Fp4Format.NVFP4)
    if cfg.o_fp4 is Fp4Format.NVFP4:
        assert (b2.dtype, b2.w_dtype, b2.block_size) == (_FP4, _FP4, 16) and blk._quant_dy_block.scale_in is True
    elif cfg.o_fp4 is Fp4Format.MXFP4:
        assert (b2.dtype, b2.w_dtype, b2.block_size) == (_E4M3, _FP4, 32) and blk._quant_dy_block.sf_layout == "gemm"
    assert blk.w_qkvg_dtype == cfg.w_dtype and blk.w_o_dtype == (_FP4 if cfg.o_fp4 else _E4M3) and blk.o_fp4 is cfg.o_fp4
    _reaches_the_rubin_gate(blk)
    g, t = blk.geom, blk.batch * blk.seq_len
    dm, n, hd = g.d_model, g.n_qkvg, g.h_q * g.d_head
    dev = "cuda"
    art = _good_artifacts(cfg, blk)
    blk._check_artifacts(**art)  # the right set is accepted
    u8 = lambda *shape: torch.zeros(*shape, dtype=torch.uint8, device=dev)  # noqa: E731
    cases = []
    if cfg.w_qkvg_fp4:
        cases += [
            ("w_qkvg_t", dict(w_qkvg_t=torch.zeros(dm, n, dtype=_E4M3, device=dev))),  # e4m3 under an e2m1 spec
            ("w_qkvg_t", dict(w_qkvg_t=u8(dm, n // 2))),  # uint8: the .view hint
            ("w_qkvg_t", dict(w_qkvg_t=u8(dm, n).view(_FP4))),  # a LOGICAL [dm, N] fp4 tensor
            ("w_qkvg_t", dict(w_qkvg_t=u8(n // 2, dm).view(_FP4).t())),  # a .t() view
        ]
    else:
        cases += [("w_qkvg_t", dict(w_qkvg_t=u8(dm, n // 2).view(_FP4)))]  # e2m1 under an e4m3 spec
    if cfg.o_fp4 is not None:
        other = Fp4Format.MXFP4 if cfg.o_fp4 is Fp4Format.NVFP4 else Fp4Format.NVFP4
        cases += [
            ("w_o_t", dict(w_o_t=None)),
            ("w_o_t_sf", dict(w_o_t_sf=None)),
            ("w_o_t_sf", dict(w_o_t_sf=u8(sf_blob_bytes(hd, dm, other.block_size)))),  # the OTHER format's blob: a byte-count decline
            ("w_o_t_sf", dict(w_o_t_sf=u8(sf_blob_bytes(hd, dm, cfg.o_fp4.block_size)).view(other.sf_torch_dtype))),  # right size, wrong dtype
            ("w_o_t", dict(w_o_t=torch.zeros(hd, dm, dtype=_E4M3, device=dev))),
            ("w_o_t", dict(w_o_t=u8(hd, dm // 2))),
            ("w_o_t", dict(w_o_t=u8(hd, dm).view(_FP4))),
        ]
    else:
        cases += [
            ("w_o_t", dict(w_o_t=u8(hd, dm // 2).view(_FP4))),  # given without o_fp4
            ("w_o_t_sf", dict(w_o_t_sf=u8(sf_blob_bytes(hd, dm, 16)))),
        ]
    for name, over in cases:
        with pytest.raises(ValueError, match=name):
            blk._check_artifacts(**{**art, **over})
    # the weight gates at check_support, both ways, by the weight's name (the field is in the message)
    for name, over, needle in (
        ("w_qkvg", dict(w_qkvg=torch.zeros(n, dm, dtype=_E4M3, device=dev) if cfg.w_qkvg_fp4 else u8(n, dm // 2).view(_FP4)), "w_qkvg_dtype"),
        ("w_o", dict(w_o=u8(dm, hd // 2) if cfg.o_fp4 else u8(dm, hd // 2).view(_FP4)), "o_fp4"),
    ):
        blk2 = _placeholder_block(cfg, **over)
        with pytest.raises(ValueError, match=name) as ei:
            blk2.check_support()
        assert needle in str(ei.value), str(ei.value)
    if cfg.o_fp4 is not None:
        with pytest.raises(ValueError, match="w_o"):  # a LOGICAL [dm, HD] fp4 w_o
            _placeholder_block(cfg, w_o=u8(dm, hd).view(_FP4)).check_support()
    # an fp4 h is not served (h stays e4m3 in every fp4 mode)
    with pytest.raises(ValueError, match="saved.h"):
        _placeholder_block(cfg, h=u8(2, 512, dm // 2).view(_FP4)).check_support()


@requires_cuda
def test_fp4_modes_are_unspellable_on_a_quant_spec_and_the_forwards_own_declines_stand():
    """``o_fp4`` / ``w_qkvg_dtype`` do not exist on ``QuantSpec`` (``TypeError`` at construction: unrepresentable, not declined); the fp4 modes
    with a FUSED training forward stay the forward's own typed declines (``fuse_norm_rope`` + ``fuse_gate`` + ``save_for_backward``; an
    e2m1 ``W_qkvg`` with ``fuse_norm_rope``); ``B*S % 32 != 0`` with a projection weight gradient is the inherited MXFP8 decline; the NVFP4
    ``dY`` cast's own geometry gate accepts the block's ``d_head = 256`` and declines ``d_head % 64 != 0`` naming ``d_head`` (unreachable
    through the block, whose SDPA flavor pins ``d_head``)."""
    import test_block_fp4 as fwd_fp4

    with pytest.raises(TypeError):
        QuantSpec(scale_q=1.0, scale_k=1.0, scale_v=1.0, scale_o=1.0, descale_h=1.0, descale_w_qkvg=1.0, descale_w_o=1.0, o_fp4=Fp4Format.NVFP4)
    with pytest.raises(TypeError):
        QuantSpec(scale_q=1.0, scale_k=1.0, scale_v=1.0, scale_o=1.0, descale_h=1.0, descale_w_qkvg=1.0, descale_w_o=1.0, w_qkvg_dtype=_FP4)
    for fmt in Fp4Format:
        with pytest.raises(ValueError, match="incompatible with save_for_backward"):
            fwd_fp4._decl_block_fp4o(fmt, save_for_backward=True, **fwd_fp4._FUSED)
    with pytest.raises(NotImplementedError, match="fuse_norm_rope"):  # one fusion knob alone: the quantized pipelines' both-or-neither rule
        fwd_fp4._decl_block_fp4w(fuse_norm_rope=True, inplace_qkv=True)
    if not fwd_fp4._fork_has_fp4_arm():  # the fused projection fork is rendered for an e4m3 B (the forward suite's own pin, inverted when the arm lands)
        with pytest.raises(NotImplementedError, match="e4m3 B"):
            fwd_fp4._decl_block_fp4w(**fwd_fp4._FUSED).check_support()
    with pytest.raises(ValueError, match="need_dw_qkvg"):
        _placeholder_block(_CFG_BY_NAME["w4_o_nvfp4"], b=1, s=1000).check_support()
    # the NVFP4 dY cast's geometry gate (the kernel's host check the stage's check_support runs): d_head = 256 passes, one step off it is the
    # typed decline naming d_head -- the one reject of the fp4 surface the block cannot reach (its SDPA flavor pins d_head = 256)
    from cudnn.gated_attention_block.api import _QUANTIZE_FP4_THREADS
    from cudnn.gated_attention_block.kernels.quantize_fp4 import validate_shape

    validate_shape(256, _QUANTIZE_FP4_THREADS, Fp4Format.NVFP4.block_size)
    with pytest.raises(ValueError, match="d_head"):
        validate_shape(256 - 32, _QUANTIZE_FP4_THREADS, Fp4Format.NVFP4.block_size)


@requires_cuda
def test_the_fp4_matrix_declares_what_the_module_says():
    """Host: the five configurations are the fp4 weight modes'; every cell is one of the MXFP8 module's with the projection weight gradient and
    ``B*S % 32 == 0``; the NVFP4 configurations are exactly those with the two-level cast; the census list covers every configuration."""
    assert [c.name for c in _CFGS] == ["w4", "o_nvfp4", "o_mxfp4", "w4_o_nvfp4", "w4_o_mxfp4"]
    assert {c.o_fp4 for c in _CFGS} == {None, Fp4Format.NVFP4, Fp4Format.MXFP4} and sum(c.w_qkvg_fp4 for c in _CFGS) == 3
    for c in _CELLS:
        assert c.need_dw_qkvg and c.t % 32 == 0 and c.need_dw_o, c.id
    assert [c.name for c in _NVFP4_CFGS] == ["o_nvfp4", "w4_o_nvfp4"] and len(_O_CFGS) == 4
    assert {c.name for c, _ in _CENSUS} == {c.name for c in _CFGS}
    assert isinstance(_SEEDED_BF16_FORM_ASSERTED, bool) and isinstance(_M_ROW_BUDGET_ASSERTED, bool)
