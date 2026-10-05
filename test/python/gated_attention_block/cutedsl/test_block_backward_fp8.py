# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The per-tensor fp8 (e4m3) BACKWARD of the gated attention block -- ``GatedAttentionBlockBwd(quant=QuantSpec, ...)`` over the
quantized training forward's record -- accept (Rubin) and reject (any CUDA device) suite.

The ACCEPT cells run on Rubin over the real chain (the quantized backward's API, kernels, stages and oracle on one tree);
the REJECT cells are written against the declared API contract and run on any CUDA device.  The hypothesised bounds were
calibrated on the first Rubin run (the margins table at the end of this docstring): magnitudes are printed on every cell and
never widened -- a miss is reported with its magnitude and asked about (tolerances are sacred).

The accept matrix (one source: the cells table below, ids ``<shape>-<norm|rope_only>``), test geometry ``d_model 512, h_q 8,
h_kv 2, D 256, rope 64``::

    s256_causal_b1      S=256  causal B=1 GQA 8/2  norm + rope_only
    s512_causal_b2      S=512  causal B=2 GQA 8/2  norm + rope_only      THE bitwise / graph / CUPTI cell
    s992_causal_b1      S=992  causal B=1 GQA 8/2  norm                  S % 128 != 0: the adapter's padded launches
    s1000_causal_b2     S=1000 causal B=2 GQA 8/2  norm                  padded both sides (S % 256 != 0) at B = 2
    s1000_causal_b1     S=1000 causal B=1 GQA 8/2  norm                  T = 1000 WITH the weight gradients: a ragged K the MN-major wgrads serve
    s256_dense_b1       S=256  dense  B=1 GQA 8/2  norm + rope_only
    s1024_dense_b1_mha  S=1024 dense  B=1 MHA 8/8  norm                  no dK fold
    s512_dense_b2       S=512  dense  B=2 GQA 8/2  rope_only
    s512_causal_b1_mha  S=512  causal B=1 MHA 8/8  norm                  the row's MHA dK arm (EPI_QUANT, no fold)
    s256_causal_b2_rope S=256  causal B=2 GQA 8/2  rope_only             n_kv = 1 at B = 2 (phase drift)
    s512_causal_b2_calib S=512 causal B=2 GQA 8/2  norm                  a CALIBRATED scale_dp (the chart recipe)
    s1000_causal_b1_dgrad_only S=1000 causal B=1 GQA 8/2 norm            need_dw_o=False, need_dw_qkvg=False: the partial-need_* path

Stage-localised bounds -- none invented, each applied where it was calibrated: the quantizers and every derived scalar
BITWISE (torch's saturating RNE cast; the "current" scale formula ``grad_scale_from_amax``) -- ``og8`` against the KERNELS'
own bf16 ``O_gated``: the forward workspace's ``o8`` bytes (kernel vs kernel) and torch's cast of the forward gate stage's
``O * sigmoid(G)`` re-run over the same O / gate, because both kernels spell the sigmoid as the tanh identity (``tanh(g / 2)
/ 2 + 1 / 2``, one MUFU) whose last fp32 bit differs from torch's ``sigmoid`` on a handful of elements that sit on a bf16
rounding midpoint (21-83 of 2-4M codes at three cells of the first run); the torch-sigmoid composition is reported and every
differing code is required to sit on such an element; ``delta`` bitwise the SDPA chain's own ``dot_do_o`` over the same bf16
O / dO; the GEMMs on the block's own e4m3 operands under the GEMM suite's
bound (``rtol 2^-7``, ``atol = rtol * max|ref|``: an fp8 input is exact in fp64); the SDPA stage under the fp8 row's
recipe (``_FP8_GRAD_TOL`` atol 0.08 / rtol 0.2 with ``assert_close_fp8_grad``'s flip budget, ``amax_dP`` under
``_AMAX_DS_TOL``); ``dh / dW_*`` against the oracle SEEDED with the block's own bf16 dQ / dK / dV -- and fed the record's
exact LSE, bf16 pre-gate O and bf16 GATE band, the inputs the backward reads (the first full run fed the oracle's own fp64
attention O and exact projection instead: the fp8 forward's P cast then put dG at 1.2-3.1x and dW_o at 3.8-5.3x the bound on
every cell, and the gate's bf16 rounding alone still flipped 0.1-0.6 % of the og8 codes -- composition gaps of the reference,
not kernel margins) -- under the bf16 block's bound (``rtol 2^-6``, ``atol 2^-7 * max|ref|``,
``cos >= 0.999``; ``dW_norm``: ``2^-5 * mass + 1e-2 * |ref|``) -- a HYPOTHESIS until the first Rubin run; end-to-end against
the fully MODELLED oracle and the unquantized-gradient one is printed (``cos``, ``max|diff| / max|ref|``, the rows outside the
bf16 bound against the ``1e-5 x rows x keys`` row budget) and asserted only once the measured margin is known.

Launch count: ``expected_fp8_launches`` (host-checkable) -- the block's own launches (16 with every gradient: scalar init;
amax + quantize dY; B2; B3; quantize dO; B1; the Q / K recompute; the q8 / k8 / v8 quantizes; B5+B6; amax + quantize
dqkvg; B7; B8) + 1 dW_norm reduce under qk_norm + the fp8 row's ``fill_i32 + zero_amax + c * (2 + q) + fold dV + fold dK
(GQA) + 3 * q_padded + 2 * kv_padded + zero_ws`` -- 24 at ``s512_causal_b2-norm`` (c = 1, q = 1), 23 rope_only, 23 MHA,
29 padded (s992 and both s1000 cells with weight gradients), 27 dgrad-only padded.  No fold copy-outs on the fp8 row (its
folds write the caller's dK / dV).

Rejects match the ATTRIBUTE NAME only (``match="quant"``, ``"scale_dp"``, ``"thd"``, ...): the message prose is owned and
pinned by the API's own test module, so a wording change touches one test.

Known accept-matrix fact from the CPU calibration of the module constants (``FP8_SCALE_S_LOG2 = 8`` and
``FP8_GRAD_SCALE_MARGIN_LOG2 = 0`` both confirmed): at ``scale_dp = 1.0`` the row tolerance is near-vacuous for dQ / dK
(their ``max|ref|`` is below ``atol``, while 93-99 % of the fp32 dS values flush to zero in e4m3); the calibrated-``scale_dp``
cell is the one whose dQ / dK check has teeth -- ``_SCALE_DP_DEFAULT`` is the one constant to flip if the matrix should run
calibrated throughout.

Margins of the first full run (Rubin cc 10.7, 204 SMs, SM clock locked at 2376 MHz; worst cell as a fraction of the bound named
for that stage, the quantizers bitwise on every cell, ``amax_dP`` equal to the reference's ``max|dS|`` on every cell, the launch
counts 24 / 23 / 23 / 29 / 29 / 29 / 27 as predicted; "rows outside" = rows of dh (tokens) / dW (output rows) with a cell outside
the bf16 bound against the ``1e-5 x rows x keys`` row budget).  The seeded oracle is fed the record's LSE, O and gate band; the
five ``dw_qkvg`` cells above 1.0 are ONE near-amax ``dqkvg8`` code each (an e4m3 ulp there is ``32 / scale_dqkvg``) moving one
``dW_qkvg`` row by ``flip * h[t, :]`` -- 2-17 rows, inside the row budget on every cell -- and are left FAILING under the per-cell
bound until the bound's form is decided (never widened here)::

    cell                           dO    B1    B7    B8   bands dq_pre/dg/dk_pre  dqkvg8 flips  og8 flips  dh    dw_qkvg dw_o   dWq_n dWk_n  rows outside dh / dw_qkvg / dw_o (budget)
    s256_causal_b1-norm            0.245 0.213 0.204 0.187 0.122/0.193/0.115      12456         0          0.308 0.971   0.148  0.123 0.159  0/256 (13.1) / 0/5120 (13.1) / 0/512 (1.31)
    s256_causal_b1-rope_only       0.245 0.167 0.160 0.179 0.083/0.186/0.111      8142          0          0.159 0.677   0.124  -     -      0/256 (13.1) / 0/5120 (13.1) / 0/512 (1.31)
    s512_causal_b2-norm            0.245 0.170 0.184 0.155 0.149/0.201/0.118      51144         0          0.474 1.514   -      -     -      0/1024 (52.4) / 3/5120 (52.4) / 0/512 (5.24)
    s512_causal_b2-rope_only       0.245 0.201 0.194 0.176 0.099/0.201/0.086      52793         0          0.175 0.576   0.143  -     -      0/1024 (52.4) / 0/5120 (52.4) / 0/512 (5.24)
    s992_causal_b1-norm            0.245 0.215 0.212 0.202 0.149/0.211/0.130      50829         0          0.479 1.981   -      -     -      0/992 (50.8) / 8/5120 (50.8) / 0/512 (5.08)
    s1000_causal_b2-norm           0.137 0.168 0.182 0.217 0.136/0.159/0.126      101756        83         0.297 1.216   -      -     -      0/2000 (102) / 2/5120 (102) / 0/512 (10.2)
    s1000_causal_b1-norm           0.245 0.208 0.215 0.203 0.149/0.211/0.130      51150         0          0.476 2.015   -      -     -      0/1000 (51.2) / 8/5120 (51.2) / 0/512 (5.12)
    s256_dense_b1-norm             0.245 0.207 0.186 0.214 0.143/0.225/0.124      13470         0          0.688 0.919   0.145  0.172 0.163  0/256 (13.1) / 0/5120 (13.1) / 0/512 (1.31)
    s256_dense_b1-rope_only        0.245 0.169 0.197 0.187 0.126/0.203/0.126      8595          0          0.412 0.212   0.126  -     -      0/256 (13.1) / 0/5120 (13.1) / 0/512 (1.31)
    s1024_dense_b1_mha-norm        0.147 0.181 0.171 0.223 0.143/0.169/0.132      79685         51         0.609 0.647   0.310  0.105 0.129  0/1024 (83.9) / 0/8192 (83.9) / 0/512 (5.24)
    s512_dense_b2-rope_only        0.245 0.204 0.183 0.227 0.107/0.213/0.135      96377         0          0.536 0.165   0.143  -     -      0/1024 (52.4) / 0/5120 (52.4) / 0/512 (5.24)
    s512_causal_b1_mha-norm        0.147 0.201 0.202 0.156 0.127/0.243/0.161      33573         21         0.277 2.009   -      -     -      0/512 (41.9) / 17/8192 (41.9) / 0/512 (2.62)
    s256_causal_b2_rope-rope_only  0.245 0.213 0.199 0.164 0.094/0.193/0.080      16378         0          0.223 0.712   0.148  -     -      0/512 (26.2) / 0/5120 (26.2) / 0/512 (2.62)
    s512_causal_b2_calib-norm      0.245 0.170 0.184 0.156 0.149/0.201/0.114      49448         0          0.290 0.872   0.127  0.135 0.152  0/1024 (52.4) / 0/5120 (52.4) / 0/512 (5.24)
    s1000_causal_b1_dgrad_only-norm 0.245 -     -     0.203 0.149/0.211/0.130      51150         -          0.476 -       -      0.132 0.167  0/1000 (51.2) / - / -
"""

import dataclasses
import inspect
import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block fp8 backward tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

from cudnn.gated_attention_block import (  # noqa: E402
    GatedAttentionBlockBwd,
    GatedAttentionBlockGeometry,
    SavedForBackward,
    gated_attention_block_backward,
)
from cudnn.gated_attention_block import api_bwd as _api_bwd  # noqa: E402
from cudnn.gated_attention_block.api import MxQuantSpec, QuantSpec, _cols, _view  # noqa: E402
from cudnn.gated_attention_block.kernels import quantize as _quantize  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_reference import FP8_E4M3_MAX, RefGeometry, gated_attention_block_fp8_bwd_reference, quant_e4m3  # noqa: E402
from gated_block_stream_probe import park_the_default_stream  # noqa: E402
from test_block_backward import (  # noqa: E402
    _ATOL_FRAC,
    _COMMON,
    _COS_MIN,
    _KNOBS,
    _RTOL,
    _alloc_grads,
    _assert_dw_norm_close,
    _assert_grad_close,
    _cos,
    _make_dy,
)
from test_block_training_forward import _alloc_saved, _declare_quant, _dense_tail_declined, _dequantized_bf16_inputs, _run_training_quant  # noqa: E402

_SM107 = (10, 7)
_E4M3 = torch.float8_e4m3fn


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


# The REGISTERED marker of cutedsl/conftest.py (the skip is applied at collection).
requires_rubin = pytest.mark.requires_rubin
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

# ---------------------------------------------------------------------------
# The declaration surface this module is written against
# ---------------------------------------------------------------------------

# The appended keyword-only parameters of the quantized backward (append-only, defaulted, LAST).
_FP8_INIT_KWARGS = ("quant", "grad_scaling")
_FP8_EXECUTE_KWARGS = ("scale_dp", "scale_dy", "scale_do", "scale_dqkvg")


def test_the_fp8_surface_is_an_appended_keyword_only_tail():
    """Host, no GPU: the quantized backward's parameters are the LAST parameters of ``__init__``, ``execute`` and the convenience
    wrapper, keyword-only and defaulted, in the declared order (public signatures evolve append-only)."""
    for fn, names in (
        (GatedAttentionBlockBwd.__init__, _FP8_INIT_KWARGS),
        (GatedAttentionBlockBwd.execute, _FP8_EXECUTE_KWARGS),
        (gated_attention_block_backward, _FP8_INIT_KWARGS + _FP8_EXECUTE_KWARGS),
    ):
        tail = list(inspect.signature(fn).parameters.values())[-len(names) :]
        assert [p.name for p in tail] == list(names), (fn.__qualname__, [p.name for p in tail])
        assert all(p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is not inspect.Parameter.empty for p in tail), fn.__qualname__


def _api_const(name: str):
    """A module constant of ``api_bwd`` (``FP8_SCALE_S_LOG2``, ``QUANT_SCALAR_SLOTS``, ...) read at CALL time, never re-literalled here."""
    return getattr(_api_bwd, name)


# ---------------------------------------------------------------------------
# The accept matrix -- named once
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Cell:
    shape: str
    s: int
    causal: bool
    b: int
    h_kv: int
    qk_norm: bool
    need_dw_o: bool = True
    need_dw_qkvg: bool = True
    scale_dp: object = "default"  # a float, "default" (= _SCALE_DP_DEFAULT) or "calibrated" (the chart recipe)
    note: str = ""

    @property
    def id(self) -> str:
        return f"{self.shape}-{'norm' if self.qk_norm else 'rope_only'}"

    @property
    def geom_kw(self) -> dict:
        return {**_COMMON, "h_kv": self.h_kv, "qk_norm": self.qk_norm, "is_causal": self.causal}

    @property
    def bwd_kw(self) -> dict:
        kw = {}
        if not self.need_dw_o:
            kw["need_dw_o"] = False
        if not self.need_dw_qkvg:
            kw["need_dw_qkvg"] = False
        return kw

    @property
    def group(self) -> int:
        return _COMMON["h_q"] // self.h_kv


def _both(shape, s, causal, b, h_kv, **kw):
    return [_Cell(shape, s, causal, b, h_kv, True, **kw), _Cell(shape, s, causal, b, h_kv, False, **kw)]


_CELLS = (
    _both("s256_causal_b1", 256, True, 1, 2)
    + _both("s512_causal_b2", 512, True, 2, 2, note="THE bitwise / graph / CUPTI cell")
    + [_Cell("s992_causal_b1", 992, True, 1, 2, True, note="S % 128 != 0: the adapter's padded launches (992 % 16 == 0)")]
    + [_Cell("s1000_causal_b2", 1000, True, 2, 2, True, note="S % 128 != 0 and S % 256 != 0 at B = 2: padded on both sides")]
    + [
        _Cell(
            "s1000_causal_b1", 1000, True, 1, 2, True, note="T = 1000 WITH the weight gradients: a ragged token count the MN-major wgrads serve (no B*S rule)"
        )
    ]
    + _both("s256_dense_b1", 256, False, 1, 2)
    + [_Cell("s1024_dense_b1_mha", 1024, False, 1, 8, True, note="MHA: no dK fold")]
    + [_Cell("s512_dense_b2", 512, False, 2, 2, False)]
    + [_Cell("s512_causal_b1_mha", 512, True, 1, 8, True, note="the row's MHA dK arm under causal: EPI_QUANT into the caller's dK, no fold")]
    + [_Cell("s256_causal_b2_rope", 256, True, 2, 2, False, note="rope_only x causal (no reduce launch), B = 2 at n_kv = 1")]
    + [_Cell("s512_causal_b2_calib", 512, True, 2, 2, True, scale_dp="calibrated", note="the bitwise cell's geometry at the chart recipe's scale_dp")]
    + [
        _Cell(
            "s1000_causal_b1_dgrad_only",
            1000,
            True,
            1,
            2,
            True,
            need_dw_o=False,
            need_dw_qkvg=False,
            note="the partial-need_* path under quant: dw_* None, og8 == -1, B1 / B7 absent, no side stream",
        )
    ]
)
_BY_ID = {c.id: c for c in _CELLS}
assert len(_BY_ID) == len(_CELLS) == 15, "the matrix ids must be unique: 12 rows, three of them in both qk_norm arms"
_MATRIX = pytest.mark.parametrize("cell", _CELLS, ids=[c.id for c in _CELLS])
_BITWISE_CELL = _BY_ID["s512_causal_b2-norm"]
_SCALE_DP_DEFAULT = 1.0  # the matrix's default scale_dp (module docstring: the calibrated regime is the one with teeth for dQ / dK)
_KNOB_SETS = pytest.mark.parametrize("knobs", list(_KNOBS.values()), ids=list(_KNOBS))


# ---------------------------------------------------------------------------
# The launch-count formula (host-checkable; the CUPTI cell asserts formula == expected == len(kernels))
# ---------------------------------------------------------------------------


def fp8_block_launch_table(*, qk_norm: bool, need_dw_o: bool = True, need_dw_qkvg: bool = True, need_dh: bool = True) -> list:
    """The block's OWN launches under ``quant``, in launch order, as ``(label, count, present)`` -- the module docstring's
    table as data, so the count is derived from it and never re-literalled."""
    return [
        ("init_scalars", 1, True),
        ("amax dY", 1, True),
        ("quantize dY", 1, True),
        ("B2 out_proj dgrad", 1, True),
        ("B3 sigmoid_gate_bwd (fp8 arm: dO, dG, og8, delta, amax_dO)", 1, True),
        ("quantize dO", 1, True),
        ("B1 out_proj wgrad", 1, need_dw_o),
        ("recompute Q, K (qk_norm_rope)", 1, True),
        ("quantize q8 / k8 / v8", 3, True),
        ("B5+B6 qk_norm_rope_bwd", 1, True),
        ("dW_norm reduce", 1, qk_norm),
        ("amax dqkvg", 1, True),
        ("quantize dqkvg", 1, True),
        ("B7 qkv_gate wgrad", 1, need_dw_qkvg),
        ("B8 qkv_gate dgrad", 1, need_dh),
    ]


def fp8_row_launches(*, group: int, chunks: int, dq_launches: int, q_padded: bool, kv_padded: bool, zero_ws: bool) -> int:
    """The fp8 SDPA row's launches under an EXTERNAL delta: ``fill_i32`` (the uniform kv length, unconditional on a dense
    plan) + ``_zero_amax`` + per head chunk ``main + dK + dq_launches x dQ`` + the dV fold (always) + the dK fold (GQA only)
    + the staging pads (3 on the q side: q, dO, lse; 2 on the kv side: k, v) + the dS zero-fill when the adapter says so.
    NO ``dot_do_o`` (the caller's delta replaces it) and NO fold copy-outs (the folds write the caller's dK / dV)."""
    return 1 + 1 + chunks * (2 + dq_launches) + 1 + (1 if group > 1 else 0) + (3 if q_padded else 0) + (2 if kv_padded else 0) + (1 if zero_ws else 0)


def expected_fp8_launches(
    *,
    qk_norm: bool,
    group: int,
    chunks: int = 1,
    dq_launches: int = 1,
    q_padded: bool = False,
    kv_padded: bool = False,
    zero_ws: bool = False,
    need_dw_o: bool = True,
    need_dw_qkvg: bool = True,
    need_dh: bool = True,
) -> int:
    """CUPTI kernel records of ONE execute of the fp8 backward = the block's table + the row's launches."""
    block = sum(n for _label, n, present in fp8_block_launch_table(qk_norm=qk_norm, need_dw_o=need_dw_o, need_dw_qkvg=need_dw_qkvg, need_dh=need_dh) if present)
    return block + fp8_row_launches(group=group, chunks=chunks, dq_launches=dq_launches, q_padded=q_padded, kv_padded=kv_padded, zero_ws=zero_ws)


def fp8_launch_formula_from_facts(blk) -> int:
    """The formula recomputed from the ADAPTER's own facts (chunks, the dQ rendering's ``b_head_group``, padding, zero-fill)
    and the block's declaration -- so a change on either side is visible (the bf16 suite's pattern)."""
    from cudnn.sdpa.bwd.kernels.sm107.prepared_host import _dq_launches

    g, impl = blk.geom, blk._sdpa._impl
    assert impl.external_delta is True, "the quantized backward ALWAYS hands the row the gate backward's delta"
    grp = g.h_q // g.h_kv
    chunks = (blk.batch // impl._b_chunk) * (g.h_q // impl._qh_chunk)
    return expected_fp8_launches(
        qk_norm=g.qk_norm,
        group=grp,
        chunks=chunks,
        dq_launches=_dq_launches(grp, impl._dq_b_head_group),
        q_padded=bool(impl._q_padded),
        kv_padded=bool(impl._kv_padded),
        zero_ws=bool(impl._zero_ws),
        need_dw_o=blk.need_dw_o,
        need_dw_qkvg=blk.need_dw_qkvg,
        need_dh=blk.need_dh,
    )


def _padded(cell: _Cell) -> tuple:
    """The adapter's staging facts from the shape alone: q side at 128-row tiles, kv side at 256-row blocks."""
    return cell.s % 128 != 0, cell.s % 256 != 0


# (cell id, expected count): the predictions every CUPTI cell checks the formula AND the profiler against.
_LAUNCH_EXPECTATIONS = [
    ("s512_causal_b2-norm", 24),
    ("s512_causal_b2-rope_only", 23),
    ("s512_causal_b1_mha-norm", 23),
    ("s992_causal_b1-norm", 29),
    ("s1000_causal_b2-norm", 29),
    ("s1000_causal_b1-norm", 29),
    ("s1000_causal_b1_dgrad_only-norm", 27),
]


def test_fp8_launch_formula_reproduces_the_declared_counts():
    """Host, no GPU: the formula over the matrix cells' facts (c = 1, one dQ launch per chunk under the single-launch dQ
    rendering, no zero-fill at the plain causal / dense cells) gives the declared counts -- 24 at the bitwise cell (norm,
    GQA), 23 rope_only, 23 MHA, 29 at the three padded GQA cells with weight gradients (+3 q pads, +2 kv pads), 27 at the padded dgrad-only cell
    (B1 and B7 gone) -- and 28 at a padded MHA shape (no matrix cell runs it: the prediction is pinned here only).  The
    block's own table sums to 16 with every gradient, 17 with the reduce; the row adds 7 (GQA) / 6 (MHA) unpadded."""
    for cell_id, want in _LAUNCH_EXPECTATIONS:
        c = _BY_ID[cell_id]
        q_pad, kv_pad = _padded(c)
        got = expected_fp8_launches(qk_norm=c.qk_norm, group=c.group, q_padded=q_pad, kv_padded=kv_pad, need_dw_o=c.need_dw_o, need_dw_qkvg=c.need_dw_qkvg)
        assert got == want, (cell_id, got, want)
    assert expected_fp8_launches(qk_norm=True, group=1, q_padded=True, kv_padded=True) == 28  # a padded MHA shape
    assert sum(n for _l, n, p in fp8_block_launch_table(qk_norm=False) if p) == 16
    assert sum(n for _l, n, p in fp8_block_launch_table(qk_norm=True) if p) == 17
    assert fp8_row_launches(group=4, chunks=1, dq_launches=1, q_padded=False, kv_padded=False, zero_ws=False) == 7
    assert fp8_row_launches(group=1, chunks=1, dq_launches=1, q_padded=False, kv_padded=False, zero_ws=False) == 6
    # two head chunks (c = 2, the 397B geometry at long S): +3 per extra chunk
    assert expected_fp8_launches(qk_norm=True, group=16, chunks=2) == 27


# ---------------------------------------------------------------------------
# Building an fp8 backward: the quantized training forward supplies the record exactly as a user would
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_tf32():
    """An fp32 torch reference is a TF32 reference on Blackwell+ unless pinned (``allow_tf32``); the oracles are fp64 but the pin is printed anyway."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"\nallow_tf32={torch.backends.cuda.matmul.allow_tf32}")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def _declare_fp8_bwd(dy, saved, inp, geom, **bwd_kw) -> GatedAttentionBlockBwd:
    """``GatedAttentionBlockBwd(...)`` over a quantized record (the quantized inputs dict's weights, norm weights and cos / sin)."""
    return GatedAttentionBlockBwd(dy, saved, inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], geom, **bwd_kw)


def _dev_scalar(value: float) -> torch.Tensor:
    return torch.full((1,), float(value), dtype=torch.float32, device="cuda")


def _execute_fp8(blk, inp, saved, dy, grads, ws, *, scale_dp, scale_dy=None, scale_do=None, scale_dqkvg=None, current_stream=None):
    kw = dict(scale_dp=scale_dp)
    if scale_dy is not None or scale_do is not None or scale_dqkvg is not None:
        kw.update(scale_dy=scale_dy, scale_do=scale_do, scale_dqkvg=scale_dqkvg)
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


def _fp8_decl(geom_kw, batch, seq_len, *, quant="spec", **bwd_kw):
    """A DECLARED (not compiled) fp8 backward over a DECLARED quantized training forward's record -- CUDA tensors, no launch,
    any CUDA device.  ``quant="spec"`` binds the record's own ``QuantSpec``; ``None`` declares a bf16 backward over it."""
    r = _declare_quant(geom_kw, batch, seq_len, "fp8")
    saved = _alloc_saved(r.geom, r.inp, batch, seq_len, save_mode="proj_slab", act_dtype=torch.bfloat16)
    dy = _make_dy(r.out)
    kw = dict(bwd_kw)
    if quant == "spec":
        kw["quant"] = r.spec
    elif quant is not None:
        kw["quant"] = quant
    blk = _declare_fp8_bwd(dy, saved, r.inp, r.geom, **kw)
    return SimpleNamespace(blk=blk, fwd=r, inp=r.inp, spec=r.spec, saved=saved, dy=dy, out=r.out, geom=r.geom, geom_kw=geom_kw, batch=batch, seq_len=seq_len)


def _declare_then_check(make):
    """The decline may fire at ``__init__`` or at ``check_support`` (both are "before any stage"); either is accepted."""
    make().check_support()


_MEMO: dict = {}


def _test_python_root() -> str:
    """``test/python`` -- the root the SDPA suites' ``sdpa.*`` modules (``fp8_ref``, ``fp8``, ``helpers``) import from; pinned on
    ``sys.path`` so the imports below resolve from any cwd."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _calibrated_scale_dp(blk, ws) -> float:
    """The chart recipe: ONE execute at ``scale_dp = 1.0`` (done by the caller), then ``get_fp8_scale_factor(amax_dP)`` -- ``amax_dP``
    is folded on the fp32 dS BEFORE the scale and the cast, so it is exact even though the e4m3 dS saturates at 1.0."""
    _test_python_root()
    from sdpa.helpers import get_fp8_scale_factor

    torch.cuda.synchronize()
    amax = float(blk.quant_scalars(ws)["amax_dp"].item())
    return float(get_fp8_scale_factor(amax, _E4M3))


def _backward_fp8(geom_kw, batch, seq_len, *, grad_scaling="current", scale_dp="default", memo=True, scales=None, **bwd_kw):
    """Run the quantized training forward (the record, KEPT with its workspace alive: its e4m3 bytes are the reference the
    recomputed ``q8 / k8 / v8`` are pinned against), declare / compile / run the fp8 backward and read its scalar block back.
    ``scale_dp``: a float, ``"default"`` (``_SCALE_DP_DEFAULT``) or ``"calibrated"`` (one warm-up at 1.0, then the chart recipe,
    then the real run).  ``scales`` = ``dict(scale_dy=, scale_do=, scale_dqkvg=)`` floats under ``grad_scaling="delayed"``.
    Memoised per declaration so the contract tests reuse one compiled block."""
    key = (tuple(sorted(geom_kw.items())), batch, seq_len, grad_scaling, str(scale_dp), tuple(sorted((scales or {}).items())), tuple(sorted(bwd_kw.items())))
    if memo and key in _MEMO:
        return _MEMO[key]
    r = _run_training_quant(geom_kw, batch, seq_len, "fp8")
    inp, saved = r.inp, r.saved
    dy = _make_dy(r.out)
    blk = _declare_fp8_bwd(dy, saved, inp, r.geom, quant=r.spec, grad_scaling=grad_scaling, **bwd_kw)
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    grads = _alloc_grads(blk)
    scale_dp_t = _dev_scalar(1.0 if scale_dp == "calibrated" else (_SCALE_DP_DEFAULT if scale_dp == "default" else scale_dp))
    scale_ts = {k: _dev_scalar(v) for k, v in (scales or {}).items()}
    _execute_fp8(blk, inp, saved, dy, grads, ws, scale_dp=scale_dp_t, **scale_ts)
    if scale_dp == "calibrated":
        scale_dp_t.fill_(_calibrated_scale_dp(blk, ws))
        _execute_fp8(blk, inp, saved, dy, grads, ws, scale_dp=scale_dp_t, **scale_ts)
    torch.cuda.synchronize()
    res = SimpleNamespace(
        blk=blk,
        fwd=r,
        inp=inp,
        spec=r.spec,
        saved=saved,
        dy=dy,
        out=r.out,
        ws=ws,
        grads=grads,
        scale_dp=float(scale_dp_t.item()),
        scale_dp_t=scale_dp_t,
        scale_ts=scale_ts,
        scalars={k: float(v.item()) for k, v in blk.quant_scalars(ws).items()},
        geom=r.geom,
        geom_kw=geom_kw,
        batch=batch,
        seq_len=seq_len,
        grad_scaling=grad_scaling,
    )
    if memo:
        _MEMO[key] = res
    return res


def _cell_backward(cell: _Cell, **extra):
    return _backward_fp8(cell.geom_kw, cell.b, cell.s, scale_dp=cell.scale_dp, **cell.bwd_kw, **extra)


def _twin_fp8(res, *, poison=0xFF, grad_scaling=None, scales=None, **bwd_kw):
    """A second fp8 block over the SAME record / dy / inputs / scale_dp as ``res`` (different knobs or recipe), compiled and run
    once into a poisoned workspace and NaN-filled gradients; returns ``(blk, ws, grads)`` -- the bitwise comparand of ``res``."""
    inp = res.inp
    blk = _declare_fp8_bwd(res.dy, res.saved, inp, res.geom, quant=res.spec, grad_scaling=grad_scaling or res.grad_scaling, **bwd_kw)
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda").fill_(poison)
    grads = _alloc_grads(blk, fill=float("nan"))
    scale_ts = {k: _dev_scalar(v) for k, v in (scales or {}).items()} if scales is not None else res.scale_ts
    _execute_fp8(blk, inp, res.saved, res.dy, grads, ws, scale_dp=res.scale_dp_t, **scale_ts)
    torch.cuda.synchronize()
    return blk, ws, grads


def _slots(res) -> dict:
    """The block's materialised intermediates after execute, read out of the workspace: the e4m3 slots as e4m3 views, the
    bf16 bands it quantized from, the SDPA stage's bf16 outputs."""
    blk, g = res.blk, res.geom
    t, d, act = blk.batch * blk.seq_len, g.d_head, blk.act_dtype
    lay = blk._layout()
    o_q, o_g, o_k, o_v = g.qkvg_offsets
    dqkvg = _view(res.ws, lay.dqkvg, (t, g.n_qkvg), act)
    return dict(
        dy8=_view(res.ws, lay.dy8, (t, g.d_model), _E4M3),
        do8=_view(res.ws, lay.do8, (t, g.h_q, d), _E4M3),
        og8=_view(res.ws, lay.og8, (t, g.h_q, d), _E4M3) if lay.og8 >= 0 else None,
        q8=_view(res.ws, lay.q8, (t, g.h_q, d), _E4M3),
        k8=_view(res.ws, lay.k8, (t, g.h_kv, d), _E4M3),
        v8=_view(res.ws, lay.v8, (t, g.h_kv, d), _E4M3),
        dqkvg8=_view(res.ws, lay.dqkvg8, (t, g.n_qkvg), _E4M3),
        do=_view(res.ws, lay.do_gated, (t, g.h_q, d), act),  # B3 wrote dO = dO_gated * sigmoid(gate) IN PLACE over B2's output
        dqkvg=dqkvg,
        dg=dqkvg[:, o_g : o_g + g.h_q * d],
        dq=_view(res.ws, lay.dq, (t, g.h_q, d), act),
        dk=_view(res.ws, lay.dk, (t, g.h_kv, d), act),
        dv=_view(res.ws, lay.dv, (t, g.h_kv, d), act),
        lay=lay,
    )


def _delta(res) -> torch.Tensor:
    """The block's ``delta`` region B3 wrote (ALWAYS under quant): fp32 ``[B, H_q, S_pad]``, the row's external delta."""
    lay = res.blk._layout()
    shape = tuple(res.blk._sdpa.delta_shape)
    assert lay.delta >= 0, "the quantized backward always carves delta (the external delta is mandatory)"
    return _view(res.ws, lay.delta, shape, torch.float32)


def _scalar_block(res) -> torch.Tensor:
    """The whole fp32 scalar block as one view (every slot, in ``QUANT_SCALAR_SLOTS`` order)."""
    lay = res.blk._layout()
    return _view(res.ws, lay.quant_scalars, (len(_api_const("QUANT_SCALAR_SLOTS")),), torch.float32)


def _grad_scale(amax: float) -> float:
    """The "current" recipe at the module's margin: the ONE formula the kernel mirrors bitwise."""
    return float(_quantize.grad_scale_from_amax(amax, _api_const("FP8_GRAD_SCALE_MARGIN_LOG2")))


def _f32(x: float) -> float:
    return float(np.float32(x))


def _f32_mul(a: float, b: float) -> float:
    """ONE fp32 RN multiply -- the alpha products as the publishing kernel forms them."""
    return float(np.float32(np.float32(a) * np.float32(b)))


def _assert_e4m3_bitwise(got8: torch.Tensor, src: torch.Tensor, scale: float, what: str) -> None:
    """``got8`` is BITWISE ``sat_e4m3(src * scale)`` (torch's saturating RNE cast -- bit-exact vs the kernels')."""
    want = quant_e4m3(src.reshape(got8.shape), scale)
    same = got8.view(torch.uint8) == want.view(torch.uint8)
    n_bad = int((~same).sum())
    print(f"{what}: {n_bad} of {same.numel()} e4m3 codes differ from torch's cast at scale {scale:g}")
    assert n_bad == 0, f"{what}: {n_bad} e4m3 codes differ from sat_e4m3(src * {scale:g})"


def _assert_og8_bitwise_the_kernels_o_gated(og8: torch.Tensor, o: torch.Tensor, gate: torch.Tensor, scale_o: float, fwd_o8: torch.Tensor) -> None:
    """``og8`` (B3's fp8 arm) is BITWISE (1) the forward workspace's ``o8`` -- the forward's quantize of the forward's bf16 ``O_gated``
    over the same O / gate (kernel vs kernel: "recomputed bit-exactly") -- and (2) torch's saturating RNE cast, at ``scale_o``, of
    the KERNELS' bf16 ``O_gated``: the forward gate stage re-run over the same O / gate, the exact bf16 words B3 computes before
    its cvt.  The torch composition ``bf16(O * torch.sigmoid(G))`` is NOT the kernels' function: both kernels spell the sigmoid
    as the tanh identity (``tanh(g / 2) / 2 + 1 / 2``, one MUFU per element) whose last fp32 bit differs from torch's on a handful
    of elements that sit on a bf16 rounding midpoint -- so its code count is REPORTED, and every differing code is required to
    sit on an element where the kernels' bf16 ``O_gated`` differs from torch's (the e4m3 cast itself is torch's wherever the bf16
    inputs agree)."""
    from test_sigmoid_gate_bwd import _forward_og

    t, h, d = (int(x) for x in og8.shape)
    assert torch.equal(og8.view(torch.uint8), fwd_o8.view(torch.uint8)), "og8 is not bitwise the forward workspace's o8 (kernel vs kernel)"
    og16_kernel = _forward_og(o, gate, h, d)
    _assert_e4m3_bitwise(og8, og16_kernel, scale_o, "og8 (vs the kernels' bf16 O_gated)")
    og16_torch = (o.float() * torch.sigmoid(gate.float())).to(torch.bfloat16)
    code_differs = og8.view(torch.uint8) != quant_e4m3(og16_torch, scale_o).view(torch.uint8)
    bf16_differs = og16_kernel != og16_torch
    print(
        f"og8 vs the torch-sigmoid composition: {int(code_differs.sum())} of {code_differs.numel()} codes differ; the kernels' bf16 O_gated differs "
        f"from torch's on {int(bf16_differs.sum())} elements (the tanh-identity sigmoid's last fp32 bit at a bf16 rounding midpoint)"
    )
    assert bool((code_differs <= bf16_differs).all()), "an og8 code differs from torch's cast on an element whose bf16 O_gated agrees with torch's"


def _report_close(got: torch.Tensor, ref64: torch.Tensor, what: str) -> float:
    """The bf16 block's bound as a NUMBER -- ``max|diff|``, ``max|ref|``, the worst cell as a fraction of ``atol + rtol * |ref|``
    (``_RTOL`` / ``_ATOL_FRAC``) and the cosine -- printed and returned, never asserted: the localisation print of a stage whose
    output is judged downstream."""
    got64, ref64 = got.detach().double().reshape(-1), ref64.detach().double().reshape(-1)
    rtol, atol_frac = _RTOL[got.dtype], _ATOL_FRAC[got.dtype]
    ref_max = ref64.abs().max().item()
    diff = (got64 - ref64).abs()
    worst = (diff / (atol_frac * ref_max + rtol * ref64.abs())).max().item()
    print(
        f"{what}: max|diff|={diff.max().item():.4g} max|ref|={ref_max:.4g} worst cell {worst:.3f} of the bf16 bound cos={_cos(got64, ref64):.6f} (printed, not asserted)"
    )
    return worst


def _rows_outside(got: torch.Tensor, ref64: torch.Tensor) -> tuple:
    """``(rows outside, rows)`` of the bf16 block's bound, a ROW being one token of ``dh`` or one output row of a ``dW`` -- the
    statistic of the row-budgeted form (``1e-5 x rows x keys``) the flip class downstream of an e4m3 cast is judged by."""
    g2 = got.detach().double().reshape(-1, got.shape[-1])
    r2 = ref64.detach().double().reshape(-1, got.shape[-1])
    outside = ((g2 - r2).abs() > _ATOL_FRAC[got.dtype] * r2.abs().max() + _RTOL[got.dtype] * r2.abs()).any(dim=1)
    return int(outside.sum()), int(g2.shape[0])


def _report_seeded_intermediates(res, v: dict, ref: dict) -> None:
    """Localisation between B4 and the outputs (no new bound): the block's bf16 ``dqkvg`` bands (B3's dG, B5+B6's dQ_pre / dK_pre)
    against the seeded oracle's fp64 bands as a fraction of the bf16 block's bound, PRINTED; the slab's V band ``torch.equal``
    the block's own dV slot (the norm backward copies it bit for bit -- a copy, asserted); and the e4m3 flips of ``dqkvg8``
    against the cast of the oracle's own bf16-rounded ``dqkvg`` at the block's scale -- a flip at ``[t, n]`` moves the whole
    ``dW_qkvg`` row ``n`` by ``flip * h[t, :]`` and the whole ``dh`` row ``t`` by ``flip * W_qkvg[n, :]`` (the rank-1 flip-class shape,
    budgeted by rows), so a seeded ``dw_qkvg`` / ``dh`` residual is attributable to them."""
    g, sc = res.geom, res.scalars
    t, d = res.batch * res.seq_len, g.d_head
    o_q, o_g, o_k, o_v = g.qkvg_offsets
    dqkvg = v["dqkvg"]
    bands = ((o_q, g.h_q, "dq_pre", "dq_pre"), (o_g, g.h_q, "dg", "dg"), (o_k, g.h_kv, "dk_pre", "dk_pre"), (o_v, g.h_kv, "dv", "dv_band"))
    ref_slab = torch.empty(t, g.n_qkvg, dtype=torch.float64, device=dqkvg.device)
    for off, heads, name, key in bands:
        _report_close(_cols(dqkvg, off, heads, d), ref[key], f"band {name} vs the seeded oracle")
        ref_slab[:, off : off + heads * d] = ref[key].reshape(t, heads * d)
    assert torch.equal(_cols(dqkvg, o_v, g.h_kv, d).contiguous(), v["dv"]), "the slab's V band is not bitwise the block's own dV slot"
    flips = v["dqkvg8"].view(torch.uint8) != quant_e4m3(ref_slab.to(torch.bfloat16), sc["scale_dqkvg"]).view(torch.uint8)
    rows_t, cols_n = torch.nonzero(flips, as_tuple=True)
    print(
        f"dqkvg8 vs e4m3(bf16(seeded oracle dqkvg)) at scale {sc['scale_dqkvg']:g}: {int(flips.sum())} of {flips.numel()} codes differ "
        f"({int((dqkvg != ref_slab.to(torch.bfloat16)).sum())} bf16 cells differ before the cast); {int(torch.unique(cols_n).numel())} dW_qkvg rows "
        f"and {int(torch.unique(rows_t).numel())} dh rows touched"
    )
    if v["og8"] is not None:
        # the oracle's og8 = e4m3(bf16(O_record * sigmoid(gate_record)) * scale_o) under torch's fp64 sigmoid: a flip at [t, j] moves
        # COLUMN j of dW_o = dy8^T . og8 (the kernels' tanh-identity sigmoid is the remaining difference)
        og_flips = v["og8"].view(torch.uint8) != ref["og8"].reshape(v["og8"].shape).view(torch.uint8)
        _rows, cols_j = torch.nonzero(og_flips.reshape(t, -1), as_tuple=True)
        print(
            f"og8 vs the seeded oracle's og8 (the record's O and gate, torch's sigmoid): {int(og_flips.sum())} of {og_flips.numel()} codes differ; "
            f"{int(torch.unique(cols_j).numel())} dW_o columns touched"
        )
    for name, keys in (("dh", g.n_qkvg), ("dw_qkvg", t), ("dw_o", t)):
        if res.grads.get(name) is not None:
            n_out, n_rows = _rows_outside(res.grads[name], ref[name])
            print(f"{name} vs the seeded oracle: {n_out} of {n_rows} rows outside the bf16 bound (row budget 1e-5 x rows x keys = {1e-5 * n_rows * keys:.3g})")


def _row_tol() -> tuple:
    """The fp8 SDPA row suite's own recipe -- its tolerance constants and ``assert_close_fp8_grad`` -- imported, never re-literalled."""
    frost_dir = os.path.join(_test_python_root(), "sdpa", "frost")
    if frost_dir not in sys.path:
        sys.path.insert(0, frost_dir)
    from sdpa.fp8 import assert_close_fp8_grad
    from test_sdpa_bwd_fp8_sm107 import _AMAX_DS_TOL, _FP8_GRAD_TOL

    return _FP8_GRAD_TOL, _AMAX_DS_TOL, assert_close_fp8_grad


def _gemm_bound(out: torch.Tensor, ref64: torch.Tensor, what: str) -> None:
    """The GEMM suite's bound (``test_proj_gemm_bwd._assert_close_vs_fp64``): an fp8 input is EXACT in fp64, so the same derivation holds."""
    from test_proj_gemm_bwd import _assert_close_vs_fp64

    _assert_close_vs_fp64(out, ref64, what)


def _row_reference(res, v: dict):
    """The fp8 row's reference over the block's OWN operands and conditions: e4m3 ``q8 / k8 / v8 / do8``, the forward's exact LSE,
    the block's ``delta`` (the SAME tensor the kernel consumed), ``scale_s = 2 ** FP8_SCALE_S_LOG2``, the caller's ``scale_dp``,
    bf16 gradients.  Returns ``(dq, dk, dv)`` fp32 BSHD (bf16-rounded like the row's output), the fp32 ``max |dS|`` it reduces into
    ``amax_dP``, and the intermediates for ``assert_close_fp8_grad``'s flip attribution."""
    _test_python_root()
    from sdpa.fp8_ref import compute_ref_backward

    blk, g, sc, sp = res.blk, res.geom, res.scalars, res.spec
    b, s, d = blk.batch, blk.seq_len, g.d_head
    s_scale = 2.0 ** _api_const("FP8_SCALE_S_LOG2")
    bshd = lambda x, h: x.reshape(b, s, h, d)  # noqa: E731
    o_dead = v["og8"] if v["og8"] is not None else v["do8"]
    dq, dk, dv, _dsink, _dp_amax_raw, _dqa, _dka, _dva, inter = compute_ref_backward(
        bshd(v["q8"], g.h_q),
        bshd(v["k8"], g.h_kv),
        bshd(v["v8"], g.h_kv),
        bshd(o_dead, g.h_q),
        bshd(v["do8"], g.h_q),
        g.scale,
        1.0 / sp.scale_q,
        1.0 / sp.scale_k,
        1.0 / sp.scale_v,
        s_scale,
        1.0 / s_scale,
        _E4M3,
        1.0 / sp.scale_o,
        sc["descale_do"],
        torch.bfloat16,
        left_bound=None,
        right_bound=0 if g.is_causal else None,
        diag_align=None,
        stats=res.saved.lse[..., None],
        return_intermediates=True,
        quantize_ds=True,
        dP_scale=res.scale_dp,
        quantize_grads=False,
        delta=_delta(res)[..., :s],
    )
    amax_ds = float(inter["ds_scaled"].abs().max()) / res.scale_dp
    return dq.to(torch.bfloat16).float(), dk.to(torch.bfloat16).float(), dv.to(torch.bfloat16).float(), amax_ds, inter


def _oracle(res, *, modelled: bool, seeded: Optional[dict] = None) -> dict:
    """The fp8 backward oracle fed the block's OWN conditions: its read-back gradient scales, ``2 ** FP8_SCALE_S_LOG2``, its
    ``scale_dp`` and the SAME ``delta`` the kernel consumed -- and, for the MODELLED and seeded oracles, the record's exact LSE,
    the record's bf16 pre-gate O and the record's bf16 GATE band (the inputs the backward reads: the kernel recomputes P from
    that LSE, and the gate backward forms dG / og8 from that O and that gate); ``seeded`` substitutes the block's bf16 dQ / dK /
    dV.  The unquantized-gradient oracle (U) keeps its own fp64 attention O / LSE / projection on purpose: it is the all-in
    informational reference."""
    sc = res.scalars
    g, t = res.geom, res.batch * res.seq_len
    _o_q, o_g, _o_k, _o_v = g.qkvg_offsets
    record = dict(lse=res.saved.lse, o=res.saved.o, gate=_cols(res.saved.proj_slab.view(t, g.n_qkvg), o_g, g.h_q, g.d_head)) if modelled else {}
    return gated_attention_block_fp8_bwd_reference(
        res.inp,
        RefGeometry(**res.geom_kw),
        res.spec,
        res.dy,
        scale_dy=sc["scale_dy"],
        scale_do=sc["scale_do"],
        scale_dqkvg=sc["scale_dqkvg"],
        scale_s=2.0 ** _api_const("FP8_SCALE_S_LOG2"),
        scale_dp=res.scale_dp,
        delta=_delta(res)[..., : res.seq_len],
        modelled=modelled,
        seeded=seeded,
        **record,
    )


def _oracle_m(res) -> dict:
    return _oracle(res, modelled=True)


def _oracle_u(res) -> dict:
    return _oracle(res, modelled=False)


def _oracle_seeded(res) -> dict:
    v = _slots(res)
    b, s, g = res.batch, res.seq_len, res.geom
    seed = dict(dq=v["dq"].view(b, s, g.h_q, g.d_head), dk=v["dk"].view(b, s, g.h_kv, g.d_head), dv=v["dv"].view(b, s, g.h_kv, g.d_head))
    return _oracle(res, modelled=True, seeded=seed)


def _print_end_to_end(tag: str, grads: dict, ref: dict, *, keys: Optional[dict] = None) -> dict:
    """``cos`` and ``max|diff| / max|ref|`` per produced gradient against an oracle -- printed, returned, never asserted here --
    plus, for the bf16 outputs, the row-budgeted statistic an (M) assertion would use (the flip-class shape, budgeted like
    ``assert_close_fp8_grad``): the number of ROWS with a cell outside the bf16 block's bound against ``1e-5 x rows x keys``
    (``keys`` = the reduction length feeding a row)."""
    out = {}
    for name in ("dh", "dw_qkvg", "dw_o", "dw_q_norm", "dw_k_norm"):
        got = grads.get(name)
        if got is None or ref.get(name) is None:
            continue
        got64, ref64 = got.detach().double(), ref[name].detach().double()
        max_rel = ((got64 - ref64).abs().max() / ref64.abs().max().clamp_min(1e-300)).item()
        out[name] = dict(cos=_cos(got64, ref64), max_rel=max_rel)
        line = f"{tag} {name}: cos={out[name]['cos']:.6f} max|diff|/max|ref|={max_rel:.4g}"
        if got.dtype in _RTOL and keys and name in keys:
            g2, r2 = got64.reshape(-1, got64.shape[-1]), ref64.reshape(-1, got64.shape[-1])  # a ROW = one token of dh, one output row of a dW
            outside = ((g2 - r2).abs() > _ATOL_FRAC[got.dtype] * r2.abs().max() + _RTOL[got.dtype] * r2.abs()).any(dim=1)
            out[name]["rows_outside"] = int(outside.sum())
            line += (
                f" rows outside the bf16 bound: {int(outside.sum())} of {g2.shape[0]} (row budget 1e-5 x rows x keys = {1e-5 * g2.shape[0] * keys[name]:.3g})"
            )
        print(line)
    return out


# ---------------------------------------------------------------------------
# ACCEPT (Rubin; integration pending)
# ---------------------------------------------------------------------------


@requires_rubin
@_MATRIX
def test_fp8_stage_localised_bounds(cell):
    """Every stage of the quantized backward against the bound calibrated FOR IT, on the block's own operands (module docstring):

    quantizers BITWISE -- ``dy8 == sat_e4m3(bf16 dY * scale_dy)`` with the READ-BACK scale, ``do8`` over the stored bf16 dO,
    ``dqkvg8`` over the bf16 slab, ``og8`` ``torch.equal`` the forward workspace's ``o8`` bytes AND the cast of the KERNELS' bf16
    ``O_gated`` at the forward's ``scale_o`` (``_assert_og8_bitwise_the_kernels_o_gated``: the tanh-identity sigmoid is not
    torch's to the last fp32 bit), ``q8 / k8 / v8`` ``torch.equal`` the FORWARD's bytes in the record run's workspace
    (decision 8, "recomputed bit-exactly", pinned against the forward -- never against a torch cast of the backward's own
    recompute); scalars BITWISE -- ``scale_* == grad_scale_from_amax(amax_*)``, ``descale == 1 / scale``, the alphas ONE fp32
    product each, ``amax_* == max|.|`` of the tensor the pass READ, ``descale_dp == 1 / scale_dp``; ``delta`` bitwise the
    chain's own ``dot_do_o`` over the same bf16 O / dO with an exactly-zero pad tail; the GEMMs on the block's e4m3 operands
    vs fp64 under the GEMM suite's bound (B1 / B7 / B8; B2 is read through dO -- B3 overwrote its output in place -- under the
    bf16 block's bound); the SDPA stage ``ws.dq / dk / dv`` vs the row's reference on the block's own ``q8 / k8 / v8 / do8``,
    ``saved.lse``, the block's ``delta`` and scalars under ``_FP8_GRAD_TOL`` + the flip budget, ``amax_dP`` under
    ``_AMAX_DS_TOL``; ``dh / dW_qkvg / dW_o / dW_*_norm`` vs the oracle SEEDED with the block's own dQ / dK / dV under the bf16
    block's bound (HYPOTHESIS: magnitudes printed, calibrated on the first Rubin run, never widened)."""
    res = _cell_backward(cell)
    blk, g, sc, sp, saved = res.blk, res.geom, res.scalars, res.spec, res.saved
    b, s, d = res.batch, res.seq_len, g.d_head
    t = b * s
    v = _slots(res)
    # --- quantizers, bitwise -------------------------------------------------------------------------------------------
    _assert_e4m3_bitwise(v["dy8"], res.dy, sc["scale_dy"], "dy8")
    _assert_e4m3_bitwise(v["do8"], v["do"], sc["scale_do"], "do8")
    _assert_e4m3_bitwise(v["dqkvg8"], v["dqkvg"], sc["scale_dqkvg"], "dqkvg8")
    flay = res.fwd.blk._layout()
    for name, h in (("q8", g.h_q), ("k8", g.h_kv), ("v8", g.h_kv)):
        fwd_bytes = _view(res.fwd.ws, getattr(flay, name), (t, h, d), _E4M3).view(torch.uint8)
        assert torch.equal(v[name].view(torch.uint8), fwd_bytes), f"{name}: the recomputed e4m3 operand is not bitwise the forward's"
    _o_q, o_g, _o_k, _o_v = g.qkvg_offsets
    gate = _cols(saved.proj_slab.view(t, g.n_qkvg), o_g, g.h_q, d)  # the GATE band of the slab, strided, as the gate kernels read it
    if blk.need_dw_o:
        assert v["og8"] is not None
        _assert_og8_bitwise_the_kernels_o_gated(v["og8"], saved.o.view(t, g.h_q, d), gate, sp.scale_o, _view(res.fwd.ws, flay.o8, (t, g.h_q, d), _E4M3))
    else:
        assert v["og8"] is None and v["lay"].og8 == -1
    # --- scalars, bitwise ----------------------------------------------------------------------------------------------
    for n, src in (("dy", res.dy), ("do", v["do"]), ("dqkvg", v["dqkvg"])):
        assert sc[f"amax_{n}"] == src.float().abs().max().item(), (n, sc[f"amax_{n}"], src.float().abs().max().item())
        assert sc[f"scale_{n}"] == _grad_scale(sc[f"amax_{n}"]), (n, sc[f"scale_{n}"], _grad_scale(sc[f"amax_{n}"]))
        assert sc[f"descale_{n}"] == _f32(1.0 / sc[f"scale_{n}"]), n
    assert sc["alpha_b1"] == _f32_mul(sc["descale_dy"], 1.0 / sp.scale_o)
    assert sc["alpha_b2"] == _f32_mul(sc["descale_dy"], sp.descale_w_o)
    assert sc["alpha_b7"] == _f32_mul(sc["descale_dqkvg"], sp.descale_h)
    assert sc["alpha_b8"] == _f32_mul(sc["descale_dqkvg"], sp.descale_w_qkvg)
    assert sc["descale_dp"] == _f32(1.0 / res.scale_dp)
    # --- delta, bitwise ------------------------------------------------------------------------------------------------
    from test_sigmoid_gate_bwd import _chain_dot_do_o

    delta = _delta(res)
    want = _chain_dot_do_o(saved.o, v["do"].view(b, s, g.h_q, d))
    assert delta.shape == want.shape and torch.equal(delta[..., :s], want[..., :s]), "delta is not bitwise the chain's dot_do_o"
    assert torch.equal(delta[..., s:], torch.zeros_like(delta[..., s:])), "the delta pad tail must be exact zeros"
    # --- the GEMMs on the block's own e4m3 operands ---------------------------------------------------------------------
    dy8_64 = v["dy8"].double() * (1.0 / sc["scale_dy"])
    wo64 = res.inp["w_o"].double() * sp.descale_w_o  # [d_model, H_q*D]
    do_gated64 = dy8_64 @ wo64  # B2's fp64 product (bf16-rounded by the epilogue, then gated in place by B3)
    do_ref64 = do_gated64.to(torch.bfloat16).double().view(t, g.h_q, d) * torch.sigmoid(gate.double())
    _assert_grad_close(v["do"], do_ref64, "dO (B2 + B3 composite; B2's output is overwritten in place)")
    if blk.need_dw_o:
        og8_64 = v["og8"].double() * (1.0 / sp.scale_o)
        _gemm_bound(res.grads["dw_o"], dy8_64.t() @ og8_64.view(t, g.h_q * d), "B1 dW_o = dy8^T . og8 (alpha_b1)")
    dqkvg8_64 = v["dqkvg8"].double() * (1.0 / sc["scale_dqkvg"])
    if blk.need_dw_qkvg:
        h8_64 = saved.h.view(t, g.d_model).double() * sp.descale_h
        _gemm_bound(res.grads["dw_qkvg"], dqkvg8_64.t() @ h8_64, "B7 dW_qkvg = dqkvg8^T . h8 (alpha_b7)")
    if blk.need_dh:
        wq64 = res.inp["w_qkvg"].double() * sp.descale_w_qkvg  # [N, d_model]
        _gemm_bound(res.grads["dh"].view(t, g.d_model), dqkvg8_64 @ wq64, "B8 dh = dqkvg8 . W_qkvg8 (alpha_b8)")
    # --- the SDPA stage under the row's recipe --------------------------------------------------------------------------
    grad_tol, amax_tol, assert_close_fp8_grad = _row_tol()
    dq_ref, dk_ref, dv_ref, amax_ds_ref, _inter = _row_reference(res, v)
    refs = dict(dq=dq_ref, dk=dk_ref, dv=dv_ref)
    for name, h, tag in (("dq", g.h_q, "dQ"), ("dk", g.h_kv, "dK"), ("dv", g.h_kv, "dV")):
        # keys = the reduction length feeding each d-row (s_kv for dQ, s_q for dK / dV): self-attention, both are S
        assert_close_fp8_grad(v[name].view(b, s, h, d).float(), refs[name], grad_tol["atol"], grad_tol["rtol"], tag, keys=s)
    amax_dp = sc["amax_dp"]
    print(f"amax_dP {amax_dp:.6g} vs the reference's max|dS| {amax_ds_ref:.6g} (scale_dp {res.scale_dp:g}, amax_dP * scale_dp = {amax_dp * res.scale_dp:.4g})")
    assert abs(amax_dp - amax_ds_ref) <= amax_tol["atol"] + amax_tol["rtol"] * amax_ds_ref, (amax_dp, amax_ds_ref)
    # --- downstream of B4: the SEEDED oracle under the bf16 block's bound ------------------------------------------------
    ref = _oracle_seeded(res)
    _report_seeded_intermediates(res, v, ref)
    worst = {}
    for name in ("dh", "dw_qkvg", "dw_o"):
        if res.grads[name] is not None:
            worst[name] = _assert_grad_close(res.grads[name], ref[name], f"{name} vs the seeded oracle")
    for name in ("dw_q_norm", "dw_k_norm"):
        if res.grads[name] is not None:
            worst[name] = _assert_dw_norm_close(res.grads[name], ref[name], ref[name + "_mass"], f"{name} vs the seeded oracle")
    print(f"{cell.id}: worst cells (fraction of the bound) {worst}")


@requires_rubin
@_MATRIX
def test_fp8_end_to_end_vs_the_oracles(cell):
    """End to end against (M) the fully MODELLED oracle (every backward quantization point, the row's reference inside) and
    (U) the unquantized-gradient STE oracle (forward points only, fp64 backward): ``cos`` and ``max|diff| / max|ref|`` PRINTED
    per gradient; (M) is asserted only after the first run's margin is known (row-budgeted like ``assert_close_fp8_grad``), so
    this cell pins finiteness and the report, nothing more, until then."""
    res = _cell_backward(cell)
    for name, ten in res.grads.items():
        if ten is not None:
            assert torch.isfinite(ten).all(), f"{name}: non-finite cells"
    g, t = res.geom, res.batch * res.seq_len
    keys = dict(dh=g.n_qkvg, dw_qkvg=t, dw_o=t)  # the reduction length feeding each row: dh over N, the weight gradients over the tokens
    m = _print_end_to_end(f"{cell.id} (M)", res.grads, _oracle_m(res), keys=keys)
    u = _print_end_to_end(f"{cell.id} (U)", res.grads, _oracle_u(res), keys=keys)
    assert m and u


@requires_rubin
@_KNOB_SETS
def test_fp8_two_runs_are_bitwise(knobs):
    """Two executes of the same block over the same record / dy / scale_dp are ``torch.equal`` on every gradient AND on the
    scalar block, with the workspace poisoned 0xFF between them, under every knob set (``fuse_gate_bwd`` is inert under quant;
    ``fuse_wgrad_overlap`` moves B1 / B7 to the side stream): the block's only atomics are int32 ``atomicMax`` of non-negative
    fp32 bit patterns -- order-free."""
    res = _cell_backward(_BITWISE_CELL, **knobs)
    sb1 = _scalar_block(res).clone()
    blk, ws, grads = _twin_fp8(res, **knobs)
    for name, ten in grads.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), f"{name}: two runs differ (knobs={knobs})"
    assert torch.equal(_view(ws, blk._layout().quant_scalars, sb1.shape, torch.float32), sb1), "the scalar block differs between two runs"


@requires_rubin
def test_fp8_fuse_wgrad_overlap_is_bitwise_the_in_order_block():
    """``fuse_wgrad_overlap=True`` is a scheduling knob: every gradient and the scalar block ``torch.equal`` the in-order block's
    over the same record; the side-stream GEMMs read the ``alpha_b1 / alpha_b7`` slots and the e4m3 operands written on the
    launch stream before their fork events (B1 after the dO quantize, B7 after the dqkvg quantize) -- the scalar block is
    poisoned between the two runs so a mis-placed fork reading a stale alpha cannot hide behind an equal value."""
    res = _cell_backward(_BITWISE_CELL)
    blk, ws, grads = _twin_fp8(res, fuse_wgrad_overlap=True)
    for name, ten in grads.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), f"{name}: fuse_wgrad_overlap differs from the in-order block"
    assert torch.equal(_view(ws, blk._layout().quant_scalars, (len(_api_const("QUANT_SCALAR_SLOTS")),), torch.float32), _scalar_block(res))


@requires_rubin
def test_fp8_fuse_gate_bwd_is_inert_under_quant():
    """The external delta is MANDATORY under quant, so ``fuse_gate_bwd`` has no second arm: both values are accepted and give
    ``torch.equal`` gradients and scalar blocks (a knob computes the same function under any value); the stage's adapter is
    built with ``external_delta=True`` either way."""
    res = _cell_backward(_BITWISE_CELL)
    assert res.blk._sdpa._impl.external_delta is True
    blk, ws, grads = _twin_fp8(res, fuse_gate_bwd=True)
    assert blk._sdpa._impl.external_delta is True
    for name, ten in grads.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), f"{name}: fuse_gate_bwd changed the gradients under quant"


@requires_rubin
def test_fp8_delayed_replays_current_bitwise():
    """A ``grad_scaling="delayed"`` block fed the "current" run's READ-BACK ``scale_dy / scale_do / scale_dqkvg`` as device scalars
    gives ``torch.equal`` gradients AND an equal scalar block: the two recipes are ONE code path with two writers (the amax
    passes run under both; the quantize kernel publishes scale / descale / alphas from the given or the derived scale)."""
    res = _cell_backward(_BITWISE_CELL)
    sc = res.scalars
    scales = dict(scale_dy=sc["scale_dy"], scale_do=sc["scale_do"], scale_dqkvg=sc["scale_dqkvg"])
    blk, ws, grads = _twin_fp8(res, grad_scaling="delayed", scales=scales)
    for name, ten in grads.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), f"{name}: the delayed replay differs from the current run"
    assert torch.equal(_view(ws, blk._layout().quant_scalars, (len(_api_const("QUANT_SCALAR_SLOTS")),), torch.float32), _scalar_block(res))


@requires_rubin
@pytest.mark.parametrize("cell_id, expected", _LAUNCH_EXPECTATIONS, ids=[c for c, _e in _LAUNCH_EXPECTATIONS])
def test_fp8_launch_count_is_honest(cell_id, expected):
    """CUPTI kernel records of one execute == the launch table (module docstring): 24 at the bitwise cell (norm, GQA 8/2, c = 1,
    one dQ launch per chunk), 23 rope_only (no reduce), 23 MHA (no dK fold), 29 at the two padded GQA cells, 27 at the padded
    dgrad-only cell; the same under both recipes and every knob; no hidden memcpy, memsets bounded as the forward's count is.
    The formula is ALSO recomputed from the adapter's own facts (``fp8_launch_formula_from_facts``) so a change on either side
    is visible; ``external_delta is True`` is asserted there."""
    from torch.profiler import ProfilerActivity, profile

    cell = _BY_ID[cell_id]
    res = _cell_backward(cell)
    blk = res.blk
    formula = fp8_launch_formula_from_facts(blk)
    grads = _alloc_grads(blk)
    _execute_fp8(blk, res.inp, res.saved, res.dy, grads, res.ws, scale_dp=res.scale_dp_t)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        _execute_fp8(blk, res.inp, res.saved, res.dy, grads, res.ws, scale_dp=res.scale_dp_t)
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    if not names:
        pytest.skip("torch.profiler recorded no CUDA events (CUPTI unavailable on this node); the launch count is unverified here")
    memsets = [n for n in names if "memset" in n.lower()]
    memcpys = [n for n in names if "memcpy" in n.lower()]
    kernels = [n for n in names if n not in memsets and n not in memcpys]
    print(f"\n{len(kernels)} kernels (formula {formula}, expected {expected}), {len(memsets)} memsets, {len(memcpys)} memcpys:\n  " + "\n  ".join(names))
    assert not memcpys, f"a hidden copy on the execute path: {memcpys}"
    assert len(memsets) <= 1, f"unexpected memsets: {memsets}"
    assert formula == expected, (formula, expected)
    assert len(kernels) == expected, (len(kernels), expected, kernels)


@requires_rubin
def test_fp8_cuda_graph_capture_replays_bitwise():
    """One ``execute`` captured into a CUDA graph on a side torch stream replays bitwise the eager run and recomputes over a
    NEW ``dy`` and a NEW ``scale_dp`` written in place through the captured pointers (the scalar-init launch, the amax atomics
    and the quantize publishes are all device work on the launch stream: capturable, no host readback).  The capture itself
    launches nothing; the block allocates nothing."""
    res = _cell_backward(_BY_ID["s256_causal_b1-norm"])
    blk = res.blk
    dy2 = res.dy.clone()
    sdp2 = res.scale_dp_t.clone()
    ws = torch.empty_like(res.ws)
    grads = _alloc_grads(blk, fill=float("nan"))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _execute_fp8(blk, res.inp, res.saved, dy2, grads, ws, scale_dp=sdp2)  # warm-up on the capture stream
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
            _execute_fp8(blk, res.inp, res.saved, dy2, grads, ws, scale_dp=sdp2)
        torch.cuda.synchronize()
        for name, ten in grads.items():
            if ten is not None:
                assert torch.isnan(ten).all(), f"{name}: the capture launched work"
        graph.replay()
        torch.cuda.synchronize()
        for name, ten in grads.items():
            if ten is not None:
                assert torch.equal(ten, res.grads[name]), f"{name}: the replay differs from the eager run"
        dy3 = _make_dy(res.out, seed=7)
        dy2.copy_(dy3)
        sdp2.fill_(res.scale_dp * 2.0)
        ws.fill_(0xFF)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        ref = _alloc_grads(blk, fill=float("nan"))
        ws_ref = torch.empty_like(ws).fill_(0xFF)
        _execute_fp8(blk, res.inp, res.saved, dy3, ref, ws_ref, scale_dp=_dev_scalar(res.scale_dp * 2.0))
        torch.cuda.synchronize()
        for name, ten in grads.items():
            if ten is not None:
                assert torch.isfinite(ten).all() and torch.equal(ten, ref[name]), f"{name}: the replay over new inputs differs from eager"
    finally:  # a graph left to the cyclic GC resets itself inside a later test's capture
        graph.reset()


@requires_rubin
@_KNOB_SETS
def test_fp8_workspace_size_is_honest(knobs):
    """``get_workspace_size()`` is exact and never exceeded: the carve (every appended quant region present, 256-B aligned, the
    scalar block 256 B) + the fp8 adapter's scratch (``scratch_workspace_bytes()``, no ``delta`` region: the block's own one takes
    its place) + the GEMM scratch (the max over the four K64 fp8 plans) exactly; a buffer 4096 B larger keeps its tail untouched;
    two executes allocate nothing; every e4m3 region and the scalar block are WRITTEN (no 0xFF byte survives in them)."""
    from cudnn.gated_attention_block.api import _WS_ALIGN

    res = _cell_backward(_BITWISE_CELL, **knobs)
    blk = res.blk
    size = blk.get_workspace_size()
    lay = blk._layout()
    assert size == lay.total_bytes and size % _WS_ALIGN == 0
    plans = blk.gemm_plans
    assert len(plans) == 4 and all(p.mma_tile_k_bytes == 64 and p.has_alpha for p in plans.values())
    assert lay.gemm_scratch_bytes >= max(p.workspace_bytes for p in plans.values()) >= 1
    assert lay.sdpa_bwd_bytes == blk._sdpa.scratch_workspace_bytes()
    assert lay.quant_scalars >= 0 and lay.quant_scalars % 256 == 0 and lay.delta >= 0 and lay.o_gated == -1 and lay.recompute_v == -1
    for name in ("dy8", "do8", "q8", "k8", "v8", "dqkvg8"):
        assert getattr(lay, name) >= 0 and getattr(lay, name) % _WS_ALIGN == 0, name
    ws = torch.full((size + 4096,), 0xFF, dtype=torch.uint8, device="cuda")
    grads = _alloc_grads(blk)
    _execute_fp8(blk, res.inp, res.saved, res.dy, grads, ws[:size], scale_dp=res.scale_dp_t)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    _execute_fp8(blk, res.inp, res.saved, res.dy, grads, ws[:size], scale_dp=res.scale_dp_t)
    _execute_fp8(blk, res.inp, res.saved, res.dy, grads, ws[:size], scale_dp=res.scale_dp_t)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before, "execute allocated on the hot path"
    assert torch.equal(ws[size:], torch.full((4096,), 0xFF, dtype=torch.uint8, device="cuda")), "bytes past get_workspace_size() were written"
    for name, ten in grads.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), name
    sb = _view(ws[:size], lay.quant_scalars, (len(_api_const("QUANT_SCALAR_SLOTS")),), torch.float32)
    assert torch.isfinite(sb).all(), "a scalar slot was never written (0xFF = NaN)"


@requires_rubin
@_MATRIX
def test_fp8_amax_times_scale_never_exceeds_448(cell):
    """Every published (amax, scale) pair of the scalar block satisfies ``amax * scale <= 448`` -- the in-kernel formula's guarantee
    for the amax the kernel READ (dY, dO over the STORED bf16 dO, dqkvg) -- and ``amax_dP * scale_dp <= 448`` at the calibrated
    cell (the chart recipe's assertion); at ``scale_dp = 1.0`` the product is printed (saturation there is the regime, not a bug)."""
    res = _cell_backward(cell)
    sc = res.scalars
    for n in ("dy", "do", "dqkvg"):
        prod = sc[f"amax_{n}"] * sc[f"scale_{n}"]
        print(f"{cell.id}: amax_{n} * scale_{n} = {prod:.4g}")
        assert prod <= FP8_E4M3_MAX, (n, sc[f"amax_{n}"], sc[f"scale_{n}"])
        assert sc[f"amax_{n}"] * (2.0 * sc[f"scale_{n}"]) > FP8_E4M3_MAX or sc[f"amax_{n}"] == 0.0, f"{n}: the scale is not the largest power of two (margin 0)"
    print(f"{cell.id}: amax_dP * scale_dp = {sc['amax_dp'] * res.scale_dp:.4g} (scale_dp {res.scale_dp:g})")
    if cell.scale_dp == "calibrated":
        assert sc["amax_dp"] * res.scale_dp <= FP8_E4M3_MAX


@requires_rubin
@pytest.mark.parametrize("how", ["ambient", "explicit"])
def test_fp8_a_caller_stream_orders_every_stage(how):
    """Every stage -- the scalar init, the amax / quantize kernels, the four fp8 GEMMs, the fp8 SDPA adapter -- launches on ONE
    stream, the caller's: ambient (``with torch.cuda.stream(s):``) or explicit (``current_stream=``).  The default stream is parked
    behind a long spin and the workspace is zeroed on the side stream right after the block, so a stage enqueued on the default
    stream runs late and the gradients differ from the default-stream run -- which they must equal BITWISE."""
    import cuda.bindings.driver as cuda_drv

    res = _cell_backward(_BY_ID["s256_causal_b1-norm"])
    side = torch.cuda.Stream()
    ws2 = torch.zeros_like(res.ws)
    grads2 = _alloc_grads(res.blk, fill=0)
    torch.cuda.synchronize()
    park_the_default_stream()
    with torch.cuda.stream(side):
        dy2 = res.dy.clone()
        sdp2 = res.scale_dp_t.clone()
        cs = None if how == "ambient" else cuda_drv.CUstream(side.cuda_stream)
        _execute_fp8(res.blk, res.inp, res.saved, dy2, grads2, ws2, scale_dp=sdp2, current_stream=cs)
        ws2.zero_()
    torch.cuda.synchronize()
    for name, ten in grads2.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), f"{name}: a stage escaped the caller's stream ({how})"


@requires_rubin
def test_fp8_convenience_wrapper_matches_the_class():
    """``gated_attention_block_backward(..., quant=, grad_scaling=, scale_dp=)`` allocates, caches the compiled block (``quant``
    and ``grad_scaling`` in the key) and delegates: its outputs are ``torch.equal`` the class path's."""
    res = _cell_backward(_BY_ID["s256_causal_b1-norm"])
    inp, saved = res.inp, res.saved
    for t_ in (saved.h, inp["w_qkvg"], inp["w_o"], inp["w_q_norm"], inp["w_k_norm"]):
        t_.requires_grad_(True)
    try:
        out = gated_attention_block_backward(
            res.dy,
            saved,
            inp["w_qkvg"],
            inp["w_q_norm"],
            inp["w_k_norm"],
            inp["cos"],
            inp["sin"],
            inp["w_o"],
            res.geom,
            quant=res.spec,
            scale_dp=res.scale_dp_t,
        )
        torch.cuda.synchronize()
        for name in ("dh", "dw_qkvg", "dw_o", "dw_q_norm", "dw_k_norm"):
            assert torch.equal(out[name], res.grads[name]), name
    finally:
        for t_ in (saved.h, inp["w_qkvg"], inp["w_o"], inp["w_q_norm"], inp["w_k_norm"]):
            t_.requires_grad_(False)


@requires_rubin
def test_fp8_execute_scalar_contracts_are_typed():
    """On a COMPILED fp8 block (execute refuses an uncompiled block first), the scalar inputs both directions (Rule 1): ``scale_dp``
    missing; a CPU / fp64 / 2-element ``scale_dp``; ``scale_dy / scale_do / scale_dqkvg`` given under "current"; missing under
    "delayed"; and on a compiled bf16 block ``scale_dp`` given -- each a ``ValueError`` naming the attribute, before any launch."""
    res = _cell_backward(_BY_ID["s256_causal_b1-norm"])
    blk, inp, saved, dy, ws = res.blk, res.inp, res.saved, res.dy, res.ws
    grads = _alloc_grads(blk)
    run = lambda **kw: _execute_fp8(blk, inp, saved, dy, grads, ws, **kw)  # noqa: E731
    with pytest.raises(ValueError, match="scale_dp"):
        run(scale_dp=None)
    with pytest.raises(ValueError, match="scale_dp"):
        run(scale_dp=torch.full((1,), 1.0, dtype=torch.float32))
    with pytest.raises(ValueError, match="scale_dp"):
        run(scale_dp=torch.full((1,), 1.0, dtype=torch.float64, device="cuda"))
    with pytest.raises(ValueError, match="scale_dp"):
        run(scale_dp=torch.full((2,), 1.0, dtype=torch.float32, device="cuda"))
    with pytest.raises(ValueError, match="scale_dy"):
        run(scale_dp=res.scale_dp_t, scale_dy=_dev_scalar(1.0))
    delayed = _backward_fp8(res.geom_kw, res.batch, res.seq_len, grad_scaling="delayed", scales=dict(scale_dy=1.0, scale_do=1.0, scale_dqkvg=1.0))
    with pytest.raises(ValueError, match="scale_dy|scale_do|scale_dqkvg"):
        _execute_fp8(delayed.blk, delayed.inp, delayed.saved, delayed.dy, _alloc_grads(delayed.blk), delayed.ws, scale_dp=delayed.scale_dp_t)


@requires_rubin
def test_fp8_dense_tail_has_no_record():
    """A dense ``S % 128 != 0`` has no quantized record to run a backward over: the quantized forward declines it typed at
    ``check_support`` (no padding mask and no causal mask covering the KV tail) -- the complement of the padded causal cells."""
    _dense_tail_declined({**_COMMON, "qk_norm": True, "is_causal": False}, 1, 992, "fp8")


# ---------------------------------------------------------------------------
# REJECT (any CUDA device; a DECLARED block, no compile, no launch)
# ---------------------------------------------------------------------------


@requires_cuda
def test_a_quantized_record_on_a_bf16_block_is_the_typed_decline():
    """The bf16 backward handed the fp8 training record AS WRITTEN (``saved.h`` = e4m3 codes) raises the record decline naming
    the e4m3 codes -- the pre-existing pin, kept; the quantized backward (``quant=QuantSpec``) is what consumes that record."""
    r = _fp8_decl(dict(_COMMON), 1, 256, quant=None)
    with pytest.raises(ValueError, match="e4m3 codes"):
        r.blk.check_support()


@requires_cuda
def test_e4m3_weights_without_quant_are_typed():
    """A bf16-declared backward (dequantized record) given e4m3 ``w_qkvg`` / ``w_o`` samples is declined naming the operand."""
    r = _fp8_decl(dict(_COMMON), 1, 256, quant=None)
    inp16 = _dequantized_bf16_inputs(r.inp, r.spec, "fp8")
    saved16 = dataclasses.replace(r.saved, h=inp16["h"])
    blk = _declare_fp8_bwd(r.dy, saved16, {**inp16, "w_qkvg": r.inp["w_qkvg"]}, r.geom)
    with pytest.raises(ValueError, match="w_qkvg"):
        blk.check_support()
    blk = _declare_fp8_bwd(r.dy, saved16, {**inp16, "w_o": r.inp["w_o"]}, r.geom)
    with pytest.raises(ValueError, match="w_o"):
        blk.check_support()


@requires_cuda
def test_fp8_reject_quant_of_a_wrong_type():
    """``quant`` that is not a ``QuantSpec`` (a string, a dict) is a typed decline naming the attribute."""
    with pytest.raises((TypeError, ValueError), match="quant"):
        _declare_then_check(lambda: _fp8_decl(dict(_COMMON), 1, 256, quant="e4m3").blk)


@requires_cuda
def test_fp8_reject_grad_scaling_vocabulary():
    """``grad_scaling`` outside ``("current", "delayed")`` -- a declaration ATTRIBUTE (numerics-changing), never a knob."""
    with pytest.raises(ValueError, match="grad_scaling"):
        _declare_then_check(lambda: _fp8_decl(dict(_COMMON), 1, 256, grad_scaling="static").blk)
    assert _api_const("_GRAD_SCALING") == ("current", "delayed")


@requires_cuda
def test_fp8_reject_quant_over_a_bf16_record():
    """``quant=QuantSpec`` over a record whose ``saved.h`` is bf16 (a bf16 forward's record, or the dequantized h): the fp8-declared
    backward needs the quantized forward's record -- a typed decline naming ``saved.h``."""
    r = _fp8_decl(dict(_COMMON), 1, 256, quant=None)
    inp16 = _dequantized_bf16_inputs(r.inp, r.spec, "fp8")
    saved16 = dataclasses.replace(r.saved, h=inp16["h"])
    with pytest.raises(ValueError, match="saved.h"):
        _declare_then_check(lambda: _declare_fp8_bwd(r.dy, saved16, r.inp, r.geom, quant=r.spec))


@requires_cuda
def test_fp8_reject_bf16_weights_with_quant():
    """``quant=QuantSpec`` with bf16 ``w_qkvg`` / ``w_o`` samples: the quantized backward's weights are the forward's e4m3 codes."""
    r = _fp8_decl(dict(_COMMON), 1, 256, quant=None)
    inp16 = _dequantized_bf16_inputs(r.inp, r.spec, "fp8")
    with pytest.raises(ValueError, match="w_qkvg|w_o|quant"):
        _declare_then_check(lambda: _declare_fp8_bwd(r.dy, r.saved, {**r.inp, "w_qkvg": inp16["w_qkvg"], "w_o": inp16["w_o"]}, r.geom, quant=r.spec))


@requires_cuda
def test_fp8_reject_thd_with_quant():
    """``thd=True`` with ``quant``: dense-only for now (the fp8 row's packed chain serves no external delta, and the block's delta
    contract forbids the row's own pre-pass) -- declined typed AT DECLARATION, naming BOTH attributes, and the message does NOT
    tell the caller to build the plan without ``external_delta`` (the adapter's text must never surface here).  Host-side, over
    placeholders shaped like a packed record (``[T, d_model]`` bf16 dy, e4m3 ``saved.h``, int32 ``saved.seq_lens``)."""
    r = _fp8_decl(dict(_COMMON), 1, 256, quant=None)
    t, dm = 256, _COMMON["d_model"]
    dy = torch.empty(t, dm, dtype=torch.bfloat16, device="cuda")
    h8 = torch.empty(t, dm, dtype=_E4M3, device="cuda")
    lens = torch.tensor([128, 128], dtype=torch.int32, device="cuda")
    z = torch.empty(0, device="cuda")
    saved = SavedForBackward(h=h8, gate=z, o=z, lse=z, rstd_q=z, rstd_k=z, seq_lens=lens, seq_lens_form="lengths")
    with pytest.raises(ValueError, match="thd") as ei:
        _declare_then_check(lambda: _declare_fp8_bwd(dy, saved, r.inp, r.geom, quant=r.spec, thd=True, num_sequences=2, max_seq_len=256))
    msg = str(ei.value)
    assert "quant" in msg, msg
    assert "external_delta" not in msg, msg


@requires_cuda
def test_fp8_reject_mxquantspec():
    """An ``MxQuantSpec`` on the per-tensor fp8 backward is a typed ``NotImplementedError`` (the MXFP8 backward is its own row)."""
    r = _fp8_decl(dict(_COMMON), 1, 256, quant=None)
    mx = MxQuantSpec(descale_w_o=r.spec.descale_w_o, scale_o=r.spec.scale_o)
    with pytest.raises(NotImplementedError, match="MxQuantSpec|quant"):
        _declare_then_check(lambda: _declare_fp8_bwd(r.dy, r.saved, r.inp, r.geom, quant=mx))


@requires_cuda
def test_fp8_reject_e5m2_spec_validate():
    """``QuantSpec.validate()`` serves e4m3 ONLY -- ``QuantSpec``'s own decline, independent of the backward."""
    r = _fp8_decl(dict(_COMMON), 1, 256, quant=None)
    with pytest.raises(NotImplementedError, match="QuantSpec"):
        dataclasses.replace(r.spec, dtype=torch.float8_e5m2).validate()


@requires_cuda
def test_fp8_reject_e5m2_through_the_block():
    """An e5m2 ``QuantSpec`` through the block's declaration surfaces ``QuantSpec``'s typed decline, before any stage."""
    r = _fp8_decl(dict(_COMMON), 1, 256, quant=None)
    e5 = dataclasses.replace(r.spec, dtype=torch.float8_e5m2)
    with pytest.raises(NotImplementedError, match="QuantSpec|e5m2|quant"):
        _declare_then_check(lambda: _declare_fp8_bwd(r.dy, r.saved, r.inp, r.geom, quant=e5))


@requires_cuda
def test_fp8_reject_fp16_dy_under_quant():
    """The quantized backward's activation dtype is bf16 (the quantized forward's record is bf16): an fp16 ``sample_dy`` with
    ``quant`` is a typed decline naming ``dy``."""
    r = _fp8_decl(dict(_COMMON), 1, 256, quant=None)
    with pytest.raises((ValueError, NotImplementedError), match="dy"):
        _declare_then_check(lambda: _declare_fp8_bwd(r.dy.to(torch.float16), r.saved, r.inp, r.geom, quant=r.spec))


@requires_cuda
def test_fp8_reject_window_knobs_the_row_cannot_serve():
    """``window_left=0`` and ``window_right > 0`` are declined by the BLOCK under quant exactly as under bf16, naming the field."""
    with pytest.raises(NotImplementedError, match="window_left"):
        _declare_then_check(lambda: _fp8_decl({**_COMMON, "window_left": 0}, 1, 256).blk)
    with pytest.raises(NotImplementedError, match="window_right"):
        _declare_then_check(lambda: _fp8_decl({**_COMMON, "is_causal": True, "window_right": 64}, 1, 256).blk)


@requires_cuda
def test_fp8_reject_padding():
    """Padding -- ``seq_lens_present=True``, or a record carrying a ``seq_lens`` tensor -- is declined under quant at declaration,
    naming ``seq_lens`` (the presence is the fact, never the values: no device sync)."""
    with pytest.raises(NotImplementedError, match="seq_lens"):
        _declare_then_check(lambda: _fp8_decl(dict(_COMMON), 1, 256, seq_lens_present=True).blk)
    r = _fp8_decl(dict(_COMMON), 1, 256, quant=None)
    lens = torch.full((1,), 256, dtype=torch.int32, device="cuda")
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with pytest.raises(NotImplementedError, match="seq_lens"):
            _declare_then_check(lambda: _declare_fp8_bwd(r.dy, dataclasses.replace(r.saved, seq_lens=lens), r.inp, r.geom, quant=r.spec))
    finally:
        torch.cuda.set_sync_debug_mode(prev)


@requires_cuda
def test_fp8_reject_need_star_both_directions():
    """``need_*`` all False leaves no work; ``need_dw_norms=True`` under a RoPE-only geometry has no norm weights -- both typed under
    quant exactly as under bf16."""
    with pytest.raises(ValueError, match="need_dh"):
        _declare_then_check(lambda: _fp8_decl(dict(_COMMON), 1, 256, need_dh=False, need_dw_qkvg=False, need_dw_o=False, need_dw_norms=False).blk)
    with pytest.raises(ValueError, match="need_dw_norms"):
        _declare_then_check(lambda: _fp8_decl({**_COMMON, "qk_norm": False}, 1, 256, need_dw_norms=True).blk)


@requires_cuda
def test_fp8_reject_fuse_wgrad_overlap_without_a_wgrad():
    """``fuse_wgrad_overlap=True`` with neither weight gradient has nothing to overlap -- typed, naming the knob, under quant too."""
    with pytest.raises(ValueError, match="fuse_wgrad_overlap"):
        _declare_then_check(lambda: _fp8_decl(dict(_COMMON), 1, 256, need_dw_o=False, need_dw_qkvg=False, fuse_wgrad_overlap=True).blk)


@requires_cuda
def test_fp8_a_ragged_token_count_is_served_with_its_weight_gradients():
    """There is NO ``B*S % 16`` decline on the fp8 backward: the two weight-gradient GEMMs contract over the token axis with
    MN-major e4m3 operands (an M-major A, an N-major B) and the TMA 16-byte contiguous-extent rule binds an operand's CONTIGUOUS
    axis only -- so ``S = 1000, B = 1`` passes every block-level check with the default ``need_*``, with one weight gradient and
    with none (on a non-Rubin host the device gate is the only decline left, and it names Rubin; on Rubin the cell runs in the
    matrix as ``s1000_causal_b1``)."""
    for kw in ({}, dict(need_dw_qkvg=False), dict(need_dw_o=False, need_dw_qkvg=False)):
        r = _fp8_decl(dict(_COMMON), 1, 1000, **kw)
        try:
            r.blk.check_support()
        except NotImplementedError as e:
            assert _cc() != _SM107 and "Rubin" in str(e), str(e)


@requires_cuda
def test_the_matrix_declares_what_the_module_says():
    """Host, no launch: every matrix row is a shape the quantized forward CAN record (a causal tail at S % 128 != 0 and a dense
    multiple of 128 only), the ragged-token row (T = 1000) keeps its weight gradients (no B*S rule), the dgrad-only row drops
    exactly the two wgrads, and the calibrated row is the bitwise cell's geometry."""
    for c in _CELLS:
        assert c.causal or c.s % 128 == 0, f"{c.id}: a dense S % 128 != 0 has no record"
    served, only = _BY_ID["s1000_causal_b1-norm"], _BY_ID["s1000_causal_b1_dgrad_only-norm"]
    assert (served.b * served.s) % 16 == 8 and served.bwd_kw == {} and served.need_dw_o and served.need_dw_qkvg
    assert only.bwd_kw == dict(need_dw_o=False, need_dw_qkvg=False)
    calib, base = _BY_ID["s512_causal_b2_calib-norm"], _BITWISE_CELL
    assert calib.scale_dp == "calibrated" and (calib.s, calib.causal, calib.b, calib.h_kv, calib.qk_norm) == (
        base.s,
        base.causal,
        base.b,
        base.h_kv,
        base.qk_norm,
    )
