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
    s512_causal_b2_scale_dp_1 S=512 causal B=2 GQA 8/2 norm             scale_dp = 1.0 (every other cell: the CALIBRATED scale_dp, the chart recipe)
    s1000_causal_b1_dgrad_only S=1000 causal B=1 GQA 8/2 norm            need_dw_o=False, need_dw_qkvg=False: the partial-need_* path
    s992_causal_b1_mha  S=992  causal B=1 MHA 8/8  norm                  LAUNCH COUNT + the bitwise layer (not a matrix cell): the pads without the dK fold, 28
    s384_causal_b1      S=384  causal B=1 GQA 8/2  norm                  LAUNCH COUNT + the bitwise layer (not a matrix cell): the kv-side pads ALONE (S % 256 != 0 only), 26

Stage-localised bounds -- none invented, each applied where it was calibrated: the quantizers and every derived scalar
BITWISE (torch's saturating RNE cast; the "current" scale formula ``grad_scale_from_amax``) -- ``og8`` against the KERNELS'
own bf16 ``O_gated``: the forward workspace's ``o8`` bytes (kernel vs kernel) and torch's cast of the forward gate stage's
``O * sigmoid(G)`` re-run over the same O / gate, because both kernels spell the sigmoid as the tanh identity (``tanh(g / 2)
/ 2 + 1 / 2``, one MUFU) whose last fp32 bit differs from torch's ``sigmoid`` on a handful of elements that sit on a bf16
rounding midpoint (21-83 of 2-4M codes at three cells of the first run); the torch-sigmoid composition is reported and every
differing code is required to sit on such an element; ``delta`` bitwise the SDPA chain's own ``dot_do_o`` over the same bf16
O / dO; the GEMMs on the block's own e4m3 operands under the GEMM suite's
bound (``rtol 2^-7``, ``atol = rtol * max|ref|``: an fp8 input is exact in fp64) -- B1 / B7 / B8 directly; B2 (the out-projection
dgrad) only as the composite ``dO = bf16(B2) * sigmoid(G)`` under the bf16 block's bound, because the gate backward overwrites
B2's output in place (B2 on its own is under the GEMM bound in the stage module's ``B2_do_gated`` cell); the SDPA stage under the fp8 row's
recipe (``_FP8_GRAD_TOL`` atol 0.08 / rtol 0.2 with ``assert_close_fp8_grad``'s flip budget, ``amax_dP`` under
``_AMAX_DS_TOL``); ``dh / dW_*`` against the oracle SEEDED with the block's own bf16 dQ / dK / dV -- and fed the record's
exact LSE, bf16 pre-gate O, bf16 GATE band and e4m3 ``q8 / k8 / v8``, the inputs the backward reads (the first full run fed the
oracle's own fp64 attention O and exact projection instead: the fp8 forward's P cast then put dG at 1.2-3.1x and dW_o at
3.8-5.3x the bound on every cell, and the gate's bf16 rounding alone still flipped 0.1-0.6 % of the og8 codes; the second run
fed its own cast of its fp64 Q / K / V to the modelled SDPA stage: a few per cent of the codes flipped against the record's,
and a P recomputed from them under the record's LSE put the modelled dh at cos 0.996 -- composition gaps of the reference,
not kernel margins) -- ``dh / dW_o`` under the bf16 block's bound (``rtol 2^-6``, ``atol 2^-7 * max|ref|``,
``cos >= 0.999``; ``dW_norm``: ``2^-5 * mass + 1e-2 * |ref|``) and ``dW_qkvg`` in the module's row-budgeted form WITH its attribution
(``_assert_seeded_dw_qkvg_row_budgeted``: the rows with a cell outside that bound against ``1e-5 x rows x keys``, keys = T -- the
statistic ``assert_close_fp8_grad`` and the (M) layer are judged by -- and every such row BOTH a row a ``dqkvg8`` code flip touched
AND a slab column whose PRE-cast bf16 band is itself inside that bound against the seeded oracle (the bands sit at 0.08-0.24 of
it): the per-cell bound is where the slab's single near-amax e4m3 flips land, 1.02-1.65x on 7 of 15 cells, each flip at ``[t, n]``
moving exactly row ``n`` by ``flip * h[t, :]``, and the pre-cast condition is what tells the CAST's rounding from a band's miss --
a defect upstream of the cast flips codes in exactly the columns it corrupts, so "touched" alone would hold by construction; the
per-cell worst stays printed); end-to-end against
the fully MODELLED oracle and the unquantized-gradient one is printed (``cos``, ``max|diff| / max|ref|``, the rows outside the
bf16 bound against the ``1e-5 x rows x keys`` row budget), and the (M) one is ASSERTED in exactly that row-budget form by
``test_fp8_end_to_end_modelled_is_row_budgeted`` now that the first run measured it.  The modelled oracle's SDPA stage is fed the
kernel's inputs -- the record's LSE, e4m3 ``q8 / k8 / v8`` and the block's bf16 dO (its ``delta`` is that dO's row-sum) -- so the
end-to-end difference is the SDPA stage's kernel-vs-reference difference propagated (fed its own cast of its fp64 chain instead, a
few per cent of those codes flipped against the record's LSE / delta and the modelled dh sat at cos 0.996, 75 % of its rows outside:
a composition gap, not a kernel margin).  What used to propagate was the kernel's GQA dK / dV fold -- bf16 per-Q-head partials
summed, where the reference rounds once (relative RMS 2.7e-3 on dK / dV under GQA, 0 under MHA) -- which put 14-81 of the 5120
``dW_qkvg`` rows and the ``dh`` of the three dense cells outside the bf16 bound against row budgets of 13-52 (OVER on 10 of 15
cells, every one under GQA; the two MHA cells inside).  The row now folds from fp32 partials (the kernel's dK is bitwise the
once-rounded reference's under GQA, dV within 1.4e-4 relative RMS of it), and every cell is inside its budget (the table at the end:
at most 18 dw_qkvg rows outside of a 41.9-row budget, 0 dh rows).  The row budget is asserted on every output of every cell --
``dw_o`` everywhere and ``dh / dw_qkvg`` on the MHA cells in ``test_fp8_end_to_end_modelled_is_row_budgeted``, ``dh / dw_qkvg`` of
the 13 GQA cells (the chain the fold sits in) in ``test_fp8_end_to_end_modelled_gqa_fold_is_row_budgeted`` -- plainly: the strict
``xfail`` the 10 over-budget cells carried while the fold rounded ``group`` times is gone; never widened.  The margins are a property
of ONE dataset: the forward inputs and dY are device-Philox draws (``torch.Generator(device="cuda")`` in the reference's input
builder and in ``_make_dy``), which torch lays out by grid size -- the part's SM count -- so a Rubin part with another SM count (the
212-SM parts the bf16 module's ``dW_norm`` noise floor was calibrated on) draws different tensors under the same seed.  A failure of
this layer on another SM count is the dataset moving a cell, not the kernel: re-measure there before touching the form; the durable
fix -- drawing this module's dataset on a CPU generator, SM-independent, and re-recording both tables -- is a follow-up.
The SDPA
stage's kernel-vs-reference difference is characterised per cell (``_report_stage_difference``: relative RMS, d-rows outside the
bf16 bound form, the d-rows carrying 90 % of the squared difference) AND asserted under the bf16 block's bound form
(``_assert_grad_close`` on the stage's bf16 dQ / dK / dV: 0 d-rows outside on every cell of both regimes), because the row
recipe's absolute ``atol 0.08`` is at or above ``max|dQ| / max|dK|`` at this geometry in true units (at any ``scale_dp``): it
cannot tell a sparse flip class from a diffuse miss, nor an all-zero output from the right one.

Launch count: ``expected_fp8_launches`` (host-checkable) -- the block's own launches (10 with every gradient: the fused
PROLOGUE (scalar init -- the zeroing, ``descale_dp`` and the plan-time constants from its kernel arguments -- + dY amax partials + the
Q / K rebuild with its e4m3 epilogue + v8); quantize dY; B2; B3 (+ per-CTA partials of max |dO| and max |dG|); quantize dO (reduces
B3's dO partials); B1; B5+B6 (+ the bands' per-CTA partials); the fused EPILOGUE (dW_norm reduce + quantize dqkvg over the dG and
band partials -- one launch under either qk_norm arm); B7; B8) + the fp8 row's ``setup (fill_i32 + the amax resets, one launch) +
c * (2 + q) + fold (dV, and dK under GQA, one launch) + 3 * q_padded + 2 * kv_padded + zero_ws`` -- 15 at ``s512_causal_b2-norm``
(c = 1, q = 1), 15 rope_only, 15 MHA, 20 padded (s992 and both s1000 cells with weight gradients), 18 dgrad-only padded, 20 padded x
MHA, 17 at the kv-side pads alone (S = 384: the ``+ 2`` measured apart from the ``+ 3``); 17 / 17 / 16 / 22 / 20 / 21 / 19 before the
row merged its setup and fold launches, 24 / 23 / 23 / 29 / 27 / 28 / 26 before the launch fusion.  No fold copy-outs on the fp8 row
(its folds write the caller's dK / dV).  The two launch-count-only cells also get the bitwise + finiteness layer (no oracle).

Rejects match the ATTRIBUTE NAME only (``match="quant"``, ``"scale_dp"``, ``"thd"``, ...): the message prose is owned and
pinned by the API's own test module, so a wording change touches one test.

The matrix runs at the CALIBRATED ``scale_dp`` (``_SCALE_DP_DEFAULT = "calibrated"``: one warm-up execute at ``scale_dp = 1.0``,
``get_fp8_scale_factor(amax_dP)`` -- the chart recipe -- then the measured run), because of a fact from the CPU calibration of the
module constants (``FP8_SCALE_S_LOG2 = 8`` and ``FP8_GRAD_SCALE_MARGIN_LOG2 = 0`` both confirmed): at ``scale_dp = 1.0`` the row
tolerance is near-vacuous for dQ / dK (their ``max|ref|`` is below ``atol``, while 93-99 % of the fp32 dS values flush to zero in
e4m3 -- an all-zero dQ would pass), and the flip proof of the SDPA-stage pin only has evidence to weigh where dS is resolved.  One
cell (``s512_causal_b2_scale_dp_1``) keeps ``scale_dp = 1.0``: the under-scaled regime the chart's warm-up execute runs at, where
the kernel must still be finite, bitwise-stable and exact in its ``amax_dP``.  ``_SCALE_DP_DEFAULT`` is the one constant to flip.

Margins of the calibrated-scale_dp run (Rubin cc 10.7, 204 SMs, SM clock locked at 2376 MHz; worst cell as a fraction of the bound named for
that stage; "rows outside" = rows of dh (tokens) / dW (output rows) with a cell outside the bf16 bound against the
``1e-5 x rows x keys`` row budget).  The seeded oracle is fed the record's LSE, O, gate band, e4m3 ``q8 / k8 / v8`` and the block's
bf16 dO.  The ``dw_qkvg`` cells above 1.0 (7 of 15, 1.021-1.649x) are near-amax ``dqkvg8`` code flips (an e4m3 ulp there is
``32 / scale_dqkvg``), each moving one ``dW_qkvg`` row by ``flip * h[t, :]``: 1-18 rows outside against row budgets of 13-102, every one
a row a flip touched whose pre-cast slab column sits inside its band's bound (the bands at 0.08-0.24 of it).  ``dw_qkvg`` is
therefore judged in the module's row-budgeted form with that attribution asserted (``_assert_seeded_dw_qkvg_row_budgeted``: budget,
flip-touched, pre-cast band inside -- counts of this run, conditions of the form), ``dh / dw_o / dW_norm`` under the per-cell
bound::

    cell                            dO    B1    B7    B8   bands dq_pre/dg/dk_pre  dqkvg8 flips  og8 flips  dh    dw_qkvg dw_o   dWq_n dWk_n  rows outside dh / dw_qkvg / dw_o (budget)
    s256_causal_b1-norm             0.245 0.213 0.204 0.155 0.122/0.193/0.116      12467         0          0.273 1.056   -      -     -      0/256 (13.1) / 2/5120 (13.1) / 0/512 (1.31)
    s256_causal_b1-rope_only        0.245 0.167 0.160 0.184 0.083/0.186/0.089      8016          0          0.167 0.677   0.124  -     -      0/256 (13.1) / 0/5120 (13.1) / 0/512 (1.31)
    s512_causal_b2-norm             0.245 0.170 0.184 0.156 0.149/0.201/0.114      49448         0          0.290 0.872   0.127  0.135 0.152  0/1024 (52.4) / 0/5120 (52.4) / 0/512 (5.24)
    s512_causal_b2-rope_only        0.245 0.201 0.194 0.171 0.099/0.201/0.118      31456         0          0.187 0.576   0.143  -     -      0/1024 (52.4) / 0/5120 (52.4) / 0/512 (5.24)
    s992_causal_b1-norm             0.245 0.215 0.212 0.148 0.149/0.211/0.126      47622         0          0.309 1.388   -      -     -      0/992 (50.8) / 6/5120 (50.8) / 0/512 (5.08)
    s1000_causal_b2-norm            0.137 0.168 0.182 0.201 0.152/0.159/0.140      96030         83         0.720 1.649   -      -     -      0/2000 (102) / 4/5120 (102) / 0/512 (10.2)
    s1000_causal_b1-norm            0.245 0.208 0.215 0.152 0.149/0.211/0.141      47875         0          0.224 1.146   -      -     -      0/1000 (51.2) / 4/5120 (51.2) / 0/512 (5.12)
    s256_dense_b1-norm              0.245 0.207 0.186 0.216 0.132/0.225/0.124      13190         0          0.776 1.021   -      -     -      0/256 (13.1) / 1/5120 (13.1) / 0/512 (1.31)
    s256_dense_b1-rope_only         0.245 0.169 0.197 0.180 0.109/0.203/0.122      8487          0          0.423 0.212   0.126  -     -      0/256 (13.1) / 0/5120 (13.1) / 0/512 (1.31)
    s1024_dense_b1_mha-norm         0.147 0.181 0.171 0.185 0.143/0.169/0.135      68543         51         0.544 0.840   0.310  0.152 0.135  0/1024 (83.9) / 0/8192 (83.9) / 0/512 (5.24)
    s512_dense_b2-rope_only         0.245 0.204 0.183 0.206 0.121/0.213/0.145      33827         0          0.463 0.165   0.143  -     -      0/1024 (52.4) / 0/5120 (52.4) / 0/512 (5.24)
    s512_causal_b1_mha-norm         0.147 0.201 0.202 0.141 0.119/0.243/0.160      31777         21         0.215 1.544   -      -     -      0/512 (41.9) / 18/8192 (41.9) / 0/512 (2.62)
    s256_causal_b2_rope-rope_only   0.245 0.213 0.199 0.169 0.094/0.193/0.085      16142         0          0.233 0.712   0.148  -     -      0/512 (26.2) / 0/5120 (26.2) / 0/512 (2.62)
    s512_causal_b2_scale_dp_1-norm  0.245 0.170 0.184 0.155 0.149/0.201/0.118      51144         0          0.474 1.514   -      -     -      0/1024 (52.4) / 3/5120 (52.4) / 0/512 (5.24)
    s1000_causal_b1_dgrad_only-norm 0.245 -     -     0.152 0.149/0.211/0.141      47875         -          0.224 -       -      0.140 0.144  0/1000 (51.2) / - / -

(M) end-to-end in the row-budget form (``test_fp8_end_to_end_modelled_is_row_budgeted`` on ``dw_o`` everywhere and on ``dh /
dw_qkvg`` of the MHA cells, ``test_fp8_end_to_end_modelled_gqa_fold_is_row_budgeted`` on ``dh / dw_qkvg`` of the 13 GQA cells; the
modelled oracle fed the record's LSE, O, gate band, e4m3 q8 / k8 / v8 and the block's bf16 dO): cos and rows outside the bf16 bound /
rows (budget ``1e-5 x rows x keys``) per output -- INSIDE the budget on every cell since the SDPA row folds its GQA dK / dV from fp32
per-Q-head partials (one rounding, like the reference; with bf16 partials 10 of 15 cells were over it, 14-81 dw_qkvg rows outside and
17-142 dh rows on the dense cells).  Rubin (cc 10.7, 204 SMs), with fp32 partials::

    cell                             (M) dh: cos  rows out/rows (budget)  (M) dw_qkvg: cos  rows out/rows (budget)  (M) dw_o: cos  rows out/rows (budget)  verdict
    s256_causal_b1-norm              0.999992 0/256 (13.1)   0.999993 3/5120 (13.1)   0.999999 0/512 (1.31)  inside
    s256_causal_b1-rope_only         0.999996 0/256 (13.1)   0.999996 0/5120 (13.1)   0.999999 0/512 (1.31)  inside
    s512_causal_b2-norm              0.999992 0/1024 (52.4)  0.999992 3/5120 (52.4)   0.999999 0/512 (5.24)  inside
    s512_causal_b2-rope_only         0.999996 0/1024 (52.4)  0.999996 0/5120 (52.4)   0.999999 0/512 (5.24)  inside
    s992_causal_b1-norm              0.999992 0/992 (50.8)   0.999993 7/5120 (50.8)   0.999999 0/512 (5.08)  inside
    s1000_causal_b2-norm             0.999992 0/2000 (102)   0.999993 3/5120 (102)    0.999999 0/512 (10.2)  inside
    s1000_causal_b1-norm             0.999992 0/1000 (51.2)  0.999993 9/5120 (51.2)   0.999999 0/512 (5.12)  inside
    s256_dense_b1-norm               0.999992 0/256 (13.1)   0.999992 0/5120 (13.1)   0.999999 0/512 (1.31)  inside
    s256_dense_b1-rope_only          0.999996 0/256 (13.1)   0.999997 0/5120 (13.1)   0.999999 0/512 (1.31)  inside
    s1024_dense_b1_mha-norm          0.999991 0/1024 (83.9)  0.999993 0/8192 (83.9)   0.999999 0/512 (5.24)  inside
    s512_dense_b2-rope_only          0.999996 0/1024 (52.4)  0.999996 0/5120 (52.4)   0.999999 0/512 (5.24)  inside
    s512_causal_b1_mha-norm          0.999993 0/512 (41.9)   0.999993 18/8192 (41.9)  0.999999 0/512 (2.62)  inside
    s256_causal_b2_rope-rope_only    0.999996 0/512 (26.2)   0.999996 0/5120 (26.2)   0.999999 0/512 (2.62)  inside
    s512_causal_b2_scale_dp_1-norm   0.999993 0/1024 (52.4)  0.999993 2/5120 (52.4)   0.999999 0/512 (5.24)  inside
    s1000_causal_b1_dgrad_only-norm  0.999992 0/1000 (51.2)  -        -               -        -             inside
"""

import dataclasses
import gc
import inspect
import os
import sys
import time
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

from cudnn.gated_attention_block import GatedAttentionBlockBwd, SavedForBackward, gated_attention_block_backward  # noqa: E402
from cudnn.gated_attention_block import api_bwd as _api_bwd  # noqa: E402
from cudnn.gated_attention_block.api import Fp4Format, MxQuantSpec, _cols, _view  # noqa: E402
from cudnn.gated_attention_block.kernels import fp8_bwd_fused as _fused  # noqa: E402
from cudnn.gated_attention_block.kernels import quantize as _quantize  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_reference import FP8_E4M3_MAX, RefGeometry, gated_attention_block_fp8_bwd_reference, quant_e4m3  # noqa: E402
from gated_block_stream_probe import park_the_default_stream  # noqa: E402
from test_block_backward import (  # noqa: E402
    _ATOL_FRAC,
    _COMMON,
    _KNOBS,
    _RTOL,
    _alloc_grads,
    _assert_dw_norm_close,
    _assert_grad_close,
    _cos,
    _declare_bwd,
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

# The appended keyword-only parameters of the quantized backward (append-only, defaulted): the per-tensor fp8 tail, which the MXFP8
# backward's four artifacts (``test_block_backward_mxfp8.py``) follow as the LAST parameters of ``execute`` and the wrapper.
_FP8_INIT_KWARGS = ("quant", "grad_scaling")
_FP8_EXECUTE_KWARGS = ("scale_dp", "scale_dy", "scale_do", "scale_dqkvg")
_MX_EXECUTE_KWARGS = ("h_t", "h_t_sf", "w_qkvg_t", "w_qkvg_t_sf")
_FP4_EXECUTE_KWARGS = ("w_o_t", "w_o_t_sf")  # the fp4 weight modes' appended pair, after the MXFP8 artifacts


def test_the_fp8_surface_is_an_appended_keyword_only_tail():
    """Host, no GPU: the quantized backward's parameters are the LAST parameters of ``__init__``, ``execute`` and the convenience
    wrapper -- the fp8 tail directly followed by the MXFP8 backward's four appended artifacts and the fp4 weight modes' two on
    ``execute`` and the wrapper -- keyword-only and defaulted, in the declared order (public signatures evolve append-only)."""
    for fn, names in (
        (GatedAttentionBlockBwd.__init__, _FP8_INIT_KWARGS),
        (GatedAttentionBlockBwd.execute, _FP8_EXECUTE_KWARGS + _MX_EXECUTE_KWARGS + _FP4_EXECUTE_KWARGS),
        (gated_attention_block_backward, _FP8_INIT_KWARGS + _FP8_EXECUTE_KWARGS + _MX_EXECUTE_KWARGS + _FP4_EXECUTE_KWARGS),
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
    scale_dp: object = "default"  # a float, "default" (= _SCALE_DP_DEFAULT, the calibrated chart recipe) or "calibrated" explicitly
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
    + [
        _Cell(
            "s512_causal_b2_scale_dp_1",
            512,
            True,
            2,
            2,
            True,
            scale_dp=1.0,
            note="the bitwise cell's geometry at scale_dp = 1.0: the under-scaled regime (most of dS flushes to zero in e4m3) the chart's warm-up runs at",
        )
    ]
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
# Launch-count-only cells: NOT in the matrix (no GEMM / SDPA / seeded / end-to-end layer) -- shapes whose launch-count arm no
# matrix cell reaches, run by ``test_fp8_launch_count_is_honest`` and the bitwise + finiteness layer
# (``test_fp8_launch_only_cells_are_finite_and_quantize_bitwise``).  padded x MHA: the q / kv staging pads WITHOUT the dK fold;
# S = 384: the kv-side pads ALONE (S % 128 == 0, S % 256 != 0 -- the "+2" term measured apart from the "+3").
_LAUNCH_ONLY_CELLS = [
    _Cell("s992_causal_b1_mha", 992, True, 1, 8, True, note="padded x MHA (+3 q pads, +2 kv pads, no dK fold): launch count + the bitwise layer"),
    _Cell("s384_causal_b1", 384, True, 1, 2, True, note="kv-side pads ONLY (S % 128 == 0, S % 256 != 0): +2 without the +3; launch count + the bitwise layer"),
]
_BY_ID = {c.id: c for c in _CELLS + _LAUNCH_ONLY_CELLS}
assert len(_CELLS) == 15 and len(_BY_ID) == len(_CELLS) + len(_LAUNCH_ONLY_CELLS), "the cell ids must be unique: 12 matrix rows, three in both qk_norm arms"
_MATRIX = pytest.mark.parametrize("cell", _CELLS, ids=[c.id for c in _CELLS])
_BITWISE_CELL = _BY_ID["s512_causal_b2-norm"]
# The matrix's default scale_dp: the CALIBRATED one (the chart recipe -- one warm-up execute at 1.0, then get_fp8_scale_factor(amax_dP),
# then the measured run), because at scale_dp = 1.0 the row tolerance is near-vacuous for dQ / dK (module docstring); one cell
# keeps scale_dp = 1.0 for that regime.  A float here runs the whole matrix at one scale instead.
_SCALE_DP_DEFAULT = "calibrated"
_KNOB_SETS = pytest.mark.parametrize("knobs", list(_KNOBS.values()), ids=list(_KNOBS))
# The (M) row budget under GQA: the fp8 SDPA row folds its GQA dK / dV from fp32 per-Q-head partials and rounds ONCE, like the
# reference, so dh / dw_qkvg of the 13 GQA cells are asserted plainly in test_fp8_end_to_end_modelled_gqa_fold_is_row_budgeted (with
# bf16 partials summed -- `group` roundings -- 10 of them were over the 1e-5 x rows x keys row budget and carried a strict xfail).
# The margins are the 204-SM dataset's (device-Philox draws follow the SM count; module docstring): on a part with another SM count
# a cell can move -- re-measure there before reading a failure as the kernel's; the SM-independent dataset is a follow-up.
_GQA_CELLS = [c for c in _CELLS if c.group > 1]
_GQA_FOLD_MATRIX = pytest.mark.parametrize("cell", _GQA_CELLS, ids=[c.id for c in _GQA_CELLS])


# ---------------------------------------------------------------------------
# The launch-count formula (host-checkable; the CUPTI cell asserts formula == expected == len(kernels))
# ---------------------------------------------------------------------------


def fp8_block_launch_table(*, qk_norm: bool, need_dw_o: bool = True, need_dw_qkvg: bool = True, need_dh: bool = True) -> list:
    """The block's OWN launches under ``quant``, in launch order, as ``(label, count, present)`` -- the module docstring's
    table as data, so the count is derived from it and never re-literalled.  The prologue is ONE launch for the scalar init,
    the dY amax partials, the Q / K rebuild with its e4m3 epilogue and v8; the epilogue ONE launch for the dW_norm reduce
    (its blocks exist under ``qk_norm`` only -- the launch exists regardless, for the dqkvg quantize) and the dqkvg quantize;
    every gradient amax is per-CTA partials of its producer (B3's dO partials for the dO quantize, B3's dG + B5+B6's band
    partials for the dqkvg quantize), reduced by the consuming launch -- no amax pass, no atomic."""
    return [
        ("prologue: init_scalars + amax dY partials + recompute Q, K -> q8 / k8 + v8", 1, True),
        ("quantize dY (reduces the partials, publishes amax_dy; persistent)", 1, True),
        ("B2 out_proj dgrad", 1, True),
        ("B3 sigmoid_gate_bwd (fp8 arm: dO, dG, og8, delta, per-CTA partials of max |dO| and max |dG|; persistent)", 1, True),
        ("quantize dO (reduces B3's dO partials, publishes amax_do; persistent)", 1, True),
        ("B1 out_proj wgrad", 1, need_dw_o),
        ("B5+B6 qk_norm_rope_bwd (+ the bands' per-CTA partials)", 1, True),
        ("epilogue: dW_norm reduce (qk_norm) + quantize dqkvg (reduces the dG + band partials, publishes amax_dqkvg)", 1, True),
        ("B7 qkv_gate wgrad", 1, need_dw_qkvg),
        ("B8 qkv_gate dgrad", 1, need_dh),
    ]


def fp8_row_launches(*, group: int, chunks: int, dq_launches: int, q_padded: bool, kv_padded: bool, zero_ws: bool) -> int:
    """The fp8 SDPA row's launches under an EXTERNAL delta: ONE setup launch (``_fp8_setup``: the uniform kv-length fill, unconditional
    on a dense plan, together with the amax resets) + per head chunk ``main + dK + dq_launches x dQ`` + ONE fold launch (the dV fold
    always; under GQA the dK fold shares its launch -- ``fold_quant_pair``) + the staging pads (3 on the q side: q, dO, lse; 2 on the
    kv side: k, v) + the dS zero-fill when the adapter says so.  NO ``dot_do_o`` (the caller's delta replaces it) and NO fold
    copy-outs (the folds write the caller's dK / dV).  ``group`` no longer moves the count (it selects the fold kernel's form)."""
    del group  # the GQA dK fold rides the dV fold's launch
    return 1 + chunks * (2 + dq_launches) + 1 + (3 if q_padded else 0) + (2 if kv_padded else 0) + (1 if zero_ws else 0)


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
    chunks = -(-blk.batch // impl._b_chunk) * -(-g.h_q // impl._qh_chunk)  # ceil-div: the adapter's chunk loops cover every batch / head
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
    ("s512_causal_b2-norm", 15),
    ("s512_causal_b2-rope_only", 15),
    ("s512_causal_b1_mha-norm", 15),
    ("s992_causal_b1-norm", 20),
    ("s1000_causal_b2-norm", 20),
    ("s1000_causal_b1-norm", 20),
    ("s1000_causal_b1_dgrad_only-norm", 18),
    ("s992_causal_b1_mha-norm", 20),  # launch-count only: padded x MHA
    ("s384_causal_b1-norm", 17),  # launch-count only: the kv-side pads alone (+2, no +3)
]


def test_fp8_launch_formula_reproduces_the_declared_counts():
    """Host, no GPU: the formula over the matrix cells' facts (c = 1, one dQ launch per chunk under the single-launch dQ
    rendering, no zero-fill at the plain causal / dense cells) gives the declared counts -- 15 at the bitwise cell (norm,
    GQA), 15 rope_only (the epilogue launch carries the dqkvg quantize whether or not it has reduce blocks), 15 MHA (the row's dK
    fold shares the dV fold's launch under GQA, so the group no longer moves the count), 20 at the three padded GQA cells with weight
    gradients (+3 q pads, +2 kv pads), 18 at the padded dgrad-only cell (B1 and B7 gone), 20 at the padded MHA cell, 17 at the
    kv-side-only padded cell (S = 384: the +2 without the +3) -- the last two launch-count-only cells.  The block's own table sums to
    10 with every gradient under either qk_norm arm; the row adds 5 unpadded whatever the group (one setup launch, one fold launch)."""
    for cell_id, want in _LAUNCH_EXPECTATIONS:
        c = _BY_ID[cell_id]
        q_pad, kv_pad = _padded(c)
        got = expected_fp8_launches(qk_norm=c.qk_norm, group=c.group, q_padded=q_pad, kv_padded=kv_pad, need_dw_o=c.need_dw_o, need_dw_qkvg=c.need_dw_qkvg)
        assert got == want, (cell_id, got, want)
    assert sum(n for _l, n, p in fp8_block_launch_table(qk_norm=False) if p) == 10
    assert sum(n for _l, n, p in fp8_block_launch_table(qk_norm=True) if p) == 10
    assert fp8_row_launches(group=4, chunks=1, dq_launches=1, q_padded=False, kv_padded=False, zero_ws=False) == 5
    assert fp8_row_launches(group=1, chunks=1, dq_launches=1, q_padded=False, kv_padded=False, zero_ws=False) == 5
    # two head chunks (c = 2, the 397B geometry at long S): +3 per extra chunk
    assert expected_fp8_launches(qk_norm=True, group=16, chunks=2) == 18


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
    ``scale_dp``: a float, ``"default"`` (``_SCALE_DP_DEFAULT``) or ``"calibrated"`` (one warm-up at 1.0, then the chart recipe
    ``get_fp8_scale_factor(amax_dP)``, then the measured run).  ``scales`` = ``dict(scale_dy=, scale_do=, scale_dqkvg=)`` floats under ``grad_scaling="delayed"``.
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
    scale_dp = _SCALE_DP_DEFAULT if scale_dp == "default" else scale_dp
    scale_dp_t = _dev_scalar(1.0 if scale_dp == "calibrated" else scale_dp)
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


def _resolved_scale_dp(cell: _Cell):
    """``"calibrated"`` or the float the cell runs at (``"default"`` resolved through ``_SCALE_DP_DEFAULT``)."""
    return _SCALE_DP_DEFAULT if cell.scale_dp == "default" else cell.scale_dp


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


def _row_budget(rows: int, keys: int) -> float:
    """``assert_close_fp8_grad``'s row budget -- ``1e-5 x rows x keys``, at least 1 (``keys`` = the reduction length feeding a
    row) -- the ONE formula every row-budgeted statistic of this module is judged against (printed and asserted alike)."""
    return max(1.0, 1e-5 * rows * keys)


def _rows_outside_mask(got: torch.Tensor, ref64: torch.Tensor) -> torch.Tensor:
    """Per ROW -- one token of ``dh``, one output row of a ``dW`` -- whether any cell sits outside the bf16 block's bound
    (``2^-7 max|ref| + 2^-6 |ref|``): a bool vector over the rows, the statistic of the row-budgeted form."""
    g2 = got.detach().double().reshape(-1, got.shape[-1])
    r2 = ref64.detach().double().reshape(-1, got.shape[-1])
    return ((g2 - r2).abs() > _ATOL_FRAC[got.dtype] * r2.abs().max() + _RTOL[got.dtype] * r2.abs()).any(dim=1)


def _rows_outside(got: torch.Tensor, ref64: torch.Tensor) -> tuple:
    """``(rows outside, rows)`` of the bf16 block's bound (``_rows_outside_mask``) -- the statistic of the row-budgeted form
    (``_row_budget``) the flip class downstream of an e4m3 cast is judged by."""
    outside = _rows_outside_mask(got, ref64)
    return int(outside.sum()), int(outside.numel())


def _report_seeded_intermediates(res, v: dict, ref: dict) -> dict:
    """Localisation between B4 and the outputs (no new bound): the block's bf16 ``dqkvg`` bands (B3's dG, B5+B6's dQ_pre / dK_pre)
    against the seeded oracle's fp64 bands as a fraction of the bf16 block's bound, PRINTED; the slab's V band ``torch.equal``
    the block's own dV slot (the norm backward copies it bit for bit -- a copy, asserted); and the e4m3 flips of ``dqkvg8``
    against the cast of the oracle's own bf16-rounded ``dqkvg`` at the block's scale -- a flip at ``[t, n]`` moves the whole
    ``dW_qkvg`` row ``n`` by ``flip * h[t, :]`` and the whole ``dh`` row ``t`` by ``flip * W_qkvg[n, :]`` (the rank-1 flip-class shape,
    budgeted by rows), so a seeded ``dw_qkvg`` / ``dh`` residual is attributable to them.  Returns that flip evidence, computed
    ONCE here and consumed by ``_assert_seeded_dw_qkvg_row_budgeted``: ``flips`` (the bool ``[T, N]`` mask), ``dqkvg8_ref`` (the
    oracle's cast), ``dw_qkvg_rows`` / ``dh_rows`` (the distinct slab columns ``n`` / tokens ``t`` a flip touched) and
    ``band_col_worst`` (per slab column ``n``: the PRE-cast ``dqkvg[:, n]``'s worst cell as a fraction of its band's bf16 bound -- the
    statistic ``_report_close`` prints per band, kept per column, so a row outside downstream can be asked whether its column was
    inside BEFORE the cast)."""
    g, sc = res.geom, res.scalars
    t, d = res.batch * res.seq_len, g.d_head
    o_q, o_g, o_k, o_v = g.qkvg_offsets
    dqkvg = v["dqkvg"]
    bands = ((o_q, g.h_q, "dq_pre", "dq_pre"), (o_g, g.h_q, "dg", "dg"), (o_k, g.h_kv, "dk_pre", "dk_pre"), (o_v, g.h_kv, "dv", "dv_band"))
    ref_slab = torch.empty(t, g.n_qkvg, dtype=torch.float64, device=dqkvg.device)
    band_col_worst = torch.empty(g.n_qkvg, dtype=torch.float64, device=dqkvg.device)  # per slab column: the PRE-cast band's worst cell / its band's bound
    for off, heads, name, key in bands:
        _report_close(_cols(dqkvg, off, heads, d), ref[key], f"band {name} vs the seeded oracle")
        band_ref = ref[key].reshape(t, heads * d)
        ref_slab[:, off : off + heads * d] = band_ref
        # the bound _report_close just printed the band's worst of -- anchored on the BAND's max|ref| -- kept per column, for the
        # attribution of a dW_qkvg row outside (_assert_seeded_dw_qkvg_row_budgeted): was its pre-cast column inside it?
        band_bound = _ATOL_FRAC[dqkvg.dtype] * band_ref.abs().max() + _RTOL[dqkvg.dtype] * band_ref.abs()
        band_col_worst[off : off + heads * d] = ((_cols(dqkvg, off, heads, d).reshape(t, heads * d).double() - band_ref).abs() / band_bound).amax(dim=0)
    assert torch.equal(_cols(dqkvg, o_v, g.h_kv, d).contiguous(), v["dv"]), "the slab's V band is not bitwise the block's own dV slot"
    dqkvg8_ref = quant_e4m3(ref_slab.to(torch.bfloat16), sc["scale_dqkvg"])
    flips = v["dqkvg8"].view(torch.uint8) != dqkvg8_ref.view(torch.uint8)
    rows_t, cols_n = torch.nonzero(flips, as_tuple=True)
    dw_qkvg_rows, dh_rows = torch.unique(cols_n), torch.unique(rows_t)
    print(
        f"dqkvg8 vs e4m3(bf16(seeded oracle dqkvg)) at scale {sc['scale_dqkvg']:g}: {int(flips.sum())} of {flips.numel()} codes differ "
        f"({int((dqkvg != ref_slab.to(torch.bfloat16)).sum())} bf16 cells differ before the cast); {int(dw_qkvg_rows.numel())} dW_qkvg rows "
        f"and {int(dh_rows.numel())} dh rows touched"
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
    for name, keys in _row_keys(res).items():
        if res.grads.get(name) is not None:
            n_out, n_rows = _rows_outside(res.grads[name], ref[name])
            print(
                f"{name} vs the seeded oracle: {n_out} of {n_rows} rows outside the bf16 bound (row budget 1e-5 x rows x keys = {_row_budget(n_rows, keys):.3g})"
            )
    return dict(flips=flips, dqkvg8_ref=dqkvg8_ref, dw_qkvg_rows=dw_qkvg_rows, dh_rows=dh_rows, band_col_worst=band_col_worst)


def _qkvg_band(g, n: int) -> str:
    """Which band of the ``[T, N]`` slab (``q`` / ``g`` / ``k`` / ``v``) row ``n`` of ``dW_qkvg`` belongs to, with its head and d index."""
    d = g.d_head
    for off, heads, name in zip(g.qkvg_offsets, (g.h_q, g.h_q, g.h_kv, g.h_kv), ("q", "g", "k", "v")):
        if off <= n < off + heads * d:
            return f"{name} head {(n - off) // d} d {(n - off) % d}"
    return "?"


def _assert_seeded_dw_qkvg_row_budgeted(res, v: dict, ref: dict, flip_ev: dict, what: str) -> float:
    """The SEEDED ``dW_qkvg`` in the module's row-budgeted form WITH its attribution.  ``dW_qkvg = dqkvg8^T . h8`` is where the
    slab's single near-amax ``dqkvg8`` e4m3 flips land (an ulp there is ``32 / scale_dqkvg``; 1.02-1.65x the per-cell bound on 7 of
    15 cells of the calibrated run), and a flip at ``[t, n]`` moves exactly ``dW_qkvg`` row ``n`` (by ``flip * h[t, :]``) -- so the form
    that describes the class is the one ``assert_close_fp8_grad`` and the (M) layer use, with the attribution the mechanism
    implies, three conditions on top of finiteness: (1) the rows with a cell outside the bf16 block's bound stay within
    ``_row_budget`` (``1e-5 x rows x keys``, keys = T); (2) EVERY such row is a row a ``dqkvg8`` flip touched
    (``flip_ev["dw_qkvg_rows"]``); (3) for EVERY such row ``n`` the PRE-cast slab column ``dqkvg[:, n]`` is itself inside the bf16
    block's bound of its band against the seeded oracle (``flip_ev["band_col_worst"][n] <= 1``; the bands sit at 0.08-0.24 of it) --
    the flip is the CAST's rounding, not a band's miss.  (2) alone is weak at the matrix's flip counts (8016-96030 flips over the
    4608 Q / G / K columns of the GQA cells (6144 under MHA) touch most of them; the V band has 0: the seeded dV is a bitwise copy), and a defect upstream of the cast
    flips codes in exactly the columns it corrupts, so its rows are "touched" by construction -- (3) is the discriminator, (1)
    catches a diffuse miss, (2) a row no flip reaches.  Printed: the per-cell worst (``_report_close``, the magnitude the per-cell
    bound would judge), and per row outside its band, its flip count, its pre-cast column's worst, and its worst before / after the
    flips' rank-1 term ``sum_t (dqkvg8 - dqkvg8_ref)[t, n] / scale_dqkvg . h[t, :]`` is removed (the oracle casts at the block's
    scale, so what remains is GEMM rounding whatever caused the flips: a magnitude, not a discriminator).  Returns the per-cell
    worst."""
    got, ref64 = res.grads["dw_qkvg"], ref["dw_qkvg"]
    assert torch.isfinite(got).all(), f"{what}: non-finite cells"  # _rows_outside_mask is NaN-blind: NaN > bound is False
    worst = _report_close(got, ref64, what)
    outside = _rows_outside_mask(got, ref64)
    rows_out = torch.nonzero(outside).flatten()
    n_out, n_rows = int(rows_out.numel()), int(outside.numel())
    budget = _row_budget(n_rows, _row_keys(res)["dw_qkvg"])
    unexplained = rows_out[~torch.isin(rows_out, flip_ev["dw_qkvg_rows"].to(rows_out.device))]
    band_worst = flip_ev["band_col_worst"].to(rows_out.device)[rows_out]  # each row's PRE-cast slab column: worst cell / its band's bound
    not_the_casts = rows_out[band_worst > 1.0]
    if n_out:
        g, sp, t = res.geom, res.spec, res.batch * res.seq_len
        got64, r64 = got.detach().double(), ref64.detach().double()
        bound = _ATOL_FRAC[got.dtype] * r64.abs().max() + _RTOL[got.dtype] * r64.abs()
        h64 = res.saved.h.view(t, g.d_model).double() * sp.descale_h
        code_diff = (v["dqkvg8"].float()[:, rows_out].double() - flip_ev["dqkvg8_ref"].float()[:, rows_out].double()) / res.scalars["scale_dqkvg"]
        flip_term = code_diff.t() @ h64  # [rows outside, d_model]: the flips' rank-1 contributions to each row
        before = ((got64[rows_out] - r64[rows_out]).abs() / bound[rows_out]).amax(dim=1)
        after = ((got64[rows_out] - r64[rows_out] - flip_term).abs() / bound[rows_out]).amax(dim=1)
        n_flips = flip_ev["flips"][:, rows_out].sum(dim=0)
        for i, n in enumerate(rows_out.tolist()):
            print(
                f"{what}: row {n} ({_qkvg_band(g, n)}) outside the bf16 bound -- worst {before[i].item():.3f} of the bound, {int(n_flips[i])} dqkvg8 "
                f"flips in its column, {after[i].item():.3f} after removing their rank-1 term; its pre-cast slab column at "
                f"{band_worst[i].item():.3f} of its band's bound"
            )
    print(
        f"{what}: {n_out} of {n_rows} rows outside the bf16 bound (row budget 1e-5 x rows x keys = {budget:.3g}), {int(unexplained.numel())} of them untouched "
        f"by a dqkvg8 flip, {int(not_the_casts.numel())} with the pre-cast slab column itself outside its band's bound"
    )
    assert n_out <= budget, (
        f"{what}: {n_out} of {n_rows} rows outside the bf16 bound exceed the 1e-5 x rows x keys row budget {budget:.3g} -- rows {rows_out.tolist()}; "
        f"untouched by a dqkvg8 flip: {unexplained.tolist()}; pre-cast slab column outside its band's bound: {not_the_casts.tolist()}"
    )
    assert unexplained.numel() == 0, (
        f"{what}: rows {unexplained.tolist()} are outside the bf16 bound and no dqkvg8 code flip touched them (not the cast's flip class) -- "
        f"rows outside {rows_out.tolist()}, row budget {budget:.3g}"
    )
    assert not_the_casts.numel() == 0, (
        f"{what}: rows {not_the_casts.tolist()} are outside the bf16 bound and their PRE-cast slab columns are themselves outside the band's bf16 "
        f"bound against the seeded oracle ({[round(x, 3) for x in band_worst[band_worst > 1.0].tolist()]} of it): a miss of the band upstream of "
        f"the cast, not the cast's flip class -- rows outside {rows_out.tolist()}, row budget {budget:.3g}"
    )
    return worst


def _report_stage_difference(got: torch.Tensor, ref: torch.Tensor, what: str) -> None:
    """The SDPA stage's kernel-vs-reference difference CHARACTERISED (printed; the bf16 bound FORM it reports is then asserted by
    the caller through ``_assert_grad_close``).  The row recipe's absolute ``atol 0.08`` is at or above ``max|dQ| / max|dK|`` at
    this geometry in TRUE units -- at any ``scale_dp`` -- so the row pin alone cannot tell the row's flip class (SPARSE: a few
    d-rows, each one e4m3 step of a dS / P value times an operand row) from a diffuse miss (every row off by a few percent), nor
    an all-zero output from the right one; this print characterises the difference, and the bf16 bound form is the pin with
    teeth -- the (M) end-to-end sees whichever class it is, propagated.  Per output: the relative RMS, ``max|diff| / max|ref|``, the
    d-rows with a cell outside the bf16 block's bound FORM (``2^-7 max|ref| + 2^-6 |ref|``, the statistic the (M) layer is judged
    by) and how many d-rows carry 90 % of the squared difference."""
    g2 = got.detach().double().reshape(-1, got.shape[-1])
    r2 = ref.detach().double().reshape(-1, got.shape[-1])
    diff = g2 - r2
    ref_max = r2.abs().max().item()
    rel_rms = (diff.norm() / r2.norm().clamp_min(1e-300)).item()
    outside = (diff.abs() > _ATOL_FRAC[torch.bfloat16] * ref_max + _RTOL[torch.bfloat16] * r2.abs()).any(dim=1)
    row_sq = torch.sort((diff * diff).sum(dim=1), descending=True).values
    cum = torch.cumsum(row_sq, dim=0)
    n90 = int((cum < 0.9 * cum[-1]).sum().item()) + 1 if cum[-1].item() > 0 else 0
    print(
        f"{what} kernel vs the row's reference: rel RMS {rel_rms:.3g}, max|diff|/max|ref| {diff.abs().max().item() / max(ref_max, 1e-300):.3g}, "
        f"{int(outside.sum())} of {g2.shape[0]} d-rows outside the bf16 bound form, {n90} d-rows carry 90 % of the squared difference (the form is asserted next)"
    )


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
    ``amax_dP``, and ``ref_bwd(selection)`` -- the same reference re-run with ``return_intermediates=selection``, the evidence
    ``assert_close_fp8_grad`` consults to PROVE a bad d-row is one e4m3 midpoint flip of a P / dS value."""
    _test_python_root()
    from sdpa.fp8_ref import compute_ref_backward

    blk, g, sc, sp = res.blk, res.geom, res.scalars, res.spec
    b, s, d = blk.batch, blk.seq_len, g.d_head
    s_scale = 2.0 ** _api_const("FP8_SCALE_S_LOG2")
    bshd = lambda x, h: x.reshape(b, s, h, d)  # noqa: E731
    o_dead = v["og8"] if v["og8"] is not None else v["do8"]

    def ref_bwd(return_intermediates=True):
        return compute_ref_backward(
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
            return_intermediates=return_intermediates,
            quantize_ds=True,
            dP_scale=res.scale_dp,
            quantize_grads=False,
            delta=_delta(res)[..., :s],
        )

    dq, dk, dv, _dsink, _dp_amax_raw, _dqa, _dka, _dva, inter = ref_bwd(True)
    amax_ds = float(inter["ds_scaled"].abs().max()) / res.scale_dp
    return dq.to(torch.bfloat16).float(), dk.to(torch.bfloat16).float(), dv.to(torch.bfloat16).float(), amax_ds, ref_bwd


def _oracle(res, *, modelled: bool, seeded: Optional[dict] = None) -> dict:
    """The fp8 backward oracle fed the block's OWN conditions: its read-back gradient scales, ``2 ** FP8_SCALE_S_LOG2``, its
    ``scale_dp`` and the SAME ``delta`` the kernel consumed -- and, for the MODELLED and seeded oracles, the record's exact LSE,
    the record's bf16 pre-gate O, the record's bf16 GATE band, the record's e4m3 ``q8 / k8 / v8`` (the block's recompute, pinned
    bitwise the forward's bytes) and the block's bf16 dO (the gate backward's output the dO quantize read; the block's ``delta`` is
    its row-sum) -- the inputs the SDPA kernel consumed: the kernel recomputes P from that LSE over those codes, forms dP from the
    cast of that dO, and the gate backward forms dG / og8 from that O and that gate; ``seeded`` substitutes the block's bf16 dQ /
    dK / dV.  The unquantized-gradient oracle (U) keeps its own fp64 attention O / LSE / projection / operands / dO on purpose: it
    is the all-in informational reference."""
    sc = res.scalars
    g, t = res.geom, res.batch * res.seq_len
    _o_q, o_g, _o_k, _o_v = g.qkvg_offsets
    record = {}
    if modelled:
        v = _slots(res)
        record = dict(
            lse=res.saved.lse,
            o=res.saved.o,
            gate=_cols(res.saved.proj_slab.view(t, g.n_qkvg), o_g, g.h_q, g.d_head),
            q8=v["q8"],
            k8=v["k8"],
            v8=v["v8"],
            do=v["do"],
        )
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
    if not hasattr(res, "oracle_m"):  # memoised on the run: the report cell and the row-budget cell read the same oracle
        res.oracle_m = _oracle(res, modelled=True)
    return res.oracle_m


def _oracle_u(res) -> dict:
    return _oracle(res, modelled=False)


def _oracle_seeded(res) -> dict:
    v = _slots(res)
    b, s, g = res.batch, res.seq_len, res.geom
    seed = dict(dq=v["dq"].view(b, s, g.h_q, g.d_head), dk=v["dk"].view(b, s, g.h_kv, g.d_head), dv=v["dv"].view(b, s, g.h_kv, g.d_head))
    return _oracle(res, modelled=True, seeded=seed)


def _print_end_to_end(tag: str, grads: dict, ref: dict, *, keys: Optional[dict] = None) -> dict:
    """``cos`` and ``max|diff| / max|ref|`` per produced gradient against an oracle -- printed, returned, never asserted here --
    plus, for the bf16 outputs, the row-budgeted statistic the (M) assertion uses (the flip-class shape, budgeted like
    ``assert_close_fp8_grad``): the number of ROWS with a cell outside the bf16 block's bound against the budget ``1e-5 x rows x
    keys``, at least 1 (``keys`` = the reduction length feeding a row), returned as ``rows_outside / rows / row_budget``."""
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
            budget = _row_budget(g2.shape[0], keys[name])  # assert_close_fp8_grad: 1e-5 of rows x keys, at least 1
            out[name].update(rows_outside=int(outside.sum()), rows=int(g2.shape[0]), row_budget=budget)
            line += f" rows outside the bf16 bound: {int(outside.sum())} of {g2.shape[0]} (row budget 1e-5 x rows x keys = {budget:.3g})"
        print(line)
    return out


# ---------------------------------------------------------------------------
# ACCEPT (Rubin)
# ---------------------------------------------------------------------------


def _assert_quantizers_scalars_delta_bitwise(res) -> dict:
    """The BITWISE layer of a cell (module docstring): the quantizers (``dy8`` / ``do8`` / ``dqkvg8`` against torch's saturating
    cast at the READ-BACK scales; ``q8 / k8 / v8`` ``torch.equal`` the FORWARD's bytes; ``og8`` against the forward's ``o8`` and
    the kernels' bf16 ``O_gated``), every scalar of the block (``amax_* == max|.|`` of what the pass read, ``scale_* ==
    grad_scale_from_amax(amax_*)``, ``descale == 1 / scale``, the alphas ONE fp32 product each, ``descale_dp == 1 / scale_dp``)
    and ``delta`` (the chain's own ``dot_do_o`` with an exactly-zero pad tail).  Shared by the matrix cells and the
    launch-count-only cells; returns the materialised intermediates (``_slots``)."""
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
    return v


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
    ``saved.lse``, the block's ``delta`` and scalars under ``_FP8_GRAD_TOL`` + the flip budget -- a bad d-row above the magnitude
    cap PROVED one e4m3 midpoint flip from the reference's own intermediates (``operand`` / ``flip_unit`` / ``intermediates``),
    at the calibrated ``scale_dp`` the matrix runs at (module docstring) -- AND under the bf16 block's bound form on the stage's
    bf16 output (``_assert_grad_close``: the stricter pin the characterisation measured green on every cell; the row recipe's
    absolute ``atol 0.08`` alone cannot reject an all-zero dQ / dK at this geometry) -- ``amax_dP`` under
    ``_AMAX_DS_TOL``; ``dh / dW_o / dW_*_norm`` vs the oracle SEEDED with the block's own dQ / dK / dV under the bf16 block's
    bound, and ``dW_qkvg`` in the module's row-budgeted form with its attribution (``_assert_seeded_dw_qkvg_row_budgeted``: the rows
    with a cell outside that bound within ``1e-5 x rows x T``, every one a row a ``dqkvg8`` code flip touched AND a slab column whose
    pre-cast band is itself inside the bound, the cast's rounding and not a band's miss -- the per-cell bound sat at 1.02-1.65x on 7
    of 15 cells of the calibrated run, each a single near-amax flip moving one row; the per-cell worst stays printed)."""
    res = _cell_backward(cell)
    blk, g, sc, sp, saved = res.blk, res.geom, res.scalars, res.spec, res.saved
    b, s, d = res.batch, res.seq_len, g.d_head
    t = b * s
    v = _assert_quantizers_scalars_delta_bitwise(res)
    _o_q, o_g, _o_k, _o_v = g.qkvg_offsets
    gate = _cols(saved.proj_slab.view(t, g.n_qkvg), o_g, g.h_q, d)  # the GATE band of the slab, strided, as the gate kernels read it
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
    dq_ref, dk_ref, dv_ref, amax_ds_ref, ref_bwd = _row_reference(res, v)
    refs = dict(dq=dq_ref, dk=dk_ref, dv=dv_ref)
    # The flip proof's evidence (assert_close_fp8_grad): the dequantized BSHD operand a flipped intermediate multiplies (K for dQ,
    # Q for dK, dO for dV), that intermediate's descale (dS's for dQ / dK, P's for dV) and the reference re-run on the bad rows --
    # a bad d-row above the magnitude cap must be ONE e4m3 midpoint flip of a P / dS value, or the cell fails.
    s_scale = 2.0 ** _api_const("FP8_SCALE_S_LOG2")
    operands = dict(
        dq=v["k8"].view(b, s, g.h_kv, d).float() / sp.scale_k,
        dk=v["q8"].view(b, s, g.h_q, d).float() / sp.scale_q,
        dv=v["do8"].view(b, s, g.h_q, d).float() * sc["descale_do"],
    )
    flip_unit = dict(dq=1.0 / res.scale_dp, dk=1.0 / res.scale_dp, dv=1.0 / s_scale)
    for name, h, tag in (("dq", g.h_q, "dQ"), ("dk", g.h_kv, "dK"), ("dv", g.h_kv, "dV")):
        # keys = the reduction length feeding each d-row (s_kv for dQ, s_q for dK / dV): self-attention, both are S
        assert_close_fp8_grad(
            v[name].view(b, s, h, d).float(),
            refs[name],
            grad_tol["atol"],
            grad_tol["rtol"],
            tag,
            keys=s,
            operand=operands[name],
            flip_unit=flip_unit[name],
            intermediates=lambda sel: ref_bwd(sel)[8],
            fp8_dtype=_E4M3,
            out_dtype=torch.bfloat16,
        )
        _report_stage_difference(v[name].view(b, s, h, d), refs[name], tag)
        # The stricter pin the characterisation measured green on every cell of both regimes (0 d-rows outside): the bf16
        # block's bound FORM (atol 2^-7 max|ref| + rtol 2^-6 |ref|, cos >= 0.999) on the stage's bf16 output -- an all-zero dQ
        # / dK passes the row recipe's absolute atol 0.08 at this geometry (max|dQ| <= 0.08 in true units at any scale_dp);
        # it does not pass this.
        _assert_grad_close(v[name].view(b, s, h, d), refs[name].double(), f"{tag} vs the row's reference (the bf16 bound form)")
    amax_dp = sc["amax_dp"]
    print(f"amax_dP {amax_dp:.6g} vs the reference's max|dS| {amax_ds_ref:.6g} (scale_dp {res.scale_dp:g}, amax_dP * scale_dp = {amax_dp * res.scale_dp:.4g})")
    assert abs(amax_dp - amax_ds_ref) <= amax_tol["atol"] + amax_tol["rtol"] * amax_ds_ref, (amax_dp, amax_ds_ref)
    # --- downstream of B4: the SEEDED oracle under the bf16 block's bound ------------------------------------------------
    ref = _oracle_seeded(res)
    flip_ev = _report_seeded_intermediates(res, v, ref)
    worst = {}
    for name in ("dh", "dw_o"):
        if res.grads[name] is not None:
            worst[name] = _assert_grad_close(res.grads[name], ref[name], f"{name} vs the seeded oracle")
    if res.grads["dw_qkvg"] is not None:
        # dW_qkvg = dqkvg8^T . h8 is where the slab's single near-amax e4m3 flips land (1.02-1.65x the per-cell bound on 7 of 15
        # cells), one dW_qkvg row per flip: judged in the row-budgeted form the (M) layer uses, every row outside a flip-touched row
        # whose pre-cast slab column is itself inside its band's bound (the cast's rounding, not a band's miss).
        worst["dw_qkvg"] = _assert_seeded_dw_qkvg_row_budgeted(res, v, ref, flip_ev, "dw_qkvg vs the seeded oracle")
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
    keys = _row_keys(res)
    m = _print_end_to_end(f"{cell.id} (M)", res.grads, _oracle_m(res), keys=keys)
    u = _print_end_to_end(f"{cell.id} (U)", res.grads, _oracle_u(res), keys=keys)
    assert m and u


def _report_oracle_do_disagreement(cell_id: str, res, ref: dict) -> None:
    """Localisation print: how far the oracle's OWN dO (its once-rounded fp64 gradient w.r.t. the pre-gate O) is from the block's
    twice-rounded bf16 dO, in bf16 elements and in ``do8`` codes its cast WOULD flip -- the disagreement the modelled SDPA stage no
    longer sees because it is fed the block's dO (and the block's ``delta`` is that dO's row-sum)."""
    v = _slots(res)
    do16 = ref["do"].reshape(v["do"].shape).to(torch.bfloat16)
    n16 = int((do16 != v["do"]).sum())
    n8 = int((quant_e4m3(do16, res.scalars["scale_do"]).view(torch.uint8) != v["do8"].view(torch.uint8)).sum())
    print(
        f"{cell_id}: the oracle's own bf16 dO differs from the block's on {n16} of {do16.numel()} elements; its e4m3 cast would differ from do8 on "
        f"{n8} codes (the modelled SDPA stage is fed the block's dO, so none reach it)"
    )


def _row_keys(res) -> dict:
    """The reduction length feeding each ROW of a bf16 output: ``dh`` (a token row) over N, the weight gradients (an output row)
    over the tokens -- the ``keys`` of the ``1e-5 x rows x keys`` row budget."""
    return dict(dh=res.geom.n_qkvg, dw_qkvg=res.batch * res.seq_len, dw_o=res.batch * res.seq_len)


def _m_over_the_row_budget(cell: _Cell, res, ref: dict, names: tuple) -> dict:
    """The (M) end-to-end PRINTED for every output (``_print_end_to_end``, the oracle's dO disagreement first) and, for ``names``, the
    outputs whose rows outside the bf16 bound exceed the ``1e-5 x rows x keys`` row budget, as ``{name: (rows outside, rows,
    budget)}`` -- empty when the budget holds on every one of them."""
    _report_oracle_do_disagreement(cell.id, res, ref)
    m = _print_end_to_end(f"{cell.id} (M)", res.grads, ref, keys=_row_keys(res))
    assert m, cell.id
    return {n: (m[n]["rows_outside"], m[n]["rows"], m[n]["row_budget"]) for n in names if n in m and m[n]["rows_outside"] > m[n]["row_budget"]}


@requires_rubin
@_MATRIX
def test_fp8_end_to_end_modelled_is_row_budgeted(cell):
    """The (M) end-to-end asserted in the ONE form named for it: the bf16 block's bound with the SDPA stage's flip class propagated
    linearly and budgeted by ROWS like ``assert_close_fp8_grad`` (``1e-5 x rows x keys``, at least 1; ``keys`` = the reduction
    length feeding a row), on every output whose chain has NO fold -- ``dw_o`` on every cell (``dy8^T . og8``: nothing of the
    SDPA backward in it) and ``dh / dw_qkvg`` on the MHA cells (``group == 1``: the row writes dK / dV once) -- never widened.
    The modelled oracle's SDPA stage is fed the kernel's own inputs (the record's LSE, ``q8 / k8 / v8`` and the block's bf16 dO; its
    ``delta`` is that dO's row-sum), so what this layer measures is the SDPA stage's kernel-vs-reference difference propagated through
    the modelled casts -- inside the budget on every output asserted here (``dw_o`` 0 rows outside on every cell; the MHA cells' ``dh``
    0 and ``dw_qkvg`` 0 / 18 of 8192 against 83.9 / 41.9).  Fed its own cast of its fp64 chain instead, the oracle's ``q8 / k8 / v8``
    and ``do8`` flipped a few per cent of their codes against the record's LSE / delta and this layer read 75 % of the rows outside --
    a composition gap, removed, not a margin.  ``dh / dw_qkvg`` of the GQA cells -- the chain WITH the fold, over fp32 per-Q-head
    partials rounded once -- are ``test_fp8_end_to_end_modelled_gqa_fold_is_row_budgeted``."""
    res = _cell_backward(cell)
    fold_free = ("dh", "dw_qkvg", "dw_o") if cell.group == 1 else ("dw_o",)
    over = _m_over_the_row_budget(cell, res, _oracle_m(res), fold_free)
    assert (
        not over
    ), f"{cell.id}: (M) rows outside the bf16 bound exceed the 1e-5 x rows x keys row budget on an output with no fold in its chain (rows outside, rows, budget): {over}"


@requires_rubin
@_GQA_FOLD_MATRIX
def test_fp8_end_to_end_modelled_gqa_fold_is_row_budgeted(cell):
    """The (M) row budget (the form of ``test_fp8_end_to_end_modelled_is_row_budgeted``) on ``dh / dw_qkvg`` of the 13 GQA cells -- the
    outputs whose chain carries the fp8 SDPA row's GQA dK / dV fold.  The fold sums fp32 per-Q-head partials and rounds ONCE, like the
    reference (the kernel's dK is bitwise the once-rounded reference's under GQA, dV within 1.4e-4 relative RMS of it), and every cell
    is inside its budget (module docstring, second table: ``dw_qkvg`` 0-9 of 5120 rows outside against budgets of 13-102, ``dh`` 0
    rows) -- so the assertion is plain on every cell, never widened.  With bf16 partials summed (``group`` roundings) the same chain put
    ``dw_qkvg`` 14-81 of 5120 rows outside against budgets of 13-52 and ``dh`` 17 / 66 / 142 token rows on the three dense GQA cells
    (0-3 elsewhere; at ``scale_dp = 1.0`` the flushed dS put 76 / 340 outside on the two rope_only ones) -- OVER on 10 of the 13,
    which carried a strict ``xfail`` until the fold rounded once.  Note on the form: the token is the reduction axis of
    ``dW_qkvg = dqkvg8^T . h8``, so a perturbed token row moves EVERY row of a band at once; a per-row budget describes ``dh`` (one
    token, one row), not a weight gradient -- which is why a diffuse fold difference lands here while the seeded layer's single flips
    stay inside it.  The margins are the 204-SM dataset's (module docstring: the inputs are device-Philox draws, laid out by the SM
    count): on a part with another SM count a cell can move -- re-measure there before reading a failure as the kernel's."""
    res = _cell_backward(cell)
    over = _m_over_the_row_budget(cell, res, _oracle_m(res), ("dh", "dw_qkvg"))
    assert (
        not over
    ), f"{cell.id}: (M) dh / dw_qkvg rows outside the bf16 bound exceed the 1e-5 x rows x keys row budget under GQA (rows outside, rows, budget): {over}"


@requires_rubin
@_KNOB_SETS
def test_fp8_two_runs_are_bitwise(knobs):
    """Two executes of the SAME block over the same record / dy / scale_dp -- the second into a workspace poisoned 0xFF and
    NaN-filled gradients -- are ``torch.equal`` on every gradient AND on the scalar block, and so is a FRESH block over the same
    record (compiled anew, poisoned the same way), under every knob set (``fuse_gate_bwd`` is inert under quant;
    ``fuse_wgrad_overlap`` moves B1 / B7 to the side stream): the block's only atomics are int32 ``atomicMax`` of non-negative
    fp32 bit patterns -- order-free -- and nothing an execute reads survives from the previous one."""
    res = _cell_backward(_BITWISE_CELL, **knobs)
    sb1 = _scalar_block(res).clone()
    # (1) the memoised block again, into a poisoned workspace and NaN-filled gradients
    ws2 = torch.empty_like(res.ws).fill_(0xFF)
    grads2 = _alloc_grads(res.blk, fill=float("nan"))
    _execute_fp8(res.blk, res.inp, res.saved, res.dy, grads2, ws2, scale_dp=res.scale_dp_t, **res.scale_ts)
    torch.cuda.synchronize()
    for name, ten in grads2.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), f"{name}: a second execute of the same block differs (knobs={knobs})"
    assert torch.equal(_view(ws2, res.blk._layout().quant_scalars, sb1.shape, torch.float32), sb1), "the scalar block differs between two executes of one block"
    # (2) a fresh block over the same record
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
    """CUPTI kernel records of one execute == the launch table (module docstring): 15 at the bitwise cell (norm, GQA 8/2, c = 1,
    one dQ launch per chunk), 15 rope_only (the epilogue launch stays for the quantize), 15 MHA (the row's dK fold shares the dV
    fold's launch under GQA, so the group no longer moves the count), 20 at the three padded GQA cells with weight gradients, 20 at
    the padded MHA cell (launch count only), 18 at the padded dgrad-only cell, 17 at the kv-side-only padded cell; the same under
    both recipes and every knob; no hidden memcpy and NO memset (the prologue's scalar-block init and the fp8 row's fills are
    kernels, counted above -- a memset appearing here would be a new, uncounted write on the execute path).
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
    assert not memsets, f"a hidden memset on the execute path (the scalar init and the row's fills are kernels): {memsets}"
    assert formula == expected, (formula, expected)
    assert len(kernels) == expected, (len(kernels), expected, kernels)


@requires_rubin
@pytest.mark.parametrize("cell", _LAUNCH_ONLY_CELLS, ids=[c.id for c in _LAUNCH_ONLY_CELLS])
def test_fp8_launch_only_cells_are_finite_and_quantize_bitwise(cell):
    """The launch-count-only cells (padded x MHA; the kv-side pads alone) run the whole backward for their count -- this is
    their numerics layer, without an oracle: every gradient finite, and the BITWISE layer of the matrix cells
    (``_assert_quantizers_scalars_delta_bitwise``: the quantizers at the read-back scales, ``q8 / k8 / v8`` against the
    forward's bytes, ``og8``, every scalar, ``delta`` with its zero pad tail).  The GEMM / SDPA / seeded bounds stay with the
    matrix cells; padded x MHA shares its const_expr arms with ``s992_causal_b1`` (the pads) and ``s512_causal_b1_mha`` (the
    MHA dK arm), so this layer is what pins their COMBINATION at block level."""
    res = _cell_backward(cell)
    for name, ten in res.grads.items():
        if ten is not None:
            assert torch.isfinite(ten).all(), f"{cell.id}: {name} has non-finite cells"
    _assert_quantizers_scalars_delta_bitwise(res)


@requires_rubin
def test_fp8_cuda_graph_capture_replays_bitwise():
    """One ``execute`` captured into a CUDA graph on a side torch stream replays bitwise the eager run and recomputes over a
    NEW ``dy`` and a NEW ``scale_dp`` written in place through the captured pointers (the prologue's scalar init, the amax
    partials / atomics and the quantize publishes are all device work on the launch stream: capturable, no host readback).  The
    capture itself launches nothing; the block allocates nothing."""
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
    its place) + the GEMM scratch (EXACTLY the max over the four K64 fp8 plans); a buffer 4096 B larger keeps its tail untouched;
    two executes allocate nothing; every e4m3 region is WRITTEN in full (no 0xFF byte survives: 0xFF is the e4m3 NaN, which a
    saturating cast of finite data never produces), and so are the fp32 ``delta`` region and the scalar block's slots (no NaN)."""
    from cudnn.gated_attention_block.api import _WS_ALIGN

    res = _cell_backward(_BITWISE_CELL, **knobs)
    blk = res.blk
    size = blk.get_workspace_size()
    lay = blk._layout()
    assert size == lay.total_bytes and size % _WS_ALIGN == 0
    plans = blk.gemm_plans
    assert len(plans) == 4 and all(p.mma_tile_k_bytes == 64 and p.has_alpha for p in plans.values())
    assert lay.gemm_scratch_bytes == max(p.workspace_bytes for p in plans.values()) >= 1
    assert lay.sdpa_bwd_bytes == blk._sdpa.scratch_workspace_bytes()
    assert lay.quant_scalars >= 0 and lay.quant_scalars % 256 == 0 and lay.delta >= 0 and lay.o_gated == -1 and lay.recompute_v == -1
    # the bf16 rebuild buffers are NOT carved: the prologue's rebuild writes q8 / k8 straight out of its registers
    assert lay.recompute == -1 and lay.recompute_k == -1
    assert (
        lay.amax_partials >= 0 and lay.amax_partials % _WS_ALIGN == 0 and lay.amax_partials_n == blk._prologue.n_partials_cap >= blk._prologue.n_partials() >= 1
    )
    # the producers' partials: the gate backward's two arrays at its persistent cap (what it writes is its grid, at most the cap),
    # the norm backward's at EXACTLY its grid; every region 256-B aligned and carved after the dY partials
    gb, nb = blk._gate_bwd, blk._norm_bwd
    assert lay.gate_partials_n == gb.n_partials_cap >= gb.n_partials() >= 1 and lay.band_partials_n == nb.n_amax_partials() == sum(nb.n_ctas()) >= 3
    assert lay.amax_partials < lay.amax_partials_do < lay.amax_partials_dg < lay.amax_partials_bands
    for name in ("amax_partials_do", "amax_partials_dg", "amax_partials_bands"):
        assert getattr(lay, name) % _WS_ALIGN == 0, name
    for name in ("dy8", "do8", "q8", "k8", "v8", "dqkvg8"):
        assert getattr(lay, name) >= 0 and getattr(lay, name) % _WS_ALIGN == 0, name
    ws = torch.full((size + 4096,), 0xFF, dtype=torch.uint8, device="cuda")
    grads = _alloc_grads(blk)
    _execute_fp8(blk, res.inp, res.saved, res.dy, grads, ws[:size], scale_dp=res.scale_dp_t)
    torch.cuda.synchronize()
    # The allocation pin in the caching allocator's COUNTER form (test_block_training_forward.py): the cumulative allocation
    # count cannot be lowered by an unrelated release and still rises for a temporary the execute frees before returning; the
    # allocator peak is the second witness for such a temporary's bytes.  Every object the execute reads stays alive across it.
    gc.collect()
    live = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    n0 = torch.cuda.memory_stats()["allocation.all.allocated"]
    _execute_fp8(blk, res.inp, res.saved, res.dy, grads, ws[:size], scale_dp=res.scale_dp_t)
    _execute_fp8(blk, res.inp, res.saved, res.dy, grads, ws[:size], scale_dp=res.scale_dp_t)
    torch.cuda.synchronize()
    n1 = torch.cuda.memory_stats()["allocation.all.allocated"]
    peak = torch.cuda.max_memory_allocated()
    assert n1 == n0, f"the fp8 backward made {n1 - n0} CUDA allocation(s) on the execute path (allocation.all.allocated {n0} -> {n1})"
    assert peak <= live, f"a temporary on the fp8 backward's execute path: the allocator peak rose from {live} to {peak} bytes"
    assert torch.equal(ws[size:], torch.full((4096,), 0xFF, dtype=torch.uint8, device="cuda")), "bytes past get_workspace_size() were written"
    for name, ten in grads.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), name
    sb = _view(ws[:size], lay.quant_scalars, (len(_api_const("QUANT_SCALAR_SLOTS")),), torch.float32)
    assert torch.isfinite(sb).all(), "a scalar slot was never written (0xFF = NaN)"
    g, t, d = blk.geom, blk.batch * blk.seq_len, blk.geom.d_head
    e4m3_regions = dict(dy8=(lay.dy8, t * g.d_model), do8=(lay.do8, t * g.h_q * d), q8=(lay.q8, t * g.h_q * d), k8=(lay.k8, t * g.h_kv * d))
    e4m3_regions.update(v8=(lay.v8, t * g.h_kv * d), dqkvg8=(lay.dqkvg8, t * g.n_qkvg))
    if lay.og8 >= 0:
        e4m3_regions["og8"] = (lay.og8, t * g.h_q * d)
    for name, (off, nbytes) in e4m3_regions.items():
        survivors = int((ws[off : off + nbytes] == 0xFF).sum())
        assert survivors == 0, f"{name}: {survivors} of {nbytes} e4m3 bytes still hold the 0xFF poison -- never written"
    delta = _view(ws[:size], lay.delta, tuple(blk._sdpa.delta_shape), torch.float32)
    assert torch.isfinite(delta).all(), "a delta element (pad rows included) was never written (0xFFFFFFFF = NaN)"


@requires_rubin
@_MATRIX
def test_fp8_amax_times_scale_never_exceeds_448(cell):
    """Every published (amax, scale) pair of the scalar block satisfies ``amax * scale <= 448`` -- the in-kernel formula's guarantee
    for the amax the kernel READ (dY, dO over the STORED bf16 dO, dqkvg) -- and ``amax_dP * scale_dp <= 448`` at every calibrated
    cell (the matrix default; the chart recipe's assertion); at the ``scale_dp = 1.0`` cell the product is printed (the under-scaled
    regime: nothing saturates there, the cast flushes most of dS to zero instead)."""
    res = _cell_backward(cell)
    sc = res.scalars
    for n in ("dy", "do", "dqkvg"):
        prod = sc[f"amax_{n}"] * sc[f"scale_{n}"]
        print(f"{cell.id}: amax_{n} * scale_{n} = {prod:.4g}")
        assert prod <= FP8_E4M3_MAX, (n, sc[f"amax_{n}"], sc[f"scale_{n}"])
        assert sc[f"amax_{n}"] * (2.0 * sc[f"scale_{n}"]) > FP8_E4M3_MAX or sc[f"amax_{n}"] == 0.0, f"{n}: the scale is not the largest power of two (margin 0)"
    print(f"{cell.id}: amax_dP * scale_dp = {sc['amax_dp'] * res.scale_dp:.4g} (scale_dp {res.scale_dp:g})")
    if _resolved_scale_dp(cell) == "calibrated":
        assert sc["amax_dp"] * res.scale_dp <= FP8_E4M3_MAX


@requires_rubin
@pytest.mark.parametrize("how", ["ambient", "explicit"])
def test_fp8_a_caller_stream_orders_every_stage(how):
    """Every stage -- the fused prologue and epilogue, the quantize kernels, the four fp8 GEMMs, the fp8 SDPA adapter -- launches
    on ONE stream, the caller's: ambient (``with torch.cuda.stream(s):``) or explicit (``current_stream=``).  The default stream is parked
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
def test_fp8_workspace_view_cache_keys_on_the_pointer_and_is_bitwise():
    """The block keeps ONE entry of typed workspace views (``_workspace_views``): a second execute over the SAME buffer reuses it
    (the same namespace object), a different buffer rebuilds it (the key is ``(data_ptr, numel, device)``, checked on every call --
    never assumed), the gradients and the scalar block are ``torch.equal`` across hits and misses, ``release_workspace_views()``
    drops the entry (a caller freeing its workspace while the block lives pins nothing), and the convenience wrapper releases
    after every call (its workspace dies at return)."""
    res = _cell_backward(_BITWISE_CELL)
    blk = res.blk
    # the memoised block may have run other tests' workspaces since the cell was built: bind res.ws first (a hit or a miss), then
    # the entry is res.ws's and the next execute over it is a HIT
    grads = _alloc_grads(blk, fill=float("nan"))
    _execute_fp8(blk, res.inp, res.saved, res.dy, grads, res.ws, scale_dp=res.scale_dp_t)
    torch.cuda.synchronize()
    assert blk._ws_views is not None and blk._ws_views_key == (res.ws.data_ptr(), res.ws.numel(), res.ws.device)
    views = blk._ws_views
    for name, ten in grads.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), f"{name}: the rebuilt views changed the gradients"
    grads = _alloc_grads(blk, fill=float("nan"))
    _execute_fp8(blk, res.inp, res.saved, res.dy, grads, res.ws, scale_dp=res.scale_dp_t)  # the same buffer: a HIT
    torch.cuda.synchronize()
    assert blk._ws_views is views, "a second execute over the same workspace rebuilt the views"
    for name, ten in grads.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), f"{name}: a cache hit changed the gradients"
    ws2 = torch.empty_like(res.ws).fill_(0xFF)
    grads2 = _alloc_grads(blk, fill=float("nan"))
    _execute_fp8(blk, res.inp, res.saved, res.dy, grads2, ws2, scale_dp=res.scale_dp_t)  # another buffer: a MISS, rebuilt
    torch.cuda.synchronize()
    assert blk._ws_views is not views and blk._ws_views_key == (ws2.data_ptr(), ws2.numel(), ws2.device)
    for name, ten in grads2.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), f"{name}: a cache miss changed the gradients"
    assert torch.equal(_view(ws2, blk._layout().quant_scalars, (len(_api_const("QUANT_SCALAR_SLOTS")),), torch.float32), _scalar_block(res))
    blk.release_workspace_views()
    assert blk._ws_views is None and blk._ws_views_key is None
    _execute_fp8(blk, res.inp, res.saved, res.dy, grads2, ws2, scale_dp=res.scale_dp_t)  # rebuilt after the release
    torch.cuda.synchronize()
    assert blk._ws_views is not None and blk._ws_views_key[0] == ws2.data_ptr()
    for name, ten in grads2.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), name
    # the convenience wrapper allocates its workspace per call and releases the views at return
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
        cached = [b for b in _api_bwd._BWD_CACHE.values() if b.quant is not None and b.batch == res.batch and b.seq_len == res.seq_len]
        assert cached and all(b._ws_views is None for b in cached), "the wrapper left the per-call workspace's views cached"
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
    """``thd=True`` with ``quant``: dense-only for now (the block's packed quantized arm -- the fp8 row's THD chain, which serves an
    external delta, reading the gate backward's packed bf16 delta -- is a follow-up) -- declined typed AT DECLARATION, naming BOTH
    attributes, and the message does NOT tell the caller to build the plan without ``external_delta`` (the adapter's text must never
    surface here).  Host-side, over placeholders shaped like a packed record (``[T, d_model]`` bf16 dy, e4m3 ``saved.h``, int32
    ``saved.seq_lens``)."""
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
    """An ``MxQuantSpec`` selects the MXFP8 backward (its own arm and its own suite, ``test_block_backward_mxfp8.py``), never the
    per-tensor fp8 one: the declaration constructs the MXFP8 stage list, and its fp4 weight modes construct that list's fp4 arms
    (``test_block_backward_fp4.py``) -- an e4m3 ``w_o`` handed to such a block is ``check_support``'s decline, by the weight's name."""
    r = _fp8_decl(dict(_COMMON), 1, 256, quant=None)
    mx = MxQuantSpec(descale_w_o=r.spec.descale_w_o, scale_o=r.spec.scale_o)
    blk = _declare_fp8_bwd(r.dy, r.saved, r.inp, r.geom, quant=mx)
    assert isinstance(blk.quant, MxQuantSpec) and type(blk._sdpa).__name__ == "_SdpaBwdMxfp8"
    assert (
        type(blk._prologue).__name__ == "_MxQuantPrologue" and type(blk._epilogue).__name__ == "_MxQuantEpilogue"
    )  # the MXFP8 fused launches, not the fp8 ones
    blk4 = _declare_fp8_bwd(r.dy, r.saved, r.inp, r.geom, quant=MxQuantSpec(descale_w_o=1.0, scale_o=1.0, o_fp4=Fp4Format.NVFP4))
    assert blk4.o_fp4 is Fp4Format.NVFP4 and blk4._out_proj_dgrad.block_scale and blk4._gate_bwd.want_dy_descale
    assert type(blk4._prologue).__name__ == "_MxQuantPrologue"
    with pytest.raises(ValueError, match="w_o"):
        blk4.check_support()


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
def test_fp8_reject_a_geometry_only_the_ldg_rebuild_can_tile():
    """The fused prologue rebuilds Q / K with the TMA norm + RoPE kernel, whose tile is ``tile_rows`` consecutive heads of one token:
    it must divide ``h_q``, be a multiple of ``h_kv`` and of the CTA's 4 warps (``_QkNormRope.resolve_tile_rows``, fitted in 16..1).
    Nothing fits ``h_q = 20`` MHA or ``h_q = 6`` over ``h_kv = 2``: the prologue stage declines typed under ``quant`` -- on ANY CUDA
    device, from the geometry alone, naming the head counts and the LDG kernel -- and on Rubin the whole block's ``check_support``
    surfaces that message (the prologue is its first stage; every block-level check passes).  The bf16 backward over the same
    geometry resolves its rebuild to the LDG kernel and passes every block-level check (the TMA validator types ``tile_rows=0``
    instead of dividing by it).  Every quant geometry the matrix runs (8/2, 8/8) tiles; this pin is where the quantized backward
    stops."""
    for h_q, h_kv in ((20, 20), (6, 2)):
        geom_kw = {**_COMMON, "h_q": h_q, "h_kv": h_kv}
        blk = _fp8_decl(geom_kw, 1, 256).blk
        assert blk._stages[0] is blk._prologue and blk._prologue._rebuild.resolve_tile_rows() == 0
        with pytest.raises(NotImplementedError, match="fp8_bwd_prologue.*LDG kernel") as ei:
            blk._prologue.check_support()  # the stage's own contract: device-free, so it runs on this host too
        assert f"h_q={h_q}" in str(ei.value) and f"h_kv={h_kv}" in str(ei.value), str(ei.value)
        if _cc() == _SM107:
            with pytest.raises(NotImplementedError, match="fp8_bwd_prologue.*LDG kernel"):
                blk.check_support()
        bf16 = _declare_bwd(geom_kw, 1, 256).blk  # a bf16 backward over a bf16 record (the quantized record itself is a decline without quant)
        assert bf16._recompute_qk.resolve_impl() == "ldg" and bf16._prologue is None
        try:
            bf16.check_support()
        except NotImplementedError as e:  # off Rubin the device gate is the only decline left, and it names Rubin
            assert _cc() != _SM107 and "Rubin" in str(e), str(e)


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
    exactly the two wgrads, the matrix runs at the calibrated ``scale_dp`` and exactly one row -- the bitwise cell's geometry --
    runs at ``scale_dp = 1.0``; the (M) GQA layer's parametrization is the 13 GQA matrix cells (the fold is a GQA mechanism: no MHA
    cell), every one a PLAIN assertion -- the fold rounds once, so no cell carries an ``xfail``."""
    for c in _CELLS + _LAUNCH_ONLY_CELLS:
        assert c.causal or c.s % 128 == 0, f"{c.id}: a dense S % 128 != 0 has no record"
    for c in _LAUNCH_ONLY_CELLS:  # launch-count arms the matrix does not reach, and nothing it already runs
        assert c.causal and c.id not in {m.id for m in _CELLS}, c.id
    pad_mha, kv_only = _BY_ID["s992_causal_b1_mha-norm"], _BY_ID["s384_causal_b1-norm"]
    assert pad_mha.s % 128 != 0 and pad_mha.h_kv == _COMMON["h_q"], "padded x MHA: both pad terms without the dK fold"
    assert kv_only.s % 128 == 0 and kv_only.s % 256 != 0 and kv_only.h_kv < _COMMON["h_q"], "the kv-side pads alone (+2, no +3)"
    assert _padded(pad_mha) == (True, True) and _padded(kv_only) == (False, True)
    served, only = _BY_ID["s1000_causal_b1-norm"], _BY_ID["s1000_causal_b1_dgrad_only-norm"]
    assert (served.b * served.s) % 16 == 8 and served.bwd_kw == {} and served.need_dw_o and served.need_dw_qkvg
    assert only.bwd_kw == dict(need_dw_o=False, need_dw_qkvg=False)
    assert _SCALE_DP_DEFAULT == "calibrated" and _resolved_scale_dp(_BITWISE_CELL) == "calibrated"
    unit, base = _BY_ID["s512_causal_b2_scale_dp_1-norm"], _BITWISE_CELL
    assert unit.scale_dp == 1.0 and (unit.s, unit.causal, unit.b, unit.h_kv, unit.qk_norm) == (base.s, base.causal, base.b, base.h_kv, base.qk_norm)
    assert [c.id for c in _CELLS if _resolved_scale_dp(c) != "calibrated"] == [unit.id]
    assert [c.id for c in _GQA_CELLS] == [c.id for c in _CELLS if c.group > 1] and len(_GQA_CELLS) == 13
    assert only.id in {c.id for c in _GQA_CELLS} and all(c.group > 1 for c in _GQA_CELLS), "the fold is a GQA mechanism: no MHA cell in its layer"
    assert all(isinstance(p, _Cell) for p in _GQA_FOLD_MATRIX.args[1]), "the fold rounds once: no (M) GQA cell carries an xfail (a plain _Cell each)"


# ---------------------------------------------------------------------------
# The plan-time constants are SLOTS the prologue's init job stores (nothing is filled on the device at compile)
# ---------------------------------------------------------------------------

# The slot tuple's ABI: the fifteen pre-existing slots keep their indices, the QuantSpec's constants are the appended tail.
_PRE_EXISTING_SLOTS = (
    "amax_dy", "amax_do", "amax_dqkvg", "amax_dp",
    "scale_dy", "descale_dy", "scale_do", "descale_do", "scale_dqkvg", "descale_dqkvg",
    "alpha_b1", "alpha_b2", "alpha_b7", "alpha_b8", "descale_dp",
)  # fmt: skip
_CONST_SLOTS = (
    "scale_q", "scale_k", "scale_v", "scale_o",
    "descale_q", "descale_k", "descale_v", "descale_o",
    "descale_w_o", "descale_h", "descale_w_qkvg",
    "scale_s", "descale_s", "scale_dqkv",
)  # fmt: skip


@requires_cuda
def test_fp8_plan_time_constants_are_init_launch_arguments_not_compile_time_fills():
    """Host, any CUDA device, no compile and no launch: the ``QuantSpec``'s plan-time constants are SLOTS of the scalar block --
    the tail ``QUANT_CONST_SLOTS`` appended after the fifteen pre-existing slots, whose names and indices are unchanged (the
    stride stays 4 B, the block stays inside its 256 B) -- and the scalar-init launch stores them from its KERNEL ARGUMENTS on
    every execute: a declared block holds the constants' VALUES as Python floats only (``_quant_const_values()``, in slot order,
    the ``QuantSpec``'s static scales, their reciprocals, the alpha factors, ``scale_s`` / ``descale_s`` and ``scale_dqkv = 1.0``), its
    fused prologue -- whose first job is the scalar init -- is declared over the whole block with the constants' slot range and hands
    its own rebuild / v8 jobs the static scales as kernel ARGUMENTS (never a slot of the block that launch is writing), the kernel
    ABI reserves an argument per constant (``run_init_scalars(consts=)`` / ``compile_init_scalars(const_slot0, n_consts)`` on the
    standalone init kernel, ``run_fp8_bwd_prologue(consts=)`` / ``compile_fp8_bwd_prologue(const_slot0, n_consts)`` on the fused
    launch -- appended and defaulted), and neither ``compile()`` nor the values method spells a device allocation -- so there is no
    compile-time fill an execute on another stream could race (the GPU half of this pin is the first-use test below)."""
    slots, consts = _api_const("QUANT_SCALAR_SLOTS"), _api_const("QUANT_CONST_SLOTS")
    assert slots[: len(_PRE_EXISTING_SLOTS)] == _PRE_EXISTING_SLOTS, "the pre-existing slots and their indices are an ABI (append-only)"
    assert slots[len(_PRE_EXISTING_SLOTS) :] == consts == _CONST_SLOTS, "the constants are the appended tail, in kernel-argument order"
    assert len(set(slots)) == len(slots) == 29 and len(consts) == 14 <= _quantize.MAX_INIT_CONSTS
    assert len(slots) * _api_const("QUANT_SCALAR_STRIDE") <= _api_const("QUANT_SCALARS_BYTES") and _api_const("QUANT_SCALAR_STRIDE") == 4
    r = _fp8_decl(dict(_COMMON), 1, 256)
    blk, spec = r.blk, r.spec
    assert blk._quant_vals is None and not hasattr(blk, "_quant_dev") and not hasattr(blk, "_quant_consts"), "the compile-time device constants are gone"
    vals = blk._quant_const_values()
    assert tuple(vals) == consts, (tuple(vals), consts)
    assert all(type(v) is float and np.isfinite(v) for v in vals.values()), vals
    s_log2 = _api_const("FP8_SCALE_S_LOG2")
    want = dict(
        scale_q=spec.scale_q,
        scale_k=spec.scale_k,
        scale_v=spec.scale_v,
        scale_o=spec.scale_o,
        descale_q=1.0 / spec.scale_q,
        descale_k=1.0 / spec.scale_k,
        descale_v=1.0 / spec.scale_v,
        descale_o=1.0 / spec.scale_o,
        descale_w_o=spec.descale_w_o,
        descale_h=spec.descale_h,
        descale_w_qkvg=spec.descale_w_qkvg,
        scale_s=2.0**s_log2,
        descale_s=2.0**-s_log2,
        scale_dqkv=1.0,
    )
    assert vals == {k: float(v) for k, v in want.items()}, (vals, want)
    init = blk._prologue  # the fused prologue's first job is the scalar init: it carries the init body's slot facts
    assert blk._stages[0] is init and not hasattr(blk, "_init_scalars"), "the standalone init launch is the prologue's job now"
    assert (init.n_slots, init.const_slot0, init.n_consts) == (len(slots), slots.index(consts[0]), len(consts)) == (29, 15, 14)
    run_sig = inspect.signature(_quantize.run_init_scalars).parameters
    assert list(run_sig)[:5] == ["r", "slots", "scale_dp", "descale_dp_out", "consts"] and run_sig["consts"].default == ()
    compile_sig = inspect.signature(_quantize.compile_init_scalars).parameters
    # the three slot facts first; the appended ``descale_dp`` (default True = this arm's artifact, the one with the reciprocal) is the MXFP8
    # backward's switch and leaves the fp8 artifact byte-identical
    assert list(compile_sig)[:3] == ["n_slots", "const_slot0", "n_consts"] and (compile_sig["const_slot0"].default, compile_sig["n_consts"].default) == (0, 0)
    assert list(compile_sig)[3:] == ["descale_dp"] and compile_sig["descale_dp"].default is True
    pro_run = inspect.signature(_fused.run_fp8_bwd_prologue).parameters
    assert pro_run["consts"].kind is inspect.Parameter.KEYWORD_ONLY and pro_run["consts"].default == ()
    pro_compile = inspect.signature(_fused.compile_fp8_bwd_prologue).parameters
    assert list(pro_compile)[-2:] == ["const_slot0", "n_consts"] and (pro_compile["const_slot0"].default, pro_compile["n_consts"].default) == (0, 0)
    exe = inspect.getsource(GatedAttentionBlockBwd._execute_quant)
    assert (
        'scale_q=vals["scale_q"]' in exe and 'scale_k=vals["scale_k"]' in exe and 'scale_v=vals["scale_v"]' in exe
    ), "the prologue's rebuild / v8 jobs take the static scales as kernel arguments (Python floats), never slot views of the block it writes"
    assert "consts=tuple(vals[n] for n in QUANT_CONST_SLOTS)" in exe, "the prologue's init job is handed every plan-time constant, in slot order"
    assert blk.quant is not None and "GatedAttentionBlockBwd" in type(blk).__name__
    src = inspect.getsource(GatedAttentionBlockBwd.compile) + inspect.getsource(GatedAttentionBlockBwd._quant_const_values)
    assert (
        "torch.full" not in src and "torch.empty" not in src and "torch.zeros" not in src and "device=" not in src
    ), "compile() must write nothing to the device"


@requires_rubin
def test_fp8_quant_scalars_hold_the_plan_time_constants():
    """After an execute the tail of ``quant_scalars()`` holds the ``QuantSpec``'s plan-time constants EXACTLY -- each slot the fp32
    RN of the Python value the init launch was handed (``np.float32(v)``) -- and the consumers read THOSE slots: the four alphas
    are ONE fp32 product of a published descale with its factor slot (``alpha_b1 = descale_dy * descale_o``, ``alpha_b2 =
    descale_dy * descale_w_o``, ``alpha_b7 = descale_dqkvg * descale_h``, ``alpha_b8 = descale_dqkvg * descale_w_qkvg``),
    ``scale_s`` / ``descale_s`` are ``2**FP8_SCALE_S_LOG2`` and its reciprocal, ``scale_dqkv`` is 1."""
    res = _cell_backward(_BITWISE_CELL)
    sc, vals = res.scalars, res.blk._quant_const_values()
    for name in _api_const("QUANT_CONST_SLOTS"):
        assert sc[name] == _f32(vals[name]), (name, sc[name], vals[name])
    s_log2 = _api_const("FP8_SCALE_S_LOG2")
    assert (sc["scale_s"], sc["descale_s"], sc["scale_dqkv"]) == (2.0**s_log2, 2.0**-s_log2, 1.0)
    assert sc["alpha_b1"] == _f32_mul(sc["descale_dy"], sc["descale_o"]) and sc["alpha_b2"] == _f32_mul(sc["descale_dy"], sc["descale_w_o"])
    assert sc["alpha_b7"] == _f32_mul(sc["descale_dqkvg"], sc["descale_h"]) and sc["alpha_b8"] == _f32_mul(sc["descale_dqkvg"], sc["descale_w_qkvg"])
    assert sc["descale_q"] == _f32(1.0 / res.spec.scale_q) and sc["scale_o"] == _f32(res.spec.scale_o)


def _device_records_of(prof, path) -> tuple:
    """``(records, streams)`` of a profile's device-side activity -- kernels, memcpys, memsets -- read from the exported chrome
    trace, the one place torch's profiler exposes the STREAM a record ran on: ``records`` the ``(category, name)`` list,
    ``streams`` the set of stream ids they ran on."""
    import json

    prof.export_chrome_trace(str(path))
    with open(path) as f:
        events = json.load(f)["traceEvents"]
    device = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    return [(e["cat"], e["name"]) for e in device], {e.get("args", {}).get("stream") for e in device}


@requires_rubin
@pytest.mark.parametrize("how", ["class", "wrapper"])
def test_fp8_first_use_on_an_explicit_stream_reads_nothing_the_ambient_stream_wrote(how, tmp_path):
    """A FRESH block (compiled inside the test: a cache miss) executed for the FIRST time on an explicit non-blocking stream that
    is NOT the ambient one.  The plan-time constants (``descale_q / k / v / o``, ``scale_s / descale_s``, the alpha factors, the
    static scales, ``scale_dqkv``) are slots of the scalar block that the scalar-init launch stores from its kernel arguments ON
    THE EXECUTION STREAM; device tensors filled at ``compile()`` instead would be enqueued on the AMBIENT stream, which an execute
    on another stream never waits for -- a first use behind pending ambient work would consume them before the fills landed.
    This is the STRUCTURAL half of the pin, without a timing window (the DELAYED-PRODUCER half -- the ambient stream parked from
    before ``compile()`` until past the constants' first consumer -- is the next test):
    the CLASS -- ``compile()`` records NO device-side activity at all (CUPTI: no kernel, memcpy or memset -- there is no
    producer to delay) and leaves the allocation counter unchanged, and every device-side record of the first
    ``execute(current_stream=)`` ran on ONE stream, the execution stream, with every gradient and the whole scalar block
    bitwise the synchronised run's; the WRAPPER -- a cache-miss call compiles inside, OUTSIDE its stream context, and every
    device-side record of the whole call (compile included) ran on that one stream, with bitwise gradients."""
    import cuda.bindings.driver as cuda_drv
    from torch.profiler import ProfilerActivity, profile

    res = _cell_backward(_BY_ID["s256_causal_b1-norm"])
    inp, saved = res.inp, res.saved
    side = torch.cuda.Stream()  # a non-blocking pool stream: nothing orders it against the legacy default stream
    ambient = torch.cuda.default_stream()
    assert torch.cuda.current_stream() == ambient and side != ambient
    cs = cuda_drv.CUstream(side.cuda_stream)
    torch.cuda.synchronize()
    if how == "class":
        blk = _declare_fp8_bwd(res.dy, saved, inp, res.geom, quant=res.spec)
        blk.check_support()
        ws = torch.empty_like(res.ws).fill_(0xFF)
        grads = _alloc_grads(res.blk, fill=float("nan"))
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            blk.compile()
            torch.cuda.synchronize()
        compile_records, _ = _device_records_of(prof, tmp_path / "compile.json")
        assert torch.cuda.memory_allocated() == before, "compile() allocated a CUDA tensor"
        assert blk.get_workspace_size() == ws.numel()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            _execute_fp8(blk, inp, saved, res.dy, grads, ws, scale_dp=res.scale_dp_t, current_stream=cs)
            side.synchronize()  # the EXECUTION stream only
        records, streams = _device_records_of(prof, tmp_path / "first_use_class.json")
        if not records:
            pytest.skip("torch.profiler recorded no CUDA events (CUPTI unavailable on this node); the compile-enqueues-nothing pin is unverified here")
        assert compile_records == [], f"compile() enqueued device work -- a fill an execute on another stream could race: {compile_records}"
        assert len(streams) == 1, f"the first execute ran on {len(streams)} streams ({streams}): device work escaped the execution stream -- {records}"
        got_block = _view(ws, blk._layout().quant_scalars, (len(_api_const("QUANT_SCALAR_SLOTS")),), torch.float32)
    else:
        for t_ in (saved.h, inp["w_qkvg"], inp["w_o"], inp["w_q_norm"], inp["w_k_norm"]):
            t_.requires_grad_(True)
        kept = dict(_api_bwd._BWD_CACHE)
        _api_bwd._BWD_CACHE.clear()  # a cache MISS: the wrapper compiles inside the call, outside its stream context
        try:
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
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
                    current_stream=cs,
                    quant=res.spec,
                    scale_dp=res.scale_dp_t,
                )
                side.synchronize()
            grads = {k: out[k] for k in ("dh", "dw_qkvg", "dw_o", "dw_q_norm", "dw_k_norm")}
            got_block = None
        finally:
            _api_bwd._BWD_CACHE.clear()  # drop the entry the wrapper call added on its cache miss before restoring the kept ones
            _api_bwd._BWD_CACHE.update(kept)
            for t_ in (saved.h, inp["w_qkvg"], inp["w_o"], inp["w_q_norm"], inp["w_k_norm"]):
                t_.requires_grad_(False)
        records, streams = _device_records_of(prof, tmp_path / "first_use_wrapper.json")
        if not records:
            pytest.skip("torch.profiler recorded no CUDA events (CUPTI unavailable on this node); the one-stream pin is unverified here")
        assert len(streams) == 1, f"the first use ran on {len(streams)} streams ({streams}): device work escaped the execution stream -- {records}"
    torch.cuda.synchronize()
    for name, ten in grads.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), f"{name}: the first use on the explicit stream differs from the synchronised run ({how})"
    if got_block is not None:
        assert torch.equal(got_block, _scalar_block(res)), "the scalar block (the plan-time constants included) differs from the synchronised run"


def _poison_the_small_pool(n: int = 64) -> None:
    """NaN into ``n`` freed 512-B blocks of the caching allocator's small pool (a 1-element fp32 tensor is one such block), so a
    device fill that has NOT landed yet reads as NaN instead of hiding behind a stale RIGHT value left in a reused block."""
    junk = [torch.full((1,), float("nan"), dtype=torch.float32, device="cuda") for _ in range(n)]
    torch.cuda.synchronize()
    del junk


# The nominal park of the delayed-producer first-use test (``torch.cuda._sleep`` counts cycles at ~2 GHz): 2-4x the 7-14 s a
# fresh block's ``compile()`` took on Rubin (cc 10.7, 204 SMs) across runs.  The premise it must hold is MEASURED by the test; a
# lost premise is retried once with a park sized from the measured compile, never passed over.
_FIRST_USE_PARK_S = 30.0


def _record_the_first_use_premise(monkeypatch, ambient: torch.cuda.Stream, log: list) -> None:
    """Wrap ``GatedAttentionBlockBwd.compile`` so every block compiled from here on records (one dict appended to ``log``) its
    compile wall time, whether ``ambient`` was still busy when ``compile()`` returned, and whether it was still busy when
    quantize-dY -- stage 2 of an execute, launch 2 (the fused prologue, launch 1, is the launch whose init job STORES the plan-time
    constants; quantize-dY is their FIRST consumer: ``alpha_b1`` / ``alpha_b2`` from the ``descale_o`` / ``descale_w_o`` slots) --
    was issued.  The stage's own ``execute`` is put back as soon as that is recorded, so no block keeps the recording wrap past
    its first issue of the stage."""
    orig = GatedAttentionBlockBwd.compile

    def compile_recorded(self):
        rec = {}
        log.append(rec)
        t0 = time.time()
        orig(self)
        rec["compile_s"] = time.time() - t0
        rec["parked_after_compile"] = not ambient.query()
        stage = self._quant_dy
        orig_exec = stage.execute

        def first_consumer(*a, **kw):
            rec["parked_at_first_consumer"] = not ambient.query()
            del stage.execute  # the premise is recorded: the class method is the stage's execute again, before this issue proceeds
            return orig_exec(*a, **kw)

        stage.execute = first_consumer

    monkeypatch.setattr(GatedAttentionBlockBwd, "compile", compile_recorded)


@requires_rubin
@pytest.mark.parametrize("how", ["class", "wrapper"])
def test_fp8_first_use_on_an_explicit_stream_with_the_ambient_stream_parked_past_compile(how, monkeypatch):
    """The DELAYED-PRODUCER half of the first-use pin: ambient and execution streams genuinely different, and the ambient
    stream busy (a long spin) from BEFORE ``compile()`` until PAST the constants' first consumer.  Device tensors filled at
    ``compile()`` would sit behind that spin while quantize-dY reads them on the execution stream -- NaN alphas, NaN gradients
    (the previous tree fails this test); the slots the prologue's init job stores on the execution stream cannot (every gradient and
    the scalar block are bitwise the synchronised run's).

    Why the park is sized against ``compile()`` and not the execute: on a FRESH block the fused prologue (stage 1, launch 1: the
    scalar init that stores the constants, the dY amax partials, the Q / K rebuild and v8) and quantize-dY (stage 2, launch 2: it
    reduces the prologue's partials and casts) are issued on the execution stream while the ambient stream is still busy, and
    only the THIRD stage -- the first FROST GEMM's lazy first-use setup, launch 3 -- blocks the host until the device is idle
    (measured on Rubin: the premise record below asserts the first two stages were issued while the park was still running, and
    the GEMM launch returns when the park ends).  So the window runs from the fill to stage 2 (launch 2, quantize-dY, is the
    constants' first consumer), a park shorter than
    ``compile()`` (7-14 s for a fresh block across runs) proves nothing, and the premise is ASSERTED, never assumed: the ambient stream
    was still busy when ``compile()`` returned AND when quantize-dY was issued (a test-side wrap of both records it); a lost
    premise retries once with a park sized from the measured compile time, then fails.  The caching allocator's small pool is
    poisoned with NaN first, so an unlanded fill cannot hide behind a stale right value in a reused block.  Both entry points:
    the CLASS (``compile()`` then ``execute(current_stream=)``) and the WRAPPER (a cache-miss call compiles inside, outside
    its stream context; the block it compiled is dropped from the process-global cache afterwards, so no later call reuses
    it)."""
    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("torch.cuda._sleep is the park; the matmul fallback lasts a few hundred ms, shorter than compile()")
    import cuda.bindings.driver as cuda_drv

    res = _cell_backward(_BY_ID["s256_causal_b1-norm"])
    inp, saved = res.inp, res.saved
    side = torch.cuda.Stream()  # a non-blocking pool stream: nothing orders it against the legacy default stream
    ambient = torch.cuda.default_stream()
    assert torch.cuda.current_stream() == ambient and side != ambient
    cs = cuda_drv.CUstream(side.cuda_stream)
    log = []
    _record_the_first_use_premise(monkeypatch, ambient, log)
    park_s = _FIRST_USE_PARK_S
    grads = got_block = None
    for _attempt in range(2):
        torch.cuda.synchronize()
        _poison_the_small_pool()
        if how == "class":
            blk = _declare_fp8_bwd(res.dy, saved, inp, res.geom, quant=res.spec)
            blk.check_support()
            ws = torch.empty_like(res.ws).fill_(0xFF)
            grads = _alloc_grads(res.blk, fill=float("nan"))
            torch.cuda.synchronize()
            park_the_default_stream(park_s)  # the AMBIENT stream is busy from here, through compile() and past the first consumer
            blk.compile()
            _execute_fp8(blk, inp, saved, res.dy, grads, ws, scale_dp=res.scale_dp_t, current_stream=cs)
            side.synchronize()  # the EXECUTION stream only
            got_block = _view(ws, blk._layout().quant_scalars, (len(_api_const("QUANT_SCALAR_SLOTS")),), torch.float32)
        else:
            for t_ in (saved.h, inp["w_qkvg"], inp["w_o"], inp["w_q_norm"], inp["w_k_norm"]):
                t_.requires_grad_(True)
            kept = dict(_api_bwd._BWD_CACHE)
            _api_bwd._BWD_CACHE.clear()  # a cache MISS: the wrapper compiles inside the call, outside its stream context
            try:
                park_the_default_stream(park_s)
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
                    current_stream=cs,
                    quant=res.spec,
                    scale_dp=res.scale_dp_t,
                )
                side.synchronize()
                grads = {k: out[k] for k in ("dh", "dw_qkvg", "dw_o", "dw_q_norm", "dw_k_norm")}
            finally:
                _api_bwd._BWD_CACHE.clear()  # drop the block this call compiled (its stage was wrapped above); the cache is left as found
                _api_bwd._BWD_CACHE.update(kept)
                for t_ in (saved.h, inp["w_qkvg"], inp["w_o"], inp["w_q_norm"], inp["w_k_norm"]):
                    t_.requires_grad_(False)
        torch.cuda.synchronize()
        rec = log[-1]
        if rec["parked_after_compile"] and rec.get("parked_at_first_consumer"):
            break
        park_s = 2.0 * rec["compile_s"] + 10.0  # the premise was lost: size the park from the measured compile and retry once
    else:
        pytest.fail(
            "premise lost twice: compile() outlasted the ambient stream's park, so a compile-time fill could not have been delayed "
            f"past the first consumer ({log[-2:]})"
        )
    for name, ten in grads.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), f"{name}: the first use behind the busy ambient stream differs from the synchronised run ({how})"
    if got_block is not None:
        assert torch.equal(got_block, _scalar_block(res)), "the scalar block (the plan-time constants included) differs from the synchronised run"
