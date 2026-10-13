# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Served-domain mirror of the two graph-level softmax levers on the cc 10.7 FROST forward rows.

The levers are python-only op attributes of ``graph.sdpa`` / ``sdpa_fp8`` / ``sdpa_mxfp8``:

* ``softmax_precision=cudnn.data_type.HALF`` -- the f16x2 exponent arm of the quantized
  (per-tensor FP8 and MXFP8) kernels.  ``FLOAT`` / ``None`` is the f32 pipeline every row runs;
  half (f16/bf16) inputs never take the arm.
* ``attn_scale_prefolded=True`` -- the CALLER pre-multiplied Q by ``attn_scale * log2(e)``, leaves
  ``attn_scale`` unset, and the engine applies no scale (the kernel evaluates ``2^(S - m)`` on the
  raw QK^T).  Served by the MXFP8 and half rows on every exact flavor; declined on per-tensor FP8
  (that kernel folds ``descale_q * descale_k`` into its softmax scale), on paged KV (the paged
  bodies apply the scale in-kernel) and -- a deliberate simplification of the adapter's rule -- on
  half THD (192, 128), whose single-Q leg loads a body without the arm (the cga2 body of the same
  flavor would serve it; the mirror stays on the conservative side).

A SET attribute makes the graph backend-unlowerable: when no FROST row serves it, ``build_plans``
raises ``cudnnGraphNotSupportedError`` and the test harness WAIVEs (skips) -- silently dropping the
knob draw.  ``served_softmax_knob_sets`` mirrors the rows so a sweep never asks for an unserved set,
and a sweep that asked for a NON-default set turns a waive into a failure (the mirror or a row's
claim is wrong).  Pure python, no GPU; ``test_softmax_knobs.py`` pins it against the engine rows
(``cudnn.sdpa.fwd.engines.ENGINE_SPECS``) and the config tables (``config_sm107``).
"""

import math

import cudnn

# Q pre-multiplied by attn_scale * LOG2E makes the raw QK^T a log2-domain score; a reference that
# evaluates softmax(LN2 * S_folded) = 2^S_folded then sees the same attention as the kernel.
LOG2E = math.log2(math.e)
LN2 = math.log(2.0)

FAMILIES = ("half", "fp8", "mxfp8")
# The exact kernel flavors of the cc 10.7 rows (the mirror is exact on these; any other head-dim
# pair is served through an envelope, or not at all, and gets the default set only).
EXACT_FLAVORS = frozenset({(128, 128), (192, 128), (256, 256), (512, 512)})
# (softmax_precision, attn_scale_prefolded): None/False = the f32 pipeline with the in-kernel scale.
DEFAULT_KNOB_SET = (None, False)
HALF = cudnn.data_type.HALF


def served_softmax_knob_sets(family, d_qk, d_v, *, paged=False, thd=False):
    """The ``(softmax_precision, attn_scale_prefolded)`` sets the cc 10.7 FROST row of ``family``
    serves for an exact flavor on this path.  The first entry is always ``DEFAULT_KNOB_SET``.

    ``family``: "half" (f16/bf16 ``graph.sdpa``), "fp8" (per-tensor ``graph.sdpa_fp8``) or "mxfp8"
    (``graph.sdpa_mxfp8``).  ``paged``: K/V behind page tables.  ``thd``: ragged (packed) Q.  A path
    the row does not serve at all (per-tensor FP8 paged KV, MXFP8 THD outside d256 or over pools, MXFP8 pools outside
    d128 / d256)
    yields the default set only -- the harness then waives the case for the path, not for a knob."""
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}; got {family!r}")
    flavor = (d_qk, d_v)
    if flavor not in EXACT_FLAVORS:
        return (DEFAULT_KNOB_SET,)
    quantized = family in ("fp8", "mxfp8")
    # Paths the quantized rows do not serve on cc 10.7: paged KV on the per-tensor FP8 row, THD on the MXFP8 row outside d256
    # (the row's thd_d_shapes, #1488) or over pools, and MXFP8 pools outside d128 / d256 (the row's paged_d_shapes).  The
    # f16x2 exponent arm is a softmax-warp constant, so it composes with the page loader; the fold stays declined over paged
    # KV on every row (engines.mismatch).
    path_served = (
        not (family == "fp8" and paged)
        and not (family == "mxfp8" and thd and (paged or flavor != (256, 256)))
        and not (family == "mxfp8" and paged and flavor not in ((128, 128), (256, 256)))
    )
    half_exp = quantized and path_served
    fold = family in ("half", "mxfp8") and path_served and not paged and not (family == "half" and thd and flavor == (192, 128))
    sets = [DEFAULT_KNOB_SET]
    if half_exp:
        sets.append((HALF, False))
    if fold:
        sets.append((None, True))
    if half_exp and fold:
        sets.append((HALF, True))
    return tuple(sets)


def knob_set_for_case(served, case_index):
    """``served[(case_index - 1) % len(served)]`` for a 1-based sweep case index: exactly uniform
    over the served set for any case count that is a multiple of its size, and it consumes no
    rng draw (the geometry of a seeded case is independent of the knob matrix)."""
    return served[(case_index - 1) % len(served)]


def knob_set_label(knob_set):
    """Short name of a knob set for messages: FLOAT, HALF, FLOAT+fold, HALF+fold."""
    precision, prefolded = knob_set
    name = "HALF" if precision == HALF else "FLOAT"
    return name + ("+fold" if prefolded else "")


def is_default_knob_set(knob_set):
    precision, prefolded = knob_set
    return (precision is None or precision == cudnn.data_type.FLOAT) and not prefolded


def prefold_factor(attn_scale):
    """The factor the caller multiplies Q by under ``attn_scale_prefolded=True``."""
    return attn_scale * LOG2E
