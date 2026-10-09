# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The additive attention bias and the per-key boolean mask on the Rubin (SM107) f16/bf16 small head-dim tiles -- the RESIDENT
row-block arm of ``sm107/prefill_d128_f16.py`` (``TemplateParams.bias_mode = 1`` / ``kv_mask_form = 1``).

One work item (a CTA's ``TILES_Q x TILE_M = 256`` query rows of one (batch, head)) loads its bias rows x its whole KV range into
SMEM ONCE (the TMA-LDG warp, after the item's Q / K / V prologue loads; one ``expect_tx`` for the FULL boxes of every tile), and
the softmax warps read their row slice with 128-byte-swizzled lane loads -- the first 64-column chunk BEFORE the ``bmm1_done``
wait -- and add it to the raw scores pre-scale as one ``fma.rn.f32x2`` per pair (``s + bias / scale``: the exp2 argument's form
is unchanged), then apply the band mask AND the per-key mask word (``kv_mask``: 4 byte loads + 4 warp ballots per 128-column
tile) in ONE bit-word pass with the true -inf, so a row whose every key is masked ends the item with ``total_sum == 0`` exactly
and the correction epilogue SELECTs O = 0 / Stats = -inf.  The block's budget is what the carveout leaves beside the Q / O slabs
and the K / V ring, in whole 128-column tiles -- 3 at the 32-wide tile (S_kv <= 384), 1 / 2 at the 64-wide one at cga 1 / 2
(128 / 256), none at d128 -- a BIAS cell above it takes the STREAMED ring where a 2- / 3-stage ring of its tiles fits beside the
operands (``bias_mode = 2``, test_sdpa_bias_streamed_sm107.py) and is a TYPED decline only where neither form fits (the one-batch-entry
item at the 64-wide tile's cga1); the kv_mask ALONE stages nothing in SMEM, so it carries no S_kv budget and the ring-wrap axis (n_kv = 4 / 5 / 8) runs on
it.  The bias is ``[1|B, 1|H_q, S_q, S_kv]`` in the Q/K/V dtype with a contiguous key dim and 16-byte strides (a COMPACT bias needs
S_kv % 8 == 0 -- the acceptance-grid cells S = 100 / 217 take a padded row stride, 104 / 224); the mask ``[1|B, 1, 1, S_kv]`` BOOLEAN
(a python-only ``sdpa`` input the backend has no field for).

Three tiers: (1) CPU contract -- the row's accept / decline table (budgets, dtypes, legs, shapes) and the adapter twin agreeing on
every cell, the config factory's derived block geometry and raising couplings, the analyzer's bias / kv_mask facts and the
malformed-shape rejections, the append-only records; (2) host rendering pins -- the no-bias renderings of every served arm
byte-identical to the committed pre-bias cubin records, the bias / kv_mask renderings keeping the register split, the bit-word
mask form, no cluster drain and no new spills; (3) GPU cells on a Rubin device -- the fp64 oracle (softmax of scale * q.k + bias
with the padded / banded / per-key masks) over the degenerate-input matrix within the resident budget, the bias-magnitude cells,
the fully masked kv_mask row against the zero-length sequence, the IDENTITY check (bias = 0, mask = all True) BITWISE against the
plain kernel, and the typed declines on real graphs.
"""

import dataclasses
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from frost_test_utils import assert_no_new_spills, make_dense_stats, requires_dsl, requires_rubin, run_sass_probe, sass_probe_source

pytestmark = [pytest.mark.L0, requires_dsl]

_ROW = "sdpa_fwd_prefill_sm107"
_KERNEL = Path(__file__).resolve().parents[4] / "python" / "cudnn" / "sdpa" / "fwd" / "kernels" / "sm107" / "prefill_d128_f16.py"
_RENDERINGS = Path(__file__).resolve().parent / "renderings"
# (head-dim tile, cga) -> the resident block's S_kv budget, DERIVED by config_sm107 from the carveout (pinned here as the
# numbers the row and the adapter must agree on).
_BUDGET = {(32, 1): 384, (64, 1): 128, (64, 2): 256, (128, 1): 0, (128, 2): 0}


@pytest.fixture(autouse=True)
def _sm107_target_for_metadata(monkeypatch):
    """Model SM107 eligibility on other GPUs without requiring their DSL to target Rubin (the sm107 suite's fixture)."""
    from cudnn.frost import buffers

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)


# --------------------------------------------------------------------------- tier 1: CPU contract


class _IrTensor:
    """A stand-in for the graph's IR tensor where ``mismatch`` reads its dtype (``facts.bias_t.get_data_type()``)."""

    def __init__(self, dt):
        self._dt = dt

    def get_data_type(self):
        return self._dt


def _facts(**kw):
    import cudnn
    from cudnn.sdpa import graph_analyzer as ga

    base = dict(b=2, h_q=4, h_kv=4, s_q=256, s_kv=256, d_qk=32, d_v=32, dtype=cudnn.data_type.BFLOAT16, dtype_o=cudnn.data_type.BFLOAT16, device_cc=(10, 7))
    base.update(kw)
    return ga.SdpaGraphFacts(**base)


def _bias_kw(b=1, h=4, s_q=256, s_kv=256, dt=None, row=None):
    """``row`` = a padded row stride (the TMA 16-byte rule at an S_kv % 8 != 0: 224 for S_kv 217, 104 for 100); None = compact."""
    import cudnn

    dt = cudnn.data_type.BFLOAT16 if dt is None else dt
    row = s_kv if row is None else row
    return dict(has_bias=True, bias_t=_IrTensor(dt), bias_dim=(b, h, s_q, s_kv), bias_stride=(h * s_q * row, s_q * row, row, 1), bias_dtype=dt)


def _mask_kw(b=2, s_kv=256, dt=None):
    import cudnn

    dt = cudnn.data_type.BOOLEAN if dt is None else dt
    return dict(has_kv_mask=True, kv_mask_t=_IrTensor(dt), kv_mask_dim=(b, 1, 1, s_kv), kv_mask_stride=(s_kv, s_kv, s_kv, 1), kv_mask_dtype=dt)


def _caps():
    from cudnn.sdpa.fwd import engines

    return {s.name: s.capabilities for s in engines.ENGINE_SPECS}[_ROW]


def test_row_claims_the_bias_on_the_derived_shapes_and_budgets():
    """The serving set and the S_kv budgets are DERIVED from the SMEM arithmetic (one function for the row, the adapter and the
    config factory, rule 8b'): 3 tiles at the 32-wide tile, 1 / 2 at the 64-wide one (cga 1 / 2), none at d128."""
    import cudnn
    from cudnn.sdpa.fwd import config_sm107 as c

    caps = _caps()
    # BOOLEAN = the bit word; HALF / BFLOAT16 / FLOAT = the additive vector (test_sdpa_bias_tile_n_additive_sm107.py pins its gates)
    assert caps.bias and caps.bias_io_dtype_only and caps.kv_mask
    assert caps.kv_mask_dtypes == frozenset({cudnn.data_type.BOOLEAN, cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT})
    assert caps.bias_d_shapes == c.SM107_BIAS_D_SHAPES == frozenset({(32, 32), (64, 64)})
    for (d, cga), s_kv_max in _BUDGET.items():
        tiles = c.sm107_bias_resident_tiles(d, cga)
        assert c.sm107_bias_resident_s_kv_max(d, cga) == s_kv_max == tiles * 128, (d, cga, tiles)
        # the arithmetic: (320 KiB - 2 KiB fixed - Q/O slabs - K/V ring) // one 256 x 128 bf16 tile (64 KiB)
        stages = c._stages_kv_d128(2, cga, mxfp8=False, tile_k=d)
        operands = c.sdpa_smem_bytes(128, 128, d, d, 2, stages, cga, 2, 2)
        assert tiles == max(0, (c.SMEM_USABLE_BYTES - c._SMEM_FIXED_OVERHEAD - operands) // (256 * 128 * 2)), (d, cga)
    assert c.sm107_bias_block_rows() == 256 and c.BIAS_SWZ_BYTES == 128


# (name, facts kwargs, knobs kwargs, accept?) -- the row's table; every DECLINE is typed (a reason string from mismatch) and the
# adapter twin below raises NotImplementedError / ValueError on the same cell.
_CELLS = [
    ("bias d32 S256", dict(**_bias_kw()), None, True),
    ("bias d32 S384 = the 3-tile budget", dict(s_q=384, s_kv=384, **_bias_kw(s_q=384, s_kv=384)), None, True),
    # The (n, n+1) PAIR item: a batch-broadcast bias on >= 2 flattened rows runs it (a 128-row block, twice the kv columns: 768 / 384
    # at d32 / d64 cga1); a per-batch bias [B, ., S, S] or a single row keeps the one-batch-entry item and ITS budget (384 / 128 / 256).
    ("bias d32 S512: the pair item's 6-tile block", dict(s_q=512, s_kv=512, **_bias_kw(s_q=512, s_kv=512)), None, True),
    ("per-batch bias d32 S512 over the 3-tile budget: the streamed ring", dict(s_q=512, s_kv=512, **_bias_kw(b=2, s_q=512, s_kv=512)), None, True),
    ("one-row bias d32 S512 over the 3-tile budget: the streamed ring", dict(b=1, s_q=512, s_kv=512, **_bias_kw(s_q=512, s_kv=512)), None, True),
    ("bias d64 S128 cga1", dict(d_qk=64, d_v=64, s_q=128, s_kv=128, **_bias_kw(s_q=128, s_kv=128)), None, True),
    ("bias d64 S256 cga1: the pair item's 3-tile block", dict(d_qk=64, d_v=64, **_bias_kw()), dict(cga=1), True),
    ("per-batch bias d64 S256 cga1 over the 1-tile budget", dict(d_qk=64, d_v=64, **_bias_kw(b=2)), dict(cga=1), False),
    ("bias d64 S256 cga2 (2 tiles)", dict(d_qk=64, d_v=64, **_bias_kw()), dict(cga=2), True),
    ("bias d64 S256 knob-less: some width holds it (cga1 pair; cga2 one-batch-entry)", dict(d_qk=64, d_v=64, **_bias_kw()), None, True),
    ("bias d64 S384: the pair item's 3-tile block at cga1", dict(d_qk=64, d_v=64, s_q=384, s_kv=384, **_bias_kw(s_q=384, s_kv=384)), None, True),
    (
        "per-batch bias d64 S384 over both widths' budgets: the cga2 streamed ring",
        dict(d_qk=64, d_v=64, s_q=384, s_kv=384, **_bias_kw(b=2, s_q=384, s_kv=384)),
        None,
        True,
    ),
    ("bias d128: no block fits", dict(d_qk=128, d_v=128, **_bias_kw()), None, False),
    ("bias d24 rides the 32-wide tile", dict(d_qk=24, d_v=24, **_bias_kw()), None, True),
    ("fp32 bias", dict(**_bias_kw(dt="FLOAT")), None, False),
    ("per-batch bias [B, H, S, S]", dict(**_bias_kw(b=2)), None, True),
    ("head-broadcast bias [1, 1, S, S]", dict(**_bias_kw(h=1)), None, True),
    (
        "bias with a non-contiguous key dim",
        dict(has_bias=True, bias_t=_IrTensor("BF16"), bias_dim=(1, 4, 256, 256), bias_stride=(262144, 65536, 1, 256), bias_dtype="BF16"),
        None,
        False,
    ),
    ("bias + THD", dict(thd=True, padded=True, **_bias_kw()), None, False),
    ("bias + split-KV", dict(**_bias_kw()), dict(split_kv=2), False),
    ("bias + PackGQA", dict(h_kv=2, **_bias_kw()), dict(pack_gqa=True), False),
    ("bias + score outputs", dict(has_score_max=True, **_bias_kw()), None, False),
    ("bias + padded + sink", dict(padded=True, has_sink=True, **_bias_kw()), None, True),
    ("bias + causal + SWA", dict(causal=True, window_left=64, **_bias_kw()), None, True),
    ("kv_mask bool", dict(**_mask_kw()), None, True),
    ("kv_mask broadcast [1, 1, 1, S]", dict(**_mask_kw(b=1)), None, True),
    ("kv_mask int8", dict(**_mask_kw(dt="INT8")), None, False),
    ("kv_mask + bias", dict(**_bias_kw(), **_mask_kw()), None, True),
    ("kv_mask d128", dict(d_qk=128, d_v=128, **_mask_kw()), None, False),
    ("kv_mask d64 S256 knob-less: no block, the tile's own width", dict(d_qk=64, d_v=64, **_mask_kw()), None, True),
    ("kv_mask d32 S512 above the BIAS budget: served (nothing staged)", dict(s_q=512, s_kv=512, **_mask_kw(s_kv=512)), None, True),
    ("kv_mask d32 S1024 (n_kv 8)", dict(s_q=1024, s_kv=1024, **_mask_kw(s_kv=1024)), None, True),
    ("kv_mask d64 S512 cga2 above the bias budget: served", dict(d_qk=64, d_v=64, s_q=512, s_kv=512, **_mask_kw(s_kv=512)), dict(cga=2), True),
    (
        "kv_mask + bias S512: the bias carries the budget (the pair item holds 512)",
        dict(s_q=512, s_kv=512, **_bias_kw(s_q=512, s_kv=512), **_mask_kw(s_kv=512)),
        None,
        True,
    ),
    (
        "kv_mask + per-batch bias S512: over the one-batch-entry budget -- the streamed ring",
        dict(s_q=512, s_kv=512, **_bias_kw(b=2, s_q=512, s_kv=512), **_mask_kw(s_kv=512)),
        None,
        True,
    ),
    ("bias S64: one tile", dict(s_q=64, s_kv=64, **_bias_kw(s_q=64, s_kv=64)), None, True),
    ("bias S100 compact: S_kv % 8 != 0 is not 16-B strided", dict(s_q=100, s_kv=100, **_bias_kw(s_q=100, s_kv=100)), None, False),
    ("bias S217 compact (the acceptance-grid customer cell)", dict(s_q=217, s_kv=217, **_bias_kw(s_q=217, s_kv=217)), None, False),
    ("bias S217 with a padded row stride 224", dict(s_q=217, s_kv=217, **_bias_kw(s_q=217, s_kv=217, row=224)), None, True),
    ("bias S100 with a padded row stride 104", dict(s_q=100, s_kv=100, **_bias_kw(s_q=100, s_kv=100, row=104)), None, True),
    ("plain d32 (no arm)", dict(), None, True),
]


def _resolve_dtypes(kw):
    import cudnn

    names = {"FLOAT": cudnn.data_type.FLOAT, "BF16": cudnn.data_type.BFLOAT16, "INT8": cudnn.data_type.INT8}
    out = dict(kw)
    for key in ("bias_dtype", "kv_mask_dtype"):
        if isinstance(out.get(key), str):
            out[key] = names[out[key]]
    for key in ("bias_t", "kv_mask_t"):
        t = out.get(key)
        if t is not None and isinstance(t._dt, str):
            out[key] = _IrTensor(names[t._dt])
    return out


@pytest.mark.parametrize("name, fkw, kkw, accept", _CELLS, ids=[c[0] for c in _CELLS])
def test_row_accepts_and_declines_the_bias_and_kv_mask(name, fkw, kkw, accept):
    """The engine row's table: an accept is ``mismatch() is None`` and the selected flavor is a bias shape; a decline names its
    reason (budget / dtype / leg / shape).  Every declined cell must be a REAL decline on a REAL graph too (the GPU tier)."""
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    caps = _caps()
    f = _facts(**_resolve_dtypes(fkw))
    knobs = SdpaFwdKnobs(**kkw) if kkw else None
    why = engines.mismatch(caps, f, knobs)
    assert (why is None) == accept, (name, why)
    if accept and (f.has_bias or f.has_kv_mask):
        assert engines._selected_d_shape(caps, f, knobs) in caps.bias_d_shapes, name


def test_heuristics_pick_the_narrowest_width_whose_block_holds_s_kv():
    """The row's knob-less eligibility and the heuristics' width agree (rule 8b'): a d64 PER-BATCH bias graph at S_kv 256 gets cga2
    plans (the one-batch-entry item's 2-tile block) and NO cga1 plan from ``recommend``, while a batch-BROADCAST bias at the same
    shape runs the (n, n+1) pair item on cga1 (its 128-row block holds 384 there); at S_kv 128 the tuned cga1 stays; a d32
    graph stays on its cga1-only tile; a per-batch bias above every width's RESIDENT budget takes the STREAMED ring at the narrowest
    width whose ring fits (d64: cga2, the only width with a 2-stage ring for the 256-row item; d32: cga1).  A kv_mask ALONE stages no
    block: it keeps the tile's own width (cga1) at every S_kv."""
    from cudnn.sdpa.fwd.heuristics import recommend

    offered = {_ROW: 20599}
    for fkw, want in (
        (dict(d_qk=64, d_v=64, **_bias_kw(b=2)), {2}),
        (dict(d_qk=64, d_v=64, **_bias_kw()), {1}),
        (dict(d_qk=64, d_v=64, **_mask_kw()), {1}),
        (dict(s_q=1024, s_kv=1024, **_mask_kw(s_kv=1024)), {1}),
        (dict(d_qk=64, d_v=64, **_bias_kw(b=2), **_mask_kw()), {2}),
        (dict(d_qk=64, d_v=64, **_bias_kw(), **_mask_kw()), {1}),
        (dict(d_qk=64, d_v=64, s_q=128, s_kv=128, **_bias_kw(s_q=128, s_kv=128)), {1}),
        (dict(**_bias_kw()), {1}),
        (dict(s_q=384, s_kv=384, **_bias_kw(s_q=384, s_kv=384)), {1}),
        (dict(d_qk=64, d_v=64, s_q=384, s_kv=384, **_bias_kw(b=2, s_q=384, s_kv=384)), {2}),  # the cga2 streamed ring
        (dict(s_q=512, s_kv=512, **_bias_kw(b=2, s_q=512, s_kv=512)), {1}),  # the one-batch-entry item's streamed ring at cga1
        (dict(s_q=512, s_kv=512, **_bias_kw(s_q=512, s_kv=512)), {1}),
    ):
        plans = recommend("A", _facts(device_sm_count=204, **_resolve_dtypes(fkw)), offered)
        assert {p.knobs.cga for p in plans} == want, (fkw, [p.knobs for p in plans])
        assert all((p.knobs.split_kv or 1) == 1 and not p.knobs.pack_gqa for p in plans), [p.knobs for p in plans]


def _desc(shape, dtype, name, stride=None):
    """A standalone-adapter descriptor.  The ACTIVATIONS (``q`` / ``k`` / ``v`` / ``o``, shape (B, H, S, D)) default to the row's
    envelope, BSHD-physical (stride order 3, 1, 2, 0) -- the layout the prepared launch binds zero-copy, so the adapter cells are the
    row's twin; a (B, H, S, D)-contiguous activation would take the adapter's staging-copy path, which carries neither the bias nor the
    kv_mask and is therefore a typed decline in ``check_support`` with a resident-arm operand bound.  Every other tensor (the bias, the
    kv_mask, the lengths) defaults to its contiguous strides."""
    from cudnn.api_base import TensorDesc

    if stride is None and name in ("q", "k", "v", "o") and len(shape) == 4:
        b, h, s, d = (int(x) for x in shape)
        stride = (s * h * d, d, h * d, 1)
    stride = tuple(int(x) for x in torch.empty(shape).stride()) if stride is None else tuple(stride)
    return TensorDesc(
        dtype=dtype, shape=tuple(shape), stride=stride, stride_order=TensorDesc._compute_stride_order(tuple(shape), stride), device="cuda", name=name
    )


@pytest.fixture
def _rubin_device_for_the_adapter(monkeypatch):
    """The standalone adapter reads the device capability at construction: model a Rubin device on any host."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 7))


def _api(d=32, *, s_q=256, s_kv=256, b=2, h=4, h_kv=None, dtype=torch.bfloat16, bias=None, kv_mask=None, bias_dtype=None, bias_stride=None, **kw):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    h_kv = h if h_kv is None else h_kv
    q, o = (_desc((b, h, s_q, d), dtype, n) for n in ("q", "o"))
    k, v = (_desc((b, h_kv, s_kv, d), dtype, n) for n in ("k", "v"))
    if bias is not None:
        kw["sample_bias"] = _desc(bias, bias_dtype or dtype, "bias", stride=bias_stride)
    if kv_mask is not None:
        kw["sample_kv_mask"] = _desc(kv_mask, torch.bool, "kv_mask")
    return SdpaFwdDslSm100(q, k, v, o, None, **kw)


# (name, adapter kwargs, accept?, flavor, bias_mode, kv_mask_form, cta_mma): a knob-less cell lowers at the narrowest width
# whose block holds S_kv (the heuristics' pick), a requested width is honoured or declined, never degraded.
_ADAPTER_CELLS = [
    # the one-batch-entry item at the 32-wide tile STREAMS from two kv tiles on (the measured mode pick: +6..+13 % over its resident block)
    ("bias d32 S256", dict(bias=(1, 4, 256, 256)), True, (32, 32), 2, 0, 1),
    ("bias d32 S384", dict(s_q=384, s_kv=384, bias=(1, 4, 384, 384)), True, (32, 32), 1, 0, 1),
    # The (n, n+1) pair item: a batch-broadcast bias on >= 2 rows lowers it (128-row block, cga1); a per-batch bias keeps the old cells.
    ("bias d32 S512: the one-batch-entry item's streamed ring (S_q % 256 == 0)", dict(s_q=512, s_kv=512, bias=(1, 4, 512, 512)), True, (32, 32), 2, 0, 1),
    ("per-batch bias d32 S512 over the 3-tile budget: the streamed ring", dict(s_q=512, s_kv=512, bias=(2, 4, 512, 512)), True, (32, 32), 2, 0, 1),
    ("bias d64 S128", dict(d=64, s_q=128, s_kv=128, bias=(1, 4, 128, 128)), True, (64, 64), 1, 0, 1),
    ("bias d64 S256 cga1: the pair item", dict(d=64, bias=(1, 4, 256, 256), cga=1), True, (64, 64), 1, 0, 1),
    ("per-batch bias d64 S256 cga1 over the 1-tile budget", dict(d=64, bias=(2, 4, 256, 256), cga=1), False, None, 0, 0, None),
    ("bias d64 S256 cga2", dict(d=64, bias=(1, 4, 256, 256), cga=2), True, (64, 64), 1, 0, 2),
    ("bias d64 S256 knob-less: the adapter lowers the pair item at cga1", dict(d=64, bias=(1, 4, 256, 256)), True, (64, 64), 1, 0, 1),
    ("per-batch bias d64 S256 knob-less: the adapter lowers at cga2", dict(d=64, bias=(2, 4, 256, 256)), True, (64, 64), 1, 0, 2),
    ("bias d64 S384: the pair item at cga1", dict(d=64, s_q=384, s_kv=384, bias=(1, 4, 384, 384)), True, (64, 64), 1, 0, 1),
    ("per-batch bias d64 S384 over both widths: the cga2 streamed ring", dict(d=64, s_q=384, s_kv=384, bias=(2, 4, 384, 384)), True, (64, 64), 2, 0, 2),
    ("bias d128", dict(d=128, bias=(1, 4, 256, 256)), False, None, 0, 0, None),
    ("bias d32 cga2 lands on the 64-wide tile (2 tiles)", dict(bias=(1, 4, 256, 256), cga=2), True, (64, 64), 1, 0, 2),
    ("fp32 bias", dict(bias=(1, 4, 256, 256), bias_dtype=torch.float32), False, None, 0, 0, None),
    ("bias wrong S_kv extent", dict(bias=(1, 4, 256, 128)), False, None, 0, 0, None),
    ("bias non-contiguous key dim", dict(bias=(1, 4, 256, 256), bias_stride=(262144, 65536, 1, 256)), False, None, 0, 0, None),
    ("bias + split", dict(bias=(1, 4, 256, 256), split_kv=2), False, None, 0, 0, None),
    ("bias + sink + padded", dict(bias=(1, 4, 256, 256), has_sink=True, seq_kv_lens_present=True), True, (32, 32), 1, 0, 1),
    ("kv_mask", dict(kv_mask=(2, 1, 1, 256)), True, (32, 32), 0, 1, 1),
    ("kv_mask d64 S256 knob-less: no block, cga1", dict(d=64, kv_mask=(2, 1, 1, 256)), True, (64, 64), 0, 1, 1),
    ("kv_mask d32 S1024 above the bias budget", dict(s_q=1024, s_kv=1024, kv_mask=(2, 1, 1, 1024)), True, (32, 32), 0, 1, 1),
    ("kv_mask d64 S512 cga2 requested", dict(d=64, s_q=512, s_kv=512, kv_mask=(2, 1, 1, 512), cga=2), True, (64, 64), 0, 1, 2),
    (
        # the measured width table: a bias WITH a mask runs the 64-wide tile, where the pair's resident block holds S_kv 512 (+12 % over
        # the one-batch-entry ring at the same width, +18 % over the 128-wide ring)
        "kv_mask + bias S512: the pair item's resident block on the 64-wide tile",
        dict(s_q=512, s_kv=512, kv_mask=(2, 1, 1, 512), bias=(1, 4, 512, 512)),
        True,
        (32, 32),
        1,
        1,
        1,
    ),
    (
        "kv_mask + per-batch bias S512: over the one-batch-entry budget -- the streamed ring",
        dict(s_q=512, s_kv=512, kv_mask=(2, 1, 1, 512), bias=(2, 4, 512, 512)),
        True,
        (32, 32),
        2,
        1,
        1,
    ),
    # (the standalone adapter wants the padding mask for a dense S_kv % 128 != 0 tail; the GRAPH path synthesizes it -- the GPU cells)
    ("bias S64 one tile (padded)", dict(s_q=64, s_kv=64, bias=(1, 4, 64, 64), seq_kv_lens_present=True), True, (32, 32), 1, 0, 1),
    ("bias S217 compact", dict(s_q=217, s_kv=217, bias=(1, 4, 217, 217), seq_kv_lens_present=True), False, None, 0, 0, None),
    (
        "bias S217 padded row stride 224",
        dict(s_q=217, s_kv=217, bias=(1, 4, 217, 217), bias_stride=(4 * 217 * 224, 217 * 224, 224, 1), seq_kv_lens_present=True),
        True,
        (32, 32),
        1,
        0,
        1,
    ),
    ("kv_mask + bias", dict(kv_mask=(2, 1, 1, 256), bias=(1, 1, 256, 256)), True, (32, 32), 1, 1, 1),
    ("kv_mask wrong extent", dict(kv_mask=(2, 1, 1, 128)), False, None, 0, 0, None),
    ("kv_mask d128", dict(d=128, kv_mask=(2, 1, 1, 256)), False, None, 0, 0, None),
]


@pytest.mark.parametrize("name, kw, accept, flavor, bias_mode, kv_mask_form, cta_mma", _ADAPTER_CELLS, ids=[c[0] for c in _ADAPTER_CELLS])
def test_adapter_check_support_is_the_rows_twin(_rubin_device_for_the_adapter, name, kw, accept, flavor, bias_mode, kv_mask_form, cta_mma):
    """The standalone adapter's ``check_support`` (rule 8b twin): the same cells accept with the compile-time record carrying
    ``bias_mode`` / ``kv_mask_form`` on the expected flavor at the expected width, or decline with one of the two TYPED
    exceptions."""
    if not accept:
        with pytest.raises((NotImplementedError, ValueError)):
            _api(**kw).check_support()
        return
    api = _api(**kw)
    assert api.check_support() is True, name
    assert api.flavor == flavor, (name, api.flavor)
    params = api.template_params()
    assert (params.bias_mode, params.kv_mask_form) == (bias_mode, kv_mask_form), (name, params)
    assert params.d_flavor == flavor[0]
    assert params.cta_mma == cta_mma, (name, params.cta_mma)


def test_config_factory_derives_the_block_and_raises_on_each_coupling():
    """Every block field follows the tile and the cga (frost-kernels s5 "bind the couplings"): the tile count from the carveout,
    256 rows, two 64-column boxes per 128-column tile, the 128-B swizzle, eight releasing warps, the descriptor-fed region under
    the 256 KiB version-0 window; each coupling moved alone raises with the failure it would produce."""
    from cudnn.sdpa.fwd import config_sm107 as c
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    for d, cga, tiles in ((32, 1, 3), (64, 1, 1), (64, 2, 2)):
        cfg, _ = {32: c.make_cfg_d32, 64: c.make_cfg_d64}[d](TemplateParams(dtype_qkv=2, cta_mma=cga, bias_mode=1, kv_mask_form=1, seq_kv_lens_present=True))
        assert (
            cfg.BIAS_MODE,
            cfg.KV_MASK_FORM,
            cfg.BIAS_RESIDENT_TILES,
            cfg.BIAS_BLOCK_ROWS,
            cfg.BIAS_SWZ_BYTES,
            cfg.BIAS_BOXES_PER_TILE,
            cfg.BIAS_EMPTY_ARRIVERS,
        ) == (1, 1, tiles, 256, 128, 2, 8,), (d, cga, cfg)
        assert cfg.BIAS_EMPTY_ARRIVERS == cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_WG_WARPS
        assert c.bias_resident_block_bytes(cfg) == tiles * 256 * 128 * 2
        assert c.operand_region_end_bytes(cfg) <= c.TCGEN05_V0_ADDR_LIMIT
        total = c.operand_region_end_bytes(cfg) + c.bias_resident_block_bytes(cfg) + c._SMEM_FIXED_OVERHEAD
        assert total <= c.SMEM_USABLE_BYTES, (d, cga, total)
        for field, value, word in (
            ("BIAS_EMPTY_ARRIVERS", 4, "softmax warp count"),
            ("BIAS_BOXES_PER_TILE", 1, "boxes"),
            ("BIAS_RESIDENT_TILES", 0, "at least one"),
            ("BIAS_BLOCK_ROWS", 128, "TILES_Q x TILE_M"),
            ("BIAS_SWZ_BYTES", 64, "128-B swizzle"),
        ):
            with pytest.raises(ValueError, match=re.escape(word)):
                c._validate_bias(dataclasses.replace(cfg, **{field: value}), "probe")
    plain, _ = c.make_cfg_d32(TemplateParams(dtype_qkv=2, cta_mma=1))
    assert (plain.BIAS_MODE, plain.BIAS_RESIDENT_TILES, plain.BIAS_BLOCK_ROWS, plain.KV_MASK_FORM, plain.BIAS_EMPTY_ARRIVERS) == (0, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="no resident bias block fits"):
        c.make_cfg_d128(TemplateParams(dtype_qkv=2, cta_mma=2, bias_mode=1))
    with pytest.raises(ValueError, match="f16/bf16 prefill kernel only"):
        c.make_cfg_d32(TemplateParams(dtype_qkv=0, cta_mma=1, bias_mode=1))
    with pytest.raises(ValueError, match="dense, unsplit"):
        c.make_cfg_d32(TemplateParams(dtype_qkv=2, cta_mma=1, bias_mode=1, pack_gqa=True, qh_per_kh=2))
    with pytest.raises(ValueError, match="bias_mode must be 0 / 1"):  # 2 is the streamed ring, 3 its cluster-multicast form (their own modules); 4 is nothing
        c.make_cfg_d32(TemplateParams(dtype_qkv=2, cta_mma=1, bias_mode=4))
    # TILE_N as a Cfg field (M-2): 64 traces the one-chunk arm, 256 is refused by the TMEM predicate at two sub-tiles.
    cfg64, _ = c.make_cfg_d32(TemplateParams(dtype_qkv=2, cta_mma=1, tile_n=64))
    assert cfg64.TILE_N == 64 and cfg64.N_BMM2_CHUNKS == 1 and cfg64.BMM2_CHUNK_SIZE == 64
    with pytest.raises(ValueError, match="STATS"):
        c.make_cfg_d32(TemplateParams(dtype_qkv=2, cta_mma=1, tile_n=256))
    with pytest.raises(ValueError, match="TILE_N"):  # the head-dim validator's chunking predicate fires first: typed either way
        c.make_cfg_d32(TemplateParams(dtype_qkv=2, cta_mma=1, tile_n=48))


def test_template_params_and_capabilities_are_appended():
    """The three TemplateParams fields default to the pre-bias record (every existing module key renders byte-identically) and
    sit last; the four Capabilities fields sit last (the append-only contracts)."""
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    from cudnn.sdpa.fwd.engines import Capabilities

    p = TemplateParams()
    assert (p.bias_mode, p.kv_mask_form, p.tile_n) == (0, 0, 128)
    assert [f.name for f in dataclasses.fields(TemplateParams)][-10:] == [
        "bias_mode",
        "kv_mask_form",
        "tile_n",
        "n_chunk",
        "bias_stages",
        "padded_tail_form",  # the padded PLAIN body's tail form (the small tiles' pre-masked tail)
        "dead_subtile_quiet",  # the one-batch-entry item's dead-sub-tile quiet body
        "bias_max_form",  # the row-max form of the bias / kv_mask body (test_sdpa_bias_max_form_sm107.py)
        "softmax_wgs",  # the third softmax warp group of the TILE_N-64 pair item, default 2
        "s_slots",  # the double-buffered S of the small tiles at TILE_N 64 (default 1)
    ]  # appended in that order
    assert (p.padded_tail_form, p.dead_subtile_quiet, p.softmax_wgs, p.s_slots) == (
        0,
        0,
        2,
        1,
    ), "a record without the fields renders the legacy tail, the full body, the two-group body and one S slot"
    # ... followed by the per-shape TILE_N domain (the TILE_N 64 cells of the small tiles), appended after them.
    assert [f.name for f in dataclasses.fields(Capabilities)][-5:] == ["bias_d_shapes", "bias_io_dtype_only", "kv_mask", "kv_mask_dtypes", "tile_ns_by_d_shape"]


def _graph(b=2, h=4, s_q=256, s_kv=256, d=32, *, bias_dims=None, bias_dtype=None, kv_mask_dims=None, kv_mask_dtype=None):
    import cudnn

    io = cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)

    def t(dims, dt, name):
        stride = tuple(int(x) for x in torch.empty(dims).stride())
        return g.tensor(dim=list(dims), stride=list(stride), data_type=dt, name=name)

    kw = dict(name="sdpa", q=t((b, h, s_q, d), io, "q"), k=t((b, h, s_kv, d), io, "k"), v=t((b, h, s_kv, d), io, "v"), generate_stats=True, attn_scale=0.25)
    if bias_dims is not None:
        kw["bias"] = t(bias_dims, bias_dtype or io, "bias")
    if kv_mask_dims is not None:
        kw["kv_mask"] = t(kv_mask_dims, kv_mask_dtype or cudnn.data_type.BOOLEAN, "kv_mask")
    o, stats = g.sdpa(**kw)
    o.set_output(True).set_dim([b, h, s_q, d]).set_stride([h * s_q * d, d, h * d, 1])
    stats.set_output(True).set_dim([b, h, s_q, 1]).set_stride([h * s_q, s_q, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    return g


def test_analyzer_describes_the_bias_and_kv_mask_and_rejects_malformed_shapes():
    """The forward bias-shape validation the graph layer lacked: a bias whose rank is not 4 or whose score extent is not
    (S_q, S_kv) is ``invalid`` (malformed for every engine); the four broadcast forms are described as facts.  ``kv_mask`` is a
    python-only ``sdpa`` input (the node is python-engines-only once bound) described with its dims / dtype."""
    import cudnn
    from cudnn import _pygraph
    from cudnn.sdpa import graph_analyzer as ga

    assert _pygraph._CAPTURED_OPS["sdpa"]["python_only_in_kwargs"] == ("kv_mask",)
    for dims in ((1, 4, 256, 256), (2, 4, 256, 256), (1, 1, 256, 256), (2, 1, 256, 256)):
        f = ga.analyze(_graph(bias_dims=dims))
        assert f.invalid is None and f.has_bias and f.bias_dim == dims and f.bias_dtype == cudnn.data_type.BFLOAT16 and f.bias_stride[3] == 1, dims
    for dims in ((1, 4, 256, 128), (1, 4, 128, 256), (3, 4, 256, 256), (1, 3, 256, 256)):
        f = ga.analyze(_graph(bias_dims=dims))
        assert f.invalid is not None and "bias" in f.invalid, (dims, f.invalid)
    g = _graph(kv_mask_dims=(2, 1, 1, 256))
    assert g._unlowerable_node() is not None, "a bound kv_mask makes the sdpa node python-engines-only"
    f = ga.analyze(g)
    assert f.invalid is None and f.has_kv_mask and f.kv_mask_dim == (2, 1, 1, 256) and f.kv_mask_dtype == cudnn.data_type.BOOLEAN and f.kv_mask_stride[3] == 1
    f = ga.analyze(_graph(kv_mask_dims=(1, 1, 1, 256)))
    assert f.invalid is None and f.has_kv_mask and f.kv_mask_dim[0] == 1
    for dims in ((2, 1, 1, 128), (2, 4, 1, 256), (3, 1, 1, 256)):
        f = ga.analyze(_graph(kv_mask_dims=dims))
        assert f.invalid is not None and "kv_mask" in f.invalid, (dims, f.invalid)
    assert ga.analyze(_graph()).has_kv_mask is False


def test_kernel_module_constants_and_mask_sites():
    """The per-kernel constant ``BIAS_PRELOAD_CHUNKS`` is in its domain, the bias barriers ride ``make_classic_bars(bias_resident=,
    bias_ring_stages=)`` (the two bias FORMS fold on CFG.BIAS_MODE), and every mask site of the file calls ``apply_mask_chunk`` directly
    (the one bit-word form; no ``form=``)."""
    src = _KERNEL.read_text()
    assert (
        "BIAS_PRELOAD_CHUNKS: int = sm107_bias_preload_chunks(CFG.TILE_N)" in src
    ), "BIAS_PRELOAD_CHUNKS is the per-rendering constant (one arithmetic, rule 8b')"
    assert src.count("BIAS_PRELOAD_CHUNKS: int =") == 1
    from cudnn.sdpa.fwd import config_sm107 as cfg107

    # the table: nothing ahead of the wait on the 128-wide tile (+1.8..+3.2 % on the resident bias forms at S_kv >= 384, +0.0..+0.6 % on the
    # streamed ring; the confirm of the applied constant +2.1 % resident / +0.3..+0.5 % ring), one chunk on the 64-wide one (the resident
    # bias + kv_mask body -- the chooser's pick for a masked bias graph -- loses 2.5..2.7 % without it; the other 64-wide forms are mixed);
    # every value in the kernel's domain; an unrelated width keeps the one-chunk default
    assert cfg107.SM107_BIAS_PRELOAD_CHUNKS_BY_TILE_N == {128: 0, 64: 1}
    assert [cfg107.sm107_bias_preload_chunks(w) for w in (128, 64, 256)] == [0, 1, 1]
    assert all(cfg107.sm107_bias_preload_chunks(w) in (0, 1, 2) for w in (32, 64, 128, 256))
    # the bias barriers ride the factory's appended keywords; the multi-line call carries the S / P slot ring and the multicast ring's release count
    assert (
        "s_stages=CFG.TILES_Q * CFG.S_SLOTS,\n        bias_resident=_BIAS_RESIDENT,\n        bias_ring_stages=CFG.STAGES_BIAS if _BIAS_STREAMED else 0," in src
    )
    assert "form=" not in "".join(ln for ln in src.splitlines() if "apply_mask_chunk(" in ln)
    assert 'mask_value=float("-inf")' in src and "row_max_for_exp2(total_max)" in src
    assert "_kv_empty | (total_sum == cutlass.Float32(0.0))" in src, "the fully masked row's SELECT"
    assert "issue_bias_block_load(" in src and "load_bias_row_chunk(" in src and "kv_mask_keep_words(" in src
    # The exp2 MUFU / FMA split PROBE: a module constant that is OFF (the split loses on cc 10.7 at every head dim measured), folded at
    # trace time with the adapter's cc 10.0-only field, and every chunk exp2 site a trace-time ternary on it (off = the plain MUFU exp2,
    # the byte-identical rendering the md5 record above pins).
    assert re.search(r"^EXP2_FMA_SPLIT: bool = False$", src, re.M), "EXP2_FMA_SPLIT must stay False: the split is a measured loss on cc 10.7"
    assert "_E2E_ENABLED = bool(PARAMS.exp2_fma_split) or EXP2_FMA_SPLIT" in src
    sites = [ln for ln in src.splitlines() if "cute.math.exp2(chunk_S_" in ln]
    assert len(sites) == 4 and all("exp2_mixed(" in ln and "if _E2E_ENABLED else" in ln for ln in sites), sites


# --------------------------------------------------------------------------- tier 2: host rendering pins (sm_107a trace-compile)

_PROBE_BODY = """
d = params_kw.pop("__d")
params = TemplateParams(**params_kw)
mod = _load_sm100_kernel_module((d, d), params, rubin=True)
print("BIAS_MODE", mod.CFG.BIAS_MODE)
print("KV_MASK_FORM", mod.CFG.KV_MASK_FORM)
print("BIAS_RESIDENT_TILES", mod.CFG.BIAS_RESIDENT_TILES)
mod.compile(d_qk=d, d_v=d, has_lse=True, lse_kind="dense", has_bias=bool(mod.CFG.BIAS_MODE), has_kv_mask=bool(mod.CFG.KV_MASK_FORM))
"""
_COUNTS = {
    "USETMAXREG": ("USETMAXREG",),
    "STL": (" STL",),
    "LDL": (" LDL",),
    "FMNMX3": ("FMNMX3",),
    "R2P": (" R2P ",),
    "CGAERRBAR": ("CGAERRBAR",),
    "FFMA2": (" FFMA2",),
    "VOTE": ("VOTE",),
}
# The committed pre-bias record's arms (renderings/md5_fwd_d128_file_bias_off_sm_107a.txt) -> their TemplateParams.
_RECORD_ARMS = {
    "d32_dense_bf16": {"__d": 32, "cta_mma": 1, "dtype_qkv": 2},
    "d32_causal_f16": {"__d": 32, "cta_mma": 1, "dtype_qkv": 3, "window_right": 0},
    "d32_padded_bf16": {"__d": 32, "cta_mma": 1, "dtype_qkv": 2, "seq_kv_lens_present": True},
    "d64_dense_bf16": {"__d": 64, "cta_mma": 1, "dtype_qkv": 2},
    "d64_causal_f16": {"__d": 64, "cta_mma": 1, "dtype_qkv": 3, "window_right": 0},
    "d64_padded_bf16_cga2": {"__d": 64, "cta_mma": 2, "dtype_qkv": 2, "seq_kv_lens_present": True},
}


def _record():
    f = _RENDERINGS / "md5_fwd_d128_file_bias_off_sm_107a.txt"
    assert f.is_file(), f
    dsl, want = None, {}
    for ln in f.read_text().splitlines():
        if ln.startswith("dsl="):
            dsl = ln[len("dsl=") :].strip()
        m = re.match(r"fwd_d128_file sm_107a (\S+) rc=0 cubin_md5=([0-9a-f]{32})", ln)
        if m:
            want[m.group(1)] = m.group(2)
    return dsl, want


def _probe(tmp_path, params, tag):
    return run_sass_probe(tmp_path, probe_src=sass_probe_source(_PROBE_BODY, _COUNTS), arch="sm_107a", params=params, tag=tag)


def test_pre_bias_md5_record_is_committed_and_complete():
    dsl, want = _record()
    assert dsl and dsl.startswith("nvidia-cutlass-dsl "), dsl
    assert set(want) == set(_RECORD_ARMS), (sorted(want), sorted(_RECORD_ARMS))


@pytest.mark.parametrize("arm", list(_RECORD_ARMS))
def test_no_bias_renderings_are_byte_identical_to_the_pre_bias_record(tmp_path, arm):
    """With neither arm requested the kernel traces byte-identically to the tree BEFORE the resident block landed: every bias /
    kv_mask kernel argument is None-specialised and every seam sits under a const_expr on the Cfg fields (host trace-compile)."""
    from cudnn.frost.buffers import cutedsl_state

    dsl, want = _record()
    _installed, version = cutedsl_state()
    have = " ".join(version) if version else None
    if dsl is not None and have != dsl:
        pytest.skip(f"the md5 record was rendered with {dsl}; installed {have}: the cubin is a function of the DSL build, re-render the record")
    probe = _probe(tmp_path, _RECORD_ARMS[arm], f"bias_off_{arm}")
    assert probe.expect["BIAS_MODE"] == 0 and probe.expect["KV_MASK_FORM"] == 0
    assert probe.cubin_md5 == want[arm], f"{arm}: cubin md5 {probe.cubin_md5} != the pre-bias record's {want[arm]} -- the no-bias rendering changed"


@pytest.mark.parametrize(
    "d, cta, extra, base_arm",
    [
        (32, 1, {"bias_mode": 1, "kv_mask_form": 1, "seq_kv_lens_present": True}, "d32_padded_bf16"),
        (32, 1, {"bias_mode": 1}, "d32_dense_bf16"),
        (32, 1, {"kv_mask_form": 1}, "d32_dense_bf16"),
        (64, 2, {"bias_mode": 1, "kv_mask_form": 1, "seq_kv_lens_present": True}, "d64_padded_bf16_cga2"),
    ],
    ids=["d32_bias_kvmask_padded", "d32_bias_only", "d32_kvmask_only", "d64_bias_kvmask_cga2_padded"],
)
def test_bias_renderings_keep_the_split_the_mask_form_and_the_spill_count(tmp_path, d, cta, extra, base_arm):
    """Each bias / kv_mask rendering traced for sm_107a keeps the per-role register split (USETMAXREG > 0), no cluster-scope
    drain (CGAERRBAR 0), spills no more than the no-bias rendering of the same mask arm and width, carries the packed bias add
    (FFMA2) and -- wherever a mask is on -- the bit-word form (R2P > 0) with the software row max (FMNMX3 > 0); the kv_mask arm
    adds the warp ballots (VOTE above the base)."""
    base = _probe(tmp_path, _RECORD_ARMS[base_arm], f"base_{base_arm}")
    params = dict(_RECORD_ARMS[base_arm])
    params.update({"__d": d, "cta_mma": cta, "dtype_qkv": 2})
    params.update(extra)
    probe = _probe(tmp_path, params, f"bias_on_{d}_{cta}_{'_'.join(sorted(extra))}")
    assert probe.expect["BIAS_MODE"] == extra.get("bias_mode", 0) and probe.expect["KV_MASK_FORM"] == extra.get("kv_mask_form", 0)
    assert probe.cubin_md5 != base.cubin_md5
    assert probe.stats["USETMAXREG"] > 0 and probe.stats["CGAERRBAR"] == 0, probe.stats
    assert_no_new_spills(probe.stats, base.stats, tag=f"d{d} cga{cta} {sorted(extra)} vs {base_arm}: ")
    assert probe.stats["FMNMX3"] > 0, probe.stats
    if extra.get("bias_mode"):
        assert probe.stats["FFMA2"] >= 128, probe.stats
        assert probe.expect["BIAS_RESIDENT_TILES"] >= 1
    if extra.get("kv_mask_form") or extra.get("seq_kv_lens_present"):
        assert probe.stats["R2P"] > 0, probe.stats
    if extra.get("kv_mask_form"):
        assert probe.stats["VOTE"] > base.stats["VOTE"], (probe.stats, base.stats)


# --------------------------------------------------------------------------- tier 3: GPU cells (a Rubin device)

_DTYPES = [torch.float16, torch.bfloat16]
_DTYPE_IDS = ["fp16", "bf16"]
# The existing f16 row tolerances (test_sdpa_fwd_dsl_sm107.py / test_sdpa_fwd_head_dim_sm107.py): O against the fp64 oracle (an
# fp32 accumulation, one 16-bit rounding at the store), the LSE pair at 1e-4 / 1e-4 (the bias enters the fp32 score by ONE
# FFMA of bias * (1 / scale), then the scale multiply: one more fp32 rounding than the eager s * scale + bias -- pinned by the
# bias-magnitude cells, not widened for them).
_TOL_O = dict(atol=5e-2, rtol=3e-2)
_TOL_LSE = dict(atol=1e-4, rtol=1e-4)


def _bhsd(b, h, s, d, dtype):
    return torch.randn(b, s, h, d, device="cuda", dtype=dtype).transpose(1, 2)


def _ref(
    q,
    k,
    v,
    *,
    scale,
    bias=None,
    kv_mask=None,
    causal=False,
    bottom_right=False,
    seq_q_lens=None,
    seq_kv_lens=None,
    sinks=None,
    window_left=None,
    right_bound=None,
):
    """fp64 oracle of the whole contract: softmax_k(scale * q.k + bias + maskbias) with the key mask, the padding and the band;
    a row without a live key reads O = 0 / LSE = -inf (the empty-row contract); the bias enters in its 16-bit value."""
    b, h_q, s_q, _ = q.shape
    _, h_kv, s_kv, _ = v.shape
    dev = q.device
    g = h_q // h_kv
    k_ref = k.repeat_interleave(g, dim=1).double()
    v_ref = v.repeat_interleave(g, dim=1).double()
    scores = torch.matmul(q.double(), k_ref.transpose(-1, -2)) * scale
    if bias is not None:
        scores = scores + bias.double()  # broadcast over the dims declared 1
    q_lens = seq_q_lens.flatten().to(device=dev, dtype=torch.int64) if seq_q_lens is not None else torch.full((b,), s_q, dtype=torch.int64, device=dev)
    kv_lens = seq_kv_lens.flatten().to(device=dev, dtype=torch.int64) if seq_kv_lens is not None else torch.full((b,), s_kv, dtype=torch.int64, device=dev)
    i = torch.arange(s_q, device=dev).view(1, 1, s_q, 1)
    j = torch.arange(s_kv, device=dev).view(1, 1, 1, s_kv)
    diag = i + (kv_lens - q_lens).view(b, 1, 1, 1) if bottom_right else i
    dead_q = i >= q_lens.view(b, 1, 1, 1)
    masked = dead_q | (j >= kv_lens.view(b, 1, 1, 1))
    if kv_mask is not None:
        masked = masked | (~kv_mask.view(kv_mask.shape[0], 1, 1, s_kv))
    if causal:
        masked = masked | (j > diag)
    if right_bound is not None:
        masked = masked | (j > diag + right_bound)
    if window_left is not None:
        masked = masked | (j < diag - window_left)
    scores = scores.masked_fill(masked, float("-inf"))
    full = scores if sinks is None else torch.cat([scores, sinks.view(1, h_q, 1, 1).double().expand(b, h_q, s_q, 1)], dim=-1)
    smax = full.max(dim=-1).values
    ssum = torch.exp(full - smax.unsqueeze(-1)).sum(dim=-1).nan_to_num(0.0)
    lse = smax + torch.log(ssum)
    o = torch.matmul(torch.softmax(full, dim=-1).nan_to_num(0.0)[..., :s_kv], v_ref)
    dead = dead_q.squeeze(-1)
    return dict(o=o, lse=lse.masked_fill(dead, float("-inf")), empty=torch.isneginf(lse) | dead)


def _build(q, k, v, *, scale, bias=None, kv_mask=None, sdpa_kwargs=None, seq_len_kv=None, seq_len_q=None, sink=None, stats=True):
    """The graph with its variant pack (bias / kv_mask bound through ``tensor_like``: the bias as the classic ``bias`` operand,
    the mask as the python-only ``kv_mask`` input)."""
    import cudnn

    b, h_q, s_q, _ = q.shape
    d_v = v.shape[-1]
    dtype = q.dtype
    o_gpu = torch.empty(b, s_q, h_q, d_v, device="cuda", dtype=dtype).transpose(1, 2)
    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tq, tk, tv = g.tensor_like(q), g.tensor_like(k), g.tensor_like(v)
    kw = dict(name="sdpa", q=tq, k=tk, v=tv, generate_stats=stats, attn_scale=scale)
    vp = {tq: q, tk: k, tv: v}
    if bias is not None:
        tb = g.tensor_like(bias)
        kw["bias"] = tb
        vp[tb] = bias
    if kv_mask is not None:
        tm = g.tensor_like(kv_mask)
        kw["kv_mask"] = tm
        vp[tm] = kv_mask
    if seq_len_kv is not None:
        slk = g.tensor_like(seq_len_kv)
        kw.update(seq_len_kv=slk, use_padding_mask=True)
        vp[slk] = seq_len_kv
        slq_t = seq_len_q if seq_len_q is not None else torch.full((b, 1, 1, 1), s_q, dtype=torch.int32, device="cuda")
        slq = g.tensor_like(slq_t)
        kw["seq_len_q"] = slq
        vp[slq] = slq_t
    if sink is not None:
        st = g.tensor_like(sink)
        kw["sink_token"] = st
        vp[st] = sink
    kw.update(sdpa_kwargs or {})
    o, stats_t = g.sdpa(**kw)
    o.set_output(True).set_dim(o_gpu.shape).set_stride(o_gpu.stride())
    stats_gpu = None
    if stats:
        stats_gpu = make_dense_stats(b, h_q, s_q, "contiguous")
        stats_t.set_output(True).set_dim(stats_gpu.shape).set_stride(stats_gpu.stride()).set_data_type(cudnn.data_type.FLOAT)
        vp[stats_t] = stats_gpu
    g.validate()
    g.build_operation_graph()
    return g, vp, o, o_gpu, stats_gpu


def _frost_plans(g):
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    return names, [i for i, n in enumerate(names) if n == _ROW or n.startswith(_ROW + "[")]


def _run(q, k, v, *, scale, bias=None, kv_mask=None, sdpa_kwargs=None, seq_len_kv=None, seq_len_q=None, sink=None, stats=True, cga=None, tile_n=None):
    """Build the graph, pin the Rubin f16 row (optionally a cga and / or a KV tile width through the public autotune record;
    ``tile_n=None`` keeps the heuristics' pick), launch TWICE on sentinel-filled outputs (the two-launch delta) and return O,
    Stats and the plan's knobs (``knobs.tile_n`` = the width the executed plan carries)."""
    import cudnn

    g, vp, o, o_gpu, stats_gpu = _build(
        q, k, v, scale=scale, bias=bias, kv_mask=kv_mask, sdpa_kwargs=sdpa_kwargs, seq_len_kv=seq_len_kv, seq_len_q=seq_len_q, sink=sink, stats=stats
    )
    g.create_execution_plans([cudnn.heur_mode.A])
    names, ours = _frost_plans(g)
    idx = next((i for i in ours if g.plans[i].knobs.split_kv in (None, 1)), None)
    assert idx is not None, f"no unsplit {_ROW} plan in {names}"
    picked = g.plans[idx].knobs
    picked_cga, picked_tile_n = picked.cga, picked.tile_n
    want_cga = picked_cga if cga is None else cga
    want_tile_n = picked_tile_n if tile_n is None else tile_n
    if (want_cga, want_tile_n) != (picked_cga, picked_tile_n):
        engine_id, record = g.get_engine_and_knobs_at_index(idx)
        record = dict(record)
        if want_cga != picked_cga:
            record[cudnn.knob_type.TILE_CGA_M] = want_cga
        if want_tile_n != picked_tile_n:
            record[cudnn.knob_type.TILE_N] = want_tile_n
        g.create_execution_plan(engine_id, record)
        idx = g.get_execution_plan_count() - 1
        picked_cga, picked_tile_n = want_cga, want_tile_n
    g.select_plan(idx)
    g.check_support()
    g.build_plans()
    assert (g.plans[idx].knobs.cga, g.plans[idx].knobs.tile_n) == (picked_cga, picked_tile_n), (g.plans[idx].knobs, picked_cga, picked_tile_n)
    vp[o] = o_gpu
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)

    def launch():
        o_gpu.fill_(float("nan"))  # sentinel: an unwritten live row stays NaN
        if stats_gpu is not None:
            stats_gpu.fill_(float("nan"))
        g.execute(vp, ws)
        torch.cuda.synchronize()
        return o_gpu.clone(), (stats_gpu.clone() if stats_gpu is not None else None)

    o1, st1 = launch()
    o2, st2 = launch()
    assert torch.equal(o1, o2) and (st1 is None or torch.equal(st1, st2)), "two-launch delta on O / Stats"
    return dict(o=o_gpu, stats=stats_gpu, knobs=SimpleNamespace(cga=picked_cga, tile_n=picked_tile_n), names=names)


def _check(out, ref, *, has_stats=True):
    """Keyless rows EXACTLY 0 / -inf (never NaN, never a residue); live rows against the fp64 oracle."""
    empty = ref["empty"]
    o = out["o"]
    assert torch.isfinite(o).all(), "O has a non-finite (an unwritten sentinel or a NaN residue)"
    assert (o[empty] == 0).all(), "a keyless row's O must be exactly 0 (a SELECT, never a multiply by zero)"
    torch.testing.assert_close(o[~empty].double(), ref["o"][~empty], **_TOL_O)
    if has_stats:
        lse = out["stats"].squeeze(-1)
        assert torch.isneginf(lse[empty]).all(), "a keyless row's Stats must be exactly -inf"
        torch.testing.assert_close(lse[~empty].double(), ref["lse"][~empty], **_TOL_LSE)


# The degenerate-input matrix inside the resident budget (384 kv columns at the 32-wide tile): the KV ring sweep n_kv = 1, 2, 3
# (the budget edge) with B*H > 1, ONE CTA / one tile in the whole launch (S_kv 128 and the acceptance grid's 64), a tail tile (S_kv
# % 8 == 0 for a COMPACT bias TMA: 304, 376; the grid's 217 / 100 with a PADDED row stride 224 / 104 -- the compact form is a typed
# decline, pinned on a real graph below),
# the band masks (causal, right band, SWA + padding), GQA, a zero-length KV batch (an EMPTY work item: no bias barrier is touched),
# a zero-key batch WITH a sink (Stats = sink_logit, O = 0) and the padded-Q trim (bottom-right causal + bias is refused by the
# graph contract -- `_MASK_ONLY_CASES` carries the bottom-right keyless rows on the kv_mask arm alone); the bias in its four
# broadcast forms; the bias-MAGNITUDE cells (N(0, 1)
# x {1, 10} and the models' finite -1e9 key fold, which is NOT an empty row by contract); the 64-wide tile at cga1 (1 tile) and
# cga2 (2 tiles).  ``bias``: the form -- "h" = [1, H, S, S], "1" = [1, 1, S, S], "b" = [B, H, S, S], "b1" = [B, 1, S, S].
_CASES = {
    "dense_one_tile_bh1": dict(b=1, h=1, h_kv=1, s_q=128, s_kv=128),
    "dense_nkv1": dict(b=2, h=2, h_kv=2, s_q=128, s_kv=128),
    "dense_nkv2": dict(b=2, h=4, h_kv=4, s_q=256, s_kv=256),
    "dense_nkv3_384": dict(b=2, h=2, h_kv=2, s_q=384, s_kv=384),
    "dense_tail_304": dict(b=2, h=2, h_kv=2, s_q=304, s_kv=304),
    "causal_384": dict(b=1, h=4, h_kv=4, s_q=384, s_kv=384, causal=True),
    "causal_tail_376": dict(b=2, h=2, h_kv=2, s_q=376, s_kv=376, causal=True),
    "right_band_64_tail_304": dict(b=2, h=2, h_kv=2, s_q=304, s_kv=304, right_bound=64),
    "causal_swa128_padded_384": dict(b=2, h=2, h_kv=2, s_q=384, s_kv=384, causal=True, window_left=128, kv_lens=[384, 200]),
    "causal_gqa_8_2": dict(b=2, h=8, h_kv=2, s_q=256, s_kv=256, causal=True),
    "padded_kv_zero_batch": dict(b=3, h=4, h_kv=4, s_q=256, s_kv=256, kv_lens=[256, 0, 100]),
    "causal_sink_padded_q_trim": dict(b=3, h=8, h_kv=4, s_q=256, s_kv=256, causal=True, q_lens=[256, 129, 200], kv_lens=[200, 256, 0], sink=True),
    "bias_head_broadcast": dict(b=2, h=4, h_kv=4, s_q=256, s_kv=256, bias="1"),
    "bias_per_batch": dict(b=2, h=4, h_kv=4, s_q=256, s_kv=256, bias="b"),
    "bias_per_batch_head_broadcast": dict(b=2, h=4, h_kv=4, s_q=256, s_kv=256, bias="b1"),
    "bias_x10": dict(b=2, h=4, h_kv=4, s_q=256, s_kv=256, bias_scale=10.0),
    "bias_x10_causal_384": dict(b=1, h=4, h_kv=4, s_q=384, s_kv=384, causal=True, bias_scale=10.0),
    "bias_neg1e9_key_fold": dict(b=2, h=4, h_kv=4, s_q=256, s_kv=256, neg_fold=True),
    "d64_dense_128": dict(b=2, h=4, h_kv=4, s_q=128, s_kv=128, d=64),
    "d64_padded_zero_batch_128": dict(b=3, h=2, h_kv=2, s_q=128, s_kv=128, d=64, kv_lens=[128, 0, 77]),
    "d64_dense_256_cga2": dict(b=2, h=2, h_kv=2, s_q=256, s_kv=256, d=64, cga=2),
    "d64_causal_256_cga2": dict(b=2, h=2, h_kv=2, s_q=256, s_kv=256, d=64, cga=2, causal=True),
    "d24_dense_256": dict(b=2, h=4, h_kv=4, s_q=256, s_kv=256, d=24),
    "dense_one_tile_64": dict(b=2, h=2, h_kv=2, s_q=64, s_kv=64),
    "tail_217_padded_stride_bias": dict(b=2, h=2, h_kv=2, s_q=217, s_kv=217, bias_row_stride=224),
    "causal_tail_100_padded_stride_bias": dict(b=2, h=2, h_kv=2, s_q=100, s_kv=100, causal=True, bias_row_stride=104),
}


# The kv_mask arm ALONE: bottom-right causal + `bias` is not a graph (the FE validation shared with the backend refuses it),
# so the keyless bottom-right rows (q_lens > kv_lens: the rows above the diagonal have no key) + sink + padded-Q trim run
# with the per-key mask only.
_MASK_ONLY_CASES = {
    "bottom_right_keyless_sink": dict(b=3, h=8, h_kv=4, s_q=256, s_kv=256, bottom_right=True, q_lens=[256, 129, 0], kv_lens=[200, 256, 128], sink=True),
}

# The kv_mask ALONE above the BIAS budget: nothing is staged in SMEM, so the -inf masked body runs the ring-wrap axis the bias cells
# cannot reach (sdpa-invariants s9: n_kv = STAGES_KV = 4, one past it = 5, two wraps = 8 with and without a tail), at the tile's own
# width (cga1) on the 32- and the 64-wide tile.
_MASK_RING_CASES = {
    "dense_nkv4_512": dict(b=2, h=2, h_kv=2, s_q=512, s_kv=512),
    "causal_nkv5_640": dict(b=2, h=2, h_kv=2, s_q=640, s_kv=640, causal=True),
    "padded_nkv8_tail_900": dict(b=2, h=2, h_kv=2, s_q=900, s_kv=900, kv_lens=[900, 500]),
    "dense_nkv8_1024": dict(b=2, h=2, h_kv=2, s_q=1024, s_kv=1024),
    "d64_dense_nkv4_512": dict(b=2, h=2, h_kv=2, s_q=512, s_kv=512, d=64),
}


def _ref_of(c, q, k, v, kw):
    """The oracle under exactly the graph's conditions (sdpa-invariants s8): every flag of the case has its reference branch."""
    return _ref(
        q, k, v, scale=kw["scale"], bias=kw["bias"], kv_mask=kw["kv_mask"], causal=bool(c.get("causal") or c.get("bottom_right")),
        bottom_right=bool(c.get("bottom_right")), seq_q_lens=kw["seq_len_q"], seq_kv_lens=kw["seq_len_kv"],
        sinks=kw["sink"].flatten() if kw["sink"] is not None else None, window_left=c.get("window_left"), right_bound=c.get("right_bound"),
    )  # fmt: skip


def _case_inputs(case, dtype, *, seed=0, kv_mask_p=None, table=None):
    torch.manual_seed(seed)
    c = (_CASES if table is None else table)[case]
    d = c.get("d", 32)
    q = _bhsd(c["b"], c["h"], c["s_q"], d, dtype)
    k = _bhsd(c["b"], c["h_kv"], c["s_kv"], d, dtype)
    v = _bhsd(c["b"], c["h_kv"], c["s_kv"], d, dtype)
    form = c.get("bias", "h")
    bb = c["b"] if form in ("b", "b1") else 1
    bh = c["h"] if form in ("h", "b") else 1
    # A padded row stride (`bias_row_stride`: the TMA 16-byte rule at an S_kv % 8 != 0) makes the bias a strided VIEW; compact otherwise.
    row = c.get("bias_row_stride", c["s_kv"])
    bias = (torch.randn(bb, bh, c["s_q"], row, device="cuda", dtype=torch.float32) * c.get("bias_scale", 1.0)).to(dtype)[..., : c["s_kv"]]
    if c.get("neg_fold"):
        # the models' finite key fold: 0 on a live key, -1e9 on a masked one (rounded to the io dtype like the models' bias)
        keep = torch.rand(bb, 1, 1, c["s_kv"], device="cuda") > 0.3
        bias = torch.where(keep, torch.zeros_like(bias), torch.full_like(bias, -1e9))
    kv_mask = None
    if kv_mask_p is not None:
        kv_mask = torch.rand(c["b"], 1, 1, c["s_kv"], device="cuda") > kv_mask_p
    kv_lens = torch.tensor(c["kv_lens"], dtype=torch.int32, device="cuda").view(c["b"], 1, 1, 1) if "kv_lens" in c else None
    q_lens = torch.tensor(c["q_lens"], dtype=torch.int32, device="cuda").view(c["b"], 1, 1, 1) if "q_lens" in c else None
    sink = torch.randn(1, c["h"], 1, 1, dtype=torch.float32, device="cuda") if c.get("sink") else None
    causal = bool(c.get("causal") or c.get("bottom_right"))
    sdpa_kwargs = dict(use_causal_mask_bottom_right=True) if c.get("bottom_right") else (dict(use_causal_mask=True) if causal else {})
    window_left = c.get("window_left")
    if window_left is not None:
        sdpa_kwargs["sliding_window_length"] = window_left + 1  # the graph's length L keeps L keys up to the diagonal: W = L - 1
    right_bound = c.get("right_bound")
    if right_bound is not None:
        sdpa_kwargs["diagonal_band_right_bound"] = right_bound
    scale = 1.0 / math.sqrt(d) if not c.get("bottom_right") else 0.75
    run_kw = dict(scale=scale, bias=bias, kv_mask=kv_mask, sdpa_kwargs=sdpa_kwargs, seq_len_kv=kv_lens, seq_len_q=q_lens, sink=sink, cga=c.get("cga"))
    return q, k, v, run_kw, _ref_of(c, q, k, v, run_kw)


def _assert_plan(out, case):
    c = _CASES[case]
    assert out["knobs"].cga == c.get("cga", 1), (case, out["knobs"])


@requires_rubin
@pytest.mark.parametrize("case", list(_CASES))
def test_bias_matches_the_fp64_oracle(case):
    """The additive bias over the degenerate-input matrix inside the resident budget (bf16), every broadcast form and the
    magnitude cells: O and Stats against the fp64 oracle, keyless rows exactly 0 / -inf, two launches bitwise."""
    q, k, v, kw, ref = _case_inputs(case, torch.bfloat16)
    out = _run(q, k, v, **kw)
    _assert_plan(out, case)
    _check(out, ref)
    c = _CASES[case]
    if "kv_lens" in c and 0 in c["kv_lens"]:
        # The zero-length KV batch on the OUTPUT directly (sdpa-invariants s8): O = 0 on every row; Stats = -inf without a sink, the
        # sink logit (per head) on the live rows with one -- a keyless row with a sink holds the sink's mass alone -- and -inf on the
        # rows the padded-Q trim removes.
        empty_b = c["kv_lens"].index(0)
        st = out["stats"][empty_b].squeeze(-1)
        assert (out["o"][empty_b] == 0).all(), "a zero-length KV batch: O = 0 (a SELECT, never a multiply by zero)"
        if kw["sink"] is None:
            assert torch.isneginf(st).all(), "a zero-length KV batch without a sink: Stats = -inf"
        else:
            q_len = c["q_lens"][empty_b] if "q_lens" in c else c["s_q"]
            torch.testing.assert_close(st[:, :q_len], kw["sink"].flatten().to(st.dtype).view(-1, 1).expand(c["h"], q_len), **_TOL_LSE)
            assert torch.isneginf(st[:, q_len:]).all(), "the padded-Q trim: Stats = -inf at and past seq_len_q"


@requires_rubin
@pytest.mark.parametrize("case", ["dense_nkv2", "causal_swa128_padded_384", "causal_sink_padded_q_trim", "bias_x10", "d64_dense_256_cga2"])
def test_bias_matches_the_fp64_oracle_fp16(case):
    """The fp16 member of the row's dtype set (an fp16 bias with fp16 activations), five arms."""
    q, k, v, kw, ref = _case_inputs(case, torch.float16)
    out = _run(q, k, v, **kw)
    _assert_plan(out, case)
    _check(out, ref)


@requires_rubin
@pytest.mark.parametrize(
    "case", ["dense_nkv2", "dense_nkv3_384", "causal_384", "causal_swa128_padded_384", "padded_kv_zero_batch", "causal_gqa_8_2", "d64_dense_256_cga2"]
)
@pytest.mark.parametrize("with_bias", [False, True], ids=["mask_only", "mask_and_bias"])
def test_kv_mask_matches_the_fp64_oracle(case, with_bias):
    """The per-key boolean mask (30 % of the keys False) alone and composed with the bias, the band masks, the padding and a
    zero-length batch: a masked key is exactly a padded one in the oracle."""
    q, k, v, kw, ref = _case_inputs(case, torch.bfloat16, kv_mask_p=0.3)
    if not with_bias:
        kw["bias"] = None
        ref = _ref_of(_CASES[case], q, k, v, kw)
    out = _run(q, k, v, **kw)
    _assert_plan(out, case)
    _check(out, ref)


@requires_rubin
@pytest.mark.parametrize("case", list(_MASK_ONLY_CASES))
def test_kv_mask_composes_with_bottom_right_causal_keyless_rows_and_the_sink(case):
    """The per-key mask under the bottom-right band: the rows above the diagonal (no key) read Stats = sink_logit / O = 0,
    the trimmed rows -inf / 0, the live rows match the oracle with 30 % of the keys masked."""
    q, k, v, kw, _ = _case_inputs(case, torch.bfloat16, kv_mask_p=0.3, table=_MASK_ONLY_CASES)
    kw["bias"] = None
    out = _run(q, k, v, **kw)
    _check(out, _ref_of(_MASK_ONLY_CASES[case], q, k, v, kw))


@requires_rubin
@pytest.mark.parametrize("case", list(_MASK_RING_CASES))
def test_kv_mask_alone_serves_any_s_kv_across_the_ring_wrap(case):
    """The kv_mask ALONE has no resident budget (nothing staged in SMEM): S_kv 512 / 640 / 900 / 1024 = n_kv 4 (the ring depth), 5
    (one past it), 8 (two wraps; 900 = a tail + right padding) on the -inf masked body, at the tile's own width (cga1), 30 % of the
    keys False, against the fp64 oracle; two launches bitwise."""
    q, k, v, kw, _ = _case_inputs(case, torch.bfloat16, kv_mask_p=0.3, table=_MASK_RING_CASES)
    kw["bias"] = None
    out = _run(q, k, v, **kw)
    assert out["knobs"].cga == 1, (case, out["knobs"])
    _check(out, _ref_of(_MASK_RING_CASES[case], q, k, v, kw))


@requires_rubin
def test_bottom_right_causal_with_a_bias_is_refused_by_the_graph_contract():
    """ONE graph contract for the backend and FROST: bottom-right causal + bias is not a graph (the shared validation), so
    no engine -- the resident block included -- is asked; the case above runs the kv_mask arm alone."""
    import cudnn

    q, k, v, kw, _ = _case_inputs("bottom_right_keyless_sink", torch.bfloat16, table=_MASK_ONLY_CASES)
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="Bottom right causal mask is only supported with is_bias=False"):
        _build(
            q, k, v, scale=kw["scale"], bias=kw["bias"], sdpa_kwargs=kw["sdpa_kwargs"], seq_len_kv=kw["seq_len_kv"], seq_len_q=kw["seq_len_q"], sink=kw["sink"]
        )


@requires_rubin
def test_kv_mask_fully_masked_row_equals_the_zero_length_sequence():
    """A batch whose every key is False (the DATA mask the band geometry cannot see) reads O = 0 / Stats = -inf exactly -- the
    softmax ends the item with total_sum == 0 and the epilogue SELECTs -- and is BITWISE what seq_len_kv = 0 produces for it; a
    broadcast [1, 1, 1, S_kv] mask serves every batch."""
    torch.manual_seed(1)
    b, h, s, d = 3, 4, 256, 32
    q, k, v = _bhsd(b, h, s, d, torch.bfloat16), _bhsd(b, h, s, d, torch.bfloat16), _bhsd(b, h, s, d, torch.bfloat16)
    scale = 1.0 / math.sqrt(d)
    mask = torch.rand(b, 1, 1, s, device="cuda") > 0.3
    mask[1] = False  # batch 1: no live key at all
    out = _run(q, k, v, scale=scale, kv_mask=mask)
    ref = _ref(q, k, v, scale=scale, kv_mask=mask)
    _check(out, ref)
    assert (out["o"][1] == 0).all() and torch.isneginf(out["stats"][1]).all()
    kv_lens = torch.tensor([s, 0, s], dtype=torch.int32, device="cuda").view(b, 1, 1, 1)
    out_len = _run(q, k, v, scale=scale, kv_mask=mask, seq_len_kv=kv_lens)
    assert torch.equal(out_len["o"][1], out["o"][1]) and torch.equal(out_len["stats"][1], out["stats"][1])
    _check(out_len, _ref(q, k, v, scale=scale, kv_mask=mask, seq_kv_lens=kv_lens))
    one = torch.rand(1, 1, 1, s, device="cuda") > 0.5
    out_b = _run(q, k, v, scale=scale, kv_mask=one)
    _check(out_b, _ref(q, k, v, scale=scale, kv_mask=one))


@requires_rubin
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("case", ["dense_nkv2", "dense_nkv3_384", "causal_384", "padded_kv_zero_batch", "d64_dense_128"])
def test_identity_bias_zero_and_mask_all_true_is_bitwise_the_plain_kernel(case, dtype, monkeypatch):
    """bias = 0 and kv_mask = all True change no value: the biased rendering (the -inf masked body on every tile, the software
    row max, fma(0, 1/scale, s) == s, a keep word of all ones) is BITWISE the plain kernel on O and Stats -- as is each arm
    alone -- at the KV tile width the arm RUNS.  Every arm runs knob-less, so the heuristics' width table decides its rendering
    (``config_sm107.sm107_tile_n_pick``: the narrow tile for a bias WITH a mask on every case here, for a bias alone on the padding
    leg and for either arm alone at one 128-wide kv tile; 128 otherwise), and the plain twin is pinned to THAT width through the
    public record: the 64- and the 128-wide renderings reduce the row in a different number of steps, so the identity is a property
    of one rendering, never a cross-width claim (a plain 128 vs a biased 64 differ in the last bits of O and Stats).  And at ONE row-max
    form: the kv_mask-only plan's default is the hardware upper-bound max (``bias_max_form`` 1, test_sdpa_bias_max_form_sm107.py) whose
    m on a band-masked tile is the RAW max where the plain body takes the exact masked one -- equal within rounding, not bitwise -- so
    the three labels are pinned to the software tree through the sweep seam here; the dense-cell identity of the default plan is pinned
    in that module."""
    from cudnn.sdpa.fwd import config_sm107 as cfg107

    monkeypatch.setattr(cfg107, "_BIAS_MAX_FORM_OVERRIDE_FOR_SWEEPS", 0)
    q, k, v, kw, _ = _case_inputs(case, dtype)
    c = _CASES[case]
    plain_by_width = {}

    def plain_at(width):
        if width not in plain_by_width:
            plain_by_width[width] = _run(q, k, v, **{**kw, "bias": None, "kv_mask": None}, tile_n=width)
            assert plain_by_width[width]["knobs"].tile_n == width, plain_by_width[width]["knobs"]
        return plain_by_width[width]

    def table_pick(has_bias, has_kv_mask):
        return cfg107.sm107_tile_n_pick(c.get("d", 32), c["s_kv"], has_bias=has_bias, has_kv_mask=has_kv_mask, padded="kv_lens" in c)

    zero = torch.zeros(1, 1, c["s_q"], c["s_kv"], device="cuda", dtype=dtype)
    ones = torch.ones(c["b"], 1, 1, c["s_kv"], device="cuda", dtype=torch.bool)
    for label, b_arg, m_arg in (("bias=0 + mask=all-true", zero, ones), ("bias=0", zero, None), ("mask=all-true", None, ones)):
        out = _run(q, k, v, **{**kw, "bias": b_arg, "kv_mask": m_arg})
        width = out["knobs"].tile_n
        assert width == table_pick(b_arg is not None, m_arg is not None), (label, out["knobs"])
        plain = plain_at(width)
        assert torch.equal(out["o"], plain["o"]), f"{label} at TILE_N {width}: O differs from the plain kernel"
        assert torch.equal(out["stats"], plain["stats"]), f"{label} at TILE_N {width}: Stats differ from the plain kernel"


@requires_rubin
def test_declines_on_real_graphs():
    """The typed declines on REAL graphs: an fp32 bias and a COMPACT bias at an S_kv % 8 != 0 (217 -- the TMA 16-byte stride rule)
    leave the FROST row out of the plan list (the backend may serve them), while the SAME S_kv = 217 bias with a padded row stride (224)
    is served; a bias above the RESIDENT budget (S_kv 512 on one row) is served by the STREAMED ring (test_sdpa_bias_streamed_sm107.py
    pins its form); a kv_mask ALONE above the bias budget (S_kv 512) is served; a kv_mask the row cannot serve (int8) has no engine at all
    (the backend has no kv_mask field); a kv_mask WITH a bias at S_kv 512 is served (the ring) where it used to have no engine."""
    import cudnn

    torch.manual_seed(2)
    d = 32
    for s, bias_dtype in ((256, torch.float32), (217, torch.bfloat16)):
        q, k, v = (_bhsd(1, 2, s, d, torch.bfloat16) for _ in range(3))
        bias = torch.randn(1, 2, s, s, device="cuda").to(bias_dtype)
        g, *_ = _build(q, k, v, scale=1.0 / math.sqrt(d), bias=bias)
        try:
            g.create_execution_plans([cudnn.heur_mode.A])
            names, ours = _frost_plans(g)
        except cudnn.cudnnGraphNotSupportedError:
            names, ours = [], []
        assert not ours, (s, bias_dtype, names)
    q, k, v = (_bhsd(1, 2, 512, d, torch.bfloat16) for _ in range(3))
    g, *_ = _build(q, k, v, scale=1.0 / math.sqrt(d), bias=torch.randn(1, 2, 512, 512, device="cuda").to(torch.bfloat16))
    g.create_execution_plans([cudnn.heur_mode.A])
    names, ours = _frost_plans(g)
    assert ours, ("a one-row bias above the resident budget is served by the streamed ring", names)
    q, k, v = (_bhsd(1, 2, 217, d, torch.bfloat16) for _ in range(3))
    padded = torch.randn(1, 2, 217, 224, device="cuda").to(torch.bfloat16)[..., :217]
    g, *_ = _build(q, k, v, scale=1.0 / math.sqrt(d), bias=padded)
    g.create_execution_plans([cudnn.heur_mode.A])
    names, ours = _frost_plans(g)
    assert ours, ("a padded-stride bias at S_kv 217 is served", names)
    q, k, v = (_bhsd(1, 2, 512, d, torch.bfloat16) for _ in range(3))
    mask = torch.ones(1, 1, 1, 512, device="cuda", dtype=torch.bool)
    g, *_ = _build(q, k, v, scale=1.0 / math.sqrt(d), kv_mask=mask)
    g.create_execution_plans([cudnn.heur_mode.A])
    names, ours = _frost_plans(g)
    assert ours, ("a kv_mask alone above the bias budget is served", names)
    bias = torch.randn(1, 2, 512, 512, device="cuda").to(torch.bfloat16)
    g, *_ = _build(q, k, v, scale=1.0 / math.sqrt(d), bias=bias, kv_mask=mask)
    g.create_execution_plans([cudnn.heur_mode.A])
    names, ours = _frost_plans(g)
    assert ours, ("a kv_mask WITH a bias above the resident budget is served by the streamed ring", names)
    q, k, v = (_bhsd(1, 2, 256, d, torch.bfloat16) for _ in range(3))
    g, *_ = _build(q, k, v, scale=1.0 / math.sqrt(d), kv_mask=torch.ones(1, 1, 1, 256, device="cuda", dtype=torch.int8))
    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        g.create_execution_plans([cudnn.heur_mode.A])
        names, ours = _frost_plans(g)
        assert not ours, names
        raise cudnn.cudnnGraphNotSupportedError("no FROST plan for an int8 kv_mask and the backend has no kv_mask field")
