# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""FROST SDPA-forward capability declarations + spec table.

One engine per architecture x phase x geometry, named

    sdpa_fwd_<phase>_sm<arch>[_d<dqk>[x<dv>]]

(dtype is NOT part of the identity: a cell's engine serves every dtype its
kernel handles — fp16 and bf16 today — via ``Capabilities.dtypes``.)

The shared analyzer (``cudnn.sdpa.graph_analyzer.analyze``) parses the graph
once into :class:`SdpaGraphFacts`; each engine's probe is a cheap field-by-field
candidate match against its :class:`Capabilities` row below. Architecture
resource feasibility remains a lowering responsibility. Adding an engine is
one ``Capabilities``/spec row plus (usually) one kernel template.

New implementation rows identify one main body and its local variants. Their
``lower`` hook consumes an immutable selection and returns an executor through
the shared parser/binder/launch framework. A fixed helper chain (for example
THD descriptor setup, attention, and split combine) remains one engine.
Performance selection lives in heuristics. Unmigrated legacy rows still select
among multiple bodies internally; the explicit half D256, wide-half and FP8 D256/D512
rows are the migration boundary, not a claim that every row already follows it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Callable, Optional, Tuple

import cudnn

from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL
from cudnn.frost.buffers import CUTEDSL_MIN_VERSION, cutedsl_arch_requirement_error, cutedsl_state, cutedsl_too_old
from cudnn.sdpa import band
from .kernels.sm90.d64_f16_one_wg import contract as _sm90_d64_owner
from .kernels.sm90.f16_three_wg import contract as _sm90_three_wg_owner
from .kernels.sm90.d128_f16_multi_q import contract as _sm90_d128_owner
from cudnn.sdpa import graph_analyzer as ga
from cudnn.sdpa.fwd.config_sm80 import graph_tile_domains as _sm80_graph_tile_domains
from cudnn.sdpa.fwd.config_sm100 import (
    d256_mxfp8_pair_leg,
    supports_d64_fp8_swap_ab,
    supports_dense_fp8_swap_ab,
    SM100_SCORE_STATS_SHAPES,
    SM100_THD_PACK_GQA_SHAPES,
    SM100_PAGED_FP8_THD_SHAPES,
    d256_wide_decode_launch,
    decode_d256_q_tile,
    pack_gqa_supported,
    supports_paged_prefill_cga1,
    supports_paged_d256_pack_gqa,
    supports_nonpaged_d128_thd_prefill,
    supports_sm100_nonpaged_d128_thd_single_cta,
    supports_paged_d128_partial_pack_gqa,
    supports_nonpaged_d128_thd_pack_gqa,
    supports_thd_split,
    supports_scalar_kv_tail_split,
    supports_quantized_split_sink,
    supports_thd_split_sink,
)
from cudnn.sdpa.fwd.config_sm100 import (
    SM100_BIAS_D_SHAPES,
    SM100_DENSE_ONLY_D_SHAPES,
    SM100_NATIVE_D576_SHAPE,
    SM100_FP32_VECTOR_WITH_BIAS_TILE_N,
    SM100_SMALL_D_CGAS,
    SM100_TILE_NS_BY_D_SHAPE,
    sm100_bias_mode_pick,
    sm100_bias_resident_s_kv_max,
    sm100_tile_n_pick,
)
from cudnn.sdpa.fwd.config_d128_family import BIAS_MODE_RESIDENT, BIAS_MODE_STREAMED
from cudnn.sdpa.fwd import config_sm107 as _config_sm107
from cudnn.sdpa.fwd.kernels.sm107 import d512_half_contract as _sm107_wide_half
from cudnn.sdpa.fwd.kernels.sm100.d256_f16_contract import (
    LOOKAHEAD as _D256_LOOKAHEAD,
    SWAP_AB as _D256_SWAP_AB,
    IMPLEMENTATIONS as _D256_IMPLEMENTATIONS,
    in_pilot_domain as _in_d256_pilot_domain,
    mismatch as _d256_implementation_mismatch,
)
from .kernels.sm100.d512_f16_contract import (
    ROLE_SPLIT as _D512_ROLE_SPLIT,
    TWO_BY_TWO as _D512_TWO_BY_TWO,
    IMPLEMENTATIONS as _D512_IMPLEMENTATIONS,
    mismatch as _d512_implementation_mismatch,
)
from .kernels.sm100.d192_fp8_multi_q.contract import (
    NAME as _D192_FP8_MULTI_Q,
    mismatch as _d192_fp8_implementation_mismatch,
)
from .kernels.sm100.d256_fp8_single_q.contract import (
    NAME as _D256_FP8_SINGLE_Q,
    mismatch as _d256_fp8_implementation_mismatch,
)
from .kernels.sm100.d512_fp8_role_split.contract import (
    NAME as _D512_FP8_ROLE_SPLIT,
    mismatch as _d512_fp8_implementation_mismatch,
)
from .kernels.sm100 import d512_mxfp8_contract as _d512_mxfp8
from .kernels.sm100.d128_fp8_multi_q import contract as _small_fp8_multi_q
from .kernels.sm107.d512_fp8_role_split import contract as _sm107_d512_fp8_owner
from .kernels.sm107.d192_fp8_multi_q import contract as _sm107_d192_fp8_owner
from .kernels.sm107.d256_fp8_single_q import contract as _sm107_d256_fp8_owner
from .kernels.sm107.d128_fp8_multi_q import contract as _sm107_small_fp8_owner
from .kernels.sm107.d128_fp8_swap_ab import contract as _sm107_swap_fp8_owner
from .kernels.sm100.d128_fp8_dual_kv import contract as _small_fp8_dual_kv
from .kernels.sm100.d128_fp8_swap_ab import contract as _small_fp8_swap_ab
from .kernels.sm100.d192_f16_multi_q import contract as _d192_half_multi_q
from .kernels.sm100.d128_f16_single_q import contract as _half_single_q
from .kernels.sm100.d128_f16_multi_q import contract as _small_half_multi_q
from .kernels.sm100.d64_f16_quadrant import contract as _half_quadrant

_SMALL_FP8_CONTRACTS = {module.NAME: module for module in (_small_fp8_multi_q, _small_fp8_dual_kv, _small_fp8_swap_ab)}
_HALF_CONTRACTS = {module.NAME: module for module in (_d192_half_multi_q, _half_single_q, _small_half_multi_q, _half_quadrant)}

from .kernels.sm120.d512_f16_paired.contract import (
    DEVICE_CCS as _SM120_DEVICE_CCS,
    NAME as _SM120_D512_PAIRED,
    mismatch as _sm120_d512_paired_mismatch,
)
from .kernels.sm120.d512_fp8_paired.contract import (
    NAME as _SM120_D512_FP8_PAIRED,
    mismatch as _sm120_d512_fp8_paired_mismatch,
)
from .kernels.sm120.d512_mxfp8_paired.contract import (
    NAME as _SM120_D512_MXFP8_PAIRED,
    mismatch as _sm120_d512_mxfp8_paired_mismatch,
)
from .kernels.sm120.quantized_general_contract import (
    FP8 as _SM120_FP8_GENERAL,
    IMPLEMENTATIONS as _SM120_QUANT_GENERAL_IMPLEMENTATIONS,
    MXFP8 as _SM120_MXFP8_GENERAL,
    SHAPES as _SM120_QUANT_GENERAL_SHAPES,
    mismatch as _sm120_quant_general_mismatch,
)
from .kernels.sm80.half_flavor_contract import (
    IMPLEMENTATIONS as _SM80_HALF_IMPLEMENTATIONS,
    SHAPES as _SM80_HALF_SHAPES,
    mismatch as _sm80_half_mismatch,
)
from .kernels.sm120.half_exact_contract import (
    ALL_COMPUTE as _SM120_D256_ALL_COMPUTE,
    D256 as _SM120_D256,
    GENERAL as _SM120_F16_GENERAL,
    IMPLEMENTATIONS as _SM120_HALF_EXACT_IMPLEMENTATIONS,
    KEY_GROUP as _SM120_DECODE_KEY_GROUP,
    SHAPES as _SM120_HALF_EXACT_SHAPES,
    explicit_geometry as _sm120_half_exact_explicit_geometry,
    mismatch as _sm120_half_exact_mismatch,
)

from cudnn.sdpa.fwd.config_sm107 import (
    KV_MASK_ADDITIVE_GRANULE_ELEMS,
    SM107_BIAS_D_SHAPES,
    SM107_EPILOGUE_GATE_SHAPES,
    SM107_F16_SMALL_D_SHAPES,
    SM107_F16_THD_SHAPES,
    SM107_FP8_THD_SHAPES,
    SM107_FP8_SPLIT_SHAPES,
    SM107_MXFP8_SPLIT_SHAPES,
    supports_fp8_padded_split,
    SM107_FP8_PACK_GQA_SHAPES,
    SM107_MXFP8_THD_SHAPES,
    SM107_SCORE_STATS_SHAPES,
    SM107_SMALL_D_CGAS,
    SM107_FP32_VECTOR_WITH_BIAS_TILE_N,
    SM107_TILE_NS_BY_D_SHAPE,
    SM107_FP8_TILE_NS_BY_D_SHAPE,
    supports_native_d128_narrow_tile,
    supports_native_d128_single_q,
)
from cudnn.sdpa.fwd.config_sm120 import (
    D512_FLAVOR,
    WIDE_F16_FLAVORS,
    GENERAL_HEAD_TILE_MAX as _SM120_GENERAL_HEAD_TILE_MAX,
    HEAD_TILE_GRANULE as _SM120_HEAD_TILE_GRANULE,
    SEQ_Q_TILES as _SM120_SEQ_Q_TILES,
    PAGED_D_SHAPES as _SM120_PAGED_D_SHAPES,
    paged_head_dims_supported as _sm120_paged_head_dims_supported,
    decode_kv_stage as _sm120_decode_kv_stage,
    head_tiles as _sm120_head_tiles,
    is_decode_tile as _sm120_is_decode_tile,
    tile_domain as _sm120_tile_domain,
    supports_mxfp8_split_sink,
)

# The DSL adapters (api_dsl) and cuda.bindings are LOWERING dependencies, not
# support-check ones: importing them here would drag the CuTe DSL (~1.0 s, 357
# modules) into every process that merely asks whether an engine could serve a
# graph. ENGINE_SPECS therefore names its adapter, and _adapter() resolves it
# at build time. The SM107 target gate lazily probes the DSL once; other
# capability checks stay import-free.
_SM100 = "SdpaFwdDslSm100"
_SM120 = "SdpaFwdDslSm120"
_SM80 = "SdpaFwdDslSm80"
_SM90 = "SdpaFwdDslSm90"


def _adapter(name: str):
    from cudnn.sdpa.fwd import api_dsl

    return getattr(api_dsl, name)


def _cuda_driver():
    from cuda.bindings import driver

    return driver


_LOG = logging.getLogger(__name__)

# Arch ranges the spec rows below serve, inclusive, encoded major*10 + minor as
# in engines/manifest.py. RANGES, not the exact device families that exist
# today: an sm100 kernel runs on the whole sm100 line, so enumerating members
# silently declines the parts that ship later -- Rubin (sm107) and Thor (sm110)
# are meant to reuse these kernels and an exact {(10,0), (10,3)} excluded both.
# Within the range, cc10.3+ additionally enables the fused LDTM.STAT row-max for
# MXFP8 — handled in the lowering, from the device capability.
_BLACKWELL = (100, 119)
_BLACKWELL_GEFORCE = (120, 129)


@dataclass(frozen=True)
class SdpaFwdKnobs:
    """Per-plan tuning request for the SDPA-forward engines.

    Typed fields internally; ``None`` means "no preference". Travels as
    ``PlanConfig.knobs``; each engine's :class:`Capabilities` row advertises the
    domain it honors, and the probe rejects the engine for any request outside
    that domain (a knob is honored or the engine is ineligible — never silently
    degraded).

    At the public surface every tuning field is one ``cudnn.knob_type`` of the
    shared vocabulary (``to_public`` / ``from_public``), the same dict shape a
    backend plan reports: ``tile_m/tile_n`` and ``cga`` reuse the backend's
    ``TILE_M`` / ``TILE_N`` / ``TILE_CGA_M`` (a cga of 2 IS a 2-CTA M-cluster,
    the backend's own encoding of its 2-CTA SDPA variant); ``sched_policy``,
    ``pack_gqa`` and ``split_kv`` are frontend-only types (the backend's
    split-KV counterpart, ``STREAM_K``, is an on/off mode, not a chunk count,
    hence the distinct ``SPLIT_KV``).

    Knobs are performance-only: every value computes the same function, so an
    autotuner may pick any of them. The softmax accumulator precision (the
    cc 10.7 f16x2 exponent arm) changes numerics and is therefore NOT a knob: it
    is the ``sdpa(..., softmax_precision=)`` op attribute, read from the graph
    into ``SdpaGraphFacts.softmax_precision`` and gated by each row's
    ``Capabilities.softmax_precisions`` in :func:`mismatch`.
    """

    sched_policy: Optional[int] = None  # tile-scheduler policy (SCHED_NATURAL, ...)
    tile_m: Optional[int] = None  # Q sequence tile width
    tile_n: Optional[int] = None  # KV sequence tile width
    cga: Optional[int] = None  # physical cluster M width; legacy wide rows retain their historical pair encoding
    pack_gqa: Optional[bool] = None  # Head packing for GQA/MQA
    # KV-split count: each Q tile's KV range cut into this many chunks, each
    # run by its own CTA, recombined by the split_combine pass. 1 = off.
    split_kv: Optional[int] = None
    # Warp-specialization topology: 0 = existing topology, 1 = one full Q tile.
    warp_spec_cfg: Optional[int] = None
    # Explicit MMA operand orientation; unset preserves the existing lowering.
    swap_ab: Optional[int] = None

    # field name -> cudnn.knob_type member name (resolved lazily: the compiled
    # module is not importable at class-definition time in every build).
    _PUBLIC_KNOBS = (
        ("sched_policy", "SCHED_POLICY"),
        ("tile_m", "TILE_M"),
        ("tile_n", "TILE_N"),
        ("cga", "TILE_CGA_M"),
        ("pack_gqa", "PACK_GQA"),
        ("split_kv", "SPLIT_KV"),
        ("warp_spec_cfg", "WARP_SPEC_CFG"),
        ("swap_ab", "SWAP_AB"),
    )

    def to_public(self) -> dict:
        """``{cudnn.knob_type: int}`` for every tuning field that is set (bools as 0/1)."""
        kt = cudnn.knob_type
        out = {}
        for field, member in self._PUBLIC_KNOBS:
            value = getattr(self, field)
            if value is not None:
                out[getattr(kt, member)] = int(value)
        return out

    @classmethod
    def from_public(cls, public: dict) -> "SdpaFwdKnobs":
        """Inverse of :meth:`to_public`. Rejects knob types this operation has no field for."""
        kt = cudnn.knob_type
        by_type = {getattr(kt, member): field for field, member in cls._PUBLIC_KNOBS}
        kwargs = {}
        for knob, value in public.items():
            knob = kt(int(knob)) if not isinstance(knob, kt) else knob
            field = by_type.get(knob)
            if field is None:
                raise ValueError(f"knob {knob.name} is not a tuning axis of the SDPA-forward engines")
            if isinstance(value, bool):
                if field != "pack_gqa":
                    raise ValueError(f"knob {knob.name} value must be an int, got {value!r}")
                value = int(value)
            if not isinstance(value, int):
                # A persisted record carries ints; "0" or 1.0 would round-trip to
                # a different native knob than the one recorded.
                raise ValueError(f"knob {knob.name} value must be an int, got {value!r}")
            if field == "pack_gqa" and value not in (0, 1):
                raise ValueError(f"knob PACK_GQA must be 0 or 1, got {value}")
            kwargs[field] = bool(value) if field == "pack_gqa" else value
        return cls(**kwargs)


@dataclass(frozen=True)
class Capabilities:
    """What one ENGINE can serve — the envelope of graphs (and, later, requested
    tuning knobs) its lowering can honor. An engine spanning several kernels
    declares the union its lowering can actually deliver. Compared
    field-by-field against SdpaGraphFacts in the probe."""

    # Arch RANGE, inclusive, encoded major*10 + minor as in engines/manifest.py.
    # A range and not a set of exact device families: an sm100 kernel runs on
    # everything in the sm100 line, so enumerating the parts that exist today
    # silently declines the ones that ship tomorrow. Rubin (sm107) and Thor
    # (sm110) are meant to reuse these kernels and an exact set excluded both.
    sm_lo: int
    sm_hi: int
    phase: str
    # Head-dim DOMAIN. One engine per arch x dtype family: the head dim is a
    # LOWERING concern — the adapter picks the smallest kernel flavor whose
    # native shape covers the graph (api_dsl._pick_flavor) — not an engine
    # identity. ``d_shapes`` is the set of NATIVE flavor shapes (d_qk, d_v)
    # that lowering picks among.
    d_shapes: frozenset
    # Envelope + alignment rule. When > 0: any graph (d_qk, d_v) componentwise
    # <= some native shape AND a multiple of this is served via TMA
    # zero-padding — the kernel's descriptors carry the ACTUAL extents, so
    # padded contraction columns load as exact zeros (S/softmax unchanged) and
    # O stores past d_v are OOB-clipped. 8 = the TMA 16-byte global-stride
    # rule at 2 bytes/elem (f16/bf16), 16 = the same rule at 1 byte/elem
    # (per-tensor FP8), 1 = no constraint (the SM80 lowering pads host-side).
    # 0 = NO envelope: exact native shapes only (MXFP8, whose SF plumbing is
    # not audited for zero-padding).
    d_pad_multiple: int = 8
    # Shapes whose kernels carry the THD leg. None = THD (when the ``thd``
    # capability is set) serves the same head-dim domain as dense — the f16
    # THD compile key carries the head dims, so THD rides the envelope. A set
    # = THD graphs must match one of these NATIVE shapes exactly: the
    # quantized rows' packed THD compile key carries no head-dim entries
    # (native-tile contract), so their envelope is dense-only.
    thd_d_shapes: Optional[frozenset] = None
    # Per-native-shape ENVELOPE FLOOR: a tuple of ``((d_qk, d_v), min_dim)``
    # pairs (a tuple, not a dict, so the row stays hashable). A shape with a
    # floor serves graphs whose head dims are in ``(min_dim, shape]`` rather
    # than ``(0, shape]``, so a kernel whose geometry is tuned for a large head
    # dim does not silently swallow a much smaller graph at a multiple-x
    # zero-padding cost. The floor applies to BOTH head dims and only to
    # ENVELOPE matches — an exact ``d_shapes`` hit always passes. Example: the
    # d512 FP8 flavor floors at 256, so it serves (256, 512] on both dims
    # (at most 2x padding) and a d256 graph is declined rather than routed
    # onto it. ``()`` = no floors.
    d_envelope_floors: tuple = ()
    dtypes: frozenset = frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16})  # cudnn.data_type, see graph_analyzer
    is_mxfp8: bool = False  # block-scale MXFP8 engine (FP8 in + per-32-block E8M0 SF)
    is_fp8: bool = False  # per-tensor FP8 engine (FP8 in + scalar descales)
    # O dtype domain, declared only by the quantized rows: elsewhere O must
    # equal Q, which facts.uniform_dtype already enforces.
    out_dtypes: frozenset = frozenset()
    # Block-scaled O domain (facts.o_block_scale): 0 = plain O; 16 = FP4_E2M1 O
    # + E4M3 scale per 16 d in ``sf_o``; 32 = FP8_E4M3 O + UE8M0 scale per 32.
    o_block_scales: frozenset = frozenset({0})

    # optional features a graph may request
    bias: bool = False
    dropout: bool = False
    score_mod: bool = False
    paged_kv: bool = False
    alibi: bool = False
    block_mask: bool = False
    rng_dump: bool = False
    score_max: bool = False  # per-row/tile score-max side output
    score_sum_exp: bool = False  # per-row/tile sum-of-exp side output
    dynamic_scale: bool = False
    unfuse_fma: bool = False
    # Stats written as (max + ln(sum_exp)) * log2(e) (sdpa(stats_use_log2=True)): the
    # kernel epilogue (or the split-KV combine) scales the LSE by log2(e).
    stats_log2: bool = False
    seq_q_trim: bool = False
    right_band_widening: bool = False

    # The band axes a graph may request, as the PRE-MODEL declaration spelling:
    # causal (right bound 0), bottom-right anchor, sliding window (left bound),
    # right-band widening (right bound > 0).  They remain the spelling every
    # long-standing row uses and are mapped ONCE, in __post_init__ below, onto
    # ``band`` -- the canonical set this probe decides with.  A row that needs a
    # RESTRICTED claim (e.g. an unmasked-only future SM89 row) declares ``band``
    # directly and leaves these False; declaring both is allowed only while they
    # agree exactly, and disagreeing raises rather than one silently winning.
    causal: bool = False
    bottom_right: bool = False
    swa: bool = False
    padded: bool = False
    sink: bool = False
    # Optional dtype subset for sink support. None means every dtype served by
    # the engine; a subset lets one exact-shape flavor decline an unsupported
    # low-precision sink path without affecting its non-sink coverage.
    sink_dtypes: Optional[frozenset] = None
    stats: bool = False
    # The adapter accepts lse_tensor=None, so a stats-less graph needs no
    # dummy-LSE workspace chunk. Every row sets True. The SM100/SM120 kernels
    # None-specialize the LSE store (compiled out entirely); the SM80 kernels
    # still compute LSE into a kernel-internal buffer when unbound — a
    # pre-existing allocation on that path, tracked with its other
    # TemplateParams-conversion follow-ups.
    lse_optional: bool = False
    # Most THD forward lowerings derive addresses from compact length prefixes.
    # The SM100/SM103 ungated, unsplit CGA2 half D128/D256/D576 graph lowering
    # reads independent Q/O/Stats origins and nonpaged K/V origins; the SM90
    # D512 graph lowering reads independent Q/K/V/O/Stats origins. See
    # SUPPORT_MATRIX_TRACKER.md for this scope and the remaining sibling gaps;
    # thd=True alone does not promise support for padding between sequences.
    thd: bool = False
    # cu_seq_len_q / cu_seq_len_kv (B+1,) prefix sums (cuDNN 9.24+). Serving
    # rows consume the form on THD host-side (lens = adjacent differences of
    # the inherent tolist); dense cu graphs stay declined until the kernels
    # grow a CU read mode (len = cu[b+1] - cu[b]) — see mismatch().
    cu_seq_len: bool = False
    # Dense padded + stats: the per-batch seq_len_q LSE trim (padded q-rows
    # write LSE=-inf / O=0, cuDNN >= 9.14). Every kernel carries the dense
    # padded-Q trim itself (a graph with per-batch Q lengths compiles the
    # SEQ_Q_LENS_PRESENT specialization), so there is no separate capability.
    padded_stats: bool = False
    # Whether the row serves s_q == 1 (decode-shaped) graphs.
    decode: bool = True

    # Escape hatch for kernels whose persistent multi-wave rescheduling is
    # numerically invalid: eligible only when the whole grid fits in one wave
    # (B*H_q*ceil(S_q/512) <= SM_count / CTA_MMA). NO CURRENT ROW SETS THIS.
    # Historical user: the fp8 d128 row, whose multi-wave wrong-O bug (14-19%
    # mismatched elements, some tiles left unwritten NaN) was root-caused to
    # the classic-pipeline TMEM stats race — the next tile's prologue BMM1
    # overwrote the S_acc-head (total_max, total_sum) before the correction
    # epilogue read them — and fixed by the mb_stats_read barrier (same fix as
    # the f16 kernel's; see sm100/prefill_d128_fp8.py / _common_blackwell.Bars).
    single_wave_only: bool = False
    # Serve ragged S_kv on unmasked graphs through the kernel's padded mask
    # compiled against the scalar S_kv (TemplateParams.kv_tail_mask, no
    # per-batch lengths buffer; #1425). Only rows that opt in use it; the
    # KV-tail rule is waived.
    skv_tail_via_padding: bool = False

    # Dense layout envelope this engine accepts:
    #   "bshd"       — Q/K/V/O must be BSHD-physical (stride order 3,1,2,0).
    #   "dense_flex" — any B/H/S stride permutation, padded (oversized)
    #                  strides included, as long as the head dim is
    #                  innermost-contiguous (stride 1) and the strides are
    #                  non-broadcast / non-overlapping (facts.dense_layout;
    #                  see graph_analyzer.dense_layout_ok). The DSL executor
    #                  normalizes such tensors to the kernel's canonical
    #                  BSHD-compact buffers (zero-copy when already BSHD).
    # THD (ragged) graphs always require the BSHD-order packing regardless —
    # the ragged lowering rebuilds packed [1,T,H,D] views and only that
    # packing is defined for it.
    layouts: frozenset[str] = frozenset({"bshd"})
    skv_tile: int = 128  # KV tail rule (waived when padded / causal covers the tail)

    # Tuning-knob domains this engine's lowering honors (see SdpaFwdKnobs).
    sched_policies: frozenset[int] = frozenset({SCHED_NATURAL})
    tile_ms: frozenset[int] = frozenset()
    tile_ns: frozenset[int] = frozenset()
    cgas: frozenset[int] = frozenset()
    # Shape-specific CGA domains for rows that lower several native flavors.
    # ``cgas`` remains the default. A split-specific entry further narrows the
    # domain for split-KV plans without changing the unsplit public domain.
    # Per-d-shape scheduler domain, mirroring cgas_by_d_shape. Needed because
    # sched_policies is row-wide while LPT support is per-KERNEL: the SM107 port
    # dropped `lpt_q_tiles_in_cga_units=True` on most flavors, so a row-wide
    # claim would advertise LPT for kernels that write nothing under it.
    # An entry here OVERRIDES sched_policies for that flavor.
    sched_policies_by_d_shape: tuple[tuple[tuple[int, int], frozenset[int]], ...] = ()
    cgas_by_d_shape: tuple[tuple[tuple[int, int], frozenset[int]], ...] = ()
    split_cgas_by_d_shape: tuple[tuple[tuple[int, int], frozenset[int]], ...] = ()
    pack_gqas: frozenset[bool] = frozenset({False})
    # Does this row's lowering wire the KV-split path (kernel SplitHelpers +
    # adapter carving the partial slabs + launching the combine)? A GATE, not a
    # domain: WHICH splits are worth trying is a device-derived search space
    # (heuristics.split_kv_candidates), not a per-row constant. Fail-closed —
    # a row that accepts split_kv > 1 without the plumbing leaves untouched
    # partial slots at lse_partial = 0, which corrupt the combine's
    # log-sum-exp rather than raising.
    split_kv_supported: bool = False
    # Shapes whose kernel flavors wire SplitHelpers. None = every flavor in
    # d_shapes does (f16/SM120). A set = split_kv > 1 is honored only when
    # the graph's dims are covered by a member (the quantized families wire
    # the split path in the d128 flavor only).
    split_d_shapes: Optional[frozenset] = None
    # Softmax-precision domain (cudnn.data_type values). Empty = unserved.
    # Arch-dependent membership (the f16x2 exponent arm exists only in the
    # cc 10.7 quantized kernels) is expressed by SPLITTING the row per arch line —
    # each row declares exactly what its own lowering carries — not by a
    # knob x arch notch here.
    softmax_precisions: frozenset[int] = frozenset()
    # PackGQA, like split-KV, can be wired per FLAVOR rather than per row: the
    # SM107 line ships it only in the d128 per-tensor FP8 kernel, while the
    # ported d256/d512 siblings skip it.  None = every shape the row serves.
    # APPENDED at the end deliberately: Capabilities evolves append-only, so a
    # positional construction of an older field never silently rebinds.
    pack_gqa_d_shapes: Optional[frozenset] = None
    # THD graphs whose Stats has NO ragged offsets (per-batch padded (b, s_max, h)
    # rows, FlashInfer's form): the kernel stores per batch and the adapter
    # fills the tail rows with -inf. Rows whose kernels lack the per-batch THD
    # store keep False and decline the form (the backend serves it).
    thd_padded_stats: bool = False
    # Fused epilogue gate: the graph tail ``sdpa(virtual O_v) -> sigmoid(G) ->
    # mul(O_v, s)`` (facts.has_epilogue_gate, cudnn._sdpa_tail) lowered as the
    # kernel's ``O := O * sigmoid(G)`` epilogue (TemplateParams.epilogue_gate;
    # the Rubin d256 f16/bf16 and per-tensor FP8 kernels).  A FEATURE, not a
    # knob -- the graph asks for it, so it follows the thd_d_shapes /
    # split_d_shapes / pack_gqa_d_shapes idiom: ``epilogue_gate_d_shapes`` is
    # the set of NATIVE shapes whose kernel carries the gate, matched on the
    # graph's EXACT (d_qk, d_v) -- never through the zero-padding envelope
    # (the gate tile is TILE_O wide and the padded columns would gate garbage).
    # None = every shape in d_shapes (no row says that today).  The standalone
    # adapter's check_support carries the twin of this claim (rule 8b) through
    # the same constant (config_sm107.SM107_EPILOGUE_GATE_SHAPES, rule 8b').
    epilogue_gate: bool = False
    epilogue_gate_d_shapes: Optional[frozenset] = None
    # dtype domain of G.  None = G must equal Q's dtype (the half kernels stage
    # the gate in Q's storage dtype); the quantized rows name the half dtype
    # their kernel stages it in (bf16 on the Rubin FP8 d256 kernel).
    epilogue_gate_dtypes: Optional[frozenset] = None
    # Flavors whose kernel packs a PROPER DIVISOR of the GQA group when the
    # group does not divide tile_m (partial PackGQA: 96/8 packs 4 of its 12
    # heads per token row-group, 48/8 packs 2; config_sm100.pack_gqa_group_size).
    # None = every flavor keeps the full-ratio contract (the group must divide
    # tile_m, or PACK_GQA=1 is declined; SM120 packs any group up to tile_m:
    # pack_gqa_requires_divisible).  Native on the SM100 d128 / d256 f16
    # kernels and the native SM107 dense d512 body; the other d192x128 /
    # d512 kernels pack HEADS_PER_TILE = G.
    # APPENDED after the epilogue-gate fields (append-only contract above):
    # test_capabilities_positional_prefix_is_append_only pins the tail.
    pack_gqa_partial_d_shapes: Optional[frozenset] = None
    # Shapes whose kernel flavors wire the PAGED_KV specialization (block-table
    # indirection on the K/V TMA loads). None = every flavor in d_shapes. A set
    # = a paged graph is served only when the flavor the lowering SELECTS
    # (_selected_d_shape, the smallest covering envelope) is a member — the
    # test is on the selection, not the raw dims, so mixed head dims that ride a
    # wired envelope ((256, 128) -> d256) are served and an unwired selection
    # ((512, 128) -> d512) is declined. Kernel-body gates mirror it
    # (config_sm100._PAGED_KV_FLAVORS is the backstop).  APPENDED after
    # pack_gqa_partial_d_shapes (append-only contract above; the same test
    # pins it).
    paged_d_shapes: Optional[frozenset] = None
    # Native THD prefill flavors whose worklist and Stats index packed query
    # heads. Empty is fail-closed; the separate ragged-Q decode leg is unchanged.
    # Appended to preserve positional construction of existing capabilities.
    thd_pack_gqa_d_shapes: frozenset[tuple[int, int]] = frozenset()
    # attn_scale = 0. The SM100/SM107/SM120 kernels fold the scale into exp2 after an unscaled, -inf-masked
    # running max, which a zero scale turns into NaN (#1435); SM80 and SM90 specialize on the scale's sign. Appended last.
    zero_scale: bool = False
    # Shapes whose kernel flavors carry the pre-folded-scale arm (the op attribute
    # sdpa(attn_scale_prefolded=True): Q carries attn_scale * log2(e), the kernel
    # traces no per-score scale).  None = unserved (the default: a row opts in).
    # Matched on the flavor the lowering SELECTS (_selected_d_shape), like
    # paged_d_shapes.  The paged-KV bodies and the single-CTA half THD legs
    # (packed split / D192 single-Q) run bodies without the arm and are declined
    # by rule in mismatch().  APPENDED after zero_scale (append-only
    # contract above; the same test pins it).
    attn_scale_prefolded_d_shapes: Optional[frozenset] = None
    # Score side outputs -- the graph's optional ``score_max`` / ``score_sum_exp``
    # fp32 (B, H_q, S_q, 1) ports (``score_max`` / ``score_sum_exp`` above are the
    # row-level flags): per Q row the softmax max in NATURAL units (scale * max_j
    # s_j, the sink folded in as one more column) and the fp32 row-sum of
    # exp(s - max), the two operands the LSE is built from, so Stats ==
    # score_max + ln(score_sum_exp) to fp32 rounding; a keyless row reads -inf / 0
    # (with a sink sink_logit / 1), the same SELECT and store guard as the LSE;
    # ``stats_use_log2`` leaves them untouched (natural units always).  This set
    # follows the paged_d_shapes idiom: None = every shape in d_shapes when the
    # flags are set, a set = served only when the flavor the lowering SELECTS
    # (_selected_d_shape) is a member -- the kernel FILE owns the store.  Dense,
    # unsplit, unpaged plans on the 2-CTA prefill tile only (mismatch holds the
    # conditions); the standalone adapter's check_support reads the same constant
    # (config_sm100 / config_sm107.SM10x_SCORE_STATS_SHAPES, rule 8b').  APPENDED
    # after attn_scale_prefolded_d_shapes (append-only contract above).
    score_stats_d_shapes: Optional[frozenset] = None
    # Shapes that carry the row's DENSE leg only -- the Rubin 32- / 64-wide
    # tiles of the f16/bf16 d128 file (config_sm107.SM107_F16_SMALL_D_SHAPES).
    # ``_selected_d_shape`` takes such a shape only for a graph that asks for
    # nothing beyond that leg: no THD, no paged KV, no split_kv > 1, no
    # PackGQA, no score side outputs, and a cga (when requested) inside the
    # shape's cgas_by_d_shape domain.  Any other graph walks PAST it to the
    # next covering shape -- the d128 flavor it rode before the tile existed,
    # where those legs are wired and validated -- instead of a decline, so a
    # new native tile never withdraws a graph the row served through the
    # envelope, and a persisted autotune record keeps working (a cga2 request
    # at D <= 32 lands on the 64-wide tile, the smallest covering shape
    # validated at that width).  Empty on every other row: their native shapes
    # either carry a leg or decline it (the SM100 native d64's split / score
    # declines describe THAT kernel; the row never served d64 through an
    # envelope, so there is no route to fall back to).  The standalone adapter
    # walks the same predicate (config_sm107.sm107_small_tile_leg /
    # sm107_f16_tile_width(leg=, cga=), rule 8b').  APPENDED after
    # score_stats_d_shapes (append-only contract above).
    dense_only_d_shapes: frozenset[tuple[int, int]] = frozenset()
    # Additive bias on the RESIDENT-row-block arm (the Rubin f16/bf16 d128-file
    # tiles, config_sm107.SM107_BIAS_D_SHAPES) -- beside the row-level ``bias``
    # flag above: the native shapes whose kernel carries the block (None = every
    # shape of a ``bias`` row serves it, the SM80 row's semantics), and whether
    # the bias must be the Q/K/V dtype (the SM80 row also takes fp32; the block
    # arm loads the io dtype's 16-bit pairs).  The resident block holds the work item's
    # rows x its whole KV range (``config_sm107.sm107_bias_resident_s_kv_max``); above
    # that budget the kernel streams the bias as a 2- / 3-stage per-kv-tile ring
    # (``BIAS_MODE = 2``), so ``mismatch`` asks ``config_sm107.sm107_bias_mode_pick``
    # (one arithmetic with the adapter and the config factory, rule 8b') and declines
    # typed only where NEITHER form fits at any width of the flavor's domain.
    # Dense, unsplit, unpaged, non-PackGQA, ungated, without the score side
    # outputs; the bias is ``[1|B, 1|H_q, S_q, S_kv]`` with a contiguous key dim
    # and 16-byte strides (S_kv % 8 == 0 for a compact bias: the TMA rule).
    bias_d_shapes: Optional[frozenset] = None
    bias_io_dtype_only: bool = False
    # The per-key mask ``kv_mask`` (python-only input: ``[1|B, 1, 1, S_kv]``), with the
    # dtype set it is served in: BOOLEAN = a key with a False entry is masked for every
    # query (a query whose every key is masked reads O = 0 / Stats = -inf); the Q/K/V
    # dtype or FLOAT = an ADDITIVE per-key vector folded into the scores pre-scale like
    # the bias (the models' finite -1e9 fold, preserved as a value), S_kv % 8 == 0 with
    # 16-byte strides for its granule loads.  Served by the same arm as the bias (the
    # same shapes and S_kv-free legs); a 16-bit vector must be the Q/K/V dtype.
    kv_mask: bool = False
    kv_mask_dtypes: frozenset = frozenset()
    # Per-d-shape KV tile-width domain, mirroring sched_policies_by_d_shape: ``tile_ns``
    # stays the row-wide floor (and the heuristics' default), an entry here REPLACES it
    # for the flavor the lowering selects -- the Rubin f16/bf16 32- / 64-wide tiles
    # render TILE_N 64 as well as 128 (config_sm107.SM107_TILE_NS_BY_D_SHAPE, the
    # adapter's twin), the d128 rendering of the same file keeps 128.  APPENDED after
    # kv_mask_dtypes (append-only contract above).
    tile_ns_by_d_shape: tuple[tuple[tuple[int, int], frozenset[int]], ...] = ()
    # Whether PackGQA requires tile_m to be divisible by the GQA ratio G (or, on
    # a partial flavor, to share a factor with it).  False on the SM120 rows:
    # any G up to tile_m packs, tile_m % G != 0 included
    # (config_sm100.pack_gqa_group_size).  APPENDED after tile_ns_by_d_shape
    # (append-only contract above).
    pack_gqa_requires_divisible: bool = True
    # Native flavors with dense Q only. Unlike thd_d_shapes, this preserves
    # the THD envelope of every other flavor on the row. Append-only.
    thd_excluded_d_shapes: frozenset[tuple[int, int]] = frozenset()
    # Native shapes with half Q/O and independent FP8 K/V storage. SM120
    # paged loads also use the row's existing eight-aligned head envelopes.
    # This is plain sdpa: no FP8-Q quantization or scalar-descale contract.
    mixed_fp8_kv_d_shapes: frozenset[tuple[int, int]] = frozenset()
    # Pure declaration constraints, shared with the standalone adapter.
    # Appended to retain the historical positional constructor.
    declaration_check: Optional[Callable[["ga.SdpaGraphFacts"], Optional[str]]] = None
    # Native flavors with a bounded set of extra legs beyond dense execution.
    # All requested legs must be in the set; unsupported combinations retain
    # the next covering flavor. Appended to preserve positional construction.
    native_legs_by_d_shape: tuple[tuple[tuple[int, int], frozenset[str]], ...] = ()

    # The canonical band-support SET this row claims (cudnn.sdpa.band), the
    # model the probe decides with.  APPENDED at the end for the append-only
    # contract; ``None`` means "derive from the legacy flags above" and a row
    # needing a restricted claim passes an explicit BandSupport instead (e.g.
    # ``band=BandSupport.unmasked_only()`` for an unmasked-only kernel).
    # __post_init__ resolves it, so ``capabilities.band`` is never None after
    # construction -- and a legacy flag that is SET but outside an explicit
    # claim is refused there, rather than letting field order decide.
    band: Optional["band.BandSupport"] = None
    # Independent physical pool capacities are separate from shared H/page geometry.
    independent_paged_pool_counts: bool = False

    def __post_init__(self) -> None:
        """Single normalization layer: legacy mask flags -> canonical band set."""
        if self.band is None:
            object.__setattr__(
                self,
                "band",
                band.BandSupport.from_legacy_flags(
                    causal=self.causal,
                    bottom_right=self.bottom_right,
                    swa=self.swa,
                    right_band_widening=self.right_band_widening,
                ),
            )
            return
        # A band-only declaration is the point of the model, so only the flags
        # that are actually SET constrain the explicit claim: each of them must
        # name an axis the declared band serves (the legacy spelling is then a
        # redundant, narrower view of the same claim).  The all-default legacy
        # block is the spelling of "nothing declared" and leaves the band alone.
        legacy_axes = (
            ("causal", self.causal, band.RIGHT_CAUSAL in self.band.right),
            ("right_band_widening", self.right_band_widening, band.RIGHT_FINITE in self.band.right),
            ("swa", self.swa, band.LEFT_WINDOW in self.band.left),
            ("bottom_right", self.bottom_right, band.ANCHOR_BOTTOM_RIGHT in self.band.anchors),
        )
        conflicting = sorted(name for name, set_flag, served in legacy_axes if set_flag and not served)
        if conflicting:
            raise ValueError(
                "Capabilities declares an explicit band support "
                f"{self.band.as_dict()} that does not serve the legacy mask flags {conflicting} set with it; the two "
                "spellings must state the same claim (clear the flags or widen the band) -- conflicting declarations "
                "are never silently resolved"
            )

    # Append-only public warp-topology knob domain.
    warp_spec_cfgs: frozenset[int] = frozenset({0})
    swap_abs: frozenset[int] = frozenset()
    # A fixed main implementation, when this row has completed the explicit
    # implementation migration. Empty keeps the unmigrated row's contract.
    implementation: str = ""
    # Exact (target, head-shape) cells transferred to another engine. Retain
    # the old native flavor for its other envelopes/targets, without offering
    # a duplicate automatic route or accepting an obsolete explicit replay.
    delegated_exact: frozenset = frozenset()

    # Per-d-shape Q tile-height domain, the tile_m twin of tile_ns_by_d_shape above, for a
    # row whose Q tile differs by head dims (the SM90 row, whose exact (128, 128) graphs also
    # take a 128 x 128 tile, the dedicated D128 template, beside the 64 x 64 tile of the
    # D512 body).  The sched_policies_by_d_shape idiom: an entry OVERRIDES tile_ms for the
    # d-shape the lowering SELECTS (effective_tile_ms, as effective_tile_ns reads
    # tile_ns_by_d_shape).  () = the row-wide tile_ms holds for every d-shape (every row but
    # SM90).  APPENDED after warp_spec_cfgs (append-only contract above; the same test pins
    # it).
    tile_ms_by_d_shape: tuple[tuple[tuple[int, int], frozenset[int]], ...] = ()
    # Shapes whose kernel flavors carry the own-block-excluded causal bound (the op
    # attribute sdpa_mxfp8(causal_exclude_own_block=32): keys j < 32 * floor(i / 32)).
    # None = unserved (the default: a row opts in).  Matched on the flavor the
    # lowering SELECTS (_selected_d_shape).  Dense, unpadded, unpaged
    # plans only (split-KV and PackGQA included) -- mismatch() holds those conditions.  APPENDED after
    # tile_ms_by_d_shape (append-only contract above).
    causal_exclude_own_block_d_shapes: Optional[frozenset] = None


# The block sizes the own-block-excluded causal bound is wired for: the MXFP8 V
# scale block.  Any power of two dividing the q sub-tile would keep the kernel's
# unmasked-segment split exact; only the size with a use is claimed.
CAUSAL_EXCLUDE_OWN_BLOCK_SIZES = frozenset({32})


def _sm90_d128_tile_mismatch(facts: "ga.SdpaGraphFacts", knobs: "SdpaFwdKnobs") -> Optional[str]:
    """Why the SM90 row's 128 x 128 tile, the dedicated D128 template (#378), cannot run this graph and knob set, or None.

    A tile of 128 on either axis names that template, so the other axis must be 128 or unset; then the graph's own
    refusal with the knobs' split and packing (config_sm90.d128_graph_decline: config_sm90.d128_decline's exact
    (128, 128) head dims and nothing in config_sm90.D128_NOT_SERVED, then the row's S_q == 1 and unpadded bottom-right
    S_q > S_kv declines)."""
    from cudnn.sdpa.fwd.config_sm90 import D128_FLAVOR, FLAVOR_TILES, d128_graph_decline

    tile_m, tile_n = FLAVOR_TILES[D128_FLAVOR]
    if knobs.tile_m not in (None, tile_m) or knobs.tile_n not in (None, tile_n):
        return f"SM90's D128 template runs a {tile_m} x {tile_n} tile; requested tile_m={knobs.tile_m}/tile_n={knobs.tile_n}"
    return d128_graph_decline(facts, pack_gqa=knobs.pack_gqa is True, split_kv=knobs.split_kv or 1)


def _mixed_fp8_kv(capabilities: Capabilities, facts: "ga.SdpaGraphFacts") -> bool:
    return (
        not (facts.is_fp8 or facts.is_mxfp8 or facts.is_backward)
        and facts.dtype in (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16)
        and facts.kv_dtype in (cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2)
        and facts.uniform_qo_dtype
        and (
            (facts.d_qk, facts.d_v) in capabilities.mixed_fp8_kv_d_shapes
            or (
                capabilities.sm_lo == 120
                and _selected_d_shape(capabilities, facts) in capabilities.mixed_fp8_kv_d_shapes
                and _sm120_paged_head_dims_supported(facts.d_qk, facts.d_v, decode=True, d512=True, d576=True)
            )
        )
    )


def _band_covers_kv_tail(facts: "ga.SdpaGraphFacts") -> bool:
    """True when the causal band (plain or right-widened) provably masks every
    KV column >= S_kv, so a ragged KV tail cannot leak into the softmax.

    The band x geometry rule itself lives on the canonical model
    (``band.BandFacts.covers_kv_tail``); this is the facts-taking alias the
    lowering, the split gates and the KV-tail rule below share."""
    return band.BandFacts.from_sdpa_facts(facts).covers_kv_tail(facts.s_q, facts.s_kv)


def _synth_kv_padding(capabilities: Capabilities, facts: "ga.SdpaGraphFacts") -> bool:
    """True when this graph's ragged S_kv is served through the kernel's padded
    mask compiled against the scalar S_kv (the adapter's kv_tail_mask, #1425)
    — the one path split-KV cannot ride.

    Only a DENSE, mask-free graph whose S_kv is not a multiple of the KV tile
    takes it. A padded graph already carries real per-batch lengths; a paged
    graph is padded by construction (per-batch ``seq_len_kv`` is mandatory and
    bounds the walk on device), so a declared ``paged_attention_max_seq_len_kv``
    that is not a tile multiple — FlashInfer passes its true max verbatim,
    e.g. 4000 — never selects this path and must not cost the launch its
    split; THD carries its lengths via cu_seqlens. The split gates in
    :func:`mismatch` and ``heuristics._split_points`` share this predicate with
    the lowering so the three cannot drift apart again (they did: the gates
    lacked the padded/paged terms and declined every paged decode graph whose
    declared max was not a 128-multiple).
    """
    return (
        capabilities.skv_tail_via_padding
        and not facts.padded
        and not facts.has_paged_kv
        and not facts.thd
        and facts.s_kv % (capabilities.skv_tile or 128) != 0
        and not _band_covers_kv_tail(facts)
    )


_RAGGED_OFFSET_DTYPES = (cudnn.data_type.INT32, cudnn.data_type.INT64)


def _ragged_row_divisor(t) -> Optional[int]:
    """Offset-units-per-token divisor of a ragged tensor: its declared TOKEN
    stride (the ``S`` stride of the (B, H, S, D) declaration -- ``H * D`` for a
    packed buffer, larger with a token-stride gap) over the offset multiplier,
    when the tensor carries int32 / int64 ragged offsets and the multiplier
    divides that stride.  cuDNN's contract: ``offset[b] * multiplier`` is the
    ELEMENT offset of sequence ``b``'s first token, so the kernel reads
    ``offset[b] // divisor`` as its token row.  None when the tensor cannot
    be addressed that way."""
    off = getattr(t, "ragged_offset", None)
    if off is None or off.get_data_type() not in _RAGGED_OFFSET_DTYPES:
        return None
    stride = tuple(int(s) for s in t.get_stride())
    if len(stride) != 4:
        return None
    token_stride = stride[2]
    mult = int(t.get_ragged_offset_multiplier() or 1)
    if mult <= 0 or token_stride <= 0 or token_stride % mult != 0:
        return None
    return token_stride // mult


def _thd_decode_leg_int64(facts: "ga.SdpaGraphFacts") -> bool:
    """Whether a ``_thd_decode_leg`` graph's ragged offsets are int64 (one width
    for Q, O and Stats -- a mixed graph is declined by ``_thd_decode_leg``)."""
    return facts.q_t.ragged_offset.get_data_type() == cudnn.data_type.INT64


def _thd_decode_leg(capabilities: Capabilities, facts: "ga.SdpaGraphFacts") -> bool:
    """True when a THD (ragged Q/O/Stats) graph rides the SM100 d128 DECODE tile's
    ragged-Q leg (``sm100/decode_d128_f16.py``, ``TemplateParams.ragged_q``)
    instead of the prefill tile's THD leg: the FlashInfer prefill-style paged
    graph at one token per sequence.

    The leg keeps the dense grid over the declared batch, reads each batch's Q
    ragged offset on device as its row coordinate over the packed Q view, and
    always splits the KV walk so the combine pass -- not the kernel -- places
    the final O / Stats rows at their ragged offsets. Hence the shape of the
    predicate: the (128, 128) half flavor on SM100/SM103/SM107,
    S_q(max) == 1, PAGED K/V (a ragged K/V needs the THD leg's
    clamped descriptors), ragged Stats when Stats are requested (a per-batch
    padded Stats has no ragged base to place rows at), int32 offsets whose
    multiplier divides the row, per-batch ``seq_len_kv`` (the dense kernel's
    SEQ_KV read; the cu form is not plumbed) and no fused epilogue gate
    (unsplittable); a sink rides it (the combine folds it once). Twin of
    ``SdpaFwdDslSm100.thd_decode_leg``; keep in lockstep.
    """
    if not (facts.thd and facts.has_paged_kv and facts.s_q == 1):
        return False
    if facts.is_fp8 or facts.is_mxfp8 or getattr(facts, "has_epilogue_gate", False):
        return False
    if capabilities.sm_lo not in (100, 107) or capabilities.is_fp8 or capabilities.is_mxfp8:
        return False
    if (facts.d_qk, facts.d_v) != (128, 128) or _selected_d_shape(capabilities, facts) != (128, 128):
        return False
    if facts.cu_seq_kv_t is not None:
        return False
    if _ragged_row_divisor(facts.q_t) is None or _ragged_row_divisor(facts.o_t) is None:
        return False
    if facts.stats_t is not None and _ragged_row_divisor(facts.stats_t) is None:
        return False
    # One offset width for every ragged operand (the kernels compile one read width).
    widths = {t.ragged_offset.get_data_type() for t in (facts.q_t, facts.o_t) + ((facts.stats_t,) if facts.stats_t is not None else ())}
    return len(widths) == 1


def decode_d256_q_tile_for_row(capabilities: Capabilities, s_q: int, pack_g: int, wide_launch: bool = False, paged: bool = False, masked: bool = True) -> int:
    """The d256 decode tile's N extent (the Q rows of one token unit) a ROW routes ``s_q``
    tokens packed ``pack_g`` heads per token onto, or 0 for the prefill tile: the Rubin
    row's rule (``config_sm107.decode_d256_q_tile``, table-driven: the 32- and 64-column
    tiles routed in one unit, then two token units of the widest admitted width for a packed
    MTP step) on cc 10.7, the Blackwell line's (``config_sm100.decode_d256_q_tile``: the
    16-column tile in one unit, and the 32-column one too for a ``wide_launch`` --
    :func:`config_sm100.d256_wide_decode_launch`) otherwise.  ``wide_launch`` (appended) is
    the Blackwell line's launch-size reading (:func:`d256_wide_decode_launch_for_facts`);
    ``paged`` / ``masked`` (appended) are the graph's cache kind and causal / window mask --
    the Rubin 64-column width's form predicate (``config_sm107.decode_d256_wide_tile_admits``)
    declines a mask-free PAGED step, which keeps the (16, 32) table.  Each line reads its own
    arguments only.  The facts-level twin of ``SdpaFwdDslSm100._decode_q_tile_for``'s arch
    dispatch; the heuristics' decode geometry reads it too -- keep the three in lockstep
    (:func:`_facts_decode_form` reads the two Rubin facts off ``SdpaGraphFacts``)."""
    if capabilities.sm_lo == 107:
        return _config_sm107.decode_d256_q_tile(s_q, pack_g, paged=bool(paged), masked=bool(masked))
    return decode_d256_q_tile(s_q, pack_g, wide_launch=wide_launch)


def d256_wide_decode_launch_for_facts(facts: "ga.SdpaGraphFacts", pack_g: int) -> bool:
    """:func:`config_sm100.d256_wide_decode_launch` on a graph's declared shape: batch x
    (heads / ``pack_g``) units over ``s_kv`` keys, never for a windowed graph.  The
    facts-level twin of ``SdpaFwdDslSm100._decode_q_tile_for``'s launch argument."""
    return d256_wide_decode_launch(facts.b, facts.h_q // max(1, pack_g), facts.s_kv, facts.window_left is not None)


def _facts_decode_form(facts: "ga.SdpaGraphFacts") -> Tuple[bool, bool]:
    """``(paged, masked)`` of a graph for :func:`decode_d256_q_tile_for_row`: the paged KV cache, and the causal /
    window mask exactly as the kernels' MASK_CAUSAL / MASK_SWA bits derive (``config_sm100._mask_flags_from``: a right
    band resolved -- ``right_bound`` 0 = causal, ``causal`` its effective reading -- or a left window)."""
    return bool(facts.has_paged_kv), bool(facts.causal) or facts.right_bound is not None or facts.window_left is not None


def _facts_decode_form_kw(facts: "ga.SdpaGraphFacts") -> dict:
    """:func:`_facts_decode_form` as the ``paged`` / ``masked`` keywords of :func:`decode_d256_q_tile_for_row`."""
    paged, masked = _facts_decode_form(facts)
    return dict(paged=paged, masked=masked)


def d256_decode_tile_selected(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", pack_g: int = 1, split_kv: Optional[int] = None) -> bool:
    """Whether the f16/bf16 row lowers this graph onto the d256 DECODE tile -- the
    swap-AB tile, ``sm100/d256_f16_swap_ab/kernel.py`` on the Blackwell row and
    ``sm107/decode_d256_f16.py`` on the Rubin row -- the facts-level twin of
    ``SdpaFwdDslSm100._decode_q_tile`` (and of the heuristics' cost-model gate):
    half inputs, dense (not THD) queries, the (256, 256) flavor, no pre-folded scale
    (not wired on the decode tile), and ``S_q x pack_g`` packed Q rows within the
    row's routed envelope (``decode_d256_q_tile_for_row``: 16 rows in one unit on the
    Blackwell line, which also routes the 32-column tile for a launch big enough to be
    HBM-bound; on cc 10.7 the 32-column tile for every launch, in up to two token units
    for a packed MTP step; ``pack_g`` is the decode tile's WHOLE-group packing when the plan
    packs, else 1).

    The fused epilogue gate: the tile has no gate seams, so a GATED graph rides it only
    when the plan SPLITS (``split_kv`` >= 2) -- the gate then moves into the split
    combine (``sm100/split_combine`` ``gate=True``: ``O *= sigmoid(G)`` on the fp32
    merged value, the fused kernels' own arithmetic; ``SdpaFwdDslSm100._gate_in_combine``)
    on the Rubin row, which claims the gate; unsplit, the d256 prefill kernel's fused
    epilogue serves it.  ``split_kv=None`` asks whether SOME plan rides the tile (the
    facts-only question), so a gated graph answers True there.  A gated graph whose
    caller enabled execute-time shape overrides never rides it: the split combine
    binds G to the plan's declared (B, H_q, S_q, D_v) (``prepared.CombineGate``), so
    such a graph keeps the unsplit fused-gate kernel (``mismatch`` names the
    overrides).  Keep the three in lockstep."""
    if capabilities.implementation in _D256_IMPLEMENTATIONS:
        return capabilities.implementation == _D256_SWAP_AB
    if capabilities.sm_lo == 100 and _in_d256_pilot_domain(facts):
        # Only the central policy generator uses the old union in this slice.
        # Real engines have a fixed implementation and take the branch above.
        from .heuristics import select_sm100_d256_geometry

        return select_sm100_d256_geometry(facts, pack_g=pack_g).implementation == _D256_SWAP_AB
    gate_ok = not facts.has_epilogue_gate or (
        capabilities.sm_lo == 107 and capabilities.epilogue_gate and (split_kv is None or split_kv > 1) and not facts.shape_overrides
    )
    return (
        capabilities.sm_lo in (100, 107)
        and capabilities.sm_hi <= _BLACKWELL[1]
        and not (facts.is_fp8 or facts.is_mxfp8)
        and facts.uniform_dtype
        and not facts.thd
        and gate_ok
        and not facts.attn_scale_prefolded
        and _selected_d_shape(capabilities, facts) == (256, 256)
        and decode_d256_q_tile_for_row(
            capabilities, facts.s_q, pack_g, wide_launch=d256_wide_decode_launch_for_facts(facts, pack_g), **_facts_decode_form_kw(facts)
        )
        > 0
    )


def _decode_tile_pack_g(facts: "ga.SdpaGraphFacts", knobs: Optional[SdpaFwdKnobs]) -> int:
    """The decode tile's packing for a knob set: the whole GQA ratio when it packs, else 1."""
    return (facts.h_q // facts.h_kv) if (knobs is not None and knobs.pack_gqa and facts.h_kv) else 1


def _decode_tile_pack_g_choices(facts: "ga.SdpaGraphFacts", knobs: Optional[SdpaFwdKnobs]) -> Tuple[int, ...]:
    """The packings a decode-tile question may assume: a knob set's own (:func:`_decode_tile_pack_g`); with NO knob set -- the
    facts-only question the heuristics' ``_eligible`` asks before any set is built -- every packing SOME plan may carry: unpacked,
    and the whole group where there is one.  A graph only the packed form serves (4 / 2 at ``S_q`` 33 over pages under the causal
    mask: 66 packed rows = two 64-column units, while 33 unpacked rows are past the 32-column tile) therefore keeps its row on the
    ranked list, where the heuristics propose the packed set (``_pack_gqa_eligible`` reads the same group) and ``mismatch`` declines
    the unpacked one.  Until 2026-10-15 the facts-only reading asked the unpacked question alone and dropped the row, so the ranked
    list offered nothing where the adapter and the facts twins routed the packed form -- the gate behind the graph-level pin of the
    Rubin decode suite (``test_decode_routing_boundary``), which the host twin test there now pins over a grid."""
    if knobs is not None:
        return (_decode_tile_pack_g(facts, knobs),)
    group = (facts.h_q // facts.h_kv) if facts.h_kv else 0
    return (1, group) if (group > 1 and facts.h_q % facts.h_kv == 0) else (1,)


def d256_decode_tile_served(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", knobs: Optional[SdpaFwdKnobs], split_kv: Optional[int] = None) -> bool:
    """:func:`d256_decode_tile_selected` for a knob set's own packing, or -- with no knob set -- for SOME plan: any packing of
    :func:`_decode_tile_pack_g_choices` the row routes onto the tile."""
    return any(d256_decode_tile_selected(capabilities, facts, pack_g, split_kv) for pack_g in _decode_tile_pack_g_choices(facts, knobs))


def _decode_tile_split_kv(knobs: Optional[SdpaFwdKnobs]) -> Optional[int]:
    """The split a knob set lowers with, for the decode tile's gate question: an
    unrequested split (None) lowers unsplit, so it reads 1; no knob set at all
    (``knobs is None``, the facts-only question) reads None (= some plan)."""
    return None if knobs is None else (knobs.split_kv or 1)


def _thd_decode_leg_divisors(facts: "ga.SdpaGraphFacts") -> tuple:
    """The (Q, O, Stats) elements-per-token divisors of a ``_thd_decode_leg`` graph
    (Stats: 1 when no Stats output is declared)."""
    return (
        _ragged_row_divisor(facts.q_t),
        _ragged_row_divisor(facts.o_t),
        _ragged_row_divisor(facts.stats_t) if facts.stats_t is not None else 1,
    )


def rubin_dense_d128_shared_leg(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", knobs: Optional[SdpaFwdKnobs] = None) -> bool:
    """Historical cc 10.7 dense half D128 decode/packing admission predicate (issue #1472).

    This includes the d64 envelope and excludes the pre-folded scale. The loader uses the shared decode tile
    at CGA1 and native SM107 prefill for packed CGA2. The historical predicate name and admission limits
    are retained for policy consumers. Twin of SdpaFwdDslSm100._rubin_shared_dense_leg; keep the gates in lockstep.

    ``knobs`` identifies the selected flavor. Native D64 dense and PackGQA plans
    use their own tile; a split request can still select D128 and this shared
    leg. A knob-less D64 graph selects the native tile and is not a shared leg."""
    return (
        capabilities.sm_lo == 107
        and not (facts.is_fp8 or facts.is_mxfp8)
        and not facts.thd
        and not facts.has_paged_kv
        and _selected_d_shape(capabilities, facts, knobs) == (128, 128)
        and not facts.attn_scale_prefolded
    )


def paged_thd_split_domain(capabilities: Capabilities, facts: "ga.SdpaGraphFacts") -> bool:
    """The paged subset of the bounded THD split contract."""
    return facts.has_paged_kv and thd_split_domain(capabilities, facts)


def thd_split_domain(capabilities: Capabilities, facts: "ga.SdpaGraphFacts") -> bool:
    """Fixed packed-Q bounds whose partial workspace is caller-owned."""
    return (
        capabilities.sm_lo in (100, 107)
        and (
            not facts.shape_overrides
            or (facts.max_total_seq_len_q is not None and 0 < facts.max_total_seq_len_q <= facts.b * facts.s_q)
            # D576 partials retain the native binder's B*S_q capacity clamp
            # when the graph omits the optional tighter total-Q bound.
            or ((facts.d_qk, facts.d_v) == SM100_NATIVE_D576_SHAPE and facts.max_total_seq_len_q is None and facts.b * facts.s_q > 0)
        )
        and (
            not facts.has_sink
            or supports_thd_split_sink(
                (facts.d_qk, facts.d_v),
                device_cc=facts.device_cc,
                fp8=facts.is_fp8 or facts.is_mxfp8,
                thd=facts.thd,
                paged=facts.has_paged_kv,
                max_q=facts.s_q,
            )
        )
        and not facts.has_epilogue_gate
        and supports_thd_split(
            (facts.d_qk, facts.d_v),
            device_cc=facts.device_cc,
            fp8=facts.is_fp8 or facts.is_mxfp8,
            thd=facts.thd,
            paged=facts.has_paged_kv,
            max_q=facts.s_q,
            padded_stats=facts.stats_t is not None and getattr(facts.stats_t, "ragged_offset", None) is None,
        )
    )


def _prepared_decline_reason(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", split_kv: Optional[int]) -> Optional[str]:
    """Pure admission for the graph's normalized VariantPack executor.

    Shared by planning and lowering; no adapter, tensor framework or compilation.
    ``split_kv=None`` asks whether the provider has a possible prepared plan.
    A complete assignment additionally checks the final O store's layout.
    Runtime geometry still has to fit the compiled binder's per-call contract.
    """
    if capabilities.sm_lo == 90 and facts.shape_overrides:
        return "SM90 prepared launch retains fixed graph geometry; shape/stride overrides are unsupported"
    if capabilities.sm_lo not in (90, 100, 107, 120):
        return "this engine has no prepared shape/stride override executor"
    if (facts.is_fp8 or facts.is_mxfp8) and capabilities.sm_lo == 107 and facts.device_cc != (10, 7):
        return "prepared SM107 FP8/MXFP8 requires device cc 10.7"
    if facts.o_block_scale and (facts.sf_o_t is None or facts.dtype_o != {16: cudnn.data_type.FP4_E2M1, 32: cudnn.data_type.FP8_E4M3}.get(facts.o_block_scale)):
        return "prepared block-scaled output requires the matching O and SF_O declarations"
    if facts.o_block_scale and (facts.thd or facts.has_paged_kv or (split_kv or 1) > 1 or facts.shape_overrides or facts.has_epilogue_gate):
        return "prepared block-scaled outputs require fixed dense unsplit plans"
    if facts.is_mxfp8 and (
        capabilities.sm_lo not in (100, 107, 120)
        or (
            capabilities.sm_lo == 107
            and (split_kv or 1) > 1
            and (
                (facts.d_qk, facts.d_v) not in SM107_MXFP8_SPLIT_SHAPES
                or facts.dtype_o not in (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16)
                or facts.has_paged_kv
                or facts.thd
            )
        )
        or not capabilities.is_mxfp8
        or facts.dtype_o not in (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2, cudnn.data_type.FP4_E2M1)
        or (facts.has_epilogue_gate and capabilities.sm_lo != 107)
        or (facts.shape_overrides and not facts.thd)
    ):
        return "prepared MXFP8 serves SM100 fixed dense or bounded THD, SM107 fixed dense (split-KV at D128 with half outputs) or bounded THD scalar outputs, and SM120 fixed dense or THD plans"
    if facts.is_fp8 and (
        capabilities.sm_lo not in (100, 107, 120)
        or facts.dtype_o not in (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2, cudnn.data_type.FP4_E2M1)
        or (facts.has_epilogue_gate and capabilities.sm_lo != 107)
    ):
        return "prepared FP8 serves SM100, SM107 or SM120 scalar-scaled outputs"
    if capabilities.sm_lo == 120 and facts.has_paged_kv:
        if facts.thd or facts.is_fp8 or facts.is_mxfp8 or not _sm120_paged_head_dims_supported(facts.d_qk, facts.d_v, decode=True, d512=True, d576=True):
            return "SM120 paged KV requires FP16/BF16 dense Q with head dims through 256 or D512/D576-V512 wide envelopes, divisible by 8"
    if facts.has_bias and capabilities.sm_lo == 120 and facts.shape_overrides:
        return "SM120 additive bias does not support execute-time shape/stride overrides"
    if facts.has_bias and capabilities.bias_d_shapes is None and capabilities.sm_lo != 120:
        # The resident-block rows bind the bias through the prepared binder (bind_dense validates its runtime geometry);
        # the SM80 row's bias travels through its own adapter and stays off the prepared path.
        return "prepared overrides cannot use bias"
    if facts.thd:
        if facts.has_epilogue_gate:
            return "prepared THD overrides cannot use an epilogue gate"
        if (split_kv or 1) > 1:
            if facts.has_sink and not getattr(cudnn._pybind_module._SdpaThdBinder, "supports_paged_split_sink", False):
                return "packed split sinks require the matching native cuDNN Frontend extension"
            # Ragged-Q decode binds offsets through its dense launch. The
            # paged D128 THD leg instead owns bounded packed partial regions.
            return None if _thd_decode_leg(capabilities, facts) or thd_split_domain(capabilities, facts) else "prepared THD overrides cannot use split-KV"
        return None
    if facts.has_epilogue_gate and (split_kv or 1) > 1 and facts.shape_overrides:
        # The d256 decode tile's split applies the gate in its combine, whose gate binding is fixed to the declared
        # (B, H_q, S_q, D_v) (prepared.CombineGate); an override-enabled graph may change either at execute.
        return (
            "prepared dense overrides cannot ride the gate-in-combine split (its gate binding is fixed to the declared (B, H_q, S_q, D_v)); "
            "a graph with execute-time shape overrides keeps the unsplit fused-gate kernel"
        )
    if facts.cu_seq_q_t is not None or facts.cu_seq_kv_t is not None:
        return "prepared dense overrides require per-batch lengths, not prefix sums"
    from cudnn.sdpa.fwd.config_sm100 import dense_bind_strides

    if capabilities.sm_lo == 90:
        from cudnn.sdpa.fwd.config_sm90 import dense_bind_strides

    tensors = [facts.q_t] + ([] if facts.has_paged_kv else [facts.k_t, facts.v_t])
    if facts.has_epilogue_gate:
        tensors.append(facts.epilogue_gate_t)
    if split_kv is not None and split_kv <= 1:
        tensors.append(facts.o_t)
    for tensor in tensors:
        if tensor is None:
            return "prepared overrides require declared operands"
        from cudnn.graph_types import storage_geometry

        geometry = storage_geometry(tensor.get_dim(), tensor.get_stride(), tensor.get_data_type())
        width = 1 if tensor.get_data_type() in (cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2, cudnn.data_type.FP4_E2M1) else 2
        if geometry is None or dense_bind_strides(*geometry, width) is None:
            return "prepared overrides require input and unsplit-output layouts that bind without a copy"
    if facts.o_t is None or not ga.dense_layout_ok(tuple(facts.o_t.get_dim()), tuple(facts.o_t.get_stride())):
        return "prepared overrides require a non-overlapping dense output layout"
    return None


def _beyond_dense_leg(facts: "ga.SdpaGraphFacts", knobs: Optional[SdpaFwdKnobs], allowed_legs: frozenset[str] = frozenset()) -> bool:
    """Whether any requested feature exceeds a tile's dense and allowed native legs."""
    return (
        _config_sm107.sm107_small_tile_leg(
            thd=facts.thd,
            paged=facts.has_paged_kv,
            score_stats=facts.has_score_max or facts.has_score_sum_exp,
            prefolded=facts.attn_scale_prefolded,
            split_kv=knobs.split_kv if knobs is not None else 1,
            pack_gqa=knobs.pack_gqa if knobs is not None else False,
            allowed_legs=allowed_legs,
        )
        is not None
    )


def _selected_d_shape(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", knobs: Optional[SdpaFwdKnobs] = None) -> Optional[tuple[int, int]]:
    """Smallest native flavor whose envelope covers this graph -- honouring the
    per-shape envelope floors, so the knob domains (cga, split) describe the
    flavor the lowering will actually pick (api_dsl._pick_flavor walks the same
    floors). An exact native shape is always its own selection -- except a
    restricted tile, which is taken only when every requested feature fits its
    native legs (``_beyond_dense_leg`` False) at a cga inside its domain (``knobs.cga``
    unset or in ``cgas_by_d_shape``); otherwise the walk continues to the next
    covering shape, the route the graph took before the tile existed.  Callers
    that know the knobs pass them (``mismatch``, ``effective_cgas``); a
    knob-less call sees the graph-fact legs only."""

    exact = (facts.d_qk, facts.d_v)
    if exact in capabilities.d_shapes and exact not in capabilities.dense_only_d_shapes and exact not in dict(capabilities.native_legs_by_d_shape):
        # The exact shape is the smallest covering one; explicit engines declare thousands of exact pairs.
        return exact
    floors = dict(capabilities.d_envelope_floors)
    covering = [
        shape
        for shape in capabilities.d_shapes
        if facts.d_qk <= shape[0] and facts.d_v <= shape[1] and ((facts.d_qk, facts.d_v) == shape or min(facts.d_qk, facts.d_v) > floors.get(shape, 0))
    ]
    if capabilities.dense_only_d_shapes or capabilities.native_legs_by_d_shape:
        native_legs = dict(capabilities.native_legs_by_d_shape)
        restricted = capabilities.dense_only_d_shapes | native_legs.keys()
        cga = knobs.cga if knobs is not None else None
        domains = dict(capabilities.cgas_by_d_shape)
        covering = [
            shape
            for shape in covering
            if shape not in restricted
            or (not _beyond_dense_leg(facts, knobs, native_legs.get(shape, frozenset())) and (cga is None or cga in domains.get(shape, capabilities.cgas)))
        ]
    return min(covering, key=lambda shape: (shape[0], shape[1])) if covering else None


def sm120_decode_tile(capabilities: Capabilities, tile_m: Optional[int]) -> bool:
    """Whether ``tile_m`` runs the SM120 half DECODE tile (sm120/decode_f16_key_group/kernel.py):
    the f16/bf16 SM120 row with ``tile_m`` 16 or 32."""
    return capabilities.sm_lo == 120 and not (capabilities.is_fp8 or capabilities.is_mxfp8) and tile_m is not None and _sm120_is_decode_tile(tile_m)


def _sm120_tile_mismatch(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", knobs: SdpaFwdKnobs) -> Optional[str]:
    """Why the SM120 kernel table has no CTA tile for ``knobs`` at this graph's
    head dims (None when it has one, or on any other row). ``tile_m`` and
    ``tile_n`` each pass their own domain in ``mismatch``; the template serving
    the head dims takes only the pairs ``config_sm120.tile_domain`` lists, and
    the decode tile (tile_m 16 / 32) adds its own graph conditions. Twin of the
    tile lines of ``api_dsl.SdpaFwdDslSm120.check_support``."""
    if capabilities.sm_lo != 120 or (knobs.tile_m is None and knobs.tile_n is None):
        return None
    if sm120_decode_tile(capabilities, knobs.tile_m):
        if facts.thd:
            return "the SM120 decode tile (tile_m 16 / 32) is dense-only; THD graphs run the prefill templates"
        if max(facts.d_qk, facts.d_v) > _SM120_GENERAL_HEAD_TILE_MAX:
            return f"the SM120 decode tile serves head dims up to {_SM120_GENERAL_HEAD_TILE_MAX}; graph has D_QK={facts.d_qk}/D_V={facts.d_v}"
        stage = _sm120_decode_kv_stage(*_sm120_head_tiles(facts.d_qk, facts.d_v, _SM120_HEAD_TILE_GRANULE))
        if knobs.tile_n is not None and knobs.tile_n != stage:
            return f"the SM120 decode tile streams {stage}-key K/V stages at D=({facts.d_qk}, {facts.d_v}); requested tile_n={knobs.tile_n}"
        return None
    domain = _sm120_tile_domain(facts.d_qk, facts.d_v, capabilities.is_fp8 or capabilities.is_mxfp8)
    # An unset tile_m runs the first sequence Q tile the table lists, else the
    # table's largest; an unset tile_n is picked from the table by SMEM fit.
    tile_m = knobs.tile_m
    if tile_m is None:
        tile_m = _SM120_SEQ_Q_TILES[0] if any(m == _SM120_SEQ_Q_TILES[0] for m, _ in domain) else max(m for m, _ in domain)
    served = (tile_m, knobs.tile_n) in domain if knobs.tile_n is not None else any(m == tile_m for m, _ in domain)
    if not served:
        return f"the SM120 kernel for D=({facts.d_qk}, {facts.d_v}) has no tile_m={knobs.tile_m}, tile_n={knobs.tile_n} tile; it takes {sorted(domain)}"
    return None


def pack_gqa_partial(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", knobs: Optional[SdpaFwdKnobs] = None) -> bool:
    """Whether the flavor this graph lowers onto packs a proper divisor of a
    GQA group that does not divide tile_m (``pack_gqa_partial_d_shapes``).

    A pinned plan supplies its knobs: packing can select a wider native tile
    than the graph's unpacked flavor. Fact-only callers retain their domain.
    """
    return capabilities.pack_gqa_partial_d_shapes is not None and _selected_d_shape(capabilities, facts, knobs) in capabilities.pack_gqa_partial_d_shapes


def effective_sched_policies(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", knobs: Optional[SdpaFwdKnobs] = None) -> frozenset[int]:
    """Scheduler domain of the native flavor the lowering will pick.

    Row-wide `sched_policies` is the floor; a `sched_policies_by_d_shape` entry
    for the selected flavor replaces it. That is what lets one Rubin row serve a
    d256 kernel that honours LPT next to flavors that do not.  ``knobs`` (when
    the caller has them) route a dense-only tile's non-dense legs to the flavor
    that carries them (``_selected_d_shape``), and name the SM120 decode tile,
    which walks its units in launch order (SCHED_NATURAL only).
    """
    if knobs is not None and sm120_decode_tile(capabilities, knobs.tile_m):
        return frozenset({SCHED_NATURAL})
    selected = _selected_d_shape(capabilities, facts, knobs)
    if selected is not None:
        for shape, shape_domain in capabilities.sched_policies_by_d_shape:
            if shape == selected:
                return shape_domain
    return capabilities.sched_policies


def _sm80_explicit_tile_domains(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", knobs: Optional[SdpaFwdKnobs]):
    if capabilities.sm_lo == 80 and knobs is not None and (knobs.tile_m is not None or knobs.tile_n is not None):
        try:
            return _sm80_graph_tile_domains(facts.d_qk, facts.d_v)
        except ValueError:
            return frozenset(), frozenset()
    return None


def effective_swap_abs(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", knobs: SdpaFwdKnobs) -> frozenset[int]:
    """The existing prefill and swap-AB decode bodies on the same legal graph."""
    if not capabilities.swap_abs or not facts.is_fp8 or facts.o_block_scale or facts.softmax_precision not in (None, cudnn.data_type.FLOAT):
        return frozenset()
    if facts.has_bias or facts.has_kv_mask or facts.has_epilogue_gate or facts.has_score_max or facts.has_score_sum_exp or facts.attn_scale_prefolded:
        return frozenset()
    shape = _selected_d_shape(capabilities, facts, knobs)
    geometry = dict(
        device_cc=facts.device_cc,
        paged=facts.has_paged_kv,
        thd=facts.thd,
        batch_size=facts.b,
        s_q=facts.s_q,
        h_q=facts.h_q,
        h_kv=facts.h_kv,
        pack_gqa=knobs.pack_gqa,
        cga=knobs.cga,
    )
    if not (supports_d64_fp8_swap_ab(shape, **geometry) or supports_dense_fp8_swap_ab(shape, **geometry, tile_n=knobs.tile_n)):
        return frozenset()
    return capabilities.swap_abs


def effective_tile_ms(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", knobs: Optional[SdpaFwdKnobs] = None) -> frozenset[int]:
    """Q tile-height domain of the d-shape the lowering will pick.  Explicit SM80 tile knobs read the native query
    tiles of the graph's head dims (``_sm80_explicit_tile_domains``), which keeps the default proposal domain
    unchanged.  Otherwise row-wide ``tile_ms`` is the floor, and a ``tile_ms_by_d_shape`` entry for the selected
    d-shape replaces it, as ``effective_sched_policies`` does for the scheduler and ``effective_tile_ns`` for the KV
    tile."""
    sm80_domains = _sm80_explicit_tile_domains(capabilities, facts, knobs)
    if sm80_domains is not None:
        return sm80_domains[0]
    selected = _selected_d_shape(capabilities, facts, knobs)
    if selected is not None:
        for shape, shape_domain in capabilities.tile_ms_by_d_shape:
            if shape == selected:
                return shape_domain
    return capabilities.tile_ms


def effective_tile_ns(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", knobs: Optional[SdpaFwdKnobs] = None) -> frozenset[int]:
    """KV tile-width domain of the native flavor the lowering will pick: row-wide ``tile_ns`` is the floor, a
    ``tile_ns_by_d_shape`` entry for the selected flavor replaces it (the Rubin small tiles render 64 and 128, the
    native D128 rendering additionally admits explicit 64 on its double-score pipeline)."""
    sm80_domains = _sm80_explicit_tile_domains(capabilities, facts, knobs)
    if sm80_domains is not None:
        return sm80_domains[1]
    selected = _selected_d_shape(capabilities, facts, knobs)
    if (
        selected == (64, 64)
        and facts.thd
        and not (facts.is_fp8 or facts.is_mxfp8)
        and (capabilities.sm_lo == 100 or (knobs is not None and (knobs.split_kv or 1) > 1))
    ):
        # Native SM100 D64 THD and the shared split body use N128. The N64
        # small-tile factory serves dense, unpacked inputs on SM100 only.
        return frozenset({128})
    if capabilities.sm_lo == 107 and facts.is_fp8 and not facts.is_mxfp8:
        # Explicit per-tensor widths do not inherit the half-precision resident-arm
        # default picker. Keep the row's default128 while admitting the D64 option.
        return dict(SM107_FP8_TILE_NS_BY_D_SHAPE).get(selected, capabilities.tile_ns)
    if capabilities.sm_lo == 107 and supports_native_d128_narrow_tile(
        selected,
        device_cc=facts.device_cc,
        half_precision=facts.dtype in (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16),
        thd=facts.thd,
        paged=facts.has_paged_kv,
        cga=knobs.cga if knobs is not None else None,
        split_kv=(knobs.split_kv or 1) if knobs is not None else 1,
        has_bias=facts.has_bias,
        has_kv_mask=facts.has_kv_mask,
    ):
        return frozenset({64, 128})
    if selected is not None:
        for shape, shape_domain in capabilities.tile_ns_by_d_shape:
            if shape == selected:
                return shape_domain
    return capabilities.tile_ns


def packed_thd_split(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", knobs: SdpaFwdKnobs) -> bool:
    """Whether these knobs split a THD graph through the prepared packed split host (``_thd_split_host``)."""
    split_cga = knobs.cga if knobs.cga is not None else (1 if (facts.d_qk, facts.d_v) == (64, 64) else 2)
    return (
        split_cga == (2 if (facts.d_qk, facts.d_v) in ((256, 256), SM100_NATIVE_D576_SHAPE) else 1)
        and (knobs.split_kv or 1) > 1
        and thd_split_domain(capabilities, facts)
    )


# CUDA caps gridDim.y and gridDim.z at 65535 (#1573).
_GRID_YZ_MAX = 65535


def launch_grid_mismatch(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", knobs: SdpaFwdKnobs) -> Optional[str]:
    """Why these knobs' launch grid puts more than 65535 on gridDim.y or .z, else ``None``.

    The NATURAL scheduler launches heads and batch x split_kv on y/z; the LPT
    policies and the SM100+/SM120 ragged grids are flat. The d256 decode tile
    keeps batch on y, the SM80/SM90 ragged grids keep heads and batch on y/z, and
    the split-KV combine keeps heads and batch on y/z, under every policy; the
    packed THD split combine keeps only heads on y. Packed
    GQA launches fewer head rows; ``h_q`` bounds them from above because the
    packing factor is per kernel flavor.
    """
    batch, heads = facts.b * (knobs.split_kv or 1), facts.h_q
    if max(batch, heads) <= _GRID_YZ_MAX:
        return None
    if capabilities.sm_lo in (80, 90) and (knobs.split_kv or 1) > 1:
        if max(facts.b, heads, knobs.split_kv) <= _GRID_YZ_MAX:
            return None
        return "SM80/SM90 split-KV requires batch, heads and split count each at most 65535"
    if packed_thd_split(capabilities, facts, knobs):
        # The packed combine launches ((t_q+3)//4, heads, 1) and the split attention a flat grid.
        if heads <= _GRID_YZ_MAX:
            return None
        return f"the packed split-KV combine launches heads={heads} on gridDim.y, past the CUDA limit of {_GRID_YZ_MAX}"
    if (knobs.split_kv or 1) > 1 and max(facts.b, heads) > _GRID_YZ_MAX:
        return f"the split-KV combine launches heads={heads} and batch={facts.b} on gridDim.y/z, past the CUDA limit of {_GRID_YZ_MAX}"
    if d256_decode_tile_selected(capabilities, facts, _decode_tile_pack_g(facts, knobs), _decode_tile_split_kv(knobs)):
        if facts.b <= _GRID_YZ_MAX:
            return None
        return f"the d256 decode tile launches batch={facts.b} on gridDim.y, past the CUDA limit of {_GRID_YZ_MAX}"
    if facts.thd:
        if capabilities.sm_lo not in (80, 90):
            return None
        return f"the ragged grid launches heads={heads} and batch={facts.b} on gridDim.y/z, past the CUDA limit of {_GRID_YZ_MAX}"
    if knobs.sched_policy not in (None, SCHED_NATURAL):
        return None
    return (
        f"SCHED_POLICY=NATURAL launches heads={heads} and batch x split_kv={batch} on gridDim.y/z, "
        f"past the CUDA limit of {_GRID_YZ_MAX}; the LPT policies launch a flat grid"
    )


def sm100_nonpaged_d128_thd_single_cta_leg(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", *, cga, split_kv) -> bool:
    """The SM100 single-CTA D128 half THD prefill leg of a graph (``config_sm100.supports_sm100_nonpaged_d128_thd_single_cta``):
    half inputs, no pre-folded scale, one query head per KV head, a plain mask."""
    return (
        capabilities.sm_lo == 100
        and not facts.attn_scale_prefolded
        and facts.dtype in (cudnn.data_type.BFLOAT16, cudnn.data_type.HALF)
        and supports_sm100_nonpaged_d128_thd_single_cta(
            (facts.d_qk, facts.d_v),
            device_cc=facts.device_cc,
            fp8=facts.is_fp8 or facts.is_mxfp8,
            thd=facts.thd,
            paged=facts.has_paged_kv,
            cga=cga,
            split_kv=split_kv or 1,
            one_head_per_kv=facts.h_kv > 0 and facts.h_q == facts.h_kv,
            plain_mask=(
                facts.window_left is None
                and not facts.right_band_widening
                and not facts.has_sink
                and not facts.has_bias
                and not facts.has_kv_mask
                and not facts.has_epilogue_gate
            ),
        )
    )


def native_small_half_dense_paged_leg(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", knobs: Optional[SdpaFwdKnobs] = None) -> bool:
    return capabilities.sm_lo == 107 and _config_sm107.supports_native_small_half_dense_paged(
        _selected_d_shape(capabilities, facts, knobs),
        device_cc=facts.device_cc,
        fp8=facts.is_fp8 or facts.is_mxfp8,
        thd=facts.thd,
        paged=facts.has_paged_kv,
    )


def native_d512_dense_leg(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", knobs: Optional[SdpaFwdKnobs] = None) -> bool:
    return capabilities.sm_lo == 107 and _config_sm107.supports_native_d512_dense_pack_split(
        _selected_d_shape(capabilities, facts, knobs),
        device_cc=facts.device_cc,
        fp8=facts.is_fp8 or facts.is_mxfp8,
        thd=facts.thd,
        paged=facts.has_paged_kv,
        cga=knobs.cga if knobs is not None else None,
    )


def native_d192_paged_leg(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", knobs: Optional[SdpaFwdKnobs] = None) -> bool:
    return capabilities.sm_lo == 107 and _config_sm107.supports_native_d192_paged(
        _selected_d_shape(capabilities, facts, knobs),
        device_cc=facts.device_cc,
        fp8=facts.is_fp8 or facts.is_mxfp8,
        paged=facts.has_paged_kv,
        cga=knobs.cga if knobs is not None else None,
        split_kv=(knobs.split_kv or 1) if knobs is not None else 1,
    )


# K/V bytes of a graph (B x KV heads x S_kv x (d_qk + d_v), one byte per element) up to which the single-CTA per-tensor fp8
# d128 / d192x128 builds are listed on the 107 line.
_SM107_D192_PT_CGA1_KV_BYTES_MAX = 96 * 1024 * 1024


def effective_cgas(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", split_kv: Optional[int] = None, cga: Optional[int] = None) -> frozenset[int]:
    """CGA domain of the native flavor and split leg selected by the graph.

    ``cga`` is the REQUESTED width when the caller validates one (``mismatch``):
    on a row with ``dense_only_d_shapes`` the flavor a request lands on depends
    on it (a cga2 request at D <= 32 selects the 64-wide tile, not the
    cga1-only 32-wide one), so the domain returned is that flavor's.  Without
    it (the heuristics' auto pick) the domain is the dense selection's."""

    if capabilities.implementation in _sm107_wide_half.IMPLEMENTATIONS:
        return frozenset({_sm107_wide_half.explicit_geometry(capabilities.implementation, facts).cluster_m})
    if capabilities.implementation == _sm107_swap_fp8_owner.NAME:
        return frozenset({1})
