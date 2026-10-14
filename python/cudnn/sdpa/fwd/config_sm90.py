# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Kernel-facing configuration for the FROST SM90 SDPA forward templates.

Two templates ship: the dedicated D128 template (``sm90/prefill_d128_f16.py``,
issue #378), a 128 x 128 tile for the exact (128, 128) dense declarations it
serves, and the D512 body (``sm90/prefill_d512_f16.py``, with its native
128/256/512-column storage and the exact dense D64 body beside it) on a 64 x 64
tile for every head dim up to 512 and every declared feature. A pinned tile
names the template; with none pinned ``pick_flavor`` chooses once, from the head
dims and the declared features (``NOT_SERVED``). The frozen ``TemplateParams``
record contains only graph declarations and selected tuning knobs. Runtime
lengths, pointers, and packed token totals stay out of compile keys.
Graph-adapter declaration logic lives with the adapter; this module keeps the
small interface consumed by the native D64/D128/D256/D512 kernels.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass

from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_FP16, SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL

SCALE_POSITIVE = 1
SCALE_ZERO = 0
SCALE_NEGATIVE = -1


# Maximum envelope. The adapter selects native 128/256/512-column storage and
# contraction; exact dense D64 keeps its separate body. GMEM maps carry the actual
# extents, with zero-filled loads and clipped stores at head tails. Eight elements
# satisfy the TMA 16-byte global-stride rule for both half types.
D_TILE = 512
D_ALIGN = 8
# Sequence tile geometry shared by the native head widths.
TILE_M = 64
TILE_N = 64

# Kernel flavors: the SM90 templates a declaration lowers onto, keyed by (D_QK, D_V),
# smallest first. D512 is the D512 body: every head-dim pair up to its tile
# (head_dims_mismatch), on its native 128/256/512-column storage (head_tile) or, for
# the exact dense D64 subset, the D64 body (native_d64_mismatch). D128 (issue #378) is
# the dedicated 128 x 128 template; it serves exact (128, 128) dense declarations outside
# NOT_SERVED. The adapter, the engine row and the tests share these spellings.
D512_FLAVOR = (D_TILE, D_TILE)
D128_FLAVOR = (128, 128)
FLAVORS = (D128_FLAVOR, D512_FLAVOR)
# Each flavor's fixed CTA tile, (tile_m, tile_n): Q rows by keys. A pinned tile names the
# flavor whose tile it is (a 128 tile the D128 template, a 64 tile the D512 body), and the
# adapter refuses a tile no flavor of the declaration has; the engine row states the same
# (Capabilities.tile_ms_by_d_shape / tile_ns_by_d_shape).
FLAVOR_TILES = {D512_FLAVOR: (TILE_M, TILE_N), D128_FLAVOR: (128, 128)}
# Native-range floor per flavor, in the engine row's d_envelope_floors vocabulary: D128
# serves head dims in (127, 128], i.e. exactly (128, 128). D512 has no template floor: it
# serves every pair up to its tile (the engine row's floors are ROW decisions,
# engines._sm90_spec).
FLAVOR_FLOORS = {D128_FLAVOR: 127}
# What a native flavor does not serve, by declared feature (declared_features' vocabulary),
# with the label its refusal names. pick_flavor never selects a flavor for a declaration that
# names one of its features, so such a declaration runs on the D512 body, which serves them
# all; validate_params(flavor=...) refuses them with these labels, in this order.
D128_NOT_SERVED = {
    "thd": "THD (ragged) layouts",
    "paged": "paged KV",
    "split_kv": "split-KV",
    "sink": "attention sinks",
    "window_left": "a sliding window",
    "window_right": "a widened right band",
    "lengths": "per-batch lengths",
    "pack_gqa": "PackGQA",
}
NOT_SERVED = {D128_FLAVOR: D128_NOT_SERVED}


def declared_features(*, thd=False, sink=False, window_left=None, window_right=None, lengths=False, pack_gqa=False, paged=False, split_kv=1) -> frozenset[str]:
    """The features one declaration names, in the vocabulary of ``NOT_SERVED``.

    ``window_left`` is the sliding-window offset and ``window_right`` a widened right bound,
    each None when absent; a ``window_right`` of 0 is plain causal and names nothing.
    ``lengths`` is any per-batch Q or KV length tensor, ``paged`` a paged K/V pool and
    ``split_kv`` the KV partition count (1, unsplit, names nothing).
    """
    named = (
        ("thd", thd),
        ("paged", paged),
        ("split_kv", split_kv != 1),
        ("sink", sink),
        ("window_left", window_left is not None),
        ("window_right", bool(window_right)),
        ("lengths", lengths),
        ("pack_gqa", pack_gqa),
    )
    return frozenset(feature for feature, on in named if on)


def native_d64_mismatch(
    *,
    thd=False,
    sink=False,
    pack_gqa=False,
    q_lengths=False,
    kv_lengths=False,
    window_left=None,
    window_right=None,
    bottom_right=False,
    s_q=0,
    s_kv=0,
    split_kv=1,
):
    """The bounded native D64 leg, shared by graph and standalone admission."""
    if split_kv != 1:
        return "native SM90 D64 does not support split-KV"
    if thd or sink or pack_gqa or q_lengths or kv_lengths or window_left is not None or window_right is not None:
        return "native SM90 D64 serves dense, unpadded, unpacked Q/K/V without sinks or window bands"
    if bottom_right and s_q > s_kv:
        return "unpadded bottom-right S_q > S_kv is not yet supported on SM90"
    return None


def head_dims_mismatch(d_qk, d_v) -> str | None:
    """Why ``(d_qk, d_v)`` is outside the D envelope, or None; D_QK != D_V is served."""
    for value in (d_qk, d_v):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or not 0 < value <= D_TILE or value % D_ALIGN:
            return (
                f"SM90 SDPA serves positive D_QK/D_V multiples of {D_ALIGN} up to its {D_TILE}-column tile "
                f"(TMA 16-byte global-stride rule); graph has D_QK={d_qk}/D_V={d_v}"
            )
    return None


def pick_flavor(d_qk: int, d_v: int, flavors: tuple[tuple[int, int], ...] = (D512_FLAVOR,), declared: frozenset[str] = frozenset()) -> tuple[int, int]:
    """The template that serves a ``(d_qk, d_v)`` declaration naming ``declared``, among ``flavors``.

    The smallest native flavor in ``flavors`` that covers the pair above its floor
    (``FLAVOR_FLOORS``) and serves every declared feature (``NOT_SERVED``), else D512,
    the D512 body. Total: an out-of-envelope pair still answers D512, and
    ``head_dims_mismatch`` declines it, so the adapter's existing NotImplementedError is
    unchanged. ``flavors`` is the caller's pool (the flavors with a template,
    ``api_dsl._SM90_KERNEL_FILES``, as ``api_dsl._pick_flavor``'s ``candidates`` on the
    SM100 line); the default pool is D512 alone, so every declaration selects D512.
    ``declared`` is ``declared_features`` of the declaration.
    """
    for flavor in FLAVORS:
        if flavor == D512_FLAVOR or flavor not in flavors or declared & NOT_SERVED.get(flavor, {}).keys():
            continue
        if d_qk <= flavor[0] and d_v <= flavor[1] and min(d_qk, d_v) > FLAVOR_FLOORS.get(flavor, 0):
            return flavor
    return D512_FLAVOR


def d128_decline(d_qk: int, d_v: int, declared: frozenset[str] = frozenset()) -> str | None:
    """Why the D128 template (SM90's 128 x 128 tile) does not serve a ``(d_qk, d_v)``
    declaration naming ``declared``, or None: the refusal a pinned 128 tile meets where
    ``pick_flavor`` keeps the declaration on the D512 body. The head dims first (exactly
    (128, 128), ``FLAVOR_FLOORS``), then the first ``NOT_SERVED`` feature in table order."""
    if pick_flavor(d_qk, d_v, FLAVORS) != D128_FLAVOR:
        return f"SM90's 128 x 128 tile (the D128 template) serves exact (128, 128) head dims; graph has D_QK={d_qk}/D_V={d_v}"
    for feature, label in D128_NOT_SERVED.items():
        if feature in declared:
            return f"SM90's 128 x 128 tile (the D128 template) does not serve {label}"
    return None


def dense_bind_strides(shape: tuple, strides: tuple, elem_bytes: int) -> tuple | None:
    """SM90 binds every covering dense permutation with TMA-aligned live strides."""
    from cudnn.sdpa.graph_analyzer import dense_layout_ok

    if not dense_layout_ok(shape, strides):
        return None
    bs, hs, ss = (int(shape[3]) if size == 1 else int(stride) for size, stride in zip(shape[:3], strides))
    if any(stride * elem_bytes % 16 for stride in (bs, hs, ss)):
        return None
    return bs, ss, hs


@dataclass(frozen=True)
class TemplateParams:
    """Per-graph choices that change the traced SM90 kernel specialization."""

    dtype_qkv: int = DTYPE_FP16
    causal: bool = False  # Top-left on dense unpadded S_q == S_kv is spelled bottom-right by the adapter.
    thd_varlen: bool = False
    has_lse: bool = False
    pack_gqa: bool = False
    qh_per_kh: int = 1
    sched_policy: int = SCHED_NATURAL
    scale_mode: int = SCALE_POSITIVE
    # Dense (B,) length reads; THD sets seq_kv_lens_present, its metadata operand.
    seq_q_lens_present: bool = False
    seq_kv_lens_present: bool = False
    # Diagonal band, appended so every earlier specialization keeps its defaults.
    # `window_left` is the sliding-window offset W (cuDNN left bound - 1): key k is visible
    # from row q when k >= q + delta - W. `window_right` is a widened right bound R > 0
    # (k <= q + delta + R); plain causal stays `causal=True`. `bottom_right=None` is the
    # original alignment, bottom-right exactly when causal; SdpaFwdDslSm90.check_support canonicalizes it.
    window_left: int | None = None
    window_right: int | None = None
    bottom_right: bool | None = None
    # Per-Q-head FP32 sink logits: an extra bound operand in these specializations only.
    has_sink: bool = False
    # Stats written as (max + ln(sum_exp)) * log2(e) (sdpa(stats_use_log2=True)): the Stats store
    # scales the natural-log LSE by log2(e).
    stats_log2: bool = False
    # THD caller-buffer origins: per port in (Q, K, V, O, Stats) order, ``(bits, multiplier)`` of the
    # port's bound ragged offsets (int32 / int64, and the offset unit in elements), or None for a port
    # bound without offsets, which keeps the compact prefix. None for the whole field keeps the packed
    # layout. The setup launch turns each offset into a token origin, so padded THD (gaps between
    # sequences, TE's cu_seqlens_padded) is read and written where the caller put it.
    thd_origins: tuple | None = None
    split_kv: int = 1
    # Native contraction/storage envelope; independent of runtime lengths.
    head_tile: int = 512
    # Dense queries with native TMA loads from independently mapped K/V pages.
    page_size: int = 0
    # Step 7 speed switch of the D128 template (sm90/prefill_d128_f16.py), appended so every earlier specialization
    # keeps its defaults: False renders the v1 loop. intra_wg_overlap issues QK(t) and PV(t-1) as two WGMMA groups in
    # flight together inside each compute warpgroup, and computes the same O and Stats bit for bit, up to the sign of a
    # zero O element. SdpaFwdDslSm90 sets the shipped value (api_dsl.D128_INTRA_WG_OVERLAP) for the D128 flavor only;
    # D512 refuses it.
    intra_wg_overlap: bool = False


# The ports of ``TemplateParams.thd_origins``, in order.
THD_ORIGIN_PORTS = ("q", "k", "v", "o", "stats")


def thd_origin_maps(b: int) -> int:
    """Tensor maps of a caller-origin THD plan: per-sequence O in ``0..B-1``, the dead-unit pad at
    ``B``, then per-sequence K at ``B+1..2B`` and V at ``2B+1..3B``."""
    return 3 * b + 1


def thd_origins_off(b: int) -> int:
    """Int32 word offset of the Int64 ``[5, B]`` token origins, right after the maps."""
    from cudnn.frost.tile_dsl.thd import TENSOR_MAP_QWORDS, THD_MAPS_OFF

    return THD_MAPS_OFF(b) + thd_origin_maps(b) * TENSOR_MAP_QWORDS * 2


def thd_meta_words(b: int, origins: bool) -> int:
    """Int32 words of the THD metadata operand: the shared layout with its ``B + 3`` maps or, with
    caller-buffer origins, the ``3B + 1`` maps and then the origins."""
    from cudnn.frost.tile_dsl.thd import THD_MAPS_META_WORDS

    return thd_origins_off(b) + 2 * len(THD_ORIGIN_PORTS) * b if origins else THD_MAPS_META_WORDS(b)


# A band bound at or beyond S_q + S_kv never masks a key inside the declared envelopes, and
# every band bound is added to Int32 token coordinates in the kernel.
BAND_COORDINATE_LIMIT = 2**30


def validate_params(params: TemplateParams, flavor: tuple[int, int] = D512_FLAVOR) -> None:
    """Validate the immutable specialization before constructing a kernel.

    ``flavor`` names the template asking. D512 (the default) runs the checks below
    unchanged, messages included, and refuses the D128 template's intra-warpgroup
    overlap switch. D128 first refuses its ``NOT_SERVED`` features (the
    declarations pick_flavor keeps on D512, so the adapter never reaches this refusal),
    then runs the same checks under its own name.
    """
    if flavor == D128_FLAVOR:
        declared = declared_features(
            thd=params.thd_varlen,
            sink=params.has_sink,
            window_left=params.window_left,
            window_right=params.window_right,
            lengths=params.seq_q_lens_present or params.seq_kv_lens_present,
            pack_gqa=params.pack_gqa,
            paged=params.page_size != 0,
            split_kv=params.split_kv,
        )
        for feature, label in D128_NOT_SERVED.items():
            if feature in declared:
                raise ValueError(f"SM90 D128 does not serve {label}; the D512 body (tile 64) serves such declarations")
    elif flavor != D512_FLAVOR:
        raise ValueError(f"SM90 has no {flavor} template; flavors: {sorted(FLAVORS)}")
    if flavor == D512_FLAVOR and params.intra_wg_overlap:
        raise ValueError("SM90 D512 has no intra-warpgroup overlap switch; it is a D128 template field")
    name = f"SM90 D{flavor[0]}"
    tile_m = FLAVOR_TILES[flavor][0]
    if type(params.page_size) is not int or params.page_size < 0:
        raise ValueError("SM90 page_size must be a nonnegative integer")
    if params.page_size:
        p = params.page_size
        if p % 8 or (p < TILE_N and TILE_N % p) or (p >= TILE_N and p % TILE_N):
            raise ValueError("SM90 page_size must be divisible by 8 and divide 64 or be a multiple of 64")
        if params.thd_varlen or not params.seq_kv_lens_present:
            raise ValueError("SM90 paged KV requires dense Q and device KV lengths")
    if type(params.head_tile) is not int or params.head_tile not in (128, 256, 512):
        raise ValueError("SM90 head_tile must be 128, 256 or 512")
    if type(params.split_kv) is not int or not 1 <= params.split_kv <= 65535:
        raise ValueError("SM90 split_kv must be an integer in [1, 65535]")
    if params.thd_varlen and params.split_kv > 1:
        raise ValueError("SM90 split-KV requires dense Q")
    if params.dtype_qkv not in (DTYPE_FP16, DTYPE_BF16):
        raise ValueError(f"{name} requires uniform FP16 or BF16 Q/K/V/O")
    from .config_sm100 import pack_gqa_supported

    if params.qh_per_kh < 1 or (params.pack_gqa and not pack_gqa_supported(params.qh_per_kh, 1, tile_m, partial=True)):
        raise ValueError(f"{name} PackGQA requires a positive head ratio sharing a factor with tile_m={tile_m}")
    if params.sched_policy not in (SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2):
        raise ValueError(f"{name} supports only NATURAL/LPT/LPT_L2 single-tile scheduling")
    if params.thd_varlen and params.sched_policy != SCHED_NATURAL:
        raise ValueError(f"{name} THD uses the natural single-tile work decoder")
    if params.scale_mode not in (SCALE_POSITIVE, SCALE_ZERO, SCALE_NEGATIVE):
        raise ValueError(f"{name} scale mode must be positive, zero, or negative")
    if params.thd_varlen:
        if not params.seq_kv_lens_present:
            raise ValueError(f"{name} thd_varlen requires seq_kv_lens_present (the THD metadata tensor)")
        if params.seq_q_lens_present:
            raise ValueError(f"{name} seq_q_lens_present is dense-only (THD carries per-sequence Q lengths via cu_seqlens)")
    if params.thd_origins is not None:
        if not params.thd_varlen or len(params.thd_origins) != len(THD_ORIGIN_PORTS):
            raise ValueError(f"{name} thd_origins is THD-only and lists Q, K, V, O and Stats")
        for port, entry in zip(THD_ORIGIN_PORTS, params.thd_origins):
            if entry is not None and (len(entry) != 2 or entry[0] not in (32, 64) or type(entry[1]) is not int or entry[1] < 1):
                raise ValueError(f"{name} thd_origins[{port}] must be (32 or 64, multiplier >= 1)")
        if params.thd_origins[4] is not None and not params.has_lse:
            raise ValueError(f"{name} Stats origins require has_lse")
    if params.causal and params.window_right is not None:
        raise ValueError(f"{name} plain causal is causal=True; window_right holds only a widened right bound")
    for bound_name, bound, low in (("window_left", params.window_left, 0), ("window_right", params.window_right, 1)):
        if bound is not None and (type(bound) is not int or not low <= bound < BAND_COORDINATE_LIMIT):
            raise ValueError(f"{name} {bound_name} must be an integer in [{low}, 2**30)")
    if params.bottom_right is not None:
        if params.window_left is None and params.window_right is None and not params.causal:
            raise ValueError(f"{name} diagonal alignment needs a band bound")
        if params.bottom_right == params.causal:
            raise ValueError(f"{name} bottom_right repeats the alignment causal implies; the canonical spelling is None")
