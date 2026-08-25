# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Compile-time configuration for the FROST SM120 SDPA prefill template."""

from __future__ import annotations

from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL
from dataclasses import dataclass
from typing import Optional

from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3, DTYPE_E5M2, DTYPE_FP16  # noqa: F401  (DTYPE_E4M3 re-exported for the FP8 template)

SEQ_Q_TILES = (128, 64)
SEQ_KV_TILES = (128, 64)
HEAD_TILE_GRANULE = 16
SUPPORTED_HEAD_TILE_MIN = 16
SUPPORTED_HEAD_TILE_MAX = 256
SUPPORTED_HEAD_TILES = tuple(range(SUPPORTED_HEAD_TILE_MIN, SUPPORTED_HEAD_TILE_MAX + 1, HEAD_TILE_GRANULE))
FP8_HEAD_TILE_GRANULE = 32
SUPPORTED_HEAD_TILES_FP8 = tuple(range(FP8_HEAD_TILE_GRANULE, SUPPORTED_HEAD_TILE_MAX + 1, FP8_HEAD_TILE_GRANULE))

# SMEM the SM120 parts expose to a kernel. The adapter asks cutlass for the
# authoritative number at build time; this constant lets the ranking answer
# "would this tile even fit" without importing the DSL.
SMEM_CAPACITY_BYTES = 101376


def smem_bytes(d_qk: int, d_v: int, q_tile: int, kv_tile: int, itemsize: int = 2, out_itemsize: Optional[int] = None) -> int:
    """SMEM the SM120 prefill kernel needs for one specialization.

    One K tile (D_QK wide) plus one V tile (D_V wide), aliased with the
    q_tile x D_V output staging tile. The two terms size INDEPENDENTLY:
    ``itemsize`` is the QKV element, ``out_itemsize`` the staged output's, and
    FP8 differs on exactly that (1-byte KV, half-precision O).

    Lives here rather than in the adapter because the ranking must not propose
    a tile the kernel cannot fit, and two answers to that question is how a plan
    list fills with entries that decline at build.
    """
    return max(kv_tile * (d_qk + d_v) * itemsize, q_tile * d_v * (itemsize if out_itemsize is None else out_itemsize)) + 16


@dataclass(frozen=True)
class TemplateParams:
    """Per-graph parameters that change the traced SM120 kernel.

    Tensor geometry deliberately stays out of this record. Scalar dimensions
    are inputs to the template module's per-shape ``compile()`` cache, while
    strides follow the fixed compact-BSHD kernel contract. This frozen record
    identifies the import-time specialization shared by all compatible shapes.
    """

    dtype_qkv: int = DTYPE_FP16
    # O dtype (frost tile_dsl codes, same table as dtype_qkv). The f16
    # template stores O at the input dtype and ignores this; the FP8 template
    # selects its quantizing-store epilogue from it (E4M3/E5M2/BF16/FP16).
    dtype_o: int = DTYPE_FP16
    # The mask is ONE diagonal band (same model as config_sm100 / the analyzer
    # facts): per-side offsets from the diagonal, None = unbounded on that
    # side. window_right = 0 is plain causal; window_right > 0 widens the
    # diagonal right by R columns (cuDNN's diagonal_band_right_bound).
    # bottom_right anchors the band's diagonal at the bottom-right corner.
    window_left: int | None = None
    window_right: int | None = None
    bottom_right: bool = False
    seq_q_lens_present: bool = False
    seq_kv_lens_present: bool = False
    has_sink: bool = False
    thd_varlen: bool = False
    sched_policy: int = SCHED_NATURAL
    q_tile: int = SEQ_Q_TILES[0]
    kv_tile: int = SEQ_KV_TILES[0]
    pack_gqa: bool = False
    # KV split: each Q tile's KV-tile range [min_kv_tile, num_kv_tiles) is cut
    # into ``split_kv`` contiguous chunks, each run by its own CTA writing a
    # partial (O, LSE) that kernels/split_combine_sm100.py reduces.  1 = off.
    # Opt-in only -- the graph front door never selects it.
    split_kv: int = 1


def validate_params(
    params: TemplateParams,
    allowed_dtypes: tuple[int, ...] = (DTYPE_BF16, DTYPE_FP16),
    allowed_o_dtypes: tuple[int, ...] = (DTYPE_BF16, DTYPE_FP16),
    allow_right_band: bool = True,
) -> None:
    """Validate the SM120 template specialization.

    Reachable failures should already have been rejected by the engine
    capabilities or adapter support checks; this validation is a backstop for
    direct template use. ``allowed_dtypes``/``allowed_o_dtypes`` default to the
    FP16/BF16 template's sets (which stores O at the input dtype and plumbs no
    quantizing epilogue); the FP8 template passes its own — FP8 in, any of its
    four quantizing-store epilogues out. ``allow_right_band=False`` rejects a
    widened right band for templates that do not plumb one.
    """

    if params.dtype_qkv not in allowed_dtypes:
        raise ValueError(f"SM120 SDPA: dtype_qkv must be one of {allowed_dtypes}; got {params.dtype_qkv}")
    if params.dtype_o not in allowed_o_dtypes:
        raise ValueError(f"SM120 SDPA: dtype_o must be one of {allowed_o_dtypes}; got {params.dtype_o}")
    if params.split_kv < 1:
        raise ValueError(f"SM120 SDPA: split_kv must be >= 1 (1 = KV-split off); got {params.split_kv}")
    if params.split_kv > 1:
        # Each of these would need extra machinery in the combine pass.
        if params.thd_varlen:
            raise ValueError("SM120 SDPA: split_kv > 1 is dense-only (THD packs its own flat grid)")
        if params.has_sink:
            # The sink logit is folded into the softmax denominator per tile, so
            # every split would add its own copy of it.
            raise ValueError("SM120 SDPA: split_kv > 1 with an attention sink is not supported")
        if params.sched_policy != SCHED_NATURAL:
            # The split index rides the NATURAL grid's batch axis (y = batch +
            # split*B); LPT / LPT_L2 flatten the grid to 1-D and derive the batch
            # from the linear tile id, so there is no axis left to fold it into.
            raise ValueError(f"SM120 SDPA: split_kv > 1 currently requires SCHED_NATURAL; got sched_policy={params.sched_policy}")

    if params.window_right is not None and params.window_right < 0:
        raise ValueError(f"SM120 SDPA: window_right must be None (unbounded) or >= 0 (0 = plain causal); got {params.window_right}")
    if not allow_right_band and params.window_right not in (None, 0):
        raise ValueError(f"SM120 SDPA: right-band widening is not plumbed for this template; window_right must be None or 0, got {params.window_right}")
    if params.bottom_right and params.window_right is None:
        raise ValueError("SM120 SDPA: bottom_right anchors the band's diagonal and requires a right bound (window_right)")
    if params.window_left is not None and params.window_left < 0:
        raise ValueError(f"SM120 SDPA: window_left must be non-negative; got {params.window_left}")
    if params.q_tile not in SEQ_Q_TILES:
        raise ValueError(f"SM120 SDPA: q_tile must be one of {SEQ_Q_TILES}; got {params.q_tile}")
    if params.kv_tile not in SEQ_KV_TILES:
        raise ValueError(f"SM120 SDPA: kv_tile must be one of {SEQ_KV_TILES}; got {params.kv_tile}")
    if params.thd_varlen:
        if not params.seq_kv_lens_present:
            raise ValueError("SM120 SDPA: thd_varlen requires seq_kv_lens_present (the THD metadata tensor)")
        if params.seq_q_lens_present:
            raise ValueError("SM120 SDPA: seq_q_lens_present is dense-only (THD carries per-sequence Q lengths via cu_seqlens)")
    if params.pack_gqa and params.thd_varlen:
        raise ValueError("SM120 SDPA: pack_gqa is not supported with thd_varlen")
