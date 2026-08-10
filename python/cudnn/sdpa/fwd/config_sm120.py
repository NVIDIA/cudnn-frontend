# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Compile-time configuration for the FROST SM120 SDPA prefill template."""

from __future__ import annotations

from dataclasses import dataclass

from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_FP16

SEQ_Q_TILES = (128, 64)
SEQ_KV_TILES = (128, 64)
SUPPORTED_HEAD_TILES = tuple(range(16, 257, 16))

# SMEM the SM120 parts expose to a kernel. The adapter asks cutlass for the
# authoritative number at build time; this constant lets the ranking answer
# "would this tile even fit" without importing the DSL.
SMEM_CAPACITY_BYTES = 101376


def smem_bytes(d_qk: int, d_v: int, q_tile: int, kv_tile: int, itemsize: int = 2) -> int:
    """SMEM the SM120 prefill kernel needs for one specialization.

    One K tile (D_QK wide) plus one V tile (D_V wide), aliased with the
    q_tile x D_V output staging tile. Lives here rather than in the adapter
    because the ranking must not propose a tile the kernel cannot fit, and the
    two answering that question differently is how a plan list fills up with
    entries that decline at build.
    """
    return max(kv_tile * (d_qk + d_v), q_tile * d_v) * itemsize + 16


@dataclass(frozen=True)
class TemplateParams:
    """Per-graph parameters that change the traced SM120 kernel.

    Tensor geometry deliberately stays out of this record. Scalar dimensions
    are inputs to the template module's per-shape ``compile()`` cache, while
    strides follow the fixed compact-BSHD kernel contract. This frozen record
    identifies the import-time specialization shared by all compatible shapes.
    """

    dtype_qkv: int = DTYPE_FP16
    # The mask is ONE diagonal band (same model as config_sm100 / the analyzer
    # facts): per-side offsets from the diagonal, None = unbounded on that
    # side. This kernel serves window_right in {None, 0} only (plain causal;
    # right-band widening is not plumbed here). bottom_right anchors the
    # band's diagonal at the bottom-right corner.
    window_left: int | None = None
    window_right: int | None = None
    bottom_right: bool = False
    seq_q_lens_present: bool = False
    seq_kv_lens_present: bool = False
    has_sink: bool = False
    thd_varlen: bool = False
    q_tile: int = SEQ_Q_TILES[0]
    kv_tile: int = SEQ_KV_TILES[0]


def validate_params(params: TemplateParams) -> None:
    """Validate the SM120 template specialization.

    Reachable failures should already have been rejected by the engine
    capabilities or adapter support checks; this validation is a backstop for
    direct template use.
    """

    if params.dtype_qkv not in (DTYPE_BF16, DTYPE_FP16):
        raise ValueError(f"SM120 SDPA: dtype_qkv must be DTYPE_BF16 ({DTYPE_BF16}) or DTYPE_FP16 ({DTYPE_FP16}); got {params.dtype_qkv}")
    if params.window_right not in (None, 0):
        raise ValueError(f"SM120 SDPA: window_right must be None (unbounded) or 0 (causal) — right-band widening is not plumbed; got {params.window_right}")
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
