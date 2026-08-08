# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Compile-time configuration for the FROST SM120 SDPA prefill template."""

from __future__ import annotations

from dataclasses import dataclass

from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3, DTYPE_FP16  # noqa: F401  (DTYPE_E4M3 re-exported for the FP8 template)

SEQ_Q_TILES = (128, 64)
SEQ_KV_TILES = (128, 64)
SUPPORTED_HEAD_TILES = tuple(range(16, 257, 16))


def fp8_tile_choice(s_q: int, s_kv: int, h_q: int, batch: int, sm_count: int) -> tuple[int, int]:
    """(q_tile, kv_tile) for the SM120 per-tensor FP8 cell.

    FP8 only: this is read off the fp8 kernel's own measurements and the f16
    cell's optimum differs, so the two must not share a default.

    ``kv_tile=128`` unconditionally -- fastest in all 28 shapes measured. An
    earlier revision of this kernel staged P through SMEM, and that traffic
    grew with the KV tile, which made 64 the better choice; the shfl path
    removed it and the optimum moved. A tile rule is a property of the kernel
    it was measured on, not of the hardware.

    ``q_tile=64`` when the grid cannot fill the machine AND each CTA has enough
    KV tiles for the extra Q-tile loop to amortize: halving the Q tile doubles
    the CTA count, which only pays while SMs sit idle, and adds loop overhead
    that only long sequences absorb. Worth 1.5x at 64 CTAs. The 1.5x SM-count
    bound is where the two stop trading evenly: 240 CTAs still want the finer
    tile on this part, 320 want the coarser one by 1.19x.
    """
    if sm_count <= 0:
        return 128, 128
    grid = -(-s_q // 128) * h_q * batch
    kv_tiles = -(-s_kv // 128)
    fine = grid * 2 <= sm_count or (grid * 2 <= 3 * sm_count and kv_tiles >= 12)
    return (64 if fine else 128), 128


@dataclass(frozen=True)
class TemplateParams:
    """Per-graph parameters that change the traced SM120 kernel.

    Tensor geometry deliberately stays out of this record. Scalar dimensions
    are inputs to the template module's per-shape ``compile()`` cache, while
    strides follow the fixed compact-BSHD kernel contract. This frozen record
    identifies the import-time specialization shared by all compatible shapes.
    """

    dtype_qkv: int = DTYPE_FP16
    is_causal: bool = False
    causal_bottom_right: bool = False
    window_size_left: int | None = None
    seq_q_lens_present: bool = False
    seq_kv_lens_present: bool = False
    has_sink: bool = False
    thd_varlen: bool = False
    q_tile: int = SEQ_Q_TILES[0]
    kv_tile: int = SEQ_KV_TILES[0]


def validate_params(params: TemplateParams, allowed_dtypes: tuple[int, ...] = (DTYPE_BF16, DTYPE_FP16)) -> None:
    """Validate the SM120 template specialization.

    Reachable failures should already have been rejected by the engine
    capabilities or adapter support checks; this validation is a backstop for
    direct template use. ``allowed_dtypes`` defaults to the FP16/BF16 template's
    set; the FP8 template passes its own.
    """

    if params.dtype_qkv not in allowed_dtypes:
        raise ValueError(f"SM120 SDPA: dtype_qkv must be one of {allowed_dtypes}; got {params.dtype_qkv}")
    if params.causal_bottom_right and not params.is_causal:
        raise ValueError("SM120 SDPA: causal_bottom_right requires is_causal=True")
    if params.window_size_left is not None and params.window_size_left < 0:
        raise ValueError(f"SM120 SDPA: window_size_left must be non-negative; got {params.window_size_left}")
    if params.q_tile not in SEQ_Q_TILES:
        raise ValueError(f"SM120 SDPA: q_tile must be one of {SEQ_Q_TILES}; got {params.q_tile}")
    if params.kv_tile not in SEQ_KV_TILES:
        raise ValueError(f"SM120 SDPA: kv_tile must be one of {SEQ_KV_TILES}; got {params.kv_tile}")
    if params.thd_varlen:
        if not params.seq_kv_lens_present:
            raise ValueError("SM120 SDPA: thd_varlen requires seq_kv_lens_present (the THD metadata tensor)")
        if params.seq_q_lens_present:
            raise ValueError("SM120 SDPA: seq_q_lens_present is dense-only (THD carries per-sequence Q lengths via cu_seqlens)")
