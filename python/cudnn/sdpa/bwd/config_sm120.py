# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Compile-time configuration for the FROST SM120 SDPA backward template."""

from __future__ import annotations

from dataclasses import dataclass

from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_FP16

SEQ_Q_TILES = (32, 64, 128)
SEQ_KV_TILES = (64, 128)
SUPPORTED_HEAD_DIMS = (32, 64, 128, 192, 256)


def padded_head_dim(d: int) -> "int | None":
    """Smallest native kernel head-dim size >= ``d``, or ``None`` when ``d`` exceeds them all."""

    return min((b for b in SUPPORTED_HEAD_DIMS if b >= d), default=None)


def padded_head_dims(d_qk: int, d_v: int) -> "tuple[int, int] | None":
    """Native kernel head-dim sizes for a head-dim pair."""

    d_qk_pad = padded_head_dim(d_qk)
    d_v_pad = padded_head_dim(d_v)
    if d_qk_pad is None or d_v_pad is None:
        return None
    # Unequal dims must both be multiples of 64 (one smem swizzle).
    if d_v_pad != d_qk_pad:
        d_v_pad = max(d_v_pad, 64)
    return d_qk_pad, d_v_pad


@dataclass(frozen=True)
class TemplateParams:
    """Per-graph parameters that change the traced SM120 backward kernel.

    Tensor geometry deliberately stays out of this record. Scalar dimensions
    are inputs to the template module's per-shape ``compile()`` cache. This
    frozen record identifies the import-time specialization shared by all
    compatible shapes.
    """

    dtype_qkv: int = DTYPE_FP16
    is_causal: bool = False
    causal_top_left: bool = False
    window_size_left: int | None = None
    window_size_right: int | None = None
    deterministic: bool = False
    use_pdl: bool = True
    q_tile: int = 0
    kv_tile: int = 0
    # Padding mask: per-batch int32 lengths; seq_q is only valid with seq_kv.
    seq_kv_lens_present: bool = False
    seq_q_lens_present: bool = False
    # Sink Attention
    sink_present: bool = False
    dsink_present: bool = False


def validate_params(params: TemplateParams) -> None:
    """Validate the SM120 backward template specialization.

    Reachable failures should already have been rejected by the engine
    capabilities or adapter support checks; this validation is a backstop for
    direct template use.
    """

    if params.dtype_qkv not in (DTYPE_BF16, DTYPE_FP16):
        raise ValueError(f"SM120 SDPA bwd: dtype_qkv must be DTYPE_BF16 ({DTYPE_BF16}) or DTYPE_FP16 ({DTYPE_FP16}); got {params.dtype_qkv}")
    if params.causal_top_left and not params.is_causal:
        raise ValueError("SM120 SDPA bwd: causal_top_left requires is_causal=True")
    if params.window_size_left is not None and params.window_size_left < 0:
        raise ValueError(f"SM120 SDPA bwd: window_size_left must be non-negative; got {params.window_size_left}")
    if params.window_size_right is not None and params.window_size_right < 0:
        raise ValueError(f"SM120 SDPA bwd: window_size_right must be non-negative; got {params.window_size_right}")
    if params.window_size_right is not None and not params.is_causal:
        raise ValueError("SM120 SDPA bwd: window_size_right widens the causal diagonal and requires is_causal=True")
    if params.seq_q_lens_present and not params.seq_kv_lens_present:
        raise ValueError("SM120 SDPA bwd: seq_q_lens_present requires seq_kv_lens_present (padding mask)")
    if params.dsink_present and not params.sink_present:
        raise ValueError("SM120 SDPA bwd: dsink_present requires sink_present (a dSink output needs the sink logits)")
    if params.q_tile not in (0,) + SEQ_Q_TILES:
        raise ValueError(f"SM120 SDPA bwd: q_tile must be one of {(0,) + SEQ_Q_TILES} (0 = per-head-dim default); got {params.q_tile}")
    if params.kv_tile not in (0,) + SEQ_KV_TILES:
        raise ValueError(f"SM120 SDPA bwd: kv_tile must be one of {(0,) + SEQ_KV_TILES} (0 = per-head-dim default); got {params.kv_tile}")
