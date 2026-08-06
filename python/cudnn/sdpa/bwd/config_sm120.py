# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Compile-time configuration for the FROST SM120 SDPA backward template."""

from __future__ import annotations

from dataclasses import dataclass

from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_FP16

SEQ_Q_TILES = (64, 128)
SEQ_KV_TILES = (64, 128)
SUPPORTED_HEAD_DIMS = (32, 64, 128)


@dataclass(frozen=True)
class TemplateParams:
    """Per-graph parameters that change the traced SM120 backward kernel.

    Tensor geometry deliberately stays out of this record. Scalar dimensions
    are inputs to the template module's per-shape ``compile()`` cache, while
    strides follow the fixed compact-BSHD kernel contract. This frozen record
    identifies the import-time specialization shared by all compatible shapes.
    """

    dtype_qkv: int = DTYPE_FP16
    is_causal: bool = False
    causal_top_left: bool = False
    use_pdl: bool = True
    q_tile: int = 0
    kv_tile: int = 0


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
    if params.q_tile not in (0,) + SEQ_Q_TILES:
        raise ValueError(f"SM120 SDPA bwd: q_tile must be one of {(0,) + SEQ_Q_TILES} (0 = per-head-dim default); got {params.q_tile}")
    if params.kv_tile not in (0,) + SEQ_KV_TILES:
        raise ValueError(f"SM120 SDPA bwd: kv_tile must be one of {(0,) + SEQ_KV_TILES} (0 = per-head-dim default); got {params.kv_tile}")
