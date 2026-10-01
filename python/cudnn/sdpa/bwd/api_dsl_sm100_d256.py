# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""cuDNN-frontend adapter of the ``sdpa_bwd_sm100_d256`` row: the SM100 / SM103 (cc 10.0-10.6) d=256 bf16 / fp16 SDPA
backward.

It IS the Rubin half chain (:mod:`cudnn.sdpa.bwd.api_dsl_sm107`: ``delta = rowsum(dO . O)`` -> the main kernel with dV
in TMEM and a kv-major ``[B, H_chunk, S_kv, S_q]`` dS workspace -> dK = dS . Q and dQ = dS^T . K as the
``bprop_matmul_blackwell`` GEMMs at the (256, 256) cluster tile -> the GQA fold; one prepared pointer-host artifact
per plan, ``prepared_sm107.compile_plan`` / ``kernels/sm107/prepared_host.host_f16``) with ONE substitution: the main
kernel is the **2x2-datapath** body ``kernels/bprop_d256_2x2_f16.py`` at ``datapath_2x2_profile = 1``
(``config_d256_2x2``): ``tcgen05.mma.cta_group::2`` with the collective M = 128 -- 64 kv rows per CTA, a 128-row kv
block per cga2 pair -- every operand an SMEM SS operand, 512 non-exclusive TMEM columns, 210 KiB of SMEM.  That
footprint is what lets a bf16 d=256 backward exist under SM100's 227 KiB / 512 columns at all; the 4x1 Rubin body
needs 576 exclusive columns and 327 KiB.

What stays the Rubin row's, by construction: the kernel's launch ABI (``host_f16`` runs it unchanged), the kv WRITE
PAIR of 256 rows (``STAGE3_GRAN_ROWS``: both 128-row blocks of a pair walk the identical q range, so the stage-3 GEMMs'
K-trim stays tight and no mask needs the workspace zero-filled), the S_q / S_kv padding to 128 / 256, the head / batch
chunking, the workspace carve, the standalone ``execute`` and every decline.  The stage-3 tile (256, 256) is passed
EXPLICITLY here -- ``api_dsl_sm107._stage3_cgrp_tile_mn`` keys on the Rubin line by design and is not widened -- and the
stage-3 granularity comes from the 2x2 config's ``kv_pad_rows_2x2`` (the pair, NOT the 128-row block).

Codegen: ``--gpu-arch sm_{sm}a`` of the device the plan is built for (100 / 103; ``prepared_host._check_target`` admits
SM100-SM119 for the half chain), descriptor version 0.
"""

from __future__ import annotations

from cudnn.sdpa.bwd import config_d256_2x2 as _cfg2x2
from cudnn.sdpa.bwd import config_sm107 as _cfg
from cudnn.sdpa.bwd.api_dsl_sm107 import _2X2_KERNEL_FILE, _STAGE3_TILE_D256, SdpaBwdDslSm107

_SM100_D256_TEMPLATE_TAG = "sdpa_bwd_sm100_d256_main"


class SdpaBwdDslSm100D256(SdpaBwdDslSm107):
    """``sdpa_bwd_sm100_d256``: d_qk = d_v = 256, bf16 / fp16, on the SM100 line (cc 10.0-10.6) over the 2x2-datapath body."""

    _FAMILY = _cfg.FAMILY_F16
    _NAME = "sdpa_bwd_sm100_d256"
    _BATCH_CHUNKING = True
    _IO_DTYPES = SdpaBwdDslSm107._IO_DTYPES

    def _datapath_2x2_profile(self) -> int:
        """Profile 1 (one 64-row sub-block per CTA, 128-row kv block, 227 KiB, descriptor version 0) -- NOT the Rubin twin
        constant ``BWD_D256_2X2``: this row has no 4x1 body to fall back to."""
        return _cfg2x2.PROFILE_SM100

    def _kernel_file(self) -> str:
        return _2X2_KERNEL_FILE

    def _template_tag(self) -> str:
        return _SM100_D256_TEMPLATE_TAG

    def _stage3_tile_mn(self, sm: int) -> tuple:
        """The d = 256 cluster tile, explicitly: cluster 2x1, 256 x 256 per pair, the 256-row M tile that equals the kv write pair
        (``_stage3_cgrp_tile_mn`` keys on 107 <= sm <= 119 by design and stays that way)."""
        return _STAGE3_TILE_D256


__all__ = ["SdpaBwdDslSm100D256"]
