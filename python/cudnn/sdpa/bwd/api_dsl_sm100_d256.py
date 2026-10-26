# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""cuDNN-frontend adapter of the ``sdpa_bwd_sm100_d256`` row: the SM100 / SM103 (cc 10.0-10.6) d=256 bf16 / fp16 SDPA
backward, with a native dense D192x128 specialization. Q/K and V/O keep
their actual widths in tensor maps; the delta and grouped-gradient reductions use
separate widths. No head-dimension staging or extra launch is added.

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

from dataclasses import asdict

import torch

from cudnn.sdpa.bwd import api_dsl_sm107 as _sm107
from cudnn.sdpa.bwd import config_d256_2x2 as _cfg2x2
from cudnn.sdpa.bwd import config_sm107 as _cfg
from cudnn.sdpa.bwd import prepared_sm107 as _prepared
from cudnn.sdpa.bwd.api_dsl_sm107 import _2X2_KERNEL_FILE, _STAGE3_TILE_D256, SdpaBwdDslSm107

_SM100_D256_TEMPLATE_TAG = "sdpa_bwd_sm100_d256_main"


class SdpaBwdDslSm100D256(SdpaBwdDslSm107):
    """SM100 half backward: native D256 or dense D192x128 geometry."""

    _FAMILY = _cfg.FAMILY_F16
    _NAME = "sdpa_bwd_sm100_d256"
    _BATCH_CHUNKING = True
    _BATCH_PADDED_BSHD = True
    _THD_SUPPORTED = True
    _THD_ORIGINS_SUPPORTED = True
    _IO_DTYPES = SdpaBwdDslSm107._IO_DTYPES

    # The half row serves per-batch kv lengths on its standalone surface (its body reads ``seq_kv_lens[b]`` under the
    # padded-mask arm); the 2x2 body's padded arm has not run, so this row keeps the decline (a typed error below).
    _PER_BATCH_KV_LENS = False

    def _check_head_dims(self) -> None:
        if (self.head_dim_qk, self.head_dim_v) == (192, 128):
            self._value_error_if(self.thd, f"{self._NAME}: D192x128 backward currently requires dense BSHD storage")
            return
        super()._check_head_dims()

    def _allow_head_subgroups(self) -> bool:
        """The dense BF16 chain folds global dK/dV partials after all head chunks.

        Smaller chunks share their group's single K head in the grouped dQ
        rendering. Keep FP16, packed, per-member dQ and other arch lines on
        their original whole-group choices.
        """
        return not self.thd and self.dtype == torch.bfloat16 and _sm107.DQ_SINGLE_LAUNCH and 100 <= _prepared._sm(self) <= 106

    def _chunk_partials_enabled(self) -> bool:
        """A complete GQA group can be folded before its partial buffer is reused."""
        return (
            not self.thd
            and self.dtype == torch.bfloat16
            and self._gqa_group > 1
            and not self._q_padded
            and not self._kv_padded
            and self._qh_chunk % self._gqa_group == 0
            and (self._b_chunk < self.batch_size or self._qh_chunk < self.h_q)
        )

    def _family_scratch_shapes(self, kv_rows: int, gqa: bool):
        plan = super()._family_scratch_shapes(kv_rows, gqa)
        if self._chunk_partials_enabled():
            return [
                (
                    (name, (self._b_chunk, kv_rows, self._qh_chunk, self.head_dim_v if name == "dv_part" else self.head_dim_qk), dtype)
                    if name in ("dv_part", "dk_part")
                    else (name, shape, dtype)
                )
                for name, shape, dtype in plan
            ]
        return plan

    def check_support(self) -> bool:
        """This row's own declines first (what the 2x2 body has not run), then the half row's backstops."""
        self._value_error_if(
            bool(self.seq_kv_lens_present),
            f"{self._NAME}: padding masks / per-batch kv lengths (seq_kv_lens) are not implemented on the 2x2 body (the cc 10.7 half row serves them)",
        )
        return super().check_support()

    def _datapath_2x2_profile(self) -> int:
        """Profile 1 (one 64-row sub-block per CTA, 128-row kv block, 227 KiB, descriptor version 0) -- NOT the Rubin twin
        constant ``BWD_D256_2X2``: this row has no 4x1 body to fall back to."""
        return _cfg2x2.PROFILE_SM100

    def _template_params(self):
        params = super()._template_params()
        if (self.head_dim_qk, self.head_dim_v) == (192, 128):
            return _cfg2x2.TemplateParams2x2(**asdict(params), head_dim_qk=192, head_dim_v=128)
        return params

    def _kernel_file(self) -> str:
        return _2X2_KERNEL_FILE

    def _template_tag(self) -> str:
        return _SM100_D256_TEMPLATE_TAG

    def _stage3_tile_mn(self, sm: int) -> tuple:
        """The d = 256 cluster tile, explicitly: cluster 2x1, 256 x 256 per pair, the 256-row M tile that equals the kv write pair
        (``_stage3_cgrp_tile_mn`` keys on 107 <= sm <= 119 by design and stays that way)."""
        return _STAGE3_TILE_D256


__all__ = ["SdpaBwdDslSm100D256"]
