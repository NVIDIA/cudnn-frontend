# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""cuDNN-frontend adapter of the ``sdpa_bwd_sm107_d512`` row: the cc 10.7 (the Rubin-line, 107 <= SM <= 119) d in
(256, 512] bf16 / fp16 SDPA backward.

It IS the SM100 large-head-dim chain (:class:`cudnn.sdpa.bwd.api_dsl.SdpaBwdDslSm100`: ``delta = rowsum(dO . O)`` -> the
stage-2 kernel writing the S / dS ``[B, H_chunk, S_q, S_kv]`` workspaces -> dV = S^T . dO, dK = dS^T . Q, dQ = dS . K as the
``bprop_matmul_blackwell`` GEMMs at the (512, 512) cluster tile -> the GQA fold; one prepared pointer-host artifact per plan,
``prepared_sm100.compile_plan`` / ``kernels/sm100/prepared_host``) with ONE substitution: stage 2 is ALWAYS the
**2x2-datapath** body ``kernels/sm107/bprop_d512_f16_2x2.py`` at the cc 10.7 ring arm (``stages_kv=8, cast_stages=2`` under
the 325 KiB usable line -- 320 KiB of slabs, descriptor version 0 at zero margin), regardless of the SM100 row's
``api_dsl.STAGE2_2X2`` module constant: the 4x1 role split was never run on this line, the twin was (first launches
2026-10-01 on a cc 10.7 board: dense / causal / bottom-right / SWA / right band / GQA / non-tile / S=8K against the fp32
reference), and the deeper ring is the design's reason for a cc 10.7 row at all.

What stays the SM100 row's, by construction: the head-chunk budget and the workspace carve, the stage-3 renderings
(``mm_lo`` / ``mm_hi`` at ``causal_gran = CLUSTER_Q_ROWS = 256``) and the causal zero-fill, ``scratch_workspace_bytes``,
the standalone ``execute``.  What this row declines on day one, each flipped together with a board-run accept test and a
tracker line (the ``sdpa_bwd_sm107`` growth pattern): THD / ragged (the SM100 row serves it; the cc 10.7 board has not run
it -- and the packed path's multi-chunk correctness is open, Track B item B0) and ``dense_flex`` staging layouts (BSHD-physical
io only).

Rule 7: the public DSL 4.7.0 lacks ``sm_107a``; :meth:`check_support` declines through ``cutedsl_arch_requirement_error``
with the installed version BEFORE the SM100 backstops run.  That gate knows cc 10.7 (the one part of the line that exists
today); on a later Rubin-line part whose target the installed DSL lacks, the DSL's own target check declines inside
``compile`` -- the same shape as the ``sdpa_bwd_sm107`` d256 rows.  Codegen ``--gpu-arch sm_{sm}a`` of the device the plan is built
for (``prepared_host.compile_host`` admits 107..119 as a range); artifact symbol ``frost_sdpa_bwd_sm107_d512_prepared``.
"""

from __future__ import annotations

from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm100

# The stage-2 file this row renders (relative to ``kernels/``, loaded through ``api_dsl._sm100_kernel_path``), and the
# template-module cache tag of its rendering (the fwd fork's spelling: ``sdpa_fwd_sm107_f16_d512_2x2``).
_SM107_STAGE2_FILE_2X2 = "sm107/bprop_d512_f16_2x2.py"
_SM107_STAGE2_TAG_2X2 = "sdpa_bwd_sm107_stage2_2x2"


class SdpaBwdDslSm107D512(SdpaBwdDslSm100):
    """``sdpa_bwd_sm107_d512``: d in (256, 512], bf16 / fp16, on the Rubin-line (cc 10.7 - 11.9) over the 2x2-datapath stage 2."""

    _NAME = "sdpa_bwd_sm107_d512"
    _STAGE2_TAG = _SM107_STAGE2_TAG_2X2

    # --- capability backstop -------------------------------------------------
    def check_support(self) -> bool:
        """Rule 7 first (a DSL without ``sm_107a`` declines by version, never inside the DSL), then this row's day-one
        declines, then the SM100 chain's backstops."""
        from cudnn.frost.buffers import cutedsl_arch_requirement_error
        from cudnn.frost.device import compute_capability, resolve_device

        error = cutedsl_arch_requirement_error(compute_capability(resolve_device(self.q_desc.device)))
        if error:
            raise NotImplementedError(error)
        n = self._NAME
        # Day-one declines (plan A4): each flips together with its board-run accept test and tracker line.
        self._value_error_if(self.thd, f"{n}: THD / ragged is not implemented on this row yet (the SM100 row's packed path has not run on cc 10.7)")
        self._value_error_if(
            bool(self._stage_in or self._stage_out),
            f"{n}: {', '.join(self._stage_in + self._stage_out)} must be BSHD-physical (stride order 3,1,2,0); the staging path is not claimed on this row",
        )
        return super().check_support()

    # --- stage-2 template selection -------------------------------------------
    def _stage2_file(self) -> str:
        """Always the cc 10.7 2x2 sibling file -- never the 4x1 role split, whatever ``api_dsl.STAGE2_2X2`` says."""
        return _SM107_STAGE2_FILE_2X2

    def _stage2_record(self, stage2_fields: dict):
        """Always the 8-stage ring arm (``kernels/sm107/bprop_d512_f16_2x2.py::RUBIN_ARM``, the ONE spelling: the kernel
        ``_require``s the same three values on the config it builds, so the two cannot drift apart).

        The record's other levers keep the SM100 twin's defaults, MEASURED on the board (2026-10-01, whole-backward CUDA-event
        medians, A/B/A x3 at B=1 H=128 S=8192 d=512 bf16 dense, control spread 0.08 %): ``stages_acc=4`` +0.04 % (noise);
        ``kv_share=1`` (pair-local K / V, no cross-pair multicast) 0.35-0.42 % FASTER (median -0.36 %) -- real but below any
        flip threshold, and it would break the fork's PTX identity with the SM100 body that the review surface rests on; a
        shared lever for both siblings if it is ever pursued (Rule 9), not a cc 10.7 delta."""
        from cudnn.sdpa.bwd.config_sm100 import SM107_USABLE_DYN_SMEM_2X2, TemplateParams2x2

        return TemplateParams2x2(**stage2_fields, stages_kv=8, cast_stages=2, smem_cap_bytes=SM107_USABLE_DYN_SMEM_2X2)


__all__ = ["SdpaBwdDslSm107D512"]
