# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FROST kernels of the gated attention block -- a FLAT package, Rubin (``sm_107a``) only.

The block targets one arch, so there is no arch sub-package here: arch is the
ONLY axis that earns a directory (engine contract § 8) and a single arch earns
none.  Every other axis -- stage, dtype family -- stays in the filename.  By
stage of the pipeline in ``api.py``::

    proj_gemm.py                stages (1) and (6): plan records + runners over the
                                RENDERED FROST GEMM template (writes no kernel)
    proj_gemm_norm_rope.py      stage (1) with (2)+(3) fused into its epilogue -- a
                                bf16 FORK of the rendered GEMM template
    proj_gemm_norm_rope_fp8.py  its FP8 twin: (1)+(2)+(3)+quantize, compact e4m3
                                q8 / k8 / v8 + bf16 gate16 out -- also a fork
    qk_norm_rope.py             stages (2)+(3), per-lane LDG pipeline
    qk_norm_rope_tma.py         stages (2)+(3), TMA-staged A/B of the above
    quantize.py                 (3q) / (5q): per-tensor e4m3 quantize pass (unfused FP8)
    elementwise.py              stage (5) sigmoid gate, and (3b) V compaction

**The SDPA stage owns no file here.**  It drives the shipped forward adapter
``cudnn.sdpa.fwd.api_dsl.SdpaFwdDslSm100`` in EVERY configuration; the
sigmoid-gate epilogue behind the block's ``fuse_gate`` is a production feature
of ``sdpa/fwd/kernels/sm107/prefill_d256_{f16,fp8}.py`` behind
``TemplateParams.epilogue_gate`` (engine rows: ``epilogue_gate_d_shapes``),
reached through ``SdpaFwdDslSm100(sample_gate=...)`` / ``execute(gate=...)``.

``proj_gemm_norm_rope*.py`` are forks of the rendered GEMM template BY DECISION
(their epilogues are the block's own norm / RoPE / quantize math on the GEMM
accumulator, not a feature the GEMM engine serves).  The two SDPA forks that
preceded the production ``epilogue_gate`` were deleted when it landed.
"""
