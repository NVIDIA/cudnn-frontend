# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""The CuTe DSL kernels of CuteMoK: the BF16 and MXFP8 forward / backward megakernels
(``persistent_bf16_fwd`` / ``persistent_bf16_bwd`` / ``persistent_mxfp8_fwd`` /
``persistent_mxfp8_bwd``), their epilogues, and the scheduling and communication operators.

The kernels require CuTe DSL >= 4.8.0 (its cu13 build), checked on first use (``_version_gate``); the few
QuACK 0.5.0 helpers they build on are adapted in ``_compat``.  ``cudnn.moe_ep._cutemok.functional`` and ``cudnn.moe_ep._cutemok.ops`` import
the kernel modules on first use; importing this package imports none of them.
"""
