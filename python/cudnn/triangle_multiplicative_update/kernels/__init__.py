# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernels of the triangle multiplicative update: ``trimul_proj`` (K1, the LayerNorm-prologue gated dual projection on the
``cudnn.gemm.frost.ln_prologue`` substrate -> the channel-major planes ``a``, ``b``), ``contraction`` (K2, the batched
GEMM contract on the FROST GEMM engine), ``trimul_tail`` (K3, the accumulators arm of the same substrate: LayerNorm of ``y`` and ``z``,
the output projection gated by the output gate -> the pair-major update)."""
