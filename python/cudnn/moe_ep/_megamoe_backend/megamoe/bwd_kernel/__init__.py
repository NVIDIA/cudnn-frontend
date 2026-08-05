# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Backward (dgrad) MegaMoE kernel package — see megamoe/BWD_DESIGN.md.

M2.A: backward variant of the standalone Sm100SwigluMxfp8Fc12Kernel
(gemm1 dA = dout@W2^T -> SwiGLU-bwd epilogue vs the fc1_c stash ->
gemm2 [dg,du]@W13^T).  M2.B grafts it into the mega kernel (metadata-
driven dout gather + token-back dx).
"""
