# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triangle multiplicative update (AlphaFold-class "outgoing" / "incoming" triangle multiplication) on the FROST kernels.

Three launches, one engine (forward):

  K1  ``kernels.trimul_proj``   LayerNorm over the channel + the gated dual projection (value || gate) + sigmoid * mul + mask
                                -> the channel-major planes ``a``, ``b`` of shape ``[C, N, N]``
  K2  ``kernels.contraction``   the per-channel contraction as ONE batched GEMM on the FROST GEMM engine (batch = C, K = N):
                                outgoing ``y[c] = a[c] @ b[c]^T``, incoming ``y[c] = a[c]^T @ b[c]`` -> channel-major ``y``
  K3  ``kernels.trimul_tail``   LayerNorm over the channel of ``y`` + the output gate (sigmoid of a projection of the LayerNormed input)
                                * the output projection -> the pair-major update

The channel-major planes are the contract between the stages: K2 needs no kernel of its own -- the dense FROST GEMM template serves
the batched matmul graph directly (``contraction.contraction_graph``), so the contraction inherits every tile config, cluster shape and
epilogue of that engine.  The public API (cuEquivariance's ``triangle_multiplicative_update`` signature) and the fused K1 / K3 kernels
land in later commits; this package currently carries the K2 contract and its byte / FLOP model.
"""

__all__ = []
