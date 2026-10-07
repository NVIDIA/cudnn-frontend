# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""K2 -- the per-channel contraction of the triangle multiplicative update as ONE batched GEMM (batch = C, M = N = K = N_res).

Per channel ``c`` of the channel-major planes ``a``, ``b`` (``[C, N, N]``, bf16):

  outgoing   ``y[c][i, j] = sum_k a[c][i, k] * b[c][j, k]``   -> A = ``a[c]`` K-major, B = ``b[c]`` K-major  (the GEMM's TN form)
  incoming   ``y[c][i, j] = sum_k a[c][k, i] * b[c][k, j]``   -> A = ``a[c]`` M-major, B = ``b[c]`` N-major  (MN-major operands)

Both directions are the SAME storage read through different strides, so the graph below declares the operand majors on the tensor
strides and the FROST GEMM engine's dense template serves either without a copy or a transpose kernel.  The output is the channel-major
``y`` (``[C, N, N]``, row-major, bf16), accumulated in fp32.

The graph is a plain ``cudnn.pygraph`` matmul with a 3-D batch: it reaches the ``frost_gemm`` engine through the ordinary graph
lifecycle, and :func:`cudnn.gemm.frost.compiler.jit_from_cudnn_graph` compiles it directly for a pinned tile config
(:data:`DESIGN_TILE_CONFIG`).  The kernel-order operand views the compiled instance takes are :func:`kernel_operand_views`.

Byte / FLOP model (the roofline every measurement of this stage is read against): the compulsory traffic reads ``a`` and ``b`` once and
writes ``y`` once -- :func:`contraction_bytes` = ``6 N^2 C`` at 2 B per element -- and :func:`contraction_flop` = ``2 C N^3``; the
arithmetic intensity is ``N / 3`` FLOP per byte, so the stage is bandwidth-bound up to ``N ~ 3 x (peak FLOP/s) / (HBM B/s)``.
"""

from __future__ import annotations

from typing import Tuple

DIRECTIONS = ("outgoing", "incoming")

#: The design tile config of the contraction: a 2-CTA (cta_group::2) pair computing a 256 x 256 output tile (128 x 256 per CTA),
#: 128-byte K tiles (64 bf16 elements), cluster 2x1 (no operand multicast).  The small-N alternative is the 1-CTA 128 x 128 tile.
DESIGN_TILE_CONFIG = "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma"
SMALL_N_TILE_CONFIG = "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma"

_BF16_BYTES = 2


def operand_majors(direction: str) -> Tuple[str, str]:
    """``(a_major, b_major)`` of the GEMM operands for one direction: ``("k", "k")`` outgoing, ``("m", "n")`` incoming."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    return ("k", "k") if direction == "outgoing" else ("m", "n")


def operand_strides(n_res: int, direction: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """The strides of the graph tensors ``A [C, M, K]`` and ``B [C, K, N]`` that spell the operand majors over the channel-major storage."""
    a_major, b_major = operand_majors(direction)
    nn = n_res * n_res
    a_stride = (nn, n_res, 1) if a_major == "k" else (nn, 1, n_res)
    b_stride = (nn, 1, n_res) if b_major == "k" else (nn, n_res, 1)
    return a_stride, b_stride


def contraction_graph(cudnn_module, channels: int, n_res: int, direction: str, *, io_data_type=None):
    """The batched-matmul graph of the contraction: ``A [C, N, N]`` and ``B [C, N, N]`` with the direction's strides, ``Y`` bf16 row-major.

    ``cudnn_module`` is the imported ``cudnn`` package (passed in so this module stays import-cheap).  The io dtype defaults to bf16;
    intermediate / compute dtypes are fp32.  Returns the graph; the operand tensors are its ``A`` / ``B`` names, the output ``Y``.
    """
    io = io_data_type if io_data_type is not None else cudnn_module.data_type.BFLOAT16
    g = cudnn_module.pygraph(io_data_type=io, intermediate_data_type=cudnn_module.data_type.FLOAT, compute_data_type=cudnn_module.data_type.FLOAT)
    a_stride, b_stride = operand_strides(n_res, direction)
    a = g.tensor(name="A", dim=[channels, n_res, n_res], stride=list(a_stride))
    b = g.tensor(name="B", dim=[channels, n_res, n_res], stride=list(b_stride))
    y = g.matmul(A=a, B=b, name="Y")
    y.set_output(True)
    return g


def kernel_operand_views(a, b, direction: str):
    """The operand views a compiled FROST instance of :func:`contraction_graph` takes, in kernel axis order (``A [C, M, K]``,
    ``B [C, N, K]``) with the declared contiguity: the channel-major tensors as stored for the outgoing direction, their transposed
    views (no copy) for the incoming one."""
    operand_majors(direction)
    if direction == "outgoing":
        return a, b
    return a.transpose(1, 2), b.transpose(1, 2)


def reference_views(a, b, direction: str):
    """The ``(A, B)`` views such that ``y = A @ B`` (``torch.bmm``) is the contraction: outgoing ``(a, b^T)``, incoming ``(a^T, b)``."""
    operand_majors(direction)
    if direction == "outgoing":
        return a, b.transpose(1, 2)
    return a.transpose(1, 2), b


def contraction_bytes(channels: int, n_res: int, elem_bytes: int = _BF16_BYTES) -> int:
    """Compulsory HBM traffic of the stage: ``a`` and ``b`` read once, ``y`` written once (``6 N^2 C`` at bf16)."""
    return 3 * channels * n_res * n_res * elem_bytes


def contraction_flop(channels: int, n_res: int) -> int:
    """``2 C N^3`` multiply-adds counted as two FLOP each."""
    return 2 * channels * n_res**3
