# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""K2 -- the per-channel contraction of the triangle multiplicative update as ONE batched GEMM (batch = C * B, M = N = N_res, K = N_res).

Per channel ``c`` (and batch entry ``b``) of the channel-major planes ``a``, ``b`` written by K1 (``[2, C, B * N, n_pad]``, bf16; the
``[N, N]`` matrix of channel ``c`` / batch ``b`` sits at rows ``b * N ..`` with row stride ``n_pad``, the inner index padded to a
multiple of 8 and the pad written as exact zeros):

  outgoing   ``y[c][i, j] = sum_k a[c][i, k] * b[c][j, k]``   -> A = ``a[c]`` K-major, B = ``b[c]`` K-major  (the GEMM's TN form)
  incoming   ``y[c][i, j] = sum_k a[c][k, i] * b[c][k, j]``   -> A = ``a[c]`` M-major, B = ``b[c]`` N-major  (MN-major operands)

Both directions are the SAME storage read through different strides, so the graph below declares the operand majors on the tensor
strides and the FROST GEMM engine's dense template serves either without a copy or a transpose kernel.  The output is the channel-major
``y`` (``[C, B * N, n_pad]``, the ``j`` axis padded like the planes -- K3's ``y_shape``), accumulated in fp32.

The channel and the batch fold into ONE uniform GEMM batch: block ``(c, b)`` is at element offset ``(c * B + b) * N * n_pad`` in every
plane, so the kernel sees ``C * B`` batches of stride ``N * n_pad`` (:func:`contraction_geometry`).  The pad plays two roles:

* **outgoing** -- the padded ``k`` axis is the GEMM's K.  Declaring ``K = n_pad`` is exact (the pad is zero in BOTH operands) and
  satisfies the engine's TMA rule that a K-major operand's K extent is a multiple of 8 elements (16 bytes); ``M`` and ``N`` stay ``N_res``
  (TMA zero-fills the loads and clips the stores past them).
* **incoming** -- the contiguous axis of both operands is the ``i`` / ``j`` extent ``N_res`` itself (an M-major A, an N-major B), so the
  same TMA rule needs ``N_res % 8 == 0``; :func:`incoming_supported` is the typed decline the API raises otherwise (a transposed-store
  K1 arm lifts it later).

The graph is a plain ``cudnn.pygraph`` matmul with a 3-D batch: it reaches the ``frost_gemm`` engine through the ordinary graph
lifecycle, and :func:`cudnn.gemm.frost.compiler.jit_from_cudnn_graph` compiles it directly for a pinned tile config
(:func:`tile_config_for`: the per-N pick of the measured prototype; :data:`DESIGN_TILE_CONFIG` where nothing was measured).  The
kernel-order operand views the compiled instance takes are :func:`k2_operand_views` (planes) / :func:`kernel_operand_views` (dense).

Byte / FLOP model (the roofline every measurement of this stage is read against): the compulsory traffic reads ``a`` and ``b`` once and
writes ``y`` once -- :func:`contraction_bytes` = ``6 N^2 C`` at 2 B per element -- and :func:`contraction_flop` = ``2 C N^3``; the
arithmetic intensity is ``N / 3`` FLOP per byte, so the stage is bandwidth-bound up to ``N ~ 3 x (peak FLOP/s) / (HBM B/s)``.
"""

from __future__ import annotations

from typing import Optional, Tuple

DIRECTIONS = ("outgoing", "incoming")

#: The default tile config of the contraction (DESIGN_TILE_CONFIG): a 2-CTA (cta_group::2) pair computing a 256 x 256 output tile (128 x 256
#: per CTA), 128-byte K tiles (64 bf16 elements), cluster 2x1 (no operand multicast).  The small-N alternative is the 1-CTA 128 x 128 tile.
DESIGN_TILE_CONFIG = "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma"
SMALL_N_TILE_CONFIG = "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma"
#: The 2-CTA 256x256 pair tile with the A operand multicast across 2 / 4 CTAs along the output's N axis (cluster 2x2 / 2x4).
MULTICAST_A2_TILE_CONFIG = "CONFIG_sm100_128x256x128_128x256x32_cluster2x2_2ctamma"
MULTICAST_A4_TILE_CONFIG = "CONFIG_sm100_128x256x128_128x256x32_cluster2x4_2ctamma"
#: The 1-CTA 128 x 128 tile with the B operand multicast across the 2 CTAs of a cluster 2x1.
SMALL_N_MULTICAST_B2_TILE_CONFIG = "CONFIG_sm100_128x128x128_128x128x32_cluster2x1_1ctamma"

# The per-N tile picks live in ``cudnn.triangle_multiplicative_update.heuristics`` (the measured table, by hidden width and batch);
# :func:`tile_config_for` reads them from there.

#: A TMA-fed operand's CONTIGUOUS extent must be a multiple of 8 bf16 elements (16 bytes) -- the GEMM engine's rule; the planes' inner
#: pad satisfies it for the outgoing direction, the incoming direction needs ``N_res`` itself to.
CONTIGUOUS_EXTENT_MULTIPLE = 8

_BF16_BYTES = 2


def operand_majors(direction: str) -> Tuple[str, str]:
    """``(a_major, b_major)`` of the GEMM operands for one direction: ``("k", "k")`` outgoing, ``("m", "n")`` incoming."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    return ("k", "k") if direction == "outgoing" else ("m", "n")


def n_pad(n_res: int) -> int:
    """The padded inner extent of the channel-major planes (a multiple of 8: a 16-byte-aligned TMA row stride) -- K1's rule."""
    return -(-int(n_res) // CONTIGUOUS_EXTENT_MULTIPLE) * CONTIGUOUS_EXTENT_MULTIPLE


def incoming_supported(n_res: int) -> bool:
    """The incoming direction reads MN-major operands whose contiguous extent is ``N_res``: the TMA rule needs it to be a multiple of 8."""
    return int(n_res) % CONTIGUOUS_EXTENT_MULTIPLE == 0


def contraction_geometry(channels: int, batch: int, n_res: int) -> Tuple[int, int, int]:
    """``(batch_count, block_stride, n_pad)`` of the folded GEMM batch over the planes: ``C * B`` blocks of ``N * n_pad`` elements."""
    npd = n_pad(n_res)
    return int(channels) * int(batch), int(n_res) * npd, npd


def k_extent(n_res: int, direction: str) -> int:
    """The GEMM's K extent: the padded ``n_pad`` for the outgoing direction (the pad is zero in both operands), ``N_res`` for incoming."""
    operand_majors(direction)
    return n_pad(n_res) if direction == "outgoing" else int(n_res)


def operand_strides(n_res: int, direction: str, *, batch: int = 1, padded: bool = False) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """The strides of the graph tensors ``A [C*B, M, K]`` and ``B [C*B, K, N]`` that spell the operand majors over the channel-major storage.

    ``padded=False`` (the dense ``[C, N, N]`` planes of the prototype): block stride ``N^2``, row stride ``N``.  ``padded=True`` (K1's
    planes): block stride ``N * n_pad``, row stride ``n_pad``.  ``batch`` only names the fold (the strides are per block either way)."""
    a_major, b_major = operand_majors(direction)
    n_res = int(n_res)
    row = n_pad(n_res) if padded else n_res
    block = n_res * row
    a_stride = (block, row, 1) if a_major == "k" else (block, 1, row)
    b_stride = (block, 1, row) if b_major == "k" else (block, row, 1)
    return a_stride, b_stride


def contraction_graph(cudnn_module, channels: int, n_res: int, direction: str, *, batch: int = 1, padded: bool = False, io_data_type=None):
    """The batched-matmul graph of the contraction: ``A [C*B, N, K]`` and ``B [C*B, K, N]`` with the direction's strides, ``Y`` bf16.

    ``cudnn_module`` is the imported ``cudnn`` package (passed in so this module stays import-cheap).  The io dtype defaults to bf16;
    intermediate / compute dtypes are fp32.  ``padded=True`` declares K1's planes (``K = n_pad`` outgoing; incoming needs ``N % 8 == 0``).
    Returns the graph; the operand tensors are its ``A`` / ``B`` names, the output ``Y``.
    """
    io = io_data_type if io_data_type is not None else cudnn_module.data_type.BFLOAT16
    if padded and direction == "incoming" and not incoming_supported(n_res):
        raise ValueError(
            f"the incoming contraction reads MN-major operands whose contiguous extent is N = {n_res}; the GEMM engine's TMA rule needs a "
            f"multiple of {CONTIGUOUS_EXTENT_MULTIPLE} (16 bytes of bf16)"
        )
    g = cudnn_module.pygraph(io_data_type=io, intermediate_data_type=cudnn_module.data_type.FLOAT, compute_data_type=cudnn_module.data_type.FLOAT)
    a_stride, b_stride = operand_strides(n_res, direction, batch=batch, padded=padded)
    k = k_extent(n_res, direction) if padded else int(n_res)
    cb = int(channels) * int(batch)
    a = g.tensor(name="A", dim=[cb, int(n_res), k], stride=list(a_stride))
    b = g.tensor(name="B", dim=[cb, k, int(n_res)], stride=list(b_stride))
    y = g.matmul(A=a, B=b, name="Y")
    y.set_output(True)
    return g


def kernel_operand_views(a, b, direction: str):
    """The operand views a compiled FROST instance of :func:`contraction_graph` takes, in kernel axis order (``A [batch, M, K]``,
    ``B [batch, N, K]``) with the declared contiguity: the channel-major tensors as stored for the outgoing direction, their transposed
    views (no copy) for the incoming one."""
    operand_majors(direction)
    if direction == "outgoing":
        return a, b
    return a.transpose(1, 2), b.transpose(1, 2)


def k2_slack_bytes(n_res: int, elem_bytes: int = _BF16_BYTES) -> int:
    """Bytes the caller must keep readable right after the ``b`` plane: the outgoing GEMM at ``N % 8 != 0`` reads the ``B`` operand as
    ``n_pad`` rows per block (:func:`k2_operand_views`), so the LAST block over-reads ``(n_pad - N) * n_pad`` elements past the plane."""
    npd = n_pad(n_res)
    return (npd - int(n_res)) * npd * elem_bytes


def k2_operand_views(plane_a, plane_b, y, *, channels: int, batch: int, n_res: int, direction: str):
    """The kernel-order views over K1's planes and K3's ``y`` (all ``[C, B * N, n_pad]`` contiguous): ``(A, B, Y)`` for the compiled instance.

    ``A`` is ``[C*B, M = N, K]`` and ``B`` ``[C*B, N', K]`` with the direction's contiguity (outgoing: ``K = n_pad`` contiguous; incoming:
    the transposed views, ``M`` / ``N`` contiguous), ``Y`` the ``[C*B, N, N']`` view of ``y`` with row stride ``n_pad``.  No copy, no allocation.

    ``N' = n_pad`` for the outgoing direction (``N' = N`` where ``N % 8 == 0``): the engine stores its output in 16-byte vectors and needs the
    output's contiguous extent to be a multiple of 8 elements, so the GEMM's N covers the padded columns -- ``B`` is read as ``n_pad`` ROWS
    per block (the extra rows are the NEXT block's first rows, or the workspace slack after the last block: :func:`k2_slack_bytes`) and
    their products land ONLY in the pad columns ``j in [N, n_pad)`` of ``y``, which K3 never consumes (its row statistics and stores cover
    ``j < N``; its tests prove it with NaN-filled pad columns).  The valid output ``y[:, :, :N]`` is the exact contraction over ``K = n_pad``
    (the pad is zero in both planes)."""
    operand_majors(direction)
    if direction == "incoming" and not incoming_supported(n_res):
        raise ValueError(f"the incoming contraction needs N % {CONTIGUOUS_EXTENT_MULTIPLE} == 0, got N={n_res}")
    cb, block, npd = contraction_geometry(channels, batch, n_res)
    n_res = int(n_res)
    a3 = plane_a.view(cb, n_res, npd)
    if direction == "incoming":
        b3 = plane_b.view(cb, n_res, npd)
        y3 = y.view(cb, n_res, npd)  # n_pad == n_res here
        return a3.transpose(1, 2), b3.transpose(1, 2), y3
    y3 = y.view(cb, n_res, npd)
    if npd == n_res:
        return a3, plane_b.view(cb, n_res, npd), y3
    flat = plane_b.view(-1)
    need = (cb - 1) * block + npd * npd  # elements the over-read view spans from the plane's first element
    room = flat.untyped_storage().nbytes() // flat.element_size() - flat.storage_offset()
    if room < need:
        raise ValueError(
            f"the outgoing contraction at N={n_res} reads the b plane as {npd} rows per block: its storage must extend k2_slack_bytes(N) = "
            f"{k2_slack_bytes(n_res)} bytes past the plane (the API's workspace does; a standalone plane needs that slack allocated after it)"
        )
    b3 = flat.as_strided((cb, npd, npd), (block, npd, 1))  # n_pad rows per block: the over-read of the docstring
    return a3, b3, y3


def reference_views(a, b, direction: str):
    """The ``(A, B)`` views such that ``y = A @ B`` (``torch.bmm``) is the contraction: outgoing ``(a, b^T)``, incoming ``(a^T, b)``."""
    operand_majors(direction)
    if direction == "outgoing":
        return a, b.transpose(1, 2)
    return a.transpose(1, 2), b


def tile_config_for(n_res: int, override: Optional[str] = None, *, channels: int = 128, batch: int = 1, direction: str = "outgoing") -> str:
    """The contraction's tile config name for ``(N_res, c, B, direction)``: the measured per-geometry pick of the heuristics table (keyed by
    hidden width, N bucket, batch bucket and direction); ``override`` names a catalog config instead (a performance knob: every config computes
    the same function; the API validates the name in ``check_support``)."""
    if override is not None:
        return str(override)
    from ..heuristics import k2_tile_for

    return k2_tile_for(int(n_res), int(channels), int(batch), direction)


def contraction_bytes(channels: int, n_res: int, elem_bytes: int = _BF16_BYTES, *, batch: int = 1) -> int:
    """Compulsory HBM traffic of the stage: ``a`` and ``b`` read once, ``y`` written once (``6 N^2 C`` at bf16, per batch entry)."""
    return 3 * int(batch) * channels * n_res * n_res * elem_bytes


def contraction_flop(channels: int, n_res: int, *, batch: int = 1) -> int:
    """``2 C N^3`` multiply-adds counted as two FLOP each (per batch entry)."""
    return 2 * int(batch) * channels * n_res**3


__all__ = [
    "CONTIGUOUS_EXTENT_MULTIPLE",
    "DESIGN_TILE_CONFIG",
    "DIRECTIONS",
    "MULTICAST_A2_TILE_CONFIG",
    "MULTICAST_A4_TILE_CONFIG",
    "SMALL_N_MULTICAST_B2_TILE_CONFIG",
    "SMALL_N_TILE_CONFIG",
    "contraction_bytes",
    "contraction_flop",
    "contraction_geometry",
    "contraction_graph",
    "incoming_supported",
    "k2_operand_views",
    "k2_slack_bytes",
    "k_extent",
    "kernel_operand_views",
    "n_pad",
    "operand_majors",
    "operand_strides",
    "reference_views",
    "tile_config_for",
]
