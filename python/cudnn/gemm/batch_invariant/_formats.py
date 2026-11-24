# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from FlashInfer batch_invariant/spec.py (e2d163fb3d46).

"""Numeric formats and operand alignment for batch-invariant GEMM."""

FP8_FMTS = ("e4m3", "e5m2")
#: The block-scale fp8 class (module docstring): e4m3 elements, fp32 scales per 1x128 block of A and per 128x128 block of B,
#: the scales applied by the kernel's promotion pass once per K block (no instruction operand).
FP8_BLOCKSCALE_FMTS = ("e4m3_bs",)
BLOCK_K = 128  # elements of K per scale block: the class constant (scale_granularity_mnk = (1, 128, 128))
SCALE_BLOCK_N = 128  # output columns per weight scale block
NVFP4_FMTS = ("nvfp4", "nvfp4_k128")
MXFP4_FMTS = ("mxfp4", "mxfp4_k128")
MXFP8_FMTS = ("mxfp8", "mxfp8_k64")
MX_MIXED_FMTS = ("mxfp4w8a", "mxfp4w8a_k64")
MX_FMTS = MXFP8_FMTS + MXFP4_FMTS + MX_MIXED_FMTS  # the UE8M0 formats
FP4_FMTS = NVFP4_FMTS + MXFP4_FMTS
BLOCK_SCALED = frozenset(MX_FMTS + NVFP4_FMTS)  # formats whose step takes scale-factor operands (the leaf carries them)
ALPHA_GIVEN = frozenset(FP4_FMTS + MX_MIXED_FMTS)  # formats whose chain scalar is the caller's alpha, not the product of two per-tensor scales
MMA_K_OF = {
    "bf16": 16,
    "e4m3": 32,
    "e5m2": 32,
    "mxfp8": 32,
    "mxfp8_k64": 64,
    "nvfp4": 64,
    "nvfp4_k128": 128,
    "mxfp4": 64,
    "mxfp4_k128": 128,
    "mxfp4w8a": 32,
    "mxfp4w8a_k64": 64,
    "e4m3_bs": 32,
}
ELEM_BITS_OF = {
    "bf16": 16,
    "e4m3": 8,
    "e5m2": 8,
    "mxfp8": 8,
    "mxfp8_k64": 8,
    "nvfp4": 4,
    "nvfp4_k128": 4,
    "mxfp4": 4,
    "mxfp4_k128": 4,
    "mxfp4w8a": 8,
    "mxfp4w8a_k64": 8,
    "e4m3_bs": 8,
}
#: Per-operand element widths: the mixed classes take 8-bit activations (A) against 4-bit weights (B); every other format
#: is symmetric.
A_ELEM_BITS_OF = dict(ELEM_BITS_OF)
B_ELEM_BITS_OF = {**ELEM_BITS_OF, "mxfp4w8a": 4, "mxfp4w8a_k64": 4}
SF_VEC = {
    "mxfp8": 32,
    "mxfp8_k64": 32,
    "nvfp4": 16,
    "nvfp4_k128": 16,
    "mxfp4": 32,
    "mxfp4_k128": 32,
    "mxfp4w8a": 32,
    "mxfp4w8a_k64": 32,
}  # elements per block scale
SF_ATOM_ROWS, SF_ATOM_KBLOCKS = 128, 4


def chunk_unit(in_fmt: str) -> int:
    """The K granule of the fixed-chunk rule (``schedule.chunks_for``): a chunk is a whole number of MMA steps and, for the
    block-scaled formats, a whole number of scale-factor atom columns (the 128x4 layout stores 4 consecutive K-blocks of
    ``SF_VEC`` elements together, so a chunk boundary must fall on a ``4 * SF_VEC`` element boundary): the lcm of the two.
    16 / 32 for bf16 / fp8, 128 for the MX classes (4 x 32 elements: 4 steps of the sm_100 instruction, 2 of the sm_107 one)."""
    import math

    mk = MMA_K_OF[in_fmt]
    if in_fmt not in BLOCK_SCALED:
        return mk
    atom = 4 * SF_VEC[in_fmt]
    return mk * atom // math.gcd(mk, atom)


MMA_RATE_OF = {
    "bf16": 1.0,
    "e4m3": 2.0,
    "e5m2": 2.0,
    "mxfp8": 2.0,
    "mxfp8_k64": 2.0,
    "nvfp4": 4.0,
    "nvfp4_k128": 4.0,
    "mxfp4": 4.0,
    "mxfp4_k128": 4.0,
    "mxfp4w8a": 2.0,  # the mixed step runs on the ``kind::mxf8f6f4`` pipeline: the 8-bit kind's rate
    "mxfp4w8a_k64": 2.0,
    "e4m3_bs": 2.0,
}


def k_granule(in_fmt: str) -> int:
    """The K granule of the body's operand rows: a whole number of block scales and of 16-byte TMA rows of the narrower
    operand (32 elements for every block-scaled format), or 128 elements for the sm_100 mixed class, whose e2m1 weight the
    TMA unpacks into the instruction's 8-bit container in 64-byte (128-element) runs of the global row: the K extent and
    the row stride must be whole runs (``blocks/blockscaled_mx_mixed.py``)."""
    import math

    if in_fmt == "mxfp4w8a":
        return 128
    row = 128 // min(A_ELEM_BITS_OF[in_fmt], B_ELEM_BITS_OF[in_fmt])  # elements per 16-byte row of the narrower operand
    vec = SF_VEC.get(in_fmt, MMA_K_OF[in_fmt])
    return vec * row // math.gcd(vec, row)
