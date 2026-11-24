# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Quantized instruction widths, fixed K folds, and FP32 epilogue operations."""

from dataclasses import dataclass
from typing import Optional, Tuple

import cutlass.cute as cute
from cutlass import Float32
from cutlass.cute.math import RoundingMode

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


def sf_bytes(rows: int, k: int, in_fmt: str = "mxfp8") -> int:
    """Bytes of the 128x4 swizzled scale-factor buffer of a ``(rows, k)`` operand of the format: whole atoms."""
    return -(-rows // SF_ATOM_ROWS) * SF_ATOM_ROWS * -(-k // (SF_VEC[in_fmt] * SF_ATOM_KBLOCKS)) * SF_ATOM_KBLOCKS


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


MMA_K_BLOCKS = 4
BREUSE_FMTS = ("nvfp4_k128", "mxfp8_k64", "mxfp4_k128", "mxfp4w8a_k64")


@dataclass(frozen=True)
class Knobs:
    mma_tiler_mn: Tuple[int, int]
    cluster_shape_mn: Tuple[int, int] = (1, 1)
    use_2cta_instrs: bool = False
    use_tma_store = True
    cta_chunks = False
    b_reuse = False
    cluster_k = 1
    k_block = 0
    swap_ab = False

    @property
    def tile_m(self):
        return self.mma_tiler_mn[0]

    @property
    def tile_n(self):
        return self.mma_tiler_mn[1]

    @property
    def inst_m(self):
        return self.tile_m

    @property
    def cta_tile_mn(self):
        return self.tile_m // (2 if self.use_2cta_instrs else 1), self.tile_n


@dataclass(frozen=True)
class MainloopBlock:
    in_fmt: str
    k_block_override: int = 0
    inner_steps_override: Optional[int] = None

    @property
    def step(self):
        return MMA_K_OF[self.in_fmt]

    @property
    def k_block(self):
        return self.inner_steps_override * self.step if self.inner_steps_override else self.k_block_override or 1024 // ELEM_BITS_OF[self.in_fmt]

    @property
    def inner(self):
        return self.k_block // self.step

    def lowering(self):
        return self

    def inner_steps(self):
        return range(self.inner)

    @staticmethod
    def accumulate(block, j):
        return True if j > 0 else block != 0


class CuteF32:
    const = Float32

    @staticmethod
    def add(a, b):
        return cute.math.add(a, b, rounding=RoundingMode.NEAREST_EVEN)

    @staticmethod
    def mul(a, b):
        return cute.math.mul(a, b, rounding=RoundingMode.NEAREST_EVEN)

    @staticmethod
    def fma2(a, b, c):
        return cute.arch.fma_packed_f32x2(a, b, c, rnd="rn")

    @staticmethod
    def widen(a):
        return a.to(Float32)


@dataclass(frozen=True)
class EpilogueChain:
    """The elementwise ops after the fold, in order: ``mul.rn alpha`` for the scaled (fp8) classes, ``add.rn bias`` for
    the biased chain (the bias of the output column, widened to binary32 exactly: ``BIAS_NOTE``), then exactly one
    ``cvt.rn <out>``. One object, one ``apply``: the spec term (:func:`output_elem`, ``fp`` = the term algebra), the
    kernel (``blocks/epilogue.py``: ``fp`` = ``CuteF32`` for the ops before the conversion, the vector ``.to(out dtype)``
    for the conversion itself) and the fixup all run this code."""

    ops: Tuple[str, ...]

    @staticmethod
    def for_class(out_fmt: str, scaled: bool, bias: bool = False) -> "EpilogueChain":
        return EpilogueChain((("mul.rn alpha",) if scaled else ()) + (("add.rn bias",) if bias else ()) + (f"cvt.rn {out_fmt}",))

    @property
    def out_fmt(self) -> str:
        assert self.ops and self.ops[-1].startswith("cvt.rn "), self.ops
        return self.ops[-1].split()[1]

    @property
    def pre(self) -> Tuple[str, ...]:
        """The ops before the conversion (all in fp32)."""
        return self.ops[:-1]

    @property
    def scaled(self) -> bool:
        return "mul.rn alpha" in self.pre

    @property
    def biased(self) -> bool:
        return "add.rn bias" in self.pre

    def apply_pre(self, fp, x, alpha=None, bias=None):
        """``alpha`` is the scaled chain's runtime scalar, ``bias`` the biased chain's bias of this element's column (a
        binary32 value: the bf16 bias widened exactly)."""
        for op in self.pre:
            if op == "mul.rn alpha":
                assert alpha is not None, "the scaled chain needs alpha"
                x = fp.mul(x, alpha)
            elif op == "add.rn bias":
                assert bias is not None, "the biased chain needs the column's bias"
                x = fp.add(x, bias)
            else:  # a new chain op must be added here, nowhere else
                raise ValueError(f"unknown epilogue op {op!r}")
        return x

    def apply(self, fp, x, alpha=None, bias=None):
        return fp.cvt_out(self.apply_pre(fp, x, alpha, bias), self.out_fmt)

    def describe(self) -> str:
        return " -> ".join(self.ops)
