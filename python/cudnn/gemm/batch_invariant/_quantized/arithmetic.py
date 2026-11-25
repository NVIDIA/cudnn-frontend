# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from FlashInfer's flashinfer/gemm/batch_invariant/{spec,generator}.py and
# flashinfer/batch_invariant/fp_interface/cute.py (e2d163fb3d46) for cuDNN GEMM plans.

"""Quantized instruction widths, fixed K folds, and FP32 epilogue operations."""

from dataclasses import dataclass
from typing import Optional, Tuple

import cutlass.cute as cute
from cutlass import Float32
from cutlass.cute.math import RoundingMode

from .._formats import ELEM_BITS_OF, MMA_K_OF

MMA_K_BLOCKS = 4


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
