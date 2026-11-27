# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The arithmetic of the batch-invariant RMSNorm: the class of a row, a function of its length and dtype.

A row ``x[0:H]`` with weight ``w[0:H]`` and epsilon ``eps`` produces

    y[i]   = out(mul.rn(mul.rn(x[i], rstd), w[i]))
    rstd   = div.rn(1, sqrt.rn(add.rn(div.rn(ssq, H), eps)))

where every operand is first widened exactly to fp32, ``out`` is the
round-to-nearest-even conversion to the output dtype, and ``ssq`` is the sum
of squares in this fixed order:

1. The row is cut into granules of ``V`` consecutive elements: the largest
   power of two that divides ``H``, up to one 16-byte vector of the input
   (``V = 8`` for 16-bit inputs, ``4`` for fp32, when ``H`` is a multiple of
   it). Granule ``g`` belongs to lane ``g % L``.
2. Lane ``l`` folds its granules in ascending order, and the elements of a
   granule in ascending order, as ``acc = fma.rn(x, x, acc)`` from ``acc = +0``.
3. The lanes form groups of ``min(L, 32)``. Each group sums its lanes by a
   butterfly with offsets 1, 2, 4, ... (``s[l] = add.rn(s[l], s[l ^ offset])``).
4. The group sums are added in ascending group order with ``add.rn``, starting
   from the first group's sum.

``V`` and ``L`` depend on the hidden size and the input dtype only, so the bits
of a row never depend on the batch size, the row's position, the SM count or
the placement that computed it. The kernel's placements (``_knobs``) map
several lanes to one thread or several rows to one CTA without changing the
order above.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Bytes of the widest granule: one 16-byte vector of the input.
VECTOR_BYTES = 16
#: Lanes of the class at most: one CTA of 1024 threads holds a row at one lane per thread.
MAX_LANES = 1024
#: Lanes of the class at least, unless the row has fewer granules.
MIN_LANES = 32
#: Row elements a lane folds at the widest placement before the class adds lanes, in 16-byte vectors of the input.
VECTORS_PER_LANE = 4
#: Elements of one row that a thread may hold in registers across the reduction.
MAX_ELEMENTS_PER_THREAD = 64

ELEMENT_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}


def _pow2_floor(n: int) -> int:
    return 1 << (n.bit_length() - 1)


@dataclass(frozen=True)
class RmsNormClass:
    """The arithmetic of one batch-invariant RMSNorm class.

    ``hidden`` is the row length H, ``in_dtype`` the input dtype tag, ``vector``
    the granule size V and ``lanes`` the lane count L.
    """

    hidden: int
    in_dtype: str
    vector: int
    lanes: int

    @property
    def granules(self) -> int:
        return self.hidden // self.vector

    @property
    def granules_per_lane(self) -> int:
        """Granules of the lane that folds the most (lanes ``< granules % lanes`` fold one more)."""
        return -(-self.granules // self.lanes)

    @property
    def group(self) -> int:
        """Lanes of one butterfly group."""
        return min(self.lanes, 32)

    @property
    def groups(self) -> int:
        return self.lanes // self.group

    @property
    def name(self) -> str:
        return f"rmsnorm.bi.v1[{self.in_dtype},H={self.hidden},V={self.vector},L={self.lanes}]"


def class_for(hidden: int, in_dtype: str) -> RmsNormClass:
    """The class of a row of ``hidden`` elements of ``in_dtype``.

    Raises ``NotImplementedError`` for shapes the class does not define.
    """
    if in_dtype not in ELEMENT_BYTES:
        raise NotImplementedError(f"batch_invariant_rmsnorm does not support input dtype {in_dtype}")
    if hidden <= 0:
        raise NotImplementedError(f"batch_invariant_rmsnorm requires a positive hidden size, got {hidden}")
    vector = min(hidden & -hidden, VECTOR_BYTES // ELEMENT_BYTES[in_dtype])
    granules = hidden // vector
    per_lane = VECTORS_PER_LANE * VECTOR_BYTES // ELEMENT_BYTES[in_dtype]
    lanes = min(_pow2_floor(granules), max(MIN_LANES, min(MAX_LANES, _pow2_floor(max(1, hidden // per_lane)))))
    cls = RmsNormClass(hidden, in_dtype, vector, lanes)
    if cls.granules_per_lane * vector > MAX_ELEMENTS_PER_THREAD:
        raise NotImplementedError(f"batch_invariant_rmsnorm supports at most {MAX_LANES * MAX_ELEMENTS_PER_THREAD} elements per row, got {hidden}")
    return cls
