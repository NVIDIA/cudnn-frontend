# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent element origins for packed or dense ports of a THD graph."""

import hashlib
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.typing import Pointer

FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


@cute.jit
def sequence_base(offsets, batch, packed_base, batch_stride: cutlass.Constexpr = None, multiplier: cutlass.Constexpr = 1):
    if cutlass.const_expr(offsets is not None):
        pointer = Pointer(cutlass.make_array_view(offsets).data_ptr(), dtype=offsets.element_type)
        return cutlass.Int64(pointer[cutlass.Int64(batch) * cutlass.Int64(offsets.stride[0])]) * cutlass.Int64(multiplier)
    elif cutlass.const_expr(batch_stride is not None):
        return cutlass.Int64(batch) * cutlass.Int64(batch_stride)
    else:
        return packed_base
