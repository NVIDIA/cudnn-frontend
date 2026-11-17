# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device helpers shared by the SM80 backward bodies."""

import cutlass
import cutlass.cute as cute
from cutlass.cute.arch.nvvm_wrappers import inline_ptx


@cute.jit
def _atomic_add_f32(addr_i64, val):
    """``atom.global.add.f32`` on a 32-bit GMEM float element.  ``addr_i64`` is
    an element pointer offset (Int64 byte address via the view's data_ptr +
    element offset); the returned old value is discarded."""
    inline_ptx(
        "atom.global.add.f32 $0, [$1], $2;",
        write_only_types=[cutlass.Float32],
        read_only_args=[addr_i64, val],
    )
