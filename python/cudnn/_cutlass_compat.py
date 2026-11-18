# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CuTeDSL names from their current modules, with fallbacks for older wheels.

Imported by kernel modules only to keep the package entry point lazy.
"""

try:
    from cutlass import AddressSpace
except ImportError:
    from cutlass.cute import AddressSpace

try:
    from cutlass.tensor_utils import LayoutEnum
except ModuleNotFoundError as error:
    if error.name != "cutlass.tensor_utils":
        raise
    from cutlass.utils import LayoutEnum

try:
    from cutlass.memory import SmemAllocator, TmemAllocator, get_num_tmem_alloc_cols, get_smem_capacity_in_bytes
except ModuleNotFoundError as error:
    if error.name != "cutlass.memory":
        raise
    from cutlass.utils import SmemAllocator, TmemAllocator, get_num_tmem_alloc_cols, get_smem_capacity_in_bytes

try:
    from cutlass.base_dsl.enums import Arch
except ModuleNotFoundError as error:
    if error.name != "cutlass.base_dsl.enums":
        raise
    from cutlass.base_dsl.arch import Arch

from cutlass.cute import FastDivmodDivisorV2 as FastDivmodDivisor
from cutlass.cute import fast_divmod_create_divisor_v2 as fast_divmod_create_divisor
from cutlass.cute.arch import setmaxregister_decrease, setmaxregister_increase
from cutlass.cute.nvgpu import OperandMajorMode
