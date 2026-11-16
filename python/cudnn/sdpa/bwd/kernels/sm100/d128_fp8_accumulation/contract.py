# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100 D64/D128 FP8 accumulation backward identity (dense and grouped launches); no runtime imports."""

from ..quantized_contract import D128_FP8_ACCUMULATION as NAME

KERNEL_FILE = "sm100/d128_fp8_accumulation/kernel.py"
HELPERS_FILE = "sm100/d128_fp8_accumulation/helpers.py"
__all__ = ["HELPERS_FILE", "KERNEL_FILE", "NAME"]
