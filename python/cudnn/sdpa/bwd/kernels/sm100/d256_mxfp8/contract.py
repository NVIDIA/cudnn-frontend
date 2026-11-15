# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100 D256 MXFP8 dK/dV + dQ chain identity; no runtime imports."""

from ..quantized_contract import D256_MXFP8 as NAME

DKDV_FILE = "sm100/d256_mxfp8/dkdv.py"
DQ_FILE = "sm100/d256_mxfp8/dq.py"
__all__ = ["DKDV_FILE", "DQ_FILE", "NAME"]
