# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SM100 FP8 key-owner dK/dV + query-owner dQ pair (also run in block-scale mode); no runtime imports.

Engines that run it: sdpa_bwd_sm100_d192_fp8 at (192, 128), sdpa_bwd_sm100_d128_fp8_pair at (128, 128) and
sdpa_bwd_sm100_mxfp8_pair for the MXFP8 pair heads.
"""

DKDV_FILE = "sm100/fp8_pair/dkdv.py"
DQ_FILE = "sm100/fp8_pair/dq.py"
