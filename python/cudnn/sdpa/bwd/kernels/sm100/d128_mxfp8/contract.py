# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100 / SM103 D128 MXFP8 backward identity; no runtime imports."""

NAME = "sdpa_bwd_sm100_d128_mxfp8"
KERNEL_FILE = "sm100/d128_mxfp8/chain.py"  # the compiled entry: prep.py -> dkdv.py -> dq.py, helpers in common.py
