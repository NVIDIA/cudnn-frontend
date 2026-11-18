# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SM107 MXFP8 D128 multi-Q block-scaled body (two Q tiles per CTA pair): identity only."""

NAME = "sdpa_fwd_sm107_d128_mxfp8_multi_q"
TEMPLATE_NAME = "prefill_d128_mxfp8"
KERNEL_FILE = "sm107/d128_mxfp8_multi_q/kernel.py"
SHAPE = (128, 128)
