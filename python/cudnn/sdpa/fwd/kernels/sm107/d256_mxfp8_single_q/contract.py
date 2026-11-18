# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SM107 MXFP8 D256 single-Q block-scaled body: identity only."""

NAME = "sdpa_fwd_sm107_d256_mxfp8_single_q"
TEMPLATE_NAME = "prefill_d256_mxfp8"
KERNEL_FILE = "sm107/d256_mxfp8_single_q/kernel.py"
SHAPE = (256, 256)
