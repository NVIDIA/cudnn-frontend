# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SM107 MXFP8 D192/V128 multi-Q block-scaled body: identity only."""

NAME = "sdpa_fwd_sm107_d192_mxfp8_multi_q"
TEMPLATE_NAME = "prefill_d192_d128_mxfp8"
KERNEL_FILE = "sm107/d192_mxfp8_multi_q/kernel.py"
SHAPE = (192, 128)
