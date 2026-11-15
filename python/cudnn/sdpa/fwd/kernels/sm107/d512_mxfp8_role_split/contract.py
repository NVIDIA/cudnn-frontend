# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SM107 MXFP8 D512 role-split block-scaled body: identity only."""

NAME = "sdpa_fwd_sm107_d512_mxfp8_role_split"
TEMPLATE_NAME = "prefill_d512_mxfp8"
KERNEL_FILE = "sm107/d512_mxfp8_role_split/kernel.py"
SHAPE = (512, 512)
