# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The native SM107 D256 materialized-dS body."""

NAME = "sdpa_bwd_sm107_d256_f16_4x1"
LEGACY_ROW = "sdpa_bwd_sm107"
KERNEL_FILE = "sm107/d256_f16_4x1/kernel.py"
TEMPLATE_TAG = "sdpa_bwd_sm107_main_f16"
