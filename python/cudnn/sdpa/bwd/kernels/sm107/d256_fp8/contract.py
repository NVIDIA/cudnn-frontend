# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SM107 FP8 materialized-dS body."""

NAME = "sdpa_bwd_sm107_fp8"
KERNEL_FILE = "sm107/d256_fp8/kernel.py"
TEMPLATE_TAG = "sdpa_bwd_sm107_main_fp8"
