# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Identity of the small-head FP8 dual_kv implementation."""

NAME = "sdpa_fwd_sm100_d128_fp8_dual_kv"
TEMPLATE_NAME = "prefill_d128_fp8"
KERNEL_FILE = "sm100/d128_fp8_dual_kv/kernel.py"
LEGACY_ROW = "sdpa_fwd_prefill_sm100_fp8"
