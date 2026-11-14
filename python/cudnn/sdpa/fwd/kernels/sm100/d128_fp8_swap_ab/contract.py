# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Identity of the small-head FP8 swap_ab implementation."""

NAME = "sdpa_fwd_sm100_d128_fp8_swap_ab"
TEMPLATE_NAME = "decode_d128_fp8"
KERNEL_FILE = "sm100/d128_fp8_swap_ab/kernel.py"
LEGACY_ROW = "sdpa_fwd_prefill_sm100_fp8"
