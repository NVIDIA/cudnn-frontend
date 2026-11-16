# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed main-implementation identity; selection policy lives outside this directory."""

NAME = "sdpa_bwd_sm100_d128_accumulation"
LEGACY_ROW = "sdpa_bwd_sm100_d128"
KERNEL_FILE = "sm100/d128_f16_accum/kernel.py"
