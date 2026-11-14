# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM120 per-tensor FP8 general identity; no runtime imports."""

NAME = "sdpa_fwd_sm120_fp8_general"
TEMPLATE_NAME = "prefill_fp8"
KERNEL_FILE = "sm120/fp8_general/kernel.py"
LEGACY_ROW = "sdpa_fwd_prefill_sm120_fp8"
QUERY_TILES = (64, 128)
DEFAULT_QUERY_TILE = 128
