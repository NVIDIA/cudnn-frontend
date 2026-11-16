# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM120 MXFP8 general identity; no runtime imports."""

NAME = "sdpa_fwd_sm120_mxfp8_general"
TEMPLATE_NAME = "prefill_mxfp8"
KERNEL_FILE = "sm120/mxfp8_general/kernel.py"
LEGACY_ROW = "sdpa_fwd_prefill_sm120_mxfp8"
QUERY_TILES = (64, 128)
DEFAULT_QUERY_TILE = 128
