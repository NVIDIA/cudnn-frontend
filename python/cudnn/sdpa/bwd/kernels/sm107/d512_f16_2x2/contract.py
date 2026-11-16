# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fixed backward identity; the existing public engine record is preserved."""

NAME = "sdpa_bwd_sm107_d512"
KERNEL_FILE = "sm107/d512_f16_2x2/kernel.py"
STAGE2_TAG = "sdpa_bwd_sm107_stage2_2x2"

__all__ = ["KERNEL_FILE", "NAME", "STAGE2_TAG"]
