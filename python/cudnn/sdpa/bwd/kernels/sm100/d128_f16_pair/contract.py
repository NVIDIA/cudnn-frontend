# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identity of the fixed deterministic small-half backward chain."""

NAME = "sdpa_bwd_sm100_d128_pair"
LEGACY_ROW = "sdpa_bwd_sm100_d128"
KERNEL_FILES = ("kernel_dkv.py", "kernel_dq.py")
