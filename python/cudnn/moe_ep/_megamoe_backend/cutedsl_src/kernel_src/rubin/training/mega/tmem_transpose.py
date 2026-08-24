# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Rubin-training source-copy shim for the 16x32 TMEM transpose core.

``_TmemTranspose16x32Core`` is the register-level transpose helper shared with
the Blackwell swap-AB epilogue.  It is arch-compatible (identical math), so we
re-export it through a marked import rather than re-porting the transpose, and
rather than reaching into another kernel product's directory at port time --
the kernel_export script inlines the source here.
"""

# <<<MEGA_REPO_CONTROL : COPY_FROM_IMPORT>>>
from ....blackwell.inference.mega.block_scaled_swap_ab_fc12_epilogue import (
    _TmemTranspose16x32Core,
)


__all__ = ["_TmemTranspose16x32Core"]
