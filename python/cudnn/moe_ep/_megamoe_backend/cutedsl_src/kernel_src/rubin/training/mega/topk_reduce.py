# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Rubin-training source-copy shim for the compatible Blackwell TopK reduction.

Identical implementation to the Blackwell / inference ``TopkReduce``; kept as a
marked import so ``rubin.training.mega`` stays a self-contained deliverable and
the kernel_export script can inline the source instead of pulling in another
kernel product's directory.
"""

# <<<MEGA_REPO_CONTROL : COPY_FROM_IMPORT>>>
from ....blackwell.inference.mega.topk_reduce import TopkReduce


__all__ = ["TopkReduce"]
