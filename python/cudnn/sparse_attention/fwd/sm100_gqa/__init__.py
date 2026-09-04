# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100 GQA-substrate sparse-attention forward kernel (PR4).

Re-exports ``sparse_attention_forward_wrapper`` -- the entry point
``cudnn.sparse_attention.fwd.api._get_gqa_substrate_kernel`` probes for.
"""

from .dispatch import sparse_attention_forward_wrapper

__all__ = ["sparse_attention_forward_wrapper"]
