# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared PyTorch-to-cuDNN dtype mappings for GNN operations."""

from typing import Dict

import torch

TORCH_DTYPE_TO_CUDNN: Dict[torch.dtype, int] = {
    torch.float32: 0,  # CUDNN_DATA_FLOAT
    torch.float16: 2,  # CUDNN_DATA_HALF
    torch.bfloat16: 9,  # CUDNN_DATA_BFLOAT16
}

TORCH_INDEX_DTYPE_TO_CUDNN: Dict[torch.dtype, int] = {
    torch.int32: 4,  # CUDNN_DATA_INT32
    torch.int64: 10,  # CUDNN_DATA_INT64
}
