# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recurrent-state layout swap between FLA's K-major ``[N, H, K, V]`` and cuDNN's
V-major ``[N, H, V, K]``; compact in both directions so the kernels see stride-1
innermost buffers for the state and for its gradient."""

from __future__ import annotations

import torch


class TransposeState(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state):
        return state.transpose(-1, -2).contiguous()

    @staticmethod
    def backward(ctx, grad):
        return grad.transpose(-1, -2).contiguous()
