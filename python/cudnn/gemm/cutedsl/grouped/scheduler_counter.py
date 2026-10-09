# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime scheduler counters initialized by a caller on the execution stream."""


def validate_scheduler_counter(counter, a, enabled=True):
    if counter is None:
        return
    import torch

    if not enabled:
        raise ValueError("scheduler_counter_tensor requires dense block-scaled inputs and dynamic scheduling")
    if counter.dtype != torch.int32 or not counter.is_cuda or counter.device != a.device:
        raise ValueError("scheduler_counter_tensor must be a CUDA int32 tensor on the input device")
    if counter.ndim != 1 or counter.numel() < 1 or not counter.is_contiguous():
        raise ValueError("scheduler_counter_tensor must be a nonempty contiguous 1-D tensor")
