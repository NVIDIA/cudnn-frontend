# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-only schedule selection; offsets remain device data."""


def backward_schedule(tokens, channels, width, dtype, silu, native, capability, multiprocessors):
    partial_bytes = 8 if dtype == "torch.float32" else 4
    splits = max(1, min(64, (16 * 1024 * 1024) // (channels * (width + 1) * partial_bytes)))
    if capability != (10, 3) or width != 4 or not silu or not native or dtype != "torch.bfloat16":
        return "tiled", splits
    # These channel widths cover the measured shared training workloads.
    # Retain the tiled path for short streams and partial channel tiles.
    if tokens >= 8192 and channels in (768, 2048) and tokens <= 2**31 - 257:
        return "persistent", max(1, (2 * multiprocessors) // ((channels + 255) // 256))
    if channels == 256 and 4096 <= tokens < 8192:
        splits = min(splits, max(1, multiprocessors // 4))
    return "tiled", splits
