# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata validation shared by the chunk-64 Cake plans."""

import math
from cudnn.frost import buffers


def ffi_view(value, device):
    """Avoid the native exchange API's default-stream override in TVM-FFI."""
    return buffers.DeviceView(value.data_ptr(), value.shape, buffers.dtype_name(value), device)


class CakeLinearPlan:
    def workspace_bytes(self):
        return self.workspace_size

    def bind(self, names):
        self.names = tuple(names)

    def _validate(self, operands, workspace):
        spans = []
        for name, value in operands.items():
            shape, dtype = self.geometry[name]
            if tuple(value.shape) != shape or buffers.dtype_name(value) != dtype:
                raise ValueError(f"{self.plan_name}: {name} must have shape {shape} and dtype {dtype}")
            if not buffers.is_contiguous(value.shape, value.stride()):
                raise ValueError(f"{self.plan_name}: {name} must be contiguous")
            ptr = value.data_ptr()
            if ptr == 0 or ptr % 16:
                raise ValueError(f"{self.plan_name}: {name} must be nonnull and 16-byte aligned")
            if tuple(value.__dlpack_device__()) != (2, self.device):
                raise ValueError(f"{self.plan_name}: {name} must be on cuda:{self.device}")
            spans.append((name, ptr, ptr + math.prod(shape) * buffers.DTYPE_ITEMSIZE[dtype]))
        ws = workspace.view(0, "uint8", (self.workspace_size,))
        spans.append(("workspace", ws.data_ptr(), ws.data_ptr() + self.workspace_size))
        for i, (name, start, end) in enumerate(spans):
            for other, lo, hi in spans[:i]:
                if start < hi and lo < end and (name in self.outputs or other in self.outputs or "workspace" in (name, other)):
                    raise ValueError(f"{self.plan_name}: {name} overlaps {other}")
