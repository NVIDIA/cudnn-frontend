# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only alias validation for caller-owned dGLU scratch."""

import math

from cudnn.frost import buffers
from cudnn.tensor_adapter import get_data_ptr, get_strides, is_jax_array, is_torch_tensor


def validate_workspace_aliases(workspace_ptr, workspace_bytes, **operands):
    """Check the carved byte range, allowing disjoint views of one allocation.

    Pointer-table contents are not read: callers own the non-aliasing and lifetime
    contract for pointed-to allocations, just as they own pointer validity.
    """
    workspace_end = workspace_ptr + workspace_bytes
    for name, tensor in operands.items():
        if tensor is None:
            continue
        if is_torch_tensor(tensor):
            if not tensor.numel():
                continue
            begin, itemsize = tensor.data_ptr(), tensor.element_size()
            shape, strides = tensor.shape, None if tensor.is_contiguous() else tensor.stride()
        elif is_jax_array(tensor):
            # JAX does not accept DLPack's stream=-1 sentinel. Read the same
            # host metadata used by the tensor adapter; never export/sync here.
            if not tensor.size:
                continue
            begin, itemsize = get_data_ptr(tensor), tensor.dtype.itemsize
            shape, strides = tensor.shape, get_strides(tensor)
        else:
            begin, shape, strides, dtype, _ = buffers.probe(tensor)
            if 0 in shape:
                continue
            itemsize = buffers.DTYPE_ITEMSIZE[dtype]
        if strides is None:
            end = begin + math.prod(shape) * itemsize
        else:
            offsets = [(extent - 1) * stride * itemsize for extent, stride in zip(shape, strides)]
            end = begin + sum(max(offset, 0) for offset in offsets) + itemsize
            begin += sum(min(offset, 0) for offset in offsets)
        if workspace_ptr < end and begin < workspace_end:
            raise ValueError(f"workspace must not overlap {name}")
