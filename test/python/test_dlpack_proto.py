# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A DeviceView's DLPack struct is copied from a per-layout prototype.

Everything in a ``DLManagedTensor`` except ``data`` is a property of the layout,
so the struct is filled once per (shape, dtype, device) and memmove'd per call:
measured 1.68 us to assign nine ctypes fields against 0.45 us to copy the 72
bytes. Same shape as the backend's kernel-argument handling
(``src/common/include/runtimeKernel.h``), where a prefilled blob carries an
(offset, uid, UpdateMethod) per mutable field; here there is exactly one
mutable field.

The struct must be FRESH per capsule even so: cute's ``from_dlpack`` aliases it
rather than copying, so a struct shared between two capsules is read after
someone else re-pointed it.
"""

import ctypes

import pytest
import torch

from cudnn.frost import buffers


@pytest.mark.L0
def test_capsule_decodes_to_the_right_buffer():
    t = torch.arange(24, dtype=torch.float32, device="cuda").reshape(2, 3, 4)
    view = buffers.DeviceView(t.data_ptr(), (2, 3, 4), "float32", t.device.index or 0)
    back = torch.from_dlpack(view)
    torch.testing.assert_close(back, t)


@pytest.mark.L0
def test_two_capsules_from_one_view_do_not_share_a_struct():
    """cute aliases the struct, so a shared one would be rewritten under it."""
    t = torch.zeros(8, dtype=torch.float32, device="cuda")
    view = buffers.DeviceView(t.data_ptr(), (8,), "float32", t.device.index or 0)
    a, b = view.__dlpack__(), view.__dlpack__()
    addr = ctypes.pythonapi.PyCapsule_GetPointer
    addr.restype = ctypes.c_void_p
    addr.argtypes = [ctypes.py_object, ctypes.c_char_p]
    assert addr(a, b"dltensor") != addr(b, b"dltensor")


@pytest.mark.L0
def test_prototypes_are_shared_across_views_of_one_layout():
    """The prototype is the cache; the struct is not."""
    t = torch.zeros(4, 5, dtype=torch.bfloat16, device="cuda")
    dev = t.device.index or 0
    v1 = buffers.DeviceView(t.data_ptr(), (4, 5), "bfloat16", dev)
    v2 = buffers.DeviceView(t.data_ptr() + 64, (4, 5), "bfloat16", dev)
    v1.__dlpack__()
    v2.__dlpack__()
    assert v1._proto is v2._proto


@pytest.mark.L0
@pytest.mark.parametrize("dtype,torch_dtype", [("float32", torch.float32), ("bfloat16", torch.bfloat16), ("int32", torch.int32), ("uint8", torch.uint8)])
def test_dtypes_round_trip(dtype, torch_dtype):
    t = torch.ones(6, dtype=torch_dtype, device="cuda")
    view = buffers.DeviceView(t.data_ptr(), (6,), dtype, t.device.index or 0)
    back = torch.from_dlpack(view)
    assert back.dtype is torch_dtype and back.shape == (6,)
    torch.testing.assert_close(back, t)
