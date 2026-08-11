# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A DeviceView exports DLPack through a slot, which owns the struct it hands out.

The view itself has nowhere to put a ``DLManagedTensor`` that must outlive the
capsule: cute's ``from_dlpack`` aliases the struct rather than copying it, and
the view cannot know when the consumer is done. Holding the struct on the view
behind a no-op deleter — which is what this did — reads freed memory as soon as
a consumer outlives the view.
"""

import ctypes
import gc

import pytest
import torch

from cudnn.frost import buffers

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="DLPack export is over device pointers")


def _capsule_address(capsule):
    fn = ctypes.pythonapi.PyCapsule_GetPointer
    fn.restype = ctypes.c_void_p
    fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return fn(capsule, b"dltensor")


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
    assert _capsule_address(a) != _capsule_address(b)


@pytest.mark.L0
def test_capsule_outlives_the_view_it_came_from():
    """The regression: the struct belongs to the capsule, not to the view.

    An unconsumed capsule used to point at a struct the view held in a list,
    so dropping the view left the consumer decoding freed memory.
    """
    t = torch.arange(16, dtype=torch.float32, device="cuda")
    view = buffers.DeviceView(t.data_ptr(), (16,), "float32", t.device.index or 0)
    capsule = view.__dlpack__()
    del view
    gc.collect()
    torch.testing.assert_close(torch.from_dlpack(capsule), t)


@pytest.mark.L0
@pytest.mark.parametrize("dtype,torch_dtype", [("float32", torch.float32), ("bfloat16", torch.bfloat16), ("int32", torch.int32), ("uint8", torch.uint8)])
def test_dtypes_round_trip(dtype, torch_dtype):
    t = torch.ones(6, dtype=torch_dtype, device="cuda")
    view = buffers.DeviceView(t.data_ptr(), (6,), dtype, t.device.index or 0)
    back = torch.from_dlpack(view)
    assert back.dtype is torch_dtype and back.shape == (6,)
    torch.testing.assert_close(back, t)
