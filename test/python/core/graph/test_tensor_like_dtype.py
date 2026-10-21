# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""tensor_like() declares the dtype the buffer has, whatever framework made it.

execute() describes each operand with ``_buffer_dtype_to_cudnn(data.dtype)``,
one table for torch, numpy/cupy dtypes and dtype names. The declaration has to
come from the same table: when tensor_like() only knew torch dtypes, a float32
or int32 cupy / jax / numpy array fell back to the graph's io_data_type and was
compiled as HALF while execute() still described it as FLOAT / INT32.

Nothing here needs a device, and only the last test needs torch. A numpy array,
or an object that carries a numpy dtype the way cupy and jax arrays do, is enough.
"""

import pytest

np = pytest.importorskip("numpy")

import cudnn
from cudnn._pygraph import pygraph
from cudnn.datatypes import _buffer_dtype_to_cudnn

pytestmark = pytest.mark.L0


class _CupyLike:
    """A cupy.ndarray look-alike: numpy dtype, byte strides, no torch-style stride()."""

    def __init__(self, a):
        self.shape = a.shape
        self.dtype = a.dtype
        self.strides = a.strides


class _NamedDtype:
    """A dtype only its name identifies, like an ml_dtypes bfloat16 when the table was built before ml_dtypes was imported."""

    def __init__(self, name):
        self.name = name


class _Buffer:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


def _half_graph():
    return pygraph(io_data_type=cudnn.data_type.HALF, compute_data_type=cudnn.data_type.FLOAT)


@pytest.mark.parametrize(
    "make, expected",
    [
        (lambda: np.zeros((4, 8), np.float32), "FLOAT"),
        (lambda: _CupyLike(np.zeros((4, 8), np.float32)), "FLOAT"),
        (lambda: _CupyLike(np.zeros((4, 8), np.int32)), "INT32"),
        (lambda: _Buffer((4, 8), _NamedDtype("bfloat16")), "BFLOAT16"),
    ],
    ids=["numpy_f32", "cupy_like_f32", "cupy_like_i32", "named_bf16"],
)
def test_tensor_like_declares_the_buffer_dtype_not_io_data_type(make, expected):
    g = _half_graph()
    t = g.tensor_like(make())
    assert t.data_type == getattr(cudnn.data_type, expected)


def test_op_builder_framework_operand_matches_what_execute_describes():
    """Op builders route framework arrays through tensor_like(); the declaration
    and execute()'s description of the same buffer must agree."""
    a = _CupyLike(np.zeros((1, 4, 8), np.float32))
    b = _CupyLike(np.zeros((1, 8, 2), np.float32))
    g = _half_graph()
    g.matmul(A=a, B=b)
    (node,) = g.nodes
    assert node.inputs["A"].data_type == _buffer_dtype_to_cudnn(a.dtype) == cudnn.data_type.FLOAT
    assert node.inputs["B"].data_type == _buffer_dtype_to_cudnn(b.dtype) == cudnn.data_type.FLOAT


def test_unknown_dtype_still_falls_back_to_io_data_type():
    g = _half_graph()
    t = g.tensor_like(_Buffer((4, 8), _NamedDtype("not_a_dtype")))
    assert t.data_type == cudnn.data_type.HALF


def test_torch_tensor_dtype_is_unchanged():
    torch = pytest.importorskip("torch")
    g = _half_graph()
    assert g.tensor_like(torch.zeros(4, 8, dtype=torch.float32)).data_type == cudnn.data_type.FLOAT
    assert g.tensor_like(torch.zeros(4, 8, dtype=torch.bfloat16)).data_type == cudnn.data_type.BFLOAT16
