# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""The native record projection must preserve every observed and effective buffer fact."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import cudnn
from cudnn.sdpa.fwd import prepared as prep

pytestmark = [pytest.mark.L0]


def _project(native, indices):
    return prep.facts_of_roles(SimpleNamespace(native=native), indices)


def _accessor_projection(native, indices):
    """Independent oracle using individual native metadata accessors."""
    result = []
    for index in indices:
        code, bits = native.dtype(index)
        nbytes = native.observed_bytes(index)
        result.append(
            prep.BufferFacts(
                native.pointer(index),
                prep._DTYPE_BY_CODE.get((code, bits), ""),
                tuple(native.observed_device(index)),
                -1 if nbytes < 0 else nbytes // max(1, (int(bits) + 7) // 8),
                tuple(native.shape(index)),
                tuple(native.stride(index)),
            )
        )
    return result


@pytest.mark.parametrize("code,bits,lanes", [(4, 16, 1), (2, 32, 1), (0, 32, 1), (1, 8, 1), (17, 4, 2), (255, 0, 1)])
@pytest.mark.parametrize(
    "shape,strides,nbytes,expected_strides",
    [
        ((2, 3), (8, 1), 22, (8, 1)),
        ((2, 3), (), 12, (3, 1)),
        ((0, 3), (), 0, (3, 1)),
        ((2, 0, 3), (), 0, (0, 3, 1)),
        ((2, 1, 3), (), 12, (3, 3, 1)),
        ((1, 3), (2**40, 1), 6, (2**40, 1)),
        ((), (), -1, ()),
    ],
)
def test_native_facts_projection_preserves_dtype_layout_device_span_and_order(code, bits, lanes, shape, strides, nbytes, expected_strides):
    native = cudnn._pybind_module.VariantPackNative(3)
    for i, device in enumerate(((2, 1), (1, 0), (-1, -1))):
        native.set_operand(i, 4096 * (i + 1), shape, strides, code, bits, lanes, nbytes, *device)
    indices = [2, 0, 2, 1]
    projected = _project(native, indices)
    assert projected == _accessor_projection(native, indices)
    assert tuple(native.stride(0)) == expected_strides
    assert all(f.strides == expected_strides for f in projected)
    assert [f.ptr for f in projected] == [12288, 4096, 12288, 8192]
    assert [f.device for f in projected] == [(-1, -1), (2, 1), (-1, -1), (1, 0)]
    assert all(isinstance(f, prep.BufferFacts) and isinstance(f.shape, tuple) and isinstance(f.strides, tuple) for f in projected)
    again = _project(native, indices)
    assert again == projected and all(a is not b for a, b in zip(again, projected)), "records belong to this call"
    with pytest.raises(AttributeError):
        projected[0].ptr = 0


def test_native_facts_keep_observed_capacity_separate_from_declared_and_overridden_geometry():
    native = cudnn._pybind_module.VariantPackNative(1)
    layout = cudnn._pybind_module.DeclaredLayout(1)
    layout.set(0, [1, 8, 4], [32, 4, 1], 2, 4, 16, 1, False)
    native.set_operand(0, 4096, [64], [1], 1, 8, 1, 64, 2, 1)
    assert native.describe_from(layout, []) == [0]
    described = _project(native, [0])[0]
    assert described == _accessor_projection(native, [0])[0]
    assert described.dtype == "bfloat16" and described.span == 32 and described.shape == (1, 8, 4)
    native.override_many(layout, [0], [[1, 4, 4]], [[16, 4, 1]])
    overridden = _project(native, [0])[0]
    assert overridden == _accessor_projection(native, [0])[0]
    assert overridden.shape == (1, 4, 4) and overridden.span == 32 and overridden.device == (2, 1)
    assert described.shape == (1, 8, 4), "an existing record cannot change when its pack changes"


@pytest.mark.parametrize("device,nbytes,match", [((1, 0), 4096, "CUDA device"), ((2, 1), 4096, "CUDA device"), ((2, 0), 4094, "spans")])
def test_native_facts_still_reject_wrong_devices_and_short_observed_storage(device, nbytes, match):
    native = cudnn._pybind_module.VariantPackNative(1)
    native.set_operand(0, 4096, [2, 8, 1, 128], [1024, 128, 1024, 1], 4, 16, 1, nbytes, *device)
    fact = _project(native, [0])[0]
    assert fact == _accessor_projection(native, [0])[0]
    with pytest.raises(ValueError, match=match):
        prep._dense_role(SimpleNamespace(b=2, device_index=0), {"q": fact}, "q", 8, 128, 1, "bfloat16")


def test_native_facts_empty_unfilled_invalid_indices_and_constructor_errors():
    native = cudnn._pybind_module.VariantPackNative(1)
    assert _project(native, []) == []
    assert _project(native, [0]) == _accessor_projection(native, [0])
    with pytest.raises(IndexError):
        _project(native, [1])

    def reject(*args):
        raise RuntimeError("record constructor failed")

    with pytest.raises(RuntimeError, match="record constructor failed"):
        native._facts_as([0], reject, prep._DTYPE_BY_CODE)


def test_native_facts_records_are_independent_across_concurrent_packs():
    def project(i):
        native = cudnn._pybind_module.VariantPackNative(1)
        native.set_operand(0, 4096 * (i + 1), [i + 1, 8], [], 4, 16, 1, (i + 1) * 16, 2, i % 2)
        return _project(native, [0])[0]

    with ThreadPoolExecutor(4) as pool:
        actual = list(pool.map(project, range(32)))
    assert [f.ptr for f in actual] == [4096 * (i + 1) for i in range(32)]
    assert [f.shape for f in actual] == [(i + 1, 8) for i in range(32)]
    assert [f.device for f in actual] == [(2, i % 2) for i in range(32)]
