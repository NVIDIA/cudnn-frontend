# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks for byte/nibble storage through public backend dispatch.

The final launcher is intercepted. No synthetic address or CPU carrier reaches
CUDA; the same native validator runs at the real backend launch boundary.
"""

import cudnn
import pytest
import torch

from cudnn.engines.base import PlanConfig

pytestmark = pytest.mark.L0

# Physical bits per logical backend element, independent of framework itemsize.
_BITS = {
    cudnn.data_type.HALF: 16,
    cudnn.data_type.BFLOAT16: 16,
    cudnn.data_type.FLOAT: 32,
    cudnn.data_type.DOUBLE: 64,
    cudnn.data_type.INT8: 8,
    cudnn.data_type.UINT8: 8,
    cudnn.data_type.INT32: 32,
    cudnn.data_type.UINT32: 32,
    cudnn.data_type.INT64: 64,
    cudnn.data_type.FP8_E4M3: 8,
    cudnn.data_type.FP8_E5M2: 8,
    cudnn.data_type.FP8_E8M0: 8,
    cudnn.data_type.FP8_E5M3: 8,
    cudnn.data_type.FP4_E2M1: 4,
    cudnn.data_type.INT4: 4,
    cudnn.data_type.BOOLEAN: 1,
    cudnn.data_type.BYTE_BOOLEAN: 8,
}


class _LaunchRecorder:
    def __init__(self, graph, affine=True):
        self.graph, self.affine, self.calls = graph, affine, 0

    def _prepare_variant_pack_template(self):
        pass

    def _get_variant_pack_uids_sorted(self):
        return sorted(self.graph._tensor_by_uid)

    def _get_buffer_bounds(self):
        return [
            (uid, tensor.dim, tensor.stride, tensor.data_type, _BITS[tensor.data_type] if self.affine else 0, True)
            for uid, tensor in sorted(self.graph._tensor_by_uid.items())
        ]

    def _execute_with_raw_ptrs(self, *args):
        self.calls += 1

    def _execute_ordered_pack(self, pack, *args):
        pack._validate_backend_capacity()
        self.calls += 1


def _graph(dtype, shape, stride, affine=True):
    graph = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    operands = [graph.tensor(name=name, dim=shape, stride=stride, data_type=dtype) for name in ("a", "b")]
    output = graph.matmul(A=operands[0], B=operands[1]).set_output(True).set_dim(shape).set_stride(stride).set_data_type(dtype)
    operands.append(output)
    recorder = _LaunchRecorder(graph, affine)
    graph._lowered_graph = recorder
    graph._sorted_uids = sorted(t.get_uid() for t in operands)
    graph._plans = [PlanConfig(engine_id=0, knobs={}, cpp_index=0)]
    graph._plan_index = 0
    graph._planning_done = graph._is_built = True
    return graph, operands, recorder


@pytest.mark.parametrize("dtype,required", [(dtype, bits * 8) for dtype, bits in _BITS.items()])
@pytest.mark.parametrize("entry", ["execute", "execute_plan_at_index"])
@pytest.mark.parametrize("slot", [0, 2])
def test_byte_carrier_short_by_one_is_rejected(dtype, required, entry, slot):
    graph, tensors, recorder = _graph(dtype, [1, 8, 8], [64, 8, 1])
    storage = torch.empty(required + 32, dtype=torch.uint8)
    values = {tensor: storage[:required] for tensor in tensors}
    getattr(graph, entry)(values)
    assert recorder.calls == 1
    values[tensors[slot]] = storage[: required - 1]
    with pytest.raises(ValueError, match=f"tensor uid {tensors[slot].get_uid()} requires a {required}-byte buffer but received {required - 1} bytes"):
        getattr(graph, entry)(values)
    assert recorder.calls == 1
    values[tensors[slot]] = storage[:required].view(torch.float32)
    getattr(graph, entry)(values)
    assert recorder.calls == 2
    values[tensors[slot]] = storage.data_ptr()
    getattr(graph, entry)(values)
    assert recorder.calls == 3


@pytest.mark.parametrize("dtype,required", [(cudnn.data_type.HALF, 176), (cudnn.data_type.FP4_E2M1, 44)])
def test_declared_stride_holes_count_toward_the_buffer_bound(dtype, required):
    # Last element index is 7 * 12 + 3 = 87; FP4 packs two per byte.
    graph, tensors, recorder = _graph(dtype, [1, 8, 4], [96, 12, 1])
    values = {tensor: torch.empty(required, dtype=torch.uint8) for tensor in tensors}
    graph.execute(values)
    assert recorder.calls == 1
    values[tensors[0]] = torch.empty(required - 1, dtype=torch.uint8)
    with pytest.raises(ValueError, match=f"requires a {required}-byte buffer"):
        graph.execute(values)
    assert recorder.calls == 1


def test_special_layout_has_no_generic_affine_allocation_bound():
    graph, tensors, recorder = _graph(cudnn.data_type.HALF, [2, 4, 128], [512, 128, 1], affine=False)
    # Ragged/reordered allocation contracts cannot be inferred from this logical
    # envelope. Their family validation must not be replaced by a dense count.
    graph.execute({tensor: torch.empty(128, dtype=torch.uint8) for tensor in tensors})
    assert recorder.calls == 1


def test_special_layout_preserves_backend_override_spelling():
    graph, tensors, recorder = _graph(cudnn.data_type.HALF, [2, 4, 128], [512, 128, 1], affine=False)
    # Backend ragged metadata can combine logical dimensions with a physical
    # stride rank. Capacity checks must not reinterpret it as a dense tensor.
    graph.execute(
        {tensor: torch.empty(128, dtype=torch.uint8) for tensor in tensors},
        override_uids=[t.get_uid() for t in tensors],
        override_shapes=[[2, 4, 128]] * 3,
        override_strides=[[128, 1]] * 3,
    )
    assert recorder.calls == 1


@pytest.mark.parametrize("dtype,required", [(cudnn.data_type.BFLOAT16, 256), (cudnn.data_type.FP8_E4M3, 128), (cudnn.data_type.FP4_E2M1, 64)])
def test_override_cannot_enlarge_the_observed_allocation(dtype, required):
    graph, tensors, recorder = _graph(dtype, [1, 8, 8], [64, 8, 1])
    values = {tensor: torch.empty(required, dtype=torch.uint8) for tensor in tensors}
    overrides = dict(override_uids=[t.get_uid() for t in tensors], override_shapes=[[2, 8, 8]] * 3, override_strides=[[64, 8, 1]] * 3)
    graph.execute(values, **overrides)
    assert recorder.calls == 1
    values[tensors[0]] = torch.empty(required - 1, dtype=torch.uint8)
    with pytest.raises(ValueError, match=f"requires a {required}-byte buffer"):
        graph.execute(values, **overrides)
    assert recorder.calls == 1


def test_smaller_override_keeps_the_declared_storage_envelope():
    graph, tensors, recorder = _graph(cudnn.data_type.HALF, [2, 8, 8], [64, 8, 1])
    values = {tensor: torch.empty(256, dtype=torch.uint8) for tensor in tensors}
    overrides = dict(override_uids=[t.get_uid() for t in tensors], override_shapes=[[1, 8, 8]] * 3, override_strides=[[64, 8, 1]] * 3)
    graph.execute(values, **overrides)
    assert recorder.calls == 1
    values[tensors[-1]] = values[tensors[-1]][:128]
    with pytest.raises(ValueError, match="requires a 256-byte buffer but received 128 bytes"):
        graph.execute(values, **overrides)
    assert recorder.calls == 1


def test_backend_lowering_replaces_preexisting_unbounded_metadata():
    graph, tensors, recorder = _graph(cudnn.data_type.HALF, [1, 8, 8], [64, 8, 1])
    graph._lowered_graph = None
    values = {tensor.get_uid(): torch.empty(128, dtype=torch.uint8) for tensor in tensors}
    graph._normalize(values, None)
    graph._prepare_ordered_binding_schema()
    graph._lowered_graph = recorder
    values[tensors[0].get_uid()] = torch.empty(127, dtype=torch.uint8)
    with pytest.raises(ValueError, match="requires a 128-byte buffer"):
        graph.execute(values)
    assert recorder.calls == 0


class _DLPackOnly:
    def __init__(self, tensor):
        self.tensor = tensor

    def __dlpack__(self, *args, **kwargs):
        return self.tensor.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()


class _NegativeStride:
    shape, dtype = (64,), torch.float16

    def __init__(self, pointer):
        self.pointer = pointer

    def data_ptr(self):
        return self.pointer

    def stride(self):
        return (-1,)


@pytest.mark.parametrize("producer,observed", [("dlpack", 127), ("python_negative", 2), ("native_negative", 2)])
def test_observable_span_does_not_become_unknown(producer, observed):
    graph, tensors, recorder = _graph(cudnn.data_type.HALF, [1, 8, 8], [64, 8, 1])
    storage = torch.empty(128, dtype=torch.uint8)
    values = {tensor: storage for tensor in tensors}
    if producer == "dlpack":
        values[tensors[0]] = _DLPackOnly(storage[:127])
    elif producer == "python_negative":
        values[tensors[0]] = _NegativeStride(storage.data_ptr() + 126)
    else:
        values[tensors[0]] = cudnn._pybind_module.make_operand_buffer(storage.data_ptr() + 126, [64], 2, 16, 0, [-1])
    with pytest.raises(ValueError, match=f"requires a 128-byte buffer but received {observed} bytes"):
        graph.execute(values)
    assert recorder.calls == 0
