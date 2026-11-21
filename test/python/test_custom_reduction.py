# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import collections
import gc
import inspect
import itertools
import json
import threading
import weakref

import pytest

import cudnn

pytestmark = pytest.mark.L0

# The headers that the module was compiled with and the loaded library must both be 9.29 or later.
_custom_reduction_available = cudnn._pybind_module._compiled_cudnn_version >= 92900 and cudnn.backend_version() >= 92900

requires_cudnn_v9_29 = pytest.mark.skipif(
    not _custom_reduction_available,
    reason="Custom reduction requires cuDNN v9.29.0",
)


def test_custom_reduction_surface_is_public():
    assert hasattr(cudnn, "custom_reduction_identity")
    assert hasattr(cudnn.pygraph, "custom_reduction")
    assert "custom_reduction_identity" in cudnn.__all__
    assert hasattr(cudnn.NodeType, "CUSTOM_REDUCTION")
    assert hasattr(cudnn, "reduction_modifiers")
    assert "reduction_modifiers" in cudnn.__all__
    assert hasattr(cudnn.data_type, "BYTE_BOOLEAN")


def test_custom_reduction_node_type_preserves_existing_values():
    # Pin established values; future append-only members may follow CUSTOM_REDUCTION.
    assert cudnn.NodeType.BLOCK_SCALE_DEQUANTIZE.value == 42
    assert cudnn.NodeType.CUSTOM_REDUCTION.value == 57


def test_custom_reduction_refuses_a_cudnn_backend_older_than_9_29(cudnn_handle):
    if cudnn_handle is None:
        pytest.skip("cuDNN backend is unavailable")
    if _custom_reduction_available:
        pytest.skip("requires cuDNN headers or a cuDNN backend older than 9.29.0")

    # The API is present with every backend. Older headers or an older backend refuse the operation when the graph is
    # built, before the reducer runs.
    calls = []

    def counted_reducer(child, lhs, rhs):
        calls.append(1)
        return _add_reducer(child, lhs, rhs)

    graph = _float_graph()
    _float_streams(graph, 1, reducer=counted_reducer)
    graph.validate()
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="Custom reduction requires cuDNN v9.29.0"):
        graph._lowered_graph.build_operation_graph()
    assert calls == []


def test_custom_reduction_identity_is_typed_and_bit_exact():
    numeric_cases = [
        (cudnn.data_type.INT8, -1, "FF"),
        (cudnn.data_type.UINT8, 0xA5, "A5"),
        (cudnn.data_type.INT32, -2, "FFFFFFFE"),
        (cudnn.data_type.UINT32, 0xFFFFFFFF, "FFFFFFFF"),
        (cudnn.data_type.INT64, -3, "FFFFFFFFFFFFFFFD"),
        (cudnn.data_type.FLOAT, 1.0, "3F800000"),
        (cudnn.data_type.DOUBLE, 1.0, "3FF0000000000000"),
    ]
    for dtype, value, expected_bits in numeric_cases:
        identity = cudnn.custom_reduction_identity(dtype, value)
        assert identity.data_type == dtype
        assert identity.raw_hex == expected_bits

    raw_cases = [
        (cudnn.data_type.HALF, 0x3C01, "3C01"),
        (cudnn.data_type.BFLOAT16, 0x3F81, "3F81"),
        (cudnn.data_type.FLOAT, 0x80000000, "80000000"),
        (cudnn.data_type.DOUBLE, 0x8000000000000000, "8000000000000000"),
        (cudnn.data_type.FP8_E4M3, 0xA5, "A5"),
        (cudnn.data_type.FP8_E5M2, 0x5A, "5A"),
    ]
    for dtype, bits, expected_bits in raw_cases:
        identity = cudnn.custom_reduction_identity.from_bits(dtype, bits)
        assert identity.data_type == dtype
        assert identity.raw_hex == expected_bits


def test_custom_reduction_identity_rejects_lossy_or_invalid_construction():
    for dtype in (
        cudnn.data_type.HALF,
        cudnn.data_type.BFLOAT16,
        cudnn.data_type.FP8_E4M3,
        cudnn.data_type.FP8_E5M2,
    ):
        with pytest.raises(ValueError, match="from_bits"):
            cudnn.custom_reduction_identity(dtype, 1.0)

    with pytest.raises(ValueError, match="out of range"):
        cudnn.custom_reduction_identity(cudnn.data_type.INT8, 128)
    with pytest.raises(ValueError, match="out of range"):
        cudnn.custom_reduction_identity(cudnn.data_type.UINT32, -1)
    with pytest.raises(ValueError, match="do not fit"):
        cudnn.custom_reduction_identity.from_bits(cudnn.data_type.HALF, 0x10000)
    with pytest.raises(ValueError, match="Unsupported"):
        cudnn.custom_reduction_identity(cudnn.data_type.BOOLEAN, True)


def test_custom_reduction_public_signature():
    parameters = inspect.signature(cudnn.pygraph.custom_reduction).parameters
    assert list(parameters) == [
        "self",
        "inputs",
        "reducer",
        "reducer_semantic_key",
        "reduction_dims",
        "identity_values",
        "compute_data_types",
        "compute_data_type",
        "name",
    ]
    assert parameters["compute_data_types"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["compute_data_types"].default is None
    assert parameters["compute_data_type"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["compute_data_type"].default is None
    assert parameters["name"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["name"].default == ""


def test_custom_reduction_dynamic_stream_order_keepdims_and_uid_lowering(cudnn_handle):
    if cudnn_handle is None:
        pytest.skip("cuDNN backend is unavailable")

    def two_stream_reducer(child, lhs, rhs):
        return [
            child.add(
                a=lhs[0],
                b=rhs[0],
                compute_data_type=cudnn.data_type.FLOAT,
                name="stream_0_add",
            ),
            child.max(
                input0=lhs[1],
                input1=rhs[1],
                compute_data_type=cudnn.data_type.FLOAT,
                name="stream_1_max",
            ),
        ]

    graph = cudnn.pygraph(
        handle=cudnn_handle,
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    x0 = graph.tensor(dim=[2, 3, 4], stride=[12, 4, 1], data_type=cudnn.data_type.FLOAT, name="X0")
    x1 = graph.tensor(dim=[2, 3, 4], stride=[12, 4, 1], data_type=cudnn.data_type.FLOAT, name="X1")
    outputs = graph.custom_reduction(
        [x0, x1],
        two_stream_reducer,
        "test.python.custom_reduction.two_stream.v1",
        [0, 2],
        [
            cudnn.custom_reduction_identity(cudnn.data_type.FLOAT, 0.0),
            cudnn.custom_reduction_identity.from_bits(cudnn.data_type.FLOAT, 0xFF800000),
        ],
        compute_data_types=[cudnn.data_type.FLOAT, cudnn.data_type.FLOAT],
        name="custom_two_stream",
    )

    expected_dims = [[1, 3, 1], [1, 3, 1]]
    expected_strides = [[3, 1, 3], [3, 1, 3]]
    assert [tensor.get_dim() for tensor in outputs] == expected_dims
    assert [tensor.get_stride() for tensor in outputs] == expected_strides

    graph.validate()
    ordered_ir_tensors = [x0, x1, *outputs]
    assert [graph._cpp_tensors[tensor.get_uid()].get_uid() for tensor in ordered_ir_tensors] == [tensor.get_uid() for tensor in ordered_ir_tensors]
    cpp_outputs = [graph._cpp_tensors[tensor.get_uid()] for tensor in outputs]
    assert [tensor.get_dim() for tensor in cpp_outputs] == expected_dims
    assert [tensor.get_stride() for tensor in cpp_outputs] == expected_strides
    assert [tensor.get_dim() for tensor in outputs] == [tensor.get_dim() for tensor in cpp_outputs]
    assert [tensor.get_stride() for tensor in outputs] == [tensor.get_stride() for tensor in cpp_outputs]

    description = json.loads(repr(graph._lowered_graph))
    custom_node = next(node for node in description["nodes"] if node["tag"] == "CUSTOM_REDUCTION")
    input_ports = custom_node["inputs"]
    output_ports = custom_node["outputs"]
    assert isinstance(input_ports, dict)
    assert isinstance(output_ports, dict)
    expected_port_order = ["0", "1"]
    assert list(input_ports) == expected_port_order
    assert [input_ports[port] for port in expected_port_order] == [x0.get_uid(), x1.get_uid()]
    assert list(output_ports) == expected_port_order
    assert [output_ports[port] for port in expected_port_order] == [
        outputs[0].get_uid(),
        outputs[1].get_uid(),
    ]


@requires_cudnn_v9_29
def test_custom_reduction_executes_float_relu_body_with_default_upper_clip(cudnn_handle):
    if cudnn_handle is None:
        pytest.skip("cuDNN backend is unavailable")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Custom reduction requires the TensorIR MemBound engine on data-center Blackwell")

    # max(lhs, rhs) written as rhs + relu(lhs - rhs). Without RELU_FWD the body returns lhs, so the result would stay
    # at the identity. -1000 is an identity on either side because the data stay in [-25, 25].
    def relu_max_reducer(child, lhs, rhs):
        difference = child.sub(a=lhs[0], b=rhs[0], compute_data_type=cudnn.data_type.FLOAT, name="relu_max_sub")
        excess = child.relu(input=difference, compute_data_type=cudnn.data_type.FLOAT, name="relu_max_relu")
        return [child.add(a=rhs[0], b=excess, compute_data_type=cudnn.data_type.FLOAT, name="relu_max_add")]

    outer, reduced, inner = 4, 64, 8
    i = torch.arange(outer).view(outer, 1, 1)
    r = torch.arange(reduced).view(1, reduced, 1)
    k = torch.arange(inner).view(1, 1, inner)
    # Multiples of 0.5 in [-25, 25], so every difference and sum is exact. Column (0, 0) is all negative.
    x_cpu = ((37 * i + 11 * r + 5 * k) % 101).to(torch.float32) * 0.5 - 25.0
    x_cpu[0, :, 0] = -1.0 - (torch.arange(reduced) % 40).to(torch.float32) * 0.5
    expected = torch.amax(x_cpu, dim=1, keepdim=True)

    graph = cudnn.pygraph(
        handle=cudnn_handle,
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    x = graph.tensor(dim=list(x_cpu.shape), stride=list(x_cpu.stride()), data_type=cudnn.data_type.FLOAT, name="X")
    y = graph.custom_reduction(
        [x],
        relu_max_reducer,
        "test.python.custom_reduction.relu_max.v1",
        [1],
        [cudnn.custom_reduction_identity(cudnn.data_type.FLOAT, -1000.0)],
        compute_data_types=[cudnn.data_type.FLOAT],
        name="custom_relu_max",
    )[0]
    y.set_output(True)
    assert y.get_dim() == [outer, 1, inner]
    graph.build([cudnn.heur_mode.A])

    x_gpu = x_cpu.cuda()
    y_gpu = torch.full(list(expected.shape), float("nan"), dtype=torch.float32, device="cuda")
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    torch.cuda.synchronize()  # The fixture handle uses a separate nonblocking stream.
    graph.execute({x: x_gpu, y: y_gpu}, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()
    assert torch.equal(y_gpu.cpu(), expected)


@requires_cudnn_v9_29
def test_two_custom_reducers_capture_distinct_callbacks_until_build(cudnn_handle):
    if cudnn_handle is None:
        pytest.skip("cuDNN backend is unavailable")

    callback_calls = []
    retained_child_proxies = []

    def add_reducer(child, lhs, rhs):
        callback_calls.append("ADD")
        retained_child_proxies.append(child)
        return [
            child.add(
                a=lhs[0],
                b=rhs[0],
                compute_data_type=cudnn.data_type.FLOAT,
                name="python_add_reducer",
            )
        ]

    def max_reducer(child, lhs, rhs):
        callback_calls.append("MAX")
        retained_child_proxies.append(child)
        return [
            child.max(
                input0=lhs[0],
                input1=rhs[0],
                compute_data_type=cudnn.data_type.FLOAT,
                name="python_max_reducer",
            )
        ]

    graph = cudnn.pygraph(
        handle=cudnn_handle,
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    x_add = graph.tensor(dim=[2, 4], stride=[4, 1], data_type=cudnn.data_type.FLOAT, name="X_ADD")
    x_max = graph.tensor(dim=[2, 4], stride=[4, 1], data_type=cudnn.data_type.FLOAT, name="X_MAX")

    add_output = graph.custom_reduction(
        [x_add],
        add_reducer,
        "test.python.custom_reduction.add.v1",
        [1],
        [cudnn.custom_reduction_identity(cudnn.data_type.FLOAT, 0.0)],
        compute_data_types=[cudnn.data_type.FLOAT],
        name="custom_add",
    )[0]
    max_output = graph.custom_reduction(
        [x_max],
        max_reducer,
        "test.python.custom_reduction.max.v1",
        [1],
        [cudnn.custom_reduction_identity(cudnn.data_type.FLOAT, float("-inf"))],
        compute_data_types=[cudnn.data_type.FLOAT],
        name="custom_max",
    )[0]
    add_output.set_output(True)
    max_output.set_output(True)

    assert add_output.get_dim() == [2, 1]
    assert max_output.get_dim() == [2, 1]

    # validate() lowers the public Python IR and installs each reducer on its
    # C++ node, but expansion (and therefore callback invocation) happens only
    # in build_operation_graph(). Retain only the lowered graph before build so
    # the C++ per-node captures are responsible for callable lifetime.
    graph.validate()
    assert callback_calls == []
    lowered_graph = graph._lowered_graph
    del graph, add_reducer, max_reducer
    gc.collect()

    lowered_graph.build_operation_graph()
    assert callback_calls == ["ADD", "MAX"]
    assert len(retained_child_proxies) == 2

    # The callback's logical graph-building window is over, but retaining its
    # proxy must remain memory-safe. This specifically guards against passing a
    # Python reference to a stack-local PyGraph.
    del lowered_graph
    gc.collect()
    assert all("nodes" in json.loads(repr(proxy._target)) for proxy in retained_child_proxies)


def test_custom_reduction_floating_identity_is_exact():
    exact_cases = [
        (cudnn.data_type.FLOAT, 0.5, "3F000000"),
        (cudnn.data_type.FLOAT, -2, "C0000000"),
        (cudnn.data_type.FLOAT, float("inf"), "7F800000"),
        (cudnn.data_type.FLOAT, float("-inf"), "FF800000"),
        (cudnn.data_type.FLOAT, 3.4028234663852886e38, "7F7FFFFF"),
        (cudnn.data_type.DOUBLE, 0.1, "3FB999999999999A"),
        (cudnn.data_type.DOUBLE, float("-inf"), "FFF0000000000000"),
    ]
    for dtype, value, expected_bits in exact_cases:
        identity = cudnn.custom_reduction_identity(dtype, value)
        assert identity.data_type == dtype
        assert identity.raw_hex == expected_bits

    for dtype in (cudnn.data_type.FLOAT, cudnn.data_type.DOUBLE):
        with pytest.raises(TypeError, match="must be a real number"):
            cudnn.custom_reduction_identity(dtype, True)
        with pytest.raises(TypeError, match="must be a real number"):
            cudnn.custom_reduction_identity(dtype, "1.0")
        with pytest.raises(TypeError, match="must be a real number"):
            cudnn.custom_reduction_identity(dtype, 1 + 0j)
        with pytest.raises(ValueError, match="NaN, which is not an identity"):
            cudnn.custom_reduction_identity(dtype, float("nan"))

    with pytest.raises(ValueError, match="not exactly representable as FLOAT"):
        cudnn.custom_reduction_identity(cudnn.data_type.FLOAT, 0.1)
    with pytest.raises(ValueError, match="not exactly representable as FLOAT"):
        cudnn.custom_reduction_identity(cudnn.data_type.FLOAT, 2**24 + 1)
    with pytest.raises(ValueError, match="not exactly representable as DOUBLE"):
        cudnn.custom_reduction_identity(cudnn.data_type.DOUBLE, 2**53 + 1)
    with pytest.raises(ValueError, match="out of range for FLOAT"):
        cudnn.custom_reduction_identity(cudnn.data_type.FLOAT, 1e40)
    with pytest.raises(ValueError, match="out of range for DOUBLE"):
        cudnn.custom_reduction_identity(cudnn.data_type.DOUBLE, 2**1024)
    with pytest.raises(ValueError, match="out of range for INT64"):
        cudnn.custom_reduction_identity(cudnn.data_type.INT64, 2**70)
    with pytest.raises(ValueError, match="out of range for UINT32"):
        cudnn.custom_reduction_identity(cudnn.data_type.UINT32, 2**64)


def test_custom_reduction_identity_accepts_numpy_scalars():
    np = pytest.importorskip("numpy")
    float32_max = np.finfo(np.float32).max
    exact_cases = [
        (cudnn.data_type.FLOAT, np.float32(0.5), "3F000000"),
        (cudnn.data_type.FLOAT, np.float32(0.1), "3DCCCCCD"),
        (cudnn.data_type.FLOAT, float32_max, "7F7FFFFF"),
        (cudnn.data_type.FLOAT, np.finfo(np.float32).min, "FF7FFFFF"),
        (cudnn.data_type.FLOAT, np.float32("-inf"), "FF800000"),
        (cudnn.data_type.FLOAT, np.int64(-2), "C0000000"),
        (cudnn.data_type.DOUBLE, np.float32(0.1), "3FB99999A0000000"),
        (cudnn.data_type.DOUBLE, np.float16(-1.5), "BFF8000000000000"),
        (cudnn.data_type.INT8, np.int8(-1), "FF"),
        (cudnn.data_type.UINT8, np.uint8(0xA5), "A5"),
        (cudnn.data_type.INT32, np.int64(-2), "FFFFFFFE"),
        (cudnn.data_type.UINT32, np.uint32(0xFFFFFFFF), "FFFFFFFF"),
        (cudnn.data_type.INT64, np.int64(-3), "FFFFFFFFFFFFFFFD"),
    ]
    for dtype, value, expected_bits in exact_cases:
        identity = cudnn.custom_reduction_identity(dtype, value)
        assert identity.data_type == dtype
        assert identity.raw_hex == expected_bits

    for dtype in (cudnn.data_type.FLOAT, cudnn.data_type.DOUBLE, cudnn.data_type.INT32):
        with pytest.raises(TypeError, match="must be"):
            cudnn.custom_reduction_identity(dtype, np.bool_(True))
    with pytest.raises(TypeError, match="must be an integer"):
        cudnn.custom_reduction_identity(cudnn.data_type.INT32, np.float32(1.0))
    with pytest.raises(ValueError, match="NaN, which is not an identity"):
        cudnn.custom_reduction_identity(cudnn.data_type.FLOAT, np.float32("nan"))
    with pytest.raises(ValueError, match="not exactly representable as FLOAT"):
        cudnn.custom_reduction_identity(cudnn.data_type.FLOAT, np.float64(0.1))
    with pytest.raises(ValueError, match="out of range for FLOAT"):
        cudnn.custom_reduction_identity(cudnn.data_type.FLOAT, np.float64(float32_max) * 2)
    with pytest.raises(ValueError, match="out of range for INT8"):
        cudnn.custom_reduction_identity(cudnn.data_type.INT8, np.int16(128))
    with pytest.raises(ValueError, match="out of range for INT64"):
        cudnn.custom_reduction_identity(cudnn.data_type.INT64, np.uint64(2**63))
    if np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant:
        with pytest.raises(ValueError, match="not exactly representable as DOUBLE"):
            cudnn.custom_reduction_identity(cudnn.data_type.DOUBLE, np.longdouble(1) / 3)


def test_custom_reduction_byte_boolean_identity():
    true_identity = cudnn.custom_reduction_identity(cudnn.data_type.BYTE_BOOLEAN, True)
    false_identity = cudnn.custom_reduction_identity(cudnn.data_type.BYTE_BOOLEAN, False)
    assert true_identity.data_type == cudnn.data_type.BYTE_BOOLEAN
    assert (true_identity.raw_hex, false_identity.raw_hex) == ("01", "00")

    for bits, expected_bits in ((0x00, "00"), (0x01, "01"), (0xFF, "FF")):
        identity = cudnn.custom_reduction_identity.from_bits(cudnn.data_type.BYTE_BOOLEAN, bits)
        assert identity.data_type == cudnn.data_type.BYTE_BOOLEAN
        assert identity.raw_hex == expected_bits

    with pytest.raises(ValueError, match="do not fit"):
        cudnn.custom_reduction_identity.from_bits(cudnn.data_type.BYTE_BOOLEAN, 0x100)
    for value in (1, 0, 1.0):
        with pytest.raises(TypeError, match="BYTE_BOOLEAN must be a bool"):
            cudnn.custom_reduction_identity(cudnn.data_type.BYTE_BOOLEAN, value)


@requires_cudnn_v9_29
def test_custom_reduction_byte_boolean_stream_builds(cudnn_handle):
    if cudnn_handle is None:
        pytest.skip("cuDNN backend is unavailable")

    def or_reducer(child, lhs, rhs):
        result = child.logical_or(lhs[0], rhs[0], compute_data_type=cudnn.data_type.BYTE_BOOLEAN, name="pred_or")
        result.set_data_type(cudnn.data_type.BOOLEAN)
        return [result]

    graph = _float_graph()
    x = graph.tensor(dim=[4, 31], stride=[31, 1], data_type=cudnn.data_type.BYTE_BOOLEAN, name="PRED")
    y = graph.custom_reduction(
        [x],
        or_reducer,
        "test.python.custom_reduction.pred_or.v1",
        [1],
        [cudnn.custom_reduction_identity(cudnn.data_type.BYTE_BOOLEAN, False)],
    )[0]
    y.set_output(True)
    graph.validate()
    assert y.get_dim() == [4, 1]
    node = _custom_node(graph)
    assert (node["identity_types"], node["identity_raw"]) == (["BYTE_BOOLEAN"], ["00"])
    # The backend decides whether it supports a BYTE_BOOLEAN reduction; the frontend forwards its decision.
    try:
        graph._lowered_graph.build_operation_graph()
    except cudnn.cudnnGraphNotSupportedError as error:
        assert "CUDNN_STATUS_NOT_SUPPORTED" in str(error)


def _float_graph():
    return cudnn.pygraph(
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )


def _add_reducer(child, lhs, rhs):
    return [child.add(lhs[i], rhs[i], compute_data_type=cudnn.data_type.FLOAT, name=f"add{i}") for i in range(len(lhs))]


def _float_streams(graph, count, reducer=_add_reducer, **kwargs):
    inputs = [graph.tensor(dim=[2, 3], stride=[3, 1], data_type=cudnn.data_type.FLOAT, name=f"X{i}") for i in range(count)]
    outputs = graph.custom_reduction(
        inputs,
        reducer,
        "test.python.custom_reduction.streams.v1",
        [1],
        [cudnn.custom_reduction_identity(cudnn.data_type.FLOAT, 0.0)] * count,
        **kwargs,
    )
    if outputs:
        outputs[-1].set_output(True)
    return outputs


def _custom_node(graph):
    description = json.loads(repr(graph._lowered_graph))
    return next(node for node in description["nodes"] if node["tag"] == "CUSTOM_REDUCTION")


def test_custom_reduction_stream_count_is_1_to_8():
    with pytest.raises(ValueError, match="at least one input stream"):
        _float_streams(_float_graph(), 0)

    graph = _float_graph()
    assert len(_float_streams(graph, 8)) == 8
    graph.validate()

    graph = _float_graph()
    _float_streams(graph, 9)
    with pytest.raises(RuntimeError, match="at most 8 input streams; got 9"):
        graph.validate()


def test_custom_reduction_refuses_null_inputs_reducers_and_results(cudnn_handle):
    graph = _float_graph()
    x = graph.tensor(dim=[2, 3], stride=[3, 1], data_type=cudnn.data_type.FLOAT, name="X")
    identity = cudnn.custom_reduction_identity(cudnn.data_type.FLOAT, 0.0)
    with pytest.raises(ValueError, match="input stream 1 is not set"):
        graph.custom_reduction([x, None], _add_reducer, "test.null_input.v1", [1], [identity, identity])
    with pytest.raises(TypeError, match="reducer must be callable"):
        graph.custom_reduction([x], None, "test.null_reducer.v1", [1], [identity])

    if cudnn_handle is None:
        pytest.skip("cuDNN backend is unavailable")
    if not _custom_reduction_available:
        pytest.skip("Custom reduction requires cuDNN v9.29.0")
    graph = _float_graph()
    _float_streams(graph, 1, reducer=lambda child, lhs, rhs: [None])
    graph.validate()
    with pytest.raises(RuntimeError, match="returned a null tensor for stream 0"):
        graph._lowered_graph.build_operation_graph()


def test_custom_reduction_compute_type_spellings_and_class():
    graph = _float_graph()
    _float_streams(graph, 2, compute_data_type=cudnn.data_type.FLOAT)
    graph.validate()
    assert _custom_node(graph)["compute_data_types"] == ["FLOAT", "FLOAT"]

    graph = _float_graph()
    _float_streams(graph, 2)
    graph.validate()
    assert _custom_node(graph)["compute_data_types"] == []

    with pytest.raises(ValueError, match="compute_data_types or compute_data_type, not both"):
        _float_streams(
            _float_graph(),
            1,
            compute_data_types=[cudnn.data_type.FLOAT],
            compute_data_type=cudnn.data_type.FLOAT,
        )

    graph = _float_graph()
    _float_streams(graph, 1, compute_data_type=cudnn.data_type.INT32)
    with pytest.raises(RuntimeError, match="same integer or floating-point class"):
        graph.validate()


@requires_cudnn_v9_29
def test_custom_reduction_body_constants(cudnn_handle):
    if cudnn_handle is None:
        pytest.skip("cuDNN backend is unavailable")

    def floor_reducer(scalar_type):
        def reducer(child, lhs, rhs):
            larger = child.max(lhs[0], rhs[0], compute_data_type=cudnn.data_type.FLOAT, name="body_max")
            floor = child.tensor_scalar(-1000.0, scalar_type)
            return [child.max(larger, floor, compute_data_type=cudnn.data_type.FLOAT, name="body_floor")]

        return reducer

    graph = _float_graph()
    _float_streams(graph, 1, reducer=floor_reducer(cudnn.scalar_type.COMPILE_TIME_CONST))
    graph.validate()
    graph._lowered_graph.build_operation_graph()

    graph = _float_graph()
    _float_streams(graph, 1, reducer=floor_reducer(cudnn.scalar_type.RUNTIME_PARAM))
    graph.validate()
    with pytest.raises(RuntimeError, match="must be virtual or a compile-time constant"):
        graph._lowered_graph.build_operation_graph()


def test_custom_reduction_modifiers_match_the_cpp_keys():
    for helper, key in (
        (cudnn.reduction_modifiers.sum, "cudnn_frontend.reduction_modifiers.sum.v1"),
        (cudnn.reduction_modifiers.max, "cudnn_frontend.reduction_modifiers.max.v1"),
        (cudnn.reduction_modifiers.min, "cudnn_frontend.reduction_modifiers.min.v1"),
    ):
        arguments = helper()
        assert set(arguments) == {"reducer", "reducer_semantic_key"}
        assert arguments["reducer_semantic_key"] == key
        assert callable(arguments["reducer"])


@requires_cudnn_v9_29
def test_custom_reduction_modifiers_execute(cudnn_handle):
    if cudnn_handle is None:
        pytest.skip("cuDNN backend is unavailable")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Custom reduction requires the TensorIR MemBound engine on data-center Blackwell")

    x_cpu = ((torch.arange(4 * 64 * 8).view(4, 64, 8) * 37) % 101).to(torch.float32) - 50.0
    cases = (
        ("sum", 0.0, torch.sum(x_cpu, dim=1, keepdim=True)),
        ("max", float("-inf"), torch.amax(x_cpu, dim=1, keepdim=True)),
        ("min", float("inf"), torch.amin(x_cpu, dim=1, keepdim=True)),
    )
    for helper_name, identity, expected in cases:
        graph = cudnn.pygraph(
            handle=cudnn_handle,
            io_data_type=cudnn.data_type.FLOAT,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        x = graph.tensor(dim=list(x_cpu.shape), stride=list(x_cpu.stride()), data_type=cudnn.data_type.FLOAT, name="X")
        y = graph.custom_reduction(
            [x],
            reduction_dims=[1],
            identity_values=[cudnn.custom_reduction_identity(cudnn.data_type.FLOAT, identity)],
            name=f"modifier_{helper_name}",
            **getattr(cudnn.reduction_modifiers, helper_name)(),
        )[0]
        y.set_output(True)
        graph.build([cudnn.heur_mode.A])
        assert _custom_node(graph)["reducer_semantic_key"] == f"cudnn_frontend.reduction_modifiers.{helper_name}.v1"

        x_gpu = x_cpu.cuda()
        y_gpu = torch.full(list(expected.shape), float("nan"), dtype=torch.float32, device="cuda")
        workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
        torch.cuda.synchronize()
        graph.execute({x: x_gpu, y: y_gpu}, workspace, handle=cudnn_handle)
        torch.cuda.synchronize()
        assert torch.equal(y_gpu.cpu(), expected), helper_name


@requires_cudnn_v9_29
def test_custom_reduction_docs_argmax_executes(cudnn_handle):
    if cudnn_handle is None:
        pytest.skip("cuDNN backend is unavailable")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Custom reduction requires the TensorIR MemBound engine on data-center Blackwell")

    # Values in a small integer range, so most rows hold their maximum more than once.
    x_cpu = ((torch.arange(4 * 64 * 8).view(4, 64, 8) * 37) % 23).to(torch.float32)
    idx_cpu = torch.arange(64, dtype=torch.int32).view(1, 64, 1).expand(4, 64, 8).contiguous()
    expected = torch.argmax(x_cpu, dim=1, keepdim=True).to(torch.int32)

    graph = cudnn.pygraph(
        handle=cudnn_handle,
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    x = graph.tensor(dim=list(x_cpu.shape), stride=list(x_cpu.stride()), data_type=cudnn.data_type.FLOAT, name="X")
    idx = graph.tensor(dim=list(idx_cpu.shape), stride=list(idx_cpu.stride()), data_type=cudnn.data_type.INT32, name="IDX")
    # The argmax example of docs/operations/CustomReduction.md.
    value, index = graph.custom_reduction(
        [x, idx],
        reduction_dims=[1],
        **cudnn.reduction_modifiers.argmax(cudnn.data_type.FLOAT),
    )
    index.set_output(True)
    graph.build([cudnn.heur_mode.A])

    x_gpu, idx_gpu = x_cpu.cuda(), idx_cpu.cuda()
    index_gpu = torch.full(list(expected.shape), -1, dtype=torch.int32, device="cuda")
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    torch.cuda.synchronize()
    graph.execute({x: x_gpu, idx: idx_gpu, index: index_gpu}, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()
    assert torch.equal(index_gpu.cpu(), expected)


@requires_cudnn_v9_29
def test_custom_reduction_plan_serialize_roundtrip(cudnn_handle):
    if cudnn_handle is None:
        pytest.skip("cuDNN backend is unavailable")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Custom reduction requires the TensorIR MemBound engine on data-center Blackwell")

    calls = []
    argmax = cudnn.reduction_modifiers.argmax(cudnn.data_type.FLOAT)

    def counted_argmax(child, lhs, rhs):
        calls.append(1)
        return argmax["reducer"](child, lhs, rhs)

    x_cpu = ((torch.arange(4 * 64 * 8).view(4, 64, 8) * 37) % 23).to(torch.float32)
    idx_cpu = torch.arange(64, dtype=torch.int32).view(1, 64, 1).expand(4, 64, 8).contiguous()
    expected = torch.argmax(x_cpu, dim=1, keepdim=True).to(torch.int32)

    graph = cudnn.pygraph(
        handle=cudnn_handle,
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    x = graph.tensor(dim=list(x_cpu.shape), stride=list(x_cpu.stride()), data_type=cudnn.data_type.FLOAT, name="X")
    idx = graph.tensor(dim=list(idx_cpu.shape), stride=list(idx_cpu.stride()), data_type=cudnn.data_type.INT32, name="IDX")
    _, index = graph.custom_reduction(
        [x, idx],
        counted_argmax,
        "test.python.custom_reduction.argmax_roundtrip.v1",
        [1],
        argmax["identity_values"],
    )
    index.set_output(True)
    graph.build([cudnn.heur_mode.A])

    x_gpu, idx_gpu = x_cpu.cuda(), idx_cpu.cuda()
    original = torch.full(list(expected.shape), -1, dtype=torch.int32, device="cuda")
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    torch.cuda.synchronize()
    graph.execute({x: x_gpu, idx: idx_gpu, index: original}, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()
    assert torch.equal(original.cpu(), expected)

    data = graph.serialize()
    calls_after_build = len(calls)
    assert calls_after_build >= 1

    restored_handle = cudnn.create_handle()
    try:
        restored_graph = cudnn.pygraph(handle=restored_handle)
        restored_graph.deserialize(restored_handle, data, enforce_precompiled=True)
        restored = torch.full(list(expected.shape), -1, dtype=torch.int32, device="cuda")
        restored_workspace = torch.empty(restored_graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
        torch.cuda.synchronize()
        restored_graph.execute({x: x_gpu, idx: idx_gpu, index: restored}, restored_workspace, handle=restored_handle)
        torch.cuda.synchronize()
    finally:
        cudnn.destroy_handle(restored_handle)

    assert torch.equal(restored, original)
    assert torch.equal(restored.cpu(), expected)
    assert len(calls) == calls_after_build


def _reduction_after_relu(graph, output_stride=None):
    x = graph.tensor(dim=[2, 64, 3], stride=[192, 3, 1], data_type=cudnn.data_type.FLOAT, name="X")
    activation = graph.relu(x, compute_data_type=cudnn.data_type.FLOAT, name="producer")
    y = graph.custom_reduction(
        [activation],
        _add_reducer,
        "test.python.custom_reduction.after_relu.v1",
        [1],
        [cudnn.custom_reduction_identity(cudnn.data_type.FLOAT, 0.0)],
        name="after_relu",
    )[0]
    # The input shape comes from the producer, so the output shape is known only at validate().
    assert y.get_dim() == []
    y.set_output(True)
    if output_stride is not None:
        y.set_stride(output_stride)
    return x, y


def test_custom_reduction_keeps_an_assigned_output_stride_after_shape_inference(cudnn_handle):
    if cudnn_handle is None:
        pytest.skip("cuDNN backend is unavailable")

    for output_stride, expected_stride in ((None, [3, 1, 1]), ([10, 6, 2], [10, 6, 2])):
        graph = _float_graph()
        _, y = _reduction_after_relu(graph, output_stride)
        graph.validate()
        assert y.get_dim() == [2, 1, 3]
        assert y.get_stride() == expected_stride
        assert graph._cpp_tensors[y.get_uid()].get_stride() == expected_stride

    if not _custom_reduction_available:
        pytest.skip("Custom reduction requires cuDNN v9.29.0")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Custom reduction requires the TensorIR MemBound engine on data-center Blackwell")

    x_gpu = torch.arange(2 * 64 * 3, dtype=torch.float32, device="cuda").view(2, 64, 3)
    expected = x_gpu.sum(dim=1, keepdim=True)
    for raw_pointers in (False, True):
        graph = cudnn.pygraph(
            handle=cudnn_handle,
            io_data_type=cudnn.data_type.FLOAT,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        x, y = _reduction_after_relu(graph, [10, 6, 2])
        graph.build([cudnn.heur_mode.A])
        # 15 elements hold the padded [2, 1, 3] layout; the 9 padding elements must keep the sentinel.
        storage = torch.full([15], -777.0, dtype=torch.float32, device="cuda")
        y_gpu = storage.as_strided([2, 1, 3], [10, 6, 2])
        workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
        torch.cuda.synchronize()
        if raw_pointers:
            graph.execute({x: x_gpu.data_ptr(), y: y_gpu.data_ptr()}, workspace.data_ptr(), handle=cudnn_handle)
        else:
            graph.execute({x: x_gpu, y: y_gpu}, workspace, handle=cudnn_handle)
        torch.cuda.synchronize()
        assert torch.equal(y_gpu, expected), raw_pointers
        written = torch.zeros(15, dtype=torch.bool, device="cuda")
        written.as_strided([2, 1, 3], [10, 6, 2]).fill_(True)
        assert torch.all(storage[~written] == -777.0), raw_pointers


def test_custom_reduction_infers_the_native_stride_for_an_assigned_dim(cudnn_handle):
    if cudnn_handle is None:
        pytest.skip("cuDNN backend is unavailable")

    # The caller sets the output dim but no stride. As in C++, an unset stride is the packed NHWC stride of the dim.
    graph = _float_graph()
    _, y = _reduction_after_relu(graph)
    y.set_dim([2, 1, 3])
    graph.validate()
    assert y.get_stride() == [3, 1, 1]
    assert graph._cpp_tensors[y.get_uid()].get_stride() == [3, 1, 1]


@requires_cudnn_v9_29
def test_custom_reduction_lowered_graph_has_no_uncollectable_cycle(cudnn_handle):
    if cudnn_handle is None:
        pytest.skip("cuDNN backend is unavailable")

    # The lowering adapter should release the native graph when the reducer itself captures no parent state.
    for expand in (False, True):
        graph = _float_graph()
        _float_streams(graph, 1)
        graph.validate()
        if expand:
            graph._lowered_graph.build_operation_graph()
        lowered_graph = weakref.ref(graph._lowered_graph)
        del graph
        gc.collect()
        assert lowered_graph() is None, f"expand={expand}"


@requires_cudnn_v9_29
@pytest.mark.parametrize("state", ["unbuilt", "built", "failed"])
def test_custom_reduction_collects_a_callback_that_reads_its_parent(cudnn_handle, state):
    if cudnn_handle is None:
        pytest.skip("cuDNN backend is unavailable")

    def construct_and_drop():
        graph = _float_graph()

        def reducer(child, lhs, rhs):
            # Reading the captured graph is allowed; no parent mutation is involved.
            gc.collect()
            assert graph is not None
            if state == "failed":
                raise ValueError("retained callback failure")
            return _add_reducer(child, lhs, rhs)

        _float_streams(graph, 1, reducer=reducer)
        graph.validate()
        lowered = graph._lowered_graph
        if state == "built":
            lowered.build_operation_graph()
        elif state == "failed":
            with pytest.raises(RuntimeError, match="retained callback failure"):
                lowered.build_operation_graph()
        return weakref.ref(graph), weakref.ref(lowered)

    references = construct_and_drop()
    gc.collect()
    assert all(reference() is None for reference in references)


@requires_cudnn_v9_29
def test_custom_reduction_retained_child_releases_its_owner_after_the_child_is_dropped(cudnn_handle):
    if cudnn_handle is None:
        pytest.skip("cuDNN backend is unavailable")

    retained = []

    def construct_and_drop():
        graph = _float_graph()

        def reducer(child, lhs, rhs):
            retained.append(child)
            return _add_reducer(child, lhs, rhs)

        _float_streams(graph, 1, reducer=reducer)
        graph.validate()
        graph._lowered_graph.build_operation_graph()
        return weakref.ref(graph._lowered_graph)

    reference = construct_and_drop()
    gc.collect()
    assert len(retained) == 1
    assert reference() is not None
    retained.clear()
    gc.collect()
    assert reference() is None


@requires_cudnn_v9_29
def test_custom_reduction_added_to_a_temporary_sdpa_child_keeps_its_callback(cudnn_handle):
    if cudnn_handle is None:
        pytest.skip("cuDNN backend is unavailable")

    calls = []

    def construct_and_drop():
        graph = cudnn._pybind_module.backend_graph(
            io_data_type=cudnn.data_type.HALF,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )

        def reducer(child, lhs, rhs):
            gc.collect()
            calls.append(1)
            return _add_reducer(child, lhs, rhs)

        def score_mod(child, scores):
            total = child.custom_reduction(
                [scores],
                reducer,
                "test.python.sdpa_child_lifetime.v1",
                [3],
                [cudnn.custom_reduction_identity(cudnn.data_type.FLOAT, 0.0)],
            )[0]
            return child.add(scores, total)

        q, k, v = (graph.tensor(name=name, dim=[1, 1, 64, 64], stride=[4096, 4096, 64, 1], data_type=cudnn.data_type.HALF) for name in "qkv")
        output, _ = graph.sdpa(
            q=q,
            k=k,
            v=v,
            attn_scale=0.125,
            generate_stats=False,
            compute_data_type=cudnn.data_type.FLOAT,
            score_mod=score_mod,
            implementation=cudnn.attention_implementation.COMPOSITE,
        )
        output.set_output(True).set_data_type(cudnn.data_type.HALF)
        graph.validate()
        lowered = graph
        try:
            lowered.build_operation_graph()
        except cudnn.cudnnGraphNotSupportedError as error:
            # This control checks callback lifetime before the backend's pattern decision.
            assert "CUDNN_STATUS_NOT_SUPPORTED" in str(error)
        assert calls == [1]
        return weakref.ref(lowered)

    reference = construct_and_drop()
    gc.collect()
    assert reference() is None


@requires_cudnn_v9_29
def test_custom_reduction_reducer_may_use_a_captured_parent_scalar(cudnn_handle):
    if cudnn_handle is None:
        pytest.skip("cuDNN backend is unavailable")

    graph = _float_graph()
    floor = graph.tensor_scalar(-1000.0, cudnn.scalar_type.COMPILE_TIME_CONST)
    lowered_floors = []

    def floor_reducer(child, lhs, rhs):
        larger = child.max(lhs[0], rhs[0], compute_data_type=cudnn.data_type.FLOAT, name="captured_max")
        result = child.max(larger, floor, compute_data_type=cudnn.data_type.FLOAT, name="captured_floor")
        lowered_floors.append(child._lower(floor))
        return [result]

    _float_streams(graph, 1, reducer=floor_reducer)
    graph.validate()
    # The reducer runs while the lowered graph builds, after the public graph is gone. It still lowers the captured
    # parent tensor on demand.
    lowered_graph = graph._lowered_graph
    public_graph = weakref.ref(graph)
    del graph
    gc.collect()
    assert public_graph() is None
    lowered_graph.build_operation_graph()
    assert len(lowered_floors) == 1
    assert lowered_floors[0].get_uid() == floor.get_uid()


@requires_cudnn_v9_29
def test_custom_reduction_reducer_exception_is_reported_and_the_next_build_retries(cudnn_handle):
    if cudnn_handle is None:
        pytest.skip("cuDNN backend is unavailable")

    calls = []

    def failing_once_reducer(child, lhs, rhs):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("python reducer failure")
        return _add_reducer(child, lhs, rhs)

    graph = _float_graph()
    _float_streams(graph, 1, reducer=failing_once_reducer)
    graph.validate()
    # The native build returns an error status; the binding raises it with the exception text of the reducer.
    with pytest.raises(RuntimeError, match=r"reducer callback threw an exception: (?s:.*)ValueError: python reducer failure"):
        graph._lowered_graph.build_operation_graph()
    assert len(calls) == 1
    graph._lowered_graph.build_operation_graph()
    assert len(calls) == 2


@requires_cudnn_v9_29
def test_custom_reduction_refuses_a_nested_reducer_when_the_reducer_builds_its_child_graph(cudnn_handle):
    if cudnn_handle is None:
        pytest.skip("cuDNN backend is unavailable")

    identity = cudnn.custom_reduction_identity(cudnn.data_type.FLOAT, 0.0)
    inner_calls = []
    child_errors = []

    def inner_reducer(child, lhs, rhs):
        inner_calls.append(1)
        # No result: a build that runs this reducer stops before any backend call.
        return []

    def outer_reducer(child, lhs, rhs):
        outputs = child.custom_reduction([lhs[0]], inner_reducer, "test.python.nested_inner.v1", [0], [identity], name="nested")
        try:
            child.build_operation_graph()
        except Exception as error:
            child_errors.append(error)
        return outputs

    graph = _float_graph()
    _float_streams(graph, 1, reducer=outer_reducer)
    graph.validate()
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="may contain only pointwise operations"):
        graph._lowered_graph.build_operation_graph()
    assert inner_calls == []
    assert len(child_errors) == 1
    assert isinstance(child_errors[0], cudnn.cudnnGraphNotSupportedError)
    assert "may contain only pointwise operations" in str(child_errors[0])


@requires_cudnn_v9_29
def test_custom_reduction_refuses_a_self_nesting_reducer_without_recursion(cudnn_handle):
    if cudnn_handle is None:
        pytest.skip("cuDNN backend is unavailable")

    identity = cudnn.custom_reduction_identity(cudnn.data_type.FLOAT, 0.0)
    calls = []
    child_errors = []

    def self_nesting_reducer(child, lhs, rhs):
        calls.append(1)
        # The depth limit keeps a build without the guard finite, so that it fails the test and does not overflow the
        # stack.
        if len(calls) > 4:
            return _add_reducer(child, lhs, rhs)
        outputs = child.custom_reduction([lhs[0]], self_nesting_reducer, "test.python.self_nesting.v1", [0], [identity], name="self_nested")
        try:
            child.build_operation_graph()
        except Exception as error:
            child_errors.append(error)
        return outputs

    graph = _float_graph()
    _float_streams(graph, 1, reducer=self_nesting_reducer)
    graph.validate()
    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        graph._lowered_graph.build_operation_graph()
    assert len(calls) == 1
    assert len(child_errors) == 1
    assert "may contain only pointwise operations" in str(child_errors[0])


@requires_cudnn_v9_29
def test_custom_reduction_refuses_a_nested_reducer_when_another_thread_builds_the_child_graph(cudnn_handle):
    if cudnn_handle is None:
        pytest.skip("cuDNN backend is unavailable")

    identity = cudnn.custom_reduction_identity(cudnn.data_type.FLOAT, 0.0)
    inner_calls = []
    child_errors = []

    def inner_reducer(child, lhs, rhs):
        inner_calls.append(1)
        # No result: a build that runs this reducer stops before any backend call.
        return []

    def outer_reducer(child, lhs, rhs):
        outputs = child.custom_reduction([lhs[0]], inner_reducer, "test.python.nested_worker.v1", [0], [identity], name="nested")

        def build_child():
            try:
                child.build_operation_graph()
            except Exception as error:
                child_errors.append(error)

        # join() releases the GIL, so the worker builds while this reducer waits.
        worker = threading.Thread(target=build_child)
        worker.start()
        worker.join()
        return outputs

    graph = _float_graph()
    _float_streams(graph, 1, reducer=outer_reducer)
    graph.validate()
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="may contain only pointwise operations"):
        graph._lowered_graph.build_operation_graph()
    assert inner_calls == []
    assert len(child_errors) == 1
    assert isinstance(child_errors[0], cudnn.cudnnGraphNotSupportedError)
    assert "may contain only pointwise operations" in str(child_errors[0])


@requires_cudnn_v9_29
@pytest.mark.parametrize("on_worker", [False, True])
def test_custom_reduction_refuses_an_sdpa_score_mod_with_a_custom_reduction_in_a_reducer_body(cudnn_handle, on_worker):
    if cudnn_handle is None:
        pytest.skip("cuDNN backend is unavailable")

    identity = cudnn.custom_reduction_identity(cudnn.data_type.FLOAT, 0.0)
    score_mod_calls = []
    inner_calls = []
    child_errors = []

    def inner_reducer(child, lhs, rhs):
        inner_calls.append(1)
        # No result: a build that runs this reducer stops before any backend call.
        return []

    def score_mod(graph, scores):
        score_mod_calls.append(1)
        graph.custom_reduction([scores], inner_reducer, "test.python.score_mod_inner.v1", [3], [identity], name="in_score_mod")
        return scores

    def outer_reducer(child, lhs, rhs):
        dims = [1, 1, 64, 64]
        strides = [4096, 4096, 64, 1]
        q, k, v = (child.tensor(name=name, dim=dims, stride=strides, data_type=cudnn.data_type.HALF) for name in "qkv")
        o, _ = child.sdpa(
            q=q,
            k=k,
            v=v,
            attn_scale=0.125,
            generate_stats=False,
            compute_data_type=cudnn.data_type.FLOAT,
            score_mod=score_mod,
            implementation=cudnn.attention_implementation.COMPOSITE,
            name="sdpa_in_body",
        )
        o.set_output(True)

        def build_child():
            try:
                child.build_operation_graph()
            except Exception as error:
                child_errors.append(error)

        if on_worker:
            worker = threading.Thread(target=build_child)
            worker.start()
            worker.join()
        else:
            build_child()
        return _add_reducer(child, lhs, rhs)

    graph = _float_graph()
    _float_streams(graph, 1, reducer=outer_reducer)
    graph.validate()
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="may contain only pointwise operations"):
        graph._lowered_graph.build_operation_graph()
    assert inner_calls == []
    assert score_mod_calls == []
    assert len(child_errors) == 1
    assert isinstance(child_errors[0], cudnn.cudnnGraphNotSupportedError)
    assert "may contain only pointwise operations" in str(child_errors[0])


@requires_cudnn_v9_29
def test_custom_reduction_reducer_may_build_an_unrelated_custom_reduction_graph(cudnn_handle):
    if cudnn_handle is None:
        pytest.skip("cuDNN backend is unavailable")

    unrelated_calls = []
    unrelated_errors = []
    outer_calls = []

    def unrelated_reducer(child, lhs, rhs):
        unrelated_calls.append(1)
        return _add_reducer(child, lhs, rhs)

    unrelated = _float_graph()
    _float_streams(unrelated, 1, reducer=unrelated_reducer)
    unrelated.validate()

    def outer_reducer(child, lhs, rhs):
        outer_calls.append(1)
        try:
            unrelated._lowered_graph.build_operation_graph()
        except Exception as error:
            unrelated_errors.append(error)
        return _add_reducer(child, lhs, rhs)

    graph = _float_graph()
    _float_streams(graph, 1, reducer=outer_reducer)
    graph.validate()
    graph._lowered_graph.build_operation_graph()
    assert outer_calls == [1]
    assert unrelated_errors == []
    assert unrelated_calls == [1]


@requires_cudnn_v9_29
def test_custom_reduction_reducer_may_return_a_captured_parent_tensor(cudnn_handle):
    if cudnn_handle is None:
        pytest.skip("cuDNN backend is unavailable")

    graph = _float_graph()
    floor = graph.tensor_scalar(-1000.0, cudnn.scalar_type.COMPILE_TIME_CONST)
    _float_streams(graph, 1, reducer=lambda child, lhs, rhs: [floor])
    graph.validate()
    graph._lowered_graph.build_operation_graph()


def test_custom_reduction_output_type_follows_a_later_input_type():
    graph = _float_graph()
    x = graph.tensor(dim=[2, 3], stride=[3, 1], name="X")
    y = graph.custom_reduction(
        [x],
        _add_reducer,
        "test.python.custom_reduction.late_input_type.v1",
        [1],
        [cudnn.custom_reduction_identity.from_bits(cudnn.data_type.HALF, 0)],
    )[0]
    y.set_output(True)
    # The builder copied the type of X (unset, so the graph io type). The caller sets the type of X after that.
    x.set_data_type(cudnn.data_type.HALF)
    graph.validate()
    assert y.data_type == cudnn.data_type.HALF
    assert graph._cpp_tensors[y.get_uid()].get_data_type() == cudnn.data_type.HALF


def test_custom_reduction_keeps_an_explicit_output_type_when_the_input_type_changes():
    graph = _float_graph()
    x = graph.tensor(dim=[2, 3], stride=[3, 1], data_type=cudnn.data_type.FLOAT, name="X")
    y = graph.custom_reduction(
        [x],
        _add_reducer,
        "test.python.custom_reduction.explicit_output_type.v1",
        [1],
        [cudnn.custom_reduction_identity.from_bits(cudnn.data_type.HALF, 0)],
    )[0]
    y.set_output(True)
    # The explicit type equals the type that the builder derived. It must stay when the input type changes later, so
    # that the native validation sees the mismatch.
    y.set_data_type(cudnn.data_type.FLOAT)
    x.set_data_type(cudnn.data_type.HALF)
    with pytest.raises(Exception, match="output and input data types must match"):
        graph.validate()
    assert y.data_type == cudnn.data_type.FLOAT


def test_byte_boolean_maps_to_torch_bool_and_boolean_keeps_it():
    torch = pytest.importorskip("torch")
    assert cudnn.datatypes._cudnn_to_torch_data_type(cudnn.data_type.BYTE_BOOLEAN) is torch.bool
    assert cudnn.datatypes._cudnn_to_torch_data_type(cudnn.data_type.BOOLEAN) is torch.bool
    assert cudnn.datatypes._torch_to_cudnn_data_type(torch.bool) == cudnn.data_type.BOOLEAN


def test_custom_reduction_wrapper_passes_a_namedtuple_argument_as_a_tuple():
    pytest.importorskip("torch")

    class _StopBeforeBuild(Exception):
        pass

    # The SDPA dropout argument is often a namedtuple; the wrapper must not rebuild it with its own constructor.
    Inputs = collections.namedtuple("Inputs", "first second")
    with pytest.raises(_StopBeforeBuild):
        with cudnn.Graph(
            io_data_type=cudnn.data_type.FLOAT,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        ) as graph:
            inputs = Inputs(*(graph.tensor(dim=[2, 3], stride=[3, 1], data_type=cudnn.data_type.FLOAT, name=f"X{i}") for i in range(2)))
            outputs = graph.custom_reduction(
                inputs,
                _add_reducer,
                "test.python.custom_reduction.wrapper_namedtuple.v1",
                [1],
                [cudnn.custom_reduction_identity(cudnn.data_type.FLOAT, 0.0)] * 2,
                name="wrapper_namedtuple",
            )
            assert len(outputs) == 2
            raise _StopBeforeBuild()


_ARG_RULES = ("torch", "xla", "ignore_nan")


def test_reduction_modifiers_arg_helpers_set_the_rule_key_and_exact_identities():
    dt = cudnn.data_type
    rule = cudnn.reduction_modifiers.ArgNanRule
    cases = [
        (cudnn.reduction_modifiers.argmax, dt.FLOAT, dt.INT32, rule.TORCH, "argmax.torch", ("FF800000", "7FFFFFFF")),
        (cudnn.reduction_modifiers.argmax, dt.BFLOAT16, dt.INT32, rule.XLA, "argmax.xla", ("FF80", "7FFFFFFF")),
        (cudnn.reduction_modifiers.argmax, dt.HALF, dt.INT32, rule.IGNORE_NAN, "argmax.ignore_nan", ("7E00", "7FFFFFFF")),
        (cudnn.reduction_modifiers.argmin, dt.HALF, dt.INT32, rule.TORCH, "argmin.torch", ("7C00", "7FFFFFFF")),
        (cudnn.reduction_modifiers.argmin, dt.FLOAT, dt.INT32, rule.XLA, "argmin.xla", ("7F800000", "7FFFFFFF")),
        (cudnn.reduction_modifiers.argmin, dt.BFLOAT16, dt.INT32, rule.IGNORE_NAN, "argmin.ignore_nan", ("7FC0", "7FFFFFFF")),
        (cudnn.reduction_modifiers.argmax, dt.DOUBLE, dt.INT64, rule.TORCH, "argmax.torch", ("FFF0000000000000", "7FFFFFFFFFFFFFFF")),
        (cudnn.reduction_modifiers.argmax, dt.FLOAT, dt.INT64, rule.IGNORE_NAN, "argmax.ignore_nan", ("7FC00000", "7FFFFFFFFFFFFFFF")),
    ]
    for helper, value_type, index_type, nan_rule, key, raw in cases:
        arguments = helper(value_type, index_type, nan_rule)
        assert set(arguments) == {"reducer", "reducer_semantic_key", "identity_values"}
        assert callable(arguments["reducer"])
        assert arguments["reducer_semantic_key"] == f"cudnn_frontend.reduction_modifiers.{key}.v1"
        assert [identity.data_type for identity in arguments["identity_values"]] == [value_type, index_type]
        assert tuple(identity.raw_hex for identity in arguments["identity_values"]) == raw

    defaulted = cudnn.reduction_modifiers.argmax(dt.FLOAT)
    assert defaulted["reducer_semantic_key"] == "cudnn_frontend.reduction_modifiers.argmax.torch.v1"
    assert defaulted["identity_values"][1].data_type == dt.INT32


def test_reduction_modifiers_rule_takes_the_enum_or_its_value():
    for helper in (cudnn.reduction_modifiers.argmax, cudnn.reduction_modifiers.argmin):
        for name in _ARG_RULES:
            by_value = helper(cudnn.data_type.FLOAT, rule=name)
            by_enum = helper(cudnn.data_type.FLOAT, rule=cudnn.reduction_modifiers.ArgNanRule(name))
            assert by_value["reducer_semantic_key"] == by_enum["reducer_semantic_key"]
            assert by_value["reducer_semantic_key"].endswith(f".{name}.v1")
        with pytest.raises(ValueError, match="rule must be"):
            helper(cudnn.data_type.FLOAT, rule="first_nan")
    keys = {
        getattr(cudnn.reduction_modifiers, kind)(cudnn.data_type.FLOAT, rule=name)["reducer_semantic_key"]
        for kind in ("argmax", "argmin")
        for name in _ARG_RULES
    }
    assert len(keys) == 6


class _BodyValue:
    def __init__(self, name):
        self.name = name
        self.data_type = None

    def set_data_type(self, data_type):
        self.data_type = data_type
        return self

    def get_data_type(self):
        return self.data_type


class _RecordingBody:
    """Records the operations that a reducer adds, in order, for a host model of the body."""

    def __init__(self):
        self.operations = []

    def __getattr__(self, method):
        def operation(*operands, compute_data_type, name):
            result = _BodyValue(str(len(self.operations)))
            self.operations.append((method, operands, compute_data_type, result))
            return result

        return operation


# The body operations as the reducer body evaluates them: ordered compares are false on NaN, CMP_NEQ is unordered.
_BODY_SEMANTICS = {
    "cmp_gt": lambda a, b: a > b,
    "cmp_ge": lambda a, b: a >= b,
    "cmp_lt": lambda a, b: a < b,
    "cmp_le": lambda a, b: a <= b,
    "cmp_eq": lambda a, b: a == b,
    "cmp_neq": lambda a, b: not (a == b),
    "logical_and": lambda a, b: a and b,
    "logical_or": lambda a, b: a or b,
    "logical_not": lambda a: not a,
    "binary_select": lambda x, b, t: x if t else b,
}


def _record_arg_body(arguments, value_type=cudnn.data_type.FLOAT):
    body = _RecordingBody()
    lhs = [_BodyValue("vl").set_data_type(value_type), _BodyValue("il").set_data_type(cudnn.data_type.INT32)]
    rhs = [_BodyValue("vr").set_data_type(value_type), _BodyValue("ir").set_data_type(cudnn.data_type.INT32)]
    results = arguments["reducer"](body, lhs, rhs)
    return body.operations, lhs, rhs, results


def _identity_float(identity):
    import struct

    return struct.unpack(">f", bytes.fromhex(identity.raw_hex))[0]


def _model_combine(operations, lhs, rhs, results):
    def combine(left, right):
        env = {id(lhs[0]): left[0], id(lhs[1]): left[1], id(rhs[0]): right[0], id(rhs[1]): right[1]}
        for method, operands, _, result in operations:
            env[id(result)] = _BODY_SEMANTICS[method](*(env[id(operand)] for operand in operands))
        return env[id(results[0])], env[id(results[1])]

    return combine


def _host_arg_index(values, is_max, rule):
    import math

    nan_positions = [i for i, v in enumerate(values) if math.isnan(v)]
    numbers = [v for v in values if not math.isnan(v)]
    if rule in ("torch", "xla") and nan_positions:
        return set(nan_positions) if rule == "xla" else {nan_positions[0]}
    if not numbers:
        return {0}
    best = max(numbers) if is_max else min(numbers)
    return {next(i for i, v in enumerate(values) if v == best)}


def _fold_mismatches(combine, identity, is_max, rule, reference_rule):
    import itertools

    specials = [float("nan"), float("inf"), float("-inf"), 3.0, 1.0, 0.0, -0.0, 1e-40]
    mismatches = folds = 0
    for length in (1, 2, 3):
        for values in itertools.product(specials, repeat=length):
            accepted = _host_arg_index(values, is_max, reference_rule)
            for order in itertools.permutations(range(length)):
                for sides in itertools.product((False, True), repeat=length):
                    accumulator = identity
                    for position, element_left in zip(order, sides):
                        element = (values[position], position)
                        accumulator = combine(element, accumulator) if element_left else combine(accumulator, element)
                    folds += 1
                    mismatches += accumulator[1] not in accepted
    return mismatches, folds


def _expected_arg_operations(is_max, rule):
    at_least, at_most = ("cmp_ge", "cmp_le") if is_max else ("cmp_le", "cmp_ge")
    strictly = "cmp_gt" if is_max else "cmp_lt"
    if rule == "xla":
        return [
            (strictly, ("vl", "vr")),
            ("cmp_neq", ("vl", "vl")),
            ("logical_or", ("0", "1")),
            ("binary_select", ("vl", "vr", "2")),
            ("cmp_eq", ("vl", "vr")),
            ("cmp_lt", ("il", "ir")),
            ("logical_and", ("4", "5")),
            ("logical_or", ("2", "6")),
            ("binary_select", ("il", "ir", "7")),
        ], ("3", "8")
    if rule == "torch":
        middle = [
            ("cmp_neq", ("vl", "vl")),
            (at_least, ("vl", "vr")),
            ("logical_or", ("1", "2")),
            (at_most, ("vl", "vr")),
            ("logical_not", ("4",)),
            ("cmp_eq", ("vr", "vr")),
            ("logical_and", ("5", "6")),
        ]
    else:
        middle = [
            (at_least, ("vl", "vr")),
            ("cmp_neq", ("vr", "vr")),
            ("logical_or", ("1", "2")),
            (strictly, ("vl", "vr")),
            ("cmp_eq", ("vl", "vl")),
            ("logical_and", ("2", "5")),
            ("logical_or", ("4", "6")),
        ]
    return [
        ("cmp_lt", ("il", "ir")),
        *middle,
        ("binary_select", ("3", "7", "0")),
        ("binary_select", ("il", "ir", "8")),
        ("binary_select", ("vl", "vr", "8")),
    ], ("10", "9")


def test_reduction_modifiers_arg_bodies_hold_their_rule_on_a_host_model():
    # Checks the operations and port order of each body, then folds every row of 1 to 3 special values from the
    # helper's identity in every element order and operand side, with the reducer body's compare semantics.
    for is_max in (True, False):
        helper = cudnn.reduction_modifiers.argmax if is_max else cudnn.reduction_modifiers.argmin
        for rule in _ARG_RULES:
            arguments = helper(cudnn.data_type.FLOAT, rule=rule)
            operations, lhs, rhs, results = _record_arg_body(arguments)
            names = {id(value): value.name for value in (*lhs, *rhs)}
            names.update({id(result): result.name for *_, result in operations})
            recorded = [(method, tuple(names[id(operand)] for operand in operands)) for method, operands, _, _ in operations]
            expected, expected_results = _expected_arg_operations(is_max, rule)
            assert recorded == expected, (is_max, rule)
            assert tuple(names[id(result)] for result in results) == expected_results
            assert [result.data_type for result in results] == [cudnn.data_type.FLOAT, cudnn.data_type.INT32]
            # A compare or select of values computes in the value type; every other operation in the index type.
            for (method, recorded_operands), (_, _, compute_type, _) in zip(recorded, operations):
                expected_type = cudnn.data_type.FLOAT if recorded_operands[0] in ("vl", "vr") else cudnn.data_type.INT32
                assert compute_type == expected_type, (is_max, rule, method, recorded_operands)

            value_identity, index_identity = arguments["identity_values"]
            identity = (_identity_float(value_identity), int(index_identity.raw_hex, 16))
            combine = _model_combine(operations, lhs, rhs, results)
            mismatches, folds = _fold_mismatches(combine, identity, is_max, rule, rule)
            assert folds == 25104
            assert mismatches == 0, (is_max, rule)
            # The identity is two-sided: it leaves every element on either side, whatever its index.
            for element in itertools.product((float("nan"), float("inf"), float("-inf"), 1.0, 0.0, -0.0), (0, 1, 7)):
                assert combine(identity, element)[1] == element[1], (is_max, rule, element)
                assert combine(element, identity)[1] == element[1], (is_max, rule, element)

    # Discrimination: the XLA body does not meet the torch rule (two NaN values), and IGNORE_NAN with the
    # (-inf, largest index) identity of the other rules gives the largest index, not 0, for a row of NaN only.
    xla = cudnn.reduction_modifiers.argmax(cudnn.data_type.FLOAT, rule="xla")
    operations, lhs, rhs, results = _record_arg_body(xla)
    mismatches, _ = _fold_mismatches(_model_combine(operations, lhs, rhs, results), (float("-inf"), 0), True, "xla", "torch")
    assert mismatches > 0
    ignore_nan = cudnn.reduction_modifiers.argmax(cudnn.data_type.FLOAT, rule="ignore_nan")
    operations, lhs, rhs, results = _record_arg_body(ignore_nan)
    combine = _model_combine(operations, lhs, rhs, results)
    assert combine((float("-inf"), 2**31 - 1), (float("nan"), 0))[1] == 2**31 - 1


def _require_custom_reduction_gpu(cudnn_handle):
    if cudnn_handle is None:
        pytest.skip("cuDNN backend is unavailable")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Custom reduction requires the TensorIR MemBound engine on data-center Blackwell")
    return torch


_TORCH_TO_CUDNN = {
    "float32": cudnn.data_type.FLOAT,
    "bfloat16": cudnn.data_type.BFLOAT16,
    "float16": cudnn.data_type.HALF,
    "float64": cudnn.data_type.DOUBLE,
}


def _arg_graph(cudnn_handle, torch, x, index_type, kind, rule, output="index"):
    value_type = _TORCH_TO_CUDNN[str(x.dtype).split(".")[-1]]
    graph = cudnn.pygraph(
        handle=cudnn_handle,
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    x_t = graph.tensor(dim=list(x.shape), stride=list(x.stride()), data_type=value_type, name="X")
    idx_t = graph.tensor(dim=list(x.shape), stride=list(x.stride()), data_type=index_type, name="IDX")
    value, index = graph.custom_reduction(
        [x_t, idx_t], reduction_dims=[1], name=f"{kind}_{rule}", **getattr(cudnn.reduction_modifiers, kind)(value_type, index_type, rule)
    )
    (index if output == "index" else value).set_output(True)
    return graph, x_t, idx_t, value, index


def _run_arg(cudnn_handle, torch, x, kind, rule):
    graph, x_t, idx_t, _, index = _arg_graph(cudnn_handle, torch, x, cudnn.data_type.INT32, kind, rule)
    graph.build([cudnn.heur_mode.A])
    rows, extent = x.shape
    iota = torch.arange(extent, dtype=torch.int32, device="cuda").expand(rows, extent).contiguous()
    out = torch.full((rows, 1), -7, dtype=torch.int32, device="cuda")
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    torch.cuda.synchronize()
    graph.execute({x_t: x, idx_t: iota, index: out}, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()
    return out.view(-1).long()


def _arg_check(torch, x, got, kind, rule):
    """Number of rows whose index breaks the rule: torch.argmax / argmin for TORCH, any NaN position of a row with NaN
    for XLA, the first best number (the first position for a row of NaN only) for IGNORE_NAN."""
    is_max = kind == "argmax"
    rows = torch.arange(x.shape[0], device=x.device)
    in_range = (got >= 0) & (got < x.shape[1])
    if not bool(in_range.all()):
        return int((~in_range).sum())
    if rule == "torch":
        want = torch.argmax(x, dim=1) if is_max else torch.argmin(x, dim=1)
        return int((got != want).sum())
    xf = x.float()
    nan = torch.isnan(xf)
    numbers = xf.masked_fill(nan, float("-inf") if is_max else float("inf"))
    best = numbers.amax(dim=1, keepdim=True) if is_max else numbers.amin(dim=1, keepdim=True)
    first_best = torch.argmax(((numbers == best) & ~nan).to(torch.uint8), dim=1)
    has_number = (~nan).any(dim=1)
    if rule == "xla":
        has_nan = nan.any(dim=1)
        ok = torch.where(has_nan, nan[rows, got], got == first_best)
    else:
        ok = got == torch.where(has_number, first_best, torch.zeros_like(first_best))
    return int((~ok).sum())


def _arg_parity_rows(torch, dtype, extent):
    """Rows of the NaN, tie, signed zero, infinity and subnormal classes, then random rows."""
    nan, inf = float("nan"), float("inf")
    tiny = torch.finfo(dtype).tiny if dtype != torch.float32 else 1e-40
    rows = []

    def row(**positions):
        r = torch.randn(extent, dtype=torch.float32) * 3
        for position, value in positions.items():
            r[int(position[1:])] = value
        return r

    last = extent - 1
    rows += [row(), row(p3=50.0, p9=50.0), row(p5=nan), row(p0=nan, **{f"p{last}": nan}), row(p4=nan, p5=nan)]
    rows += [row(p7=inf, p2=nan), row(p2=inf, p7=nan), row(p1=inf, p6=inf), row(p1=-inf, p6=-inf)]
    zeros = -torch.rand(extent) - 1
    for first, second in ((0.0, -0.0), (-0.0, 0.0)):
        r = zeros.clone()
        r[3], r[8] = first, second
        rows.append(r)
    rows.append(torch.full((extent,), -inf))
    rows.append(torch.full((extent,), inf))
    all_nan = torch.full((extent,), nan)
    all_nan[1::2] = -nan
    rows.append(all_nan)
    neginf_nan = torch.full((extent,), -inf)
    neginf_nan[0] = nan
    rows.append(neginf_nan)
    sub = torch.full((extent,), -inf)
    sub[4], sub[9] = tiny / 2, tiny / 4
    rows.append(sub)
    rows.append(-sub)
    x = torch.stack(rows).to(dtype)
    # Random 16-bit words as bf16: about 1 in 256 is NaN, most rows hold one or more.
    words = torch.randint(0, 1 << 16, (64, extent), dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    return torch.cat([x, words.to(dtype)]).cuda()


@requires_cudnn_v9_29
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16", "float16"])
def test_reduction_modifiers_arg_rules_match_their_reference(cudnn_handle, dtype_name):
    torch = _require_custom_reduction_gpu(cudnn_handle)
    torch.manual_seed(7)
    dtype = getattr(torch, dtype_name)
    x = _arg_parity_rows(torch, dtype, 257)
    for kind in ("argmax", "argmin"):
        for rule in _ARG_RULES:
            got = _run_arg(cudnn_handle, torch, x, kind, rule)
            assert _arg_check(torch, x, got, kind, rule) == 0, (dtype_name, kind, rule)


@requires_cudnn_v9_29
def test_reduction_modifiers_argmax_matches_torch_on_every_bfloat16_pattern(cudnn_handle):
    torch = _require_custom_reduction_gpu(cudnn_handle)
    words = torch.arange(1 << 16, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    x = torch.full((1 << 16, 16), float("-inf"), dtype=torch.bfloat16)
    x[:, 3] = words
    x[:, 11] = words
    x = x.cuda()
    for kind in ("argmax", "argmin"):
        background = float("-inf") if kind == "argmax" else float("inf")
        x[:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]] = background
        for rule in _ARG_RULES:
            got = _run_arg(cudnn_handle, torch, x, kind, rule)
            assert _arg_check(torch, x, got, kind, rule) == 0, (kind, rule)


@requires_cudnn_v9_29
def test_reduction_modifiers_forward_the_backend_decision_on_other_stream_types(cudnn_handle):
    torch = _require_custom_reduction_gpu(cudnn_handle)
    # The frontend builds the body for any stream type, and the backend decides: the build either runs and gives the
    # torch index, or forwards the backend's not-supported status.
    for value_dtype, index_type, index_dtype in (
        (torch.float32, cudnn.data_type.INT64, torch.int64),
        (torch.float64, cudnn.data_type.INT32, torch.int32),
    ):
        x = torch.randn(4, 64, dtype=value_dtype, device="cuda")
        graph, x_t, idx_t, _, index = _arg_graph(cudnn_handle, torch, x, index_type, "argmax", "torch")
        try:
            graph.build([cudnn.heur_mode.A])
        except cudnn.cudnnGraphNotSupportedError as error:
            assert "CUDNN_STATUS_NOT_SUPPORTED" in str(error)
            continue
        iota = torch.arange(64, dtype=index_dtype, device="cuda").expand(4, 64).contiguous()
        out = torch.full((4, 1), -1, dtype=index_dtype, device="cuda")
        workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
        torch.cuda.synchronize()
        graph.execute({x_t: x, idx_t: iota, index: out}, workspace, handle=cudnn_handle)
        torch.cuda.synchronize()
        assert torch.equal(out.view(-1).long(), torch.argmax(x, dim=1))

    # The value and the index both as graph outputs: the backend decides.
    x = torch.randn(4, 64, dtype=torch.float32, device="cuda")
    graph, x_t, idx_t, value, index = _arg_graph(cudnn_handle, torch, x, cudnn.data_type.INT32, "argmax", "torch")
    value.set_output(True)
    graph.validate()
    try:
        graph.build([cudnn.heur_mode.A])
    except cudnn.cudnnGraphNotSupportedError as error:
        assert "CUDNN_STATUS_NOT_SUPPORTED" in str(error)
        return
    iota = torch.arange(64, dtype=torch.int32, device="cuda").expand(4, 64).contiguous()
    out_value = torch.full((4, 1), float("nan"), dtype=torch.float32, device="cuda")
    out_index = torch.full((4, 1), -1, dtype=torch.int32, device="cuda")
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    torch.cuda.synchronize()
    graph.execute({x_t: x, idx_t: iota, value: out_value, index: out_index}, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()
    assert torch.equal(out_index.view(-1).long(), torch.argmax(x, dim=1))
    assert torch.equal(out_value.view(-1), torch.amax(x, dim=1))


@requires_cudnn_v9_29
def test_reduction_modifiers_argmax_after_a_pointwise_producer(cudnn_handle):
    # The producer examples of docs/operations/CustomReduction.md: Gumbel-max with a [B, 1] temperature, a masked
    # argmax, and the recovered token max(p_t - p_d, 0) * inv_q.
    torch = _require_custom_reduction_gpu(cudnn_handle)
    torch.manual_seed(11)
    B, V = 8, 4099
    x_cpu = (torch.randn(B, V) * 3).to(torch.bfloat16)
    t_cpu = torch.rand(B, 1) + 0.5
    g_cpu = -torch.log(-torch.log(torch.rand(B, V).clamp(1e-20, 1 - 1e-7)))
    mask_cpu = torch.rand(B, V) < 0.3
    p_t_cpu, p_d_cpu = torch.softmax(torch.randn(B, V), dim=1), torch.softmax(torch.randn(B, V), dim=1)
    inv_q_cpu = 1.0 / (torch.rand(B, V) + 0.1)
    idx_cpu = torch.arange(V, dtype=torch.int32).expand(B, V).contiguous()

    def run(producer, expected):
        graph = cudnn.pygraph(
            handle=cudnn_handle,
            io_data_type=cudnn.data_type.FLOAT,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        feeds = {}

        def tensor(name, value, data_type):
            t = graph.tensor(dim=list(value.shape), stride=list(value.stride()), data_type=data_type, name=name)
            feeds[t] = value.cuda()
            return t

        y = producer(graph, tensor)
        idx = tensor("IDX", idx_cpu, cudnn.data_type.INT32)
        _, index = graph.custom_reduction([y, idx], reduction_dims=[1], **cudnn.reduction_modifiers.argmax(cudnn.data_type.FLOAT))
        index.set_output(True)
        graph.build([cudnn.heur_mode.A])
        feeds[index] = torch.full((B, 1), -1, dtype=torch.int32, device="cuda")
        workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
        torch.cuda.synchronize()
        graph.execute(feeds, workspace, handle=cudnn_handle)
        torch.cuda.synchronize()
        assert torch.equal(feeds[index].view(-1).cpu().long(), expected)

    def gumbel(graph, tensor):
        x, t, g = tensor("X", x_cpu, cudnn.data_type.BFLOAT16), tensor("T", t_cpu, cudnn.data_type.FLOAT), tensor("G", g_cpu, cudnn.data_type.FLOAT)
        return graph.add(graph.div(x, t), g)

    def masked(graph, tensor):
        x, mask = tensor("X", x_cpu, cudnn.data_type.BFLOAT16), tensor("MASK", mask_cpu, cudnn.data_type.BYTE_BOOLEAN)
        return graph.binary_select(x, graph.tensor_scalar(float("-inf"), cudnn.scalar_type.COMPILE_TIME_CONST), mask)

    def recovered(graph, tensor):
        p_t, p_d = tensor("P_T", p_t_cpu, cudnn.data_type.FLOAT), tensor("P_D", p_d_cpu, cudnn.data_type.FLOAT)
        inv_q = tensor("INV_Q", inv_q_cpu, cudnn.data_type.FLOAT)
        return graph.mul(graph.max(graph.sub(p_t, p_d), graph.tensor_scalar(0.0, cudnn.scalar_type.COMPILE_TIME_CONST)), inv_q)

    run(gumbel, torch.argmax(x_cpu.float() / t_cpu + g_cpu, dim=1))
    run(masked, torch.argmax(torch.where(mask_cpu, x_cpu.float(), torch.tensor(float("-inf"))), dim=1))
    run(recovered, torch.argmax((p_t_cpu - p_d_cpu).clamp(min=0.0) * inv_q_cpu, dim=1))
