# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw graph DLPack descriptors must describe storage before graph inference.

An FP8 buffer misdeclared as NOT_SET inherits the graph's default IO dtype.
Mixed E4M3/E5M2 saved O and dO then compute wrong backward gradients. Exercise
the actual C++ binding without depending on a framework's FP8 DLPack exporter.
"""

import ctypes
import math

import cudnn
import pytest

pytestmark = pytest.mark.L0


class _Device(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _Dtype(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _Tensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _Device),
        ("ndim", ctypes.c_int),
        ("dtype", _Dtype),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _ManagedTensor(ctypes.Structure):
    _fields_ = [("dl_tensor", _Tensor), ("manager_ctx", ctypes.c_void_p), ("deleter", ctypes.c_void_p)]


class _Buffer:
    """Own metadata/storage throughout tensor_like's synchronous inspection."""

    def __init__(self, code, bits=8, lanes=1, strides=None):
        self.storage = (ctypes.c_uint8 * 512)()
        self.shape = (ctypes.c_int64 * 3)(2, 3, 4)
        self.strides = None if strides is None else (ctypes.c_int64 * 3)(*strides)
        self.managed = _ManagedTensor(
            _Tensor(ctypes.addressof(self.storage), _Device(1, 0), 3, _Dtype(code, bits, lanes), self.shape, self.strides, 0), None, None
        )

    def __dlpack__(self):
        new_capsule = ctypes.pythonapi.PyCapsule_New
        new_capsule.restype = ctypes.py_object
        new_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        # The binding only inspects this capsule; it neither consumes nor retains it.
        return new_capsule(ctypes.addressof(self.managed), b"dltensor", None)


# DLPack codes are the public ABI in dlpack.h, independent of Torch/NumPy names.
_STORAGE_TYPES = [
    pytest.param(10, 8, cudnn.data_type.FP8_E4M3, id="e4m3fn"),
    pytest.param(12, 8, cudnn.data_type.FP8_E5M2, id="e5m2"),
    pytest.param(14, 8, cudnn.data_type.FP8_E8M0, id="e8m0fnu"),
    pytest.param(2, 16, cudnn.data_type.HALF, id="fp16"),
    pytest.param(4, 16, cudnn.data_type.BFLOAT16, id="bf16"),
    pytest.param(2, 32, cudnn.data_type.FLOAT, id="fp32"),
    pytest.param(2, 64, cudnn.data_type.DOUBLE, id="fp64"),
    pytest.param(0, 8, cudnn.data_type.INT8, id="int8"),
    pytest.param(0, 32, cudnn.data_type.INT32, id="int32"),
    pytest.param(0, 64, cudnn.data_type.INT64, id="int64"),
    pytest.param(1, 8, cudnn.data_type.UINT8, id="uint8"),
    pytest.param(6, 8, cudnn.data_type.BOOLEAN, id="bool"),
]


@pytest.mark.parametrize("code,bits,expected", _STORAGE_TYPES)
@pytest.mark.parametrize("strides", [None, (24, 1, 6)], ids=["compact", "strided"])
@pytest.mark.parametrize(
    "default", [cudnn.data_type.HALF, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2], ids=["default-half", "default-e4", "default-e5"]
)
def test_raw_tensor_like_preserves_storage_dtype(code, bits, expected, strides, default):
    # Explicit null handle: descriptor construction does not need a CUDA context.
    graph = cudnn._pybind_module.backend_graph(handle=0, io_data_type=default)
    buffer = _Buffer(code, bits, strides=strides)
    desc = graph.tensor_like(buffer)
    assert desc.get_data_type() == expected
    assert desc.get_dim() == [2, 3, 4]
    assert desc.get_stride() == list(strides or (12, 4, 1))
    assert desc.get_is_pass_by_value()


@pytest.mark.parametrize("code,expected", [(10, cudnn.data_type.FP8_E4M3), (12, cudnn.data_type.FP8_E5M2), (14, cudnn.data_type.FP8_E8M0)])
@pytest.mark.parametrize("bits,lanes", [(16, 1), (8, 2), (8, 0)], ids=["wrong-width", "vector-lanes", "zero-lanes"])
def test_raw_tensor_like_does_not_mislabel_non_scalar_fp8(code, expected, bits, lanes):
    graph = cudnn._pybind_module.backend_graph(handle=0)
    buffer = _Buffer(code, bits, lanes)
    # Declining unrepresentable types need not use a particular future error policy.
    assert graph.tensor_like(buffer).get_data_type() != expected


@pytest.mark.parametrize("code", [7, 8, 9, 11, 13], ids=["e3m4", "e4m3", "e4m3b11fnuz", "e4m3fnuz", "e5m2fnuz"])
def test_raw_tensor_like_does_not_alias_other_fp8_encodings(code):
    graph = cudnn._pybind_module.backend_graph(handle=0)
    buffer = _Buffer(code)
    assert graph.tensor_like(buffer).get_data_type() not in (cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2, cudnn.data_type.FP8_E8M0)


@pytest.mark.parametrize("io_name", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("mixed_o", [False, True], ids=["matched-O", "mixed-O"])
def test_raw_fp8_backward_uses_saved_o_storage_dtype(io_name, mixed_o):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (9, 10):
        pytest.skip("Native FP8 backward requires Hopper or Blackwell")
    if cudnn.backend_version() < 92700:
        pytest.skip("This native FP8 backward regression requires cuDNN 9.27+")
    io_type = getattr(torch, io_name)
    other_name = "float8_e5m2" if io_name == "float8_e4m3fn" else "float8_e4m3fn"
    o_type = getattr(torch, other_name) if mixed_o else io_type
    fe_type = cudnn.data_type.FP8_E4M3 if io_name == "float8_e4m3fn" else cudnn.data_type.FP8_E5M2
    sq, sk, d = 65, 128, 128
    handle = cudnn.create_handle()
    try:
        cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)
        graph = cudnn._pybind_module.backend_graph(
            handle=handle.backend_handle, io_data_type=fe_type, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT
        )
        bindings = {}

        def attr(buffer):
            desc = graph.tensor_like(buffer)
            bindings[desc] = buffer
            return desc

        def scalar(value):
            return attr(torch.full((1, 1, 1, 1), value, device="cuda", dtype=torch.float32))

        q, k, v, do = [torch.ones((1, 1, s, d), device="cuda", dtype=torch.float32).to(io_type) for s in (sq, sk, sk, sq)]
        # Uniform attention supplies independent, valid saved forward state.
        o = torch.ones(q.shape, device="cuda", dtype=torch.float32).to(o_type)
        stats = torch.full((1, 1, sq, 1), d * 0.125 + math.log(sk), device="cuda", dtype=torch.float32)
        outputs = graph.sdpa_fp8_backward(
            q=attr(q),
            k=attr(k),
            v=attr(v),
            dO=attr(do),
            o=attr(o),
            stats=attr(stats),
            attn_scale=0.125,
            descale_q=scalar(1),
            descale_k=scalar(1),
            descale_v=scalar(1),
            descale_o=scalar(1),
            descale_dO=scalar(1),
            descale_s=scalar(1 / 128),
            scale_s=scalar(128),
            descale_dP=scalar(1),
            scale_dP=scalar(1),
            scale_dQ=scalar(1),
            scale_dK=scalar(1),
            scale_dV=scalar(1),
            use_deterministic_algorithm=False,
        )
        buffers = [torch.empty_like(t) for t in (q, k, v)] + [torch.empty((1, 1, 1, 1), device="cuda") for _ in range(4)]
        assert len(outputs) == len(buffers)
        for desc, buffer in zip(outputs, buffers):
            desc.set_output(True).set_dim(buffer.shape).set_stride(buffer.stride()).set_data_type(
                cudnn.data_type.FLOAT if buffer.dtype == torch.float32 else fe_type
            )
            bindings[desc] = buffer
        graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        workspace = torch.empty(max(1, graph.get_workspace_size_plan_at_index(0)), device="cuda", dtype=torch.uint8)
        pack = {desc.get_uid(): buffer.data_ptr() for desc, buffer in bindings.items()}

        def execute():
            graph._execute_plan_at_index(pack, workspace.data_ptr(), 0, handle.backend_handle)

        def poison():
            for buffer in buffers[:3]:
                buffer.view(torch.uint8).fill_(0x7F)  # NaN in both FP8 encodings
            for buffer in buffers[3:]:
                buffer.fill_(math.nan)

        def check(multiplier):
            refs = [0, 0, multiplier * sq / sk, 0, 0, multiplier * sq / sk, 0]
            for buffer, value in zip(buffers, refs):
                expected = torch.full(buffer.shape, value, device="cuda", dtype=torch.float32).to(buffer.dtype).float()
                torch.testing.assert_close(buffer.float(), expected, rtol=0.02, atol=0.01)

        poison()
        execute()
        check(1)
        # Capture on the explicitly bound handle stream, then change dO and
        # poison outputs: an empty/stale replay must fail the independent oracle.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        cudnn.set_stream(handle, stream.cuda_stream)
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured, stream=stream):
            execute()
        torch.cuda.current_stream().wait_stream(stream)
        cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)
        do.copy_(torch.full(do.shape, 2, device="cuda", dtype=torch.float32))
        poison()
        captured.replay()
        check(2)
    finally:
        cudnn.destroy_handle(handle)
