# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import cudnn
import pytest
import torch

pytestmark = pytest.mark.L0

# This is CUDA C++ source compiled into the producer, not a Python callback.
SOURCE = r"""
__device__ void decode(const FortWeightDecodeTileV1& t, const void* storage,
    const void* const* auxiliary, void*, void*, FortWeightDecodeValue* output) {
    const int bits=int(t.constants[0]), scaling=int(t.constants[1]);
    const long long block_k=t.constants[2], block_n=t.constants[3];
    const long long blocks_n=(t.full_n+block_n-1)/block_n;
    const auto* block_scales=static_cast<const float*>(auxiliary[0]);
    const auto* global_scale=static_cast<const float*>(auxiliary[1]);
    for (int i=t.thread_id; i<t.tile_k*t.tile_n; i+=t.thread_count) {
        const int row=i/t.tile_n, col=i%t.tile_n;
        if (row>=t.valid_k || col>=t.valid_n) continue;
        const long long k=t.k_begin+row, n=t.n_begin+col;
        const long long bit_index=(k*t.full_n+n)*bits;
        const unsigned code=(static_cast<const unsigned char*>(storage)[bit_index/8]
            >> (bit_index%8)) & ((1u<<bits)-1);
        // Give the code its signed numerical meaning, then dequantize it.
        const int value=int(code)-((code & (1u<<(bits-1))) ? (1<<bits) : 0);
        float weight=float(value);
        if (scaling & 2) weight *= block_scales[(k/block_k)*blocks_n+n/block_n];
        if (scaling & 1) weight *= global_scale[0];
        // Both scale operations precede the single rounding to the MMA type.
        output[row*t.output_stride+col]=fort_weight_decode_from_float(weight);
    }
}
"""


def make_graph(handle, bits=4, scaling=3, dtype=cudnn.data_type.HALF, **program):
    m, k, n, lda, block_k, block_n = 37, 83, 75, 88, 24, 20
    byte_count = (k * n * bits + 7) // 8
    scale_count = ((k + block_k - 1) // block_k) * ((n + block_n - 1) // block_n)
    graph = cudnn.pygraph(io_data_type=dtype, intermediate_data_type=dtype, compute_data_type=cudnn.data_type.FLOAT, handle=handle)
    a = graph.tensor(dim=[1, m, k], stride=[m * lda, lda, 1], name="A", uid=1)
    weights = graph.tensor(dim=[1, 1, byte_count], stride=[byte_count, byte_count, 1], data_type=cudnn.data_type.UINT8, name="weights", uid=2)
    block = graph.tensor(dim=[1, 1, scale_count], stride=[scale_count, scale_count, 1], data_type=cudnn.data_type.FLOAT, name="block", uid=5)
    global_ = graph.tensor(dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT, name="global", uid=6)
    kwargs = dict(source=SOURCE, entry="decode", constants=[bits, scaling, block_k, block_n])
    kwargs.update(program)
    b = graph.weight_dequantize(weights, auxiliaries=[block, global_], out_dims=[1, k, n], name="dequant", **kwargs)
    b.set_data_type(dtype).set_uid(3)
    c = graph.matmul(a, b, name="gemm")
    c.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_uid(4)
    return graph, (a, weights, block, global_, c)


def build_prototype(graph):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("Experimental backend currently supports SM120 only")
    try:
        graph.validate()
        graph.build_operation_graph()
    except cudnn.cudnnGraphNotSupportedError as error:
        # An older header/library is an expected skip. Never hide a broken
        # supported graph, descriptor finalization, compilation, or execution.
        if any(
            message in str(error)
            for message in (
                "CUDNN_WEIGHT_DECODE_ABI_VERSION",
                "Backend does not provide weight-decode program ABI 1",
                "Weight dequantization requires the prototype backend",
            )
        ):
            pytest.skip(str(error))
        raise
    graph.create_execution_plan(10, {})
    graph.check_support()
    graph.build_plans()
    assert graph.get_workspace_size() == 0


@pytest.mark.parametrize("bits", [8, 4, 2])
@pytest.mark.parametrize("scaling", [1, 2, 3], ids=["global", "block", "block_global"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_weight_dequantize_numerics(cudnn_handle, bits, scaling, dtype):
    graph, tensors = make_graph(cudnn_handle, bits, scaling, dtype)
    build_prototype(graph)
    m, k, n, lda, block_k, block_n = 37, 83, 75, 88, 24, 20
    blocks_n = (n + block_n - 1) // block_n
    scale_count = ((k + block_k - 1) // block_k) * blocks_n
    # Deterministic CPU inputs, independent of the GPU's SM count.
    i = torch.arange(k * n, dtype=torch.int64)
    original = ((i * 13 + i // 7) % (1 << bits) - (1 << (bits - 1))).reshape(k, n)
    codes = (original.flatten() & ((1 << bits) - 1)).tolist()
    packed = bytearray((k * n * bits + 7) // 8)
    for index, code in enumerate(codes):
        packed[index * bits // 8] |= code << ((index * bits) % 8)
    a_cpu = (((torch.arange(m)[:, None] * 3 + torch.arange(k)[None, :] * 7) % 13 - 6) / 16).to(dtype)
    # Use the handle fixture's stream for copies, execution and capture.
    stream = torch.cuda.ExternalStream(cudnn.get_stream(cudnn_handle))
    with torch.cuda.stream(stream):
        a_gpu = torch.zeros((1, m, lda), dtype=dtype, device="cuda")[:, :, :k]
        a_gpu.copy_(a_cpu)
        w_gpu = torch.tensor(list(packed), dtype=torch.uint8, device="cuda")
        block_gpu = torch.empty(scale_count, dtype=torch.float32, device="cuda")
        global_gpu = torch.empty(1, dtype=torch.float32, device="cuda")
        c_gpu = torch.empty((1, m, n), dtype=torch.float32, device="cuda")
        bindings = dict(zip(tensors, (a_gpu, w_gpu, block_gpu, global_gpu, c_gpu)))
        for iteration in range(3):
            scales = ((torch.arange(scale_count) % 7 + 1 + (2 if iteration == 2 else 0)) / 16).float()
            global_scale = 0.375 if iteration == 0 else 0.625
            block_gpu.copy_(scales)
            global_gpu.fill_(global_scale)
            graph.execute(bindings, None)
            expected_b = original.float()
            if scaling & 2:
                block_ids = (torch.arange(k)[:, None] // block_k) * blocks_n + torch.arange(n)[None, :] // block_n
                expected_b = expected_b * scales[block_ids]
            if scaling & 1:
                expected_b = expected_b * global_scale
            expected = a_cpu.double() @ expected_b.to(dtype).double()
            torch.testing.assert_close(c_gpu[0].cpu().double(), expected, atol=0.005, rtol=0.005)

        # One representative case checks public plan serialization and capture.
        if bits == 4 and scaling == 3 and dtype == torch.float16:
            saved = graph.serialize()
            restored = cudnn.pygraph(handle=cudnn_handle)
            restored.deserialize(cudnn_handle, saved)
            uid_bindings = {t.get_uid(): data for t, data in bindings.items()}
            restored.execute(uid_bindings, None)
            torch.testing.assert_close(c_gpu[0].cpu().double(), expected, atol=0.005, rtol=0.005)
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture, stream=stream):
                graph.execute(bindings, None)
            c_gpu.fill_(float("nan"))
            capture.replay()
            torch.testing.assert_close(c_gpu[0].cpu().double(), expected, atol=0.005, rtol=0.005)


@pytest.mark.parametrize(
    "program", [{"entry": "decode;"}, {"abi_version": 2}, {"source": ""}, {"source": "x\0y"}, {"input_alignment": 3}, {"stage_smem_bytes": -1}]
)
def test_weight_dequantize_invalid_program(cudnn_handle, program):
    graph, _ = make_graph(cudnn_handle, **program)
    with pytest.raises((RuntimeError, ValueError)):
        graph.validate()


def test_weight_dequantize_graph_identity(cudnn_handle):
    graph, _ = make_graph(cudnn_handle)
    graph.validate()
    description = json.loads(repr(graph))
    op = description["nodes"][0]
    assert op["tag"] == "WEIGHT_DEQUANTIZE"
    assert op["program"]["source"] == SOURCE
    assert [op["inputs"][f"AUX_{i}"] for i in range(2)] == [5, 6]
    changed, _ = make_graph(cudnn_handle, source=SOURCE + "\n// a different program")
    changed.validate()
    assert changed.key() != graph.key()
