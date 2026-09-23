# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public aligned-HCA support, execution-contract and autograd tests."""

from dataclasses import replace
import gc
from unittest.mock import patch

import pytest
import torch

pytestmark = pytest.mark.gpu_exclusive


@pytest.fixture(autouse=True)
def require_gb300():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("Aligned HCA requires GB300")
    pytest.importorskip("cutlass")
    pytest.importorskip("triton")


def declarations():
    from cudnn.api_base import TensorDesc

    def desc(shape, stride, dtype):
        return TensorDesc(
            dtype=dtype, shape=shape, stride=stride, stride_order=tuple(reversed(range(len(shape)))), device=torch.device("cuda", torch.cuda.current_device())
        )

    q = desc((4096, 128, 512), (65536, 512, 1), torch.bfloat16)
    kv = desc((4752, 512), (512, 1), torch.bfloat16)
    lse = desc((4096, 128), (128, 1), torch.float32)
    sink = desc((128,), (1,), torch.float32)
    return [q, kv, q, q, lse, sink, q, kv, sink]


def constant_tensors():
    return [torch.full(desc.shape, 0.01 * (index + 1), device=desc.device, dtype=desc.dtype) for index, desc in enumerate(declarations())]


@pytest.mark.L0
def test_public_exports_and_metadata_only_support():
    import cudnn
    from cudnn import DSA, AlignedHCABackward, aligned_hca_backward_wrapper
    from cudnn.deepseek_sparse_attention import AlignedHCABackward as family_class
    from cudnn.api_base import APIBase

    assert DSA.AlignedHCABackward is family_class is AlignedHCABackward
    assert DSA.aligned_hca_backward_wrapper is aligned_hca_backward_wrapper
    assert issubclass(cudnn.AlignedHCABackward, APIBase)
    before = torch.cuda.memory_allocated()
    for rank in range(16):
        api = AlignedHCABackward(*declarations(), cp_rank=rank)
        assert api.check_support()
        with pytest.raises(RuntimeError, match="Compile"):
            api.scratch_workspace_bytes()
    assert torch.cuda.memory_allocated() == before


@pytest.mark.L0
def test_workspace_has_no_paired_input_buffer():
    from cudnn.deepseek_sparse_attention.aligned_hca_backward._kernels import reduce_local_keys, reduce_local_rows
    from cudnn.deepseek_sparse_attention.aligned_hca_backward._plan import _HcaPlan
    from cudnn.deepseek_sparse_attention.aligned_hca_backward._workspace import workspace_layout

    for rank in range(16):
        layout = workspace_layout(rank)
        assert {buffer.name for buffer in layout.buffers} == {"packed_keys", "normalization", "sink_partial", "p", "ds", "dk", "dv"}
        assert layout.group_tokens == (128 if rank in (3, 5, 7, 9, 11, 12, 13, 14, 15) else 32)
        assert layout.groups * layout.group_tokens == 4096
        assert layout.keys == ((128 + layout.group_tokens + (rank + 1) * 32 + 63) // 64) * 64
        assert layout.nbytes % 256 == 0
        for previous, current in zip(layout.buffers, layout.buffers[1:]):
            assert current.offset % 256 == 0
            assert current.offset >= previous.offset + previous.nbytes
        assert layout.buffers[-1].offset + layout.buffers[-1].nbytes <= layout.nbytes
        schedule = _HcaPlan(rank, 512**-0.5, "cuda")._schedule()
        assert len(schedule) == 6
        assert schedule[0][-1] == {"num_warps": 4}
        assert schedule[2][-1] == {"num_warps": 8, "num_stages": 4 if rank in (0, 3, 4, 9, 11, 13, 14, 15) else 3}
        assert schedule[3][0] is (reduce_local_rows if layout.group_tokens == 32 else reduce_local_keys)
        assert schedule[3][1] == ((1056 if layout.group_tokens == 32 else 16896), 1, 1)
        assert schedule[3][2] == ("dk", "dv", "dkv")
        assert schedule[3][3] == (layout.keys, layout.groups, 512, layout.group_tokens, *((2048,) if layout.group_tokens == 32 else (128,)))


@pytest.mark.L0
@pytest.mark.parametrize(
    "index,field,value,error",
    [
        (1, "shape", (4736, 512), NotImplementedError),
        (0, "stride", (65537, 512, 1), NotImplementedError),
        (4, "dtype", torch.bfloat16, ValueError),
        (8, "shape", (64,), NotImplementedError),
        (1, "device", torch.device("cpu"), ValueError),
    ],
)
def test_unsupported_declarations(index, field, value, error):
    from cudnn import AlignedHCABackward

    tensors = declarations()
    tensors[index] = replace(tensors[index], **{field: value})
    with pytest.raises(error):
        AlignedHCABackward(*tensors, cp_rank=0).check_support()


@pytest.mark.L0
@pytest.mark.parametrize("rank,scale", [(-1, 0.1), (16, 0.1), (True, 0.1), (0, float("nan")), (0, True)])
def test_invalid_configuration(rank, scale):
    from cudnn import AlignedHCABackward

    with pytest.raises(ValueError):
        AlignedHCABackward(*declarations(), cp_rank=rank, softmax_scale=scale).check_support()


@pytest.mark.L1
@pytest.mark.parametrize("rank", [0, 3, 15])
def test_execute_contract(rank):
    from cuda.bindings import driver as cuda
    from triton.runtime.jit import JITFunction
    from cudnn import AlignedHCABackward

    before_compile = torch.cuda.memory_allocated()
    api = AlignedHCABackward(*declarations(), cp_rank=rank)
    api.compile()
    assert torch.cuda.memory_allocated() == before_compile
    tensors = constant_tensors()
    workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    default, alternate = torch.cuda.current_stream(), torch.cuda.Stream()
    with patch.object(JITFunction, "run", side_effect=AssertionError("execute dispatched Triton JIT")):
        expected = {name: tensor.clone() for name, tensor in api.execute(*tensors, workspace).items()}
        alternate.wait_stream(default)
        api.execute(*tensors, workspace, current_stream=alternate)
        alternate.synchronize()
        before_execute = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        api.execute(*tensors, workspace, current_stream=cuda.CUstream(alternate.cuda_stream))
        alternate.synchronize()
        assert torch.cuda.max_memory_allocated() == before_execute

        fresh = constant_tensors()
        fresh_workspace = torch.empty_like(workspace)
        alternate.wait_stream(default)
        with torch.cuda.stream(alternate):
            torch.cuda._sleep(2_000_000)
            for tensor in fresh[6:]:
                tensor.fill_(float("nan"))
        actual = api.execute(*fresh, fresh_workspace, current_stream=alternate.cuda_stream)
        default.wait_stream(alternate)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            api.execute(*fresh, fresh_workspace)
        graph.replay()
        torch.cuda.synchronize()
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


@pytest.mark.L1
def test_execute_rejects_invalid_arguments():
    from cudnn import AlignedHCABackward

    tensors = constant_tensors()
    api = AlignedHCABackward(*declarations(), cp_rank=0)
    with pytest.raises(RuntimeError, match="Compile"):
        api.execute(*tensors, torch.empty(0, device="cuda", dtype=torch.uint8))
    api.compile()
    workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    for index, value in [(1, tensors[1][:4736]), (6, tensors[0]), (7, tensors[1]), (8, tensors[5]), (4, tensors[4][:, :64])]:
        changed = list(tensors)
        changed[index] = value
        with pytest.raises(ValueError):
            api.execute(*changed, workspace)
    for invalid in (workspace[:-256], workspace[1:], workspace.view(torch.float32)):
        with pytest.raises(ValueError):
            api.execute(*tensors, invalid)
    for invalid in (1, 2, -1):
        with pytest.raises(ValueError):
            api.execute(*tensors, workspace, current_stream=invalid)
    for invalid in (True, torch.tensor(0, device="cuda")):
        with pytest.raises(TypeError):
            api.execute(*tensors, workspace, current_stream=invalid)
    actual = api.execute(*tensors, workspace, current_stream=0)
    assert list(actual.keys()) == ["dq", "dkv", "d_sink"]


@pytest.mark.L1
@pytest.mark.parametrize("rank", [0, 3, 15])
def test_numerical_reference_and_wrapper_graph_replay(rank):
    from cudnn import aligned_hca_backward_wrapper
    from cudnn.deepseek_sparse_attention.aligned_hca_backward import api as api_module
    from triton.runtime.jit import JITFunction
    from fe_api.dsa.aligned_hca_test_utils import reference_case, assert_reference

    inputs, expected = reference_case(rank, 751 + rank)
    actual = aligned_hca_backward_wrapper(*inputs, cp_rank=rank)
    assert_reference(actual, expected)
    del actual
    torch.cuda.synchronize()
    graphs = []
    with (
        patch.object(JITFunction, "run", side_effect=AssertionError("warm wrapper dispatched JIT")),
        patch.object(api_module._HcaPlan, "compile", side_effect=AssertionError("warm wrapper recompiled")),
    ):
        for _ in range(2):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = aligned_hca_backward_wrapper(*inputs, cp_rank=rank)
            graphs.append((graph, outputs))
        gc.collect()
        torch.cuda.empty_cache()
        for seed in (1771, 8811):
            fresh, expected = reference_case(rank, seed + rank)
            for destination, source in zip(inputs, fresh):
                destination.copy_(source)
            del fresh
            for graph, outputs in reversed(graphs):
                graph.replay()
                torch.cuda.synchronize()
                assert_reference(outputs, expected)


@pytest.mark.L1
def test_wrapper_capture_requires_warmup():
    from cudnn import aligned_hca_backward_wrapper
    from cudnn.deepseek_sparse_attention.aligned_hca_backward import api as api_module

    tensors = constant_tensors()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with patch.object(api_module, "_WRAPPER_APIS", {}):
        with torch.cuda.graph(graph):
            with pytest.raises(RuntimeError, match="Warm"):
                aligned_hca_backward_wrapper(*tensors[:6], cp_rank=0, dq=tensors[6], dkv=tensors[7], d_sink=tensors[8])
