"""
Tests for zero-element (no-op) graph support, e.g. SDPA with batch size 0.

See https://github.com/NVIDIA/cudnn-frontend/issues/101: graphs whose output
tensors are all zero-element (a dimension of size 0) are treated as no-ops.
They validate and build successfully, report a workspace size of 0, and
execute() launches no work.
"""

import cudnn
import pytest
import torch

from test_utils import torch_fork_set_rng


def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return cudnn.data_type.HALF
    elif torch_type == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    elif torch_type == torch.float32:
        return cudnn.data_type.FLOAT
    else:
        raise ValueError("Unsupported tensor data type.")


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_sdpa_batch_size_zero(cudnn_handle):
    b, h, s_q, s_kv, d = 0, 10, 128, 16, 64

    dtype = torch.float16
    q_gpu = torch.zeros(b, h, s_q, d, device="cuda", dtype=dtype)
    k_gpu = torch.zeros(b, h, s_kv, d, device="cuda", dtype=dtype)
    v_gpu = torch.zeros(b, h, s_kv, d, device="cuda", dtype=dtype)
    o_gpu = torch.zeros(b, h, s_q, d, device="cuda", dtype=dtype)

    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=cudnn_handle, stream=stream)

    graph = cudnn.pygraph(
        handle=cudnn_handle,
        io_data_type=convert_to_cudnn_type(dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    q = graph.tensor_like(q_gpu)
    k = graph.tensor_like(k_gpu)
    v = graph.tensor_like(v_gpu)

    o, _ = graph.sdpa(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        generate_stats=False,
        attn_scale=1.0 / (d**0.5),
    )
    o.set_output(True).set_dim(o_gpu.size()).set_stride(o_gpu.stride())

    graph.validate()
    assert graph.is_zero_element_graph()

    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    graph.check_support()
    graph.build_plans()

    assert graph.get_workspace_size() == 0
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    variant_pack = {
        q: q_gpu,
        k: k_gpu,
        v: v_gpu,
        o: o_gpu,
    }
    graph.execute(variant_pack, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()

    assert o_gpu.shape == (b, h, s_q, d)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_pointwise_zero_element(cudnn_handle):
    dims = (0, 8, 16)

    dtype = torch.float32
    a_gpu = torch.zeros(*dims, device="cuda", dtype=dtype)
    b_gpu = torch.zeros(*dims, device="cuda", dtype=dtype)
    c_gpu = torch.zeros(*dims, device="cuda", dtype=dtype)

    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=cudnn_handle, stream=stream)

    graph = cudnn.pygraph(
        handle=cudnn_handle,
        io_data_type=convert_to_cudnn_type(dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    a = graph.tensor_like(a_gpu)
    b = graph.tensor_like(b_gpu)

    c = graph.add(a=a, b=b)
    c.set_output(True).set_dim(c_gpu.size()).set_stride(c_gpu.stride())

    graph.build([cudnn.heur_mode.A])
    assert graph.is_zero_element_graph()
    assert graph.get_workspace_size() == 0
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    variant_pack = {a: a_gpu, b: b_gpu, c: c_gpu}
    graph.execute(variant_pack, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_mixed_zero_element_graph_rejected(cudnn_handle):
    # Matmul with a contracted dimension of 0: inputs are zero-element but the
    # output is not. This would require zero-filling the output, which cuDNN
    # does not support; expect a clear validation error.
    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=cudnn_handle, stream=stream)

    graph = cudnn.pygraph(
        handle=cudnn_handle,
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    a = graph.tensor(name="A", dim=[1, 4, 0], stride=[1, 1, 1])
    b = graph.tensor(name="B", dim=[1, 0, 8], stride=[1, 1, 1])

    c = graph.matmul(name="matmul", A=a, B=b)
    c.set_output(True)

    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="zero-element"):
        graph.validate()
    assert not graph.is_zero_element_graph()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
