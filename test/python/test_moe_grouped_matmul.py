"""
Test suite for MoE Grouped Matmul and MoE Grouped Matmul Bwd Python API.
Based on samples/cpp/moe_grouped_matmul/moe_grouped_matmul.cpp
"""

import cudnn
import pytest
import torch

from test_utils import torch_fork_set_rng


def get_cublaslt_version() -> int:
    """Return the cublasLt runtime version, or 0 if the library cannot be loaded."""
    import ctypes

    for libname in ["libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"]:
        try:
            return ctypes.CDLL(libname).cublasLtGetVersion()
        except OSError:
            continue
    return 0


@pytest.mark.skipif(
    cudnn.backend_version() < 91800,
    reason="moe_grouped_matmul requires cuDNN >= 9.18.0",
)
@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_bf16_moe_grouped_matmul_fwd(cudnn_handle):
    # problem size
    num_experts = 36
    token_num = 2000
    weight_size = 248
    hidden_size = 520

    first_token_offset_values = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        127,
        255,
        383,
        483,
        515,
        643,
        718,
        924,
        1100,
        1200,
        1300,
        1400,
        1500,
        1600,
        1700,
        1800,
        1900,
    ]

    graph = cudnn.pygraph(
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )

    # token: [1, T, H], BFLOAT16, row-major
    tensor_token = graph.tensor(
        name="token",
        dim=[1, token_num, hidden_size],
        stride=[token_num * hidden_size, hidden_size, 1],
        data_type=cudnn.data_type.BFLOAT16,
    )

    # weight: [E, H, N], BFLOAT16, column-major in H×N
    tensor_weight = graph.tensor(
        name="weight",
        dim=[num_experts, hidden_size, weight_size],
        stride=[hidden_size * weight_size, 1, hidden_size],
        data_type=cudnn.data_type.BFLOAT16,
    )

    # first_token_offset: [E, 1, 1], INT32
    tensor_first_token_offset = graph.tensor(
        name="first_token_offset",
        dim=[num_experts, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.INT32,
    )

    # moe_grouped_matmul: token × weight → output per expert
    tensor_output = graph.moe_grouped_matmul(
        tensor_token,
        tensor_weight,
        tensor_first_token_offset,
        mode=cudnn.moe_grouped_matmul_mode.NONE,
        compute_data_type=cudnn.data_type.FLOAT,
        name="moe_grouped_matmul",
    )
    # output shape [1, T, N] is inferred; row-major stride [T*N, N, 1]
    tensor_output.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    graph.check_support()
    graph.build_plans()

    # allocate device buffers
    token_data = torch.randn(token_num * hidden_size, dtype=torch.bfloat16, device="cuda")
    # weight: [E, H, N] column-major → total elements = E * H * N
    weight_data = torch.randn(num_experts * hidden_size * weight_size, dtype=torch.bfloat16, device="cuda")
    first_token_offset_data = torch.tensor(first_token_offset_values, dtype=torch.int32, device="cuda")
    output_data = torch.empty(token_num * weight_size, dtype=torch.bfloat16, device="cuda")

    workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda")

    graph.execute(
        {
            tensor_token: token_data,
            tensor_weight: weight_data,
            tensor_first_token_offset: first_token_offset_data,
            tensor_output: output_data,
        },
        workspace,
        handle=cudnn_handle,
    )


@pytest.mark.skipif(
    cudnn.backend_version() < 92200,
    reason="moe_grouped_matmul_bwd requires cuDNN >= 9.22.0",
)
@pytest.mark.skipif(
    get_cublaslt_version() < 130500,
    reason="moe_grouped_matmul_bwd requires cublasLt >= 13.5",
)
@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_bf16_moe_grouped_matmul_bwd(cudnn_handle):
    """
    BF16 MoE Grouped Matmul backward pass (dweight computation).
    Mirrors C++ TEST_CASE "BF16 MoeGroupedMatmulBwd".
    """
    # problem size
    num_experts = 36
    token_num = 2000
    weight_size = 248
    hidden_size = 520

    first_token_offset_values = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        127,
        255,
        383,
        483,
        515,
        643,
        718,
        924,
        1100,
        1200,
        1300,
        1400,
        1500,
        1600,
        1700,
        1800,
        1900,
    ]

    graph = cudnn.pygraph(
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )

    # doutput: [1, T, N], BFLOAT16, row-major
    tensor_doutput = graph.tensor(
        name="doutput",
        dim=[1, token_num, weight_size],
        stride=[token_num * weight_size, weight_size, 1],
        data_type=cudnn.data_type.BFLOAT16,
    )

    # token: [1, T, H], BFLOAT16, row-major
    tensor_token = graph.tensor(
        name="token",
        dim=[1, token_num, hidden_size],
        stride=[token_num * hidden_size, hidden_size, 1],
        data_type=cudnn.data_type.BFLOAT16,
    )

    # first_token_offset: [E, 1, 1], INT32
    tensor_first_token_offset = graph.tensor(
        name="first_token_offset",
        dim=[num_experts, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.INT32,
    )

    # moe_grouped_matmul_bwd: computes dweight = token^T × doutput per expert
    tensor_dweight = graph.moe_grouped_matmul_bwd(
        tensor_doutput,
        tensor_token,
        tensor_first_token_offset,
        compute_data_type=cudnn.data_type.FLOAT,
        name="moe_grouped_matmul_bwd",
    )
    # dweight shape [E, H, N] is inferred; column-major stride [H*N, 1, H]
    tensor_dweight.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    graph.check_support()
    graph.build_plans()

    # allocate device buffers
    doutput_data = torch.randn(token_num * weight_size, dtype=torch.bfloat16, device="cuda")
    token_data = torch.randn(token_num * hidden_size, dtype=torch.bfloat16, device="cuda")
    first_token_offset_data = torch.tensor(first_token_offset_values, dtype=torch.int32, device="cuda")
    # dweight: [E, H, N] column-major → total elements = E * H * N
    dweight_data = torch.empty(num_experts * hidden_size * weight_size, dtype=torch.bfloat16, device="cuda")

    workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda")

    graph.execute(
        {
            tensor_doutput: doutput_data,
            tensor_token: token_data,
            tensor_first_token_offset: first_token_offset_data,
            tensor_dweight: dweight_data,
        },
        workspace,
        handle=cudnn_handle,
    )
