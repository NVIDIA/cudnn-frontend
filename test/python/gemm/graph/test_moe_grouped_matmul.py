# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


def get_compute_capability() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


# ---------------------------------------------------------------------------
# Numeric oracle + layout helpers
#
# Layouts (from the graph tensor definitions below):
#   token   [1, T, H] row-major          -> token[t, h]   = data[t*H + h]
#   weight  [E, H, N] stride [H*N, 1, H]  -> weight[e,h,n] = data[e*H*N + h + n*H]
#                                             i.e. expert block is column-major [H,N]
#   output  [1, T, N] row-major          -> output[t, n]  = data[t*N + n]
# Expert e owns token rows [offset[e], offset[e+1]) with offset[E] := T.
# This turns the previously execute-only harness into a checked one, so silent
# wrong-result / grouped-offset / empty-expert defects (NVBug 6192149-class,
# 5921085 scatter OOB) are actually caught.
# ---------------------------------------------------------------------------


def _expert_weight_HN(weight_data, e, H, N):
    """Reconstruct expert e's [H, N] weight matrix from the column-major flat block."""
    block = weight_data[e * H * N : (e + 1) * H * N]
    return block.view(N, H).t().float()  # data[h + n*H] -> M[h, n]


def moe_fwd_reference(token_data, weight_data, offsets, E, T, H, N):
    tok = token_data.view(T, H).float()
    out = torch.zeros(T, N, dtype=torch.float32, device=token_data.device)
    bounds = list(offsets) + [T]
    for e in range(E):
        lo, hi = bounds[e], bounds[e + 1]
        if hi > lo:
            out[lo:hi] = tok[lo:hi] @ _expert_weight_HN(weight_data, e, H, N)
    return out  # [T, N]


def moe_bwd_reference(doutput_data, token_data, offsets, E, T, H, N):
    """dweight[e] = token[e-rows]^T @ doutput[e-rows], returned as [E, H, N] (column-major flat)."""
    tok = token_data.view(T, H).float()
    do = doutput_data.view(T, N).float()
    bounds = list(offsets) + [T]
    dw = torch.zeros(E, H, N, dtype=torch.float32, device=token_data.device)
    for e in range(E):
        lo, hi = bounds[e], bounds[e + 1]
        if hi > lo:
            dw[e] = tok[lo:hi].t() @ do[lo:hi]  # [H,N]
    return dw  # [E, H, N]


def _moe_tol(contract_dim):
    # bf16 IO, fp32 accumulate; scale with sqrt(contraction length) like the matmul fuzzer.
    import math

    s = max(1.0, math.sqrt(contract_dim / 128.0))
    return 2e-2 * s, 2e-2 * s


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
    torch.cuda.synchronize()

    # Numeric oracle (was previously execute-only).
    ref = moe_fwd_reference(token_data, weight_data, first_token_offset_values, num_experts, token_num, hidden_size, weight_size)
    rtol, atol = _moe_tol(hidden_size)
    torch.testing.assert_close(output_data.view(token_num, weight_size).float(), ref, rtol=rtol, atol=atol)


@pytest.mark.skipif(
    cudnn.backend_version() < 92200,
    reason="moe_grouped_matmul_bwd requires cuDNN >= 9.22.0",
)
@pytest.mark.skipif(
    get_cublaslt_version() < 130500,
    reason="moe_grouped_matmul_bwd requires cublasLt >= 13.5",
)
@pytest.mark.skipif(
    get_compute_capability() < 90 or get_compute_capability() >= 120,
    reason="moe_grouped_matmul_bwd requires SM90 - SM119 architectures",
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
    torch.cuda.synchronize()

    # Numeric oracle (was previously execute-only). dweight is [E,H,N] column-major flat:
    # dweight[e,h,n] = data[e*H*N + h + n*H]  ==  data.view(E,N,H)[e].t()
    ref = moe_bwd_reference(doutput_data, token_data, first_token_offset_values, num_experts, token_num, hidden_size, weight_size)
    dw_actual = dweight_data.view(num_experts, weight_size, hidden_size).transpose(1, 2).float()
    rtol, atol = _moe_tol(token_num)  # contraction is over tokens
    torch.testing.assert_close(dw_actual, ref, rtol=rtol, atol=atol)


def _rand_offsets(E, T, rng):
    """Non-decreasing first-token offsets, offset[0]=0; duplicates => empty experts."""
    starts = sorted(rng.randint(0, T) for _ in range(E))
    starts[0] = 0
    return starts


@pytest.mark.skipif(
    cudnn.backend_version() < 91800,
    reason="moe_grouped_matmul requires cuDNN >= 9.18.0",
)
@pytest.mark.L0
@pytest.mark.parametrize("seed", list(range(16)))
def test_bf16_moe_grouped_matmul_fwd_randomized(cudnn_handle, seed):
    """Randomized experts/tokens/offsets (incl. empty experts) + numeric oracle.

    The original harness used one fixed shape and never checked the result. This
    exercises grouped-offset / empty-expert / token-boundary handling against a
    PyTorch per-expert reference. Caught class: 6192149 (grouped MoE numerics),
    5921085 (scatter OOB on uneven offsets).
    """
    import random as _random

    rng = _random.Random(seed)

    num_experts = rng.choice([2, 4, 8, 17, 36, 64])
    token_num = rng.choice([16, 64, 200, 555, 2000])
    hidden_size = rng.choice([64, 128, 256, 520])
    weight_size = rng.choice([64, 128, 248, 256])
    # Force at least one empty expert in ~half the configs.
    offsets = _rand_offsets(num_experts, token_num, rng)
    if seed % 2 == 0 and num_experts >= 2:
        offsets[1] = 0  # expert 0 empty

    torch.manual_seed(seed)

    graph = cudnn.pygraph(
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )
    tensor_token = graph.tensor(
        name="token",
        dim=[1, token_num, hidden_size],
        stride=[token_num * hidden_size, hidden_size, 1],
        data_type=cudnn.data_type.BFLOAT16,
    )
    tensor_weight = graph.tensor(
        name="weight",
        dim=[num_experts, hidden_size, weight_size],
        stride=[hidden_size * weight_size, 1, hidden_size],
        data_type=cudnn.data_type.BFLOAT16,
    )
    tensor_first_token_offset = graph.tensor(
        name="first_token_offset",
        dim=[num_experts, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.INT32,
    )
    tensor_output = graph.moe_grouped_matmul(
        tensor_token,
        tensor_weight,
        tensor_first_token_offset,
        mode=cudnn.moe_grouped_matmul_mode.NONE,
        compute_data_type=cudnn.data_type.FLOAT,
        name="moe_grouped_matmul",
    )
    tensor_output.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    try:
        graph.check_support()
    except Exception as e:
        pytest.skip(f"unsupported config: {e}")
    graph.build_plans()

    token_data = torch.randn(token_num * hidden_size, dtype=torch.bfloat16, device="cuda")
    weight_data = torch.randn(num_experts * hidden_size * weight_size, dtype=torch.bfloat16, device="cuda")
    first_token_offset_data = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    output_data = torch.empty(token_num * weight_size, dtype=torch.bfloat16, device="cuda")
    workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda")

    graph.execute(
        {tensor_token: token_data, tensor_weight: weight_data, tensor_first_token_offset: first_token_offset_data, tensor_output: output_data},
        workspace,
        handle=cudnn_handle,
    )
    torch.cuda.synchronize()

    ref = moe_fwd_reference(token_data, weight_data, offsets, num_experts, token_num, hidden_size, weight_size)
    rtol, atol = _moe_tol(hidden_size)
    torch.testing.assert_close(output_data.view(token_num, weight_size).float(), ref, rtol=rtol, atol=atol)
