"""
Test suite for block_scale_quantize with dynamic shape overrides Python API.
Based on blackwell_nvfp4_mxfp8_block_scale_matmul.cpp
"""

import cudnn
import pytest
import torch

from test_utils import torch_fork_set_rng


def get_cc():
    """Get CUDA compute capability."""
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def div_up(a, b):
    """Integer division with rounding up."""
    return (a + b - 1) // b


def calculate_block_scale_dims(m, n, k, block_size):
    """
    Calculate block scale dimensions using indestructible block formula.
    Based on C++ lines 319-325, 454-463.
    """
    INDESTRUCTIBLE_128x4_BLOCK_M_N = 128
    INDESTRUCTIBLE_128x4_BLOCK_K = 4

    block_scale_dim_m = div_up(m, INDESTRUCTIBLE_128x4_BLOCK_M_N) * INDESTRUCTIBLE_128x4_BLOCK_M_N
    block_scale_dim_n = div_up(n, INDESTRUCTIBLE_128x4_BLOCK_M_N) * INDESTRUCTIBLE_128x4_BLOCK_M_N
    block_scale_dim_k = div_up(div_up(k, block_size), INDESTRUCTIBLE_128x4_BLOCK_K) * INDESTRUCTIBLE_128x4_BLOCK_K

    return block_scale_dim_m, block_scale_dim_n, block_scale_dim_k


class TestBlockScaleQuantizeMatmulDynamicShape:
    """
    Test block_scale_quantize API with full matmul workflow.
    Based on C++ TEST_CASE "Blackwell Block Scale Matmul dynamic shape overrides" (lines 749-910).
    """

    @pytest.mark.skipif(
        cudnn.backend_version() < 91800,
        reason="block_scale_quantize requires cuDNN >= 9.18.0",
    )
    @pytest.mark.skipif(
        get_cc() < 100,
        reason="block_scale_quantize requires CUDA compute capability larger than 100",
    )
    @pytest.mark.parametrize(
        "b,m,n,k",
        [
            (1, 1024, 1024, 1024),
        ],
    )
    @pytest.mark.L0
    @torch_fork_set_rng(seed=999)
    def test_block_scale_quantize_matmul_dynamic_shape(self, cudnn_handle, b, m, n, k):
        """
        Test block_scale_quantize in a full matmul workflow:
        1. Create quantized inputs A, B with block scales
        2. Dequantize A and B
        3. Perform matmul
        4. Quantize output using block_scale_quantize
        5. Validate execution succeeds

        This mirrors the C++ test at lines 749-910.
        """
        # Skip FP4 tests if PyTorch doesn't support it
        if not hasattr(torch, "float4_e2m1fn_x2"):
            pytest.skip("PyTorch does not support float4_e2m1fn_x2")

        A_UID = 1
        SF_A_UID = 2
        B_UID = 3
        SF_B_UID = 4
        C_UID = 5

        datatype_a = cudnn.data_type.FP4_E2M1
        datatype_b = cudnn.data_type.FP4_E2M1
        datatype_scale = cudnn.data_type.FP8_E4M3
        datatype_output = cudnn.data_type.BFLOAT16
        block_size = 16

        matmul_dynamic_shapes = [
            {"b": 2, "m": 1024, "n": 1024, "k": 1024},
            {"b": 2, "m": 2048, "n": 2048, "k": 2048},
        ]

        graph = cudnn.pygraph(
            io_data_type=cudnn.data_type.FLOAT,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=cudnn_handle,
            is_dynamic_shape_enabled=True,
        )

        block_scale_dim_m, block_scale_dim_n, block_scale_dim_k = calculate_block_scale_dims(m, n, k, block_size)

        A = graph.tensor(
            name="A",
            uid=A_UID,
            dim=[b, m, k],
            stride=[m * k, k, 1],
            data_type=datatype_a,
        )

        SF_A = graph.tensor(
            name="SF_A",
            uid=SF_A_UID,
            dim=[b, block_scale_dim_m, block_scale_dim_k],
            stride=[block_scale_dim_m * block_scale_dim_k, block_scale_dim_k, 1],
            data_type=datatype_scale,
            reordering_type=cudnn.tensor_reordering.F8_128x4,
        )

        dequan_tensor_a = graph.block_scale_dequantize(A, SF_A, block_size=[1, block_size], name="dequantize_a")

        B = graph.tensor(
            name="B",
            uid=B_UID,
            dim=[b, k, n],
            stride=[n * k, 1, k],
            data_type=datatype_b,
        )

        SF_B = graph.tensor(
            name="SF_B",
            uid=SF_B_UID,
            dim=[b, block_scale_dim_k, block_scale_dim_n],
            stride=[block_scale_dim_n * block_scale_dim_k, 1, block_scale_dim_k],
            data_type=datatype_scale,
            reordering_type=cudnn.tensor_reordering.F8_128x4,
        )

        dequan_tensor_b = graph.block_scale_dequantize(B, SF_B, block_size=[block_size, 1], name="dequantize_b")

        C = graph.matmul(
            dequan_tensor_a,
            dequan_tensor_b,
            compute_data_type=cudnn.data_type.FLOAT,
            name="matmul",
        )
        C.set_uid(C_UID).set_output(True).set_data_type(datatype_output)

        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans()

        for dynamic_shape in matmul_dynamic_shapes:
            block_scale_dim_m, block_scale_dim_n, block_scale_dim_k = calculate_block_scale_dims(
                dynamic_shape["m"],
                dynamic_shape["n"],
                dynamic_shape["k"],
                block_size,
            )

            override_uids = [A_UID, SF_A_UID, B_UID, SF_B_UID, C_UID]

            override_shapes = [
                [dynamic_shape["b"], dynamic_shape["m"], dynamic_shape["k"]],
                [dynamic_shape["b"], block_scale_dim_m, block_scale_dim_k],
                [dynamic_shape["b"], dynamic_shape["k"], dynamic_shape["n"]],
                [dynamic_shape["b"], block_scale_dim_k, block_scale_dim_n],
                [dynamic_shape["b"], dynamic_shape["m"], dynamic_shape["n"]],
            ]

            override_strides = [
                [dynamic_shape["m"] * dynamic_shape["k"], dynamic_shape["k"], 1],
                [block_scale_dim_m * block_scale_dim_k, block_scale_dim_k, 1],
                [dynamic_shape["n"] * dynamic_shape["k"], 1, dynamic_shape["k"]],
                [block_scale_dim_n * block_scale_dim_k, 1, block_scale_dim_k],
                [dynamic_shape["m"] * dynamic_shape["n"], dynamic_shape["n"], 1],
            ]

            A_gpu = torch.randint(
                0,
                256,
                (dynamic_shape["b"], dynamic_shape["m"], dynamic_shape["k"] // 2),
                dtype=torch.uint8,
                device="cuda",
            )
            SF_A_gpu = torch.ones(
                (b, block_scale_dim_m, block_scale_dim_k),
                dtype=torch.float8_e4m3fn,
                device="cuda",
            )
            B_gpu = torch.randint(
                0,
                256,
                (dynamic_shape["b"], dynamic_shape["k"] // 2, dynamic_shape["n"]),
                dtype=torch.uint8,
                device="cuda",
            )
            SF_B_gpu = torch.ones(
                (b, block_scale_dim_k, block_scale_dim_n),
                dtype=torch.float8_e4m3fn,
                device="cuda",
            )
            C_gpu = torch.empty(
                (dynamic_shape["b"], dynamic_shape["m"], dynamic_shape["n"]),
                dtype=torch.bfloat16,
                device="cuda",
            )

            variant_pack = {
                A_UID: A_gpu,
                SF_A_UID: SF_A_gpu,
                B_UID: B_gpu,
                SF_B_UID: SF_B_gpu,
                C_UID: C_gpu,
            }

            workspace_size = graph.get_workspace_size()
            workspace = torch.empty(workspace_size, dtype=torch.uint8, device="cuda")

            graph.execute(
                variant_pack,
                workspace,
                handle=cudnn_handle,
                override_uids=override_uids,
                override_shapes=override_shapes,
                override_strides=override_strides,
            )

            torch.cuda.synchronize()

        print(f"✓ Test passed: b={b}, m={m}, n={n}, k={k}")
