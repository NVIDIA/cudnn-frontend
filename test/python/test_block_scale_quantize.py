"""
Test suite for block_scale_quantize Python API.
Based on blackwell_nvfp4_mxfp8_block_scale_matmul.cpp
"""

import cudnn
import pytest
import torch

from test_utils import torch_fork_set_rng


def get_cc():
    """Get CUDA compute capability."""
    (major, minor) = torch.cuda.get_device_capability()
    return major * 10 + minor


def div_up(a, b):
    """Integer division with rounding up."""
    return (a + b - 1) // b


def convert_to_cudnn_type(torch_type):
    """Convert PyTorch dtype to cuDNN data type."""
    if torch_type == torch.float16:
        return cudnn.data_type.HALF
    elif torch_type == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    elif torch_type == torch.float32:
        return cudnn.data_type.FLOAT
    elif torch_type == torch.int32:
        return cudnn.data_type.INT32
    elif torch_type == torch.int64:
        return cudnn.data_type.INT64
    elif torch_type == torch.float8_e4m3fn:
        return cudnn.data_type.FP8_E4M3
    elif torch_type == torch.float8_e5m2fn:
        return cudnn.data_type.FP8_E5M2
    elif torch_type == torch.float8_e8m0fnu:
        return cudnn.data_type.FP8_E8M0
    elif hasattr(torch, "float4_e2m1fn_x2") and torch_type == torch.float4_e2m1fn_x2:
        return cudnn.data_type.FP4_E2M1
    else:
        raise ValueError(f"Unsupported tensor data type: {torch_type}")


def calculate_block_scale_dims(m, n, k, block_size):
    """
    Calculate block scale dimensions using indestructible block formula.
    Based on C++ lines 319-325, 454-463.
    """
    INDESTRUCTIBLE_128x4_BLOCK_M_N = 128
    INDESTRUCTIBLE_128x4_BLOCK_K = 4

    block_scale_dim_m = (
        div_up(m, INDESTRUCTIBLE_128x4_BLOCK_M_N) * INDESTRUCTIBLE_128x4_BLOCK_M_N
    )
    block_scale_dim_n = (
        div_up(n, INDESTRUCTIBLE_128x4_BLOCK_M_N) * INDESTRUCTIBLE_128x4_BLOCK_M_N
    )
    block_scale_dim_k = (
        div_up(div_up(k, block_size), INDESTRUCTIBLE_128x4_BLOCK_K)
        * INDESTRUCTIBLE_128x4_BLOCK_K
    )

    # For output quantization (lines 461-463)
    block_scale_dim_out_m = block_scale_dim_m
    block_scale_dim_out_n = (
        div_up(div_up(n, block_size), INDESTRUCTIBLE_128x4_BLOCK_K)
        * INDESTRUCTIBLE_128x4_BLOCK_K
    )

    return (
        block_scale_dim_m,
        block_scale_dim_n,
        block_scale_dim_k,
        block_scale_dim_out_m,
        block_scale_dim_out_n,
    )


class TestBlockScaleQuantizeMatmul:
    """
    Test block_scale_quantize API with full matmul workflow.
    Based on C++ TEST_CASE "Blackwell Block Scale Matmul Quantize" (lines 417-566).
    """

    @pytest.mark.skipif(
        cudnn.backend_version() < 91400,
        reason="block_scale_quantize requires cuDNN >= 9.14.0",
    )
    @pytest.mark.skipif(
        get_cc() < 100 or get_cc() >= 110,
        reason="block_scale_quantize requires CUDA compute capability 100-109",
    )
    @pytest.mark.parametrize(
        "b,m,n,k,block_size,dtype_a,dtype_b,dtype_scale,dtype_output",
        [
            # FP4 tests - representative sample
            (1, 256, 256, 256, 16, "FP4_E2M1", "FP4_E2M1", "FP8_E4M3", "FP4_E2M1"),
            # FP8 E4M3 x E4M3 tests
            (1, 128, 128, 128, 32, "FP8_E4M3", "FP8_E4M3", "FP8_E8M0", "FP8_E4M3"),
            (1, 128, 128, 128, 32, "FP8_E4M3", "FP8_E4M3", "FP8_E8M0", "FP8_E5M2"),
            # FP8 E4M3 x E5M2 mixed tests
            (1, 128, 128, 128, 32, "FP8_E4M3", "FP8_E5M2", "FP8_E8M0", "FP8_E4M3"),
            (1, 128, 128, 128, 32, "FP8_E4M3", "FP8_E5M2", "FP8_E8M0", "FP8_E5M2"),
            # FP8 E5M2 x E4M3 mixed tests
            (1, 128, 128, 128, 32, "FP8_E5M2", "FP8_E4M3", "FP8_E8M0", "FP8_E4M3"),
            (1, 128, 128, 128, 32, "FP8_E5M2", "FP8_E4M3", "FP8_E8M0", "FP8_E5M2"),
        ],
    )
    @pytest.mark.L0
    @torch_fork_set_rng(seed=42)
    def test_block_scale_quantize_matmul(
        self,
        cudnn_handle,
        b,
        m,
        n,
        k,
        block_size,
        dtype_a,
        dtype_b,
        dtype_scale,
        dtype_output,
    ):
        """
        Test block_scale_quantize in a full matmul workflow:
        1. Create quantized inputs A, B with block scales
        2. Dequantize A and B
        3. Perform matmul
        4. Quantize output using block_scale_quantize
        5. Validate execution succeeds

        This mirrors the C++ test at lines 417-566.
        """
        # Skip FP4 tests if PyTorch doesn't support it
        if dtype_a == "FP4_E2M1" or dtype_b == "FP4_E2M1" or dtype_output == "FP4_E2M1":
            if not hasattr(torch, "float4_e2m1fn_x2"):
                pytest.skip("PyTorch does not support float4_e2m1fn_x2")

        # Map string dtype names to cudnn data types
        dtype_map = {
            "FP4_E2M1": cudnn.data_type.FP4_E2M1,
            "FP8_E4M3": cudnn.data_type.FP8_E4M3,
            "FP8_E5M2": cudnn.data_type.FP8_E5M2,
            "FP8_E8M0": cudnn.data_type.FP8_E8M0,
        }

        datatype_a = dtype_map[dtype_a]
        datatype_b = dtype_map[dtype_b]
        datatype_scale = dtype_map[dtype_scale]
        datatype_output = dtype_map[dtype_output]

        # Calculate block scale dimensions using indestructible block formula (lines 454-463)
        (
            block_scale_dim_m,
            block_scale_dim_n,
            block_scale_dim_k,
            block_scale_dim_out_m,
            block_scale_dim_out_n,
        ) = calculate_block_scale_dims(m, n, k, block_size)

        # Create graph (lines 484-487)
        g = cudnn.pygraph(
            io_data_type=cudnn.data_type.HALF,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=cudnn_handle,
        )

        # Input tensors A and B (lines 489-499)
        tensor_a = g.tensor(
            name="tensor_a",
            dim=[b, m, k],
            stride=[m * k, k, 1],
            data_type=datatype_a,
        )

        tensor_b = g.tensor(
            name="tensor_b",
            dim=[b, k, n],
            stride=[k * n, 1, k],
            data_type=datatype_b,
        )

        # Block scale tensors with F8_128x4 reordering (lines 501-513)
        block_descale_a = g.tensor(
            name="block_descale_a",
            dim=[b, block_scale_dim_m, block_scale_dim_k],
            stride=[block_scale_dim_m * block_scale_dim_k, block_scale_dim_k, 1],
            data_type=datatype_scale,
            reordering_type=cudnn.tensor_reordering.F8_128x4,
        )

        block_descale_b = g.tensor(
            name="block_descale_b",
            dim=[b, block_scale_dim_k, block_scale_dim_n],
            stride=[block_scale_dim_n * block_scale_dim_k, 1, block_scale_dim_k],
            data_type=datatype_scale,
            reordering_type=cudnn.tensor_reordering.F8_128x4,
        )

        # Dequantize A (lines 515-517)
        dequant_tensor_a = g.block_scale_dequantize(
            tensor_a, block_descale_a, block_size=[1, block_size], name="dequantize_a"
        )

        # Dequantize B (lines 519-521)
        dequant_tensor_b = g.block_scale_dequantize(
            tensor_b, block_descale_b, block_size=[block_size, 1], name="dequantize_b"
        )

        # Matmul (lines 523-526)
        tensor_c = g.matmul(
            dequant_tensor_a,
            dequant_tensor_b,
            compute_data_type=cudnn.data_type.FLOAT,
            name="matmul",
        )

        # ⭐ BLOCK SCALE QUANTIZE - THE KEY OPERATION (lines 528-531)
        tensor_d, block_scale = g.block_scale_quantize(
            tensor_c,
            block_size=block_size,
            axis=2,
            transpose=False,
            name="quantize_output",
        )

        # Set output properties (lines 533-536)
        tensor_d.set_output(True).set_data_type(datatype_output)
        block_scale.set_output(True).set_data_type(datatype_scale).set_reordering_type(
            cudnn.tensor_reordering.F8_128x4
        )

        # Build and validate graph (lines 540-551)
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        g.check_support()
        g.build_plans()

        # Allocate tensors
        # Note: For simplicity, we allocate based on dimensions. In production,
        # you'd calculate exact sizes using get_element_size_in_bits as in C++

        # Create dummy input data (in practice, these would be properly quantized)
        # Using uint8 as a generic container since we're just testing the graph execution
        if dtype_a == "FP4_E2M1":
            # FP4 is packed, so size is smaller
            tensor_a_data = torch.randint(
                0, 16, (b, m, k // 2), dtype=torch.uint8, device="cuda"
            )
        elif dtype_a == "FP8_E4M3":
            tensor_a_data = torch.randint(
                0, 256, (b, m, k), dtype=torch.uint8, device="cuda"
            )
        elif dtype_a == "FP8_E5M2":
            tensor_a_data = torch.randint(
                0, 256, (b, m, k), dtype=torch.uint8, device="cuda"
            )
        else:
            tensor_a_data = torch.randn((b, m, k), dtype=torch.float16, device="cuda")

        if dtype_b == "FP4_E2M1":
            tensor_b_data = torch.randint(
                0, 16, (b, k, n // 2), dtype=torch.uint8, device="cuda"
            )
        elif dtype_b == "FP8_E4M3":
            tensor_b_data = torch.randint(
                0, 256, (b, k, n), dtype=torch.uint8, device="cuda"
            )
        elif dtype_b == "FP8_E5M2":
            tensor_b_data = torch.randint(
                0, 256, (b, k, n), dtype=torch.uint8, device="cuda"
            )
        else:
            tensor_b_data = torch.randn((b, k, n), dtype=torch.float16, device="cuda")

        # Scale tensors
        if dtype_scale == "FP8_E4M3":
            scale_a_data = torch.ones(
                (b, block_scale_dim_m, block_scale_dim_k),
                dtype=torch.float8_e4m3fn,
                device="cuda",
            )
            scale_b_data = torch.ones(
                (b, block_scale_dim_k, block_scale_dim_n),
                dtype=torch.float8_e4m3fn,
                device="cuda",
            )
            scale_output_data = torch.empty(
                (b, block_scale_dim_out_m, block_scale_dim_out_n),
                dtype=torch.float8_e4m3fn,
                device="cuda",
            )
        else:  # FP8_E8M0
            scale_a_data = torch.ones(
                (b, block_scale_dim_m, block_scale_dim_k),
                dtype=torch.float8_e8m0fnu,
                device="cuda",
            )
            scale_b_data = torch.ones(
                (b, block_scale_dim_k, block_scale_dim_n),
                dtype=torch.float8_e8m0fnu,
                device="cuda",
            )
            scale_output_data = torch.empty(
                (b, block_scale_dim_out_m, block_scale_dim_out_n),
                dtype=torch.float8_e8m0fnu,
                device="cuda",
            )

        # Output tensor
        if dtype_output == "FP4_E2M1":
            tensor_d_data = torch.empty(
                (b, m, n // 2), dtype=torch.uint8, device="cuda"
            )
        elif dtype_output == "FP8_E4M3":
            tensor_d_data = torch.empty((b, m, n), dtype=torch.uint8, device="cuda")
        elif dtype_output == "FP8_E5M2":
            tensor_d_data = torch.empty((b, m, n), dtype=torch.uint8, device="cuda")
        else:
            tensor_d_data = torch.empty((b, m, n), dtype=torch.float16, device="cuda")

        # Get workspace
        workspace = torch.empty(
            g.get_workspace_size(), device="cuda", dtype=torch.uint8
        )

        # Execute (lines 557-565)
        variant_pack = {
            tensor_a: tensor_a_data,
            tensor_b: tensor_b_data,
            block_descale_a: scale_a_data,
            block_descale_b: scale_b_data,
            tensor_d: tensor_d_data,
            block_scale: scale_output_data,
        }

        g.execute(variant_pack, workspace, handle=cudnn_handle)

        # If we reach here without exceptions, the test passed
        assert tensor_d_data is not None
        assert scale_output_data is not None

        print(
            f"✓ Test passed: b={b}, m={m}, n={n}, k={k}, "
            f"dtype_a={dtype_a}, dtype_b={dtype_b}, "
            f"dtype_scale={dtype_scale}, dtype_output={dtype_output}"
        )
