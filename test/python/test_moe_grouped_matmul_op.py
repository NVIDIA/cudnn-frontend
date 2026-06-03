"""
Correctness tests for MoE Grouped Matmul.

Uses Mixtral/DeepSeek-inspired shapes.
"""

import pytest
import torch
import cudnn


def get_cc():
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def ref_moe_grouped_matmul(token, weight, first_token_offset, num_experts):
    """Reference: loop over experts, matmul with column-major weight."""
    _, M, K = token.shape
    _, _, N = weight.shape
    output = torch.zeros(1, M, N, dtype=token.dtype, device=token.device)
    for b_e in range(num_experts):
        start = first_token_offset[b_e, 0, 0].item()
        end = first_token_offset[b_e + 1, 0, 0].item() if b_e + 1 < num_experts else M
        if start < end:
            output[0, start:end] = (token[0, start:end].float() @ weight[b_e].float()).to(token.dtype)
    return output


class TestMoEGroupedMatmul:

    @pytest.mark.L0
    @pytest.mark.skipif(cudnn.backend_version() < 91800, reason="MoE requires cuDNN >= 9.18.0")
    @pytest.mark.parametrize(
        "num_experts,tokens,K,N",
        [
            (4, 512, 256, 256),  # small
            (8, 1024, 512, 512),  # Mixtral-like
            (4, 256, 256, 256),  # minimal
        ],
        ids=["4exp_512tok", "8exp_1024tok", "4exp_256tok"],
    )
    @pytest.mark.parametrize("dtype", [torch.float16], ids=["fp16"])
    def test_moe_forward(self, cudnn_handle, num_experts, tokens, K, N, dtype):
        torch.manual_seed(42)

        tpe = tokens // num_experts
        actual_tokens = tpe * num_experts

        token = torch.randn(1, actual_tokens, K, dtype=dtype, device="cuda")
        weight_raw = torch.randn(num_experts, N, K, dtype=dtype, device="cuda")
        weight = weight_raw.transpose(1, 2)  # col-major inner
        fto_vals = torch.arange(num_experts, dtype=torch.int32, device="cuda") * tpe
        fto = fto_vals.reshape(-1, 1, 1)

        stream = torch.cuda.current_stream().cuda_stream
        cudnn.set_stream(handle=cudnn_handle, stream=stream)

        from cudnn.experimental.ops import moe_grouped_matmul

        result = moe_grouped_matmul(token, weight, fto, mode="none", top_k=1)

        Y_ref = ref_moe_grouped_matmul(token, weight, fto, num_experts)

        torch.testing.assert_close(Y_ref, result, atol=0.125, rtol=0.125)
