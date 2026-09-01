# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observable contracts for the standalone grouped weight-only NVFP4 API."""

from __future__ import annotations

import pytest
import torch
from cuda.bindings import driver as cuda


def _imports():
    pytest.importorskip("cutlass.cute", reason="requires the nvidia-cutlass-dsl extra")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("PyTorch build has no float8_e4m3fn tensor dtype")
    from cudnn.gemm.cutedsl.grouped.weight_only_nvfp4 import GroupedGemmWeightOnlyNvfp4, grouped_gemm_weight_only_nvfp4

    return GroupedGemmWeightOnlyNvfp4, grouped_gemm_weight_only_nvfp4


def _empty_inputs(*, epilogue: str, experts: int = 2, rows: int = 5):
    if epilogue == "squared_relu":
        k, n = 2688, 1856
    elif epilogue == "linear":
        k, n = 1856, 2688
    else:
        raise ValueError(epilogue)
    return {
        "routed_tokens": torch.empty((1, rows, k), dtype=torch.bfloat16, device="cuda"),
        "packed_weight": torch.empty((experts, n, k // 2), dtype=torch.uint8, device="cuda"),
        "weight_scale": torch.empty((experts, n, k // 16), dtype=torch.float8_e4m3fn, device="cuda"),
        "first_token_offset": torch.empty((experts, 1, 1), dtype=torch.int32, device="cuda"),
        "factor": torch.empty((experts, 1, 1), dtype=torch.float32, device="cuda"),
        "output": torch.empty((1, rows, n), dtype=torch.bfloat16, device="cuda"),
    }


def _make_op(api, tensors, epilogue):
    return api(
        tensors["routed_tokens"],
        tensors["packed_weight"],
        tensors["weight_scale"],
        tensors["first_token_offset"],
        tensors["factor"],
        tensors["output"],
        epilogue=epilogue,
    )


@pytest.mark.L0
@pytest.mark.parametrize("epilogue", ("linear", "squared_relu"))
def test_check_support_accepts_public_tensor_contract(monkeypatch, epilogue):
    api, _ = _imports()
    tensors = _empty_inputs(epilogue=epilogue)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: (10, 0))

    op = _make_op(api, tensors, epilogue)

    assert op.check_support() is True


@pytest.mark.L0
def test_contract_rejects_unsupported_semantics_and_layout(monkeypatch):
    api, _ = _imports()
    tensors = _empty_inputs(epilogue="linear")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: (10, 0))

    with pytest.raises(NotImplementedError, match="supports only"):
        _make_op(api, tensors, "squared_relu")

    tensors["packed_weight"] = torch.empty(
        (2, 1856 // 2, 2688),
        dtype=torch.uint8,
        device="cuda",
    ).transpose(1, 2)
    op = _make_op(api, tensors, "linear")
    with pytest.raises(ValueError, match="packed_weight tensor stride mismatch"):
        op.check_support()


def _runtime_inputs(epilogue: str):
    tensors = _empty_inputs(epilogue=epilogue, rows=5)
    packed = tensors["packed_weight"]
    scale = tensors["weight_scale"]
    tokens = tensors["routed_tokens"]
    experts, n, packed_k = packed.shape
    k = packed_k * 2

    byte_index = torch.arange(packed.numel(), device="cuda", dtype=torch.int64).reshape(packed.shape)
    low = (byte_index.remainder(7) + 1).to(torch.uint8)
    high = ((3 * byte_index + 2).remainder(7) + 1).to(torch.uint8)
    packed.copy_(low | (high << 4))
    scale_values = torch.tensor((0.25, 0.5, 1.0, 2.0), dtype=torch.float8_e4m3fn, device="cuda")
    scale_index = torch.arange(scale.numel(), device="cuda", dtype=torch.int64).reshape(scale.shape)
    scale.copy_(scale_values[scale_index.remainder(4)])
    token_index = torch.arange(tokens.numel(), device="cuda", dtype=torch.float32).reshape(tokens.shape)
    tokens.copy_(((token_index.remainder(11) - 5) / 16).to(torch.bfloat16))
    tensors["first_token_offset"].copy_(torch.tensor((0, 3), dtype=torch.int32, device="cuda").view(experts, 1, 1))
    tensors["factor"].copy_(torch.tensor((2.0**-13, 2.0**-14), dtype=torch.float32, device="cuda").view(experts, 1, 1))

    e2m1 = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0), device="cuda")
    reference = torch.empty((1, 5, n), dtype=torch.bfloat16, device="cuda")
    starts = (0, 3)
    for expert, begin in enumerate(starts):
        end = starts[expert + 1] if expert + 1 < experts else 5
        unpacked = torch.stack((e2m1[(packed[expert] & 0xF).long()], e2m1[(packed[expert] >> 4).long()]), dim=-1).flatten(-2)
        decoded = (unpacked * scale[expert].float().repeat_interleave(16, dim=1)).to(torch.bfloat16)
        result = ((tokens[0, begin:end].float() @ decoded.float().t()) * tensors["factor"][expert, 0, 0]).to(torch.bfloat16)
        if epilogue == "squared_relu":
            result = torch.relu(result)
            result = (result * result).to(torch.bfloat16)
        reference[0, begin:end] = result
    return tensors, reference


@pytest.mark.L1
@pytest.mark.parametrize("epilogue", ("linear", "squared_relu"))
def test_wrapper_matches_semantic_reference_on_supported_gpu(epilogue):
    _, wrapper = _imports()
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("the kernel requires an SM100 or SM103 GPU")
    tensors, reference = _runtime_inputs(epilogue)

    result = wrapper(
        tensors["routed_tokens"],
        tensors["packed_weight"],
        tensors["weight_scale"],
        tensors["first_token_offset"],
        tensors["factor"],
        epilogue=epilogue,
    )
    torch.cuda.synchronize()

    assert result[0] is result["output"]
    torch.testing.assert_close(result["output"], reference, atol=0, rtol=0)


@pytest.mark.L1
def test_class_api_writes_caller_output_on_nondefault_stream():
    api, _ = _imports()
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("the kernel requires an SM100 or SM103 GPU")
    tensors, reference = _runtime_inputs("linear")
    op = _make_op(api, tensors, "linear")
    assert op.check_support()
    op.compile()
    output_ptr = tensors["output"].data_ptr()
    stream = torch.cuda.Stream()

    op.execute(
        tensors["routed_tokens"],
        tensors["packed_weight"],
        tensors["weight_scale"],
        tensors["first_token_offset"],
        tensors["factor"],
        tensors["output"],
        current_stream=cuda.CUstream(stream.cuda_stream),
    )
    stream.synchronize()

    assert tensors["output"].data_ptr() == output_ptr
    torch.testing.assert_close(tensors["output"], reference, atol=0, rtol=0)
