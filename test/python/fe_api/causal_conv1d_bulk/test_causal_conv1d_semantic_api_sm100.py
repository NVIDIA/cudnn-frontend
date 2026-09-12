# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L1 GPU correctness for the model-facing causal-conv state adapter."""

from __future__ import annotations

import pytest
import torch
from fe_api.causal_conv1d_bulk.reference import causal_conv1d_bulk_reference

pytestmark = [
    pytest.mark.L1,
    pytest.mark.gpu_exclusive,
    pytest.mark.xdist_group(name="gpu_exclusive"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def _require_native_route() -> None:
    try:
        from cudnn._causal_conv1d_arch import is_functional_arch
        from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old
    except (ImportError, OSError) as error:
        pytest.skip(f"CuTe DSL dependencies unavailable: {error}")
    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        pytest.skip("causal_conv1d state requires nvidia-cutlass-dsl>=4.7.0")
    capability = torch.cuda.get_device_capability()
    if not is_functional_arch(capability):
        pytest.skip(f"unsupported compute capability {capability}")


def test_glm_mixed_fp32_weight_contract_matches_fp32_oracle() -> None:
    """Keep the exact GLM D=24576 parameter and gradient dtypes native."""

    _require_native_route()
    from cudnn.ops.causal_conv1d import (
        _get_causal_conv1d_last_route,
        causal_conv1d,
    )

    generator = torch.Generator(device="cuda").manual_seed(20260903)
    batch, tokens, channels = 1, 16, 24576
    x_btd = (
        torch.randn(
            (batch, tokens, channels),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.25
    ).requires_grad_()
    weight = (
        torch.randn(
            (channels, 4),
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.25
    ).requires_grad_()
    dy_btd = (
        torch.randn(
            x_btd.shape,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.25
    )

    reference_x = x_btd.detach().float().requires_grad_()
    reference_weight = weight.detach().requires_grad_()
    expected_btd, _ = causal_conv1d_bulk_reference(
        reference_x,
        reference_weight,
    )
    expected_btd.backward(dy_btd.float())

    actual = causal_conv1d(
        x_btd.transpose(1, 2),
        weight,
        activation="silu",
    )
    assert _get_causal_conv1d_last_route() == "native-autograd"
    actual.backward(dy_btd.transpose(1, 2))

    assert actual.dtype == torch.bfloat16
    assert x_btd.grad.dtype == torch.bfloat16
    assert weight.grad.dtype == torch.float32
    torch.testing.assert_close(actual.float(), expected_btd.transpose(1, 2), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(x_btd.grad.float(), reference_x.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(weight.grad, reference_weight.grad, atol=1e-1, rtol=5e-2)


@pytest.mark.parametrize("packed", [False, True], ids=("dense", "packed"))
def test_public_w_minus_one_state_matches_oracle_and_initial_gradient(
    packed: bool,
) -> None:
    """Exercise public state shape, output/final values, and dInitial mapping."""

    _require_native_route()
    from cudnn.ops.causal_conv1d import (
        _get_causal_conv1d_last_route,
        causal_conv1d,
    )

    generator = torch.Generator(device="cuda").manual_seed(20260902 + int(packed))
    batch, tokens, channels = (1, 6, 8) if packed else (2, 5, 8)
    offsets = (0, 1, 3, 6) if packed else None
    num_sequences = len(offsets) - 1 if offsets is not None else batch
    cu_seqlens = None if offsets is None else torch.tensor(offsets, device="cuda", dtype=torch.int32)

    x_btd = (
        torch.randn(
            (batch, tokens, channels),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.25
    ).requires_grad_()
    public_x = x_btd.transpose(1, 2)
    weight = (torch.randn((channels, 4), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25).requires_grad_()
    bias = (torch.randn(channels, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25).requires_grad_()
    initial_backing = torch.randn(
        (num_sequences, 3, channels),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    initial_states = (initial_backing * 0.25).transpose(1, 2).detach().requires_grad_()
    final_states_out = torch.empty(
        (num_sequences, 3, channels),
        device="cuda",
        dtype=torch.bfloat16,
    ).transpose(1, 2)
    dy_btd = torch.randn(x_btd.shape, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25
    d_final_states = (
        torch.randn(
            initial_states.shape,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.25
    )

    reference_x = x_btd.detach().float().requires_grad_()
    reference_weight = weight.detach().float().requires_grad_()
    reference_bias = bias.detach().float().requires_grad_()
    reference_initial_states = initial_states.detach().float().requires_grad_()
    reference_full_state = torch.cat(
        (torch.zeros_like(reference_initial_states[..., :1]), reference_initial_states),
        dim=-1,
    )
    expected_btd, expected_full_final = causal_conv1d_bulk_reference(
        reference_x,
        reference_weight,
        bias=reference_bias,
        cu_seqlens=cu_seqlens,
        initial_state=reference_full_state,
    )
    expected_final_states = expected_full_final[..., 1:]
    d_full_final = torch.cat(
        (torch.zeros_like(d_final_states[..., :1]), d_final_states),
        dim=-1,
    )
    torch.autograd.backward(
        (expected_btd, expected_full_final),
        (dy_btd.float(), d_full_final.float()),
    )

    actual, actual_final_states = causal_conv1d(
        public_x,
        weight,
        bias,
        "silu",
        cu_seqlens=cu_seqlens,
        initial_states=initial_states,
        return_final_states=True,
        final_states_out=final_states_out,
    )
    assert _get_causal_conv1d_last_route() == "native-autograd"
    torch.autograd.backward(
        (actual, actual_final_states),
        (dy_btd.transpose(1, 2), d_final_states),
    )

    assert actual.shape == public_x.shape
    assert actual.stride() == public_x.stride()
    assert actual_final_states is final_states_out
    assert actual_final_states.shape == (num_sequences, channels, 3)
    torch.testing.assert_close(actual.float(), expected_btd.transpose(1, 2), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(actual_final_states.float(), expected_final_states, atol=0, rtol=0)
    torch.testing.assert_close(initial_states.grad.float(), reference_initial_states.grad, atol=5e-2, rtol=5e-2)


def test_public_inference_separates_bf16_and_fp32_weight_plans() -> None:
    """A same-shape FP32 call must not reuse the preceding BF16 plan."""

    _require_native_route()
    from cudnn.ops.causal_conv1d import (
        _get_causal_conv1d_last_route,
        causal_conv1d,
    )

    generator = torch.Generator(device="cuda").manual_seed(20260904)
    x_btd = torch.randn((1, 8, 16), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25
    weight_bf16 = torch.randn((16, 4), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25
    weight_fp32 = torch.randn((16, 4), device="cuda", dtype=torch.float32, generator=generator) * 0.25

    expected_bf16, _ = causal_conv1d_bulk_reference(x_btd, weight_bf16)
    expected_fp32, _ = causal_conv1d_bulk_reference(x_btd, weight_fp32)

    with torch.no_grad():
        actual_bf16 = causal_conv1d(x_btd.transpose(1, 2), weight_bf16, activation="silu")
        assert _get_causal_conv1d_last_route() == "native-inference"
        actual_fp32 = causal_conv1d(x_btd.transpose(1, 2), weight_fp32, activation="silu")
        assert _get_causal_conv1d_last_route() == "native-inference"

    torch.testing.assert_close(actual_bf16.float(), expected_bf16.transpose(1, 2).float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(actual_fp32.float(), expected_fp32.transpose(1, 2).float(), atol=3e-2, rtol=3e-2)
