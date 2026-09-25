# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU correctness for the experimental bulk backward."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from cudnn._causal_conv1d_arch import F32X2_COMPUTE_CAPABILITIES, is_functional_arch
from fe_api.causal_conv1d_bulk.reference import causal_conv1d_bulk_reference

pytestmark = pytest.mark.L1


def _load_autograd_prototype():
    try:
        from cudnn.causal_conv1d_bulk_sm100 import (
            CausalConv1dBulkAutogradPrototype,
        )
    except (ImportError, OSError) as error:
        pytest.skip(f"CuTe DSL dependencies unavailable: {error}")
    return CausalConv1dBulkAutogradPrototype


def _dense_reference(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    tokens = x.shape[1]
    preactivation = F.conv1d(
        x.transpose(1, 2),
        weight.unsqueeze(1),
        bias,
        padding=3,
        groups=x.shape[2],
    )[
        ..., :tokens
    ].transpose(1, 2)
    return F.silu(preactivation)


def _reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    offsets: tuple[int, ...] | None,
) -> torch.Tensor:
    if offsets is None:
        return _dense_reference(x, weight, bias)
    return torch.cat(
        [_dense_reference(x[:, start:end], weight, bias) for start, end in zip(offsets[:-1], offsets[1:])],
        dim=1,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("shape", "offsets", "schedule", "with_bias"),
    [
        ((2, 65, 8), None, "t64", False),
        ((1, 70, 8), (0, 1, 3, 6, 70), "t64", False),
        ((1, 70, 8), (0, 1, 3, 6, 70), "t64-partial", False),
        ((2, 65, 8), None, "t64", True),
        ((1, 70, 8), (0, 1, 3, 6, 70), "t64", True),
        ((1, 70, 8), (0, 1, 3, 6, 70), "t64-partial", True),
        ((1, 16384, 8), None, "auto", True),
        ((1, 64, 512), None, "v4-stream", False),
        ((1, 49, 512), None, "v4-stream", False),
        ((1, 65, 256), None, "v2-cpasync", False),
        # This shape exposes a one-token final tile unless the planner folds it
        # into the preceding G8 tile.
        ((1, 8242, 2048), None, "v2-cpasync", False),
    ],
)
def test_stateless_backward_matches_independent_recurrence(shape, offsets, schedule, with_bias):
    capability = torch.cuda.get_device_capability()
    if not is_functional_arch(capability):
        pytest.skip(f"unsupported compute capability {capability}")
    if schedule == "v2-cpasync" and capability not in F32X2_COMPUTE_CAPABILITIES:
        pytest.skip(f"packed-f32x2 is unsupported on compute capability {capability}")

    prototype = _load_autograd_prototype()
    generator = torch.Generator(device="cuda").manual_seed(20260829)
    x = (torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25).requires_grad_()
    weight = (
        torch.randn(
            (shape[2], 4),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.25
    ).requires_grad_()
    bias = (torch.randn(shape[2], device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25).requires_grad_() if with_bias else None
    dy = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25
    cu_seqlens = None if offsets is None else torch.tensor(offsets, device="cuda", dtype=torch.int32)

    reference_x = x.detach().float().requires_grad_()
    reference_weight = weight.detach().float().requires_grad_()
    reference_bias = bias.detach().float().requires_grad_() if bias is not None else None
    expected = _reference(reference_x, reference_weight, reference_bias, offsets)
    expected.backward(dy.float())

    operation = prototype(x, weight, cu_seqlens, schedule=schedule, sample_bias=bias)
    actual = operation(x, weight, cu_seqlens, bias=bias)
    actual.backward(dy)

    torch.testing.assert_close(actual.float(), expected, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(x.grad.float(), reference_x.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(weight.grad.float(), reference_weight.grad, atol=1e-1, rtol=5e-2)
    if bias is not None:
        torch.testing.assert_close(bias.grad.float(), reference_bias.grad, atol=1e-1, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("shape", "schedule"),
    [
        ((1, 64, 512), "v4-stream"),
        ((1, 8192, 256), "v2-cpasync"),
    ],
)
def test_streaming_backward_supports_fp32_weight(shape, schedule):
    capability = torch.cuda.get_device_capability()
    if not is_functional_arch(capability):
        pytest.skip(f"unsupported compute capability {capability}")
    if schedule == "v2-cpasync" and capability not in F32X2_COMPUTE_CAPABILITIES:
        pytest.skip(f"packed-f32x2 is unsupported on compute capability {capability}")

    prototype = _load_autograd_prototype()
    generator = torch.Generator(device="cuda").manual_seed(20260903)
    x = (
        torch.randn(
            shape,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.25
    ).requires_grad_()
    weight = (
        torch.randn(
            (shape[2], 4),
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.25
    ).requires_grad_()
    dy = (
        torch.randn(
            shape,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.25
    )

    reference_x = x.detach().float().requires_grad_()
    reference_weight = weight.detach().requires_grad_()
    expected = _dense_reference(reference_x, reference_weight, None)
    expected.backward(dy.float())

    operation = prototype(x, weight, schedule=schedule)
    actual = operation(x, weight)
    actual.backward(dy)

    assert operation.backward_backend.kernel_variant in (
        "vec4-stream",
        "vec2-cpasync",
    )
    assert weight.grad.dtype == torch.float32
    torch.testing.assert_close(actual.float(), expected, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(x.grad.float(), reference_x.grad, atol=1.25e-1, rtol=5e-2)
    torch.testing.assert_close(weight.grad, reference_weight.grad, atol=2.0, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vec2_cpasync_fast_tanh_extremes_and_exact_cancellation():
    capability = torch.cuda.get_device_capability()
    if capability not in F32X2_COMPUTE_CAPABILITIES:
        pytest.skip(f"packed-f32x2 is unsupported on compute capability {capability}")

    prototype = _load_autograd_prototype()
    shape = (1, 64, 512)
    x = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    weight = torch.zeros((shape[2], 4), device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        levels = torch.tensor((-32.0, -16.0, -8.0, -4.0, -1.0, 0.0, 1.0, 4.0, 8.0, 16.0, 32.0), device="cuda", dtype=torch.bfloat16)
        x[..., :256] = levels.repeat((shape[1] * 256 + levels.numel() - 1) // levels.numel())[: shape[1] * 256].view(shape[1], 256)
        weight[:256, 3] = 1.0

        # For every interior token these four terms cancel z and dX exactly:
        # z = 96 - 96 + 64 - 64 and dX = -64 + 64 - 96 + 96.
        x[..., 256:] = 1.0
        weight[256:, 0] = 96.0
        weight[256:, 1] = -96.0
        weight[256:, 2] = 64.0
        weight[256:, 3] = -64.0
    x.requires_grad_()
    weight.requires_grad_()
    dy = torch.full_like(x, 1.359375)

    reference_x = x.detach().float().requires_grad_()
    reference_weight = weight.detach().float().requires_grad_()
    expected = _dense_reference(reference_x, reference_weight, None)
    expected.backward(dy.float())

    operation = prototype(x, weight, schedule="v2-cpasync")
    assert operation.backward_backend.kernel_variant == "vec2-cpasync"
    actual = operation(x, weight)
    actual.backward(dy)

    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(weight.grad).all()
    torch.testing.assert_close(x.grad.float(), reference_x.grad, atol=1.25e-1, rtol=5e-2)
    torch.testing.assert_close(weight.grad.float(), reference_weight.grad, atol=2.0, rtol=5e-2)
    torch.testing.assert_close(x.grad[0, 3:61, 256:], torch.zeros_like(x.grad[0, 3:61, 256:]), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("shape", "offsets", "schedule", "with_bias"),
    [
        ((2, 1, 8), None, "t64", False),
        ((2, 2, 8), None, "t64", True),
        ((2, 3, 8), None, "t64-partial", False),
        ((2, 4, 8), None, "t64-partial", True),
        ((1, 10, 8), (0, 1, 3, 6, 10), "t64", True),
        ((1, 10, 8), (0, 1, 3, 6, 10), "t64-partial", False),
    ],
)
def test_stateful_backward_matches_tail4_recurrence(shape, offsets, schedule, with_bias):
    """Exercise every positive short-sequence boundary and both dw schedules."""

    capability = torch.cuda.get_device_capability()
    if not is_functional_arch(capability):
        pytest.skip(f"unsupported compute capability {capability}")

    prototype = _load_autograd_prototype()
    generator = torch.Generator(device="cuda").manual_seed(20260830 + shape[1] + int(with_bias))
    num_sequences = shape[0] if offsets is None else len(offsets) - 1
    x = (torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25).requires_grad_()
    weight = (torch.randn((shape[2], 4), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25).requires_grad_()
    bias = (torch.randn(shape[2], device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25).requires_grad_() if with_bias else None
    initial_state = (torch.randn((num_sequences, shape[2], 4), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25).requires_grad_()
    dy = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25
    d_final_state = torch.randn(initial_state.shape, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25
    cu_seqlens = None if offsets is None else torch.tensor(offsets, device="cuda", dtype=torch.int32)

    reference_x = x.detach().float().requires_grad_()
    reference_weight = weight.detach().float().requires_grad_()
    reference_bias = bias.detach().float().requires_grad_() if bias is not None else None
    reference_initial_state = initial_state.detach().float().requires_grad_()
    expected_output, expected_final_state = causal_conv1d_bulk_reference(
        reference_x,
        reference_weight,
        bias=reference_bias,
        cu_seqlens=cu_seqlens,
        initial_state=reference_initial_state,
    )
    torch.autograd.backward(
        (expected_output, expected_final_state),
        (dy.float(), d_final_state.float()),
    )

    operation = prototype(
        x,
        weight,
        cu_seqlens,
        schedule=schedule,
        sample_bias=bias,
        sample_initial_state=initial_state,
        output_final_state=True,
    )
    actual = operation(
        x,
        weight,
        cu_seqlens,
        bias=bias,
        initial_state=initial_state,
        output_final_state=True,
    )
    torch.autograd.backward(
        (actual["output_tensor"], actual["final_state_tensor"]),
        (dy, d_final_state),
    )

    torch.testing.assert_close(actual["output_tensor"].float(), expected_output, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(actual["final_state_tensor"].float(), expected_final_state, atol=0, rtol=0)
    torch.testing.assert_close(x.grad.float(), reference_x.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(weight.grad.float(), reference_weight.grad, atol=1e-1, rtol=5e-2)
    torch.testing.assert_close(initial_state.grad.float(), reference_initial_state.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(initial_state.grad[..., 0], torch.zeros_like(initial_state.grad[..., 0]), atol=0, rtol=0)
    if bias is not None:
        torch.testing.assert_close(bias.grad.float(), reference_bias.grad, atol=1e-1, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_final_state_gradient_without_initial_state_reaches_only_surviving_tokens():
    capability = torch.cuda.get_device_capability()
    if not is_functional_arch(capability):
        pytest.skip(f"unsupported compute capability {capability}")

    prototype = _load_autograd_prototype()
    generator = torch.Generator(device="cuda").manual_seed(20260831)
    shape = (1, 6, 8)
    offsets = (0, 1, 3, 6)
    x = (torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25).requires_grad_()
    weight = (torch.randn((shape[2], 4), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25).requires_grad_()
    dy = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25
    d_final_state = torch.randn((len(offsets) - 1, shape[2], 4), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25
    cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.int32)

    reference_x = x.detach().float().requires_grad_()
    reference_weight = weight.detach().float().requires_grad_()
    expected_output, expected_final_state = causal_conv1d_bulk_reference(
        reference_x,
        reference_weight,
        cu_seqlens=cu_seqlens,
    )
    torch.autograd.backward((expected_output, expected_final_state), (dy.float(), d_final_state.float()))

    operation = prototype(x, weight, cu_seqlens, output_final_state=True)
    actual = operation(x, weight, cu_seqlens, output_final_state=True)
    torch.autograd.backward((actual["output_tensor"], actual["final_state_tensor"]), (dy, d_final_state))

    torch.testing.assert_close(x.grad.float(), reference_x.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(weight.grad.float(), reference_weight.grad, atol=1e-1, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_final_state_gradient_is_accumulated_before_the_single_bf16_dx_cast():
    capability = torch.cuda.get_device_capability()
    if not is_functional_arch(capability):
        pytest.skip(f"unsupported compute capability {capability}")

    prototype = _load_autograd_prototype()
    x = torch.zeros((1, 1, 8), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.zeros((8, 4), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with torch.no_grad():
        weight[:, 3] = 96.0
    dy = torch.full_like(x, 1.359375)
    d_final_state = torch.zeros((1, 8, 4), device="cuda", dtype=torch.bfloat16)
    d_final_state[..., 3] = -65.0

    operation = prototype(x, weight, output_final_state=True)
    actual = operation(x, weight, output_final_state=True)
    torch.autograd.backward(
        (actual["output_tensor"], actual["final_state_tensor"]),
        (dy, d_final_state),
    )

    # At z=0, SiLU'=0.5: 1.359375 * 0.5 * 96 - 65 = 0.25.
    torch.testing.assert_close(x.grad, torch.full_like(x, 0.25), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_initial_state_gradient_without_final_state_output():
    capability = torch.cuda.get_device_capability()
    if not is_functional_arch(capability):
        pytest.skip(f"unsupported compute capability {capability}")

    prototype = _load_autograd_prototype()
    generator = torch.Generator(device="cuda").manual_seed(20260901)
    shape = (1, 3, 8)
    x = (torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25).requires_grad_()
    weight = (torch.randn((shape[2], 4), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25).requires_grad_()
    initial_state = (torch.randn((shape[0], shape[2], 4), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25).requires_grad_()
    dy = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25

    reference_x = x.detach().float().requires_grad_()
    reference_weight = weight.detach().float().requires_grad_()
    reference_initial_state = initial_state.detach().float().requires_grad_()
    expected_output, _ = causal_conv1d_bulk_reference(
        reference_x,
        reference_weight,
        initial_state=reference_initial_state,
    )
    expected_output.backward(dy.float())

    operation = prototype(x, weight, sample_initial_state=initial_state)
    actual = operation(x, weight, initial_state=initial_state)
    assert isinstance(actual, torch.Tensor)
    actual.backward(dy)

    torch.testing.assert_close(x.grad.float(), reference_x.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(weight.grad.float(), reference_weight.grad, atol=1e-1, rtol=5e-2)
    torch.testing.assert_close(initial_state.grad.float(), reference_initial_state.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(initial_state.grad[..., 0], torch.zeros_like(initial_state.grad[..., 0]), atol=0, rtol=0)
