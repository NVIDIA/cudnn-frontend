# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for the fused Conv3D video-VAE APIs."""

import math
from functools import partial

import pytest
import torch
import torch.nn.functional as F
from cuda.bindings import driver as cuda
from cudnn.conv.frost._cutedsl import requirement_error as cutedsl_requirement_error


def _has_supported_gpu() -> bool:
    """Return whether the current CUDA device is supported by these kernels."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in (
        (10, 0),
        (10, 3),
    )


_dsl_error = cutedsl_requirement_error("Conv3D post-operation tests", (4, 9))
requires_cutedsl = pytest.mark.skipif(_dsl_error is not None, reason=_dsl_error or "CuTeDSL version is supported")
requires_gpu = pytest.mark.skipif(not _has_supported_gpu(), reason="requires SM100 or SM103")


def _make_inputs(input_channels: int, output_channels: int, shape: tuple[int, ...] = (1, 4, 6, 7)):
    """Create BF16 convolution operands, packed weights, and the expected output shape."""
    from cudnn import pack_conv3d_weight_sm100

    n, t, h, w = shape
    shape = (*shape, input_channels)
    output_shape = (n, t - 2, h - 2, w - 2, output_channels)
    input = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = (
        torch.randn(
            output_channels,
            input_channels,
            3,
            3,
            3,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    return input, weight, pack_conv3d_weight_sm100(weight), output_shape


def _conv_reference(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute valid Conv3D through Torch with NTHWC input and output."""
    return F.conv3d(input.permute(0, 4, 1, 2, 3), weight).permute(0, 2, 3, 4, 1)


def _norm_silu_reference(
    conv: torch.Tensor,
    bias: torch.Tensor,
    gamma: torch.Tensor,
    previous: torch.Tensor | None,
    residual: torch.Tensor | None,
    residual_bias: torch.Tensor | None,
):
    """Compute rounded normalization/SiLU with padded history and cache outputs."""
    value = (conv.float() + bias.float()).to(torch.bfloat16)
    if residual is not None:
        skip = residual if residual_bias is None else (residual.float() + residual_bias.float()).to(torch.bfloat16)
        value = (value.float() + skip.float()).to(torch.bfloat16)

    normalized = F.normalize(value.float(), dim=-1).to(torch.bfloat16)
    normalized = (normalized.float() * math.sqrt(value.shape[-1])).to(torch.bfloat16)
    affine = (normalized.float() * gamma.float()).to(torch.bfloat16)
    activated = F.silu(affine.float()).to(torch.bfloat16)
    joined = activated if previous is None else torch.cat((previous, activated), dim=1)
    history = 0 if previous is None else previous.shape[1]
    padded = F.pad(joined, (0, 0, 1, 1, 1, 1, 2 - history, 0))
    return padded, joined[:, -2:].contiguous(), value if residual is not None else None


def _norm_silu_output_reference(
    conv: torch.Tensor,
    bias: torch.Tensor,
    gamma: torch.Tensor,
    residual: torch.Tensor | None,
    residual_bias: torch.Tensor | None,
):
    """Return contiguous normalized activations and the optional residual sum."""
    padded, _, residual_output = _norm_silu_reference(conv, bias, gamma, None, residual, residual_bias)
    return padded[:, 2:, 1:-1, 1:-1, :].contiguous(), residual_output


def _spatial_reference(
    conv: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    residual_bias: torch.Tensor | None,
) -> torch.Tensor:
    """Compute rounded convolution bias/residual addition with bottom/right padding."""
    biased_conv = (conv.float() + bias.float()).to(torch.bfloat16)
    skip = residual if residual_bias is None else (residual.float() + residual_bias.float()).to(torch.bfloat16)
    value = (biased_conv.float() + skip.float()).to(torch.bfloat16)
    return F.pad(value, (0, 0, 0, 1, 0, 1))


def _assert_warm_execute_contract(execute):
    """Check prepared execution without counting setup or reference work."""
    execute()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    previous_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            execute()
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before


@pytest.mark.L0
def test_public_exports():
    """Verify top-level and family exports resolve to the same public objects."""
    import cudnn
    from cudnn.conv import cutedsl
    from cudnn.conv.cutedsl import conv3d_postops

    for name in conv3d_postops.__all__:
        assert getattr(cudnn, name) is getattr(conv3d_postops, name)
        assert getattr(cutedsl, name) is getattr(conv3d_postops, name)


@pytest.mark.L0
def test_pack_conv3d_weight_rejects_unsupported_channels():
    """Verify weight packing rejects unsupported channel pairs."""
    from cudnn import pack_conv3d_weight_sm100

    weight = torch.empty((128, 128, 3, 3, 3), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="unsupported channel pair"):
        pack_conv3d_weight_sm100(weight)


@pytest.mark.L0
def test_pack_conv3d_weight_layout_and_padding():
    """Verify filter reordering and zero-filled channel padding."""
    from cudnn import pack_conv3d_weight_sm100

    weight = torch.arange(160 * 160 * 27, dtype=torch.bfloat16).reshape(160, 160, 3, 3, 3)
    packed = pack_conv3d_weight_sm100(weight)

    assert packed.shape == (160, 3, 3, 3, 192)
    torch.testing.assert_close(packed[..., :160], weight.permute(0, 2, 3, 4, 1), rtol=0, atol=0)
    torch.testing.assert_close(packed[..., 160:], torch.zeros_like(packed[..., 160:]), rtol=0, atol=0)


@requires_cutedsl
@requires_gpu
@pytest.mark.parametrize(
    "frames,history_frames,height,width",
    (
        pytest.param(1, 0, 4, 5, marks=pytest.mark.L0),
        pytest.param(4, 1, 4, 5, marks=pytest.mark.L0),
        pytest.param(1, 2, 4, 5, marks=pytest.mark.L0),
        pytest.param(4, 2, 96, 97, marks=pytest.mark.L1, id="persistent-tile-reuse"),
    ),
)
@torch.inference_mode()
def test_causal_conv3d_class_and_wrapper(frames, history_frames, height, width):
    """Check causal packing, convolution, caching, changed-input graph replay, and wrapper results."""
    from cudnn import (
        CausalConv3dWithCacheSm100,
        causal_conv3d_with_cache_wrapper_sm100,
        pack_causal_conv3d_weight_sm100,
    )

    torch.manual_seed(2)
    video = torch.randn((2, 12, frames + 2, height, width), device="cuda", dtype=torch.bfloat16) * 0.1
    input = video[:, :, 1 : frames + 1]
    previous = torch.randn((2, history_frames, height, width, 12), device="cuda", dtype=torch.bfloat16) * 0.1 if history_frames else None
    cache_frames = min(2, frames + history_frames)
    weight = torch.randn((160, 12, 3, 3, 3), device="cuda", dtype=torch.bfloat16) * 0.1
    packed_weight = pack_causal_conv3d_weight_sm100(weight)
    padded_input = torch.empty((2, frames + 2, height + 2, width + 2, 16), device="cuda", dtype=torch.bfloat16)
    cache_output = torch.empty((2, cache_frames, height, width, 12), device="cuda", dtype=torch.bfloat16)
    output = torch.empty((2, frames, height, width, 160), device="cuda", dtype=torch.bfloat16)

    plan = CausalConv3dWithCacheSm100(
        input,
        packed_weight,
        padded_input,
        cache_output,
        output,
        previous,
    )
    assert plan.check_support()
    plan.compile()
    _assert_warm_execute_contract(lambda: plan.execute(input, packed_weight, padded_input, cache_output, output, previous))

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        plan.execute(input, packed_weight, padded_input, cache_output, output, previous)

    for _ in range(3):
        input.normal_(std=0.1)
        if previous is not None:
            previous.normal_(std=0.1)
        padded_input.fill_(float("nan"))
        cache_output.fill_(float("nan"))
        output.fill_(float("nan"))
        graph.replay()

        current = input.permute(0, 2, 3, 4, 1)
        joined = torch.cat((previous, current), dim=1) if previous is not None else current
        expected_padded = F.pad(joined, (0, 4, 1, 1, 1, 1, 2 - history_frames, 0))
        expected = F.conv3d(expected_padded.permute(0, 4, 1, 2, 3)[:, :12], weight).permute(0, 2, 3, 4, 1)
        torch.testing.assert_close(padded_input, expected_padded, atol=0, rtol=0)
        torch.testing.assert_close(cache_output, joined[:, -2:], atol=0, rtol=0)
        torch.testing.assert_close(output, expected, atol=0.02, rtol=0.02)

    wrapped = causal_conv3d_with_cache_wrapper_sm100(input, packed_weight, previous)
    torch.testing.assert_close(wrapped["output"], expected, atol=0.02, rtol=0.02)
    torch.testing.assert_close(wrapped["cache_output"], joined[:, -2:], atol=0, rtol=0)

    misaligned_weight = torch.empty(packed_weight.numel() + 1, device="cuda", dtype=packed_weight.dtype)[1:].view_as(packed_weight)
    invalid = CausalConv3dWithCacheSm100(input, misaligned_weight, padded_input, cache_output, output, previous)
    with pytest.raises(ValueError, match="packed_weight must be 16-byte aligned"):
        invalid.check_support()
    if previous is not None:
        shifted_previous = torch.empty(previous.numel() + 4, device="cuda", dtype=previous.dtype)[4:].view_as(previous)
        shifted_previous.copy_(previous)
        rebound = causal_conv3d_with_cache_wrapper_sm100(input, packed_weight, shifted_previous)
        torch.testing.assert_close(rebound["output"], expected, atol=0.02, rtol=0.02)
        torch.testing.assert_close(rebound["cache_output"], joined[:, -2:], atol=0, rtol=0)
    invalid_previous = torch.empty((2, 3, height, width, 12), device="cuda", dtype=input.dtype)
    invalid = CausalConv3dWithCacheSm100(input, packed_weight, padded_input, cache_output, output, invalid_previous)
    with pytest.raises(ValueError, match="previous must contain one or two frames"):
        invalid.check_support()
    aliased_cache = padded_input.view(-1)[: cache_output.numel()].view_as(cache_output)
    with pytest.raises(ValueError, match="must not overlap"):
        CausalConv3dWithCacheSm100(input, packed_weight, padded_input, aliased_cache, output, previous).check_support()


@pytest.mark.L0
@pytest.mark.parametrize("version", ("4.6.2", "4.8.0"))
@pytest.mark.parametrize(
    "class_name,required_samples",
    (
        ("Conv3dRawSm100", 3),
        ("Conv3dRmsNormSiluSm100", 5),
        ("Conv3dRmsNormSiluPadSm100", 6),
        ("Conv3dBiasResidualPadSm100", 5),
        ("CausalConv3dWithCacheSm100", 5),
        ("RmsNormSiluPadSm100", 4),
    ),
)
def test_conv3d_requires_cutedsl_4_9(monkeypatch, version, class_name, required_samples):
    """Verify all plans reject older public DSL versions before tensor validation."""
    import cudnn
    from cudnn.frost import buffers

    # The version gate must run before tensor validation or kernel imports.
    sample = torch.empty(0, dtype=torch.bfloat16)
    plan = getattr(cudnn, class_name)(*[sample] * required_samples)

    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", version)))
    with pytest.raises(
        NotImplementedError,
        match=r"requires nvidia-cutlass-dsl >= 4\.9; found " + version.replace(".", r"\."),
    ):
        plan.check_support()


@pytest.mark.L0
@requires_cutedsl
@requires_gpu
@pytest.mark.parametrize("case", ("channels", "layout", "alignment", "history"))
def test_conv3d_declines_unsupported_metadata(case):
    """Verify support checks reject unsupported channels, layouts, alignment, and history."""
    from cudnn import Conv3dRmsNormSiluPadSm100

    channels = 640 if case == "channels" else 160
    input_channels = 320 if case == "channels" else 160
    packed_channels = (input_channels + 63) // 64 * 64
    input = torch.empty((1, 3, 4, 4, input_channels), dtype=torch.bfloat16, device="cuda")
    weight = torch.empty((channels, 3, 3, 3, packed_channels), dtype=torch.bfloat16, device="cuda")
    bias = torch.empty(channels, dtype=torch.bfloat16, device="cuda")
    gamma = torch.empty_like(bias)
    padded = torch.empty((1, 3, 4, 4, channels), dtype=torch.bfloat16, device="cuda")
    cache = torch.empty((1, 1, 2, 2, channels), dtype=torch.bfloat16, device="cuda")
    if case == "layout":
        input = input.transpose(2, 3)
    elif case == "alignment":
        input = torch.empty(input.numel() + 1, dtype=input.dtype, device=input.device)[1:].view_as(input)
    expected = {
        "channels": "96, 160, 192, 256, or 320 output channels",
        "layout": "input must be contiguous",
        "alignment": "16-byte aligned",
        "history": "history_frames",
    }
    plan = Conv3dRmsNormSiluPadSm100(input, weight, bias, gamma, padded, cache, history_frames=3 if case == "history" else 0)
    with pytest.raises(ValueError, match=expected[case]):
        plan.check_support()


@requires_cutedsl
@requires_gpu
@pytest.mark.parametrize(
    "shape",
    (
        pytest.param((1, 4, 6, 7), marks=pytest.mark.L0),
        pytest.param((2, 6, 98, 98), marks=pytest.mark.L1, id="persistent-tile-reuse"),
    ),
)
@torch.inference_mode()
def test_conv3d_raw_class_and_wrapper(shape, monkeypatch):
    """Check raw convolution, sample validation, and reuse of the wrapper's compiled plan."""
    from cudnn import Conv3dRawSm100, conv3d_raw_wrapper_sm100

    torch.manual_seed(3)
    input, weight, packed_weight, output_shape = _make_inputs(320, 640, shape)
    output = torch.empty(output_shape, device="cuda", dtype=torch.bfloat16)
    plan = Conv3dRawSm100(input, packed_weight, output)
    assert plan.check_support()
    plan.compile()
    _assert_warm_execute_contract(lambda: plan.execute(input, packed_weight, output))

    expected = _conv_reference(input, weight)
    torch.testing.assert_close(output, expected, atol=0.02, rtol=0.02)
    replacement = torch.randn_like(input) * 0.1
    replacement_output = torch.empty_like(output)
    plan.execute(replacement, packed_weight, replacement_output)
    torch.testing.assert_close(replacement_output, _conv_reference(replacement, weight), atol=0.02, rtol=0.02)
    wrapped = conv3d_raw_wrapper_sm100(input, packed_weight)
    torch.testing.assert_close(wrapped, expected, atol=0.02, rtol=0.02)

    from cutlass import cute

    with monkeypatch.context() as patch:
        patch.setattr(cute, "compile", lambda *args, **kwargs: pytest.fail("warm wrapper recompiled"))
        wrapped = conv3d_raw_wrapper_sm100(input, packed_weight)
    torch.testing.assert_close(wrapped, expected, atol=0.02, rtol=0.02)

    with pytest.raises(ValueError, match="output tensor shape mismatch"):
        Conv3dRawSm100(input, packed_weight, output[..., :-1]).check_support()

    aliased_output = packed_weight.view(-1)[: output.numel()].view_as(output) if output.numel() <= packed_weight.numel() else None
    if aliased_output is not None:
        with pytest.raises(ValueError, match="must not overlap packed_weight"):
            Conv3dRawSm100(input, packed_weight, aliased_output).check_support()
    with torch.inference_mode(False), torch.enable_grad():
        grad_input = input.clone().requires_grad_()
        with pytest.raises(RuntimeError, match="inference-only"):
            Conv3dRawSm100(grad_input, packed_weight, output).check_support()


@requires_cutedsl
@requires_gpu
@pytest.mark.parametrize("channels", (160, 320, 640))
@pytest.mark.parametrize(
    "frames,history_frames",
    (
        pytest.param(1, 0, marks=pytest.mark.L0),
        pytest.param(4, 2, marks=pytest.mark.L0),
        pytest.param(4, 1, marks=pytest.mark.L1),
        pytest.param(1, 2, marks=pytest.mark.L1),
    ),
)
@torch.inference_mode()
def test_rmsnorm_silu_pad_class_and_wrapper(channels, frames, history_frames):
    """Check standalone normalization/preparation, optional additions, and graph replay."""
    from cudnn import (
        RmsNormSiluPadSm100,
        rmsnorm_silu_pad_wrapper_sm100,
    )

    torch.manual_seed(4)
    shape = (2, frames, 4, 5, channels)
    input = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1
    gamma = torch.randn(channels, device="cuda", dtype=torch.bfloat16)
    input_bias = torch.randn(channels, device="cuda", dtype=torch.bfloat16) * 0.1
    previous = torch.randn((2, history_frames, 4, 5, channels), device="cuda", dtype=torch.bfloat16) if history_frames else None
    residual = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1
    residual_bias = torch.randn(channels, device="cuda", dtype=torch.bfloat16) * 0.1
    padded_output = torch.empty((2, frames + 2, 6, 7, channels), device="cuda", dtype=torch.bfloat16)
    cache_output = torch.empty((2, min(2, frames + history_frames), 4, 5, channels), device="cuda", dtype=torch.bfloat16)
    residual_output = torch.empty_like(input)

    with pytest.raises(ValueError, match="residual_output must not overlap input"):
        RmsNormSiluPadSm100(input, gamma, padded_output, cache_output, sample_previous=previous, sample_residual_output=input).check_support()

    plan = RmsNormSiluPadSm100(
        input,
        gamma,
        padded_output,
        cache_output,
        input_bias,
        previous,
        residual,
        residual_bias,
        residual_output,
    )
    assert plan.check_support()
    plan.compile()
    plan.execute(
        input,
        gamma,
        padded_output,
        cache_output,
        input_bias,
        previous,
        residual,
        residual_bias,
        residual_output,
    )

    expected = _norm_silu_reference(input, input_bias, gamma, previous, residual, residual_bias)
    _assert_warm_execute_contract(
        lambda: plan.execute(input, gamma, padded_output, cache_output, input_bias, previous, residual, residual_bias, residual_output)
    )
    torch.testing.assert_close(padded_output, expected[0], atol=0.02, rtol=0.02)
    torch.testing.assert_close(cache_output, expected[1], atol=0.02, rtol=0.02)
    torch.testing.assert_close(residual_output, expected[2], atol=0.02, rtol=0.02)

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        plan.execute(
            input,
            gamma,
            padded_output,
            cache_output,
            input_bias,
            previous,
            residual,
            residual_bias,
            residual_output,
        )
    input.copy_(torch.randn_like(input) * 0.1)
    graph.replay()
    expected = _norm_silu_reference(input, input_bias, gamma, previous, residual, residual_bias)
    torch.testing.assert_close(padded_output, expected[0], atol=0.02, rtol=0.02)

    wrapped = rmsnorm_silu_pad_wrapper_sm100(
        input,
        gamma,
        input_bias,
        previous,
        residual,
        residual_bias,
    )
    torch.testing.assert_close(wrapped["padded_output"], expected[0], atol=0.02, rtol=0.02)
    torch.testing.assert_close(wrapped["cache_output"], expected[1], atol=0.02, rtol=0.02)
    torch.testing.assert_close(wrapped["residual_output"], expected[2], atol=0.02, rtol=0.02)

    without_residual = rmsnorm_silu_pad_wrapper_sm100(input, gamma, input_bias, previous)
    expected_without_residual = _norm_silu_reference(input, input_bias, gamma, previous, None, None)
    torch.testing.assert_close(without_residual["padded_output"], expected_without_residual[0], atol=0.02, rtol=0.02)
    torch.testing.assert_close(without_residual["cache_output"], expected_without_residual[1], atol=0.02, rtol=0.02)
    assert without_residual["residual_output"] is None

    saved = rmsnorm_silu_pad_wrapper_sm100(input, gamma, input_bias, previous, save_input=True)
    torch.testing.assert_close(saved["residual_output"], (input.float() + input_bias.float()).bfloat16(), atol=0, rtol=0)
    torch.testing.assert_close(saved["padded_output"], expected_without_residual[0], atol=0.02, rtol=0.02)


@pytest.mark.L0
@requires_cutedsl
@requires_gpu
@torch.inference_mode()
def test_conv3d_rmsnorm_silu_contiguous_class_and_wrapper():
    """Check contiguous fused normalization output and the saved residual sum."""
    from cudnn import Conv3dRmsNormSiluSm100, conv3d_rmsnorm_silu_wrapper_sm100

    torch.manual_seed(5)
    input, weight, packed_weight, output_shape = _make_inputs(320, 320)
    bias = torch.randn(320, device="cuda", dtype=torch.bfloat16) * 0.1
    gamma = torch.randn(320, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(output_shape, device="cuda", dtype=torch.bfloat16)
    residual_bias = torch.randn(320, device="cuda", dtype=torch.bfloat16) * 0.1
    output = torch.empty(output_shape, device="cuda", dtype=torch.bfloat16)
    residual_output = torch.empty_like(output)

    plan = Conv3dRmsNormSiluSm100(
        input,
        packed_weight,
        bias,
        gamma,
        output,
        residual,
        residual_bias,
        residual_output,
    )
    assert plan.check_support()
    plan.compile()
    plan.execute(
        input,
        packed_weight,
        bias,
        gamma,
        output,
        residual,
        residual_bias,
        residual_output,
    )

    expected = _norm_silu_output_reference(_conv_reference(input, weight), bias, gamma, residual, residual_bias)
    torch.testing.assert_close(output, expected[0], atol=0.02, rtol=0.02)
    torch.testing.assert_close(residual_output, expected[1], atol=0.02, rtol=0.02)

    wrapped = conv3d_rmsnorm_silu_wrapper_sm100(input, packed_weight, bias, gamma, residual, residual_bias)
    torch.testing.assert_close(wrapped["output"], expected[0], atol=0.02, rtol=0.02)
    torch.testing.assert_close(wrapped["residual_output"], expected[1], atol=0.02, rtol=0.02)


@requires_cutedsl
@requires_gpu
@pytest.mark.parametrize("channels", (160, 320))
@pytest.mark.parametrize(
    "frames,history_frames",
    (
        pytest.param(1, 0, marks=pytest.mark.L0),
        pytest.param(4, 2, marks=pytest.mark.L0),
        pytest.param(1, 1, marks=pytest.mark.L1),
        pytest.param(1, 2, marks=pytest.mark.L1),
    ),
)
@torch.inference_mode()
def test_conv3d_rmsnorm_silu_pad_class_and_wrapper(channels, frames, history_frames):
    """Check fused padding outputs, preserved history, sample validation, and graph replay."""
    from cudnn import (
        Conv3dRmsNormSiluPadSm100,
        conv3d_rmsnorm_silu_pad_wrapper_sm100,
    )

    torch.manual_seed(7)
    input, weight, packed_weight, output_shape = _make_inputs(channels, channels, (1, frames + 2, 6, 7))
    bias = torch.randn(channels, device="cuda", dtype=torch.bfloat16) * 0.1
    gamma = torch.randn(channels, device="cuda", dtype=torch.bfloat16)
    previous = torch.randn((1, history_frames, 4, 5, channels), device="cuda", dtype=torch.bfloat16) if history_frames else None
    residual = torch.randn(output_shape, device="cuda", dtype=torch.bfloat16)
    residual_bias = torch.randn(channels, device="cuda", dtype=torch.bfloat16) * 0.1
    padded = torch.full((1, frames + 2, 6, 7, channels), float("nan"), device="cuda", dtype=torch.bfloat16)
    cache = torch.full((1, min(2, frames + history_frames), 4, 5, channels), float("nan"), device="cuda", dtype=torch.bfloat16)
    residual_output = torch.empty(output_shape, device="cuda", dtype=torch.bfloat16)

    def copy_history(padded_output, cache_output):
        """Fill the caller-owned history regions of padded output and cache."""
        if previous is not None:
            padded_output[:, 2 - history_frames : 2, 1:-1, 1:-1, :].copy_(previous)
            old_frames = max(cache_output.shape[1] - frames, 0)
            if old_frames:
                cache_output[:, :old_frames].copy_(previous[:, -old_frames:])

    # Prepopulate caller-owned history; the kernel must preserve it exactly.
    copy_history(padded, cache)

    plan = Conv3dRmsNormSiluPadSm100(
        input,
        packed_weight,
        bias,
        gamma,
        padded,
        cache,
        history_frames,
        residual,
        residual_bias,
        residual_output,
    )
    assert plan.check_support()
    plan.compile()
    plan.execute(
        input,
        packed_weight,
        bias,
        gamma,
        padded,
        cache,
        residual,
        residual_bias,
        residual_output,
    )
    _assert_warm_execute_contract(lambda: plan.execute(input, packed_weight, bias, gamma, padded, cache, residual, residual_bias, residual_output))
    with pytest.raises(ValueError, match="residual and residual_output"):
        Conv3dRmsNormSiluPadSm100(
            input,
            packed_weight,
            bias,
            gamma,
            padded,
            cache,
            history_frames,
            sample_residual_bias=residual_bias,
            sample_residual_output=residual_output,
        ).check_support()

    expected = _norm_silu_reference(
        _conv_reference(input, weight),
        bias,
        gamma,
        previous,
        residual,
        residual_bias,
    )
    torch.testing.assert_close(padded, expected[0], atol=0.02, rtol=0.02)
    torch.testing.assert_close(cache, expected[1], atol=0.02, rtol=0.02)
    torch.testing.assert_close(residual_output, expected[2], atol=0.02, rtol=0.02)
    if previous is not None:
        torch.testing.assert_close(padded[:, 2 - history_frames : 2, 1:-1, 1:-1, :], previous, atol=0, rtol=0)
        old_frames = max(cache.shape[1] - frames, 0)
        if old_frames:
            torch.testing.assert_close(cache[:, :old_frames], previous[:, -old_frames:], atol=0, rtol=0)

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        plan.execute(
            input,
            packed_weight,
            bias,
            gamma,
            padded,
            cache,
            residual,
            residual_bias,
            residual_output,
        )
    input.copy_(torch.randn_like(input) * 0.1)
    graph.replay()
    graph_expected = _norm_silu_reference(
        _conv_reference(input, weight),
        bias,
        gamma,
        previous,
        residual,
        residual_bias,
    )
    torch.testing.assert_close(padded, graph_expected[0], atol=0.02, rtol=0.02)

    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    wrapped = conv3d_rmsnorm_silu_pad_wrapper_sm100(
        input,
        packed_weight,
        bias,
        gamma,
        history_frames,
        residual,
        residual_bias,
        current_stream=cuda.CUstream(side_stream.cuda_stream),
    )
    torch.cuda.current_stream().wait_stream(side_stream)
    copy_history(wrapped["padded_output"], wrapped["cache_output"])
    torch.testing.assert_close(wrapped["padded_output"], graph_expected[0], atol=0.02, rtol=0.02)
    torch.testing.assert_close(wrapped["cache_output"], graph_expected[1], atol=0.02, rtol=0.02)
    torch.testing.assert_close(wrapped["residual_output"], graph_expected[2], atol=0.02, rtol=0.02)


@pytest.mark.L0
@requires_cutedsl
@requires_gpu
@torch.inference_mode()
def test_conv3d_bias_residual_pad_class_and_wrapper():
    """Check bias/residual fusion and zero-filled bottom/right padding."""
    from cudnn import Conv3dBiasResidualPadSm100, conv3d_bias_residual_pad_wrapper_sm100

    torch.manual_seed(11)
    input, weight, packed_weight, output_shape = _make_inputs(160, 320)
    bias = torch.randn(320, device="cuda", dtype=torch.bfloat16) * 0.1
    residual = torch.randn(output_shape, device="cuda", dtype=torch.bfloat16)
    residual_bias = torch.randn(320, device="cuda", dtype=torch.bfloat16) * 0.1
    padded = torch.empty((1, 2, 5, 6, 320), device="cuda", dtype=torch.bfloat16)

    plan = Conv3dBiasResidualPadSm100(input, packed_weight, bias, residual, padded, residual_bias)
    assert plan.check_support()
    plan.compile()
    plan.execute(input, packed_weight, bias, residual, padded, residual_bias)

    expected = _spatial_reference(_conv_reference(input, weight), bias, residual, residual_bias)
    torch.testing.assert_close(padded, expected, atol=0.02, rtol=0.02)

    wrapped = conv3d_bias_residual_pad_wrapper_sm100(input, packed_weight, bias, residual, residual_bias)
    torch.testing.assert_close(wrapped["padded_output"], expected, atol=0.02, rtol=0.02)


@pytest.mark.L1
@requires_cutedsl
@requires_gpu
@torch.inference_mode()
@pytest.mark.parametrize(
    "variant,input_channels,output_channels",
    (
        ("raw", 160, 160),
        ("raw", 160, 320),
        ("raw", 320, 320),
        ("raw", 640, 640),
        ("norm_silu_contiguous", 160, 160),
        ("norm_silu_contiguous", 160, 320),
        ("norm_silu_contiguous", 320, 320),
        ("norm_silu_pad", 160, 160),
        ("norm_silu_pad", 160, 320),
        ("norm_silu_pad", 320, 320),
        ("bias_residual_pad", 160, 160),
        ("bias_residual_pad", 160, 320),
        ("bias_residual_pad", 320, 320),
        ("bias_residual_pad", 320, 640),
        ("bias_residual_pad", 640, 640),
    ),
)
def test_supported_channel_pairs(variant, input_channels, output_channels):
    """Compare each supported convolution channel pair against the Torch reference."""
    from cudnn import (
        conv3d_bias_residual_pad_wrapper_sm100,
        conv3d_raw_wrapper_sm100,
        conv3d_rmsnorm_silu_pad_wrapper_sm100,
        conv3d_rmsnorm_silu_wrapper_sm100,
        pack_conv3d_weight_sm100,
    )

    torch.manual_seed(input_channels + output_channels)
    input = torch.randn((1, 3, 5, 5, input_channels), device="cuda", dtype=torch.bfloat16) * 0.1
    weight = (
        torch.randn(
            (output_channels, input_channels, 3, 3, 3),
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    packed_weight = pack_conv3d_weight_sm100(weight)
    bias = torch.randn(output_channels, device="cuda", dtype=torch.bfloat16) * 0.1
    conv = _conv_reference(input, weight)

    if variant == "raw":
        actual = conv3d_raw_wrapper_sm100(input, packed_weight)
        torch.testing.assert_close(actual, conv, atol=0.02, rtol=0.02)
    elif variant in ("norm_silu_pad", "norm_silu_contiguous"):
        gamma = torch.randn(output_channels, device="cuda", dtype=torch.bfloat16)
        if variant == "norm_silu_contiguous":
            expected = _norm_silu_output_reference(conv, bias, gamma, None, None)
            actual = conv3d_rmsnorm_silu_wrapper_sm100(input, packed_weight, bias, gamma)
            torch.testing.assert_close(actual["output"], expected[0], atol=0.02, rtol=0.02)
        else:
            expected = _norm_silu_reference(conv, bias, gamma, None, None, None)
            actual = conv3d_rmsnorm_silu_pad_wrapper_sm100(input, packed_weight, bias, gamma)
            torch.testing.assert_close(actual["padded_output"], expected[0], atol=0.02, rtol=0.02)
            torch.testing.assert_close(actual["cache_output"], expected[1], atol=0.02, rtol=0.02)
    else:
        residual = torch.randn_like(conv)
        expected = _spatial_reference(conv, bias, residual, None)
        actual = conv3d_bias_residual_pad_wrapper_sm100(input, packed_weight, bias, residual)
        torch.testing.assert_close(actual["padded_output"], expected, atol=0.02, rtol=0.02)


@requires_cutedsl
@requires_gpu
@pytest.mark.L1
@pytest.mark.parametrize(
    "ci,co,dtype",
    [
        (96, 96, torch.bfloat16),  # N96, partial K64.
        (96, 192, torch.float32),  # N192, aligned K32.
        (16, 384, torch.bfloat16),  # Input smaller than one K tile.
        (192, 384, torch.float32),  # Multiple N192 tiles.
        (512, 256, torch.bfloat16),  # N256.
        (48, 1024, torch.float32),  # Partial K32 and multiple N256 tiles.
        (160, 320, torch.float32),  # Multiple N160 tiles.
        (640, 640, torch.float32),  # Wide reduction.
    ],
)
@torch.inference_mode()
def test_extended_raw_channels(ci, co, dtype):
    """Sample tile widths, K padding, and multi-tile outputs across BF16/TF32."""
    from cudnn import Conv3dRawSm100, conv3d_raw_wrapper_sm100, pack_conv3d_weight_sm100

    torch.manual_seed(ci + co)
    x = torch.randn((2, 3, 7, 9, ci), device="cuda", dtype=dtype) * 0.1
    weight = torch.randn((co, ci, 3, 3, 3), device="cuda", dtype=dtype) * (27 * ci) ** -0.5
    tf32 = dtype == torch.float32
    packed = pack_conv3d_weight_sm100(weight, allow_tf32=tf32)
    output = torch.empty((2, 1, 5, 7, co), device="cuda", dtype=dtype)
    workspace = torch.empty_like(x) if tf32 else None
    plan = Conv3dRawSm100(x, packed, output, allow_tf32=tf32, sample_workspace=workspace)
    plan.check_support()
    plan.compile()
    execute = partial(plan.execute, x, packed, output, workspace=workspace)
    _assert_warm_execute_contract(execute)
    if tf32:
        # Isolate accumulation accuracy from operand rounding. The packed weight
        # and declared input workspace contain the exact TF32 operands.
        reference_weight = packed[..., :ci].permute(0, 4, 1, 2, 3).contiguous()
        with torch.backends.cudnn.flags(allow_tf32=False):
            expected = _conv_reference(workspace, reference_weight)
        torch.testing.assert_close(output, expected, atol=3e-5, rtol=3e-4)
    else:
        expected = _conv_reference(x, weight)
        torch.testing.assert_close(output, expected, atol=0.002, rtol=0.02)
    wrapped = conv3d_raw_wrapper_sm100(x, packed, allow_tf32=tf32)
    torch.testing.assert_close(wrapped, output, atol=0, rtol=0)


@requires_cutedsl
@requires_gpu
@pytest.mark.L1
@pytest.mark.parametrize(
    "channels,dtype",
    [(96, torch.float32), (192, torch.bfloat16), (256, torch.float32), (384, torch.bfloat16), (1024, torch.float32)],
)
@torch.inference_mode()
def test_extended_standalone_postops(channels, dtype):
    """Sample scalar, subwarp, and full-warp reductions with dynamic history."""
    from cudnn import RmsNormSiluPadSm100

    torch.manual_seed(channels)
    plans = []
    for n, t, h, w, history in ((1, 1, 3, 5, 1), (2, 3, 5, 7, 2)):
        x = torch.randn((n, t, h, w, channels), device="cuda", dtype=dtype)
        gamma = torch.randn(channels, device="cuda", dtype=dtype)
        bias = torch.randn_like(gamma) * 0.1
        skip = torch.randn_like(x)
        skip_bias = torch.randn_like(gamma) * 0.1
        previous = torch.randn((n, history, h, w, channels), device="cuda", dtype=dtype)
        padded = torch.empty((n, t + 2, h + 2, w + 2, channels), device="cuda", dtype=dtype)
        cache = torch.empty((n, min(2, t + history), h, w, channels), device="cuda", dtype=dtype)
        saved = torch.empty_like(x)
        plan = RmsNormSiluPadSm100(x, gamma, padded, cache, bias, previous, skip, skip_bias, saved)
        plan.check_support()
        plan.compile()
        execute = partial(plan.execute, x, gamma, padded, cache, bias, previous, skip, skip_bias, saved)
        _assert_warm_execute_contract(execute)
        value = ((x.float() + bias.float()).to(dtype).float() + (skip.float() + skip_bias.float()).to(dtype).float()).to(dtype)
        norm = F.normalize(value.float(), dim=-1).to(dtype)
        affine = ((norm.float() * math.sqrt(channels)).to(dtype).float() * gamma.float()).to(dtype)
        activated = F.silu(affine.float()).to(dtype)
        joined = torch.cat((previous, activated), dim=1)
        expected = F.pad(joined, (0, 0, 1, 1, 1, 1, 2 - history, 0))
        tol = 0.02 if dtype == torch.bfloat16 else 3e-6
        torch.testing.assert_close(padded, expected, atol=tol, rtol=tol)
        torch.testing.assert_close(cache, joined[:, -2:], atol=tol, rtol=tol)
        torch.testing.assert_close(saved, value, atol=0, rtol=0)
        plans.append(plan)
    assert plans[0]._compiled_kernel is plans[1]._compiled_kernel


@requires_cutedsl
@requires_gpu
@pytest.mark.L1
@pytest.mark.parametrize(
    "channels,dtype,padded,use_residual",
    [
        (96, torch.bfloat16, False, False),  # Eight-warp epilogue.
        (96, torch.float32, True, True),  # Residual selects sixteen warps.
        (160, torch.float32, True, False),  # Register budget for TF32 C160.
        (192, torch.bfloat16, True, True),  # BF16 register redistribution.
        (192, torch.float32, False, False),
        (256, torch.bfloat16, True, True),  # N256 reduction.
        (320, torch.bfloat16, True, False),  # Two N160 tiles per reduction.
        (320, torch.float32, False, True),
    ],
)
@torch.inference_mode()
def test_extended_fused_norm(channels, dtype, padded, use_residual):
    """Reuse geometry/history specializations, preserving caller-owned regions on replay."""
    from cudnn import Conv3dRmsNormSiluPadSm100, Conv3dRmsNormSiluSm100, pack_conv3d_weight_sm100

    torch.manual_seed(channels)
    tf32 = dtype == torch.float32
    weight = torch.randn((channels, channels, 3, 3, 3), device="cuda", dtype=dtype) * (27 * channels) ** -0.5
    packed = pack_conv3d_weight_sm100(weight, allow_tf32=tf32)
    plans = []
    for n, t, h, w, history in ((1, 1, 3, 5, 0), (2, 1, 7, 9, 2), (1, 3, 5, 7, 1)):
        x = torch.randn((n, t + 2, h + 2, w + 2, channels), device="cuda", dtype=dtype) * 0.1
        bias = torch.randn(channels, device="cuda", dtype=dtype) * 0.1
        gamma = torch.randn_like(bias)
        skip = torch.randn((n, t, h, w, channels), device="cuda", dtype=dtype) * 0.1
        skip_bias = torch.randn_like(bias) * 0.1
        saved = torch.empty_like(skip) if use_residual else None
        workspace = torch.empty_like(x) if tf32 else None
        output = torch.empty((n, t + 2, h + 2, w + 2, channels), device="cuda", dtype=dtype) if padded else torch.empty_like(skip)
        cache = torch.empty((n, min(2, t + history), h, w, channels), device="cuda", dtype=dtype) if padded else None
        if not use_residual:
            skip = skip_bias = None
        if padded:
            plan = Conv3dRmsNormSiluPadSm100(
                x, packed, bias, gamma, output, cache, history, skip, skip_bias, saved, allow_tf32=tf32, sample_workspace=workspace
            )
            execute = partial(plan.execute, x, packed, bias, gamma, output, cache, skip, skip_bias, saved, workspace=workspace)
        else:
            plan = Conv3dRmsNormSiluSm100(x, packed, bias, gamma, output, skip, skip_bias, saved, allow_tf32=tf32, sample_workspace=workspace)
            execute = partial(plan.execute, x, packed, bias, gamma, output, skip, skip_bias, saved, workspace=workspace)
        plan.check_support()
        plan.compile()
        if padded:
            output.fill_(float("nan"))
            cache.fill_(float("nan"))
        _assert_warm_execute_contract(execute)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                execute()
            x.mul_(0.75)
            graph.replay()
            with torch.backends.cudnn.flags(allow_tf32=False):
                conv = _conv_reference(workspace, packed[..., :channels].permute(0, 4, 1, 2, 3).contiguous()) if tf32 else _conv_reference(x, weight)
            value = (conv.float() + bias.float()).to(dtype)
            if use_residual:
                value = (value.float() + (skip.float() + skip_bias.float()).to(dtype).float()).to(dtype)
            norm = F.normalize(value.float(), dim=-1).to(dtype)
            affine = ((norm.float() * math.sqrt(channels)).to(dtype).float() * gamma.float()).to(dtype)
            expected = F.silu(affine.float()).to(dtype)
            tol = 0.04 if dtype == torch.bfloat16 else 2e-4
            current = output[:, 2:, 1:-1, 1:-1] if padded else output
            torch.testing.assert_close(current, expected, atol=tol, rtol=tol)
            if use_residual:
                torch.testing.assert_close(saved, value, atol=tol, rtol=tol)
            if padded:
                torch.testing.assert_close(cache[:, -min(2, t) :], expected[:, -2:], atol=tol, rtol=tol)
                assert torch.isnan(output[:, 2 - history : 2, 1:-1, 1:-1]).all()
                if t == 1 and history:
                    assert torch.isnan(cache[:, :1]).all()
                assert torch.count_nonzero(output[:, :, 0]) == 0
                assert torch.count_nonzero(output[:, :, -1]) == 0
                assert torch.count_nonzero(output[:, :, :, 0]) == 0
                assert torch.count_nonzero(output[:, :, :, -1]) == 0
        finally:
            graph.reset()
        plans.append(plan)
    assert all(plan._compiled_kernel is plans[0]._compiled_kernel for plan in plans)


@requires_cutedsl
@requires_gpu
@pytest.mark.L1
@torch.inference_mode()
def test_tf32_causal_dynamic_packing():
    """Reuse packing across sliced strides/history, retaining unrounded FP32 cache."""
    from cudnn import (
        CausalConv3dWithCacheSm100,
        causal_conv3d_with_cache_wrapper_sm100,
        pack_causal_conv3d_weight_sm100,
    )

    torch.manual_seed(17)
    weight = torch.randn((160, 12, 3, 3, 3), device="cuda") * 0.03
    packed = pack_causal_conv3d_weight_sm100(weight, allow_tf32=True)
    plans = []
    for n, t, h, w, history in ((1, 1, 4, 5, 1), (2, 4, 7, 9, 2)):
        video = torch.randn((n, 12, t + 2, h, w * 2), device="cuda") * 0.1
        x = video[:, :, 1 : t + 1, :, ::2]
        previous = torch.randn((n, history, h, w, 12), device="cuda") * 0.1
        padded = torch.empty((n, t + 2, h + 2, w + 2, 16), device="cuda")
        cache = torch.empty((n, 2, h, w, 12), device="cuda")
        output = torch.empty((n, t, h, w, 160), device="cuda")
        plan = CausalConv3dWithCacheSm100(x, packed, padded, cache, output, previous, allow_tf32=True)
        plan.check_support()
        plan.compile()
        execute = partial(plan.execute, x, packed, padded, cache, output, previous)
        _assert_warm_execute_contract(execute)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                execute()
            for scale in (0.75, 1.25):
                x.mul_(scale)
                previous.mul_(scale)
                graph.replay()
                joined = torch.cat((previous, x.permute(0, 2, 3, 4, 1)), dim=1)
                torch.testing.assert_close(cache, joined[:, -2:], atol=0, rtol=0)
                # Input rounding is exactly a binary32 -> TF32 RN-even conversion.
                expected_padded = F.pad(joined, (0, 4, 1, 1, 1, 1, 2 - history, 0))
                bits = expected_padded.view(torch.int32)
                expected_padded = ((bits + 4095 + ((bits >> 13) & 1)) & -8192).view(torch.float32)
                torch.testing.assert_close(padded, expected_padded, atol=0, rtol=0)
                with torch.backends.cudnn.flags(allow_tf32=False):
                    expected = _conv_reference(padded, packed[..., :16].permute(0, 4, 1, 2, 3).contiguous())
                torch.testing.assert_close(output, expected, atol=2e-5, rtol=3e-4)
        finally:
            graph.reset()
        wrapped = causal_conv3d_with_cache_wrapper_sm100(x, packed, previous, allow_tf32=True)
        torch.testing.assert_close(wrapped["output"], output, atol=0, rtol=0)
        plans.append(plan)
    assert plans[0]._compiled_pack is plans[1]._compiled_pack
    assert plans[0]._compiled_kernel is plans[1]._compiled_kernel


@pytest.mark.L0
def test_tf32_weight_rounding():
    """Verify RN-even ties and K32 padding without requiring a GPU."""
    from cudnn import pack_conv3d_weight_sm100

    bits = torch.tensor([0x3F801000, 0x3F803000, -1082126336, 0x7F800000, 0x7FC00000], dtype=torch.int32)
    values = bits.view(torch.float32)
    weight = values.repeat((96 * 96 * 27 + 4) // 5)[: 96 * 96 * 27].reshape(96, 96, 3, 3, 3)
    with pytest.raises(ValueError, match="allow_tf32"):
        pack_conv3d_weight_sm100(weight)
    packed = pack_conv3d_weight_sm100(weight, allow_tf32=True)
    assert packed.shape == (96, 3, 3, 3, 96)
    expected_bits = torch.tensor([0x3F800000, 0x3F804000, -1082130432, 0x7F800000, 0x7FC00000], dtype=torch.int32)
    actual = packed.permute(0, 4, 1, 2, 3).contiguous().reshape(-1)[:5].view(torch.int32)
    assert actual.tolist() == expected_bits.tolist()


@requires_cutedsl
@requires_gpu
@pytest.mark.L1
@torch.inference_mode()
def test_fp32_permission_workspace_and_wide_coordinates():
    """Require explicit TF32/workspace and exercise the wide coordinate launch."""
    from cudnn import Conv3dRawSm100, pack_conv3d_weight_sm100
    from cudnn.conv.cutedsl.conv3d_postops._compile import compile_conv
    from cudnn.conv.cutedsl.conv3d_postops._config import ConvSpecialization

    x = torch.randn((1, 3, 5, 7, 96), device="cuda") * 0.1
    weight = pack_conv3d_weight_sm100(torch.randn((96, 96, 3, 3, 3), device="cuda") * 0.03, allow_tf32=True)
    output = torch.empty((1, 1, 3, 5, 96), device="cuda")
    workspace = torch.empty_like(x)
    with pytest.raises(ValueError, match="allow_tf32"):
        Conv3dRawSm100(x, weight, output, sample_workspace=workspace).check_support()
    with pytest.raises(ValueError, match="workspace is required"):
        Conv3dRawSm100(x, weight, output, allow_tf32=True).check_support()
    plan = Conv3dRawSm100(x, weight, output, allow_tf32=True, sample_workspace=workspace)
    plan.check_support()
    plan.compile()
    plan.execute(x, weight, output, workspace=workspace)
    expected = output.clone()
    spec = ConvSpecialization(96, 96, plan.config.ctas, "tf32")
    plan._compiled_kernel = compile_conv(plan.descs["input"].device, spec, "raw", False, False, True)
    plan.execute(x, weight, output, workspace=workspace)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="workspace must not overlap input"):
        Conv3dRawSm100(x, weight, output, allow_tf32=True, sample_workspace=x).check_support()


@requires_cutedsl
@requires_gpu
@pytest.mark.L1
@pytest.mark.parametrize("kind", ["raw_tf32", "norm_fp32", "causal_bf16"])
@torch.inference_mode()
def test_replan_reloads_artifact(kind, tmp_path, monkeypatch):
    """Reload each launch topology from disk with JIT forbidden, then replay changed input."""
    from cudnn import (
        CausalConv3dWithCacheSm100,
        Conv3dRawSm100,
        RmsNormSiluPadSm100,
        pack_causal_conv3d_weight_sm100,
        pack_conv3d_weight_sm100,
    )
    from cudnn.frost import compiled_cache
    from cutlass import cute

    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE", str(tmp_path))
    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO", "0")
    dtype = torch.bfloat16 if kind.endswith("bf16") else torch.float32
    x = torch.randn((1, 12, 1, 3, 5) if kind == "causal_bf16" else (1, 3, 5, 7, 96), device="cuda", dtype=dtype) * 0.1
    gamma = torch.ones(96, device="cuda", dtype=dtype)
    if kind.startswith("raw"):
        weight = pack_conv3d_weight_sm100(torch.randn((96, 96, 3, 3, 3), device="cuda", dtype=dtype) * 0.03, allow_tf32=kind == "raw_tf32")
    elif kind == "causal_bf16":
        weight = pack_causal_conv3d_weight_sm100(torch.randn((160, 12, 3, 3, 3), device="cuda", dtype=dtype) * 0.03)

    def build():
        if kind.startswith("raw"):
            out = torch.empty((1, 1, 3, 5, 96), device="cuda", dtype=dtype)
            workspace = torch.empty_like(x) if kind == "raw_tf32" else None
            plan = Conv3dRawSm100(x, weight, out, allow_tf32=kind == "raw_tf32", sample_workspace=workspace)
            call = partial(plan.execute, x, weight, out, workspace=workspace)
        elif kind == "norm_fp32":
            out = torch.empty((1, 5, 7, 9, 96), device="cuda", dtype=dtype)
            cache = torch.empty((1, 2, 5, 7, 96), device="cuda", dtype=dtype)
            plan = RmsNormSiluPadSm100(x, gamma, out, cache)
            call = partial(plan.execute, x, gamma, out, cache)
        else:
            out = torch.empty((1, 1, 3, 5, 160), device="cuda", dtype=dtype)
            padded = torch.empty((1, 3, 5, 7, 16), device="cuda", dtype=dtype)
            cache = torch.empty((1, 1, 3, 5, 12), device="cuda", dtype=dtype)
            plan = CausalConv3dWithCacheSm100(x, weight, padded, cache, out)
            call = partial(plan.execute, x, weight, padded, cache, out)
        plan.check_support()
        plan.compile()
        return call, out

    first, expected = build()
    first()
    counts = compiled_cache.stats()

    def forbidden(*args, **kwargs):
        raise AssertionError("replanning invoked JIT instead of reloading the artifact")

    monkeypatch.setattr(cute, "compile", forbidden)
    second, actual = build()
    assert compiled_cache.stats()["hits"] > counts["hits"]
    assert compiled_cache.stats()["misses"] == counts["misses"]
    second()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            second()
        x.mul_(0.5)
        actual.fill_(float("nan"))
        graph.replay()
        first()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    finally:
        graph.reset()


@pytest.mark.L0
@requires_cutedsl
@requires_gpu
def test_wrapper_cache_rebinds_inputs():
    """Cached plans bind new compatible storage and observe updates during graph replay."""
    from cudnn import rmsnorm_silu_pad_wrapper_sm100

    shape = (1, 2, 4, 4, 96)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    gamma = torch.ones(96, device="cuda", dtype=x.dtype)
    rmsnorm_silu_pad_wrapper_sm100(x, gamma, save_input=True)
    replacement = torch.randn_like(x)
    result = rmsnorm_silu_pad_wrapper_sm100(replacement, gamma, save_input=True)
    torch.testing.assert_close(result["residual_output"], replacement, atol=0, rtol=0)

    with pytest.raises(ValueError, match="contiguous"):
        rmsnorm_silu_pad_wrapper_sm100(x.transpose(2, 3), gamma, save_input=True)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        captured = rmsnorm_silu_pad_wrapper_sm100(x, gamma, save_input=True)
    x.copy_(replacement)
    graph.replay()
    torch.testing.assert_close(captured["residual_output"], replacement, atol=0, rtol=0)
    expected = _norm_silu_reference(x, torch.zeros_like(gamma), gamma, None, None, None)
    torch.testing.assert_close(captured["padded_output"], expected[0], atol=0.02, rtol=0.02)


@pytest.mark.L0
@requires_cutedsl
@requires_gpu
@pytest.mark.parametrize("explicit_stream", (False, True))
@torch.inference_mode()
def test_wrapper_stream_lifetimes(explicit_stream):
    """Keep foreign-stream inputs alive; allocate owned outputs on the launch stream."""
    from cudnn import rmsnorm_silu_pad_wrapper_sm100

    shape = (1, 2, 4, 4, 96)
    x = torch.full(shape, 2.0, device="cuda", dtype=torch.bfloat16)
    gamma = torch.ones(96, device="cuda", dtype=x.dtype)
    rmsnorm_silu_pad_wrapper_sm100(x, gamma, save_input=True)
    consumer = torch.cuda.Stream()
    consumer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(consumer):
        torch.cuda._sleep(20_000_000)
    if explicit_stream:
        result = rmsnorm_silu_pad_wrapper_sm100(x, gamma, save_input=True, current_stream=cuda.CUstream(consumer.cuda_stream))
    else:
        with torch.cuda.stream(consumer):
            result = rmsnorm_silu_pad_wrapper_sm100(x, gamma, save_input=True)
    with torch.cuda.stream(consumer):
        saved = result["residual_output"].clone()
    del x, gamma, result
    # Reuse allocations on the producer stream while the consumer is still busy.
    churn = [torch.full(shape, -100.0, device="cuda", dtype=torch.bfloat16) for _ in range(16)]
    torch.cuda.current_stream().wait_stream(consumer)
    torch.testing.assert_close(saved, torch.full_like(saved, 2.0), atol=0, rtol=0)
    del churn
