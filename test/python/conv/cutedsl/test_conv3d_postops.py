# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for the fused Conv3D video-VAE APIs."""

import math

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
    invalid_previous = torch.empty((2, 3, height, width, 12), device="cuda", dtype=input.dtype)
    invalid = CausalConv3dWithCacheSm100(input, packed_weight, padded_input, cache_output, output, invalid_previous)
    with pytest.raises(ValueError, match="previous must contain one or two frames"):
        invalid.check_support()


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
    expected = {"channels": "160 or 320 output channels", "layout": "input must be contiguous", "alignment": "16-byte aligned", "history": "history_frames"}
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
    """Check raw convolution, runtime validation, and reuse of the wrapper's compiled plan."""
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
    wrapped = conv3d_raw_wrapper_sm100(input, packed_weight)
    torch.testing.assert_close(wrapped, expected, atol=0.02, rtol=0.02)

    from cutlass import cute

    with monkeypatch.context() as patch:
        patch.setattr(cute, "compile", lambda *args, **kwargs: pytest.fail("warm wrapper recompiled"))
        wrapped = conv3d_raw_wrapper_sm100(input, packed_weight)
    torch.testing.assert_close(wrapped, expected, atol=0.02, rtol=0.02)

    with pytest.raises(ValueError, match="output shape mismatch"):
        plan.execute(input, packed_weight, output[..., :-1])

    aliased_output = packed_weight.view(-1)[: output.numel()].view_as(output) if output.numel() <= packed_weight.numel() else None
    if aliased_output is not None:
        with pytest.raises(ValueError, match="must not overlap packed_weight"):
            plan.execute(input, packed_weight, aliased_output)
    with torch.inference_mode(False), torch.enable_grad():
        grad_input = input.clone().requires_grad_()
        with pytest.raises(RuntimeError, match="inference-only"):
            plan.execute(grad_input, packed_weight, output)


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
    """Check fused padding outputs, preserved history, runtime guards, and graph replay."""
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
    with pytest.raises(ValueError, match="residual presence"):
        plan.execute(input, packed_weight, bias, gamma, padded, cache, residual_bias=residual_bias, residual_output=residual_output)

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
