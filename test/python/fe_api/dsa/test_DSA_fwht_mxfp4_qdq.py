# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the DeepSeek-V4 normalized H128 plus MXFP4 QDQ operation."""

from importlib import import_module

import pytest
import torch


def _require_supported_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] not in (10, 11, 12):
        pytest.skip("the current inline E2M1 conversion path requires an SM100-, SM110-, or SM120-family GPU")


def _require_operation():
    """Return the operation only when all lazy optional dependencies exist."""

    _require_supported_device()
    try:
        import_module("cudnn.ops._fwht_mxfp4_qdq_cutedsl")
        from cudnn.ops import fwht_mxfp4_qdq
    except ImportError as error:
        pytest.skip(f"CuTe DSL optional dependencies are unavailable: {error}")
    return fwht_mxfp4_qdq


def _pow2_ceil_positive(values: torch.Tensor) -> torch.Tensor:
    bits = values.contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) - 127
    mantissa = bits & 0x7FFFFF
    ceil_exponent = exponent + (mantissa != 0).to(torch.int32)
    return ((ceil_exponent + 127) << 23).contiguous().view(torch.float32)


def _e2m1_rne(values: torch.Tensor) -> torch.Tensor:
    sign = torch.where(torch.signbit(values), -1.0, 1.0)
    magnitude = values.abs()
    quantized = torch.where(
        magnitude <= 0.25,
        0.0,
        torch.where(
            magnitude < 0.75,
            0.5,
            torch.where(
                magnitude <= 1.25,
                1.0,
                torch.where(
                    magnitude < 1.75,
                    1.5,
                    torch.where(
                        magnitude <= 2.5,
                        2.0,
                        torch.where(
                            magnitude < 3.5,
                            3.0,
                            torch.where(magnitude <= 5.0, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    return quantized * sign


def _qdq_reference(rotated_bf16: torch.Tensor) -> torch.Tensor:
    """Official FP32 scale/divide path starting at the rounded BF16 boundary."""

    rows = rotated_bf16.numel() // 128
    groups = rotated_bf16.view(rows, 4, 32).float()
    amax = groups.abs().amax(dim=-1).clamp_min(6.0 * (2.0**-126))
    scale = _pow2_ceil_positive(amax * (1.0 / 6.0))
    normalized = (groups / scale.unsqueeze(-1)).clamp(-6.0, 6.0)
    output = _e2m1_rne(normalized) * scale.unsqueeze(-1)
    return output.reshape(rotated_bf16.shape).to(torch.bfloat16)


def _packed_bf16_qdq_reference(rotated_bf16: torch.Tensor) -> torch.Tensor:
    """Host model of the device kernel's packed power-of-two QDQ."""

    rows = rotated_bf16.numel() // 128
    groups = rotated_bf16.view(rows, 4, 32)
    amax = groups.float().abs().amax(dim=-1).clamp_min(6.0 * (2.0**-126))
    scale = _pow2_ceil_positive(amax * (1.0 / 6.0))
    inverse_scale_bf16 = scale.reciprocal().to(torch.bfloat16)
    normalized_bf16 = (groups * inverse_scale_bf16.unsqueeze(-1)).to(torch.bfloat16)
    quantized_bf16 = _e2m1_rne(normalized_bf16.float().clamp(-6.0, 6.0)).to(torch.bfloat16)
    output = quantized_bf16 * scale.to(torch.bfloat16).unsqueeze(-1)
    return output.reshape(rotated_bf16.shape).to(torch.bfloat16)


def _fwht_fp32(input_tensor: torch.Tensor) -> torch.Tensor:
    rows = input_tensor.numel() // 128
    transformed = input_tensor.view(rows, 128).float()
    for half_width in (1, 2, 4, 8, 16, 32, 64):
        pairs = transformed.reshape(rows, -1, 2, half_width)
        low = pairs[:, :, 0, :]
        high = pairs[:, :, 1, :]
        transformed = torch.cat((low + high, low - high), dim=-1).reshape(rows, 128)
    return transformed


def _reference(input_tensor: torch.Tensor) -> torch.Tensor:
    transformed = _fwht_fp32(input_tensor)
    transformed = (transformed * (128.0**-0.5)).to(torch.bfloat16)
    return _qdq_reference(transformed).reshape(input_tensor.shape)


def _legacy_combined_round_reference(input_tensor: torch.Tensor) -> torch.Tensor:
    """Host model of the earlier combined-rounding schedule for regression."""

    rows = input_tensor.numel() // 128
    groups = _fwht_fp32(input_tensor).view(rows, 4, 32)
    norm = 128.0**-0.5
    amax = (groups.abs().amax(dim=-1) * norm).to(torch.bfloat16).float()
    amax = amax.clamp_min(6.0 * (2.0**-126))
    scale = _pow2_ceil_positive(amax * (1.0 / 6.0))
    normalized_bf16 = (groups * (norm / scale.unsqueeze(-1))).to(torch.bfloat16)
    output = _e2m1_rne(normalized_bf16.float().clamp(-6.0, 6.0)) * scale.unsqueeze(-1)
    return output.reshape(input_tensor.shape).to(torch.bfloat16)


@pytest.mark.L0
def test_reference_e2m1_rne_ties_and_neighbors_on_host():
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
    below = torch.nextafter(boundaries, torch.zeros_like(boundaries))
    above = torch.nextafter(boundaries, torch.full_like(boundaries, float("inf")))

    assert torch.equal(_e2m1_rne(below), torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]))
    assert torch.equal(_e2m1_rne(boundaries), torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]))
    assert torch.equal(_e2m1_rne(above), torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]))
    assert torch.equal(_e2m1_rne(-boundaries), -torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]))

    signed_zero = torch.tensor([-32768, 0], dtype=torch.int16).view(torch.bfloat16).float()
    quantized_zero = _e2m1_rne(signed_zero).to(torch.bfloat16)
    assert torch.equal(quantized_zero.view(torch.int16), torch.tensor([-32768, 0], dtype=torch.int16))


@pytest.mark.L0
def test_reference_ue8m0_power_of_two_boundaries_on_host():
    exponents = torch.tensor([-125, -64, -1, 0, 1, 64, 125], dtype=torch.int32)
    exact = torch.ldexp(torch.ones(exponents.shape, dtype=torch.float32), exponents)
    below = torch.nextafter(exact, torch.zeros_like(exact))
    above = torch.nextafter(exact, torch.full_like(exact, float("inf")))

    assert torch.equal(_pow2_ceil_positive(below), exact)
    assert torch.equal(_pow2_ceil_positive(exact), exact)
    assert torch.equal(_pow2_ceil_positive(above), exact * 2.0)

    minimum_normal = torch.ldexp(torch.ones(1, dtype=torch.float32), torch.tensor([-126]))
    floor_ratio = torch.tensor([6.0 * (2.0**-126)], dtype=torch.float32) * torch.tensor([1.0 / 6.0], dtype=torch.float32)
    assert torch.equal(floor_ratio, minimum_normal)
    assert torch.equal(_pow2_ceil_positive(floor_ratio), minimum_normal)

    maximum_bf16 = torch.tensor([0x7F7F], dtype=torch.int16).view(torch.bfloat16).float()
    maximum_scale = torch.ldexp(torch.ones(1, dtype=torch.float32), torch.tensor([126]))
    assert torch.equal(_pow2_ceil_positive(maximum_bf16 * (1.0 / 6.0)), maximum_scale)


@pytest.mark.L1
def test_packed_bf16_qdq_matches_reference_for_every_finite_bf16_on_host():
    positive_bits = torch.arange(0x0000, 0x7F80, dtype=torch.int32)
    finite_bits = torch.cat((positive_bits, positive_bits | 0x8000))
    bit_layouts = (
        finite_bits,
        finite_bits.reshape(128, -1).T.contiguous().reshape(-1),
    )

    for bit_layout in bit_layouts:
        rotated_bf16 = bit_layout.to(torch.int16).view(torch.bfloat16).reshape(-1, 128)
        expected = _qdq_reference(rotated_bf16)
        actual = _packed_bf16_qdq_reference(rotated_bf16)

        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.L1
def test_packed_scale_bytes_match_reference_for_every_positive_finite_bf16_on_host():
    positive_bits = torch.arange(0x0000, 0x7F80, dtype=torch.int32)
    clamped_bits = torch.maximum(positive_bits, torch.tensor(0x01C0, dtype=torch.int32))

    # The optimized kernel carries two group maxima in one packed word. Pair
    # ascending and descending values so both halfwords exhaust the BF16 range.
    packed_amax = clamped_bits.to(torch.int64) | (clamped_bits.flip(0).to(torch.int64) << 16)
    biased_amax = packed_amax - 0x00C100C1
    actual_low = (biased_amax & 0xFFFF) >> 7
    actual_high = biased_amax >> 23

    amax = positive_bits.to(torch.int16).view(torch.bfloat16).float()
    floor = torch.tensor(6.0 * (2.0**-126), dtype=torch.float32)
    scale = _pow2_ceil_positive(torch.maximum(amax, floor) * torch.tensor(1.0 / 6.0, dtype=torch.float32))
    expected = (scale.contiguous().view(torch.int32).to(torch.int64) >> 23) & 0xFF

    assert torch.equal(actual_low, expected)
    assert torch.equal(actual_high, expected.flip(0))


@pytest.mark.L0
@pytest.mark.parametrize("rows", [15, 16, 17])
def test_reference_is_row_local_at_current_cta_boundary_on_host(rows):
    torch.manual_seed(20260829 + rows)
    input_tensor = torch.randn((rows, 128), dtype=torch.bfloat16)

    batched = _reference(input_tensor)
    rowwise = torch.cat([_reference(input_tensor[row : row + 1]) for row in range(rows)])

    assert torch.equal(batched, rowwise)


def _subnormal_double_rounding_case():
    input_tensor = torch.zeros((1, 128), dtype=torch.bfloat16)
    input_bits = input_tensor.view(torch.int16)
    input_bits[0, 0] = 0x0191
    input_bits[0, 1] = 0x00D7

    expected = _reference(input_tensor).view(torch.int16)
    legacy = _legacy_combined_round_reference(input_tensor).view(torch.int16)

    return input_tensor, expected, legacy


@pytest.mark.L0
def test_bf16_boundary_precedes_inverse_scale_on_host():
    """Lock down the double-rounding case fixed by the baseline implementation."""

    _, expected, legacy = _subnormal_double_rounding_case()

    assert torch.equal(expected[0, 0::2], torch.full((64,), 0x0040, dtype=torch.int16))
    assert torch.equal(expected[0, 1::2], torch.zeros(64, dtype=torch.int16))
    assert torch.equal(legacy, torch.full((1, 128), 0x0040, dtype=torch.int16))


@pytest.mark.L0
def test_actual_kernel_observes_bf16_boundary_before_inverse_scale():
    fwht_mxfp4_qdq = _require_operation()

    input_tensor, expected, legacy = _subnormal_double_rounding_case()
    actual = fwht_mxfp4_qdq(input_tensor.cuda()).cpu().view(torch.int16)

    assert torch.equal(actual, expected)
    assert not torch.equal(actual, legacy)


@pytest.mark.L0
@pytest.mark.parametrize("shape", [(1, 128), (31, 128), (33, 128), (2, 3, 4, 128)])
def test_fwht_mxfp4_qdq_matches_exact_recipe(shape):
    fwht_mxfp4_qdq = _require_operation()

    torch.manual_seed(20260829)
    input_tensor = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    expected = _reference(input_tensor)
    actual = fwht_mxfp4_qdq(input_tensor)

    assert actual.shape == input_tensor.shape
    assert actual.dtype == input_tensor.dtype
    assert actual.is_contiguous()
    assert torch.equal(actual, expected)


@pytest.mark.L0
@pytest.mark.parametrize("compute_capability", [(9, 0), (13, 0)])
def test_fwht_mxfp4_qdq_rejects_unsupported_arch_before_compile(monkeypatch, compute_capability):
    _require_operation()
    from cudnn.ops import _fwht_mxfp4_qdq_cutedsl as kernel_module

    input_tensor = torch.zeros((1, 128), device="cuda", dtype=torch.bfloat16)
    output_tensor = torch.empty_like(input_tensor)
    monkeypatch.setattr(kernel_module.torch.cuda, "get_device_capability", lambda _device: compute_capability)
    monkeypatch.setattr(kernel_module.cute, "compile", lambda *_args, **_kwargs: pytest.fail("unsupported targets must not compile"))

    with pytest.raises(RuntimeError, match=r"SM100-, SM110-, or SM120-family.*compute capability"):
        kernel_module.run_fwht_mxfp4_qdq(input_tensor, output_tensor)


@pytest.mark.L0
@pytest.mark.parametrize("rows", [15, 16, 17, 63, 64, 65, 127, 128, 129])
def test_fwht_mxfp4_qdq_cta_row_boundaries(rows):
    fwht_mxfp4_qdq = _require_operation()

    torch.manual_seed(20260830 + rows)
    input_tensor = torch.randn((rows, 128), device="cuda", dtype=torch.bfloat16)
    actual = fwht_mxfp4_qdq(input_tensor)

    assert torch.equal(actual, _reference(input_tensor))


@pytest.mark.L0
@torch.no_grad()
def test_fwht_mxfp4_qdq_torch_library_contract():
    fwht_mxfp4_qdq = _require_operation()

    input_tensor = torch.randn((65, 128), device="cuda", dtype=torch.bfloat16)
    # Importing the public function performs the lazy registration.
    assert callable(fwht_mxfp4_qdq)
    primitive = torch.ops.cudnn.fwht_mxfp4_qdq_primitive.default
    test_utils = (
        "test_schema",
        "test_faketensor",
        "test_aot_dispatch_dynamic",
    )

    results = torch.library.opcheck(
        primitive,
        (input_tensor,),
        test_utils=test_utils,
    )

    assert results == {test: "SUCCESS" for test in test_utils}


@pytest.mark.L1
@torch.no_grad()
def test_fwht_mxfp4_qdq_torch_compile_fullgraph():
    fwht_mxfp4_qdq = _require_operation()

    torch.manual_seed(20260901)
    input_tensor = torch.randn((65, 128), device="cuda", dtype=torch.bfloat16)
    expected = fwht_mxfp4_qdq(input_tensor)
    compiled = torch.compile(fwht_mxfp4_qdq, fullgraph=True)
    actual = compiled(input_tensor)

    assert torch.equal(actual, expected)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@torch.no_grad()
def test_fwht_mxfp4_qdq_respects_non_default_torch_stream():
    fwht_mxfp4_qdq = _require_operation()
    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("torch.cuda._sleep is unavailable")

    torch.manual_seed(20260902)
    input_tensor = torch.zeros((65, 128), device="cuda", dtype=torch.bfloat16)
    replacement = torch.randn_like(input_tensor)
    expected = _reference(replacement)

    # Warm the compile/cache path before placing the side stream behind a gate.
    fwht_mxfp4_qdq(input_tensor)
    torch.cuda.synchronize()

    # Park the side stream before the input update. A launch incorrectly sent
    # to the default stream overtakes the update and reads the old zeros, while
    # a launch on the current side stream is ordered after the replacement.
    side_stream = torch.cuda.Stream()
    with torch.cuda.stream(side_stream):
        torch.cuda._sleep(500_000_000)
        input_tensor.copy_(replacement)
        actual = fwht_mxfp4_qdq(input_tensor)

    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


@pytest.mark.L0
@torch.no_grad()
def test_fwht_mxfp4_qdq_warmed_cuda_graph_capture_and_replay():
    fwht_mxfp4_qdq = _require_operation()

    torch.manual_seed(20260903)
    static_input = torch.zeros((65, 128), device="cuda", dtype=torch.bfloat16)
    replay_inputs = [torch.randn_like(static_input), torch.randn_like(static_input)]
    expected = [_reference(value) for value in replay_inputs]

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        fwht_mxfp4_qdq(static_input)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = fwht_mxfp4_qdq(static_input)

    for replay_input, replay_expected in zip(replay_inputs, expected):
        static_input.copy_(replay_input)
        captured_output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured_output, replay_expected)


@pytest.mark.L0
def test_fwht_mxfp4_qdq_empty_does_not_launch():
    fwht_mxfp4_qdq = _require_operation()

    input_tensor = torch.empty((0, 128), device="cuda", dtype=torch.bfloat16)
    output = fwht_mxfp4_qdq(input_tensor)
    assert output.shape == input_tensor.shape
    assert output.numel() == 0


@pytest.mark.L0
def test_fwht_mxfp4_qdq_rejects_implicit_layout_conversion():
    fwht_mxfp4_qdq = _require_operation()

    input_tensor = torch.empty((128, 7), device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    assert input_tensor.shape == (7, 128)
    assert not input_tensor.is_contiguous()
    with pytest.raises(ValueError, match="must be contiguous"):
        fwht_mxfp4_qdq(input_tensor)


@pytest.mark.L0
def test_fwht_mxfp4_qdq_rejects_misaligned_contiguous_view():
    fwht_mxfp4_qdq = _require_operation()

    storage = torch.empty((129,), device="cuda", dtype=torch.bfloat16)
    input_tensor = storage[1:].view(1, 128)
    assert input_tensor.is_contiguous()
    assert input_tensor.data_ptr() % 32 != 0
    with pytest.raises(ValueError, match="32-byte aligned"):
        fwht_mxfp4_qdq(input_tensor)


@pytest.mark.L0
def test_fwht_mxfp4_qdq_is_explicitly_inference_only():
    fwht_mxfp4_qdq = _require_operation()

    input_tensor = torch.randn((1, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with pytest.raises(NotImplementedError, match="inference-only"):
        fwht_mxfp4_qdq(input_tensor)
