# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the benchmark-private SM100 NVFP4 quantizer."""

import pytest
import torch

from cudnn.gemm.ops._nvfp4_quantize import nvfp4_quantize


def _is_sm100():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)


def _swizzle_128x4(logical, m, k):
    """Independent implementation of the documented F8_128x4 byte mapping."""
    rows = (m + 127) // 128 * 128
    cols = (k // 16 + 3) // 4 * 4
    padded = torch.zeros((rows, cols), dtype=torch.uint8, device=logical.device)
    padded[:m, : k // 16] = logical
    return padded.view(rows // 128, 4, 32, cols // 4, 4).transpose(1, 3).reshape(rows, cols)


def _known_byte_problem(m, k):
    # After the exact BF16 pqs multiply, every 16-value block is the complete
    # E2M1 codebook at scale 1. This stays away from reciprocal/tie ambiguity
    # and gives a literal expected packed-byte sequence.
    half_values = torch.tensor(
        [
            -3.0,
            -2.0,
            -1.5,
            -1.0,
            -0.75,
            -0.5,
            -0.25,
            0.0,
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
            1.5,
            2.0,
            3.0,
        ],
        device="cuda",
        dtype=torch.bfloat16,
    )
    x = half_values.repeat(m, k // 16).contiguous()
    pqs = torch.full((k,), 2.0, device="cuda", dtype=torch.bfloat16)
    global_scale = torch.ones(1, device="cuda", dtype=torch.float32)

    packed_block = torch.tensor(
        [0xEF, 0xCD, 0xAB, 0x09, 0x10, 0x32, 0x54, 0x76],
        device="cuda",
        dtype=torch.uint8,
    )
    expected = packed_block.repeat(m, k // 16)
    one_e4m3 = torch.ones((), device="cuda", dtype=torch.float8_e4m3fn).view(torch.uint8)
    logical_sf = one_e4m3.expand(m, k // 16)
    expected_sf = _swizzle_128x4(logical_sf, m, k)
    return x, pqs, global_scale, expected, expected_sf


@pytest.mark.L0
@pytest.mark.skipif(not _is_sm100(), reason="NVFP4 quantization requires SM100")
def test_nvfp4_quantize_generic_exact_bytes():
    x, pqs, global_scale, expected, expected_sf = _known_byte_problem(37, 256)
    out, sf = nvfp4_quantize(x, global_scale, pqs)

    assert torch.equal(out, expected)
    assert torch.equal(sf, expected_sf)


@pytest.mark.L1
@pytest.mark.skipif(not _is_sm100(), reason="NVFP4 quantization requires SM100")
@pytest.mark.parametrize("m", [1, 512, 4096])
@pytest.mark.parametrize("k", [3072, 12288])
def test_nvfp4_quantize_qwen_shapes_exact_bytes(m, k):
    x, pqs, global_scale, expected, expected_sf = _known_byte_problem(m, k)
    out, sf = nvfp4_quantize(x, global_scale, pqs)

    assert out.shape == (m, k // 2)
    assert sf.shape == ((m + 127) // 128 * 128, (k // 16 + 3) // 4 * 4)
    assert torch.equal(out, expected)
    assert torch.equal(sf, expected_sf)


@pytest.mark.L0
@pytest.mark.skipif(not _is_sm100(), reason="NVFP4 quantization requires SM100")
def test_nvfp4_quantize_optional_pqs_and_caller_outputs():
    torch.manual_seed(7)
    x = torch.randn((37, 256), device="cuda", dtype=torch.bfloat16)
    global_scale = ((448.0 * 6.0) / x.float().abs().amax()).reshape(())

    expected, expected_sf = nvfp4_quantize(x, global_scale)
    out = torch.empty_like(expected)
    sf = torch.empty_like(expected_sf)
    got, got_sf = nvfp4_quantize(
        x,
        global_scale,
        torch.ones(256, device="cuda", dtype=torch.bfloat16),
        out=out,
        scale_factors=sf,
    )

    assert got is out
    assert got_sf is sf
    assert torch.equal(got, expected)
    assert torch.equal(got_sf, expected_sf)

    with pytest.raises(ValueError, match=r"shape \(\) or \(1,\)"):
        nvfp4_quantize(x, global_scale.reshape(1, 1))


@pytest.mark.L0
@pytest.mark.skipif(not _is_sm100(), reason="NVFP4 quantization requires SM100")
def test_nvfp4_quantize_uses_current_stream(monkeypatch):
    import cudnn.gemm.ops._nvfp4_quantize as module

    launches = []

    class FakeExtension:
        @staticmethod
        def launch(*args):
            launches.append(args)

    monkeypatch.setattr(module, "_load_extension", lambda: FakeExtension)
    x = torch.empty((37, 256), device="cuda", dtype=torch.bfloat16)
    global_scale = torch.ones(1, device="cuda", dtype=torch.float32)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        out, sf = module.nvfp4_quantize(x, global_scale)

    assert launches[-1][-2] == stream.cuda_stream
    assert launches[-1][0] == x.data_ptr()
    assert launches[-1][3] == out.data_ptr()
    assert launches[-1][4] == sf.data_ptr()


@pytest.mark.L0
def test_nvfp4_quantize_rejects_before_lazy_build(monkeypatch):
    import cudnn.gemm.ops._nvfp4_quantize as module

    def unexpected_build():
        raise AssertionError("unsupported input reached the lazy compiler")

    monkeypatch.setattr(module, "_load_extension", unexpected_build)
    with pytest.raises(ValueError, match="CUDA tensor"):
        module.nvfp4_quantize(
            torch.empty((1, 256), dtype=torch.bfloat16),
            torch.ones(1, dtype=torch.float32),
        )
