# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch-invariant MXFP8 and MXFP4 quantization: row independence, replay, an independent torch reference, FlashInfer parity."""

from pathlib import Path

import pytest
import torch

import cudnn
from cudnn.quantize.batch_invariant.layout import unswizzle_sf
from determinism_utils import assert_bitwise, assert_golden, assert_replays
from quant_reference import adversarial_input, e2m1_pack, make_input

pytestmark = pytest.mark.L0

LAYOUTS = ["128x4", "8x4", "linear"]
FORMATS = ["mxfp8", "mxfp4"]
DTYPES = [torch.bfloat16, torch.float16, torch.float32]


def quantize(fmt, x, layout="128x4", **kwargs):
    """Returns the codes as uint8 bytes and the scale bytes as a flat uint8 vector."""
    fn = cudnn.ops.mxfp8_quantize_forward if fmt == "mxfp8" else cudnn.ops.mxfp4_quantize_forward
    codes, sf = fn(x, layout, **kwargs)
    return codes.view(torch.uint8), sf.reshape(-1)


def logical_scales(sf, m, k, layout):
    return unswizzle_sf(sf, m, k, 32, layout)


def reference_codes(fmt, x):
    """TE's MX recipe in torch: UE8M0 ceil of amax * (1 / max_norm), then round-to-nearest-even onto the element grid."""
    from sdpa.mxfp8_quant import e8m0_ceil, exp2_rcp

    m, k = x.shape
    blocks = x.float().reshape(m, k // 32, 32)
    max_norm = 448.0 if fmt == "mxfp8" else 6.0
    e = e8m0_ceil(blocks.abs().amax(-1) * torch.tensor(1.0 / max_norm, dtype=torch.float32, device=x.device))
    scaled = (blocks * exp2_rcp(e).unsqueeze(-1)).clamp(-max_norm, max_norm)
    if fmt == "mxfp8":
        return scaled.to(torch.float8_e4m3fn).reshape(m, k).view(torch.uint8), e
    return e2m1_pack(scaled), e


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("fmt", FORMATS)
def test_rows_are_independent_of_the_batch(quant_device, fmt, dtype, layout):
    m, k = 135, 1056  # M crosses a 128-row tile; K = 33 scale blocks pads the 4-column scale tile
    x = make_input(quant_device, dtype, m, k, 11)
    snapshot = assert_replays(lambda: dict(zip(("codes", "sf"), quantize(fmt, x, layout))))
    codes, sf = snapshot["codes"], snapshot["sf"]
    scales = logical_scales(sf, m, k, layout)

    rows = [129, 0, 129, 134, 17]
    c2, s2 = quantize(fmt, x[rows], layout)
    assert_bitwise({"codes": c2, "scales": logical_scales(s2, len(rows), k, layout)}, {"codes": codes[rows], "scales": scales[rows]}, "reorder and duplicate")
    c1, s1 = quantize(fmt, x[129:130], layout)
    assert_bitwise({"codes": c1, "scales": logical_scales(s1, 1, k, layout)}, {"codes": codes[129:130], "scales": scales[129:130]}, "rollout row")

    # The other rows change; this row does not.
    y = x.clone()
    y[:129] = make_input(quant_device, dtype, 129, k, 12)
    c3, s3 = quantize(fmt, y, layout)
    assert_bitwise({"codes": c3[129:], "scales": logical_scales(s3, m, k, layout)[129:]}, {"codes": codes[129:], "scales": scales[129:]}, "neighbours changed")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("fmt", FORMATS)
def test_matches_independent_torch_reference(quant_device, fmt, dtype):
    m, k = 70, 512
    x = make_input(quant_device, dtype, m, k, 21)
    codes, sf = quantize(fmt, x, "linear")
    ref_codes, ref_e = reference_codes(fmt, x)
    assert_bitwise({"scales": sf.view(m, k // 32)}, {"scales": ref_e}, "scale bytes")
    assert_bitwise({"codes": codes.reshape(m, -1)}, {"codes": ref_codes.reshape(m, -1)}, "codes")


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("fmt", FORMATS)
def test_matches_flashinfer_batch_invariant(quant_device, fmt, dtype, layout):
    flashinfer = pytest.importorskip("flashinfer")
    if fmt == "mxfp4" and dtype == torch.float32:
        pytest.skip("FlashInfer's MXFP4 quantizer takes 16-bit inputs")
    fl = {"128x4": flashinfer.SfLayout.layout_128x4, "8x4": flashinfer.SfLayout.layout_8x4, "linear": flashinfer.SfLayout.layout_linear}[layout]
    for m, k in ((1, 32), (5, 128), (300, 4096)):
        x = make_input(quant_device, dtype, m, k, 31)
        if fmt == "mxfp8":
            want = flashinfer.mxfp8_quantize(x, sf_swizzle_layout=fl, backend="auto", batch_invariant=True)
        else:
            want = flashinfer.mxfp4_quantize(x, sfLayout=fl, backend="auto", batch_invariant=True)
        fn = cudnn.ops.mxfp8_quantize_forward if fmt == "mxfp8" else cudnn.ops.mxfp4_quantize_forward
        got = fn(x, layout)
        assert_bitwise({"codes": got[0].view(torch.uint8), "sf": got[1]}, {"codes": want[0].view(torch.uint8), "sf": want[1]}, f"{fmt} M={m} K={k}")


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("fmt", FORMATS)
def test_boundary_inputs_match_independent_torch_reference(quant_device, fmt, dtype):
    # Zero and signed-zero blocks, subnormal and near-max values, rounding midpoints, a wide exponent range and outliers.
    m, k = 64, 512
    x = adversarial_input(quant_device, dtype, m, k, 91)
    codes, sf = quantize(fmt, x, "linear")
    ref_codes, ref_e = reference_codes(fmt, x)
    assert_bitwise({"scales": sf.view(m, k // 32)}, {"scales": ref_e}, "scale bytes")
    assert_bitwise({"codes": codes.reshape(m, -1)}, {"codes": ref_codes.reshape(m, -1)}, "codes")


@pytest.mark.parametrize("nonfinite", [False, True], ids=["finite", "with_inf_nan"])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("fmt", FORMATS)
def test_boundary_rows_are_independent_of_the_batch(quant_device, fmt, layout, nonfinite):
    # A row holding inf or NaN must neither change nor be changed by its neighbours.
    m, k = 135, 1056
    x = adversarial_input(quant_device, torch.bfloat16, m, k, 92, nonfinite)
    codes, sf = assert_replays(lambda: dict(zip(("codes", "sf"), quantize(fmt, x, layout)))).values()
    scales = logical_scales(sf, m, k, layout)
    for rows in ([129, 0, 129, 134, 17], [5], list(range(m - 1, -1, -1))):
        c, s = quantize(fmt, x[rows], layout)
        assert_bitwise({"codes": c, "scales": logical_scales(s, len(rows), k, layout)}, {"codes": codes[rows], "scales": scales[rows]}, f"rows {rows[:3]}")


@pytest.mark.parametrize("nonfinite", [False, True], ids=["finite", "with_inf_nan"])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32], ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("fmt", FORMATS)
def test_boundary_inputs_match_flashinfer_batch_invariant(quant_device, fmt, dtype, layout, nonfinite):
    flashinfer = pytest.importorskip("flashinfer")
    if fmt == "mxfp4" and dtype == torch.float32:
        pytest.skip("FlashInfer's MXFP4 quantizer takes 16-bit inputs")
    fl = {"128x4": flashinfer.SfLayout.layout_128x4, "8x4": flashinfer.SfLayout.layout_8x4, "linear": flashinfer.SfLayout.layout_linear}[layout]
    x = adversarial_input(quant_device, dtype, 70, 1056, 93, nonfinite)
    if fmt == "mxfp8":
        want = flashinfer.mxfp8_quantize(x, sf_swizzle_layout=fl, backend="auto", batch_invariant=True)
    else:
        want = flashinfer.mxfp4_quantize(x, sfLayout=fl, backend="auto", batch_invariant=True)
    fn = cudnn.ops.mxfp8_quantize_forward if fmt == "mxfp8" else cudnn.ops.mxfp4_quantize_forward
    got = fn(x, layout)
    assert_bitwise({"codes": got[0].view(torch.uint8), "sf": got[1]}, {"codes": want[0].view(torch.uint8), "sf": want[1]}, f"{fmt} boundary")


@pytest.mark.parametrize("fmt", FORMATS)
def test_quantizes_on_the_current_stream(quant_device, fmt):
    x = make_input(quant_device, torch.bfloat16, 64, 4096, 41)
    expected = quantize(fmt, x)
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        y = torch.zeros_like(x)
        torch.cuda._sleep(50_000_000)  # keep the fill pending on the side stream while the quantizer is enqueued
        y.copy_(x)
        got = quantize(fmt, y)
    side.synchronize()
    assert_bitwise({"codes": got[0], "sf": got[1]}, {"codes": expected[0], "sf": expected[1]}, "side stream")


@pytest.mark.parametrize("fmt", FORMATS)
def test_strided_and_misaligned_inputs(quant_device, fmt):
    wide = make_input(quant_device, torch.bfloat16, 40, 160, 51)
    view = wide[:, 32:96]  # a row stride of 160 elements and a 64-byte offset
    odd = torch.empty(40 * 64 + 8, device=quant_device, dtype=torch.bfloat16)[8:].view(40, 64)  # 16-byte offset
    odd.copy_(view)
    shifted = torch.empty(40 * 64 + 1, device=quant_device, dtype=torch.bfloat16)[1:].view(40, 64)  # 2-byte offset
    shifted.copy_(view)
    want = quantize(fmt, view.contiguous())
    for name, t in (("strided", view), ("offset16", odd), ("offset2", shifted)):
        got = quantize(fmt, t)
        assert_bitwise({"codes": got[0], "sf": got[1]}, {"codes": want[0], "sf": want[1]}, name)


def test_mxfp8_pads_k_to_the_alignment(quant_device):
    x = make_input(quant_device, torch.bfloat16, 7, 96, 61)
    codes, sf = cudnn.ops.mxfp8_quantize_forward(x, "linear", alignment=128)
    assert codes.shape == (7, 128) and codes.dtype == torch.float8_e4m3fn
    plain, plain_sf = quantize("mxfp8", x, "linear")
    assert_bitwise({"codes": codes.view(torch.uint8)[:, :96]}, {"codes": plain}, "payload")
    assert not codes.view(torch.uint8)[:, 96:].any() and not sf.view(7, 4)[:, 3].any(), "padding is zero bytes under a zero scale"


@pytest.mark.parametrize("fmt", FORMATS)
def test_empty_batch_and_leading_dims(quant_device, fmt):
    fn = cudnn.ops.mxfp8_quantize_forward if fmt == "mxfp8" else cudnn.ops.mxfp4_quantize_forward
    codes, sf = fn(torch.empty(0, 64, device=quant_device, dtype=torch.bfloat16))
    assert codes.shape[0] == 0 and codes.shape[-1] == (64 if fmt == "mxfp8" else 32)
    x = make_input(quant_device, torch.bfloat16, 24, 128, 71)
    flat = quantize(fmt, x, "linear")
    cube = fn(x.view(2, 12, 128), "linear")
    assert cube[0].shape[:2] == (2, 12)
    assert_bitwise(
        {"codes": cube[0].reshape(24, -1).view(torch.uint8), "sf": cube[1].reshape(-1)}, {"codes": flat[0].reshape(24, -1), "sf": flat[1]}, "leading dims"
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_unsupported_inputs_raise(quant_device, fmt):
    fn = cudnn.ops.mxfp8_quantize_forward if fmt == "mxfp8" else cudnn.ops.mxfp4_quantize_forward
    good = torch.ones(4, 64, device=quant_device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="K"):
        fn(torch.ones(4, 48, device=quant_device, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="dtype"):
        fn(good.double())
    with pytest.raises(ValueError, match="layout"):
        fn(good, "64x4")
    with pytest.raises(ValueError, match="CUDA"):
        fn(good.cpu())


@pytest.mark.parametrize("fmt", FORMATS)
def test_golden(quant_device, fmt):
    # Dyadic inputs keep the golden independent of the generator's float stream.
    g = torch.Generator().manual_seed(81)
    x = (torch.randint(-64, 65, (9, 96), generator=g).float() / 8).to(device=quant_device, dtype=torch.bfloat16)
    codes, sf = quantize(fmt, x, "128x4")
    ref_codes, _ = reference_codes(fmt, x)
    assert_bitwise({"codes": codes.reshape(9, -1)}, {"codes": ref_codes.reshape(9, -1)}, "independent reference")
    assert_golden(f"batch_invariant_{fmt}_quantize_bf16_128x4", {"x": x}, {"codes": codes, "sf": sf}, baseline_dir=Path(__file__).with_name("goldens"))


@pytest.mark.parametrize("fmt", FORMATS)
def test_golden_boundary_inputs(quant_device, fmt):
    # Boundary families built from float32 values that bf16 holds exactly, so the golden does not depend on the generator's float stream.
    x = adversarial_input("cpu", torch.bfloat16, 16, 256, 82).to(quant_device)
    codes, sf = quantize(fmt, x, "128x4")
    ref_codes, _ = reference_codes(fmt, x)
    assert_bitwise({"codes": codes.reshape(16, -1)}, {"codes": ref_codes.reshape(16, -1)}, "independent reference")
    assert_golden(f"batch_invariant_{fmt}_quantize_boundary_bf16_128x4", {"x": x}, {"codes": codes, "sf": sf}, baseline_dir=Path(__file__).with_name("goldens"))
