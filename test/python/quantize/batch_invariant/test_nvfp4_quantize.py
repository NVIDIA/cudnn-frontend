# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch-invariant NVFP4 quantization, per tensor and per token: row independence, replay, an independent torch reference, FlashInfer parity."""

from pathlib import Path

import pytest
import torch

import cudnn
from cudnn.quantize.batch_invariant.layout import unswizzle_sf
from determinism_utils import assert_bitwise, assert_golden, assert_replays
from quant_reference import adversarial_input, make_input, nvfp4_reference

pytestmark = pytest.mark.L0

LAYOUTS = ["128x4", "8x4", "linear"]
DTYPES = [torch.bfloat16, torch.float16, torch.float32]
INV_BASE = 1.0 / (448 * 6)
MODES = [False, True]
MODE_IDS = ["per_tensor", "per_token"]


def global_scale(per_token):
    return INV_BASE if per_token else 0.37


def quantize(x, layout="128x4", per_token=False, scale=None, **kwargs):
    """Returns the codes and the flat scale bytes, plus the per-token scale vector when there is one."""
    out = cudnn.ops.nvfp4_quantize_forward(x, global_scale(per_token) if scale is None else scale, layout, per_token=per_token, **kwargs)
    return (out[0], out[1].reshape(-1), *out[2:])


def logical_scales(sf, m, k, layout):
    return unswizzle_sf(sf, m, k, 16, layout)


def as_outputs(out, m, k, layout):
    named = {"codes": out[0], "scales": logical_scales(out[1], m, k, layout)}
    if len(out) == 3:
        named["per_token_scale"] = out[2]
    return named


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("per_token", MODES, ids=MODE_IDS)
def test_rows_are_independent_of_the_batch(quant_device, per_token, dtype, layout):
    m, k = 135, 1056  # M crosses a 128-row tile; K = 66 scale blocks pads the 4-column scale tile
    x = make_input(quant_device, dtype, m, k, 11)
    out = assert_replays(lambda: dict(zip(("codes", "sf", "pts")[: 3 if per_token else 2], quantize(x, layout, per_token))))
    full = {"codes": out["codes"], "scales": logical_scales(out["sf"], m, k, layout)}
    if per_token:
        full["per_token_scale"] = out["pts"]

    def rows_of(idx):
        return {name: value[idx] for name, value in full.items()}

    rows = [129, 0, 129, 134, 17]
    assert_bitwise(as_outputs(quantize(x[rows], layout, per_token), len(rows), k, layout), rows_of(rows), "reorder and duplicate")
    assert_bitwise(as_outputs(quantize(x[129:130], layout, per_token), 1, k, layout), rows_of(slice(129, 130)), "rollout row")

    # The other rows change; this row does not.
    y = x.clone()
    y[:129] = make_input(quant_device, dtype, 129, k, 12)
    assert_bitwise({n: v[129:] for n, v in as_outputs(quantize(y, layout, per_token), m, k, layout).items()}, rows_of(slice(129, None)), "neighbours changed")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("per_token", MODES, ids=MODE_IDS)
def test_matches_independent_torch_reference(quant_device, per_token, dtype):
    m, k = 70, 512
    x = make_input(quant_device, dtype, m, k, 21)
    codes, sf, *extra = quantize(x, "linear", per_token)
    want = nvfp4_reference(x, global_scale(per_token), per_token)
    assert_bitwise({"scales": sf.view(m, k // 16)}, {"scales": want[1]}, "scale bytes")
    assert_bitwise({"codes": codes}, {"codes": want[0]}, "codes")
    if per_token:
        assert_bitwise({"pts": extra[0]}, {"pts": want[2]}, "per-token scale")


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("per_token", MODES, ids=MODE_IDS)
def test_matches_flashinfer_batch_invariant(quant_device, per_token, dtype, layout):
    flashinfer = pytest.importorskip("flashinfer")
    fl = {"128x4": flashinfer.SfLayout.layout_128x4, "8x4": flashinfer.SfLayout.layout_8x4, "linear": flashinfer.SfLayout.layout_linear}[layout]
    for m, k in ((1, 16), (5, 128), (300, 4096)):
        x = make_input(quant_device, dtype, m, k, 31)
        want = flashinfer.nvfp4_quantize(x, global_scale(per_token), sfLayout=fl, per_token_activation=per_token, backend="auto", batch_invariant=True)
        got = cudnn.ops.nvfp4_quantize_forward(x, global_scale(per_token), layout, per_token=per_token)
        assert len(got) == len(want)
        assert_bitwise({str(i): g for i, g in enumerate(got)}, {str(i): w for i, w in enumerate(want)}, f"M={m} K={k}")


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("per_token", MODES, ids=MODE_IDS)
def test_boundary_inputs_match_independent_torch_reference(quant_device, per_token, dtype):
    # Zero and signed-zero blocks, subnormal and near-max values, rounding midpoints, a wide exponent range and outliers.
    m, k = 64, 512
    x = adversarial_input(quant_device, dtype, m, k, 91)
    codes, sf, *extra = quantize(x, "linear", per_token)
    want = nvfp4_reference(x, global_scale(per_token), per_token)
    assert_bitwise({"scales": sf.view(m, k // 16)}, {"scales": want[1]}, "scale bytes")
    assert_bitwise({"codes": codes}, {"codes": want[0]}, "codes")
    if per_token:
        assert_bitwise({"pts": extra[0]}, {"pts": want[2]}, "per-token scale")


@pytest.mark.parametrize("nonfinite", [False, True], ids=["finite", "with_inf_nan"])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("per_token", MODES, ids=MODE_IDS)
def test_boundary_rows_are_independent_of_the_batch(quant_device, per_token, layout, nonfinite):
    # A row holding inf or NaN must neither change nor be changed by its neighbours.
    m, k = 135, 1056
    x = adversarial_input(quant_device, torch.bfloat16, m, k, 92, nonfinite)
    out = assert_replays(lambda: dict(zip(("codes", "sf", "pts")[: 3 if per_token else 2], quantize(x, layout, per_token))))
    full = {"codes": out["codes"], "scales": logical_scales(out["sf"], m, k, layout)}
    if per_token:
        full["per_token_scale"] = out["pts"]
    for rows in ([129, 0, 129, 134, 17], [5], list(range(m - 1, -1, -1))):
        assert_bitwise(as_outputs(quantize(x[rows], layout, per_token), len(rows), k, layout), {n: v[rows] for n, v in full.items()}, f"rows {rows[:3]}")


@pytest.mark.parametrize("nonfinite", [False, True], ids=["finite", "with_inf_nan"])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("per_token", MODES, ids=MODE_IDS)
def test_boundary_inputs_match_flashinfer_batch_invariant(quant_device, per_token, dtype, layout, nonfinite):
    flashinfer = pytest.importorskip("flashinfer")
    fl = {"128x4": flashinfer.SfLayout.layout_128x4, "8x4": flashinfer.SfLayout.layout_8x4, "linear": flashinfer.SfLayout.layout_linear}[layout]
    x = adversarial_input(quant_device, dtype, 70, 1056, 93, nonfinite)
    want = flashinfer.nvfp4_quantize(x, global_scale(per_token), sfLayout=fl, per_token_activation=per_token, backend="auto", batch_invariant=True)
    got = cudnn.ops.nvfp4_quantize_forward(x, global_scale(per_token), layout, per_token=per_token)
    assert len(got) == len(want)
    assert_bitwise({str(i): g for i, g in enumerate(got)}, {str(i): w for i, w in enumerate(want)}, "boundary")


def test_global_scale_forms_agree(quant_device):
    x = make_input(quant_device, torch.bfloat16, 33, 256, 41)
    host = quantize(x, scale=0.37)
    device = quantize(x, scale=torch.tensor([0.37], dtype=torch.float32, device=quant_device))
    on_cpu = quantize(x, scale=torch.tensor(0.37))
    for name, other in (("device tensor", device), ("cpu tensor", on_cpu)):
        assert_bitwise({"codes": other[0], "sf": other[1]}, {"codes": host[0], "sf": host[1]}, name)


def test_global_scale_changes_the_block_scales(quant_device):
    x = make_input(quant_device, torch.bfloat16, 8, 256, 42)
    a, b = quantize(x, scale=0.37), quantize(x, scale=0.74)
    assert not torch.equal(a[1], b[1]), "the global scale is an input of the per-tensor class"


@pytest.mark.parametrize("per_token", MODES, ids=MODE_IDS)
def test_quantizes_on_the_current_stream(quant_device, per_token):
    x = make_input(quant_device, torch.bfloat16, 64, 4096, 43)
    expected = quantize(x, per_token=per_token)
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        y = torch.zeros_like(x)
        torch.cuda._sleep(50_000_000)  # keep the fill pending on the side stream while the quantizer is enqueued
        y.copy_(x)
        got = quantize(y, per_token=per_token)
    side.synchronize()
    assert_bitwise({str(i): g for i, g in enumerate(got)}, {str(i): w for i, w in enumerate(expected)}, "side stream")


@pytest.mark.parametrize("per_token", MODES, ids=MODE_IDS)
def test_strided_and_misaligned_inputs(quant_device, per_token):
    wide = make_input(quant_device, torch.bfloat16, 40, 160, 51)
    view = wide[:, 32:96]  # a row stride of 160 elements and a 64-byte offset
    shifted = torch.empty(40 * 64 + 1, device=quant_device, dtype=torch.bfloat16)[1:].view(40, 64)  # 2-byte offset
    shifted.copy_(view)
    want = quantize(view.contiguous(), per_token=per_token)
    for name, t in (("strided", view), ("offset2", shifted)):
        got = quantize(t, per_token=per_token)
        assert_bitwise({str(i): g for i, g in enumerate(got)}, {str(i): w for i, w in enumerate(want)}, name)


@pytest.mark.parametrize("per_token", MODES, ids=MODE_IDS)
def test_empty_batch_and_leading_dims(quant_device, per_token):
    empty = cudnn.ops.nvfp4_quantize_forward(torch.empty(0, 64, device=quant_device, dtype=torch.bfloat16), global_scale(per_token), per_token=per_token)
    assert empty[0].shape == (0, 32) and len(empty) == (3 if per_token else 2)
    x = make_input(quant_device, torch.bfloat16, 24, 128, 71)
    flat = quantize(x, "linear", per_token)
    cube = quantize(x.view(2, 12, 128), "linear", per_token)
    assert cube[0].shape == (2, 12, 64)
    assert_bitwise(
        {"codes": cube[0].reshape(24, 64), "sf": cube[1], **({"pts": cube[2]} if per_token else {})},
        {"codes": flat[0], "sf": flat[1], **({"pts": flat[2]} if per_token else {})},
        "leading dims",
    )


def test_unsupported_inputs_raise(quant_device):
    fn = cudnn.ops.nvfp4_quantize_forward
    good = torch.ones(4, 64, device=quant_device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="K"):
        fn(torch.ones(4, 24, device=quant_device, dtype=torch.bfloat16), 1.0)
    with pytest.raises(ValueError, match="dtype"):
        fn(good.double(), 1.0)
    with pytest.raises(ValueError, match="layout"):
        fn(good, 1.0, "64x4")
    with pytest.raises(ValueError, match="global scale"):
        fn(good, torch.ones(2, device=quant_device))
    with pytest.raises(ValueError, match="CUDA"):
        fn(good.cpu(), 1.0)


@pytest.mark.parametrize("per_token", MODES, ids=MODE_IDS)
def test_golden(quant_device, per_token):
    # Dyadic inputs keep the golden independent of the generator's float stream.
    g = torch.Generator().manual_seed(81)
    x = (torch.randint(-64, 65, (9, 96), generator=g).float() / 8).to(device=quant_device, dtype=torch.bfloat16)
    codes, sf, *extra = quantize(x, "128x4", per_token)
    want = nvfp4_reference(x, global_scale(per_token), per_token)
    assert_bitwise({"codes": codes}, {"codes": want[0]}, "independent reference")
    outputs = {"codes": codes, "sf": sf, **({"per_token_scale": extra[0]} if per_token else {})}
    assert_golden(
        f"batch_invariant_nvfp4_quantize_{'per_token' if per_token else 'per_tensor'}_bf16_128x4",
        {"x": x},
        outputs,
        baseline_dir=Path(__file__).with_name("goldens"),
    )


@pytest.mark.parametrize("per_token", MODES, ids=MODE_IDS)
def test_golden_boundary_inputs(quant_device, per_token):
    # Boundary families built from float32 values that bf16 holds exactly, so the golden does not depend on the generator's float stream.
    x = adversarial_input("cpu", torch.bfloat16, 16, 256, 82).to(quant_device)
    codes, sf, *extra = quantize(x, "128x4", per_token)
    want = nvfp4_reference(x, global_scale(per_token), per_token)
    assert_bitwise({"codes": codes}, {"codes": want[0]}, "independent reference")
    outputs = {"codes": codes, "sf": sf, **({"per_token_scale": extra[0]} if per_token else {})}
    assert_golden(
        f"batch_invariant_nvfp4_quantize_boundary_{'per_token' if per_token else 'per_tensor'}_bf16_128x4",
        {"x": x},
        outputs,
        baseline_dir=Path(__file__).with_name("goldens"),
    )
