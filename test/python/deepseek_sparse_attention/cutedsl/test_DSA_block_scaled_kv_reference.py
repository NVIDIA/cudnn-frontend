# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host checks of the block-scaled KV reference (``block_scaled_kv_reference``).

- The exactness class of each format's widening to BF16, over every code and
  scale byte: NVFP4 is lossless (also in FP16); MXFP8 rounds once and is exact
  for scale bytes 3..246.
- The quantizers write the codes and scales that decode to the pinned writers'
  QDQ values in ``benchmark/rope_qdq``.
- ``pack_rows`` builds rows of the width ``_kv_formats`` registers, and
  ``row_views`` splits them back without a copy; ``quantize_rows`` /
  ``widen_rows`` are the same steps by format name.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.L0

if not all(hasattr(torch, name) for name in ("float8_e8m0fnu", "float4_e2m1fn_x2")):
    pytest.skip("torch build lacks float8_e8m0fnu / float4_e2m1fn_x2", allow_module_level=True)

from deepseek_sparse_attention.cutedsl.block_scaled_kv_reference import (  # noqa: E402
    e2m1_values,
    ROW_LAYOUTS,
    pack_rows,
    quantize_mxfp8,
    quantize_nvfp4,
    quantize_rows,
    row_views,
    unpack_e2m1,
    widen_kv,
    widen_mxfp8,
    widen_nvfp4,
    widen_rows,
)

_ALL_BYTES = torch.arange(256, dtype=torch.int32).to(torch.uint8)
_E4M3_NAN_BYTES = (0x7F, 0xFF)


def _pinned_reference(name: str):
    """Load ``benchmark/rope_qdq/<name>.py``, FE's mirror of a pinned DeepSeek writer, by path."""
    path = Path(__file__).resolve().parents[4] / "benchmark" / "rope_qdq" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"rope_qdq_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nvfp4_widening_is_exact_for_every_code_and_scale_byte():
    # One 16-channel block per scale byte, holding the 16 codes in order.
    codes = torch.arange(16, dtype=torch.uint8).repeat(256, 1)
    kv = (codes[:, 0::2] | (codes[:, 1::2] << 4)).view(torch.float4_e2m1fn_x2)
    kv_sf = _ALL_BYTES.view(torch.float8_e4m3fn)[:, None]
    widened = widen_nvfp4(kv, kv_sf).double()
    exact = e2m1_values(codes).double() * kv_sf.double()

    finite = torch.ones(256, dtype=torch.bool)
    finite[list(_E4M3_NAN_BYTES)] = False
    assert torch.equal(widened[finite], exact[finite])
    assert torch.equal(torch.signbit(widened[finite]), torch.signbit(exact[finite]))
    assert torch.equal(exact[finite].to(torch.float16).double(), exact[finite]), "an NVFP4 product is not exact in FP16"
    assert bool(exact[finite].abs().max() == 6 * 448) and bool(exact[finite][exact[finite] != 0].abs().min() == 2.0**-10)
    assert bool(widened[~finite].isnan().all())


def test_mxfp8_widening_rounds_once_and_is_exact_for_scale_bytes_3_to_246():
    codes = _ALL_BYTES.repeat(256, 1)  # row = scale byte, column = code byte
    kv = codes.view(torch.float8_e4m3fn)
    kv_sf = _ALL_BYTES.view(torch.float8_e8m0fnu)[:, None].repeat(1, 8)
    widened = widen_mxfp8(kv, kv_sf)

    finite_code = torch.ones(256, dtype=torch.bool)
    finite_code[list(_E4M3_NAN_BYTES)] = False
    exact = kv.double() * torch.pow(2.0, torch.arange(256, dtype=torch.float64) - 127)[:, None]
    # Every finite code at every finite scale byte: the exact product rounded once to BF16 (to infinity above its range).
    assert torch.equal(widened[:255, finite_code].view(torch.int16), exact[:255, finite_code].to(torch.bfloat16).view(torch.int16))
    # Lossless exactly on scale bytes 3..246; bytes 2 and 247 each round some code.
    lossless = (widened.double() == exact)[:, finite_code].all(dim=1)
    assert lossless.nonzero().flatten().tolist() == list(range(3, 247))
    # NaN codes and the NaN scale byte 0xFF widen to NaN.
    assert bool(widened[:, ~finite_code].isnan().all()) and bool(widened[255].isnan().all())


def test_mxfp8_quantizer_decodes_to_the_pinned_writer_qdq():
    references = _pinned_reference("references_fp8")
    g = torch.Generator().manual_seed(0)
    x = torch.cat(
        (
            torch.randn(64, 512, generator=g),
            torch.randn(64, 512, generator=g) * 1e3,
            torch.randn(16, 512, generator=g) * 1e-7,  # below the 1e-4 amax floor
            torch.zeros(1, 512),
        )
    ).to(torch.bfloat16)
    kv, kv_sf = quantize_mxfp8(x)
    assert kv.dtype == torch.float8_e4m3fn and kv.shape == x.shape
    assert kv_sf.dtype == torch.float8_e8m0fnu and kv_sf.shape == (x.shape[0], 16)
    assert torch.equal(widen_mxfp8(kv, kv_sf).view(torch.int16), references.qdq_values(x).view(torch.int16))
    # The amax floor fixes the smallest scale byte at 105: ceil(log2(1e-4 / 448)) + 127.
    assert int(kv_sf.view(torch.uint8)[-17:].max()) == 105


def test_nvfp4_quantizer_decodes_to_the_pinned_writer_qdq():
    references = _pinned_reference("references_compkv")
    g = torch.Generator().manual_seed(1)
    x = torch.cat(
        (
            torch.randn(64, 512, generator=g),
            torch.randn(64, 512, generator=g) * 1e3,  # scales saturate at 448
            torch.randn(16, 512, generator=g) * 1e-5,  # below the 2**-9 scale floor
            torch.zeros(1, 512),
        )
    ).to(torch.bfloat16)
    kv, kv_sf = quantize_nvfp4(x)
    assert kv.dtype == torch.float4_e2m1fn_x2 and kv.shape == (x.shape[0], 256)
    assert kv_sf.dtype == torch.float8_e4m3fn and kv_sf.shape == (x.shape[0], 32)
    assert torch.equal(widen_nvfp4(kv, kv_sf).view(torch.int16), references.qdq_values(x).view(torch.int16))
    # Ties go to the even code, and channel 2i is the low nibble of byte i.
    ties = torch.tensor([[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25] + [6.0] + [0.0] * 7], dtype=torch.float32)
    tie_kv, tie_sf = quantize_nvfp4(ties)
    assert float(tie_sf.float()) == 1.0
    assert unpack_e2m1(tie_kv)[0, :9].tolist() == [0, 2, 2, 4, 4, 6, 6, 8, 7]
    assert tie_kv.view(torch.uint8)[0, 0].item() == 0x20


@pytest.mark.parametrize("name, quantize", [("mxfp8_528", quantize_mxfp8), ("nvfp4_288", quantize_nvfp4)])
def test_packed_rows_have_the_registered_width(name, quantize):
    from cudnn.deepseek_sparse_attention.sparse_attention_forward._kv_formats import KV_FORMATS

    fmt = KV_FORMATS[name]
    x = torch.randn(3, fmt.head_dim_qk).to(torch.bfloat16)
    rows = pack_rows(*quantize(x))
    assert rows.dtype == torch.uint8 and rows.shape == (3, fmt.record_bytes)
    code_dtype, code_bytes, scale_bytes, scale_dtype = ROW_LAYOUTS[name]
    assert code_bytes + scale_bytes == fmt.record_bytes
    assert torch.equal(quantize_rows(x, name), rows)
    assert torch.equal(widen_rows(rows, name).view(torch.int16), widen_kv(*quantize(x)).view(torch.int16))


@pytest.mark.parametrize("fmt", ["mxfp8", "nvfp4"])
@pytest.mark.parametrize("row_pitch", [None, 544])
def test_rows_hold_codes_then_scales_and_views_copy_nothing(fmt, row_pitch):
    x = torch.randn(5, 512).to(torch.bfloat16)
    if fmt == "mxfp8":
        kv, kv_sf = quantize_mxfp8(x)
        layout = (torch.float8_e4m3fn, 512, 16, torch.float8_e8m0fnu)
    else:
        kv, kv_sf = quantize_nvfp4(x)
        layout = (torch.float4_e2m1fn_x2, 256, 32, torch.float8_e4m3fn)
    rows = pack_rows(kv, kv_sf, row_pitch=row_pitch)
    pitch = layout[1] + layout[2] if row_pitch is None else row_pitch
    assert rows.shape == (5, pitch)
    view_kv, view_sf = row_views(rows, *layout)
    assert view_kv.stride() == (pitch, 1) and view_sf.stride() == (pitch, 1)
    assert view_sf.data_ptr() == view_kv.data_ptr() + layout[1]
    assert torch.equal(view_kv.view(torch.uint8), kv.view(torch.uint8)) and torch.equal(view_sf.view(torch.uint8), kv_sf.view(torch.uint8))
    assert torch.equal(rows[:, layout[1] + layout[2] :], torch.full_like(rows[:, layout[1] + layout[2] :], 0xFF))
