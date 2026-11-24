# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers of the FP8 row quantization decode step's test modules: the launch arms, the operand builders (built on the
HOST), the independent reference and the device gate.  Not a test module (no ``test_`` prefix); imported by name from the sibling
modules (pytest puts their directory on ``sys.path``).

THE REFERENCE is written here from the operation's definition -- exact arithmetic on the host in fp64 with every fp32 rounding
point spelled; no kernel helper of the package is imported -- and it is a statement about BYTES::

    amax    = max |x_j|                                  over the row: exact
    scale   = max( fp32(amax / 448), FLOOR )             FLOOR = the fp32 bits 0x36924925 = fp32(1 / (448 * 512))
    inv     = fp32( 1 / scale )
    p_j     = fp32( x_j * inv )
    code_j  = e4m3( p_j )                                nearest E4M3 value, ties to the even code, saturating at +-448, sign kept

``fp32(.)`` is one rounding to nearest-even.  A quotient or a product of two fp32 values formed in fp64 and rounded once to
fp32 is the correctly rounded fp32 result (fp64 holds more than twice the bits of fp32 plus two), so no fp32 kernel of the host
is trusted for a division.  The E4M3 values are built from the format (:func:`e4m3_values`): one sign bit, four exponent bits
with bias 7, three mantissa bits; subnormals at exponent 0; 448 the largest finite magnitude.
"""

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from cudnn.row_quantize.frost import row_quantize_fp8_decode as entry  # noqa: E402

WIDTH = 17408
VEC = 8
BF16, U8, I16, E4M3, FP32, FP64 = torch.bfloat16, torch.uint8, torch.int16, torch.float8_e4m3fn, torch.float32, torch.float64
E4M3_MAX = 448.0
FLOOR_BITS = 0x36924925  # fp32(1 / (448 * 512))
KNEE_PATTERN = 0x3B00  # the bf16 pattern of 2**-9: the row maximum at which amax / 448 equals the floor
#: the finite non-negative bf16 patterns: 0 .. 0x7F7F (0x7F80 is +infinity)
BF16_FINITE = 0x7F80
#: the served launch arms as ``(pdl, pdl_release)``: off; wait only; release + wait
LAUNCH_ARMS = ((None, None), (True, False), (True, None))
LAUNCH_ARM_IDS = ("plain", "wait", "release-wait")


def make_entry(device, arm=(None, None), width=WIDTH):
    return entry.RowQuantizeFp8Decode(device, width, pdl=arm[0], pdl_release=arm[1])


def f32(bits):
    """The fp32 value of a bit pattern, as a 0-d host tensor."""
    return torch.tensor(bits, dtype=torch.int32).view(FP32)


def bf16_from_patterns(patterns):
    """bf16 values from their 16-bit patterns (an integer tensor of values 0 .. 0xFFFF)."""
    return (patterns.to(torch.int32) & 0xFFFF).to(I16).view(BF16)


# ---------------------------------------------------------------------------------------------------------------------------
# the reference
# ---------------------------------------------------------------------------------------------------------------------------
def e4m3_values():
    """The 127 finite non-negative E4M3 magnitudes by code ``0 .. 0x7E`` (fp64, increasing): ``m / 8 x 2**-6`` at exponent field 0,
    ``(1 + m / 8) x 2**(e - 7)`` otherwise; code ``0x7E`` is 448."""
    code = torch.arange(0x7F, dtype=torch.int64)
    e, m = code >> 3, (code & 7).to(FP64)
    return torch.where(e == 0, m / 8 * 2.0**-6, (1 + m / 8) * torch.exp2((e - 7).to(FP64)))


def e4m3_encode(p):
    """The E4M3 code bytes (host uint8) of fp32 values without a NaN: the nearest E4M3 magnitude, an exact tie to the EVEN code,
    saturation at 448, the sign bit of the value kept (a negative value that rounds to zero is the code ``0x80``)."""
    values = e4m3_values()
    a = p.abs().to(FP64).clamp(max=E4M3_MAX)
    hi = torch.searchsorted(values, a).clamp(max=0x7E)  # the first code whose magnitude is not below a
    lo = (hi - 1).clamp(min=0)
    below, above = a - values[lo], values[hi] - a
    take_hi = (above < below) | ((above == below) & (hi % 2 == 0))
    code = torch.where(take_hi, hi, lo)
    return (code | (torch.signbit(p).to(torch.int64) << 7)).to(U8)


def row_scales(amax):
    """``(scale, inv)`` (host fp32) of row maxima ``amax`` (host fp32, finite, non-negative): the two correctly rounded
    divisions and the floor."""
    floor = f32(FLOOR_BITS)
    scale = torch.maximum((amax.to(FP64) / E4M3_MAX).to(FP32), floor)
    inv = (1.0 / scale.to(FP64)).to(FP32)
    return scale, inv


def reference(x):
    """``(codes uint8 [rows, width], scales fp32 [rows, 1])`` of a finite bf16 ``x`` (any device; the arithmetic runs on the
    host): the module docstring's five lines."""
    xf = x.detach().cpu().to(FP32)
    assert bool(torch.isfinite(xf).all()), "the reference is defined for finite rows"
    scale, inv = row_scales(xf.abs().amax(dim=1))
    product = (xf.to(FP64) * inv.to(FP64)[:, None]).to(FP32)
    return e4m3_encode(product), scale[:, None].contiguous()


def assert_reference(codes, scales, x, label=""):
    """Every code byte and every scale bit of the live rows equals the reference's; a mismatch names its count and its first
    position."""
    rows = x.shape[0]
    want_codes, want_scales = reference(x)
    got_codes = codes.detach()[:rows].view(U8).cpu()
    got_scales = scales.detach()[:rows].cpu()
    assert got_scales.shape == want_scales.shape and got_scales.dtype == FP32, f"{label}: scales are {tuple(got_scales.shape)} {got_scales.dtype}"
    differ = got_scales.view(torch.int32) != want_scales.view(torch.int32)
    if bool(differ.any()):
        row = int(differ.nonzero()[0, 0])
        got, want = int(got_scales.view(torch.int32)[row, 0]) & 0xFFFFFFFF, int(want_scales.view(torch.int32)[row, 0]) & 0xFFFFFFFF
        raise AssertionError(f"{label}: {int(differ.sum())} of {rows} scales differ from the reference; row {row}: got 0x{got:08x}, want 0x{want:08x}")
    differ = got_codes != want_codes
    if bool(differ.any()):
        row, column = (int(v) for v in differ.nonzero()[0])
        raise AssertionError(
            f"{label}: {int(differ.sum())} of {differ.numel()} codes differ from the reference; first at ({row}, {column}): got "
            f"0x{int(got_codes[row, column]):02x}, want 0x{int(want_codes[row, column]):02x} for x = {float(x[row, column])!r}"
        )


# ---------------------------------------------------------------------------------------------------------------------------
# operands (built on the host: one seed gives one set of tensors on every GPU)
# ---------------------------------------------------------------------------------------------------------------------------
def make_operands(rows, device="cpu", seed=0, kind="random", width=WIDTH):
    """``x`` bf16 ``[rows, width]``, finite, from a CPU generator and moved to ``device``.  ``random``: normal values under a
    per-row magnitude spread over twenty octaves and a per-vector magnitude spread over eight.  ``structured``: the same draw
    with, cycling over the rows, a zero row, a row of negative zeros, a row below the floor's knee, a row whose ONE dominant
    element sits in the last column, in the first column, in the last vector of the second pass and in the first vector of the
    third, a row of two values of opposite sign, and a row whose maximum is held by many elements of both signs."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.randn(rows, width // VEC, VEC, generator=gen)
    row_mag = torch.exp2(torch.empty(rows, 1, 1).uniform_(-12.0, 8.0, generator=gen))
    vec_mag = torch.exp2(torch.empty(rows, width // VEC, 1).uniform_(-8.0, 0.0, generator=gen))
    x = (base * vec_mag * row_mag).reshape(rows, width).to(BF16)
    if kind == "structured":
        for row in range(rows):
            shape = row % 10
            if shape == 0:
                x[row] = 0.0
            elif shape == 1:
                x[row] = -0.0
            elif shape == 2:
                x[row] = (x[row].float() * 2.0**-30).to(BF16)  # far below the knee: the floor scale
            elif shape in (3, 4, 5, 6):
                column = (width - 1, 0, 2 * 1024 * VEC - 1, 2 * 1024 * VEC)[shape - 3]
                x[row] = (x[row].float() * 2.0**-9).to(BF16)
                x[row, column] = -700.0 if row % 2 else 300.0
            elif shape == 7:
                x[row] = torch.tensor([1.5, -1.5]).to(BF16).repeat(width // 2)
            elif shape == 8:
                top = x[row].float().abs().max()
                x[row, ::37] = top.to(BF16)
                x[row, 5::91] = (-top).to(BF16)
    elif kind != "random":
        raise ValueError(kind)
    assert bool(torch.isfinite(x.float()).all())
    return x.to(device)


def spike_rows(patterns, columns, width=WIDTH):
    """bf16 ``[n, width]`` of zero rows with ONE non-zero element each: row ``i`` holds the bf16 pattern ``patterns[i]`` at column
    ``columns[i]`` (two integer tensors of ``n`` entries).  The row maximum of row ``i`` is that element's magnitude."""
    x = torch.zeros(len(patterns), width, dtype=BF16)
    x[torch.arange(len(patterns)), columns] = bf16_from_patterns(patterns)
    return x


def spike_reference(patterns):
    """``(code uint8 [n], scale fp32 [n])`` of the ONE non-zero element of each row of :func:`spike_rows` -- every other code of
    such a row is the code of ``+0`` -- by the reference's five lines."""
    value = bf16_from_patterns(patterns).to(FP32)
    scale, inv = row_scales(value.abs())
    return e4m3_encode((value.to(FP64) * inv.to(FP64)).to(FP32)), scale


def _device_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    dev = torch.device("cuda", torch.cuda.current_device())
    cc = torch.cuda.get_device_capability(dev)
    if not entry.cc_supported(cc):
        pytest.skip(f"the FP8 row quantization decode step serves cc 10.x; this device is {cc[0]}.{cc[1]}")
    msg = entry._dsl_error(cc)
    if msg is not None:
        pytest.skip(msg)
    return dev
