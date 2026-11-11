# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers of the attention gate + quantize decode step's test modules: the operand builders, the fp64 reference with its two
assertions (the bracket and the true-nearest margin rule), the torch model of the rounding convention on the device, and the device gate.
Not a test module (no ``test_`` prefix); imported by name from the sibling modules (pytest puts their directory on ``sys.path``).

THE REFERENCE.  The step's contract is a rounding CHAIN (``e4m3(bf16(a / (1 + exp(-g))) * (1 / s))``), so a tolerance on values is the
wrong tool; the reference is three statements about BYTES, strongest last:

1. *The bracket* (independent of any GPU arithmetic).  With ``t = a / (1 + exp(-g))`` in fp64 on the bf16 operands (``exp(-g)`` = +inf where
   it exceeds the fp32 maximum, the kernel's fp32 overflow) and ``lo``, ``hi`` the two bf16 values that bracket ``t`` (equal where ``t`` is a
   bf16 value), EVERY byte is ``q(lo)`` or ``q(hi)``, ``q(v) = e4m3(clamp(fp32(v) * inv, +-448))`` with ``inv`` the correctly rounded fp32
   reciprocal of the scale: one fp32 multiply and one nearest-even cast, both exactly reproducible in torch.  The kernel's fp32 chain
   (the exponential, one add, one division) stays within a few fp32 units of ``t``, 2^-16 of a bf16 unit, so its bf16 rounding can only land on
   a neighbour of ``t``; after that the kernel and ``q`` do the same two exact operations.
2. *The true-nearest margin rule* (catches a kernel that truncates, or rounds the other way, which the bracket alone would pass).  Wherever
   ``t`` is further than ``MARGIN_ULP32`` fp32 units from the midpoint of its bracket, the byte is ``q`` of the TRUE nearest bf16 of ``t``
   (chosen by comparing ``t`` with the midpoint in fp64, never by a cast that rounds twice).  No share, no floor: inside the margin the fp32
   quotient may sit on the other side of (or exactly on) the midpoint, which is the chain's own behaviour, and statement 1 still holds there.
3. *The torch model on the device* (``device_model``): the same chain spelled with torch's CUDA ``exp``, asserted BITWISE under the recorded
   flag ``DEVICE_MODEL_BITWISE``.  It is not independent of the toolchain (both exponentials are the same algorithm), which is why it comes
   last; if the two lowerings ever part, the flag changes with the measurement and statements 1 and 2 stand.
"""

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from cudnn.attention_gate.frost import gate_quant_decode as entry  # noqa: E402
from cudnn.attention_gate.frost.kernel import gate_quant_decode_config as kcfg  # noqa: E402

FLT_MAX = 3.4028234663852886e38
E4M3_MAX = 448.0
SCALES = (0.01, 0.02, 1.0, 2.0**-10, 37.5)  # the scales of the reference cells (a small, a typical, unit, a power of two, a large one)

# MEASURED on a cc 10.3 part (CuTe DSL 4.8.0) over EVERY (attn, gate) bf16 pair, 2^32 pairs per scale, at the five SCALES -- the census
# module beside this one is the measurement and its command (``pytest -m L2 attention_gate/frost/test_gate_quant_decode_census.py``):
# 0 bytes outside the bracket; the bytes that are not q(true nearest) number 2 / 2 / 32 / 32 / 0 and sit at most 0.82 fp32 units from
# the bracket's midpoint; the torch model's bf16 value leaves the true nearest on 998,698 of the 4,261,478,400 finite pairs, at most 1.17
# fp32 units from the midpoint.  The rule's margin is the larger measurement DOUBLED, rounded up.  Never widen it to pass a cell: a byte
# beyond the margin that is not q(true nearest) is a finding, and the census module fails when a runtime measures more than half of it.
MARGIN_ULP32 = 2.5

# RECORDED (same measurement): the torch model on the device equals the entry on every finite pair at every scale (5 x 4,261,478,400 bytes).
# Only a commit that carries a measurement showing the two exponential lowerings part (another compiler flavour, another torch) may set
# this False; statements 1 and 2 of the module docstring are then the whole reference.
DEVICE_MODEL_BITWISE = True


def scale_tensor(value, device, zero_d=False):
    return torch.tensor(value if zero_d else [value], dtype=torch.float32, device=device)


# every (attn, gate) pair of these lands in the first elements of an ``adversarial`` slab: saturation (|a| / scale >= 448 at the small scales),
# signed zeros, tiny and near-maximal magnitudes, E4M3 rounding midpoints at scale 1 (1.0625, 17), gates that saturate the sigmoid on either
# side (+-40), sit at the fp32 overflow of the exponential (+-88.5 / +-89, +-104) or far beyond it (+-700)
ADVERSARIAL_ATTN = (448.0, -448.0, 1e4, -1e4, 0.0, -0.0, 1e-30, -1e-30, 3.3895e38, -3.3895e38, 1.0625, -1.0625, 17.0, -17.0, 0.0146484375, 4.4765625)
ADVERSARIAL_GATE = (40.0, -40.0, 0.0, -0.0, 88.5, -88.5, 89.0, -89.0, 1e-30, 20.0, -20.0, 0.5, -0.5, 104.0, -104.0, 700.0, -700.0, 16.625, -16.625, 2.0)


def make_operands(rows, width, device, seed=0, kind="random"):
    """``attn`` / ``gate`` ``[rows, width]`` bf16, NaN-free and finite, from a CPU generator (the same values on every GPU): ``random`` =
    attn N(0, 1), gate 3 N(0, 1); ``adversarial`` = the same draw with every pair of the two special lists written from element 0 (and,
    where the slab holds it, once more three elements later so the pairs meet other positions inside a thread's eight elements), plus the
    smallest bf16 subnormals of both signs."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    attn = torch.randn(rows, width, generator=gen).to(device).bfloat16()
    gate = (torch.randn(rows, width, generator=gen) * 3).to(device).bfloat16()
    if kind == "adversarial":
        sa = torch.tensor(ADVERSARIAL_ATTN, device=device).bfloat16()
        sa = torch.cat([sa, torch.tensor([1, -32767], dtype=torch.int16, device=device).view(torch.bfloat16)])  # +-(smallest subnormal)
        sg = torch.tensor(ADVERSARIAL_GATE, device=device).bfloat16()
        pa, pg = sa.repeat_interleave(sg.numel()), sg.repeat(sa.numel())
        n = min(pa.numel(), attn.numel())
        attn.view(-1)[:n] = pa[:n]
        gate.view(-1)[:n] = pg[:n]
        if attn.numel() > 2 * pa.numel() + 3:
            attn.view(-1)[pa.numel() + 3 : 2 * pa.numel() + 3] = pa
            gate.view(-1)[pa.numel() + 3 : 2 * pa.numel() + 3] = pg
    elif kind != "random":
        raise ValueError(kind)
    assert bool(torch.isfinite(attn.float()).all()) and bool(torch.isfinite(gate.float()).all())
    return attn, gate


def _bits(v):
    return v.view(torch.int16).to(torch.int32) & 0xFFFF


def _from_bits(b):
    return b.to(torch.int16).view(torch.bfloat16)


def quantize(v_bf16, scale):
    """``q(v)``: the static E4M3 quantization of a bf16 value -- one fp32 multiply by the correctly rounded fp32 reciprocal of ``scale``, a
    clamp to +-448 (torch's cast reads NaN from 480 up where the hardware convert saturates; the clamp makes the two one function on every
    finite value) and torch's nearest-even E4M3 cast.  uint8 codes."""
    inv = torch.ones((), dtype=torch.float32, device=v_bf16.device) / scale.float().reshape(())
    return (v_bf16.float() * inv).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)


def fp64_bracket(attn, gate):
    """The fp64 gated value ``t`` of finite bf16 operands with its bf16 bracket: ``(t, v, other, nearest, margin)`` -- ``v`` and ``other``
    are the two bf16 values that bracket ``t`` (equal where ``t`` is a bf16 value), ``nearest`` the TRUE round-to-nearest-even bf16 of
    ``t``, ``margin`` the distance of ``t`` from the bracket's midpoint in fp32 units of ``t`` (+inf where ``t`` is a bf16 value)."""
    ex = torch.exp(-gate.double())
    ex = torch.where(ex > FLT_MAX, torch.full_like(ex, float("inf")), ex)  # the kernel's fp32 exponential overflows there: the gated value is +-0
    t = attn.double() / (1.0 + ex)
    v = t.float().bfloat16()  # ONE end of the bracket (both roundings are monotone); which one is decided below, in fp64
    vb = _bits(v)
    sign, mag = vb & 0x8000, vb & 0x7FFF
    v64 = v.double()
    away = _from_bits(sign | (mag + 1).clamp(max=0x7F80))
    toward = _from_bits(sign | (mag - 1).clamp(min=0))
    other = torch.where(t.abs() > v64.abs(), away, torch.where(t.abs() < v64.abs(), toward, v))
    o64 = other.double()
    exact = t == v64
    dv, do = (t - v64).abs(), (t - o64).abs()
    tie = (dv == do) & ~exact  # t on the midpoint at fp64 resolution (e.g. an odd subnormal halved by a gate of 0): nearest-even decides
    nearest = torch.where(exact | (dv < do) | (tie & ((vb & 1) == 0)), v, other)
    _, exponent = torch.frexp(t.abs())
    ulp32 = torch.ldexp(torch.ones_like(t), exponent.clamp(min=-125) - 24)
    margin = torch.where(exact, torch.full_like(t, float("inf")), (t - (v64 + o64) * 0.5).abs() / ulp32)
    return t, v, other, nearest, margin


def device_model(attn, gate, scale):
    """The torch model of the convention on the operands' device: ``e4m3(bf16(a / (1 + exp(-g))) * inv)`` -- the sigmoid as a DIVISION (the
    product ``a * sigmoid(g)`` is another rounding), torch's ``exp``, the bf16 rounding of the quotient, then :func:`quantize`.  The same
    spelling as the staged-store model of the split-KV combine's suite (``test/python/sdpa/frost/test_split_combine_gate_e4m3.py``); keep
    the two in step.  uint8 codes."""
    a, g = attn.float(), gate.float()
    return quantize((a / (1.0 + torch.exp(-g))).to(torch.bfloat16), scale)


def product_form_model(attn, gate, scale):
    """NOT the convention: the PRODUCT form ``e4m3(bf16(a * sigmoid(g)) * inv)``.  One code apart from the step on rare elements; kept so a
    cell can pin that the two are different functions."""
    return quantize((attn.float() * torch.sigmoid(gate.float())).to(torch.bfloat16), scale)


def assert_reference(out, attn, gate, scale, label=""):
    """The three statements of the module docstring on one launch's bytes (``out``: E4M3 or its uint8 view; finite operands)."""
    got = out.view(torch.uint8)
    assert got.shape == attn.shape, f"{label}: output shape {tuple(got.shape)} for operands {tuple(attn.shape)}"
    _, v, other, nearest, margin = fp64_bracket(attn, gate)
    q_v, q_o = quantize(v, scale), quantize(other, scale)
    outside = (got != q_v) & (got != q_o)
    n_out = int(outside.sum())
    assert (
        n_out == 0
    ), f"{label}: {n_out} of {got.numel()} bytes outside the bf16 bracket of the fp64 gated value (first at flat index {int(outside.reshape(-1).nonzero()[0])})"
    off = (margin > MARGIN_ULP32) & (got != quantize(nearest, scale))
    n_off = int(off.sum())
    assert n_off == 0, (
        f"{label}: {n_off} bytes are not the code of the true nearest bf16 although the gated value is more than {MARGIN_ULP32} fp32 units "
        f"from its bracket's midpoint (first at flat index {int(off.reshape(-1).nonzero()[0])}, margin {float(margin[off].min()):.3f})"
    )
    if DEVICE_MODEL_BITWISE:
        model = device_model(attn, gate, scale)
        n_model = int((got != model).sum())
        assert n_model == 0, f"{label}: {n_model} bytes differ from the torch model of the convention on the device (recorded bitwise)"


def _device_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    dev = torch.device("cuda", torch.cuda.current_device())
    cc = torch.cuda.get_device_capability(dev)
    if not entry.cc_supported(cc):
        pytest.skip(f"the attention gate + quantize decode step serves cc 10.x; this device is {cc[0]}.{cc[1]}")
    msg = entry._dsl_error(cc)
    if msg is not None:
        pytest.skip(msg)
    return dev
