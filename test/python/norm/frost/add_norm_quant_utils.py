# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers of the add + RMSNorm + quantize decode step's test modules: the form tables, the operand builders (drawn on the
HOST), the independent references and the device gate.  Not a test module (no ``test_`` prefix); imported by name from the sibling
modules (pytest puts their directory on ``sys.path``).

THE REFERENCE.  The step's contract is a rounding CHAIN, so a tolerance on values is the wrong tool; the reference is statements
about BYTES.  All of it is torch math on the host, independent of the kernel's arithmetic:

1. *The normalized value* ``y`` (:func:`assert_y`).  With ``v = x + residual`` (exact in fp64 on bf16 operands) and
   ``y* = v (1 + w) / sqrt(mean(v^2) + eps)`` in fp64, ``lo`` and ``hi`` are the two bf16 values that bracket ``y*`` (equal where
   ``y*`` is a bf16 value).  EVERY output element is ``lo`` or ``hi``: the kernel's fp32 value before its one rounding to bf16 is within
   ``Y_REL_BOUND = 20 x 2^-24`` relative of ``y*`` -- 18 roundings on the deepest path of the all-positive sum of squares (8 fused steps,
   5 + 5 butterfly levels), halved by the root, one each for the division, the ``eps`` addition, the fp32 ``x + residual`` and
   ``1 + w``, two multiplications, and 2^-22 for the approximate reciprocal root -- three orders of magnitude below half a bf16 unit, so
   the rounding can only land on a neighbour of ``y*``.  A set membership: nothing to widen.  On random operands the count of elements
   that are NOT the true nearest bf16 of ``y*`` is also bounded from the arithmetic: an element can leave the nearest only if ``y*``
   lies within ``Y_REL_BOUND`` of a bf16 midpoint, the bf16 spacing is at least 2^-8 relative, so the share is at most
   ``p = 2 x Y_REL_BOUND x 2^8``, and the count over ``n`` elements at most ``ceil(p n + 6 sqrt(p n))``.
2. *The residual* (:func:`residual_reference`): bitwise ``bf16(fp32(x) + fp32(residual))`` -- one IEEE addition, one nearest-even cast.
3. *The E4M3 codes* (:func:`e4m3_codes`): bitwise the static quantization of the kernel's OWN ``y``: one fp32 multiplication by the
   correctly rounded fp32 reciprocal of the scale, a clamp to +-448 and torch's nearest-even E4M3 cast (the spelling of
   ``test/python/sdpa/frost/test_split_combine_gate_e4m3.py::_staged_twin`` -- keep the two in step).
4. *The NVFP4 block scales* (:func:`nvfp4_scale_bytes`): bitwise ``e4m3( gs x (amax x fp32(1/6)) )`` of the kernel's own ``y`` -- two fp32
   multiplications and one cast, no approximate operation -- read through the 128 x 4 layout rule (:func:`scale_bytes_of_rows`).
5. *The NVFP4 codes* (:func:`assert_nvfp4_codes`).  The encode scale goes through two approximate reciprocals, so the reference is a
   bracket: with ``q* = gs / F`` in fp64 (``F`` the fp32 value of the stored block scale) every code is ``e2m1(y q* (1 - r))`` or
   ``e2m1(y q* (1 + r))`` with ``r = RCP_REL_BOUND`` -- at most two ADJACENT codes, and one code for every element not within ``r`` of
   an E2M1 rounding midpoint.  A zero block scale encodes signed zeros.  ``RCP_REL_BOUND = 5 x 2^-23``: one unit in the last place per
   approximate reciprocal (its documented bound), doubled, plus one fp32 unit for the two multiplications.  A code outside the
   bracket is a finding with its magnitude, never a reason to widen.

EXACT OPERANDS (:func:`exact_operands`): rows of ``x = +-1`` with a zero residual have a sum of squares of exactly ``hidden`` and a mean
of exactly 1, so ``y_j = bf16(+-(1 + w_j) / sqrt(1 + eps))`` rounds to exactly ``+-(1 + w_j)`` for every bf16 target ``t_j = 1 + w_j``
whose ``w_j`` is a bf16 value: the WEIGHT places ``y`` on chosen values (E2M1 midpoints, saturation, zero blocks).
"""

import math

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from cudnn.norm.frost import add_norm_quant_decode as entry  # noqa: E402
from cudnn.norm.frost.kernel import add_norm_quant_decode_config as kcfg  # noqa: E402

HIDDEN = 5120
EPS = 1e-6
BF16, U8, E4M3 = torch.bfloat16, torch.uint8, torch.float8_e4m3fn
E4M3_MAX = 448.0
E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # positive codes 0 .. 7; code | 8 is the negative
#: the seven E2M1 rounding midpoints; the E2M1 rule below is the tree's (``e2m1_codes`` / ``pack_e2m1`` of
#: ``test/python/gated_attention_block/cutedsl/gated_block_reference.py``): nearest, ties to the EVEN code, saturation at 6, sign kept.
E2M1_MIDPOINTS = tuple((a + b) / 2 for a, b in zip(E2M1_GRID[:-1], E2M1_GRID[1:]))
FP4_MAX_RCP_BITS = 0x3E2AAAAB  # fp32(1 / 6): the kernel's constant
Y_REL_BOUND = 20 * 2.0**-24
Y_OFF_NEAREST_SHARE = 2 * Y_REL_BOUND * 2.0**8
RCP_REL_BOUND = 5 * 2.0**-23

#: the fifteen forms: ``id -> (add, store_residual, bf16, quant)``, residual mode x output set
RESIDUAL_MODES = (("add", True, False), ("store", False, True), ("none", False, False))
OUTPUT_SETS = (
    ("bf16", True, None),
    ("bf16-e4m3", True, "e4m3_static"),
    ("e4m3", False, "e4m3_static"),
    ("bf16-nvfp4", True, "nvfp4"),
    ("nvfp4", False, "nvfp4"),
)
FORMS = {f"{rname}-{oname}": (add, store, bf16, quant) for rname, add, store in RESIDUAL_MODES for oname, bf16, quant in OUTPUT_SETS}
#: the served launch arms as ``(pdl, pdl_release)``: off; wait only; wait + release
LAUNCH_ARMS = ((None, None), (True, False), (True, None))
LAUNCH_ARM_IDS = ("plain", "wait", "wait-release")
#: one form per numerical path AND per residual mode
DETERMINISM_FORMS = ("add-bf16-e4m3", "add-bf16-nvfp4", "add-bf16", "store-e4m3", "none-e4m3", "none-bf16-nvfp4")
E4M3_SCALES = (0.01, 0.02, 1.0, 37.5, 2.0**-7)
NVFP4_SCALES = (1.0, 0.25, 37.0, 2.0**10, 2.0**-3)


def form_kwargs(form):
    """The constructor keywords of a form id."""
    add, store, bf16, quant = FORMS[form]
    return dict(add=add, store_residual=store, bf16=bf16, quant=quant)


def make_entry(device, form, arm=(None, None), hidden=HIDDEN, eps=EPS):
    return entry.AddNormQuantDecode(device, hidden, eps, pdl=arm[0], pdl_release=arm[1], **form_kwargs(form))


def scale_tensor(value, device, zero_d=False):
    return torch.tensor(value if zero_d else [value], dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------------------------------------------------------
# operands (drawn on the host: one seed gives one set of tensors on every GPU)
# ---------------------------------------------------------------------------------------------------------------------------
def make_operands(rows, hidden=HIDDEN, device="cpu", seed=0, kind="random"):
    """``(x, residual, weight)`` bf16, NaN-free and finite, from a CPU generator and moved to ``device``: ``random`` =
    ``x ~ 3 N(0, 1)``, ``residual ~ 10 N(0, 1)``, ``weight ~ 0.3 N(0, 1)``; ``structured`` = the same draw with a few large entries
    per row, a block of zeros, alternating-sign runs and negative zeros, and row 0's residual cancelling its ``x`` on a stretch."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(rows, hidden, generator=gen) * 3).bfloat16()
    residual = (torch.randn(rows, hidden, generator=gen) * 10).bfloat16()
    weight = (torch.randn(hidden, generator=gen) * 0.3).bfloat16()
    if kind == "structured":
        big = torch.randint(0, hidden, (rows, 4), generator=gen)
        x.scatter_(1, big, torch.tensor([[300.0, -700.0, 1500.0, -40.0]]).bfloat16().expand(rows, 4))
        x[:, 64:96] = 0.0
        residual[:, 64:80] = 0.0
        x[:, 128:160] = torch.tensor([1.0, -1.0]).bfloat16().repeat(16)
        x[:, 200:208] = -0.0
        residual[0, 256:512] = -x[0, 256:512]
        weight[300:316] = -1.0  # 1 + w == 0: a block of zero outputs
        weight[400:408] = -2.5  # a negative (1 + w)
    elif kind != "random":
        raise ValueError(kind)
    assert bool(torch.isfinite(x.float()).all()) and bool(torch.isfinite(residual.float()).all()) and bool(torch.isfinite(weight.float()).all())
    return x.to(device), residual.to(device), weight.to(device)


def edge_operands(device="cpu", hidden=HIDDEN, seed=0):
    """``(x, residual, weight)`` of seven finite EDGE rows on top of a random draw: 0 an all-zero row (``x`` and ``residual``);
    1 negative zeros only; 2 one dominant element among zeros; 3 tiny values whose squares underflow fp32 (the mean is below ``eps``);
    4 a large in-domain row (magnitudes near 1e15); 5 ``residual == -x`` (the sum cancels to an all-zero ``v``); 6 a plain random row
    (the neighbour that must not notice the others)."""
    x, residual, weight = make_operands(7, hidden, "cpu", seed, "random")
    x[0], residual[0] = 0.0, 0.0
    x[1], residual[1] = -0.0, -0.0
    x[2], residual[2] = 0.0, 0.0
    x[2, 1234] = 3.0e4
    x[3] = (x[3].float() * 1e-25).bfloat16()
    residual[3] = (residual[3].float() * 1e-25).bfloat16()
    x[4] = (x[4].float() * 1e15).bfloat16()
    residual[4] = (residual[4].float() * 1e14).bfloat16()
    residual[5] = -x[5]
    return x.to(device), residual.to(device), weight.to(device)


def bf16_exact(values):
    """``values`` (fp32 tensor) as bf16 after asserting that every one IS a bf16 value."""
    out = values.float().bfloat16()
    assert bool((out.float() == values.float()).all()), "a target of an exact cell is not a bf16 value"
    return out


def exact_operands(targets, signs, device="cpu", hidden=HIDDEN):
    """``(x, residual, weight, y)`` for the EXACT construction: ``targets`` ``[hidden]`` non-negative bf16-valued fp32 (the wanted
    ``|y|`` per column), ``signs`` ``[rows, hidden]`` of +-1 (the sign of ``x``).  ``x = signs``, ``residual = 0``, ``weight =
    targets - 1``; the expected output ``y = signs x targets`` (a zero target gives the signed zero of ``x``'s sign)."""
    assert targets.shape == (hidden,) and signs.shape[1] == hidden and bool((signs.abs() == 1).all()) and bool((targets >= 0).all())
    weight = bf16_exact(targets.float() - 1.0)
    assert bool((weight.float() + 1.0 == bf16_exact(targets).float()).all()), "1 + w is not the target in fp32"
    x = bf16_exact(signs.float())
    residual = torch.zeros_like(x)
    y = bf16_exact(signs.float() * targets.float())
    return x.to(device), residual.to(device), weight.to(device), y.to(device)


def exact_targets(hidden=HIDDEN):
    """``[hidden]`` targets of the exact cells, per 16-element block: blocks 0-3 a block maximum of ``6 x 2^m`` with the seven E2M1
    midpoints times ``2^m`` twice and a zero (power-of-two block scales: exact codes); blocks 8-9 a maximum of 9.0 and 10.5 with the
    midpoints times 1.5 and 1.75 (block scales 1.5 and 1.75 at global scale 1: every midpoint element is an EXACT tie decided by the
    approximate reciprocal); block 20 all zero; block 21 all at the maximum; the rest 1.0."""
    mids = torch.tensor(E2M1_MIDPOINTS)
    t = torch.ones(hidden)
    for i, (peak, unit) in enumerate(((6.0, 1.0), (12.0, 2.0), (0.75, 0.125), (96.0, 16.0))):
        t[16 * i : 16 * i + 16] = torch.cat([torch.tensor([peak]), mids * unit, mids * unit, torch.tensor([0.0])])
    for i, (peak, unit) in ((8, (9.0, 1.5)), (9, (10.5, 1.75))):
        t[16 * i : 16 * i + 16] = torch.cat([torch.tensor([peak]), mids * unit, mids * unit, torch.tensor([0.0])])
    t[320:336] = 0.0
    t[336:352] = 6.0
    return t


def exact_signs(rows, hidden=HIDDEN):
    """``[rows, hidden]`` signs of +-1 for the exact cells from a CPU generator: row 0 all positive, row 1 all negative."""
    gen = torch.Generator(device="cpu").manual_seed(9)
    signs = torch.where(torch.rand(rows, hidden, generator=gen) < 0.5, -1.0, 1.0)
    signs[0] = 1.0
    signs[1] = -1.0
    return signs


# ---------------------------------------------------------------------------------------------------------------------------
# the references (host torch math on copies; every function returns host tensors)
# ---------------------------------------------------------------------------------------------------------------------------
def _bits(v):
    return v.view(torch.int16).to(torch.int32) & 0xFFFF


def _from_bits(b):
    return b.to(torch.int16).view(torch.bfloat16)


def fp32_eps(eps=EPS):
    """The kernel's constant: ``eps`` rounded to fp32, as an fp64 value."""
    return float(torch.tensor(eps, dtype=torch.float32).double())


def y_bracket(x, residual, weight, add, eps=EPS):
    """``(lo_or_hi_a, lo_or_hi_b, nearest)`` bf16 host tensors: the two bf16 values that bracket the fp64 normalized value ``y*`` of
    finite bf16 operands (equal where ``y*`` is a bf16 value) and its TRUE nearest-even bf16 (decided by comparing ``y*`` with the
    bracket's midpoint in fp64, never by a cast that rounds twice)."""
    v = x.detach().cpu().double()
    if add:
        v = v + residual.detach().cpu().double()
    g = 1.0 + weight.detach().cpu().double().reshape(1, -1)
    t = v * g / torch.sqrt((v * v).mean(dim=1, keepdim=True) + fp32_eps(eps))
    a = t.float().bfloat16()  # ONE end of the bracket (both roundings are monotone); which one is decided below, in fp64
    ab = _bits(a)
    sign, mag = ab & 0x8000, ab & 0x7FFF
    a64 = a.double()
    away = _from_bits(sign | (mag + 1).clamp(max=0x7F80))
    toward = _from_bits(sign | (mag - 1).clamp(min=0))
    b = torch.where(t.abs() > a64.abs(), away, torch.where(t.abs() < a64.abs(), toward, a))
    b64 = b.double()
    exact = t == a64
    da, db = (t - a64).abs(), (t - b64).abs()
    tie = (da == db) & ~exact
    nearest = torch.where(exact | (da < db) | (tie & ((ab & 1) == 0)), a, b)
    return a, b, nearest


def assert_y(out, x, residual, weight, add, eps=EPS, share_bound=False, label="", bracket=None):
    """Statement 1 of the module docstring on the kernel's ``out`` (bf16 ``[rows, hidden]``): every element is one of the two bf16
    neighbours of the fp64 value; with ``share_bound`` (random operands) the count off the true nearest is within the bound from the
    arithmetic.  Compared as bit patterns (a signed zero is its own value).  ``bracket`` is a precomputed :func:`y_bracket` of the
    same operands (the bracket does not depend on a scale)."""
    got = _bits(out.detach().cpu())
    a, b, nearest = bracket if bracket is not None else y_bracket(x, residual, weight, add, eps)
    outside = (got != _bits(a)) & (got != _bits(b))
    n_out = int(outside.sum())
    assert (
        n_out == 0
    ), f"{label}: {n_out} of {got.numel()} elements outside the bf16 bracket of the fp64 value (first at flat index {int(outside.reshape(-1).nonzero()[0])})"
    off = int((got != _bits(nearest)).sum())
    if share_bound:
        n = got.numel()
        limit = math.ceil(Y_OFF_NEAREST_SHARE * n + 6 * math.sqrt(Y_OFF_NEAREST_SHARE * n))
        assert off <= limit, f"{label}: {off} of {n} elements are not the true nearest bf16; the arithmetic bounds the count by {limit}"
    return off


def residual_reference(x, residual, add):
    """Statement 2: the residual after the call -- ``bf16(fp32(x) + fp32(residual))`` under ``add``, ``x`` otherwise (host tensor)."""
    if not add:
        return x.detach().cpu().clone()
    return (x.detach().cpu().float() + residual.detach().cpu().float()).bfloat16()


def e4m3_codes(y, scale):
    """Statement 3: the static E4M3 quantization of a bf16 ``y`` as uint8 codes (host tensor)."""
    inv = torch.ones((), dtype=torch.float32) / scale.detach().cpu().float().reshape(())
    return (y.detach().cpu().float() * inv).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3).view(U8)


def e2m1_codes(a):
    """fp32 / fp64 -> 4-bit E2M1 codes (uint8 ``0 .. 15``, sign in bit 3): nearest on the grid, ties to the EVEN code, saturation at
    6, sign kept (a negative zero is code 8).  NaN is refused."""
    if bool(torch.isnan(a).any()):
        raise ValueError("e2m1_codes: NaN is out of the quantizer's contract")
    mag = a.abs().double()
    mids = torch.tensor(E2M1_MIDPOINTS, dtype=torch.float64)
    lo = torch.bucketize(mag, mids, right=False)
    hi = torch.bucketize(mag, mids, right=True)
    code = torch.where(lo % 2 == 0, lo, hi)
    return (code.to(U8) | (torch.signbit(a).to(U8) << 3)).contiguous()


def pack_e2m1(codes):
    """E2M1 codes ``[..., K]`` -> bytes ``[..., K / 2]``: the EVEN element in the low nibble."""
    c = codes.to(U8) & 0xF
    return (c[..., 0::2] | (c[..., 1::2] << 4)).contiguous()


def unpack_e2m1(packed):
    """Bytes ``[..., K / 2]`` -> codes ``[..., K]`` (low nibble first)."""
    u = packed.view(U8)
    return torch.stack([u & 0xF, u >> 4], dim=-1).flatten(-2)


def nvfp4_scale_bytes(y, global_scale):
    """Statement 4: the block-scale bytes ``[rows, hidden / 16]`` of a bf16 ``y`` in LOGICAL (row, block) order (host tensor)."""
    yc = y.detach().cpu().float()
    amax = yc.abs().reshape(yc.shape[0], -1, 16).amax(dim=2)
    sixth = torch.tensor(FP4_MAX_RCP_BITS, dtype=torch.int32).view(torch.float32)
    gs = global_scale.detach().cpu().float().reshape(())
    return (gs * (amax * sixth)).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3).view(U8)


def scale_byte_index(rows, hidden=HIDDEN):
    """Host long ``[rows, hidden / 16]``: the byte of block scale ``(row, block)`` in a scale buffer, through the layout rule
    (``kcfg.scale_byte_offset``, the host twin of the kernel's index, evaluated on index tensors)."""
    cfg = kcfg.Cfg(HIDDEN=hidden, EPS=EPS, ADD=False, STORE_RESIDUAL=False, BF16_OUT=False, QUANT="nvfp4", PDL=False, PDL_RELEASE=False)
    row = torch.arange(rows, dtype=torch.long).reshape(-1, 1)
    block = torch.arange(hidden // 16, dtype=torch.long).reshape(1, -1)
    return kcfg.scale_byte_offset(cfg, row, block)


def scale_bytes_of_rows(scales, rows, hidden=HIDDEN):
    """The block-scale bytes of rows ``0 .. rows - 1`` in LOGICAL (row, block) order, read from a scale buffer in the 128 x 4 layout.
    Host uint8 ``[rows, hidden / 16]``."""
    return scales.detach().cpu().view(U8).reshape(-1)[scale_byte_index(rows, hidden)]


def scale_bytes_layout_mask(scale_buffer_rows, rows, hidden=HIDDEN):
    """A host bool mask over the flat bytes of a ``[scale_buffer_rows, hidden / 16]`` scale buffer: True on the bytes of the LIVE
    rows ``0 .. rows - 1`` (the bytes the kernel writes), False on the pad rows' bytes (which it never writes)."""
    mask = torch.zeros(scale_buffer_rows * (hidden // 16), dtype=torch.bool)
    mask[scale_byte_index(rows, hidden).reshape(-1)] = True
    return mask


def nvfp4_code_bracket(y, scale_bytes, global_scale):
    """Statement 5: ``(lo, hi)`` host uint8 E2M1 codes ``[rows, hidden]`` -- every code of the kernel is one of the two.
    ``scale_bytes`` are the stored block scales in logical order (the kernel's own bytes: statement 4 pins them separately)."""
    yc = y.detach().cpu().double()
    f = scale_bytes.view(E4M3).double().repeat_interleave(16, dim=1)
    gs = global_scale.detach().cpu().double().reshape(())
    q = torch.where(f != 0, gs / torch.where(f != 0, f, torch.ones_like(f)), torch.zeros_like(f))
    a = yc * q  # a zero block scale: the signed zero of y
    return e2m1_codes(a * (1 - RCP_REL_BOUND)), e2m1_codes(a * (1 + RCP_REL_BOUND))


def assert_nvfp4_codes(codes, y, scale_bytes, global_scale, label=""):
    """Statement 5 on the kernel's packed codes (uint8 ``[rows, hidden / 2]``); returns the count of elements whose bracket holds two
    codes (the elements within the reciprocal bound of an E2M1 midpoint)."""
    got = unpack_e2m1(codes.detach().cpu())
    lo, hi = nvfp4_code_bracket(y, scale_bytes, global_scale)
    outside = (got != lo) & (got != hi)
    n_out = int(outside.sum())
    assert (
        n_out == 0
    ), f"{label}: {n_out} of {got.numel()} NVFP4 codes outside the bracket of the approximate reciprocals (first at flat index {int(outside.reshape(-1).nonzero()[0])})"
    return int((lo != hi).sum())


def twin_form(form):
    """The form with the same residual mode and quant that also writes the bf16 value."""
    mode, _, rest = form.partition("-")
    return form if rest.startswith("bf16") else f"{mode}-bf16-{rest}"


def assert_reference(form, got, x, res0, weight, scale, y=None, share_bound=False, label="", bracket=None):
    """The references of this module on ONE launch's bytes (``got``: the residual after the call and every output the form
    produces, by name; ``res0`` the residual before it): the residual, the normalized value (where the form writes it; otherwise ``y``
    is the output of :func:`twin_form`'s entry on the same operands, already checked), the E4M3 codes, the NVFP4 block scales and the
    code bracket, and the zero pad rows of a scale buffer of exactly the call's tiles.  ``bracket`` is the precomputed fp64 bracket
    of the same operands."""
    add, store, bf16, quant = FORMS[form]
    rows = x.shape[0]
    if add or store:
        assert torch.equal(_bits(got["residual"].detach().cpu()), _bits(residual_reference(x, res0, add))), f"{label}: the residual"
    else:
        assert "residual" not in got
    if bf16:
        assert_y(got["out"], x, res0, weight, add, share_bound=share_bound, label=label, bracket=bracket)
        y = got["out"]
    else:
        assert "out" not in got and y is not None
    if quant == "e4m3_static":
        assert got["codes"].dtype == E4M3 and tuple(got["codes"].shape) == (rows, HIDDEN)
        want = e4m3_codes(y, scale)
        n = int((got["codes"].view(U8).cpu() != want).sum())
        assert n == 0, f"{label}: {n} E4M3 codes differ from the static quantization of the kernel's own bf16 value"
    if quant == "nvfp4":
        assert got["codes"].dtype == U8 and tuple(got["codes"].shape) == (rows, HIDDEN // 2)
        assert got["scales"].dtype == E4M3 and tuple(got["scales"].shape) == (kcfg.scale_rows(rows), HIDDEN // 16)
        stored = scale_bytes_of_rows(got["scales"], rows)
        n = int((stored != nvfp4_scale_bytes(y, scale)).sum())
        assert n == 0, f"{label}: {n} NVFP4 block scales differ from e4m3(gs x (amax x fp32(1/6))) of the kernel's own bf16 value"
        assert_nvfp4_codes(got["codes"], y, stored, scale, label)
        pad = ~scale_bytes_layout_mask(kcfg.scale_rows(rows), rows)
        assert bool((got["scales"].view(U8).cpu().reshape(-1)[pad] == 0).all()), f"{label}: the scale buffer's pad rows are zero"
    if quant is None:
        assert "codes" not in got and "scales" not in got


def _device_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    dev = torch.device("cuda", torch.cuda.current_device())
    cc = torch.cuda.get_device_capability(dev)
    if not entry.cc_supported(cc):
        pytest.skip(f"the add + RMSNorm + quantize decode step serves cc 10.x; this device is {cc[0]}.{cc[1]}")
    msg = entry._dsl_error(cc)
    if msg is not None:
        pytest.skip(msg)
    return dev
