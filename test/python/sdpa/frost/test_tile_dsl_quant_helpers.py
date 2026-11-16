# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The global-scale NVFP4 helpers of ``cudnn.frost.tile_dsl.pointwise`` -- ``rcp_approx_ftz``, ``e4m3_round``,
``silu_ex2_div_approx``, ``nvfp4_block_scale_global`` -- and ``cudnn.frost.tile_dsl.barrier.named_barrier_sync_unaligned``.

Each helper is a rounding CONVENTION or a synchronization CONTRACT, so each cell compares raw bits with a reference that is
named here; no cell has a tolerance on a helper's own result.

* host twins (torch on the CPU, no kernel): the constants of the spellings; the E4M3 pair as torch's ``float32 ->
  float8_e4m3fn`` cast and its exact widening, on its rounding corners; the chain's code as two IEEE fp32 multiplies in the
  helper's association, with operand pairs on which the OTHER association lands on another code (found by a search, so the
  device cell can prove the association).
* host rendering (a file probe in a fresh process, ``sm_100a``; ``sm_107a`` where the compiler flavour serves it): every helper
  alone -- the PTX instructions and constants of its docstring, and with a disassembler the SASS facts the spellings exist
  for (the chain keeps both reciprocals under the zero guard and has no select; the convert has a zero upper operand).
* device (the SM100 line; ``tile_dsl_quant_helpers_probe.py``), bit-exact:
  - ``e4m3_round``: byte and value against the host cast and widening, on all 65,536 bf16 values but the NaN patterns and on
    fp32 values around every rounding boundary of the format.
  - ``rcp_approx_ftz``: against ``cute.arch.rcp_approx`` (the same instruction as a compiler intrinsic) on all 65,536 bf16
    values and a dense fp32 draw; against the host's IEEE reciprocal it is EXACT on every power of two and at most one unit in
    the last place off elsewhere (normal operand, normal result); the ``.ftz`` cases by value.
  - ``silu_ex2_div_approx``: against its four instructions issued as four statements on all 65,536 bf16 values; within a
    derived bracket of the fp64 SiLU where the block's documented accuracy holds; the saturated ends by value.
  - ``nvfp4_block_scale_global``: the byte against the host twin on every non-negative finite bf16 block maximum x 64 scales,
    a dense random draw and the association pairs; the encode scale against the reciprocal probe composed on the host
    (``rcp(value * rcp(scale))`` with the multiply on the host), and zero exactly where the code decodes to zero.
  - ``named_barrier_sync_unaligned``: a two-warp subset of a four-warp CTA hands a late store over through shared memory.
"""

import importlib
import math
import os
import struct
import subprocess
import sys

import pytest
import torch

from frost_test_utils import arch_known_to_the_dsl, nvdisasm_candidates, requires_blackwell, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]

_PROBE = "tile_dsl_quant_helpers_probe"
_PROBE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PROBE + ".py")
E4M3 = torch.float8_e4m3fn
SIXTH_BITS = 0x3E2AAAAB  # fp32(1/6): the bits of pointwise.opaque_fp4_max_rcp
SILU_CONSTANT_BITS = 0xBFB8AA3B  # fp32(-log2 e): the multiplier inside the SiLU block
FP32_MIN_NORMAL = 2.0**-126
ASSOCIATION_SCALES = (6.0, 3.0)  # scales with block maxima on which (scale * amax) * 1/6 and scale * (amax * 1/6) round to two codes


def _probe():
    """The probe module, imported lazily (its DSL imports must not run at collection on a box without the DSL)."""
    return importlib.import_module(_PROBE)


def _f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32)


def _all_bf16() -> torch.Tensor:
    """All 65,536 bf16 bit patterns widened exactly to fp32, in pattern order (the sign in the upper half)."""
    return (torch.arange(65536, dtype=torch.int32) << 16).view(torch.float32)


def _nonnegative_finite_bf16() -> torch.Tensor:
    return (torch.arange(0x7F80, dtype=torch.int32) << 16).view(torch.float32)


def _first(mask: torch.Tensor, *columns) -> str:
    """The first failing element as text: every column's value there, fp32 columns as hex bits."""
    k = int(mask.nonzero()[0])
    return ", ".join(hex(int(_bits(c)[k]) & 0xFFFFFFFF) if c.dtype == torch.float32 else str(int(c[k])) for c in columns)


# ---------------------------------------------------------------------------
# Host twins: the references of the device cells
# ---------------------------------------------------------------------------


def e4m3_round_twin(x: torch.Tensor):
    """``(byte, value)`` of ``e4m3_round`` on the host: torch's ``float32 -> float8_e4m3fn`` cast (round-to-nearest-even,
    saturating at +-448) and the exact widening of that code."""
    code = x.detach().to(torch.float32).cpu().to(E4M3)
    return code.view(torch.uint8).to(torch.int32), code.to(torch.float32)


def chain_twin(amax: torch.Tensor, scale: torch.Tensor):
    """``(byte, value)`` of the chain's block scale on the host: two IEEE fp32 multiplies, ``amax * fp32(1/6)`` FIRST."""
    sixth = torch.tensor(_f32(SIXTH_BITS), dtype=torch.float32)
    return e4m3_round_twin(scale.cpu() * (amax.cpu() * sixth))


def chain_other_association(amax: torch.Tensor, scale: torch.Tensor):
    sixth = torch.tensor(_f32(SIXTH_BITS), dtype=torch.float32)
    return e4m3_round_twin((scale.cpu() * amax.cpu()) * sixth)


def association_pairs():
    """``(amax, scale)`` pairs on which the two associations of the chain's product round to different E4M3 codes."""
    amax = _nonnegative_finite_bf16()
    found_a, found_s = [], []
    for s in ASSOCIATION_SCALES:
        scale = torch.full_like(amax, s)
        differ = chain_twin(amax, scale)[0] != chain_other_association(amax, scale)[0]
        found_a.append(amax[differ])
        found_s.append(scale[differ])
    return torch.cat(found_a), torch.cat(found_s)


def e4m3_boundary_values() -> torch.Tensor:
    """fp32 values on and next to every rounding boundary of E4M3: each positive code's value, the midpoint to the next code
    and both fp32 neighbours of each, both signs; the zero boundary ``2^-10``; the saturation band."""
    codes = torch.arange(0, 0x7F, dtype=torch.uint8).view(E4M3).to(torch.float32)  # 0 .. 448, increasing
    mid = (codes[:-1].double() + codes[1:].double()) / 2.0  # exact in fp64; every midpoint is an fp32 value
    pts = torch.cat([codes, mid.to(torch.float32)])
    up = torch.nextafter(pts, torch.full_like(pts, float("inf")))
    down = torch.nextafter(pts, torch.full_like(pts, -float("inf")))
    extra = torch.tensor([448.0, 449.0, 464.0, 480.0, 1e6, 3.0e38, float("inf"), 2.0**-10, 2.0**-11, 1e-30, 1.1754944e-38, 1e-45], dtype=torch.float32)
    extra = torch.cat([extra, torch.nextafter(extra, torch.full_like(extra, float("inf"))), torch.nextafter(extra, torch.zeros_like(extra))])
    pos = torch.cat([pts, up, down, extra])
    return torch.cat([pos, -pos]).contiguous()


def silu_fp64(x: torch.Tensor) -> torch.Tensor:
    x64 = x.double()
    return x64 / (1.0 + torch.exp(-x64))


def silu_margin(x: torch.Tensor) -> torch.Tensor:
    """The relative bracket of the block around the fp64 SiLU, in units of ``2^-23``, DERIVED from its four instructions:
    the exponent ``fl(x * fp32(-log2 e))`` carries half a unit of its own plus the constant's error ``2^-26.16`` -- together
    a relative error of ``0.62 |x|`` units on ``e = 2^(-x log2 e)`` -- and ``ex2.approx`` adds 2; both reach the result with
    the weight ``e / (1 + e)``; the add contributes half a unit and ``div.approx`` 2; one unit of slack."""
    x64 = x.double()
    weight = torch.sigmoid(-x64)
    return 4.0 + weight * (2.0 + 0.62 * x64.abs())


# ---------------------------------------------------------------------------
# Host cells
# ---------------------------------------------------------------------------


def test_constants_of_the_spellings():
    """The two constants a rendering carries: the SiLU block multiplies by fp32(-log2 e) = ``0xBFB8AA3B``; the chain by fp32(1/6)
    = ``0x3E2AAAAB`` (and the probe's four-statement reference uses the same multiplier)."""
    from cudnn.frost.tile_dsl.pointwise import E2M1_MAX_RCP_BITS

    assert struct.unpack("<I", struct.pack("<f", -math.log2(math.e)))[0] == SILU_CONSTANT_BITS
    assert struct.unpack("<I", struct.pack("<f", _probe().SILU_NEG_LOG2_E))[0] == SILU_CONSTANT_BITS
    assert E2M1_MAX_RCP_BITS == SIXTH_BITS == struct.unpack("<I", struct.pack("<f", 1.0 / 6.0))[0]


def test_e4m3_pair_twin_on_its_rounding_corners():
    """What ``(byte, value)`` means, pinned on the host twin the device cell compares with: zero up to ``2^-10`` (a tie that
    rounds to the even code, zero), the smallest code from the next fp32 value, ties to the even code, saturation at 448 for
    every larger finite value and for infinity, the sign carried by the code."""
    cases = [
        (0.0, 0x00, 0.0),
        (2.0**-10, 0x00, 0.0),
        (float(torch.nextafter(torch.tensor(2.0**-10), torch.tensor(1.0))), 0x01, 2.0**-9),
        (2.0**-9, 0x01, 2.0**-9),
        (1.5 * 2.0**-9, 0x02, 2.0**-8),
        (1.0, 0x38, 1.0),
        (1.0625, 0x38, 1.0),
        (1.1875, 0x3A, 1.25),
        (448.0, 0x7E, 448.0),
        (449.0, 0x7E, 448.0),
        (1.0e6, 0x7E, 448.0),
        (float("inf"), 0x7E, 448.0),
        (-1.0, 0xB8, -1.0),
        (-1.0e6, 0xFE, -448.0),
    ]
    byte, value = e4m3_round_twin(torch.tensor([c[0] for c in cases], dtype=torch.float32))
    assert byte.tolist() == [c[1] for c in cases]
    assert value.tolist() == [c[2] for c in cases]
    byte, value = e4m3_round_twin(torch.tensor([-0.0, -(2.0**-10)], dtype=torch.float32))
    assert byte.tolist() == [0x80, 0x80] and _bits(value).tolist() == [-(1 << 31)] * 2  # the zero code keeps the sign: -0.0


def test_chain_association_is_observable():
    """``scale * (amax * fp32(1/6))`` and ``(scale * amax) * fp32(1/6)`` are two functions: at each of the scales searched some
    bf16 block maxima round to two different codes.  The pairs feed the device cell, which then proves the association."""
    amax, scale = association_pairs()
    assert amax.numel() >= 2 * len(ASSOCIATION_SCALES), amax.numel()
    for s in ASSOCIATION_SCALES:
        assert bool((scale == s).any()), f"no bf16 block maximum separates the two associations at scale {s}"
    first = chain_twin(amax, scale)[0]
    other = chain_other_association(amax, scale)[0]
    assert bool((first != other).all())
    assert int((first - other).abs().max()) == 1  # neighbouring codes: a rounding boundary, not noise


# ---------------------------------------------------------------------------
# Host rendering: every helper alone
# ---------------------------------------------------------------------------

_UNSERVED = "Unsupported GPU architecture"  # the DSL's message when its compiler flavour has no such target


def _render(tmp_path, arch):
    dump = tmp_path / f"dump_{arch}"
    dump.mkdir()
    env = dict(os.environ, CUTE_DSL_DUMP_DIR=str(dump), CUTE_DSL_KEEP="ptx,cubin")
    proc = subprocess.run([sys.executable, _PROBE_PATH, arch, str(dump), *nvdisasm_candidates()], capture_output=True, text=True, timeout=900, env=env)
    return proc


@pytest.mark.parametrize("arch", ["sm_100a", "sm_107a"])
def test_rendering_of_every_helper(tmp_path, arch):
    """Each helper rendered alone for ``arch`` (no device of that class needed).  ASSERTED: the compile, and the PTX of each
    docstring.  With a disassembler that decodes the target, also the SASS: the chain's two reciprocals both carry the
    predicate of the zero guard and there is no select (the opaque spelling, not the intrinsic); the convert takes a zero
    upper operand; the subset barrier is a barrier instruction; nothing spills.  ``sm_107a`` is SKIPPED where the installed
    compiler flavour does not serve that target; the SASS half is skipped where no disassembler decodes the cubin."""
    if not arch_known_to_the_dsl(arch):
        pytest.skip(f"this cutlass-dsl has no {arch}")
    probe = _probe()
    proc = _render(tmp_path, arch)
    if proc.returncode != 0 and arch != "sm_100a" and _UNSERVED in (proc.stdout + proc.stderr):
        pytest.skip(f"the installed compiler flavour of the DSL does not serve {arch}")
    assert proc.returncode == 0, f"{arch} trace-compile failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    lines = proc.stdout.splitlines()
    for helper in probe.RENDER_HELPERS:
        assert f"COMPILED {helper} {arch}" in lines, proc.stdout[-2000:]
    pin = {h: {} for h in probe.RENDER_HELPERS}
    for ln in lines:
        if ln.startswith("PIN "):
            _, helper, key, count = ln.split()
            pin[helper][key] = int(count)
    zero = dict.fromkeys(next(iter(pin.values())), 0)
    want = {
        "rcp": dict(zero, rcp_approx_ftz=1),
        "e4m3": dict(zero, e4m3_convert=1, e4m3_widen=1, u32_from_u16=1, u16_from_u32=1),
        "silu": dict(zero, ex2_approx_ftz=1, div_approx=1, silu_constant=1),
        "chain": dict(zero, rcp_approx_ftz=2, e4m3_convert=1, e4m3_widen=1, u32_from_u16=1, u16_from_u32=1, sixth_bits=1),
        "barrier": dict(zero, barrier_sync_unaligned=1),
        "fence": dict(zero, bar_sync_aligned=1),
    }
    assert pin == want, f"{arch}: the rendered PTX is not the documented instruction set: {pin}"
    if not nvdisasm_candidates():
        pytest.skip("compiled + PTX pinned; no nvdisasm executable to try (CUDA_PATH unset and none on PATH)")
    if any(ln.startswith("SKIPSASS") for ln in lines):
        pytest.skip(f"compiled + PTX pinned; SASS half skipped: {[ln for ln in lines if ln.startswith(('SKIPSASS', 'REJECT'))][:4]}")
    sass = {h: {} for h in probe.RENDER_HELPERS}
    for ln in lines:
        if ln.startswith("SASS "):
            _, helper, key, count = ln.split()
            sass[helper][key] = int(count)
    print(f"\n{arch} SASS: {sass}")
    for helper in probe.RENDER_HELPERS:
        assert sass[helper]["spill"] == 0, (helper, sass[helper])
    assert sass["rcp"]["mufu_rcp"] == 1 and sass["rcp"]["fsel"] == 0, sass["rcp"]
    assert sass["e4m3"]["e4m3_convert"] == 1 and sass["e4m3"]["e4m3_convert_zero_upper"] == 1, sass["e4m3"]
    assert sass["silu"]["mufu_ex2"] == 1 and sass["silu"]["mufu_rcp"] == 1, sass["silu"]  # div.approx = one reciprocal and a multiply
    chain = sass["chain"]
    assert chain["mufu_rcp"] == 2 and chain["mufu_rcp_predicated"] == 2 and chain["fsel"] == 0, f"the zero guard is not a guarded region: {chain}"
    assert chain["e4m3_convert"] == 1 and chain["e4m3_convert_zero_upper"] == 1, chain
    assert sass["barrier"]["bar_sync"] == 1 and sass["fence"]["bar_sync"] == 1, (sass["barrier"], sass["fence"])


# ---------------------------------------------------------------------------
# Device cells (the SM100 line): raw bits against the named references
# ---------------------------------------------------------------------------


def _device():
    return torch.device("cuda", torch.cuda.current_device())


@requires_blackwell
def test_e4m3_round_is_the_cast_and_its_exact_widening():
    """Byte and value of ``e4m3_round`` equal the host twin (torch's cast and the widening of its code) bit for bit on every
    bf16 value that is not a NaN -- zeros of both signs, subnormals, the whole normal range, both infinities -- and on fp32
    values on and beside every rounding boundary of the format; the byte is always in ``[0, 255]``."""
    probe = _probe()
    x = torch.cat([_all_bf16(), e4m3_boundary_values()])
    rows, byte = probe.run_elementwise(x.to(_device()))
    byte, value = byte.cpu(), rows["e4m3_value"].cpu()
    assert int(byte.min()) >= 0 and int(byte.max()) <= 255
    want_byte, want_value = e4m3_round_twin(x)
    check = ~torch.isnan(x)
    assert int(check.sum()) == x.numel() - 254  # every bf16 NaN pattern (2 x 127) is left out; nothing else is
    bad = (byte != want_byte) & check
    assert not bool(bad.any()), f"{int(bad.sum())} bytes differ from the cast (x, byte, cast: {_first(bad, x, byte, want_byte)})"
    bad = (_bits(value) != _bits(want_value)) & check
    assert not bool(bad.any()), f"{int(bad.sum())} values differ from the widened code (x, value, twin: {_first(bad, x, value, want_value)})"


@requires_blackwell
def test_rcp_approx_ftz_is_the_hardware_reciprocal():
    """``rcp_approx_ftz`` on all 65,536 bf16 values and 2^20 fp32 values drawn log-uniformly over the normal range:

    * bitwise ``cute.arch.rcp_approx`` -- the opaque statement and the intrinsic are one function;
    * against the host's IEEE reciprocal (normal operand, normal result): EXACT on every power of two, never more than one
      unit in the last place off, and not the IEEE reciprocal everywhere (it is the approximate instruction);
    * ``.ftz``: a zero or subnormal operand gives the infinity of its sign, an operand whose reciprocal is subnormal or an
      infinite operand gives the zero of its sign."""
    probe = _probe()
    gen = torch.Generator(device="cpu").manual_seed(11)
    drawn = torch.exp2(torch.empty(1 << 20).uniform_(-126.0, 126.0, generator=gen)).to(torch.float32)
    drawn = torch.where(torch.rand(1 << 20, generator=gen) < 0.5, drawn, -drawn)
    x = torch.cat([_all_bf16(), drawn]).contiguous()
    rows, _ = probe.run_elementwise(x.to(_device()))
    got, intrinsic = rows["rcp"].cpu(), rows["rcp_intrinsic"].cpu()
    bad = _bits(got) != _bits(intrinsic)
    assert not bool(bad.any()), f"{int(bad.sum())} results differ from cute.arch.rcp_approx (x, helper, intrinsic: {_first(bad, x, got, intrinsic)})"
    ieee = torch.ones((), dtype=torch.float32) / x  # the host's IEEE division: independent of any GPU arithmetic
    finite = torch.isfinite(x)
    normal = finite & (x.abs() >= FP32_MIN_NORMAL) & torch.isfinite(ieee) & (ieee.abs() >= FP32_MIN_NORMAL)
    ulps = (_bits(got).to(torch.int64) - _bits(ieee).to(torch.int64)).abs()
    assert int(normal.sum()) > (1 << 20), int(normal.sum())
    bad = normal & (ulps > 1)
    assert not bool(bad.any()), f"{int(bad.sum())} reciprocals are more than one unit from the IEEE value (x, helper, IEEE: {_first(bad, x, got, ieee)})"
    assert int((normal & (ulps == 1)).sum()) > 0, "bitwise the IEEE reciprocal on every operand: not the approximate instruction this helper names"
    power_of_two = normal & ((_bits(x) & 0x007FFFFF) == 0)
    assert int(power_of_two.sum()) >= 2 * 250  # both signs of the bf16 powers of two whose reciprocal is normal
    bad = power_of_two & (ulps != 0)
    assert not bool(bad.any()), f"{int(bad.sum())} powers of two are not inverted exactly (x, helper: {_first(bad, x, got)})"
    same_sign = torch.signbit(got) == torch.signbit(x)
    to_infinity = finite & (x.abs() < FP32_MIN_NORMAL)  # zeros and subnormal operands
    assert int(to_infinity.sum()) == 2 * 128 and bool((torch.isinf(got) & same_sign)[to_infinity].all())
    to_zero = (finite & (x.abs() >= FP32_MIN_NORMAL) & (ieee.abs() < FP32_MIN_NORMAL)) | torch.isinf(x)
    assert int(to_zero.sum()) > 2 and bool(((got == 0) & same_sign)[to_zero].all())


@requires_blackwell
def test_silu_ex2_div_approx_is_its_four_instructions():
    """``silu_ex2_div_approx`` on all 65,536 bf16 values:

    * bitwise the four instructions issued as four statements (the multiply by fp32(-log2 e), ``ex2.approx.ftz.f32``, the add
      of 1, ``div.approx.f32``) on every value that is not a NaN: the block is exactly that chain, nothing contracted;
    * for ``|x| <= 87`` within the bracket ``silu_margin`` derives around the fp64 SiLU -- it IS the SiLU;
    * the ends: ``x >= 89`` returns ``x`` (the exponential is flushed to zero), ``x <= -89`` returns ``-0`` (it overflows),
      a zero returns itself."""
    probe = _probe()
    x = _all_bf16()
    rows, _ = probe.run_elementwise(x.to(_device()))
    got, composed = rows["silu"].cpu(), rows["silu_four_statements"].cpu()
    check = ~torch.isnan(x)
    bad = (_bits(got) != _bits(composed)) & check
    assert not bool(bad.any()), f"{int(bad.sum())} results differ from the four statements (x, block, statements: {_first(bad, x, got, composed)})"
    finite = torch.isfinite(x)
    core = finite & (x.abs() <= 87.0) & (x != 0)
    assert int(core.sum()) > 27000
    exact = silu_fp64(x)
    deviation = (got.double() - exact).abs() / exact.abs().clamp_min(1e-300) * 2.0**23
    bad = core & (deviation > silu_margin(x))
    assert not bool(
        bad.any()
    ), f"{int(bad.sum())} results leave the derived bracket of the fp64 SiLU (x, block: {_first(bad, x, got)}; units {float(deviation[bad].max()):.2f})"
    high = finite & (x >= 89.0)
    assert int(high.sum()) > 0 and bool((_bits(got) == _bits(x))[high].all())
    low = finite & (x <= -89.0)
    assert int(low.sum()) > 0 and bool((_bits(got)[low] == -(1 << 31)).all())
    zeros = x == 0
    assert int(zeros.sum()) == 2 and bool((_bits(got) == _bits(x))[zeros].all())


@requires_blackwell
def test_nvfp4_block_scale_global_is_the_documented_chain():
    """``nvfp4_block_scale_global`` on every non-negative finite bf16 block maximum x 64 scales in ``[2^-100, 2^100]``, 2^20
    random pairs, and the pairs that separate the two associations of the product:

    * the byte is the host twin's (two IEEE fp32 multiplies, ``amax * fp32(1/6)`` first, then the cast) -- which the
      association pairs make a proof of the association, and of the bits of the 1/6;
    * the encode scale is zero exactly where the code decodes to zero, never an infinity or a NaN;
    * elsewhere it is ``rcp(value * rcp(scale))`` bit for bit, composed on the host from the reciprocal probe (the multiply
      is an IEEE multiply on the host): two reciprocals and one multiply, nothing fused or reassociated."""
    probe = _probe()
    dev = _device()
    gen = torch.Generator(device="cpu").manual_seed(12)
    grid_amax = _nonnegative_finite_bf16()
    scales = torch.cat([torch.exp2(torch.empty(40).uniform_(-100.0, 100.0, generator=gen)), torch.exp2(torch.empty(16).uniform_(-8.0, 8.0, generator=gen))])
    scales = torch.cat([scales, torch.tensor([1.0, 2.0**-100, 2.0**100, 6.0, 1.0 / 6.0, 448.0, 0.1, 37.5])]).to(torch.float32)
    assert scales.numel() == 64
    drawn_amax = (torch.randint(0, 0x7F80, (1 << 20,), generator=gen, dtype=torch.int32) << 16).view(torch.float32)
    drawn_scale = torch.exp2(torch.empty(1 << 20).uniform_(-100.0, 100.0, generator=gen)).to(torch.float32)
    pair_amax, pair_scale = association_pairs()
    amax = torch.cat([grid_amax.repeat_interleave(64), drawn_amax, pair_amax]).contiguous()
    scale = torch.cat([scales.repeat(grid_amax.numel()), drawn_scale, pair_scale]).contiguous()
    byte, encode = probe.run_chain(amax.to(dev), scale.to(dev))
    byte, encode = byte.cpu(), encode.cpu()
    want_byte, value = chain_twin(amax, scale)
    bad = byte != want_byte
    assert not bool(bad.any()), f"{int(bad.sum())} bytes differ from the host twin (amax, scale, byte, twin: {_first(bad, amax, scale, byte, want_byte)})"
    other = chain_other_association(amax, scale)[0]
    assert int((byte != other).sum()) >= pair_amax.numel(), "no pair told the two associations apart"
    zero = value == 0
    assert int(zero.sum()) > 1000 and int((~zero).sum()) > 1000  # both arms of the guard are exercised
    assert bool((_bits(encode)[zero] == 0).all()), "a zero block scale must give a zero multiplier (+0)"
    assert bool(torch.isfinite(encode).all())
    reciprocal_of_scale = probe.run_elementwise(scale.to(dev))[0]["rcp"].cpu()
    product = (value * reciprocal_of_scale).contiguous()  # an IEEE fp32 multiply on the host
    composed = probe.run_elementwise(product.to(dev))[0]["rcp"].cpu()
    bad = (_bits(encode) != _bits(composed)) & ~zero
    assert not bool(
        bad.any()
    ), f"{int(bad.sum())} encode scales differ from rcp(value * rcp(scale)) (amax, scale, helper, composed: {_first(bad, amax, scale, encode, composed)})"


@requires_blackwell
def test_named_barrier_sync_unaligned_synchronizes_a_warp_subset():
    """Two of a CTA's four warps -- a writer that stores late and a reader -- meet on ``named_barrier_sync_unaligned`` from
    their own warp-role branches; the two other warps never reach it.  In 50 launches of 2,048 CTAs the reader copies out
    exactly what the writer stored (the host twin: the same chain, the add on the host and the reciprocal from the probe)."""
    probe = _probe()
    dev = _device()
    gen = torch.Generator(device="cpu").manual_seed(13)
    blocks = 2048
    src = (torch.rand(blocks * 32, generator=gen) + 0.5).to(torch.float32).to(dev).contiguous()
    want = src.clone()
    for _ in range(probe.DELAY):
        want = probe.run_elementwise((want + 1.0).contiguous())[0]["rcp"].clone()
    assert bool(torch.isfinite(want).all()) and not torch.equal(want, src)
    for launch in range(50):
        got = probe.run_barrier(src)
        bad = _bits(got) != _bits(want)
        assert not bool(bad.any()), f"launch {launch}: {int(bad.sum())} of {got.numel()} elements read behind the barrier are not what the writer stored"
