# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fp4 (e2m1) block-quantization primitives of ``cudnn.frost.tile_dsl.pointwise``:
``fp32_to_fp4_pack``, ``opaque_fp4_max_rcp``, ``e8m0_from_amax(inv_max=)``,
``e4m3_scale_from_amax``, ``div_rn_f32``.

Two tiers, neither needs a cc 10.x device:

* tier 1, pure torch / Python: the constants the kernel and the oracle must share
  (``fp32(1/6)`` = ``0x3E2AAAAB``, the e4m3 ``2^-9`` scale floor and its rounding
  corners), the e2m1 round-to-nearest-even table on EVERY midpoint -- a torch emulation
  of ``cvt.rn.satfinite.e2m1x2.f32`` pinned against an independent port (torchao's
  ``_f32_to_floatx_unpacked``) -- and the reciprocal-multiply midpoint flip that makes
  ``div.rn.f32`` load-bearing;
* tier 2, an sm_107a trace-compile of a tiny kernel that uses every primitive (any box whose
  cutlass-dsl knows ``sm_107a``; no device match needed), and -- when ``$CUDA_PATH/bin`` or
  ``$PATH`` carries an ``nvdisasm`` that decodes it -- the SASS: zero spills, the hardware
  ``F2FP`` converts for e2m1 / e4m3 / ue8m0, and for the NVFP4 arm one ``FCHK`` per division
  (the ``div.rn.f32`` RCP + FFMA fixup, never a bare ``MUFU.RCP``).  The compile itself is
  asserted; only the SASS half is a SKIP when no candidate nvdisasm decodes the cubin.

The kernel-vs-oracle ``torch.equal`` on a Rubin device belongs to the quantize kernel's own
suite (``fe_api/gated_attention_block/test_quantize_fp4.py``); this module holds what that test
takes as given.
"""

import glob
import importlib.util
import os
import shutil
import struct
import subprocess
import sys
import textwrap

import pytest
import torch

from frost_test_utils import _dsl_installed, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]

if _dsl_installed():
    from cudnn.frost.tile_dsl.pointwise import (
        E2M1_MAX_RCP_BITS,
        E2M1_PER_CVT,
        E4M3_MIN_SUBNORMAL_BITS,
        E8M0_RCP_E4M3_MAX_BITS,
        fp32_to_fp4_pack,
        opaque_f32_bits,
    )

_TEST_PYTHON = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# e2m1: sign(1) exponent(2) mantissa(1); codes 0..7 in value order, 8..15 their negatives.
E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MAX = 6.0
# Every rounding midpoint of the positive grid, and the even CODE each must round to.
E2M1_MIDPOINTS = {0.25: 0, 0.75: 2, 1.25: 2, 1.75: 4, 2.5: 4, 3.5: 6, 5.0: 6}


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _bits_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


# ---------------------------------------------------------------------------
# Torch emulations of the hardware ops (what the Rubin tier-3 test compares codes against)
# ---------------------------------------------------------------------------


def e2m1_rne_codes(x: torch.Tensor) -> torch.Tensor:
    """``cvt.rn.satfinite.e2m1x2.f32`` per element: the 4-bit e2m1 code (uint8, low nibble).

    Nearest grid value, a tie goes to the EVEN code, |x| >= 6 saturates to code 7 (6.0), the sign
    bit is kept (so -0.0 -> 0x8 like the hardware).  fp32 in; no NaN handling (out of contract)."""
    xf = x.to(torch.float32)
    mag = xf.abs().clamp(max=E2M1_MAX)
    # index of the largest grid value <= |x|, and its successor
    lo = torch.searchsorted(E2M1_GRID, mag, right=True) - 1
    lo = lo.clamp(0, len(E2M1_GRID) - 1)
    hi = (lo + 1).clamp(max=len(E2M1_GRID) - 1)
    d_lo = mag - E2M1_GRID[lo]
    d_hi = E2M1_GRID[hi] - mag
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (hi % 2 == 0))
    code = torch.where(pick_hi, hi, lo).to(torch.uint8)
    sign = (xf.view(torch.int32) < 0).to(torch.uint8) << 3
    return code | sign


def e4m3_scale_emulation(amax: torch.Tensor) -> torch.Tensor:
    """``e4m3_scale_from_amax``: ``e4m3_rn(max(amax * fp32(1/6), 2^-9))`` -- the SAME constant bits."""
    sixth = torch.tensor(_bits_f32(E2M1_MAX_RCP_BITS), dtype=torch.float32)
    floor = torch.tensor(_bits_f32(E4M3_MIN_SUBNORMAL_BITS), dtype=torch.float32)
    return torch.maximum(amax.to(torch.float32) * sixth, floor).to(torch.float8_e4m3fn)


def _torchao_e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """The independent reference: torchao's ``_f32_to_floatx_unpacked(x, 2, 1)`` as vendored in
    ``test/python/test_low_precision_matmul.py`` (path-loaded here, not collected)."""
    path = os.path.join(_TEST_PYTHON, "test_low_precision_matmul.py")
    if _TEST_PYTHON not in sys.path:
        sys.path.insert(0, _TEST_PYTHON)  # its ``import test_utils``
    spec = importlib.util.spec_from_file_location("_fp4_torchao_port_for_pointwise_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._f32_to_floatx_unpacked(x.to(torch.float32).contiguous(), 2, 1)


# ---------------------------------------------------------------------------
# Tier 1: constants and rounding corners
# ---------------------------------------------------------------------------


def test_fp4_max_rcp_is_fp32_one_sixth():
    """``opaque_fp4_max_rcp`` carries fp32(1/6) = 0x3E2AAAAB: the ONE constant the kernel and the oracle multiply by."""
    assert E2M1_MAX_RCP_BITS == 0x3E2AAAAB
    assert _f32_bits(1.0 / E2M1_MAX) == E2M1_MAX_RCP_BITS
    assert torch.tensor(1.0 / E2M1_MAX, dtype=torch.float32).view(torch.int32).item() == E2M1_MAX_RCP_BITS
    # the MXFP8 twin is untouched by the parametrisation
    assert E8M0_RCP_E4M3_MAX_BITS == _f32_bits(1.0 / 448.0) == 0x3B124925


def test_multiply_by_one_sixth_is_not_divide_by_six():
    """The two spellings differ by one fp32 ulp on ~1/3 of inputs, i.e. across e4m3 / e8m0 rounding corners --
    which is why BOTH sides spell the scale as ``amax * fp32(1/6)``."""
    g = torch.Generator().manual_seed(0)
    amax = torch.rand(100_000, generator=g) * 3000.0
    sixth = torch.tensor(_bits_f32(E2M1_MAX_RCP_BITS), dtype=torch.float32)
    differ = ((amax * sixth) != (amax / 6.0)).float().mean().item()
    assert 0.2 < differ < 0.5, differ


def test_e4m3_scale_floor_is_the_smallest_nonzero_e4m3():
    """``2^-9`` = 0x3B000000 is e4m3's minimum subnormal: one binade below rounds to ZERO (the infinite-encode
    hazard the floor exists for), the floor itself is code 0x01, and the subnormal ties round to the even code."""
    assert E4M3_MIN_SUBNORMAL_BITS == 0x3B000000 == _f32_bits(2.0**-9)
    e4m3 = torch.float8_e4m3fn

    def cast(v):
        t = torch.tensor(v, dtype=torch.float32).to(e4m3)
        return t.float().item(), t.view(torch.uint8).item()

    assert cast(2.0**-10) == (0.0, 0x00)  # below the floor: zero
    assert cast(2.0**-9) == (2.0**-9, 0x01)  # the floor: smallest nonzero
    assert cast(1.5 * 2.0**-9) == (2.0**-8, 0x02)  # tie 0x01 | 0x02 -> even
    assert cast(2.5 * 2.0**-9) == (2.0**-8, 0x02)  # tie 0x02 | 0x03 -> even
    assert cast(3.0 * 2.0**-9) == (3.0 * 2.0**-9, 0x03)
    assert cast(448.0) == (448.0, 0x7E)  # e4m3 max normal; the hardware's satfinite ceiling


@pytest.mark.parametrize("amax", [0.0, 2.0**-30, 6.0 * 2.0**-10, 6.0 * 2.0**-9, 1e-3, 1.0, 100.0, 6.0 * 448.0, 6.0 * 449.0, 5000.0])
def test_e4m3_scale_emulation_is_nonzero_and_saturates(amax):
    """The scale the NVFP4 arm produces (emulated with the kernel's constants) is never 0 -- an all-zero block
    encodes as 0 / sf = 0, not NaN -- and never above 448."""
    sf = e4m3_scale_emulation(torch.tensor([amax])).float().item()
    assert sf > 0.0, (amax, sf)
    assert sf <= 448.0, (amax, sf)
    if amax == 0.0:
        assert sf == 2.0**-9
    if amax >= 6.0 * 448.0:
        assert sf == 448.0


def test_e2m1_rne_table_on_every_midpoint():
    """Nearest on ``{0, .5, 1, 1.5, 2, 3, 4, 6}``, ties to the EVEN code: 0.25->0, 0.75->1, 1.25->1, 1.75->2,
    2.5->2, 3.5->4, 5->4; |x| >= 6 saturates; the sign bit rides along."""
    for mid, code in E2M1_MIDPOINTS.items():
        got = e2m1_rne_codes(torch.tensor([mid, -mid]))
        assert got.tolist() == [code, code | 0x8], (mid, got.tolist())
    grid = e2m1_rne_codes(E2M1_GRID)
    assert grid.tolist() == list(range(8))
    assert e2m1_rne_codes(torch.tensor([6.0, 6.5, 100.0, 1e30, -7.0])).tolist() == [7, 7, 7, 7, 15]
    assert e2m1_rne_codes(torch.tensor([0.1, 0.2, 0.3, 0.26, 0.74, 0.76, -0.0])).tolist() == [0, 0, 1, 1, 1, 2, 8]


def test_e2m1_rne_emulation_matches_the_torchao_port():
    """The emulation above vs an independent implementation, on every midpoint (both signs), the grid, the
    saturation band and a dense random sweep -- so the Rubin ``torch.equal`` test compares the kernel against
    a rounding rule two implementations agree on."""
    pts = [m for m in E2M1_MIDPOINTS] + E2M1_GRID.tolist() + [6.5, 7.0, 100.0, 0.1, 0.2, 0.3, 0.26, 0.74, 0.76]
    x = torch.tensor(pts + [-p for p in pts], dtype=torch.float32)
    g = torch.Generator().manual_seed(1)
    x = torch.cat([x, (torch.rand(200_000, generator=g) - 0.5) * 16.0])
    ref = _torchao_e2m1_codes(x).to(torch.uint8)
    ours = e2m1_rne_codes(x)  # both keep the sign of -0.0 (code 0x8), as the hardware does
    mism = (ref != ours).nonzero().flatten()
    assert mism.numel() == 0, f"{mism.numel()} mismatches, first: x={x[mism[:8]].tolist()} ref={ref[mism[:8]].tolist()} ours={ours[mism[:8]].tolist()}"


def test_reciprocal_multiply_flips_an_exact_e2m1_midpoint():
    """WHY ``div_rn_f32``: ``0.5859375 / 0.46875`` is EXACTLY 1.25 (a tie -> code 1.0), while
    ``0.5859375 * fp32(1 / 0.46875)`` is 1.2500001 (-> code 1.5).  Same fp32 inputs, a whole code apart."""
    x = torch.tensor(0.5859375, dtype=torch.float32)
    sf = torch.tensor(0.46875, dtype=torch.float32)  # an e4m3 value (0.46875 = 15/32)
    assert sf.to(torch.float8_e4m3fn).float().item() == 0.46875
    q_div = x / sf
    q_rcp = x * (torch.tensor(1.0, dtype=torch.float32) / sf)
    assert q_div.item() == 1.25
    assert q_rcp.item() != 1.25 and q_rcp.view(torch.int32).item() == 0x3FA00001
    assert e2m1_rne_codes(q_div).item() == 2  # 1.0 (tie -> even)
    assert e2m1_rne_codes(q_rcp).item() == 3  # 1.5


def test_e2m1_pack_convention_low_nibble_is_the_even_element():
    """The byte order ``fp32_to_fp4_pack`` documents: byte i = code(x[2i]) | code(x[2i+1]) << 4, word 0 = bytes 0..3.
    Pinned here as the emulation the Rubin bitwise test packs its oracle with."""
    codes = torch.arange(16, dtype=torch.uint8)  # element k -> code k
    packed = (codes[0::2] | (codes[1::2] << 4)).tolist()
    assert packed == [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]
    words = [int.from_bytes(bytes(packed[4 * w : 4 * w + 4]), "little") for w in range(2)]
    assert words == [0x76543210, 0xFEDCBA98]
    assert E2M1_PER_CVT == 2 and 16 // E2M1_PER_CVT == 8  # eight cvts per 16-element call


def test_fp32_to_fp4_pack_refuses_a_wrong_count():
    """Typed at trace time, before any asm is emitted (no DSL context needed to hit it)."""
    with pytest.raises(ValueError, match="expected 16 input values, got 15"):
        fp32_to_fp4_pack([None] * 15)
    with pytest.raises(ValueError, match="expected 16 input values, got 32"):
        fp32_to_fp4_pack([None] * 32)


@pytest.mark.parametrize("bits", [-1, 1 << 32, 0x1_3E2AAAAB])
def test_opaque_f32_bits_refuses_a_non_32bit_pattern(bits):
    with pytest.raises(ValueError, match="32-bit pattern"):
        opaque_f32_bits(bits)


# ---------------------------------------------------------------------------
# Tier 2: sm_107a trace-compile of every primitive + SASS
# ---------------------------------------------------------------------------


def _nvdisasm_candidates():
    """Executables to TRY, most likely to decode sm_107a first; the probe verifies each and skips if none does.

    A public CUDA 13.x ``nvdisasm`` reports ``Cannot decode architecture 'SM107a'``, so the toolkit
    ``$CUDA_PATH`` names is tried first and the one on ``$PATH`` second -- both are hints, not authorities."""
    cands = []
    if os.environ.get("CUDA_PATH"):
        cands.append(os.path.join(os.environ["CUDA_PATH"], "bin", "nvdisasm"))
    on_path = shutil.which("nvdisasm")
    if on_path:
        cands.append(on_path)
    return [c for c in dict.fromkeys(cands) if os.path.isfile(c) and os.access(c, os.X_OK)]


def _sm107a_known_to_the_dsl() -> bool:
    """cutlass-dsl < 4.8.0.dev0 has no ``sm_107a`` (``Arch.from_string`` KeyError); FROST's floor is 4.7.0."""
    try:
        from cutlass.base_dsl.enums import Arch

        Arch.from_string("sm_107a")
        return True
    except Exception:
        return False


N_ELEMS = 16  # one fp32_to_fp4_pack call; the NVFP4 arm divides each element once
_SASS_KEYS = ("SPILL", "E2M1", "E4M3", "E4M3_UNPACK", "E8M0", "FCHK", "MUFU_RCP", "FFMA", "FMNMX3", "LINES")

_SASS_PROBE = textwrap.dedent("""
    import glob, os, subprocess, sys
    mode, dump, cands = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump  # read once, at the first cutlass import
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream
    from cudnn.frost.tile_dsl.pointwise import abs_max_tree, div_rn_f32, e4m3_scale_from_amax, e8m0_from_amax, fp32_to_fp4_pack, opaque_fp4_max_rcp
    from cudnn.frost.tile_dsl.tma import ld_global_v4, st_global, st_global_v2

    N = %(n)d

    @cute.kernel
    def probe(mSrc: cute.Tensor, mDst: cute.Tensor, nvfp4: cutlass.Constexpr[bool]):
        # one lane = one block of N fp32: amax -> scale -> codes, exactly the quantizer's per-lane body
        tidx = cutlass.Int32(cute.arch.thread_idx()[0])
        src = mSrc.iterator.toint() + tidx.to(cutlass.Int64) * cutlass.Int64(N * 4)
        dst = mDst.iterator.toint() + tidx.to(cutlass.Int64) * cutlass.Int64(16)
        vals = []
        for j in cutlass.range_constexpr(N // 4):
            for w in ld_global_v4(src + cutlass.Int64(j * 16), cutlass.Float32):
                vals.append(w)
        amax = abs_max_tree(vals)
        q = []
        if cutlass.const_expr(nvfp4):
            sf, byte = e4m3_scale_from_amax(amax)
            for i in cutlass.range_constexpr(N):
                q.append(div_rn_f32(vals[i], sf))
        else:
            rcp, byte = e8m0_from_amax(amax, inv_max=opaque_fp4_max_rcp())
            for i in cutlass.range_constexpr(N):
                q.append(vals[i] * rcp)
        packed = fp32_to_fp4_pack(q)
        st_global_v2(dst, [packed[0], packed[1]], cutlass.Int32)
        st_global(dst + cutlass.Int64(8), byte, cutlass.Int32)

    @cute.jit
    def launch(src: cute.Tensor, dst: cute.Tensor, nvfp4: cutlass.Constexpr[bool], stream: cuda.CUstream):
        probe(src, dst, nvfp4).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    src = cute.runtime.make_fake_tensor(dtype=cutlass.Float32, shape=(cute.sym_int(),), stride=(1,), assumed_align=16)
    dst = cute.runtime.make_fake_tensor(dtype=cutlass.Int32, shape=(cute.sym_int(),), stride=(1,), assumed_align=16)
    # --keep-cubin, NOT --keep-sass: the latter runs the DSL's own wheel nvdisasm, which ICEs on sm_107a.
    cute.compile(launch, src, dst, mode == "nvfp4", make_fake_stream(use_tvm_ffi_env_stream=False), options="--enable-tvm-ffi --gpu-arch sm_107a --keep-cubin")
    print("COMPILED", mode)
    cubins = glob.glob(os.path.join(dump, "*.sm_107a.cubin"))
    if not cubins:
        print("FAIL no .sm_107a.cubin landed in", dump)
        sys.exit(3)
    sass = None
    for nvd in cands:
        try:
            proc = subprocess.run([nvd, "-c", cubins[0]], capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as exc:  # vanished / not executable / wedged: the next candidate
            print("REJECT", nvd, "->", repr(exc))
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            sass = proc.stdout.splitlines()
            print("NVDISASM", nvd)
            break
        print("REJECT", nvd, "->", (proc.stderr.strip().splitlines() or [str(proc.returncode)])[-1])
    if sass is None:
        print("SKIP no nvdisasm candidate decodes sm_107a")
        sys.exit(0)
    print("SPILL", sum(1 for ln in sass if "STL" in ln or "LDL" in ln))
    print("E2M1", sum(1 for ln in sass if "F2FP" in ln and "E2M1" in ln))
    print("E4M3", sum(1 for ln in sass if "F2FP" in ln and "E4M3.F32" in ln))
    print("E4M3_UNPACK", sum(1 for ln in sass if "F2FP" in ln and "F16.E4M3" in ln))
    print("E8M0", sum(1 for ln in sass if "F2FP" in ln and ".E8." in ln and ".RP" in ln))
    print("FCHK", sum(1 for ln in sass if "FCHK" in ln))
    print("MUFU_RCP", sum(1 for ln in sass if "MUFU.RCP" in ln))
    print("FFMA", sum(1 for ln in sass if "FFMA" in ln))
    print("FMNMX3", sum(1 for ln in sass if "FMNMX3" in ln))
    print("LINES", len(sass))
    """) % {"n": N_ELEMS}


@pytest.mark.parametrize("mode", ["nvfp4", "mxfp4"])
def test_sm107_trace_compile_of_every_primitive(mode, tmp_path):
    """Compile a kernel that uses every fp4 primitive for Rubin here (no device match needed) -- that half is
    ASSERTED on any box whose DSL knows ``sm_107a``.  With an nvdisasm that decodes it, also the SASS:

    * ``SPILL == 0``;
    * the codes come from the hardware ``F2FP...E2M1`` (8 per 16 elements -- a LOWER bound, ptxas may fuse);
    * NVFP4: the scale is a hardware ``F2FP...E4M3.F32`` widened by ``F2FP.F16.E4M3``, and the division is the
      IEEE ``div.rn.f32`` lowering -- ONE ``FCHK`` per element, at least as many ``MUFU.RCP`` seeds and a
      multi-``FFMA`` Newton fixup -- never a bare ``MUFU.RCP`` (that would be an approximate reciprocal-multiply,
      which flips e2m1 midpoints);
    * MXFP4: the scale is the ``cvt.rp...ue8m0x2`` (``F2FP...E8...RP``) and there is NO division at all.

    SKIPS the SASS half (never fails) when no candidate nvdisasm decodes ``sm_107a`` -- an L0 verdict must not
    turn on which toolkit ``CUDA_PATH`` happens to name."""
    if not _sm107a_known_to_the_dsl():
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0.dev0, --pre)")
    cands = _nvdisasm_candidates()
    dump = tmp_path / f"fp4_primitives_{mode}"
    dump.mkdir()
    # a FILE, not ``python -c``: the DSL preprocessor re-reads a kernel's source through ``inspect.getsource``
    probe_py = tmp_path / f"fp4_primitives_probe_{mode}.py"
    probe_py.write_text(_SASS_PROBE)
    proc = subprocess.run([sys.executable, str(probe_py), mode, str(dump), *cands], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"trace-compile failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    lines = proc.stdout.splitlines()
    assert f"COMPILED {mode}" in lines, proc.stdout[-2000:]
    assert glob.glob(str(dump / "*.sm_107a.cubin")), "no .sm_107a.cubin landed in CUTE_DSL_DUMP_DIR"
    if not cands:
        pytest.skip("compiled; no nvdisasm executable to try for the SASS half (CUDA_PATH unset and none on PATH)")
    if any(ln.startswith("SKIP") for ln in lines):
        pytest.skip(f"compiled; SASS half skipped: {[ln for ln in lines if ln.startswith(('SKIP', 'REJECT'))]}")
    stats = {k: int(v) for k, v in (ln.split() for ln in lines if ln.split() and ln.split()[0] in _SASS_KEYS)}
    print(f"\n[{mode}] sm_107a SASS: {stats} via {[ln for ln in lines if ln.startswith('NVDISASM')]}")
    assert stats["SPILL"] == 0, f"{mode}: {stats['SPILL']} STL/LDL in the sm_107a cubin"
    assert stats["E2M1"] >= N_ELEMS // E2M1_PER_CVT, f"{mode}: fewer e2m1 cvts than the source issues -- {stats}"
    assert stats["FMNMX3"] > 0, stats
    if mode == "nvfp4":
        assert stats["E4M3"] >= 1 and stats["E4M3_UNPACK"] >= 1 and stats["E8M0"] == 0, stats
        assert stats["FCHK"] == N_ELEMS, f"nvfp4: expected one FCHK (div.rn range check) per element -- {stats}"
        assert stats["MUFU_RCP"] >= stats["FCHK"], f"nvfp4: every FCHK pairs with a MUFU.RCP seed -- {stats}"
        assert stats["FFMA"] >= 4 * N_ELEMS, f"nvfp4: the div.rn Newton fixup is >= 4 FFMA per division -- {stats}"
    else:
        assert stats["E8M0"] >= 1 and stats["E4M3"] == 0, stats
        assert stats["FCHK"] == 0 and stats["MUFU_RCP"] == 0, f"mxfp4: a power-of-two scale needs no division -- {stats}"
