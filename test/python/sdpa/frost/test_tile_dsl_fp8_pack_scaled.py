# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``tile_dsl.pointwise.fp32_to_fp8_pack_scaled`` / ``fp32_to_fp8x2_scaled`` -- Rubin's fused DEscale-and-pack
``cvt.rn.satfinite.scaled::n1::ue8m0.{e4m3,e5m2}x2.f32`` behind one helper with a portable FMUL arm.

The contract under test (MEASURED on Rubin, cc 10.7, 204 SMs; the instruction probe's records are retained internally):
``byte_i = fp8_rn_satfinite(ftz(x_i) * 2^(127 - e))`` -- **the ``.b8`` operand is a DEscale exponent, so the block's E8M0
scale byte (``e8m0_from_amax(...)[1]``) is passed AS IS** and the hardware divides; ``fused=False`` is the bit-identical
``x * e8m0_rcp(e)`` + ``fp32_to_fp8_pack`` spelling for normal fp32 inputs and ``e <= 253``.

**The ``.b8`` scale has a PROVENANCE rule** (MEASURED 2026-09-30 on the DSL's libnvptxcompiler and a second standalone ptxas, sm_107a;
``pointwise.py`` header): per-lane DATA (a load, an ``e8m0_from_amax`` byte) and the ``cvt.u8.u32`` IMMEDIATE assemble; a
register ptxas can prove constant (``opaque_i32(Int32(119))``) or a KERNEL PARAMETER ICEs ptxas (C7907).  The probe therefore
takes its byte from every production provenance (``SRC_*``): ``gmem`` (the block's byte from memory), ``pyint`` /
``traced_const`` (P's fixed byte 119 as a Python int / a constant ``Int32`` -> the helper's immediate form), ``smem`` (an SF
atom the kernel filled itself, ``sP_SF``), ``amax`` (``e8m0_from_amax(abs_max_tree(block))`` -- the dS
quantizer), ``redux`` (``e8m0_from_amax(warp_abs_max_f32(x))`` -- the along-kv column scale of dS), ``param`` (a kernel scalar:
the tripwire that must keep ICEing until ptxas is fixed).

Three tiers, one probe kernel (one lane = one 16-value block + the SWAPPED first pair -- both entry points, one scale byte;
swapped because identical operands let ptxas CSE the pair's cvt with the pack's, MEASURED: 8 not 9 SCALE_BY_C):

* tier 1, any box whose cutlass-dsl knows ``sm_107a`` (no device match needed): the sm_107a trace-compile of the probe
  is ASSERTED, and with an ``nvdisasm`` that decodes the cubin (``$CUDA_PATH/bin`` or ``$PATH``; the DSL's own wheel one
  may not) the SASS pins from THAT kernel: fused -> 8 ``F2FP.SATFINITE.<FMT>.F32.PACK_AB_MERGE_C.SCALE_BY_C`` per
  16-value call (+1 for the pair twin), ZERO ``FMUL`` and zero unscaled ``F2FP``; unfused -> 16..18 ``FMUL`` (one per
  distinct element -- the pair's two products CSE with the pack's on this toolchain, a bound not a literal) and 8 (+1)
  unscaled ``F2FP``, zero ``SCALE_BY_C``.  Per provenance: the constant sources carry ``cvt.u8.u32 sf, 119;`` in the PTX,
  ``smem`` an ``LDS``, ``redux`` a ``REDUX``; ``param`` FAILS in ptxas with C7907 (the tripwire).  The SASS half SKIPS
  without a decoding nvdisasm; a compile failure is a FAIL.
* tier 1b, the arch gate: ``fused=True`` traced for ``sm_100a`` is refused AT TRACE TIME by the helper with an error that
  names ``sm_107a`` -- never left to ptxas's ``Illegal modifier '.scaled::n1::ue8m0'`` -- and ``fused=False`` compiles
  there (the portable arm).  ``fused`` itself is the API layer's fact (``compute_capability(dev) == (10, 7)`` -> a
  ``TemplateParams`` flag, the ``api_dsl.py`` ``_EXP2_FMA_SPLIT_CC`` idiom); the trace-time check is the backstop.
* tier 2, ``requires_rubin`` (a Rubin GPU): fused vs unfused BIT-IDENTICAL on normal fp32 inputs for ``e in [0, 253]``,
  both dtypes, both entry points; the fused bytes == the torch DEscale oracle WITH input FTZ on every lane, edge rows
  included (``e = 255 -> 0x7f7f``; ``+-inf -> +-max``; ``NaN -> 0x7f``; ``e = 0`` with ``x = 2^-118 -> max``; ``e = 254 ->
  2^-127``; an fp32-SUBNORMAL input is flushed to +-0 by the fused op and scaled exactly by the FMUL arm -- the one
  documented divergence); the unfused bytes == the same oracle WITHOUT input FTZ; the pair twin == bytes 1 / 0 of the
  16-pack on every lane; and EVERY assembling provenance (constant 119 as the immediate, the SMEM-filled atom, the amax
  and the redux bytes) == the oracle at the byte each lane used.
"""

import glob
import importlib.util
import os
import re
import struct
import subprocess
import sys
import textwrap

import pytest
import torch

from frost_test_utils import arch_known_to_the_dsl, nvdisasm_candidates, requires_dsl, requires_rubin

pytestmark = [pytest.mark.L0, requires_dsl]

N_PACK = 16  # values per fp32_to_fp8_pack_scaled call
LANES_PER_BLOCK = 128
P_SF_BYTE = 119  # cuDNN's fixed MXFP8 P scale: 127 - 8 -> x * 2^8

# The scale byte's provenance (the probe's ``src`` Constexpr) -- mirrored in _PROBE_SRC.
SRC_GMEM, SRC_PYINT, SRC_TRACED_CONST, SRC_SMEM, SRC_AMAX, SRC_REDUX, SRC_PARAM = range(7)
SRC_NAMES = {"gmem": SRC_GMEM, "pyint": SRC_PYINT, "traced_const": SRC_TRACED_CONST, "smem": SRC_SMEM, "amax": SRC_AMAX, "redux": SRC_REDUX, "param": SRC_PARAM}
CONSTANT_SOURCES = (SRC_PYINT, SRC_TRACED_CONST, SRC_SMEM)  # every lane's byte is P_SF_BYTE; the probe stores no byte for the first two

# The probe lives in a FILE (the DSL preprocessor re-reads a kernel's source through ``inspect.getsource``); the same
# text serves the device tier (imported) and the SASS / gate tiers (a subprocess per target arch, arm and provenance).
_PROBE_SRC = textwrap.dedent('''
    """Probe: one lane = one 16-value block (``fp32_to_fp8_pack_scaled``) + the swapped first pair (``fp32_to_fp8x2_scaled``);
    the scale byte comes from the provenance ``src`` selects (SRC_* below)."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.experimental import primitives as nvvm

    from cudnn.frost.tile_dsl.pointwise import abs_max_tree, e8m0_from_amax, fp32_to_fp8_pack_scaled, fp32_to_fp8x2_scaled, warp_abs_max_f32
    from cudnn.frost.tile_dsl.tma import ld_global, ld_global_v4, st_global, st_global_v4

    N = 16
    P_SF_BYTE = 119
    SRC_GMEM, SRC_PYINT, SRC_TRACED_CONST, SRC_SMEM, SRC_AMAX, SRC_REDUX, SRC_PARAM = range(7)


    @cute.kernel
    def probe(mSrc: cute.Tensor, mSf: cute.Tensor, mDst: cute.Tensor, mPair: cute.Tensor, mSfOut: cute.Tensor, sf_param: cutlass.Int32,
              e5m2: cutlass.Constexpr[bool], fused: cutlass.Constexpr[bool], src: cutlass.Constexpr[int]):
        tidx = cutlass.Int32(cute.arch.thread_idx()[0])
        bidx = cutlass.Int32(cute.arch.block_idx()[0])
        lane = bidx * cutlass.Int32(cute.arch.block_dim()[0]) + tidx
        lane64 = lane.to(cutlass.Int64)
        base = mSrc.iterator.toint() + lane64 * cutlass.Int64(N * 4)
        vals = []
        for j in cutlass.range_constexpr(N // 4):
            for w in ld_global_v4(base + cutlass.Int64(j * 16), cutlass.Float32):
                vals.append(w)
        if cutlass.const_expr(src == SRC_GMEM):
            sf = ld_global(mSf.iterator.toint() + lane64 * cutlass.Int64(4), cutlass.Int32)  # the block's byte from memory
        elif cutlass.const_expr(src == SRC_PYINT):
            sf = P_SF_BYTE  # a Python int: P's fixed byte -> the helper's cvt.u8.u32 IMMEDIATE
        elif cutlass.const_expr(src == SRC_TRACED_CONST):
            sf = cutlass.Int32(P_SF_BYTE)  # a constant Int32 still holding its Python value -> the same immediate
        elif cutlass.const_expr(src == SRC_SMEM):
            smem = cutlass.memory.SmemAllocator()  # (cutlass.utils.SmemAllocator is deprecated in 4.8.0)
            atom = smem.allocate_array(cutlass.Int32, 32)
            if tidx == 0:
                atom[0] = cutlass.Int32(P_SF_BYTE)  # the kernel fills its own SF atom (sP_SF) ...
            nvvm.barrier_cta_sync()
            sf = atom[0]  # ... and every lane loads the byte: data to ptxas (one LDS)
        elif cutlass.const_expr(src == SRC_AMAX):
            _rcp, sf = e8m0_from_amax(abs_max_tree(vals))  # the dS quantizer's byte: cvt.rp.satfinite.ue8m0x2 & 0xff
        elif cutlass.const_expr(src == SRC_REDUX):
            _rcp, sf = e8m0_from_amax(warp_abs_max_f32(vals[0]))  # the dS along-kv column amax, one CREDUX, warp-uniform
        else:
            sf = sf_param  # a kernel scalar parameter (ld.param): ICEs ptxas -- the tripwire
        dtype = cutlass.Float8E5M2 if cutlass.const_expr(e5m2) else cutlass.Float8E4M3FN
        packed = fp32_to_fp8_pack_scaled(vals, sf, dtype=dtype, fused=fused)
        st_global_v4(mDst.iterator.toint() + lane64 * cutlass.Int64(N), [packed[0], packed[1], packed[2], packed[3]], cutlass.Int32)
        # the pair twin on the SWAPPED first pair: identical operands would let ptxas CSE it with the pack's first cvt
        pair = fp32_to_fp8x2_scaled(vals[1], vals[0], sf, dtype=dtype, fused=fused)
        st_global(mPair.iterator.toint() + lane64 * cutlass.Int64(4), cutlass.Int32(pair), cutlass.Int32)
        if cutlass.const_expr(src not in (SRC_PYINT, SRC_TRACED_CONST)):
            st_global(mSfOut.iterator.toint() + lane64 * cutlass.Int64(4), sf, cutlass.Int32)  # the byte each lane used, for the oracle


    probe.set_name_prefix("cudnn", remove_cutlass_symbol=True)


    @cute.jit
    def launch(src_t: cute.Tensor, sf: cute.Tensor, dst: cute.Tensor, pair: cute.Tensor, sf_out: cute.Tensor, sf_param: cutlass.Int32, n_blocks: cutlass.Int32,
               e5m2: cutlass.Constexpr[bool], fused: cutlass.Constexpr[bool], src: cutlass.Constexpr[int], stream: cuda.CUstream):
        probe(src_t, sf, dst, pair, sf_out, sf_param, e5m2, fused, src).launch(grid=(n_blocks, 1, 1), block=(128, 1, 1), smem=256, stream=stream)


    if __name__ == "__main__":
        # argv: <arch> <fused 0/1> <e5m2 0/1> <src 0..6> <dump dir> [nvdisasm candidates...]; CUTE_DSL_DUMP_DIR is set by the caller.
        import glob, os, re, subprocess, sys
        from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

        arch, fused_s, e5m2_s, src_i, dump, cands = sys.argv[1], sys.argv[2] == "1", sys.argv[3] == "1", int(sys.argv[4]), sys.argv[5], sys.argv[6:]

        def fake(dt):
            return make_fake_tensor(dtype=dt, shape=(cute.sym_int(),), stride=(1,), assumed_align=16)

        # --keep-cubin / --keep-ptx, NOT --keep-sass: the latter runs the DSL's own wheel nvdisasm, which may not decode the target.
        try:
            cute.compile(launch, fake(cutlass.Float32), fake(cutlass.Int32), fake(cutlass.Int32), fake(cutlass.Int32), fake(cutlass.Int32), cutlass.Int32(P_SF_BYTE),
                         cutlass.Int32(1), e5m2_s, fused_s, src_i, make_fake_stream(use_tvm_ffi_env_stream=False),
                         options=f"--enable-tvm-ffi --gpu-arch {arch} --keep-cubin --keep-ptx")
        except Exception as exc:  # a trace-time refusal or a ptxas failure: report WHICH, for the gate and the tripwire
            msg = f"{type(exc).__name__}: {exc}"
            print("COMPILE_FAILED", arch, "fused" if fused_s else "unfused", "src", src_i)
            print("FAILED_IN_PTXAS", int("ptxas" in msg), "C7907", int("C7907" in msg or "Internal compiler error" in msg), "ILLEGAL_MODIFIER", int("Illegal modifier" in msg))
            print(msg[-3000:])
            sys.exit(4)
        print("COMPILED", arch, "fused" if fused_s else "unfused", "e5m2" if e5m2_s else "e4m3", "src", src_i)
        ptxs = glob.glob(os.path.join(dump, f"*.{arch}.ptx"))
        if ptxs:
            ptx = open(ptxs[0]).read()
            print("PTX_SCALED_CVT", ptx.count("scaled::n1::ue8m0"))
            print("PTX_CVT_U8_IMM", ptx.count(f"cvt.u8.u32 sf, {P_SF_BYTE};"))
            print("PTX_CVT_U8_REG", len(re.findall(r"cvt\\.u8\\.u32 sf, %r\\d+;", ptx)))
        cubins = glob.glob(os.path.join(dump, f"*.{arch}.cubin"))
        if not cubins:
            print("FAIL no cubin landed in", dump, os.listdir(dump)); sys.exit(3)
        sass = None
        for nvd in cands:
            try:
                proc = subprocess.run([nvd, "-c", cubins[0]], capture_output=True, text=True, timeout=120)
            except (OSError, subprocess.SubprocessError) as exc:
                print("REJECT", nvd, "->", repr(exc)); continue
            if proc.returncode == 0 and proc.stdout.strip():
                sass = proc.stdout.splitlines(); print("NVDISASM", nvd); break
            print("REJECT", nvd, "->", (proc.stderr.strip().splitlines() or [str(proc.returncode)])[-1])
        if sass is None:
            print("SKIP no nvdisasm candidate decodes", arch); sys.exit(0)
        fmt = "E5M2" if e5m2_s else "E4M3"
        print("SCALE_BY_C", sum(1 for ln in sass if "F2FP" in ln and "SCALE_BY_C" in ln))
        print("SCALE_BY_C_FMT", sum(1 for ln in sass if "F2FP" in ln and "SCALE_BY_C" in ln and fmt in ln))
        print("F2FP_PLAIN", sum(1 for ln in sass if "F2FP.SATFINITE" in ln and f".{fmt}." in ln and "SCALE_BY_C" not in ln))
        print("F2FP_UE8M0", sum(1 for ln in sass if "F2FP" in ln and ".E4M3." not in ln and ".E5M2." not in ln))  # the e8m0 derivation's cvt
        print("FMUL", sum(1 for ln in sass if re.search(r"\\bFMUL\\b", ln)))
        print("SHF", sum(1 for ln in sass if "SHF.L.U32" in ln))
        print("LDS", sum(1 for ln in sass if re.search(r"\\bLDS\\b", ln)))
        print("REDUX", sum(1 for ln in sass if "REDUX" in ln))
        print("SPILL", sum(1 for ln in sass if "STL" in ln or "LDL" in ln))
        print("LINES", len(sass))
    ''')

_SASS_KEYS = (
    "SCALE_BY_C",
    "SCALE_BY_C_FMT",
    "F2FP_PLAIN",
    "F2FP_UE8M0",
    "FMUL",
    "SHF",
    "LDS",
    "REDUX",
    "SPILL",
    "LINES",
    "PTX_SCALED_CVT",
    "PTX_CVT_U8_IMM",
    "PTX_CVT_U8_REG",
)


def _write_probe(tmp_path):
    probe_py = tmp_path / "fp8_pack_scaled_probe.py"
    probe_py.write_text(_PROBE_SRC)
    return probe_py


def _run_probe(tmp_path, *, arch: str, fused: bool, e5m2: bool, src: int = SRC_GMEM, timeout: int = 900):
    """Trace-compile the probe for ``arch`` in a fresh interpreter (CUTE_DSL_DUMP_DIR must be set before ``import cutlass``)."""
    tag = f"{arch}_{'fused' if fused else 'unfused'}_{'e5m2' if e5m2 else 'e4m3'}_src{src}"
    dump = tmp_path / f"dump_{tag}"
    dump.mkdir()
    env = dict(os.environ, CUTE_DSL_DUMP_DIR=str(dump), CUTE_DSL_KEEP="ptx,cubin")
    argv = [sys.executable, str(_write_probe(tmp_path)), arch, "1" if fused else "0", "1" if e5m2 else "0", str(src), str(dump), *nvdisasm_candidates()]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env), dump


def _stats(proc):
    return {k: int(v) for k, v in (ln.split() for ln in proc.stdout.splitlines() if ln.split() and ln.split()[0] in _SASS_KEYS)}


def _sass_or_skip(proc, lines):
    if not nvdisasm_candidates():
        pytest.skip("compiled; no nvdisasm executable to try for the SASS half (CUDA_PATH unset and none on PATH)")
    if any(ln.startswith("SKIP") for ln in lines):
        pytest.skip(f"compiled; SASS half skipped: {[ln for ln in lines if ln.startswith(('SKIP', 'REJECT'))]}")
    return _stats(proc)


# ---------------------------------------------------------------------------
# Tier 1: the sm_107a trace-compile and its SASS
# ---------------------------------------------------------------------------

N_PAIRS = N_PACK // 2 + 1  # the 16-pack's eight cvts + the pair twin


@pytest.mark.parametrize("e5m2", [False, True], ids=["e4m3", "e5m2"])
@pytest.mark.parametrize("fused", [True, False], ids=["fused", "unfused"])
def test_sm107a_sass_pins(tmp_path, e5m2, fused):
    """The probe (byte from GMEM) compiles for sm_107a on any box (asserted); with a decoding nvdisasm, the fused build
    carries exactly 8 + 1 ``F2FP...SCALE_BY_C`` (one per pair: the 16-pack and the swapped pair twin), no ``FMUL`` at all
    (the descale multiply is gone) and no unscaled ``F2FP``; the unfused twin carries 16..18 ``FMUL`` (one per distinct
    element; whether the pair twin's two products CSE with the pack's is the toolchain's call -- a bound, like the spill
    pins) and 8 + 1 unscaled ``F2FP`` and no ``SCALE_BY_C``.  Neither spills.  The ``.b8`` extraction's SASS spelling
    (``SHF.L.U32`` in the standalone instruction probe, folded into the F2FP's C-operand PRMTs here) is reported, not
    pinned."""
    if not arch_known_to_the_dsl("sm_107a"):
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0)")
    proc, dump = _run_probe(tmp_path, arch="sm_107a", fused=fused, e5m2=e5m2)
    assert proc.returncode == 0, f"sm_107a trace-compile failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    lines = proc.stdout.splitlines()
    assert any(ln.startswith("COMPILED sm_107a") for ln in lines), proc.stdout[-2000:]
    assert glob.glob(str(dump / "*.sm_107a.cubin")), "no .sm_107a.cubin landed in CUTE_DSL_DUMP_DIR"
    stats = _sass_or_skip(proc, lines)
    print(f"\n[{'e5m2' if e5m2 else 'e4m3'} {'fused' if fused else 'unfused'} gmem] sm_107a SASS: {stats}")
    assert stats["SPILL"] == 0, stats
    if fused:
        assert stats["SCALE_BY_C"] == N_PAIRS, f"expected {N_PAIRS} F2FP...SCALE_BY_C (8 per 16-pack + 1 pair) -- {stats}"
        assert stats["SCALE_BY_C_FMT"] == N_PAIRS, f"the SCALE_BY_C cvts must carry the requested format -- {stats}"
        assert stats["FMUL"] == 0, f"the fused arm must emit NO FMUL (the descale multiply is inside the cvt) -- {stats}"
        assert stats["F2FP_PLAIN"] == 0, f"the fused arm must emit no unscaled F2FP -- {stats}"
        assert stats["PTX_CVT_U8_REG"] == 2 and stats["PTX_CVT_U8_IMM"] == 0, f"a traced byte is the cvt.u8.u32's REGISTER operand (pack + pair) -- {stats}"
    else:
        assert stats["SCALE_BY_C"] == 0, f"the unfused arm must not use the scaled cvt -- {stats}"
        assert N_PACK <= stats["FMUL"] <= N_PACK + 2, f"expected one FMUL per distinct element (16, up to 18 if the pair twin's products do not CSE) -- {stats}"
        assert stats["F2FP_PLAIN"] == N_PAIRS, f"expected {N_PAIRS} unscaled F2FP in the unfused arm -- {stats}"


# (provenance, fused) -> the pins that identify it; every row must ASSEMBLE for sm_107a.  A constant byte: the 16-pack takes
# the cvt.u8.u32 IMMEDIATE (8 SCALE_BY_C), the pair twin the FMUL arm (its lone scaled cvt with an immediate scale ICEs
# ptxas, MEASURED 2026-09-30: 2 FMUL + 1 plain cvt instead).  amax / redux: the 1 FMUL is e8m0_from_amax's amax * 1/448
# and the 1 F2FP_UE8M0 its cvt.rp.satfinite.ue8m0x2 -- neither is the pack's.
_CONST_FUSED = dict(SCALE_BY_C=N_PACK // 2, FMUL=2, F2FP_PLAIN=1, PTX_SCALED_CVT=N_PACK // 2, PTX_CVT_U8_IMM=1, PTX_CVT_U8_REG=0)
_PROVENANCE_PINS = {
    ("pyint", True): _CONST_FUSED,
    ("pyint", False): dict(SCALE_BY_C=0, F2FP_PLAIN=N_PAIRS, PTX_SCALED_CVT=0),
    ("traced_const", True): _CONST_FUSED,
    ("smem", True): dict(SCALE_BY_C=N_PAIRS, FMUL=0, F2FP_PLAIN=0, PTX_CVT_U8_REG=2, PTX_CVT_U8_IMM=0, LDS=1),
    ("amax", True): dict(SCALE_BY_C=N_PAIRS, FMUL=1, F2FP_PLAIN=0, F2FP_UE8M0=1, PTX_CVT_U8_REG=2, PTX_CVT_U8_IMM=0),
    ("redux", True): dict(SCALE_BY_C=N_PAIRS, FMUL=1, F2FP_PLAIN=0, F2FP_UE8M0=1, PTX_CVT_U8_REG=2, PTX_CVT_U8_IMM=0, REDUX=1),
}


@pytest.mark.parametrize("src_name,fused", list(_PROVENANCE_PINS), ids=[f"{s}-{'fused' if f else 'unfused'}" for s, f in _PROVENANCE_PINS])
def test_sm107a_sass_pins_per_scale_byte_provenance(tmp_path, src_name, fused):
    """Every production provenance of the scale byte ASSEMBLES for sm_107a: the data-derived ones (the SMEM-filled atom --
    one ``LDS`` --, the amax byte -- its one FMUL is ``amax * 1/448`` --, the redux byte -- one ``REDUX``) lower to 8 + 1
    ``SCALE_BY_C`` with no pack FMUL; the constant 119 (Python int and constant ``Int32``) lowers to the 16-pack's 8
    ``SCALE_BY_C`` through the ``cvt.u8.u32`` IMMEDIATE (the one constant form ptxas accepts -- the register-materialized
    constant of the first draft ICEd, see the tripwire below) plus the pair twin's FMUL fallback (2 FMUL + 1 plain cvt: a
    lone scaled cvt with an immediate scale ICEs).  The Python-int constant through the FMUL arm folds the multiplier: no
    scaled cvt, 8 + 1 plain packs."""
    if not arch_known_to_the_dsl("sm_107a"):
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0)")
    proc, dump = _run_probe(tmp_path, arch="sm_107a", fused=fused, e5m2=False, src=SRC_NAMES[src_name])
    assert proc.returncode == 0, f"sm_107a trace-compile of the {src_name} provenance failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    lines = proc.stdout.splitlines()
    assert glob.glob(str(dump / "*.sm_107a.cubin")), "no .sm_107a.cubin landed in CUTE_DSL_DUMP_DIR"
    ptx_stats = _stats(proc)
    for key in ("PTX_SCALED_CVT", "PTX_CVT_U8_IMM", "PTX_CVT_U8_REG"):
        if key in _PROVENANCE_PINS[(src_name, fused)]:
            assert ptx_stats[key] == _PROVENANCE_PINS[(src_name, fused)][key], f"{src_name}: {key} -- {ptx_stats}"
    stats = _sass_or_skip(proc, lines)
    print(f"\n[{src_name} {'fused' if fused else 'unfused'}] sm_107a SASS: {stats}")
    assert stats["SPILL"] == 0, stats
    for key, want in _PROVENANCE_PINS[(src_name, fused)].items():
        if not key.startswith("PTX_"):
            assert stats[key] == want, f"{src_name} {'fused' if fused else 'unfused'}: {key} = {stats[key]}, expected {want} -- {stats}"
    if not fused:
        assert N_PACK <= stats["FMUL"] <= N_PACK + 2, f"the FMUL arm keeps one multiply per distinct element -- {stats}"


def test_kernel_parameter_scale_byte_still_ices_ptxas(tmp_path):
    """THE TRIPWIRE of the provenance rule (``pointwise.py`` header): a scale byte that is a KERNEL PARAMETER (``ld.param``,
    warp-uniform) makes ptxas die on sm_107a with ``(C7907) Missing Mercury ISA version for instruction MAD`` -- an ICE the
    helper cannot pre-empt because a traced value carries no provenance.  MEASURED 2026-09-30 on libnvptxcompiler 13.4.46
    (the DSL's) and a second standalone ptxas build; the same for ``opaque_i32(Int32(119))``, ``tid & 0 | 119`` and every
    laundering detour tried.  When this test FAILS because the compile SUCCEEDS, ptxas has been fixed: relax the rule in the
    header and in ``_scale_byte_operand``'s docstring, and delete this test."""
    if not arch_known_to_the_dsl("sm_107a"):
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0)")
    proc, _dump = _run_probe(tmp_path, arch="sm_107a", fused=True, e5m2=False, src=SRC_PARAM)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, (
        "ptxas now ASSEMBLES a kernel-parameter scale byte for sm_107a -- the provenance rule can be relaxed "
        f"(pointwise.py header, _scale_byte_operand); delete this tripwire:\n{proc.stdout[-2000:]}"
    )
    assert "FAILED_IN_PTXAS 1" in out and "C7907 1" in out, f"the failure must be ptxas's C7907 ICE, not a trace-time error:\n{out[-3000:]}"
    assert "ILLEGAL_MODIFIER 0" in out, f"an 'Illegal modifier' means the target was not sm_107a:\n{out[-3000:]}"


# ---------------------------------------------------------------------------
# Tier 1b: the arch gate (RED until the helper refuses at trace time)
# ---------------------------------------------------------------------------


def test_fused_arm_is_refused_at_trace_time_for_a_non_rubin_target(tmp_path):
    """``fused=True`` under ``--gpu-arch sm_100a`` must fail in the HELPER, at trace time, with a message that names the
    only target the instruction assembles for (``sm_107a``) and the helper -- not in ptxas with ``Illegal modifier``
    (the failure the gate exists to pre-empt).  ``fused=False`` is the portable arm and compiles for sm_100a."""
    if not arch_known_to_the_dsl("sm_100a"):
        pytest.skip("this cutlass-dsl has no sm_100a")
    proc, _dump = _run_probe(tmp_path, arch="sm_100a", fused=True, e5m2=False)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"fused=True traced for sm_100a must be refused; it compiled:\n{proc.stdout[-2000:]}"
    assert (
        "Illegal modifier" not in out and "ILLEGAL_MODIFIER 1" not in out
    ), f"the refusal came from ptxas, not from the helper's trace-time gate:\n{out[-3000:]}"
    assert "sm_107a" in out and "fp32_to_fp8_pack_scaled" in out, f"the trace-time refusal must name sm_107a and the helper:\n{out[-3000:]}"
    proc2, _dump2 = _run_probe(tmp_path, arch="sm_100a", fused=False, e5m2=False)
    assert proc2.returncode == 0, f"fused=False must compile for sm_100a (the portable FMUL arm):\n{proc2.stdout[-2000:]}\n{proc2.stderr[-2000:]}"


def test_constant_scale_byte_out_of_range_is_refused():
    """A compile-time constant outside the byte, or above e8m0_rcp's e <= 253 in the FMUL arm, is a typed refusal at trace
    time -- no kernel needed (the check runs before any IR is emitted)."""
    from cudnn.frost.tile_dsl.pointwise import _scale_byte_operand

    assert _scale_byte_operand(P_SF_BYTE, fused=True, who="t") == (P_SF_BYTE, None)
    assert _scale_byte_operand(253, fused=False, who="t") == (253, None)
    with pytest.raises(ValueError, match=r"must be in \[0, 255\]"):
        _scale_byte_operand(256, fused=True, who="t")
    with pytest.raises(ValueError, match=r"must be in \[0, 255\]"):
        _scale_byte_operand(-1, fused=False, who="t")
    with pytest.raises(ValueError, match=r"e8m0_rcp\), valid for e <= 253"):
        _scale_byte_operand(254, fused=False, who="t")
    assert _scale_byte_operand(254, fused=True, who="t") == (254, None), "the fused 16-pack handles e = 254 natively (2^-127)"
    with pytest.raises(ValueError, match=r"e8m0_rcp\), valid for e <= 253"):
        _scale_byte_operand(254, fused=True, who="t", const_takes_fmul=True)  # the pair twin's constant takes the FMUL arm even when fused
    # a bool is refused as a TYPE error that names the CALLER (`who`), so a caller of the pair twin is not sent to the 16-pack
    with pytest.raises(TypeError, match=r"^t: .*not a bool"):
        _scale_byte_operand(True, fused=True, who="t")
    import cutlass

    from cudnn.frost.tile_dsl.pointwise import fp32_to_fp8_pack_scaled, fp32_to_fp8x2_scaled

    with pytest.raises(TypeError, match=r"^fp32_to_fp8_pack_scaled: .*not a bool"):
        fp32_to_fp8_pack_scaled([None] * 16, True, dtype=cutlass.Float8E4M3FN, fused=False)  # the byte is checked before any value is touched
    with pytest.raises(TypeError, match=r"^fp32_to_fp8x2_scaled: .*not a bool"):
        fp32_to_fp8x2_scaled(None, None, True, dtype=cutlass.Float8E4M3FN, fused=False)


# ---------------------------------------------------------------------------
# Tier 2: numerics on a Rubin device
# ---------------------------------------------------------------------------

_E4M3_MAX, _E5M2_MAX = 448.0, 57344.0
_FP32_MIN_NORMAL = 2.0**-126


def _f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def e8m0_descale(e: torch.Tensor) -> torch.Tensor:
    """``2^(127 - e)`` as fp32: ``e = 254 -> 2^-127`` (the fp32 subnormal 0x00400000), ``e = 255 -> NaN`` (the e8m0 NaN)."""
    e64 = e.to(torch.int64)
    bits = torch.where(e64 == 254, torch.full_like(e64, 0x00400000), (254 - e64) << 23)
    bits = torch.where(e64 == 255, torch.full_like(e64, 0x7FC00000), bits)
    return bits.to(torch.int32).view(torch.float32)


def torch_scaled_pack(x: torch.Tensor, e: torch.Tensor, fmt: str, *, ftz_inputs: bool) -> torch.Tensor:
    """The torch DEscale oracle of the instruction: ``fp8(x * 2^(127-e))`` per element, e5m2 clamped to +-57344 first (torch's
    e5m2 cast does not saturate; e4m3fn's does), NaN passed through (0x7f), ``e = 255`` -> 0x7f in EVERY byte, and --
    with ``ftz_inputs`` -- fp32-subnormal inputs flushed to signed zero, which is what the hardware does.  ``x``:
    ``[n, k]`` fp32, ``e``: ``[n]`` int; returns uint8 ``[n, k]``."""
    x = x.float()
    if ftz_inputs:
        sub = (x != 0) & (x.abs() < _FP32_MIN_NORMAL)
        x = torch.where(sub, torch.copysign(torch.zeros_like(x), x), x)
    y = x * e8m0_descale(e).unsqueeze(-1)
    m = _E4M3_MAX if fmt == "e4m3" else _E5M2_MAX
    y = torch.where(torch.isnan(y), y, y.clamp(-m, m))
    dt = torch.float8_e4m3fn if fmt == "e4m3" else torch.float8_e5m2
    b = y.to(dt).view(torch.uint8)
    return torch.where((e == 255).unsqueeze(-1), torch.full_like(b, 0x7F), b)


# Edge lanes (one 16-value block + its scale byte each); the rest of the lanes are random normal fp32 with e in [0, 253].
_EDGE_A = [1.0, -1.0, 17.0, 19.0, 21.0, 23.0, 448.0, 449.0, 464.0, 1000.0, 3e38, -3e38, 2.0**-9, 1.5 * 2.0**-10, 2.0**-10, 0.0]
_EDGE_B = [
    float("inf"),
    -float("inf"),
    float("nan"),
    -0.0,
    57344.0,
    61440.0,
    -1e6,
    1e-30,
    2.0**-6,
    2.0**-7,
    2.0**-8,
    2.0**-14,
    2.0**-16,
    2.0**-17,
    1.0625,
    1.4375,
]
_EDGE_SUB = [
    _f32(0x00080000),
    _f32(0x000116C2),
    -_f32(0x00080000),
    2.0**-130,
    1e-40,
    -1e-40,
    2.0**-127,
    2.0**-140,
    1.0,
    2.0,
    0.0,
    -0.0,
    448.0,
    0.5,
    2.0**-6,
    3.0,
]
_EDGE_LANES = [
    # (values, e, tag)
    (_EDGE_A, 127, "identity scale, ties, saturation, min subnormal"),
    (_EDGE_B, 127, "inf / NaN / signed zero / e5m2 range"),
    (_EDGE_A, 255, "e8m0 NaN byte -> 0x7f7f"),
    (_EDGE_B, 255, "e8m0 NaN byte -> 0x7f7f (inf / NaN inputs)"),
    ([2.0**-118, 1e-30, 1.0, 2.0, -(2.0**-118), 2.0**-120, 2.0**-125, 0.0, 3e38, -1.0, 2.0**-119, 2.0**-118 * 1.5, 0.75, 1e6, -1e6, 7.0], 0, "e = 0 is 2^+127"),
    ([448.0, 3e38, 1.0, -3e38, 1e30, 2.0**100, 0.0, -0.0, 1e38, -1e38, 2.0**126, 2.0**125, 1.5 * 2.0**126, 2.0**120, 0.5, 1e10], 254, "e = 254 is 2^-127"),
    (_EDGE_SUB, 0, "fp32-subnormal inputs at e = 0 (fused flushes, FMUL scales)"),
    (_EDGE_SUB, 1, "fp32-subnormal inputs at e = 1"),
    (_EDGE_SUB, 2, "fp32-subnormal inputs at e = 2"),
    (_EDGE_A, P_SF_BYTE, "P's fixed byte 119 = x * 2^8"),
    (_EDGE_A, 126, "e - 127 = -1 -> x * 2"),
    (_EDGE_A, 128, "e - 127 = +1 -> x / 2"),
]


def _device_inputs(n_blocks: int, seed: int = 20260929):
    n = n_blocks * LANES_PER_BLOCK
    g = torch.Generator(device="cpu").manual_seed(seed)
    mag = torch.exp2(torch.rand(n, N_PACK, generator=g) * 60.0 - 30.0)  # |x| log-uniform in 2^-30 .. 2^30: never an fp32 subnormal
    x = torch.randn(n, N_PACK, generator=g) * mag
    e = torch.randint(0, 254, (n,), generator=g, dtype=torch.int32)  # [0, 253]: e8m0_rcp's contract
    for i, (vals, ee, _tag) in enumerate(_EDGE_LANES):
        x[i] = torch.tensor(vals, dtype=torch.float32)
        e[i] = ee
    return x.contiguous(), e.contiguous()


def _lane_masks(x: torch.Tensor, e: torch.Tensor):
    """Which lanes the bit-identity contract covers: no fp32-subnormal input and e <= 253."""
    has_sub = ((x != 0) & (x.abs() < _FP32_MIN_NORMAL)).any(dim=1)
    return (~has_sub) & (e <= 253)


def _launch_probe(mod, x, e, *, e5m2: bool, fused: bool, src: int):
    """One device run of the probe: returns ``(bytes [n, 16] uint8, pair [n] int32, byte_used [n] int32)``."""
    import cuda.bindings.driver as _cuda_driver
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    dev = x.device
    n = x.shape[0]
    n_blocks = n // LANES_PER_BLOCK
    dst = torch.full((n, N_PACK // 4), -1, dtype=torch.int32, device=dev)
    pair = torch.full((n,), -1, dtype=torch.int32, device=dev)
    sf_out = torch.full((n,), -1, dtype=torch.int32, device=dev)
    stream = _cuda_driver.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    args = (
        from_dlpack(x, assumed_align=16),
        from_dlpack(e, assumed_align=16),
        from_dlpack(dst, assumed_align=16),
        from_dlpack(pair, assumed_align=16),
        from_dlpack(sf_out, assumed_align=16),
        cutlass.Int32(P_SF_BYTE),
        cutlass.Int32(n_blocks),
    )
    fn = cute.compile(mod.launch, *args, e5m2, fused, src, stream)
    fn(*args, stream)
    torch.cuda.synchronize()
    assert int((dst == -1).all(dim=1).sum().item()) == 0 and int((pair == -1).sum().item()) == 0, "unwritten lanes"
    if src in (SRC_PYINT, SRC_TRACED_CONST):
        sf_out = torch.full((n,), P_SF_BYTE, dtype=torch.int32, device=dev)  # the probe stores no byte for a constant source
    else:
        assert int((sf_out == -1).sum().item()) == 0, "the probe must store the byte every lane used"
    return dst.view(torch.uint8).reshape(n, N_PACK), pair, sf_out


def _load_probe_module(tmp_path):
    spec = importlib.util.spec_from_file_location("_fp8_pack_scaled_probe_device", str(_write_probe(tmp_path)))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _check_pair_twin(bytes_, pair_, tag):
    """(4) the pair twin (fed vals[1], vals[0]) is bytes 1 / 0 of the 16-pack, zero-extended."""
    pair_lo = (pair_ & 0xFF).to(torch.uint8)
    pair_hi = ((pair_ >> 8) & 0xFF).to(torch.uint8)
    assert torch.equal(pair_lo, bytes_[:, 1]) and torch.equal(pair_hi, bytes_[:, 0]), f"pair twin != pack bytes 1/0 ({tag})"
    assert int((pair_ >> 16).abs().sum().item()) == 0, f"the pair twin must zero-extend its Uint16 ({tag})"


@requires_rubin
@pytest.mark.parametrize("fmt", ["e4m3", "e5m2"])
def test_fused_and_unfused_arms_match_each_other_and_the_descale_oracle(tmp_path, fmt):
    """On a Rubin device (byte from GMEM): (1) fused == unfused bitwise on every lane inside the bit-identity contract
    (normal fp32 inputs, e <= 253), for the 16-pack AND the pair twin; (2) fused == the torch DEscale oracle WITH input FTZ on
    EVERY lane, edge rows included; (3) unfused == the same oracle WITHOUT input FTZ on the contract lanes plus the
    subnormal-input lanes (the FMUL arm scales an fp32 subnormal exactly -- the one documented divergence); (4) the pair twin
    == bytes 1 / 0 of the 16-pack for both arms; (5) the literal measured edge-row bytes on the edge lanes.
    """
    mod = _load_probe_module(tmp_path)
    dev = torch.device("cuda")
    x_cpu, e_cpu = _device_inputs(8)
    x, e = x_cpu.to(dev), e_cpu.to(dev)
    n = x.shape[0]
    e5m2 = fmt == "e5m2"
    out = {fused: _launch_probe(mod, x, e, e5m2=e5m2, fused=fused, src=SRC_GMEM) for fused in (True, False)}
    contract = _lane_masks(x, e)
    assert int(contract.sum().item()) > n // 2
    fused_bytes, fused_pair, _ = out[True]
    unfused_bytes, unfused_pair, _ = out[False]
    for arm, (bytes_, pair_, _sf) in out.items():
        _check_pair_twin(bytes_, pair_, "fused" if arm else "unfused")
    # (1) fused == unfused inside the contract
    mism = (fused_bytes != unfused_bytes).any(dim=1) & contract
    assert int(mism.sum().item()) == 0, (
        f"{int(mism.sum().item())} contract lanes differ between the fused and the FMUL arm; first: lane {int(mism.nonzero()[0])}, e={int(e[mism.nonzero()[0]])}, "
        f"x={x[mism.nonzero()[0]].tolist()}, fused={fused_bytes[mism.nonzero()[0]].tolist()}, unfused={unfused_bytes[mism.nonzero()[0]].tolist()}"
    )
    # (2) fused == the FTZ oracle everywhere, (3) unfused == the non-FTZ oracle where e8m0_rcp is in contract (e <= 253)
    ref_ftz = torch_scaled_pack(x, e, fmt, ftz_inputs=True)
    ref_exact = torch_scaled_pack(x, e, fmt, ftz_inputs=False)
    bad = (fused_bytes != ref_ftz).any(dim=1)
    assert (
        int(bad.sum().item()) == 0
    ), f"fused arm vs FTZ oracle: {int(bad.sum().item())} lanes; first lane {int(bad.nonzero()[0])} ({_EDGE_LANES[int(bad.nonzero()[0])][2] if int(bad.nonzero()[0]) < len(_EDGE_LANES) else 'random'}): x={x[bad.nonzero()[0]].tolist()} e={int(e[bad.nonzero()[0]])} got={fused_bytes[bad.nonzero()[0]].tolist()} ref={ref_ftz[bad.nonzero()[0]].tolist()}"
    ok_unfused = e <= 253
    bad_u = (unfused_bytes != ref_exact).any(dim=1) & ok_unfused
    assert (
        int(bad_u.sum().item()) == 0
    ), f"unfused arm vs exact-scaling oracle: {int(bad_u.sum().item())} lanes; first lane {int(bad_u.nonzero()[0])}: x={x[bad_u.nonzero()[0]].tolist()} e={int(e[bad_u.nonzero()[0]])} got={unfused_bytes[bad_u.nonzero()[0]].tolist()} ref={ref_exact[bad_u.nonzero()[0]].tolist()}"
    # the documented divergence is real: a subnormal input at a tiny e flushes in the fused arm and scales in the FMUL arm
    sub_lane = [i for i, (_v, ee, tag) in enumerate(_EDGE_LANES) if tag.startswith("fp32-subnormal inputs at e = 0")][0]
    assert not torch.equal(fused_bytes[sub_lane], unfused_bytes[sub_lane]), "expected the fp32-subnormal lane to expose the input-FTZ divergence"
    assert fused_bytes[sub_lane, 0].item() == 0x00 and unfused_bytes[sub_lane, 0].item() != 0x00
    # (5) the literal measured edge rows (fused arm; e4m3 / e5m2 bytes)
    fb = fused_bytes.cpu()
    a = {v: i for i, v in enumerate(_EDGE_A)}
    b = {repr(v): i for i, v in enumerate(_EDGE_B)}
    if fmt == "e4m3":
        assert fb[0, a[1.0]].item() == 0x38 and fb[0, a[17.0]].item() == 0x58 and fb[0, a[19.0]].item() == 0x5A, fb[0].tolist()
        assert fb[0, a[448.0]].item() == 0x7E and fb[0, a[449.0]].item() == 0x7E and fb[0, a[464.0]].item() == 0x7E and fb[0, a[3e38]].item() == 0x7E, fb[
            0
        ].tolist()
        assert fb[0, a[2.0**-9]].item() == 0x01 and fb[0, a[1.5 * 2.0**-10]].item() == 0x01 and fb[0, a[2.0**-10]].item() == 0x00, fb[0].tolist()
        assert fb[1, b[repr(float("inf"))]].item() == 0x7E and fb[1, b[repr(-float("inf"))]].item() == 0xFE and fb[1, b[repr(float("nan"))]].item() == 0x7F, fb[
            1
        ].tolist()
        assert fb[9, a[1.0]].item() == 0x78 and fb[9, a[-1.0]].item() == 0xF8, fb[9].tolist()  # byte 119: +-1.0 * 2^8 = +-256 -> e4m3 0x78 / 0xf8
    else:
        assert fb[0, a[1.0]].item() == 0x3C and fb[0, a[17.0]].item() == 0x4C and fb[0, a[19.0]].item() == 0x4D, fb[0].tolist()
        assert fb[0, a[448.0]].item() == 0x5F and fb[0, a[3e38]].item() == 0x7B, fb[0].tolist()
        assert fb[1, b[repr(57344.0)]].item() == 0x7B and fb[1, b[repr(61440.0)]].item() == 0x7B, fb[1].tolist()
        assert fb[1, b[repr(float("inf"))]].item() == 0x7B and fb[1, b[repr(-float("inf"))]].item() == 0xFB and fb[1, b[repr(float("nan"))]].item() == 0x7F, fb[
            1
        ].tolist()
        assert fb[9, a[1.0]].item() == 0x5C and fb[9, a[-1.0]].item() == 0xDC, fb[9].tolist()  # byte 119: +-1.0 * 2^8 = +-256 -> e5m2 0x5c / 0xdc
    assert fb[1, b[repr(-0.0)]].item() == 0x80, "signed zero must survive"
    assert (fb[2] == 0x7F).all() and (fb[3] == 0x7F).all(), "e = 255 (the e8m0 NaN byte) -> 0x7f in every byte whatever the inputs"
    assert fb[4, 0].item() == (0x7E if fmt == "e4m3" else 0x60), "e = 0 with x = 2^-118: 2^9 saturates e4m3 (0x7e) and is exactly 512 in e5m2 (0x60)"
    assert fb[5, 0].item() == 0x00, "448 * 2^-127 is far below the fp8 grid"
    assert fb[5, 1].item() == (0x3E if fmt == "e4m3" else 0x3F), "3e38 * 2^-127 = 1.763 -> 1.75 in both formats (no fp32 intermediate)"


@requires_rubin
@pytest.mark.parametrize("src_name", ["pyint", "traced_const", "smem", "amax", "redux"])
def test_every_assembling_provenance_matches_the_descale_oracle_on_device(tmp_path, src_name):
    """The fused arm's bytes equal the torch DEscale oracle (input FTZ) at THE BYTE EACH LANE USED, for every provenance that
    assembles: the constant 119 as the ``cvt.u8.u32`` immediate (Python int and constant ``Int32`` -- P's path, ``x * 2^8``,
    so lane 0's 1.0 / -1.0 read 0x78 / 0xf8), the SMEM-filled atom, the amax-derived byte (the probe stores it; on the
    inf-carrying edge lanes it is 254) and the warp-uniform redux byte.  For the constant the FMUL arm is run too and is
    bitwise the fused arm on the contract lanes (the P path's ``fused=False`` fallback is exact).  Pair twin on every lane."""
    src = SRC_NAMES[src_name]
    mod = _load_probe_module(tmp_path)
    dev = torch.device("cuda")
    x_cpu, _e_unused = _device_inputs(8)
    x = x_cpu.to(dev)
    e_gmem = torch.full((x.shape[0],), P_SF_BYTE, dtype=torch.int32, device=dev)  # ignored by every source here but SRC_GMEM
    bytes_, pair_, e_used = _launch_probe(mod, x, e_gmem, e5m2=False, fused=True, src=src)
    _check_pair_twin(bytes_, pair_, f"{src_name} fused")
    if src in CONSTANT_SOURCES:
        assert bool((e_used == P_SF_BYTE).all().item()), f"{src_name}: every lane's byte is the constant {P_SF_BYTE}"
    else:
        assert int(((e_used < 0) | (e_used > 255)).sum().item()) == 0, f"{src_name}: e8m0_from_amax bytes must be in [0, 255]: {e_used.unique().tolist()}"
        assert int(e_used.unique().numel()) > 4, f"{src_name}: the data-derived bytes should vary across lanes: {e_used.unique().tolist()}"
    ref = torch_scaled_pack(x, e_used, "e4m3", ftz_inputs=True)
    bad = (bytes_ != ref).any(dim=1)
    assert int(bad.sum().item()) == 0, (
        f"{src_name}: fused arm vs FTZ oracle at each lane's byte: {int(bad.sum().item())} lanes; first lane {int(bad.nonzero()[0])}: "
        f"e={int(e_used[bad.nonzero()[0]])} x={x[bad.nonzero()[0]].tolist()} got={bytes_[bad.nonzero()[0]].tolist()} ref={ref[bad.nonzero()[0]].tolist()}"
    )
    if src in CONSTANT_SOURCES:
        fb = bytes_.cpu()
        assert fb[0, 0].item() == 0x78 and fb[0, 1].item() == 0xF8, f"{src_name}: byte 119 must read as x * 2^8 (+-1.0 -> 0x78 / 0xf8): {fb[0].tolist()}"
    if src == SRC_PYINT:
        ubytes, upair, _ = _launch_probe(mod, x, e_gmem, e5m2=False, fused=False, src=src)
        _check_pair_twin(ubytes, upair, "pyint unfused")
        contract = _lane_masks(x, e_used)
        mism = (bytes_ != ubytes).any(dim=1) & contract
        assert (
            int(mism.sum().item()) == 0
        ), f"pyint: fused (immediate) vs FMUL arm differ on {int(mism.sum().item())} contract lanes; first {int(mism.nonzero()[0])}"
        ref_exact = torch_scaled_pack(x, e_used, "e4m3", ftz_inputs=False)
        bad_u = (ubytes != ref_exact).any(dim=1)
        assert int(bad_u.sum().item()) == 0, f"pyint unfused vs exact oracle: {int(bad_u.sum().item())} lanes"
