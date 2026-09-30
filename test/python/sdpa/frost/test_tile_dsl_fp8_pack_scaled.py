# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``tile_dsl.pointwise.fp32_to_fp8_pack_scaled`` / ``fp32_to_fp8x2_scaled`` -- Rubin's fused DEscale-and-pack
``cvt.rn.satfinite.scaled::n1::ue8m0.{e4m3,e5m2}x2.f32`` behind one helper with a portable FMUL arm.

The contract under test (MEASURED on fractal-ts2-128 cc 10.7, ``frost_dev/results/cvt_scaled_probe_2026-09-29/NUMERICS.md``):
``byte_i = fp8_rn_satfinite(ftz(x_i) * 2^(127 - e))`` -- **the ``.b8`` operand is a DEscale exponent, so the block's E8M0
scale byte (``e8m0_from_amax(...)[1]``) is passed AS IS** and the hardware divides; ``fused=False`` is the bit-identical
``x * e8m0_rcp(e)`` + ``fp32_to_fp8_pack`` spelling for normal fp32 inputs and ``e <= 253``.

Three tiers, one probe kernel (one lane = one 16-value block + the SWAPPED first pair -- both entry points, one scale byte;
swapped because identical operands let ptxas CSE the pair's cvt with the pack's, MEASURED: 8 not 9 SCALE_BY_C):

* tier 1, any box whose cutlass-dsl knows ``sm_107a`` (no device match needed): the sm_107a trace-compile of the probe
  is ASSERTED, and with an ``nvdisasm`` that decodes the cubin (``$CUDA_PATH/bin`` or ``$PATH``; the DSL's own wheel one
  may not) the SASS pins from THAT kernel: fused -> 8 ``F2FP.SATFINITE.<FMT>.F32.PACK_AB_MERGE_C.SCALE_BY_C`` per
  16-value call (+1 for the pair twin), ZERO ``FMUL`` and zero unscaled ``F2FP``; unfused -> 16 ``FMUL`` (one per DISTINCT
  element: the pair's two products CSE with the pack's) and 8 (+1) unscaled ``F2FP``, zero ``SCALE_BY_C``.  The SASS half SKIPS without a decoding nvdisasm; a compile failure is a FAIL.
* tier 1b, the arch gate: ``fused=True`` traced for ``sm_100a`` is refused AT TRACE TIME by the helper with an error that
  names ``sm_107a`` -- never left to ptxas's ``Illegal modifier '.scaled::n1::ue8m0'`` -- and ``fused=False`` compiles
  there (the portable arm).  ``fused`` itself is the API layer's fact (``compute_capability(dev) == (10, 7)`` -> a
  ``TemplateParams`` flag, the ``api_dsl.py`` ``_EXP2_FMA_SPLIT_CC`` idiom); the trace-time check is the backstop.
* tier 2, ``requires_rubin`` (fractal GPU): fused vs unfused BIT-IDENTICAL on normal fp32 inputs for ``e in [0, 253]``,
  both dtypes, both entry points; the fused bytes == the torch DEscale oracle WITH input FTZ on every lane, edge rows
  included (``e = 255 -> 0x7f7f``; ``+-inf -> +-max``; ``NaN -> 0x7f``; ``e = 0`` with ``x = 2^-118 -> max``; ``e = 254 ->
  2^-127``; an fp32-SUBNORMAL input is flushed to +-0 by the fused op and scaled exactly by the FMUL arm -- the one
  documented divergence); the unfused bytes == the same oracle WITHOUT input FTZ; the pair twin == bytes 1 / 0 of the
  16-pack on every lane.
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

from frost_test_utils import _dsl_installed, arch_known_to_the_dsl, nvdisasm_candidates, requires_dsl, requires_rubin

pytestmark = [pytest.mark.L0, requires_dsl]

N_PACK = 16  # values per fp32_to_fp8_pack_scaled call
LANES_PER_BLOCK = 128

# The probe lives in a FILE (the DSL preprocessor re-reads a kernel's source through ``inspect.getsource``); the same
# text serves the device tier (imported) and the SASS / gate tiers (a subprocess per target arch and arm).
_PROBE_SRC = textwrap.dedent('''
    """Probe: one lane = one 16-value block (``fp32_to_fp8_pack_scaled``) + the swapped first pair (``fp32_to_fp8x2_scaled``)."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from cudnn.frost.tile_dsl.pointwise import fp32_to_fp8_pack_scaled, fp32_to_fp8x2_scaled
    from cudnn.frost.tile_dsl.tma import ld_global, ld_global_v4, st_global, st_global_v4

    N = 16


    @cute.kernel
    def probe(mSrc: cute.Tensor, mSf: cute.Tensor, mDst: cute.Tensor, mPair: cute.Tensor, e5m2: cutlass.Constexpr[bool], fused: cutlass.Constexpr[bool]):
        tidx = cutlass.Int32(cute.arch.thread_idx()[0])
        bidx = cutlass.Int32(cute.arch.block_idx()[0])
        lane = bidx * cutlass.Int32(cute.arch.block_dim()[0]) + tidx
        lane64 = lane.to(cutlass.Int64)
        src = mSrc.iterator.toint() + lane64 * cutlass.Int64(N * 4)
        vals = []
        for j in cutlass.range_constexpr(N // 4):
            for w in ld_global_v4(src + cutlass.Int64(j * 16), cutlass.Float32):
                vals.append(w)
        sf = ld_global(mSf.iterator.toint() + lane64 * cutlass.Int64(4), cutlass.Int32)
        dtype = cutlass.Float8E5M2 if cutlass.const_expr(e5m2) else cutlass.Float8E4M3FN
        packed = fp32_to_fp8_pack_scaled(vals, sf, dtype=dtype, fused=fused)
        st_global_v4(mDst.iterator.toint() + lane64 * cutlass.Int64(N), [packed[0], packed[1], packed[2], packed[3]], cutlass.Int32)
        # the pair twin on the SWAPPED first pair: identical operands would let ptxas CSE it with the pack's first cvt
        pair = fp32_to_fp8x2_scaled(vals[1], vals[0], sf, dtype=dtype, fused=fused)
        st_global(mPair.iterator.toint() + lane64 * cutlass.Int64(4), cutlass.Int32(pair), cutlass.Int32)


    probe.set_name_prefix("cudnn", remove_cutlass_symbol=True)


    @cute.jit
    def launch(src: cute.Tensor, sf: cute.Tensor, dst: cute.Tensor, pair: cute.Tensor, n_blocks: cutlass.Int32, e5m2: cutlass.Constexpr[bool], fused: cutlass.Constexpr[bool], stream: cuda.CUstream):
        probe(src, sf, dst, pair, e5m2, fused).launch(grid=(n_blocks, 1, 1), block=(128, 1, 1), stream=stream)


    if __name__ == "__main__":
        # argv: <arch> <fused 0/1> <e5m2 0/1> <dump dir> [nvdisasm candidates...]; CUTE_DSL_DUMP_DIR is set by the caller.
        import glob, os, re, subprocess, sys
        from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

        arch, fused_s, e5m2_s, dump, cands = sys.argv[1], sys.argv[2] == "1", sys.argv[3] == "1", sys.argv[4], sys.argv[5:]

        def fake(dt):
            return make_fake_tensor(dtype=dt, shape=(cute.sym_int(),), stride=(1,), assumed_align=16)

        # --keep-cubin, NOT --keep-sass: the latter runs the DSL's own wheel nvdisasm, which may not decode the target.
        cute.compile(launch, fake(cutlass.Float32), fake(cutlass.Int32), fake(cutlass.Int32), fake(cutlass.Int32), cutlass.Int32(1), e5m2_s, fused_s,
                     make_fake_stream(use_tvm_ffi_env_stream=False), options=f"--enable-tvm-ffi --gpu-arch {arch} --keep-cubin")
        print("COMPILED", arch, "fused" if fused_s else "unfused", "e5m2" if e5m2_s else "e4m3")
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
        print("F2FP_PLAIN", sum(1 for ln in sass if "F2FP.SATFINITE" in ln and "PACK_AB_MERGE_C" in ln and "SCALE_BY_C" not in ln))
        print("FMUL", sum(1 for ln in sass if re.search(r"\\bFMUL\\b", ln)))
        print("SHF", sum(1 for ln in sass if "SHF.L.U32" in ln))
        print("SPILL", sum(1 for ln in sass if "STL" in ln or "LDL" in ln))
        print("LINES", len(sass))
    ''')

_SASS_KEYS = ("SCALE_BY_C", "SCALE_BY_C_FMT", "F2FP_PLAIN", "FMUL", "SHF", "SPILL", "LINES")


def _write_probe(tmp_path):
    probe_py = tmp_path / "fp8_pack_scaled_probe.py"
    probe_py.write_text(_PROBE_SRC)
    return probe_py


def _run_probe(tmp_path, *, arch: str, fused: bool, e5m2: bool, timeout: int = 900):
    """Trace-compile the probe for ``arch`` in a fresh interpreter (CUTE_DSL_DUMP_DIR must be set before ``import cutlass``)."""
    tag = f"{arch}_{'fused' if fused else 'unfused'}_{'e5m2' if e5m2 else 'e4m3'}"
    dump = tmp_path / f"dump_{tag}"
    dump.mkdir()
    env = dict(os.environ, CUTE_DSL_DUMP_DIR=str(dump), CUTE_DSL_KEEP="cubin")
    argv = [sys.executable, str(_write_probe(tmp_path)), arch, "1" if fused else "0", "1" if e5m2 else "0", str(dump), *nvdisasm_candidates()]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env), dump


# ---------------------------------------------------------------------------
# Tier 1: the sm_107a trace-compile and its SASS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("e5m2", [False, True], ids=["e4m3", "e5m2"])
@pytest.mark.parametrize("fused", [True, False], ids=["fused", "unfused"])
def test_sm107a_sass_pins(tmp_path, e5m2, fused):
    """The probe compiles for sm_107a on any box (asserted); with a decoding nvdisasm, the fused build carries exactly
    8 + 1 ``F2FP...SCALE_BY_C`` (one per pair: the 16-pack and the swapped pair twin), no ``FMUL`` at all (the descale
    multiply is gone) and no unscaled ``F2FP``; the unfused twin carries 16 ``FMUL`` (one per distinct element; the pair
    twin's two products are the pack's) and 8 + 1 unscaled ``F2FP`` and no ``SCALE_BY_C``.  Neither spills.  The
    ``.b8`` extraction's SASS spelling (``SHF.L.U32`` in the probe of NUMERICS.md, folded into the load here) is
    reported, not pinned."""
    if not arch_known_to_the_dsl("sm_107a"):
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0)")
    proc, dump = _run_probe(tmp_path, arch="sm_107a", fused=fused, e5m2=e5m2)
    assert proc.returncode == 0, f"sm_107a trace-compile failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    lines = proc.stdout.splitlines()
    assert any(ln.startswith("COMPILED sm_107a") for ln in lines), proc.stdout[-2000:]
    assert glob.glob(str(dump / "*.sm_107a.cubin")), "no .sm_107a.cubin landed in CUTE_DSL_DUMP_DIR"
    if not nvdisasm_candidates():
        pytest.skip("compiled; no nvdisasm executable to try for the SASS half (CUDA_PATH unset and none on PATH)")
    if any(ln.startswith("SKIP") for ln in lines):
        pytest.skip(f"compiled; SASS half skipped: {[ln for ln in lines if ln.startswith(('SKIP', 'REJECT'))]}")
    stats = {k: int(v) for k, v in (ln.split() for ln in lines if ln.split() and ln.split()[0] in _SASS_KEYS)}
    print(f"\n[{'e5m2' if e5m2 else 'e4m3'} {'fused' if fused else 'unfused'}] sm_107a SASS: {stats}")
    assert stats["SPILL"] == 0, stats
    n_pairs = N_PACK // 2 + 1  # the 16-pack's eight cvts + the pair twin
    if fused:
        assert stats["SCALE_BY_C"] == n_pairs, f"expected {n_pairs} F2FP...SCALE_BY_C (8 per 16-pack + 1 pair) -- {stats}"
        assert stats["SCALE_BY_C_FMT"] == n_pairs, f"the SCALE_BY_C cvts must carry the requested format -- {stats}"
        assert stats["FMUL"] == 0, f"the fused arm must emit NO FMUL (the descale multiply is inside the cvt) -- {stats}"
        assert stats["F2FP_PLAIN"] == 0, f"the fused arm must emit no unscaled F2FP -- {stats}"
    else:
        assert stats["SCALE_BY_C"] == 0, f"the unfused arm must not use the scaled cvt -- {stats}"
        assert stats["FMUL"] == N_PACK, f"expected one FMUL per distinct element (16; the pair twin's products CSE with the pack's) -- {stats}"
        assert stats["F2FP_PLAIN"] == n_pairs, f"expected {n_pairs} unscaled F2FP in the unfused arm -- {stats}"


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
    assert "Illegal modifier" not in out, f"the refusal came from ptxas, not from the helper's trace-time gate:\n{out[-3000:]}"
    assert "sm_107a" in out and "fp32_to_fp8_pack_scaled" in out, f"the trace-time refusal must name sm_107a and the helper:\n{out[-3000:]}"
    proc2, _dump2 = _run_probe(tmp_path, arch="sm_100a", fused=False, e5m2=False)
    assert proc2.returncode == 0, f"fused=False must compile for sm_100a (the portable FMUL arm):\n{proc2.stdout[-2000:]}\n{proc2.stderr[-2000:]}"


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
    """The torch DEscale oracle of NUMERICS.md: ``fp8(x * 2^(127-e))`` per element, e5m2 clamped to +-57344 first (torch's
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
    (_EDGE_A, 119, "P's fixed byte 119 = x * 2^8"),
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


@requires_rubin
@pytest.mark.parametrize("fmt", ["e4m3", "e5m2"])
def test_fused_and_unfused_arms_match_each_other_and_the_descale_oracle(tmp_path, fmt):
    """On a Rubin device: (1) fused == unfused bitwise on every lane inside the bit-identity contract (normal fp32 inputs,
    e <= 253), for the 16-pack AND the pair twin; (2) fused == the torch DEscale oracle WITH input FTZ on EVERY lane, edge
    rows included; (3) unfused == the same oracle WITHOUT input FTZ on the contract lanes plus the subnormal-input lanes
    (the FMUL arm scales an fp32 subnormal exactly -- the one documented divergence); (4) the pair twin == bytes 1 / 0 of the 16-pack for both arms; (5) literal NUMERICS.md bytes on the edge lanes.
    """
    import cuda.bindings.driver as _cuda_driver
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    spec = importlib.util.spec_from_file_location("_fp8_pack_scaled_probe_device", str(_write_probe(tmp_path)))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    dev = torch.device("cuda")
    n_blocks = 8
    x_cpu, e_cpu = _device_inputs(n_blocks)
    x, e = x_cpu.to(dev), e_cpu.to(dev)
    n = x.shape[0]
    e5m2 = fmt == "e5m2"
    stream = _cuda_driver.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    out = {}
    for fused in (True, False):
        dst = torch.full((n, N_PACK // 4), -1, dtype=torch.int32, device=dev)
        pair = torch.full((n,), -1, dtype=torch.int32, device=dev)
        args = (
            from_dlpack(x, assumed_align=16),
            from_dlpack(e, assumed_align=16),
            from_dlpack(dst, assumed_align=16),
            from_dlpack(pair, assumed_align=16),
            cutlass.Int32(n_blocks),
        )
        fn = cute.compile(mod.launch, *args, e5m2, fused, stream)
        fn(*args, stream)
        torch.cuda.synchronize()
        assert int((dst == -1).all(dim=1).sum().item()) == 0 and int((pair == -1).sum().item()) == 0, "unwritten lanes"
        out[fused] = (dst.view(torch.uint8).reshape(n, N_PACK), pair)
    contract = _lane_masks(x, e)
    assert int(contract.sum().item()) > n // 2
    fused_bytes, fused_pair = out[True]
    unfused_bytes, unfused_pair = out[False]
    # (4) the pair twin (fed vals[1], vals[0]) is bytes 1 / 0 of the 16-pack -- both arms, every lane
    for arm, (bytes_, pair_) in out.items():
        pair_lo = (pair_ & 0xFF).to(torch.uint8)
        pair_hi = ((pair_ >> 8) & 0xFF).to(torch.uint8)
        assert torch.equal(pair_lo, bytes_[:, 1]) and torch.equal(pair_hi, bytes_[:, 0]), f"pair twin != pack bytes 1/0 ({'fused' if arm else 'unfused'})"
        assert int((pair_ >> 16).abs().sum().item()) == 0, "the pair twin must zero-extend its Uint16"
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
    # (5) literal NUMERICS.md rows (fused arm; e4m3 / e5m2 columns of its section 5 table)
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
