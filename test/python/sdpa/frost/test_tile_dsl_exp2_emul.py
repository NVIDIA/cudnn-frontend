# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``tile_dsl.pointwise.exp2_emul_pair`` -- 2^x on the FMA pipe (no MUFU.EX2) for a softmax exp burst.

Three tiers:

* tier 1, pure torch on any host: a bit-level model of the helper's PTX (clamp, ``add.rm`` floor
  split, the three ``fma.rn`` of the degree-3 minimax, the exponent insertion) against the exact
  2^x -- pins ``EXP2_EMUL_MAX_REL_ERR`` from both sides (the bound holds AND is tight), the exact
  integers, and the clamp / denormal edge;
* tier 2, an sm_100a trace-compile of a probe kernel that calls the helper (any box whose DSL knows
  ``sm_100a``, no device match needed) and -- when ``$CUDA_PATH/bin`` or ``$PATH`` carries an
  ``nvdisasm`` that decodes it -- the SASS: ZERO ``MUFU.EX2`` and the packed FFMA2 / FADD2 the
  polynomial and the floor split lower to.  The compile is asserted; the SASS half SKIPS without a
  decoding nvdisasm;
* tier 3, on a cc 10.x device (the ``.f32x2`` ops are SM100+): the helper's output is BITWISE the
  tier-1 model over a sweep that covers integers, the clamp, the denormal edge and random fractions.

The kernel that consumes it (``sdpa/fwd/kernels/sm100/prefill_d128_mxfp8.py``) pins its own
MUFU.EX2 count and its fp64-reference LSE in ``test_sdpa_fwd_mxfp8_sm100.py``.
"""

import glob
import os
import shutil
import subprocess
import sys
import textwrap

import pytest
import torch

from frost_test_utils import _SM, _dsl_installed, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]

if _dsl_installed():
    import cuda.bindings.driver as _cuda_driver
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from cudnn.frost.tile_dsl.pointwise import (
        EXP2_EMUL_C1_BITS,
        EXP2_EMUL_C2_BITS,
        EXP2_EMUL_C3_BITS,
        EXP2_EMUL_CLAMP_BITS,
        EXP2_EMUL_FLOOR_MAGIC_BITS,
        EXP2_EMUL_MAX_REL_ERR,
        exp2_emul_pair,
    )
    from cudnn.frost.tile_dsl.tma import ld_global_v4, st_global_v2

    PAIRS_PER_LANE = 8  # 16 fp32 per lane: four ld.global.v4 in, eight st.global.v2 out

    @cute.kernel
    def _probe_kernel(mSrc: cute.Tensor, mDst: cute.Tensor):
        tidx = cutlass.Int32(cute.arch.thread_idx()[0])
        bidx = cutlass.Int32(cute.arch.block_idx()[0])
        lane = bidx * cutlass.Int32(cute.arch.block_dim()[0]) + tidx
        src = mSrc.iterator.toint() + lane.to(cutlass.Int64) * cutlass.Int64(2 * PAIRS_PER_LANE * 4)
        dst = mDst.iterator.toint() + lane.to(cutlass.Int64) * cutlass.Int64(2 * PAIRS_PER_LANE * 4)
        vals = []
        for j in cutlass.range_constexpr(2 * PAIRS_PER_LANE // 4):
            for w in ld_global_v4(src + cutlass.Int64(j * 16), cutlass.Float32):
                vals.append(w)
        for p in cutlass.range_constexpr(PAIRS_PER_LANE):
            lo, hi = exp2_emul_pair(vals[2 * p], vals[2 * p + 1])
            st_global_v2(dst + cutlass.Int64(p * 8), [lo, hi], cutlass.Float32)

    _probe_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    @cute.jit
    def _probe_host(src: cute.Tensor, dst: cute.Tensor, n_blocks: cutlass.Int32, stream: _cuda_driver.CUstream):
        _probe_kernel(src, dst).launch(grid=(n_blocks, 1, 1), block=(128, 1, 1), stream=stream)


# ---------------------------------------------------------------------------
# Tier 1: the host model and the error bound
# ---------------------------------------------------------------------------


def _f32_from_bits(bits: int) -> torch.Tensor:
    return torch.tensor([bits], dtype=torch.int64).to(torch.int32).view(torch.float32)[0]


def exp2_emul_model(x: torch.Tensor) -> torch.Tensor:
    """``exp2_emul_pair``'s PTX, op for op, on an fp32 tensor (CPU or CUDA).

    ``fma.rn.f32x2`` is modelled as an fp64 product + sum rounded once to fp32.  The fp32 x fp32
    product is exact in fp64; the sum rounds only when the addend's exponent is far above the
    product's, and a second rounding to fp32 then differs from a true single-rounding fma only when
    the exact value sits within 2^-53 (relative) of an fp32 rounding midpoint -- probability ~2^-28
    per op, so a bitwise device comparison over a few thousand values does not see it (and if it
    ever did, the mismatch would be exactly one fp32 ulp on one element).  ``.ftz`` is a no-op on
    these inputs: every operand is >= 2^-126 in magnitude or exactly 0."""
    if x.dtype != torch.float32:
        raise TypeError(f"exp2_emul_model: fp32 inputs only, got {x.dtype}")
    dev = x.device
    clamp = _f32_from_bits(EXP2_EMUL_CLAMP_BITS).to(dev)
    magic = _f32_from_bits(EXP2_EMUL_FLOOR_MAGIC_BITS).to(dev)
    c1, c2, c3 = (_f32_from_bits(b).to(dev) for b in (EXP2_EMUL_C1_BITS, EXP2_EMUL_C2_BITS, EXP2_EMUL_C3_BITS))
    x = torch.maximum(x, clamp)  # max.ftz.f32 x, -127
    # add.rm.ftz.f32x2: x + 1.5 * 2^23 rounded toward -inf.  The fp64 sum is exact (|x| < 2^22);
    # round it to fp32 to nearest, then step down one ulp wherever nearest rounded UP.
    t64 = x.double() + magic.double()
    t = t64.float()
    t = torch.where(t.double() > t64, torch.nextafter(t, torch.full_like(t, float("-inf"))), t)
    n = t - magic  # sub.rn: both integers below 2^24, exact
    f = x - n  # sub.rn: exact (f in [0, 1), n = floor(x))

    def fma(a, b, c):
        return (a.double() * b.double() + c.double()).float()

    p = fma(f, c3, c2)
    p = fma(p, f, c1)
    p = fma(p, f, torch.ones((), dtype=torch.float32, device=dev))
    # shl.b32 23 of t's bit pattern == n << 23 (mod 2^32: the magic's high bits shift out), then
    # add.s32 into p's bit pattern -- n lands in the exponent field.
    t_bits = t.view(torch.int32).to(torch.int64)
    p_bits = p.view(torch.int32).to(torch.int64)
    out = ((t_bits << 23) + p_bits) & 0xFFFFFFFF
    out = torch.where(out >= 2**31, out - 2**32, out)
    return out.to(torch.int32).view(torch.float32)


def _sweep_inputs(dev="cpu") -> torch.Tensor:
    """Every integer in [-130, 8], the clamp and denormal edges, every 64th fp32 in [-1, 0) (the
    error depends on the fraction only, and this grid resolves it to 4e-6), and uniform samples of
    [-126, 8] -- the softmax feeds ``S*scale - (max - 4)`` <= 8."""
    g = torch.Generator(device="cpu").manual_seed(0)
    ints = torch.arange(-130, 9, dtype=torch.float32)
    edges = torch.tensor([-127.0, -126.99, -126.5, -126.0000001, -126.0, -125.999, -1e-7, -0.0, 0.0, 1e-7, 7.999999], dtype=torch.float32)
    # Negative floats: a SMALLER magnitude is a SMALLER signed int32 bit pattern, so the grid runs from -(2^-24) up to -1.0.
    a_bits = torch.tensor([-1.0], dtype=torch.float32).view(torch.int32).item()
    b_bits = torch.tensor([-(2.0**-24)], dtype=torch.float32).view(torch.int32).item()
    frac_grid = torch.arange(min(a_bits, b_bits), max(a_bits, b_bits), 64, dtype=torch.int64).to(torch.int32).view(torch.float32)
    uniform = torch.rand(4_000_000, generator=g, dtype=torch.float32) * 134.0 - 126.0
    return torch.cat([ints, edges, frac_grid, uniform]).to(dev)


def test_exp2_emul_model_error_bound_holds_and_is_tight():
    """max |model / 2^x - 1| over the sweep is <= EXP2_EMUL_MAX_REL_ERR (8.8e-5) and >= 8.7e-5: the
    constant the docstring quotes is neither optimistic nor slack (the polynomial's worst fraction is
    f = 0.1008, measured 8.77e-5)."""
    x = _sweep_inputs()
    in_range = x >= -126.0  # the exponent insertion is exact from 2^-126 up (see the edge test below)
    y = exp2_emul_model(x[in_range]).double()
    exact = torch.exp2(x[in_range].double())
    rel = ((y - exact).abs() / exact).max().item()
    print(f"\nexp2_emul model: max rel err {rel:.4e} over {int(in_range.sum())} inputs in [-126, 8]")
    assert rel <= EXP2_EMUL_MAX_REL_ERR, f"model error {rel:.4e} exceeds the documented bound {EXP2_EMUL_MAX_REL_ERR:.1e}"
    assert rel >= 8.7e-5, f"model error {rel:.4e} is far below the documented bound -- re-measure and tighten EXP2_EMUL_MAX_REL_ERR"


def test_exp2_emul_model_integers_are_exact():
    """p(0) == 1.0 exactly, so every integer x in [-126, 8] evaluates to exactly 2^x."""
    x = torch.arange(-126, 9, dtype=torch.float32)
    assert torch.equal(exp2_emul_model(x).double(), torch.exp2(x.double()))


def test_exp2_emul_model_clamp_and_denormal_edge():
    """x <= -127 clamps to the -127 lane and reads exactly 0.0 (exponent field 0, zero mantissa);
    x in (-127, -126) puts a denormal bit pattern below 2^-126 in the output (NOT 2^x -- the exponent
    insertion has no denormal path), which is the documented, backend-identical behaviour and is
    irrelevant against a softmax row max of 2^4 and above."""
    below = torch.tensor([-127.0, -128.0, -200.0, -1e6, float("-inf")], dtype=torch.float32)
    assert torch.equal(exp2_emul_model(below), torch.zeros_like(below))
    edge = torch.linspace(-126.999, -126.001, 257, dtype=torch.float32)
    y = exp2_emul_model(edge)
    assert (y >= 0).all() and (y.double() < 2.0**-126).all(), "the (-127, -126) band must read as a denormal-class value"
    # From 2^-126 up the insertion is exact to the polynomial's bound.
    just_in = torch.linspace(-126.0, -125.0, 1025, dtype=torch.float32)
    rel = (exp2_emul_model(just_in).double() / torch.exp2(just_in.double()) - 1).abs().max().item()
    assert rel <= EXP2_EMUL_MAX_REL_ERR, rel


# ---------------------------------------------------------------------------
# Tier 2: sm_100a trace-compile of the probe kernel + SASS (no device match needed)
# ---------------------------------------------------------------------------


def _nvdisasm_candidates():
    """Executables to TRY; the probe verifies each and skips if none decodes the cubin (hints, not authorities)."""
    cands = []
    if os.environ.get("CUDA_PATH"):
        cands.append(os.path.join(os.environ["CUDA_PATH"], "bin", "nvdisasm"))
    on_path = shutil.which("nvdisasm")
    if on_path:
        cands.append(on_path)
    return [c for c in dict.fromkeys(cands) if os.path.isfile(c) and os.access(c, os.X_OK)]


def _sm100a_known_to_the_dsl() -> bool:
    try:
        from cutlass.base_dsl.enums import Arch

        Arch.from_string("sm_100a")
        return True
    except Exception:
        return False


_SASS_KEYS = ("MUFU_EX2", "MUFU", "FFMA2", "FADD2", "FMNMX", "SPILL", "LINES")

_SASS_PROBE = textwrap.dedent("""
    import glob, os, subprocess, sys
    dump, cands = sys.argv[1], sys.argv[2:]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump  # read once, at the first cutlass import
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream
    from cudnn.frost.tile_dsl.pointwise import exp2_emul_pair
    from cudnn.frost.tile_dsl.tma import ld_global_v4, st_global_v2

    PAIRS = %(pairs)d

    @cute.kernel
    def probe(mSrc: cute.Tensor, mDst: cute.Tensor):
        tidx = cutlass.Int32(cute.arch.thread_idx()[0])
        src = mSrc.iterator.toint() + tidx.to(cutlass.Int64) * cutlass.Int64(2 * PAIRS * 4)
        dst = mDst.iterator.toint() + tidx.to(cutlass.Int64) * cutlass.Int64(2 * PAIRS * 4)
        vals = []
        for j in cutlass.range_constexpr(2 * PAIRS // 4):
            for w in ld_global_v4(src + cutlass.Int64(j * 16), cutlass.Float32):
                vals.append(w)
        for p in cutlass.range_constexpr(PAIRS):
            lo, hi = exp2_emul_pair(vals[2 * p], vals[2 * p + 1])
            st_global_v2(dst + cutlass.Int64(p * 8), [lo, hi], cutlass.Float32)

    @cute.jit
    def launch(src: cute.Tensor, dst: cute.Tensor, stream: cuda.CUstream):
        probe(src, dst).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)

    src = cute.runtime.make_fake_tensor(dtype=cutlass.Float32, shape=(cute.sym_int(),), stride=(1,), assumed_align=16)
    dst = cute.runtime.make_fake_tensor(dtype=cutlass.Float32, shape=(cute.sym_int(),), stride=(1,), assumed_align=16)
    # --keep-cubin, NOT --keep-sass: the latter runs the DSL's own wheel nvdisasm, which may not decode the target.
    cute.compile(launch, src, dst, make_fake_stream(use_tvm_ffi_env_stream=False), options="--enable-tvm-ffi --gpu-arch sm_100a --keep-cubin")
    print("COMPILED")
    cubins = glob.glob(os.path.join(dump, "*.sm_100a.cubin"))
    if not cubins:
        print("FAIL no .sm_100a.cubin landed in", dump)
        sys.exit(3)
    sass = None
    for nvd in cands:
        try:
            proc = subprocess.run([nvd, "-c", cubins[0]], capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as exc:
            print("REJECT", nvd, "->", repr(exc))
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            sass = proc.stdout.splitlines()
            print("NVDISASM", nvd)
            break
        print("REJECT", nvd, "->", (proc.stderr.strip().splitlines() or [str(proc.returncode)])[-1])
    if sass is None:
        print("SKIP no nvdisasm candidate decodes sm_100a")
        sys.exit(0)
    print("MUFU_EX2", sum(1 for ln in sass if "MUFU.EX2" in ln))
    print("MUFU", sum(1 for ln in sass if "MUFU." in ln))
    print("FFMA2", sum(1 for ln in sass if " FFMA2" in ln))
    print("FADD2", sum(1 for ln in sass if " FADD2" in ln))
    print("FMNMX", sum(1 for ln in sass if " FMNMX" in ln))
    print("SPILL", sum(1 for ln in sass if "STL" in ln or "LDL" in ln))
    print("LINES", len(sass))
    """)


def test_exp2_emul_pair_sm100a_sass_has_no_mufu(tmp_path):
    """The helper compiles for sm_100a on any box (asserted), and its SASS carries NO ``MUFU.EX2``
    (the whole point), at least 3 packed ``FFMA2`` (the minimax) and 3 packed ``FADD2`` (the floor
    split's add.rm + two subs) per pair, 2 ``FMNMX`` (the clamps) per pair, and no spill.  SKIPS
    the SASS half when no candidate nvdisasm decodes the cubin."""
    if not _sm100a_known_to_the_dsl():
        pytest.skip("this cutlass-dsl has no sm_100a")
    cands = _nvdisasm_candidates()
    dump = tmp_path / "exp2_emul_sm100a"
    dump.mkdir()
    pairs = 8
    # a FILE, not ``python -c``: the DSL preprocessor re-reads a kernel's source through ``inspect.getsource``
    probe_py = tmp_path / "exp2_emul_probe.py"
    probe_py.write_text(_SASS_PROBE % {"pairs": pairs})
    proc = subprocess.run([sys.executable, str(probe_py), str(dump), *cands], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"sm_100a trace-compile failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    lines = proc.stdout.splitlines()
    assert "COMPILED" in lines, proc.stdout[-2000:]
    assert glob.glob(str(dump / "*.sm_100a.cubin")), "no .sm_100a.cubin landed in CUTE_DSL_DUMP_DIR"
    if not cands:
        pytest.skip("compiled; no nvdisasm executable to try for the SASS half (CUDA_PATH unset and none on PATH)")
    if any(ln.startswith("SKIP") for ln in lines):
        pytest.skip(f"compiled; SASS half skipped: {[ln for ln in lines if ln.startswith(('SKIP', 'REJECT'))]}")
    stats = {k: int(v) for k, v in (ln.split() for ln in lines if ln.split() and ln.split()[0] in _SASS_KEYS)}
    print(f"\nexp2_emul_pair x{pairs} sm_100a SASS: {stats} via {[ln for ln in lines if ln.startswith('NVDISASM')]}")
    assert stats["MUFU_EX2"] == 0, f"the emulation must issue no MUFU.EX2 -- {stats}"
    assert stats["FFMA2"] >= 3 * pairs, f"expected >= 3 packed FFMA2 per pair (the degree-3 minimax) -- {stats}"
    assert stats["FADD2"] >= 3 * pairs, f"expected >= 3 packed FADD2 per pair (add.rm floor + two subs) -- {stats}"
    assert stats["FMNMX"] >= 2 * pairs, f"expected >= 2 FMNMX per pair (the -127 clamps) -- {stats}"
    assert stats["SPILL"] == 0, stats


# ---------------------------------------------------------------------------
# Tier 3: bitwise against the model on a cc 10.x device
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    _SM is None or not (100 <= _SM <= 119), reason="exp2_emul_pair uses the SM100-line packed .f32x2 ops; have " + ("no GPU" if _SM is None else f"sm_{_SM}")
)
def test_exp2_emul_pair_matches_the_model_bitwise_on_device():
    """The device helper reproduces the tier-1 model bit for bit over a 4096-value sweep (integers,
    the clamp and denormal edges, random fractions over [-126, 8]) -- so the error bound the model
    pins is the helper's -- and stays within EXP2_EMUL_MAX_REL_ERR of the exact 2^x from 2^-126 up."""
    dev = torch.device("cuda")
    n_blocks = 2
    n_vals = n_blocks * 128 * 2 * PAIRS_PER_LANE  # 4096
    g = torch.Generator(device="cpu").manual_seed(1)
    parts = [
        torch.arange(-130, 9, dtype=torch.float32),
        torch.tensor([-127.0, -126.99, -126.5, -126.0, -125.999, -1e-7, -0.0, 0.0, 1e-7, 7.999999, -0.8991529, 2.1010334], dtype=torch.float32),
    ]
    parts.append(torch.rand(n_vals - sum(p.numel() for p in parts), generator=g, dtype=torch.float32) * 134.0 - 126.0)
    x = torch.cat(parts).to(dev).contiguous()
    assert x.numel() == n_vals
    y = torch.full_like(x, float("nan"))
    stream = _cuda_driver.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    args = (from_dlpack(x, assumed_align=16), from_dlpack(y, assumed_align=16), cutlass.Int32(n_blocks), stream)
    cute.compile(_probe_host, *args)(*args)
    torch.cuda.synchronize()
    ref = exp2_emul_model(x)
    mism = (y.view(torch.int32) != ref.view(torch.int32)).nonzero().flatten()
    assert (
        mism.numel() == 0
    ), f"{mism.numel()} of {n_vals} device values differ from the model: x={x[mism[:8]].tolist()} device={y[mism[:8]].tolist()} model={ref[mism[:8]].tolist()}"
    in_range = x >= -126.0
    rel = ((y[in_range].double() - torch.exp2(x[in_range].double())).abs() / torch.exp2(x[in_range].double())).max().item()
    assert rel <= EXP2_EMUL_MAX_REL_ERR, rel
