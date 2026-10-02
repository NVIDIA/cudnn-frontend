# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``tile_dsl.pointwise.e8m0_pair_u`` + ``fp32_to_fp8x4_scaled_pairs`` -- the along-kv (per-COLUMN) MXFP8 quantizer of a lane that
holds one ROW: one ``redux.sync.max.abs.f32`` per column (the warp = the 32-row block), the column pair's E8M0 bytes by the shipped
rule on the two warp-uniform amaxes as ONE packed multiply (``e8m0_pair_u``), and every element converted by a LONE
``cvt.rn.satfinite.scaled::n1::ue8m0.e4m3x2.f32`` with ITS column's byte (``fp32_to_fp8x4_scaled_pairs``, ``fused=True`` on cc 10.7;
``fused=False`` = ``e8m0_rcp`` FMUL + the plain pair cvt).  The sm107 d256 MXFP8 backward's ``ds_dq`` payload is this shape.

* tier 1 (any box whose cutlass-dsl knows ``sm_107a``): the sm_107a trace-compile of a one-warp probe (lane = row, 64 columns, both
  arms + both pair helpers) is asserted; with a decoding nvdisasm the SASS pins: 64 ``CREDUX.MAXABS.F32``, 64 ``SCALE_BY_C`` (the fused
  arm: one lone cvt per element), 32 plain ``E4M3`` cvts (the FMUL arm: one pair cvt per two elements), 64 ``F2FP...E8`` (32 ``e8m0_pair_u``
  + 32 ``e8m0_pair``), >= 32 ``FMUL2`` (``e8m0_pair_u``'s one packed multiply per pair), no spill.  ``MOV R, UR`` is REPORTED: a CREDUX
  result lands in a uniform register and sm_107a ptxas moves it into the vector file before any ALU op reads it, in every spelling
  probed (13 forms, 64 moves each -- the floor; ``pointwise.e8m0_pair_u``).
* tier 2 (``requires_rubin``): on random log-uniform-magnitude normal fp32 rows (no fp32 subnormal: the one documented divergence of
  the fused op) (1) ``e8m0_pair_u`` == ``e8m0_pair`` BITWISE on every column pair (the same rule, op for op); (2) both == the torch
  rule ``ue8m0_rp(amax * fp32(1/448))`` per column; (3) the fused and the FMUL arms are BITWISE identical; (4) both == the torch
  DEscale oracle ``e4m3_rn_satfinite(x * 2^(127 - e_col))`` with element i in byte i of word i // 4.
"""

import glob
import importlib.util
import os
import struct
import subprocess
import sys
import textwrap

import pytest
import torch

from frost_test_utils import arch_known_to_the_dsl, nvdisasm_candidates, requires_dsl, requires_rubin

pytestmark = [pytest.mark.L0, requires_dsl]

N_COLS = 64  # the softmax lane's q columns (one warpgroup half of the d256 backward)
E8M0_RCP_E4M3_MAX_BITS = 0x3B124925  # fp32(1/448), the rule's constant (pointwise.E8M0_RCP_E4M3_MAX_BITS)

_PROBE_SRC = textwrap.dedent('''
    """Probe: one warp per block, lane = row of 64 fp32; per column a redux abs-max; the column pairs' E8M0 words by e8m0_pair_u AND
    e8m0_pair; the 16 payload words by fp32_to_fp8x4_scaled_pairs, fused and FMUL arms; everything stored per lane."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from cudnn.frost.tile_dsl.pointwise import e8m0_pair, e8m0_pair_u, fp32_to_fp8x4_scaled_pairs, opaque_e4m3_max_rcp_in_lane, warp_abs_max_f32
    from cudnn.frost.tile_dsl.tma import ld_global, st_global

    N = 64


    @cute.kernel
    def probe(mX: cute.Tensor, mFused: cute.Tensor, mUnfused: cute.Tensor, mPairU: cute.Tensor, mPairRef: cute.Tensor):
        tidx = cutlass.Int32(cute.arch.thread_idx()[0])
        bidx = cutlass.Int32(cute.arch.block_idx()[0])
        lane = bidx * cutlass.Int32(32) + tidx
        row64 = lane.to(cutlass.Int64) * cutlass.Int64(N * 4)
        inv = opaque_e4m3_max_rcp_in_lane(tidx.to(cutlass.Float32))
        xs = []
        for j in cutlass.range_constexpr(N):
            xs.append(ld_global(mX.iterator.toint() + row64 + cutlass.Int64(4 * j), cutlass.Float32))
        for g in cutlass.range_constexpr(N // 4):
            am = [warp_abs_max_f32(xs[4 * g + i]) for i in range(4)]
            p01 = e8m0_pair_u(am[0], am[1], inv)
            p23 = e8m0_pair_u(am[2], am[3], inv)
            _r0, _r1, q01 = e8m0_pair(am[0], am[1])
            _r2, _r3, q23 = e8m0_pair(am[2], am[3])
            vals = [xs[4 * g + i] for i in range(4)]
            w_f = fp32_to_fp8x4_scaled_pairs(vals, p01, p23, dtype=cutlass.Float8E4M3FN, fused=True)
            w_u = fp32_to_fp8x4_scaled_pairs(vals, q01, q23, dtype=cutlass.Float8E4M3FN, fused=False)
            word64 = lane.to(cutlass.Int64) * cutlass.Int64(N) + cutlass.Int64(4 * g)
            st_global(mFused.iterator.toint() + word64, w_f, cutlass.Int32)
            st_global(mUnfused.iterator.toint() + word64, w_u, cutlass.Int32)
            pair64 = lane.to(cutlass.Int64) * cutlass.Int64(N * 2) + cutlass.Int64(8 * g)
            st_global(mPairU.iterator.toint() + pair64, p01, cutlass.Int32)
            st_global(mPairU.iterator.toint() + pair64 + cutlass.Int64(4), p23, cutlass.Int32)
            st_global(mPairRef.iterator.toint() + pair64, q01, cutlass.Int32)
            st_global(mPairRef.iterator.toint() + pair64 + cutlass.Int64(4), q23, cutlass.Int32)


    probe.set_name_prefix("cudnn", remove_cutlass_symbol=True)


    @cute.jit
    def launch(x: cute.Tensor, fused: cute.Tensor, unfused: cute.Tensor, pair_u: cute.Tensor, pair_ref: cute.Tensor, n_warps: cutlass.Int32, stream: cuda.CUstream):
        probe(x, fused, unfused, pair_u, pair_ref).launch(grid=(n_warps, 1, 1), block=(32, 1, 1), stream=stream)


    if __name__ == "__main__":
        # argv: <arch> <dump dir> [nvdisasm candidates...]; CUTE_DSL_DUMP_DIR set by the caller (keeps ptx + cubin).
        import glob, os, re, subprocess, sys
        from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

        arch, dump, cands = sys.argv[1], sys.argv[2], sys.argv[3:]

        def fake(dt):
            return make_fake_tensor(dtype=dt, shape=(cute.sym_int(),), stride=(1,), assumed_align=16)

        cute.compile(launch, fake(cutlass.Float32), fake(cutlass.Int32), fake(cutlass.Int32), fake(cutlass.Int32), fake(cutlass.Int32), cutlass.Int32(1),
                     make_fake_stream(use_tvm_ffi_env_stream=False), options=f"--enable-tvm-ffi --gpu-arch {arch} --keep-cubin --keep-ptx")
        print("COMPILED", arch)
        ptxs = glob.glob(os.path.join(dump, f"*.{arch}.ptx"))
        if not ptxs:
            print("FAIL no ptx landed in", dump, os.listdir(dump)); sys.exit(3)
        ptx = open(ptxs[0]).read()
        print("PTX REDUX_MAX_ABS", ptx.count("redux.sync.max.abs.f32"))
        print("PTX SCALED_CVT", ptx.count("cvt.rn.satfinite.scaled::n1::ue8m0.e4m3x2.f32"))
        print("PTX E8M0_CVT", ptx.count("cvt.rp.satfinite.ue8m0x2.f32"))
        print("PTX MUL_F32X2", ptx.count("mul.rn.f32x2"))
        cubins = glob.glob(os.path.join(dump, f"*.{arch}.cubin"))
        if not cubins:
            print("FAIL no cubin landed in", dump); sys.exit(3)
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
        def cnt(*subs):
            return sum(1 for ln in sass if all(sb in ln for sb in subs))
        print("SASS CREDUX", cnt("CREDUX.MAXABS.F32"))
        print("SASS SCALE_BY_C", cnt("F2FP", "SCALE_BY_C"))
        print("SASS PLAIN_E4M3", sum(1 for ln in sass if "F2FP" in ln and "E4M3" in ln and "SCALE_BY_C" not in ln))
        print("SASS E8M0_CVT", cnt("F2FP", ".E8.F32"))
        print("SASS FMUL2", sum(1 for ln in sass if re.search(r"\\bFMUL2\\b", ln)))
        print("SASS FMUL", sum(1 for ln in sass if re.search(r"\\bFMUL\\b", ln)))
        print("SASS MOV_R_UR", sum(1 for ln in sass if re.search(r"\\bMOV R\\d+, UR\\d+", ln)))
        print("SASS SPILL", sum(1 for ln in sass if " STL" in ln or " LDL" in ln))
        print("SASS LINES", len(sass))
    ''')


def _write_probe(tmp_path):
    probe_py = tmp_path / "fp8_x4_scaled_pairs_probe.py"
    probe_py.write_text(_PROBE_SRC)
    return probe_py


def test_sm107a_form(tmp_path):
    """sm_107a trace-compile: PTX carries 64 redux, 64 lone scaled cvts (fused), 33 + 32 ... E8M0 cvts (one per pair per helper), 32
    packed multiplies; SASS (with a decoding nvdisasm): 64 CREDUX, 64 SCALE_BY_C, 32 plain e4m3 cvts, 64 E8 cvts, >= 32 FMUL2, no spill."""
    if not arch_known_to_the_dsl("sm_107a"):
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0)")
    dump = tmp_path / "dump_sm107a"
    dump.mkdir()
    env = dict(os.environ, CUTE_DSL_DUMP_DIR=str(dump), CUTE_DSL_KEEP="ptx,cubin")
    proc = subprocess.run(
        [sys.executable, str(_write_probe(tmp_path)), "sm_107a", str(dump), *nvdisasm_candidates()], capture_output=True, text=True, timeout=900, env=env
    )
    assert proc.returncode == 0, f"sm_107a trace-compile failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    lines = proc.stdout.splitlines()
    assert "COMPILED sm_107a" in lines, proc.stdout[-2000:]
    ptx = {ln.split()[1]: int(ln.split()[2]) for ln in lines if ln.startswith("PTX ")}
    assert ptx["REDUX_MAX_ABS"] == N_COLS, ptx
    assert ptx["SCALED_CVT"] == N_COLS, f"one lone scaled cvt per element on the fused arm -- {ptx}"
    assert ptx["E8M0_CVT"] == N_COLS, f"one e8m0 cvt per column pair per helper (e8m0_pair_u + e8m0_pair) -- {ptx}"
    assert ptx["MUL_F32X2"] == N_COLS // 2, f"e8m0_pair_u: ONE packed multiply per column pair -- {ptx}"
    if not nvdisasm_candidates():
        pytest.skip("compiled + PTX pinned; no nvdisasm executable to try (CUDA_PATH unset and none on PATH)")
    if any(ln.startswith("SKIP") for ln in lines):
        pytest.skip(f"compiled + PTX pinned; SASS half skipped: {[ln for ln in lines if ln.startswith(('SKIP', 'REJECT'))]}")
    sass = {ln.split()[1]: int(ln.split()[2]) for ln in lines if ln.startswith("SASS ")}
    print(f"\nfp8 x4 scaled pairs sm_107a PTX {ptx} SASS {sass} (MOV R,UR = {sass['MOV_R_UR']}: the uniform-register floor, one per CREDUX)")
    assert sass["CREDUX"] == N_COLS, sass
    assert sass["SCALE_BY_C"] == N_COLS, f"the fused arm is one SCALE_BY_C per element -- {sass}"
    assert sass["PLAIN_E4M3"] == N_COLS // 2, f"the FMUL arm is one plain pair cvt per two elements -- {sass}"
    assert sass["E8M0_CVT"] == N_COLS, sass
    assert sass["FMUL2"] >= N_COLS // 2, f"e8m0_pair_u's packed multiplies -- {sass}"
    assert sass["SPILL"] == 0, sass


def _f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def ue8m0_rp(scaled: torch.Tensor) -> torch.Tensor:
    """``cvt.rp.satfinite.ue8m0x2.f32`` of a finite non-negative fp32: the smallest power of two >= ``scaled`` as a biased exponent
    (``0 -> 0x00``; a power of two is its own exponent; saturates at 254).  Exact through ``frexp`` (no log2 rounding)."""
    m, k = torch.frexp(scaled)  # scaled = m * 2^k, m in [0.5, 1)
    e = torch.where(m == 0.5, k - 1, k) + 127  # ceil(log2(scaled)) + 127
    e = torch.where(scaled == 0, torch.zeros_like(e), e)
    return e.clamp(0, 254).to(torch.int32)


def torch_oracle(x: torch.Tensor):
    """``x``: ``[n_warps, 32, 64]`` fp32.  Returns ``(e [n_warps, 64] int32, words [n_warps, 32, 16] int32)`` = the per-column E8M0
    byte of the rule ``ue8m0_rp(amax * fp32(1/448))`` and the e4m3 payload ``e4m3_rn_satfinite(x * 2^(127 - e))``, element i in byte
    i of word i // 4."""
    amax = x.abs().amax(dim=1)  # [n_warps, 64]
    scaled = amax * torch.tensor(_f32(E8M0_RCP_E4M3_MAX_BITS), dtype=torch.float32, device=x.device)
    e = ue8m0_rp(scaled)
    descale = ((254 - e.to(torch.int64)) << 23).to(torch.int32).view(torch.float32)  # 2^(127 - e), e <= 253 on these inputs
    y = (x * descale.unsqueeze(1)).clamp(-448.0, 448.0)
    b = y.to(torch.float8_e4m3fn).view(torch.uint8)  # [n_warps, 32, 64]
    words = b.reshape(x.shape[0], 32, 16, 4).to(torch.int32)
    words = words[..., 0] | (words[..., 1] << 8) | (words[..., 2] << 16) | (words[..., 3] << 24)
    return e, words


def _inputs(n_warps: int, seed: int = 20261001) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    mag = torch.exp2(torch.rand(n_warps, 1, N_COLS, generator=g) * 40.0 - 20.0)  # per column magnitude 2^-20 .. 2^20 (never an fp32 subnormal)
    x = torch.randn(n_warps, 32, N_COLS, generator=g) * mag
    x[0, :, :4] = 0.0  # an all-zero column block: byte 0x00, payload 0
    x[1, 0, 8] = 448.0  # a block amax exactly 448: scaled == 1.0 -> e = 127 (the power-of-two corner)
    x[1, 1:, 8] = 0.0
    x[1, 0, 9] = 449.0  # just above: e = 128
    x[1, 1:, 9] = 0.0
    return x.contiguous()


@requires_rubin
def test_pairs_and_payload_match_the_shipped_rule_and_the_oracle_on_device(tmp_path):
    import cuda.bindings.driver as _cuda_driver
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    spec = importlib.util.spec_from_file_location("_fp8_x4_scaled_pairs_probe_device", str(_write_probe(tmp_path)))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    dev = torch.device("cuda")
    n_warps = 256
    x = _inputs(n_warps).to(dev)
    fused = torch.full((n_warps * 32 * 16,), -1, dtype=torch.int32, device=dev)
    unfused = torch.full_like(fused, -1)
    pair_u = torch.full((n_warps * 32 * 32,), -1, dtype=torch.int32, device=dev)
    pair_ref = torch.full_like(pair_u, -1)
    stream = _cuda_driver.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    args = (
        from_dlpack(x.reshape(-1), assumed_align=16),
        from_dlpack(fused, assumed_align=16),
        from_dlpack(unfused, assumed_align=16),
        from_dlpack(pair_u, assumed_align=16),
        from_dlpack(pair_ref, assumed_align=16),
        cutlass.Int32(n_warps),
    )
    cute.compile(mod.launch, *args, stream)(*args, stream)
    torch.cuda.synchronize()
    fused = fused.reshape(n_warps, 32, 16)
    unfused = unfused.reshape(n_warps, 32, 16)
    pair_u = pair_u.reshape(n_warps, 32, 32)
    pair_ref = pair_ref.reshape(n_warps, 32, 32)
    assert int((pair_u == -1).sum()) == 0 and int((pair_ref == -1).sum()) == 0, "unwritten pair words"
    # (1) e8m0_pair_u == e8m0_pair bitwise on every (lane, pair): the same rule, op for op
    n_pair = int((pair_u != pair_ref).sum())
    assert n_pair == 0, f"e8m0_pair_u differs from e8m0_pair on {n_pair} of {pair_u.numel()} pair words"
    # the pair words are warp-uniform
    assert torch.equal(pair_u, pair_u[:, :1, :].expand_as(pair_u)), "a pair word differs across the lanes of a warp"
    # (2) the bytes == the torch rule per column
    e_ref, words_ref = torch_oracle(x)
    e_dev = torch.stack([pair_u[:, 0, :] & 0xFF, (pair_u[:, 0, :] >> 8) & 0xFF], dim=-1).reshape(n_warps, N_COLS)
    n_e = int((e_dev != e_ref).sum())
    assert (
        n_e == 0
    ), f"E8M0 bytes differ from ue8m0_rp(amax * fp32(1/448)) on {n_e} of {e_dev.numel()} columns; first: dev {e_dev[e_dev != e_ref][:8].tolist()} ref {e_ref[e_dev != e_ref][:8].tolist()}"
    assert int((pair_u >> 16).abs().sum()) == 0, "bits 16..31 of a pair word must be zero"
    assert int(e_ref[0, :4].sum()) == 0 and int(e_ref[1, 8]) == 127 and int(e_ref[1, 9]) == 128, "the planted corners are not what the test thinks"
    # (3) fused == FMUL arm bitwise (normal inputs, e <= 253)
    n_arm = int((fused != unfused).sum())
    assert n_arm == 0, f"fused and FMUL arms differ on {n_arm} of {fused.numel()} payload words"
    # (4) == the torch DEscale oracle, element i in byte i of word i // 4
    bad = fused != words_ref
    assert (
        int(bad.sum()) == 0
    ), f"payload differs from e4m3(x * 2^(127 - e_col)) on {int(bad.sum())} of {fused.numel()} words; first: dev {fused[bad][:4].tolist()} ref {words_ref[bad][:4].tolist()}"
    assert int(fused[0, :, 0].abs().sum()) == 0, "an all-zero block quantizes to zero payload bytes"
