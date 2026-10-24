# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The LayerNorm-prologue dual GEMM's tcgen05 SMEM-descriptor builder (``ln_prologue.tcgen05_smem_desc_builder``) -- host tests, no GPU.

The two bodies build their operand ring-ROOT descriptors through ONE builder resolved against the installed CuTe DSL: ``Tcgen05SmemDesc.build(...,
version=1)`` where the DSL's build() takes a version (nvidia-cutlass-dsl 4.8.0+), the non-versioned build() of the released 4.7.0 floor on the
sm100 body (whose 227 KiB part cannot place a root at or past the 256 KiB line where that builder's 14-bit start-address field wraps), and a
typed decline on the sm107 body (whose rings end at 288 KiB) when the DSL has no versioned builder.  Pinned here on the INSTALLED DSL, so one
module proves the floor DSL's path under DSL 4.7.0 and the versioned path under DSL 4.8.0:

  * the dispatch (which spelling each resolution calls, the decline's text) against fake build() signatures, and the resolution on the live DSL;
  * ``build_ln_dual_gemm`` declines typed, BEFORE loading a body, where the sm107 body needs the wide field and the builder is non-versioned;
  * the source pin: every ring-root descriptor of both bodies goes through the builder (no raw ``Tcgen05SmemDesc.build`` / ``version=`` literal);
  * the ENCODINGS: a trace-compiled sm_100a micro-kernel stores descriptors built from constant operands (the bodies' LBO / SBO / swizzle sets at
    ring roots below and above the line); the PTX immediates agree bit-for-bit between the sm100 body's builder and the non-versioned NVVM op
    (= what the 4.7.0 build() lowers to) below the line and match the PTX ISA layout, while at / above the line the non-versioned op drops
    start-address bit 14 -- the wide field the sm107 body requires;
  * K1 and K3 RENDER for sm_100a through the builder (the production c 128 picks), the resolution the body reports matching the DSL.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import textwrap

import pytest

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Triangle multiplicative update descriptor-builder tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)
pytest.importorskip("cutlass")

pytestmark = pytest.mark.L0

from cudnn.gemm.frost import ln_prologue as sub  # noqa: E402
from cudnn.triangle_multiplicative_update.kernels import trimul_proj as k1  # noqa: E402
from cudnn.triangle_multiplicative_update.kernels import trimul_tail as k3  # noqa: E402
from sdpa.frost import frost_test_utils as sass_utils  # noqa: E402  -- arch_known_to_the_dsl (the sm_100a trace-compile gate)

LIMIT = sub.TCGEN05_SMEM_DESC_LEGACY_ADDR_LIMIT  # 262144


def _installed_build_is_versioned() -> bool:
    from cutlass.experimental.primitives import Tcgen05SmemDesc

    return "version" in inspect.signature(Tcgen05SmemDesc.build).parameters


def _fake_build(versioned: bool):
    """A stand-in ``Tcgen05SmemDesc.build`` with the 4.8.0 (versioned) or the 4.7.0 (non-versioned) signature; records every call's kwargs."""
    calls = []
    if versioned:

        def build(cls, start_address, leading_byte_offset=0, stride_byte_offset=0, version=1, base_offset=0, layout=0, leading_dim_mode=0, k_segment_offset=0):
            calls.append(
                dict(
                    start_address=start_address, leading_byte_offset=leading_byte_offset, stride_byte_offset=stride_byte_offset, version=version, layout=layout
                )
            )
            return "desc"

    else:

        def build(cls, start_address, leading_byte_offset=0, stride_byte_offset=0, base_offset=0, layout=0, leading_dim_mode=0):
            calls.append(dict(start_address=start_address, leading_byte_offset=leading_byte_offset, stride_byte_offset=stride_byte_offset, layout=layout))
            return "desc"

    return build, calls


def _patch_build(monkeypatch, versioned: bool):
    from cutlass.experimental.primitives import Tcgen05SmemDesc

    build, calls = _fake_build(versioned)
    monkeypatch.setattr(Tcgen05SmemDesc, "build", classmethod(build))
    return calls


def test_wide_start_address_is_required_by_the_rubin_body_only():
    """The 320 KiB Rubin carveout reaches past the non-versioned builder's 256 KiB window; the 227 KiB Blackwell one cannot (whatever the layout)."""
    assert LIMIT == 256 * 1024 and sub.TCGEN05_SMEM_DESC_VERSION == 1
    assert sub.smem_desc_wide_start_address_required("sm107") is True
    assert sub.smem_desc_wide_start_address_required("sm100") is False
    assert sub.SMEM_BUDGET_BYTES_BY_ARCH["sm100"] < LIMIT < sub.SMEM_BUDGET_BYTES_BY_ARCH["sm107"]
    with pytest.raises(ValueError, match="rendered body"):
        sub.smem_desc_wide_start_address_required("sm90")


def test_builder_passes_version_only_where_the_installed_build_has_it(monkeypatch):
    """A versioned build(): BOTH bodies get ``version=1`` explicitly (never the DSL default).  A non-versioned build() (the released 4.7.0): the
    sm100 body calls it as-is, the sm107 body gets a declining builder -- a typed ``NotImplementedError`` naming the installed DSL and the
    requirement on first use, the same text ``tcgen05_smem_desc_requirement_error`` reports -- and the body module itself still loads."""
    from cudnn.frost.buffers import cutedsl_state

    calls = _patch_build(monkeypatch, versioned=True)
    for arch in ("sm100", "sm107"):
        b = sub.tcgen05_smem_desc_builder(arch)
        assert (b.arch, b.path) == (arch, "versioned") and sub.tcgen05_smem_desc_requirement_error(arch) is None
        assert b(1024, leading_byte_offset=16, stride_byte_offset=1024, layout=2) == "desc"
    assert calls == [dict(start_address=1024, leading_byte_offset=16, stride_byte_offset=1024, version=sub.TCGEN05_SMEM_DESC_VERSION, layout=2)] * 2

    calls = _patch_build(monkeypatch, versioned=False)
    legacy = sub.tcgen05_smem_desc_builder("sm100")
    assert legacy.path == "legacy" and sub.tcgen05_smem_desc_requirement_error("sm100") is None
    assert legacy(8192, leading_byte_offset=8192, stride_byte_offset=1024, layout=2) == "desc"
    assert calls == [dict(start_address=8192, leading_byte_offset=8192, stride_byte_offset=1024, layout=2)]  # no version kwarg: the 4.7.0 spelling
    declined = sub.tcgen05_smem_desc_builder("sm107")
    assert declined.path == "declined"
    msg = sub.tcgen05_smem_desc_requirement_error("sm107")
    _, version = cutedsl_state()
    assert msg and "sm107" in msg and str(LIMIT) in msg and "'version'" in msg and "nvidia-cutlass-dsl >= 4.8.0" in msg and version[1] in msg
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        declined(1024, leading_byte_offset=16, stride_byte_offset=1024, layout=2)
    assert calls == calls[:1]  # the declining builder never reaches the DSL


def test_builder_resolution_on_the_installed_dsl():
    """On the live DSL: a build() with ``version`` resolves both bodies to the versioned path; without it (the 4.7.0 floor) the sm100 body to the
    legacy path and the sm107 body to the typed decline -- the requirement the sm_107a target gate already holds at 4.8.0+."""
    versioned = _installed_build_is_versioned()
    assert sub.tcgen05_smem_desc_builder("sm100").path == ("versioned" if versioned else "legacy")
    assert sub.tcgen05_smem_desc_builder("sm107").path == ("versioned" if versioned else "declined")
    assert (sub.tcgen05_smem_desc_requirement_error("sm107") is None) is versioned
    assert sub.tcgen05_smem_desc_requirement_error("sm100") is None


def test_build_ln_dual_gemm_declines_typed_before_loading_the_sm107_body_without_the_versioned_builder(monkeypatch):
    """The substrate's own gate: a DSL that knows sm_107a but carries the non-versioned build() (the target gate passes) is declined with the
    descriptor requirement BEFORE the body is loaded; the sm100 body passes the same gate and proceeds to the load."""
    import cudnn.frost.buffers as buffers
    import cudnn.frost.template_loader as loader

    class _Loaded(Exception):
        pass

    def _no_load(*a, **k):
        raise _Loaded()

    _patch_build(monkeypatch, versioned=False)
    monkeypatch.setattr(buffers, "cutedsl_arch_requirement_error", lambda cc: None)
    monkeypatch.setattr(loader, "load_template", _no_load)
    p = k1.proj_params(128, 128, has_bias=True, has_mask=True)
    with pytest.raises(NotImplementedError, match="versioned tcgen05 SMEM-descriptor builder"):
        sub.build_ln_dual_gemm.__wrapped__(p, "sm107")  # __wrapped__: past the lru_cache, so the gate runs whatever this process compiled before
    with pytest.raises(_Loaded):
        sub.build_ln_dual_gemm.__wrapped__(p, "sm100")


def test_both_bodies_build_every_ring_root_through_the_builder():
    """Source pin: the four ring-root descriptor sites of each body call ``_DESC_BUILD`` (one builder per body, resolved from ``BODY_ARCH``), and no
    raw ``Tcgen05SmemDesc.build(`` / ``version=`` literal remains -- a raw ``version=`` is the 4.7.0 TypeError this module exists for."""
    for arch in ("sm107", "sm100"):
        src = open(sub.body_path(arch)).read()
        assert src.count("_DESC_BUILD(") == 4, arch
        assert "_DESC_BUILD = tcgen05_smem_desc_builder(BODY_ARCH)" in src, arch
        code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
        assert "Tcgen05SmemDesc.build(" not in code and "version=" not in code and "DESC_VERSION" not in code, arch


# ---------------------------------------------------------------------------------------------------------------------------------------------
# The ENCODINGS.  Constant operands let LLVM fold every descriptor into a PTX immediate; the probe reads them back from the stores.
# Ring roots: below the line (every address a 227 KiB cc 10.0 / 10.3 CTA can hold; the sm100 body's rings end at 192 KiB) and at / above it
# (reachable on the 327 KiB Rubin carveout only: the sm107 body's rings end at 288 KiB).  (LBO, SBO): the bodies' K-major SW128 bf16 operands
# (A and B: 16 / 1024) and the K3 arm's MN-major SW128 bf16 A0 (8192 / 1024); SWIZZLE_128B (tcgen05 code 2) throughout.
_ROOTS = (1024, 16384, 99328, 197632, 247808, 261120, 262144, 263168, 293888)
_LBO_SBO = ((16, 1024), (8192, 1024))
_CASES = [(r, l, s) for r in _ROOTS for (l, s) in _LBO_SBO]


def _expected(root: int, lbo: int, sbo: int, address_bits: int) -> int:
    """The PTX ISA tcgen05 SMEM-descriptor layout: start / LBO / SBO in 16-B units (the non-versioned op keeps 14 bits of each, the versioned
    op 15), version 1 at bits 46-48, base offset 0, lead-dim mode 0, SWIZZLE_128B (2) at bits 61-63."""
    mask = (1 << address_bits) - 1
    return ((root >> 4) & mask) | (((lbo >> 4) & mask) << 16) | (((sbo >> 4) & mask) << 32) | (1 << 46) | (2 << 61)


_ENCODING_PROBE = r"""
import glob, json, os, re, shutil, sys, time
dump, arch, modes, cases = sys.argv[1], sys.argv[2], sys.argv[3].split(","), json.loads(sys.argv[4])
os.environ["CUTE_DSL_DUMP_DIR"] = dump          # read once, at the first cutlass import
os.environ["CUTE_DSL_KEEP"] = "ptx"              # the constant-folded descriptors are PTX store immediates
os.environ["CUTE_DSL_ARCH"] = arch               # no device needed for a trace-compile
import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as prims
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.experimental.primitives import descriptors as _descs
from cuda.bindings import driver as cuda
from cudnn.gemm.frost import ln_prologue as sub
SWZ = prims.Tcgen05SmemSwizzle.SWIZZLE_128B
BUILDERS = {a: sub.tcgen05_smem_desc_builder(a) for a in ("sm100", "sm107")}

def build(mode, root, lbo, sbo):
    if mode in BUILDERS:  # the production builder of that body
        return BUILDERS[mode](root, leading_byte_offset=lbo, stride_byte_offset=sbo, layout=SWZ)
    if mode == "op_v1":   # the non-versioned NVVM op: what the 4.7.0 build() lowers to (module-private, present in 4.7.0 and 4.8.0)
        return _descs._tcgen05_mma_smem_desc(root >> 4, lbo >> 4, sbo >> 4, 0, 0, int(SWZ))
    raise ValueError(mode)

def make_host(mode):
    @cute.kernel
    def _k(out: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            for i in cutlass.range_constexpr(len(cases)):
                r, l, s = cases[i]
                out[i] = cutlass.Int64(build(mode, r, l, s))

    @cute.jit
    def _host(out: cute.Tensor, stream: cuda.CUstream):
        _k(out).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    return _host

def parse(path):
    txt = open(path).read()
    imm = {}
    for m in re.finditer(r"mov\.(?:b64|u64|s64)\s+(%rd\d+),\s*(-?\d+);", txt):
        imm[m.group(1)] = int(m.group(2)) & 0xFFFFFFFFFFFFFFFF
    vals = {}
    for m in re.finditer(r"st\.global(?:\.v(\d))?\.(?:u64|b64|s64)\s+\[(%rd\d+)(?:\+(\d+))?\],\s*\{?([^;}]+)\}?;", txt):
        width, off, ops = int(m.group(1) or 1), int(m.group(3) or 0), [o.strip() for o in m.group(4).split(",")]
        assert len(ops) == width, (ops, width)
        for k, op in enumerate(ops):
            vals[off // 8 + k] = (int(op) & 0xFFFFFFFFFFFFFFFF) if re.fullmatch(r"-?\d+", op) else imm.get(op)
    return vals

for mode in modes:
    print("PATH", mode, BUILDERS[mode].path if mode in BUILDERS else "nvvm")
    t0 = time.time()
    fake_out = make_fake_compact_tensor(cutlass.Int64, (len(cases),), stride_order=(0,), assumed_align=16)
    cute.compile(make_host(mode), fake_out, stream=make_fake_stream(use_tvm_ffi_env_stream=False))
    ptx = sorted((f for f in glob.glob(os.path.join(dump, "*.ptx")) if os.path.getmtime(f) >= t0 - 1), key=os.path.getmtime)
    assert ptx, ("no PTX dumped", mode, os.listdir(dump))
    kept = os.path.join(dump, mode + ".ptx")
    shutil.copyfile(ptx[-1], kept)  # the DSL names the dump by the function signature: every mode overwrites the same file
    vals = parse(kept)
    for i in range(len(cases)):
        print("DESC", mode, i, "none" if vals.get(i) is None else "0x%016x" % vals[i])
print("DONE")
"""


def test_descriptor_encodings_agree_below_the_line_on_the_installed_dsl(tmp_path):
    """Trace-compile for sm_100a (no GPU) the descriptors of the bodies' operand sets at ring roots below and above the 256 KiB line through the
    sm100 body's builder, the non-versioned NVVM op (= the 4.7.0 build()) and -- where the DSL has it -- the sm107 body's versioned builder:
    below the line every path encodes the SAME 64-bit value, the PTX ISA layout; at / above it the non-versioned op drops start-address bit 14
    (the wrap the Rubin body's wide-field requirement exists for) while the versioned builder keeps it."""
    if not sass_utils.arch_known_to_the_dsl("sm_100a"):
        pytest.skip("this cutlass-dsl has no sm_100a")
    versioned = _installed_build_is_versioned()
    modes = ["sm100", "op_v1"] + (["sm107"] if versioned else [])
    probe = tmp_path / "encoding_probe.py"  # a FILE: the DSL reads the kernel's source
    probe.write_text(textwrap.dedent(_ENCODING_PROBE))
    dump = tmp_path / "dump"
    dump.mkdir()
    proc = subprocess.run([sys.executable, str(probe), str(dump), "sm_100a", ",".join(modes), json.dumps(_CASES)], capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0 and "DONE" in proc.stdout, f"the encoding probe failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    paths = {ln.split()[1]: ln.split()[2] for ln in proc.stdout.splitlines() if ln.startswith("PATH ")}
    assert paths["sm100"] == ("versioned" if versioned else "legacy") and paths["op_v1"] == "nvvm"
    vals = {m: {} for m in modes}
    for ln in proc.stdout.splitlines():
        if ln.startswith("DESC "):
            _, mode, i, v = ln.split()
            vals[mode][int(i)] = None if v == "none" else int(v, 16)
    print(
        f"\nsm_100a descriptor immediates ({len(_CASES)} cases): "
        + json.dumps({m: {str(i): (None if v is None else f"0x{v:016x}") for i, v in d.items()} for m, d in vals.items()})
    )
    assert all(len(vals[m]) == len(_CASES) and None not in vals[m].values() for m in modes), vals
    assert any(r >= LIMIT for r, _, _ in _CASES) and any(r < LIMIT for r, _, _ in _CASES)
    for i, (root, lbo, sbo) in enumerate(_CASES):
        narrow, wide = _expected(root, lbo, sbo, 14), _expected(root, lbo, sbo, 15)
        assert vals["op_v1"][i] == narrow, (i, root, hex(vals["op_v1"][i]), hex(narrow))
        if root < LIMIT:
            assert narrow == wide and vals["sm100"][i] == narrow, (i, root, hex(vals["sm100"][i]), hex(narrow))
            if versioned:
                assert vals["sm107"][i] == narrow
        else:
            assert wide == narrow | (1 << 14) and vals["sm100"][i] == (wide if versioned else narrow), (i, root, hex(vals["sm100"][i]))
            if versioned:
                assert vals["sm107"][i] == wide, (i, root, hex(vals["sm107"][i]), hex(wide))


_RENDER_PROBE = """
import glob, hashlib, json, os, sys
dump, arch, params_json = sys.argv[1], sys.argv[2], sys.argv[3]
os.environ["CUTE_DSL_DUMP_DIR"] = dump
os.environ["CUTE_DSL_KEEP"] = "cubin"
os.environ["CUTE_DSL_ARCH"] = arch
os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"  # a compiled-plan cache HIT skips the render
from cudnn.gemm.frost import ln_prologue
from cudnn.triangle_multiplicative_update.kernels import trimul_proj as k1
from cudnn.triangle_multiplicative_update.kernels import trimul_tail as k3
kw = json.loads(params_json)
if kw["arm"] == "k1":
    p = k1.proj_params(kw["k"], kw["k"], has_bias=True, has_mask=True, want_stats=False, pass_order="interleaved", clc_stages=2, ln_pass_form="registers", epi_math="packed")
else:
    p = k3.tail_params(kw["k"], kw["k"], has_bias=True, want_stats=False, a0_prologue="transpose", ln_pass_form="registers", epi_math="packed")
plan = ln_prologue.build_ln_dual_gemm(p, "sm100")
print("DESC_PATH", plan.module._DESC_BUILD.path)
print("AB_STAGES", plan.module.ab_stages)
cubins = sorted(glob.glob(os.path.join(dump, "*.cubin")), key=os.path.getmtime)
print("CUBIN_MD5", hashlib.md5(open(cubins[-1], "rb").read()).hexdigest() if cubins else "none")
print("DONE")
"""


@pytest.mark.parametrize("arm,k", [("k1", 128), ("k3", 128)])
def test_k1_and_k3_render_for_sm_100a_through_the_builder(tmp_path, arm, k):
    """Trace-compile the sm100 body's production c 128 renderings of K1 (the LayerNorm-prologue projection) and K3 (the output stage) for sm_100a
    (no GPU) in a fresh interpreter: the render succeeds on the installed DSL -- the versioned path on a DSL whose build() takes ``version``, the
    legacy path on the 4.7.0 floor (the ``version=`` TypeError this module exists for) -- and the body reports the resolution the DSL has."""
    if not sass_utils.arch_known_to_the_dsl("sm_100a"):
        pytest.skip("this cutlass-dsl has no sm_100a")
    dump = tmp_path / f"{arm}_{k}"
    dump.mkdir()
    proc = subprocess.run(
        [sys.executable, "-c", _RENDER_PROBE, str(dump), "sm_100a", json.dumps({"arm": arm, "k": k})], capture_output=True, text=True, timeout=1500
    )
    assert proc.returncode == 0 and "DONE" in proc.stdout, f"sm_100a trace-compile of {arm} K={k} failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    out = {ln.split()[0]: ln.split()[1] for ln in proc.stdout.splitlines() if len(ln.split()) == 2 and ln.split()[0].isupper()}
    print(f"\n{arm} K={k} sm_100a: builder {out['DESC_PATH']}, ab_stages {out['AB_STAGES']}, cubin md5 {out['CUBIN_MD5']}")
    assert out["DESC_PATH"] == ("versioned" if _installed_build_is_versioned() else "legacy")
    assert int(out["AB_STAGES"]) == sub.ring_stages("sm100", "accumulators" if arm == "k3" else "columns")
    assert out["CUBIN_MD5"] != "none"
