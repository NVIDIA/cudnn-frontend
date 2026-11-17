# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``tile_dsl.wgmma`` SM90 forms and the geometry parameters of the ``sm90/_common_hopper`` MMA helpers.

* host (no device): ``wgmma._rs_asm`` builds the RS inline assembly from N. Its m64n256k16 text and constraints are
  develop's literal byte for byte (the D512 PV instruction), m64n128k16 numbers its operands C ``$0..$63`` (tied inputs
  ``$64..$127``), A ``$128..$131`` and B ``$132``, and every other N is refused.
* sm_90a trace-compile (any box whose cutlass-dsl knows ``sm_90a``; no Hopper device, ``CUTE_DSL_ARCH`` needs none): each
  form lowers to exactly one ``wgmma.mma_async`` of its shape and operand kind, and the D128-geometry step (``mma_qk``
  issue-only plus the caller's wait, then RS ``mma_pv`` over eight k16 steps) lowers to 8 SS + 8 RS ``m64n128k16``, two
  commits and exactly two ``wait_group 0``: an internal wait in issue-only mode would make three. When a ptxas is on
  ``$CUDA_PATH/bin`` or ``$PATH`` it also assembles that PTX and must report no serialized ``wgmma.mma_async`` (skipped
  when that ptxas is older than the DSL's PTX ISA).

These run on every FROST lane (no SM90 FROST lane exists), which is the only CI coverage the SM90 helpers get.
"""

import itertools
import os
import re
import shutil
import subprocess
import sys
import textwrap

import pytest

from frost_test_utils import arch_known_to_the_dsl, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]


def _develop_rs(n, kind, accumulate_bit, transpose_bit):
    """Develop's RS m64n{n}k16 text and constraints, built inline in _wgmma_rs before the builder existed (#2426's
    native-width RS form, any C slot count)."""
    count = n // 2
    registers = "{" + ",".join(f"${i}" for i in range(count)) + "}, {" + ",".join(f"${2 * count + i}" for i in range(4)) + f"}}, ${2 * count + 4}"
    asm = f"wgmma.mma_async.sync.aligned.m64n{2 * count}k16.f32.{kind}.{kind} {registers}, {accumulate_bit}, 1, 1, {transpose_bit};"
    constraints = ",".join(["=f"] * count + [str(i) for i in range(count)] + ["r"] * 4 + ["l"])
    return asm, constraints


@pytest.mark.parametrize("n", (64, 128, 256))
@pytest.mark.parametrize("kind, accumulate_bit, transpose_bit", list(itertools.product(("f16", "bf16"), (0, 1), (0, 1))))
def test_rs_asm_is_develops_literal(n, kind, accumulate_bit, transpose_bit):
    from cudnn.frost.tile_dsl.wgmma import _rs_asm

    assert _rs_asm(n, kind, accumulate_bit, transpose_bit) == _develop_rs(n, kind, accumulate_bit, transpose_bit)


@pytest.mark.parametrize("n", (64, 128, 256))
def test_rs_asm_numbers_outputs_tied_inputs_a_and_b(n):
    from cudnn.frost.tile_dsl.wgmma import _rs_asm

    asm, constraints = _rs_asm(n, "bf16", 1, 1)
    slots = n // 2
    m = re.fullmatch(rf"wgmma\.mma_async\.sync\.aligned\.m64n{n}k16\.f32\.bf16\.bf16 \{{([^}}]*)\}}, \{{([^}}]*)\}}, (\$\d+), 1, 1, 1, 1;", asm)
    assert m, asm[:100]
    outputs, a_words, b = m.groups()
    assert outputs.split(",") == [f"${i}" for i in range(slots)]
    assert a_words.split(",") == [f"${n + i}" for i in range(4)]
    assert b == f"${n + 4}"
    assert constraints.split(",") == ["=f"] * slots + [str(i) for i in range(slots)] + ["r"] * 4 + ["l"]
    used = {int(x) for x in re.findall(r"\$(\d+)", asm)}
    assert max(used) == 2 * slots + 4 and not used & set(range(slots, n)), sorted(used)[-6:]


@pytest.mark.parametrize("n", (0, 32, 192, 512))
def test_rs_asm_refuses_other_n(n):
    from cudnn.frost.tile_dsl.wgmma import _rs_asm

    with pytest.raises(ValueError, match="n64, n128 or n256"):
        _rs_asm(n, "f16", 1, 1)


_PROBE_SRC = textwrap.dedent('''
    """Probe: one warpgroup issues one SM90 WGMMA form (or the D128 helper step) over SMEM it never fills; C goes to GMEM."""
    from types import SimpleNamespace

    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.experimental import primitives as prims

    from cudnn.frost.tile_dsl.tma import st_global
    from cudnn.frost.tile_dsl.wgmma import wgmma_m64nNk16_f32
    from cudnn.sdpa.fwd.kernels.sm90._common_hopper import mma_pv, mma_qk, wgmma_descriptor

    # form -> (C slots, register A): SS n64 / n128 / n256, RS n128 / n256.
    FORMS = {0: (32, False), 1: (64, False), 2: (128, False), 3: (64, True), 4: (128, True)}
    D128_STEP = 5


    @cute.kernel
    def probe(mOut: cute.Tensor, form: cutlass.Constexpr[int]):
        tid = cutlass.Int32(cute.arch.thread_idx()[0])
        sA = cutlass.Array(cutlass.BFloat16, 128 * 128, alignment=1024, space=cutlass.AddressSpace.smem)
        sB = cutlass.Array(cutlass.BFloat16, 128 * 128, alignment=1024, space=cutlass.AddressSpace.smem)
        a_desc, b_desc = wgmma_descriptor(sA), wgmma_descriptor(sB)
        if cutlass.const_expr(form == D128_STEP):
            output = cutlass.Array(cutlass.Float32, 64, alignment=16)
            output.store((0.0,) * 64)
            scores = cutlass.Array(cutlass.Float32, 64, alignment=16)
            v_desc = wgmma_descriptor(sB, mn_major=True, slab_rows=128)
            params = SimpleNamespace(q_desc=a_desc, k_desc=b_desc, v_desc=v_desc, o_regs=output)
            mma_qk(params, scores, ab_dtype=cutlass.BFloat16, c_slots=64, k_steps=8, q_slab_rows=64, k_slab_rows=128, wg_wait=-1)
            prims.wgmma_wait_group_sync_aligned(0)
            # P in new registers, as after the softmax: QK's raw C registers fed straight in as the RS A operand
            # make ptxas 13.4 serialize the PV group (C7511), a pattern no attention kernel has.
            scale = cutlass.Float32(tid) * cutlass.Float32(0.125)
            words = cutlass.Vector.from_elements([scores[2 * w] * scale for w in range(32)], cutlass.Float32).bitcast(cutlass.Uint32)
            mma_pv(params, words, True, ab_dtype=cutlass.BFloat16, k_steps=8)
            c = output.load(0, 64)
        else:
            slots, register_a = FORMS[form]
            c = cutlass.Vector.from_elements((cutlass.Float32(0.0),) * slots, cutlass.Float32)
            # RS mirrors D512's PV (register P, MN-major V); SS mirrors its QK (two K-major descriptors).
            a = cutlass.Vector.from_elements((cutlass.Uint32(0),) * 4, cutlass.Uint32) if cutlass.const_expr(register_a) else a_desc
            b = wgmma_descriptor(sB, mn_major=True, slab_rows=128) if cutlass.const_expr(register_a) else b_desc
            prims.wgmma_fence_aligned()
            c = wgmma_m64nNk16_f32(c, a, b, cutlass.BFloat16, transpose_b=register_a, accumulate=False)
            prims.wgmma_commit_group_sync_aligned()
            prims.wgmma_wait_group_sync_aligned(0)
        for i in cutlass.range_constexpr(c.shape[0]):
            st_global(mOut.iterator.toint() + (tid * c.shape[0] + i).to(cutlass.Int64) * cutlass.Int64(4), c[i], cutlass.Float32)


    probe.set_name_prefix("cudnn", remove_cutlass_symbol=True)


    @cute.jit
    def launch(out: cute.Tensor, form: cutlass.Constexpr[int], stream: cuda.CUstream):
        probe(out, form).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)


    if __name__ == "__main__":
        # argv: <arch> <form> <dump dir>; the caller sets CUTE_DSL_DUMP_DIR / CUTE_DSL_KEEP / CUTE_DSL_ARCH.
        import glob, os, re, sys
        from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

        arch, form, dump = sys.argv[1], int(sys.argv[2]), sys.argv[3]
        out = make_fake_tensor(dtype=cutlass.Float32, shape=(cute.sym_int(),), stride=(1,), assumed_align=16)
        cute.compile(launch, out, form, make_fake_stream(use_tvm_ffi_env_stream=False), options=f"--enable-tvm-ffi --gpu-arch {arch} --keep-ptx")
        print("COMPILED", arch, form)
        ptxs = glob.glob(os.path.join(dump, "**", f"*.{arch}.ptx"), recursive=True)
        if len(ptxs) != 1:
            print("FAIL expected one ptx in", dump, ptxs); sys.exit(3)
        ptx = open(ptxs[0]).read()
        print("PTX_PATH", ptxs[0])
        for n in (64, 128, 256):
            head = f"wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16"
            print("PTX", f"N{n}", ptx.count(head))
            # RS: the A operand is a brace list right after the C list; SS: a descriptor register.
            print("PTX", f"RS{n}", len(re.findall(re.escape(head) + r"\\s*\\{[^}]*\\},\\s*\\{", ptx)))
        print("PTX COMMIT", ptx.count("wgmma.commit_group.sync.aligned"))
        print("PTX WAIT0", len(re.findall(r"wgmma\\.wait_group\\.sync\\.aligned\\s+0;", ptx)))
        print("PTX WAIT_ANY", ptx.count("wgmma.wait_group.sync.aligned"))
''')

# form -> {PTX key: count}; keys not listed must be 0 among N*/RS*.
_EXPECT = {
    0: {"N64": 1},
    1: {"N128": 1},
    2: {"N256": 1},
    3: {"N128": 1, "RS128": 1},
    4: {"N256": 1, "RS256": 1},
    5: {"N128": 16, "RS128": 8},
}


def _ptxas():
    cands = [os.path.join(os.environ["CUDA_PATH"], "bin", "ptxas")] if os.environ.get("CUDA_PATH") else []
    cands.append(shutil.which("ptxas") or "")
    return next((c for c in cands if c and os.path.isfile(c) and os.access(c, os.X_OK)), None)


@pytest.mark.parametrize("form", sorted(_EXPECT), ids=["ss_n64", "ss_n128", "ss_n256", "rs_n128", "rs_n256", "d128_qk_issue_only_then_rs_pv"])
def test_sm90a_trace_compile_lowers_each_form(tmp_path, form):
    if not arch_known_to_the_dsl("sm_90a"):
        pytest.skip("this cutlass-dsl has no sm_90a")
    dump = tmp_path / f"dump_{form}"
    dump.mkdir()
    probe_py = tmp_path / "wgmma_probe.py"
    probe_py.write_text(_PROBE_SRC)
    env = dict(os.environ, CUTE_DSL_DUMP_DIR=str(dump), CUTE_DSL_KEEP="ptx", CUTE_DSL_ARCH="sm_90a", CUDNN_FRONTEND_DISABLE_COMPILED_CACHE="1")
    proc = subprocess.run([sys.executable, str(probe_py), "sm_90a", str(form), str(dump)], capture_output=True, text=True, timeout=900, env=env)
    assert proc.returncode == 0, f"sm_90a trace-compile of form {form} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    lines = proc.stdout.splitlines()
    ptx = {ln.split()[1]: int(ln.split()[2]) for ln in lines if ln.startswith("PTX ")}
    want = {key: 0 for key in ("N64", "N128", "N256", "RS64", "RS128", "RS256")} | _EXPECT[form]
    assert {key: ptx[key] for key in want} == want, ptx
    if form == 5:
        assert ptx["COMMIT"] == 2 and ptx["WAIT0"] == 2 and ptx["WAIT_ANY"] == 2, f"issue-only mma_qk must add no wait of its own: {ptx}"
        ptxas = _ptxas()
        if ptxas is None:
            pytest.skip("PTX pinned; no ptxas on CUDA_PATH or PATH to check wgmma serialization")
        ptx_path = next(ln.split(maxsplit=1)[1] for ln in lines if ln.startswith("PTX_PATH "))
        asm = subprocess.run([ptxas, "-arch=sm_90a", "-v", "-o", str(tmp_path / "probe.cubin"), ptx_path], capture_output=True, text=True, timeout=600)
        if asm.returncode != 0 and "Unsupported .version" in asm.stderr:
            # A toolkit ptxas older than the DSL's PTX ISA gives no verdict either way (the nvdisasm pins skip the same way).
            pytest.skip(f"PTX pinned; {ptxas} cannot assemble this DSL's PTX ISA: {asm.stderr.strip().splitlines()[-1]}")
        assert asm.returncode == 0, asm.stderr[-3000:]
        assert "wgmma.mma_async instructions are serialized" not in asm.stdout + asm.stderr, (asm.stdout + asm.stderr)[-3000:]
