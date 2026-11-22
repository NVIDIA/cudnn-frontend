# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``tile_dsl.handover`` -- the release add and the poll-then-acquire of a hand-over between two CTAs through global memory,
the reader's two anchors (``tile_dsl.consume.consumed_zero`` and ``tile_dsl.handover.acquired_zero``) and
``tile_dsl.barrier.timer_sleep`` (the delay of a stress probe).

Host-only pins of what the forms compile to (a trace-compile for ``sm_100a``; no device, nothing is launched).  One probe:
CTA 0's lanes store four floats each and call ``release_add``; CTA 1's lanes call ``poll_then_acquire`` for the lane count
and load the same floats from an address that takes ``consumed_zero`` of the acquired value.  The probe is rendered three
times: with the plain poll, with ``warp_uniform=True`` (whole warps polling as one: the loop is left on a warp vote), and
as the hand-over to a reader through the async proxy that takes no release-zero -- the writer calls
``release_add(word, async_reader=True)``, the reader polls as whole warps and its address takes
``acquired_zero(count, lanes)``.  All three must read the same way; only the last two carry the vote, only the last the
proxy fence.

PTX: exactly one ``atom.global.release.add`` (no ``acq_rel``, no scope wider or narrower than the default GPU scope), exactly
one ``ld.acquire.gpu`` and at least one ``ld.relaxed.gpu``; one ``fence.proxy.async.global`` with ``async_reader`` and none
without.

Disassembly (required: the pins FAIL when no ``nvdisasm`` is found or none decodes the arch -- :func:`_disassemblers` says
where one is looked for):

* the writer: EVERY 16-byte data store stands in front of the ONE ``MEMBAR.ALL.GPU``, which stands in front of the ONE
  reduction on the word -- the assembler folds a warp's adds into one reduction of the lane count; with ``async_reader``
  the proxy fence stands between the last data store and that barrier;
* the reader: at least one strong load (the poll) in front of the ONE strong load that carries ``CCTL.IVALL`` (the acquire),
  EVERY data load behind it, and the data loads' address register computed from the acquired value (the anchor), for
  ``consumed_zero`` and for ``acquired_zero`` alike.

The sleep is one ``nanosleep`` in a counted loop of the PTX (the assembler may unroll a short one).  The hand-over itself is
run by the kernel that uses the forms, under its own stress with a positive control.
"""

import glob
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import textwrap

import pytest

from frost_test_utils import arch_known_to_the_dsl, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]

ARCH = "sm_100a"
FORM_HANDOVER, FORM_SLEEP, FORM_HANDOVER_UNIFORM, FORM_HANDOVER_COUNT_ZERO = 0, 1, 2, 3
SLEEP_UNITS = 3


def _disassemblers() -> list:
    """Every ``nvdisasm`` the pins may try, in preference order: ``$NVDISASM``, ``$CUDA_PATH/bin`` and ``$CUDA_HOME/bin``,
    ``$PATH``, the ``nvidia-cuda-nvdisasm`` wheel of this interpreter (``nvidia/cu*/bin``) and the toolkit's default install
    prefix (``/usr/local/cuda*/bin``, newest first).  Only existing executables; the probe takes the first that decodes."""
    cands = [os.environ.get("NVDISASM", "")]
    cands += [os.path.join(os.environ[var], "bin", "nvdisasm") for var in ("CUDA_PATH", "CUDA_HOME") if os.environ.get(var)]
    cands.append(shutil.which("nvdisasm") or "")
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        spec = None
    for root in list(getattr(spec, "submodule_search_locations", None) or []):
        cands += sorted(glob.glob(os.path.join(root, "cu*", "bin", "nvdisasm")), reverse=True)
    cands += sorted(glob.glob("/usr/local/cuda*/bin/nvdisasm"), reverse=True)
    return [c for c in dict.fromkeys(cands) if c and os.path.isfile(c) and os.access(c, os.X_OK)]


def _require_sass(sass, skipped) -> None:
    """The disassembly pins are required coverage: without a listing they fail, they do not skip."""
    assert sass, (
        f"no nvdisasm decodes {ARCH}, so the disassembly pins did not run: candidates {_disassemblers()}; what each said: {skipped}.  "
        "Point NVDISASM or CUDA_PATH at a CUDA toolkit whose nvdisasm knows the arch"
    )


_PROBE_SRC = textwrap.dedent('''
    """Probe: 2 CTAs x 64 lanes.  CTA 0: every lane stores four floats to global memory, then release_add(word).  CTA 1:
    every lane polls the word for the lane count, acquires, and loads the four floats from an address that takes the
    acquired value through consumed_zero -- or, in the count-zero form, through acquired_zero (no release-zero is read by
    the reader, and the writer fences for a reader through the async proxy)."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from cudnn.frost.tile_dsl.barrier import timer_sleep
    from cudnn.frost.tile_dsl.consume import consumed_zero, release_zero
    from cudnn.frost.tile_dsl.handover import acquired_zero, poll_then_acquire, release_add

    LANES = 64
    FORM_HANDOVER, FORM_SLEEP, FORM_HANDOVER_UNIFORM, FORM_HANDOVER_COUNT_ZERO = 0, 1, 2, 3


    @cute.kernel
    def probe(mWord: cute.Tensor, mData: cute.Tensor, mOut: cute.Tensor, form: cutlass.Constexpr[int], units: cutlass.Constexpr[int], rz_arg: cutlass.Int32):
        rz = release_zero(rz_arg)
        tidx = cutlass.Int32(cute.arch.thread_idx()[0])
        bidx = cutlass.Int32(cute.arch.block_idx()[0])
        if cutlass.const_expr(form == FORM_SLEEP):
            timer_sleep(units)
            mOut[tidx] = cutlass.Float32(1.0)
        else:
            word = mWord.iterator
            if bidx == cutlass.Int32(0):
                dst = (mData.iterator + tidx * cutlass.Int32(4)).raw_ptr()
                dst.store(cutlass.Vector.from_elements(tuple(mOut[tidx + cutlass.Int32(g)] for g in range(4)), cutlass.Float32), alignment=16)
                release_add(word, 1, form == FORM_HANDOVER_COUNT_ZERO)
            else:
                count = poll_then_acquire(word, cutlass.Int32(LANES), form != FORM_HANDOVER)
                zero = acquired_zero(count, LANES) if cutlass.const_expr(form == FORM_HANDOVER_COUNT_ZERO) else consumed_zero(count, rz)
                src = (mData.iterator + (tidx * cutlass.Int32(4) + zero)).raw_ptr()
                chunk = src.load(count=4, alignment=16)
                mOut[tidx + cutlass.Int32(LANES)] = chunk[0] + chunk[1] + chunk[2] + chunk[3]


    probe.set_name_prefix("cudnn", remove_cutlass_symbol=True)


    @cute.jit
    def launch(word: cute.Tensor, data: cute.Tensor, out: cute.Tensor, form: cutlass.Constexpr[int], units: cutlass.Constexpr[int], stream: cuda.CUstream):
        probe(word, data, out, form, units, cutlass.Int32(0)).launch(grid=(2, 1, 1), block=(LANES, 1, 1), stream=stream)


    if __name__ == "__main__":
        # argv: <arch> <form> <units> <dump dir> [nvdisasm candidates...]; CUTE_DSL_DUMP_DIR is set by the caller (keeps ptx + cubin).
        import glob, os, subprocess, sys
        from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

        arch, form, units, dump, cands = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5:]
        word = make_fake_tensor(dtype=cutlass.Int32, shape=(cute.sym_int(),), stride=(1,), assumed_align=4)
        data = make_fake_tensor(dtype=cutlass.Float32, shape=(cute.sym_int(),), stride=(1,), assumed_align=16)
        out = make_fake_tensor(dtype=cutlass.Float32, shape=(cute.sym_int(),), stride=(1,), assumed_align=16)
        cute.compile(launch, word, data, out, form, units, make_fake_stream(use_tvm_ffi_env_stream=False), options=f"--enable-tvm-ffi --gpu-arch {arch} --keep-cubin --keep-ptx")
        print("COMPILED", arch, form)
        ptxs = glob.glob(os.path.join(dump, f"*.{arch}.ptx"))
        if not ptxs:
            print("FAIL no ptx landed in", dump, os.listdir(dump)); sys.exit(3)
        ptx = open(ptxs[0]).read()
        for key, pat in (("RELEASE_ADD", "atom.global.release.add"), ("ACQ_REL", "acq_rel"), ("ATOM", "atom."), ("LD_ACQUIRE_GPU", "ld.acquire.gpu.global"),
                         ("LD_RELAXED_GPU", "ld.relaxed.gpu.global"), ("LD_ACQUIRE", "ld.acquire"), ("NANOSLEEP", "nanosleep.u32 1000;"), ("VOTE_ALL", "vote.sync.all"),
                         ("FENCE_PROXY_ASYNC_GLOBAL", "fence.proxy.async.global")):
            print("PTX", key, ptx.count(pat))
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
        for line in sass:
            if "/*" in line and ";" in line:
                print("INSTR", line.split("*/", 1)[1].strip())
    ''')


def _run_probe(tmp_path, form: int, timeout: int = 900):
    probe_py = tmp_path / "handover_probe.py"
    probe_py.write_text(_PROBE_SRC)
    dump = tmp_path / f"dump_{form}"
    dump.mkdir()
    env = dict(os.environ, CUTE_DSL_DUMP_DIR=str(dump), CUTE_DSL_KEEP="ptx,cubin", CUDA_VISIBLE_DEVICES="")
    argv = [sys.executable, str(probe_py), ARCH, str(form), str(SLEEP_UNITS), str(dump), *_disassemblers()]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
    assert proc.returncode == 0, f"{ARCH} trace-compile of the probe (form {form}) failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    lines = proc.stdout.splitlines()
    assert any(ln.startswith(f"COMPILED {ARCH}") for ln in lines), proc.stdout[-2000:]
    ptx = {ln.split()[1]: int(ln.split()[2]) for ln in lines if ln.startswith("PTX ") and len(ln.split()) == 3}
    skipped = [ln for ln in lines if ln.startswith(("SKIP", "REJECT"))] if not any(ln.startswith("NVDISASM") for ln in lines) else []
    sass = [ln[len("INSTR ") :] for ln in lines if ln.startswith("INSTR ")]
    return ptx, sass, skipped


def _mnemonic(instr: str) -> str:
    return re.sub(r"^@!?U?P\w+\s+", "", instr).split()[0].rstrip(";")


@pytest.fixture(
    scope="module", params=[FORM_HANDOVER, FORM_HANDOVER_UNIFORM, FORM_HANDOVER_COUNT_ZERO], ids=["plain", "warp_uniform", "count_zero_async_reader"]
)
def handover(request, tmp_path_factory):
    if not arch_known_to_the_dsl(ARCH):
        pytest.skip(f"this cutlass-dsl has no {ARCH}")
    return (*_run_probe(tmp_path_factory.mktemp("handover"), request.param), request.param)


def test_release_add_and_poll_then_acquire_ptx(handover):
    """One release add at the default (GPU) scope whose result nobody reads; relaxed polls and exactly one acquire load, both
    at GPU scope; the proxy fence with ``async_reader`` only."""
    ptx, _, _, form = handover
    print(f"\nPTX {ptx}")
    assert (ptx["VOTE_ALL"] >= 1) == (form != FORM_HANDOVER), f"the warp-uniform poll, and only it, leaves its loop on a warp vote -- {ptx}"
    assert ptx["FENCE_PROXY_ASYNC_GLOBAL"] == (
        1 if form == FORM_HANDOVER_COUNT_ZERO else 0
    ), f"one proxy fence for a reader through the async proxy, none otherwise -- {ptx}"
    assert ptx["RELEASE_ADD"] == 1 and ptx["ATOM"] == 1 and ptx["ACQ_REL"] == 0, ptx
    assert ptx["LD_ACQUIRE_GPU"] == 1 and ptx["LD_ACQUIRE"] == 1, f"exactly one acquire load, at GPU scope -- {ptx}"
    assert ptx["LD_RELAXED_GPU"] >= 1, f"the poll is a relaxed load at GPU scope -- {ptx}"


def test_release_add_is_one_reduction_per_warp_behind_the_stores(handover):
    """The writer's stores, then (for a reader through the async proxy) the proxy fence, then ONE GPU-scope memory barrier,
    then ONE reduction on the word."""
    _, sass, skipped, form = handover
    _require_sass(sass, skipped)
    names = [_mnemonic(instr) for instr in sass]
    barriers = [i for i, name in enumerate(names) if name == "MEMBAR.ALL.GPU"]
    reductions = [i for i, name in enumerate(names) if name in ("REDG.E.ADD.STRONG.GPU", "ATOMG.E.ADD.STRONG.GPU")]
    data_stores = [i for i, name in enumerate(names) if name.startswith("STG.E.128")]  # the writer's four floats: its only 16-byte stores
    print(f"\nMEMBAR.ALL.GPU at {barriers}, reductions at {reductions} ({[names[i] for i in reductions]}), 16-byte data stores at {data_stores}")
    assert len(barriers) == 1 and len(reductions) == 1, "a warp's release adds fold into one barrier and one reduction"
    assert data_stores, "the writer's 16-byte data store is in the listing"
    assert (
        max(data_stores) < barriers[0] < reductions[0]
    ), "EVERY data store of the lane stands in front of the barrier of its release (none sunk behind the add)"
    fences = [i for i, name in enumerate(names) if name.startswith("FENCE.VIEW.ASYNC")]
    print(f"proxy fences at {fences} ({[names[i] for i in fences]})")
    if form == FORM_HANDOVER_COUNT_ZERO:
        assert len(fences) == 1 and max(data_stores) < fences[0] < barriers[0], "the proxy fence stands behind the last data store and in front of the release"
    else:
        assert not fences, "no proxy fence without async_reader"


def test_poll_then_acquire_is_strong_polls_and_one_invalidating_load_in_front_of_the_data_loads(handover):
    """Strong loads poll; exactly one strong load carries the first-level invalidation (the acquire), behind every poll; the
    data loads stand behind it."""
    _, sass, skipped, _ = handover
    _require_sass(sass, skipped)
    names = [_mnemonic(instr) for instr in sass]
    strong = [i for i, name in enumerate(names) if name.startswith("LDG.E") and "STRONG.GPU" in name and ".128" not in name]
    invalidations = [i for i, name in enumerate(names) if name == "CCTL.IVALL"]
    data_loads = [i for i, name in enumerate(names) if name.startswith("LDG.E.128")]
    print(f"\nstrong word loads at {strong}, CCTL.IVALL at {invalidations}, data loads at {data_loads} ({[names[i] for i in data_loads]})")
    assert len(invalidations) == 1, "one acquire load: one first-level invalidation, none inside the poll loop"
    acquire = max(i for i in strong if i < invalidations[0])
    assert invalidations[0] == acquire + 1, "the invalidation belongs to the last strong load"
    assert any(i < acquire for i in strong), "at least one poll in front of the acquire load"
    assert data_loads and min(data_loads) > invalidations[0], "the data loads stand behind the acquire load"
    # the anchor: the data loads' address register is (re)defined behind the acquire load, by a chain that starts at the acquired register
    acquired = re.search(r"LDG\.E\S*\s+(R\d+),", sass[acquire]).group(1)
    address = re.search(r"\[(R\d+)\.64", sass[min(data_loads)]).group(1)
    tainted, defined_behind = {acquired}, False
    for instr in sass[acquire + 1 : min(data_loads)]:
        operands = re.sub(r"^@!?U?P\w+\s+", "", instr).split(None, 1)
        regs = re.findall(r"\bR\d+\b", operands[1]) if len(operands) > 1 else []
        if regs and any(r in tainted for r in regs[1:]):
            tainted.add(regs[0])
        elif regs and regs[0] in tainted and regs[0] != acquired:
            tainted.discard(regs[0])
        defined_behind = defined_behind or (bool(regs) and regs[0] == address)
    print(f"acquired {acquired}; the data loads' address register {address}; registers computed from the acquired value {sorted(tainted)}")
    assert (
        defined_behind and address in tainted
    ), "the data loads' address takes a value from the acquire load (consumed_zero / acquired_zero): no load can be formed in front of it"


def test_timer_sleep_is_a_counted_loop_of_timer_sleeps(tmp_path):
    """``timer_sleep(n)``: one ``nanosleep.u32 1000`` in a loop that counts to ``n`` in a PTX register."""
    from cudnn.frost.tile_dsl.barrier import sleep_ptx

    text = sleep_ptx(SLEEP_UNITS)
    assert text.count("nanosleep.u32 1000;") == 1 and f"setp.lt.u32 P1, n, {SLEEP_UNITS};" in text and "@P1 bra LAB_SLEEP;" in text
    with pytest.raises(ValueError):
        sleep_ptx(0)
    if not arch_known_to_the_dsl(ARCH):
        pytest.skip(f"this cutlass-dsl has no {ARCH}")
    ptx, sass, skipped = _run_probe(tmp_path, FORM_SLEEP)
    assert ptx["NANOSLEEP"] == 1 and ptx["ATOM"] == 0 and ptx["LD_ACQUIRE"] == 0, ptx
    _require_sass(sass, skipped)
    sleeps = sum(1 for instr in sass if _mnemonic(instr).startswith("NANOSLEEP"))
    assert 1 <= sleeps <= SLEEP_UNITS, f"{sleeps} NANOSLEEP for a loop of {SLEEP_UNITS}"
