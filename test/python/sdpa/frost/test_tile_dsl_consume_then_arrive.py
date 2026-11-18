# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``cudnn.frost.tile_dsl.consume`` -- the mbarrier release that depends on the values loaded from the stage it frees
(``python/cudnn/AGENTS.md`` Rule 11), on the self-test ring of ``warm_stress_ring.py``.

Host tier (any box whose CuTe DSL knows the target; nothing is launched).  One child process renders the ring for every
target the DSL knows and traces the misuse cases:

* every helper form: the generated PTX has ONE arrive whose address is computed, through an ``and`` with the kernel's
  release-zero parameter, from all twenty loaded shared-memory registers of the release (``warm_stress.assert_release_dependence``,
  a definition-use walk); with a decoding disassembler, every loaded register is read in front of the arrive in the SASS;
* the three controls have no such site: the unsafe twin (a plain arrive), the helper's chain with an inline-asm zero as the
  mask, and the chain under an entry guard on the kernel argument.  The last two are the regression tests of the two
  sentences of the contract that forbid them; they can only pass or skip (a toolchain that stops folding them);
* the trace-time refusals of the contract, and the accepted call shapes (inside dynamic regions, through a ``@cute.jit``
  function, mixed value types).

Device tier (an SM100-line GPU).  The positive control first: the unsafe twin must be SEEN to fail here
(``warm_stress.stress_validation``).  Then each helper form, with its uses pushed behind the release, must read zero wrong
bytes in 3,000 back-to-back launches and 3,000 launches behind a shared-memory poison launch.  Where the twin does not
reproduce, the cells are skipped with ``warm stress not validated on <part>`` -- a coverage gap, never a silent pass.
A zero is a rate bound for this ring; the proof of the order is the rendering pinned by the host tier.
"""

import os
import subprocess
import sys
import textwrap

import pytest

import warm_stress
from frost_test_utils import nvdisasm_candidates, process_watchdog, requires_blackwell, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]

ARCHS = ("sm_100a", "sm_103a", "sm_107a", "sm_120a")
HELPER_FORMS = ("elected", "all", "counted", "counted_all", "peer", "peer_all", "peer_relaxed", "sched1", "sched2")
CONTROL_FORMS = ("plain", "folded_zero", "guarded")
DEVICE_FORMS = ("elected", "all", "counted", "peer", "sched2")
WORDS_PER_RELEASE = 20
# form -> (text the arrive must contain, predicated, counted)
ARRIVE_SPELLING = {
    "elected": ("mbarrier.arrive.shared::cta.b64", True, False),
    "all": ("mbarrier.arrive", False, False),
    "counted": ("mbarrier.arrive.shared::cta.b64", True, True),
    "counted_all": ("mbarrier.arrive", False, True),
    "peer": ("mbarrier.arrive.release.cta.shared::cluster.b64", True, False),
    "peer_all": ("mbarrier.arrive.release.cta.shared::cluster.b64", False, False),
    "peer_relaxed": ("mbarrier.arrive.relaxed.cluster.shared::cluster.b64", True, False),
    "sched1": ("mbarrier.arrive.shared::cta.b64", True, False),
    "sched2": ("mbarrier.arrive.release.cta.shared::cluster.b64", True, False),
}
# misuse case -> a fragment of the TypeError the helper raises at trace time
REFUSALS = {
    "none": "holds None",
    "empty": "empty sequence",
    "python_number": "Python number",
    "constant": "compile-time constant",
    "boolean": "Boolean",
    "duplicate": "same value is passed twice",
    "vector_and_element": "a vector and one of its elements",
    "element_twice": "same vector element is passed twice",
    "zero_as_consumed": "not a consumed value",
    "literal_zero": "must be a KERNEL ARGUMENT",
    "computed_zero": "release-zero parameter, unmodified",
    "raw_zero": "must be the ReleaseZero",
    "guarded_zero": "has a use that is not this module's mask",
    "zero_used_elsewhere": "has a use that is not this module's mask",
    "python_pred": "lane predicate",
    "bad_count": "positive arrival count",
}
# accepted call shape -> loaded shared-memory registers its one arrive must be computed from
ACCEPTED = {"dynamic_regions": 4, "across_jit": 4, "mixed_types": 8}

_PROBE_SRC = textwrap.dedent('''
    """Child of test_tile_dsl_consume_then_arrive.py: argv = <dump dir> <test/python dir> <arch,...>.  Renders the ring forms for
    every listed target, then traces the misuse and accepted call shapes of the helper on a small site kernel (sm_100a)."""
    import glob, os, re, shutil, sys

    dump, test_python, archs = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    sys.path.insert(0, test_python)
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
    from cutlass.experimental import primitives as nvvm

    import warm_stress_ring as ring
    from cudnn.frost.tile_dsl.consume import consume_then_arrive, release_zero
    from cudnn.frost.tile_dsl.regtile import RegTile

    def keep(tag):
        """Move what the last compile dumped to <tag>.ptx / <tag>.cubin."""
        for ext in ("ptx", "cubin"):
            for path in glob.glob(os.path.join(dump, "cutlass_*." + ext)):
                shutil.move(path, os.path.join(dump, tag + "." + ext))

    def known(arch):
        try:
            from cutlass.base_dsl.enums import Arch
            Arch.from_string(arch)
            return True
        except Exception:
            return False

    x = make_fake_tensor(cutlass.Int32, (cute.sym_int(),), (1,), assumed_align=16)
    out = make_fake_tensor(cutlass.Int32, (cute.sym_int(),), (1,), assumed_align=16)
    stream = make_fake_stream(use_tvm_ffi_env_stream=False)
    for arch in archs:
        if not known(arch):
            print("UNKNOWN", arch, flush=True)
            continue
        for form in list(ring.CONTROL_FORMS) + list(ring.HELPER_FORMS):
            try:
                cute.compile(ring.warm_stress_ring_launch, x, out, cutlass.Int32(0), cutlass.Int32(0), form, True, ring.cluster_of(form), stream,
                             options=f"--enable-tvm-ffi --gpu-arch {arch} --keep-cubin --keep-ptx")
            except Exception as exc:
                # a target the DSL lists but its compiler flavour does not build (sm_107a with the CUDA 12 flavour)
                if "Unsupported GPU architecture" not in str(exc):
                    raise
                print("UNKNOWN", arch, flush=True)
                print("WHY", arch, "::", " ".join(re.sub(re.escape(chr(27) + "[") + "[0-9;]*m", "", str(exc)).split())[:300], flush=True)
                break
            keep(f"{arch}__{form}")
            print("COMPILED", arch, form, flush=True)


    @cute.jit
    def release_through_a_jit_function(mb, vec, rz):
        consume_then_arrive(mb, vec, rz)


    @cute.kernel
    def consume_site_kernel(mOut: cute.Tensor, steps: cutlass.Int32, rz_arg: cutlass.Int32, CASE: cutlass.Constexpr[str]):
        tid, _, _ = cute.arch.thread_idx()
        words = cutlass.Array(cutlass.Int32, 128, alignment=16, space=cutlass.AddressSpace.smem)
        wide = cutlass.Array(cutlass.Int64, 32, alignment=16, space=cutlass.AddressSpace.smem)
        reals = cutlass.Array(cutlass.Float32, 128, alignment=16, space=cutlass.AddressSpace.smem)
        halves = cutlass.Array(cutlass.Float16, 32, alignment=16, space=cutlass.AddressSpace.smem)
        small = cutlass.Array(cutlass.Uint8, 32, alignment=16, space=cutlass.AddressSpace.smem)
        mb = cutlass.Array(cutlass.Int64, 2, alignment=8, space=cutlass.AddressSpace.smem)
        if tid == 0:
            nvvm.mbarrier_init(mb, 32)
        nvvm.fence_mbarrier_init()
        nvvm.barrier_cta_sync()
        vec = words.load(tid * 4, vector_size=4, alignment=16)
        acc = vec[0] + vec[1] + vec[2] + vec[3]
        if cutlass.const_expr(CASE == "none"):
            consume_then_arrive(mb, None, release_zero(rz_arg))
        elif cutlass.const_expr(CASE == "empty"):
            consume_then_arrive(mb, (), release_zero(rz_arg))
        elif cutlass.const_expr(CASE == "python_number"):
            consume_then_arrive(mb, (vec, 3), release_zero(rz_arg))
        elif cutlass.const_expr(CASE == "constant"):
            consume_then_arrive(mb, (vec, cutlass.Int32(7)), release_zero(rz_arg))
        elif cutlass.const_expr(CASE == "boolean"):
            consume_then_arrive(mb, (vec, vec[0] == cutlass.Int32(0)), release_zero(rz_arg))
        elif cutlass.const_expr(CASE == "duplicate"):
            consume_then_arrive(mb, (vec, vec), release_zero(rz_arg))
        elif cutlass.const_expr(CASE == "vector_and_element"):
            consume_then_arrive(mb, (vec, vec[1]), release_zero(rz_arg))
        elif cutlass.const_expr(CASE == "element_twice"):
            consume_then_arrive(mb, (vec[0], vec[0]), release_zero(rz_arg))
        elif cutlass.const_expr(CASE == "zero_as_consumed"):
            consume_then_arrive(mb, (vec, release_zero(rz_arg)), release_zero(rz_arg))
        elif cutlass.const_expr(CASE == "literal_zero"):
            consume_then_arrive(mb, vec, release_zero(cutlass.Int32(0)))
        elif cutlass.const_expr(CASE == "computed_zero"):
            consume_then_arrive(mb, vec, release_zero(rz_arg & cutlass.Int32(1)))
        elif cutlass.const_expr(CASE == "raw_zero"):
            consume_then_arrive(mb, vec, rz_arg)
        elif cutlass.const_expr(CASE == "guarded_zero"):
            if rz_arg == cutlass.Int32(0):
                consume_then_arrive(mb, vec, release_zero(rz_arg))
        elif cutlass.const_expr(CASE == "zero_used_elsewhere"):
            acc = acc + rz_arg
            consume_then_arrive(mb, vec, release_zero(rz_arg))
        elif cutlass.const_expr(CASE == "python_pred"):
            consume_then_arrive(mb, vec, release_zero(rz_arg), pred=True)
        elif cutlass.const_expr(CASE == "bad_count"):
            consume_then_arrive(mb, vec, release_zero(rz_arg), pred=nvvm.elect_sync(), count=0)
        elif cutlass.const_expr(CASE == "dynamic_regions"):
            rz = release_zero(rz_arg)
            i = cutlass.Int32(0)
            while i < steps:
                if tid < 16:
                    again = words.load(tid * 4, vector_size=4, alignment=16)
                    consume_then_arrive(mb, again, rz)
                    acc = acc + again[0] + again[1] + again[2] + again[3]
                i = i + 1
        elif cutlass.const_expr(CASE == "across_jit"):
            release_through_a_jit_function(mb, vec, release_zero(rz_arg))
        else:  # mixed_types: a 64-bit word, a float vector as a register tile, a half and a byte beside an Int32 scalar
            w64 = wide.subview(tid).load()
            tile = RegTile(reals.load(tid * 4, vector_size=4, alignment=16))
            half = halves.subview(tid).load()
            byte = small.subview(tid).load()
            scalar = words.subview(tid).load()
            consume_then_arrive(mb, (w64, [tile, half], (byte, scalar)), release_zero(rz_arg))
            acc = acc + cutlass.Int32(scalar)
        mOut[tid] = acc


    consume_site_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


    @cute.jit
    def consume_site_launch(out: cute.Tensor, CASE: cutlass.Constexpr[str], stream: cuda.CUstream):
        consume_site_kernel(out, cutlass.Int32(2), cutlass.Int32(0), CASE).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


    refused, accepted = sys.argv[4].split(","), sys.argv[5].split(",")
    for case in refused + accepted:
        try:
            cute.compile(consume_site_launch, out, case, stream, options="--enable-tvm-ffi --gpu-arch sm_100a --keep-cubin --keep-ptx")
            keep(f"site__{case}")
            print("TRACED", case, flush=True)
        except Exception as exc:
            text = " ".join(str(exc).split())
            print("REFUSED", case, type(exc).__name__, "::", text[:600], flush=True)
    print("PROBE-DONE", flush=True)
    ''')


class _Renders:
    def __init__(self, dump, lines, stderr):
        self.dump, self.lines, self.stderr = dump, lines, stderr

    def has(self, arch, form):
        return f"COMPILED {arch} {form}" in self.lines

    def ptx(self, tag):
        with open(os.path.join(self.dump, tag + ".ptx")) as f:
            return f.read()

    def cubin(self, tag):
        return os.path.join(self.dump, tag + ".cubin")

    def require(self, arch, form):
        if f"UNKNOWN {arch}" in self.lines:
            why = next((ln for ln in self.lines if ln.startswith(f"WHY {arch} ")), "")
            pytest.skip(f"this CuTe DSL has no {arch} target" + (f" ({why})" if why else ""))
        assert self.has(arch, form), f"the {form} ring did not compile for {arch}:\n" + "\n".join(self.lines[-20:]) + "\n" + self.stderr[-3000:]


@pytest.fixture(scope="module")
def renders(tmp_path_factory):
    """ONE child process: the ring forms for every target the DSL knows, then the misuse / accepted site cases (host only, nothing launched)."""
    root = tmp_path_factory.mktemp("consume_renders")
    dump = root / "dump"
    dump.mkdir()
    probe = root / "consume_probe.py"
    probe.write_text(_PROBE_SRC)
    test_python = os.path.dirname(os.path.abspath(warm_stress.__file__))
    env = dict(os.environ, CUTE_DSL_DUMP_DIR=str(dump), CUTE_DSL_KEEP="ptx,cubin")
    argv = [sys.executable, str(probe), str(dump), test_python, ",".join(ARCHS), ",".join(REFUSALS), ",".join(ACCEPTED)]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=1200, env=env)
    lines = proc.stdout.splitlines()
    assert proc.returncode == 0 and "PROBE-DONE" in lines, f"the render child failed (rc {proc.returncode}):\n{proc.stdout[-3000:]}\n{proc.stderr[-4000:]}"
    return _Renders(str(dump), lines, proc.stderr)


def _disassemble(cubin):
    """The SASS lines of ``cubin``, or ``None`` when no disassembler on this box decodes it."""
    for nvdisasm in nvdisasm_candidates():
        try:
            proc = subprocess.run([nvdisasm, "-c", cubin], capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.splitlines()
    return None


def _unread_loaded_registers(sass):
    """``(loaded registers, those not read in front of the release)`` of the ring's consumer block, read in layout order: from the
    first 16-byte stage load to the first arrive behind the last one.  The consumer step is straight-line code, so layout order
    is execution order there."""
    import re

    body = [ln.split("*/", 1)[1].strip() for ln in sass if "/*" in ln and "*/" in ln and re.search(r"/\*[0-9a-f]{4,}\*/", ln)]
    loads = [i for i, ln in enumerate(body) if re.search(r"\bLDS\.128\b", ln)]
    assert len(loads) == 5, f"expected the ring's five 16-byte stage loads, found {len(loads)}"
    release = next(i for i in range(loads[-1], len(body)) if "SYNCS.ARRIVE" in body[i])
    loaded = set()
    for i in loads:
        first = int(re.search(r"LDS\.128\s+R(\d+)", body[i]).group(1))
        loaded.update(range(first, first + 4))
    read = set()
    for ln in body[loads[0] : release]:
        operands = ln.split(",", 1)[1] if "," in ln else ""
        read.update(int(r) for r in re.findall(r"\bR(\d+)\b", operands))
    return loaded, loaded - read


# ---------------------------------------------------------------------------
# Host tier: the rendering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form", HELPER_FORMS)
@pytest.mark.parametrize("arch", ARCHS)
def test_the_arrive_of_every_form_is_computed_from_every_loaded_register(renders, arch, form):
    """ONE arrive, and its address reaches all twenty loaded registers through an ``and`` with the release-zero parameter (the kernel's
    fourth); the arrive itself is the form's plain twin (spelling, predicate, count as a register)."""
    renders.require(arch, form)
    sites = warm_stress.assert_release_dependence(renders.ptx(f"{arch}__{form}"), sites=1, words=WORDS_PER_RELEASE)
    assert len(sites) == 1, f"the ring has one release site, the PTX has {len(sites)} arrives"
    site = sites[0]
    assert site.parameter.endswith("_param_3"), f"the mask is not the release-zero (the kernel's fourth parameter): {site.parameter}"
    assert site.loads == 5 and site.loaded_registers == WORDS_PER_RELEASE, (site.loads, site.loaded_registers)
    spelling, predicated, counted = ARRIVE_SPELLING[form]
    assert spelling in site.text, site.text
    assert site.text.startswith("@%p") == predicated, f"{form}: predicate -- {site.text}"
    count_operand = site.text.split("]", 1)[1].strip(" ;,")
    assert bool(count_operand) == counted, f"{form}: a count operand is expected exactly on the counted forms -- {site.text}"
    if counted and predicated:
        assert count_operand.startswith("%r"), f"{form}: the count of a predicated arrive must be a register, never an immediate -- {site.text}"


@pytest.mark.parametrize("form", CONTROL_FORMS)
@pytest.mark.parametrize("arch", ARCHS)
def test_the_controls_have_no_dependent_arrive(renders, arch, form):
    """The unsafe twin releases with a plain arrive.  The two lost-dependence controls show what the contract forbids: with an
    inline-asm zero the mask is not a kernel parameter; under an entry guard on the argument the optimizer replaces the mask by a
    constant and no ``and`` with the parameter is left."""
    renders.require(arch, form)
    sites = warm_stress.release_dependence(renders.ptx(f"{arch}__{form}"))
    assert len(sites) == 1, sites
    if sites[0].dependent and form == "guarded":
        pytest.skip("this toolchain no longer folds a release-zero that is compared in front of the release: revisit the sentence in Rule 11")
    assert not sites[0].dependent, f"{form}: {sites[0]}"


@pytest.mark.parametrize("form", HELPER_FORMS + CONTROL_FORMS)
@pytest.mark.parametrize("arch", ARCHS)
def test_disassembly_reads_the_loaded_registers_in_front_of_the_release(renders, arch, form):
    """With a decoding disassembler: every register of the five stage loads is read in front of the arrive for a helper form; the
    twin and both lost-dependence controls leave loaded registers unread there (the assembler folds the inline-asm zero)."""
    renders.require(arch, form)
    sass = _disassemble(renders.cubin(f"{arch}__{form}"))
    if sass is None:
        pytest.skip(f"no disassembler on this box decodes {arch}; the PTX pins of this module ran")
    loaded, unread = _unread_loaded_registers(sass)
    assert len(loaded) == WORDS_PER_RELEASE
    if form in HELPER_FORMS:
        assert not unread, f"{arch} {form}: registers {sorted(unread)} of the stage loads are not read in front of the release"
    elif form == "plain":
        assert unread, f"{arch}: the unsafe twin reads every loaded register in front of its release -- it is no longer a positive control"
    elif not unread:
        pytest.skip(f"{arch} {form}: this toolchain keeps the dependence of the forbidden spelling: revisit the sentence in Rule 11")


@pytest.mark.parametrize("case", list(REFUSALS))
def test_misuse_is_refused_at_trace_time(renders, case):
    """Each way to pass nothing, a constant, a derived duplicate or a visible zero raises with a message that names it."""
    line = next((ln for ln in renders.lines if ln.startswith(f"REFUSED {case} ") or ln == f"TRACED {case}"), None)
    assert line is not None, renders.lines[-30:]
    assert line.startswith("REFUSED"), f"the {case} misuse traced without an error"
    assert REFUSALS[case] in line, line


@pytest.mark.parametrize("case", list(ACCEPTED))
def test_accepted_call_shapes_keep_the_dependence(renders, case):
    """Inside a dynamic loop and branch, through a ``@cute.jit`` function, and with 64-bit, float, half and byte values: every
    arrive is computed from every loaded register of the call."""
    assert f"TRACED {case}" in renders.lines, [ln for ln in renders.lines if case in ln]
    sites = warm_stress.release_dependence(renders.ptx(f"site__{case}"))
    assert sites, "the site kernel lost its arrive"
    for site in sites:  # the optimizer may unroll the loop of `dynamic_regions`: every copy of the arrive keeps the dependence
        assert site.dependent and site.loaded_registers >= ACCEPTED[case] and site.parameter.endswith("_param_2"), site


# ---------------------------------------------------------------------------
# Device tier: the ring under the warm stress, behind its positive control
# ---------------------------------------------------------------------------

# Each cell builds what it launches IN FRONT of its process watchdog (the first CuTe DSL import of the process, the control
# ring, the poison kernel, the ring under test): the watchdog times launches only, so a slow import on a cold file system
# cannot read as a hung kernel.
device_tier = [requires_blackwell, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


def _build_ring(form):
    """The ring under test, compiled in front of the cell's watchdog.  A DSL whose compiler flavour has no target for this
    GPU skips the cell (a coverage gap), as the control does through ``stress_validation``."""
    import warm_stress_ring

    try:
        return warm_stress_ring.Ring(form)
    except Exception as exc:
        if "Unsupported GPU architecture" not in str(exc):
            raise
        pytest.skip(f"this CuTe DSL does not build the ring for this GPU: {' '.join(str(exc).split())[:300]}")


def _ring_case(ring):
    import warm_stress_ring

    return warm_stress.StressCase(
        name=f"consume ring {ring.release} (cluster {ring.cluster})",
        launch=ring.launch,
        outputs={"sums": ring.out},
        thread_blocks=ring.blocks,
        refills_per_launch=ring.refills_per_launch,
        reference={"sums": ring.reference},
        locate=warm_stress_ring.locate,
    )


@pytest.mark.parametrize("cluster", [pytest.param(1, marks=device_tier), pytest.param(2, marks=device_tier)])
def test_unsafe_twin_is_seen(cluster):
    """The positive control: the ring with a plain arrive fails on this GPU, alone and in clusters of two.  A part where it does not
    is reported as not validated and skipped -- the helper cells below then skip as well."""
    warm_stress.build_launchers(cluster=cluster)
    with process_watchdog(300, f"the unsafe twin of the test ring (cluster {cluster})"):
        validation = warm_stress.stress_validation(cluster=cluster)
    print("\n" + validation.text)
    if not validation.reproduced:
        pytest.skip(validation.text)
    assert validation.failures > 0 and validation.launches == warm_stress.CONTROL_LAUNCHES


@pytest.mark.parametrize("form", [pytest.param(form, marks=device_tier) for form in DEVICE_FORMS])
def test_helper_ring_reads_zero(form):
    """3,000 back-to-back + 3,000 poisoned launches of the ring released through the helper, uses pushed behind the release: no
    differing byte against the independent reference, in the process where the twin of the same launch shape failed."""
    ring = _build_ring(form)
    warm_stress.build_launchers(cluster=ring.cluster)
    with process_watchdog(420, f"the consume ring ({form})"):
        validation = warm_stress.stress_validation(cluster=ring.cluster)
        report = warm_stress.assert_warm_stress(_ring_case(ring), validation=validation)
    print("\n" + report.text())
    assert report.failures == 0 and report.launches == warm_stress.DEFAULT_WARM + warm_stress.DEFAULT_POISONED


@pytest.mark.parametrize("form", [pytest.param(form, marks=device_tier) for form in ("folded_zero", "guarded")])
def test_the_stress_sees_a_lost_dependence(form):
    """Where the twin fails, the ring whose mask the toolchain can see through fails as well: the stress has power against a
    dependence that was lost without any error.  It passes or skips, never fails."""
    ring = _build_ring(form)
    warm_stress.build_launchers(cluster=1)
    with process_watchdog(300, f"the lost-dependence control ({form})"):
        validation = warm_stress.stress_validation(cluster=1)
        if not validation.reproduced:
            pytest.skip(validation.text)
        report = warm_stress.warm_stress(_ring_case(ring), warm=warm_stress.CONTROL_LAUNCHES, poisoned=0, flush_mib=0, stop_at_first=False)
    print("\n" + report.text())
    if report.failures == 0:
        pytest.skip(f"the {form} ring read 0 of {report.launches} where the twin failed: this toolchain keeps the dependence of the forbidden spelling")
    assert report.failures > 0
