# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``tile_dsl.barrier.arrive_on_leader_release`` / ``Producer.LEADER_RELEASE`` -- the cross-CTA arrive that PUBLISHES
generic SMEM stores to the leader.

``arrive_on_leader`` is ``relaxed=True`` at cluster scope: right when the data it publishes is an async-proxy TMEM write
already completed by ``tcgen05_wait(STORE)``.  A follower that writes an MMA operand into ITS OWN
SMEM with generic stores (``store_swizzled``), fences it into the async proxy and hands it to a leader-issued
``cta_group::2`` MMA needs the arrive to RELEASE those stores to the leader's acquiring wait, or the MMA may read the
slab stale -- a load-dependent first-launch race.  The
form is ``mbarrier.arrive.release.cta.shared::cluster.b64`` on the mapa'd leader mbar (the DSL's own default for a remote
arrive and the SM100 dkdv chain's ``producer_commit``), NEVER ``.release.cluster``: that puts a ``MEMBAR.ALL.GPU`` +
``CGAERRBAR`` drain ahead of every arrive (removing that drain took a cga2 kernel from 47 % to 92 % of SOL).

Tiers, one probe (2 CTAs x 32 lanes; every lane writes a pattern into its CTA's slab, fences, arrives on the leader's
mbar; the leader waits and reads the FOLLOWER's slab over DSMEM):

* tier 1 (any box whose cutlass-dsl knows ``sm_107a``): the trace-compile is asserted; the PTX carries exactly ONE
  ``mbarrier.arrive.release.cta.shared::cluster.b64`` for the release form (zero ``.release.cluster``), the relaxed
  control keeps its ``.relaxed.cluster`` form, the cga1 form falls through to a plain local arrive; with a decoding
  nvdisasm the SASS of every form has ``CGAERRBAR == 0`` and ``MEMBAR.ALL.GPU == 0`` (the drain the rule forbids).
* tier 2 (``requires_rubin``): the probe RUNS -- the leader reads the follower's 32-lane pattern exactly, for the release
  form, the relaxed control and the cga1 form (a single-launch smoke of the publish-then-read shape; the 12-process x
  {relaxed, release} x {follower delay} micro-probe under an async-proxy consumer belongs with the first kernel that
  consumes the release arrive).

The two ``mbar_arrive_on_peer(..., release_cta=True)`` arms ride the same probe (forms ``peer_release`` / ``peer_release_pred``): the
UNPREDICATED arm (``mbarrier_arrive(mapa(mb, peer), scope=CTA, relaxed=False)``, no kernel calls it today -- pinned here so it cannot
rot unseen) and the predicated inline-PTX arm (the multicast bias ring's cross-CTA stage credit, one elected lane per warp; its kernel
module pins the rendering, this module pins the helper).  Both must read exactly one ``.release.cta.shared::cluster`` arrive, zero
relaxed, zero ``.release.cluster``, and no GPU-scope drain in SASS; the probe also prints every ``SYNCS.ARRIVE`` mnemonic it decodes
(``SASS_ARRIVE_MNEMONICS``), because the suffix is the lowering shape, not the memory order.
"""

import glob
import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest
import torch

from frost_test_utils import arch_known_to_the_dsl, nvdisasm_candidates, requires_dsl, requires_rubin

pytestmark = [pytest.mark.L0, requires_dsl]

FORM_RELEASE, FORM_RELAXED, FORM_CGA1, FORM_PEER_RELEASE, FORM_PEER_RELEASE_PRED = 0, 1, 2, 3, 4
FORMS = {"release": FORM_RELEASE, "relaxed": FORM_RELAXED, "cga1": FORM_CGA1, "peer_release": FORM_PEER_RELEASE, "peer_release_pred": FORM_PEER_RELEASE_PRED}
RELEASE_FORMS = (FORM_RELEASE, FORM_PEER_RELEASE, FORM_PEER_RELEASE_PRED)
LANES = 32

_PROBE_SRC = textwrap.dedent('''
    """Probe: 2 CTAs x 32 lanes; each lane stores a pattern into its CTA's SMEM slab, fences, arrives on the LEADER's
    mbar (32 x 2 = 64 arrives); the leader waits and copies the follower's slab (DSMEM read) to GMEM."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.experimental import primitives as nvvm

    from cudnn.frost.tile_dsl.barrier import MBarrier, Producer, Scope, cga_arrive, cga_wait
    from cudnn.frost.tile_dsl.tma import st_global

    LANES = 32
    FORM_RELEASE, FORM_RELAXED, FORM_CGA1, FORM_PEER_RELEASE, FORM_PEER_RELEASE_PRED = 0, 1, 2, 3, 4


    @cute.kernel
    def probe(mOut: cute.Tensor, form: cutlass.Constexpr[int]):
        cta_group = 1 if form == FORM_CGA1 else 2
        producer = Producer.LEADER if form in (FORM_RELAXED, FORM_CGA1) else Producer.LEADER_RELEASE
        # the predicated peer form fires ONE elected lane per warp (one warp per CTA here); every other form arrives bare on all 32 lanes
        arrivers = 1 if form == FORM_PEER_RELEASE_PRED else LANES
        tidx = cutlass.Int32(cute.arch.thread_idx()[0])
        cta = cutlass.Int32(cute.arch.block_idx_in_cluster()) if cutlass.const_expr(cta_group == 2) else cutlass.Int32(0)
        slab = cutlass.Array(cutlass.Int32, LANES, alignment=16, space=cutlass.AddressSpace.smem)
        bar_raw = cutlass.Array(cutlass.Int64, 1, alignment=8, space=cutlass.AddressSpace.smem)
        # init count = SUM(issuing lanes): arrivers x cta_group CTAs (32 bare lanes, or the one elected lane of the predicated peer form)
        bar = MBarrier(base_ptr=bar_raw, stages=1, init_count=arrivers * cta_group, producer=int(producer), scope=int(Scope.LEADER))
        if tidx == 0:
            bar.init()
        nvvm.fence_mbarrier_init()
        nvvm.barrier_cta_sync()
        if cutlass.const_expr(cta_group == 2):
            cga_arrive()
            cga_wait()
        # the follower's generic store of "its MMA operand", then the production recipe: fence into the async proxy, arrive
        slab.subview(tidx).store(cta * cutlass.Int32(1000) + tidx * cutlass.Int32(7) + cutlass.Int32(1))
        nvvm.fence_proxy("async.shared", space="cta")
        if cutlass.const_expr(form == FORM_PEER_RELEASE):
            # mbar_arrive_on_peer(release_cta=True) UNPREDICATED = mbarrier_arrive(mapa(mb, 0), scope=CTA, relaxed=False); no kernel caller today
            bar.arrive_on_peer(cutlass.Int32(0), release_cta=True)
        elif cutlass.const_expr(form == FORM_PEER_RELEASE_PRED):
            # the multicast bias ring's stage credit: the predicated inline-PTX .release.cta arrive from one elected lane per warp
            bar.arrive_on_peer(cutlass.Int32(0), pred=nvvm.elect_sync(), release_cta=True)
        else:
            bar.arrive(cta_group=cta_group, leader_cta_id=cutlass.Int32(0))
        if cta == 0:
            bar.wait(0)
            # the leader reads the PEER's slab (cga2: CTA 1; cga1: mapa(self, 0) is the local slab)
            peer_elem = nvvm.mapa(slab.subview(tidx), cutlass.Int32(cta_group - 1))
            v = nvvm.inline_ptx("ld.shared::cluster.b32 $0, [$1];", write_only_types=[cutlass.Int32], read_only_args=[peer_elem.ir_value()])
            st_global(mOut.iterator.toint() + tidx.to(cutlass.Int64) * cutlass.Int64(4), v, cutlass.Int32)
        else:
            # the follower publishes its own slab from a local read as the control row
            st_global(mOut.iterator.toint() + cutlass.Int64(LANES * 4) + tidx.to(cutlass.Int64) * cutlass.Int64(4), slab.subview(tidx).load(), cutlass.Int32)
        if cutlass.const_expr(cta_group == 2):
            # P15: the follower must stay resident until the leader has read its SMEM
            cga_arrive()
            cga_wait()


    probe.set_name_prefix("cudnn", remove_cutlass_symbol=True)


    @cute.jit
    def launch(out: cute.Tensor, form: cutlass.Constexpr[int], stream: cuda.CUstream):
        if cutlass.const_expr(form == FORM_CGA1):
            probe(out, form).launch(grid=(1, 1, 1), block=(LANES, 1, 1), stream=stream)
        else:
            probe(out, form).launch(grid=(2, 1, 1), block=(LANES, 1, 1), cluster=(2, 1, 1), stream=stream)


    if __name__ == "__main__":
        # argv: <arch> <form 0/1/2> <dump dir> [nvdisasm candidates...]; CUTE_DSL_DUMP_DIR is set by the caller (keeps ptx + cubin).
        import glob, os, subprocess, sys
        from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

        arch, form, dump, cands = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:]
        out = make_fake_tensor(dtype=cutlass.Int32, shape=(cute.sym_int(),), stride=(1,), assumed_align=16)
        cute.compile(launch, out, form, make_fake_stream(use_tvm_ffi_env_stream=False), options=f"--enable-tvm-ffi --gpu-arch {arch} --keep-cubin --keep-ptx")
        print("COMPILED", arch, form)
        ptxs = glob.glob(os.path.join(dump, f"*.{arch}.ptx"))
        if not ptxs:
            print("FAIL no ptx landed in", dump, os.listdir(dump)); sys.exit(3)
        ptx = open(ptxs[0]).read()
        for key, pat in (("RELEASE_CTA_CLUSTER", "mbarrier.arrive.release.cta.shared::cluster.b64"),
                         ("RELEASE_CLUSTER_CLUSTER", "mbarrier.arrive.release.cluster.shared::cluster.b64"),
                         ("RELAXED_CLUSTER_CLUSTER", "mbarrier.arrive.relaxed.cluster.shared::cluster.b64"),
                         ("ANY_CLUSTER_ARRIVE", "shared::cluster.b64 _"),
                         ("LOCAL_ARRIVE", "mbarrier.arrive.shared")):
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
        print("SASS CGAERRBAR", sum(1 for ln in sass if "CGAERRBAR" in ln))
        print("SASS MEMBAR_GPU", sum(1 for ln in sass if "MEMBAR.ALL.GPU" in ln))
        print("SASS MEMBAR", sum(1 for ln in sass if "MEMBAR" in ln))
        print("SASS ARRIVE", sum(1 for ln in sass if "SYNCS.ARRIVE" in ln))
        print("SASS LINES", len(sass))
        import re
        mnem = sorted({m.group(0) for ln in sass for m in [re.search(r"U?SYNCS\.ARRIVE[\w.]*", ln)] if m})
        print("SASS_ARRIVE_MNEMONICS", " ".join(mnem) if mnem else "none")
    ''')


def _write_probe(tmp_path):
    probe_py = tmp_path / "release_arrive_probe.py"
    probe_py.write_text(_PROBE_SRC)
    return probe_py


def _run_probe(tmp_path, *, arch: str, form: int, timeout: int = 900):
    dump = tmp_path / f"dump_{arch}_{form}"
    dump.mkdir()
    env = dict(os.environ, CUTE_DSL_DUMP_DIR=str(dump), CUTE_DSL_KEEP="ptx,cubin")
    argv = [sys.executable, str(_write_probe(tmp_path)), arch, str(form), str(dump), *nvdisasm_candidates()]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)


def _parse(lines):
    ptx = {ln.split()[1]: int(ln.split()[2]) for ln in lines if ln.startswith("PTX ") and len(ln.split()) == 3}
    sass = {ln.split()[1]: int(ln.split()[2]) for ln in lines if ln.startswith("SASS ") and len(ln.split()) == 3}
    return ptx, sass


# ---------------------------------------------------------------------------
# Tier 1: PTX form + SASS drain pins on an sm_107a trace-compile
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form_name", list(FORMS))
def test_sm107a_release_arrive_ptx_form_and_no_gpu_drain(tmp_path, form_name):
    """release / peer_release / peer_release_pred -> exactly one ``mbarrier.arrive.release.cta.shared::cluster.b64`` and NO
    ``.release.cluster`` (the two ``mbar_arrive_on_peer(release_cta=True)`` arms, unpredicated and predicated, read the same form as
    ``arrive_on_leader_release``); relaxed -> the ``.relaxed.cluster`` form (the ``arrive_on_leader`` control, untouched); cga1 -> a
    plain local arrive and no cluster arrive at all.  With a decoding nvdisasm: ``CGAERRBAR == 0`` and ``MEMBAR.ALL.GPU == 0`` for
    every form, and the decoded ``SYNCS.ARRIVE`` mnemonics are printed (``-s``) for the record."""
    if not arch_known_to_the_dsl("sm_107a"):
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0)")
    form = FORMS[form_name]
    proc = _run_probe(tmp_path, arch="sm_107a", form=form)
    assert proc.returncode == 0, f"sm_107a trace-compile of the {form_name} probe failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    lines = proc.stdout.splitlines()
    assert any(ln.startswith("COMPILED sm_107a") for ln in lines), proc.stdout[-2000:]
    ptx, sass = _parse(lines)
    mnem = [ln for ln in lines if ln.startswith("SASS_ARRIVE_MNEMONICS")]
    print(f"\n[{form_name}] PTX {ptx} SASS {sass} {mnem[0] if mnem else ''}")
    assert ptx["RELEASE_CLUSTER_CLUSTER"] == 0, f"a .release.cluster arrive is a MEMBAR.ALL.GPU + CGAERRBAR drain per arrive -- {ptx}"
    if form in RELEASE_FORMS:
        assert ptx["RELEASE_CTA_CLUSTER"] == 1 and ptx["RELAXED_CLUSTER_CLUSTER"] == 0, ptx
    elif form == FORM_RELAXED:
        assert ptx["RELAXED_CLUSTER_CLUSTER"] == 1 and ptx["RELEASE_CTA_CLUSTER"] == 0, ptx
    else:
        assert ptx["ANY_CLUSTER_ARRIVE"] == 0 and ptx["LOCAL_ARRIVE"] >= 1, f"cga1 must fall through to a plain local arrive -- {ptx}"
    if not nvdisasm_candidates():
        pytest.skip("compiled + PTX pinned; no nvdisasm executable to try for the SASS half (CUDA_PATH unset and none on PATH)")
    if any(ln.startswith("SKIP") for ln in lines):
        pytest.skip(f"compiled + PTX pinned; SASS half skipped: {[ln for ln in lines if ln.startswith(('SKIP', 'REJECT'))]}")
    assert sass["CGAERRBAR"] == 0 and sass["MEMBAR_GPU"] == 0, f"{form_name}: a GPU-scope drain sits before the cluster arrive -- {sass}"


# ---------------------------------------------------------------------------
# Tier 2: the publish-then-read runs on a Rubin device
# ---------------------------------------------------------------------------


@requires_rubin
@pytest.mark.parametrize("form_name", list(FORMS))
def test_leader_reads_the_followers_slab_after_the_arrive(tmp_path, form_name):
    """The leader's 32 lanes read the follower's pattern ``1000 + 7 * lane + 1`` exactly (cga2 -- the release form, the relaxed
    control and both ``arrive_on_peer(release_cta=True)`` arms), or their own ``7 * lane + 1`` at cga1; the follower's control row is
    its own slab."""
    import cuda.bindings.driver as _cuda_driver
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    spec = importlib.util.spec_from_file_location("_release_arrive_probe_device", str(_write_probe(tmp_path)))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    form = FORMS[form_name]
    dev = torch.device("cuda")
    out = torch.full((2 * LANES,), -1, dtype=torch.int32, device=dev)
    stream = _cuda_driver.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    arg = from_dlpack(out, assumed_align=16)
    fn = cute.compile(mod.launch, arg, form, stream)
    fn(arg, stream)
    torch.cuda.synchronize()
    lane = torch.arange(LANES, dtype=torch.int32, device=dev)
    if form == FORM_CGA1:
        expect_leader = 7 * lane + 1
        assert torch.equal(out[:LANES], expect_leader), out[:LANES].tolist()
    else:
        expect_leader = 1000 + 7 * lane + 1
        assert torch.equal(out[:LANES], expect_leader), f"leader read of the follower's slab: {out[:LANES].tolist()}"
        assert torch.equal(out[LANES:], expect_leader), f"follower control row: {out[LANES:].tolist()}"
