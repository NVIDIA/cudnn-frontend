# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers of the d128 grouped THD cta_group::2 pair-body test modules (``test_sdpa_bwd_d128_2cta_stail_sm100.py``,
``test_sdpa_bwd_d128_2cta_stail_xfa4_sm100.py``, ``test_sdpa_bwd_d128_cta_pair_heuristic.py``, ``test_sdpa_bwd_d128_dkv_accum_sm100.py``):
the direct-entry adapter construction, the part gate, the host-render template with the PTX census of the pair chain, and the constants of the
fresh-process pair-skew detectors.  No test here (not collected): pytest imports it by name."""

import os
import re
import subprocess
import sys

import pytest
import torch

from d128_grouped_test_utils import adapter_arguments, check_case, make_workspace, poison
from d128_thd_test_utils import ROLES, SCALE
from frost_test_utils import ptx_kernel_md5s
from test_sdpa_bwd_d128_accum_sm100 import _require_accumulation

_ADAPTER_MODULE = "cudnn.sdpa.bwd.api_dsl_sm100_d128_2cta"
_HOST_MODULE = "cudnn.sdpa.bwd.kernels.sm100.prepared_d128_grouped_2cta_host"
_IMPLEMENTATION = "accumulation_grouped_thd_2cta"
RECORD_ARCH = "sm_100a"  # the arch every committed rendering record is pinned on (frost_test_utils.read_kernel_record's default)
# Exit codes of the fresh-process rows (the parents read them by name): the watchdog's HANG kill and the typed decline of the pair body.
CHILD_EXIT_HANG = 70  # frost_test_utils.process_watchdog: os._exit(70) inside the kernel stage = HANG -> the barrier AUDIT, never a bigger budget
CHILD_EXIT_SKIP = 77  # the child's pair body DECLINED by name on this device (make_adapter_2cta -> pytest.skip): the parent SKIPS with the reason
_CHILD_SKIP_LINE = "CHILD_SKIP "


def _adapter_class():
    """The 2-CTA adapter of THIS tree; a venv whose ``cudnn`` is another checkout has no such module (AGENTS.md: editable installs pin one checkout)."""
    import importlib

    import cudnn

    try:
        return importlib.import_module(_ADAPTER_MODULE).SdpaBwdDslSm100D128Cta2
    except ImportError as error:  # an explicit raise (pytest.skip() is one too, but static analysis cannot see it)
        raise pytest.skip.Exception(
            f"{_ADAPTER_MODULE} is not importable from {cudnn.__file__}: this venv's cudnn is not the tree under test ({error})"
        ) from None


def _require_2cta():
    """The pair-body GPU rows' part gate reads the library's own cc set (``CTA_PAIR_CCS``: the SM100 line, cc 10.0 / 10.3 -- never a copy of
    it), so the forced two-CTA rows admit exactly the devices the library admits (#1985) and skip elsewhere.  Host rows never call it."""
    from cudnn.sdpa.bwd.d128_grouped_2cta_contract import CTA_PAIR_CCS

    _require_accumulation()
    cc = torch.cuda.get_device_capability()
    if cc not in CTA_PAIR_CCS:
        pytest.skip(f"forced cta_group::2 d128 backward requires cc in {CTA_PAIR_CCS}; got {cc}")


def _check_support_or_skip(api):
    """``api.check_support()`` with the library's TYPED decline (``NotImplementedError``: the cc / cluster / DSL / field predicates, by name)
    mapped to a pytest skip that carries the decline text -- a device the pair body does not serve SKIPS every pair-body row, never fails it
    (the sm80 / sm103 / sm120 CI lanes).  A ``ValueError`` (a field outside its domain) and any other exception propagate: those are bugs.
    The library returns ``True`` or raises; a silent ``False`` is a library bug and fails here by name."""
    try:
        supported = api.check_support()
    except NotImplementedError as error:
        raise pytest.skip.Exception(f"pair body declined on this device: {error}") from None  # an explicit raise: static analysis sees it
    assert supported is True, f"{type(api).__name__}.check_support() returned {supported!r} without a typed decline: a silent False is a library bug"
    return api


def make_adapter_2cta(case, *, compile_plan=True, declaration=None, **fields):
    """The suite's adapter construction (``d128_grouped_test_utils.make_adapter``) on the 2-CTA subclass (``cta_pair = "on"``: the pair body at
    any S) plus any typed field set on the instance before ``check_support()`` (``diag_2cta`` / ``mask_form`` / ``warp_roles`` / ``dkv_accum`` /
    ``debug_clk``); ``declaration`` selects the storage declaration helper (the grouped one by default; the MHA utils' for an MHA case)."""
    from cudnn.api_base import TensorDesc

    if declaration is None:
        from d128_grouped_test_utils import declaration

    cls = _adapter_class()
    samples = {}
    for role in ROLES:
        shape, stride = declaration(case, role)
        samples["sample_" + role] = TensorDesc(
            dtype=torch.float32 if role == "stats" else case.dtype,
            shape=shape,
            stride=stride,
            stride_order=tuple(sorted(range(4), key=lambda i: (stride[i], shape[i]))),
            device=case.tensors["q"].device,
            name=role,
        )
    thd = getattr(case, "thd", True)  # the MHA utils' case is always THD and carries no flag
    api = cls(
        **samples,
        is_causal=case.causal,
        scale_softmax=SCALE,
        thd=thd,
        max_total_seq_len_q=case.capacities[0] if thd else None,
        max_total_seq_len_kv=case.capacities[1] if thd else None,
        thd_stats_token_major=case.stats_layout == "TH1",
        thd_stats_head_stride=case.head_pitch if case.stats_layout == "HN" else None,
        thd_ragged_offsets=case.offset_declarations,
    )
    for name, value in fields.items():
        setattr(api, name, value)
    if not compile_plan:
        return api
    _check_support_or_skip(api)
    api.compile()
    assert api._prepared.native is not None
    assert api._prepared.artifact.implementation == _IMPLEMENTATION
    return api


def child_adapter(make, *args, **fields):
    """In a FRESH child process (the skew / hang-repro rows): build the pair body through ``make`` (``make_adapter_2cta_stail`` or a sibling) or
    exit ``CHILD_EXIT_SKIP`` with the decline on stdout -- the parent maps it to ``pytest.skip`` (``skip_if_child_declined``)."""
    try:
        return make(*args, **fields)
    except pytest.skip.Exception as skipped:
        print(f"{_CHILD_SKIP_LINE}{skipped}", flush=True)
        sys.exit(CHILD_EXIT_SKIP)


def skip_if_child_declined(proc, what):
    """The parent side of ``child_adapter``: a child that exited ``CHILD_EXIT_SKIP`` skips this row with the child's reason (the decline text);
    every other exit code is the caller's to read (``CHILD_EXIT_HANG``, 0, the child's own error)."""
    if proc.returncode != CHILD_EXIT_SKIP:
        return
    line = next((ln[len(_CHILD_SKIP_LINE) :] for ln in proc.stdout.splitlines() if ln.startswith(_CHILD_SKIP_LINE)), None)
    pytest.skip(f"[{what}] the child's {line or 'pair body declined by name (no reason line printed)'}")


def run_case_2cta(case, repeats=2, **fields):
    """Execute the pair chain ``repeats`` times on a poisoned workspace and gate every output against the suite's fp64 reference."""
    api = make_adapter_2cta(case, **fields)
    workspace = make_workspace(api.scratch_workspace_bytes())
    for _ in range(repeats):
        poison(case, workspace)
        api.execute(**adapter_arguments(case), workspace=workspace.tensor)
        torch.cuda.synchronize()
        check_case(case, workspace)
    return api, workspace


# ------------------------------------------------------------------------------------------------------------ host renders (no GPU)
_RENDER = """
import os, sys, re, glob
os.environ["CUTE_DSL_DUMP_DIR"] = sys.argv[1]
os.environ["CUTE_DSL_KEEP"] = "ptx"
os.environ["CUTE_DSL_ARCH"] = {arch!r}
os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
import cutlass
from {host} import compile_host
ports = ("q", "k", "v", "o", "do", "dq", "dk", "dv", "stats")
B, H, HK = 2, 8, 2
ts = lambda p: H if p == "stats" else (HK if p in ("k", "v", "dk", "dv") else H) * 128
specs = tuple((i, 1, ts(p), "int32") for i, p in enumerate(ports))
compile_host((B, H, HK, True), specs, tuple("int32" for _ in ports), cutlass.BFloat16, True, {sm}, **{form_kw})  # sm = the device cc as the prepared module passes it
ptx_file = sorted(glob.glob(os.path.join(sys.argv[1], "*.ptx")), key=os.path.getmtime)[-1]
ptx = open(ptx_file, "rb").read()
import hashlib
print("PTX_MD5", hashlib.md5(ptx).hexdigest())
print("PTX_FILE", ptx_file)
print("PTX_TARGET", re.search(rb"^\\.target\\s+(\\S+)", ptx, re.M).group(1).decode())
for name, pat in {census}.items():
    print("CENSUS", name, len(re.findall(pat.encode(), ptx)))
for action, imm in re.findall(rb"setmaxnreg\\.(inc|dec)\\.sync\\.aligned\\.u32\\s+(\\d+)", ptx):
    print("SETMAXNREG", action.decode(), imm.decode())
print("RENDER_OK")
"""
# The PTX census of the pair chain (every row a regex over the sm_100a PTX text; the expected counts live beside each module's record).
CENSUS = {
    "mma_cta_group_2": r"tcgen05\.mma\.cta_group::2",
    "mma_cta_group_1": r"tcgen05\.mma\.cta_group::1",
    "commit_multicast": r"tcgen05\.commit\.cta_group::2\.mbarrier::arrive::one\.shared::cluster\.multicast::cluster",
    "tma_cta_group_2": r"multicast::cluster\.cta_group::2",  # the one-CTA-mask multicast form of every cga2 load
    "tma_plain_cta_group_2": r"complete_tx::bytes\.cta_group::2",  # a load without the multicast modifier: none in this chain
    "dsmem_ship": r"cp\.async\.bulk\.shared::cluster\.shared::cta",
    "reduce_add": r"cp\.reduce\.async\.bulk",
    "try_cancel": r"clusterlaunchcontrol\.try_cancel",
    "tmem_alloc_cta2": r"tcgen05\.alloc\.cta_group::2",
    "local_mem": r"\.local",
    "entry": r"\.entry\s+cudnn_kernel_bprop_d128_accum_grouped_2cta_kernel",  # an earlier pair body's entry: absent from this chain
    "nanosleep": r"nanosleep\.u32",  # the test-only DEBUG_*_DELAY_US levers: absent from the default rendering
    "mbarrier_arrive": r"mbarrier\.arrive",  # every arrive site of the body's header section 5 (a DIAGNOSTIC probe never moves one)
    "try_wait": r"mbarrier\.try_wait",  # the wait sites
}
# The pair chain's default counts (the body's header section 5 / 11): the base MMA count of the one-pass form plus the dQ issue at TWO
# sites (in-tile dQ(t-1) + the tail dQ(hi-1)) = 72, the nine MMA_COMMIT rows + the S^T lookahead pair + the zero-trip dq_done + the tail's
# dq_done commit = 13, twelve cga2 loads, one DSMEM ship, four dQ reduce-adds per tile.
EXPECTED_BASE = {
    "mma_cta_group_2": 56,
    "mma_cta_group_1": 0,  # never mixed
    "commit_multicast": 12,
    "tma_cta_group_2": 12,  # K1 2 + V 2 + K2 2 + Q1 2 + dO1 2 + dO2 1 + Q2 1
    "tma_plain_cta_group_2": 0,
    "dsmem_ship": 1,
    "reduce_add": 4,
    "try_cancel": 1,
    "tmem_alloc_cta2": 1,
    "local_mem": 0,
    "entry": 1,
    "nanosleep": 0,
    "mbarrier_arrive": 41,
    "try_wait": 63,
}


def _setmaxnreg_of():
    """The setmaxnreg immediates the register split renders: inc x (softmax + epilogue roles = 2 sites), dec x (service roles = 5 sites)."""
    from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta import CFG

    return sorted([("inc", CFG.SOFTMAX_REGS), ("inc", CFG.EPILOGUE_REGS)] + [("dec", CFG.SERVICE_REGS)] * (CFG.SERVICE_WARPS + 1))


def render_2cta_kernels(tmp_path, form_kw, census, arch=RECORD_ARCH):
    """Host trace-compile of the pair chain for ``arch`` (the record arch sm_100a by default; sm_103a = the GB300 target the prepared host
    selects on cc 10.3) in a fresh process -> (PTX md5 of the module, census, [(setmaxnreg action, immediate), ...], {kernel: md5 of its `.entry`
    region} in entry order -- the per-kernel record form, frost_test_utils.ptx_kernel_md5s).  No device needed: the host's ``--gpu-arch sm_{sm}a``
    option (the production mechanism, ``sm`` from the device cc) selects the target -- it wins over ``CUTE_DSL_ARCH``, which is set to agree."""
    _adapter_class()  # the tree under test must be the one this venv imports
    sm = int(re.fullmatch(r"sm_(\d+)a", arch).group(1))  # the prepared module's `sm` (cc major * 10 + minor) -> the host's `--gpu-arch sm_{sm}a`
    script = _RENDER.format(host=_HOST_MODULE, census=repr(census), form_kw=repr(form_kw), arch=arch, sm=sm)
    proc = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=600, env=dict(os.environ))
    assert proc.returncode == 0 and "RENDER_OK" in proc.stdout, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert re.search(r"PTX_TARGET (\S+)", proc.stdout).group(1) == arch, proc.stdout[-500:]
    md5 = re.search(r"PTX_MD5 ([0-9a-f]{32})", proc.stdout).group(1)
    counts = {m.group(1): int(m.group(2)) for m in re.finditer(r"CENSUS (\S+) (\d+)", proc.stdout)}
    regs = [(m.group(1), int(m.group(2))) for m in re.finditer(r"SETMAXNREG (inc|dec) (\d+)", proc.stdout)]
    with open(re.search(r"PTX_FILE (\S+)", proc.stdout).group(1)) as f:
        kernels = ptx_kernel_md5s(f.read())
    return md5, counts, regs, kernels


def render_2cta(tmp_path, form_kw, census):
    """Host trace-compile of the pair chain for sm_100a in a fresh process -> (PTX md5, census, [(setmaxnreg action, immediate), ...])."""
    return render_2cta_kernels(tmp_path, form_kw, census)[:3]


# ------------------------------------------------------------------------------------------------------------ the pair-skew detectors (fresh child)
GLOBAL_PREFIX = "FROST_D128_2CTA_DEBUG_"
LEVER_LINE = re.compile(r'^DEBUG_(\w+)_DELAY_US: int = int\(globals\(\)\.get\("FROST_D128_2CTA_DEBUG_\1_DELAY_US", 0\)\)', re.M)
CHILD_TIMEOUT_S = 900  # import (~100 s cold) + JIT + the CPU fp64 reference + the watchdog budget; a TimeoutExpired is read as a HANG
WATCHDOG_S = 240.0  # execute x2 + capture + replay x4 on the skewed body; compile is outside it (a slow compiler is not a hang)
# Cells (the suite's CPU generator, seed 149: one dataset on every SM count; the SM count is printed by the child as the dataset class).
# ``skew``: uniform-ish causal with lq == lk so EVERY pair has live q tiles (1..n per item), pairs x heads x batch > SMs (a cluster runs
# 2-3 LIVE items -> tile 0 of a second item with a populated sDS / recv half), the second sequence one pair shorter, and the half-dead last
# pair (seq_k % 256 == 128).  ``p14``: n_q == 1, items > SMs, zero-trip kv blocks past seq_q.
CELLS = {
    "skew": "make_case((skv, skv - 256), (skv, skv - 256), hq=8, hkv=1, dtype=torch.bfloat16, causal=True, stats_layout='TH1')",
    "p14": "make_case((128, 64), (skv, skv - 256), hq=8, hkv=1, dtype=torch.bfloat16, causal=True, stats_layout='TH1')",
}
SKV = "256 * ((sms // (8 * 2)) + 2) + 128"  # pairs x (hq=8) x (b=2) > sms, last pair half-dead
# The four pair-skew sites every pair body carries (the barrier audit's lever forms).  ``lever`` = <SITE>; ``cell`` names the case builder
# above; ``red`` = what the detector shows on the unsound protocol; ``green`` = the expected PASS.  Line anchors are function names.
SITES_BASE = {
    "follower_own_half": dict(
        lever="FOLLOWER_OWN_HALF",
        role="epilogue warps 8-11 of the FOLLOWER (rank 1)",
        site="_epilogue_loop: between _ship_xchg(...) and the own-half _store_words(slab, own_base, ...) of tile 0 of the second item",
        delay_us=300,
        isolates="ds_pair_ready (leader MMA wait <- follower relay arrive_on_peer(release_cta=True)): the follower's XCHG has landed on the leader "
        "(leader ds_recv_full done) and the leader's own half is written (leader ds_smem done); only the relay's release.cta arrive holds dQ(t)",
        red="relay removed / relaxed (no release.cta): dQ rows [64, 128) of every tile read the follower's STALE own half of sDS",
        green="fp64 gate on dQ / dK / dV, eager x2 + capture / replay x4, no hang",
        cell="skew",
    ),
    "leader_ship": dict(
        lever="LEADER_SHIP",
        role="epilogue warps 8-11 of the LEADER (rank 0)",
        site="_epilogue_loop: before _ship_xchg(...) (the XCHG half is written, the ship is held)",
        delay_us=300,
        isolates="the relay's ds_recv_full wait on the follower + the routing of the leader's cp.async.bulk ship to the FOLLOWER's ds_recv_full",
        red="relay waits ds_smem only: dQ rows [64, 128) built on a STALE recv half (the peer's previous XCHG)",
        green="fp64 gate on dQ / dK / dV, eager x2 + capture / replay x4, no hang",
        cell="skew",
    ),
    "follower_epilogue_item_end": dict(
        lever="FOLLOWER_EPILOGUE_ITEM_END",
        role="epilogue warps 8-11 of the FOLLOWER (rank 1)",
        site="_epilogue_loop: before the item-end _native_out(... CFG.DV_COL ...) (the follower's dV / dK TMEM readout is held)",
        delay_us=300,
        isolates="dvk_empty's PAIR-wide init 64 (store warp bare arrive x 2 CTAs -> leader MMA item-top wait)",
        red="init 32 (per-CTA): the leader's dV(lo) / dK(lo) accumulate=False of item k+1 OVERWRITES the follower's unread TMEM -> dK / dV of item k wrong",
        green="fp64 gate on dQ / dK / dV, eager x2 + capture / replay x4, no hang",
        cell="skew",
    ),
    "loader_zero_trip": dict(
        lever="LOADER",
        role="TMA load warp of EITHER CTA",
        site="_load_loop: right after _credit(bars.sched_empty[...]) at the top of every item",
        delay_us=20,  # >= 10 us: longer than a zero-trip item's whole store chain (~2-4 us)
        isolates="k2_empty's back-edge through its single waiter on the zero-trip path; the half-dead last pair makes the causal kv blocks past "
        "seq_q zero-trip",
        red="a loader that observes k2_empty once while it completes twice -> parity alias -> the next live item's MMA waits loads never armed -> HANG "
        "(child exit 70 / 124); numerics correct until then",
        green="PASS (dv_full = EPILOGUE_LANES + STORE_LANES = 160: the loader arrives dv_full after its k2_empty observation)",
        cell="p14",
    ),
}
