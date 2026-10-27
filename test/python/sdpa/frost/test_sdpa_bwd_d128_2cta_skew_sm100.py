# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pair-skew DETECTORS for the d128 grouped THD backward on the cta_group::2 datapath (``bprop_d128_accum_grouped_2cta_f16.py``).

The barrier-inspector's AUDIT (2026-10-26 23:10 UTC, body @ b2d05784; fix A landed in d0a669c7) specified four skew-lever FORMS,
one per cross-CTA object whose protocol a natural warp skew cannot reach on the validation cells.  Each form sleeps ONE role of
ONE CTA at ONE site through the house lever -- a module global read when the body executes, default 0 = byte-identical
rendering (the parent module's PTX census pins ``nanosleep`` = 0 on the default rendering)::

    DEBUG_<SITE>_DELAY_US: int = int(globals().get("FROST_D128_2CTA_DEBUG_<SITE>_DELAY_US", 0))
    if cutlass.const_expr(DEBUG_<SITE>_DELAY_US > 0):
        for _us in cutlass.range(DEBUG_<SITE>_DELAY_US, unroll=1):
            nvvm.inline_ptx("nanosleep.u32 1000;", read_only_args=[])

The test injects the global BEFORE the body executes (``find_spec`` -> ``module_from_spec`` -> ``setattr`` -> ``sys.modules`` ->
``exec_module``) in a FRESH process with the compiled cache disabled (the cache does not key the lever; RULE-GAPS 201), on a cell
with MORE PAIRS THAN SMs (a cluster runs a second CLC item: the stale-slab forms need tile 0 of a second item, the zero-trip
form needs an item mix), under the house ``process_watchdog`` (a wedged launch never returns to Python: the daemon timer exits
the child with 70 = HANG; the ``d512`` ``delayed empty observer`` tests are the pattern), and asserts the grouped suite's
independent FP64 gate (``check_case``: dQ / dK / dV apart, storage canaries) eagerly twice and under CUDA-graph capture +
replay four times.

Skip guard (G13: a required check that skips is UNCOVERED and is reported as such by ``-ra``): a site whose lever the body does
not expose yet skips BY NAME; the test inverts into a live detector the moment the code-writer adds the lever at the site --
nothing in this module changes.  ``LOADER`` (lever 4) is exposed since d0a669c7 (RED = HANG on the pre-fix body per that
commit's record; GREEN with ``dv_full`` = 160).  The RED form of every site is documented in ``SITES``; a RED here is a
barrier finding for the AUDIT, never a budget change.
"""

import os
import pathlib
import re
import subprocess
import sys

import pytest
import torch

from frost_test_utils import requires_dsl
from test_sdpa_bwd_d128_2cta_sm100 import _adapter_class, _require_2cta

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.xdist_group(name="gpu_exclusive")]

_BODY_MODULE = "cudnn.sdpa.bwd.kernels.sm100.bprop_d128_accum_grouped_2cta_f16"
_BODY_REL = pathlib.Path("python/cudnn/sdpa/bwd/kernels/sm100/bprop_d128_accum_grouped_2cta_f16.py")
_TREE = pathlib.Path(__file__).resolve().parents[4]  # test/python/sdpa/frost/<file> -> the checkout
_GLOBAL_PREFIX = "FROST_D128_2CTA_DEBUG_"
_LEVER_LINE = re.compile(r'^DEBUG_(\w+)_DELAY_US: int = int\(globals\(\)\.get\("FROST_D128_2CTA_DEBUG_\1_DELAY_US", 0\)\)', re.M)
_CHILD_TIMEOUT_S = 900  # import (~100 s cold) + JIT + the CPU fp64 reference + the watchdog budget; a TimeoutExpired is read as a HANG
_WATCHDOG_S = 240.0  # execute x2 + capture + replay x4 on the skewed body; compile is outside it (a slow compiler is not a hang)

# The inspector's four forms (AUDIT § "Skew levers").  ``lever`` = <SITE>; ``cell`` names the case builder below; ``red`` = what the
# detector shows on the broken protocol; ``green`` = the expected PASS.  Line anchors are function names (line numbers move).
SITES = {
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
        isolates="the relay's ds_recv_full wait on the follower + the P9 routing of the leader's cp.async.bulk ship to the FOLLOWER's ds_recv_full",
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
        site="_load_loop: right after _credit(bars.sched_empty[...]) at the top of every item (exposed since d0a669c7)",
        delay_us=20,  # >= 10 us: longer than a zero-trip item's whole store chain (~2-4 us)
        isolates="k2_empty's back-edge through its single waiter on the zero-trip path (AUDIT finding 1, P14-B); the P14 row's half-dead last pair "
        "makes the causal kv blocks past seq_q zero-trip",
        red="pre-fix body (b2d05784): k2_empty completes twice before the loader observes once -> parity alias -> the next live item's MMA waits "
        "loads never armed -> HANG (child exit 70 / 124); numerics correct until then",
        green="PASS after fix A (dv_full = EPILOGUE_LANES + STORE_LANES = 160; the loader arrives dv_full after its k2_empty observation)",
        cell="p14",
    ),
}

# Cells (the suite's CPU generator, seed 149: one dataset on every SM count; the SM count is printed by the child as the dataset class).
# ``skew``: uniform-ish causal with lq == lk so EVERY pair has live q tiles (1..n per item), pairs x heads x batch > SMs (a cluster runs
# 2-3 LIVE items -> tile 0 of a second item with a populated sDS / recv half), the second sequence one pair shorter, and the half-dead last
# pair (seq_k % 256 == 128).  ``p14``: the parent module's P14 row verbatim (n_q == 1, items > SMs, zero-trip kv blocks past seq_q).
_CELLS = {
    "skew": "make_case((skv, skv - 256), (skv, skv - 256), hq=8, hkv=1, dtype=torch.bfloat16, causal=True, stats_layout='TH1')",
    "p14": "make_case((128, 64), (skv, skv - 256), hq=8, hkv=1, dtype=torch.bfloat16, causal=True, stats_layout='TH1')",
}
_SKV = "256 * ((sms // (8 * 2)) + 2) + 128"  # pairs x (hq=8) x (b=2) > sms, last pair half-dead


def _body_source():
    path = _TREE / _BODY_REL
    if not path.exists():
        pytest.skip(f"{path} is not in this checkout: the pair body belongs to task/frost-sdpa-bwd-d128-gqa-thd-2cta-b200")
    return path.read_text()


def _exposed_levers(source=None):
    """The <SITE> tokens whose house-form lever line the body carries (read from SOURCE, so the host can answer without importing)."""
    return set(_LEVER_LINE.findall(_body_source() if source is None else source))


def _require_lever(site):
    row = SITES[site]
    if row["lever"] not in _exposed_levers():
        pytest.skip(
            f"UNCOVERED: the pair body does not expose {_GLOBAL_PREFIX}{row['lever']}_DELAY_US yet (code-writer: the house lever at "
            f"{row['site']}); form {site}: {row['delay_us']} us on the {row['role']}; isolates {row['isolates']}; RED = {row['red']}"
        )


# ------------------------------------------------------------------------------------------------------------ host (no GPU)
def test_2cta_skew_levers_follow_the_house_form():
    """Every lever the body exposes is one of the inspector's four, spelled in the house form (module global + const_expr nanosleep loop,
    default 0), and lever 4 (LOADER) is present since fix A; the inventory of present / absent forms is printed for ``-ra`` readers."""
    source = _body_source()
    exposed = _exposed_levers(source)
    known = {row["lever"] for row in SITES.values()}
    assert exposed <= known, f"levers outside the inspector's four forms (undocumented traced code): {sorted(exposed - known)}"
    assert "LOADER" in exposed, "lever 4 (DEBUG_LOADER_DELAY_US) landed with AUDIT fix A (d0a669c7) and must stay"
    for lever in sorted(exposed):
        name = f"DEBUG_{lever}_DELAY_US"
        assert re.search(rf"cutlass\.const_expr\({name} > 0\)", source), f"{name}: the sleep must be under const_expr (default 0 = no traced code)"
        assert re.search(rf"cutlass\.range\({name}, unroll=1\)", source), f"{name}: the house loop form"
        assert source.count(f'globals().get("{_GLOBAL_PREFIX}{lever}_DELAY_US", 0)') == 1, f"{name}: one default-0 read"
        assert f"{_GLOBAL_PREFIX}{lever}" not in os.environ, f"{name} is a module global for tests, never an environment variable"
    assert source.count('nvvm.inline_ptx("nanosleep.u32 1000;"') == len(exposed), "one nanosleep loop per exposed lever"
    inventory = {site: ("present" if row["lever"] in exposed else "ABSENT (skips: uncovered)") for site, row in SITES.items()}
    print("\n2-CTA skew levers:", inventory)


# ------------------------------------------------------------------------------------------------------------ GPU (fresh process, watchdog)
_CHILD = """
import os, sys, importlib.util
from pathlib import Path
os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"  # the compile cache does not key the lever: never serve the unarmed body
sys.path.insert(0, {testdir!r})
name = {body!r}
spec = importlib.util.find_spec(name)
mod = importlib.util.module_from_spec(spec)
setattr(mod, {global_name!r}, {delay})  # BEFORE the module executes: the body reads it into DEBUG_<SITE>_DELAY_US (const_expr)
sys.modules[name] = mod
spec.loader.exec_module(mod)
assert getattr(mod, {attr!r}) == {delay}, getattr(mod, {attr!r}, None)
import torch
import cudnn
assert Path(cudnn.__file__).resolve() == Path({expected_fe!r}).resolve(), cudnn.__file__
from d128_grouped_test_utils import adapter_arguments, check_case, make_case, make_workspace, poison
from frost_test_utils import process_watchdog
from test_sdpa_bwd_d128_2cta_sm100 import make_adapter_2cta
sms = torch.cuda.get_device_properties(0).multi_processor_count
skv = {skv}
print("DATASET sms=%d skv=%d (inputs: the suite's CPU generator, seed 149)" % (sms, skv), flush=True)
case = {cell}
api = make_adapter_2cta(case)  # check_support + compile + the implementation pin, OUTSIDE the watchdog
workspace = make_workspace(api.scratch_workspace_bytes())
args = adapter_arguments(case)
with process_watchdog({watchdog}, "2-CTA pair-skew {site}"):
    for _ in range(2):
        poison(case, workspace)
        api.execute(**args, workspace=workspace.tensor)
        torch.cuda.synchronize()
        check_case(case, workspace)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=stream):
        api.execute(**args, workspace=workspace.tensor)
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(4):
        poison(case, workspace)
        graph.replay()
        torch.cuda.synchronize()
        check_case(case, workspace)
    graph.reset()
print("SKEW_OK {site} {delay}", flush=True)
"""


@pytest.mark.parametrize("site", list(SITES))
def test_sdpa_bwd_d128_2cta_survives_pair_skew(site, tmp_path):
    """One inspector form per id (``SITES[site]``: role, site, delay, what it isolates, the RED form, the expected PASS).  A fresh process
    arms ``FROST_D128_2CTA_DEBUG_<SITE>_DELAY_US`` before the body executes, runs the ``skew`` / ``p14`` cell (more pairs than SMs) under
    ``process_watchdog`` and asserts the fp64 gate eagerly and under capture / replay.  Exit 70 (watchdog) or a subprocess timeout =
    HANG -> the barrier AUDIT, never a bigger budget; a gate miss = the RED form's wrong rows, named by ``check_case``."""
    _require_lever(site)  # FIRST: a missing lever is reported as UNCOVERED by name on any host, GPU or not
    _require_2cta()
    _adapter_class()  # the venv's cudnn must be THIS tree (the child asserts the same path)
    import cudnn

    row = SITES[site]
    script = _CHILD.format(
        testdir=str(pathlib.Path(__file__).resolve().parent),
        body=_BODY_MODULE,
        global_name=f"{_GLOBAL_PREFIX}{row['lever']}_DELAY_US",
        attr=f"DEBUG_{row['lever']}_DELAY_US",
        delay=row["delay_us"],
        expected_fe=cudnn.__file__,
        skv=_SKV,
        cell=_CELLS[row["cell"]],
        watchdog=_WATCHDOG_S,
        site=site,
    )
    (tmp_path / f"skew_{site}.py").write_text(script)
    print(f"\n[{site}] {row['role']} sleeps {row['delay_us']} us at {row['site']}; isolates {row['isolates']}; RED = {row['red']}; GREEN = {row['green']}")
    try:
        proc = subprocess.run(
            [sys.executable, str(tmp_path / f"skew_{site}.py")], capture_output=True, text=True, timeout=_CHILD_TIMEOUT_S, env=dict(os.environ)
        )
    except subprocess.TimeoutExpired as e:
        (tmp_path / f"skew_{site}.log").write_text((e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""))
        pytest.fail(f"[{site}] HANG: the child did not return within {_CHILD_TIMEOUT_S} s (RED form: {row['red']}) -> barrier AUDIT, never a bigger budget")
    (tmp_path / f"skew_{site}.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    if proc.returncode == 70:
        pytest.fail(
            f"[{site}] HANG: process_watchdog killed the child after {_WATCHDOG_S:.0f} s (RED form: {row['red']})\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    assert proc.returncode == 0 and f"SKEW_OK {site} {row['delay_us']}" in proc.stdout, f"rc={proc.returncode}\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
