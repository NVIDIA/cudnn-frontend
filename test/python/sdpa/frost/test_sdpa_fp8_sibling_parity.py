# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sibling-lockstep fences for the per-tensor FP8 kernel pair.

``prefill_d128_fp8_sm107.py`` is a hunk-symmetric sibling of the SM100 file
with a small set of INTENTIONAL divergences (Rubin K=64 MMA geometry, deeper
KV ring, baked LDTM/rowsum). Everything else must move in lockstep — and has
silently not, three times: #574 (compile-signature skew), #602 (amax ABI
drift), #585/#653 (the LPT scheduler landed in one file only, breaking every
causal FP8 graph on the sibling's arch). These tests turn that class of
drift into a red test at the PR that causes it.

The source-scan tests need no GPU and no DSL — they run everywhere.
"""

import inspect
import re
from pathlib import Path

import pytest

from frost_test_utils import requires_dsl

from cudnn.sdpa.fwd.config_sm100 import TemplateParams

pytestmark = pytest.mark.L0

_E4M3, _BF16_OUT = 0, 2
_SIBLINGS = ("prefill_d128_fp8_sm100.py", "prefill_d128_fp8_sm107.py")
# CFG fields the sibling intentionally re-derives (dataclasses.replace at
# import). Anything else differing is drift.
_CFG_DIVERGENCE_ALLOWLIST = {"TILE_K_HW_BMM1", "TILE_K_HW_BMM2", "STAGES_KV"}


def _kernel_dir() -> Path:
    from cudnn.sdpa.fwd import api_dsl

    return Path(api_dsl.__file__).parent / "kernels"


def _decode_calls(src: str):
    """Every ``_dispatch_decode_*`` call block in a kernel source."""
    out = []
    for m in re.finditer(r"_dispatch_decode_(initial|payload)\(", src):
        depth, i = 0, m.end() - 1
        while True:
            c = src[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        out.append(src[m.start() : i])
    return out


def test_sched_decode_call_sites_carry_lpt_args():
    """#653's fence: every scheduler decode call in BOTH files must pass
    qh_per_kh and seqlen_kv (SCHED_LPT_L2 needs them at every call site; a
    file whose calls lack them silently loses LPT support)."""
    counts = {}
    for f in _SIBLINGS:
        src = (_kernel_dir() / f).read_text()
        calls = _decode_calls(src)
        assert calls, f"{f}: no scheduler decode call sites found (parser drift?)"
        for blk in calls:
            assert "qh_per_kh" in blk and "seqlen_kv" in blk, f"{f}: decode call missing LPT args:\n{blk}"
        counts[f] = len(calls)
    assert counts[_SIBLINGS[0]] == counts[_SIBLINGS[1]], f"decode call-site count drift: {counts}"


def test_sched_domain_declared_in_both_sources():
    """The decode-domain constant must exist in both files and match — the
    adapter's defaulting clamp and the engine rows are tied to it."""
    doms = {}
    for f in _SIBLINGS:
        src = (_kernel_dir() / f).read_text()
        m = re.search(r"SUPPORTED_SCHED_POLICIES = frozenset\(\{([^}]*)\}\)", src)
        assert m, f"{f}: SUPPORTED_SCHED_POLICIES declaration missing"
        doms[f] = {x.strip() for x in m.group(1).split(",") if x.strip()}
    assert doms[_SIBLINGS[0]] == doms[_SIBLINGS[1]], f"sched domain drift: {doms}"


def _load(rubin):
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module

    return _load_sm100_kernel_module(
        (128, 128),
        TemplateParams(dtype_qkv=_E4M3, dtype_o=_BF16_OUT),
        fp8=True,
        pertensor=True,
        rubin=rubin,
    )


@requires_dsl
def test_compile_signatures_in_lockstep():
    """#574's fence: the shared adapter calls both modules' ``compile`` with
    the same keywords — the signatures must be identical."""
    sig100 = inspect.signature(_load(rubin=False).compile)
    sig107 = inspect.signature(_load(rubin=True).compile)
    assert list(sig100.parameters) == list(sig107.parameters), f"compile param drift: {list(sig100.parameters)} vs {list(sig107.parameters)}"


@requires_dsl
def test_sched_domains_match_adapter_table():
    """#1's invariant: file constants == the adapter's routing-level domain
    table, for both routes. A default may never be a value the served kernel
    cannot decode."""
    from cudnn.sdpa.fwd.api_dsl import _sm100_sched_domain

    for rubin in (False, True):
        mod = _load(rubin=rubin)
        assert mod.SUPPORTED_SCHED_POLICIES == _sm100_sched_domain(rubin), (
            f"rubin={rubin}: kernel file declares {sorted(mod.SUPPORTED_SCHED_POLICIES)}, " f"adapter table says {sorted(_sm100_sched_domain(rubin))}"
        )


@requires_dsl
def test_cfg_divergence_is_exactly_the_allowlist():
    """Every CFG field must match between siblings except the declared Rubin
    re-derivations — a new silent divergence is drift until allowlisted."""
    import dataclasses

    cfg100 = _load(rubin=False).CFG
    cfg107 = _load(rubin=True).CFG
    diverged = {f.name for f in dataclasses.fields(cfg100) if getattr(cfg100, f.name) != getattr(cfg107, f.name)}
    assert diverged == _CFG_DIVERGENCE_ALLOWLIST, (
        f"CFG divergence drift: unexpected={sorted(diverged - _CFG_DIVERGENCE_ALLOWLIST)}, " f"missing={sorted(_CFG_DIVERGENCE_ALLOWLIST - diverged)}"
    )
