# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The SM100 d512 f16/bf16 forward on the 2x2 DATAPATH (sm100/prefill_d512_f16_2x2.py, TemplateParams.mma_2x2).

Host-only: the Cfg / mbarrier-ledger pins, the byte-identity of the 4x1 role-split renderings (cubin md5 with and
without the appended field, against the pin recorded before the field existed), the arrive-site counts of the kernel
source, and the SASS pins of the 2x2 cubin.  GPU (requires_blackwell, pre-cc-10.7): the graph API under the
``two_by_two`` fixture (api_dsl.D512_2X2 flipped for the test) on the d512 cases plus the directed cells the 2x2 atom
needs -- column-half-skewed S (the row-max exchange), a rescale storm, non-tile-multiple seqlens, causal
S_q = S_kv = 512 (cluster-union bounds), SWA with empty tiles, q-trim, PackGQA g4 / g64 (g128 stays role-split), THD
(served-template asserted), the (257..512) envelope head dims (384/384, 448/320) vs an fp64 reference, the CGA_M=2 vs
CGA_M=4 bitwise twin through the direct template ABI, a persistent multi-tile-per-CTA cell (per-row check), and the
pair-skew DETECTOR of the cross-twin O u V alias gate (the test-only DEBUG_STG_DELAY_US lever; RED on a per-CTA
mb_o_empty, GREEN on the pair-wide one).  The existing d512 ids of test_sdpa_fwd_dsl_sm100.py re-run under the twin
through that file's autouse ``d512_arm`` fixture (``-k "(d512 or dsv4) and two_by_two"``)."""

import importlib.util
import json as _json
import math
import os
import subprocess as _subprocess
import sys as _sys
import textwrap as _textwrap

import pytest
import torch

from test_utils import torch_fork_set_rng

from cudnn.sdpa.fwd.engines import engine_name
from frost_test_utils import _SM, assert_no_new_spills, launch_f16, requires_blackwell, requires_dsl, run_sass_probe, sass_probe_source

import test_sdpa_fwd_dsl_sm100 as _dsl

pytestmark = requires_dsl

_KERNEL_FILE = "sm100/prefill_d512_f16_2x2.py"
_TEMPLATE = "prefill_d512_f16_2x2"
_ROLE_SPLIT_TEMPLATE = "prefill_d512_f16"
_D = 512
_pre_rubin = pytest.mark.skipif(_SM == 107, reason="the 2x2 d512 kernel's cc 10.7 sibling lands in its own lane; the twin stays off on cc 10.7")


def _kernels_dir():
    from cudnn.sdpa.fwd import api_dsl

    return os.path.join(os.path.dirname(os.path.abspath(api_dsl.__file__)), "kernels")


# ---------------------------------------------------------------------------------------------------- host-only: Cfg pins


@pytest.mark.L0
def test_config_default_arm_is_unchanged():
    """A record without the field (or with it False) builds the role-split CfgD512, dataclass-equal."""
    from cudnn.sdpa.fwd.config_sm100 import CfgD512, TemplateParams, make_cfg_d512

    c0, t0 = make_cfg_d512(TemplateParams())
    c1, t1 = make_cfg_d512(TemplateParams(mma_2x2=False))
    assert isinstance(c0, CfgD512) and c0 == c1 and t0 == t1
    assert c0.READ_TILE_ARRIVERS == 25 and c0.TILE_M == 128


@pytest.mark.L0
def test_two_by_two_host_slots_match_role_split():
    """The native dense binder (`fwd/prepared.py`) admits kernel templates BY NAME and fills host slots BY NAME, so the twin
    is served natively iff (a) the gate names `prefill_d512_f16_2x2` and (b) its `_host` takes the role split's runtime slot
    list -- both arch lines' twins against sm100/prefill_d512_f16.py.  RED-first: flipping `D512_2X2 = True` without (a)
    demoted every dense d512 plan to the Python observation path (10 width-512 cells of test_sdpa_native_prefill_binding
    red on the sm100 CI lane, `_dense_spec.native is None`, every numerics suite still green)."""
    import ast

    from cudnn.sdpa.fwd import prepared

    def runtime_slots(rel):
        with open(os.path.join(_kernels_dir(), rel)) as f:
            tree = ast.parse(f.read())
        host = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_host")
        return [a.arg for a in host.args.args if not (a.annotation is not None and "Constexpr" in ast.unparse(a.annotation))]

    parent = runtime_slots("sm100/prefill_d512_f16.py")
    assert runtime_slots(_KERNEL_FILE) == parent
    assert runtime_slots("sm107/prefill_d512_f16_2x2.py") == parent
    with open(prepared.__file__) as f:
        tree = ast.parse(f.read())
    gates = [
        {e.value for e in node.elts if isinstance(e, ast.Constant)}
        for node in ast.walk(tree)
        if isinstance(node, ast.Tuple) and any(isinstance(e, ast.Constant) and e.value == _ROLE_SPLIT_TEMPLATE for e in node.elts)
    ]
    assert gates and all(_TEMPLATE in g for g in gates), gates


@pytest.mark.L0
def test_config_2x2_pins_and_ledger_formulas():
    """The 2x2 Cfg: geometry, register split, and every mbarrier arrival-count formula (P3) re-derived from the
    role counts the kernel dispatches: 4 softmax + 4 correction + TMA-LDG + TMA-STG warps credit the scheduler on
    EVERY CTA of the cluster, the MMA warp only on the pair leader."""
    from cudnn.sdpa.fwd.config_sm100 import CfgD512X2, TemplateParams, _validate_cfg_d512_2x2, d512_2x2_smem_bytes, make_cfg_d512, make_cfg_d512_2x2
    from dataclasses import replace

    cfg, tma = make_cfg_d512(TemplateParams(mma_2x2=True))
    assert isinstance(cfg, CfgD512X2)
    assert (cfg.TILE_M, cfg.TILE_N, cfg.TILE_K, cfg.TILE_O) == (64, 128, 512, 512)
    assert (cfg.CGA_M, cfg.CGA_N, cfg.CTA_MMA, cfg.KV_SHARE, cfg.Q_SUPERS_PER_CLUSTER, cfg.ROWS_PER_CLUSTER) == (4, 1, 2, 2, 4, 256)
    assert (cfg.TOTAL_WARPS, cfg.SOFTMAX_REGS, cfg.CORRECTION_REGS, cfg.OTHER_REGS) == (12, 192, 208, 40)
    assert cfg.OTHER_REGS + cfg.CORRECTION_REGS + cfg.SOFTMAX_REGS <= 512 and 32 * 4 * (192 + 208 + 40) <= 65536
    crediting_warps_per_cta = cfg.SOFTMAX_WG_WARPS + cfg.CORRECTION_WARPS + 1 + 1
    assert crediting_warps_per_cta == 10
    assert cfg.READ_TILE_ARRIVERS == crediting_warps_per_cta * cfg.CGA_M + cfg.CGA_M // cfg.CTA_MMA == 42
    assert cfg.KV_EMPTY_ARRIVERS == cfg.CGA_M // cfg.CTA_MMA == 2
    assert cfg.O_CHUNK_ARRIVERS == cfg.CORR_LANES // 2 == 64
    assert cfg.O_EMPTY_ARRIVERS == cfg.ONE_WARP * cfg.KV_SHARE == 64  # own + twin TMA-STG warps (pair-wide O u V gate)
    assert cfg.PAIR_LANES == cfg.SOFTMAX_LANES * cfg.CTA_MMA == cfg.CORR_LANES * cfg.CTA_MMA == 256
    # alpha ring (2) + tile stats PER RING SLOT (2 x 2): cols 384..389 (the fixed 386/387 pair raced the alpha ring).
    assert cfg.O_TMEM_COLS + cfg.XFER_STAGES * cfg.S_TMEM_COLS + 2 + 2 * 2 == 390 <= cfg.TMEM_COLS
    assert cfg.XFER_STAGES >= cfg.BMM1_LOOKAHEAD + 1
    smem = d512_2x2_smem_bytes(cfg)
    assert smem["data"] == 65536 + 2 * 32768 + 2 * 32768 + 2 * 16384 and smem["n_bars"] == 35
    assert smem["total"] == 232312 <= cfg.SMEM_CAP_BYTES == 227 * 1024
    assert (tma.QK_ITERS, tma.VO_ITERS, tma.QK_GRANU_ELEMS) == (8, 8, 64)
    # The CGA_M=2 bring-up arm: one pair, own-bit loads, CfgD256's 21 credits.
    c2, _ = make_cfg_d512_2x2(TemplateParams(mma_2x2=True), cga_m=2)
    assert (c2.READ_TILE_ARRIVERS, c2.KV_EMPTY_ARRIVERS, c2.KV_SHARE, c2.ROWS_PER_CLUSTER) == (21, 1, 1, 128)
    assert c2.O_EMPTY_ARRIVERS == 32  # no twin: own TMA-STG warp only
    # The S/P slot-reuse invariant raises at trace time when violated.
    with pytest.raises(ValueError, match="XFER_STAGES"):
        _validate_cfg_d512_2x2(replace(cfg, BMM1_LOOKAHEAD=2))
    with pytest.raises(ValueError, match="READ_TILE_ARRIVERS"):
        _validate_cfg_d512_2x2(replace(cfg, READ_TILE_ARRIVERS=25))
    with pytest.raises(ValueError, match="O_CHUNK_ARRIVERS|column half"):
        _validate_cfg_d512_2x2(replace(cfg, O_CHUNK_ARRIVERS=128))
    with pytest.raises(ValueError, match="O_EMPTY_ARRIVERS"):
        _validate_cfg_d512_2x2(replace(cfg, O_EMPTY_ARRIVERS=32))  # the per-CTA gate the review's FATAL-1 found


@pytest.mark.L0
def test_config_2x2_rejects_off_contract_records():
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams, make_cfg_d512, make_cfg_d512_2x2

    with pytest.raises(ValueError, match="BF16/FP16"):
        make_cfg_d512(TemplateParams(mma_2x2=True, dtype_qkv=0, dtype_o=2))
    with pytest.raises(ValueError, match="qh_per_kh"):
        make_cfg_d512(TemplateParams(mma_2x2=True, pack_gqa=True, qh_per_kh=128))
    assert make_cfg_d512(TemplateParams(mma_2x2=True, pack_gqa=True, qh_per_kh=64))[0].PACK_G == 64
    with pytest.raises(ValueError, match="mma_2x2"):
        make_cfg_d512_2x2(TemplateParams())
    # split_kv builds the Cfg (phase 1.5 arm); the adapter twin declines it (see test_twin_declines_split_and_g128).
    assert make_cfg_d512(TemplateParams(mma_2x2=True, split_kv=4))[0].SPLIT_KV == 4


# ------------------------------------------------------------------------------------------- host-only: source ledger pins

# Arrive SITES in the kernel source per barrier (the per-phase SUMS are the Cfg constants above): a new site on a
# per-lane barrier changes its init count, so the count of sites is pinned next to the ledger.
_ARRIVE_SITE_PINS = {
    "mb_p_full[": 1,  # one per-lane release arrive per softmax iteration
    "mb_stat_full[": 2,  # per-iteration alpha + tile-end stats
    "mb_stat_empty[": 3,  # correction: kv_left consume + per-iteration + tile-end
    "mb_bmm2_ready[": 8,  # correction: 2 (kv_left) + 2 (fast arm) + 2 (slow arm) + 2 (lever-off arm)
    "mb_o_full[": 2,  # epilogue: staged + fp32-partials arms (64 lanes of one half each)
    "mb_o_empty.arrive": 1,  # TMA-STG warp, all lanes, own copy ...
    "mb_o_empty.arrive_on_peer": 1,  # ... + the twin's copy under KV_SHARE=2 = O_EMPTY_ARRIVERS (32 x KV_SHARE)
    "mb_empty_mainloop.arrive": 1,
    "mb_tmem_dealloc.arrive": 1,  # local ...
    "mb_tmem_dealloc.arrive_on_peer": 1,  # ... + on-peer = PAIR_LANES per CTA
    "mb_q_empty.arrive": 1,  # MMA commit after the tile's last BMM1
    "mb_bmm1_done[": 1,
    "mb_bmm2_done[": 2,  # per-iteration (last N-block) + empty-tile commit
}


@pytest.mark.L0
def test_kernel_source_arrive_sites_match_the_ledger():
    src = open(os.path.join(_kernels_dir(), _KERNEL_FILE)).read()
    for key, n in _ARRIVE_SITE_PINS.items():
        got = src.count(key + "].arrive(" if key.endswith("[") else key + "(")
        # the subscripted barriers are spelled `bars.mb_x[<idx>].arrive(`; count the `].arrive(` that follow the name
        if key.endswith("["):
            got = sum(1 for ln in src.splitlines() if key in ln and "].arrive(" in ln)
        assert got == n, f"{key}: {got} arrive sites, ledger says {n}"
    # The P publish keeps the proxy fence on EVERY lane immediately before the release arrive.
    lines = src.splitlines()
    idx = [i for i, ln in enumerate(lines) if "mb_p_full[" in ln and "].arrive(" in ln]
    assert len(idx) == 1 and 'fence_proxy("async.shared", space="cta")' in lines[idx[0] - 1]
    assert src.count("make_sdpa_helpers(") == 1 and "kv_shared_cluster=True" in src
    assert 'set_name_prefix("cudnn", remove_cutlass_symbol=True)' in src


_CROSS_PAIR_BARS = ("mb_k_full", "mb_k_empty", "mb_v_full", "mb_v_empty", "mb_o_empty")


def _bars_constructor_calls(src):
    """{field: MBarrier(...) call text} of the ``return D512X2Bars(`` block of make_d512_2x2_bars (paren-matched, so the
    black-wrapped multi-line constructors are whole)."""
    body = src[src.index("def make_d512_2x2_bars(") :]
    body = body[body.index("return D512X2Bars(") :]
    body = body[: body.index("\ndef ")]  # stop at the next function (make_classic_bars reuses the field names)
    calls, i = {}, 0
    while True:
        j = body.find("=MBarrier(", i)
        if j < 0:
            break
        name = body[body.rfind("\n", 0, j) + 1 : j].strip()
        depth, k = 0, j + len("=MBarrier")
        while True:
            depth += {"(": 1, ")": -1}.get(body[k], 0)
            k += 1
            if depth == 0:
                break
        calls[name] = body[j:k]
        i = k
    return calls


@pytest.mark.L0
def test_two_by_two_cross_pair_waits_poll():
    """Every barrier whose phase the OTHER pair can complete (k/v_empty: both leaders' commits mask 0xF; o_empty: the twin's
    arrive_on_peer; k/v_full: the twin pair's TMA bytes) is waited with the non-blocking test_wait poll (MBarrier.poll), and
    no pair-local barrier is: a waiter parked in the barrier unit loses a cross-pair wake-up under GPU time-slicing (the
    d512 2x2 backward hung within 2-74 launches on every parking form).  The kernel threads its POLL_CROSS_PAIR_WAITS
    constant (default ON) into make_d512_2x2_bars; the contention run (lane_d512_fprop/fix/run_contention.sh) flips it to
    0 for RED.  Source pins (the bundle allocates SMEM arrays, so it cannot be built on the host)."""
    import inspect

    from cudnn.frost.tile_dsl import barrier as _barrier
    from cudnn.sdpa.fwd.kernels import _common_blackwell as _common

    calls = _bars_constructor_calls(inspect.getsource(_common))
    assert len(calls) == 16, sorted(calls)
    for name, call in calls.items():
        assert ("poll=cross_pair_poll" in call) is (name in _CROSS_PAIR_BARS), (name, call)
    assert "cross_pair_poll: bool = True" in inspect.getsource(_common.make_d512_2x2_bars)
    # MBarrier(poll=True) dispatches to the test_wait poll; the poll never parks (no try_wait anywhere in it), and it has
    # the two-phase shape: POLL_TIGHT_ITERS tight tests, then a plain TIMER nanosleep (NOT the event-sleep NANOSLEEP.SYNCS)
    # of POLL_SLEEP_NS between tests -- a tight loop on the MMA / TMA-LDG warp starved the compute warps that share its SMSP
    # (the d512 backward measured 2.2x slower with the tight form).
    assert (_barrier.POLL_TIGHT_ITERS, _barrier.POLL_SLEEP_NS) == (32, 128)
    default_ptx = _barrier.poll_ptx(_barrier.POLL_TIGHT_ITERS, _barrier.POLL_SLEEP_NS)
    assert "mbarrier.test_wait.parity.acquire.cta" in default_ptx and "try_wait" not in default_ptx
    assert "nanosleep.u32 128;" in default_ptx and "setp.lt.u32 P1, n, 32;" in default_ptx
    tight_ptx = _barrier.poll_ptx(32, 0)  # the sm107 fork's pure tight loop
    assert "nanosleep" not in tight_ptx and "mbarrier.test_wait.parity.acquire.cta" in tight_ptx
    for bad in ((0, 128), (-1, 0)):
        with pytest.raises(ValueError, match="tight_iters"):
            _barrier.poll_ptx(*bad)
    # wait_poll's constexpr defaults and MBarrier's append-only shape fields equal the module constants (no rendering change).
    sig = inspect.signature(_barrier.wait_poll)
    assert (sig.parameters["tight_iters"].default, sig.parameters["sleep_ns"].default) == (32, 128)
    fields = _barrier.MBarrier.__dataclass_fields__
    assert (fields["poll_tight"].default, fields["poll_sleep_ns"].default) == (32, 128) and list(fields)[-4:] == [
        "poll",
        "poll_tight",
        "poll_sleep_ns",
        "stage_idx",
    ]
    wait_src = inspect.getsource(_barrier.MBarrier.wait)
    assert "if cutlass.const_expr(self.poll):" in wait_src and "tight_iters=self.poll_tight, sleep_ns=self.poll_sleep_ns" in wait_src
    src = open(os.path.join(_kernels_dir(), _KERNEL_FILE)).read()
    assert src.count("bars = make_d512_2x2_bars(") == 1 and "cross_pair_poll=POLL_CROSS_PAIR_WAITS)" in src
    assert 'bool(int(globals().get("FROST_D512_2X2_POLL_CROSS_PAIR_WAITS", 1)))' in src  # default ON (sm107 spelling)


# ------------------------------------------------------------------------------------------- host-only: byte identity

# Cubin md5 of sm100/prefill_d512_f16.py (bf16 dense, has_lse, sm_100a) RECORDED BEFORE TemplateParams.mma_2x2 existed
# (job c9d07061, lane_d512_fprop/md5_pins_before.log, nvidia-cutlass-dsl 4.7.0 + CUDA 13.3 ptxas).  Toolchain-specific:
# a different ptxas renders a different cubin; the WITH-vs-WITHOUT-field equality below is the toolchain-independent half.
_ROLE_SPLIT_PRE_FIELD_MD5 = {"dense": "0a5e0d71bc3714475394cd70cdc9a58d", "causal": "0138a912f5390f06c5ff031df9739c46"}
_ROLE_SPLIT_SPECS = {"dense": {}, "causal": {"window_right": 0, "sched_policy": 2}}

_ROLE_SPLIT_PROBE = sass_probe_source("""
    params = TemplateParams(dtype_qkv=2, dtype_o=2, **params_kw)
    mod = _load_sm100_kernel_module((512, 512), params, fp8=False, pertensor=False, rubin=False)
    print("IS_2X2", int(mod.__file__.endswith("_2x2.py")))
    mod.compile(d_qk=512, d_v=512, has_lse=True, lse_kind="dense")
    """)


@pytest.mark.L0
@pytest.mark.parametrize("spec", sorted(_ROLE_SPLIT_SPECS))
def test_role_split_rendering_is_byte_identical(tmp_path, spec):
    """The appended field changes no cubin byte of the 4x1 kernel: TemplateParams() and TemplateParams(mma_2x2=False)
    render the same cubin, equal to the pre-field pin on the pinning toolchain."""
    without = run_sass_probe(tmp_path, probe_src=_ROLE_SPLIT_PROBE, arch="sm_100a", params=_ROLE_SPLIT_SPECS[spec], tag=f"rs_{spec}_nofield")
    explicit = run_sass_probe(
        tmp_path, probe_src=_ROLE_SPLIT_PROBE, arch="sm_100a", params={"mma_2x2": False, **_ROLE_SPLIT_SPECS[spec]}, tag=f"rs_{spec}_false"
    )
    assert without.expect["IS_2X2"] == 0 and explicit.expect["IS_2X2"] == 0
    assert without.cubin_md5 == explicit.cubin_md5, "mma_2x2=False must render the role-split kernel byte-identically"
    if without.cubin_md5 != _ROLE_SPLIT_PRE_FIELD_MD5[spec]:
        pytest.skip(
            f"cubin md5 {without.cubin_md5} differs from the pin recorded on the pinning toolchain (ptxas / DSL changed); the with/without-field identity above held"
        )


# ------------------------------------------------------------------------------------------- host-only: 2x2 SASS pins

_2X2_SASS_COUNTS = {
    "STL": ("STL",),
    "LDL": ("LDL",),
    "UTMASTG": ("UTMASTG",),
    "UTCHMMA": ("UTCHMMA",),
    "CGAERRBAR": ("CGAERRBAR",),
    "MEMBAR_GPU": ("MEMBAR.ALL.GPU",),
    "SYNCS_ARRIVE": (" SYNCS.ARRIVE",),
}
_2X2_PROBE = sass_probe_source(
    """
    params = TemplateParams(dtype_qkv=2, dtype_o=2, mma_2x2=True, **params_kw)
    mod = _load_sm100_kernel_module((512, 512), params, fp8=False, pertensor=False, rubin=False)
    print("IS_2X2", int(mod.__file__.endswith("_2x2.py")))
    print("CGA_M", int(mod.CFG.CGA_M))
    print("SMEM_TOTAL", int(mod._SMEM["total"]))
    mod.compile(d_qk=512, d_v=512, has_lse=True, lse_kind="dense")
    """,
    counts=_2X2_SASS_COUNTS,
)
# MEASURED on this branch's toolchain (nvidia-cutlass-dsl 4.7.0 + CUDA 13.3 ptxas, sm_100a); STL / LDL are BOUNDS
# (frost_test_utils.SPILL_TOLERANCE), the rest exact structure: 8 UTMASTG = the streamed O store's eight subtiles,
# 0 CGAERRBAR / MEMBAR.ALL.GPU = no cluster-scope release on any per-iteration path.
_2X2_SPECS = {"dense": {}, "causal": {"window_right": 0}}
_2X2_PINS = {"dense": {"STL": 0, "LDL": 0}, "causal": {"STL": 0, "LDL": 0}}


@pytest.mark.L0
@pytest.mark.parametrize("spec", sorted(_2X2_SPECS))
def test_2x2_sass_pins(tmp_path, spec):
    probe = run_sass_probe(tmp_path, probe_src=_2X2_PROBE, arch="sm_100a", params=_2X2_SPECS[spec], tag=f"d512_2x2_{spec}")
    assert probe.expect["IS_2X2"] == 1 and probe.expect["CGA_M"] == 4
    assert probe.expect["SMEM_TOTAL"] == 232312
    assert probe.stats["UTMASTG"] == 8, probe.stats
    assert probe.stats["CGAERRBAR"] == 0 and probe.stats["MEMBAR_GPU"] == 0, probe.stats
    assert probe.stats["UTCHMMA"] > 0
    assert_no_new_spills(probe.stats, _2X2_PINS[spec], f"[{spec}] ")


# ------------------------------------------------------------------------------------------------------ GPU: fixtures


@pytest.fixture
def two_by_two(monkeypatch):
    """Flip the call-time twin so every d512 half plan built in the test lowers onto the 2x2 kernel."""
    from cudnn.sdpa.fwd import api_dsl

    monkeypatch.setattr(api_dsl, "D512_2X2", True)
    yield


def _served_template(graph):
    """The template file stem of the plan the graph built (``kernel_template``)."""
    eng = graph.selected_engine
    assert eng is not None, "a FROST plan must be selected"
    api = getattr(getattr(eng, "_compiled", None), "kernel_template", None)
    if api is not None:
        return api
    for p in (
        getattr(graph, "_compiled_plans", {}).values() if isinstance(getattr(graph, "_compiled_plans", None), dict) else getattr(graph, "_compiled_plans", [])
    ):
        c = getattr(p, "_compiled", None)
        if c is not None and hasattr(c, "kernel_template"):
            return c.kernel_template
    raise AssertionError("could not read kernel_template off the built plan")


def _run_graph(q, k, v, *, scale, dtype, sdpa_kwargs, seq_len_kv=None, seq_len_q=None, sink=None, pack_gqa=None, return_stats=False, expect_template=_TEMPLATE):
    """_run_dsl_graph plus the template assertion (which kernel served the plan)."""
    import cudnn

    b, h_q, s_q, _ = q.shape
    d_v = v.shape[-1]
    o_gpu = torch.empty(b, s_q, h_q, d_v, device="cuda", dtype=dtype).transpose(1, 2)
    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tq, tk, tv = g.tensor_like(q), g.tensor_like(k), g.tensor_like(v)
    kw = dict(name="sdpa", q=tq, k=tk, v=tv, generate_stats=return_stats, attn_scale=scale)
    vp = {tq: q, tk: k, tv: v}
    if seq_len_kv is not None:
        slk = g.tensor_like(seq_len_kv)
        kw["seq_len_kv"] = slk
        kw["use_padding_mask"] = True
        vp[slk] = seq_len_kv
        slq_t = seq_len_q if seq_len_q is not None else torch.full((b, 1, 1, 1), s_q, dtype=torch.int32, device="cuda")
        slq = g.tensor_like(slq_t)
        kw["seq_len_q"] = slq
        vp[slq] = slq_t
    if sink is not None:
        st = g.tensor_like(sink)
        kw["sink_token"] = st
        vp[st] = sink
    kw.update(sdpa_kwargs)
    o, stats = g.sdpa(**kw)
    o.set_output(True).set_dim(o_gpu.shape).set_stride(o_gpu.stride())
    stats_gpu = None
    if return_stats:
        stats_gpu = _dsl.make_dense_stats(b, h_q, s_q, "contiguous")
        stats.set_output(True).set_dim(stats_gpu.shape).set_stride(stats_gpu.stride()).set_data_type(cudnn.data_type.FLOAT)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    # split_kv pinned to 1: the heuristics split small B*H causal graphs over KV, and the twin (correctly) keeps
    # split plans on the role-split kernel in phase 1 -- this file tests the 2x2 kernel, not the split heuristic.
    _dsl._select_engine(g, engine_name(arch=_dsl._ARCH), pack_gqa=pack_gqa, split_kv=1)
    g.check_support()
    g.build_plans()
    assert _served_template(g) == expect_template
    vp[o] = o_gpu
    if stats_gpu is not None:
        vp[stats] = stats_gpu
    g.execute(vp, torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8))
    torch.cuda.synchronize()
    return (o_gpu, stats_gpu) if return_stats else o_gpu


_DTYPES = [torch.float16, torch.bfloat16]
_DTYPE_IDS = ["fp16", "bf16"]
_TOL = dict(atol=5e-2, rtol=3e-2)

# ------------------------------------------------------------------------------------------------------ GPU: graph API


@requires_blackwell
@_pre_rubin
@pytest.mark.L0
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("is_causal", [False, True], ids=["dense", "causal"])
@torch_fork_set_rng(seed=0)
def test_two_by_two_graph_api(two_by_two, dtype, is_causal):
    """The existing dsv4_d512 graph-API case served by the 2x2 kernel (b=2 h=8 s=256 -> one 4-CTA cluster per head)."""
    _dsl._require_dsl()
    b, h, s = 2, 8, 256
    scale = 1.0 / math.sqrt(_D)
    q, k, v = (_dsl._bhsd(b, h, s, _D, dtype) for _ in range(3))
    o, stats = _run_graph(q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=dict(use_causal_mask=is_causal), return_stats=True)
    o_ref, lse_ref = _dsl._ref_sdpa_full(q, k, v, scale=scale, is_causal=is_causal, return_stats=True)
    torch.testing.assert_close(o, o_ref, **_TOL)
    torch.testing.assert_close(stats.squeeze(-1), lse_ref, **_TOL)


@requires_blackwell
@_pre_rubin
@pytest.mark.L0
@torch_fork_set_rng(seed=1)
def test_two_by_two_role_split_untouched_when_twin_off(monkeypatch):
    """With the twin switched OFF (`api_dsl.D512_2X2 = False`; True is the default since 2026-10-06) the same graph keeps
    the role-split kernel -- the A/B control."""
    from cudnn.sdpa.fwd import api_dsl

    monkeypatch.setattr(api_dsl, "D512_2X2", False)
    _dsl._require_dsl()
    b, h, s = 1, 4, 256
    scale = 1.0 / math.sqrt(_D)
    q, k, v = (_dsl._bhsd(b, h, s, _D, torch.bfloat16) for _ in range(3))
    o = _run_graph(q, k, v, scale=scale, dtype=torch.bfloat16, sdpa_kwargs=dict(use_causal_mask=True), expect_template=_ROLE_SPLIT_TEMPLATE)
    torch.testing.assert_close(o, _dsl._ref_sdpa_full(q, k, v, scale=scale, is_causal=True), **_TOL)


@requires_blackwell
@_pre_rubin
@pytest.mark.L0
@pytest.mark.parametrize("mask", ["causal_br", "swa", "band", "band_br", "swa_br", "band_swa", "padded"])
@torch_fork_set_rng(seed=2)
def test_two_by_two_mask_family(two_by_two, mask):
    """The causal-family masks + KV padding on the 2x2 kernel (bottom-right, sliding window, right band)."""
    _dsl._require_dsl()
    dtype = torch.bfloat16
    b, h = 2, 4
    s_q, s_kv = (128, 256) if mask in ("causal_br", "band_br", "swa_br") else (256, 256)
    scale = 1.0 / math.sqrt(_D)
    q = _dsl._bhsd(b, h, s_q, _D, dtype)
    k = _dsl._bhsd(b, h, s_kv, _D, dtype)
    v = _dsl._bhsd(b, h, s_kv, _D, dtype)
    seq_len_kv = None
    if mask == "padded":
        seq_len_kv = torch.tensor([s_kv - 76, s_kv - 16], dtype=torch.int32, device="cuda").view(b, 1, 1, 1)
        graph_kw, ref_kw = {}, dict(seq_kv_lens=seq_len_kv.flatten())
    else:
        graph_kw, ref_kw = _dsl._mask_graph_kwargs(mask), _dsl._mask_ref_kwargs(mask)
    o = _run_graph(q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=graph_kw, seq_len_kv=seq_len_kv)
    torch.testing.assert_close(o, _dsl._ref_sdpa_full(q, k, v, scale=scale, **ref_kw), **_TOL)


@requires_blackwell
@_pre_rubin
@pytest.mark.L0
@torch_fork_set_rng(seed=3)
def test_two_by_two_causal_512_cluster_union_bounds(two_by_two):
    """Causal S_q = S_kv = 512: one 4-CTA cluster covers the whole triangle, so the two pairs' natural causal
    ranges differ (rows 0..127 vs 128..255) -- the cluster-UNION bounds must keep them on the identical KV range
    (anything else deadlocks the shared k/v_empty ring) and the per-cell mask must do the trimming."""
    _dsl._require_dsl()
    dtype = torch.float16
    b, h, s = 1, 2, 512
    scale = 1.0 / math.sqrt(_D)
    q, k, v = (_dsl._bhsd(b, h, s, _D, dtype) for _ in range(3))
    o, stats = _run_graph(q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=dict(use_causal_mask=True), return_stats=True)
    o_ref, lse_ref = _dsl._ref_sdpa_full(q, k, v, scale=scale, is_causal=True, return_stats=True)
    torch.testing.assert_close(o, o_ref, **_TOL)
    torch.testing.assert_close(stats.squeeze(-1), lse_ref, **_TOL)


@requires_blackwell
@_pre_rubin
@pytest.mark.L0
@pytest.mark.parametrize("s_q,s_kv", [(200, 333), (65, 129), (4096 + 64, 4096 + 64)], ids=["200x333", "65x129", "4160x4160"])
@torch_fork_set_rng(seed=4)
def test_two_by_two_non_tile_multiple_seqlens(two_by_two, s_q, s_kv):
    """Seqlens that are not multiples of 64 / 128: the 64-row Q boxes zero-fill, the KV tail is padded-masked."""
    _dsl._require_dsl()
    dtype = torch.bfloat16
    b, h = 1, 2
    scale = 1.0 / math.sqrt(_D)
    q = _dsl._bhsd(b, h, s_q, _D, dtype)
    k = _dsl._bhsd(b, h, s_kv, _D, dtype)
    v = _dsl._bhsd(b, h, s_kv, _D, dtype)
    seq_len_kv = torch.full((b, 1, 1, 1), s_kv, dtype=torch.int32, device="cuda")
    o = _run_graph(q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=dict(use_causal_mask=True), seq_len_kv=seq_len_kv)
    torch.testing.assert_close(o, _dsl._ref_sdpa_full(q, k, v, scale=scale, is_causal=True, seq_kv_lens=seq_len_kv.flatten()), **_TOL)


@requires_blackwell
@_pre_rubin
@pytest.mark.L0
@torch_fork_set_rng(seed=5)
def test_two_by_two_swa_empty_tiles_and_q_trim(two_by_two):
    """Sliding window past the padded KV tail (empty KV loops: the empty-mainloop protocol and the O u V alias
    phase bookkeeping) plus the dense padded-Q trim (rows >= seq_len_q[b] -> O = 0, LSE = -inf)."""
    _dsl._require_dsl()
    dtype = torch.float16
    b, h, s_q, s_kv, W = 2, 2, 512, 512, 100
    scale = 1.0 / math.sqrt(_D)
    q, k, v = (_dsl._bhsd(b, h, s_q, _D, dtype) for _ in range(3))
    seq_len_kv = torch.tensor([160, 512], dtype=torch.int32, device="cuda").view(b, 1, 1, 1)
    seq_len_q = torch.tensor([300, 450], dtype=torch.int32, device="cuda").view(b, 1, 1, 1)
    o, stats = _run_graph(
        q,
        k,
        v,
        scale=scale,
        dtype=dtype,
        sdpa_kwargs=dict(use_causal_mask=True, sliding_window_length=W + 1),
        seq_len_kv=seq_len_kv,
        seq_len_q=seq_len_q,
        return_stats=True,
    )
    o_ref, lse_ref = _dsl._ref_sdpa_full(
        q, k, v, scale=scale, is_causal=True, swa_window=W, seq_q_lens=seq_len_q.flatten(), seq_kv_lens=seq_len_kv.flatten(), return_stats=True
    )
    torch.testing.assert_close(o, o_ref.nan_to_num(0.0), **_TOL)
    finite = torch.isfinite(lse_ref)
    torch.testing.assert_close(stats.squeeze(-1)[finite], lse_ref[finite], **_TOL)
    assert torch.isneginf(stats.squeeze(-1)[~finite]).all(), "trimmed / windowed-out rows must carry LSE = -inf"


@requires_blackwell
@_pre_rubin
@pytest.mark.L0
@pytest.mark.parametrize("stats_use_log2", [False, True], ids=["ln", "log2"])
@torch_fork_set_rng(seed=6)
def test_two_by_two_sink_and_stats(two_by_two, stats_use_log2):
    _dsl._require_dsl()
    dtype = torch.bfloat16
    b, h, s = 1, 4, 384
    scale = 1.0 / math.sqrt(_D)
    q, k, v = (_dsl._bhsd(b, h, s, _D, dtype) for _ in range(3))
    sink = torch.randn(1, h, 1, 1, device="cuda", dtype=torch.float32)
    o, stats = _run_graph(
        q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=dict(use_causal_mask=True, stats_use_log2=stats_use_log2), sink=sink, return_stats=True
    )
    o_ref, lse_ref = _dsl._ref_sdpa_full(q, k, v, scale=scale, is_causal=True, sinks=sink.flatten(), return_stats=True)
    if stats_use_log2:
        lse_ref = lse_ref * math.log2(math.e)
    torch.testing.assert_close(o, o_ref, **_TOL)
    torch.testing.assert_close(stats.squeeze(-1), lse_ref, **_TOL)


@requires_blackwell
@_pre_rubin
@pytest.mark.L0
@pytest.mark.parametrize("h_q,h_kv,expect", [(8, 2, _TEMPLATE), (64, 1, _TEMPLATE), (128, 1, _ROLE_SPLIT_TEMPLATE)], ids=["g4", "g64", "g128_role_split"])
@torch_fork_set_rng(seed=7)
def test_two_by_two_pack_gqa(two_by_two, h_q, h_kv, expect):
    """PackGQA on the 64-row tile: G=4 and G=64 pack whole groups onto the 2x2 kernel; G=128 does not divide 64 and
    stays on the role-split kernel (the twin declines it)."""
    _dsl._require_dsl()
    dtype = torch.bfloat16
    b, s_q, s_kv = 2, 40, 256
    scale = 1.0 / math.sqrt(_D)
    q = _dsl._bhsd(b, h_q, s_q, _D, dtype)
    k = _dsl._bhsd(b, h_kv, s_kv, _D, dtype)
    v = _dsl._bhsd(b, h_kv, s_kv, _D, dtype)
    o = _run_graph(q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=dict(use_causal_mask=True), pack_gqa=True, expect_template=expect)
    torch.testing.assert_close(o, _dsl._ref_sdpa_full(q, k, v, scale=scale, is_causal=True), **_TOL)


@requires_blackwell
@_pre_rubin
@pytest.mark.L0
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@torch_fork_set_rng(seed=8)
def test_two_by_two_thd(two_by_two, dtype):
    """Packed THD (two sequences of unequal length, per-sequence causal) through the persistent scheduler with
    256-row units; the sentinel outside the packed region must come back untouched."""
    _dsl._require_dsl()
    H = 4
    seq_lens = [333, 150]
    cu = [0]
    for s in seq_lens:
        cu.append(cu[-1] + s)
    T = cu[-1]
    scale = 1.0 / math.sqrt(_D)
    q_pk, k_pk, v_pk = (torch.randn(T, H, _D, device="cuda", dtype=dtype) for _ in range(3))
    served = []
    o_stor = _dsl._run_dsl_thd_graph(
        q_pk,
        k_pk,
        v_pk,
        cu,
        cu,
        seq_lens,
        seq_lens,
        scale=scale,
        dtype=dtype,
        H_q=H,
        H_kv=H,
        d=_D,
        mask="causal",
        on_graph=lambda g: served.append(_served_template(g)),
    )
    assert served == [_TEMPLATE], f"the THD plan must be served by the 2x2 kernel, got {served}"
    o_pk = o_stor[: T * H * _D].view(T, H, _D)
    for bi, s in enumerate(seq_lens):
        qs, ks, vs = (t[cu[bi] : cu[bi + 1]].permute(1, 0, 2).unsqueeze(0) for t in (q_pk, k_pk, v_pk))
        o_ref = _dsl._ref_sdpa_full(qs, ks, vs, scale=scale, is_causal=True)[0].permute(1, 0, 2)
        torch.testing.assert_close(o_pk[cu[bi] : cu[bi + 1]], o_ref, **_TOL)
    assert (o_stor[T * H * _D :] == _dsl._THD_SENTINEL).all()


def _ref_fp64(q, k, v, *, scale, is_causal):
    """fp64 BHSD reference (O in fp64 -> caller's dtype compare, LSE fp64) for the envelope cells, where the TMA zero-fill /
    clip of the 64-col boxes is what is under test and the fp32 reference's own rounding should not be in the budget."""
    s = torch.matmul(q.double(), k.double().transpose(-1, -2)) * scale
    if is_causal:
        s_q, s_kv = q.shape[2], k.shape[2]
        i = torch.arange(s_q, device=q.device).view(s_q, 1)
        j = torch.arange(s_kv, device=q.device).view(1, s_kv)
        s = s.masked_fill(j > i, float("-inf"))
    return torch.matmul(torch.softmax(s, dim=-1), v.double()), torch.logsumexp(s, dim=-1)


@requires_blackwell
@_pre_rubin
@pytest.mark.L0
@pytest.mark.parametrize("d_qk,d_v", [(384, 384), (448, 320)], ids=["d384", "d448_d320"])
@pytest.mark.parametrize("is_causal", [False, True], ids=["dense", "causal"])
@torch_fork_set_rng(seed=13)
def test_two_by_two_envelope_head_dims(two_by_two, d_qk, d_v, is_causal):
    """Head dims in (256, 512] ride the (512, 512) flavor (TMA zero-fill of the Q/K/V 64-col boxes, clip of the O boxes),
    so under the twin they land on the 2x2 kernel with d_qk < TILE_K and d_v < TILE_O: dense + causal with Stats vs fp64.
    b=1 h=2 s=512 -> two clusters per head, four KV iterations."""
    _dsl._require_dsl()
    dtype = torch.bfloat16
    b, h, s = 1, 2, 512
    scale = 1.0 / math.sqrt(d_qk)
    q = _dsl._bhsd(b, h, s, d_qk, dtype)
    k = _dsl._bhsd(b, h, s, d_qk, dtype)
    v = _dsl._bhsd(b, h, s, d_v, dtype)
    o, stats = _run_graph(q, k, v, scale=scale, dtype=dtype, sdpa_kwargs=dict(use_causal_mask=is_causal), return_stats=True)
    o_ref, lse_ref = _ref_fp64(q, k, v, scale=scale, is_causal=is_causal)
    assert o.shape[-1] == d_v and not torch.isnan(o).any()
    torch.testing.assert_close(o.double(), o_ref, **_TOL)
    torch.testing.assert_close(stats.squeeze(-1).double(), lse_ref, **_TOL)


@requires_blackwell
@_pre_rubin
@pytest.mark.L0
def test_twin_declines_split_and_g128(two_by_two):
    """The twin's domain, at the ADAPTER: with the call-time twin on, a split-KV plan (`split_kv > 1`) keeps `mma_2x2=False`
    in its record (the role split serves it; the twin's split arm is not validated) while the unsplit control routes to the
    twin; PackGQA G=128 is refused by the config.  `SdpaFwdDslSm100.template_params()` is the rule under test."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams, make_cfg_d512

    _dsl._require_dsl()
    dtype = torch.bfloat16
    b, h, s = 1, 2, 512
    q, k, v, o = (_dsl._bhsd(b, h, s, _D, dtype) for _ in range(4))
    lse = torch.empty(b, h, s, dtype=torch.float32, device="cuda")
    records = {}
    for split in (2, 1):
        api = SdpaFwdDslSm100(sample_q=q, sample_k=k, sample_v=v, sample_o=o, sample_lse=lse, scale_softmax=1.0 / math.sqrt(_D), split_kv=split)
        assert api.check_support()
        records[split] = api.template_params()
    assert records[2].mma_2x2 is False and records[2].split_kv == 2, records[2]
    assert records[1].mma_2x2 is True and records[1].split_kv == 1, records[1]
    # the config's own statement of the same domain
    assert make_cfg_d512(TemplateParams(mma_2x2=True, split_kv=4))[0].SPLIT_KV == 4
    with pytest.raises(ValueError):
        make_cfg_d512(TemplateParams(mma_2x2=True, pack_gqa=True, qh_per_kh=128))


@requires_blackwell
@pytest.mark.L0
def test_two_by_two_default_plan_binds_natively(monkeypatch):
    """The default d512 half plan is the twin AND carries the native dense binder; the role-split arm keeps its own.
    `_dense_spec.native` is what graph.execute and the standalone path dispatch on -- None is the Python observation path,
    correct but not the shipped host path.  Runs on both arch lines (cc 10.7 serves the same template name from sm107/)."""
    from cudnn.sdpa.fwd import api_dsl
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    _dsl._require_dsl()
    dtype = torch.bfloat16
    b, h, s = 1, 2, 512
    q, k, v, o = (_dsl._bhsd(b, h, s, _D, dtype) for _ in range(4))
    lse = torch.empty(b, h, s, dtype=torch.float32, device="cuda")
    for arm, expect in ((True, _TEMPLATE), (False, _ROLE_SPLIT_TEMPLATE)):
        monkeypatch.setattr(api_dsl, "D512_2X2", arm)
        api = SdpaFwdDslSm100(sample_q=q, sample_k=k, sample_v=v, sample_o=o, sample_lse=lse, scale_softmax=1.0 / math.sqrt(_D))
        assert api.check_support()
        api.compile()
        assert api.kernel_template == expect, (arm, api.kernel_template)
        assert api._dense_spec.native is not None, (arm, expect)


# ------------------------------------------------------------------------------------------- GPU: direct template cells


def _params_digest(params) -> str:
    """The record's part of a hand-set ``FROST_SOURCE_DIGEST``.  The compiled-plan cache keys a ``compile()`` on the module
    digest + the call's arguments ONLY -- the mask, dtype and every other TemplateParams field live in the module -- so two
    loads of one tag whose records differ MUST carry different digests, or the cache serves the first load's artifact to
    the second (a causal load handed the dense kernel: RED on this file's multi-tile and pair-skew causal cells, dense
    attention under a causal reference, whenever the dense cell compiled first)."""
    import hashlib

    return hashlib.sha1(repr(params).encode()).hexdigest()[:12]


def _load_2x2(params, cga_m, tag, *, stg_delay_us=0):
    """Exec the template the way the loader does, with the CGA_M arm (and, for the alias-gate detector, the test-only
    DEBUG_STG_DELAY_US skew lever) injected next to the params.  The digest names the record too (``_params_digest``)."""
    path = os.path.join(_kernels_dir(), _KERNEL_FILE)
    spec = importlib.util.spec_from_file_location(f"cudnn.frost._templates.test2x2_{tag}_{cga_m}_{stg_delay_us}", path)
    mod = importlib.util.module_from_spec(spec)
    setattr(mod, "FROST_TEMPLATE_PARAMS", params)
    setattr(mod, "FROST_D512_2X2_CGA_M", cga_m)
    if stg_delay_us:
        setattr(mod, "FROST_D512_2X2_DEBUG_STG_DELAY_US", int(stg_delay_us))
    setattr(mod, "FROST_SOURCE_DIGEST", f"test2x2_{tag}_{cga_m}_{stg_delay_us}_{_params_digest(params)}")
    spec.loader.exec_module(mod)
    return mod


def _direct_launch(mod, q, k, v, scale, *, causal, seq_kv=None):
    import cutlass
    import cuda.bindings.driver as cuda_driver

    B, SQ, H, _ = q.shape
    KH, SKV = k.shape[2], k.shape[1]
    fn = mod.compile(d_qk=_D, d_v=_D, has_lse=True, lse_kind="dense")
    o = torch.full((B, SQ, H, _D), float("nan"), device="cuda", dtype=q.dtype)
    lse = torch.full((B, H, SQ), float("nan"), device="cuda", dtype=torch.float32)
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    launch_f16(
        fn,
        q,
        k,
        v,
        o,
        lse,
        torch.zeros(H, dtype=torch.float32, device="cuda"),
        seq_kv if seq_kv is not None else torch.zeros(B, dtype=torch.int32, device="cuda"),
        torch.zeros(1, dtype=torch.int64, device="cuda"),
        (B, H, KH, SQ, SKV, 0),
        cutlass.Float32(scale * math.log2(math.e)),
        cutlass.Int32(0),
        0,
        stream=stream,
        host=mod._host,
    )
    torch.cuda.synchronize()
    return o, lse


def _ref_bshd(q, k, v, scale, causal):
    rep = q.shape[2] // k.shape[2]
    qf, kf, vf = q.float(), k.float().repeat_interleave(rep, dim=2), v.float().repeat_interleave(rep, dim=2)
    s = torch.einsum("bqhd,bkhd->bhqk", qf, kf) * scale
    if causal:
        i = torch.arange(q.shape[1], device=q.device).view(-1, 1)
        j = torch.arange(k.shape[1], device=q.device).view(1, -1)
        s = s.masked_fill((j > i).view(1, 1, q.shape[1], k.shape[1]), float("-inf"))
    return torch.einsum("bhqk,bkhd->bqhd", torch.softmax(s, dim=-1), vf), torch.logsumexp(s, dim=-1)


@requires_blackwell
@_pre_rubin
@pytest.mark.L0
@torch_fork_set_rng(seed=9)
def test_two_by_two_cga2_vs_cga4_bitwise():
    """The CGA_M=2 bring-up arm (one pair, own-bit loads) and the CGA_M=4 twin-multicast arm run the same
    per-pair arithmetic: O and LSE must be BITWISE identical (the multicast only changes who issues the bytes)."""
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    dtype = torch.float16
    B, H, KH, SQ, SKV = 1, 4, 4, 512, 1024
    scale = 1.0 / math.sqrt(_D)
    q = torch.randn(B, SQ, H, _D, device="cuda", dtype=dtype)
    k = torch.randn(B, SKV, KH, _D, device="cuda", dtype=dtype)
    v = torch.randn(B, SKV, KH, _D, device="cuda", dtype=dtype)
    params = TemplateParams(mma_2x2=True, dtype_qkv=3, dtype_o=3, window_right=0)
    outs = {}
    for cga_m in (4, 2):
        mod = _load_2x2(params, cga_m, "twin")
        assert mod.CFG.CGA_M == cga_m and mod.KV_SHARE == cga_m // 2
        outs[cga_m] = _direct_launch(mod, q, k, v, scale, causal=True)
    assert torch.equal(outs[4][0], outs[2][0]) and torch.equal(outs[4][1], outs[2][1])
    o_ref, lse_ref = _ref_bshd(q, k, v, scale, True)
    torch.testing.assert_close(outs[4][0].float(), o_ref, **_TOL)
    torch.testing.assert_close(outs[4][1], lse_ref, **_TOL)


@requires_blackwell
@_pre_rubin
@pytest.mark.L0
@pytest.mark.parametrize("cell", ["skewed_halves", "rescale_storm"])
@torch_fork_set_rng(seed=10)
def test_two_by_two_directed_numerics(cell):
    """skewed_halves: every odd 64-key half of each 128-key tile is scaled by 2^10, so lanes r and r + 64 see
    row maxima ~1000x apart -- wrong without the row-max exchange (the half with the smaller max would exponentiate
    unnormalised).  rescale_storm: K tile t scaled by 2^t drives alpha != 1 on every iteration (the slow correction
    arm and the per-N-block credits on every step)."""
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    dtype = torch.bfloat16
    B, H, SQ, SKV = 1, 2, 256, 1024
    scale = 1.0 / math.sqrt(_D)
    q = torch.randn(B, SQ, H, _D, device="cuda", dtype=dtype)
    k = torch.randn(B, SKV, H, _D, device="cuda", dtype=dtype)
    v = torch.randn(B, SKV, H, _D, device="cuda", dtype=dtype)
    kf = k.float()
    if cell == "skewed_halves":
        kf = kf.view(B, SKV // 64, 64, H, _D)
        kf[:, 1::2] *= 2.0**10
        k = (kf.view(B, SKV, H, _D) / 2.0**5).to(dtype)
    else:
        kf = kf.view(B, SKV // 128, 128, H, _D)
        for t in range(SKV // 128):
            kf[:, t] *= 2.0 ** min(t, 12)
        k = (kf.view(B, SKV, H, _D) / 2.0**6).to(dtype)
    mod = _load_2x2(TemplateParams(mma_2x2=True, dtype_qkv=2, dtype_o=2), 4, "directed")
    o, lse = _direct_launch(mod, q, k, v, scale, causal=False)
    o_ref, lse_ref = _ref_bshd(q, k, v, scale, False)
    assert not torch.isnan(o).any()
    torch.testing.assert_close(o.float(), o_ref, **_TOL)
    torch.testing.assert_close(lse, lse_ref, **_TOL)


# ------------------------------------------------------------------------------- GPU: persistent multi-tile / alias-gate cells


def _bad_blocks(o, o_ref):
    """(batch, 64-row q block, head, 64-col d_v block) cells with any element outside _TOL -- the signature of an SMEM slab
    corruption (one TMA subtile = 64 rows x 64 d_v).  Empty when the output is clean."""
    err = (o.float() - o_ref).abs()
    over = err > (_TOL["atol"] + _TOL["rtol"] * o_ref.abs())
    B, SQ, H, D = over.shape
    cells = over.view(B, SQ // 64, 64, H, D // 64, 64).amax(dim=5).amax(dim=2)
    return cells.nonzero().tolist()


def _assert_rows_close(o, lse, o_ref, lse_ref, what):
    bad = _bad_blocks(o, o_ref)
    assert not torch.isnan(o).any(), f"{what}: NaN in O"
    assert not bad, f"{what}: {len(bad)} corrupted (b, q_block64, h, d_v_block64) cells, first 16: {bad[:16]}"
    torch.testing.assert_close(o.float(), o_ref, **_TOL)
    torch.testing.assert_close(lse, lse_ref, **_TOL)


@requires_blackwell
@_pre_rubin
@pytest.mark.L0
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@torch_fork_set_rng(seed=11)
def test_two_by_two_persistent_multi_tile(causal):
    """Several Q tiles per CTA: B1 H8 S_q 4096 S_kv 8192 = 128 four-CTA clusters over the 34 co-resident ones -> ~4 tiles
    per CTA and ~90 tile boundaries per cluster, dense and causal, checked PER ROW.  Exercises every cross-tile protocol
    under natural pair skew: mb_q_empty, the pair-wide O u V alias gate (mb_o_empty), the stat-ring tile-end step, the
    sXchg cross-tile reuse.  Random N(0, 1) V makes any slab corruption visible against O, a softmax average (|O| < ~0.3)."""
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    dtype = torch.bfloat16
    B, H, SQ, SKV = 1, 8, 4096, 8192
    scale = 1.0 / math.sqrt(_D)
    q = torch.randn(B, SQ, H, _D, device="cuda", dtype=dtype)
    k = torch.randn(B, SKV, H, _D, device="cuda", dtype=dtype)
    v = torch.randn(B, SKV, H, _D, device="cuda", dtype=dtype)
    kw = dict(mma_2x2=True, dtype_qkv=2, dtype_o=2)
    if causal:
        kw["window_right"] = 0
    mod = _load_2x2(TemplateParams(**kw), 4, "multi")
    assert mod.KV_SHARE == 2 and mod.DEBUG_STG_DELAY_US == 0
    o, lse = _direct_launch(mod, q, k, v, scale, causal=causal)
    o_ref, lse_ref = _ref_bshd(q, k, v, scale, causal)
    _assert_rows_close(o, lse, o_ref, lse_ref, f"multi-tile {'causal' if causal else 'dense'}")


@requires_blackwell
@_pre_rubin
@pytest.mark.L0
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@torch_fork_set_rng(seed=12)
def test_two_by_two_twin_alias_gate_under_pair_skew(causal):
    """DETECTOR for the cross-CTA O u V alias race (review FATAL-1).  sO aliases the V ring, and under KV_SHARE=2 the
    TWIN CTA (cta ^ 2) multicasts its V(t+1) share into MY sVO -- so the gate before a tile's first V issue must cover
    BOTH twins' O(t) stores (mb_o_empty init 32 x KV_SHARE: own arrive + arrive_on_peer(cta ^ 2) from the TMA-STG warp).
    The test-only DEBUG_STG_DELAY_US lever holds PAIR 0's O store for ~300 us per tile after its first subtile is ready
    while pair 1 finishes, stores O(t) and issues V(t+1).  Under a per-CTA gate pair 1's share (V subtile 1 of each
    sub-chunk = sVO bytes [16K, 32K) and [48K, 64K) = O subtiles 2, 3, 6, 7 = d_v [128, 256) and [384, 512)) lands on
    pair 0's staged O(t) before its store reads it: RED on q rows 0..127 of every 256-row cluster block except each
    CTA's last tile.  B1 H4 S_q 4096 S_kv 2048 = 64 clusters over 34 resident -> 30 clusters run two tiles."""
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    dtype = torch.bfloat16
    B, H, SQ, SKV = 1, 4, 4096, 2048
    scale = 1.0 / math.sqrt(_D)
    q = torch.randn(B, SQ, H, _D, device="cuda", dtype=dtype)
    k = torch.randn(B, SKV, H, _D, device="cuda", dtype=dtype)
    v = torch.randn(B, SKV, H, _D, device="cuda", dtype=dtype)
    kw = dict(mma_2x2=True, dtype_qkv=2, dtype_o=2)
    if causal:
        kw["window_right"] = 0
    mod = _load_2x2(TemplateParams(**kw), 4, "skew", stg_delay_us=300)
    assert mod.KV_SHARE == 2 and mod.DEBUG_STG_DELAY_US == 300
    o, lse = _direct_launch(mod, q, k, v, scale, causal=causal)
    o_ref, lse_ref = _ref_bshd(q, k, v, scale, causal)
    _assert_rows_close(o, lse, o_ref, lse_ref, f"pair-skew {'causal' if causal else 'dense'}")


# ------------------------------------------------------------------- GPU: the time-slicing hang detector + negative control

# One child = one process = one CUDA context.  ``load`` launches the shipped role-split d512 forward back to back (the second
# context that makes the GPU time-slice); ``twin`` runs the 2x2 forward for N launches with a per-launch wall budget enforced
# by polling a CUDA event from Python (a wedged launch never returns, so the budget is the only way out) and exits 3 on a
# hang.  Levers (JSON) are the kernel's loader-style module globals (FROST_D512_2X2_POLL_CROSS_PAIR_WAITS=0 renders the
# pre-fix parking waits).  B=1 H=128 S=8192 dense; the values are irrelevant to the schedule.
_CONTENTION_CHILD = _textwrap.dedent(r"""
    import importlib.util, json, math, os, sys, time
    role, n, budget_s, levers, utils_dir = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), json.loads(sys.argv[4]), sys.argv[5]
    os.environ.setdefault("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", "1")
    sys.path.insert(0, utils_dir)
    import torch
    import cutlass
    import cuda.bindings.driver as cuda_driver
    from cudnn.sdpa.fwd import api_dsl
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    from frost_test_utils import launch_f16
    B, H, S, D = 1, 128, 8192, 512
    if role == "twin":
        path = os.path.join(os.path.dirname(os.path.abspath(api_dsl.__file__)), "kernels", "sm100/prefill_d512_f16_2x2.py")
        spec = importlib.util.spec_from_file_location("cudnn.frost._templates.contention_twin", path)
        mod = importlib.util.module_from_spec(spec)
        setattr(mod, "FROST_TEMPLATE_PARAMS", TemplateParams(mma_2x2=True, dtype_qkv=2, dtype_o=2))
        setattr(mod, "FROST_D512_2X2_CGA_M", 4)
        for key, val in levers.items():
            setattr(mod, key, val)
        setattr(mod, "FROST_SOURCE_DIGEST", "contention_twin_" + "_".join(f"{k}{v}" for k, v in sorted(levers.items())))
        spec.loader.exec_module(mod)
        assert mod.KV_SHARE == 2
    else:
        mod = _load_sm100_kernel_module((512, 512), TemplateParams(dtype_qkv=2, dtype_o=2), fp8=False, pertensor=False, rubin=False)
    fn = mod.compile(d_qk=D, d_v=D, has_lse=True, lse_kind="dense")
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    o = torch.empty(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(B, H, S, device="cuda", dtype=torch.float32)
    sinks, seq_kv, o_desc = torch.zeros(H, device="cuda"), torch.zeros(B, dtype=torch.int32, device="cuda"), torch.zeros(1, dtype=torch.int64, device="cuda")
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    scale_log2 = cutlass.Float32(math.log2(math.e) / math.sqrt(D))
    def run():
        launch_f16(fn, q, k, v, o, lse, sinks, seq_kv, o_desc, (B, H, H, S, S, 0), scale_log2, cutlass.Int32(0), 0, stream=stream, host=mod._host)
    print(f"[{role}] kernel {os.path.basename(mod.__file__)} poll={getattr(mod, 'POLL_CROSS_PAIR_WAITS', None)}", flush=True)
    hist = []  # per-launch seconds: a starved launch (seconds, then the wall) reads differently from a wedged one (ms, ms, never)
    for i in range(n):
        t0 = time.time()
        run()
        ev = torch.cuda.Event()
        ev.record()
        while not ev.query():
            time.sleep(0.005)
            if time.time() - t0 > budget_s:
                import subprocess
                try:
                    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=True).stdout
                except Exception as e:  # a missing or stuck nvidia-smi must not turn the 45 s hang exit into a 40 min one
                    apps = f"<nvidia-smi unavailable: {e!r}>"
                print(f"[{role}] HANG: launch {i + 1} exceeded {budget_s:.0f} s; history (s): " + " ".join(f"{h:.2f}" for h in hist[-30:]), flush=True)
                print(f"[{role}] compute processes at the hang (all GPUs; may include this child; GPU UUID, PID, process name), "
                      f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}: {apps.strip().splitlines()}", flush=True)
                os._exit(3)
        hist.append(time.time() - t0)
        if i == 0:
            print(f"[{role}] ready", flush=True)
    torch.cuda.synchronize()
    print(f"[{role}] done {n} launches", flush=True)
    """)


def _contention_run(tmp_path, *, twin_levers: dict, n_twin: int, budget_s: float, tag: str):
    """Start the role-split load child, wait until it is launching, run the twin child to completion (or its hang exit),
    stop the load.  Returns the twin's CompletedProcess; both logs are under ``tmp_path`` for the failure message."""
    import time

    import frost_test_utils

    utils_dir = os.path.dirname(os.path.abspath(frost_test_utils.__file__))
    script = tmp_path / "contention_child.py"
    script.write_text(_CONTENTION_CHILD)
    load_log = tmp_path / f"{tag}_load.log"
    with open(load_log, "w") as f:
        load = _subprocess.Popen([_sys.executable, str(script), "load", "1000000", "600", "{}", utils_dir], stdout=f, stderr=_subprocess.STDOUT, text=True)
    try:
        t0 = time.time()
        while "[load] ready" not in load_log.read_text():
            if load.poll() is not None:
                pytest.fail(f"the role-split load child exited early:\n{load_log.read_text()[-3000:]}")
            if time.time() - t0 > 600:
                pytest.fail(f"the role-split load child did not start launching within 600 s:\n{load_log.read_text()[-3000:]}")
            time.sleep(1.0)
        twin = _subprocess.run(
            [_sys.executable, str(script), "twin", str(n_twin), str(budget_s), _json.dumps(twin_levers), utils_dir],
            capture_output=True,
            text=True,
            timeout=1800,
        )
    finally:
        load.kill()
        load.wait()
    (tmp_path / f"{tag}_twin.log").write_text(twin.stdout + "\n--- stderr ---\n" + twin.stderr)
    return twin


@requires_blackwell
@_pre_rubin
@pytest.mark.L0
@pytest.mark.xdist_group(name="gpu_exclusive")
def test_two_by_two_survives_gpu_time_slicing(tmp_path):
    """THE runnable detector of the cross-pair lost-wake-up hang (python/cudnn/sdpa/AGENTS.md, 2x2 section): a second CUDA
    context launching the role-split d512 forward back to back makes the GPU time-slice; the 2x2 forward must then complete
    100 launches at B=1 H=128 S=8192 with every launch returning inside 30 s.  Its cross-pair barriers (k/v_full, k/v_empty,
    o_empty) are waited with the non-blocking test_wait poll; the backward's identical ring protocol hung within 2-74
    launches on every parking form (lane_d512_bprop/fix).  A hang exits the child with 3 after the budget (the stuck
    context dies with it), so the suite never wedges.  ~2 minutes (two compiles + 100 time-sliced launches)."""
    twin = _contention_run(tmp_path, twin_levers={}, n_twin=100, budget_s=30.0, tag="fixed")
    assert twin.returncode == 0 and "[twin] done 100 launches" in twin.stdout, f"rc={twin.returncode}\n{twin.stdout[-3000:]}\n{twin.stderr[-3000:]}"


@requires_blackwell
@_pre_rubin
@pytest.mark.L0
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.gpu_exclusive
@pytest.mark.xfail(
    strict=False, reason="the FORWARD's parking form has not reproduced the hang yet (0 in 1700+ time-sliced launches on 2026-10-01); the backward's did"
)
def test_two_by_two_parking_wait_form_under_time_slicing(tmp_path):
    """NEGATIVE CONTROL of the detector above: FROST_D512_2X2_POLL_CROSS_PAIR_WAITS=0 renders the pre-fix kernel (the parking
    try_wait on the cross-pair barriers too) and is EXPECTED to hang within 300 time-sliced launches, as the backward did
    (launch 2 / 23 / 25 / 74 in four of four runs).  On the forward the hang has NOT been reproduced yet (300/300 at S=8192,
    800/800 at S=16384, 600/600 causal S=16384 under two load contexts, lane_d512_fprop/fix/cont_red_park*.log), so the assertion is xfail(strict=False):
    an XPASS here is the forward reproducing the mechanism -- record its log.  It can wedge a kernel for the 30 s budget
    before the child dies, hence ``gpu_exclusive``: deselect it on a GPU other jobs share."""
    twin = _contention_run(tmp_path, twin_levers={"FROST_D512_2X2_POLL_CROSS_PAIR_WAITS": 0}, n_twin=300, budget_s=30.0, tag="parking")
    assert twin.returncode == 3 and "HANG" in twin.stdout, f"the parking wait form did not hang in 300 launches: rc={twin.returncode}\n{twin.stdout[-1500:]}"


# ---------------------------------------------------------------- GPU: tile-stats ring race (empty tile after a live one), dead rows


def _stair_inputs(*, B=4, H=8, KH=4, SQ=1024, SKV=512, step=12.0):
    """The sm107 lane's 'rising stair' (lane_d512_fprop/sm107/stair_probe.py `repro ... multi`): Q[..., 0] = +1 / 0 / -1 by
    32-row group, K[..., 0] = step * floor(key / 128) -> every KV tile raises the +1 rows' max by step/2 log2 (> RESCALE_THRESHOLD
    8 for step 12: the slow correction arm on every iteration), S is equal within a tile; random V."""
    dev, dt = "cuda", torch.bfloat16
    rows = torch.arange(SQ, device=dev)
    sign = torch.tensor([0.0, 1.0, -1.0], device=dev)[(rows // 32) % 3]
    q = torch.zeros(B, SQ, H, _D, device=dev, dtype=dt)
    q[..., 0] = sign.view(1, SQ, 1).to(dt)
    keys = torch.arange(SKV, device=dev)
    k = torch.zeros(B, SKV, KH, _D, device=dev, dtype=dt)
    k[..., 0] = (step * torch.div(keys, 128, rounding_mode="floor").float()).view(1, SKV, 1).to(dt)
    v = (torch.randn(B, SKV, KH, _D, device=dev) * 0.5).to(dt)
    return q, k, v


def _direct_launch_trim(mod, q, k, v, scale, *, seq_kv, q_lens):
    """_direct_launch with the dense padded-KV lengths and the q-trim lengths (seq_q_lens_addr) bound."""
    import cutlass
    import cuda.bindings.driver as cuda_driver

    B, SQ, H, _ = q.shape
    KH, SKV = k.shape[2], k.shape[1]
    fn = mod.compile(d_qk=_D, d_v=_D, has_lse=True, lse_kind="dense")
    o = torch.full((B, SQ, H, _D), float("nan"), device="cuda", dtype=q.dtype)
    lse = torch.full((B, H, SQ), float("nan"), device="cuda", dtype=torch.float32)
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    launch_f16(
        fn,
        q,
        k,
        v,
        o,
        lse,
        torch.zeros(H, dtype=torch.float32, device="cuda"),
        seq_kv,
        torch.zeros(1, dtype=torch.int64, device="cuda"),
        (B, H, KH, SQ, SKV, 0),
        cutlass.Float32(scale * math.log2(math.e)),
        cutlass.Int32(0),
        q_lens.data_ptr(),
        stream=stream,
        host=mod._host,
    )
    torch.cuda.synchronize()
    return o, lse


def _ref_trim(q, k, v, scale, q_lens):
    o_ref, lse_ref = _ref_bshd(q, k, v, scale, False)
    for bi in range(q.shape[0]):
        o_ref[bi, int(q_lens[bi]) :] = 0.0
        lse_ref[bi, :, int(q_lens[bi]) :] = float("-inf")
    return o_ref, lse_ref


@requires_blackwell
@_pre_rubin
@pytest.mark.L0
@torch_fork_set_rng(seed=14)
def test_two_by_two_stats_ring_slot_race_empty_after_live():
    """The tile stats (total_max_safe, final_sum) ride the 2-deep alpha ring (mb_stat_full / mb_stat_empty) but used to live in the
    FIXED TMEM columns 386 / 387.  A fixed column is protected by the ring only when the next writer waits the SAME slot's empty; an
    EMPTY tile (q-trim: q_len 0, or every row past q_len) is ONE ring step, so its stats store waited mb_stat_empty[1 - s] (the
    consume of the previous tile's last alpha), not mb_stat_empty[s] (the read of its stats): a softmax lane entering the empty tile
    while its correction lane was still in the slow arm (two O rescales between that alpha consume and the stats read) overwrote
    386/387 with (-, 0) and the correction published the row DEAD (O = 0, LSE = -inf).  Slot-indexed stats (386 + 2 * ring slot) put
    every ring step's payload in its own columns.  Found by the sm107 lane (stair_probe.py `repro padded qtrim qlens gqa sq1024
    multi`), reproduced on the B200 with this body: B4 H8 KH4 Sq1024 Skv512, q_lens (1024, 641, 0, 1024) -> the +1 rows of the FULL
    batches came back dead.  Only slow-arm rows lose the race, which is why random-data multi-tile cells never saw it."""
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    q, k, v = _stair_inputs()
    B, SQ, SKV = q.shape[0], q.shape[1], k.shape[1]
    scale = 0.5
    q_lens = torch.tensor([SQ, 641, 0, SQ], dtype=torch.int32, device="cuda")
    seq_kv = torch.full((B,), SKV, dtype=torch.int32, device="cuda")
    mod = _load_2x2(TemplateParams(mma_2x2=True, dtype_qkv=2, dtype_o=2, qh_per_kh=2, seq_kv_lens_present=True, seq_q_lens_present=True), 4, "stair")
    o, lse = _direct_launch_trim(mod, q, k, v, scale, seq_kv=seq_kv, q_lens=q_lens)
    o_ref, lse_ref = _ref_trim(q, k, v, scale, q_lens)
    dead = torch.isneginf(lse) & torch.isfinite(lse_ref)
    assert not dead.any(), f"{int(dead.sum())} live rows published DEAD (LSE = -inf); (b, h, row) first 12: {dead.nonzero()[:12].tolist()}"
    _assert_rows_close(o, lse, o_ref, lse_ref, "stair stats-ring")  # the LSE compare skips the -inf rows via the exact match of -inf


@requires_blackwell
@_pre_rubin
@pytest.mark.L0
@torch_fork_set_rng(seed=15)
def test_two_by_two_trimmed_rows_with_nan_inputs_store_zero():
    """Dead / trimmed rows are zeroed by a SELECT, never by `o * beta` with beta = 0: a q-trimmed row (row >= seq_len_q[b]) whose Q
    memory holds NaN (a poisoned padded tail) has S = P = O = NaN in TMEM, and NaN * 0 = NaN would reach the output.  The fp32-partials
    arm already selects; this pins the staged bf16 arm.  Rows >= q_len must come back exactly 0 with LSE = -inf, the live rows exact
    vs the reference."""
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    dtype = torch.bfloat16
    B, H, SQ, SKV = 2, 2, 256, 512
    scale = 1.0 / math.sqrt(_D)
    q = torch.randn(B, SQ, H, _D, device="cuda", dtype=dtype)
    k = torch.randn(B, SKV, H, _D, device="cuda", dtype=dtype)
    v = torch.randn(B, SKV, H, _D, device="cuda", dtype=dtype)
    q_lens = torch.tensor([200, 70], dtype=torch.int32, device="cuda")
    for bi in range(B):
        q[bi, int(q_lens[bi]) :] = float("nan")  # the trimmed rows' memory is poisoned
    seq_kv = torch.full((B,), SKV, dtype=torch.int32, device="cuda")
    mod = _load_2x2(TemplateParams(mma_2x2=True, dtype_qkv=2, dtype_o=2, seq_kv_lens_present=True, seq_q_lens_present=True), 4, "nantrim")
    o, lse = _direct_launch_trim(mod, q, k, v, scale, seq_kv=seq_kv, q_lens=q_lens)
    q_ref = q.clone()
    for bi in range(B):
        q_ref[bi, int(q_lens[bi]) :] = 0.0
    o_ref, lse_ref = _ref_trim(q_ref, k, v, scale, q_lens)
    for bi in range(B):
        ql = int(q_lens[bi])
        assert (o[bi, ql:] == 0).all(), f"batch {bi}: trimmed rows carry non-zero / NaN output (NaN count {int(torch.isnan(o[bi, ql:]).sum())})"
        assert torch.isneginf(lse[bi, :, ql:]).all()
    _assert_rows_close(o, lse, o_ref, lse_ref, "nan-trim")
