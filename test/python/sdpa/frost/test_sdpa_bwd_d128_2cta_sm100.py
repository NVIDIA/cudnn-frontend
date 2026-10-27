# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 grouped THD d128 backward on the cta_group::2 datapath (``bprop_d128_accum_grouped_2cta_f16.py``).

RED-first: the GPU rows below were committed with the pair body's first trace-compile, before it ran; every one is the grouped
suite's independent FP64 gate (``check_case``: per-sequence atol / rtol + cosine, storage canaries, exact zeros for empty attention),
dQ / dK / dV apart.  Route: the direct 2-CTA adapter ``api_dsl_sm100_d128_2cta.SdpaBwdDslSm100D128Cta2`` (the graph hook is deferred).

Rows (header section 12 of the kernel): uniform pairs, a GQA ratio of 8, a varlen ramp, non-causal, the grouped suite's degenerate
THD row (empty sequences, a sequence shorter than one pair, one token), a half-dead pair (seq_k % 256 in (0, 128]: CTA 1 of the
last pair owns no live key), all-empty attention, and the P14 row (n_q == 1 with more pairs than SMs, so a cluster claims a
second CLC item and the zero-trip protocol runs on most of them).  The host rendering test needs no GPU.
"""

import os
import pathlib
import re
import subprocess
import sys

import pytest
import torch

from d128_grouped_test_utils import adapter_arguments, check_case, make_case, make_workspace, poison
from d128_thd_test_utils import ROLES, SCALE
from frost_test_utils import requires_dsl
from test_sdpa_bwd_d128_accum_sm100 import _require_accumulation

pytestmark = [pytest.mark.L0, requires_dsl]

_ADAPTER_MODULE = "cudnn.sdpa.bwd.api_dsl_sm100_d128_2cta"
_HOST_MODULE = "cudnn.sdpa.bwd.kernels.sm100.prepared_d128_grouped_2cta_host"
_IMPLEMENTATION = "accumulation_grouped_thd_2cta"
_PAIR_CCS = ((10, 0), (10, 3))


def _adapter_class():
    """The 2-CTA adapter of THIS tree; a venv whose ``cudnn`` is another checkout has no such module (AGENTS.md: editable installs pin one checkout)."""
    import importlib

    import cudnn

    try:
        return importlib.import_module(_ADAPTER_MODULE).SdpaBwdDslSm100D128Cta2
    except ImportError as error:
        pytest.skip(f"{_ADAPTER_MODULE} is not importable from {cudnn.__file__}: this venv's cudnn is not the tree under test ({error})")


def _require_2cta():
    _require_accumulation()
    cc = torch.cuda.get_device_capability()
    if cc not in _PAIR_CCS:
        pytest.skip(f"the cta_group::2 d128 backward is served on SM100 / SM103 only; got {cc}")


def make_adapter_2cta(case, *, compile_plan=True, declaration=None):
    """The suite's adapter construction (``d128_grouped_test_utils.make_adapter``) on the 2-CTA subclass; ``declaration`` selects the
    storage declaration helper (the grouped one by default; the MHA utils' for an MHA case)."""
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
    if not compile_plan:
        return api
    assert api.check_support()
    api.compile()
    assert api._prepared.native is not None
    assert api._prepared.artifact.implementation == _IMPLEMENTATION
    return api


def _run_case(case, repeats=2):
    api = make_adapter_2cta(case)
    workspace = make_workspace(api.scratch_workspace_bytes())
    for _ in range(repeats):
        poison(case, workspace)
        api.execute(**adapter_arguments(case), workspace=workspace.tensor)
        torch.cuda.synchronize()
        check_case(case, workspace)
    return api, workspace


# ------------------------------------------------------------------------------------------------------------ host (no GPU)
def test_2cta_config_is_the_truth():
    """Every body constant is a derived, asserted ``CFG`` field (G10 config-as-truth); the kernel's map set follows it."""
    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta as cfgmod

    c = cfgmod.CFG
    assert (c.CTA_MMA, c.CGA_M, c.PAIR_KV_ROWS, c.Q_HALF, c.D_HALF) == (2, 2, 256, 64, 64)
    assert (c.MAPS_PER_SEQUENCE, c.MAP_BOX_ROWS) == (9, (64, 128, 128, 64, 64, 128, 128, 128, 128))
    assert c.DQ_SLICE_ORDER == (2, 3, 0, 1) and c.Q_TILES_PER_PAIR == 2
    assert [c.ts_k_perm(k) for k in range(8)] == [0, 1, 4, 5, 2, 3, 6, 7]
    assert (c.PACK_QUARTER_COLS, c.PACK_GROUP_COLS, c.K_STEPS_PER_BOX, c.K2_HALF_BYTES) == (16, 32, 4, 16384)
    assert (c.S_COL, c.DP_COL, c.DS_COL, c.DQ_COL, c.DV_COL, c.DK_COL, c.TMEM_COLS) == (0, 128, 128, 192, 256, 384, 512)
    assert len(cfgmod.BAR_LAYOUT) == 32 and sum(r[2] for r in cfgmod.BAR_LAYOUT) == 43
    # barrier-inspector AUDIT 2026-10-26 fix A: dv_full counts the loader's k2_empty observation (no new row, the x T cadence kept)
    counts = {row[0]: row[3] for row in cfgmod.BAR_LAYOUT}
    assert c.DV_FULL_LANES == c.EPILOGUE_LANES + c.STORE_LANES == 160 and counts["dv_full"] == c.DV_FULL_LANES
    assert counts["k2_empty"] == c.STORE_LANES == 32 and counts["dk_full"] == c.EPILOGUE_LANES == 128
    assert c.PAIR_MMA_ORDERS == ("dk_first", "s_first") and c.PAIR_MMA_ORDER_DEFAULT in c.PAIR_MMA_ORDERS
    assert c.TMA_LOAD_FORMS == ("multicast", "plain") and c.TMA_LOAD_FORM_DEFAULT in c.TMA_LOAD_FORMS
    # the pair probes' SHARED code space (day 3: + the s_tail-only D8s / D10 / D9); this body defines the first three (D2-pair / D7 / D8)
    assert c.DIAG_2CTA_NAMES == ("dq_reduce_add_off", "dp_empty_wait_off", "ds_gates_off", "exchange_off", "stats_one_chunk", "sdq_sts_off")
    assert c.DIAG_2CTA_NAMES_ON_MAP == c.DIAG_2CTA_NAMES[:3]
    # the two perf-only knobs: the store-warp wait form and the register split (the entry formula holds for every split)
    assert c.STORE_WAIT_FORMS == ("retry", "two_instr") and c.STORE_WAIT_FORM_DEFAULT == "retry"
    assert c.REG_SPLIT_NAMES == ("144/144/80", "136/136/104") and c.REG_SPLIT_DEFAULT == "144/144/80"
    for s_regs, e_regs, v_regs in c.REG_SPLITS.values():
        assert (c.SOFTMAX_WARPS * s_regs + c.EPILOGUE_WARPS * e_regs + c.SERVICE_WARPS * v_regs) * 32 == c.REGFILE == 65536
    cfgmod.validate_config()


def test_2cta_pair_mma_order_is_typed():
    """The leader MMA issue order is a typed field: ``None`` = the config default, a name outside ``CFG.PAIR_MMA_ORDERS`` declines BY NAME
    (never a silent default), the adapter reader prefixes the engine name; the kernel Constexpr follows the name."""
    from types import SimpleNamespace

    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta as cfgmod
    from cudnn.sdpa.bwd.prepared_sm100_d128_grouped_2cta import pair_mma_order

    c = cfgmod.CFG
    # the default is pinned BY NAME to its measurement (benchmarker ORDER A/B/A 2026-10-27: dk_first = main kernel -2.78 % @ 16K,
    # -1.55 % @ 32K vs s_first, band 1 %); a flip of this line needs a new A/B/A row, never a validate ladder alone
    assert c.PAIR_MMA_ORDER_DEFAULT == "s_first"
    assert cfgmod.resolve_pair_mma_order(None) == c.PAIR_MMA_ORDER_DEFAULT
    assert [cfgmod.resolve_pair_mma_order(o) for o in c.PAIR_MMA_ORDERS] == list(c.PAIR_MMA_ORDERS)
    assert cfgmod.s_lookahead_after_dk("dk_first") is True and cfgmod.s_lookahead_after_dk("s_first") is False
    for bad in ("bogus", True, 1, ""):
        with pytest.raises(ValueError, match="pair_mma_order must be one of"):
            cfgmod.resolve_pair_mma_order(bad)
    api = SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", pair_mma_order="s_first")
    assert pair_mma_order(api) == "s_first" and pair_mma_order(SimpleNamespace(_NAME="x")) == c.PAIR_MMA_ORDER_DEFAULT
    with pytest.raises(ValueError, match="sdpa_bwd_sm100_d128: pair_mma_order must be one of"):
        pair_mma_order(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", pair_mma_order="s_lookahead_first"))


def test_2cta_tma_load_form_is_typed():
    """The cga2 TMA load form (kernel header section 10 LOAD FORM) is a typed field: ``None`` = the config default (multicast, the validated
    one-CTA-mask form), a name outside ``CFG.TMA_LOAD_FORMS`` declines BY NAME, the adapter reader prefixes the engine name; the kernel
    Constexpr follows the name.  The default is pinned BY NAME: a flip needs the plain form's A/B/A row first (RULE-GAPS 240)."""
    from types import SimpleNamespace

    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta as cfgmod
    from cudnn.sdpa.bwd.prepared_sm100_d128_grouped_2cta import tma_load_form

    c = cfgmod.CFG
    assert c.TMA_LOAD_FORM_DEFAULT == "multicast"
    assert cfgmod.resolve_tma_load_form(None) == c.TMA_LOAD_FORM_DEFAULT
    assert [cfgmod.resolve_tma_load_form(f) for f in c.TMA_LOAD_FORMS] == list(c.TMA_LOAD_FORMS)
    assert cfgmod.tma_load_plain("plain") is True and cfgmod.tma_load_plain("multicast") is False and cfgmod.tma_load_plain(None) is False
    for bad in ("bogus", True, 1, "", "unicast"):
        with pytest.raises(ValueError, match="tma_load_form must be one of"):
            cfgmod.resolve_tma_load_form(bad)
    api = SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", tma_load_form="plain")
    assert tma_load_form(api) == "plain" and tma_load_form(SimpleNamespace(_NAME="x")) == c.TMA_LOAD_FORM_DEFAULT
    with pytest.raises(ValueError, match="sdpa_bwd_sm100_d128: tma_load_form must be one of"):
        tma_load_form(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", tma_load_form="multicast_2cta"))


def test_2cta_diag_2cta_is_typed():
    """The pair body's DIAGNOSTIC probe (kernel header section 11; PLAN day-2 rank 1) is ONE typed field: ``None`` = no probe (the shipped
    body, THE default), a ``CFG.DIAG_2CTA_NAMES`` name = that probe (output WRONG BY DESIGN), anything else declines BY NAME; the adapter
    reader prefixes the engine name; the kernel Constexpr is a distinct non-zero code per name and 0 for None (the kernel's module constants
    DIAG_* are derived from it, never literals)."""
    from types import SimpleNamespace

    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta as cfgmod
    from cudnn.sdpa.bwd.kernels.sm100 import bprop_d128_accum_grouped_2cta_f16 as body
    from cudnn.sdpa.bwd.prepared_sm100_d128_grouped_2cta import diag_2cta

    c = cfgmod.CFG
    assert cfgmod.resolve_diag_2cta(None) is None and cfgmod.diag_2cta_code(None) == 0 == body.DIAG_NONE
    assert [cfgmod.resolve_diag_2cta(n) for n in c.DIAG_2CTA_NAMES] == list(c.DIAG_2CTA_NAMES)
    codes = [cfgmod.diag_2cta_code(n) for n in c.DIAG_2CTA_NAMES]
    assert codes[:3] == [body.DIAG_DQ_REDUCE_ADD_OFF, body.DIAG_DP_EMPTY_WAIT_OFF, body.DIAG_DS_GATES_OFF] and len(set(codes)) == 6 and 0 not in codes
    # the s_tail-only probes are in the shared code space but NOT defined on this map: the per-map resolver declines them by name
    from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta_stail import resolve_diag_2cta_for_map

    for name in c.DIAG_2CTA_NAMES[3:]:
        with pytest.raises(ValueError, match="is not defined on tmem_map 'dp_tail'"):
            resolve_diag_2cta_for_map(name, "dp_tail")
    for bad in ("bogus", True, 1, "", "D7", "dp_empty_off"):
        with pytest.raises(ValueError, match="diag_2cta must be None or one of"):
            cfgmod.resolve_diag_2cta(bad)
    api = SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", diag_2cta="ds_gates_off")
    assert diag_2cta(api) == "ds_gates_off" and diag_2cta(SimpleNamespace(_NAME="x")) is None
    with pytest.raises(ValueError, match="sdpa_bwd_sm100_d128: diag_2cta must be None or one of"):
        diag_2cta(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", diag_2cta="dq_reduce_add"))
    from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128

    assert SdpaBwdDslSm100D128.diag_2cta is None  # the record default: no probe


def test_2cta_knobs_are_typed():
    """The two perf-only knobs (kernel header section 4 / 11) are typed fields: ``None`` = the config default (pinned BY NAME: a flip needs the
    knob process's A/B/A row, RULE-GAPS 240), a name outside the domain declines BY NAME, the adapter readers prefix the engine name; the
    kernel Constexprs follow the names (``store_wait_spin`` bool; ``reg_split_code`` index -> the three setmaxnreg counts, never literals)."""
    from types import SimpleNamespace

    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta as cfgmod
    from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128
    from cudnn.sdpa.bwd.prepared_sm100_d128_grouped_2cta import reg_split, store_wait_form

    c = cfgmod.CFG
    assert c.STORE_WAIT_FORM_DEFAULT == "retry" and c.REG_SPLIT_DEFAULT == "144/144/80"  # the validated defaults, by name
    assert cfgmod.resolve_store_wait_form(None) == "retry" and cfgmod.store_wait_spin(None) is False and cfgmod.store_wait_spin("two_instr") is True
    assert cfgmod.resolve_reg_split(None) == c.REG_SPLIT_DEFAULT and cfgmod.reg_split_code(None) == 0 and cfgmod.reg_split_code("136/136/104") == 1
    assert cfgmod.reg_split_counts(0) == (c.SOFTMAX_REGS, c.EPILOGUE_REGS, c.SERVICE_REGS) == (144, 144, 80)
    assert cfgmod.reg_split_counts(1) == (144 - c.M5_REG_SHIFT, 144 - c.M5_REG_SHIFT, 80 + (8 + 4) * c.M5_REG_SHIFT // 4) == (136, 136, 104)
    for bad in ("bogus", True, 1, "", "spin", "2_instr"):
        with pytest.raises(ValueError, match="store_wait_form must be one of"):
            cfgmod.resolve_store_wait_form(bad)
    for bad in ("bogus", True, 1, "", "136", "144/144/80/0", "136/136/104 "):
        with pytest.raises(ValueError, match="reg_split must be one of"):
            cfgmod.resolve_reg_split(bad)
    api = SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", store_wait_form="two_instr", reg_split="136/136/104")
    assert (store_wait_form(api), reg_split(api)) == ("two_instr", "136/136/104")
    assert (store_wait_form(SimpleNamespace(_NAME="x")), reg_split(SimpleNamespace(_NAME="x"))) == ("retry", "144/144/80")
    with pytest.raises(ValueError, match="sdpa_bwd_sm100_d128: store_wait_form must be one of"):
        store_wait_form(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", store_wait_form="nanosleep"))
    with pytest.raises(ValueError, match="sdpa_bwd_sm100_d128: reg_split must be one of"):
        reg_split(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", reg_split="128/128/128"))
    assert SdpaBwdDslSm100D128.store_wait_form is None and SdpaBwdDslSm100D128.reg_split is None  # the record defaults


def test_2cta_contract_adds_two_maps_per_sequence():
    """The 2-CTA workspace is the grouped THD workspace with nine per-sequence maps instead of seven; the partials follow."""
    from cudnn.sdpa.bwd import d128_grouped_2cta_contract as c2
    from cudnn.sdpa.bwd import d128_grouped_contract as c1

    for batch, heads, tq, tkv in ((1, 2, 7, 5), (3, 8, 4097, 2049), (4, 32, 32768, 32768)):
        r1, t1 = c1.thd_workspace_layout(batch, heads, tq, tkv)
        r2, t2 = c2.thd_workspace_layout(batch, heads, tq, tkv)
        assert r2[4][1] == (9 * batch * 16,) and r1[4][1] == (7 * batch * 16,)
        assert r2[:4] == r1[:4] and t2 >= t1
        assert r2[5][1] == r1[5][1] and r2[6][1] == r1[6][1]  # the partial regions keep their shapes, moved by the extra maps
    assert c2.THD == _IMPLEMENTATION
    with pytest.raises(ValueError):
        c2.validate_thd_bounds(1, 2, 1, 2**31 - 200, 128, 16, 16, 16, 16)


_RENDER = """
import os, sys, re, glob
os.environ["CUTE_DSL_DUMP_DIR"] = sys.argv[1]
os.environ["CUTE_DSL_KEEP"] = "ptx"
os.environ["CUTE_DSL_ARCH"] = "sm_100a"
os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
import cutlass
from {host} import compile_host
ports = ("q", "k", "v", "o", "do", "dq", "dk", "dv", "stats")
B, H, HK = 2, 8, 2
ts = lambda p: H if p == "stats" else (HK if p in ("k", "v", "dk", "dv") else H) * 128
specs = tuple((i, 1, ts(p), "int32") for i, p in enumerate(ports))
compile_host((B, H, HK, True), specs, tuple("int32" for _ in ports), cutlass.BFloat16, True, 100, **{form_kw})
ptx = open(sorted(glob.glob(os.path.join(sys.argv[1], "*.ptx")), key=os.path.getmtime)[-1], "rb").read()
import hashlib
print("PTX_MD5", hashlib.md5(ptx).hexdigest())
for name, pat in {census}.items():
    print("CENSUS", name, len(re.findall(pat.encode(), ptx)))
for action, imm in re.findall(rb"setmaxnreg\\.(inc|dec)\\.sync\\.aligned\\.u32\\s+(\\d+)", ptx):
    print("SETMAXNREG", action.decode(), imm.decode())
print("RENDER_OK")
"""
_CENSUS = {
    "mma_cta_group_2": r"tcgen05\.mma\.cta_group::2",
    "mma_cta_group_1": r"tcgen05\.mma\.cta_group::1",
    "commit_multicast": r"tcgen05\.commit\.cta_group::2\.mbarrier::arrive::one\.shared::cluster\.multicast::cluster",
    "tma_cta_group_2": r"multicast::cluster\.cta_group::2",  # the multicast form (the default): a one-CTA mask per load
    "tma_plain_cta_group_2": r"complete_tx::bytes\.cta_group::2",  # the plain form: no multicast modifier (tma_load_form = "plain")
    "dsmem_ship": r"cp\.async\.bulk\.shared::cluster\.shared::cta",
    "reduce_add": r"cp\.reduce\.async\.bulk",
    "try_cancel": r"clusterlaunchcontrol\.try_cancel",
    "tmem_alloc_cta2": r"tcgen05\.alloc\.cta_group::2",
    "local_mem": r"\.local",
    "entry": r"\.entry\s+cudnn_kernel_bprop_d128_accum_grouped_2cta_kernel",
    "nanosleep": r"nanosleep\.u32",  # the test-only DEBUG_LOADER_DELAY_US lever: absent from the default rendering
    "mbarrier_arrive": r"mbarrier\.arrive",  # every arrive site of header section 5 (a DIAGNOSTIC probe never moves one)
    "try_wait": r"mbarrier\.try_wait",  # the wait sites; a probe removes exactly its own (D7: 1, D8: 3)
}
_EXPECTED = {
    "mma_cta_group_2": 56,  # S^T 8 + S^T lookahead 8 + dP^T 8 + dV 8 + dK 8 + dQ 16
    "mma_cta_group_1": 0,  # P10: never mixed
    "commit_multicast": 12,  # the nine MMA_COMMIT rows + the S^T lookahead pair + the zero-trip dq_done
    "tma_cta_group_2": 12,  # K1 2 + V 2 + K2 2 + Q1 2 + dO1 2 + dO2 1 + Q2 1
    "tma_plain_cta_group_2": 0,  # the default form carries the mask on every load
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
# The DIAGNOSTIC probes (typed field ``diag_2cta``): what each removes from the default census and nothing else.
_DIAG_CENSUS_DELTA = {
    "dq_reduce_add_off": {"reduce_add": 0},  # D2-pair: the four reduce-add ISSUES
    "dp_empty_wait_off": {"try_wait": _EXPECTED["try_wait"] - 1},  # D7: the leader's per-tile dp_empty WAIT (the P15 drain stays)
    "ds_gates_off": {"try_wait": _EXPECTED["try_wait"] - 3},  # D8: ds_smem / ds_recv_full / ds_pair_ready WAITs before dQ(t)
}
_RECORD = pathlib.Path(__file__).resolve().parent / "renderings" / "md5_d128_accum_grouped_2cta_default_sm_100a.txt"


def _setmaxnreg_of(split):
    """The setmaxnreg immediates a register split renders: inc x (softmax + epilogue roles = 2 sites), dec x (service roles = 5 sites)."""
    from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta import CFG

    s_regs, e_regs, v_regs = CFG.REG_SPLITS[split]
    return sorted([("inc", s_regs), ("inc", e_regs)] + [("dec", v_regs)] * (CFG.SERVICE_WARPS + 1))  # MMA leader + follower + load + store + sched


def _render(tmp_path, form_kw):
    """Host trace-compile of the 2-CTA chain for sm_100a in a fresh process -> (PTX md5, census, [(setmaxnreg action, immediate), ...])."""
    _adapter_class()  # the tree under test must be the one this venv imports
    script = _RENDER.format(host=_HOST_MODULE, census=repr(_CENSUS), form_kw=repr(form_kw))
    proc = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=600, env=dict(os.environ))
    assert proc.returncode == 0 and "RENDER_OK" in proc.stdout, proc.stdout[-3000:] + proc.stderr[-3000:]
    md5 = re.search(r"PTX_MD5 ([0-9a-f]{32})", proc.stdout).group(1)
    census = {m.group(1): int(m.group(2)) for m in re.finditer(r"CENSUS (\S+) (\d+)", proc.stdout)}
    regs = [(m.group(1), int(m.group(2))) for m in re.finditer(r"SETMAXNREG (inc|dec) (\d+)", proc.stdout)]
    return md5, census, regs


def _render_census(tmp_path, form_kw):
    return _render(tmp_path, form_kw)[1]


def _parse_record(path):
    """-> (dsl line or None, md5 or None) of the default-rendering record (RULE-GAPS 241: the 2-CTA chain's own md5 pin)."""
    dsl, md5 = None, None
    for ln in path.read_text().splitlines():
        if ln.startswith("dsl="):
            dsl = ln[len("dsl=") :].strip()
        m = re.match(r"d128_accum_grouped_2cta sm_100a default rc=0 ptx_md5=([0-9a-f]{32})", ln)
        if m:
            md5 = m.group(1)
    return dsl, md5


def test_2cta_default_rendering_ptx_md5_is_unchanged(tmp_path):
    """The DEFAULT rendering (no probe, the config defaults for every typed field) is PTX-IDENTICAL to the committed record: every
    typed field (pair_mma_order, tma_load_form, diag_2cta, ...) is a trace-time constant that folds to the shipped code when unset.
    Skips (never widens) on another DSL build; a deliberate protocol change re-renders the record and says so in its header."""
    from cudnn.frost.buffers import cutedsl_state

    assert _RECORD.is_file(), _RECORD
    dsl, want = _parse_record(_RECORD)
    assert dsl and dsl.startswith("nvidia-cutlass-dsl ") and want, (dsl, want)
    _installed, version = cutedsl_state()
    have = " ".join(version) if version else None
    if have != dsl:
        pytest.skip(f"the md5 record was rendered with {dsl}; installed {have}: PTX text differs by DSL build, re-render the record")
    got, census, regs = _render(tmp_path, {})
    print(f"\n2-CTA sm_100a default: PTX md5 {got} (record {want}) setmaxnreg {regs}")
    assert census == _EXPECTED, census
    assert sorted(regs) == _setmaxnreg_of("144/144/80"), regs  # the default split: 144 x 2 (inc) + 80 x 5 (dec)
    assert got == want, f"PTX md5 {got} != the record's {want} -- the DEFAULT rendering changed (a typed field no longer folds, or the body moved)"


def test_2cta_host_renders_for_sm_100a(tmp_path):
    """Host trace-compile of the 2-CTA chain for sm_100a in a fresh process (no device): the pair body's PTX census is the design."""
    census = _render_census(tmp_path, {})
    print("\n2-CTA sm_100a PTX census:", census)
    assert census == _EXPECTED, census


def test_2cta_host_renders_plain_tma_form_for_sm_100a(tmp_path):
    """``tma_load_form = "plain"`` (kernel header section 10 LOAD FORM): the twelve cga2 loads render as `cp.async.bulk.tensor ...
    complete_tx::bytes.cta_group::2` WITHOUT `.multicast::cluster` (FAv4's form); every other census row (MMAs, commits, the DSMEM ship,
    the reduce-adds, no local memory, no nanosleep) is the default's -- the form moves no barrier and traces no other code."""
    census = _render_census(tmp_path, {"tma_load_form": "plain"})
    print("\n2-CTA sm_100a PTX census (plain TMA form):", census)
    expected = dict(_EXPECTED, tma_cta_group_2=0, tma_plain_cta_group_2=_EXPECTED["tma_cta_group_2"])
    assert census == expected, census


@pytest.mark.parametrize("probe", sorted(_DIAG_CENSUS_DELTA))
def test_2cta_host_renders_diag_probes_for_sm_100a(tmp_path, probe):
    """``diag_2cta = <probe>`` (kernel header section 11 DIAGNOSTIC): each probe removes exactly its own ISSUE (D2-pair: the four reduce-adds)
    or WAIT sites (D7: one, D8: three) and nothing else -- every mbarrier.arrive site, every MMA, commit, ship and load of the default census
    stays (the barrier table is identical; output wrong by design is the probe's point, never rendered as a default)."""
    md5, census, _ = _render(tmp_path, {"diag_2cta": probe})
    print(f"\n2-CTA sm_100a PTX census ({probe}): md5 {md5} {census}")
    assert census == dict(_EXPECTED, **_DIAG_CENSUS_DELTA[probe]), census


def test_2cta_host_renders_two_instr_wait_form_for_sm_100a(tmp_path):
    """``store_wait_form = "two_instr"`` (header section 4 KNOB): the store warp's six mbarrier waits render as the hint-less inline-PTX spin;
    the PTX census is the default's row for row (the same 63 try_wait sites, every arrive, MMA, commit, ship, load and reduce-add) and the
    register split is untouched -- only the retry mechanics of six waits move."""
    md5, census, regs = _render(tmp_path, {"store_wait_form": "two_instr"})
    print(f"\n2-CTA sm_100a PTX census (two_instr wait form): md5 {md5} {census}")
    assert census == _EXPECTED, census
    assert sorted(regs) == _setmaxnreg_of("144/144/80"), regs


def test_2cta_host_renders_reg_split_136_for_sm_100a(tmp_path):
    """``reg_split = "136/136/104"`` (header section 4 KNOB, M5'): the setmaxnreg immediates move to 136 x 2 (inc) + 104 x 5 (dec) and nothing
    else does -- the PTX census is the default's row for row (no local memory: a spill would show as `.local`)."""
    md5, census, regs = _render(tmp_path, {"reg_split": "136/136/104"})
    print(f"\n2-CTA sm_100a PTX census (reg split 136/136/104): md5 {md5} {census} setmaxnreg {regs}")
    assert census == _EXPECTED, census
    assert sorted(regs) == _setmaxnreg_of("136/136/104"), regs


def test_2cta_route_declines_dense_and_mha_typed():
    """The pair body serves grouped THD only: dense and MHA requests decline BY NAME before any compile."""
    _require_2cta()
    dense = make_case((256, 256), (256, 256), hq=4, hkv=2, thd=False)
    with pytest.raises(NotImplementedError, match="2-CTA datapath requires THD"):
        make_adapter_2cta(dense, compile_plan=False).check_support()
    from d128_thd_test_utils import declaration as mha_declaration
    from d128_thd_test_utils import make_case as make_mha_case

    mha = make_mha_case((129, 3), (129, 3), heads=2)  # the grouped utils refuse hq == hkv; the MHA utils build the MHA storage
    with pytest.raises(NotImplementedError, match="grouped heads"):
        make_adapter_2cta(mha, compile_plan=False, declaration=mha_declaration).check_support()


# ------------------------------------------------------------------------------------------------------------ GPU (the fp64 gate)
@pytest.mark.parametrize(
    "lq,lk,hq,hkv,causal,layout,offset_dtype,units",
    [
        pytest.param((256, 256), (256, 256), 4, 2, True, "TH1", torch.int32, "tokens", id="uniform-256-causal"),
        pytest.param((1024,), (1024,), 8, 1, True, "HN", torch.int32, "tokens", id="uniform-1024-gqa8-causal"),
        pytest.param((384, 384), (384, 384), 4, 2, False, "TH1", torch.int64, "elements", id="uniform-384-noncausal"),
        pytest.param((256, 512, 768, 1024), (256, 512, 768, 1024), 4, 2, True, "TH1", torch.int32, "tokens", id="ramp-1024-causal"),
        pytest.param((129, 0, 127, 1), (65, 128, 0, 3), 4, 2, True, "TH1", torch.int32, "tokens", id="suite-degenerate-causal"),
        pytest.param((129, 0, 127, 1), (65, 128, 0, 3), 6, 2, False, "HN", torch.int64, "elements", id="suite-degenerate-noncausal"),
        pytest.param((65, 0, 3), (17, 129, 0), 6, 2, True, "HN", torch.int64, "elements", id="short-pair-half-dead"),
        pytest.param((300, 300), (300, 300), 3, 1, True, "TH1", torch.int64, "elements", id="pair-tail-300"),
    ],
)
def test_2cta_thd_matrix(lq, lk, hq, hkv, causal, layout, offset_dtype, units):
    _require_2cta()
    case = make_case(lq, lk, hq=hq, hkv=hkv, thd=True, dtype=torch.bfloat16, causal=causal, stats_layout=layout, offset_dtype=offset_dtype, units=units)
    _run_case(case)


def test_2cta_thd_all_empty():
    """Every sequence empty: exact zeros, no map dereference (the dead-pair protocol on BOTH CTAs)."""
    _require_2cta()
    case = make_case((0, 0), (0, 0), hq=3, hkv=1, dtype=torch.bfloat16, causal=True, stats_layout="HN", envelope=(128, 128))
    _run_case(case)


def test_2cta_persistent_reuse_more_pairs_than_sms():
    """P14 row: n_q == 1 with more pairs than SMs (a cluster claims a second CLC item) plus a half-dead last pair (seq_k % 256 == 128)."""
    _require_2cta()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    hq, b = 8, 2
    skv = 256 * ((sms // (hq * b)) + 2) + 128  # pairs x heads x batch > SMs, and the last pair's second CTA owns no live key
    case = make_case((128, 64), (skv, skv - 256), hq=hq, hkv=1, dtype=torch.bfloat16, causal=True, stats_layout="TH1")
    _run_case(case)


_LEVER = """
import os, sys, importlib, importlib.util
os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
sys.path.insert(0, {testdir!r})
name = "cudnn.sdpa.bwd.kernels.sm100.bprop_d128_accum_grouped_2cta_f16"
spec = importlib.util.find_spec(name)
mod = importlib.util.module_from_spec(spec)
setattr(mod, "FROST_D128_2CTA_DEBUG_LOADER_DELAY_US", {delay})
sys.modules[name] = mod
spec.loader.exec_module(mod)
assert mod.DEBUG_LOADER_DELAY_US == {delay}, mod.DEBUG_LOADER_DELAY_US
import torch
from d128_grouped_test_utils import make_case
from test_sdpa_bwd_d128_2cta_sm100 import _run_case
sms = torch.cuda.get_device_properties(0).multi_processor_count
hq, b = 8, 2
skv = 256 * ((sms // (hq * b)) + 2) + 128
case = make_case((128, 64), (skv, skv - 256), hq=hq, hkv=1, dtype=torch.bfloat16, causal=True, stats_layout="TH1")
_run_case(case, repeats=2)
print("LEVER_OK", {delay})
"""


def test_2cta_loader_delay_survives_zero_trip_items():
    """barrier-inspector AUDIT 2026-10-26 lever 4 = the k2_empty back-edge detector (finding 1, fix A): the LOADER of both CTAs sleeps
    ~20 us at the top of every item, longer than a zero-trip item's whole store chain, on the P14 row (a cluster runs a second item;
    the half-dead last pair's causal kv blocks past seq_q are zero-trip).  Pre-fix body: k2_empty completes twice before its single
    waiter observes it -> the loader's parity aliases -> the next live item's MMA waits loads never armed -> HANG (the subprocess times
    out = RED).  With dv_full = DV_FULL_LANES the store's next k2_empty arrive waits the loader's observation -> GREEN.  A fresh process:
    the lever is a module global read when the body executes (the d512 2x2 form), and the compile cache must not serve the unarmed body."""
    _require_2cta()
    script = _LEVER.format(testdir=str(pathlib.Path(__file__).resolve().parent), delay=20)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=900, env=dict(os.environ))
    assert proc.returncode == 0 and "LEVER_OK 20" in proc.stdout, proc.stdout[-3000:] + proc.stderr[-3000:]


def test_2cta_mma_orders_agree_bitwise_on_dk_dv():
    """The two leader issue orders (header section 5 MMA ISSUE ORDER) are the SAME function: only S^T(t+1)'s issue position moves, every
    accumulation chain per GEMM is unchanged, so dK / dV (deterministic native partials + the fp32 group fold) agree BITWISE between the
    orders on one case (G10 section 8: a knob is bitwise under every served value); dQ (reduce-add order across pairs) passes the fp64
    gate under each.  Two tiles per pair so the lookahead is exercised (n_q = 4 at s 512)."""
    _require_2cta()
    from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta import CFG

    case = make_case((256, 512), (256, 512), hq=4, hkv=2, thd=True, dtype=torch.bfloat16, causal=True, stats_layout="TH1")
    grads = {}
    for order in CFG.PAIR_MMA_ORDERS:
        api = make_adapter_2cta(case, compile_plan=False)
        api.pair_mma_order = order
        assert api.check_support()
        api.compile()
        assert api._prepared.artifact.implementation == _IMPLEMENTATION
        workspace = make_workspace(api.scratch_workspace_bytes())
        poison(case, workspace)
        api.execute(**adapter_arguments(case), workspace=workspace.tensor)
        torch.cuda.synchronize()
        check_case(case, workspace)
        grads[order] = {role: case.tensors[role][0].detach().clone() for role in ("dk", "dv")}
    a, b = (grads[o] for o in CFG.PAIR_MMA_ORDERS)
    for role in ("dk", "dv"):
        assert torch.equal(a[role].view(torch.int16), b[role].view(torch.int16)), f"{role}: the two MMA orders differ bitwise"


def test_2cta_tma_load_forms_agree_bitwise_on_dk_dv():
    """The two cga2 TMA load forms (header section 10 LOAD FORM) move the SAME bytes into the SAME SMEM image completing on the same
    leader mbarrier: dK / dV agree BITWISE between the forms (G10 section 8: a knob is bitwise under every served value); dQ passes
    the fp64 gate under each.  Two tiles per pair, two live items per cluster class."""
    _require_2cta()
    from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta import CFG

    case = make_case((256, 512), (256, 512), hq=4, hkv=2, thd=True, dtype=torch.bfloat16, causal=True, stats_layout="TH1")
    grads = {}
    for form in CFG.TMA_LOAD_FORMS:
        api = make_adapter_2cta(case, compile_plan=False)
        api.tma_load_form = form
        assert api.check_support()
        api.compile()
        assert api._prepared.artifact.implementation == _IMPLEMENTATION
        workspace = make_workspace(api.scratch_workspace_bytes())
        poison(case, workspace)
        api.execute(**adapter_arguments(case), workspace=workspace.tensor)
        torch.cuda.synchronize()
        check_case(case, workspace)
        grads[form] = {role: case.tensors[role][0].detach().clone() for role in ("dk", "dv")}
    a, b = (grads[f] for f in CFG.TMA_LOAD_FORMS)
    for role in ("dk", "dv"):
        assert torch.equal(a[role].view(torch.int16), b[role].view(torch.int16)), f"{role}: the two TMA load forms differ bitwise"


@pytest.mark.parametrize("field,values", [("store_wait_form", ("retry", "two_instr")), ("reg_split", ("144/144/80", "136/136/104"))])
def test_2cta_knobs_agree_bitwise_on_dk_dv(field, values):
    """Each KNOB is the same function under every served value (G10 section 8): dK / dV agree BITWISE between the two values of the knob on
    one case (the wait form moves retry mechanics only; the register split moves setmaxnreg counts only), dQ passes the fp64 gate under each.
    Two tiles per pair, two live items per cluster class."""
    _require_2cta()
    case = make_case((256, 512), (256, 512), hq=4, hkv=2, thd=True, dtype=torch.bfloat16, causal=True, stats_layout="TH1")
    grads = {}
    for value in values:
        api = make_adapter_2cta(case, compile_plan=False)
        setattr(api, field, value)
        assert api.check_support()
        api.compile()
        assert api._prepared.artifact.implementation == _IMPLEMENTATION
        workspace = make_workspace(api.scratch_workspace_bytes())
        poison(case, workspace)
        api.execute(**adapter_arguments(case), workspace=workspace.tensor)
        torch.cuda.synchronize()
        check_case(case, workspace)
        grads[value] = {role: case.tensors[role][0].detach().clone() for role in ("dk", "dv")}
    a, b = (grads[v] for v in values)
    for role in ("dk", "dv"):
        assert torch.equal(a[role].view(torch.int16), b[role].view(torch.int16)), f"{role}: the two {field} values differ bitwise"


def test_2cta_launch_count_and_cluster_facts():
    """The chain is five launches with the pair MAIN kernel; the kernel header's facts (cluster, pair rows) come from CFG."""
    _require_2cta()
    import importlib

    body = importlib.import_module("cudnn.sdpa.bwd.kernels.sm100.bprop_d128_accum_grouped_2cta_f16")
    assert (body.CFG.CGA_M, body.CFG.CGA_N, body.CFG.PAIR_KV_ROWS) == (2, 1, 256)
    case = make_case((256, 256), (256, 256), hq=4, hkv=2, thd=True, dtype=torch.bfloat16, causal=True)
    api, workspace = _run_case(case, repeats=1)
    names = []
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        api.execute(**adapter_arguments(case), workspace=workspace.tensor)
        torch.cuda.synchronize()
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            names.append(event.name)
    kernels = [n for n in names if n.startswith("cudnn_kernel_")]
    assert len(kernels) == 5, kernels
    assert any("bprop_d128_accum_grouped_2cta_kernel" in n for n in kernels), kernels
    assert any("setup_thd_d128_2cta_kernel" in n for n in kernels), kernels
