# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The grouped THD d128 backward's typed op-level field ``dkv_accum`` (fp32_reduce): how the GQA group's dK / dV are accumulated.

``"bf16_partials"`` (the default; the field unset renders BYTE-IDENTICALLY): the main kernel stores one bf16 dK / dV partial per Q head and
the ``dkv_reduce_thd_origins`` fold launch sums the group in fp32 (five launches).  ``"fp32_reduce"``: a per-KV-head fp32 dK / dV workspace
(carved from the partial regions, zero-filled by the ``delta_zero_thd`` launch) that EVERY Q-head owner reduce-adds its fp32 tile into
(``cp.reduce.async.bulk.tensor`` add.f32 -- the dQ slices' primitive -- from the same alias staging, two rounds per item: dV then dK), the
fold launch gone, the bf16 convert folded into ``cast_dq_thd`` (four launches).  A ROUNDING CHANGE (the group accumulates in fp32 in a
non-deterministic order instead of 8 bf16-rounded partials summed in a fixed order): an op ATTRIBUTE the caller sets, never a knob; dQ is
untouched by the field.  Both bodies carry it: the 1-CTA body (``bprop_d128_accum_grouped_f16`` through ``bprop_d128_accum_thd_f16``) and
the cta_group::2 pair body (``bprop_d128_accum_grouped_2cta_stail_f16``).

Host rows (no GPU): the typed domain, config = truth (the fp32 staging slots of both bodies), the default renderings of BOTH chains equal to
their committed records with the field unset AND with ``bf16_partials`` named, the fp32 renderings' census (the reduce-adds up by the eight
dK / dV slices, the bf16 tile stores gone, the fold entry gone, the P7 commit / wait groups and the two ring credits / waits of the dK
staging's WAR edge), the typed declines.  GPU rows: the fp64-oracle ladder on both bodies at the smallest shapes, the launch census
(5 -> 4) with the R9 gate, dQ bitwise vs the default where one kv block feeds it (one reduce-add contributor), dK / dV within the suite's
own tolerance (never widened) with the max error of both arms recorded side by side, the declines by name (dense, MHA).
"""

import json
import pathlib
import re
import subprocess
import sys
import textwrap

import pytest
import torch

from d128_grouped_test_utils import adapter_arguments, check_case, make_adapter, make_case, make_workspace, poison
from frost_test_utils import D128_THD_SHARED_KERNELS, arch_known_to_the_dsl, assert_kernel_record, ptx_kernel_md5s, read_kernel_record, requires_dsl
from test_sdpa_bwd_d128_2cta_stail_sm100 import make_adapter_2cta_stail
from test_sdpa_bwd_d128_accum_sm100 import _require_accumulation
from test_sdpa_bwd_d128_diag_dq_reduce_add_sm100 import _RECORD as _RECORD_1CTA
from test_sdpa_bwd_d128_diag_dq_reduce_add_sm100 import _RECORD_TOKEN as _RECORD_TOKEN_1CTA
from test_sdpa_bwd_d128_diag_dq_reduce_add_sm100 import _skip_unless_recorded_dsl
from test_sdpa_bwd_d128_2cta_stail_sm100 import _RECORD as _RECORD_STAIL
from test_sdpa_bwd_d128_2cta_stail_sm100 import _RECORD_TOKEN as _RECORD_TOKEN_STAIL
from test_sdpa_bwd_dsl_sm107 import _TOL

pytestmark = [pytest.mark.L0, requires_dsl]

FIELD = "dkv_accum"
FORMS = ("bf16_partials", "fp32_reduce")
_FP32 = {FIELD: "fp32_reduce"}
_GEOMETRY = {"1cta": (4, 32, 4), "pair": (2, 8, 2)}  # the record geometries of the two chains' md5 pins
# PTX census per chain: the fp32 form adds the 8 dK / dV reduce-add slices (4 -> 12), deletes the 4 bf16 tile stores, drops the fold entry
# (5 -> 4 kernels); the store warp credits the ring after the dV slices' source reads (the dK staging's WAR edge: +4 arrives on the 1-CTA body's
# bootstrapped 4-slot ring; the pair's pre-armed ring is untouched); wait SITES: the 1-CTA epilogue's dK round waits the 4 slots where the bf16 form waited 2 (one per
# tensor's half 0) -> 45 - 2 + 4 = 47, the pair adds ONE new row dvk_stage_empty (store -> epilogue, between its two rounds): +1 arrive / +1 wait (68 -> 69); the P7 bulk groups: per-slice commits (1-CTA 5 -> 13,
# waits 5 -> 9), the pair's one commit per tensor (5 -> 6, waits 5 -> 6).  No local memory, no nanosleep under either form.
_CENSUS = {
    "REDUCE_ADD_TENSOR": r"cp\.reduce\.async\.bulk\.tensor",
    "BULK_TENSOR_STORE": r"cp\.async\.bulk\.tensor\.4d\.global\.shared::cta",
    "MBARRIER_ARRIVE": r"mbarrier\.arrive",
    "MBARRIER_WAIT": r"mbarrier\.try_wait",
    "BULK_COMMIT": r"cp\.async\.bulk\.commit_group",
    "BULK_WAIT": r"cp\.async\.bulk\.wait_group",
    "LOCAL": r"\.local",
    "NANOSLEEP": r"nanosleep\.u32",
    "ENTRY_FOLD": r"\.entry\s+cudnn_kernel_dkv_reduce_thd_origins_kernel",
    "ENTRIES": r"\.entry\s+",
}
_EXPECTED_FP32_DELTA = {
    "1cta": dict(REDUCE_ADD_TENSOR=(4, 12), BULK_TENSOR_STORE=(4, 0), MBARRIER_ARRIVE=(30, 34), MBARRIER_WAIT=(45, 47), BULK_COMMIT=(5, 13), BULK_WAIT=(5, 9)),
    # the pair's default stats form is the tma_ring (the stats ring's arm + release, its two waits): arrive 42 / wait 70 under bf16_partials
    "pair": dict(REDUCE_ADD_TENSOR=(4, 12), BULK_TENSOR_STORE=(4, 0), MBARRIER_ARRIVE=(42, 43), MBARRIER_WAIT=(70, 71), BULK_COMMIT=(5, 6), BULK_WAIT=(5, 6)),
}

_PROBE = textwrap.dedent(r"""
    import glob, hashlib, json, os, re, sys
    dump, chain, fields, spec, census = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), json.loads(sys.argv[4]), json.loads(sys.argv[5])
    os.environ["CUTE_DSL_DUMP_DIR"] = dump
    os.environ["CUTE_DSL_KEEP"] = "ptx"
    os.environ["CUTE_DSL_ARCH"] = "sm_100a"
    os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
    import cutlass
    B, H, HK = spec
    ports = ("q", "k", "v", "o", "do", "dq", "dk", "dv", "stats")
    kv = ("k", "v", "dk", "dv")
    specs = tuple((i, 1, H if p == "stats" else (HK if p in kv else H) * 128, "int32") for i, p in enumerate(ports))
    dtypes = tuple("int32" for _ in ports)
    if chain == "pair":
        from cudnn.sdpa.bwd.kernels.sm100.prepared_d128_grouped_2cta_host import compile_host
        compile_host((B, H, HK, True), specs, dtypes, cutlass.BFloat16, True, 100, **fields)
    else:
        from cudnn.sdpa.bwd.kernels.sm100.prepared_d128_grouped_thd_host import compile_host
        compile_host((B, H, HK, True), specs, dtypes, cutlass.BFloat16, True, 100, **fields)
    ptxs = sorted(glob.glob(os.path.join(dump, "*.ptx")), key=os.path.getmtime)
    if not ptxs:
        print("FAIL no ptx dumped into", dump, os.listdir(dump)); sys.exit(3)
    txt = open(ptxs[-1], "rb").read()
    print("PTX_MD5", hashlib.md5(txt).hexdigest())
    print("PTX_FILE", ptxs[-1])
    for key, pat in census.items():
        print("CENSUS", key, len(re.findall(pat.encode(), txt)))
    """)


def _render_kernels(tmp_path, chain, **fields):
    """Host trace-compile ONE chain (1cta | pair) for sm_100a in a fresh process with the typed fields -> (PTX md5, census, {kernel: md5})."""
    if not arch_known_to_the_dsl("sm_100a"):
        pytest.skip("this cutlass-dsl has no sm_100a")
    tag = "_".join(f"{k}_{v}" for k, v in sorted(fields.items())) or "default"
    dump = tmp_path / f"sm100a_dkv_accum_{chain}_{tag}"
    dump.mkdir()
    script = dump / "ptx_probe.py"
    script.write_text(_PROBE)
    cmd = [sys.executable, str(script), str(dump), chain, json.dumps(fields), json.dumps(_GEOMETRY[chain]), json.dumps(_CENSUS)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"sm_100a trace-compile of the {chain} chain ({fields}) failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    md5, ptx_file, census = None, None, {}
    for ln in proc.stdout.splitlines():
        if ln.startswith("PTX_MD5 "):
            md5 = ln.split()[1]
        elif ln.startswith("PTX_FILE "):
            ptx_file = ln.split(" ", 1)[1]
        elif ln.startswith("CENSUS "):
            _, key, n = ln.split()
            census[key] = int(n)
    assert md5 and ptx_file and set(census) == set(_CENSUS), proc.stdout[-2000:]
    with open(ptx_file) as f:
        kernels = ptx_kernel_md5s(f.read())
    return md5, census, kernels


def _render(tmp_path, chain, **fields):
    """Host trace-compile ONE chain (1cta | pair) for sm_100a in a fresh process with the typed fields -> (PTX md5, census)."""
    return _render_kernels(tmp_path, chain, **fields)[:2]


# The per-kernel default records of the two chains (token, variant, file): the pair chain's in the S-tail module, the one-CTA chain's in the
# diag module; both exclude the shared group fold (frost_test_utils.D128_THD_SHARED_KERNELS).
_RECORDS = {"pair": (_RECORD_TOKEN_STAIL, "default", _RECORD_STAIL), "1cta": (_RECORD_TOKEN_1CTA, "grouped_thd", _RECORD_1CTA)}


def _record(chain):
    """(dsl line, {kernel: md5}) of the chain's committed default record."""
    token, variant, path = _RECORDS[chain]
    dsl, want = read_kernel_record(path, token)
    return dsl, want.get(variant, {})


# ------------------------------------------------------------------------------------------------------------ host (no GPU)
def test_dkv_accum_domain_is_typed():
    """The vocabulary, the default, the resolver's declines BY NAME, the record default on the adapter, the Constexpr it folds to."""
    from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128
    from cudnn.sdpa.bwd.d128_grouped_contract import DKV_ACCUM_DEFAULT, DKV_ACCUM_FORMS, dkv_accum_fp32, resolve_dkv_accum

    assert DKV_ACCUM_FORMS == FORMS and DKV_ACCUM_DEFAULT == "bf16_partials" == FORMS[0]
    assert SdpaBwdDslSm100D128.dkv_accum is None  # the record default: unset = bf16_partials = the shipping chain, byte-identical
    assert resolve_dkv_accum(None) == "bf16_partials" and resolve_dkv_accum("bf16_partials") == "bf16_partials"
    assert resolve_dkv_accum("fp32_reduce") == "fp32_reduce"
    assert dkv_accum_fp32(None) is False and dkv_accum_fp32("bf16_partials") is False and dkv_accum_fp32("fp32_reduce") is True
    for bad in ("fp32", "FP32_REDUCE", "bf16", "", 1, True, b"fp32_reduce"):
        with pytest.raises(ValueError, match="dkv_accum"):
            resolve_dkv_accum(bad)
        with pytest.raises(ValueError, match="dkv_accum"):
            dkv_accum_fp32(bad)


def test_dkv_accum_config_is_the_truth():
    """Every number the fp32 store path uses is a config field: the 1-CTA body's four [M x DQ_BOX] fp32 slices fill exactly the 64 KiB ALIAS slab
    (= one fp32 [M x D] tile); the pair body's four slots are the dV / dK bf16 staging bytes re-cut (K2's two halves + ds_own(rank) + XCHG),
    1024-B aligned, disjoint, 16 KiB each; the fp32 workspace per KV head fits inside the bf16 per-Q-head partial region on every grouped ratio."""
    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as stail
    from cudnn.sdpa.bwd.config_sm100_d128_accum import CFG
    from cudnn.sdpa.bwd.d128_grouped_contract import ACCUM_ITEMSIZE, PARTIAL_ITEMSIZE, thd_fp32_accum_bytes

    assert CFG.BPE_ACC == ACCUM_ITEMSIZE == 4 and PARTIAL_ITEMSIZE == 2
    assert CFG.DKV_ACC_SLICES == CFG.D // CFG.DQ_BOX == 4 and CFG.DKV_ACC_SLICE_BYTES == CFG.M * CFG.DQ_BOX * CFG.BPE_ACC == CFG.SLICE_BYTES
    assert CFG.DKV_ACC_SLICES * CFG.DKV_ACC_SLICE_BYTES == 2 * CFG.OPERAND_BYTES == CFG.M * CFG.D * CFG.BPE_ACC  # the ALIAS slab = one fp32 tile
    assert [CFG.dkv_acc_slot(i) for i in range(CFG.DKV_ACC_SLICES)] == [CFG.ALIAS + i * CFG.SLICE_BYTES for i in range(4)]
    assert (CFG.DV_COL, CFG.DK_COL) == (256, 384) and CFG.DK_COL + CFG.D <= CFG.TMEM_COLS
    c = stail.CFG
    assert c.DKV_ACC_SLICES == c.D // c.DQ_BOX == 4 and (c.DKV_ACC_R1_SLICES, c.DKV_ACC_R2_SLICES) == (5, 3)
    assert c.DKV_ACC_SLICE_BYTES == c.M * c.DQ_BOX * c.BPE_ACC == 16384
    r1 = [c.dkv_acc_r1_slot(i) for i in range(c.DKV_ACC_R1_SLICES)]
    r2 = [c.dkv_acc_r2_slot(i) for i in range(c.DKV_ACC_R2_SLICES)]
    assert r1 == [c.K2, c.K2 + c.DKV_ACC_SLICE_BYTES, c.DS, c.DS + c.DS_HALF_BYTES, c.XCHG] and r2 == r1[:3], (r1, r2)
    assert all(s % 1024 == 0 and c.K2 <= s and s + c.DKV_ACC_SLICE_BYTES <= c.DQ for s in r1) and len(set(r1)) == 5  # inside [K2, DQ): never the sDQ ring
    spans = sorted((s, s + c.DKV_ACC_SLICE_BYTES) for s in r1)
    assert all(a_end <= b_start for (_, a_end), (b_start, _) in zip(spans, spans[1:]))  # disjoint
    assert {r1[2], r1[3]} == {c.ds_own(0), c.ds_recv(0)}  # both dS halves: dead from dq_done(hi-1) until the next item's dS writes (the dvk_empty chain)
    rows = {row[0]: (row[3], row[4]) for row in stail.BAR_LAYOUT}
    assert rows["dvk_stage_empty"] == (c.STORE_LANES, "thread") and stail.BAR_LAYOUT[-1][0] == "dvk_stage_empty"  # the NEW row: store -> epilogue
    assert thd_fp32_accum_bytes(1024, 4) == 1024 * 4 * 128 * 4
    for heads, kv_heads in ((2, 1), (4, 2), (6, 2), (8, 1), (32, 4)):
        assert thd_fp32_accum_bytes(2048, kv_heads) <= 2048 * heads * 128 * PARTIAL_ITEMSIZE  # the fp32 accumulator fits the partial region


@pytest.mark.parametrize("chain", ["1cta", "pair"])
def test_dkv_accum_default_renderings_are_unchanged_and_bf16_partials_named_is_the_default(tmp_path, chain):
    """The pin: with the field unset BOTH chains render the committed record md5 (the 1-CTA grouped THD record of the diag module at 4 / 32 / 4,
    the S-tail record at 2 / 8 / 2), and ``dkv_accum="bf16_partials"`` named explicitly renders the SAME bytes (skips, never widens, on a DSL
    build that differs from the record's)."""
    dsl, want = _record(chain)
    assert dsl and want, (chain, dsl, want)
    _skip_unless_recorded_dsl(dsl)
    md5_default, census_default, kernels_default = _render_kernels(tmp_path, chain)
    md5_named, census_named = _render(tmp_path, chain, **{FIELD: "bf16_partials"})
    print(f"\n{chain} default: {md5_default} {census_default} kernels {kernels_default}")
    token, variant, _path = _RECORDS[chain]
    assert_kernel_record(kernels_default, want, token=token, variant=variant, excluded=D128_THD_SHARED_KERNELS, what=f"{chain}: the DEFAULT rendering")
    assert md5_named == md5_default and census_named == census_default, f"{chain}: bf16_partials named must render the default bytes"
    assert census_default["ENTRY_FOLD"] == 1 and census_default["ENTRIES"] == 5


@pytest.mark.parametrize("chain", ["1cta", "pair"])
def test_dkv_accum_fp32_renders_for_sm_100a_with_the_documented_census(tmp_path, chain):
    """``fp32_reduce`` renders a DIFFERENT image with exactly the documented deltas (kernel header section 11 of each body): +8 tensor reduce-adds,
    -4 bf16 tile stores, the fold entry gone (4 kernels), the store warp's post-dV-read ring credits and the epilogue's dK WAR waits, the P7
    groups; no local memory, no nanosleep."""
    md5_default, default = _render(tmp_path, chain)
    md5_fp32, fp32 = _render(tmp_path, chain, **_FP32)
    print(f"\n{chain} fp32_reduce: {md5_fp32} {fp32}")
    assert md5_fp32 != md5_default, f"{chain}: the fp32 rendering must differ from the default (the field folded to nothing)"
    for key, (want_default, want_fp32) in _EXPECTED_FP32_DELTA[chain].items():
        assert default[key] == want_default, f"{chain} default {key}: {default[key]} != {want_default}"
        assert fp32[key] == want_fp32, f"{chain} fp32_reduce {key}: {fp32[key]} != {want_fp32}"
    assert fp32["ENTRY_FOLD"] == 0 and fp32["ENTRIES"] == 4 and default["ENTRIES"] == 5
    assert fp32["LOCAL"] == 0 and fp32["NANOSLEEP"] == 0


def test_dkv_accum_declines_by_name_at_the_pair_chain_entry():
    """A value outside the domain and the one unbuilt combination (``fp32_reduce`` under ``warp_roles = "16w_fa4"``) decline BY NAME at the pair
    chain's compile_host before any trace."""
    from cudnn.sdpa.bwd.kernels.sm100.prepared_d128_grouped_2cta_host import compile_host

    with pytest.raises(ValueError, match="dkv_accum must be one of"):
        compile_host((2, 8, 2, True), (None,) * 9, (None,) * 9, None, True, 100, dkv_accum="fp32")
    with pytest.raises(ValueError, match="dkv_accum 'fp32_reduce' is not built under warp_roles '16w_fa4'"):
        compile_host((2, 8, 2, True), (None,) * 9, (None,) * 9, None, True, 100, warp_roles="16w_fa4", **_FP32)


def _code_lines(text):
    """Statement lines only: the docstring, blank, comment-only lines and inline comments dropped (a source assertion reads code, not prose)."""
    doc = re.search(r'"""[\s\S]*?"""', text)
    if doc:
        text = text[: doc.start()] + text[doc.end() :]
    return [ln.split("#")[0].rstrip() for ln in text.splitlines() if ln.split("#")[0].strip() and not ln.strip().startswith("@")]


def test_dkv_accum_fp32_store_path_credits_the_ring_after_the_dv_reads():
    """SOURCE assertion (the audit's rows): the 1-CTA store path issues the dV slices, then ``tma_store_wait`` -> the ``dq_empty[i].arrive()``
    releases BEFORE the dk_full wait (its bootstrapped ring carries the dK staging's WAR edge) and its epilogue's dK round waits them before it
    stores; the pair's rounds go through the NEW row ``dvk_stage_empty`` (commit -> drain -> arrive -> the dk_full wait; the epilogue waits it
    between dv_full and dk_full) and never touch the pre-armed dq_empty ring; the pair's new row is initialised only under the field."""
    import cudnn.sdpa.bwd.kernels.sm100.bprop_d128_accum_grouped_2cta_stail_f16 as pair_body
    import cudnn.sdpa.bwd.kernels.sm100.bprop_d128_accum_thd_f16 as one_body

    one = pathlib.Path(one_body.__file__).read_text()
    store = "\n".join(_code_lines(one[one.index("def _store_loop(") : one.index("def _scheduler_loop(")]))
    fp32 = store[store.index("const_expr(dkv_fp32)") : store.index("else:", store.index("const_expr(dkv_fp32)"))]
    order = [fp32.index("tma_store_wait("), fp32.index("dq_empty[i].arrive()"), fp32.index("dk_full")]
    assert order == sorted(order), f"1-CTA: the post-dV credits must precede the dk_full wait: {order}"
    epi = "\n".join(_code_lines(one[one.index("def _native_epilogue_fp32(") : one.index("def _epilogue_loop(")]))
    assert epi.index("_wait(bars.dq_empty[i], empty_phase)") < epi.index("_store_dq_slice(slab, row, vals, i)"), "1-CTA: the dK round waits before it stores"
    pair = pathlib.Path(pair_body.__file__).read_text()
    rounds = "\n".join(_code_lines(pair[pair.index("def _dkv_fp32_reduce_rounds(") : pair.index("def _ts(")]))
    order = [rounds.index("tma_store_commit()"), rounds.index("tma_store_wait(0)"), rounds.index("dvk_stage_empty.arrive()"), rounds.index("dk_full")]
    assert order == sorted(order), f"pair: commit -> drain -> the dvk_stage_empty release -> the dk_full wait: {order}"
    assert "dq_empty" not in rounds, "pair: the fp32 rounds never touch the pre-armed dq_empty ring"
    stage = "\n".join(_code_lines(pair[pair.index("def _dkv_fp32_stage(") : pair.index("def _dkv_fp32_reduce_rounds(")]))
    assert stage.index("dv_full.arrive()") < stage.index("_wait(bars.dvk_stage_empty, stage_state.phase") < stage.index("dk_full.arrive()")
    assert stage.rstrip().endswith("return advance(stage_state, 1)") and "dq_empty" not in stage, "pair: one dvk_stage_empty flip per item, the ring untouched"
    assert re.search(r"_native_epilogue_fp32\(", one) and re.search(r"_stage_fp32_slice\(", pair)
    init = _code_lines(pair[pair.index("def bprop_d128_accum_grouped_2cta_stail_kernel(") : pair.index("def main_thd_2cta_stail_host(")])
    assert any("dkv_fp32 or i != DVK_STAGE_EMPTY_ROW" in ln for ln in init), "pair: the new row's init is guarded by the field (no user under the default)"


# ------------------------------------------------------------------------------------------------------------ GPU rows
def _make_1cta(case, *, compile_plan=True, **fields):
    api = make_adapter(case, compile_plan=False)
    for name, value in fields.items():
        setattr(api, name, value)
    if not compile_plan:
        return api
    assert api.check_support()
    api.compile()
    return api


def _make_pair(case, *, compile_plan=True, **fields):
    return make_adapter_2cta_stail(case, compile_plan=compile_plan, **fields)


_BODIES = {"1cta": _make_1cta, "pair": _make_pair}
_BODY_IDS = [pytest.param("1cta", id="1cta"), pytest.param("pair", id="pair")]


def _run(case, body, repeats=2, **fields):
    api = _BODIES[body](case, **fields)
    workspace = make_workspace(api.scratch_workspace_bytes())
    for _ in range(repeats):
        poison(case, workspace)
        api.execute(**adapter_arguments(case), workspace=workspace.tensor)
        torch.cuda.synchronize()
        check_case(case, workspace)
    return api, workspace


def _grads(case):
    return {role: case.tensors[role][0].detach().clone() for role in ("dq", "dk", "dv")}


def _max_err(case, grads, role):
    """(max abs error, max error in bf16 ulps at the reference magnitude) of ``role`` over the live rows against the fp64 oracle sequences."""
    worst_abs, worst_ulp = 0.0, 0.0
    lens = case.lens_q if role == "dq" else case.lens_kv
    for i, (start, length) in enumerate(zip(case.origins[role], lens, strict=True)):
        if not length:
            continue
        got, ref = grads[role][start : start + length].float().cpu(), case.sequences[i][role].float()
        err = (got - ref).abs()
        ulp = torch.ldexp(torch.ones_like(ref), torch.floor(torch.log2(ref.abs().clamp_min(2.0**-126))) - 7)  # one bf16 ulp at |ref|
        worst_abs = max(worst_abs, float(err.max()))
        worst_ulp = max(worst_ulp, float((err / ulp).max()))
    return worst_abs, worst_ulp


@pytest.mark.parametrize("body", _BODY_IDS)
@pytest.mark.parametrize(
    "lq,lk,hq,hkv,causal,layout,offset_dtype,units,envelope",
    [
        pytest.param((129, 0, 127, 1), (65, 128, 0, 3), 4, 2, True, "TH1", torch.int32, "tokens", None, id="suite-degenerate-causal"),
        pytest.param((129, 0, 127, 1), (65, 128, 0, 3), 6, 2, False, "HN", torch.int64, "elements", None, id="suite-degenerate-noncausal"),
        pytest.param((0, 0), (0, 0), 3, 1, True, "HN", torch.int64, "elements", (128, 128), id="all-empty"),
        pytest.param((300, 128, 1000), (2000, 700, 1000), 4, 2, True, "TH1", torch.int32, "tokens", None, id="zero-trip-mix"),
        pytest.param((300, 300), (300, 300), 3, 1, True, "TH1", torch.int64, "elements", None, id="tail-300-gqa3"),
        pytest.param((1024,), (1024,), 8, 1, True, "HN", torch.int32, "tokens", None, id="uniform-1024-gqa8"),
    ],
)
def test_dkv_accum_fp32_matrix(body, lq, lk, hq, hkv, causal, layout, offset_dtype, units, envelope):
    """The fp64-oracle ladder under ``fp32_reduce`` on both bodies: S % 256 != 0 tails, an all-empty batch, the zero-trip mix (items with N = 3, 1,
    0 between live ones), n_q == 1 items, the 8-head group; every row at the suite's own gate (``_TOL`` / cosine), two executes per case."""
    _require_accumulation()
    case = make_case(
        lq, lk, hq=hq, hkv=hkv, thd=True, dtype=torch.bfloat16, causal=causal, stats_layout=layout, offset_dtype=offset_dtype, units=units, envelope=envelope
    )
    api, _ = _run(case, body, **_FP32)
    assert api._prepared.artifact.dkv_accum == "fp32_reduce"


@pytest.mark.parametrize("body", _BODY_IDS)
def test_dkv_accum_fp32_n_q_one_more_items_than_sms(body):
    """P14 row: n_q == 1 with more items than SMs (a CTA / pair claims a second CLC item: the two fp32 rounds per item re-arm the ring across
    items), a half-dead last block and zero-trip blocks past seq_q."""
    _require_accumulation()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    hq, b = 8, 2
    skv = 256 * ((sms // (hq * b)) + 2) + 128
    case = make_case((128, 64), (skv, skv - 256), hq=hq, hkv=1, dtype=torch.bfloat16, causal=True, stats_layout="TH1")
    _run(case, body, **_FP32)


@pytest.mark.parametrize("body", _BODY_IDS)
def test_dkv_accum_fp32_launch_census_is_four_with_no_host_work(body, monkeypatch):
    """The fold launch is gone: four ``cudnn_kernel_`` launches (setup, delta_zero, MAIN, cast_dq) and no fold entry; the R9 gate (no allocation,
    no host read, no Python tensor plumbing over three executes) holds as on the default chain."""
    _require_accumulation()
    import cutlass.cute as cute
    from cudnn.sdpa.bwd import prepared

    case = make_case((256, 256), (256, 256), hq=4, hkv=2, thd=True, dtype=torch.bfloat16, causal=True)
    api, workspace = _run(case, body, repeats=1, **_FP32)
    args = adapter_arguments(case)
    torch.cuda.synchronize()
    sync_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        before = torch.cuda.memory_stats()["allocation.all.allocated"]

        def forbidden(*a, **k):
            raise AssertionError("the fp32_reduce chain used Python tensor plumbing, allocation, host read or compilation on the execute path")

        with monkeypatch.context() as patch:
            patch.setattr(cute, "compile", forbidden)
            patch.setattr(prepared, "facts_of_roles", forbidden)
            for name in ("empty", "empty_like", "zeros", "zeros_like", "ones", "full", "tensor"):
                patch.setattr(torch, name, forbidden)
            for name in ("view", "reshape", "as_strided", "permute", "transpose", "cpu", "item", "tolist"):
                patch.setattr(torch.Tensor, name, forbidden)
            for _ in range(3):
                api.execute(**args, workspace=workspace.tensor)
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    finally:
        torch.cuda.set_sync_debug_mode(sync_mode)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        api.execute(**args, workspace=workspace.tensor)
        torch.cuda.synchronize()
    kernels = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA and e.name.startswith("cudnn_kernel_")]
    assert len(kernels) == 4, kernels
    assert not any("dkv_reduce" in n for n in kernels), kernels
    # the delta launch by the chain's stats form: the compact `delta_zero_thd_d128_kernel` (1-CTA; the pair under ldg) or the padded-stats
    # `delta_zero_stats_thd_d128_2cta_kernel` (the pair's default tma_ring form)
    assert any("cast_dq_thd_d128_kernel" in n for n in kernels) and any("delta_zero" in n for n in kernels), kernels
    check_case(case, workspace)


@pytest.mark.parametrize("body", _BODY_IDS)
def test_dkv_accum_fp32_dq_bitwise_vs_default_on_one_kv_block_and_dk_dv_recorded(body):
    """Twin vs the default on ONE dataset: dQ is untouched by the field, so where ONE kv block feeds every dQ element (S_kv <= 128 for the 1-CTA
    body, <= 256 for the pair: one reduce-add contributor into the zero-filled fp32 image) it is BITWISE equal by construction; dK / dV are a
    DIFFERENT rounding (fp32 group accumulate vs 8 bf16 partials summed in fp32) and are gated by the suite's own tolerance under each arm,
    their max error against the fp64 oracle recorded side by side (never a new tolerance)."""
    _require_accumulation()
    lk = (128, 100) if body == "1cta" else (256, 200)
    case = make_case((300, 200), lk, hq=8, hkv=2, thd=True, dtype=torch.bfloat16, causal=False, stats_layout="TH1")
    grads = {}
    for tag, fields in (("bf16_partials", {}), ("fp32_reduce", _FP32)):
        _run(case, body, repeats=1, **fields)
        grads[tag] = _grads(case)
    assert torch.equal(grads["bf16_partials"]["dq"].view(torch.int16), grads["fp32_reduce"]["dq"].view(torch.int16)), "dQ must be bitwise: one kv block"
    for role in ("dk", "dv"):
        a, b = _max_err(case, grads["bf16_partials"], role), _max_err(case, grads["fp32_reduce"], role)
        print(f"\n{body} {role}: max|err| bf16_partials {a[0]:.3e} ({a[1]:.2f} bf16 ulp) | fp32_reduce {b[0]:.3e} ({b[1]:.2f} bf16 ulp)")
        for i, (start, length) in enumerate(zip(case.origins[role], case.lens_kv, strict=True)):
            x, y = grads["bf16_partials"][role][start : start + length].float(), grads["fp32_reduce"][role][start : start + length].float()
            torch.testing.assert_close(y, x, **_TOL[case.dtype], msg=lambda m: f"{role}/seq{i}: bf16_partials vs fp32_reduce: {m}")


@pytest.mark.parametrize("body", _BODY_IDS)
def test_dkv_accum_fp32_dq_at_the_gate_vs_default_over_many_kv_blocks(body):
    """Several kv blocks per dQ element: dQ differs between the arms only by the fp32 add ORDER of the same reduce-adds (never bitwise, a tolerance
    twin at the suite's gate after each arm passed the fp64 gate); dK / dV at the gate as well, the max errors recorded."""
    _require_accumulation()
    case = make_case((512, 768), (512, 768), hq=4, hkv=2, thd=True, dtype=torch.bfloat16, causal=True, stats_layout="TH1")
    grads = {}
    for tag, fields in (("bf16_partials", {}), ("fp32_reduce", _FP32)):
        _run(case, body, repeats=1, **fields)
        grads[tag] = _grads(case)
    for role in ("dq", "dk", "dv"):
        lens = case.lens_q if role == "dq" else case.lens_kv
        a, b = _max_err(case, grads["bf16_partials"], role), _max_err(case, grads["fp32_reduce"], role)
        print(f"\n{body} {role}: max|err| bf16_partials {a[0]:.3e} ({a[1]:.2f} bf16 ulp) | fp32_reduce {b[0]:.3e} ({b[1]:.2f} bf16 ulp)")
        for i, (start, length) in enumerate(zip(case.origins[role], lens, strict=True)):
            x, y = grads["bf16_partials"][role][start : start + length].float(), grads["fp32_reduce"][role][start : start + length].float()
            torch.testing.assert_close(y, x, **_TOL[case.dtype], msg=lambda m: f"{role}/seq{i}: bf16_partials vs fp32_reduce: {m}")


def _make_mha_1cta(case, **fields):
    """The MHA THD suite's adapter construction (d128_thd_test_utils.make_adapter) with the typed fields set BEFORE check_support()."""
    from cudnn.api_base import TensorDesc
    from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128
    from d128_thd_test_utils import ROLES as MHA_ROLES
    from d128_thd_test_utils import SCALE as MHA_SCALE
    from d128_thd_test_utils import declaration as mha_declaration

    samples = {}
    for role in MHA_ROLES:
        shape, stride = mha_declaration(case, role)
        samples["sample_" + role] = TensorDesc(
            dtype=torch.float32 if role == "stats" else case.dtype,
            shape=shape,
            stride=stride,
            stride_order=tuple(sorted(range(4), key=lambda i: (stride[i], shape[i]))),
            device=case.tensors["q"].device,
            name=role,
        )
    api = SdpaBwdDslSm100D128(
        **samples,
        is_causal=case.causal,
        scale_softmax=MHA_SCALE,
        thd=True,
        max_total_seq_len_q=case.capacities[0],
        max_total_seq_len_kv=case.capacities[1],
        thd_stats_token_major=case.stats_layout == "TH1",
        thd_stats_head_stride=case.head_pitch if case.stats_layout == "HN" else None,
        thd_ragged_offsets=case.offset_declarations,
    )
    for name, value in fields.items():
        setattr(api, name, value)
    return api


def test_dkv_accum_fp32_declines_dense_grouped_and_mha_by_name():
    """The fp32 form is served on the THD grouped chain only: the dense grouped route and the MHA routes decline it BY NAME in check_support,
    before any compile; a value outside the vocabulary is a ValueError there."""
    from d128_thd_test_utils import make_case as make_mha_case

    _require_accumulation()
    dense = make_case((256, 256), (256, 256), hq=4, hkv=2, thd=False)
    with pytest.raises(NotImplementedError, match="fp32_reduce"):
        _make_1cta(dense, compile_plan=False, **_FP32).check_support()
    with pytest.raises(NotImplementedError, match="fp32_reduce"):
        _make_mha_1cta(make_mha_case((129, 3), (129, 3), heads=2), **_FP32).check_support()
    with pytest.raises(ValueError, match="dkv_accum"):
        _make_1cta(make_case((129, 3), (65, 3), hq=4, hkv=2, thd=True), compile_plan=False, dkv_accum="fp32").check_support()
