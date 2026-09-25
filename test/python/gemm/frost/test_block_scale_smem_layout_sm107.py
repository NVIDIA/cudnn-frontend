# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rubin (sm107) block-scale GEMM: the SMEM ring layout vs the 14-bit
``Tcgen05SmemDesc.build()`` start-address field.  NO GPU.

``Tcgen05SmemDesc.build()`` lowers through the non-versioned ``_tcgen05_mma_smem_desc``
intrinsic (cutlass-dsl ``experimental/primitives/descriptors.py:513-522``) and keeps
14 bits of ``addr >> 4``: a root at or above 262144 B wraps to the bottom of SMEM with
no error.  ``advance_start_address`` (``:413-425``) is an unmasked encoded add, so only
the ring ROOTS -- the ``build()`` operands -- are exposed.  The sm100 block-scale
templates used to declare their rings A, B, SFA, SFB; on Rubin's 327 KiB carveout
(``334848`` B budget) the SF ring roots landed at 258-302 KiB, the SF UTCCP copied
A-operand bytes into the SF TMEM columns, and ~100 % of the outputs were NaN/inf --
60 of the 80 ``-k sm107`` block-scale GPU tests.  The GPU sweep that established the
rule (mxfp8 128x128, M=N=256 K=512, forced ``ab_stages``, Rubin dev node 2026-09-15):

    stages   SFA root   verdict
       9      296960    FAIL
       8      264192    FAIL
       7      231424    PASS (bit-exact)
       6      198656    PASS

and the no-GPU predictor "FAIL iff any build() root >= 262144" reproduced that
suite's 60 failed / 20 passed partition exactly.  Fix: declare the (small) SF rings
BEFORE the A/B rings, so every root sits far below the line and only the per-stage
``advance_start_address`` offsets cross it; the renderer models the roots
(``_block_scale_smem_desc_roots``), TRIMS ``ab_stages`` where a deeper ring would
still put one past the line (only the MoE template needs it: its TMA-store staging
buffer sits ahead of the rings, and at sm107 512x128 nvfp4 a 192 KiB A ring behind it
puts B[0] at 263168 with 3 stages), and refuses a layout a single stage cannot fit.

This module pins all of it without a GPU: the arch is pinned to 107 and the SMEM
budget to Rubin's, the block-scale tile constants are rendered (no ``cute.compile``),
and the SMEM arrays are laid out in the templates' declaration order -- the
predictor above, kept here as the independent model the compiler is checked against.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

import cudnn
import cudnn.gemm.frost  # noqa: F401  (installs recorder)
from cudnn.gemm.frost import arch_family as AF
from cudnn.gemm.frost import compiler as C
from cudnn.gemm.frost import tile_config as TC
from cudnn.gemm.frost.graph_analyzer import analyze_with_binding
from cudnn.gemm.frost.kernel_registry import select_template
from cudnn.gemm.frost.sm100 import compiler as C100
from cudnn.gemm.frost.tile_config import CATALOG, by_name

pytestmark = pytest.mark.L0

# max(optin 227 KiB, oversized 327 KiB) -- what tile_config._sm_smem_budget_bytes_of
# reports on cc 10.7, and the budget every failing GPU test was sized against.
RUBIN_BUDGET = 334848
SM100_BUDGET = 227 * 1024
WRAP = C100._TCGEN05_DESC_BUILD_ADDR_LIMIT

_TEMPLATES_DIR = Path(C100.__file__).resolve().parent / "kernel_templates"
_DENSE_TEMPLATE = "sm100_block_scale_matmul.py"
_MOE_TEMPLATE = "sm100_moe_grouped_block_scale_matmul_fwd.py"

# (block size, data dtype, scale dtype) -- the three combos the GPU suites run.
_COMBOS = {
    "nvfp4": (16, cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E4M3),
    "mxfp4": (32, cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E8M0),
    "mxfp8": (32, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E8M0),
}

_BPE = {
    "cutlass.Float16": 2,
    "cutlass.BFloat16": 2,
    "cutlass.Float32": 4,
    "cutlass.Int8": 1,
    "cutlass.Uint8": 1,
    "cutlass.Float8E4M3FN": 1,
    "cutlass.Float8E5M2": 1,
}


# --- no-GPU rendering -------------------------------------------------------------


def _pin(monkeypatch, *, arch: int, budget: int) -> None:
    """Render for ``arch`` at ``budget`` bytes of SMEM whatever GPU (if any) is present."""
    monkeypatch.setattr(AF, "current_arch", lambda device=None: arch)
    monkeypatch.setattr(C, "_current_arch", lambda device=None: arch)
    monkeypatch.setattr(C100, "_current_arch", lambda device=None: arch)
    monkeypatch.setattr(TC, "_sm_smem_budget_bytes", lambda device=None: budget)
    monkeypatch.setattr(TC, "_sm_smem_budget_bytes_of", lambda device: budget)
    # Rendered as constants only; keep the render off the device.
    monkeypatch.setattr(TC, "l2_swizzle_budget_bytes", lambda device=None: 42 << 20)
    monkeypatch.setattr(C100, "_grid_num_clusters", lambda cfg, device=None: 132)


@pytest.fixture
def rubin(monkeypatch):
    _pin(monkeypatch, arch=107, budget=RUBIN_BUDGET)


@pytest.fixture
def blackwell(monkeypatch):
    _pin(monkeypatch, arch=100, budget=SM100_BUDGET)


def _dense_graph(combo: str, M: int = 256, N: int = 256, K: int = 512):
    block, dt, sf_dt = _COMBOS[combo]
    sf_k = K // block
    g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1], data_type=dt)
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K], data_type=dt)
    sf_kw = dict(reordering_type=cudnn.tensor_reordering.F8_128x4)
    SFA = g.tensor(name="SFA", dim=[1, M, sf_k], stride=[M * sf_k, sf_k, 1], data_type=sf_dt, **sf_kw)
    SFB = g.tensor(name="SFB", dim=[1, sf_k, N], stride=[sf_k * N, 1, sf_k], data_type=sf_dt, **sf_kw)
    Ad = g.block_scale_dequantize(input=A, descale=SFA, block_size=[1, block])
    Bd = g.block_scale_dequantize(input=B, descale=SFB, block_size=[block, 1])
    out = g.matmul(A=Ad, B=Bd, name="mm")
    out.set_output(True).set_data_type(cudnn.data_type.HALF)
    return g


def _moe_graph(combo: str, E: int = 2, S: int = 512, N: int = 256, K: int = 512, num_groups: int = 2):
    block, dt, sf_dt = _COMBOS[combo]
    sf_k = K // block
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=dt)
    w = g.tensor(name="weight", dim=[E, K, N], stride=[K * N, 1, K], data_type=dt)
    sf_kw = dict(reordering_type=cudnn.tensor_reordering.F8_128x4)
    SFA = g.tensor(name="SFA", dim=[1, S, sf_k], stride=[S * sf_k, sf_k, 1], data_type=sf_dt, **sf_kw)
    SFB = g.tensor(name="SFB", dim=[E, sf_k, N], stride=[sf_k * N, 1, sf_k], data_type=sf_dt, **sf_kw)
    fto = g.tensor(name="first_token_offset", dim=[num_groups, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    tok_d = g.block_scale_dequantize(input=tok, descale=SFA, block_size=[1, block])
    w_d = g.block_scale_dequantize(input=w, descale=SFB, block_size=[block, 1])
    out = g.moe_grouped_matmul(tok_d, w_d, fto, mode=cudnn.moe_grouped_matmul_mode.NONE, compute_data_type=cudnn.data_type.FLOAT, name="moe")
    out.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    return g


def _chain(graph):
    chain, _binding = analyze_with_binding(graph)
    return chain


def _render(cfg, chain):
    """The block-scale tile constants the loader would inject -- no cute.compile."""
    tmpl = select_template(chain, cfg)
    reject = tmpl.accepts(chain, cfg)
    assert reject is None, reject
    return tmpl, C100._render_block_scale_tile_constants(cfg, chain, tmpl)


def _consts(txt: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in re.finditer(r"^([A-Za-z_][A-Za-z0-9_]*) = ([^\n#]+?)\s*(?:#.*)?$", txt, re.M):
        out.setdefault(m.group(1), m.group(2).strip())
    return out


_ROOTS_LINE = re.compile(r"^# tcgen05 SMEM descriptor roots \(.*?\): (.*)$", re.M)
_TRIM_LINE = re.compile(r"^# ab_stages trimmed from (\d+) to (\d+):", re.M)


def _rendered_roots(txt: str) -> dict[str, int]:
    """The roots the COMPILER modelled and checked, as it renders them beside ab_stages."""
    m = _ROOTS_LINE.search(txt)
    assert m, "the renderer no longer emits its descriptor-root line"
    return {name: int(off) for name, off in (tok.split("=") for tok in m.group(1).split())}


def _trim(txt: str) -> tuple[int, int] | None:
    """(budget stages, rendered stages) when the renderer trimmed the ring depth, else None."""
    m = _TRIM_LINE.search(txt)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _budget_stages(consts: dict[str, str], tmpl, chain, cfg) -> int:
    """The depth the SMEM budget alone allows -- what the renderer starts from."""
    na, nb, nsa, nsb = (int(consts[k]) for k in ("num_a_operands", "num_b_operands", "num_sfa_operands", "num_sfb_operands"))
    per_stage = (
        na * int(consts["sA_packed_elems"]) + nb * int(consts["sB_packed_elems"]) + nsa * int(consts["sfa_smem_bytes"]) + nsb * int(consts["sfb_smem_bytes"])
    )
    reserved = C100._smem_d_bytes(cfg, chain) if C100._use_tma_store_epi(chain, cfg) else 0
    return max(1, TC.smem_ab_stages(per_stage, smem_fixed_reserve=tmpl.smem_fixed_reserve, extra_smem_bytes=reserved))


# --- the predictor: lay the template's SMEM declarations out in order --------------


def _align(x: int, a: int) -> int:
    return -(-x // a) * a


def _model_roots(consts: dict[str, str], template: str, *, tma_store: bool, sf_first: bool = True, stages: int | None = None) -> dict[str, int]:
    """Byte offset of every ring root, walking the template's ``cutlass.Array``
    declarations in source order (the DSL allocates SMEM sequentially, honouring each
    array's alignment).  Exactly the model that predicted the 60/20 GPU partition;
    ``sf_first=False`` is the pre-fix order, ``stages`` overrides ``ab_stages`` for
    the forced-depth sweep."""
    st = int(consts["ab_stages"]) if stages is None else stages
    acc = int(consts["acc_stages"])
    na, nb, nsa, nsb = (int(consts[k]) for k in ("num_a_operands", "num_b_operands", "num_sfa_operands", "num_sfb_operands"))
    # Packed elements are 8-bit (Float4E2M1FNx2 / Uint8 / Float8), so these ARE the stage bytes.
    sA, sB = int(consts["sA_packed_elems"]), int(consts["sB_packed_elems"])
    sfa, sfb = int(consts["sfa_smem_bytes"]), int(consts["sfb_smem_bytes"])

    off = 1024  # _smem_sys_reserved
    off += 8 * (3 * st + 2 * acc + 1)  # ab_full / sf_full / ab_empty / acc_empty / acc_full / tmem_dealloc mbars
    off += 4  # tmem_ptr_i32
    if template == _DENSE_TEMPLATE:
        off = _align(off, 16) + 16  # _clc_response_raw (CLC_SCHED_STAGES = 1)
        off += 8 + 8  # clc_full / clc_empty mbars
    else:
        assert template == _MOE_TEMPLATE, template
        off = _align(off, 16) + 4 * 2 * 8  # sched_storage: SCHED_STAGES x SCHED_SLOT_WORDS i32
        off += 8 * 2 + 8 * 2  # sched_full / sched_empty mbars
        off = _align(off, 16) + 4 * 2  # sched_bcast_slot
        off += 8 * 2 + 8 * 2  # sched_bcast_full / sched_bcast_empty mbars
        off = _align(off, 128) + 128 * (na + nsa)  # per-operand TMA tensormap scratch (16 qwords each)
        if tma_store:  # MoE stages its TMA-store buffer AHEAD of the rings
            smem_d = 2 * int(consts["epi_stage_rows"]) * int(consts["epi_row_elems"]) * int(consts["epi_slot_widen"]) * _BPE[consts["cd_dtype"]]
            off = _align(off, 1024) + smem_d
            off = _align(off, 128) + 128 * int(consts["n_tma_outputs"])

    roots: dict[str, int] = {}

    def ring(name: str, nbytes: int) -> None:
        nonlocal off
        off = _align(off, 1024)
        roots[name] = off
        off += nbytes

    sf = [(f"SFA[{i}]", sfa * st) for i in range(nsa)] + [(f"SFB[{j}]", sfb * st) for j in range(nsb)]
    ab = [(f"A[{i}]", sA * st) for i in range(na)] + [(f"B[{j}]", sB * st) for j in range(nb)]
    for name, nbytes in sf + ab if sf_first else ab + sf:
        ring(name, nbytes)
    return roots


def _wrapped(roots: dict[str, int]) -> dict[str, int]:
    return {name: off for name, off in roots.items() if off >= WRAP}


def _cross_check(rendered: dict[str, int], model: dict[str, int], *, slack: int) -> None:
    """The compiler's roots must agree with the predictor: same rings, never BELOW the
    exact layout (its header is the template's fixed reserve, an upper bound) and
    never more than one alignment quantum above it."""
    assert rendered.keys() == model.keys(), (rendered, model)
    for name in model:
        assert 0 <= rendered[name] - model[name] <= slack, (name, rendered[name], model[name])


# --- the configs the sm107 GPU suites run -------------------------------------------


def _sm107_suite_configs() -> list[str]:
    """Every (geometry, cta_group) `pytest gemm/frost/test_block_scale_matmul.py -k sm107`
    plans, spelled as catalog names (superset of the parametrizations there)."""
    names = [
        "CONFIG_sm100_128x128x128_128x128x64_cluster1x1_1ctamma",
        "CONFIG_sm100_128x256x128_128x256x64_cluster1x1_1ctamma",
        "CONFIG_sm100_128x128x128_128x128x64_cluster1x2_1ctamma",
        "CONFIG_sm100_128x128x128_128x128x64_cluster2x1_2ctamma",
        "CONFIG_sm100_128x256x128_128x256x64_cluster2x1_1ctamma",
        "CONFIG_sm100_128x256x128_128x256x64_cluster2x1_2ctamma",
        "CONFIG_sm100_128x256x128_128x256x64_cluster2x2_2ctamma",
        "CONFIG_sm100_128x128x128_128x128x64_cluster4x1_2ctamma",
        "CONFIG_sm100_128x128x128_128x128x64_cluster4x2_2ctamma",
        "CONFIG_sm100_128x128x128_128x128x64_cluster4x1_1ctamma",
        "CONFIG_sm100_128x128x128_128x128x64_cluster1x4_1ctamma",
        "CONFIG_sm100_128x128x128_128x128x64_cluster2x2_1ctamma",
        "CONFIG_sm100_256x256x128_128x256x64_cluster2x4_2ctamma",
    ]
    # test_sm107_block_scale_matmul_multi_mma_m: several MMA instructions along M.
    for cta_m, cta_n in [(128, 256), (256, 128), (256, 256), (512, 128)]:
        for cta_group, cluster in [(1, "cluster1x1"), (2, "cluster2x1")]:
            names.append(f"CONFIG_sm100_{cta_m}x{cta_n}x128_128x{cta_n}x64_{cluster}_{cta_group}ctamma")
    return sorted(set(names))


def _sm107_moe_cases() -> list[tuple[str, str, dict]]:
    """(config, combo, graph kwargs) for every plan the sm107 block of
    test_moe_grouped_block_scale_matmul_fwd.py makes: test_e2e_sm107 (E=2, S=1024,
    K=512), test_e2e_sm107_multi_mma_m (E=4, S=4*group_m, K=256) and the mixed
    128x128 geometry (all three combos here, homogeneous)."""
    cases = []
    for name in ("CONFIG_sm100_128x256x128_128x256x64_cluster2x1_2ctamma", "CONFIG_sm100_128x256x128_128x256x64_cluster1x1_1ctamma"):
        for combo in sorted(_COMBOS):
            cases.append((name, combo, dict(E=2, S=1024, N=256, K=512, num_groups=4)))
    for cta_m, cta_n in [(128, 256), (256, 128), (256, 256), (512, 128)]:
        for cta_group, cluster in [(1, "cluster1x1"), (2, "cluster2x1")]:
            group_m = cta_m * cta_group if cta_m == 512 else 128
            for combo in ("nvfp4", "mxfp8"):
                cases.append(
                    (
                        f"CONFIG_sm100_{cta_m}x{cta_n}x128_128x{cta_n}x64_{cluster}_{cta_group}ctamma",
                        combo,
                        dict(E=4, S=4 * group_m, N=256, K=256, num_groups=4),
                    )
                )
    for combo in sorted(_COMBOS):
        cases.append(("CONFIG_sm100_128x128x128_128x128x64_cluster1x1_1ctamma", combo, dict(E=2, S=512, N=256, K=512, num_groups=2)))
    return cases


# The only sm107 geometries whose rings cannot all clear the line at the budget depth
# even with the SF rings first -- MoE only (its TMA-store buffer sits ahead of the
# rings): (config, combo) -> (budget stages, rendered stages).  Both returned NaN
# before the fix; the dense template never trims.
_EXPECTED_MOE_TRIMS = {
    ("CONFIG_sm100_512x128x128_128x128x64_cluster1x1_1ctamma", "nvfp4"): (3, 2),
    ("CONFIG_sm100_512x128x128_128x128x64_cluster2x1_2ctamma", "nvfp4"): (3, 2),
    ("CONFIG_sm100_256x128x128_128x128x64_cluster2x1_2ctamma", "mxfp8"): (7, 6),
}


def _catalog_smem_geometries(mma_tile_k_bytes: int) -> list[str]:
    """One catalog config per distinct SMEM layout at that MMA-K width: the cluster
    shape only changes multicast, not the ring sizes, so 9600 K64 entries collapse to
    a few hundred."""
    seen: dict[tuple, str] = {}
    for c in CATALOG:
        if c.pipeline != "sm100" or c.mma_tile_k_bytes != mma_tile_k_bytes or c.cta_tile_k_bytes != 128:
            continue
        key = (c.cta_tile_m, c.cta_tile_n, c.cta_group, c.mma_tile_m, c.mma_tile_n, c.mma_size_m, c.epi_tile_m, c.split_k_slices)
        seen.setdefault(key, c.name)
    return sorted(seen.values())


# --- tests --------------------------------------------------------------------------


@pytest.mark.parametrize("template", [_DENSE_TEMPLATE, _MOE_TEMPLATE])
def test_sf_rings_are_declared_before_ab_rings(template):
    """Source order IS the SMEM layout.  Both block-scale templates must declare the
    scale-factor rings before the A/B rings, every ring root must be a
    ``Tcgen05SmemDesc.build()`` operand (that is WHY the order matters), and the
    reason must be written next to the declarations so nobody reorders it back."""
    src = (_TEMPLATES_DIR / template).read_text()

    def decl(name: str) -> int:
        hits = [m.start() for m in re.finditer(rf"^    {name} = \[\n", src, re.M)]
        assert len(hits) == 1, (template, name, hits)
        return hits[0]

    sfa, sfb, a, b = (decl(n) for n in ("smem_sfa_list", "smem_sfb_list", "smem_a_list", "smem_b_list"))
    assert sfa < sfb < a < b, f"{template}: rings must be declared SFA, SFB, A, B; got offsets {(sfa, sfb, a, b)}"

    smem_d = src.index("smem_d_ptr = cutlass.Array(")
    if template == _MOE_TEMPLATE:
        assert smem_d < sfa, "the MoE template stages its TMA-store buffer AHEAD of the rings (the compiler's header model assumes so)"
    else:
        assert smem_d > b, "the dense template stages its TMA-store buffer AFTER the rings (the compiler's header model assumes so)"

    for lst in ("smem_a_list", "smem_b_list", "smem_sfa_list", "smem_sfb_list"):
        pat = rf"Tcgen05SmemDesc\.build\(\s*start_address={lst}\["
        assert re.search(pat, src), f"{template}: {lst} is no longer a Tcgen05SmemDesc.build() root"

    comment = src[max(0, sfa - 1600) : sfa]
    assert (
        "descriptors.py:513-522" in comment and "14 bits" in comment
    ), f"{template}: the 14-bit build() mask rationale must sit right above the ring declarations"


def test_model_reproduces_the_forced_stage_gpu_sweep(rubin):
    """The predictor's calibration: the forced-``ab_stages`` sweep on the Rubin dev
    node (mxfp8 128x128, M=N=256 K=512) failed at 9 and 8 stages and passed bit-exact
    at 7 and 6.  Under the PRE-fix ring order the model puts the SFA root exactly where
    the sweep saw the verdict flip -- and the natural depth at Rubin's budget is 9."""
    cfg = by_name("CONFIG_sm100_128x128x128_128x128x64_cluster1x1_1ctamma")
    tmpl, txt = _render(cfg, _chain(_dense_graph("mxfp8")))
    assert tmpl.file == _DENSE_TEMPLATE
    consts = _consts(txt)
    assert int(consts["ab_stages"]) == 9
    seen = {}
    for stages, sfa_root, fails in [(9, 296960, True), (8, 264192, True), (7, 231424, False), (6, 198656, False)]:
        roots = _model_roots(consts, tmpl.file, tma_store=False, sf_first=False, stages=stages)
        seen[stages] = roots
        assert roots["SFA[0]"] == sfa_root, (stages, roots)
        assert bool(_wrapped(roots)) is fails, (stages, roots)
    # With the SF rings first the same 9 stages keep every root below the line.
    fixed = _model_roots(consts, tmpl.file, tma_store=False, sf_first=True)
    assert not _wrapped(fixed), fixed
    assert fixed["SFA[0]"] == 2048 and fixed["B[0]"] == 159744, fixed


@pytest.mark.parametrize("combo", sorted(_COMBOS))
def test_sm107_suite_configs_stay_below_the_line(rubin, combo):
    """Every (geometry, cta_group) the sm107 block-scale GPU suite plans, at Rubin's
    budget: the compiler's roots and the predictor's agree and all sit below
    262144 -- while the PRE-fix order wraps on most of them (the 60 red tests).
    The pre-fix count is pinned so the model cannot drift into 'everything passes'."""
    chain = _chain(_dense_graph(combo))
    pre_fix_wrapped = 0
    for name in _sm107_suite_configs():
        cfg = by_name(name)
        tmpl, txt = _render(cfg, chain)
        assert tmpl.file == _DENSE_TEMPLATE, (name, tmpl.file)
        consts = _consts(txt)
        rendered = _rendered_roots(txt)
        model = _model_roots(consts, tmpl.file, tma_store=False)
        _cross_check(rendered, model, slack=1024)
        assert not _wrapped(rendered), (name, combo, rendered)
        assert not _wrapped(model), (name, combo, model)
        # The dense template keeps every stage the budget allows -- the fix costs nothing here.
        assert _trim(txt) is None and int(consts["ab_stages"]) == _budget_stages(consts, tmpl, chain, cfg), (name, combo, consts["ab_stages"])
        pre_fix = _model_roots(consts, tmpl.file, tma_store=False, sf_first=False)
        if _wrapped(pre_fix):
            pre_fix_wrapped += 1
        if name.startswith("CONFIG_sm100_128x128x128_128x128x64_cluster1x1_"):
            assert _wrapped(pre_fix), (name, combo, pre_fix)  # the forced-stage sweep's shape wrapped at every combo
    # 2026-09-15 GPU run: 60 of the 71 planning tests failed.  Per unique geometry the
    # pre-fix order wraps on these many of the 19; the survivors are the ones whose
    # A+B rings stop short of the line (nvfp4 128x256 at 5 stages: SFA root 247808,
    # the 11 GPU passes; 256x256 / 512x128 whose SF rings start at 223232-247808).
    assert len(_sm107_suite_configs()) == 19
    assert pre_fix_wrapped == {"nvfp4": 16, "mxfp4": 15, "mxfp8": 15}[combo], (combo, pre_fix_wrapped)


def test_sm107_moe_suite_stays_below_the_line_with_exactly_two_trims(rubin):
    """Same layout rule for the grouped (MoE) block-scale template, whose TMA-store
    staging buffer sits AHEAD of the rings and pushes every root up: every plan the
    sm107 MoE suite makes renders below the line, and the renderer trims the ring
    depth on exactly the geometries that cannot clear it otherwise (pinned, so a
    silent widening or narrowing of the trim set is loud).  Pre-fix, every one of
    these wrapped on an SF root (the MoE twin of the 60 red dense tests)."""
    trims = {}
    for name, combo, kw in _sm107_moe_cases():
        cfg = by_name(name)
        chain = _chain(_moe_graph(combo, **kw))
        tmpl, txt = _render(cfg, chain)
        assert tmpl.file == _MOE_TEMPLATE, (name, tmpl.file)
        consts = _consts(txt)
        tma_store = C100._use_tma_store_epi(chain, cfg)
        rendered = _rendered_roots(txt)
        model = _model_roots(consts, tmpl.file, tma_store=tma_store)
        _cross_check(rendered, model, slack=2048)
        assert not _wrapped(rendered), (name, combo, rendered)
        assert not _wrapped(model), (name, combo, model)
        budget = _budget_stages(consts, tmpl, chain, cfg)
        # Pre-fix (A, B, SFA, SFB at the budget depth) every one of these wrapped on an SF root.
        assert _wrapped(_model_roots(consts, tmpl.file, tma_store=tma_store, sf_first=False, stages=budget)), (name, combo, budget)
        trim = _trim(txt)
        if trim is None:
            assert int(consts["ab_stages"]) == budget, (name, combo, consts["ab_stages"], budget)
        else:
            assert trim == (budget, int(consts["ab_stages"])), (name, combo, trim, budget)
            assert trim[1] == budget - 1, (name, combo, trim)  # minimal: one stage less clears the line
            trims[(name, combo)] = trim
    assert trims == _EXPECTED_MOE_TRIMS, trims


@pytest.mark.parametrize("combo", sorted(_COMBOS))
def test_every_k64_catalog_geometry_renders_below_the_line(rubin, combo):
    """The guard is a typed failure for a layout nobody validated -- it must be SILENT
    on every catalog geometry the 64-byte-K (Rubin) block-scale pipeline can plan."""
    chain = _chain(_dense_graph(combo))
    rendered_n = 0
    for name in _catalog_smem_geometries(64):
        cfg = by_name(name)
        tmpl = select_template(chain, cfg)
        if tmpl.accepts(chain, cfg) is not None:
            continue
        try:
            txt = C100._render_block_scale_tile_constants(cfg, chain, tmpl)
        except NotImplementedError:
            continue  # a typed decline ahead of the guard (SF geometry, TMEM budget, ...)
        rendered_n += 1
        rendered = _rendered_roots(txt)
        assert not _wrapped(rendered), (name, combo, rendered)
        _cross_check(rendered, _model_roots(_consts(txt), tmpl.file, tma_store=False), slack=1024)
    # The block-scale pipeline plans 128-multiple tiles only (the SF 128x4 swizzle
    # rejects the other 372 geometries up front) and 512x256 exceeds the TMEM budget:
    # cta_m in {128, 256, 512} x cta_n in {128, 256} x cta_group in {1, 2} minus two.
    # Pinned exactly so a catalog widening re-reads this sweep instead of diluting it.
    assert rendered_n == 10, rendered_n


@pytest.mark.parametrize("combo", sorted(_COMBOS))
def test_sm100_budget_never_reaches_the_line(blackwell, combo):
    """227 KiB cannot put a root past 256 KiB; the reorder moves SMEM offsets on SM100
    too, so pin that the guard stays silent and the compiler still models the roots."""
    chain = _chain(_dense_graph(combo))
    rendered_n = 0
    for name in _catalog_smem_geometries(32):
        cfg = by_name(name)
        tmpl = select_template(chain, cfg)
        if tmpl.accepts(chain, cfg) is not None:
            continue
        try:
            txt = C100._render_block_scale_tile_constants(cfg, chain, tmpl)
        except NotImplementedError:
            continue
        rendered_n += 1
        rendered = _rendered_roots(txt)
        assert max(rendered.values()) < SM100_BUDGET, (name, rendered)
        assert not _wrapped(rendered), (name, combo, rendered)
    assert rendered_n == 6, rendered_n  # 128-multiple tiles minus the 5 the 227 KiB TMEM/SMEM budget declines


def test_guard_names_the_wrapping_root():
    """Unit check of the compiler's model + guard, on a layout that crosses the line
    (16 stages of the 128x128 mxfp8 geometry: the A ring alone is 256 KiB)."""
    roots = C100._block_scale_smem_desc_roots(
        header_bytes=2048,
        rings=[("SFA[0]", 16 * 512), ("SFB[0]", 16 * 512), ("A[0]", 16 * 16384), ("B[0]", 16 * 16384)],
    )
    assert roots == {"SFA[0]": 2048, "SFB[0]": 10240, "A[0]": 18432, "B[0]": 280576}
    cfg = by_name("CONFIG_sm100_128x128x128_128x128x64_cluster1x1_1ctamma")
    with pytest.raises(ValueError, match=r"B\[0\] at 280576 B.*ab_stages=16.*14 bits.*version-1 descriptor"):
        C100._check_block_scale_desc_roots(cfg, 16, roots)
    # Below the line: silent.
    C100._check_block_scale_desc_roots(cfg, 9, {"SFA[0]": 2048, "B[0]": WRAP - 1024})


def test_renderer_trims_a_depth_whose_root_would_wrap(rubin, monkeypatch):
    """End to end through the renderer: force a budget depth (16) whose A ring pushes
    the B root past the line.  The renderer falls back to the deepest ring that clears
    it -- the same answer the predictor gives -- and says so beside ab_stages."""
    # Force the budget-derived depth to 16 AND lift the SMEM budget, so the
    # renderer's alignment re-fit (#1074) keeps 16 and only the desc-root guard
    # can trim from there.
    monkeypatch.setattr(TC, "smem_ab_stages", lambda *a, **k: 16)
    monkeypatch.setattr(TC, "smem_ab_budget_bytes", lambda *a, **k: 1 << 30)
    cfg = by_name("CONFIG_sm100_128x128x128_128x128x64_cluster1x1_1ctamma")
    chain = _chain(_dense_graph("mxfp8"))
    tmpl = select_template(chain, cfg)
    txt = C100._render_block_scale_tile_constants(cfg, chain, tmpl)
    consts = _consts(txt)
    deepest_clear = max(st for st in range(1, 17) if not _wrapped(_model_roots(consts, tmpl.file, tma_store=False, stages=st)))
    # 15 stages: SF rings 2 x 7680 -> A root 18432, A ring 245760 -> B root 264192 (wraps); 14 clears at 245760.
    assert int(consts["ab_stages"]) == deepest_clear == 14, consts["ab_stages"]
    assert _trim(txt) == (16, deepest_clear)
    assert not _wrapped(_rendered_roots(txt))


def test_renderer_refuses_when_even_one_stage_wraps(rubin):
    """A header that already sits at the line leaves no depth to trim to: the plan
    dies with the typed ValueError instead of rendering a kernel that returns NaN."""
    cfg = by_name("CONFIG_sm100_128x128x128_128x128x64_cluster1x1_1ctamma")
    chain = _chain(_dense_graph("mxfp8"))
    tmpl = dataclasses.replace(select_template(chain, cfg), smem_fixed_reserve=WRAP)
    with pytest.raises(ValueError, match=r"SMEM ring root\(s\) SFA\[0\] at 262144 B, .* sit at or above 262144 B with ab_stages=1"):
        C100._render_block_scale_tile_constants(cfg, chain, tmpl)


def test_rubin_budget_is_what_the_probe_and_the_suite_use():
    """The number this module pins the layout against is the one tile_config derives
    on cc 10.7 -- max(227 KiB opt-in, 327 KiB oversized carveout) = 334848."""
    assert RUBIN_BUDGET == max(227 * 1024, 327 * 1024)
    assert WRAP == 1 << 18
    assert dataclasses.is_dataclass(by_name("CONFIG_sm100_128x128x128_128x128x64_cluster1x1_1ctamma"))
