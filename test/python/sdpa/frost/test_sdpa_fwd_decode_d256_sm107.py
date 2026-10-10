# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The d256 DECODE tile of the FROST Rubin (SM107) f16/bf16 engine (sm107/decode_d256_f16.py).

The Rubin sibling of the SM100 swap-AB tile: the same body (KV tokens on the MMA M
axis, the packed Q rows on N, one cta_group::1 CTA per unit) under
``config_sm107.make_cfg_d256_decode``, routed on cc 10.7 by the same record field
(``TemplateParams.decode_q_tile``) under the Rubin route ``config_sm107.decode_d256_q_tile``:
f16/bf16 d256 graphs whose S_q x packed heads fit 16 rows ride the 16-column tile as on
SM100, rows in (16, 32] ride the 32-column tile, and a packed MTP step of up to two TOKEN
UNITS (24/2 at S_q = 4: two units of two tokens x 12 heads, each streaming the KV range once)
rides it too (``Q_TOKEN_UNITS``).  Same graph contract,
same engine row (``sdpa_fwd_prefill_sm107``): every graph test pins the engine with
``select_engine`` and asserts WHICH template served the plan through the executor's
``kernel_template`` (a decline or a fallback to the prefill tile fails instead of
passing on the wrong kernel).  The reference is an fp32 torch softmax over the dense
K/V the pools were built from (shared with the SM100 suite).

Coverage, the SM100 case list re-run on the Rubin tile: paged (page 16/32/64/128,
NHD/HND) and dense padded caches, mixed lengths incl. 0 and 1, the head groups 8:1 /
16:1 / 32:2 / 16:4 / MHA and the 24/2 geometry (G = 12), MTP bottom-right causal at S_q
2 (8:1) and 4 (4:1) with per-batch Q lengths, the 24/2 MTP steps S_q = 2 / 3 / 4 on the
32-column tile in one and two token units (dense and paged, bottom-right, per-batch Q
lengths, split and unsplit; the token-unit forms BITWISE each other and the unpacked
form), sliding window, right band, sink (dense and paged, keyless rows), Stats natural
and base-2, fp16 and bf16, CUDA-graph replay under ``set_sync_debug_mode("error")``
(Rule 3), the 32-column tile at the template level, and the routing boundary (a third
token unit, an unpacked step above 32 tokens, THD queries, d128).

The row claims the SM100 tile's PackGQA and split-KV on this tile (``pack_gqa_d_shapes``
carries (256, 256) for exactly this route; the SM107 D256 split rule exempts it), so the
SM100 suite's policy assertions apply here verbatim: whole-group packing (24/2: 12 live
rows + 4 zero tail rows per (batch, KV head) unit), the decode split model
(``choose_decode_tile_split_kv``: the eager-safe lead, the captured runner-up, swapped under
``is_cuda_graph_replay_expected``), plus the three equalities the claim rests on -- paged ==
dense BITWISE on the same tokens, packed == unpacked BITWISE, and split == unsplit within a
DERIVED budget (one output ulp from the single cast of two differently-associated fp32 sums,
plus the half-precision P quantization term, ``_assert_split_matches_unsplit``).  The
32-row shapes (32/2 at S_q = 2) now ride the 32-column tile PACKED (the former unpacked pin
inverted with the Rubin route); past the route -- a third token unit (24/2 at S_q = 5), an
unpacked step above 32 tokens -- the Rubin d256 prefill kernel wires no PackGQA, so a packed
request there is a typed decline (the REJECT cases) -- and the tile's packing / split
partials are still driven at the TEMPLATE level too (the adapter's module loader +
``launch_f16``), the kernel-level evidence under the graph-level claim.

Structural pins (host-runnable, no Rubin GPU): the module is the SM100 body modulo the
declared Rubin deltas (the twin-diff snapshot), ``DESC_VERSION`` derived from the layout
and wired into every ``SmemTile``, ``SPIN_RING_WAITS`` on exactly the per-KV-iteration
ring waits, the config twin's backstops, and the CuTe DSL version gate (Rule 7).
"""

import math
import re
from pathlib import Path

import pytest

import torch

from cudnn.frost.compiled_cache import positional_entry
from cudnn.sdpa.fwd.heuristics import choose_decode_tile_split_kv

from frost_test_utils import _is_plan_for, launch_f16, offers_engine, requires_dsl, requires_rubin, select_engine
from test_sdpa_fwd_decode_d256_sm100 import _pools, _ref, _served_by

D = 256
DECODE = "decode_d256_f16"
PREFILL = "prefill_d256_f16"
ARCH = "sm107"


def _dsl_floor_reason():
    """Rule 7: the Rubin tile needs a CuTe DSL with the sm_107a target (the public 4.8.0
    wheel); below that floor every test here SKIPS (never fails) with the typed message."""
    try:
        from cudnn.frost.buffers import cutedsl_arch_requirement_error

        return cutedsl_arch_requirement_error((10, 7))
    except Exception as e:  # pragma: no cover - a DSL that cannot be probed reads as below the floor
        return f"cannot probe the CuTe DSL for the sm_107a target: {e}"


_DSL_FLOOR_WHY = _dsl_floor_reason()
requires_sm107_dsl = pytest.mark.skipif(_DSL_FLOOR_WHY is not None, reason=_DSL_FLOOR_WHY or "")
# Module-wide, built AFTER ``requires_sm107_dsl`` exists: every test here -- the GPU cases and the
# host-runnable structural pins that import the Rubin module through ``_load`` -- skips (never fails)
# below the sm_107a DSL floor (Rule 7); the probe that asserts the floor's typed message does so by
# monkeypatch above the floor, so it skips below it too.
pytestmark = [pytest.mark.L0, requires_dsl, requires_sm107_dsl]


def _gpu(f):
    """A test that runs the tile: a Rubin-line GPU and a DSL with its target."""
    return requires_sm107_dsl(requires_rubin(f))


def _engine():
    from cudnn.sdpa.fwd.engines import engine_name

    return engine_name(arch=ARCH)


def _load(rubin=True, **params):
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    return _load_sm100_kernel_module((D, D), TemplateParams(**params), rubin=rubin)


def _code_lines(src):
    """Source with whole-line comments dropped -- the module's prose quotes the
    spellings these pins count."""
    return "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))


def _wait_sites(code):
    """Every mbarrier wait call site of the kernel source: (target, balanced argument text)."""
    out = []
    for m in re.finditer(r"^\s*(?:bars\.)?(mb_\w+)(?:\[[^\]]*\])*\.wait\(", code, re.M):
        target = m.group(1)
        i, depth = m.end(), 1
        while depth:
            depth += (code[i] == "(") - (code[i] == ")")
            i += 1
        out.append((target, code[m.end() : i - 1]))
    return out


# --- structural pins: the port is the SM100 body plus the declared Rubin deltas -------


def test_sm107_decode_routes_to_the_rubin_sibling():
    """The decode record routes to sm107/decode_d256_f16.py on cc 10.7 (ahead of the shared
    paged-prefill arm: the paged record too) and to the SM100 file otherwise."""
    for extra in ({}, dict(paged_kv=True, page_size=16)):
        mod = _load(rubin=True, decode_q_tile=16, seq_kv_lens_present=True, **extra)
        assert mod.__file__.endswith("/sm107/decode_d256_f16.py") and "sm107_decode" in mod.__name__, mod.__name__
    sm100 = _load(rubin=False, decode_q_tile=16, seq_kv_lens_present=True)
    assert sm100.__file__.endswith("/sm100/decode_d256_f16.py") and "sm100_decode" in sm100.__name__, sm100.__name__


def test_sm107_decode_body_matches_the_sm100_body():
    """The SNAPSHOT pin: the Rubin module is the SM100 body line for line, modulo the
    declared deltas -- the config import, the DSL gate + the two module constants, the
    ``desc_version=DESC_VERSION`` keyword on every SmemTile, the ``spin=SPIN_RING_WAITS``
    keyword on the ring waits and the header prose.  A fix that lands in one file and
    not the other fails here; so does a body change smuggled into the port."""
    from cudnn.sdpa.fwd import api_dsl

    kdir = Path(api_dsl.__file__).parent / "kernels"
    sm100 = (kdir / "sm100" / "decode_d256_f16.py").read_text(encoding="utf-8")
    sm107 = (kdir / "sm107" / "decode_d256_f16.py").read_text(encoding="utf-8")

    def _body(src):
        # everything after the module docstring, comments dropped
        end = src.index('"""', src.index('"""') + 3) + 3
        return [ln for ln in _code_lines(src[end:]).splitlines() if ln.strip()]

    a, b = _body(sm100), _body(sm107)
    b = [ln for ln in b if ln.strip() not in ("desc_version=DESC_VERSION,",)]
    b = [ln.replace(", spin=SPIN_RING_WAITS)", ")") for ln in b]
    b = [
        ln
        for ln in b
        if not ln.startswith(
            (
                "DESC_VERSION: int = ",
                "SPIN_RING_WAITS: bool = ",
                "_DSL_ARCH_ERROR = ",
                "if _DSL_ARCH_ERROR is not None:",
                '    raise NotImplementedError(f"decode_d256_f16 (sm107)',
            )
        )
    ]
    b = [ln for ln in b if ln != "from cudnn.frost.buffers import cutedsl_arch_requirement_error"]
    b = [
        ln.replace(
            "from cudnn.sdpa.fwd.config_sm107 import TemplateParams, d256_decode_desc_version, make_cfg_d256_decode",
            "from cudnn.sdpa.fwd.config_sm100 import TemplateParams, make_cfg_d256_decode",
        )
        for ln in b
    ]
    assert a == b, "the Rubin decode module diverged from the SM100 body beyond the declared deltas:\n" + "\n".join(
        f"{'-' if x else '+'} {x or y}" for x, y in zip(a, b) if x != y
    )


@pytest.mark.parametrize("n_q", [16, 32])
def test_sm107_decode_descriptor_version_matches_the_smem_budget(n_q):
    """DESC_VERSION is derived from the layout (config_sm107.d256_decode_desc_version) and
    is 0 at both tile widths: the highest MMA-operand slab (P^T) starts at 200 / 208 KiB,
    under the 256 KiB version-0 window, and the whole tile (209 / 226 KiB) fits the
    STANDARD 227 KiB carveout (no oversized-SMEM launch, no L1 shrink).  A record that
    moved an operand past the line must flip the module constant, never a tile literal."""
    from cudnn.sdpa.fwd.config_sm107 import d256_decode_desc_version, d256_decode_smem_layout

    mod = _load(decode_q_tile=n_q, seq_kv_lens_present=True)
    lay = d256_decode_smem_layout(mod.CFG)
    assert mod.DESC_VERSION == d256_decode_desc_version(mod.CFG) == 0, (mod.DESC_VERSION, lay)
    assert lay["p_start"] == (200 if n_q == 16 else 208) * 1024 and lay["p_start"] < 256 * 1024, lay
    assert lay["fits_standard_carveout"] and lay["total"] <= 227 * 1024, lay
    assert (mod.CFG.N_Q, mod.CFG.SOFTMAX_WARPS, mod.CFG.TOTAL_WARPS, mod.CFG.STAGES_KV, mod.CFG.TILE_K_HW) == (n_q, n_q // 4, n_q // 4 + 2, 3, 16)


def test_sm107_decode_every_smem_tile_takes_the_module_desc_version():
    """Every SmemTile of the module is wired to DESC_VERSION (4 tiles: Q^T, the K/V ring as
    K and as V^T, P^T); a re-literalled desc_version bypasses the constant."""
    mod = _load(decode_q_tile=16, seq_kv_lens_present=True)
    code = _code_lines(Path(mod.__file__).read_text(encoding="utf-8"))
    n_tiles = len(re.findall(r"\bSmemTile\(", code))
    assert n_tiles == 4 == code.count("desc_version=DESC_VERSION"), (n_tiles, code.count("desc_version=DESC_VERSION"))
    assert not re.search(r"desc_version=[01]\b", code), "a re-literalled desc_version bypasses DESC_VERSION"


def test_sm107_decode_ring_waits_take_the_module_spin_constant():
    """SPIN_RING_WAITS holds the measured per-kernel value (False: the sleeping form the SM100
    body runs, until this tile's own A/B), is defined exactly once, and every per-KV-iteration
    RING wait -- the TMA warp's three kv_empty waits, the MMA warp's kv_full x3 / s_empty x2 /
    p_full, the softmax warps' s_full and in-loop bmm2_done: 11 sites -- passes it, while the
    three one-shot waits (q_full, the epilogue's last bmm2_done, tmem_dealloc) keep the default."""
    mod = _load(decode_q_tile=16, seq_kv_lens_present=True)
    assert mod.SPIN_RING_WAITS is False
    code = _code_lines(Path(mod.__file__).read_text(encoding="utf-8"))
    assert len(re.findall(r"^SPIN_RING_WAITS: bool = (?:True|False)$", code, re.M)) == 1
    assert not re.search(r"spin=(?:True|False)\b", code), "a spin= literal at a call site bypasses SPIN_RING_WAITS"
    sites = _wait_sites(code)
    spun = sorted(t for t, args in sites if "spin=SPIN_RING_WAITS" in args)
    idle = sorted(t for t, args in sites if "spin=SPIN_RING_WAITS" not in args)
    assert spun == sorted(["mb_kv_empty"] * 3 + ["mb_kv_full"] * 3 + ["mb_s_empty"] * 2 + ["mb_p_full", "mb_s_full", "mb_bmm2_done"]), spun
    assert idle == ["mb_bmm2_done", "mb_q_full", "mb_tmem_dealloc"], idle
    assert code.count("spin=") == 11, "a spin= outside a .wait( call"


def test_sm107_decode_config_twin_backstops():
    """config_sm107.make_cfg_d256_decode: the SM100 record's predicates plus the Rubin ones --
    the gate / pre-folded scale / other-template records are refused by name, the Rubin prefill
    factory refuses a decode record (the loader routes on it first), and both tile widths build."""
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    from cudnn.sdpa.fwd.config_sm107 import d256_decode_desc_version, make_cfg_d256, make_cfg_d256_decode

    base = dict(decode_q_tile=16, seq_kv_lens_present=True)
    for bad, pattern in (
        (dict(epilogue_gate=True), "epilogue gate is not wired on the decode tile"),
        (dict(softmax_scale_prefolded=True), "softmax_scale_prefolded is not wired on the decode tile"),
        (dict(mma_2x2=True), "select other templates"),
        (dict(decode_tile=True), "select other templates"),
        (dict(thd_varlen=True), "THD"),
        (dict(pack_gqa=True, qh_per_kh=32), "fit the Q tile"),
        (dict(paged_kv=True, page_size=48), "page_size must be a positive multiple of 8"),
    ):
        with pytest.raises(ValueError, match=pattern):
            make_cfg_d256_decode(TemplateParams(**{**base, **bad}))
    with pytest.raises(ValueError, match="decode_q_tile"):
        make_cfg_d256(TemplateParams(**base))
    with pytest.raises(ValueError, match="decode_q_tile"):
        make_cfg_d256_decode(TemplateParams())
    cfg, _ = make_cfg_d256_decode(TemplateParams(**base))
    assert (cfg.N_Q, cfg.TILE_N, cfg.STAGES_KV, cfg.TILE_K_HW, d256_decode_desc_version(cfg)) == (16, 128, 3, 16, 0)
    wide, _ = make_cfg_d256_decode(TemplateParams(decode_q_tile=32, seq_kv_lens_present=True, dtype_qkv=2))
    assert (wide.N_Q, wide.SOFTMAX_WARPS, wide.TOTAL_WARPS, wide.P_SWZ_BYTES, d256_decode_desc_version(wide)) == (32, 8, 10, 64, 0)


def test_sm107_decode_module_declines_below_the_dsl_floor(monkeypatch):
    """Rule 7 probe: a DSL without the sm_107a target makes the module raise the TYPED decline
    (NotImplementedError naming the installed version) before any kernel body is defined.  A
    params record no other test loads, so the loader cannot serve a cached module."""
    from cudnn.frost import buffers
    from cudnn.frost.template_loader import load_template
    from cudnn.sdpa.fwd import api_dsl
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: False)
    want = buffers.cutedsl_arch_requirement_error((10, 7))
    assert want is not None and "sm_107a" in want
    path = str(Path(api_dsl.__file__).parent / "kernels" / "sm107" / "decode_d256_f16.py")
    with pytest.raises(NotImplementedError, match=re.escape(want)):
        load_template(path, TemplateParams(decode_q_tile=16, seq_kv_lens_present=True, stats_log2=True, window_left=77, window_right=5), tag="dsl_floor_probe")


def test_decode_q_tile_rule():
    """The SM100 routing rule is untouched (16 rows ride the decode tile; the 32-column tile is a
    valid config, D256_DECODE_MAX_Q_ROWS, that line does not select) and the Rubin rule
    (config_sm107.decode_d256_q_tile) extends it: rows in (16, 32] ride the 32-column tile in one
    unit, a PACKED group may take two TOKEN UNITS of it (the MTP step), an unpacked / MHA step one
    unit only; past that (a third unit, a group wider than the tile) the prefill kernel.  The unit
    count is the kernel's own derivation (ceil(S_q / (N_Q // G)), decode_d256_q_units)."""
    from cudnn.sdpa.fwd import config_sm107
    from cudnn.sdpa.fwd.config_sm100 import D256_DECODE_MAX_Q_ROWS, D256_DECODE_ROUTED_MAX_Q_ROWS, decode_d256_q_tile, decode_d256_q_units

    assert (D256_DECODE_ROUTED_MAX_Q_ROWS, D256_DECODE_MAX_Q_ROWS) == (16, 32)
    assert decode_d256_q_tile(1, 12) == 16  # the 24/2 geometry: a group that does not divide the tile still fits it
    assert decode_d256_q_tile(1, 16) == decode_d256_q_tile(2, 8) == decode_d256_q_tile(16, 1) == 16
    for s_q, g in ((2, 16), (1, 32), (17, 1), (32, 1), (3, 16), (33, 1), (1, 64), (0, 16), (2, 12), (4, 12)):
        assert decode_d256_q_tile(s_q, g) == 0, (s_q, g)
    # The Rubin route.
    rubin = config_sm107.decode_d256_q_tile
    assert (config_sm107.D256_DECODE_ROUTED_MAX_Q_ROWS, config_sm107.D256_DECODE_ROUTED_MAX_TOKEN_UNITS) == (32, 2)
    for s_q, g in ((1, 12), (1, 16), (2, 8), (16, 1), (2, 6), (1, 1)):
        assert rubin(s_q, g) == 16 and decode_d256_q_units(s_q, g, 16) == 1, (s_q, g)  # one unit of the 16-column tile, as on SM100
    for s_q, g, units in ((2, 12, 1), (3, 12, 2), (4, 12, 2), (2, 16, 1), (4, 16, 2), (4, 8, 1), (8, 8, 2), (1, 32, 1), (2, 32, 2), (4, 6, 1), (10, 6, 2)):
        assert rubin(s_q, g) == 32 and decode_d256_q_units(s_q, g, 32) == units, (s_q, g, units)  # the 32-column tile, packed
    for s_q in (17, 24, 32):
        assert rubin(s_q, 1) == 32 and decode_d256_q_units(s_q, 1, 32) == 1, s_q  # an unpacked / MHA step of up to 32 tokens: one unit
    for s_q, g in ((5, 12), (5, 16), (9, 8), (3, 32), (11, 6), (33, 1), (64, 1), (1, 64), (0, 16), (4, 0)):
        assert rubin(s_q, g) == 0, (s_q, g)  # past the route: a third unit, an unpacked step above 32 tokens, a group wider than the tile
    assert decode_d256_q_units(4, 12, 0) == 0 and decode_d256_q_units(1, 64, 32) == 0
    from cudnn.sdpa.fwd import engines

    caps107 = next(s.capabilities for s in engines.ENGINE_SPECS if s.name == _engine())
    caps100 = next(s.capabilities for s in engines.ENGINE_SPECS if s.name == engines.engine_name(arch="sm100"))
    assert engines.decode_d256_q_tile_for_row(caps107, 4, 12) == 32 and engines.decode_d256_q_tile_for_row(caps100, 4, 12) == 0


def test_sm107_row_and_predicate_agree_on_the_decode_tile():
    """The facts-level twin (engines.d256_decode_tile_selected) of the adapter's _decode_q_tile on the
    Rubin row: a decode-shaped half d256 graph is selected (paged or dense, sink or not); a gated,
    pre-folded, THD, quantized or wider graph is not -- and the row's paged rule admits exactly the
    selected form while the paged PREFILL pipeline keeps its THD-only scope."""
    import cudnn
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

    caps = next(s.capabilities for s in engines.ENGINE_SPECS if s.name == _engine())
    assert caps.decode is True
    base = dict(
        b=4, h_q=24, h_kv=2, s_q=1, s_kv=4096, d_qk=D, d_v=D, dtype=cudnn.data_type.BFLOAT16, padded=True, has_paged_kv=True, page_size=16, device_cc=(10, 7)
    )
    f = SdpaGraphFacts(**base)
    assert engines.d256_decode_tile_selected(caps, f, 1) and engines.d256_decode_tile_selected(caps, f, 12)
    assert engines.mismatch(caps, f, None) is None
    assert engines.mismatch(caps, SdpaGraphFacts(**{**base, "has_sink": True}), None) is None
    for change in (dict(s_q=33), dict(thd=True), dict(attn_scale_prefolded=True), dict(d_qk=128, d_v=128)):
        g = SdpaGraphFacts(**{**base, **change})
        assert not engines.d256_decode_tile_selected(caps, g, 1), change
    assert engines.d256_decode_tile_selected(caps, SdpaGraphFacts(**{**base, "s_q": 17}), 1)  # an unpacked 17-token step: the 32-column tile
    # The fused-gate tail: selected through a SPLIT only (the gate rides the split combine; the
    # tile has no gate seams), so an unsplit set is not the tile's and the facts-only question says yes.
    gated = SdpaGraphFacts(**{**base, "has_epilogue_gate": True, "epilogue_gate_dtype": cudnn.data_type.BFLOAT16})
    assert engines.d256_decode_tile_selected(caps, gated, 12) and engines.d256_decode_tile_selected(caps, gated, 12, split_kv=2)
    assert not engines.d256_decode_tile_selected(caps, gated, 12, split_kv=1)
    for change in (dict(s_q=64), dict(d_qk=128, d_v=128)):
        g = SdpaGraphFacts(**{**base, **change})
        reason = engines.mismatch(caps, g, None)
        assert reason and "Rubin paged KV requires THD" in reason, (change, reason)
    assert engines.d256_decode_tile_selected(caps, SdpaGraphFacts(**{**base, "s_q": 2}), 16)  # 32 packed rows: the 32-column tile, one unit
    assert engines.d256_decode_tile_selected(caps, SdpaGraphFacts(**{**base, "s_q": 4}), 12)  # 48 packed rows: two token units
    assert not engines.d256_decode_tile_selected(caps, SdpaGraphFacts(**{**base, "s_q": 5}), 12)  # 60 packed rows: a third unit, past the route


def test_sm107_row_predicate_and_heuristics_agree_on_the_gated_decode_tile():
    """The gate-in-combine contract at the facts level, on the Rubin row: a GATED decode-shaped half
    d256 graph (the ``mul(O_v, sigmoid(G))`` tail) is served on the decode tile through its SPLIT --
    the tile has no gate seams, the split combine applies the gate -- packed or not, paged or dense;
    its unsplit form is the d256 PREFILL kernel's fused epilogue (dense only: the paged prefill kernel
    has no gate), so mismatch admits exactly the sets the lowering runs and the heuristics propose
    exactly those (rule 4: never a declined set): the decode model's split floored at 2 leads, packed
    12:1; a paged gated graph lists no unsplit set at all; a padded dense cache (no split) keeps the
    fused prefill kernel; a prefill-shaped gated graph keeps the old contract (unsplit, unpacked)."""
    import cudnn
    from cudnn.engines.manifest import MANIFEST
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs
    from cudnn.sdpa.fwd.heuristics import _pack_gqa_eligible, _pack_gqa_group, _split_points, recommend
    from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

    caps = next(s.capabilities for s in engines.ENGINE_SPECS if s.name == _engine())
    fam = next(r for r in MANIFEST if r.factory == "FrostSdpaFwdEngines")
    offered = {_engine(): fam.engine_id + fam.slots[_engine()].slot}
    base = dict(
        b=4,
        h_q=24,
        h_kv=2,
        s_q=1,
        s_kv=4096,
        d_qk=D,
        d_v=D,
        dtype=cudnn.data_type.BFLOAT16,
        padded=True,
        has_paged_kv=True,
        page_size=16,
        device_cc=(10, 7),
        device_sm_count=204,
        has_epilogue_gate=True,
        epilogue_gate_dtype=cudnn.data_type.BFLOAT16,
    )

    def sets(f):
        out = [p.knobs for p in recommend("A", f, offered) if p.engine_id == offered[_engine()]]
        for k in out:
            assert engines.mismatch(caps, f, k) is None, (k, engines.mismatch(caps, f, k))
        return [(k.pack_gqa is True, k.split_kv or 1) for k in out]

    paged = SdpaGraphFacts(**base)
    assert engines.mismatch(caps, paged, None) is None
    assert engines.mismatch(caps, paged, SdpaFwdKnobs(pack_gqa=True, split_kv=2)) is None
    assert engines.mismatch(caps, paged, SdpaFwdKnobs(pack_gqa=False, split_kv=4)) is None
    why = engines.mismatch(caps, paged, SdpaFwdKnobs(pack_gqa=False, split_kv=1))
    assert why and "paged" in why and "gate" in why and "combine" in why, why  # the paged prefill kernel has no gate
    why = engines.mismatch(caps, paged, SdpaFwdKnobs(pack_gqa=True, split_kv=1))
    assert why and "decode tile" in why, why
    assert _pack_gqa_eligible(caps, paged, 128, 2) and not _pack_gqa_eligible(caps, paged, 128, 1) and _pack_gqa_group(caps, paged, 128, True, 2) == 12
    paged_sets = sets(paged)
    assert paged_sets[0] == (True, 16) and all(
        s > 1 for _, s in paged_sets
    ), paged_sets  # 8 units x 32 tiles: the model splits 16 ways; no unsplit gated paged plan exists
    assert _split_points(caps, paged, 128, 128, 2, pack_g=12) == [16]
    # Paged + gate + SINK: no plan at all on this row -- a sink never splits (the shared no-split rule)
    # and the paged prefill kernel has no gate -- so the facts-level question is a typed decline too
    # (never "served" with an empty proposal list), the knob-level split is the sink's own decline, and
    # the heuristics propose nothing.  The UNGATED paged sink graph stays served (the tile folds the
    # sink per Q row, unsplit) -- test_sm107_row_and_predicate_agree_on_the_decode_tile pins that.
    paged_sink = SdpaGraphFacts(**{**base, "has_sink": True})
    why = engines.mismatch(caps, paged_sink, None)
    assert why and "paged" in why and "gate" in why and "sink" in why, why
    assert engines.mismatch(caps, paged_sink, SdpaFwdKnobs(pack_gqa=True, split_kv=2)) is not None
    assert engines.mismatch(caps, paged_sink, SdpaFwdKnobs(pack_gqa=True, split_kv=1)) is not None
    assert sets(paged_sink) == []
    # Dense UNPADDED: the same split lead; the unsplit runner-up is the prefill kernel's fused gate, unpacked.
    dense = SdpaGraphFacts(**{**base, "has_paged_kv": False, "page_size": 0, "padded": False})
    assert engines.mismatch(caps, dense, SdpaFwdKnobs(pack_gqa=False, split_kv=1)) is None
    why = engines.mismatch(caps, dense, SdpaFwdKnobs(pack_gqa=True, split_kv=1))
    assert why and "decode tile" in why, why  # a packed unsplit gated plan does not exist on this row
    dense_sets = sets(dense)
    assert dense_sets[0] == (True, 16) and (False, 1) in dense_sets and (True, 1) not in dense_sets, dense_sets
    assert _split_points(caps, dense, 128, 128, 2, pack_g=12) == [16, 1]
    # Dense PADDED: the shared no-split rule holds, so the gated graph keeps the fused prefill kernel.
    padded_dense = SdpaGraphFacts(**{**base, "has_paged_kv": False, "page_size": 0})
    assert sets(padded_dense) == [(False, 1)]
    # The serving shape (b=32 x 2 KV heads, 32 tiles), whose ungated leading plan is UNSPLIT: gated, the leading plan is split 2.
    serving = SdpaGraphFacts(**{**base, "b": 32, "h_q": 32})
    assert sets(serving)[0] == (True, 2), sets(serving)
    assert sets(SdpaGraphFacts(**{**base, "b": 32, "h_q": 32, "has_epilogue_gate": False}))[0] == (True, 1)
    # Prefill-shaped gated graphs keep the old contract: unsplit, unpacked; paged + gate past the tile is declined.
    prefill = SdpaGraphFacts(**{**base, "s_q": 512, "s_kv": 512, "has_paged_kv": False, "page_size": 0, "padded": False})
    assert sets(prefill) == [(False, 1)]
    why = engines.mismatch(caps, prefill, SdpaFwdKnobs(split_kv=2))
    assert why and "gate" in why, why
    why = engines.mismatch(caps, SdpaGraphFacts(**{**base, "s_q": 512}), None)
    assert why and "paged" in why and "gate" in why, why


# --- the graph path on a Rubin GPU -------------------------------------------------------------


def _run_graph(
    *,
    B,
    H,
    KH,
    s_q,
    lens,
    d=D,
    page=16,
    hnd=True,
    dtype=torch.float16,
    stats=True,
    stats_log2=False,
    causal_br=False,
    window_left=None,
    window_right=None,
    sink=False,
    q_lens=None,
    expect=DECODE,
    seed=0,
    graph_kwargs=None,
    split_kv=None,
    pack_gqa=None,
    padded=True,
    capture=None,
    gate=None,
):
    """Build cuDNN's paged (``page`` > 0) or dense SDPA graph -- padded (per-batch
    ``seq_len_kv`` / ``seq_len_q``, the default) or, with ``padded=False`` on a dense cache
    of uniform tile-aligned ``lens``, UNPADDED (no padding mask at all: the one dense form
    the shared split rules admit) -- pin the Rubin FROST engine (its first entry: the
    heuristics' own choice, or with ``split_kv`` / ``pack_gqa`` its ranked entry carrying
    those knobs, a runner-up when the heuristics lead elsewhere), assert the serving
    template, execute under the D2H detector and compare O / Stats against the SM100
    suite's fp32 reference.  Returns the pinned plan (its knobs are what the row-contract
    tests read); ``capture`` (a dict) additionally receives the fp32 ``o`` ([B, S_q, H, d]),
    the ``lse`` ([B, H, S_q], or None) and the kernel ``module`` the executor loaded, for
    the bitwise comparisons.

    ``gate``: None = the plain sdpa graph; ``"random"`` = the fused epilogue-gate TAIL
    (``sdpa(virtual O_v) -> sigmoid(G) -> mul(O_v, s)``, ``cudnn._sdpa_tail``) with a +-2 sigma
    G in the IO dtype; a number = that gate logit on every element (+-1e4: an exactly
    saturated sigmoid, 0: exactly one half).  The reference is then multiplied by
    ``sigmoid(G)`` in fp32; the captured ``gate`` is the BSHD fp32 G."""
    import cudnn
    import cudnn.sdpa  # noqa: F401

    torch.manual_seed(seed)
    dev = "cuda"
    S = max(max(lens), 1)
    assert padded or (not page and set(lens) == {S} and S % 128 == 0 and q_lens is None), "the unpadded form: a dense cache of uniform tile-aligned lengths"
    scale = 1.0 / math.sqrt(d)
    q_gpu = torch.randn(B, s_q, H, d, device=dev, dtype=dtype).transpose(1, 2)  # BHSD strides over a BSHD buffer
    k_dense = torch.randn(B, S, KH, d, device=dev, dtype=dtype)
    v_dense = torch.randn(B, S, KH, d, device=dev, dtype=dtype)
    gate_gpu = None
    if gate is not None:
        gate_gpu = (
            (torch.randn(B, s_q, H, d, device=dev) * 2.0 if gate == "random" else torch.full((B, s_q, H, d), float(gate), device=dev)).to(dtype).transpose(1, 2)
        )
    o_gpu = torch.empty(B, s_q, H, d, device=dev, dtype=dtype).transpose(1, 2)
    seq_kv = torch.tensor(lens, dtype=torch.int32, device=dev)
    seq_q = torch.tensor(q_lens if q_lens is not None else [s_q] * B, dtype=torch.int32, device=dev)
    # ``sink``: False = no sink, True = random logits, a number = that logit on every head.
    sinks = None if sink is False else (torch.randn(H, device=dev) * 2.0 if sink is True else torch.full((H,), float(sink), device=dev))

    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, **(graph_kwargs or {}))
    q = g.tensor_like(q_gpu)
    kw = dict(name="sdpa", generate_stats=stats, attn_scale=scale, use_padding_mask=padded, stats_use_log2=stats_log2)
    if page:
        k_c, v_c, bt = _pools(k_dense, v_dense, page, hnd, seed)
        k, v = g.tensor_like(k_c), g.tensor_like(v_c)
        tk, tv = g.tensor_like(bt), g.tensor_like(bt)
        kw.update(paged_attention_k_table=tk, paged_attention_v_table=tv, paged_attention_max_seq_len_kv=S)
    else:
        k_c, v_c = k_dense.transpose(1, 2), v_dense.transpose(1, 2)  # BHSD view of the BSHD buffers
        k, v = g.tensor_like(k_c), g.tensor_like(v_c)
    slq, slk = seq_q.view(B, 1, 1, 1), seq_kv.view(B, 1, 1, 1)
    kw.update(q=q, k=k, v=v)
    if padded:
        sq_t, sk_t = g.tensor_like(slq), g.tensor_like(slk)
        kw.update(seq_len_q=sq_t, seq_len_kv=sk_t)
    if causal_br:
        kw["use_causal_mask_bottom_right"] = True
    elif window_right is not None:
        kw["use_causal_mask"] = window_right == 0
        if window_right > 0:
            kw["diagonal_band_right_bound"] = window_right
    if window_left is not None:
        kw["sliding_window_length"] = window_left + 1
    sink_t = None
    if sinks is not None:
        sink_gpu = sinks.view(1, H, 1, 1).contiguous()
        sink_t = g.tensor_like(sink_gpu)
        kw["sink_token"] = sink_t
    o, st = g.sdpa(**kw)
    gt = None
    if gate_gpu is not None:
        # The three-node gate tail: the sdpa output stays VIRTUAL but declared (dim + stride),
        # the mul output is the graph's real O (docs/operations/Attention.md, "Fused epilogue gate").
        o.set_dim(q_gpu.shape).set_stride(q_gpu.stride())
        gt = g.tensor_like(gate_gpu)
        o = g.mul(a=o, b=g.sigmoid(input=gt, name="sig"), name="gated")
        o.set_data_type(io)
    o.set_output(True).set_dim(q_gpu.shape).set_stride(q_gpu.stride())
    stats_gpu = None
    if stats:
        stats_gpu = torch.empty(B, H, s_q, 1, device=dev, dtype=torch.float32)
        st.set_output(True).set_dim(stats_gpu.shape).set_stride(stats_gpu.stride()).set_data_type(cudnn.data_type.FLOAT)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    plan = select_engine(g, _engine(), pack_gqa=pack_gqa, split_kv=split_kv)
    idx = g._plan_index
    g.check_support()
    g.build_plans()
    assert _served_by(g, idx) == expect, f"plan {g.get_plan_name_at_index(idx)} served by {_served_by(g, idx)}, expected {expect}"
    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    vp = {q: q_gpu, k: k_c, v: v_c, o: o_gpu}
    if padded:
        vp.update({sq_t: slq, sk_t: slk})
    if page:
        vp.update({tk: bt, tv: bt})
    if stats:
        vp[st] = stats_gpu
    if sinks is not None:
        vp[sink_t] = sink_gpu
    if gt is not None:
        vp[gt] = gate_gpu
    # Rule 3: the execute path reads the per-batch lengths and page tables on
    # device; any blocking D2H here is a bug, not a slow path.
    torch.cuda.set_sync_debug_mode("error")
    try:
        g.execute(vp, ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()

    ref_o, ref_lse = _ref(
        q_gpu.transpose(1, 2), k_dense, v_dense, lens, q_lens, scale, causal_br=causal_br, window_left=window_left, window_right=window_right, sinks=sinks
    )
    gate_f32 = None
    if gate_gpu is not None:
        # The reference gates its fp32 O (the gated prefill kernels' and the combine's
        # convention: one rounding, of O32 * sigmoid(G)); the suite tolerance absorbs the
        # one-ulp difference to a reference that would round O to the IO dtype first.
        gate_f32 = gate_gpu.transpose(1, 2).float()
        ref_o = ref_o * torch.sigmoid(gate_f32)
    out = o_gpu.transpose(1, 2).float()
    assert not torch.isnan(out).any(), "NaN in O"
    torch.testing.assert_close(out, ref_o, atol=2e-2 if dtype == torch.float16 else 5e-2, rtol=0)
    dead = torch.isinf(ref_lse)
    if dead.any():
        assert out.transpose(1, 2)[dead].abs().max().item() == 0.0, "dead rows must write O := 0"
    if stats:
        got_lse = stats_gpu.view(B, H, s_q)
        exp_lse = ref_lse * (1.4426950408889634 if stats_log2 else 1.0)
        torch.testing.assert_close(got_lse[~dead], exp_lse[~dead], atol=5e-3, rtol=0)
        if dead.any():
            assert torch.isinf(got_lse[dead]).all() and (got_lse[dead] < 0).all(), "dead rows must write LSE := -inf"
    if capture is not None:
        capture.update(
            o=out.clone(),
            lse=stats_gpu.view(B, H, s_q).clone() if stats else None,
            module=g._compiled_plans[idx]._compiled.kernel_module,
            q=q_gpu.transpose(1, 2).clone(),
            k=k_dense.clone(),
            v=v_dense.clone(),
            lens=list(lens),
            scale=scale,
            gate=gate_f32.clone() if gate_f32 is not None else None,
        )
    return plan


def _sm_count():
    return torch.cuda.get_device_properties(0).multi_processor_count


def _assert_decode_tile_plan(plan, *, G, units=None, kv_tiles=None, replay=False, gated=False, q_tile=16):
    """The SM100 suite's policy on the Rubin row, now that it claims PackGQA + split-KV on the tile
    (the inversion of the former unpacked / unsplit pin): the plan is PACKED exactly when there is a
    group to pack (``G > 1``; MHA stays the bit-exact unpacked fold) and, for a shape the decode split
    model governs (``units`` x ``kv_tiles`` given: an unmasked, sink-free paged or unpadded cache), its
    split is that model's eager-safe choice -- the captured optimum under ``replay`` -- at THIS part's
    SM count.  Shapes the shared no-split rules bind (a padded dense cache, a sink, per-batch Q lengths)
    pass no model inputs and are pinned unsplit by their caller.  ``gated``: the graph carries the
    fused-gate tail, which rides the tile through its split only (the combine applies the gate), so
    the model's "do not split" reads as the smallest split, 2.  ``units`` counts the TOKEN units too
    (batch x head groups x units per group); ``q_tile`` is the tile's N extent the model costs (32 on
    the Rubin MTP forms)."""
    assert (plan.knobs.pack_gqa is True) == (G > 1), (G, plan.knobs)
    if units is not None:
        want = choose_decode_tile_split_kv(units=units, kv_tiles=kv_tiles, sm_count=_sm_count(), q_tile=q_tile, **({"launch_cost": 0.0} if replay else {}))
        if gated:
            want = max(want, 2)
        assert (plan.knobs.split_kv or 1) == want, (plan.knobs, want)


def _ulp(mag, mbits):
    """The output dtype's ulp at each element's own binade (``mag`` fp64, non-negative), floored at the
    dtype's SUBNORMAL step: f16 (10 mantissa bits, min normal 2^-14) spaces every value below 6.1e-5 by
    2^-24, bf16 (7 bits, min normal 2^-126) by 2^-133 -- a cancellation element of O at 1e-5 is an f16
    subnormal, and its one-ulp rounding flip is 6e-8, not the 1e-8 the normal-range formula would say."""
    _, exp = torch.frexp(mag.cpu())  # mag = m * 2**exp, m in [0.5, 1): the dtype's ulp in that binade is 2**(exp - 1 - mbits)
    emin = {7: -126, 10: -14}[mbits]
    ulp = torch.ldexp(torch.ones_like(mag.cpu()), exp - 1 - mbits)
    return torch.clamp(ulp, min=2.0 ** (emin - mbits)).to(mag.device)


def _assert_split_matches_unsplit(a, b, dtype, *, q, k, v, lens, scale, what, gate=None):
    """The DERIVED budget between the split and the unsplit plan of one graph, per output element:

        |a - b| <= ulp_dtype(max(|a|, |b|)) + sigmoid(g) * 2 * eps_P * sum_j p_j |v_j| / l

    Two terms, two causes.  (1) The output is rounded ONCE to the output dtype from two fp32
    values that are the same quantity associated differently (the split path renormalises each
    partial by exp(lse_s - lse) in the combine, the unsplit path rescales its running accumulator
    per KV tile), so the two roundings agree or differ by one output ulp at the element's own
    binade.  (2) Before that rounding the two fp32 values are NOT equal to fp32 precision: the
    tile quantizes P = exp(s - m) to the half IO dtype for BMM2 (the P^T operand must match V's
    dtype) at the running max m of ITS OWN split, so every p_j carries a relative rounding
    error |delta_j| <= eps_P = 2^-(mbits+1) that differs between the two paths; each path's
    sum_j p_j (1 + delta_j) v_j / l sits within eps_P * sum_j p_j |v_j| / l of the exact value,
    so their DIFFERENCE is within TWICE that -- the row's softmax-weighted mean of |V|, computed
    here from the same inputs in fp32, a rigorous (not statistical) bound on the two-path
    difference (one eps_P would bound each path alone).  Term (2) is what makes an element near
    ZERO (cancellation) differ by far more than one ulp of its own tiny magnitude; it also exceeds
    one output ulp of a mid-magnitude element (|O| ~ 2^-6: eps_P * mean|V| ~ 2^-8 in bf16 against
    a 2^-13 ulp), so the ONE assertion is the two-term budget -- the per-binade ulp counts are
    reported, not pinned.  Still 6x (bf16) / 20x (f16) under the suite's reference tolerance, and
    any structural defect (a wrong combine weight, a dropped split, a stale partial) lands orders
    of magnitude above it.  Both plans match the fp32 reference on their own; the measured
    magnitudes are printed so the log carries them.  Dense / padded rows without a causal or
    window mask only.

    ``gate`` (the fp32 BSHD G of a GATED graph): both paths multiply their fp32 value by the
    SAME sigmoid(G) -- the same MUFU.TANH arithmetic on the same input -- before the single
    rounding, so term (2) scales by sigmoid(g) per element and term (1) is unchanged."""
    mbits = {torch.bfloat16: 7, torch.float16: 10}[dtype]
    eps_p = 2.0 ** -(mbits + 1)
    B, s_q, H, d = q.shape
    KH = k.shape[2]
    assert s_q == 1 and H % KH == 0, (q.shape, k.shape)
    G = H // KH
    qf = q.float().view(B, H, d)  # [B, H, d]
    kf = k.float().transpose(1, 2)  # [B, KH, S, d]
    vf = v.float().transpose(1, 2).abs()  # [B, KH, S, d]
    kv_h = torch.arange(H, device=q.device) // G
    scores = torch.einsum("bhd,bhsd->bhs", qf, kf[:, kv_h]) * scale  # [B, H, S]
    S = kf.shape[2]
    keep = torch.arange(S, device=q.device)[None, :] < torch.tensor(lens, device=q.device)[:, None]  # [B, S]
    scores = scores.masked_fill(~keep[:, None, :], float("-inf"))
    prob = torch.softmax(scores, dim=-1)  # rows with no live key are NaN: excluded below
    pv_abs = torch.einsum("bhs,bhsd->bhd", prob.nan_to_num(0.0), vf[:, kv_h])  # sum_j p_j |v_j| / l, [B, H, d]
    live = torch.isfinite(prob).all(dim=-1)  # [B, H]
    a64, b64 = a.double().view(B, H, d), b.double().view(B, H, d)
    mag = torch.maximum(a64.abs(), b64.abs())
    ulp = _ulp(mag, mbits)
    weight = torch.sigmoid(gate.double().view(B, H, d)) if gate is not None else 1.0
    budget = ulp + weight * (2.0 * eps_p) * pv_abs.double()
    diff = (a64 - b64).abs()
    diff = diff[live]
    budget, ulp, mag = budget[live], ulp[live], mag[live]
    ratio = (diff / budget).max().item()
    big = mag >= 2.0**-6
    big_ulps = (diff[big] / ulp[big]).max().item() if big.any() else 0.0
    small_abs = diff[~big].max().item() if (~big).any() else 0.0
    print(
        f"{what}: max |diff| = {diff.max().item():.3e}; {(diff > 0).double().mean().item():.4%} of the elements differ; "
        f"elements with |O| >= 2^-6: max {big_ulps:.2f} output ulp; elements below: max |diff| {small_abs:.3e} "
        f"(eps_P * mean|V| budget there {budget[~big].min().item() if (~big).any() else 0.0:.3e}..{budget[~big].max().item() if (~big).any() else 0.0:.3e}); "
        f"max |diff| / budget = {ratio:.3f}"
    )
    assert ratio <= 1.0, (what, ratio)


# --- accept: the decode tile serves the contract on Rubin --------------------------------------


@_gpu
@pytest.mark.parametrize("hnd", [False, True], ids=["NHD", "HND"])
@pytest.mark.parametrize("page", [16, 32, 64, 128])
def test_decode_graph_paged_page_sizes(page, hnd):
    """32/2-head decode (PackGQA 16:1 on the tile) over pages of every admitted size: mixed lengths
    incl. 0 and 1, a tile-unaligned tail, a length ending on a page/tile boundary; Stats out; the
    split is the decode model's at 10 units x 8 tiles."""
    _assert_decode_tile_plan(_run_graph(B=5, H=32, KH=2, s_q=1, lens=[300, 77, 0, 1, 1024], page=page, hnd=hnd), G=16, units=5 * 2, kv_tiles=8)


@_gpu
@pytest.mark.parametrize(("H", "KH"), [(8, 1), (16, 1), (32, 2), (16, 4), (4, 4), (24, 2)], ids=["8to1", "16to1", "32to2", "16to4", "mha", "24to2"])
def test_decode_graph_head_groups(H, KH):
    """PackGQA 8:1 and 16:1 (one and two tokens per 16-row tile at S_q = 1), 32:2, a 4-wide group,
    MHA (one row per unit, unpacked) and the 24/2 geometry (G = 12: 12 live rows + 4 zero tail rows
    per unit -- the whole-group packing no prefill tile has) -- every one decode-shaped on the Rubin
    row, packed exactly when there is a group, split by the decode model (3 x KH units x 8 tiles)."""
    G = H // KH
    _assert_decode_tile_plan(_run_graph(B=3, H=H, KH=KH, s_q=1, lens=[1000, 129, 640], page=32, dtype=torch.bfloat16), G=G, units=3 * (H // G), kv_tiles=8)


@_gpu
@pytest.mark.parametrize(("s_q", "H"), [(2, 16), (4, 8)], ids=["sq2_8to1", "sq4_4to1"])
def test_decode_graph_mtp_bottom_right(s_q, H):
    """MTP: S_q in {2, 4} bottom-right causal over a paged cache, packed 8:1 and 4:1 (16 rows either
    way: two and four tokens per 16-row tile), with per-batch Q lengths below S_q (dense padded-Q
    trim: O := 0 / LSE := -inf past them, the diagonal anchored at seq_len_kv[b] - seq_len_q[b]);
    the per-batch Q lengths bind the shared no-split rule."""
    plan = _run_graph(B=3, H=H, KH=2, s_q=s_q, lens=[700, 130, 5], q_lens=[s_q, max(1, s_q - 1), 0], page=16, causal_br=True)
    _assert_decode_tile_plan(plan, G=H // 2)
    assert (plan.knobs.split_kv or 1) == 1, plan.knobs


def _mtp_module_facts(mod):
    return (mod.N_Q, mod.HEADS_PER_TILE, mod.Q_BOX_TOKENS, mod.Q_BOX_ROWS, mod.Q_TOKEN_UNITS, mod.CFG.SOFTMAX_WARPS, mod.CFG.TOTAL_WARPS)


@_gpu
@pytest.mark.parametrize("s_q", [2, 3, 4], ids=["sq2_one_unit", "sq3_two_units_half_filled", "sq4_two_units"])
@pytest.mark.parametrize("page", [0, 16], ids=["dense_padded", "page16_HND"])
def test_decode_graph_mtp_qwen_24_2_rides_the_32_column_tile_in_token_units(s_q, page):
    """The MTP step of the 24/2 geometry (G = 12) on the Rubin route: S_q = 2 (24 rows) rides the
    32-column tile in ONE unit, S_q = 3 / 4 (36 / 48 rows) in TWO token units of two tokens x 12
    heads (the second unit half-filled at S_q = 3), bottom-right causal over a dense padded or a
    paged cache with per-batch Q lengths below S_q (dense padded-Q trim across the units: O := 0 /
    LSE := -inf past them, the diagonal anchored at seq_len_kv[b] - seq_len_q[b]); the plan is
    PACKED 12:1 and the module the executor loaded is the 32-column tile with the token-unit axis
    (N_Q = 32, HEADS_PER_TILE = 12, Q_BOX_TOKENS = 2, 24 live + 8 zero tail rows per unit, 10 warps).
    The per-batch Q lengths bind the shared no-split rule (unsplit)."""
    cap = {}
    plan = _run_graph(
        B=3, H=24, KH=2, s_q=s_q, lens=[700, 130, 5], q_lens=[s_q, max(1, s_q - 1), 1], page=page, hnd=True, causal_br=True, dtype=torch.bfloat16, capture=cap
    )
    _assert_decode_tile_plan(plan, G=12)
    assert (plan.knobs.split_kv or 1) == 1, plan.knobs
    assert _mtp_module_facts(cap["module"]) == (32, 12, 2, 24, True, 8, 10), _mtp_module_facts(cap["module"])


@_gpu
@pytest.mark.parametrize("s_q", [2, 4], ids=["sq2", "sq4"])
def test_decode_graph_mtp_qwen_24_2_splits_in_token_units(s_q):
    """The MTP step over an UNPADDED dense cache and over pages (the forms the split rules admit):
    the plan is packed 12:1 on the 32-column tile and its split is the decode model's at the
    token-unit geometry -- B x 2 KV heads x ceil(S_q / 2) units, the 32-column tile's cost -- the
    fp32 partials of every token unit recombined by the shared combine against the reference."""
    units = 2 * 2 * -(-s_q // 2)
    plan = _run_graph(B=2, H=24, KH=2, s_q=s_q, lens=[4096, 4096], page=0, padded=False, causal_br=True, dtype=torch.bfloat16)
    _assert_decode_tile_plan(plan, G=12, units=units, kv_tiles=32, q_tile=32)
    plan = _run_graph(B=2, H=24, KH=2, s_q=s_q, lens=[4000, 1000], page=16, causal_br=True, dtype=torch.bfloat16)
    _assert_decode_tile_plan(plan, G=12, units=units, kv_tiles=32, q_tile=32)
    if (plan.knobs.split_kv or 1) == 1:
        plan = _run_graph(B=2, H=24, KH=2, s_q=s_q, lens=[4000, 1000], page=16, causal_br=True, dtype=torch.bfloat16, split_kv=4)
        assert (plan.knobs.split_kv or 1) == 4, plan.knobs


@_gpu
def test_decode_graph_mtp_token_unit_forms_equal_each_other_and_the_unpacked_form_bitwise(monkeypatch):
    """S_q = 4 at 24/2 over pages, three lowerings of ONE graph: (a) the Rubin route -- two token
    units of two tokens on the 32-column tile, (b) four single-token units of the 16-column tile
    (the route pinned to the narrow tile for this test), (c) the unpacked plan (one unit of four
    tokens per (batch, Q head), the pre-route lead).  A column of S^T does not depend on which N
    column, which unit or which tile width it occupies, and every unit streams the same KV tiles in
    the same order, so O and LSE are BITWISE equal across the three -- the token-unit axis moves
    rows between CTAs, it never changes the arithmetic.  All pinned unsplit."""
    from cudnn.sdpa.fwd import api_dsl, config_sm107

    kw = dict(B=3, H=24, KH=2, s_q=4, lens=[1000, 129, 640], page=32, dtype=torch.bfloat16, seed=5, causal_br=True)
    wide, narrow, unpacked = {}, {}, {}
    pw = _run_graph(**kw, pack_gqa=True, split_kv=1, capture=wide)
    assert pw.knobs.pack_gqa is True and _mtp_module_facts(wide["module"])[:4] == (32, 12, 2, 24)
    # (b): the narrow tile in four token units -- the adapter's route pinned to N = 16 for this geometry (the engine's
    # row rule and the heuristics keep the real route: the packed set is eligible either way, the lowering decides the width).
    monkeypatch.setattr(
        api_dsl, "_decode_d256_q_tile_sm107", lambda s_q, pack_g: 16 if (s_q, pack_g) == (4, 12) else config_sm107.decode_d256_q_tile(s_q, pack_g)
    )
    pn = _run_graph(**kw, pack_gqa=True, split_kv=1, capture=narrow)
    monkeypatch.undo()
    assert pn.knobs.pack_gqa is True and _mtp_module_facts(narrow["module"])[:4] == (16, 12, 1, 12), _mtp_module_facts(narrow["module"])
    pu = _run_graph(**kw, pack_gqa=False, split_kv=1, capture=unpacked)
    assert pu.knobs.pack_gqa is not True and _mtp_module_facts(unpacked["module"])[:4] == (16, 1, 16, 16)
    for name, other in (("four single-token units (N = 16)", narrow), ("unpacked", unpacked)):
        assert torch.equal(wide["o"], other["o"]), f"two token units != {name}: max |diff| {(wide['o'] - other['o']).abs().max().item():.3e}"
        assert torch.equal(wide["lse"], other["lse"]), f"two token units != {name} LSE: max |diff| {(wide['lse'] - other['lse']).abs().max().item():.3e}"


@_gpu
def test_decode_graph_mtp_qwen_24_2_paged_equals_dense_bitwise():
    """The two-token-unit form over a dense padded cache and over page pools (S_q = 4, 24/2, packed
    12:1, both unsplit): the paged specialization only redirects the K/V tile loads, so O and LSE
    are BITWISE equal across the units."""
    kw = dict(B=3, H=24, KH=2, s_q=4, lens=[700, 130, 5], dtype=torch.bfloat16, seed=3, causal_br=True)
    dense, paged = {}, {}
    pd = _run_graph(**kw, page=0, capture=dense)
    pp = _run_graph(**kw, page=16, hnd=True, split_kv=1, capture=paged)
    _assert_decode_tile_plan(pd, G=12)
    _assert_decode_tile_plan(pp, G=12)
    assert _mtp_module_facts(dense["module"])[:4] == (32, 12, 2, 24) == _mtp_module_facts(paged["module"])[:4]
    assert torch.equal(dense["o"], paged["o"]), f"paged != dense: max |diff| {(dense['o'] - paged['o']).abs().max().item():.3e}"
    assert torch.equal(dense["lse"], paged["lse"]), f"paged != dense LSE: max |diff| {(dense['lse'] - paged['lse']).abs().max().item():.3e}"


@_gpu
def test_decode_graph_mtp_32_2_rides_the_32_column_tile_in_one_unit():
    """32/2 at S_q = 2 (32 packed rows, the former unpacked pin): one unit of the 32-column tile,
    packed 16:1 (Q_BOX_TOKENS = 2, no tail rows), bottom-right over pages, against the reference."""
    cap = {}
    plan = _run_graph(B=3, H=32, KH=2, s_q=2, lens=[700, 130, 5], page=16, causal_br=True, dtype=torch.float16, capture=cap)
    _assert_decode_tile_plan(plan, G=16)
    assert _mtp_module_facts(cap["module"]) == (32, 16, 2, 32, True, 8, 10), _mtp_module_facts(cap["module"])


@_gpu
@pytest.mark.parametrize("page", [0, 16], ids=["dense_padded", "page16_HND"])
@pytest.mark.parametrize(
    ("H", "KH", "s_q", "facts"),
    [(8, 1, 6, (32, 8, 4, 32, True, 8, 10)), (12, 2, 10, (32, 6, 5, 30, True, 8, 10)), (4, 4, 24, (32, 1, 32, 32, True, 8, 10))],
    ids=["8to1_sq6_two_units_half_filled", "12to2_sq10_two_units_tail_rows", "mha_sq24_one_wide_unit"],
)
def test_decode_graph_token_units_at_the_other_geometries_inside_the_route(H, KH, s_q, facts, page):
    """The route admits more geometries than the 24/2 steps the MTP perf table measured, and each
    class needs a GPU cell of its own: 8:1 at S_q = 6 (G = 8 divides the tile: two units of four
    tokens x 8 heads, the second half-filled, no tail rows -- the liveness guard folds out), 12:2 at
    S_q = 10 (G = 6: two units of five tokens x 6 heads with TWO zero tail rows per unit -- the
    tail-row liveness guard's second instance: rows 30 / 31 of unit 0 alias unit 1's first token on
    heads 0 / 1) and an MHA bottom-right step of 24 tokens (one unit of the 32-column tile, unpacked,
    24 live rows), over a dense padded and a paged cache with per-batch Q lengths below S_q (the
    trim across the units; for the two-unit cases one sequence's second unit lies entirely past its
    Q length and must write O := 0 / LSE := -inf).  Correctness only -- these forms are the same tile
    with the same stream count per group as the measured 24/2 form, but their perf is not in the
    table.  The per-batch Q lengths bind the shared no-split rule (unsplit)."""
    cap = {}
    plan = _run_graph(
        B=3, H=H, KH=KH, s_q=s_q, lens=[700, 130, 5], q_lens=[s_q, max(1, s_q - 1), 1], page=page, hnd=True, causal_br=True, dtype=torch.bfloat16, capture=cap
    )
    _assert_decode_tile_plan(plan, G=H // KH)
    assert (plan.knobs.split_kv or 1) == 1, plan.knobs
    assert _mtp_module_facts(cap["module"]) == facts, _mtp_module_facts(cap["module"])


@_gpu
@pytest.mark.parametrize(("H", "KH", "s_q"), [(8, 1, 6), (12, 2, 10)], ids=["8to1_sq6", "12to2_sq10"])
def test_decode_graph_token_units_at_the_other_geometries_split_in_token_units(H, KH, s_q):
    """The split forms of the two-unit geometries above over pages (the form the split rules
    admit): the plan is packed on the 32-column tile and its split is the decode model's at the
    token-unit geometry -- B x KH head groups x 2 units, the 32-column tile's cost -- the fp32
    partials of every token unit (the G = 6 units with their tail rows included) recombined by the
    shared combine against the reference; pinned to split 4 when the model picks unsplit."""
    units = 2 * KH * 2
    plan = _run_graph(B=2, H=H, KH=KH, s_q=s_q, lens=[4000, 1000], page=16, causal_br=True, dtype=torch.bfloat16)
    _assert_decode_tile_plan(plan, G=H // KH, units=units, kv_tiles=32, q_tile=32)
    if (plan.knobs.split_kv or 1) == 1:
        plan = _run_graph(B=2, H=H, KH=KH, s_q=s_q, lens=[4000, 1000], page=16, causal_br=True, dtype=torch.bfloat16, split_kv=4)
        assert (plan.knobs.split_kv or 1) == 4, plan.knobs


@_gpu
def test_decode_graph_sliding_window_bottom_right():
    """Sliding window (left bound) on the bottom-right diagonal, S_q = 1, packed 16:1."""
    _assert_decode_tile_plan(_run_graph(B=2, H=32, KH=2, s_q=1, lens=[700, 130], page=16, causal_br=True, window_left=200), G=16)


@_gpu
def test_decode_graph_right_band_dense():
    """Top-left causal widened by a right band on a dense padded cache, S_q = 2, packed 4:1 (8 rows);
    the padded dense cache binds the shared no-split rule."""
    plan = _run_graph(B=2, H=8, KH=2, s_q=2, lens=[300, 129], page=0, window_right=100)
    _assert_decode_tile_plan(plan, G=4)
    assert (plan.knobs.split_kv or 1) == 1, plan.knobs


@_gpu
def test_decode_graph_sink_dense():
    """Attention sink folded once per Q row over a dense padded cache (S_q = 2, packed 8:1); a
    keyless batch keeps the sink's finite LSE and O := 0; a sink never splits."""
    plan = _run_graph(B=3, H=16, KH=2, s_q=2, lens=[700, 0, 300], page=0, sink=True)
    _assert_decode_tile_plan(plan, G=8)
    assert (plan.knobs.split_kv or 1) == 1, plan.knobs


@_gpu
@pytest.mark.parametrize("s_q", [1, 2])
def test_decode_graph_sink_paged(s_q):
    """The sink fold over a paged cache (page 16) at S_q = 1 and 2 -- the one paged + sink form
    the Rubin row serves (the paged PREFILL pipeline there is THD-only without a sink; the
    decode tile walks the block table itself).  A keyless batch keeps the sink's finite LSE
    and O := 0; packed 8:1, and a sink never splits."""
    plan = _run_graph(B=3, H=16, KH=2, s_q=s_q, lens=[700, 0, 300], page=16, sink=True)
    _assert_decode_tile_plan(plan, G=8)
    assert (plan.knobs.split_kv or 1) == 1, plan.knobs


@_gpu
@pytest.mark.parametrize("sink_logit", [-100.0, -1000.0, -2.0], ids=["sink_-100", "sink_-1000", "sink_-2_control"])
def test_decode_graph_sink_keyless_rows_keep_the_sink_logit(sink_logit):
    """A row with no live key and a sink has exactly one softmax column, so its LSE is the
    sink logit itself and O := 0 -- for the empty batch (KV length 0) and for the
    bottom-right row a one-key batch masks entirely alike; finite sinks far below zero."""
    _assert_decode_tile_plan(_run_graph(B=3, H=16, KH=2, s_q=2, lens=[0, 1, 128], page=0, sink=sink_logit, causal_br=True), G=8)


@_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["f16", "bf16"])
def test_decode_graph_stats_log2(dtype):
    """Base-2 Stats (stats_use_log2) on the decode tile, both half dtypes, packed 16:1."""
    _assert_decode_tile_plan(_run_graph(B=2, H=32, KH=2, s_q=1, lens=[700, 130], page=16, dtype=dtype, stats_log2=True), G=16, units=4, kv_tiles=6)


@_gpu
def test_decode_graph_dense_padded_no_stats():
    plan = _run_graph(B=2, H=32, KH=2, s_q=1, lens=[1000, 77], page=0, stats=False, dtype=torch.bfloat16)
    _assert_decode_tile_plan(plan, G=16)
    assert (plan.knobs.split_kv or 1) == 1, plan.knobs  # a padded dense cache: the shared no-split rule


@_gpu
def test_decode_graph_serving_shape_leads_unsplit_with_the_split_as_runner_up():
    """b=32 x 2 KV heads over 4096 keys (the serving shape) at S_q=1 on the Rubin row: the LEADING
    plan is packed 16:1 and does not split -- split 2 saves GPU time but costs an eager caller a
    second host launch per execute -- and the captured caller's split-2 plan is the runner-up,
    reachable by select_plan.  Both run and match the reference.  The literal splits are what the
    decode model answers at 148 / 204 / 212 SMs alike (test_split_kv_heuristic pins the policy); on
    another part the plans must still agree with the model.  The engine's ranked entries carry
    exactly that: a packed unsplit lead and a packed split-2 entry, every one admissible."""
    import cudnn

    sm = _sm_count()
    shape = dict(B=32, H=32, KH=2, s_q=1, lens=[4096] * 32, page=16, stats=False, dtype=torch.bfloat16)
    lead = _run_graph(**shape)
    _assert_decode_tile_plan(lead, G=16, units=64, kv_tiles=32)
    captured = choose_decode_tile_split_kv(units=64, kv_tiles=32, sm_count=sm, launch_cost=0.0)
    runner = _run_graph(**shape, split_kv=captured)
    assert runner.knobs.split_kv == captured and runner.knobs.pack_gqa is True, runner.knobs
    if sm in (148, 204, 212):
        assert (lead.knobs.split_kv, runner.knobs.split_kv) == (1, 2), (lead.knobs, runner.knobs)
    # The engine's ranked entries for the serving shape: the packed unsplit lead, a packed split entry.
    torch.manual_seed(0)
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q = g.tensor(dim=[32, 32, 1, D], stride=[32 * D, D, 32 * D, 1], data_type=cudnn.data_type.BFLOAT16, name="q")
    k_dense = torch.randn(32, 4096, 2, D, device="cuda", dtype=torch.bfloat16)
    k_c, v_c, bt = _pools(k_dense, k_dense, 16, True, 0)
    k, v = g.tensor_like(k_c), g.tensor_like(v_c)
    tk, tv = g.tensor_like(bt), g.tensor_like(bt)
    sq_t = g.tensor(dim=[32, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32, name="sq")
    sk_t = g.tensor(dim=[32, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32, name="sk")
    o, _ = g.sdpa(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        generate_stats=False,
        attn_scale=1.0 / math.sqrt(D),
        use_padding_mask=True,
        seq_len_q=sq_t,
        seq_len_kv=sk_t,
        paged_attention_k_table=tk,
        paged_attention_v_table=tv,
        paged_attention_max_seq_len_kv=4096,
    )
    o.set_output(True).set_dim([32, 32, 1, D]).set_stride([32 * D, D, 32 * D, 1])
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    ours = [g.plans[i].knobs for i, n in enumerate(names) if _is_plan_for(n, _engine())]
    assert ours and ours[0].pack_gqa is True and (ours[0].split_kv or 1) == 1, ours
    assert any(kn.pack_gqa is True and kn.split_kv == captured for kn in ours), ours
    # The same shape one token wider (32/2 at S_q=2 bottom-right: 32 packed rows) is the 32-column
    # tile's -- ROUTED on the Rubin row (config_sm107.decode_d256_q_tile), so the step packs 16:1 on
    # the 32-column tile in one unit (two tokens x 16 heads, no tail rows) and takes the decode split
    # model's choice at that tile's cost (64 units x 32 tiles, q_tile 32).  The former pin ("rides the
    # tile UNPACKED, unsplit") inverted with the route, as announced.
    mtp = _run_graph(B=32, H=32, KH=2, s_q=2, lens=[4096] * 32, page=16, stats=False, dtype=torch.bfloat16, causal_br=True)
    _assert_decode_tile_plan(mtp, G=16, units=64, kv_tiles=32, q_tile=32)
    # Its 16-row sibling (16/2 at S_q=2: 8:1 packing) is decode-shaped PACKED and follows the serving
    # shape's policy: unsplit lead, split-2 runner-up.
    mtp16 = _run_graph(B=32, H=16, KH=2, s_q=2, lens=[4096] * 32, page=16, stats=False, dtype=torch.bfloat16, causal_br=True)
    _assert_decode_tile_plan(mtp16, G=8, units=64, kv_tiles=32)


@_gpu
def test_decode_graph_serving_shape_leads_with_the_split_when_replay_is_expected():
    """The serving shape on a graph created with is_cuda_graph_replay_expected=True: the captured
    caller's optimum (split 2) LEADS and the eager-safe unsplit plan follows as a runner-up,
    reachable by select_plan.  The hint changes no numerics: both plans run and match the reference."""
    sm = _sm_count()
    shape = dict(B=32, H=32, KH=2, s_q=1, lens=[4096] * 32, page=16, stats=False, dtype=torch.bfloat16, graph_kwargs=dict(is_cuda_graph_replay_expected=True))
    lead = _run_graph(**shape)
    _assert_decode_tile_plan(lead, G=16, units=64, kv_tiles=32, replay=True)
    eager = choose_decode_tile_split_kv(units=64, kv_tiles=32, sm_count=sm)
    runner = _run_graph(**shape, split_kv=eager)
    assert runner.knobs.split_kv == eager and runner.knobs.pack_gqa is True, runner.knobs
    if sm in (148, 204, 212):
        assert (lead.knobs.split_kv, runner.knobs.split_kv) == (2, 1), (lead.knobs, runner.knobs)


@_gpu
def test_decode_graph_small_batch_splits_and_recombines():
    """b=8 x 2 KV heads is 16 units: unsplit they would stream 32 tiles each on 16 of the SMs, so
    the decode model splits (8 ways: 128 CTAs, and the GPU saving covers the second launch) and the
    recombined O / LSE over mixed lengths (incl. 1 and 77) match the reference."""
    plan = _run_graph(B=8, H=32, KH=2, s_q=1, lens=[4096, 4000, 129, 1, 2048, 4096, 300, 77], page=16, dtype=torch.bfloat16)
    _assert_decode_tile_plan(plan, G=16, units=16, kv_tiles=32)
    if _sm_count() in (148, 204, 212):
        assert plan.knobs.split_kv == 8, plan.knobs


@_gpu
def test_decode_graph_deep_split_empty_ranges():
    """Six units over a 4096-key table: the decode rule splits 16 ways (96 CTAs), so the 1- and
    129-key batches leave most split ranges empty next to live ones; those must yield -inf / 0
    partials the combine ignores."""
    plan = _run_graph(B=3, H=32, KH=2, s_q=1, lens=[4096, 1, 129], page=16)
    _assert_decode_tile_plan(plan, G=16, units=6, kv_tiles=32)
    assert plan.knobs.split_kv >= (8 if _sm_count() >= 96 else 2), plan.knobs


@_gpu
def test_decode_graph_mha_unpacked_split_recombines():
    """ONE unpacked graph-path split (the row's split claim reads 'packed or not'): MHA 4/4 at B = 2
    over a 4096-key table -- 8 units x 32 tiles, which the decode model splits 16 ways, unpacked --
    recombines to the reference: the kernel / adapter partial-dtype and workspace agreement a
    template-level split test cannot see, on the unpacked form."""
    plan = _run_graph(B=2, H=4, KH=4, s_q=1, lens=[4096, 3000], page=16, dtype=torch.bfloat16)
    _assert_decode_tile_plan(plan, G=1, units=8, kv_tiles=32)
    assert plan.knobs.pack_gqa is not True and plan.knobs.split_kv >= 2, plan.knobs
    if _sm_count() in (148, 204, 212):
        assert plan.knobs.split_kv == 16, plan.knobs


# --- the three equalities the row's claim rests on ------------------------------------------------


@_gpu
def test_decode_graph_qwen_24_2_packs_twelve_rows_into_the_tile():
    """The acceptance rule at the Qwen 24/2 geometry, graph level: the plan is PACKED 12:1 and the
    module the executor loaded packs the whole group into the 16-row tile (HEADS_PER_TILE = 12,
    Q_BOX_TOKENS = 1, 12 live rows + 4 zero tail rows), one CTA per (batch, KV head) unit instead of
    twelve; the output matches the reference and the split is the decode model's (8 units x 8 tiles)."""
    cap = {}
    plan = _run_graph(B=4, H=24, KH=2, s_q=1, lens=[700, 130, 5, 1024], page=16, dtype=torch.bfloat16, capture=cap)
    _assert_decode_tile_plan(plan, G=12, units=4 * 2, kv_tiles=8)
    mod = cap["module"]
    assert "sm107" in mod.__name__ and mod.__file__.endswith("/sm107/decode_d256_f16.py"), mod.__name__
    assert (mod.HEADS_PER_TILE, mod.Q_BOX_TOKENS, mod.Q_BOX_ROWS, mod.N_Q - mod.Q_BOX_ROWS, mod.N_Q, mod.CFG.SPLIT_KV) == (
        12,
        1,
        12,
        4,
        16,
        plan.knobs.split_kv or 1,
    )


@_gpu
@pytest.mark.parametrize(("page", "hnd"), [(16, True), (64, False)], ids=["page16_HND", "page64_NHD"])
def test_decode_graph_paged_equals_dense_bitwise(page, hnd):
    """The same tokens through a dense padded cache and through page pools (24/2, packed 12:1, both
    unsplit -- the padded dense graph cannot split, the paged one is pinned to its unsplit entry):
    the tile's PAGED_KV specialization only redirects the K/V tile loads through the block table
    (same tiles, same accumulation order, the padding mask zeroing the same columns), so O and LSE
    are BITWISE equal."""
    kw = dict(B=4, H=24, KH=2, s_q=1, lens=[700, 130, 5, 1024], dtype=torch.bfloat16, seed=3)
    dense, paged = {}, {}
    pd = _run_graph(**kw, page=0, capture=dense)
    pp = _run_graph(**kw, page=page, hnd=hnd, split_kv=1, capture=paged)
    _assert_decode_tile_plan(pd, G=12)
    _assert_decode_tile_plan(pp, G=12)
    assert (pd.knobs.split_kv or 1) == 1 == (pp.knobs.split_kv or 1), (pd.knobs, pp.knobs)
    assert torch.equal(dense["o"], paged["o"]), f"paged != dense: max |diff| {(dense['o'] - paged['o']).abs().max().item():.3e}"
    assert torch.equal(dense["lse"], paged["lse"]), f"paged != dense LSE: max |diff| {(dense['lse'] - paged['lse']).abs().max().item():.3e}"


@_gpu
def test_decode_graph_packed_equals_unpacked_bitwise():
    """24/2 over pages: the packed plan (12 live rows in one 16-row unit per (batch, KV head)) and the
    unpacked plan (one live row per (batch, Q head) unit, the former row contract) compute each Q
    row's softmax over the same KV tiles in the same order -- a column of S^T does not depend on
    which N column it occupies -- so O and LSE are BITWISE equal.  Both pinned unsplit."""
    kw = dict(B=3, H=24, KH=2, s_q=1, lens=[1000, 129, 640], page=32, dtype=torch.bfloat16, seed=5)
    packed, unpacked = {}, {}
    pp = _run_graph(**kw, pack_gqa=True, split_kv=1, capture=packed)
    pu = _run_graph(**kw, pack_gqa=False, split_kv=1, capture=unpacked)
    assert pp.knobs.pack_gqa is True and pu.knobs.pack_gqa is not True, (pp.knobs, pu.knobs)
    assert (packed["module"].HEADS_PER_TILE, unpacked["module"].HEADS_PER_TILE) == (12, 1)
    assert torch.equal(packed["o"], unpacked["o"]), f"packed != unpacked: max |diff| {(packed['o'] - unpacked['o']).abs().max().item():.3e}"
    assert torch.equal(packed["lse"], unpacked["lse"]), f"packed != unpacked LSE: max |diff| {(packed['lse'] - unpacked['lse']).abs().max().item():.3e}"


@_gpu
@pytest.mark.parametrize(
    ("form", "dtype", "page"),
    [("paged", torch.bfloat16, 16), ("paged", torch.float16, 16), ("paged", torch.bfloat16, 64), ("dense_unpadded", torch.bfloat16, 0)],
    ids=["paged_bf16", "paged_f16", "paged_page64_HND_bf16", "dense_unpadded_bf16"],
)
def test_decode_graph_split_equals_unsplit_within_the_combine_rounding(form, dtype, page):
    """The decode model's split (the leading plan: 8 ways at b=8 x 2 KV heads over pages -- page 16
    and page 64 HND -- 16 ways at b=3 x 2 KV heads over an UNPADDED dense cache -- the dense split the
    row claims) against the pinned unsplit plan, same graph: NOT bitwise, and bounded by the derived
    two-term budget of _assert_split_matches_unsplit (one output ulp of the element's binade from the
    single cast of two differently-associated fp32 sums, plus twice eps_P times the row's
    softmax-weighted mean |V| from the half-precision P each path quantizes at its own running max).
    The fp32 LSEs (m + log l, no quantized P in them) agree to fp32 rounding.  Both plans match the
    fp32 reference on their own; the measured magnitudes are printed."""
    if form == "paged":
        kw = dict(B=8, H=32, KH=2, s_q=1, lens=[4096, 4000, 129, 1, 2048, 4096, 300, 77], page=page, hnd=True, dtype=dtype, seed=11)
        G, units, kv_tiles = 16, 16, 32
    else:
        kw = dict(B=3, H=24, KH=2, s_q=1, lens=[4096] * 3, page=0, padded=False, dtype=dtype, seed=13)
        G, units, kv_tiles = 12, 6, 32
    split, unsplit = {}, {}
    ps = _run_graph(**kw, capture=split)
    _assert_decode_tile_plan(ps, G=G, units=units, kv_tiles=kv_tiles)
    assert ps.knobs.split_kv > 1, ps.knobs
    pu = _run_graph(**kw, split_kv=1, capture=unsplit)
    assert pu.knobs.pack_gqa is True and (pu.knobs.split_kv or 1) == 1, pu.knobs
    _assert_split_matches_unsplit(
        split["o"],
        unsplit["o"],
        dtype,
        q=unsplit["q"],
        k=unsplit["k"],
        v=unsplit["v"],
        lens=unsplit["lens"],
        scale=unsplit["scale"],
        what=f"split {ps.knobs.split_kv} vs unsplit O ({form}, {dtype})",
    )
    assert torch.equal(split["q"], unsplit["q"]) and torch.equal(split["k"], unsplit["k"]), "the two runs must see the same inputs"
    lse_s, lse_u = split["lse"], unsplit["lse"]
    live = torch.isfinite(lse_u)
    lse_diff = (lse_s[live] - lse_u[live]).abs().max().item()
    print(f"split {ps.knobs.split_kv} vs unsplit LSE ({form}): max |diff| = {lse_diff:.3e} on |LSE| up to {lse_u[live].abs().max().item():.2f}")
    torch.testing.assert_close(lse_s[live], lse_u[live], atol=1e-5, rtol=4e-6)


# --- the template level: the tile's packing and split partials, under the row's claims --------------


def _launch_template(mod, *, q, k_view, v_view, lens, scale, splits, dtype, bt=None, page=0, seq_q_lens_addr=0):
    """Drive the Rubin decode module directly through launch_f16 (dense or paged, split or not)
    and recombine the fp32 partials with the shared combine; returns (O, LSE)."""
    import cutlass
    import cuda.bindings.driver as cuda_driver

    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb
    from test_sdpa_fwd_split_kv_sm100 import _partial_kwargs, _partial_o_dtype, _partial_tag

    B, s_q, H, _ = q.shape
    KH = k_view.shape[2]
    dev = q.device
    fn = mod.compile(d_qk=D, d_v=D, has_lse=True, paged_hnd=(bt is not None))
    o_p = torch.zeros(splits * B, s_q, H, D, device=dev, dtype=_partial_o_dtype(splits, dtype))
    lse_p = torch.zeros(splits * B, H, s_q, device=dev, dtype=torch.float32)
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    launch_f16(
        fn,
        q,
        k_view,
        v_view,
        o_p,
        lse_p,
        torch.zeros(H, dtype=torch.float32, device=dev),
        torch.tensor(lens, dtype=torch.int32, device=dev),
        torch.zeros(1, dtype=torch.int64, device=dev),
        (B, H, KH, s_q, 0 if bt is not None else k_view.shape[1], 0),
        cutlass.Float32(scale * math.log2(math.e)),
        cutlass.Int32(0),
        seq_q_lens_addr,
        **_partial_kwargs(splits, o_p),
        block_table_tensor=bt,
        block_table_v_tensor=bt,
        page_size=page,
        stream=stream,
        host=mod._host,
    )
    if splits == 1:
        torch.cuda.synchronize()
        return o_p, lse_p
    o_out = torch.zeros(B, s_q, H, D, device=dev, dtype=dtype)
    lse_out = torch.zeros(B, H, s_q, device=dev, dtype=torch.float32)
    cfn = positional_entry(comb.compile_ptr(dtype_o="f16" if dtype == torch.float16 else "bf16", has_lse=True, dtype_partial=_partial_tag(splits, dtype)))
    cfn(o_p.data_ptr(), lse_p.data_ptr(), o_out.data_ptr(), lse_out.data_ptr(), (B, H, s_q, D), splits, o_out.stride(), lse_out.stride(), int(stream))
    torch.cuda.synchronize()
    return o_out, lse_out


@_gpu
@pytest.mark.parametrize("splits", [1, 2], ids=["unsplit", "split2"])
def test_decode_kernel_qwen_24_2_packs_twelve_rows(splits):
    """The 24/2 geometry on the tile's WHOLE-GROUP packing: HEADS_PER_TILE = 12, one token per
    16-row Q box, 12 live rows and 4 zero-filled tail rows per (batch, KV head) unit -- driven
    through the module loader (pack_gqa=True, qh_per_kh=12) over a paged cache, unsplit and
    split 2 with the fp32 partials recombined, against the fp32 reference -- the kernel-level
    evidence under the row's PackGQA claim (test_decode_graph_qwen_24_2_packs_twelve_rows_into_the_tile
    is its graph-level twin)."""
    B, H, KH, P, s_q = 3, 24, 2, 16, 1
    lens = [700, 130, 5]
    dtype = torch.bfloat16
    torch.manual_seed(7)
    S = max(lens)
    q = torch.randn(B, s_q, H, D, device="cuda", dtype=dtype)
    k_dense = torch.randn(B, S, KH, D, device="cuda", dtype=dtype)
    v_dense = torch.randn(B, S, KH, D, device="cuda", dtype=dtype)
    k_c, v_c, bt4 = _pools(k_dense, v_dense, P, True, seed=7)
    mod = _load(dtype_qkv=2, seq_kv_lens_present=True, paged_kv=True, page_size=P, split_kv=splits, pack_gqa=True, qh_per_kh=H // KH, decode_q_tile=16)
    assert (mod.HEADS_PER_TILE, mod.Q_BOX_TOKENS, mod.Q_BOX_ROWS, mod.N_Q - mod.Q_BOX_ROWS, mod.CFG.SPLIT_KV) == (12, 1, 12, 4, splits)
    o, lse = _launch_template(
        mod,
        q=q,
        k_view=k_c.permute(0, 2, 1, 3),
        v_view=v_c.permute(0, 2, 1, 3),
        lens=lens,
        scale=1.0 / math.sqrt(D),
        splits=splits,
        dtype=dtype,
        bt=bt4.view(B, -1),
        page=P,
    )
    ref_o, ref_lse = _ref(q, k_dense, v_dense, lens, None, 1.0 / math.sqrt(D))
    torch.testing.assert_close(o.float(), ref_o, atol=5e-2, rtol=0)
    torch.testing.assert_close(lse, ref_lse, atol=5e-3, rtol=0)


@_gpu
@pytest.mark.parametrize("splits", [1, 2], ids=["unsplit", "split2"])
def test_decode_kernel_two_column_groups(splits):
    """The 32-column tile (N_Q = 32: two softmax column groups, 8 softmax warps over the same 128
    TMEM lanes) compiles and computes correctly on Rubin but is not routed (the graph tests pin
    those shapes on the prefill tile) -- S_q = 2 x 16:1 packing = 32 rows over a paged cache,
    unsplit and split 2 with the partials recombined."""
    B, H, KH, P, s_q = 3, 32, 2, 16, 2
    lens = [700, 130, 5]
    dtype = torch.float16
    torch.manual_seed(5)
    S = max(lens)
    q = torch.randn(B, s_q, H, D, device="cuda", dtype=dtype)
    k_dense = torch.randn(B, S, KH, D, device="cuda", dtype=dtype)
    v_dense = torch.randn(B, S, KH, D, device="cuda", dtype=dtype)
    k_c, v_c, bt4 = _pools(k_dense, v_dense, P, True, seed=5)
    mod = _load(dtype_qkv=3, seq_kv_lens_present=True, paged_kv=True, page_size=P, split_kv=splits, pack_gqa=True, qh_per_kh=H // KH, decode_q_tile=32)
    assert (mod.CFG.N_Q, mod.CFG.SOFTMAX_WARPS, mod.COL_GROUPS, mod.DESC_VERSION) == (32, 8, 2, 0)
    o, lse = _launch_template(
        mod,
        q=q,
        k_view=k_c.permute(0, 2, 1, 3),
        v_view=v_c.permute(0, 2, 1, 3),
        lens=lens,
        scale=1.0 / math.sqrt(D),
        splits=splits,
        dtype=dtype,
        bt=bt4.view(B, -1),
        page=P,
    )
    ref_o, ref_lse = _ref(q, k_dense, v_dense, lens, None, 1.0 / math.sqrt(D))
    torch.testing.assert_close(o.float(), ref_o, atol=2e-2, rtol=0)
    torch.testing.assert_close(lse, ref_lse, atol=5e-3, rtol=0)


@_gpu
@pytest.mark.parametrize(("n_q", "s_q", "units"), [(32, 4, 2), (16, 4, 4), (32, 3, 2)], ids=["n32_sq4_2units", "n16_sq4_4units", "n32_sq3_half_unit"])
def test_decode_kernel_token_units_qwen_24_2(n_q, s_q, units):
    """The TOKEN-UNIT axis at the template level (the module loader + launch_f16): the 24/2 geometry
    at S_q = 3 / 4 over a paged cache, the 32-column tile in two units of two tokens (the second
    half-filled at S_q = 3) and the 16-column tile in four single-token units -- the host entry
    derives ceil(S_q / Q_BOX_TOKENS) units from the runtime S_q and every unit writes its own
    tokens' rows (the output buffer is zero-poisoned: a unit that did not run reads as a failure)
    -- against the fp32 reference, unsplit and split 2 with the fp32 partials recombined."""
    B, H, KH, P = 3, 24, 2, 16
    lens = [700, 130, 5]
    dtype = torch.bfloat16
    torch.manual_seed(13 + s_q)
    S = max(lens)
    q = torch.randn(B, s_q, H, D, device="cuda", dtype=dtype)
    k_dense = torch.randn(B, S, KH, D, device="cuda", dtype=dtype)
    v_dense = torch.randn(B, S, KH, D, device="cuda", dtype=dtype)
    k_c, v_c, bt4 = _pools(k_dense, v_dense, P, True, seed=13)
    for splits in (1, 2):
        mod = _load(dtype_qkv=2, seq_kv_lens_present=True, paged_kv=True, page_size=P, split_kv=splits, pack_gqa=True, qh_per_kh=H // KH, decode_q_tile=n_q)
        assert (mod.N_Q, mod.HEADS_PER_TILE, mod.Q_BOX_TOKENS, mod.Q_TOKEN_UNITS) == (n_q, 12, n_q // 12, True)
        assert -(-s_q // mod.Q_BOX_TOKENS) == units
        o, lse = _launch_template(
            mod,
            q=q,
            k_view=k_c.permute(0, 2, 1, 3),
            v_view=v_c.permute(0, 2, 1, 3),
            lens=lens,
            scale=1.0 / math.sqrt(D),
            splits=splits,
            dtype=dtype,
            bt=bt4.view(B, -1),
            page=P,
        )
        ref_o, ref_lse = _ref(q, k_dense, v_dense, lens, None, 1.0 / math.sqrt(D))
        torch.testing.assert_close(o.float(), ref_o, atol=5e-2, rtol=0)
        torch.testing.assert_close(lse, ref_lse, atol=5e-3, rtol=0)


@_gpu
def test_decode_kernel_dense_split_partials_recombine():
    """Dense padded K/V, split 4 at the routed width (N_Q = 16, unpacked 16:1 at S_q = 1): the
    fp32 partials over mixed lengths (an empty range next to live ones) recombine to the
    reference -- the kernel-level evidence under the row's dense split claim
    (test_decode_graph_split_equals_unsplit_within_the_combine_rounding is its graph-level twin)."""
    B, H, KH, s_q = 3, 8, 2, 1
    lens = [4096, 1, 129]
    dtype = torch.bfloat16
    torch.manual_seed(11)
    S = max(lens)
    q = torch.randn(B, s_q, H, D, device="cuda", dtype=dtype)
    k_dense = torch.randn(B, S, KH, D, device="cuda", dtype=dtype)
    v_dense = torch.randn(B, S, KH, D, device="cuda", dtype=dtype)
    mod = _load(dtype_qkv=2, seq_kv_lens_present=True, split_kv=4, pack_gqa=True, qh_per_kh=H // KH, decode_q_tile=16)
    assert (mod.HEADS_PER_TILE, mod.CFG.SPLIT_KV, mod.PAGED_KV) == (4, 4, False)
    o, lse = _launch_template(mod, q=q, k_view=k_dense, v_view=v_dense, lens=lens, scale=1.0 / math.sqrt(D), splits=4, dtype=dtype)
    ref_o, ref_lse = _ref(q, k_dense, v_dense, lens, None, 1.0 / math.sqrt(D))
    torch.testing.assert_close(o.float(), ref_o, atol=5e-2, rtol=0)
    torch.testing.assert_close(lse, ref_lse, atol=5e-3, rtol=0)


# --- routing boundary and the adapter ---------------------------------------------------------------


@_gpu
def test_decode_adapter_routes_decode_shaped_graphs_only():
    """The adapter's routing rule on cc 10.7 (the acceptance rule, config_sm107.decode_d256_q_tile):
    16 for S_q x pack_g <= 16 rows, 32 for the 32-column tile's forms (rows in (16, 32] in one
    unit; a packed group in up to two token units) and 0 past them.  Positives through
    check_support on paged pools (the form the Rubin adapter admits PackGQA on): the 24/2 geometry
    packed (12 rows), 16:1 packed, the 16-row MTP step, 16 MHA rows, the unpacked 32/2 step and the
    24/2 MTP steps at S_q = 2 / 4 (one / two token units of the 32-column tile); the rule's zeros
    through the same adapter's shape-explicit twin (_decode_q_tile_for, what check_support and
    _decode_q_tile both read), because a paged packed graph PAST the route is not a graph the Rubin
    row serves at all (its paged prefill pipeline is THD-only) -- asserted as the typed decline."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    def api(*, s_q, H, KH, pack):
        B, P, S = 2, 16, 256
        dtype = torch.bfloat16
        k_dense = torch.randn(B, S, KH, D, device="cuda", dtype=dtype)
        k_c, v_c, _ = _pools(k_dense, k_dense, P, False, seed=1)
        q_gpu = torch.randn(B, s_q, H, D, device="cuda", dtype=dtype).transpose(1, 2)
        o_gpu = torch.empty_like(q_gpu)
        lse = torch.empty(B, H, s_q, device="cuda", dtype=torch.float32)
        a = SdpaFwdDslSm100(
            sample_q=q_gpu,
            sample_k=k_c,
            sample_v=v_c,
            sample_o=o_gpu,
            sample_lse=lse,
            seq_kv_lens_present=True,
            paged_page_size=P,
            paged_max_seq_len_kv=S,
            pack_gqa=pack,
        )
        a.check_support()
        return a

    packed = api(s_q=1, H=24, KH=2, pack=True)
    assert packed._decode_q_tile() == 16  # 12 packed rows: the 24/2 geometry
    assert api(s_q=1, H=32, KH=2, pack=True)._decode_q_tile() == 16  # 16 packed rows
    assert api(s_q=2, H=16, KH=2, pack=True)._decode_q_tile() == 16  # the 16-row MTP step
    assert api(s_q=16, H=4, KH=4, pack=False)._decode_q_tile() == 16  # 16 MHA rows
    assert api(s_q=1, H=32, KH=2, pack=False)._decode_q_tile() == 16  # unpacked: one row per head
    assert api(s_q=2, H=24, KH=2, pack=True)._decode_q_tile() == 32  # the 24/2 MTP step at S_q = 2: 24 rows, one unit of the wide tile
    assert api(s_q=4, H=24, KH=2, pack=True)._decode_q_tile() == 32  # S_q = 4: 48 rows, two token units of two tokens
    assert api(s_q=2, H=32, KH=2, pack=True)._decode_q_tile() == 32  # 32/2 at S_q = 2: 32 rows, one unit (the former unpacked pin)
    # The rule through the packed adapter's shape-explicit twin: the wide tile's forms and the zeros past the route.
    assert packed._decode_q_tile_for(1, 32, 1) == 32  # 32 packed rows at S_q = 1: one unit
    assert packed._decode_q_tile_for(3, 16, 2) == 32  # 24 packed rows
    assert packed._decode_q_tile_for(4, 16, 2) == 32  # 64 packed rows: two units of two tokens x 16 heads
    assert packed._decode_q_tile_for(1, 24, 2) == 16 and packed._decode_q_tile_for(16, 1, 1) == 16
    assert packed._decode_q_tile_for(5, 24, 2) == 0  # 60 packed rows: a THIRD token unit, past the route
    assert (
        packed._decode_q_tile_for(9, 16, 2) == 0 and packed._decode_q_tile_for(3, 32, 1) == 0 and packed._decode_q_tile_for(1, 64, 1) == 0
    )  # 9 tokens x 8 heads: 3 units
    assert packed._decode_q_tile_for(5, 16, 2) == 32  # 5 tokens x 8 heads = 40 rows: two units of 4 tokens
    unpacked = api(s_q=1, H=32, KH=2, pack=False)
    assert unpacked._decode_q_tile_for(17, 4, 4) == 32 and unpacked._decode_q_tile_for(16, 4, 4) == 16  # 17 MHA rows: the wide tile; 16: the narrow one
    assert unpacked._decode_q_tile_for(32, 4, 4) == 32 and unpacked._decode_q_tile_for(33, 4, 4) == 0  # an unpacked step above 32 tokens: past the route
    # ... and a paged packed graph past the route is the row's typed decline (naming the decode tile), never the prefill tile.
    with pytest.raises(NotImplementedError, match="decode tile"):
        api(s_q=5, H=24, KH=2, pack=True)


@_gpu
def test_decode_adapter_dense_pack_and_split_follow_the_tile():
    """The standalone adapter on cc 10.7, DENSE cache (the form its former 'Rubin half PackGQA
    requires paged KV' gate declined): ACCEPT -- a decode-shaped graph packs the whole group (24/2
    -> 12 rows) and splits (the fp32 partials + the shared combine), the 16-row MTP step packs 8:1;
    the 24/2 MTP step packs on the 32-column tile in two token units and splits; REJECT -- a packed
    graph past the route (24/2 at S_q = 5: a third token unit) and a split past the route (33 MHA
    rows) are typed declines naming the decode tile, because the d256 prefill kernel wires neither
    (never a silent unpacked / unsplit run)."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    def api(*, s_q, H, KH, pack, split=1, seq_lens=True):
        B, S = 2, 256
        dtype = torch.bfloat16
        q_gpu = torch.randn(B, s_q, H, D, device="cuda", dtype=dtype).transpose(1, 2)
        k_c = torch.randn(B, S, KH, D, device="cuda", dtype=dtype).transpose(1, 2)
        v_c = torch.randn(B, S, KH, D, device="cuda", dtype=dtype).transpose(1, 2)
        a = SdpaFwdDslSm100(
            sample_q=q_gpu,
            sample_k=k_c,
            sample_v=v_c,
            sample_o=torch.empty_like(q_gpu),
            sample_lse=torch.empty(B, H, s_q, device="cuda", dtype=torch.float32),
            seq_kv_lens_present=seq_lens,
            pack_gqa=pack,
            split_kv=split,
        )
        a.check_support()
        return a

    assert api(s_q=1, H=24, KH=2, pack=True)._decode_q_tile() == 16  # 12 packed rows over a dense cache
    assert api(s_q=2, H=16, KH=2, pack=True)._decode_q_tile() == 16  # the 16-row MTP step
    a = api(s_q=1, H=24, KH=2, pack=True, split=2, seq_lens=False)  # the dense (unpadded) split, packed
    assert a._decode_q_tile() == 16 and a.split_kv == 2
    a = api(s_q=16, H=4, KH=4, pack=False, split=4, seq_lens=False)  # 16 MHA rows, unpacked split
    assert a._decode_q_tile() == 16 and a.split_kv == 4
    assert api(s_q=2, H=32, KH=2, pack=True)._decode_q_tile() == 32  # 32 packed rows: one unit of the 32-column tile
    a = api(s_q=4, H=24, KH=2, pack=True, split=2, seq_lens=False)  # the 24/2 MTP step: two token units, the dense (unpadded) split
    assert a._decode_q_tile() == 32 and a.split_kv == 2
    a = api(s_q=17, H=4, KH=4, pack=False, split=2, seq_lens=False)  # 17 MHA rows: one unit of the 32-column tile, split
    assert a._decode_q_tile() == 32 and a.split_kv == 2
    with pytest.raises(NotImplementedError, match="d256 decode tile"):
        api(s_q=5, H=24, KH=2, pack=True)  # 60 packed rows (a third token unit): no Rubin d256 kernel packs them
    with pytest.raises(NotImplementedError, match="d256 decode tile"):
        api(s_q=33, H=4, KH=4, pack=False, split=2, seq_lens=False)  # 33 MHA rows: the prefill kernel, no dense split
    with pytest.raises(ValueError, match="unpadded dense graphs only"):
        api(s_q=1, H=24, KH=2, pack=True, split=2, seq_lens=True)  # a PADDED dense split: the shared structural rule


def _paged_graph_offers_engine(*, B, H, KH, s_q, d, lens, page=16, causal_br=False):
    """Build a paged dense-Q graph and report whether the Rubin FROST engine lists ANY plan for it
    (a typed absence at plan creation, never a bare error)."""
    import cudnn

    torch.manual_seed(0)
    S = max(lens)
    k_dense = torch.randn(B, S, KH, d, device="cuda", dtype=torch.float16)
    k_c, v_c, bt = _pools(k_dense, k_dense, page, True, 0)
    g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q = g.tensor(dim=[B, H, s_q, d], stride=[s_q * H * d, d, H * d, 1], data_type=cudnn.data_type.HALF, name="q")
    k, v = g.tensor_like(k_c), g.tensor_like(v_c)
    tk, tv = g.tensor_like(bt), g.tensor_like(bt)
    sq_t = g.tensor(dim=[B, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32, name="sq")
    sk_t = g.tensor(dim=[B, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32, name="sk")
    kw = dict(use_causal_mask_bottom_right=True) if causal_br else {}
    o, _ = g.sdpa(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        generate_stats=False,
        attn_scale=1.0 / math.sqrt(d),
        use_padding_mask=True,
        seq_len_q=sq_t,
        seq_len_kv=sk_t,
        paged_attention_k_table=tk,
        paged_attention_v_table=tv,
        paged_attention_max_seq_len_kv=S,
        **kw,
    )
    o.set_output(True).set_dim([B, H, s_q, d]).set_stride([s_q * H * d, d, H * d, 1])
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    return offers_engine(g, _engine())


@_gpu
def test_decode_routing_boundary():
    """S_q x G past the Rubin route stays on the prefill d256 tile over a DENSE cache (the Rubin
    row's prefill tile serves dense Q; the route is read at the plan's geometry: a packed group in
    up to two token units of the 32-column tile, an unpacked step of up to 32 tokens in one), while
    over a PAGED cache a dense-Q graph past the route is not served on the Rubin row at all -- its
    paged prefill pipeline is THD-only -- and a decode-shaped paged d128 graph has no Rubin decode
    tile either: both are typed absences at plan creation.  The cells INSIDE the route (24 / 32
    unpacked tokens per head, 17 MHA rows: one unit of the 32-column tile) ride the decode tile.
    INVERTS when the Rubin row serves dense-Q paged prefill, gains a d128 decode tile or widens the
    route."""
    _run_graph(
        B=2, H=8, KH=2, s_q=24, lens=[700, 130], page=0, causal_br=True, expect=DECODE
    )  # dense, 24 tokens per head: one wide unit (G = 4 packed would need 3)
    _run_graph(B=2, H=4, KH=4, s_q=17, lens=[700, 130], page=0, expect=DECODE)  # dense, 17 MHA rows: one wide unit
    _run_graph(
        B=2, H=32, KH=2, s_q=32, lens=[700, 130], page=0, causal_br=True, expect=DECODE
    )  # dense, 32 tokens per head: one wide unit (16:1 packed would need 16)
    _run_graph(B=2, H=4, KH=4, s_q=33, lens=[700, 130], page=0, expect=PREFILL)  # dense, 33 MHA rows: past the route
    mtp5 = _run_graph(
        B=2, H=24, KH=2, s_q=5, lens=[700, 130], page=0, causal_br=True, expect=DECODE
    )  # dense, 24/2 at S_q = 5: packed would need a third unit, so it rides the tile UNPACKED (5 rows per Q head)
    assert mtp5.knobs.pack_gqa is not True, mtp5.knobs
    _run_graph(B=2, H=8, KH=2, s_q=65, lens=[700, 130], page=0, causal_br=True, expect=PREFILL)  # dense, 65 tokens per head: past the route either way
    assert _paged_graph_offers_engine(B=2, H=8, KH=2, s_q=1, d=D, lens=[700, 130]), "the decode-shaped paged d256 graph IS served (the decode tile)"
    assert _paged_graph_offers_engine(B=2, H=4, KH=4, s_q=17, d=D, lens=[700, 130]), "17 MHA rows over a paged cache: one unit of the 32-column tile"
    assert _paged_graph_offers_engine(B=2, H=24, KH=2, s_q=4, d=D, lens=[700, 130], causal_br=True), "the 24/2 MTP step over a paged cache: two token units"
    assert not _paged_graph_offers_engine(
        B=2, H=4, KH=4, s_q=33, d=D, lens=[700, 130]
    ), "33 MHA rows over a paged cache: past the route, THD-only paged prefill"
    assert _paged_graph_offers_engine(
        B=2, H=24, KH=2, s_q=5, d=D, lens=[700, 130], causal_br=True
    ), "24/2 at S_q = 5 over a paged cache: decode-shaped unpacked (5 rows per head)"
    assert not _paged_graph_offers_engine(B=2, H=8, KH=2, s_q=65, d=D, lens=[700, 130], causal_br=True), "65 tokens per head over a paged cache"
    assert not _paged_graph_offers_engine(B=2, H=8, KH=2, s_q=1, d=128, lens=[700, 130]), "paged d128 decode graph: no Rubin d128 decode tile"


@_gpu
def test_decode_routing_boundary_thd_queries():
    """Ragged (THD) queries -- packed [T, H, d] Q/O storage with ragged offsets, one token per
    sequence -- over a paged cache stay on the prefill d256 tile (the Rubin row's paged THD
    pipeline; the decode tile has no THD scheduler); the graph runs under the D2H detector and
    matches the reference."""
    import cudnn
    import cudnn.sdpa  # noqa: F401

    dev, dtype = "cuda", torch.float16
    B, H, KH, P = 3, 32, 2, 16
    lens = [700, 130, 5]
    S = max(lens)
    scale = 1.0 / math.sqrt(D)
    torch.manual_seed(3)
    k_dense = torch.randn(B, S, KH, D, device=dev, dtype=dtype)
    v_dense = torch.randn(B, S, KH, D, device=dev, dtype=dtype)
    k_c, v_c, bt = _pools(k_dense, v_dense, P, True, seed=3)
    q_pk = torch.randn(B, H, D, device=dev, dtype=dtype)  # T = B tokens, one per sequence
    stride = (H * D, D, H * D, 1)  # (B, H, S_max=1, d) over the packed [T, H, d] storage
    q_gpu = q_pk.view(-1).as_strided((B, H, 1, D), stride)
    o_stor = torch.zeros(B * H * D, device=dev, dtype=dtype)
    o_gpu = o_stor.as_strided((B, H, 1, D), stride)
    slq = torch.ones(B, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    slk = torch.tensor(lens, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    ro = (torch.arange(B + 1, dtype=torch.int64, device=dev) * H * D).view(B + 1, 1, 1, 1)

    g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tq = g.tensor(dim=[B, H, 1, D], stride=list(stride), data_type=cudnn.data_type.HALF, name="q")
    k, v = g.tensor_like(k_c), g.tensor_like(v_c)
    tk, tv = g.tensor_like(bt), g.tensor_like(bt)
    sq_t, sk_t = g.tensor_like(slq), g.tensor_like(slk)
    qro, oro = g.tensor_like(ro), g.tensor_like(ro)
    tq.set_ragged_offset(qro)
    o, _ = g.sdpa(
        name="sdpa",
        q=tq,
        k=k,
        v=v,
        generate_stats=False,
        attn_scale=scale,
        use_padding_mask=True,
        seq_len_q=sq_t,
        seq_len_kv=sk_t,
        paged_attention_k_table=tk,
        paged_attention_v_table=tv,
        paged_attention_max_seq_len_kv=S,
    )
    o.set_output(True).set_dim([B, H, 1, D]).set_stride(list(stride))
    o.set_ragged_offset(oro)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    select_engine(g, _engine())
    idx = g._plan_index
    g.check_support()
    g.build_plans()
    assert _served_by(g, idx) == PREFILL, f"plan {g.get_plan_name_at_index(idx)} served by {_served_by(g, idx)}, expected {PREFILL}"
    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    torch.cuda.set_sync_debug_mode("error")
    try:
        g.execute({tq: q_gpu, k: k_c, v: v_c, tk: bt, tv: bt, sq_t: slq, sk_t: slk, qro: ro, oro: ro, o: o_gpu}, ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    ref_o, _ = _ref(q_pk.view(B, 1, H, D), k_dense, v_dense, lens, None, scale)
    torch.testing.assert_close(o_stor.view(B, 1, H, D).float(), ref_o, atol=2e-2, rtol=0)


@_gpu
def test_decode_adapter_cuda_graph_replay_no_host_sync():
    """The adapter's execute path captured once at fixed B; seq_lens CONTENT changes between
    replays (Rule 3, the SM100 suite's protocol), in the SM100 suite's packed split_kv=4 form:
    the two launches (the tile's partials, the combine) replay without a host sync."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    B, H, KH, P, S = 8, 32, 2, 16, 1024
    dev, dtype = "cuda", torch.float16
    torch.manual_seed(2)
    k_dense = torch.randn(B, S, KH, D, device=dev, dtype=dtype)
    v_dense = torch.randn(B, S, KH, D, device=dev, dtype=dtype)
    k_c, v_c, bt4 = _pools(k_dense, v_dense, P, False, seed=2)
    bt = bt4.view(B, -1)
    q_gpu = torch.randn(B, 1, H, D, device=dev, dtype=dtype).transpose(1, 2)
    o_gpu = torch.empty(B, 1, H, D, device=dev, dtype=dtype).transpose(1, 2)
    lse = torch.empty(B, H, 1, device=dev, dtype=torch.float32)
    seq_lens = torch.full((B,), 1000, dtype=torch.int32, device=dev)
    seq_q = torch.ones(B, dtype=torch.int32, device=dev)
    api = SdpaFwdDslSm100(
        sample_q=q_gpu,
        sample_k=k_c,
        sample_v=v_c,
        sample_o=o_gpu,
        sample_lse=lse,
        seq_kv_lens_present=True,
        seq_q_lens_present=True,
        paged_page_size=P,
        paged_max_seq_len_kv=S,
        split_kv=4,
        pack_gqa=True,
    )
    api.check_support()
    api.compile()
    assert api.kernel_template == DECODE and "sm107" in type(api._k_mod).__module__ + api._k_mod.__name__
    assert api.split_kv == 4 and api._k_mod.CFG.SPLIT_KV == 4 and api._k_mod.HEADS_PER_TILE == 16
    ws = torch.empty(max(api.scratch_workspace_bytes(), 1), device=dev, dtype=torch.uint8)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        api.execute(q_gpu, k_c, v_c, o_gpu, lse_tensor=lse, seq_kv_lens=seq_lens, seq_q_lens=seq_q, block_table=bt, workspace=ws)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        prev = torch.cuda.get_sync_debug_mode()
        with torch.cuda.graph(g, stream=s):
            torch.cuda.set_sync_debug_mode("error")
            try:
                api.execute(q_gpu, k_c, v_c, o_gpu, lse_tensor=lse, seq_kv_lens=seq_lens, seq_q_lens=seq_q, block_table=bt, workspace=ws)
            finally:
                torch.cuda.set_sync_debug_mode(prev)
        scale = 1.0 / math.sqrt(D)
        for new_lens in ([5, 1024, 77, 128, 129, 1, 512, 1000], [1024] * B, [0, 1, 2, 3, 4, 5, 6, 7]):
            seq_lens.copy_(torch.tensor(new_lens, dtype=torch.int32))
            g.replay()
            torch.cuda.synchronize()
            ref_o, ref_lse = _ref(q_gpu.transpose(1, 2), k_dense, v_dense, new_lens, None, scale)
            live = ~torch.isinf(ref_lse)
            torch.testing.assert_close(o_gpu.transpose(1, 2).float(), ref_o, atol=2e-2, rtol=0)
            torch.testing.assert_close(lse.view(B, H, 1)[live], ref_lse[live], atol=5e-3, rtol=0)
    finally:
        g.reset()


# --- the gate in the combine: a GATED graph rides the tile through its split -------------------------
#
# The decode tile has no epilogue-gate seams (config_sm107.make_cfg_d256_decode refuses the record),
# so a graph with the fused-gate tail ``sdpa(virtual O_v) -> sigmoid(G) -> mul(O_v, s)`` rode the d256
# PREFILL kernel's fused epilogue -- unsplit, unpacked, dense only.  Now its SPLIT rides the tile: the
# tile writes its fp32 partials as any split does and the shared combine applies ``O *= sigmoid(G)``
# to the merged fp32 value before the single cast (``sm100/split_combine`` ``gate=True``, the fused
# kernels' own ``h * tanh(g / 2) + h`` arithmetic -- FROST's one-rounding convention; vLLM's split
# merge and the gated attention block's torch reference round the merged O to the IO dtype BEFORE the
# fp32 gate).  At split_kv == 1 the gated graph keeps the prefill kernel (``api_dsl._gate_in_combine``,
# ``engines.d256_decode_tile_selected(..., split_kv)``, the heuristics' floor of 2).


@_gpu
def test_decode_graph_gated_qwen_24_2_splits_with_the_gate_in_the_combine():
    """The acceptance rule, gated: the Qwen 24/2 geometry WITH the fused-gate tail over pages leads
    on the DECODE tile, packed 12:1, split by the decode model floored at 2 (8 units x 32 tiles: 16
    ways), the module compiled UNGATED (TemplateParams.epilogue_gate False: the tile has no gate
    seams) and the gate applied by the split combine; O matches the gated fp32 reference, LSE the
    plain one, and a keyless batch's rows are EXACTLY 0 under a random gate."""
    cap = {}
    plan = _run_graph(B=4, H=24, KH=2, s_q=1, lens=[4096, 130, 0, 1024], page=16, dtype=torch.bfloat16, gate="random", capture=cap)
    _assert_decode_tile_plan(plan, G=12, units=8, kv_tiles=32, gated=True)
    assert plan.knobs.split_kv >= 2, plan.knobs
    mod = cap["module"]
    assert mod.__file__.endswith("/sm107/decode_d256_f16.py"), mod.__name__
    assert (mod.HEADS_PER_TILE, mod.CFG.SPLIT_KV, mod.PARAMS.epilogue_gate) == (12, plan.knobs.split_kv, False), (
        mod.HEADS_PER_TILE,
        mod.CFG.SPLIT_KV,
        mod.PARAMS,
    )
    assert (cap["o"][2] == 0).all(), "the keyless batch must be EXACTLY zero (a SELECT after the gate fma, never residue times a gate)"
    assert torch.isneginf(cap["lse"][2]).all()


@_gpu
@pytest.mark.parametrize(
    ("dtype", "page", "hnd"),
    [(torch.bfloat16, 16, True), (torch.float16, 64, False)],
    ids=["paged16_HND_bf16", "paged64_NHD_f16"],
)
def test_decode_graph_gated_paged_split_matches_the_reference(dtype, page, hnd):
    """The gated split over PAGES (the serving form; the paged prefill kernel has no gate, so this is
    the one gated paged plan the row has): b=8 x 2 KV heads over mixed lengths incl. 0 and 1, page 16
    HND and page 64 NHD, packed 16:1, the model's split; O matches the gated reference, the keyless
    batch is exactly 0 and the 1-key batch gates one key's row."""
    cap = {}
    plan = _run_graph(B=8, H=32, KH=2, s_q=1, lens=[4096, 4000, 129, 1, 2048, 0, 300, 77], page=page, hnd=hnd, dtype=dtype, gate="random", capture=cap)
    _assert_decode_tile_plan(plan, G=16, units=16, kv_tiles=32, gated=True)
    assert plan.knobs.split_kv >= 2 and cap["module"].PARAMS.epilogue_gate is False, (plan.knobs, cap["module"].PARAMS)
    assert (cap["o"][5] == 0).all() and torch.isneginf(cap["lse"][5]).all(), "the keyless batch: exactly 0 / -inf under a random gate"


@_gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
def test_decode_graph_gated_split_matches_the_fused_unsplit_gate_within_the_budget(dtype):
    """THE acceptance: split + gate-in-combine (the decode tile) against the UNSPLIT fused gate (the
    d256 prefill kernel's epilogue, the same graph pinned split_kv=1 -- a different kernel; a dense
    UNPADDED cache, the one form both plans serve: the paged prefill kernel has no gate), same inputs,
    same G.  Both apply the same sigmoid(G) arithmetic (fmul2 / tanh.approx / ffma2) to their fp32
    value before the single cast, so the gated difference is the attention's own: the derived
    two-term budget of _assert_split_matches_unsplit with term (2) scaled by sigmoid(g) -- one output
    ulp of the element's binade from the single rounding of two differently-associated sums, plus
    sigmoid(g) x 2 eps_P x the row's softmax-weighted mean |V| from the half-precision P each kernel
    quantizes at its own running max.  The exact max |dO| is printed (the record).  LSE is gate-free
    on both kernels and agrees to a few fp32 ulps (two summation orders of one fp32 log-sum-exp)."""
    kw = dict(B=3, H=24, KH=2, s_q=1, lens=[4096] * 3, page=0, padded=False, dtype=dtype, seed=29)
    split, fused = {}, {}
    ps = _run_graph(**kw, gate="random", capture=split)
    _assert_decode_tile_plan(ps, G=12, units=6, kv_tiles=32, gated=True)
    assert ps.knobs.split_kv >= 2 and split["module"].PARAMS.epilogue_gate is False, (ps.knobs, split["module"].PARAMS)
    pu = _run_graph(**kw, gate="random", split_kv=1, expect=PREFILL, capture=fused)
    assert pu.knobs.pack_gqa is not True and (pu.knobs.split_kv or 1) == 1 and fused["module"].PARAMS.epilogue_gate is True, (pu.knobs, fused["module"].PARAMS)
    assert (
        torch.equal(split["q"], fused["q"]) and torch.equal(split["k"], fused["k"]) and torch.equal(split["gate"], fused["gate"])
    ), "the two plans must see the same inputs"
    _assert_split_matches_unsplit(
        split["o"],
        fused["o"],
        dtype,
        q=fused["q"],
        k=fused["k"],
        v=fused["v"],
        lens=fused["lens"],
        scale=fused["scale"],
        gate=fused["gate"],
        what=f"gate-in-combine split {ps.knobs.split_kv} (decode tile) vs the fused unsplit gate (prefill kernel) O (dense unpadded, {dtype})",
    )
    lse_s, lse_u = split["lse"], fused["lse"]
    live = torch.isfinite(lse_u)
    lse_diff = (lse_s[live] - lse_u[live]).abs().max().item()
    print(
        f"gate-in-combine split {ps.knobs.split_kv} vs fused unsplit LSE (dense unpadded): max |diff| = {lse_diff:.3e} on |LSE| up to {lse_u[live].abs().max().item():.2f}"
    )
    torch.testing.assert_close(lse_s[live], lse_u[live], atol=5e-5, rtol=1e-5)


@_gpu
def test_decode_graph_gate_in_combine_is_bitwise_the_ungated_plan_at_exact_sigmoids():
    """Bitwise against the unfused gate-after-combine reference WHERE THE ARITHMETIC IS IDENTICAL: the
    same split plan without the gate tail loads the same ungated module and so computes the same fp32
    merged value; sigmoid(+1e4) == 1, sigmoid(0) == 1/2 and sigmoid(-1e4) == 0 are exact in the
    combine's tanh form (tanh.approx saturates to +-1 and tanh(0) == 0; ``h * t + h`` with h the
    half-scaled value) and in the reference, and a power-of-two scaling commutes with the rounding,
    so the gated O is BITWISE the ungated O, the ungated O times one half, and exactly 0.  LSE is
    bitwise the ungated LSE in all three: the gate never touches the partials."""
    kw = dict(B=8, H=32, KH=2, s_q=1, lens=[4096, 4000, 129, 1, 2048, 4096, 300, 77], page=16, dtype=torch.bfloat16, seed=17)
    plain = {}
    pp = _run_graph(**kw, capture=plain)
    _assert_decode_tile_plan(pp, G=16, units=16, kv_tiles=32)
    assert pp.knobs.split_kv > 1, pp.knobs
    for logit, want in ((1e4, plain["o"]), (0.0, plain["o"] * 0.5), (-1e4, torch.zeros_like(plain["o"]))):
        cap = {}
        pg = _run_graph(**kw, gate=logit, split_kv=pp.knobs.split_kv, capture=cap)
        assert pg.knobs.split_kv == pp.knobs.split_kv and pg.knobs.pack_gqa is True, (pg.knobs, pp.knobs)
        assert cap["module"].__file__ == plain["module"].__file__ and cap["module"].PARAMS == plain["module"].PARAMS, "one ungated module serves both plans"
        assert torch.equal(cap["o"], want), f"gate logit {logit}: max |diff| {(cap['o'] - want).abs().max().item():.3e}"
        assert torch.equal(cap["lse"], plain["lse"]), f"gate logit {logit}: the LSE must not depend on the gate"


@_gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
def test_decode_graph_gate_in_combine_vs_the_unfused_gate_after_combine_reference(dtype):
    """Where the arithmetic is NOT identical the budget is stated: the unfused references (vLLM's
    split merge, the gated attention block's torch reference) round the merged O to the IO dtype
    BEFORE the fp32 gate and round again after it; this pass rounds ONCE.  Per element, with b the
    ungated plan's O (the same merged value, rounded) and r = round(b * sigmoid(g)):

        |a - r| <= ulp(max(|a|, |r|)) + sigmoid(g) * ulp(|b|) / 2 + |b| * 2^-11

    -- the two final roundings, the intermediate rounding the reference carries (at most half an ulp
    of the merged value, scaled by the gate; ulp(O32) <= ulp(b)), and the approximate tanh behind the
    combine's sigmoid (tanh.approx.f32: max relative error 2^-11 per the PTX ISA, i.e. 2^-12 on the
    sigmoid, doubled here as margin); ulp() is the dtype's, floored at its subnormal step (an f16 O
    element near zero IS subnormal).  Rigorous given that tanh bound; the measured ratio is printed."""
    kw = dict(B=8, H=32, KH=2, s_q=1, lens=[4096, 4000, 129, 1, 2048, 4096, 300, 77], page=16, dtype=dtype, seed=19)
    plain, gated = {}, {}
    pp = _run_graph(**kw, capture=plain)
    pg = _run_graph(**kw, gate="random", split_kv=pp.knobs.split_kv, capture=gated)
    assert pg.knobs.split_kv == pp.knobs.split_kv > 1, (pg.knobs, pp.knobs)
    mbits = {torch.bfloat16: 7, torch.float16: 10}[dtype]
    b = plain["o"].double()
    s = torch.sigmoid(gated["gate"].double())
    r = (plain["o"] * torch.sigmoid(gated["gate"])).to(dtype).double()  # the unfused convention: the IO-dtype O, the fp32 gate, the IO dtype again
    a = gated["o"].double()
    live = torch.isfinite(plain["lse"]).transpose(1, 2).unsqueeze(-1).expand_as(a)  # [B, S_q, H, d]
    budget = _ulp(torch.maximum(a.abs(), r.abs()), mbits) + s * _ulp(b.abs(), mbits) * 0.5 + b.abs() * 2.0**-11
    diff = (a - r).abs()
    ratios = torch.where(live, diff / budget, torch.zeros_like(diff))
    ratio = ratios.max().item()
    worst = ratios.argmax()
    big = live & (b.abs() >= 0.5)  # where the output rounding (<= ulp(a) / 2, <= 2^-(mbits + 2) of |b|) cannot hide the sigmoid's own deviation
    implied = ((a / b) - s).abs()[big].max().item() if big.any() else float("nan")
    print(
        f"gate-in-combine vs the unfused gate-after-combine reference ({dtype}): max |diff| = {diff[live].max().item():.3e}, "
        f"{(diff[live] > 0).double().mean().item():.4%} of the elements differ, max |diff| / budget = {ratio:.3f}; "
        f"worst element: a={a.flatten()[worst].item():.6e} r={r.flatten()[worst].item():.6e} b={b.flatten()[worst].item():.6e} "
        f"g={gated['gate'].double().flatten()[worst].item():.4f} s={s.flatten()[worst].item():.6f} budget={budget.flatten()[worst].item():.3e}; "
        f"max |a/b - sigmoid(g)| over |b| >= 0.5: {implied:.3e}"
    )
    assert ratio <= 1.0, ratio
    assert (a[~live] == 0).all(), "dead rows: exactly 0 on both"


@_gpu
def test_decode_graph_gated_packed_equals_unpacked_bitwise():
    """24/2 gated over pages: the packed gated split (12 live rows per unit) and the unpacked gated
    split (one live row per unit) at the SAME split compute each row's softmax over the same tiles in
    the same order and gate the same merged value, so O and LSE are BITWISE equal (the ungated pin,
    gated)."""
    kw = dict(B=3, H=24, KH=2, s_q=1, lens=[1000, 129, 640], page=32, dtype=torch.bfloat16, seed=5)
    packed, unpacked = {}, {}
    pp = _run_graph(**kw, gate="random", pack_gqa=True, capture=packed)
    _assert_decode_tile_plan(pp, G=12, units=6, kv_tiles=8, gated=True)
    pu = _run_graph(**kw, gate="random", pack_gqa=False, split_kv=pp.knobs.split_kv, capture=unpacked)
    assert pp.knobs.pack_gqa is True and pu.knobs.pack_gqa is not True and pu.knobs.split_kv == pp.knobs.split_kv >= 2, (pp.knobs, pu.knobs)
    assert (packed["module"].HEADS_PER_TILE, unpacked["module"].HEADS_PER_TILE) == (12, 1)
    assert torch.equal(packed["o"], unpacked["o"]), f"packed != unpacked: max |diff| {(packed['o'] - unpacked['o']).abs().max().item():.3e}"
    assert torch.equal(packed["lse"], unpacked["lse"])


@_gpu
def test_decode_adapter_gate_rides_the_combine_on_the_tile_only():
    """The standalone adapter's twin of the row's gate-in-combine claim (rule 8b): ACCEPT -- a gated
    decode-shaped d256 graph with split_kv >= 2 lowers onto the decode tile (dense unpadded or paged,
    packed or not), its template record UNGATED and the gate marked for the combine; at split_kv == 1
    the same dense graph keeps the d256 prefill kernel's fused epilogue (record gated, the tile not
    selected).  The Rubin route's wide forms ride the tile too: 17 MHA rows and 32 packed rows (32/2 at
    S_q = 2) split with the gate in the combine.  REJECT (typed, naming the combine) -- the gate at
    split 1 over pages, PackGQA with the gate unsplit, a gated split past the route (33 MHA rows, a
    packed group needing a third token unit)."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    def api(*, s_q, H, KH, pack, split=1, page=0, seq_lens=True):
        B, S = 2, 256
        dtype = torch.bfloat16
        q_gpu = torch.randn(B, s_q, H, D, device="cuda", dtype=dtype).transpose(1, 2)
        if page:
            k_dense = torch.randn(B, S, KH, D, device="cuda", dtype=dtype)
            k_c, v_c, _ = _pools(k_dense, k_dense, page, True, seed=1)
            paged_kw = dict(paged_page_size=page, paged_max_seq_len_kv=S)
        else:
            k_c = torch.randn(B, S, KH, D, device="cuda", dtype=dtype).transpose(1, 2)
            v_c = torch.randn(B, S, KH, D, device="cuda", dtype=dtype).transpose(1, 2)
            paged_kw = {}
        a = SdpaFwdDslSm100(
            sample_q=q_gpu,
            sample_k=k_c,
            sample_v=v_c,
            sample_o=torch.empty_like(q_gpu),
            sample_lse=torch.empty(B, H, s_q, device="cuda", dtype=torch.float32),
            seq_kv_lens_present=seq_lens,
            pack_gqa=pack,
            split_kv=split,
            sample_gate=torch.empty_like(q_gpu),
            **paged_kw,
        )
        a.check_support()
        return a

    a = api(s_q=1, H=24, KH=2, pack=True, split=2, seq_lens=False)  # dense unpadded, packed, split: the tile, gate in the combine
    assert a._decode_q_tile() == 16 and a._gate_in_combine() and a.template_params().epilogue_gate is False
    a = api(s_q=1, H=24, KH=2, pack=True, split=4, page=16)  # paged, packed, split: the tile
    assert a._decode_q_tile() == 16 and a._gate_in_combine() and a.template_params().epilogue_gate is False
    a = api(s_q=16, H=4, KH=4, pack=False, split=4, page=16)  # 16 MHA rows unpacked, paged, split: the tile
    assert a._decode_q_tile() == 16 and a._gate_in_combine() and a.template_params().epilogue_gate is False
    a = api(s_q=1, H=24, KH=2, pack=False, split=1, seq_lens=False)  # dense, unsplit: the prefill kernel's fused epilogue
    assert a._decode_q_tile() == 0 and not a._gate_in_combine() and a.template_params().epilogue_gate is True
    with pytest.raises(NotImplementedError, match="paged"):
        api(s_q=1, H=24, KH=2, pack=False, split=1, page=16)  # the paged prefill kernel has no gate; unsplit, the tile does not serve it
    with pytest.raises(NotImplementedError, match="PackGQA"):
        api(s_q=1, H=24, KH=2, pack=True, split=1, seq_lens=False)  # a packed unsplit gated plan exists on no Rubin kernel
    a = api(s_q=17, H=4, KH=4, pack=False, split=2, seq_lens=False)  # 17 MHA rows: one unit of the 32-column tile, split, gate in the combine
    assert a._decode_q_tile() == 32 and a._gate_in_combine() and a.template_params().epilogue_gate is False
    a = api(s_q=2, H=32, KH=2, pack=True, split=2, page=16)  # 32 packed rows: the 32-column tile, split over pages, gate in the combine
    assert a._decode_q_tile() == 32 and a._gate_in_combine() and a.template_params().epilogue_gate is False
    with pytest.raises(NotImplementedError, match="split_kv"):
        api(s_q=33, H=4, KH=4, pack=False, split=2, seq_lens=False)  # 33 MHA rows: past the route, the prefill kernel has no dense split
    with pytest.raises(NotImplementedError, match="decode tile|PackGQA"):
        api(s_q=5, H=24, KH=2, pack=True, split=2, page=16)  # 24/2 at S_q = 5 packed: a third token unit, past the route; no Rubin d256 kernel packs it


@_gpu
def test_decode_adapter_gated_split_rejects_a_gate_off_the_plans_device():
    """The gate-in-combine split binds G outside the native dense binder, so the binding keeps the plan's
    CUDA-device check itself (prepared.CombineGate): a same-shape CPU gate is the SAME typed error on the
    tile's split (the combine's binding) as on the unsplit fused-gate prefill kernel (the native binder) --
    never an address forwarded to a launch."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    B, H, KH, S = 2, 24, 2, 256
    dtype = torch.bfloat16
    q = torch.randn(B, 1, H, D, device="cuda", dtype=dtype).transpose(1, 2)
    k = torch.randn(B, S, KH, D, device="cuda", dtype=dtype).transpose(1, 2)
    v = torch.randn(B, S, KH, D, device="cuda", dtype=dtype).transpose(1, 2)
    o = torch.empty_like(q)
    lse = torch.empty(B, H, 1, device="cuda", dtype=torch.float32)
    gate_cpu = torch.randn(B, 1, H, D, dtype=dtype).transpose(1, 2)
    for split, pack in ((2, True), (1, False)):  # the tile's split (gate in the combine) and the prefill kernel's fused gate
        api = SdpaFwdDslSm100(
            sample_q=q,
            sample_k=k,
            sample_v=v,
            sample_o=o,
            sample_lse=lse,
            seq_kv_lens_present=False,
            pack_gqa=pack,
            split_kv=split,
            sample_gate=torch.empty_like(q),
        )
        api.check_support()
        api.compile()
        assert api._gate_in_combine() == (split > 1)
        kw = dict(lse_tensor=lse, gate=gate_cpu)
        if api.scratch_workspace_bytes():
            kw["workspace"] = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
        with pytest.raises(ValueError, match="gate must be on this plan's CUDA device"):
            api.execute(q, k, v, o, **kw)


@_gpu
def test_decode_adapter_gated_split_cuda_graph_replay_no_host_sync():
    """The standalone adapter with ``sample_gate`` on the decode tile's split (packed 16:1, split 4
    over pages): the two launches -- the ungated tile's partials, the GATED combine -- capture once
    under the D2H detector (Rule 3) and replay with changed seq_lens AND changed gate CONTENT (G's
    address is bound at capture, its values read at replay), matching the gated reference each time."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    B, H, KH, P, S = 8, 32, 2, 16, 1024
    dev, dtype = "cuda", torch.float16
    torch.manual_seed(2)
    k_dense = torch.randn(B, S, KH, D, device=dev, dtype=dtype)
    v_dense = torch.randn(B, S, KH, D, device=dev, dtype=dtype)
    k_c, v_c, bt4 = _pools(k_dense, v_dense, P, False, seed=2)
    bt = bt4.view(B, -1)
    q_gpu = torch.randn(B, 1, H, D, device=dev, dtype=dtype).transpose(1, 2)
    gate_gpu = (torch.randn(B, 1, H, D, device=dev) * 2.0).to(dtype).transpose(1, 2)
    o_gpu = torch.empty(B, 1, H, D, device=dev, dtype=dtype).transpose(1, 2)
    lse = torch.empty(B, H, 1, device=dev, dtype=torch.float32)
    seq_lens = torch.full((B,), 1000, dtype=torch.int32, device=dev)
    seq_q = torch.ones(B, dtype=torch.int32, device=dev)
    api = SdpaFwdDslSm100(
        sample_q=q_gpu,
        sample_k=k_c,
        sample_v=v_c,
        sample_o=o_gpu,
        sample_lse=lse,
        seq_kv_lens_present=True,
        seq_q_lens_present=True,
        paged_page_size=P,
        paged_max_seq_len_kv=S,
        split_kv=4,
        pack_gqa=True,
        sample_gate=gate_gpu,
    )
    api.check_support()
    api.compile()
    assert api.kernel_template == DECODE and api._gate_in_combine() and api._k_mod.PARAMS.epilogue_gate is False
    assert api.split_kv == 4 and api._k_mod.CFG.SPLIT_KV == 4 and api._k_mod.HEADS_PER_TILE == 16
    assert api._dense_spec.gate_expect is None and api._dense_spec.combine.gate is not None, "the gate is the combine's, not the binder's"
    ws = torch.empty(max(api.scratch_workspace_bytes(), 1), device=dev, dtype=torch.uint8)
    kw = dict(lse_tensor=lse, seq_kv_lens=seq_lens, seq_q_lens=seq_q, block_table=bt, workspace=ws, gate=gate_gpu)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        api.execute(q_gpu, k_c, v_c, o_gpu, **kw)
    torch.cuda.synchronize()
    first = o_gpu.clone()
    with torch.cuda.stream(s):
        api.execute(q_gpu, k_c, v_c, o_gpu, **kw)
    torch.cuda.synchronize()
    assert torch.equal(o_gpu, first), "two-launch delta on the gated O"
    g = torch.cuda.CUDAGraph()
    try:
        prev = torch.cuda.get_sync_debug_mode()
        with torch.cuda.graph(g, stream=s):
            torch.cuda.set_sync_debug_mode("error")
            try:
                api.execute(q_gpu, k_c, v_c, o_gpu, **kw)
            finally:
                torch.cuda.set_sync_debug_mode(prev)
        scale = 1.0 / math.sqrt(D)
        gen = torch.Generator(device=dev).manual_seed(33)
        for new_lens in ([5, 1024, 77, 128, 129, 1, 512, 1000], [1024] * B, [0, 1, 2, 3, 4, 5, 6, 7]):
            seq_lens.copy_(torch.tensor(new_lens, dtype=torch.int32))
            gate_gpu.copy_((torch.randn(B, 1, H, D, device=dev, generator=gen) * 2.0).to(dtype).transpose(1, 2))
            g.replay()
            torch.cuda.synchronize()
            ref_o, ref_lse = _ref(q_gpu.transpose(1, 2), k_dense, v_dense, new_lens, None, scale)
            ref_o = ref_o * torch.sigmoid(gate_gpu.transpose(1, 2).float())
            live = ~torch.isinf(ref_lse)  # [B, H, 1]
            torch.testing.assert_close(o_gpu.transpose(1, 2).float(), ref_o, atol=2e-2, rtol=0)
            dead = (~live).view(B, 1, H)  # the [B, S_q=1, H] rows of the BSHD output
            if dead.any():
                assert (o_gpu.transpose(1, 2)[dead].float() == 0).all(), "a keyless row must be EXACTLY 0 under the gate, every replay"
            torch.testing.assert_close(lse.view(B, H, 1)[live], ref_lse[live], atol=5e-3, rtol=0)
    finally:
        g.reset()
