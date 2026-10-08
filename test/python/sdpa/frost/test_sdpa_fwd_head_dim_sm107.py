# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The small head-dim tiles of the Rubin (SM107) f16/bf16 forward row: ``TILE_K = TILE_O`` in {32, 64} as Cfg flavors of the d128 file.

``sm107/prefill_d128_f16.py`` is one body with three head-dim geometries (``TemplateParams.d_flavor`` 32 / 64 / 128 ->
``config_sm107.make_cfg_d32`` / ``make_cfg_d64`` / ``make_cfg_d128``).  A graph's head dim lands on the SMALLEST width covering
max(d_qk, d_v) -- 8 / 16 / 24 -> 32, 40 / 48 / 56 -> 64, the natives on themselves -- and the TMA load zero-fills the columns
past D (exact zero terms in QK^T and in P.V's padded V columns; the ``expect_tx`` count is the FULL box; the O store clips at
d_v).  D % 8 != 0 is a typed decline (the TMA 16-byte global-stride rule at 2 B/elem).  Everything coupled to the tile -- the
swizzle atoms, the TMA iteration counts, the TMEM O slot width, the BMM1 k-step count -- is derived from ``CFG.*`` and pinned by
``_validate_head_dim_tile``; the d128 rendering is byte-identical to the one-geometry file (``test_sdpa_fwd_score_stats.py``
holds the cubin md5 records).

The small tiles carry the row's DENSE leg only (``Capabilities.dense_only_d_shapes``): a graph that asks for split-KV, the score
side outputs, THD, paged KV, PackGQA or a cga outside the tile's domain is selected PAST them onto the flavor that carries the leg
-- the d128 flavor it rode before the tiles existed (a cga2 request at D <= 32: the 64-wide tile) -- so nothing the row served
before the tiles is withdrawn, and THD / paged KV / PackGQA at D <= 64 keep exactly their pre-tile typed declines.

Three tiers: (1) CPU contract -- the row's accept / decline table per D, the ROUTE table per (D, leg) with the three pick
implementations agreeing on every cell (the row's envelope walk, the adapter's flavor pool, ``sm107_f16_tile_width(leg=, cga=)``),
the one constant feeding the row's cga domain, the adapter's cga twin and the config factory, the factories' derived facts and
raising couplings, the loader binding ``d_flavor`` to the flavor, the static expect_tx pin; (2) host rendering pins -- an sm_107a
trace-compile of each new rendering keeps the register split (``USETMAXREG`` > 0) and spills no more than the d128 rendering of
the same mask arm; (3) GPU cells on a Rubin device -- the fp64 oracle over the degenerate-input matrix at every served D (8 .. 64
on the small tiles, 128 on the d128 tile of the same file, both dtypes), the bitwise pin D = 24 vs an explicitly zero-padded D =
32 (48 vs 64), the d64 tile at cga2, the routed legs (split-KV and the score outputs at D = 24 / 64 on the d128 flavor, cga2 at
D = 24 / 32 on the 64-wide tile -- bitwise the zero-padded D = 64 cga2 run), optional and base-2 Stats.  A D = 24 cell that
completes under the runner's timeout IS the expect_tx pin: a byte count short of the full box would hang the consumer on the
first tile.
"""

import dataclasses
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from frost_test_utils import _SM, assert_no_new_spills, make_dense_stats, requires_dsl, requires_rubin, run_sass_probe, sass_probe_source

pytestmark = [pytest.mark.L0, requires_dsl]

_ROW = "sdpa_fwd_prefill_sm107"
# D -> the tile width the row serves it on (config_sm107.sm107_f16_tile_width); every member is a GPU oracle cell below.
_SERVED = {8: 32, 16: 32, 24: 32, 32: 32, 40: 64, 48: 64, 56: 64, 64: 64}
_DECLINED = (4, 12, 20, 36, 60)  # D % 8 != 0: the TMA 16-byte global-stride rule, a typed decline
_KERNEL = Path(__file__).resolve().parents[4] / "python" / "cudnn" / "sdpa" / "fwd" / "kernels" / "sm107" / "prefill_d128_f16.py"


@pytest.fixture(autouse=True)
def _sm107_target_for_metadata(monkeypatch):
    """Model SM107 eligibility on other GPUs without requiring their DSL to target Rubin (the sm107 suite's fixture)."""
    from cudnn.frost import buffers

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)


# --------------------------------------------------------------------------- tier 1: CPU contract


def _facts(**kw):
    import cudnn
    from cudnn.sdpa import graph_analyzer as ga

    base = dict(b=2, h_q=8, h_kv=8, s_q=512, s_kv=512, d_qk=32, d_v=32, dtype=cudnn.data_type.HALF, dtype_o=cudnn.data_type.HALF, device_cc=(10, 7))
    base.update(kw)
    return ga.SdpaGraphFacts(**base)


def _caps():
    from cudnn.sdpa.fwd import engines

    return {s.name: s.capabilities for s in engines.ENGINE_SPECS}[_ROW]


def _rubin_pool():
    from cudnn.sdpa.fwd.api_dsl import _SM107_FLAVORS, _SM107_KERNEL_FILES

    return tuple(f for f in _SM107_FLAVORS if f in _SM107_KERNEL_FILES)


def test_row_serves_every_small_head_dim_on_the_next_width():
    """The three implementations of the pick agree on every served D (rule 8b'): the row's envelope walk
    (``engines._selected_d_shape``), the adapter's flavor pool (``api_dsl._pick_flavor``) and ``sm107_f16_tile_width``; a D above
    64 keeps landing on the d128 tile (72..128) or the wider rows (136..512); a non-multiple of 8 is a typed decline."""
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.api_dsl import _pick_flavor
    from cudnn.sdpa.fwd.config_sm107 import SM107_F16_SMALL_D_SHAPES, SM107_HEAD_DIM_TILE_WIDTHS, sm107_f16_tile_width

    caps, pool = _caps(), _rubin_pool()
    assert SM107_HEAD_DIM_TILE_WIDTHS == (32, 64, 128) and SM107_F16_SMALL_D_SHAPES <= caps.d_shapes
    for d, width in _SERVED.items():
        f = _facts(d_qk=d, d_v=d)
        assert engines.mismatch(caps, f) is None, (d, engines.mismatch(caps, f))
        assert engines._selected_d_shape(caps, f) == (width, width) == _pick_flavor(d, d, pool), d
        assert sm107_f16_tile_width(d, d) == width, d
    for d in (72, 96, 120, 128):
        assert engines._selected_d_shape(caps, _facts(d_qk=d, d_v=d)) == (128, 128) == _pick_flavor(d, d, pool) and sm107_f16_tile_width(d, d) == 128, d
    for d, want in (
        (136, (256, 256)),
        (192, (256, 256)),
        (256, (256, 256)),
        (512, (512, 512)),
    ):  # a square d192 head needs the d256 tile (the native pair is 192 x 128)
        assert engines._selected_d_shape(caps, _facts(d_qk=d, d_v=d)) == want == _pick_flavor(d, d, pool) and sm107_f16_tile_width(d, d) is None, d
    assert engines._selected_d_shape(caps, _facts(d_qk=192, d_v=128)) == (192, 128), "the d192xd128 native pair is untouched"
    for d in _DECLINED:
        why = engines.mismatch(caps, _facts(d_qk=d, d_v=d))
        assert why is not None and "multiples of 8" in why, (d, why)
        assert sm107_f16_tile_width(d, d) is None, d
    # asymmetric heads ride the width covering BOTH extents
    assert engines._selected_d_shape(caps, _facts(d_qk=48, d_v=32)) == (64, 64) == _pick_flavor(48, 32, pool) and sm107_f16_tile_width(48, 32) == 64
    assert engines._selected_d_shape(caps, _facts(d_qk=24, d_v=40)) == (64, 64) and sm107_f16_tile_width(24, 40) == 64
    assert _pick_flavor(24, 24, None) == (64, 64), "Blackwell has no 32-wide tile: its smallest flavor stays d64"


def test_row_serves_the_dense_leg_on_the_small_tiles():
    """Accept on the small tiles: the dense feature set at d32 (both dtypes, causal / bottom-right / right band / SWA / padded /
    sink / GQA / Stats natural and base-2 / no Stats), cga1 at d32, cga1 and cga2 at d64; the auto pick is cga1 on both."""
    import cudnn
    from cudnn.sdpa.fwd import engines, heuristics
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    caps = _caps()
    spec = {s.name: s for s in engines.ENGINE_SPECS}[_ROW]
    accept = (
        dict(),
        dict(dtype=cudnn.data_type.BFLOAT16, dtype_o=cudnn.data_type.BFLOAT16),
        dict(causal=True),
        dict(causal=True, bottom_right=True),
        dict(right_band_widening=True),
        dict(causal=True, window_left=64),
        dict(padded=True),
        dict(has_sink=True),
        dict(h_kv=2),
        dict(wants_stats=True, has_stats_log2=True),
        dict(wants_stats=False),
    )
    for kw in accept:
        assert engines.mismatch(caps, _facts(**kw)) is None, (kw, engines.mismatch(caps, _facts(**kw)))
        assert engines._selected_d_shape(caps, _facts(**kw)) == (32, 32), kw
    assert engines.mismatch(caps, _facts(), SdpaFwdKnobs(cga=1)) is None
    f64 = _facts(d_qk=64, d_v=64)
    assert engines.mismatch(caps, f64, SdpaFwdKnobs(cga=1)) is None and engines.mismatch(caps, f64, SdpaFwdKnobs(cga=2)) is None
    assert engines.effective_cgas(caps, _facts()) == frozenset({1}) and engines.effective_cgas(caps, f64) == frozenset({1, 2})
    assert engines.effective_cgas(caps, _facts(d_qk=128, d_v=128)) == frozenset({2}), "the d128 tile keeps its cga2-only domain"
    for f in (_facts(), f64):
        assert heuristics._auto_sched_cga(spec, f, split_kv=1, sched_policy=0) == (0, 1), "the small tiles default to cga1"


# The ROUTE table of the legs beyond the dense one, per head dim: (leg, facts kwargs, knobs, adapter kwargs) -> the width the
# graph lands on.  "128" = the d128 flavor the graph rode before the small tiles (served exactly as then); 64 = the 64-wide tile
# (a cga2 request at D <= 32 lands on the smallest covering tile validated at cga2); None = a typed decline that predates the
# tiles (THD / paged KV are exact-shape legs, Rubin half PackGQA needs paged KV or a D128 packed split) -- the decline's words.
_LEGS = (
    ("split-KV", dict(), dict(split_kv=2), dict(split_kv=2), {8: 128, 24: 128, 32: 128, 48: 128, 64: 128, 128: 128}),
    ("split-KV", dict(), dict(split_kv=2, cga=2), dict(split_kv=2, cga=2), {8: 128, 24: 128, 32: 128, 48: 128, 64: 128, 128: 128}),
    ("score side outputs", dict(has_score_max=True, has_score_sum_exp=True), None, dict(score=True), {8: 128, 24: 128, 32: 128, 48: 128, 64: 128, 128: 128}),
    (
        "score side outputs",
        dict(has_score_max=True, has_score_sum_exp=True),
        dict(cga=2),
        dict(score=True, cga=2),
        {8: 128, 24: 128, 32: 128, 48: 128, 64: 128, 128: 128},
    ),
    (None, dict(), dict(cga=2), dict(cga=2), {8: 64, 24: 64, 32: 64, 48: 64, 64: 64, 128: 128}),
    (None, dict(), dict(cga=1), dict(cga=1), {8: 32, 24: 32, 32: 32, 48: 64, 64: 64, 128: None}),
    ("THD", dict(thd=True, padded=True), None, dict(thd=True), {8: None, 24: None, 32: None, 48: None, 64: None, 128: 128}),
    ("THD", dict(has_paged_kv=True, padded=True, page_size=128, thd=True), None, None, {8: None, 24: None, 32: None, 48: None, 64: None, 128: 128}),
    ("PackGQA", dict(h_kv=2), dict(pack_gqa=True), dict(h_kv=2, pack_gqa=True), {8: None, 24: None, 32: None, 48: None, 64: None, 128: None}),
)
_DECLINE_WORDS = {"THD": "THD", "PackGQA": "PackGQA", None: "cga"}


def test_row_routes_every_leg_beyond_dense_past_the_small_tiles():
    """The small tiles carry the dense leg ONLY (``Capabilities.dense_only_d_shapes``): split-KV and the score side outputs at
    D <= 64 select the d128 flavor (the pre-tile route, where ``split_d_shapes`` / ``score_stats_d_shapes`` name it) and are
    SERVED, a cga2 request at D <= 32 selects the 64-wide tile, cga1 the small tile; THD / paged KV / PackGQA keep their pre-tile
    typed declines.  The three pick implementations agree on every cell (rule 8b'): the row's walk (``_selected_d_shape`` with
    the knobs), ``sm107_f16_tile_width(leg=, cga=)`` and the adapter's flavor; ``effective_cgas`` with the requested cga is the
    domain of the flavor the request lands on (the knob record `{TILE_CGA_M: 2}` at D <= 32 keeps working)."""
    from cudnn.sdpa.fwd import engines, heuristics
    from cudnn.sdpa.fwd.config_sm107 import sm107_f16_tile_width, sm107_small_tile_leg
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    caps = _caps()
    spec = {s.name: s for s in engines.ENGINE_SPECS}[_ROW]
    assert caps.dense_only_d_shapes == frozenset({(32, 32), (64, 64)})
    assert sm107_small_tile_leg() is None and sm107_small_tile_leg(split_kv=1, pack_gqa=False) is None
    for kw, name in (
        (dict(thd=True), "THD"),
        (dict(paged=True), "paged KV"),
        (dict(split_kv=2), "split-KV"),
        (dict(pack_gqa=True), "PackGQA"),
        (dict(score_stats=True), "score side outputs"),
    ):
        assert sm107_small_tile_leg(**kw) == name, kw
    for leg, fkw, kkw, akw, want in _LEGS:
        knobs = SdpaFwdKnobs(**kkw) if kkw else None
        for d, width in want.items():
            f = _facts(d_qk=d, d_v=d, **fkw)
            why = engines.mismatch(caps, f, knobs)
            selected = engines._selected_d_shape(caps, f, knobs)
            as_leg = sm107_small_tile_leg(
                thd=f.thd,
                paged=f.has_paged_kv,
                split_kv=kkw.get("split_kv") if kkw else 1,
                pack_gqa=kkw.get("pack_gqa") if kkw else False,
                score_stats=f.has_score_max,
            )
            assert as_leg == leg, (leg, d, as_leg)
            picked = sm107_f16_tile_width(d, d, leg=leg, cga=kkw.get("cga") if kkw else None)
            if width is None:
                assert why is not None and _DECLINE_WORDS[leg] in why, (leg, kkw, d, why)
                continue
            assert why is None, (leg, kkw, d, why)
            assert selected == (width, width) == (picked, picked), (leg, kkw, d, selected, picked)
            if knobs is not None and knobs.cga is not None:
                assert knobs.cga in engines.effective_cgas(caps, f, knobs.split_kv, knobs.cga), (leg, kkw, d)
            if akw is not None:
                api = _api(d, **akw)
                assert api.check_support(), (leg, kkw, d)
                assert api.flavor == (width, width), (leg, kkw, d, api.flavor)
                params = api.template_params()
                assert params.d_flavor == width and params.cta_mma == (kkw.get("cga") if kkw and kkw.get("cga") else (1 if width <= 64 else 2)), (
                    leg,
                    kkw,
                    d,
                    params,
                )
    f24 = _facts(d_qk=24, d_v=24)
    assert engines.effective_cgas(caps, f24) == frozenset({1}), "the auto domain at D <= 32 is the 32-wide tile's"
    assert engines.effective_cgas(caps, f24, None, 2) == frozenset({1, 2}), "a cga2 request lands on the 64-wide tile"
    assert engines.effective_cgas(caps, f24, 2) == frozenset({2}), "a split graph lands on the d128 flavor"
    assert heuristics._auto_sched_cga(spec, f24, split_kv=2, sched_policy=0) == (0, 2)
    assert heuristics._auto_sched_cga(spec, _facts(d_qk=24, d_v=24, has_score_max=True, has_score_sum_exp=True), split_kv=1, sched_policy=0) == (0, 2)
    # a cga nobody serves is still a typed decline naming the domain of the flavor it would land on
    why = engines.mismatch(caps, f24, SdpaFwdKnobs(cga=3))
    assert why is not None and "cga=3" in why and "[2]" in why, why


def test_one_constant_feeds_the_row_the_adapter_and_the_factory():
    """``SM107_SMALL_D_CGAS`` is read by the row (cgas_by_d_shape), the adapter's cga twin (``supported_cgas_for``) and the config
    factory (which refuses any other width) -- the three cannot drift (rule 8b').  Same for the shapes and the kernel-file map."""
    from cudnn.sdpa.fwd.api_dsl import _SM107_KERNEL_FILES, supported_cgas_for
    from cudnn.sdpa.fwd.config_sm107 import SM107_F16_SMALL_D_SHAPES, SM107_SMALL_D_CGAS, TemplateParams, make_cfg_d32, make_cfg_d64

    caps = _caps()
    assert set(SM107_SMALL_D_CGAS) == SM107_F16_SMALL_D_SHAPES == {(32, 32), (64, 64)}
    assert SM107_SMALL_D_CGAS == {(32, 32): frozenset({1}), (64, 64): frozenset({1, 2})}
    by_shape = dict(caps.cgas_by_d_shape)
    for shape, domain in SM107_SMALL_D_CGAS.items():
        assert by_shape[shape] == domain, shape
        assert supported_cgas_for(shape, fp8=False, device_cc=(10, 7)) == tuple(sorted(domain)), shape
        assert _SM107_KERNEL_FILES[shape] == _SM107_KERNEL_FILES[(128, 128)] == "sm107/prefill_d128_f16.py", shape
    with pytest.raises(ValueError, match="outside the validated widths"):
        make_cfg_d32(TemplateParams(dtype_qkv=2, cta_mma=2))
    for cta in (1, 2):
        cfg, _ = make_cfg_d64(TemplateParams(dtype_qkv=2, cta_mma=cta))
        assert cfg.CTA_MMA == cta


@pytest.mark.parametrize("width", [32, 64])
def test_config_factories_derive_everything_from_the_tile(width):
    """Every coupled value follows TILE_K = TILE_O = width -- swizzle atoms, TMA iteration counts, the ring depth, the TMEM map, the
    SMEM footprint -- and the factory's validator raises on each coupling when it is moved alone (frost-kernels.md s5 / s6)."""
    from cudnn.sdpa.fwd import config_sm107 as c

    make = {32: c.make_cfg_d32, 64: c.make_cfg_d64}[width]
    for cta in sorted(c.SM107_SMALL_D_CGAS[(width, width)]):
        cfg, tma = make(c.TemplateParams(dtype_qkv=2, cta_mma=cta))
        assert type(cfg).__name__ == f"CfgD{width}" and cfg.TILE_K == cfg.TILE_O == width and cfg.TILE_N == cfg.TILE_M == 128 and cfg.TILES_Q == 2
        assert cfg.STAGES_KV == 4, "the small tiles run the 4-deep ring at both widths"
        assert cfg.Q_SWZ_BYTES == cfg.K_SWZ_BYTES == c.q_swz_bytes(width, 2) == (64 if width == 32 else 128)
        assert cfg.V_SWZ_BYTES == c.v_swz_bytes(width, cta, 2) == (64 if (width // cta) == 32 else 128)
        assert cfg.O_SWZ_BYTES == c.o_swz_bytes(width, 2) == (64 if width == 32 else 128)
        assert tma.QK_ITERS == width * 2 // cfg.Q_SWZ_BYTES == 1 and tma.QK_GRANU_ELEMS == width
        assert tma.VO_ITERS == width * 2 // cfg.V_SWZ_BYTES and tma.VO_ITERS // cta == 1, "one V box per tile per CTA"
        assert cfg.N_BMM2_CHUNKS * cfg.BMM2_CHUNK_SIZE == cfg.TILE_N == 128, "the softmax chunk is TILE_N-derived, not TILE_O's"
        lay = c.d128_family_tmem_layout(cfg)
        assert lay.S_OFF == (0, 128) and lay.P_OFF == (64, 192) and lay.O_OFF == (256, 256 + width) and lay.STATS_OFF == 544 and lay.TOTAL_COLS == 576
        smem = c.sdpa_smem_bytes(cfg.TILE_M, cfg.TILE_N, cfg.TILE_K, cfg.TILE_O, cfg.TILES_Q, cfg.STAGES_KV, cfg.CTA_MMA, cfg.BPE, cfg.BPE_O)
        # Q + O slabs (2 x 128 x width x 2 B each) + the K and V rings (4 x 128 x width x 2 B each, halved per CTA at cga2)
        assert smem == 2 * (2 * 128 * width * 2) + 2 * (4 * 128 * width * 2 // cta), smem
    cfg, _ = make(c.TemplateParams(dtype_qkv=2, cta_mma=1))
    for field, value, word in (
        ("Q_SWZ_BYTES", 128 if width == 32 else 64, "Q/K swizzle"),
        ("V_SWZ_BYTES", 32, "V swizzle"),
        ("O_SWZ_BYTES", 128 if width == 32 else 64, "O swizzle"),
        ("BMM2_CHUNK_SIZE", 32, "TILE_N-derived"),
        ("TILE_O", 48, "TILE_O one of"),
        ("TILE_O", 128 if width == 32 else 32, "TILE_K == TILE_O"),
    ):
        with pytest.raises(ValueError, match=re.escape(word)):
            c._validate_head_dim_tile(dataclasses.replace(cfg, **{field: value}), "probe")
    with pytest.raises(ValueError, match="f16/bf16 prefill kernel only"):
        make(c.TemplateParams(dtype_qkv=0, cta_mma=1))
    with pytest.raises(ValueError, match="THD/varlen"):
        make(c.TemplateParams(dtype_qkv=2, cta_mma=1, thd_varlen=True, seq_kv_lens_present=True))
    with pytest.raises(ValueError, match="epilogue_gate"):
        make(c.TemplateParams(dtype_qkv=2, cta_mma=1, epilogue_gate=True))


def test_d128_tmem_map_is_the_historical_one():
    """The derived map at TILE_K = TILE_O = 128 is exactly what the kernel always had (S 0 / 128, P 64 / 192, O 256 / 384, STATS 544)
    -- the d128 rendering does not move (the cubin md5 records in test_sdpa_fwd_score_stats.py are the binary pin)."""
    from cudnn.sdpa.fwd import config_sm107 as c

    for cta in (1, 2):
        cfg, _ = c.make_cfg_d128(c.TemplateParams(dtype_qkv=2, cta_mma=cta))
        lay = c.d128_family_tmem_layout(cfg)
        assert (lay.S_OFF, lay.P_OFF, lay.O_OFF, lay.STATS_OFF, lay.TOTAL_COLS) == ((0, 128), (64, 192), (256, 384), 544, 576)


def test_loader_binds_the_flavor_to_the_record():
    """``_load_sm100_kernel_module((32, 32) | (64, 64), <bare record>, rubin=True)`` traces the small geometry: the loader sets
    ``d_flavor`` from the flavor, so no caller can pair the key with the d128 geometry under a d32 tag."""
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    for width, o1 in ((32, 288), (64, 320), (128, 384)):
        mod = _load_sm100_kernel_module((width, width), TemplateParams(dtype_qkv=2, cta_mma=1 if width < 128 else 2), rubin=True)
        assert mod.CFG.TILE_K == mod.CFG.TILE_O == width and mod.PARAMS.d_flavor == width, width
        assert (mod.LAYOUT.S0_OFF, mod.LAYOUT.S1_OFF, mod.LAYOUT.P0_OFF, mod.LAYOUT.P1_OFF, mod.LAYOUT.O0_OFF, mod.LAYOUT.O1_OFF, mod.LAYOUT.STATS_OFF) == (
            0,
            128,
            64,
            192,
            256,
            o1,
            544,
        )
        assert mod.LAYOUT.TOTAL_COLS == 576 and mod.DESC_VERSION == 0 and mod.SPIN_RING_WAITS is True
        assert f"d{width}" in mod.__name__, mod.__name__
        assert sorted(mod._MAKE_CFG) == [32, 64, 128]


def test_expect_tx_is_the_full_box_and_every_box_is_the_tile():
    """Static pin of the zero-fill contract (frost-tile-dsl.md s5: an overshooting TMA box delivers the FULL box's bytes): every
    per-tile element count is a TILE product, every transaction byte count is that count x BPE x CTA_MMA, and no head-dim
    EXTENT (d_qk / d_v) reaches a byte count or a box -- the extents live in the descriptors only."""
    code = "\n".join(ln for ln in _KERNEL.read_text().splitlines() if not ln.lstrip().startswith("#"))
    for line in (
        "qBufferElems = CFG.TILE_M * CFG.TILE_K",
        "kBufferElems = CFG.TILE_N * CFG.TILE_K // CFG.CTA_MMA",
        "vBufferElems = CFG.TILE_O * CFG.TILE_N // CFG.CTA_MMA",
        "qTmaTransactionBytes = qBufferElems * CFG.BPE * CFG.CTA_MMA",
        "kTmaTransactionBytes = kBufferElems * CFG.BPE * CFG.CTA_MMA",
        "vTmaTransactionBytes = vBufferElems * CFG.BPE * CFG.CTA_MMA",
        "qk_box_q = (1, CFG.TILE_M, 1, TMA_QK_GRANU_ELEMS)",
        "qk_box_k = (1, CFG.TILE_N // CFG.CTA_MMA, 1, TMA_QK_GRANU_ELEMS)",
        "vo_box_v = (1, CFG.TILE_N, 1, TMA_VO_GRANU_ELEMS)",
        "vo_box_o = (1, CFG.TILE_M, 1, _O_GRANU_ELEMS)",
    ):
        assert line in code, line
    for bad in (r"TransactionBytes\s*=.*\bd_(qk|v)\b", r"box_[qkvo]\s*=.*\bd_(qk|v)\b", r"BufferElems\s*=.*\bd_(qk|v)\b"):
        assert not re.search(bad, code), bad
    assert "_MAKE_CFG = {32: make_cfg_d32, 64: make_cfg_d64, 128: make_cfg_d128}" in code
    assert "d128_family_tmem_layout(CFG)" in code and not re.search(r"^\s*(S1|O0|O1|STATS)_OFF: int = \d+\s*$", code, re.M), "no TMEM literal survives"
    assert re.search(r"CHUNK = CFG\.BMM2_CHUNK_SIZE", code) and not re.search(r"^\s*CHUNK = 64\s*$", code, re.M)


def _desc(shape, dtype, name):
    from cudnn.api_base import TensorDesc

    b, h, s, d = shape
    stride = (s * h * d, d, h * d, 1)
    return TensorDesc(
        dtype=dtype, shape=tuple(shape), stride=stride, stride_order=TensorDesc._compute_stride_order(tuple(shape), stride), device="cuda", name=name
    )


def _api(d, *, s=256, h=4, h_kv=None, dtype=torch.bfloat16, score=False, **kw):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    h_kv = h if h_kv is None else h_kv
    q, o = (_desc((1, h, s, d), dtype, n) for n in ("q", "o"))
    k, v = (_desc((1, h_kv, s, d), dtype, n) for n in ("k", "v"))
    if score:  # the two score side outputs, declared like Stats (b, h, s, 1) fp32
        kw["sample_score_max"] = _desc((1, h, s, 1), torch.float32, "score_max")
        kw["sample_score_sum_exp"] = _desc((1, h, s, 1), torch.float32, "score_sum_exp")
    return SdpaFwdDslSm100(q, k, v, o, None, **kw)


@pytest.fixture(autouse=True)
def _rubin_device_for_the_adapter(monkeypatch):
    """The standalone adapter reads the device capability at construction: model a Rubin device on any host."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 7))


def test_adapter_check_support_twins_and_record():
    """The standalone adapter on a Rubin device: a small head dim is admitted, the compile-time record carries the tile's
    ``d_flavor`` and cga1; the legs beyond the dense one route like the row (cga2 at d24 -> the 64-wide tile, split-KV / the
    score outputs -> the d128 flavor); the row's declines have their typed twins (THD, a non-multiple of 8)."""
    for d, width in ((24, 32), (32, 32), (48, 64), (64, 64), (8, 32)):
        api = _api(d)
        assert api.check_support(), d
        assert api.flavor == (width, width), (d, api.flavor)
        params = api.template_params()
        assert params.d_flavor == width and params.cta_mma == 1, (d, params.d_flavor, params.cta_mma)
    api = _api(48, cga=2)
    assert api.check_support() and api.template_params().cta_mma == 2
    api = _api(24, cga=2)
    assert api.check_support() and api.flavor == (64, 64) and api.template_params().cta_mma == 2 and api.template_params().d_flavor == 64
    for kw in (dict(split_kv=2), dict(score=True), dict(split_kv=2, cga=2)):
        api = _api(24, **kw)
        assert api.check_support() and api.flavor == (128, 128), (kw, api.flavor)
        assert api.template_params().d_flavor == 128 and api.template_params().cta_mma == 2, kw
    with pytest.raises((ValueError, NotImplementedError), match="THD"):
        _api(24, thd=True).check_support()
    with pytest.raises((ValueError, NotImplementedError)):
        _api(20).check_support()
    api128 = _api(128)
    assert api128.check_support() and api128.template_params().d_flavor == 128 and api128.template_params().cta_mma == 2


# --------------------------------------------------------------------------- tier 2: host rendering pins (sm_107a trace-compile)

_PROBE_BODY = """
d = params_kw.pop("__d")
params = TemplateParams(**params_kw)
mod = _load_sm100_kernel_module((d, d), params, rubin=True)
print("TILE_K", mod.CFG.TILE_K)
print("O1_OFF", mod.LAYOUT.O1_OFF)
mod.compile(d_qk=d, d_v=d, has_lse=True)
"""
_COUNTS = {"USETMAXREG": ("USETMAXREG",), "STL": (" STL",), "LDL": (" LDL",), "FMNMX3": ("FMNMX3",), "R2P": (" R2P ",), "CGAERRBAR": ("CGAERRBAR",)}


def _probe(tmp_path, d, cta, mask, tag):
    params = {"__d": d, "dtype_qkv": 2, "cta_mma": cta}
    if mask == "causal":
        params["window_right"] = 0
    return run_sass_probe(tmp_path, probe_src=sass_probe_source(_PROBE_BODY, _COUNTS), arch="sm_107a", params=params, tag=tag)


@pytest.mark.parametrize("mask", ["dense", "causal"])
def test_small_tile_renderings_keep_the_register_split_and_spill_no_more(tmp_path, mask):
    """Each new rendering (d32 cga1, d64 cga1, d64 cga2) traced for sm_107a keeps the per-role register split (USETMAXREG > 0 --
    a rendering at 0 runs flat at the entry count) and spills no more than the d128 rendering of the SAME mask arm at the same
    width (the d128 cga1 arm is the like-for-like base for the cga1 tiles, its cga2 arm for the cga2 tile); no cluster-scope
    drain (CGAERRBAR 0); the masked arm keeps the bit-word form (R2P > 0).  The probe prints the module's geometry so a probe
    that traced the d128 geometry under a small-tile key fails here, not in a GPU test."""
    base1 = _probe(tmp_path, 128, 1, mask, f"d128_cga1_{mask}")
    base2 = _probe(tmp_path, 128, 2, mask, f"d128_cga2_{mask}")
    assert base1.expect["TILE_K"] == 128 and base1.expect["O1_OFF"] == 384
    for d, cta, base in ((32, 1, base1), (64, 1, base1), (64, 2, base2)):
        probe = _probe(tmp_path, d, cta, mask, f"d{d}_cga{cta}_{mask}")
        assert probe.expect["TILE_K"] == d and probe.expect["O1_OFF"] == 256 + d, (d, probe.expect)
        assert probe.stats["USETMAXREG"] > 0, (d, cta, mask, probe.stats)
        assert probe.stats["CGAERRBAR"] == 0, (d, cta, mask, probe.stats)
        assert_no_new_spills(probe.stats, base.stats, tag=f"d{d} cga{cta} {mask} vs d128 cga{cta}: ")
        if mask == "causal":
            assert probe.stats["R2P"] > 0 and probe.stats["FMNMX3"] > 0, (d, cta, probe.stats)


# --------------------------------------------------------------------------- tier 3: GPU cells (a Rubin device)

_DTYPES = [torch.float16, torch.bfloat16]
_DTYPE_IDS = ["fp16", "bf16"]
# Tolerances DERIVED from the kernel's arithmetic (the score-side-output suite's analysis, test_sdpa_fwd_score_stats.py):
#   O: the suite's half-precision O bound (an fp32 accumulation over <= 1024 keys, one bf16 / f16 rounding at the store).
#   LSE: max * ln 2 + ln(sum) in fp32 from the fp32 S accumulation (|err| <= D * 2^-24 * sum|q k| ~ 2e-4 at these magnitudes, one
#        FMUL + one lg2.approx + one FADD) and a row-sum of <= 1024 ex2.approx terms (< 4e-4 relative -> < 4e-4 absolute in the
#        log): atol 1e-3 is >= 2x the bound; the measured cells sit at ~1e-6.
_TOL_O = dict(atol=5e-2, rtol=3e-2)
# The existing f16 row tolerances (test_sdpa_fwd_dsl_sm107.py: the O pair of its oracle cells, the 1e-4 / 1e-4 LSE pair of its
# log-sum-exp pins); the small tiles' cells measure max|dLSE| ~1e-6 against the fp64 oracle.
_TOL_LSE = dict(atol=1e-4, rtol=1e-4)


def _bhsd(b, h, s, d, dtype):
    return torch.randn(b, s, h, d, device="cuda", dtype=dtype).transpose(1, 2)


def _ref(q, k, v, *, scale, causal=False, bottom_right=False, seq_q_lens=None, seq_kv_lens=None, sinks=None, window_left=None, right_bound=None):
    """fp64 oracle of the whole contract: O and the natural-log LSE per row class (keyless rows -inf / 0)."""
    b, h_q, s_q, _ = q.shape
    _, h_kv, s_kv, _ = v.shape
    dev = q.device
    g = h_q // h_kv
    k_ref = k.repeat_interleave(g, dim=1).double()
    v_ref = v.repeat_interleave(g, dim=1).double()
    scores = torch.matmul(q.double(), k_ref.transpose(-1, -2)) * scale
    q_lens = seq_q_lens.flatten().to(device=dev, dtype=torch.int64) if seq_q_lens is not None else torch.full((b,), s_q, dtype=torch.int64, device=dev)
    kv_lens = seq_kv_lens.flatten().to(device=dev, dtype=torch.int64) if seq_kv_lens is not None else torch.full((b,), s_kv, dtype=torch.int64, device=dev)
    i = torch.arange(s_q, device=dev).view(1, 1, s_q, 1)
    j = torch.arange(s_kv, device=dev).view(1, 1, 1, s_kv)
    diag = i + (kv_lens - q_lens).view(b, 1, 1, 1) if bottom_right else i
    dead_q = i >= q_lens.view(b, 1, 1, 1)
    masked = dead_q | (j >= kv_lens.view(b, 1, 1, 1))
    if causal:
        masked = masked | (j > diag)
    if right_bound is not None:
        masked = masked | (j > diag + right_bound)
    if window_left is not None:
        masked = masked | (j < diag - window_left)
    scores = scores.masked_fill(masked, float("-inf"))
    full = scores if sinks is None else torch.cat([scores, sinks.view(1, h_q, 1, 1).double().expand(b, h_q, s_q, 1)], dim=-1)
    smax = full.max(dim=-1).values
    ssum = torch.exp(full - smax.unsqueeze(-1)).sum(dim=-1).nan_to_num(0.0)
    lse = smax + torch.log(ssum)
    o = torch.matmul(torch.softmax(full, dim=-1).nan_to_num(0.0)[..., :s_kv], v_ref)
    dead = dead_q.squeeze(-1)
    return dict(o=o, lse=lse.masked_fill(dead, float("-inf")), smax=smax.masked_fill(dead, float("-inf")), empty=torch.isneginf(lse) | dead)


def _run(
    q, k, v, *, scale, sdpa_kwargs=None, seq_len_kv=None, seq_len_q=None, sink=None, stats=True, stats_log2=False, cga=None, split_kv=None, score_outputs=False
):
    """Build the graph, pin the Rubin f16 row (optionally a cga / split_kv knob through the public autotune record, optionally
    the two score side outputs), launch TWICE on sentinel-filled outputs (the two-launch delta) and return O, Stats, the side
    outputs and the plan's knobs."""
    import cudnn

    b, h_q, s_q, _ = q.shape
    d_v = v.shape[-1]
    dtype = q.dtype
    o_gpu = torch.empty(b, s_q, h_q, d_v, device="cuda", dtype=dtype).transpose(1, 2)
    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tq, tk, tv = g.tensor_like(q), g.tensor_like(k), g.tensor_like(v)
    kw = dict(name="sdpa", q=tq, k=tk, v=tv, generate_stats=stats, attn_scale=scale, stats_use_log2=stats_log2)
    vp = {tq: q, tk: k, tv: v}
    side = {}
    if score_outputs:  # the graph's two optional ports, declared like Stats (b, h, s, 1) fp32
        for name in ("score_max", "score_sum_exp"):
            buf = make_dense_stats(b, h_q, s_q, "contiguous")
            t = g.tensor(dim=list(buf.shape), stride=list(buf.stride()), data_type=cudnn.data_type.FLOAT, name=name)
            kw[name] = t
            vp[t] = buf
            side[name] = buf
    if seq_len_kv is not None:
        slk = g.tensor_like(seq_len_kv)
        kw.update(seq_len_kv=slk, use_padding_mask=True)
        vp[slk] = seq_len_kv
        slq_t = seq_len_q if seq_len_q is not None else torch.full((b, 1, 1, 1), s_q, dtype=torch.int32, device="cuda")
        slq = g.tensor_like(slq_t)
        kw["seq_len_q"] = slq
        vp[slq] = slq_t
    if sink is not None:
        st = g.tensor_like(sink)
        kw["sink_token"] = st
        vp[st] = sink
    kw.update(sdpa_kwargs or {})
    o, stats_t = g.sdpa(**kw)
    o.set_output(True).set_dim(o_gpu.shape).set_stride(o_gpu.stride())
    stats_gpu = None
    if stats:
        stats_gpu = make_dense_stats(b, h_q, s_q, "contiguous")
        stats_t.set_output(True).set_dim(stats_gpu.shape).set_stride(stats_gpu.stride()).set_data_type(cudnn.data_type.FLOAT)
        vp[stats_t] = stats_gpu
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    ours = [i for i, n in enumerate(names) if n == _ROW or n.startswith(_ROW + "[")]
    idx = next((i for i in ours if g.plans[i].knobs.split_kv in (None, 1)), None)
    assert idx is not None, f"no unsplit {_ROW} plan in {names}"
    picked_cga, picked_split = g.plans[idx].knobs.cga, g.plans[idx].knobs.split_kv or 1
    if (cga is not None and picked_cga != cga) or (split_kv is not None and picked_split != split_kv):
        # The heuristics rank ONE knob point per row (the auto pick); another member of the row's cga / split domain is requested
        # the way an autotuner replays it -- the public record (engine_id, {knob_type: int}) through create_execution_plan.
        engine_id, record = g.get_engine_and_knobs_at_index(idx)
        record = dict(record)
        if cga is not None:
            record[cudnn.knob_type.TILE_CGA_M] = cga
            picked_cga = cga
        if split_kv is not None:
            record[cudnn.knob_type.SPLIT_KV] = split_kv
            picked_split = split_kv
        g.create_execution_plan(engine_id, record)
        idx = g.get_execution_plan_count() - 1
    g.select_plan(idx)
    knobs = SimpleNamespace(cga=picked_cga, split_kv=picked_split)
    g.check_support()
    g.build_plans()
    vp[o] = o_gpu
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)

    def launch():
        o_gpu.fill_(float("nan"))  # sentinel: an unwritten live row stays NaN
        if stats_gpu is not None:
            stats_gpu.fill_(float("nan"))
        for buf in side.values():
            buf.fill_(float("nan"))
        g.execute(vp, ws)
        torch.cuda.synchronize()
        return o_gpu.clone(), (stats_gpu.clone() if stats_gpu is not None else None), {n: buf.clone() for n, buf in side.items()}

    o1, st1, side1 = launch()
    o2, st2, side2 = launch()
    assert torch.equal(o1, o2) and (st1 is None or torch.equal(st1, st2)), "two-launch delta on O / Stats"
    assert all(torch.equal(side1[n], side2[n]) for n in side), "two-launch delta on the score side outputs"
    return dict(o=o_gpu, stats=stats_gpu, knobs=knobs, **side)


def _check(out, ref, *, has_stats=True, stats_log2=False):
    """Keyless rows EXACTLY 0 / -inf (never NaN, never a residue); live rows against the fp64 oracle."""
    empty = ref["empty"]
    o = out["o"]
    assert torch.isfinite(o).all(), "O has a non-finite (an unwritten sentinel or a NaN residue)"
    assert (o[empty] == 0).all(), "a keyless row's O must be exactly 0 (a SELECT, never a multiply by zero)"
    torch.testing.assert_close(o[~empty].double(), ref["o"][~empty], **_TOL_O)
    if has_stats:
        lse = out["stats"].squeeze(-1)
        assert torch.isneginf(lse[empty]).all(), "a keyless row's Stats must be exactly -inf"
        ref_lse = ref["lse"][~empty] * (math.log2(math.e) if stats_log2 else 1.0)
        torch.testing.assert_close(lse[~empty].double(), ref_lse, **_TOL_LSE)


_CASES = {
    # the degenerate-input matrix (sdpa-invariants.md s9): the KV ring sweep n_kv = 1 (the P14 phase-drift shape), 2, 3, 4 (the ring
    # depth), 5 (depth + 1), 8 -- every one with B*H > 1; tail tiles (S not a multiple of 128; the row's synthesized KV padding on the
    # dense one, whose Q tail also exercises the OOB-row store guard); GQA; a zero-length KV batch; keyless bottom-right rows with a
    # sink and padded-Q trim; keyless rows past a left window; a right band; ONE CTA / one tile in the whole launch (B*H = 1, n_kv = 1)
    "dense_one_tile_bh1": dict(b=1, h=1, h_kv=1, s_q=128, s_kv=128),
    "dense_nkv1": dict(b=2, h=2, h_kv=2, s_q=128, s_kv=128),
    "dense_nkv2": dict(b=2, h=4, h_kv=4, s_q=256, s_kv=256),
    "dense_nkv3_tail_300": dict(b=2, h=2, h_kv=2, s_q=300, s_kv=300),
    "dense_nkv4_512": dict(b=2, h=2, h_kv=2, s_q=512, s_kv=512),
    "dense_nkv5_640": dict(b=2, h=2, h_kv=2, s_q=640, s_kv=640),
    "dense_nkv8_1024": dict(b=2, h=2, h_kv=2, s_q=1024, s_kv=1024),
    "causal_tail_385": dict(b=1, h=4, h_kv=4, s_q=385, s_kv=385, causal=True),
    "right_band_64_tail_300": dict(b=2, h=2, h_kv=2, s_q=300, s_kv=300, right_bound=64),
    "causal_swa128_padded_385": dict(b=2, h=2, h_kv=2, s_q=385, s_kv=385, causal=True, window_left=128, kv_lens=[385, 200]),
    "causal_gqa_8_2": dict(b=2, h=8, h_kv=2, s_q=256, s_kv=256, causal=True),
    "padded_kv_zero_batch": dict(b=3, h=4, h_kv=4, s_q=256, s_kv=256, kv_lens=[256, 0, 100]),
    "bottom_right_keyless_sink": dict(b=3, h=8, h_kv=4, s_q=256, s_kv=256, bottom_right=True, q_lens=[256, 129, 0], kv_lens=[200, 256, 128], sink=True),
}


def _case_inputs(case, d, dtype, *, seed=0):
    torch.manual_seed(seed)
    c = _CASES[case]
    q = _bhsd(c["b"], c["h"], c["s_q"], d, dtype)
    k = _bhsd(c["b"], c["h_kv"], c["s_kv"], d, dtype)
    v = _bhsd(c["b"], c["h_kv"], c["s_kv"], d, dtype)
    kv_lens = torch.tensor(c["kv_lens"], dtype=torch.int32, device="cuda").view(c["b"], 1, 1, 1) if "kv_lens" in c else None
    q_lens = torch.tensor(c["q_lens"], dtype=torch.int32, device="cuda").view(c["b"], 1, 1, 1) if "q_lens" in c else None
    sink = torch.randn(1, c["h"], 1, 1, dtype=torch.float32, device="cuda") if c.get("sink") else None
    causal = bool(c.get("causal") or c.get("bottom_right"))
    sdpa_kwargs = dict(use_causal_mask_bottom_right=True) if c.get("bottom_right") else (dict(use_causal_mask=True) if causal else {})
    window_left = c.get("window_left")
    if window_left is not None:
        sdpa_kwargs["sliding_window_length"] = window_left + 1  # the graph's length L keeps L keys up to the diagonal: W = L - 1
    right_bound = c.get("right_bound")
    if right_bound is not None:
        sdpa_kwargs["diagonal_band_right_bound"] = right_bound  # alone: with use_causal_mask the bound is forced back to 0
    scale = 1.0 / math.sqrt(d) if not c.get("bottom_right") else 0.75
    ref = _ref(
        q, k, v, scale=scale, causal=causal, bottom_right=bool(c.get("bottom_right")), seq_q_lens=q_lens, seq_kv_lens=kv_lens,
        sinks=sink.flatten() if sink is not None else None, window_left=window_left, right_bound=right_bound,
    )  # fmt: skip
    return q, k, v, dict(scale=scale, sdpa_kwargs=sdpa_kwargs, seq_len_kv=kv_lens, seq_len_q=q_lens, sink=sink), ref


def _assert_small_tile_plan(out, d):
    """The route evidence: the plan's TILE_CGA_M knob is 1 on the small tiles (D <= 64; the d128 tile is cga2-only)."""
    assert out["knobs"].cga == (1 if d <= 64 else 2) and out["knobs"].split_kv == 1, (d, out["knobs"])


@requires_rubin
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("d", [8, 16, 24, 32, 48, 64, 128])
@pytest.mark.parametrize("case", list(_CASES))
def test_small_tiles_match_the_fp64_oracle(case, d, dtype):
    """Every served head dim of the file's three geometries -- 8 / 16 / 24 / 32 on the 32-wide tile, 48 / 64 on the 64-wide one,
    128 on the d128 tile (the same file, byte-identical to its one-geometry past) -- over the degenerate-input matrix, both dtypes."""
    q, k, v, kw, ref = _case_inputs(case, d, dtype)
    out = _run(q, k, v, **kw)
    _assert_small_tile_plan(out, d)
    _check(out, ref)
    c = _CASES[case]
    if "kv_lens" in c and 0 in c["kv_lens"]:
        empty_b = c["kv_lens"].index(0)
        assert (out["o"][empty_b] == 0).all() and torch.isneginf(out["stats"][empty_b]).all(), "a zero-length KV batch: O = 0, Stats = -inf"
    if "window_left" in c:
        first_dead = c["kv_lens"][1] + c["window_left"]
        lse = out["stats"].squeeze(-1)
        assert torch.isneginf(lse[1, :, first_dead:]).all() and (out["o"][1, :, first_dead:] == 0).all(), "rows past the window are keyless"
        assert torch.isfinite(lse[1, :, :first_dead]).all() and torch.isfinite(lse[0]).all()


@requires_rubin
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("d", [40, 56])
@pytest.mark.parametrize("case", ["dense_nkv2", "causal_tail_385", "padded_kv_zero_batch"])
def test_every_other_served_head_dim_matches_the_fp64_oracle(case, d, dtype):
    """The remaining members of the served set (40 / 56 on the 64-wide tile), both dtypes, three arms."""
    q, k, v, kw, ref = _case_inputs(case, d, dtype)
    out = _run(q, k, v, **kw)
    _assert_small_tile_plan(out, d)
    _check(out, ref)


@requires_rubin
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("d", [24, 64])
@pytest.mark.parametrize("case", ["dense_nkv2", "causal_gqa_8_2", "dense_nkv8_1024"])
def test_split_kv_at_small_d_rides_the_d128_flavor(case, d, dtype):
    """A split_kv = 2 request at D <= 64 -- the pre-tile autotune record {TILE_CGA_M: 2, SPLIT_KV: 2} -- is SERVED on the d128 flavor
    the graph rode before the small tiles (only that flavor admits cga2 with a split at D <= 64; the small tiles carry the dense leg
    only) and matches the oracle and the unsplit small-tile run.  Dense, sink-free, tile-multiple S: the split leg's own envelope
    (padded / synthesized-tail graphs are its typed declines on every flavor)."""
    q, k, v, kw, ref = _case_inputs(case, d, dtype)
    out = _run(q, k, v, split_kv=2, cga=2, **kw)
    assert out["knobs"].split_kv == 2 and out["knobs"].cga == 2, out["knobs"]
    _check(out, ref)
    plain = _run(q, k, v, **kw)
    _assert_small_tile_plan(plain, d)
    torch.testing.assert_close(out["o"].double(), plain["o"].double(), **_TOL_O)


@requires_rubin
@pytest.mark.parametrize("d", [24, 64])
@pytest.mark.parametrize("case", ["dense_nkv2", "causal_tail_385", "padded_kv_zero_batch", "bottom_right_keyless_sink"])
def test_score_outputs_at_small_d_ride_the_d128_flavor(case, d):
    """A graph with the score side outputs at D <= 64 is SERVED on the d128 flavor that stores them (the pre-tile route; TILE_CGA_M =
    2): O / Stats against the oracle, ``score_max`` = the scaled row max of the oracle (the sink folded in), ``Stats == score_max +
    ln(score_sum_exp)`` to fp32 rounding, keyless rows (-inf, 0)."""
    q, k, v, kw, ref = _case_inputs(case, d, torch.bfloat16)
    out = _run(q, k, v, score_outputs=True, **kw)
    assert out["knobs"].cga == 2 and out["knobs"].split_kv == 1, out["knobs"]
    _check(out, ref)
    smax, ssum, lse = (out[n].squeeze(-1) for n in ("score_max", "score_sum_exp", "stats"))
    empty = ref["empty"]
    assert torch.isneginf(smax[empty]).all() and (ssum[empty] == 0).all(), "a keyless row reads (-inf, 0)"
    torch.testing.assert_close(smax[~empty].double(), ref["smax"][~empty], **_TOL_LSE)
    torch.testing.assert_close(lse[~empty], smax[~empty] + torch.log(ssum[~empty]), atol=1e-4, rtol=1e-5)


@requires_rubin
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("d", [24, 32])
def test_cga2_at_d_le_32_lands_on_the_64_wide_tile(d, dtype):
    """A cga2 request at D <= 32 (the persisted record {TILE_CGA_M: 2} of the pre-tile d128 route) is SERVED on the 64-wide tile,
    the smallest covering one validated at cga2 -- the d32 tile is cga1-only -- and is BITWISE the D = 64 cga2 run on the same
    data zero-padded to 64 columns (the zero-fill contract at the second width)."""
    q, k, v, kw, ref = _case_inputs("causal_gqa_8_2", d, dtype)
    out = _run(q, k, v, cga=2, **kw)
    assert out["knobs"].cga == 2, out["knobs"]
    _check(out, ref)
    qp, kp, vp = (torch.nn.functional.pad(t.transpose(1, 2), (0, 64 - d)).transpose(1, 2) for t in (q, k, v))
    wide = _run(qp, kp, vp, cga=2, **kw)
    assert torch.equal(wide["o"][..., :d], out["o"]) and torch.equal(wide["stats"], out["stats"]), "bitwise the zero-padded d64 cga2 run"


@requires_rubin
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("d, width", [(24, 32), (48, 64), (8, 32), (40, 64)])
def test_padded_head_dim_is_bitwise_the_explicitly_zero_padded_tile(d, width, dtype):
    """The zero-fill contract (plan risk 5): a D = 24 graph is BITWISE the D = 32 graph on the same data zero-padded to 32 columns
    -- O's first 24 columns and the Stats -- because the TMA's OOB zero fill and an explicit zero are the same operand (exact zero
    terms in QK^T; the padded V columns' O is never stored at D = 24).  Both runs land on the same tile."""
    q, k, v, kw, _ = _case_inputs("causal_gqa_8_2", d, dtype)
    narrow = _run(q, k, v, **kw)
    pad = width - d
    qp, kp, vp = (torch.nn.functional.pad(t.transpose(1, 2), (0, pad)).transpose(1, 2) for t in (q, k, v))
    wide = _run(qp, kp, vp, **kw)
    assert narrow["knobs"].cga == wide["knobs"].cga == 1
    assert torch.equal(wide["o"][..., :d], narrow["o"]), "O must be bitwise the zero-padded run's first D columns"
    assert (wide["o"][..., d:] == 0).all(), "the padded V columns produce an exactly-zero O"
    assert torch.equal(wide["stats"], narrow["stats"]), "Stats must be bitwise"


@requires_rubin
@pytest.mark.parametrize("case", ["dense_nkv2", "causal_tail_385", "padded_kv_zero_batch", "bottom_right_keyless_sink"])
def test_d64_tile_at_cga2_matches_the_oracle(case):
    """The 64-wide tile's second validated width: the cga2 plan pinned through its knob (the per-CTA V operand is the 64-byte
    MN-major atom the SM100 d64 cga2 build runs)."""
    q, k, v, kw, ref = _case_inputs(case, 64, torch.bfloat16)
    out = _run(q, k, v, cga=2, **kw)
    assert out["knobs"].cga == 2
    _check(out, ref)
    out1 = _run(q, k, v, cga=1, **kw)
    torch.testing.assert_close(out["o"].double(), out1["o"].double(), **_TOL_O)


@requires_rubin
@pytest.mark.parametrize("d", [32, 64])
def test_optional_and_base2_stats_on_the_small_tiles(d):
    """Stats compiled out leaves O bitwise; stats_use_log2 scales the Stats alone."""
    q, k, v, kw, ref = _case_inputs("causal_gqa_8_2", d, torch.bfloat16)
    with_stats = _run(q, k, v, **kw)
    without = _run(q, k, v, stats=False, **kw)
    assert without["stats"] is None and torch.equal(without["o"], with_stats["o"])
    log2 = _run(q, k, v, stats_log2=True, **kw)
    _check(log2, ref, stats_log2=True)
    torch.testing.assert_close(log2["stats"], with_stats["stats"] * math.log2(math.e), atol=1e-4, rtol=1e-5)
