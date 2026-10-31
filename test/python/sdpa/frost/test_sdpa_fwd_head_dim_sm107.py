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

D32 carries dense plans. D64 also carries nonpaged, unsplit THD and PackGQA.
Split-KV and score side outputs use a larger flavor; a cga2 request at D <= 32
uses the 64-wide tile. THD requires exact native dimensions, and paged D64 THD
remains declined.

Three tiers: (1) CPU contract -- the row's accept / decline table per D, the ROUTE table per (D, leg) with the three pick
implementations agreeing on every cell (the row's envelope walk, the adapter's flavor pool, ``sm107_f16_tile_width(leg=, cga=)``),
the one constant feeding the row's cga domain, the adapter's cga twin and the config factory, the factories' derived facts and
raising couplings, the loader binding ``d_flavor`` to the flavor, the static expect_tx pin; (2) host rendering pins -- an sm_107a
trace-compile of each new rendering keeps the register split (``USETMAXREG`` > 0) and spills no more than the d128 rendering of
the same mask arm; (3) GPU cells on a Rubin device -- the fp64 oracle over the degenerate-input matrix (the shared ``small_tile_cells``,
one matrix for both arch lines) on the 32- and 64-wide tiles in bf16 with the fp16 member on the degenerate trio, one accept cell at
D = 16 and at D = 56, the bitwise pin D = 8 / 24 vs an explicitly zero-padded D = 32 (40 / 48 vs 64: the zero-fill contract, one
TMA mechanism), the d64 tile at cga2, the routed legs (split-KV and the score outputs at D = 24 / 64 on the d128 flavor, cga2 at
D = 24 / 32 on the 64-wide tile -- bitwise the zero-padded D = 64 cga2 run), optional and base-2 Stats.  A D = 24 cell that
completes under the runner's timeout IS the expect_tx pin: a byte count short of the full box would hang the consumer on the
first tile.  The d128 geometry of the same file is byte-pinned to its one-geometry past (``test_sdpa_fwd_score_stats.py``), so
it is not re-oracled here.
"""

import ast
import dataclasses
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from frost_test_utils import assert_no_new_spills, requires_dsl, requires_rubin, run_sass_probe, sass_probe_source
from small_tile_cells import CASES, TOL_O, case_inputs, check, check_degenerate_rows, run_graph
from test_sdpa_fwd_decode_d256_sm107 import requires_sm107_dsl

pytestmark = [pytest.mark.L0, requires_dsl]

_ROW = "sdpa_fwd_prefill_sm107"
# D -> the tile width the row serves it on (config_sm107.sm107_f16_tile_width); every member is a GPU cell below (an oracle cell or
# the bitwise zero-fill pin).
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


@requires_sm107_dsl
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


@requires_sm107_dsl
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
    # The d128 flavor carries two tiles behind TILE_CGA_M on cc 10.7: cga2 = the Rubin prefill pipeline, cga1 = the shared SM100
    # decode tile compiled for cc 10.7 (issue #1472); the small tiles do not change that domain.
    assert engines.effective_cgas(caps, _facts(d_qk=128, d_v=128)) == frozenset({1, 2}), "the d128 flavor keeps its two-tile cga domain"
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
    # a cga1 request at D = 128 is SERVED on the d128 flavor since the shared SM100 decode tile was compiled for cc 10.7 (issue #1472)
    (None, dict(), dict(cga=1), dict(cga=1), {8: 32, 24: 32, 32: 32, 48: 64, 64: 64, 128: 128}),
    ("THD", dict(thd=True, padded=True), None, dict(thd=True), {8: None, 24: None, 32: None, 48: None, 64: 64, 128: 128}),
    ("THD", dict(has_paged_kv=True, padded=True, page_size=128, thd=True), None, None, {8: None, 24: None, 32: None, 48: None, 64: None, 128: 128}),
    # Dense PackGQA uses the native D64 tile for logical widths up to 64.
    ("PackGQA", dict(h_kv=2), dict(pack_gqa=True), dict(h_kv=2, pack_gqa=True), {8: 64, 24: 64, 32: 64, 48: 64, 64: 64, 128: 128}),
)
_DECLINE_WORDS = {"THD": "THD", "PackGQA": "PackGQA", None: "cga"}


@requires_sm107_dsl
def test_row_routes_every_leg_beyond_dense_past_the_small_tiles():
    """Graph, adapter and config agree on each tile's native feature domain.

    D64 carries THD and PackGQA; split and score outputs use D128. Combined THD
    and paged requests remain declined, and D32 retains its dense-only domain.
    """
    from cudnn.sdpa.fwd import engines, heuristics
    from cudnn.sdpa.fwd.config_sm107 import sm107_f16_tile_width, sm107_small_tile_leg
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    caps = _caps()
    spec = {s.name: s for s in engines.ENGINE_SPECS}[_ROW]
    assert caps.dense_only_d_shapes == frozenset({(32, 32)})
    assert dict(caps.native_legs_by_d_shape)[(64, 64)] == frozenset({"THD", "PackGQA"})
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
    # a split graph lands on the d128 flavor, whose two tiles both split on cc 10.7 (the shared decode tile at cga1 too)
    assert engines.effective_cgas(caps, f24, 2) == frozenset({1, 2}), "a split graph lands on the d128 flavor"
    assert heuristics._auto_sched_cga(spec, f24, split_kv=2, sched_policy=0) == (0, 2)
    assert heuristics._auto_sched_cga(spec, _facts(d_qk=24, d_v=24, has_score_max=True, has_score_sum_exp=True), split_kv=1, sched_policy=0) == (0, 2)
    # a cga nobody serves is still a typed decline naming the domain of the flavor it would land on
    why = engines.mismatch(caps, f24, SdpaFwdKnobs(cga=3))
    assert why is not None and "cga=3" in why and "[1, 2]" in why, why


@requires_sm107_dsl
@pytest.mark.parametrize("d_qk,d_v", [(8, 8), (24, 16), (32, 32), (48, 32), (64, 64)])
@pytest.mark.parametrize("cga", [None, 1, 2])
def test_partial_pack_checks_the_selected_native_tile(d_qk, d_v, cga):
    """A packed plan may use D64 even when the unpacked graph selects D32."""
    from cudnn.sdpa.fwd import engines

    caps = _caps()
    facts = _facts(d_qk=d_qk, d_v=d_v, h_q=24, h_kv=2)
    knobs = engines.SdpaFwdKnobs(pack_gqa=True, cga=cga)
    assert engines._selected_d_shape(caps, facts, knobs) == (64, 64)
    assert engines.mismatch(caps, facts, knobs) is None


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
    if width == 32:
        with pytest.raises(ValueError, match="per-tensor FP8 D64"):
            make(c.TemplateParams(dtype_qkv=0, cta_mma=1))
    else:
        for dtype in (0, 1):
            for cga in (1, 2):
                quantized, _ = make(c.TemplateParams(dtype_qkv=dtype, dtype_o=2, cta_mma=cga, split_kv=4))
                assert quantized.TILE_K == quantized.TILE_O == quantized.TILE_K_HW_BMM1 == 64
                assert quantized.V_SWZ_BYTES == 64 // cga
                with pytest.raises(ValueError, match="FP8 THD"):
                    make(c.TemplateParams(dtype_qkv=dtype, cta_mma=cga, thd_varlen=True, seq_kv_lens_present=True))
    if width == 32:
        with pytest.raises(ValueError, match="THD/varlen"):
            make(c.TemplateParams(dtype_qkv=2, cta_mma=1, thd_varlen=True, seq_kv_lens_present=True))
    else:
        cfg, _ = make(c.TemplateParams(dtype_qkv=2, cta_mma=1, thd_varlen=True, seq_kv_lens_present=True))
        assert cfg.THD_VARLEN and cfg.TILE_K == cfg.TILE_O == 64
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
    """Static pin of the zero-fill contract (frost-tile-dsl.md s5: an overshooting TMA box delivers the FULL box's bytes): the aggregate boxes cover each tile, every
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
        "HEADS_PER_TILE = CFG.PACK_G if CFG.PACK_GQA else 1",
        "TOKENS_PER_TILE = CFG.TILE_M // HEADS_PER_TILE",
        "qk_box_q = (1, TOKENS_PER_TILE, HEADS_PER_TILE, TMA_QK_GRANU_ELEMS)",
        "qk_box_k = (1, K_BOX_ROWS, 1, TMA_QK_GRANU_ELEMS)",
        "vo_box_v = (1, V_BOX_ROWS, 1, TMA_VO_GRANU_ELEMS)",
        "vo_box_o = (1, TOKENS_PER_TILE, HEADS_PER_TILE, _O_GRANU_ELEMS)",
    ):
        assert line in code, line
    # Evaluate the kernel's box expressions: packing trades token rows for
    # heads without changing the transfer volume or the unpacked geometry.
    names = ("HEADS_PER_TILE", "TOKENS_PER_TILE", "qk_box_q", "vo_box_o")
    expressions = {
        target.id: node.value
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in names
    }
    for packed, pack_g in ((False, 1), (True, 1), (True, 2), (True, 4), (True, 8), (True, 16)):
        for granularity in (16, 32, 64):
            env = dict(CFG=SimpleNamespace(TILE_M=128, PACK_GQA=packed, PACK_G=pack_g), TMA_QK_GRANU_ELEMS=granularity, _O_GRANU_ELEMS=granularity)
            for name in names:
                env[name] = eval(compile(ast.Expression(expressions[name]), str(_KERNEL), "eval"), {"__builtins__": {}}, env)
            for name in ("qk_box_q", "vo_box_o"):
                assert math.prod(env[name]) == 128 * granularity, (packed, pack_g, name, env[name])
                if not packed or pack_g == 1:
                    assert env[name] == (1, 128, 1, granularity)
    # Evaluate the source's page subdivision and byte-count expressions together.
    # Per-page boxes must still deliver the full expect_tx volume, including
    # zero-filled head-dim tails; nonpaged boxes retain the original geometry.
    names = (
        "_K_TILE_ROWS",
        "K_BOX_ROWS",
        "V_BOX_ROWS",
        "K_BOXES",
        "V_BOXES",
        "qk_box_k",
        "vo_box_v",
        "kBufferElems",
        "vBufferElems",
        "kTmaTransactionBytes",
        "vTmaTransactionBytes",
    )
    expressions = {
        target.id: node.value
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in names
    }
    for cga in (1, 2):
        for tile_n in (64, 128):
            for width in (32, 64, 128):
                for page in (0, 8, 16, 32, 64, 128, 256):
                    cfg = SimpleNamespace(TILE_N=tile_n, TILE_K=width, TILE_O=width, CTA_MMA=cga, BPE=2)
                    env = dict(CFG=cfg, PAGED_KV=bool(page), PAGE_SIZE=page, TMA_QK_GRANU_ELEMS=32, TMA_VO_GRANU_ELEMS=16)
                    for name in names:
                        env[name] = eval(compile(ast.Expression(expressions[name]), str(_KERNEL), "eval"), {"__builtins__": {"min": min}}, env)
                    assert math.prod(env["qk_box_k"]) * (width // 32) * env["K_BOXES"] * 2 * cga == env["kTmaTransactionBytes"]
                    assert math.prod(env["vo_box_v"]) * (width // cga // 16) * env["V_BOXES"] * 2 * cga == env["vTmaTransactionBytes"]
                    assert env["K_BOX_ROWS"] * env["K_BOXES"] == tile_n // cga
                    assert env["V_BOX_ROWS"] * env["V_BOXES"] == tile_n
                    if page:
                        assert env["K_BOX_ROWS"] <= page and env["V_BOX_ROWS"] <= page
                    else:
                        assert env["qk_box_k"] == (1, tile_n // cga, 1, 32)
                        assert env["vo_box_v"] == (1, tile_n, 1, 16)
                        assert env["K_BOXES"] == env["V_BOXES"] == 1
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


@requires_sm107_dsl
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
if d == 128 and params.cta_mma == 1:
    # The like-for-like base of the cga1 tiles is the d128 FILE's own cga1 rendering; through the loader, dense d128 at
    # TILE_CGA_M=1 on cc 10.7 is the shared SM100 decode tile (issue #1472), another body -- load the file directly.
    from cudnn.sdpa.fwd.api_dsl import _SM107_KERNEL_FILES, _load_kernel_template
    mod = _load_kernel_template(_SM107_KERNEL_FILES[(d, d)], params, "sdpa_fwd_sm107_f16_d128_cga1_probe")
else:
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

# The degenerate-input matrix (``CASES``), the fp64 oracle, the pinned graph run and the output checks are the shared
# ``small_tile_cells`` -- one matrix for both arch lines (``test_sdpa_small_d_sm100.py`` runs the same cases on its 32-wide tile).
_DT_IDS = {torch.float16: "fp16", torch.bfloat16: "bf16"}


def _run(q, k, v, **kw):
    return run_graph(q, k, v, row=_ROW, **kw)


def _assert_small_tile_plan(out, d):
    """The route evidence: the plan's TILE_CGA_M knob is 1 on the small tiles (D <= 64); the d128 flavor runs either of its two tiles
    (cga2 = the Rubin prefill pipeline, cga1 = the shared decode tile compiled for cc 10.7, the heuristics' pick per shape)."""
    assert out["knobs"].cga == 1 if d <= 64 else out["knobs"].cga in (1, 2), (d, out["knobs"])
    assert out["knobs"].split_kv == 1, (d, out["knobs"])


# One oracle per served tile over the whole matrix in bf16; the fp16 member on the cells where the dtype enters the io loads and the
# P / O casts of a degenerate row (the dense base, a zero-length KV batch, keyless bottom-right rows with a sink and the padded-Q
# trim) on both tiles; D 16 as the one served width of the 32-wide tile that neither the grid nor the bitwise zero-fill pin below names
# (D 8 / 24 / 40 / 48 ARE that pin: one TMA zero fill, one mechanism).
_FP16_CASES = ("dense_nkv2", "padded_kv_zero_batch", "bottom_right_keyless_sink")
_ORACLE_CELLS = (
    [(case, d, torch.bfloat16) for d in (32, 64) for case in CASES]
    + [(case, d, torch.float16) for d in (32, 64) for case in _FP16_CASES]
    + [("dense_nkv2", 16, torch.bfloat16)]
)


@requires_rubin
@pytest.mark.parametrize("case, d, dtype", _ORACLE_CELLS, ids=[f"{c}-{d}-{_DT_IDS[t]}" for c, d, t in _ORACLE_CELLS])
def test_small_tiles_match_the_fp64_oracle(case, d, dtype):
    """The two small geometries of the file -- the 32-wide tile at D 32 (and 16), the 64-wide tile at D 64 -- over the degenerate-input
    matrix: every case in bf16, the degenerate trio in fp16 too."""
    q, k, v, kw, ref = case_inputs(case, d, dtype)
    out = _run(q, k, v, **kw)
    _assert_small_tile_plan(out, d)
    check(out, ref)
    check_degenerate_rows(out, case)


@requires_rubin
def test_every_other_served_head_dim_matches_the_fp64_oracle():
    """D 56: the one served width of the 64-wide tile that neither the grid above (32 / 64) nor the bitwise zero-fill pin below (40 /
    48) names -- one accept cell, the only one that would catch a TMA box / stride fault specific to that width."""
    q, k, v, kw, ref = case_inputs("dense_nkv2", 56, torch.bfloat16)
    out = _run(q, k, v, **kw)
    _assert_small_tile_plan(out, 56)
    check(out, ref)


_SPLIT_CELLS = [("dense_nkv2", 24), ("dense_nkv8_1024", 64)]


@requires_rubin
@pytest.mark.parametrize("case, d", _SPLIT_CELLS, ids=[f"{c}-{d}" for c, d in _SPLIT_CELLS])
def test_split_kv_at_small_d_rides_the_d128_flavor(case, d):
    """A split_kv = 2 request at D <= 64 -- the pre-tile autotune record {TILE_CGA_M: 2, SPLIT_KV: 2} -- is SERVED on the d128 flavor
    the graph rode before the small tiles (only that flavor admits cga2 with a split at D <= 64; the small tiles carry the dense leg
    only) and matches the oracle and the unsplit small-tile run.  Dense, sink-free, tile-multiple S: the split leg's own envelope
    (padded / synthesized-tail graphs are its typed declines on every flavor).  The routing is the claim; the split body's own oracle
    is the pre-existing d128 suite's."""
    q, k, v, kw, ref = case_inputs(case, d, torch.bfloat16)
    out = _run(q, k, v, split_kv=2, cga=2, **kw)
    assert out["knobs"].split_kv == 2 and out["knobs"].cga == 2, out["knobs"]
    check(out, ref)
    plain = _run(q, k, v, **kw)
    _assert_small_tile_plan(plain, d)
    torch.testing.assert_close(out["o"].double(), plain["o"].double(), **TOL_O)


_SCORE_CELLS = [("dense_nkv2", 24), ("bottom_right_keyless_sink", 64)]


@requires_rubin
@pytest.mark.parametrize("case, d", _SCORE_CELLS, ids=[f"{c}-{d}" for c, d in _SCORE_CELLS])
def test_score_outputs_at_small_d_ride_the_d128_flavor(case, d):
    """A graph with the score side outputs at D <= 64 is SERVED on the d128 flavor that stores them (the pre-tile route; TILE_CGA_M =
    2): O / Stats against the oracle, ``score_max`` = the scaled row max of the oracle (the sink folded in), ``Stats == score_max +
    ln(score_sum_exp)`` to fp32 rounding, keyless rows (-inf, 0).  The score arm's own oracle is ``test_sdpa_fwd_score_stats.py``."""
    q, k, v, kw, ref = case_inputs(case, d, torch.bfloat16)
    out = _run(q, k, v, score_outputs=True, **kw)
    assert out["knobs"].cga == 2 and out["knobs"].split_kv == 1, out["knobs"]
    check(out, ref)
    smax, ssum, lse = (out[n].squeeze(-1) for n in ("score_max", "score_sum_exp", "stats"))
    empty = ref["empty"]
    assert torch.isneginf(smax[empty]).all() and (ssum[empty] == 0).all(), "a keyless row reads (-inf, 0)"
    torch.testing.assert_close(smax[~empty].double(), ref["smax"][~empty], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(lse[~empty], smax[~empty] + torch.log(ssum[~empty]), atol=1e-4, rtol=1e-5)


@requires_rubin
@pytest.mark.parametrize("d", [24, 32])
def test_cga2_at_d_le_32_lands_on_the_64_wide_tile(d):
    """A cga2 request at D <= 32 (the persisted record {TILE_CGA_M: 2} of the pre-tile d128 route) is SERVED on the 64-wide tile,
    the smallest covering one validated at cga2 -- the d32 tile is cga1-only -- and is BITWISE the D = 64 cga2 run on the same
    data zero-padded to 64 columns (the zero-fill contract at the second width; the routing does not read the dtype)."""
    q, k, v, kw, ref = case_inputs("causal_gqa_8_2", d, torch.bfloat16)
    out = _run(q, k, v, cga=2, **kw)
    assert out["knobs"].cga == 2, out["knobs"]
    check(out, ref)
    qp, kp, vp = (torch.nn.functional.pad(t.transpose(1, 2), (0, 64 - d)).transpose(1, 2) for t in (q, k, v))
    wide = _run(qp, kp, vp, cga=2, **kw)
    assert torch.equal(wide["o"][..., :d], out["o"]) and torch.equal(wide["stats"], out["stats"]), "bitwise the zero-padded d64 cga2 run"


@requires_rubin
@pytest.mark.parametrize("d, width", [(24, 32), (48, 64), (8, 32), (40, 64)], ids=["24-32", "48-64", "8-32", "40-64"])
def test_padded_head_dim_is_bitwise_the_explicitly_zero_padded_tile(d, width):
    """The zero-fill contract (plan risk 5): a D = 24 graph is BITWISE the D = 32 graph on the same data zero-padded to 32 columns
    -- O's first 24 columns and the Stats -- because the TMA's OOB zero fill and an explicit zero are the same operand (exact zero
    terms in QK^T; the padded V columns' O is never stored at D = 24).  Both runs land on the same tile; the TMA zero fill is
    dtype-blind, so bf16 carries the pin."""
    q, k, v, kw, _ = case_inputs("causal_gqa_8_2", d, torch.bfloat16)
    narrow = _run(q, k, v, **kw)
    pad = width - d
    qp, kp, vp = (torch.nn.functional.pad(t.transpose(1, 2), (0, pad)).transpose(1, 2) for t in (q, k, v))
    wide = _run(qp, kp, vp, **kw)
    assert narrow["knobs"].cga == wide["knobs"].cga == 1
    assert torch.equal(wide["o"][..., :d], narrow["o"]), "O must be bitwise the zero-padded run's first D columns"
    assert (wide["o"][..., d:] == 0).all(), "the padded V columns produce an exactly-zero O"
    assert torch.equal(wide["stats"], narrow["stats"]), "Stats must be bitwise"


@requires_rubin
@pytest.mark.parametrize("case", ["causal_tail_385", "bottom_right_keyless_sink"])
def test_d64_tile_at_cga2_matches_the_oracle(case):
    """The 64-wide tile's second validated width: the cga2 plan pinned through its knob (the per-CTA V operand is the 64-byte
    MN-major atom the SM100 d64 cga2 build runs) on the causal tail and on keyless bottom-right rows with a sink -- the dense cga2
    tile runs under ``test_cga2_at_d_le_32_lands_on_the_64_wide_tile``."""
    q, k, v, kw, ref = case_inputs(case, 64, torch.bfloat16)
    out = _run(q, k, v, cga=2, **kw)
    assert out["knobs"].cga == 2
    check(out, ref)
    out1 = _run(q, k, v, cga=1, **kw)
    torch.testing.assert_close(out["o"].double(), out1["o"].double(), **TOL_O)


@requires_rubin
@pytest.mark.parametrize("d", [32, 64])
def test_optional_and_base2_stats_on_the_small_tiles(d):
    """Stats compiled out leaves O bitwise; stats_use_log2 scales the Stats alone."""
    q, k, v, kw, ref = case_inputs("causal_gqa_8_2", d, torch.bfloat16)
    with_stats = _run(q, k, v, **kw)
    without = _run(q, k, v, stats=False, **kw)
    assert without["stats"] is None and torch.equal(without["o"], with_stats["o"])
    log2 = _run(q, k, v, stats_log2=True, **kw)
    check(log2, ref, stats_log2=True)
    torch.testing.assert_close(log2["stats"], with_stats["stats"] * math.log2(math.e), atol=1e-4, rtol=1e-5)
