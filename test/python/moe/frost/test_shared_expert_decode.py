# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared-expert decode kernel pair (``cudnn.moe.frost.shared_expert_decode``): the config's structural invariants and derived
constants (the gate-logit partition, the SMEM tables, the ring ledger, the barrier constants) on the host, the typed declines of the host
entry (CPU tensors), the plan-time-only compile key, the lazy export and the docs rows; on a cc 10.x device the pair against an fp64
reference of the bf16 operands at the harness band, the padded-row hygiene, the CUDA-graph replay, and every knob arm bitwise with the
default (the PDL arms, the group count, the ring, the W phases, the launch-2 row count), the K_SPLIT / gate-fold / v1 arms within the band
(rounding-order axes).
The operand builders, the fp64 reference, the band check, the order model and the device gate live in ``shared_expert_utils.py`` beside
this module (shared with ``test_shared_expert_decode_determinism.py``)."""

from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from shared_expert_utils import (  # noqa: E402  (the operand builders, the references and the device gate shared with the determinism module)
    GEOMETRIES,
    SM100_READING,
    _device_or_skip,
    _gid,
    assert_in_band,
    make_operands,
    order_model,
    reference,
)

from cudnn.moe.frost import shared_expert_decode as entry  # noqa: E402
from cudnn.moe.frost.kernel import shared_expert_decode_config as kcfg  # noqa: E402

pytestmark = [pytest.mark.L0]

REPO = Path(__file__).resolve().parents[4]


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the config
# ---------------------------------------------------------------------------------------------------------------------------
def test_default_cfg_pins_k_split_per_geometry_at_one_wave():
    """K_SPLIT is a function of the geometry alone (the slice order of every output element never moves with T): 8 at both validated
    geometries -- clusters of eight, 256 CTAs either way: two row-pair groups per CTA at I 256 (32 KB of W each, two CTAs per SM at every
    T: the measured best form), one at I 128 (16 KB, four per SM at one tile; the two-group grid of 128 would leave SMs idle); one wave on a
    148- or 152-SM part; every knob default is the measured one (bulk x rows, the cluster-side gate fold, PDL off, W_PHASES 1, 64 rows)."""
    for g, n, ctas in ((GEOMETRIES[0], 2, 2), (GEOMETRIES[1], 1, 4)):
        cfg = kcfg.default_cfg(g)
        assert (cfg.K_SPLIT, cfg.GRID1, cfg.GRID2, cfg.THREADS, cfg.THREADS2, cfg.N_GROUPS_EFF, cfg.co_resident_ctas1, cfg.WAVES1) == (
            8,
            256,
            128,
            256,
            256,
            n,
            ctas,
            1,
        ), g
        assert (cfg.N_GROUPS, cfg.X_BULK_ROWS, cfg.GATE_CLUSTER_FOLD, cfg.SM_COUNT) == (0, True, True, kcfg.SM100_SM_COUNT)
        assert (cfg.PDL, cfg.PDL_PAIR, cfg.PDL_RELEASE, cfg.X_STAGES, cfg.W_PHASES, cfg.ROWS_PER_CTA, cfg.MIN_CTAS_PER_SM) == (False, False, False, 0, 1, 64, 1)
        two = kcfg.form_for(cfg, 24)
        assert (two.TOKEN_TILES, two.N_GROUPS_EFF, two.GRID1, two.co_resident_ctas1, two.WAVES1) == (2, n, 256, 2, 1), g
        for sm in (148, 152, 212):  # the same forms on every SM100-class part
            part = kcfg.default_cfg(g, sm_count=sm)
            assert (part.N_GROUPS_EFF, kcfg.form_for(part, 24).N_GROUPS_EFF) == (n, n)
    assert kcfg.CFG == kcfg.default_cfg() and kcfg.CFG_TP16 == kcfg.default_cfg(GEOMETRIES[1])
    assert kcfg.k_split_for(kcfg.Geometry(8192, 256)) == 8 and kcfg.k_split_for(kcfg.Geometry(8192, 128)) == 8
    # the v1 arm (the class of the committed ksplit2 / ksplit4 goldens): 128 CTAs, the ring, the launch-2 gate fold
    for g, s in ((GEOMETRIES[0], 2), (GEOMETRIES[1], 4)):
        v1 = kcfg.v1_cfg(g)
        assert (v1.K_SPLIT, v1.GRID1, v1.N_GROUPS_EFF, v1.X_BULK_ROWS, v1.GATE_CLUSTER_FOLD_EFF, v1.GATE_SLOTS) == (s, 128, 1, False, False, v1.GATE_PARTS)
    assert kcfg.CFG_V1 == kcfg.v1_cfg() and kcfg.CFG_V1_TP16 == kcfg.v1_cfg(GEOMETRIES[1])


def _cfg(g, k_split, tiles, **kw):
    """A form of (geometry, K_SPLIT, TOKEN_TILES): the bulk x rows where the slice fits (K_SPLIT >= 4; K_SPLIT 2 at one tile), else the ring."""
    bulk = kw.pop("X_BULK_ROWS", k_split >= 4 or (k_split == 2 and tiles == 1))
    return kcfg.make_cfg(GEOM=g, K_SPLIT=k_split, TOKEN_TILES=tiles, X_BULK_ROWS=bulk, **kw)


def test_geometry_validates_and_holds_plain_ints():
    g = kcfg.Geometry(8192, 256)
    assert isinstance(g.HIDDEN, int) and g == kcfg.Geometry(8192, 256) and "8192" in g.describe()
    for bad in ((0, 256), (8192, 0), (8192, -1), (8192, 2.5), (True, 256)):
        with pytest.raises(ValueError, match="Geometry"):
            kcfg.Geometry(*bad)


@pytest.mark.parametrize("fold", (True, False), ids=("cluster-fold", "launch2-fold"))
@pytest.mark.parametrize("g", GEOMETRIES, ids=_gid)
@pytest.mark.parametrize("k_split", (1, 2, 4, 8))
@pytest.mark.parametrize("tiles", (1, 2))
def test_gate_partition_tiles_hidden_for_every_cfg(g, k_split, tiles, fold):
    """The audit's host pin: GATE_PARTS x GATE_COLS == H for every (geometry, K_SPLIT) the knobs produce (the leaves independent of N_GROUPS),
    the fold's per-lane counts integral, GATE_SLOTS = GROUPS under the cluster fold (K_SPLIT > 1) else every leaf, and the gate warps' pair
    coverage exactly the CTA's columns (every column of H owned once across the grid)."""
    cfg = _cfg(g, k_split, tiles, GATE_CLUSTER_FOLD=fold)
    assert cfg.GATE_PARTS * cfg.GATE_COLS == g.HIDDEN and cfg.GATE_COLS % 2 == 0 and cfg.GATE_SLOTS % 32 == 0
    assert cfg.GATE_PARTS == cfg.K_SPLIT * cfg.GROUPS == cfg.GRID1 * cfg.N_GROUPS_EFF and cfg.GATE_SLOTS_PER_LANE * 32 == cfg.GATE_SLOTS
    assert cfg.GATE_SLOTS == (cfg.GROUPS if fold and k_split > 1 else cfg.GATE_PARTS) and cfg.GATE_CLUSTER_FOLD_EFF == (fold and k_split > 1)
    owned = [0] * g.HIDDEN
    for q in range(cfg.GATE_PARTS):
        ks, grp = divmod(q, cfg.GROUPS)
        k0 = ks * cfg.K_SLICE + grp * cfg.GATE_COLS
        for lane in range(32):
            for p in range(cfg.GATE_PAIRS_PER_LANE):
                col = 2 * lane + 64 * p
                if col < cfg.GATE_COLS:
                    owned[k0 + col] += 1
                    owned[k0 + col + 1] += 1
    assert all(v == 1 for v in owned), "every hidden column is summed exactly once across the grid"
    for n in (1, 2) if k_split >= 4 else (1,):  # N_GROUPS moves the grid, never the leaves (two groups of a 4096-column slice do not fit)
        assert kcfg.make_cfg(GEOM=g, K_SPLIT=k_split, TOKEN_TILES=tiles, N_GROUPS=n, X_BULK_ROWS=cfg.X_BULK_ROWS).GATE_PARTS == cfg.GATE_PARTS


@pytest.mark.parametrize("bulk", (True, False), ids=("bulk-rows", "ring"))
@pytest.mark.parametrize("g", GEOMETRIES, ids=_gid)
@pytest.mark.parametrize("k_split", (1, 2, 4, 8))
@pytest.mark.parametrize("tiles", (1, 2))
def test_smem_tables_fit_and_every_pitch_covers_the_banks(g, k_split, tiles, bulk):
    """Every row pitch is 128 n + 16 B (the eight row addresses of one ldmatrix phase cover 32 banks); launch 1's total fits the opt-in at
    every form, launch 2's too; a ring form has at least two stages and the budget rule keeps every chunk resident where it fits; a bulk
    form that does not fit (K_SPLIT 1; K_SPLIT 2 at two tiles) is the typed decline naming the ring."""
    if bulk and (k_split == 1 or (k_split == 2 and tiles == 2)):
        with pytest.raises(ValueError, match="x resident.*ring form"):
            kcfg.make_cfg(GEOM=g, K_SPLIT=k_split, TOKEN_TILES=tiles, X_BULK_ROWS=True)
        return
    cfg = kcfg.make_cfg(GEOM=g, K_SPLIT=k_split, TOKEN_TILES=tiles, X_BULK_ROWS=bulk)
    for name, (nbytes, pitch) in {**kcfg.smem_table1(cfg), **kcfg.smem_table2(cfg)}.items():
        assert nbytes >= 0, name
        if pitch is not None and pitch > 16:
            assert pitch % 128 == 16, (name, pitch)
            banks = {((r * pitch) // 4 + b) % 32 for r in range(8) for b in range(4)}
            assert len(banks) == 32, (name, pitch)
    assert sum(b for b, _ in kcfg.smem_table1(cfg).values()) == cfg.SMEM1_BYTES <= cfg.SMEM_OPTIN_BYTES
    assert sum(b for b, _ in kcfg.smem_table2(cfg).values()) == cfg.SMEM2_BYTES <= cfg.SMEM_OPTIN_BYTES
    assert cfg.WAVES1 >= 1 and cfg.co_resident_ctas1 >= 1
    if bulk:
        assert cfg.X_STAGES_EFF == 0 and cfg.X_SMEM_BYTES == cfg.X_ROWS * (cfg.K_SLICE + 8) * 2 and cfg.X_ROWS == (16 if tiles == 1 else 24)
    else:
        assert 2 <= cfg.X_STAGES_EFF <= cfg.X_CHUNKS
        if cfg.SMEM1_FIXED_BYTES + cfg.X_CHUNKS * cfg.X_CHUNK_BYTES + kcfg.SMEM_RING_MARGIN_BYTES <= cfg.SMEM1_BUDGET_BYTES:
            assert cfg.X_STAGES_EFF == cfg.X_CHUNKS, "the whole slice is resident when it fits"


def test_default_form_figures_of_record():
    """The default forms' figures of record (the kernel module's SMEM table quotes these): I 256 / K_SPLIT 8 / N_GROUPS 2: 81,328 B at one tile
    (w_s 33,024, x_s 16 x 2,064, red_s / mail_s 7,168, gmail_s 896, 48 B of mbarriers; two CTAs per SM) and 113,072 B at two tiles; the
    one-group one-tile form (the N_GROUPS 1 arm, the I 128 default) 57,200 B, four per SM with 144 B of slack; I 128: 57,200 / 81,328 B; the
    v1 arm unchanged at 202,912 / 207,008 B (I 256) and 104,608 / 176,288 B (I 128)."""
    c1, c2 = kcfg.form_for(kcfg.CFG, 6), kcfg.form_for(kcfg.CFG, 24)
    assert kcfg.smem_table1(c1) == {
        "w_s": (33024, 2064),
        "x_s": (33024, 2064),
        "red_s": (7168, 16),
        "mail_s": (7168, 16),
        "gmail_s": (896, 4),
        "mbarriers": (48, None),
    }
    assert kcfg.smem_table1(c2) == {
        "w_s": (33024, 2064),
        "x_s": (49536, 2064),
        "red_s": (14336, 16),
        "mail_s": (14336, 16),
        "gmail_s": (1792, 4),
        "mbarriers": (48, None),
    }
    assert (c1.SMEM1_BYTES, c2.SMEM1_BYTES, c1.SMEM2_BYTES, c2.SMEM2_BYTES) == (81328, 113072, 42368, 50816)
    assert (c1.co_resident_ctas1, c2.co_resident_ctas1, c1.KRED_BYTES, c2.KRED_BYTES) == (2, 2, 8064, 16128)
    n1 = kcfg.make_cfg(N_GROUPS=1)
    assert kcfg.smem_table1(n1) == {
        "w_s": (16512, 2064),
        "x_s": (33024, 2064),
        "red_s": (3584, 16),
        "mail_s": (3584, 16),
        "gmail_s": (448, 4),
        "mbarriers": (48, None),
    }
    assert (n1.SMEM1_BYTES, n1.co_resident_ctas1, n1.GRID1, n1.KRED_BYTES) == (57200, 4, 512, 4032)
    assert kcfg.SM100_SMEM_PER_SM_BYTES // 4 - kcfg.SMEM_CTA_RESERVE_BYTES - n1.SMEM1_BYTES == 144, "the one-group one-tile form's slack under four per SM"
    t1, t2 = kcfg.form_for(kcfg.CFG_TP16, 6), kcfg.form_for(kcfg.CFG_TP16, 24)
    assert (t1.SMEM1_BYTES, t2.SMEM1_BYTES, t1.SMEM2_BYTES, t2.SMEM2_BYTES, t1.co_resident_ctas1, t2.co_resident_ctas1) == (57200, 81328, 21888, 26240, 4, 2)
    v1, v2 = kcfg.form_for(kcfg.CFG_V1, 6), kcfg.form_for(kcfg.CFG_V1, 24)
    assert (v1.W_SMEM_BYTES, v1.X_STAGES_EFF, v1.X_CHUNK_BYTES, v1.X_SMEM_BYTES) == (65664, 8, 16640, 133120)
    assert (v2.X_STAGES_EFF, v2.X_CHUNK_BYTES, v2.X_SMEM_BYTES) == (4, 33280, 133120)
    assert (v1.SMEM1_BYTES, v2.SMEM1_BYTES, v1.co_resident_ctas1, v1.GMAIL_SMEM_BYTES) == (202912, 207008, 1, 0)
    u1 = kcfg.CFG_V1_TP16
    assert u1.K_SPLIT == 4 and u1.X_CHUNKS == 4 and u1.X_STAGES_EFF == 4 and u1.co_resident_ctas1 == 2 and u1.SMEM1_BYTES == 104608
    assert kcfg.form_for(u1, 24).SMEM1_BYTES == 176288
    big = kcfg.make_cfg(K_SPLIT=4, MIN_CTAS_PER_SM=2, X_BULK_ROWS=False)  # a ring arm that must co-reside twice: the budget rule halves the SM
    assert big.co_resident_ctas1 == 2 and big.SMEM1_BUDGET_BYTES == kcfg.SM100_SMEM_PER_SM_BYTES // 2 - kcfg.SMEM_CTA_RESERVE_BYTES


def test_one_wave_rule_is_the_form_axis_and_a_two_wave_default_is_the_typed_error():
    """N_GROUPS 0 picks the smallest group count that fits one wave; an explicit N_GROUPS is kept (a knob arm may choose two waves); a part
    too small for one wave under the rule declines by name."""
    assert kcfg.make_cfg(TOKEN_TILES=2).N_GROUPS_EFF == 2 and kcfg.make_cfg(TOKEN_TILES=2, N_GROUPS=1).WAVES1 == 2
    assert kcfg.make_cfg(GEOM=GEOMETRIES[1], TOKEN_TILES=2).N_GROUPS_EFF == 1 and kcfg.make_cfg(GEOM=GEOMETRIES[1], TOKEN_TILES=2, N_GROUPS=2).GRID1 == 128
    assert kcfg.make_cfg(N_GROUPS=1).WAVES1 == 1, "the one-group one-tile form fits one wave too; the rule prefers the fatter two-group CTAs"
    small = kcfg.make_cfg(TOKEN_TILES=2, SM_COUNT=64)
    assert small.N_GROUPS_EFF == 2 and small.WAVES1 == 2
    with pytest.raises(ValueError, match="needs 2 waves"):
        kcfg.form_for(kcfg.make_cfg(SM_COUNT=64), 24)
    with pytest.raises(ValueError, match="needs 2 waves"):
        kcfg.default_cfg(sm_count=64)


def test_barrier_ledger_constants_and_the_ring_walk():
    """The barrier-table constants with their derivation (the W, x and kred ledgers of every form), and the pure-Python ring ledger for every
    (X_CHUNKS, X_STAGES) the ring cfgs produce: chunk i has landed before iteration i, and a stage is refilled only by the iteration after its
    last read (behind that iteration's barrier)."""
    assert kcfg.W_MBAR_ARRIVERS == 1 and kcfg.X_MBAR_ARRIVERS == 1 and kcfg.KRED_MBAR_ARRIVERS == 1
    for g in GEOMETRIES:
        for s in (1, 2, 4, 8):
            for tiles in (1, 2):
                for phases in (1, 2):
                    for bulk in (True, False):
                        if bulk and (s == 1 or (s == 2 and tiles == 2)):
                            continue
                        cfg = kcfg.make_cfg(GEOM=g, K_SPLIT=s, TOKEN_TILES=tiles, W_PHASES=phases, X_BULK_ROWS=bulk)
                        n = cfg.N_GROUPS_EFF
                        assert cfg.W_ROWS == 8 * n <= 32 and cfg.W_PHASE_BYTES * cfg.W_PHASES == cfg.W_SLICE_BYTES == cfg.W_ROWS * cfg.K_SLICE * 2
                        assert cfg.W_PHASE_BYTES == cfg.W_ROWS * cfg.W_PHASE_ROW_BYTES and cfg.W_PHASE_ROW_BYTES % 16 == 0
                        assert cfg.ACC_TILES == n * tiles and cfg.MAIL_SMEM_BYTES == (s - 1) * cfg.ACC_TILES * 32 * 16
                        assert cfg.GMAIL_SMEM_BYTES == ((s - 1) * n * cfg.TPAD * 4 if s > 1 else 0)
                        assert cfg.KRED_BYTES == cfg.MAIL_SMEM_BYTES + cfg.GMAIL_SMEM_BYTES
                        assert cfg.WD_COPIES * cfg.THREADS2 * 16 == cfg.ROWS_PER_CTA * cfg.I * 2
                        assert cfg.ACT_COPIES * cfg.THREADS2 * 16 == cfg.TPAD * cfg.I * 2
                        if bulk:
                            assert cfg.X_ROWS <= 32 and cfg.X_ROW_BYTES == cfg.K_SLICE * 2 and cfg.X_ROW_BYTES % 16 == 0
                            continue
                        assert cfg.X_COPIES * cfg.THREADS * 16 == cfg.TPAD * kcfg.CHUNK_K * 2
                        assert cfg.X_PROLOGUE_GROUPS == cfg.X_STAGES_EFF - 1 and cfg.X_WAIT_PENDING == cfg.X_STAGES_EFF - 2
                        for stages in {cfg.X_STAGES_EFF, 2, cfg.X_CHUNKS}:
                            landed, refilled = kcfg.ring_ledger(cfg.X_CHUNKS, stages)
                            for i in range(cfg.X_CHUNKS):
                                assert landed[i] >= i + 1, (cfg.X_CHUNKS, stages, i)
                                if refilled[i] >= 0:
                                    assert refilled[i] == (i - 1) % stages, (cfg.X_CHUNKS, stages, i)
    with pytest.raises(ValueError, match="2 <= X_STAGES"):
        kcfg.ring_ledger(8, 1)


@pytest.mark.parametrize(
    "overrides, token",
    [
        (dict(THREADS=128), "THREADS"),
        (dict(K_SPLIT=3), "K_SPLIT"),
        (dict(K_SPLIT=16), "K_SPLIT"),
        (dict(TOKEN_TILES=3), "TOKEN_TILES"),
        (dict(N_GROUPS=3), "N_GROUPS"),
        (dict(W_PHASES=4), "W_PHASES"),
        (dict(ROWS_PER_CTA=16), "ROWS_PER_CTA"),
        (dict(MIN_CTAS_PER_SM=3), "MIN_CTAS_PER_SM"),
        (dict(SM_COUNT=0), "SM_COUNT"),
        (dict(X_BULK_ROWS=False, K_SPLIT=2, X_STAGES=1), "X_STAGES"),
        (dict(X_BULK_ROWS=False, K_SPLIT=2, X_STAGES=9), "X_STAGES"),
        (dict(K_SPLIT=1, N_GROUPS=1), "x resident"),
        (dict(K_SPLIT=2, TOKEN_TILES=2), "x resident"),
        (
            dict(
                K_SPLIT=4,
                MIN_CTAS_PER_SM=1,
                X_STAGES=4,
                X_BULK_ROWS=False,
                GEOM=kcfg.Geometry(8192, 256),
                SMEM_OPTIN_BYTES=100 * 1024,
                SMEM_PER_SM_BYTES=101 * 1024,
            ),
            "exceeds the per-CTA opt-in",
        ),
        (dict(X_BULK_ROWS=False, K_SPLIT=2, SMEM_OPTIN_BYTES=100 * 1024, SMEM_PER_SM_BYTES=101 * 1024), "X_STAGES"),
        (dict(SMEM_OPTIN_BYTES=300 * 1024), "SMEM_OPTIN_BYTES"),
        (dict(GEOM=kcfg.Geometry(8000, 256)), "HIDDEN"),
        (dict(GEOM=kcfg.Geometry(8192, 200)), "INTERMEDIATE"),
        (dict(GEOM=kcfg.Geometry(8192, 64)), "do not split"),
        (dict(GEOM=kcfg.Geometry(8192, 64), ROWS_PER_CTA=32, K_SPLIT=1, X_BULK_ROWS=False), "GATE_SLOTS"),
        (dict(GEOM="geometry"), "GEOM"),
    ],
)
def test_cfg_knobs_validate_by_name(overrides, token):
    with pytest.raises(ValueError, match=token):
        kcfg.make_cfg(**overrides)


def test_token_tiles_form_and_the_24_ceiling():
    assert [kcfg.token_tiles_for(t) for t in (1, 6, 16, 17, 24)] == [1, 1, 1, 2, 2]
    with pytest.raises(NotImplementedError, match="24"):
        kcfg.token_tiles_for(25)
    with pytest.raises(ValueError):
        kcfg.token_tiles_for(0)
    assert kcfg.form_for(kcfg.CFG, 6) is kcfg.CFG and kcfg.form_for(kcfg.CFG, 24).TOKEN_TILES == 2 and kcfg.form_for(kcfg.CFG, 24).N_GROUPS_EFF == 2


def test_part_readings_validate():
    assert kcfg.part_readings(*SM100_READING) == dict(SMEM_PER_SM_BYTES=SM100_READING[0], SMEM_OPTIN_BYTES=SM100_READING[1], SM_COUNT=148)
    assert kcfg.part_readings(*SM100_READING, 152)["SM_COUNT"] == 152
    with pytest.raises(ValueError, match="cannot exceed"):
        kcfg.part_readings(100, 200)
    with pytest.raises(ValueError, match="positive int"):
        kcfg.part_readings(0, 0)
    with pytest.raises(ValueError, match="sm_count"):
        kcfg.part_readings(*SM100_READING, True)


# ---------------------------------------------------------------------------------------------------------------------------
# host-only: the entry
# ---------------------------------------------------------------------------------------------------------------------------
def call_check(ops, device_cc=(10, 3), **kw):
    return entry.check_support(ops["x"], ops["w_gate_up"], ops["w_down"], ops["w_gate"], kw.pop("out", None), device_cc=device_cc, **kw)


def test_served_operands_pass_the_host_checks_at_both_geometries():
    for g in GEOMETRIES:
        for t in (1, 16, 17, 24):
            assert call_check(make_operands(t, geometry=g), geometry=g) == t
    ops = make_operands(4, x_rows=8)  # a row-strided x view inside a taller slab
    assert call_check(ops) == 4
    assert call_check(make_operands(3), out=torch.empty(5, 8192, dtype=torch.bfloat16)) == 3
    ops = make_operands(2)
    assert call_check(dict(ops, w_gate=ops["w_gate"].view(1, -1))) == 2


@pytest.mark.parametrize(
    "mutation, exc, token",
    [
        (lambda o: dict(o, x=o["x"].float()), TypeError, "x must be torch.bfloat16"),
        (lambda o: dict(o, w_down="w"), TypeError, "w_down must be a torch.Tensor"),
        (lambda o: dict(o, x=torch.zeros(25, 8192, dtype=torch.bfloat16)), NotImplementedError, "at most 24 tokens"),
        (lambda o: dict(o, x=torch.zeros(0, 8192, dtype=torch.bfloat16)), ValueError, "at least one token"),
        (lambda o: dict(o, x=torch.zeros(4, 4096, dtype=torch.bfloat16)), ValueError, r"x must be \[tokens, 8192\]"),
        (lambda o: dict(o, x=torch.zeros(4, 8196, dtype=torch.bfloat16)[:, :8192]), ValueError, "16-byte-multiple row stride"),
        (lambda o: dict(o, x=torch.zeros(8192, 4, dtype=torch.bfloat16).t()), ValueError, "x rows must be contiguous"),
        (lambda o: dict(o, w_gate_up=o["w_gate_up"].t()), ValueError, "w_gate_up must be a contiguous"),
        (lambda o: dict(o, w_gate_up=o["w_gate_up"][:256]), ValueError, r"w_gate_up must be a contiguous \[512, 8192\]"),
        (lambda o: dict(o, w_down=o["w_down"].t().contiguous()), ValueError, r"w_down must be a contiguous \[8192, 256\]"),
        (lambda o: dict(o, w_gate=o["w_gate"][:4096]), ValueError, "w_gate must be a contiguous"),
        (lambda o: dict(o, w_gate=o["w_gate"].view(2, 4096)), ValueError, "w_gate must be a contiguous"),
        (lambda o: dict(o, out=torch.empty(2, 8192, dtype=torch.bfloat16)), ValueError, r"out must be \[>= 4, 8192\]"),
        (lambda o: dict(o, out=torch.empty(4, 8192, dtype=torch.float32)), TypeError, "out must be torch.bfloat16"),
        (lambda o: dict(o, out=torch.empty(4, 8196, dtype=torch.bfloat16)[:, :8192]), ValueError, "out rows must be contiguous"),
        (lambda o: dict(o, x=torch.zeros(4, 8192 + 8, dtype=torch.bfloat16)[:, 1:8193]), ValueError, "16-byte aligned"),
    ],
)
def test_typed_declines_name_the_operand(mutation, exc, token):
    ops = mutation(make_operands(4))
    with pytest.raises(exc, match=token):
        call_check(ops, out=ops.get("out"))


def test_declines_an_unserved_geometry_by_name():
    for g in (kcfg.Geometry(8192, 512), kcfg.Geometry(8192, 64), kcfg.Geometry(4096, 256), kcfg.Geometry(16384, 256)):
        with pytest.raises(NotImplementedError, match="validated"):
            call_check(make_operands(2, geometry=g), geometry=g)
    with pytest.raises(TypeError, match="geometry must be"):
        entry.resolve_geometry((8192, 256))
    with pytest.raises(ValueError, match="one source"):
        entry.resolve_geometry(GEOMETRIES[1], kcfg.CFG)
    assert entry.resolve_geometry(None, kcfg.CFG_TP16) == GEOMETRIES[1] and entry.resolve_geometry(None) == kcfg.DEFAULT_GEOMETRY


def test_declines_an_unserved_compute_capability_by_name():
    ops = make_operands(2)
    for cc in ((9, 0), (12, 0), (12, 1), (11, 0), (13, 0)):
        with pytest.raises(NotImplementedError, match=rf"10\.x .*{cc[0]}\.{cc[1]}"):
            call_check(ops, device_cc=cc)
    for cc in ((10, 0), (10, 3), (10, 7)):
        assert call_check(ops, device_cc=cc) == 2
    assert entry.cc_supported((10, 3)) and not entry.cc_supported(None) and not entry.cc_supported((12, 0))
    assert entry.SUPPORTED_CC_RANGES == ((100, 109),)


def test_declines_cpu_operands_when_no_capability_is_given():
    with pytest.raises(NotImplementedError, match="CUDA tensors only"):
        call_check(make_operands(2), device_cc=None)


def test_binding_check_requires_every_operand_on_the_given_device():
    ops = make_operands(2)
    with pytest.raises(ValueError, match="every operand must live on one device"):
        entry.check_binding(ops["x"], ops["w_gate_up"], ops["w_down"], ops["w_gate"], device=torch.device("meta"))
    with pytest.raises(ValueError, match="w_down is on meta"):
        entry.check_binding(ops["x"], ops["w_gate_up"], ops["w_down"].to("meta"), ops["w_gate"])


def test_host_entry_keeps_the_cc10_readings_and_a_given_cfg_overrides_the_pick():
    k = entry.SharedExpertDecode("cpu")
    assert k.cfg == kcfg.CFG and k.readings == SM100_READING and k.sm_count is None and k.geometry == kcfg.DEFAULT_GEOMETRY
    k = entry.SharedExpertDecode("cpu", geometry=GEOMETRIES[1])
    assert k.cfg == kcfg.CFG_TP16
    arm = kcfg.v1_cfg()
    k = entry.SharedExpertDecode("cpu", cfg=arm)
    assert k.cfg == arm and k.cfg_given and k.cfg_for(24).X_BULK_ROWS is False
    with pytest.raises(ValueError, match="one source"):
        entry.SharedExpertDecode("cpu", geometry=GEOMETRIES[1], cfg=arm)


def test_pdl_launch_knobs_apply_on_top_of_the_pick():
    k = entry.SharedExpertDecode("cpu", pdl=True)
    assert (k.cfg.PDL, k.cfg.PDL_PAIR, k.cfg.PDL_RELEASE) == (True, False, False)
    k = entry.SharedExpertDecode("cpu", pdl=True, pdl_pair=True, pdl_release=True)
    assert (k.cfg.PDL, k.cfg.PDL_PAIR, k.cfg.PDL_RELEASE) == (True, True, True)
    k = entry.SharedExpertDecode("cpu", cfg=kcfg.make_cfg(PDL_PAIR=True), pdl_pair=False)
    assert (k.cfg.PDL, k.cfg.PDL_PAIR) == (False, False)
    assert entry.SharedExpertDecode("cpu").launch_fields(kcfg.CFG) == {}


def test_compile_key_is_plan_time_only():
    """The same key for T 1 .. 16 (one form), another for 17 .. 24 (two m-tiles); the strides and the config are in it, T is not."""
    k = entry.SharedExpertDecode("cpu")
    ops = make_operands(1)
    out = torch.empty(32, 8192, dtype=torch.bfloat16)
    k1 = k.key(1, ops["x"], out)
    assert k1 == k.key(16, ops["x"], out) and k1 != k.key(17, ops["x"], out) and k.key(17, ops["x"], out) == k.key(24, ops["x"], out)
    assert k1[0] == kcfg.CFG and k1[0].TOKEN_TILES == 1 and k.key(24, ops["x"], out)[0].TOKEN_TILES == 2
    assert k1 != k.key(1, ops["x"], torch.empty(32, 8200, dtype=torch.bfloat16)[:, :8192])
    with pytest.raises(NotImplementedError, match="24"):
        k.cfg_for(25)


def test_declines_a_dsl_below_the_floor_by_version(monkeypatch):
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.7.1")))
    assert "4.8.0" in entry._dsl_version_error() and "4.7.1" in entry._dsl_version_error()
    monkeypatch.setattr(buffers, "_DSL_STATE", (False, None))
    assert "not installed" in entry._dsl_version_error()
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.8.0")))
    assert entry._dsl_version_error() is None


def test_cudnn_level_export_resolves_to_the_entry():
    import importlib

    assert cudnn.SharedExpertDecode is entry.SharedExpertDecode
    moe, frost = importlib.import_module("cudnn.moe"), importlib.import_module("cudnn.moe.frost")
    assert moe.SharedExpertDecode is entry.SharedExpertDecode and frost.Geometry is kcfg.Geometry
    assert "SharedExpertDecode" in cudnn._LAZY_OPTIONAL_IMPORTS and "SharedExpertDecode" in cudnn._OPTIONAL_DEPENDENCY_INSTALL_HINTS


def test_import_cudnn_does_not_import_the_kernel_or_torch():
    """The lazy boundary: a fresh interpreter importing cudnn loads neither the moe package's kernel modules nor the DSL."""
    import subprocess
    import sys

    code = "import sys, cudnn; print(sorted(m for m in sys.modules if m.startswith('cudnn.moe') or m == 'cutlass'))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]", out


def test_docs_rows_exist_for_the_api():
    """The determinism status page carries the API's row (SM90 unsupported, SM100/103 supported with the cc 10.3-verified / cc 10.0-unverified
    limit stated in the cell, SM107 unverified) and the goldens README its suite section; the operations docs page and the navigation entry
    exist."""
    det = (REPO / "docs" / "operations" / "determinism.md").read_text()
    row = [l for l in det.splitlines() if l.startswith("| `SharedExpertDecode`")]
    assert len(row) == 1, "one SharedExpertDecode row on the status page"
    cells = [c.strip() for c in row[0].strip("|").split("|")]
    assert cells[3] == "Unsupported" and cells[4].startswith("Supported (cc 10.3 verified; cc 10.0 unverified") and cells[5] == "Unverified", cells
    readme = (REPO / "test" / "python" / "determinism_goldens" / "README.md").read_text()
    assert "SharedExpertDecode" in readme and "test_shared_expert_decode_determinism.py" in readme
    assert (REPO / "docs" / "operations" / "moe" / "shared_expert_decode.md").exists()
    assert "shared_expert_decode" in (REPO / "docs" / "fern" / "versions" / "latest.yml").read_text()
    assert "shared_expert_decode" in (REPO / "docs" / "operations" / "overview.md").read_text()


def test_order_model_matches_the_fp64_reference_in_band():
    """The readable order model is a faithful spec of the function (fp32, the documented trees) -- within the band of the fp64 reference."""
    ops = make_operands(6, seed=3)
    assert_in_band(order_model(ops, kcfg.CFG), reference(ops), "order model")


# ---------------------------------------------------------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------------------------------------------------------
def _run(k, ops, out=None):
    y = k(ops["x"], ops["w_gate_up"], ops["w_down"], ops["w_gate"], out)
    torch.cuda.synchronize()
    return y


def _bytes(t):
    return t.detach().contiguous().view(torch.uint8)


@pytest.mark.parametrize("structured", (False, True), ids=("random", "structured"))
@pytest.mark.parametrize("tokens", (1, 2, 4, 6, 8, 16, 17, 24))
@pytest.mark.parametrize("g", GEOMETRIES, ids=_gid)
def test_matches_the_fp64_reference_in_band(g, tokens, structured):
    dev = _device_or_skip()
    ops = make_operands(tokens, device=dev, seed=10 + tokens, geometry=g, structured=structured)
    k = entry.SharedExpertDecode(dev, geometry=g)
    y = _run(k, ops)
    assert y.shape == (tokens, g.HIDDEN) and y.dtype == torch.bfloat16 and torch.isfinite(y.float()).all()
    assert_in_band(y, reference(ops), f"{_gid(g)} T {tokens} {'structured' if structured else 'random'}")
    assert_in_band(y, order_model(ops, k.cfg_for(tokens)).to(dev), "vs the order model")


def test_entry_reads_the_device_readings_and_picks_the_form_per_tokens():
    dev = _device_or_skip()
    k = entry.SharedExpertDecode(dev)
    assert k.readings == entry.device_shared_memory(dev) and k.sm_count == torch.cuda.get_device_properties(dev).multi_processor_count
    assert k.cfg.SMEM_OPTIN_BYTES == k.readings[1] and k.cfg.SM_COUNT == k.sm_count and k.cfg_for(16).TOKEN_TILES == 1 and k.cfg_for(17).TOKEN_TILES == 2
    assert (k.cfg_for(16).N_GROUPS_EFF, k.cfg_for(17).N_GROUPS_EFF, k.cfg_for(16).WAVES1, k.cfg_for(17).WAVES1) == (2, 2, 1, 1)
    ops = make_operands(17, device=dev, seed=1)
    _run(k, ops)
    _run(k, dict(ops, x=ops["x"][:3]))
    assert len(k.compiled) == 2 and len(k.scratch) == 2 and k.scratch[k.cfg_for(17)]["act"].shape == (32, 256)
    assert k.scratch[k.cfg_for(17)]["gate_part"].shape == (32, 64)


def test_caller_owned_out_and_padded_rows_stay_untouched():
    """Rows of ``out`` beyond T, and the scratch rows >= T, keep their sentinel; the kernel writes exactly the live rows."""
    dev = _device_or_skip()
    ops = make_operands(5, device=dev, seed=2)
    k = entry.SharedExpertDecode(dev)
    out = torch.full((9, 8192), 1.5e30, dtype=torch.bfloat16, device=dev)
    y = _run(k, ops, out)
    assert y is out
    assert torch.isfinite(out[:5].float()).all() and (out[5:] == torch.tensor(1.5e30, dtype=torch.bfloat16, device=dev)).all()
    assert_in_band(out[:5], reference(ops), "caller-owned out")
    s = k.scratch[k.cfg_for(5)]
    s["act"][5:].fill_(float("nan"))
    s["gate_part"][5:].fill_(float("nan"))
    y2 = _run(k, ops).clone()
    assert torch.isnan(s["act"][5:].float()).all() and torch.isnan(s["gate_part"][5:]).all(), "the scratch rows past T are never written"
    assert torch.equal(_bytes(y2), _bytes(out[:5])), "the NaN-filled dead scratch rows do not reach the live rows (zero-filled copies)"


def test_graph_capture_replays_bitwise_with_eager():
    dev = _device_or_skip()
    k = entry.SharedExpertDecode(dev)
    layers = [make_operands(6, device=dev, seed=100 + i) for i in range(4)]
    outs = [torch.empty(6, 8192, dtype=torch.bfloat16, device=dev) for _ in layers]
    eager = [_run(k, lay).clone() for lay in layers]  # compile + warm
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for lay, o in zip(layers, outs):
            k(lay["x"], lay["w_gate_up"], lay["w_down"], lay["w_gate"], o)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for lay, o in zip(layers, outs):
            k(lay["x"], lay["w_gate_up"], lay["w_down"], lay["w_gate"], o)
    for o in outs:
        o.zero_()
    for _ in range(3):
        g.replay()
        torch.cuda.synchronize()
        for e, o in zip(eager, outs):
            assert torch.equal(_bytes(e), _bytes(o)), "graph replay bitwise with eager"


BITWISE_ARMS = [
    dict(PDL=True),
    dict(PDL_PAIR=True),
    dict(PDL=True, PDL_PAIR=True, PDL_RELEASE=True),
    dict(N_GROUPS=1),
    dict(N_GROUPS=2),
    dict(X_BULK_ROWS=False, N_GROUPS=1),  # the ring at K_SPLIT 8 needs two waves at T 24 (the one-wave rule would decline it): an explicit N_GROUPS
    dict(X_BULK_ROWS=False, X_STAGES=2, N_GROUPS=1),
    dict(W_PHASES=2),
    dict(ROWS_PER_CTA=32),
]


@pytest.mark.parametrize("tokens", (1, 6, 24))
@pytest.mark.parametrize("arm", BITWISE_ARMS, ids=lambda a: ",".join(f"{k}={v}" for k, v in a.items()))
def test_bit_neutral_knob_arms_write_the_default_bytes(arm, tokens):
    """The PDL attributes (launch 1's, launch 2's, both plus the release), one or two row-pair groups per CTA (the two-wave N_GROUPS 1 at T 24
    included), the ring at K_SPLIT 8 (two chunks, all resident, and the minimum ring; two waves at T 24, chosen knowingly by the explicit
    N_GROUPS -- the one-wave rule declines a two-wave default), the two W phases and the 32-row launch 2 move no fp32
    addition: byte-identical with the default on the same operands.  The stream is drained before a PDL launch: the caller obligation (no
    in-flight predecessor writes the weights) holds in the test as in the chain."""
    dev = _device_or_skip()
    ops = make_operands(tokens, device=dev, seed=40 + tokens)
    base = _run(entry.SharedExpertDecode(dev), ops).clone()
    torch.cuda.synchronize()
    k = entry.SharedExpertDecode(dev, cfg=kcfg.make_cfg(**arm))
    got = _run(k, ops)
    assert torch.equal(_bytes(base), _bytes(got)), f"{arm} at T {tokens}"


ORDER_ARMS = {
    "ksplit1-ring": dict(K_SPLIT=1, X_BULK_ROWS=False),
    "ksplit2-bulk": dict(K_SPLIT=2),
    "ksplit4-bulk": dict(K_SPLIT=4),
    "ksplit8-launch2-fold": dict(GATE_CLUSTER_FOLD=False),
    "v1": None,
}


@pytest.mark.parametrize("arm", sorted(ORDER_ARMS))
def test_order_arms_are_rounding_order_arms_within_the_band(arm):
    """K_SPLIT changes the slice sum and the gate tree's leaf count (64 .. 512 leaves), GATE_CLUSTER_FOLD the gate tree's shape, the v1 arm
    both: rounding-order arms -- in band against the fp64 reference and the order model, not bitwise with the default."""
    dev = _device_or_skip()
    ops = make_operands(6, device=dev, seed=77)
    cfg = kcfg.v1_cfg() if ORDER_ARMS[arm] is None else kcfg.make_cfg(**ORDER_ARMS[arm])
    y = _run(entry.SharedExpertDecode(dev, cfg=cfg), ops)
    assert_in_band(y, reference(ops), arm)
    assert_in_band(y, order_model(ops, cfg).to(dev), f"{arm} vs the order model")


def test_two_tile_form_at_sixteen_tokens_writes_the_one_tile_bytes():
    """The form boundary A/B: TOKEN_TILES 2 serving T 16 (its second m-tile all zero rows) writes the bytes of the one-tile form -- the
    zero-padded rows and the extra mma cannot touch a live row (section 4, point (2))."""
    dev = _device_or_skip()
    ops = make_operands(16, device=dev, seed=16)
    one = _run(entry.SharedExpertDecode(dev), ops).clone()
    two = _run(entry.SharedExpertDecode(dev, cfg=kcfg.make_cfg(TOKEN_TILES=2)), ops)
    assert torch.equal(_bytes(one), _bytes(two))


def test_twenty_five_tokens_is_a_typed_decline_never_a_provider_switch():
    dev = _device_or_skip()
    k = entry.SharedExpertDecode(dev)
    ops = make_operands(25, device=dev)
    with pytest.raises(NotImplementedError, match="at most 24"):
        k(ops["x"], ops["w_gate_up"], ops["w_down"], ops["w_gate"])


def test_warm_entry_rejects_a_bad_binding_before_the_compiled_pair_is_looked_up():
    dev = _device_or_skip()
    k = entry.SharedExpertDecode(dev)
    ops = make_operands(2, device=dev)
    _run(k, ops)
    with pytest.raises(TypeError, match="x must be torch.bfloat16"):
        k(ops["x"].float(), ops["w_gate_up"], ops["w_down"], ops["w_gate"])
    with pytest.raises(ValueError, match="every operand must live on one device"):
        k(ops["x"].cpu(), ops["w_gate_up"], ops["w_down"], ops["w_gate"])
