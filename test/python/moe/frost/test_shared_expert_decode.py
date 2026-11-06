# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared-expert decode kernel pair (``cudnn.moe.frost.shared_expert_decode``): the config's structural invariants and derived
constants (the gate-logit partition, the SMEM tables, the ring ledger, the barrier constants) on the host, the typed declines of the host
entry (CPU tensors), the plan-time-only compile key, the lazy export and the docs rows; on a cc 10.x device the pair against an fp64
reference of the bf16 operands at the harness band, the padded-row hygiene, the CUDA-graph replay, and every knob arm bitwise with the
default (the PDL arms, the ring depth, the W phases, the launch-2 row count), the K_SPLIT arms within the band (a rounding-order axis).
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
def test_default_cfg_pins_k_split_per_geometry_at_128_ctas():
    """K_SPLIT is a function of the geometry alone (the slice order of every output element never moves with T): 2 at I 256 (64 clusters x 2),
    4 at I 128 (32 x 4) -- 128 CTAs of launch 1 either way, one per SM on a 148- / 152-SM part."""
    for g, s in ((GEOMETRIES[0], 2), (GEOMETRIES[1], 4)):
        cfg = kcfg.default_cfg(g)
        assert (cfg.K_SPLIT, cfg.GRID1, cfg.GRID2, cfg.THREADS, cfg.THREADS2) == (s, 128, 128, 256, 256), g
        assert (cfg.PDL, cfg.PDL_PAIR, cfg.PDL_RELEASE, cfg.X_STAGES, cfg.W_PHASES, cfg.ROWS_PER_CTA, cfg.MIN_CTAS_PER_SM) == (False, False, False, 0, 1, 64, 1)
    assert kcfg.CFG == kcfg.default_cfg() and kcfg.CFG_TP16 == kcfg.default_cfg(GEOMETRIES[1])
    assert kcfg.k_split_for(kcfg.Geometry(8192, 256)) == 2 and kcfg.k_split_for(kcfg.Geometry(8192, 128)) == 4


def test_geometry_validates_and_holds_plain_ints():
    g = kcfg.Geometry(8192, 256)
    assert isinstance(g.HIDDEN, int) and g == kcfg.Geometry(8192, 256) and "8192" in g.describe()
    for bad in ((0, 256), (8192, 0), (8192, -1), (8192, 2.5), (True, 256)):
        with pytest.raises(ValueError, match="Geometry"):
            kcfg.Geometry(*bad)


@pytest.mark.parametrize("g", GEOMETRIES, ids=_gid)
@pytest.mark.parametrize("k_split", (1, 2, 4))
@pytest.mark.parametrize("tiles", (1, 2))
def test_gate_partition_tiles_hidden_for_every_cfg(g, k_split, tiles):
    """The audit's host pin: GATE_PARTS x GATE_COLS == H for every (geometry, K_SPLIT) the knobs produce, the fold's per-lane counts integral,
    and warp 1's pair coverage exactly the CTA's columns (every column of H owned once across the grid)."""
    cfg = kcfg.make_cfg(GEOM=g, K_SPLIT=k_split, TOKEN_TILES=tiles)
    assert cfg.GATE_PARTS * cfg.GATE_COLS == g.HIDDEN and cfg.GATE_COLS % 2 == 0 and cfg.GATE_PARTS % 32 == 0
    assert cfg.GATE_PARTS == cfg.GRID1 and cfg.GATE_PARTS_PER_LANE * 32 == cfg.GATE_PARTS
    owned = [0] * g.HIDDEN
    for q in range(cfg.GATE_PARTS):
        ks, c = divmod(q, cfg.CLUSTERS)
        k0 = ks * cfg.K_SLICE + c * cfg.GATE_COLS
        for lane in range(32):
            for p in range(cfg.GATE_PAIRS_PER_LANE):
                col = 2 * lane + 64 * p
                if col < cfg.GATE_COLS:
                    owned[k0 + col] += 1
                    owned[k0 + col + 1] += 1
    assert all(v == 1 for v in owned), "every hidden column is summed exactly once across the grid"


@pytest.mark.parametrize("g", GEOMETRIES, ids=_gid)
@pytest.mark.parametrize("k_split", (1, 2, 4))
@pytest.mark.parametrize("tiles", (1, 2))
def test_smem_tables_fit_and_every_pitch_covers_the_banks(g, k_split, tiles):
    """Every row pitch is 128 n + 16 B (the eight row addresses of one ldmatrix phase cover 32 banks); launch 1's total fits the opt-in at
    every form, launch 2's too; the ring has at least two stages and the budget rule keeps every chunk resident where it fits."""
    cfg = kcfg.make_cfg(GEOM=g, K_SPLIT=k_split, TOKEN_TILES=tiles)
    for name, (nbytes, pitch) in {**kcfg.smem_table1(cfg), **kcfg.smem_table2(cfg)}.items():
        assert nbytes >= 0, name
        if pitch is not None and pitch > 16:
            assert pitch % 128 == 16, (name, pitch)
            banks = {((r * pitch) // 4 + b) % 32 for r in range(8) for b in range(4)}
            assert len(banks) == 32, (name, pitch)
    assert sum(b for b, _ in kcfg.smem_table1(cfg).values()) == cfg.SMEM1_BYTES <= cfg.SMEM_OPTIN_BYTES
    assert sum(b for b, _ in kcfg.smem_table2(cfg).values()) == cfg.SMEM2_BYTES <= cfg.SMEM_OPTIN_BYTES
    assert 2 <= cfg.X_STAGES_EFF <= cfg.X_CHUNKS
    if cfg.SMEM1_FIXED_BYTES + cfg.X_CHUNKS * cfg.X_CHUNK_BYTES + kcfg.SMEM_RING_MARGIN_BYTES <= cfg.SMEM1_BUDGET_BYTES:
        assert cfg.X_STAGES_EFF == cfg.X_CHUNKS, "the whole slice is resident when it fits"


def test_default_form_figures_of_record():
    """The default form's figures of record (I 256, K_SPLIT 2; the kernel module's SMEM table quotes these): w_s 65,664, x_s 8 x 16 x 1,040
    (Tpad 16) / 4 x 32 x 1,040 (Tpad 32), 202,912 / 207,008 B in all, one CTA per SM; the I 128 default (K_SPLIT 4) 104,608 / 176,288 B
    (two CTAs per SM by shared memory at Tpad 16)."""
    c1 = kcfg.make_cfg(TOKEN_TILES=1)
    c2 = kcfg.make_cfg(TOKEN_TILES=2)
    assert (c1.W_SMEM_BYTES, c1.X_STAGES_EFF, c1.X_CHUNK_BYTES, c1.X_SMEM_BYTES) == (65664, 8, 16640, 133120)
    assert (c2.X_STAGES_EFF, c2.X_CHUNK_BYTES, c2.X_SMEM_BYTES) == (4, 33280, 133120)
    assert c1.SMEM1_BYTES == 202912 and c2.SMEM1_BYTES == 207008 and c1.SMEM2_BYTES == 42368 and c2.SMEM2_BYTES == 50816
    assert c1.co_resident_ctas1 == 1
    t16 = kcfg.CFG_TP16
    assert t16.K_SPLIT == 4 and t16.X_CHUNKS == 4 and t16.X_STAGES_EFF == 4 and t16.co_resident_ctas1 == 2 and t16.SMEM1_BYTES == 104608
    assert replace(t16, TOKEN_TILES=2).SMEM1_BYTES == 176288 and kcfg.make_cfg(GEOM=GEOMETRIES[1], K_SPLIT=4, TOKEN_TILES=2).SMEM2_BYTES == 26240
    big = kcfg.make_cfg(K_SPLIT=4, MIN_CTAS_PER_SM=2)  # the 256-CTA arm at I 256 must co-reside twice: the budget rule halves the SM
    assert big.co_resident_ctas1 == 2 and big.SMEM1_BUDGET_BYTES == kcfg.SM100_SMEM_PER_SM_BYTES // 2 - kcfg.SMEM_CTA_RESERVE_BYTES


def test_barrier_ledger_constants_and_the_ring_walk():
    """The barrier-table constants with their derivation, and the pure-Python ring ledger for every (X_CHUNKS, X_STAGES) the cfgs produce:
    chunk i has landed before iteration i, and a stage is refilled only by the iteration after its last read (behind that iteration's barrier)."""
    assert kcfg.W_MBAR_ARRIVERS == 1 and kcfg.KRED_MBAR_ARRIVERS == 1
    for g in GEOMETRIES:
        for s in (1, 2, 4):
            for tiles in (1, 2):
                for phases in (1, 2):
                    cfg = kcfg.make_cfg(GEOM=g, K_SPLIT=s, TOKEN_TILES=tiles, W_PHASES=phases)
                    assert cfg.W_PHASE_BYTES * cfg.W_PHASES == cfg.W_SLICE_BYTES == 8 * cfg.K_SLICE * 2
                    assert cfg.W_PHASE_BYTES == 8 * cfg.W_PHASE_ROW_BYTES and cfg.W_PHASE_ROW_BYTES % 16 == 0
                    assert cfg.KRED_BYTES == (s - 1) * tiles * 32 * 16 == cfg.MAIL_SMEM_BYTES
                    assert cfg.X_COPIES * cfg.THREADS * 16 == cfg.TPAD * kcfg.CHUNK_K * 2
                    assert cfg.WD_COPIES * cfg.THREADS2 * 16 == cfg.ROWS_PER_CTA * cfg.I * 2
                    assert cfg.ACT_COPIES * cfg.THREADS2 * 16 == cfg.TPAD * cfg.I * 2
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
        (dict(TOKEN_TILES=3), "TOKEN_TILES"),
        (dict(W_PHASES=4), "W_PHASES"),
        (dict(ROWS_PER_CTA=16), "ROWS_PER_CTA"),
        (dict(MIN_CTAS_PER_SM=3), "MIN_CTAS_PER_SM"),
        (dict(X_STAGES=1), "X_STAGES"),
        (dict(X_STAGES=9), "X_STAGES"),
        (
            dict(K_SPLIT=4, MIN_CTAS_PER_SM=1, X_STAGES=4, GEOM=kcfg.Geometry(8192, 256), SMEM_OPTIN_BYTES=100 * 1024, SMEM_PER_SM_BYTES=101 * 1024),
            "exceeds the per-CTA opt-in",
        ),
        (dict(SMEM_OPTIN_BYTES=100 * 1024, SMEM_PER_SM_BYTES=101 * 1024), "X_STAGES"),
        (dict(SMEM_OPTIN_BYTES=300 * 1024), "SMEM_OPTIN_BYTES"),
        (dict(GEOM=kcfg.Geometry(8000, 256)), "HIDDEN"),
        (dict(GEOM=kcfg.Geometry(8192, 200)), "INTERMEDIATE"),
        (dict(GEOM=kcfg.Geometry(8192, 64)), "do not split"),
        (dict(GEOM=kcfg.Geometry(8192, 64), ROWS_PER_CTA=32, K_SPLIT=1), "GATE_PARTS"),
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
    assert kcfg.form_for(kcfg.CFG, 6) is kcfg.CFG and kcfg.form_for(kcfg.CFG, 24).TOKEN_TILES == 2


def test_part_readings_validate():
    assert kcfg.part_readings(*SM100_READING) == dict(SMEM_PER_SM_BYTES=SM100_READING[0], SMEM_OPTIN_BYTES=SM100_READING[1])
    with pytest.raises(ValueError, match="cannot exceed"):
        kcfg.part_readings(100, 200)
    with pytest.raises(ValueError, match="positive int"):
        kcfg.part_readings(0, 0)


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
    arm = kcfg.make_cfg(K_SPLIT=1)
    k = entry.SharedExpertDecode("cpu", cfg=arm)
    assert k.cfg == arm and k.cfg_given
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
    assert k.cfg.SMEM_OPTIN_BYTES == k.readings[1] and k.cfg_for(16).TOKEN_TILES == 1 and k.cfg_for(17).TOKEN_TILES == 2
    ops = make_operands(17, device=dev, seed=1)
    _run(k, ops)
    _run(k, dict(ops, x=ops["x"][:3]))
    assert len(k.compiled) == 2 and len(k.scratch) == 2 and k.scratch[k.cfg_for(17)]["act"].shape == (32, 256)


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
    dict(X_STAGES=2),
    dict(W_PHASES=2),
    dict(ROWS_PER_CTA=32),
]


@pytest.mark.parametrize("tokens", (1, 6, 24))
@pytest.mark.parametrize("arm", BITWISE_ARMS, ids=lambda a: ",".join(f"{k}={v}" for k, v in a.items()))
def test_bit_neutral_knob_arms_write_the_default_bytes(arm, tokens):
    """The PDL attributes (launch 1's, launch 2's, both plus the release), the minimum ring, the two W phases and the 32-row launch 2 move no
    fp32 addition: byte-identical with the default on the same operands.  The stream is drained before a PDL launch: the caller obligation
    (no in-flight predecessor writes the weights) holds in the test as in the chain."""
    dev = _device_or_skip()
    ops = make_operands(tokens, device=dev, seed=40 + tokens)
    base = _run(entry.SharedExpertDecode(dev), ops).clone()
    torch.cuda.synchronize()
    k = entry.SharedExpertDecode(dev, cfg=kcfg.make_cfg(**arm))
    got = _run(k, ops)
    assert torch.equal(_bytes(base), _bytes(got)), f"{arm} at T {tokens}"


@pytest.mark.parametrize("k_split", (1, 4))
def test_k_split_arms_are_rounding_order_arms_within_the_band(k_split):
    """K_SPLIT changes the slice sum and the gate tree (64 / 128 / 256 leaves): a rounding-order arm -- in band against the fp64 reference,
    not bitwise with the default.  K_SPLIT 4 at I 256 is 256 CTAs: the two-per-SM budget form."""
    dev = _device_or_skip()
    ops = make_operands(6, device=dev, seed=77)
    arm = kcfg.make_cfg(K_SPLIT=k_split, MIN_CTAS_PER_SM=2 if k_split == 4 else 1)
    y = _run(entry.SharedExpertDecode(dev, cfg=arm), ops)
    assert_in_band(y, reference(ops), f"K_SPLIT {k_split}")
    assert_in_band(y, order_model(ops, arm).to(dev), f"K_SPLIT {k_split} vs the order model")


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
