# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Determinism of the shared-expert decode pair (``cudnn.SharedExpertDecode``): bitwise replay under side-stream contention on fresh
buffers, batch invariance of every token (alone at T 1 -- across the form boundary -- and inside T 6 / 24; the permuted rows; a duplicated
token; neighbour isolation; a row-strided x / out view inside a wider slab), the two outputs of the pair (``y`` and the inter-launch
``act`` / ``gate_part`` scratch) compared as raw bytes, the inputs unmodified by a launch, the T 24 -> 25 boundary as the typed decline, and the goldens (one small case per geometry and form; recorded with
``CUDNN_DETERMINISM_RECORD=<new dir>`` after the replay and the independent reference pass): the default form's ``ksplit8`` sets (the
one-wave default's class: eight K slices, the gate leaves folded on the cluster leader) and, on the first release's knob arm (``v1_cfg``),
the ``ksplit2`` / ``ksplit4`` sets recorded with that release -- the arm reproducing them byte for byte is the proof that the one-wave
refactor moved no first-release addition.  The
independent reference is a separate check at the harness band (``test_shared_expert_decode.py``).  Row-position invariance of the
``m16n8k16`` datapath is the premise these cells pin (the kernel module's docstring, point 3)."""

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from shared_expert_utils import GEOMETRIES, _device_or_skip, _gid, assert_in_band, make_operands, reference  # noqa: E402

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot  # noqa: E402

from cudnn.moe.frost import shared_expert_decode as entry  # noqa: E402
from cudnn.moe.frost.kernel import shared_expert_decode_config as kcfg  # noqa: E402

pytestmark = [pytest.mark.L0]


def run(k, ops, tokens=None):
    """One pair launch into FRESH buffers; the outputs compared as bytes: ``y`` [T, H] and the live rows of the two scratch tensors."""
    x = ops["x"] if tokens is None else ops["x"][:tokens]
    T = x.shape[0]
    out = torch.empty(T, x.shape[1], dtype=torch.bfloat16, device=x.device)
    k(x, ops["w_gate_up"], ops["w_down"], ops["w_gate"], out)
    s = k.scratch[k.cfg_for(T)]
    return {"y": out, "act": s["act"][:T].clone(), "gate_part": s["gate_part"][:T].clone()}


def rows(snap, index):
    return {name: value[index : index + 1] for name, value in snap.items()}


@pytest.mark.parametrize("tokens", (1, 6, 24))
@pytest.mark.parametrize("g", GEOMETRIES, ids=_gid)
def test_replay(g, tokens):
    """Eight executions on fresh output buffers under side-stream contention: every byte of y, act and gate_part matches the first."""
    dev = _device_or_skip()
    ops = make_operands(tokens, device=dev, seed=200 + tokens, geometry=g)
    k = entry.SharedExpertDecode(dev, geometry=g)
    expected = assert_replays(lambda: run(k, ops), repeats=8)
    assert_in_band(expected["y"], reference(ops), f"{_gid(g)} T {tokens}")


@pytest.mark.parametrize("g", GEOMETRIES, ids=_gid)
def test_replay_on_a_fresh_entry_and_fresh_operands(g):
    """Two entries, two identical operand sets drawn afresh: byte-identical (the compile, the scratch and the buffers all fresh)."""
    dev = _device_or_skip()
    a = snapshot(run(entry.SharedExpertDecode(dev, geometry=g), make_operands(6, device=dev, seed=9, geometry=g)))
    b = snapshot(run(entry.SharedExpertDecode(dev, geometry=g), make_operands(6, device=dev, seed=9, geometry=g)))
    assert_bitwise(a, b, "fresh entry")


@pytest.mark.parametrize("inside", (6, 24))
@pytest.mark.parametrize("g", GEOMETRIES, ids=_gid)
def test_batch_invariance(g, inside):
    """Every token of a T = ``inside`` step writes the bytes it writes alone (T 1: the one-m-tile form, row 0 of m-tile 0, so the T 24 cell
    crosses the form boundary and the row position), and inside the reversed batch; the inputs are built once and sliced."""
    dev = _device_or_skip()
    ops = make_operands(inside, device=dev, seed=300 + inside, geometry=g)
    k = entry.SharedExpertDecode(dev, geometry=g)
    batch = assert_replays(lambda: run(k, ops), repeats=3)
    perm = torch.arange(inside - 1, -1, -1, device=dev)
    reversed_batch = snapshot(run(k, dict(ops, x=ops["x"][perm].contiguous())))
    for t in range(inside):
        alone = snapshot(run(k, dict(ops, x=ops["x"][t : t + 1])))
        assert_bitwise(alone, rows(batch, t), f"{_gid(g)} token {t} alone vs inside T {inside}")
        assert_bitwise(rows(reversed_batch, inside - 1 - t), rows(batch, t), f"{_gid(g)} token {t} in the reversed batch")


@pytest.mark.parametrize("g", GEOMETRIES, ids=_gid)
def test_duplicated_token_and_neighbour_isolation(g):
    """A token duplicated inside the batch: both copies equal its alone bytes; the other rows redrawn (removal / insertion of neighbours): the
    token's bytes unchanged; a row-strided x view (the token inside a taller slab) likewise."""
    dev = _device_or_skip()
    ops = make_operands(6, device=dev, seed=400, geometry=g)
    k = entry.SharedExpertDecode(dev, geometry=g)
    alone = snapshot(run(k, dict(ops, x=ops["x"][2:3])))
    dup = torch.cat([ops["x"][:3], ops["x"][2:3], ops["x"][3:]], dim=0).contiguous()  # token 2 twice: rows 2 and 3
    got = snapshot(run(k, dict(ops, x=dup)))
    assert_bitwise(rows(got, 2), alone, f"{_gid(g)} duplicate, first copy")
    assert_bitwise(rows(got, 3), alone, f"{_gid(g)} duplicate, second copy")
    other = make_operands(24, device=dev, seed=401, geometry=g)["x"]
    mixed = torch.cat([other[:5], ops["x"][2:3], other[5:19]], dim=0).contiguous()  # 20 rows (the two-tile form), the token at row 5
    got = snapshot(run(k, dict(ops, x=mixed)))
    assert_bitwise(rows(got, 5), alone, f"{_gid(g)} neighbours redrawn, the two-tile form")
    slab = make_operands(4, device=dev, seed=402, geometry=g, x_rows=8)["x"]  # a 4-row slice of an 8-row slab (row stride H), then the same rows contiguous
    view_ops = dict(ops, x=slab)
    assert slab.shape[0] == 4
    a = snapshot(run(k, view_ops))
    b = snapshot(run(k, dict(ops, x=slab.clone())))
    assert_bitwise(a, b, f"{_gid(g)} a row slice of a taller slab vs a contiguous copy")


@pytest.mark.parametrize("tokens", (5, 20))
@pytest.mark.parametrize("g", GEOMETRIES, ids=_gid)
def test_row_strided_views_write_the_contiguous_bytes_and_leave_the_inputs_unmodified(g, tokens):
    """``x`` and ``out`` as column slices of a slab with H + 64 columns (row stride 8,256 elements, a 16-byte multiple, != H: the ``XS`` /
    ``YS`` compile key, the footprint arithmetic of the host views and the kernel's ``t x stride`` addressing): the live rows' bytes equal
    the contiguous run's, the slab columns past H keep their sentinel, and every input tensor holds the bytes it held before the launch."""
    dev = _device_or_skip()
    H = g.HIDDEN
    ops = make_operands(tokens, device=dev, seed=600 + tokens, geometry=g)
    k = entry.SharedExpertDecode(dev, geometry=g)
    before = snapshot({name: ops[name] for name in ("x", "w_gate_up", "w_down", "w_gate")})
    contiguous = snapshot(run(k, ops))
    x_slab = torch.full((tokens, H + 64), 2.0, dtype=torch.bfloat16, device=dev)
    x_slab[:, :H] = ops["x"]
    out_slab = torch.full((tokens, H + 64), 1.5e30, dtype=torch.bfloat16, device=dev)
    xv, outv = x_slab[:, :H], out_slab[:, :H]
    assert xv.stride(0) == H + 64 and not xv.is_contiguous()
    k(xv, ops["w_gate_up"], ops["w_down"], ops["w_gate"], outv)
    torch.cuda.synchronize()
    s = k.scratch[k.cfg_for(tokens)]
    strided = {"y": outv.contiguous(), "act": s["act"][:tokens].clone(), "gate_part": s["gate_part"][:tokens].clone()}
    assert_bitwise(snapshot(strided), contiguous, f"{_gid(g)} T {tokens}: a row-strided x / out view vs the contiguous run")
    assert (out_slab[:, H:] == torch.tensor(1.5e30, dtype=torch.bfloat16, device=dev)).all(), "the out slab's columns past H are untouched"
    assert (x_slab[:, H:] == 2.0).all(), "the x slab's columns past H are untouched"
    after = snapshot({name: ops[name] for name in ("x", "w_gate_up", "w_down", "w_gate")})
    assert_bitwise(after, before, f"{_gid(g)}: the inputs after two launches")


def test_twenty_four_to_twenty_five_is_the_typed_decline():
    dev = _device_or_skip()
    k = entry.SharedExpertDecode(dev)
    ops = make_operands(25, device=dev, seed=25)
    snapshot(run(k, ops, tokens=24))
    with pytest.raises(NotImplementedError, match="at most 24"):
        run(k, ops)


@pytest.mark.parametrize("tiles", (1, 2))
@pytest.mark.parametrize("g", GEOMETRIES, ids=_gid)
def test_golden(g, tiles):
    """One small case per geometry and form: three executions, the independent reference, then the saved hashes (recorded into a NEW
    directory with ``CUDNN_DETERMINISM_RECORD`` only after the replay and reference cells pass; a change of the pinned K_SPLIT or of the
    gate tree is a new golden set with a stated reason -- the one-wave default's ``ksplit8`` sets: K_SPLIT 2 / 4 -> 8 and the cluster-side
    gate fold)."""
    dev = _device_or_skip()
    tokens = 3 if tiles == 1 else 19
    ops = make_operands(tokens, device=dev, seed=500 + tiles, geometry=g)
    k = entry.SharedExpertDecode(dev, geometry=g)
    actual = assert_replays(lambda: run(k, ops), repeats=3)
    assert_in_band(actual["y"], reference(ops), f"{_gid(g)} golden T {tokens}")
    form = k.cfg_for(tokens)
    assert form.TOKEN_TILES == tiles and form.K_SPLIT == kcfg.k_split_for(g) == 8 and form.X_BULK_ROWS and form.GATE_CLUSTER_FOLD_EFF
    assert_golden(f"shared-expert-decode-{_gid(g)}-tiles{tiles}-ksplit{k.cfg.K_SPLIT}", ops, actual)


@pytest.mark.parametrize("tiles", (1, 2))
@pytest.mark.parametrize("g", GEOMETRIES, ids=_gid)
def test_v1_arm_reproduces_the_v1_goldens(g, tiles):
    """The v1 knob arm (``kcfg.v1_cfg``: K_SPLIT 2 at I 256 / 4 at I 128, the x ring, the launch-2 gate fold, one group per CTA) on the SAME
    operands as :func:`test_golden` writes the bytes of the ``ksplit2`` / ``ksplit4`` goldens recorded with the first release: the shared
    code paths (the mainloop, the folds, launch 2) moved no first-release addition."""
    dev = _device_or_skip()
    tokens = 3 if tiles == 1 else 19
    ops = make_operands(tokens, device=dev, seed=500 + tiles, geometry=g)
    arm = kcfg.v1_cfg(g)
    k = entry.SharedExpertDecode(dev, cfg=arm)
    actual = assert_replays(lambda: run(k, ops), repeats=3)
    assert_in_band(actual["y"], reference(ops), f"{_gid(g)} v1 arm T {tokens}")
    form = k.cfg_for(tokens)
    assert form.TOKEN_TILES == tiles and not form.X_BULK_ROWS and not form.GATE_CLUSTER_FOLD_EFF and form.N_GROUPS_EFF == 1
    assert_golden(f"shared-expert-decode-{_gid(g)}-tiles{tiles}-ksplit{arm.K_SPLIT}", ops, actual)
