# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Rubin index-list sparse d256 forward (``kernels/sm107/sparse_d256_f16.py``) over PRESCRIBED per-CTA item sequences:
every ring barrier's slot and parity must come from a pipeline state carried across the work items a persistent CTA runs.

A per-item restart of the parity arithmetic passes every one-item-per-CTA cell and fails on an odd / even MIX of tile counts
between consecutive items: ``1, 1, 1`` leaves the ``s_empty`` ring a phase behind (a hang), ``3, 1, 2`` reads a stale score
slot, mixes whose tile counts do not sum to a multiple of the K/V ring depth desynchronise that ring.  ``sparse_item_sequences``
builds work lists on which those sequences are the typical per-CTA order (waves of the device's SM count along the token
axis, the tile count set through ``block_lens``), holds the open-tail phase constant through the per-batch KV length, poisons
the outputs with a sentinel (a surviving one = a row never written), poisons the tensor-memory residue through NaN query rows
on the item before the sequence, and appends batches of KV length 0 (dead items in a run) next to the dead waves.

Host tier (any box): the builder's arithmetic -- the count that yields a tile count round-trips through the kernel's clamp,
every designed wave lands its target, a ``0`` wave is dead with no open block and tail-only with one, the natural form
interleaves the four residues, the filler region costs one tile per item.

Rubin tier (``requires_rubin``): the six sequences x the five tail phases, each with two launches.  Pins: no sentinel on any
row (every item ran and stored), no non-finite output on a row whose query is finite (NaN residue did not travel), dead rows
``O = 0`` / ``LSE = -inf`` exactly, the live rows within the sparse module's budget (atol 2e-2 on O and LSE, tighter than the dense sm107 suite's 5e-2 on O -- nothing widened),
and the two launches bitwise.  A 12-fresh-process census of the same cells is a results-tree driver, not a test.
"""

import pytest
import torch

from frost_test_utils import requires_dsl, requires_rubin
from sparse_item_sequences import (
    CARRIED_STATE_SEQUENCES,
    build_item_sequence_work_list,
    count_for_tiles,
    default_scale,
    evaluate,
    expected_item_bounds,
    poisoned_outputs,
    reference_outputs,
    run_sparse_forward,
    tiles_of,
)

pytestmark = [pytest.mark.L0, requires_dsl]

ATOL = 2e-2  # on O and LSE, no rtol: tighter than the dense sm107 suite's O budget (atol 5e-2 / rtol 3e-2) -- never widened here
_PHASES = [None, 0, 1, 2, 3]
_PHASE_IDS = ["natural", "tail0", "tail1", "tail2", "tail3"]


def _seq_id(seq) -> str:
    return "n" + "-".join(str(n) for n in seq)


# ============================================================================ host: the builder's arithmetic
def test_count_for_tiles_round_trips_through_the_kernel_clamp():
    for h in (0, 1):
        assert tiles_of(count_for_tiles(0, h), h) == 1, "a dead / tail-only item still runs one clamped tile"
        for n in range(1, 17):
            assert tiles_of(count_for_tiles(n, h), h) == n
        # 17 tiles need the full 512-entry list AND an open block; without one the kernel runs 16.
        assert tiles_of(count_for_tiles(17, 1), 1) == 17
        assert tiles_of(count_for_tiles(17, 0), 0) == 16
    assert count_for_tiles(17, 1) == 512 and count_for_tiles(3, 1) == 95 and count_for_tiles(3, 0) == 96


@pytest.mark.parametrize("phase", _PHASES, ids=_PHASE_IDS)
@pytest.mark.parametrize("seq", CARRIED_STATE_SEQUENCES, ids=_seq_id)
def test_work_list_waves_land_their_tile_counts(seq, phase):
    R = 8
    reps = 2 if len(seq) < 3 else 1
    wl = build_item_sequence_work_list(seq=seq, phase=phase, resident_ctas=R, reps=reps, H=4, KH=2, dtype=torch.bfloat16, device="cpu", dead_batches=1)
    B, S, H, KH, SKV = wl.shape
    assert (B, H, KH) == (2, 4, 2) and S == wl.designed_start + len(wl.waves) * R
    tiles, dead = wl.expected_tiles, wl.expected_dead
    # Fillers: one tile each (tail-only or dead), the first designed token admits the largest count of the sequence.
    assert (tiles[0, : wl.designed_start] == 1).all()
    for w, n in enumerate(wl.waves):
        lo, hi = wl.designed_start + w * R, wl.designed_start + (w + 1) * R
        tok = torch.arange(lo, hi)
        if phase is None:
            h_tok = ((tok + 1) % 4 != 0).to(torch.long)
            want = torch.tensor([tiles_of(count_for_tiles(n, int(h)), int(h)) for h in h_tok])
            assert torch.equal(tiles[0, lo:hi], want)
            assert torch.equal(dead[0, lo:hi], (n == 0) & (h_tok == 0))
        else:
            h = 1 if phase else 0
            assert (tiles[0, lo:hi] == tiles_of(count_for_tiles(n, h), h)).all()
            assert dead[0, lo:hi].all() == ((n == 0) and phase == 0) and (dead[0, lo:hi].any() == ((n == 0) and phase == 0))
    # The KV-length-0 batch is dead everywhere, and the kernel learns that length ONLY from the tensor -- so it travels even in
    # the natural form (batch 0's entry = the full extent); the designed items' visible range is the batch length (the tail phase).
    assert dead[1].all() and wl.kv_lens[1] == 0
    assert wl.seq_kv_lens is not None and wl.seq_kv_lens.tolist() == [wl.kv_lens[0], 0]
    if phase is not None:
        assert wl.kv_lens[0] % 4 == phase and SKV == wl.kv_lens[0]
    else:
        assert wl.kv_lens[0] == SKV == S
    # Without a dead batch the natural form passes no lengths at all (a different specialization of the kernel).
    assert build_item_sequence_work_list(seq=seq, phase=None, resident_ctas=R, reps=reps, H=4, KH=2, dtype=torch.bfloat16, device="cpu").seq_kv_lens is None
    # NaN query rows: the last filler wave (or the whole filler region when it is shorter than a wave), every head, every batch.
    n_nan = min(R, wl.designed_start)
    assert int(wl.nan_tokens.sum()) == n_nan and wl.nan_tokens[wl.designed_start - n_nan : wl.designed_start].all()
    assert torch.isnan(wl.q[:, wl.nan_tokens].float()).all() and not torch.isnan(wl.q[:, ~wl.nan_tokens].float()).any()
    # The list rows hold exactly `count` live entries then -1 (the valid-prefix contract); every live id is a complete block.
    counts = torch.minimum(wl.block_lens[0].to(torch.long), torch.div(torch.arange(S) + 1, 4, rounding_mode="floor").clamp_max(wl.top_k))
    live = (wl.block_ids[0] >= 0).sum(dim=1).to(torch.long)
    assert torch.equal(live, torch.minimum(counts, torch.full_like(counts, wl.block_ids.shape[2])))
    assert int(wl.block_ids[0].max()) < SKV // 4


def test_expected_item_bounds_matches_the_kernel_header_formula():
    pos = torch.arange(0, 2064)
    kv = torch.full_like(pos, 2050)
    # block_lens = the list width: count = min(512, (pos + 1) // 4), tail = ((min(pos + 1, 2050)) % 4 != 0)
    tiles, dead = expected_item_bounds(pos, torch.full_like(pos, 512), kv, 512)
    assert tiles[0] == 1 and not dead[0], "pos 0: count 0, the open block alone"
    assert tiles[3] == 1 and not dead[3], "pos 3: one complete block, no tail -- one live tile"
    assert tiles[2047] == 16 and tiles[2048] == 17 and tiles[2049] == 17, "512 blocks + the open tail of the 2050-long range"
    assert (tiles[2050:] == 17).all(), "past the range the visible count is the length (2050 % 4 == 2: an open block)"
    # count 0: dead exactly where the visible range has no open block (pos % 4 == 3 below the length, then 2050 % 4 == 2 -> never)
    tiles_z, dead_z = expected_item_bounds(pos, torch.zeros_like(pos), kv, 512)
    assert (tiles_z == 1).all() and torch.equal(dead_z, (pos < 2050) & (pos % 4 == 3))
    tiles0, dead0 = expected_item_bounds(pos, torch.zeros_like(pos), torch.zeros_like(pos), 512)
    assert dead0.all() and (tiles0 == 1).all(), "a KV length of 0 is dead at every position, one clamped tile"


# ============================================================================ Rubin: the sequences
def _stream():
    import cuda.bindings.driver as cuda

    return cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))


def _bitwise(a: torch.Tensor, b: torch.Tensor) -> bool:
    view = torch.int16 if a.element_size() == 2 else torch.int32
    return torch.equal(a.contiguous().view(view), b.contiguous().view(view))


def _run_sequence_cell(*, seq, phase, H=24, KH=2, dtype=torch.bfloat16, seed=0):
    dev = torch.device("cuda")
    R = torch.cuda.get_device_properties(dev).multi_processor_count
    reps = 2 if len(seq) < 3 else 1
    has_dead_wave = 0 in seq
    wl = build_item_sequence_work_list(
        seq=seq,
        phase=phase,
        resident_ctas=R,
        reps=reps,
        H=H,
        KH=KH,
        dtype=dtype,
        device=dev,
        dead_batches=1 if has_dead_wave else 0,
        nan_waves=(-1,) if has_dead_wave else (),
        seed=seed,
    )
    scale = default_scale()
    outs = []
    for _ in range(2):
        o, lse = poisoned_outputs(wl)
        run_sparse_forward(wl, o, lse, scale, stream=_stream())
        outs.append((o, lse))
    (o, lse), (o2, lse2) = outs
    ref_o, ref_lse = reference_outputs(wl, scale)
    m = evaluate(wl, o, lse, ref_o, ref_lse, ATOL)
    assert not m["failures"], (wl.describe(), m)
    assert _bitwise(o, o2) and _bitwise(lse, lse2), "two launches must be bitwise"
    return wl, m


@requires_rubin
@pytest.mark.parametrize("phase", _PHASES, ids=_PHASE_IDS)
@pytest.mark.parametrize("seq", CARRIED_STATE_SEQUENCES, ids=_seq_id)
def test_sparse_core_carried_states_over_item_sequences(seq, phase):
    wl, m = _run_sequence_cell(seq=seq, phase=phase)
    d = wl.describe()
    print(
        f"\nsparse sequences {d['seq']} tail {d['phase']}: items {d['items']} (designed tiles {d['designed_tiles_hist']}, dead rows {d['dead_rows']}, "
        f"NaN tokens {d['nan_tokens']}) max|dO| {m['max_abs_o']:.5f} max|dLSE| {m['max_abs_lse']:.6f} on {m['live_rows_checked']} rows (budget {ATOL})"
    )


@requires_rubin
@pytest.mark.parametrize(
    "H, KH, dtype",
    [pytest.param(16, 1, torch.bfloat16, id="group16-one-kv-head"), pytest.param(24, 2, torch.float16, id="f16")],
)
def test_sparse_core_carried_states_other_geometries(H, KH, dtype):
    wl, m = _run_sequence_cell(seq=(3, 1, 2), phase=1, H=H, KH=KH, dtype=dtype)
    print(f"\nsparse sequences {wl.seq} tail 1 at {H}/{KH} {dtype}: max|dO| {m['max_abs_o']:.5f} max|dLSE| {m['max_abs_lse']:.6f} (budget {ATOL})")
