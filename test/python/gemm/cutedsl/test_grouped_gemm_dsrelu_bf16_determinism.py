# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bytewise repeatability and batch invariance of the BF16 grouped GEMM dsrelu backward (grouped_gemm_dsrelu_wrapper_sm100),
and the Rule 11 warm stress of its kernel.

Direction: backward. Outputs: dZ (d_row_tensor) and the recomputed forward d_srelu, BF16 (M, N, 1), and dprob, FP32
(M, 1, 1); rows past the routed total are never compared. The forward (C, D) is
test_grouped_gemm_srelu_bf16_determinism.py. The weight gradient is a separate API (grouped wgrad) and not covered here.

Each dZ and d_srelu element depends only on its row of G and Z_save, its expert's weights and its p. dprob of a row is
the sum over N of srelu(Z_save) * dH: with deterministic=True the kernel writes one FP32 partial per (row, N tile) and
execute() adds their strict left fold, so it depends on that row alone too; with the default atomic reduction the N
tiles add in arrival order, so atomic dprob is not bit-reproducible by design and is left out of the byte comparisons
of the atomic cell (its dZ and d_srelu are compared). The batch-invariance cases build one target expert segment (512
rows of G, Z_save and p, and its N-major weights) once and place it alone, packed between ragged neighbours with an
empty expert, with changed neighbours (other rows, other probabilities, other weights), reordered to the last expert
slot, duplicated in two slots, and inside a batch large enough that every persistent cluster runs several tiles. The
upstream gradient of the target is never changed. K = 192 is three K blocks.

N = 320 has a partial last N tile and two dprob slots (256-column N tiles). A sum of two FP32 terms has the same bits in
either order, so those cells cannot tell the fold order, the tile-to-slot mapping or the atomic reduction apart. The
deterministic cells therefore also run at N = 1856: eight slots, the last N tile 64 columns wide, where those orders
round differently, so its goldens pin the fold across revisions.

Cells: the plain squared-ReLU and the tanh-clamped one in both tanh modes ("approx", the default, and "accurate") with
deterministic dprob at N = 320 and N = 1856, and the default (approx, atomic) for dZ and d_srelu at N = 320. Replays run
with fresh output buffers (the wrapper allocates them) and concurrent work on another stream. An independent FP32
reference check (not less accurate than the native path; MUFU tanh.approx with its measured ratio reported as
APPROX_RATIO lines) is kept separate from the byte comparisons. One golden per deterministic cell
(test_dsrelu_bf16_backward_golden) records the target's bytes under CUDNN_DETERMINISM_RECORD; without a recorded golden
for this runtime it skips, except under CUDNN_DETERMINISM_GOLDENS (the strict check), where it fails.

test_dsrelu_bf16_dprob_fold_is_a_strict_left_fold runs the fold kernel alone on 3, 4, 8 and 14 slots per row (eight
and fourteen are N = 1856 and N = 3584), 805 rows (the last 256-thread block partial), mixed-magnitude slots and a
non-zero starting dprob: dprob must become, byte for byte, its start plus (((s0 + s1) + s2) + ...) in FP32, every
slot must be +0.0 afterwards, and a thread block's worth of guard rows behind both buffers must keep its bytes. On the
same inputs the reversed fold, accumulation slot by slot into dprob and (from four slots) a pairwise tree each give
different bytes, so the comparison would catch them. It compares with sequential torch additions, not a golden, so it
also runs on runtimes with no recorded goldens.

The tile-info ring and the Z_save C stage are shared-memory stages the kernel hands back to their producers (Rule 11
in python/cudnn/AGENTS.md), so they have a warm stress cell (test_dsrelu_bf16_tile_info_and_c_stage_warm_stress) on a
shape that refills both in one launch.
"""

from __future__ import annotations

import math

import pytest
import torch

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from gemm.cutedsl._workspace import ws
from gemm.cutedsl.test_grouped_gemm_srelu_bf16_utils import (
    CLAMP_SCALE,
    approx_tanh,
    assert_bitwise_equal,
    assert_not_worse_than_native,
    assert_values_equal,
    bf16_rows,
    bitwise,
    dprob_slots,
    dsrelu_api,
    dsrelu_execute,
    exact_backward_reference,
    exact_prob,
    exact_z,
    make_problem,
    mn1,
    native_backward,
    offsets_from_counts,
    persistent_clusters,
    reference_backward,
    require_bf16_srelu_gpu,
    sm_count,
)

pytestmark = pytest.mark.L0

K = 192
TARGET_ROWS = 512
EXPERTS = 4

DETERMINISTIC_CELLS = [
    pytest.param(320, None, "approx", True, id="bwd-bf16-e4-n320-k192-plain-deterministic"),
    pytest.param(320, CLAMP_SCALE, "approx", True, id="bwd-bf16-e4-n320-k192-clamped-approx-deterministic"),
    pytest.param(320, CLAMP_SCALE, "accurate", True, id="bwd-bf16-e4-n320-k192-clamped-accurate-deterministic"),
    pytest.param(1856, None, "approx", True, id="bwd-bf16-e4-n1856-k192-plain-deterministic"),
    pytest.param(1856, CLAMP_SCALE, "approx", True, id="bwd-bf16-e4-n1856-k192-clamped-approx-deterministic"),
    pytest.param(1856, CLAMP_SCALE, "accurate", True, id="bwd-bf16-e4-n1856-k192-clamped-accurate-deterministic"),
]
CELLS = DETERMINISTIC_CELLS + [pytest.param(320, CLAMP_SCALE, "approx", False, id="bwd-bf16-e4-n320-k192-clamped-approx-atomic")]

# The fold test: 3 * 256 + 37 rows leaves the fold kernel's last 256-thread block partial; GUARD_BYTE fills the rows
# behind its buffers.
FOLD_ROWS = 805
FOLD_CELLS = [pytest.param(slots, id=f"dprob-fold-fp32-m{FOLD_ROWS}-s{slots}") for slots in (3, 4, 8, 14)]
GUARD_BYTE = 0xA5


@pytest.fixture(autouse=True)
def require_sm100():
    require_bf16_srelu_gpu()


def _segment(rows, n, seed):
    """One expert segment: G rows (rows, K), Z_save rows (rows, n) and their probabilities (rows,), built once."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    grad = torch.randn((rows, K), generator=generator, device="cuda").to(torch.bfloat16)
    z_save = ((torch.rand((rows, n), generator=generator, device="cuda") * 10.0 - 5.0) * 4.0).to(torch.bfloat16)
    prob = torch.rand((rows,), generator=generator, device="cuda") * 4.3 - 1.0
    return grad, z_save, prob


def _weight(n, seed):
    """One expert's logical (n, K) weight and its N-major storage [K, n] (an FC2 weight used for its dgrad)."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    logical = (torch.randn((n, K), generator=generator, device="cuda") / math.sqrt(K)).to(torch.bfloat16)
    return logical, logical.t().contiguous()


def _batch(slots):
    """The wrapper's operands for per-expert slots ``[(segment or None, weight), ...]``, and each slot's first row."""
    counts = [0 if segment is None else segment[0].shape[0] for segment, _ in slots]
    segments = [segment for segment, _ in slots if segment is not None]
    begins = [sum(counts[:slot]) for slot in range(len(slots))]
    storages = [weight[1] for _, weight in slots]
    return {
        "a": mn1(torch.cat([segment[0] for segment in segments]).contiguous()),
        "c": mn1(torch.cat([segment[1] for segment in segments]).contiguous()),
        "prob": torch.cat([segment[2] for segment in segments]).contiguous().reshape(-1, 1, 1),
        "offsets": offsets_from_counts(counts),
        "b_ptrs": torch.tensor([storage.data_ptr() for storage in storages], dtype=torch.int64, device="cuda"),
        "weights": [weight[0] for _, weight in slots],
        "storages": storages,
        "counts": counts,
        "begins": begins,
    }


def _inputs(batch):
    return {
        "g": batch["a"][:, :, 0],
        "z_save": batch["c"][:, :, 0],
        "prob": batch["prob"],
        "offsets": batch["offsets"],
        "b_ptrs": batch["b_ptrs"],
        "b": torch.stack(batch["storages"]),
    }


def _run(batch, n, clamp_scale, tanh_mode, deterministic):
    from cudnn import grouped_gemm_dsrelu_wrapper_sm100

    return grouped_gemm_dsrelu_wrapper_sm100(
        a_tensor=batch["a"],
        c_tensor=batch["c"],
        b_ptrs=batch["b_ptrs"],
        n=n,
        b_dtype=torch.bfloat16,
        b_major="n",
        padded_offsets=batch["offsets"],
        prob_tensor=batch["prob"],
        tanh_clamp_scale=clamp_scale,
        tanh_mode=tanh_mode,
        deterministic=deterministic,
    )


def _rows(out, begin, rows, deterministic):
    """The byte-compared outputs of rows ``[begin, begin + rows)``; atomic dprob is not bit-reproducible by design."""
    result = {"d_row": out["d_row_tensor"][begin : begin + rows, :, 0], "d_srelu": out["d_srelu_tensor"][begin : begin + rows, :, 0]}
    if deterministic:
        result["dprob"] = out["dprob_tensor"][begin : begin + rows, 0, 0]
    return result


class _Pool:
    """The target segment and every neighbour at width ``n``, built once."""

    def __init__(self, n):
        self.n = n
        self.target = _segment(TARGET_ROWS, n, seed=950)
        self.w_target = _weight(n, seed=951)
        self.w = [_weight(n, seed=960 + slot) for slot in range(EXPERTS)]
        self.w_changed = [_weight(n, seed=970 + slot) for slot in range(EXPERTS)]
        self.n0, self.n3 = _segment(256, n, seed=980), _segment(768, n, seed=981)
        self.n0_changed, self.n3_changed = _segment(256, n, seed=982), _segment(768, n, seed=983)

    def alone(self):
        return _batch([(None, self.w[0]), (None, self.w[1]), (self.target, self.w_target), (None, self.w[3])])

    def big(self):
        """Neighbours for the large batch: about SM count M tiles of 256 rows (2 * SM count tiles of 256 x 256 at
        N = 320, 8 * SM count at N = 1856) over SM count / 2 clusters, so every persistent cluster runs several tiles."""
        half = sm_count() // 2
        return _segment(256 * half, self.n, seed=984), _segment(256 * (sm_count() - half), self.n, seed=985)


def _replayed_alone(pool, clamp_scale, tanh_mode, deterministic, *, repeats):
    """The target alone: the independent reference check, then its replays (fresh outputs, concurrent work)."""
    batch = pool.alone()
    problem = {"a": batch["a"], "b_logical": torch.stack(batch["weights"]), "counts": batch["counts"], "rows": TARGET_ROWS, "n": pool.n}
    out = _run(batch, pool.n, clamp_scale, tanh_mode, deterministic)
    reference = reference_backward(problem, batch["c"], batch["prob"], clamp_scale)
    native = native_backward(problem, batch["c"], batch["prob"], clamp_scale)
    report = approx_tanh(clamp_scale, tanh_mode)
    for name, key, ref_key in (("dZ", "d_row_tensor", "dz"), ("d_srelu", "d_srelu_tensor", "h"), ("dprob", "dprob_tensor", "dprob")):
        assert_not_worse_than_native(name, out[key], reference[ref_key], native[ref_key], routed=TARGET_ROWS, report_ratio=report)
    alone = assert_replays(lambda: _rows(_run(batch, pool.n, clamp_scale, tanh_mode, deterministic), 0, TARGET_ROWS, deterministic), repeats=repeats)
    return batch, alone


@pytest.mark.parametrize("n, clamp_scale, tanh_mode, deterministic", CELLS)
def test_dsrelu_bf16_backward_replay_and_batch_invariance(n, clamp_scale, tanh_mode, deterministic):
    """Replays of a scheduling-heavy batch are byte-identical; the target segment's dZ, d_srelu and (deterministic)
    dprob bytes are the same alone, packed between ragged neighbours with an empty expert, with changed neighbours,
    reordered, duplicated and in the large batch; no call mutates its inputs."""
    pool = _Pool(n)
    _, alone = _replayed_alone(pool, clamp_scale, tanh_mode, deterministic, repeats=4)
    w, wc, t, wt = pool.w, pool.w_changed, pool.target, pool.w_target
    big0, big3 = pool.big()
    arrangements = {
        "packed": ([(pool.n0, w[0]), (None, w[1]), (t, wt), (pool.n3, w[3])], [2]),
        "changed-neighbours": ([(pool.n0_changed, wc[0]), (None, wc[1]), (t, wt), (pool.n3_changed, wc[3])], [2]),
        "reordered": ([(pool.n3, w[3]), (pool.n0, w[0]), (None, w[1]), (t, wt)], [3]),
        "duplicated": ([(t, wt), (pool.n0, w[0]), (t, wt), (None, w[3])], [0, 2]),
        "large": ([(big0, w[0]), (None, w[1]), (t, wt), (big3, w[3])], [2]),
    }
    for name, (slots, target_slots) in arrangements.items():
        batch = _batch(slots)
        before = snapshot(_inputs(batch))
        out = _run(batch, n, clamp_scale, tanh_mode, deterministic)
        for slot in target_slots:
            assert_bitwise(snapshot(_rows(out, batch["begins"][slot], TARGET_ROWS, deterministic)), alone, f"{name}, expert slot {slot}")
        assert_bitwise(snapshot(_inputs(batch)), before, f"{name}: inputs mutated")

    large = _batch(arrangements["large"][0])
    routed = sum(large["counts"])
    assert_replays(lambda: _rows(_run(large, n, clamp_scale, tanh_mode, deterministic), 0, routed, deterministic), repeats=8)


@pytest.mark.parametrize("n, clamp_scale, tanh_mode, deterministic", DETERMINISTIC_CELLS)
def test_dsrelu_bf16_backward_golden(n, clamp_scale, tanh_mode, deterministic):
    """One golden per deterministic cell, on the target segment alone; recorded only after its replay and the
    reference check pass (determinism_goldens/README.md). A runtime with no recorded baseline skips this test alone."""
    pool = _Pool(n)
    batch, alone = _replayed_alone(pool, clamp_scale, tanh_mode, deterministic, repeats=3)
    activation = "plain" if clamp_scale is None else f"clamped{int(clamp_scale)}-{tanh_mode}"
    assert_golden(
        f"grouped-gemm-dsrelu-bf16-bwd-discrete-nmajor-e{EXPERTS}-n{n}-k{K}-m{TARGET_ROWS}-{activation}-deterministic",
        {"g": batch["a"][:, :, 0], "z_save": batch["c"][:, :, 0], "prob": batch["prob"], "b": pool.w_target[1], "offsets": batch["offsets"]},
        alone,
    )


def _fold_operands(rows, slots, seed):
    """FP32 slots ``(rows, slots, 1)`` and a starting dprob ``(rows, 1, 1)``, every value randn * 2^e with e in
    [-24, 24], drawn on the CPU so every GPU and runtime folds the same values."""
    generator = torch.Generator().manual_seed(seed)

    def mixed(shape):
        mantissa = torch.randn(shape, generator=generator, dtype=torch.float32)
        return torch.ldexp(mantissa, torch.randint(-24, 25, shape, generator=generator))

    return mixed((rows, slots, 1)).cuda(), mixed((rows, 1, 1)).cuda()


def _guarded(values, guard_rows):
    """``(copy, guard)``: a copy of ``values`` at the front of a CUDA buffer that has ``guard_rows`` more rows of
    GUARD_BYTE behind it. The copy is contiguous at storage offset 0, as fold_dprob_slots's flat views need."""
    buffer = torch.empty((values.shape[0] + guard_rows) * values[0].numel(), dtype=values.dtype, device=values.device)
    buffer.view(torch.uint8).fill_(GUARD_BYTE)
    copy = buffer[: values.numel()].view(values.shape)
    copy.copy_(values)
    return copy, buffer[values.numel() :]


def _left_fold(terms):
    """``((t0 + t1) + t2) + ...`` over dim 1, one FP32 addition at a time."""
    total = terms[:, 0]
    for j in range(1, terms.shape[1]):
        total = total + terms[:, j]
    return total


def _pairwise_tree(terms):
    """``(t0 + t1) + (t2 + t3)`` and so on, level by level; an odd last term moves up a level unchanged."""
    level = [terms[:, j] for j in range(terms.shape[1])]
    while len(level) > 1:
        level = [level[i] + level[i + 1] for i in range(0, len(level) - 1, 2)] + level[len(level) - len(level) % 2 :]
    return level[0]


@pytest.mark.parametrize("num_slots", FOLD_CELLS)
def test_dsrelu_bf16_dprob_fold_is_a_strict_left_fold(num_slots):
    """The deterministic dprob fold alone: ``dprob += (((s0 + s1) + s2) + ...)`` per row, the bytes of sequential FP32
    additions, then every slot +0.0; folding the zeroed slots again leaves dprob unchanged. A thread block's worth of
    guard rows behind the slots and behind dprob keeps its bytes, so the partial last block stores nothing past either
    end. On these inputs the reversed fold, accumulation slot by slot into dprob and (from four slots) a pairwise tree
    each give other bytes, so the comparison would catch any of them."""
    from cuda.bindings import driver as cuda

    from cudnn.gemm.cutedsl.grouped.dsrelu._dprob_fold import _THREADS, fold_dprob_slots

    assert FOLD_ROWS % _THREADS, "the row count must leave the fold's last thread block partial"
    slot_values, start = _fold_operands(FOLD_ROWS, num_slots, seed=1200 + num_slots)
    terms, start_rows = slot_values[:, :, 0], start[:, 0, 0]
    expected = start_rows + _left_fold(terms)
    others = {
        "the reversed fold": start_rows + _left_fold(terms.flip(1)),
        "accumulation slot by slot into dprob": _left_fold(torch.cat([start, slot_values], dim=1)[:, :, 0]),
    }
    if num_slots >= 4:
        others["a pairwise tree"] = start_rows + _pairwise_tree(terms)
    for name, other in others.items():
        assert not torch.equal(bitwise(other), bitwise(expected)), f"{name} has the left fold's bytes on every row: the test could not tell them apart"

    slots, slots_guard = _guarded(slot_values, _THREADS)
    dprob, dprob_guard = _guarded(start, _THREADS)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    for launch in ("the fold", "folding the zeroed slots again"):
        fold_dprob_slots(slots, dprob, stream)
        assert_bitwise_equal(f"dprob after {launch}", dprob[:, 0, 0], expected)
        assert not slots.view(torch.int32).any(), f"{launch} must leave every slot +0.0"
        for name, guard in (("slots", slots_guard), ("dprob", dprob_guard)):
            assert bool((guard.view(torch.uint8) == GUARD_BYTE).all()), f"{launch} stored past the end of {name}"


@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
def test_dsrelu_bf16_tile_info_and_c_stage_warm_stress():
    """Rule 11 at run time for the backward kernel's tile-info ring (sInfo: two stages released through
    consume_then_arrive by the TMA, MMA, epilogue and Z_save-load warps) and its Z_save C stage (two 128 x 32 stages the
    epilogue reads with ordinary shared-memory loads and releases through consume_then_arrive, refilled by TMA): the
    shipping 256 x 256 2-CTA configuration, discrete N-major weights, deterministic dprob (kernel and fold, so every
    output is bit-reproducible), on a persistent grid where every cluster runs about four tiles of eight 32-column
    subtiles. warm_stress.assert_warm_stress launches it back to back and behind a shared-memory poison launch and
    compares every dZ, d_srelu and dprob byte with an independent float64 reference: exact-valued operands (odd,
    never-zero dH, Z_save in {1, 2, 3}, positive power-of-two probabilities), and dprob is the helper's sentinel fill
    plus the exact row sum, one FP32 addition, because execute() accumulates into it. The kernel runs one thread block
    per SM (its shared memory), which the report flags: a zero is a rate bound for this shape on this GPU, behind the
    cluster-2 control; the proof of the order is the rendering pin,
    test_grouped_gemm_dsrelu_bf16.py::test_dsrelu_bf16_rendering_orders_every_release."""
    from warm_stress import SENTINEL_BYTE, StressCase, assert_warm_stress, stress_validation

    n, k = 512, 64
    clusters = persistent_clusters((2, 1))
    m_tiles = 2 * clusters  # 256-row tiles; with two 256-column N tiles, four tiles per cluster
    counts = [256 * (m_tiles // 4), 0, 256 * (m_tiles // 2)]
    counts.append(256 * m_tiles - sum(counts))
    problem = make_problem(counts, n, k, discrete=True, b_major="n", exact=True, seed=991)
    rows = problem["rows"]
    z_save = exact_z(rows, n, seed=992, positive=True)
    prob = exact_prob(rows, seed=993, positive=True)
    d_row, d_srelu = bf16_rows(rows, n), bf16_rows(rows, n)
    dprob = torch.zeros((rows, 1, 1), dtype=torch.float32, device="cuda")
    slots = dprob_slots(rows, n)
    api = dsrelu_api(problem, z_save, prob, d_row, dprob, d_srelu=d_srelu, dprob_workspace=slots, deterministic=True)
    assert api.check_support()
    api.compile()
    workspace = ws(api)

    def launch():
        dsrelu_execute(api, problem, z_save, prob, d_row, dprob, d_srelu=d_srelu, dprob_workspace=slots, workspace=workspace)

    reference = exact_backward_reference(problem["a"], problem["b_logical"], counts, z_save, prob)
    sentinel = torch.empty(rows, dtype=torch.float32, device="cuda")
    sentinel.view(torch.uint8).fill_(SENTINEL_BYTE)
    dprob_reference = (sentinel + reference["dprob"]).contiguous()
    dprob.reshape(-1).view(torch.uint8).fill_(SENTINEL_BYTE)
    launch()
    assert_values_equal("dZ", d_row[:, :, 0], reference["d_row"])
    assert_values_equal("d_srelu", d_srelu[:, :, 0], reference["d_srelu"])
    assert_values_equal("dprob (sentinel + sum)", dprob[:, 0, 0], dprob_reference)
    assert not slots.any(), "execute() must leave the dprob slots zeroed"
    tiles = m_tiles * (n // 256)
    tile_info_refills = 2 * (tiles - clusters)  # per CTA: one fill per tile plus the end marker, minus the two fresh stages
    c_stage_refills = 2 * (8 * tiles - 2 * clusters)  # per CTA: eight 128 x 32 Z_save subtiles per tile, minus the two fresh stages
    case = StressCase(
        f"BF16 dsrelu backward tile-info ring ({tile_info_refills} refills) and Z_save C stage ({c_stage_refills} refills) "
        f"(discrete N-major, 256x256 2-CTA, deterministic dprob, N {n}, {tiles} tiles over {clusters} clusters)",
        launch,
        {"d_row": d_row, "d_srelu": d_srelu, "dprob": dprob},
        thread_blocks=2 * clusters,
        refills_per_launch=min(tile_info_refills, c_stage_refills),
        reference={"d_row": reference["d_row"].contiguous(), "d_srelu": reference["d_srelu"].contiguous(), "dprob": dprob_reference},
    )
    report = assert_warm_stress(case, validation=stress_validation(d_row.device, cluster=2), warm=1000, poisoned=1000)
    print("\n" + report.text())
    assert not slots.any(), "the stress must leave the dprob slots zeroed"
