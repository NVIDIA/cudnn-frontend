# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bytewise repeatability and batch invariance of the BF16 grouped GEMM srelu forward (grouped_gemm_srelu_wrapper_sm100),
and the Rule 11 warm stress of its kernel.

Direction: forward. Outputs: C = bf16(A @ B_e^T) (the saved pre-activation) and D = bf16(p * srelu(Z)), both BF16
(M, N, 1); rows past the routed total are never written and never compared. This forward slice has no
gradient outputs; BF16 backward coverage belongs to the dependent dSReLU change.

Every output element depends only on its row of A, its expert's weights and its p: an expert segment starts at a
256-aligned row, so its M tiles and the fixed N and K tiling do not depend on where it sits or on its neighbours. The
batch-invariance cases therefore build one target expert segment (512 rows, its weights and probabilities) once and
place it alone, packed between ragged neighbours with an empty expert, with changed neighbours (other rows, other
probabilities, other weights), reordered to the last expert slot, duplicated in two slots, and inside a batch large
enough that every persistent cluster runs several tiles. Its C and D bytes must not change. Weights are discrete (one
pointer per expert slot), so a moved or duplicated segment keeps its weights. N = 320 has a partial last N tile; K =
192 is three K blocks.

Cells: the plain squared-ReLU, and the tanh-clamped one in both tanh modes ("approx", the default, and "accurate"),
each with act_input "acc" and "saved" (plain once per act_input: plain mode ignores tanh_mode). Replays run with
fresh output buffers (the wrapper allocates them) and concurrent work on another stream (determinism_utils.
assert_replays). An independent FP32 reference check (not less accurate than the native path; MUFU tanh.approx with
its measured ratio reported as an APPROX_RATIO line) is kept separate from the byte comparisons. One golden per cell
(test_srelu_bf16_forward_golden) records the target's bytes under CUDNN_DETERMINISM_RECORD; without a recorded golden
for this runtime it skips, except under CUDNN_DETERMINISM_GOLDENS (the strict check), where it fails.

The tile-info ring is a shared-memory stage the kernel hands back to its producer (Rule 11 in python/cudnn/AGENTS.md),
so it has a warm stress cell (test_srelu_bf16_tile_info_ring_warm_stress) on a shape that refills it in one launch.
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
    assert_not_worse_than_native,
    assert_values_equal,
    bf16_rows,
    exact_forward_reference,
    exact_prob,
    make_problem,
    mn1,
    native_forward,
    offsets_from_counts,
    persistent_clusters,
    reference_forward,
    require_bf16_srelu_gpu,
    sm_count,
    srelu_api,
    srelu_execute,
)

pytestmark = pytest.mark.L0

N, K = 320, 192
TARGET_ROWS = 512
EXPERTS = 4

CELLS = [
    pytest.param(None, "approx", "acc", id="fwd-bf16-e4-n320-k192-plain-acc"),
    pytest.param(None, "approx", "saved", id="fwd-bf16-e4-n320-k192-plain-saved"),
    pytest.param(CLAMP_SCALE, "approx", "acc", id="fwd-bf16-e4-n320-k192-clamped-approx-acc"),
    pytest.param(CLAMP_SCALE, "approx", "saved", id="fwd-bf16-e4-n320-k192-clamped-approx-saved"),
    pytest.param(CLAMP_SCALE, "accurate", "acc", id="fwd-bf16-e4-n320-k192-clamped-accurate-acc"),
    pytest.param(CLAMP_SCALE, "accurate", "saved", id="fwd-bf16-e4-n320-k192-clamped-accurate-saved"),
]


@pytest.fixture(autouse=True)
def require_sm100():
    require_bf16_srelu_gpu()


def _segment(rows, seed):
    """One expert segment: A rows (rows, K) and their probabilities (rows,), built once and reused unchanged."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    a = (torch.randn((rows, K), generator=generator, device="cuda") * 8.0).to(torch.bfloat16)
    prob = torch.rand((rows,), generator=generator, device="cuda") * 4.3 - 1.0
    return a, prob


def _weight(seed):
    """One expert's (N, K) weight, stored K-major (b_major "k")."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn((N, K), generator=generator, device="cuda") / math.sqrt(K)).to(torch.bfloat16)


def _batch(slots):
    """The wrapper's operands for per-expert slots ``[(segment or None, weight), ...]``, and each slot's first row."""
    counts = [0 if segment is None else segment[0].shape[0] for segment, _ in slots]
    segments = [segment for segment, _ in slots if segment is not None]
    a = torch.cat([segment[0] for segment in segments]).contiguous()
    prob = torch.cat([segment[1] for segment in segments]).contiguous()
    weights = [weight for _, weight in slots]
    begins = [sum(counts[:slot]) for slot in range(len(slots))]
    return {
        "a": mn1(a),
        "prob": prob.reshape(-1, 1, 1),
        "offsets": offsets_from_counts(counts),
        "b_ptrs": torch.tensor([weight.data_ptr() for weight in weights], dtype=torch.int64, device="cuda"),
        "weights": weights,
        "counts": counts,
        "begins": begins,
    }


def _inputs(batch):
    return {"a": batch["a"][:, :, 0], "prob": batch["prob"], "offsets": batch["offsets"], "b_ptrs": batch["b_ptrs"], "b": torch.stack(batch["weights"])}


def _run(batch, clamp_scale, tanh_mode, act_input):
    from cudnn import grouped_gemm_srelu_wrapper_sm100

    out = grouped_gemm_srelu_wrapper_sm100(
        a_tensor=batch["a"],
        b_ptrs=batch["b_ptrs"],
        n=N,
        b_dtype=torch.bfloat16,
        padded_offsets=batch["offsets"],
        prob_tensor=batch["prob"],
        tanh_clamp_scale=clamp_scale,
        tanh_mode=tanh_mode,
        act_input=act_input,
    )
    return out["c_tensor"], out["d_tensor"]


def _rows(outputs, begin, rows):
    c, d = outputs
    return {"c": c[begin : begin + rows, :, 0], "d": d[begin : begin + rows, :, 0]}


class _Pool:
    """The target segment and every neighbour, built once."""

    def __init__(self):
        self.target = _segment(TARGET_ROWS, seed=900)
        self.w_target = _weight(seed=901)
        self.w = [_weight(seed=910 + slot) for slot in range(EXPERTS)]
        self.w_changed = [_weight(seed=920 + slot) for slot in range(EXPERTS)]
        self.n0, self.n3 = _segment(256, seed=930), _segment(768, seed=931)
        self.n0_changed, self.n3_changed = _segment(256, seed=932), _segment(768, seed=933)

    def alone(self):
        return _batch([(None, self.w[0]), (None, self.w[1]), (self.target, self.w_target), (None, self.w[3])])

    def big(self):
        """Neighbours for the large batch: about 2 * SM count tiles of 256 x 256 over SM count / 2 clusters, so every
        persistent cluster runs several tiles."""
        half = sm_count() // 2
        return _segment(256 * half, seed=934), _segment(256 * (sm_count() - half), seed=935)


def _replayed_alone(pool, clamp_scale, tanh_mode, act_input, *, repeats):
    """The target alone: its replays (fresh outputs, concurrent work) and the independent reference check."""
    batch = pool.alone()
    alone = assert_replays(lambda: _rows(_run(batch, clamp_scale, tanh_mode, act_input), 0, TARGET_ROWS), repeats=repeats)
    problem = {"a": batch["a"], "b_logical": torch.stack(batch["weights"]), "counts": batch["counts"], "rows": TARGET_ROWS, "n": N}
    z_ref, h_ref = reference_forward(problem, batch["prob"], clamp_scale)
    z_nat, h_nat = native_forward(problem, batch["prob"], clamp_scale)
    assert_not_worse_than_native("C", alone["c"], z_ref, z_nat, routed=TARGET_ROWS)
    assert_not_worse_than_native("D", alone["d"], h_ref, h_nat, routed=TARGET_ROWS, report_ratio=approx_tanh(clamp_scale, tanh_mode))
    return batch, alone


@pytest.mark.parametrize("clamp_scale, tanh_mode, act_input", CELLS)
def test_srelu_bf16_forward_replay_and_batch_invariance(clamp_scale, tanh_mode, act_input):
    """Replays of a scheduling-heavy batch are byte-identical; the target segment's C and D bytes are the same alone,
    packed between ragged neighbours with an empty expert, with changed neighbours, reordered, duplicated and in the
    large batch; no call mutates its inputs."""
    pool = _Pool()
    _, alone = _replayed_alone(pool, clamp_scale, tanh_mode, act_input, repeats=4)
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
        outputs = _run(batch, clamp_scale, tanh_mode, act_input)
        for slot in target_slots:
            assert_bitwise(snapshot(_rows(outputs, batch["begins"][slot], TARGET_ROWS)), alone, f"{name}, expert slot {slot}")
        assert_bitwise(snapshot(_inputs(batch)), before, f"{name}: inputs mutated")

    large = _batch(arrangements["large"][0])
    routed = sum(large["counts"])
    assert_replays(lambda: _rows(_run(large, clamp_scale, tanh_mode, act_input), 0, routed), repeats=8)


@pytest.mark.parametrize("clamp_scale, tanh_mode, act_input", CELLS)
def test_srelu_bf16_forward_golden(clamp_scale, tanh_mode, act_input):
    """One golden per cell, on the target segment alone; recorded only after its replay and the reference check pass
    (determinism_goldens/README.md). A runtime with no recorded baseline skips this test alone (a coverage gap)."""
    pool = _Pool()
    batch, alone = _replayed_alone(pool, clamp_scale, tanh_mode, act_input, repeats=3)
    activation = "plain" if clamp_scale is None else f"clamped{int(clamp_scale)}-{tanh_mode}"
    assert_golden(
        f"grouped-gemm-srelu-bf16-fwd-discrete-e{EXPERTS}-n{N}-k{K}-m{TARGET_ROWS}-{activation}-{act_input}",
        {"a": batch["a"][:, :, 0], "prob": batch["prob"], "b": pool.w_target, "offsets": batch["offsets"]},
        alone,
    )


@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
def test_srelu_bf16_tile_info_ring_warm_stress():
    """Rule 11 at run time for the forward kernel's tile-info ring (sInfo: two stages, released through
    consume_then_arrive by the TMA, MMA and epilogue warps, refilled by the scheduler warp): the shipping 256 x 256
    2-CTA discrete configuration on a persistent grid where every cluster runs about four tiles, so each CTA's ring is
    refilled inside one launch. warm_stress.assert_warm_stress launches it back to back and behind a shared-memory
    poison launch and compares every C and D byte with an independent float64 reference (exact-valued operands: odd,
    never-zero accumulators and positive power-of-two probabilities, so the reference has the kernel's bytes, signed
    zeros included). The kernel runs one thread block per SM (its shared memory), which the report flags: a zero is a
    rate bound for this shape on this GPU, behind the cluster-2 control; the proof of the order is the rendering
    pin, test_grouped_gemm_srelu_bf16.py::test_srelu_bf16_rendering_orders_every_release."""
    from warm_stress import StressCase, assert_warm_stress, stress_validation

    n, k = 512, 64
    clusters = persistent_clusters((2, 1))
    m_tiles = 2 * clusters  # 256-row tiles; with two 256-column N tiles, four tiles per cluster
    counts = [256 * (m_tiles // 4), 0, 256 * (m_tiles // 2)]
    counts.append(256 * m_tiles - sum(counts))
    problem = make_problem(counts, n, k, discrete=True, exact=True, seed=941)
    rows = problem["rows"]
    prob = exact_prob(rows, seed=942, positive=True)
    c, d = bf16_rows(rows, n), bf16_rows(rows, n)
    api = srelu_api(problem, prob, c, d)
    assert api.check_support()
    api.compile()
    workspace = ws(api)

    def launch():
        srelu_execute(api, problem, prob, c, d, workspace=workspace)

    ref_c, ref_d = exact_forward_reference(problem["a"], problem["b_logical"], counts, prob)
    launch()
    assert_values_equal("C", c[:, :, 0], ref_c)
    assert_values_equal("D", d[:, :, 0], ref_d)
    tiles = m_tiles * (n // 256)
    refills = 2 * (tiles - clusters)  # per CTA: one fill per tile plus the end marker, minus the two fresh stages
    case = StressCase(
        f"BF16 srelu forward tile-info ring (discrete, 256x256 2-CTA, N {n}, {tiles} tiles over {clusters} clusters)",
        launch,
        {"c": c, "d": d},
        thread_blocks=2 * clusters,
        refills_per_launch=refills,
        reference={"c": ref_c.contiguous(), "d": ref_d.contiguous()},
    )
    report = assert_warm_stress(case, validation=stress_validation(c.device, cluster=2), warm=1000, poisoned=1000)
    print("\n" + report.text())
