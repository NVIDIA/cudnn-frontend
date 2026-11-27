# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Wentao Guo, Phuong Nguyen

"""Bitwise replay, batch invariance, and warm stress for the SM90 BF16 grouped GEMM GLU and dGLU.

dGLU's dprob is an atomic sum across N tiles, so only its D (dgate | dup) is held bitwise.
"""

from __future__ import annotations


import pytest
import torch

from cudnn.tensor_adapter import has_wgmma

from determinism_utils import assert_bitwise, assert_replays, snapshot
from gemm.cutedsl.test_grouped_gemm_glu_sm90 import check_dglu, check_glu, make_problem, run_dglu, run_glu, to_gate_then_up

pytestmark = pytest.mark.L0

OPS = ["glu", "dglu"]


@pytest.fixture(autouse=True)
def require_sm90():
    if not torch.cuda.is_available() or not has_wgmma(torch.cuda.get_device_capability()[0]):
        pytest.skip("SM90 is required")


def as_gate_then_up(op, problem):
    """The same problem with GLU's weights or dGLU's saved C stored gate_then_up."""
    if op == "glu":
        return dict(problem, weights=to_gate_then_up(problem["weights"], 1))
    return dict(problem, c=to_gate_then_up(problem["c"], 1))


def outputs(op, problem, **kwargs):
    if kwargs.get("glu_layout") == "gate_then_up":
        problem = as_gate_then_up(op, problem)
    if op == "glu":
        result = run_glu(problem, **kwargs)
        return {"c": result["c_tensor"], "d": result["d_tensor"]} if kwargs.get("generate_c", True) else {"d": result["d_tensor"]}
    return {"d": run_dglu(problem, **kwargs)["d_row_tensor"]}


def check(op, problem, **kwargs):
    if op == "glu":
        check_glu(problem, run_glu(problem, **kwargs), with_c=kwargs.get("generate_c", True))
    else:
        check_dglu(problem, run_dglu(problem, **kwargs))


def segment_problem(segments, n=768, k=640):
    """Pack the given (count, seed) segments; each segment's rows, weights, alpha, beta and prob come from its seed alone."""
    parts = [make_problem((count,), n=n, k=k, seed=seed) for count, seed in segments]
    return dict(
        parts[0],
        a=torch.cat([p["a"][: p["counts"][0]] for p in parts]),
        c=torch.cat([p["c"][: p["counts"][0]] for p in parts]),
        weights=torch.cat([p["weights"] for p in parts]),
        offsets=torch.tensor([c for c, _ in segments], dtype=torch.int64).cumsum(0).to(torch.int32).cuda(),
        alpha=torch.tensor([float(1 + seed % 5) * 0.25 for _, seed in segments], device="cuda"),
        beta=torch.tensor([float(1 + seed % 3) * 0.5 for _, seed in segments], device="cuda"),
        prob=torch.cat([p["prob"][: p["counts"][0]] for p in parts]),
        counts=tuple(c for c, _ in segments),
    )


def rows_of(result, segments, index):
    begin = sum(count for count, _ in segments[:index])
    return {name: value[begin : begin + segments[index][0]] for name, value in result.items()}


@pytest.mark.parametrize("op", OPS)
def test_sm90_glu_replay_fresh_buffers_with_concurrent_work(op):
    # Enough tiles to cross many dynamic claims; assert_replays' second-stream noise shifts the claim order.
    problem = make_problem((1024, 0, 768, 256, 2048), n=2304, k=1536, seed=7)
    check(op, problem)
    inputs = lambda: snapshot({"a": problem["a"], "weights": problem["weights"], "prob": problem["prob"], "c": problem["c"]})
    original = inputs()
    assert_replays(lambda: outputs(op, problem), repeats=8)
    assert_bitwise(inputs(), original, "input bytes")


@pytest.mark.parametrize("op", OPS)
def test_sm90_glu_batch_invariance(op):
    target = (512, 11)
    alone = [target]
    packed = [(768, 3), target, (256, 5), (1280, 9)]
    reordered = [(1280, 9), (256, 5), target, (768, 3)]
    duplicated = [target, (256, 5), target]
    expected = snapshot(outputs(op, segment_problem(alone)))
    for name, segments in (("packed", packed), ("reordered", reordered), ("duplicated", duplicated)):
        result = snapshot(outputs(op, segment_problem(segments)))
        for index, segment in enumerate(segments):
            if segment == target:
                assert_bitwise(rows_of(result, segments, index), expected, f"{name} segment {index}")
    changed = snapshot(outputs(op, segment_problem([(768, 4), target, (256, 6)])))
    assert_bitwise(rows_of(changed, [(768, 4), target, (256, 6)], 1), expected, "changed neighbors")


@pytest.mark.parametrize("op", OPS)
def test_sm90_glu_warm_stress(op):
    import cutlass
    from warm_stress import StressCase, assert_warm_stress

    # K=4096 refills every A/B stage many times within a tile; 8 experts span several persistent tiles per CTA.
    problem = make_problem((512,) * 8, n=2048, k=4096, seed=13)
    check(op, problem)
    torch.cuda.synchronize()
    dprob = torch.zeros_like(problem["prob"])
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = outputs(op, problem) if op == "glu" else {"d": run_dglu(problem, dprob=dprob)["d_row_tensor"]}
        clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(2)
        report = assert_warm_stress(
            StressCase(
                f"SM90 grouped GEMM {op} M=8x512 N=2048 K=4096",
                launch=graph.replay,
                outputs=captured,
                thread_blocks=2 * clusters,
                refills_per_launch=4096 // 64,
            )
        )
        print(report.text())
    finally:
        graph.reset()


# The tile-info slots (2 stages) turn over once per tile, dGLU's C stages once per C subtile. K = 64 makes each tile one
# k-block: 64 experts x 256 rows at N = 3072 give 1536 (cooperative 128x256) or 2048 (pingpong 128x192) tiles, about 12
# or 16 per CTA on H100, so the producers wait on an empty stage at nearly every release, and a refill that overtook a
# pending load would show.
TILE_INFO_STRESS = {
    "glu-cooperative": ("glu", dict(schedule="cooperative", mma_tiler_mn=(128, 256), cluster_shape_mn=(2, 1))),
    "glu-pingpong": ("glu", dict(schedule="pingpong", mma_tiler_mn=(128, 192), cluster_shape_mn=(2, 1))),
    "dglu": ("dglu", dict(mma_tiler_mn=(128, 256), cluster_shape_mn=(2, 1))),
}


@pytest.mark.parametrize("case", list(TILE_INFO_STRESS))
def test_sm90_glu_tile_info_warm_stress(case):
    import cutlass
    from warm_stress import StressCase, assert_warm_stress

    op, kwargs = TILE_INFO_STRESS[case]
    problem = make_problem((256,) * 64, n=3072, k=64, seed=17)
    check(op, problem, **kwargs)
    torch.cuda.synchronize()
    clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(2)
    tile_m, tile_n = kwargs["mma_tiler_mn"]
    tiles = (sum(problem["counts"]) // tile_m) * (problem["n"] // tile_n)
    dprob = torch.zeros_like(problem["prob"])
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = outputs(op, problem, **kwargs) if op == "glu" else {"d": run_dglu(problem, dprob=dprob, **kwargs)["d_row_tensor"]}
        report = assert_warm_stress(
            StressCase(
                f"SM90 grouped GEMM {case} tile-info{' and C' if op == 'dglu' else ''} M=64x256 N=3072 K=64 ({tiles} tiles)",
                launch=graph.replay,
                outputs=captured,
                thread_blocks=2 * clusters,
                refills_per_launch=-(-tiles // (2 * clusters)),  # tile-info fills per CTA (dGLU's C stages: per subtile)
            )
        )
        print(report.text())
    finally:
        graph.reset()


# ---- GLU forward, pingpong schedule: 128x192 on (2, 1) clusters without the saved C ----

PINGPONG = dict(schedule="pingpong", generate_c=False)


def test_sm90_glu_pingpong_replay_fresh_buffers_with_concurrent_work():
    problem = make_problem((1024, 0, 768, 256, 2048), n=2304, k=1536, seed=7)
    check("glu", problem, **PINGPONG)
    assert_replays(lambda: outputs("glu", problem, **PINGPONG), repeats=8)


def test_sm90_glu_pingpong_batch_invariance():
    target = (512, 11)
    expected = snapshot(outputs("glu", segment_problem([target]), **PINGPONG))
    segments = [(768, 3), target, (256, 5), target]
    result = snapshot(outputs("glu", segment_problem(segments), **PINGPONG))
    for index in (1, 3):
        assert_bitwise(rows_of(result, segments, index), expected, f"packed segment {index}")


# ---- glu_layout="gate_then_up" ----

GATE_THEN_UP = {
    "glu-cooperative": ("glu", dict(schedule="cooperative", glu_layout="gate_then_up")),
    "glu-pingpong": ("glu", dict(schedule="pingpong", mma_tiler_mn=(128, 192), cluster_shape_mn=(2, 1), glu_layout="gate_then_up")),
    "dglu": ("dglu", dict(glu_layout="gate_then_up")),
}


def interleaved_bytes(op, result):
    """gate_then_up outputs in interleaved_32 order (GLU's D has no branches)."""
    inverse = lambda x: x.reshape(x.shape[0], 2, -1, 32).transpose(1, 2).reshape(x.shape[0], -1)  # noqa: E731
    return {name: value if (op == "glu" and name == "d") else inverse(value.view(value.shape[0], -1)) for name, value in result.items()}


@pytest.mark.parametrize("case", list(GATE_THEN_UP))
def test_sm90_glu_gate_then_up_replay_fresh_buffers_with_concurrent_work(case):
    # Replays under second-stream noise are bitwise equal, and equal to interleaved_32's bytes in branch order.
    op, kwargs = GATE_THEN_UP[case]
    problem = make_problem((1024, 0, 768, 256, 2048), n=2304, k=1536, seed=7)
    interleaved_kwargs = {name: value for name, value in kwargs.items() if name != "glu_layout"}
    rows = sum(problem["counts"])
    expected = {name: value[:rows] for name, value in snapshot(outputs(op, problem, **interleaved_kwargs)).items()}
    replayed = assert_replays(lambda: outputs(op, problem, **kwargs), repeats=8)
    assert_bitwise(interleaved_bytes(op, {name: value[:rows] for name, value in replayed.items()}), expected, "gate_then_up vs interleaved_32")


@pytest.mark.parametrize("case", list(GATE_THEN_UP))
def test_sm90_glu_gate_then_up_batch_invariance(case):
    op, kwargs = GATE_THEN_UP[case]
    target = (512, 11)
    expected = snapshot(outputs(op, segment_problem([target]), **kwargs))
    for name, segments in (("packed", [(768, 3), target, (256, 5), target]), ("reordered", [(1280, 9), target, (768, 4)])):
        result = snapshot(outputs(op, segment_problem(segments), **kwargs))
        for index, segment in enumerate(segments):
            if segment == target:
                assert_bitwise(rows_of(result, segments, index), expected, f"{name} segment {index}")
