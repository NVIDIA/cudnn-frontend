# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Wentao Guo, Phuong Nguyen

"""Bitwise replay, batch invariance and warm stress for the SM90 BF16 grouped GEMM (forward and dgrad)."""

from __future__ import annotations

import pytest
import torch

from cudnn.tensor_adapter import has_wgmma

from determinism_utils import assert_bitwise, assert_replays, snapshot
from gemm.cutedsl.test_grouped_gemm_sm90 import check, make_problem, run

pytestmark = pytest.mark.L0

DIRECTIONS = ["fwd", "dgrad"]
SCHEDULES = ["cooperative", "pingpong"]


@pytest.fixture(autouse=True)
def require_sm90():
    if not torch.cuda.is_available() or not has_wgmma(torch.cuda.get_device_capability()[0]):
        pytest.skip("SM90 is required")


def outputs(problem, **kwargs):
    result = run(problem, **kwargs)
    return {"c": result["c_tensor"], "d": result["d_tensor"]}


def segment_problem(direction, segments, n=768, k=640):
    """Pack the given (count, seed) segments; each segment's A rows, weights, alpha and prob come from its seed alone."""
    parts = [make_problem((count,), n=n, k=k, direction=direction, seed=seed) for count, seed in segments]
    return dict(
        parts[0],
        a=torch.cat([p["a"][: p["counts"][0]] for p in parts]),
        weights=torch.cat([p["weights"] for p in parts]),
        offsets=torch.tensor([c for c, _ in segments], dtype=torch.int64).cumsum(0).to(torch.int32).cuda(),
        alpha=torch.tensor([float(1 + seed % 5) * 0.25 for _, seed in segments], device="cuda"),
        prob=torch.cat([p["prob"][: p["counts"][0]] for p in parts]),
        counts=tuple(c for c, _ in segments),
    )


def rows_of(result, segments, index):
    begin = sum(count for count, _ in segments[:index])
    return {name: value[begin : begin + segments[index][0]] for name, value in result.items()}


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("tile,cluster", [(None, None), ((64, 256), (1, 2))], ids=["default", "64x256-c12"])
def test_sm90_replay_fresh_buffers_with_concurrent_work(direction, tile, cluster):
    # Enough tiles to cross many dynamic claims; assert_replays' second-stream noise shifts the claim order.
    problem = make_problem((1024, 0, 768, 256, 2048), n=2304, k=1536, direction=direction, seed=7)
    check(problem, run(problem, mma_tiler_mn=tile, cluster_shape_mn=cluster))
    inputs = lambda: snapshot({"a": problem["a"], "weights": problem["weights"], "prob": problem["prob"]})
    original = inputs()
    assert_replays(lambda: outputs(problem, mma_tiler_mn=tile, cluster_shape_mn=cluster), repeats=8)
    assert_bitwise(inputs(), original, "input bytes")


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_sm90_batch_invariance(direction):
    target = (512, 11)
    alone = [target]
    packed = [(768, 3), target, (256, 5), (1280, 9)]
    reordered = [(1280, 9), (256, 5), target, (768, 3)]
    duplicated = [target, (256, 5), target]
    expected = snapshot(outputs(segment_problem(direction, alone)))
    for name, segments in (("packed", packed), ("reordered", reordered), ("duplicated", duplicated)):
        result = snapshot(outputs(segment_problem(direction, segments)))
        for index, segment in enumerate(segments):
            if segment == target:
                assert_bitwise(rows_of(result, segments, index), expected, f"{name} segment {index}")
    # Different neighbor data leaves the target's bytes unchanged.
    changed = snapshot(outputs(segment_problem(direction, [(768, 4), target, (256, 6)])))
    assert_bitwise(rows_of(changed, [(768, 4), target, (256, 6)], 1), expected, "changed neighbors")


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_sm90_warm_stress(direction):
    import cutlass
    from warm_stress import StressCase, assert_warm_stress

    # K=4096 refills every A/B stage many times within a tile; 8 experts span several persistent tiles per CTA.
    problem = make_problem((512,) * 8, n=2048, k=4096, direction=direction, seed=13)
    result = run(problem)
    check(problem, result)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = run(problem)
        clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(2)
        report = assert_warm_stress(
            StressCase(
                f"SM90 grouped GEMM {direction} M=8x512 N=2048 K=4096",
                launch=graph.replay,
                outputs={"c": captured["c_tensor"], "d": captured["d_tensor"]},
                thread_blocks=2 * clusters,
                refills_per_launch=4096 // 64,
            )
        )
        print(report.text())
    finally:
        graph.reset()


# The tile-info slots (2 stages) turn over once per tile. K = 64 makes each tile one k-block, so 64 experts x 256 rows at
# N = 3072 give 1536 (cooperative 128x256) or 2048 (pingpong 128x192) tiles, about 12 or 16 per CTA on H100, and the
# scheduler warp waits on an empty slot at nearly every release, so a refill that overtook a pending load would show.
TILE_INFO_STRESS = {
    "cooperative": dict(mma_tiler_mn=(128, 256), cluster_shape_mn=(2, 1)),
    "pingpong": dict(mma_tiler_mn=(128, 192), cluster_shape_mn=(2, 1)),
}


@pytest.mark.parametrize("schedule", list(TILE_INFO_STRESS))
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_sm90_tile_info_warm_stress(direction, schedule):
    import cutlass
    from warm_stress import StressCase, assert_warm_stress

    kwargs = dict(TILE_INFO_STRESS[schedule], schedule=schedule)
    problem = make_problem((256,) * 64, n=3072, k=64, direction=direction, seed=17)
    check(problem, run(problem, **kwargs))
    torch.cuda.synchronize()
    clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(2)
    tile_m, tile_n = kwargs["mma_tiler_mn"]
    tiles = (sum(problem["counts"]) // tile_m) * (problem["n"] // tile_n)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = run(problem, **kwargs)
        report = assert_warm_stress(
            StressCase(
                f"SM90 grouped GEMM {direction} {schedule} tile-info M=64x256 N=3072 K=64 ({tiles} tiles)",
                launch=graph.replay,
                outputs={"c": captured["c_tensor"], "d": captured["d_tensor"]},
                thread_blocks=2 * clusters,
                refills_per_launch=-(-tiles // (2 * clusters)),  # tile-info fills per CTA
            )
        )
        print(report.text())
    finally:
        graph.reset()


# ---- pingpong schedule (byte equality with cooperative: test_sm90_pingpong_matches_cooperative) ----


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_sm90_pingpong_replay_fresh_buffers_with_concurrent_work(direction):
    # The claim order moves tiles between CTAs and so between the two warp groups; the bytes must not move.
    kwargs = dict(mma_tiler_mn=(128, 192), cluster_shape_mn=(2, 1), schedule="pingpong")
    problem = make_problem((1024, 0, 768, 256, 2048), n=2304, k=1536, direction=direction, seed=7)
    check(problem, run(problem, **kwargs))
    inputs = lambda: snapshot({"a": problem["a"], "weights": problem["weights"], "prob": problem["prob"]})
    original = inputs()
    assert_replays(lambda: outputs(problem, **kwargs), repeats=8)
    assert_bitwise(inputs(), original, "input bytes")


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_sm90_pingpong_batch_invariance(direction):
    target = (512, 11)
    expected = snapshot(outputs(segment_problem(direction, [target]), schedule="pingpong"))
    for name, segments in (("packed", [(768, 3), target, (256, 5), (1280, 9)]), ("duplicated", [target, (256, 5), target])):
        result = snapshot(outputs(segment_problem(direction, segments), schedule="pingpong"))
        for index, segment in enumerate(segments):
            if segment == target:
                assert_bitwise(rows_of(result, segments, index), expected, f"{name} segment {index}")
