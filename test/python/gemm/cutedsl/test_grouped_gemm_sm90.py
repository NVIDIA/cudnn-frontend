# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Wentao Guo, Phuong Nguyen

"""SM90 (Hopper) coverage for grouped_gemm_wrapper_sm90 / GroupedGemmSm90: forward (K-major B) and dgrad (N-major B)."""

from __future__ import annotations

import contextlib

import pytest
import torch

from cudnn.tensor_adapter import has_wgmma

pytestmark = pytest.mark.L0


@contextlib.contextmanager
def override_schedule(schedule):
    """Force the SM90 BF16 grouped GEMMs' warp-group schedule ("cooperative" / "pingpong"); None keeps the heuristic."""
    import cudnn.gemm.cutedsl.grouped.unfused.moe_grouped_gemm_sm90 as sm90

    saved, sm90._SCHEDULE_OVERRIDE = sm90._SCHEDULE_OVERRIDE, schedule
    try:
        yield
    finally:
        sm90._SCHEDULE_OVERRIDE = saved


@pytest.fixture(autouse=True)
def require_sm90():
    if not torch.cuda.is_available() or not has_wgmma(torch.cuda.get_device_capability()[0]):
        pytest.skip("SM90 is required")


def make_problem(counts, n, k, *, direction="fwd", seed=0, out_dtype=torch.bfloat16):
    """Padded (256-row) expert segments; ``direction="dgrad"`` passes B as the canonical N-major (E, K, N) buffer."""
    assert all(count % 256 == 0 for count in counts)
    generator = torch.Generator(device="cuda").manual_seed(20261121 + seed)
    experts, m = len(counts), max(sum(counts), 256)
    a = torch.randn((m, k), generator=generator, device="cuda", dtype=torch.bfloat16) * 0.125
    # fwd: weights (E, N, K) row-major (K-major B). dgrad: the forward weights (E, n_out=K, k_in=N), read N-major.
    storage_shape = (experts, n, k) if direction == "fwd" else (experts, k, n)
    weights = torch.randn(storage_shape, generator=generator, device="cuda", dtype=torch.bfloat16) * 0.125
    offsets = torch.tensor(counts, dtype=torch.int64).cumsum(0).to(torch.int32).cuda()
    alpha = torch.linspace(0.5, 1.5, experts, dtype=torch.float32, device="cuda")
    prob = torch.rand((m,), generator=generator, device="cuda", dtype=torch.float32)
    return dict(a=a, weights=weights, offsets=offsets, alpha=alpha, prob=prob, n=n, k=k, counts=counts, direction=direction, out_dtype=out_dtype)


def reference(problem):
    """FP32 per-expert C = alpha * A @ B^T and D = prob * C."""
    a, weights = problem["a"].float(), problem["weights"].float()
    c = torch.zeros((a.shape[0], problem["n"]), dtype=torch.float32, device="cuda")
    begin = 0
    for expert, count in enumerate(problem["counts"]):
        w = weights[expert] if problem["direction"] == "fwd" else weights[expert].t()
        c[begin : begin + count] = problem["alpha"][expert] * (a[begin : begin + count] @ w.t())
        begin += count
    return c, c * problem["prob"][:, None]


def run(problem, schedule=None, generate_c=True, **kwargs):
    """``schedule`` forces the SM90 schedule ("cooperative" / "pingpong"); None keeps the heuristic."""
    import cudnn

    with override_schedule(schedule):
        return cudnn.grouped_gemm_wrapper_sm90(
            a_tensor=problem["a"],
            padded_offsets=problem["offsets"],
            alpha_tensor=problem["alpha"],
            b_tensor=problem["weights"],
            b_major="k" if problem["direction"] == "fwd" else "n",
            prob_tensor=problem["prob"],
            c_dtype=problem["out_dtype"],
            d_dtype=problem["out_dtype"],
            generate_c=generate_c,
            **kwargs,
        )


def check(problem, result):
    expected_c, expected_d = reference(problem)
    rows = sum(problem["counts"])
    for name, actual, expected in (("c", result["c_tensor"], expected_c), ("d", result["d_tensor"], expected_d)):
        assert not torch.isnan(actual[:rows].float()).any(), name
        torch.testing.assert_close(actual[:rows].float(), expected[:rows].to(actual.dtype).float(), rtol=2e-2, atol=3e-2, msg=name)


@pytest.mark.parametrize("direction", ["fwd", "dgrad"])
@pytest.mark.parametrize(
    "tile,cluster",
    [(None, None), ((128, 256), (1, 1)), ((128, 128), (1, 1)), ((128, 192), (2, 1)), ((64, 256), (1, 2)), ((64, 64), (2, 2))],
    ids=["default", "128x256-c11", "128x128-c11", "128x192-c21", "64x256-c12", "64x64-c22"],
)
def test_sm90_grouped_gemm_tiles(direction, tile, cluster):
    problem = make_problem((256, 0, 768, 512), n=1024, k=512, direction=direction)
    check(problem, run(problem, mma_tiler_mn=tile, cluster_shape_mn=cluster))


@pytest.mark.parametrize("direction", ["fwd", "dgrad"])
@pytest.mark.parametrize("n,k", [(96, 256), (416, 288), (7168, 2048), (2048, 7168)], ids=["N96", "N416-K288", "N7168-K2048", "N2048-K7168"])
def test_sm90_grouped_gemm_edges(direction, n, k):
    # N below one tile and off the 256 grid, K off the 64 grid, and the deepseek_v3 FC2 / dgrad shapes.
    problem = make_problem((512, 256, 0, 256), n=n, k=k, direction=direction, seed=n + k)
    check(problem, run(problem))


@pytest.mark.parametrize("experts", [1, 8, 32, 256])
def test_sm90_grouped_gemm_experts(experts):
    counts = [256 * ((i * 7) % 4) for i in range(experts)]
    counts[-1] = 256  # the last expert's end is the token count
    problem = make_problem(tuple(counts), n=512, k=256, seed=experts)
    check(problem, run(problem))


def test_sm90_grouped_gemm_all_empty():
    # Contract: with no routed rows the kernel exits before any store, so caller-supplied C and D stay untouched.
    problem = make_problem((0, 0, 0), n=256, k=256)
    c, d = (torch.full((256, 256), 7.0, dtype=torch.bfloat16, device="cuda") for _ in range(2))
    result = run(problem, c_tensor=c, d_tensor=d)
    torch.cuda.synchronize()
    assert result["d_tensor"].shape == (256, 256)
    assert bool((c == 7.0).all()) and bool((d == 7.0).all())


@pytest.mark.parametrize("direction", ["fwd", "dgrad"])
@pytest.mark.parametrize("schedule", ["cooperative", "pingpong"])
def test_sm90_grouped_gemm_zero_rows(direction, schedule):
    # A rank with no routed tokens: (0, K) A and 0-row C/D/prob. N/K are unique here: a cached 256-row compile of
    # the same signature rejects 0-row operands (their stride order differs), on SM100 as well.
    problem = make_problem((0, 0), n=352, k=192, direction=direction)
    problem.update(a=problem["a"][:0], prob=problem["prob"][:0])
    result = run(problem, schedule=schedule)
    torch.cuda.synchronize()
    assert result["c_tensor"].shape == (0, 352) and result["d_tensor"].shape == (0, 352)


def test_sm90_grouped_gemm_fp16_and_legacy_layout():
    import cudnn

    problem = make_problem((256, 512), n=512, k=256, out_dtype=torch.float16)
    check(problem, run(problem))
    # The kernel-facing (m, k, 1) A, (n, k, l) B and (m, 1, 1) prob views give the same bytes.
    m, k = problem["a"].shape
    legacy = cudnn.grouped_gemm_wrapper_sm90(
        a_tensor=problem["a"].as_strided((m, k, 1), (k, 1, m * k)),
        padded_offsets=problem["offsets"],
        alpha_tensor=problem["alpha"],
        b_tensor=problem["weights"].permute(1, 2, 0),
        prob_tensor=problem["prob"].view(m, 1, 1),
        c_dtype=torch.float16,
        d_dtype=torch.float16,
    )
    assert torch.equal(legacy["d_tensor"].view(m, -1), run(problem)["d_tensor"])


def test_sm90_grouped_gemm_static_request_runs_dynamic():
    # SM90 always runs the dynamic scheduler; use_dynamic_sched=False is accepted.
    problem = make_problem((256, 256), n=256, k=256)
    check(problem, run(problem, use_dynamic_sched=False))


def test_sm90_grouped_gemm_rejects_unsupported():
    problem = make_problem((256, 256), n=256, k=256)
    with pytest.raises(NotImplementedError, match="bias"):
        run(problem, bias_tensor=torch.zeros((2, 256), dtype=torch.bfloat16, device="cuda").t())
    fp32 = dict(problem, out_dtype=torch.float32)
    with pytest.raises((NotImplementedError, ValueError)):
        run(fp32)
    with pytest.raises((NotImplementedError, ValueError, AssertionError)):
        run(problem, mma_tiler_mn=(256, 256))
    import cudnn

    ptrs = torch.tensor([w.data_ptr() for w in problem["weights"]], dtype=torch.int64, device="cuda")
    with pytest.raises((NotImplementedError, AssertionError)):
        cudnn.grouped_gemm_wrapper_sm90(
            a_tensor=problem["a"],
            padded_offsets=problem["offsets"],
            alpha_tensor=problem["alpha"],
            b_ptrs=ptrs,
            n=256,
            b_dtype=torch.bfloat16,
            prob_tensor=problem["prob"],
        )


@pytest.mark.parametrize("tile,cluster", [((128, 256), (4, 1)), ((64, 128), (8, 1)), ((128, 128), (4, 2))], ids=["128-c41", "64-c81", "128-c42"])
def test_sm90_grouped_gemm_rejects_cluster_m_over_expert_pad(tile, cluster):
    # A cluster M tile over the 256-row expert pad would write the next expert's rows.
    problem = make_problem((256, 256), n=256, k=256)
    with pytest.raises(ValueError):
        run(problem, mma_tiler_mn=tile, cluster_shape_mn=cluster)


# ---- pingpong schedule: each MMA warp group owns alternate tiles of the scheduler's sequence ----

# Pingpong runs one configuration, 128x192 tiles on (2, 1) clusters; (None, None) picks it.
PINGPONG_CASES = [(None, None), ((128, 192), (2, 1))]
PINGPONG_IDS = ["default", "128x192-c21"]


def single_cluster(monkeypatch, cluster):
    """Run on one cluster (the overlap margin leaves one active), so each CTA's tile count is the problem's."""
    import cutlass

    clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster[0] * cluster[1])
    monkeypatch.setenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", str(clusters - 1))


@pytest.mark.parametrize("direction", ["fwd", "dgrad"])
@pytest.mark.parametrize("tile,cluster", PINGPONG_CASES, ids=PINGPONG_IDS)
def test_sm90_pingpong_tiles(direction, tile, cluster):
    # Ragged experts with empty ones and enough tiles that CTAs run several, in odd and even counts.
    problem = make_problem((512, 0, 1280, 256, 0, 768, 1024), n=1536, k=320, direction=direction, seed=31)
    check(problem, run(problem, mma_tiler_mn=tile, cluster_shape_mn=cluster, schedule="pingpong"))


@pytest.mark.parametrize("direction", ["fwd", "dgrad"])
@pytest.mark.parametrize(
    "counts,n",
    [((256,), 192), ((256, 0, 768, 0, 256), 576), ((256, 0, 512, 256), 576), ((256, 512, 0), 800)],
    ids=["one-tile", "odd-15", "even-12", "odd-n-tail"],
)
def test_sm90_pingpong_one_cluster_parity(monkeypatch, direction, counts, n):
    # One (2, 1) cluster of 128x192 tiles: per CTA 1, 15, 12 and 15 tiles; warp group 1 skips the odd tiles' A/B stages.
    single_cluster(monkeypatch, (2, 1))
    problem = make_problem(counts, n=n, k=576, direction=direction, seed=len(counts) + n)
    check(problem, run(problem, mma_tiler_mn=(128, 192), cluster_shape_mn=(2, 1), schedule="pingpong"))


@pytest.mark.parametrize("direction", ["fwd", "dgrad"])
def test_sm90_cooperative_one_cluster(monkeypatch, direction):
    # One (2, 1) cluster of 128x256 tiles: 15 tiles per CTA, so the tile_info stages are reused (a racecheck target).
    single_cluster(monkeypatch, (2, 1))
    problem = make_problem((256, 0, 768, 0, 256), n=768, k=576, direction=direction, seed=43)
    check(problem, run(problem, mma_tiler_mn=(128, 256), cluster_shape_mn=(2, 1), schedule="cooperative"))


@pytest.mark.parametrize("direction", ["fwd", "dgrad"])
def test_sm90_pingpong_matches_cooperative(direction):
    # 128x192 runs the same 64x192 WGMMAs per 64 rows in both schedules (two warp groups on one tile, or one per tile).
    problem = make_problem((768, 0, 256, 1280), n=1344, k=704, direction=direction, seed=37)
    cooperative = run(problem, mma_tiler_mn=(128, 192), cluster_shape_mn=(2, 1), schedule="cooperative")
    pingpong = run(problem, mma_tiler_mn=(128, 192), cluster_shape_mn=(2, 1), schedule="pingpong")
    check(problem, pingpong)
    for name in ("c_tensor", "d_tensor"):
        assert torch.equal(pingpong[name], cooperative[name]), name


@pytest.mark.parametrize("direction", ["fwd", "dgrad"])
@pytest.mark.parametrize("schedule", ["cooperative", "pingpong"])
def test_sm90_random_scales_without_c(direction, schedule):
    # D = A @ B * (alpha * prob) with random per-expert alpha and per-row prob, with and without C: the same D bytes.
    problem = make_problem((512, 0, 768, 256), n=2304, k=448, direction=direction, seed=53)
    generator = torch.Generator(device="cuda").manual_seed(53)
    problem["alpha"] = torch.rand((4,), generator=generator, device="cuda") * 3.0 - 1.0
    problem["prob"] = torch.rand(problem["prob"].shape, generator=generator, device="cuda") * 2.0
    without_c = run(problem, schedule=schedule, generate_c=False)
    with_c = run(problem, schedule=schedule)
    check(problem, with_c)
    _, expected_d = reference(problem)
    rows = sum(problem["counts"])
    actual = without_c["d_tensor"][:rows].float()
    torch.testing.assert_close(actual, expected_d[:rows].to(torch.bfloat16).float(), rtol=2e-2, atol=3e-2)
    assert torch.equal(without_c["d_tensor"][:rows], with_c["d_tensor"][:rows])


def test_sm90_pingpong_fp16_genc_and_legacy_layout():
    import cudnn

    problem = make_problem((256, 0, 512), n=640, k=256, out_dtype=torch.float16, seed=41)
    result = run(problem, schedule="pingpong")
    check(problem, result)
    m, k = problem["a"].shape
    with override_schedule("pingpong"):
        legacy = cudnn.grouped_gemm_wrapper_sm90(
            a_tensor=problem["a"].as_strided((m, k, 1), (k, 1, m * k)),
            padded_offsets=problem["offsets"],
            alpha_tensor=problem["alpha"],
            b_tensor=problem["weights"].permute(1, 2, 0),
            prob_tensor=problem["prob"].view(m, 1, 1),
            c_dtype=torch.float16,
            d_dtype=torch.float16,
        )
    assert torch.equal(legacy["d_tensor"].view(m, -1), result["d_tensor"])


def test_sm90_pingpong_all_empty():
    problem = make_problem((0, 0, 0), n=256, k=256)
    result = run(problem, schedule="pingpong")
    torch.cuda.synchronize()
    assert result["d_tensor"].shape == (256, 256)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(mma_tiler_mn=(128, 256), schedule="pingpong"), "pingpong"),
        (dict(mma_tiler_mn=(128, 224), schedule="pingpong"), "pingpong"),
        (dict(mma_tiler_mn=(128, 160), schedule="pingpong"), "pingpong"),
        (dict(mma_tiler_mn=(128, 128), schedule="pingpong"), "pingpong"),
        (dict(mma_tiler_mn=(64, 256), cluster_shape_mn=(1, 2), schedule="pingpong"), "pingpong"),
        (dict(mma_tiler_mn=(128, 192), cluster_shape_mn=(1, 2), schedule="pingpong"), "pingpong"),
        (dict(mma_tiler_mn=(128, 192), cluster_shape_mn=(4, 1), schedule="pingpong"), "pingpong"),
        (dict(schedule="stream_k"), "SM90 schedule override"),
    ],
    ids=["128x256", "128x224", "128x160", "128x128", "64x256-c12", "128x192-c12", "128x192-c41", "unknown-schedule"],
)
def test_sm90_pingpong_rejects(kwargs, match):
    # Pingpong runs only 128x192 tiles on (2, 1) clusters; an unknown schedule override raises too.
    problem = make_problem((256, 256), n=256, k=256)
    with pytest.raises(ValueError, match=match):
        run(problem, **kwargs)


@pytest.mark.parametrize(
    "n,k,tile,cluster,expected",
    [
        (2048, 1024, None, None, (True, (128, 192))),
        (2048, 1024, (128, 192), (2, 1), (True, (128, 192))),
        (2048, 3072, None, None, (True, (128, 192))),
        (2048, 3136, None, None, (False, (128, 256))),
        (4096, 4096, None, None, (False, (128, 256))),
        (1024, 1024, None, None, (False, (128, 256))),
        (2048, 1024, (128, 256), None, (False, (128, 256))),
        (2048, 1024, (128, 160), None, (False, (128, 160))),
        (2048, 1024, None, (1, 2), (False, (128, 256))),
        (2048, 1024, (128, 192), (1, 2), (False, (128, 192))),
    ],
    ids=[
        "N2048-K1024",
        "explicit-128x192-c21",
        "N2048-K3072",
        "N2048-K3136",
        "N4096-K4096",
        "N1024-K1024",
        "explicit-128x256",
        "explicit-128x160",
        "c12",
        "128x192-c12",
    ],
)
def test_sm90_schedule_heuristic(n, k, tile, cluster, expected):
    # Pingpong for K <= 3072 and N >= 2048 on 128x192 tiles and (2, 1) clusters.
    from cudnn.gemm.cutedsl.grouped.unfused._bf16_api import GroupedGemmBf16API

    m, dev = 256, "cuda"
    a = torch.zeros((m, k), dtype=torch.bfloat16, device=dev)
    b = torch.zeros((1, n, k), dtype=torch.bfloat16, device=dev)
    c = torch.zeros((m, n), dtype=torch.bfloat16, device=dev)
    d = torch.zeros((m, n), dtype=torch.bfloat16, device=dev)
    common = dict(sample_padded_offsets=torch.full((1,), m, dtype=torch.int32, device=dev), sample_alpha=torch.ones(1, device=dev))
    common.update(sample_b=b, sample_prob=torch.ones((m,), device=dev), mma_tiler_mn=tile, cluster_shape_mn=cluster)
    api = GroupedGemmBf16API(a, c, d, arch="sm90", **common)
    assert (api._pingpong, api.mma_tiler_mn) == expected


def test_sm90_names_match_sm100_signatures():
    import inspect

    import cudnn

    assert inspect.signature(cudnn.grouped_gemm_wrapper_sm90) == inspect.signature(cudnn.grouped_gemm_wrapper_sm100)
    for method in ("__init__", "check_support", "compile", "execute"):
        assert inspect.signature(getattr(cudnn.GroupedGemmSm90, method)) == inspect.signature(getattr(cudnn.GroupedGemmSm100, method))


def test_sm90_rejects_sm100_names():
    import cudnn

    problem = make_problem((256,), n=256, k=256)
    with pytest.raises(RuntimeError, match="requires SM100"):
        cudnn.grouped_gemm_wrapper_sm100(
            a_tensor=problem["a"],
            padded_offsets=problem["offsets"],
            alpha_tensor=problem["alpha"],
            b_tensor=problem["weights"],
            prob_tensor=problem["prob"],
        )
