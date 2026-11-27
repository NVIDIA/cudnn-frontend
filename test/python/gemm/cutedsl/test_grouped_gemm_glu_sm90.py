# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Wentao Guo, Phuong Nguyen

"""SM90 (Hopper) coverage for the BF16 grouped GEMM GLU (forward) and dGLU (backward): the *Glu*Sm90 names."""

from __future__ import annotations

import pytest
import torch

from cudnn.tensor_adapter import has_wgmma

from gemm.cutedsl.test_grouped_gemm_sm90 import override_schedule

pytestmark = pytest.mark.L0

BLOCK = 32  # interleaved_32: accumulator column block 2q is gate, block 2q + 1 is up


@pytest.fixture(autouse=True)
def require_sm90():
    if not torch.cuda.is_available() or not has_wgmma(torch.cuda.get_device_capability()[0]):
        pytest.skip("SM90 is required")


def split_gate_up(x):
    """(m, 2n) interleaved_32 columns -> (gate, up), each (m, n)."""
    m, two_n = x.shape
    blocks = x.view(m, two_n // (2 * BLOCK), 2, BLOCK)
    return blocks[:, :, 0].reshape(m, -1), blocks[:, :, 1].reshape(m, -1)


def join_gate_up(gate, up):
    m, n = gate.shape
    return torch.stack((gate.view(m, n // BLOCK, BLOCK), up.view(m, n // BLOCK, BLOCK)), dim=2).reshape(m, 2 * n)


def to_gate_then_up(x, dim):
    """interleaved_32 -> gate_then_up along ``dim`` (2n wide): the gate blocks in order, then the up blocks."""
    two_n = x.shape[dim]
    order = torch.arange(two_n, device=x.device).view(two_n // (2 * BLOCK), 2, BLOCK).permute(1, 0, 2).reshape(-1)
    return x.index_select(dim, order)


def segments(counts):
    begin = 0
    for expert, count in enumerate(counts):
        yield expert, slice(begin, begin + count)
        begin += count


def make_problem(counts, n, k, *, seed=0, out_dtype=torch.bfloat16, b_major="k"):
    """Padded (256-row) expert segments; ``n`` is the GEMM N (the GLU's C width, the dGLU's gate/up width)."""
    assert all(count % 256 == 0 for count in counts)
    generator = torch.Generator(device="cuda").manual_seed(20261122 + seed)
    experts, m = len(counts), max(sum(counts), 256)
    a = torch.randn((m, k), generator=generator, device="cuda", dtype=torch.bfloat16) * 0.25
    storage = (experts, n, k) if b_major == "k" else (experts, k, n)
    weights = torch.randn(storage, generator=generator, device="cuda", dtype=torch.bfloat16) * 0.125
    offsets = torch.tensor(counts, dtype=torch.int64).cumsum(0).to(torch.int32).cuda()
    alpha = torch.linspace(0.5, 1.5, experts, dtype=torch.float32, device="cuda")
    beta = torch.linspace(1.25, 0.75, experts, dtype=torch.float32, device="cuda")
    prob = torch.rand((m,), generator=generator, device="cuda", dtype=torch.float32)
    # The forward's saved pre-activation C (gate | up), wide enough to cross the GeGLU clamps.
    c = (torch.randn((m, 2 * n), generator=generator, device="cuda", dtype=torch.float32) * 4).to(out_dtype)
    return dict(a=a, weights=weights, offsets=offsets, alpha=alpha, beta=beta, prob=prob, c=c, n=n, k=k, counts=counts, out_dtype=out_dtype, b_major=b_major)


def gemm_reference(problem, scale):
    """FP32 per-expert scale[e] * A @ B^T, (m, n)."""
    a, weights = problem["a"].float(), problem["weights"].float()
    acc = torch.zeros((a.shape[0], problem["n"]), dtype=torch.float32, device="cuda")
    for expert, rows in segments(problem["counts"]):
        w = weights[expert] if problem["b_major"] == "k" else weights[expert].t()
        acc[rows] = scale[expert] * (a[rows] @ w.t())
    return acc


def expert_rows(problem, per_expert):
    out = torch.zeros((problem["a"].shape[0], 1), dtype=torch.float32, device="cuda")
    for expert, rows in segments(problem["counts"]):
        out[rows] = per_expert[expert]
    return out


# ---- GLU forward ----


def glu_reference(problem, act, linear_offset):
    c = gemm_reference(problem, problem["alpha"])
    gate, up = split_gate_up(c)
    if act == "geglu":
        gate, up = gate.clamp(max=7.0), up.clamp(-7.0, 7.0)
        d = (up + linear_offset) * gate * torch.sigmoid(1.702 * gate)
    else:
        d = up * gate * torch.sigmoid(gate)
    return c, d * problem["prob"][:, None]


def run_glu(problem, act="swiglu", schedule=None, **kwargs):
    """``schedule`` forces the SM90 schedule ("cooperative" / "pingpong"); None keeps the heuristic."""
    import cudnn

    with override_schedule(schedule):
        return cudnn.grouped_gemm_glu_wrapper_sm90(
            a_tensor=problem["a"],
            sfa_tensor=None,
            padded_offsets=problem["offsets"],
            alpha_tensor=problem["alpha"],
            b_tensor=problem["weights"],
            prob_tensor=problem["prob"],
            c_dtype=problem["out_dtype"],
            d_dtype=problem["out_dtype"],
            act_func=act,
            **{"generate_c": True, **kwargs},
        )


def check_glu(problem, result, act="swiglu", linear_offset=None, with_c=True):
    linear_offset = (1.0 if act == "geglu" else 0.0) if linear_offset is None else linear_offset
    expected_c, expected_d = glu_reference(problem, act, linear_offset)
    rows = sum(problem["counts"])
    checked = (("c", result["c_tensor"], expected_c),) if with_c else ()
    for name, actual, expected in checked + (("d", result["d_tensor"], expected_d),):
        assert not torch.isnan(actual[:rows].float()).any(), name
        torch.testing.assert_close(actual[:rows].float(), expected[:rows].to(actual.dtype).float(), rtol=2e-2, atol=3e-2, msg=name)


@pytest.mark.parametrize("act", ["swiglu", "geglu"])
@pytest.mark.parametrize(
    "tile,cluster",
    [(None, None), ((128, 256), (1, 1)), ((128, 128), (1, 1)), ((128, 192), (2, 1)), ((64, 256), (1, 2)), ((64, 128), (2, 2))],
    ids=["default", "128x256-c11", "128x128-c11", "128x192-c21", "64x256-c12", "64x128-c22"],
)
def test_sm90_glu_tiles(act, tile, cluster):
    problem = make_problem((256, 0, 768, 512), n=1024, k=512)
    check_glu(problem, run_glu(problem, act, mma_tiler_mn=tile, cluster_shape_mn=cluster), act)


@pytest.mark.parametrize("n,k", [(128, 256), (448, 288), (4096, 7168)], ids=["N128", "N448-K288", "N4096-K7168"])
def test_sm90_glu_edges(n, k):
    # N below one tile and off the 256 grid, K off the 64 grid, and the deepseek_v3 FC1 shape.
    problem = make_problem((512, 256, 0, 256), n=n, k=k, seed=n + k)
    check_glu(problem, run_glu(problem))


@pytest.mark.parametrize("experts", [1, 32])
def test_sm90_glu_experts(experts):
    counts = [256 * ((i * 7) % 4) for i in range(experts)]
    counts[-1] = 256
    problem = make_problem(tuple(counts), n=512, k=256, seed=experts)
    check_glu(problem, run_glu(problem))


def test_sm90_glu_geglu_linear_offset_fp16_and_all_empty():
    problem = make_problem((256, 512), n=512, k=256, out_dtype=torch.float16)
    check_glu(problem, run_glu(problem, "geglu", linear_offset=0.25), "geglu", 0.25)
    empty = make_problem((0, 0, 0), n=256, k=256)
    result = run_glu(empty)
    torch.cuda.synchronize()
    assert result["d_tensor"].shape == (256, 128)


def test_sm90_glu_rejects_unsupported():
    problem = make_problem((256, 256), n=256, k=256)
    with pytest.raises(NotImplementedError, match="bias"):
        run_glu(problem, bias_tensor=torch.zeros((2, 256), dtype=torch.bfloat16, device="cuda").t())
    with pytest.raises((NotImplementedError, ValueError)):
        run_glu(dict(problem, out_dtype=torch.float32))
    with pytest.raises(NotImplementedError, match="divisible by 128"):
        # gate_then_up needs each branch's N (here 320 / 2) to be a multiple of 128.
        run_glu(make_problem((256, 256), n=320, k=256), glu_layout="gate_then_up")
    with pytest.raises((NotImplementedError, ValueError)):
        run_glu(problem, mma_tiler_mn=(128, 96))
    for tile, cluster in (((128, 256), (4, 1)), ((64, 128), (8, 1))):
        with pytest.raises(ValueError):
            run_glu(problem, mma_tiler_mn=tile, cluster_shape_mn=cluster)


# ---- GLU forward, pingpong schedule: 128x192 tiles on (2, 1) clusters ----

NO_C = dict(schedule="pingpong", generate_c=False)


@pytest.mark.parametrize("act", ["swiglu", "geglu"])
@pytest.mark.parametrize("tile,cluster", [(None, None), ((128, 192), (2, 1))], ids=["default", "128x192-c21"])
def test_sm90_glu_pingpong_tiles(act, tile, cluster):
    # Ragged experts with empty ones and enough tiles that CTAs run several, in odd and even counts.
    problem = make_problem((512, 0, 1280, 256, 0, 768, 1024), n=1536, k=320, seed=33)
    check_glu(problem, run_glu(problem, act, mma_tiler_mn=tile, cluster_shape_mn=cluster, **NO_C), act, with_c=False)


@pytest.mark.parametrize("with_c", [False, True], ids=["no-c", "with-c"])
@pytest.mark.parametrize("counts,n", [((256,), 192), ((256, 0, 768, 0, 256), 576), ((256, 0, 512, 256), 576)], ids=["one-tile", "odd-15", "even-12"])
def test_sm90_glu_pingpong_one_cluster_parity(monkeypatch, counts, n, with_c):
    # One (2, 1) cluster of 128x192 tiles: per CTA 1, 15 and 12 tiles; warp group 1 skips the odd tiles' A/B stages.
    import cutlass

    clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(2)
    monkeypatch.setenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", str(clusters - 1))
    problem = make_problem(counts, n=n, k=576, seed=len(counts) + n)
    result = run_glu(problem, mma_tiler_mn=(128, 192), cluster_shape_mn=(2, 1), schedule="pingpong", generate_c=with_c)
    check_glu(problem, result, with_c=with_c)


def test_sm90_glu_pingpong_matches_cooperative_and_fp16():
    # 128x192 runs the same 64x192 WGMMAs per 64 rows in both schedules (two warp groups on one tile, or one per tile).
    problem = make_problem((768, 0, 256, 1280), n=1536, k=704, seed=37)
    tile = dict(mma_tiler_mn=(128, 192), cluster_shape_mn=(2, 1), generate_c=False)
    cooperative = run_glu(problem, "geglu", schedule="cooperative", **tile)
    pingpong = run_glu(problem, "geglu", schedule="pingpong", **tile)
    check_glu(problem, pingpong, "geglu", with_c=False)
    assert torch.equal(pingpong["d_tensor"], cooperative["d_tensor"])
    fp16 = make_problem((256, 512), n=576, k=256, out_dtype=torch.float16)
    check_glu(fp16, run_glu(fp16, "geglu", linear_offset=0.25, **NO_C), "geglu", 0.25, with_c=False)


@pytest.mark.parametrize("act", ["swiglu", "geglu"])
def test_sm90_glu_pingpong_with_c(act):
    # Pingpong stores C before the activation; C and D match the cooperative 128x192 bitwise. Odd and even tile counts.
    problem = make_problem((512, 0, 1280, 256, 0, 768, 1024), n=1536, k=320, seed=41)
    tile = dict(mma_tiler_mn=(128, 192), cluster_shape_mn=(2, 1))
    pingpong = run_glu(problem, act, schedule="pingpong", **tile)
    check_glu(problem, pingpong, act)
    cooperative = run_glu(problem, act, schedule="cooperative", **tile)
    for name in ("c_tensor", "d_tensor"):
        assert torch.equal(pingpong[name], cooperative[name]), name
    fp16 = make_problem((256, 512), n=576, k=256, out_dtype=torch.float16)
    check_glu(fp16, run_glu(fp16, "geglu", linear_offset=0.25, schedule="pingpong"), "geglu", 0.25)


@pytest.mark.parametrize("act", ["swiglu", "geglu"])
def test_sm90_glu_pingpong_with_c_zero_rows(act):
    # Pingpong with C on a rank with no routed tokens (0-row A, C and prob; unique N/K, see the fwd zero-row test) and
    # with every expert empty (no tile is claimed).
    problem = make_problem((0, 0), n=704, k=192)
    problem.update(a=problem["a"][:0], prob=problem["prob"][:0], c=problem["c"][:0])
    result = run_glu(problem, act, schedule="pingpong")
    torch.cuda.synchronize()
    assert result["c_tensor"].shape == (0, 704) and result["d_tensor"].shape == (0, 352)
    empty = make_problem((0, 0, 0), n=1536, k=320)
    result = run_glu(empty, act, schedule="pingpong")
    torch.cuda.synchronize()
    assert result["c_tensor"].shape == (256, 1536) and result["d_tensor"].shape == (256, 768)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(mma_tiler_mn=(128, 128), generate_c=False),
        dict(mma_tiler_mn=(128, 160), generate_c=False),
        dict(mma_tiler_mn=(128, 256), generate_c=False),
        dict(mma_tiler_mn=(64, 256), cluster_shape_mn=(1, 2), generate_c=False),
        dict(mma_tiler_mn=(128, 192), cluster_shape_mn=(1, 2), generate_c=False),
    ],
    ids=["128x128", "128x160", "128x256", "64x256-c12", "128x192-c12"],
)
def test_sm90_glu_pingpong_rejects(kwargs):
    # Pingpong runs only 128x192 on (2, 1) clusters.
    problem = make_problem((256, 256), n=256, k=256)
    with pytest.raises(ValueError, match="pingpong"):
        run_glu(problem, schedule="pingpong", **kwargs)


# ---- GLU forward, glu_layout="gate_then_up" ----

GATE_THEN_UP_GLU = {
    "cooperative-128x256-c21": dict(schedule="cooperative", mma_tiler_mn=(128, 256), cluster_shape_mn=(2, 1)),
    "cooperative-128x192-c21": dict(schedule="cooperative", mma_tiler_mn=(128, 192), cluster_shape_mn=(2, 1)),
    "pingpong-128x192-c21": dict(schedule="pingpong", mma_tiler_mn=(128, 192), cluster_shape_mn=(2, 1)),
    "128x128-c11": dict(mma_tiler_mn=(128, 128), cluster_shape_mn=(1, 1)),
    "64x256-c12": dict(mma_tiler_mn=(64, 256), cluster_shape_mn=(1, 2)),
    "64x128-c22": dict(mma_tiler_mn=(64, 128), cluster_shape_mn=(2, 2)),
}


@pytest.mark.parametrize("with_c", [True, False], ids=["with-c", "no-c"])
@pytest.mark.parametrize("case", list(GATE_THEN_UP_GLU))
@pytest.mark.parametrize("act", ["swiglu", "geglu"])
def test_sm90_glu_gate_then_up_matches_interleaved(act, case, with_c):
    # The same weights stored gate_then_up give interleaved_32's D bytes and its C bytes in branch order: the same
    # products in the same K order. N = 2048 ends the 192-wide tiles in a partial tile, whose gate rows past N/2 must
    # not reach C; an empty expert and unequal ones.
    problem = make_problem((512, 0, 768, 256), n=2048, k=320, seed=51)
    kwargs = dict(GATE_THEN_UP_GLU[case], generate_c=with_c)
    interleaved = run_glu(problem, act, **kwargs)
    check_glu(problem, interleaved, act, with_c=with_c)
    gate_then_up = run_glu(dict(problem, weights=to_gate_then_up(problem["weights"], 1)), act, glu_layout="gate_then_up", **kwargs)
    rows = sum(problem["counts"])
    assert torch.equal(gate_then_up["d_tensor"][:rows], interleaved["d_tensor"][:rows])
    if with_c:
        assert torch.equal(gate_then_up["c_tensor"][:rows], to_gate_then_up(interleaved["c_tensor"][:rows], 1))


def test_sm90_glu_gate_then_up_default_fp16_experts_and_empty():
    # The default schedule and tile, FP16 C/D with a GeGLU linear_offset, 32 experts, and all-empty experts.
    counts = [256 * ((i * 7) % 4) for i in range(32)]
    counts[-1] = 256
    problem = make_problem(tuple(counts), n=1536, k=256, seed=55, out_dtype=torch.float16)
    interleaved = run_glu(problem, "geglu", linear_offset=0.25)
    check_glu(problem, interleaved, "geglu", 0.25)
    gate_then_up = run_glu(dict(problem, weights=to_gate_then_up(problem["weights"], 1)), "geglu", linear_offset=0.25, glu_layout="gate_then_up")
    rows = sum(counts)
    assert torch.equal(gate_then_up["d_tensor"][:rows], interleaved["d_tensor"][:rows])
    assert torch.equal(gate_then_up["c_tensor"][:rows], to_gate_then_up(interleaved["c_tensor"][:rows], 1))
    empty = make_problem((0, 0, 0), n=256, k=256)
    result = run_glu(empty, glu_layout="gate_then_up")
    torch.cuda.synchronize()
    assert result["d_tensor"].shape == (256, 128)


def test_sm90_glu_gate_then_up_advertised():
    import cudnn

    for owner in (cudnn.GroupedGemmGluSm90, cudnn.grouped_gemm_glu_wrapper_sm90, cudnn.GroupedGemmDgluSm90, cudnn.grouped_gemm_dglu_wrapper_sm90):
        assert owner.supports_glu_layouts == ("interleaved_32", "gate_then_up")
        assert owner.supports_gate_then_up_precisions == ("bf16",)
    assert cudnn.GroupedGemmGluSm90.preferred_gate_then_up_mma_tiler is None


# ---- dGLU backward ----


def dglu_reference(problem, act, linear_offset, round_dgrad=False):
    """D (dgate | dup) and dprob of dGLU, from dY @ W^T scaled by alpha^2 and the saved C scaled by beta."""
    g = gemm_reference(problem, problem["alpha"] ** 2)
    if round_dgrad:
        g = g.to(torch.bfloat16).float()
    beta = expert_rows(problem, problem["beta"])
    x1, x2 = (half.float() * beta for half in split_gate_up(problem["c"]))
    prob = problem["prob"][:, None]
    if act == "dgeglu":
        y1, y2 = x1.clamp(max=7.0), x2.clamp(-7.0, 7.0) + linear_offset
        s = torch.sigmoid(1.702 * y1)
        dprob = (y1 * s * y2 * g).sum(dim=1)
        d1 = (x1 <= 7.0) * g * prob * s * (1 + 1.702 * y1 * (1 - s)) * y2
        d2 = ((x2 >= -7.0) & (x2 <= 7.0)) * g * prob * y1 * s
    else:
        s = torch.sigmoid(x1)
        dprob = (g * x2 * x1 * s).sum(dim=1)
        d1 = g * prob * x2 * s * (1 + x1 * (1 - s))
        d2 = g * prob * x1 * s
    return join_gate_up(d1, d2), dprob


def run_dglu(problem, act="dswiglu", dprob=None, **kwargs):
    import cudnn

    dprob = torch.zeros_like(problem["prob"]) if dprob is None else dprob
    return cudnn.grouped_gemm_dglu_wrapper_sm90(
        a_tensor=problem["a"],
        c_tensor=problem["c"],
        sfa_tensor=None,
        padded_offsets=problem["offsets"],
        alpha_tensor=problem["alpha"],
        beta_tensor=problem["beta"],
        prob_tensor=problem["prob"],
        dprob_tensor=dprob,
        b_tensor=problem["weights"],
        b_major=problem["b_major"],
        d_dtype=problem["out_dtype"],
        act_func=act,
        **kwargs,
    )


def check_dglu(problem, result, act="dswiglu", linear_offset=None, round_dgrad=False):
    linear_offset = (1.0 if act == "dgeglu" else 0.0) if linear_offset is None else linear_offset
    expected_d, expected_dprob = dglu_reference(problem, act, linear_offset, round_dgrad)
    rows = sum(problem["counts"])
    actual_d = result["d_row_tensor"].view(result["d_row_tensor"].shape[0], -1)
    assert not torch.isnan(actual_d[:rows].float()).any()
    torch.testing.assert_close(actual_d[:rows].float(), expected_d[:rows].to(actual_d.dtype).float(), rtol=2e-2, atol=5e-2, msg="d")
    dprob = result["dprob_tensor"].view(-1)
    torch.testing.assert_close(dprob[:rows], expected_dprob[:rows], rtol=2e-2, atol=2e-1 * (problem["n"] / 256) ** 0.5, msg="dprob")


@pytest.mark.parametrize("act", ["dswiglu", "dgeglu"])
@pytest.mark.parametrize("b_major", ["k", "n"])
@pytest.mark.parametrize(
    "tile,cluster",
    [(None, None), ((128, 128), (1, 1)), ((128, 256), (1, 2)), ((128, 128), (2, 2))],
    ids=["default", "128x128-c11", "128x256-c12", "128x128-c22"],
)
def test_sm90_dglu_tiles(act, b_major, tile, cluster):
    problem = make_problem((256, 0, 768, 512), n=512, k=1024, b_major=b_major)
    check_dglu(problem, run_dglu(problem, act, mma_tiler_mn=tile, cluster_shape_mn=cluster), act)


def test_sm90_dglu_one_cluster(monkeypatch):
    # One (2, 1) cluster of 128x256 tiles: 15 tiles per CTA, so the tile_info stages are reused (a racecheck target).
    import cutlass

    clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(2)
    monkeypatch.setenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", str(clusters - 1))
    problem = make_problem((256, 0, 768, 0, 256), n=768, k=576, seed=45)
    check_dglu(problem, run_dglu(problem, mma_tiler_mn=(128, 256), cluster_shape_mn=(2, 1)))


@pytest.mark.parametrize("n,k", [(96, 256), (416, 288), (2048, 7168)], ids=["N96", "N416-K288", "N2048-K7168"])
def test_sm90_dglu_edges(n, k):
    # N below one tile and off the 128 grid, K off the 64 grid, and the deepseek_v3 FC2 dgrad shape.
    problem = make_problem((512, 256, 0, 256), n=n, k=k, seed=n + k, b_major="n")
    check_dglu(problem, run_dglu(problem))


@pytest.mark.parametrize("experts", [1, 32])
def test_sm90_dglu_experts(experts):
    counts = [256 * ((i * 7) % 4) for i in range(experts)]
    counts[-1] = 256
    problem = make_problem(tuple(counts), n=256, k=512, seed=experts)
    check_dglu(problem, run_dglu(problem))


@pytest.mark.parametrize("act", ["dswiglu", "dgeglu"])
def test_sm90_dglu_round_offset_and_fp16(act):
    problem = make_problem((256, 512), n=256, k=512, out_dtype=torch.float16)
    check_dglu(problem, run_dglu(problem, act, round_dgrad_to_input_dtype=True, linear_offset=0.5), act, 0.5, round_dgrad=True)


def test_sm90_dglu_all_empty():
    problem = make_problem((0, 0, 0), n=256, k=256)
    result = run_dglu(problem)
    torch.cuda.synchronize()
    assert result["d_row_tensor"].view(256, -1).shape == (256, 512)
    assert torch.count_nonzero(result["dprob_tensor"]) == 0


def test_sm90_glu_and_dglu_zero_rows():
    # A rank with no routed tokens: 0-row A, C and prob give empty outputs (unique N/K, see the fwd zero-row test).
    problem = make_problem((0, 0), n=448, k=192)
    problem.update(a=problem["a"][:0], prob=problem["prob"][:0], c=problem["c"][:0])
    glu = run_glu(problem)
    dglu = run_dglu(problem)
    torch.cuda.synchronize()
    assert glu["c_tensor"].shape == (0, 448) and glu["d_tensor"].shape == (0, 224)
    assert tuple(dglu["d_row_tensor"].shape) == (0, 896) and dglu["dprob_tensor"].numel() == 0


def test_sm90_dglu_rejects_unsupported():
    problem = make_problem((256, 256), n=256, k=256)
    with pytest.raises(NotImplementedError, match="dbias"):
        run_dglu(problem, generate_dbias=True)
    with pytest.raises((NotImplementedError, ValueError)):
        run_dglu(dict(problem, out_dtype=torch.float32))
    with pytest.raises((NotImplementedError, ValueError)):
        run_dglu(problem, mma_tiler_mn=(64, 256))
    with pytest.raises((NotImplementedError, ValueError)):
        run_dglu(problem, mma_tiler_mn=(128, 256), cluster_shape_mn=(4, 1))
    with pytest.raises(ValueError, match="divisible by 128"):
        # gate_then_up needs each branch's N (the weight N) to be a multiple of 128.
        run_dglu(make_problem((256, 256), n=192, k=256), glu_layout="gate_then_up")


@pytest.mark.parametrize("act", ["dswiglu", "dgeglu"])
@pytest.mark.parametrize("b_major", ["k", "n"])
@pytest.mark.parametrize(
    "tile,cluster",
    [(None, None), ((128, 128), (1, 1)), ((128, 256), (1, 2)), ((128, 128), (2, 2))],
    ids=["default", "128x128-c11", "128x256-c12", "128x128-c22"],
)
def test_sm90_dglu_gate_then_up_matches_interleaved(act, b_major, tile, cluster):
    # C stored gate_then_up gives interleaved_32's D in branch order and its deterministic dprob bytes. N = 384 ends the
    # 256-wide tiles in a partial tile, whose gate blocks past N must load zeros (not up columns) and not be stored.
    problem = make_problem((256, 0, 768, 512), n=384, k=512, b_major=b_major, seed=53)
    kwargs = dict(mma_tiler_mn=tile, cluster_shape_mn=cluster, deterministic=True)
    interleaved = run_dglu(problem, act, **kwargs)
    check_dglu(problem, interleaved, act)
    gate_then_up = run_dglu(dict(problem, c=to_gate_then_up(problem["c"], 1)), act, glu_layout="gate_then_up", **kwargs)
    rows = sum(problem["counts"])
    d = lambda result: result["d_row_tensor"].view(result["d_row_tensor"].shape[0], -1)[:rows]  # noqa: E731
    assert torch.equal(d(gate_then_up), to_gate_then_up(d(interleaved), 1))
    assert torch.equal(gate_then_up["dprob_tensor"].view(-1)[:rows], interleaved["dprob_tensor"].view(-1)[:rows])


@pytest.mark.parametrize("act", ["dswiglu", "dgeglu"])
def test_sm90_dglu_gate_then_up_atomic_round_offset_fp16_and_empty(act):
    # The atomic dprob, round_dgrad_to_input_dtype, a linear_offset and FP16 C/D; then all-empty experts.
    problem = make_problem((256, 0, 512), n=256, k=512, out_dtype=torch.float16, seed=57)
    kwargs = dict(round_dgrad_to_input_dtype=True, linear_offset=0.5)
    interleaved = run_dglu(problem, act, **kwargs)
    check_dglu(problem, interleaved, act, 0.5, round_dgrad=True)
    gate_then_up = run_dglu(dict(problem, c=to_gate_then_up(problem["c"], 1)), act, glu_layout="gate_then_up", **kwargs)
    rows = sum(problem["counts"])
    d = lambda result: result["d_row_tensor"].view(result["d_row_tensor"].shape[0], -1)[:rows]  # noqa: E731
    assert torch.equal(d(gate_then_up), to_gate_then_up(d(interleaved), 1))
    torch.testing.assert_close(gate_then_up["dprob_tensor"].view(-1)[:rows], interleaved["dprob_tensor"].view(-1)[:rows], rtol=1e-4, atol=1e-4)
    empty = make_problem((0, 0, 0), n=256, k=256)
    result = run_dglu(dict(empty, c=to_gate_then_up(empty["c"], 1)), glu_layout="gate_then_up")
    torch.cuda.synchronize()
    assert result["d_row_tensor"].view(256, -1).shape == (256, 512)
    assert torch.count_nonzero(result["dprob_tensor"]) == 0


def test_sm90_glu_names_match_sm100_signatures():
    import inspect

    import cudnn

    for sm100, sm90 in (("GroupedGemmGluSm100", "GroupedGemmGluSm90"), ("GroupedGemmDgluSm100", "GroupedGemmDgluSm90")):
        for method in ("__init__", "check_support", "compile", "execute"):
            assert inspect.signature(getattr(getattr(cudnn, sm90), method)) == inspect.signature(getattr(getattr(cudnn, sm100), method))
    for name in ("grouped_gemm_glu_wrapper", "grouped_gemm_dglu_wrapper"):
        assert inspect.signature(getattr(cudnn, f"{name}_sm90")) == inspect.signature(getattr(cudnn, f"{name}_sm100"))


def test_sm90_glu_rejects_sm100_names():
    import cudnn

    problem = make_problem((256, 256), n=256, k=256)
    with pytest.raises(RuntimeError, match="requires SM100"):
        cudnn.grouped_gemm_glu_wrapper_sm100(
            a_tensor=problem["a"],
            sfa_tensor=None,
            padded_offsets=problem["offsets"],
            alpha_tensor=problem["alpha"],
            b_tensor=problem["weights"],
            prob_tensor=problem["prob"],
        )
    with pytest.raises(RuntimeError, match="requires SM100"):
        cudnn.grouped_gemm_dglu_wrapper_sm100(
            a_tensor=problem["a"],
            c_tensor=problem["c"],
            sfa_tensor=None,
            padded_offsets=problem["offsets"],
            alpha_tensor=problem["alpha"],
            beta_tensor=problem["beta"],
            prob_tensor=problem["prob"],
            dprob_tensor=torch.zeros_like(problem["prob"]),
            b_tensor=problem["weights"],
        )
