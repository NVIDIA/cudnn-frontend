# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic-counter tile claim (``atomic_counter_claim=True``) for the SM100 grouped GEMM wgrad kernels.

By default the wgrad kernels launch a grid of every output tile and rebalance it with cluster launch control (CLC). With
``atomic_counter_claim`` they launch persistent clusters that claim tiles from an int32 counter in the kernel workspace.
Every output tile is still computed by one CTA (pair) over its expert's whole token range, so the bytes cannot depend on
the schedule: each case compares the counter path with the CLC path byte for byte.
"""

import importlib.util
import inspect
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import cudnn
from cudnn.gemm.cutedsl.grouped.wgrad import api as wgrad_api
from determinism_utils import assert_bitwise, snapshot
from gemm.cutedsl.test_grouped_gemm_wgrad_bf16_utils import (
    assert_grouped_gemm_wgrad_close,
    grouped_gemm_wgrad_bf16_reference,
    make_grouped_gemm_wgrad_bf16_problem,
)
from gemm.cutedsl.test_grouped_gemm_wgrad_utils import (
    _wgrad_assemble_scales_2d2d,
    _wgrad_create_scale_tensor,
    _wgrad_independent_reference,
    check_ref_grouped_gemm_wgrad,
)


@pytest.fixture(autouse=True)
def require_sm100():
    from cudnn.tensor_adapter import has_tcgen05

    if not torch.cuda.is_available() or not has_tcgen05(torch.cuda.get_device_capability()[0]):
        pytest.skip("Requires SM100+ for grouped GEMM wgrad")


# Empty experts, one-group experts and uneven K. BF16 groups are multiples of 256 tokens, MXFP8 groups of 128 (one K tile).
BF16_GROUPS = (256, 0, 1024, 256, 0, 0, 512, 2048)
MXFP8_GROUPS = (128, 0, 384, 1024, 0, 128, 2048, 256)
M, N = 2048, 2048
# (mma_tiler_mn, cluster_shape_mn): a one-CTA cluster claims without the cluster broadcast, a (2, 1) pair through it.
L0_CONFIGS = [((128, 256), (1, 1)), ((256, 256), (2, 1))]
L1_CONFIGS = [((128, 128), (1, 2)), ((128, 256), (2, 1)), ((256, 256), (4, 1))]
CONFIGS = [pytest.param(c, marks=pytest.mark.L0) for c in L0_CONFIGS] + [pytest.param(c, marks=pytest.mark.L1) for c in L1_CONFIGS]
MXFP8_L1_CONFIGS = [((128, 128), (1, 2)), ((128, 256), (2, 1)), ((256, 256), (4, 1))]
MXFP8_CONFIGS = [pytest.param(((128, 128), (1, 1)), marks=pytest.mark.L0), pytest.param(((256, 256), (2, 1)), marks=pytest.mark.L0)] + [
    pytest.param(c, marks=pytest.mark.L1) for c in MXFP8_L1_CONFIGS
]


def _config_id(config):
    (tm, tn), (cm, cn) = config
    return f"tile{tm}x{tn}-cluster{cm}x{cn}"


def _expert_ptrs(output):
    return torch.tensor([expert.data_ptr() for expert in output], dtype=torch.int64, device="cuda")


def _fresh_output(like, initial):
    """NaN-filled for an overwrite, so a tile that is never written differs; a copy of ``initial`` when accumulating."""
    return torch.full_like(like, float("nan")) if initial is None else initial.clone()


# ---------------------------------------------------------------------------
# BF16
# ---------------------------------------------------------------------------


def _bf16_wrapper(problem, config, *, counter, accumulate=False, initial=None, output_mode="dense", input_order="tensor2d"):
    tile, cluster = config
    output = _fresh_output(problem["output"], initial)
    discrete = output_mode == "discrete"
    cudnn.grouped_gemm_wgrad_wrapper_sm100(
        a_tensor=problem["a"],
        b_tensor=problem["b"],
        sfa_tensor=None,
        sfb_tensor=None,
        offsets_tensor=problem["offsets"],
        output_mode=output_mode,
        wgrad_tensor=None if discrete else output,
        wgrad_ptrs=_expert_ptrs(output) if discrete else None,
        acc_dtype=torch.float32,
        wgrad_dtype=output.dtype,
        mma_tiler_mn=tile,
        cluster_shape_mn=cluster,
        accumulate_on_output=accumulate,
        input_order=input_order,
        atomic_counter_claim=counter,
    )
    torch.cuda.synchronize()
    return output


@pytest.mark.parametrize("config", CONFIGS, ids=_config_id)
@pytest.mark.parametrize("accumulate", [False, True], ids=["overwrite", "accumulate"])
@pytest.mark.parametrize("output_mode", ["dense", "discrete"])
@pytest.mark.parametrize("input_order", ["tensor2d", "tensor_ragged"])
def test_counter_matches_clc_bf16(config, accumulate, output_mode, input_order):
    problem = make_grouped_gemm_wgrad_bf16_problem(
        group_k_list=BF16_GROUPS, m=M, n=N, input_order=input_order, discrete=output_mode == "discrete", initial_value=0.25 if accumulate else None
    )
    initial = torch.full_like(problem["output"], 0.25) if accumulate else None
    kwargs = dict(accumulate=accumulate, initial=initial, output_mode=output_mode, input_order=input_order)
    clc = _bf16_wrapper(problem, config, counter=False, **kwargs)
    counter = _bf16_wrapper(problem, config, counter=True, **kwargs)
    assert_bitwise({"wgrad": counter}, snapshot({"wgrad": clc}), "atomic counter vs CLC")
    assert_grouped_gemm_wgrad_close(counter, grouped_gemm_wgrad_bf16_reference(problem, accumulate=accumulate))


@pytest.mark.L0
@pytest.mark.parametrize("config", L0_CONFIGS, ids=_config_id)
def test_counter_matches_clc_bf16_unpadded_groups(config):
    """1-token and 255-token experts are off the BF16 contract (256-token groups): a K tile reads past its expert, so
    there is no reference. Both schedules still run the same tile computations, so the claim order must not change a byte."""
    groups = (1, 0, 255, 256, 0, 512)
    tokens = sum(groups)
    generator = torch.Generator(device="cuda").manual_seed(20261124)
    # M-major A and N-major B: any token offset is a legal TMA coordinate.
    a = (torch.randn((tokens, M), device="cuda", generator=generator) * 0.125).to(torch.bfloat16).t()
    b = (torch.randn((tokens, N), device="cuda", generator=generator) * 0.125).to(torch.bfloat16)
    offsets = torch.tensor(groups, dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)
    problem = {"a": a, "b": b, "offsets": offsets, "output": torch.empty((len(groups), M, N), dtype=torch.bfloat16, device="cuda")}
    clc = _bf16_wrapper(problem, config, counter=False)
    counter = _bf16_wrapper(problem, config, counter=True)
    assert_bitwise({"wgrad": counter}, snapshot({"wgrad": clc}), "atomic counter vs CLC")


# ---------------------------------------------------------------------------
# MXFP8 (E4M3 data, E8M0 scale factors per 32 tokens)
# ---------------------------------------------------------------------------


def _make_mxfp8_problem(groups, m, n, *, seed, a_major="k", b_major="k", initial_value=None):
    torch.manual_seed(seed)
    tokens = sum(groups)
    # {-1, 0, 1} and scales {1, 2} are exact in E4M3 / E8M0.
    logical_a = torch.randint(-1, 2, (m, tokens), device="cuda").to(torch.bfloat16)
    logical_b = torch.randint(-1, 2, (tokens, n), device="cuda").to(torch.bfloat16)
    a = logical_a.to(torch.float8_e4m3fn)
    a = a.contiguous() if a_major == "k" else a.t().contiguous().t()
    b = logical_b.to(torch.float8_e4m3fn)
    b = b.t().contiguous().t() if b_major == "k" else b.contiguous()
    raw_sfa = [_wgrad_create_scale_tensor((m, -(-k // 32)), torch.float8_e8m0fnu) for k in groups]
    raw_sfb = [_wgrad_create_scale_tensor((n, -(-k // 32)), torch.float8_e8m0fnu) for k in groups]
    reference = _wgrad_independent_reference(logical_a.float(), logical_b.float(), raw_sfa, raw_sfb, list(groups), 32, torch.bfloat16)
    if initial_value is not None:
        reference = (reference.float() + initial_value).to(torch.bfloat16)
    return {
        "a": a,
        "b": b,
        "sfa": _wgrad_assemble_scales_2d2d(raw_sfa, m),
        "sfb": _wgrad_assemble_scales_2d2d(raw_sfb, n),
        "offsets": torch.tensor(groups, dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32),
        "output": torch.empty((len(groups), m, n), dtype=torch.bfloat16, device="cuda"),
        "reference": reference,
    }


def _mxfp8_wrapper(problem, config, *, counter, accumulate=False, initial=None, output_mode="dense", input_order="tensor2d"):
    tile, cluster = config
    output = _fresh_output(problem["output"], initial)
    discrete = output_mode == "discrete"
    cudnn.grouped_gemm_wgrad_wrapper_sm100(
        a_tensor=problem["a"],
        b_tensor=problem["b"],
        sfa_tensor=problem["sfa"],
        sfb_tensor=problem["sfb"],
        offsets_tensor=problem["offsets"],
        output_mode=output_mode,
        wgrad_tensor=None if discrete else output,
        wgrad_ptrs=_expert_ptrs(output) if discrete else None,
        acc_dtype=torch.float32,
        wgrad_dtype=output.dtype,
        mma_tiler_mn=tile,
        cluster_shape_mn=cluster,
        sf_vec_size=32,
        accumulate_on_output=accumulate,
        input_order=input_order,
        atomic_counter_claim=counter,
    )
    torch.cuda.synchronize()
    return output


@pytest.mark.parametrize("config", MXFP8_CONFIGS, ids=_config_id)
@pytest.mark.parametrize("accumulate", [False, True], ids=["overwrite", "accumulate"])
@pytest.mark.parametrize("output_mode", ["dense", "discrete"])
def test_counter_matches_clc_mxfp8(config, accumulate, output_mode):
    problem = _make_mxfp8_problem(MXFP8_GROUPS, M, N, seed=3, initial_value=0.25 if accumulate else None)
    initial = torch.full_like(problem["output"], 0.25) if accumulate else None
    kwargs = dict(accumulate=accumulate, initial=initial, output_mode=output_mode)
    clc = _mxfp8_wrapper(problem, config, counter=False, **kwargs)
    counter = _mxfp8_wrapper(problem, config, counter=True, **kwargs)
    assert_bitwise({"wgrad": counter}, snapshot({"wgrad": clc}), "atomic counter vs CLC")
    check_ref_grouped_gemm_wgrad(counter, problem["reference"])


@pytest.mark.L1
@pytest.mark.parametrize("config", L0_CONFIGS, ids=_config_id)
def test_counter_matches_clc_mxfp8_tensor_ragged(config):
    problem = _make_mxfp8_problem(MXFP8_GROUPS, M, N, seed=4)
    clc = _mxfp8_wrapper(problem, config, counter=False, input_order="tensor_ragged")
    counter = _mxfp8_wrapper(problem, config, counter=True, input_order="tensor_ragged")
    assert_bitwise({"wgrad": counter}, snapshot({"wgrad": clc}), "atomic counter vs CLC")


# ---------------------------------------------------------------------------
# Class API: poisoned workspace and outputs, eager and CUDA Graph replay
# ---------------------------------------------------------------------------


def _class_op(precision, problem, config, *, counter):
    tile, cluster = config
    kwargs = dict(
        sample_a=problem["a"],
        sample_b=problem["b"],
        sample_sfa=problem.get("sfa"),
        sample_sfb=problem.get("sfb"),
        sample_offsets=problem["offsets"],
        sample_wgrad=problem["output"],
        acc_dtype=torch.float32,
        mma_tiler_mn=tile,
        cluster_shape_mn=cluster,
        sf_vec_size=32 if precision == "mxfp8" else 16,
        atomic_counter_claim=counter,
    )
    op = cudnn.GroupedGemmWgradSm100(**kwargs)
    assert op.check_support()
    op.compile()
    return op


def _class_execute(precision, op, problem, output, workspace=None):
    kwargs = dict(
        a_tensor=problem["a"],
        b_tensor=problem["b"],
        sfa_tensor=problem.get("sfa"),
        sfb_tensor=problem.get("sfb"),
        offsets_tensor=problem["offsets"],
        wgrad_tensor=output,
    )
    if workspace is not None:
        # The block-scaled backend takes a caller workspace as descriptor_workspace; BF16 takes the wrapper's per-call one.
        kwargs["descriptor_workspace" if precision == "mxfp8" else "_workspace"] = workspace
    op.execute(**kwargs)


def _problem(precision, *, seed=11):
    if precision == "mxfp8":
        return _make_mxfp8_problem(MXFP8_GROUPS, M, N, seed=seed)
    return make_grouped_gemm_wgrad_bf16_problem(group_k_list=BF16_GROUPS, m=M, n=N)


def _workspace_bytes(precision, op):
    if precision == "mxfp8":
        return cudnn.get_grouped_gemm_wgrad_workspace_size_sm100(len(MXFP8_GROUPS), atomic_counter_claim=True)
    return op._implementation._scratch_nbytes


@pytest.mark.L0
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize("config", L0_CONFIGS, ids=_config_id)
def test_counter_replays_on_poisoned_buffers(precision, config):
    """Three eager launches, then three CUDA Graph replays, each on a NaN-filled output and a poisoned workspace: the
    helper kernel must reset the counter (and rebuild the descriptors) in front of every launch."""
    problem = _problem(precision)
    expected = snapshot({"wgrad": (_mxfp8_wrapper if precision == "mxfp8" else _bf16_wrapper)(problem, config, counter=False)})
    op = _class_op(precision, problem, config, counter=True)
    nbytes = _workspace_bytes(precision, op)
    assert nbytes >= 4
    for launch, poison in enumerate((0xFF, 0x00, 0xA5)):
        output = torch.full_like(problem["output"], float("nan"))
        workspace = torch.full((nbytes,), poison, dtype=torch.uint8, device="cuda")
        _class_execute(precision, op, problem, output, workspace)
        torch.cuda.synchronize()
        assert_bitwise({"wgrad": output}, expected, f"eager launch {launch}")

    output = torch.full_like(problem["output"], float("nan"))
    workspace = torch.full((nbytes,), 0xFF, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            _class_execute(precision, op, problem, output, workspace)
        for replay, poison in enumerate((0xFF, 0x7F, 0x01)):
            output.fill_(float("nan"))
            workspace.fill_(poison)
            graph.replay()
            torch.cuda.synchronize()
            assert_bitwise({"wgrad": output}, expected, f"graph replay {replay}")
    finally:
        graph.reset()


# ---------------------------------------------------------------------------
# Warm stress of the claim broadcast slot (each claim read from the slot is consumed before the slot is released)
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
def test_counter_claim_slot_warm_stress(monkeypatch, precision):
    """In a (2, 1) cluster the leader CTA broadcasts every claim through a shared-memory slot that each CTA reads and then
    releases to the leader. The references are independent of the counter path: the CLC path's output bytes, and the
    counter's final value, one claim per tile plus each cluster's final empty claim. The TMA descriptors in front of the
    counter are rebuilt by the helper kernel on every launch and are not part of the claim; they are not compared."""
    import cutlass
    from warm_stress import StressCase, assert_warm_stress

    monkeypatch.delenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", raising=False)
    config = ((256, 256), (2, 1))
    problem = _problem(precision)
    groups = MXFP8_GROUPS if precision == "mxfp8" else BF16_GROUPS
    expected = (_mxfp8_wrapper if precision == "mxfp8" else _bf16_wrapper)(problem, config, counter=False)
    op = _class_op(precision, problem, config, counter=True)
    output = torch.full_like(problem["output"], float("nan"))
    _class_execute(precision, op, problem, output)
    torch.cuda.synchronize()
    # The counter is the workspace's last 4 bytes (a view of the kernel's storage, not a copy).
    counter = op._implementation._workspace[-4:]

    clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(2)
    tiles = len(groups) * (M // 256) * (N // 256)
    reference = {"wgrad": expected.clone(), "counter": torch.tensor([tiles + clusters], dtype=torch.int32, device="cuda").view(torch.uint8)}
    assert_bitwise({"wgrad": output}, {"wgrad": reference["wgrad"]}, "cold counter launch vs CLC")
    claims = int(counter.view(torch.int32).item())
    assert claims == tiles + clusters, f"the counter ended at {claims}, not {tiles} tiles + {clusters} clusters"

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            _class_execute(precision, op, problem, output)
        report = assert_warm_stress(
            StressCase(
                f"wgrad {precision} atomic counter claim slot tile=256x256 cluster=(2, 1) M=N={M} groups={groups}",
                launch=graph.replay,
                outputs={"wgrad": output, "counter": counter},
                thread_blocks=2 * clusters,
                refills_per_launch=(tiles + clusters) // clusters,
                reference=reference,
            )
        )
        print(report.text())
    finally:
        graph.reset()


# ---------------------------------------------------------------------------
# Two same-signature MXFP8 wrapper calls on two streams (the BF16 cases live in test_grouped_gemm_bf16_overlap.py)
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.parametrize("output_mode", ["dense", "discrete"])
def test_counter_mxfp8_overlapping_streams(output_mode):
    """The wrapper allocates the counter (and the descriptors) per call, so overlapping calls cannot share a counter."""
    experts, rows, m, n = 32, 512, 4096, 1024
    pair = [_make_mxfp8_problem((rows,) * experts, m, n, seed=seed) for seed in (21, 22)]
    config = ((256, 256), (2, 1))
    expected = [snapshot({"wgrad": _mxfp8_wrapper(problem, config, counter=True, output_mode=output_mode)}) for problem in pair]
    for problem, reference in zip(pair, expected):
        check_ref_grouped_gemm_wgrad(reference["wgrad"], problem["reference"])
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for iteration in range(40):
        torch.cuda.synchronize()
        results = []
        for stream, problem in zip(streams, pair):
            with torch.cuda.stream(stream):
                output = torch.full_like(problem["output"], float("nan"))
                discrete = output_mode == "discrete"
                cudnn.grouped_gemm_wgrad_wrapper_sm100(
                    a_tensor=problem["a"],
                    b_tensor=problem["b"],
                    sfa_tensor=problem["sfa"],
                    sfb_tensor=problem["sfb"],
                    offsets_tensor=problem["offsets"],
                    output_mode=output_mode,
                    wgrad_tensor=None if discrete else output,
                    wgrad_ptrs=_expert_ptrs(output) if discrete else None,
                    acc_dtype=torch.float32,
                    wgrad_dtype=output.dtype,
                    mma_tiler_mn=config[0],
                    cluster_shape_mn=config[1],
                    sf_vec_size=32,
                    atomic_counter_claim=True,
                )
                results.append({"wgrad": output})
        torch.cuda.synchronize()
        for index, (result, reference) in enumerate(zip(results, expected)):
            assert_bitwise(snapshot(result), reference, f"iteration {iteration} call {index}")


# ---------------------------------------------------------------------------
# More persistent clusters than output tiles
# ---------------------------------------------------------------------------

_CHILD_ENV = "CUDNN_WGRAD_COUNTER_TEST_CHILD"

# Multi-CTA clusters that are not a single 2-CTA pair: every CTA releases each broadcast claim with a remote arrive on the
# leader CTA, and nothing else (no 2-CTA TMEM dealloc handshake across the whole cluster) holds the leader until the
# peers' release of the last, invalid claim has landed.
CLUSTERED_EXHAUSTION_CONFIGS = [((128, 128), (1, 2)), ((128, 256), (2, 1)), ((256, 256), (2, 2))]


@pytest.mark.L0
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize("config", L0_CONFIGS + CLUSTERED_EXHAUSTION_CONFIGS, ids=_config_id)
def test_more_clusters_than_tiles(request, precision, config):
    """4 experts x 512 x 512 = 8 to 32 cluster tiles on every cluster the GPU holds: most clusters' first claim is
    already past the last tile. Such a cluster must publish only the sentinel and exit (publishing its invalid first
    claim ahead of the sentinel left the sentinel's stage unreleased: a hang), and in a multi-CTA cluster its leader must
    not exit before every CTA has released that claim. Bytes equal CLC's.

    The case runs in a child pytest with a deadline, so a hang (or a launch failure that poisons the CUDA context) fails
    this test instead of stalling or breaking the session."""
    if os.environ.get(_CHILD_ENV) == "1":
        if precision == "mxfp8":
            problem = _make_mxfp8_problem((384, 0, 128, 640), 512, 512, seed=17)
            clc = _mxfp8_wrapper(problem, config, counter=False)
            counter = _mxfp8_wrapper(problem, config, counter=True)
            check_ref_grouped_gemm_wgrad(counter, problem["reference"])
        else:
            problem = make_grouped_gemm_wgrad_bf16_problem(group_k_list=(512, 0, 256, 768), m=512, n=512)
            clc = _bf16_wrapper(problem, config, counter=False)
            counter = _bf16_wrapper(problem, config, counter=True)
            assert_grouped_gemm_wgrad_close(counter, grouped_gemm_wgrad_bf16_reference(problem))
        assert_bitwise({"wgrad": counter}, snapshot({"wgrad": clc}), "more clusters than tiles: atomic counter vs CLC")
        return

    test_timeout = float(os.environ.get("CUDNN_TEST_TIMEOUT", "1500"))
    timeout = min(600.0, 0.6 * test_timeout) if test_timeout > 0 else 900.0
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "pytest",
        f"{Path(__file__).resolve()}::{request.node.name}",
        *(["-n0"] if importlib.util.find_spec("xdist") else []),
        "-o",
        "addopts=",
        "-p",
        "no:cacheprovider",
        "-q",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=dict(os.environ, **{_CHILD_ENV: "1"}), start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        pytest.fail(f"timed out after {timeout:.0f} s: the kernel hangs\n{out[-4000:]}")
    assert proc.returncode == 0 and "1 passed" in out, out[-8000:]


EXHAUSTION_LAUNCHES = 1000


def _cluster_tiles(config, experts, m, n):
    """Output tiles of a whole cluster: a 2-CTA MMA tile of 256 rows is 128 rows per CTA."""
    (tile_m, tile_n), (cluster_m, cluster_n) = config
    cluster_tile_m = (tile_m // 2 if tile_m == 256 else tile_m) * cluster_m
    cluster_tile_n = tile_n * cluster_n
    return experts * -(-m // cluster_tile_m) * -(-n // cluster_tile_n)


@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("precision", [pytest.param("bf16", marks=pytest.mark.L0), pytest.param("mxfp8", marks=pytest.mark.L1)])
@pytest.mark.parametrize("config", CLUSTERED_EXHAUSTION_CONFIGS, ids=_config_id)
def test_counter_clustered_exhaustion_stress(monkeypatch, precision, config):
    """Fewer cluster tiles than persistent clusters, launched again and again with a cold L2. Every cluster ends on an
    invalid claim: most at their first claim (initial_work_tile_info), the rest after one tile (advance_to_next_work).
    Each CTA of a cluster releases that last claim with a remote arrive on the leader CTA's shared memory, and the
    leader waits for those arrives before it exits. Each launch must finish, take exactly one invalid claim per cluster
    (the counter ends at tiles + clusters) and match CLC's bytes.

    This is exhaustion coverage for the wgrad counter path, not a reproducer of the exit race: a 2Dx2D claim maps to a
    tile without loading the expert offsets, so a peer's release follows its read closely. The race reproducer is
    test_grouped_gemm_dynamic_sched_cluster_exit.py (2Dx3D, offset walks over 1024 experts)."""
    import cutlass

    monkeypatch.delenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", raising=False)
    m = n = 512
    if precision == "mxfp8":
        groups = (384, 0, 128, 640)
        problem = _make_mxfp8_problem(groups, m, n, seed=17)
        expected = snapshot({"wgrad": _mxfp8_wrapper(problem, config, counter=False)})
    else:
        groups = (512, 0, 256, 768)
        problem = make_grouped_gemm_wgrad_bf16_problem(group_k_list=groups, m=m, n=n)
        expected = snapshot({"wgrad": _bf16_wrapper(problem, config, counter=False)})
    tiles = _cluster_tiles(config, len(groups), m, n)
    _, (cluster_m, cluster_n) = config
    clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_m * cluster_n)
    if clusters <= tiles:
        pytest.skip(f"{clusters} persistent clusters do not outnumber {tiles} cluster tiles on this GPU")

    op = _class_op(precision, problem, config, counter=True)
    output = torch.full_like(problem["output"], float("nan"))
    flush = torch.empty(128 << 20, dtype=torch.float32, device="cuda")  # 512 MB, larger than the L2
    for launch in range(EXHAUSTION_LAUNCHES):
        flush.fill_(float(launch))
        output.fill_(float("nan"))
        _class_execute(precision, op, problem, output)
        torch.cuda.synchronize()
        # The counter is the workspace's last 4 bytes (a view of the kernel's storage, not a copy).
        claims = int(op._implementation._workspace[-4:].view(torch.int32).item())
        assert claims == tiles + clusters, f"launch {launch}: the counter ended at {claims}, not {tiles} tiles + {clusters} clusters"
        if launch % 100 == 0 or launch == EXHAUSTION_LAUNCHES - 1:
            assert_bitwise({"wgrad": output}, expected, f"launch {launch}: atomic counter vs CLC")


# ---------------------------------------------------------------------------
# API surface, sizes and rejections
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_atomic_counter_claim_api_surface():
    wrapper = inspect.signature(cudnn.grouped_gemm_wgrad_wrapper_sm100).parameters["atomic_counter_claim"]
    assert wrapper.kind is inspect.Parameter.KEYWORD_ONLY and wrapper.default is False
    init = list(inspect.signature(cudnn.GroupedGemmWgradSm100.__init__).parameters.values())
    # Appended after input_order, so earlier positional calls keep their meaning; only the PDL options follow it.
    names = [p.name for p in init]
    at = names.index("atomic_counter_claim")
    assert names[at - 1] == "input_order" and names[at + 1 :] == ["use_pdl", "launch_pdl_dependents_at_entry"]
    assert all(p.default is False for p in init[at:])
    size = inspect.signature(cudnn.get_grouped_gemm_wgrad_workspace_size_sm100).parameters["atomic_counter_claim"]
    assert size.kind is inspect.Parameter.KEYWORD_ONLY and size.default is False
    for mode in ("dense", "discrete"):
        for order in ("tensor2d", "tensor_ragged"):
            base = cudnn.get_grouped_gemm_wgrad_workspace_size_sm100(8, output_mode=mode, input_order=order)
            assert cudnn.get_grouped_gemm_wgrad_workspace_size_sm100(8, output_mode=mode, input_order=order, atomic_counter_claim=True) == base + 4


@pytest.mark.L0
def test_atomic_counter_claim_grid_and_workspace():
    from cudnn.gemm.cutedsl.grouped.moe_persistent_scheduler import MoESchedulerParams
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode, WGradInputOrder
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad import BlockScaledMoEGroupedGemmWgradKernel
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_grouped_gemm_wgrad import MoEGroupedGemmWgradBF16Kernel

    def grid(counter):
        params = MoESchedulerParams("2Dx2D", (8, 1024, 2048), (128, 256, 64), (2, 1), atomic_counter_claim=counter)
        return MoESchedulerParams.get_grid_shape(params, 74)

    # CLC: every tile of every expert (hidden 2048 / 128, intermediate 1024 / 256, 8 experts); counter: persistent clusters.
    assert tuple(int(x) for x in grid(False)) == (16, 4, 8)
    assert tuple(grid(True)) == (2, 1, 74)

    for mode in (MoEWeightMode.DENSE, MoEWeightMode.DISCRETE):
        for order in (WGradInputOrder.Tensor2D, WGradInputOrder.TensorRagged):
            common = dict(expert_cnt=8, weight_mode=mode, input_order=order)
            for kernel_class, extra in ((MoEGroupedGemmWgradBF16Kernel, {}), (BlockScaledMoEGroupedGemmWgradKernel, {"sf_vec_size": 32})):
                default = kernel_class(**extra, **common)
                counted = kernel_class(**extra, **common, atomic_counter_claim=True)
                assert counted.get_workspace_bytes() == default.get_workspace_bytes() + 4
                assert counted.get_desc_workspace_bytes() == default.get_desc_workspace_bytes() == default.get_workspace_bytes()


@pytest.mark.L0
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
def test_atomic_counter_claim_rejections(precision):
    problem = _make_mxfp8_problem((128, 256), 256, 256, seed=1) if precision == "mxfp8" else make_grouped_gemm_wgrad_bf16_problem(m=256, n=256)
    config = ((128, 128), (1, 1))
    with pytest.raises(ValueError, match="atomic_counter_claim must be a bool"):
        _class_op(precision, problem, config, counter=1)
    with pytest.raises(ValueError, match="atomic_counter_claim must be a bool"):
        (_mxfp8_wrapper if precision == "mxfp8" else _bf16_wrapper)(problem, config, counter="yes")

    default_op = _class_op(precision, problem, config, counter=False)
    op = _class_op(precision, problem, config, counter=True)
    # The wrapper allocates the counter per call only for the counter path.
    if precision == "mxfp8":
        assert default_op._implementation._scratch_nbytes == 0
        assert op._implementation._scratch_nbytes == op._implementation._workspace_bytes
        # A descriptor workspace sized without the counter is refused.
        short = torch.empty(cudnn.get_grouped_gemm_wgrad_workspace_size_sm100(2), dtype=torch.uint8, device="cuda")
        with pytest.raises(ValueError, match="descriptor_workspace requires at least"):
            _class_execute(precision, op, problem, torch.empty_like(problem["output"]), short)
    else:
        assert op._implementation._scratch_nbytes == default_op._implementation._scratch_nbytes + 4


@pytest.mark.L0
def test_atomic_counter_claim_rejected_on_the_jax_mxfp8_path():
    jnp = pytest.importorskip("jax.numpy")
    a = jnp.zeros((128, 128), dtype=jnp.float8_e4m3fn)
    with pytest.raises(ValueError, match="atomic_counter_claim is unsupported for the JAX MXFP8 path"):
        wgrad_api.grouped_gemm_wgrad_wrapper_sm100(a, a, a, a, jnp.zeros((1,), dtype=jnp.int32), sf_vec_size=32, atomic_counter_claim=True)
