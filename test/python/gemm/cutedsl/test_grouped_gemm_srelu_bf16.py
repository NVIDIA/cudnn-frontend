# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BF16 coverage for the grouped GEMM + weighted squared-ReLU forward (grouped_gemm_srelu_wrapper_sm100 and
GroupedGemmSreluSm100).

Numerics against an FP32 reference and the native path (test_grouped_gemm_srelu_bf16_utils.py), the class API's
caller-owned scratch (recipe R2), the Rule 8 / R9 detectors (no host synchronization, no allocation, a first execute
inside CUDA-graph capture), the wrapper's plan cache, per-launch rebinding of the discrete weight descriptors, the lazy
facade and the support checks. Bitwise repeatability, batch invariance and the Rule 11 warm stress of the kernel are
in test_grouped_gemm_srelu_bf16_determinism.py.
"""

from __future__ import annotations

import inspect
import logging
import re

import pytest
import torch

from gemm.cutedsl._workspace import ws
from gemm.cutedsl.test_grouped_gemm_srelu_bf16_utils import (
    CLAMP_SCALE,
    DEVICE_VALUES_ENV,
    CompileSpy,
    NEMOTRON3_SHAPE,
    PUBLIC_DEFAULT,
    RENDER_CHILD_PRELUDE,
    SMALL_K_SHAPE,
    aliasing_views,
    allocator_state,
    approx_tanh,
    assert_bitwise_equal,
    assert_near_native,
    assert_not_worse_than_native,
    assert_release_rendering,
    assert_values_equal,
    b_kwargs,
    bf16_rows,
    count_wrapper_compiles,
    device_values_validated,
    eager_forward,
    exact_forward_reference,
    exact_prob,
    make_problem,
    make_prob,
    mn1,
    native_forward,
    offsets_from_counts,
    other_bf16_device,
    pointer_values,
    reference_forward,
    render_env,
    require_bf16_srelu_gpu,
    require_no_host_value_reads,
    run_python_child,
    sm_count,
    srelu_api,
    srelu_execute,
    srelu_fp32,
    store_weights,
    sync_debug_error,
    ternary_weights,
)

# (tanh_clamp_scale, tanh_mode): the plain squared-ReLU (no tanh; the mode is ignored, "approx" is the
# public default) and the tanh-clamped one in both tanh modes.
ACTIVATIONS = [
    pytest.param(None, "approx", id="plain"),
    pytest.param(CLAMP_SCALE, "accurate", id="clamped-accurate"),
    pytest.param(CLAMP_SCALE, "approx", id="clamped-approx"),
]
TANH_MODE_ERROR = "tanh_mode must be 'approx' or 'accurate'"


@pytest.fixture(autouse=True)
def require_sm100():
    require_bf16_srelu_gpu()


def run_srelu(problem, prob, *, clamp_scale, tanh_mode=PUBLIC_DEFAULT, act_input="acc", **kwargs):
    """The public wrapper on ``problem``; ``tanh_mode`` left at PUBLIC_DEFAULT is not passed at all."""
    from cudnn import grouped_gemm_srelu_wrapper_sm100

    if tanh_mode is not PUBLIC_DEFAULT:
        kwargs["tanh_mode"] = tanh_mode
    out = grouped_gemm_srelu_wrapper_sm100(
        a_tensor=problem["a"],
        padded_offsets=problem["offsets"],
        prob_tensor=prob,
        tanh_clamp_scale=clamp_scale,
        act_input=act_input,
        **b_kwargs(problem),
        **kwargs,
    )
    assert out["d_col_tensor"] is None and out["amax_tensor"] is None
    assert out["sfd_row_tensor"] is None and out["sfd_col_tensor"] is None
    return out["c_tensor"], out["d_tensor"]


def intercepted_srelu_api(problem, prob, c, d, monkeypatch, **config):
    """A checked (not compiled) facade whose compiled consumer is a recorder: execute() runs every check and the
    workspace carve, and a launch only appends its arguments to the returned list."""
    api = srelu_api(problem, prob, c, d, **config)
    assert api.check_support()
    launches = []
    monkeypatch.setattr(api._implementation, "_compiled_kernel", lambda *args: launches.append(args))
    return api, launches


# --------------------------------------------------------------------------- numerics

IDENTITY_CELLS = [
    pytest.param(clamp_scale, tanh_mode, width, discrete, act_input, id=f"{label}-{width}-{'discrete' if discrete else 'dense'}-{act_input}")
    for clamp_scale, tanh_mode, label in ((None, "approx", "plain"), (CLAMP_SCALE, "accurate", "clamped-accurate"))
    for width in (256, 1856)
    for discrete in (False, True)
    for act_input in ("acc", "saved")
] + [
    pytest.param(CLAMP_SCALE, "approx", 1856, discrete, "acc", id=f"clamped-approx-1856-{'discrete' if discrete else 'dense'}-acc")
    for discrete in (False, True)
]


@pytest.mark.L0
@pytest.mark.parametrize("clamp_scale, tanh_mode, width, discrete, act_input", IDENTITY_CELLS)
def test_srelu_bf16_identity_weights_match_native(clamp_scale, tanh_mode, width, discrete, act_input):
    """With B_e = I the FP32 accumulator is A exactly, so the kernel reduces to its epilogue.

    C must be A bit for bit in every mode. The plain squared-ReLU uses the native rounding points and
    operation order, so D must equal the eager activation bit for bit. The explicit
    tanh_mode="accurate" is an FP32 Cephes tanh rather than libdevice tanhf, so the clamped D may differ
    from native by one BF16 ulp on a small fraction of the elements. MUFU tanh.approx (the default) is
    held to the error criterion instead: D not less accurate than the native activation against the
    FP32 activation of A, with its measured ratio reported. N = 1856 has a partial last N tile.
    """
    counts = [256, 0, 512]
    problem = make_problem(counts, width, width, discrete=discrete, identity=True, seed=7, a_scale=8.0)
    prob = make_prob(problem["rows"], seed=8)
    c, d = run_srelu(problem, prob, clamp_scale=clamp_scale, tanh_mode=tanh_mode, act_input=act_input)
    routed = problem["routed"]
    assert_bitwise_equal("C (Z_save)", c[:routed], problem["a"][:routed])
    native = eager_forward(problem["a"][:routed, :, 0], prob[:routed, :, 0], clamp_scale)
    if clamp_scale is None:
        assert_bitwise_equal("D (plain)", d[:routed, :, 0], native)
    elif tanh_mode == "accurate":
        assert_near_native("D (clamped accurate)", d[:routed, :, 0], native)
    else:
        reference = srelu_fp32(problem["a"][:routed, :, 0].float(), clamp_scale) * prob[:routed, :, 0]
        assert_not_worse_than_native("D", d[:routed], reference, native, routed=routed, report_ratio=approx_tanh(clamp_scale, tanh_mode))


@pytest.mark.L0
@pytest.mark.parametrize("act_input", ["acc", "saved"])
@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
@pytest.mark.parametrize("clamp_scale, tanh_mode", ACTIVATIONS)
def test_srelu_bf16_against_reference(clamp_scale, tanh_mode, discrete, act_input):
    """Random weights, uneven expert counts with an empty expert and rows past the routed total.

    C and D are compared with the FP32 reference and must not be less accurate than the native
    path (BF16 Z, then the eager activation); MUFU tanh.approx too, with its measured ratio reported.
    """
    counts = [512, 0, 256, 768]
    problem = make_problem(counts, 512, 384, extra_rows=256, discrete=discrete, seed=11, a_scale=8.0)
    prob = make_prob(problem["rows"], seed=12)
    c, d = run_srelu(problem, prob, clamp_scale=clamp_scale, tanh_mode=tanh_mode, act_input=act_input)
    z_ref, h_ref = reference_forward(problem, prob, clamp_scale)
    z_nat, h_nat = native_forward(problem, prob, clamp_scale)
    routed = problem["routed"]
    assert_not_worse_than_native("C (Z_save)", c, z_ref, z_nat, routed=routed)
    assert_not_worse_than_native("D (H)", d, h_ref, h_nat, routed=routed, report_ratio=approx_tanh(clamp_scale, tanh_mode))


@pytest.mark.L0
@pytest.mark.parametrize("clamp_scale, tanh_mode", ACTIVATIONS)
def test_srelu_bf16_saved_input_reproduces_activation_of_c(clamp_scale, tanh_mode):
    """act_input="saved" evaluates the activation at bf16(acc), the C tensor the backward reloads:
    D equals the eager activation of C bit for bit (plain), within one BF16 ulp on a small fraction
    (accurate tanh), and is not less accurate than it against the FP32 activation of C (approx)."""
    problem = make_problem([256, 512], 256, 256, seed=21, a_scale=8.0)
    prob = make_prob(problem["rows"], seed=22)
    c, d = run_srelu(problem, prob, clamp_scale=clamp_scale, tanh_mode=tanh_mode, act_input="saved")
    native = eager_forward(c[:, :, 0], prob[:, :, 0], clamp_scale)
    if clamp_scale is None:
        assert_bitwise_equal("D vs eager(C)", d[:, :, 0], native)
    elif tanh_mode == "accurate":
        assert_near_native("D vs eager(C)", d[:, :, 0], native)
    else:
        reference = srelu_fp32(c[:, :, 0].float(), clamp_scale) * prob[:, :, 0]
        report = approx_tanh(clamp_scale, tanh_mode)
        assert_not_worse_than_native("D_vs_activation_of_C", d, reference, native, routed=problem["rows"], report_ratio=report)


@pytest.mark.L0
def test_srelu_bf16_nan_propagates():
    """NaN in the pre-activation propagates to D in every mode on the BF16 backend (max.NaN relu,
    min.NaN cap; the block-scaled kernel maps it to +0 instead).

    With B = I the NaN of A[3, 5] reaches Z[3, 5] (and, through NaN * 0, possibly the rest of row
    3), never another row.
    """
    problem = make_problem([256], 64, 64, identity=True, seed=31)
    problem["a"][3, 5, 0] = float("nan")
    prob = make_prob(problem["rows"], seed=32, low=0.5, high=1.5)
    other_rows = torch.arange(problem["rows"], device="cuda") != 3
    for clamp_scale, tanh_mode in ((None, "approx"), (None, "accurate"), (CLAMP_SCALE, "accurate"), (CLAMP_SCALE, "approx")):
        _, d = run_srelu(problem, prob, clamp_scale=clamp_scale, tanh_mode=tanh_mode)
        nan = torch.isnan(d[:, :, 0])
        assert bool(nan[3, 5]), (clamp_scale, tanh_mode)
        assert not bool(nan[other_rows].any()), (clamp_scale, tanh_mode)


@pytest.mark.L0
def test_srelu_bf16_zero_prob_rows_give_zero():
    """p = 0 gives exactly zero, plain and clamped (at the default tanh mode)."""
    problem = make_problem([512], 128, 64, seed=41, a_scale=8.0)
    prob = make_prob(problem["rows"], seed=42, low=0.25, high=1.0)
    prob[::3] = 0.0
    for clamp_scale in (None, CLAMP_SCALE):
        _, d = run_srelu(problem, prob, clamp_scale=clamp_scale)
        assert not d[::3].any()


@pytest.mark.L0
def test_srelu_bf16_dynamic_scheduler_matches_static():
    """The dynamic tile scheduler only reorders whole tiles: the result is bitwise the same."""
    problem = make_problem([256, 0, 768, 512], 384, 256, seed=51, a_scale=8.0)
    prob = make_prob(problem["rows"], seed=52)
    static = run_srelu(problem, prob, clamp_scale=CLAMP_SCALE)
    dynamic = run_srelu(problem, prob, clamp_scale=CLAMP_SCALE, use_dynamic_sched=True)
    for name, a, b in zip(("C", "D"), static, dynamic):
        assert_bitwise_equal(f"{name} dynamic vs static", a[: problem["routed"]], b[: problem["routed"]])


@pytest.mark.L0
@pytest.mark.parametrize("mma_tiler_mn, cluster_shape_mn", [((128, 256), (1, 1)), ((256, 128), (2, 1))], ids=["1cta-256n", "2cta-128n"])
def test_srelu_bf16_tile_configs(mma_tiler_mn, cluster_shape_mn):
    """Both MMA widths at the explicit accurate tanh: not less accurate than native, no allowance."""
    problem = make_problem([256, 512], 320, 192, discrete=True, seed=61, a_scale=8.0)
    prob = make_prob(problem["rows"], seed=62)
    c, d = run_srelu(problem, prob, clamp_scale=CLAMP_SCALE, tanh_mode="accurate", mma_tiler_mn=mma_tiler_mn, cluster_shape_mn=cluster_shape_mn)
    z_ref, h_ref = reference_forward(problem, prob, CLAMP_SCALE)
    z_nat, h_nat = native_forward(problem, prob, CLAMP_SCALE)
    assert_not_worse_than_native("C", c, z_ref, z_nat, routed=problem["routed"])
    assert_not_worse_than_native("D", d, h_ref, h_nat, routed=problem["routed"])


@pytest.mark.L0
def test_srelu_bf16_n_major_discrete_weights():
    problem = make_problem([512, 256], 256, 128, discrete=True, b_major="n", seed=71, a_scale=8.0)
    prob = make_prob(problem["rows"], seed=72)
    c, d = run_srelu(problem, prob, clamp_scale=None)
    z_ref, h_ref = reference_forward(problem, prob, None)
    z_nat, h_nat = native_forward(problem, prob, None)
    assert_not_worse_than_native("C", c, z_ref, z_nat, routed=problem["routed"])
    assert_not_worse_than_native("D", d, h_ref, h_nat, routed=problem["routed"])


@pytest.mark.L1
@pytest.mark.parametrize("clamp_scale, tanh_mode", [(None, "approx"), (CLAMP_SCALE, "accurate")], ids=["plain", "clamped-accurate"])
@pytest.mark.parametrize("shape", [NEMOTRON3_SHAPE, SMALL_K_SHAPE], ids=["nemotron3", "e8-k512-n3584"])
def test_srelu_bf16_model_shapes(shape, clamp_scale, tanh_mode):
    experts, k, n = shape["experts"], shape["k"], shape["n"]
    counts = [256 * ((3 * e) % 5) for e in range(experts)]  # uneven, with empty experts
    problem = make_problem(counts, n, k, discrete=True, seed=111, a_scale=8.0)
    prob = make_prob(problem["rows"], seed=112, low=0.0, high=1.0)
    c, d = run_srelu(problem, prob, clamp_scale=clamp_scale, tanh_mode=tanh_mode)
    z_ref, h_ref = reference_forward(problem, prob, clamp_scale)
    z_nat, h_nat = native_forward(problem, prob, clamp_scale)
    assert_not_worse_than_native("C", c, z_ref, z_nat, routed=problem["routed"])
    assert_not_worse_than_native("D", d, h_ref, h_nat, routed=problem["routed"])


# --------------------------------------------------------------------------- class API and CUDA graphs


@pytest.mark.L0
def test_srelu_bf16_class_api_leaves_unrouted_rows_untouched():
    """The class API, with the caller's workspace of scratch_workspace_bytes() bytes, writes only the
    routed rows of the caller's C and D, and matches the wrapper bit for bit (same tanh mode)."""
    problem = make_problem([256, 0, 512], 256, 128, extra_rows=256, seed=81, a_scale=8.0)
    prob = make_prob(problem["rows"], seed=82)
    rows, n = problem["rows"], problem["n"]
    c, d = bf16_rows(rows, n, fill=7.0), bf16_rows(rows, n, fill=7.0)
    api = srelu_api(problem, prob, c, d, tanh_clamp_scale=CLAMP_SCALE, tanh_mode="approx")
    assert api.check_support()
    api.compile()
    srelu_execute(api, problem, prob, c, d, workspace=ws(api))
    routed = problem["routed"]
    assert bool((c[routed:] == 7.0).all()) and bool((d[routed:] == 7.0).all()), "rows past the routed total were written"
    c_w, d_w = run_srelu(problem, prob, clamp_scale=CLAMP_SCALE, tanh_mode="approx")
    assert_bitwise_equal("class C vs wrapper", c[:routed], c_w[:routed])
    assert_bitwise_equal("class D vs wrapper", d[:routed], d_w[:routed])


@pytest.mark.L0
def test_srelu_bf16_cuda_graph_replay():
    """Captured after eager warm-up, the graph replays new inputs exactly as an eager call computes them."""
    require_no_host_value_reads()
    problem = make_problem([256, 0, 512], 256, 128, discrete=True, seed=91, a_scale=8.0)
    prob = make_prob(problem["rows"], seed=92)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            run_srelu(problem, prob, clamp_scale=CLAMP_SCALE)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            c_graph, d_graph = run_srelu(problem, prob, clamp_scale=CLAMP_SCALE)
        fresh = make_problem([256, 0, 512], 256, 128, discrete=True, seed=93, a_scale=8.0)
        problem["a"].copy_(fresh["a"])
        prob.copy_(make_prob(problem["rows"], seed=94))
        graph.replay()
        c_eager, d_eager = run_srelu(problem, prob, clamp_scale=CLAMP_SCALE)
        torch.cuda.synchronize()
        routed = problem["routed"]
        assert_bitwise_equal("C replay vs eager", c_graph[:routed], c_eager[:routed])
        assert_bitwise_equal("D replay vs eager", d_graph[:routed], d_eager[:routed])
    finally:
        graph.reset()


# --------------------------------------------------------------------------- Rule 8 / recipe R9 detectors


@pytest.mark.L0
@pytest.mark.parametrize("entry", ["wrapper", "class"])
def test_srelu_bf16_explicit_stream_capture(entry):
    """An explicit launch stream owns the work even while another stream is current."""
    import cuda.bindings.driver as cuda

    require_no_host_value_reads()
    problem = make_problem([256, 0, 256], 256, 128, discrete=True, seed=231, exact=True)
    prob = exact_prob(problem["rows"], seed=232)
    caller = torch.cuda.current_stream()
    launch = torch.cuda.Stream()
    launch.wait_stream(caller)
    api = None
    with torch.cuda.stream(launch):
        if entry == "class":
            c, d = bf16_rows(problem["rows"], problem["n"]), bf16_rows(problem["rows"], problem["n"])
            api = srelu_api(problem, prob, c, d)
            assert api.check_support()
            api.compile()
            workspace = ws(api)

    def execute():
        stream = cuda.CUstream(launch.cuda_stream)
        if api is None:
            return run_srelu(problem, prob, clamp_scale=None, current_stream=stream)
        srelu_execute(api, problem, prob, c, d, workspace=workspace, current_stream=stream)
        return c, d

    execute()
    launch.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(launch):
            with torch.cuda.graph(graph, stream=launch):
                with torch.cuda.stream(caller):
                    captured = execute()
        with torch.cuda.stream(launch):
            problem["a"].neg_()
            prob.mul_(2)
            graph.replay()
        caller.wait_stream(launch)
        expected = exact_forward_reference(problem["a"], problem["b_logical"], problem["counts"], prob)
        for name, actual, reference in zip(("C", "D"), captured, expected):
            assert_values_equal(name, actual[: problem["routed"], :, 0], reference)
    finally:
        graph.reset()


@pytest.mark.L0
@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
def test_srelu_bf16_wrapper_never_synchronizes(discrete, monkeypatch):
    """Rule 8 / recipe R9: a cold wrapper call (backend selection, check_support, compile, per-call
    outputs and scratch, launch) and a warm call run under torch.cuda.set_sync_debug_mode("error");
    the caller's mode is restored afterwards. Offsets and pointer tables are never read on the host."""
    from cudnn.gemm.cutedsl.grouped.srelu import api as srelu_module

    require_no_host_value_reads()
    compiles = count_wrapper_compiles(monkeypatch, srelu_module, "GroupedGemmSreluSm100", "_cache_of_GroupedGemmSreluSm100Objects")
    problem = make_problem([256, 512], 256, 128, discrete=discrete, seed=101, a_scale=8.0)
    prob = make_prob(problem["rows"], seed=102)
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    with sync_debug_error():
        cold = run_srelu(problem, prob, clamp_scale=CLAMP_SCALE)
        warm = run_srelu(problem, prob, clamp_scale=CLAMP_SCALE)
    assert torch.cuda.get_sync_debug_mode() == previous
    torch.cuda.synchronize()
    assert len(compiles) == 1, "the guarded first call must have been the cold build"
    for name, a, b in zip(("C", "D"), cold, warm):
        assert_bitwise_equal(f"{name} cold vs warm", a, b)


@pytest.mark.L0
def test_srelu_bf16_class_api_allocates_nothing_and_never_synchronizes():
    """Rule 8 / recipe R9 on the class API with caller-owned outputs and workspace: construction,
    check_support(), compile() and three executes run under sync-debug "error", and the caching
    allocator records no allocation (neither bytes held nor allocations made) across compile() and
    the three executes: compile() binds fake tensors and pointers only, execute() carves the caller's
    workspace and launches."""
    require_no_host_value_reads()
    problem = make_problem([256, 0, 512], 256, 128, extra_rows=256, discrete=True, seed=141, a_scale=8.0)
    prob = make_prob(problem["rows"], seed=142)
    c, d = bf16_rows(problem["rows"], 256), bf16_rows(problem["rows"], 256)
    torch.cuda.synchronize()
    with sync_debug_error():
        api = srelu_api(problem, prob, c, d, tanh_clamp_scale=CLAMP_SCALE, tanh_mode="approx")
        assert api.check_support()
        workspace = ws(api)
        before = allocator_state()
        api.compile()
        compiled = allocator_state()
        for _ in range(3):
            srelu_execute(api, problem, prob, c, d, workspace=workspace)
        executed = allocator_state()
    torch.cuda.synchronize()
    assert compiled == before, f"compile() allocated: (bytes held, allocations made) {before} -> {compiled}"
    assert executed == before, f"three executes allocated: (bytes held, allocations made) {before} -> {executed}"
    c_w, d_w = run_srelu(problem, prob, clamp_scale=CLAMP_SCALE, tanh_mode="approx")
    routed = problem["routed"]
    assert_bitwise_equal("class C vs wrapper", c[:routed], c_w[:routed])
    assert_bitwise_equal("class D vs wrapper", d[:routed], d_w[:routed])


@pytest.mark.L0
def test_srelu_bf16_first_execute_inside_capture_then_changed_replay():
    """Rule 8: the class API's first execute happens inside CUDA-graph capture, with offsets and
    pointer tensors no execute has seen and a framework-neutral workspace view. It synchronizes
    nothing (sync-debug "error" scoped to the execute inside the capture) and allocates nothing.
    Replays then follow changed A, prob, expert boundaries and weight pointers; each is compared
    exactly with an independent float64 reference on NaN-filled outputs."""
    from cudnn.frost.buffers import DeviceView

    require_no_host_value_reads()
    counts, n, k = [512, 0, 256], 256, 128
    problem = make_problem(counts, n, k, extra_rows=256, discrete=True, exact=True, seed=151)
    rows = problem["rows"]
    prob = exact_prob(rows, seed=152)
    c, d = bf16_rows(rows, n, fill=float("nan")), bf16_rows(rows, n, fill=float("nan"))
    api = srelu_api(problem, prob, c, d)
    assert api.check_support()
    api.compile()
    scratch = ws(api)
    scratch.fill_(0xFF)  # stale bytes the first launch's helper kernel must overwrite
    view = DeviceView(scratch.data_ptr(), (scratch.numel(),), "uint8", scratch.device.index)
    problem["offsets"] = problem["offsets"].clone()
    problem["b_ptrs"] = problem["b_ptrs"].clone()
    alive = [problem["b_storage"]]
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            before = allocator_state()
            with sync_debug_error():
                srelu_execute(api, problem, prob, c, d, workspace=view)
            assert allocator_state() == before, "the first execute allocated inside the capture"
        for step, new_counts in enumerate(([512, 0, 256], [256, 256, 0], [0, 768, 256])):
            fresh = make_problem(new_counts, n, k, extra_rows=rows - sum(new_counts), exact=True, seed=160 + step)
            problem["a"].copy_(fresh["a"])
            prob.copy_(exact_prob(rows, seed=170 + step))
            problem["offsets"].copy_(offsets_from_counts(new_counts))
            storage = store_weights(fresh["b_logical"], "k")
            alive.append(storage)
            problem["b_ptrs"].copy_(pointer_values(storage))
            c.fill_(float("nan"))
            d.fill_(float("nan"))
            graph.replay()
            routed = sum(new_counts)
            ref_c, ref_d = exact_forward_reference(problem["a"], fresh["b_logical"], new_counts, prob)
            assert_values_equal(f"replay {step} C", c[:routed, :, 0], ref_c)
            assert_values_equal(f"replay {step} D", d[:routed, :, 0], ref_d)
            assert bool(torch.isnan(c[routed:]).all()) and bool(torch.isnan(d[routed:]).all()), f"replay {step} wrote rows past the routed total"
    finally:
        graph.reset()


# --------------------------------------------------------------------------- the wrapper's plan cache


@pytest.mark.L0
def test_srelu_bf16_plain_mode_shares_one_kernel_across_tanh_modes(monkeypatch):
    """The plain squared-ReLU evaluates no tanh: the default (omitted), "approx" and "accurate" are all
    accepted, hit one wrapper cache entry and one compile, and give the same bytes."""
    from cudnn import grouped_gemm_srelu_wrapper_sm100
    from cudnn.gemm.cutedsl.grouped.srelu import api as srelu_module

    assert inspect.signature(grouped_gemm_srelu_wrapper_sm100).parameters["tanh_mode"].default == "approx"
    assert inspect.signature(srelu_module.GroupedGemmSreluSm100).parameters["tanh_mode"].default == "approx"
    compiles = count_wrapper_compiles(monkeypatch, srelu_module, "GroupedGemmSreluSm100", "_cache_of_GroupedGemmSreluSm100Objects")
    problem = make_problem([256, 0, 512], 256, 128, discrete=True, seed=171, a_scale=8.0)
    prob = make_prob(problem["rows"], seed=172)
    default = run_srelu(problem, prob, clamp_scale=None)
    for mode in ("approx", "accurate"):
        out = run_srelu(problem, prob, clamp_scale=None, tanh_mode=mode)
        for name, got, want in zip(("C", "D"), out, default):
            assert_bitwise_equal(f"plain {name}, tanh_mode={mode} vs default", got, want)
    assert len(compiles) == 1, f"plain mode compiled {len(compiles)} kernels for three tanh_mode spellings"
    assert len(srelu_module._cache_of_GroupedGemmSreluSm100Objects) == 1


@pytest.mark.L0
def test_srelu_bf16_wrapper_reuses_one_plan_across_runtime_values(monkeypatch):
    """Cache-miss regression: M, the offset values and the weight pointers are runtime values of one
    plan. Four calls that change all three (empty experts included) compile once and keep one cache
    entry, and every call equals the exact float64 reference."""
    from cudnn.gemm.cutedsl.grouped.srelu import api as srelu_module

    compiles = count_wrapper_compiles(monkeypatch, srelu_module, "GroupedGemmSreluSm100", "_cache_of_GroupedGemmSreluSm100Objects")
    for step, counts in enumerate(([256, 0, 512], [768, 256, 0], [0, 0, 256], [512, 512, 512])):
        problem = make_problem(counts, 256, 128, discrete=True, exact=True, seed=180 + step)
        prob = exact_prob(problem["rows"], seed=190 + step)
        c, d = run_srelu(problem, prob, clamp_scale=None)
        ref_c, ref_d = exact_forward_reference(problem["a"], problem["b_logical"], counts, prob)
        assert_values_equal(f"call {step} C", c[:, :, 0], ref_c)
        assert_values_equal(f"call {step} D", d[:, :, 0], ref_d)
    assert len(compiles) == 1, f"{len(compiles)} compiles for one plan"
    assert len(srelu_module._cache_of_GroupedGemmSreluSm100Objects) == 1


# --------------------------------------------------------------------------- discrete weight descriptors

DESCRIPTOR_TILINGS = [
    pytest.param((256, 256), (2, 1), False, id="2cta-static"),
    pytest.param((128, 128), (1, 1), False, id="1cta-static"),
    pytest.param((256, 256), (2, 1), True, id="2cta-dynamic"),
]


@pytest.mark.L0
@pytest.mark.parametrize("mma_tiler_mn, cluster_shape_mn, use_dynamic_sched", DESCRIPTOR_TILINGS)
def test_srelu_bf16_discrete_weight_descriptors_follow_new_pointers(mma_tiler_mn, cluster_shape_mn, use_dynamic_sched):
    """The per-expert B descriptors live in the caller's workspace and every launch's helper kernel
    rewrites them at the same address; the TMA descriptor cache outlives launches, so each CTA must
    acquire an expert's descriptor before loading through it (moe_utils.acquire_tma_desc).

    One compiled API and one workspace buffer: two eager calls whose fresh pointer tables name freshly
    allocated weights, then a captured graph replayed after b_ptrs.copy_() of new weights (the last
    replay with new expert boundaries too). Weights are randint(-1, 2); outputs are NaN-filled before
    every launch and compared exactly with an independent float64 reference. Every weight set stays
    alive, so a stale descriptor reads other values rather than reused memory. M is sized to the SM
    count, so CTAs process several tiles and revisit experts within and across launches. A detector,
    not a proof: a stale descriptor is timing-dependent. The workspace is filled with 0xFF before the
    first eager call and zeroed before the second and before the capture.
    """
    require_no_host_value_reads()
    n, k = 256, 128
    m_tiles = sm_count()
    counts = [256 * (m_tiles // 4), 0, 256 * (m_tiles // 3)]
    counts.append(256 * m_tiles - sum(counts))
    problem = make_problem(counts, n, k, discrete=True, exact=True, seed=201)
    rows = problem["rows"]
    prob = exact_prob(rows, seed=202)
    c, d = bf16_rows(rows, n), bf16_rows(rows, n)
    api = srelu_api(problem, prob, c, d, mma_tiler_mn=mma_tiler_mn, cluster_shape_mn=cluster_shape_mn, use_dynamic_sched=use_dynamic_sched)
    assert api.check_support()
    api.compile()
    workspace = ws(api)
    alive = [problem["b_storage"]]

    def fresh_weights(seed):
        logical = ternary_weights(problem["experts"], n, k, seed=seed)
        alive.append(store_weights(logical, "k"))
        return logical, pointer_values(alive[-1])

    def launch_and_check(label, logical, counts_now, launch):
        c.fill_(float("nan"))
        d.fill_(float("nan"))
        launch()
        routed = sum(counts_now)
        ref_c, ref_d = exact_forward_reference(problem["a"], logical, counts_now, prob)
        assert_values_equal(f"{label} C", c[:routed, :, 0], ref_c)
        assert_values_equal(f"{label} D", d[:routed, :, 0], ref_d)
        assert bool(torch.isnan(c[routed:]).all()) and bool(torch.isnan(d[routed:]).all()), f"{label} wrote rows past the routed total"

    # test/AGENTS.md: poison the caller workspace with both 0xFF and 0x00; each launch's helper kernel rewrites every slot it reads.
    for call, (seed, poison) in enumerate(((211, 0xFF), (212, 0x00))):
        logical, pointers = fresh_weights(seed)
        problem["b_ptrs"] = pointers.cuda()
        workspace.fill_(poison)
        launch_and_check(f"eager call {call}", logical, counts, lambda: srelu_execute(api, problem, prob, c, d, workspace=workspace))
    workspace.zero_()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            srelu_execute(api, problem, prob, c, d, workspace=workspace)
        late_counts = [256 * (m_tiles // 2), 256 * (m_tiles // 4), 0, 0]
        for replay, (seed, counts_now) in enumerate(((213, counts), (214, counts), (215, late_counts))):
            logical, pointers = fresh_weights(seed)
            problem["b_ptrs"].copy_(pointers)
            problem["offsets"].copy_(offsets_from_counts(counts_now))
            launch_and_check(f"replay {replay}", logical, counts_now, graph.replay)
    finally:
        graph.reset()


# --------------------------------------------------------------------------- caller-owned scratch (recipe R2)


@pytest.mark.L0
@pytest.mark.parametrize("discrete, use_dynamic_sched", [(False, False), (True, False), (True, True)], ids=["dense", "discrete-static", "discrete-dynamic"])
def test_srelu_bf16_scratch_workspace_bytes(discrete, use_dynamic_sched):
    """The facade's BF16 scratch_workspace_bytes() runs the lazy check_support() and reports a positive
    multiple of 128 that covers one 128-byte descriptor slot per expert in discrete mode."""
    problem = make_problem([256, 0, 512], 64, 64, discrete=discrete, seed=221)
    rows = problem["rows"]
    api = srelu_api(problem, make_prob(rows, seed=222), bf16_rows(rows, 64), bf16_rows(rows, 64), use_dynamic_sched=use_dynamic_sched)
    nbytes = api.scratch_workspace_bytes()
    assert nbytes >= 128 and nbytes % 128 == 0, nbytes
    if discrete:
        assert nbytes >= 128 * problem["experts"], nbytes
    assert api.scratch_workspace_bytes() == nbytes


@pytest.mark.L0
@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
def test_srelu_bf16_workspace_rejected_before_launch(discrete, monkeypatch):
    """Recipe R2: execute() requires the caller's workspace and checks it before any launch, also for
    a call with no rows: missing, too short, misaligned, non-contiguous, host and wrong-device buffers
    raise; the compiled consumer is intercepted, so no rejected buffer reaches a kernel. A correct
    buffer launches once per call with rows, and not at all without rows."""
    from cudnn.frost.buffers import DeviceView

    problem = make_problem([256, 0, 512], 64, 64, discrete=discrete, seed=231)
    rows = problem["rows"]
    prob = make_prob(rows, seed=232)
    c, d = bf16_rows(rows, 64), bf16_rows(rows, 64)
    api, launches = intercepted_srelu_api(problem, prob, c, d, monkeypatch)
    nbytes = api.scratch_workspace_bytes()
    index = torch.cuda.current_device()
    storage = torch.empty(2 * nbytes + 256, dtype=torch.uint8, device="cuda")
    base = -storage.data_ptr() % 128
    aligned = storage[base : base + nbytes]
    host = torch.empty(nbytes + 128, dtype=torch.uint8)
    host = host[-host.data_ptr() % 128 :][:nbytes]
    if torch.cuda.device_count() > 1:
        other = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{(index + 1) % torch.cuda.device_count()}")
    else:
        other = DeviceView(aligned.data_ptr(), (nbytes,), "uint8", index + 1)  # an ordinal that does not exist
    cases = [
        (None, f"requires a {nbytes}-byte workspace"),
        (aligned[: nbytes - 64], f"needs a {nbytes}-byte workspace, got {nbytes - 64} bytes"),
        (storage[base + 64 : base + 64 + nbytes], "must be 128-byte aligned"),
        (storage[base : base + 2 * nbytes : 2], "must be contiguous"),
        (host, "workspace must be a CUDA device buffer"),
        (other, "workspace must be on|does not exist"),
    ]
    no_rows = dict(
        a_tensor=mn1(torch.empty((0, 64), dtype=torch.bfloat16, device="cuda")),
        c_tensor=mn1(torch.empty((0, 64), dtype=torch.bfloat16, device="cuda")),
        d_tensor=mn1(torch.empty((0, 64), dtype=torch.bfloat16, device="cuda")),
        prob_tensor=torch.empty((0, 1, 1), dtype=torch.float32, device="cuda"),
    )
    for workspace, match in cases:
        for overrides in ({}, no_rows):
            with pytest.raises(ValueError, match=match):
                srelu_execute(api, problem, prob, c, d, workspace=workspace, **overrides)
    assert not launches, "a rejected workspace reached the launch"
    srelu_execute(api, problem, prob, c, d, workspace=aligned)
    assert len(launches) == 1
    srelu_execute(api, problem, prob, c, d, workspace=aligned, **no_rows)
    assert len(launches) == 1, "a call with no rows launched"


@pytest.mark.L0
@pytest.mark.parametrize(
    "discrete, operand",
    [(False, name) for name in ("a_tensor", "c_tensor", "d_tensor", "padded_offsets", "prob_tensor", "b_tensor")] + [(True, "b_ptrs")],
)
def test_srelu_bf16_workspace_alias_rejected_before_launch(discrete, operand, monkeypatch):
    """Recipe R2 (test/AGENTS.md, caller workspace alias regressions): scratch that overlaps an operand's
    byte span is rejected before launch with that operand's name; the same operand right after the
    workspace in one allocation is accepted. Every other operand stays clear of the tested interval."""
    if operand in ("padded_offsets", "b_ptrs"):
        require_no_host_value_reads()  # the uninitialized views would fail the debug value checks first
    problem = make_problem([256, 0, 512], 64, 64, discrete=discrete, seed=241)
    rows = problem["rows"]
    prob = make_prob(rows, seed=242)
    c, d = bf16_rows(rows, 64), bf16_rows(rows, 64)
    api, launches = intercepted_srelu_api(problem, prob, c, d, monkeypatch)
    nbytes = api.scratch_workspace_bytes()
    operands = dict(a_tensor=problem["a"], c_tensor=c, d_tensor=d, padded_offsets=problem["offsets"], prob_tensor=prob)
    operands.update({"b_ptrs": problem["b_ptrs"]} if discrete else {"b_tensor": problem["b"]})
    storage, disjoint, overlapping = aliasing_views(operands[operand], nbytes)
    srelu_execute(api, problem, prob, c, d, workspace=storage[:nbytes], **{operand: disjoint})
    assert len(launches) == 1
    with pytest.raises(ValueError, match="workspace must not overlap " + operand):
        srelu_execute(api, problem, prob, c, d, workspace=storage[:nbytes], **{operand: overlapping})
    assert len(launches) == 1, "an aliased workspace reached the launch"


@pytest.mark.L0
def test_block_scaled_srelu_takes_no_workspace(request):
    """The block-scaled backend keeps its own descriptor workspace: the facade reports 0 scratch bytes
    and its execute() rejects workspace= (a provided but unused argument raises, Rule 1)."""
    from cudnn import GroupedGemmSreluSm100
    from gemm.cutedsl.test_grouped_gemm_srelu_utils import (
        allocate_grouped_gemm_input_tensors,
        allocate_grouped_gemm_output_tensors,
        grouped_gemm_srelu_init,
    )

    cfg = grouped_gemm_srelu_init(
        request, torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn, "n", torch.float32, (256, 256), (2, 1), 32, torch.float8_e8m0fnu, False, True
    )
    inputs = allocate_grouped_gemm_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        l=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
    )
    outputs = allocate_grouped_gemm_output_tensors(
        tensor_m=inputs["tensor_m"],
        n=cfg["n"],
        l=cfg["l"],
        ab_dtype=cfg["ab_dtype"],
        c_dtype=cfg["c_dtype"],
        d_dtype=cfg["d_dtype"],
        cd_major=cfg["cd_major"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
    )
    api = GroupedGemmSreluSm100(
        sample_a=inputs["a_tensor"],
        sample_b=inputs["b_tensor"],
        sample_c=outputs["c_tensor"],
        sample_d=outputs["d_tensor"],
        sample_sfa=inputs["sfa_tensor"],
        sample_sfb=inputs["sfb_tensor"],
        sample_padded_offsets=inputs["padded_offsets_tensor"],
        sample_alpha=inputs["alpha_tensor"],
        sample_amax=outputs.get("amax_tensor"),
        sample_d_col=outputs["d_col_tensor"],
        sample_sfd_row=outputs.get("sfd_row_tensor"),
        sample_sfd_col=outputs.get("sfd_col_tensor"),
        sample_norm_const=inputs.get("norm_const_tensor"),
        sample_prob=inputs.get("prob_tensor"),
        acc_dtype=cfg["acc_dtype"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sf_vec_size=cfg["sf_vec_size"],
        vector_f32=cfg["vector_f32"],
        m_aligned=cfg["m_aligned"],
        discrete_col_sfd=cfg["discrete_col_sfd"],
    )
    try:
        assert api.check_support()
    except (ValueError, NotImplementedError) as exc:
        pytest.skip(f"Unsupported block-scaled testcase: {exc}")
    assert api.scratch_workspace_bytes() == 0
    with pytest.raises(ValueError, match="workspace is supported only by the BF16 kernel"):
        api.execute(
            a_tensor=inputs["a_tensor"],
            b_tensor=inputs["b_tensor"],
            c_tensor=outputs["c_tensor"],
            d_tensor=outputs["d_tensor"],
            sfa_tensor=inputs["sfa_tensor"],
            sfb_tensor=inputs["sfb_tensor"],
            padded_offsets=inputs["padded_offsets_tensor"],
            alpha_tensor=inputs["alpha_tensor"],
            d_col_tensor=outputs["d_col_tensor"],
            sfd_row_tensor=outputs.get("sfd_row_tensor"),
            sfd_col_tensor=outputs.get("sfd_col_tensor"),
            norm_const_tensor=inputs.get("norm_const_tensor"),
            prob_tensor=inputs.get("prob_tensor"),
            amax_tensor=outputs.get("amax_tensor"),
            workspace=torch.empty(128, dtype=torch.uint8, device="cuda"),
        )


# --------------------------------------------------------------------------- lazy facade and compile placement


@pytest.mark.L0
def test_srelu_bf16_facade_defers_validation_to_check_support(caplog):
    """The facade only stores its arguments: backend selection, the implementation's constructor and its
    checks run in check_support(), which compile() and scratch_workspace_bytes() reach too. Constructing
    never raises for these errors; an unknown framework is still refused at construction. execute()
    before compile() raises; a BF16 scale control at execute() is refused by name. The experimental
    warning is logged once per implementation class, at check_support()."""
    import numpy as np

    from cudnn import GroupedGemmSreluSm100
    from cudnn.api_base import _reset_experimental_api_warning_registry

    problem = make_problem([256, 256], 64, 64, seed=251)
    rows = problem["rows"]
    prob = make_prob(rows, seed=252)
    c, d = bf16_rows(rows, 64), bf16_rows(rows, 64)
    base = dict(sample_a=problem["a"], sample_c=c, sample_d=d, sample_padded_offsets=problem["offsets"], sample_prob=prob)
    cases = [
        (dict(sample_b=problem["b"].to(torch.float8_e4m3fn)), "mixed dtype families"),
        (dict(sample_b=problem["b"], sample_alpha=torch.ones(2, device="cuda")), "grouped_gemm_srelu_sm100: BF16 forbids scale control sample_alpha"),
        (dict(sample_b=problem["b"], num_experts=2, b_shape=(64, 64), b_dtype=torch.bfloat16), "but not both"),
        (dict(sample_b=problem["b"], tanh_mode=None), re.escape(f"{TANH_MODE_ERROR}, got None")),
        (dict(sample_b=problem["b"], tanh_mode="fast"), re.escape(f"{TANH_MODE_ERROR}, got 'fast'")),
        (dict(sample_b=problem["b"], act_input="z"), "act_input"),
    ]
    for extra, match in cases:
        for entry in ("check_support", "compile", "scratch_workspace_bytes"):
            api = GroupedGemmSreluSm100(**base, **extra)
            with pytest.raises(ValueError, match=match):
                getattr(api, entry)()
    execute_args = dict(a_tensor=problem["a"], sfa_tensor=None, padded_offsets=problem["offsets"], alpha_tensor=None, c_tensor=c, d_tensor=d)
    execute_args.update(b_tensor=problem["b"], prob_tensor=prob, workspace=torch.empty(128, dtype=torch.uint8, device="cuda"))
    with pytest.raises(RuntimeError, match="Kernel not compiled"):
        GroupedGemmSreluSm100(**base, sample_b=problem["b"]).execute(**execute_args)
    with pytest.raises(ValueError, match="Unsupported tensor framework 'numpy'"):
        GroupedGemmSreluSm100(sample_a=np.zeros((256, 64, 1), dtype=np.float32), sample_b=problem["b"])

    _reset_experimental_api_warning_registry()
    caplog.set_level(logging.WARNING)
    caplog.clear()
    api = GroupedGemmSreluSm100(**base, sample_b=problem["b"])
    assert "GroupedGemmSreluBf16API is an experimental API" not in caplog.text, "the facade's constructor built the implementation"
    assert api.check_support()
    assert caplog.text.count("GroupedGemmSreluBf16API is an experimental API") == 1
    with pytest.raises(RuntimeError, match="Kernel not compiled"):
        api.execute(**execute_args)
    with pytest.raises(ValueError, match="BF16 forbids scale control alpha_tensor"):
        api.execute(**{**execute_args, "alpha_tensor": torch.ones(2, device="cuda")})
    assert GroupedGemmSreluSm100(**base, sample_b=problem["b"]).check_support()
    assert caplog.text.count("GroupedGemmSreluBf16API is an experimental API") == 1


@pytest.mark.L0
@pytest.mark.parametrize("placement", ["current-device", "other-device"])
def test_srelu_bf16_compile_pins_operand_device_and_compile_lock(placement, monkeypatch):
    """check_support() reads the capability of the operands' GPU and compile() runs cute.compile under
    the process-wide DSL compile lock with the operands' device current, restoring the caller's device;
    with two GPUs the operands live on the one that is not current."""
    import cutlass.cute as cute

    import cudnn.gemm.cutedsl.grouped.srelu._bf16_api as bf16_api

    caller = torch.cuda.current_device()
    index = caller
    if placement == "other-device":
        if torch.cuda.device_count() < 2:
            pytest.skip("needs two visible GPUs")
        index = (caller + 1) % torch.cuda.device_count()
        if torch.cuda.get_device_capability(index) not in ((10, 0), (10, 3)):
            pytest.skip(f"cuda:{index} is not SM100/SM103")
    capabilities = []
    real_capability = bf16_api.get_compute_capability

    def capability(*args, **kwargs):
        capabilities.append(args[0] if args else kwargs.get("device_index"))
        return real_capability(*args, **kwargs)

    spy = CompileSpy(cute.compile)
    monkeypatch.setattr(bf16_api, "get_compute_capability", capability)
    monkeypatch.setattr(cute, "compile", spy)
    with torch.cuda.device(index):
        problem = make_problem([256, 0, 512], 64, 64, discrete=True, seed=261)
        prob = make_prob(problem["rows"], seed=262)
        c, d = bf16_rows(problem["rows"], 64), bf16_rows(problem["rows"], 64)
    api = srelu_api(problem, prob, c, d)
    assert api.check_support()
    api.compile()
    assert torch.cuda.current_device() == caller, "compile() did not restore the caller's device"
    assert capabilities and all(seen == index for seen in capabilities), f"capability queried for {capabilities}, operands on cuda:{index}"
    assert spy.calls and all(seen == (index, True) for seen in spy.calls), f"(device, lock held) at cute.compile: {spy.calls}, operands on cuda:{index}"


# --------------------------------------------------------------------------- support checks


def _wrapper_error(problem, prob, match, **kwargs):
    with pytest.raises((ValueError, RuntimeError), match=match):
        run_srelu(problem, prob, **{"clamp_scale": CLAMP_SCALE, **kwargs})


@pytest.mark.L0
def test_srelu_bf16_rejects_invalid_configurations():
    from cudnn import grouped_gemm_srelu_wrapper_sm100

    problem = make_problem([256, 256], 128, 64, seed=121)
    prob = make_prob(problem["rows"], seed=122)
    alpha = torch.ones(2, dtype=torch.float32, device="cuda")
    _wrapper_error(problem, prob, "BF16 forbids scale control alpha_tensor", alpha_tensor=alpha)
    _wrapper_error(problem, prob, "BF16 forbids scale control bias_tensor", bias_tensor=torch.zeros((128, 2), dtype=torch.bfloat16, device="cuda"))
    for clamp_scale in (CLAMP_SCALE, None):
        _wrapper_error(problem, prob, re.escape(f"{TANH_MODE_ERROR}, got None"), clamp_scale=clamp_scale, tanh_mode=None)
        _wrapper_error(problem, prob, re.escape(f"{TANH_MODE_ERROR}, got 'fast'"), clamp_scale=clamp_scale, tanh_mode="fast")
    _wrapper_error(problem, prob, "act_input", act_input="z")
    for bad_scale in (0.0, -2.0, float("inf"), float("nan"), 1e-40):
        _wrapper_error(problem, prob, "clamp_scale", clamp_scale=bad_scale)
    _wrapper_error(problem, prob, "c_dtype", c_dtype=torch.float16)
    with pytest.raises(ValueError, match="prob_tensor is required"):
        run_srelu(problem, None, clamp_scale=CLAMP_SCALE)

    narrow = make_problem([256], 48, 64, seed=123)
    _wrapper_error(narrow, make_prob(256, seed=124), "multiple of 32")
    odd_k = make_problem([256], 64, 36, seed=125)
    _wrapper_error(odd_k, make_prob(256, seed=126), "multiple of 8")

    fp8_b = problem["b"].to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="mixed dtype"):
        run_srelu({**problem, "b": fp8_b}, prob, clamp_scale=CLAMP_SCALE)
    # The weights are checked before the backend is selected, for both backends.
    pointers = torch.zeros(2, dtype=torch.int64, device="cuda")
    with pytest.raises(ValueError, match="not both"):
        run_srelu(problem, prob, clamp_scale=None, b_ptrs=pointers)
    with pytest.raises(ValueError, match="Must provide either"):
        grouped_gemm_srelu_wrapper_sm100(a_tensor=problem["a"], padded_offsets=problem["offsets"], prob_tensor=prob)
    with pytest.raises(ValueError, match="n and b_dtype are required for discrete mode"):
        grouped_gemm_srelu_wrapper_sm100(a_tensor=problem["a"], b_ptrs=pointers, padded_offsets=problem["offsets"], prob_tensor=prob)


SRELU_DEVICE_VALUE_CHILD = r"""
from cudnn import GroupedGemmSreluSm100
from cudnn.gemm.cutedsl.grouped import backend_utils

assert backend_utils.DEBUG_VALIDATE_DEVICE_VALUES, "CUDNN_FE_GROUPED_GEMM_VALIDATE_DEVICE_VALUES=1 was not read at import"


def mn1(t):
    rows, cols = t.shape
    return t.as_strided((rows, cols, 1), (cols, 1, rows * cols))


rows, n, k, experts = 512, 64, 64, 2
a = mn1(torch.randn(rows, k, device="cuda").to(torch.bfloat16))
weights = torch.randn(experts, n, k, device="cuda").to(torch.bfloat16)
b_ptrs = torch.tensor([weights[e].data_ptr() for e in range(experts)], dtype=torch.int64, device="cuda")
prob = torch.rand(rows, 1, 1, device="cuda")
c = mn1(torch.empty(rows, n, dtype=torch.bfloat16, device="cuda"))
d = mn1(torch.empty(rows, n, dtype=torch.bfloat16, device="cuda"))
offsets = torch.tensor([256, 512], dtype=torch.int32, device="cuda")
api = GroupedGemmSreluSm100(
    sample_a=a, sample_c=c, sample_d=d, sample_padded_offsets=offsets, sample_prob=prob, num_experts=experts, b_shape=(n, k), b_dtype=torch.bfloat16
)
assert api.check_support()
launches = []
api._implementation._compiled_kernel = lambda *args: launches.append(args)
workspace = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")


def run(**overrides):
    kwargs = dict(a_tensor=a, sfa_tensor=None, padded_offsets=offsets, alpha_tensor=None, c_tensor=c, d_tensor=d, b_ptrs=b_ptrs, prob_tensor=prob)
    kwargs.update(overrides)
    api.execute(workspace=workspace, **kwargs)


def ends(values):
    return torch.tensor(values, dtype=torch.int32, device="cuda")


for values, pattern in (
    ([128, 512], r"padded_offsets\[0\] must be 256-aligned, got 128"),
    ([256, 768], r"padded_offsets last value must be in \[1, 512\], got 768"),
    ([512, 256], r"padded_offsets must be a non-decreasing cumulative sum"),
    ([0, 0], r"padded_offsets last value must be in \[1, 512\], got 0"),
):
    expect_value_error(pattern, lambda: run(padded_offsets=ends(values)))
for entries in ([weights[0].data_ptr(), 0], [weights[0].data_ptr(), weights[1].data_ptr() + 8]):
    expect_value_error(r"b_ptrs entries must be non-null and 16-byte aligned", lambda: run(b_ptrs=torch.tensor(entries, dtype=torch.int64, device="cuda")))
assert not launches, "a rejected value reached the launch"
run()
assert len(launches) == 1, launches
empty = dict(
    a_tensor=mn1(torch.empty(0, k, dtype=torch.bfloat16, device="cuda")),
    c_tensor=mn1(torch.empty(0, n, dtype=torch.bfloat16, device="cuda")),
    d_tensor=mn1(torch.empty(0, n, dtype=torch.bfloat16, device="cuda")),
    prob_tensor=torch.empty(0, 1, 1, device="cuda"),
)
run(padded_offsets=ends([0, 0]), **empty)
assert len(launches) == 1, "a call with no rows launched"
print("SRELU_DEVICE_VALUE_CHECKS_OK")
"""


@pytest.mark.L0
def test_srelu_bf16_offset_and_pointer_values_checked_only_on_request(tmp_path, monkeypatch):
    """Rule 3: execute() never reads offsets or the pointer table to the host. In this process invalid
    values reach the intercepted launch without a host read. In a fresh process with
    CUDNN_FE_GROUPED_GEMM_VALIDATE_DEVICE_VALUES=1 set before import, the same kind of values is rejected
    before launch (256-alignment, last offset in [1, M] -- all-zero offsets included --, non-decreasing
    ends, non-null 16-byte-aligned pointers), valid ones launch, and a call with no rows returns before
    any value check (all-zero offsets accepted)."""
    from cudnn.gemm.cutedsl.grouped import backend_utils

    if not device_values_validated():

        def forbidden(*args, **kwargs):
            raise AssertionError("execute() read device values back to the host")

        for name in ("_check_not_capturing", "_host_int_values", "_host_pointer_values"):
            monkeypatch.setattr(backend_utils, name, forbidden)
        problem = make_problem([256, 256], 64, 64, discrete=True, seed=271)
        prob = make_prob(problem["rows"], seed=272)
        c, d = bf16_rows(problem["rows"], 64), bf16_rows(problem["rows"], 64)
        api, launches = intercepted_srelu_api(problem, prob, c, d, monkeypatch)
        bad_offsets = torch.tensor([128, 768], dtype=torch.int32, device="cuda")
        bad_pointers = torch.tensor([0, 8], dtype=torch.int64, device="cuda")
        srelu_execute(api, problem, prob, c, d, workspace=ws(api), padded_offsets=bad_offsets, b_ptrs=bad_pointers)
        assert len(launches) == 1
    run_python_child(SRELU_DEVICE_VALUE_CHILD, env={DEVICE_VALUES_ENV: "1"}, cwd=tmp_path, marker="SRELU_DEVICE_VALUE_CHECKS_OK")


@pytest.mark.L0
@pytest.mark.parametrize("call", ["wrapper", "class"])
def test_srelu_bf16_launches_on_the_operands_device(call):
    """python/cudnn/AGENTS.md ("Compiled CuTe launchables belong to their CUDA context", "Restore the caller's device
    after a launch on another device"): the operands live on a GPU that is not current and no stream is given. The
    wrapper allocates its outputs and scratch on that GPU's current stream and launches there; the class API
    compiles and launches on the operands' device. Both equal the same call made with that GPU current, and the
    caller's device stays current."""
    caller, other = other_bf16_device()
    with torch.cuda.device(other):
        problem = make_problem([256, 0, 512], 256, 128, discrete=True, seed=281)
        prob = make_prob(problem["rows"], seed=282)
        expected_c, expected_d = run_srelu(problem, prob, clamp_scale=CLAMP_SCALE)
        c, d = bf16_rows(problem["rows"], 256, fill=float("nan")), bf16_rows(problem["rows"], 256, fill=float("nan"))
    if call == "wrapper":
        c, d = run_srelu(problem, prob, clamp_scale=CLAMP_SCALE)
    else:
        api = srelu_api(problem, prob, c, d, tanh_clamp_scale=CLAMP_SCALE)
        assert api.check_support()
        api.compile()
        workspace = ws(api, device=f"cuda:{other}")
        srelu_execute(api, problem, prob, c, d, workspace=workspace)
    assert torch.cuda.current_device() == caller, "the call did not restore the caller's device"
    torch.cuda.synchronize(other)
    routed = problem["routed"]
    assert c.device.index == other and d.device.index == other
    assert_bitwise_equal("C launched from another current device", c[:routed], expected_c[:routed])
    assert_bitwise_equal("D launched from another current device", d[:routed], expected_d[:routed])


BF16_DSL_GATE_CHILD = r"""
import cudnn.gemm.cutedsl.discrete_grouped  # noqa: F401
import cudnn.gemm.cutedsl.grouped as grouped
from cudnn.frost import buffers

BF16_KERNEL_MODULES = (
    "cudnn.frost.tile_dsl.barrier",
    "cudnn.frost.tile_dsl.consume",
    "cudnn.gemm.cutedsl.grouped.srelu.moe_grouped_gemm_srelu",
    "cudnn.gemm.cutedsl.grouped.srelu._bf16_api",
)


def loaded():
    return [name for name in BF16_KERNEL_MODULES if name in sys.modules]


assert not loaded(), f"importing the grouped GEMM families imported {loaded()}"

# A CuTe DSL that clears the tile_dsl floor (4.7.0) but not the BF16 kernels' DSL_FLOOR (4.8.0): the gate reads
# only this cached package metadata.
buffers._DSL_STATE = (True, ("nvidia-cutlass-dsl", "4.7.1"))


def mn1(t):
    rows, cols = t.shape
    return t.as_strided((rows, cols, 1), (cols, 1, rows * cols))


def expect_version_error(call):
    try:
        call()
    except ImportError as exc:
        assert re.search(r"requires nvidia-cutlass-dsl >= 4\.8\.0; found 4\.7\.1", str(exc)), exc
    else:
        raise AssertionError("no ImportError naming the installed CuTe DSL version")


rows, n, k, experts = 512, 64, 64, 2
a = mn1(torch.randn(rows, k, device="cuda").to(torch.bfloat16))
z = mn1(torch.randn(rows, n, device="cuda").to(torch.bfloat16))
weights = torch.randn(experts, n, k, device="cuda").to(torch.bfloat16)
b_ptrs = torch.tensor([weights[e].data_ptr() for e in range(experts)], dtype=torch.int64, device="cuda")
prob = torch.rand(rows, 1, 1, device="cuda")
offsets = torch.tensor([256, 512], dtype=torch.int32, device="cuda")
discrete = dict(num_experts=experts, b_shape=(n, k), b_dtype=torch.bfloat16)
srelu = grouped.GroupedGemmSreluSm100(sample_a=a, sample_c=mn1(torch.empty_like(z[:, :, 0])), sample_d=mn1(torch.empty_like(z[:, :, 0])), sample_padded_offsets=offsets, sample_prob=prob, **discrete)
expect_version_error(srelu.check_support)
expect_version_error(lambda: grouped.grouped_gemm_srelu_wrapper_sm100(a_tensor=a, b_ptrs=b_ptrs, n=n, b_dtype=torch.bfloat16, padded_offsets=offsets, prob_tensor=prob))
assert not loaded(), f"the declined BF16 calls imported {loaded()}"
print("BF16_DSL_GATE_OK")
"""


@pytest.mark.L0
def test_srelu_bf16_kernel_is_imported_only_behind_the_dsl_gate(tmp_path):
    """python/cudnn/AGENTS.md Rule 7: the BF16 kernel modules import cutlass.experimental (cudnn.frost.tile_dsl) at
    module scope, which the block-scaled and JAX paths never need. In a fresh process, importing the grouped and
    discrete-grouped GEMM families imports neither them nor the tile_dsl primitives; with the installed CuTe DSL
    reported as 4.7.1, which clears the tile_dsl floor (4.7.0) but not the kernels' (4.8.0), the BF16 check_support()
    of both facades and both BF16 wrappers raise an ImportError naming 4.8.0 and 4.7.1, still without importing them."""
    run_python_child(BF16_DSL_GATE_CHILD, env={}, cwd=tmp_path, marker="BF16_DSL_GATE_OK")


SRELU_RENDER_CHILD = r"""
from cudnn import GroupedGemmSreluSm100

rows, n, k, experts = 512, 256, 128, 2
for weights in ("discrete", "dense"):
    kwargs = dict(
        sample_a=mn1(torch.randn(rows, k, device="cuda").to(torch.bfloat16)),
        sample_c=mn1(torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")),
        sample_d=mn1(torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")),
        sample_padded_offsets=torch.tensor([256, 512], dtype=torch.int32, device="cuda"),
        sample_prob=torch.rand(rows, 1, 1, device="cuda"),
        tanh_clamp_scale=16.0,
        tanh_mode="approx",
    )
    if weights == "discrete":
        kwargs.update(num_experts=experts, b_shape=(n, k), b_dtype=torch.bfloat16, b_major="k")
    else:
        kwargs.update(sample_b=torch.randn(experts, n, k, device="cuda").to(torch.bfloat16).permute(1, 2, 0))
    api = GroupedGemmSreluSm100(**kwargs)
    assert api.check_support()
    api.compile()
    save_ptx(api, weights)
print("SRELU_RENDER_OK")
"""


@pytest.mark.L0
def test_srelu_bf16_rendering_orders_every_release(tmp_path):
    """python/cudnn/AGENTS.md Rule 11, the proof, in the forward kernel's PTX for discrete and dense weights: the six
    tile-info releases (TMA, MMA and epilogue warps, two each) arrive at addresses computed from the four loaded words
    through the release-zero argument; every warp-subset named barrier is the unaligned form; only discrete weights
    acquire the per-expert tensormaps. A fresh process compiles both kernels with the DSL keeping their PTX."""
    env, ptx_dir = render_env(tmp_path)
    run_python_child(RENDER_CHILD_PRELUDE + SRELU_RENDER_CHILD, env=env, cwd=tmp_path, marker="SRELU_RENDER_OK")
    for weights in ("discrete", "dense"):
        ptx = (ptx_dir / f"{weights}.ptx").read_text()
        assert_release_rendering(f"srelu {weights}", ptx, sites=6, wide_words=None, acquire=weights == "discrete")


@pytest.mark.L0
@pytest.mark.parametrize("capability", [(10, 7), (9, 0), (12, 0)], ids=["sm107", "sm90", "sm120"])
def test_srelu_bf16_rejects_other_architectures(monkeypatch, capability):
    import cudnn.gemm.cutedsl.grouped.srelu._bf16_api as bf16_api
    from cudnn import GroupedGemmSreluSm100

    monkeypatch.setattr(bf16_api, "get_compute_capability", lambda *args, **kwargs: capability)
    problem = make_problem([256], 64, 64, seed=131)
    prob = make_prob(256, seed=132)
    c = mn1(torch.empty((256, 64), dtype=torch.bfloat16, device="cuda"))
    api = GroupedGemmSreluSm100(
        sample_a=problem["a"], sample_b=problem["b"], sample_c=c, sample_d=c, sample_padded_offsets=problem["offsets"], sample_prob=prob
    )
    with pytest.raises(RuntimeError, match="SM100, SM103 only"):
        api.check_support()


@pytest.mark.L0
def test_block_scaled_srelu_activation_options():
    """The block-scaled path computes tanh with MUFU tanh.approx from the FP32 accumulator: with a clamp
    scale it refuses "accurate", it refuses act_input="saved", and it validates tanh_mode as the BF16
    backend does (None and unknown strings raise). Plain mode accepts either mode, so the default
    ("approx") too; clamped mode accepts the default. The wrapper refuses before it reads the FP8
    operands, the facade in check_support() before it builds the block-scaled implementation."""
    from cudnn import GroupedGemmSreluSm100, grouped_gemm_srelu_wrapper_sm100
    from cudnn.gemm.cutedsl.grouped.srelu.api import _check_block_scaled_activation_options

    a = torch.zeros((256, 128, 1), dtype=torch.float8_e4m3fn, device="cuda")
    b = torch.zeros((128, 128, 1), dtype=torch.float8_e4m3fn, device="cuda")
    rejected = [
        (
            dict(tanh_clamp_scale=CLAMP_SCALE, tanh_mode="accurate"),
            re.escape("grouped_gemm_srelu_sm100: tanh_mode='accurate' with tanh_clamp_scale is supported only by the BF16 kernel"),
        ),
        (dict(act_input="saved"), re.escape("grouped_gemm_srelu_sm100: act_input='saved' is supported only by the BF16 kernel")),
        (dict(tanh_clamp_scale=CLAMP_SCALE, tanh_mode=None), re.escape(f"{TANH_MODE_ERROR}, got None")),
        (dict(tanh_mode=None), re.escape(f"{TANH_MODE_ERROR}, got None")),
        (dict(tanh_mode="fast"), re.escape(f"{TANH_MODE_ERROR}, got 'fast'")),
    ]
    for kwargs, match in rejected:
        with pytest.raises(ValueError, match=match):
            grouped_gemm_srelu_wrapper_sm100(a_tensor=a, b_tensor=b, **kwargs)
        with pytest.raises(ValueError, match=match):
            GroupedGemmSreluSm100(sample_a=a, sample_b=b, **kwargs).check_support()
    for clamp_scale, tanh_mode in ((None, "approx"), (None, "accurate"), (CLAMP_SCALE, "approx")):
        _check_block_scaled_activation_options("grouped_gemm_srelu_sm100", clamp_scale, tanh_mode, "acc")


@pytest.mark.L0
def test_srelu_bf16_rejects_jax_arrays():
    """The BF16 backend is torch-only: BF16 JAX arrays are refused by the wrapper (before any allocation)
    and by the facade's constructor."""
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from cudnn import GroupedGemmSreluSm100, grouped_gemm_srelu_wrapper_sm100

    a = jnp.zeros((256, 64, 1), dtype=jnp.bfloat16)
    b = jnp.zeros((64, 64, 1), dtype=jnp.bfloat16)
    offsets = jnp.asarray([256], dtype=jnp.int32)
    prob = jnp.ones((256, 1, 1), dtype=jnp.float32)
    match = "The BF16 grouped GEMM SReLU backend supports torch tensors only"
    with pytest.raises(ValueError, match=match):
        grouped_gemm_srelu_wrapper_sm100(a_tensor=a, b_tensor=b, padded_offsets=offsets, prob_tensor=prob)
    with pytest.raises(ValueError, match=match):
        GroupedGemmSreluSm100(sample_a=a, sample_b=b, sample_padded_offsets=offsets, sample_prob=prob)


@pytest.mark.L0
@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
def test_srelu_bf16_rejects_per_thread_stream(discrete, monkeypatch):
    """Recipe R1: CUstream(2) (cudaStreamPerThread) has no safe torch stream. The wrapper refuses it
    where it allocates the outputs; the discrete class API where it records the pointer table's stream,
    after the workspace checks and before the launch."""
    from cuda.bindings import driver as cuda

    problem = make_problem([256, 256], 64, 64, discrete=discrete, seed=281)
    prob = make_prob(problem["rows"], seed=282)
    with pytest.raises(ValueError, match="cudaStreamPerThread"):
        run_srelu(problem, prob, clamp_scale=None, current_stream=cuda.CUstream(2))
    if discrete:
        c, d = bf16_rows(problem["rows"], 64), bf16_rows(problem["rows"], 64)
        api, launches = intercepted_srelu_api(problem, prob, c, d, monkeypatch)
        with pytest.raises(ValueError, match="cudaStreamPerThread"):
            srelu_execute(api, problem, prob, c, d, workspace=ws(api), current_stream=cuda.CUstream(2))
        assert not launches
