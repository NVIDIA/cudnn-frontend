# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BF16 coverage for the grouped GEMM + weighted squared-ReLU backward (grouped_gemm_dsrelu_wrapper_sm100 and
GroupedGemmDsreluSm100).

Numerics against an FP32 reference and the native path (test_grouped_gemm_srelu_bf16_utils.py), the class API's
caller-owned scratch (recipe R2), the Rule 8 / R9 detectors (no host synchronization, no allocation -- the
deterministic dprob fold included --, a first execute inside CUDA-graph capture), the wrapper's plan cache, per-launch
rebinding of the discrete weight descriptors, the lazy facade and the support checks. Bitwise repeatability, batch
invariance and the Rule 11 warm stress of the kernel are in test_grouped_gemm_dsrelu_bf16_determinism.py.
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
    assert_dprob_within_bound,
    assert_near_native,
    assert_not_worse_than_native,
    assert_release_rendering,
    assert_values_equal,
    b_kwargs,
    bf16_rows,
    count_wrapper_compiles,
    device_values_validated,
    dprob_slots,
    dprob_sum_bound,
    dsrelu_api,
    dsrelu_execute,
    eager_backward,
    eager_forward,
    exact_backward_reference,
    exact_prob,
    exact_z,
    make_problem,
    make_prob,
    make_z,
    mn1,
    native_backward,
    offsets_from_counts,
    other_bf16_device,
    pointer_values,
    reference_backward,
    render_env,
    require_bf16_srelu_gpu,
    require_no_host_value_reads,
    run_python_child,
    sm_count,
    store_weights,
    sync_debug_error,
    ternary_weights,
)
from test_utils import assert_bitwise_runs

ACTIVATIONS = [
    pytest.param(None, "approx", id="plain"),
    pytest.param(CLAMP_SCALE, "accurate", id="clamped-accurate"),
    pytest.param(CLAMP_SCALE, "approx", id="clamped-approx"),
]
TANH_MODE_ERROR = "tanh_mode must be 'approx' or 'accurate'"


@pytest.fixture(autouse=True)
def require_sm100():
    require_bf16_srelu_gpu()


def run_dsrelu(problem, z_save, prob, *, clamp_scale, tanh_mode=PUBLIC_DEFAULT, deterministic=False, generate_d_srelu=True, **kwargs):
    """``problem["a"]`` is G (the FC2 output gradient), ``problem`` B the FC2 weight as (N, D).
    ``tanh_mode`` left at PUBLIC_DEFAULT is not passed at all."""
    from cudnn import grouped_gemm_dsrelu_wrapper_sm100

    if tanh_mode is not PUBLIC_DEFAULT:
        kwargs["tanh_mode"] = tanh_mode
    out = grouped_gemm_dsrelu_wrapper_sm100(
        a_tensor=problem["a"],
        c_tensor=z_save,
        padded_offsets=problem["offsets"],
        prob_tensor=prob,
        tanh_clamp_scale=clamp_scale,
        deterministic=deterministic,
        generate_d_srelu=generate_d_srelu,
        **b_kwargs(problem),
        **kwargs,
    )
    for name in ("d_col_tensor", "dbias_tensor", "amax_tensor", "sfd_row_tensor", "sfd_col_tensor", "sfd_col_d_srelu_tensor"):
        assert out[name] is None, name
    assert (out["d_srelu_tensor"] is not None) == generate_d_srelu
    return out


def intercepted_dsrelu_api(problem, z_save, prob, d_row, dprob, monkeypatch, **config):
    """A checked (not compiled) facade whose compiled consumer and dprob fold are recorders: execute() runs every
    check and the workspace carve, and a launch only appends ("kernel" | "fold", arguments) to the returned list."""
    import cudnn.gemm.cutedsl.grouped.dsrelu._bf16_api as dsrelu_bf16_api

    api = dsrelu_api(problem, z_save, prob, d_row, dprob, **config)
    assert api.check_support()
    launches = []
    monkeypatch.setattr(api._implementation, "_compiled_kernel", lambda *args: launches.append(("kernel", args)))
    monkeypatch.setattr(dsrelu_bf16_api, "fold_dprob_slots", lambda *args: launches.append(("fold", args)))
    return api, launches


# --------------------------------------------------------------------------- numerics

IDENTITY_CELLS = [
    pytest.param(
        clamp_scale,
        tanh_mode,
        width,
        discrete,
        deterministic,
        id=f"{label}-{width}-{'discrete' if discrete else 'dense'}-{'deterministic' if deterministic else 'atomic'}",
    )
    for clamp_scale, tanh_mode, label in ((None, "approx", "plain"), (CLAMP_SCALE, "accurate", "clamped-accurate"))
    for width in (256, 1856)
    for discrete in (False, True)
    for deterministic in (True, False)
] + [
    pytest.param(CLAMP_SCALE, "approx", 1856, discrete, True, id=f"clamped-approx-1856-{'discrete' if discrete else 'dense'}-deterministic")
    for discrete in (False, True)
]


@pytest.mark.L0
@pytest.mark.parametrize("clamp_scale, tanh_mode, width, discrete, deterministic", IDENTITY_CELLS)
def test_dsrelu_bf16_identity_weights_match_native(clamp_scale, tanh_mode, width, discrete, deterministic):
    """With B_e = I the FP32 accumulator is the BF16 upstream gradient exactly.

    The plain squared-ReLU dZ and the recomputed forward d_srelu use the native rounding points and
    operation order, so they equal the eager backward bit for bit. With the explicit
    tanh_mode="accurate" the clamped ones may differ by one BF16 ulp on a small fraction of the
    elements (FP32 Cephes tanh instead of libdevice tanhf, and dZ multiplies in the order
    (dH * p) * 2c * (1 - t^2)). MUFU tanh.approx (the default) is held to the error criterion: dZ and
    d_srelu not less accurate than native against the FP32 reference, with the measured ratio reported.
    dprob is within the FP32 summation bound of the exact sum in either reduction mode, with the
    approx per-term allowance for tanh.approx.
    """
    counts = [256, 0, 512]
    problem = make_problem(counts, width, width, discrete=discrete, b_major="n", identity=True, seed=7)
    z_save = make_z(problem["rows"], width, seed=8)
    prob = make_prob(problem["rows"], seed=9)
    out = run_dsrelu(problem, z_save, prob, clamp_scale=clamp_scale, tanh_mode=tanh_mode, deterministic=deterministic)
    routed = problem["routed"]
    g, z, p = problem["a"][:routed, :, 0], z_save[:routed, :, 0], prob[:routed, :, 0]
    dz_native, _ = eager_backward(g, z, p, clamp_scale)
    h_native = eager_forward(z, p, clamp_scale)
    reference = reference_backward(problem, z_save, prob, clamp_scale)
    report = approx_tanh(clamp_scale, tanh_mode)
    if clamp_scale is None:
        assert_bitwise_equal("dZ (plain)", out["d_row_tensor"][:routed, :, 0], dz_native)
        assert_bitwise_equal("d_srelu (plain)", out["d_srelu_tensor"][:routed, :, 0], h_native)
    elif tanh_mode == "accurate":
        assert_near_native("dZ (clamped accurate)", out["d_row_tensor"][:routed, :, 0], dz_native)
        assert_near_native("d_srelu (clamped accurate)", out["d_srelu_tensor"][:routed, :, 0], h_native)
    else:
        native = native_backward(problem, z_save, prob, clamp_scale)
        assert_not_worse_than_native("dZ", out["d_row_tensor"], reference["dz"], native["dz"], routed=routed, report_ratio=report)
        assert_not_worse_than_native("d_srelu", out["d_srelu_tensor"], reference["h"], native["h"], routed=routed, report_ratio=report)
    bound = dprob_sum_bound(z_save, reference["dh"], clamp_scale, "approx" if report else "accurate")
    assert_dprob_within_bound("dprob", out["dprob_tensor"], reference["dprob"], bound, routed=routed)


@pytest.mark.L0
@pytest.mark.parametrize("deterministic", [True, False], ids=["deterministic", "atomic"])
@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
@pytest.mark.parametrize("clamp_scale, tanh_mode", ACTIVATIONS)
def test_dsrelu_bf16_against_reference(clamp_scale, tanh_mode, discrete, deterministic):
    """Random FC2 weights, uneven counts with an empty expert and rows past the routed total.

    dZ, d_srelu and dprob must not be less accurate than the native path (dH rounded to BF16, then
    the eager backward); MUFU tanh.approx too, with its measured ratio reported. dprob of the unrouted rows
    stays zero.
    """
    counts = [512, 0, 256, 768]
    problem = make_problem(counts, 512, 384, extra_rows=256, discrete=discrete, b_major="n", seed=11)
    z_save = make_z(problem["rows"], 512, seed=12)
    prob = make_prob(problem["rows"], seed=13)
    out = run_dsrelu(problem, z_save, prob, clamp_scale=clamp_scale, tanh_mode=tanh_mode, deterministic=deterministic)
    reference = reference_backward(problem, z_save, prob, clamp_scale)
    native = native_backward(problem, z_save, prob, clamp_scale)
    routed = problem["routed"]
    report = approx_tanh(clamp_scale, tanh_mode)
    assert_not_worse_than_native("dZ", out["d_row_tensor"], reference["dz"], native["dz"], routed=routed, report_ratio=report)
    assert_not_worse_than_native("d_srelu", out["d_srelu_tensor"], reference["h"], native["h"], routed=routed, report_ratio=report)
    assert_not_worse_than_native("dprob", out["dprob_tensor"], reference["dprob"], native["dprob"], routed=routed, report_ratio=report)
    assert not out["dprob_tensor"][routed:].any(), "dprob of rows past the routed total must stay 0"


@pytest.mark.L0
@pytest.mark.parametrize("clamp_scale, tanh_mode", ACTIVATIONS)
def test_dsrelu_bf16_atomic_and_deterministic_agree(clamp_scale, tanh_mode):
    """The dprob reduction mode changes only dprob, and only by FP32 summation order."""
    problem = make_problem([256, 0, 768], 1856, 256, discrete=True, b_major="n", seed=21)
    z_save = make_z(problem["rows"], 1856, seed=22)
    prob = make_prob(problem["rows"], seed=23)
    det = run_dsrelu(problem, z_save, prob, clamp_scale=clamp_scale, tanh_mode=tanh_mode, deterministic=True)
    atomic = run_dsrelu(problem, z_save, prob, clamp_scale=clamp_scale, tanh_mode=tanh_mode, deterministic=False)
    routed = problem["routed"]
    assert_bitwise_equal("dZ", atomic["d_row_tensor"][:routed], det["d_row_tensor"][:routed])
    assert_bitwise_equal("d_srelu", atomic["d_srelu_tensor"][:routed], det["d_srelu_tensor"][:routed])
    dh = reference_backward(problem, z_save, prob, clamp_scale)["dh"]
    bound = dprob_sum_bound(z_save, dh, clamp_scale, "accurate")
    assert_dprob_within_bound("dprob atomic vs deterministic", atomic["dprob_tensor"], det["dprob_tensor"], bound, routed=routed)


@pytest.mark.L0
@pytest.mark.parametrize("clamp_scale, tanh_mode", [(None, "approx"), (CLAMP_SCALE, "accurate")], ids=["plain", "clamped-accurate"])
def test_dsrelu_bf16_deterministic_is_bitwise_reproducible(clamp_scale, tanh_mode):
    problem = make_problem([768, 0, 512, 256], 1856, 512, discrete=True, b_major="n", seed=31)
    z_save = make_z(problem["rows"], 1856, seed=32)
    prob = make_prob(problem["rows"], seed=33)

    def launch():
        out = run_dsrelu(problem, z_save, prob, clamp_scale=clamp_scale, tanh_mode=tanh_mode, deterministic=True)
        # 1-D views: assert_bitwise_runs views each output as bytes, which needs a unit last stride.
        return tuple(out[key].reshape(-1) for key in ("d_row_tensor", "dprob_tensor", "d_srelu_tensor"))

    assert_bitwise_runs(launch, label=f"dsrelu bf16 deterministic s={clamp_scale} tanh_mode={tanh_mode}")


@pytest.mark.L0
@pytest.mark.parametrize("deterministic", [True, False], ids=["deterministic", "atomic"])
def test_dsrelu_bf16_generate_d_srelu_false(deterministic):
    problem = make_problem([256, 512], 256, 192, discrete=True, b_major="n", seed=41)
    z_save = make_z(problem["rows"], 256, seed=42)
    prob = make_prob(problem["rows"], seed=43)
    with_h = run_dsrelu(problem, z_save, prob, clamp_scale=CLAMP_SCALE, deterministic=True)
    without_h = run_dsrelu(problem, z_save, prob, clamp_scale=CLAMP_SCALE, deterministic=deterministic, generate_d_srelu=False)
    assert_bitwise_equal("dZ", without_h["d_row_tensor"], with_h["d_row_tensor"])
    if deterministic:
        assert_bitwise_equal("dprob", without_h["dprob_tensor"], with_h["dprob_tensor"])
    else:
        reference = reference_backward(problem, z_save, prob, CLAMP_SCALE)
        bound = dprob_sum_bound(z_save, reference["dh"], CLAMP_SCALE, "approx")
        assert_dprob_within_bound("atomic dprob without H", without_h["dprob_tensor"], reference["dprob"], bound, routed=problem["routed"])


@pytest.mark.L0
@pytest.mark.parametrize("clamp_scale, tanh_mode", ACTIVATIONS)
@pytest.mark.parametrize("deterministic", [False, True], ids=["atomic", "deterministic"])
def test_dsrelu_bf16_nan_propagates(clamp_scale, tanh_mode, deterministic):
    """A NaN saved activation reaches dZ, recomputed H and dprob in just its own row."""
    problem = make_problem([256, 0, 256], 64, 64, seed=44, identity=True)
    z_save = make_z(problem["rows"], 64, seed=45)
    prob = make_prob(problem["rows"], seed=46)
    z_save[3, 5, 0] = float("nan")
    out = run_dsrelu(problem, z_save, prob, clamp_scale=clamp_scale, tanh_mode=tanh_mode, deterministic=deterministic)
    for name in ("d_row_tensor", "d_srelu_tensor"):
        expected_nan = torch.zeros_like(out[name], dtype=torch.bool)
        expected_nan[3, 5, 0] = True
        assert torch.equal(torch.isnan(out[name]), expected_nan), name
        assert bool(torch.isfinite(out[name][~expected_nan]).all()), name
    expected_nan = torch.zeros_like(out["dprob_tensor"], dtype=torch.bool)
    expected_nan[3] = True
    assert torch.equal(torch.isnan(out["dprob_tensor"]), expected_nan)
    assert bool(torch.isfinite(out["dprob_tensor"][~expected_nan]).all())


@pytest.mark.L0
def test_dsrelu_bf16_default_determinism_follows_torch(monkeypatch):
    """Changing the global setting selects the matching plan; explicit False still opts out."""
    import cudnn.gemm.cutedsl.grouped.dsrelu.api as module

    problem = make_problem([256, 0, 256], 320, 128, discrete=True, seed=47, exact=True)
    z_save = exact_z(problem["rows"], problem["n"], seed=48)
    prob = exact_prob(problem["rows"], seed=49)
    compiles = count_wrapper_compiles(monkeypatch, module, "GroupedGemmDsreluSm100", "_cache_of_GroupedGemmDsreluSm100Objects")
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        for enabled in (False, True):
            torch.use_deterministic_algorithms(enabled)
            implicit = run_dsrelu(problem, z_save, prob, clamp_scale=None, deterministic=None)
            explicit = run_dsrelu(problem, z_save, prob, clamp_scale=None, deterministic=enabled)
            for name in ("d_row_tensor", "d_srelu_tensor", "dprob_tensor"):
                assert_bitwise_equal(name, implicit[name], explicit[name])
        assert len(compiles) == 2
        assert [plan._implementation.deterministic for plan in compiles] == [False, True]
        run_dsrelu(problem, z_save, prob, clamp_scale=None, deterministic=False)
        assert len(compiles) == 2, "explicit False did not reuse the atomic plan"
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn)


@pytest.mark.L0
@pytest.mark.parametrize("clamp_scale, tanh_mode", ACTIVATIONS)
def test_srelu_dsrelu_bf16_recomputed_activation_matches_forward(clamp_scale, tanh_mode):
    """With act_input="saved" the forward evaluates the activation at the C it returns, so the
    backward's d_srelu (the recomputed FC2 input) equals the forward D bit for bit, when the
    backward's tanh_mode matches the forward's."""
    from cudnn import grouped_gemm_srelu_wrapper_sm100

    counts = [512, 0, 256]
    n, k, d = 448, 256, 320
    fc1 = make_problem(counts, n, k, extra_rows=256, discrete=True, b_major="k", seed=51, a_scale=8.0)
    prob = make_prob(fc1["rows"], seed=52)
    forward = grouped_gemm_srelu_wrapper_sm100(
        a_tensor=fc1["a"],
        padded_offsets=fc1["offsets"],
        prob_tensor=prob,
        tanh_clamp_scale=clamp_scale,
        tanh_mode=tanh_mode,
        act_input="saved",
        **b_kwargs(fc1),
    )
    fc2 = make_problem(counts, n, d, extra_rows=256, discrete=True, b_major="n", seed=53)
    backward = run_dsrelu(fc2, forward["c_tensor"], prob, clamp_scale=clamp_scale, tanh_mode=tanh_mode, deterministic=True)
    routed = fc1["routed"]
    assert_bitwise_equal("d_srelu vs forward D", backward["d_srelu_tensor"][:routed], forward["d_tensor"][:routed])


@pytest.mark.L0
@pytest.mark.parametrize("discrete, b_major", [(False, "k"), (False, "n"), (True, "k"), (True, "n")], ids=["dense-k", "dense-n", "discrete-k", "discrete-n"])
def test_dsrelu_bf16_weight_layouts(discrete, b_major):
    """Every weight layout at the explicit accurate tanh: not less accurate than native, no allowance."""
    problem = make_problem([256, 512], 320, 192, discrete=discrete, b_major=b_major, seed=71)
    z_save = make_z(problem["rows"], 320, seed=72)
    prob = make_prob(problem["rows"], seed=73)
    out = run_dsrelu(problem, z_save, prob, clamp_scale=CLAMP_SCALE, tanh_mode="accurate", deterministic=True)
    reference = reference_backward(problem, z_save, prob, CLAMP_SCALE)
    native = native_backward(problem, z_save, prob, CLAMP_SCALE)
    routed = problem["routed"]
    for name, key, ref_key in (("dZ", "d_row_tensor", "dz"), ("d_srelu", "d_srelu_tensor", "h"), ("dprob", "dprob_tensor", "dprob")):
        assert_not_worse_than_native(name, out[key], reference[ref_key], native[ref_key], routed=routed)


@pytest.mark.L0
def test_dsrelu_bf16_dynamic_scheduler_matches_static():
    problem = make_problem([256, 0, 768, 512], 384, 256, discrete=True, b_major="n", seed=81)
    z_save = make_z(problem["rows"], 384, seed=82)
    prob = make_prob(problem["rows"], seed=83)
    static = run_dsrelu(problem, z_save, prob, clamp_scale=CLAMP_SCALE, deterministic=True)
    dynamic = run_dsrelu(problem, z_save, prob, clamp_scale=CLAMP_SCALE, deterministic=True, use_dynamic_sched=True)
    routed = problem["routed"]
    for key in ("d_row_tensor", "d_srelu_tensor", "dprob_tensor"):
        assert_bitwise_equal(f"{key} dynamic vs static", dynamic[key][:routed], static[key][:routed])


@pytest.mark.L0
@pytest.mark.parametrize("mma_tiler_mn, cluster_shape_mn", [((128, 256), (1, 1)), ((256, 128), (2, 1))], ids=["1cta-256n", "2cta-128n"])
def test_dsrelu_bf16_tile_configs(mma_tiler_mn, cluster_shape_mn):
    """Both MMA widths at the explicit accurate tanh: not less accurate than native, no allowance."""
    problem = make_problem([256, 512], 320, 192, discrete=True, b_major="n", seed=91)
    z_save = make_z(problem["rows"], 320, seed=92)
    prob = make_prob(problem["rows"], seed=93)
    out = run_dsrelu(
        problem,
        z_save,
        prob,
        clamp_scale=CLAMP_SCALE,
        tanh_mode="accurate",
        deterministic=True,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
    )
    reference = reference_backward(problem, z_save, prob, CLAMP_SCALE)
    native = native_backward(problem, z_save, prob, CLAMP_SCALE)
    for name, key, ref_key in (("dZ", "d_row_tensor", "dz"), ("dprob", "dprob_tensor", "dprob")):
        assert_not_worse_than_native(name, out[key], reference[ref_key], native[ref_key], routed=problem["routed"])


@pytest.mark.L1
@pytest.mark.parametrize("deterministic", [True, False], ids=["deterministic", "atomic"])
@pytest.mark.parametrize("clamp_scale, tanh_mode", [(None, "approx"), (CLAMP_SCALE, "accurate")], ids=["plain", "clamped-accurate"])
@pytest.mark.parametrize("shape", [NEMOTRON3_SHAPE, SMALL_K_SHAPE], ids=["nemotron3", "e8-k512-n3584"])
def test_dsrelu_bf16_model_shapes(shape, clamp_scale, tanh_mode, deterministic):
    experts, d, n = shape["experts"], shape["k"], shape["n"]
    counts = [256 * ((3 * e) % 5) for e in range(experts)]  # uneven, with empty experts
    problem = make_problem(counts, n, d, discrete=True, b_major="n", seed=121)
    z_save = make_z(problem["rows"], n, seed=122)
    prob = make_prob(problem["rows"], seed=123, low=0.0, high=1.0)
    out = run_dsrelu(problem, z_save, prob, clamp_scale=clamp_scale, tanh_mode=tanh_mode, deterministic=deterministic)
    reference = reference_backward(problem, z_save, prob, clamp_scale)
    native = native_backward(problem, z_save, prob, clamp_scale)
    for name, key, ref_key in (("dZ", "d_row_tensor", "dz"), ("d_srelu", "d_srelu_tensor", "h"), ("dprob", "dprob_tensor", "dprob")):
        assert_not_worse_than_native(name, out[key], reference[ref_key], native[ref_key], routed=problem["routed"])


# --------------------------------------------------------------------------- class API and CUDA graphs


@pytest.mark.L0
def test_dsrelu_bf16_class_api_deterministic_finishes_reduction():
    """Through the class API (caller workspace of scratch_workspace_bytes() bytes), deterministic
    execute() adds the folded slots to dprob and leaves the slot buffer zeroed, so it can be reused;
    it matches the wrapper (same tanh mode). The block-scaled class sequence -- execute(), then
    reduce_dprob_workspace(slots, dprob), then slots.zero_() -- gives the same dprob bits here,
    while the non-portable return-value form reduce_dprob_workspace(slots) returns zeros."""
    from cudnn import GroupedGemmDsreluSm100

    problem = make_problem([256, 0, 512], 512, 256, extra_rows=256, b_major="n", seed=61)
    rows, n = problem["rows"], problem["n"]
    z_save = make_z(rows, n, seed=62)
    prob = make_prob(rows, seed=63)
    d_row, d_srelu = bf16_rows(rows, n), bf16_rows(rows, n)
    dprob = torch.zeros((rows, 1, 1), dtype=torch.float32, device="cuda")
    slots = dprob_slots(rows, n)
    api = dsrelu_api(
        problem, z_save, prob, d_row, dprob, d_srelu=d_srelu, dprob_workspace=slots, deterministic=True, tanh_clamp_scale=CLAMP_SCALE, tanh_mode="approx"
    )
    assert api.check_support()
    api.compile()
    workspace = ws(api)

    def execute(dprob_out):
        dsrelu_execute(api, problem, z_save, prob, d_row, dprob_out, d_srelu=d_srelu, dprob_workspace=slots, workspace=workspace)

    execute(dprob)
    torch.cuda.synchronize()
    assert not slots.any(), "execute() must leave the dprob slots zeroed"
    wrapper = run_dsrelu(problem, z_save, prob, clamp_scale=CLAMP_SCALE, tanh_mode="approx", deterministic=True)
    routed = problem["routed"]
    assert_bitwise_equal("class dZ vs wrapper", d_row[:routed], wrapper["d_row_tensor"][:routed])
    assert_bitwise_equal("class d_srelu vs wrapper", d_srelu[:routed], wrapper["d_srelu_tensor"][:routed])
    assert_bitwise_equal("class dprob vs wrapper", dprob, wrapper["dprob_tensor"])
    assert not GroupedGemmDsreluSm100.reduce_dprob_workspace(slots).any(), "the documented return-value form no longer returns zeros"
    dprob_again = torch.zeros_like(dprob)
    execute(dprob_again)
    assert_bitwise_equal("dprob from the reused (re-zeroed) slot buffer", dprob_again, dprob)
    GroupedGemmDsreluSm100.reduce_dprob_workspace(slots, dprob_again)
    slots.zero_()
    assert_bitwise_equal("dprob after the portable reduce_dprob_workspace(slots, dprob); slots.zero_()", dprob_again, dprob)


@pytest.mark.L0
def test_dsrelu_bf16_cuda_graph_replay():
    require_no_host_value_reads()
    problem = make_problem([256, 0, 512], 1856, 256, discrete=True, b_major="n", seed=101)
    z_save = make_z(problem["rows"], 1856, seed=102)
    prob = make_prob(problem["rows"], seed=103)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            run_dsrelu(problem, z_save, prob, clamp_scale=CLAMP_SCALE, deterministic=True)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = run_dsrelu(problem, z_save, prob, clamp_scale=CLAMP_SCALE, deterministic=True)
        problem["a"].copy_(make_problem([256, 0, 512], 1856, 256, discrete=True, b_major="n", seed=104)["a"])
        z_save.copy_(make_z(problem["rows"], 1856, seed=105))
        prob.copy_(make_prob(problem["rows"], seed=106))
        graph.replay()
        eager = run_dsrelu(problem, z_save, prob, clamp_scale=CLAMP_SCALE, deterministic=True)
        torch.cuda.synchronize()
        routed = problem["routed"]
        for key in ("d_row_tensor", "d_srelu_tensor", "dprob_tensor"):
            assert_bitwise_equal(f"{key} replay vs eager", captured[key][:routed], eager[key][:routed])
    finally:
        graph.reset()


# --------------------------------------------------------------------------- Rule 8 / recipe R9 detectors


@pytest.mark.L0
@pytest.mark.parametrize("deterministic", [True, False], ids=["deterministic", "atomic"])
def test_dsrelu_bf16_wrapper_never_synchronizes(deterministic, monkeypatch):
    """Rule 8 / recipe R9: a cold wrapper call (backend selection, check_support, compile -- the dprob
    fold's too --, per-call outputs, slots and scratch, launch) and a warm call run under
    torch.cuda.set_sync_debug_mode("error"); the caller's mode is restored afterwards."""
    from cudnn.gemm.cutedsl.grouped.dsrelu import _dprob_fold as dprob_fold
    from cudnn.gemm.cutedsl.grouped.dsrelu import api as dsrelu_module

    require_no_host_value_reads()
    compiles = count_wrapper_compiles(monkeypatch, dsrelu_module, "GroupedGemmDsreluSm100", "_cache_of_GroupedGemmDsreluSm100Objects")
    # The fold is compiled once per device for the whole process: empty that cache too, so the cold call builds it.
    monkeypatch.setattr(dprob_fold, "_compiled", {})
    problem = make_problem([256, 512], 256, 128, discrete=True, b_major="n", seed=111)
    z_save = make_z(problem["rows"], 256, seed=112)
    prob = make_prob(problem["rows"], seed=113)
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    with sync_debug_error():
        cold = run_dsrelu(problem, z_save, prob, clamp_scale=CLAMP_SCALE, deterministic=deterministic)
        warm = run_dsrelu(problem, z_save, prob, clamp_scale=CLAMP_SCALE, deterministic=deterministic)
    assert torch.cuda.get_sync_debug_mode() == previous
    torch.cuda.synchronize()
    assert len(compiles) == 1, "the guarded first call must have been the cold build"
    # Atomic dprob adds in tile-arrival order, so only the deterministic dprob is compared bitwise.
    for key in ("d_row_tensor", "d_srelu_tensor") + (("dprob_tensor",) if deterministic else ()):
        assert_bitwise_equal(f"{key} cold vs warm", cold[key], warm[key])


@pytest.mark.L0
def test_dsrelu_bf16_class_api_allocates_nothing_and_never_synchronizes(monkeypatch):
    """Rule 8 / recipe R9 on the deterministic class API with caller-owned outputs, slots and workspace:
    construction, check_support(), compile() (which also compiles the dprob fold) and three executes
    (kernel plus fold) run under sync-debug "error", and the caching allocator records no allocation
    (neither bytes held nor allocations made) across compile() and the three executes."""
    from cudnn.gemm.cutedsl.grouped.dsrelu import _dprob_fold as dprob_fold

    require_no_host_value_reads()
    # The fold is compiled once per device for the whole process: empty that cache, so compile() builds it here.
    monkeypatch.setattr(dprob_fold, "_compiled", {})
    problem = make_problem([256, 0, 512], 256, 128, extra_rows=256, discrete=True, b_major="n", seed=141)
    rows, n = problem["rows"], problem["n"]
    z_save = make_z(rows, n, seed=142)
    prob = make_prob(rows, seed=143)
    d_row, d_srelu = bf16_rows(rows, n), bf16_rows(rows, n)
    dprob = torch.zeros((rows, 1, 1), dtype=torch.float32, device="cuda")
    slots = dprob_slots(rows, n)
    torch.cuda.synchronize()
    with sync_debug_error():
        api = dsrelu_api(
            problem, z_save, prob, d_row, dprob, d_srelu=d_srelu, dprob_workspace=slots, deterministic=True, tanh_clamp_scale=CLAMP_SCALE, tanh_mode="approx"
        )
        assert api.check_support()
        workspace = ws(api)
        before = allocator_state()
        api.compile()
        compiled = allocator_state()
        for _ in range(3):
            dsrelu_execute(api, problem, z_save, prob, d_row, dprob, d_srelu=d_srelu, dprob_workspace=slots, workspace=workspace)
        executed = allocator_state()
    torch.cuda.synchronize()
    assert compiled == before, f"compile() allocated: (bytes held, allocations made) {before} -> {compiled}"
    assert executed == before, f"three executes allocated: (bytes held, allocations made) {before} -> {executed}"
    assert not slots.any(), "execute() must leave the dprob slots zeroed"
    wrapper = run_dsrelu(problem, z_save, prob, clamp_scale=CLAMP_SCALE, tanh_mode="approx", deterministic=True)
    routed = problem["routed"]
    assert_bitwise_equal("class dZ vs wrapper", d_row[:routed], wrapper["d_row_tensor"][:routed])
    assert_bitwise_equal("class d_srelu vs wrapper", d_srelu[:routed], wrapper["d_srelu_tensor"][:routed])
    # dprob accumulated three launches: three times the deterministic left fold of the same slots.
    assert_bitwise_equal("class dprob vs 3 x wrapper", dprob, wrapper["dprob_tensor"] + wrapper["dprob_tensor"] + wrapper["dprob_tensor"])


@pytest.mark.L0
def test_dsrelu_bf16_first_execute_inside_capture_then_changed_replay():
    """Rule 8: the deterministic class API's first execute (kernel and dprob fold) happens inside CUDA-
    graph capture, with offsets and pointer tensors no execute has seen and a framework-neutral
    workspace view. It synchronizes nothing (sync-debug "error" scoped to the execute inside the
    capture) and allocates nothing. Replays then follow changed G, Z_save, prob, expert boundaries and
    weight pointers; each is compared exactly with an independent float64 reference on NaN-filled
    dZ / d_srelu, with dprob zeroed inside the graph."""
    from cudnn.frost.buffers import DeviceView

    require_no_host_value_reads()
    counts, n, k = [512, 0, 256], 256, 128
    problem = make_problem(counts, n, k, extra_rows=256, discrete=True, b_major="n", exact=True, seed=151)
    rows = problem["rows"]
    z_save = exact_z(rows, n, seed=152)
    prob = exact_prob(rows, seed=153)
    d_row, d_srelu = bf16_rows(rows, n, fill=float("nan")), bf16_rows(rows, n, fill=float("nan"))
    dprob = torch.zeros((rows, 1, 1), dtype=torch.float32, device="cuda")
    slots = dprob_slots(rows, n)
    api = dsrelu_api(problem, z_save, prob, d_row, dprob, d_srelu=d_srelu, dprob_workspace=slots, deterministic=True)
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
            dprob.zero_()
            with sync_debug_error():
                dsrelu_execute(api, problem, z_save, prob, d_row, dprob, d_srelu=d_srelu, dprob_workspace=slots, workspace=view)
            assert allocator_state() == before, "the first execute allocated inside the capture"
        for step, new_counts in enumerate(([512, 0, 256], [256, 256, 0], [0, 768, 256])):
            fresh = make_problem(new_counts, n, k, extra_rows=rows - sum(new_counts), b_major="n", exact=True, seed=160 + step)
            problem["a"].copy_(fresh["a"])
            z_save.copy_(exact_z(rows, n, seed=170 + step))
            prob.copy_(exact_prob(rows, seed=180 + step))
            problem["offsets"].copy_(offsets_from_counts(new_counts))
            storage = store_weights(fresh["b_logical"], "n")
            alive.append(storage)
            problem["b_ptrs"].copy_(pointer_values(storage))
            d_row.fill_(float("nan"))
            d_srelu.fill_(float("nan"))
            graph.replay()
            routed = sum(new_counts)
            reference = exact_backward_reference(problem["a"], fresh["b_logical"], new_counts, z_save, prob)
            assert_values_equal(f"replay {step} dZ", d_row[:routed, :, 0], reference["d_row"])
            assert_values_equal(f"replay {step} d_srelu", d_srelu[:routed, :, 0], reference["d_srelu"])
            assert_values_equal(f"replay {step} dprob", dprob[:routed, 0, 0], reference["dprob"])
            assert bool(torch.isnan(d_row[routed:]).all()) and bool(torch.isnan(d_srelu[routed:]).all()), f"replay {step} wrote rows past the routed total"
            assert not dprob[routed:].any(), f"replay {step}: dprob of rows past the routed total must stay 0"
        assert not slots.any(), "the replays must leave the dprob slots zeroed"
    finally:
        graph.reset()


# --------------------------------------------------------------------------- the wrapper's plan cache


@pytest.mark.L0
def test_dsrelu_bf16_plain_mode_shares_one_kernel_across_tanh_modes(monkeypatch):
    """The plain squared-ReLU evaluates no tanh: the default (omitted), "approx" and "accurate" are all
    accepted, hit one wrapper cache entry and one compile, and give the same bytes."""
    from cudnn import grouped_gemm_dsrelu_wrapper_sm100
    from cudnn.gemm.cutedsl.grouped.dsrelu import api as dsrelu_module

    assert inspect.signature(grouped_gemm_dsrelu_wrapper_sm100).parameters["tanh_mode"].default == "approx"
    assert inspect.signature(dsrelu_module.GroupedGemmDsreluSm100).parameters["tanh_mode"].default == "approx"
    compiles = count_wrapper_compiles(monkeypatch, dsrelu_module, "GroupedGemmDsreluSm100", "_cache_of_GroupedGemmDsreluSm100Objects")
    problem = make_problem([256, 0, 512], 256, 128, discrete=True, b_major="n", seed=171)
    z_save = make_z(problem["rows"], 256, seed=172)
    prob = make_prob(problem["rows"], seed=173)
    default = run_dsrelu(problem, z_save, prob, clamp_scale=None, deterministic=True)
    for mode in ("approx", "accurate"):
        out = run_dsrelu(problem, z_save, prob, clamp_scale=None, tanh_mode=mode, deterministic=True)
        for key in ("d_row_tensor", "d_srelu_tensor", "dprob_tensor"):
            assert_bitwise_equal(f"plain {key}, tanh_mode={mode} vs default", out[key], default[key])
    assert len(compiles) == 1, f"plain mode compiled {len(compiles)} kernels for three tanh_mode spellings"
    assert len(dsrelu_module._cache_of_GroupedGemmDsreluSm100Objects) == 1


@pytest.mark.L0
def test_dsrelu_bf16_accepts_use_dsrelu_reuse(monkeypatch):
    """The BF16 epilogue always computes relu(C)^2 (clamped: c^2) once for both d_srelu and dprob, which
    is what use_dsrelu_reuse=True asks for. The wrapper accepts True with one cache entry and one compile
    per activation shared with False, and the class API accepts it too; every output equals the False bits."""
    from cudnn.gemm.cutedsl.grouped.dsrelu import api as dsrelu_module

    compiles = count_wrapper_compiles(monkeypatch, dsrelu_module, "GroupedGemmDsreluSm100", "_cache_of_GroupedGemmDsreluSm100Objects")
    problem = make_problem([256, 0, 512], 256, 128, discrete=True, b_major="n", seed=311)
    rows, routed = problem["rows"], problem["routed"]
    z_save = make_z(rows, 256, seed=312)
    prob = make_prob(rows, seed=313)
    for clamp_scale in (None, CLAMP_SCALE):
        reuse_off = run_dsrelu(problem, z_save, prob, clamp_scale=clamp_scale, deterministic=True)
        reuse_on = run_dsrelu(problem, z_save, prob, clamp_scale=clamp_scale, deterministic=True, use_dsrelu_reuse=True)
        for key in ("d_row_tensor", "d_srelu_tensor", "dprob_tensor"):
            assert_bitwise_equal(f"{key}, clamp_scale={clamp_scale}: use_dsrelu_reuse=True vs False", reuse_on[key], reuse_off[key])
    assert len(compiles) == 2, f"{len(compiles)} compiles for two activations: use_dsrelu_reuse reached the plan"
    assert len(dsrelu_module._cache_of_GroupedGemmDsreluSm100Objects) == 2

    # The class API, clamped like the last wrapper call.
    d_row, d_srelu = bf16_rows(rows, 256), bf16_rows(rows, 256)
    dprob = torch.zeros((rows, 1, 1), dtype=torch.float32, device="cuda")
    slots = dprob_slots(rows, 256)
    api = dsrelu_api(
        problem, z_save, prob, d_row, dprob, d_srelu=d_srelu, dprob_workspace=slots, deterministic=True, tanh_clamp_scale=CLAMP_SCALE, use_dsrelu_reuse=True
    )
    assert api.check_support()
    api.compile()
    dsrelu_execute(api, problem, z_save, prob, d_row, dprob, d_srelu=d_srelu, dprob_workspace=slots, workspace=ws(api))
    assert_bitwise_equal("class dZ with use_dsrelu_reuse=True vs the wrapper's False", d_row[:routed], reuse_off["d_row_tensor"][:routed])
    assert_bitwise_equal("class d_srelu with use_dsrelu_reuse=True vs the wrapper's False", d_srelu[:routed], reuse_off["d_srelu_tensor"][:routed])
    assert_bitwise_equal("class dprob with use_dsrelu_reuse=True vs the wrapper's False", dprob, reuse_off["dprob_tensor"])


@pytest.mark.L0
def test_dsrelu_bf16_wrapper_reuses_one_plan_across_runtime_values(monkeypatch):
    """Cache-miss regression: M, the offset values and the weight pointers are runtime values of one
    plan. Four calls that change all three (empty experts included) compile once and keep one cache
    entry, and every call equals the exact float64 reference."""
    from cudnn.gemm.cutedsl.grouped.dsrelu import api as dsrelu_module

    compiles = count_wrapper_compiles(monkeypatch, dsrelu_module, "GroupedGemmDsreluSm100", "_cache_of_GroupedGemmDsreluSm100Objects")
    for step, counts in enumerate(([256, 0, 512], [768, 256, 0], [0, 0, 256], [512, 512, 512])):
        problem = make_problem(counts, 256, 128, discrete=True, b_major="n", exact=True, seed=180 + step)
        z_save = exact_z(problem["rows"], 256, seed=190 + step)
        prob = exact_prob(problem["rows"], seed=200 + step)
        out = run_dsrelu(problem, z_save, prob, clamp_scale=None, deterministic=False)
        reference = exact_backward_reference(problem["a"], problem["b_logical"], counts, z_save, prob)
        assert_values_equal(f"call {step} dZ", out["d_row_tensor"][:, :, 0], reference["d_row"])
        assert_values_equal(f"call {step} d_srelu", out["d_srelu_tensor"][:, :, 0], reference["d_srelu"])
        assert_values_equal(f"call {step} dprob", out["dprob_tensor"][:, 0, 0], reference["dprob"])
    assert len(compiles) == 1, f"{len(compiles)} compiles for one plan"
    assert len(dsrelu_module._cache_of_GroupedGemmDsreluSm100Objects) == 1


# --------------------------------------------------------------------------- discrete weight descriptors

DESCRIPTOR_TILINGS = [
    pytest.param((256, 256), (2, 1), False, id="2cta-static"),
    pytest.param((128, 128), (1, 1), False, id="1cta-static"),
    pytest.param((256, 256), (2, 1), True, id="2cta-dynamic"),
]


@pytest.mark.L0
@pytest.mark.parametrize("mma_tiler_mn, cluster_shape_mn, use_dynamic_sched", DESCRIPTOR_TILINGS)
def test_dsrelu_bf16_discrete_weight_descriptors_follow_new_pointers(mma_tiler_mn, cluster_shape_mn, use_dynamic_sched):
    """The per-expert B descriptors live in the caller's workspace and every launch's helper kernel
    rewrites them at the same address; the TMA descriptor cache outlives launches, so each CTA must
    acquire an expert's descriptor before loading through it (moe_utils.acquire_tma_desc).

    One compiled API (atomic dprob) and one workspace buffer: two eager calls whose fresh pointer
    tables name freshly allocated N-major weights, then a captured graph replayed after b_ptrs.copy_()
    of new weights (the last replay with new expert boundaries too). Weights are randint(-1, 2); dZ and
    d_srelu are NaN-filled and dprob zeroed before every launch, and all three are compared exactly with
    an independent float64 reference (integer sums, so the atomic order is exact too). Every weight set
    stays alive; M is sized to the SM count, so CTAs revisit experts within and across launches. A
    detector, not a proof: a stale descriptor is timing-dependent. The workspace is filled with 0xFF
    before the first eager call and zeroed before the second and before the capture.
    """
    require_no_host_value_reads()
    n, k = 256, 128
    m_tiles = sm_count()
    counts = [256 * (m_tiles // 4), 0, 256 * (m_tiles // 3)]
    counts.append(256 * m_tiles - sum(counts))
    problem = make_problem(counts, n, k, discrete=True, b_major="n", exact=True, seed=201)
    rows = problem["rows"]
    z_save = exact_z(rows, n, seed=202)
    prob = exact_prob(rows, seed=203)
    d_row, d_srelu = bf16_rows(rows, n), bf16_rows(rows, n)
    dprob = torch.zeros((rows, 1, 1), dtype=torch.float32, device="cuda")
    api = dsrelu_api(
        problem, z_save, prob, d_row, dprob, d_srelu=d_srelu, mma_tiler_mn=mma_tiler_mn, cluster_shape_mn=cluster_shape_mn, use_dynamic_sched=use_dynamic_sched
    )
    assert api.check_support()
    api.compile()
    workspace = ws(api)
    alive = [problem["b_storage"]]

    def fresh_weights(seed):
        logical = ternary_weights(problem["experts"], n, k, seed=seed)
        alive.append(store_weights(logical, "n"))
        return logical, pointer_values(alive[-1])

    def launch_and_check(label, logical, counts_now, launch):
        d_row.fill_(float("nan"))
        d_srelu.fill_(float("nan"))
        dprob.zero_()
        launch()
        routed = sum(counts_now)
        reference = exact_backward_reference(problem["a"], logical, counts_now, z_save, prob)
        assert_values_equal(f"{label} dZ", d_row[:routed, :, 0], reference["d_row"])
        assert_values_equal(f"{label} d_srelu", d_srelu[:routed, :, 0], reference["d_srelu"])
        assert_values_equal(f"{label} dprob", dprob[:routed, 0, 0], reference["dprob"])
        assert bool(torch.isnan(d_row[routed:]).all()) and bool(torch.isnan(d_srelu[routed:]).all()), f"{label} wrote rows past the routed total"

    def execute():
        dsrelu_execute(api, problem, z_save, prob, d_row, dprob, d_srelu=d_srelu, workspace=workspace)

    # test/AGENTS.md: poison the caller workspace with both 0xFF and 0x00; each launch's helper kernel rewrites every slot it reads.
    for call, (seed, poison) in enumerate(((211, 0xFF), (212, 0x00))):
        logical, pointers = fresh_weights(seed)
        problem["b_ptrs"] = pointers.cuda()
        workspace.fill_(poison)
        launch_and_check(f"eager call {call}", logical, counts, execute)
    workspace.zero_()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            execute()
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
def test_dsrelu_bf16_scratch_workspace_bytes(discrete, use_dynamic_sched):
    """The facade's BF16 scratch_workspace_bytes() runs the lazy check_support() and reports a positive
    multiple of 128 that covers one 128-byte descriptor slot per expert in discrete mode; the
    deterministic dprob slots are not part of it."""
    problem = make_problem([256, 0, 512], 64, 64, discrete=discrete, b_major="n", seed=221)
    rows = problem["rows"]
    api = dsrelu_api(
        problem,
        make_z(rows, 64, seed=222),
        make_prob(rows, seed=223),
        bf16_rows(rows, 64),
        torch.zeros((rows, 1, 1), dtype=torch.float32, device="cuda"),
        dprob_workspace=dprob_slots(rows, 64),
        deterministic=True,
        use_dynamic_sched=use_dynamic_sched,
    )
    nbytes = api.scratch_workspace_bytes()
    assert nbytes >= 128 and nbytes % 128 == 0, nbytes
    if discrete:
        assert nbytes >= 128 * problem["experts"], nbytes
    assert api.scratch_workspace_bytes() == nbytes


def _dsrelu_operands(rows, n):
    z_save = make_z(rows, n, seed=233)
    prob = make_prob(rows, seed=234)
    d_row, d_srelu = bf16_rows(rows, n), bf16_rows(rows, n)
    dprob = torch.zeros((rows, 1, 1), dtype=torch.float32, device="cuda")
    return z_save, prob, d_row, d_srelu, dprob, dprob_slots(rows, n)


@pytest.mark.L0
@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
def test_dsrelu_bf16_workspace_rejected_before_launch(discrete, monkeypatch):
    """Recipe R2: execute() requires the caller's workspace and checks it before any launch, also for
    a call with no rows: missing, too short, misaligned, non-contiguous, host and wrong-device buffers
    raise; the compiled consumer and the dprob fold are intercepted, so no rejected buffer reaches a
    kernel. A correct buffer launches kernel and fold once per call with rows, nothing without rows."""
    from cudnn.frost.buffers import DeviceView

    problem = make_problem([256, 0, 512], 64, 64, discrete=discrete, b_major="n", seed=231)
    rows = problem["rows"]
    z_save, prob, d_row, d_srelu, dprob, slots = _dsrelu_operands(rows, 64)
    api, launches = intercepted_dsrelu_api(problem, z_save, prob, d_row, dprob, monkeypatch, d_srelu=d_srelu, dprob_workspace=slots, deterministic=True)

    def execute(workspace, **overrides):
        dsrelu_execute(api, problem, z_save, prob, d_row, dprob, d_srelu=d_srelu, dprob_workspace=slots, workspace=workspace, **overrides)

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
        d_row_tensor=mn1(torch.empty((0, 64), dtype=torch.bfloat16, device="cuda")),
        d_srelu_tensor=mn1(torch.empty((0, 64), dtype=torch.bfloat16, device="cuda")),
        prob_tensor=torch.empty((0, 1, 1), dtype=torch.float32, device="cuda"),
        dprob_tensor=torch.zeros((0, 1, 1), dtype=torch.float32, device="cuda"),
        dprob_workspace_tensor=dprob_slots(0, 64),
    )
    for workspace, match in cases:
        for overrides in ({}, no_rows):
            with pytest.raises(ValueError, match=match):
                execute(workspace, **overrides)
    assert not launches, "a rejected workspace reached the launch"
    execute(aligned)
    assert [kind for kind, _ in launches] == ["kernel", "fold"]
    execute(aligned, **no_rows)
    assert len(launches) == 2, "a call with no rows launched"


@pytest.mark.L0
@pytest.mark.parametrize(
    "discrete, operand",
    [
        (False, name)
        for name in (
            "a_tensor",
            "c_tensor",
            "d_row_tensor",
            "d_srelu_tensor",
            "padded_offsets",
            "prob_tensor",
            "dprob_tensor",
            "dprob_workspace_tensor",
            "b_tensor",
        )
    ]
    + [(True, "b_ptrs")],
)
def test_dsrelu_bf16_workspace_alias_rejected_before_launch(discrete, operand, monkeypatch):
    """Recipe R2 (test/AGENTS.md, caller workspace alias regressions): scratch that overlaps an operand's
    byte span -- the deterministic dprob slots included -- is rejected before launch with that operand's
    name; the same operand right after the workspace in one allocation is accepted. Every other operand
    stays clear of the tested interval."""
    if operand in ("padded_offsets", "b_ptrs"):
        require_no_host_value_reads()  # the uninitialized views would fail the debug value checks first
    problem = make_problem([256, 0, 512], 64, 64, discrete=discrete, b_major="n", seed=241)
    rows = problem["rows"]
    z_save, prob, d_row, d_srelu, dprob, slots = _dsrelu_operands(rows, 64)
    api, launches = intercepted_dsrelu_api(problem, z_save, prob, d_row, dprob, monkeypatch, d_srelu=d_srelu, dprob_workspace=slots, deterministic=True)
    nbytes = api.scratch_workspace_bytes()
    operands = dict(
        a_tensor=problem["a"],
        c_tensor=z_save,
        d_row_tensor=d_row,
        d_srelu_tensor=d_srelu,
        padded_offsets=problem["offsets"],
        prob_tensor=prob,
        dprob_tensor=dprob,
        dprob_workspace_tensor=slots,
    )
    operands.update({"b_ptrs": problem["b_ptrs"]} if discrete else {"b_tensor": problem["b"]})
    storage, disjoint, overlapping = aliasing_views(operands[operand], nbytes)

    def execute(view):
        dsrelu_execute(api, problem, z_save, prob, d_row, dprob, d_srelu=d_srelu, dprob_workspace=slots, workspace=storage[:nbytes], **{operand: view})

    execute(disjoint)
    assert [kind for kind, _ in launches] == ["kernel", "fold"]
    with pytest.raises(ValueError, match="workspace must not overlap " + operand):
        execute(overlapping)
    assert len(launches) == 2, "an aliased workspace reached the launch"


def _mxfp8_dsrelu_problem(request):
    """A complete dense MXFP8 dSReLU problem (FP8 A/B, BF16 C, FP8 outputs with scale factors) of the block-scaled suite."""
    from gemm.cutedsl.test_grouped_gemm_dsrelu_utils import (
        allocate_grouped_gemm_dsrelu_tensors,
        allocate_grouped_gemm_input_tensors,
        grouped_gemm_dsrelu_init,
    )

    cfg = grouped_gemm_dsrelu_init(
        request=request,
        ab_dtype=torch.float8_e4m3fn,
        c_dtype=torch.bfloat16,
        d_dtype=torch.float8_e4m3fn,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=False,
        discrete_col_sfd=True,
        b_major="k",
    )
    inputs = allocate_grouped_gemm_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        l=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        b_major=cfg["b_major"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
    )
    inputs, outputs = allocate_grouped_gemm_dsrelu_tensors(
        tensor_m=inputs["tensor_m"],
        n=cfg["n"],
        l=cfg["l"],
        ab_dtype=cfg["ab_dtype"],
        c_dtype=cfg["c_dtype"],
        d_dtype=cfg["d_dtype"],
        cd_major=cfg["cd_major"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        input_tensors=inputs,
    )
    return cfg, inputs, outputs


@pytest.mark.L0
def test_block_scaled_dsrelu_takes_no_workspace(request):
    """The block-scaled backend keeps its own descriptor workspace: the facade reports 0 scratch bytes
    and its execute() rejects workspace= (a provided but unused argument raises, Rule 1)."""
    from cudnn import GroupedGemmDsreluSm100

    cfg, inputs, outputs = _mxfp8_dsrelu_problem(request)
    api = GroupedGemmDsreluSm100(
        sample_a=inputs["a_tensor"],
        sample_b=inputs["b_tensor"],
        sample_c=inputs["c_tensor"],
        sample_d_row=outputs["d_row_tensor"],
        sample_d_col=outputs["d_col_tensor"],
        sample_d_srelu=outputs["d_srelu_tensor"],
        sample_sfa=inputs["sfa_tensor"],
        sample_sfb=inputs["sfb_tensor"],
        sample_padded_offsets=inputs["padded_offsets_tensor"],
        sample_alpha=inputs["alpha_tensor"],
        sample_prob=inputs["prob_tensor"],
        sample_dprob=outputs["dprob_tensor"],
        sample_amax=outputs.get("amax_tensor"),
        sample_sfd_row=outputs.get("sfd_row_tensor"),
        sample_sfd_col=outputs.get("sfd_col_tensor"),
        sample_sfd_col_d_srelu=outputs.get("sfd_col_d_srelu_tensor"),
        sample_norm_const=inputs.get("norm_const_tensor"),
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
            c_tensor=inputs["c_tensor"],
            d_row_tensor=outputs["d_row_tensor"],
            d_col_tensor=outputs["d_col_tensor"],
            d_srelu_tensor=outputs["d_srelu_tensor"],
            sfa_tensor=inputs["sfa_tensor"],
            sfb_tensor=inputs["sfb_tensor"],
            padded_offsets=inputs["padded_offsets_tensor"],
            alpha_tensor=inputs["alpha_tensor"],
            prob_tensor=inputs["prob_tensor"],
            dprob_tensor=outputs["dprob_tensor"],
            sfd_row_tensor=outputs.get("sfd_row_tensor"),
            sfd_col_tensor=outputs.get("sfd_col_tensor"),
            sfd_col_d_srelu_tensor=outputs.get("sfd_col_d_srelu_tensor"),
            norm_const_tensor=inputs.get("norm_const_tensor"),
            amax_tensor=outputs.get("amax_tensor"),
            workspace=torch.empty(128, dtype=torch.uint8, device="cuda"),
        )


# --------------------------------------------------------------------------- lazy facade and compile placement


@pytest.mark.L0
def test_dsrelu_bf16_facade_defers_validation_to_check_support(caplog):
    """The facade only stores its arguments: backend selection, the implementation's constructor and its
    checks run in check_support(), which compile() and scratch_workspace_bytes() reach too. Constructing
    never raises for these errors; an unknown framework is still refused at construction. execute()
    before compile() raises; a BF16 scale control at execute() is refused by name. The experimental
    warning is logged once per implementation class, at check_support()."""
    import numpy as np

    from cudnn import GroupedGemmDsreluSm100
    from cudnn.api_base import _reset_experimental_api_warning_registry

    problem = make_problem([256, 256], 64, 64, b_major="n", seed=251)
    rows = problem["rows"]
    z_save = make_z(rows, 64, seed=252)
    prob = make_prob(rows, seed=253)
    d_row = bf16_rows(rows, 64)
    dprob = torch.zeros((rows, 1, 1), dtype=torch.float32, device="cuda")
    base = dict(sample_a=problem["a"], sample_c=z_save, sample_d_row=d_row, sample_padded_offsets=problem["offsets"], sample_prob=prob, sample_dprob=dprob)
    cases = [
        (dict(sample_b=problem["b"].to(torch.float8_e4m3fn)), "mixed dtype families"),
        (dict(sample_b=problem["b"], sample_alpha=torch.ones(2, device="cuda")), "grouped_gemm_dsrelu_sm100: BF16 forbids scale control sample_alpha"),
        (dict(sample_b=problem["b"], num_experts=2, b_shape=(64, 64), b_dtype=torch.bfloat16), "but not both"),
        (dict(sample_b=problem["b"], tanh_mode=None), re.escape(f"{TANH_MODE_ERROR}, got None")),
        (dict(sample_b=problem["b"], tanh_mode="fast"), re.escape(f"{TANH_MODE_ERROR}, got 'fast'")),
        (dict(sample_b=problem["b"], deterministic=True), "sample_dprob_workspace is required"),
    ]
    for extra, match in cases:
        for entry in ("check_support", "compile", "scratch_workspace_bytes"):
            api = GroupedGemmDsreluSm100(**base, **extra)
            with pytest.raises(ValueError, match=match):
                getattr(api, entry)()
    execute_args = dict(
        a_tensor=problem["a"],
        c_tensor=z_save,
        d_row_tensor=d_row,
        d_col_tensor=None,
        sfa_tensor=None,
        padded_offsets=problem["offsets"],
        alpha_tensor=None,
        prob_tensor=prob,
        dprob_tensor=dprob,
        b_tensor=problem["b"],
        workspace=torch.empty(128, dtype=torch.uint8, device="cuda"),
    )
    with pytest.raises(RuntimeError, match="Kernel not compiled"):
        GroupedGemmDsreluSm100(**base, sample_b=problem["b"]).execute(**execute_args)
    with pytest.raises(ValueError, match="Unsupported tensor framework 'numpy' for GroupedGemmDsreluSm100"):
        GroupedGemmDsreluSm100(sample_a=np.zeros((256, 64, 1), dtype=np.float32), sample_b=problem["b"])

    _reset_experimental_api_warning_registry()
    caplog.set_level(logging.WARNING)
    caplog.clear()
    api = GroupedGemmDsreluSm100(**base, sample_b=problem["b"])
    assert "GroupedGemmDsreluBf16API is an experimental API" not in caplog.text, "the facade's constructor built the implementation"
    assert api.check_support()
    assert caplog.text.count("GroupedGemmDsreluBf16API is an experimental API") == 1
    with pytest.raises(RuntimeError, match="Kernel not compiled"):
        api.execute(**execute_args)
    with pytest.raises(ValueError, match="BF16 forbids scale control alpha_tensor"):
        api.execute(**{**execute_args, "alpha_tensor": torch.ones(2, device="cuda")})
    assert GroupedGemmDsreluSm100(**base, sample_b=problem["b"]).check_support()
    assert caplog.text.count("GroupedGemmDsreluBf16API is an experimental API") == 1


@pytest.mark.L0
@pytest.mark.parametrize("placement", ["current-device", "other-device"])
def test_dsrelu_bf16_compile_pins_operand_device_and_compile_lock(placement, monkeypatch):
    """check_support() reads the capability of the operands' GPU and compile() -- the kernel and the
    deterministic dprob fold -- runs cute.compile under the process-wide DSL compile lock with the
    operands' device current, restoring the caller's device; with two GPUs the operands live on the
    one that is not current."""
    import cutlass.cute as cute

    import cudnn.gemm.cutedsl.grouped.srelu._bf16_api as srelu_bf16_api
    from cudnn.gemm.cutedsl.grouped.dsrelu import _dprob_fold as dprob_fold

    caller = torch.cuda.current_device()
    index = caller
    if placement == "other-device":
        if torch.cuda.device_count() < 2:
            pytest.skip("needs two visible GPUs")
        index = (caller + 1) % torch.cuda.device_count()
        if torch.cuda.get_device_capability(index) not in ((10, 0), (10, 3)):
            pytest.skip(f"cuda:{index} is not SM100/SM103")
    capabilities = []
    real_capability = srelu_bf16_api.get_compute_capability

    def capability(*args, **kwargs):
        capabilities.append(args[0] if args else kwargs.get("device_index"))
        return real_capability(*args, **kwargs)

    spy = CompileSpy(cute.compile)
    # The dSReLU support check calls the shared device check of srelu/_bf16_api.py.
    monkeypatch.setattr(srelu_bf16_api, "get_compute_capability", capability)
    monkeypatch.setattr(cute, "compile", spy)
    # The fold is compiled once per device for the whole process: empty that cache, so compile() builds it here.
    monkeypatch.setattr(dprob_fold, "_compiled", {})
    with torch.cuda.device(index):
        problem = make_problem([256, 0, 512], 64, 64, discrete=True, b_major="n", seed=261)
        rows = problem["rows"]
        z_save, prob, d_row, d_srelu, dprob, slots = _dsrelu_operands(rows, 64)
    api = dsrelu_api(problem, z_save, prob, d_row, dprob, d_srelu=d_srelu, dprob_workspace=slots, deterministic=True)
    assert api.check_support()
    api.compile()
    assert torch.cuda.current_device() == caller, "compile() did not restore the caller's device"
    assert capabilities and all(seen == index for seen in capabilities), f"capability queried for {capabilities}, operands on cuda:{index}"
    assert spy.calls and all(seen == (index, True) for seen in spy.calls), f"(device, lock held) at cute.compile: {spy.calls}, operands on cuda:{index}"
    assert "_DprobFoldKernel" in spy.kinds, f"the dprob fold was not compiled through cute.compile: {spy.kinds}"


# --------------------------------------------------------------------------- support checks


@pytest.mark.L0
def test_dsrelu_bf16_rejects_invalid_configurations():
    from cudnn import GroupedGemmDsreluSm100, grouped_gemm_dsrelu_wrapper_sm100

    problem = make_problem([256, 256], 128, 64, discrete=True, b_major="n", seed=131)
    z_save = make_z(problem["rows"], 128, seed=132)
    prob = make_prob(problem["rows"], seed=133)

    def expect(match, **kwargs):
        with pytest.raises((ValueError, RuntimeError), match=match):
            run_dsrelu(problem, z_save, prob, **{"clamp_scale": CLAMP_SCALE, **kwargs})

    expect("BF16 forbids scale control alpha_tensor", alpha_tensor=torch.ones(2, dtype=torch.float32, device="cuda"))
    expect("BF16 forbids scale control generate_dbias", generate_dbias=True)
    expect("d_dtype", d_dtype=torch.float16)
    for clamp_scale in (CLAMP_SCALE, None):
        expect(re.escape(f"{TANH_MODE_ERROR}, got None"), clamp_scale=clamp_scale, tanh_mode=None)
        expect(re.escape(f"{TANH_MODE_ERROR}, got 'fast'"), clamp_scale=clamp_scale, tanh_mode="fast")
    with pytest.raises(ValueError, match="c_tensor"):
        run_dsrelu(problem, None, prob, clamp_scale=CLAMP_SCALE)

    narrow = make_problem([256], 48, 64, discrete=True, b_major="n", seed=134)
    with pytest.raises(ValueError, match="multiple of 32"):
        run_dsrelu(narrow, make_z(256, 48, seed=135), make_prob(256, seed=136), clamp_scale=CLAMP_SCALE)
    odd_k = make_problem([256], 64, 36, discrete=True, b_major="n", seed=137)
    with pytest.raises(ValueError, match="multiple of 8"):
        run_dsrelu(odd_k, make_z(256, 64, seed=138), make_prob(256, seed=139), clamp_scale=CLAMP_SCALE)

    # The weights are checked before the backend is selected, for both backends.
    with pytest.raises(ValueError, match="Must provide either"):
        grouped_gemm_dsrelu_wrapper_sm100(a_tensor=problem["a"], c_tensor=z_save, padded_offsets=problem["offsets"], prob_tensor=prob)
    with pytest.raises(ValueError, match="n and b_dtype are required for discrete mode"):
        grouped_gemm_dsrelu_wrapper_sm100(a_tensor=problem["a"], c_tensor=z_save, b_ptrs=problem["b_ptrs"], padded_offsets=problem["offsets"], prob_tensor=prob)

    dense = make_problem([256, 256], 128, 64, b_major="n", seed=140)
    with pytest.raises(ValueError, match="not both"):
        run_dsrelu(dense, z_save, prob, clamp_scale=None, b_ptrs=problem["b_ptrs"])
    rows = dense["rows"]
    d_row = mn1(torch.empty((rows, 128), dtype=torch.bfloat16, device="cuda"))
    dprob = torch.zeros((rows, 1, 1), dtype=torch.float32, device="cuda")
    api = GroupedGemmDsreluSm100(
        sample_a=dense["a"],
        sample_b=dense["b"],
        sample_c=z_save,
        sample_d_row=d_row,
        sample_padded_offsets=dense["offsets"],
        sample_prob=prob,
        sample_dprob=dprob,
        deterministic=True,
    )
    with pytest.raises(ValueError, match="sample_dprob_workspace is required"):
        api.check_support()


DSRELU_DEVICE_VALUE_CHILD = r"""
from cudnn import GroupedGemmDsreluSm100
from cudnn.gemm.cutedsl.grouped import backend_utils

assert backend_utils.DEBUG_VALIDATE_DEVICE_VALUES, "CUDNN_FE_GROUPED_GEMM_VALIDATE_DEVICE_VALUES=1 was not read at import"


def mn1(t):
    rows, cols = t.shape
    return t.as_strided((rows, cols, 1), (cols, 1, rows * cols))


rows, n, k, experts = 512, 64, 64, 2
a = mn1(torch.randn(rows, k, device="cuda").to(torch.bfloat16))
z = mn1(torch.randn(rows, n, device="cuda").to(torch.bfloat16))
weights = torch.randn(experts, k, n, device="cuda").to(torch.bfloat16)  # N-major: [K, N] per expert
b_ptrs = torch.tensor([weights[e].data_ptr() for e in range(experts)], dtype=torch.int64, device="cuda")
prob = torch.rand(rows, 1, 1, device="cuda")
dprob = torch.zeros(rows, 1, 1, device="cuda")
d_row = mn1(torch.empty(rows, n, dtype=torch.bfloat16, device="cuda"))
offsets = torch.tensor([256, 512], dtype=torch.int32, device="cuda")
api = GroupedGemmDsreluSm100(
    sample_a=a,
    sample_c=z,
    sample_d_row=d_row,
    sample_padded_offsets=offsets,
    sample_prob=prob,
    sample_dprob=dprob,
    num_experts=experts,
    b_shape=(n, k),
    b_dtype=torch.bfloat16,
    b_major="n",
)
assert api.check_support()
launches = []
api._implementation._compiled_kernel = lambda *args: launches.append(args)
workspace = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")


def run(**overrides):
    kwargs = dict(
        a_tensor=a, c_tensor=z, d_row_tensor=d_row, d_col_tensor=None, sfa_tensor=None, padded_offsets=offsets, alpha_tensor=None,
        prob_tensor=prob, dprob_tensor=dprob, b_ptrs=b_ptrs,
    )
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
    d_row_tensor=mn1(torch.empty(0, n, dtype=torch.bfloat16, device="cuda")),
    prob_tensor=torch.empty(0, 1, 1, device="cuda"),
    dprob_tensor=torch.zeros(0, 1, 1, device="cuda"),
)
run(padded_offsets=ends([0, 0]), **empty)
assert len(launches) == 1, "a call with no rows launched"
print("DSRELU_DEVICE_VALUE_CHECKS_OK")
"""


@pytest.mark.L0
def test_dsrelu_bf16_offset_and_pointer_values_checked_only_on_request(tmp_path, monkeypatch):
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
        problem = make_problem([256, 256], 64, 64, discrete=True, b_major="n", seed=271)
        rows = problem["rows"]
        z_save, prob, d_row, d_srelu, dprob, _ = _dsrelu_operands(rows, 64)
        api, launches = intercepted_dsrelu_api(problem, z_save, prob, d_row, dprob, monkeypatch)
        bad_offsets = torch.tensor([128, 768], dtype=torch.int32, device="cuda")
        bad_pointers = torch.tensor([0, 8], dtype=torch.int64, device="cuda")
        dsrelu_execute(api, problem, z_save, prob, d_row, dprob, workspace=ws(api), padded_offsets=bad_offsets, b_ptrs=bad_pointers)
        assert [kind for kind, _ in launches] == ["kernel"]
    run_python_child(DSRELU_DEVICE_VALUE_CHILD, env={DEVICE_VALUES_ENV: "1"}, cwd=tmp_path, marker="DSRELU_DEVICE_VALUE_CHECKS_OK")


@pytest.mark.L0
def test_dsrelu_bf16_deterministic_dprob_side_stream_unordered_init():
    """The BF16 wrapper zero-fills dprob and the deterministic slot buffer -- both accumulated into -- on current_stream,
    not on torch's stream (R1/R2; the block-scaled twin is test_grouped_gemm_dsrelu.py::
    test_grouped_gemm_dsrelu_deterministic_dprob_side_stream_unordered_init). Poisoned blocks of exactly their sizes
    are freed into torch's stream pool, so an allocation there gets those bytes back, and torch's stream is kept busy,
    so a zero-fill queued there would land after the kernel and the fold accumulated on the caller's stream. dZ,
    d_srelu and dprob must equal bit for bit the same call on torch's stream."""
    from cuda.bindings import driver as cuda

    from cudnn.gemm.cutedsl.grouped.dsrelu.api import _dprob_n_slots

    n = 512
    problem = make_problem([256, 0, 512], n, 128, discrete=True, b_major="n", seed=301)
    rows = problem["rows"]
    z_save = make_z(rows, n, seed=302)
    prob = make_prob(rows, seed=303)
    expected = run_dsrelu(problem, z_save, prob, clamp_scale=CLAMP_SCALE, deterministic=True)
    torch.cuda.synchronize()
    slots = _dprob_n_slots(n, (256, 256), (2, 1), True)
    assert slots > 1, "the slot buffer needs more than one N tile to be exercised"
    side = torch.cuda.Stream()
    for shape in ((rows, slots, 1), (rows, 1, 1)):
        poison = torch.full(shape, 1e30, dtype=torch.float32, device="cuda")
        del poison
    torch.cuda._sleep(200_000_000)
    on_side = run_dsrelu(problem, z_save, prob, clamp_scale=CLAMP_SCALE, deterministic=True, current_stream=cuda.CUstream(side.cuda_stream))
    torch.cuda.synchronize()
    routed = problem["routed"]
    assert torch.isfinite(on_side["dprob_tensor"]).all(), "dprob picked up a poisoned block: zero-filled on the wrong stream"
    for key in ("d_row_tensor", "d_srelu_tensor", "dprob_tensor"):
        assert_bitwise_equal(f"{key} on a caller stream vs torch's", on_side[key][:routed], expected[key][:routed])


@pytest.mark.L0
@pytest.mark.parametrize("call", ["wrapper", "class"])
def test_dsrelu_bf16_launches_on_the_operands_device(call):
    """python/cudnn/AGENTS.md ("Compiled CuTe launchables belong to their CUDA context", "Restore the caller's device
    after a launch on another device"), deterministic, so the dprob fold launches too: the operands live on a GPU
    that is not current and no stream is given. The wrapper allocates its outputs, dprob buffers and scratch on that
    GPU's current stream and launches there; the class API compiles and launches on the operands' device. Both equal
    the same call made with that GPU current, and the caller's device stays current."""
    caller, other = other_bf16_device()
    with torch.cuda.device(other):
        problem = make_problem([256, 0, 512], 256, 128, discrete=True, b_major="n", seed=291)
        rows = problem["rows"]
        z_save = make_z(rows, 256, seed=292)
        prob = make_prob(rows, seed=293)
        expected = run_dsrelu(problem, z_save, prob, clamp_scale=CLAMP_SCALE, deterministic=True)
        d_row, d_srelu = bf16_rows(rows, 256, fill=float("nan")), bf16_rows(rows, 256, fill=float("nan"))
        dprob = torch.zeros((rows, 1, 1), dtype=torch.float32, device="cuda")
        slots = dprob_slots(rows, 256)
    if call == "wrapper":
        out = run_dsrelu(problem, z_save, prob, clamp_scale=CLAMP_SCALE, deterministic=True)
        d_row, d_srelu, dprob = out["d_row_tensor"], out["d_srelu_tensor"], out["dprob_tensor"]
    else:
        api = dsrelu_api(problem, z_save, prob, d_row, dprob, d_srelu=d_srelu, dprob_workspace=slots, deterministic=True, tanh_clamp_scale=CLAMP_SCALE)
        assert api.check_support()
        api.compile()
        workspace = ws(api, device=f"cuda:{other}")
        dsrelu_execute(api, problem, z_save, prob, d_row, dprob, d_srelu=d_srelu, dprob_workspace=slots, workspace=workspace)
    assert torch.cuda.current_device() == caller, "the call did not restore the caller's device"
    torch.cuda.synchronize(other)
    routed = problem["routed"]
    for name, got, want in (
        ("dZ", d_row, expected["d_row_tensor"]),
        ("d_srelu", d_srelu, expected["d_srelu_tensor"]),
        ("dprob", dprob, expected["dprob_tensor"]),
    ):
        assert got.device.index == other, f"{name} is on cuda:{got.device.index}"
        assert_bitwise_equal(f"{name} launched from another current device", got[:routed], want[:routed])


DSRELU_RENDER_CHILD = r"""
from cudnn import GroupedGemmDsreluSm100

rows, n, d, experts = 512, 256, 128, 2
for weights in ("discrete", "dense"):
    kwargs = dict(
        sample_a=mn1(torch.randn(rows, d, device="cuda").to(torch.bfloat16)),
        sample_c=mn1(torch.randn(rows, n, device="cuda").to(torch.bfloat16)),
        sample_d_row=mn1(torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")),
        sample_d_srelu=mn1(torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")),
        sample_padded_offsets=torch.tensor([256, 512], dtype=torch.int32, device="cuda"),
        sample_prob=torch.rand(rows, 1, 1, device="cuda"),
        sample_dprob=torch.zeros(rows, 1, 1, device="cuda"),
        sample_dprob_workspace=torch.zeros(GroupedGemmDsreluSm100.dprob_workspace_shape(rows, n), device="cuda"),
        deterministic=True,
        tanh_clamp_scale=16.0,
        tanh_mode="approx",
    )
    if weights == "discrete":
        kwargs.update(num_experts=experts, b_shape=(n, d), b_dtype=torch.bfloat16, b_major="n")
    else:
        kwargs.update(sample_b=torch.randn(experts, d, n, device="cuda").to(torch.bfloat16).permute(2, 1, 0))
    api = GroupedGemmDsreluSm100(**kwargs)
    assert api.check_support()
    api.compile()
    save_ptx(api, weights)
print("DSRELU_RENDER_OK")
"""


@pytest.mark.L0
def test_dsrelu_bf16_rendering_orders_every_release(tmp_path):
    """python/cudnn/AGENTS.md Rule 11, the proof, in the backward kernel's PTX for discrete and dense (N-major)
    weights: the eight tile-info releases (TMA, MMA, epilogue and Z_save-load warps, two each) arrive at addresses
    computed from the four loaded words, and the Z_save stage release from the 16 registers its loads wrote, through
    the release-zero argument; every warp-subset named barrier is the unaligned form; only discrete weights acquire
    the per-expert tensormaps. A fresh process compiles both kernels with the DSL keeping their PTX."""
    env, ptx_dir = render_env(tmp_path)
    run_python_child(RENDER_CHILD_PRELUDE + DSRELU_RENDER_CHILD, env=env, cwd=tmp_path, marker="DSRELU_RENDER_OK")
    for weights in ("discrete", "dense"):
        ptx = (ptx_dir / f"{weights}.ptx").read_text()
        assert_release_rendering(f"dsrelu {weights}", ptx, sites=9, wide_words=16, acquire=weights == "discrete")


@pytest.mark.L0
@pytest.mark.parametrize("capability", [(10, 7), (9, 0), (12, 0)], ids=["sm107", "sm90", "sm120"])
def test_dsrelu_bf16_rejects_other_architectures(monkeypatch, capability):
    import cudnn.gemm.cutedsl.grouped.srelu._bf16_api as bf16_api
    from cudnn import GroupedGemmDsreluSm100

    monkeypatch.setattr(bf16_api, "get_compute_capability", lambda *args, **kwargs: capability)
    problem = make_problem([256], 64, 64, b_major="n", seed=141)
    z_save = make_z(256, 64, seed=142)
    prob = make_prob(256, seed=143)
    d_row = mn1(torch.empty((256, 64), dtype=torch.bfloat16, device="cuda"))
    api = GroupedGemmDsreluSm100(
        sample_a=problem["a"],
        sample_b=problem["b"],
        sample_c=z_save,
        sample_d_row=d_row,
        sample_padded_offsets=problem["offsets"],
        sample_prob=prob,
        sample_dprob=torch.zeros((256, 1, 1), dtype=torch.float32, device="cuda"),
    )
    with pytest.raises(RuntimeError, match="SM100, SM103 only"):
        api.check_support()


@pytest.mark.L0
def test_block_scaled_dsrelu_activation_options():
    """The block-scaled kernel evaluates tanh with MUFU tanh.approx: with a clamp scale it refuses
    "accurate", and it validates tanh_mode as the BF16 backend does (None and unknown strings raise).
    Plain mode accepts either mode, so the default ("approx") too; clamped mode accepts the default.
    The wrapper refuses before it reads the FP8 operands, the facade in check_support() before it
    builds the block-scaled implementation."""
    from cudnn import GroupedGemmDsreluSm100, grouped_gemm_dsrelu_wrapper_sm100
    from cudnn.gemm.cutedsl.grouped.srelu.api import _check_block_scaled_activation_options

    a = torch.zeros((256, 128, 1), dtype=torch.float8_e4m3fn, device="cuda")
    b = torch.zeros((128, 128, 1), dtype=torch.float8_e4m3fn, device="cuda")
    rejected = [
        (
            dict(tanh_clamp_scale=CLAMP_SCALE, tanh_mode="accurate"),
            re.escape("grouped_gemm_dsrelu_sm100: tanh_mode='accurate' with tanh_clamp_scale is supported only by the BF16 kernel"),
        ),
        (dict(tanh_clamp_scale=CLAMP_SCALE, tanh_mode=None), re.escape(f"{TANH_MODE_ERROR}, got None")),
        (dict(tanh_mode=None), re.escape(f"{TANH_MODE_ERROR}, got None")),
        (dict(tanh_mode="fast"), re.escape(f"{TANH_MODE_ERROR}, got 'fast'")),
    ]
    for kwargs, match in rejected:
        with pytest.raises(ValueError, match=match):
            grouped_gemm_dsrelu_wrapper_sm100(a_tensor=a, b_tensor=b, **kwargs)
        with pytest.raises(ValueError, match=match):
            GroupedGemmDsreluSm100(sample_a=a, sample_b=b, **kwargs).check_support()
    for clamp_scale, tanh_mode in ((None, "approx"), (None, "accurate"), (CLAMP_SCALE, "approx")):
        _check_block_scaled_activation_options("grouped_gemm_dsrelu_sm100", clamp_scale, tanh_mode, "acc")


@pytest.mark.L0
def test_block_scaled_dsrelu_generate_d_srelu_false(request):
    """generate_d_srelu=False on the block-scaled path skips d_srelu and its column scale factors
    without changing the other outputs. MXFP8, so the outputs carry row and column scale factors."""
    from cudnn import grouped_gemm_dsrelu_wrapper_sm100
    from cuda.bindings import driver as cuda

    cfg, inputs, _ = _mxfp8_dsrelu_problem(request)
    common = dict(
        a_tensor=inputs["a_tensor"],
        b_tensor=inputs["b_tensor"],
        c_tensor=inputs["c_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        sfb_tensor=inputs["sfb_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        prob_tensor=inputs["prob_tensor"],
        norm_const_tensor=inputs.get("norm_const_tensor"),  # FP8 outputs carry scale factors
        acc_dtype=cfg["acc_dtype"],
        d_dtype=cfg["d_dtype"],
        cd_major=cfg["cd_major"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sf_vec_size=cfg["sf_vec_size"],
        vector_f32=cfg["vector_f32"],
        m_aligned=cfg["m_aligned"],
        discrete_col_sfd=cfg["discrete_col_sfd"],
        deterministic=True,
        current_stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    with_h = grouped_gemm_dsrelu_wrapper_sm100(**common)
    without_h = grouped_gemm_dsrelu_wrapper_sm100(**common, generate_d_srelu=False)
    torch.cuda.synchronize()
    assert with_h["d_srelu_tensor"] is not None and with_h["sfd_col_d_srelu_tensor"] is not None
    assert without_h["d_srelu_tensor"] is None and without_h["sfd_col_d_srelu_tensor"] is None
    # The outputs the block-scaled determinism checks compare; they leave the scale factors out too.
    for key in ("d_row_tensor", "d_col_tensor", "dprob_tensor"):
        assert_bitwise_equal(key, without_h[key], with_h[key])


@pytest.mark.L0
def test_dsrelu_bf16_rejects_jax_arrays():
    """The BF16 backend is torch-only: BF16 JAX arrays are refused by the wrapper (before any allocation)
    and, the facade accepting JAX arrays at construction, by its check_support()."""
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from cudnn import GroupedGemmDsreluSm100, grouped_gemm_dsrelu_wrapper_sm100

    a = jnp.zeros((256, 64, 1), dtype=jnp.bfloat16)
    b = jnp.zeros((64, 64, 1), dtype=jnp.bfloat16)
    c = jnp.zeros((256, 64, 1), dtype=jnp.bfloat16)
    offsets = jnp.asarray([256], dtype=jnp.int32)
    prob = jnp.ones((256, 1, 1), dtype=jnp.float32)
    match = "The BF16 grouped GEMM dSReLU backend supports torch tensors only"
    with pytest.raises(ValueError, match=match):
        grouped_gemm_dsrelu_wrapper_sm100(a_tensor=a, b_tensor=b, c_tensor=c, padded_offsets=offsets, prob_tensor=prob)
    api = GroupedGemmDsreluSm100(sample_a=a, sample_b=b, sample_c=c, sample_padded_offsets=offsets, sample_prob=prob)
    with pytest.raises(ValueError, match=match):
        api.check_support()


@pytest.mark.L0
def test_jax_dense_dsrelu_rejects_generate_d_srelu_false():
    """The JAX block-scaled dense path always writes d_srelu, so generate_d_srelu=False is refused there
    rather than ignored (before any operand is read)."""
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from cudnn import grouped_gemm_dsrelu_wrapper_sm100

    a = jnp.zeros((256, 128, 1), dtype=jnp.float8_e4m3fn)
    b = jnp.zeros((128, 128, 1), dtype=jnp.float8_e4m3fn)
    c = jnp.zeros((256, 128, 1), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="generate_d_srelu is unsupported for the JAX block-scaled path"):
        grouped_gemm_dsrelu_wrapper_sm100(a_tensor=a, b_tensor=b, c_tensor=c, generate_d_srelu=False)


@pytest.mark.L0
@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
def test_dsrelu_bf16_rejects_per_thread_stream(discrete, monkeypatch):
    """Recipe R1: CUstream(2) (cudaStreamPerThread) has no safe torch stream. The wrapper refuses it
    where it allocates the outputs; the discrete class API where it records the pointer table's stream,
    after the workspace checks and before the launch."""
    from cuda.bindings import driver as cuda

    problem = make_problem([256, 256], 64, 64, discrete=discrete, b_major="n", seed=281)
    z_save = make_z(problem["rows"], 64, seed=282)
    prob = make_prob(problem["rows"], seed=283)
    with pytest.raises(ValueError, match="cudaStreamPerThread"):
        run_dsrelu(problem, z_save, prob, clamp_scale=None, current_stream=cuda.CUstream(2))
    if discrete:
        _, _, d_row, _, dprob, _ = _dsrelu_operands(problem["rows"], 64)
        api, launches = intercepted_dsrelu_api(problem, z_save, prob, d_row, dprob, monkeypatch)
        with pytest.raises(ValueError, match="cudaStreamPerThread"):
            dsrelu_execute(api, problem, z_save, prob, d_row, dprob, workspace=ws(api), current_stream=cuda.CUstream(2))
        assert not launches


DSRELU_DSL_GATE_CHILD = r"""
import cudnn.gemm.cutedsl.discrete_grouped  # noqa: F401
import cudnn.gemm.cutedsl.grouped as grouped
from cudnn.frost import buffers

BF16_KERNEL_MODULES = (
    "cudnn.frost.tile_dsl.barrier",
    "cudnn.frost.tile_dsl.consume",
    "cudnn.gemm.cutedsl.grouped.srelu.moe_grouped_gemm_srelu",
    "cudnn.gemm.cutedsl.grouped.srelu._bf16_api",
    "cudnn.gemm.cutedsl.grouped.dsrelu.moe_grouped_gemm_dsrelu",
    "cudnn.gemm.cutedsl.grouped.dsrelu._bf16_api",
    "cudnn.gemm.cutedsl.grouped.dsrelu._dprob_fold",
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
dsrelu = grouped.GroupedGemmDsreluSm100(
    sample_a=a,
    sample_c=z,
    sample_d_row=mn1(torch.empty_like(z[:, :, 0])),
    sample_padded_offsets=offsets,
    sample_prob=prob,
    sample_dprob=torch.zeros(rows, 1, 1, device="cuda"),
    b_major="n",
    **discrete,
)
expect_version_error(dsrelu.check_support)
expect_version_error(
    lambda: grouped.grouped_gemm_dsrelu_wrapper_sm100(a_tensor=a, c_tensor=z, b_ptrs=b_ptrs, n=n, b_dtype=torch.bfloat16, b_major="n", padded_offsets=offsets, prob_tensor=prob)
)
assert not loaded(), f"the declined BF16 calls imported {loaded()}"
print("BF16_DSRELU_DSL_GATE_OK")
"""


@pytest.mark.L0
def test_dsrelu_bf16_kernels_are_imported_only_behind_the_dsl_gate(tmp_path):
    run_python_child(DSRELU_DSL_GATE_CHILD, env={}, cwd=tmp_path, marker="BF16_DSRELU_DSL_GATE_OK")
