# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Grouped GEMM dSReLU Backward Kernel (SM100+)

This module tests the contiguous grouped block-scaled GEMM backward pass
with dSReLU activation gradient for MoE (Mixture of Experts) workloads.
"""

import contextlib

import torch
import pytest
from test_utils import torch_fork_set_rng, assert_bitwise_runs, bitwise_bits
from fe_api.grouped_gemm.test_grouped_gemm_dsrelu_utils import (
    run_grouped_gemm_dsrelu_ref,
    with_grouped_gemm_dsrelu_params_fp4,
    with_grouped_gemm_dsrelu_params_fp8,
    allocate_grouped_gemm_dsrelu_tensors,
    allocate_grouped_gemm_input_tensors,
    check_ref_grouped_gemm_dsrelu,
    grouped_gemm_dsrelu_init,
)
from fe_api.grouped_gemm.test_discrete_grouped_gemm_swiglu_utils import (
    allocate_discrete_input_tensors,
    discrete_grouped_gemm_init,
)

GROUPED_GEMM_DSRELU_DYNAMIC_SHAPES_M_VALUES = [64, 320, 576, 832, 1088, 1344, 1600, 1856, 2112, 2368]

DISCRETE_GROUPED_GEMM_DSRELU_SUPPORTED_CONFIGS = [
    pytest.param(torch.float4_e2m1fn_x2, torch.bfloat16, torch.bfloat16, "k", id="fp4-k-major"),
    pytest.param(torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn, "k", id="fp8-k-major"),
    pytest.param(torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn, "n", id="fp8-n-major"),
]


def _dense_ref_inputs_from_discrete(inputs):
    ref_inputs = dict(inputs)
    ref_inputs["b_ref"] = torch.cat(inputs["b_ref_list"], dim=2)
    ref_inputs["sfb_ref"] = torch.cat(inputs["sfb_ref_list"], dim=2)
    return ref_inputs


def _prepare_discrete_dsrelu_inputs(inputs):
    inputs["alpha_tensor"] = torch.ones_like(inputs["alpha_tensor"])
    inputs["prob_tensor"] = torch.ones_like(inputs["prob_tensor"])
    return inputs


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_grouped_gemm_dsrelu_params_fp4
def test_grouped_gemm_dsrelu_compile_execute_fp4(
    ab_dtype,
    c_dtype,
    d_dtype,
    b_major,
    cd_major,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    vector_f32,
    discrete_col_sfd,
    request,
):
    _test_grouped_gemm_dsrelu_compile_execute(
        ab_dtype=ab_dtype,
        c_dtype=c_dtype,
        d_dtype=d_dtype,
        b_major=b_major,
        cd_major=cd_major,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
        vector_f32=vector_f32,
        discrete_col_sfd=discrete_col_sfd,
        request=request,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_grouped_gemm_dsrelu_params_fp8
def test_grouped_gemm_dsrelu_compile_execute_fp8(
    ab_dtype,
    c_dtype,
    d_dtype,
    b_major,
    cd_major,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    vector_f32,
    discrete_col_sfd,
    request,
):
    _test_grouped_gemm_dsrelu_compile_execute(
        ab_dtype=ab_dtype,
        c_dtype=c_dtype,
        d_dtype=d_dtype,
        b_major=b_major,
        cd_major=cd_major,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
        vector_f32=vector_f32,
        discrete_col_sfd=discrete_col_sfd,
        request=request,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_grouped_gemm_dsrelu_params_fp4
def test_grouped_gemm_dsrelu_wrapper_fp4(
    ab_dtype,
    c_dtype,
    d_dtype,
    b_major,
    cd_major,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    vector_f32,
    discrete_col_sfd,
    request,
):
    _test_grouped_gemm_dsrelu_wrapper(
        ab_dtype=ab_dtype,
        c_dtype=c_dtype,
        d_dtype=d_dtype,
        b_major=b_major,
        cd_major=cd_major,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
        vector_f32=vector_f32,
        discrete_col_sfd=discrete_col_sfd,
        request=request,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_grouped_gemm_dsrelu_params_fp8
def test_grouped_gemm_dsrelu_wrapper_fp8(
    ab_dtype,
    c_dtype,
    d_dtype,
    b_major,
    cd_major,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    vector_f32,
    discrete_col_sfd,
    request,
):
    _test_grouped_gemm_dsrelu_wrapper(
        ab_dtype=ab_dtype,
        c_dtype=c_dtype,
        d_dtype=d_dtype,
        b_major=b_major,
        cd_major=cd_major,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
        vector_f32=vector_f32,
        discrete_col_sfd=discrete_col_sfd,
        request=request,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize(
    "ab_dtype",
    [
        pytest.param(torch.float4_e2m1fn_x2, id="fp4"),
        pytest.param(torch.float8_e4m3fn, id="fp8"),
    ],
)
def test_grouped_gemm_dsrelu_wrapper_cache_partial_dynamic_smoke(request, monkeypatch, ab_dtype):
    compile_count, cache_entries = _test_grouped_gemm_dsrelu_wrapper_dynamic_shape_cache_behavior(
        request=request,
        monkeypatch=monkeypatch,
        use_full_dynamic=False,
        ab_dtype=ab_dtype,
    )

    assert compile_count == 1
    assert cache_entries == 1


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize(
    "ab_dtype",
    [
        pytest.param(torch.float4_e2m1fn_x2, id="fp4"),
        pytest.param(torch.float8_e4m3fn, id="fp8"),
    ],
)
def test_grouped_gemm_dsrelu_wrapper_cache_full_dynamic_smoke(request, monkeypatch, ab_dtype):
    compile_count, cache_entries = _test_grouped_gemm_dsrelu_wrapper_dynamic_shape_cache_behavior(
        request=request,
        monkeypatch=monkeypatch,
        use_full_dynamic=True,
        ab_dtype=ab_dtype,
    )

    assert compile_count == 1
    assert cache_entries == 1


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize(
    "ab_dtype",
    [
        pytest.param(torch.float4_e2m1fn_x2, id="fp4"),
        pytest.param(torch.float8_e4m3fn, id="fp8"),
    ],
)
def test_grouped_gemm_dsrelu_wrapper_cache_zero_m_after_compile_partial_dynamic_smoke(request, monkeypatch, ab_dtype):
    compile_count, cache_entries = _test_grouped_gemm_dsrelu_wrapper_zero_m_after_compile_cache_behavior(
        request=request,
        monkeypatch=monkeypatch,
        use_full_dynamic=False,
        ab_dtype=ab_dtype,
    )

    assert compile_count == 1
    assert cache_entries == 1


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize(
    "ab_dtype",
    [
        pytest.param(torch.float4_e2m1fn_x2, id="fp4"),
        pytest.param(torch.float8_e4m3fn, id="fp8"),
    ],
)
def test_grouped_gemm_dsrelu_wrapper_cache_zero_m_after_compile_full_dynamic_smoke(request, monkeypatch, ab_dtype):
    compile_count, cache_entries = _test_grouped_gemm_dsrelu_wrapper_zero_m_after_compile_cache_behavior(
        request=request,
        monkeypatch=monkeypatch,
        use_full_dynamic=True,
        ab_dtype=ab_dtype,
    )

    assert compile_count == 1
    assert cache_entries == 1


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize(
    "ab_dtype",
    [
        pytest.param(torch.float4_e2m1fn_x2, id="fp4"),
        pytest.param(torch.float8_e4m3fn, id="fp8"),
    ],
)
def test_grouped_gemm_dsrelu_wrapper_cache_zero_m_before_compile_partial_dynamic_smoke(request, monkeypatch, ab_dtype):
    compile_count, cache_entries = _test_grouped_gemm_dsrelu_wrapper_zero_m_before_compile_cache_behavior(
        request=request,
        monkeypatch=monkeypatch,
        use_full_dynamic=False,
        ab_dtype=ab_dtype,
    )

    assert compile_count == 1
    assert cache_entries == 1


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize(
    "ab_dtype",
    [
        pytest.param(torch.float4_e2m1fn_x2, id="fp4"),
        pytest.param(torch.float8_e4m3fn, id="fp8"),
    ],
)
def test_grouped_gemm_dsrelu_wrapper_cache_zero_m_before_compile_full_dynamic_smoke(request, monkeypatch, ab_dtype):
    compile_count, cache_entries = _test_grouped_gemm_dsrelu_wrapper_zero_m_before_compile_cache_behavior(
        request=request,
        monkeypatch=monkeypatch,
        use_full_dynamic=True,
        ab_dtype=ab_dtype,
    )

    assert compile_count == 1
    assert cache_entries == 1


@pytest.mark.L0
@torch_fork_set_rng(seed=7)
def test_grouped_gemm_dsrelu_wrapper_uint8_raw_fp4_smoke(request):
    try:
        from cudnn import grouped_gemm_dsrelu_wrapper_sm100
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = grouped_gemm_dsrelu_init(
        request=request,
        ab_dtype=torch.uint8,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=16,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=True,
        discrete_col_sfd=False,
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
    inputs, _ = allocate_grouped_gemm_dsrelu_tensors(
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

    outputs = grouped_gemm_dsrelu_wrapper_sm100(
        a_tensor=inputs["a_tensor"],
        b_tensor=inputs["b_tensor"],
        c_tensor=inputs["c_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        sfb_tensor=inputs["sfb_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        prob_tensor=inputs["prob_tensor"],
        acc_dtype=cfg["acc_dtype"],
        d_dtype=cfg["d_dtype"],
        cd_major=cfg["cd_major"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sf_vec_size=cfg["sf_vec_size"],
        vector_f32=cfg["vector_f32"],
        m_aligned=cfg["m_aligned"],
        discrete_col_sfd=cfg["discrete_col_sfd"],
        current_stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )

    torch.cuda.synchronize()
    assert torch.isfinite(outputs["d_row_tensor"].float()).all()
    assert torch.isfinite(outputs["dprob_tensor"].float()).all()
    assert torch.count_nonzero(outputs["d_row_tensor"]).item() > 0


def _run_dsrelu_case(case, **wrapper_kwargs):
    """Run the wrapper once over an already-built case, so repeats share identical inputs.

    ``dprob_tensor`` is deliberately not passed: the wrapper then allocates and zeroes a
    fresh one per call, which is what lets two runs be compared against each other.
    """
    from cudnn import grouped_gemm_dsrelu_wrapper_sm100
    from cuda.bindings import driver as cuda

    inputs, cfg = case
    wrapper_kwargs.setdefault("current_stream", cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    return grouped_gemm_dsrelu_wrapper_sm100(
        a_tensor=inputs["a_tensor"],
        b_tensor=inputs["b_tensor"],
        c_tensor=inputs["c_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        sfb_tensor=inputs["sfb_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        prob_tensor=inputs["prob_tensor"],
        # Required whenever the fp8 path allocates sfd_row/sfd_col: the API rejects a
        # descriptor set where these three are not all-None or all-present.
        norm_const_tensor=inputs.get("norm_const_tensor"),
        acc_dtype=cfg["acc_dtype"],
        d_dtype=cfg["d_dtype"],
        cd_major=cfg["cd_major"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sf_vec_size=cfg["sf_vec_size"],
        vector_f32=cfg["vector_f32"],
        m_aligned=cfg["m_aligned"],
        discrete_col_sfd=cfg["discrete_col_sfd"],
        **wrapper_kwargs,
    )


def _build_dsrelu_case(request, ab_dtype, c_dtype, d_dtype, sf_vec_size, sf_dtype, vector_f32, discrete_col_sfd, n_override=None, group_m_override=None):
    cfg = grouped_gemm_dsrelu_init(
        request=request,
        ab_dtype=ab_dtype,
        c_dtype=c_dtype,
        d_dtype=d_dtype,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
        vector_f32=vector_f32,
        discrete_col_sfd=discrete_col_sfd,
        b_major="k",
    )
    if n_override is not None:
        # Everything downstream reads cfg["n"], so overriding it here keeps the allocators,
        # the wrapper call and the reference in agreement.
        cfg["n"] = n_override
    if group_m_override is not None:
        # The default is [256] * l -- every expert the same size, so every expert owns the same
        # number of M-blocks. Anything that assumes a uniform stride between experts survives
        # that; only a ragged distribution tells a real segment mapping from a fake one.
        cfg["group_m_list"] = group_m_override
        cfg["l"] = len(group_m_override)
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
    inputs, _ = allocate_grouped_gemm_dsrelu_tensors(
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
    return inputs, cfg


@contextlib.contextmanager
def _torch_deterministic_algorithms():
    """Scope torch's global determinism flag, restoring whatever the session had.

    warn_only is captured and restored too: hardcoding it on the way out would downgrade a
    session running under strict determinism into warning-only for every later test.
    """
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)


def _assert_dprob_deterministic(case, baseline, deterministic, relaunch, ref_inputs=None, dprob_rtol=1e-4, dprob_atol=1e-4, ref_kwargs=None):
    """The three properties deterministic=True has to hold, shared by dense and discrete.

    ``relaunch`` produces another deterministic run; it is called repeatedly, because a
    reduction-order race is timing-dependent and one repeat proves little.

    ``ref_kwargs`` is forwarded to the final reference check. It exists for the clamped fp8
    caller, whose block-scaled outputs need a wider rtol for a reason that has nothing to do
    with determinism (see _clamp_ref_tolerances); default behaviour is unchanged.
    """
    torch.cuda.synchronize()
    inputs, cfg = case

    # The caller-visible shape is unchanged -- the per-N-tile workspace is internal.
    assert deterministic["dprob_tensor"].shape == baseline["dprob_tensor"].shape
    assert torch.count_nonzero(deterministic["dprob_tensor"]).item() > 0

    # 1. Repeated runs agree to the last bit, not merely to a tolerance.
    assert_bitwise_runs(lambda: (relaunch()["dprob_tensor"],), label="dprob")

    # 2. Reordering a float sum must not change what is being summed: a partial dropped by a
    # gap in the per-subtile slots, or double-counted by a second writer to an N-tile slot,
    # would show up here and nowhere else. The tolerance is not delicate -- both modes sum
    # the same terms, so the honest difference is fp32 reordering (~1e-5 here), while a
    # dropped or duplicated partial moves dprob by tens of percent. It does not transfer across
    # n, though -- dprob sums n terms -- so callers at a larger n pass a looser pair.
    torch.testing.assert_close(deterministic["dprob_tensor"], baseline["dprob_tensor"], rtol=dprob_rtol, atol=dprob_atol)

    # dprob is the only output the flag touches; dA and friends are single-write per element
    # and must come back untouched. d_col only joins that list when the fp8 scale-factor path
    # is active -- otherwise the kernel leaves it at whatever its torch.empty_strided
    # allocation happened to contain, which differs between two independent calls for reasons
    # that have nothing to do with determinism. (The reference check skips it for the same
    # reason: run_grouped_gemm_dsrelu_ref only builds d_col_ref under generate_sfd.)
    unchanged = ["d_row_tensor", "d_srelu_tensor"]
    if deterministic.get("sfd_col_tensor") is not None:
        unchanged.append("d_col_tensor")
    for key in unchanged:
        assert torch.equal(deterministic[key], baseline[key]), f"{key} changed under deterministic=True"

    # 3. The deterministic path clears the same bar as the default one, not merely agrees
    # with it: the reference the non-deterministic wrapper tests run, over the whole backward
    # output set (dprob, dA row/col, d_srelu, amax, scale factors).
    check_ref_grouped_gemm_dsrelu(inputs if ref_inputs is None else ref_inputs, deterministic, cfg, **(ref_kwargs or {}), skip_ref=cfg["skip_ref"])


# n defaults to 512 against an mma_tiler_mn[1] of 256, so grid_n is 2 and the cross-N-tile
# reduction -- level 2 of the fix -- is actually exercised. A config with n <= 256 would
# only cover level 1.
GROUPED_GEMM_DSRELU_DETERMINISTIC_CONFIGS = [
    pytest.param(torch.uint8, torch.bfloat16, torch.bfloat16, 16, torch.float8_e8m0fnu, True, False, id="fp4"),
    pytest.param(torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn, 32, torch.float8_e8m0fnu, False, True, id="fp8"),
]


@pytest.mark.L0
@torch_fork_set_rng(seed=17)
@pytest.mark.parametrize(
    "ab_dtype,c_dtype,d_dtype,sf_vec_size,sf_dtype,vector_f32,discrete_col_sfd",
    GROUPED_GEMM_DSRELU_DETERMINISTIC_CONFIGS,
)
def test_grouped_gemm_dsrelu_deterministic_dprob(request, ab_dtype, c_dtype, d_dtype, sf_vec_size, sf_dtype, vector_f32, discrete_col_sfd):
    """deterministic=True makes dprob bit-exact without changing what it computes."""
    try:
        import cudnn  # noqa: F401
        from cuda.bindings import driver as cuda  # noqa: F401
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    case = _build_dsrelu_case(request, ab_dtype, c_dtype, d_dtype, sf_vec_size, sf_dtype, vector_f32, discrete_col_sfd)

    # Only the baseline probe may skip. Once it has run, the config is supported, so a
    # failure from the deterministic run is a real failure -- catching it here would turn
    # exactly the regression this test exists to find into a green skip.
    try:
        baseline = _run_dsrelu_case(case, deterministic=False)
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    deterministic = _run_dsrelu_case(case, deterministic=True)
    _assert_dprob_deterministic(case, baseline, deterministic, lambda: _run_dsrelu_case(case, deterministic=True))

    # Unset, the flag follows torch -- the path most callers will actually take.
    with _torch_deterministic_algorithms():
        from_torch = _run_dsrelu_case(case)
    torch.cuda.synchronize()
    assert torch.equal(bitwise_bits(from_torch["dprob_tensor"]), bitwise_bits(deterministic["dprob_tensor"]))


# b_major does not reach the dprob path, so the n-major config would only re-prove what
# fp8-k-major already does; the two dtypes are kept because they compile different kernels.
DISCRETE_DETERMINISTIC_CONFIGS = [c for c in DISCRETE_GROUPED_GEMM_DSRELU_SUPPORTED_CONFIGS if c.id != "fp8-n-major"]


@pytest.mark.L0
@torch_fork_set_rng(seed=29)
def test_grouped_gemm_dsrelu_deterministic_dprob_side_stream(request):
    """The slot reduction must be ordered against the kernel on the caller's stream.

    Issued on torch's current stream instead, it would read dprob before the kernel that
    writes it had finished. The race is timing-dependent, so a green run here is not proof
    of correct ordering -- but it does exercise the non-current-stream path, which every
    other test misses by passing torch's own stream.
    """
    try:
        import cudnn  # noqa: F401
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    case = _build_dsrelu_case(request, torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn, 32, torch.float8_e8m0fnu, False, True)
    inputs, cfg = case

    # Deliberately NOT entering torch.cuda.stream(side): the divergence between the wrapper's
    # stream and torch's current stream is the whole point. Inside that context the two would
    # be the same stream and the ordering bug could not appear.
    side = torch.cuda.Stream()
    try:
        on_side = _run_dsrelu_case(case, deterministic=True, current_stream=cuda.CUstream(side.cuda_stream))
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    torch.cuda.synchronize()

    check_ref_grouped_gemm_dsrelu(inputs, on_side, cfg, skip_ref=cfg["skip_ref"])


@pytest.mark.L0
@torch_fork_set_rng(seed=37)
def test_grouped_gemm_dsrelu_deterministic_dprob_side_stream_unordered_init(request):
    """The dprob workspace must be zeroed on the caller's stream, not on torch's.

    The kernel *accumulates* into dprob, so the zeroing has to be visible to it. Allocating
    with torch.zeros puts that memset on torch's current stream, which is unordered against a
    caller-supplied one -- the kernel is then free to atomic-add onto whatever the allocator
    handed back, and deterministic=True silently stops being deterministic.

    Where the plain side-stream test above is only opportunistic, this one forces the failure:
    a same-sized block is poisoned and freed so torch's allocator hands those exact bytes to
    the wrapper, and torch's stream is occupied so a memset queued there lands well after the
    kernel has run. Against the buggy ordering dprob comes back carrying the poison; against
    the fixed ordering it is bit-identical to the same case run on torch's own stream.
    """
    try:
        import cudnn  # noqa: F401
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    # White-box on purpose: the poison block only gets reused if it matches the workspace byte
    # for byte, and the slot count is exactly what this test must not hardcode.
    from cudnn.gemm.cutedsl.grouped.dsrelu.api import _dprob_n_slots

    case = _build_dsrelu_case(request, torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn, 32, torch.float8_e8m0fnu, False, True)
    inputs, cfg = case

    try:
        expected = _run_dsrelu_case(case, deterministic=True)
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    torch.cuda.synchronize()

    valid_m = inputs["a_tensor"].shape[0]
    slots = _dprob_n_slots(cfg["n"], cfg["mma_tiler_mn"], cfg["cluster_shape_mn"], True)
    assert slots > 1, "config must have more than one N-tile for this to exercise the slot workspace"

    side = torch.cuda.Stream()
    # Freed, not held: the bytes stay in the allocator's pool for torch's current stream, so
    # the wrapper's next same-shaped allocation gets them back still carrying the poison.
    poison = torch.full((valid_m, slots, 1), 1e30, dtype=torch.float32, device=inputs["a_tensor"].device)
    del poison
    # ~100 ms of occupancy on torch's stream. A memset mistakenly queued there cannot reach the
    # buffer until after the kernel on `side` has already accumulated into it.
    torch.cuda._sleep(200_000_000)

    on_side = _run_dsrelu_case(case, deterministic=True, current_stream=cuda.CUstream(side.cuda_stream))
    torch.cuda.synchronize()

    assert torch.isfinite(on_side["dprob_tensor"]).all(), "dprob picked up the poisoned block -- workspace zeroed on the wrong stream"
    assert torch.equal(
        bitwise_bits(on_side["dprob_tensor"]), bitwise_bits(expected["dprob_tensor"])
    ), "deterministic dprob differs between torch's stream and a caller-supplied stream"
    check_ref_grouped_gemm_dsrelu(inputs, on_side, cfg, skip_ref=cfg["skip_ref"])


@pytest.mark.L0
@torch_fork_set_rng(seed=41)
def test_grouped_gemm_dsrelu_deterministic_dbias(request):
    """deterministic=True makes dbias bit-exact without changing what it computes.

    dbias contends across M-tiles, not N-tiles like dprob, so it gets its own fp32 workspace
    indexed by absolute M-block plus a per-expert segment sum. The properties it has to hold are
    the same ones dprob does, so this reuses that checklist and adds the dbias-specific parts.

    Run on a ragged expert distribution -- see the comment on the case below; the default one
    cannot fail this test.
    """
    try:
        import cudnn  # noqa: F401
        from cuda.bindings import driver as cuda  # noqa: F401
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    # Ragged on purpose. Measured (job 466159, 16 launches per config): at the default
    # [256] * 4 the NON-deterministic dbias is already bit-stable, so assert_bitwise_runs below
    # would pass whether or not the fix works. At this distribution it varies 15/15, which is
    # what makes the assertion mean something.
    case = _build_dsrelu_case(
        request,
        torch.float8_e4m3fn,
        torch.bfloat16,
        torch.float8_e4m3fn,
        32,
        torch.float8_e8m0fnu,
        False,
        True,
        group_m_override=[256, 512, 256, 1024],
    )
    inputs, cfg = case

    try:
        baseline = _run_dsrelu_case(case, deterministic=False, generate_dbias=True)
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    torch.cuda.synchronize()

    deterministic = _run_dsrelu_case(case, deterministic=True, generate_dbias=True)
    torch.cuda.synchronize()

    # Everything dprob has to hold, dbias holds too -- including that the flag leaves the other
    # outputs untouched -- so the dprob checklist covers it. check_ref now asserts dbias against
    # the reference as well, which is what actually pins the per-expert segment mapping.
    _assert_dprob_deterministic(case, baseline, deterministic, lambda: _run_dsrelu_case(case, deterministic=True, generate_dbias=True))

    # The caller-visible dbias keeps its (expert, n, 1) bf16 shape; the fp32 M-block workspace and
    # the segment sum are internal to the wrapper.
    assert deterministic["dbias_tensor"].dtype == torch.bfloat16
    assert torch.count_nonzero(deterministic["dbias_tensor"]).item() > 0

    # The two paths sum the same terms but not the same way: the default rounds to bf16 on every
    # M-tile, the deterministic one keeps fp32 partials and narrows once, which makes it the more
    # accurate of the pair. So they agree only to bf16 resolution at dbias's magnitude, and the
    # bound is scaled accordingly -- job 459169 failed a flat 2e-2 on a 16.0 difference that is
    # ~2 ULP of bf16 there.
    det_dbias, base_dbias = deterministic["dbias_tensor"].float(), baseline["dbias_tensor"].float()
    torch.testing.assert_close(det_dbias, base_dbias, rtol=5e-2, atol=max(base_dbias.abs().max().item(), 1.0) * 5e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=71)
@pytest.mark.parametrize("deterministic", [False, True], ids=["default", "deterministic"])
def test_grouped_gemm_dsrelu_dbias_cache_keys_on_n(request, deterministic, monkeypatch):
    """Two n values that share a dprob slot count must not share a dbias kernel.

    Under the default full-dynamic key every tensor drops its shape, because the kernel compiles
    with symbolic extents -- but _make_dbias_fake frees only dim 0, so dbias's n stays baked, and
    nothing else in the dense key carries n (b_tensor.shape[2] is l; dprob_n_slots is too coarse,
    n=384 and n=512 both give 2). Before the fix the second call reused the first kernel and
    CuTeDSL rejected it: "Mismatched dbias_tensor.shape[1] ... expected to be 384" (job 469460).
    Pre-existing on the default path, which is why this is parametrized over both modes.
    """
    try:
        import cudnn  # noqa: F401
        from cuda.bindings import driver as cuda  # noqa: F401
        from cudnn.gemm.cutedsl.grouped.dsrelu import api as dsrelu_api
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    monkeypatch.setattr(dsrelu_api, "_cache_of_GroupedGemmDsreluSm100Objects", {})
    args = (torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn, 32, torch.float8_e8m0fnu, False, True)
    for n in (384, 512):
        case = _build_dsrelu_case(request, *args, n_override=n)
        try:
            outputs = _run_dsrelu_case(case, deterministic=deterministic, generate_dbias=True)
        except (ValueError, NotImplementedError) as e:
            pytest.skip(f"Unsupported testcase: {e}")
        torch.cuda.synchronize()
        assert outputs["dbias_tensor"].shape == (case[1]["l"], n, 1)
        check_ref_grouped_gemm_dsrelu(case[0], outputs, case[1], skip_ref=case[1]["skip_ref"])


@pytest.mark.L0
@torch_fork_set_rng(seed=67)
def test_grouped_gemm_dsrelu_deterministic_class_api(request):
    """The class API under determinism: same output args as the default, plus explicit scratch.

    Every other determinism test drives the wrapper, which hides the workspaces. This is the only
    coverage of the contract a direct GroupedGemmDsreluSm100 caller sees -- that dprob and dbias
    keep their shapes and dtypes, that the scratch is a separate argument sized by the public
    helpers, and that the public reductions turn it into the documented outputs.
    """
    try:
        import cudnn  # noqa: F401
        from cuda.bindings import driver as cuda
        from cudnn import GroupedGemmDsreluSm100
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    args = (torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn, 32, torch.float8_e8m0fnu, False, True)
    case = _build_dsrelu_case(request, *args, group_m_override=[256, 512, 256, 1024])
    inputs, cfg = case

    # The wrapper run is the oracle: same inputs, so the class must reproduce it exactly.
    try:
        expected = _run_dsrelu_case(case, deterministic=True, generate_dbias=True)
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    torch.cuda.synchronize()

    valid_m, n, l = inputs["a_tensor"].shape[0], cfg["n"], cfg["l"]
    dev = inputs["a_tensor"].device
    dprob = torch.zeros((valid_m, 1, 1), dtype=torch.float32, device=dev)
    dbias = torch.zeros((l, n, 1), dtype=torch.bfloat16, device=dev)
    dprob_ws = torch.zeros(GroupedGemmDsreluSm100.dprob_workspace_shape(valid_m, n), dtype=torch.float32, device=dev)
    dbias_ws = torch.zeros(GroupedGemmDsreluSm100.dbias_workspace_shape(valid_m, n), dtype=torch.bfloat16, device=dev)

    # The output args are byte-for-byte the ones the non-deterministic path takes.
    assert dprob.shape == expected["dprob_tensor"].shape and dprob.dtype == expected["dprob_tensor"].dtype
    assert dbias.shape == expected["dbias_tensor"].shape and dbias.dtype == expected["dbias_tensor"].dtype

    d_row = torch.empty_strided((valid_m, n, 1), (n, 1, valid_m * n), dtype=cfg["d_dtype"], device=dev)
    d_col = torch.empty_strided((valid_m, n, 1), (n, 1, valid_m * n), dtype=cfg["d_dtype"], device=dev)
    op = GroupedGemmDsreluSm100(
        sample_a=inputs["a_tensor"],
        sample_b=inputs["b_tensor"],
        sample_c=inputs["c_tensor"],
        sample_d_row=d_row,
        sample_d_col=d_col,
        sample_sfa=inputs["sfa_tensor"],
        sample_sfb=inputs["sfb_tensor"],
        sample_padded_offsets=inputs["padded_offsets_tensor"],
        sample_alpha=inputs["alpha_tensor"],
        sample_prob=inputs["prob_tensor"],
        sample_dprob=dprob,
        sample_dbias=dbias,
        sample_dprob_workspace=dprob_ws,
        sample_dbias_workspace=dbias_ws,
        sample_sfd_row=expected["sfd_row_tensor"],
        sample_sfd_col=expected["sfd_col_tensor"],
        sample_norm_const=inputs.get("norm_const_tensor"),
        acc_dtype=cfg["acc_dtype"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sf_vec_size=cfg["sf_vec_size"],
        vector_f32=cfg["vector_f32"],
        m_aligned=cfg["m_aligned"],
        discrete_col_sfd=cfg["discrete_col_sfd"],
        deterministic=True,
    )
    try:
        assert op.check_support()
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    op.compile()

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    op.execute(
        a_tensor=inputs["a_tensor"],
        b_tensor=inputs["b_tensor"],
        c_tensor=inputs["c_tensor"],
        d_row_tensor=d_row,
        d_col_tensor=d_col,
        sfa_tensor=inputs["sfa_tensor"],
        sfb_tensor=inputs["sfb_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        prob_tensor=inputs["prob_tensor"],
        dprob_tensor=dprob,
        dbias_tensor=dbias,
        dprob_workspace_tensor=dprob_ws,
        dbias_workspace_tensor=dbias_ws,
        sfd_row_tensor=expected["sfd_row_tensor"],
        sfd_col_tensor=expected["sfd_col_tensor"],
        norm_const_tensor=inputs.get("norm_const_tensor"),
        current_stream=stream,
    )
    # Both accumulate onto the output, so the zeroed buffers above end up holding the result --
    # no copy_ needed, and the same semantics the kernel has when the flag is off.
    GroupedGemmDsreluSm100.reduce_dprob_workspace(dprob_ws, dprob, stream)
    GroupedGemmDsreluSm100.reduce_dbias_workspace(dbias_ws, inputs["padded_offsets_tensor"], dbias, cfg["mma_tiler_mn"], stream)
    torch.cuda.synchronize()

    assert torch.equal(bitwise_bits(dprob), bitwise_bits(expected["dprob_tensor"])), "class API dprob differs from the wrapper"
    assert torch.equal(bitwise_bits(dbias), bitwise_bits(expected["dbias_tensor"])), "class API dbias differs from the wrapper"


@pytest.mark.L0
@torch_fork_set_rng(seed=59)
def test_grouped_gemm_dsrelu_deterministic_dbias_cache_is_m_dynamic(request, monkeypatch):
    """Deterministic dbias must not add token-count dependence to the compile cache key.

    The workspace has one row per CTA M-tile, so its dim 0 follows valid_m; keyed on the full
    shape it would recompile for most steps of an MoE loop. Asserted as "no more entries than
    the default path produces" rather than "exactly one": measured (job 466362) this config
    already recompiles per valid_m without dbias at all -- elements 28/33/34 of the key are the
    d_col batch stride and the sfd_col shape/strides, all of which carry M. That is pre-existing
    and orthogonal; what this pins is that the flag adds nothing on top.

    Only reachable with CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL=0 -- the default path drops tensor
    shapes from the key entirely, which is why the other dbias tests cannot see this.
    """
    try:
        import cudnn  # noqa: F401
        from cuda.bindings import driver as cuda  # noqa: F401
        from cudnn.gemm.cutedsl.grouped.dsrelu import api as dsrelu_api
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    monkeypatch.setenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "0")
    args = (torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn, 32, torch.float8_e8m0fnu, False, True)

    def entries_for(deterministic):
        monkeypatch.setattr(dsrelu_api, "_cache_of_GroupedGemmDsreluSm100Objects", {})
        for group_m_list in ([256] * 4, [256, 512, 256, 512]):
            case = _build_dsrelu_case(request, *args, group_m_override=group_m_list)
            outputs = _run_dsrelu_case(case, deterministic=deterministic, generate_dbias=True)
            torch.cuda.synchronize()
            check_ref_grouped_gemm_dsrelu(case[0], outputs, case[1], skip_ref=case[1]["skip_ref"])
        return len(dsrelu_api._cache_of_GroupedGemmDsreluSm100Objects)

    try:
        baseline_entries = entries_for(False)
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    assert entries_for(True) == baseline_entries, "deterministic dbias added a token-count-dependent cache key"


@pytest.mark.L0
@torch_fork_set_rng(seed=61)
def test_grouped_gemm_dsrelu_deterministic_dbias_zero_tokens(request):
    """A zero-token call must still return the documented dbias, not the slot workspace.

    valid_m == 0 skips the kernel entirely, so the reduction runs on an empty workspace: the
    one-hot is (expert_cnt, 0) and the matmul has an empty contraction. It has to come back
    (expert_cnt, n, 1) bf16 and zeroed, the same as the non-deterministic path.
    """
    try:
        import cudnn  # noqa: F401
        from cuda.bindings import driver as cuda  # noqa: F401
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    args = (torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn, 32, torch.float8_e8m0fnu, False, True)
    case = _build_dsrelu_case(request, *args)
    inputs, cfg = case
    # Zero every group: valid_m becomes 0 while the descriptors stay well-formed.
    inputs = dict(inputs)
    inputs["padded_offsets_tensor"] = torch.zeros_like(inputs["padded_offsets_tensor"])
    try:
        outputs = _run_dsrelu_case((inputs, cfg), deterministic=True, generate_dbias=True)
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    torch.cuda.synchronize()

    assert outputs["dbias_tensor"].shape == (cfg["l"], cfg["n"], 1)
    assert outputs["dbias_tensor"].dtype == torch.bfloat16
    assert torch.count_nonzero(outputs["dbias_tensor"]).item() == 0
    assert outputs["dprob_tensor"].shape[1:] == (1, 1)


@pytest.mark.L1
@torch_fork_set_rng(seed=53)
def test_grouped_gemm_dsrelu_deterministic_at_scale(request):
    """The one config in this file where the default path is non-deterministic in *both* outputs.

    Measured (job 466159, 16 launches per config): at l=4 / [256] * 4 / n=512 -- what every other
    determinism test here uses -- the non-deterministic dprob AND dbias are both already bit-stable,
    so those tests cannot distinguish a working fix from an invalid one. At l=8 / [1024] * 8 / n=2048
    both vary 15/15. This is the case that actually demonstrates the flag does something.

    L1 rather than L0: it is roughly 16x the work of the smoke configs.
    """
    try:
        import cudnn  # noqa: F401
        from cuda.bindings import driver as cuda  # noqa: F401
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    case = _build_dsrelu_case(
        request,
        torch.float8_e4m3fn,
        torch.bfloat16,
        torch.float8_e4m3fn,
        32,
        torch.float8_e8m0fnu,
        False,
        True,
        n_override=2048,
        group_m_override=[1024] * 8,
    )
    inputs, cfg = case
    try:
        baseline = _run_dsrelu_case(case, deterministic=False, generate_dbias=True)
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    deterministic = _run_dsrelu_case(case, deterministic=True, generate_dbias=True)
    # dprob sums n terms, so fp32 reassociation grows with n: the helper's 1e-4 is calibrated at
    # the default n=512, and 1 element in 8192 lands at 1.8e-4 relative here (job 466172). That is
    # reordering, not a dropped partial -- those move dprob by tens of percent, not by 5e-4.
    _assert_dprob_deterministic(
        case,
        baseline,
        deterministic,
        lambda: _run_dsrelu_case(case, deterministic=True, generate_dbias=True),
        dprob_rtol=1e-3,
        dprob_atol=1e-3,
    )
    check_ref_grouped_gemm_dsrelu(inputs, deterministic, cfg, skip_ref=cfg["skip_ref"])


@pytest.mark.L0
@torch_fork_set_rng(seed=47)
@pytest.mark.parametrize("group_m_list", [[256, 512, 256, 1024], [512, 256, 1024, 256]], ids=["ragged", "ragged-reordered"])
def test_grouped_gemm_dsrelu_deterministic_dbias_ragged_experts(request, group_m_list):
    """Experts of different sizes own different numbers of M-blocks.

    The default config is [256] * 4 -- every expert exactly two M-blocks -- so a segment sum
    that simply assumed a fixed stride between experts would pass every other dbias test here.
    These distributions give 2/4/2/8 and 4/2/8/2 blocks, so a fixed-stride or off-by-one
    mapping lands one expert's rows in another and the reference check fails.
    """
    try:
        import cudnn  # noqa: F401
        from cuda.bindings import driver as cuda  # noqa: F401
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    case = _build_dsrelu_case(
        request, torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn, 32, torch.float8_e8m0fnu, False, True, group_m_override=group_m_list
    )
    inputs, cfg = case
    try:
        outputs = _run_dsrelu_case(case, deterministic=True, generate_dbias=True)
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    torch.cuda.synchronize()

    # Each expert's row must be non-trivial, or a mapping that collapsed everything into one
    # expert could still satisfy the reference check on the rows it happened to fill.
    per_expert_nonzero = (outputs["dbias_tensor"].float().abs().sum(dim=(1, 2)) > 0).tolist()
    assert all(per_expert_nonzero), f"empty dbias rows: {per_expert_nonzero}"
    check_ref_grouped_gemm_dsrelu(inputs, outputs, cfg, skip_ref=cfg["skip_ref"])


@pytest.mark.L0
@torch_fork_set_rng(seed=43)
def test_grouped_gemm_dsrelu_deterministic_dbias_segments(request):
    """The per-expert segment sum must follow the real expert boundaries at a second N.

    A one-hot built off the wrong offsets still produces plausible finite numbers and still
    agrees with the non-deterministic path to a loose tolerance, so only the reference catches
    it -- check_ref_grouped_gemm_dsrelu compares against a dbias summed per expert from the
    unquantized dA. n=768 rather than the default 512, so the segment sum is exercised against
    a workspace whose row stride differs from the one the test above already covers.
    """
    try:
        import cudnn  # noqa: F401
        from cuda.bindings import driver as cuda  # noqa: F401
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    case = _build_dsrelu_case(request, torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn, 32, torch.float8_e8m0fnu, False, True, n_override=768)
    inputs, cfg = case
    try:
        outputs = _run_dsrelu_case(case, deterministic=True, generate_dbias=True)
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    torch.cuda.synchronize()

    assert outputs["dbias_tensor"].shape == (cfg["l"], 768, 1)
    check_ref_grouped_gemm_dsrelu(inputs, outputs, cfg, skip_ref=cfg["skip_ref"])


@pytest.mark.L0
@torch_fork_set_rng(seed=31)
def test_grouped_gemm_dsrelu_deterministic_dprob_slot_count_cache(request):
    """Two N values with different slot counts must not share a compiled kernel.

    Under CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL the cache key drops tensor shapes, but the
    dprob slot extent is static in the compiled descriptor. n=512 and n=768 give 2 and 3
    slots; if the second reuses the first's kernel, its dprob comes back wrong.
    """
    try:
        import cudnn  # noqa: F401
        from cuda.bindings import driver as cuda  # noqa: F401
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    args = (torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn, 32, torch.float8_e8m0fnu, False, True)
    for n in (512, 768):
        case = _build_dsrelu_case(request, *args, n_override=n)
        inputs, cfg = case
        try:
            outputs = _run_dsrelu_case(case, deterministic=True)
        except (ValueError, NotImplementedError) as e:
            pytest.skip(f"Unsupported testcase: {e}")
        torch.cuda.synchronize()
        check_ref_grouped_gemm_dsrelu(inputs, outputs, cfg, skip_ref=cfg["skip_ref"])


@pytest.mark.L0
@torch_fork_set_rng(seed=23)
@pytest.mark.parametrize("ab_dtype,c_dtype,d_dtype,b_major", DISCRETE_DETERMINISTIC_CONFIGS)
def test_grouped_gemm_dsrelu_deterministic_dprob_discrete(request, ab_dtype, c_dtype, d_dtype, b_major):
    """Same guarantee in discrete-weight mode, which builds its cache key on its own branch."""
    try:
        from cudnn import grouped_gemm_dsrelu_wrapper_sm100
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = discrete_grouped_gemm_init(
        request=request,
        ab_dtype=ab_dtype,
        c_dtype=c_dtype,
        d_dtype=d_dtype,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=False,
        discrete_col_sfd=False,
        b_major=b_major,
    )

    inputs = _prepare_discrete_dsrelu_inputs(
        allocate_discrete_input_tensors(
            n=cfg["n"],
            k=cfg["k"],
            num_experts=cfg["l"],
            group_m_list=cfg["group_m_list"],
            ab_dtype=cfg["ab_dtype"],
            sf_dtype=cfg["sf_dtype"],
            sf_vec_size=cfg["sf_vec_size"],
            m_aligned=cfg["m_aligned"],
            b_major=cfg["b_major"],
        )
    )
    inputs, _ = allocate_grouped_gemm_dsrelu_tensors(
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

    def run(**kwargs):
        return grouped_gemm_dsrelu_wrapper_sm100(
            a_tensor=inputs["a_tensor"],
            c_tensor=inputs["c_tensor"],
            sfa_tensor=inputs["sfa_tensor"],
            padded_offsets=inputs["padded_offsets_tensor"],
            alpha_tensor=inputs["alpha_tensor"],
            prob_tensor=inputs["prob_tensor"],
            b_ptrs=inputs["b_ptrs_tensor"],
            sfb_ptrs=inputs["sfb_ptrs_tensor"],
            n=cfg["n"],
            b_dtype=inputs["b_list"][0].dtype,
            b_major=cfg["b_major"],
            norm_const_tensor=inputs.get("norm_const_tensor"),
            acc_dtype=cfg["acc_dtype"],
            d_dtype=cfg["d_dtype"],
            cd_major=cfg["cd_major"],
            mma_tiler_mn=cfg["mma_tiler_mn"],
            cluster_shape_mn=cfg["cluster_shape_mn"],
            sf_vec_size=cfg["sf_vec_size"],
            vector_f32=cfg["vector_f32"],
            m_aligned=cfg["m_aligned"],
            discrete_col_sfd=cfg["discrete_col_sfd"],
            current_stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            **kwargs,
        )

    baseline = run(deterministic=False)
    deterministic = run(deterministic=True)
    _assert_dprob_deterministic(
        (inputs, cfg),
        baseline,
        deterministic,
        lambda: run(deterministic=True),
        ref_inputs=_dense_ref_inputs_from_discrete(inputs),
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=13)
@pytest.mark.parametrize("ab_dtype,c_dtype,d_dtype,b_major", DISCRETE_GROUPED_GEMM_DSRELU_SUPPORTED_CONFIGS)
def test_grouped_gemm_dsrelu_discrete_compile_execute(request, ab_dtype, c_dtype, d_dtype, b_major):
    try:
        from cudnn import GroupedGemmDsreluSm100
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = discrete_grouped_gemm_init(
        request=request,
        ab_dtype=ab_dtype,
        c_dtype=c_dtype,
        d_dtype=d_dtype,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=False,
        discrete_col_sfd=False,
        b_major=b_major,
    )

    inputs = _prepare_discrete_dsrelu_inputs(
        allocate_discrete_input_tensors(
            n=cfg["n"],
            k=cfg["k"],
            num_experts=cfg["l"],
            group_m_list=cfg["group_m_list"],
            ab_dtype=cfg["ab_dtype"],
            sf_dtype=cfg["sf_dtype"],
            sf_vec_size=cfg["sf_vec_size"],
            m_aligned=cfg["m_aligned"],
            b_major=cfg["b_major"],
        )
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

    api = GroupedGemmDsreluSm100(
        sample_a=inputs["a_tensor"],
        sample_c=inputs["c_tensor"],
        sample_d_row=outputs["d_row_tensor"],
        sample_d_col=outputs["d_col_tensor"],
        sample_d_srelu=outputs["d_srelu_tensor"],
        sample_sfa=inputs["sfa_tensor"],
        sample_padded_offsets=inputs["padded_offsets_tensor"],
        sample_alpha=inputs["alpha_tensor"],
        sample_prob=inputs["prob_tensor"],
        sample_dprob=outputs["dprob_tensor"],
        num_experts=cfg["l"],
        b_shape=(cfg["n"], cfg["k"]),
        b_dtype=inputs["b_list"][0].dtype,
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
        b_major=cfg["b_major"],
    )

    try:
        assert api.check_support(), "Unsupported testcase"
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    api.compile()
    api.execute(
        a_tensor=inputs["a_tensor"],
        b_ptrs=inputs["b_ptrs_tensor"],
        sfb_ptrs=inputs["sfb_ptrs_tensor"],
        c_tensor=inputs["c_tensor"],
        d_row_tensor=outputs["d_row_tensor"],
        d_col_tensor=outputs["d_col_tensor"],
        d_srelu_tensor=outputs["d_srelu_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        prob_tensor=inputs["prob_tensor"],
        dprob_tensor=outputs["dprob_tensor"],
        sfd_row_tensor=outputs.get("sfd_row_tensor"),
        sfd_col_tensor=outputs.get("sfd_col_tensor"),
        sfd_col_d_srelu_tensor=outputs.get("sfd_col_d_srelu_tensor"),
        norm_const_tensor=inputs.get("norm_const_tensor"),
        amax_tensor=outputs.get("amax_tensor"),
        current_stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )

    torch.cuda.synchronize()
    check_ref_grouped_gemm_dsrelu(
        _dense_ref_inputs_from_discrete(inputs),
        outputs,
        cfg,
        skip_ref=cfg["skip_ref"],
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=13)
@pytest.mark.parametrize("ab_dtype,c_dtype,d_dtype,b_major", DISCRETE_GROUPED_GEMM_DSRELU_SUPPORTED_CONFIGS)
def test_grouped_gemm_dsrelu_discrete_wrapper(request, ab_dtype, c_dtype, d_dtype, b_major):
    try:
        from cudnn import grouped_gemm_dsrelu_wrapper_sm100
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = discrete_grouped_gemm_init(
        request=request,
        ab_dtype=ab_dtype,
        c_dtype=c_dtype,
        d_dtype=d_dtype,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=False,
        discrete_col_sfd=False,
        b_major=b_major,
    )

    inputs = _prepare_discrete_dsrelu_inputs(
        allocate_discrete_input_tensors(
            n=cfg["n"],
            k=cfg["k"],
            num_experts=cfg["l"],
            group_m_list=cfg["group_m_list"],
            ab_dtype=cfg["ab_dtype"],
            sf_dtype=cfg["sf_dtype"],
            sf_vec_size=cfg["sf_vec_size"],
            m_aligned=cfg["m_aligned"],
            b_major=cfg["b_major"],
        )
    )
    inputs, _ = allocate_grouped_gemm_dsrelu_tensors(
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

    outputs = grouped_gemm_dsrelu_wrapper_sm100(
        a_tensor=inputs["a_tensor"],
        c_tensor=inputs["c_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        prob_tensor=inputs["prob_tensor"],
        b_ptrs=inputs["b_ptrs_tensor"],
        sfb_ptrs=inputs["sfb_ptrs_tensor"],
        n=cfg["n"],
        b_dtype=inputs["b_list"][0].dtype,
        b_major=cfg["b_major"],
        norm_const_tensor=inputs.get("norm_const_tensor"),
        acc_dtype=cfg["acc_dtype"],
        d_dtype=cfg["d_dtype"],
        cd_major=cfg["cd_major"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sf_vec_size=cfg["sf_vec_size"],
        vector_f32=cfg["vector_f32"],
        m_aligned=cfg["m_aligned"],
        discrete_col_sfd=cfg["discrete_col_sfd"],
        current_stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )

    torch.cuda.synchronize()
    wrapper_outputs = {
        "d_row_tensor": outputs["d_row_tensor"],
        "d_col_tensor": outputs["d_col_tensor"],
        "d_srelu_tensor": outputs["d_srelu_tensor"],
        "dprob_tensor": outputs["dprob_tensor"],
        "dbias_tensor": outputs["dbias_tensor"],
        "amax_tensor": outputs["amax_tensor"],
        "sfd_row_tensor": outputs["sfd_row_tensor"],
        "sfd_col_tensor": outputs["sfd_col_tensor"],
        "sfd_col_d_srelu_tensor": outputs["sfd_col_d_srelu_tensor"],
    }
    check_ref_grouped_gemm_dsrelu(
        _dense_ref_inputs_from_discrete(inputs),
        wrapper_outputs,
        cfg,
        skip_ref=cfg["skip_ref"],
    )


"""
GroupedGemmDsrelu API with explicit check_support, compile, and execute paths.
Use this method when running one static configuration for each GroupedGemmDsrelu object.
"""


def _test_grouped_gemm_dsrelu_compile_execute(
    ab_dtype,
    c_dtype,
    d_dtype,
    b_major,
    cd_major,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    vector_f32,
    discrete_col_sfd,
    request,
):
    try:
        from cudnn import GroupedGemmDsreluSm100
        from cuda.bindings import driver as cuda
    except ImportError as e:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = grouped_gemm_dsrelu_init(
        request=request,
        ab_dtype=ab_dtype,
        c_dtype=c_dtype,
        d_dtype=d_dtype,
        cd_major=cd_major,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
        vector_f32=vector_f32,
        discrete_col_sfd=discrete_col_sfd,
        b_major=b_major,
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

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
        assert api.check_support(), "Unsupported testcase"
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    api.compile()
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
        current_stream=stream,
    )

    torch.cuda.synchronize()
    check_ref_grouped_gemm_dsrelu(
        inputs,
        outputs,
        cfg,
        skip_ref=cfg["skip_ref"],
    )


"""
GroupedGemmDsrelu API with grouped_gemm_dsrelu_wrapper:
Use the wrapper to directly call GroupedGemmDsrelu without explicit setup and compilation.
"""


def _test_grouped_gemm_dsrelu_wrapper(
    ab_dtype,
    c_dtype,
    d_dtype,
    b_major,
    cd_major,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    vector_f32,
    discrete_col_sfd,
    request,
):
    try:
        from cudnn import grouped_gemm_dsrelu_wrapper_sm100
        from cuda.bindings import driver as cuda
    except ImportError as e:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = grouped_gemm_dsrelu_init(
        request=request,
        ab_dtype=ab_dtype,
        c_dtype=c_dtype,
        d_dtype=d_dtype,
        cd_major=cd_major,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
        vector_f32=vector_f32,
        discrete_col_sfd=discrete_col_sfd,
        b_major=b_major,
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

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

    inputs, _ = allocate_grouped_gemm_dsrelu_tensors(
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

    try:
        for _ in range(2):  # Run twice to test caching path
            wrapper_outputs = grouped_gemm_dsrelu_wrapper_sm100(
                a_tensor=inputs["a_tensor"],
                b_tensor=inputs["b_tensor"],
                c_tensor=inputs["c_tensor"],
                sfa_tensor=inputs["sfa_tensor"],
                sfb_tensor=inputs["sfb_tensor"],
                padded_offsets=inputs["padded_offsets_tensor"],
                alpha_tensor=inputs["alpha_tensor"],
                prob_tensor=inputs["prob_tensor"],
                norm_const_tensor=inputs.get("norm_const_tensor"),
                acc_dtype=cfg["acc_dtype"],
                d_dtype=cfg["d_dtype"],
                cd_major=cfg["cd_major"],
                mma_tiler_mn=cfg["mma_tiler_mn"],
                cluster_shape_mn=cfg["cluster_shape_mn"],
                sf_vec_size=cfg["sf_vec_size"],
                vector_f32=cfg["vector_f32"],
                m_aligned=cfg["m_aligned"],
                discrete_col_sfd=cfg["discrete_col_sfd"],
                current_stream=stream,
            )
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    torch.cuda.synchronize()
    check_ref_grouped_gemm_dsrelu(
        inputs,
        wrapper_outputs,
        cfg,
        skip_ref=cfg["skip_ref"],
    )


def _test_grouped_gemm_dsrelu_wrapper_dynamic_shape_cache_behavior(
    request,
    monkeypatch,
    use_full_dynamic,
    ab_dtype,
):
    try:
        from cudnn import grouped_gemm_dsrelu_wrapper_sm100
        from cudnn.gemm.cutedsl.grouped.dsrelu import api as grouped_gemm_dsrelu_api
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    if use_full_dynamic:
        monkeypatch.setenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1")
    else:
        monkeypatch.delenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", raising=False)

    grouped_gemm_dsrelu_api._cache_of_GroupedGemmDsreluSm100Objects.clear()

    compile_count = {"value": 0}
    original_compile = grouped_gemm_dsrelu_api.GroupedGemmDsreluSm100.compile

    def counted_compile(self):
        compile_count["value"] += 1
        return original_compile(self)

    monkeypatch.setattr(grouped_gemm_dsrelu_api.GroupedGemmDsreluSm100, "compile", counted_compile)

    d_dtype = torch.float8_e4m3fn if ab_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] else torch.bfloat16

    cfg = grouped_gemm_dsrelu_init(
        request=request,
        ab_dtype=ab_dtype,
        c_dtype=torch.bfloat16,
        d_dtype=d_dtype,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=False,
        discrete_col_sfd=ab_dtype in [torch.float8_e4m3fn, torch.float8_e5m2],
        b_major="k",
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    try:
        for group_m in GROUPED_GEMM_DSRELU_DYNAMIC_SHAPES_M_VALUES:
            group_m_list = [group_m] * cfg["l"]
            inputs = allocate_grouped_gemm_input_tensors(
                n=cfg["n"],
                k=cfg["k"],
                l=cfg["l"],
                group_m_list=group_m_list,
                ab_dtype=cfg["ab_dtype"],
                b_major=cfg["b_major"],
                sf_dtype=cfg["sf_dtype"],
                sf_vec_size=cfg["sf_vec_size"],
                m_aligned=cfg["m_aligned"],
            )

            inputs, _ = allocate_grouped_gemm_dsrelu_tensors(
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

            wrapper_outputs = grouped_gemm_dsrelu_wrapper_sm100(
                a_tensor=inputs["a_tensor"],
                b_tensor=inputs["b_tensor"],
                c_tensor=inputs["c_tensor"],
                sfa_tensor=inputs["sfa_tensor"],
                sfb_tensor=inputs["sfb_tensor"],
                padded_offsets=inputs["padded_offsets_tensor"],
                alpha_tensor=inputs["alpha_tensor"],
                prob_tensor=inputs["prob_tensor"],
                norm_const_tensor=inputs.get("norm_const_tensor"),
                acc_dtype=cfg["acc_dtype"],
                d_dtype=cfg["d_dtype"],
                cd_major=cfg["cd_major"],
                mma_tiler_mn=cfg["mma_tiler_mn"],
                cluster_shape_mn=cfg["cluster_shape_mn"],
                sf_vec_size=cfg["sf_vec_size"],
                vector_f32=cfg["vector_f32"],
                m_aligned=cfg["m_aligned"],
                discrete_col_sfd=cfg["discrete_col_sfd"],
                current_stream=stream,
            )
            torch.cuda.synchronize()
            # check_ref_grouped_gemm_dsrelu(
            #     inputs,
            #     wrapper_outputs,
            #     cfg,
            #     skip_ref=cfg["skip_ref"],
            # )
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    finally:
        cache_entries = len(grouped_gemm_dsrelu_api._cache_of_GroupedGemmDsreluSm100Objects)
        grouped_gemm_dsrelu_api._cache_of_GroupedGemmDsreluSm100Objects.clear()

    return compile_count["value"], cache_entries


def _test_grouped_gemm_dsrelu_wrapper_zero_m_after_compile_cache_behavior(
    request,
    monkeypatch,
    use_full_dynamic,
    ab_dtype,
):
    return _test_grouped_gemm_dsrelu_wrapper_zero_m_cache_behavior(
        request=request,
        monkeypatch=monkeypatch,
        use_full_dynamic=use_full_dynamic,
        ab_dtype=ab_dtype,
        group_m_values=[512, 0],
    )


def _test_grouped_gemm_dsrelu_wrapper_zero_m_before_compile_cache_behavior(
    request,
    monkeypatch,
    use_full_dynamic,
    ab_dtype,
):
    return _test_grouped_gemm_dsrelu_wrapper_zero_m_cache_behavior(
        request=request,
        monkeypatch=monkeypatch,
        use_full_dynamic=use_full_dynamic,
        ab_dtype=ab_dtype,
        group_m_values=[0, 512],
    )


def _test_grouped_gemm_dsrelu_wrapper_zero_m_cache_behavior(
    request,
    monkeypatch,
    use_full_dynamic,
    ab_dtype,
    group_m_values,
):
    try:
        from cudnn import grouped_gemm_dsrelu_wrapper_sm100
        from cudnn.gemm.cutedsl.grouped.dsrelu import api as grouped_gemm_dsrelu_api
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    if use_full_dynamic:
        monkeypatch.setenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1")
    else:
        monkeypatch.delenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", raising=False)

    grouped_gemm_dsrelu_api._cache_of_GroupedGemmDsreluSm100Objects.clear()

    compile_count = {"value": 0}
    original_compile = grouped_gemm_dsrelu_api.GroupedGemmDsreluSm100.compile

    def counted_compile(self):
        compile_count["value"] += 1
        return original_compile(self)

    monkeypatch.setattr(grouped_gemm_dsrelu_api.GroupedGemmDsreluSm100, "compile", counted_compile)

    d_dtype = torch.float8_e4m3fn if ab_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] else torch.bfloat16

    cfg = grouped_gemm_dsrelu_init(
        request=request,
        ab_dtype=ab_dtype,
        c_dtype=torch.bfloat16,
        d_dtype=d_dtype,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=False,
        discrete_col_sfd=ab_dtype in [torch.float8_e4m3fn, torch.float8_e5m2],
        b_major="k",
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    try:
        for group_m in group_m_values:
            group_m_list = [group_m] * cfg["l"]
            inputs = allocate_grouped_gemm_input_tensors(
                n=cfg["n"],
                k=cfg["k"],
                l=cfg["l"],
                group_m_list=group_m_list,
                ab_dtype=cfg["ab_dtype"],
                b_major=cfg["b_major"],
                sf_dtype=cfg["sf_dtype"],
                sf_vec_size=cfg["sf_vec_size"],
                m_aligned=cfg["m_aligned"],
            )

            inputs, _ = allocate_grouped_gemm_dsrelu_tensors(
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

            grouped_gemm_dsrelu_wrapper_sm100(
                a_tensor=inputs["a_tensor"],
                b_tensor=inputs["b_tensor"],
                c_tensor=inputs["c_tensor"],
                sfa_tensor=inputs["sfa_tensor"],
                sfb_tensor=inputs["sfb_tensor"],
                padded_offsets=inputs["padded_offsets_tensor"],
                alpha_tensor=inputs["alpha_tensor"],
                prob_tensor=inputs["prob_tensor"],
                norm_const_tensor=inputs.get("norm_const_tensor"),
                acc_dtype=cfg["acc_dtype"],
                d_dtype=cfg["d_dtype"],
                cd_major=cfg["cd_major"],
                mma_tiler_mn=cfg["mma_tiler_mn"],
                cluster_shape_mn=cfg["cluster_shape_mn"],
                sf_vec_size=cfg["sf_vec_size"],
                vector_f32=cfg["vector_f32"],
                m_aligned=cfg["m_aligned"],
                discrete_col_sfd=cfg["discrete_col_sfd"],
                current_stream=stream,
            )
            torch.cuda.synchronize()
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    finally:
        cache_entries = len(grouped_gemm_dsrelu_api._cache_of_GroupedGemmDsreluSm100Objects)
        grouped_gemm_dsrelu_api._cache_of_GroupedGemmDsreluSm100Objects.clear()

    return compile_count["value"], cache_entries


# =============================================================================
# Soft-clamped dSReLU (tanh_clamp_scale)
# =============================================================================
#
# tanh_clamp_scale=s replaces relu(C) with c = s*tanh(relu(C)/s) throughout the backward:
#     dgrad = g * 2*c*(1-t^2) * w      (was  g * 2*relu(C) * w)
#     dprob = sum_n c^2 * g            (was  sum_n relu(C)^2 * g)
#     d_srelu regen = c^2 * w          (was  relu(C)^2 * w)
# Unclamped is the s -> infinity limit. See the srelu test file for why the cache key and
# the vector_f32 axis are correctness requirements rather than coverage nice-to-haves.
#
# The backward has three consumers of the activation, and the third (d_srelu regen, used
# only under FP8 activation recompute) is the one a partial implementation drops -- so
# generate_d_srelu is exercised explicitly below rather than left to the default.

GROUPED_GEMM_DSRELU_CLAMP_CONFIGS = [
    pytest.param(torch.uint8, torch.bfloat16, torch.bfloat16, 16, torch.float8_e8m0fnu, True, False, id="fp4-vector_f32"),
    pytest.param(torch.uint8, torch.bfloat16, torch.bfloat16, 16, torch.float8_e8m0fnu, False, False, id="fp4-scalar_f32"),
    pytest.param(torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn, 32, torch.float8_e8m0fnu, True, True, id="fp8-vector_f32"),
    pytest.param(torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn, 32, torch.float8_e8m0fnu, False, True, id="fp8-scalar_f32"),
]

GROUPED_GEMM_DSRELU_CLAMP_SCALES = [
    pytest.param(None, id="unclamped"),
    pytest.param(16.0, id="clamp16"),
    pytest.param(32.0, id="clamp32"),
]


# Block-scaled (MXFP8) outputs vs an independently quantized reference, once the kernel is
# perturbed at all -- measured on GB300 (sm_103) at the clamp's 7.7e-6 max tanh perturbation
# (repos/sqrelu_tanh/tools/):
#   * the e8m0 scale factors do not flip at all, even though ~25% of the 32-element block
#     amaxes in this harness sit exactly ON an exponent boundary;
#   * d_row/d_col show rare one-ULP straddles -- 12 of 524288 elements (2.3e-5), max ratio
#     exactly one e4m3 ULP (1.125), none beyond;
#   * dprob (plain fp32, not block-scaled) shows no outliers.
# A boundary straddle is a one-ULP event produced by an arbitrarily small cause, which no
# blanket rtol below 12.5% can distinguish from a real error. The clamped fp8 cases therefore
# allow one ULP of relative slack; the tight elementwise check of the epilogue math lives on
# the bf16-output fp4 configs above.
_FP8_E4M3_ULP_RTOL = 2.0**-3


def _is_block_scaled_output(d_dtype, discrete_col_sfd):
    return d_dtype is torch.float8_e4m3fn and discrete_col_sfd


def _clamp_ref_tolerances(d_dtype, tanh_clamp_scale):
    if d_dtype is torch.float8_e4m3fn and tanh_clamp_scale is not None:
        return {"rtol": _FP8_E4M3_ULP_RTOL + 1e-2}
    return {}


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("use_dsrelu_reuse", [False, True], ids=["no_reuse", "reuse"])
@pytest.mark.parametrize("tanh_clamp_scale", GROUPED_GEMM_DSRELU_CLAMP_SCALES)
@pytest.mark.parametrize(
    "ab_dtype,c_dtype,d_dtype,sf_vec_size,sf_dtype,vector_f32,discrete_col_sfd",
    GROUPED_GEMM_DSRELU_CLAMP_CONFIGS,
)
def test_grouped_gemm_dsrelu_wrapper_tanh_clamp_scale(
    request, ab_dtype, c_dtype, d_dtype, sf_vec_size, sf_dtype, vector_f32, discrete_col_sfd, tanh_clamp_scale, use_dsrelu_reuse
):
    """All three backward consumers agree with the torch reference, on both f32 variants.

    use_dsrelu_reuse is crossed in because it routes dprob and d_srelu through a different
    pair of helpers (compute_relu2 + compute_dprob_from_relu2 + scale_srelu_from_relu2)
    than the default path -- the clamped value has to flow correctly through both.
    """
    try:
        import cudnn  # noqa: F401
        from cuda.bindings import driver as cuda  # noqa: F401
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    case = _build_dsrelu_case(request, ab_dtype, c_dtype, d_dtype, sf_vec_size, sf_dtype, vector_f32, discrete_col_sfd)
    inputs, cfg = case
    cfg["tanh_clamp_scale"] = tanh_clamp_scale

    try:
        outputs = _run_dsrelu_case(case, tanh_clamp_scale=tanh_clamp_scale, use_dsrelu_reuse=use_dsrelu_reuse)
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    torch.cuda.synchronize()
    # The wrapper always regenerates d_srelu; assert it so a future change that makes it
    # conditional cannot silently drop the third backward consumer from this test.
    assert outputs.get("d_srelu_tensor") is not None, "wrapper must return the regenerated activation"

    if tanh_clamp_scale is not None and _is_block_scaled_output(d_dtype, discrete_col_sfd):
        # See the comment on _is_block_scaled_output: the code-for-code comparison of the
        # MXFP8 outputs is not a valid oracle for a perturbed kernel. dprob is, and it is the
        # consumer that would catch a wrong c^2, so assert it at full tightness.
        if not cfg["skip_ref"]:
            ref_tensors = run_grouped_gemm_dsrelu_ref(
                a_ref=inputs["a_ref"],
                b_ref=inputs["b_ref"],
                c_ref=inputs["c_ref"],
                sfa_ref=inputs["sfa_ref"],
                sfb_ref=inputs["sfb_ref"],
                alpha_tensor=inputs["alpha_tensor"],
                prob_tensor=inputs["prob_tensor"],
                aligned_group_m_list=inputs["aligned_group_m_list"],
                valid_m=inputs["valid_m"],
                generate_dbias=outputs.get("dbias_tensor") is not None,
                generate_amax=outputs.get("amax_tensor") is not None,
                generate_sfd=outputs.get("sfd_row_tensor") is not None and outputs.get("sfd_col_tensor") is not None,
                norm_const_tensor=inputs.get("norm_const_tensor"),
                c_dtype=cfg["c_dtype"],
                d_dtype=cfg["d_dtype"],
                sf_vec_size=cfg["sf_vec_size"],
                sf_dtype=cfg["sf_dtype"],
                tanh_clamp_scale=tanh_clamp_scale,
            )
            torch.testing.assert_close(outputs["dprob_tensor"].float(), ref_tensors["dprob_ref"].float(), atol=1e-1, rtol=1e-2)
            assert torch.isfinite(outputs["d_srelu_tensor"].float()).all()
            assert torch.isfinite(outputs["d_row_tensor"].float()).all()
    else:
        check_ref_grouped_gemm_dsrelu(inputs, outputs, cfg, skip_ref=cfg["skip_ref"])


@pytest.mark.L0
@torch_fork_set_rng(seed=19)
@pytest.mark.parametrize(
    "ab_dtype,c_dtype,d_dtype,sf_vec_size,sf_dtype,vector_f32,discrete_col_sfd",
    # fp4 only: the fp8 entry has a block-scaled D whose codes are not a valid oracle for a
    # perturbed kernel (see _is_block_scaled_output), and _assert_dprob_deterministic ends in a
    # full reference check. Determinism itself is value-agnostic, so the fp4 entry covers it.
    [c for c in GROUPED_GEMM_DSRELU_DETERMINISTIC_CONFIGS if c.id != "fp8"],
)
def test_grouped_gemm_dsrelu_deterministic_dprob_clamped(request, ab_dtype, c_dtype, d_dtype, sf_vec_size, sf_dtype, vector_f32, discrete_col_sfd):
    """deterministic=True still holds when dprob sums c^2 instead of relu(C)^2.

    The slot-parking reduction is value-agnostic, so this is expected to pass for free --
    which is exactly why it is worth asserting: 'expected to work by construction' is the
    class of claim that silently stops being true.
    """
    try:
        import cudnn  # noqa: F401
        from cuda.bindings import driver as cuda  # noqa: F401
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    tanh_clamp_scale = 16.0
    case = _build_dsrelu_case(request, ab_dtype, c_dtype, d_dtype, sf_vec_size, sf_dtype, vector_f32, discrete_col_sfd)
    case[1]["tanh_clamp_scale"] = tanh_clamp_scale

    try:
        baseline = _run_dsrelu_case(case, deterministic=False, tanh_clamp_scale=tanh_clamp_scale)
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    deterministic = _run_dsrelu_case(case, deterministic=True, tanh_clamp_scale=tanh_clamp_scale)
    _assert_dprob_deterministic(
        case,
        baseline,
        deterministic,
        lambda: _run_dsrelu_case(case, deterministic=True, tanh_clamp_scale=tanh_clamp_scale),
        ref_kwargs=_clamp_ref_tolerances(d_dtype, tanh_clamp_scale),
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=31)
@pytest.mark.parametrize(
    "ab_dtype,c_dtype,d_dtype,sf_vec_size,sf_dtype,vector_f32,discrete_col_sfd",
    GROUPED_GEMM_DSRELU_DETERMINISTIC_CONFIGS,
)
def test_grouped_gemm_dsrelu_tanh_clamp_scale_none_is_bit_identical(request, ab_dtype, c_dtype, d_dtype, sf_vec_size, sf_dtype, vector_f32, discrete_col_sfd):
    """tanh_clamp_scale=None must be BIT-identical to not passing the argument at all.

    The backward counterpart of the srelu test of the same name, and the direct evidence
    for the additive-only claim on this kernel: the clamped branch is const_expr'd out, so
    an existing caller should get the same traced program and the same bits. Covers the
    fp4 and fp8 configurations, which compile different kernels.
    """
    try:
        import cudnn  # noqa: F401
        from cuda.bindings import driver as cuda  # noqa: F401
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    case = _build_dsrelu_case(request, ab_dtype, c_dtype, d_dtype, sf_vec_size, sf_dtype, vector_f32, discrete_col_sfd)

    try:
        omitted = _run_dsrelu_case(case)  # exactly the pre-change call signature
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    explicit_none = _run_dsrelu_case(case, tanh_clamp_scale=None)
    torch.cuda.synchronize()

    # d_col is only written under the fp8 scale-factor path; elsewhere it keeps its
    # allocation contents, which differ between two independent calls for reasons that
    # have nothing to do with this flag (same caveat as the determinism test above).
    compared = ["d_row_tensor", "d_srelu_tensor", "dprob_tensor"]
    if omitted.get("sfd_col_tensor") is not None:
        compared += ["d_col_tensor", "sfd_row_tensor", "sfd_col_tensor"]
    if omitted.get("amax_tensor") is not None:
        compared.append("amax_tensor")

    for key in compared:
        a, b = omitted[key], explicit_none[key]
        if a is None and b is None:
            continue
        assert a is not None and b is not None, f"{key} missing from one of the two runs"
        assert torch.equal(a, b), f"{key} differs between omitted and tanh_clamp_scale=None"


@pytest.mark.L0
@torch_fork_set_rng(seed=37)
@pytest.mark.parametrize("vector_f32", [False, True], ids=["scalar_f32", "vector_f32"])
def test_grouped_gemm_dsrelu_tanh_clamp_scale_saturated_dgrad_is_zero(request, vector_f32):
    """Deep in the saturated tail the row gradient must be EXACTLY zero.

    dgrad carries the factor 1 - t^2, which goes to zero as t -> 1, so for relu(C) >> s the
    row output is 0 with no tolerance argument needed -- the one assertion in this file
    sensitive to the sign of 1 - t^2 (a t above 1 from tanh.approx would come back small,
    non-zero and sign-flipped, invisible to every fp8-tolerance test). The kernel caps t at
    1; this asserts the cap holds.
    """
    try:
        import cudnn  # noqa: F401
        from cuda.bindings import driver as cuda  # noqa: F401
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    # d_dtype bf16 (the fp4 config): 0.0 is exactly representable and there is no output
    # quantization to blur an exact-zero check.
    case = _build_dsrelu_case(request, torch.uint8, torch.bfloat16, torch.bfloat16, 16, torch.float8_e8m0fnu, vector_f32, False)
    inputs, cfg = case

    # Drive every element far past saturation. 512 is exact in bf16, so c_tensor and the
    # fp32 c_ref agree exactly and relu(C)/s is ~1e3 even at the largest scale tested.
    saturated = 512.0
    inputs["c_tensor"].fill_(saturated)
    inputs["c_ref"].fill_(saturated)

    tanh_clamp_scale = 0.5
    try:
        outputs = _run_dsrelu_case(case, tanh_clamp_scale=tanh_clamp_scale)
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    torch.cuda.synchronize()

    d_row = outputs["d_row_tensor"].float()
    assert torch.equal(d_row, torch.zeros_like(d_row)), (
        "saturated dgrad is not exactly zero: max |d_row| = "
        f"{d_row.abs().max().item():.6g}, min = {d_row.min().item():.6g}. "
        "A negative value here means 1 - t^2 went negative, i.e. tanh returned t > 1."
    )

    # dprob keeps the c^2 term, which saturates to s^2 rather than to zero, so it is a live
    # control: if the whole epilogue had simply produced zeros the assertion above would be
    # vacuous.
    dprob = outputs["dprob_tensor"].float()
    assert torch.count_nonzero(dprob).item() > 0, "dprob is all zero -- the epilogue produced nothing"


@pytest.mark.L0
@torch_fork_set_rng(seed=23)
def test_grouped_gemm_dsrelu_tanh_clamp_scale_cache_key(request, monkeypatch):
    """Two clamp scales in one process must compile two kernels, not reuse the first.

    ``s`` is folded at trace time, so a cache key missing it returns the s=16 kernel for an
    s=32 call -- wrong numbers, no error. This is the backward-side twin of the srelu test
    of the same name; both dense and discrete keys are covered because the wrapper picks
    between them on the weight mode, and this config is the dense one.
    """
    try:
        import cudnn  # noqa: F401
        from cudnn.gemm.cutedsl.grouped.dsrelu import api as dsrelu_api
        from cuda.bindings import driver as cuda  # noqa: F401
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    monkeypatch.setattr(dsrelu_api, "_cache_of_GroupedGemmDsreluSm100Objects", {})

    compile_count = {"value": 0}
    original_compile = dsrelu_api.GroupedGemmDsreluSm100.compile

    def counted_compile(self):
        compile_count["value"] += 1
        return original_compile(self)

    monkeypatch.setattr(dsrelu_api.GroupedGemmDsreluSm100, "compile", counted_compile)

    args = (torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn, 32, torch.float8_e8m0fnu, False, True)
    case = _build_dsrelu_case(request, *args)
    inputs, cfg = case

    results = {}
    for tanh_clamp_scale in (16.0, 32.0):
        try:
            results[tanh_clamp_scale] = _run_dsrelu_case(case, tanh_clamp_scale=tanh_clamp_scale)
        except (ValueError, NotImplementedError) as e:
            pytest.skip(f"Unsupported testcase: {e}")
        torch.cuda.synchronize()

    assert compile_count["value"] == 2, f"expected one compile per tanh_clamp_scale, got {compile_count['value']}"

    # This config has an MXFP8 block-scaled D, so only dprob is a valid elementwise oracle
    # (see _is_block_scaled_output). That is enough here: the point of this test is the cache
    # key, and dprob moving with s is exactly what a stale kernel would fail to do.
    if not cfg["skip_ref"]:
        for tanh_clamp_scale, outputs in results.items():
            ref_tensors = run_grouped_gemm_dsrelu_ref(
                a_ref=inputs["a_ref"],
                b_ref=inputs["b_ref"],
                c_ref=inputs["c_ref"],
                sfa_ref=inputs["sfa_ref"],
                sfb_ref=inputs["sfb_ref"],
                alpha_tensor=inputs["alpha_tensor"],
                prob_tensor=inputs["prob_tensor"],
                aligned_group_m_list=inputs["aligned_group_m_list"],
                valid_m=inputs["valid_m"],
                generate_dbias=outputs.get("dbias_tensor") is not None,
                generate_amax=outputs.get("amax_tensor") is not None,
                generate_sfd=outputs.get("sfd_row_tensor") is not None and outputs.get("sfd_col_tensor") is not None,
                norm_const_tensor=inputs.get("norm_const_tensor"),
                c_dtype=cfg["c_dtype"],
                d_dtype=cfg["d_dtype"],
                sf_vec_size=cfg["sf_vec_size"],
                sf_dtype=cfg["sf_dtype"],
                tanh_clamp_scale=tanh_clamp_scale,
            )
            torch.testing.assert_close(outputs["dprob_tensor"].float(), ref_tensors["dprob_ref"].float(), atol=1e-1, rtol=1e-2)

    # A kernel that ignored tanh_clamp_scale would pass the count check and, if the reference
    # shared the bug, the reference check too. Different s must move every consumer.
    if not cfg["skip_ref"]:
        assert not torch.equal(results[16.0]["d_row_tensor"], results[32.0]["d_row_tensor"])
        assert not torch.equal(results[16.0]["dprob_tensor"], results[32.0]["dprob_tensor"])
        assert not torch.equal(results[16.0]["d_srelu_tensor"], results[32.0]["d_srelu_tensor"])


@pytest.mark.L0
@torch_fork_set_rng(seed=29)
@pytest.mark.parametrize("c_dtype", [torch.bfloat16, torch.float32], ids=["c_bf16", "c_fp32"])
def test_grouped_gemm_dsrelu_tanh_clamp_scale_regen_matches_forward(request, c_dtype):
    """The backward's d_srelu regen must reproduce the forward kernel's own output.

    Under FP8 activation recompute the fc2 input is regenerated by the backward rather than
    saved, so forward output and backward regen have to be the same function. They are two
    separate kernels with two separate copies of the tanh expression, and nothing else in the
    suite compares them to each other -- each is only ever checked against the torch oracle.

    Sensitivity, stated rather than assumed: the forward applies the activation to its fp32
    accumulator while the backward applies it to the SAVED C, so the two disagree by the C
    round-trip no matter how well the tanh forms match. That floor is not guessed here, it is
    computed from the inputs (``floor`` below) and used as the tolerance. With c_dtype=bf16
    the floor is ~1e-2 relative, which is coarser than a pure fastmath-tanh discrepancy
    (~1e-3), so THAT case cannot detect a subtly different tanh form -- it catches gross
    mismatches (a swapped s vs 1/s, a missing square, prob folded in twice). The c_fp32 case
    is the sharp one: C then stores the accumulator exactly, the floor collapses, and the two
    tanh expressions are compared against each other with nothing in between.
    """
    try:
        import cudnn  # noqa: F401
        from cudnn import grouped_gemm_srelu_wrapper_sm100
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    tanh_clamp_scale = 16.0
    # bf16 D so the comparison is not floored by fp8 output quantization instead.
    args = (torch.uint8, c_dtype, torch.bfloat16, 16, torch.float8_e8m0fnu, True, False)
    case = _build_dsrelu_case(request, *args)
    inputs, cfg = case
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # 1. Forward over this case's A/B/SF/prob: gives both the stored C and the activation
    #    the forward actually produced from its fp32 accumulator.
    try:
        fwd = grouped_gemm_srelu_wrapper_sm100(
            a_tensor=inputs["a_tensor"],
            b_tensor=inputs["b_tensor"],
            sfa_tensor=inputs["sfa_tensor"],
            sfb_tensor=inputs["sfb_tensor"],
            padded_offsets=inputs["padded_offsets_tensor"],
            alpha_tensor=inputs["alpha_tensor"],
            norm_const_tensor=inputs.get("norm_const_tensor"),
            prob_tensor=inputs["prob_tensor"],
            acc_dtype=cfg["acc_dtype"],
            c_dtype=cfg["c_dtype"],
            d_dtype=cfg["d_dtype"],
            cd_major=cfg["cd_major"],
            mma_tiler_mn=cfg["mma_tiler_mn"],
            cluster_shape_mn=cfg["cluster_shape_mn"],
            sf_vec_size=cfg["sf_vec_size"],
            vector_f32=cfg["vector_f32"],
            m_aligned=cfg["m_aligned"],
            discrete_col_sfd=cfg["discrete_col_sfd"],
            tanh_clamp_scale=tanh_clamp_scale,
            current_stream=stream,
        )
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    torch.cuda.synchronize()

    # 2. Backward fed the forward's own C, regenerating the activation. The forward returns
    #    only the valid_m rows while the backward's C carries the padded tensor_m, so copy
    #    into the case's existing buffer rather than substituting a differently shaped one.
    valid_m = inputs["valid_m"]
    bwd_inputs = dict(inputs)
    bwd_inputs["c_tensor"] = inputs["c_tensor"].clone()
    bwd_inputs["c_tensor"][:valid_m].copy_(fwd["c_tensor"])
    try:
        bwd = _run_dsrelu_case((bwd_inputs, cfg), tanh_clamp_scale=tanh_clamp_scale)
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    torch.cuda.synchronize()

    d_fwd = fwd["d_tensor"].float()[:valid_m]
    d_regen = bwd["d_srelu_tensor"].float()[:valid_m]
    assert d_fwd.shape == d_regen.shape, f"{d_fwd.shape} vs {d_regen.shape}"

    # 3. The C round-trip floor, computed from this run's own C rather than assumed.
    c_saved = fwd["c_tensor"].float()[:valid_m]
    prob = inputs["prob_tensor"].float().expand(-1, cfg["n"], -1)[:valid_m]

    def _act(x):
        t = torch.tanh(torch.relu(x) / tanh_clamp_scale)
        return (tanh_clamp_scale * t) ** 2 * prob

    # The forward applied the activation to its fp32 accumulator; the backward applied it to
    # c_saved, which is that accumulator rounded to c_dtype. The unknown is the rounding, and
    # it is bounded by one ulp -- so propagate one ulp of C through the activation and take
    # that as the floor. For c_dtype=fp32 the ulp is 2^-24 and the floor collapses, which is
    # what makes that parametrization the sharp one.
    c_ulp_rel = 2.0**-8 if c_dtype is torch.bfloat16 else 2.0**-24
    c_ulp = c_saved.abs() * c_ulp_rel
    floor = torch.maximum((_act(c_saved + c_ulp) - _act(c_saved)).abs(), (_act(c_saved - c_ulp) - _act(c_saved)).abs()).max().item()

    # Both sides are stored in d_dtype (bf16 here) and the two kernels multiply prob in a
    # different order, so allow a couple of ulps of the output on top.
    d_ulp = d_fwd.abs().max().item() * 2.0**-8
    atol = floor + 2 * d_ulp
    gap = (d_fwd - d_regen).abs().max().item()
    assert gap <= atol, f"forward output and backward d_srelu regen disagree by {gap:.4g}, above the computed floor {atol:.4g}"


@pytest.mark.L0
def test_grouped_gemm_dsrelu_tanh_clamp_scale_validation(request):
    """tanh_clamp_scale must be finite and positive; the API rejects the rest at construction."""
    try:
        import cudnn  # noqa: F401
        from cuda.bindings import driver as cuda  # noqa: F401
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    args = (torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn, 32, torch.float8_e8m0fnu, False, True)
    case = _build_dsrelu_case(request, *args)

    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="tanh_clamp_scale"):
            _run_dsrelu_case(case, tanh_clamp_scale=bad)


@pytest.mark.L0
@torch_fork_set_rng(seed=101)
@pytest.mark.parametrize("tanh_clamp_scale", [None, 16.0, 32.0], ids=["unclamped", "clamp16", "clamp32"])
def test_clamped_srelu_backward_is_the_forward_derivative(tanh_clamp_scale):
    """The backward formulas really are the derivative of the forward one -- checked by autograd.

    Closes a loop nothing else in this suite closes. Every other test compares a KERNEL against
    a torch REFERENCE, but the forward reference (test_grouped_gemm_srelu_utils) and the backward
    reference (test_grouped_gemm_dsrelu_utils) are two halves of the same hand-derived math:

        forward   y      = (s*tanh(relu(x)/s))^2 * w
        backward  dy/dx  = g * 2c(1-t^2) * w
                  dy/dw  = sum_n c^2 * g

    If that derivative were wrong, the kernels and the references would be wrong TOGETHER and
    every kernel-vs-reference test would still pass. So differentiate the forward expression
    with autograd and compare against the closed forms the backward reference and the kernel
    both use. Pure torch, fp64, no GPU kernel involved -- this validates the math, and the rest
    of the file validates that the kernels implement it.

    The unclamped case is kept as a control: it must reduce to d/dx relu(x)^2 = 2*relu(x).
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, n = 64, 48
    # fp64 so the comparison is limited by the formulas, not by rounding. Scaled to straddle
    # the clamp: relu(x)/s spans roughly [0, 4] at s=16, so both the near-linear and the
    # saturated regimes are exercised in one tensor.
    x = (torch.randn(m, n, dtype=torch.float64, device=dev) * 40.0).requires_grad_(True)
    w = torch.randn(m, 1, dtype=torch.float64, device=dev).requires_grad_(True)
    g = torch.randn(m, n, dtype=torch.float64, device=dev)

    def forward(xx, ww):
        """Exactly what run_grouped_gemm_srelu_ref computes, clamped or not."""
        if tanh_clamp_scale is None:
            return torch.relu(xx) ** 2 * ww
        t = torch.tanh(torch.relu(xx) / tanh_clamp_scale)
        return (tanh_clamp_scale * t) ** 2 * ww

    (forward(x, w) * g).sum().backward()

    with torch.no_grad():
        if tanh_clamp_scale is None:
            # The s -> infinity limit: dgrad reduces to 2*relu(x), dprob to sum relu(x)^2 * g.
            closed_dx = 2 * torch.relu(x) * g * w
            closed_dw = (torch.relu(x) ** 2 * g).sum(dim=-1, keepdim=True)
        else:
            t = torch.tanh(torch.relu(x) / tanh_clamp_scale)
            c = tanh_clamp_scale * t
            closed_dx = 2 * c * (1 - t**2) * g * w
            closed_dw = ((c**2) * g).sum(dim=-1, keepdim=True)

    torch.testing.assert_close(x.grad, closed_dx, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(w.grad, closed_dw, rtol=1e-10, atol=1e-10)

    # Non-vacuous on both sides of the clamp: without saturated elements this would only be
    # testing the near-linear regime, where clamped and unclamped agree anyway.
    if tanh_clamp_scale is not None:
        ratio = torch.relu(x) / tanh_clamp_scale
        assert (ratio > 3.0).any(), "no saturated element -- the clamped branch is untested"
        assert (ratio < 0.5).any() and (ratio > 0).any(), "no near-linear element"
