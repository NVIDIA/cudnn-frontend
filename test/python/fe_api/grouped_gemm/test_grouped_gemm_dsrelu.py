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
        current_stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        **wrapper_kwargs,
    )


def _build_dsrelu_case(request, ab_dtype, c_dtype, d_dtype, sf_vec_size, sf_dtype, vector_f32, discrete_col_sfd):
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
    """Scope torch's global determinism flag, restoring whatever the session had."""
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=True)


def _assert_dprob_deterministic(case, baseline, deterministic, relaunch, ref_inputs=None):
    """The three properties deterministic=True has to hold, shared by dense and discrete.

    ``relaunch`` produces another deterministic run; it is called repeatedly, because a
    reduction-order race is timing-dependent and one repeat proves little.
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
    # would show up here and nowhere else.
    scale = baseline["dprob_tensor"].abs().max().item()
    torch.testing.assert_close(deterministic["dprob_tensor"], baseline["dprob_tensor"], rtol=1e-4, atol=1e-4 * max(scale, 1.0))

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
    check_ref_grouped_gemm_dsrelu(inputs if ref_inputs is None else ref_inputs, deterministic, cfg, skip_ref=cfg["skip_ref"])


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

    try:
        baseline = _run_dsrelu_case(case, deterministic=False)
        deterministic = _run_dsrelu_case(case, deterministic=True)
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
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
