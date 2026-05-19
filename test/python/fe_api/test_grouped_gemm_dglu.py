"""
Tests for Unified Grouped GEMM dGLU Backward Kernel (SM100+)

Tests the GroupedGemmDgluSm100 API which supports both dense (contiguous)
and discrete weight modes, with dSwiGLU and dGeGLU activations.
"""

import torch
import pytest
from test_utils import torch_fork_set_rng
from fe_api.test_fe_api_utils import DYNAMIC_SHAPES_M_VALUES
from fe_api.test_grouped_gemm_swiglu_utils import (
    grouped_gemm_swiglu_init,
    allocate_grouped_gemm_input_tensors as allocate_grouped_gemm_input_tensors_base,
)
from fe_api.test_grouped_gemm_dswiglu_utils import (
    with_grouped_gemm_dswiglu_params_fp4,
    with_grouped_gemm_dswiglu_params_fp8,
    with_grouped_gemm_dswiglu_params_dbias_fp4,
    with_grouped_gemm_dswiglu_params_dbias_fp8,
    allocate_grouped_gemm_dswiglu_tensors,
    check_ref_grouped_gemm_dswiglu,
)
from fe_api.test_discrete_grouped_gemm_dswiglu_utils import (
    discrete_dswiglu_init,
    allocate_discrete_dswiglu_input_tensors,
    allocate_discrete_dswiglu_output_tensors,
    check_ref_discrete_dswiglu,
)

with_scheduler_modes = pytest.mark.parametrize(
    "use_dynamic_sched",
    [False, True],
    ids=["static_sched", "dynamic_sched"],
)


def allocate_grouped_gemm_input_tensors(*args, **kwargs):
    """Restore the upstream dGLU test-input range for backward kernels."""

    tensors = allocate_grouped_gemm_input_tensors_base(*args, **kwargs)

    alpha_tensor = tensors["alpha_tensor"]
    tensors["alpha_tensor"] = torch.randint(1, 2, alpha_tensor.shape, dtype=torch.float32, device=alpha_tensor.device)

    beta_tensor = tensors["beta_tensor"]
    tensors["beta_tensor"] = torch.randint(1, 2, beta_tensor.shape, dtype=torch.float32, device=beta_tensor.device)

    prob_tensor = tensors["prob_tensor"]
    tensors["prob_tensor"] = torch.randint(1, 2, prob_tensor.shape, dtype=torch.float32, device=prob_tensor.device)

    return tensors


def _apply_grouped_gemm_cfg_overrides(cfg, cfg_overrides=None):
    if cfg_overrides is None:
        return cfg

    cfg = dict(cfg)
    cfg.update(cfg_overrides)
    if "group_m_list" in cfg_overrides:
        cfg["group_m_list"] = list(cfg["group_m_list"])
        cfg["l"] = len(cfg["group_m_list"])
    return cfg


# ---------------------------------------------------------------------------
#  Dense mode: Class API
# ---------------------------------------------------------------------------


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_scheduler_modes
@with_grouped_gemm_dswiglu_params_fp4
def test_grouped_gemm_dglu_dense_compile_execute_fp4(
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
    use_dynamic_sched,
    request,
):
    _test_grouped_gemm_dglu_dense_compile_execute(
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
        use_dynamic_sched=use_dynamic_sched,
        request=request,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_scheduler_modes
@with_grouped_gemm_dswiglu_params_fp8
def test_grouped_gemm_dglu_dense_compile_execute_fp8(
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
    use_dynamic_sched,
    request,
):
    _test_grouped_gemm_dglu_dense_compile_execute(
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
        use_dynamic_sched=use_dynamic_sched,
        request=request,
    )


# ---------------------------------------------------------------------------
#  Dense mode: Wrapper API
# ---------------------------------------------------------------------------


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_scheduler_modes
@with_grouped_gemm_dswiglu_params_fp4
def test_grouped_gemm_dglu_dense_wrapper_fp4(
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
    use_dynamic_sched,
    request,
):
    _test_grouped_gemm_dglu_dense_wrapper(
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
        use_dynamic_sched=use_dynamic_sched,
        request=request,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_scheduler_modes
@with_grouped_gemm_dswiglu_params_fp8
def test_grouped_gemm_dglu_dense_wrapper_fp8(
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
    use_dynamic_sched,
    request,
):
    _test_grouped_gemm_dglu_dense_wrapper(
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
        use_dynamic_sched=use_dynamic_sched,
        request=request,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_dglu_dense_compile_execute_rectangular_zero_prob(request):
    def input_mutator(inputs, _cfg):
        inputs["prob_tensor"].zero_()
        inputs["alpha_tensor"].copy_(torch.tensor([1.0, -1.25, 0.75], dtype=torch.float32, device=inputs["alpha_tensor"].device))
        inputs["beta_tensor"].copy_(torch.tensor([0.5, -1.0, 1.5], dtype=torch.float32, device=inputs["beta_tensor"].device))

    inputs, outputs, _ = _test_grouped_gemm_dglu_dense_compile_execute(
        ab_dtype=torch.float4_e2m1fn_x2,
        c_dtype=torch.bfloat16,
        d_dtype=torch.float32,
        b_major="k",
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(1, 1),
        sf_vec_size=16,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=False,
        discrete_col_sfd=False,
        request=request,
        cfg_overrides={
            "n": 192,
            "k": 320,
            "group_m_list": [96, 320, 128],
        },
        input_mutator=input_mutator,
    )

    assert torch.count_nonzero(outputs["d_row_tensor"][: inputs["valid_m"]]).item() == 0
    assert torch.count_nonzero(outputs["dprob_tensor"][: inputs["valid_m"]]).item() > 0


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_grouped_gemm_dswiglu_params_dbias_fp4
def test_grouped_gemm_dglu_dense_compile_execute_with_dbias_fp4(
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
    _test_grouped_gemm_dglu_dense_compile_execute(
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
        generate_dbias=True,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_grouped_gemm_dswiglu_params_dbias_fp8
def test_grouped_gemm_dglu_dense_compile_execute_with_dbias_fp8(
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
    _test_grouped_gemm_dglu_dense_compile_execute(
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
        generate_dbias=True,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_grouped_gemm_dswiglu_params_dbias_fp4
def test_grouped_gemm_dglu_dense_wrapper_with_dbias_fp4(
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
    _test_grouped_gemm_dglu_dense_wrapper(
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
        generate_dbias=True,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_grouped_gemm_dswiglu_params_dbias_fp8
def test_grouped_gemm_dglu_dense_wrapper_with_dbias_fp8(
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
    _test_grouped_gemm_dglu_dense_wrapper(
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
        generate_dbias=True,
    )


# ---------------------------------------------------------------------------
#  Impl: Dense Class API
# ---------------------------------------------------------------------------


def _test_grouped_gemm_dglu_dense_compile_execute(
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
    cfg_overrides=None,
    input_mutator=None,
    generate_dbias=False,
    use_dynamic_sched=False,
):
    try:
        from cudnn import GroupedGemmDgluSm100
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("cudnn optional dependencies not installed")

    cfg = grouped_gemm_swiglu_init(
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
    cfg = _apply_grouped_gemm_cfg_overrides(cfg, cfg_overrides)

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

    inputs, outputs = allocate_grouped_gemm_dswiglu_tensors(
        tensor_m=inputs["tensor_m"],
        n=cfg["n"],
        l=cfg["l"],
        ab_dtype=cfg["ab_dtype"],
        c_dtype=cfg["c_dtype"],
        d_dtype=cfg["d_dtype"],
        cd_major=cfg["cd_major"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        generate_dbias=generate_dbias,
        input_tensors=inputs,
    )

    if input_mutator is not None:
        input_mutator(inputs, cfg)

    # Use the new unified dGLU API in dense mode
    api = GroupedGemmDgluSm100(
        sample_a=inputs["a_tensor"],
        sample_c=inputs["c_tensor"],
        sample_d_row=outputs["d_row_tensor"],
        sample_d_col=outputs["d_col_tensor"],
        sample_sfa=inputs["sfa_tensor"],
        sample_padded_offsets=inputs["padded_offsets_tensor"],
        sample_alpha=inputs["alpha_tensor"],
        sample_beta=inputs["beta_tensor"],
        sample_prob=inputs["prob_tensor"],
        sample_dprob=outputs["dprob_tensor"],
        sample_dbias=outputs.get("dbias_tensor"),
        # Dense mode:
        sample_b=inputs["b_tensor"],
        sample_sfb=inputs["sfb_tensor"],
        # Optional:
        sample_sfd_row=outputs.get("sfd_row_tensor"),
        sample_sfd_col=outputs.get("sfd_col_tensor"),
        sample_amax=outputs.get("amax_tensor"),
        sample_norm_const=inputs.get("norm_const_tensor"),
        # Configuration:
        acc_dtype=cfg["acc_dtype"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sf_vec_size=cfg["sf_vec_size"],
        vector_f32=cfg["vector_f32"],
        m_aligned=cfg["m_aligned"],
        discrete_col_sfd=cfg["discrete_col_sfd"],
        act_func="dswiglu",
        use_dynamic_sched=use_dynamic_sched,
    )

    try:
        assert api.check_support(), "Unsupported testcase"
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    api.compile()
    api.execute(
        a_tensor=inputs["a_tensor"],
        c_tensor=inputs["c_tensor"],
        d_row_tensor=outputs["d_row_tensor"],
        d_col_tensor=outputs["d_col_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        beta_tensor=inputs["beta_tensor"],
        prob_tensor=inputs["prob_tensor"],
        dprob_tensor=outputs["dprob_tensor"],
        dbias_tensor=outputs.get("dbias_tensor"),
        b_tensor=inputs["b_tensor"],
        sfb_tensor=inputs["sfb_tensor"],
        sfd_row_tensor=outputs.get("sfd_row_tensor"),
        sfd_col_tensor=outputs.get("sfd_col_tensor"),
        amax_tensor=outputs.get("amax_tensor"),
        norm_const_tensor=inputs.get("norm_const_tensor"),
        current_stream=stream,
    )

    torch.cuda.synchronize()
    check_ref_grouped_gemm_dswiglu(inputs, outputs, cfg, skip_ref=cfg["skip_ref"])
    return inputs, outputs, cfg


# ---------------------------------------------------------------------------
#  Impl: Dense Wrapper API
# ---------------------------------------------------------------------------


def _test_grouped_gemm_dglu_dense_wrapper(
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
    cfg_overrides=None,
    input_mutator=None,
    generate_dbias=False,
    use_dynamic_sched=False,
):
    try:
        from cudnn import grouped_gemm_dglu_wrapper_sm100
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("cudnn optional dependencies not installed")

    cfg = grouped_gemm_swiglu_init(
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
    cfg = _apply_grouped_gemm_cfg_overrides(cfg, cfg_overrides)

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

    inputs, outputs = allocate_grouped_gemm_dswiglu_tensors(
        tensor_m=inputs["tensor_m"],
        n=cfg["n"],
        l=cfg["l"],
        ab_dtype=cfg["ab_dtype"],
        c_dtype=cfg["c_dtype"],
        d_dtype=cfg["d_dtype"],
        cd_major=cfg["cd_major"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        generate_dbias=generate_dbias,
        input_tensors=inputs,
    )

    if input_mutator is not None:
        input_mutator(inputs, cfg)

    try:
        for _ in range(2):  # Run twice to test caching path
            outputs["dprob_tensor"].zero_()
            wrapper_outputs = grouped_gemm_dglu_wrapper_sm100(
                a_tensor=inputs["a_tensor"],
                c_tensor=inputs["c_tensor"],
                sfa_tensor=inputs["sfa_tensor"],
                padded_offsets=inputs["padded_offsets_tensor"],
                alpha_tensor=inputs["alpha_tensor"],
                beta_tensor=inputs["beta_tensor"],
                prob_tensor=inputs["prob_tensor"],
                dprob_tensor=outputs["dprob_tensor"],
                generate_dbias=generate_dbias,
                # Dense mode:
                b_tensor=inputs["b_tensor"],
                sfb_tensor=inputs["sfb_tensor"],
                # Common:
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
                act_func="dswiglu",
                use_dynamic_sched=use_dynamic_sched,
                current_stream=stream,
            )
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    torch.cuda.synchronize()
    check_ref_grouped_gemm_dswiglu(inputs, wrapper_outputs, cfg, skip_ref=cfg["skip_ref"])
    return inputs, wrapper_outputs, cfg


@pytest.mark.L0
@torch_fork_set_rng(seed=2)
@pytest.mark.parametrize(
    "ab_dtype",
    [
        pytest.param(torch.float4_e2m1fn_x2, id="fp4"),
        pytest.param(torch.float8_e4m3fn, id="fp8"),
    ],
)
def test_grouped_gemm_dglu_dense_wrapper_cache_partial_dynamic_smoke(request, monkeypatch, ab_dtype):
    compile_count, cache_entries = _test_grouped_gemm_dglu_dense_wrapper_dynamic_m_cache_behavior(
        request=request,
        monkeypatch=monkeypatch,
        use_full_dynamic=False,
        ab_dtype=ab_dtype,
    )

    assert compile_count == 1
    assert cache_entries == 1


@pytest.mark.L0
@torch_fork_set_rng(seed=3)
@pytest.mark.parametrize("ab_dtype", [pytest.param(torch.float4_e2m1fn_x2, id="fp4")])
def test_grouped_gemm_dglu_dense_wrapper_cache_full_dynamic_smoke(request, monkeypatch, ab_dtype):
    compile_count, cache_entries = _test_grouped_gemm_dglu_dense_wrapper_dynamic_nk_cache_behavior(
        request=request,
        monkeypatch=monkeypatch,
        ab_dtype=ab_dtype,
    )

    assert compile_count == 1
    assert cache_entries == 1


def _test_grouped_gemm_dglu_dense_wrapper_dynamic_m_cache_behavior(request, monkeypatch, use_full_dynamic, ab_dtype):
    try:
        from cudnn import grouped_gemm_dglu_wrapper_sm100
        from cudnn.grouped_gemm.grouped_gemm_dglu import api as grouped_gemm_dglu_api
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    if use_full_dynamic:
        monkeypatch.setenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1")
    else:
        monkeypatch.delenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", raising=False)

    grouped_gemm_dglu_api._cache_of_GroupedGemmDgluSm100Objects.clear()

    compile_count = {"value": 0}

    def counted_compile(self):
        compile_count["value"] += 1

    monkeypatch.setattr(grouped_gemm_dglu_api.GroupedGemmDgluSm100, "check_support", lambda self: True)
    monkeypatch.setattr(grouped_gemm_dglu_api.GroupedGemmDgluSm100, "compile", counted_compile)
    monkeypatch.setattr(grouped_gemm_dglu_api.GroupedGemmDgluSm100, "execute", lambda self, **kwargs: None)

    d_dtype = torch.float8_e4m3fn if ab_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] else torch.bfloat16
    cfg = grouped_gemm_swiglu_init(
        request=request,
        ab_dtype=ab_dtype,
        c_dtype=torch.bfloat16,
        d_dtype=d_dtype,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32 if ab_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] else 16,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=False,
        discrete_col_sfd=False,
        b_major="k",
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    try:
        for group_m in DYNAMIC_SHAPES_M_VALUES:
            inputs = allocate_grouped_gemm_input_tensors(
                n=cfg["n"],
                k=cfg["k"],
                l=cfg["l"],
                group_m_list=[group_m] * cfg["l"],
                ab_dtype=cfg["ab_dtype"],
                b_major=cfg["b_major"],
                sf_dtype=cfg["sf_dtype"],
                sf_vec_size=cfg["sf_vec_size"],
                m_aligned=cfg["m_aligned"],
            )
            inputs, outputs = allocate_grouped_gemm_dswiglu_tensors(
                tensor_m=inputs["tensor_m"],
                n=cfg["n"],
                l=cfg["l"],
                ab_dtype=cfg["ab_dtype"],
                c_dtype=cfg["c_dtype"],
                d_dtype=cfg["d_dtype"],
                cd_major=cfg["cd_major"],
                sf_dtype=cfg["sf_dtype"],
                sf_vec_size=cfg["sf_vec_size"],
                generate_dbias=False,
                input_tensors=inputs,
            )

            grouped_gemm_dglu_wrapper_sm100(
                a_tensor=inputs["a_tensor"],
                c_tensor=inputs["c_tensor"],
                sfa_tensor=inputs["sfa_tensor"],
                padded_offsets=inputs["padded_offsets_tensor"],
                alpha_tensor=inputs["alpha_tensor"],
                beta_tensor=inputs["beta_tensor"],
                prob_tensor=inputs["prob_tensor"],
                dprob_tensor=outputs["dprob_tensor"],
                b_tensor=inputs["b_tensor"],
                sfb_tensor=inputs["sfb_tensor"],
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
                act_func="dswiglu",
                current_stream=stream,
            )
    finally:
        cache_entries = len(grouped_gemm_dglu_api._cache_of_GroupedGemmDgluSm100Objects)
        grouped_gemm_dglu_api._cache_of_GroupedGemmDgluSm100Objects.clear()

    return compile_count["value"], cache_entries


def _test_grouped_gemm_dglu_dense_wrapper_dynamic_nk_cache_behavior(request, monkeypatch, ab_dtype):
    try:
        from cudnn import grouped_gemm_dglu_wrapper_sm100
        from cudnn.grouped_gemm.grouped_gemm_dglu import api as grouped_gemm_dglu_api
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    monkeypatch.setenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1")
    grouped_gemm_dglu_api._cache_of_GroupedGemmDgluSm100Objects.clear()

    compile_count = {"value": 0}

    def counted_compile(self):
        compile_count["value"] += 1

    monkeypatch.setattr(grouped_gemm_dglu_api.GroupedGemmDgluSm100, "check_support", lambda self: True)
    monkeypatch.setattr(grouped_gemm_dglu_api.GroupedGemmDgluSm100, "compile", counted_compile)
    monkeypatch.setattr(grouped_gemm_dglu_api.GroupedGemmDgluSm100, "execute", lambda self, **kwargs: None)

    cfg = grouped_gemm_swiglu_init(
        request=request,
        ab_dtype=ab_dtype,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=16,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=False,
        discrete_col_sfd=False,
        b_major="k",
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    try:
        for n, k in [(cfg["n"], cfg["k"]), (cfg["n"] + 256, cfg["k"] + 128)]:
            inputs = allocate_grouped_gemm_input_tensors(
                n=n,
                k=k,
                l=cfg["l"],
                group_m_list=cfg["group_m_list"],
                ab_dtype=cfg["ab_dtype"],
                b_major=cfg["b_major"],
                sf_dtype=cfg["sf_dtype"],
                sf_vec_size=cfg["sf_vec_size"],
                m_aligned=cfg["m_aligned"],
            )
            inputs, outputs = allocate_grouped_gemm_dswiglu_tensors(
                tensor_m=inputs["tensor_m"],
                n=n,
                l=cfg["l"],
                ab_dtype=cfg["ab_dtype"],
                c_dtype=cfg["c_dtype"],
                d_dtype=cfg["d_dtype"],
                cd_major=cfg["cd_major"],
                sf_dtype=cfg["sf_dtype"],
                sf_vec_size=cfg["sf_vec_size"],
                generate_dbias=False,
                input_tensors=inputs,
            )

            grouped_gemm_dglu_wrapper_sm100(
                a_tensor=inputs["a_tensor"],
                c_tensor=inputs["c_tensor"],
                sfa_tensor=inputs["sfa_tensor"],
                padded_offsets=inputs["padded_offsets_tensor"],
                alpha_tensor=inputs["alpha_tensor"],
                beta_tensor=inputs["beta_tensor"],
                prob_tensor=inputs["prob_tensor"],
                dprob_tensor=outputs["dprob_tensor"],
                b_tensor=inputs["b_tensor"],
                sfb_tensor=inputs["sfb_tensor"],
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
                act_func="dswiglu",
                current_stream=stream,
            )
    finally:
        cache_entries = len(grouped_gemm_dglu_api._cache_of_GroupedGemmDgluSm100Objects)
        grouped_gemm_dglu_api._cache_of_GroupedGemmDgluSm100Objects.clear()

    return compile_count["value"], cache_entries


# ---------------------------------------------------------------------------
#  Discrete mode: Class API
# ---------------------------------------------------------------------------


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_scheduler_modes
@pytest.mark.parametrize("act_func", ["dswiglu", "dgeglu"])
def test_grouped_gemm_dglu_discrete_compile_execute_fp4(act_func, use_dynamic_sched, request):
    _test_grouped_gemm_dglu_discrete_compile_execute(
        ab_dtype=torch.float4_e2m1fn_x2,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=False,
        discrete_col_sfd=False,
        act_func=act_func,
        use_dynamic_sched=use_dynamic_sched,
        request=request,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_scheduler_modes
@pytest.mark.parametrize("act_func", ["dswiglu", "dgeglu"])
@pytest.mark.parametrize("b_major", ["k", "n"])
def test_grouped_gemm_dglu_discrete_compile_execute_fp8(act_func, b_major, use_dynamic_sched, request):
    _test_grouped_gemm_dglu_discrete_compile_execute(
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
        discrete_col_sfd=False,
        act_func=act_func,
        use_dynamic_sched=use_dynamic_sched,
        request=request,
        b_major=b_major,
    )


# ---------------------------------------------------------------------------
#  Discrete mode: Wrapper API
# ---------------------------------------------------------------------------


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_scheduler_modes
@pytest.mark.parametrize("act_func", ["dswiglu", "dgeglu"])
def test_grouped_gemm_dglu_discrete_wrapper_fp4(act_func, use_dynamic_sched, request):
    _test_grouped_gemm_dglu_discrete_wrapper(
        ab_dtype=torch.float4_e2m1fn_x2,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=False,
        discrete_col_sfd=False,
        act_func=act_func,
        use_dynamic_sched=use_dynamic_sched,
        request=request,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_scheduler_modes
@pytest.mark.parametrize("act_func", ["dswiglu", "dgeglu"])
@pytest.mark.parametrize("b_major", ["k", "n"])
def test_grouped_gemm_dglu_discrete_wrapper_fp8(act_func, b_major, use_dynamic_sched, request):
    _test_grouped_gemm_dglu_discrete_wrapper(
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
        discrete_col_sfd=False,
        act_func=act_func,
        use_dynamic_sched=use_dynamic_sched,
        request=request,
        b_major=b_major,
    )


def _test_grouped_gemm_dglu_discrete_compile_execute(
    ab_dtype,
    c_dtype,
    d_dtype,
    cd_major,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    vector_f32,
    discrete_col_sfd,
    act_func,
    request,
    b_major="k",
    generate_dbias=False,
    use_dynamic_sched=False,
):
    try:
        from cudnn import GroupedGemmDgluSm100
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("cudnn optional dependencies not installed")

    cfg = discrete_dswiglu_init(
        request,
        ab_dtype,
        c_dtype,
        d_dtype,
        cd_major,
        acc_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        sf_vec_size,
        sf_dtype,
        vector_f32,
        discrete_col_sfd,
        act_func,
        b_major=b_major,
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    inputs = allocate_discrete_dswiglu_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        num_experts=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        c_dtype=cfg["c_dtype"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
        b_major=cfg["b_major"],
    )

    outputs = allocate_discrete_dswiglu_output_tensors(
        tensor_m=inputs["tensor_m"],
        n=cfg["n"],
        num_experts=cfg["l"],
        ab_dtype=cfg["ab_dtype"],
        d_dtype=cfg["d_dtype"],
        cd_major=cfg["cd_major"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        generate_dbias=generate_dbias,
    )

    api = GroupedGemmDgluSm100(
        sample_a=inputs["a_tensor"],
        sample_c=inputs["c_tensor"],
        sample_d_row=outputs["d_row_tensor"],
        sample_d_col=outputs["d_col_tensor"],
        sample_sfa=inputs["sfa_tensor"],
        sample_padded_offsets=inputs["padded_offsets_tensor"],
        sample_alpha=inputs["alpha_tensor"],
        sample_beta=inputs["beta_tensor"],
        sample_prob=inputs["prob_tensor"],
        sample_dprob=inputs["dprob_tensor"],
        sample_dbias=outputs.get("dbias_tensor"),
        num_experts=len(inputs["b_list"]),
        b_shape=(cfg["n"], cfg["k"]),
        b_dtype=inputs["b_list"][0].dtype,
        sample_sfd_row=outputs.get("sfd_row_tensor"),
        sample_sfd_col=outputs.get("sfd_col_tensor"),
        sample_amax=outputs.get("amax_tensor"),
        sample_norm_const=inputs.get("norm_const_tensor"),
        acc_dtype=cfg["acc_dtype"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sf_vec_size=cfg["sf_vec_size"],
        vector_f32=cfg["vector_f32"],
        m_aligned=cfg["m_aligned"],
        discrete_col_sfd=cfg["discrete_col_sfd"],
        act_func=cfg["act_func"],
        b_major=cfg["b_major"],
        use_dynamic_sched=use_dynamic_sched,
    )

    try:
        assert api.check_support(), "Unsupported testcase"
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    api.compile()
    api.execute(
        a_tensor=inputs["a_tensor"],
        c_tensor=inputs["c_tensor"],
        d_row_tensor=outputs["d_row_tensor"],
        d_col_tensor=outputs["d_col_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        beta_tensor=inputs["beta_tensor"],
        prob_tensor=inputs["prob_tensor"],
        dprob_tensor=inputs["dprob_tensor"],
        dbias_tensor=outputs.get("dbias_tensor"),
        b_ptrs=inputs["b_ptrs_tensor"],
        sfb_ptrs=inputs["sfb_ptrs_tensor"],
        sfd_row_tensor=outputs.get("sfd_row_tensor"),
        sfd_col_tensor=outputs.get("sfd_col_tensor"),
        amax_tensor=outputs.get("amax_tensor"),
        norm_const_tensor=inputs.get("norm_const_tensor"),
        current_stream=stream,
    )

    torch.cuda.synchronize()
    check_ref_discrete_dswiglu(inputs, outputs, cfg, skip_ref=cfg["skip_ref"])


def _test_grouped_gemm_dglu_discrete_wrapper(
    ab_dtype,
    c_dtype,
    d_dtype,
    cd_major,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    vector_f32,
    discrete_col_sfd,
    act_func,
    request,
    b_major="k",
    generate_dbias=False,
    use_dynamic_sched=False,
):
    try:
        from cudnn import grouped_gemm_dglu_wrapper_sm100
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("cudnn optional dependencies not installed")

    cfg = discrete_dswiglu_init(
        request,
        ab_dtype,
        c_dtype,
        d_dtype,
        cd_major,
        acc_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        sf_vec_size,
        sf_dtype,
        vector_f32,
        discrete_col_sfd,
        act_func,
        b_major=b_major,
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    inputs = allocate_discrete_dswiglu_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        num_experts=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        c_dtype=cfg["c_dtype"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
        b_major=cfg["b_major"],
    )

    try:
        for _ in range(2):  # Run twice to test caching path
            inputs["dprob_tensor"].zero_()
            outputs = grouped_gemm_dglu_wrapper_sm100(
                a_tensor=inputs["a_tensor"],
                c_tensor=inputs["c_tensor"],
                sfa_tensor=inputs["sfa_tensor"],
                padded_offsets=inputs["padded_offsets_tensor"],
                alpha_tensor=inputs["alpha_tensor"],
                beta_tensor=inputs["beta_tensor"],
                prob_tensor=inputs["prob_tensor"],
                dprob_tensor=inputs["dprob_tensor"],
                generate_dbias=generate_dbias,
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
                act_func=cfg["act_func"],
                use_dynamic_sched=use_dynamic_sched,
                current_stream=stream,
            )
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    torch.cuda.synchronize()
    check_ref_discrete_dswiglu(inputs, outputs, cfg, skip_ref=cfg["skip_ref"])


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_dglu_discrete_compile_execute_with_dbias(request):
    _test_grouped_gemm_dglu_discrete_compile_execute(
        ab_dtype=torch.float4_e2m1fn_x2,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=False,
        discrete_col_sfd=False,
        act_func="dswiglu",
        request=request,
        generate_dbias=True,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_dglu_discrete_wrapper_with_dbias(request):
    _test_grouped_gemm_dglu_discrete_wrapper(
        ab_dtype=torch.float4_e2m1fn_x2,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=False,
        discrete_col_sfd=False,
        act_func="dswiglu",
        request=request,
        generate_dbias=True,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=4)
def test_grouped_gemm_dglu_discrete_wrapper_cache_dynamic_m_smoke(request, monkeypatch):
    compile_count, cache_entries = _test_grouped_gemm_dglu_discrete_wrapper_dynamic_m_cache_behavior(
        request=request,
        monkeypatch=monkeypatch,
    )

    assert compile_count == 1
    assert cache_entries == 1


def _test_grouped_gemm_dglu_discrete_wrapper_dynamic_m_cache_behavior(request, monkeypatch):
    try:
        from cudnn import grouped_gemm_dglu_wrapper_sm100
        from cudnn.grouped_gemm.grouped_gemm_dglu import api as grouped_gemm_dglu_api
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    grouped_gemm_dglu_api._cache_of_GroupedGemmDgluSm100Objects.clear()

    compile_count = {"value": 0}
    original_compile = grouped_gemm_dglu_api.GroupedGemmDgluSm100.compile

    def counted_compile(self):
        compile_count["value"] += 1
        return original_compile(self)

    monkeypatch.setattr(grouped_gemm_dglu_api.GroupedGemmDgluSm100, "compile", counted_compile)

    cfg = discrete_dswiglu_init(
        request=request,
        ab_dtype=torch.float4_e2m1fn_x2,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=False,
        discrete_col_sfd=False,
        act_func="dswiglu",
        b_major="k",
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    try:
        for group_m in DYNAMIC_SHAPES_M_VALUES:
            inputs = allocate_discrete_dswiglu_input_tensors(
                n=cfg["n"],
                k=cfg["k"],
                num_experts=cfg["l"],
                group_m_list=[group_m] * cfg["l"],
                ab_dtype=cfg["ab_dtype"],
                c_dtype=cfg["c_dtype"],
                sf_dtype=cfg["sf_dtype"],
                sf_vec_size=cfg["sf_vec_size"],
                m_aligned=cfg["m_aligned"],
                b_major=cfg["b_major"],
            )
            inputs["dprob_tensor"].zero_()

            grouped_gemm_dglu_wrapper_sm100(
                a_tensor=inputs["a_tensor"],
                c_tensor=inputs["c_tensor"],
                sfa_tensor=inputs["sfa_tensor"],
                padded_offsets=inputs["padded_offsets_tensor"],
                alpha_tensor=inputs["alpha_tensor"],
                beta_tensor=inputs["beta_tensor"],
                prob_tensor=inputs["prob_tensor"],
                dprob_tensor=inputs["dprob_tensor"],
                b_ptrs=inputs["b_ptrs_tensor"],
                sfb_ptrs=inputs["sfb_ptrs_tensor"],
                n=cfg["n"],
                b_dtype=cfg["ab_dtype"],
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
                act_func=cfg["act_func"],
                current_stream=stream,
            )
            torch.cuda.synchronize()
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    finally:
        cache_entries = len(grouped_gemm_dglu_api._cache_of_GroupedGemmDgluSm100Objects)
        grouped_gemm_dglu_api._cache_of_GroupedGemmDgluSm100Objects.clear()

    return compile_count["value"], cache_entries


# ---------------------------------------------------------------------------
#  Linear-offset plumbing tests (act_func="dgeglu")
# ---------------------------------------------------------------------------
#
# The backward kernel computes
#     dx_gate = x_gate_filter * g * sigmoid_out * (1 + 1.702 * y_gate * (1 - sigmoid_out)) * (y_up + linear_offset)
#     dx_up   = x_up_filter   * g * y_gate * sigmoid_out
#     prob_grad += y_gate * sigmoid_out * (y_up + linear_offset) * fc2_dgrad
# with y_gate / y_up taken from interleaved 32-column blocks of the forward
# activation tensor C (and clamped to [-7.0, 7.0] / [..., 7.0] inside the
# kernel). Because dx_up is independent of ``linear_offset``, varying
# ``linear_offset`` while keeping every other input fixed must leave the
# "up" 32-column blocks of d_row / d_col exactly unchanged. The "gate"
# blocks must change, and ``dprob_tensor`` must also change.


def _interleaved_block_indices(n_out: int, device) -> "tuple[torch.Tensor, torch.Tensor]":
    group = 32
    assert n_out % group == 0 and (n_out // group) % 2 == 0
    cols = torch.arange(n_out, device=device)
    block_cols = cols.view(n_out // group, group)
    return block_cols[0::2].reshape(-1), block_cols[1::2].reshape(-1)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("linear_offset", [0.5, 1.0])
def test_grouped_gemm_dglu_discrete_wrapper_linear_offset_dgeglu(linear_offset, request):
    """Verify ``linear_offset`` correctly reaches the backward kernel.

    Plumbing checks performed:

    * ``dx_up`` (odd 32-column blocks of ``d_row_tensor``) is independent of
      ``linear_offset`` and must be bit-identical between a run with
      ``linear_offset=0.0`` and a run with the parameterized value.
    * ``dx_gate`` (even 32-column blocks) and ``dprob_tensor`` must differ
      between the two runs (otherwise the kernel ignored the new value).
    """
    try:
        from cudnn import grouped_gemm_dglu_wrapper_sm100
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("cudnn optional dependencies not installed")

    cfg = discrete_dswiglu_init(
        request=request,
        ab_dtype=torch.float4_e2m1fn_x2,
        c_dtype=torch.bfloat16,
        d_dtype=torch.float32,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=False,
        discrete_col_sfd=False,
        act_func="dgeglu",
        b_major="k",
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    inputs = allocate_discrete_dswiglu_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        num_experts=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        c_dtype=cfg["c_dtype"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
        b_major=cfg["b_major"],
    )

    common_kwargs = dict(
        a_tensor=inputs["a_tensor"],
        c_tensor=inputs["c_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        beta_tensor=inputs["beta_tensor"],
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
        act_func="dgeglu",
        current_stream=stream,
    )

    try:
        # First run: linear_offset = 0.0
        inputs["dprob_tensor"].zero_()
        dprob_zero_in = inputs["dprob_tensor"]
        out_zero = grouped_gemm_dglu_wrapper_sm100(
            dprob_tensor=dprob_zero_in,
            linear_offset=0.0,
            **common_kwargs,
        )
        torch.cuda.synchronize()
        d_row_zero = out_zero["d_row_tensor"].clone()
        dprob_zero = out_zero["dprob_tensor"].clone()

        # Second run: linear_offset = parameterized value, with a freshly zeroed dprob
        inputs["dprob_tensor"].zero_()
        out_offset = grouped_gemm_dglu_wrapper_sm100(
            dprob_tensor=inputs["dprob_tensor"],
            linear_offset=linear_offset,
            **common_kwargs,
        )
        torch.cuda.synchronize()
        d_row_offset = out_offset["d_row_tensor"].clone()
        dprob_offset = out_offset["dprob_tensor"].clone()
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    valid_m = inputs["valid_m"]
    n_out = 2 * cfg["n"]
    gate_idx, up_idx = _interleaved_block_indices(n_out, d_row_zero.device)

    # dx_up does not depend on linear_offset -> must be bit-identical
    torch.testing.assert_close(
        d_row_zero[:valid_m].index_select(1, up_idx).cpu(),
        d_row_offset[:valid_m].index_select(1, up_idx).cpu(),
        atol=0.0,
        rtol=0.0,
    )

    # dx_gate must move with linear_offset (otherwise the kernel ignored it).
    gate_zero = d_row_zero[:valid_m].index_select(1, gate_idx).float()
    gate_offset = d_row_offset[:valid_m].index_select(1, gate_idx).float()
    assert not torch.equal(gate_zero, gate_offset), "dx_gate did not change when linear_offset was varied"

    # dprob also picks up linear_offset (it must differ unless prob_tensor or
    # forward activations happen to be identically zero, which would be a
    # degenerate test setup).
    assert not torch.equal(dprob_zero, dprob_offset), "dprob did not change when linear_offset was varied"


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_dglu_discrete_wrapper_linear_offset_default_dgeglu_is_one(request):
    """Default ``linear_offset=None`` must reproduce ``linear_offset=1.0`` for dgeglu.

    Mirrors the forward-side default test to guard the documented backwards-
    compatibility behavior.
    """
    try:
        from cudnn import grouped_gemm_dglu_wrapper_sm100
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("cudnn optional dependencies not installed")

    cfg = discrete_dswiglu_init(
        request=request,
        ab_dtype=torch.float4_e2m1fn_x2,
        c_dtype=torch.bfloat16,
        d_dtype=torch.float32,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=False,
        discrete_col_sfd=False,
        act_func="dgeglu",
        b_major="k",
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    inputs = allocate_discrete_dswiglu_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        num_experts=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        c_dtype=cfg["c_dtype"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
        b_major=cfg["b_major"],
    )

    common_kwargs = dict(
        a_tensor=inputs["a_tensor"],
        c_tensor=inputs["c_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        beta_tensor=inputs["beta_tensor"],
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
        act_func="dgeglu",
        current_stream=stream,
    )

    try:
        inputs["dprob_tensor"].zero_()
        out_default = grouped_gemm_dglu_wrapper_sm100(
            dprob_tensor=inputs["dprob_tensor"],
            **common_kwargs,
        )
        torch.cuda.synchronize()
        d_row_default = out_default["d_row_tensor"].clone()
        dprob_default = out_default["dprob_tensor"].clone()

        inputs["dprob_tensor"].zero_()
        out_one = grouped_gemm_dglu_wrapper_sm100(
            dprob_tensor=inputs["dprob_tensor"],
            linear_offset=1.0,
            **common_kwargs,
        )
        torch.cuda.synchronize()
        d_row_one = out_one["d_row_tensor"].clone()
        dprob_one = out_one["dprob_tensor"].clone()
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    valid_m = inputs["valid_m"]
    torch.testing.assert_close(
        d_row_default[:valid_m].cpu(),
        d_row_one[:valid_m].cpu(),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        dprob_default[:valid_m].cpu(),
        dprob_one[:valid_m].cpu(),
        atol=0.0,
        rtol=0.0,
    )


# ---------------------------------------------------------------------------
#  Alpha + clamp plumbing tests (act_func="dgeglu")
# ---------------------------------------------------------------------------


def _run_dgeglu_wrapper(request, *, alpha, glu_clamp_max, glu_clamp_min, linear_offset, cfg=None, inputs=None):
    """Helper: run grouped_gemm_dglu_wrapper_sm100 with act_func="dgeglu" once.

    If *cfg* and *inputs* are supplied they are reused (no fresh allocation),
    allowing callers to compare runs that differ only in runtime knobs.
    """

    from cudnn import grouped_gemm_dglu_wrapper_sm100
    from cuda.bindings import driver as cuda

    if cfg is None:
        cfg = discrete_dswiglu_init(
            request=request,
            ab_dtype=torch.float4_e2m1fn_x2,
            c_dtype=torch.bfloat16,
            d_dtype=torch.float32,
            cd_major="n",
            acc_dtype=torch.float32,
            mma_tiler_mn=(256, 256),
            cluster_shape_mn=(2, 1),
            sf_vec_size=32,
            sf_dtype=torch.float8_e8m0fnu,
            vector_f32=False,
            discrete_col_sfd=False,
            act_func="dgeglu",
            b_major="k",
        )

    if inputs is None:
        inputs = allocate_discrete_dswiglu_input_tensors(
            n=cfg["n"],
            k=cfg["k"],
            num_experts=cfg["l"],
            group_m_list=cfg["group_m_list"],
            ab_dtype=cfg["ab_dtype"],
            c_dtype=cfg["c_dtype"],
            sf_dtype=cfg["sf_dtype"],
            sf_vec_size=cfg["sf_vec_size"],
            m_aligned=cfg["m_aligned"],
            b_major=cfg["b_major"],
        )

    inputs["dprob_tensor"].zero_()

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    out = grouped_gemm_dglu_wrapper_sm100(
        a_tensor=inputs["a_tensor"],
        c_tensor=inputs["c_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        beta_tensor=inputs["beta_tensor"],
        prob_tensor=inputs["prob_tensor"],
        dprob_tensor=inputs["dprob_tensor"],
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
        act_func="dgeglu",
        linear_offset=linear_offset,
        geglu_alpha=alpha,
        glu_clamp_max=glu_clamp_max,
        glu_clamp_min=glu_clamp_min,
        current_stream=stream,
    )
    torch.cuda.synchronize()
    return cfg, inputs, out


def _make_dgeglu_clamp_sensitive_inputs(inputs, n):
    n_out = 2 * n
    gate_idx, up_idx = _interleaved_block_indices(n_out, inputs["c_tensor"].device)

    with torch.no_grad():
        inputs["alpha_tensor"].fill_(1.0)
        inputs["beta_tensor"].fill_(1.0)
        inputs["prob_tensor"].fill_(1.0)

        inputs["c_tensor"][:, gate_idx, :] = 3.0
        inputs["c_tensor"][:, up_idx, :] = 1.0


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("alpha", [1.0, 1.702, 2.0])
def test_grouped_gemm_dglu_discrete_wrapper_alpha_dgeglu(alpha, request):
    """Verify that varying ``geglu_alpha`` reaches the backward kernel.

    ``dx_up = g * y_gate * sigmoid(alpha * y_gate)`` depends on ``alpha`` only
    through ``sigmoid(alpha * y_gate)``, while ``dx_gate`` and ``dprob``
    additionally pick up the ``(1 + alpha * y_gate * (1 - sigmoid))`` factor.
    Two runs differing only in ``alpha`` must produce different ``d_row``
    tensors and different ``dprob`` outputs (otherwise ``alpha`` was ignored).
    """
    try:
        import cudnn  # noqa: F401
        from cuda.bindings import driver as cuda  # noqa: F401
    except ImportError:
        pytest.skip("cudnn optional dependencies not installed")

    try:
        cfg, inputs, out_a = _run_dgeglu_wrapper(
            request,
            alpha=alpha,
            glu_clamp_max=7.0,
            glu_clamp_min=-7.0,
            linear_offset=1.0,
        )
        d_row_a = out_a["d_row_tensor"].clone()
        dprob_a = out_a["dprob_tensor"].clone()

        _, _, out_b = _run_dgeglu_wrapper(
            request,
            alpha=1.0 if alpha != 1.0 else 2.5,
            glu_clamp_max=7.0,
            glu_clamp_min=-7.0,
            linear_offset=1.0,
            cfg=cfg,
            inputs=inputs,
        )
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    valid_m = inputs["valid_m"]
    assert not torch.equal(
        d_row_a[:valid_m].float().cpu(),
        out_b["d_row_tensor"][:valid_m].float().cpu(),
    ), "d_row did not change when alpha was varied"
    assert not torch.equal(
        dprob_a[:valid_m].float().cpu(),
        out_b["dprob_tensor"][:valid_m].float().cpu(),
    ), "dprob did not change when alpha was varied"


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("glu_clamp_max", [3.5, 7.0, 1.0e6])
def test_grouped_gemm_dglu_discrete_wrapper_clamp_max_dgeglu(glu_clamp_max, request):
    """Verify that varying ``glu_clamp_max`` reaches the backward kernel.

    Two runs differing only in ``glu_clamp_max`` must produce different
    gradient outputs (otherwise the clamp/mask was ignored). The very-large
    ``1.0e6`` value exercises the "effectively no clamp" path; ``3.5``
    saturates a meaningful fraction of the inputs.
    """
    try:
        import cudnn  # noqa: F401
        from cuda.bindings import driver as cuda  # noqa: F401
    except ImportError:
        pytest.skip("cudnn optional dependencies not installed")

    try:
        cfg, inputs, _ = _run_dgeglu_wrapper(
            request,
            alpha=1.702,
            glu_clamp_max=7.0,
            glu_clamp_min=-7.0,
            linear_offset=1.0,
        )
        _make_dgeglu_clamp_sensitive_inputs(inputs, cfg["n"])

        _, _, out_a = _run_dgeglu_wrapper(
            request,
            alpha=1.702,
            glu_clamp_max=glu_clamp_max,
            glu_clamp_min=-glu_clamp_max,
            linear_offset=1.0,
            cfg=cfg,
            inputs=inputs,
        )
        d_row_a = out_a["d_row_tensor"].clone()

        _, _, out_b = _run_dgeglu_wrapper(
            request,
            alpha=1.702,
            glu_clamp_max=2.0 if glu_clamp_max != 2.0 else 5.0,
            glu_clamp_min=-(2.0 if glu_clamp_max != 2.0 else 5.0),
            linear_offset=1.0,
            cfg=cfg,
            inputs=inputs,
        )
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    valid_m = inputs["valid_m"]
    assert not torch.equal(
        d_row_a[:valid_m].float().cpu(),
        out_b["d_row_tensor"][:valid_m].float().cpu(),
    ), "d_row did not change when glu_clamp_max was varied"


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_dglu_discrete_wrapper_alpha_default_is_1702(request):
    """Default ``geglu_alpha`` / clamp values must reproduce the explicit defaults."""
    try:
        from cudnn import grouped_gemm_dglu_wrapper_sm100
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("cudnn optional dependencies not installed")

    cfg = discrete_dswiglu_init(
        request=request,
        ab_dtype=torch.float4_e2m1fn_x2,
        c_dtype=torch.bfloat16,
        d_dtype=torch.float32,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=False,
        discrete_col_sfd=False,
        act_func="dgeglu",
        b_major="k",
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    inputs = allocate_discrete_dswiglu_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        num_experts=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        c_dtype=cfg["c_dtype"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
        b_major=cfg["b_major"],
    )

    common_kwargs = dict(
        a_tensor=inputs["a_tensor"],
        c_tensor=inputs["c_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        beta_tensor=inputs["beta_tensor"],
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
        act_func="dgeglu",
        linear_offset=1.0,
        current_stream=stream,
    )

    try:
        inputs["dprob_tensor"].zero_()
        out_default = grouped_gemm_dglu_wrapper_sm100(
            dprob_tensor=inputs["dprob_tensor"],
            **common_kwargs,
        )
        torch.cuda.synchronize()
        d_row_default = out_default["d_row_tensor"].clone()
        dprob_default = out_default["dprob_tensor"].clone()

        inputs["dprob_tensor"].zero_()
        out_explicit = grouped_gemm_dglu_wrapper_sm100(
            dprob_tensor=inputs["dprob_tensor"],
            geglu_alpha=1.702,
            glu_clamp_max=7.0,
            glu_clamp_min=-7.0,
            **common_kwargs,
        )
        torch.cuda.synchronize()
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    valid_m = inputs["valid_m"]
    torch.testing.assert_close(
        d_row_default[:valid_m].cpu(),
        out_explicit["d_row_tensor"][:valid_m].cpu(),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        dprob_default[:valid_m].cpu(),
        out_explicit["dprob_tensor"][:valid_m].cpu(),
        atol=0.0,
        rtol=0.0,
    )
