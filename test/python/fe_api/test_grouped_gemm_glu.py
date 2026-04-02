"""
Tests for Unified Grouped GEMM GLU Forward Kernel (SM100+)

Tests the GroupedGemmGluSm100 API which supports both dense (contiguous)
and discrete weight modes, with SwiGLU and GeGLU activations.
"""

import torch
import pytest
from test_utils import torch_fork_set_rng
from fe_api.test_fe_api_utils import DYNAMIC_SHAPES_M_VALUES
from fe_api.test_grouped_gemm_swiglu_utils import (
    grouped_gemm_swiglu_init,
    with_grouped_gemm_swiglu_params_fp4,
    with_grouped_gemm_swiglu_params_fp8,
    with_grouped_gemm_swiglu_params_bias_fp4,
    with_grouped_gemm_swiglu_params_bias_fp8,
    allocate_grouped_gemm_input_tensors,
    allocate_grouped_gemm_output_tensors,
    check_ref_grouped_gemm_swiglu,
)
from fe_api.test_discrete_grouped_gemm_swiglu_utils import (
    discrete_grouped_gemm_init,
    allocate_discrete_input_tensors,
    allocate_discrete_output_tensors,
    check_ref_discrete_grouped_gemm,
)

with_scheduler_modes = pytest.mark.parametrize(
    "use_dynamic_sched",
    [False, True],
    ids=["static_sched", "dynamic_sched"],
)


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
#  Dense mode: Class API (reuses same tensor setup as contiguous swiglu)
# ---------------------------------------------------------------------------


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_scheduler_modes
@with_grouped_gemm_swiglu_params_fp4
def test_grouped_gemm_glu_dense_compile_execute_fp4(
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
    use_dynamic_sched,
    request,
):
    _test_grouped_gemm_glu_dense_compile_execute(
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
        use_dynamic_sched=use_dynamic_sched,
        request=request,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_scheduler_modes
@with_grouped_gemm_swiglu_params_fp8
def test_grouped_gemm_glu_dense_compile_execute_fp8(
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
    use_dynamic_sched,
    request,
):
    _test_grouped_gemm_glu_dense_compile_execute(
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
        use_dynamic_sched=use_dynamic_sched,
        request=request,
    )


# ---------------------------------------------------------------------------
#  Dense mode: Wrapper API
# ---------------------------------------------------------------------------


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_scheduler_modes
@with_grouped_gemm_swiglu_params_fp4
def test_grouped_gemm_glu_dense_wrapper_fp4(
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
    use_dynamic_sched,
    request,
):
    _test_grouped_gemm_glu_dense_wrapper(
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
        use_dynamic_sched=use_dynamic_sched,
        request=request,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_scheduler_modes
@with_grouped_gemm_swiglu_params_fp8
def test_grouped_gemm_glu_dense_wrapper_fp8(
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
    use_dynamic_sched,
    request,
):
    _test_grouped_gemm_glu_dense_wrapper(
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
        use_dynamic_sched=use_dynamic_sched,
        request=request,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_glu_dense_compile_execute_rectangular_zero_prob(request):
    def input_mutator(inputs, _cfg):
        inputs["prob_tensor"].zero_()
        inputs["alpha_tensor"].copy_(torch.tensor([1.0, -1.5, 0.75], dtype=torch.float32, device=inputs["alpha_tensor"].device))

    inputs, outputs, _ = _test_grouped_gemm_glu_dense_compile_execute(
        ab_dtype=torch.float4_e2m1fn_x2,
        c_dtype=torch.bfloat16,
        d_dtype=torch.float32,
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
            "n": 384,
            "k": 192,
            "group_m_list": [128, 320, 96],
        },
        input_mutator=input_mutator,
    )

    assert torch.count_nonzero(outputs["c_tensor"][: inputs["valid_m"]]).item() > 0
    assert torch.count_nonzero(outputs["d_tensor"][: inputs["valid_m"]]).item() == 0


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_grouped_gemm_swiglu_params_bias_fp4
def test_grouped_gemm_glu_dense_compile_execute_with_bias_fp4(
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
    request,
):
    _test_grouped_gemm_glu_dense_compile_execute(
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
        request=request,
        enable_bias=True,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_grouped_gemm_swiglu_params_bias_fp8
def test_grouped_gemm_glu_dense_compile_execute_with_bias_fp8(
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
    request,
):
    _test_grouped_gemm_glu_dense_compile_execute(
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
        request=request,
        enable_bias=True,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_grouped_gemm_swiglu_params_bias_fp4
def test_grouped_gemm_glu_dense_wrapper_with_bias_fp4(
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
    request,
):
    _test_grouped_gemm_glu_dense_wrapper(
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
        request=request,
        enable_bias=True,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_grouped_gemm_swiglu_params_bias_fp8
def test_grouped_gemm_glu_dense_wrapper_with_bias_fp8(
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
    request,
):
    _test_grouped_gemm_glu_dense_wrapper(
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
        request=request,
        enable_bias=True,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_glu_dense_compile_execute_with_bias_partial_n(request):
    _test_grouped_gemm_glu_dense_compile_execute(
        ab_dtype=torch.float4_e2m1fn_x2,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(128, 256),
        cluster_shape_mn=(1, 1),
        sf_vec_size=32,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=True,
        discrete_col_sfd=False,
        request=request,
        cfg_overrides={"n": 384},
        enable_bias=True,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_glu_dense_wrapper_with_bias_partial_n(request):
    _test_grouped_gemm_glu_dense_wrapper(
        ab_dtype=torch.float4_e2m1fn_x2,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(128, 256),
        cluster_shape_mn=(1, 1),
        sf_vec_size=32,
        sf_dtype=torch.float8_e8m0fnu,
        vector_f32=True,
        discrete_col_sfd=False,
        request=request,
        cfg_overrides={"n": 384},
        enable_bias=True,
    )


# ---------------------------------------------------------------------------
#  Impl: Dense Class API
# ---------------------------------------------------------------------------


def _test_grouped_gemm_glu_dense_compile_execute(
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
    request,
    cfg_overrides=None,
    input_mutator=None,
    enable_bias=False,
    use_dynamic_sched=False,
):
    try:
        from cudnn import GroupedGemmGluSm100
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("cudnn optional dependencies not installed")

    cfg = grouped_gemm_swiglu_init(
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
        enable_bias=enable_bias,
    )
    cfg = _apply_grouped_gemm_cfg_overrides(cfg, cfg_overrides)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    inputs = allocate_grouped_gemm_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        l=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
        enable_bias=cfg["enable_bias"],
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

    if input_mutator is not None:
        input_mutator(inputs, cfg)

    # Use the new unified GLU API in dense mode
    api = GroupedGemmGluSm100(
        sample_a=inputs["a_tensor"],
        sample_c=outputs["c_tensor"],
        sample_d=outputs["d_tensor"],
        sample_sfa=inputs["sfa_tensor"],
        sample_padded_offsets=inputs["padded_offsets_tensor"],
        sample_alpha=inputs["alpha_tensor"],
        sample_d_col=outputs["d_col_tensor"],
        sample_bias=inputs.get("bias_tensor"),
        # Dense mode:
        sample_b=inputs["b_tensor"],
        sample_sfb=inputs["sfb_tensor"],
        # Optional:
        sample_sfd_row=outputs.get("sfd_row_tensor"),
        sample_sfd_col=outputs.get("sfd_col_tensor"),
        sample_amax=outputs.get("amax_tensor"),
        sample_norm_const=inputs.get("norm_const_tensor"),
        sample_prob=inputs.get("prob_tensor"),
        # Configuration:
        acc_dtype=cfg["acc_dtype"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sf_vec_size=cfg["sf_vec_size"],
        vector_f32=cfg["vector_f32"],
        m_aligned=cfg["m_aligned"],
        discrete_col_sfd=cfg["discrete_col_sfd"],
        act_func="swiglu",
        use_dynamic_sched=use_dynamic_sched,
    )

    try:
        assert api.check_support(), "Unsupported testcase"
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    api.compile()
    api.execute(
        a_tensor=inputs["a_tensor"],
        c_tensor=outputs["c_tensor"],
        d_tensor=outputs["d_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        b_tensor=inputs["b_tensor"],
        sfb_tensor=inputs["sfb_tensor"],
        bias_tensor=inputs.get("bias_tensor"),
        d_col_tensor=outputs["d_col_tensor"],
        sfd_row_tensor=outputs.get("sfd_row_tensor"),
        sfd_col_tensor=outputs.get("sfd_col_tensor"),
        amax_tensor=outputs.get("amax_tensor"),
        norm_const_tensor=inputs.get("norm_const_tensor"),
        prob_tensor=inputs.get("prob_tensor"),
        current_stream=stream,
    )

    check_ref_grouped_gemm_swiglu(inputs, outputs, cfg, skip_ref=cfg["skip_ref"])
    return inputs, outputs, cfg


# ---------------------------------------------------------------------------
#  Impl: Dense Wrapper API
# ---------------------------------------------------------------------------


def _test_grouped_gemm_glu_dense_wrapper(
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
    request,
    cfg_overrides=None,
    input_mutator=None,
    enable_bias=False,
    use_dynamic_sched=False,
):
    try:
        from cudnn import grouped_gemm_glu_wrapper_sm100
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("cudnn optional dependencies not installed")

    cfg = grouped_gemm_swiglu_init(
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
        enable_bias=enable_bias,
    )
    cfg = _apply_grouped_gemm_cfg_overrides(cfg, cfg_overrides)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    inputs = allocate_grouped_gemm_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        l=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
        enable_bias=cfg["enable_bias"],
    )

    if input_mutator is not None:
        input_mutator(inputs, cfg)

    try:
        for _ in range(2):  # Run twice to test caching path
            outputs = grouped_gemm_glu_wrapper_sm100(
                a_tensor=inputs["a_tensor"],
                sfa_tensor=inputs["sfa_tensor"],
                padded_offsets=inputs["padded_offsets_tensor"],
                alpha_tensor=inputs["alpha_tensor"],
                bias_tensor=inputs.get("bias_tensor"),
                # Dense mode:
                b_tensor=inputs["b_tensor"],
                sfb_tensor=inputs["sfb_tensor"],
                # Common:
                norm_const_tensor=inputs.get("norm_const_tensor"),
                prob_tensor=inputs.get("prob_tensor"),
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
                act_func="swiglu",
                use_dynamic_sched=use_dynamic_sched,
                current_stream=stream,
            )
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    check_ref_grouped_gemm_swiglu(inputs, outputs, cfg, skip_ref=cfg["skip_ref"])
    return inputs, outputs, cfg


@pytest.mark.L0
@torch_fork_set_rng(seed=1)
@pytest.mark.parametrize(
    "ab_dtype",
    [
        pytest.param(torch.float4_e2m1fn_x2, id="fp4"),
        pytest.param(torch.float8_e4m3fn, id="fp8"),
    ],
)
def test_grouped_gemm_glu_dense_wrapper_cache_partial_dynamic_smoke(request, monkeypatch, ab_dtype):
    compile_count, cache_entries = _test_grouped_gemm_glu_dense_wrapper_dynamic_m_cache_behavior(
        request=request,
        monkeypatch=monkeypatch,
        use_full_dynamic=False,
        ab_dtype=ab_dtype,
    )

    assert compile_count == 1
    assert cache_entries == 1


@pytest.mark.L0
@torch_fork_set_rng(seed=2)
@pytest.mark.parametrize("ab_dtype", [pytest.param(torch.float4_e2m1fn_x2, id="fp4")])
def test_grouped_gemm_glu_dense_wrapper_cache_full_dynamic_smoke(request, monkeypatch, ab_dtype):
    compile_count, cache_entries = _test_grouped_gemm_glu_dense_wrapper_dynamic_nk_cache_behavior(
        request=request,
        monkeypatch=monkeypatch,
        ab_dtype=ab_dtype,
    )

    assert compile_count == 1
    assert cache_entries == 1


def _test_grouped_gemm_glu_dense_wrapper_dynamic_m_cache_behavior(request, monkeypatch, use_full_dynamic, ab_dtype):
    try:
        from cudnn import grouped_gemm_glu_wrapper_sm100
        from cudnn.grouped_gemm.grouped_gemm_glu import api as grouped_gemm_glu_api
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    if use_full_dynamic:
        monkeypatch.setenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1")
    else:
        monkeypatch.delenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", raising=False)

    grouped_gemm_glu_api._cache_of_GroupedGemmGluSm100Objects.clear()

    compile_count = {"value": 0}

    def counted_compile(self):
        compile_count["value"] += 1

    monkeypatch.setattr(grouped_gemm_glu_api.GroupedGemmGluSm100, "check_support", lambda self: True)
    monkeypatch.setattr(grouped_gemm_glu_api.GroupedGemmGluSm100, "compile", counted_compile)
    monkeypatch.setattr(grouped_gemm_glu_api.GroupedGemmGluSm100, "execute", lambda self, **kwargs: None)

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
                sf_dtype=cfg["sf_dtype"],
                sf_vec_size=cfg["sf_vec_size"],
                m_aligned=cfg["m_aligned"],
            )

            grouped_gemm_glu_wrapper_sm100(
                a_tensor=inputs["a_tensor"],
                sfa_tensor=inputs["sfa_tensor"],
                padded_offsets=inputs["padded_offsets_tensor"],
                alpha_tensor=inputs["alpha_tensor"],
                b_tensor=inputs["b_tensor"],
                sfb_tensor=inputs["sfb_tensor"],
                norm_const_tensor=inputs.get("norm_const_tensor"),
                prob_tensor=inputs.get("prob_tensor"),
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
                act_func="swiglu",
                current_stream=stream,
            )
    finally:
        cache_entries = len(grouped_gemm_glu_api._cache_of_GroupedGemmGluSm100Objects)
        grouped_gemm_glu_api._cache_of_GroupedGemmGluSm100Objects.clear()

    return compile_count["value"], cache_entries


def _test_grouped_gemm_glu_dense_wrapper_dynamic_nk_cache_behavior(request, monkeypatch, ab_dtype):
    try:
        from cudnn import grouped_gemm_glu_wrapper_sm100
        from cudnn.grouped_gemm.grouped_gemm_glu import api as grouped_gemm_glu_api
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    monkeypatch.setenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1")
    grouped_gemm_glu_api._cache_of_GroupedGemmGluSm100Objects.clear()

    compile_count = {"value": 0}

    def counted_compile(self):
        compile_count["value"] += 1

    monkeypatch.setattr(grouped_gemm_glu_api.GroupedGemmGluSm100, "check_support", lambda self: True)
    monkeypatch.setattr(grouped_gemm_glu_api.GroupedGemmGluSm100, "compile", counted_compile)
    monkeypatch.setattr(grouped_gemm_glu_api.GroupedGemmGluSm100, "execute", lambda self, **kwargs: None)

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
                sf_dtype=cfg["sf_dtype"],
                sf_vec_size=cfg["sf_vec_size"],
                m_aligned=cfg["m_aligned"],
            )

            grouped_gemm_glu_wrapper_sm100(
                a_tensor=inputs["a_tensor"],
                sfa_tensor=inputs["sfa_tensor"],
                padded_offsets=inputs["padded_offsets_tensor"],
                alpha_tensor=inputs["alpha_tensor"],
                b_tensor=inputs["b_tensor"],
                sfb_tensor=inputs["sfb_tensor"],
                norm_const_tensor=inputs.get("norm_const_tensor"),
                prob_tensor=inputs.get("prob_tensor"),
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
                act_func="swiglu",
                current_stream=stream,
            )
    finally:
        cache_entries = len(grouped_gemm_glu_api._cache_of_GroupedGemmGluSm100Objects)
        grouped_gemm_glu_api._cache_of_GroupedGemmGluSm100Objects.clear()

    return compile_count["value"], cache_entries


# ---------------------------------------------------------------------------
#  Discrete mode: Class API
# ---------------------------------------------------------------------------


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_scheduler_modes
@pytest.mark.parametrize("act_func", ["swiglu", "geglu"])
def test_grouped_gemm_glu_discrete_compile_execute_fp4(act_func, use_dynamic_sched, request):
    _test_grouped_gemm_glu_discrete_compile_execute(
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
@pytest.mark.parametrize("act_func", ["swiglu", "geglu"])
@pytest.mark.parametrize("b_major", ["k", "n"])
def test_grouped_gemm_glu_discrete_compile_execute_fp8(act_func, b_major, use_dynamic_sched, request):
    _test_grouped_gemm_glu_discrete_compile_execute(
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
@pytest.mark.parametrize("act_func", ["swiglu", "geglu"])
def test_grouped_gemm_glu_discrete_wrapper_fp4(act_func, use_dynamic_sched, request):
    _test_grouped_gemm_glu_discrete_wrapper(
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
@pytest.mark.parametrize("act_func", ["swiglu", "geglu"])
@pytest.mark.parametrize("b_major", ["k", "n"])
def test_grouped_gemm_glu_discrete_wrapper_fp8(act_func, b_major, use_dynamic_sched, request):
    _test_grouped_gemm_glu_discrete_wrapper(
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


def _test_grouped_gemm_glu_discrete_compile_execute(
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
    enable_bias=False,
    use_dynamic_sched=False,
):
    try:
        from cudnn import GroupedGemmGluSm100
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("cudnn optional dependencies not installed")

    cfg = discrete_grouped_gemm_init(
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

    inputs = allocate_discrete_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        num_experts=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
        b_major=cfg["b_major"],
        enable_bias=enable_bias,
    )

    outputs = allocate_discrete_output_tensors(
        tensor_m=inputs["tensor_m"],
        n=cfg["n"],
        num_experts=cfg["l"],
        ab_dtype=cfg["ab_dtype"],
        c_dtype=cfg["c_dtype"],
        d_dtype=cfg["d_dtype"],
        cd_major=cfg["cd_major"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
    )

    api = GroupedGemmGluSm100(
        sample_a=inputs["a_tensor"],
        sample_c=outputs["c_tensor"],
        sample_d=outputs["d_tensor"],
        sample_sfa=inputs["sfa_tensor"],
        sample_padded_offsets=inputs["padded_offsets_tensor"],
        sample_alpha=inputs["alpha_tensor"],
        sample_d_col=outputs["d_col_tensor"],
        sample_bias=inputs.get("bias_tensor"),
        num_experts=len(inputs["b_list"]),
        b_shape=(cfg["n"], cfg["k"]),
        b_dtype=inputs["b_list"][0].dtype,
        sample_sfd_row=outputs.get("sfd_row_tensor"),
        sample_sfd_col=outputs.get("sfd_col_tensor"),
        sample_amax=outputs.get("amax_tensor"),
        sample_norm_const=inputs.get("norm_const_tensor"),
        sample_prob=inputs.get("prob_tensor"),
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
        c_tensor=outputs["c_tensor"],
        d_tensor=outputs["d_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        b_ptrs=inputs["b_ptrs_tensor"],
        sfb_ptrs=inputs["sfb_ptrs_tensor"],
        bias_tensor=inputs.get("bias_tensor"),
        d_col_tensor=outputs["d_col_tensor"],
        sfd_row_tensor=outputs.get("sfd_row_tensor"),
        sfd_col_tensor=outputs.get("sfd_col_tensor"),
        amax_tensor=outputs.get("amax_tensor"),
        norm_const_tensor=inputs.get("norm_const_tensor"),
        prob_tensor=inputs.get("prob_tensor"),
        current_stream=stream,
    )

    check_ref_discrete_grouped_gemm(inputs, outputs, cfg, skip_ref=cfg["skip_ref"])


def _test_grouped_gemm_glu_discrete_wrapper(
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
    enable_bias=False,
    use_dynamic_sched=False,
):
    try:
        from cudnn import grouped_gemm_glu_wrapper_sm100
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("cudnn optional dependencies not installed")

    cfg = discrete_grouped_gemm_init(
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

    inputs = allocate_discrete_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        num_experts=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
        b_major=cfg["b_major"],
        enable_bias=enable_bias,
    )

    try:
        for _ in range(2):  # Run twice to test caching path
            outputs = grouped_gemm_glu_wrapper_sm100(
                a_tensor=inputs["a_tensor"],
                sfa_tensor=inputs["sfa_tensor"],
                padded_offsets=inputs["padded_offsets_tensor"],
                alpha_tensor=inputs["alpha_tensor"],
                b_ptrs=inputs["b_ptrs_tensor"],
                sfb_ptrs=inputs["sfb_ptrs_tensor"],
                bias_tensor=inputs.get("bias_tensor"),
                n=cfg["n"],
                b_dtype=inputs["b_list"][0].dtype,
                b_major=cfg["b_major"],
                norm_const_tensor=inputs.get("norm_const_tensor"),
                prob_tensor=inputs.get("prob_tensor"),
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
                act_func=cfg["act_func"],
                use_dynamic_sched=use_dynamic_sched,
                current_stream=stream,
            )
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    check_ref_discrete_grouped_gemm(inputs, outputs, cfg, skip_ref=cfg["skip_ref"])


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_glu_discrete_compile_execute_with_bias(request):
    _test_grouped_gemm_glu_discrete_compile_execute(
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
        act_func="swiglu",
        request=request,
        enable_bias=True,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_glu_discrete_wrapper_with_bias(request):
    _test_grouped_gemm_glu_discrete_wrapper(
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
        act_func="swiglu",
        request=request,
        enable_bias=True,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=3)
def test_grouped_gemm_glu_discrete_wrapper_cache_dynamic_m_smoke(request, monkeypatch):
    compile_count, cache_entries = _test_grouped_gemm_glu_discrete_wrapper_dynamic_m_cache_behavior(
        request=request,
        monkeypatch=monkeypatch,
    )

    assert compile_count == 1
    assert cache_entries == 1


def _test_grouped_gemm_glu_discrete_wrapper_dynamic_m_cache_behavior(request, monkeypatch):
    try:
        from cudnn import grouped_gemm_glu_wrapper_sm100
        from cudnn.grouped_gemm.grouped_gemm_glu import api as grouped_gemm_glu_api
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    grouped_gemm_glu_api._cache_of_GroupedGemmGluSm100Objects.clear()

    compile_count = {"value": 0}
    original_compile = grouped_gemm_glu_api.GroupedGemmGluSm100.compile

    def counted_compile(self):
        compile_count["value"] += 1
        return original_compile(self)

    monkeypatch.setattr(grouped_gemm_glu_api.GroupedGemmGluSm100, "compile", counted_compile)

    cfg = discrete_grouped_gemm_init(
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
        act_func="swiglu",
        b_major="k",
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    try:
        for group_m in DYNAMIC_SHAPES_M_VALUES:
            inputs = allocate_discrete_input_tensors(
                n=cfg["n"],
                k=cfg["k"],
                num_experts=cfg["l"],
                group_m_list=[group_m] * cfg["l"],
                ab_dtype=cfg["ab_dtype"],
                sf_dtype=cfg["sf_dtype"],
                sf_vec_size=cfg["sf_vec_size"],
                m_aligned=cfg["m_aligned"],
                b_major=cfg["b_major"],
            )

            grouped_gemm_glu_wrapper_sm100(
                a_tensor=inputs["a_tensor"],
                sfa_tensor=inputs["sfa_tensor"],
                padded_offsets=inputs["padded_offsets_tensor"],
                alpha_tensor=inputs["alpha_tensor"],
                b_ptrs=inputs["b_ptrs_tensor"],
                sfb_ptrs=inputs["sfb_ptrs_tensor"],
                n=cfg["n"],
                b_dtype=cfg["ab_dtype"],
                b_major=cfg["b_major"],
                norm_const_tensor=inputs.get("norm_const_tensor"),
                prob_tensor=inputs.get("prob_tensor"),
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
                act_func=cfg["act_func"],
                current_stream=stream,
            )
            torch.cuda.synchronize()
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    finally:
        cache_entries = len(grouped_gemm_glu_api._cache_of_GroupedGemmGluSm100Objects)
        grouped_gemm_glu_api._cache_of_GroupedGemmGluSm100Objects.clear()

    return compile_count["value"], cache_entries
