import pytest
import torch

import cudnn
from test_utils import torch_fork_set_rng

from fe_api.test_gemm_dsrelu_utils import (
    allocate_gemm_dsrelu_outputs,
    allocate_gemm_dsrelu_tensors,
    check_ref_gemm_dsrelu,
    gemm_dsrelu_init,
    with_gemm_dsrelu_params_fp4,
)


def _run_class_api(cfg, inputs, outputs):
    op = cudnn.GemmDsreluSm100(
        sample_a=inputs["a_tensor"],
        sample_b=inputs["b_tensor"],
        sample_c=inputs["c_tensor"],
        sample_d=outputs["d_tensor"],
        sample_dprob=outputs["dprob_tensor"],
        sample_sfa=inputs["sfa_tensor"],
        sample_sfb=inputs["sfb_tensor"],
        sample_prob=inputs["prob_tensor"],
        sample_sfd=outputs["sfd_tensor"],
        sample_amax=outputs["amax_tensor"],
        sample_norm_const=outputs["norm_const_tensor"],
        alpha=cfg["alpha"],
        acc_dtype=cfg["acc_dtype"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sf_vec_size=cfg["sf_vec_size"],
        vector_f32=cfg["vector_f32"],
    )
    try:
        assert op.check_support()
    except (ValueError, NotImplementedError, RuntimeError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    op.compile()
    op.execute(
        a_tensor=inputs["a_tensor"],
        b_tensor=inputs["b_tensor"],
        c_tensor=inputs["c_tensor"],
        d_tensor=outputs["d_tensor"],
        dprob_tensor=outputs["dprob_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        sfb_tensor=inputs["sfb_tensor"],
        prob_tensor=inputs["prob_tensor"],
        sfd_tensor=outputs["sfd_tensor"],
        amax_tensor=outputs["amax_tensor"],
        norm_const_tensor=outputs["norm_const_tensor"],
        alpha=cfg["alpha"],
    )
    torch.cuda.synchronize()


def _run_wrapper_api(cfg, inputs):
    try:
        result = None
        for _ in range(2):
            result = cudnn.gemm_dsrelu_wrapper_sm100(
                a_tensor=inputs["a_tensor"],
                b_tensor=inputs["b_tensor"],
                c_tensor=inputs["c_tensor"],
                sfa_tensor=inputs["sfa_tensor"],
                sfb_tensor=inputs["sfb_tensor"],
                prob_tensor=inputs["prob_tensor"],
                alpha=cfg["alpha"],
                d_major=cfg["c_major"],
                d_dtype=cfg["d_dtype"],
                acc_dtype=cfg["acc_dtype"],
                mma_tiler_mn=cfg["mma_tiler_mn"],
                cluster_shape_mn=cfg["cluster_shape_mn"],
                norm_const_tensor=(
                    None if cfg["d_dtype"] not in {torch.float8_e4m3fn, torch.float8_e5m2} else torch.tensor([1.0], dtype=torch.float32, device="cuda")
                ),
                sf_vec_size=cfg["sf_vec_size"],
                vector_f32=cfg["vector_f32"],
            )
    except (ValueError, NotImplementedError, RuntimeError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    torch.cuda.synchronize()
    return {
        "d_tensor": result["d_tensor"],
        "dprob_tensor": result["dprob_tensor"],
        "amax_tensor": result["amax_tensor"],
        "sfd_tensor": result["sfd_tensor"],
        "norm_const_tensor": None,
    }


def _make_dense_dsrelu_cfg(request, m: int, n: int = 256, k: int = 512, l: int = 2):
    cfg = gemm_dsrelu_init(
        request,
        "k",
        "k",
        "n",
        torch.float4_e2m1fn_x2,
        torch.bfloat16,
        torch.bfloat16,
        torch.float32,
        (256, 256),
        (2, 1),
        16,
        torch.float8_e8m0fnu,
        False,
    )
    cfg["m"] = m
    cfg["n"] = n
    cfg["k"] = k
    cfg["l"] = l
    return cfg


def _test_gemm_dsrelu_wrapper_dynamic_m_cache_behavior(request, monkeypatch, use_dynamic_m):
    try:
        from cudnn import gemm_dsrelu_wrapper_sm100
        from cudnn.gemm_dsrelu import api as gemm_dsrelu_api
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    if use_dynamic_m:
        monkeypatch.setenv("CUDNN_FE_GEMM_DYNAMIC_M", "1")
    else:
        monkeypatch.delenv("CUDNN_FE_GEMM_DYNAMIC_M", raising=False)

    gemm_dsrelu_api._cache_of_GemmDsreluSm100Objects.clear()
    compile_count = {"value": 0}

    def counted_compile(self):
        compile_count["value"] += 1

    monkeypatch.setattr(gemm_dsrelu_api.GemmDsreluSm100, "check_support", lambda self: True)
    monkeypatch.setattr(gemm_dsrelu_api.GemmDsreluSm100, "compile", counted_compile)
    monkeypatch.setattr(gemm_dsrelu_api.GemmDsreluSm100, "execute", lambda self, **kwargs: None)

    try:
        for m in (256, 384):
            cfg = _make_dense_dsrelu_cfg(request, m)
            inputs = allocate_gemm_dsrelu_tensors(cfg)
            gemm_dsrelu_wrapper_sm100(
                a_tensor=inputs["a_tensor"],
                b_tensor=inputs["b_tensor"],
                c_tensor=inputs["c_tensor"],
                sfa_tensor=inputs["sfa_tensor"],
                sfb_tensor=inputs["sfb_tensor"],
                prob_tensor=inputs["prob_tensor"],
                alpha=cfg["alpha"],
                d_major=cfg["c_major"],
                d_dtype=cfg["d_dtype"],
                acc_dtype=cfg["acc_dtype"],
                mma_tiler_mn=cfg["mma_tiler_mn"],
                cluster_shape_mn=cfg["cluster_shape_mn"],
                norm_const_tensor=None,
                sf_vec_size=cfg["sf_vec_size"],
                vector_f32=cfg["vector_f32"],
            )
    finally:
        cache_entries = len(gemm_dsrelu_api._cache_of_GemmDsreluSm100Objects)
        gemm_dsrelu_api._cache_of_GemmDsreluSm100Objects.clear()

    return compile_count["value"], cache_entries


def _test_gemm_dsrelu_wrapper_full_dynamic_cache_behavior(request, monkeypatch):
    try:
        from cudnn import gemm_dsrelu_wrapper_sm100
        from cudnn.gemm_dsrelu import api as gemm_dsrelu_api
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    monkeypatch.setenv("CUDNN_FE_GEMM_DYNAMIC_MNKL", "1")
    monkeypatch.delenv("CUDNN_FE_GEMM_DYNAMIC_M", raising=False)
    gemm_dsrelu_api._cache_of_GemmDsreluSm100Objects.clear()
    compile_count = {"value": 0}

    def counted_compile(self):
        compile_count["value"] += 1

    monkeypatch.setattr(gemm_dsrelu_api.GemmDsreluSm100, "check_support", lambda self: True)
    monkeypatch.setattr(gemm_dsrelu_api.GemmDsreluSm100, "compile", counted_compile)
    monkeypatch.setattr(gemm_dsrelu_api.GemmDsreluSm100, "execute", lambda self, **kwargs: None)

    try:
        for mnkl in ((256, 256, 512, 2), (384, 384, 640, 3)):
            cfg = _make_dense_dsrelu_cfg(request, *mnkl)
            inputs = allocate_gemm_dsrelu_tensors(cfg)
            gemm_dsrelu_wrapper_sm100(
                a_tensor=inputs["a_tensor"],
                b_tensor=inputs["b_tensor"],
                c_tensor=inputs["c_tensor"],
                sfa_tensor=inputs["sfa_tensor"],
                sfb_tensor=inputs["sfb_tensor"],
                prob_tensor=inputs["prob_tensor"],
                alpha=cfg["alpha"],
                d_major=cfg["c_major"],
                d_dtype=cfg["d_dtype"],
                acc_dtype=cfg["acc_dtype"],
                mma_tiler_mn=cfg["mma_tiler_mn"],
                cluster_shape_mn=cfg["cluster_shape_mn"],
                norm_const_tensor=None,
                sf_vec_size=cfg["sf_vec_size"],
                vector_f32=cfg["vector_f32"],
            )
    finally:
        cache_entries = len(gemm_dsrelu_api._cache_of_GemmDsreluSm100Objects)
        gemm_dsrelu_api._cache_of_GemmDsreluSm100Objects.clear()

    return compile_count["value"], cache_entries


@pytest.mark.L0
@torch_fork_set_rng(seed=10)
@with_gemm_dsrelu_params_fp4
def test_gemm_dsrelu_compile_execute_fp4(
    a_major,
    b_major,
    c_major,
    ab_dtype,
    c_dtype,
    d_dtype,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    vector_f32,
    request,
):
    cfg = gemm_dsrelu_init(
        request,
        a_major,
        b_major,
        c_major,
        ab_dtype,
        c_dtype,
        d_dtype,
        acc_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        sf_vec_size,
        sf_dtype,
        vector_f32,
    )
    inputs = allocate_gemm_dsrelu_tensors(cfg)
    outputs = allocate_gemm_dsrelu_outputs(cfg)
    _run_class_api(cfg, inputs, outputs)
    check_ref_gemm_dsrelu(inputs, outputs, cfg, check_d=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=11)
@with_gemm_dsrelu_params_fp4
def test_gemm_dsrelu_wrapper_fp4(
    a_major,
    b_major,
    c_major,
    ab_dtype,
    c_dtype,
    d_dtype,
    acc_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    sf_vec_size,
    sf_dtype,
    vector_f32,
    request,
):
    cfg = gemm_dsrelu_init(
        request,
        a_major,
        b_major,
        c_major,
        ab_dtype,
        c_dtype,
        d_dtype,
        acc_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        sf_vec_size,
        sf_dtype,
        vector_f32,
    )
    inputs = allocate_gemm_dsrelu_tensors(cfg)
    outputs = _run_wrapper_api(cfg, inputs)
    check_ref_gemm_dsrelu(inputs, outputs, cfg, check_d=True)


@pytest.mark.L0
@torch_fork_set_rng(seed=12)
def test_gemm_dsrelu_wrapper_cache_static_m_smoke(request, monkeypatch):
    compile_count, cache_entries = _test_gemm_dsrelu_wrapper_dynamic_m_cache_behavior(request, monkeypatch, use_dynamic_m=False)

    assert compile_count == 2
    assert cache_entries == 2


@pytest.mark.L0
@torch_fork_set_rng(seed=13)
def test_gemm_dsrelu_wrapper_cache_dynamic_m_smoke(request, monkeypatch):
    compile_count, cache_entries = _test_gemm_dsrelu_wrapper_dynamic_m_cache_behavior(request, monkeypatch, use_dynamic_m=True)

    assert compile_count == 1
    assert cache_entries == 1


@pytest.mark.L0
@torch_fork_set_rng(seed=14)
def test_gemm_dsrelu_wrapper_dynamic_m_fp4(request, monkeypatch):
    try:
        import cudnn
        from cudnn.gemm_dsrelu import api as gemm_dsrelu_api
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    monkeypatch.setenv("CUDNN_FE_GEMM_DYNAMIC_M", "1")
    gemm_dsrelu_api._cache_of_GemmDsreluSm100Objects.clear()

    try:
        for m in (256, 384):
            cfg = _make_dense_dsrelu_cfg(request, m)
            inputs = allocate_gemm_dsrelu_tensors(cfg)
            outputs = cudnn.gemm_dsrelu_wrapper_sm100(
                a_tensor=inputs["a_tensor"],
                b_tensor=inputs["b_tensor"],
                c_tensor=inputs["c_tensor"],
                sfa_tensor=inputs["sfa_tensor"],
                sfb_tensor=inputs["sfb_tensor"],
                prob_tensor=inputs["prob_tensor"],
                alpha=cfg["alpha"],
                d_major=cfg["c_major"],
                d_dtype=cfg["d_dtype"],
                acc_dtype=cfg["acc_dtype"],
                mma_tiler_mn=cfg["mma_tiler_mn"],
                cluster_shape_mn=cfg["cluster_shape_mn"],
                norm_const_tensor=None,
                sf_vec_size=cfg["sf_vec_size"],
                vector_f32=cfg["vector_f32"],
            )
            check_ref_gemm_dsrelu(inputs, outputs, cfg, check_d=True)

        assert len(gemm_dsrelu_api._cache_of_GemmDsreluSm100Objects) == 1
    finally:
        gemm_dsrelu_api._cache_of_GemmDsreluSm100Objects.clear()


@pytest.mark.L0
@torch_fork_set_rng(seed=15)
def test_gemm_dsrelu_wrapper_cache_full_dynamic_smoke(request, monkeypatch):
    compile_count, cache_entries = _test_gemm_dsrelu_wrapper_full_dynamic_cache_behavior(request, monkeypatch)

    assert compile_count == 1
    assert cache_entries == 1


@pytest.mark.L0
@torch_fork_set_rng(seed=16)
def test_gemm_dsrelu_wrapper_full_dynamic_fp4(request, monkeypatch):
    try:
        import cudnn
        from cudnn.gemm_dsrelu import api as gemm_dsrelu_api
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    monkeypatch.setenv("CUDNN_FE_GEMM_DYNAMIC_MNKL", "1")
    monkeypatch.delenv("CUDNN_FE_GEMM_DYNAMIC_M", raising=False)
    gemm_dsrelu_api._cache_of_GemmDsreluSm100Objects.clear()

    try:
        for mnkl in ((256, 256, 512, 2), (384, 384, 640, 3)):
            cfg = _make_dense_dsrelu_cfg(request, *mnkl)
            inputs = allocate_gemm_dsrelu_tensors(cfg)
            outputs = cudnn.gemm_dsrelu_wrapper_sm100(
                a_tensor=inputs["a_tensor"],
                b_tensor=inputs["b_tensor"],
                c_tensor=inputs["c_tensor"],
                sfa_tensor=inputs["sfa_tensor"],
                sfb_tensor=inputs["sfb_tensor"],
                prob_tensor=inputs["prob_tensor"],
                alpha=cfg["alpha"],
                d_major=cfg["c_major"],
                d_dtype=cfg["d_dtype"],
                acc_dtype=cfg["acc_dtype"],
                mma_tiler_mn=cfg["mma_tiler_mn"],
                cluster_shape_mn=cfg["cluster_shape_mn"],
                norm_const_tensor=None,
                sf_vec_size=cfg["sf_vec_size"],
                vector_f32=cfg["vector_f32"],
            )
            check_ref_gemm_dsrelu(inputs, outputs, cfg, check_d=True)

        assert len(gemm_dsrelu_api._cache_of_GemmDsreluSm100Objects) == 1
    finally:
        gemm_dsrelu_api._cache_of_GemmDsreluSm100Objects.clear()
