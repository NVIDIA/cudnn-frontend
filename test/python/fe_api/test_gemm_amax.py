import json

import torch

import pytest

from test_utils import torch_fork_set_rng
from fe_api.test_gemm_amax_utils import (
    with_gemm_amax_params_fp4,
    with_gemm_amax_params_fp8,
)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_gemm_amax_params_fp4
def test_gemm_amax_compile_execute_fp4(
    a_major,
    b_major,
    c_major,
    ab_dtype,
    sf_dtype,
    c_dtype,
    acc_dtype,
    sf_vec_size,
    mma_tiler_mn,
    cluster_shape_mn,
    request,
):
    _test_gemm_amax_compile_execute(
        a_major=a_major,
        b_major=b_major,
        c_major=c_major,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        acc_dtype=acc_dtype,
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        request=request,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_gemm_amax_params_fp8
def test_gemm_amax_compile_execute_fp8(
    a_major,
    b_major,
    c_major,
    ab_dtype,
    sf_dtype,
    c_dtype,
    acc_dtype,
    sf_vec_size,
    mma_tiler_mn,
    cluster_shape_mn,
    request,
):
    _test_gemm_amax_compile_execute(
        a_major=a_major,
        b_major=b_major,
        c_major=c_major,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        acc_dtype=acc_dtype,
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        request=request,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_gemm_amax_params_fp4
def test_gemm_amax_wrapper_fp4(
    a_major,
    b_major,
    c_major,
    ab_dtype,
    sf_dtype,
    c_dtype,
    acc_dtype,
    sf_vec_size,
    mma_tiler_mn,
    cluster_shape_mn,
    request,
):
    _test_gemm_amax_wrapper(
        a_major=a_major,
        b_major=b_major,
        c_major=c_major,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        acc_dtype=acc_dtype,
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        request=request,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_gemm_amax_params_fp8
def test_gemm_amax_wrapper_fp8(
    a_major,
    b_major,
    c_major,
    ab_dtype,
    sf_dtype,
    c_dtype,
    acc_dtype,
    sf_vec_size,
    mma_tiler_mn,
    cluster_shape_mn,
    request,
):
    _test_gemm_amax_wrapper(
        a_major=a_major,
        b_major=b_major,
        c_major=c_major,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        acc_dtype=acc_dtype,
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        request=request,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_gemm_amax_aot_export_load(tmp_path, monkeypatch):
    pytest.importorskip("cutlass", reason="CuTe DSL is not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    from cudnn import gemm_amax_wrapper_sm100
    from cudnn.gemm_amax import api as gemm_amax_api
    from fe_api.test_gemm_amax_utils import (
        allocate_input_tensors,
        check_ref_gemm_amax,
    )

    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        pytest.skip(f"GemmAmax AOT requires SM100+, found SM{major}{minor}")

    a_torch, a_ref, b_torch, b_ref, sfa_torch, sfa_ref, sfb_torch, sfb_ref = allocate_input_tensors(
        512,
        256,
        256,
        1,
        torch.float8_e5m2,
        torch.float8_e8m0fnu,
        32,
        "k",
        "k",
    )
    gemm_amax_api._cache_of_GemmAmaxSm100Objects.clear()
    original_export_aot = gemm_amax_api.GemmAmaxSm100.export_aot
    export_calls = 0

    def counting_export_aot(self, *args, **kwargs):
        nonlocal export_calls
        export_calls += 1
        return original_export_aot(self, *args, **kwargs)

    monkeypatch.setattr(gemm_amax_api.GemmAmaxSm100, "export_aot", counting_export_aot)
    monkeypatch.setenv("CUDNN_FE_AOT_MODE", "write")
    monkeypatch.setenv("CUDNN_FE_AOT_DIR", str(tmp_path))
    exported = gemm_amax_wrapper_sm100(a_torch, b_torch, sfa_torch, sfb_torch)

    metadata_files = list(tmp_path.glob("*.json"))
    assert len(metadata_files) == 1
    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    assert metadata["identity"]["kernel_name"] == "GemmAmaxSm100"
    assert metadata["symbol"].startswith("cudnnfe_GemmAmaxSm100_")
    check_ref_gemm_amax(a_ref, b_ref, sfa_ref, sfb_ref, exported["c_tensor"], exported["amax_tensor"])
    assert export_calls == 1

    cached = gemm_amax_wrapper_sm100(a_torch, b_torch, sfa_torch, sfb_torch)
    check_ref_gemm_amax(a_ref, b_ref, sfa_ref, sfb_ref, cached["c_tensor"], cached["amax_tensor"])
    assert export_calls == 1

    monkeypatch.setenv("CUDNN_FE_AOT_MODE", "read")
    loaded = gemm_amax_wrapper_sm100(a_torch, b_torch, sfa_torch, sfb_torch)

    check_ref_gemm_amax(a_ref, b_ref, sfa_ref, sfb_ref, loaded["c_tensor"], loaded["amax_tensor"])


"""
GemmAmax API with explicit set_params, compile, and execute paths. 
Use this method when running one static configuration for each GemmAmax object.
"""


def _test_gemm_amax_compile_execute(
    a_major,
    b_major,
    c_major,
    ab_dtype,
    sf_dtype,
    c_dtype,
    acc_dtype,
    sf_vec_size,
    mma_tiler_mn,
    cluster_shape_mn,
    request,
):
    try:
        from cudnn import GemmAmaxSm100
        from cuda.bindings import driver as cuda
        from fe_api.test_gemm_amax_utils import (
            allocate_input_tensors,
            allocate_output_tensors,
            check_ref_gemm_amax,
            gemm_amax_init,
        )
    except ImportError as e:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")
    cfg = gemm_amax_init(
        request,
        a_major,
        b_major,
        c_major,
        ab_dtype,
        sf_dtype,
        c_dtype,
        acc_dtype,
        sf_vec_size,
        mma_tiler_mn,
        cluster_shape_mn,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    a_torch, a_ref, b_torch, b_ref, sfa_torch, sfa_ref, sfb_torch, sfb_ref = allocate_input_tensors(
        cfg["m"],
        cfg["n"],
        cfg["k"],
        cfg["l"],
        cfg["ab_dtype"],
        cfg["sf_dtype"],
        cfg["sf_vec_size"],
        cfg["a_major"],
        cfg["b_major"],
    )
    c_torch, amax_torch = allocate_output_tensors(cfg["m"], cfg["n"], cfg["l"], cfg["c_dtype"], cfg["c_major"])

    gemm = GemmAmaxSm100(
        sample_a=a_torch,
        sample_b=b_torch,
        sample_sfa=sfa_torch,
        sample_sfb=sfb_torch,
        sample_c=c_torch,
        sample_amax=amax_torch,
        acc_dtype=cfg["acc_dtype"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sf_vec_size=cfg["sf_vec_size"],
    )
    try:
        assert gemm.check_support(), "Unsupported testcase"
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    gemm.compile()
    gemm.execute(
        a_tensor=a_torch,
        b_tensor=b_torch,
        sfa_tensor=sfa_torch,
        sfb_tensor=sfb_torch,
        c_tensor=c_torch,
        amax_tensor=amax_torch,
        current_stream=stream,
    )

    check_ref_gemm_amax(a_ref, b_ref, sfa_ref, sfb_ref, c_torch, amax_torch, skip_ref=cfg["skip_ref"])


"""
GemmAmax API with gemm_amax_wrapper:
Use the wrapper to directly call GemmAmax without explicit setup and compilation.
"""


def _test_gemm_amax_wrapper(
    a_major,
    b_major,
    c_major,
    ab_dtype,
    sf_dtype,
    c_dtype,
    acc_dtype,
    sf_vec_size,
    mma_tiler_mn,
    cluster_shape_mn,
    request,
):
    try:
        from cudnn import gemm_amax_wrapper_sm100
        from cuda.bindings import driver as cuda
        from fe_api.test_gemm_amax_utils import (
            allocate_input_tensors,
            allocate_output_tensors,
            check_ref_gemm_amax,
            gemm_amax_init,
        )
    except ImportError as e:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")
    cfg = gemm_amax_init(
        request,
        a_major,
        b_major,
        c_major,
        ab_dtype,
        sf_dtype,
        c_dtype,
        acc_dtype,
        sf_vec_size,
        mma_tiler_mn,
        cluster_shape_mn,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    a_torch, a_ref, b_torch, b_ref, sfa_torch, sfa_ref, sfb_torch, sfb_ref = allocate_input_tensors(
        cfg["m"],
        cfg["n"],
        cfg["k"],
        cfg["l"],
        cfg["ab_dtype"],
        cfg["sf_dtype"],
        cfg["sf_vec_size"],
        cfg["a_major"],
        cfg["b_major"],
    )

    try:
        for _ in range(2):  # Run twice to test caching path
            c_torch, amax_torch = gemm_amax_wrapper_sm100(
                a_tensor=a_torch,
                b_tensor=b_torch,
                sfa_tensor=sfa_torch,
                sfb_tensor=sfb_torch,
                c_major=cfg["c_major"],
                c_dtype=cfg["c_dtype"],
                acc_dtype=cfg["acc_dtype"],
                mma_tiler_mn=cfg["mma_tiler_mn"],
                cluster_shape_mn=cfg["cluster_shape_mn"],
                sf_vec_size=cfg["sf_vec_size"],
                stream=stream,
            )
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    check_ref_gemm_amax(a_ref, b_ref, sfa_ref, sfb_ref, c_torch, amax_torch, skip_ref=cfg["skip_ref"])
