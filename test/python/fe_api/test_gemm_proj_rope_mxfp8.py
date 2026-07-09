import torch

import pytest

from test_utils import torch_fork_set_rng
from fe_api.test_gemm_proj_rope_mxfp8_utils import with_gemm_proj_rope_mxfp8_params


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_gemm_proj_rope_mxfp8_params
def test_gemm_proj_rope_mxfp8_compile_execute(tokens, w_out_in, request):
    _test_gemm_proj_rope_mxfp8_compile_execute(tokens=tokens, w_out_in=w_out_in, request=request)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_gemm_proj_rope_mxfp8_params
def test_gemm_proj_rope_mxfp8_wrapper(tokens, w_out_in, request):
    _test_gemm_proj_rope_mxfp8_wrapper(tokens=tokens, w_out_in=w_out_in, request=request)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_gemm_proj_rope_mxfp8_check_support_rejects_unaligned_tokens(request):
    """check_support() must reject a token count that is not a multiple of TILE_M."""
    try:
        from cudnn import GemmProjRopeMxfp8Sm100
        from fe_api.test_gemm_proj_rope_mxfp8_utils import (
            allocate_input_tensors,
            allocate_output_tensors,
            gemm_proj_rope_mxfp8_init,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    gemm_proj_rope_mxfp8_init(request, tokens=2048, w_out_in=False)  # arch skip if needed
    tokens = 2048 + 32  # not a multiple of TILE_M (128)
    x, w, cos, sin = allocate_input_tensors(tokens, w_out_in=False)
    out_fp8_row, out_scales_row, out_fp8_col, out_scales_col = allocate_output_tensors(tokens - 32)  # any outputs
    obj = GemmProjRopeMxfp8Sm100(
        sample_x=x,
        sample_w=w,
        sample_cos=cos,
        sample_sin=sin,
        sample_out_fp8_row=out_fp8_row,
        sample_out_scales_row=out_scales_row,
        sample_out_fp8_col=out_fp8_col,
        sample_out_scales_col=out_scales_col,
        w_out_in=False,
    )
    with pytest.raises(ValueError):
        obj.check_support()


"""
GemmProjRopeMxfp8 API with explicit check_support, compile, and execute paths.
Use this method when running one static configuration per object.
"""


def _test_gemm_proj_rope_mxfp8_compile_execute(tokens, w_out_in, request):
    try:
        from cudnn import GemmProjRopeMxfp8Sm100
        from cuda.bindings import driver as cuda
        from fe_api.test_gemm_proj_rope_mxfp8_utils import (
            allocate_input_tensors,
            allocate_output_tensors,
            check_ref_gemm_proj_rope_mxfp8,
            gemm_proj_rope_mxfp8_init,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = gemm_proj_rope_mxfp8_init(request, tokens, w_out_in)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    x, w, cos, sin = allocate_input_tensors(cfg["tokens"], cfg["w_out_in"])
    outputs = allocate_output_tensors(cfg["tokens"])

    gemm = GemmProjRopeMxfp8Sm100(
        sample_x=x,
        sample_w=w,
        sample_cos=cos,
        sample_sin=sin,
        sample_out_fp8_row=outputs[0],
        sample_out_scales_row=outputs[1],
        sample_out_fp8_col=outputs[2],
        sample_out_scales_col=outputs[3],
        w_out_in=cfg["w_out_in"],
    )
    try:
        assert gemm.check_support(), "Unsupported testcase"
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    gemm.compile()
    gemm.execute(x, w, cos, sin, *outputs, current_stream=stream)

    check_ref_gemm_proj_rope_mxfp8(x, w, cos, sin, outputs, cfg["w_out_in"], skip_ref=cfg["skip_ref"])


"""
GemmProjRopeMxfp8 API via the high-level wrapper (no explicit setup/compile).
"""


def _test_gemm_proj_rope_mxfp8_wrapper(tokens, w_out_in, request):
    try:
        from cudnn import gemm_proj_rope_mxfp8_wrapper_sm100
        from cuda.bindings import driver as cuda
        from fe_api.test_gemm_proj_rope_mxfp8_utils import (
            allocate_input_tensors,
            check_ref_gemm_proj_rope_mxfp8,
            gemm_proj_rope_mxfp8_init,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = gemm_proj_rope_mxfp8_init(request, tokens, w_out_in)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    x, w, cos, sin = allocate_input_tensors(cfg["tokens"], cfg["w_out_in"])

    try:
        for _ in range(2):  # run twice to exercise the caching path
            out = gemm_proj_rope_mxfp8_wrapper_sm100(x, w, cos, sin, w_out_in=cfg["w_out_in"], stream=stream)
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    # TupleDict supports both key access and tuple unpacking.
    outputs = (out["out_fp8_row"], out["out_scales_row"], out["out_fp8_col"], out["out_scales_col"])
    check_ref_gemm_proj_rope_mxfp8(x, w, cos, sin, outputs, cfg["w_out_in"], skip_ref=cfg["skip_ref"])
