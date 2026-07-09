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


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_gemm_proj_rope_mxfp8_check_support_rejects_kdim_mismatch(request):
    """check_support() must reject x whose contraction dim does not match the weight's."""
    try:
        import torch

        from cudnn import GemmProjRopeMxfp8Sm100
        from fe_api.test_gemm_proj_rope_mxfp8_utils import (
            Q_LORA,
            allocate_input_tensors,
            allocate_output_tensors,
            gemm_proj_rope_mxfp8_init,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    gemm_proj_rope_mxfp8_init(request, tokens=2048, w_out_in=False)  # arch skip if needed
    tokens = 2048
    x, w, cos, sin = allocate_input_tensors(tokens, w_out_in=False)
    x = torch.randn(tokens, Q_LORA + 8, dtype=x.dtype, device=x.device)  # inner dim != w's K
    out_fp8_row, out_scales_row, out_fp8_col, out_scales_col = allocate_output_tensors(tokens)
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


@pytest.mark.L0
def test_gemm_proj_rope_mxfp8_run_rejects_kdim_mismatch():
    """run() is a public entry that skips check_support(); it must still reject an x
    whose GEMM contraction dim does not match the weight's, before reaching the kernel.

    The structural checks at the top of run() fire before any CUDA work, so plain CPU
    tensors are enough to exercise the rejection (no SM100 device required)."""
    try:
        from cudnn.gemm_proj_rope_mxfp8 import run
        from fe_api.test_gemm_proj_rope_mxfp8_utils import (
            BLOCK,
            HEAD_DIM,
            NUM_HEADS,
            QK_ROPE,
            Q_LORA,
            Q_OUT,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    tokens = 128  # a multiple of TILE_M so validation reaches the k-dim check
    x = torch.empty(tokens, Q_LORA + 8, dtype=torch.bfloat16)  # inner dim != w's K
    w = torch.empty(Q_LORA, Q_OUT, dtype=torch.bfloat16)  # [in, out], K = Q_LORA
    cos = torch.empty(tokens, QK_ROPE, dtype=torch.bfloat16)
    sin = torch.empty(tokens, QK_ROPE, dtype=torch.bfloat16)
    out_fp8_row = torch.empty(tokens, NUM_HEADS, HEAD_DIM, dtype=torch.float8_e4m3fn)
    out_scales_row = torch.empty(tokens, NUM_HEADS, HEAD_DIM // BLOCK, dtype=torch.uint8)
    out_fp8_col = torch.empty(tokens, NUM_HEADS, HEAD_DIM, dtype=torch.float8_e4m3fn)
    out_scales_col = torch.empty(tokens // BLOCK, NUM_HEADS, HEAD_DIM, dtype=torch.uint8)

    with pytest.raises(ValueError):
        run(x, w, cos, sin, out_fp8_row, out_scales_row, out_fp8_col, out_scales_col, w_out_in=False)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("tokens", [128, 256])
@pytest.mark.parametrize("w_out_in", [False, True])
def test_gemm_proj_rope_mxfp8_reference_contract(tokens, w_out_in):
    """Exercise the PyTorch reference oracle directly (no SM100 kernel required).

    Gives ``gemm_proj_rope_mxfp8_reference`` contract coverage for both weight
    layouts and valid token multiples: the four outputs must have the documented
    shapes/dtypes and carry only finite (dequantized) values."""
    try:
        from cudnn.gemm_proj_rope_mxfp8 import gemm_proj_rope_mxfp8_reference
        from fe_api.test_gemm_proj_rope_mxfp8_utils import (
            BLOCK,
            HEAD_DIM,
            NUM_HEADS,
            QK_ROPE,
            Q_LORA,
            Q_OUT,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(tokens, Q_LORA, dtype=torch.bfloat16, device=dev) * 0.5
    # w is stored [out, in] for w_out_in else [in, out]; both project Q_LORA -> Q_OUT.
    w_shape = (Q_OUT, Q_LORA) if w_out_in else (Q_LORA, Q_OUT)
    w = torch.randn(*w_shape, dtype=torch.bfloat16, device=dev) * 0.02
    cos = torch.randn(tokens, QK_ROPE, dtype=torch.bfloat16, device=dev)
    sin = torch.randn(tokens, QK_ROPE, dtype=torch.bfloat16, device=dev)

    qr, sr, qc, sc = gemm_proj_rope_mxfp8_reference(x, w, cos, sin, w_out_in=w_out_in)

    assert qr.shape == (tokens, NUM_HEADS, HEAD_DIM)
    assert sr.shape == (tokens, NUM_HEADS, HEAD_DIM // BLOCK)
    assert qc.shape == (tokens, NUM_HEADS, HEAD_DIM)
    assert sc.shape == (tokens // BLOCK, NUM_HEADS, HEAD_DIM)
    assert qr.dtype == torch.float8_e4m3fn and qc.dtype == torch.float8_e4m3fn
    assert sr.dtype == torch.uint8 and sc.dtype == torch.uint8
    # No NaN/Inf should leak through the e8m0/e4m3 quantization path.
    assert torch.isfinite(qr.float()).all() and torch.isfinite(qc.float()).all()


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("tokens", [128, 256])
def test_gemm_proj_rope_mxfp8_reference_w_out_in_equivalence(tokens):
    """The oracle must be layout-invariant: feeding the transposed weight with
    w_out_in=True reproduces the w_out_in=False result (both map to the same
    logical B=[out, in] operand)."""
    try:
        from cudnn.gemm_proj_rope_mxfp8 import gemm_proj_rope_mxfp8_reference
        from fe_api.test_gemm_proj_rope_mxfp8_utils import QK_ROPE, Q_LORA, Q_OUT
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(tokens, Q_LORA, dtype=torch.bfloat16, device=dev) * 0.5
    w_in_out = torch.randn(Q_LORA, Q_OUT, dtype=torch.bfloat16, device=dev) * 0.02
    cos = torch.randn(tokens, QK_ROPE, dtype=torch.bfloat16, device=dev)
    sin = torch.randn(tokens, QK_ROPE, dtype=torch.bfloat16, device=dev)

    ref_in_out = gemm_proj_rope_mxfp8_reference(x, w_in_out, cos, sin, w_out_in=False)
    ref_out_in = gemm_proj_rope_mxfp8_reference(x, w_in_out.t().contiguous(), cos, sin, w_out_in=True)

    names = ("out_fp8_row", "out_scales_row", "out_fp8_col", "out_scales_col")
    for name, a, b in zip(names, ref_in_out, ref_out_in):
        # Identical logical operands; allow a tiny fraction of e4m3 boundary flips from
        # differing fp32 matmul reduction orders, but the layouts must otherwise agree.
        frac_equal = (a.float() == b.float()).float().mean().item()
        assert frac_equal >= 0.999, f"{name} differs between w_out_in layouts: matched={frac_equal:.4f}"


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
