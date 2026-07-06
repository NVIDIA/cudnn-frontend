from .gemm_proj_rope_mxfp8 import (
    gemm_proj_rope_mxfp8_host,
    NUM_HEADS,
    HEAD_DIM,
    BLOCK,
    TILE_M,
)

from cuda.bindings import driver as cuda
import logging
import torch
from typing import Optional

import cutlass
import cutlass.utils
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from cudnn.api_base import APIBase, TupleDict


class GemmProjRopeMxfp8Sm100(APIBase):
    """Fused projection GEMM + per-head YARN RoPE + dual-direction MXFP8 quantize (SM100).
    """

    def __init__(
        self,
        sample_x: torch.Tensor,
        sample_w: torch.Tensor,
        sample_cos: torch.Tensor,
        sample_sin: torch.Tensor,
        sample_q_fp8_row: torch.Tensor,
        sample_q_scales_row: torch.Tensor,
        sample_q_fp8_col: torch.Tensor,
        sample_q_scales_col: torch.Tensor,
        w_out_in: bool = False,
    ):
        super().__init__()

        self._warn_experimental_api()
        self._logger.debug("Entering __init__")

        self.x_desc = self._make_tensor_desc(sample_x, name="sample_x")
        self.w_desc = self._make_tensor_desc(sample_w, name="sample_w")
        self.cos_desc = self._make_tensor_desc(sample_cos, name="sample_cos")
        self.sin_desc = self._make_tensor_desc(sample_sin, name="sample_sin")
        self.q_fp8_row_desc = self._make_tensor_desc(sample_q_fp8_row, name="sample_q_fp8_row")
        self.q_scales_row_desc = self._make_tensor_desc(sample_q_scales_row, name="sample_q_scales_row")
        self.q_fp8_col_desc = self._make_tensor_desc(sample_q_fp8_col, name="sample_q_fp8_col")
        self.q_scales_col_desc = self._make_tensor_desc(sample_q_scales_col, name="sample_q_scales_col")

        self.w_out_in = bool(w_out_in)
        self.tokens = int(sample_x.shape[0])

        # The cute program is traced from real sample tensors at compile() time; kept only
        # until then, then released (mirrors the sample_* teardown in the sibling APIs).
        self._samples = (
            sample_x, sample_w, sample_cos, sample_sin,
            sample_q_fp8_row, sample_q_scales_row, sample_q_fp8_col, sample_q_scales_col,
        )
        self._logger.debug(
            f"__init__ completed: x {self.x_desc.shape}, w {self.w_desc.shape}, "
            f"w_out_in {self.w_out_in}"
        )

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        self._check_dtype(self.x_desc, dtype=torch.bfloat16, name="x")
        self._check_dtype(self.w_desc, dtype=torch.bfloat16, name="w")
        self._check_dtype(self.cos_desc, dtype=torch.bfloat16, name="cos")
        self._check_dtype(self.sin_desc, dtype=torch.bfloat16, name="sin")
        self._check_dtype(self.q_fp8_row_desc, dtype=torch.float8_e4m3fn, name="q_fp8_row")
        self._check_dtype(self.q_fp8_col_desc, dtype=torch.float8_e4m3fn, name="q_fp8_col")
        self._check_dtype(self.q_scales_row_desc, dtype=torch.uint8, name="q_scales_row")
        self._check_dtype(self.q_scales_col_desc, dtype=torch.uint8, name="q_scales_col")

        self._value_error_if(
            self.tokens % TILE_M != 0,
            f"tokens ({self.tokens}) must be a multiple of TILE_M ({TILE_M})",
        )

        self._logger.debug("Checking environment")
        self._runtime_error_if(not torch.cuda.is_available(), "CUDA is not available")
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        compute_capability = major * 10 + minor
        self._runtime_error_if(
            compute_capability < 100,
            f"GemmProjRopeMxfp8 requires SM100+ compute capability, but found "
            f"SM{compute_capability} on device {device}",
        )

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def _to_cute_tensors(
        self,
        x, w, cos, sin, q_fp8_row, q_scales_row, q_fp8_col, q_scales_col,
    ):
        """Wrap the torch tensors as (layout-dynamic) cute tensors for the host program.

        ``w`` may be stored [in, out] (default) or the TE-native [out, in] (``w_out_in``);
        both are presented to the kernel as the logical [out, in] B operand.
        """
        mA = from_dlpack(x.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=1)
        if self.w_out_in:
            mB = from_dlpack(w.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=1)
        else:
            mB = from_dlpack(w.detach().transpose(0, 1), assumed_align=16).mark_layout_dynamic(leading_dim=0)
        mCos = from_dlpack(cos.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=1)
        mSin = from_dlpack(sin.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=1)
        mQrow = from_dlpack(q_fp8_row, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        mSrow = from_dlpack(q_scales_row, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        mQcol = from_dlpack(q_fp8_col, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        mScol = from_dlpack(q_scales_col, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        return mA, mB, mCos, mSin, mQrow, mSrow, mQcol, mScol

    def compile(self) -> None:
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            self._logger.debug("Kernel already compiled; skipping recompilation")
            return

        cute_tensors = self._to_cute_tensors(*self._samples)
        grid_m = self.tokens // TILE_M
        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(1)
        swizzle_size = 8
        # Trace/compile on the current stream (a runtime argument; execute() supplies its own).
        compile_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        self._logger.debug("Compiling gemm_proj_rope_mxfp8_host")
        self._compiled_kernel = cute.compile(
            gemm_proj_rope_mxfp8_host,
            *cute_tensors,
            grid_m,
            max_active_clusters,
            swizzle_size,
            compile_stream,
        )

        self._samples = None
        self._logger.debug("Kernel compiled successfully")

    def execute(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        q_fp8_row: torch.Tensor,
        q_scales_row: torch.Tensor,
        q_fp8_col: torch.Tensor,
        q_scales_col: torch.Tensor,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        self._logger.debug("Entering execute")
        current_stream = self._get_default_stream(current_stream)

        self._runtime_error_if(
            self._compiled_kernel is None,
            "GemmProjRopeMxfp8Sm100 kernel not compiled; call compile() first",
        )

        cute_tensors = self._to_cute_tensors(
            x, w, cos, sin, q_fp8_row, q_scales_row, q_fp8_col, q_scales_col
        )
        self._compiled_kernel(*cute_tensors, current_stream)
        self._logger.debug("Executed with compiled kernel successfully")


_logger = logging.getLogger(__name__)
_cache_of_GemmProjRopeMxfp8Sm100Objects = {}


def gemm_proj_rope_mxfp8_wrapper_sm100(
    x: torch.Tensor,
    w: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    w_out_in: bool = False,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Allocate outputs, (compile-and-)cache the kernel, run it, and return the MXFP8 Q tensors.

    Args:
        x: ``[tokens, Q_LORA]`` bf16 activations (``tokens % TILE_M == 0``).
        w: projection weight -- ``[Q_LORA, NUM_HEADS*HEAD_DIM]`` (``w_out_in=False``) or the
            TE-native transposed ``[NUM_HEADS*HEAD_DIM, Q_LORA]`` (``w_out_in=True``).
        cos, sin: ``[tokens, QK_ROPE]`` bf16 rotary tables.
        w_out_in: whether ``w`` is stored ``[out, in]``.
        stream: optional CUDA stream; defaults to the current torch stream.

    Returns:
        ``TupleDict(q_fp8_row, q_scales_row, q_fp8_col, q_scales_col)`` -- rowwise / columnwise
        MXFP8 (E4M3) data and E8M0 scales.
    """
    tokens = x.shape[0]
    device = x.device

    q_fp8_row = torch.empty(tokens, NUM_HEADS, HEAD_DIM, dtype=torch.float8_e4m3fn, device=device)
    q_scales_row = torch.empty(tokens, NUM_HEADS, HEAD_DIM // BLOCK, dtype=torch.uint8, device=device)
    q_fp8_col = torch.empty(tokens, NUM_HEADS, HEAD_DIM, dtype=torch.float8_e4m3fn, device=device)
    q_scales_col = torch.empty(tokens // BLOCK, NUM_HEADS, HEAD_DIM, dtype=torch.uint8, device=device)

    cache_key = (
        tuple(x.shape),
        tuple(w.shape),
        x.dtype,
        w.dtype,
        cos.dtype,
        bool(w_out_in),
    )
    if cache_key in _cache_of_GemmProjRopeMxfp8Sm100Objects:
        _logger.debug("gemm_proj_rope_mxfp8_wrapper_sm100: using cached GemmProjRopeMxfp8Sm100 object")
        obj = _cache_of_GemmProjRopeMxfp8Sm100Objects[cache_key]
    else:
        _logger.debug("gemm_proj_rope_mxfp8_wrapper_sm100: creating new GemmProjRopeMxfp8Sm100 object")
        obj = GemmProjRopeMxfp8Sm100(
            sample_x=x,
            sample_w=w,
            sample_cos=cos,
            sample_sin=sin,
            sample_q_fp8_row=q_fp8_row,
            sample_q_scales_row=q_scales_row,
            sample_q_fp8_col=q_fp8_col,
            sample_q_scales_col=q_scales_col,
            w_out_in=w_out_in,
        )
        assert obj.check_support()
        obj.compile()
        _cache_of_GemmProjRopeMxfp8Sm100Objects[cache_key] = obj

    obj.execute(
        x, w, cos, sin, q_fp8_row, q_scales_row, q_fp8_col, q_scales_col, current_stream=stream
    )

    return TupleDict(
        q_fp8_row=q_fp8_row,
        q_scales_row=q_scales_row,
        q_fp8_col=q_fp8_col,
        q_scales_col=q_scales_col,
    )
