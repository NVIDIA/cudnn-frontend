# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend-only API for the fused projection GEMM + YARN RoPE + dual-direction MXFP8 quantize.

The operation (GEMM -> per-head RoPE on the trailing 64 -> rowwise+columnwise MXFP8 output) has two
kernels that differ only in the GEMM *input* precision, exposed as sibling APIBase classes:

  * ``GemmProjRopeMxfp8Bf16InSm100``  -- BF16 inputs (x, w bf16), BF16 GEMM.
  * ``GemmProjRopeMxfp8Mxfp8InSm100`` -- MXFP8 inputs (E4M3 codes + E8M0 block scales), MXFP8 GEMM.

``gemm_proj_rope_mxfp8_wrapper_sm100`` selects between them by the dtype of ``x``/``w`` (which must
match), allocates outputs, drives the class lifecycle, and returns a ``TupleDict``.
"""

from .gemm_proj_rope_mxfp8_bf16in import (
    gemm_proj_rope_mxfp8_host as _bf16in_host,
    HEAD_DIM,
    QK_ROPE,
    BLOCK,
    TILE_M,
)
from .gemm_proj_rope_mxfp8_mxfp8in import (
    gemm_proj_rope_mxfp8_host as _mxfp8in_host,
    _as_e8m0 as _mxfp8_as_e8m0,
)

from cuda.bindings import driver as cuda
import logging
import torch
from typing import Optional

import cutlass.utils
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from cudnn.api_base import APIBase, TensorDesc, TupleDict


# ======================================================================================
# BF16-input sibling (bf16 GEMM)
# ======================================================================================
class GemmProjRopeMxfp8Bf16InSm100(APIBase):
    """Fused BF16-input projection GEMM + per-head YARN RoPE + dual-direction MXFP8 quantize (SM100)."""

    def __init__(
        self,
        sample_x: torch.Tensor,
        sample_w: torch.Tensor,
        sample_cos: torch.Tensor,
        sample_sin: torch.Tensor,
        sample_out_fp8_row: torch.Tensor,
        sample_out_scales_row: torch.Tensor,
        sample_out_fp8_col: torch.Tensor,
        sample_out_scales_col: torch.Tensor,
        w_out_in: bool = False,
    ):
        super().__init__()
        self._warn_experimental_api()
        self._logger.debug("Entering __init__")

        self.x_desc = self._make_tensor_desc(sample_x, name="sample_x")
        self.w_desc = self._make_tensor_desc(sample_w, name="sample_w")
        self.cos_desc = self._make_tensor_desc(sample_cos, name="sample_cos")
        self.sin_desc = self._make_tensor_desc(sample_sin, name="sample_sin")
        self.out_fp8_row_desc = self._make_tensor_desc(sample_out_fp8_row, name="sample_out_fp8_row")
        self.out_scales_row_desc = self._make_tensor_desc(sample_out_scales_row, name="sample_out_scales_row")
        self.out_fp8_col_desc = self._make_tensor_desc(sample_out_fp8_col, name="sample_out_fp8_col")
        self.out_scales_col_desc = self._make_tensor_desc(sample_out_scales_col, name="sample_out_scales_col")

        self.w_out_in = bool(w_out_in)
        self.tokens = int(sample_x.shape[0])
        proj_dim = int(sample_w.shape[0] if self.w_out_in else sample_w.shape[1])
        self.num_heads = proj_dim // HEAD_DIM

        self._samples = (
            sample_x,
            sample_w,
            sample_cos,
            sample_sin,
            sample_out_fp8_row,
            sample_out_scales_row,
            sample_out_fp8_col,
            sample_out_scales_col,
        )
        self._logger.debug(f"__init__ completed: x {self.x_desc.shape}, w {self.w_desc.shape}, w_out_in {self.w_out_in}")

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        self._check_dtype(self.x_desc, dtype=torch.bfloat16, name="x")
        self._check_dtype(self.w_desc, dtype=torch.bfloat16, name="w")
        self._check_dtype(self.cos_desc, dtype=torch.bfloat16, name="cos")
        self._check_dtype(self.sin_desc, dtype=torch.bfloat16, name="sin")
        self._check_dtype(self.out_fp8_row_desc, dtype=torch.float8_e4m3fn, name="out_fp8_row")
        self._check_dtype(self.out_fp8_col_desc, dtype=torch.float8_e4m3fn, name="out_fp8_col")
        self._check_dtype(self.out_scales_row_desc, dtype=torch.uint8, name="out_scales_row")
        self._check_dtype(self.out_scales_col_desc, dtype=torch.uint8, name="out_scales_col")

        self._value_error_if(
            self.tokens % TILE_M != 0,
            f"tokens ({self.tokens}) must be a multiple of TILE_M ({TILE_M})",
        )
        self._value_error_if(
            len(self.x_desc.shape) != 2 or self.x_desc.shape[0] != self.tokens,
            f"x must be [tokens, Q_LORA]; got {tuple(self.x_desc.shape)}",
        )
        proj_dim = self.w_desc.shape[0] if self.w_out_in else self.w_desc.shape[1]
        self._value_error_if(
            len(self.w_desc.shape) != 2 or proj_dim % HEAD_DIM != 0,
            f"w projected dim must be an integer multiple of HEAD_DIM ({HEAD_DIM}); got weight "
            f"shape {tuple(self.w_desc.shape)} with w_out_in={self.w_out_in}",
        )
        k_dim = self.w_desc.shape[1] if self.w_out_in else self.w_desc.shape[0]
        self._value_error_if(
            self.x_desc.shape[1] != k_dim,
            f"x contraction dim ({self.x_desc.shape[1]}) must match w's ({k_dim}); "
            f"x {tuple(self.x_desc.shape)}, w {tuple(self.w_desc.shape)}, w_out_in={self.w_out_in}",
        )
        for name, desc in (("cos", self.cos_desc), ("sin", self.sin_desc)):
            self._value_error_if(
                tuple(desc.shape) != (self.tokens, QK_ROPE),
                f"{name} must be [tokens, QK_ROPE] = [{self.tokens}, {QK_ROPE}]; got {tuple(desc.shape)}",
            )
        num_heads = self.num_heads
        expected = {
            "out_fp8_row": (self.out_fp8_row_desc, (self.tokens, num_heads, HEAD_DIM)),
            "out_scales_row": (self.out_scales_row_desc, (self.tokens, num_heads, HEAD_DIM // BLOCK)),
            "out_fp8_col": (self.out_fp8_col_desc, (self.tokens, num_heads, HEAD_DIM)),
            "out_scales_col": (self.out_scales_col_desc, (self.tokens // BLOCK, num_heads, HEAD_DIM)),
        }
        for name, (desc, shape) in expected.items():
            self._value_error_if(
                tuple(desc.shape) != shape,
                f"{name} must have shape {shape}; got {tuple(desc.shape)}",
            )

        _check_same_cuda_device(
            self,
            self.x_desc,
            self.w_desc,
            self.cos_desc,
            self.sin_desc,
            self.out_fp8_row_desc,
            self.out_scales_row_desc,
            self.out_fp8_col_desc,
            self.out_scales_col_desc,
        )
        _check_contiguous(
            self,
            x=self.x_desc,
            w=self.w_desc,
            cos=self.cos_desc,
            sin=self.sin_desc,
            out_fp8_row=self.out_fp8_row_desc,
            out_scales_row=self.out_scales_row_desc,
            out_fp8_col=self.out_fp8_col_desc,
            out_scales_col=self.out_scales_col_desc,
        )
        _check_sm100(self)

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def _to_cute_tensors(self, x, w, cos, sin, out_fp8_row, out_scales_row, out_fp8_col, out_scales_col):
        """``w`` may be [in, out] (default) or TE-native [out, in] (``w_out_in``); both present as [out, in]."""
        mA = from_dlpack(x.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=1)
        if self.w_out_in:
            mB = from_dlpack(w.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=1)
        else:
            mB = from_dlpack(w.detach().transpose(0, 1), assumed_align=16).mark_layout_dynamic(leading_dim=0)
        mCos = from_dlpack(cos.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=1)
        mSin = from_dlpack(sin.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=1)
        mQrow = from_dlpack(out_fp8_row, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        mSrow = from_dlpack(out_scales_row, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        mQcol = from_dlpack(out_fp8_col, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        mScol = from_dlpack(out_scales_col, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        return mA, mB, mCos, mSin, mQrow, mSrow, mQcol, mScol

    def compile(self) -> None:
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        cute_tensors = self._to_cute_tensors(*self._samples)
        grid_m = self.tokens // TILE_M
        max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(1)
        swizzle_size = 8
        compile_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self._compiled_kernel = cute.compile(
            _bf16in_host,
            *cute_tensors,
            grid_m,
            self.num_heads,
            max_active_clusters,
            swizzle_size,
            compile_stream,
        )
        self._samples = None
        self._logger.debug("Kernel compiled successfully")

    def execute(
        self,
        x,
        w,
        cos,
        sin,
        out_fp8_row,
        out_scales_row,
        out_fp8_col,
        out_scales_col,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        current_stream = self._get_default_stream(current_stream)
        self._runtime_error_if(
            self._compiled_kernel is None,
            "GemmProjRopeMxfp8Bf16InSm100 kernel not compiled; call compile() first",
        )
        cute_tensors = self._to_cute_tensors(x, w, cos, sin, out_fp8_row, out_scales_row, out_fp8_col, out_scales_col)
        self._compiled_kernel(*cute_tensors, current_stream)


# ======================================================================================
# MXFP8-input sibling (mxfp8 GEMM)
# ======================================================================================
class GemmProjRopeMxfp8Mxfp8InSm100(APIBase):
    """Fused MXFP8-input projection GEMM + per-head YARN RoPE + dual-direction MXFP8 quantize (SM100).

    Inputs are pre-quantized MXFP8: FP8 (E4M3) codes ``x_code``/``w_code`` plus E8M0 rowwise block
    scales ``x_scale``/``w_scale``. The weight is TE-native ``[out, in] = [N, K]`` (the wrapper
    transposes the [in, out] layout before constructing this API).
    """

    def __init__(
        self,
        sample_x_code: torch.Tensor,
        sample_x_scale: torch.Tensor,
        sample_w_code: torch.Tensor,
        sample_w_scale: torch.Tensor,
        sample_cos: torch.Tensor,
        sample_sin: torch.Tensor,
        sample_out_fp8_row: torch.Tensor,
        sample_out_scales_row: torch.Tensor,
        sample_out_fp8_col: torch.Tensor,
        sample_out_scales_col: torch.Tensor,
    ):
        super().__init__()
        self._warn_experimental_api()
        self._logger.debug("Entering __init__")

        self.x_code_desc = self._make_tensor_desc(sample_x_code, name="sample_x_code")
        self.x_scale_desc = self._make_tensor_desc(sample_x_scale, name="sample_x_scale")
        self.w_code_desc = self._make_tensor_desc(sample_w_code, name="sample_w_code")
        self.w_scale_desc = self._make_tensor_desc(sample_w_scale, name="sample_w_scale")
        self.cos_desc = self._make_tensor_desc(sample_cos, name="sample_cos")
        self.sin_desc = self._make_tensor_desc(sample_sin, name="sample_sin")
        self.out_fp8_row_desc = self._make_tensor_desc(sample_out_fp8_row, name="sample_out_fp8_row")
        self.out_scales_row_desc = self._make_tensor_desc(sample_out_scales_row, name="sample_out_scales_row")
        self.out_fp8_col_desc = self._make_tensor_desc(sample_out_fp8_col, name="sample_out_fp8_col")
        self.out_scales_col_desc = self._make_tensor_desc(sample_out_scales_col, name="sample_out_scales_col")

        self.tokens = int(sample_x_code.shape[0])
        proj_dim = int(sample_w_code.shape[0])  # weight is [N, K]
        self.k_dim = int(sample_w_code.shape[1])
        self.num_heads = proj_dim // HEAD_DIM

        self._samples = (
            sample_x_code,
            sample_x_scale,
            sample_w_code,
            sample_w_scale,
            sample_cos,
            sample_sin,
            sample_out_fp8_row,
            sample_out_scales_row,
            sample_out_fp8_col,
            sample_out_scales_col,
        )
        self._logger.debug(f"__init__ completed: x_code {self.x_code_desc.shape}, w_code {self.w_code_desc.shape}")

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        self._check_dtype(self.x_code_desc, dtype=torch.float8_e4m3fn, name="x_code")
        self._check_dtype(self.w_code_desc, dtype=torch.float8_e4m3fn, name="w_code")
        self._check_dtype(self.x_scale_desc, dtype=torch.uint8, name="x_scale")
        self._check_dtype(self.w_scale_desc, dtype=torch.uint8, name="w_scale")
        self._check_dtype(self.cos_desc, dtype=torch.bfloat16, name="cos")
        self._check_dtype(self.sin_desc, dtype=torch.bfloat16, name="sin")
        self._check_dtype(self.out_fp8_row_desc, dtype=torch.float8_e4m3fn, name="out_fp8_row")
        self._check_dtype(self.out_fp8_col_desc, dtype=torch.float8_e4m3fn, name="out_fp8_col")
        self._check_dtype(self.out_scales_row_desc, dtype=torch.uint8, name="out_scales_row")
        self._check_dtype(self.out_scales_col_desc, dtype=torch.uint8, name="out_scales_col")

        self._value_error_if(
            self.tokens % TILE_M != 0,
            f"tokens ({self.tokens}) must be a multiple of TILE_M ({TILE_M})",
        )
        self._value_error_if(
            len(self.w_code_desc.shape) != 2 or (self.num_heads * HEAD_DIM) != self.w_code_desc.shape[0],
            f"w_code must be [N, K] with N a multiple of HEAD_DIM ({HEAD_DIM}); got {tuple(self.w_code_desc.shape)}",
        )
        self._value_error_if(self.k_dim % BLOCK != 0, f"K ({self.k_dim}) must be a multiple of the MXFP8 block ({BLOCK})")
        self._value_error_if(
            len(self.x_code_desc.shape) != 2 or self.x_code_desc.shape != (self.tokens, self.k_dim),
            f"x_code must be [tokens, K] = [{self.tokens}, {self.k_dim}]; got {tuple(self.x_code_desc.shape)}",
        )
        # E8M0 rowwise block scales: one uint8 per 32-wide block along K.
        self._value_error_if(
            tuple(self.x_scale_desc.shape) != (self.tokens, self.k_dim // BLOCK),
            f"x_scale must be [tokens, K//{BLOCK}] = [{self.tokens}, {self.k_dim // BLOCK}]; got {tuple(self.x_scale_desc.shape)}",
        )
        self._value_error_if(
            tuple(self.w_scale_desc.shape) != (self.num_heads * HEAD_DIM, self.k_dim // BLOCK),
            f"w_scale must be [N, K//{BLOCK}] = [{self.num_heads * HEAD_DIM}, {self.k_dim // BLOCK}]; " f"got {tuple(self.w_scale_desc.shape)}",
        )
        for name, desc in (("cos", self.cos_desc), ("sin", self.sin_desc)):
            self._value_error_if(
                tuple(desc.shape) != (self.tokens, QK_ROPE),
                f"{name} must be [tokens, QK_ROPE] = [{self.tokens}, {QK_ROPE}]; got {tuple(desc.shape)}",
            )
        num_heads = self.num_heads
        expected = {
            "out_fp8_row": (self.out_fp8_row_desc, (self.tokens, num_heads, HEAD_DIM)),
            "out_scales_row": (self.out_scales_row_desc, (self.tokens, num_heads, HEAD_DIM // BLOCK)),
            "out_fp8_col": (self.out_fp8_col_desc, (self.tokens, num_heads, HEAD_DIM)),
            "out_scales_col": (self.out_scales_col_desc, (self.tokens // BLOCK, num_heads, HEAD_DIM)),
        }
        for name, (desc, shape) in expected.items():
            self._value_error_if(
                tuple(desc.shape) != shape,
                f"{name} must have shape {shape}; got {tuple(desc.shape)}",
            )

        # The SFB scale relay copies a 256-row (2*128) block shared across even/odd n_idx; the last
        # head stays in-bounds only when num_heads is even (the odd-n_idx -QK_ROPE shift covers the
        # pair). An odd num_heads reads past w_scale's [num_heads*HEAD_DIM, K//32] bounds.
        self._value_error_if(
            num_heads % 2 != 0,
            f"num_heads ({num_heads}) must be even for the MXFP8-input kernel (SFB scale-relay pairing)",
        )
        _check_same_cuda_device(
            self,
            self.x_code_desc,
            self.x_scale_desc,
            self.w_code_desc,
            self.w_scale_desc,
            self.cos_desc,
            self.sin_desc,
            self.out_fp8_row_desc,
            self.out_scales_row_desc,
            self.out_fp8_col_desc,
            self.out_scales_col_desc,
        )
        _check_contiguous(
            self,
            x_code=self.x_code_desc,
            x_scale=self.x_scale_desc,
            w_code=self.w_code_desc,
            w_scale=self.w_scale_desc,
            cos=self.cos_desc,
            sin=self.sin_desc,
            out_fp8_row=self.out_fp8_row_desc,
            out_scales_row=self.out_scales_row_desc,
            out_fp8_col=self.out_fp8_col_desc,
            out_scales_col=self.out_scales_col_desc,
        )
        _check_sm100(self)

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def _grid_params(self):
        grid_m = self.tokens // TILE_M
        t2r_x8 = self.tokens >= 2048
        limit = min(grid_m, self.num_heads)
        if limit < 4:
            swizzle_size = 4
        else:
            v = 1
            while v * 2 <= limit:
                v *= 2
            swizzle_size = v
        return grid_m, t2r_x8, swizzle_size

    def _to_cute_tensors(self, x_code, x_scale, w_code, w_scale, cos, sin, out_fp8_row, out_scales_row, out_fp8_col, out_scales_col):
        mA = from_dlpack(x_code.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=1)
        mSFA = _mxfp8_as_e8m0(x_scale)
        mB = from_dlpack(w_code.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=1)
        mSFB = _mxfp8_as_e8m0(w_scale)
        mCos = from_dlpack(cos.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=1)
        mSin = from_dlpack(sin.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=1)
        mQrow = from_dlpack(out_fp8_row, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        mSrow = from_dlpack(out_scales_row, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        mQcol = from_dlpack(out_fp8_col, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        mScol = from_dlpack(out_scales_col, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        return mA, mSFA, mB, mSFB, mCos, mSin, mQrow, mSrow, mQcol, mScol

    def compile(self) -> None:
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        cute_tensors = self._to_cute_tensors(*self._samples)
        grid_m, t2r_x8, swizzle_size = self._grid_params()
        max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(1)
        k_scale_words = self.k_dim // 128  # compact-scale uint32 words per row = K // 128 (deduced, not hardcoded)
        compile_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self._compiled_kernel = cute.compile(
            _mxfp8in_host,
            *cute_tensors,
            grid_m,
            self.num_heads,
            max_active_clusters,
            swizzle_size,
            t2r_x8,
            k_scale_words,
            compile_stream,
        )
        self._samples = None
        self._logger.debug("Kernel compiled successfully")

    def execute(
        self,
        x_code,
        x_scale,
        w_code,
        w_scale,
        cos,
        sin,
        out_fp8_row,
        out_scales_row,
        out_fp8_col,
        out_scales_col,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        current_stream = self._get_default_stream(current_stream)
        self._runtime_error_if(
            self._compiled_kernel is None,
            "GemmProjRopeMxfp8Mxfp8InSm100 kernel not compiled; call compile() first",
        )
        cute_tensors = self._to_cute_tensors(
            x_code,
            x_scale,
            w_code,
            w_scale,
            cos,
            sin,
            out_fp8_row,
            out_scales_row,
            out_fp8_col,
            out_scales_col,
        )
        self._compiled_kernel(*cute_tensors, current_stream)


# ======================================================================================
# shared check helpers
# ======================================================================================
def _check_same_cuda_device(api, *descs):
    devices = {d.device for d in descs}
    api._value_error_if(
        len(devices) != 1 or next(iter(devices)).type != "cuda",
        f"all tensors must be on a single CUDA device; got devices {sorted(str(d) for d in devices)}",
    )


def _check_contiguous(api, **named_descs):
    # The epilogue writes/reads with hardcoded row strides (cos/sin: QK_ROPE*2; outputs:
    # num_heads*HEAD_DIM) and the TMA / scale-relay assume packed operands, so every tensor must be
    # C-contiguous -- a strided/sliced view would silently read/write the wrong addresses.
    for name, desc in named_descs.items():
        api._value_error_if(
            tuple(desc.stride) != TensorDesc._compute_contiguous_stride(tuple(desc.shape)),
            f"{name} must be C-contiguous; got shape {tuple(desc.shape)} stride {tuple(desc.stride)}",
        )


def _check_sm100(api):
    api._runtime_error_if(not torch.cuda.is_available(), "CUDA is not available")
    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    compute_capability = major * 10 + minor
    api._runtime_error_if(
        compute_capability < 100,
        f"GemmProjRopeMxfp8 requires SM100+ compute capability, but found SM{compute_capability} on device {device}",
    )


# ======================================================================================
# dtype-dispatch wrapper
# ======================================================================================
_logger = logging.getLogger(__name__)
_bf16in_obj_cache = {}
_mxfp8in_obj_cache = {}


def gemm_proj_rope_mxfp8_wrapper_sm100(
    x: torch.Tensor,
    w: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    x_scale: Optional[torch.Tensor] = None,
    w_scale: Optional[torch.Tensor] = None,
    w_out_in: bool = True,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Allocate outputs, dispatch to the bf16- or mxfp8-input API by x/w dtype, and return MXFP8 Q.

    Args:
        x: ``[tokens, K]`` activations. ``bfloat16`` -> BF16 GEMM; ``float8_e4m3fn`` -> MXFP8 GEMM.
        w: projection weight, SAME dtype as x. ``[out, in]`` (``w_out_in=True``, default / TE-native)
            or ``[in, out]`` (``w_out_in=False``).
        cos, sin: ``[tokens, QK_ROPE]`` bfloat16 rotary tables.
        x_scale, w_scale: E8M0 rowwise block scales for the MXFP8 path (``x_scale [tokens, K//32]``,
            ``w_scale [out, K//32]``, uint8). Must be None for bf16; required for float8_e4m3fn.
        w_out_in: whether w is stored ``[out, in]`` (also transposes w_scale on the MXFP8 path).
        stream: optional CUDA stream; defaults to the current torch stream.

    Returns:
        ``TupleDict(out_fp8_row, out_scales_row, out_fp8_col, out_scales_col)``.
    """
    assert x.dtype == w.dtype, f"x and w must share a dtype (both bfloat16 or both float8_e4m3fn); got x {x.dtype}, w {w.dtype}"

    tokens = x.shape[0]
    device = x.device
    proj_dim = w.shape[0] if w_out_in else w.shape[1]
    num_heads = proj_dim // HEAD_DIM

    out_fp8_row = torch.empty(tokens, num_heads, HEAD_DIM, dtype=torch.float8_e4m3fn, device=device)
    out_scales_row = torch.empty(tokens, num_heads, HEAD_DIM // BLOCK, dtype=torch.uint8, device=device)
    out_fp8_col = torch.empty(tokens, num_heads, HEAD_DIM, dtype=torch.float8_e4m3fn, device=device)
    out_scales_col = torch.empty(tokens // BLOCK, num_heads, HEAD_DIM, dtype=torch.uint8, device=device)

    if x.dtype == torch.bfloat16:
        assert x_scale is None and w_scale is None, "bf16 inputs must not be given MXFP8 scales (x_scale/w_scale); those are for the float8_e4m3fn path"
        key = (tuple(x.shape), tuple(w.shape), bool(w_out_in), device)
        obj = _bf16in_obj_cache.get(key)
        if obj is None:
            obj = GemmProjRopeMxfp8Bf16InSm100(
                sample_x=x,
                sample_w=w,
                sample_cos=cos,
                sample_sin=sin,
                sample_out_fp8_row=out_fp8_row,
                sample_out_scales_row=out_scales_row,
                sample_out_fp8_col=out_fp8_col,
                sample_out_scales_col=out_scales_col,
                w_out_in=w_out_in,
            )
            assert obj.check_support()
            obj.compile()
            _bf16in_obj_cache[key] = obj
        obj.execute(x, w, cos, sin, out_fp8_row, out_scales_row, out_fp8_col, out_scales_col, current_stream=stream)

    elif x.dtype == torch.float8_e4m3fn:
        assert x_scale is not None and w_scale is not None, "MXFP8 (float8_e4m3fn) inputs require x_scale and w_scale (E8M0 rowwise block scales)"
        # the mxfp8in kernel expects the weight as [out, in]; transpose code + scale for [in, out].
        if w_out_in:
            wc, ws = w, w_scale
        else:
            wc, ws = w.T.contiguous(), w_scale.T.contiguous()
        key = (tuple(x.shape), tuple(wc.shape), device)
        obj = _mxfp8in_obj_cache.get(key)
        if obj is None:
            obj = GemmProjRopeMxfp8Mxfp8InSm100(
                sample_x_code=x,
                sample_x_scale=x_scale,
                sample_w_code=wc,
                sample_w_scale=ws,
                sample_cos=cos,
                sample_sin=sin,
                sample_out_fp8_row=out_fp8_row,
                sample_out_scales_row=out_scales_row,
                sample_out_fp8_col=out_fp8_col,
                sample_out_scales_col=out_scales_col,
            )
            assert obj.check_support()
            obj.compile()
            _mxfp8in_obj_cache[key] = obj
        obj.execute(
            x,
            x_scale,
            wc,
            ws,
            cos,
            sin,
            out_fp8_row,
            out_scales_row,
            out_fp8_col,
            out_scales_col,
            current_stream=stream,
        )

    else:
        raise AssertionError(f"unsupported input dtype {x.dtype}; expected bfloat16 (BF16 GEMM) or float8_e4m3fn (MXFP8 GEMM)")

    return TupleDict(
        out_fp8_row=out_fp8_row,
        out_scales_row=out_scales_row,
        out_fp8_col=out_fp8_col,
        out_scales_col=out_scales_col,
    )
