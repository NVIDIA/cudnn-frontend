from typing import Tuple, Optional
import math

from cuda.bindings import driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32

from cudnn.api_base import APIBase
from cudnn.datatypes import _convert_to_cutlass_data_type

from ..utils import make_tensor_strided_like
from .fmha import BlackwellFusedMultiHeadAttentionForward
from . import fmha_helpers as fmha_utils


class CompressionAttention(APIBase):
    def __init__(
        self,
        sample_q: torch.Tensor,
        sample_k: torch.Tensor,
        sample_v: torch.Tensor,
        sample_o: torch.Tensor,
        sample_lse: Optional[torch.Tensor] = None,
        sample_cum_seqlen_q: Optional[torch.Tensor] = None,
        sample_cum_seqlen_k: Optional[torch.Tensor] = None,
        qk_acc_dtype: torch.dtype = torch.float32,
        pv_acc_dtype: torch.dtype = torch.float32,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        is_persistent: bool = False,
        scale_q: float = 1.0,
        scale_k: float = 1.0,
        scale_v: float = 1.0,
        inv_scale_o: float = 1.0,
        scale_softmax: Optional[float] = None,
    ):
        super().__init__()
        self._kernel = BlackwellFusedMultiHeadAttentionForward

        self._logger.warning("CompressionAttention is an experimental API")
        self._logger.debug("Entering __init__")

        self.sample_q = sample_q
        self.sample_k = sample_k
        self.sample_v = sample_v
        self.sample_o = sample_o
        self.sample_lse = sample_lse
        self.enable_lse = sample_lse is not None
        self.sample_cum_seqlen_q = sample_cum_seqlen_q
        self.sample_cum_seqlen_k = sample_cum_seqlen_k

        # Types and kernel configuration
        self.qk_acc_dtype_torch = qk_acc_dtype
        self.pv_acc_dtype_torch = pv_acc_dtype
        self.mma_tiler_mn = mma_tiler_mn
        self.is_persistent = is_persistent

        # Scale config
        self.scale_q = scale_q
        self.scale_k = scale_k
        self.scale_v = scale_v
        self.inv_scale_o = inv_scale_o
        self.scale_softmax = scale_softmax

        # Derived attributes (populated in check_support)
        self.batch_size = None
        self.s_q = None
        self.s_k = None
        self.h_q = None
        self.h_k = None
        self.h_r = None
        self.head_dim = None
        self.problem_size = None
        self._compiled_kernel = None

        self._logger.debug(
            f"__init__ completed with args: sample_q {sample_q.shape}, sample_k {sample_k.shape}, sample_v {sample_v.shape}, sample_o {sample_o.shape}, sample_lse {sample_lse.shape if sample_lse is not None else 'None'}, sample_cum_seqlen_q {sample_cum_seqlen_q.shape if sample_cum_seqlen_q is not None else 'None'}, sample_cum_seqlen_k {sample_cum_seqlen_k.shape if sample_cum_seqlen_k is not None else 'None'}, qk_acc_dtype {qk_acc_dtype}, pv_acc_dtype {pv_acc_dtype}, mma_tiler_mn {mma_tiler_mn}, is_persistent {is_persistent}, scale_q {scale_q}, scale_k {scale_k}, scale_v {scale_v}, inv_scale_o {inv_scale_o}, scale_softmax {scale_softmax}"
        )

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        # shape normalization and validation
        self._logger.debug("Checking shape normalization and validation")
        if self.sample_q.ndim == 4:
            self.input_layout = "B,H,S,D"

            b, h_qo, s_qo, d_qk = self.sample_q.shape
            b, h_kv, s_kv, d_qk = self.sample_k.shape
            b, h_kv, s_kv, d_v = self.sample_v.shape
            b, h_q, s_qo, d_v = self.sample_o.shape

            if self.sample_q.shape != (b, h_qo, s_qo, d_qk):
                raise ValueError(
                    f"Input shape mismatch: expected Q tensor shape {b, h_qo, s_qo, d_qk}, got {self.sample_q.shape}"
                )
            if self.sample_k.shape != (b, h_kv, s_kv, d_qk):
                raise ValueError(
                    f"Input shape mismatch: expected K tensor shape {b, h_kv, s_kv, d_qk}, got {self.sample_k.shape}"
                )
            if self.sample_v.shape != (b, h_kv, s_kv, d_v):
                raise ValueError(
                    f"Input shape mismatch: expected V tensor shape {b, h_kv, s_kv, d_v}, got {self.sample_v.shape}"
                )
            if self.sample_o.shape != (b, h_q, s_qo, d_v):
                raise ValueError(
                    f"Output shape mismatch: expected O tensor shape {b, h_q, s_qo, d_v}, got {self.sample_o.shape}"
                )
            if self.enable_lse:
                self.sample_lse = self._unpad_tensor_to_ndim(
                    self.sample_lse, 3, "sample_lse"
                )
                if self.sample_lse.shape != (b, h_q, s_qo):
                    raise ValueError(
                        f"Output shape mismatch: expected LSE tensor shape {b, h_q, s_qo}, got {self.sample_lse.shape}"
                    )
                if not self.sample_lse.is_contiguous():
                    raise ValueError("LSE tensor must be contiguous")
            if (
                self.sample_cum_seqlen_q is not None
                or self.sample_cum_seqlen_k is not None
            ):
                self._logger.warning(
                    "sample_cum_seqlen_q and sample_cum_seqlen_k are ignored for B,H,S,D layout"
                )

            # Shapes
            self.batch_size = b
            self.s_q = s_qo
            self.s_kv = s_kv
            self.h_q = h_q
            self.h_kv = h_kv
            self.h_r = h_q // h_kv
            self.head_dim = d_qk
        elif self.sample_q.ndim == 3:
            self.input_layout = "T,H,D"

            t, h_q, d_qk = self.sample_q.shape
            t_kv, h_kv, d_qk = self.sample_k.shape  # T has been compressed for K and V
            t_kv, h_kv, d_v = self.sample_v.shape
            t, h_q, d_v = self.sample_o.shape

            if self.sample_q.shape != (t, h_q, d_qk):
                raise ValueError(
                    f"Input shape mismatch: expected Q tensor shape {t, h_q, d_qk}, got {self.sample_q.shape}"
                )
            if self.sample_k.shape != (t_kv, h_kv, d_qk):
                raise ValueError(
                    f"Input shape mismatch: expected K tensor shape {t_kv, h_kv, d_qk}, got {self.sample_k.shape}"
                )
            if self.sample_v.shape != (t_kv, h_kv, d_v):
                raise ValueError(
                    f"Input shape mismatch: expected V tensor shape {t_kv, h_kv, d_v}, got {self.sample_v.shape}"
                )
            if self.sample_o.shape != (t, h_q, d_v):
                raise ValueError(
                    f"Output shape mismatch: expected O tensor shape {t, h_q, d_v}, got {self.sample_o.shape}"
                )
            if self.enable_lse:
                self.sample_lse = self._unpad_tensor_to_ndim(
                    self.sample_lse, 2, "sample_lse"
                )
                if self.sample_lse.shape != (t, h_q):
                    raise ValueError(
                        f"Output shape mismatch: expected LSE tensor shape {t, h_q}, got {self.sample_lse.shape}"
                    )

            if self.sample_cum_seqlen_q is None or self.sample_cum_seqlen_k is None:
                raise ValueError(
                    f"sample_cum_seqlen_q and sample_cum_seqlen_k must be provided for T,H,D layout, got {self.sample_cum_seqlen_q} and {self.sample_cum_seqlen_k}"
                )
            self.sample_cum_seqlen_q = self._unpad_tensor_to_ndim(
                self.sample_cum_seqlen_q, 1, "sample_cum_seqlen_q"
            )
            self.sample_cum_seqlen_k = self._unpad_tensor_to_ndim(
                self.sample_cum_seqlen_k, 1, "sample_cum_seqlen_k"
            )
            if self.sample_cum_seqlen_q.ndim != 1 or self.sample_cum_seqlen_k.ndim != 1:
                raise ValueError(
                    f"sample_cum_seqlen_q and sample_cum_seqlen_k must be 1D tensors, got {self.sample_cum_seqlen_q.ndim}D and {self.sample_cum_seqlen_k.ndim}D"
                )
            if self.sample_cum_seqlen_q.dtype not in {
                torch.int32,
                torch.int64,
            } or self.sample_cum_seqlen_k.dtype not in {torch.int32, torch.int64}:
                raise ValueError(
                    f"sample_cum_seqlen_q and sample_cum_seqlen_k must be int32 or int64, got {self.sample_cum_seqlen_q.dtype} and {self.sample_cum_seqlen_k.dtype}"
                )
            if len(self.sample_cum_seqlen_q) != len(self.sample_cum_seqlen_k):
                raise ValueError(
                    f"sample_cum_seqlen_q and sample_cum_seqlen_k must have the same length, got {len(self.sample_cum_seqlen_q)} and {len(self.sample_cum_seqlen_k)}"
                )

            self.batch_size = len(self.sample_cum_seqlen_q) - 1
            self.s_q = None
            self.s_kv = None
            self.h_q = h_q
            self.h_kv = h_kv
            self.h_r = h_q // h_kv
            self.head_dim = d_qk

        else:
            raise ValueError(
                f"Invalid input layout: sample_q must be rank-3 (T,H,D) or rank-4 (B,H,S,D), got {self.sample_q.ndim}"
            )
        if d_qk != d_v:
            raise ValueError("D_qk must match D_v")
        if d_qk not in {32, 64, 128}:
            raise ValueError("Head dimension D_qk must be 32, 64, or 128")
        if h_q % h_kv != 0:
            raise ValueError("H_q must be divisible by H_k (GQA/MQA constraint)")

        self._logger.debug("Checking dtypes")
        in_dtype = self.sample_q.dtype
        out_dtype = self.sample_o.dtype
        if self.sample_k.dtype != in_dtype or self.sample_v.dtype != in_dtype:
            raise ValueError(
                f"Inputs must have the same dtype, got K {self.sample_k.dtype}, V {self.sample_v.dtype} for Q {in_dtype}"
            )
        if in_dtype not in {torch.float16, torch.bfloat16, torch.float8_e4m3fn}:
            raise ValueError(
                f"Inputs must be Float16, BFloat16, or Float8E4M3FN, got {in_dtype}"
            )
        if out_dtype not in {torch.float16, torch.bfloat16, torch.float8_e4m3fn}:
            raise ValueError(
                f"Outputs must be Float16, BFloat16, or Float8E4M3FN, got {out_dtype}"
            )
        if self.qk_acc_dtype_torch not in {torch.float32}:
            raise ValueError(
                f"qk_acc_dtype must be Float32, got {self.qk_acc_dtype_torch}"
            )
        if self.pv_acc_dtype_torch not in {torch.float32}:
            raise ValueError(
                f"pv_acc_dtype must be Float32, got {self.pv_acc_dtype_torch}"
            )

        # Scale defaults
        if self.scale_softmax is None:
            self._logger.debug("No scale_softmax provided, using default 1/sqrt(d)")
            self.scale_softmax = 1.0 / math.sqrt(self.head_dim)

        # Environment checks
        self._logger.debug("Checking environment")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        compute_capability = major * 10 + minor
        if compute_capability < 100:
            raise RuntimeError(
                f"CompressionAttention requires SM100+ compute capability, but found SM{compute_capability} on device {device}"
            )
        if compute_capability == 103:
            raise RuntimeError("cuteDSL is not supported on SM103")

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def compile(self, current_stream: Optional[cuda.CUstream] = None) -> None:
        self._logger.debug("Entering compile")
        current_stream = self._get_default_stream(current_stream)
        self._ensure_support_checked()

        fmha_kernel = self._kernel(
            _convert_to_cutlass_data_type(self.qk_acc_dtype_torch),
            _convert_to_cutlass_data_type(self.pv_acc_dtype_torch),
            (*self.mma_tiler_mn, self.head_dim),
            self.is_persistent,
            mask_type=fmha_utils.MaskType.COMPRESSED_CAUSAL_MASK,
        )

        # Scales
        log2_e = math.log2(math.exp(1.0))
        scale_softmax = self.scale_q * self.scale_k * self.scale_softmax
        scale_softmax_log2 = scale_softmax * log2_e
        scale_output = self.scale_v * self.inv_scale_o

        s_q = (
            self.s_q
            if self.input_layout == "B,H,S,D"
            else max(self.sample_cum_seqlen_q).item()
        )
        s_kv = (
            self.s_kv
            if self.input_layout == "B,H,S,D"
            else max(self.sample_cum_seqlen_k).item()
        )
        self.problem_size = (
            self.batch_size,
            s_q,
            s_q,
            s_kv,
            self.h_q,
            self.h_kv,
            self.head_dim,
        )

        self._logger.debug("Compiling CompressionAttention kernel with cute.compile")
        self._compiled_kernel = cute.compile(
            fmha_kernel,
            q_iter=from_dlpack(self.sample_q, assumed_align=16).iterator,
            q_stride=(
                self.sample_q.transpose(1, 2).stride()
                if self.input_layout == "B,H,S,D"
                else (self.sample_q.stride()[0], *self.sample_q.stride())
            ),
            k_iter=from_dlpack(self.sample_k, assumed_align=16).iterator,
            k_stride=(
                self.sample_k.transpose(1, 2).stride()
                if self.input_layout == "B,H,S,D"
                else (self.sample_k.stride()[0], *self.sample_k.stride())
            ),
            v_iter=from_dlpack(self.sample_v, assumed_align=16).iterator,
            v_stride=(
                self.sample_v.transpose(1, 2).stride()
                if self.input_layout == "B,H,S,D"
                else (self.sample_v.stride()[0], *self.sample_v.stride())
            ),
            o_iter=from_dlpack(self.sample_o, assumed_align=16).iterator,
            o_stride=(
                self.sample_o.transpose(1, 2).stride()
                if self.input_layout == "B,H,S,D"
                else (self.sample_o.stride()[0], *self.sample_o.stride())
            ),
            problem_size=self.problem_size,
            cum_seqlen_q=(
                from_dlpack(self.sample_cum_seqlen_q, assumed_align=16)
                if self.input_layout == "T,H,D"
                else None
            ),
            cum_seqlen_k=(
                from_dlpack(self.sample_cum_seqlen_k, assumed_align=16)
                if self.input_layout == "T,H,D"
                else None
            ),
            lse_iter=(
                from_dlpack(self.sample_lse, assumed_align=16).iterator
                if self.enable_lse
                else None
            ),
            lse_stride=(
                self.sample_lse.transpose(1, 2).stride()
                if self.input_layout == "B,H,S,D"
                else (0, *self.sample_lse.stride())
            ),
            scale_softmax_log2=scale_softmax_log2,
            scale_softmax=scale_softmax,
            scale_output=scale_output,
            window_size_left=None,
            window_size_right=Int32(0),
            stream=current_stream,
        )
        self._logger.debug("Kernel compiled successfully")

    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        lse_tensor: Optional[torch.Tensor] = None,
        cum_seqlen_q_tensor: Optional[torch.Tensor] = None,
        cum_seqlen_k_tensor: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
        skip_compile: bool = False,
        scale_q: Optional[float] = None,
        scale_k: Optional[float] = None,
        scale_v: Optional[float] = None,
        inv_scale_o: Optional[float] = None,
        scale_softmax: Optional[float] = None,
    ) -> None:
        self._logger.debug("Entering execute")
        current_stream = self._get_default_stream(current_stream)

        if self.enable_lse:
            if lse_tensor is None:
                raise ValueError(
                    "kernel was compiled with lse_tensor provided, but lse_tensor was not provided during execute"
                )
            lse_tensor = self._unpad_tensor_to_ndim(
                lse_tensor, o_tensor.ndim - 1, "lse_tensor"
            )
        if self.input_layout == "T,H,D":
            if cum_seqlen_q_tensor is None or cum_seqlen_k_tensor is None:
                raise ValueError(
                    f"cum_seqlen_q_tensor and cum_seqlen_k_tensor must be provided for T,H,D layout, got {cum_seqlen_q_tensor} and {cum_seqlen_k_tensor}"
                )
            cum_seqlen_q_tensor = self._unpad_tensor_to_ndim(
                cum_seqlen_q_tensor, 1, "cum_seqlen_q_tensor"
            )
            cum_seqlen_k_tensor = self._unpad_tensor_to_ndim(
                cum_seqlen_k_tensor, 1, "cum_seqlen_k_tensor"
            )

        # Scale values
        scale_q = self.scale_q if scale_q is None else scale_q
        scale_k = self.scale_k if scale_k is None else scale_k
        scale_v = self.scale_v if scale_v is None else scale_v
        inv_scale_o = self.inv_scale_o if inv_scale_o is None else inv_scale_o
        scale_softmax = self.scale_softmax if scale_softmax is None else scale_softmax
        log2_e = math.log2(math.e)
        scale_softmax_val = scale_q * scale_k * scale_softmax
        scale_softmax_log2_val = scale_softmax_val * log2_e
        scale_output_val = scale_v * inv_scale_o

        if not skip_compile:
            if self._compiled_kernel is None:
                raise ValueError("CompressionAttention kernel not compiled")
            self._logger.debug("Executing with compiled kernel")
            self._compiled_kernel(
                q_iter=from_dlpack(
                    (
                        q_tensor.transpose(1, 2)
                        if self.input_layout == "B,H,S,D"
                        else q_tensor
                    ),
                    assumed_align=16,
                ).iterator,
                k_iter=from_dlpack(
                    (
                        k_tensor.transpose(1, 2)
                        if self.input_layout == "B,H,S,D"
                        else k_tensor
                    ),
                    assumed_align=16,
                ).iterator,
                v_iter=from_dlpack(
                    (
                        v_tensor.transpose(1, 2)
                        if self.input_layout == "B,H,S,D"
                        else v_tensor
                    ),
                    assumed_align=16,
                ).iterator,
                o_iter=from_dlpack(
                    (
                        o_tensor.transpose(1, 2)
                        if self.input_layout == "B,H,S,D"
                        else o_tensor
                    ),
                    assumed_align=16,
                ).iterator,
                problem_size=self.problem_size,
                cum_seqlen_q=(
                    from_dlpack(cum_seqlen_q_tensor, assumed_align=16).iterator
                    if self.input_layout == "T,H,D"
                    else None
                ),
                cum_seqlen_k=(
                    from_dlpack(cum_seqlen_k_tensor, assumed_align=16).iterator
                    if self.input_layout == "T,H,D"
                    else None
                ),
                lse_iter=(
                    from_dlpack(lse_tensor, assumed_align=16).iterator
                    if self.enable_lse
                    else None
                ),
                scale_softmax_log2=scale_softmax_log2_val,
                scale_softmax=scale_softmax_val,
                scale_output=scale_output_val,
                window_size_left=None,
                window_size_right=Int32(0),
                stream=current_stream,
            )
            self._logger.debug("Executed with compiled kernel successfully")
        else:
            self._logger.debug("Executing without compiled kernel (JIT)")
            fmha_kernel = self._kernel(
                _convert_to_cutlass_data_type(self.qk_acc_dtype_torch),
                _convert_to_cutlass_data_type(self.pv_acc_dtype_torch),
                (*self.mma_tiler_mn, self.head_dim),
                self.is_persistent,
                mask_type=fmha_utils.MaskType.COMPRESSED_CAUSAL_MASK,
            )
            fmha_kernel(
                q_iter=from_dlpack(
                    (
                        q_tensor.transpose(1, 2)
                        if self.input_layout == "B,H,S,D"
                        else q_tensor
                    ),
                    assumed_align=16,
                ).iterator,
                q_stride=(
                    q_tensor.transpose(1, 2).stride()
                    if self.input_layout == "B,H,S,D"
                    else (q_tensor.stride()[0], *q_tensor.stride())
                ),
                k_iter=from_dlpack(
                    (
                        k_tensor.transpose(1, 2)
                        if self.input_layout == "B,H,S,D"
                        else k_tensor
                    ),
                    assumed_align=16,
                ).iterator,
                k_stride=(
                    k_tensor.transpose(1, 2).stride()
                    if self.input_layout == "B,H,S,D"
                    else (k_tensor.stride()[0], *k_tensor.stride())
                ),
                v_iter=from_dlpack(
                    (
                        v_tensor.transpose(1, 2)
                        if self.input_layout == "B,H,S,D"
                        else v_tensor
                    ),
                    assumed_align=16,
                ).iterator,
                v_stride=(
                    v_tensor.transpose(1, 2).stride()
                    if self.input_layout == "B,H,S,D"
                    else (v_tensor.stride()[0], *v_tensor.stride())
                ),
                o_iter=from_dlpack(
                    (
                        o_tensor.transpose(1, 2)
                        if self.input_layout == "B,H,S,D"
                        else o_tensor
                    ),
                    assumed_align=16,
                ).iterator,
                o_stride=(
                    o_tensor.transpose(1, 2).stride()
                    if self.input_layout == "B,H,S,D"
                    else (o_tensor.stride()[0], *o_tensor.stride())
                ),
                problem_size=self.problem_size,
                cum_seqlen_q=(
                    from_dlpack(cum_seqlen_q_tensor, assumed_align=16)
                    if self.input_layout == "T,H,D"
                    else None
                ),
                cum_seqlen_k=(
                    from_dlpack(cum_seqlen_k_tensor, assumed_align=16)
                    if self.input_layout == "T,H,D"
                    else None
                ),
                lse_iter=(
                    from_dlpack(lse_tensor, assumed_align=16).iterator
                    if self.enable_lse
                    else None
                ),
                lse_stride=(
                    lse_tensor.transpose(1, 2).stride()
                    if self.input_layout == "B,H,S,D"
                    else (0, *lse_tensor.stride())
                ),
                scale_softmax_log2=scale_softmax_log2_val,
                scale_softmax=scale_softmax_val,
                scale_output=scale_output_val,
                window_size_left=None,
                window_size_right=Int32(0),
                stream=current_stream,
            )
            self._logger.debug("Executed successfully")


import logging

_logger = logging.getLogger(__name__)
_cache_of_CompressionAttentionObjects = {}


def compression_attention_wrapper(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    cum_seqlen_q_tensor: Optional[torch.Tensor] = None,
    cum_seqlen_k_tensor: Optional[torch.Tensor] = None,
    enable_lse: bool = False,
    o_dtype: Optional[torch.dtype] = None,
    qk_acc_dtype: torch.dtype = torch.float32,
    pv_acc_dtype: torch.dtype = torch.float32,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    is_persistent: bool = False,
    scale_q: float = 1.0,
    scale_k: float = 1.0,
    scale_v: float = 1.0,
    inv_scale_o: float = 1.0,
    scale_softmax: Optional[float] = None,
    stream: Optional[cuda.CUstream] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compression Attention Wrapper that returns output (and optionally LSE) tensors directly.

    Returns:
        tuple: (o_tensor, lse_tensor | None)
    """
    _logger.debug(
        "compression_attention_wrapper: Creating empty output tensor o and optional lse"
    )

    o_tensor, lse_tensor = None, None
    o_dtype = o_dtype if o_dtype is not None else q_tensor.dtype
    if q_tensor.ndim == 4:  # bshd
        b, h_q, s_q, d = q_tensor.shape
        _, h_k, s_k, d_v = v_tensor.shape

        o_tensor = make_tensor_strided_like(
            q_tensor, (b, h_q, s_q, d_v), dtype=o_dtype, device=q_tensor.device
        )
        if enable_lse:
            lse_tensor = torch.empty(
                b, h_q, s_q, dtype=torch.float32, device=q_tensor.device
            ).contiguous()
    elif q_tensor.ndim == 3:  # thd
        t, h_q, d = q_tensor.shape
        _, h_k, d_v = v_tensor.shape

        o_tensor = make_tensor_strided_like(
            q_tensor, (t, h_q, d_v), dtype=o_dtype, device=q_tensor.device
        )
        if enable_lse:
            lse_tensor = (
                torch.empty(1, h_q, t, dtype=torch.float32, device=q_tensor.device)
                .contiguous()
                .permute(2, 1, 0)
            )
    else:
        raise ValueError(
            f"Invalid input layout: q_tensor must be rank-4 (B,H,S,D) or rank-3 (T,H,D), got {q_tensor.ndim}"
        )

    cache_key = (
        q_tensor.shape,
        k_tensor.shape,
        v_tensor.shape,
        cum_seqlen_q_tensor.shape if cum_seqlen_q_tensor is not None else None,
        cum_seqlen_k_tensor.shape if cum_seqlen_k_tensor is not None else None,
        q_tensor.dtype,
        k_tensor.dtype,
        v_tensor.dtype,
        cum_seqlen_q_tensor.dtype if cum_seqlen_q_tensor is not None else None,
        cum_seqlen_k_tensor.dtype if cum_seqlen_k_tensor is not None else None,
        q_tensor.stride(),
        k_tensor.stride(),
        v_tensor.stride(),
        cum_seqlen_q_tensor.stride() if cum_seqlen_q_tensor is not None else None,
        cum_seqlen_k_tensor.stride() if cum_seqlen_k_tensor is not None else None,
        enable_lse,
        o_dtype,
        qk_acc_dtype,
        pv_acc_dtype,
        mma_tiler_mn,
        is_persistent,
        scale_q,
        scale_k,
        scale_v,
        inv_scale_o,
        scale_softmax,
    )
    if cache_key in _cache_of_CompressionAttentionObjects:
        _logger.debug(
            "compression_attention_wrapper: Using previously cached CompressionAttention object"
        )
        comp_attn = _cache_of_CompressionAttentionObjects[cache_key]
        comp_attn.execute(
            q_tensor=q_tensor,
            k_tensor=k_tensor,
            v_tensor=v_tensor,
            o_tensor=o_tensor,
            lse_tensor=lse_tensor,
            cum_seqlen_q_tensor=cum_seqlen_q_tensor,
            cum_seqlen_k_tensor=cum_seqlen_k_tensor,
            current_stream=stream,
        )
    else:
        _logger.debug(
            "compression_attention_wrapper: No cached object found, creating new CompressionAttention object"
        )
        comp_attn = CompressionAttention(
            sample_q=q_tensor,
            sample_k=k_tensor,
            sample_v=v_tensor,
            sample_o=o_tensor,
            sample_lse=lse_tensor,
            sample_cum_seqlen_q=cum_seqlen_q_tensor,
            sample_cum_seqlen_k=cum_seqlen_k_tensor,
            qk_acc_dtype=qk_acc_dtype,
            pv_acc_dtype=pv_acc_dtype,
            mma_tiler_mn=mma_tiler_mn,
            is_persistent=is_persistent,
            scale_q=scale_q,
            scale_k=scale_k,
            scale_v=scale_v,
            inv_scale_o=inv_scale_o,
            scale_softmax=scale_softmax,
        )
        assert comp_attn.check_support()
        comp_attn.compile()
        comp_attn.execute(
            q_tensor=q_tensor,
            k_tensor=k_tensor,
            v_tensor=v_tensor,
            o_tensor=o_tensor,
            lse_tensor=lse_tensor,
            cum_seqlen_q_tensor=cum_seqlen_q_tensor,
            cum_seqlen_k_tensor=cum_seqlen_k_tensor,
            current_stream=stream,
        )
        _cache_of_CompressionAttentionObjects[cache_key] = comp_attn

    return o_tensor, lse_tensor
