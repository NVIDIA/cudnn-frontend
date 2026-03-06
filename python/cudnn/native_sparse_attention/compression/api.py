from typing import Tuple, Optional
import math

from cuda.bindings import driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_stream
from cutlass.cute.typing import Int32

from cudnn.api_base import APIBase, TupleDict
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

        self.q_desc = self._make_tensor_desc(sample_q, name="sample_q")
        self.k_desc = self._make_tensor_desc(sample_k, name="sample_k")
        self.v_desc = self._make_tensor_desc(sample_v, name="sample_v")
        self.o_desc = self._make_tensor_desc(sample_o, name="sample_o")
        self.lse_desc = self._make_tensor_desc(sample_lse, name="sample_lse")
        self.enable_lse = sample_lse is not None
        self.cum_seqlen_q_desc = self._unpad_tensor_to_ndim(self._make_tensor_desc(sample_cum_seqlen_q, name="sample_cum_seqlen_q"), 1, "sample_cum_seqlen_q")
        self.cum_seqlen_k_desc = self._unpad_tensor_to_ndim(self._make_tensor_desc(sample_cum_seqlen_k, name="sample_cum_seqlen_k"), 1, "sample_cum_seqlen_k")
        self.max_cum_seqlen_q = int(sample_cum_seqlen_q.max().item()) if sample_cum_seqlen_q is not None else None
        self.max_cum_seqlen_k = int(sample_cum_seqlen_k.max().item()) if sample_cum_seqlen_k is not None else None

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
            f"__init__ completed with args: sample_q {self.q_desc.shape}, sample_k {self.k_desc.shape}, sample_v {self.v_desc.shape}, sample_o {self.o_desc.shape}, sample_lse {self.lse_desc.shape if self.lse_desc is not None else 'None'}, sample_cum_seqlen_q {self.cum_seqlen_q_desc.shape if self.cum_seqlen_q_desc is not None else 'None'}, sample_cum_seqlen_k {self.cum_seqlen_k_desc.shape if self.cum_seqlen_k_desc is not None else 'None'}, qk_acc_dtype {qk_acc_dtype}, pv_acc_dtype {pv_acc_dtype}, mma_tiler_mn {mma_tiler_mn}, is_persistent {is_persistent}, scale_q {scale_q}, scale_k {scale_k}, scale_v {scale_v}, inv_scale_o {inv_scale_o}, scale_softmax {scale_softmax}"
        )

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        # shape normalization and validation
        self._logger.debug("Checking shape normalization and validation")
        if self.q_desc.ndim == 4:
            self.input_layout = "B,H,S,D"

            b, h_qo, s_qo, d_qk = self.q_desc.shape
            b, h_kv, s_kv, d_qk = self.k_desc.shape
            b, h_kv, s_kv, d_v = self.v_desc.shape
            b, h_q, s_qo, d_v = self.o_desc.shape

            self._check_tensor_shape(self.q_desc, (b, h_qo, s_qo, d_qk), name="Q")
            self._check_tensor_shape(self.k_desc, (b, h_kv, s_kv, d_qk), name="K")
            self._check_tensor_shape(self.v_desc, (b, h_kv, s_kv, d_v), name="V")
            self._check_tensor_shape(self.o_desc, (b, h_q, s_qo, d_v), name="O")
            if self.enable_lse:

                self.lse_desc = self._unpad_tensor_to_ndim(self.lse_desc, 3, name="LSE")
                self._check_tensor_shape(self.lse_desc, (b, h_q, s_qo), name="LSE")
                self._check_tensor_stride(
                    self.lse_desc, stride=(h_q * s_qo, s_qo, 1), name="LSE", extra_error_msg="LSE tensor must be contiguous"
                )  # TODO @mingyangw: contiguous check
            if self.cum_seqlen_q_desc is not None or self.cum_seqlen_k_desc is not None:
                self._logger.warning("cum_seqlen_q and cum_seqlen_k are ignored for B,H,S,D layout")

            # Shapes
            self.batch_size = b
            self.s_q = s_qo
            self.s_kv = s_kv
            self.h_q = h_q
            self.h_kv = h_kv
            self.h_r = h_q // h_kv
            self.head_dim = d_qk
        elif self.q_desc.ndim == 3:
            self.input_layout = "T,H,D"

            t, h_q, d_qk = self.q_desc.shape
            t_kv, h_kv, d_qk = self.k_desc.shape  # T has been compressed for K and V
            t_kv, h_kv, d_v = self.v_desc.shape
            t, h_q, d_v = self.o_desc.shape

            self._check_tensor_shape(self.q_desc, (t, h_q, d_qk), name="Q")
            self._check_tensor_shape(self.k_desc, (t_kv, h_kv, d_qk), name="K")
            self._check_tensor_shape(self.v_desc, (t_kv, h_kv, d_v), name="V")
            self._check_tensor_shape(self.o_desc, (t, h_q, d_v), name="O")
            if self.enable_lse:
                self.lse_desc = self._unpad_tensor_to_ndim(self.lse_desc, 2, name="LSE")
                self._check_tensor_shape(self.lse_desc, (t, h_q), name="LSE")

            if self.cum_seqlen_q_desc is None or self.cum_seqlen_k_desc is None:
                raise ValueError(f"cum_seqlen_q and cum_seqlen_k must be provided for T,H,D layout, got {self.cum_seqlen_q_desc} and {self.cum_seqlen_k_desc}")

            if self.cum_seqlen_q_desc.ndim != 1 or self.cum_seqlen_k_desc.ndim != 1:
                raise ValueError(f"cum_seqlen_q and cum_seqlen_k must be 1D tensors, got {self.cum_seqlen_q_desc.ndim} and {self.cum_seqlen_k_desc.ndim}")
            self._check_dtype(self.cum_seqlen_q_desc, [torch.int32, torch.int64], name="cum_seqlen_q")
            self._check_dtype(self.cum_seqlen_k_desc, [torch.int32, torch.int64], name="cum_seqlen_k")
            if self.cum_seqlen_q_desc.shape[0] != self.cum_seqlen_k_desc.shape[0]:
                raise ValueError(
                    f"cum_seqlen_q and cum_seqlen_k must have the same length, got {self.cum_seqlen_q_desc.shape[0]} and {self.cum_seqlen_k_desc.shape[0]}"
                )

            self.batch_size = self.cum_seqlen_q_desc.shape[0] - 1
            self.s_q = None
            self.s_kv = None
            self.h_q = h_q
            self.h_kv = h_kv
            self.h_r = h_q // h_kv
            self.head_dim = d_qk

        else:
            raise ValueError(f"Invalid input layout: q must be rank-3 (T,H,D) or rank-4 (B,H,S,D), got {self.q_desc.ndim}")
        if d_qk != d_v:
            raise ValueError("D_qk must match D_v")
        if d_qk not in {32, 64, 128}:
            raise ValueError("Head dimension D_qk must be 32, 64, or 128")
        if h_q % h_kv != 0:
            raise ValueError("H_q must be divisible by H_k (GQA/MQA constraint)")

        self._logger.debug("Checking dtypes")
        in_dtype = self._check_dtype(self.q_desc, dtype=[torch.float16, torch.bfloat16, torch.float8_e4m3fn], name="Q")
        self._check_dtype(self.k_desc, dtype=in_dtype, name="K", extra_error_msg="K must have the same dtype as Q")
        self._check_dtype(self.v_desc, dtype=in_dtype, name="V", extra_error_msg="V must have the same dtype as Q")
        self._check_dtype(self.o_desc, dtype=[torch.float16, torch.bfloat16, torch.float8_e4m3fn], name="O")
        self._check_dtype(self.qk_acc_dtype_torch, dtype=torch.float32, name="qk_acc_dtype", extra_error_msg="qk_acc_dtype must be Float32")
        self._check_dtype(self.pv_acc_dtype_torch, dtype=torch.float32, name="pv_acc_dtype", extra_error_msg="pv_acc_dtype must be Float32")

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
            raise RuntimeError(f"CompressionAttention requires SM100+ compute capability, but found SM{compute_capability} on device {device}")
        if compute_capability == 103:
            raise RuntimeError("cuteDSL is not supported on SM103")

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def compile(self) -> None:
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            self._logger.debug("Kernel already compiled; skipping recompilation")
            return

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

        s_q = self.s_q if self.input_layout == "B,H,S,D" else self.max_cum_seqlen_q
        s_kv = self.s_kv if self.input_layout == "B,H,S,D" else self.max_cum_seqlen_k
        self.problem_size = (
            self.batch_size,
            s_q,
            s_q,
            s_kv,
            self.h_q,
            self.h_kv,
            self.head_dim,
        )
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)

        self._logger.debug("Compiling CompressionAttention kernel with cute.compile")
        if self.input_layout == "B,H,S,D":
            _q_desc = self.q_desc.transpose(1, 2)
            _k_desc = self.k_desc.transpose(1, 2)
            _v_desc = self.v_desc.transpose(1, 2)
            _o_desc = self.o_desc.transpose(1, 2)
            _lse_desc = self.lse_desc.transpose(1, 2) if self.enable_lse else None
        elif self.input_layout == "T,H,D":
            _q_desc = self.q_desc.as_strided(size=(1, *self.q_desc.shape), stride=(self.q_desc.stride[0], *self.q_desc.stride))
            _k_desc = self.k_desc.as_strided(size=(1, *self.k_desc.shape), stride=(self.k_desc.stride[0], *self.k_desc.stride))
            _v_desc = self.v_desc.as_strided(size=(1, *self.v_desc.shape), stride=(self.v_desc.stride[0], *self.v_desc.stride))
            _o_desc = self.o_desc.as_strided(size=(1, *self.o_desc.shape), stride=(self.o_desc.stride[0], *self.o_desc.stride))
            # lse_local = self.sample_lse.as_strided(size=(1, *self.sample_lse.shape), stride=(self.sample_lse.stride()[0], *self.sample_lse.stride()))
            _lse_desc = self.lse_desc.unsqueeze(0) if self.enable_lse else None
        else:
            raise NotImplementedError(f"Invalid input layout: {self.input_layout}")

        # breakpoint()
        _compiled_kernel = cute.compile(
            fmha_kernel,
            Q=self._make_fake_cute_tensor_from_desc(_q_desc, assumed_align=16),
            K=self._make_fake_cute_tensor_from_desc(_k_desc, assumed_align=16),
            V=self._make_fake_cute_tensor_from_desc(_v_desc, assumed_align=16),
            O=self._make_fake_cute_tensor_from_desc(_o_desc, assumed_align=16),
            problem_size=self.problem_size,
            cum_seqlen_q=(self._make_fake_cute_tensor_from_desc(self.cum_seqlen_q_desc, assumed_align=16) if self.input_layout == "T,H,D" else None),
            cum_seqlen_k=(self._make_fake_cute_tensor_from_desc(self.cum_seqlen_k_desc, assumed_align=16) if self.input_layout == "T,H,D" else None),
            LSE=(self._make_fake_cute_tensor_from_desc(_lse_desc, assumed_align=16) if self.enable_lse else None),
            scale_softmax_log2=scale_softmax_log2,
            scale_softmax=scale_softmax,
            scale_output=scale_output,
            window_size_left=None,
            window_size_right=Int32(0),
            stream=fake_stream,
            options="--enable-tvm-ffi",
        )

        def tensor_api(
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            problem_size,
            cum_seqlen_q,
            cum_seqlen_k,
            lse_tensor,
            scale_softmax_log2,
            scale_softmax,
            scale_output,
            window_size_left,
            window_size_right,
            stream,
        ):
            if self.enable_lse:
                lse_tensor = self._unpad_tensor_to_ndim(lse_tensor, self.o_desc.ndim - 1, "lse_tensor")
            if self.input_layout == "B,H,S,D":
                q_tensor = q_tensor.transpose(1, 2)
                k_tensor = k_tensor.transpose(1, 2)
                v_tensor = v_tensor.transpose(1, 2)
                o_tensor = o_tensor.transpose(1, 2)
                lse_tensor = lse_tensor.transpose(1, 2) if self.enable_lse else None
            elif self.input_layout == "T,H,D":
                q_tensor = q_tensor.as_strided(size=(1, *q_tensor.shape), stride=(q_tensor.stride()[0], *q_tensor.stride()))
                k_tensor = k_tensor.as_strided(size=(1, *k_tensor.shape), stride=(k_tensor.stride()[0], *k_tensor.stride()))
                v_tensor = v_tensor.as_strided(size=(1, *v_tensor.shape), stride=(v_tensor.stride()[0], *v_tensor.stride()))
                o_tensor = o_tensor.as_strided(size=(1, *o_tensor.shape), stride=(o_tensor.stride()[0], *o_tensor.stride()))
                lse_tensor = lse_tensor.unsqueeze(0) if self.enable_lse else None
                cum_seqlen_q = self._unpad_tensor_to_ndim(cum_seqlen_q, 1, "cum_seqlen_q")
                cum_seqlen_k = self._unpad_tensor_to_ndim(cum_seqlen_k, 1, "cum_seqlen_k")

            return _compiled_kernel(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                problem_size,
                cum_seqlen_q,
                cum_seqlen_k,
                lse_tensor,
                scale_softmax_log2,
                scale_softmax,
                scale_output,
                window_size_left,
                window_size_right,
                stream,
            )

        self._compiled_kernel = tensor_api

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
                raise ValueError("kernel was compiled with lse_tensor provided, but lse_tensor was not provided during execute")

        if self.input_layout == "T,H,D":
            if cum_seqlen_q_tensor is None or cum_seqlen_k_tensor is None:
                raise ValueError("cum_seqlen_q_tensor and cum_seqlen_k_tensor must be provided during execute for T,H,D layout")

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

        if self._compiled_kernel is None:
            raise ValueError("CompressionAttention kernel not compiled")
        self._logger.debug("Executing with compiled kernel")
        self._compiled_kernel(
            q_tensor=q_tensor,
            k_tensor=k_tensor,
            v_tensor=v_tensor,
            o_tensor=o_tensor,
            problem_size=self.problem_size,
            cum_seqlen_q=(cum_seqlen_q_tensor if self.input_layout == "T,H,D" else None),
            cum_seqlen_k=(cum_seqlen_k_tensor if self.input_layout == "T,H,D" else None),
            lse_tensor=lse_tensor,
            scale_softmax_log2=scale_softmax_log2_val,
            scale_softmax=scale_softmax_val,
            scale_output=scale_output_val,
            window_size_left=None,
            window_size_right=Int32(0),
            stream=current_stream,
        )
        self._logger.debug("Executed with compiled kernel successfully")


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
) -> TupleDict:
    """
    Compression Attention Wrapper that returns output (and optionally LSE) tensors.

    Returns:
        TupleDict: (o_tensor, lse_tensor | None)
    """
    _logger.debug("compression_attention_wrapper: Creating empty output tensor o and optional lse")

    o_tensor, lse_tensor = None, None
    o_dtype = o_dtype if o_dtype is not None else q_tensor.dtype
    if q_tensor.ndim == 4:  # bshd
        b, h_q, s_q, d = q_tensor.shape
        _, h_k, s_k, d_v = v_tensor.shape

        o_tensor = make_tensor_strided_like(q_tensor, (b, h_q, s_q, d_v), dtype=o_dtype, device=q_tensor.device)
        if enable_lse:
            lse_tensor = torch.empty(b, h_q, s_q, dtype=torch.float32, device=q_tensor.device).contiguous()
    elif q_tensor.ndim == 3:  # thd
        t, h_q, d = q_tensor.shape
        _, h_k, d_v = v_tensor.shape

        o_tensor = make_tensor_strided_like(q_tensor, (t, h_q, d_v), dtype=o_dtype, device=q_tensor.device)
        if enable_lse:
            lse_tensor = torch.empty(1, h_q, t, dtype=torch.float32, device=q_tensor.device).contiguous().permute(2, 1, 0)
    else:
        raise ValueError(f"Invalid input layout: q_tensor must be rank-4 (B,H,S,D) or rank-3 (T,H,D), got {q_tensor.ndim}")

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
        _logger.debug("compression_attention_wrapper: Using previously cached CompressionAttention object")
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
        _logger.debug("compression_attention_wrapper: No cached object found, creating new CompressionAttention object")
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

    return TupleDict(
        o_tensor=o_tensor,
        lse_tensor=lse_tensor,
    )
