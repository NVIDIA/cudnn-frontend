from .NSA_select_attn_fwd_hmma import HopperSelectAttentionFwd
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.api_base import APIBase, TupleDict

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_stream
from cuda.bindings import driver as cuda
import torch
from typing import Tuple, Optional
import math


class SelectionAttention(APIBase):
    def __init__(
        self,
        sample_q: torch.Tensor,
        sample_k: torch.Tensor,
        sample_v: torch.Tensor,
        sample_o: torch.Tensor,
        sample_l: torch.Tensor,
        sample_m: torch.Tensor,
        sample_block_indices: torch.Tensor,
        sample_block_counts: torch.Tensor,
        sample_cum_seqlen_q: Optional[torch.Tensor] = None,
        sample_cum_seqlen_k: Optional[torch.Tensor] = None,
        max_s_q: Optional[int] = 1024,
        max_s_k: Optional[int] = 1024,
        acc_dtype: torch.dtype = torch.float32,
        block_size: int = 64,
        scale_softmax: Optional[float] = None,
    ):
        super().__init__()
        self._kernel = HopperSelectAttentionFwd

        self._logger.warning("SelectionAttention is an experimental API")
        self._logger.debug("Entering __init__")

        self.q_desc = self._make_tensor_desc(sample_q, name="sample_q")
        self.k_desc = self._make_tensor_desc(sample_k, name="sample_k")
        self.v_desc = self._make_tensor_desc(sample_v, name="sample_v")
        self.o_desc = self._make_tensor_desc(sample_o, name="sample_o")
        self.l_desc = self._make_tensor_desc(sample_l, name="sample_l")
        self.m_desc = self._make_tensor_desc(sample_m, name="sample_m")
        self.block_indices_desc = self._make_tensor_desc(sample_block_indices, name="sample_block_indices")
        self.block_counts_desc = self._make_tensor_desc(sample_block_counts, name="sample_block_counts")
        self.cum_seqlen_q_desc = self._make_tensor_desc(sample_cum_seqlen_q, name="sample_cum_seqlen_q")
        self.cum_seqlen_k_desc = self._make_tensor_desc(sample_cum_seqlen_k, name="sample_cum_seqlen_k")
        if sample_cum_seqlen_q is not None and sample_cum_seqlen_k is not None:
            if not torch.equal(sample_cum_seqlen_q, sample_cum_seqlen_k):
                raise NotImplementedError("sample_cum_seqlen_k is not yet supported. Must be None or identical to sample_cum_seqlen_q")
        self.max_s_q = max_s_q
        self.max_s_k = max_s_k

        # Types and kernel configuration
        self.acc_dtype = acc_dtype
        self.block_size = block_size

        # Derived attributes (populated in check_support)
        self.input_layout = None
        self.dtype = None
        self.h_q = None
        self.h_kv = None
        self.gqa_group_size = None
        self.head_dim = None
        self.value_dim = None

        self.scale_softmax = scale_softmax

        self._logger.debug(
            f"__init__ completed with args: sample_q {self.q_desc.shape}, sample_k {self.k_desc.shape}, sample_v {self.v_desc.shape}, sample_o {self.o_desc.shape}, sample_l {self.l_desc.shape}, sample_m {self.m_desc.shape}, sample_block_indices {self.block_indices_desc.shape}, sample_block_counts {self.block_counts_desc.shape}, sample_cum_seqlen_q {self.cum_seqlen_q_desc.shape if self.cum_seqlen_q_desc is not None else 'None'}, sample_cum_seqlen_k {self.cum_seqlen_k_desc.shape if self.cum_seqlen_k_desc is not None else 'None'}, acc_dtype {acc_dtype}, max_s_q {max_s_q}, max_s_k {max_s_k}, block_size {block_size}, scale_softmax {scale_softmax}"
        )

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        # Shape normalization and validation
        self._logger.debug("Checking shape normalization and validation")
        if self.q_desc.ndim == 4:
            # B, H_q, S, D  format
            self.input_layout = "B,H,S,D"

            raise NotImplementedError("B, H_q, S, D format not implemented")
        elif self.q_desc.ndim == 3:
            # T, H_q, D  format
            self.input_layout = "T,H,D"

            t, h_q, d_qk = self.q_desc.shape
            t, h_kv, d_qk = self.k_desc.shape
            t, h_kv, d_v = self.v_desc.shape
            t, h_q, d_v = self.o_desc.shape

            self._check_tensor_shape(self.q_desc, (t, h_q, d_qk), name="Q")
            self._check_tensor_shape(self.k_desc, (t, h_kv, d_qk), name="K")
            self._check_tensor_shape(self.v_desc, (t, h_kv, d_v), name="V")
            self._check_tensor_shape(self.o_desc, (t, h_q, d_v), name="O")
            self.l_desc = self._unpad_tensor_to_ndim(self.l_desc, 2, "sample_l")
            self._check_tensor_shape(self.l_desc, (t, h_q), name="L")
            self.m_desc = self._unpad_tensor_to_ndim(self.m_desc, 2, "sample_m")
            self._check_tensor_shape(self.m_desc, (t, h_q), name="M")
            if self.cum_seqlen_q_desc is None:
                raise ValueError(f"cum_seqlen_q must be provided for T,H,D format, got {self.cum_seqlen_q_desc}")
            if self.max_s_q is None:
                raise ValueError(f"max_s_q must be provided for T,H,D format, got {self.max_s_q}")
            if self.max_s_k is not None and self.max_s_q != self.max_s_k:
                raise NotImplementedError(f"SelectionAttention requires max_s_q and max_s_k to be identical, but got {self.max_s_q} and {self.max_s_k}")

            self.batch_size = self.cum_seqlen_q_desc.shape[0] - 1
            if self.batch_size <= 0:
                raise ValueError(f"batch_size (len(cum_seqlen_q) - 1) must be greater than 0, got {self.batch_size}")
            if self.cum_seqlen_q_desc.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"cum_seqlen_q must be int32 or int64, got {self.cum_seqlen_q_desc.dtype}")

            if self.block_indices_desc.shape[:2] != (t, h_kv) and self.block_indices_desc.ndim != 3:
                raise ValueError(f"block_indices shape mismatch: expected {(t, h_kv, 'K')}, got {tuple(self.block_indices_desc.shape)}")
            if self.block_counts_desc.shape != (t, h_kv):
                raise ValueError(f"block_counts shape mismatch: expected {(t, h_kv)}, got {tuple(self.block_counts_desc.shape)}")
            if self.block_indices_desc.dtype != torch.int32 or self.block_counts_desc.dtype != torch.int32:
                raise ValueError(f"block_indices and block_counts must be int32, got {self.block_indices_desc.dtype} and {self.block_counts_desc.dtype}")
        else:
            raise ValueError(f"q must be rank-3 (T,H,D) or rank-4 (B,H,S,D), got {self.q_desc.ndim}")

        # Shared derived attributes
        if h_q % h_kv != 0:
            raise ValueError("H_q must be a multiple of H_kv (GQA/MQA constraint)")
        self.h_q = h_q
        self.h_kv = h_kv
        self.gqa_group_size = h_q // h_kv
        self.head_dim = d_qk
        self.value_dim = d_v

        # Validate dtypes and config
        self._logger.debug("Checking dtypes and config")
        self.dtype = self._check_dtype(self.q_desc, dtype=[torch.float16, torch.bfloat16], name="Q")
        _ = self._check_dtype(self.k_desc, dtype=self.dtype, name="K", extra_error_msg="K must have the same dtype as Q")
        _ = self._check_dtype(self.v_desc, dtype=self.dtype, name="V", extra_error_msg="V must have the same dtype as Q")
        _ = self._check_dtype(self.o_desc, dtype=self.dtype, name="O", extra_error_msg="O must have the same dtype as Q")
        _ = self._check_dtype(self.acc_dtype, dtype=torch.float32, name="Acc", extra_error_msg="acc_dtype must be Float32")

        if self.block_size not in {16, 32, 64}:
            raise ValueError("block_size must be 16, 32, or 64")

        # Compute default scale_softmax if needed
        if self.scale_softmax is None:
            self.scale_softmax = 1.0 / math.sqrt(self.head_dim)

        if not torch.cuda.is_available():
            self._logger.error("CUDA is not available")
            raise RuntimeError("CUDA is not available")

        self._logger.debug("Checking environment")
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        compute_capability = major * 10 + minor
        if compute_capability < 90:
            self._logger.error(f"Requires SM90+ compute capability, but found SM{compute_capability} on device {device}")
            raise RuntimeError(f"Requires SM90+ compute capability, but found SM{compute_capability} on device {device}")
        if compute_capability == 103:
            raise RuntimeError("cuteDSL SelectionAttention is not supported on SM103")

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def compile(self) -> None:
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            self._logger.debug("Kernel already compiled; skipping recompilation")
            return

        selection_attention = self._kernel(
            head_dim=self.head_dim,
            value_dim=self.value_dim,
            GQA_group_size=self.gqa_group_size,
            block_size=self.block_size,
            dtype=_convert_to_cutlass_data_type(self.dtype),
            acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
        )

        if self.input_layout == "T,H,D":
            _q_desc = self.q_desc.unsqueeze(0)
            _k_desc = self.k_desc.unsqueeze(0)
            _v_desc = self.v_desc.unsqueeze(0)
            _o_desc = self.o_desc.unsqueeze(0)
            _l_desc = self.l_desc.unsqueeze(0)
            _m_desc = self.m_desc.unsqueeze(0)
        else:
            raise NotImplementedError(f"Invalid input layout: {self.input_layout}")

        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)

        self._logger.debug("Compiling selection_attention")
        _compiled_kernel = cute.compile(
            selection_attention,
            Q=self._make_fake_cute_tensor_from_desc(_q_desc, assumed_align=128),
            K=self._make_fake_cute_tensor_from_desc(_k_desc, assumed_align=128),
            V=self._make_fake_cute_tensor_from_desc(_v_desc, assumed_align=128),
            O=self._make_fake_cute_tensor_from_desc(_o_desc, assumed_align=128),
            L=self._make_fake_cute_tensor_from_desc(_l_desc),
            M=self._make_fake_cute_tensor_from_desc(_m_desc),
            block_indices=self._make_fake_cute_tensor_from_desc(self.block_indices_desc),
            block_counts=self._make_fake_cute_tensor_from_desc(self.block_counts_desc),
            max_length=self.max_s_q,
            seq_offsets=self._make_fake_cute_tensor_from_desc(self.cum_seqlen_q_desc),
            softmax_scale=self.scale_softmax,
            stream=fake_stream,
            options="--enable-tvm-ffi",
        )

        def tensor_api(
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            l_tensor,
            m_tensor,
            block_indices_tensor,
            block_counts_tensor,
            cum_seqlen_q_tensor,
            softmax_scale,
            stream,
        ):
            # assumed T,H,D format
            q_tensor = q_tensor.unsqueeze(0)
            k_tensor = k_tensor.unsqueeze(0)
            v_tensor = v_tensor.unsqueeze(0)
            o_tensor = o_tensor.unsqueeze(0)
            l_tensor = self._unpad_tensor_to_ndim(l_tensor, 2, "l_tensor").unsqueeze(0)
            m_tensor = self._unpad_tensor_to_ndim(m_tensor, 2, "m_tensor").unsqueeze(0)

            return _compiled_kernel(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                l_tensor,
                m_tensor,
                block_indices_tensor,
                block_counts_tensor,
                self.max_s_q,
                cum_seqlen_q_tensor,
                softmax_scale,
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
        l_tensor: torch.Tensor,
        m_tensor: torch.Tensor,
        block_indices_tensor: torch.Tensor,
        block_counts_tensor: torch.Tensor,
        cum_seqlen_q_tensor: Optional[torch.Tensor] = None,
        cum_seqlen_k_tensor: Optional[torch.Tensor] = None,
        scale_softmax: Optional[float] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ):
        self._logger.debug("Entering execute")
        current_stream = self._get_default_stream(current_stream)

        scale_softmax = self.scale_softmax if scale_softmax is None else scale_softmax

        if self._compiled_kernel is None:
            raise RuntimeError("SelectionAttention kernel not compiled")
        self._logger.debug("Executing with compiled kernel")
        self._compiled_kernel(
            q_tensor=q_tensor,
            k_tensor=k_tensor,
            v_tensor=v_tensor,
            o_tensor=o_tensor,
            l_tensor=l_tensor,
            m_tensor=m_tensor,
            block_indices_tensor=block_indices_tensor,
            block_counts_tensor=block_counts_tensor,
            cum_seqlen_q_tensor=cum_seqlen_q_tensor,
            softmax_scale=scale_softmax,
            stream=current_stream,
        )
        self._logger.debug("Executed with compiled kernel successfully")


import logging

_logger = logging.getLogger(__name__)
_cache_of_SelectionAttentionObjects = {}


def selection_attention_wrapper(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    block_indices_tensor: torch.Tensor,
    block_counts_tensor: torch.Tensor,
    cum_seqlen_q_tensor: Optional[torch.Tensor] = None,
    cum_seqlen_k_tensor: Optional[torch.Tensor] = None,
    block_size: int = 64,
    scale_softmax: Optional[float] = None,
    o_dtype: Optional[torch.dtype] = None,
    acc_dtype: torch.dtype = torch.float32,
    max_s_q: Optional[int] = None,
    max_s_k: Optional[int] = None,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """
    Selection Attention Wrapper that returns output tensors.

    Returns:
        TupleDict: (o_tensor, l_tensor, m_tensor) - Output, logsumexp, and max tensors
    """
    _logger.debug("selection_attention_wrapper: Creating empty output tensors o, l, and m")

    max_s_q = max(cum_seqlen_q_tensor[1:] - cum_seqlen_q_tensor[:-1]).item() if max_s_q is None else max_s_q
    max_s_k = max(cum_seqlen_k_tensor[1:] - cum_seqlen_k_tensor[:-1]).item() if max_s_k is None else max_s_k

    t, h_q, d = q_tensor.shape
    _, h_kv, d_v = v_tensor.shape

    o_dtype = o_dtype if o_dtype is not None else q_tensor.dtype
    o_tensor = torch.empty((t, h_q, d_v), dtype=o_dtype, device=q_tensor.device)
    l_tensor = torch.empty((t, h_q, 1), dtype=torch.float32, device=q_tensor.device)
    m_tensor = torch.empty((t, h_q, 1), dtype=torch.float32, device=q_tensor.device)

    cache_key = (
        q_tensor.shape,
        k_tensor.shape,
        v_tensor.shape,
        block_indices_tensor.shape,
        block_counts_tensor.shape,
        cum_seqlen_q_tensor.shape,
        cum_seqlen_k_tensor.shape,
        q_tensor.dtype,
        k_tensor.dtype,
        v_tensor.dtype,
        q_tensor.stride(),
        k_tensor.stride(),
        v_tensor.stride(),
        block_indices_tensor.stride(),
        block_counts_tensor.stride(),
        cum_seqlen_q_tensor.stride(),
        cum_seqlen_k_tensor.stride(),
        block_size,
        scale_softmax,
        acc_dtype,
        max_s_q,
        max_s_k,
    )
    if cache_key in _cache_of_SelectionAttentionObjects:
        _logger.debug("selection_attention_wrapper: Using previously cached SelectionAttention object")
        selection_attention = _cache_of_SelectionAttentionObjects[cache_key]
        selection_attention.execute(
            q_tensor=q_tensor,
            k_tensor=k_tensor,
            v_tensor=v_tensor,
            o_tensor=o_tensor,
            l_tensor=l_tensor,
            m_tensor=m_tensor,
            block_indices_tensor=block_indices_tensor,
            block_counts_tensor=block_counts_tensor,
            cum_seqlen_q_tensor=cum_seqlen_q_tensor,
            cum_seqlen_k_tensor=cum_seqlen_k_tensor,
            scale_softmax=scale_softmax,
            current_stream=stream,
        )
    else:
        _logger.debug("selection_attention_wrapper: No previously cached SelectionAttention object found, creating new SelectionAttention object")
        selection_attention = SelectionAttention(
            sample_q=q_tensor,
            sample_k=k_tensor,
            sample_v=v_tensor,
            sample_o=o_tensor,
            sample_l=l_tensor,
            sample_m=m_tensor,
            sample_block_indices=block_indices_tensor,
            sample_block_counts=block_counts_tensor,
            sample_cum_seqlen_q=cum_seqlen_q_tensor,
            sample_cum_seqlen_k=cum_seqlen_k_tensor,
            acc_dtype=acc_dtype,
            max_s_q=max_s_q,
            max_s_k=max_s_k,
            block_size=block_size,
            scale_softmax=scale_softmax,
        )
        assert selection_attention.check_support()
        selection_attention.compile()
        selection_attention.execute(
            q_tensor=q_tensor,
            k_tensor=k_tensor,
            v_tensor=v_tensor,
            o_tensor=o_tensor,
            l_tensor=l_tensor,
            m_tensor=m_tensor,
            block_indices_tensor=block_indices_tensor,
            block_counts_tensor=block_counts_tensor,
            cum_seqlen_q_tensor=cum_seqlen_q_tensor,
            cum_seqlen_k_tensor=cum_seqlen_k_tensor,
            scale_softmax=scale_softmax,
            current_stream=stream,
        )
        _cache_of_SelectionAttentionObjects[cache_key] = selection_attention

    return TupleDict(
        o_tensor=o_tensor,
        l_tensor=l_tensor,
        m_tensor=m_tensor,
    )
