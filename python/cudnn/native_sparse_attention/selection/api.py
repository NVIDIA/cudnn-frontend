from .NSA_select_attn_fwd_hmma import HopperSelectAttentionFwd
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.api_base import APIBase

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
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

        # Store sample tensors only; defer validation to check_support
        self.sample_q = sample_q
        self.sample_k = sample_k
        self.sample_v = sample_v
        self.sample_o = sample_o
        self.sample_l = sample_l
        self.sample_m = sample_m
        self.sample_block_indices = sample_block_indices
        self.sample_block_counts = sample_block_counts
        self.sample_cum_seqlen_q = sample_cum_seqlen_q
        self.sample_cum_seqlen_k = sample_cum_seqlen_k
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
            f"__init__ completed with args: sample_q {sample_q.shape}, sample_k {sample_k.shape}, sample_v {sample_v.shape}, sample_o {sample_o.shape}, sample_l {sample_l.shape}, sample_m {sample_m.shape}, sample_block_indices {sample_block_indices.shape}, sample_block_counts {sample_block_counts.shape}, sample_cum_seqlen_q {sample_cum_seqlen_q.shape if sample_cum_seqlen_q is not None else 'None'}, sample_cum_seqlen_k {sample_cum_seqlen_k.shape if sample_cum_seqlen_k is not None else 'None'}, acc_dtype {acc_dtype}, max_s_q {max_s_q}, max_s_k {max_s_k}, block_size {block_size}, scale_softmax {scale_softmax}"
        )

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        # Shape normalization and validation
        self._logger.debug("Checking shape normalization and validation")
        if self.sample_q.ndim == 4:
            # B, H_q, S, D  format
            self.input_layout = "B,H,S,D"

            raise NotImplementedError("B, H_q, S, D format not implemented")
        elif self.sample_q.ndim == 3:
            # T, H_q, D  format
            self.input_layout = "T,H,D"

            t, h_q, d_qk = self.sample_q.shape
            t, h_kv, d_qk = self.sample_k.shape
            t, h_kv, d_v = self.sample_v.shape
            t, h_q, d_v = self.sample_o.shape

            if self.sample_q.shape != (t, h_q, d_qk):
                raise ValueError(
                    f"Input shape mismatch: expected Q tensor shape {t, h_q, d_qk}, got {self.sample_q.shape}"
                )
            if self.sample_k.shape != (t, h_kv, d_qk):
                raise ValueError(
                    f"Input shape mismatch: expected K tensor shape {t, h_kv, d_qk}, got {self.sample_k.shape}"
                )
            if self.sample_v.shape != (t, h_kv, d_v):
                raise ValueError(
                    f"Input shape mismatch: expected V tensor shape {t, h_kv, d_v}, got {self.sample_v.shape}"
                )
            if self.sample_o.shape != (t, h_q, d_v):
                raise ValueError(
                    f"Output shape mismatch: expected O tensor shape {t, h_q, d_v}, got {self.sample_o.shape}"
                )
            self.sample_l = self._unpad_tensor_to_ndim(self.sample_l, 2, "sample_l")
            if self.sample_l.shape != (t, h_q):
                raise ValueError(
                    f"Output shape mismatch: expected L tensor shape {t, h_q}, got {self.sample_l.shape}"
                )
            self.sample_m = self._unpad_tensor_to_ndim(self.sample_m, 2, "sample_m")
            if self.sample_m.shape != (t, h_q):
                raise ValueError(
                    f"Output shape mismatch: expected M tensor shape {t, h_q}, got {self.sample_m.shape}"
                )

            if self.sample_cum_seqlen_q is None:
                raise ValueError(
                    f"sample_cum_seqlen_q must be provided for T,H,D format, got {self.sample_cum_seqlen_q}"
                )
            if self.sample_cum_seqlen_k is not None and not torch.equal(
                self.sample_cum_seqlen_q, self.sample_cum_seqlen_k
            ):
                raise NotImplementedError(
                    f"SelectionAttention requires sample_cum_seqlen_q and sample_cum_seqlen_k to be identical, but got {self.sample_cum_seqlen_q} and {self.sample_cum_seqlen_k}"
                )
            if self.max_s_q is None:
                raise ValueError(
                    f"max_s_q must be provided for T,H,D format, got {self.max_s_q}"
                )
            if self.max_s_k is not None and self.max_s_q != self.max_s_k:
                raise NotImplementedError(
                    f"SelectionAttention requires max_s_q and max_s_k to be identical, but got {self.max_s_q} and {self.max_s_k}"
                )

            self.batch_size = len(self.sample_cum_seqlen_q) - 1
            if self.batch_size <= 0:
                raise ValueError(
                    f"batch_size (len(sample_cum_seqlen_q) - 1) must be greater than 0, got {self.batch_size}"
                )
            if self.sample_cum_seqlen_q.dtype not in (torch.int32, torch.int64):
                raise ValueError(
                    f"sample_cum_seqlen_q must be int32 or int64, got {self.sample_cum_seqlen_q.dtype}"
                )

            if (
                self.sample_block_indices.shape[:2] != (t, h_kv)
                and self.sample_block_indices.ndim != 3
            ):
                raise ValueError(
                    f"sample_block_indices shape mismatch: expected {(t, h_kv, 'K')}, got {tuple(self.sample_block_indices.shape)}"
                )
            if self.sample_block_counts.shape != (t, h_kv):
                raise ValueError(
                    f"sample_block_counts shape mismatch: expected {(t, h_kv)}, got {tuple(self.sample_block_counts.shape)}"
                )
            if (
                self.sample_block_indices.dtype != torch.int32
                or self.sample_block_counts.dtype != torch.int32
            ):
                raise ValueError(
                    f"sample_block_indices and sample_block_counts must be int32, got {self.sample_block_indices.dtype} and {self.sample_block_counts.dtype}"
                )
        else:
            raise ValueError(
                f"sample_q must be rank-3 (T,H,D) or rank-4 (B,H,S,D), got {self.sample_q.ndim}"
            )

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
        self.dtype = self.sample_q.dtype
        if not (
            self.dtype
            == self.sample_k.dtype
            == self.sample_v.dtype
            == self.sample_o.dtype
        ):
            raise ValueError("All input/output tensors must have the same dtype")
        if self.dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError("dtype must be Float16 or BFloat16")
        if self.acc_dtype not in {torch.float32}:
            raise ValueError("acc_dtype must be Float32")
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
            self._logger.error(
                f"Requires SM90+ compute capability, but found SM{compute_capability} on device {device}"
            )
            raise RuntimeError(
                f"Requires SM90+ compute capability, but found SM{compute_capability} on device {device}"
            )
        if compute_capability == 103:
            raise RuntimeError("cuteDSL SelectionAttention is not supported on SM103")

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def _reshape_tensors(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        l: torch.Tensor,
        m: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Reshape tensors from input format to kernel expected format:
        - Q: (gqa_group_size, d, T, h_kv)
        - K: (T, d, h_kv)
        - V: (T, d_v, h_kv)
        - O: (gqa_group_size, d_v, T, h_kv)
        - L: (gqa_group_size, T, h_kv)
        - M: (gqa_group_size, T, h_kv)
        """
        if self.input_layout == "B,H,S,D":
            raise NotImplementedError("B,H,S,D format not implemented")
        elif self.input_layout == "T,H,D":
            T, h_q, d = q.shape
            _, h_kv, _ = k.shape
            _, _, d_v = v.shape

            # Reshape Q: (T, H_q, D) -> (gqa_group_size, D, T, H_kv)
            q_reshaped = q.view(T, h_kv, self.gqa_group_size, d).permute(2, 3, 0, 1)
            # Reshape K: (T, H_kv, D) -> (T, D, H_kv)
            k_reshaped = k.permute(0, 2, 1)
            # Reshape V: (T, H_kv, D_v) -> (T, D_v, H_kv)
            v_reshaped = v.permute(0, 2, 1)
            # Reshape O: (T, H_q, D_v) -> (gqa_group_size, D_v, T, H_kv)
            o_reshaped = o.view(T, h_kv, self.gqa_group_size, d_v).permute(2, 3, 0, 1)
            # Reshape L: (T, H_q) -> (gqa_group_size, T, H_kv)
            l_reshaped = l.view(T, h_kv, self.gqa_group_size).permute(2, 0, 1)
            # Reshape M: (T, H_q) -> (gqa_group_size, T, H_kv)
            m_reshaped = m.view(T, h_kv, self.gqa_group_size).permute(2, 0, 1)
        else:
            raise ValueError(f"Invalid input layout: {self.input_layout}")

        # Temporary: assert that no memory is copied during reshape
        # Long term, we'd instead want to handle copying output tensors back to their original tensors
        def shares_memory(original, reshaped):
            return original.data_ptr() == reshaped.data_ptr()

        if not shares_memory(q, q_reshaped):
            raise ValueError(
                "Q tensor memory changed during reshape - expected view operation"
            )
        if not shares_memory(k, k_reshaped):
            raise ValueError(
                "K tensor memory changed during reshape - expected view operation"
            )
        if not shares_memory(v, v_reshaped):
            raise ValueError(
                "V tensor memory changed during reshape - expected view operation"
            )
        if not shares_memory(o, o_reshaped):
            raise ValueError(
                "O tensor memory changed during reshape - expected view operation"
            )
        if not shares_memory(l, l_reshaped):
            raise ValueError(
                "L tensor memory changed during reshape - expected view operation"
            )
        if not shares_memory(m, m_reshaped):
            raise ValueError(
                "M tensor memory changed during reshape - expected view operation"
            )

        return q_reshaped, k_reshaped, v_reshaped, o_reshaped, l_reshaped, m_reshaped

    def compile(self, current_stream: Optional[cuda.CUstream] = None) -> None:
        self._logger.debug("Entering compile")
        current_stream = self._get_default_stream(current_stream)
        self._ensure_support_checked()

        selection_attention = self._kernel(
            head_dim=self.head_dim,
            value_dim=self.value_dim,
            GQA_group_size=self.gqa_group_size,
            block_size=self.block_size,
            dtype=_convert_to_cutlass_data_type(self.dtype),
            acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
        )

        self._logger.debug("Reshaping tensors to kernel expected format")
        q_reshaped, k_reshaped, v_reshaped, o_reshaped, l_reshaped, m_reshaped = (
            self._reshape_tensors(
                self.sample_q,
                self.sample_k,
                self.sample_v,
                self.sample_o,
                self.sample_l,
                self.sample_m,
            )
        )

        mQ = from_dlpack(q_reshaped, assumed_align=128)
        mK = from_dlpack(k_reshaped, assumed_align=128)
        mV = from_dlpack(v_reshaped, assumed_align=128)
        mO = from_dlpack(o_reshaped, assumed_align=128)
        mL = from_dlpack(l_reshaped)
        mM = from_dlpack(m_reshaped)
        m_block_indices = from_dlpack(self.sample_block_indices)
        m_block_counts = from_dlpack(self.sample_block_counts)
        m_cum_seqlen_q = from_dlpack(self.sample_cum_seqlen_q)
        # m_cum_seqlen_k = from_dlpack(self.sample_cum_seqlen_k) # unused

        self._logger.debug("Compiling selection_attention")
        self._compiled_kernel = cute.compile(
            selection_attention,
            Q=mQ,
            K=mK,
            V=mV,
            O=mO,
            L=mL,
            M=mM,
            block_indices=m_block_indices,
            block_counts=m_block_counts,
            max_length=self.max_s_q,
            seq_offsets=m_cum_seqlen_q,
            softmax_scale=self.scale_softmax,
            stream=current_stream,
        )
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
        skip_compile: bool = False,
    ):
        self._logger.debug("Entering execute")
        current_stream = self._get_default_stream(current_stream)

        self._logger.debug("Reshaping tensors to kernel expected format")
        l_tensor = self._unpad_tensor_to_ndim(l_tensor, 2, "l_tensor")
        m_tensor = self._unpad_tensor_to_ndim(m_tensor, 2, "m_tensor")
        q_reshaped, k_reshaped, v_reshaped, o_reshaped, l_reshaped, m_reshaped = (
            self._reshape_tensors(
                q_tensor, k_tensor, v_tensor, o_tensor, l_tensor, m_tensor
            )
        )

        mQ = from_dlpack(q_reshaped, assumed_align=128)
        mK = from_dlpack(k_reshaped, assumed_align=128)
        mV = from_dlpack(v_reshaped, assumed_align=128)
        mO = from_dlpack(o_reshaped, assumed_align=128)
        mL = from_dlpack(l_reshaped)
        mM = from_dlpack(m_reshaped)
        m_block_indices = from_dlpack(block_indices_tensor)
        m_block_counts = from_dlpack(block_counts_tensor)
        m_cum_seqlen_q = from_dlpack(cum_seqlen_q_tensor)
        # m_cum_seqlen_k = from_dlpack(cum_seqlen_k_tensor) # unused

        scale_softmax = self.scale_softmax if scale_softmax is None else scale_softmax

        if not skip_compile:
            if self._compiled_kernel is None:
                raise RuntimeError("SelectionAttention kernel not compiled")
            self._logger.debug("Executing with compiled kernel")
            self._compiled_kernel(
                Q=mQ,
                K=mK,
                V=mV,
                O=mO,
                L=mL,
                M=mM,
                block_indices=m_block_indices,
                block_counts=m_block_counts,
                max_length=self.max_s_q,
                seq_offsets=m_cum_seqlen_q,
                softmax_scale=scale_softmax,
                stream=current_stream,
            )
            self._logger.debug("Executed with compiled kernel successfully")
        else:
            self._logger.debug("Executing without compiled kernel (JIT)")
            selection_attention = self._kernel(
                head_dim=self.head_dim,
                value_dim=self.value_dim,
                GQA_group_size=self.gqa_group_size,
                block_size=self.block_size,
                dtype=_convert_to_cutlass_data_type(self.dtype),
                acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
            )
            selection_attention(
                Q=mQ,
                K=mK,
                V=mV,
                O=mO,
                L=mL,
                M=mM,
                block_indices=m_block_indices,
                block_counts=m_block_counts,
                max_length=self.max_s_q,
                seq_offsets=m_cum_seqlen_q,
                softmax_scale=scale_softmax,
                stream=current_stream,
            )
            self._logger.debug("Executed successfully")


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Selection Attention Wrapper that returns output tensors directly.

    Returns:
        tuple: (o_tensor, l_tensor, m_tensor) - Output, logsumexp, and max tensors
    """
    _logger.debug(
        "selection_attention_wrapper: Creating empty output tensors o, l, and m"
    )

    max_s_q = (
        max(cum_seqlen_q_tensor[1:] - cum_seqlen_q_tensor[:-1]).item()
        if max_s_q is None
        else max_s_q
    )
    max_s_k = (
        max(cum_seqlen_k_tensor[1:] - cum_seqlen_k_tensor[:-1]).item()
        if max_s_k is None
        else max_s_k
    )

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
        _logger.debug(
            "selection_attention_wrapper: Using previously cached SelectionAttention object"
        )
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
        _logger.debug(
            "selection_attention_wrapper: No previously cached SelectionAttention object found, creating new SelectionAttention object"
        )
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

    return o_tensor, l_tensor, m_tensor
