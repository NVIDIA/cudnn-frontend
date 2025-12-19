from .nsa_top_k_reduction_fwd import FineGrainedReductionQK

from cuda.bindings import driver as cuda
import torch
from typing import Tuple, Optional
import math

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.api_base import APIBase


class TopKReduction(APIBase):
    """
    Top-K Reduction for Native Sparse Attention.

    This class performs top-k reduction on attention scores to identify the most important
    key-value pairs for each query position.

    Note:
        The returned values calculated by the kernel exclude the first block and neighboring blocks from the reduction.
        As a result, it is expected to see rows of all -inf values and -1 values in the final topk_scores and topk_indices output tensors, respectively.
    """

    def __init__(
        self,
        sample_q: torch.Tensor,
        sample_k: torch.Tensor,
        sample_lse: torch.Tensor,
        sample_topk_scores: torch.Tensor,
        sample_topk_indices: torch.Tensor,
        sample_cum_seqlen_q: Optional[torch.Tensor] = None,
        sample_cum_seqlen_k: Optional[torch.Tensor] = None,
        max_s_q: Optional[int] = None,
        max_s_k: Optional[int] = None,
        acc_dtype: torch.dtype = torch.float32,
        k_value: int = 16,
        selection_block_size: int = 64,
        compress_stride: int = 32,
        is_causal: bool = True,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        scale_softmax: Optional[float] = None,
    ):
        super().__init__()
        self._kernel = FineGrainedReductionQK

        self._logger.warning("TopKReduction is an experimental API")
        self._logger.debug("Entering __init__")

        self.sample_q = sample_q
        self.sample_k = sample_k
        self.sample_lse = sample_lse
        self.sample_topk_scores = sample_topk_scores
        self.sample_topk_indices = sample_topk_indices
        self.sample_cum_seqlen_q = sample_cum_seqlen_q
        self.sample_cum_seqlen_k = sample_cum_seqlen_k

        self.max_s_q = max_s_q
        self.max_s_k = max_s_k
        self.acc_dtype = acc_dtype
        self.k_value = k_value
        self.selection_block_size = selection_block_size
        self.compress_stride = compress_stride
        self.is_causal = is_causal
        self.mma_tiler_mn = mma_tiler_mn
        self.scale_softmax = scale_softmax

        # Derived attributes (TODO)

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        # Shape normalization and validation
        self._logger.debug("Checking shape normalization and validation")
        if self.sample_cum_seqlen_q is None and self.sample_cum_seqlen_k is None:
            self.input_layout = "B,H,S,D"

        elif (
            self.sample_cum_seqlen_q is not None
            and self.sample_cum_seqlen_k is not None
        ):
            self.input_layout = "T,H,D"

            if self.sample_q.ndim == 3:
                self._logger.info("reshaping q_tensor from T,H,D to 1,H,T,D")
                self.sample_q = self.sample_q.unsqueeze(0).transpose(1, 2)
            if self.sample_k.ndim == 3:
                self._logger.info("reshaping k_tensor from T,H,D to 1,H,T,D")
                self.sample_k = self.sample_k.unsqueeze(0).transpose(1, 2)
            if self.sample_lse.ndim == 2:
                self._logger.info("reshaping lse_tensor from T,H to 1,T,H")
                self.sample_lse = self.sample_lse.unsqueeze(0).transpose(1, 2)
            elif self.sample_lse.ndim == 3:
                self._logger.info("reshaping lse_tensor from T,H,1 to 1,H,T")
                self.sample_lse = (
                    self._unpad_tensor_to_ndim(self.sample_lse, 2, "sample_lse")
                    .unsqueeze(0)
                    .transpose(1, 2)
                )
            if self.sample_topk_scores.ndim == 3:
                self._logger.info("reshaping topk_scores_tensor from T,H,D to 1,H,T,D")
                self.sample_topk_scores = self.sample_topk_scores.unsqueeze(
                    0
                ).transpose(1, 2)
            if self.sample_topk_indices.ndim == 3:
                self._logger.info("reshaping topk_indices_tensor from T,H,D to 1,H,T,D")
                self.sample_topk_indices = self.sample_topk_indices.unsqueeze(
                    0
                ).transpose(1, 2)
            if self.sample_cum_seqlen_q.ndim != 1:
                self._logger.info(
                    "cum_seqlen_q must be 1D tensor. Attempting to squeeze last dimension(s)"
                )
                for _ in range(self.sample_cum_seqlen_q.ndim - 1):
                    self.sample_cum_seqlen_q = self.sample_cum_seqlen_q.squeeze(-1)
                if self.sample_cum_seqlen_q.ndim != 1:
                    raise ValueError(
                        f"cum_seqlen_q must be 1D tensor, got {self.sample_cum_seqlen_q.ndim}D"
                    )
            if self.sample_cum_seqlen_k.ndim != 1:
                self._logger.info(
                    "cum_seqlen_k must be 1D tensor. Attempting to squeeze last dimension(s)"
                )
                for _ in range(self.sample_cum_seqlen_k.ndim - 1):
                    self.sample_cum_seqlen_k = self.sample_cum_seqlen_k.squeeze(-1)
                if self.sample_cum_seqlen_k.ndim != 1:
                    raise ValueError(
                        f"cum_seqlen_k must be 1D tensor, got {self.sample_cum_seqlen_k.ndim}D"
                    )
            if self.max_s_q is None:
                self._logger.warning(
                    "max_s_q not provided, inferring from cum_seqlen_q"
                )
                self.max_s_q = (
                    (self.sample_cum_seqlen_q[1:] - self.sample_cum_seqlen_q[:-1])
                    .max()
                    .item()
                )
            if self.max_s_k is None:
                self._logger.warning(
                    "max_s_k not provided, inferring from cum_seqlen_k"
                )
                self.max_s_k = (
                    (self.sample_cum_seqlen_k[1:] - self.sample_cum_seqlen_k[:-1])
                    .max()
                    .item()
                )
        else:
            raise ValueError(
                f"cum_seqlen_q and cum_seqlen_k must be None or both not None, got {self.sample_cum_seqlen_q} and {self.sample_cum_seqlen_k}"
            )

        b, h_q, s_q, d = self.sample_q.shape
        b, h_k, s_k, d = self.sample_k.shape
        if self.sample_q.shape != (b, h_q, s_q, d):
            raise ValueError(
                f"Input shape mismatch: expected Q tensor shape {b, h_q, s_q, d}, got {self.sample_q.shape}"
            )
        if self.sample_k.shape != (b, h_k, s_k, d):
            raise ValueError(
                f"Input shape mismatch: expected K tensor shape {b, h_k, s_k, d}, got {self.sample_k.shape}"
            )
        if self.sample_lse.shape == (b, h_q, s_q, 1):
            self._logger.info(
                "reshaping lse_tensor from (b, h_q, s_q, 1) to (b, h_q, s_q)"
            )
            self.sample_lse = self.sample_lse.squeeze(-1)
        if self.sample_lse.shape != (b, h_q, s_q):
            raise ValueError(
                f"Input shape mismatch: expected LSE tensor shape {b, h_q, s_q}, got {self.sample_lse.shape}"
            )
        if self.sample_lse.stride(-1) != 1:
            self._logger.warning(
                "lse_tensor is expected to have leading stride in last dimension of shape (b, h_q, s_q), copying lse_tensor to contiguous"
            )
            self.sample_lse = self.sample_lse.contiguous()
        if self.sample_topk_scores.shape != (b, h_k, s_q, self.k_value):
            raise ValueError(
                f"Input shape mismatch: expected TopK Scores tensor shape {b, h_k, s_q, self.k_value}, got {self.sample_topk_scores.shape}"
            )
        if self.sample_topk_indices.shape != (b, h_k, s_q, self.k_value):
            raise ValueError(
                f"Input shape mismatch: expected TopK Indices tensor shape {b, h_k, s_q, self.k_value}, got {self.sample_topk_indices.shape}"
            )

        self.batch_size = (
            b
            if (self.input_layout == "B,H,S,D")
            else (len(self.sample_cum_seqlen_q) - 1)
        )
        self.h_q, self.h_k, self.head_dim = h_q, h_k, d
        if self.input_layout == "B,H,S,D":
            self.max_s_q, self.max_s_k = s_q, s_k

        self._logger.debug("Checking dtypes")
        if self.sample_q.dtype != self.sample_k.dtype:
            raise ValueError(
                f"Q and K must have the same dtype, got {self.sample_q.dtype} and {self.sample_k.dtype}"
            )
        self.dtype = self.sample_q.dtype
        if self.sample_lse.dtype != self.acc_dtype:
            raise ValueError(
                f"LSE and Accumulator must have the same dtype, got {self.sample_lse.dtype} and {self.acc_dtype}"
            )
        if self.sample_topk_scores.dtype != self.acc_dtype:
            raise ValueError(
                f"TopK Scores and Accumulator must have the same dtype, got {self.sample_topk_scores.dtype} and {self.acc_dtype}"
            )
        if self.sample_topk_indices.dtype != torch.int32:
            raise ValueError(
                f"TopK Indices must be int32, got {self.sample_topk_indices.dtype}"
            )
        if self.input_layout == "T,H,D":
            if (
                self.sample_cum_seqlen_q.dtype != torch.int32
                or self.sample_cum_seqlen_k.dtype != torch.int32
            ):
                raise ValueError(
                    f"cum_seqlen_q and cum_seqlen_k tensors must be int32, got {self.sample_cum_seqlen_q.dtype} and {self.sample_cum_seqlen_k.dtype}"
                )

        # Environment checks
        self._logger.debug("Checking environment")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        compute_capability = major * 10 + minor
        if compute_capability < 100:
            raise RuntimeError(
                f"TopKReduction requires SM100+ compute capability, but found SM{compute_capability} on device {device}"
            )
        if compute_capability == 103:
            raise RuntimeError("cuteDSL TopKReduction is not supported on SM103")

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def compile(self, current_stream: Optional[cuda.CUstream] = None) -> None:
        self._logger.debug("Entering compile")
        current_stream = self._get_default_stream(current_stream)
        self._ensure_support_checked()

        mma_tiler = (*self.mma_tiler_mn, self.head_dim)

        topk_reduction = self._kernel(
            element_dtype=_convert_to_cutlass_data_type(self.dtype),
            acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
            k_value=self.k_value,
            selection_block_size=self.selection_block_size,
            compress_block_sliding_stride=self.compress_stride,
            mma_tiler=mma_tiler,
            is_causal=self.is_causal,
        )

        scale_softmax = (
            1.0 / math.sqrt(self.head_dim)
            if self.scale_softmax is None
            else self.scale_softmax
        )
        log2_e = math.log2(math.e)
        softmax_scale_log2_e = scale_softmax * log2_e
        problem_size = (
            self.batch_size,
            self.max_s_q,
            self.max_s_k,
            self.h_q,
            self.h_k,
            self.head_dim,
        )

        sample_q_cute = from_dlpack(
            self.sample_q, assumed_align=16
        ).mark_layout_dynamic(leading_dim=3)
        sample_k_cute = from_dlpack(
            self.sample_k, assumed_align=16
        ).mark_layout_dynamic(leading_dim=3)
        sample_lse_cute = from_dlpack(
            self.sample_lse, assumed_align=16
        ).mark_layout_dynamic(leading_dim=2)
        sample_topk_scores_cute = from_dlpack(
            self.sample_topk_scores, assumed_align=16
        ).mark_layout_dynamic(leading_dim=3)
        sample_topk_indices_cute = from_dlpack(
            self.sample_topk_indices, assumed_align=16
        ).mark_layout_dynamic(leading_dim=3)
        sample_cum_seqlen_q_cute = (
            from_dlpack(self.sample_cum_seqlen_q).mark_layout_dynamic()
            if self.input_layout == "T,H,D"
            else None
        )
        sample_cum_seqlen_k_cute = (
            from_dlpack(self.sample_cum_seqlen_k).mark_layout_dynamic()
            if self.input_layout == "T,H,D"
            else None
        )

        self._compiled_kernel = cute.compile(
            topk_reduction,
            problem_size=problem_size,
            Q=sample_q_cute,
            K=sample_k_cute,
            LSE=sample_lse_cute,
            Topk_scores=sample_topk_scores_cute,
            Topk_indices=sample_topk_indices_cute,
            softmax_scale_log2_e=softmax_scale_log2_e,
            cumulative_s_q=sample_cum_seqlen_q_cute,
            cumulative_s_k=sample_cum_seqlen_k_cute,
            stream=current_stream,
        )
        self._logger.debug("Kernel compiled successfully")

    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        lse_tensor: torch.Tensor,
        topk_scores_tensor: torch.Tensor,
        topk_indices_tensor: torch.Tensor,
        cumulative_s_q_tensor: Optional[torch.Tensor] = None,
        cumulative_s_k_tensor: Optional[torch.Tensor] = None,
        skip_compile: bool = False,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        self._logger.debug("Entering execute")
        current_stream = self._get_default_stream(current_stream)

        if self.input_layout == "T,H,D":
            if cumulative_s_q_tensor is None or cumulative_s_k_tensor is None:
                raise ValueError(
                    "cumulative_s_q_tensor and cumulative_s_k_tensor are required when using T,H,D layout"
                )
            if q_tensor.ndim == 3:
                self._logger.info("reshaping q_tensor from T,H,D to 1,H,T,D")
                q_tensor = q_tensor.unsqueeze(0).transpose(1, 2)
            if k_tensor.ndim == 3:
                self._logger.info("reshaping k_tensor from T,H,D to 1,H,T,D")
                k_tensor = k_tensor.unsqueeze(0).transpose(1, 2)
            if lse_tensor.ndim == 2:
                self._logger.info("reshaping lse_tensor from T,H to 1,H,T")
                lse_tensor = lse_tensor.unsqueeze(0).transpose(1, 2)
            elif lse_tensor.ndim == 3:
                self._logger.info("reshaping lse_tensor from T,H,1 to 1,H,T")
                lse_tensor = (
                    self._unpad_tensor_to_ndim(lse_tensor, 2, "lse_tensor")
                    .unsqueeze(0)
                    .transpose(1, 2)
                )
            if topk_scores_tensor.ndim == 3:
                self._logger.info("reshaping topk_scores_tensor from T,H,D to 1,H,T,D")
                topk_scores_tensor = topk_scores_tensor.unsqueeze(0).transpose(1, 2)
            if topk_indices_tensor.ndim == 3:
                self._logger.info("reshaping topk_indices_tensor from T,H,D to 1,H,T,D")
                topk_indices_tensor = topk_indices_tensor.unsqueeze(0).transpose(1, 2)

        if lse_tensor.ndim == 4:
            self._logger.info("reshaping lse_tensor to remove trailing dimension")
            lse_tensor = lse_tensor.squeeze(-1)
        if lse_tensor.stride(-1) != 1:
            self._logger.warning(
                "lse_tensor is expected to have leading stride in last dimension of shape (b, h_q, s_q), copying lse_tensor to contiguous"
            )
            lse_tensor = lse_tensor.contiguous()

        q_cute = from_dlpack(q_tensor, assumed_align=16).mark_layout_dynamic(
            leading_dim=3
        )
        k_cute = from_dlpack(k_tensor, assumed_align=16).mark_layout_dynamic(
            leading_dim=3
        )
        lse_cute = from_dlpack(lse_tensor, assumed_align=16).mark_layout_dynamic(
            leading_dim=2
        )
        topk_scores_cute = from_dlpack(
            topk_scores_tensor, assumed_align=16
        ).mark_layout_dynamic(leading_dim=3)
        topk_indices_cute = from_dlpack(
            topk_indices_tensor, assumed_align=16
        ).mark_layout_dynamic(leading_dim=3)
        cumulative_s_q_cute = (
            from_dlpack(cumulative_s_q_tensor).mark_layout_dynamic()
            if self.input_layout == "T,H,D"
            else None
        )
        cumulative_s_k_cute = (
            from_dlpack(cumulative_s_k_tensor).mark_layout_dynamic()
            if self.input_layout == "T,H,D"
            else None
        )
        scale_softmax = (
            1.0 / math.sqrt(self.head_dim)
            if self.scale_softmax is None
            else self.scale_softmax
        )
        log2_e = math.log2(math.e)
        softmax_scale_log2_e = scale_softmax * log2_e
        problem_size = (
            self.batch_size,
            self.max_s_q,
            self.max_s_k,
            self.h_q,
            self.h_k,
            self.head_dim,
        )

        if not skip_compile:
            if self._compiled_kernel is None:
                raise ValueError("TopKReduction kernel not compiled")
            self._logger.debug("Executing with compiled kernel")
            self._compiled_kernel(
                problem_size=problem_size,
                Q=q_cute,
                K=k_cute,
                LSE=lse_cute,
                Topk_scores=topk_scores_cute,
                Topk_indices=topk_indices_cute,
                softmax_scale_log2_e=softmax_scale_log2_e,
                cumulative_s_q=cumulative_s_q_cute,
                cumulative_s_k=cumulative_s_k_cute,
                stream=current_stream,
            )
            self._logger.debug("Executed with compiled kernel successfully")
        else:
            self._logger.debug("Executing without compiled kernel (JIT)")
            topk_reduction = self._kernel(
                element_dtype=_convert_to_cutlass_data_type(self.dtype),
                acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
                k_value=self.k_value,
                selection_block_size=self.selection_block_size,
                compress_block_sliding_stride=self.compress_stride,
                mma_tiler=(*self.mma_tiler_mn, self.head_dim),
                is_causal=self.is_causal,
            )
            topk_reduction(
                problem_size=problem_size,
                Q=q_cute,
                K=k_cute,
                LSE=lse_cute,
                Topk_scores=topk_scores_cute,
                Topk_indices=topk_indices_cute,
                softmax_scale_log2_e=softmax_scale_log2_e,
                cumulative_s_q=cumulative_s_q_cute,
                cumulative_s_k=cumulative_s_k_cute,
                stream=current_stream,
            )
            self._logger.debug("Executed successfully")


import logging

_logger = logging.getLogger(__name__)
_cache_of_TopKReductionObjects = {}


def topk_reduction_wrapper(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    lse_tensor: torch.Tensor,
    cum_seqlen_q_tensor: Optional[torch.Tensor] = None,
    cum_seqlen_k_tensor: Optional[torch.Tensor] = None,
    max_s_q: Optional[int] = None,
    max_s_k: Optional[int] = None,
    acc_dtype: torch.dtype = torch.float32,
    k_value: int = 16,
    selection_block_size: int = 64,
    compress_stride: int = 32,
    is_causal: bool = True,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    scale_softmax: Optional[float] = None,
    current_stream: Optional[cuda.CUstream] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    _logger.debug("topk_reduction_wrapper: Entering topk_reduction_wrapper")
    topk_scores_tensor, topk_indices_tensor = None, None
    if cum_seqlen_q_tensor is not None and cum_seqlen_k_tensor is not None:  # T,H,D
        total_seq_len_q = cum_seqlen_q_tensor[-1].item()
        h_k = k_tensor.shape[1]
        topk_scores_tensor = torch.empty(
            total_seq_len_q, h_k, k_value, dtype=acc_dtype, device=q_tensor.device
        )
        topk_indices_tensor = torch.empty(
            total_seq_len_q, h_k, k_value, dtype=torch.int32, device=q_tensor.device
        )
    elif cum_seqlen_q_tensor is None and cum_seqlen_k_tensor is None:  # B,H,S,D
        b, _, s_q, _ = q_tensor.shape
        _, h_k, _, _ = k_tensor.shape
        topk_scores_tensor = torch.empty(
            b, s_q, h_k, k_value, dtype=acc_dtype, device=q_tensor.device
        ).transpose(1, 2)
        topk_indices_tensor = torch.empty(
            b, s_q, h_k, k_value, dtype=torch.int32, device=q_tensor.device
        ).transpose(1, 2)
    else:
        raise ValueError(
            f"cum_seqlen_q_tensor and cum_seqlen_k_tensor must either both be None (B,H,S,D) or both not None (T,H,D), got {cum_seqlen_q_tensor} and {cum_seqlen_k_tensor}"
        )

    cache_key = (
        q_tensor.shape,
        k_tensor.shape,
        lse_tensor.shape,
        cum_seqlen_q_tensor.shape if cum_seqlen_q_tensor is not None else None,
        cum_seqlen_k_tensor.shape if cum_seqlen_k_tensor is not None else None,
        q_tensor.dtype,
        k_tensor.dtype,
        lse_tensor.dtype,
        cum_seqlen_q_tensor.dtype if cum_seqlen_q_tensor is not None else None,
        cum_seqlen_k_tensor.dtype if cum_seqlen_k_tensor is not None else None,
        q_tensor.stride(),
        k_tensor.stride(),
        lse_tensor.stride(),
        cum_seqlen_q_tensor.stride() if cum_seqlen_q_tensor is not None else None,
        cum_seqlen_k_tensor.stride() if cum_seqlen_k_tensor is not None else None,
        max_s_q,
        max_s_k,
        acc_dtype,
        k_value,
        selection_block_size,
        compress_stride,
        is_causal,
        mma_tiler_mn,
        scale_softmax,
    )

    if cache_key in _cache_of_TopKReductionObjects:
        _logger.debug(
            "topk_reduction_wrapper: Using previously cached TopKReduction object"
        )
        topk_reduction = _cache_of_TopKReductionObjects[cache_key]
        topk_reduction.execute(
            q_tensor=q_tensor,
            k_tensor=k_tensor,
            lse_tensor=lse_tensor,
            topk_scores_tensor=topk_scores_tensor,
            topk_indices_tensor=topk_indices_tensor,
            cumulative_s_q_tensor=cum_seqlen_q_tensor,
            cumulative_s_k_tensor=cum_seqlen_k_tensor,
            current_stream=current_stream,
        )
        return topk_scores_tensor, topk_indices_tensor
    else:
        topk_reduction = TopKReduction(
            sample_q=q_tensor,
            sample_k=k_tensor,
            sample_lse=lse_tensor,
            sample_topk_scores=topk_scores_tensor,
            sample_topk_indices=topk_indices_tensor,
            sample_cum_seqlen_q=cum_seqlen_q_tensor,
            sample_cum_seqlen_k=cum_seqlen_k_tensor,
            max_s_q=max_s_q,
            max_s_k=max_s_k,
            acc_dtype=acc_dtype,
            k_value=k_value,
            selection_block_size=selection_block_size,
            compress_stride=compress_stride,
            is_causal=is_causal,
            mma_tiler_mn=mma_tiler_mn,
            scale_softmax=scale_softmax,
        )
        assert topk_reduction.check_support()
        topk_reduction.compile(current_stream=current_stream)
        topk_reduction.execute(
            q_tensor=q_tensor,
            k_tensor=k_tensor,
            lse_tensor=lse_tensor,
            topk_scores_tensor=topk_scores_tensor,
            topk_indices_tensor=topk_indices_tensor,
            cumulative_s_q_tensor=cum_seqlen_q_tensor,
            cumulative_s_k_tensor=cum_seqlen_k_tensor,
            current_stream=current_stream,
        )
        _cache_of_TopKReductionObjects[cache_key] = topk_reduction

    return topk_scores_tensor, topk_indices_tensor
