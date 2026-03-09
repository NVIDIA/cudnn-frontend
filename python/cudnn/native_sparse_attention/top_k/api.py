from .nsa_top_k_reduction_fwd import FineGrainedReductionQK

from cuda.bindings import driver as cuda
import torch
from typing import Tuple, Optional
import math

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_stream

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.api_base import APIBase, TupleDict


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

        self.q_desc = self._make_tensor_desc(sample_q, name="sample_q")
        self.k_desc = self._make_tensor_desc(sample_k, name="sample_k")
        self.lse_desc = self._make_tensor_desc(sample_lse, name="sample_lse")
        self.topk_scores_desc = self._make_tensor_desc(sample_topk_scores, name="sample_topk_scores")
        self.topk_indices_desc = self._make_tensor_desc(sample_topk_indices, name="sample_topk_indices")
        self.cum_seqlen_q_desc = self._make_tensor_desc(sample_cum_seqlen_q, name="sample_cum_seqlen_q")
        self.cum_seqlen_k_desc = self._make_tensor_desc(sample_cum_seqlen_k, name="sample_cum_seqlen_k")

        self.max_s_q = max_s_q
        if self.max_s_q is None and sample_cum_seqlen_q is not None:
            self._logger.warning("max_s_q not provided, inferring from cum_seqlen_q")
            self.max_s_q = (sample_cum_seqlen_q[1:] - sample_cum_seqlen_q[:-1]).max().item()
        self.max_s_k = max_s_k
        if self.max_s_k is None and sample_cum_seqlen_k is not None:
            self._logger.warning("max_s_k not provided, inferring from cum_seqlen_k")
            self.max_s_k = (sample_cum_seqlen_k[1:] - sample_cum_seqlen_k[:-1]).max().item()
        self.acc_dtype = acc_dtype
        self.k_value = k_value
        self.selection_block_size = selection_block_size
        self.compress_stride = compress_stride
        self.is_causal = is_causal
        self.mma_tiler_mn = mma_tiler_mn
        self.scale_softmax = scale_softmax

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        self._logger.debug("Checking shape normalization and validation")

        if self.cum_seqlen_q_desc is None and self.cum_seqlen_k_desc is None:
            self.input_layout = "B,H,S,D"
        elif self.cum_seqlen_q_desc is not None and self.cum_seqlen_k_desc is not None:
            self.input_layout = "T,H,D"

            if self.q_desc.ndim == 3:
                self._logger.info("reshaping T,H,D tensors to 1,H,T,D")
                self.q_desc = self.q_desc.unsqueeze(0).transpose(1, 2)
                self.k_desc = self.k_desc.unsqueeze(0).transpose(1, 2)
                self.lse_desc = self.lse_desc.unsqueeze(0).transpose(1, 2)
                self.topk_scores_desc = self.topk_scores_desc.unsqueeze(0).transpose(1, 2)
                self.topk_indices_desc = self.topk_indices_desc.unsqueeze(0).transpose(1, 2)
                self.cum_seqlen_q_desc = self._unpad_tensor_to_ndim(self.cum_seqlen_q_desc, 1, "sample_cum_seqlen_q")
                self.cum_seqlen_k_desc = self._unpad_tensor_to_ndim(self.cum_seqlen_k_desc, 1, "sample_cum_seqlen_k")
        else:
            raise ValueError(f"cum_seqlen_q and cum_seqlen_k must be both None or both not None, got {self.cum_seqlen_q_desc} and {self.cum_seqlen_k_desc}")

        b, h_q, s_q, d = self.q_desc.shape
        b, h_k, s_k, d = self.k_desc.shape
        self._check_tensor_shape(self.q_desc, (b, h_q, s_q, d), name="Q")
        self._check_tensor_shape(self.k_desc, (b, h_k, s_k, d), name="K")
        self.lse_desc = self._unpad_tensor_to_ndim(self.lse_desc, 3, "sample_lse")
        self._check_tensor_shape(self.lse_desc, (b, h_q, s_q), name="LSE")
        if self.lse_desc.stride[-1] != 1:
            self._logger.warning("lse_tensor is expected to have leading stride in last dimension of shape (b, h_q, s_q), copying lse_tensor to contiguous")
            self.lse_desc = self.lse_desc.contiguous()
        self._check_tensor_shape(self.topk_scores_desc, (b, h_k, s_q, self.k_value), name="TopK Scores")
        self._check_tensor_shape(self.topk_indices_desc, (b, h_k, s_q, self.k_value), name="TopK Indices")

        self.batch_size = b if (self.input_layout == "B,H,S,D") else (self.cum_seqlen_q_desc.shape[0] - 1)
        self.h_q, self.h_k, self.head_dim = h_q, h_k, d
        if self.input_layout == "B,H,S,D":
            self.max_s_q, self.max_s_k = s_q, s_k

        self._logger.debug("Checking dtypes")
        self.dtype = self._check_dtype(self.q_desc, [torch.float16, torch.bfloat16], name="Q")
        _ = self._check_dtype(self.k_desc, self.dtype, name="K", extra_error_msg="K must have the same dtype as Q")
        _ = self._check_dtype(self.lse_desc, self.acc_dtype, name="LSE", extra_error_msg="LSE must have the same dtype as Accumulator")
        _ = self._check_dtype(self.topk_scores_desc, self.acc_dtype, name="TopK Scores", extra_error_msg="TopK Scores must have the same dtype as Accumulator")
        _ = self._check_dtype(self.topk_indices_desc, torch.int32, name="TopK Indices")

        if self.input_layout == "T,H,D":
            self._check_dtype(self.cum_seqlen_q_desc, [torch.int32], name="cum_seqlen_q")
            self._check_dtype(self.cum_seqlen_k_desc, [torch.int32], name="cum_seqlen_k")

        # Environment checks
        self._logger.debug("Checking environment")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        compute_capability = major * 10 + minor
        if compute_capability < 100:
            raise RuntimeError(f"TopKReduction requires SM100+ compute capability, but found SM{compute_capability} on device {device}")
        if compute_capability == 103:
            raise RuntimeError("cuteDSL TopKReduction is not supported on SM103")

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def compile(self) -> None:
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            self._logger.debug("Kernel already compiled; skipping recompilation")
            return

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

        scale_softmax = 1.0 / math.sqrt(self.head_dim) if self.scale_softmax is None else self.scale_softmax
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

        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
        _compiled_kernel = cute.compile(
            topk_reduction,
            problem_size=problem_size,
            Q=self._make_fake_cute_tensor_from_desc(self.q_desc, assumed_align=16),  # .mark_layout_dynamic(leading_dim=3),
            K=self._make_fake_cute_tensor_from_desc(self.k_desc, assumed_align=16),  # .mark_layout_dynamic(leading_dim=3),
            LSE=self._make_fake_cute_tensor_from_desc(self.lse_desc, assumed_align=16),  # .mark_layout_dynamic(leading_dim=2),
            Topk_scores=self._make_fake_cute_tensor_from_desc(self.topk_scores_desc, assumed_align=16),  # .mark_layout_dynamic(leading_dim=3),
            Topk_indices=self._make_fake_cute_tensor_from_desc(self.topk_indices_desc, assumed_align=16),  # .mark_layout_dynamic(leading_dim=3),
            softmax_scale_log2_e=softmax_scale_log2_e,
            cumulative_s_q=(
                self._make_fake_cute_tensor_from_desc(self.cum_seqlen_q_desc, assumed_align=16) if self.input_layout == "T,H,D" else None
            ),  # .mark_layout_dynamic()
            cumulative_s_k=(
                self._make_fake_cute_tensor_from_desc(self.cum_seqlen_k_desc, assumed_align=16) if self.input_layout == "T,H,D" else None
            ),  # .mark_layout_dynamic()
            stream=fake_stream,
            options="--enable-tvm-ffi",
        )

        def tensor_api(
            problem_size,
            q_tensor,
            k_tensor,
            lse_tensor,
            topk_scores_tensor,
            topk_indices_tensor,
            softmax_scale_log2_e,
            cumulative_s_q_tensor,
            cumulative_s_k_tensor,
            stream,
        ):

            if self.input_layout == "T,H,D":
                if q_tensor.ndim == 3:
                    q_tensor = q_tensor.unsqueeze(0).transpose(1, 2)
                    k_tensor = k_tensor.unsqueeze(0).transpose(1, 2)

                    lse_tensor = lse_tensor.unsqueeze(0).transpose(1, 2)
                    topk_scores_tensor = topk_scores_tensor.unsqueeze(0).transpose(1, 2)
                    topk_indices_tensor = topk_indices_tensor.unsqueeze(0).transpose(1, 2)
            lse_tensor = self._unpad_tensor_to_ndim(lse_tensor, 3, "lse_tensor")
            if lse_tensor.stride(-1) != 1:
                self._logger.warning("lse_tensor is expected to have leading stride in last dimension of shape (b, h_q, s_q), copying lse_tensor to contiguous")
                lse_tensor = lse_tensor.contiguous()

            return _compiled_kernel(
                problem_size,
                q_tensor,
                k_tensor,
                lse_tensor,
                topk_scores_tensor,
                topk_indices_tensor,
                softmax_scale_log2_e,
                cumulative_s_q_tensor,
                cumulative_s_k_tensor,
                stream,
            )

        self._compiled_kernel = tensor_api

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
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        self._logger.debug("Entering execute")
        current_stream = self._get_default_stream(current_stream)

        scale_softmax = 1.0 / math.sqrt(self.head_dim) if self.scale_softmax is None else self.scale_softmax
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
        if self._compiled_kernel is None:
            raise ValueError("TopKReduction kernel not compiled")
        if self.input_layout == "T,H,D":
            if cumulative_s_q_tensor is None or cumulative_s_k_tensor is None:
                raise ValueError("cumulative_s_q_tensor and cumulative_s_k_tensor must be provided during execute for T,H,D layout")
        self._logger.debug("Executing with compiled kernel")
        self._compiled_kernel(
            problem_size=problem_size,
            q_tensor=q_tensor,
            k_tensor=k_tensor,
            lse_tensor=lse_tensor,
            topk_scores_tensor=topk_scores_tensor,
            topk_indices_tensor=topk_indices_tensor,
            softmax_scale_log2_e=softmax_scale_log2_e,
            cumulative_s_q_tensor=cumulative_s_q_tensor,
            cumulative_s_k_tensor=cumulative_s_k_tensor,
            stream=current_stream,
        )
        self._logger.debug("Executed with compiled kernel successfully")


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
) -> TupleDict:

    _logger.debug("topk_reduction_wrapper: Entering topk_reduction_wrapper")
    topk_scores_tensor, topk_indices_tensor = None, None
    if cum_seqlen_q_tensor is not None and cum_seqlen_k_tensor is not None:  # T,H,D
        total_seq_len_q = cum_seqlen_q_tensor[-1].item()
        h_k = k_tensor.shape[1]
        topk_scores_tensor = torch.empty(total_seq_len_q, h_k, k_value, dtype=acc_dtype, device=q_tensor.device)
        topk_indices_tensor = torch.empty(total_seq_len_q, h_k, k_value, dtype=torch.int32, device=q_tensor.device)
    elif cum_seqlen_q_tensor is None and cum_seqlen_k_tensor is None:  # B,H,S,D
        b, _, s_q, _ = q_tensor.shape
        _, h_k, _, _ = k_tensor.shape
        topk_scores_tensor = torch.empty(b, s_q, h_k, k_value, dtype=acc_dtype, device=q_tensor.device).transpose(1, 2)
        topk_indices_tensor = torch.empty(b, s_q, h_k, k_value, dtype=torch.int32, device=q_tensor.device).transpose(1, 2)
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
        _logger.debug("topk_reduction_wrapper: Using previously cached TopKReduction object")
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
        topk_reduction.compile()
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

    return TupleDict(
        topk_scores_tensor=topk_scores_tensor,
        topk_indices_tensor=topk_indices_tensor,
    )
