# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from typing import Optional, Tuple
import logging
import math

from cuda.bindings import driver as cuda
import cutlass
import torch

import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_stream

from cudnn.api_base import APIBase, TupleDict
from cudnn.datatypes import _convert_to_cutlass_data_type

from .fmha_forward_sm100_d256 import BlackwellFusedMultiHeadAttentionForward
from ..fmha_utils import MaskEnum


class SdpafwdSm100D256(APIBase):
    """API class for d=256 SDPA forward (SM100+) using the FE OSS CUTE DSL kernel."""

    def __init__(
        self,
        sample_q: torch.Tensor,
        sample_k: torch.Tensor,
        sample_v: torch.Tensor,
        sample_o: torch.Tensor,
        sample_lse: torch.Tensor,
        sample_cum_seqlen_q: Optional[torch.Tensor] = None,
        sample_cum_seqlen_k: Optional[torch.Tensor] = None,
        max_s_q: Optional[int] = None,
        max_s_k: Optional[int] = None,
        qk_acc_dtype: torch.dtype = torch.float32,
        pv_acc_dtype: torch.dtype = torch.float32,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        is_causal: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        scale_softmax: Optional[float] = None,
        scale_output: float = 1.0,
    ):
        super().__init__()
        self._kernel = BlackwellFusedMultiHeadAttentionForward

        self._logger.warning("SdpafwdSm100D256 is an experimental API")
        self._logger.debug("Entering __init__")

        if sample_cum_seqlen_q is not None:
            if sample_cum_seqlen_q.numel() < 2:
                raise ValueError("sample_cum_seqlen_q must contain at least 2 elements")
            self._sample_s_q_max = int((sample_cum_seqlen_q[1:] - sample_cum_seqlen_q[:-1]).max().item())
        else:
            self._sample_s_q_max = None

        if sample_cum_seqlen_k is not None:
            if sample_cum_seqlen_k.numel() < 2:
                raise ValueError("sample_cum_seqlen_k must contain at least 2 elements")
            self._sample_s_k_max = int((sample_cum_seqlen_k[1:] - sample_cum_seqlen_k[:-1]).max().item())
        else:
            self._sample_s_k_max = None

        self.s_q_max = int(max_s_q) if max_s_q is not None else self._sample_s_q_max
        self.s_k_max = int(max_s_k) if max_s_k is not None else self._sample_s_k_max

        self.q_desc = self._make_tensor_desc(sample_q, name="q")
        self.k_desc = self._make_tensor_desc(sample_k, name="k")
        self.v_desc = self._make_tensor_desc(sample_v, name="v")
        self.o_desc = self._make_tensor_desc(sample_o, name="o")
        self.lse_desc = self._unpad_tensor_to_ndim(self._make_tensor_desc(sample_lse, name="lse"), self.q_desc.ndim - 1, "lse")
        self.cum_seqlen_q_desc = self._unpad_tensor_to_ndim(self._make_tensor_desc(sample_cum_seqlen_q, name="cum_seqlen_q"), 1, "cum_seqlen_q")
        self.cum_seqlen_k_desc = self._unpad_tensor_to_ndim(self._make_tensor_desc(sample_cum_seqlen_k, name="cum_seqlen_k"), 1, "cum_seqlen_k")

        self.qk_acc_dtype = qk_acc_dtype
        self.pv_acc_dtype = pv_acc_dtype
        self.mma_tiler_mn = mma_tiler_mn
        self.is_causal = is_causal
        self.window_size_left, self.window_size_right = window_size
        self.scale_softmax = scale_softmax
        self.scale_output = float(scale_output)

        self.input_layout = None
        self.dtype = None
        self.problem_size = None
        self.mask_type = None

        self.h_k = None
        self.h_q = None
        self.head_dim = None
        self.batch_size = None

        self._logger.debug("__init__ completed")

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        if self.cum_seqlen_q_desc is None and self.cum_seqlen_k_desc is None:
            self.input_layout = "B,H,S,D"
            for desc_name in ["q_desc", "k_desc", "v_desc", "o_desc"]:
                tensor_desc = getattr(self, desc_name)
                self._value_error_if(tensor_desc.ndim != 4, f"{tensor_desc.name} must be rank-4 for B,H,S,D layout, got {tensor_desc.ndim}")
                self._value_error_if(
                    tensor_desc.stride_order != (3, 1, 2, 0),
                    f"{tensor_desc.name} must have d,h,s,b stride order (3, 1, 2, 0), got {tensor_desc.stride_order}",
                )
                setattr(self, desc_name, tensor_desc.transpose(1, 2))
            self.lse_desc = self._unpad_tensor_to_ndim(self.lse_desc, 3, "lse")
            self._value_error_if(self.lse_desc is None, "sample_lse is required")
            self._value_error_if(not self.lse_desc.is_contiguous(), "lse_tensor must be contiguous for B,H,S,D layout")
        elif self.cum_seqlen_q_desc is not None and self.cum_seqlen_k_desc is not None:
            self.input_layout = "T,H,D"
            for desc_name in ["q_desc", "k_desc", "v_desc", "o_desc"]:
                tensor_desc = getattr(self, desc_name)
                if tensor_desc.ndim == 3:
                    setattr(self, desc_name, tensor_desc.unsqueeze(0))
                elif tensor_desc.ndim == 4:
                    self._value_error_if(
                        tensor_desc.shape[0] != 1,
                        f"{tensor_desc.name} must have batch dimension 1 for T,H,D layout (1, t, h, d), got {tensor_desc.shape[0]}",
                    )
                else:
                    raise ValueError(f"{tensor_desc.name} must be rank-3 or rank-4 for T,H,D layout, got {tensor_desc.ndim}")
            self.lse_desc = self._unpad_tensor_to_ndim(self.lse_desc, 2, "lse")
        else:
            raise ValueError(f"cum_seqlen_q and cum_seqlen_k must be both None or both not None, got {self.cum_seqlen_q_desc} and {self.cum_seqlen_k_desc}")

        b, s_qo, h_qo, d_qk = self.q_desc.shape
        _, s_kv, h_kv, d_v = self.v_desc.shape

        self._check_tensor_shape(self.q_desc, (b, s_qo, h_qo, d_qk), name="Q")
        self._check_tensor_shape(self.k_desc, (b, s_kv, h_kv, d_qk), name="K")
        self._check_tensor_shape(self.v_desc, (b, s_kv, h_kv, d_v), name="V")
        self._check_tensor_shape(self.o_desc, (b, s_qo, h_qo, d_v), name="O")

        self._value_error_if(self.lse_desc is None, "sample_lse is required")
        if self.input_layout == "B,H,S,D":
            self._check_tensor_shape(self.lse_desc, (b, h_qo, s_qo), name="LSE")
        else:
            self._check_tensor_shape(self.lse_desc, (s_qo, h_qo), name="LSE")

        self._value_error_if(d_qk != d_v, f"D_qk must match D_v, got {d_qk} and {d_v}")
        self._value_error_if(h_qo % h_kv != 0, f"H_q must be divisible by H_k, got {h_qo} and {h_kv}")

        self.dtype = self._check_dtype(self.q_desc, [torch.float16, torch.bfloat16], name="Q")
        for tensor_desc in [self.k_desc, self.v_desc, self.o_desc]:
            self._check_dtype(tensor_desc, self.dtype, name=tensor_desc.name, extra_error_msg=f"{tensor_desc.name} must match Q dtype")
        self._check_dtype(self.lse_desc, torch.float32, name="LSE")
        self._check_dtype(self.qk_acc_dtype, torch.float32, name="qk_acc_dtype", extra_error_msg="Only float32 accumulator is supported")
        self._check_dtype(self.pv_acc_dtype, torch.float32, name="pv_acc_dtype", extra_error_msg="Only float32 accumulator is supported")

        if self.input_layout == "T,H,D":
            self._check_dtype(self.cum_seqlen_q_desc, [torch.int32], name="cum_seqlen_q")
            self._check_dtype(self.cum_seqlen_k_desc, [torch.int32], name="cum_seqlen_k")
            self._value_error_if(
                self.cum_seqlen_q_desc.shape != self.cum_seqlen_k_desc.shape,
                f"cum_seqlen_q and cum_seqlen_k must have same shape, got {self.cum_seqlen_q_desc.shape} and {self.cum_seqlen_k_desc.shape}",
            )
            self.batch_size = int(self.cum_seqlen_q_desc.shape[0] - 1)
            self._value_error_if(self.batch_size <= 0, f"Invalid varlen batch_size={self.batch_size}")
            self._value_error_if(self._sample_s_q_max is None, "sample_cum_seqlen_q is required for T,H,D layout")
            self._value_error_if(self._sample_s_k_max is None, "sample_cum_seqlen_k is required for T,H,D layout")
            if self.s_q_max is None:
                self.s_q_max = self._sample_s_q_max
            if self.s_k_max is None:
                self.s_k_max = self._sample_s_k_max
            self._value_error_if(
                self.s_q_max < self._sample_s_q_max,
                f"max_s_q must be >= inferred max from sample_cum_seqlen_q ({self._sample_s_q_max}), got {self.s_q_max}",
            )
            self._value_error_if(
                self.s_k_max < self._sample_s_k_max,
                f"max_s_k must be >= inferred max from sample_cum_seqlen_k ({self._sample_s_k_max}), got {self.s_k_max}",
            )
        else:
            self.batch_size = b
            self.s_q_max = s_qo
            self.s_k_max = s_kv

        self.h_k = h_kv
        self.h_q = h_qo
        self.head_dim = d_qk

        self._value_error_if(self.head_dim != 256, f"head_dim must be 256, got {self.head_dim}")
        self._value_error_if(self.mma_tiler_mn != (128, 128), f"mma_tiler_mn must be (128, 128), got {self.mma_tiler_mn}")

        if self.scale_softmax is None or self.scale_softmax == 0.0:
            self.scale_softmax = 1.0 / math.sqrt(self.head_dim)

        if self.is_causal:
            self.window_size_right = 0

        self._value_error_if(
            self.window_size_left >= self.s_k_max - 1,
            f"window_size_left must be less than s_k_max - 1 (s_k_max={self.s_k_max}), got {self.window_size_left}",
        )
        self._value_error_if(
            self.window_size_right >= self.s_q_max - 1,
            f"window_size_right must be less than s_q_max - 1 (s_q_max={self.s_q_max}), got {self.window_size_right}",
        )
        if not self.is_causal:
            self._not_implemented_error_if(
                (self.window_size_left, self.window_size_right) != (-1, -1),
                f"window_size must be (-1, -1) for non-causal mode, got {self.window_size_left} and {self.window_size_right}",
            )

        self.mask_type = MaskEnum.WINDOW_MASK_INFERENCE
        if (not self.is_causal) and (self.window_size_left, self.window_size_right) == (-1, -1):
            if self.input_layout == "T,H,D" or s_kv % self.mma_tiler_mn[1] != 0:
                self.mask_type = MaskEnum.RESIDUAL_MASK

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device = self.q_desc.device
        major, minor = torch.cuda.get_device_capability(device)
        compute_capability = major * 10 + minor
        if compute_capability < 100:
            raise RuntimeError(f"SdpafwdSm100D256 requires SM100+, found SM{compute_capability} on device {device}")

        self.problem_size = (
            self.batch_size,
            self.s_q_max,
            self.s_k_max,
            self.h_q,
            self.h_k,
            self.head_dim,
        )

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def compile(self) -> None:
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            self._logger.debug("Kernel already compiled; skipping recompilation")
            return

        sdpa_fwd = self._kernel(
            qk_acc_dtype=_convert_to_cutlass_data_type(self.qk_acc_dtype),
            pv_acc_dtype=_convert_to_cutlass_data_type(self.pv_acc_dtype),
            mma_tiler=(*self.mma_tiler_mn, self.head_dim),
            is_persistent=False,
            mask_type=self.mask_type,
        )
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
        scale_softmax_log2 = self.scale_softmax * math.log2(math.exp(1.0))
        window_size_left = None if self.window_size_left < 0 else cutlass.Int32(self.window_size_left)
        window_size_right = None if self.window_size_right < 0 else cutlass.Int32(self.window_size_right)

        self._logger.debug("Compiling sdpa forward kernel with cute.compile")
        _compiled_kernel = cute.compile(
            sdpa_fwd,
            q_tensor=self._make_fake_cute_tensor_from_desc(self.q_desc, assumed_align=64),
            k_tensor=self._make_fake_cute_tensor_from_desc(self.k_desc, assumed_align=64),
            v_tensor=self._make_fake_cute_tensor_from_desc(self.v_desc, assumed_align=64),
            o_tensor=self._make_fake_cute_tensor_from_desc(self.o_desc, assumed_align=64),
            problem_size=self.problem_size,
            cum_seqlen_q=self._make_fake_cute_tensor_from_desc(self.cum_seqlen_q_desc, assumed_align=16),
            cum_seqlen_k=self._make_fake_cute_tensor_from_desc(self.cum_seqlen_k_desc, assumed_align=16),
            lse_tensor=self._make_fake_cute_tensor_from_desc(self.lse_desc, assumed_align=16),
            scale_softmax_log2=scale_softmax_log2,
            scale_softmax=self.scale_softmax,
            scale_output=self.scale_output,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            stream=fake_stream,
            options="--enable-tvm-ffi",
        )

        def tensor_api(
            q_tensor: torch.Tensor,
            k_tensor: torch.Tensor,
            v_tensor: torch.Tensor,
            o_tensor: torch.Tensor,
            lse_tensor: torch.Tensor,
            cum_seqlen_q: Optional[torch.Tensor],
            cum_seqlen_k: Optional[torch.Tensor],
            scale_softmax: float,
            scale_output: float,
            stream: cuda.CUstream,
        ) -> None:
            if self.input_layout == "B,H,S,D":
                q_tensor, k_tensor, v_tensor, o_tensor = (
                    q_tensor.transpose(1, 2),
                    k_tensor.transpose(1, 2),
                    v_tensor.transpose(1, 2),
                    o_tensor.transpose(1, 2),
                )
                lse_tensor = self._unpad_tensor_to_ndim(lse_tensor, 3, "lse_tensor")
            elif self.input_layout == "T,H,D":
                q_tensor, k_tensor, v_tensor, o_tensor = (
                    q_tensor.unsqueeze(0) if q_tensor.ndim == 3 else q_tensor,
                    k_tensor.unsqueeze(0) if k_tensor.ndim == 3 else k_tensor,
                    v_tensor.unsqueeze(0) if v_tensor.ndim == 3 else v_tensor,
                    o_tensor.unsqueeze(0) if o_tensor.ndim == 3 else o_tensor,
                )
                lse_tensor = self._unpad_tensor_to_ndim(lse_tensor, 2, "lse_tensor")
                cum_seqlen_q = self._unpad_tensor_to_ndim(cum_seqlen_q, 1, "cum_seqlen_q")
                cum_seqlen_k = self._unpad_tensor_to_ndim(cum_seqlen_k, 1, "cum_seqlen_k")
            else:
                raise NotImplementedError(f"Invalid input layout: {self.input_layout}")

            _compiled_kernel(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                self.problem_size,
                cum_seqlen_q,
                cum_seqlen_k,
                lse_tensor,
                scale_softmax * math.log2(math.exp(1.0)),
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
        lse_tensor: torch.Tensor,
        cum_seqlen_q_tensor: Optional[torch.Tensor] = None,
        cum_seqlen_k_tensor: Optional[torch.Tensor] = None,
        scale_softmax: Optional[float] = None,
        scale_output: Optional[float] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        self._logger.debug("Entering execute")
        current_stream = self._get_default_stream(current_stream)

        if self._compiled_kernel is None:
            raise RuntimeError("SdpafwdSm100D256 kernel is not compiled")

        if self.input_layout == "T,H,D":
            self._value_error_if(cum_seqlen_q_tensor is None, "cum_seqlen_q_tensor is required for T,H,D layout")
            self._value_error_if(cum_seqlen_k_tensor is None, "cum_seqlen_k_tensor is required for T,H,D layout")
        elif self.input_layout == "B,H,S,D":
            self._value_error_if(cum_seqlen_q_tensor is not None, "cum_seqlen_q_tensor must be None for B,H,S,D layout")
            self._value_error_if(cum_seqlen_k_tensor is not None, "cum_seqlen_k_tensor must be None for B,H,S,D layout")
        else:
            raise NotImplementedError(f"Invalid input layout: {self.input_layout}")

        scale_softmax_val = self.scale_softmax if (scale_softmax is None or scale_softmax == 0.0) else scale_softmax
        scale_output_val = self.scale_output if scale_output is None else float(scale_output)

        self._compiled_kernel(
            q_tensor=q_tensor,
            k_tensor=k_tensor,
            v_tensor=v_tensor,
            o_tensor=o_tensor,
            lse_tensor=lse_tensor,
            cum_seqlen_q=cum_seqlen_q_tensor,
            cum_seqlen_k=cum_seqlen_k_tensor,
            scale_softmax=scale_softmax_val,
            scale_output=scale_output_val,
            stream=current_stream,
        )
        self._logger.debug("Execute completed")


_logger = logging.getLogger(__name__)
_cache_of_SdpafwdSm100D256Objects = {}


def _allocate_lse_tensor(
    q_tensor: torch.Tensor,
    cum_seqlen_q_tensor: Optional[torch.Tensor],
) -> torch.Tensor:
    if cum_seqlen_q_tensor is None:
        if q_tensor.ndim != 4:
            raise ValueError(f"Expected BHSD q_tensor to be rank-4, got {q_tensor.ndim}")
        return torch.empty((q_tensor.shape[0], q_tensor.shape[1], q_tensor.shape[2]), dtype=torch.float32, device=q_tensor.device)

    if q_tensor.ndim == 3:
        return torch.empty((q_tensor.shape[0], q_tensor.shape[1]), dtype=torch.float32, device=q_tensor.device)
    if q_tensor.ndim == 4:
        return torch.empty((q_tensor.shape[1], q_tensor.shape[2]), dtype=torch.float32, device=q_tensor.device)
    raise ValueError(f"Expected THD q_tensor to be rank-3 or rank-4, got {q_tensor.ndim}")


def sdpa_fwd_wrapper_sm100_d256(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    cum_seqlen_q_tensor: Optional[torch.Tensor] = None,
    cum_seqlen_k_tensor: Optional[torch.Tensor] = None,
    max_s_q: Optional[int] = None,
    max_s_k: Optional[int] = None,
    qk_acc_dtype: torch.dtype = torch.float32,
    pv_acc_dtype: torch.dtype = torch.float32,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    is_causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    scale_softmax: Optional[float] = None,
    scale_output: float = 1.0,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Convenience wrapper for the d=256 SDPA forward SM100 kernel."""

    o_tensor = torch.empty_like(q_tensor)
    lse_tensor = _allocate_lse_tensor(q_tensor, cum_seqlen_q_tensor)

    cache_max_s_q = max_s_q
    cache_max_s_k = max_s_k
    if cache_max_s_q is None and cum_seqlen_q_tensor is not None:
        cache_max_s_q = int((cum_seqlen_q_tensor[1:] - cum_seqlen_q_tensor[:-1]).max().item())
    if cache_max_s_k is None and cum_seqlen_k_tensor is not None:
        cache_max_s_k = int((cum_seqlen_k_tensor[1:] - cum_seqlen_k_tensor[:-1]).max().item())

    cache_key = (
        q_tensor.shape,
        k_tensor.shape,
        v_tensor.shape,
        q_tensor.stride(),
        k_tensor.stride(),
        v_tensor.stride(),
        q_tensor.dtype,
        k_tensor.dtype,
        v_tensor.dtype,
        cum_seqlen_q_tensor.shape if cum_seqlen_q_tensor is not None else None,
        cum_seqlen_q_tensor.stride() if cum_seqlen_q_tensor is not None else None,
        cum_seqlen_q_tensor.dtype if cum_seqlen_q_tensor is not None else None,
        cum_seqlen_k_tensor.shape if cum_seqlen_k_tensor is not None else None,
        cum_seqlen_k_tensor.stride() if cum_seqlen_k_tensor is not None else None,
        cum_seqlen_k_tensor.dtype if cum_seqlen_k_tensor is not None else None,
        cache_max_s_q,
        cache_max_s_k,
        qk_acc_dtype,
        pv_acc_dtype,
        mma_tiler_mn,
        is_causal,
        window_size,
        scale_softmax,
        float(scale_output),
        q_tensor.device,
    )

    if cache_key in _cache_of_SdpafwdSm100D256Objects:
        _logger.debug("sdpa_fwd_wrapper_sm100_d256: Using cached SdpafwdSm100D256 object")
        sdpa_fwd = _cache_of_SdpafwdSm100D256Objects[cache_key]
    else:
        _logger.debug("sdpa_fwd_wrapper_sm100_d256: No cached object found, creating new SdpafwdSm100D256 object")
        sdpa_fwd = SdpafwdSm100D256(
            sample_q=q_tensor,
            sample_k=k_tensor,
            sample_v=v_tensor,
            sample_o=o_tensor,
            sample_lse=lse_tensor,
            sample_cum_seqlen_q=cum_seqlen_q_tensor,
            sample_cum_seqlen_k=cum_seqlen_k_tensor,
            max_s_q=max_s_q,
            max_s_k=max_s_k,
            qk_acc_dtype=qk_acc_dtype,
            pv_acc_dtype=pv_acc_dtype,
            mma_tiler_mn=mma_tiler_mn,
            is_causal=is_causal,
            window_size=window_size,
            scale_softmax=scale_softmax,
            scale_output=scale_output,
        )
        assert sdpa_fwd.check_support(), "Unsupported configuration"
        sdpa_fwd.compile()
        _cache_of_SdpafwdSm100D256Objects[cache_key] = sdpa_fwd

    sdpa_fwd.execute(
        q_tensor=q_tensor,
        k_tensor=k_tensor,
        v_tensor=v_tensor,
        o_tensor=o_tensor,
        lse_tensor=lse_tensor,
        cum_seqlen_q_tensor=cum_seqlen_q_tensor,
        cum_seqlen_k_tensor=cum_seqlen_k_tensor,
        scale_softmax=scale_softmax,
        scale_output=scale_output,
        current_stream=current_stream,
    )

    return TupleDict(
        o_tensor=o_tensor,
        lse_tensor=lse_tensor,
    )
