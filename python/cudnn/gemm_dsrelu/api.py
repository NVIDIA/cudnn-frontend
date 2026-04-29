# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda
from cutlass.cute.runtime import make_fake_stream

from cudnn.api_base import APIBase, TupleDict, ceil_div, is_power_of_2
from cudnn.datatypes import _convert_to_cutlass_data_type

from .dense_blockscaled_gemm_persistent_dsrelu_quant import (
    Sm100BlockScaledPersistentDenseGemmKernel,
)


def _major_from_stride_order(stride_order: Tuple[int, ...], mode0_label: str, mode1_label: str) -> str:
    if stride_order == (0, 1, 2):
        return mode0_label
    if stride_order == (1, 0, 2):
        return mode1_label
    raise ValueError(f"Unsupported stride order {stride_order}")


class GemmDsreluSm100(APIBase):
    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_b: torch.Tensor,
        sample_c: torch.Tensor,
        sample_d: torch.Tensor,
        sample_dprob: torch.Tensor,
        sample_sfa: torch.Tensor,
        sample_sfb: torch.Tensor,
        sample_prob: torch.Tensor,
        sample_sfd: Optional[torch.Tensor] = None,
        sample_amax: Optional[torch.Tensor] = None,
        sample_norm_const: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        acc_dtype: torch.dtype = torch.float32,
        mma_tiler_mn: Tuple[int, int] = (256, 256),
        cluster_shape_mn: Optional[Tuple[int, int]] = None,
        sf_vec_size: int = 16,
        vector_f32: bool = False,
    ):
        super().__init__()

        self._warn_experimental_api()

        self.a_desc = self._make_tensor_desc(sample_a, name="sample_a")
        self.b_desc = self._make_tensor_desc(sample_b, name="sample_b")
        self.c_desc = self._make_tensor_desc(sample_c, name="sample_c")
        self.d_desc = self._make_tensor_desc(sample_d, name="sample_d")
        self.dprob_desc = self._make_tensor_desc(sample_dprob, name="sample_dprob")
        self.sfa_desc = self._make_tensor_desc(sample_sfa, name="sample_sfa")
        self.sfb_desc = self._make_tensor_desc(sample_sfb, name="sample_sfb")
        self.prob_desc = self._make_tensor_desc(sample_prob, name="sample_prob")
        self.sfd_desc = self._make_tensor_desc(sample_sfd, name="sample_sfd")
        self.amax_desc = self._unpad_tensor_to_ndim(self._make_tensor_desc(sample_amax, name="sample_amax"), 1, "amax")
        self.norm_const_desc = self._unpad_tensor_to_ndim(
            self._make_tensor_desc(sample_norm_const, name="sample_norm_const"),
            1,
            "norm_const",
        )

        self.alpha = alpha
        self.acc_dtype = acc_dtype
        self.mma_tiler_mn = mma_tiler_mn
        self.cluster_shape_mn = cluster_shape_mn if cluster_shape_mn is not None else ((2, 1) if mma_tiler_mn[0] == 256 else (1, 1))
        self.sf_vec_size = sf_vec_size
        self.vector_f32 = vector_f32
        self.num_cluster_overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))

        self._interpret_uint8_as_fp4x2 = True
        self._kernel = Sm100BlockScaledPersistentDenseGemmKernel

    def check_support(self) -> bool:
        m, k, l = self._tensor_shape(self.a_desc, name="sample_a")
        n, b_k, b_l = self._tensor_shape(self.b_desc, name="sample_b")

        self._value_error_if((b_k, b_l) != (k, l), f"B shape mismatch: expected (*, {k}, {l}), got {(n, b_k, b_l)}")
        self._check_tensor_shape(self.c_desc, (m, n, l), "C")
        self._check_tensor_shape(self.d_desc, (m, n, l), "D")
        self._check_tensor_shape(self.prob_desc, (m, 1, l), "prob")
        self._check_tensor_shape(self.dprob_desc, (m, 1, l), "dprob")

        rest_k = ceil_div(ceil_div(k, self.sf_vec_size), 4)
        self._check_tensor_shape(self.sfa_desc, (32, 4, ceil_div(m, 128), 4, rest_k, l), "SFA")
        self._check_tensor_shape(self.sfb_desc, (32, 4, ceil_div(n, 128), 4, rest_k, l), "SFB")

        if self.sfd_desc is not None:
            rest_n = ceil_div(ceil_div(n, self.sf_vec_size), 4)
            self._check_tensor_shape(self.sfd_desc, (32, 4, ceil_div(m, 128), 4, rest_n, l), "SFD")

        self._check_tensor_shape(self.amax_desc, (1,), "amax")
        self._check_tensor_shape(self.norm_const_desc, (1,), "norm_const")

        self.ab_dtype = self._check_dtype(
            self.a_desc,
            dtype=[torch.float4_e2m1fn_x2, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2],
            name="A",
        )
        self._check_dtype(self.b_desc, dtype=self.ab_dtype, name="B", extra_error_msg="A and B must have the same dtype")
        self.c_dtype = self._check_dtype(
            self.c_desc,
            dtype=[torch.float16, torch.bfloat16, torch.float32],
            name="C",
        )
        self.d_dtype = self._check_dtype(
            self.d_desc,
            dtype=[torch.float16, torch.bfloat16, torch.float32, torch.float8_e4m3fn, torch.float8_e5m2],
            name="D",
        )
        self._check_dtype(self.prob_desc, dtype=torch.float32, name="prob")
        self._check_dtype(self.dprob_desc, dtype=torch.float32, name="dprob")

        self.sf_dtype = self._check_dtype(self.sfa_desc, dtype=[torch.float8_e8m0fnu, torch.float8_e4m3fn], name="SFA")
        self._check_dtype(self.sfb_desc, dtype=self.sf_dtype, name="SFB", extra_error_msg="SFB must have the same dtype as SFA")
        self._check_dtype(self.sfd_desc, dtype=self.sf_dtype, name="SFD", extra_error_msg="SFD must have the same dtype as SFA")

        self._check_dtype(self.acc_dtype, dtype=torch.float32, name="Accumulator")

        self._value_error_if(self.sf_vec_size not in {16, 32}, f"sf_vec_size must be 16 or 32, got {self.sf_vec_size}")
        self._value_error_if(
            self._is_fp8(self.d_desc) and (self.sfd_desc is None or self.norm_const_desc is None), "sfd and norm_const are required when D is FP8"
        )
        self._value_error_if(
            self._is_fp4x2(self.ab_dtype) and self.d_dtype in {torch.float8_e4m3fn, torch.float8_e5m2}, "FP4 input with FP8 output is not supported"
        )

        a_major = _major_from_stride_order(self.a_desc.stride_order, "m", "k")
        b_major = _major_from_stride_order(self.b_desc.stride_order, "n", "k")
        c_major = _major_from_stride_order(self.c_desc.stride_order, "m", "n")
        d_major = _major_from_stride_order(self.d_desc.stride_order, "m", "n")
        self._value_error_if(c_major != d_major, f"C and D must share the same layout, got {c_major} and {d_major}")

        self._value_error_if(
            self.mma_tiler_mn[0] not in {128, 256} or self.mma_tiler_mn[1] not in {64, 128, 192, 256},
            f"Unsupported mma_tiler_mn {self.mma_tiler_mn}",
        )
        self._value_error_if(
            not (
                self.cluster_shape_mn[0] > 0
                and self.cluster_shape_mn[1] > 0
                and self.cluster_shape_mn[0] * self.cluster_shape_mn[1] <= 16
                and is_power_of_2(self.cluster_shape_mn[0])
                and is_power_of_2(self.cluster_shape_mn[1])
            ),
            f"Invalid cluster shape {self.cluster_shape_mn}",
        )

        self._runtime_error_if(not torch.cuda.is_available(), "CUDA is not available")
        major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        self._runtime_error_if(major * 10 + minor < 100, f"GemmDsreluSm100 requires SM100+, found SM{major}{minor}")

        self._value_error_if(
            not self._kernel.can_implement(
                ab_dtype=_convert_to_cutlass_data_type(self.ab_dtype, interpret_uint8_as_fp4x2=self._interpret_uint8_as_fp4x2),
                sf_dtype=_convert_to_cutlass_data_type(self.sf_dtype),
                sf_vec_size=self.sf_vec_size,
                d_dtype=_convert_to_cutlass_data_type(self.d_dtype),
                mma_tiler_mn=self.mma_tiler_mn,
                cluster_shape_mn=self.cluster_shape_mn,
                m=m,
                n=n,
                k=k,
                l=l,
                a_major=a_major,
                b_major=b_major,
                d_major=d_major,
            ),
            "Unsupported configuration for dense dsReLU kernel",
        )

        self._is_supported = True
        return True

    def compile(self) -> None:
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return

        gemm = self._kernel(
            sf_vec_size=self.sf_vec_size,
            mma_tiler_mn=self.mma_tiler_mn,
            cluster_shape_mn=self.cluster_shape_mn,
            vector_f32=self.vector_f32,
        )

        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(self.cluster_shape_mn[0] * self.cluster_shape_mn[1])
        max_active_clusters -= self.num_cluster_overlap_margin
        self._value_error_if(max_active_clusters <= 0, "max_active_clusters must be > 0 after overlap margin")

        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
        epilogue_op = lambda x, y: cute.where(x > 0, x, cute.full_like(x, 0)) * 2 * y
        use_full_dynamic = os.environ.get(_DENSE_GEMM_DYNAMIC_MNKL_ENV) is not None
        use_dynamic_m = not use_full_dynamic and os.environ.get(_DENSE_GEMM_DYNAMIC_M_ENV) is not None

        if use_dynamic_m:
            valid_m = cute.sym_int()

            a_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.a_desc.dtype,
                shape=(valid_m, *self.a_desc.shape[1:]),
                stride_order=self.a_desc.stride_order,
            )
            b_cute_fake = self._make_fake_cute_tensor_from_desc(self.b_desc, assumed_align=16)
            c_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.c_desc.dtype,
                shape=(valid_m, *self.c_desc.shape[1:]),
                stride_order=self.c_desc.stride_order,
            )
            d_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.d_desc.dtype,
                shape=(valid_m, *self.d_desc.shape[1:]),
                stride_order=self.d_desc.stride_order,
            )
            prob_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.prob_desc.dtype,
                shape=(valid_m, *self.prob_desc.shape[1:]),
                stride_order=self.prob_desc.stride_order,
            )
            dprob_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.dprob_desc.dtype,
                shape=(valid_m, *self.dprob_desc.shape[1:]),
                stride_order=self.dprob_desc.stride_order,
            )

            tensor_m_128 = cute.sym_int()
            stride_tensor_m_128 = cute.sym_int(divisibility=32 * 4 * 4)
            sfa_cute_fake = self._make_fake_cute_tensor(
                dtype=self.sfa_desc.dtype,
                shape=(32, 4, tensor_m_128, 4, self.sfa_desc.shape[4], self.sfa_desc.shape[5]),
                stride=(16, 4, self.sfa_desc.stride[2], 1, 512, stride_tensor_m_128),
            )
            sfb_cute_fake = self._make_fake_cute_tensor_from_desc(self.sfb_desc, assumed_align=16)

            sfd_cute_fake = None
            if self.sfd_desc is not None:
                stride_sfd_m = cute.sym_int(divisibility=32 * 4 * 4)
                sfd_cute_fake = self._make_fake_cute_tensor(
                    dtype=self.sfd_desc.dtype,
                    shape=(32, 4, tensor_m_128, 4, self.sfd_desc.shape[4], self.sfd_desc.shape[5]),
                    stride=(16, 4, self.sfd_desc.stride[2], 1, 512, stride_sfd_m),
                )
        elif use_full_dynamic:
            valid_m = cute.sym_int()
            n_sym = cute.sym_int()
            k_sym = cute.sym_int()
            l_sym = cute.sym_int()

            a_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.a_desc.dtype,
                shape=(valid_m, k_sym, l_sym),
                stride_order=self.a_desc.stride_order,
                dynamic_mode=self.a_desc.stride_order[0],
                divisibility=32 if self._is_fp4x2(self.ab_dtype) else 16,
            )
            b_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.b_desc.dtype,
                shape=(n_sym, k_sym, l_sym),
                stride_order=self.b_desc.stride_order,
                dynamic_mode=self.b_desc.stride_order[0],
                divisibility=32 if self._is_fp4x2(self.ab_dtype) else 16,
            )
            c_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.c_desc.dtype,
                shape=(valid_m, n_sym, l_sym),
                stride_order=self.c_desc.stride_order,
                dynamic_mode=self.c_desc.stride_order[0],
                divisibility=8 if self._is_f16(self.c_desc.dtype) else 16,
            )
            d_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.d_desc.dtype,
                shape=(valid_m, n_sym, l_sym),
                stride_order=self.d_desc.stride_order,
                dynamic_mode=self.d_desc.stride_order[0],
                divisibility=8 if self._is_f16(self.d_desc.dtype) else 16,
            )
            prob_cute_fake = self._make_fake_cute_tensor(
                dtype=self.prob_desc.dtype,
                shape=(valid_m, *self.prob_desc.shape[1:-1], l_sym),
                stride=(1, 1, valid_m),
            )
            dprob_cute_fake = self._make_fake_cute_tensor(
                dtype=self.dprob_desc.dtype,
                shape=(valid_m, *self.dprob_desc.shape[1:-1], l_sym),
                stride=(l_sym, l_sym, 1),
            )

            tensor_m_128 = cute.sym_int()
            rest_k = cute.sym_int()
            stride_rest_k = cute.sym_int(divisibility=32 * 4 * 4)
            stride_tensor_m_128 = cute.sym_int(divisibility=32 * 4 * 4)
            sfa_shape = list(self.sfa_desc.shape)
            sfa_shape[2] = tensor_m_128
            sfa_shape[4] = rest_k
            sfa_shape[5] = l_sym
            sfa_stride = list(self.sfa_desc.stride)
            sfa_stride[2] = stride_rest_k
            sfa_stride[5] = stride_tensor_m_128
            sfa_cute_fake = self._make_fake_cute_tensor(
                dtype=self.sfa_desc.dtype,
                shape=tuple(sfa_shape),
                stride=tuple(sfa_stride),
            )

            tensor_n_128 = cute.sym_int()
            stride_sfb_rest_k = cute.sym_int(divisibility=32 * 4 * 4)
            stride_sfb_tensor_n_128 = cute.sym_int(divisibility=32 * 4 * 4)
            sfb_cute_fake = self._make_fake_cute_tensor(
                dtype=self.sfb_desc.dtype,
                shape=(32, 4, tensor_n_128, 4, rest_k, l_sym),
                stride=(16, 4, stride_sfb_tensor_n_128, 1, 512, stride_sfb_rest_k),
            )

            sfd_cute_fake = None
            if self.sfd_desc is not None:
                rest_n = cute.sym_int()
                stride_sfd_rest_n = cute.sym_int(divisibility=32 * 4 * 4)
                stride_sfd_tensor_m_128 = cute.sym_int(divisibility=32 * 4 * 4)
                sfd_shape = list(self.sfd_desc.shape)
                sfd_shape[2] = tensor_m_128
                sfd_shape[4] = rest_n
                sfd_shape[5] = l_sym
                sfd_stride = list(self.sfd_desc.stride)
                sfd_stride[2] = stride_sfd_rest_n
                sfd_stride[5] = stride_sfd_tensor_m_128
                sfd_cute_fake = self._make_fake_cute_tensor(
                    dtype=self.sfd_desc.dtype,
                    shape=tuple(sfd_shape),
                    stride=tuple(sfd_stride),
                )
        else:
            a_cute_fake = self._make_fake_cute_tensor_from_desc(self.a_desc, assumed_align=16)
            b_cute_fake = self._make_fake_cute_tensor_from_desc(self.b_desc, assumed_align=16)
            c_cute_fake = self._make_fake_cute_tensor_from_desc(self.c_desc, assumed_align=16)
            d_cute_fake = self._make_fake_cute_tensor_from_desc(self.d_desc, assumed_align=16)
            prob_cute_fake = self._make_fake_cute_tensor_from_desc(self.prob_desc, assumed_align=16)
            dprob_cute_fake = self._make_fake_cute_tensor_from_desc(self.dprob_desc, assumed_align=16)
            sfa_cute_fake = self._make_fake_cute_tensor_from_desc(self.sfa_desc, assumed_align=16)
            sfb_cute_fake = self._make_fake_cute_tensor_from_desc(self.sfb_desc, assumed_align=16)
            sfd_cute_fake = self._make_fake_cute_tensor_from_desc(self.sfd_desc, assumed_align=16)

        compiled = cute.compile(
            gemm,
            a_tensor=a_cute_fake,
            b_tensor=b_cute_fake,
            sfa_tensor=sfa_cute_fake,
            sfb_tensor=sfb_cute_fake,
            c_tensor=c_cute_fake,
            d_tensor=d_cute_fake,
            prob_tensor=prob_cute_fake,
            dprob_tensor=dprob_cute_fake,
            amax_tensor=self._make_fake_cute_tensor_from_desc(self.amax_desc, assumed_align=16),
            sfd_tensor=sfd_cute_fake,
            norm_const_tensor=self._make_fake_cute_tensor_from_desc(self.norm_const_desc, assumed_align=16),
            alpha=self.alpha,
            max_active_clusters=max_active_clusters,
            stream=fake_stream,
            epilogue_op=epilogue_op,
            options="--enable-tvm-ffi",
        )

        def tensor_api(
            a_tensor: torch.Tensor,
            b_tensor: torch.Tensor,
            sfa_tensor: torch.Tensor,
            sfb_tensor: torch.Tensor,
            c_tensor: torch.Tensor,
            d_tensor: torch.Tensor,
            prob_tensor: torch.Tensor,
            dprob_tensor: torch.Tensor,
            amax_tensor: Optional[torch.Tensor],
            sfd_tensor: Optional[torch.Tensor],
            norm_const_tensor: Optional[torch.Tensor],
            alpha: float,
            stream: cuda.CUstream,
        ) -> None:
            compiled(
                a_tensor,
                b_tensor,
                sfa_tensor,
                sfb_tensor,
                c_tensor,
                d_tensor,
                prob_tensor,
                dprob_tensor,
                self._unpad_tensor_to_ndim(amax_tensor, 1, "amax"),
                sfd_tensor,
                self._unpad_tensor_to_ndim(norm_const_tensor, 1, "norm_const"),
                alpha,
                stream,
            )

        self._compiled_kernel = tensor_api

    def execute(
        self,
        a_tensor: torch.Tensor,
        b_tensor: torch.Tensor,
        c_tensor: torch.Tensor,
        d_tensor: torch.Tensor,
        dprob_tensor: torch.Tensor,
        sfa_tensor: torch.Tensor,
        sfb_tensor: torch.Tensor,
        prob_tensor: torch.Tensor,
        sfd_tensor: Optional[torch.Tensor] = None,
        amax_tensor: Optional[torch.Tensor] = None,
        norm_const_tensor: Optional[torch.Tensor] = None,
        alpha: Optional[float] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        self._runtime_error_if(self._compiled_kernel is None, "GemmDsreluSm100 kernel not compiled; call compile() first")
        current_stream = self._get_default_stream(current_stream)
        self._compiled_kernel(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            sfa_tensor=sfa_tensor,
            sfb_tensor=sfb_tensor,
            c_tensor=c_tensor,
            d_tensor=d_tensor,
            prob_tensor=prob_tensor,
            dprob_tensor=dprob_tensor,
            amax_tensor=amax_tensor,
            sfd_tensor=sfd_tensor,
            norm_const_tensor=norm_const_tensor,
            alpha=self.alpha if alpha is None else alpha,
            stream=current_stream,
        )


_logger = logging.getLogger(__name__)
_cache_of_GemmDsreluSm100Objects = {}
_DENSE_GEMM_DYNAMIC_M_ENV = "CUDNN_FE_GEMM_DYNAMIC_M"
_DENSE_GEMM_DYNAMIC_MNKL_ENV = "CUDNN_FE_GEMM_DYNAMIC_MNKL"


def _allocate_dense_output(shape: Tuple[int, int, int], major: str, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    m, n, l = shape
    if major == "m":
        return torch.empty_strided((m, n, l), (1, m, m * n), dtype=dtype, device=device)
    if major == "n":
        return torch.empty_strided((m, n, l), (n, 1, m * n), dtype=dtype, device=device)
    raise ValueError(f"major must be 'm' or 'n', got {major}")


def gemm_dsrelu_wrapper_sm100(
    a_tensor: torch.Tensor,
    b_tensor: torch.Tensor,
    c_tensor: torch.Tensor,
    sfa_tensor: torch.Tensor,
    sfb_tensor: torch.Tensor,
    prob_tensor: torch.Tensor,
    alpha: float = 1.0,
    d_major: str = "n",
    d_dtype: torch.dtype = torch.bfloat16,
    acc_dtype: torch.dtype = torch.float32,
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    norm_const_tensor: Optional[torch.Tensor] = None,
    sf_vec_size: int = 16,
    vector_f32: bool = False,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    m, k, l = a_tensor.shape
    n, _, _ = b_tensor.shape

    d_tensor = _allocate_dense_output((m, n, l), d_major, d_dtype, a_tensor.device)
    dprob_tensor = torch.zeros((m, 1, l), dtype=torch.float32, device=a_tensor.device)

    sfd_tensor = None
    if d_dtype in {torch.float8_e4m3fn, torch.float8_e5m2}:
        sf_k = ceil_div(n, sf_vec_size)
        mma_shape = (
            l,
            ceil_div(m, 128),
            ceil_div(sf_k, 4),
            32,
            4,
            4,
        )
        sfd_tensor = torch.empty(mma_shape, dtype=sfa_tensor.dtype, device=a_tensor.device).permute(3, 4, 1, 5, 2, 0)

    amax_tensor = None
    if a_tensor.dtype in {torch.float4_e2m1fn_x2, torch.uint8} and d_dtype in {torch.bfloat16, torch.float16, torch.float32}:
        amax_tensor = torch.full((1,), float("-inf"), dtype=torch.float32, device=a_tensor.device)

    use_full_dynamic = os.environ.get(_DENSE_GEMM_DYNAMIC_MNKL_ENV) is not None
    use_dynamic_m = not use_full_dynamic and os.environ.get(_DENSE_GEMM_DYNAMIC_M_ENV) is not None

    def stride_order(tensor: torch.Tensor) -> Tuple[int, ...]:
        return tuple(i for i, s in sorted(enumerate(tensor.stride()), key=lambda x: x[1]))

    def tensor_signature(tensor: Optional[torch.Tensor]) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        return tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype

    def dynamic_compact_signature(tensor: Optional[torch.Tensor]) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        return tuple(tensor.shape[1:]), stride_order(tensor), tensor.dtype

    def dynamic_tensor_signature(tensor: Optional[torch.Tensor]) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        return None, stride_order(tensor), tensor.dtype

    def dynamic_m_tensor_signature(
        tensor: Optional[torch.Tensor], static_shape_suffix: Optional[Tuple[int, ...]], dynamic_stride_dims: Tuple[int, ...] = ()
    ) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        stride_signature = tuple(None if i in dynamic_stride_dims else s for i, s in enumerate(tensor.stride()))
        return static_shape_suffix, stride_signature, tensor.dtype

    cache_key = (
        use_full_dynamic,
        use_dynamic_m,
        *(dynamic_tensor_signature(a_tensor) if use_full_dynamic else dynamic_compact_signature(a_tensor) if use_dynamic_m else tensor_signature(a_tensor)),
        *(dynamic_tensor_signature(b_tensor) if use_full_dynamic else tensor_signature(b_tensor)),
        *(dynamic_tensor_signature(c_tensor) if use_full_dynamic else dynamic_compact_signature(c_tensor) if use_dynamic_m else tensor_signature(c_tensor)),
        *(dynamic_tensor_signature(d_tensor) if use_full_dynamic else dynamic_compact_signature(d_tensor) if use_dynamic_m else tensor_signature(d_tensor)),
        *(
            dynamic_tensor_signature(dprob_tensor)
            if use_full_dynamic
            else dynamic_compact_signature(dprob_tensor) if use_dynamic_m else tensor_signature(dprob_tensor)
        ),
        d_dtype,
        *(
            dynamic_tensor_signature(sfa_tensor)
            if use_full_dynamic
            else (
                dynamic_m_tensor_signature(sfa_tensor, (sfa_tensor.shape[4], sfa_tensor.shape[5]), dynamic_stride_dims=(5,))
                if use_dynamic_m
                else tensor_signature(sfa_tensor)
            )
        ),
        *(dynamic_tensor_signature(sfb_tensor) if use_full_dynamic else tensor_signature(sfb_tensor)),
        *(
            dynamic_tensor_signature(prob_tensor)
            if use_full_dynamic
            else dynamic_compact_signature(prob_tensor) if use_dynamic_m else tensor_signature(prob_tensor)
        ),
        norm_const_tensor.shape if norm_const_tensor is not None else None,
        norm_const_tensor.stride() if norm_const_tensor is not None else None,
        norm_const_tensor.dtype if norm_const_tensor is not None else None,
        alpha,
        acc_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        d_major,
        sf_vec_size,
        vector_f32,
    )

    op = _cache_of_GemmDsreluSm100Objects.get(cache_key)
    if op is None:
        op = GemmDsreluSm100(
            sample_a=a_tensor,
            sample_b=b_tensor,
            sample_c=c_tensor,
            sample_d=d_tensor,
            sample_dprob=dprob_tensor,
            sample_sfa=sfa_tensor,
            sample_sfb=sfb_tensor,
            sample_prob=prob_tensor,
            sample_sfd=sfd_tensor,
            sample_amax=amax_tensor,
            sample_norm_const=norm_const_tensor,
            alpha=alpha,
            acc_dtype=acc_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sf_vec_size=sf_vec_size,
            vector_f32=vector_f32,
        )
        assert op.check_support(), "Unsupported testcase"
        op.compile()
        _cache_of_GemmDsreluSm100Objects[cache_key] = op

    op.execute(
        a_tensor=a_tensor,
        b_tensor=b_tensor,
        c_tensor=c_tensor,
        d_tensor=d_tensor,
        dprob_tensor=dprob_tensor,
        sfa_tensor=sfa_tensor,
        sfb_tensor=sfb_tensor,
        prob_tensor=prob_tensor,
        sfd_tensor=sfd_tensor,
        amax_tensor=amax_tensor,
        norm_const_tensor=norm_const_tensor,
        alpha=alpha,
        current_stream=stream,
    )

    return TupleDict(
        d_tensor=d_tensor,
        dprob_tensor=dprob_tensor,
        amax_tensor=amax_tensor,
        sfd_tensor=sfd_tensor,
    )
