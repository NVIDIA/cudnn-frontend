# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from .dense_gemm_persistent_swiglu import (
    PersistentDenseGemmKernel,
)
from .dense_blockscaled_gemm_persistent_swiglu_interleaved_quant import (
    Sm100BlockScaledPersistentDenseGemmKernel,
)
from cuda.bindings import driver as cuda
import torch
from typing import Tuple, Optional

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_stream

from cudnn.api_base import ApiBaseTorch, TupleDict, ceil_div
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.gemm_validation import (
    require_contiguous_alignment,
    require_gemm_shapes,
    resolve_max_active_clusters,
)
from .validation import validate_quantized_gemm_swiglu
import os


class GemmSwigluSm100(ApiBaseTorch):
    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_b: torch.Tensor,
        sample_ab12: torch.Tensor,
        sample_c: torch.Tensor,
        alpha: float = 1.0,
        acc_dtype: torch.dtype = torch.float32,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        cluster_shape_mn: Optional[Tuple[int, int]] = None,
        ### Quantize only arguments
        sample_sfa: Optional[torch.Tensor] = None,
        sample_sfb: Optional[torch.Tensor] = None,
        sample_amax: Optional[torch.Tensor] = None,
        sample_sfc: Optional[torch.Tensor] = None,
        sample_norm_const: Optional[torch.Tensor] = None,
        sf_vec_size: int = 16,
        vector_f32: bool = False,
        ab12_stages: int = 4,
    ):
        super().__init__()
        self._interpret_uint8_as_fp4x2 = True

        self._warn_experimental_api()
        self._logger.debug("Entering __init__")

        self.a_desc = self._make_tensor_desc(sample_a, name="sample_a")
        self.b_desc = self._make_tensor_desc(sample_b, name="sample_b")
        self.ab12_desc = self._make_tensor_desc(sample_ab12, name="sample_ab12")
        self.c_desc = self._make_tensor_desc(sample_c, name="sample_c")
        self.alpha = alpha
        self.acc_dtype = acc_dtype
        self.mma_tiler_mn = mma_tiler_mn

        ### Quantize only arguments
        self.sfa_desc = self._make_tensor_desc(sample_sfa, name="sample_sfa")
        self.sfb_desc = self._make_tensor_desc(sample_sfb, name="sample_sfb")
        self.sfc_desc = self._make_tensor_desc(sample_sfc, name="sample_sfc")
        self.amax_desc = self._unpad_tensor_to_ndim(self._make_tensor_desc(sample_amax, name="sample_amax"), 1, "amax")
        self.norm_const_desc = self._unpad_tensor_to_ndim(self._make_tensor_desc(sample_norm_const, name="sample_norm_const"), 1, "norm_const")
        self.sf_vec_size = sf_vec_size
        self.vector_f32 = vector_f32
        self.ab12_stages = ab12_stages
        self.num_cluster_overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
        self._logger.debug(f"setting num_cluster_overlap_margin: {self.num_cluster_overlap_margin}")

        # Kernel selection
        if self.sfa_desc is None and self.sfb_desc is None and self.amax_desc is None and self.sfc_desc is None and self.norm_const_desc is None:
            self._logger.debug("No quantization arguments provided, using regular GEMM swiglu kernel")
            self._kernel = PersistentDenseGemmKernel
        else:
            self._logger.debug("Quantization arguments provided, using quantized GEMM swiglu kernel")
            self._kernel = Sm100BlockScaledPersistentDenseGemmKernel

        if cluster_shape_mn is None:
            self.cluster_shape_mn = (2, 2) if self.mma_tiler_mn[0] == self._kernel.TWO_CTA_MMA_TILER_M else (1, 1)
        else:
            self.cluster_shape_mn = cluster_shape_mn

        self._logger.debug(
            f"__init__ completed with args: sample_a {self.a_desc.shape}, sample_b {self.b_desc.shape}, sample_ab12 {self.ab12_desc.shape}, sample_c {self.c_desc.shape}, alpha {alpha}, acc_dtype {acc_dtype}, mma_tiler_mn {mma_tiler_mn}, cluster_shape_mn {cluster_shape_mn}, sample_sfa {self.sfa_desc.shape if self.sfa_desc is not None else None}, sample_sfb {self.sfb_desc.shape if self.sfb_desc is not None else None}, sample_amax {self.amax_desc.shape if self.amax_desc is not None else None}, sample_sfc {self.sfc_desc.shape if self.sfc_desc is not None else None}, sample_norm_const {self.norm_const_desc.shape if self.norm_const_desc is not None else None}, sf_vec_size {sf_vec_size}, vector_f32 {vector_f32}, ab12_stages {ab12_stages}"
        )

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        self._logger.debug("Checking tensor shapes, strides, and dtypes")
        m, n, k, l = require_gemm_shapes(
            self._tensor_shape(self.a_desc, name="sample_a"),
            self._tensor_shape(self.b_desc, name="sample_b"),
        )
        n_2 = self._kernel.get_output_n(n)
        if self._kernel is PersistentDenseGemmKernel:
            self._check_tensor_shape(self.ab12_desc, (m, n, l), "AB12")
            self._check_tensor_shape(self.c_desc, (m, n_2, l), "C")
            self._check_tensor_stride(self.a_desc, stride=[(1, m, m * k), (k, 1, m * k)])
            self._check_tensor_stride(self.b_desc, stride=[(1, n, n * k), (k, 1, n * k)])
            self._check_tensor_stride(self.ab12_desc, stride=[(1, m, m * n), (n, 1, m * n)])
            self._check_tensor_stride(self.c_desc, stride=[(1, m, m * n_2), (n_2, 1, m * n_2)])
            self._value_error_if(
                self.ab12_desc.stride_order != self.c_desc.stride_order,
                f"AB12 and C tensor stride orders must match, got {self.ab12_desc.stride_order} and {self.c_desc.stride_order}",
            )

        self._logger.debug("Checking data types")
        if self._kernel is PersistentDenseGemmKernel:
            self.ab_dtype = self._check_dtype(
                self.a_desc,
                dtype=[
                    torch.float16,
                    torch.bfloat16,
                    torch.float32,
                    torch.float8_e4m3fn,
                    torch.float8_e5m2,
                ],
                name="A",
            )
            match self.acc_dtype:
                case torch.float32:
                    self.ab12_dtype = self._check_dtype(
                        self.ab12_desc,
                        dtype=[
                            torch.float32,
                            torch.float16,
                            torch.bfloat16,
                            torch.float8_e4m3fn,
                            torch.float8_e5m2,
                        ],
                        name="AB12 (for float32 acc_dtype)",
                    )
                    self._not_implemented_error_if(
                        self._is_fp8(self.ab12_dtype),
                        f"ab12_dtype {{torch.float8_e5m2, torch.float8_e4m3fn}} is currently disabled",
                    )
                case torch.float16:
                    self.ab12_dtype = self._check_dtype(
                        self.ab12_desc,
                        dtype=[torch.float16, torch.bfloat16],
                        name="AB12 (for float16 acc_dtype)",
                    )
                    self._check_dtype(
                        self.a_desc,
                        dtype=[torch.float16, torch.float8_e4m3fn, torch.float8_e5m2],
                        name="A/B (for float16 acc_dtype)",
                    )
                case _:
                    raise ValueError(f"Unsupported acc_dtype: expected one of {{torch.float32, torch.float16}}, got {self.acc_dtype}")
            self.c_dtype = self._check_dtype(self.c_desc, dtype=[torch.float16, torch.bfloat16], name="C")
        elif self._kernel is not Sm100BlockScaledPersistentDenseGemmKernel:
            raise NotImplementedError(f"Unreachable: invalid kernel type {self._kernel}")

        if self._kernel is PersistentDenseGemmKernel:
            self._check_dtype(
                self.b_desc,
                dtype=self.ab_dtype,
                name="B",
                extra_error_msg="A and B must have the same dtype",
            )

        self._logger.debug("Checking MMA tile shape and cluster shape")

        self.mma_tiler_mn = self._kernel.require_mma_tiler(self.mma_tiler_mn)
        self.cluster_shape_mn = self._kernel.require_cluster_shape(
            self.cluster_shape_mn,
            mma_tiler_mn=self.mma_tiler_mn,
        )

        if self._kernel is Sm100BlockScaledPersistentDenseGemmKernel:
            plan = validate_quantized_gemm_swiglu(
                self.a_desc,
                self.b_desc,
                self.ab12_desc,
                self.c_desc,
                sfa=self.sfa_desc,
                sfb=self.sfb_desc,
                amax=self.amax_desc,
                sfc=self.sfc_desc,
                norm_const=self.norm_const_desc,
                acc_dtype=self.acc_dtype,
                output_n=n_2,
                sf_vec_size=self.sf_vec_size,
                supported_sf_vec_sizes=self._kernel.SF_VEC_SIZES,
                mma_tiler_mn=self.mma_tiler_mn,
            )
            self.ab_dtype = self.a_desc.dtype
            self.ab12_dtype = self.ab12_desc.dtype
            self.c_dtype = self.c_desc.dtype
            self.sf_dtype = self.sfa_desc.dtype
            self._logger.debug("Resolved quantized GEMM + SwiGLU plan: %s", plan)
        else:
            self._logger.debug("Checking tensor alignment")
            ab_bits = _convert_to_cutlass_data_type(
                self.ab_dtype,
                interpret_uint8_as_fp4x2=self._interpret_uint8_as_fp4x2,
            ).width
            ab12_bits = _convert_to_cutlass_data_type(
                self.ab12_dtype,
                interpret_uint8_as_fp4x2=self._interpret_uint8_as_fp4x2,
            ).width
            c_bits = _convert_to_cutlass_data_type(
                self.c_dtype,
                interpret_uint8_as_fp4x2=self._interpret_uint8_as_fp4x2,
            ).width
            require_contiguous_alignment("A", m if self.a_desc.stride_order == (0, 1, 2) else k, ab_bits)
            require_contiguous_alignment("B", n if self.b_desc.stride_order == (0, 1, 2) else k, ab_bits)
            require_contiguous_alignment("AB12", m if self.ab12_desc.stride_order == (0, 1, 2) else n, ab12_bits)
            require_contiguous_alignment("C", m if self.c_desc.stride_order == (0, 1, 2) else n_2, c_bits)

        self._logger.debug("Checking environment")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        compute_capability = major * 10 + minor
        if compute_capability < 100:
            raise RuntimeError(f"GemmSwiglu requires SM100+ compute capability, but found SM{compute_capability} on device {device}")

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def compile(self) -> None:
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            self._logger.debug("Kernel already compiled; skipping recompilation")
            return

        if self._kernel is PersistentDenseGemmKernel:
            gemm_swiglu = self._kernel(
                acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
                use_2cta_instrs=(self.mma_tiler_mn[0] == self._kernel.TWO_CTA_MMA_TILER_M),
                mma_tiler_mn=self.mma_tiler_mn,
                cluster_shape_mn=self.cluster_shape_mn,
            )
        elif self._kernel is Sm100BlockScaledPersistentDenseGemmKernel:
            gemm_swiglu = self._kernel(
                sf_vec_size=self.sf_vec_size,
                mma_tiler_mn=self.mma_tiler_mn,
                cluster_shape_mn=self.cluster_shape_mn,
                vector_f32=self.vector_f32,
                ab12_stages=self.ab12_stages,
            )
        else:
            raise NotImplementedError(f"Unreachable: invalid kernel type {self._kernel}")

        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = resolve_max_active_clusters(
            hardware_info.get_max_active_clusters(self.cluster_shape_mn[0] * self.cluster_shape_mn[1]),
            self.num_cluster_overlap_margin,
        )

        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)

        if self._kernel is PersistentDenseGemmKernel:
            self._logger.debug("Compiling gemm_swiglu")
            _compiled_kernel = cute.compile(
                gemm_swiglu,
                a=self._make_fake_cute_tensor_from_desc(self.a_desc),
                b=self._make_fake_cute_tensor_from_desc(self.b_desc),
                ab12=self._make_fake_cute_tensor_from_desc(self.ab12_desc),
                c=self._make_fake_cute_tensor_from_desc(self.c_desc),
                alpha=self.alpha,
                max_active_clusters=max_active_clusters,
                stream=fake_stream,
                options="--enable-tvm-ffi",
            )

            def tensor_api(
                a_tensor: torch.Tensor,
                b_tensor: torch.Tensor,
                ab12_tensor: torch.Tensor,
                c_tensor: torch.Tensor,
                alpha: float,
                stream: cuda.CUstream,
            ) -> None:
                _compiled_kernel(
                    a_tensor,
                    b_tensor,
                    ab12_tensor,
                    c_tensor,
                    alpha,
                    stream,
                )

            self._compiled_kernel = tensor_api
        elif self._kernel is Sm100BlockScaledPersistentDenseGemmKernel:
            self._logger.debug("Compiling gemm_swiglu_blockscaled_quantized")
            _compiled_kernel = cute.compile(
                gemm_swiglu,
                a_tensor=self._make_fake_cute_tensor_from_desc(self.a_desc, assumed_align=16),
                b_tensor=self._make_fake_cute_tensor_from_desc(self.b_desc, assumed_align=16),
                sfa_tensor=self._make_fake_cute_tensor_from_desc(self.sfa_desc, assumed_align=16),
                sfb_tensor=self._make_fake_cute_tensor_from_desc(self.sfb_desc, assumed_align=16),
                c_tensor=self._make_fake_cute_tensor_from_desc(self.c_desc, assumed_align=16),
                ab12_tensor=self._make_fake_cute_tensor_from_desc(self.ab12_desc, assumed_align=8),
                amax_tensor=self._make_fake_cute_tensor_from_desc(self.amax_desc, assumed_align=16),
                sfc_tensor=self._make_fake_cute_tensor_from_desc(self.sfc_desc, assumed_align=16),
                norm_const_tensor=self._make_fake_cute_tensor_from_desc(self.norm_const_desc, assumed_align=16),
                alpha=self.alpha,
                max_active_clusters=max_active_clusters,
                stream=fake_stream,
                options="--enable-tvm-ffi",
            )

            def tensor_api(
                a_tensor: torch.Tensor,
                b_tensor: torch.Tensor,
                ab12_tensor: torch.Tensor,
                c_tensor: torch.Tensor,
                sfa_tensor: Optional[torch.Tensor],
                sfb_tensor: Optional[torch.Tensor],
                amax_tensor: Optional[torch.Tensor],
                sfc_tensor: Optional[torch.Tensor],
                norm_const_tensor: Optional[torch.Tensor],
                alpha: float,
                stream: cuda.CUstream,
            ) -> None:
                amax_tensor = self._unpad_tensor_to_ndim(amax_tensor, 1, "amax")
                norm_const_tensor = self._unpad_tensor_to_ndim(norm_const_tensor, 1, "norm_const")
                _compiled_kernel(
                    a_tensor,
                    b_tensor,
                    sfa_tensor,
                    sfb_tensor,
                    c_tensor,
                    ab12_tensor,
                    amax_tensor,
                    sfc_tensor,
                    norm_const_tensor,
                    alpha,
                    stream,
                )

            self._compiled_kernel = tensor_api

        self._logger.debug("Kernel compiled successfully")

    def execute(
        self,
        a_tensor: torch.Tensor,
        b_tensor: torch.Tensor,
        ab12_tensor: torch.Tensor,
        c_tensor: torch.Tensor,
        sfa_tensor: Optional[torch.Tensor] = None,
        sfb_tensor: Optional[torch.Tensor] = None,
        amax_tensor: Optional[torch.Tensor] = None,
        sfc_tensor: Optional[torch.Tensor] = None,
        norm_const_tensor: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        self._logger.debug("Entering execute")
        current_stream = self._get_default_stream(current_stream)

        self._runtime_error_if(
            self._compiled_kernel is None,
            "GemmSwigluSm100 kernel not compiled; call compile() first",
        )
        self._logger.debug("Executing with compiled kernel")

        if self._kernel is PersistentDenseGemmKernel:
            self._compiled_kernel(
                a_tensor=a_tensor,
                b_tensor=b_tensor,
                ab12_tensor=ab12_tensor,
                c_tensor=c_tensor,
                alpha=alpha,
                stream=current_stream,
            )
        elif self._kernel is Sm100BlockScaledPersistentDenseGemmKernel:
            self._compiled_kernel(
                a_tensor=a_tensor,
                b_tensor=b_tensor,
                ab12_tensor=ab12_tensor,
                c_tensor=c_tensor,
                sfa_tensor=sfa_tensor,
                sfb_tensor=sfb_tensor,
                amax_tensor=amax_tensor,
                sfc_tensor=sfc_tensor,
                norm_const_tensor=norm_const_tensor,
                alpha=alpha,
                stream=current_stream,
            )
        else:
            raise NotImplementedError(f"Unreachable: invalid kernel type {self._kernel}")

        self._logger.debug("Executed with compiled kernel successfully")


import logging

_logger = logging.getLogger(__name__)
_cache_of_GemmSwigluSm100Objects = {}


def gemm_swiglu_wrapper_sm100(
    a_tensor: torch.Tensor,
    b_tensor: torch.Tensor,
    alpha: float = 1.0,
    c_major: str = "n",
    ab12_dtype: torch.dtype = torch.float32,
    c_dtype: torch.dtype = torch.float16,
    acc_dtype: torch.dtype = torch.float32,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    ### Quantize only arguments
    sfa_tensor: Optional[torch.Tensor] = None,
    sfb_tensor: Optional[torch.Tensor] = None,
    norm_const_tensor: Optional[torch.Tensor] = None,
    sf_vec_size: int = 16,
    vector_f32: bool = False,
    ab12_stages: int = 4,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:

    _logger.debug("gemm_swiglu_wrapper_sm100: Creating empty output tensors ab12 and c")
    m, k, l = a_tensor.shape
    n, k, l = b_tensor.shape
    ab12_tensor, c_tensor = None, None
    if c_major == "m":
        ab12_tensor = torch.empty_strided((m, n, l), (1, m, m * n), dtype=ab12_dtype, device=a_tensor.device)
        c_tensor = torch.empty_strided((m, n // 2, l), (1, m, m * n // 2), dtype=c_dtype, device=a_tensor.device)
    elif c_major == "n":
        ab12_tensor = torch.empty_strided((m, n, l), (n, 1, m * n), dtype=ab12_dtype, device=a_tensor.device)
        c_tensor = torch.empty_strided(
            (m, n // 2, l),
            (n // 2, 1, m * n // 2),
            dtype=c_dtype,
            device=a_tensor.device,
        )
    else:
        raise ValueError(f"c_major must be either 'm' or 'n', got {c_major}")

    sfc_tensor, amax_tensor = None, None
    if sfa_tensor is not None and sfb_tensor is not None:
        _logger.debug("gemm_swiglu_wrapper_sm100: Detected sfa_tensor and sfb_tensor, constructing quantized output tensors")
        if c_dtype in {torch.float8_e5m2, torch.float8_e4m3fn}:
            _logger.debug("gemm_swiglu_wrapper_sm100: Detected fp8 c_dtype, constructing sfc_tensor")

            sf_k = ceil_div(n // 2, sf_vec_size)
            mma_shape = (
                l,
                ceil_div(m, 128),
                ceil_div(sf_k, 4),
                32,
                4,
                4,
            )
            mma_permute_order = (3, 4, 1, 5, 2, 0)
            sfc_tensor = torch.empty(
                mma_shape,
                dtype=torch.float8_e8m0fnu,
                device=a_tensor.device,
            ).permute(mma_permute_order)
        if a_tensor.dtype in {torch.float4_e2m1fn_x2, torch.uint8} and c_dtype == torch.bfloat16:
            _logger.debug("gemm_swiglu_wrapper_sm100: Detected fp4 ab_dtype and bf16 c_dtype, constructing amax_tensor")
            amax_tensor = torch.full((1, 1, 1), -float("inf"), device=a_tensor.device, dtype=torch.float32)

    cache_key = (
        a_tensor.shape,
        b_tensor.shape,
        a_tensor.dtype,
        b_tensor.dtype,
        a_tensor.stride(),
        b_tensor.stride(),
        alpha,
        c_major,
        ab12_dtype,
        c_dtype,
        acc_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        sfa_tensor.shape if sfa_tensor is not None else None,
        sfb_tensor.shape if sfb_tensor is not None else None,
        sfa_tensor.stride() if sfa_tensor is not None else None,
        sfb_tensor.stride() if sfb_tensor is not None else None,
        sfa_tensor.dtype if sfa_tensor is not None else None,
        sfb_tensor.dtype if sfb_tensor is not None else None,
        norm_const_tensor.shape if norm_const_tensor is not None else None,
        norm_const_tensor.stride() if norm_const_tensor is not None else None,
        norm_const_tensor.dtype if norm_const_tensor is not None else None,
        sf_vec_size,
        vector_f32,
        ab12_stages,
    )
    if cache_key in _cache_of_GemmSwigluSm100Objects:
        _logger.debug("gemm_swiglu_wrapper_sm100: Using previously cached GemmSwigluSm100 object")
        gemm_swiglu = _cache_of_GemmSwigluSm100Objects[cache_key]
        gemm_swiglu.execute(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            ab12_tensor=ab12_tensor,
            c_tensor=c_tensor,
            sfa_tensor=sfa_tensor,
            sfb_tensor=sfb_tensor,
            amax_tensor=amax_tensor,
            sfc_tensor=sfc_tensor,
            norm_const_tensor=norm_const_tensor,
            alpha=alpha,
            current_stream=stream,
        )
    else:
        _logger.debug("gemm_swiglu_wrapper_sm100: No previously cached GemmSwigluSm100 object found, creating new GemmSwigluSm100 object")
        gemm_swiglu = GemmSwigluSm100(
            sample_a=a_tensor,
            sample_b=b_tensor,
            sample_ab12=ab12_tensor,
            sample_c=c_tensor,
            alpha=alpha,
            acc_dtype=acc_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sample_sfa=sfa_tensor,
            sample_sfb=sfb_tensor,
            sample_amax=amax_tensor,
            sample_sfc=sfc_tensor,
            sample_norm_const=norm_const_tensor,
            sf_vec_size=sf_vec_size,
            vector_f32=vector_f32,
            ab12_stages=ab12_stages,
        )
        assert gemm_swiglu.check_support(), "Unsupported testcase"
        gemm_swiglu.compile()
        gemm_swiglu.execute(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            ab12_tensor=ab12_tensor,
            c_tensor=c_tensor,
            sfa_tensor=sfa_tensor,
            sfb_tensor=sfb_tensor,
            amax_tensor=amax_tensor,
            sfc_tensor=sfc_tensor,
            norm_const_tensor=norm_const_tensor,
            alpha=alpha,
            current_stream=stream,
        )
        _cache_of_GemmSwigluSm100Objects[cache_key] = gemm_swiglu

    return TupleDict(
        ab12_tensor=ab12_tensor,
        c_tensor=c_tensor,
        sfc_tensor=sfc_tensor,
        amax_tensor=amax_tensor,
    )
