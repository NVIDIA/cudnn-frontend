# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

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

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.api_base import APIBase, TupleDict, ceil_div, is_power_of_2
import os


class GemmSwigluSm100(APIBase):
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

        self._logger.warning("GemmSwigluSm100 is an experimental API")
        self._logger.debug("Entering __init__")

        self.a_desc = self._make_tensor_desc(sample_a, name="sample_a")
        self.b_desc = self._make_tensor_desc(sample_b, name="sample_b")
        self.ab12_desc = self._make_tensor_desc(sample_ab12, name="sample_ab12")
        self.c_desc = self._make_tensor_desc(sample_c, name="sample_c")
        self.alpha = alpha
        self.acc_dtype = acc_dtype
        self.mma_tiler_mn = mma_tiler_mn
        if cluster_shape_mn is None:
            self.cluster_shape_mn = (1, 1) if not self.mma_tiler_mn[0] == 256 else (2, 2)
        else:
            self.cluster_shape_mn = cluster_shape_mn

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
        print(f"setting num_cluster_overlap_margin: {self.num_cluster_overlap_margin}")

        # Kernel selection
        if self.sfa_desc is None and self.sfb_desc is None and self.amax_desc is None and self.sfc_desc is None and self.norm_const_desc is None:
            self._logger.debug("No quantization arguments provided, using regular GEMM swiglu kernel")
            self._kernel = PersistentDenseGemmKernel
        else:
            self._logger.debug("Quantization arguments provided, using quantized GEMM swiglu kernel")
            self._kernel = Sm100BlockScaledPersistentDenseGemmKernel

        self._logger.debug(
            f"__init__ completed with args: sample_a {self.a_desc.shape}, sample_b {self.b_desc.shape}, sample_ab12 {self.ab12_desc.shape}, sample_c {self.c_desc.shape}, alpha {alpha}, acc_dtype {acc_dtype}, mma_tiler_mn {mma_tiler_mn}, cluster_shape_mn {cluster_shape_mn}, sample_sfa {self.sfa_desc.shape if self.sfa_desc is not None else None}, sample_sfb {self.sfb_desc.shape if self.sfb_desc is not None else None}, sample_amax {self.amax_desc.shape if self.amax_desc is not None else None}, sample_sfc {self.sfc_desc.shape if self.sfc_desc is not None else None}, sample_norm_const {self.norm_const_desc.shape if self.norm_const_desc is not None else None}, sf_vec_size {sf_vec_size}, vector_f32 {vector_f32}, ab12_stages {ab12_stages}"
        )

        self._interpret_uint8_as_fp4x2 = True

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        self._logger.debug("Checking tensor shapes, strides, and dtypes")
        m, k, l = self.a_desc.shape
        n, k, l = self.b_desc.shape
        m, n, l = self.ab12_desc.shape
        m, n_2, l = self.c_desc.shape

        self._check_tensor_shape(self.a_desc, (m, k, l), "A")
        self._check_tensor_shape(self.b_desc, (n, k, l), "B")
        self._check_tensor_shape(self.ab12_desc, (m, n, l), "AB12")
        self._check_tensor_shape(self.c_desc, (m, n // 2, l), "C")

        if self._kernel is Sm100BlockScaledPersistentDenseGemmKernel:
            rest_k = ceil_div(ceil_div(k, self.sf_vec_size), 4)
            self._check_tensor_shape(self.sfa_desc, (32, 4, ceil_div(m, 128), 4, rest_k, l), "SFA")
            self._check_tensor_shape(self.sfb_desc, (32, 4, ceil_div(n, 128), 4, rest_k, l), "SFB")
            self._check_tensor_shape(self.amax_desc, (1,), "amax")
            rest_n2 = ceil_div(ceil_div(n // 2, self.sf_vec_size), 4)
            self._check_tensor_shape(self.sfc_desc, (32, 4, ceil_div(m, 128), 4, rest_n2, l), "SFC")
            self._check_tensor_shape(self.norm_const_desc, (1,), "norm_const")

        _ = self._check_tensor_stride(self.a_desc, stride=[(1, m, m * k), (k, 1, m * k)])
        _ = self._check_tensor_stride(self.b_desc, stride=[(1, n, n * k), (k, 1, n * k)])
        _ = self._check_tensor_stride(self.ab12_desc, stride=[(1, m, m * n), (n, 1, m * n)])
        _ = self._check_tensor_stride(self.c_desc, stride=[(1, m, m * n_2), (n_2, 1, m * n_2)])
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
        elif self._kernel is Sm100BlockScaledPersistentDenseGemmKernel:
            self._value_error_if(
                self.sfa_desc is None or self.sfb_desc is None,
                "sfa and sfb must be provided for quantized GEMM swiglu kernel",
            )

            self.ab_dtype = self._check_dtype(
                self.a_desc,
                dtype=[
                    torch.float4_e2m1fn_x2,
                    torch.uint8,
                    torch.float8_e5m2,
                    torch.float8_e4m3fn,
                ],
                name="A (for quantized GEMM swiglu kernel)",
            )
            self.acc_dtype = self._check_dtype(
                self.acc_dtype,
                dtype=torch.float32,
                name="Accumulator (for quantized GEMM swiglu kernel)",
            )
            self.ab12_dtype = self._check_dtype(
                self.ab12_desc,
                dtype=[
                    torch.float32,
                    torch.float16,
                    torch.bfloat16,
                    torch.float8_e4m3fn,
                    torch.float8_e5m2,
                ],
                name="AB12 (for quantized GEMM swiglu kernel)",
            )
            self.c_dtype = self._check_dtype(
                self.c_desc,
                dtype=[
                    torch.float32,
                    torch.float16,
                    torch.bfloat16,
                    torch.float8_e4m3fn,
                    torch.float8_e5m2,
                ],
                name="C (for quantized GEMM swiglu kernel)",
            )

            self._value_error_if(
                self._is_fp4x2(self.ab_dtype) and self._is_fp8(self.c_dtype),
                "Invalid dtype combination: fp4 ab_dtype is not compatible with fp8 c_dtype (recommended bf16)",
            )

            self._value_error_if(
                self._is_fp8(self.c_dtype) and (self.sfc_desc is None or self.norm_const_desc is None),
                "sfc and norm_const must be provided when c_dtype is fp8",
            )
            self._value_error_if(
                (self._is_fp4x2(self.ab_dtype) and self.c_dtype == torch.bfloat16) and (self.amax_desc is None),
                "amax must be provided when ab_dtype is fp4 and c_dtype is bf16",
            )

            self._not_implemented_error_if(
                self.c_dtype == torch.float32 and self.ab12_dtype == torch.float32,
                "float32 c_dtype and float32 ab12_dtype currently disabled due to kernel bug",
            )

            self._value_error_if(
                self.sf_vec_size not in {16, 32},
                f"sf_vec_size must be 16 or 32 when ab_dtype is {{torch.float8_e5m2, torch.float8_e4m3fn}}, got {self.sf_vec_size}",
            )
            self.sf_dtype = self._check_dtype(
                self.sfa_desc,
                dtype=[torch.float8_e8m0fnu, torch.float8_e4m3fn],
                name="SFA",
            )
            self._check_dtype(
                self.sfb_desc,
                dtype=self.sf_dtype,
                name="SFB",
                extra_error_msg="SFB must have the same dtype as SFA",
            )
            self._check_dtype(
                self.sfc_desc,
                dtype=self.sf_dtype,
                name="SFC",
                extra_error_msg="SFC must have the same dtype as SFA",
            )
            if self._is_fp8(self.ab_dtype):
                self._value_error_if(
                    not (self.sf_dtype == torch.float8_e8m0fnu and self.sf_vec_size == 32),
                    "Invalid ab_dtype and sf_dtype/sf_vec_size combination: fp8 ab_dtype requires float8_e8m0fnu sf_dtype and 32 sf_vec_size",
                )
            elif self._is_fp4x2(self.ab_dtype):
                self._value_error_if(
                    self.sf_dtype == torch.float8_e4m3fn and self.sf_vec_size == 32,
                    "Invalid ab_dtype and sf_dtype/sf_vec_size combination: fp4 ab_dtype not supported with float8_e4m3fn sf_dtype and 32 sf_vec_size",
                )

            if self._is_fp4x2(self.ab_dtype):
                self._value_error_if(
                    self.a_desc.stride_order != (1, 0, 2) or self.b_desc.stride_order != (1, 0, 2),
                    "Invalid A or B tensor stride: fp4 dtype requires k-major layout",
                )
                self._value_error_if(
                    self.ab12_desc.stride_order != (1, 0, 2),
                    "Invalid AB12 tensor stride: fp4 dtype requires n-major layout",
                )
        self._check_dtype(
            self.b_desc,
            dtype=self.ab_dtype,
            name="B",
            extra_error_msg="A and B must have the same dtype",
        )

        self._logger.debug("Checking MMA tile shape and cluster shape")

        self._value_error_if(
            self.mma_tiler_mn[0] not in [128, 256],
            f"Invalid MMA tile shape: expected mma_tiler_mn[0] in {{128, 256}}, got {self.mma_tiler_mn[0]}",
        )
        if self._kernel is PersistentDenseGemmKernel:
            self._value_error_if(
                self.mma_tiler_mn[1] not in range(32, 257, 32),
                f"Invalid MMA tile shape: expected mma_tiler_mn[1] in {{32, 64, ..., 224, 256}}, got {self.mma_tiler_mn[1]}",
            )

        elif self._kernel is Sm100BlockScaledPersistentDenseGemmKernel:
            if self._is_fp4x2(self.ab_dtype):
                self._value_error_if(
                    self.mma_tiler_mn[1] not in range(64, 257, 64),
                    f"Invalid MMA tile shape: expected mma_tiler_mn[1] in {{64, 128, 192, 256}}, got {self.mma_tiler_mn[1]}",
                )
            else:
                if self._is_fp8(self.ab_dtype):
                    self._value_error_if(
                        self._is_fp8(self.c_dtype) or self._is_fp8(self.ab12_dtype) or self.ab12_dtype == torch.float32,
                        "For MXFP8 inputs for blockscaled quantized GEMM swiglu kernel, ab12_dtype and c_dtype cannot be FP8. ab12_dtype also cannot be float32",
                    )

        self._value_error_if(
            self.cluster_shape_mn[0] % (2 if self.mma_tiler_mn[0] == 256 else 1) != 0,
            "Invalid cluster shape: cluster_shape_mn[0] must be divisible by 2 if mma_tiler_mn[0] == 256",
        )
        self._value_error_if(
            not (
                self.cluster_shape_mn[0] * self.cluster_shape_mn[1] <= 16
                and self.cluster_shape_mn[0] > 0
                and self.cluster_shape_mn[1] > 0
                and is_power_of_2(self.cluster_shape_mn[0])
                and is_power_of_2(self.cluster_shape_mn[1])
            ),
            f"Invalid cluster shape: expected values to be powers of 2 and cluster_shape_mn[0] * cluster_shape_mn[1] <= 16, got {self.cluster_shape_mn[0]},{self.cluster_shape_mn[1]}",
        )

        if self._kernel is PersistentDenseGemmKernel:
            use_2cta_instrs = self.mma_tiler_mn[0] == 256
            self._value_error_if(
                not use_2cta_instrs and self.cluster_shape_mn != (1, 1),
                "Invalid cluster shape: cluster_shape must be (1, 1) when use_2cta_instrs=False",
            )
            if self.cluster_shape_mn != (1, 1) and self.mma_tiler_mn[0] == 128:
                self._value_error_if(
                    self.mma_tiler_mn != (128, 128),
                    "Invalid MMA tile shape: for non-1x1 cluster shape and 128xmma tile shape, mma_tiler_mn must be (128, 128)",
                )

        self._logger.debug("Checking tensor alignment")

        def check_contigous_16B_alignment(dtype, stride_order, tensor_shape):
            is_mode0_major = stride_order == (0, 1, 2)
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // (_convert_to_cutlass_data_type(dtype, interpret_uint8_as_fp4x2=self._interpret_uint8_as_fp4x2).width)
            return num_major_elements % num_contiguous_elements == 0

        self._value_error_if(
            not (
                check_contigous_16B_alignment(self.ab_dtype, self.a_desc.stride_order, (m, k, l))
                and check_contigous_16B_alignment(self.ab_dtype, self.b_desc.stride_order, (n, k, l))
                and check_contigous_16B_alignment(self.ab12_dtype, self.ab12_desc.stride_order, (m, n, l))
            ),
            "Invalid tensor alignment: tensors must be 16B aligned",
        )

        if self._kernel is Sm100BlockScaledPersistentDenseGemmKernel:
            self._value_error_if(
                m % self.mma_tiler_mn[0] != 0 or n % self.mma_tiler_mn[1] != 0,
                "Invalid tensor alignment: m and n must be aligned to mma_tiler_mn",
            )

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
                use_2cta_instrs=(self.mma_tiler_mn[0] == 256),
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
        max_active_clusters = hardware_info.get_max_active_clusters(self.cluster_shape_mn[0] * self.cluster_shape_mn[1])
        max_active_clusters -= self.num_cluster_overlap_margin
        self._value_error_if(
            max_active_clusters <= 0,
            "max_active_clusters must be > 0 after applying overlap margin; reduce CUDNNFE_CLUSTER_OVERLAP_MARGIN",
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
