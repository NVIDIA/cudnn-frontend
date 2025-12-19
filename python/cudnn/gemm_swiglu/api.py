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
    PersistentDenseGemmKernelNoDlpack,
)
from .dense_blockscaled_gemm_persistent_swiglu_interleaved_quant import (
    Sm100BlockScaledPersistentDenseGemmKernel,
    Sm100BlockScaledPersistentDenseGemmKernelNoDlpack,
)
from cuda.bindings import driver as cuda
import torch
from typing import Tuple, Optional

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_ptr
from packaging import version
import cutlass.cute.math as math

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.api_base import APIBase, ceil_div


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

        self.sample_a = sample_a
        self.sample_b = sample_b
        self.sample_ab12 = sample_ab12
        self.sample_c = sample_c
        self.alpha = alpha
        self.acc_dtype = acc_dtype
        self.mma_tiler_mn = mma_tiler_mn
        if cluster_shape_mn is None:
            self.cluster_shape_mn = (
                (1, 1) if not self.mma_tiler_mn[0] == 256 else (2, 2)
            )
        else:
            self.cluster_shape_mn = cluster_shape_mn

        ### Quantize only arguments
        self.sample_sfa = sample_sfa
        self.sample_sfb = sample_sfb
        self.sample_sfc = sample_sfc
        self.sample_amax = self._unpad_tensor_to_ndim(sample_amax, 1, "amax")
        self.sample_norm_const = self._unpad_tensor_to_ndim(
            sample_norm_const, 1, "norm_const"
        )
        self.sf_vec_size = sf_vec_size
        self.vector_f32 = vector_f32
        self.ab12_stages = ab12_stages

        # Kernel selection
        if (
            self.sample_sfa is None
            and self.sample_sfb is None
            and self.sample_amax is None
            and self.sample_sfc is None
            and self.sample_norm_const is None
        ):
            self._logger.debug(
                "No quantization arguments provided, using regular GEMM swiglu kernel"
            )
            self._kernel = PersistentDenseGemmKernel
        else:
            self._logger.debug(
                "Quantization arguments provided, using quantized GEMM swiglu kernel"
            )
            self._kernel = Sm100BlockScaledPersistentDenseGemmKernel

        self._logger.debug(
            f"__init__ completed with args: sample_a {sample_a.shape}, sample_b {sample_b.shape}, sample_ab12 {sample_ab12.shape}, sample_c {sample_c.shape}, alpha {alpha}, acc_dtype {acc_dtype}, mma_tiler_mn {mma_tiler_mn}, cluster_shape_mn {cluster_shape_mn}, sample_sfa {sample_sfa.shape if sample_sfa is not None else None}, sample_sfb {sample_sfb.shape if sample_sfb is not None else None}, sample_amax {sample_amax.shape if sample_amax is not None else None}, sample_sfc {sample_sfc.shape if sample_sfc is not None else None}, sample_norm_const {sample_norm_const.shape if sample_norm_const is not None else None}, sf_vec_size {sf_vec_size}, vector_f32 {vector_f32}, ab12_stages {ab12_stages}"
        )

        self._interpret_uint8_as_fp4x2 = True

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        self._logger.debug("Checking tensor shapes, strides, and dtypes")
        m, k, l = self._tensor_shape(self.sample_a, name="sample_a")
        n, k, l = self._tensor_shape(self.sample_b, name="sample_b")
        m, n, l = self._tensor_shape(self.sample_ab12, name="sample_ab12")
        m, n_2, l = self._tensor_shape(self.sample_c, name="sample_c")

        self._check_tensor_shape(self.sample_a, (m, k, l), "A")
        self._check_tensor_shape(self.sample_b, (n, k, l), "B")
        self._check_tensor_shape(self.sample_ab12, (m, n, l), "AB12")
        self._check_tensor_shape(self.sample_c, (m, n // 2, l), "C")

        if self._kernel in {
            Sm100BlockScaledPersistentDenseGemmKernel,
            Sm100BlockScaledPersistentDenseGemmKernelNoDlpack,
        }:
            rest_k = ceil_div(ceil_div(k, self.sf_vec_size), 4)
            self._check_tensor_shape(
                self.sample_sfa, (32, 4, ceil_div(m, 128), 4, rest_k, l), "SFA"
            )
            self._check_tensor_shape(
                self.sample_sfb, (32, 4, ceil_div(n, 128), 4, rest_k, l), "SFB"
            )
            self._check_tensor_shape(self.sample_amax, (1,), "amax")
            rest_n2 = ceil_div(ceil_div(n // 2, self.sf_vec_size), 4)
            self._check_tensor_shape(
                self.sample_sfc, (32, 4, ceil_div(m, 128), 4, rest_n2, l), "SFC"
            )
            self._check_tensor_shape(self.sample_norm_const, (1,), "norm_const")

        _, self.a_stride_order = self._check_tensor_stride(
            self.sample_a, stride=[(1, m, m * k), (k, 1, m * k)]
        )
        _, self.b_stride_order = self._check_tensor_stride(
            self.sample_b, stride=[(1, n, n * k), (k, 1, n * k)]
        )
        _, self.ab12_stride_order = self._check_tensor_stride(
            self.sample_ab12, stride=[(1, m, m * n), (n, 1, m * n)]
        )
        _, self.c_stride_order = self._check_tensor_stride(
            self.sample_c, stride=[(1, m, m * n_2), (n_2, 1, m * n_2)]
        )
        self._value_error_if(
            self.ab12_stride_order != self.c_stride_order,
            f"AB12 and C tensor stride orders must match, got {self.ab12_stride_order} and {self.c_stride_order}",
        )

        self._logger.debug("Checking data types")
        if self._kernel in {
            PersistentDenseGemmKernel,
            PersistentDenseGemmKernelNoDlpack,
        }:
            self.ab_dtype = self._check_dtype(
                self.sample_a,
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
                        self.sample_ab12,
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
                        self.sample_ab12,
                        dtype=[torch.float16, torch.bfloat16],
                        name="AB12 (for float16 acc_dtype)",
                    )
                    self._check_dtype(
                        self.ab_dtype,
                        dtype=[torch.float16, torch.float8_e4m3fn, torch.float8_e5m2],
                        name="A/B (for float16 acc_dtype)",
                    )
                case _:
                    raise ValueError(
                        f"Unsupported acc_dtype: expected one of {{torch.float32, torch.float16}}, got {self.acc_dtype}"
                    )
            self.c_dtype = self._check_dtype(
                self.sample_c, dtype=[torch.float16, torch.bfloat16], name="C"
            )
        elif self._kernel in {
            Sm100BlockScaledPersistentDenseGemmKernel,
            Sm100BlockScaledPersistentDenseGemmKernelNoDlpack,
        }:
            self._value_error_if(
                self.sample_sfa is None or self.sample_sfb is None,
                "sfa and sfb must be provided for quantized GEMM swiglu kernel",
            )

            self.ab_dtype = self._check_dtype(
                self.sample_a,
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
                self.sample_ab12,
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
                self.sample_c,
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
                self._is_fp8(self.c_dtype)
                and (self.sample_sfc is None or self.sample_norm_const is None),
                "sfc and norm_const must be provided when c_dtype is fp8",
            )
            self._value_error_if(
                (self._is_fp4x2(self.ab_dtype) and self.c_dtype == torch.bfloat16)
                and (self.sample_amax is None),
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
                self.sample_sfa,
                dtype=[torch.float8_e8m0fnu, torch.float8_e4m3fn],
                name="SFA",
            )
            self._check_dtype(
                self.sample_sfb,
                dtype=self.sf_dtype,
                name="SFB",
                extra_error_msg="SFB must have the same dtype as SFA",
            )
            self._check_dtype(
                self.sample_sfc,
                dtype=self.sf_dtype,
                name="SFC",
                extra_error_msg="SFC must have the same dtype as SFA",
            )
            if self._is_fp8(self.ab_dtype):
                self._value_error_if(
                    not (
                        self.sf_dtype == torch.float8_e8m0fnu and self.sf_vec_size == 32
                    ),
                    "Invalid ab_dtype and sf_dtype/sf_vec_size combination: fp8 ab_dtype requires float8_e8m0fnu sf_dtype and 32 sf_vec_size",
                )
            elif self._is_fp4x2(self.ab_dtype):
                self._value_error_if(
                    self.sf_dtype == torch.float8_e4m3fn and self.sf_vec_size == 32,
                    "Invalid ab_dtype and sf_dtype/sf_vec_size combination: fp4 ab_dtype not supported with float8_e4m3fn sf_dtype and 32 sf_vec_size",
                )

            if self._is_fp4x2(self.ab_dtype):
                self._value_error_if(
                    self.a_stride_order != (1, 0, 2)
                    or self.b_stride_order != (1, 0, 2),
                    "Invalid A or B tensor stride: fp4 dtype requires k-major layout",
                )
                self._value_error_if(
                    self.ab12_stride_order != (1, 0, 2),
                    "Invalid AB12 tensor stride: fp4 dtype requires n-major layout",
                )
        self._check_dtype(
            self.sample_b,
            dtype=self.ab_dtype,
            name="B",
            extra_error_msg="A and B must have the same dtype",
        )

        self._logger.debug("Checking MMA tile shape and cluster shape")

        def is_power_of_2(x):
            return x > 0 and (x & (x - 1)) == 0

        self._value_error_if(
            self.mma_tiler_mn[0] not in [128, 256],
            f"Invalid MMA tile shape: expected mma_tiler_mn[0] in {{128, 256}}, got {self.mma_tiler_mn[0]}",
        )
        if self._kernel in {
            PersistentDenseGemmKernel,
            PersistentDenseGemmKernelNoDlpack,
        }:
            self._value_error_if(
                self.mma_tiler_mn[1] not in range(32, 257, 32),
                f"Invalid MMA tile shape: expected mma_tiler_mn[1] in {{32, 64, ..., 224, 256}}, got {self.mma_tiler_mn[1]}",
            )

        elif self._kernel in {
            Sm100BlockScaledPersistentDenseGemmKernel,
            Sm100BlockScaledPersistentDenseGemmKernelNoDlpack,
        }:
            if self._is_fp4x2(self.ab_dtype):
                self._value_error_if(
                    self.mma_tiler_mn[1] not in range(64, 257, 64),
                    f"Invalid MMA tile shape: expected mma_tiler_mn[1] in {{64, 128, 192, 256}}, got {self.mma_tiler_mn[1]}",
                )
            else:
                self._value_error_if(
                    self.mma_tiler_mn[1] not in [256],
                    f"Invalid MMA tile shape: MXFP8 Quantized kernel only supports tile_n=256, got {self.mma_tiler_mn[1]}",
                )

            if (
                self.mma_tiler_mn == (256, 256)
                and self.cluster_shape_mn != (1, 1)
                and self.sf_vec_size == 32
                and self.sf_dtype == torch.float8_e8m0fnu
            ):
                self._value_error_if(
                    not (
                        self.ab12_dtype == torch.bfloat16
                        and self.c_dtype == torch.bfloat16
                    ),
                    "Invalid MMA tile shape/cluster shape/dtype combination: for 256x256mma tile shape, non-1x1 cluster shape, 32 sf_vec_size, float8_e8m0fnu sf_dtype: ab12_dtype must be bfloat16 and c_dtype must be bfloat16",
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

        if self._kernel in {
            PersistentDenseGemmKernel,
            PersistentDenseGemmKernelNoDlpack,
        }:
            use_2cta_instrs = self.mma_tiler_mn[0] == 256
            self._value_error_if(
                not use_2cta_instrs and self.cluster_shape_mn != (1, 1),
                "Invalid cluster shape: cluster_shape must be (1, 1) when use_2cta_instrs=False",
            )
            self._value_error_if(
                not use_2cta_instrs and self.ab12_dtype == torch.float32,
                "Invalid ab12_dtype: use_2cta_instrs=False is incompatbile with float32 accumulator",
            )

            self._value_error_if(
                self.mma_tiler_mn == (128, 128)
                and self.cluster_shape_mn == (1, 1)
                and self.ab12_stride_order != (0, 1, 2),
                "Invalid MMA tile shape and AB12 stride order combination: (128, 128) mma tile shape with 1x1 cluster shape is only supported with ab12 stride_order (0, 1, 2)",
            )
            self._value_error_if(
                self.mma_tiler_mn != (128, 128) and self.ab12_stride_order != (0, 1, 2),
                f"Invalid AB12 tensor stride order: for non-128x128mma tile shape, ab12 stride_order must be (0, 1, 2), got {self.ab12_stride_order}",
            )
            if self.cluster_shape_mn != (1, 1) and self.mma_tiler_mn[0] == 128:
                self._value_error_if(
                    self.mma_tiler_mn != (128, 128),
                    "Invalid MMA tile shape: for non-1x1 cluster shape and 128xmma tile shape, mma_tiler_mn must be (128, 128)",
                )
            self._not_implemented_error_if(
                self.mma_tiler_mn[0] == 256 and self.ab12_dtype == torch.float32,
                "mma_tiler_mn[0] == 256 and ab12_dtype == torch.float32 currently disabled",
            )

        self._logger.debug("Checking tensor alignment")

        def check_contigous_16B_alignment(dtype, stride_order, tensor_shape):
            is_mode0_major = stride_order == (0, 1, 2)
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = (
                16
                * 8
                // (
                    _convert_to_cutlass_data_type(
                        dtype, interpret_uint8_as_fp4x2=self._interpret_uint8_as_fp4x2
                    ).width
                )
            )
            return num_major_elements % num_contiguous_elements == 0

        self._value_error_if(
            not (
                check_contigous_16B_alignment(
                    self.ab_dtype, self.a_stride_order, (m, k, l)
                )
                and check_contigous_16B_alignment(
                    self.ab_dtype, self.b_stride_order, (n, k, l)
                )
                and check_contigous_16B_alignment(
                    self.ab12_dtype, self.ab12_stride_order, (m, n, l)
                )
            ),
            "Invalid tensor alignment: tensors must be 16B aligned",
        )

        if self._kernel in {
            Sm100BlockScaledPersistentDenseGemmKernel,
            Sm100BlockScaledPersistentDenseGemmKernelNoDlpack,
        }:
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
            raise RuntimeError(
                f"GemmSwiglu requires SM100+ compute capability, but found SM{compute_capability} on device {device}"
            )
        if compute_capability == 103:
            raise RuntimeError("cuteDSL GemmSwiglu is not supported on SM103")

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def compile(self, current_stream: Optional[cuda.CUstream] = None) -> None:
        self._logger.debug("Entering compile")
        current_stream = self._get_default_stream(current_stream)
        self._ensure_support_checked()

        torch_version = version.parse(torch.__version__)
        is_ab_fp4 = self._is_fp4x2(self.ab_dtype)
        is_ab12_fp4 = self._is_fp4x2(self.ab12_dtype)
        is_ab_fp8 = self._is_fp8(self.ab_dtype)
        is_ab12_fp8 = self._is_fp8(self.ab12_dtype)
        _fp8_dlpack_supported = version.parse(
            torch_version.base_version
        ) >= version.parse("2.10.0")
        use_no_dlpack_kernel = (
            is_ab_fp4
            or is_ab12_fp4
            or ((is_ab_fp8 or is_ab12_fp8) and not _fp8_dlpack_supported)
        )

        if use_no_dlpack_kernel:
            self._logger.debug(
                "Running no_dlpack kernel wrapper due to fp4 dtype or fp8 dtype on incompatible torch version"
            )
            if self._kernel is PersistentDenseGemmKernel:
                self._kernel = PersistentDenseGemmKernelNoDlpack
            elif self._kernel is Sm100BlockScaledPersistentDenseGemmKernel:
                self._kernel = Sm100BlockScaledPersistentDenseGemmKernelNoDlpack
            else:
                raise NotImplementedError(
                    f"Unreachable: invalid kernel type {self._kernel}"
                )

        gemm_swiglu = None
        if self._kernel in (
            PersistentDenseGemmKernel,
            PersistentDenseGemmKernelNoDlpack,
        ):
            gemm_swiglu = self._kernel(
                acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
                use_2cta_instrs=(self.mma_tiler_mn[0] == 256),
                mma_tiler_mn=self.mma_tiler_mn,
                cluster_shape_mn=self.cluster_shape_mn,
            )
        elif self._kernel in (
            Sm100BlockScaledPersistentDenseGemmKernel,
            Sm100BlockScaledPersistentDenseGemmKernelNoDlpack,
        ):
            gemm_swiglu = self._kernel(
                sf_vec_size=self.sf_vec_size,
                mma_tiler_mn=self.mma_tiler_mn,
                cluster_shape_mn=self.cluster_shape_mn,
                vector_f32=self.vector_f32,
                ab12_stages=self.ab12_stages,
            )
        else:
            raise NotImplementedError(
                f"Unreachable: invalid kernel type {self._kernel}"
            )

        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(
            self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        )

        if self._kernel is PersistentDenseGemmKernel:
            self._logger.debug("Compiling gemm_swiglu (dlpack)")
            self._compiled_kernel = cute.compile(
                gemm_swiglu,
                a=from_dlpack(self.sample_a),
                b=from_dlpack(self.sample_b),
                ab12=from_dlpack(self.sample_ab12),
                c=from_dlpack(self.sample_c),
                alpha=self.alpha,
                max_active_clusters=max_active_clusters,
                stream=current_stream,
            )
        elif self._kernel is Sm100BlockScaledPersistentDenseGemmKernel:
            self._logger.debug("Compiling gemm_swiglu_blockscaled_quantized (dlpack)")
            self._compiled_kernel = cute.compile(
                gemm_swiglu,
                a_tensor=from_dlpack(self.sample_a, assumed_align=16),
                b_tensor=from_dlpack(self.sample_b, assumed_align=16),
                sfa_tensor=from_dlpack(self.sample_sfa, assumed_align=16),
                sfb_tensor=from_dlpack(self.sample_sfb, assumed_align=16),
                c_tensor=from_dlpack(self.sample_c, assumed_align=16),
                ab12_tensor=from_dlpack(self.sample_ab12, assumed_align=8),
                amax_tensor=(
                    from_dlpack(self.sample_amax, assumed_align=16)
                    if self.sample_amax is not None
                    else None
                ),
                sfc_tensor=(
                    from_dlpack(self.sample_sfc, assumed_align=16)
                    if self.sample_sfc is not None
                    else None
                ),
                norm_const_tensor=(
                    from_dlpack(self.sample_norm_const)
                    if self.sample_norm_const is not None
                    else None
                ),
                alpha=self.alpha,
                max_active_clusters=max_active_clusters,
                stream=current_stream,
            )
        elif self._kernel in (
            PersistentDenseGemmKernelNoDlpack,
            Sm100BlockScaledPersistentDenseGemmKernelNoDlpack,
        ):
            # Create cute pointers/tensors manually to avoid DLPack requirements
            # c (output) is always fp16/bf16 and is safe to use directly with dlpack
            a_ptr, a_shape, a_stride_order = self._make_cute_tensor_descriptor(
                self.sample_a, name="A"
            )
            b_ptr, b_shape, b_stride_order = self._make_cute_tensor_descriptor(
                self.sample_b, name="B"
            )
            ab12_ptr, ab12_shape, ab12_stride_order = self._make_cute_tensor_descriptor(
                self.sample_ab12, name="AB12"
            )

            if self._kernel is PersistentDenseGemmKernelNoDlpack:
                self._compiled_kernel = cute.compile(
                    gemm_swiglu,
                    a_ptr=a_ptr,
                    a_shape=a_shape,
                    a_order=a_stride_order,
                    b_ptr=b_ptr,
                    b_shape=b_shape,
                    b_order=b_stride_order,
                    ab12_ptr=ab12_ptr,
                    ab12_shape=ab12_shape,
                    ab12_order=ab12_stride_order,
                    c_cute=from_dlpack(self.sample_c),
                    alpha=self.alpha,
                    max_active_clusters=max_active_clusters,
                    stream=current_stream,
                )
            elif self._kernel is Sm100BlockScaledPersistentDenseGemmKernelNoDlpack:
                c_ptr, c_shape, c_stride_order = self._make_cute_tensor_descriptor(
                    self.sample_c, name="C"
                )
                sfa_ptr, sfa_shape, sfa_stride_order = (
                    self._make_cute_tensor_descriptor(self.sample_sfa, name="SFA")
                )
                sfb_ptr, sfb_shape, sfb_stride_order = (
                    self._make_cute_tensor_descriptor(self.sample_sfb, name="SFB")
                )
                amax_ptr, amax_shape, amax_stride_order = (
                    self._make_cute_tensor_descriptor(self.sample_amax, name="AMAX")
                )
                sfc_ptr, sfc_shape, sfc_stride_order = (
                    self._make_cute_tensor_descriptor(self.sample_sfc, name="SFC")
                )
                norm_const_ptr, norm_const_shape, norm_const_stride_order = (
                    self._make_cute_tensor_descriptor(
                        self.sample_norm_const, name="NORM_CONST"
                    )
                )

                self._compiled_kernel = cute.compile(
                    gemm_swiglu,
                    a_ptr=a_ptr,
                    a_shape=a_shape,
                    a_order=a_stride_order,
                    b_ptr=b_ptr,
                    b_shape=b_shape,
                    b_order=b_stride_order,
                    sfa_ptr=sfa_ptr,
                    sfa_shape=sfa_shape,
                    sfa_order=sfa_stride_order,
                    sfb_ptr=sfb_ptr,
                    sfb_shape=sfb_shape,
                    sfb_order=sfb_stride_order,
                    c_ptr=c_ptr,
                    c_shape=c_shape,
                    c_order=c_stride_order,
                    ab12_ptr=ab12_ptr,
                    ab12_shape=ab12_shape,
                    ab12_order=ab12_stride_order,
                    amax_ptr=amax_ptr,
                    amax_shape=amax_shape,
                    amax_order=amax_stride_order,
                    sfc_ptr=sfc_ptr,
                    sfc_shape=sfc_shape,
                    sfc_order=sfc_stride_order,
                    norm_const_ptr=norm_const_ptr,
                    norm_const_shape=norm_const_shape,
                    norm_const_order=norm_const_stride_order,
                    alpha=self.alpha,
                    max_active_clusters=max_active_clusters,
                    stream=current_stream,
                )
            else:
                raise NotImplementedError(
                    f"Unreachable: invalid kernel type {self._kernel}"
                )
        else:
            raise NotImplementedError(
                f"Unreachable: invalid kernel type {self._kernel}"
            )
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
        skip_compile: bool = False,
    ) -> None:
        self._logger.debug("Entering execute")
        current_stream = self._get_default_stream(current_stream)

        if not skip_compile:
            self._runtime_error_if(
                self._compiled_kernel is None,
                "GemmSwigluSm100 kernel not compiled; call compile() first or use execute(skip_compile=True)",
            )
            self._logger.debug("Executing with compiled kernel")

            if self._kernel is PersistentDenseGemmKernel:
                self._compiled_kernel(
                    a=from_dlpack(a_tensor),
                    b=from_dlpack(b_tensor),
                    ab12=from_dlpack(ab12_tensor),
                    c=from_dlpack(c_tensor),
                    alpha=alpha,
                    stream=current_stream,
                )
            elif self._kernel is Sm100BlockScaledPersistentDenseGemmKernel:
                amax_tensor = self._unpad_tensor_to_ndim(amax_tensor, 1, "amax")
                norm_const_tensor = self._unpad_tensor_to_ndim(
                    norm_const_tensor, 1, "norm_const"
                )
                self._compiled_kernel(
                    a_tensor=from_dlpack(a_tensor, assumed_align=16),
                    b_tensor=from_dlpack(b_tensor, assumed_align=16),
                    sfa_tensor=from_dlpack(sfa_tensor, assumed_align=16),
                    sfb_tensor=from_dlpack(sfb_tensor, assumed_align=16),
                    c_tensor=from_dlpack(c_tensor, assumed_align=16),
                    ab12_tensor=from_dlpack(ab12_tensor, assumed_align=8),
                    amax_tensor=(
                        from_dlpack(amax_tensor, assumed_align=16)
                        if amax_tensor is not None
                        else None
                    ),
                    sfc_tensor=(
                        from_dlpack(sfc_tensor, assumed_align=16)
                        if sfc_tensor is not None
                        else None
                    ),
                    norm_const_tensor=(
                        from_dlpack(norm_const_tensor)
                        if norm_const_tensor is not None
                        else None
                    ),
                    alpha=alpha,
                    stream=current_stream,
                )
            elif self._kernel in (
                PersistentDenseGemmKernelNoDlpack,
                Sm100BlockScaledPersistentDenseGemmKernelNoDlpack,
            ):
                a_ptr = self._make_cute_pointer(a_tensor, assumed_align=16)
                b_ptr = self._make_cute_pointer(b_tensor, assumed_align=16)
                ab12_ptr = self._make_cute_pointer(ab12_tensor, assumed_align=16)

                if self._kernel is PersistentDenseGemmKernelNoDlpack:
                    self._compiled_kernel(
                        a_ptr=a_ptr,
                        b_ptr=b_ptr,
                        ab12_ptr=ab12_ptr,
                        c_cute=from_dlpack(c_tensor),
                        alpha=alpha,
                        stream=current_stream,
                    )
                elif self._kernel is Sm100BlockScaledPersistentDenseGemmKernelNoDlpack:
                    amax_tensor = self._unpad_tensor_to_ndim(amax_tensor, 1, "amax")
                    norm_const_tensor = self._unpad_tensor_to_ndim(
                        norm_const_tensor, 1, "norm_const"
                    )
                    c_ptr = self._make_cute_pointer(c_tensor, assumed_align=16)
                    sfa_ptr = self._make_cute_pointer(sfa_tensor, assumed_align=16)
                    sfb_ptr = self._make_cute_pointer(sfb_tensor, assumed_align=16)
                    amax_ptr = self._make_cute_pointer(amax_tensor, assumed_align=16)
                    sfc_ptr = self._make_cute_pointer(sfc_tensor, assumed_align=16)
                    norm_const_ptr = self._make_cute_pointer(norm_const_tensor)
                    self._compiled_kernel(
                        a_ptr=a_ptr,
                        b_ptr=b_ptr,
                        sfa_ptr=sfa_ptr,
                        sfb_ptr=sfb_ptr,
                        c_ptr=c_ptr,
                        ab12_ptr=ab12_ptr,
                        amax_ptr=amax_ptr,
                        sfc_ptr=sfc_ptr,
                        norm_const_ptr=norm_const_ptr,
                        alpha=alpha,
                        stream=current_stream,
                    )
                else:
                    raise NotImplementedError(
                        f"Unreachable: invalid kernel type {type(self._compiled_kernel)}"
                    )
            else:
                raise NotImplementedError(
                    f"Unreachable: invalid kernel type {type(self._compiled_kernel)}"
                )
            self._logger.debug("Executed with compiled kernel successfully")
        else:  # skip_compile
            self._logger.debug("Executing without compiled kernel (JIT)")

            if self._kernel is PersistentDenseGemmKernel:
                gemm_swiglu = self._kernel(
                    acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
                    use_2cta_instrs=(self.mma_tiler_mn[0] == 256),
                    mma_tiler_mn=self.mma_tiler_mn,
                    cluster_shape_mn=self.cluster_shape_mn,
                )
                gemm_swiglu(
                    a=from_dlpack(a_tensor),
                    b=from_dlpack(b_tensor),
                    ab12=from_dlpack(ab12_tensor),
                    c=from_dlpack(c_tensor),
                    alpha=alpha,
                    max_active_clusters=cutlass.utils.HardwareInfo().get_max_active_clusters(
                        self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
                    ),
                    stream=current_stream,
                )
            elif self._kernel is PersistentDenseGemmKernelNoDlpack:
                gemm_swiglu = self._kernel(
                    acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
                    use_2cta_instrs=(self.mma_tiler_mn[0] == 256),
                    mma_tiler_mn=self.mma_tiler_mn,
                    cluster_shape_mn=self.cluster_shape_mn,
                )

                a_ptr, a_shape, a_stride_order = self._make_cute_tensor_descriptor(
                    a_tensor, name="A"
                )
                b_ptr, b_shape, b_stride_order = self._make_cute_tensor_descriptor(
                    b_tensor, name="B"
                )
                ab12_ptr, ab12_shape, ab12_stride_order = (
                    self._make_cute_tensor_descriptor(ab12_tensor, name="AB12")
                )

                gemm_swiglu(
                    a_ptr=a_ptr,
                    a_shape=a_shape,
                    a_order=a_stride_order,
                    b_ptr=b_ptr,
                    b_shape=b_shape,
                    b_order=b_stride_order,
                    ab12_ptr=ab12_ptr,
                    ab12_shape=ab12_shape,
                    ab12_order=ab12_stride_order,
                    c_cute=from_dlpack(c_tensor),
                    alpha=alpha,
                    max_active_clusters=cutlass.utils.HardwareInfo().get_max_active_clusters(
                        self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
                    ),
                    stream=current_stream,
                )
            elif self._kernel is Sm100BlockScaledPersistentDenseGemmKernel:
                gemm_swiglu = self._kernel(
                    sf_vec_size=self.sf_vec_size,
                    mma_tiler_mn=self.mma_tiler_mn,
                    cluster_shape_mn=self.cluster_shape_mn,
                    vector_f32=self.vector_f32,
                    ab12_stages=self.ab12_stages,
                )
                amax_tensor = self._unpad_tensor_to_ndim(amax_tensor, 1, "amax")
                norm_const_tensor = self._unpad_tensor_to_ndim(
                    norm_const_tensor, 1, "norm_const"
                )
                gemm_swiglu(
                    a_tensor=from_dlpack(a_tensor, assumed_align=16),
                    b_tensor=from_dlpack(b_tensor, assumed_align=16),
                    sfa_tensor=from_dlpack(sfa_tensor, assumed_align=16),
                    sfb_tensor=from_dlpack(sfb_tensor, assumed_align=16),
                    c_tensor=from_dlpack(c_tensor, assumed_align=16),
                    ab12_tensor=from_dlpack(ab12_tensor, assumed_align=8),
                    amax_tensor=(
                        from_dlpack(amax_tensor, assumed_align=16)
                        if amax_tensor is not None
                        else None
                    ),
                    sfc_tensor=(
                        from_dlpack(sfc_tensor, assumed_align=16)
                        if sfc_tensor is not None
                        else None
                    ),
                    norm_const_tensor=(
                        from_dlpack(norm_const_tensor)
                        if norm_const_tensor is not None
                        else None
                    ),
                    alpha=alpha,
                    max_active_clusters=cutlass.utils.HardwareInfo().get_max_active_clusters(
                        self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
                    ),
                    stream=current_stream,
                )
            elif self._kernel is Sm100BlockScaledPersistentDenseGemmKernelNoDlpack:
                gemm_swiglu = self._kernel(
                    sf_vec_size=self.sf_vec_size,
                    mma_tiler_mn=self.mma_tiler_mn,
                    cluster_shape_mn=self.cluster_shape_mn,
                    vector_f32=self.vector_f32,
                    ab12_stages=self.ab12_stages,
                )
                amax_tensor = self._unpad_tensor_to_ndim(amax_tensor, 1, "amax")
                norm_const_tensor = self._unpad_tensor_to_ndim(
                    norm_const_tensor, 1, "norm_const"
                )

                a_ptr, a_shape, a_stride_order = self._make_cute_tensor_descriptor(
                    a_tensor, name="A"
                )
                b_ptr, b_shape, b_stride_order = self._make_cute_tensor_descriptor(
                    b_tensor, name="B"
                )
                ab12_ptr, ab12_shape, ab12_stride_order = (
                    self._make_cute_tensor_descriptor(ab12_tensor, name="AB12")
                )
                c_ptr, c_shape, c_stride_order = self._make_cute_tensor_descriptor(
                    c_tensor, name="C"
                )
                sfa_ptr, sfa_shape, sfa_stride_order = (
                    self._make_cute_tensor_descriptor(sfa_tensor, name="SFA")
                )
                sfb_ptr, sfb_shape, sfb_stride_order = (
                    self._make_cute_tensor_descriptor(sfb_tensor, name="SFB")
                )
                amax_ptr, amax_shape, amax_stride_order = (
                    self._make_cute_tensor_descriptor(amax_tensor, name="AMAX")
                )
                sfc_ptr, sfc_shape, sfc_stride_order = (
                    self._make_cute_tensor_descriptor(sfc_tensor, name="SFC")
                )
                norm_const_ptr, norm_const_shape, norm_const_stride_order = (
                    self._make_cute_tensor_descriptor(
                        norm_const_tensor, name="NORM_CONST"
                    )
                )

                gemm_swiglu(
                    a_ptr=a_ptr,
                    a_shape=a_shape,
                    a_order=a_stride_order,
                    b_ptr=b_ptr,
                    b_shape=b_shape,
                    b_order=b_stride_order,
                    sfa_ptr=sfa_ptr,
                    sfa_shape=sfa_shape,
                    sfa_order=sfa_stride_order,
                    sfb_ptr=sfb_ptr,
                    sfb_shape=sfb_shape,
                    sfb_order=sfb_stride_order,
                    c_ptr=c_ptr,
                    c_shape=c_shape,
                    c_order=c_stride_order,
                    ab12_ptr=ab12_ptr,
                    ab12_shape=ab12_shape,
                    ab12_order=ab12_stride_order,
                    amax_ptr=amax_ptr,
                    amax_shape=amax_shape,
                    amax_order=amax_stride_order,
                    sfc_ptr=sfc_ptr,
                    sfc_shape=sfc_shape,
                    sfc_order=sfc_stride_order,
                    norm_const_ptr=norm_const_ptr,
                    norm_const_shape=norm_const_shape,
                    norm_const_order=norm_const_stride_order,
                    alpha=alpha,
                    max_active_clusters=cutlass.utils.HardwareInfo().get_max_active_clusters(
                        self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
                    ),
                    stream=current_stream,
                )
            else:
                raise NotImplementedError(
                    f"Unreachable: invalid kernel type {type(self._kernel)}"
                )
            self._logger.debug("Executed without compiled kernel (JIT) successfully")


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
) -> Tuple[torch.Tensor, ...]:

    _logger.debug("gemm_swiglu_wrapper_sm100: Creating empty output tensors ab12 and c")
    m, k, l = a_tensor.shape
    n, k, l = b_tensor.shape
    ab12_tensor, c_tensor = None, None
    if c_major == "m":
        ab12_tensor = torch.empty_strided(
            (m, n, l), (1, m, m * n), dtype=ab12_dtype, device=a_tensor.device
        )
        c_tensor = torch.empty_strided(
            (m, n // 2, l), (1, m, m * n // 2), dtype=c_dtype, device=a_tensor.device
        )
    elif c_major == "n":
        ab12_tensor = torch.empty_strided(
            (m, n, l), (n, 1, m * n), dtype=ab12_dtype, device=a_tensor.device
        )
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
        _logger.debug(
            "gemm_swiglu_wrapper_sm100: Detected sfa_tensor and sfb_tensor, constructing quantized output tensors"
        )
        if c_dtype in {torch.float8_e5m2, torch.float8_e4m3fn}:
            _logger.debug(
                "gemm_swiglu_wrapper_sm100: Detected fp8 c_dtype, constructing sfc_tensor"
            )

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
        if (
            a_tensor.dtype in {torch.float4_e2m1fn_x2, torch.uint8}
            and c_dtype == torch.bfloat16
        ):
            _logger.debug(
                "gemm_swiglu_wrapper_sm100: Detected fp4 ab_dtype and bf16 c_dtype, constructing amax_tensor"
            )
            amax_tensor = torch.full(
                (1, 1, 1), -float("inf"), device=a_tensor.device, dtype=torch.float32
            )

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
        _logger.debug(
            "gemm_swiglu_wrapper_sm100: Using previously cached GemmSwigluSm100 object"
        )
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
        _logger.debug(
            "gemm_swiglu_wrapper_sm100: No previously cached GemmSwigluSm100 object found, creating new GemmSwigluSm100 object"
        )
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
        gemm_swiglu.compile(current_stream=stream)
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

    if sfa_tensor is not None and sfb_tensor is not None:
        return ab12_tensor, c_tensor, sfc_tensor, amax_tensor
    else:
        return ab12_tensor, c_tensor
