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

from cuda.bindings import driver as cuda
import torch
from typing import Tuple, Optional

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_ptr
from packaging import version
import cutlass.cute.math as math

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.api_base import APIBase


class GemmSwigluSm100(APIBase):
    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_b: torch.Tensor,
        sample_c: torch.Tensor,
        sample_glu: torch.Tensor,
        alpha: float = 1.0,
        acc_dtype: torch.dtype = torch.float32,
        use_2cta_instrs: bool = False,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        cluster_shape_mn: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self._kernel = PersistentDenseGemmKernel

        self._logger.warning("GemmSwigluSm100 is an experimental API")
        self._logger.debug("Entering __init__")

        self.sample_a = sample_a
        self.sample_b = sample_b
        self.sample_c = sample_c
        self.sample_glu = sample_glu
        self.alpha = alpha
        self.acc_dtype = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.mma_tiler_mn = mma_tiler_mn
        if cluster_shape_mn is None:
            self.cluster_shape_mn = (1, 1) if not use_2cta_instrs else (2, 2)
        else:
            self.cluster_shape_mn = cluster_shape_mn

        self._logger.debug(
            f"__init__ completed with args: sample_a {sample_a.shape}, sample_b {sample_b.shape}, sample_c {sample_c.shape}, sample_glu {sample_glu.shape}, alpha {alpha}, acc_dtype {acc_dtype}, use_2cta_instrs {use_2cta_instrs}, mma_tiler_mn {mma_tiler_mn}, cluster_shape_mn {cluster_shape_mn}"
        )

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        self._logger.debug("Checking tensor shapes, strides, and dtypes")
        m, k, l = self.sample_a.shape
        n, k, l = self.sample_b.shape
        m, n, l = self.sample_c.shape
        m, n_2, l = self.sample_glu.shape
        ab_dtype = self.sample_a.dtype
        c_dtype = self.sample_c.dtype
        acc_dtype = self.acc_dtype
        glu_dtype = self.sample_glu.dtype

        if self.sample_a.shape != (m, k, l):
            raise ValueError(
                f"Input/Output shape mismatch: expected A tensor shape {m, k, l}, got {self.sample_a.shape}"
            )
        if self.sample_b.shape != (n, k, l):
            raise ValueError(
                f"Input/Output shape mismatch: expected B tensor shape {n, k, l}, got {self.sample_b.shape}"
            )
        if self.sample_c.shape != (m, n, l):
            raise ValueError(
                f"Input/Output shape mismatch: expected C tensor shape {m, n, l}, got {self.sample_c.shape}"
            )
        if self.sample_glu.shape != (m, n // 2, l):
            raise ValueError(
                f"Input/Output shape mismatch: expected GLU tensor shape {m, n // 2, l}, got {self.sample_glu.shape}"
            )
        if self.sample_a.dtype != self.sample_b.dtype:
            raise ValueError(
                f"A and B tensor dtypes must match, got {self.sample_a.dtype} and {self.sample_b.dtype}"
            )

        if self.sample_a.stride() == (1, m, m * k):
            self.a_major = "m"
        elif self.sample_a.stride() == (k, 1, m * k):
            self.a_major = "k"
        else:
            raise ValueError(
                f"Unsupported A tensor stride: expected {{(1, m, m * k), (k, 1, m * k)}}, got {self.sample_a.stride()}"
            )

        if self.sample_b.stride() == (1, n, n * k):
            self.b_major = "n"
        elif self.sample_b.stride() == (k, 1, n * k):
            self.b_major = "k"
        else:
            raise ValueError(
                f"Unsupported B tensor stride: expected {{(1, n, n * k), (k, 1, n * k)}}, got {self.sample_b.stride()}"
            )

        if self.sample_c.stride() == (1, m, m * n) and self.sample_glu.stride() == (
            1,
            m,
            m * n_2,
        ):
            self.c_major = "m"
        elif self.sample_c.stride() == (n, 1, m * n) and self.sample_glu.stride() == (
            n_2,
            1,
            m * n_2,
        ):
            self.c_major = "n"
        else:
            raise ValueError(
                f"Unsupported C/glu tensor stride: expected C stride (1, m, m*n) glu stride (1, m, m*n/2) or C stride (n, 1, m*n) glu stride (n/2, 1, m*n/2), got {self.sample_c.stride()} and {self.sample_glu.stride()}"
            )

        self._logger.debug("Checking data types")
        if ab_dtype not in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        }:
            raise ValueError(
                f"Unsupported ab_dtype: expected {{float16, bfloat16, float32, float8_e4m3fn, float8_e5m2}}, got {ab_dtype}"
            )

        if acc_dtype not in {torch.float32, torch.float16}:
            raise ValueError(
                f"Unsupported acc_dtype: expected {{float32, float16}}, got {acc_dtype}"
            )

        if acc_dtype == torch.float32:
            if c_dtype not in {
                torch.float32,
                torch.float16,
                torch.bfloat16,
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            }:
                raise ValueError(
                    f"Unsupported c_dtype: for float32 acc_dtype, expected {{float32, float16, bfloat16, float8_e4m3fn, float8_e5m2}}, got {c_dtype}"
                )
            if c_dtype in {torch.float8_e5m2, torch.float8_e4m3fn}:
                raise NotImplementedError(
                    f"c_dtype {{torch.float8_e5m2, torch.float8_e4m3fn}} is currently disabled"
                )
        elif acc_dtype == torch.float16:
            if ab_dtype not in {
                torch.float16,
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            }:
                raise ValueError(
                    f"Unsupported ab_dtype: for float16 acc_dtype, expected {{float16, float8_e4m3fn, float8_e5m2}}, got {ab_dtype}"
                )
            if c_dtype not in {
                torch.bfloat16,
                torch.float16,
            }:
                raise ValueError(
                    f"Unsupported c_dtype: for float16 acc_dtype, expected {{bfloat16, float16}}, got {c_dtype}"
                )

        if glu_dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError(
                f"Unsupported glu_dtype: expected {{float16, bfloat16}}, got {glu_dtype}"
            )
        self.ab_dtype = ab_dtype
        self.c_dtype = c_dtype
        self.glu_dtype = glu_dtype

        self._logger.debug("Checking MMA tile shape and cluster shape")

        def is_power_of_2(x):
            return x > 0 and (x & (x - 1)) == 0

        if self.mma_tiler_mn[0] not in [128, 256]:
            raise ValueError(
                f"Invalid MMA tile shape: expected mma_tiler_mn[0] in {{128, 256}}, got {self.mma_tiler_mn[0]}"
            )
        if self.mma_tiler_mn[0] == 256 and not self.use_2cta_instrs:
            raise ValueError(
                "Unsupported MMA tile shape: 256xmma tile shape is only supported with use_2cta_instrs=True"
            )
        if self.mma_tiler_mn[1] not in range(32, 257, 32):
            raise ValueError(
                f"Invalid MMA tile shape: expected mma_tiler_mn[1] in {{32, 64, ..., 224, 256}}, got {self.mma_tiler_mn[1]}"
            )
        if not (self.cluster_shape_mn[0] % (2 if self.use_2cta_instrs else 1) == 0):
            raise ValueError("Invalid cluster shape")
        if not self.use_2cta_instrs and self.cluster_shape_mn != (1, 1):
            raise ValueError(
                "Invalid cluster shape: cluster_shape must be (1, 1) when use_2cta_instrs=False"
            )
        if not self.use_2cta_instrs and c_dtype == torch.float32:
            raise ValueError(
                "Invalid c_dtype: self.use_2cta_instrs=False is incompatbile with float32 accumulator"
            )
        if not (
            self.cluster_shape_mn[0] * self.cluster_shape_mn[1] <= 16
            and self.cluster_shape_mn[0] > 0
            and self.cluster_shape_mn[1] > 0
            and is_power_of_2(self.cluster_shape_mn[0])
            and is_power_of_2(self.cluster_shape_mn[1])
        ):
            raise ValueError(
                f"Invalid cluster shape: expected values to be powers of 2 and cluster_shape_mn[0] * cluster_shape_mn[1] <= 16, got {self.cluster_shape_mn[0]},{self.cluster_shape_mn[1]}"
            )

        if (
            self.mma_tiler_mn == (128, 128)
            and self.cluster_shape_mn == (1, 1)
            and self.c_major != "m"
        ):
            raise ValueError(
                "Invalid MMA tile shape and C major combination: (128, 128) mma tile shape with 1x1 cluster shape is only supported with c_major='m'"
            )
        if self.mma_tiler_mn != (128, 128) and self.c_major != "m":
            raise ValueError(
                f"Invalid c_major: for non-128x128mma tile shape, c_major must be 'm', got {self.c_major}"
            )
        if self.cluster_shape_mn != (1, 1) and self.mma_tiler_mn[0] == 128:
            if self.mma_tiler_mn != (
                128,
                128,
            ):
                raise ValueError(
                    "Invalid MMA tile shape: for non-1x1 cluster shape and 128xmma tile shape, mma_tiler_mn must be (128, 128)"
                )
        if self.mma_tiler_mn[0] == 256 and c_dtype == torch.float32:
            raise NotImplementedError(
                "mma_tiler_mn[0] == 256 and c_dtype == torch.float32 currently disabled"
            )

        self._logger.debug("Checking tensor alignment")

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = (
                16 * 8 // (_convert_to_cutlass_data_type(dtype).width)
            )
            return num_major_elements % num_contiguous_elements == 0

        if not (
            check_contigous_16B_alignment(ab_dtype, self.a_major == "m", (m, k, l))
            and check_contigous_16B_alignment(ab_dtype, self.b_major == "n", (n, k, l))
            and check_contigous_16B_alignment(c_dtype, self.c_major == "m", (m, n, l))
        ):
            raise ValueError("Invalid tensor alignment: tensors must be 16B aligned")

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

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def compile(self, current_stream: Optional[cuda.CUstream] = None) -> None:
        self._logger.debug("Entering compile")
        current_stream = self._get_default_stream(current_stream)
        self._ensure_support_checked()

        torch_version = version.parse(torch.__version__)
        _fp8_dlpack_supported = version.parse(
            torch_version.base_version
        ) >= version.parse("2.10.0")
        use_no_dlpack_kernel = (
            self.ab_dtype in {torch.float8_e5m2, torch.float8_e4m3fn}
            or self.c_dtype in {torch.float8_e5m2, torch.float8_e4m3fn}
        ) and not _fp8_dlpack_supported

        if use_no_dlpack_kernel:
            self._logger.debug(
                "Running no_dlpack kernel wrapper due to fp8 dtype on incompatible torch version"
            )
            self._kernel = PersistentDenseGemmKernelNoDlpack
        else:
            self._kernel = PersistentDenseGemmKernel

        gemm_swiglu = self._kernel(
            _convert_to_cutlass_data_type(self.acc_dtype),
            self.use_2cta_instrs,
            self.mma_tiler_mn,
            self.cluster_shape_mn,
        )
        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(
            self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        )

        if not use_no_dlpack_kernel:
            self._logger.debug("Compiling gemm_swiglu (dlpack)")
            self._compiled_kernel = cute.compile(
                gemm_swiglu,
                from_dlpack(self.sample_a),
                from_dlpack(self.sample_b),
                from_dlpack(self.sample_c),
                from_dlpack(self.sample_glu),
                self.alpha,
                max_active_clusters,
                current_stream,
                lambda x: x / (1 + math.exp(-1 * x, True)),
            )
        else:  # use_no_dlpack
            # Create cute pointers/tensors manually to avoid DLPack requirements
            # glu is always fp16/bf16 and is safe to use directly with dlpack
            a_ptr = make_ptr(
                _convert_to_cutlass_data_type(self.sample_a.dtype),
                self.sample_a.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            b_ptr = make_ptr(
                _convert_to_cutlass_data_type(self.sample_b.dtype),
                self.sample_b.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            c_ptr = make_ptr(
                _convert_to_cutlass_data_type(self.sample_c.dtype),
                self.sample_c.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            a_shape = tuple(self.sample_a.shape)
            b_shape = tuple(self.sample_b.shape)
            c_shape = tuple(self.sample_c.shape)
            a_order = (1, 0, 2) if self.a_major == "k" else (0, 1, 2)
            b_order = (1, 0, 2) if self.b_major == "k" else (0, 1, 2)
            c_order = (1, 0, 2) if self.c_major == "n" else (0, 1, 2)

            self._compiled_kernel = cute.compile(
                gemm_swiglu,
                a_ptr,
                tuple(self.sample_a.shape),
                a_order,
                b_ptr,
                tuple(self.sample_b.shape),
                b_order,
                c_ptr,
                tuple(self.sample_c.shape),
                c_order,
                from_dlpack(self.sample_glu),
                self.alpha,
                max_active_clusters,
                current_stream,
                lambda x: x / (1 + math.exp(-1 * x, True)),
            )
        self._logger.debug("Kernel compiled successfully")

    def execute(
        self,
        a_tensor: torch.Tensor,
        b_tensor: torch.Tensor,
        c_tensor: torch.Tensor,
        glu_tensor: torch.Tensor,
        alpha: float = 1.0,
        current_stream: Optional[cuda.CUstream] = None,
        skip_compile: bool = False,
    ) -> None:
        self._logger.debug("Entering execute")
        current_stream = self._get_default_stream(current_stream)

        torch_version = version.parse(torch.__version__)
        _fp8_dlpack_supported = version.parse(
            torch_version.base_version
        ) >= version.parse("2.10.0")
        use_no_dlpack_kernel = (
            self.ab_dtype in {torch.float8_e5m2, torch.float8_e4m3fn}
            or self.c_dtype in {torch.float8_e5m2, torch.float8_e4m3fn}
        ) and not _fp8_dlpack_supported

        if not use_no_dlpack_kernel:
            if not skip_compile:
                if self._compiled_kernel is None:
                    raise RuntimeError(
                        "GemmSwigluSm100 kernel not compiled; call compile() first or use execute(skip_compile=True)"
                    )
                self._logger.debug("Executing with compiled kernel")
                self._compiled_kernel(
                    from_dlpack(a_tensor),
                    from_dlpack(b_tensor),
                    from_dlpack(c_tensor),
                    from_dlpack(glu_tensor),
                    alpha,
                    current_stream,
                )
                self._logger.debug("Executed with compiled kernel successfully")
            else:
                self._logger.debug("Executing without compiled kernel (JIT)")
                gemm_swiglu = self._kernel(
                    _convert_to_cutlass_data_type(self.acc_dtype),
                    self.use_2cta_instrs,
                    self.mma_tiler_mn,
                    self.cluster_shape_mn,
                )
                gemm_swiglu(
                    from_dlpack(a_tensor),
                    from_dlpack(b_tensor),
                    from_dlpack(c_tensor),
                    from_dlpack(glu_tensor),
                    alpha,
                    cutlass.utils.HardwareInfo().get_max_active_clusters(
                        self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
                    ),
                    current_stream,
                    lambda x: x / (1 + math.exp(-1 * x, True)),
                )
        else:  # use_no_dlpack
            a_ptr = make_ptr(
                _convert_to_cutlass_data_type(a_tensor.dtype),
                a_tensor.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            b_ptr = make_ptr(
                _convert_to_cutlass_data_type(b_tensor.dtype),
                b_tensor.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            c_ptr = make_ptr(
                _convert_to_cutlass_data_type(c_tensor.dtype),
                c_tensor.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )

            if not skip_compile:
                if self._compiled_kernel is None:
                    raise RuntimeError(
                        "GemmSwigluSm100 kernel not compiled; call compile() first or use execute(skip_compile=True)"
                    )
                self._logger.debug("Executing with compiled kernel")
                self._compiled_kernel(
                    a_ptr,
                    b_ptr,
                    c_ptr,
                    from_dlpack(glu_tensor),
                    alpha,
                    current_stream,
                )
                self._logger.debug("Executed with compiled kernel successfully")
            else:
                self._logger.debug("Executing without compiled kernel (JIT)")
                gemm_swiglu = self._kernel(
                    _convert_to_cutlass_data_type(self.acc_dtype),
                    self.use_2cta_instrs,
                    self.mma_tiler_mn,
                    self.cluster_shape_mn,
                )

                a_shape = tuple(a_tensor.shape)
                b_shape = tuple(b_tensor.shape)
                c_shape = tuple(c_tensor.shape)
                a_order = (1, 0, 2) if self.a_major == "k" else (0, 1, 2)
                b_order = (1, 0, 2) if self.b_major == "k" else (0, 1, 2)
                c_order = (1, 0, 2) if self.c_major == "n" else (0, 1, 2)

                gemm_swiglu(
                    a_ptr,
                    tuple(a_tensor.shape),
                    a_order,
                    b_ptr,
                    tuple(b_tensor.shape),
                    b_order,
                    c_ptr,
                    tuple(c_tensor.shape),
                    c_order,
                    from_dlpack(glu_tensor),
                    alpha,
                    cutlass.utils.HardwareInfo().get_max_active_clusters(
                        self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
                    ),
                    current_stream,
                )
            self._logger.debug("Executed successfully")


import logging

_logger = logging.getLogger(__name__)
_cache_of_GemmSwigluSm100Objects = {}


def gemm_swiglu_wrapper_sm100(
    a_tensor: torch.Tensor,
    b_tensor: torch.Tensor,
    alpha: float = 1.0,
    c_major: str = "n",
    c_dtype: torch.dtype = torch.float32,
    glu_dtype: torch.dtype = torch.float16,
    acc_dtype: torch.dtype = torch.float32,
    use_2cta_instrs: bool = False,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    stream: Optional[cuda.CUstream] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    _logger.debug("gemm_swiglu_wrapper_sm100: Creating empty output tensors c and glu")
    m, k, l = a_tensor.shape
    n, k, l = b_tensor.shape
    c_tensor, glu_tensor = None, None
    if c_major == "m":

        c_tensor = torch.empty_strided(
            (m, n, l), (1, m, m * n), dtype=c_dtype, device="cuda"
        )
        glu_tensor = torch.empty_strided(
            (m, n // 2, l), (1, m, m * n // 2), dtype=glu_dtype, device="cuda"
        )
    elif c_major == "n":
        c_tensor = torch.empty_strided(
            (m, n, l), (n, 1, m * n), dtype=c_dtype, device="cuda"
        )
        glu_tensor = torch.empty_strided(
            (m, n // 2, l), (n // 2, 1, m * n // 2), dtype=glu_dtype, device="cuda"
        )
    else:
        raise ValueError(f"c_major must be either 'm' or 'n', got {c_major}")

    cache_key = (
        a_tensor.shape,
        b_tensor.shape,
        a_tensor.dtype,
        b_tensor.dtype,
        a_tensor.stride(),
        b_tensor.stride(),
        alpha,
        c_major,
        c_dtype,
        glu_dtype,
        acc_dtype,
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
    )
    if cache_key in _cache_of_GemmSwigluSm100Objects:
        _logger.debug(
            "gemm_swiglu_wrapper_sm100: Using previously cached GemmSwigluSm100 object"
        )
        gemm_swiglu = _cache_of_GemmSwigluSm100Objects[cache_key]
        gemm_swiglu.execute(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            c_tensor=c_tensor,
            glu_tensor=glu_tensor,
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
            sample_c=c_tensor,
            sample_glu=glu_tensor,
            alpha=alpha,
            acc_dtype=acc_dtype,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
        )
        assert gemm_swiglu.check_support(), "Unsupported testcase"
        gemm_swiglu.compile(current_stream=stream)
        gemm_swiglu.execute(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            c_tensor=c_tensor,
            glu_tensor=glu_tensor,
            alpha=alpha,
            current_stream=stream,
        )
        _cache_of_GemmSwigluSm100Objects[cache_key] = gemm_swiglu

    return c_tensor, glu_tensor
