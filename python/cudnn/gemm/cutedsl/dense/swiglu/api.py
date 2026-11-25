# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy dense GEMM + SwiGLU names. They lower to the dense GLU op with ``act_func="swiglu"``."""

from typing import Any, Optional, Tuple

import cutlass
from cuda.bindings import driver as cuda

from cudnn.api_base import TupleDict
from ..glu.api import GemmGlu, gemm_glu_forward


class GemmSwigluSm100(GemmGlu):
    """Persistent GEMM + SwiGLU for SM100: the dense GLU plan with ``act_func="swiglu"``."""


def gemm_swiglu_wrapper_sm100(
    a_tensor: Any,
    b_tensor: Any,
    alpha: float = 1.0,
    c_major: str = "n",
    ab12_dtype: Any = cutlass.Float32,
    c_dtype: Any = cutlass.Float16,
    acc_dtype: Any = cutlass.Float32,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    ### Quantize only arguments
    sfa_tensor: Optional[Any] = None,
    sfb_tensor: Optional[Any] = None,
    norm_const_tensor: Optional[Any] = None,
    sf_vec_size: int = 16,
    vector_f32: bool = False,
    ab12_stages: int = 4,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Dense GEMM + SwiGLU. Same as ``cudnn.ops.gemm_glu_forward`` with ``act_func="swiglu"``."""
    return gemm_glu_forward(
        a_tensor,
        b_tensor,
        alpha=alpha,
        c_major=c_major,
        ab12_dtype=ab12_dtype,
        c_dtype=c_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sfa_tensor=sfa_tensor,
        sfb_tensor=sfb_tensor,
        norm_const_tensor=norm_const_tensor,
        sf_vec_size=sf_vec_size,
        vector_f32=vector_f32,
        ab12_stages=ab12_stages,
        stream=stream,
    )
