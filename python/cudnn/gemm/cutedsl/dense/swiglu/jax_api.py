# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy JAX entry point for dense GEMM + SwiGLU. It lowers to ``cudnn.ops.gemm_glu_forward_jax``."""

from typing import Any, Optional, Tuple

import cutlass

from ..glu.jax_api import gemm_glu_forward_jax


def gemm_swiglu_jax_sm100(
    a_tensor: Any,
    b_tensor: Any,
    alpha: float = 1.0,
    ab12_dtype: Any = cutlass.Float32,
    c_dtype: Any = cutlass.Float16,
    acc_dtype: Any = cutlass.Float32,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    ### Quantize only arguments
    sfa_tensor: Optional[Any] = None,
    sfb_tensor: Optional[Any] = None,
    sf_vec_size: int = 16,
    vector_f32: bool = False,
    ab12_stages: int = 4,
) -> Tuple[Any, Any]:
    """Dense GEMM + SwiGLU as an XLA custom call. Same as ``gemm_glu_forward_jax`` with ``act_func="swiglu"``."""
    return gemm_glu_forward_jax(
        a_tensor,
        b_tensor,
        alpha=alpha,
        ab12_dtype=ab12_dtype,
        c_dtype=c_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sfa_tensor=sfa_tensor,
        sfb_tensor=sfb_tensor,
        sf_vec_size=sf_vec_size,
        vector_f32=vector_f32,
        ab12_stages=ab12_stages,
    )
