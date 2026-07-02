# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for dense grouped GEMM + GLU + Hadamard on SM100."""

from __future__ import annotations

from functools import lru_cache
import os
from typing import Any, NamedTuple, Optional

import jax.numpy as jnp

from ..._jax.cutedsl import BufferSpec, call_cutedsl
from ..._jax.gemm import (
    block_scale_tensor_spec,
    gemm_a_tensor_spec,
    gemm_b_tensor_spec,
    gemm_c_tensor_spec,
    probability_tensor_spec,
    require_array,
)
from ..._jax.grouped_gemm import (
    grouped_bias_tensor_spec,
    grouped_workspace_tensor_spec,
    require_grouped_block_scales,
    require_grouped_gemm_inputs,
    require_grouped_probability,
)
from ..._jax.validation import require_dtype
from ...gemm_validation import (
    require_contiguous_alignment,
    require_shape,
    resolve_max_active_clusters,
)


class GroupedGemmGluHadamardResult(NamedTuple):
    """Functional outputs from grouped GEMM + GLU + Hadamard."""

    c_tensor: Any
    d_tensor: Any
    amax_tensor: Any
    post_rht_amax_tensor: Any


def hadamard_values(size: int) -> tuple[tuple[int, ...], ...]:
    """Return an unnormalized Sylvester Hadamard matrix as Python values."""

    if size < 1 or size & (size - 1):
        raise ValueError(f"Hadamard size must be a positive power of two, got {size}")
    matrix = ((1,),)
    while len(matrix) < size:
        matrix = tuple(row + row for row in matrix) + tuple(row + tuple(-value for value in row) for row in matrix)
    return matrix


@lru_cache(maxsize=None)
def _make_launcher(
    *,
    acc_dtype: Any,
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    sf_vec_size: int,
    vector_f32: bool,
    expert_cnt: int,
    act_func: str,
    has_bias: bool,
    has_hadamard: bool,
    use_dynamic_sched: bool,
    use_tmem_post_rht_amax: bool,
    cluster_overlap_margin: int,
):
    def launch(stream, *args):
        import cutlass
        from cutlass.cute.nvgpu import OperandMajorMode
        from cutlass.jax import jax_to_cutlass_dtype

        from ..moe_utils import MoEWeightMode
        from .moe_blockscaled_grouped_gemm_glu_hadamard import BlockScaledMoEGroupedGemmGluHadamardKernel

        arg_idx = 0

        def take():
            nonlocal arg_idx
            value = args[arg_idx]
            arg_idx += 1
            return value

        a = take()
        b = take()
        sfa = take()
        sfb = take()
        padded_offsets = take()
        alpha = take()
        prob = take()
        hadamard = take() if has_hadamard else None
        bias = take() if has_bias else None
        c = take()
        d = take()
        amax = take()
        post_rht_amax = take()
        workspace = take()
        if arg_idx != len(args):
            raise RuntimeError(f"Unexpected grouped GEMM argument count: consumed {arg_idx}, received {len(args)}")

        kernel = BlockScaledMoEGroupedGemmGluHadamardKernel(
            sf_vec_size=sf_vec_size,
            acc_dtype=jax_to_cutlass_dtype(acc_dtype),
            use_2cta_instrs=True,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            vectorized_f32=vector_f32,
            expert_cnt=expert_cnt,
            weight_mode=MoEWeightMode.DENSE,
            use_dynamic_sched=use_dynamic_sched,
            act_func=act_func,
            enable_bias=has_bias,
            use_tmem_post_rht_amax=use_tmem_post_rht_amax,
        )
        max_active_clusters = resolve_max_active_clusters(
            cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1]),
            cluster_overlap_margin,
        )
        kernel(
            a,
            b,
            sfa,
            sfb,
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int64(0),
            OperandMajorMode.K,
            workspace.iterator,
            c,
            d,
            amax,
            post_rht_amax,
            padded_offsets,
            alpha,
            prob,
            hadamard,
            bias,
            max_active_clusters,
            stream,
            linear_offset=cutlass.Float32(1.0 if act_func == "geglu" else 0.0),
        )

    return launch


def grouped_gemm_glu_hadamard_wrapper_sm100(
    a_tensor: Any,
    b_tensor: Any,
    sfa_tensor: Any,
    sfb_tensor: Any,
    padded_offsets: Any,
    alpha_tensor: Any,
    prob_tensor: Any,
    bias_tensor: Optional[Any] = None,
    hadamard_tensor: Optional[Any] = None,
    acc_dtype: Any = None,
    c_dtype: Any = None,
    d_dtype: Any = None,
    cd_major: str = "n",
    mma_tiler_mn: tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[tuple[int, int]] = None,
    sf_vec_size: int = 16,
    vector_f32: bool = False,
    m_aligned: int = 256,
    act_func: str = "swiglu",
    use_dynamic_sched: bool = False,
    use_tmem_post_rht_amax: bool = False,
    *,
    b_major: str = "k",
) -> GroupedGemmGluHadamardResult:
    """Compute dense native-FP4 grouped GEMM with GLU and Hadamard amax.

    A and B use JAX's native ``float4_e2m1fn`` dtype with logical shapes
    ``(M, K, 1)`` and ``(N, K, L)``. Raw ``uint8`` FP4 payloads and discrete
    per-expert pointer arrays are intentionally not part of this API.

    Runtime ``padded_offsets`` values must be nondecreasing multiples of 256,
    must not exceed M, and must end at the number of rows whose outputs are
    meaningful. Rows beyond the final offset are returned as zero. Empty
    experts produce zero amax values. These value constraints are trusted
    while tracing with :func:`jax.jit`.
    """

    from cutlass import Float32
    from cutlass.jax import jax_to_cutlass_dtype

    from .moe_blockscaled_grouped_gemm_glu_hadamard import BlockScaledMoEGroupedGemmGluHadamardKernel

    kernel = BlockScaledMoEGroupedGemmGluHadamardKernel
    m, n, k, experts, ab_dtype = require_grouped_gemm_inputs(
        a_tensor,
        b_tensor,
        padded_offsets,
        alpha_tensor,
        max_experts=kernel.MAX_EXPERTS,
        valid_ab_dtypes=(jnp.float4_e2m1fn,),
    )
    if m % kernel.FIX_PAD_SIZE:
        raise ValueError(f"M must be divisible by {kernel.FIX_PAD_SIZE}, got {m}")
    if n % 64:
        raise ValueError(f"N must be divisible by 64, got {n}")
    if m_aligned != kernel.FIX_PAD_SIZE:
        raise ValueError(f"m_aligned must be {kernel.FIX_PAD_SIZE}, got {m_aligned}")
    if sf_vec_size not in kernel.SF_VEC_SIZES:
        raise ValueError(f"sf_vec_size must be one of {kernel.SF_VEC_SIZES}, got {sf_vec_size}")
    if act_func not in ("swiglu", "geglu", "srelu"):
        raise ValueError(f"act_func must be 'swiglu', 'geglu', or 'srelu', got {act_func!r}")
    if b_major != "k":
        raise ValueError(f"b_major must be 'k' for native FP4, got {b_major!r}")
    if cd_major != "n":
        raise ValueError(f"cd_major must be 'n', got {cd_major!r}")

    sf_dtype = require_grouped_block_scales(
        sfa_tensor,
        sfb_tensor,
        m=m,
        n=n,
        k=k,
        experts=experts,
        sf_vec_size=sf_vec_size,
        valid_dtypes=(jnp.float8_e8m0fnu, jnp.float8_e4m3fn),
    )
    if sf_dtype == jnp.dtype(jnp.float8_e4m3fn) and sf_vec_size == 32:
        raise ValueError("float8_e4m3fn scales require sf_vec_size=16")
    require_grouped_probability("prob_tensor", prob_tensor, m=m)

    if bias_tensor is not None:
        bias_shape = require_array("bias_tensor", bias_tensor, 2)
        require_shape("bias_tensor", bias_shape, (n, experts))
        require_dtype("bias_tensor.dtype", bias_tensor, (jnp.float16, jnp.bfloat16, jnp.float32))

    acc_dtype = require_dtype("acc_dtype", acc_dtype, (jnp.float32,), default=jnp.float32)
    c_dtype = require_dtype("c_dtype", c_dtype, (jnp.float16, jnp.bfloat16), default=jnp.bfloat16)
    d_dtype = require_dtype("d_dtype", d_dtype, (jnp.float16, jnp.bfloat16), default=jnp.bfloat16)

    mma_tiler_mn = kernel.require_mma_tiler(mma_tiler_mn)
    if cluster_shape_mn is None:
        cluster_shape_mn = (2, 1)
    cluster_shape_mn = kernel.require_cluster_shape(cluster_shape_mn, mma_tiler_mn=mma_tiler_mn)

    output_n = n if act_func == "srelu" else n // 2
    if output_n % kernel.HADAMARD_SIZE:
        raise ValueError(f"D's N dimension must be divisible by {kernel.HADAMARD_SIZE}, got {output_n}")
    require_contiguous_alignment("a_tensor", k, 4)
    require_contiguous_alignment("b_tensor", k, 4)
    require_contiguous_alignment("c_tensor", n, c_dtype.itemsize * 8)
    require_contiguous_alignment("d_tensor", output_n, d_dtype.itemsize * 8)

    if not kernel.can_implement(
        jax_to_cutlass_dtype(ab_dtype),
        jax_to_cutlass_dtype(sf_dtype),
        sf_vec_size,
        Float32,
        jax_to_cutlass_dtype(d_dtype),
        True,
        mma_tiler_mn,
        cluster_shape_mn,
        m,
        n,
        k,
        experts,
        "k",
        "k",
        "n",
        m_aligned,
    ):
        raise ValueError("Unsupported grouped GEMM GLU Hadamard configuration")

    a_spec = gemm_a_tensor_spec("k")
    b_spec = gemm_b_tensor_spec("k")
    output_spec = gemm_c_tensor_spec("n")
    scale_spec = block_scale_tensor_spec()
    inputs = [
        a_tensor,
        b_tensor,
        sfa_tensor,
        sfb_tensor,
        padded_offsets,
        alpha_tensor,
        prob_tensor,
    ]
    input_specs = [
        a_spec,
        b_spec,
        scale_spec,
        scale_spec,
        None,
        None,
        probability_tensor_spec(),
    ]

    has_hadamard = bool(use_tmem_post_rht_amax)
    if has_hadamard:
        if hadamard_tensor is None:
            hadamard_tensor = jnp.asarray(hadamard_values(kernel.HADAMARD_SIZE), dtype=jnp.bfloat16)
        else:
            hadamard_shape = require_array("hadamard_tensor", hadamard_tensor, 2)
            require_shape("hadamard_tensor", hadamard_shape, (kernel.HADAMARD_SIZE, kernel.HADAMARD_SIZE))
            require_dtype("hadamard_tensor.dtype", hadamard_tensor, (jnp.bfloat16,))
        inputs.append(hadamard_tensor)
        input_specs.append(None)
    elif hadamard_tensor is not None:
        raise ValueError("hadamard_tensor is used only when use_tmem_post_rht_amax=True")

    if bias_tensor is not None:
        inputs.append(bias_tensor)
        input_specs.append(grouped_bias_tensor_spec())

    workspace_bytes = max(kernel.get_dense_workspace_bytes(bool(use_dynamic_sched)), 1)
    c_tensor, d_tensor, amax_tensor, post_rht_amax_tensor = call_cutedsl(
        _make_launcher(
            acc_dtype=acc_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sf_vec_size=sf_vec_size,
            vector_f32=bool(vector_f32),
            expert_cnt=experts,
            act_func=act_func,
            has_bias=bias_tensor is not None,
            has_hadamard=has_hadamard,
            use_dynamic_sched=bool(use_dynamic_sched),
            use_tmem_post_rht_amax=bool(use_tmem_post_rht_amax),
            cluster_overlap_margin=int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0")),
        ),
        inputs,
        outputs=(
            BufferSpec("c_tensor", (m, n, 1), c_dtype, tensor_spec=output_spec, fill_value=0),
            BufferSpec("d_tensor", (m, output_n, 1), d_dtype, tensor_spec=output_spec, fill_value=0),
            BufferSpec("amax_tensor", (experts, 1), jnp.float32, fill_value=0.0),
            BufferSpec("post_rht_amax_tensor", (experts, 1), jnp.float32, fill_value=0.0),
        ),
        workspaces=(
            BufferSpec(
                "workspace",
                (workspace_bytes,),
                jnp.uint8,
                tensor_spec=grouped_workspace_tensor_spec(),
                fill_value=0,
            ),
        ),
        input_specs=input_specs,
        use_static_tensors=True,
    )
    return GroupedGemmGluHadamardResult(
        c_tensor=c_tensor,
        d_tensor=d_tensor,
        amax_tensor=amax_tensor,
        post_rht_amax_tensor=post_rht_amax_tensor,
    )


__all__ = ["GroupedGemmGluHadamardResult", "grouped_gemm_glu_hadamard_wrapper_sm100"]
