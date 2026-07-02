# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for dense-weight grouped GEMM + dGLU on SM100."""

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
    require_16_byte_extent,
)
from ..._jax.grouped_gemm import (
    grouped_workspace_tensor_spec,
    require_grouped_fp8_scales,
    require_grouped_gemm_inputs,
    require_grouped_probability,
    require_grouped_vector,
)
from ..._jax.validation import require_dtype
from ...gemm_validation import (
    block_scale_shape,
    require_shape,
    resolve_max_active_clusters,
)


class GroupedGemmDgluResult(NamedTuple):
    """Functional outputs from dense-weight grouped GEMM + dGLU."""

    d_row_tensor: Any
    d_col_tensor: Any
    dprob_tensor: Any
    dbias_tensor: Any | None
    amax_tensor: None
    sfd_row_tensor: Any
    sfd_col_tensor: Any


@lru_cache(maxsize=None)
def _make_launcher(
    *,
    acc_dtype: Any,
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    sf_vec_size: int,
    vector_f32: bool,
    discrete_col_sfd: bool,
    expert_cnt: int,
    generate_dbias: bool,
    use_dynamic_sched: bool,
    act_func: str,
    epilogue_op: str,
    linear_offset: float,
    geglu_alpha: float,
    glu_clamp_max: float,
    glu_clamp_min: float,
    cluster_overlap_margin: int,
):
    def launch(stream, *args):
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.nvgpu import OperandMajorMode
        from cutlass.jax import jax_to_cutlass_dtype

        from ..moe_utils import MoEWeightMode
        from .moe_blockscaled_grouped_gemm_dglu_dbias import (
            BlockScaledMoEGroupedGemmDgluDbiasKernel,
        )

        arg_idx = 0

        def take():
            nonlocal arg_idx
            value = args[arg_idx]
            arg_idx += 1
            return value

        a = take()
        b = take()
        c = take()
        sfa = take()
        sfb = take()
        padded_offsets = take()
        alpha = take()
        beta = take()
        prob = take()
        norm_const = take()

        d_row = take()
        d_col = take()
        dprob = take()
        dbias = take() if generate_dbias else None
        sfd_row = take()
        sfd_col = take()
        workspace = take()
        if arg_idx != len(args):
            raise RuntimeError(f"Unexpected grouped GEMM argument count: consumed {arg_idx}, received {len(args)}")

        if epilogue_op == "relu":

            def epilogue(x):
                return cute.where(x > 0, x, cute.full_like(x, 0))

        elif epilogue_op == "srelu":

            def epilogue(x):
                return cute.where(x > 0, x, cute.full_like(x, 0)) ** 2

        else:

            def epilogue(x):
                return x

        kernel = BlockScaledMoEGroupedGemmDgluDbiasKernel(
            sf_vec_size=sf_vec_size,
            acc_dtype=jax_to_cutlass_dtype(acc_dtype),
            use_2cta_instrs=mma_tiler_mn[0] == BlockScaledMoEGroupedGemmDgluDbiasKernel.TWO_CTA_MMA_TILER_M,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            vectorized_f32=vector_f32,
            discrete_col_sfd=discrete_col_sfd,
            expert_cnt=expert_cnt,
            weight_mode=MoEWeightMode.DENSE,
            use_dynamic_sched=use_dynamic_sched,
            act_func=act_func,
        )
        max_active_clusters = resolve_max_active_clusters(
            cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1]),
            cluster_overlap_margin,
        )
        kernel(
            a,
            b,
            sfb,
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int64(0),
            OperandMajorMode.K,
            workspace.iterator,
            c,
            d_row,
            d_col,
            sfa,
            sfd_row,
            sfd_col,
            None,
            norm_const,
            padded_offsets,
            alpha,
            beta,
            prob,
            dprob,
            dbias,
            max_active_clusters,
            stream,
            epilogue_op=epilogue,
            linear_offset=linear_offset,
            geglu_alpha=geglu_alpha,
            glu_clamp_max=glu_clamp_max,
            glu_clamp_min=glu_clamp_min,
        )

    return launch


def grouped_gemm_dglu_wrapper_sm100(
    a_tensor: Any,
    c_tensor: Any,
    sfa_tensor: Any,
    padded_offsets: Any,
    alpha_tensor: Any,
    beta_tensor: Any,
    prob_tensor: Any,
    b_tensor: Any,
    sfb_tensor: Any,
    generate_dbias: bool = False,
    norm_const_tensor: Optional[Any] = None,
    acc_dtype: Any = None,
    d_dtype: Any = None,
    cd_major: str = "n",
    mma_tiler_mn: tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[tuple[int, int]] = None,
    sf_vec_size: int = 32,
    vector_f32: bool = False,
    m_aligned: int = 256,
    discrete_col_sfd: bool = False,
    act_func: str = "dswiglu",
    linear_offset: Optional[float] = None,
    geglu_alpha: float = 1.702,
    glu_clamp_max: float = 7.0,
    glu_clamp_min: float = -7.0,
    epilogue_op: Optional[str] = None,
    use_dynamic_sched: bool = False,
    *,
    b_major: str = "k",
) -> GroupedGemmDgluResult:
    """Compute an MXFP8 dense-weight grouped GEMM with fused dGLU.

    ``dprob_tensor`` and optional ``dbias_tensor`` are fresh, zero-initialized
    JAX results. The FP8 output and its E8M0 scale factors are public results;
    temporary scheduler storage is owned by XLA.
    """

    from .moe_blockscaled_grouped_gemm_dglu_dbias import (
        BlockScaledMoEGroupedGemmDgluDbiasKernel,
    )

    kernel = BlockScaledMoEGroupedGemmDgluDbiasKernel
    m, n, k, experts, ab_dtype = require_grouped_gemm_inputs(
        a_tensor,
        b_tensor,
        padded_offsets,
        alpha_tensor,
        max_experts=kernel.MAX_EXPERTS,
    )
    if n % 32:
        raise ValueError(f"b_tensor N must be divisible by 32 for dGLU, got {n}")
    if sf_vec_size != kernel.FP8_SF_VEC_SIZE:
        raise ValueError(f"FP8 grouped GEMM requires sf_vec_size={kernel.FP8_SF_VEC_SIZE}, got {sf_vec_size}")
    if m_aligned != kernel.FIX_PAD_SIZE:
        raise ValueError(f"m_aligned must be {kernel.FIX_PAD_SIZE}, got {m_aligned}")
    require_grouped_fp8_scales(
        sfa_tensor,
        sfb_tensor,
        m=m,
        n=n,
        k=k,
        experts=experts,
        sf_vec_size=sf_vec_size,
    )

    output_n = 2 * n
    require_shape("c_tensor", tuple(getattr(c_tensor, "shape", ())), (m, output_n, 1))
    c_dtype = require_dtype(
        "c_tensor.dtype",
        c_tensor,
        (jnp.float32, jnp.float16, jnp.bfloat16, jnp.float8_e4m3fn, jnp.float8_e5m2),
    )
    fp8_dtypes = {
        jnp.dtype(jnp.float8_e4m3fn),
        jnp.dtype(jnp.float8_e5m2),
    }
    if vector_f32 and c_dtype in fp8_dtypes:
        raise ValueError("vector_f32 does not support an FP8 c_tensor")
    require_grouped_probability("prob_tensor", prob_tensor, m=m)
    require_grouped_vector("beta_tensor", beta_tensor, length=experts)
    if norm_const_tensor is None:
        raise ValueError("norm_const_tensor is required for an FP8 output")
    require_grouped_vector("norm_const_tensor", norm_const_tensor, length=1)

    acc_dtype = require_dtype("acc_dtype", acc_dtype, (jnp.float32,), default=jnp.float32)
    d_dtype = require_dtype(
        "d_dtype",
        d_dtype,
        (jnp.float8_e4m3fn, jnp.float8_e5m2),
        default=ab_dtype,
    )
    if cd_major != "n":
        raise ValueError(f"cd_major must be 'n', got {cd_major!r}")
    if act_func not in ("dswiglu", "dgeglu"):
        raise ValueError(f"act_func must be 'dswiglu' or 'dgeglu', got {act_func!r}")
    if linear_offset is None:
        linear_offset = 1.0 if act_func == "dgeglu" else 0.0
    normalized_epilogue = "identity" if epilogue_op in (None, "none", "identity") else epilogue_op
    if normalized_epilogue not in ("identity", "relu", "srelu"):
        raise ValueError(f"epilogue_op must be None, 'none', 'identity', 'relu', or 'srelu', got {epilogue_op!r}")

    mma_tiler_mn = kernel.require_mma_tiler(mma_tiler_mn)
    if cluster_shape_mn is None:
        cluster_shape_mn = (2, 1) if mma_tiler_mn[0] == kernel.TWO_CTA_MMA_TILER_M else (1, 1)
    cluster_shape_mn = kernel.require_cluster_shape(cluster_shape_mn, mma_tiler_mn=mma_tiler_mn)

    a_spec = gemm_a_tensor_spec("k")
    b_spec = gemm_b_tensor_spec(b_major)
    output_spec = gemm_c_tensor_spec("n")
    scale_spec = block_scale_tensor_spec()
    require_16_byte_extent("a_tensor", k, ab_dtype)
    require_16_byte_extent("b_tensor", n if b_major == "n" else k, ab_dtype)
    require_16_byte_extent("c_tensor", output_n, c_dtype)
    require_16_byte_extent("d_tensor", output_n, d_dtype)

    outputs = [
        BufferSpec("d_row_tensor", (m, output_n, 1), d_dtype, tensor_spec=output_spec),
        BufferSpec("d_col_tensor", (m, output_n, 1), d_dtype, tensor_spec=output_spec),
        BufferSpec("dprob_tensor", (m, 1, 1), jnp.float32, fill_value=0.0),
    ]
    if generate_dbias:
        outputs.append(
            BufferSpec(
                "dbias_tensor",
                (experts, output_n, 1),
                jnp.bfloat16,
                fill_value=0.0,
            )
        )
    outputs.extend(
        (
            BufferSpec(
                "sfd_row_tensor",
                block_scale_shape(m, output_n, 1, sf_vec_size),
                jnp.float8_e8m0fnu,
                tensor_spec=scale_spec,
            ),
            BufferSpec(
                "sfd_col_tensor",
                block_scale_shape(output_n, m, 1, sf_vec_size),
                jnp.float8_e8m0fnu,
                tensor_spec=scale_spec,
            ),
        )
    )

    workspace_bytes = max(kernel.get_dense_workspace_bytes(bool(use_dynamic_sched)), 1)
    results = call_cutedsl(
        _make_launcher(
            acc_dtype=acc_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sf_vec_size=sf_vec_size,
            vector_f32=bool(vector_f32),
            discrete_col_sfd=bool(discrete_col_sfd),
            expert_cnt=experts,
            generate_dbias=bool(generate_dbias),
            use_dynamic_sched=bool(use_dynamic_sched),
            act_func=act_func,
            epilogue_op=normalized_epilogue,
            linear_offset=float(linear_offset),
            geglu_alpha=float(geglu_alpha),
            glu_clamp_max=float(glu_clamp_max),
            glu_clamp_min=float(glu_clamp_min),
            cluster_overlap_margin=int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0")),
        ),
        (
            a_tensor,
            b_tensor,
            c_tensor,
            sfa_tensor,
            sfb_tensor,
            padded_offsets,
            alpha_tensor,
            beta_tensor,
            prob_tensor,
            norm_const_tensor,
        ),
        outputs=outputs,
        workspaces=(
            BufferSpec(
                "workspace",
                (workspace_bytes,),
                jnp.uint8,
                tensor_spec=grouped_workspace_tensor_spec(),
            ),
        ),
        input_specs=(
            a_spec,
            b_spec,
            output_spec,
            scale_spec,
            scale_spec,
            None,
            None,
            None,
            probability_tensor_spec(),
            None,
        ),
        use_static_tensors=True,
    )
    result_idx = 0
    d_row_tensor = results[result_idx]
    result_idx += 1
    d_col_tensor = results[result_idx]
    result_idx += 1
    dprob_tensor = results[result_idx]
    result_idx += 1
    dbias_tensor = results[result_idx] if generate_dbias else None
    result_idx += int(bool(generate_dbias))
    sfd_row_tensor = results[result_idx]
    sfd_col_tensor = results[result_idx + 1]
    return GroupedGemmDgluResult(
        d_row_tensor=d_row_tensor,
        d_col_tensor=d_col_tensor,
        dprob_tensor=dprob_tensor,
        dbias_tensor=dbias_tensor,
        amax_tensor=None,
        sfd_row_tensor=sfd_row_tensor,
        sfd_col_tensor=sfd_col_tensor,
    )


__all__ = ["GroupedGemmDgluResult", "grouped_gemm_dglu_wrapper_sm100"]
