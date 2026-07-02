# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for dense-weight grouped GEMM + dSReLU on SM100."""

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


class GroupedGemmDsreluResult(NamedTuple):
    """Functional outputs from dense-weight grouped GEMM + dSReLU."""

    d_row_tensor: Any
    d_col_tensor: Any
    d_srelu_tensor: Any
    dprob_tensor: Any
    dbias_tensor: Any | None
    amax_tensor: None
    sfd_row_tensor: Any
    sfd_col_tensor: Any
    sfd_col_d_srelu_tensor: Any


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
    use_dsrelu_reuse: bool,
    cluster_overlap_margin: int,
):
    def launch(stream, *args):
        import cutlass
        from cutlass.cute.nvgpu import OperandMajorMode
        from cutlass.jax import jax_to_cutlass_dtype

        from ..moe_utils import MoEWeightMode
        from .moe_blockscaled_grouped_gemm_dsrelu_quant import (
            BlockScaledMoEGroupedGemmQuantBwdKernel,
            EpilogueType,
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
        prob = take()
        norm_const = take()

        d_row = take()
        d_col = take()
        d_srelu = take()
        dprob = take()
        dbias = take() if generate_dbias else None
        sfd_row = take()
        sfd_col = take()
        sfd_col_d_srelu = take()
        workspace = take()
        if arg_idx != len(args):
            raise RuntimeError(f"Unexpected grouped GEMM argument count: consumed {arg_idx}, received {len(args)}")

        kernel = BlockScaledMoEGroupedGemmQuantBwdKernel(
            sf_vec_size=sf_vec_size,
            acc_dtype=jax_to_cutlass_dtype(acc_dtype),
            use_2cta_instrs=mma_tiler_mn[0] == BlockScaledMoEGroupedGemmQuantBwdKernel.TWO_CTA_MMA_TILER_M,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            vectorized_f32=vector_f32,
            generate_sfd=True,
            discrete_col_sfd=discrete_col_sfd,
            expert_cnt=expert_cnt,
            weight_mode=MoEWeightMode.DENSE,
            use_dynamic_sched=use_dynamic_sched,
            epilogue_type=EpilogueType.DSRELU.value,
            generate_dbias=generate_dbias,
            generate_d_srelu=True,
            use_dsrelu_reuse=use_dsrelu_reuse,
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
            prob,
            dprob,
            dbias,
            d_srelu,
            sfd_col_d_srelu,
            max_active_clusters,
            stream,
        )

    return launch


def grouped_gemm_dsrelu_wrapper_sm100(
    a_tensor: Any,
    b_tensor: Any,
    c_tensor: Any,
    sfa_tensor: Any,
    sfb_tensor: Any,
    padded_offsets: Any,
    alpha_tensor: Any,
    prob_tensor: Any,
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
    use_dynamic_sched: bool = False,
    use_dsrelu_reuse: bool = False,
    *,
    b_major: str = "k",
) -> GroupedGemmDsreluResult:
    """Compute an MXFP8 dense-weight grouped GEMM with fused dSReLU.

    The mutable Torch outputs are represented as functional JAX results.
    ``dprob_tensor`` and optional ``dbias_tensor`` start at zero before the
    kernel's atomic reductions. Temporary scheduler storage is owned by XLA.
    """

    from .moe_blockscaled_grouped_gemm_dsrelu_quant import (
        BlockScaledMoEGroupedGemmQuantBwdKernel,
    )

    kernel = BlockScaledMoEGroupedGemmQuantBwdKernel
    m, n, k, experts, ab_dtype = require_grouped_gemm_inputs(
        a_tensor,
        b_tensor,
        padded_offsets,
        alpha_tensor,
        max_experts=kernel.MAX_EXPERTS,
    )
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

    require_shape("c_tensor", tuple(getattr(c_tensor, "shape", ())), (m, n, 1))
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
    require_16_byte_extent("c_tensor", n, c_dtype)
    require_16_byte_extent("d_tensor", n, d_dtype)

    outputs = [
        BufferSpec("d_row_tensor", (m, n, 1), d_dtype, tensor_spec=output_spec),
        BufferSpec("d_col_tensor", (m, n, 1), d_dtype, tensor_spec=output_spec),
        BufferSpec("d_srelu_tensor", (m, n, 1), d_dtype, tensor_spec=output_spec),
        BufferSpec(
            "dprob_tensor",
            (m, 1, 1),
            jnp.float32,
            tensor_spec=probability_tensor_spec(),
            fill_value=0.0,
        ),
    ]
    if generate_dbias:
        outputs.append(
            BufferSpec(
                "dbias_tensor",
                (experts, n, 1),
                jnp.bfloat16,
                fill_value=0.0,
            )
        )
    outputs.extend(
        (
            BufferSpec(
                "sfd_row_tensor",
                block_scale_shape(m, n, 1, sf_vec_size),
                jnp.float8_e8m0fnu,
                tensor_spec=scale_spec,
            ),
            BufferSpec(
                "sfd_col_tensor",
                block_scale_shape(n, m, 1, sf_vec_size),
                jnp.float8_e8m0fnu,
                tensor_spec=scale_spec,
            ),
            BufferSpec(
                "sfd_col_d_srelu_tensor",
                block_scale_shape(n, m, 1, sf_vec_size),
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
            use_dsrelu_reuse=bool(use_dsrelu_reuse),
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
    d_srelu_tensor = results[result_idx]
    result_idx += 1
    dprob_tensor = results[result_idx]
    result_idx += 1
    dbias_tensor = results[result_idx] if generate_dbias else None
    result_idx += int(bool(generate_dbias))
    sfd_row_tensor = results[result_idx]
    sfd_col_tensor = results[result_idx + 1]
    sfd_col_d_srelu_tensor = results[result_idx + 2]
    return GroupedGemmDsreluResult(
        d_row_tensor=d_row_tensor,
        d_col_tensor=d_col_tensor,
        d_srelu_tensor=d_srelu_tensor,
        dprob_tensor=dprob_tensor,
        dbias_tensor=dbias_tensor,
        amax_tensor=None,
        sfd_row_tensor=sfd_row_tensor,
        sfd_col_tensor=sfd_col_tensor,
        sfd_col_d_srelu_tensor=sfd_col_d_srelu_tensor,
    )


__all__ = ["GroupedGemmDsreluResult", "grouped_gemm_dsrelu_wrapper_sm100"]
