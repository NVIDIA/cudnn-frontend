# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for contiguous grouped GEMM + SwiGLU on SM100."""

from __future__ import annotations

from functools import partial
import os
from typing import Any, Optional

import jax
import jax.numpy as jnp

from .._jax_api import (
    ApiBaseJax,
    BLOCK_SCALE_MODE,
    make_buffer_desc,
    FIX_PAD_SIZE,
    MAX_EXPERTS,
    TWO_CTA_MMA_TILER_M,
    TupleDict,
    PROBABILITY_MODE,
    as_gemm_tensor_desc,
    block_scale_shape,
    call_cutedsl,
    gemm_a_mode,
    gemm_b_mode,
    gemm_output_mode,
    is_fp4_dtype,
    is_fp8_dtype,
    require_16_byte_extent,
    require_dtype,
    require_grouped_cluster_shape,
    require_grouped_gemm_inputs,
    require_grouped_input_scales,
    require_grouped_mma_tiler,
    require_grouped_probability,
    require_grouped_vector,
    require_layout,
)


def _launch(
    stream,
    *args,
    acc_dtype: Any,
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    sf_vec_size: int,
    vector_f32: bool,
    discrete_col_sfd: bool,
    expert_cnt: int,
    has_prob: bool,
    generate_sfd: bool,
    has_amax: bool,
    max_active_clusters: int,
):
    from cutlass.jax import jax_to_cutlass_dtype

    from .grouped_gemm_swiglu_quant import BlockScaledContiguousGroupedGemmKernel

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
    norm_const = take() if generate_sfd else None
    prob = take() if has_prob else None
    c = take()
    d = take()
    d_col = take()
    amax = take() if has_amax else None
    sfd_row = take() if generate_sfd else None
    sfd_col = take() if generate_sfd else None
    if arg_idx != len(args):
        raise RuntimeError(
            f"Unexpected grouped GEMM argument count: consumed {arg_idx}, received {len(args)}"
        )

    kernel = BlockScaledContiguousGroupedGemmKernel(
        sf_vec_size=sf_vec_size,
        acc_dtype=jax_to_cutlass_dtype(acc_dtype),
        use_2cta_instrs=mma_tiler_mn[0] == TWO_CTA_MMA_TILER_M,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        vector_f32=vector_f32,
        generate_sfd=generate_sfd,
        discrete_col_sfd=discrete_col_sfd,
        expert_cnt=expert_cnt,
        use_mono_increase_expert_idx=True,
    )
    kernel(
        a,
        b,
        c,
        d,
        d_col,
        sfa,
        sfb,
        sfd_row,
        sfd_col,
        amax,
        norm_const,
        padded_offsets,
        alpha,
        prob,
        max_active_clusters,
        stream,
    )


def _grouped_gemm_swiglu_impl(
    a_tensor: Any,
    b_tensor: Any,
    sfa_tensor: Any,
    sfb_tensor: Any,
    padded_offsets: Any,
    alpha_tensor: Any,
    norm_const_tensor: Optional[Any] = None,
    prob_tensor: Optional[Any] = None,
    acc_dtype: Any = None,
    c_dtype: Any = None,
    d_dtype: Any = None,
    output_layout: str = "LMN",
    mma_tiler_mn: tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[tuple[int, int]] = None,
    sf_vec_size: int = 32,
    vector_f32: bool = False,
    m_aligned: int = 256,
    discrete_col_sfd: bool = False,
    cluster_overlap_margin: int = 0,
    *,
    _validate_only: bool = False,
) -> TupleDict | dict[str, Any]:
    """Compute a contiguous block-scaled grouped GEMM and fused SwiGLU.

    ``a_tensor`` has public shape ``(1, M, K)`` and contains all padded expert rows.
    ``b_tensor`` has public shape ``(L, N, K)``, where ``L`` is the expert count.
    ``padded_offsets`` supplies each expert's cumulative padded row boundary at
    runtime. Matrix outputs use public ``LMN`` order.

    FP4 inputs use JAX's native logical dtype; raw packed ``uint8`` storage is
    deliberately excluded. Configuration values are static under JAX tracing.
    """

    output_layout = require_layout("output_layout", output_layout, ("LMN",))
    a_mode = gemm_a_mode("LMK")
    b_mode = gemm_b_mode("LNK")
    output_mode = gemm_output_mode(output_layout, name="output_layout")
    a_desc = as_gemm_tensor_desc("a_tensor", a_tensor, mode=a_mode)
    b_desc = as_gemm_tensor_desc("b_tensor", b_tensor, mode=b_mode)
    m, n, k, experts, ab_dtype = require_grouped_gemm_inputs(
        a_desc,
        b_desc,
        padded_offsets,
        alpha_tensor,
        max_experts=MAX_EXPERTS,
    )
    if n % 64:
        raise ValueError(f"b_tensor N must be divisible by 64 for SwiGLU, got {n}")
    if m_aligned != FIX_PAD_SIZE:
        raise ValueError(f"m_aligned must be {FIX_PAD_SIZE}, got {m_aligned}")
    require_grouped_input_scales(
        sfa_tensor,
        sfb_tensor,
        m=m,
        n=n,
        k=k,
        experts=experts,
        sf_vec_size=sf_vec_size,
        ab_dtype=ab_dtype,
    )
    if prob_tensor is not None:
        require_grouped_probability("prob_tensor", prob_tensor, m=m)

    acc_dtype = require_dtype(
        acc_dtype, (jnp.float32,), name="acc_dtype", default=jnp.float32
    )
    c_dtype = require_dtype(
        c_dtype,
        (
            jnp.float32,
            jnp.float16,
            jnp.bfloat16,
            jnp.float8_e4m3fn,
            jnp.float8_e5m2,
            jnp.float4_e2m1fn,
        ),
        name="c_dtype",
        default=jnp.bfloat16,
    )
    if is_fp4_dtype(ab_dtype):
        valid_d_dtypes = (jnp.float16, jnp.bfloat16, jnp.float32)
    else:
        valid_d_dtypes = (
            jnp.float16,
            jnp.bfloat16,
            jnp.float8_e4m3fn,
            jnp.float8_e5m2,
            jnp.float4_e2m1fn,
        )
    d_dtype = require_dtype(
        d_dtype, valid_d_dtypes, name="d_dtype", default=jnp.bfloat16
    )
    if (
        is_fp4_dtype(ab_dtype)
        and sf_vec_size == 16
        and d_dtype == jnp.dtype(jnp.float32)
    ):
        raise NotImplementedError(
            "FP4 with sf_vec_size=16 does not support a float32 D output"
        )
    generate_sfd = is_fp8_dtype(ab_dtype)
    if generate_sfd:
        if norm_const_tensor is None:
            raise ValueError("norm_const_tensor is required for FP8 inputs")
        require_grouped_vector("norm_const_tensor", norm_const_tensor, length=1)
    else:
        norm_const_tensor = None
        discrete_col_sfd = False
    if vector_f32 and c_dtype in {
        jnp.dtype(jnp.float8_e4m3fn),
        jnp.dtype(jnp.float8_e5m2),
    }:
        raise ValueError("vector_f32 does not support an FP8 c_dtype")
    mma_tiler_mn = require_grouped_mma_tiler(mma_tiler_mn)
    if cluster_shape_mn is None:
        cluster_shape_mn = (2, 1) if mma_tiler_mn[0] == TWO_CTA_MMA_TILER_M else (1, 1)
    cluster_shape_mn = require_grouped_cluster_shape(
        cluster_shape_mn, mma_tiler_mn=mma_tiler_mn
    )
    if mma_tiler_mn[1] == 128 and d_dtype in {
        jnp.dtype(jnp.float8_e4m3fn),
        jnp.dtype(jnp.float8_e5m2),
    }:
        raise NotImplementedError("FP8 output requires mma_tiler_mn[1] == 256")

    output_n = n // 2
    require_16_byte_extent("a_tensor", k, ab_dtype)
    require_16_byte_extent("b_tensor", k, ab_dtype)
    require_16_byte_extent("c_tensor", n, c_dtype)
    require_16_byte_extent("d_tensor", output_n, d_dtype)

    if _validate_only:
        return {
            "acc_dtype": acc_dtype,
            "c_dtype": c_dtype,
            "d_dtype": d_dtype,
            "mma_tiler_mn": mma_tiler_mn,
            "cluster_shape_mn": cluster_shape_mn,
        }

    inputs = [
        a_tensor,
        b_tensor,
        sfa_tensor,
        sfb_tensor,
        padded_offsets,
        alpha_tensor,
    ]
    input_descs = [
        a_desc,
        b_desc,
        as_gemm_tensor_desc("sfa_tensor", sfa_tensor, mode=BLOCK_SCALE_MODE),
        as_gemm_tensor_desc("sfb_tensor", sfb_tensor, mode=BLOCK_SCALE_MODE),
        as_gemm_tensor_desc("padded_offsets", padded_offsets),
        as_gemm_tensor_desc("alpha_tensor", alpha_tensor),
    ]
    if generate_sfd:
        inputs.append(norm_const_tensor)
        input_descs.append(as_gemm_tensor_desc("norm_const_tensor", norm_const_tensor))
    if prob_tensor is not None:
        inputs.append(prob_tensor)
        input_descs.append(
            as_gemm_tensor_desc("prob_tensor", prob_tensor, mode=PROBABILITY_MODE)
        )

    has_amax = d_dtype in {jnp.dtype(jnp.float16), jnp.dtype(jnp.bfloat16)}
    outputs = [
        make_buffer_desc("c_tensor", (1, m, n), c_dtype, mode=output_mode),
        make_buffer_desc(
            "d_tensor", (1, m, output_n), d_dtype, mode=output_mode
        ),
        make_buffer_desc(
            "d_col_tensor", (1, m, output_n), d_dtype, mode=output_mode
        ),
    ]
    if has_amax:
        outputs.append(
            make_buffer_desc(
                "amax_tensor", (experts, 1), jnp.float32, init_value=-float("inf")
            )
        )
    if generate_sfd:
        outputs.extend(
            (
                make_buffer_desc(
                    "sfd_row_tensor",
                    block_scale_shape(m, output_n, 1, sf_vec_size),
                    jnp.float8_e8m0fnu,
                    mode=BLOCK_SCALE_MODE,
                ),
                make_buffer_desc(
                    "sfd_col_tensor",
                    block_scale_shape(output_n, m, 1, sf_vec_size),
                    jnp.float8_e8m0fnu,
                    mode=BLOCK_SCALE_MODE,
                ),
            )
        )
    results = call_cutedsl(
        _launch,
        inputs,
        input_descs=input_descs,
        static_args={
            "acc_dtype": acc_dtype,
            "mma_tiler_mn": mma_tiler_mn,
            "cluster_shape_mn": cluster_shape_mn,
            "sf_vec_size": sf_vec_size,
            "vector_f32": bool(vector_f32),
            "discrete_col_sfd": bool(discrete_col_sfd),
            "expert_cnt": experts,
            "has_prob": prob_tensor is not None,
            "generate_sfd": generate_sfd,
            "has_amax": has_amax,
            "cluster_overlap_margin": int(cluster_overlap_margin),
        },
        outputs=outputs,
    )
    result_idx = 0
    c_tensor = results[result_idx]
    result_idx += 1
    d_tensor = results[result_idx]
    result_idx += 1
    d_col_tensor = results[result_idx]
    result_idx += 1
    amax_tensor = results[result_idx] if has_amax else None
    result_idx += int(has_amax)
    sfd_row_tensor = results[result_idx] if generate_sfd else None
    sfd_col_tensor = results[result_idx + 1] if generate_sfd else None
    return TupleDict(
        c_tensor=c_tensor,
        d_tensor=d_tensor,
        d_col_tensor=d_col_tensor,
        amax_tensor=amax_tensor,
        sfd_row_tensor=sfd_row_tensor,
        sfd_col_tensor=sfd_col_tensor,
    )


class GroupedGemmSwigluSm100(ApiBaseJax):
    """Sample-signature-bound JAX callable for grouped GEMM + SwiGLU."""

    def __init__(
        self,
        sample_a_tensor: Any,
        sample_b_tensor: Any,
        sample_sfa_tensor: Any,
        sample_sfb_tensor: Any,
        sample_padded_offsets: Any,
        sample_alpha_tensor: Any,
        sample_norm_const_tensor: Optional[Any] = None,
        sample_prob_tensor: Optional[Any] = None,
        acc_dtype: Any = None,
        c_dtype: Any = None,
        d_dtype: Any = None,
        output_layout: str = "LMN",
        mma_tiler_mn: tuple[int, int] = (256, 256),
        cluster_shape_mn: Optional[tuple[int, int]] = None,
        sf_vec_size: int = 32,
        vector_f32: bool = False,
        m_aligned: int = 256,
        discrete_col_sfd: bool = False,
    ) -> None:
        super().__init__()
        output_layout = require_layout("output_layout", output_layout, ("LMN",))
        a_mode = gemm_a_mode("LMK")
        b_mode = gemm_b_mode("LNK")
        self._sample_descs = {
            "a_tensor": self.make_tensor_desc(
                sample_a_tensor, mode=a_mode, name="sample_a_tensor"
            ),
            "b_tensor": self.make_tensor_desc(
                sample_b_tensor, mode=b_mode, name="sample_b_tensor"
            ),
            "sfa_tensor": self.make_tensor_desc(
                sample_sfa_tensor, mode=BLOCK_SCALE_MODE, name="sample_sfa_tensor"
            ),
            "sfb_tensor": self.make_tensor_desc(
                sample_sfb_tensor, mode=BLOCK_SCALE_MODE, name="sample_sfb_tensor"
            ),
            "padded_offsets": self.make_tensor_desc(
                sample_padded_offsets, name="sample_padded_offsets"
            ),
            "alpha_tensor": self.make_tensor_desc(
                sample_alpha_tensor, name="sample_alpha_tensor"
            ),
            "norm_const_tensor": self.make_optional_tensor_desc(
                sample_norm_const_tensor, name="sample_norm_const_tensor"
            ),
            "prob_tensor": self.make_optional_tensor_desc(
                sample_prob_tensor,
                mode=PROBABILITY_MODE,
                name="sample_prob_tensor",
            ),
        }
        self._config = {
            "acc_dtype": self.as_optional_dtype(acc_dtype),
            "c_dtype": self.as_optional_dtype(c_dtype),
            "d_dtype": self.as_optional_dtype(d_dtype),
            "output_layout": output_layout,
            "mma_tiler_mn": tuple(mma_tiler_mn),
            "cluster_shape_mn": (
                None if cluster_shape_mn is None else tuple(cluster_shape_mn)
            ),
            "sf_vec_size": sf_vec_size,
            "vector_f32": vector_f32,
            "m_aligned": m_aligned,
            "discrete_col_sfd": discrete_col_sfd,
            "cluster_overlap_margin": int(
                os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0")
            ),
        }

        self._sample_descs = self.freeze_mapping(self._sample_descs)
        self._config = self.freeze_mapping(self._config)

    def _check_support(self) -> None:
        resolved = _grouped_gemm_swiglu_impl(
            self._sample_descs["a_tensor"],
            self._sample_descs["b_tensor"],
            self._sample_descs["sfa_tensor"],
            self._sample_descs["sfb_tensor"],
            self._sample_descs["padded_offsets"],
            self._sample_descs["alpha_tensor"],
            self._sample_descs["norm_const_tensor"],
            self._sample_descs["prob_tensor"],
            **self._config,
            _validate_only=True,
        )
        self._config = self.freeze_mapping({**self._config, **resolved})

    def __call__(
        self,
        a_tensor: Any,
        b_tensor: Any,
        sfa_tensor: Any,
        sfb_tensor: Any,
        padded_offsets: Any,
        alpha_tensor: Any,
        norm_const_tensor: Optional[Any] = None,
        prob_tensor: Optional[Any] = None,
    ) -> TupleDict:
        return super().__call__(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            padded_offsets,
            alpha_tensor,
            norm_const_tensor,
            prob_tensor,
        )

    def _call_impl(
        self,
        a_tensor: Any,
        b_tensor: Any,
        sfa_tensor: Any,
        sfb_tensor: Any,
        padded_offsets: Any,
        alpha_tensor: Any,
        norm_const_tensor: Optional[Any] = None,
        prob_tensor: Optional[Any] = None,
    ) -> TupleDict:
        values = {
            "a_tensor": a_tensor,
            "b_tensor": b_tensor,
            "sfa_tensor": sfa_tensor,
            "sfb_tensor": sfb_tensor,
            "padded_offsets": padded_offsets,
            "alpha_tensor": alpha_tensor,
            "norm_const_tensor": norm_const_tensor,
            "prob_tensor": prob_tensor,
        }
        self.check_tensor_signatures(self._sample_descs, values)
        return _grouped_gemm_swiglu_impl(**values, **self._config)


@partial(
    jax.jit,
    static_argnames=(
        "acc_dtype",
        "c_dtype",
        "d_dtype",
        "output_layout",
        "mma_tiler_mn",
        "cluster_shape_mn",
        "sf_vec_size",
        "vector_f32",
        "m_aligned",
        "discrete_col_sfd",
    ),
)
def grouped_gemm_swiglu_wrapper_sm100(
    a_tensor: Any,
    b_tensor: Any,
    sfa_tensor: Any,
    sfb_tensor: Any,
    padded_offsets: Any,
    alpha_tensor: Any,
    norm_const_tensor: Optional[Any] = None,
    prob_tensor: Optional[Any] = None,
    acc_dtype: Any = None,
    c_dtype: Any = None,
    d_dtype: Any = None,
    output_layout: str = "LMN",
    mma_tiler_mn: tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[tuple[int, int]] = None,
    sf_vec_size: int = 32,
    vector_f32: bool = False,
    m_aligned: int = 256,
    discrete_col_sfd: bool = False,
) -> TupleDict:
    """Compute a contiguous block-scaled grouped GEMM and fused SwiGLU."""

    return GroupedGemmSwigluSm100(
        a_tensor,
        b_tensor,
        sfa_tensor,
        sfb_tensor,
        padded_offsets,
        alpha_tensor,
        norm_const_tensor,
        prob_tensor,
        acc_dtype=acc_dtype,
        c_dtype=c_dtype,
        d_dtype=d_dtype,
        output_layout=output_layout,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        vector_f32=vector_f32,
        m_aligned=m_aligned,
        discrete_col_sfd=discrete_col_sfd,
    )(
        a_tensor,
        b_tensor,
        sfa_tensor,
        sfb_tensor,
        padded_offsets,
        alpha_tensor,
        norm_const_tensor,
        prob_tensor,
    )


__all__ = [
    "GroupedGemmSwigluSm100",
    "grouped_gemm_swiglu_wrapper_sm100",
]
