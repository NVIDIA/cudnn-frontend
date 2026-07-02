# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for contiguous grouped GEMM + SwiGLU on SM100."""

from __future__ import annotations

from functools import lru_cache
import os
from typing import Any, NamedTuple, Optional

import jax.numpy as jnp

from ..._jax.api_base import ApiBaseJax, BufferSpec, call_cutedsl
from ..._jax.gemm import (
    block_scale_tensor_spec,
    gemm_a_tensor_spec,
    gemm_b_tensor_spec,
    gemm_c_tensor_spec,
    probability_tensor_spec,
    require_16_byte_extent,
)
from ..._jax.grouped_gemm import (
    require_grouped_fp8_scales,
    require_grouped_gemm_inputs,
    require_grouped_probability,
    require_grouped_vector,
)
from ..._jax.validation import require_dtype
from ...gemm_validation import (
    block_scale_shape,
    resolve_max_active_clusters,
)
from .._jax_api import check_call_signatures, immutable_mapping


class GroupedGemmSwigluResult(NamedTuple):
    """Functional outputs from contiguous grouped GEMM + SwiGLU."""

    c_tensor: Any
    d_tensor: Any
    d_col_tensor: Any
    amax_tensor: Any | None
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
    has_prob: bool,
    has_amax: bool,
    cluster_overlap_margin: int,
):
    def launch(stream, *args):
        import cutlass
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
        norm_const = take()
        prob = take() if has_prob else None
        c = take()
        d = take()
        d_col = take()
        amax = take() if has_amax else None
        sfd_row = take()
        sfd_col = take()
        if arg_idx != len(args):
            raise RuntimeError(f"Unexpected grouped GEMM argument count: consumed {arg_idx}, received {len(args)}")

        kernel = BlockScaledContiguousGroupedGemmKernel(
            sf_vec_size=sf_vec_size,
            acc_dtype=jax_to_cutlass_dtype(acc_dtype),
            use_2cta_instrs=mma_tiler_mn[0] == BlockScaledContiguousGroupedGemmKernel.TWO_CTA_MMA_TILER_M,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            vector_f32=vector_f32,
            generate_sfd=True,
            discrete_col_sfd=discrete_col_sfd,
            expert_cnt=expert_cnt,
            use_mono_increase_expert_idx=True,
        )
        max_active_clusters = resolve_max_active_clusters(
            cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1]),
            cluster_overlap_margin,
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

    return launch


def _grouped_gemm_swiglu_impl(
    a_tensor: Any,
    b_tensor: Any,
    sfa_tensor: Any,
    sfb_tensor: Any,
    padded_offsets: Any,
    alpha_tensor: Any,
    norm_const_tensor: Any,
    prob_tensor: Optional[Any] = None,
    acc_dtype: Any = None,
    c_dtype: Any = None,
    d_dtype: Any = None,
    cd_major: str = "n",
    mma_tiler_mn: tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[tuple[int, int]] = None,
    sf_vec_size: int = 32,
    vector_f32: bool = False,
    m_aligned: int = 256,
    discrete_col_sfd: bool = False,
    *,
    _validate_only: bool = False,
) -> GroupedGemmSwigluResult | dict[str, Any]:
    """Compute MXFP8 contiguous grouped GEMM and fused SwiGLU.

    ``a_tensor`` has shape ``(M, K, 1)`` and contains all padded expert rows.
    ``b_tensor`` has shape ``(N, K, L)``, where ``L`` is the expert count.
    ``padded_offsets`` supplies each expert's cumulative padded row boundary at
    runtime. Inputs and outputs use the same physical layouts as the Torch API.

    This binding supports FP8 inputs with E8M0 scales and ``sf_vec_size=32``.
    Configuration values are static when traced by :func:`jax.jit`.
    """

    from .grouped_gemm_swiglu_quant import BlockScaledContiguousGroupedGemmKernel

    kernel = BlockScaledContiguousGroupedGemmKernel
    m, n, k, experts, ab_dtype = require_grouped_gemm_inputs(
        a_tensor,
        b_tensor,
        padded_offsets,
        alpha_tensor,
        max_experts=kernel.MAX_EXPERTS,
    )
    if n % 64:
        raise ValueError(f"b_tensor N must be divisible by 64 for SwiGLU, got {n}")
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
    require_grouped_vector("norm_const_tensor", norm_const_tensor, length=1)
    if prob_tensor is not None:
        require_grouped_probability("prob_tensor", prob_tensor, m=m)

    acc_dtype = require_dtype("acc_dtype", acc_dtype, (jnp.float32,), default=jnp.float32)
    c_dtype = require_dtype(
        "c_dtype",
        c_dtype,
        (jnp.float32, jnp.float16, jnp.bfloat16, jnp.float8_e4m3fn, jnp.float8_e5m2),
        default=jnp.bfloat16,
    )
    d_dtype = require_dtype(
        "d_dtype",
        d_dtype,
        (jnp.float16, jnp.bfloat16, jnp.float8_e4m3fn, jnp.float8_e5m2),
        default=jnp.bfloat16,
    )
    if vector_f32 and c_dtype in {
        jnp.dtype(jnp.float8_e4m3fn),
        jnp.dtype(jnp.float8_e5m2),
    }:
        raise ValueError("vector_f32 does not support an FP8 c_dtype")
    if cd_major != "n":
        raise ValueError(f"cd_major must be 'n', got {cd_major!r}")

    mma_tiler_mn = kernel.require_mma_tiler(mma_tiler_mn)
    if cluster_shape_mn is None:
        cluster_shape_mn = (2, 1) if mma_tiler_mn[0] == kernel.TWO_CTA_MMA_TILER_M else (1, 1)
    cluster_shape_mn = kernel.require_cluster_shape(cluster_shape_mn, mma_tiler_mn=mma_tiler_mn)
    if mma_tiler_mn[1] == 128 and d_dtype in {
        jnp.dtype(jnp.float8_e4m3fn),
        jnp.dtype(jnp.float8_e5m2),
    }:
        raise NotImplementedError("FP8 output requires mma_tiler_mn[1] == 256")

    output_n = n // 2
    a_spec = gemm_a_tensor_spec("k")
    b_spec = gemm_b_tensor_spec("k")
    output_spec = gemm_c_tensor_spec("n")
    scale_spec = block_scale_tensor_spec()
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
        norm_const_tensor,
    ]
    input_specs = [a_spec, b_spec, scale_spec, scale_spec, None, None, None]
    if prob_tensor is not None:
        inputs.append(prob_tensor)
        input_specs.append(probability_tensor_spec())

    has_amax = d_dtype in {jnp.dtype(jnp.float16), jnp.dtype(jnp.bfloat16)}
    output_specs = [
        BufferSpec("c_tensor", (m, n, 1), c_dtype, tensor_spec=output_spec),
        BufferSpec("d_tensor", (m, output_n, 1), d_dtype, tensor_spec=output_spec),
        BufferSpec("d_col_tensor", (m, output_n, 1), d_dtype, tensor_spec=output_spec),
    ]
    if has_amax:
        output_specs.append(BufferSpec("amax_tensor", (experts, 1), jnp.float32, fill_value=-float("inf")))
    output_specs.extend(
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
    results = call_cutedsl(
        _make_launcher(
            acc_dtype=acc_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sf_vec_size=sf_vec_size,
            vector_f32=bool(vector_f32),
            discrete_col_sfd=bool(discrete_col_sfd),
            expert_cnt=experts,
            has_prob=prob_tensor is not None,
            has_amax=has_amax,
            cluster_overlap_margin=int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0")),
        ),
        inputs,
        outputs=output_specs,
        input_specs=input_specs,
        use_static_tensors=True,
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
    sfd_row_tensor = results[result_idx]
    sfd_col_tensor = results[result_idx + 1]
    return GroupedGemmSwigluResult(
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
        sample_norm_const_tensor: Any,
        sample_prob_tensor: Optional[Any] = None,
        acc_dtype: Any = None,
        c_dtype: Any = None,
        d_dtype: Any = None,
        cd_major: str = "n",
        mma_tiler_mn: tuple[int, int] = (256, 256),
        cluster_shape_mn: Optional[tuple[int, int]] = None,
        sf_vec_size: int = 32,
        vector_f32: bool = False,
        m_aligned: int = 256,
        discrete_col_sfd: bool = False,
    ) -> None:
        super().__init__()
        self._sample_descs = {
            "a_tensor": self.make_tensor_desc(sample_a_tensor, name="sample_a_tensor"),
            "b_tensor": self.make_tensor_desc(sample_b_tensor, name="sample_b_tensor"),
            "sfa_tensor": self.make_tensor_desc(sample_sfa_tensor, name="sample_sfa_tensor"),
            "sfb_tensor": self.make_tensor_desc(sample_sfb_tensor, name="sample_sfb_tensor"),
            "padded_offsets": self.make_tensor_desc(sample_padded_offsets, name="sample_padded_offsets"),
            "alpha_tensor": self.make_tensor_desc(sample_alpha_tensor, name="sample_alpha_tensor"),
            "norm_const_tensor": self.make_tensor_desc(sample_norm_const_tensor, name="sample_norm_const_tensor"),
            "prob_tensor": self.make_optional_tensor_desc(sample_prob_tensor, name="sample_prob_tensor"),
        }
        self._config = {
            "acc_dtype": self.as_optional_dtype(acc_dtype),
            "c_dtype": self.as_optional_dtype(c_dtype),
            "d_dtype": self.as_optional_dtype(d_dtype),
            "cd_major": cd_major,
            "mma_tiler_mn": tuple(mma_tiler_mn),
            "cluster_shape_mn": (None if cluster_shape_mn is None else tuple(cluster_shape_mn)),
            "sf_vec_size": sf_vec_size,
            "vector_f32": vector_f32,
            "m_aligned": m_aligned,
            "discrete_col_sfd": discrete_col_sfd,
        }

        self._sample_descs = immutable_mapping(self._sample_descs)
        self._config = immutable_mapping(self._config)

    def _check_support(self) -> bool:
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
        self._config = immutable_mapping({**self._config, **resolved})
        return True

    def __call__(
        self,
        a_tensor: Any,
        b_tensor: Any,
        sfa_tensor: Any,
        sfb_tensor: Any,
        padded_offsets: Any,
        alpha_tensor: Any,
        norm_const_tensor: Any,
        prob_tensor: Optional[Any] = None,
    ) -> GroupedGemmSwigluResult:
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
        norm_const_tensor: Any,
        prob_tensor: Optional[Any] = None,
    ) -> GroupedGemmSwigluResult:
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
        check_call_signatures(self, self._sample_descs, values)
        return _grouped_gemm_swiglu_impl(**values, **self._config)


def grouped_gemm_swiglu_wrapper_sm100(
    a_tensor: Any,
    b_tensor: Any,
    sfa_tensor: Any,
    sfb_tensor: Any,
    padded_offsets: Any,
    alpha_tensor: Any,
    norm_const_tensor: Any,
    prob_tensor: Optional[Any] = None,
    acc_dtype: Any = None,
    c_dtype: Any = None,
    d_dtype: Any = None,
    cd_major: str = "n",
    mma_tiler_mn: tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[tuple[int, int]] = None,
    sf_vec_size: int = 32,
    vector_f32: bool = False,
    m_aligned: int = 256,
    discrete_col_sfd: bool = False,
) -> GroupedGemmSwigluResult:
    """Compute MXFP8 contiguous grouped GEMM and fused SwiGLU."""

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
        cd_major=cd_major,
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
    "GroupedGemmSwigluResult",
    "GroupedGemmSwigluSm100",
    "grouped_gemm_swiglu_wrapper_sm100",
]
