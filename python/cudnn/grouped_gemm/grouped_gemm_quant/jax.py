# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for dense-weight grouped GEMM quantization on SM100."""

from __future__ import annotations

import os
from typing import Any, Optional

import jax.numpy as jnp

from ..._jax.api_base import (
    ApiBaseJax,
    BufferSpec,
    TupleDict,
    call_cutedsl,
    require_array,
    require_dtype,
)
from ..._jax.gemm import (
    as_gemm_tensor_desc,
    block_scale_tensor_spec,
    gemm_a_tensor_spec,
    gemm_b_tensor_spec,
    gemm_c_tensor_spec,
    probability_tensor_spec,
    require_16_byte_extent,
    require_layout,
)
from ..._jax.grouped_gemm import (
    grouped_bias_tensor_spec,
    grouped_workspace_tensor_spec,
    require_grouped_fp8_scales,
    require_grouped_gemm_inputs,
    require_grouped_probability,
    require_grouped_vector,
)
from ...gemm_validation import (
    block_scale_shape,
    resolve_max_active_clusters,
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
    has_bias: bool,
    has_row_scale: bool,
    quantized_output: bool,
    use_dynamic_sched: bool,
    cluster_overlap_margin: int,
):
    import cutlass
    from cutlass.cute.nvgpu import OperandMajorMode
    from cutlass.jax import jax_to_cutlass_dtype

    from .grouped_gemm_quant import BlockScaledMoEGroupedGemmQuantKernel

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
    row_scale = take() if has_row_scale else None
    bias = take() if has_bias else None
    norm_const = take() if quantized_output else None
    d = take()
    d_col = take() if quantized_output else None
    amax = None if quantized_output else take()
    sfd_row = take() if quantized_output else None
    sfd_col = take() if quantized_output else None
    workspace = take()
    if arg_idx != len(args):
        raise RuntimeError(f"Unexpected grouped GEMM argument count: consumed {arg_idx}, received {len(args)}")

    kernel = BlockScaledMoEGroupedGemmQuantKernel(
        sf_vec_size=sf_vec_size,
        acc_dtype=jax_to_cutlass_dtype(acc_dtype),
        use_2cta_instrs=mma_tiler_mn[0] == BlockScaledMoEGroupedGemmQuantKernel.TWO_CTA_MMA_TILER_M,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        vectorized_f32=vector_f32,
        generate_sfd=quantized_output,
        discrete_col_sfd=discrete_col_sfd,
        enable_bias=has_bias,
        expert_cnt=expert_cnt,
        use_dynamic_sched=use_dynamic_sched,
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
        d,
        d_col,
        sfa,
        sfd_row,
        sfd_col,
        amax,
        norm_const,
        padded_offsets,
        alpha,
        row_scale,
        bias,
        prob,
        max_active_clusters,
        stream,
    )


def _grouped_gemm_quant_impl(
    a_tensor: Any,
    sfa_tensor: Any,
    padded_offsets: Any,
    alpha_tensor: Any,
    b_tensor: Any,
    sfb_tensor: Any,
    bias_tensor: Optional[Any] = None,
    norm_const_tensor: Optional[Any] = None,
    prob_tensor: Optional[Any] = None,
    row_scale_tensor: Optional[Any] = None,
    acc_dtype: Any = None,
    d_dtype: Any = None,
    output_layout: str = "LMN",
    mma_tiler_mn: tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[tuple[int, int]] = None,
    sf_vec_size: int = 32,
    vector_f32: bool = False,
    m_aligned: int = 256,
    discrete_col_sfd: bool = False,
    use_dynamic_sched: bool = False,
    cluster_overlap_margin: int = 0,
    *,
    b_layout: str = "LNK",
    _validate_only: bool = False,
) -> TupleDict | dict[str, Any]:
    """Compute an MXFP8 dense-weight grouped GEMM with optional quantization.

    The grouped dimension is represented by B's leading ``L`` dimension and by the
    runtime ``padded_offsets`` tensor. FP8 outputs return row/column E8M0 scale
    factors; FP16/BF16 outputs return an initialized per-expert amax reduction.
    Temporary scheduler storage is owned by XLA and is not returned.
    """

    from .grouped_gemm_quant import BlockScaledMoEGroupedGemmQuantKernel

    kernel = BlockScaledMoEGroupedGemmQuantKernel
    output_layout = require_layout("output_layout", output_layout, ("LMN",))
    b_layout = require_layout("b_layout", b_layout, ("LNK", "LKN"))
    a_spec = gemm_a_tensor_spec("LMK")
    b_spec = gemm_b_tensor_spec(b_layout)
    output_spec = gemm_c_tensor_spec(output_layout, name="output_layout")
    a_desc = as_gemm_tensor_desc("a_tensor", a_tensor, a_spec)
    b_desc = as_gemm_tensor_desc("b_tensor", b_tensor, b_spec)
    m, n, k, experts, ab_dtype = require_grouped_gemm_inputs(
        a_desc,
        b_desc,
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
    if prob_tensor is None:
        raise ValueError("prob_tensor is required; pass ones when no gating is needed")
    require_grouped_probability("prob_tensor", prob_tensor, m=m)
    if row_scale_tensor is not None:
        require_grouped_vector("row_scale_tensor", row_scale_tensor, length=m)
    if bias_tensor is not None:
        require_array(
            bias_tensor,
            name="bias_tensor",
            shape=(n, experts),
            dtype=(jnp.float16, jnp.bfloat16, jnp.float32),
        )

    acc_dtype = require_dtype(acc_dtype, (jnp.float32,), name="acc_dtype", default=jnp.float32)
    d_dtype = require_dtype(
        d_dtype,
        (jnp.float16, jnp.bfloat16, jnp.float8_e4m3fn, jnp.float8_e5m2),
        name="d_dtype",
        default=jnp.bfloat16,
    )
    quantized_output = d_dtype in {
        jnp.dtype(jnp.float8_e4m3fn),
        jnp.dtype(jnp.float8_e5m2),
    }
    if quantized_output:
        if norm_const_tensor is None:
            raise ValueError("norm_const_tensor is required for an FP8 output")
        require_grouped_vector("norm_const_tensor", norm_const_tensor, length=1)
    else:
        norm_const_tensor = None
    mma_tiler_mn = kernel.require_mma_tiler(mma_tiler_mn)
    if cluster_shape_mn is None:
        cluster_shape_mn = (2, 1) if mma_tiler_mn[0] == kernel.TWO_CTA_MMA_TILER_M else (1, 1)
    cluster_shape_mn = kernel.require_cluster_shape(cluster_shape_mn, mma_tiler_mn=mma_tiler_mn)

    scale_spec = block_scale_tensor_spec()
    require_16_byte_extent("a_tensor", k, ab_dtype)
    require_16_byte_extent("b_tensor", n if b_layout == "LKN" else k, ab_dtype)
    require_16_byte_extent("d_tensor", n, d_dtype)

    if _validate_only:
        return {
            "acc_dtype": acc_dtype,
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
    if row_scale_tensor is not None:
        inputs.append(row_scale_tensor)
        input_specs.append(None)
    if bias_tensor is not None:
        inputs.append(bias_tensor)
        input_specs.append(grouped_bias_tensor_spec())
    if quantized_output:
        inputs.append(norm_const_tensor)
        input_specs.append(None)

    outputs = [BufferSpec("d_tensor", (1, m, n), d_dtype, tensor_spec=output_spec)]
    if quantized_output:
        outputs.extend(
            (
                BufferSpec("d_col_tensor", (1, m, n), d_dtype, tensor_spec=output_spec),
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
            )
        )
    else:
        outputs.append(BufferSpec("amax_tensor", (experts, 1), jnp.float32, fill_value=-float("inf")))

    workspace_bytes = max(kernel.get_dense_workspace_bytes(bool(use_dynamic_sched)), 1)
    results = call_cutedsl(
        _launch,
        inputs,
        static_args={
            "acc_dtype": acc_dtype,
            "mma_tiler_mn": mma_tiler_mn,
            "cluster_shape_mn": cluster_shape_mn,
            "sf_vec_size": sf_vec_size,
            "vector_f32": bool(vector_f32),
            "discrete_col_sfd": bool(discrete_col_sfd),
            "expert_cnt": experts,
            "has_bias": bias_tensor is not None,
            "has_row_scale": row_scale_tensor is not None,
            "quantized_output": quantized_output,
            "use_dynamic_sched": bool(use_dynamic_sched),
            "cluster_overlap_margin": int(cluster_overlap_margin),
        },
        outputs=outputs,
        workspaces=(
            BufferSpec(
                "workspace",
                (workspace_bytes,),
                jnp.uint8,
                tensor_spec=grouped_workspace_tensor_spec(),
            ),
        ),
        input_specs=input_specs,
    )
    if quantized_output:
        d_tensor, d_col_tensor, sfd_row_tensor, sfd_col_tensor = results
        amax_tensor = None
    else:
        d_tensor, amax_tensor = results
        d_col_tensor = None
        sfd_row_tensor = None
        sfd_col_tensor = None
    return TupleDict(
        d_tensor=d_tensor,
        d_col_tensor=d_col_tensor,
        amax_tensor=amax_tensor,
        sfd_row_tensor=sfd_row_tensor,
        sfd_col_tensor=sfd_col_tensor,
    )


class GroupedGemmQuantSm100(ApiBaseJax):
    """Sample-signature-bound JAX callable for grouped GEMM quantization."""

    def __init__(
        self,
        sample_a_tensor: Any,
        sample_sfa_tensor: Any,
        sample_padded_offsets: Any,
        sample_alpha_tensor: Any,
        sample_b_tensor: Any,
        sample_sfb_tensor: Any,
        sample_bias_tensor: Optional[Any] = None,
        sample_norm_const_tensor: Optional[Any] = None,
        sample_prob_tensor: Optional[Any] = None,
        sample_row_scale_tensor: Optional[Any] = None,
        acc_dtype: Any = None,
        d_dtype: Any = None,
        output_layout: str = "LMN",
        mma_tiler_mn: tuple[int, int] = (256, 256),
        cluster_shape_mn: Optional[tuple[int, int]] = None,
        sf_vec_size: int = 32,
        vector_f32: bool = False,
        m_aligned: int = 256,
        discrete_col_sfd: bool = False,
        use_dynamic_sched: bool = False,
        *,
        b_layout: str = "LNK",
    ) -> None:
        super().__init__()
        output_layout = require_layout("output_layout", output_layout, ("LMN",))
        b_layout = require_layout("b_layout", b_layout, ("LNK", "LKN"))
        a_spec = gemm_a_tensor_spec("LMK")
        b_spec = gemm_b_tensor_spec(b_layout)
        scale_spec = block_scale_tensor_spec()
        self._sample_descs = {
            "a_tensor": self.make_tensor_desc(sample_a_tensor, tensor_spec=a_spec, name="sample_a_tensor"),
            "sfa_tensor": self.make_tensor_desc(sample_sfa_tensor, tensor_spec=scale_spec, name="sample_sfa_tensor"),
            "padded_offsets": self.make_tensor_desc(sample_padded_offsets, name="sample_padded_offsets"),
            "alpha_tensor": self.make_tensor_desc(sample_alpha_tensor, name="sample_alpha_tensor"),
            "b_tensor": self.make_tensor_desc(sample_b_tensor, tensor_spec=b_spec, name="sample_b_tensor"),
            "sfb_tensor": self.make_tensor_desc(sample_sfb_tensor, tensor_spec=scale_spec, name="sample_sfb_tensor"),
            "bias_tensor": self.make_optional_tensor_desc(
                sample_bias_tensor,
                tensor_spec=grouped_bias_tensor_spec(),
                name="sample_bias_tensor",
            ),
            "norm_const_tensor": self.make_optional_tensor_desc(sample_norm_const_tensor, name="sample_norm_const_tensor"),
            "prob_tensor": self.make_optional_tensor_desc(
                sample_prob_tensor,
                tensor_spec=probability_tensor_spec(),
                name="sample_prob_tensor",
            ),
            "row_scale_tensor": self.make_optional_tensor_desc(sample_row_scale_tensor, name="sample_row_scale_tensor"),
        }
        self._config = {
            "acc_dtype": self.as_optional_dtype(acc_dtype),
            "d_dtype": self.as_optional_dtype(d_dtype),
            "output_layout": output_layout,
            "mma_tiler_mn": tuple(mma_tiler_mn),
            "cluster_shape_mn": (None if cluster_shape_mn is None else tuple(cluster_shape_mn)),
            "sf_vec_size": sf_vec_size,
            "vector_f32": vector_f32,
            "m_aligned": m_aligned,
            "discrete_col_sfd": discrete_col_sfd,
            "use_dynamic_sched": use_dynamic_sched,
            "b_layout": b_layout,
            "cluster_overlap_margin": int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0")),
        }

        self._sample_descs = self.freeze_mapping(self._sample_descs)
        self._config = self.freeze_mapping(self._config)

    def _check_support(self) -> None:
        resolved = _grouped_gemm_quant_impl(
            self._sample_descs["a_tensor"],
            self._sample_descs["sfa_tensor"],
            self._sample_descs["padded_offsets"],
            self._sample_descs["alpha_tensor"],
            self._sample_descs["b_tensor"],
            self._sample_descs["sfb_tensor"],
            self._sample_descs["bias_tensor"],
            self._sample_descs["norm_const_tensor"],
            self._sample_descs["prob_tensor"],
            self._sample_descs["row_scale_tensor"],
            **self._config,
            _validate_only=True,
        )
        self._config = self.freeze_mapping({**self._config, **resolved})

    def __call__(
        self,
        a_tensor: Any,
        sfa_tensor: Any,
        padded_offsets: Any,
        alpha_tensor: Any,
        b_tensor: Any,
        sfb_tensor: Any,
        bias_tensor: Optional[Any] = None,
        norm_const_tensor: Optional[Any] = None,
        prob_tensor: Optional[Any] = None,
        row_scale_tensor: Optional[Any] = None,
    ) -> TupleDict:
        return super().__call__(
            a_tensor,
            sfa_tensor,
            padded_offsets,
            alpha_tensor,
            b_tensor,
            sfb_tensor,
            bias_tensor,
            norm_const_tensor,
            prob_tensor,
            row_scale_tensor,
        )

    def _call_impl(
        self,
        a_tensor: Any,
        sfa_tensor: Any,
        padded_offsets: Any,
        alpha_tensor: Any,
        b_tensor: Any,
        sfb_tensor: Any,
        bias_tensor: Optional[Any] = None,
        norm_const_tensor: Optional[Any] = None,
        prob_tensor: Optional[Any] = None,
        row_scale_tensor: Optional[Any] = None,
    ) -> TupleDict:
        values = {
            "a_tensor": a_tensor,
            "sfa_tensor": sfa_tensor,
            "padded_offsets": padded_offsets,
            "alpha_tensor": alpha_tensor,
            "b_tensor": b_tensor,
            "sfb_tensor": sfb_tensor,
            "bias_tensor": bias_tensor,
            "norm_const_tensor": norm_const_tensor,
            "prob_tensor": prob_tensor,
            "row_scale_tensor": row_scale_tensor,
        }
        self.check_tensor_signatures(self._sample_descs, values)
        return _grouped_gemm_quant_impl(**values, **self._config)


def grouped_gemm_quant_wrapper_sm100(
    a_tensor: Any,
    sfa_tensor: Any,
    padded_offsets: Any,
    alpha_tensor: Any,
    b_tensor: Any,
    sfb_tensor: Any,
    bias_tensor: Optional[Any] = None,
    norm_const_tensor: Optional[Any] = None,
    prob_tensor: Optional[Any] = None,
    row_scale_tensor: Optional[Any] = None,
    acc_dtype: Any = None,
    d_dtype: Any = None,
    output_layout: str = "LMN",
    mma_tiler_mn: tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[tuple[int, int]] = None,
    sf_vec_size: int = 32,
    vector_f32: bool = False,
    m_aligned: int = 256,
    discrete_col_sfd: bool = False,
    use_dynamic_sched: bool = False,
    *,
    b_layout: str = "LNK",
) -> TupleDict:
    """Compute an MXFP8 dense-weight grouped GEMM with optional quantization."""

    op = GroupedGemmQuantSm100(
        a_tensor,
        sfa_tensor,
        padded_offsets,
        alpha_tensor,
        b_tensor,
        sfb_tensor,
        bias_tensor,
        norm_const_tensor,
        prob_tensor,
        row_scale_tensor,
        acc_dtype=acc_dtype,
        d_dtype=d_dtype,
        output_layout=output_layout,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        vector_f32=vector_f32,
        m_aligned=m_aligned,
        discrete_col_sfd=discrete_col_sfd,
        use_dynamic_sched=use_dynamic_sched,
        b_layout=b_layout,
    )
    return op(
        a_tensor,
        sfa_tensor,
        padded_offsets,
        alpha_tensor,
        b_tensor,
        sfb_tensor,
        bias_tensor,
        norm_const_tensor,
        prob_tensor,
        row_scale_tensor,
    )


__all__ = [
    "GroupedGemmQuantSm100",
    "grouped_gemm_quant_wrapper_sm100",
]
