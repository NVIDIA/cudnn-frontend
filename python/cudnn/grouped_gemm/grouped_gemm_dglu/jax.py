# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for dense-weight grouped GEMM + dGLU on SM100."""

from __future__ import annotations

from functools import partial
import os
from typing import Any, Optional

import jax
import jax.numpy as jnp

from .._jax_api import (
    ApiBaseJax,
    make_buffer_desc,
    FIX_PAD_SIZE,
    MAX_EXPERTS,
    TWO_CTA_MMA_TILER_M,
    TupleDict,
    as_dtype,
    as_gemm_tensor_desc,
    block_scale_shape,
    block_scale_tensor_spec,
    call_cutedsl,
    dense_workspace_bytes,
    gemm_a_tensor_spec,
    gemm_b_tensor_spec,
    gemm_c_tensor_spec,
    grouped_workspace_tensor_spec,
    is_fp4_dtype,
    is_fp8_dtype,
    probability_tensor_spec,
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
    generate_dbias: bool,
    generate_sfd: bool,
    has_amax: bool,
    use_dynamic_sched: bool,
    act_func: str,
    epilogue_op: str,
    linear_offset: float,
    geglu_alpha: float,
    glu_clamp_max: float,
    glu_clamp_min: float,
    max_active_clusters: int,
):
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
    norm_const = take() if generate_sfd else None

    d_row = take()
    d_col = take()
    dprob = take()
    dbias = take() if generate_dbias else None
    amax = take() if has_amax else None
    sfd_row = take() if generate_sfd else None
    sfd_col = take() if generate_sfd else None
    workspace = take()
    if arg_idx != len(args):
        raise RuntimeError(
            f"Unexpected grouped GEMM argument count: consumed {arg_idx}, received {len(args)}"
        )

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
        use_2cta_instrs=mma_tiler_mn[0] == TWO_CTA_MMA_TILER_M,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        vectorized_f32=vector_f32,
        discrete_col_sfd=discrete_col_sfd,
        expert_cnt=expert_cnt,
        weight_mode=MoEWeightMode.DENSE,
        use_dynamic_sched=use_dynamic_sched,
        act_func=act_func,
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
        amax,
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


def _grouped_gemm_dglu_impl(
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
    output_layout: str = "LMN",
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
    cluster_overlap_margin: int = 0,
    *,
    b_layout: str = "LNK",
    _validate_only: bool = False,
) -> TupleDict | dict[str, Any]:
    """Compute a dense-weight block-scaled grouped GEMM with fused dGLU.

    ``dprob_tensor`` and optional ``dbias_tensor`` are fresh, zero-initialized
    JAX results. Native FP4 inputs use logical array shapes; raw-byte
    reinterpretation remains outside the JAX API.
    """

    output_layout = require_layout("output_layout", output_layout, ("LMN",))
    b_layout = require_layout("b_layout", b_layout, ("LNK", "LKN"))
    a_spec = gemm_a_tensor_spec("LMK")
    b_spec = gemm_b_tensor_spec(b_layout)
    output_spec = gemm_c_tensor_spec(output_layout, name="output_layout")
    a_desc = as_gemm_tensor_desc("a_tensor", a_tensor, a_spec)
    b_desc = as_gemm_tensor_desc("b_tensor", b_tensor, b_spec)
    c_desc = as_gemm_tensor_desc("c_tensor", c_tensor, output_spec)
    m, n, k, experts, ab_dtype = require_grouped_gemm_inputs(
        a_desc,
        b_desc,
        padded_offsets,
        alpha_tensor,
        max_experts=MAX_EXPERTS,
    )
    if n % 32:
        raise ValueError(f"b_tensor N must be divisible by 32 for dGLU, got {n}")
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

    output_n = 2 * n
    expected_c_shape = (m, output_n, 1)
    if c_desc.shape != expected_c_shape:
        raise ValueError(
            f"c_tensor must have canonical shape {expected_c_shape}, got {c_desc.shape}"
        )
    require_dtype(
        c_desc,
        (
            jnp.float32,
            jnp.float16,
            jnp.bfloat16,
            jnp.float8_e4m3fn,
            jnp.float8_e5m2,
        ),
        name="c_tensor.dtype",
    )
    c_dtype = as_dtype(c_desc)
    fp8_dtypes = {
        jnp.dtype(jnp.float8_e4m3fn),
        jnp.dtype(jnp.float8_e5m2),
    }
    if vector_f32 and c_dtype in fp8_dtypes:
        raise ValueError("vector_f32 does not support an FP8 c_tensor")
    require_grouped_probability("prob_tensor", prob_tensor, m=m)
    require_grouped_vector("beta_tensor", beta_tensor, length=experts)
    acc_dtype = require_dtype(
        acc_dtype, (jnp.float32,), name="acc_dtype", default=jnp.float32
    )
    if is_fp4_dtype(ab_dtype):
        valid_d_dtypes = (jnp.float16, jnp.bfloat16, jnp.float32)
        default_d_dtype = jnp.bfloat16
    else:
        valid_d_dtypes = (jnp.float8_e4m3fn, jnp.float8_e5m2)
        default_d_dtype = ab_dtype
    d_dtype = require_dtype(
        d_dtype, valid_d_dtypes, name="d_dtype", default=default_d_dtype
    )
    generate_sfd = is_fp8_dtype(ab_dtype)
    has_amax = d_dtype in {jnp.dtype(jnp.float16), jnp.dtype(jnp.bfloat16)}
    if generate_sfd:
        if norm_const_tensor is None:
            raise ValueError("norm_const_tensor is required for FP8 inputs")
        require_grouped_vector("norm_const_tensor", norm_const_tensor, length=1)
    else:
        norm_const_tensor = None
        discrete_col_sfd = False
    if is_fp4_dtype(ab_dtype) and b_layout != "LNK":
        raise ValueError("Native FP4 B must use the K-major LNK layout")
    if (
        is_fp4_dtype(ab_dtype)
        and sf_vec_size == 16
        and d_dtype == jnp.dtype(jnp.float32)
        and not generate_dbias
    ):
        raise NotImplementedError(
            "FP4 with sf_vec_size=16 and float32 D requires generate_dbias=True"
        )
    if act_func not in ("dswiglu", "dgeglu"):
        raise ValueError(f"act_func must be 'dswiglu' or 'dgeglu', got {act_func!r}")
    if linear_offset is None:
        linear_offset = 1.0 if act_func == "dgeglu" else 0.0
    normalized_epilogue = (
        "identity" if epilogue_op in (None, "none", "identity") else epilogue_op
    )
    if normalized_epilogue not in ("identity", "relu", "srelu"):
        raise ValueError(
            f"epilogue_op must be None, 'none', 'identity', 'relu', or 'srelu', got {epilogue_op!r}"
        )

    mma_tiler_mn = require_grouped_mma_tiler(
        mma_tiler_mn, allowed_m=(128, 256), allowed_n=(256,)
    )
    if cluster_shape_mn is None:
        cluster_shape_mn = (2, 1) if mma_tiler_mn[0] == TWO_CTA_MMA_TILER_M else (1, 1)
    cluster_shape_mn = require_grouped_cluster_shape(
        cluster_shape_mn, mma_tiler_mn=mma_tiler_mn
    )

    scale_spec = block_scale_tensor_spec()
    require_16_byte_extent("a_tensor", k, ab_dtype)
    require_16_byte_extent("b_tensor", n if b_layout == "LKN" else k, ab_dtype)
    require_16_byte_extent("c_tensor", output_n, c_dtype)
    require_16_byte_extent("d_tensor", output_n, d_dtype)

    if _validate_only:
        return {
            "acc_dtype": acc_dtype,
            "d_dtype": d_dtype,
            "mma_tiler_mn": mma_tiler_mn,
            "cluster_shape_mn": cluster_shape_mn,
            "linear_offset": linear_offset,
            "epilogue_op": normalized_epilogue,
        }

    outputs = [
        make_buffer_desc(
            "d_row_tensor", (1, m, output_n), d_dtype, tensor_spec=output_spec
        ),
        make_buffer_desc(
            "d_col_tensor", (1, m, output_n), d_dtype, tensor_spec=output_spec
        ),
        make_buffer_desc(
            "dprob_tensor",
            (1, 1, m),
            jnp.float32,
            tensor_spec=probability_tensor_spec(),
            init_value=0.0,
        ),
    ]
    if generate_dbias:
        outputs.append(
            make_buffer_desc(
                "dbias_tensor",
                (experts, output_n, 1),
                jnp.bfloat16,
                init_value=0.0,
            )
        )
    if has_amax:
        outputs.append(
            make_buffer_desc(
                "amax_tensor",
                (experts, 2, 1),
                jnp.float32,
                init_value=-float("inf"),
            )
        )
    if generate_sfd:
        outputs.extend(
            (
                make_buffer_desc(
                    "sfd_row_tensor",
                    block_scale_shape(m, output_n, 1, sf_vec_size),
                    jnp.float8_e8m0fnu,
                    tensor_spec=scale_spec,
                ),
                make_buffer_desc(
                    "sfd_col_tensor",
                    block_scale_shape(output_n, m, 1, sf_vec_size),
                    jnp.float8_e8m0fnu,
                    tensor_spec=scale_spec,
                ),
            )
        )

    workspace_bytes = max(dense_workspace_bytes(bool(use_dynamic_sched)), 1)
    inputs = [
        a_tensor,
        b_tensor,
        c_tensor,
        sfa_tensor,
        sfb_tensor,
        padded_offsets,
        alpha_tensor,
        beta_tensor,
        prob_tensor,
    ]
    input_specs = [
        a_spec,
        b_spec,
        output_spec,
        scale_spec,
        scale_spec,
        None,
        None,
        None,
        probability_tensor_spec(),
    ]
    if generate_sfd:
        inputs.append(norm_const_tensor)
        input_specs.append(None)
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
            "generate_dbias": bool(generate_dbias),
            "generate_sfd": generate_sfd,
            "has_amax": has_amax,
            "use_dynamic_sched": bool(use_dynamic_sched),
            "act_func": act_func,
            "epilogue_op": normalized_epilogue,
            "linear_offset": float(linear_offset),
            "geglu_alpha": float(geglu_alpha),
            "glu_clamp_max": float(glu_clamp_max),
            "glu_clamp_min": float(glu_clamp_min),
            "cluster_overlap_margin": int(cluster_overlap_margin),
        },
        outputs=outputs,
        workspaces=(
            make_buffer_desc(
                "workspace",
                (workspace_bytes,),
                jnp.uint8,
                tensor_spec=grouped_workspace_tensor_spec(),
            ),
        ),
        input_specs=input_specs,
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
    amax_tensor = results[result_idx] if has_amax else None
    result_idx += int(has_amax)
    sfd_row_tensor = results[result_idx] if generate_sfd else None
    sfd_col_tensor = results[result_idx + 1] if generate_sfd else None
    return TupleDict(
        d_row_tensor=d_row_tensor,
        d_col_tensor=d_col_tensor,
        dprob_tensor=dprob_tensor,
        dbias_tensor=dbias_tensor,
        amax_tensor=amax_tensor,
        sfd_row_tensor=sfd_row_tensor,
        sfd_col_tensor=sfd_col_tensor,
    )


class GroupedGemmDgluSm100(ApiBaseJax):
    """Sample-signature-bound JAX callable for grouped GEMM + dGLU."""

    def __init__(
        self,
        sample_a_tensor: Any,
        sample_c_tensor: Any,
        sample_sfa_tensor: Any,
        sample_padded_offsets: Any,
        sample_alpha_tensor: Any,
        sample_beta_tensor: Any,
        sample_prob_tensor: Any,
        sample_b_tensor: Any,
        sample_sfb_tensor: Any,
        generate_dbias: bool = False,
        sample_norm_const_tensor: Optional[Any] = None,
        acc_dtype: Any = None,
        d_dtype: Any = None,
        output_layout: str = "LMN",
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
        b_layout: str = "LNK",
    ) -> None:
        super().__init__()
        output_layout = require_layout("output_layout", output_layout, ("LMN",))
        b_layout = require_layout("b_layout", b_layout, ("LNK", "LKN"))
        a_spec = gemm_a_tensor_spec("LMK")
        b_spec = gemm_b_tensor_spec(b_layout)
        output_spec = gemm_c_tensor_spec(output_layout, name="output_layout")
        scale_spec = block_scale_tensor_spec()
        self._sample_descs = {
            "a_tensor": self.make_tensor_desc(
                sample_a_tensor, tensor_spec=a_spec, name="sample_a_tensor"
            ),
            "c_tensor": self.make_tensor_desc(
                sample_c_tensor, tensor_spec=output_spec, name="sample_c_tensor"
            ),
            "sfa_tensor": self.make_tensor_desc(
                sample_sfa_tensor, tensor_spec=scale_spec, name="sample_sfa_tensor"
            ),
            "padded_offsets": self.make_tensor_desc(
                sample_padded_offsets, name="sample_padded_offsets"
            ),
            "alpha_tensor": self.make_tensor_desc(
                sample_alpha_tensor, name="sample_alpha_tensor"
            ),
            "beta_tensor": self.make_tensor_desc(
                sample_beta_tensor, name="sample_beta_tensor"
            ),
            "prob_tensor": self.make_tensor_desc(
                sample_prob_tensor,
                tensor_spec=probability_tensor_spec(),
                name="sample_prob_tensor",
            ),
            "b_tensor": self.make_tensor_desc(
                sample_b_tensor, tensor_spec=b_spec, name="sample_b_tensor"
            ),
            "sfb_tensor": self.make_tensor_desc(
                sample_sfb_tensor, tensor_spec=scale_spec, name="sample_sfb_tensor"
            ),
            "norm_const_tensor": self.make_optional_tensor_desc(
                sample_norm_const_tensor, name="sample_norm_const_tensor"
            ),
        }
        self._config = {
            "generate_dbias": generate_dbias,
            "acc_dtype": self.as_optional_dtype(acc_dtype),
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
            "act_func": act_func,
            "linear_offset": linear_offset,
            "geglu_alpha": geglu_alpha,
            "glu_clamp_max": glu_clamp_max,
            "glu_clamp_min": glu_clamp_min,
            "epilogue_op": epilogue_op,
            "use_dynamic_sched": use_dynamic_sched,
            "b_layout": b_layout,
            "cluster_overlap_margin": int(
                os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0")
            ),
        }

        self._sample_descs = self.freeze_mapping(self._sample_descs)
        self._config = self.freeze_mapping(self._config)

    def _check_support(self) -> None:
        resolved = _grouped_gemm_dglu_impl(
            self._sample_descs["a_tensor"],
            self._sample_descs["c_tensor"],
            self._sample_descs["sfa_tensor"],
            self._sample_descs["padded_offsets"],
            self._sample_descs["alpha_tensor"],
            self._sample_descs["beta_tensor"],
            self._sample_descs["prob_tensor"],
            self._sample_descs["b_tensor"],
            self._sample_descs["sfb_tensor"],
            norm_const_tensor=self._sample_descs["norm_const_tensor"],
            **self._config,
            _validate_only=True,
        )
        self._config = self.freeze_mapping({**self._config, **resolved})

    def __call__(
        self,
        a_tensor: Any,
        c_tensor: Any,
        sfa_tensor: Any,
        padded_offsets: Any,
        alpha_tensor: Any,
        beta_tensor: Any,
        prob_tensor: Any,
        b_tensor: Any,
        sfb_tensor: Any,
        norm_const_tensor: Optional[Any] = None,
    ) -> TupleDict:
        return super().__call__(
            a_tensor,
            c_tensor,
            sfa_tensor,
            padded_offsets,
            alpha_tensor,
            beta_tensor,
            prob_tensor,
            b_tensor,
            sfb_tensor,
            norm_const_tensor,
        )

    def _call_impl(
        self,
        a_tensor: Any,
        c_tensor: Any,
        sfa_tensor: Any,
        padded_offsets: Any,
        alpha_tensor: Any,
        beta_tensor: Any,
        prob_tensor: Any,
        b_tensor: Any,
        sfb_tensor: Any,
        norm_const_tensor: Optional[Any] = None,
    ) -> TupleDict:
        values = {
            "a_tensor": a_tensor,
            "c_tensor": c_tensor,
            "sfa_tensor": sfa_tensor,
            "padded_offsets": padded_offsets,
            "alpha_tensor": alpha_tensor,
            "beta_tensor": beta_tensor,
            "prob_tensor": prob_tensor,
            "b_tensor": b_tensor,
            "sfb_tensor": sfb_tensor,
            "norm_const_tensor": norm_const_tensor,
        }
        self.check_tensor_signatures(self._sample_descs, values)
        return _grouped_gemm_dglu_impl(**values, **self._config)


@partial(
    jax.jit,
    static_argnames=(
        "generate_dbias",
        "acc_dtype",
        "d_dtype",
        "output_layout",
        "mma_tiler_mn",
        "cluster_shape_mn",
        "sf_vec_size",
        "vector_f32",
        "m_aligned",
        "discrete_col_sfd",
        "act_func",
        "linear_offset",
        "geglu_alpha",
        "glu_clamp_max",
        "glu_clamp_min",
        "epilogue_op",
        "use_dynamic_sched",
        "b_layout",
    ),
)
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
    output_layout: str = "LMN",
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
    b_layout: str = "LNK",
) -> TupleDict:
    """Compute an MXFP8 dense-weight grouped GEMM with fused dGLU."""

    op = GroupedGemmDgluSm100(
        a_tensor,
        c_tensor,
        sfa_tensor,
        padded_offsets,
        alpha_tensor,
        beta_tensor,
        prob_tensor,
        b_tensor,
        sfb_tensor,
        generate_dbias=generate_dbias,
        sample_norm_const_tensor=norm_const_tensor,
        acc_dtype=acc_dtype,
        d_dtype=d_dtype,
        output_layout=output_layout,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        vector_f32=vector_f32,
        m_aligned=m_aligned,
        discrete_col_sfd=discrete_col_sfd,
        act_func=act_func,
        linear_offset=linear_offset,
        geglu_alpha=geglu_alpha,
        glu_clamp_max=glu_clamp_max,
        glu_clamp_min=glu_clamp_min,
        epilogue_op=epilogue_op,
        use_dynamic_sched=use_dynamic_sched,
        b_layout=b_layout,
    )
    return op(
        a_tensor,
        c_tensor,
        sfa_tensor,
        padded_offsets,
        alpha_tensor,
        beta_tensor,
        prob_tensor,
        b_tensor,
        sfb_tensor,
        norm_const_tensor,
    )


__all__ = [
    "GroupedGemmDgluSm100",
    "grouped_gemm_dglu_wrapper_sm100",
]
