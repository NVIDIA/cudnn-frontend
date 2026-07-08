# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for dense grouped GEMM + GLU + Hadamard on SM100."""

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
    HADAMARD_SIZE,
    MAX_EXPERTS,
    SF_VEC_SIZES,
    TupleDict,
    as_gemm_tensor_desc,
    block_scale_tensor_spec,
    call_cutedsl,
    dense_workspace_bytes,
    gemm_a_tensor_spec,
    gemm_b_tensor_spec,
    gemm_c_tensor_spec,
    grouped_bias_tensor_spec,
    grouped_workspace_tensor_spec,
    probability_tensor_spec,
    require_array,
    require_contiguous_alignment,
    require_dtype,
    require_grouped_block_scales,
    require_grouped_cluster_shape,
    require_grouped_gemm_inputs,
    require_grouped_mma_tiler,
    require_grouped_probability,
    require_layout,
)


def hadamard_values(size: int) -> tuple[tuple[int, ...], ...]:
    """Return an unnormalized Sylvester Hadamard matrix as Python values."""

    if size < 1 or size & (size - 1):
        raise ValueError(f"Hadamard size must be a positive power of two, got {size}")
    matrix = ((1,),)
    while len(matrix) < size:
        matrix = tuple(row + row for row in matrix) + tuple(
            row + tuple(-value for value in row) for row in matrix
        )
    return matrix


def _launch(
    stream,
    *args,
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
    max_active_clusters: int,
):
    import cutlass
    from cutlass.cute.nvgpu import OperandMajorMode
    from cutlass.jax import jax_to_cutlass_dtype

    from ..moe_utils import MoEWeightMode
    from .moe_blockscaled_grouped_gemm_glu_hadamard import (
        BlockScaledMoEGroupedGemmGluHadamardKernel,
    )

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
        raise RuntimeError(
            f"Unexpected grouped GEMM argument count: consumed {arg_idx}, received {len(args)}"
        )

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


def _grouped_gemm_glu_hadamard_impl(
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
    output_layout: str = "LMN",
    mma_tiler_mn: tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[tuple[int, int]] = None,
    sf_vec_size: int = 16,
    vector_f32: bool = False,
    m_aligned: int = 256,
    act_func: str = "swiglu",
    use_dynamic_sched: bool = False,
    use_tmem_post_rht_amax: bool = False,
    cluster_overlap_margin: int = 0,
    *,
    b_layout: str = "LNK",
    _validate_only: bool = False,
) -> TupleDict | dict[str, Any]:
    """Compute dense native-FP4 grouped GEMM with GLU and Hadamard amax.

    A and B use JAX's native ``float4_e2m1fn`` dtype with public shapes
    ``(1, M, K)`` and ``(L, N, K)``. Raw ``uint8`` FP4 payloads and discrete
    per-expert pointer arrays are intentionally not part of this API.

    Runtime ``padded_offsets`` values must be nondecreasing multiples of 256,
    must not exceed M, and must end at the number of rows whose outputs are
    meaningful. Rows beyond the final offset are returned as zero. Empty
    experts produce zero amax values. These value constraints are trusted
    while tracing with :func:`jax.jit`.
    """

    from cutlass import Float32
    from cutlass.jax import jax_to_cutlass_dtype

    from .moe_blockscaled_grouped_gemm_glu_hadamard import (
        BlockScaledMoEGroupedGemmGluHadamardKernel,
    )

    kernel = BlockScaledMoEGroupedGemmGluHadamardKernel
    output_layout = require_layout("output_layout", output_layout, ("LMN",))
    b_layout = require_layout("b_layout", b_layout, ("LNK",))
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
        max_experts=MAX_EXPERTS,
        valid_ab_dtypes=(jnp.float4_e2m1fn,),
    )
    if m % FIX_PAD_SIZE:
        raise ValueError(f"M must be divisible by {FIX_PAD_SIZE}, got {m}")
    if n % 64:
        raise ValueError(f"N must be divisible by 64, got {n}")
    if m_aligned != FIX_PAD_SIZE:
        raise ValueError(f"m_aligned must be {FIX_PAD_SIZE}, got {m_aligned}")
    if sf_vec_size not in SF_VEC_SIZES:
        raise ValueError(
            f"sf_vec_size must be one of {SF_VEC_SIZES}, got {sf_vec_size}"
        )
    if act_func not in ("swiglu", "geglu", "srelu"):
        raise ValueError(
            f"act_func must be 'swiglu', 'geglu', or 'srelu', got {act_func!r}"
        )
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
        require_array(
            bias_tensor,
            name="bias_tensor",
            shape=(experts, n),
            dtype=(jnp.float16, jnp.bfloat16, jnp.float32),
        )

    acc_dtype = require_dtype(
        acc_dtype, (jnp.float32,), name="acc_dtype", default=jnp.float32
    )
    c_dtype = require_dtype(
        c_dtype, (jnp.float16, jnp.bfloat16), name="c_dtype", default=jnp.bfloat16
    )
    d_dtype = require_dtype(
        d_dtype, (jnp.float16, jnp.bfloat16), name="d_dtype", default=jnp.bfloat16
    )

    mma_tiler_mn = require_grouped_mma_tiler(
        mma_tiler_mn, allowed_m=(256,), allowed_n=(256,)
    )
    if cluster_shape_mn is None:
        cluster_shape_mn = (2, 1)
    cluster_shape_mn = require_grouped_cluster_shape(
        cluster_shape_mn, mma_tiler_mn=mma_tiler_mn
    )

    output_n = n if act_func == "srelu" else n // 2
    if output_n % HADAMARD_SIZE:
        raise ValueError(
            f"D's N dimension must be divisible by {HADAMARD_SIZE}, got {output_n}"
        )
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

    has_hadamard = bool(use_tmem_post_rht_amax)
    if has_hadamard and hadamard_tensor is not None:
        require_array(
            hadamard_tensor,
            name="hadamard_tensor",
            shape=(HADAMARD_SIZE, HADAMARD_SIZE),
            dtype=jnp.bfloat16,
        )
    elif not has_hadamard and hadamard_tensor is not None:
        raise ValueError(
            "hadamard_tensor is used only when use_tmem_post_rht_amax=True"
        )

    if _validate_only:
        return {
            "acc_dtype": acc_dtype,
            "c_dtype": c_dtype,
            "d_dtype": d_dtype,
            "mma_tiler_mn": mma_tiler_mn,
            "cluster_shape_mn": cluster_shape_mn,
        }

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

    if has_hadamard:
        if hadamard_tensor is None:
            hadamard_tensor = jnp.asarray(
                hadamard_values(HADAMARD_SIZE), dtype=jnp.bfloat16
            )
        inputs.append(hadamard_tensor)
        input_specs.append(None)

    if bias_tensor is not None:
        inputs.append(bias_tensor)
        input_specs.append(grouped_bias_tensor_spec())

    workspace_bytes = max(dense_workspace_bytes(bool(use_dynamic_sched)), 1)
    c_tensor, d_tensor, amax_tensor, post_rht_amax_tensor = call_cutedsl(
        _launch,
        inputs,
        static_args={
            "acc_dtype": acc_dtype,
            "mma_tiler_mn": mma_tiler_mn,
            "cluster_shape_mn": cluster_shape_mn,
            "sf_vec_size": sf_vec_size,
            "vector_f32": bool(vector_f32),
            "expert_cnt": experts,
            "act_func": act_func,
            "has_bias": bias_tensor is not None,
            "has_hadamard": has_hadamard,
            "use_dynamic_sched": bool(use_dynamic_sched),
            "use_tmem_post_rht_amax": bool(use_tmem_post_rht_amax),
            "cluster_overlap_margin": int(cluster_overlap_margin),
        },
        outputs=(
            make_buffer_desc(
                "c_tensor", (1, m, n), c_dtype, tensor_spec=output_spec, init_value=0
            ),
            make_buffer_desc(
                "d_tensor",
                (1, m, output_n),
                d_dtype,
                tensor_spec=output_spec,
                init_value=0,
            ),
            make_buffer_desc("amax_tensor", (experts, 1), jnp.float32, init_value=0.0),
            make_buffer_desc(
                "post_rht_amax_tensor", (experts, 1), jnp.float32, init_value=0.0
            ),
        ),
        workspaces=(
            make_buffer_desc(
                "workspace",
                (workspace_bytes,),
                jnp.uint8,
                tensor_spec=grouped_workspace_tensor_spec(),
                init_value=0,
            ),
        ),
        input_specs=input_specs,
    )
    return TupleDict(
        c_tensor=c_tensor,
        d_tensor=d_tensor,
        amax_tensor=amax_tensor,
        post_rht_amax_tensor=post_rht_amax_tensor,
    )


class GroupedGemmGluHadamardSm100(ApiBaseJax):
    """Sample-signature-bound JAX callable for grouped GEMM + GLU + Hadamard."""

    def __init__(
        self,
        sample_a_tensor: Any,
        sample_b_tensor: Any,
        sample_sfa_tensor: Any,
        sample_sfb_tensor: Any,
        sample_padded_offsets: Any,
        sample_alpha_tensor: Any,
        sample_prob_tensor: Any,
        sample_bias_tensor: Optional[Any] = None,
        sample_hadamard_tensor: Optional[Any] = None,
        acc_dtype: Any = None,
        c_dtype: Any = None,
        d_dtype: Any = None,
        output_layout: str = "LMN",
        mma_tiler_mn: tuple[int, int] = (256, 256),
        cluster_shape_mn: Optional[tuple[int, int]] = None,
        sf_vec_size: int = 16,
        vector_f32: bool = False,
        m_aligned: int = 256,
        act_func: str = "swiglu",
        use_dynamic_sched: bool = False,
        use_tmem_post_rht_amax: bool = False,
        *,
        b_layout: str = "LNK",
    ) -> None:
        super().__init__()
        output_layout = require_layout("output_layout", output_layout, ("LMN",))
        b_layout = require_layout("b_layout", b_layout, ("LNK",))
        a_spec = gemm_a_tensor_spec("LMK")
        b_spec = gemm_b_tensor_spec(b_layout)
        scale_spec = block_scale_tensor_spec()
        self._sample_descs = {
            "a_tensor": self.make_tensor_desc(
                sample_a_tensor, tensor_spec=a_spec, name="sample_a_tensor"
            ),
            "b_tensor": self.make_tensor_desc(
                sample_b_tensor, tensor_spec=b_spec, name="sample_b_tensor"
            ),
            "sfa_tensor": self.make_tensor_desc(
                sample_sfa_tensor, tensor_spec=scale_spec, name="sample_sfa_tensor"
            ),
            "sfb_tensor": self.make_tensor_desc(
                sample_sfb_tensor, tensor_spec=scale_spec, name="sample_sfb_tensor"
            ),
            "padded_offsets": self.make_tensor_desc(
                sample_padded_offsets, name="sample_padded_offsets"
            ),
            "alpha_tensor": self.make_tensor_desc(
                sample_alpha_tensor, name="sample_alpha_tensor"
            ),
            "prob_tensor": self.make_tensor_desc(
                sample_prob_tensor,
                tensor_spec=probability_tensor_spec(),
                name="sample_prob_tensor",
            ),
            "bias_tensor": self.make_optional_tensor_desc(
                sample_bias_tensor,
                tensor_spec=grouped_bias_tensor_spec(),
                name="sample_bias_tensor",
            ),
            "hadamard_tensor": self.make_optional_tensor_desc(
                sample_hadamard_tensor, name="sample_hadamard_tensor"
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
            "act_func": act_func,
            "use_dynamic_sched": use_dynamic_sched,
            "use_tmem_post_rht_amax": use_tmem_post_rht_amax,
            "b_layout": b_layout,
            "cluster_overlap_margin": int(
                os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0")
            ),
        }

        self._sample_descs = self.freeze_mapping(self._sample_descs)
        self._config = self.freeze_mapping(self._config)

    def _check_support(self) -> None:
        resolved = _grouped_gemm_glu_hadamard_impl(
            self._sample_descs["a_tensor"],
            self._sample_descs["b_tensor"],
            self._sample_descs["sfa_tensor"],
            self._sample_descs["sfb_tensor"],
            self._sample_descs["padded_offsets"],
            self._sample_descs["alpha_tensor"],
            self._sample_descs["prob_tensor"],
            self._sample_descs["bias_tensor"],
            self._sample_descs["hadamard_tensor"],
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
        prob_tensor: Any,
        bias_tensor: Optional[Any] = None,
        hadamard_tensor: Optional[Any] = None,
    ) -> TupleDict:
        return super().__call__(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            padded_offsets,
            alpha_tensor,
            prob_tensor,
            bias_tensor,
            hadamard_tensor,
        )

    def _call_impl(
        self,
        a_tensor: Any,
        b_tensor: Any,
        sfa_tensor: Any,
        sfb_tensor: Any,
        padded_offsets: Any,
        alpha_tensor: Any,
        prob_tensor: Any,
        bias_tensor: Optional[Any] = None,
        hadamard_tensor: Optional[Any] = None,
    ) -> TupleDict:
        values = {
            "a_tensor": a_tensor,
            "b_tensor": b_tensor,
            "sfa_tensor": sfa_tensor,
            "sfb_tensor": sfb_tensor,
            "padded_offsets": padded_offsets,
            "alpha_tensor": alpha_tensor,
            "prob_tensor": prob_tensor,
            "bias_tensor": bias_tensor,
            "hadamard_tensor": hadamard_tensor,
        }
        self.check_tensor_signatures(self._sample_descs, values)
        return _grouped_gemm_glu_hadamard_impl(**values, **self._config)


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
        "act_func",
        "use_dynamic_sched",
        "use_tmem_post_rht_amax",
        "b_layout",
    ),
)
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
    output_layout: str = "LMN",
    mma_tiler_mn: tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[tuple[int, int]] = None,
    sf_vec_size: int = 16,
    vector_f32: bool = False,
    m_aligned: int = 256,
    act_func: str = "swiglu",
    use_dynamic_sched: bool = False,
    use_tmem_post_rht_amax: bool = False,
    *,
    b_layout: str = "LNK",
) -> TupleDict:
    """Compute dense native-FP4 grouped GEMM with GLU and Hadamard amax."""

    op = GroupedGemmGluHadamardSm100(
        a_tensor,
        b_tensor,
        sfa_tensor,
        sfb_tensor,
        padded_offsets,
        alpha_tensor,
        prob_tensor,
        bias_tensor,
        hadamard_tensor,
        acc_dtype=acc_dtype,
        c_dtype=c_dtype,
        d_dtype=d_dtype,
        output_layout=output_layout,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        vector_f32=vector_f32,
        m_aligned=m_aligned,
        act_func=act_func,
        use_dynamic_sched=use_dynamic_sched,
        use_tmem_post_rht_amax=use_tmem_post_rht_amax,
        b_layout=b_layout,
    )
    return op(
        a_tensor,
        b_tensor,
        sfa_tensor,
        sfb_tensor,
        padded_offsets,
        alpha_tensor,
        prob_tensor,
        bias_tensor,
        hadamard_tensor,
    )


__all__ = [
    "GroupedGemmGluHadamardSm100",
    "grouped_gemm_glu_hadamard_wrapper_sm100",
]
