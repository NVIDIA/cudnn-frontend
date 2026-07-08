# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for the discrete-weight grouped GEMM + SwiGLU kernel."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from ... import data_type
from ..._jax.compiler import compile_options_for_target
from ...gemm.helpers import (
    block_scale_shape,
    require_16_byte_alignment,
    require_compact_major,
    require_tensor_shape,
)
from ..._jax import JaxTensorDesc, TupleDict
from ..._jax.datatypes import normalize_jax_dtype
from .._jax_common import (
    DiscreteGroupedGemmJaxBase,
    FP8_DTYPES,
    SUPPORTED_COMPUTE_CAPABILITIES,
)


class DiscreteGroupedGemmSwigluSm100(DiscreteGroupedGemmJaxBase):
    """JAX callable for grouped GEMM + SwiGLU using stacked expert weights.

    ``sample_b`` and ``sample_sfb`` retain an explicit expert dimension.  They
    are passed to XLA as ordinary operands, so their storage remains live for
    the custom call.  During lowering the discrete kernel derives per-expert
    addresses and builds the same TMA descriptor workspace used by the Torch
    pointer-table API.
    """

    def __init__(
        self,
        sample_a: Any,
        sample_b: Any,
        sample_sfa: Any,
        sample_sfb: Any,
        sample_padded_offsets: Any,
        sample_alpha: Any,
        *,
        sample_norm_const: Any | None = None,
        sample_prob: Any | None = None,
        sample_bias: Any | None = None,
        c_dtype: Any | None = None,
        d_dtype: Any | None = None,
        acc_dtype: Any | None = None,
        mma_tiler_mn: tuple[int, int] = (256, 256),
        cluster_shape_mn: tuple[int, int] | None = None,
        sf_vec_size: int = 32,
        vector_f32: bool = False,
        m_aligned: int = 256,
        discrete_col_sfd: bool = False,
        act_func: str = "swiglu",
        use_dynamic_sched: bool = False,
        a_layout: str = "LMK",
        b_layout: str = "LNK",
        output_layout: str = "LMN",
    ) -> None:
        self._initialize_common(
            sample_a,
            sample_b,
            sample_sfa,
            sample_sfb,
            sample_padded_offsets,
            sample_alpha,
            acc_dtype=acc_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sf_vec_size=sf_vec_size,
            vector_f32=vector_f32,
            m_aligned=m_aligned,
            use_dynamic_sched=use_dynamic_sched,
            a_layout=a_layout,
            b_layout=b_layout,
            output_layout=output_layout,
        )
        self.norm_const_desc = (
            None
            if sample_norm_const is None
            else self._to_tensor_desc(sample_norm_const, "sample_norm_const")
        )
        self._uses_implicit_prob = sample_prob is None
        if sample_prob is None:
            self.prob_desc = JaxTensorDesc.from_shape(
                (1, 1, self.a_desc.shape[0]),
                jnp.float32,
                name="sample_prob",
                mode=self.probability_mode,
            )
        else:
            self.prob_desc = self._to_tensor_desc(
                sample_prob, "sample_prob", mode=self.probability_mode
            )
        self.bias_desc = (
            None
            if sample_bias is None
            else self._to_tensor_desc(sample_bias, "sample_bias", mode=self.bias_mode)
        )
        self.c_dtype = normalize_jax_dtype(c_dtype, jnp.bfloat16, "c_dtype")
        self.d_dtype = normalize_jax_dtype(d_dtype, jnp.bfloat16, "d_dtype")
        self.discrete_col_sfd = discrete_col_sfd
        self.act_func = act_func

        self.generate_sfd = self.norm_const_desc is not None
        self.c_desc = self.d_desc = self.d_col_desc = None
        self.sfd_row_desc = self.sfd_col_desc = self.amax_desc = None

    def check_support(self) -> bool:
        self._check_common()
        if self.act_func not in ("swiglu", "geglu"):
            raise ValueError(
                f"act_func must be 'swiglu' or 'geglu', got {self.act_func!r}"
            )
        if not isinstance(self.discrete_col_sfd, bool):
            raise TypeError(
                f"discrete_col_sfd must be a bool, got {type(self.discrete_col_sfd).__name__}"
            )
        if self.discrete_col_sfd and not self.generate_sfd:
            raise ValueError(
                "discrete_col_sfd requires sample_norm_const and generated SFD outputs"
            )

        if self.norm_const_desc is not None:
            require_tensor_shape(self.norm_const_desc, (1,), label="norm_const")
            if self.norm_const_desc.cudnn_dtype != data_type.FLOAT:
                raise ValueError(
                    f"norm_const must have float32 dtype, got {self.norm_const_desc.dtype}"
                )
        require_tensor_shape(self.prob_desc, (self.m, 1, 1), label="prob")
        if self.prob_desc.cudnn_dtype != data_type.FLOAT:
            raise ValueError(
                f"prob must have float32 dtype, got {self.prob_desc.dtype}"
            )
        if self.bias_desc is not None:
            require_tensor_shape(
                self.bias_desc, (self.n, self.expert_cnt), label="bias"
            )
            if self.bias_desc.cudnn_dtype not in (data_type.HALF, data_type.BFLOAT16):
                raise ValueError(
                    f"bias must have float16 or bfloat16 dtype, got {self.bias_desc.dtype}"
                )

        c_cudnn_dtype = self._output_cudnn_dtype(self.c_dtype, "c_dtype")
        d_cudnn_dtype = self._output_cudnn_dtype(self.d_dtype, "d_dtype")
        if self.ab_dtype == data_type.FP4_E2M1:
            if c_cudnn_dtype not in (data_type.HALF, data_type.BFLOAT16):
                raise NotImplementedError("FP4 A and B require float16 or bfloat16 C")
            if d_cudnn_dtype not in (
                data_type.HALF,
                data_type.BFLOAT16,
                data_type.FLOAT,
            ):
                raise ValueError(f"FP4 A and B do not support D dtype {self.d_dtype}")
            if self.sf_vec_size == 16 and d_cudnn_dtype == data_type.FLOAT:
                raise NotImplementedError(
                    "FP4 A and B with sf_vec_size=16 do not support float32 D"
                )
        elif d_cudnn_dtype not in (
            data_type.HALF,
            data_type.BFLOAT16,
            *FP8_DTYPES,
            data_type.FP4_E2M1,
        ):
            raise ValueError(f"FP8 A and B do not support D dtype {self.d_dtype}")

        output_n = self.n // 2
        self.c_desc = self._canonical_desc(
            (self.m, self.n, 1), self.c_dtype, "c_tensor", mode=self.output_mode
        )
        self.d_desc = self._canonical_desc(
            (self.m, output_n, 1), self.d_dtype, "d_tensor", mode=self.output_mode
        )
        self.d_col_desc = self._canonical_desc(
            (self.m, output_n, 1), self.d_dtype, "d_col_tensor", mode=self.output_mode
        )
        for desc, label in (
            (self.c_desc, "C"),
            (self.d_desc, "D"),
            (self.d_col_desc, "D_col"),
        ):
            if require_compact_major(desc, "m", "n") != "n":
                raise ValueError(f"{label} must use an N-major output layout")
            require_16_byte_alignment(desc)

        if self.generate_sfd:
            self.sfd_row_desc = self._canonical_desc(
                block_scale_shape(self.m, output_n, 1, self.sf_vec_size),
                self.sfa_desc.dtype,
                "sfd_row_tensor",
                mode=self.scale_mode,
            )
            self.sfd_col_desc = self._canonical_desc(
                block_scale_shape(output_n, self.m, 1, self.sf_vec_size),
                self.sfa_desc.dtype,
                "sfd_col_tensor",
                mode=self.scale_mode,
            )
        else:
            self.sfd_row_desc = self.sfd_col_desc = None
        self.amax_desc = (
            self._canonical_desc(
                (self.expert_cnt, 1),
                jnp.float32,
                "amax_tensor",
                init_value=float("-inf"),
            )
            if d_cudnn_dtype in (data_type.HALF, data_type.BFLOAT16)
            else None
        )
        return True

    @staticmethod
    def _output_cudnn_dtype(dtype: Any, name: str) -> data_type:
        from ..._jax.datatypes import jax_to_cudnn_dtype

        resolved = jax_to_cudnn_dtype(dtype)
        allowed = {
            data_type.FLOAT,
            data_type.HALF,
            data_type.BFLOAT16,
            data_type.FP8_E4M3,
            data_type.FP8_E5M2,
            data_type.FP4_E2M1,
        }
        if resolved not in allowed:
            raise ValueError(f"{name} has unsupported dtype {dtype}")
        return resolved

    def __call__(
        self,
        a_tensor: Any,
        b_tensor: Any,
        sfa_tensor: Any,
        sfb_tensor: Any,
        padded_offsets: Any,
        alpha_tensor: Any,
        norm_const_tensor: Any | None = None,
        prob_tensor: Any | None = None,
        bias_tensor: Any | None = None,
        *,
        linear_offset: float | None = None,
        geglu_alpha: float = 1.702,
        glu_clamp_max: float = 7.0,
        glu_clamp_min: float = -7.0,
    ) -> TupleDict:
        self.check_support()
        self._check_runtime_common(
            a_tensor, b_tensor, sfa_tensor, sfb_tensor, padded_offsets, alpha_tensor
        )
        self._check_optional_input(norm_const_tensor, self.norm_const_desc)
        if self._uses_implicit_prob:
            if prob_tensor is not None:
                raise ValueError(
                    "prob_tensor must be None when sample_prob was omitted"
                )
            prob_tensor = jnp.ones((1, 1, self.m), dtype=jnp.float32)
        else:
            self._check_tensor_signature(prob_tensor, self.prob_desc)
        self._check_optional_input(bias_tensor, self.bias_desc)
        if linear_offset is None:
            linear_offset = 1.0 if self.act_func == "geglu" else 0.0

        if self.m == 0:
            return TupleDict(
                c_tensor=self._materialize_output_desc(self.c_desc),
                d_tensor=self._materialize_output_desc(self.d_desc),
                d_col_tensor=self._materialize_output_desc(self.d_col_desc),
                amax_tensor=self._materialize_output_desc(self.amax_desc),
                sfd_row_tensor=self._materialize_output_desc(self.sfd_row_desc),
                sfd_col_tensor=self._materialize_output_desc(self.sfd_col_desc),
            )

        import cutlass
        from cutlass.cute.nvgpu import OperandMajorMode
        from cutlass.jax import jax_to_cutlass_dtype

        from .discrete_B_blockscaled_grouped_gemm_glu_bias import (
            BlockScaledDiscreteWeightGroupedGemmBiasKernel,
        )

        kernel = BlockScaledDiscreteWeightGroupedGemmBiasKernel(
            sf_vec_size=self.sf_vec_size,
            acc_dtype=jax_to_cutlass_dtype(self.acc_dtype),
            use_2cta_instrs=self.mma_tiler_mn[0] == 256,
            mma_tiler_mn=self.mma_tiler_mn,
            cluster_shape_mn=self.cluster_shape_mn,
            vectorized_f32=self.vector_f32,
            generate_sfd=self.generate_sfd,
            discrete_col_sfd=self.discrete_col_sfd,
            expert_cnt=self.expert_cnt,
            use_dynamic_sched=self.use_dynamic_sched,
            act_func=self.act_func,
            enable_bias=self.bias_desc is not None,
            stacked_expert_inputs=True,
        )
        max_active_clusters = self._get_max_active_clusters(
            self.cluster_shape_mn[0] * self.cluster_shape_mn[1],
            overlap_margin=self.num_cluster_overlap_margin,
        )
        workspace_desc = self._workspace_desc(kernel.get_workspace_bytes())

        inputs = [
            a_tensor,
            b_tensor,
            sfb_tensor,
            sfa_tensor,
            padded_offsets,
            alpha_tensor,
        ]
        input_descs = [
            self.a_desc,
            self.b_desc,
            self.sfb_desc,
            self.sfa_desc,
            self.padded_offsets_desc,
            self.alpha_desc,
        ]
        for value, desc in (
            (norm_const_tensor, self.norm_const_desc),
            (prob_tensor, self.prob_desc),
            (bias_tensor, self.bias_desc),
        ):
            if desc is not None:
                inputs.append(value)
                input_descs.append(desc)

        output_descs = [self.c_desc, self.d_desc, self.d_col_desc]
        for desc in (
            self.sfd_row_desc,
            self.sfd_col_desc,
            self.amax_desc,
        ):
            if desc is not None:
                output_descs.append(desc)

        has_norm = self.norm_const_desc is not None
        has_prob = True
        has_bias = self.bias_desc is not None
        has_sfd = self.sfd_row_desc is not None
        has_amax = self.amax_desc is not None

        def launch(stream: Any, *args: Any) -> None:
            arg_index = 0

            def take() -> Any:
                nonlocal arg_index
                value = args[arg_index]
                arg_index += 1
                return value

            a, b, sfb, sfa, offsets, alpha = (take() for _ in range(6))
            norm_const = take() if has_norm else None
            prob = take() if has_prob else None
            bias = take() if has_bias else None
            c, d, d_col = (take() for _ in range(3))
            sfd_row = take() if has_sfd else None
            sfd_col = take() if has_sfd else None
            amax = take() if has_amax else None
            workspace = take()
            if arg_index != len(args):
                raise RuntimeError(
                    f"Unexpected discrete grouped SwiGLU buffer count: consumed {arg_index}, received {len(args)}"
                )

            kernel(
                a,
                b.iterator,
                sfb.iterator,
                cutlass.Int32(self.n),
                cutlass.Int32(self.k),
                cutlass.Int64(self.k),
                OperandMajorMode.K,
                workspace.iterator,
                c,
                d,
                d_col,
                sfa,
                sfd_row,
                sfd_col,
                amax,
                norm_const,
                offsets,
                alpha,
                prob,
                bias,
                max_active_clusters,
                stream,
                lambda x: x,
                cutlass.Float32(linear_offset),
                cutlass.Float32(geglu_alpha),
                cutlass.Float32(glu_clamp_max),
                cutlass.Float32(glu_clamp_min),
            )

        results = self._call_kernel(
            tuple(inputs),
            launch=launch,
            output_descs=tuple(output_descs),
            input_descs=tuple(input_descs),
            workspace_descs=(workspace_desc,),
            compile_options=compile_options_for_target(self.compute_capability),
        )
        result_index = 0

        def result() -> Any:
            nonlocal result_index
            value = results[result_index]
            result_index += 1
            return value

        c_result, d_result, d_col_result = result(), result(), result()
        sfd_row_result = result() if has_sfd else None
        sfd_col_result = result() if has_sfd else None
        amax_result = result() if has_amax else None
        return TupleDict(
            c_tensor=c_result,
            d_tensor=d_result,
            d_col_tensor=d_col_result,
            amax_tensor=amax_result,
            sfd_row_tensor=sfd_row_result,
            sfd_col_tensor=sfd_col_result,
        )

    def _check_optional_input(
        self,
        value: Any | None,
        desc: JaxTensorDesc | None,
    ) -> None:
        if (value is None) != (desc is None):
            name = "optional input" if desc is None else desc.name
            raise ValueError(
                f"{name} presence must match the sample passed to the constructor"
            )
        if desc is not None:
            self._check_tensor_signature(value, desc)


@partial(
    jax.jit,
    static_argnames=(
        "c_dtype",
        "d_dtype",
        "acc_dtype",
        "mma_tiler_mn",
        "cluster_shape_mn",
        "sf_vec_size",
        "vector_f32",
        "m_aligned",
        "discrete_col_sfd",
        "act_func",
        "use_dynamic_sched",
        "a_layout",
        "b_layout",
        "output_layout",
        "linear_offset",
        "geglu_alpha",
        "glu_clamp_max",
        "glu_clamp_min",
    ),
)
def discrete_grouped_gemm_swiglu_wrapper_sm100(
    a_tensor: Any,
    b_tensor: Any,
    sfa_tensor: Any,
    sfb_tensor: Any,
    padded_offsets: Any,
    alpha_tensor: Any,
    norm_const_tensor: Any | None = None,
    prob_tensor: Any | None = None,
    bias_tensor: Any | None = None,
    *,
    c_dtype: Any | None = None,
    d_dtype: Any | None = None,
    acc_dtype: Any | None = None,
    mma_tiler_mn: tuple[int, int] = (256, 256),
    cluster_shape_mn: tuple[int, int] | None = None,
    sf_vec_size: int = 32,
    vector_f32: bool = False,
    m_aligned: int = 256,
    discrete_col_sfd: bool = False,
    act_func: str = "swiglu",
    use_dynamic_sched: bool = False,
    a_layout: str = "LMK",
    b_layout: str = "LNK",
    output_layout: str = "LMN",
    linear_offset: float | None = None,
    geglu_alpha: float = 1.702,
    glu_clamp_max: float = 7.0,
    glu_clamp_min: float = -7.0,
) -> TupleDict:
    """Run the discrete kernel with XLA-owned stacked expert operands."""

    operation = DiscreteGroupedGemmSwigluSm100(
        a_tensor,
        b_tensor,
        sfa_tensor,
        sfb_tensor,
        padded_offsets,
        alpha_tensor,
        sample_norm_const=norm_const_tensor,
        sample_prob=prob_tensor,
        sample_bias=bias_tensor,
        c_dtype=c_dtype,
        d_dtype=d_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        vector_f32=vector_f32,
        m_aligned=m_aligned,
        discrete_col_sfd=discrete_col_sfd,
        act_func=act_func,
        use_dynamic_sched=use_dynamic_sched,
        a_layout=a_layout,
        b_layout=b_layout,
        output_layout=output_layout,
    )
    return operation(
        a_tensor,
        b_tensor,
        sfa_tensor,
        sfb_tensor,
        padded_offsets,
        alpha_tensor,
        norm_const_tensor,
        prob_tensor,
        bias_tensor,
        linear_offset=linear_offset,
        geglu_alpha=geglu_alpha,
        glu_clamp_max=glu_clamp_max,
        glu_clamp_min=glu_clamp_min,
    )


__all__ = [
    "DiscreteGroupedGemmSwigluSm100",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "discrete_grouped_gemm_swiglu_wrapper_sm100",
]
