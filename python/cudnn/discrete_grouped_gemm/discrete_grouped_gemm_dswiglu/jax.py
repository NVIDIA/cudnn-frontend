# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for the discrete-weight grouped GEMM dSwiGLU kernel."""

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
from ..._jax import TupleDict
from ..._jax.datatypes import normalize_jax_dtype
from .._jax_common import (
    DiscreteGroupedGemmJaxBase,
    FP8_DTYPES,
    SUPPORTED_COMPUTE_CAPABILITIES,
)


class DiscreteGroupedGemmDswigluSm100(DiscreteGroupedGemmJaxBase):
    """JAX callable for the discrete grouped dSwiGLU/dGeGLU kernel.

    B and SFB are compact stacked expert arrays rather than opaque device
    pointer tables.  Keeping them as custom-call operands gives XLA complete
    buffer-liveness information while the kernel still initializes one TMA
    descriptor pair per expert.
    """

    def __init__(
        self,
        sample_a: Any,
        sample_b: Any,
        sample_c: Any,
        sample_sfa: Any,
        sample_sfb: Any,
        sample_padded_offsets: Any,
        sample_alpha: Any,
        sample_beta: Any,
        sample_prob: Any,
        *,
        sample_norm_const: Any | None = None,
        d_dtype: Any | None = None,
        acc_dtype: Any | None = None,
        mma_tiler_mn: tuple[int, int] = (256, 256),
        cluster_shape_mn: tuple[int, int] | None = None,
        sf_vec_size: int = 32,
        vector_f32: bool = False,
        m_aligned: int = 256,
        discrete_col_sfd: bool = False,
        generate_dbias: bool = False,
        act_func: str = "dswiglu",
        epilogue_op: str | None = None,
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
        self.c_desc = self._to_tensor_desc(sample_c, "sample_c", mode=self.output_mode)
        self.beta_desc = self._to_tensor_desc(sample_beta, "sample_beta")
        self.prob_desc = self._to_tensor_desc(
            sample_prob, "sample_prob", mode=self.probability_mode
        )
        self.norm_const_desc = (
            None
            if sample_norm_const is None
            else self._to_tensor_desc(sample_norm_const, "sample_norm_const")
        )
        self.d_dtype = normalize_jax_dtype(d_dtype, jnp.bfloat16, "d_dtype")
        self.discrete_col_sfd = discrete_col_sfd
        self.generate_dbias = generate_dbias
        self.act_func = act_func
        self.epilogue_op = epilogue_op

        self.generate_sfd = False
        self.d_row_desc = self.d_col_desc = self.dprob_desc = None
        self.dbias_desc = self.sfd_row_desc = self.sfd_col_desc = self.amax_desc = None

    def check_support(self) -> bool:
        self._check_common()
        if self.act_func not in ("dswiglu", "dgeglu"):
            raise ValueError(
                f"act_func must be 'dswiglu' or 'dgeglu', got {self.act_func!r}"
            )
        if self.epilogue_op not in (None, "none", "identity", "relu", "srelu"):
            raise ValueError(
                f"epilogue_op must be None, 'identity', 'relu', or 'srelu', got {self.epilogue_op!r}"
            )
        for name, value in (
            ("discrete_col_sfd", self.discrete_col_sfd),
            ("generate_dbias", self.generate_dbias),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool, got {type(value).__name__}")

        n_out = 2 * self.n
        require_tensor_shape(self.c_desc, (self.m, n_out, 1), label="C")
        if self.c_desc.cudnn_dtype not in (
            data_type.FLOAT,
            data_type.HALF,
            data_type.BFLOAT16,
        ):
            raise ValueError(
                f"C must have float32, float16, or bfloat16 dtype, got {self.c_desc.dtype}"
            )
        if require_compact_major(self.c_desc, "m", "n") != "n":
            raise ValueError("C must use an N-major output layout")
        require_16_byte_alignment(self.c_desc)

        require_tensor_shape(self.beta_desc, (self.expert_cnt,), label="beta")
        if self.beta_desc.cudnn_dtype != data_type.FLOAT:
            raise ValueError(
                f"beta must have float32 dtype, got {self.beta_desc.dtype}"
            )
        require_tensor_shape(self.prob_desc, (self.m, 1, 1), label="prob")
        if self.prob_desc.cudnn_dtype != data_type.FLOAT:
            raise ValueError(
                f"prob must have float32 dtype, got {self.prob_desc.dtype}"
            )
        if self.norm_const_desc is not None:
            require_tensor_shape(self.norm_const_desc, (1,), label="norm_const")
            if self.norm_const_desc.cudnn_dtype != data_type.FLOAT:
                raise ValueError(
                    f"norm_const must have float32 dtype, got {self.norm_const_desc.dtype}"
                )

        d_cudnn_dtype = self._output_cudnn_dtype(self.d_dtype)
        if self.ab_dtype == data_type.FP4_E2M1:
            if d_cudnn_dtype not in (
                data_type.HALF,
                data_type.BFLOAT16,
                data_type.FLOAT,
            ):
                raise ValueError(f"FP4 A and B do not support D dtype {self.d_dtype}")
        elif d_cudnn_dtype not in (
            data_type.HALF,
            data_type.BFLOAT16,
            *FP8_DTYPES,
            data_type.FP4_E2M1,
        ):
            raise ValueError(f"FP8 A and B do not support D dtype {self.d_dtype}")

        self.generate_sfd = (
            self.ab_dtype in FP8_DTYPES
            and self.sf_dtype == data_type.FP8_E8M0
            and d_cudnn_dtype in FP8_DTYPES
        )
        if self.generate_sfd and self.norm_const_desc is None:
            raise ValueError(
                "FP8 A/B and FP8 D require sample_norm_const so the kernel can generate SFD"
            )
        if not self.generate_sfd and self.norm_const_desc is not None:
            raise ValueError(
                "sample_norm_const is only used by FP8 A/B to FP8 D configurations"
            )
        if self.discrete_col_sfd and not self.generate_sfd:
            raise ValueError("discrete_col_sfd requires generated SFD outputs")

        self.d_row_desc = self._canonical_desc(
            (self.m, n_out, 1), self.d_dtype, "d_row_tensor", mode=self.output_mode
        )
        self.d_col_desc = self._canonical_desc(
            (self.m, n_out, 1), self.d_dtype, "d_col_tensor", mode=self.output_mode
        )
        for desc, label in ((self.d_row_desc, "D_row"), (self.d_col_desc, "D_col")):
            if require_compact_major(desc, "m", "n") != "n":
                raise ValueError(f"{label} must use an N-major output layout")
            require_16_byte_alignment(desc)

        self.dprob_desc = self._canonical_desc(
            (self.m, 1, 1),
            jnp.float32,
            "dprob_tensor",
            mode=self.probability_mode,
            init_value=0.0,
        )
        self.dbias_desc = (
            self._canonical_desc(
                (self.expert_cnt, n_out, 1),
                jnp.bfloat16,
                "dbias_tensor",
                init_value=0.0,
            )
            if self.generate_dbias
            else None
        )
        if self.generate_sfd:
            self.sfd_row_desc = self._canonical_desc(
                block_scale_shape(self.m, n_out, 1, self.sf_vec_size),
                self.sfa_desc.dtype,
                "sfd_row_tensor",
                mode=self.scale_mode,
            )
            self.sfd_col_desc = self._canonical_desc(
                block_scale_shape(n_out, self.m, 1, self.sf_vec_size),
                self.sfa_desc.dtype,
                "sfd_col_tensor",
                mode=self.scale_mode,
            )
        else:
            self.sfd_row_desc = self.sfd_col_desc = None
        self.amax_desc = (
            self._canonical_desc(
                (self.expert_cnt, 2, 1),
                jnp.float32,
                "amax_tensor",
                init_value=float("-inf"),
            )
            if d_cudnn_dtype in (data_type.HALF, data_type.BFLOAT16)
            else None
        )
        return True

    @staticmethod
    def _output_cudnn_dtype(dtype: Any) -> data_type:
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
            raise ValueError(f"d_dtype has unsupported dtype {dtype}")
        return resolved

    def __call__(
        self,
        a_tensor: Any,
        b_tensor: Any,
        c_tensor: Any,
        sfa_tensor: Any,
        sfb_tensor: Any,
        padded_offsets: Any,
        alpha_tensor: Any,
        beta_tensor: Any,
        prob_tensor: Any,
        norm_const_tensor: Any | None = None,
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
        self._check_tensor_signature(c_tensor, self.c_desc)
        self._check_tensor_signature(beta_tensor, self.beta_desc)
        self._check_tensor_signature(prob_tensor, self.prob_desc)
        if (norm_const_tensor is None) != (self.norm_const_desc is None):
            raise ValueError("norm_const_tensor presence must match sample_norm_const")
        if self.norm_const_desc is not None:
            self._check_tensor_signature(norm_const_tensor, self.norm_const_desc)
        if linear_offset is None:
            linear_offset = 1.0 if self.act_func == "dgeglu" else 0.0

        if self.m == 0:
            return TupleDict(
                d_row_tensor=self._materialize_output_desc(self.d_row_desc),
                d_col_tensor=self._materialize_output_desc(self.d_col_desc),
                dprob_tensor=self._materialize_output_desc(self.dprob_desc),
                dbias_tensor=self._materialize_output_desc(self.dbias_desc),
                amax_tensor=self._materialize_output_desc(self.amax_desc),
                sfd_row_tensor=self._materialize_output_desc(self.sfd_row_desc),
                sfd_col_tensor=self._materialize_output_desc(self.sfd_col_desc),
            )

        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.nvgpu import OperandMajorMode
        from cutlass.jax import jax_to_cutlass_dtype

        from .discrete_B_blockscaled_grouped_gemm_dglu_dbias import (
            BlockScaledDiscreteWeightDgluDbiasGroupedGemmKernel,
        )

        kernel = BlockScaledDiscreteWeightDgluDbiasGroupedGemmKernel(
            sf_vec_size=self.sf_vec_size,
            acc_dtype=jax_to_cutlass_dtype(self.acc_dtype),
            use_2cta_instrs=self.mma_tiler_mn[0] == 256,
            mma_tiler_mn=self.mma_tiler_mn,
            cluster_shape_mn=self.cluster_shape_mn,
            vectorized_f32=self.vector_f32,
            discrete_col_sfd=self.discrete_col_sfd,
            expert_cnt=self.expert_cnt,
            use_dynamic_sched=self.use_dynamic_sched,
            act_func=self.act_func,
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
            c_tensor,
            sfa_tensor,
            padded_offsets,
            alpha_tensor,
            beta_tensor,
            prob_tensor,
        ]
        input_descs = [
            self.a_desc,
            self.b_desc,
            self.sfb_desc,
            self.c_desc,
            self.sfa_desc,
            self.padded_offsets_desc,
            self.alpha_desc,
            self.beta_desc,
            self.prob_desc,
        ]
        if self.norm_const_desc is not None:
            inputs.append(norm_const_tensor)
            input_descs.append(self.norm_const_desc)

        output_descs = [self.d_row_desc, self.d_col_desc, self.dprob_desc]
        for desc in (
            self.dbias_desc,
            self.sfd_row_desc,
            self.sfd_col_desc,
            self.amax_desc,
        ):
            if desc is not None:
                output_descs.append(desc)

        has_norm = self.norm_const_desc is not None
        has_dbias = self.dbias_desc is not None
        has_sfd = self.sfd_row_desc is not None
        has_amax = self.amax_desc is not None

        def launch(stream: Any, *args: Any) -> None:
            arg_index = 0

            def take() -> Any:
                nonlocal arg_index
                value = args[arg_index]
                arg_index += 1
                return value

            a, b, sfb, c, sfa, offsets, alpha, beta, prob = (take() for _ in range(9))
            norm_const = take() if has_norm else None
            d_row, d_col, dprob = (take() for _ in range(3))
            dbias = take() if has_dbias else None
            sfd_row = take() if has_sfd else None
            sfd_col = take() if has_sfd else None
            amax = take() if has_amax else None
            workspace = take()
            if arg_index != len(args):
                raise RuntimeError(
                    f"Unexpected discrete grouped dSwiGLU buffer count: consumed {arg_index}, received {len(args)}"
                )

            if self.epilogue_op in (None, "none", "identity"):

                def epilogue(x):
                    return x
            elif self.epilogue_op == "relu":

                def epilogue(x):
                    return cute.where(x > 0, x, cute.full_like(x, 0))
            else:

                def epilogue(x):
                    return cute.where(x > 0, x, cute.full_like(x, 0)) ** 2

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
                d_row,
                d_col,
                sfa,
                sfd_row,
                sfd_col,
                amax,
                norm_const,
                offsets,
                alpha,
                beta,
                prob,
                dprob,
                cutlass.Float32(linear_offset),
                dbias,
                max_active_clusters,
                stream,
                epilogue,
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

        d_row_result, d_col_result, dprob_result = result(), result(), result()
        dbias_result = result() if has_dbias else None
        sfd_row_result = result() if has_sfd else None
        sfd_col_result = result() if has_sfd else None
        amax_result = result() if has_amax else None
        return TupleDict(
            d_row_tensor=d_row_result,
            d_col_tensor=d_col_result,
            dprob_tensor=dprob_result,
            dbias_tensor=dbias_result,
            amax_tensor=amax_result,
            sfd_row_tensor=sfd_row_result,
            sfd_col_tensor=sfd_col_result,
        )


@partial(
    jax.jit,
    static_argnames=(
        "d_dtype",
        "acc_dtype",
        "mma_tiler_mn",
        "cluster_shape_mn",
        "sf_vec_size",
        "vector_f32",
        "m_aligned",
        "discrete_col_sfd",
        "generate_dbias",
        "act_func",
        "epilogue_op",
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
def discrete_grouped_gemm_dswiglu_wrapper_sm100(
    a_tensor: Any,
    b_tensor: Any,
    c_tensor: Any,
    sfa_tensor: Any,
    sfb_tensor: Any,
    padded_offsets: Any,
    alpha_tensor: Any,
    beta_tensor: Any,
    prob_tensor: Any,
    norm_const_tensor: Any | None = None,
    *,
    d_dtype: Any | None = None,
    acc_dtype: Any | None = None,
    mma_tiler_mn: tuple[int, int] = (256, 256),
    cluster_shape_mn: tuple[int, int] | None = None,
    sf_vec_size: int = 32,
    vector_f32: bool = False,
    m_aligned: int = 256,
    discrete_col_sfd: bool = False,
    generate_dbias: bool = False,
    act_func: str = "dswiglu",
    epilogue_op: str | None = None,
    use_dynamic_sched: bool = False,
    a_layout: str = "LMK",
    b_layout: str = "LNK",
    output_layout: str = "LMN",
    linear_offset: float | None = None,
    geglu_alpha: float = 1.702,
    glu_clamp_max: float = 7.0,
    glu_clamp_min: float = -7.0,
) -> TupleDict:
    """Run discrete dSwiGLU with XLA-owned stacked expert operands."""

    operation = DiscreteGroupedGemmDswigluSm100(
        a_tensor,
        b_tensor,
        c_tensor,
        sfa_tensor,
        sfb_tensor,
        padded_offsets,
        alpha_tensor,
        beta_tensor,
        prob_tensor,
        sample_norm_const=norm_const_tensor,
        d_dtype=d_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        vector_f32=vector_f32,
        m_aligned=m_aligned,
        discrete_col_sfd=discrete_col_sfd,
        generate_dbias=generate_dbias,
        act_func=act_func,
        epilogue_op=epilogue_op,
        use_dynamic_sched=use_dynamic_sched,
        a_layout=a_layout,
        b_layout=b_layout,
        output_layout=output_layout,
    )
    return operation(
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
        linear_offset=linear_offset,
        geglu_alpha=geglu_alpha,
        glu_clamp_max=glu_clamp_max,
        glu_clamp_min=glu_clamp_min,
    )


__all__ = [
    "DiscreteGroupedGemmDswigluSm100",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "discrete_grouped_gemm_dswiglu_wrapper_sm100",
]
