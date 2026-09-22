# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rule 8 detectors for the fused / block-scaled grouped-GEMM APIBase owners (recipes R2, R9, R11).

Nine classes: the block-scaled GLU / dGLU implementations, SReLU, dSReLU, Quant, GLU+Hadamard,
GLU+Hadamard+Quant, and the discrete-weight SwiGLU / dSwiGLU. Each carves its scratch (TMA
descriptor slots, dynamic-scheduler counter) from the caller's ``workspace=``; ``compile()`` builds
from fakes and allocates nothing; the first execute of a shape is CUDA-graph capturable."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import pytest
import torch

from fe_api.grouped_gemm._workspace import ws
from fe_api.grouped_gemm.test_discrete_grouped_gemm_dswiglu_utils import (
    allocate_discrete_dswiglu_input_tensors,
    allocate_discrete_dswiglu_output_tensors,
    discrete_dswiglu_init,
)
from fe_api.grouped_gemm.test_discrete_grouped_gemm_swiglu_utils import (
    allocate_discrete_input_tensors,
    allocate_discrete_output_tensors,
    discrete_grouped_gemm_init,
)
from fe_api.grouped_gemm.test_grouped_gemm_dsrelu_utils import (
    allocate_grouped_gemm_dsrelu_tensors,
    allocate_grouped_gemm_input_tensors as allocate_dsrelu_input_tensors,
)
from fe_api.grouped_gemm.test_grouped_gemm_dswiglu_utils import allocate_grouped_gemm_dswiglu_tensors
from fe_api.grouped_gemm.test_grouped_gemm_quant_utils import allocate_grouped_gemm_quant_output_tensors, grouped_gemm_quant_init
from fe_api.grouped_gemm.test_grouped_gemm_srelu_utils import (
    allocate_grouped_gemm_input_tensors as allocate_srelu_input_tensors,
    allocate_grouped_gemm_output_tensors as allocate_srelu_output_tensors,
    grouped_gemm_srelu_init,
)
from fe_api.grouped_gemm.test_grouped_gemm_swiglu_utils import (
    allocate_grouped_gemm_input_tensors,
    allocate_grouped_gemm_output_tensors,
    grouped_gemm_swiglu_init,
)

# One fp8 recipe every family accepts on SM100: e4m3 A/B with e8m0 block scales (vec 32), bf16 C, fp8 D.
FP8 = dict(
    ab_dtype=torch.float8_e4m3fn,
    c_dtype=torch.bfloat16,
    d_dtype=torch.float8_e4m3fn,
    cd_major="n",
    acc_dtype=torch.float32,
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=(2, 1),
    sf_vec_size=32,
    sf_dtype=torch.float8_e8m0fnu,
)
# The Hadamard fusions are NVFP4-only with bf16 outputs.
FP4_BF16 = dict(
    ab_dtype=torch.float4_e2m1fn_x2,
    c_dtype=torch.bfloat16,
    d_dtype=torch.bfloat16,
    cd_major="n",
    acc_dtype=torch.float32,
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=(2, 1),
    sf_vec_size=16,
    sf_dtype=torch.float8_e8m0fnu,
)


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        pytest.skip("SM100 is required")


@dataclass
class Case:
    api: object
    # run(offsets, ptrs, workspace): one execute against the case's fixed operands/outputs.
    run: Callable[[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], object], None]
    offsets: torch.Tensor
    ptrs: Optional[Tuple[torch.Tensor, torch.Tensor]]
    outputs: Dict[str, Optional[torch.Tensor]]

    def fresh(self):
        """Operands the plan has never seen: a per-tensor memo cannot hide a D2H read behind them."""
        ptrs = None if self.ptrs is None else tuple(p.clone() for p in self.ptrs)
        return self.offsets.clone(), ptrs


def _supported(api) -> None:
    try:
        assert api.check_support()
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported on this device: {e}")


def _common(cfg):
    return dict(
        acc_dtype=cfg["acc_dtype"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sf_vec_size=cfg["sf_vec_size"],
        vector_f32=cfg["vector_f32"],
        m_aligned=cfg["m_aligned"],
    )


def _dense_inputs(cfg, **extra):
    return allocate_grouped_gemm_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        l=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
        **extra,
    )


def _discrete_inputs(cfg):
    return allocate_discrete_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        num_experts=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
        b_major=cfg["b_major"],
    )


def _ptrs(inputs):
    return (inputs["b_ptrs_tensor"], inputs["sfb_ptrs_tensor"])


# --------------------------------------------------------------------------- #
#  case builders: one per class and weight mode
# --------------------------------------------------------------------------- #


def _glu_blockscaled(request, discrete):
    from cudnn.gemm.cutedsl.grouped.glu._blockscaled_api import GroupedGemmGluBlockScaledAPI

    if discrete:
        cfg = discrete_grouped_gemm_init(request, **FP8)
        inputs = _discrete_inputs(cfg)
        outputs = allocate_discrete_output_tensors(
            tensor_m=inputs["tensor_m"],
            n=cfg["n"],
            num_experts=cfg["l"],
            ab_dtype=cfg["ab_dtype"],
            c_dtype=cfg["c_dtype"],
            d_dtype=cfg["d_dtype"],
            cd_major=cfg["cd_major"],
            sf_dtype=cfg["sf_dtype"],
            sf_vec_size=cfg["sf_vec_size"],
        )
        weights = dict(num_experts=cfg["l"], b_shape=(cfg["n"], cfg["k"]), b_dtype=inputs["b_list"][0].dtype, b_major=cfg["b_major"])
    else:
        cfg = grouped_gemm_swiglu_init(request, **FP8)
        inputs = _dense_inputs(cfg)
        outputs = allocate_grouped_gemm_output_tensors(
            tensor_m=inputs["tensor_m"],
            n=cfg["n"],
            l=cfg["l"],
            ab_dtype=cfg["ab_dtype"],
            c_dtype=cfg["c_dtype"],
            d_dtype=cfg["d_dtype"],
            cd_major=cfg["cd_major"],
            sf_dtype=cfg["sf_dtype"],
            sf_vec_size=cfg["sf_vec_size"],
        )
        weights = dict(sample_b=inputs["b_tensor"], sample_sfb=inputs["sfb_tensor"])
    api = GroupedGemmGluBlockScaledAPI(
        sample_a=inputs["a_tensor"],
        sample_c=outputs["c_tensor"],
        sample_d=outputs["d_tensor"],
        sample_sfa=inputs["sfa_tensor"],
        sample_padded_offsets=inputs["padded_offsets_tensor"],
        sample_alpha=inputs["alpha_tensor"],
        sample_d_col=outputs["d_col_tensor"],
        sample_amax=outputs.get("amax_tensor"),
        sample_sfd_row=outputs.get("sfd_row_tensor"),
        sample_sfd_col=outputs.get("sfd_col_tensor"),
        sample_norm_const=inputs.get("norm_const_tensor"),
        sample_prob=inputs.get("prob_tensor"),
        discrete_col_sfd=cfg["discrete_col_sfd"],
        **weights,
        **_common(cfg),
    )

    def run(offsets, ptrs, workspace):
        api.execute(
            a_tensor=inputs["a_tensor"],
            c_tensor=outputs["c_tensor"],
            d_tensor=outputs["d_tensor"],
            sfa_tensor=inputs["sfa_tensor"],
            padded_offsets=offsets,
            alpha_tensor=inputs["alpha_tensor"],
            **(dict(b_ptrs=ptrs[0], sfb_ptrs=ptrs[1]) if discrete else dict(b_tensor=inputs["b_tensor"], sfb_tensor=inputs["sfb_tensor"])),
            d_col_tensor=outputs["d_col_tensor"],
            sfd_row_tensor=outputs.get("sfd_row_tensor"),
            sfd_col_tensor=outputs.get("sfd_col_tensor"),
            amax_tensor=outputs.get("amax_tensor"),
            norm_const_tensor=inputs.get("norm_const_tensor"),
            prob_tensor=inputs.get("prob_tensor"),
            workspace=workspace,
        )

    return Case(api, run, inputs["padded_offsets_tensor"], _ptrs(inputs) if discrete else None, outputs)


def _dglu_blockscaled(request, discrete):
    from cudnn.gemm.cutedsl.grouped.dglu._blockscaled_api import GroupedGemmDgluBlockScaledAPI

    if discrete:
        cfg = discrete_dswiglu_init(request, **FP8)
        inputs = allocate_discrete_dswiglu_input_tensors(
            n=cfg["n"],
            k=cfg["k"],
            num_experts=cfg["l"],
            group_m_list=cfg["group_m_list"],
            ab_dtype=cfg["ab_dtype"],
            c_dtype=cfg["c_dtype"],
            sf_dtype=cfg["sf_dtype"],
            sf_vec_size=cfg["sf_vec_size"],
            m_aligned=cfg["m_aligned"],
            b_major=cfg["b_major"],
        )
        outputs = allocate_discrete_dswiglu_output_tensors(
            tensor_m=inputs["tensor_m"],
            n=cfg["n"],
            num_experts=cfg["l"],
            ab_dtype=cfg["ab_dtype"],
            d_dtype=cfg["d_dtype"],
            cd_major=cfg["cd_major"],
            sf_dtype=cfg["sf_dtype"],
            sf_vec_size=cfg["sf_vec_size"],
        )
        weights = dict(num_experts=cfg["l"], b_shape=(cfg["n"], cfg["k"]), b_dtype=inputs["b_list"][0].dtype, b_major=cfg["b_major"])
    else:
        cfg = grouped_gemm_swiglu_init(request, **FP8)
        inputs = _dense_inputs(cfg, b_major=cfg["b_major"])
        inputs, outputs = allocate_grouped_gemm_dswiglu_tensors(
            tensor_m=inputs["tensor_m"],
            n=cfg["n"],
            l=cfg["l"],
            ab_dtype=cfg["ab_dtype"],
            c_dtype=cfg["c_dtype"],
            d_dtype=cfg["d_dtype"],
            cd_major=cfg["cd_major"],
            sf_dtype=cfg["sf_dtype"],
            sf_vec_size=cfg["sf_vec_size"],
            input_tensors=inputs,
        )
        weights = dict(sample_b=inputs["b_tensor"], sample_sfb=inputs["sfb_tensor"])
    api = GroupedGemmDgluBlockScaledAPI(
        sample_a=inputs["a_tensor"],
        sample_c=inputs["c_tensor"],
        sample_d_row=outputs["d_row_tensor"],
        sample_d_col=outputs["d_col_tensor"],
        sample_sfa=inputs["sfa_tensor"],
        sample_padded_offsets=inputs["padded_offsets_tensor"],
        sample_alpha=inputs["alpha_tensor"],
        sample_beta=inputs["beta_tensor"],
        sample_prob=inputs["prob_tensor"],
        sample_dprob=inputs["dprob_tensor"] if discrete else outputs["dprob_tensor"],
        sample_amax=outputs.get("amax_tensor"),
        sample_sfd_row=outputs.get("sfd_row_tensor"),
        sample_sfd_col=outputs.get("sfd_col_tensor"),
        sample_norm_const=inputs.get("norm_const_tensor"),
        discrete_col_sfd=cfg["discrete_col_sfd"],
        **weights,
        **_common(cfg),
    )
    dprob = inputs["dprob_tensor"] if discrete else outputs["dprob_tensor"]

    def run(offsets, ptrs, workspace):
        api.execute(
            a_tensor=inputs["a_tensor"],
            c_tensor=inputs["c_tensor"],
            d_row_tensor=outputs["d_row_tensor"],
            d_col_tensor=outputs["d_col_tensor"],
            sfa_tensor=inputs["sfa_tensor"],
            padded_offsets=offsets,
            alpha_tensor=inputs["alpha_tensor"],
            beta_tensor=inputs["beta_tensor"],
            prob_tensor=inputs["prob_tensor"],
            dprob_tensor=dprob,
            **(dict(b_ptrs=ptrs[0], sfb_ptrs=ptrs[1]) if discrete else dict(b_tensor=inputs["b_tensor"], sfb_tensor=inputs["sfb_tensor"])),
            sfd_row_tensor=outputs.get("sfd_row_tensor"),
            sfd_col_tensor=outputs.get("sfd_col_tensor"),
            amax_tensor=outputs.get("amax_tensor"),
            norm_const_tensor=inputs.get("norm_const_tensor"),
            workspace=workspace,
        )

    return Case(api, run, inputs["padded_offsets_tensor"], _ptrs(inputs) if discrete else None, outputs)


def _srelu(request, discrete):
    from cudnn import GroupedGemmSreluSm100

    if discrete:
        cfg = discrete_grouped_gemm_init(request, **FP8)
        inputs = _discrete_inputs(cfg)
        weights = dict(num_experts=cfg["l"], b_shape=(cfg["n"], cfg["k"]), b_dtype=inputs["b_list"][0].dtype, b_major=cfg["b_major"])
    else:
        cfg = grouped_gemm_srelu_init(request, **FP8)
        inputs = allocate_srelu_input_tensors(
            n=cfg["n"],
            k=cfg["k"],
            l=cfg["l"],
            group_m_list=cfg["group_m_list"],
            ab_dtype=cfg["ab_dtype"],
            sf_dtype=cfg["sf_dtype"],
            sf_vec_size=cfg["sf_vec_size"],
            m_aligned=cfg["m_aligned"],
        )
        weights = dict(sample_b=inputs["b_tensor"], sample_sfb=inputs["sfb_tensor"])
    outputs = allocate_srelu_output_tensors(
        tensor_m=inputs["tensor_m"],
        n=cfg["n"],
        l=cfg["l"],
        ab_dtype=cfg["ab_dtype"],
        c_dtype=cfg["c_dtype"],
        d_dtype=cfg["d_dtype"],
        cd_major=cfg["cd_major"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
    )
    api = GroupedGemmSreluSm100(
        sample_a=inputs["a_tensor"],
        sample_c=outputs["c_tensor"],
        sample_d=outputs["d_tensor"],
        sample_sfa=inputs["sfa_tensor"],
        sample_padded_offsets=inputs["padded_offsets_tensor"],
        sample_alpha=inputs["alpha_tensor"],
        sample_d_col=outputs["d_col_tensor"],
        sample_amax=outputs.get("amax_tensor"),
        sample_sfd_row=outputs.get("sfd_row_tensor"),
        sample_sfd_col=outputs.get("sfd_col_tensor"),
        sample_norm_const=inputs.get("norm_const_tensor"),
        sample_prob=inputs.get("prob_tensor"),
        discrete_col_sfd=cfg["discrete_col_sfd"],
        **weights,
        **_common(cfg),
    )

    def run(offsets, ptrs, workspace):
        api.execute(
            a_tensor=inputs["a_tensor"],
            sfa_tensor=inputs["sfa_tensor"],
            padded_offsets=offsets,
            alpha_tensor=inputs["alpha_tensor"],
            c_tensor=outputs["c_tensor"],
            d_tensor=outputs["d_tensor"],
            **(dict(b_ptrs=ptrs[0], sfb_ptrs=ptrs[1]) if discrete else dict(b_tensor=inputs["b_tensor"], sfb_tensor=inputs["sfb_tensor"])),
            d_col_tensor=outputs["d_col_tensor"],
            sfd_row_tensor=outputs.get("sfd_row_tensor"),
            sfd_col_tensor=outputs.get("sfd_col_tensor"),
            amax_tensor=outputs.get("amax_tensor"),
            norm_const_tensor=inputs.get("norm_const_tensor"),
            prob_tensor=inputs.get("prob_tensor"),
            workspace=workspace,
        )

    return Case(api, run, inputs["padded_offsets_tensor"], _ptrs(inputs) if discrete else None, outputs)


def _dsrelu(request, discrete):
    from cudnn import GroupedGemmDsreluSm100

    if discrete:
        cfg = discrete_grouped_gemm_init(request, **FP8)
        inputs = _discrete_inputs(cfg)
        weights = dict(num_experts=cfg["l"], b_shape=(cfg["n"], cfg["k"]), b_dtype=inputs["b_list"][0].dtype, b_major=cfg["b_major"])
    else:
        cfg = grouped_gemm_swiglu_init(request, **FP8)
        inputs = allocate_dsrelu_input_tensors(
            n=cfg["n"],
            k=cfg["k"],
            l=cfg["l"],
            group_m_list=cfg["group_m_list"],
            ab_dtype=cfg["ab_dtype"],
            b_major=cfg["b_major"],
            sf_dtype=cfg["sf_dtype"],
            sf_vec_size=cfg["sf_vec_size"],
            m_aligned=cfg["m_aligned"],
        )
        weights = dict(sample_b=inputs["b_tensor"], sample_sfb=inputs["sfb_tensor"])
    inputs, outputs = allocate_grouped_gemm_dsrelu_tensors(
        tensor_m=inputs["tensor_m"],
        n=cfg["n"],
        l=cfg["l"],
        ab_dtype=cfg["ab_dtype"],
        c_dtype=cfg["c_dtype"],
        d_dtype=cfg["d_dtype"],
        cd_major=cfg["cd_major"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        input_tensors=inputs,
    )
    api = GroupedGemmDsreluSm100(
        sample_a=inputs["a_tensor"],
        sample_c=inputs["c_tensor"],
        sample_d_row=outputs["d_row_tensor"],
        sample_d_col=outputs["d_col_tensor"],
        sample_d_srelu=outputs["d_srelu_tensor"],
        sample_sfa=inputs["sfa_tensor"],
        sample_padded_offsets=inputs["padded_offsets_tensor"],
        sample_alpha=inputs["alpha_tensor"],
        sample_prob=inputs["prob_tensor"],
        sample_dprob=outputs["dprob_tensor"],
        sample_amax=outputs.get("amax_tensor"),
        sample_sfd_row=outputs.get("sfd_row_tensor"),
        sample_sfd_col=outputs.get("sfd_col_tensor"),
        sample_sfd_col_d_srelu=outputs.get("sfd_col_d_srelu_tensor"),
        sample_norm_const=inputs.get("norm_const_tensor"),
        discrete_col_sfd=cfg["discrete_col_sfd"],
        **weights,
        **_common(cfg),
    )

    def run(offsets, ptrs, workspace):
        api.execute(
            a_tensor=inputs["a_tensor"],
            c_tensor=inputs["c_tensor"],
            d_row_tensor=outputs["d_row_tensor"],
            d_col_tensor=outputs["d_col_tensor"],
            d_srelu_tensor=outputs["d_srelu_tensor"],
            sfa_tensor=inputs["sfa_tensor"],
            padded_offsets=offsets,
            alpha_tensor=inputs["alpha_tensor"],
            prob_tensor=inputs["prob_tensor"],
            dprob_tensor=outputs["dprob_tensor"],
            **(dict(b_ptrs=ptrs[0], sfb_ptrs=ptrs[1]) if discrete else dict(b_tensor=inputs["b_tensor"], sfb_tensor=inputs["sfb_tensor"])),
            sfd_row_tensor=outputs.get("sfd_row_tensor"),
            sfd_col_tensor=outputs.get("sfd_col_tensor"),
            sfd_col_d_srelu_tensor=outputs.get("sfd_col_d_srelu_tensor"),
            norm_const_tensor=inputs.get("norm_const_tensor"),
            amax_tensor=outputs.get("amax_tensor"),
            workspace=workspace,
        )

    return Case(api, run, inputs["padded_offsets_tensor"], _ptrs(inputs) if discrete else None, outputs)


def _quant(request, discrete):
    from cudnn import GroupedGemmQuantSm100

    cfg = grouped_gemm_quant_init(request, **FP8)
    cfg["b_major"] = "k"
    if discrete:
        inputs = _discrete_inputs(cfg)
        weights = dict(num_experts=cfg["l"], b_shape=(cfg["n"], cfg["k"]), b_dtype=cfg["ab_dtype"], b_major=cfg["b_major"])
    else:
        inputs = _dense_inputs(cfg)
        weights = dict(sample_b=inputs["b_tensor"], sample_sfb=inputs["sfb_tensor"])
    outputs = allocate_grouped_gemm_quant_output_tensors(
        tensor_m=inputs["tensor_m"],
        n=cfg["n"],
        l=cfg["l"],
        ab_dtype=cfg["ab_dtype"],
        c_dtype=cfg["c_dtype"],
        d_dtype=cfg["d_dtype"],
        cd_major=cfg["cd_major"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
    )
    api = GroupedGemmQuantSm100(
        sample_a=inputs["a_tensor"],
        sample_sfa=inputs["sfa_tensor"],
        sample_padded_offsets=inputs["padded_offsets_tensor"],
        sample_alpha=inputs["alpha_tensor"],
        sample_d=outputs["d_tensor"],
        sample_d_col=outputs["d_col_tensor"],
        sample_sfd_row=outputs.get("sfd_row_tensor"),
        sample_sfd_col=outputs.get("sfd_col_tensor"),
        sample_amax=outputs.get("amax_tensor"),
        sample_norm_const=inputs.get("norm_const_tensor"),
        sample_prob=inputs["prob_tensor"],
        discrete_col_sfd=cfg["discrete_col_sfd"],
        **weights,
        **_common(cfg),
    )

    def run(offsets, ptrs, workspace):
        api.execute(
            a_tensor=inputs["a_tensor"],
            sfa_tensor=inputs["sfa_tensor"],
            padded_offsets=offsets,
            alpha_tensor=inputs["alpha_tensor"],
            d_tensor=outputs["d_tensor"],
            **(dict(b_ptrs=ptrs[0], sfb_ptrs=ptrs[1]) if discrete else dict(b_tensor=inputs["b_tensor"], sfb_tensor=inputs["sfb_tensor"])),
            d_col_tensor=outputs["d_col_tensor"],
            sfd_row_tensor=outputs.get("sfd_row_tensor"),
            sfd_col_tensor=outputs.get("sfd_col_tensor"),
            amax_tensor=outputs.get("amax_tensor"),
            norm_const_tensor=inputs.get("norm_const_tensor"),
            prob_tensor=inputs["prob_tensor"],
            workspace=workspace,
        )

    return Case(api, run, inputs["padded_offsets_tensor"], _ptrs(inputs) if discrete else None, outputs)


def _hadamard_operands(request, discrete):
    cfg = grouped_gemm_swiglu_init(request, vector_f32=False, discrete_col_sfd=False, **FP4_BF16)
    if discrete:
        inputs = _discrete_inputs(cfg)
        weights = dict(num_experts=cfg["l"], b_shape=(cfg["n"], cfg["k"]), b_dtype=inputs["b_list"][0].dtype, b_major=cfg["b_major"])
    else:
        inputs = _dense_inputs(cfg, b_major=cfg["b_major"], enable_bias=False)
        weights = dict(sample_b=inputs["b_tensor"], sample_sfb=inputs["sfb_tensor"])
    valid_m, n, device = inputs["valid_m"], cfg["n"], inputs["a_tensor"].device

    def n_major(cols, dtype):
        return torch.empty_strided((valid_m, cols, 1), (cols, 1, valid_m * cols), dtype=dtype, device=device)

    return cfg, inputs, weights, n_major


def _glu_hadamard(request, discrete):
    from cudnn import GroupedGemmGluHadamardSm100

    cfg, inputs, weights, n_major = _hadamard_operands(request, discrete)
    n = cfg["n"]
    outputs = {
        "c_tensor": n_major(n, cfg["c_dtype"]),
        "d_tensor": n_major(n // 2, cfg["d_dtype"]),
        "amax_tensor": torch.full((cfg["l"], 1), float("-inf"), dtype=torch.float32, device=inputs["a_tensor"].device),
        "post_rht_amax_tensor": torch.full((cfg["l"], 1), float("-inf"), dtype=torch.float32, device=inputs["a_tensor"].device),
    }
    api = GroupedGemmGluHadamardSm100(
        sample_a=inputs["a_tensor"],
        sample_c=outputs["c_tensor"],
        sample_d=outputs["d_tensor"],
        sample_sfa=inputs["sfa_tensor"],
        sample_padded_offsets=inputs["padded_offsets_tensor"],
        sample_alpha=inputs["alpha_tensor"],
        sample_prob=inputs["prob_tensor"],
        sample_amax=outputs["amax_tensor"],
        sample_post_rht_amax=outputs["post_rht_amax_tensor"],
        act_func="swiglu",
        **weights,
        **_common(cfg),
    )

    def run(offsets, ptrs, workspace):
        api.execute(
            a_tensor=inputs["a_tensor"],
            c_tensor=outputs["c_tensor"],
            d_tensor=outputs["d_tensor"],
            sfa_tensor=inputs["sfa_tensor"],
            padded_offsets=offsets,
            alpha_tensor=inputs["alpha_tensor"],
            prob_tensor=inputs["prob_tensor"],
            **(dict(b_ptrs=ptrs[0], sfb_ptrs=ptrs[1]) if discrete else dict(b_tensor=inputs["b_tensor"], sfb_tensor=inputs["sfb_tensor"])),
            amax_tensor=outputs["amax_tensor"],
            post_rht_amax_tensor=outputs["post_rht_amax_tensor"],
            workspace=workspace,
        )

    return Case(api, run, inputs["padded_offsets_tensor"], _ptrs(inputs) if discrete else None, outputs)


def _glu_hadamard_quant(request, discrete):
    import cutlass

    if not hasattr(cutlass, "FloatNV8E5M3FNU"):
        pytest.skip("glu_hadamard_quant kernels require cutlass-dsl >= 4.8 (cutlass.FloatNV8E5M3FNU)")
    from cudnn import GroupedGemmGluHadamardQuantSm100

    cfg, inputs, weights, n_major = _hadamard_operands(request, discrete)
    n = cfg["n"]
    outputs = {
        "c_tensor": n_major(n, cfg["c_dtype"]),
        "d_tensor": n_major(n // 2, torch.bfloat16),
        "rht_rowwise_tensor": n_major(n // 2, torch.bfloat16),
    }
    api = GroupedGemmGluHadamardQuantSm100(
        sample_a=inputs["a_tensor"],
        sample_c=outputs["c_tensor"],
        sample_d=outputs["d_tensor"],
        sample_sfa=inputs["sfa_tensor"],
        sample_padded_offsets=inputs["padded_offsets_tensor"],
        sample_alpha=inputs["alpha_tensor"],
        sample_prob=inputs["prob_tensor"],
        sample_rht_rowwise=outputs["rht_rowwise_tensor"],
        act_func="swiglu",
        **weights,
        **_common(cfg),
    )

    def run(offsets, ptrs, workspace):
        api.execute(
            a_tensor=inputs["a_tensor"],
            c_tensor=outputs["c_tensor"],
            d_tensor=outputs["d_tensor"],
            sfa_tensor=inputs["sfa_tensor"],
            padded_offsets=offsets,
            alpha_tensor=inputs["alpha_tensor"],
            prob_tensor=inputs["prob_tensor"],
            **(dict(b_ptrs=ptrs[0], sfb_ptrs=ptrs[1]) if discrete else dict(b_tensor=inputs["b_tensor"], sfb_tensor=inputs["sfb_tensor"])),
            rht_rowwise_tensor=outputs["rht_rowwise_tensor"],
            workspace=workspace,
        )

    return Case(api, run, inputs["padded_offsets_tensor"], _ptrs(inputs) if discrete else None, outputs)


def _discrete_swiglu(request):
    from cudnn import DiscreteGroupedGemmSwigluSm100

    cfg = discrete_grouped_gemm_init(request, **FP8)
    inputs = _discrete_inputs(cfg)
    outputs = allocate_discrete_output_tensors(
        tensor_m=inputs["tensor_m"],
        n=cfg["n"],
        num_experts=cfg["l"],
        ab_dtype=cfg["ab_dtype"],
        c_dtype=cfg["c_dtype"],
        d_dtype=cfg["d_dtype"],
        cd_major=cfg["cd_major"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
    )
    api = DiscreteGroupedGemmSwigluSm100(
        sample_a=inputs["a_tensor"],
        num_experts=cfg["l"],
        b_shape=(cfg["n"], cfg["k"]),
        b_dtype=inputs["b_list"][0].dtype,
        sample_c=outputs["c_tensor"],
        sample_d=outputs["d_tensor"],
        sample_sfa=inputs["sfa_tensor"],
        sample_padded_offsets=inputs["padded_offsets_tensor"],
        sample_alpha=inputs["alpha_tensor"],
        sample_amax=outputs.get("amax_tensor"),
        sample_d_col=outputs["d_col_tensor"],
        sample_sfd_row=outputs.get("sfd_row_tensor"),
        sample_sfd_col=outputs.get("sfd_col_tensor"),
        sample_norm_const=inputs.get("norm_const_tensor"),
        sample_prob=inputs.get("prob_tensor"),
        discrete_col_sfd=cfg["discrete_col_sfd"],
        act_func=cfg["act_func"],
        b_major=cfg["b_major"],
        **_common(cfg),
    )

    def run(offsets, ptrs, workspace):
        api.execute(
            a_tensor=inputs["a_tensor"],
            b_ptrs=ptrs[0],
            c_tensor=outputs["c_tensor"],
            d_tensor=outputs["d_tensor"],
            sfa_tensor=inputs["sfa_tensor"],
            sfb_ptrs=ptrs[1],
            padded_offsets=offsets,
            alpha_tensor=inputs["alpha_tensor"],
            d_col_tensor=outputs["d_col_tensor"],
            sfd_row_tensor=outputs.get("sfd_row_tensor"),
            sfd_col_tensor=outputs.get("sfd_col_tensor"),
            norm_const_tensor=inputs.get("norm_const_tensor"),
            prob_tensor=inputs.get("prob_tensor"),
            amax_tensor=outputs.get("amax_tensor"),
            workspace=workspace,
        )

    return Case(api, run, inputs["padded_offsets_tensor"], _ptrs(inputs), outputs)


def _discrete_dswiglu(request):
    from cudnn import DiscreteGroupedGemmDswigluSm100

    cfg = discrete_dswiglu_init(request, **FP8)
    inputs = allocate_discrete_dswiglu_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        num_experts=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        c_dtype=cfg["c_dtype"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
        b_major=cfg["b_major"],
    )
    outputs = allocate_discrete_dswiglu_output_tensors(
        tensor_m=inputs["tensor_m"],
        n=cfg["n"],
        num_experts=cfg["l"],
        ab_dtype=cfg["ab_dtype"],
        d_dtype=cfg["d_dtype"],
        cd_major=cfg["cd_major"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
    )
    api = DiscreteGroupedGemmDswigluSm100(
        sample_a=inputs["a_tensor"],
        num_experts=cfg["l"],
        b_shape=(cfg["n"], cfg["k"]),
        b_dtype=inputs["b_list"][0].dtype,
        sample_c=inputs["c_tensor"],
        sample_d_row=outputs["d_row_tensor"],
        sample_d_col=outputs["d_col_tensor"],
        sample_sfa=inputs["sfa_tensor"],
        sample_padded_offsets=inputs["padded_offsets_tensor"],
        sample_alpha=inputs["alpha_tensor"],
        sample_beta=inputs["beta_tensor"],
        sample_prob=inputs["prob_tensor"],
        sample_dprob=inputs["dprob_tensor"],
        sample_amax=outputs.get("amax_tensor"),
        sample_sfd_row=outputs.get("sfd_row_tensor"),
        sample_sfd_col=outputs.get("sfd_col_tensor"),
        sample_norm_const=inputs.get("norm_const_tensor"),
        discrete_col_sfd=cfg["discrete_col_sfd"],
        act_func=cfg["act_func"],
        b_major=cfg["b_major"],
        **_common(cfg),
    )

    def run(offsets, ptrs, workspace):
        api.execute(
            a_tensor=inputs["a_tensor"],
            b_ptrs=ptrs[0],
            c_tensor=inputs["c_tensor"],
            d_row_tensor=outputs["d_row_tensor"],
            d_col_tensor=outputs["d_col_tensor"],
            sfa_tensor=inputs["sfa_tensor"],
            sfb_ptrs=ptrs[1],
            padded_offsets=offsets,
            alpha_tensor=inputs["alpha_tensor"],
            beta_tensor=inputs["beta_tensor"],
            prob_tensor=inputs["prob_tensor"],
            dprob_tensor=inputs["dprob_tensor"],
            sfd_row_tensor=outputs.get("sfd_row_tensor"),
            sfd_col_tensor=outputs.get("sfd_col_tensor"),
            amax_tensor=outputs.get("amax_tensor"),
            norm_const_tensor=inputs.get("norm_const_tensor"),
            workspace=workspace,
        )

    return Case(api, run, inputs["padded_offsets_tensor"], _ptrs(inputs), outputs)


CASES = {
    "glu_blockscaled-dense": lambda r: _glu_blockscaled(r, False),
    "glu_blockscaled-discrete": lambda r: _glu_blockscaled(r, True),
    "dglu_blockscaled-dense": lambda r: _dglu_blockscaled(r, False),
    "dglu_blockscaled-discrete": lambda r: _dglu_blockscaled(r, True),
    "srelu-dense": lambda r: _srelu(r, False),
    "srelu-discrete": lambda r: _srelu(r, True),
    "dsrelu-dense": lambda r: _dsrelu(r, False),
    "dsrelu-discrete": lambda r: _dsrelu(r, True),
    "quant-dense": lambda r: _quant(r, False),
    "quant-discrete": lambda r: _quant(r, True),
    "glu_hadamard-dense": lambda r: _glu_hadamard(r, False),
    "glu_hadamard-discrete": lambda r: _glu_hadamard(r, True),
    "glu_hadamard_quant-dense": lambda r: _glu_hadamard_quant(r, False),
    "glu_hadamard_quant-discrete": lambda r: _glu_hadamard_quant(r, True),
    "discrete_swiglu": _discrete_swiglu,
    "discrete_dswiglu": _discrete_dswiglu,
}
with_cases = pytest.mark.parametrize("case_name", list(CASES), ids=list(CASES))


def _build(request, case_name) -> Case:
    try:
        case = CASES[case_name](request)
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported on this device: {e}")
    _supported(case.api)
    return case


def _allocated() -> int:
    torch.cuda.synchronize()
    return torch.cuda.memory_stats()["allocation.all.allocated"]


def _bits(t: torch.Tensor) -> torch.Tensor:
    """Same-width integer view: bit-exact comparison for fp8/bf16 outputs that assert_close cannot take."""
    return t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32}[t.element_size()])


# --------------------------------------------------------------------------- #
#  detectors
# --------------------------------------------------------------------------- #


@pytest.mark.L0
@with_cases
def test_compile_allocates_nothing(request, case_name, compile_allocates_nothing):
    """R11: compile() builds the ABI from fake tensors and pointer placeholders; no torch.empty stand-ins."""
    compile_allocates_nothing(_build(request, case_name).api)


@pytest.mark.L0
@with_cases
def test_execute_allocates_nothing_and_never_synchronizes(request, case_name):
    """R9: three warm executes, each with operands the plan has never seen and a caller workspace,
    make no torch allocation and no host sync (the fe_api conftest arms the sync detector)."""
    case = _build(request, case_name)
    case.api.compile()
    case.run(case.offsets, case.ptrs, ws(case.api))

    runs = [(*case.fresh(), ws(case.api)) for _ in range(3)]
    before = _allocated()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for offsets, ptrs, workspace in runs:
            case.run(offsets, ptrs, workspace)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert _allocated() == before, f"{type(case.api).__name__}.execute() allocated (Rule 8, recipe R2)"


@pytest.mark.L0
@with_cases
def test_execute_requires_workspace(request, case_name):
    """R2: the workspace is caller-owned and validated before the launch; the API never falls back to allocating."""
    case = _build(request, case_name)
    case.api.compile()
    nbytes = case.api.scratch_workspace_bytes()
    assert nbytes >= 128 and nbytes % 128 == 0, nbytes

    with pytest.raises(ValueError, match=re.escape(f"{type(case.api).__name__} requires a {nbytes}-byte workspace but execute() received none")):
        case.run(case.offsets, case.ptrs, None)
    with pytest.raises(ValueError, match=r"needs a \d+-byte workspace"):
        case.run(case.offsets, case.ptrs, torch.empty(nbytes - 64, dtype=torch.uint8, device="cuda"))
    slab = torch.empty(nbytes + 128, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="must be 128-byte aligned"):
        case.run(case.offsets, case.ptrs, slab[64:])
    case.run(case.offsets, case.ptrs, slab[:nbytes])  # exactly-sized, aligned: accepted


@pytest.mark.L0
def test_first_execute_under_capture(request):
    """Rule 8: the first execute of a never-seen offsets/pointer table happens inside torch.cuda.graph
    (a serving stack's graph-cache miss), allocates nothing inside the window, and replays bit-exactly."""
    case = _build(request, "discrete_swiglu")
    case.api.compile()
    case.run(case.offsets, case.ptrs, ws(case.api))
    torch.cuda.synchronize()
    expected = {k: v.clone() for k, v in case.outputs.items() if isinstance(v, torch.Tensor)}
    for v in case.outputs.values():
        if isinstance(v, torch.Tensor):
            v.zero_()

    offsets, ptrs = case.fresh()
    workspace = ws(case.api)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # torch 2.13's capture_begin allocates on its own; count only what execute() adds.
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        case.run(offsets, ptrs, workspace)
        after = torch.cuda.memory_stats()["allocation.all.allocated"]
    assert after == before, "execute() allocated inside the capture window"

    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    for name, ref in expected.items():
        assert torch.equal(_bits(case.outputs[name]), _bits(ref)), f"{name} differs between the eager run and the captured replay"
