# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native MegaMoE kernel construction shared by tensor frameworks."""

from ._config import Mxfp8KernelConfig


def _common_kwargs(config: Mxfp8KernelConfig):
    import cutlass

    from ..cutedsl_src.quant_def import CombineFormat

    return dict(
        mma_tiler_mnk=config.mma_tiler_mnk,
        cluster_shape_mnk=config.cluster_shape_mnk,
        use_2cta_instrs=config.use_2cta_instrs,
        group_hint=config.group_hint,
        token_padding_block=config.token_padding_block,
        sf_padding_block=config.sf_padding_block,
        load_balance_mode=config.load_balance_mode,
        force_static_sched=config.force_static_sched,
        clc_bundle_size=config.clc_bundle_size,
        num_sched_stages=config.num_sched_stages,
        ab_dtype=cutlass.Float8E4M3FN,
        sf_vec_size=config.sf_vec_size,
        world_size=config.world_size,
        # Runtime rank belongs to SymmetricBufferHost, keeping the kernel descriptor rank-independent.
        local_rank=0,
        num_topk=config.top_k,
        max_tokens_per_rank=config.max_tokens_per_rank,
        max_recv_size_per_rank=config.max_recv_size_per_rank,
        data_token_capacity=config.physical_recv_pool_size,
        hidden=config.hidden,
        launch_cluster_count=config.launch_cluster_count,
        drop_on_overflow=config.kernel_drop_on_overflow,
        fc2_in_kernel_topk_reduce=config.fc2_in_kernel_topk_reduce,
        token_back_mode=config.token_back_mode,
        epi_flag_batch=config.epi_flag_batch,
        flag_batch=config.flag_batch,
        gate_up_clamp=config.gate_up_clamp,
        combine_format=CombineFormat.parse(config.combine_format),
        act_func=config.act_func,
        weight_storage_mode=config.weight_storage_mode,
    )


def make_forward_kernel(config: Mxfp8KernelConfig, *, target_arch=None, sm_count=None):
    from ..cutedsl_src.kernel_src.rubin.training.mega.fwd_glu.glu_mxfp8_mega_moe_kernel import Sm107MegaMoEMxfp8GluKernel

    return Sm107MegaMoEMxfp8GluKernel.from_kwargs(
        **_common_kwargs(config),
        static_expert_shape=(config.num_experts, config.fc1_out, config.hidden),
        generate_c=config.generate_c,
        fc2_use_bulk=config.fc2_use_bulk,
        fc2_tma_stages=config.fc2_tma_stages,
        enable_col_quant=config.enable_col_quant,
        col_quant_num_ctas=config.col_quant_num_ctas,
        **{name: value for name, value in (("target_arch", target_arch), ("sm_count", sm_count)) if value is not None},
    )


def _dgrad_selector_kwargs(config: Mxfp8KernelConfig):
    return {
        "enable_dgrad_optimizations": config.dgrad_optimization == "ds3_ep4_pattern",
        "dgrad_schedule": "optimized" if config.dgrad_optimization == "rolling" else None,
    }


def _validate_resolved_dgrad_profile(config: Mxfp8KernelConfig, kernel):
    resolved_dgrad = kernel.resolved_dgrad_config
    expected_upstream_profile = {"baseline": "explicit", "rolling": "optimized", "ds3_ep4_pattern": "ds3_ep4_v1"}[config.dgrad_optimization]
    actual_upstream_profile = resolved_dgrad["dgrad_optimization_profile"]
    if actual_upstream_profile != expected_upstream_profile:
        raise RuntimeError(
            "Rubin dgrad profile resolution drifted from the MoeEP integration contract: "
            f"requested {config.dgrad_optimization!r}, resolved {actual_upstream_profile!r}, expected {expected_upstream_profile!r}"
        )
    if config.dgrad_optimization == "ds3_ep4_pattern" and resolved_dgrad["dgrad_optimization_overrides"]:
        raise RuntimeError(
            "Rubin ds3_ep4_v1 profile for ds3_ep4_pattern resolved with unexpected downstream overrides: " f"{resolved_dgrad['dgrad_optimization_overrides']!r}"
        )


def make_backward_kernel(config: Mxfp8KernelConfig, *, target_arch=None, sm_count=None):
    from ..cutedsl_src.kernel_src.rubin.training.mega.bwd_dglu.dglu_mxfp8_mega_moe_kernel import Sm107MegaMoEMxfp8DgluKernel

    kernel = Sm107MegaMoEMxfp8DgluKernel.from_kwargs(
        **_common_kwargs(config),
        static_expert_shape=(config.num_experts, config.intermediate, config.hidden),
        dfc2_recompute=config.dfc2_recompute,
        dfc2_col_output=config.dfc2_col_output,
        enable_grad_y2_col_quant=config.enable_grad_y2_col_quant,
        num_ctas_grad_y2_col_quant=config.col_quant_num_ctas,
        **_dgrad_selector_kwargs(config),
        **{name: value for name, value in (("target_arch", target_arch), ("sm_count", sm_count)) if value is not None},
    )
    _validate_resolved_dgrad_profile(config, kernel)
    return kernel
