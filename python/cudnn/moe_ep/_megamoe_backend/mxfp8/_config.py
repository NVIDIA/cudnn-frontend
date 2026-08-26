# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Import-light static configuration for the Rubin SM107 MXFP8 kernel."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..._contracts import ForwardConfig
from ._formats import combine_wire_format


@dataclass(frozen=True)
class Mxfp8KernelConfig:
    """Code-generation constants for one dense EP subgroup kernel."""

    num_experts: int
    world_size: int
    local_rank: int
    hidden: int
    intermediate: int
    top_k: int
    max_tokens_per_rank: int
    apply_topk_in_fc1: bool
    gate_up_clamp: float | None
    generate_c: bool
    max_recv_size_per_rank: int | None = None
    drop_on_overflow: bool = True
    enable_col_quant: bool = False
    col_quant_num_ctas: int = 2368
    mma_tiler_mnk: tuple[int, int, int] = (256, 256, 128)
    cluster_shape_mnk: tuple[int, int, int] = (2, 1, 1)
    use_2cta_instrs: bool = True
    load_balance_mode: str = "static"
    force_static_sched: bool = True
    clc_bundle_size: int | None = None
    num_sched_stages: int | None = None
    token_padding_block: int = 128
    sf_padding_block: int = 128
    sf_vec_size: int = 32
    group_hint: int | None = None
    token_back_mode: str = "epi_warps"
    epi_flag_batch: tuple[int, int] = (1, 1)
    flag_batch: int = 1
    fc2_in_kernel_topk_reduce: bool = False
    act_func: str = "swiglu"
    combine_format: str = "bf16"
    fc2_use_bulk: bool = False
    fc2_tma_stages: int | None = None

    def __post_init__(self) -> None:
        if (
            self.max_recv_size_per_rank is not None
            and self.max_recv_size_per_rank <= 0
        ):
            raise ValueError("max_recv_size_per_rank must be positive")
        if self.col_quant_num_ctas <= 0:
            raise ValueError("col_quant_num_ctas must be positive")

    @classmethod
    def from_forward_config(cls, config: ForwardConfig) -> "Mxfp8KernelConfig":
        if config.ep_size < 1:
            raise ValueError("MXFP8 execution requires a positive EP size")
        if config.ep_rank < 0 or config.ep_rank >= config.ep_size:
            raise ValueError(
                f"ep_rank {config.ep_rank} is outside EP size {config.ep_size}"
            )
        if config.max_tokens_per_rank is None:
            raise ValueError("MXFP8 execution requires max_tokens_per_rank")
        max_recv_size_per_rank = (
            config.ep_size * config.max_tokens_per_rank * config.top_k
        )
        if max_recv_size_per_rank <= 0:
            raise ValueError("max_recv_size_per_rank must be positive")
        return cls(
            num_experts=config.experts_per_rank,
            world_size=config.ep_size,
            local_rank=config.ep_rank,
            hidden=config.hidden_size,
            intermediate=config.intermediate_size,
            top_k=config.top_k,
            max_tokens_per_rank=config.max_tokens_per_rank,
            apply_topk_in_fc1=config.apply_topk_in_fc1,
            gate_up_clamp=config.gate_up_clamp,
            generate_c=config.generate_c,
            max_recv_size_per_rank=max_recv_size_per_rank,
            combine_format=combine_wire_format(config.combine_format),
            enable_col_quant=(
                config.backward_wgrad_mode == "operands"
            ),
            token_padding_block=(
                config.token_padding_size
                if config.backward_wgrad_mode == "operands"
                else 128 if config.generate_c else config.token_padding_size
            ),
            sf_padding_block=config.sf_padding_size,
            group_hint=config.tuning.group_hint,
            token_back_mode=config.tuning.token_back_mode,
            epi_flag_batch=config.tuning.epi_flag_batch,
            flag_batch=config.tuning.token_in_flag_batch,
            fc2_in_kernel_topk_reduce=(
                config.tuning.reduce_topk_in_kernel
            ),
        )

    @property
    def fc1_out(self) -> int:
        return 2 * self.intermediate

    @property
    def cluster_size(self) -> int:
        return self.cluster_shape_mnk[0] * self.cluster_shape_mnk[1]

    def tuning_signature(
        self,
        launch_cluster_count: int,
    ) -> tuple[str, tuple[int, int], int, int, bool]:
        """Return the effective rank-independent transport/scheduler knobs."""

        group_hint = (
            launch_cluster_count
            if self.group_hint is None
            else self.group_hint
        )
        return (
            self.token_back_mode,
            self.epi_flag_batch,
            self.flag_batch,
            group_hint,
            self.fc2_in_kernel_topk_reduce,
        )

    def effective_config(self, launch_cluster_count: int) -> dict[str, object]:
        """Return the complete JSON-safe compile-time configuration."""

        effective_group_hint = (
            launch_cluster_count
            if self.group_hint is None
            else self.group_hint
        )
        return {
            "num_experts_per_rank": self.num_experts,
            "world_size": self.world_size,
            "hidden": self.hidden,
            "intermediate": self.intermediate,
            "top_k": self.top_k,
            "max_tokens_per_rank": self.max_tokens_per_rank,
            "max_recv_size_per_rank": self.max_recv_size_per_rank,
            "drop_on_overflow": self.drop_on_overflow,
            "apply_topk_in_fc1": self.apply_topk_in_fc1,
            "gate_up_clamp": self.gate_up_clamp,
            "generate_c": self.generate_c,
            "enable_col_quant": self.enable_col_quant,
            "col_quant_num_ctas": self.col_quant_num_ctas,
            "combine_format": self.combine_format,
            "mma_tiler_mnk": list(self.mma_tiler_mnk),
            "cluster_shape_mnk": list(self.cluster_shape_mnk),
            "use_2cta_instrs": self.use_2cta_instrs,
            "load_balance_mode": self.load_balance_mode,
            "force_static_sched": self.force_static_sched,
            "clc_bundle_size": self.clc_bundle_size,
            "num_sched_stages": self.num_sched_stages,
            "token_padding_block": self.token_padding_block,
            "sf_padding_block": self.sf_padding_block,
            "sf_vec_size": self.sf_vec_size,
            "effective_group_hint": effective_group_hint,
            "token_back_mode": self.token_back_mode,
            "epi_flag_batch": list(self.epi_flag_batch),
            "token_in_flag_batch": self.flag_batch,
            "fc2_in_kernel_topk_reduce": self.fc2_in_kernel_topk_reduce,
            "act_func": self.act_func,
            "fc2_use_bulk": self.fc2_use_bulk,
            "fc2_tma_stages": self.fc2_tma_stages,
            "launch_cluster_count": launch_cluster_count,
        }

    def compile_key(
        self,
        device: torch.device,
        architecture: tuple[int, int],
        launch_cluster_count: int,
        layout_signature: tuple,
    ) -> tuple:
        """Return a pointer/stream-independent in-process JIT cache key."""

        canonical_device = torch.device(device)
        return (
            self,
            canonical_device.index,
            architecture,
            launch_cluster_count,
            layout_signature,
        )


__all__ = ["Mxfp8KernelConfig"]
