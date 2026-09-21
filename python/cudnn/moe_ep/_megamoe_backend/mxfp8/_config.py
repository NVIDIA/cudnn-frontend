# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import-light static configuration for the Rubin SM107 MXFP8 kernel."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal

import torch

from ..._config import ResolvedMoeEpConfig
from ._formats import combine_wire_format

_PHYSICAL_POOL_ALIGNMENT = 128
MXFP8_CLUSTER_SHAPE_MNK = (2, 1, 1)


@dataclass(frozen=True)
class Mxfp8KernelConfig:
    """Fully resolved code-generation/ABI constants for one phase."""

    num_experts: int
    world_size: int
    hidden: int
    intermediate: int
    top_k: int
    max_tokens_per_rank: int
    gate_up_clamp: float | None
    generate_c: bool
    physical_recv_pool_size: int
    # The vendored upstream kernel interprets this field as a logical route limit.
    max_recv_size_per_rank: int
    kernel_drop_on_overflow: bool
    enable_col_quant: bool
    dfc2_recompute: bool
    dfc2_col_output: bool
    enable_grad_y2_col_quant: bool
    token_padding_block: int
    sf_padding_block: int
    sf_vec_size: int
    combine_format: str
    weight_storage_mode: str
    group_hint: int
    token_back_mode: str
    epi_flag_batch: tuple[int, int]
    flag_batch: int
    fc2_in_kernel_topk_reduce: bool
    col_quant_num_ctas: int = 2368
    mma_tiler_mnk: tuple[int, int, int] = (256, 256, 128)
    cluster_shape_mnk: tuple[int, int, int] = MXFP8_CLUSTER_SHAPE_MNK
    launch_cluster_count: int = 1
    use_2cta_instrs: bool = True
    load_balance_mode: str = "static"
    force_static_sched: bool = True
    clc_bundle_size: int | None = None
    num_sched_stages: int | None = None
    act_func: str = "swiglu"
    fc2_use_bulk: bool = False
    fc2_tma_stages: int | None = None
    dgrad_optimization: Literal[
        "baseline",
        "rolling",
        "ds3_ep4_v1",
    ] = "baseline"

    def __post_init__(self) -> None:
        if self.physical_recv_pool_size <= 0:
            raise ValueError("physical_recv_pool_size must be positive")
        if self.physical_recv_pool_size % _PHYSICAL_POOL_ALIGNMENT:
            raise ValueError(
                "physical_recv_pool_size must satisfy P % 128 == 0, "
                f"got P={self.physical_recv_pool_size}"
            )
        if self.max_recv_size_per_rank <= 0:
            raise ValueError("max_recv_size_per_rank must be positive")
        if self.group_hint <= 0:
            raise ValueError("group_hint must be positive")
        if self.launch_cluster_count <= 0:
            raise ValueError("launch_cluster_count must be positive")
        if self.dgrad_optimization not in (
            "baseline",
            "rolling",
            "ds3_ep4_v1",
        ):
            raise ValueError(
                "dgrad_optimization must be 'baseline', 'rolling', or "
                f"'ds3_ep4_v1', got {self.dgrad_optimization!r}"
            )
        if self.col_quant_num_ctas == -1:
            if (
                self.dgrad_optimization != "ds3_ep4_v1"
                or not self.enable_grad_y2_col_quant
            ):
                raise ValueError(
                    "col_quant_num_ctas=-1 requires ds3_ep4_v1 backward "
                    "grad-y2 column quantization"
                )
        elif self.col_quant_num_ctas <= 0:
            raise ValueError("col_quant_num_ctas must be positive")
        if self.dgrad_optimization == "ds3_ep4_v1" and self.col_quant_num_ctas != -1:
            raise ValueError(
                "dgrad_optimization='ds3_ep4_v1' requires " "col_quant_num_ctas=-1"
            )
        if self.weight_storage_mode not in ("contiguous", "discrete"):
            raise ValueError(
                "weight_storage_mode must be 'contiguous' or 'discrete', "
                f"got {self.weight_storage_mode!r}"
            )

    @classmethod
    def _for_phase(
        cls,
        config: ResolvedMoeEpConfig,
        *,
        phase: Literal[
            "inference",
            "training_forward",
            "training_backward",
        ],
        launch_cluster_count: int,
        weight_storage_mode: str,
    ) -> "Mxfp8KernelConfig":
        topology = config.topology
        public = config.public_config
        model = public.model
        parallel = public.parallel
        data_path = public.data_path
        if topology.ep_size < 1:
            raise ValueError("MXFP8 execution requires a positive EP size")
        if topology.ep_rank < 0 or topology.ep_rank >= topology.ep_size:
            raise ValueError(
                f"ep_rank {topology.ep_rank} is outside EP size " f"{topology.ep_size}"
            )
        if parallel.max_tokens_per_rank is None:
            raise ValueError("MXFP8 execution requires max_tokens_per_rank")
        if launch_cluster_count <= 0:
            raise ValueError("launch_cluster_count must be positive")
        training = phase != "inference"
        if phase == "inference":
            tuning = public.inference_tuning
        elif phase == "training_forward":
            tuning = public.training_forward_tuning
        else:
            tuning = public.training_backward_tuning
        backward = phase == "training_backward"
        dgrad_optimization = tuning.dgrad_optimization if backward else "baseline"
        token_padding_block = 128 if training else parallel.token_padding_size
        sf_padding_block = 128 if training else parallel.sf_padding_size
        receive_capacity = config.receive_capacity
        physical_recv_pool_size = receive_capacity.physical_recv_pool_rows
        if training:
            logical_route_limit = receive_capacity.training_logical_route_capacity
        else:
            logical_route_limit = receive_capacity.inference_logical_route_capacity
        ds3 = dgrad_optimization == "ds3_ep4_v1"
        return cls(
            num_experts=topology.experts_per_rank,
            world_size=topology.ep_size,
            hidden=model.hidden_size,
            intermediate=model.intermediate_size,
            top_k=model.top_k,
            max_tokens_per_rank=parallel.max_tokens_per_rank,
            gate_up_clamp=data_path.gate_up_clamp,
            generate_c=training,
            physical_recv_pool_size=physical_recv_pool_size,
            max_recv_size_per_rank=logical_route_limit,
            # Token communication must always complete its publish/tail protocol.
            # The public policy is applied after the kernel returns.
            kernel_drop_on_overflow=True,
            enable_col_quant=training,
            dfc2_recompute=backward,
            dfc2_col_output=backward,
            enable_grad_y2_col_quant=backward,
            combine_format=combine_wire_format(data_path.combine_format),
            token_padding_block=token_padding_block,
            sf_padding_block=sf_padding_block,
            sf_vec_size=32,
            group_hint=(
                launch_cluster_count if tuning.group_hint is None else tuning.group_hint
            ),
            token_back_mode=tuning.token_back_mode,
            epi_flag_batch=((4, 2) if ds3 else tuning.epi_flag_batch),
            flag_batch=tuning.token_in_flag_batch,
            fc2_in_kernel_topk_reduce=tuning.reduce_topk_in_kernel,
            weight_storage_mode=weight_storage_mode,
            launch_cluster_count=launch_cluster_count,
            col_quant_num_ctas=(-1 if ds3 else 2368),
            load_balance_mode=(
                "atomic_counter"
                if dgrad_optimization in ("rolling", "ds3_ep4_v1")
                else "static"
            ),
            num_sched_stages=(2 if ds3 else None),
            dgrad_optimization=dgrad_optimization,
        )

    @classmethod
    def for_inference(
        cls,
        config: ResolvedMoeEpConfig,
        *,
        launch_cluster_count: int,
    ) -> "Mxfp8KernelConfig":
        return cls._for_phase(
            config,
            phase="inference",
            launch_cluster_count=launch_cluster_count,
            weight_storage_mode="contiguous",
        )

    @classmethod
    def for_training_forward(
        cls,
        config: ResolvedMoeEpConfig,
        *,
        launch_cluster_count: int,
    ) -> "Mxfp8KernelConfig":
        return cls._for_phase(
            config,
            phase="training_forward",
            launch_cluster_count=launch_cluster_count,
            weight_storage_mode=(
                config.public_config.training_weight_storage_mode.value
            ),
        )

    @classmethod
    def for_training_backward(
        cls,
        config: ResolvedMoeEpConfig,
        *,
        launch_cluster_count: int,
    ) -> "Mxfp8KernelConfig":
        return cls._for_phase(
            config,
            phase="training_backward",
            launch_cluster_count=launch_cluster_count,
            weight_storage_mode=(
                config.public_config.training_weight_storage_mode.value
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
    ) -> tuple[str, tuple[int, int], int, int, bool]:
        """Return the effective rank-independent transport/scheduler knobs."""

        return (
            self.token_back_mode,
            self.epi_flag_batch,
            self.flag_batch,
            self.group_hint,
            self.fc2_in_kernel_topk_reduce,
        )

    def effective_config(self) -> dict[str, object]:
        """Return the complete JSON-safe compile-time configuration."""

        result: dict[str, object] = {}
        for definition in fields(self):
            value = getattr(self, definition.name)
            result[definition.name] = list(value) if isinstance(value, tuple) else value
        return result

    def compile_key(
        self,
        device: torch.device,
        architecture: tuple[int, int],
        layout_signature: tuple,
    ) -> tuple:
        """Return a pointer/stream-independent in-process JIT cache key."""

        canonical_device = torch.device(device)
        return (
            self,
            canonical_device.index,
            architecture,
            layout_signature,
        )


__all__ = ["MXFP8_CLUSTER_SHAPE_MNK", "Mxfp8KernelConfig"]
