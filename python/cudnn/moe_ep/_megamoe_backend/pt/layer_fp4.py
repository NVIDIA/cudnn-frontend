# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MoEEpTrainingLayerFp4 — the EP training layer with simulated-NVFP4
expert GEMMs (optionally + TurboQuant rotation, quantized bprop, stochastic
rounding on grads). See pt/quant.py for what is and isn't modeled.

Drop-in for :class:`pt.layer.MoEEpTrainingLayer`: identical dispatch /
combine / routing and gradient contract; only the expert FFN hook changes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import EpConfig
from .experts_fp4 import grouped_expert_ffn_fp4
from .layer import MoEEpTrainingLayer
from .quant import QuantConfig, make_rotation


class MoEEpTrainingLayerFp4(MoEEpTrainingLayer):
    def __init__(
        self,
        cfg: EpConfig,
        w13: torch.Tensor,
        w2: torch.Tensor,
        qcfg: QuantConfig | None = None,
        comm: str = "torch_dist",
    ):
        super().__init__(cfg, w13, w2, comm)
        self.qcfg = qcfg or QuantConfig()
        if self.qcfg.turboquant:
            if cfg.hidden_size % self.qcfg.rotation_block:
                raise ValueError(
                    f"hidden_size ({cfg.hidden_size}) must be a multiple of "
                    f"rotation_block ({self.qcfg.rotation_block})"
                )
            # Seed-deterministic => identical on every rank, like megamoe.
            self.register_buffer(
                "q_rot",
                make_rotation(
                    self.qcfg.rotation_block,
                    self.qcfg.rotation_seed,
                    device=self.w13.device,
                ),
            )
        else:
            self.q_rot = None

    def _ffn(self, x_grouped: torch.Tensor, tokens_per_expert) -> torch.Tensor:
        return grouped_expert_ffn_fp4(
            x_grouped, tokens_per_expert, self.w13, self.w2, self.qcfg, self.q_rot
        )
