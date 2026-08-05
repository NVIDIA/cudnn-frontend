# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Single-process full-expert MoE with simulated-NVFP4 expert GEMMs.

Quantized twin of :class:`pt.reference.ReferenceMoE` (same op order), used
as the single-GPU vehicle for fp4/turboquant accuracy experiments and as
the oracle for the fp4 EP parity test.
"""

from __future__ import annotations

import torch

from .experts_fp4 import grouped_expert_ffn_fp4
from .quant import QuantConfig, make_rotation
from .reference import ReferenceMoE


class ReferenceMoEFp4(ReferenceMoE):
    def __init__(
        self,
        w13: torch.Tensor,
        w2: torch.Tensor,
        qcfg: QuantConfig | None = None,
    ):
        super().__init__(w13, w2)
        self.qcfg = qcfg or QuantConfig()
        if self.qcfg.turboquant:
            hidden = self.w13.shape[2]
            if hidden % self.qcfg.rotation_block:
                raise ValueError(
                    f"hidden ({hidden}) must be a multiple of "
                    f"rotation_block ({self.qcfg.rotation_block})"
                )
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
