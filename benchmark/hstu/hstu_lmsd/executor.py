# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor setup and precompiled execution paths for HSTU LMSD benchmarks."""

from __future__ import annotations

from typing import Optional

from cuda.bindings import driver as cuda
import torch

from cudnn.hstu.hstu_lmsd import HSTULMSDBwdSm100, HSTULMSDFwdSm100
from cudnn.hstu.hstu_lmsd.cutedsl.cute_dsl_ln_mul_dropout_bwd import (
    TARGET_TILES,
)

from .model_shapes import LMSDShape


class HSTULMSDExecutor:
    """Own all operands, outputs, workspaces, and compiled LMSD APIs.

    Setup, random initialization, allocation, and JIT compilation happen in
    ``__init__`` and are therefore outside every timed region.
    """

    def __init__(
        self,
        shape: LMSDShape,
        *,
        seed: int = 0,
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.shape = shape
        self.seed = int(seed)
        self.device = torch.device("cuda" if device is None else device)
        if self.device.type != "cuda":
            raise ValueError("HSTU LMSD benchmarks require a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        major, minor = torch.cuda.get_device_capability(self.device)
        if major != 10:
            raise RuntimeError(f"HSTU LMSD benchmarks require SM10x, found SM{major}{minor}")

        n = shape.num_rows
        d = shape.hidden_size
        torch.manual_seed(self.seed)
        with torch.cuda.device(self.device):
            self.x = torch.randn((n, d), dtype=torch.bfloat16, device=self.device)
            self._u_storage = torch.randn(
                (n, shape.u_storage_width),
                dtype=torch.bfloat16,
                device=self.device,
            )
            self.u = self._u_storage[:, :d]
            self.weight = torch.randn((d,), dtype=torch.bfloat16, device=self.device)
            self.bias = torch.randn((d,), dtype=torch.bfloat16, device=self.device)
            self.dy = torch.randn((n, 3 * d), dtype=torch.bfloat16, device=self.device)

            self.y = torch.empty((n, 3 * d), dtype=torch.bfloat16, device=self.device)
            self.mean = torch.empty((n,), dtype=torch.float32, device=self.device)
            self.rstd = torch.empty((n,), dtype=torch.float32, device=self.device)
            self.mask = torch.empty((n, d), dtype=torch.int8, device=self.device)
            self.dx = torch.empty((n, d), dtype=torch.bfloat16, device=self.device)
            self.du = torch.empty((n, d), dtype=torch.bfloat16, device=self.device)
            self.dweight = torch.empty((d,), dtype=torch.bfloat16, device=self.device)
            self.dbias = torch.empty((d,), dtype=torch.bfloat16, device=self.device)
            self.dweight_workspace = torch.empty((TARGET_TILES, d), dtype=torch.float32, device=self.device)
            self.dbias_workspace = torch.empty((TARGET_TILES, d), dtype=torch.float32, device=self.device)

        self.forward_api = HSTULMSDFwdSm100(
            sample_x=self.x,
            sample_u=self.u,
            sample_weight=self.weight,
            sample_bias=self.bias,
            sample_y=self.y,
            sample_mean=self.mean,
            sample_rstd=self.rstd,
            sample_mask=self.mask,
            eps=shape.eps,
            dropout_ratio=shape.dropout_ratio,
        )
        self.forward_api.check_support()
        self.forward_api.compile()
        self.forward()

        self.backward_api = HSTULMSDBwdSm100(
            sample_dy=self.dy,
            sample_x=self.x,
            sample_u=self.u,
            sample_weight=self.weight,
            sample_bias=self.bias,
            sample_mean=self.mean,
            sample_rstd=self.rstd,
            sample_mask=self.mask,
            sample_dx=self.dx,
            sample_du=self.du,
            sample_dweight=self.dweight,
            sample_dbias=self.dbias,
            sample_dweight_workspace=self.dweight_workspace,
            sample_dbias_workspace=self.dbias_workspace,
            dropout_ratio=shape.dropout_ratio,
        )
        self.backward_api.check_support()
        self.backward_api.compile()
        self.backward()
        torch.cuda.synchronize(self.device)

    @property
    def backward_chunks(self) -> int:
        return 1

    @property
    def workspace_bytes(self) -> int:
        return 2 * TARGET_TILES * self.shape.hidden_size * 4

    def logical_bytes(self, mode: str) -> int:
        """Return public tensor bytes read or written by one execution.

        Internal dW/dB partial workspaces are deliberately excluded. The
        resulting GB/s is a stable operator-level traffic metric rather than an
        estimate of cache-line traffic inside a particular implementation.
        """

        n = self.shape.num_rows
        d = self.shape.hidden_size
        element_bytes = 2
        forward_bytes = 2 * n * d * element_bytes + 2 * d * element_bytes + 3 * n * d * element_bytes + n * d + 2 * n * 4
        backward_bytes = 5 * n * d * element_bytes + 2 * d * element_bytes + 2 * n * 4 + n * d + 2 * n * d * element_bytes + 2 * d * element_bytes
        if mode == "forward":
            return forward_bytes
        if mode == "backward":
            return backward_bytes
        if mode == "e2e":
            return forward_bytes + backward_bytes
        raise ValueError(f"unsupported benchmark mode: {mode}")

    def forward(
        self,
        stream: Optional[cuda.CUstream | torch.cuda.Stream] = None,
    ) -> None:
        self.forward_api.execute(
            x_tensor=self.x,
            u_tensor=self.u,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            y_tensor=self.y,
            mean_tensor=self.mean,
            rstd_tensor=self.rstd,
            mask_tensor=self.mask,
            seed=self.seed,
            current_stream=stream,
        )

    def backward(
        self,
        stream: Optional[cuda.CUstream | torch.cuda.Stream] = None,
    ) -> None:
        self.backward_api.execute(
            dy_tensor=self.dy,
            x_tensor=self.x,
            u_tensor=self.u,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            mean_tensor=self.mean,
            rstd_tensor=self.rstd,
            mask_tensor=self.mask,
            dx_tensor=self.dx,
            du_tensor=self.du,
            dweight_tensor=self.dweight,
            dbias_tensor=self.dbias,
            dweight_workspace=self.dweight_workspace,
            dbias_workspace=self.dbias_workspace,
            current_stream=stream,
        )

    def e2e(
        self,
        stream: Optional[cuda.CUstream | torch.cuda.Stream] = None,
    ) -> None:
        """Run the complete explicit forward-to-backward operator dataflow."""

        self.forward(stream)
        self.backward(stream)
