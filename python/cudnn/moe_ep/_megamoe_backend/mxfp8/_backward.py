# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Explicit dgrad/dprob Rubin MXFP8 backward orchestration."""

from __future__ import annotations

import torch
import torch.distributed as dist

from ..._backend import BackendUnavailableError
from ..._contracts import ForwardConfig, ValidatedBackwardRequest
from .._plan import ExecutionPlanOwner
from ._backward_compile import (
    CompiledMxfp8BackwardKernel,
    PreparedMxfp8BackwardKernel,
    compile_backward_or_get,
    prepare_backward_kernel,
)
from ._backward_dispatch import Mxfp8BackwardRedispatch
from ._backward_dprob import return_grad_topk_weights
from ._backward_launch import launch_backward_dglu
from ._backward_layout import Mxfp8BackwardLayout
from ._backward_staging import stage_backward
from ._backward_wgrad_export import export_wgrad_operands
from ._config import Mxfp8KernelConfig


class Mxfp8BackwardExecutor:
    """Own only compiled products and reusable capacity workspaces."""

    def __init__(self, config: ForwardConfig, device: torch.device) -> None:
        self.config = config
        self.device = torch.device(device)
        self.kernel_config = Mxfp8KernelConfig.from_forward_config(config)
        self._prepared: PreparedMxfp8BackwardKernel | None = None
        self._compiled: CompiledMxfp8BackwardKernel | None = None
        self._plan: ExecutionPlanOwner | None = None
        self._ep_launch_ready = config.ep_size == 1

    def _ensure_prepared(self) -> PreparedMxfp8BackwardKernel:
        if self._prepared is None:
            try:
                self._prepared = prepare_backward_kernel(
                    self.config,
                    self.kernel_config,
                    self.device,
                )
            except (ImportError, OSError) as exc:
                raise BackendUnavailableError(
                    "MoeEp MXFP8 backward requires the 'moe_ep' optional "
                    "dependencies and their shared libraries"
                ) from exc
        return self._prepared

    def _ensure_ep_launch_ready(self) -> None:
        if self._ep_launch_ready:
            return
        if self.config.ep_group is None:
            raise RuntimeError(
                "distributed MXFP8 backward requires an EP process group"
            )
        torch.cuda.current_stream(self.device).synchronize()
        dist.barrier(group=self.config.ep_group)
        self._ep_launch_ready = True

    def run(
        self,
        request: ValidatedBackwardRequest,
    ):
        prepared = self._ensure_prepared()
        if self._plan is None:
            self._plan = ExecutionPlanOwner(
                self.config,
                self.device,
                prepared.workspace_requirements,
            )

        layout = Mxfp8BackwardLayout.from_request(request)
        redispatched = Mxfp8BackwardRedispatch(request).run()
        resources = self._plan.prepare(request)
        inputs = stage_backward(request, layout, prepared, resources)
        self._compiled = compile_backward_or_get(
            prepared,
            inputs,
            resources,
        )
        self._ensure_ep_launch_ready()
        dglu = launch_backward_dglu(
            self._compiled,
            inputs,
            resources,
        )
        grad_topk_weights = return_grad_topk_weights(
            request,
            redispatched.grad_output,
        )
        if request.config.backward_wgrad_mode == "operands":
            operands = export_wgrad_operands(request, dglu)
            return dglu.grad_activation, grad_topk_weights, operands
        return dglu.grad_activation, grad_topk_weights

    def close(self) -> None:
        if self._plan is not None:
            self._plan.close()
            self._plan = None
        self._prepared = None
        self._compiled = None
        self._ep_launch_ready = self.config.ep_size == 1


__all__ = ["Mxfp8BackwardExecutor"]
