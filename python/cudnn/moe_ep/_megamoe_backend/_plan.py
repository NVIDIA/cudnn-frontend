# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lazy runtime/workspace owner for a compiled MegaMoE execution plan."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import torch

from .._contracts import ForwardConfig, ValidatedForwardRequest
from ._comm import SymmetricMemoryProvider
from ._runtime import RuntimeHandle, RuntimeManager, get_runtime_manager
from ._workspace import (
    LocalMemoryProvider,
    WorkspaceOwner,
    WorkspaceRequirements,
    WorkspaceViews,
)


@dataclass(frozen=True)
class PreparedResources:
    """Resources prepared for staging, compile, and launch integration."""

    runtime: RuntimeHandle
    workspace: WorkspaceViews


class ExecutionPlanOwner:
    """Own runtime and stable workspace without compiling or launching a kernel."""

    def __init__(
        self,
        config: ForwardConfig,
        device: torch.device,
        requirements: WorkspaceRequirements,
        *,
        runtime_manager: Optional[RuntimeManager] = None,
        symmetric_provider: Optional[SymmetricMemoryProvider] = None,
        local_provider: Optional[LocalMemoryProvider] = None,
    ) -> None:
        if config.max_tokens_per_rank != requirements.max_tokens_per_rank:
            raise ValueError(
                "workspace capacity must match ForwardConfig.max_tokens_per_rank"
            )
        self.config = config
        self.device = torch.device(device)
        self.requirements = requirements
        self._runtime_manager = runtime_manager or get_runtime_manager()
        self._symmetric_provider = symmetric_provider
        self._local_provider = local_provider
        self._runtime: Optional[RuntimeHandle] = None
        self._workspace: Optional[WorkspaceOwner] = None
        self._closed = False
        self._cleanup_required = False
        self._lock = threading.RLock()

    @property
    def prepared(self) -> bool:
        return (
            not self._cleanup_required
            and self._runtime is not None
            and self._workspace is not None
            and self._workspace.allocated
        )

    @property
    def cleanup_required(self) -> bool:
        return self._cleanup_required

    @property
    def closed(self) -> bool:
        return self._closed

    def prepare(
        self,
        request: ValidatedForwardRequest,
    ) -> PreparedResources:
        with self._lock:
            if self._closed:
                raise RuntimeError("MegaMoE execution plan is closed")
            if self._cleanup_required:
                raise RuntimeError(
                    "MegaMoE execution plan requires cleanup before prepare"
                )
            if request.config is not self.config:
                raise ValueError("request does not belong to this static plan")
            if torch.device(request.device) != self.device:
                raise ValueError(
                    f"execution plan is bound to {self.device}, got {request.device}"
                )
            if request.token_count > self.requirements.max_tokens_per_rank:
                raise ValueError(
                    f"token count {request.token_count} exceeds "
                    f"max_tokens_per_rank={self.requirements.max_tokens_per_rank}"
                )
            if (
                not self.prepared
                and torch.cuda.is_current_stream_capturing()
            ):
                raise RuntimeError(
                    "MegaMoE runtime/workspace must be warmed up before "
                    "CUDA graph capture"
                )

            if not self.prepared:
                runtime = self._runtime_manager.acquire(self.config, self.device)
                self._runtime = runtime
                try:
                    workspace = WorkspaceOwner(
                        self.requirements,
                        runtime,
                        symmetric_provider=self._symmetric_provider,
                        local_provider=self._local_provider,
                    )
                    self._workspace = workspace
                    views = workspace.views(request.token_count)
                except Exception:
                    try:
                        self._cleanup_failed_prepare()
                    except Exception:
                        self._cleanup_required = True
                        raise
                    raise
            else:
                assert self._runtime is not None
                assert self._workspace is not None
                views = self._workspace.views(request.token_count)

            return PreparedResources(
                runtime=self._runtime,
                workspace=views,
            )

    def _cleanup_failed_prepare(self) -> None:
        if self._workspace is not None:
            self._workspace.close()
            self._workspace = None
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return

            try:
                if self._workspace is not None:
                    self._workspace.close()
                    self._workspace = None
                if self._runtime is not None:
                    self._runtime.close()
                    self._runtime = None
            except Exception:
                self._cleanup_required = True
                raise
            self._cleanup_required = False
            self._closed = True

    def __enter__(self) -> "ExecutionPlanOwner":
        with self._lock:
            if self._closed:
                raise RuntimeError("MegaMoE execution plan is closed")
            return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        del exc_type, exc_value, traceback
        self.close()
        return False


__all__ = ["ExecutionPlanOwner", "PreparedResources"]
