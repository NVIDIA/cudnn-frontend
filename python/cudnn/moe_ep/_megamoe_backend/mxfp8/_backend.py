# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Single-rank and EP-subgroup MXFP8 execution backend orchestration."""

from __future__ import annotations

import threading
from dataclasses import replace

import torch
import torch.distributed as dist

from ..._backend import BackendUnavailableError
from ..._contracts import ForwardConfig, ValidatedForwardRequest
from ..._types import MoeEpTrainingWeights
from .._plan import ExecutionPlanOwner
from ._adapter import Mxfp8InputAdapter
from ._backward_compile import prepare_backward_kernel
from ._compile import (
    CompiledMxfp8Kernel,
    PreparedMxfp8Kernel,
    compile_or_get,
    prepare_kernel,
)
from ._config import Mxfp8KernelConfig
from ._launch import launch_forward


class Mxfp8Backend:
    """Own forward/backward executors and per-instance plan resources."""

    def __init__(self, config: ForwardConfig, device: torch.device) -> None:
        self.config = config
        self.device = torch.device(device)
        self.kernel_config = Mxfp8KernelConfig.from_forward_config(config)
        self._adapter = Mxfp8InputAdapter()
        self._prepared_kernel: PreparedMxfp8Kernel | None = None
        self._compiled: CompiledMxfp8Kernel | None = None
        self._plan: ExecutionPlanOwner | None = None
        self._warmed_up = False
        self._closed = False
        self._completion_event: torch.cuda.Event | None = None
        self._completion_recorded = False
        self._device_work_may_be_pending = False
        self._ep_launch_ready = config.ep_size == 1
        self._training_resource_owner = None
        self._lock = threading.RLock()

    @property
    def warmed_up(self) -> bool:
        return self._warmed_up

    @property
    def kernel_fingerprint(self) -> dict | None:
        """Fingerprint of the callable compiled by the most recent launch."""

        if self._compiled is None:
            return None
        return self._compiled.fingerprint

    def _ensure_prepared_kernel(self) -> PreparedMxfp8Kernel:
        if self._prepared_kernel is None:
            try:
                self._prepared_kernel = prepare_kernel(
                    self.config,
                    self.kernel_config,
                    self.device,
                )
            except (ImportError, OSError) as exc:
                raise BackendUnavailableError(
                    "MoeEp MXFP8 backend requires the 'moe_ep' optional "
                    "dependencies and their shared libraries"
                ) from exc
        return self._prepared_kernel

    def _ensure_ep_launch_ready(self, resources, stream) -> None:
        if self._ep_launch_ready:
            return
        # First subgroup launch only: peer metadata writes begin before the
        # kernel's first cross-rank device barrier. Ensure every rank's
        # root-zero and staging work has completed before any rank can issue
        # those writes.
        stream.synchronize()
        if resources.runtime.group is None:
            raise RuntimeError(
                "distributed MXFP8 launch requires a "
                "torch.distributed process group"
            )
        tuning_signature = self.kernel_config.tuning_signature(
            self._ensure_prepared_kernel().launch_cluster_count
        )
        rank_tuning_signatures = [None] * resources.runtime.world_size
        dist.all_gather_object(
            rank_tuning_signatures,
            tuning_signature,
            group=resources.runtime.group,
        )
        if any(
            signature != rank_tuning_signatures[0]
            for signature in rank_tuning_signatures[1:]
        ):
            raise RuntimeError(
                "MoeEp tuning must match on every expert-parallel rank; "
                f"effective signatures by rank: {rank_tuning_signatures}"
            )
        dist.barrier(group=resources.runtime.group)
        self._ep_launch_ready = True

    def forward(self, request: ValidatedForwardRequest):
        with self._lock:
            if self._closed:
                raise RuntimeError("MoeEp MXFP8 backend is closed")
            if request.device != self.device:
                raise ValueError(
                    f"MoeEp MXFP8 backend is bound to {self.device}, "
                    f"got {request.device}"
                )

            with torch.cuda.device(self.device):
                capturing = torch.cuda.is_current_stream_capturing()
                if (
                    capturing
                    and not self._adapter.weights_have_version_counters(
                        request
                    )
                ):
                    raise NotImplementedError(
                        "CUDA graph capture does not support inference tensor "
                        "weights without version counters; eager calls remain "
                        "supported and repack those weights on every call"
                    )
                if capturing and (
                    not self._warmed_up
                    or not self._adapter.has_cached_weights(request)
                ):
                    raise RuntimeError(
                        "MoeEp MXFP8 backend and weights must be warmed up "
                        "before CUDA graph capture"
                    )

                stream = torch.cuda.current_stream(self.device)
                if self._device_work_may_be_pending:
                    torch.cuda.synchronize(self.device)
                    self._device_work_may_be_pending = False
                if self._completion_event is None:
                    self._completion_event = torch.cuda.Event()
                elif self._completion_recorded and not capturing:
                    stream.wait_event(self._completion_event)

                prepared = self._ensure_prepared_kernel()
                if self._plan is None:
                    self._plan = ExecutionPlanOwner(
                        self.config,
                        self.device,
                        prepared.workspace_requirements,
                    )
                device_work_attempted = False
                try:
                    # Allocation zeroing, input staging, weight transforms,
                    # compilation, and launch can all enqueue device work.
                    # Record one completion event even if a later step fails so
                    # a retry on another stream cannot race those writes.
                    device_work_attempted = True
                    resources = self._plan.prepare(request)
                    inputs = self._adapter.stage(
                        request,
                        resources,
                        self.kernel_config,
                        local_workspace_zero_bytes=(
                            prepared.local_workspace_zero_bytes
                        ),
                        shared_workspace_zero_bytes=(
                            prepared.shared_workspace_zero_bytes
                        ),
                        pre_reduced_activation_offset=(
                            prepared.pre_reduced_activation_offset
                        ),
                        pre_reduced_activation_bytes_per_token=(
                            prepared.pre_reduced_activation_bytes_per_token
                        ),
                        pre_reduced_activation_sf_offset=(
                            prepared.pre_reduced_activation_sf_offset
                        ),
                        pre_reduced_activation_sf_bytes_per_token=(
                            prepared.pre_reduced_activation_sf_bytes_per_token
                        ),
                        col_quant_data_rows=prepared.col_quant_data_rows,
                        col_quant_sf_elements=prepared.col_quant_sf_elements,
                        fc1_c=None,
                    )
                    self._compiled = compile_or_get(
                        prepared,
                        inputs,
                        resources,
                    )
                    self._ensure_ep_launch_ready(resources, stream)
                    output = launch_forward(
                        self._compiled,
                        inputs,
                        resources,
                    )
                except (ImportError, OSError) as exc:
                    raise BackendUnavailableError(
                        "MoeEp MXFP8 backend requires the 'moe_ep' optional "
                        "dependencies and their shared libraries"
                    ) from exc
                finally:
                    if device_work_attempted and not capturing:
                        try:
                            self._completion_event.record(stream)
                            self._completion_recorded = True
                            self._device_work_may_be_pending = False
                        except Exception:
                            self._completion_recorded = False
                            self._device_work_may_be_pending = True
                            raise

                self._warmed_up = True
                return output

    def prepare_training_resources(
        self,
        weights: MoeEpTrainingWeights,
        *,
        slot_count: int,
        lane_count: int,
    ):
        """Allocate the fixed slot/lane roots used by the training graph path."""

        with self._lock:
            if self._closed:
                raise RuntimeError("MoeEp MXFP8 backend is closed")
            if self._training_resource_owner is not None:
                raise RuntimeError("MoeEp training resources already exist")
            training_config = replace(
                self.config,
                generate_c=True,
                backward_wgrad_mode="operands",
                token_padding_size=128,
                sf_padding_size=128,
            )
            training_kernel_config = Mxfp8KernelConfig.from_forward_config(
                training_config
            )
            # Graph transport must complete its cross-rank protocol before the
            # frontend applies the public trap/drop policy at graph tail.
            graph_kernel_config = replace(
                training_kernel_config,
                drop_on_overflow=True,
                # Upstream 5b89819's forward col-requant accepts token
                # padding 128/256 but fixes SF atoms at 128; its dGLU
                # auxiliaries require token and SF padding to match. The
                # graph-only fixed-capacity intersection is therefore 128.
                token_padding_block=128,
                sf_padding_block=128,
            )
            forward = prepare_kernel(
                training_config,
                graph_kernel_config,
                self.device,
            )
            backward = prepare_backward_kernel(
                training_config,
                graph_kernel_config,
                self.device,
            )
            from ._training_resources import Mxfp8TrainingResourceOwner

            owner = Mxfp8TrainingResourceOwner(
                training_config,
                self.device,
                forward,
                backward,
                weights,
                slot_count=slot_count,
                lane_count=lane_count,
            )
            try:
                owner.prepare()
            except Exception:
                owner.close()
                raise
            self._training_resource_owner = owner
            return owner

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            with torch.cuda.device(self.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "MoeEp MXFP8 backend cannot be closed during "
                        "CUDA graph capture"
                    )
                if (
                    self._plan is not None
                    or self._training_resource_owner is not None
                ):
                    torch.cuda.synchronize(self.device)
                self._adapter.close()
                if self._training_resource_owner is not None:
                    self._training_resource_owner.close()
                    self._training_resource_owner = None
                if self._plan is not None:
                    self._plan.close()
                    self._plan = None
                self._prepared_kernel = None
                self._compiled = None
                self._completion_event = None
                self._completion_recorded = False
                self._device_work_may_be_pending = False
                self._ep_launch_ready = self.config.ep_size == 1
                self._closed = True


__all__ = ["Mxfp8Backend"]
