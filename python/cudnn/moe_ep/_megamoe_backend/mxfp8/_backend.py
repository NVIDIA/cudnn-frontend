# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-rank and EP-subgroup MXFP8 execution backend orchestration."""

from __future__ import annotations

import threading

import torch
import torch.distributed as dist

from ..._backend import BackendUnavailableError
from ..._config import ResolvedMoeEpConfig
from ..._contracts import _ForwardCall
from .._plan import _InferenceRuntimeWorkspaceOwner
from ._adapter import Mxfp8InputAdapter
from ._backward_compile import prepare_backward_kernel
from ._compile import (
    CompiledMxfp8Kernel,
    PreparedMxfp8Kernel,
    compile_or_get,
    prepare_environment,
    prepare_kernel,
)
from ._config import Mxfp8KernelConfig
from ._launch import launch_forward


class Mxfp8Backend:
    """Own forward/backward executors and per-instance plan resources."""

    def __init__(
        self,
        config: ResolvedMoeEpConfig,
        device: torch.device,
    ) -> None:
        self._resolved_config = config
        resolved_device = torch.device(device)
        if resolved_device.type != "cuda":
            raise ValueError(
                f"MoeEp MXFP8 backend requires a CUDA device, got {resolved_device}"
            )
        if resolved_device.index is None:
            resolved_device = torch.device("cuda", torch.cuda.current_device())
        self._device = resolved_device
        self._adapter = Mxfp8InputAdapter(
            config.public_config.data_path.fc1_weight_layout
        )
        self._prepare_context: tuple[tuple[int, int], int] | None = None
        self._prepared_kernel: PreparedMxfp8Kernel | None = None
        self._compiled: CompiledMxfp8Kernel | None = None
        self._inference_resources: _InferenceRuntimeWorkspaceOwner | None = None
        self._warmed_up = False
        self._closed = False
        self._completion_event: torch.cuda.Event | None = None
        self._completion_recorded = False
        self._device_work_may_be_pending = False
        self._ep_config_agreed = config.topology.ep_size == 1
        self._ep_launch_ready = config.topology.ep_size == 1
        self._training_state = None
        self._lock = threading.RLock()

    @property
    def warmed_up(self) -> bool:
        return self._warmed_up

    @property
    def resolved_config(self) -> ResolvedMoeEpConfig:
        return self._resolved_config

    @property
    def device(self) -> torch.device:
        return self._device

    def _ensure_prepare_context(self) -> tuple[tuple[int, int], int]:
        if self._prepare_context is None:
            self._prepare_context = prepare_environment(self.device)
        return self._prepare_context

    @property
    def kernel_fingerprint(self) -> dict | None:
        """Fingerprint of the callable compiled by the most recent launch."""

        if self._compiled is None:
            return None
        return self._compiled.fingerprint

    @property
    def execution_fingerprint(self) -> dict | None:
        """Describe kernel identity separately from the source-weight contract."""

        if self._compiled is None:
            return None
        return {
            "kernel": self._compiled.fingerprint,
            "input_contract": {
                "fc1_weight_layout": (
                    self.resolved_config.public_config.data_path.fc1_weight_layout.value
                ),
            },
        }

    def _ensure_prepared_kernel(self) -> PreparedMxfp8Kernel:
        if self._prepared_kernel is None:
            try:
                architecture, launch_cluster_count = self._ensure_prepare_context()
                kernel_config = Mxfp8KernelConfig.for_inference(
                    self.resolved_config,
                    launch_cluster_count=launch_cluster_count,
                )
                self._prepared_kernel = prepare_kernel(
                    self.resolved_config,
                    kernel_config,
                    self.device,
                    architecture=architecture,
                )
            except (ImportError, OSError) as exc:
                raise BackendUnavailableError(
                    "MoeEp MXFP8 backend requires the 'cutedsl' and 'comm' "
                    "optional dependencies and their shared libraries"
                ) from exc
        return self._prepared_kernel

    def _ensure_ep_config_agreed(
        self,
        prepared: PreparedMxfp8Kernel,
        stream,
    ) -> None:
        if self._ep_config_agreed:
            return
        stream.synchronize()
        group = self.resolved_config.public_config.parallel.ep_group
        if group is None:
            raise RuntimeError(
                "distributed MXFP8 launch requires a " "torch.distributed process group"
            )
        capacity = self.resolved_config.receive_capacity
        signature = (
            prepared.config.tuning_signature(),
            capacity.physical_recv_pool_rows,
            capacity.inference_logical_route_capacity,
            capacity.inference_sf_pool_rows,
            prepared.config.token_padding_block,
            prepared.config.sf_padding_block,
        )
        rank_signatures = [None] * self.resolved_config.topology.ep_size
        dist.all_gather_object(
            rank_signatures,
            signature,
            group=group,
        )
        if any(item != rank_signatures[0] for item in rank_signatures[1:]):
            raise RuntimeError(
                "MoeEp inference capacity and tuning must match on every "
                "expert-parallel rank; effective signatures by rank: "
                f"{rank_signatures}"
            )
        self._ep_config_agreed = True

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
                "distributed MXFP8 launch requires a " "torch.distributed process group"
            )
        dist.barrier(group=resources.runtime.group)
        self._ep_launch_ready = True

    def forward(self, request: _ForwardCall):
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
                if capturing and not self._adapter.weights_have_version_counters(
                    request
                ):
                    raise NotImplementedError(
                        "CUDA graph capture does not support inference tensor "
                        "weights without version counters; eager calls remain "
                        "supported and repack those weights on every call"
                    )
                if capturing and (
                    not self._warmed_up or not self._adapter.has_cached_weights(request)
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
                self._ensure_ep_config_agreed(prepared, stream)
                if self._inference_resources is None:
                    self._inference_resources = _InferenceRuntimeWorkspaceOwner(
                        self.resolved_config,
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
                    resources = self._inference_resources.prepare(request)
                    inputs = self._adapter.stage(
                        request,
                        resources,
                        prepared.config,
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
                        "MoeEp MXFP8 backend requires the 'cutedsl' and 'comm' "
                        "optional dependencies and their shared libraries"
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

    def prepare_training(self):
        """Allocate private instance state for stateless training calls."""

        with self._lock:
            if self._closed:
                raise RuntimeError("MoeEp MXFP8 backend is closed")
            if self._training_state is not None:
                raise RuntimeError("MoeEp training is already prepared")
            architecture, launch_cluster_count = self._ensure_prepare_context()
            forward_kernel_config = Mxfp8KernelConfig.for_training_forward(
                self.resolved_config,
                launch_cluster_count=launch_cluster_count,
            )
            backward_kernel_config = Mxfp8KernelConfig.for_training_backward(
                self.resolved_config,
                launch_cluster_count=launch_cluster_count,
            )
            forward = prepare_kernel(
                self.resolved_config,
                forward_kernel_config,
                self.device,
                architecture=architecture,
            )
            backward = prepare_backward_kernel(
                self.resolved_config,
                backward_kernel_config,
                self.device,
                architecture=architecture,
            )
            from ._training_resources import _Mxfp8TrainingState

            state = _Mxfp8TrainingState(
                self.resolved_config,
                self.device,
                forward,
                backward,
            )
            try:
                state.prepare()
            except Exception:
                state.close()
                raise
            self._training_state = state
            return state

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
                    self._inference_resources is not None
                    or self._training_state is not None
                ):
                    torch.cuda.synchronize(self.device)
                self._adapter.close()
                if self._training_state is not None:
                    self._training_state.close()
                    self._training_state = None
                if self._inference_resources is not None:
                    self._inference_resources.close()
                    self._inference_resources = None
                self._prepare_context = None
                self._prepared_kernel = None
                self._compiled = None
                self._completion_event = None
                self._completion_recorded = False
                self._device_work_may_be_pending = False
                self._ep_launch_ready = self.resolved_config.topology.ep_size == 1
                self._closed = True


__all__ = ["Mxfp8Backend"]
