# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared graph-facing execution contract for prepared SDPA plans."""

from typing import TYPE_CHECKING, Any
from cudnn.engines.base import CompiledPlan, ExecutionContext

if TYPE_CHECKING:
    from cudnn._pygraph import pygraph


def _check_workspace(workspace, required: int, name: str) -> None:
    """A FROST executor carves its scratch out of the CALLER's workspace: no
    hidden per-execute allocation, stable pointers, CUDA-graph friendly."""
    if workspace is None:
        raise ValueError(f"{name} needs a {required}-byte workspace; execute() got none — allocate graph.get_workspace_size() bytes and pass it")
    available = workspace.numel() * workspace.element_size() if hasattr(workspace, "numel") else len(workspace)
    if available < required:
        raise ValueError(f"{name} needs a {required}-byte workspace; the buffer provides {available}")


class _FrostSdpaPlan(CompiledPlan):
    """Common SDPA plan workspace, stream and normalized variant-pack lifecycle."""

    def __init__(self, name: str, compiled: Any):
        self._name = name
        self._compiled = compiled
        # The kernel is bound to specific graph tensors; the variant pack the
        # graph API hands us covers every IO tensor of the graph, so key the
        # kernel's own operands out of it by uid (uids are eager and unique).
        self._tensors = list(compiled.binding.bound_tensors())
        # A bound tensor's uid is fixed once the graph is frozen, so read them
        # here rather than re-walking the list on every execute.
        self._uids = [t.get_uid() for t in self._tensors]
        self._workspace_bytes = int(getattr(compiled, "workspace_bytes", 0) or 0)
        # Prepared launch (plan-time argument template, positional tvm-ffi call): binds the graph's
        # normalized VariantPack; without one the plan is handed the raw uid map as before.
        self._prepared = getattr(compiled, "prepared", None)
        self._default_stream = getattr(compiled, "default_stream", None)  # the caller's current stream when the handle carries none
        self.takes_variant_pack = self._prepared is not None
        self._stream_handles: dict = {}
        native = getattr(getattr(self._prepared, "spec", None), "native", None)
        # Only a zero-scratch native launch can bypass the workspace guards
        # below. This is an internal transport hook, not another plan/API step.
        self._ordered_native = native if not self._workspace_bytes and hasattr(native, "execute_ordered") else None

    def get_workspace_size(self) -> int:
        return self._workspace_bytes

    def _stream_handle(self, raw_stream):
        if raw_stream is None:
            # Resolve on every call so capture uses the caller's current stream.
            return self._default_stream() if self._default_stream is not None else None
        cu_stream = self._stream_handles.get(raw_stream)
        if cu_stream is None:
            from cuda.bindings import driver as _drv

            cu_stream = self._stream_handles[raw_stream] = _drv.CUstream(raw_stream)
        return cu_stream

    def _execute_ordered(self, graph, schema, buffers, uids, workspace, override_uids, override_shapes, override_strides, stream):
        launch = self._prepared
        if launch._native_indices is None:
            launch._prepare_indices(graph._slot_of_uid.__getitem__)
        return self._ordered_native.execute_ordered(
            schema,
            buffers,
            uids,
            graph._data_bindings,
            workspace,
            override_uids,
            override_shapes,
            override_strides,
            launch._native_indices,
            self._stream_handle(stream),
        )

    def execute(self, graph: "pygraph", uid_to_data, ctx: ExecutionContext) -> None:
        if self._prepared is not None:
            pack = uid_to_data  # a VariantPack (takes_variant_pack)
            if pack is None:
                raise ValueError(f"{self._name}: the graph could not normalize the variant pack for this plan")
            ws_ptr = 0
            if self._workspace_bytes:
                ws_ptr, nbytes = pack.workspace, pack.workspace_bytes
                if not ws_ptr:
                    raise ValueError(
                        f"{self._name} requires a {self._workspace_bytes}-byte workspace but execute() received none; allocate graph.get_workspace_size() bytes"
                    )
                if nbytes and nbytes < self._workspace_bytes:  # 0: a bare address, size unknown
                    raise ValueError(
                        f"{self._name}: needs a {self._workspace_bytes}-byte workspace, got {nbytes} bytes (size it with graph.get_workspace_size())"
                    )
                device = getattr(ctx.workspace, "__dlpack_device__", None)
                if device is not None and tuple(device()) != (2, self._prepared.spec.device_index):
                    raise ValueError(f"{self._name}: workspace must be on CUDA device {self._prepared.spec.device_index}")
            cu_stream = self._stream_handle(ctx.stream)
            stream_int = int(cu_stream) if cu_stream is not None else 0
            self._prepared.execute(pack, ws_ptr, cu_stream, stream_int)
            return
        self._execute_tensor(uid_to_data, ctx)

    def _execute_tensor(self, uid_to_data, ctx):
        raise NotImplementedError
