# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public API for exact MiniMax Lightning Indexer decode."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from cudnn.api_base import APIBase, TensorDesc, TupleDict

from .kernel import BLOCK_SIZE, HEAD_DIM, TOP_K, decode_host, short_host

_INDEX_HEADS = 4
_MAX_TESTED_K = 32768
_INT32_BYTES = 4
_CACHE_CAPACITY = 32
_WORKSPACE_TOKEN = "_cudnn_lightning_indexer_plan_token"
_WORKSPACE_READY_EVENT = "_cudnn_lightning_indexer_ready_event"
_WORKSPACE_READY_STREAM = "_cudnn_lightning_indexer_ready_stream"
_cache: OrderedDict[tuple, "LightningIndexer"] = OrderedDict()


def _ceil_div(value: int, divisor: int) -> int:
    """Return ceil(value / divisor) for positive integers."""
    return (value + divisor - 1) // divisor


def _require_alignment(tensor: torch.Tensor, alignment: int, name: str) -> None:
    """Reject pointers that do not satisfy the raw-pointer kernel contract."""
    if tensor.data_ptr() % alignment:
        raise ValueError(f"{name} must be {alignment}-byte aligned, got pointer " f"0x{tensor.data_ptr():x}")


def _tensor_key(tensor: torch.Tensor) -> tuple:
    """Return plan-time tensor metadata without retaining its storage."""
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device.type,
        tensor.device.index,
    )


def _as_torch_stream(stream: cuda.CUstream, device: torch.device) -> torch.cuda.Stream:
    """Wrap a launch stream for allocator lifetime tracking."""
    handle = int(stream)
    if handle in (1, 2):
        raise ValueError("CUDA magic stream handles 1 and 2 are not supported; pass 0 or " "an actual stream handle")
    current = torch.cuda.current_stream(device)
    if handle == current.cuda_stream:
        return current
    default = torch.cuda.default_stream(device)
    if handle in (0, default.cuda_stream):
        return default
    return torch.cuda.ExternalStream(handle, device=device)


def _stream_context(stream: Optional[cuda.CUstream], device: torch.device):
    """Place allocations on a real explicit stream without wrapping sentinels."""
    if stream is None:
        return nullcontext()
    handle = int(stream)
    if handle in (1, 2):
        raise ValueError("CUDA magic stream handles 1 and 2 are not supported; pass 0 or " "an actual stream handle")
    torch_stream = _as_torch_stream(stream, device)
    if torch_stream.cuda_stream == torch.cuda.current_stream(device).cuda_stream:
        return nullcontext()
    return torch.cuda.stream(torch_stream)


def _resolve_stream(stream: Optional[cuda.CUstream], device: torch.device) -> cuda.CUstream:
    """Resolve the ambient PyTorch stream on the tensor's device."""
    if stream is not None:
        if int(stream) in (1, 2):
            raise ValueError("CUDA magic stream handles 1 and 2 are not supported; pass 0 " "or an actual stream handle")
        return stream
    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)


def _zero_words_async(pointer: int, count: int, stream: cuda.CUstream) -> None:
    """Stream-order a zero fill without adding a kernel launch."""
    result = cuda.cuMemsetD32Async(int(pointer), 0, int(count), int(stream))
    error = result[0] if isinstance(result, tuple) else result
    if int(error) != 0:
        raise RuntimeError(f"cuMemsetD32Async failed: {error}")


def _launch_on_device(compiled_kernel, device: torch.device, *args) -> None:
    """Launch without paying a device-guard round trip on the common device."""
    if torch.cuda.current_device() == device.index:
        compiled_kernel(*args)
        return
    with torch.cuda.device(device):
        compiled_kernel(*args)


class LightningIndexer(APIBase):
    """Prepared exact static-cache decode plan for the MiniMax-M3 indexer.

    The supported layout is q[B, 1, 4, 128] and k[B, S_k, 1, 128] in BF16.
    position_ids[B, 1] gives an explicit dense K-cache slot. The operation
    returns the current 128-token block plus the 15 highest-scoring completed
    blocks. BF16 Q/K products accumulate in FP32 before block-max reduction.
    The FP32 dot-product accumulations must remain finite. S_k is a fixed
    static-cache capacity for the lifetime of the plan.

    Output order after the current block is deterministic but otherwise not
    part of the contract. block_counts gives the number of left-packed valid
    entries; unused entries are -1.

    execute performs no allocation or host synchronization. Long-context plans
    require an initialized, exclusive caller-owned workspace from
    make_workspace. Initialization records a CUDA event so the first execution
    may safely use a different stream. A successful launch restores its arrival
    counters to zero. Do not share one workspace across overlapping launches.
    """

    def __init__(
        self,
        sample_q: torch.Tensor | TensorDesc,
        sample_k: torch.Tensor | TensorDesc,
        sample_position_ids: torch.Tensor | TensorDesc,
        sample_block_indices: torch.Tensor | TensorDesc,
        sample_block_counts: torch.Tensor | TensorDesc,
    ):
        super().__init__()
        self._warn_experimental_api()
        self.q_desc = self._make_tensor_desc(sample_q, name="sample_q")
        self.k_desc = self._make_tensor_desc(sample_k, name="sample_k")
        self.position_desc = self._make_tensor_desc(sample_position_ids, name="sample_position_ids")
        self.indices_desc = self._make_tensor_desc(sample_block_indices, name="sample_block_indices")
        self.counts_desc = self._make_tensor_desc(sample_block_counts, name="sample_block_counts")
        self.batch_size: Optional[int] = None
        self.k_capacity: Optional[int] = None
        self.num_blocks: Optional[int] = None
        self._short_context = False
        self._workspace_numel = 0
        self._schedule: Optional[tuple[int, ...]] = None
        self._workspace_token = object()

    @property
    def workspace_size(self) -> int:
        """Return required workspace size in bytes."""
        return self._workspace_numel * _INT32_BYTES

    def make_workspace(
        self,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> Optional[torch.Tensor]:
        """Allocate an initialized exclusive workspace for this plan."""
        self._ensure_support_checked()
        if self._workspace_numel == 0:
            return None
        launch_stream = _resolve_stream(current_stream, self.q_desc.device)
        with torch.cuda.device(self.q_desc.device), _stream_context(launch_stream, self.q_desc.device):
            workspace = torch.empty(
                self._workspace_numel,
                dtype=torch.int32,
                device=self.q_desc.device,
            )
        self.initialize_workspace(workspace, launch_stream)
        return workspace

    def initialize_workspace(
        self,
        workspace: torch.Tensor,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        """Reset arrival counters before first use or after a failed launch."""
        self._ensure_support_checked()
        if self._workspace_numel == 0:
            if workspace.numel() != 0:
                raise ValueError("short-context plans require no workspace")
            return
        self._validate_workspace(workspace)
        score_words = self.batch_size * _INDEX_HEADS * self.num_blocks
        counter_pointer = workspace.data_ptr() + score_words * _INT32_BYTES
        launch_stream = _resolve_stream(current_stream, workspace.device)
        torch_stream = _as_torch_stream(launch_stream, workspace.device)
        with torch.cuda.device(workspace.device):
            _zero_words_async(counter_pointer, self.batch_size, launch_stream)
            ready_event = torch.cuda.Event()
            ready_event.record(torch_stream)
        setattr(workspace, _WORKSPACE_TOKEN, self._workspace_token)
        setattr(workspace, _WORKSPACE_READY_EVENT, ready_event)
        setattr(workspace, _WORKSPACE_READY_STREAM, int(launch_stream))
        workspace.record_stream(torch_stream)

    def check_support(self) -> bool:
        """Validate the exact decode tensor and architecture contract."""
        self._value_error_if(
            self.q_desc.ndim != 4,
            f"q must be rank-4 [B, 1, 4, 128], got {self.q_desc.shape}",
        )
        self._value_error_if(
            self.k_desc.ndim != 4,
            f"k must be rank-4 [B, S_k, 1, 128], got {self.k_desc.shape}",
        )
        b, s_q, h_q, d_q = self.q_desc.shape
        b_k, s_k, h_k, d_k = self.k_desc.shape
        self._value_error_if(b < 1, "batch size must be positive")
        self._value_error_if(b > 65535, "batch size must be <= 65535")
        self._value_error_if(
            (s_q, h_q, d_q) != (1, _INDEX_HEADS, HEAD_DIM),
            f"q must have shape [B, 1, 4, 128], got {self.q_desc.shape}",
        )
        self._value_error_if(
            (b_k, h_k, d_k) != (b, 1, HEAD_DIM),
            "k must have shape [B, S_k, 1, 128] with the same batch as q, " f"got {self.k_desc.shape}",
        )
        self._value_error_if(s_k < 1, "S_k must be positive")
        self._value_error_if(
            s_k > _MAX_TESTED_K,
            f"S_k must be <= {_MAX_TESTED_K}, got {s_k}",
        )
        self._check_dtype(self.q_desc, torch.bfloat16, name="q")
        self._check_dtype(self.k_desc, torch.bfloat16, name="k")
        self._check_dtype(self.position_desc, torch.int64, name="position_ids")
        self._check_dtype(self.indices_desc, torch.int32, name="block_indices")
        self._check_dtype(self.counts_desc, torch.int32, name="block_counts")
        self._check_tensor_shape(self.position_desc, (b, 1), name="position_ids")
        self._check_tensor_shape(
            self.indices_desc,
            (b, _INDEX_HEADS, 1, TOP_K),
            name="block_indices",
        )
        self._check_tensor_shape(
            self.counts_desc,
            (b, _INDEX_HEADS, 1),
            name="block_counts",
        )

        devices = {
            self.q_desc.device,
            self.k_desc.device,
            self.position_desc.device,
            self.indices_desc.device,
            self.counts_desc.device,
        }
        self._value_error_if(
            len(devices) != 1,
            "q, k, position_ids, block_indices, and block_counts must share a device",
        )
        self._value_error_if(
            self.q_desc.device.type != "cuda",
            f"LightningIndexer requires CUDA tensors, got {self.q_desc.device}",
        )
        self._runtime_error_if(not torch.cuda.is_available(), "CUDA is not available")
        major, minor = torch.cuda.get_device_capability(self.q_desc.device)
        self._runtime_error_if(
            major < 8,
            f"LightningIndexer requires SM80+, found SM{major}{minor}",
        )

        self._value_error_if(
            (
                self.q_desc.stride[0],
                self.q_desc.stride[2],
                self.q_desc.stride[3],
            )
            != (4 * HEAD_DIM, HEAD_DIM, 1),
            "q must have dense head-major backing storage; " f"got strides {self.q_desc.stride}",
        )
        self._value_error_if(
            (
                self.k_desc.stride[0],
                self.k_desc.stride[1],
                self.k_desc.stride[3],
            )
            != (s_k * HEAD_DIM, HEAD_DIM, 1),
            "k must have dense token-major storage; " f"got strides {self.k_desc.stride}",
        )
        self._value_error_if(
            self.position_desc.stride not in ((0, 1), (1, 1)),
            "position_ids must be contiguous or batch-broadcast with strides " f"(0, 1), got {self.position_desc.stride}",
        )
        self._value_error_if(
            (
                self.indices_desc.stride[0],
                self.indices_desc.stride[1],
                self.indices_desc.stride[3],
            )
            != (_INDEX_HEADS * TOP_K, TOP_K, 1),
            "block_indices must have token-major backing storage; " f"got strides {self.indices_desc.stride}",
        )
        self._value_error_if(
            (self.counts_desc.stride[0], self.counts_desc.stride[1]) != (_INDEX_HEADS, 1),
            "block_counts must have token-major backing storage; " f"got strides {self.counts_desc.stride}",
        )

        props = torch.cuda.get_device_properties(self.q_desc.device)
        num_blocks = _ceil_div(s_k, BLOCK_SIZE)
        self.batch_size = b
        self.k_capacity = s_k
        self.num_blocks = num_blocks
        self._short_context = num_blocks <= TOP_K
        if not self._short_context:
            max_candidates = (s_k - 1) // BLOCK_SIZE
            keys_per_thread = 2 if b * max_candidates >= 1000 else 1
            ctas_per_batch = _ceil_div(max_candidates, keys_per_thread)
            head_split = 1
            dim_slice = HEAD_DIM if b * ctas_per_batch * head_split <= 2 * props.multi_processor_count else 64
            commit_groups = 2 if dim_slice == HEAD_DIM else 1
            self._schedule = (
                max_candidates,
                ctas_per_batch,
                keys_per_thread,
                dim_slice,
                commit_groups,
                head_split,
            )
            score_words = b * _INDEX_HEADS * num_blocks
            self._workspace_numel = score_words + b

        self._is_supported = True
        return True

    def compile(self) -> None:
        """Compile the shape- and schedule-specific launch host."""
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        with torch.cuda.device(self.q_desc.device):
            compile_stream = cuda.CUstream(0)
            if self._short_context:
                self._compiled_kernel = cute.compile(
                    short_host,
                    cutlass.Int64(0),
                    cutlass.Int64(0),
                    cutlass.Int64(0),
                    compile_stream,
                    self.batch_size,
                    self.k_capacity,
                    self.position_desc.stride[0],
                )
                return

            assert self._schedule is not None
            (
                max_candidates,
                ctas_per_batch,
                keys_per_thread,
                dim_slice,
                commit_groups,
                head_split,
            ) = self._schedule
            self._compiled_kernel = cute.compile(
                decode_host,
                cutlass.Int64(0),
                cutlass.Int64(0),
                cutlass.Int64(0),
                cutlass.Int64(0),
                cutlass.Int64(0),
                cutlass.Int64(0),
                cutlass.Int64(0),
                compile_stream,
                self.batch_size,
                self.k_capacity,
                self.position_desc.stride[0],
                max_candidates,
                self.num_blocks,
                ctas_per_batch,
                keys_per_thread,
                dim_slice,
                commit_groups,
                head_split,
            )

    def _validate_runtime_tensor(
        self,
        tensor: torch.Tensor,
        desc,
        name: str,
        alignment: int,
    ) -> None:
        """Validate runtime metadata against the prepared plan."""
        signature = (
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device,
        )
        expected = (desc.shape, desc.stride, desc.dtype, desc.device)
        if signature != expected:
            raise ValueError(f"{name} metadata differs from the compiled plan: " f"expected {expected}, got {signature}")
        _require_alignment(tensor, alignment, name)

    def _validate_workspace(self, workspace: torch.Tensor) -> None:
        """Validate a caller-owned workspace against this plan."""
        if (
            workspace.dtype != torch.int32
            or workspace.ndim != 1
            or not workspace.is_contiguous()
            or workspace.device != self.q_desc.device
            or workspace.numel() < self._workspace_numel
        ):
            raise ValueError("workspace must be contiguous CUDA int32 on the input device " f"with at least {self._workspace_numel} elements")
        _require_alignment(workspace, 16, "workspace")

    def _consume_workspace_initialization(
        self,
        workspace: torch.Tensor,
        launch_stream: cuda.CUstream,
    ) -> None:
        """Make first use wait for initialization without synchronizing the host."""
        if getattr(workspace, _WORKSPACE_TOKEN, None) is not self._workspace_token:
            raise ValueError("workspace is not initialized for this plan; call " "initialize_workspace before first use")
        ready_event = getattr(workspace, _WORKSPACE_READY_EVENT, None)
        if ready_event is None:
            return
        ready_stream = getattr(workspace, _WORKSPACE_READY_STREAM, None)
        if ready_stream != int(launch_stream):
            with torch.cuda.device(workspace.device):
                error, capture_status = cuda.cuStreamIsCapturing(launch_stream)
                if int(error) != 0:
                    raise RuntimeError(f"cuStreamIsCapturing failed: {error}")
                if int(capture_status) != 0:
                    raise ValueError("consume workspace initialization once before cross-stream " "CUDA Graph capture, or initialize on the capture stream")
                _as_torch_stream(launch_stream, workspace.device).wait_event(ready_event)
        setattr(workspace, _WORKSPACE_READY_EVENT, None)
        setattr(workspace, _WORKSPACE_READY_STREAM, None)

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
        block_indices: torch.Tensor,
        block_counts: torch.Tensor,
        workspace: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        """Launch on caller-owned destinations and workspace."""
        self._ensure_support_checked()
        if self._compiled_kernel is None:
            raise ValueError("LightningIndexer kernel is not compiled")
        self._validate_runtime_tensor(q, self.q_desc, "q", 16)
        self._validate_runtime_tensor(k, self.k_desc, "k", 16)
        self._validate_runtime_tensor(position_ids, self.position_desc, "position_ids", 8)
        self._validate_runtime_tensor(block_indices, self.indices_desc, "block_indices", 16)
        self._validate_runtime_tensor(block_counts, self.counts_desc, "block_counts", 16)
        launch_stream = _resolve_stream(current_stream, q.device)
        tensors = [q, k, position_ids, block_indices, block_counts]

        if self._short_context:
            if workspace is not None and workspace.numel() != 0:
                raise ValueError("short-context plans require no workspace")
            _launch_on_device(
                self._compiled_kernel,
                q.device,
                position_ids.data_ptr(),
                block_indices.data_ptr(),
                block_counts.data_ptr(),
                launch_stream,
            )
        else:
            if workspace is None:
                raise ValueError(f"workspace with at least {self.workspace_size} bytes is required")
            self._validate_workspace(workspace)
            self._consume_workspace_initialization(workspace, launch_stream)
            score_words = self.batch_size * _INDEX_HEADS * self.num_blocks
            score_pointer = workspace.data_ptr()
            counter_pointer = score_pointer + score_words * _INT32_BYTES
            _launch_on_device(
                self._compiled_kernel,
                q.device,
                q.data_ptr(),
                k.data_ptr(),
                position_ids.data_ptr(),
                block_indices.data_ptr(),
                block_counts.data_ptr(),
                score_pointer,
                counter_pointer,
                launch_stream,
            )
            tensors.append(workspace)

        consumer = _as_torch_stream(launch_stream, q.device)
        for tensor in tensors:
            tensor.record_stream(consumer)


def lightning_indexer(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    block_indices: Optional[torch.Tensor] = None,
    block_counts: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Run exact MiniMax Lightning Indexer static-cache decode.

    q uses BSHD layout [B, 1, 4, 128], k uses [B, S_k, 1, 128], and
    position_ids[B, 1] is required so dense static-cache buffers stay correct.
    K is a fixed plan capacity; growing DynamicCache tensors are not supported.
    FP32 dot-product accumulations must remain finite. Pass destinations and
    workspace to avoid allocation during repeated calls.
    """
    if (block_indices is None) != (block_counts is None):
        raise ValueError("block_indices and block_counts must both be provided or both omitted")
    b = q.shape[0]
    if block_indices is None:
        with torch.cuda.device(q.device), _stream_context(stream, q.device):
            block_indices = torch.empty(
                (b, 1, _INDEX_HEADS, TOP_K),
                dtype=torch.int32,
                device=q.device,
            ).transpose(1, 2)
            block_counts = torch.empty(
                (b, 1, _INDEX_HEADS),
                dtype=torch.int32,
                device=q.device,
            ).transpose(1, 2)

    assert block_indices is not None
    assert block_counts is not None
    props = torch.cuda.get_device_properties(q.device)
    cache_key = (
        q.device.index,
        props.major,
        props.minor,
        props.multi_processor_count,
        _tensor_key(q),
        _tensor_key(k),
        _tensor_key(position_ids),
        _tensor_key(block_indices),
        _tensor_key(block_counts),
    )
    plan = _cache.get(cache_key)
    if plan is None:
        plan = LightningIndexer(q, k, position_ids, block_indices, block_counts)
        plan.check_support()
        plan.compile()
        _cache[cache_key] = plan
        if len(_cache) > _CACHE_CAPACITY:
            _cache.popitem(last=False)
    else:
        _cache.move_to_end(cache_key)
    if plan.workspace_size:
        if workspace is None:
            workspace = plan.make_workspace(stream)
        elif getattr(workspace, _WORKSPACE_TOKEN, None) is not plan._workspace_token:
            plan.initialize_workspace(workspace, stream)
    plan.execute(
        q,
        k,
        position_ids,
        block_indices,
        block_counts,
        workspace,
        current_stream=stream,
    )
    return TupleDict(
        block_indices=block_indices,
        block_counts=block_counts,
    )


__all__ = ["LightningIndexer", "lightning_indexer"]
