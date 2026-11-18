# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Host entry point of the persistent CuTe DSL BF16 backward kernel.

``backward_bf16`` validates the inputs, prepares the workspace buffers (readiness counters, decoder metadata,
router weights and partials), compiles or reuses the executors, and launches the metadata builder and the
megakernel on the current stream.  There is no host-side expert GEMM and no per-minibatch launch loop.

Geometry (token and expert counts, hidden / intermediate sizes, macrobatch / minibatch sizes, the peer count from
the workspace), strides, schedule contents and clamp values are runtime arguments; the executor cache is keyed by
compiled code only.  The full validation runs when a launch plan is built; later calls check the per-call layouts
against the plan.  Torch and CUTLASS imports stay lazy: importing this module does not load CuTe DSL.
"""

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
import sys
import threading
from typing import Final, Optional
import weakref

from .persistent_bf16_fwd import (
    SHARED_PLACEHOLDER_SHAPE,
    routed_expert_count,
    routed_launch_tensors,
    shared_placeholder,
    shared_presence,
    validate_up_row_offset,
)
from .persistent_bf16_bwd_contract import (
    BWD_EXPERT_TABLE_CAP,
    BwdConfig,
    BwdGeometry,
    DEFAULT_COMM_TILE_SHAPE,
    CommTileShape,
    COUNTER_NAMES,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    MACROBATCH_SIZE,
    METADATA_NAMES,
    MINIBATCH_SIZE,
    NUM_COMM_SMS,
    NUM_LOCAL_EXPERTS,
    NUM_LOCAL_TOKENS,
    RAW13_NAMES,
    SCHEDULE_CAPACITY,
    TOPK,
    TaskDecode,
    TaskPlan,
    counter_descriptors,
    metadata_descriptors,
    raw13_descriptors,
)
from . import _host_common
from ._host_common import has_tail as _has_tail, persistent_executor
from ._host_common import current_cuda_context_key as _current_cuda_context_key
from ._host_common import normalize_situ_beta
from ._host_common import validate_tensor as _validate_tensor
from ._host_common import (
    LaunchPlanCache,
    checked_template,
    current_stream_handle,
    layouts_match,
    tensor_layout,
    typed_key,
    workspace_peer_tables,
)
from ._host_common import current_device_index as _current_device_index, device_capability as _device_capability


from ._arch import COMPILE_OPTIONS as _COMPILE_OPTIONS, TARGET_ARCH, TARGET_CAPABILITY
from ._version_gate import require_cutedsl

REQUIRED_CUDA_MAJOR = 13
ROUTED_DTYPE = "bfloat16"
# Executor cache identity: compiled code only; geometry, strides, schedule
# contents and clamp values are per-call runtime arguments.
_CODEGEN_EPOCH = "bwd-runtime-geometry-dsl48-v2-epsize-runtime-smalltuple-commworkerinterleave-taskdecode-swiglupeek-onegeometry-commtileshape"

_EXECUTOR_CACHE: dict[tuple[int, int, "PersistentBf16BwdPlan"], object] = {}
_EXECUTOR_CACHE_DEVICES: set[int] = set()
_EXECUTOR_CACHE_LOCK = threading.Lock()


class ToolchainMismatchError(RuntimeError):
    """The CUDA runtime, the device or the active CUTLASS DSL differs from what the backward needs."""


@dataclass(frozen=True)
class PersistentBf16BwdPlan:
    arch: str = TARGET_ARCH
    ep_size: int = 8  # runtime peer count from the workspace; any positive value, not a device constant
    num_local_experts: int = NUM_LOCAL_EXPERTS
    num_local_tokens: int = NUM_LOCAL_TOKENS
    hidden_size: int = HIDDEN_SIZE  # shared chain
    routed_hidden_size: Optional[int] = None  # routed chain; None = hidden_size (ordinary MoE)
    intermediate_size: int = INTERMEDIATE_SIZE
    topk: int = TOPK
    schedule_capacity: int = SCHEDULE_CAPACITY
    macrobatch_size: int = MACROBATCH_SIZE
    minibatch_size: int = MINIBATCH_SIZE
    num_comm_sms: int = NUM_COMM_SMS
    routed_dtype: str = ROUTED_DTYPE
    swiglu_limit: Optional[float] = None
    recompute_forward_context: bool = False  # the first macrobatch is replayed too (no routed forward context)
    # The communication tile shape (contract CommTileShape); selects the kernel variant.
    comm_tile_shape: CommTileShape = DEFAULT_COMM_TILE_SHAPE

    def validate(
        self,
        *,
        schedule_capacity: int,
    ) -> BwdGeometry:
        if self.arch != TARGET_ARCH or self.routed_dtype != ROUTED_DTYPE:
            raise NotImplementedError(f"persistent BF16 backward requires {TARGET_ARCH}/BF16; got " f"{self.arch}/{self.routed_dtype}")
        if type(self.ep_size) is not int or not 0 < self.ep_size < 2**31:
            raise ValueError(f"ep_size must be a positive Int32; got {self.ep_size!r}")
        if schedule_capacity != self.schedule_capacity:
            raise ValueError(f"schedule capacity {schedule_capacity} differs from plan {self.schedule_capacity}")
        return BwdGeometry(
            schedule_capacity=schedule_capacity,
            num_local_tokens=self.num_local_tokens,
            macrobatch_size=self.macrobatch_size,
            minibatch_size=self.minibatch_size,
            num_comm_sms=self.num_comm_sms,
            num_local_experts=self.num_local_experts,
            hidden_size=self.hidden_size,
            routed_hidden_size=self.routed_hidden_size,
            intermediate_size=self.intermediate_size,
            topk=self.topk,
            swiglu_limit=self.swiglu_limit,
            ep_size=self.ep_size,
            recompute_forward_context=self.recompute_forward_context,
            comm_tile_shape=self.comm_tile_shape,
        )


@dataclass(frozen=True)
class PersistentBf16BwdState:
    """Raw13, readiness, fixed decoder metadata, and the configured ring."""

    raw13: tuple[object, ...]
    counters: tuple[object, ...]
    metadata: tuple[object, ...]
    router_weights: object
    d_router_weight_partials: object
    # One zero-filled int32 allocation behind ``counters`` (views on 16-byte boundaries),
    # so a reused state is re-zeroed with one memset; None for caller-built states.
    counter_storage: object = None

    def __post_init__(self) -> None:
        if len(self.raw13) != len(RAW13_NAMES):
            raise ValueError("raw13 must contain exactly thirteen tensors")
        if len(self.counters) != len(COUNTER_NAMES):
            raise ValueError(f"counters must contain all {len(COUNTER_NAMES)} readiness arrays")
        if len(self.metadata) != len(METADATA_NAMES):
            raise ValueError("metadata must contain all four fixed decoder arrays")

    def raw(self, name: str):
        try:
            return self.raw13[RAW13_NAMES.index(name)]
        except ValueError as error:
            raise KeyError(name) from error

    def counter(self, name: str):
        try:
            return self.counters[COUNTER_NAMES.index(name)]
        except ValueError as error:
            raise KeyError(name) from error

    def decoder(self, name: str):
        try:
            return self.metadata[METADATA_NAMES.index(name)]
        except ValueError as error:
            raise KeyError(name) from error


# Every launch-plan cache held by a workspace's backward preparation, so that clearing the executors drops them too.
_PLAN_CACHES: "weakref.WeakSet[LaunchPlanCache]" = weakref.WeakSet()


def _new_plan_cache() -> LaunchPlanCache:
    cache = LaunchPlanCache()
    _PLAN_CACHES.add(cache)
    return cache


@dataclass(frozen=True)
class _PersistentBf16BwdPreparation:
    """The small buffers one workspace keeps between backward calls, and the launch plans built on
    them (see backward_bf16), which live and die with them: the readiness counters and the fixed
    decoder metadata (schedule-capacity sized; schedule values are never cached), and the router
    weights and their partials (macrobatch-sized: macrobatch_size x 4 B plus macrobatch_size x
    intermediate_size / 128 x 4 B per rank).  The scratch (raw13 0-6: d_x_shared, d_gate_shared and
    d_up_shared are num_local_tokens-sized; d_x_routed, d_gate_routed, d_up_routed and d_y_routed
    macrobatch-sized) and the six dW outputs are fresh allocations on every call (_allocate_raw13)."""

    key: tuple[object, ...]
    counters: tuple[object, ...]
    metadata: tuple[object, ...]
    router_weights: object
    d_router_weight_partials: object
    counter_storage: object
    launch_plans: object = field(default_factory=_new_plan_cache, compare=False)
    # Raw handles of the streams these buffers have been used on.  They outlive a call and are
    # released when the geometry changes, so on their first use on another stream they are recorded
    # on that stream once: the allocator then keeps them until that stream's work completes.
    streams: set = field(default_factory=set, compare=False)

    def buffers(self) -> tuple[object, ...]:
        return (self.counter_storage, *self.metadata, self.router_weights, self.d_router_weight_partials)


#: Backward configurations one workspace keeps prepared at once.  get_workspace shares a workspace
#: across a model's MoE layers, which may differ (E, I, SwiGLU clamp, shared expert, main_grad dtype,
#: split weights, ...); one buffer set per configuration keeps them from reallocating each other's on
#: every call, and keeps the buffers a CUDA graph captured alive while other layers run.
_MAX_PREPARATIONS = 8


def _workspace_preparations(workspace) -> dict:
    """The workspace's prepared backward configurations, by preparation key, oldest first."""
    preparations = getattr(workspace, "_cutedsl_bwd_preparations", None)
    if preparations is None:
        preparations = {}
        workspace._cutedsl_bwd_preparations = preparations
    return preparations


@dataclass(frozen=True)
class PersistentBf16BwdInputs:
    """FWD-saved activations, weights, schedule, and symmetric peer pointers."""

    x_shared: object
    x_routed: object
    gate_shared: object
    gate_routed: object
    up_shared: object
    up_routed: object
    hidden_shared: object
    hidden_routed: object
    d_y_shared: object
    w_shared_gate: object
    w_routed_gate: object
    w_shared_up: object
    w_routed_up: object
    w_shared_down: object
    w_routed_down: object
    schedule_peer_rank: object
    schedule_peer_token_idx: object
    num_tokens: object
    tokens_per_expert: object
    x_peer_ptrs: tuple[int, ...]
    d_y_peer_ptrs: tuple[int, ...]
    d_x_peer_ptrs: tuple[int, ...]
    router_peer_ptrs: tuple[int, ...]
    d_router_peer_ptrs: tuple[int, ...]


def validate_toolchain_snapshot(
    *,
    cuda_version: str,
    capability: tuple[int, int],
    cutlass_origins: tuple[str, ...] = (),
) -> None:
    """Torch's CUDA runtime, the device and the single active CUTLASS DSL origin.  The CuTe DSL version
    and its compiler flavor (cu13) are checked by ``_version_gate.require_cutedsl``."""
    if not isinstance(cuda_version, str) or not cuda_version:
        raise ToolchainMismatchError("Torch must report a CUDA 13 runtime")
    try:
        cuda_major = int(cuda_version.split(".", 1)[0])
    except (TypeError, ValueError) as error:
        raise ToolchainMismatchError(f"invalid CUDA version: {cuda_version!r}") from error
    if cuda_major != REQUIRED_CUDA_MAJOR:
        raise ToolchainMismatchError(f"requires CUDA {REQUIRED_CUDA_MAJOR}; got {cuda_version}")
    if capability != TARGET_CAPABILITY:
        raise ToolchainMismatchError(f"requires {TARGET_ARCH} (compute capability {TARGET_CAPABILITY}); got {capability}")

    normalized = tuple(str(Path(origin).resolve()) for origin in cutlass_origins if origin)
    if len(set(normalized)) != 1:
        raise ToolchainMismatchError("exactly one active CUTLASS DSL origin is required")


def _runtime_toolchain_snapshot(device) -> None:
    import torch

    spec = importlib.util.find_spec("cutlass")
    roots: list[str] = []
    if spec is not None:
        candidates = []
        if spec.origin:
            candidates.append(spec.origin)
        if spec.submodule_search_locations:
            candidates.extend(str(item) for item in spec.submodule_search_locations)
        for candidate in candidates:
            path = Path(candidate).resolve()
            roots.append(str(path.parent if path.name == "__init__.py" else path))
    validate_toolchain_snapshot(
        cuda_version=torch.version.cuda,
        capability=_device_capability(device),
        cutlass_origins=tuple(dict.fromkeys(roots)),
    )


def _prepare_public_runtime_environment() -> None:
    """The version gate (it runs once per process).  The target and TVM FFI are compile options
    (``_arch.COMPILE_OPTIONS``, part of the executor key), not environment variables."""
    require_cutedsl()


def prepare_persistent_bwd_state(
    *,
    config: BwdGeometry,
    device,
    num_tokens,
    main_grads=None,
    combined_fc1: bool = False,
    has_shared: bool = True,
    raw=None,
    storage=None,
) -> PersistentBf16BwdState:
    """Allocate raw13/readiness and capacity-only, NOT READY decoder metadata.

    Only the complete public launch initializes metadata via the device builder.
    For C<=B the last two entries alias num_tokens.  Routed dW is uninitialized; the mega
    writes every routed dW tile itself (no zero kernel).

    The execute step of a plan passes ``raw`` (its raw13, see ``_caller_raw13``) and ``storage``, the plan's
    ``(int32, float32)`` buffers (``backward_bf16_storage_lengths``): the counters, the decoder metadata, the
    router weights and their partials are then leading views of those, and nothing is allocated.  Every call
    zeroes the counters, rebuilds the metadata and rewrites the router weights and partials before reading
    them, so the views of different geometries may share the buffers.
    """

    import torch

    plan = TaskPlan(config)
    if raw is None:
        raw = _allocate_raw13(
            tuple(item.shape for item in raw13_descriptors(config)), torch.empty(0, dtype=torch.bfloat16, device=device), main_grads, combined_fc1, has_shared
        )
    counter_shapes = tuple(item.shape for item in counter_descriptors(plan))
    offsets, total = [], 0
    for shape in counter_shapes:
        offsets.append(total)
        numel = 1
        for extent in shape:
            numel *= extent
        total += -(-numel // 4) * 4
    metadata_shapes = tuple(item.shape if index < 2 or config.build_o1_decode else None for index, item in enumerate(metadata_descriptors(plan)))
    partials_shape = (config.macrobatch_size, config.intermediate_size // 128)
    if storage is None:
        counter_storage = torch.zeros(max(total, 4), dtype=torch.int32, device=device)
        decoder_metadata = tuple(num_tokens if shape is None else torch.empty(shape, dtype=torch.int32, device=device) for shape in metadata_shapes)
        router_weights = torch.empty((config.macrobatch_size,), dtype=torch.float32, device=device)
        d_router_weight_partials = torch.empty(partials_shape, dtype=torch.float32, device=device)
    else:
        words, floats = storage
        counter_storage, used = words[: max(total, 4)], max(total, 4)
        decoder_metadata = []
        for shape in metadata_shapes:
            if shape is None:
                decoder_metadata.append(num_tokens)
                continue
            decoder_metadata.append(words[used : used + shape[0]])
            used += -(-shape[0] // 4) * 4  # every view starts on a 16-byte boundary
        decoder_metadata = tuple(decoder_metadata)
        partials_offset = -(-config.macrobatch_size // 4) * 4
        partials_end = partials_offset + partials_shape[0] * partials_shape[1]
        if used > words.numel() or partials_end > floats.numel():
            raise ValueError(
                f"this backward geometry needs {used} int32 and {partials_end} float32 words of plan storage; "
                f"the plan has {words.numel()} and {floats.numel()} (it was made for a smaller schedule "
                "capacity or macrobatch size)"
            )
        router_weights = floats[: config.macrobatch_size]
        d_router_weight_partials = floats[partials_offset:partials_end].view(partials_shape)
    counters = tuple(
        counter_storage.narrow(0, offset, total_numel).view(shape)
        for offset, shape, total_numel in zip(offsets, counter_shapes, (int(torch.Size(shape).numel()) for shape in counter_shapes))
    )
    return PersistentBf16BwdState(
        raw,
        counters,
        decoder_metadata,
        router_weights,
        d_router_weight_partials,
        counter_storage,
    )


def _validate_persistent_bwd_abi(
    inputs: PersistentBf16BwdInputs,
    state: PersistentBf16BwdState,
    *,
    config: BwdGeometry,
    routed_storage_tables=None,
    up_row_offset: int = 0,
    main_grad_dtype=None,
    main_grad_storage_tables=None,
    main_grad_up_row_offset: int = 0,
    has_shared: bool = True,
) -> None:
    """Check dynamic tensor/pointer ABI without reading any device contents.

    ``routed_storage_tables`` are the Gate, Up and Down per-expert address tables when the routed
    experts are independent per-expert allocations; the weights are then one expert's own matrix
    and the dgrad names the others through per-expert TMA descriptors.  ``up_row_offset`` is the
    Up half's first row inside a combined FC1 parameter.
    """

    import torch

    if not isinstance(inputs, PersistentBf16BwdInputs):
        raise TypeError("inputs must be PersistentBf16BwdInputs")
    if not isinstance(state, PersistentBf16BwdState):
        raise TypeError("state must be PersistentBf16BwdState")
    if not isinstance(inputs.tokens_per_expert, torch.Tensor):
        raise TypeError("inputs.tokens_per_expert must be a torch.Tensor")
    if not isinstance(inputs.num_tokens, torch.Tensor):
        raise TypeError("inputs.num_tokens must be a torch.Tensor")
    if not inputs.tokens_per_expert.is_cuda or not inputs.num_tokens.is_cuda:
        raise ValueError("schedule metadata must be CUDA tensors")
    device = inputs.tokens_per_expert.device
    _validate_tensor(
        "tokens_per_expert",
        inputs.tokens_per_expert,
        shape=(config.num_local_experts,),
        dtype=torch.int32,
        device=device,
    )
    _validate_tensor(
        "num_tokens",
        inputs.num_tokens,
        shape=(1,),
        dtype=torch.int32,
        device=device,
    )
    task_plan = TaskPlan(config)
    # Split allocations: one expert's own matrix per routed weight.  Grouped: the [E, ...]
    # parameter.  Split weights come with split (per-expert) main_grad buffers.
    split_weights = routed_storage_tables is not None
    if (main_grad_storage_tables is not None) != split_weights:
        raise ValueError("split routed weights require three split main-grad storage tables")
    if split_weights and main_grad_dtype is None:
        raise ValueError("split routed weights require fused main-grad accumulation")
    for tables, label in ((routed_storage_tables, "routed_storage_tables"), (main_grad_storage_tables, "main_grad_storage_tables")):
        if tables is None:
            continue
        if len(tables) != 3:
            raise ValueError(f"{label} must hold the Gate, Up and Down per-expert address tables")
        for name, table in zip(("gate", "up", "down"), tables):
            _validate_tensor(f"{label}.{name}", table, shape=(config.num_local_experts,), dtype=torch.int64, device=inputs.tokens_per_expert.device)
    # A combined FC1 is the whole [E, 2I, H] parameter (or [2I, H] per split expert) under both
    # names, Up up_row_offset = I rows down.
    combined = validate_up_row_offset(up_row_offset, config.intermediate_size, inputs.w_routed_gate, inputs.w_routed_up)
    fc1_rows = (2 if combined else 1) * config.intermediate_size
    fc1_shape = (fc1_rows, config.routed_hidden_size) if split_weights else (config.num_local_experts, fc1_rows, config.routed_hidden_size)
    down_shape = (
        (config.routed_hidden_size, config.intermediate_size)
        if split_weights
        else (config.num_local_experts, config.routed_hidden_size, config.intermediate_size)
    )

    for name in (
        "x_peer_ptrs",
        "d_y_peer_ptrs",
        "d_x_peer_ptrs",
        "router_peer_ptrs",
        "d_router_peer_ptrs",
    ):
        pointers = getattr(inputs, name)
        if (
            not isinstance(pointers, tuple)
            or len(pointers) != config.ep_size
            or any(type(pointer) is not int or pointer <= 0 or pointer % 16 for pointer in pointers)
        ):
            raise ValueError(f"{name} must contain ep_size={config.ep_size} aligned device addresses")

    # A routed-only layer passes the placeholder for every shared context tensor and weight.
    t_i = (config.num_local_tokens, config.intermediate_size) if has_shared else SHARED_PLACEHOLDER_SHAPE
    i_h = (config.intermediate_size, config.hidden_size) if has_shared else SHARED_PLACEHOLDER_SHAPE
    h_i = (config.hidden_size, config.intermediate_size) if has_shared else SHARED_PLACEHOLDER_SHAPE
    bf16_specs = (
        ("x_shared", inputs.x_shared, (config.num_local_tokens, config.hidden_size)),
        ("x_routed", inputs.x_routed, (config.macrobatch_size, config.routed_hidden_size)),
        ("gate_shared", inputs.gate_shared, t_i),
        (
            "gate_routed",
            inputs.gate_routed,
            (config.macrobatch_size, config.intermediate_size),
        ),
        ("up_shared", inputs.up_shared, t_i),
        ("up_routed", inputs.up_routed, (config.macrobatch_size, config.intermediate_size)),
        ("hidden_shared", inputs.hidden_shared, t_i),
        (
            "hidden_routed",
            inputs.hidden_routed,
            (config.macrobatch_size, config.intermediate_size),
        ),
        ("d_y_shared", inputs.d_y_shared, (config.num_local_tokens, config.hidden_size)),
        ("w_shared_gate", inputs.w_shared_gate, i_h),
        ("w_routed_gate", inputs.w_routed_gate, fc1_shape),
        ("w_shared_up", inputs.w_shared_up, i_h),
        ("w_routed_up", inputs.w_routed_up, fc1_shape),
        ("w_shared_down", inputs.w_shared_down, h_i),
        ("w_routed_down", inputs.w_routed_down, down_shape),
    )
    for name, tensor, shape in bf16_specs:
        # The routed weights (w_routed_*) may be slices of a larger parameter; the activations and
        # context named routed_* here are this call's own allocations and stay contiguous.
        _validate_tensor(name, tensor, shape=shape, dtype=torch.bfloat16, device=device, sliced=name.startswith("w_routed_"))

    for name, tensor, shape in (
        ("schedule_peer_rank", inputs.schedule_peer_rank, (config.schedule_capacity,)),
        (
            "schedule_peer_token_idx",
            inputs.schedule_peer_token_idx,
            (config.schedule_capacity,),
        ),
    ):
        _validate_tensor(name, tensor, shape=shape, dtype=torch.int32, device=device)

    # The six weight gradients are the caller's main_grad buffers when the epilogue accumulates into
    # them, and those are float32 unless the caller keeps bfloat16 ones.  Everything else in raw13
    # is this backward's own bfloat16 scratch.
    wgrad_dtype = main_grad_dtype or torch.bfloat16
    # Routed Gate and Up gradients: one combined buffer under both slots, Up main_grad_up_row_offset
    # = I rows down (a combined main_grad, or this call's own for a combined FC1), or two
    # separate I-row ones.  Split: one expert's own allocation, the representative.
    if main_grad_up_row_offset not in (0, config.intermediate_size):
        raise ValueError("main_grad_up_row_offset must be 0 or intermediate_size")
    if (state.raw13[8] is state.raw13[10]) != bool(main_grad_up_row_offset):
        raise ValueError("main_grad_up_row_offset is intermediate_size exactly when the routed Gate " "and Up gradients are one buffer")
    wgrad_fc1 = (config.intermediate_size + main_grad_up_row_offset, config.routed_hidden_size)
    wgrad_down = (config.routed_hidden_size, config.intermediate_size)
    wgrad_shapes = {
        8: wgrad_fc1 if split_weights else (config.num_local_experts, *wgrad_fc1),
        10: wgrad_fc1 if split_weights else (config.num_local_experts, *wgrad_fc1),
        12: wgrad_down if split_weights else (config.num_local_experts, *wgrad_down),
    }
    if not has_shared:
        wgrad_shapes.update((index, SHARED_PLACEHOLDER_SHAPE) for index in SHARED_RAW13)
    for index, (descriptor, tensor) in enumerate(zip(raw13_descriptors(config), state.raw13)):
        _validate_tensor(
            f"state.{descriptor.name}",
            tensor,
            shape=wgrad_shapes.get(index, descriptor.shape),
            dtype=wgrad_dtype if index >= 7 else torch.bfloat16,
            device=device,
        )
    for descriptor, tensor in zip(counter_descriptors(task_plan), state.counters):
        _validate_tensor(
            f"state.{descriptor.name}",
            tensor,
            shape=descriptor.shape,
            dtype=torch.int32,
            device=device,
        )
    for descriptor, tensor in zip(metadata_descriptors(task_plan), state.metadata):
        _validate_tensor(
            f"state.{descriptor.name}",
            tensor,
            shape=descriptor.shape,
            dtype=torch.int32,
            device=device,
        )
    _validate_tensor(
        "state.router_weights",
        state.router_weights,
        shape=(config.macrobatch_size,),
        dtype=torch.float32,
        device=device,
    )
    _validate_tensor(
        "state.d_router_weight_partials",
        state.d_router_weight_partials,
        shape=(config.macrobatch_size, config.intermediate_size // 128),
        dtype=torch.float32,
        device=device,
    )


def _validate_persistent_bwd_schedule_values(
    inputs: PersistentBf16BwdInputs,
    *,
    num_tokens: int,
    config: BwdConfig,
) -> None:
    """Validate peer selectors and valid-prefix token indices."""

    import torch

    # The capacity tail is never a real peer. Valid-prefix padding may also be
    # -1; its peer_token_idx lane is deliberately ignored.
    peer_rank = inputs.schedule_peer_rank
    if bool(torch.any(peer_rank < -1).item()) or bool(torch.any(peer_rank >= config.ep_size).item()):
        raise ValueError(f"schedule peer ranks must be -1 or a rank below ep_size={config.ep_size}")
    if bool(torch.any(peer_rank[num_tokens:] != -1).item()):
        raise ValueError("schedule capacity tail must use peer_rank=-1")
    valid_peer = peer_rank[:num_tokens] >= 0
    valid_indices = inputs.schedule_peer_token_idx[:num_tokens][valid_peer]
    if bool(torch.any(valid_indices < 0).item()) or bool(torch.any(valid_indices >= config.num_local_tokens * config.topk).item()):
        raise ValueError("valid peers require flattened token*topk route indices")


# Launch plans (as in the FWD host), keyed by the workspace
# buffers, the weights and the geometry and kept on the workspace's backward preparation they were
# built on: the validation and the executors are resolved when a plan is built.  A call checks the
# schedule, the forward context (new tensors every step) and the shared x / dY inputs against the
# validated layouts, allocates this call's raw13 (scratch and dW outputs), fills the per-call slots
# and launches.  The templates hold only the preparation's buffers (counters, decoder metadata,
# router weights and partials), peer tables and scalars, never a per-call or caller-owned tensor.
_FORWARD_CONTEXT_NAMES: Final = (
    "x_routed",
    "gate_shared",
    "gate_routed",
    "up_shared",
    "up_routed",
    "hidden_shared",
    "hidden_routed",
)


@dataclass(frozen=True)
class _BwdLaunchPlan:
    executor: object
    template: tuple[object, ...]  # mega launch arguments with every per-call slot emptied
    metadata_executor: object
    metadata_template: tuple[object, ...]  # metadata-builder arguments without the stream
    schedule_layouts: tuple[object, ...]  # peer_rank, peer_token_idx, num_tokens, tokens_per_expert
    context_layouts: tuple[object, ...]  # the seven forward-context activations
    shared_input_layouts: tuple[object, ...]  # x_shared, d_y_shared (the workspace x / dY buffers by default)
    raw13_shapes: tuple[tuple[int, ...], ...]  # the per-call scratch (raw13 0-6) and the six dW outputs (7-12)
    raw13_proto: object  # empty bfloat16 tensor on the device: dtype / device source of the per-call new_empty
    metadata_count: int
    alias_slots: tuple[int, ...]  # mega slots of the decoder-metadata num_tokens aliases
    metadata_alias_slots: tuple[int, ...]
    # The mega and metadata modules import CUTLASS lazily, so the plan keeps their slot fillers and
    # a call needs no import statement.
    fill_slots: object
    fill_metadata_slots: object
    # The routed address table and descriptor bytes (RoutedDescriptors), built by the first launch.
    routed: object = None


def select_task_decode(num_local_experts: int, requested: TaskDecode | None = None) -> TaskDecode:
    """Task->expert decode variant of backward_bf16: the forward's rule over the BWD table
    capacity (SMEM_BINARY for E <= BWD_EXPERT_TABLE_CAP, SMEM_GROUP above it).
    Internal: ``requested`` forces a variant.  This function is the only selection point of
    backward_bf16."""

    if type(num_local_experts) is not int or num_local_experts <= 0:
        raise ValueError("num_local_experts must be a positive int")
    if requested is None:
        return TaskDecode.SMEM_BINARY if num_local_experts <= BWD_EXPERT_TABLE_CAP else TaskDecode.SMEM_GROUP
    chosen = TaskDecode(requested)
    if chosen is TaskDecode.SMEM_BINARY and num_local_experts > BWD_EXPERT_TABLE_CAP:
        raise ValueError(f"TaskDecode.SMEM_BINARY holds at most {BWD_EXPERT_TABLE_CAP} local experts, got {num_local_experts}")
    return chosen


def _executor_cache_key(
    device_index,
    context,
    plan,
    inputs,
    task_decode: TaskDecode,
    *,
    split_weights: bool = False,
    main_grad_dtype=None,
    is_situ: bool = False,
    has_shared: bool = True,
    has_tail: bool = False,
    recompute_forward_context: bool = False,
):
    # Dimensions, strides, capacities and clamp values are launch metadata.
    # Public sources are contiguous and normalized to row-major metadata before
    # the fixed transpose/permutation ABI. Singleton input strides are not code.
    tensors = tuple(
        getattr(inputs, name)
        for name in (
            "x_shared",
            "x_routed",
            "gate_shared",
            "gate_routed",
            "up_shared",
            "up_routed",
            "hidden_shared",
            "hidden_routed",
            "d_y_shared",
            "w_shared_gate",
            "w_routed_gate",
            "w_shared_up",
            "w_routed_up",
            "w_shared_down",
            "w_routed_down",
            "schedule_peer_rank",
            "schedule_peer_token_idx",
            "num_tokens",
            "tokens_per_expert",
        )
    )
    abi = tuple((str(t.dtype), len(t.shape), "normalized-row-major-source") for t in tensors)
    return (
        _CODEGEN_EPOCH,
        _COMPILE_OPTIONS,
        device_index,
        context,
        plan.arch,
        plan.routed_dtype,
        "minibatch_release",
        plan.swiglu_limit is not None,
        ("task_decode", TaskDecode(task_decode).name),
        # The routed operands' rank differs between the two layouts and the ABI above keeps
        # the rank, but the flag is what selects the code, so key it directly.
        ("routed_weights", "split" if split_weights else "grouped"),
        # Fused wgrad accumulation and its destination dtype are compile-time, so they belong
        # in the key.
        ("main_grad", "none" if main_grad_dtype is None else str(main_grad_dtype)),
        # SiTU-GLU is compile-time too; its betas are runtime arguments.
        ("situ", bool(is_situ)),
        # Compile-time: a routed-only layer has no shared tasks.
        ("has_shared", bool(has_shared)),
        # Compile-time as well: a 128-wide H / I tail tile.
        ("has_tail", bool(has_tail)),
        # Compile-time: the first macrobatch is replayed like the later ones.
        ("recompute_forward_context", bool(recompute_forward_context)),
        # Compile-time: the rows and columns of one communication tile.
        ("comm_tile_shape", (plan.comm_tile_shape.rows, plan.comm_tile_shape.columns)),
        abi,
    )


def clear_persistent_bf16_bwd_executor_cache(*, synchronize: bool = True) -> None:
    """Release compiled backward executors after their launches are quiescent."""

    import torch

    with _EXECUTOR_CACHE_LOCK:
        if synchronize:
            for device_index in sorted(_EXECUTOR_CACHE_DEVICES):
                torch.cuda.synchronize(device_index)
        _EXECUTOR_CACHE.clear()
        _EXECUTOR_CACHE_DEVICES.clear()
        for plans in list(_PLAN_CACHES):  # the plans on every workspace's preparation hold executors too
            plans.clear()
        metadata_module = sys.modules.get(f"{__package__}._persistent_bf16_bwd_metadata")
        if metadata_module is not None:
            metadata_module.clear_bwd_metadata_cache()


def _build_launch_plan(
    workspace,
    schedule,
    context: tuple[object, ...],
    weights: tuple[object, ...],
    *,
    macrobatch_size: int,
    minibatch_size: int,
    num_comm_sms: int,
    swiglu_limit: Optional[float],
    device_index: int,
    context_key: int,
    x_shared=None,
    d_y_shared=None,
    routed_storage_tables=None,
    up_row_offset: int = 0,
    main_grads=None,
    main_grad_storage_tables=None,
    main_grad_up_row_offset: int = 0,
    situ_beta=None,
    has_shared: bool = True,
    recompute_forward_context: bool = False,
    caller_buffers=None,
    comm_tile_shape=None,
) -> tuple[_BwdLaunchPlan, tuple[object, ...]]:
    """Validate one set of inputs and geometry; prepare the workspace buffers, the executors and the
    launch-argument templates.  Returns the plan and the validated raw13 (this call's scratch and dW
    outputs), which the first call uses.  ``caller_buffers``: see ``backward_bf16``."""

    import torch

    def raw13_of(config):
        shapes = tuple(item.shape for item in raw13_descriptors(config))
        if caller_buffers is None or caller_buffers[0] is None:
            return _allocate_raw13(shapes, torch.empty(0, dtype=torch.bfloat16, device=workspace.device), main_grads, bool(up_row_offset), has_shared)
        return _caller_raw13(shapes, caller_buffers[0], caller_buffers[1], main_grads, bool(up_row_offset), has_shared, workspace.device)

    shared_inputs = (workspace.x_buffer if x_shared is None else x_shared, workspace.d_y_buffer if d_y_shared is None else d_y_shared)

    main_grad_dtype = _main_grad_dtype(main_grads)
    plan = PersistentBf16BwdPlan(
        ep_size=getattr(workspace, "ep_size", -1),
        num_local_experts=routed_expert_count(routed_storage_tables, weights[1]),
        num_local_tokens=getattr(workspace, "num_local_tokens", -1),
        hidden_size=getattr(workspace, "hidden_size", -1),
        routed_hidden_size=getattr(workspace, "routed_hidden_size", None),
        # From the routed FC1 (halved when combined): a routed-only layer has no shared weights to
        # read it from.
        intermediate_size=getattr(weights[1], "shape", (-1, -1))[-2] - up_row_offset,
        topk=getattr(workspace, "topk", -1),
        schedule_capacity=getattr(workspace, "schedule_capacity", -1),
        macrobatch_size=macrobatch_size,
        minibatch_size=minibatch_size,
        num_comm_sms=num_comm_sms,
        swiglu_limit=swiglu_limit,
        recompute_forward_context=recompute_forward_context,
        comm_tile_shape=_host_common.comm_tile_shape(getattr(workspace, "routed_hidden_size", None) or getattr(workspace, "hidden_size", -1), comm_tile_shape),
    )
    inputs = PersistentBf16BwdInputs(
        x_shared=shared_inputs[0],
        x_routed=context[0],
        gate_shared=context[1],
        gate_routed=context[2],
        up_shared=context[3],
        up_routed=context[4],
        hidden_shared=context[5],
        hidden_routed=context[6],
        d_y_shared=shared_inputs[1],
        w_shared_gate=weights[0],
        w_routed_gate=weights[1],
        w_shared_up=weights[2],
        w_routed_up=weights[3],
        w_shared_down=weights[4],
        w_routed_down=weights[5],
        schedule_peer_rank=schedule.peer_rank,
        schedule_peer_token_idx=schedule.peer_token_idx,
        num_tokens=schedule.num_tokens,
        tokens_per_expert=schedule.tokens_per_expert,
        x_peer_ptrs=tuple(workspace.x_buffer_ptrs),
        d_y_peer_ptrs=tuple(workspace.d_y_buffer_ptrs),
        d_x_peer_ptrs=tuple(workspace.d_x_routed_buffer_ptrs),
        router_peer_ptrs=tuple(workspace.router_weight_buffer_ptrs),
        d_router_peer_ptrs=tuple(workspace.d_router_weight_buffer_ptrs),
    )
    workspace_index = workspace.device.index if workspace.device.index is not None else device_index
    if workspace_index != device_index:
        raise RuntimeError("the workspace CUDA device must be current")
    _prepare_public_runtime_environment()
    config = plan.validate(schedule_capacity=inputs.schedule_peer_rank.numel())
    task_decode = select_task_decode(config.num_local_experts)
    has_tail = _has_tail(config.hidden_size, config.routed_hidden_size, config.intermediate_size)
    cache_key = _executor_cache_key(
        device_index,
        context_key,
        plan,
        inputs,
        task_decode,
        split_weights=routed_storage_tables is not None,
        main_grad_dtype=main_grad_dtype,
        is_situ=situ_beta is not None,
        has_shared=has_shared,
        has_tail=has_tail,
        recompute_forward_context=recompute_forward_context,
    )
    if (x_shared is None or d_y_shared is None) and config.routed_hidden_size != config.hidden_size:
        raise ValueError("x_shared and d_y_shared are required when the routed hidden differs from the shared hidden")
    if main_grads is not None:
        validate_main_grads(
            main_grads,
            num_local_experts=config.num_local_experts,
            hidden_size=config.routed_hidden_size,
            intermediate_size=config.intermediate_size,
            split=routed_storage_tables is not None,
            main_grad_up_row_offset=main_grad_up_row_offset,
            device=workspace.device,
            has_shared=has_shared,
        )
    # Where Up's weight gradient starts in the Gate one's buffer: a combined main_grad, or
    # this call's own combined dW for a combined FC1.
    wgrad_up_row_offset = main_grad_up_row_offset if main_grads is not None else up_row_offset
    # Allocation reuse can depend on geometry; machine code reuse must not.
    preparation_key = (cache_key, config)
    preparations = _workspace_preparations(workspace)
    prepared, state = preparations.get(preparation_key), None
    if prepared is None:
        state = prepare_persistent_bwd_state(
            config=config,
            device=workspace.device,
            num_tokens=inputs.num_tokens,
            main_grads=main_grads,
            combined_fc1=bool(up_row_offset),
            has_shared=has_shared,
            **({} if caller_buffers is None else dict(raw=raw13_of(config), storage=caller_buffers[2])),
        )
        _validate_persistent_bwd_abi(
            inputs,
            state,
            config=config,
            routed_storage_tables=routed_storage_tables,
            up_row_offset=up_row_offset,
            main_grad_dtype=main_grad_dtype,
            main_grad_storage_tables=main_grad_storage_tables,
            main_grad_up_row_offset=wgrad_up_row_offset,
            has_shared=has_shared,
        )
        _runtime_toolchain_snapshot(inputs.tokens_per_expert.device)
        prepared = _PersistentBf16BwdPreparation(
            preparation_key,
            state.counters,
            # Do not retain num_tokens aliases from an old schedule.
            state.metadata if config.build_o1_decode else state.metadata[:2],
            state.router_weights,
            state.d_router_weight_partials,
            state.counter_storage,
        )
        prepared.streams.add(int(current_stream_handle(device_index)))  # the allocation stream
        if len(preparations) >= _MAX_PREPARATIONS:
            preparations.pop(next(iter(preparations)))  # the oldest configuration
        preparations[preparation_key] = prepared
    workspace._cutedsl_bwd_preparation = prepared  # the one this call's plan and launch use
    if state is None:
        state = PersistentBf16BwdState(
            raw13_of(config),
            prepared.counters,
            (prepared.metadata if config.build_o1_decode else prepared.metadata + (inputs.num_tokens, inputs.num_tokens)),
            prepared.router_weights,
            prepared.d_router_weight_partials,
            prepared.counter_storage,
        )
        _validate_persistent_bwd_abi(
            inputs,
            state,
            config=config,
            routed_storage_tables=routed_storage_tables,
            up_row_offset=up_row_offset,
            main_grad_dtype=main_grad_dtype,
            main_grad_storage_tables=main_grad_storage_tables,
            main_grad_up_row_offset=wgrad_up_row_offset,
            has_shared=has_shared,
        )
    # The routed chain reaches the symmetric buffers only through the peer tables, at the routed
    # width: check the local ones after the ABI checks, as the FWD validation does.
    for name, rows in (
        ("x_buffer", config.num_local_tokens),
        ("d_y_buffer", config.num_local_tokens),
        ("d_x_routed_buffer", config.num_local_tokens * config.topk),
    ):
        _validate_tensor(
            f"workspace.{name}", getattr(workspace, name), shape=(rows, config.routed_hidden_size), dtype=torch.bfloat16, device=inputs.tokens_per_expert.device
        )

    from ._persistent_bf16_bwd_mega import (
        bwd_per_call_slots,
        fill_bwd_launch_slots,
        make_bwd_mega_args,
        prepare_bwd_mega as prepare_persistent_bwd_mega,
    )
    from ._persistent_bf16_bwd_metadata import fill_bwd_metadata_slots, prepare_bwd_metadata_launch

    stream = current_stream_handle(device_index)
    # Derived once per launch plan, which is keyed by the tables and the weights they came from:
    # a weight that moves gives a new plan rather than a kept table that has gone stale.
    # One row per routed operand, in the order the descriptors are built: the three weights the
    # dgrad reads MN-major, the two the replay reads K-major, then the three main_grad buffers the
    # Wgrad writes.  Gate and Up appear twice among the weights -- the same allocations in two
    # views, so two descriptor schemas over one set of addresses.
    routed_rows = (
        None
        if routed_storage_tables is None
        else (routed_storage_tables[0], routed_storage_tables[1], routed_storage_tables[2], routed_storage_tables[0], routed_storage_tables[1])
    )
    if main_grad_storage_tables is not None:
        routed_rows = (*routed_rows, main_grad_storage_tables[2], main_grad_storage_tables[0], main_grad_storage_tables[1])
    # The descriptors' schema: the executor's TMA atoms over these weights' and main_grads' layouts.
    routed = routed_launch_tensors(
        routed_rows,
        config.num_local_experts,
        workspace.device,
        schema=(
            cache_key,
            tuple(
                tensor_layout(tensor)
                for tensor in (*weights[1::2], *(() if main_grads is None else (buffer for buffer in main_grads[1::2] if buffer is not None)))
            ),
        ),
    )
    routed_args = dict(
        routed_storage_tables=routed.addresses,
        routed_descriptors=routed.descriptors,
        build_descriptors=routed.build,
        up_row_offset=up_row_offset,
        main_grad_up_row_offset=wgrad_up_row_offset,
        situ_beta=(0.0, 0.0) if situ_beta is None else situ_beta,
    )
    with _EXECUTOR_CACHE_LOCK:
        executor = _EXECUTOR_CACHE.get(cache_key)
        if executor is None:
            executor = persistent_executor(
                (cache_key[0], cache_key[1], *cache_key[4:]),
                lambda: prepare_persistent_bwd_mega(
                    inputs,
                    state,
                    config=config,
                    stream=stream,
                    task_decode=task_decode,
                    split_weights=routed_storage_tables is not None,
                    accumulate_wgrad=main_grad_dtype is not None,
                    bf16_main_grad=main_grad_dtype is torch.bfloat16,
                    split_main_grads=(main_grad_storage_tables is not None),
                    is_situ=situ_beta is not None,
                    has_shared=has_shared,
                    has_tail=has_tail,
                    recompute_forward_context=recompute_forward_context,
                    **routed_args,
                ),
            )
            _EXECUTOR_CACHE[cache_key] = executor
            _EXECUTOR_CACHE_DEVICES.add(device_index)
    # The compiled functions are TVM FFI functions: torch views and plain scalars, not CuTe tensors.
    # The per-call slot positions must match the full argument builders: checked once per plan.
    metadata_count = len(state.metadata)
    aliases = tuple(i for i, tensor in enumerate(state.metadata) if tensor is inputs.num_tokens)
    alias_slots = tuple(9 + i for i in aliases)
    schedule_tensors = _schedule_tensors(schedule)
    template = checked_template(
        make_bwd_mega_args(inputs, state, config=config, stream=stream, launch=True, peer_tables=workspace_peer_tables(workspace), **routed_args),
        bwd_per_call_slots(metadata_count, alias_slots),
        lambda args: fill_bwd_launch_slots(
            args, shared_inputs[0], shared_inputs[1], schedule_tensors, weights, context, state.raw13, stream, metadata_count, alias_slots, routed.build
        ),
        "fill_bwd_launch_slots positions do not match make_bwd_mega_args",
    )
    metadata_executor, metadata_full = prepare_bwd_metadata_launch(inputs, state, config=config, stream=stream)
    metadata_alias_slots = tuple(4 + i for i in aliases)
    metadata_template = checked_template(
        metadata_full,
        (0, 1, 2, 3) + metadata_alias_slots,
        lambda args: fill_bwd_metadata_slots(args, schedule_tensors, metadata_alias_slots),
        "fill_bwd_metadata_slots positions do not match make_bwd_metadata_args",
    )
    launch_plan = _BwdLaunchPlan(
        executor=executor,
        template=template,
        metadata_executor=metadata_executor,
        metadata_template=metadata_template,
        schedule_layouts=tuple(tensor_layout(tensor) for tensor in schedule_tensors),
        context_layouts=tuple(tensor_layout(tensor) for tensor in context),
        shared_input_layouts=tuple(tensor_layout(tensor) for tensor in shared_inputs),
        # The descriptors' shapes, as the first call's _allocate_raw13 took them: a combined FC1's
        # one buffer and a routed-only layer's placeholders are rebuilt from them the same way.
        raw13_shapes=tuple(item.shape for item in raw13_descriptors(config)),
        raw13_proto=state.raw13[0].new_empty(0),
        metadata_count=metadata_count,
        alias_slots=alias_slots,
        metadata_alias_slots=metadata_alias_slots,
        fill_slots=fill_bwd_launch_slots,
        fill_metadata_slots=fill_bwd_metadata_slots,
        routed=routed,
    )
    return launch_plan, state.raw13


def _schedule_tensors(schedule) -> tuple[object, ...]:
    return (schedule.peer_rank, schedule.peer_token_idx, schedule.num_tokens, schedule.tokens_per_expert)


def _recomputed_routed_context(forward_context, workspace, intermediate_size: int, macrobatch_size: int):
    """recompute_forward_context: the forward context carries the shared activations only.  The routed ones are this
    call's uninitialized macrobatch-sized scratch, which the replay writes before any task reads it."""
    import torch
    from types import SimpleNamespace

    names = ("x_routed", "gate_routed", "up_routed", "hidden_routed")
    if any(getattr(forward_context, name) is not None for name in names):
        raise ValueError("recompute_forward_context takes a forward context without routed activations")
    routed_hidden = getattr(workspace, "routed_hidden_size", None) or workspace.hidden_size
    columns = dict(x_routed=routed_hidden, gate_routed=intermediate_size, up_routed=intermediate_size, hidden_routed=intermediate_size)
    context = {name: getattr(forward_context, name) for name in _FORWARD_CONTEXT_NAMES}
    for name in names:
        context[name] = torch.empty((macrobatch_size, columns[name]), dtype=torch.bfloat16, device=workspace.device)
    return SimpleNamespace(**context)


def _backward_context(forward_context, has_shared: bool, device) -> tuple[object, ...]:
    """The seven forward-context activations the kernel reads (``_FORWARD_CONTEXT_NAMES``).  A
    routed-only layer's context has no shared ones; the placeholder stands in for them.  Its
    shared presence must match the weights'."""
    context = tuple(getattr(forward_context, name) for name in _FORWARD_CONTEXT_NAMES)
    if shared_presence(context[1], context[3], context[5]) != has_shared:
        raise ValueError("shared weight presence must match the forward context")
    if has_shared:
        return context
    placeholder = shared_placeholder(device)
    return (context[0], placeholder, context[2], placeholder, context[4], placeholder, context[6])


def _main_grad_dtype(main_grads):
    """The dtype the fused wgrad epilogue writes, or None when it is not fused.

    The destination is FP32 or BF16, a compile-time variant; every buffer must agree.
    """
    if main_grads is None:
        return None
    present = [buffer for buffer in main_grads if buffer is not None]
    if not present:
        raise ValueError("main_grads must hold at least the three routed buffers")
    dtypes = {buffer.dtype for buffer in present}
    if len(dtypes) != 1:
        raise ValueError(f"the main_grad buffers must agree on a dtype; got {sorted(map(str, dtypes))}")
    return dtypes.pop()


def validate_main_grads(
    main_grads, *, num_local_experts: int, hidden_size: int, intermediate_size: int, split: bool, main_grad_up_row_offset: int, device, has_shared: bool = True
) -> None:
    """Check the six main_grad buffers of fused gradient accumulation.

    Order: shared/routed per matrix (gate, up, down).  All present, one dtype (FP32 or BF16),
    one device, contiguous and 16-byte aligned, each shaped like its weight.  A combined FC1's
    main_grad is one buffer under both the routed gate and up slots -- ``[E, 2I, H]``, or the
    representative ``[2I, H]`` when split -- with Up ``main_grad_up_row_offset`` = I rows down;
    separate buffers take no offset.  Checked on every call: a buffer that silently disagrees
    with its weight would lose gradient rows, since TMA clips at the tensor's bounds.

    A routed-only layer (``has_shared=False``) has no shared buffers: only the routed three are
    required and the shared slots are ignored.
    """
    import torch

    if len(main_grads) != 6:
        raise ValueError(f"main_grads must hold six buffers; got {len(main_grads)}")
    names = ("shared_gate", "routed_gate", "shared_up", "routed_up", "shared_down", "routed_down")
    combined = main_grads[1] is main_grads[3]
    i, h, e = intermediate_size, hidden_size, num_local_experts
    if main_grad_up_row_offset != (i if combined else 0):
        raise ValueError(f"main_grad_up_row_offset must be {i if combined else 0} for " f"{'one combined' if combined else 'separate'} FC1 main_grad buffers")
    fc1 = (2 * i if combined else i, h)
    down = (h, i)
    expected = ((i, h), fc1 if split else (e, *fc1), (i, h), fc1 if split else (e, *fc1), down, down if split else (e, *down))
    if not has_shared:
        if any(main_grads[index] is not None for index in (0, 2, 4)):
            raise ValueError("a routed-only layer has no shared main_grad buffers")
        names, main_grads, expected = names[1::2], main_grads[1::2], expected[1::2]
    missing = [name for name, buffer in zip(names, main_grads) if buffer is None]
    if missing:
        raise ValueError(f"main_grads is missing {missing}")
    dtype = _main_grad_dtype(main_grads)
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"main_grad buffers must be float32 or bfloat16; got {dtype}")
    for name, buffer, shape in zip(names, main_grads, expected, strict=True):
        _validate_tensor(f"main_grads.{name}", buffer, shape=shape, dtype=dtype, device=device)


#: raw13's shared slots: d_x_shared, d_gate_shared, d_up_shared and the three shared dW.
SHARED_RAW13: Final = (0, 2, 4, 7, 9, 11)


def _allocate_raw13(shapes, proto, main_grads=None, combined_fc1: bool = False, has_shared: bool = True) -> tuple[object, ...]:
    """This call's raw13 on the current stream: the seven scratch tensors (0-6) and the six dW outputs
    (7-12), one caching-allocator call each; freed when the caller drops them.  Separate allocations
    on purpose: d_x_shared (returned as d_x) and the dW are held by the caller past the call, and
    separate tensors of the forward's sizes reuse the segments the forward just freed, where one block
    the size of all internal scratch tensors fits none of them and makes the caching allocator reserve
    a new segment of that size.  ``proto`` is an empty bfloat16 tensor on the target device:
    ``new_empty`` takes dtype and device from it, which saves the keyword parsing of ``torch.empty``
    on each of the thirteen calls.  ``shapes`` are raw13_descriptors'.

    A routed-only layer allocates none of the shared slots: they are the device's placeholder, in the
    weight gradients' dtype for the three dW, which nothing touches.
    """

    if main_grads is None:
        allocated = [proto.new_empty(shape) for shape in shapes]
        if combined_fc1:
            # A combined FC1's gradient is one [E, 2I, H] buffer under both the Gate and the Up
            # slot; the kernel writes Up I rows down.
            experts, rows, columns = shapes[8]
            allocated[8] = allocated[10] = proto.new_empty((experts, 2 * rows, columns))
    else:
        # Fused accumulation: the six weight gradients *are* the caller's main_grad buffers -- only
        # the seven scratch tensors are allocated.
        allocated = [proto.new_empty(shape) for shape in shapes[:7]] + list(main_grads)
    if not has_shared:
        wgrad_dtype = proto.dtype if main_grads is None else _main_grad_dtype(main_grads)
        for index in SHARED_RAW13:
            allocated[index] = shared_placeholder(proto.device, proto.dtype if index < 7 else wgrad_dtype)
    return tuple(allocated)


def _caller_raw13(shapes, scratch, d_weights, main_grads, combined_fc1: bool, has_shared: bool, device) -> tuple[object, ...]:
    """The raw13 of a call in the execute step of a plan, which allocates nothing.

    ``scratch``: the seven scratch tensors (raw13 0-6), each at least as many rows as this call's and its
    columns exactly -- a caller may size them for its largest token count and macrobatch -- of which the leading
    rows are used.  The weight gradients are ``main_grads`` (fused accumulation), or ``d_weights``: the caller's
    six output buffers in raw13's order (shared / routed per matrix: gate, up, down), shaped like the weights,
    None where a routed-only layer has no shared weight; a combined FC1 passes its one ``[E, 2I, H]`` buffer
    under both the Gate and the Up slot.  The kernel stores into them.
    """
    import torch

    rows = []
    for index, (tensor, shape) in enumerate(zip(scratch, shapes[:7], strict=True)):
        if not has_shared and index in SHARED_RAW13:
            rows.append(None)
            continue
        if (
            type(tensor) is not torch.Tensor
            or tensor.dtype != torch.bfloat16
            or tensor.device != device
            or tensor.ndim != 2
            or tensor.shape[1] != shape[1]
            or tensor.shape[0] < shape[0]
            or not tensor.is_contiguous()
            or tensor.data_ptr() % 16
        ):
            raise ValueError(
                f"backward scratch {RAW13_NAMES[index]} must be a contiguous 16-byte aligned BF16 " f"[>= {shape[0]}, {shape[1]}] tensor on {device}"
            )
        rows.append(tensor if tensor.shape[0] == shape[0] else tensor[: shape[0]])
    if main_grads is not None:
        allocated = rows + list(main_grads)
    else:
        if d_weights is None or len(d_weights) != 6:
            raise ValueError("the backward takes main_grads or six weight-gradient output buffers")
        expected = list(shapes[7:])
        if combined_fc1:
            if d_weights[1] is not d_weights[3]:
                raise ValueError("a combined FC1 passes one weight-gradient buffer under both Gate and Up")
            experts, fc1_rows, columns = shapes[8]
            expected[1] = expected[3] = (experts, 2 * fc1_rows, columns)
        for index, (tensor, shape) in enumerate(zip(d_weights, expected)):
            if has_shared or 7 + index not in SHARED_RAW13:
                _validate_tensor(f"d_weights[{index}]", tensor, shape=shape, dtype=torch.bfloat16, device=device)
        allocated = rows + list(d_weights)
    if not has_shared:
        wgrad_dtype = torch.bfloat16 if main_grads is None else _main_grad_dtype(main_grads)
        for index in SHARED_RAW13:
            allocated[index] = shared_placeholder(device, torch.bfloat16 if index < 7 else wgrad_dtype)
    return tuple(allocated)


def backward_bf16(
    workspace,
    schedule,
    forward_context,
    shared_gate_weights,
    routed_gate_weights,
    shared_up_weights,
    routed_up_weights,
    shared_down_weights,
    routed_down_weights,
    *,
    macrobatch_size: int,
    minibatch_size: int,
    num_comm_sms: int,
    swiglu_limit: Optional[float] = None,
    x_shared=None,
    d_y_shared=None,
    routed_storage_tables=None,
    up_row_offset: int = 0,
    main_grads=None,
    main_grad_storage_tables=None,
    main_grad_up_row_offset: int = 0,
    situ_beta=None,
    recompute_forward_context: bool = False,
    caller_buffers=None,
    comm_tile_shape=None,
) -> tuple[object, ...]:
    """Run the public runtime-geometry persistent CuTe DSL BF16 backward.

    ``comm_tile_shape`` (a compiled variant; ``CuteMoKConfig.comm_tile_shape``): rows x columns of one
    communication tile, one of the contract's COMM_TILE_SHAPES or its (rows, columns); None takes
    ``_host_common.comm_tile_shape``'s default.  Only how the communication CTAs move bytes; same results.

    ``caller_buffers`` (the execute step of a plan): ``(scratch, d_weights, storage)`` -- the seven scratch
    tensors and the six weight-gradient buffers of ``_caller_raw13`` (``d_weights`` None with ``main_grads``)
    and the plan's ``(int32, float32)`` storage of ``prepare_persistent_bwd_state``.  The call then allocates
    nothing.  ``(None, None, storage)`` (the operator API): the plan's storage, the raw13 allocated per call.  By
    default it allocates its raw13, and the workspace keeps one buffer set per configuration.

    ``recompute_forward_context``: ``forward_context`` holds the shared activations only (its routed ones are None); the
    kernel replays the first macrobatch's routed X / Gate / Up like every later one's.

    ``situ_beta``: the forward's SiTU-GLU ``(beta1, beta2)``, or None for SwiGLU.

    ``x_shared`` / ``d_y_shared`` are the shared chain's input and output gradient in hidden_size
    (when the hiddens differ); ``None`` uses the workspace's symmetric x / dY buffers, which an
    ordinary MoE shares between the two chains.  The routed chain always reads the symmetric buffers
    (routed_hidden_size rows).

    Calls sharing a workspace must not overlap; callers establish ordering
    across streams and must not mutate schedules during a call. Every call
    rebuilds metadata, including after unversioned/in-place schedule updates.
    Host ABI/geometry errors are synchronous ValueError; invalid device values
    trap asynchronously (CUDA error at a later synchronization, not ValueError).
    The full validation runs when a launch plan is built; every call checks the
    schedule, forward-context and shared-input layouts and runs the metadata builder and the
    mega kernel (no zero kernel).  The returned raw13 (the seven scratch tensors and the six dW
    outputs) are this call's own allocations on the current stream; between calls the workspace
    keeps only the readiness counters and decoder metadata (schedule-capacity sized) and the router
    weights and partials (macrobatch-sized), see _PersistentBf16BwdPreparation.

    ``routed_storage_tables`` are the Gate, Up and Down per-expert base addresses when the routed
    experts are independent per-expert allocations (``cudnn.moe_ep._cutemok.ops.make_routed_weight_storage_table_bf16``
    builds one).  The three routed weights are then the representative expert alone and the dgrad
    names the others through per-expert TMA descriptors; ``up_row_offset`` is the Up half's first
    row inside a combined FC1 parameter, which a retargeted descriptor cannot express as a
    separate base address.  The tables join the plan key, so a rebuilt table is a new plan.

    The three shared weights all None: a routed-only layer, whose forward context has no shared
    activations either.  There are no shared tasks; the shared main_grad slots are ignored and the
    six shared raw13 slots come back None.
    """

    import torch

    situ_beta = normalize_situ_beta(situ_beta, swiglu_limit)
    has_shared = shared_presence(shared_gate_weights, shared_up_weights, shared_down_weights)
    if not has_shared:
        shared_gate_weights = shared_up_weights = shared_down_weights = shared_placeholder(workspace.device)
        if main_grads is not None and len(main_grads) == 6:
            main_grads = (None, main_grads[1], None, main_grads[3], None, main_grads[5])
    weights = (
        shared_gate_weights,
        routed_gate_weights,
        shared_up_weights,
        routed_up_weights,
        shared_down_weights,
        routed_down_weights,
    )
    device_index = _current_device_index()
    context_key = _current_cuda_context_key()
    tables = routed_storage_tables
    if recompute_forward_context and caller_buffers is not None:
        raise NotImplementedError("the execute step of a plan needs the forward context's routed activations")
    if recompute_forward_context:
        forward_context = _recomputed_routed_context(forward_context, workspace, int(routed_gate_weights.shape[-2]) - up_row_offset, macrobatch_size)
    context = schedule_tensors = shared_inputs = None  # set in the try; the check below reads them only with a plan
    try:
        if tables is not None:
            tables = tuple(tables)
        context = _backward_context(forward_context, has_shared, workspace.device)
        schedule_tensors = _schedule_tensors(schedule)
        # The address tables are plan sources: the plan keeps the (5, E) table derived from them.  The
        # weights are matched by address and layout (LaunchPlanCache views): the template refills them on
        # every call, and a caller may pass new Gate / Up row views of a combined FC1 on every call.
        sources = (workspace.x_buffer, workspace.d_y_buffer, *(tables or ()), *(main_grad_storage_tables or ()))
        shared_inputs = (workspace.x_buffer if x_shared is None else x_shared, workspace.d_y_buffer if d_y_shared is None else d_y_shared)
        # Everything else the validation and the arguments read: geometry, the workspace scalars
        # and peer addresses (typed), whether the shared inputs are the workspace buffers, device
        # and context, and the task->expert decode variant selected for this expert count.
        extra = typed_key(
            macrobatch_size,
            minibatch_size,
            num_comm_sms,
            swiglu_limit,
            workspace.ep_size,
            workspace.num_local_tokens,
            workspace.hidden_size,
            getattr(workspace, "routed_hidden_size", None),
            workspace.topk,
            workspace.schedule_capacity,
            workspace.x_buffer_ptrs,
            workspace.d_y_buffer_ptrs,
            workspace.d_x_routed_buffer_ptrs,
            workspace.router_weight_buffer_ptrs,
            workspace.d_router_weight_buffer_ptrs,
            x_shared is None,
            d_y_shared is None,
            tables is None,
            up_row_offset,
            None if main_grads is None else str(_main_grad_dtype(main_grads)),
            main_grad_storage_tables is None,
            main_grad_up_row_offset,
            situ_beta,
            has_shared,
            recompute_forward_context,
        ) + (
            device_index,
            context_key,
            workspace.device,
            select_task_decode(int(schedule.tokens_per_expert.shape[0])).name,
            _host_common.comm_tile_shape(getattr(workspace, "routed_hidden_size", None) or workspace.hidden_size, comm_tile_shape),
        )
        # The plan lives with the buffer set it was built on: the last one used first (one configuration
        # per workspace is the common case), then the other configurations' sets.
        prepared = workspace._cutedsl_bwd_preparation
        launch_plan = None if prepared is None else prepared.launch_plans.get(sources, extra, weights)
        if launch_plan is None:
            for other in (getattr(workspace, "_cutedsl_bwd_preparations", None) or {}).values():
                if other is not prepared:
                    launch_plan = other.launch_plans.get(sources, extra, weights)
                    if launch_plan is not None:
                        prepared = workspace._cutedsl_bwd_preparation = other
                        break
    except (AttributeError, TypeError) as error:
        launch_plan = None
        key_error = error
    else:
        key_error = None
    if (
        launch_plan is not None
        and layouts_match(schedule_tensors, launch_plan.schedule_layouts, 16)
        and layouts_match(context, launch_plan.context_layouts, 16)
        and layouts_match(shared_inputs, launch_plan.shared_input_layouts, 16)
    ):
        raw13 = None  # allocated below, after the metadata builder is enqueued
    else:
        # A new plan; malformed arguments that broke the key are reported by the full validation,
        # as on any first call.
        launch_plan, raw13 = _build_launch_plan(
            workspace,
            schedule,
            _backward_context(forward_context, has_shared, workspace.device),
            weights,
            macrobatch_size=macrobatch_size,
            minibatch_size=minibatch_size,
            num_comm_sms=num_comm_sms,
            swiglu_limit=swiglu_limit,
            device_index=device_index,
            context_key=context_key,
            x_shared=x_shared,
            d_y_shared=d_y_shared,
            routed_storage_tables=tables,
            up_row_offset=up_row_offset,
            main_grads=main_grads,
            main_grad_storage_tables=(None if main_grad_storage_tables is None else tuple(main_grad_storage_tables)),
            main_grad_up_row_offset=main_grad_up_row_offset,
            situ_beta=situ_beta,
            has_shared=has_shared,
            recompute_forward_context=recompute_forward_context,
            caller_buffers=caller_buffers,
            comm_tile_shape=comm_tile_shape,
        )
        if key_error is not None:
            raise key_error
        prepared = workspace._cutedsl_bwd_preparation  # the build may have replaced the buffers
        prepared.launch_plans.put(sources, extra, launch_plan, weights)
    stream = current_stream_handle(device_index)
    handle = int(stream)
    if handle not in prepared.streams:
        # First use of this workspace's buffers on this stream: record them there once (see
        # _PersistentBf16BwdPreparation.streams).  Per-call and caller-owned tensors are used on the
        # current stream like PyTorch's own ops, and a caller that frees them from another stream
        # synchronizes that stream itself.
        current = torch.cuda.current_stream(device_index)
        for tensor in prepared.buffers():
            tensor.record_stream(current)
        prepared.streams.add(handle)
    prepared.counter_storage.zero_()
    # All launches go to the current stream.  The metadata builder validates counts and builds the
    # tables BEFORE the mega consumes them; it runs every call (schedule values are never cached).
    # Route payload validity is the schedule producer/caller's responsibility.
    # No external zero kernel: the mega writes every routed dW tile itself (segment 0 stores, later
    # macrobatch segments reduce-add in order, empty experts store zeros), so dW never depends on
    # prior buffer contents.
    launch_plan.metadata_executor(
        *launch_plan.fill_metadata_slots(list(launch_plan.metadata_template), schedule_tensors, launch_plan.metadata_alias_slots), stream
    )
    if raw13 is None:
        # The call's scratch and dW (a plan hit): allocated after the counter memset and the metadata
        # builder are enqueued, so the host-side allocation overlaps their GPU time instead of
        # delaying the mega launch.  The allocation is stream-ordered on the same stream.
        if caller_buffers is None or caller_buffers[0] is None:
            raw13 = _allocate_raw13(launch_plan.raw13_shapes, launch_plan.raw13_proto, main_grads, bool(up_row_offset), has_shared)
        else:
            raw13 = _caller_raw13(launch_plan.raw13_shapes, caller_buffers[0], caller_buffers[1], main_grads, bool(up_row_offset), has_shared, workspace.device)
    routed = launch_plan.routed
    launch_plan.executor(
        *launch_plan.fill_slots(
            list(launch_plan.template),
            shared_inputs[0],
            shared_inputs[1],
            schedule_tensors,
            weights,
            context,
            raw13,
            stream,
            launch_plan.metadata_count,
            launch_plan.alias_slots,
            routed.build,
        )
    )
    routed.mark_built()
    if not has_shared:
        return tuple(None if index in SHARED_RAW13 else tensor for index, tensor in enumerate(raw13))
    return raw13


def backward_bf16_storage_lengths(workspace, *, num_local_experts: int, intermediate_size: int, macrobatch_size: int, minibatch_size: int) -> tuple[int, int]:
    """Lengths of a plan's ``(int32, float32)`` storage (``prepare_persistent_bwd_state``) for backwards over
    ``workspace`` with macrobatches up to ``macrobatch_size``.

    The counters and the decoder metadata grow as the minibatch and the macrobatch shrink, the router weights
    and partials with the macrobatch; so the lengths are the largest over the given sizes and the 256-row
    minibatch with the smallest and the largest macrobatch.  A call whose geometry still needs more is refused
    by ``prepare_persistent_bwd_state``, never run short."""
    words = floats = 0
    for macro, mini in ((macrobatch_size, minibatch_size), (macrobatch_size, 256), (256, 256)):
        try:
            config = BwdGeometry(
                schedule_capacity=workspace.schedule_capacity,
                num_local_tokens=workspace.num_local_tokens,
                macrobatch_size=macro,
                minibatch_size=mini,
                num_comm_sms=2,
                num_local_experts=num_local_experts,
                hidden_size=workspace.hidden_size,
                routed_hidden_size=getattr(workspace, "routed_hidden_size", None),
                intermediate_size=intermediate_size,
                topk=workspace.topk,
                ep_size=workspace.ep_size,
            )
        except ValueError:
            if mini == minibatch_size:
                raise  # the plan's own sizes
            continue  # a bound no call can use (it exceeds the int32 task domain)
        plan = TaskPlan(config)
        lengths = [item.shape[0] for item in counter_descriptors(plan)]
        used = max(sum(-(-length // 4) * 4 for length in lengths), 4)
        used += sum(-(-item.shape[0] // 4) * 4 for index, item in enumerate(metadata_descriptors(plan)) if index < 2 or config.build_o1_decode)
        words = max(words, used)
        floats = max(floats, -(-macro // 4) * 4 + macro * (intermediate_size // 128))
    return words, floats


def compile_backward_bf16(
    workspace,
    *,
    num_local_experts: int,
    intermediate_size: int,
    is_clamped: bool,
    is_situ: bool = False,
    has_shared: bool = True,
    split_weights: bool = False,
    main_grad_dtype=None,
    comm_tile_shape=None,
) -> None:
    """The plan step: compile the backward megakernel variant a layer runs and its metadata builder, without
    launching, and build the workspace's peer address tables.

    The megakernel variant is what its executor key holds besides the tensor ranks and dtypes: the SwiGLU clamp,
    SiTU-GLU, the shared expert, grouped or split routed weights, fused accumulation and its dtype
    (``main_grad_dtype``: None, FP32 or BF16), the communication tile shape (``comm_tile_shape`` as in
    ``backward_bf16``), the task decode (from ``num_local_experts``) and whether H, the routed H or I has a
    128-wide tail; the metadata builder's is the tail of the routed H or I.  Sizes are
    runtime values, so the compile runs over a small stand-in geometry (512 tokens, one expert, one 256-row
    macrobatch, 256-wide H and I plus the tail where the layer has one) whose tensors it allocates and drops.
    A later ``backward_bf16`` over real tensors computes the same keys and finds the executors.  The variant
    that replays the first macrobatch (no routed forward context) is not compiled here.
    """
    import torch
    from ._host_common import peer_address_table

    device = workspace.device
    device_index = _current_device_index()
    if (device.index if device.index is not None else device_index) != device_index:
        raise RuntimeError("the workspace CUDA device must be current")
    _prepare_public_runtime_environment()
    hidden, routed_hidden = workspace.hidden_size, getattr(workspace, "routed_hidden_size", workspace.hidden_size)
    metadata_tail = _has_tail(routed_hidden, intermediate_size)
    has_tail = _has_tail(hidden, routed_hidden, intermediate_size)
    # The stand-in sizes reproduce both flags: a tail of I serves both, a tail of the shared H alone only the
    # megakernel's.
    i, r = (384 if metadata_tail else 256), 256
    h = 384 if has_tail and not metadata_tail else 256
    tokens, macro, capacity = 512, 256, 512 * workspace.topk * 2
    plan = PersistentBf16BwdPlan(
        ep_size=workspace.ep_size,
        num_local_experts=1,
        num_local_tokens=tokens,
        hidden_size=h,
        routed_hidden_size=r,
        intermediate_size=i,
        topk=workspace.topk,
        schedule_capacity=capacity,
        macrobatch_size=macro,
        minibatch_size=macro,
        num_comm_sms=2,
        swiglu_limit=1.0 if is_clamped else None,
        # The layer's routed hidden, not the stand-in's.
        comm_tile_shape=_host_common.comm_tile_shape(routed_hidden, comm_tile_shape),
    )
    config = plan.validate(schedule_capacity=capacity)

    def bf16(*shape):
        return torch.empty(shape, dtype=torch.bfloat16, device=device)

    def routed(*shape, dtype=torch.bfloat16):  # one expert: [N, K] split, [1, N, K] grouped
        return torch.empty(shape if split_weights else (1, *shape), dtype=dtype, device=device)

    placeholder = shared_placeholder(device)
    shared = (lambda *shape: bf16(*shape)) if has_shared else (lambda *shape: placeholder)
    words = torch.zeros(2 * capacity + 8, dtype=torch.int32, device=device)
    inputs = PersistentBf16BwdInputs(
        x_shared=bf16(tokens, h),
        x_routed=bf16(macro, r),
        gate_shared=shared(tokens, i),
        gate_routed=bf16(macro, i),
        up_shared=shared(tokens, i),
        up_routed=bf16(macro, i),
        hidden_shared=shared(tokens, i),
        hidden_routed=bf16(macro, i),
        d_y_shared=bf16(tokens, h),
        w_shared_gate=shared(i, h),
        w_routed_gate=routed(i, r),
        w_shared_up=shared(i, h),
        w_routed_up=routed(i, r),
        w_shared_down=shared(h, i),
        w_routed_down=routed(r, i),
        schedule_peer_rank=words[:capacity],
        schedule_peer_token_idx=words[capacity : 2 * capacity],
        num_tokens=words[2 * capacity : 2 * capacity + 1],
        tokens_per_expert=words[2 * capacity + 4 : 2 * capacity + 5],
        x_peer_ptrs=tuple(workspace.x_buffer_ptrs),
        d_y_peer_ptrs=tuple(workspace.d_y_buffer_ptrs),
        d_x_peer_ptrs=tuple(workspace.d_x_routed_buffer_ptrs),
        router_peer_ptrs=tuple(workspace.router_weight_buffer_ptrs),
        d_router_peer_ptrs=tuple(workspace.d_router_weight_buffer_ptrs),
    )
    main_grads = None
    if main_grad_dtype is not None:

        def grad(*shape):
            return torch.empty(shape, dtype=main_grad_dtype, device=device)

        main_grads = (
            grad(i, h) if has_shared else None,
            routed(i, r, dtype=main_grad_dtype),
            grad(i, h) if has_shared else None,
            routed(i, r, dtype=main_grad_dtype),
            grad(h, i) if has_shared else None,
            routed(r, i, dtype=main_grad_dtype),
        )
    state = prepare_persistent_bwd_state(config=config, device=device, num_tokens=inputs.num_tokens, main_grads=main_grads, has_shared=has_shared)
    task_decode = select_task_decode(num_local_experts)
    cache_key = _executor_cache_key(
        device_index,
        _current_cuda_context_key(),
        plan,
        inputs,
        task_decode,
        split_weights=bool(split_weights),
        main_grad_dtype=main_grad_dtype,
        is_situ=bool(is_situ),
        has_shared=has_shared,
        has_tail=has_tail,
    )
    stream = current_stream_handle(device_index)
    from ._persistent_bf16_bwd_mega import prepare_bwd_mega
    from ._persistent_bf16_bwd_metadata import prepare_bwd_metadata_launch

    stand_ins = routed_launch_tensors(None, 1, device)
    with _EXECUTOR_CACHE_LOCK:
        if cache_key not in _EXECUTOR_CACHE:
            _EXECUTOR_CACHE[cache_key] = persistent_executor(
                (cache_key[0], cache_key[1], *cache_key[4:]),
                lambda: prepare_bwd_mega(
                    inputs,
                    state,
                    config=config,
                    stream=stream,
                    task_decode=task_decode,
                    split_weights=bool(split_weights),
                    accumulate_wgrad=main_grad_dtype is not None,
                    bf16_main_grad=main_grad_dtype is torch.bfloat16,
                    split_main_grads=bool(split_weights) and main_grad_dtype is not None,
                    is_situ=bool(is_situ),
                    has_shared=has_shared,
                    has_tail=has_tail,
                    routed_storage_tables=stand_ins.addresses,
                    routed_descriptors=stand_ins.descriptors,
                    build_descriptors=0,
                    situ_beta=(1.0, 1.0) if is_situ else (0.0, 0.0),
                ),
            )
            _EXECUTOR_CACHE_DEVICES.add(device_index)
    prepare_bwd_metadata_launch(inputs, state, config=config, stream=stream)
    tables = workspace_peer_tables(workspace)
    for pointers in (inputs.x_peer_ptrs, inputs.d_y_peer_ptrs, inputs.d_x_peer_ptrs, inputs.router_peer_ptrs, inputs.d_router_peer_ptrs):
        peer_address_table(pointers, device, tables)


__all__ = [
    "PersistentBf16BwdInputs",
    "PersistentBf16BwdPlan",
    "PersistentBf16BwdState",
    "REQUIRED_CUDA_MAJOR",
    "TARGET_ARCH",
    "ToolchainMismatchError",
    "backward_bf16",
    "backward_bf16_storage_lengths",
    "clear_persistent_bf16_bwd_executor_cache",
    "compile_backward_bf16",
    "prepare_persistent_bwd_state",
    "select_task_decode",
    "validate_toolchain_snapshot",
]
