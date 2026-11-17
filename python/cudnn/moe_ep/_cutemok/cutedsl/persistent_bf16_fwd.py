# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Host entry points of the BF16 forward megakernel: ``forward_bf16``, and ``shared_fc1_bf16`` for its shared
FC1 tasks alone.

A ``forward_bf16`` call validates its workspace, schedule, weights and geometry, allocates the per-call state (nine
outputs and five readiness counters), gets the compiled kernel variant and launches it once.

Caching.  The executor cache holds compiled code only, keyed by code and ABI structure (variant flags, tensor
ranks and dtypes, device, context, library versions) and never by a size or an address.  A launch plan, keyed by
the workspace buffers, the weights and the geometry, holds the loop-invariant launch arguments; the full
validation runs when a plan is built.  A later call checks the schedule and the shared input against the
validated layouts, allocates fresh state on the current stream, fills the per-call slots and launches.

Geometry (token counts, hidden sizes, intermediate size, expert count, top-k, macrobatch and minibatch size,
communication SMs, EP size) is runtime data.  torch and CUTLASS are imported lazily: this module imports on a
CPU-only host, and CUTLASS only after the CuTe DSL version check.  Unsupported configurations raise.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import threading
from typing import Final

from .persistent_bf16_fwd_contract import (
    CLUSTER_SIZE,
    COMM_TILE_SHAPES,
    DEFAULT_COMM_TILE_SHAPE,
    CommTileShape,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    MACROBATCH_ROWS,
    MINIBATCH_ROWS,
    NUM_LOCAL_EXPERTS,
    TOPK,
    EXPERT_TABLE_CAP,
    TaskDecode,
)
from . import _host_common
from ._host_common import current_cuda_context_key as _current_cuda_context_key
from ._host_common import has_tail as _has_tail, persistent_executor, process_independent
from ._host_common import validate_tensor as _validate_tensor
from ._host_common import (
    LaunchPlanCache,
    checked_template,
    current_stream_handle,
    layouts_match,
    normalize_situ_beta,
    tensor_layout,
    typed_key,
    workspace_peer_tables,
)
from ._host_common import current_device_index as _current_device_index


from ._arch import COMPILE_OPTIONS as _COMPILE_OPTIONS, TARGET_ARCH, TARGET_CAPABILITY, check_device
from ._version_gate import require_cutedsl

FC1_TILE_M: Final = 256
FC1_LOGICAL_N: Final = 128
CLUSTER_SHAPE: Final = (CLUSTER_SIZE, 1, 1)

# The default comm-SM count and fixed row granularities, not code-generation constants. Geometry is
# runtime data.
NUM_COMM_SMS: Final = 40
CLC_DEPTH: Final = 1
ROW_ALIGNMENT: Final = 256
HIDDEN_ROW_BLOCK: Final = 256
Y_DONE_ROW_BLOCK: Final = 128

# Executor cache identity.  The cache contains compiled code only; the launch
# plans below hold the loop-invariant arguments, and state buffers and streams
# are fresh per call.
_CODEGEN_EPOCH: Final = (
    "runtime-geometry-linear-expert-dsl48-v2-epsize-runtime-clc-drain-lane0-geomparams-backoff-taskdecode-commworkerinterleave-swapab-stm2-epicombine-tail-clip-role-elif-pfentry-commtileshape"
)
_ABI_EPOCH: Final = "persistent-bf16-nine-output-five-counter-v1"

_EXECUTOR_CACHE: dict[tuple[object, ...], object] = {}
_EXECUTOR_CACHE_DEVICES: set[int] = set()
_EXECUTOR_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class PersistentBf16Plan:
    """Runtime geometry of one call (ep_size from the workspace) plus the fixed structure: target
    architecture (SM100 / SM103) and CLC depth 1."""

    arch: str = TARGET_ARCH
    ep_size: int = 8  # runtime peer count from the workspace, not a device constant
    num_local_experts: int = NUM_LOCAL_EXPERTS
    hidden_size: int = HIDDEN_SIZE  # shared-expert hidden (x_shared, y_shared)
    routed_hidden_size: int | None = None  # routed chain hidden (x_routed, y_routed, comm rows); None = hidden_size
    intermediate_size: int = INTERMEDIATE_SIZE
    topk: int = TOPK
    macrobatch_size: int = MACROBATCH_ROWS
    minibatch_size: int = MINIBATCH_ROWS
    num_comm_sms: int = NUM_COMM_SMS
    clc_depth: int = CLC_DEPTH
    # Whether the layer has a shared expert; routed-only otherwise.
    has_shared: bool = True
    # The communication tile shape (contract CommTileShape); selects the kernel variant.
    comm_tile_shape: CommTileShape = DEFAULT_COMM_TILE_SHAPE

    @property
    def routed_hidden(self) -> int:
        return self.hidden_size if self.routed_hidden_size is None else self.routed_hidden_size

    def validate(self) -> None:
        if (self.arch, self.clc_depth) != (TARGET_ARCH, CLC_DEPTH):
            raise NotImplementedError(f"persistent BF16 requires {TARGET_ARCH} CLC1 G1D1")
        if type(self.ep_size) is not int or not 0 < self.ep_size < 2**31:
            raise ValueError(f"ep_size must be a positive Int32; got {self.ep_size!r}")
        if self.comm_tile_shape not in COMM_TILE_SHAPES:
            raise ValueError(f"comm_tile_shape must be one of {COMM_TILE_SHAPES}; got {self.comm_tile_shape!r}")
        for name in ("num_local_experts", "hidden_size", "intermediate_size", "topk", "macrobatch_size", "minibatch_size", "num_comm_sms"):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value < 2**31:
                raise ValueError(f"{name} must be a positive Int32")
        if self.routed_hidden_size is not None and (type(self.routed_hidden_size) is not int or not 0 < self.routed_hidden_size < 2**31):
            raise ValueError("routed_hidden_size must be a positive Int32")
        # The hidden extent may have a 128-wide tail (H % 256 != 0): communication copies the
        # partial tail, the Down tiles count it (ceil) and their TMA loads / stores zero-fill /
        # clip the half past H.  A routed hidden that differs from the shared one (a latent layer)
        # stays 256-aligned.
        if self.hidden_size < 256 or self.hidden_size % 128:
            raise ValueError(f"hidden_size must be at least 256 and a multiple of 128; got {self.hidden_size}")
        if self.routed_hidden != self.hidden_size and self.routed_hidden % 256:
            raise ValueError(f"routed_hidden_size must be a multiple of 256; got {self.routed_hidden}")
        # The intermediate extent is never communicated.  The forward tiles it at
        # FC1_LOGICAL_N=128 (gate/up tasks, SwiGLU) and at the K tile of 64 (Down K), so 128 is its
        # granularity.  The BF16 backward tiles it at 256 with a 128-wide tail tile.
        if self.intermediate_size < 256 or self.intermediate_size % 128:
            raise ValueError("intermediate must be at least 256 and a multiple of 128; got " f"{self.intermediate_size}")
        if self.minibatch_size % ROW_ALIGNMENT or self.macrobatch_size % self.minibatch_size:
            raise ValueError("minibatch must be 256-aligned and macrobatch a multiple of it; got " f"{self.macrobatch_size}/{self.minibatch_size}")
        if self.num_comm_sms % CLUSTER_SHAPE[0]:
            raise ValueError(f"num_comm_sms must be even; got {self.num_comm_sms}")
        if self.topk > 255:
            raise ValueError("topk must be in [1, 255], like the CUDA wrapper")


@dataclass(frozen=True)
class PersistentBf16CounterLengths:
    gate_up_tile_ready: int
    hidden_row_block_ready: int
    x_routed_ready: int
    y_routed_ready: int
    y_routed_done: int


@dataclass(frozen=True)
class PersistentBf16State:
    """The nine output tensors and five counter arrays of one forward; ``y_routed_done`` carries one extra
    word, the communication task stream's head (_fwd_dynamic_task_stream)."""

    x_routed: object
    gate_shared: object
    gate_routed: object
    up_shared: object
    up_routed: object
    hidden_shared: object
    hidden_routed: object
    y_shared: object
    y_routed: object
    gate_up_tile_ready: object
    hidden_row_block_ready: object
    x_routed_ready: object
    y_routed_ready: object
    y_routed_done: object

    @property
    def abi_outputs(self) -> tuple[object, ...]:
        return (
            self.x_routed,
            self.gate_shared,
            self.gate_routed,
            self.up_shared,
            self.up_routed,
            self.hidden_shared,
            self.hidden_routed,
            self.y_shared,
            self.y_routed,
        )

    @property
    def counters(self) -> tuple[object, ...]:
        return (
            self.gate_up_tile_ready,
            self.hidden_row_block_ready,
            self.x_routed_ready,
            self.y_routed_ready,
            self.y_routed_done,
        )


def _normalize_swiglu_limit(swiglu_limit: float | None) -> tuple[float, bool]:
    """(limit, is_clamped) of the optional clamp: None is unclamped, otherwise a non-negative number."""

    if swiglu_limit is None:
        return 0.0, False
    if type(swiglu_limit) not in (int, float) or swiglu_limit < 0:
        raise ValueError("swiglu_limit must be None or a non-negative number")
    return float(swiglu_limit), True


_RUNTIME_ENVIRONMENT: tuple[object, ...] | None = None


def _prepare_public_runtime_environment() -> tuple[object, ...]:
    """The CuTe DSL version and the compile options, for the executor key (memoized).

    The first call runs the version gate.  The target and TVM FFI are compile options
    (``_arch.COMPILE_OPTIONS``), so nothing here reads the environment or depends on when CUTLASS was
    imported.
    """

    global _RUNTIME_ENVIRONMENT
    if _RUNTIME_ENVIRONMENT is None:
        _RUNTIME_ENVIRONMENT = (
            ("nvidia-cutlass-dsl", require_cutedsl()),
            ("compile_options", _COMPILE_OPTIONS),
        )
    return _RUNTIME_ENVIRONMENT


def _tensor_abi_descriptor(name: str, tensor) -> tuple[object, ...]:
    """Describe compile-relevant tensor ABI without capturing an address."""

    return (
        name,
        len(tensor.shape),
        "contiguous-unit-last-stride-dynamic-extents-and-outer-strides",
        str(tensor.dtype),
    )


# Launch plans, keyed by the workspace buffers, the weights and the geometry: the full validation
# and the executor key run when a plan is built.  A call checks the schedule (new tensors every
# step) and the shared input against the validated layouts, allocates the state, fills the per-call
# slots and launches.
_LAUNCH_PLANS = LaunchPlanCache()


@dataclass(frozen=True)
class _FwdLaunchPlan:
    executor: object
    template: tuple[object, ...]  # launch arguments with every per-call slot emptied
    state_layout: tuple[object, ...]
    # peer_rank, peer_token_idx, num_tokens, tokens_per_expert, valid_tokens_per_expert
    schedule_layouts: tuple[object, ...]
    shared_input_layout: tuple[object, ...]  # x_shared (or workspace.x_buffer when the hiddens coincide)
    # fill_mega_launch_slots: the mega module imports CUTLASS lazily, so the plan keeps the function
    # and a call needs no import statement.
    fill_slots: object
    # The routed address table and descriptor bytes (RoutedDescriptors), built by the first launch.
    routed: object = None


def _persistent_executor_key(
    workspace,
    schedule,
    weights: tuple[object, ...],
    state: PersistentBf16State,
    *,
    plan: PersistentBf16Plan,
    swiglu_limit_value: float,
    is_clamped: bool,
    task_decode: TaskDecode,
    runtime_environment: tuple[object, ...],
    device_index: int,
    context_key: int,
    epilogue_combine: bool = False,
    split_weights: bool = False,
    is_situ: bool = False,
    has_shared: bool = True,
    shared_fc1_only: bool = False,
) -> tuple[object, ...]:
    """Key only code/ABI structure; runtime values and addresses are excluded."""

    named_tensors = (
        ("workspace.x_buffer", workspace.x_buffer),
        ("workspace.combine_buffer", workspace.combine_buffer),
        ("schedule.peer_rank", schedule.peer_rank),
        ("schedule.peer_token_idx", schedule.peer_token_idx),
        ("schedule.num_tokens", schedule.num_tokens),
        ("schedule.tokens_per_expert", schedule.tokens_per_expert),
        ("schedule.valid_tokens_per_expert", schedule.valid_tokens_per_expert),
        ("shared_gate_weights", weights[0]),
        ("routed_gate_weights", weights[1]),
        ("shared_up_weights", weights[2]),
        ("routed_up_weights", weights[3]),
        ("shared_down_weights", weights[4]),
        ("routed_down_weights", weights[5]),
        *tuple((f"state.output.{index}", tensor) for index, tensor in enumerate(state.abi_outputs)),
        *tuple((f"state.counter.{index}", tensor) for index, tensor in enumerate(state.counters)),
    )
    tensor_abi = tuple(_tensor_abi_descriptor(name, tensor) for name, tensor in named_tensors)
    return (
        "mok-persistent-bf16-executor",
        _CODEGEN_EPOCH,
        ("device", device_index),
        ("context", context_key),
        ("arch", TARGET_ARCH, TARGET_CAPABILITY),
        ("runtime", runtime_environment),
        ("swiglu", bool(is_clamped)),
        ("situ", bool(is_situ)),
        ("has_shared", bool(has_shared)),
        ("shared_fc1_only", bool(shared_fc1_only)),
        # Compile-time: the routed hidden has a 128-wide tail, one more Down tile for the communication CTAs to
        # wait for.  The shared-FC1-only launch has neither, so every size runs its one executor.
        ("has_tail", not shared_fc1_only and _has_tail(plan.routed_hidden)),
        ("task_decode", TaskDecode(task_decode).name),
        ("epilogue_combine", bool(epilogue_combine)),
        # Compile-time: the rows and columns of one communication tile.
        ("comm_tile_shape", (plan.comm_tile_shape.rows, plan.comm_tile_shape.columns)),
        ("structure", plan.clc_depth, CLUSTER_SHAPE, "bf16"),
        # The routed operands' rank differs between the two layouts and _tensor_abi_descriptor
        # keeps the rank, but the flag is what selects the code, so key it directly.
        ("routed_weights", "split" if split_weights else "grouped"),
        ("peer_pointer_abi", "int64-address-table-runtime-length"),
        ("abi", _ABI_EPOCH, tensor_abi),
    )


def select_task_decode(num_local_experts: int, requested: TaskDecode | None = None) -> TaskDecode:
    """Task->expert decode variant for a model structure (see TaskDecode in the contract):
    SMEM_BINARY when every local expert fits the shared-memory prefix, SMEM_GROUP otherwise.
    Internal: ``requested`` forces a variant (SMEM_BINARY needs E <= EXPERT_TABLE_CAP); tests force
    one by patching this function, the only selection point of forward_bf16."""

    if type(num_local_experts) is not int or num_local_experts <= 0:
        raise ValueError("num_local_experts must be a positive int")
    if requested is None:
        return TaskDecode.SMEM_BINARY if num_local_experts <= EXPERT_TABLE_CAP else TaskDecode.SMEM_GROUP
    chosen = TaskDecode(requested)
    if chosen is TaskDecode.SMEM_BINARY and num_local_experts > EXPERT_TABLE_CAP:
        raise ValueError(f"TaskDecode.SMEM_BINARY holds at most {EXPERT_TABLE_CAP} local experts, got {num_local_experts}")
    return chosen


def clear_persistent_bf16_executor_cache(*, synchronize: bool = True) -> None:
    """Release executor-only cache entries after public calls are quiescent."""

    with _EXECUTOR_CACHE_LOCK:
        if synchronize and _EXECUTOR_CACHE_DEVICES:
            import torch

            for device_index in sorted(_EXECUTOR_CACHE_DEVICES):
                torch.cuda.synchronize(device_index)
        _EXECUTOR_CACHE.clear()
        _EXECUTOR_CACHE_DEVICES.clear()
        _LAUNCH_PLANS.clear()
        clear_routed_descriptors()


def persistent_bf16_output_shapes(
    num_local_tokens: int,
    *,
    plan: PersistentBf16Plan = PersistentBf16Plan(),
) -> tuple[tuple[int, ...], ...]:
    """Shapes of the nine outputs, in ABI order: x_routed, gate_shared, gate_routed, up_shared,
    up_routed, hidden_shared, hidden_routed, y_shared, y_routed.  The routed tensors hold one
    macrobatch of rows, the shared ones ``num_local_tokens``."""

    if type(num_local_tokens) is not int or not 2 * ROW_ALIGNMENT <= num_local_tokens < 2**31 or num_local_tokens % ROW_ALIGNMENT:
        raise ValueError("num_local_tokens must be an integer multiple of 256 at least 512")
    plan.validate()
    h, hr, i, macro = plan.hidden_size, plan.routed_hidden, plan.intermediate_size, plan.macrobatch_size
    # x_routed / y_routed are routed-hidden rows; y_shared is shared-hidden.  A routed-only layer
    # has no shared outputs: one placeholder tile each, for the kernel's TMA atoms, never written.
    t_i, t_h = ((num_local_tokens, i), (num_local_tokens, h)) if plan.has_shared else ((SHARED_PLACEHOLDER_SHAPE,) * 2)
    return (
        (macro, hr),
        t_i,
        (macro, i),
        t_i,
        (macro, i),
        t_i,
        (macro, i),
        t_h,
        (macro, hr),
    )


def persistent_bf16_counter_lengths(
    num_local_tokens: int,
    schedule_capacity: int,
    *,
    plan: PersistentBf16Plan = PersistentBf16Plan(),
) -> PersistentBf16CounterLengths:
    """Lengths of the five counter arrays.  ``y_routed_done`` has one word after its 128-row blocks:
    the head of the communication task stream (``_fwd_dynamic_task_stream``), which must not alias
    the last block the compute waits for when the routed rows reach ``schedule_capacity``."""

    persistent_bf16_output_shapes(num_local_tokens, plan=plan)
    if type(schedule_capacity) is not int or not ROW_ALIGNMENT <= schedule_capacity < 2**31 or schedule_capacity % ROW_ALIGNMENT:
        raise ValueError("schedule_capacity must be a positive multiple of 256")
    shared_row_blocks = num_local_tokens // HIDDEN_ROW_BLOCK if plan.has_shared else 0
    routed_row_blocks = schedule_capacity // HIDDEN_ROW_BLOCK
    gate_up_column_blocks = plan.intermediate_size // FC1_LOGICAL_N
    # Task IDs, counters and route indices are signed 32-bit; matrix byte
    # addresses remain 64-bit and are not constrained by this task-ID bound.
    shared_columns = gate_up_column_blocks + -(-plan.hidden_size // FC1_TILE_M)
    routed_columns = gate_up_column_blocks + -(-plan.routed_hidden // FC1_TILE_M)
    capacity_minis = schedule_capacity // plan.minibatch_size + int(schedule_capacity % plan.minibatch_size != 0)
    capacity_clusters = plan.num_comm_sms // 2 + shared_row_blocks * shared_columns + capacity_minis * (plan.minibatch_size // ROW_ALIGNMENT) * routed_columns
    if capacity_clusters * 2 >= 2**31 or (shared_row_blocks + routed_row_blocks) * gate_up_column_blocks >= 2**31 or num_local_tokens * plan.topk >= 2**31:
        raise ValueError("runtime geometry exceeds the signed Int32 task/index domain")
    return PersistentBf16CounterLengths(
        gate_up_tile_ready=(shared_row_blocks + routed_row_blocks) * gate_up_column_blocks,
        hidden_row_block_ready=shared_row_blocks + routed_row_blocks,
        x_routed_ready=capacity_minis,
        y_routed_ready=capacity_minis,
        y_routed_done=schedule_capacity // Y_DONE_ROW_BLOCK + 1,  # + the communication task stream's head
    )


def _state_layout(num_local_tokens: int, schedule_capacity: int, plan: PersistentBf16Plan):
    """Shapes of the nine outputs, and the split sizes and length of the one int32 allocation that
    holds the five counters (each followed by padding to a 16-byte boundary, as the ABI assumes).
    Validates its arguments; the forward keeps the result in its launch plan."""

    shapes = persistent_bf16_output_shapes(num_local_tokens, plan=plan)
    lengths = persistent_bf16_counter_lengths(num_local_tokens, schedule_capacity, plan=plan)
    split_sizes = []
    for length in (
        lengths.gate_up_tile_ready,
        lengths.hidden_row_block_ready,
        lengths.x_routed_ready,
        lengths.y_routed_ready,
        lengths.y_routed_done,
    ):
        split_sizes += [length, -(-length // 4) * 4 - length]  # 4 int32 = 16 bytes
    counter_total = max(sum(split_sizes), 4)
    split_sizes[-1] += counter_total - sum(split_sizes)
    return shapes, tuple(split_sizes), counter_total, plan.has_shared


#: The four shared outputs' positions among the nine (gate, up, hidden, y).
SHARED_OUTPUT_INDICES: Final = (1, 3, 5, 7)


def _allocate_state(layout, device):
    """Fresh outputs (one caching-allocator call each) and counters (one zero-filled allocation
    split into the five arrays: one memset, one view op) on the current stream.  A routed-only
    layer's four shared outputs are the device's placeholder, which nothing writes."""

    import torch

    shapes, split_sizes, counter_total, has_shared = layout
    outputs = tuple(
        shared_placeholder(device) if not has_shared and index in SHARED_OUTPUT_INDICES else torch.empty(shape, dtype=torch.bfloat16, device=device)
        for index, shape in enumerate(shapes)
    )
    counters = tuple(torch.zeros(counter_total, dtype=torch.int32, device=device).split(split_sizes)[0::2])
    return outputs, counters


def _caller_state(layout, device, outputs, counters):
    """The state of a call in the execute step of a plan, which allocates nothing: ``outputs`` are the caller's
    nine buffers (None where a routed-only layer has no shared output), checked against the layout;
    ``counters`` is the plan's int32 buffer, at least as long as this call's counters, whose leading words are
    zeroed here on the current stream and split into the five arrays."""

    import torch

    shapes, split_sizes, counter_total, has_shared = layout
    outputs = tuple(outputs)
    if len(outputs) != len(shapes):
        raise ValueError(f"the forward takes {len(shapes)} output buffers; got {len(outputs)}")
    absent = () if has_shared else SHARED_OUTPUT_INDICES
    for index, shape in enumerate(shapes):
        if index not in absent:
            _validate_tensor(f"outputs[{index}]", outputs[index], shape=shape, dtype=torch.bfloat16, device=device)
    outputs = tuple(shared_placeholder(device) if index in absent else output for index, output in enumerate(outputs))
    if (
        type(counters) is not torch.Tensor
        or counters.dtype != torch.int32
        or counters.device != device
        or counters.ndim != 1
        or counters.numel() < counter_total
        or counters.data_ptr() % 16
    ):
        raise ValueError(f"counters must be a 16-byte aligned int32 [>= {counter_total}] tensor on {device}")
    return outputs, tuple(counters[:counter_total].zero_().split(split_sizes)[0::2])


#: An absent shared expert's tensors (a routed-only layer) are one 256 x 256 BF16 tile -- enough
#: for the kernel's TMA atoms, which it builds either way.  With no shared task, nothing reads or
#: writes it, so one per device stands in for every absent weight and output.
SHARED_PLACEHOLDER_SHAPE: Final = (256, 256)
_SHARED_PLACEHOLDERS: dict[object, object] = {}


def shared_placeholder(device, dtype=None):
    """The device's stand-in for a routed-only layer's absent shared tensors (never accessed),
    BF16 unless ``dtype`` says otherwise (an FP32 main_grad build's shared weight gradients)."""
    import torch

    dtype = torch.bfloat16 if dtype is None else dtype
    placeholder = _SHARED_PLACEHOLDERS.get((device, dtype))
    if placeholder is None:
        placeholder = torch.zeros(SHARED_PLACEHOLDER_SHAPE, dtype=dtype, device=device)
        _SHARED_PLACEHOLDERS[device, dtype] = placeholder
    return placeholder


def shared_presence(*tensors) -> bool:
    """Whether a layer has a shared expert: its tensors are all given or all None."""
    present = [tensor is not None for tensor in tensors]
    if any(present) != all(present):
        raise ValueError("shared tensors must be all present or all absent")
    return all(present)


#: A TMA descriptor is 128 bytes.  Duplicated from cutedsl.routed_descriptors, which imports
#: CUTLASS, because this module must import on a CPU-only host; _load_mega_runtime checks the two
#: against each other once the DSL is in.
_BYTES_PER_DESCRIPTOR = 128


def validate_up_row_offset(up_row_offset, intermediate_size: int, gate, up, gate_scale=None, up_scale=None) -> bool:
    """Whether Gate and Up are one combined FC1, checking ``up_row_offset`` against them.

    A combined FC1 is one parameter -- ``[E, 2I, H]``, or ``[2I, H]`` per split expert -- and, for
    MXFP8, its one scale, passed as both Gate and Up, with Up read ``intermediate_size`` rows down.
    Separate weights take no offset.  Anything else would read Up out of Gate's rows or past the
    parameter.
    """
    if type(up_row_offset) is not int:
        raise TypeError("up_row_offset must be an int")
    if up_row_offset == 0:
        if gate is up:
            raise ValueError("Gate and Up are one tensor, so up_row_offset must be intermediate_size")
        return False
    if up_row_offset != intermediate_size:
        raise ValueError(f"up_row_offset must be 0 or intermediate_size ({intermediate_size}); " f"got {up_row_offset}")
    if gate is not up or gate_scale is not up_scale:
        raise ValueError("a combined FC1 passes one tensor (and one scale) as both Gate and Up")
    return True


class RoutedDescriptors:
    """One set of routed operands' launch arguments: the ``(operands, E)`` int64 address table the
    kernel reads, the bytes the per-expert TMA descriptors live in, and whether they are built.

    The descriptors are built by the kernel's own launch, since their schema comes from the kernel's
    TMA atoms: on the first launch that uses this set (``build`` is 1), and never again
    (``mark_built``).  An entry lives as long as every source table does.

    Under CUDA graph capture a launch is only recorded: the descriptors (and, for an entry the
    capture created, the address snapshot) are written when the graph replays.  So a capturing
    launch leaves ``built`` false, and an entry made during a capture is ``stale``: every launch
    retakes its address snapshot from the (weakly held) source tables until an eager one has; the
    graph itself rebuilds the descriptors on every replay.
    """

    __slots__ = ("addresses", "descriptors", "built", "stale", "sources", "__weakref__")

    def __init__(self, addresses, descriptors, built, stale=False, sources=()):
        self.addresses, self.descriptors, self.built = addresses, descriptors, built
        self.stale, self.sources = stale, sources

    @property
    def build(self) -> int:
        """The kernel's ``build_descriptors`` argument for the next launch, read at that launch."""
        if self.stale:
            import torch

            tables = tuple(source() for source in self.sources)
            if any(table is None for table in tables):
                raise RuntimeError("a routed storage table these descriptors were made from was freed")
            # Now, or recorded in the capture in progress (whose replays then write it too).
            self.addresses.copy_(torch.stack(tables))
            self.stale = torch.cuda.is_current_stream_capturing()
        return 0 if self.built else 1

    def mark_built(self) -> None:
        """After the launch that built them: later launches reuse the descriptors as they are --
        unless that launch was captured into a CUDA graph, which builds them only when it replays."""
        import torch

        if not torch.cuda.is_current_stream_capturing():
            self.built = True


#: (schema, source table ids, operand count, E, device) -> RoutedDescriptors; an entry is dropped
#: when any of its source tables is freed, so a new table at a recycled id never finds a stale one.
_ROUTED_DESCRIPTORS: dict[tuple[object, ...], RoutedDescriptors] = {}
#: device -> the grouped weights' one-element stand-ins.
_GROUPED_STAND_INS: dict[str, RoutedDescriptors] = {}
_ROUTED_DESCRIPTORS_LOCK = threading.Lock()


def routed_launch_tensors(tables, num_local_experts: int, device, schema=None) -> RoutedDescriptors:
    """The routed weights' launch arguments (``RoutedDescriptors``).

    ``tables`` are the caller's per-expert address tables, one per routed operand -- three for the
    BF16 forward, six for MXFP8, where every weight is a payload and a scale.  The kernel wants
    them as one ``(operands, E)`` int64 tensor, next to the bytes it builds the per-expert TMA
    descriptors into.  ``schema`` is everything else a descriptor depends on -- the executor (its
    TMA atoms) and the layouts of the tensors the atoms are built over -- so one set is reused
    exactly while both the tables and the schema are.

    A combined FC1 shares one table between Gate and Up: the Up rows are a tile coordinate in the
    kernel, never an address offset here.

    Grouped weights get one-element stand-ins: the kernel's split path is compiled out, but the
    ABI is one shape so the executor stays positional.
    """

    import torch
    import weakref

    if tables is None:
        # One per device, never written (the split path is compiled out): a launch plan built in the execute
        # step of a plan then allocates nothing.
        with _ROUTED_DESCRIPTORS_LOCK:
            entry = _GROUPED_STAND_INS.get(str(device))
            if entry is None:
                entry = _GROUPED_STAND_INS[str(device)] = RoutedDescriptors(
                    torch.zeros(1, 1, dtype=torch.int64, device=device), torch.zeros(_BYTES_PER_DESCRIPTOR, dtype=torch.uint8, device=device), True
                )
        return entry
    tables = tuple(tables)
    key = (schema, tuple(id(table) for table in tables), len(tables), num_local_experts, str(device))
    with _ROUTED_DESCRIPTORS_LOCK:
        entry = _ROUTED_DESCRIPTORS.get(key)
        if entry is None:
            # During a CUDA graph capture torch.stack is only recorded: the entry stays stale (see
            # RoutedDescriptors) until an eager launch retakes the snapshot.
            entry = RoutedDescriptors(
                torch.stack(tables),
                torch.empty(len(tables) * num_local_experts * _BYTES_PER_DESCRIPTOR, dtype=torch.uint8, device=device),
                False,
                stale=torch.cuda.is_current_stream_capturing(),
                sources=tuple(weakref.ref(table) for table in tables),
            )
            _ROUTED_DESCRIPTORS[key] = entry
            for table in {id(table): table for table in tables}.values():
                weakref.finalize(table, _ROUTED_DESCRIPTORS.pop, key, None)
    return entry


def clear_routed_descriptors() -> None:
    """Drop every kept descriptor set (the executor caches' clear functions call this)."""
    with _ROUTED_DESCRIPTORS_LOCK:
        _ROUTED_DESCRIPTORS.clear()
        _GROUPED_STAND_INS.clear()


def prepare_persistent_bf16_state(
    *,
    num_local_tokens: int,
    schedule_capacity: int,
    device,
    plan: PersistentBf16Plan = PersistentBf16Plan(),
) -> PersistentBf16State:
    """Allocate the outputs and counters of one call without launching."""

    outputs, counters = _allocate_state(_state_layout(num_local_tokens, schedule_capacity, plan), device)
    return PersistentBf16State(*outputs, *counters)


def validate_persistent_bf16_call(
    workspace,
    schedule,
    shared_gate_weights,
    routed_gate_weights,
    shared_up_weights,
    routed_up_weights,
    shared_down_weights,
    routed_down_weights,
    state: PersistentBf16State,
    *,
    plan: PersistentBf16Plan = PersistentBf16Plan(),
    x_shared=None,
    routed_storage_tables=None,
    up_row_offset: int = 0,
) -> None:
    """Validate one call before CUTLASS is imported or the megakernel compiled; raises on a mismatch.

    ``x_shared`` is the shared-expert input ``[num_local_tokens, hidden_size]``; it defaults to
    ``workspace.x_buffer`` (the symmetric routed source) when the two hiddens coincide.

    ``routed_storage_tables`` is the Gate, Up and Down per-expert address tables when the routed
    experts are independent allocations.  The routed weights are then the representative
    expert alone -- two-dimensional, since split allocations have no common L stride -- and what
    has to be checked is that the three tables agree with each other and with that representative,
    because the kernel keeps one descriptor schema and only retargets its address.

    ``up_row_offset`` is ``intermediate_size`` when Gate and Up are one combined FC1 passed as
    both arguments (``[E, 2I, H]``, or ``[2I, H]`` per split expert, sharing one table), and 0
    for separate weights; see ``validate_up_row_offset``.
    """

    import torch

    plan.validate()
    required_workspace = (
        "device",
        "ep_size",
        "num_local_tokens",
        "hidden_size",
        "topk",
        "schedule_capacity",
        "x_buffer",
        "x_buffer_ptrs",
        "combine_buffer_ptrs",
        "combine_buffer",
        "ep_rank",
    )
    missing = tuple(name for name in required_workspace if not hasattr(workspace, name))
    if missing:
        raise TypeError(f"workspace is missing fields: {missing}")
    device = workspace.device
    if (
        workspace.ep_size != plan.ep_size
        or workspace.hidden_size != plan.hidden_size
        or getattr(workspace, "routed_hidden_size", workspace.hidden_size) != plan.routed_hidden
        or workspace.topk != plan.topk
    ):
        raise ValueError("workspace geometry does not match the runtime plan")
    hidden_size, intermediate_size = plan.hidden_size, plan.intermediate_size
    routed_hidden_size = plan.routed_hidden
    num_local_experts, topk = plan.num_local_experts, plan.topk
    persistent_bf16_output_shapes(workspace.num_local_tokens, plan=plan)
    lengths = persistent_bf16_counter_lengths(
        workspace.num_local_tokens,
        workspace.schedule_capacity,
        plan=plan,
    )
    check_device(device, "persistent BF16 forward")
    for name in ("x_buffer_ptrs", "combine_buffer_ptrs"):
        pointers = getattr(workspace, name)
        if (
            not isinstance(pointers, list)
            or len(pointers) != plan.ep_size
            or any(type(pointer) is not int or pointer <= 0 or pointer % 16 for pointer in pointers)
        ):
            raise ValueError(f"workspace.{name} must contain ep_size={plan.ep_size} aligned addresses")

    _validate_tensor(
        "workspace.x_buffer",
        workspace.x_buffer,
        shape=(workspace.num_local_tokens, routed_hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    if x_shared is None:
        if routed_hidden_size != hidden_size:
            raise ValueError("x_shared is required when the routed hidden differs from the shared hidden")
    else:
        _validate_tensor(
            "x_shared",
            x_shared,
            shape=(workspace.num_local_tokens, hidden_size),
            dtype=torch.bfloat16,
            device=device,
        )
    _validate_tensor(
        "workspace.combine_buffer",
        workspace.combine_buffer,
        shape=(workspace.num_local_tokens * topk, routed_hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    if not 0 <= workspace.ep_rank < plan.ep_size:
        raise ValueError(f"workspace.ep_rank is outside [0, ep_size={plan.ep_size})")
    if workspace.x_buffer_ptrs[workspace.ep_rank] != workspace.x_buffer.data_ptr():
        raise ValueError("workspace.x_buffer_ptrs does not own the local x buffer")
    if workspace.combine_buffer_ptrs[workspace.ep_rank] != workspace.combine_buffer.data_ptr():
        raise ValueError("workspace.combine_buffer_ptrs does not own the local combine buffer")
    split_weights = routed_storage_tables is not None
    if split_weights:
        if len(routed_storage_tables) != 3:
            raise ValueError("routed_storage_tables must hold the Gate, Up and Down per-expert address tables")
        for name, table in zip(("gate", "up", "down"), routed_storage_tables):
            _validate_tensor(
                f"routed_storage_tables.{name}",
                table,
                shape=(num_local_experts,),
                dtype=torch.int64,
                device=device,
            )
    combined = validate_up_row_offset(up_row_offset, intermediate_size, routed_gate_weights, routed_up_weights)
    # Split: the representative expert alone, one per routed weight.  Grouped: the whole
    # [E, N, K] parameter, whose L mode the kernel slices.
    fc1_shape = ((2 if combined else 1) * intermediate_size, routed_hidden_size)
    down_shape = (routed_hidden_size, intermediate_size)
    if not split_weights:
        fc1_shape = (num_local_experts, *fc1_shape)
        down_shape = (num_local_experts, *down_shape)
    shared_fc1 = (intermediate_size, hidden_size) if plan.has_shared else SHARED_PLACEHOLDER_SHAPE
    shared_down = (hidden_size, intermediate_size) if plan.has_shared else SHARED_PLACEHOLDER_SHAPE
    weight_specs = (
        ("shared_gate_weights", shared_gate_weights, shared_fc1),
        ("routed_gate_weights", routed_gate_weights, fc1_shape),
        ("shared_up_weights", shared_up_weights, shared_fc1),
        ("routed_up_weights", routed_up_weights, fc1_shape),
        ("shared_down_weights", shared_down_weights, shared_down),
        ("routed_down_weights", routed_down_weights, down_shape),
    )
    for name, tensor, shape in weight_specs:
        _validate_tensor(
            name,
            tensor,
            shape=shape,
            dtype=torch.bfloat16,
            device=device,
            # A routed weight may be a slice of a larger parameter (Gate and Up as the two halves
            # of one combined [E, 2I, H] tensor); each gets its own TMA descriptor.
            sliced=name.startswith("routed_"),
        )

    if getattr(schedule, "valid_tokens_per_expert", None) is None:
        raise ValueError(
            "schedule.valid_tokens_per_expert is missing: the CuTe DSL forward needs a schedule built by "
            "the CuTe DSL scheduler (build_schedule with CuteMoKConfig)"
        )
    schedule_specs = (
        ("schedule.peer_rank", schedule.peer_rank, (workspace.schedule_capacity,)),
        (
            "schedule.peer_token_idx",
            schedule.peer_token_idx,
            (workspace.schedule_capacity,),
        ),
        ("schedule.num_tokens", schedule.num_tokens, (1,)),
        (
            "schedule.tokens_per_expert",
            schedule.tokens_per_expert,
            (num_local_experts,),
        ),
        (
            "schedule.valid_tokens_per_expert",
            schedule.valid_tokens_per_expert,
            (num_local_experts,),
        ),
    )
    for name, tensor, shape in schedule_specs:
        _validate_tensor(
            name,
            tensor,
            shape=shape,
            dtype=torch.int32,
            device=device,
        )

    if not isinstance(state, PersistentBf16State):
        raise TypeError("state must be a PersistentBf16State")
    for name, tensor, shape in zip(
        (
            "x_routed",
            "gate_shared",
            "gate_routed",
            "up_shared",
            "up_routed",
            "hidden_shared",
            "hidden_routed",
            "y_shared",
            "y_routed",
        ),
        state.abi_outputs,
        persistent_bf16_output_shapes(workspace.num_local_tokens, plan=plan),
    ):
        _validate_tensor(
            f"state.{name}",
            tensor,
            shape=shape,
            dtype=torch.bfloat16,
            device=device,
        )
    for name, tensor, length in zip(
        (
            "gate_up_tile_ready",
            "hidden_row_block_ready",
            "x_routed_ready",
            "y_routed_ready",
            "y_routed_done",
        ),
        state.counters,
        (
            lengths.gate_up_tile_ready,
            lengths.hidden_row_block_ready,
            lengths.x_routed_ready,
            lengths.y_routed_ready,
            lengths.y_routed_done,
        ),
    ):
        _validate_tensor(
            f"state.{name}",
            tensor,
            shape=(length,),
            dtype=torch.int32,
            device=device,
        )


def _mega_call_arguments(
    workspace,
    schedule,
    weights: tuple[object, ...],
    state: PersistentBf16State,
    *,
    plan: PersistentBf16Plan,
    swiglu_limit_value: float,
    is_clamped: bool,
    stream,
    x_shared=None,
    routed_storage_tables=None,
    routed_descriptors=None,
    up_row_offset: int = 0,
    situ_beta=None,
    build_descriptors: int = 1,
) -> tuple[tuple[object, ...], dict[str, object]]:
    return (
        (
            workspace.x_buffer_ptrs,
            workspace.combine_buffer_ptrs,
            schedule.peer_rank,
            schedule.peer_token_idx,
            schedule.num_tokens,
            schedule.tokens_per_expert,
            schedule.valid_tokens_per_expert,
            workspace.x_buffer if x_shared is None else x_shared,
            *weights,
            state,
        ),
        {
            "num_local_tokens": workspace.num_local_tokens,
            "peer_tables": workspace_peer_tables(workspace),
            "schedule_capacity": workspace.schedule_capacity,
            "macrobatch_size": plan.macrobatch_size,
            "minibatch_size": plan.minibatch_size,
            "num_comm_sms": plan.num_comm_sms,
            "topk": plan.topk,
            "swiglu_limit": swiglu_limit_value,
            "stream": stream,
            "routed_storage_tables": routed_storage_tables,
            "routed_descriptors": routed_descriptors,
            "build_descriptors": build_descriptors,
            "up_row_offset": up_row_offset,
            "situ_beta": (0.0, 0.0) if situ_beta is None else situ_beta,
        },
    )


def _prepare_cached_mega_launch(
    key: tuple[object, ...],
    device_index: int,
    positional_args: tuple[object, ...],
    keyword_args: dict[str, object],
    *,
    is_clamped: bool,
    task_decode: TaskDecode,
    epilogue_combine: bool = False,
    split_weights: bool = False,
    is_situ: bool = False,
    has_shared: bool = True,
    shared_fc1_only: bool = False,
    has_tail: bool = False,
    comm_tile_shape: CommTileShape = DEFAULT_COMM_TILE_SHAPE,
) -> tuple[object, tuple[object, ...]]:
    """Get code by static signature and bind only this call's addresses."""

    make_mega_args, prepare_mega_bf16 = _load_mega_runtime()

    with _EXECUTOR_CACHE_LOCK:
        executor = _EXECUTOR_CACHE.get(key)
        if executor is None:
            executor = persistent_executor(
                process_independent(key),
                lambda: prepare_mega_bf16(
                    *positional_args,
                    is_clamped=is_clamped,
                    task_decode=task_decode,
                    epilogue_combine=epilogue_combine,
                    split_weights=split_weights,
                    is_situ=is_situ,
                    has_shared=has_shared,
                    shared_fc1_only=shared_fc1_only,
                    has_tail=has_tail,
                    comm_tile_shape=comm_tile_shape,
                    **keyword_args,
                ),
            )
            _EXECUTOR_CACHE[key] = executor
            _EXECUTOR_CACHE_DEVICES.add(device_index)

    # The compiled function is a TVM FFI function: pass the torch views and plain
    # scalars directly (CuTe DSL TVM FFI guide), not CuTe tensors built per call.
    return executor, make_mega_args(*positional_args, launch=True, **keyword_args)


def _load_mega_runtime():
    """Import CUTLASS only after the CuTe DSL version gate succeeds."""

    from ._persistent_bf16_fwd_mega import make_mega_args, prepare_mega_bf16
    from .routed_descriptors import BYTES_PER_DESCRIPTOR

    if BYTES_PER_DESCRIPTOR != _BYTES_PER_DESCRIPTOR:
        raise RuntimeError(f"a TMA descriptor is {BYTES_PER_DESCRIPTOR} bytes, not the {_BYTES_PER_DESCRIPTOR} " "this module sizes the descriptor buffer with")
    return make_mega_args, prepare_mega_bf16


def routed_expert_count(routed_storage_tables, routed_gate_weights) -> int:
    """How many local experts a call has.

    Grouped: the routed weight's leading extent.  Split: the address table's length -- the weight
    is then one expert's own allocation, so its shape says nothing about how many there are.  A
    malformed argument gives -1, which PersistentBf16Plan.validate rejects with the rest of the
    geometry rather than here, out of context.
    """
    if routed_storage_tables is None:
        return getattr(routed_gate_weights, "shape", (-1,))[0]
    try:
        shape = routed_storage_tables[0].shape
    except (AttributeError, IndexError, KeyError, TypeError):
        return -1
    return shape[0] if len(shape) == 1 else -1


def _build_launch_plan(
    workspace,
    schedule,
    weights: tuple[object, ...],
    *,
    macrobatch_size: int,
    minibatch_size: int,
    swiglu_limit: float | None,
    num_comm_sms: int,
    device_index: int,
    context_key: int,
    x_shared=None,
    epilogue_combine: bool = False,
    routed_storage_tables=None,
    up_row_offset: int = 0,
    situ_beta=None,
    has_shared: bool = True,
    outputs=None,
    counters=None,
    comm_tile_shape=None,
) -> tuple[_FwdLaunchPlan, tuple[object, ...], tuple[object, ...]]:
    """Validate one set of inputs and geometry, get its executor and build its launch plan.
    Returns the plan and the validated state (outputs, counters), which the first call uses."""

    plan = PersistentBf16Plan(
        ep_size=getattr(workspace, "ep_size", -1),
        num_local_experts=routed_expert_count(routed_storage_tables, weights[1]),
        hidden_size=getattr(workspace, "hidden_size", -1),
        routed_hidden_size=getattr(workspace, "routed_hidden_size", None),
        # From the routed FC1 (minus the Up offset when combined): a routed-only layer has no
        # shared weights to read it from.
        intermediate_size=getattr(weights[1], "shape", (-1, -1))[-2] - up_row_offset,
        has_shared=has_shared,
        topk=getattr(workspace, "topk", -1),
        macrobatch_size=macrobatch_size,
        minibatch_size=minibatch_size,
        num_comm_sms=num_comm_sms,
    )
    plan.validate()
    plan = replace(plan, comm_tile_shape=_host_common.comm_tile_shape(plan.routed_hidden, comm_tile_shape))
    task_decode = select_task_decode(plan.num_local_experts)
    swiglu_limit_value, is_clamped = _normalize_swiglu_limit(swiglu_limit)
    workspace_index = workspace.device.index if workspace.device.index is not None else device_index
    if workspace_index != device_index:
        raise RuntimeError("the workspace CUDA device must be current")
    runtime_environment = _prepare_public_runtime_environment()
    layout = _state_layout(workspace.num_local_tokens, workspace.schedule_capacity, plan)
    outputs, counters = _allocate_state(layout, workspace.device) if outputs is None else _caller_state(layout, workspace.device, outputs, counters)
    state = PersistentBf16State(*outputs, *counters)
    validate_persistent_bf16_call(
        workspace, schedule, *weights, state, plan=plan, x_shared=x_shared, routed_storage_tables=routed_storage_tables, up_row_offset=up_row_offset
    )
    key = _persistent_executor_key(
        workspace,
        schedule,
        weights,
        state,
        plan=plan,
        swiglu_limit_value=swiglu_limit_value,
        is_clamped=is_clamped,
        task_decode=task_decode,
        runtime_environment=runtime_environment,
        device_index=device_index,
        context_key=context_key,
        epilogue_combine=epilogue_combine,
        split_weights=routed_storage_tables is not None,
        is_situ=situ_beta is not None,
        has_shared=has_shared,
    )
    # The descriptors' schema: the executor's TMA atoms over these routed weights' layouts.
    routed = routed_launch_tensors(
        routed_storage_tables, plan.num_local_experts, workspace.device, schema=(key, tuple(tensor_layout(weight) for weight in weights[1::2]))
    )
    stream = current_stream_handle(device_index)
    positional_args, keyword_args = _mega_call_arguments(
        workspace,
        schedule,
        weights,
        state,
        plan=plan,
        swiglu_limit_value=swiglu_limit_value,
        is_clamped=is_clamped,
        stream=stream,
        x_shared=x_shared,
        routed_storage_tables=routed.addresses,
        routed_descriptors=routed.descriptors,
        up_row_offset=up_row_offset,
        situ_beta=situ_beta,
        build_descriptors=routed.build,
    )
    executor, runtime_args = _prepare_cached_mega_launch(
        key,
        device_index,
        positional_args,
        keyword_args,
        is_clamped=is_clamped,
        task_decode=task_decode,
        epilogue_combine=epilogue_combine,
        split_weights=routed_storage_tables is not None,
        is_situ=situ_beta is not None,
        has_shared=has_shared,
        has_tail=_has_tail(plan.routed_hidden),
        comm_tile_shape=plan.comm_tile_shape,
    )
    # The TVM FFI executor is positional; the keyword dict is built in the kernel __call__
    # parameter order (checked in prepare_mega_bf16).  The kept template holds only the peer
    # tables and scalars, so a cached plan keeps no workspace, schedule, weight or state memory.
    from ._persistent_bf16_fwd_mega import MEGA_PER_CALL_SLOTS, fill_mega_launch_slots

    schedule_tensors = _schedule_tensors(schedule)
    shared_input = workspace.x_buffer if x_shared is None else x_shared
    template = checked_template(
        runtime_args.values(),
        MEGA_PER_CALL_SLOTS,
        lambda args: fill_mega_launch_slots(args, shared_input, schedule_tensors, weights, outputs, counters, stream, routed.build),
        "fill_mega_launch_slots positions do not match make_mega_args",
    )
    launch_plan = _FwdLaunchPlan(
        executor=executor,
        template=template,
        state_layout=layout,
        schedule_layouts=tuple(tensor_layout(tensor) for tensor in schedule_tensors),
        shared_input_layout=tensor_layout(shared_input),
        fill_slots=fill_mega_launch_slots,
        routed=routed,
    )
    return launch_plan, outputs, counters


def _schedule_tensors(schedule) -> tuple[object, ...]:
    """The schedule tensors of the launch, in slot order (a schedule without
    valid_tokens_per_expert fails the full validation)."""
    return (schedule.peer_rank, schedule.peer_token_idx, schedule.num_tokens, schedule.tokens_per_expert, schedule.valid_tokens_per_expert)


def forward_bf16(
    workspace,
    schedule,
    shared_gate_weights,
    routed_gate_weights,
    shared_up_weights,
    routed_up_weights,
    shared_down_weights,
    routed_down_weights,
    *,
    macrobatch_size: int,
    minibatch_size: int,
    swiglu_limit: float | None,
    num_comm_sms: int = NUM_COMM_SMS,
    x_shared=None,
    epilogue_combine: bool = False,
    comm_tile_shape=None,
    routed_storage_tables=None,
    up_row_offset: int = 0,
    situ_beta=None,
    outputs=None,
    counters=None,
) -> tuple[object, ...]:
    """Run the BF16 forward megakernel with runtime geometry.  Returns the nine outputs in the order of
    ``PersistentBf16State.abi_outputs``.

    ``outputs`` / ``counters`` (the execute step of a plan, both or neither): the caller's nine output buffers
    in that order, shaped as ``persistent_bf16_output_shapes`` gives (None where a routed-only layer has no
    shared output), and the plan's int32 counter buffer; the call then allocates nothing and returns the
    caller's buffers.  By default the call allocates its outputs and counters.

    ``macrobatch_size`` / ``minibatch_size`` (rows) and ``num_comm_sms`` are runtime launch values;
    ``swiglu_limit`` is the clamp value, or None for the unclamped variant.

    ``epilogue_combine`` (a compiled variant): the routed Down epilogue stores the combine rows into the source
    ranks' combine buffers and the communication CTAs only pull X; with ``False`` the communication CTAs read the
    Y ring and push it.  The combine buffers receive the same rows; the ``y_routed`` output is not written with
    ``True``.

    ``comm_tile_shape`` (a compiled variant; ``CuteMoKConfig.comm_tile_shape``): rows x columns of one
    communication tile, one of the contract's COMM_TILE_SHAPES or its (rows, columns); None takes
    ``_host_common.comm_tile_shape``'s default.  Only how the communication CTAs move bytes; same outputs.

    ``x_shared`` (``[num_local_tokens, hidden_size]``) is the shared-expert input; when omitted
    the symmetric ``workspace.x_buffer`` serves both chains (shared hidden == routed hidden).

    ``routed_storage_tables`` are the Gate, Up and Down per-expert base addresses when the routed
    experts are independent allocations (``cudnn.moe_ep._cutemok.ops.make_routed_weight_storage_table_bf16`` builds
    one).  The three routed weights are then the representative expert alone and the kernel names
    the others through per-expert TMA descriptors.  The tables join the plan key, so a rebuilt
    table is a new plan and a kept one can never be stale.

    ``up_row_offset``: ``intermediate_size`` when Gate and Up are one combined FC1 passed as both
    arguments, 0 for separate weights; the kernel applies it as a tile coordinate.

    ``situ_beta``: SiTU-GLU's ``(beta1, beta2)`` instead of SwiGLU, or None.

    The three shared weights all None: a routed-only layer.  There are no shared tasks, and the
    four shared outputs come back None.

    The executor and the launch plan of each workspace, set of weights and geometry are cached;
    the schedule and the shared input are checked against the validated layouts on every call,
    and the state (outputs and counters) and the current stream are fresh for every invocation.
    """

    import torch

    situ_beta = normalize_situ_beta(situ_beta, swiglu_limit)
    has_shared = shared_presence(shared_gate_weights, shared_up_weights, shared_down_weights)
    if not has_shared:
        shared_gate_weights = shared_up_weights = shared_down_weights = shared_placeholder(workspace.device)
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
    try:
        if tables is not None:
            tables = tuple(tables)
        # The address tables are plan sources: the plan keeps the (3, E) table derived from them.
        # The weights are matched by address and layout (LaunchPlanCache views): the template refills
        # them on every call, and a caller may pass new Gate / Up row views of one FC1 on every call.
        sources = (workspace.x_buffer, workspace.combine_buffer, *(tables or ()))
        schedule_tensors = _schedule_tensors(schedule)
        shared_input = workspace.x_buffer if x_shared is None else x_shared
        # Everything else the validation and the arguments read: geometry, the workspace scalars
        # and peer addresses (typed), whether the shared input is separate, device and context,
        # and the task->expert decode variant selected for this expert count.
        extra = typed_key(
            macrobatch_size,
            minibatch_size,
            num_comm_sms,
            swiglu_limit,
            workspace.ep_size,
            workspace.ep_rank,
            workspace.hidden_size,
            getattr(workspace, "routed_hidden_size", None),
            workspace.topk,
            workspace.num_local_tokens,
            workspace.schedule_capacity,
            workspace.x_buffer_ptrs,
            workspace.combine_buffer_ptrs,
            x_shared is None,
            bool(epilogue_combine),
            tables is None,
            up_row_offset,
            situ_beta,
            has_shared,
        ) + (
            device_index,
            context_key,
            workspace.device,
            select_task_decode(routed_expert_count(tables, routed_gate_weights)).name,
            _host_common.comm_tile_shape(getattr(workspace, "routed_hidden_size", None) or workspace.hidden_size, comm_tile_shape),
        )
        launch_plan = _LAUNCH_PLANS.get(sources, extra, weights)
    except (AttributeError, TypeError, ValueError) as error:
        launch_plan = None
        key_error = error
    else:
        key_error = None
    if (
        launch_plan is not None
        and layouts_match(schedule_tensors, launch_plan.schedule_layouts, 16)
        and layouts_match((shared_input,), (launch_plan.shared_input_layout,), 16)
    ):
        outputs, counters = (
            _allocate_state(launch_plan.state_layout, workspace.device)
            if outputs is None
            else _caller_state(launch_plan.state_layout, workspace.device, outputs, counters)
        )
    else:
        # A new plan; malformed arguments that broke the key are reported by the full validation,
        # as on any first call.
        launch_plan, outputs, counters = _build_launch_plan(
            workspace,
            schedule,
            weights,
            macrobatch_size=macrobatch_size,
            minibatch_size=minibatch_size,
            swiglu_limit=swiglu_limit,
            num_comm_sms=num_comm_sms,
            device_index=device_index,
            context_key=context_key,
            x_shared=x_shared,
            epilogue_combine=bool(epilogue_combine),
            routed_storage_tables=tables,
            up_row_offset=up_row_offset,
            situ_beta=situ_beta,
            has_shared=has_shared,
            outputs=outputs,
            counters=counters,
            comm_tile_shape=comm_tile_shape,
        )
        if key_error is not None:
            raise key_error
        _LAUNCH_PLANS.put(sources, extra, launch_plan, weights)
    # Fresh state on the current stream (the counters are zeroed there), launched on the same
    # stream.  No record_stream: like PyTorch's own ops, inputs are used on the current stream,
    # and a caller that frees them from another stream synchronizes that stream itself.
    routed = launch_plan.routed
    launch_plan.executor(
        *launch_plan.fill_slots(
            list(launch_plan.template), shared_input, schedule_tensors, weights, outputs, counters, current_stream_handle(device_index), routed.build
        )
    )
    routed.mark_built()
    if not has_shared:
        outputs = tuple(None if index in SHARED_OUTPUT_INDICES else output for index, output in enumerate(outputs))
    return outputs


def forward_bf16_counter_words(
    num_local_tokens: int, schedule_capacity: int, *, hidden_size: int, routed_hidden_size: int | None, intermediate_size: int, topk: int
) -> int:
    """How many int32 words the five counters of any forward over these sizes take: the length of a plan's
    counter buffer.  The minibatch size is a runtime value of every call and the counters grow as it shrinks,
    so this is the length at the smallest one (256 rows); the macrobatch size and the communication SMs do not
    change it."""
    plan = PersistentBf16Plan(
        num_local_experts=1,
        hidden_size=hidden_size,
        routed_hidden_size=routed_hidden_size,
        intermediate_size=intermediate_size,
        topk=topk,
        macrobatch_size=ROW_ALIGNMENT,
        minibatch_size=ROW_ALIGNMENT,
        num_comm_sms=CLUSTER_SIZE,
    )
    return _state_layout(num_local_tokens, schedule_capacity, plan)[2]


def compile_forward_bf16(
    workspace,
    *,
    num_local_experts: int,
    is_clamped: bool,
    is_situ: bool = False,
    has_shared: bool = True,
    split_weights: bool = False,
    epilogue_combine: bool = False,
    comm_tile_shape=None,
) -> None:
    """The plan step: compile the megakernel variant a layer runs, without launching.

    The variant is what the executor key holds besides the tensor ranks and dtypes: the SwiGLU clamp, SiTU-GLU,
    the shared expert, grouped or split routed weights, the combine placement, the communication tile shape
    (``comm_tile_shape`` as in ``forward_bf16``), the task decode (from ``num_local_experts``) and the routed
    hidden's tail (from the workspace).  Sizes are runtime values of the kernel, so every tensor argument of the
    compile is a stand-in of the right rank and dtype: the device's 256 x 256 BF16 tile for the weights and
    outputs, one int32 word for the schedule and the counters.  The workspace gives the peer address tables, which
    this builds too.  A later ``forward_bf16`` over real tensors computes the same key and finds the executor.
    """
    import torch
    from types import SimpleNamespace

    device = workspace.device
    plan = PersistentBf16Plan(
        ep_size=workspace.ep_size,
        num_local_experts=num_local_experts,
        hidden_size=workspace.hidden_size,
        routed_hidden_size=getattr(workspace, "routed_hidden_size", None),
        intermediate_size=FC1_TILE_M,
        topk=workspace.topk,
        macrobatch_size=ROW_ALIGNMENT,
        minibatch_size=ROW_ALIGNMENT,
        num_comm_sms=CLUSTER_SIZE,
        has_shared=has_shared,
    )
    plan.validate()
    plan = replace(plan, comm_tile_shape=_host_common.comm_tile_shape(plan.routed_hidden, comm_tile_shape))
    check_device(device, "persistent BF16 forward")
    device_index = _current_device_index()
    if (device.index if device.index is not None else device_index) != device_index:
        raise RuntimeError("the workspace CUDA device must be current")
    runtime_environment = _prepare_public_runtime_environment()
    placeholder = shared_placeholder(device)
    routed_weight = placeholder if split_weights else placeholder.unsqueeze(0)  # [N, K] or [E, N, K]
    weights = (placeholder, routed_weight, placeholder, routed_weight, placeholder, routed_weight)
    zero = torch.zeros(4, dtype=torch.int32, device=device)[:1]
    schedule = SimpleNamespace(peer_rank=zero, peer_token_idx=zero, num_tokens=zero, tokens_per_expert=zero, valid_tokens_per_expert=zero)
    state = PersistentBf16State(*(placeholder,) * 9, *(zero,) * 5)
    task_decode = select_task_decode(num_local_experts)
    is_clamped, is_situ = bool(is_clamped), bool(is_situ)
    key = _persistent_executor_key(
        workspace,
        schedule,
        weights,
        state,
        plan=plan,
        swiglu_limit_value=0.0,
        is_clamped=is_clamped,
        task_decode=task_decode,
        runtime_environment=runtime_environment,
        device_index=device_index,
        context_key=_current_cuda_context_key(),
        epilogue_combine=bool(epilogue_combine),
        split_weights=bool(split_weights),
        is_situ=is_situ,
        has_shared=has_shared,
    )
    stand_ins = routed_launch_tensors(None, num_local_experts, device)
    positional_args, keyword_args = _mega_call_arguments(
        workspace,
        schedule,
        weights,
        state,
        plan=plan,
        swiglu_limit_value=0.0,
        is_clamped=is_clamped,
        stream=current_stream_handle(device_index),
        routed_storage_tables=stand_ins.addresses,
        routed_descriptors=stand_ins.descriptors,
        situ_beta=(1.0, 1.0) if is_situ else None,
        build_descriptors=0,
    )
    _prepare_cached_mega_launch(
        key,
        device_index,
        positional_args,
        keyword_args,
        is_clamped=is_clamped,
        task_decode=task_decode,
        epilogue_combine=bool(epilogue_combine),
        split_weights=bool(split_weights),
        is_situ=is_situ,
        has_shared=has_shared,
        has_tail=_has_tail(plan.routed_hidden),
        comm_tile_shape=plan.comm_tile_shape,
    )


def shared_fc1_bf16(workspace, x_shared, shared_gate_weights, shared_up_weights, *, swiglu_limit, situ_beta=None):
    """Run the forward megakernel's shared FC1 tasks alone and return ``(gate_shared, up_shared, hidden_shared)``,
    three fresh ``[num_local_tokens, intermediate_size]`` tensors: what ``forward_bf16`` keeps in its context for
    the same input and weights, bit for bit (the same task code over the same tiles).

    The launch is the megakernel's ``shared_fc1_only`` variant: no communication cluster, no routed task and no
    shared Down.  It takes no schedule, reads no symmetric buffer and waits for no other rank.  The kernel's other
    arguments are placeholders it builds descriptors over: the device's 256 x 256 tile for every routed tensor,
    the Down weights and ``y_shared``, and one zero word for every schedule tensor and routed counter.

    ``x_shared``: the shared expert's input, ``[num_local_tokens, hidden_size]``.  ``swiglu_limit`` and
    ``situ_beta`` as in ``forward_bf16``.  One executor per clamp mode and activation serves every size.  A call
    validates its arguments and builds its launch arguments; no launch plan is kept.
    """

    import torch
    from types import SimpleNamespace

    situ_beta = normalize_situ_beta(situ_beta, swiglu_limit)
    swiglu_limit_value, is_clamped = _normalize_swiglu_limit(swiglu_limit)
    device, tokens = workspace.device, workspace.num_local_tokens
    # The shared expert's sizes; the routed fields describe the placeholders (one expert, 256-row batches).
    plan = PersistentBf16Plan(
        ep_size=workspace.ep_size,
        num_local_experts=1,
        hidden_size=workspace.hidden_size,
        intermediate_size=shared_gate_weights.shape[0] if getattr(shared_gate_weights, "ndim", 0) == 2 else -1,
        topk=workspace.topk,
        macrobatch_size=ROW_ALIGNMENT,
        minibatch_size=ROW_ALIGNMENT,
        num_comm_sms=CLUSTER_SIZE,
    )
    shared_shape = persistent_bf16_output_shapes(tokens, plan=plan)[1]  # validates the plan and the token count
    check_device(device, "persistent BF16 forward")
    device_index = _current_device_index()
    if (device.index if device.index is not None else device_index) != device_index:
        raise RuntimeError("the workspace CUDA device must be current")
    _validate_tensor("x_shared", x_shared, shape=(tokens, plan.hidden_size), dtype=torch.bfloat16, device=device)
    for name, weight in (("shared_gate_weights", shared_gate_weights), ("shared_up_weights", shared_up_weights)):
        _validate_tensor(name, weight, shape=(plan.intermediate_size, plan.hidden_size), dtype=torch.bfloat16, device=device)
    runtime_environment = _prepare_public_runtime_environment()
    placeholder = shared_placeholder(device)
    routed_weight = placeholder.unsqueeze(0)  # [E, N, K] with one expert
    weights = (shared_gate_weights, routed_weight, shared_up_weights, routed_weight, placeholder, routed_weight)
    outputs = tuple(torch.empty(shared_shape, dtype=torch.bfloat16, device=device) for _ in range(3))
    # One zeroed allocation on the current stream: a word that stands in for every schedule tensor and routed
    # counter, then, 16-byte aligned as the ABI assumes, hidden_row_block_ready -- the one counter a shared FC1
    # epilogue arrives on.
    words = torch.zeros(4 + tokens // HIDDEN_ROW_BLOCK, dtype=torch.int32, device=device)
    zero = words[:1]
    schedule = SimpleNamespace(peer_rank=zero, peer_token_idx=zero, num_tokens=zero, tokens_per_expert=zero, valid_tokens_per_expert=zero)
    state = PersistentBf16State(
        placeholder, outputs[0], placeholder, outputs[1], placeholder, outputs[2], placeholder, placeholder, placeholder, zero, words[4:], zero, zero, zero
    )
    task_decode = select_task_decode(plan.num_local_experts)
    key = _persistent_executor_key(
        workspace,
        schedule,
        weights,
        state,
        plan=plan,
        swiglu_limit_value=swiglu_limit_value,
        is_clamped=is_clamped,
        task_decode=task_decode,
        runtime_environment=runtime_environment,
        device_index=device_index,
        context_key=_current_cuda_context_key(),
        is_situ=situ_beta is not None,
        shared_fc1_only=True,
    )
    positional_args, keyword_args = _mega_call_arguments(
        workspace,
        schedule,
        weights,
        state,
        plan=plan,
        swiglu_limit_value=swiglu_limit_value,
        is_clamped=is_clamped,
        stream=current_stream_handle(device_index),
        x_shared=x_shared,
        situ_beta=situ_beta,
        build_descriptors=0,
    )
    executor, runtime_args = _prepare_cached_mega_launch(
        key, device_index, positional_args, keyword_args, is_clamped=is_clamped, task_decode=task_decode, is_situ=situ_beta is not None, shared_fc1_only=True
    )
    executor(*runtime_args.values())
    return outputs


__all__ = [
    "CLC_DEPTH",
    "CLUSTER_SHAPE",
    "FC1_LOGICAL_N",
    "FC1_TILE_M",
    "PersistentBf16CounterLengths",
    "PersistentBf16Plan",
    "PersistentBf16State",
    "clear_persistent_bf16_executor_cache",
    "compile_forward_bf16",
    "forward_bf16",
    "forward_bf16_counter_words",
    "persistent_bf16_counter_lengths",
    "persistent_bf16_output_shapes",
    "prepare_persistent_bf16_state",
    "routed_expert_count",
    "RoutedDescriptors",
    "clear_routed_descriptors",
    "routed_launch_tensors",
    "select_task_decode",
    "shared_fc1_bf16",
    "validate_persistent_bf16_call",
]
