# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Host of the MXFP8 forward megakernel: routed experts on e4m3 with e8m0 scales, one launch.

The shared expert stays BF16; routed weights arrive as ``(float8_e4m3fn, uint8 scale)`` pairs from
``mxfp8_quantize``.  With the default ``gate_up_context="mxfp8"`` the forward returns thirteen tensors (the
routed Gate / Up context as e4m3 / scale pairs); with ``"bf16"`` eleven: that context is then BF16 (the
epilogue's rounded accumulators, no scales).  The other context tensors (the transposed x / Hidden pairs, the
shared Gate / Up / Hidden) are the same in both.

Geometry is runtime: the executor is keyed on the clamp mode, the task-decode variant, the Gate / Up context
precision, the kernel's variant flags and the runtime environment, never on a size (the two hiddens of the
latent MoE, ``PersistentMxfp8Plan.routed_hidden_size``, included).  Torch / CUTLASS imports are lazy.

The host path is that of the other CuTe DSL hosts: full validation when the call signature is new, address
checks on every call, one argument builder for the compile and the launch, the torch views and plain scalars
handed to the compiled TVM FFI function on the current stream, no ``record_stream``.  The state (the output
buffers and the readiness counters) is allocated per call; the context buffers share storage only with each
other, so a forward context kept until the backward keeps no call buffer alive.

``shared_fc1_mxfp8`` launches the kernel's shared Gate, Up and SwiGLU tasks alone (its ``shared_fc1_only``
variant) and returns the shared Gate / Up / Hidden of a forward whose context was not kept.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Final

from .mxfp8_contract import validate_geometry
from ._host_common import validate_tensor as _validate_tensor
from ._host_common import workspace_peer_tables
from ._host_common import has_tail as _has_tail, persistent_executor, process_independent
from ._host_common import LaunchPlanCache, SlotTemplate, layouts_match, row_major_view, tensor_layout
from ._host_common import (
    ValidatedCalls,
    current_cuda_context_key as _current_cuda_context_key,
    current_stream_handle,
    normalize_situ_beta,
    tensor_layout_signature,
    typed_key,
)
from ._host_common import current_device_index as _current_device_index
from .persistent_bf16_fwd import (
    SHARED_PLACEHOLDER_SHAPE,
    routed_expert_count,
    routed_launch_tensors,
    select_task_decode,
    shared_placeholder,
    shared_presence,
    validate_up_row_offset as _validate_up_row_offset,
)
from .persistent_bf16_fwd_contract import TaskDecode
from ._arch import COMPILE_OPTIONS as _COMPILE_OPTIONS, TARGET_ARCH, TARGET_CAPABILITY, check_device
from ._version_gate import require_cutedsl

FC1_LOGICAL_N: Final = 256

# Defaults only, never compiled model constants.
NUM_LOCAL_EXPERTS: Final = 64
HIDDEN_SIZE: Final = 4096
INTERMEDIATE_SIZE: Final = 1024
TOPK: Final = 10
MACROBATCH_ROWS: Final = 32768
MINIBATCH_ROWS: Final = 4096
NUM_COMM_SMS: Final = 42
CLC_DEPTH: Final = 1
ROW_ALIGNMENT: Final = 256
HIDDEN_ROW_BLOCK: Final = 256
Y_DONE_ROW_BLOCK: Final = 128


_CODEGEN_EPOCH: Final = (
    "mxfp8-fwd-ab6-acc1-sf6-tmem-slots-dsl48" "-v5-comm-stream-32x256-warps7-buffers2-groups4-warpquant-swiglu-split-quant-fusedfc1-bf16gateup"
)
_ABI_EPOCH: Final = "persistent-mxfp8-eleven-output-five-counter-stream-head-bf16-gate-up-v5"
# gate_up_context="mxfp8": the e4m3 / scale Gate / Up context (the thirteen-output kernel).
_CODEGEN_EPOCH_E4M3_CONTEXT: Final = (
    "mxfp8-fwd-ab6-acc1-sf6-tmem-slots-dsl48" "-v5-comm-stream-32x256-warps7-buffers2-groups4-warpquant-swiglu-split-quant-fusedfc1"
)
_ABI_EPOCH_E4M3_CONTEXT: Final = "persistent-mxfp8-thirteen-output-five-counter-stream-head-v4"
# The routed Gate / Up context precision (CuteMoKConfig.gate_up_context): "mxfp8" (default) keeps
# e4m3 / scale pairs; "bf16" keeps the epilogue's BF16 values.
GATE_UP_CONTEXTS: Final = ("mxfp8", "bf16")
# Which parts of the forward run in MXFP8.  "routed": the routed experts (the shared expert and
# the communication payload stay BF16).  A full MXFP8 path would be another mode of the same
# entry; it is not implemented.
MXFP8_PATHS: Final = ("routed",)

_EXECUTOR_CACHE: dict[tuple[object, ...], object] = {}
_EXECUTOR_CACHE_DEVICES: set[int] = set()
_EXECUTOR_CACHE_LOCK = threading.Lock()
# Host-only memos: the state's cut per geometry (emptied at _MEMO_LIMIT entries) and, as in the
# other CuTe DSL hosts, the call signatures whose full validation passed (layouts, devices, plan
# and peer addresses -- no tensor; the local address checks repeat on every call).
_STATE_LAYOUTS: dict[tuple[object, ...], tuple[object, ...]] = {}
_VALIDATED_CALLS = ValidatedCalls()


def _routed_view(tensor):
    """``routed_weight_view``'s derivation (the BF16 builder's routed operand view), for SlotTemplate."""
    view = row_major_view(tensor)
    return view if view.ndim == 2 else view.permute(1, 2, 0)


def _permute_10(tensor):
    return tensor.permute(1, 0) if tensor.ndim == 2 else None


def _permute_210(tensor):
    return tensor.permute(2, 1, 0) if tensor.ndim == 3 else None


#: The views the argument builders derive from a per-call tensor (SlotTemplate transforms).
_TRANSFORMS = (row_major_view, _routed_view, _permute_10, _permute_210)
#: Launch plans of forward_mxfp8's repeated calls (see _MxFwdPlan).
_FAST_PLANS = LaunchPlanCache()


@dataclass(frozen=True)
class _MxFwdPlan:
    """What forward_mxfp8's full path validated and built for one set of workspace, weights and geometry:
    a later call with the same ones checks its schedule's layouts, allocates its state and fills the
    template's per-call slots (SlotTemplate) -- no validation, signature or argument building."""

    executor: object
    template: object
    state_layout: tuple
    schedule_layouts: tuple
    routed: object


_MEMO_LIMIT: Final = 64


@dataclass(frozen=True)
class PersistentMxfp8BasePlan:
    """Per-call host geometry, never captured in the executor cache."""

    arch: str = TARGET_ARCH
    ep_size: int = 8  # runtime peer count from the workspace, not a device constant
    num_local_experts: int = NUM_LOCAL_EXPERTS
    hidden_size: int = HIDDEN_SIZE
    intermediate_size: int = INTERMEDIATE_SIZE
    topk: int = TOPK
    macrobatch_size: int = MACROBATCH_ROWS
    minibatch_size: int = MINIBATCH_ROWS
    num_comm_sms: int = NUM_COMM_SMS
    clc_depth: int = CLC_DEPTH
    # 256 by default; the MXFP8 forward, which validates through this plan, passes 128 for both: a
    # 128-wide intermediate and hidden tail.
    intermediate_alignment: int = 256
    hidden_alignment: int = 256

    def validate(self) -> None:
        if type(self.ep_size) is not int or not 0 < self.ep_size < 2**31:
            raise ValueError(f"ep_size must be a positive Int32; got {self.ep_size!r}")
        if self.arch != TARGET_ARCH or self.clc_depth != CLC_DEPTH:
            raise NotImplementedError(f"the MXFP8 forward requires {TARGET_ARCH} CLC1")
        validate_geometry(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_local_experts=self.num_local_experts,
            topk=self.topk,
            macrobatch_size=self.macrobatch_size,
            minibatch_size=self.minibatch_size,
            num_comm_sms=self.num_comm_sms,
            intermediate_alignment=self.intermediate_alignment,
            hidden_alignment=self.hidden_alignment,
        )


@dataclass(frozen=True)
class PersistentMxfp8Plan(PersistentMxfp8BasePlan):
    """The base plan plus the routed chain's hidden.

    Latent MoE (``functional.forward_latent_layer``, ``workspace.routed_hidden_size != hidden_size``): the
    shared expert works in ``hidden_size`` (``x_shared``, ``y_shared``), the routed experts and the
    communication rows (``x_routed`` and its quantized layouts, the combine rows, ``y_routed``) in
    ``routed_hidden``.  ``None`` is the ordinary MoE (one hidden), as in the BF16 forward's plan.
    """

    routed_hidden_size: int | None = None

    @property
    def routed_hidden(self) -> int:
        return self.hidden_size if self.routed_hidden_size is None else self.routed_hidden_size

    def validate(self) -> None:
        super().validate()
        # A latent layer's routed hidden (one that differs from the shared hidden) keeps the 256 alignment; an
        # ordinary layer's is its hidden size, which super().validate() admits with a 128-wide tail.
        if (
            self.routed_hidden_size is not None
            and self.routed_hidden_size != self.hidden_size
            and (type(self.routed_hidden_size) is not int or not 0 < self.routed_hidden_size < 2**31 or self.routed_hidden_size % 256)
        ):
            raise ValueError("routed_hidden_size must be a positive Int32 divisible by 256")


# The default plan of the two public entries that take one (frozen, shared, never mutated).
_DEFAULT_PLAN: Final = PersistentMxfp8Plan()


@dataclass(frozen=True)
class PersistentMxfp8CounterLengths:
    gate_up_tile_ready: int
    hidden_row_block_ready: int
    x_routed_ready: int
    y_routed_ready: int
    y_routed_done: int


def _normalize_swiglu_limit(swiglu_limit: float | None) -> tuple[float, bool]:
    """The optional clamp as ``(limit, is_clamped)``: ``None`` is no clamp, otherwise a finite non-negative
    number."""

    if swiglu_limit is None:
        return 0.0, False
    if type(swiglu_limit) not in (int, float) or not math.isfinite(swiglu_limit) or swiglu_limit < 0:
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


def _validate_num_local_tokens(num_local_tokens: int, *, plan: PersistentMxfp8BasePlan) -> None:
    """The plan and the local token count the forward admits."""

    plan.validate()
    if type(num_local_tokens) is not int or num_local_tokens < 512 or num_local_tokens % ROW_ALIGNMENT:
        raise ValueError("num_local_tokens must be at least 512 and divisible by 256")


def persistent_mxfp8_counter_lengths(
    num_local_tokens: int,
    schedule_capacity: int,
    *,
    plan: PersistentMxfp8BasePlan = PersistentMxfp8BasePlan(),
) -> PersistentMxfp8CounterLengths:
    """The lengths of the five readiness counter arrays for ``num_local_tokens`` shared rows and
    ``schedule_capacity`` routed rows (``x_routed_ready`` without the stream's head word)."""

    _validate_num_local_tokens(num_local_tokens, plan=plan)
    if type(schedule_capacity) is not int or schedule_capacity < ROW_ALIGNMENT or schedule_capacity % ROW_ALIGNMENT:
        raise ValueError("schedule_capacity must be a positive multiple of 256")
    shared_row_blocks = num_local_tokens // HIDDEN_ROW_BLOCK
    routed_row_blocks = schedule_capacity // HIDDEN_ROW_BLOCK
    # Ceil: the MXFP8 forward's 256-wide shared Gate / Up tiles count an I % 256 != 0 tail as a tile.
    gate_up_column_blocks = -(-plan.intermediate_size // FC1_LOGICAL_N)
    return PersistentMxfp8CounterLengths(
        gate_up_tile_ready=(shared_row_blocks + routed_row_blocks) * gate_up_column_blocks,
        hidden_row_block_ready=shared_row_blocks + routed_row_blocks,
        x_routed_ready=(schedule_capacity + plan.minibatch_size - 1) // plan.minibatch_size,
        y_routed_ready=(schedule_capacity + plan.minibatch_size - 1) // plan.minibatch_size,
        y_routed_done=schedule_capacity // Y_DONE_ROW_BLOCK,
    )


# The buffers the forward context keeps until the backward; the others live for one call.
_CONTEXT_BUFFERS: Final = frozenset(
    (
        "x_routed_t",
        "x_routed_sc_t",
        "gate_shared",
        "gate_routed_bf16",
        "up_shared",
        "up_routed_bf16",
        "hidden_shared",
        "hidden_routed_t",
        "hidden_routed_sc_t",
    )
)
_CONTEXT_BUFFERS_E4M3_CONTEXT: Final = frozenset(
    (
        "x_routed_t",
        "x_routed_sc_t",
        "gate_shared",
        "gate_routed_fp8",
        "gate_routed_sc",
        "up_shared",
        "up_routed_fp8",
        "up_routed_sc",
        "hidden_shared",
        "hidden_routed_t",
        "hidden_routed_sc_t",
    )
)


def dynamic_layout_tensor(tensor):
    """The CuTe view ``cute.compile`` traces for a torch tensor: sizes and strides runtime, the
    leading (unit-stride) dimension the last one with stride 1, so a transposed view keeps its
    own layout.  One rule for the forward, the backward and the quantizer."""
    from cutlass.cute.runtime import from_dlpack

    leading = next(dim for dim in reversed(range(tensor.ndim)) if tensor.stride(dim) == 1)
    return from_dlpack(tensor, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=leading)


# x_routed_ready carries one extra element after the minibatch counters: the communication task stream's head word
# (the next block number, taken by atomic add), zeroed with the counters.
STREAM_HEAD_WORDS = 1


def _packed_offsets(lengths: tuple[int, ...]) -> tuple[list[int], int]:
    """Where ``packed_int32_views`` puts each array (every one on a 16-byte boundary), and the total length."""
    offsets, total = [], 0
    for length in lengths:
        offsets.append(total)
        total += -(-length // 4) * 4
    return offsets, max(total, 4)


def packed_int32_views(lengths: tuple[int, ...], device, storage=None) -> tuple[object, tuple[object, ...]]:
    """One zero-filled int32 allocation split into 16-byte-aligned views, one per length.

    Zeroing the storage once replaces one memset launch per readiness array.  ``storage``: the caller's int32
    counter buffer (the execute step of a plan), at least as long as these arrays; its leading words are zeroed
    on the current stream instead of allocating.
    """

    import torch

    offsets, total = _packed_offsets(lengths)
    if storage is None:
        storage = torch.zeros((total,), dtype=torch.int32, device=device)
    elif (
        type(storage) is not torch.Tensor
        or storage.dtype != torch.int32
        or storage.device != device
        or storage.ndim != 1
        or storage.numel() < total
        or storage.data_ptr() % 16
    ):
        raise ValueError(f"counters must be a 16-byte aligned int32 [>= {total}] tensor on {device}")
    else:
        storage[:total].zero_()
    # as_strided rather than narrow: the same views at a lower host cost.
    return storage, tuple(storage.as_strided((length,), (1,), offset) for offset, length in zip(offsets, lengths, strict=True))


def leading_view(storage, shape, dtype, device, name: str):
    """A contiguous ``shape`` view of the leading elements of ``storage``: one of a plan's flat buffers (the
    execute step of a plan), sized for its largest token count and macrobatch."""
    import torch

    numel = math.prod(shape)
    if (
        type(storage) is not torch.Tensor
        or storage.dtype != dtype
        or storage.device != device
        or storage.ndim != 1
        or storage.numel() < numel
        or storage.data_ptr() % 16
    ):
        raise ValueError(
            f"{name} must be a 16-byte aligned {dtype} [>= {numel}] tensor on {device}: the plan's "
            "buffers are sized for its max_tokens_per_rank and macrobatch_size"
        )
    return storage[:numel].view(shape)


@dataclass(frozen=True)
class PersistentMxfp8State:
    """``gate_up_context="bf16"``: the MXFP8 forward's buffers and the five counters, the routed Gate / Up
    context in BF16.

    ``x_routed`` / ``hidden_routed`` are the e4m3 GEMM operands (normal layout, never
    returned).  Scales are uint8 ``(rows / 128, cols / 128, 32, 16)`` tiles.  There is no BF16
    ``gate_routed`` / ``up_routed`` ring: the routed FC1 epilogue computes SwiGLU from its own
    BF16-rounded accumulators and writes Hidden quantized, so the routed half of
    ``gate_up_tile_ready`` is never used (the shared expert still counts on it).

    Lifetimes: the buffers are views of one allocation per lifetime and dtype
    (``prepare_persistent_mxfp8_state``).  The nine
    context buffers (``_CONTEXT_BUFFERS``) share storages only with each other, so keeping the
    forward context alive keeps the context only.  ``y_shared`` is an allocation of its own: a
    gated forward context keeps it until the backward (the shared output gate's dG needs it), and
    sharing a storage with the call's macrobatch-sized ``y_routed`` would keep that alive too --
    256 MiB per layer at H 4096 and a 32K macrobatch.
    """

    x_routed: object
    x_routed_sc: object
    x_routed_t: object
    x_routed_sc_t: object
    gate_shared: object
    gate_routed_bf16: object
    up_shared: object
    up_routed_bf16: object
    hidden_shared: object
    hidden_routed: object
    hidden_routed_sc: object
    hidden_routed_t: object
    hidden_routed_sc_t: object
    y_shared: object
    y_routed: object
    gate_up_tile_ready: object
    hidden_row_block_ready: object
    x_routed_ready: object
    y_routed_ready: object
    y_routed_done: object

    @property
    def abi_outputs(self) -> tuple[object, ...]:
        """The eleven outputs in return order; the routed Gate / Up context is one BF16 tensor each."""
        return (
            self.x_routed_t,
            self.x_routed_sc_t,
            self.gate_shared,
            self.gate_routed_bf16,
            self.up_shared,
            self.up_routed_bf16,
            self.hidden_shared,
            self.hidden_routed_t,
            self.hidden_routed_sc_t,
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


@dataclass(frozen=True)
class PersistentMxfp8StateE4m3Context:
    """``gate_up_context="mxfp8"``: the MXFP8 forward's buffers and the five counters, the routed Gate / Up
    context as e4m3 / scale pairs (``PersistentMxfp8State`` keeps it in BF16).

    ``x_routed`` / ``hidden_routed`` are the e4m3 GEMM operands (normal layout, never
    returned).  Scales are uint8 ``(rows / 128, cols / 128, 32, 16)`` tiles.  There is no BF16
    ``gate_routed`` / ``up_routed`` ring: the routed FC1 epilogue computes SwiGLU from its own
    BF16-rounded accumulators and writes Hidden quantized, so the routed half of
    ``gate_up_tile_ready`` is never used (the shared expert still counts on it).

    Lifetimes: the buffers are views of one allocation per lifetime and dtype
    (``prepare_persistent_mxfp8_state``).  The eleven
    context buffers (``_CONTEXT_BUFFERS_E4M3_CONTEXT``) share storages only with each other, so keeping the
    forward context alive keeps the context only; the returned ``y_shared`` / ``y_routed`` share
    a BF16 storage with each other only.
    """

    x_routed: object
    x_routed_sc: object
    x_routed_t: object
    x_routed_sc_t: object
    gate_shared: object
    gate_routed_fp8: object
    gate_routed_sc: object
    up_shared: object
    up_routed_fp8: object
    up_routed_sc: object
    hidden_shared: object
    hidden_routed: object
    hidden_routed_sc: object
    hidden_routed_t: object
    hidden_routed_sc_t: object
    y_shared: object
    y_routed: object
    gate_up_tile_ready: object
    hidden_row_block_ready: object
    x_routed_ready: object
    y_routed_ready: object
    y_routed_done: object

    @property
    def abi_outputs(self) -> tuple[object, ...]:
        """The thirteen outputs, in return order."""
        return (
            self.x_routed_t,
            self.x_routed_sc_t,
            self.gate_shared,
            self.gate_routed_fp8,
            self.gate_routed_sc,
            self.up_shared,
            self.up_routed_fp8,
            self.up_routed_sc,
            self.hidden_shared,
            self.hidden_routed_t,
            self.hidden_routed_sc_t,
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


def _lifetime(name: str, context_buffers=_CONTEXT_BUFFERS) -> str:
    """How long a state buffer lives: the forward context's until the backward; ``y_shared`` until
    the backward when the shared output gate keeps it (otherwise one call); the rest one call."""
    if name in context_buffers:
        return "context"
    return "shared_output" if name == "y_shared" else "call"


#: The shared expert's four BF16 buffers; a routed-only layer has none.
SHARED_BUFFERS: Final = frozenset(("gate_shared", "up_shared", "hidden_shared", "y_shared"))
#: Their positions among the thirteen outputs.
SHARED_OUTPUT_INDICES: Final = (2, 5, 8, 11)
#: The same among the eleven outputs of ``gate_up_context="bf16"`` (one BF16 tensor per routed Gate / Up).
SHARED_OUTPUT_INDICES_BF16_CONTEXT: Final = (2, 4, 6, 9)
#: The state's buffers that are not returned: the e4m3 GEMM operands x and Hidden and their scales.  Every
#: other buffer is an output, in the state's field order (``abi_outputs``).
FORWARD_SCRATCH: Final = ("x_routed", "x_routed_sc", "hidden_routed", "hidden_routed_sc")


def persistent_mxfp8_buffer_specs(
    num_local_tokens: int, *, plan: PersistentMxfp8Plan | PersistentMxfp8BasePlan, gate_up_context: str = "mxfp8", has_shared: bool = True
) -> tuple:
    """``(name, shape, kind)`` of each allocation, kind in bf16 / e4m3 / scale, in the state class's field
    order (``PersistentMxfp8State``, or ``PersistentMxfp8StateE4m3Context`` for ``gate_up_context="mxfp8"``).
    The routed buffers (``x_routed`` and its layouts, ``y_routed``) are in the routed hidden
    (``PersistentMxfp8Plan.routed_hidden``; a base plan has one hidden), ``y_shared`` in the shared one.
    A routed-only layer's shared buffers are the device's placeholder tile instead (never written)."""

    _validate_num_local_tokens(num_local_tokens, plan=plan)
    t, h, i, m = num_local_tokens, plan.hidden_size, plan.intermediate_size, plan.macrobatch_size
    hr = getattr(plan, "routed_hidden", h)
    t_i, t_h = ((t, i), (t, h)) if has_shared else (SHARED_PLACEHOLDER_SHAPE,) * 2
    return (
        ("x_routed", (m, hr), "e4m3"),
        ("x_routed_sc", (m // 128, hr // 128, 32, 16), "scale"),
        ("x_routed_t", (hr, m), "e4m3"),
        ("x_routed_sc_t", (hr // 128, m // 128, 32, 16), "scale"),
        ("gate_shared", t_i, "bf16"),
        *(
            (("gate_routed_fp8", (m, i), "e4m3"), ("gate_routed_sc", (m // 128, i // 128, 32, 16), "scale"))
            if gate_up_context == "mxfp8"
            else (("gate_routed_bf16", (m, i), "bf16"),)
        ),
        ("up_shared", t_i, "bf16"),
        *(
            (("up_routed_fp8", (m, i), "e4m3"), ("up_routed_sc", (m // 128, i // 128, 32, 16), "scale"))
            if gate_up_context == "mxfp8"
            else (("up_routed_bf16", (m, i), "bf16"),)
        ),
        ("hidden_shared", t_i, "bf16"),
        ("hidden_routed", (m, i), "e4m3"),
        ("hidden_routed_sc", (m // 128, i // 128, 32, 16), "scale"),
        ("hidden_routed_t", (i, m), "e4m3"),
        ("hidden_routed_sc_t", (i // 128, m // 128, 32, 16), "scale"),
        ("y_shared", t_h, "bf16"),
        ("y_routed", (m, hr), "bf16"),
    )


def _dtype(kind: str):
    import torch

    return {"bf16": torch.bfloat16, "e4m3": torch.float8_e4m3fn, "scale": torch.uint8}[kind]


def _sizes_plan(
    *,
    hidden_size: int,
    routed_hidden_size: int | None,
    intermediate_size: int,
    topk: int,
    macrobatch_size: int = ROW_ALIGNMENT,
    minibatch_size: int = ROW_ALIGNMENT,
) -> PersistentMxfp8Plan:
    """A plan of one layer's sizes, for the buffer and counter sizes of a plan step."""
    return PersistentMxfp8Plan(
        num_local_experts=1,
        hidden_size=hidden_size,
        routed_hidden_size=routed_hidden_size,
        intermediate_size=intermediate_size,
        topk=topk,
        macrobatch_size=macrobatch_size,
        minibatch_size=minibatch_size,
        num_comm_sms=2,
        intermediate_alignment=128,
        hidden_alignment=128,
    )


def forward_mxfp8_buffer_specs(
    num_local_tokens: int,
    *,
    hidden_size: int,
    routed_hidden_size: int | None,
    intermediate_size: int,
    topk: int,
    macrobatch_size: int,
    gate_up_context: str = "mxfp8",
    has_shared: bool = True,
) -> tuple[tuple, tuple]:
    """``(outputs, scratch)``: the ``(shape, dtype)`` of each buffer a forward returns, in return order (None where
    a routed-only layer has no shared output), and of its four scratch buffers (``FORWARD_SCRATCH`` order) --
    the execute step of a plan takes both from the caller."""
    specs = persistent_mxfp8_buffer_specs(
        num_local_tokens,
        gate_up_context=gate_up_context,
        has_shared=has_shared,
        plan=_sizes_plan(
            hidden_size=hidden_size, routed_hidden_size=routed_hidden_size, intermediate_size=intermediate_size, topk=topk, macrobatch_size=macrobatch_size
        ),
    )
    outputs = tuple(None if not has_shared and name in SHARED_BUFFERS else (shape, _dtype(kind)) for name, shape, kind in specs if name not in FORWARD_SCRATCH)
    return outputs, tuple((shape, _dtype(kind)) for name, shape, kind in specs if name in FORWARD_SCRATCH)


def forward_mxfp8_counter_words(
    num_local_tokens: int,
    schedule_capacity: int,
    *,
    hidden_size: int,
    routed_hidden_size: int | None,
    intermediate_size: int,
    topk: int,
    minibatch_size: int = ROW_ALIGNMENT,
) -> int:
    """How many int32 words the five counters of a forward over these sizes take: the length of an execute
    call's counter buffer, as a call that allocates its own counters does.  They grow as the minibatch shrinks;
    the default, the smallest one (256 rows), gives a length that serves every minibatch.  The macrobatch and
    the communication SMs do not change it."""
    lengths = persistent_mxfp8_counter_lengths(
        num_local_tokens,
        schedule_capacity,
        plan=_sizes_plan(
            hidden_size=hidden_size,
            routed_hidden_size=routed_hidden_size,
            intermediate_size=intermediate_size,
            topk=topk,
            macrobatch_size=minibatch_size,
            minibatch_size=minibatch_size,
        ),
    )
    return _packed_offsets(
        (lengths.gate_up_tile_ready, lengths.hidden_row_block_ready, lengths.x_routed_ready + STREAM_HEAD_WORDS, lengths.y_routed_ready, lengths.y_routed_done)
    )[1]


def prepare_persistent_mxfp8_state(
    *,
    num_local_tokens: int,
    schedule_capacity: int,
    device,
    plan: PersistentMxfp8Plan | PersistentMxfp8BasePlan = _DEFAULT_PLAN,
    gate_up_context: str = "mxfp8",
    has_shared: bool = True,
) -> PersistentMxfp8State | PersistentMxfp8StateE4m3Context:
    """Allocate the MXFP8 forward's buffers and counters without launching anything.

    The buffers are split from one allocation per lifetime and dtype (every buffer is a
    multiple of 256 bytes: its trailing extents are multiples of 128): the context buffers
    share storages only with each other, so the forward context does not keep a call buffer
    alive into the backward.  The five counters are views of one zeroed allocation
    (``packed_int32_views``).  ``gate_up_context="bf16"`` gives a ``PersistentMxfp8State``
    (fifteen buffers, nine of them context), ``"mxfp8"`` a ``PersistentMxfp8StateE4m3Context``
    (seventeen buffers, eleven of them context).
    """

    return _allocate_state(_state_layout(num_local_tokens, schedule_capacity, plan, gate_up_context, has_shared), device, has_shared, gate_up_context)


def _allocate_state(layout, device, has_shared: bool, gate_up_context: str = "mxfp8", caller_buffers=None):
    """The state cut as ``layout`` (_state_layout) says: one allocation per group, one zeroed counter block;
    or, with ``caller_buffers``, the caller's buffers (``_caller_state``)."""
    import torch

    if caller_buffers is not None:
        return _caller_state(layout, device, has_shared, gate_up_context, caller_buffers)
    groups, counter_lengths, buffer_count = layout
    # A routed-only layer's shared buffers are left out of the groups: the placeholder stands in.
    outputs: list[object] = [None if has_shared else shared_placeholder(device)] * buffer_count
    for dtype, total, views in groups:
        storage = torch.empty((total,), dtype=dtype, device=device)
        for index, shape, stride, offset in views:
            outputs[index] = storage.as_strided(shape, stride, offset)
    _, counters = packed_int32_views(counter_lengths, device)
    state_class = PersistentMxfp8StateE4m3Context if gate_up_context == "mxfp8" else PersistentMxfp8State
    return state_class(*outputs, *counters)


def _caller_state(layout, device, has_shared: bool, gate_up_context: str, caller_buffers):
    """The state of a call in the execute step of a plan, which allocates nothing.  ``caller_buffers`` =
    ``(outputs, scratch, counters)``: the caller's buffers of the returned tensors in return order (None where a
    routed-only layer has no shared output; ``forward_mxfp8_buffer_specs`` gives their shapes and dtypes), checked
    against the layout; the caller's four flat scratch buffers (``FORWARD_SCRATCH`` order), of which the leading
    elements are used; and the caller's int32 counter buffer (``forward_mxfp8_counter_words``), whose leading
    words are zeroed on the current stream."""
    groups, counter_lengths, buffer_count = layout
    state_class = PersistentMxfp8StateE4m3Context if gate_up_context == "mxfp8" else PersistentMxfp8State
    names = _STATE_FIELD_NAMES[state_class][:buffer_count]
    specs = {index: (shape, dtype) for dtype, _, views in groups for index, shape, _, _ in views}
    outputs, scratch, counters = caller_buffers
    output_names = tuple(name for name in names if name not in FORWARD_SCRATCH)
    outputs, scratch = tuple(outputs), tuple(scratch)
    if len(outputs) != len(output_names) or len(scratch) != len(FORWARD_SCRATCH):
        raise ValueError(
            f"the MXFP8 forward takes {len(output_names)} output buffers and " f"{len(FORWARD_SCRATCH)} scratch buffers; got {len(outputs)} and {len(scratch)}"
        )
    supplied = {**dict(zip(output_names, outputs)), **dict(zip(FORWARD_SCRATCH, scratch))}
    buffers = []
    for index, name in enumerate(names):
        if index not in specs:  # a routed-only layer's shared buffer
            buffers.append(shared_placeholder(device))
            continue
        shape, dtype = specs[index]
        if name in FORWARD_SCRATCH:
            buffers.append(leading_view(supplied[name], shape, dtype, device, f"the forward scratch {name}"))
        else:
            _validate_tensor(f"outputs[{output_names.index(name)}] ({name})", supplied[name], shape=shape, dtype=dtype, device=device)
            buffers.append(supplied[name])
    _, counters = packed_int32_views(counter_lengths, device, storage=counters)
    return state_class(*buffers, *counters)


def _state_layout(
    num_local_tokens: int, schedule_capacity: int, plan: PersistentMxfp8Plan | PersistentMxfp8BasePlan, gate_up_context: str = "mxfp8", has_shared: bool = True
):
    """How a state is cut: per lifetime and dtype, the allocation's element count and each
    buffer's position, shape, contiguous strides and offset; the counter lengths; computed
    once per geometry (memoized).  A routed-only layer allocates no shared buffer."""

    # 1024 and 1024.0 are different keys
    key = typed_key(num_local_tokens, schedule_capacity, has_shared) + (plan, gate_up_context)
    layout = _STATE_LAYOUTS.get(key)
    if layout is not None:
        return layout
    specs = persistent_mxfp8_buffer_specs(num_local_tokens, plan=plan, gate_up_context=gate_up_context, has_shared=has_shared)
    context_buffers = _CONTEXT_BUFFERS_E4M3_CONTEXT if gate_up_context == "mxfp8" else _CONTEXT_BUFFERS
    by_group: dict[tuple[bool, str], list[int]] = {}
    for index, (name, _, kind) in enumerate(specs):
        if not has_shared and name in SHARED_BUFFERS:
            continue
        by_group.setdefault((_lifetime(name, context_buffers), kind), []).append(index)
    groups = []
    for (_, kind), indices in by_group.items():
        dtype = _dtype(kind)
        views, offset = [], 0
        for index in indices:
            shape = specs[index][1]
            stride, count = [], 1
            for extent in reversed(shape):
                stride.append(count)
                count *= extent
            if (count * dtype.itemsize) % 256:
                raise ValueError(f"state buffer {specs[index][0]} is not a multiple of 256 bytes")
            views.append((index, shape, tuple(reversed(stride)), offset))
            offset += count
        groups.append((dtype, offset, tuple(views)))
    lengths = persistent_mxfp8_counter_lengths(num_local_tokens, schedule_capacity, plan=plan)
    layout = (
        tuple(groups),
        (lengths.gate_up_tile_ready, lengths.hidden_row_block_ready, lengths.x_routed_ready + STREAM_HEAD_WORDS, lengths.y_routed_ready, lengths.y_routed_done),
        len(specs),
    )
    if len(_STATE_LAYOUTS) >= _MEMO_LIMIT:
        _STATE_LAYOUTS.clear()
    _STATE_LAYOUTS[key] = layout
    return layout


def validate_persistent_mxfp8_call(
    workspace,
    schedule,
    shared_gate_weights,
    routed_gate_weights,
    shared_up_weights,
    routed_up_weights,
    shared_down_weights,
    routed_down_weights,
    routed_weight_scales,
    state: PersistentMxfp8State | None = None,
    *,
    plan: PersistentMxfp8Plan | PersistentMxfp8BasePlan = _DEFAULT_PLAN,
    x_shared=None,
    up_row_offset: int = 0,
    routed_storage_tables=None,
    has_shared: bool = True,
) -> None:
    """Fail closed before importing CUTLASS: the workspace / schedule checks,
    e4m3 routed weights with their scale tiles, and the MXFP8 state when one is given
    (``forward_mxfp8`` allocates its own from the same specs and passes none).  ``x_shared`` is the
    shared expert's input ``[num_local_tokens, hidden_size]``, required when the routed hidden differs
    from the shared one (the routed weights, the communication buffers and ``routed_weight_scales`` are
    then in the routed hidden).  A base plan (one hidden) is accepted wherever a
    ``PersistentMxfp8Plan`` is.

    ``up_row_offset`` is ``intermediate_size`` when Gate and Up are one combined FC1 -- an
    ``[E, 2I, H]`` parameter, or ``[2I, H]`` per split expert, with its whole scale -- passed as
    both arguments, and 0 when they are separate ``I``-row weights.

    ``routed_storage_tables`` is the six per-expert address tables -- the Gate, Up and Down
    payloads, then their three scales -- when the routed experts are independent
    allocations.  Every routed operand is then one expert's own allocation rather than an
    ``[E, ...]`` parameter, and what has to be checked is that the tables agree with each other
    and with those representatives, because the kernel keeps one descriptor schema per operand and
    only retargets its address."""

    import torch

    plan.validate()
    routed_hidden = getattr(plan, "routed_hidden", plan.hidden_size)
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
        or getattr(workspace, "routed_hidden_size", workspace.hidden_size) != routed_hidden
        or workspace.topk != plan.topk
    ):
        raise NotImplementedError("workspace geometry (ep_size/hidden/routed hidden/topk) must match the plan")
    lengths = persistent_mxfp8_counter_lengths(workspace.num_local_tokens, workspace.schedule_capacity, plan=plan)
    check_device(device, "the MXFP8 forward")
    for name in ("x_buffer_ptrs", "combine_buffer_ptrs"):
        pointers = getattr(workspace, name)
        if (
            not isinstance(pointers, list)
            or len(pointers) != plan.ep_size
            or any(type(pointer) is not int or pointer <= 0 or pointer % 16 for pointer in pointers)
        ):
            raise ValueError(f"workspace.{name} must contain ep_size={plan.ep_size} aligned addresses")
    _validate_tensor("workspace.x_buffer", workspace.x_buffer, shape=(workspace.num_local_tokens, routed_hidden), dtype=torch.bfloat16, device=device)
    _validate_tensor(
        "workspace.combine_buffer", workspace.combine_buffer, shape=(workspace.num_local_tokens * plan.topk, routed_hidden), dtype=torch.bfloat16, device=device
    )
    if x_shared is None:
        if routed_hidden != plan.hidden_size:
            raise ValueError("x_shared is required when the routed hidden differs from the shared hidden")
    else:
        _validate_tensor("x_shared", x_shared, shape=(workspace.num_local_tokens, plan.hidden_size), dtype=torch.bfloat16, device=device)
    if not 0 <= workspace.ep_rank < plan.ep_size:
        raise ValueError(f"workspace.ep_rank is outside [0, ep_size={plan.ep_size})")
    if workspace.x_buffer_ptrs[workspace.ep_rank] != workspace.x_buffer.data_ptr():
        raise ValueError("workspace.x_buffer_ptrs does not own the local x buffer")
    if workspace.combine_buffer_ptrs[workspace.ep_rank] != workspace.combine_buffer.data_ptr():
        raise ValueError("workspace.combine_buffer_ptrs does not own the local combine buffer")

    e, h, hr, i = plan.num_local_experts, plan.hidden_size, routed_hidden, plan.intermediate_size
    split_weights = routed_storage_tables is not None
    if split_weights:
        if len(routed_storage_tables) != 6:
            raise ValueError("routed_storage_tables must hold the Gate, Up and Down payload tables and then " "their three scale tables")
        for name, table in zip(
            ("gate", "up", "down", "gate_sc", "up_sc", "down_sc"),
            routed_storage_tables,
        ):
            _validate_tensor(f"routed_storage_tables.{name}", table, shape=(e,), dtype=torch.int64, device=device)
    combined = _validate_up_row_offset(up_row_offset, i, routed_gate_weights, routed_up_weights, routed_weight_scales[0], routed_weight_scales[1])
    fc1_rows = 2 * i if combined else i
    # Split: one expert's own allocation per routed operand.  Grouped: the whole [E, ...]
    # parameter, whose L mode the kernel slices.
    fc1_shape = (fc1_rows, hr) if split_weights else (e, fc1_rows, hr)
    down_shape = (hr, i) if split_weights else (e, hr, i)
    # A routed-only layer passes the placeholder for each shared weight.
    shared_fc1, shared_down = ((i, h), (h, i)) if has_shared else (SHARED_PLACEHOLDER_SHAPE,) * 2
    for name, tensor, shape, dtype, sliced in (
        ("shared_gate_weights", shared_gate_weights, shared_fc1, torch.bfloat16, False),
        ("routed_gate_weights", routed_gate_weights, fc1_shape, torch.float8_e4m3fn, True),
        ("shared_up_weights", shared_up_weights, shared_fc1, torch.bfloat16, False),
        ("routed_up_weights", routed_up_weights, fc1_shape, torch.float8_e4m3fn, True),
        ("shared_down_weights", shared_down_weights, shared_down, torch.bfloat16, False),
        ("routed_down_weights", routed_down_weights, down_shape, torch.float8_e4m3fn, True),
    ):
        _validate_tensor(name, tensor, shape=shape, dtype=dtype, device=device, sliced=sliced)
    # A scale covers its payload's rows, expert-major over 128-row blocks; a split scale is one
    # expert's own.  The kernel takes the per-expert stride from this extent (``scale_operand``).
    scale_experts = 1 if split_weights else e
    for name, scale in zip(("routed_gate_weights_sc", "routed_up_weights_sc"), routed_weight_scales[:2], strict=True):
        _validate_tensor(name, scale, shape=(scale_experts * fc1_rows // 128, hr // 128, 32, 16), dtype=torch.uint8, device=device)
    _validate_tensor("routed_down_weights_sc", routed_weight_scales[2], shape=(scale_experts * hr // 128, i // 128, 32, 16), dtype=torch.uint8, device=device)
    for name, tensor, shape in (
        ("schedule.peer_rank", schedule.peer_rank, (workspace.schedule_capacity,)),
        ("schedule.peer_token_idx", schedule.peer_token_idx, (workspace.schedule_capacity,)),
        ("schedule.num_tokens", schedule.num_tokens, (1,)),
        ("schedule.tokens_per_expert", schedule.tokens_per_expert, (e,)),
    ):
        _validate_tensor(name, tensor, shape=shape, dtype=torch.int32, device=device)

    if state is None:
        return
    if not isinstance(state, (PersistentMxfp8State, PersistentMxfp8StateE4m3Context)):
        raise TypeError("state must be a PersistentMxfp8State or PersistentMxfp8StateE4m3Context")
    gate_up_context = "mxfp8" if isinstance(state, PersistentMxfp8StateE4m3Context) else "bf16"
    for name, shape, kind in persistent_mxfp8_buffer_specs(workspace.num_local_tokens, plan=plan, gate_up_context=gate_up_context, has_shared=has_shared):
        _validate_tensor(f"state.{name}", getattr(state, name), shape=shape, dtype=_dtype(kind), device=device)
    for name, tensor, length in zip(
        ("gate_up_tile_ready", "hidden_row_block_ready", "x_routed_ready", "y_routed_ready", "y_routed_done"),
        state.counters,
        (lengths.gate_up_tile_ready, lengths.hidden_row_block_ready, lengths.x_routed_ready + STREAM_HEAD_WORDS, lengths.y_routed_ready, lengths.y_routed_done),
        strict=True,
    ):
        _validate_tensor(f"state.{name}", tensor, shape=(length,), dtype=torch.int32, device=device)


def _normalize_mxfp8_path(mxfp8_path: str) -> str:
    """The ``mxfp8_path`` keyword of ``forward_mxfp8``: one of ``MXFP8_PATHS``."""
    if mxfp8_path not in MXFP8_PATHS:
        raise NotImplementedError(f"mxfp8_path must be one of {MXFP8_PATHS}; got {mxfp8_path!r} (the full MXFP8 path is not implemented)")
    return mxfp8_path


def _executor_key(
    *,
    is_clamped,
    task_decode,
    mxfp8_path,
    runtime_environment,
    device_index,
    context_key,
    gate_up_context="mxfp8",
    split_weights: bool = False,
    is_situ: bool = False,
    has_shared: bool = True,
    shared_fc1_only: bool = False,
    has_tail: bool = False,
) -> tuple[object, ...]:
    """Only what selects code owns an executor: the codegen / ABI epochs, the task-decode variant, the
    Gate / Up context precision, the routed weight layout, the clamp, the activation, the shared expert's
    presence, the shared-FC1-only launch, 128-wide H / I tail tiles, the runtime versions and the CUDA context.
    Sizes, the two hiddens and a combined FC1's Up row offset included, are runtime arguments."""
    epochs = (_CODEGEN_EPOCH_E4M3_CONTEXT, _ABI_EPOCH_E4M3_CONTEXT) if gate_up_context == "mxfp8" else (_CODEGEN_EPOCH, _ABI_EPOCH)
    return (
        "mok-mxfp8-fwd-runtime",
        *epochs,
        ("mxfp8_path", mxfp8_path),
        ("gate_up_context", gate_up_context),
        ("device", device_index),
        ("context", context_key),
        ("arch", TARGET_ARCH, TARGET_CAPABILITY),
        ("runtime", runtime_environment),
        ("peer_pointer_abi", "int64-address-table-runtime-length"),
        ("tensor_abi", "e4m3-e8m0-sf32-routed-bf16-shared-i32-schedule-dynamic-layout"),
        # The routed operands' rank differs between the two layouts, and the flag is what selects
        # the code, so key it directly.
        ("routed_weights", "split" if split_weights else "grouped"),
        ("is_clamped", bool(is_clamped)),
        ("is_situ", bool(is_situ)),
        ("has_shared", bool(has_shared)),
        ("shared_fc1_only", bool(shared_fc1_only)),
        ("has_tail", bool(has_tail)),
        ("task_decode", TaskDecode(task_decode).name),
    )


def clear_persistent_mxfp8_executor_cache(*, synchronize: bool = True) -> None:
    """Release executor-only cache entries after public calls are quiescent."""

    with _EXECUTOR_CACHE_LOCK:
        if synchronize and _EXECUTOR_CACHE_DEVICES:
            import torch

            for device_index in sorted(_EXECUTOR_CACHE_DEVICES):
                torch.cuda.synchronize(device_index)
        _EXECUTOR_CACHE.clear()
        _EXECUTOR_CACHE_DEVICES.clear()
        _STATE_LAYOUTS.clear()
        _VALIDATED_CALLS.clear()
        _FAST_PLANS.clear()
        _SHARED_FC1_PLACEHOLDERS.clear()


def _mega_call_arguments(
    workspace,
    schedule,
    weights: tuple[object, ...],
    state: PersistentMxfp8State | PersistentMxfp8StateE4m3Context,
    *,
    plan: PersistentMxfp8BasePlan,
    swiglu_limit_value: float,
    is_clamped: bool,
    stream,
) -> tuple[tuple[object, ...], dict[str, object]]:
    return (
        (
            workspace.x_buffer_ptrs,
            workspace.combine_buffer_ptrs,
            schedule.peer_rank,
            schedule.peer_token_idx,
            schedule.num_tokens,
            schedule.tokens_per_expert,
            workspace.x_buffer,
            *weights,
            state,
        ),
        {
            "num_local_tokens": workspace.num_local_tokens,
            "peer_tables": workspace_peer_tables(workspace),
            "topk": plan.topk,
            "schedule_capacity": workspace.schedule_capacity,
            "macrobatch_size": plan.macrobatch_size,
            "minibatch_size": plan.minibatch_size,
            "num_comm_sms": plan.num_comm_sms,
            "swiglu_limit": swiglu_limit_value,
            "stream": stream,
        },
    )


def forward_mxfp8(
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
    mxfp8_path: str = "routed",
    gate_up_context: str = "mxfp8",
    up_row_offset: int = 0,
    routed_storage_tables=None,
    situ_beta=None,
    caller_buffers=None,
) -> tuple[object, ...]:
    """Run the MXFP8 forward; routed weights are ``(e4m3, scale)`` pairs.  Returns thirteen outputs with
    the default ``gate_up_context="mxfp8"`` (``PersistentMxfp8StateE4m3Context.abi_outputs``: the Gate / Up
    context as e4m3 / scale pairs), or eleven with ``"bf16"`` (``PersistentMxfp8State.abi_outputs``).

    ``routed_storage_tables`` are the six per-expert address tables -- the Gate, Up and Down
    payloads, then their three scales -- when the routed experts are independent
    allocations; every routed operand is then the representative expert alone and the kernel names
    the others through per-expert TMA descriptors.

    ``up_row_offset``: ``intermediate_size`` when Gate and Up are one combined FC1 passed as both
    arguments (its whole payload and scale), 0 for separate weights.  The kernel reads Up that
    many rows down, payload and scale alike.
    ``mxfp8_path`` selects which parts run in MXFP8 (``MXFP8_PATHS``; only ``"routed"`` exists, and it is
    part of the executor key).  ``x_shared`` is the shared expert's input ``[num_local_tokens, hidden_size]``;
    it defaults to the workspace's x buffer and is required when ``workspace.routed_hidden_size`` differs from
    ``hidden_size`` (the latent MoE: the x buffer, the routed weights, the combine rows and the routed outputs
    are then in the routed hidden; the kernel reads both hiddens from the tensors, one executor serves both).

    ``situ_beta``: SiTU-GLU's ``(beta1, beta2)`` for the shared and routed MLPs instead of SwiGLU;
    a compile-time variant with the betas as runtime arguments.

    The three shared weights all None: a routed-only layer, with no shared tasks or buffers; the
    four shared outputs come back None.

    One host path, as the other CuTe DSL ops: the full input check runs when the call signature
    (plan, layouts, devices, peer addresses) is new and the local address checks on every call;
    the state is allocated per call; the executor (one per compiled variant), the state's cut
    per geometry, the validated signatures and the launch plans of repeated calls are the only
    memos; the launch hands this call's torch views and plain scalars to the compiled TVM FFI
    function on the current stream.  ``functional.forward`` calls this after the entry barrier.

    ``caller_buffers`` (the execute step of a plan): ``(outputs, scratch, counters)`` -- the caller's buffers of
    the returned tensors (``forward_mxfp8_buffer_specs``), the caller's four flat scratch buffers and int32
    counter buffer (``_caller_state``).  The call then allocates nothing and returns the caller's buffers.  By
    default it allocates its state.
    """

    routed = (routed_gate_weights, routed_up_weights, routed_down_weights)
    if any(not isinstance(pair, tuple) or len(pair) != 2 for pair in routed):
        raise TypeError("routed weights must be (float8_e4m3fn, uint8 scale) pairs")
    (routed_gate_fp8, routed_gate_sc), (routed_up_fp8, routed_up_sc), (routed_down_fp8, routed_down_sc) = routed
    tables = None if routed_storage_tables is None else tuple(routed_storage_tables)
    has_shared = shared_presence(shared_gate_weights, shared_up_weights, shared_down_weights)
    if not has_shared:
        shared_gate_weights = shared_up_weights = shared_down_weights = shared_placeholder(workspace.device)
    # The fast path: a repeated workspace, weights and geometry (_MxFwdPlan).
    weights_and_scales = (
        shared_gate_weights,
        routed_gate_fp8,
        shared_up_weights,
        routed_up_fp8,
        shared_down_weights,
        routed_down_fp8,
        routed_gate_sc,
        routed_up_sc,
        routed_down_sc,
    )
    try:
        fast_sources = (workspace.x_buffer, workspace.combine_buffer, *(tables or ()))
        fast_extra = typed_key(
            macrobatch_size,
            minibatch_size,
            num_comm_sms,
            swiglu_limit,
            up_row_offset,
            situ_beta,
            mxfp8_path,
            has_shared,
            workspace.ep_size,
            workspace.ep_rank,
            workspace.num_local_tokens,
            workspace.hidden_size,
            workspace.topk,
            workspace.schedule_capacity,
            workspace.x_buffer_ptrs,
            workspace.combine_buffer_ptrs,
            tables is None,
            gate_up_context,
            getattr(workspace, "routed_hidden_size", None),
        ) + (
            _current_device_index(),
            _current_cuda_context_key(),
            workspace.device,
            # The decode variant joins the key as in the BF16 plans: a plan is one executor's.
            select_task_decode(routed_expert_count(tables, routed_gate_fp8)).name,
        )
        schedule_tensors = _fast_schedule_tensors(schedule)
        # The latent MoE's x_shared is a per-call tensor the launch template does not carry: full path.
        fast = None if x_shared is not None else _FAST_PLANS.get(fast_sources, fast_extra, weights_and_scales)
    except (AttributeError, TypeError, ValueError):
        fast_extra = fast = None  # no plan is stored either (fast_sources is read only with fast_extra)
    if fast is not None and layouts_match(schedule_tensors, fast.schedule_layouts, 16):
        state = _allocate_state(fast.state_layout, workspace.device, has_shared, gate_up_context, caller_buffers)
        fast.executor(
            *fast.template.fill(
                _fast_launch_sources(workspace, weights_and_scales, schedule_tensors, state, fast_extra[-4]), _TRANSFORMS, build_descriptors=fast.routed.build
            )
        )
        fast.routed.mark_built()
        return _forward_outputs(state, has_shared)
    plan = PersistentMxfp8Plan(
        ep_size=getattr(workspace, "ep_size", -1),
        num_local_experts=routed_expert_count(tables, routed_gate_fp8),
        hidden_size=getattr(workspace, "hidden_size", -1),
        routed_hidden_size=getattr(workspace, "routed_hidden_size", None),
        # From the routed FC1 (its rows, less the Up half when combined): a routed-only layer has
        # no shared weights to read it from.
        intermediate_size=getattr(routed_gate_fp8, "shape", (-1, -1))[-2] - up_row_offset,
        topk=getattr(workspace, "topk", -1),
        macrobatch_size=macrobatch_size,
        minibatch_size=minibatch_size,
        num_comm_sms=num_comm_sms,
        intermediate_alignment=128,  # a 128-wide intermediate tail
        hidden_alignment=128,  # a 128-wide hidden tail too
    )
    plan.validate()
    if plan.routed_hidden != plan.hidden_size and x_shared is None:
        raise ValueError("x_shared is required when workspace.routed_hidden_size != hidden_size")
    swiglu_limit_value, is_clamped = _normalize_swiglu_limit(swiglu_limit)
    situ = normalize_situ_beta(situ_beta, swiglu_limit)
    # Task -> expert decode variant of the kernel (the BF16 forward's selector): a compile-time
    # attribute chosen from the local expert count, part of the executor key.
    task_decode = select_task_decode(plan.num_local_experts)
    # The kernel's other compile-time variant this call selects: one more tile for a 128-wide H / I tail.
    has_tail = _has_tail(plan.hidden_size, plan.routed_hidden, plan.intermediate_size)
    mxfp8_path = _normalize_mxfp8_path(mxfp8_path)
    if gate_up_context not in GATE_UP_CONTEXTS:
        raise ValueError(f"gate_up_context must be one of {GATE_UP_CONTEXTS}; got {gate_up_context!r}")

    import torch

    device_index = workspace.device.index if workspace.device.index is not None else _current_device_index()
    if _current_device_index() != device_index:
        raise RuntimeError("the workspace CUDA device must be current")
    runtime_environment = _prepare_public_runtime_environment()
    weights = (shared_gate_weights, routed_gate_fp8, shared_up_weights, routed_up_fp8, shared_down_weights, routed_down_fp8)
    scales = (routed_gate_sc, routed_up_sc, routed_down_sc)
    # The shared expert's input: the caller's x_shared, or the x buffer when the hiddens coincide.
    shared_input = workspace.x_buffer if x_shared is None else x_shared
    inputs = None
    try:
        inputs = (
            shared_input,
            workspace.x_buffer,
            workspace.combine_buffer,
            schedule.peer_rank,
            schedule.peer_token_idx,
            schedule.num_tokens,
            schedule.tokens_per_expert,
            *weights,
            *scales,
        )
        signature = (
            plan,
            workspace.device,
            typed_key(
                workspace.ep_rank,
                workspace.num_local_tokens,
                workspace.schedule_capacity,
                workspace.x_buffer_ptrs,
                workspace.combine_buffer_ptrs,
                up_row_offset,
                routed_up_fp8 is routed_gate_fp8,
                has_shared,
            ),
            tensor_layout_signature(inputs),
        )
        validated = _VALIDATED_CALLS.get(signature)  # hashes the signature
    except (AttributeError, TypeError):  # malformed inputs: the full checks raise the precise error
        signature = validated = None
    if validated is None:
        validate_persistent_mxfp8_call(
            workspace,
            schedule,
            *weights,
            scales,
            plan=plan,
            x_shared=x_shared,
            up_row_offset=up_row_offset,
            routed_storage_tables=tables,
            has_shared=has_shared,
        )
        if signature is not None:
            _VALIDATED_CALLS.put(signature, True)
    elif (
        workspace.x_buffer_ptrs[workspace.ep_rank] != workspace.x_buffer.data_ptr()
        or workspace.combine_buffer_ptrs[workspace.ep_rank] != workspace.combine_buffer.data_ptr()
        or any(tensor.data_ptr() % 16 for tensor in inputs)
    ):
        # Addresses are not part of the signature: the ownership and alignment checks stay per call.
        raise ValueError("the workspace pointer lists must own the local buffers and every input " "must be 16-byte aligned")
    state_layout = _state_layout(workspace.num_local_tokens, workspace.schedule_capacity, plan, gate_up_context, has_shared)
    state = _allocate_state(state_layout, workspace.device, has_shared, gate_up_context, caller_buffers)
    key = _executor_key(
        is_clamped=is_clamped,
        task_decode=task_decode,
        mxfp8_path=mxfp8_path,
        runtime_environment=runtime_environment,
        device_index=device_index,
        context_key=_current_cuda_context_key(),
        gate_up_context=gate_up_context,
        split_weights=tables is not None,
        is_situ=situ is not None,
        has_shared=has_shared,
        has_tail=has_tail,
    )
    # Built once per set of address tables and descriptor schema (the executor's atoms over these
    # routed weights' and scales' layouts).
    routed = routed_launch_tensors(tables, plan.num_local_experts, workspace.device, schema=(key, tensor_layout_signature((*weights[1::2], *scales))))
    positional_args, keyword_args = _launch_arguments(
        workspace,
        schedule,
        weights,
        scales,
        state,
        plan=plan,
        shared_input=shared_input,
        swiglu_limit_value=swiglu_limit_value,
        is_clamped=is_clamped,
        up_row_offset=up_row_offset,
        situ=situ,
        routed=routed,
        stream=current_stream_handle(device_index),
    )
    executor = _executor(
        key,
        device_index,
        positional_args,
        keyword_args,
        is_clamped=is_clamped,
        task_decode=task_decode,
        gate_up_context=gate_up_context,
        split_weights=tables is not None,
        is_situ=situ is not None,
        has_shared=has_shared,
        has_tail=has_tail,
    )

    from ._persistent_mxfp8_fwd_mega import make_mega_args_mxfp8

    runtime_args = make_mega_args_mxfp8(*positional_args, launch=True, **keyword_args)
    # Launch on the current stream (the one prepare_persistent_mxfp8_state zeroed the counters on).
    # The TVM FFI executor is positional; the dict follows __call__'s parameter order (checked in
    # prepare_mega_mxfp8).  No record_stream: like PyTorch's own ops, inputs are used on the current
    # stream, and a caller that frees them from another stream synchronizes that stream itself.
    executor(*runtime_args.values())
    if fast_extra is not None:
        # Remember this call's launch for the next ones with the same workspace, weights and geometry.
        template = SlotTemplate.build(
            runtime_args.values(),
            _fast_launch_sources(workspace, weights_and_scales, schedule_tensors, state, device_index),
            _TRANSFORMS,
            named={"build_descriptors": list(runtime_args).index("build_descriptors")},
        )
        if template is not None and x_shared is None:
            _FAST_PLANS.put(
                fast_sources,
                fast_extra,
                _MxFwdPlan(
                    executor=executor,
                    template=template,
                    state_layout=state_layout,
                    schedule_layouts=tuple(tensor_layout(tensor) for tensor in schedule_tensors),
                    routed=routed,
                ),
                weights_and_scales,
            )
    routed.mark_built()
    return _forward_outputs(state, has_shared)


def _launch_arguments(
    workspace, schedule, weights, scales, state, *, plan, shared_input, swiglu_limit_value, is_clamped, up_row_offset, situ, routed, stream
) -> tuple[tuple, dict]:
    """The megakernel's positional and keyword arguments, for its compile and its launch
    (``make_mega_args_mxfp8``)."""
    positional_args, keyword_args = _mega_call_arguments(
        workspace, schedule, weights, state, plan=plan, swiglu_limit_value=swiglu_limit_value, is_clamped=is_clamped, stream=stream
    )
    # The BF16 builder's shared input (its seventh positional, the x buffer) is the caller's x_shared when given:
    # the kernel reads the shared hidden from its width and the routed hidden from x_routed's.
    if positional_args[6] is not workspace.x_buffer:
        raise RuntimeError("_mega_call_arguments moved its shared-input (x buffer) slot")
    positional_args = (*positional_args[:6], shared_input, *positional_args[7:])
    keyword_args["routed_weight_scales"] = scales
    keyword_args["up_row_offset"] = up_row_offset
    keyword_args["situ_beta"] = (0.0, 0.0) if situ is None else situ
    # make_mega_args_mxfp8 inserts these; the BF16 builder it wraps has no use for them.
    keyword_args["routed_storage_tables"] = routed.addresses
    keyword_args["routed_descriptors"] = routed.descriptors
    keyword_args["build_descriptors"] = routed.build
    return positional_args, keyword_args


def _executor(key, device_index, positional_args, keyword_args, **variant):
    """The executor of ``key``, compiled on first use over these arguments (``prepare_mega_mxfp8``)."""
    from ._persistent_mxfp8_fwd_mega import prepare_mega_mxfp8

    with _EXECUTOR_CACHE_LOCK:
        executor = _EXECUTOR_CACHE.get(key)
        if executor is None:
            executor = persistent_executor(process_independent(key), lambda: prepare_mega_mxfp8(*positional_args, **variant, **keyword_args))
            _EXECUTOR_CACHE[key] = executor
            _EXECUTOR_CACHE_DEVICES.add(device_index)
    return executor


def compile_forward_mxfp8(
    workspace,
    *,
    num_local_experts: int,
    intermediate_size: int,
    is_clamped: bool,
    is_situ: bool = False,
    has_shared: bool = True,
    split_weights: bool = False,
    gate_up_context: str = "mxfp8",
) -> None:
    """The plan step: compile the megakernel variant a layer runs, without launching.

    The variant is what the executor key holds: the SwiGLU clamp, SiTU-GLU, the shared expert, grouped or split
    routed weights, the Gate / Up context precision, the task decode (from ``num_local_experts``) and the H / I
    tail (from the workspace and ``intermediate_size``).  Sizes are runtime values of the kernel, so every tensor
    argument of the compile is a stand-in of the right rank and dtype: the device's 256 x 256 BF16 tile for the
    shared weights and outputs, a 256-row state of one expert for the routed ones, one int32 word for the
    schedule.  The workspace gives the peer address tables (``create_workspace`` built them).  A later ``forward_mxfp8``
    over real tensors computes the same key and finds the executor.
    """
    import torch
    from types import SimpleNamespace

    device = workspace.device
    check_device(device, "the MXFP8 forward")
    device_index = _current_device_index()
    if (device.index if device.index is not None else device_index) != device_index:
        raise RuntimeError("the workspace CUDA device must be current")
    if gate_up_context not in GATE_UP_CONTEXTS:
        raise ValueError(f"gate_up_context must be one of {GATE_UP_CONTEXTS}; got {gate_up_context!r}")
    plan = PersistentMxfp8Plan(
        ep_size=workspace.ep_size,
        num_local_experts=num_local_experts,
        hidden_size=workspace.hidden_size,
        routed_hidden_size=getattr(workspace, "routed_hidden_size", None),
        intermediate_size=intermediate_size,
        topk=workspace.topk,
        macrobatch_size=ROW_ALIGNMENT,
        minibatch_size=ROW_ALIGNMENT,
        num_comm_sms=2,
        intermediate_alignment=128,
        hidden_alignment=128,
    )
    plan.validate()
    runtime_environment = _prepare_public_runtime_environment()
    stand_in = prepare_persistent_mxfp8_state(
        num_local_tokens=2 * ROW_ALIGNMENT,
        schedule_capacity=ROW_ALIGNMENT,
        device=device,
        plan=_SHARED_FC1_PLACEHOLDER_PLAN,
        gate_up_context=gate_up_context,
        has_shared=False,
    )
    placeholder = shared_placeholder(device)
    routed_weight = stand_in.x_routed if split_weights else stand_in.x_routed.unsqueeze(0)  # [N, K] or [E, N, K]
    weights = (placeholder, routed_weight, placeholder, routed_weight, placeholder, routed_weight)
    zero = torch.zeros(4, dtype=torch.int32, device=device)[:1]
    schedule = SimpleNamespace(peer_rank=zero, peer_token_idx=zero, num_tokens=zero, tokens_per_expert=zero)
    is_clamped, is_situ = bool(is_clamped), bool(is_situ)
    task_decode = select_task_decode(num_local_experts)
    has_tail = _has_tail(plan.hidden_size, plan.routed_hidden, plan.intermediate_size)
    key = _executor_key(
        is_clamped=is_clamped,
        task_decode=task_decode,
        mxfp8_path="routed",
        runtime_environment=runtime_environment,
        device_index=device_index,
        context_key=_current_cuda_context_key(),
        gate_up_context=gate_up_context,
        split_weights=bool(split_weights),
        is_situ=is_situ,
        has_shared=has_shared,
        has_tail=has_tail,
    )
    positional_args, keyword_args = _launch_arguments(
        workspace,
        schedule,
        weights,
        (stand_in.x_routed_sc,) * 3,
        stand_in,
        plan=plan,
        shared_input=workspace.x_buffer,
        swiglu_limit_value=0.0,
        is_clamped=is_clamped,
        up_row_offset=0,
        situ=(1.0, 1.0) if is_situ else None,
        routed=routed_launch_tensors(None, num_local_experts, device),
        stream=current_stream_handle(device_index),
    )
    _executor(
        key,
        device_index,
        positional_args,
        keyword_args,
        is_clamped=is_clamped,
        task_decode=task_decode,
        gate_up_context=gate_up_context,
        split_weights=bool(split_weights),
        is_situ=is_situ,
        has_shared=has_shared,
        has_tail=has_tail,
    )


#: Each state class's fields, in order (its buffers and counters): the BF16 and the e4m3 Gate / Up context.
_STATE_FIELD_NAMES = {
    state_class: tuple(field.name for field in __import__("dataclasses").fields(state_class))
    for state_class in (PersistentMxfp8State, PersistentMxfp8StateE4m3Context)
}


def _fast_schedule_tensors(schedule) -> tuple[object, ...]:
    """The schedule tensors a launch reads (the CuTe DSL scheduler's valid counts included when present)."""
    tensors = (schedule.peer_rank, schedule.peer_token_idx, schedule.num_tokens, schedule.tokens_per_expert, getattr(schedule, "valid_tokens_per_expert", None))
    return tuple(tensor for tensor in tensors if tensor is not None)


def _fast_launch_sources(workspace, weights_and_scales, schedule_tensors, state, device_index) -> tuple:
    """Every per-call object a launch argument may be (or be a view of), in one fixed order: SlotTemplate's
    sources, the same for the call that builds a template and the calls that fill it."""
    return (
        workspace.x_buffer,
        workspace.combine_buffer,
        *weights_and_scales,
        *schedule_tensors,
        *(getattr(state, name) for name in _STATE_FIELD_NAMES[type(state)]),
        current_stream_handle(device_index),
    )


def _forward_outputs(state, has_shared: bool) -> tuple[object, ...]:
    if not has_shared:
        shared_indices = SHARED_OUTPUT_INDICES if isinstance(state, PersistentMxfp8StateE4m3Context) else SHARED_OUTPUT_INDICES_BF16_CONTEXT
        return tuple(None if index in shared_indices else output for index, output in enumerate(state.abi_outputs))
    return state.abi_outputs


#: shared_fc1_mxfp8's stand-in for the routed side of its launch: the state of a layer with one expert, hidden and
#: intermediate size 256 and one 256-row macrobatch (no shared buffers).  No task of that launch reads or writes
#: these buffers, so one set per device serves every call.
_SHARED_FC1_PLACEHOLDER_PLAN: Final = PersistentMxfp8Plan(
    num_local_experts=1, hidden_size=256, intermediate_size=256, topk=1, macrobatch_size=ROW_ALIGNMENT, minibatch_size=ROW_ALIGNMENT, num_comm_sms=2
)
_SHARED_FC1_PLACEHOLDERS: dict[object, PersistentMxfp8State] = {}


def shared_fc1_mxfp8(workspace, x_shared, shared_gate_weights, shared_up_weights, *, swiglu_limit, situ_beta=None):
    """Run the MXFP8 forward megakernel's shared Gate, Up and SwiGLU tasks alone and return ``(gate_shared,
    up_shared, hidden_shared)``, three fresh BF16 ``[num_local_tokens, intermediate_size]`` tensors: what
    ``forward_mxfp8`` keeps in its context for the same input and weights, bit for bit (the same task code over
    the same tiles; the shared expert is the same in both Gate / Up context variants).

    The launch is the megakernel's ``shared_fc1_only`` variant: no communication cluster, no routed task and no
    shared Down.  It takes no schedule, reads no symmetric buffer and waits for no other rank.  The kernel's other
    arguments are placeholders it builds descriptors over: a 256 x 256 routed geometry of one expert kept per
    device, the device's BF16 tile for the Down weights and ``y_shared``, and one zero word for every schedule
    tensor.

    ``x_shared``: the shared expert's input, ``[num_local_tokens, hidden_size]``.  ``swiglu_limit`` and
    ``situ_beta`` as in ``forward_mxfp8``.  One executor per clamp mode and activation serves every size.  A call
    validates its arguments and builds its launch arguments; no launch plan is kept.
    """

    import torch
    from dataclasses import replace
    from types import SimpleNamespace

    situ = normalize_situ_beta(situ_beta, swiglu_limit)
    swiglu_limit_value, is_clamped = _normalize_swiglu_limit(swiglu_limit)
    device, tokens, hidden = workspace.device, workspace.num_local_tokens, workspace.hidden_size
    intermediate = shared_gate_weights.shape[0] if getattr(shared_gate_weights, "ndim", 0) == 2 else -1
    # The shared expert's sizes, with the forward's 128-wide hidden and intermediate tails.
    _validate_num_local_tokens(
        tokens, plan=PersistentMxfp8BasePlan(hidden_size=hidden, intermediate_size=intermediate, intermediate_alignment=128, hidden_alignment=128)
    )
    check_device(device, "the MXFP8 forward")
    device_index = device.index if device.index is not None else _current_device_index()
    if _current_device_index() != device_index:
        raise RuntimeError("the workspace CUDA device must be current")
    _validate_tensor("x_shared", x_shared, shape=(tokens, hidden), dtype=torch.bfloat16, device=device)
    for name, weight in (("shared_gate_weights", shared_gate_weights), ("shared_up_weights", shared_up_weights)):
        _validate_tensor(name, weight, shape=(intermediate, hidden), dtype=torch.bfloat16, device=device)
    runtime_environment = _prepare_public_runtime_environment()
    placeholders = _SHARED_FC1_PLACEHOLDERS.get(device)
    if placeholders is None:
        placeholders = _SHARED_FC1_PLACEHOLDERS[device] = prepare_persistent_mxfp8_state(
            num_local_tokens=2 * ROW_ALIGNMENT,
            schedule_capacity=ROW_ALIGNMENT,
            device=device,
            plan=_SHARED_FC1_PLACEHOLDER_PLAN,
            gate_up_context="bf16",
            has_shared=False,
        )
    outputs = tuple(torch.empty((tokens, intermediate), dtype=torch.bfloat16, device=device) for _ in range(3))
    # One zeroed allocation on the current stream: a word that stands in for every schedule tensor, and the two
    # counters the shared tasks use -- gate_up_tile_ready (Gate / Up arrive, SwiGLU waits) and
    # hidden_row_block_ready (SwiGLU arrives).
    shared_row_blocks = tokens // HIDDEN_ROW_BLOCK
    _, (zero, gate_up_tile_ready, hidden_row_block_ready) = packed_int32_views(
        (1, shared_row_blocks * -(-intermediate // FC1_LOGICAL_N), shared_row_blocks), device
    )
    state = replace(
        placeholders,
        gate_shared=outputs[0],
        up_shared=outputs[1],
        hidden_shared=outputs[2],
        gate_up_tile_ready=gate_up_tile_ready,
        hidden_row_block_ready=hidden_row_block_ready,
    )
    schedule = SimpleNamespace(peer_rank=zero, peer_token_idx=zero, num_tokens=zero, tokens_per_expert=zero)
    routed_weight = placeholders.x_routed.unsqueeze(0)  # e4m3 [E, N, K] with one expert
    weights = (shared_gate_weights, routed_weight, shared_up_weights, routed_weight, shared_placeholder(device), routed_weight)
    positional_args, keyword_args = _mega_call_arguments(
        workspace,
        schedule,
        weights,
        state,
        plan=_SHARED_FC1_PLACEHOLDER_PLAN,
        swiglu_limit_value=swiglu_limit_value,
        is_clamped=is_clamped,
        stream=current_stream_handle(device_index),
    )
    # The builder's shared-input slot (the x buffer), as in forward_mxfp8.
    positional_args = (*positional_args[:6], x_shared, *positional_args[7:])
    keyword_args["routed_weight_scales"] = (placeholders.x_routed_sc,) * 3
    keyword_args["situ_beta"] = (0.0, 0.0) if situ is None else situ
    keyword_args["build_descriptors"] = 0
    # The shared expert is the same code in both Gate / Up context kernels; the BF16-context one has fewer
    # routed arguments.  has_tail is always set: in this launch it only makes the tile counts over I ceilings
    # (no stream, no routed task), so one executor serves aligned sizes and sizes with a tail.
    task_decode = select_task_decode(_SHARED_FC1_PLACEHOLDER_PLAN.num_local_experts)
    key = _executor_key(
        is_clamped=is_clamped,
        task_decode=task_decode,
        mxfp8_path="routed",
        runtime_environment=runtime_environment,
        device_index=device_index,
        context_key=_current_cuda_context_key(),
        gate_up_context="bf16",
        is_situ=situ is not None,
        shared_fc1_only=True,
        has_tail=True,
    )

    from ._persistent_mxfp8_fwd_mega import make_mega_args_mxfp8

    executor = _executor(
        key,
        device_index,
        positional_args,
        keyword_args,
        is_clamped=is_clamped,
        task_decode=task_decode,
        gate_up_context="bf16",
        is_situ=situ is not None,
        shared_fc1_only=True,
        has_tail=True,
    )
    executor(*make_mega_args_mxfp8(*positional_args, launch=True, **keyword_args).values())
    return outputs


__all__ = [
    "GATE_UP_CONTEXTS",
    "MXFP8_PATHS",
    "PersistentMxfp8Plan",
    "PersistentMxfp8State",
    "PersistentMxfp8StateE4m3Context",
    "FORWARD_SCRATCH",
    "clear_persistent_mxfp8_executor_cache",
    "compile_forward_mxfp8",
    "forward_mxfp8",
    "forward_mxfp8_buffer_specs",
    "forward_mxfp8_counter_words",
    "leading_view",
    "persistent_mxfp8_buffer_specs",
    "prepare_persistent_mxfp8_state",
    "shared_fc1_mxfp8",
    "validate_persistent_mxfp8_call",
]
