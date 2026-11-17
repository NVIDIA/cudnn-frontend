# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Host side of the MXFP8 backward megakernel: input validation, per-call state, executor cache, launch.

``backward_mxfp8`` takes the workspace, the schedule, the forward context and the weights and returns
eighteen outputs (``_OUTPUTS``).  The shared expert stays BF16; the routed weights arrive as MXFP8 tuples
(Gate / Up with their transposes, the transposed Down) and the forward context is the MXFP8 forward's
(macrobatch-sized; every macrobatch past the first is replayed into it).  Its routed Gate / Up are e4m3 /
scale pairs (``CONTEXT_NAMES_E4M3_CONTEXT``, a forward with the default ``gate_up_context="mxfp8"``) or one
BF16 tensor each (``CONTEXT_NAMES``, a ``"bf16"`` forward); the kernel variant follows the context's names.
``recompute_forward_context=True`` replays macrobatch 0 as well: the caller passes the shared expert's context only and
the routed context is this call's uninitialized scratch (``prepare_persistent_mxfp8_bwd_routed_context``).

Geometry is runtime: the executor is keyed on the compile-time variant (``_executor_key``: clamp mode,
task-decode variant chosen from the local expert count, Gate / Up context precision, weight and main_grad
layouts, ...) and the runtime environment, never on a size.  Only the executor and the launch plans of
repeated calls are cached; state, stream, pointers and runtime arguments are fresh for every call.
Torch/CUTLASS imports remain lazy.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import threading
from typing import Final

from ._host_common import validate_tensor as _validate_tensor
from ._host_common import has_tail as _has_tail, persistent_executor, process_independent
from ._host_common import (
    ValidatedCalls,
    current_stream_handle,
    normalize_situ_beta,
    tensor_layout_signature,
    typed_key,
    workspace_peer_tables,
)
from ._host_common import current_cuda_context_key as _current_cuda_context_key
from ._host_common import LaunchPlanCache, SlotTemplate, layouts_match, row_major_view, tensor_layout
from ._host_common import current_device_index as _current_device_index
from .persistent_bf16_bwd import _main_grad_dtype, validate_main_grads
from .persistent_bf16_fwd import (
    SHARED_PLACEHOLDER_SHAPE,
    routed_expert_count,
    routed_launch_tensors,
    select_task_decode,
    shared_placeholder,
    shared_presence,
    validate_up_row_offset,
)
from .persistent_bf16_fwd_contract import TaskDecode
from .persistent_mxfp8_fwd import (
    _normalize_swiglu_limit,
    _packed_offsets,
    _prepare_public_runtime_environment,
    leading_view,
    packed_int32_views,
)
from ._arch import TARGET_ARCH, TARGET_CAPABILITY, check_device

# The routed GEMMs' TMA descriptors (the e4m3 A / B operands and their scales: 14 + 14 CUtensorMaps of 128 B) are
# copied into the state's ``tensormaps`` buffer by a one-warp prologue launch before every backward mega, and the
# mega's producer warps read them there (``_persistent_mxfp8_bwd_mega.ROUTED_TMA_POSITIONS``, which asserts this
# count against its atom positions).
TENSORMAP_BYTES: Final = 128
# 2 x (7 + 7): the seven e4m3 A operands (x, dY, d_gate, d_up and the transposed dY, d_gate, d_up), the seven B
# operands (gate, up, down^T, gate^T, up^T, hidden^T, x^T) and one scale tensor for each of the fourteen.
ROUTED_TENSORMAPS: Final = 28


_CODEGEN_EPOCH: Final = (
    "mxfp8-bwd-ab6-acc1-sf6-tmem-slots-dsl48"
    "-v7-stream-all-macrobatches-32x256-warps7-buffers2-groups4-dy-warpquant"
    "-tma-desc-gmem-wgrad-valid-end-swiglu-split-quant-bf16gateup"
)
_ABI_EPOCH: Final = "persistent-mxfp8-bwd-eighteen-output-nine-counter-stream-heads-per-macrobatch-tensormaps" "-bf16-gate-up-context-v6"
# The backward of a gate_up_context="mxfp8" forward (the e4m3 / scale Gate / Up context).
_CODEGEN_EPOCH_E4M3_CONTEXT: Final = (
    "mxfp8-bwd-ab6-acc1-sf6-tmem-slots-dsl48"
    "-v7-stream-all-macrobatches-32x256-warps7-buffers2-groups4-dy-warpquant"
    "-tma-desc-gmem-wgrad-valid-end-swiglu-split-quant"
)
_ABI_EPOCH_E4M3_CONTEXT: Final = "persistent-mxfp8-bwd-eighteen-output-nine-counter-stream-heads-per-macrobatch" "-tensormaps-v5"

_EXECUTOR_CACHE: dict[tuple[object, ...], object] = {}
_EXECUTOR_CACHE_DEVICES: set[int] = set()
_EXECUTOR_CACHE_LOCK = threading.Lock()
# The call signatures whose full validation passed (layouts, devices, scalars and peer
# addresses -- no tensor address; the local address checks repeat on every call).
_VALIDATED_CALLS = ValidatedCalls()
# The weight gradients, in the order fused accumulation matches them up with MOK's main_grads ABI
# (functional.MAIN_GRAD_NAMES is the same order).
_WEIGHT_GRADIENT_ORDER: Final = (
    "d_w_shared_gate",
    "d_w_routed_gate",
    "d_w_shared_up",
    "d_w_routed_up",
    "d_w_shared_down",
    "d_w_routed_down",
)
_WEIGHT_GRADIENTS: Final = frozenset(_WEIGHT_GRADIENT_ORDER)
# The eighteen outputs, in return order.
_OUTPUTS: Final = (
    "d_x_shared",
    "d_x_routed",
    "d_gate_shared",
    "d_gate_fp8_routed",
    "d_gate_sc_routed",
    "d_up_shared",
    "d_up_fp8_routed",
    "d_up_sc_routed",
    "d_hidden_shared",
    "d_hidden_routed",
    "d_y_fp8_routed",
    "d_y_sc_routed",
    "d_w_shared_gate",
    "d_w_routed_gate",
    "d_w_shared_up",
    "d_w_routed_up",
    "d_w_shared_down",
    "d_w_routed_down",
)
_COUNTERS: Final = (
    "router_weights_ready",
    "d_y_routed_ready",
    "d_hidden_ready",
    "d_gate_up_ready",
    "d_x_routed_ready",
    "replayed_x_routed_ready",
    "replayed_gate_up_ready",
    "replayed_hidden_ready",
    "routed_buffers_done",
)
#: The shared expert's scratch and weight gradients; a routed-only layer has none, and they come
#: back None.
_SHARED_BUFFERS: Final = frozenset(("d_hidden_shared", "d_gate_shared", "d_up_shared", "d_x_shared", "d_w_shared_gate", "d_w_shared_up", "d_w_shared_down"))
# The tensors of a gate_up_context="bf16" forward's context, by name: the routed Gate / Up context is one BF16
# tensor each, which the replay overwrites in place.
CONTEXT_NAMES: Final = (
    "x_fp8_t_routed",
    "x_sc_t_routed",
    "gate_shared",
    "gate_routed_bf16",
    "up_shared",
    "up_routed_bf16",
    "hidden_shared",
    "hidden_fp8_t_routed",
    "hidden_sc_t_routed",
)
# The tensors of a gate_up_context="mxfp8" forward's context: the routed Gate / Up context as e4m3 / scale pairs,
# which the replay rewrites in place next to its BF16 ring in the state.
CONTEXT_NAMES_E4M3_CONTEXT: Final = (
    "x_fp8_t_routed",
    "x_sc_t_routed",
    "gate_shared",
    "gate_fp8_routed",
    "gate_sc_routed",
    "up_shared",
    "up_fp8_routed",
    "up_sc_routed",
    "hidden_shared",
    "hidden_fp8_t_routed",
    "hidden_sc_t_routed",
)
# The shared expert's context, which the caller always supplies; every other context name is routed, and with
# ``recompute_forward_context`` this call's scratch.
SHARED_CONTEXT_NAMES: Final = ("gate_shared", "up_shared", "hidden_shared")
ROUTED_CONTEXT_NAMES: Final = tuple(dict.fromkeys(name for name in (*CONTEXT_NAMES, *CONTEXT_NAMES_E4M3_CONTEXT) if name not in SHARED_CONTEXT_NAMES))


def mxfp8_bwd_routed_context_specs(*, intermediate_size: int, macrobatch_size: int, routed_hidden_size: int, gate_up_context: str = "mxfp8") -> tuple:
    """``(name, shape, kind)`` of the routed context, kind in bf16 / e4m3 / scale: the shapes and dtypes the MXFP8
    forward saves it in (``persistent_mxfp8_fwd.persistent_mxfp8_buffer_specs``) under the backward's names --
    the transposed x and Hidden with their scales, and Gate / Up as e4m3 / scale pairs (``"mxfp8"``) or one BF16
    tensor each (``"bf16"``)."""

    i, m, hr = intermediate_size, macrobatch_size, routed_hidden_size
    gate_up = (
        (
            ("gate_fp8_routed", (m, i), "e4m3"),
            ("gate_sc_routed", (m // 128, i // 128, 32, 16), "scale"),
            ("up_fp8_routed", (m, i), "e4m3"),
            ("up_sc_routed", (m // 128, i // 128, 32, 16), "scale"),
        )
        if gate_up_context == "mxfp8"
        else (("gate_routed_bf16", (m, i), "bf16"), ("up_routed_bf16", (m, i), "bf16"))
    )
    return (
        ("x_fp8_t_routed", (hr, m), "e4m3"),
        ("x_sc_t_routed", (hr // 128, m // 128, 32, 16), "scale"),
        *gate_up,
        ("hidden_fp8_t_routed", (i, m), "e4m3"),
        ("hidden_sc_t_routed", (i // 128, m // 128, 32, 16), "scale"),
    )


def mxfp8_bwd_buffer_specs(
    *,
    num_local_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_local_experts: int,
    macrobatch_size: int,
    routed_hidden_size: int | None = None,
    gate_up_context: str = "mxfp8",
) -> tuple:
    """``(name, shape, kind)`` of one call's allocations, kind in bf16 / e4m3 / scale / f32; ``tensormaps`` (the
    routed TMA descriptors' slots) is uint8, the ``scale`` kind.  ``gate_up_context="mxfp8"`` adds the BF16
    Gate / Up replay ring (the BF16 context is its own replay target).  The routed chain's buffers (the
    replayed x, dY, d_x_routed, the routed weight gradients) are in ``routed_hidden_size`` (the latent MoE;
    ``None`` = ``hidden_size``), the shared chain's (``d_x_shared``, the shared weight gradients) in
    ``hidden_size``."""

    t, h, i, e, m = num_local_tokens, hidden_size, intermediate_size, num_local_experts, macrobatch_size
    hr = h if routed_hidden_size is None else routed_hidden_size

    def scales(rows, cols):
        return (rows // 128, cols // 128, 32, 16)

    return (
        ("x_fp8_routed", (m, hr), "e4m3"),
        ("x_sc_routed", scales(m, hr), "scale"),
        *((("gate_routed", (m, i), "bf16"), ("up_routed", (m, i), "bf16")) if gate_up_context == "mxfp8" else ()),
        ("hidden_fp8_routed", (m, i), "e4m3"),
        ("hidden_sc_routed", scales(m, i), "scale"),
        ("router_weights", (m,), "f32"),
        ("d_router_weight_partials", (m, i // 128), "f32"),
        ("d_y_fp8_routed", (m, hr), "e4m3"),
        ("d_y_sc_routed", scales(m, hr), "scale"),
        ("d_y_fp8_t_routed", (hr, m), "e4m3"),
        ("d_y_sc_t_routed", scales(hr, m), "scale"),
        ("d_hidden_shared", (t, i), "bf16"),
        ("d_hidden_routed", (m, i), "bf16"),
        ("d_gate_shared", (t, i), "bf16"),
        ("d_gate_fp8_routed", (m, i), "e4m3"),
        ("d_gate_sc_routed", scales(m, i), "scale"),
        ("d_gate_fp8_t_routed", (i, m), "e4m3"),
        ("d_gate_sc_t_routed", scales(i, m), "scale"),
        ("d_up_shared", (t, i), "bf16"),
        ("d_up_fp8_routed", (m, i), "e4m3"),
        ("d_up_sc_routed", scales(m, i), "scale"),
        ("d_up_fp8_t_routed", (i, m), "e4m3"),
        ("d_up_sc_t_routed", scales(i, m), "scale"),
        ("d_x_shared", (t, h), "bf16"),
        ("d_x_routed", (m, hr), "bf16"),
        ("d_w_shared_gate", (i, h), "bf16"),
        ("d_w_routed_gate", (e, i, hr), "bf16"),
        ("d_w_shared_up", (i, h), "bf16"),
        ("d_w_routed_up", (e, i, hr), "bf16"),
        ("d_w_shared_down", (h, i), "bf16"),
        ("d_w_routed_down", (e, hr, i), "bf16"),
        ("tensormaps", (ROUTED_TENSORMAPS * TENSORMAP_BYTES,), "scale"),  # uint8 descriptor slots
    )


def mxfp8_bwd_counter_lengths(
    *, num_local_tokens: int, intermediate_size: int, schedule_capacity: int, macrobatch_size: int, minibatch_size: int
) -> tuple[int, ...]:
    """The lengths of the nine readiness counters, in ``_COUNTERS`` order; the communication task stream's head
    words (one per macrobatch, zeroed with the counters) follow ``d_y_routed_ready``'s minibatch entries."""

    minibatches = -(-schedule_capacity // minibatch_size)
    macrobatches = -(-schedule_capacity // macrobatch_size)
    shared_row_blocks = num_local_tokens // 256
    routed_row_blocks = schedule_capacity // 256
    column_blocks = -(-intermediate_size // 256)  # an I % 256 != 0 tail is one more tile
    return (
        macrobatches,
        minibatches + macrobatches,  # + the task stream's head words (macrobatch m's is the (m + 1)-th from the end)
        (shared_row_blocks + routed_row_blocks) * column_blocks,
        shared_row_blocks + routed_row_blocks,
        minibatches,
        minibatches,
        routed_row_blocks * column_blocks,
        routed_row_blocks,
        macrobatches,
    )


def _dtype(kind: str):
    import torch

    return {"bf16": torch.bfloat16, "e4m3": torch.float8_e4m3fn, "scale": torch.uint8, "f32": torch.float32}[kind]


@dataclass(frozen=True)
class PersistentMxfp8BwdState:
    """One call's allocations (``mxfp8_bwd_buffer_specs`` names, in its order) and its nine
    counters by name (``counters``, in ``__call__``'s order).  Every buffer is its own allocation
    (``prepare_persistent_mxfp8_bwd_state``); ``tensormaps`` holds the routed TMA descriptors the
    prologue launch writes."""

    x_fp8_routed: object
    x_sc_routed: object
    hidden_fp8_routed: object
    hidden_sc_routed: object
    router_weights: object
    d_router_weight_partials: object
    d_y_fp8_routed: object
    d_y_sc_routed: object
    d_y_fp8_t_routed: object
    d_y_sc_t_routed: object
    d_hidden_shared: object
    d_hidden_routed: object
    d_gate_shared: object
    d_gate_fp8_routed: object
    d_gate_sc_routed: object
    d_gate_fp8_t_routed: object
    d_gate_sc_t_routed: object
    d_up_shared: object
    d_up_fp8_routed: object
    d_up_sc_routed: object
    d_up_fp8_t_routed: object
    d_up_sc_t_routed: object
    d_x_shared: object
    d_x_routed: object
    d_w_shared_gate: object
    d_w_routed_gate: object
    d_w_shared_up: object
    d_w_routed_up: object
    d_w_shared_down: object
    d_w_routed_down: object
    tensormaps: object
    counters: dict

    @property
    def abi_outputs(self) -> tuple[object, ...]:
        """The eighteen outputs, in return order (``_OUTPUTS``)."""
        return tuple(getattr(self, name) for name in _OUTPUTS)


@dataclass(frozen=True)
class PersistentMxfp8BwdStateE4m3Context:
    """The state of a ``gate_up_context="mxfp8"`` backward: ``PersistentMxfp8BwdState``'s buffers and
    counters plus the BF16 Gate / Up replay ring (``gate_routed`` / ``up_routed``)."""

    x_fp8_routed: object
    x_sc_routed: object
    gate_routed: object
    up_routed: object
    hidden_fp8_routed: object
    hidden_sc_routed: object
    router_weights: object
    d_router_weight_partials: object
    d_y_fp8_routed: object
    d_y_sc_routed: object
    d_y_fp8_t_routed: object
    d_y_sc_t_routed: object
    d_hidden_shared: object
    d_hidden_routed: object
    d_gate_shared: object
    d_gate_fp8_routed: object
    d_gate_sc_routed: object
    d_gate_fp8_t_routed: object
    d_gate_sc_t_routed: object
    d_up_shared: object
    d_up_fp8_routed: object
    d_up_sc_routed: object
    d_up_fp8_t_routed: object
    d_up_sc_t_routed: object
    d_x_shared: object
    d_x_routed: object
    d_w_shared_gate: object
    d_w_routed_gate: object
    d_w_shared_up: object
    d_w_routed_up: object
    d_w_shared_down: object
    d_w_routed_down: object
    tensormaps: object
    counters: dict

    @property
    def abi_outputs(self) -> tuple[object, ...]:
        """The eighteen outputs, in return order (``_OUTPUTS``)."""
        return tuple(getattr(self, name) for name in _OUTPUTS)


#: The state's buffers by name, per state class: every field but the counters (the e4m3 Gate / Up context's state
#: has two more buffers).
_STATE_FIELDS_BY_CLASS: Final = {
    state_class: tuple(field.name for field in fields(state_class) if field.name != "counters")
    for state_class in (PersistentMxfp8BwdState, PersistentMxfp8BwdStateE4m3Context)
}


def prepare_persistent_mxfp8_bwd_state(
    *,
    num_local_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_local_experts: int,
    schedule_capacity: int,
    macrobatch_size: int,
    minibatch_size: int,
    device,
    routed_hidden_size: int | None = None,
    gate_up_context: str = "mxfp8",
    main_grads=None,
    combined_fc1: bool = False,
    has_shared: bool = True,
    caller_buffers=None,
) -> PersistentMxfp8BwdState | PersistentMxfp8BwdStateE4m3Context:
    """Allocate the backward's buffers, one ``torch.empty`` per tensor, and the nine counters as
    views of one zeroed allocation.

    The buffers are not packed into one allocation per lifetime and dtype: at large macrobatches a
    packed block reaches tens of GiB, and a CUDA Graph capture that shares the forward graph's
    memory pool cannot place a block that size although the same bytes fit as separate tensors.
    The kernel addresses every buffer through its own TMA descriptor or pointer and assumes only
    16-byte alignment, which every allocation satisfies -- except ``tensormaps``, whose TMA descriptors must be
    64-byte aligned (PTX) and which the kernel reads as 128-byte aligned slots; that one is checked here.

    ``combined_fc1``: Gate and Up are one combined FC1, so their weight gradient is one ``[E, 2I, H]``
    buffer under both names, Up written ``I`` rows down.

    ``has_shared=False``: a routed-only layer allocates no shared buffer; the device's placeholder
    stands in for each (in the weight gradients' dtype for the three shared dW), never touched.

    ``caller_buffers`` (the execute step of a plan, which allocates nothing): ``(scratch, d_weights, counters,
    d_x_shared)`` -- the plan's flat buffers by name (``backward_mxfp8_plan_sizes``), of which the leading elements
    are used; the caller's six weight-gradient buffers in ``main_grads`` order (shaped like the weights, a combined
    FC1's one ``[E, 2I, H]`` buffer under both Gate and Up), or None with ``main_grads``; the plan's int32
    counter buffer, whose leading words are zeroed on the current stream; and the caller's ``d_x_shared`` when
    the two hiddens differ (None: from the scratch)."""

    import torch

    specs = mxfp8_bwd_buffer_specs(
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=num_local_experts,
        macrobatch_size=macrobatch_size,
        gate_up_context=gate_up_context,
        routed_hidden_size=routed_hidden_size,
    )
    state_fields = [name for name, _, _ in specs]
    own_combined = combined_fc1 and main_grads is None
    if own_combined:
        routed_hidden = hidden_size if routed_hidden_size is None else routed_hidden_size
        specs = [
            (name, (num_local_experts, 2 * intermediate_size, routed_hidden), kind) if name == "d_w_routed_gate" else (name, shape, kind)
            for name, shape, kind in specs
            if name != "d_w_routed_up"
        ]
    # Fused accumulation: the six weight gradients *are* the caller's main_grad buffers, so only the
    # call's scratch is allocated.
    supplied = {} if main_grads is None else {name: buffer for name, buffer in zip(_WEIGHT_GRADIENT_ORDER, main_grads, strict=True) if buffer is not None}
    counter_storage = None
    if caller_buffers is not None:
        scratch, d_weights, counter_storage, d_x_shared = caller_buffers
        outputs = {} if d_weights is None else dict(zip(_WEIGHT_GRADIENT_ORDER, d_weights, strict=True))
        if d_x_shared is not None:
            outputs["d_x_shared"] = d_x_shared
        expected = {name: (shape, _dtype(kind)) for name, shape, kind in specs}
        for name, tensor in outputs.items():
            if tensor is None or (not has_shared and name in _SHARED_BUFFERS):
                continue
            if own_combined and name == "d_w_routed_up":
                if tensor is not outputs.get("d_w_routed_gate"):
                    raise ValueError("a combined FC1 passes one weight-gradient buffer under both Gate and Up")
                continue
            _validate_tensor(name, tensor, shape=expected[name][0], dtype=expected[name][1], device=device)
            supplied[name] = tensor
    if not has_shared:
        wgrad_dtype = torch.bfloat16 if main_grads is None else _main_grad_dtype(main_grads)
        supplied.update((name, shared_placeholder(device, wgrad_dtype if name in _WEIGHT_GRADIENTS else None)) for name in _SHARED_BUFFERS)

    def buffer(name, shape, kind):
        if caller_buffers is None:
            return torch.empty(shape, dtype=_dtype(kind), device=device)
        return leading_view(scratch.get(name), shape, _dtype(kind), device, f"the backward scratch {name}")

    buffers = {name: buffer(name, shape, kind) for name, shape, kind in specs if name not in supplied}
    if buffers["tensormaps"].data_ptr() % TENSORMAP_BYTES:
        raise RuntimeError("the tensormaps buffer must be 128-byte aligned (the TMA descriptor slots)")
    _, counters = packed_int32_views(
        mxfp8_bwd_counter_lengths(
            num_local_tokens=num_local_tokens,
            intermediate_size=intermediate_size,
            schedule_capacity=schedule_capacity,
            macrobatch_size=macrobatch_size,
            minibatch_size=minibatch_size,
        ),
        device,
        storage=counter_storage,
    )
    buffers.update(supplied)
    if own_combined:
        buffers["d_w_routed_up"] = buffers["d_w_routed_gate"]
    state_class = PersistentMxfp8BwdStateE4m3Context if gate_up_context == "mxfp8" else PersistentMxfp8BwdState
    return state_class(**{name: buffers[name] for name in state_fields}, counters=dict(zip(_COUNTERS, counters, strict=True)))


def prepare_persistent_mxfp8_bwd_routed_context(
    *, intermediate_size: int, macrobatch_size: int, routed_hidden_size: int, device, gate_up_context: str = "mxfp8"
) -> dict:
    """The routed context of a ``recompute_forward_context`` backward by name (``mxfp8_bwd_routed_context_specs``): one
    ``torch.empty`` per tensor, as the state's buffers.  Nothing is initialized: the kernel replays every
    macrobatch into these buffers before it reads them."""

    import torch

    return {
        name: torch.empty(shape, dtype=_dtype(kind), device=device)
        for name, shape, kind in mxfp8_bwd_routed_context_specs(
            intermediate_size=intermediate_size, macrobatch_size=macrobatch_size, routed_hidden_size=routed_hidden_size, gate_up_context=gate_up_context
        )
    }


def _recomputed_routed_context(workspace, routed_gate, macrobatch_size, up_row_offset, gate_up_context: str) -> dict:
    """``backward_mxfp8``'s routed context with ``recompute_forward_context``, allocated from the geometry the call
    reads from its workspace and routed Gate weight.  Empty for a geometry that cannot size a buffer: the entries then
    stay None and ``validate_persistent_mxfp8_bwd_call`` reports the geometry, which it checks before the context."""

    routed_hidden_size = getattr(workspace, "routed_hidden_size", None)
    if routed_hidden_size is None:
        routed_hidden_size = getattr(workspace, "hidden_size", -1)
    intermediate_size = getattr(routed_gate, "shape", (-1, -1))[-2] - up_row_offset
    if any(type(extent) is not int or extent <= 0 or extent % 128 for extent in (intermediate_size, macrobatch_size, routed_hidden_size)):
        return {}
    return prepare_persistent_mxfp8_bwd_routed_context(
        intermediate_size=intermediate_size,
        macrobatch_size=macrobatch_size,
        routed_hidden_size=routed_hidden_size,
        device=workspace.device,
        gate_up_context=gate_up_context,
    )


def validate_persistent_mxfp8_bwd_call(
    workspace,
    schedule,
    context: dict,
    weights: dict,
    *,
    num_local_experts: int,
    intermediate_size: int,
    macrobatch_size: int,
    minibatch_size: int,
    num_comm_sms: int,
    x_shared=None,
    d_y_shared=None,
    native_columnwise: bool = False,
    up_row_offset: int = 0,
    routed_storage_tables=None,
    has_shared: bool = True,
) -> None:
    """Fail closed before importing CUTLASS: the symmetric buffers and their peer tables,
    the BF16 shared weights, the MXFP8 routed weights and the macrobatch-sized MXFP8 context.
    The symmetric buffers, the routed weights and the routed context are in the routed hidden
    (``workspace.routed_hidden_size``; the latent MoE); ``x_shared`` / ``d_y_shared`` are the shared chain's
    input and output gradient in ``hidden_size``, required when the two hiddens differ.

    ``native_columnwise`` picks the shape the transposed routed payloads must have.
    ``up_row_offset`` is ``intermediate_size`` when Gate and Up are one combined FC1 passed whole
    as both arguments -- every payload and scale of it -- and 0 for separate weights.

    ``routed_storage_tables`` are the ten per-expert address tables -- the five weights the dgrad
    reads, then their five scales -- when the routed experts are independent per-expert
    allocations.  Each routed operand is then one expert's own matrix rather than an ``[E, ...]``
    parameter, and its scale covers one expert rather than all of them."""

    import torch

    required = (
        "device",
        "ep_size",
        "ep_rank",
        "num_local_tokens",
        "hidden_size",
        "topk",
        "schedule_capacity",
        "x_buffer",
        "x_buffer_ptrs",
        "d_y_buffer",
        "d_y_buffer_ptrs",
        "d_x_routed_buffer",
        "d_x_routed_buffer_ptrs",
        "router_weight_buffer",
        "router_weight_buffer_ptrs",
        "d_router_weight_buffer",
        "d_router_weight_buffer_ptrs",
    )
    missing = tuple(name for name in required if not hasattr(workspace, name))
    if missing:
        raise TypeError(f"workspace is missing fields: {missing}")
    device = workspace.device
    check_device(device, "the MXFP8 backward")
    t, h, k = workspace.num_local_tokens, workspace.hidden_size, workspace.topk
    hr = getattr(workspace, "routed_hidden_size", None)
    hr = h if hr is None else hr
    i, e = intermediate_size, num_local_experts
    m, b, cap = macrobatch_size, minibatch_size, workspace.schedule_capacity
    for name, value in (
        ("num_local_tokens", t),
        ("hidden_size", h),
        ("routed_hidden_size", hr),
        ("topk", k),
        ("intermediate_size", i),
        ("num_local_experts", e),
        ("macrobatch_size", m),
        ("minibatch_size", b),
        ("schedule_capacity", cap),
        ("num_comm_sms", num_comm_sms),
        ("ep_size", workspace.ep_size),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive int; got {value!r}")
    # The same rules and error type as the forward contract's validate_geometry.
    # The shared hidden only needs 128 alignment (checked with the workspace); a latent layer's routed hidden keeps 256.
    for name, value in (("num_local_tokens", t), ("minibatch_size", b), *((("routed_hidden_size", hr),) if hr != h else ())):
        if value % 256:
            raise ValueError(f"{name} must be positive and divisible by 256")
    # The hidden extent may end in a 128-wide tail tile.
    if h < 256 or h % 128:
        raise ValueError("hidden_size must be at least 256 and divisible by 128")
    # So may the intermediate extent.
    if i < 256 or i % 128:
        raise ValueError("intermediate_size must be at least 256 and divisible by 128")
    if not 1 <= k <= 255:
        raise ValueError("topk must be an integer in [1, 255]")
    if m % b:
        raise ValueError("macrobatch_size must be a positive multiple of minibatch_size")
    if num_comm_sms % 2:
        raise ValueError("num_comm_sms must be a positive even integer")
    if not 0 <= workspace.ep_rank < workspace.ep_size:
        raise ValueError("workspace.ep_rank is outside [0, ep_size)")
    for name, shape, dtype in (
        ("x_buffer", (t, hr), torch.bfloat16),
        ("d_y_buffer", (t, hr), torch.bfloat16),
        ("d_x_routed_buffer", (t * k, hr), torch.bfloat16),
        ("router_weight_buffer", (t, k), torch.float32),
        ("d_router_weight_buffer", (t, k), torch.float32),
    ):
        buffer = getattr(workspace, name)
        _validate_tensor(f"workspace.{name}", buffer, shape=shape, dtype=dtype, device=device)
        pointers = getattr(workspace, f"{name}_ptrs")
        if (
            not isinstance(pointers, list)
            or len(pointers) != workspace.ep_size
            or any(type(pointer) is not int or pointer <= 0 or pointer % 16 for pointer in pointers)
        ):
            raise ValueError(f"workspace.{name}_ptrs must contain ep_size aligned addresses")
        if pointers[workspace.ep_rank] != buffer.data_ptr():
            raise ValueError(f"workspace.{name}_ptrs does not own the local buffer")
    if (x_shared is None or d_y_shared is None) and hr != h:
        raise ValueError("x_shared and d_y_shared are required when the routed hidden differs from the shared " "hidden")
    for name, tensor in (("x_shared", x_shared), ("d_y_shared", d_y_shared)):
        if tensor is not None:
            _validate_tensor(name, tensor, shape=(t, h), dtype=torch.bfloat16, device=device)
    # A routed-only layer passes the placeholder for each shared weight and context tensor.
    i_h, h_i, t_i = ((i, h), (h, i), (t, i)) if has_shared else (SHARED_PLACEHOLDER_SHAPE,) * 3
    for name, shape in (("shared_gate", i_h), ("shared_up", i_h), ("shared_down", h_i)):
        _validate_tensor(f"weights.{name}", weights[name], shape=shape, dtype=torch.bfloat16, device=device)
    # A routed payload may be a slice of a larger parameter (a combined [E, 2I, H] FC1).  The
    # _t ones are MOK's transposed weights, or -- native_columnwise -- the parameter's own shape
    # with TransformerEngine's columnwise quantization.
    split_weights = routed_storage_tables is not None
    if split_weights:
        if len(routed_storage_tables) != 10:
            raise ValueError("routed_storage_tables must hold the five weights the dgrad reads and then their " "five scales")
        for index, table in enumerate(routed_storage_tables):
            _validate_tensor(f"routed_storage_tables[{index}]", table, shape=(e,), dtype=torch.int64, device=device)

    def routed_shape(*dims):
        """One expert's own matrix when the experts are split, the [E, ...] parameter otherwise."""
        return dims if split_weights else (e, *dims)

    combined = validate_up_row_offset(up_row_offset, i, weights["routed_gate"], weights["routed_up"], weights["routed_gate_sc"], weights["routed_up_sc"])
    if combined and (weights["routed_gate_t"] is not weights["routed_up_t"] or weights["routed_gate_t_sc"] is not weights["routed_up_t_sc"]):
        raise ValueError("a combined FC1 passes one transposed payload and scale as both Gate and Up")
    fc1 = 2 * i if combined else i
    for name, shape in (
        ("routed_gate", routed_shape(fc1, hr)),
        ("routed_up", routed_shape(fc1, hr)),
        ("routed_gate_t", routed_shape(fc1, hr) if native_columnwise else routed_shape(hr, fc1)),
        ("routed_up_t", routed_shape(fc1, hr) if native_columnwise else routed_shape(hr, fc1)),
        ("routed_down_t", routed_shape(hr, i) if native_columnwise else routed_shape(i, hr)),
    ):
        _validate_tensor(f"weights.{name}", weights[name], shape=shape, dtype=torch.float8_e4m3fn, device=device, sliced=True)
    # The scales are MOK's tcgen05 layout either way: the rowwise one over the payload's rows, the
    # transposed one over its columns, expert-major over 128-row blocks.  A split scale covers one
    # expert.  Exact shapes: the kernel derives the per-expert stride from this extent
    # (scale_operand).
    scale_experts = 1 if split_weights else e
    for name, m_rows, k_columns in (
        ("routed_gate_sc", fc1, hr),
        ("routed_up_sc", fc1, hr),
        ("routed_gate_t_sc", hr, fc1),
        ("routed_up_t_sc", hr, fc1),
        ("routed_down_t_sc", i, hr),
    ):
        _validate_tensor(f"weights.{name}", weights[name], shape=(scale_experts * m_rows // 128, k_columns // 128, 32, 16), dtype=torch.uint8, device=device)
    for name, shape, dtype in (
        ("x_fp8_t_routed", (hr, m), torch.float8_e4m3fn),
        ("x_sc_t_routed", (hr // 128, m // 128, 32, 16), torch.uint8),
        ("gate_shared", t_i, torch.bfloat16),
        ("up_shared", t_i, torch.bfloat16),
        *(
            (
                ("gate_fp8_routed", (m, i), torch.float8_e4m3fn),
                ("gate_sc_routed", (m // 128, i // 128, 32, 16), torch.uint8),
                ("up_fp8_routed", (m, i), torch.float8_e4m3fn),
                ("up_sc_routed", (m // 128, i // 128, 32, 16), torch.uint8),
            )
            if "gate_fp8_routed" in context
            else (("gate_routed_bf16", (m, i), torch.bfloat16), ("up_routed_bf16", (m, i), torch.bfloat16))
        ),
        ("hidden_shared", t_i, torch.bfloat16),
        ("hidden_fp8_t_routed", (i, m), torch.float8_e4m3fn),
        ("hidden_sc_t_routed", (i // 128, m // 128, 32, 16), torch.uint8),
    ):
        _validate_tensor(f"context.{name}", context[name], shape=shape, dtype=dtype, device=device)
    for name, tensor, shape in (
        ("schedule.peer_rank", schedule.peer_rank, (cap,)),
        ("schedule.peer_token_idx", schedule.peer_token_idx, (cap,)),
        ("schedule.num_tokens", schedule.num_tokens, (1,)),
        ("schedule.tokens_per_expert", schedule.tokens_per_expert, (e,)),
    ):
        _validate_tensor(name, tensor, shape=shape, dtype=torch.int32, device=device)


def _executor_key(
    *,
    is_clamped,
    task_decode,
    runtime_environment,
    device_index,
    context_key,
    gate_up_context="mxfp8",
    split_weights: bool = False,
    main_grad_dtype=None,
    split_main_grads: bool = False,
    native_columnwise=False,
    is_situ: bool = False,
    has_shared: bool = True,
    has_tail: bool = False,
    recompute_forward_context: bool = False,
) -> tuple[object, ...]:
    """The executor cache key.  Only what selects code is in it: the codegen / ABI epochs, the compile-time
    variant flags (clamp, SiTU, shared presence, tail tiles, replay of macrobatch 0, task decode,
    Gate / Up context precision, routed weight and main_grad layouts, columnwise layout of the transposed
    weights), the target, the runtime version and the CUDA context.  Sizes, the two hiddens and a combined
    FC1's Up row offset included, are runtime arguments."""
    epochs = (_CODEGEN_EPOCH_E4M3_CONTEXT, _ABI_EPOCH_E4M3_CONTEXT) if gate_up_context == "mxfp8" else (_CODEGEN_EPOCH, _ABI_EPOCH)
    return (
        "mok-mxfp8-bwd-runtime",
        *epochs,
        ("gate_up_context", gate_up_context),
        ("device", device_index),
        ("context", context_key),
        ("arch", TARGET_ARCH, TARGET_CAPABILITY),
        ("runtime", runtime_environment),
        ("peer_pointer_abi", "int64-address-table-runtime-length"),
        ("tensor_abi", "e4m3-e8m0-sf32-routed-bf16-shared-i32-schedule-dynamic-layout"),
        ("is_clamped", bool(is_clamped)),
        ("is_situ", bool(is_situ)),
        ("has_shared", bool(has_shared)),
        # Compile-time as well: a 128-wide H / I tail tile.
        ("has_tail", bool(has_tail)),
        # Compile-time: macrobatch 0 is replayed instead of read from the forward's routed context.
        ("recompute_forward_context", bool(recompute_forward_context)),
        ("task_decode", TaskDecode(task_decode).name),
        ("native_columnwise", bool(native_columnwise)),
        # The routed operands' rank differs between the two layouts, and the flag is what selects
        # the code, so key it directly.
        ("routed_weights", "split" if split_weights else "grouped"),
        # Fused wgrad accumulation, its destination dtype and whether those buffers are split:
        # all compile-time.
        ("main_grad", "none" if main_grad_dtype is None else str(main_grad_dtype), "split" if split_main_grads else "grouped"),
    )


def clear_persistent_mxfp8_bwd_executor_cache(*, synchronize: bool = True) -> None:
    """Release executor-only cache entries after public calls are quiescent."""

    with _EXECUTOR_CACHE_LOCK:
        if synchronize and _EXECUTOR_CACHE_DEVICES:
            import torch

            for device_index in sorted(_EXECUTOR_CACHE_DEVICES):
                torch.cuda.synchronize(device_index)
        _EXECUTOR_CACHE.clear()
        _EXECUTOR_CACHE_DEVICES.clear()
        _VALIDATED_CALLS.clear()
        _FAST_PLANS.clear()


#: Launch plans of backward_mxfp8's repeated calls (see _MxBwdPlan).
_FAST_PLANS = LaunchPlanCache()


def _permute_10(tensor):
    return tensor.permute(1, 0) if tensor.ndim == 2 else None


def _permute_120(tensor):
    return tensor.permute(1, 2, 0) if tensor.ndim == 3 else None


def _permute_210(tensor):
    return tensor.permute(2, 1, 0) if tensor.ndim == 3 else None


#: The views make_mega_args_mxfp8_bwd derives from a per-call tensor (SlotTemplate transforms): row-major
#: normalization, the K-major / MN-major transposes of 2-D operands, the routed [N, K, E] operand view and the
#: native columnwise one.
_TRANSFORMS = (row_major_view, _permute_10, _permute_120, _permute_210)


@dataclass(frozen=True)
class _MxBwdPlan:
    """What backward_mxfp8's full path validated and built for one set of workspace, weights, main_grads
    and geometry: a later call with the same ones checks its schedule's, context's and d_y_shared's
    layouts, allocates its state and fills the template's per-call slots (SlotTemplate)."""

    executor: object
    template: object
    layouts: tuple
    num_local_experts: int
    intermediate_size: int
    routed: object


def _fast_bwd_sources(workspace, weights, context, main_grads, d_y_shared, schedule_tensors, state, stream) -> tuple:
    """Every per-call object a backward launch argument may be (or be a view of), in one fixed order."""
    return (
        workspace.x_buffer,
        workspace.d_y_buffer,
        workspace.d_x_routed_buffer,
        workspace.router_weight_buffer,
        workspace.d_router_weight_buffer,
        *weights.values(),
        *context.values(),
        *(() if main_grads is None else (grad for grad in main_grads if grad is not None)),
        *(() if d_y_shared is None else (d_y_shared,)),
        *schedule_tensors,
        *(() if state is None else (*(getattr(state, name) for name in _STATE_FIELDS_BY_CLASS[type(state)]), *state.counters.values())),
        stream,
    )


def _fast_bwd_schedule_tensors(schedule) -> tuple[object, ...]:
    tensors = (schedule.peer_rank, schedule.peer_token_idx, schedule.num_tokens, schedule.tokens_per_expert, getattr(schedule, "valid_tokens_per_expert", None))
    return tuple(tensor for tensor in tensors if tensor is not None)


def _bwd_outputs(state, has_shared: bool) -> tuple[object, ...]:
    if not has_shared:
        return tuple(None if name in _SHARED_BUFFERS else getattr(state, name) for name in _OUTPUTS)
    return state.abi_outputs


def backward_mxfp8(
    workspace,
    schedule,
    forward_context: dict,
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
    swiglu_limit: float | None = None,
    x_shared=None,
    d_y_shared=None,
    native_columnwise: bool = False,
    up_row_offset: int = 0,
    routed_storage_tables=None,
    main_grads=None,
    main_grad_storage_tables=None,
    main_grad_up_row_offset: int = 0,
    situ_beta=None,
    recompute_forward_context: bool = False,
    gate_up_context: str | None = None,
    caller_buffers=None,
) -> tuple[object, ...]:
    """Run the MXFP8 backward and return its eighteen outputs (``_OUTPUTS`` order).

    ``d_y_shared``: the shared chain's output gradient when it differs from the routed chain's
    (``dY * G`` under a shared output gate), BF16 ``[T, H]``.

    ``situ_beta``: the forward's SiTU-GLU ``(beta1, beta2)``, or None for SwiGLU.

    The three shared weights all None: a routed-only layer, whose context has no shared
    activations either.  There are no shared tasks or buffers; the shared main_grad slots are
    ignored and the seven shared outputs come back None.

    ``forward_context``: the MXFP8 forward context by ``CONTEXT_NAMES``, or by ``CONTEXT_NAMES_E4M3_CONTEXT`` for the
    context of a ``gate_up_context="mxfp8"`` forward (e4m3 / scale Gate / Up pairs; the kernel variant
    follows the names).  The routed parts are overwritten by the replay.  Routed weights: Gate / Up
    ``(e4m3, scale, e4m3_t, scale_t)``, Down ``(e4m3_t, scale_t)``.

    ``recompute_forward_context``: macrobatch 0 is replayed like every later macrobatch (a compile-time variant, part of
    the executor key), so no routed context is read: ``forward_context`` holds the shared expert's entries
    (``SHARED_CONTEXT_NAMES``) and its routed entries (``ROUTED_CONTEXT_NAMES``) are absent or None -- passing
    one is a ``ValueError`` -- and this call allocates them uninitialized, in the forward's shapes and dtypes
    (``prepare_persistent_mxfp8_bwd_routed_context``).  ``gate_up_context`` (``"mxfp8"``, the default, or
    ``"bf16"``) then names the routed Gate / Up representation the replay keeps.  Without ``recompute_forward_context``
    the representation is the context's own; a ``gate_up_context`` that contradicts it is a ``ValueError``.

    ``native_columnwise``: the ``_t`` payloads are TransformerEngine's columnwise data, which keeps
    the parameter's own ``(E, K, N)`` shape instead of MOK's transposed ``(E, N, K)``.  Their scales
    are in MOK's layout either way, and so is the arithmetic; the dgrads simply read an MN-major
    weight operand.  It is part of the executor key.

    ``routed_storage_tables`` / ``main_grads`` / ``main_grad_storage_tables`` /
    ``main_grad_up_row_offset``: split routed weights (ten per-expert address tables) and fused
    weight-gradient accumulation into the caller's main_grad buffers (optionally split per expert,
    optionally one combined FC1 buffer with Up that many rows down).

    ``up_row_offset``: ``intermediate_size`` when Gate and Up are one combined FC1 passed whole
    as both arguments -- payloads and scales -- and 0 for separate weights.  The kernel finds Up
    that far down every one of them: on
    the rows of the rowwise payload and of a native columnwise one, on the columns of MOK's
    transposed payload, on the M blocks of the rowwise scale and the K blocks of the transposed
    one (transposing makes the combined 2I the contraction dimension).  The workspace's
    symmetric buffers hold dY, x and the router weights of this call.  ``x_shared`` / ``d_y_shared`` are the
    shared chain's input and output gradient in ``hidden_size``; they default to the workspace's x / dY buffers
    and are required when ``workspace.routed_hidden_size`` differs from ``hidden_size`` (the latent MoE: the
    symmetric buffers, the routed weights, the routed context and the routed outputs are then in the routed
    hidden; the kernel reads both hiddens from the tensors, one executor serves both).  Only the executor is
    cached; state, stream, pointers and runtime arguments are fresh for every call.

    ``caller_buffers`` (the execute step of a plan): the plan's scratch and counters and the caller's weight
    gradient and ``d_x_shared`` buffers, as ``prepare_persistent_mxfp8_bwd_state`` takes them; the call then
    allocates nothing.  Not with ``recompute_forward_context``, whose routed context this call allocates.
    """

    gate, gate_sc, gate_t, gate_t_sc = routed_gate_weights
    up, up_sc, up_t, up_t_sc = routed_up_weights
    down_t, down_t_sc = routed_down_weights
    has_shared = shared_presence(shared_gate_weights, shared_up_weights, shared_down_weights)
    recompute_forward_context = bool(recompute_forward_context)
    if recompute_forward_context and caller_buffers is not None:
        raise NotImplementedError("the execute step of a plan needs the forward context's routed activations")
    if recompute_forward_context:
        if gate_up_context is None:
            gate_up_context = "mxfp8"
        elif gate_up_context not in ("bf16", "mxfp8"):
            raise ValueError("gate_up_context must be 'bf16' or 'mxfp8'")
        supplied = tuple(name for name in ROUTED_CONTEXT_NAMES if forward_context.get(name) is not None)
        if supplied:
            raise ValueError("recompute_forward_context=True replays the routed context: forward_context must not " f"hold {supplied}")
        # The routed entries are allocated below, once the shared ones are in place.
        context = {
            name: forward_context[name] if name in SHARED_CONTEXT_NAMES else None
            for name in (CONTEXT_NAMES_E4M3_CONTEXT if gate_up_context == "mxfp8" else CONTEXT_NAMES)
        }
    else:
        context_precision = "mxfp8" if "gate_fp8_routed" in forward_context else "bf16"
        if gate_up_context is not None and gate_up_context != context_precision:
            raise ValueError(f"gate_up_context={gate_up_context!r} does not match the forward context's routed " f"Gate / Up ({context_precision!r})")
        gate_up_context = context_precision
        context = {name: forward_context[name] for name in (CONTEXT_NAMES_E4M3_CONTEXT if gate_up_context == "mxfp8" else CONTEXT_NAMES)}
    if shared_presence(context["gate_shared"], context["up_shared"], context["hidden_shared"]) != has_shared:
        raise ValueError("shared weight presence must match the forward context")
    if not has_shared:
        placeholder = shared_placeholder(workspace.device)
        shared_gate_weights = shared_up_weights = shared_down_weights = placeholder
        context.update(gate_shared=placeholder, up_shared=placeholder, hidden_shared=placeholder)
        if main_grads is not None and len(main_grads) == 6:
            main_grads = (None, main_grads[1], None, main_grads[3], None, main_grads[5])
    if recompute_forward_context:
        context.update(_recomputed_routed_context(workspace, gate, macrobatch_size, up_row_offset, gate_up_context))
    weights = dict(
        shared_gate=shared_gate_weights,
        shared_up=shared_up_weights,
        shared_down=shared_down_weights,
        routed_gate=gate,
        routed_gate_sc=gate_sc,
        routed_up=up,
        routed_up_sc=up_sc,
        routed_gate_t=gate_t,
        routed_gate_t_sc=gate_t_sc,
        routed_up_t=up_t,
        routed_up_t_sc=up_t_sc,
        routed_down_t=down_t,
        routed_down_t_sc=down_t_sc,
    )
    # The fast path: a repeated workspace, weights, main_grads and geometry (_MxBwdPlan).
    fast_per_call = None
    try:
        fast_sources = (
            workspace.x_buffer,
            workspace.d_y_buffer,
            workspace.d_x_routed_buffer,
            workspace.router_weight_buffer,
            workspace.d_router_weight_buffer,
            *(routed_storage_tables or ()),
            *(main_grad_storage_tables or ()),
        )
        fast_views = (*weights.values(), *(() if main_grads is None else (grad for grad in main_grads if grad is not None)))
        fast_extra = typed_key(
            macrobatch_size,
            minibatch_size,
            num_comm_sms,
            swiglu_limit,
            native_columnwise,
            up_row_offset,
            main_grad_up_row_offset,
            situ_beta,
            has_shared,
            d_y_shared is None,
            routed_storage_tables is None,
            main_grad_storage_tables is None,
            gate_up_context,
            recompute_forward_context,
            getattr(workspace, "routed_hidden_size", None),
            None if main_grads is None else tuple(grad is None for grad in main_grads),
            workspace.ep_size,
            workspace.ep_rank,
            workspace.num_local_tokens,
            workspace.hidden_size,
            workspace.topk,
            workspace.schedule_capacity,
            workspace.x_buffer_ptrs,
            workspace.d_y_buffer_ptrs,
            workspace.d_x_routed_buffer_ptrs,
            workspace.router_weight_buffer_ptrs,
            workspace.d_router_weight_buffer_ptrs,
        ) + (
            _current_device_index(),
            _current_cuda_context_key(),
            workspace.device,
            # The decode variant joins the key as in the BF16 plans: a plan is one executor's.
            select_task_decode(int(routed_expert_count(routed_storage_tables, gate))).name,
        )
        fast_schedule = _fast_bwd_schedule_tensors(schedule)
        fast_per_call = (*fast_schedule, *context.values(), *(() if d_y_shared is None else (d_y_shared,)))
        # The latent MoE's x_shared is a per-call tensor the launch template does not carry: full path.
        fast = None if x_shared is not None else _FAST_PLANS.get(fast_sources, fast_extra, fast_views)
    except (AttributeError, TypeError, ValueError):
        fast_sources = fast_extra = fast = None
    if fast is not None and layouts_match(fast_per_call, fast.layouts, 16):
        state = prepare_persistent_mxfp8_bwd_state(
            num_local_tokens=workspace.num_local_tokens,
            hidden_size=workspace.hidden_size,
            intermediate_size=fast.intermediate_size,
            num_local_experts=fast.num_local_experts,
            schedule_capacity=workspace.schedule_capacity,
            macrobatch_size=macrobatch_size,
            minibatch_size=minibatch_size,
            device=workspace.device,
            main_grads=main_grads,
            combined_fc1=bool(up_row_offset),
            has_shared=has_shared,
            gate_up_context=gate_up_context,
            caller_buffers=caller_buffers,
        )
        stream = current_stream_handle(fast_extra[-4])
        fast.executor(
            *fast.template.fill(
                _fast_bwd_sources(workspace, weights, context, main_grads, d_y_shared, fast_schedule, state, stream),
                _TRANSFORMS,
                build_descriptors=fast.routed.build,
            )
        )
        fast.routed.mark_built()
        return _bwd_outputs(state, has_shared)
    # Split allocations: the weight is one expert's own matrix, so the count comes from the
    # address table rather than from a leading extent.
    num_local_experts = routed_expert_count(routed_storage_tables, gate)
    # From the routed FC1 (its row extent, less the Up half when combined): a routed-only layer has
    # no shared weights to read it from.
    intermediate_size = getattr(gate, "shape", (-1, -1))[-2] - up_row_offset
    swiglu_limit_value, is_clamped = _normalize_swiglu_limit(swiglu_limit)
    situ = normalize_situ_beta(situ_beta, swiglu_limit)
    hidden_size = getattr(workspace, "hidden_size", -1)
    routed_hidden_size = getattr(workspace, "routed_hidden_size", None)
    routed_hidden_size = hidden_size if routed_hidden_size is None else routed_hidden_size
    if routed_hidden_size != hidden_size and (x_shared is None or d_y_shared is None):
        raise ValueError("x_shared and d_y_shared are required when workspace.routed_hidden_size != hidden_size")
    # The shared chain's input and output gradient: the caller's, or the symmetric buffers when the hiddens coincide.
    shared_inputs = (workspace.x_buffer if x_shared is None else x_shared, workspace.d_y_buffer if d_y_shared is None else d_y_shared)

    import torch

    if d_y_shared is not None:
        _validate_tensor("d_y_shared", d_y_shared, shape=(workspace.num_local_tokens, workspace.hidden_size), dtype=torch.bfloat16, device=workspace.device)
    # One host path, as the forward: the full input check runs when the call signature is new,
    # the local address checks on every call.
    pointer_lists = ("x_buffer_ptrs", "d_y_buffer_ptrs", "d_x_routed_buffer_ptrs", "router_weight_buffer_ptrs", "d_router_weight_buffer_ptrs")
    inputs = None
    try:
        inputs = (
            *shared_inputs,
            workspace.x_buffer,
            workspace.d_y_buffer,
            workspace.d_x_routed_buffer,
            workspace.router_weight_buffer,
            workspace.d_router_weight_buffer,
            schedule.peer_rank,
            schedule.peer_token_idx,
            schedule.num_tokens,
            schedule.tokens_per_expert,
            *weights.values(),
            *context.values(),
        )
        signature = (
            "mxfp8-bwd",
            workspace.device,
            typed_key(
                workspace.ep_size,
                workspace.ep_rank,
                workspace.num_local_tokens,
                hidden_size,
                routed_hidden_size,
                workspace.topk,
                workspace.schedule_capacity,
                num_local_experts,
                intermediate_size,
                macrobatch_size,
                minibatch_size,
                num_comm_sms,
                native_columnwise,
                up_row_offset,
                routed_storage_tables is None,
                has_shared,
                *(getattr(workspace, name) for name in pointer_lists),
            ),
            tensor_layout_signature(inputs),
        )
        validated = _VALIDATED_CALLS.get(signature)  # hashes the signature
    except (AttributeError, TypeError):  # malformed inputs: the full checks raise the precise error
        signature = validated = None
    if validated is None:
        validate_persistent_mxfp8_bwd_call(
            workspace,
            schedule,
            context,
            weights,
            num_local_experts=num_local_experts,
            intermediate_size=intermediate_size,
            macrobatch_size=macrobatch_size,
            minibatch_size=minibatch_size,
            num_comm_sms=num_comm_sms,
            x_shared=x_shared,
            d_y_shared=d_y_shared,
            native_columnwise=native_columnwise,
            up_row_offset=up_row_offset,
            routed_storage_tables=routed_storage_tables,
            has_shared=has_shared,
        )
        if signature is not None:
            _VALIDATED_CALLS.put(signature, True)
    elif any(getattr(workspace, name)[workspace.ep_rank] != getattr(workspace, name[:-5]).data_ptr() for name in pointer_lists) or any(
        tensor.data_ptr() % 16 for tensor in inputs
    ):
        # Addresses are not part of the signature: the ownership and alignment checks stay per call.
        raise ValueError("the workspace pointer lists must own the local buffers and every input " "must be 16-byte aligned")
    device_index = workspace.device.index if workspace.device.index is not None else _current_device_index()
    if _current_device_index() != device_index:
        raise RuntimeError("the workspace CUDA device must be current")
    runtime_environment = _prepare_public_runtime_environment()
    main_grad_dtype = _main_grad_dtype(main_grads)
    if main_grad_storage_tables is not None and routed_storage_tables is None:
        raise ValueError("split main_grad buffers need split routed weights")
    if main_grad_storage_tables is not None and main_grad_dtype is None:
        raise ValueError("split main_grad buffers require fused accumulation")
    if main_grads is not None:
        validate_main_grads(
            main_grads,
            num_local_experts=num_local_experts,
            hidden_size=workspace.hidden_size,
            intermediate_size=intermediate_size,
            split=main_grad_storage_tables is not None,
            main_grad_up_row_offset=main_grad_up_row_offset,
            device=workspace.device,
            has_shared=has_shared,
        )
    state = prepare_persistent_mxfp8_bwd_state(
        num_local_tokens=workspace.num_local_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=num_local_experts,
        schedule_capacity=workspace.schedule_capacity,
        macrobatch_size=macrobatch_size,
        minibatch_size=minibatch_size,
        device=workspace.device,
        gate_up_context=gate_up_context,
        routed_hidden_size=routed_hidden_size,
        main_grads=main_grads,
        combined_fc1=bool(up_row_offset),
        has_shared=has_shared,
        caller_buffers=caller_buffers,
    )
    # Where Up's weight gradient starts in the Gate one's buffer: the caller's combined main_grad,
    # or this call's own combined dW.
    wgrad_up_row_offset = main_grad_up_row_offset if main_grads is not None else up_row_offset
    # Task -> expert decode variant (the BF16 forward's selector): compile-time, part of the executor
    # key; after the input check, so a malformed call reports its own error first.
    task_decode = select_task_decode(int(num_local_experts))
    has_tail = _has_tail(hidden_size, routed_hidden_size, intermediate_size)
    key = _executor_key(
        is_clamped=is_clamped,
        task_decode=task_decode,
        runtime_environment=runtime_environment,
        device_index=device_index,
        context_key=_current_cuda_context_key(),
        gate_up_context=gate_up_context,
        native_columnwise=native_columnwise,
        split_weights=routed_storage_tables is not None,
        main_grad_dtype=main_grad_dtype,
        split_main_grads=main_grad_storage_tables is not None,
        is_situ=situ is not None,
        has_shared=has_shared,
        has_tail=has_tail,
        recompute_forward_context=recompute_forward_context,
    )
    # The ten operands the dgrad reads, then -- when the buffers are split too -- the three
    # main_grad ones the Wgrad writes, in the Down, Gate, Up order the kernel expects.  Built once
    # per set of address tables and descriptor schema (the executor's atoms over these operands'
    # layouts).
    routed_rows = None if routed_storage_tables is None else tuple(routed_storage_tables)
    if main_grad_storage_tables is not None:
        routed_rows = (*routed_rows, main_grad_storage_tables[2], main_grad_storage_tables[0], main_grad_storage_tables[1])
    routed = routed_launch_tensors(
        routed_rows,
        int(num_local_experts),
        workspace.device,
        schema=(
            key,
            tensor_layout_signature(
                (
                    *(tensor for name, tensor in weights.items() if name.startswith("routed")),
                    *(() if main_grads is None else (buffer for buffer in main_grads[1::2] if buffer is not None)),
                )
            ),
        ),
    )
    call_kwargs = dict(
        workspace=workspace,
        schedule=schedule,
        context=context,
        weights=weights,
        state=state,
        peer_tables=workspace_peer_tables(workspace),
        num_local_tokens=workspace.num_local_tokens,
        schedule_capacity=workspace.schedule_capacity,
        intermediate_size=intermediate_size,
        num_local_experts=num_local_experts,
        topk=workspace.topk,
        macrobatch_size=macrobatch_size,
        minibatch_size=minibatch_size,
        num_comm_sms=num_comm_sms,
        swiglu_limit=swiglu_limit_value,
        stream=current_stream_handle(device_index),
        x_shared=x_shared,
        d_y_shared=d_y_shared,
        native_columnwise=native_columnwise,
        up_row_offset=up_row_offset,
        routed_storage_tables=routed.addresses,
        routed_descriptors=routed.descriptors,
        build_descriptors=routed.build,
        main_grad_up_row_offset=wgrad_up_row_offset,
        situ_beta=(0.0, 0.0) if situ is None else situ,
    )

    executor = _executor(
        key,
        device_index,
        call_kwargs,
        is_clamped=is_clamped,
        task_decode=task_decode,
        split_weights=routed_storage_tables is not None,
        main_grad_dtype=main_grad_dtype,
        split_main_grads=main_grad_storage_tables is not None,
        is_situ=situ is not None,
        has_shared=has_shared,
        has_tail=has_tail,
        recompute_forward_context=recompute_forward_context,
    )

    from ._persistent_mxfp8_bwd_mega import make_mega_args_mxfp8_bwd

    runtime_args = make_mega_args_mxfp8_bwd(launch=True, **call_kwargs)
    # Launch on the current stream (the one prepare_persistent_mxfp8_bwd_state zeroed the counters
    # on).  The TVM FFI executor is positional; the dict follows __call__'s parameter order (checked
    # in prepare_mega_mxfp8_bwd).  No record_stream: like PyTorch's own ops, inputs are used on the
    # current stream, and a caller that frees them from another stream synchronizes that stream itself.
    executor(*runtime_args.values())
    if fast_extra is not None:
        # Remember this call's launch for the next ones with the same workspace, weights and geometry.
        template = SlotTemplate.build(
            runtime_args.values(),
            _fast_bwd_sources(workspace, weights, context, main_grads, d_y_shared, fast_schedule, state, call_kwargs["stream"]),
            _TRANSFORMS,
            named={"build_descriptors": list(runtime_args).index("build_descriptors")},
        )
        if template is not None and x_shared is None:
            _FAST_PLANS.put(
                fast_sources,
                fast_extra,
                _MxBwdPlan(
                    executor=executor,
                    template=template,
                    layouts=tuple(tensor_layout(tensor) for tensor in fast_per_call),
                    num_local_experts=int(num_local_experts),
                    intermediate_size=int(intermediate_size),
                    routed=routed,
                ),
                fast_views,
            )
    routed.mark_built()
    return _bwd_outputs(state, has_shared)


def _executor(key, device_index, call_kwargs, *, main_grad_dtype, **variant):
    """The executor of ``key``, compiled on first use over these arguments (``prepare_mega_mxfp8_bwd``;
    ``native_columnwise`` travels inside ``call_kwargs``)."""
    import torch
    from ._persistent_mxfp8_bwd_mega import prepare_mega_mxfp8_bwd

    with _EXECUTOR_CACHE_LOCK:
        executor = _EXECUTOR_CACHE.get(key)
        if executor is None:
            executor = persistent_executor(
                process_independent(key),
                lambda: prepare_mega_mxfp8_bwd(
                    accumulate_wgrad=main_grad_dtype is not None, bf16_main_grad=main_grad_dtype is torch.bfloat16, **variant, **call_kwargs
                ),
            )
            _EXECUTOR_CACHE[key] = executor
            _EXECUTOR_CACHE_DEVICES.add(device_index)
    return executor


def backward_mxfp8_plan_sizes(
    *,
    num_local_tokens: int,
    hidden_size: int,
    routed_hidden_size: int | None,
    intermediate_size: int,
    num_local_experts: int,
    schedule_capacity: int,
    macrobatch_size: int,
    gate_up_context: str = "mxfp8",
    has_shared: bool = True,
) -> tuple[tuple, int]:
    """What a plan holds for MXFP8 backwards of up to these sizes: ``(scratch, counter_words)``.

    ``scratch``: ``(name, elements, dtype)`` of every buffer of ``mxfp8_bwd_buffer_specs`` but the weight
    gradients (the caller's), a routed-only layer's shared ones and, when the two hiddens differ, ``d_x_shared``
    (then the caller's output).  ``counter_words``: the length of the int32 counter buffer.  Every buffer grows
    with the token count and the macrobatch and the counters as the macrobatch and the minibatch shrink, so
    these are the sizes at the largest token count and macrobatch, and the counters' at 256-row batches."""
    import math

    routed_hidden = hidden_size if routed_hidden_size is None else routed_hidden_size
    excluded = (
        _WEIGHT_GRADIENTS | (frozenset() if has_shared else _SHARED_BUFFERS) | (frozenset(("d_x_shared",)) if routed_hidden != hidden_size else frozenset())
    )
    specs = mxfp8_bwd_buffer_specs(
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=num_local_experts,
        macrobatch_size=macrobatch_size,
        routed_hidden_size=routed_hidden,
        gate_up_context=gate_up_context,
    )
    lengths = mxfp8_bwd_counter_lengths(
        num_local_tokens=num_local_tokens, intermediate_size=intermediate_size, schedule_capacity=schedule_capacity, macrobatch_size=256, minibatch_size=256
    )
    return (tuple((name, math.prod(shape), _dtype(kind)) for name, shape, kind in specs if name not in excluded), _packed_offsets(lengths)[1])


def compile_backward_mxfp8(
    workspace,
    *,
    num_local_experts: int,
    intermediate_size: int,
    is_clamped: bool,
    is_situ: bool = False,
    has_shared: bool = True,
    split_weights: bool = False,
    main_grad_dtype=None,
    native_columnwise: bool = False,
    gate_up_context: str = "mxfp8",
) -> None:
    """The plan step: compile the backward megakernel variant a layer runs (with its tensormap prologue and the
    other launches of the same compiled call), without launching, and build the workspace's peer address tables.

    The variant is what the executor key holds: the SwiGLU clamp, SiTU-GLU, the shared expert, grouped or split
    routed weights, fused accumulation and its dtype (``main_grad_dtype``: None, FP32 or BF16; split with split
    weights), the columnwise layout of the transposed weights, the Gate / Up context precision, the task decode
    (from ``num_local_experts``) and the H / I tail.  Sizes are runtime values, so every tensor argument is a
    stand-in of the right rank and dtype over a small geometry (512 tokens, one expert, 256-wide H and I, one
    256-row macrobatch), allocated here and dropped.  A later ``backward_mxfp8`` over real tensors computes the
    same key and finds the executor.  The variant that replays the first macrobatch is not compiled here.
    """
    import torch
    from types import SimpleNamespace

    device = workspace.device
    check_device(device, "the MXFP8 backward")
    device_index = _current_device_index()
    if (device.index if device.index is not None else device_index) != device_index:
        raise RuntimeError("the workspace CUDA device must be current")
    if gate_up_context not in ("bf16", "mxfp8"):
        raise ValueError("gate_up_context must be 'bf16' or 'mxfp8'")
    runtime_environment = _prepare_public_runtime_environment()
    tokens, size = 512, 256

    def routed(dtype=torch.float8_e4m3fn):  # one expert: [N, K] split, [1, N, K] grouped
        return torch.empty((size, size) if split_weights else (1, size, size), dtype=dtype, device=device)

    scale = torch.empty((size // 128, size // 128, 32, 16), dtype=torch.uint8, device=device)
    placeholder = shared_placeholder(device)
    weights = dict(
        shared_gate=placeholder,
        shared_up=placeholder,
        shared_down=placeholder,
        routed_gate=routed(),
        routed_gate_sc=scale,
        routed_up=routed(),
        routed_up_sc=scale,
        routed_gate_t=routed(),
        routed_gate_t_sc=scale,
        routed_up_t=routed(),
        routed_up_t_sc=scale,
        routed_down_t=routed(),
        routed_down_t_sc=scale,
    )
    context = dict(
        gate_shared=placeholder,
        up_shared=placeholder,
        hidden_shared=placeholder,
        **prepare_persistent_mxfp8_bwd_routed_context(
            intermediate_size=size, macrobatch_size=size, routed_hidden_size=size, device=device, gate_up_context=gate_up_context
        ),
    )
    main_grads = None if main_grad_dtype is None else (None, routed(main_grad_dtype), None, routed(main_grad_dtype), None, routed(main_grad_dtype))
    state = prepare_persistent_mxfp8_bwd_state(
        num_local_tokens=tokens,
        hidden_size=size,
        intermediate_size=size,
        num_local_experts=1,
        schedule_capacity=size,
        macrobatch_size=size,
        minibatch_size=size,
        device=device,
        gate_up_context=gate_up_context,
        main_grads=main_grads,
        has_shared=False,
    )
    zero = torch.zeros(4, dtype=torch.int32, device=device)[:1]
    schedule = SimpleNamespace(peer_rank=zero, peer_token_idx=zero, num_tokens=zero, tokens_per_expert=zero)
    stand_ins = routed_launch_tensors(None, 1, device)
    hidden_size = workspace.hidden_size
    routed_hidden_size = getattr(workspace, "routed_hidden_size", None) or hidden_size
    task_decode = select_task_decode(int(num_local_experts))
    has_tail = _has_tail(hidden_size, routed_hidden_size, intermediate_size)
    split_main_grads = bool(split_weights) and main_grad_dtype is not None
    key = _executor_key(
        is_clamped=bool(is_clamped),
        task_decode=task_decode,
        runtime_environment=runtime_environment,
        device_index=device_index,
        context_key=_current_cuda_context_key(),
        gate_up_context=gate_up_context,
        native_columnwise=bool(native_columnwise),
        split_weights=bool(split_weights),
        main_grad_dtype=main_grad_dtype,
        split_main_grads=split_main_grads,
        is_situ=bool(is_situ),
        has_shared=has_shared,
        has_tail=has_tail,
        recompute_forward_context=False,
    )
    call_kwargs = dict(
        workspace=workspace,
        schedule=schedule,
        context=context,
        weights=weights,
        state=state,
        peer_tables=workspace_peer_tables(workspace),
        num_local_tokens=tokens,
        schedule_capacity=size,
        intermediate_size=size,
        num_local_experts=1,
        topk=workspace.topk,
        macrobatch_size=size,
        minibatch_size=size,
        num_comm_sms=2,
        swiglu_limit=0.0,
        stream=current_stream_handle(device_index),
        x_shared=None,
        d_y_shared=None,
        native_columnwise=bool(native_columnwise),
        up_row_offset=0,
        routed_storage_tables=stand_ins.addresses,
        routed_descriptors=stand_ins.descriptors,
        build_descriptors=stand_ins.build,
        main_grad_up_row_offset=0,
        situ_beta=(1.0, 1.0) if is_situ else (0.0, 0.0),
    )
    _executor(
        key,
        device_index,
        call_kwargs,
        is_clamped=bool(is_clamped),
        task_decode=task_decode,
        split_weights=bool(split_weights),
        main_grad_dtype=main_grad_dtype,
        split_main_grads=split_main_grads,
        is_situ=bool(is_situ),
        has_shared=has_shared,
        has_tail=has_tail,
        recompute_forward_context=False,
    )


__all__ = [
    "CONTEXT_NAMES",
    "CONTEXT_NAMES_E4M3_CONTEXT",
    "ROUTED_CONTEXT_NAMES",
    "SHARED_CONTEXT_NAMES",
    "PersistentMxfp8BwdState",
    "PersistentMxfp8BwdStateE4m3Context",
    "backward_mxfp8",
    "backward_mxfp8_plan_sizes",
    "clear_persistent_mxfp8_bwd_executor_cache",
    "compile_backward_mxfp8",
    "mxfp8_bwd_buffer_specs",
    "mxfp8_bwd_counter_lengths",
    "mxfp8_bwd_routed_context_specs",
    "prepare_persistent_mxfp8_bwd_routed_context",
    "prepare_persistent_mxfp8_bwd_state",
    "validate_persistent_mxfp8_bwd_call",
]
