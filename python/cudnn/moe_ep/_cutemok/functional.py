# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""The layer CuteMoK's operator API (``cudnn.moe_ep.cutemok``) calls: one expert-parallel MoE layer, forward and backward.
An implementation module, not part of the public API.

Call order: ``get_workspace`` (the symmetric-memory buffers of one EP group and one set of shapes, cached),
``build_schedule`` (once per routing: all-gathers the top-k expert ids and builds this rank's padded expert
schedule), ``forward`` (returns the output and a ``CuteMoKForwardContext``), then ``backward`` with that context.
``forward_latent_layer`` / ``backward_latent_layer`` wrap the two for a layer whose routed experts work in a
different hidden size than the shared expert.

``build_schedule``, ``forward`` and ``backward`` each run device-side barriers across the EP group
(``barrier_all``).  Token counts, hidden sizes, the expert count and top-k are runtime values of the kernels.  The
kernel modules (and with them CUTLASS) are imported on first use, not at ``import cudnn.moe_ep._cutemok``.
"""

import collections
import math
from dataclasses import dataclass, replace as _replace_fields
import sys
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True, slots=True)
class CuteMoKConfig:
    """CuteMoK runtime configuration, named after Mixture-of-Kittens' ``MoKConfig``: the same six fields and defaults,
    plus per-direction minibatch sizes and the options below.

    ``fwd_num_comm_sms`` / ``bwd_num_comm_sms``: the number of SMs the forward / backward megakernel
    gives to communication (even, and fewer than the device's SM count).

    ``bwd_recompute_num_comm_sms``: ``bwd_num_comm_sms`` of the backward that replays the first macrobatch
    too (a forward context without routed activations, or None); None means ``bwd_num_comm_sms``.  That
    backward pulls every macrobatch's x again on its communication SMs, and with MXFP8 routed weights
    quantizes it there, so it can want more of them than the backward on a saved context.

    ``minibatch_size`` serves both directions; ``fwd_minibatch_size`` / ``bwd_minibatch_size``
    override it per direction (each positive and divisible by 256).
    When both per-direction sizes are set, ``minibatch_size`` is unused.
    Only ``macrobatch_size`` (a multiple of every minibatch size) is genuinely shared: the forward
    context the backward reads is laid out in macrobatch rows, while a minibatch only orders the tasks
    and sizes each direction's own readiness counters.

    ``schedule_capacity_multiplier``: the schedule holds
    ``num_local_tokens * topk * max(2, ceil(ep_size * multiplier))`` routed rows.

    ``all_gather_top_experts_chunk_bytes``: a positive multiple of 16 that divides one rank's route
    buffer (``num_local_tokens * topk * 4`` bytes).

    The field names are a public contract and the dataclass is strict: an unknown keyword raises
    ``TypeError``, which a caller that builds the config from user settings can report as a
    configuration error.

    ``gate_up_context`` (MXFP8 routed weights only): the precision of the routed Gate / Up context the
    forward keeps for the backward -- ``"mxfp8"`` (default; e4m3 / scale pairs) or ``"bf16"`` (the
    epilogue's BF16 values: the backward's SwiGLU reads them once and skips a quantize / dequantize
    pair, at twice the context bytes).  A compile-time choice of the forward; the backward follows the
    context it is given.

    ``epilogue_combine`` (BF16 forward only, a compiled variant): the routed Down epilogue stores each
    finished row into the source rank's combine buffer and the forward's communication CTAs only pull
    X; ``False`` (default) lets the communication CTAs push the combine from the Y ring.  Same
    outputs; tune it per model.

    ``comm_tile_shape`` (BF16 forward and backward only, a compiled variant): rows x columns of one
    communication tile, ``(32, 512)`` (default) or ``(16, 1024)`` (a tuple or the contract's
    ``CommTileShape``), or ``"auto"``: 16 x 1024 when the routed hidden size is a multiple of 1024,
    otherwise 32 x 512 (chosen from the hidden size alone, a model parameter).  Each row of a tile is one
    TMA bulk copy of ``columns * 2`` bytes: 1 KiB at 32 x 512, 2 KiB at 16 x 1024, the same 32 KiB slot.
    It changes only how the communication CTAs move bytes; the results are bit-identical.  Part of the
    executor cache key; MXFP8 routed weights (their stream has its own fixed tile) take 32 x 512 under
    ``"auto"`` and refuse any other value.
    """

    fwd_num_comm_sms: int = 40
    bwd_num_comm_sms: int = 28
    minibatch_size: int = 4096
    macrobatch_size: int = 131072
    schedule_capacity_multiplier: float = 0.5
    all_gather_top_experts_chunk_bytes: int = 2048
    fwd_minibatch_size: int | None = None
    bwd_minibatch_size: int | None = None
    gate_up_context: str = "mxfp8"
    epilogue_combine: bool = False
    bwd_recompute_num_comm_sms: int | None = None
    comm_tile_shape: tuple[int, int] | str = (32, 512)

    def __post_init__(self) -> None:
        # A config built from JSON passes a list: keep the frozen config hashable.
        if isinstance(self.comm_tile_shape, list):
            object.__setattr__(self, "comm_tile_shape", tuple(self.comm_tile_shape))


#: The Mixture-of-Kittens name, kept as an alias for callers written against it.
MoKConfig = CuteMoKConfig

_CONFIG_TYPES = (CuteMoKConfig,)


def _fwd_minibatch_size(config: CuteMoKConfig) -> int:
    return config.minibatch_size if config.fwd_minibatch_size is None else config.fwd_minibatch_size


def _bwd_minibatch_size(config: CuteMoKConfig) -> int:
    return config.minibatch_size if config.bwd_minibatch_size is None else config.bwd_minibatch_size


def _bwd_num_comm_sms(config: CuteMoKConfig, recompute: bool) -> int:
    """The backward's communication SMs; ``recompute``: the launch replays the first macrobatch too."""
    if recompute and config.bwd_recompute_num_comm_sms is not None:
        return config.bwd_recompute_num_comm_sms
    return config.bwd_num_comm_sms


def _comm_tile_shape(workspace, config: CuteMoKConfig):
    """``config.comm_tile_shape`` as the contract's CommTileShape (ValueError for any other value)."""
    from .cutedsl._host_common import comm_tile_shape

    return comm_tile_shape(workspace.routed_hidden_size, config.comm_tile_shape)


def _require_default_comm_tile_shape(workspace, config: CuteMoKConfig) -> None:
    """MXFP8 routed weights: their communication stream has its own fixed tile shape ("auto" takes it)."""
    from .cutedsl._host_common import comm_tile_shape

    if config.comm_tile_shape == "auto":
        return
    if _comm_tile_shape(workspace, config) != comm_tile_shape(workspace.routed_hidden_size):
        raise ValueError("comm_tile_shape is an option of the BF16 megakernels; MXFP8 routed weights " "take the default (32, 512)")


def _routed_hidden_differs(workspace) -> bool:
    """The routed experts work in a different hidden (routed_hidden_size) than the shared expert (hidden_size)."""
    hidden = getattr(workspace, "hidden_size", None)
    return hidden is not None and getattr(workspace, "routed_hidden_size", hidden) != hidden


def _validate_config_type(config: CuteMoKConfig) -> None:
    """The first config check of every entry point."""
    if not isinstance(config, _CONFIG_TYPES):
        raise TypeError("config must be a CuteMoKConfig")


_CUTEDSL_OPS = None


def _device_ops(config: CuteMoKConfig):
    """The small operators around the megakernels, compiled once per process."""
    _validate_config_type(config)
    global _CUTEDSL_OPS
    if _CUTEDSL_OPS is None:
        # CUTLASS and the helpers adapted from QuACK are imported on first use, not at `import cudnn.moe_ep._cutemok`.
        from .cutedsl.all_gather_top_experts import all_gather_top_experts as cutedsl_all_gather
        from .cutedsl.barrier_all import barrier_all as cutedsl_barrier_all
        from .cutedsl.bwd_epilogue import bwd_epilogue_bf16
        from .cutedsl.fwd_epilogue import fwd_epilogue_bf16
        from .cutedsl.scheduler import scheduler as cutedsl_schedule

        class _CuTeOps:
            all_gather_top_experts = staticmethod(cutedsl_all_gather)
            barrier_all = staticmethod(cutedsl_barrier_all)
            schedule = staticmethod(cutedsl_schedule)
            fwd_epilogue = staticmethod(fwd_epilogue_bf16)
            bwd_epilogue = staticmethod(bwd_epilogue_bf16)

        _CUTEDSL_OPS = _CuTeOps
    return _CUTEDSL_OPS


def _use_minibatch_bwd_schedule(config: CuteMoKConfig, routed_mxfp8: bool = False) -> bool:
    """Whether the backward retires ring slots per minibatch.

    The BF16 backward does; the MXFP8 backward retires per macrobatch.
    """
    _validate_config_type(config)
    return not routed_mxfp8


@dataclass(frozen=True, slots=True)
class CuteMoKSchedule:
    peer_rank: torch.Tensor  # (schedule_capacity,) int32
    peer_token_idx: torch.Tensor  # (schedule_capacity,) int32
    num_tokens: torch.Tensor  # (1,) int32
    tokens_per_expert: torch.Tensor  # (num_local_experts,) int32, padded to 256 rows per expert
    # (num_local_experts,) int32: each local expert's unpadded routed rows.
    valid_tokens_per_expert: torch.Tensor | None = None
    # (num_local_tokens, topk) int32: the routing this schedule was built from, -1 where a token uses
    # fewer than topk experts.  The epilogues skip those slots: their rows of the combine and
    # d_x_routed buffers are never written, and the buffers are allocated rather than zeroed.
    top_experts: torch.Tensor | None = None


#: The Mixture-of-Kittens name, kept as an alias.
MoKSchedule = CuteMoKSchedule


@dataclass(frozen=True, slots=True)
class CuteMoKForwardContext:
    # The four routed entries hold the first macrobatch's activations, or are all None: ``backward`` then
    # replays the first macrobatch like the later ones (``without_routed``, ``recompute_forward_context``).
    x_routed: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None
    gate_shared: torch.Tensor
    gate_routed: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None
    up_shared: torch.Tensor
    up_routed: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None
    hidden_shared: torch.Tensor
    hidden_routed: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None
    # The shared output gate's saved BF16 S and G, per-forward tensors (never workspace views);
    # an ungated forward keeps neither.
    shared_output: torch.Tensor | None = None
    shared_output_gate: torch.Tensor | None = None
    # SiTU-GLU's (beta1, beta2) as FP32 values when the forward used it; backward uses the same.
    situ_beta: tuple[float, float] | None = None

    def without_routed(self) -> "CuteMoKForwardContext":
        """This context without the routed activations (macrobatch-sized; the shared ones are kept).
        ``backward`` recomputes them in its kernel, so they need not be held until the backward."""
        return _replace_fields(self, x_routed=None, gate_routed=None, up_routed=None, hidden_routed=None)


#: The Mixture-of-Kittens name, kept as an alias.
MoKForwardContext = CuteMoKForwardContext


@dataclass
class CuteMoKLatentForwardContext:
    """forward_latent_layer state for backward_latent_layer."""

    mok: CuteMoKForwardContext
    routed_input: torch.Tensor  # [num_local_tokens, routed_hidden_size]  x @ latent_down_weights^T
    y_routed: torch.Tensor  # [num_local_tokens, routed_hidden_size]  routed top-k reduced output


#: The earlier name, kept as an alias.
MoKLatentForwardContext = CuteMoKLatentForwardContext


@dataclass(slots=True)
class CuteMoKWorkspace:
    group_name: str  # () str
    ep_rank: int  # () int
    ep_size: int  # () int
    device: torch.device  # () torch.device
    num_local_tokens: int  # () int
    hidden_size: int  # () int  shared-expert hidden
    topk: int  # () int
    schedule_capacity: int  # () int
    # (num_local_tokens, routed_hidden_size) bfloat16 (routed source rows)
    x_buffer: torch.Tensor
    x_buffer_handle: Any  # () SymmetricMemory handle
    x_buffer_ptrs: list[int]  # (ep_size,) uintptr64
    combine_buffer: torch.Tensor  # (num_local_tokens * topk, routed_hidden_size) bfloat16
    combine_buffer_handle: Any  # () SymmetricMemory handle
    combine_buffer_ptrs: list[int]  # (ep_size,) uintptr64
    d_y_buffer: torch.Tensor  # (num_local_tokens, routed_hidden_size) bfloat16
    d_y_buffer_handle: Any  # () SymmetricMemory handle
    d_y_buffer_ptrs: list[int]  # (ep_size,) uintptr64
    d_x_routed_buffer: torch.Tensor  # (num_local_tokens * topk, routed_hidden_size) bfloat16
    d_x_routed_buffer_handle: Any  # () SymmetricMemory handle
    d_x_routed_buffer_ptrs: list[int]  # (ep_size,) uintptr64
    router_weight_buffer: torch.Tensor  # (num_local_tokens, topk) float32
    router_weight_buffer_handle: Any  # () SymmetricMemory handle
    router_weight_buffer_ptrs: list[int]  # (ep_size,) uintptr64
    d_router_weight_buffer: torch.Tensor  # (num_local_tokens, topk) float32
    d_router_weight_buffer_handle: Any  # () SymmetricMemory handle
    d_router_weight_buffer_ptrs: list[int]  # (ep_size,) uintptr64
    all_gather_top_experts_buffer: torch.Tensor  # (ep_size, num_local_tokens, topk) int32
    all_gather_top_experts_buffer_handle: Any  # () SymmetricMemory handle
    all_gather_top_experts_buffer_multicast_ptr: int  # () uintptr64
    barrier_buffer: torch.Tensor  # (1,) int32
    barrier_buffer_handle: Any  # () SymmetricMemory handle
    barrier_buffer_ptrs: list[int]  # (ep_size,) uintptr64
    barrier_buffer_multicast_ptr: int  # () uintptr64
    barrier_target: torch.Tensor  # (1,) int32
    _cutedsl_bwd_preparation: Any = None  # the BF16 backward buffer set last used
    _cutedsl_bwd_preparations: Any = None  # every prepared BF16 backward configuration's buffer set
    _cutedsl_peer_tables: Any = None  # CuTe DSL peer address tables of this workspace's symmetric buffers
    # Hidden of the routed chain (x_buffer / combine_buffer / dY / dX rows); equals
    # hidden_size unless create_workspace(routed_hidden_size=...) was given.  None = "same as hidden_size".
    routed_hidden_size: int | None = None

    def __post_init__(self) -> None:
        if self.routed_hidden_size is None:
            self.routed_hidden_size = self.hidden_size


#: The Mixture-of-Kittens name, kept as an alias.
MoKWorkspace = CuteMoKWorkspace


_WORKSPACE_CACHE: dict[tuple[str, int, int, int, int, int, int], CuteMoKWorkspace] = {}
#: get_workspace's arguments -> the workspace they validated to (see get_workspace).
_WORKSPACE_LOOKUP: dict[tuple, CuteMoKWorkspace] = {}


def validate_workspace_args(
    config: CuteMoKConfig,
    group: dist.ProcessGroup,
    *,
    device: torch.device,
    num_local_tokens: int,
    hidden_size: int,
    topk: int,
    routed_hidden_size: int | None = None,
) -> None:
    """Validates the arguments used to create or retrieve a workspace.

    Inputs:
        config:           CuteMoKConfig
        group:            torch.distributed.ProcessGroup
        device:           torch.device
        num_local_tokens: int
        hidden_size:      int
        topk:             int
        routed_hidden_size:      int | None  (routed-expert hidden; None = hidden_size)

    Outputs:
        None
    """
    _validate_config_type(config)
    if (
        type(config.schedule_capacity_multiplier) not in (int, float)
        or not math.isfinite(config.schedule_capacity_multiplier)
        or config.schedule_capacity_multiplier <= 0
    ):
        raise ValueError("schedule_capacity_multiplier must be a positive finite number")
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    if not isinstance(group, dist.ProcessGroup):
        raise TypeError("group must be a torch.distributed.ProcessGroup")
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")
    if device.type != "cuda":
        raise ValueError("device must be a CUDA device")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    if device_index != torch.cuda.current_device():
        raise ValueError("CuteMoK workspace device must be the current CUDA device")
    if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3)):
        raise NotImplementedError("CuteMoK currently requires an SM100 or SM103 GPU")
    device_properties = torch.cuda.get_device_properties(device)
    if type(num_local_tokens) is not int or num_local_tokens < 512:
        raise ValueError("num_local_tokens must be an integer at least 512")
    if num_local_tokens % 256 != 0:
        raise ValueError("num_local_tokens must be divisible by 256")
    if type(hidden_size) is not int or hidden_size <= 0:
        raise ValueError("hidden_size must be a positive integer")
    if hidden_size < 256 or hidden_size % 128 != 0:
        raise ValueError("hidden_size must be at least 256 and divisible by 128")
    if routed_hidden_size is not None and (type(routed_hidden_size) is not int or routed_hidden_size <= 0 or routed_hidden_size % 256 != 0):
        raise ValueError("routed_hidden_size must be a positive integer divisible by 256")
    if type(topk) is not int or not 0 < topk <= 255:
        raise ValueError("topk must be an integer in [1, 255]")
    # Two ring slots, each holding topk+1 aligned 2 KiB vectors, the FP32 route weights and the
    # INT32 route ids, plus the driver's reserved 1 KiB.
    fwd_epilogue_smem_bytes = 2 * ((topk + 1) * 2048 + topk * 4 + topk * 4) + 1024
    if fwd_epilogue_smem_bytes > device_properties.shared_memory_per_block_optin:
        raise ValueError("topk requires more dynamic shared memory than the device supports")

    group_name = group.group_name
    if not isinstance(group_name, str) or not group_name:
        raise RuntimeError("process group must have a nonempty group_name")
    ep_rank = dist.get_rank(group=group)
    ep_size = dist.get_world_size(group=group)
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("CuteMoK EP size must be one of 1, 4, 8, 16, 32, 64")
    if not 0 <= ep_rank < ep_size:
        raise RuntimeError("current process is not a member of the EP process group")


def _torch_distributed_features(ep_size: int):
    """The PyTorch features a workspace uses, checked on first use rather than by a version number: the
    stream-ordered ``Work.block_current_stream`` and, for more than one rank, symmetric memory
    (``torch.distributed._symmetric_memory``: ``enable_symm_mem_for_group``, ``empty``, ``rendezvous``, and the
    handles' ``buffer_ptrs`` and ``multicast_ptr``).  Returns the symmetric-memory module (None for one rank);
    raises naming what this PyTorch lacks."""
    missing = []
    c10d = getattr(torch._C, "_distributed_c10d", None)
    if not hasattr(getattr(c10d, "Work", None), "block_current_stream"):
        missing.append("torch.distributed Work.block_current_stream")
    symm_mem = None
    if ep_size > 1:
        try:
            import torch.distributed._symmetric_memory as symm_mem
        except ImportError:
            missing.append("torch.distributed._symmetric_memory")
        else:
            missing += [
                f"torch.distributed._symmetric_memory.{name}" for name in ("enable_symm_mem_for_group", "empty", "rendezvous") if not hasattr(symm_mem, name)
            ]
            handle = getattr(c10d, "_SymmetricMemory", None)
            missing += [f"symmetric-memory handle attribute {name}" for name in ("buffer_ptrs", "multicast_ptr") if not hasattr(handle, name)]
    if missing:
        raise RuntimeError(
            f"CuteMoK needs PyTorch features that torch {torch.__version__} lacks: {', '.join(missing)}; "
            "install a PyTorch release that has them (2.9 or later, built for CUDA 13)"
        )
    return symm_mem


def create_workspace(
    config: CuteMoKConfig,
    group: dist.ProcessGroup,
    *,
    device: torch.device,
    num_local_tokens: int,
    hidden_size: int,
    topk: int,
    routed_hidden_size: int | None = None,
) -> CuteMoKWorkspace:
    """Creates a new caller-owned workspace.

    ``routed_hidden_size`` is the hidden of the routed chain: the symmetric x/combine/dY/dX
    buffers carry routed-hidden rows while ``hidden_size`` stays the shared-expert hidden.  ``None`` keeps
    both equal.

    Inputs:
        config:           CuteMoKConfig
        group:            torch.distributed.ProcessGroup
        device:           torch.device
        num_local_tokens: int
        hidden_size:      int
        topk:             int
        routed_hidden_size:      int | None  (routed-expert hidden; None = hidden_size)

    Outputs:
        workspace: CuteMoKWorkspace
    """
    validate_workspace_args(
        config,
        group,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
        routed_hidden_size=routed_hidden_size,
    )
    routed_hidden = hidden_size if routed_hidden_size is None else routed_hidden_size

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    group_name = group.group_name
    ep_rank = dist.get_rank(group=group)
    ep_size = dist.get_world_size(group=group)
    symm_mem = _torch_distributed_features(ep_size)
    schedule_capacity_factor = max(2, math.ceil(ep_size * config.schedule_capacity_multiplier))

    local_shape = torch.tensor([num_local_tokens, hidden_size, topk, routed_hidden], dtype=torch.int64, device=device)
    gathered_shapes = torch.empty(ep_size * local_shape.numel(), dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(gathered_shapes, local_shape, group=group)
    gathered_shapes = gathered_shapes.view(ep_size, local_shape.numel())
    if not torch.all(gathered_shapes == local_shape).item():  # .item() here is fine since this is one-time setup
        raise ValueError("CuteMoK requires identical token, hidden, routed hidden, and top-k shapes on every EP rank")
    if ep_size > 1:
        symm_mem.enable_symm_mem_for_group(group_name)

    schedule_capacity = num_local_tokens * topk * schedule_capacity_factor

    def allocate_buffer(*shape: int, dtype: torch.dtype, zero: bool = False) -> tuple[torch.Tensor, Any, list[int]]:
        if ep_size == 1:
            if zero:
                buffer = torch.zeros(*shape, dtype=dtype, device=device)
            else:
                buffer = torch.empty(*shape, dtype=dtype, device=device)
            handle = None
            ptrs = [int(buffer.data_ptr())]
        else:
            buffer = symm_mem.empty(*shape, dtype=dtype, device=device)
            if zero:
                buffer.zero_()
            handle = symm_mem.rendezvous(buffer, group_name)
            ptrs = [int(handle.buffer_ptrs[peer_rank]) for peer_rank in range(ep_size)]
        return buffer, handle, ptrs

    # routed-chain buffers carry routed-hidden rows (== hidden rows for an ordinary MoE)
    x_buffer, x_buffer_handle, x_buffer_ptrs = allocate_buffer(num_local_tokens, routed_hidden, dtype=torch.bfloat16)
    combine_buffer, combine_buffer_handle, combine_buffer_ptrs = allocate_buffer(num_local_tokens * topk, routed_hidden, dtype=torch.bfloat16)
    d_y_buffer, d_y_buffer_handle, d_y_buffer_ptrs = allocate_buffer(num_local_tokens, routed_hidden, dtype=torch.bfloat16)
    d_x_routed_buffer, d_x_routed_buffer_handle, d_x_routed_buffer_ptrs = allocate_buffer(num_local_tokens * topk, routed_hidden, dtype=torch.bfloat16)
    router_weight_buffer, router_weight_buffer_handle, router_weight_buffer_ptrs = allocate_buffer(num_local_tokens, topk, dtype=torch.float32)
    # Zeroed once: the backward kernels write only routed slots, so with zero_unrouted_router_grads=False an
    # unrouted slot reads back zero or a gradient an earlier call wrote there, never uninitialized memory.
    d_router_weight_buffer, d_router_weight_buffer_handle, d_router_weight_buffer_ptrs = allocate_buffer(num_local_tokens, topk, dtype=torch.float32, zero=True)

    all_gather_top_experts_buffer, all_gather_top_experts_buffer_handle, _ = allocate_buffer(ep_size, num_local_tokens, topk, dtype=torch.int32)
    all_gather_top_experts_buffer_multicast_ptr = int(
        all_gather_top_experts_buffer.data_ptr() if all_gather_top_experts_buffer_handle is None else all_gather_top_experts_buffer_handle.multicast_ptr
    )

    barrier_buffer, barrier_buffer_handle, barrier_buffer_ptrs = allocate_buffer(1, dtype=torch.int32, zero=True)
    barrier_buffer_multicast_ptr = int(barrier_buffer.data_ptr() if barrier_buffer_handle is None else barrier_buffer_handle.multicast_ptr)
    # The routing all-gather and the barriers write through these two multicast addresses (the other symmetric
    # buffers are reached through their peer addresses only and need none).
    if ep_size > 1 and not (all_gather_top_experts_buffer_multicast_ptr and barrier_buffer_multicast_ptr):
        raise RuntimeError(
            "CuteMoK needs a multicast address of PyTorch symmetric memory for its routing all-gather and barrier "
            "buffers in an EP group of more than one rank; this group's GPUs or driver do not provide one "
            "(NVLink multicast)"
        )
    barrier_target = torch.zeros(1, dtype=torch.int32, device=device)

    dist.barrier(group=group, async_op=True, device_ids=[device_index]).block_current_stream()

    workspace = CuteMoKWorkspace(
        group_name=group_name,
        ep_rank=ep_rank,
        ep_size=ep_size,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
        routed_hidden_size=routed_hidden,
        schedule_capacity=schedule_capacity,
        x_buffer=x_buffer,
        x_buffer_handle=x_buffer_handle,
        x_buffer_ptrs=x_buffer_ptrs,
        combine_buffer=combine_buffer,
        combine_buffer_handle=combine_buffer_handle,
        combine_buffer_ptrs=combine_buffer_ptrs,
        d_y_buffer=d_y_buffer,
        d_y_buffer_handle=d_y_buffer_handle,
        d_y_buffer_ptrs=d_y_buffer_ptrs,
        d_x_routed_buffer=d_x_routed_buffer,
        d_x_routed_buffer_handle=d_x_routed_buffer_handle,
        d_x_routed_buffer_ptrs=d_x_routed_buffer_ptrs,
        router_weight_buffer=router_weight_buffer,
        router_weight_buffer_handle=router_weight_buffer_handle,
        router_weight_buffer_ptrs=router_weight_buffer_ptrs,
        d_router_weight_buffer=d_router_weight_buffer,
        d_router_weight_buffer_handle=d_router_weight_buffer_handle,
        d_router_weight_buffer_ptrs=d_router_weight_buffer_ptrs,
        all_gather_top_experts_buffer=all_gather_top_experts_buffer,
        all_gather_top_experts_buffer_handle=all_gather_top_experts_buffer_handle,
        all_gather_top_experts_buffer_multicast_ptr=all_gather_top_experts_buffer_multicast_ptr,
        barrier_buffer=barrier_buffer,
        barrier_buffer_handle=barrier_buffer_handle,
        barrier_buffer_ptrs=barrier_buffer_ptrs,
        barrier_buffer_multicast_ptr=barrier_buffer_multicast_ptr,
        barrier_target=barrier_target,
    )
    # The megakernels' peer address tables: built here, never by a forward or backward (a graph capture).
    from .cutedsl._host_common import build_workspace_peer_tables

    build_workspace_peer_tables(workspace)
    return workspace


def get_workspace(
    config: CuteMoKConfig,
    group: dist.ProcessGroup,
    *,
    device: torch.device,
    num_local_tokens: int,
    hidden_size: int,
    topk: int,
    routed_hidden_size: int | None = None,
) -> CuteMoKWorkspace:
    """Returns a cached workspace, creating and caching one if absent.

    Inputs:
        config:           CuteMoKConfig
        group:            torch.distributed.ProcessGroup
        device:           torch.device
        num_local_tokens: int
        hidden_size:      int
        topk:             int
        routed_hidden_size:      int | None  (routed-expert hidden; None = hidden_size)

    Outputs:
        workspace: CuteMoKWorkspace
    """
    # A repeated set of arguments skips validate_workspace_args: a caller may ask for the workspace on
    # every layer call.  The lookup is keyed by the arguments themselves (CuteMoKConfig is frozen), and only
    # remembers ones that validated; a destroyed group is caught below.
    lookup_key = None
    if (
        type(num_local_tokens) is int
        and type(hidden_size) is int
        and type(topk) is int
        and (routed_hidden_size is None or type(routed_hidden_size) is int)
        and isinstance(config, CuteMoKConfig)
        and isinstance(device, torch.device)
        and not torch.compiler.is_compiling()
    ):
        lookup_key = (config, group, device, torch.cuda.current_device(), num_local_tokens, hidden_size, topk, routed_hidden_size)
        workspace = _WORKSPACE_LOOKUP.get(lookup_key)
        if workspace is not None:
            if _group_is_registered(group):
                return workspace
            # The group was destroyed (or torch.distributed shut down) since: forget the entry, which
            # also releases the group, and validate, which raises for such a group.
            _WORKSPACE_LOOKUP.pop(lookup_key, None)

    validate_workspace_args(
        config,
        group,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
        routed_hidden_size=routed_hidden_size,
    )

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    group_name = group.group_name
    ep_size = dist.get_world_size(group=group)
    schedule_capacity_factor = max(2, math.ceil(ep_size * config.schedule_capacity_multiplier))

    cache_key = (
        group_name,
        device_index,
        num_local_tokens,
        hidden_size,
        topk,
        schedule_capacity_factor,
        hidden_size if routed_hidden_size is None else routed_hidden_size,
    )
    cached_workspace = _WORKSPACE_CACHE.get(cache_key)
    if cached_workspace is not None:
        if lookup_key is not None:
            _WORKSPACE_LOOKUP[lookup_key] = cached_workspace
        return cached_workspace

    workspace = create_workspace(
        config,
        group,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
        routed_hidden_size=routed_hidden_size,
    )
    _WORKSPACE_CACHE[cache_key] = workspace
    if lookup_key is not None:
        _WORKSPACE_LOOKUP[lookup_key] = workspace
    return workspace


def _group_is_registered(group: dist.ProcessGroup) -> bool:
    """Whether torch.distributed is initialized and ``group`` is still one of its process groups (not destroyed).

    get_workspace checks this on a lookup hit instead of rerunning validate_workspace_args, whose process-group
    queries raise for a destroyed group.  It reads c10d's group registry, the map destroy_process_group removes a
    group from; a torch without that private registry falls back to the public dist.get_rank, which raises for such
    a group.
    """
    if not dist.is_initialized():
        return False
    try:
        registered_groups = dist.distributed_c10d._world.pg_map
    except AttributeError:
        try:
            dist.get_rank(group=group)
        except (RuntimeError, ValueError):
            return False
        return True
    return group in registered_groups


def clear_workspace_cache() -> None:
    """Clears cached workspaces after all participating ranks synchronize.

    Inputs:
        None

    Outputs:
        None
    """
    from .cutedsl.barrier_all import barrier_all

    for workspace in _WORKSPACE_CACHE.values():
        barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
        torch.cuda.synchronize(workspace.device)
        workspace._cutedsl_bwd_preparation = None
        workspace._cutedsl_bwd_preparations = None
        workspace._cutedsl_peer_tables = None
    # Release the executor-only caches of whichever kernels this process compiled, after a
    # conservative synchronization of the devices they compiled for.  A module that was never
    # imported has no cache to clear.
    for module_name, clear_name in (
        ("cudnn.moe_ep._cutemok.cutedsl.persistent_bf16_fwd", "clear_persistent_bf16_executor_cache"),
        ("cudnn.moe_ep._cutemok.cutedsl.persistent_mxfp8_fwd", "clear_persistent_mxfp8_executor_cache"),
        ("cudnn.moe_ep._cutemok.cutedsl.mxfp8_quantize", "clear_mxfp8_quantize_executor_cache"),
        ("cudnn.moe_ep._cutemok.cutedsl.persistent_mxfp8_bwd", "clear_persistent_mxfp8_bwd_executor_cache"),
        ("cudnn.moe_ep._cutemok.cutedsl.persistent_bf16_bwd", "clear_persistent_bf16_bwd_executor_cache"),
        ("cudnn.moe_ep._cutemok.cutedsl.fwd_epilogue", "clear_fwd_epilogue_executor_cache"),
        ("cudnn.moe_ep._cutemok.cutedsl.bwd_epilogue", "clear_bwd_epilogue_executor_cache"),
    ):
        persistent_bf16 = sys.modules.get(module_name)
        if persistent_bf16 is None:
            continue
        clear_executor_cache = getattr(
            persistent_bf16,
            clear_name,
            None,
        )
        if clear_executor_cache is not None:
            # Executors are reused across workspace lifetimes; synchronizing
            # here keeps cache destruction conservative.
            clear_executor_cache(synchronize=True)
    _WORKSPACE_CACHE.clear()
    _WORKSPACE_LOOKUP.clear()
    _VALIDATED_SCHEDULE_CONFIGS.clear()


_VALIDATED_SCHEDULE_CONFIGS: set = set()


def _validate_schedule_config(workspace: "CuteMoKWorkspace", config: "CuteMoKConfig") -> None:
    """The scheduling and communication options of ``config`` for ``workspace``'s device and shapes."""
    device_properties = torch.cuda.get_device_properties(workspace.device)
    if type(config.fwd_num_comm_sms) is not int or config.fwd_num_comm_sms <= 0:
        raise ValueError("fwd_num_comm_sms must be a positive integer")
    if config.fwd_num_comm_sms % 2 != 0:
        raise ValueError("fwd_num_comm_sms must be even")
    if type(config.bwd_num_comm_sms) is not int or config.bwd_num_comm_sms <= 0:
        raise ValueError("bwd_num_comm_sms must be a positive integer")
    if config.bwd_num_comm_sms % 2 != 0:
        raise ValueError("bwd_num_comm_sms must be even")
    recompute_comm_sms = config.bwd_recompute_num_comm_sms
    if recompute_comm_sms is not None:
        if type(recompute_comm_sms) is not int or recompute_comm_sms <= 0:
            raise ValueError("bwd_recompute_num_comm_sms must be a positive integer or None")
        if recompute_comm_sms % 2 != 0:
            raise ValueError("bwd_recompute_num_comm_sms must be even")
    if any(count >= device_properties.multi_processor_count for count in (config.fwd_num_comm_sms, config.bwd_num_comm_sms, recompute_comm_sms or 0)):
        raise ValueError("communication-SM counts must leave at least one compute SM")
    if config.gate_up_context not in ("bf16", "mxfp8"):
        raise ValueError("gate_up_context must be 'bf16' or 'mxfp8'")
    if type(config.epilogue_combine) is not bool:
        raise TypeError("epilogue_combine must be a bool")
    _comm_tile_shape(workspace, config)
    # Each direction resolves to its own override or to the shared minibatch_size; validating the
    # resolved values keeps the message about the field the caller actually set.
    direction_minibatches = tuple(
        (name if value is not None else "minibatch_size", config.minibatch_size if value is None else value)
        for name, value in (("fwd_minibatch_size", config.fwd_minibatch_size), ("bwd_minibatch_size", config.bwd_minibatch_size))
    )
    for name, minibatch_size in direction_minibatches:
        if type(minibatch_size) is not int or minibatch_size <= 0 or minibatch_size % 256 != 0:
            raise ValueError(f"{name} must be positive and divisible by 256")
    if type(config.macrobatch_size) is not int or config.macrobatch_size <= 0:
        raise ValueError("macrobatch_size must be a positive integer")
    if any(config.macrobatch_size % minibatch_size != 0 for _, minibatch_size in direction_minibatches):
        raise ValueError("macrobatch_size must be a multiple of every minibatch size")
    if (
        type(config.all_gather_top_experts_chunk_bytes) is not int
        or config.all_gather_top_experts_chunk_bytes <= 0
        or config.all_gather_top_experts_chunk_bytes % 16 != 0
    ):
        raise ValueError("all_gather_top_experts_chunk_bytes must be a positive multiple of 16")
    if config.all_gather_top_experts_chunk_bytes + 1024 > device_properties.shared_memory_per_block_optin:
        raise ValueError("all_gather_top_experts_chunk_bytes exceeds device dynamic shared-memory capacity")
    route_buffer_bytes = workspace.num_local_tokens * workspace.topk * 4
    if route_buffer_bytes % config.all_gather_top_experts_chunk_bytes != 0:
        raise ValueError("all_gather_top_experts_chunk_bytes must divide one rank's route-buffer bytes")


def build_schedule(
    workspace: CuteMoKWorkspace,
    config: CuteMoKConfig,
    top_experts: torch.Tensor,
    *,
    num_local_experts: int,
) -> CuteMoKSchedule:
    """All-gathers routing choices and builds this rank's padded expert schedule.

    Inputs:
        workspace:         CuteMoKWorkspace
        config:            CuteMoKConfig
        top_experts:       int32 or int64 [num_local_tokens, topk]; -1 marks an unused route slot.  The kernels
                           read int32: int64 is converted on every call (an allocation and a conversion kernel),
                           and ``schedule_execute`` takes int32 only.
        num_local_experts: int

    Outputs:
        schedule: CuteMoKSchedule
    """
    return _build_schedule(workspace, config, top_experts, num_local_experts)


def _build_schedule(workspace, config, top_experts, num_local_experts, out=None):
    """The schedule build of ``build_schedule`` and of ``schedule_execute``, which passes ``out`` -- the caller's
    peer_rank, peer_token_idx and counts buffers and the plan's scratch -- so that nothing is allocated."""
    if not isinstance(workspace, CuteMoKWorkspace):
        raise TypeError("workspace must be a CuteMoKWorkspace")
    _validate_config_type(config)
    # The options are validated once per (config, device, num_local_tokens, topk) (CuteMoKConfig is
    # frozen, so the key holds its values); the caller's routes are checked on every call below.
    config_key = (config, workspace.device, workspace.num_local_tokens, workspace.topk)
    if config_key not in _VALIDATED_SCHEDULE_CONFIGS:
        _validate_schedule_config(workspace, config)
        _VALIDATED_SCHEDULE_CONFIGS.add(config_key)
    if not top_experts.is_cuda or top_experts.device != workspace.device:
        raise ValueError("top_experts must be on the workspace CUDA device")
    if top_experts.dtype not in (torch.int32, torch.int64):
        raise TypeError("top_experts must have dtype torch.int32 or torch.int64")
    if not top_experts.is_contiguous():
        raise ValueError("top_experts must be contiguous")
    if tuple(top_experts.shape) != (workspace.num_local_tokens, workspace.topk):
        raise ValueError("top_experts must have shape (num_local_tokens, topk)")
    if type(num_local_experts) is not int or num_local_experts <= 0:
        raise ValueError("num_local_experts must be a positive integer")

    ops = _device_ops(config)
    if out is not None and top_experts.dtype != torch.int32:
        raise TypeError("schedule_execute takes int32 top_experts (a conversion would allocate)")
    # int32 is what the kernels read; a caller that already has it is not charged a conversion.
    top_experts_int32 = top_experts if top_experts.dtype == torch.int32 else top_experts.to(torch.int32)
    ops.all_gather_top_experts(
        top_experts_int32,
        workspace.all_gather_top_experts_buffer,
        workspace.all_gather_top_experts_buffer_multicast_ptr,
        workspace.ep_rank,
        config.all_gather_top_experts_chunk_bytes,
    )
    ops.barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    # The four schedule tensors, then the unpadded rows per expert.
    schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert, *valid_tokens_per_expert = ops.schedule(
        workspace.all_gather_top_experts_buffer, num_local_experts, workspace.schedule_capacity, workspace.ep_rank, out=out
    )
    return CuteMoKSchedule(
        peer_rank=schedule_peer_rank,
        peer_token_idx=schedule_peer_token_idx,
        num_tokens=num_tokens,
        tokens_per_expert=tokens_per_expert,
        valid_tokens_per_expert=valid_tokens_per_expert[0] if valid_tokens_per_expert else None,
        top_experts=top_experts_int32,
    )


def validate_inputs(
    config: CuteMoKConfig,
    workspace: CuteMoKWorkspace,
    schedule: CuteMoKSchedule,
    x: torch.Tensor,
    router_weights: torch.Tensor | None = None,
    grad_output: torch.Tensor | None = None,
) -> None:
    """Validates runtime inputs against the workspace and schedule.

    Inputs:
        config:         CuteMoKConfig
        workspace:      CuteMoKWorkspace
        schedule:       CuteMoKSchedule
        x:              bfloat16 [num_local_tokens, hidden_size]
        router_weights: float32 [num_local_tokens, topk] | None
        grad_output:    bfloat16 [num_local_tokens, hidden_size] | None

    Outputs:
        None
    """
    _validate_config_type(config)
    if not isinstance(workspace, CuteMoKWorkspace):
        raise TypeError("workspace must be a CuteMoKWorkspace")
    if not isinstance(schedule, CuteMoKSchedule):
        raise TypeError("schedule must be a CuteMoKSchedule")
    expected_activation_shape = (workspace.num_local_tokens, workspace.hidden_size)
    tensors = [("x", x, torch.bfloat16, expected_activation_shape)]
    if grad_output is not None:
        tensors.append(("grad_output", grad_output, torch.bfloat16, expected_activation_shape))
    if schedule.top_experts is None:
        raise ValueError("schedule.top_experts is missing: build the schedule with build_schedule")
    tensors.append(("schedule.top_experts", schedule.top_experts, torch.int32, (workspace.num_local_tokens, workspace.topk)))
    if router_weights is not None:
        tensors.append(("router_weights", router_weights, torch.float32, (workspace.num_local_tokens, workspace.topk)))
    for tensor_name, tensor, expected_dtype, expected_shape in tensors:
        if not tensor.is_cuda or tensor.device != workspace.device or tensor.dtype != expected_dtype or not tensor.is_contiguous():
            raise ValueError(f"{tensor_name} must be contiguous {expected_dtype} on the workspace CUDA device")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{tensor_name} shape does not match the workspace")
    if schedule.peer_rank.numel() != workspace.schedule_capacity:
        raise ValueError("schedule capacity does not match the workspace")


def _normalize_mxfp8_backward_weights(gate, up, down):
    """The backward's routed MXFP8 arguments, with the optional trailing layout flag taken off.

    The backward reads Gate/Up as ``(e4m3, scale, e4m3_t, scale_t)`` and Down as
    ``(e4m3_t, scale_t)``.  A caller may append one bool saying whether the columnwise payload is
    the native ``[E, M, K]`` TransformerEngine layout rather than the transposed one
    ``cudnn.moe_ep._cutemok.ops.mxfp8_quantize`` produces; the arguments are then a five- and a three-tuple.  The
    dgrad reads a native payload as an MN-major operand instead of a K-major one; transposing it on
    the host would copy every routed weight every step.

    Returns:
        (gate, up, down, native_columnwise)
    """
    flags = {len(weights) for weights in (gate, up, down)}
    natives = []
    trimmed = []
    for weights, expected in ((gate, 4), (up, 4), (down, 2)):
        if len(weights) == expected:
            natives.append(False)
            trimmed.append(weights)
        elif len(weights) == expected + 1 and isinstance(weights[-1], bool):
            natives.append(weights[-1])
            trimmed.append(tuple(weights[:-1]))
        else:
            raise ValueError(
                f"an MXFP8 routed backward weight must have {expected} entries, or {expected + 1} "
                f"with a trailing native-columnwise flag; got {len(weights)} ({sorted(flags)})"
            )
    if len(set(natives)) != 1:
        raise ValueError("the routed weights disagree about the native-columnwise layout")
    return (*trimmed, natives[0])


@dataclass(frozen=True)
class CuteMoKSplitWeight:
    """One logical routed matrix held in independent per-expert allocations.

    For a caller that owns the routed experts as separate parameters instead of one grouped
    ``[E, N, K]`` parameter.  The allocations are identical in dtype, shape and stride and differ
    only in base address, so CuteMoK takes one representative tensor (which carries the schema the
    validators check) plus a table of the addresses, and the kernel points its TMA descriptor at the
    right one per expert.  Routed Gate, Up and Down must all be split or none of them.
    ``from_experts`` builds one from the per-expert tensors.

    ``data``: the representative expert's payload, ``[N, K]`` (BF16, or the MXFP8 e4m3 payload).
    ``storage_table``: int64 ``[num_local_experts]`` base addresses of every expert's payload
    (``ops.make_routed_weight_storage_table_bf16`` / ``_mxfp8``).
    ``scale`` / ``scale_storage_table`` (MXFP8): the representative expert's scale and the
    addresses of every expert's scale (``ops.make_routed_scale_storage_table``).

    The tables own no storage.  The caller's parameters own every payload, so there is no parallel
    ``data_tensors`` field; the tcgen05-layout scales, by contrast, are built for CuteMoK alone, so
    ``scale_tensors`` keeps them alive for as long as the table refers to them.

    ``transposed_*`` carry the MXFP8 backward's columnwise operand.  ``native_columnwise`` says it
    is TransformerEngine's own columnwise payload, in the weight's ``[N, K]`` order, rather than a
    materialized ``[K, N]`` transpose, exactly as in the tuple form (``_normalize_mxfp8_backward_weights``).

    Gate and Up may deliberately share one ``[2I, H]`` payload, table and scale: CuteMoK then applies
    the same logical row offsets a combined FC1 gets, without repacking the parameter.
    """

    data: torch.Tensor
    storage_table: torch.Tensor
    scale: torch.Tensor | None = None
    scale_storage_table: torch.Tensor | None = None
    scale_tensors: tuple[torch.Tensor, ...] | None = None
    transposed_data: torch.Tensor | None = None
    transposed_scale: torch.Tensor | None = None
    transposed_storage_table: torch.Tensor | None = None
    transposed_scale_storage_table: torch.Tensor | None = None
    transposed_scale_tensors: tuple[torch.Tensor, ...] | None = None
    native_columnwise: bool = False

    @classmethod
    def from_experts(
        cls,
        experts,
        scales=None,
        transposed=None,
        transposed_scales=None,
        *,
        native_columnwise: bool = False,
    ) -> "CuteMoKSplitWeight":
        """One routed matrix from its per-expert tensors, in local expert order; builds the address tables.

        Inputs (each a list or tuple with one tensor per local expert; the tensors of one list contiguous,
        16-byte aligned, on one CUDA device, with one dtype and shape):
            experts:           BF16 ``[N, K]`` matrices, or MXFP8 E4M3 (float8_e4m3fn) ``[N, K]`` payloads
            scales:            MXFP8: the payloads' scales, uint8 ``[N // 128, K // 128, 32, 16]`` (tcgen05 layout,
                               as ``cutemok_quantize_weights`` produces them per expert); None for BF16
            transposed:        MXFP8, for the backward: the columnwise payloads, float8_e4m3fn ``[K, N]``, or with
                               ``native_columnwise`` TransformerEngine's columnwise payloads in the ``[N, K]`` order
            transposed_scales: their scales, uint8 ``[K // 128, N // 128, 32, 16]`` (the scales of the ``[K, N]``
                               matrix in either case); given with ``transposed``
            native_columnwise: the columnwise payloads are in ``[N, K]`` order

        Gate and up of one combined FC1 (``[2I, H]`` per expert) are one weight, passed as both arguments.  The
        weight keeps the scales alive (``scale_tensors``, ``transposed_scale_tensors``) but not the payloads: the
        caller keeps every payload allocation alive, at its address, while the weight is in use.  The tables are
        new int64 CUDA tensors; a plan builds the TMA descriptors of new tables on their first call, so build the
        weight once (again after the payloads move) rather than per call.
        """
        from . import ops

        if not isinstance(experts, (list, tuple)) or not experts or not isinstance(experts[0], torch.Tensor):
            raise TypeError("experts must be a nonempty list of tensors, one per local expert")
        mxfp8 = experts[0].dtype == torch.float8_e4m3fn
        if not mxfp8 and experts[0].dtype != torch.bfloat16:
            raise TypeError(f"experts must be bfloat16 or float8_e4m3fn; got {experts[0].dtype}")
        if mxfp8 and scales is None:
            raise ValueError("MXFP8 experts need their scales")
        if not mxfp8 and (scales is not None or transposed is not None or transposed_scales is not None or native_columnwise):
            raise ValueError("BF16 experts take no scales, transposed payloads or native_columnwise")
        if (transposed is None) != (transposed_scales is None):
            raise ValueError("transposed and transposed_scales are given together")
        if native_columnwise and transposed is None:
            raise ValueError("native_columnwise describes the transposed payloads, which are not given")
        for name, tensors in (("scales", scales), ("transposed", transposed), ("transposed_scales", transposed_scales)):
            if tensors is not None and (not isinstance(tensors, (list, tuple)) or len(tensors) != len(experts)):
                raise ValueError(f"{name} must be a list with one tensor per expert ({len(experts)})")
        if not mxfp8:
            return cls(data=experts[0], storage_table=ops.make_routed_weight_storage_table_bf16(experts))
        fields = dict(
            data=experts[0],
            storage_table=ops.make_routed_weight_storage_table_mxfp8(experts),
            scale=scales[0],
            scale_storage_table=ops.make_routed_scale_storage_table(scales),
            scale_tensors=tuple(scales),
        )
        if transposed is not None:
            fields.update(
                transposed_data=transposed[0],
                transposed_scale=transposed_scales[0],
                transposed_storage_table=ops.make_routed_weight_storage_table_mxfp8(transposed),
                transposed_scale_storage_table=ops.make_routed_scale_storage_table(transposed_scales),
                transposed_scale_tensors=tuple(transposed_scales),
                native_columnwise=native_columnwise,
            )
        return cls(**fields)


#: The routed weights' per-expert address tables, or None when none of them is split.  ``scales``
#: and ``transposed`` are each a Gate/Up/Down triple or None; an MXFP8 weight is a payload *and* a
#: scale, and both are that expert's own allocation.
RoutedStorageTables = collections.namedtuple("RoutedStorageTables", "gate up down transposed scales transposed_scales")


def _routed_weight_payload(weight):
    """The plain tensor / tuple form of a routed weight argument.

    A ``CuteMoKSplitWeight`` becomes the representative expert's payload in the same shape the
    non-split forms use, so every validator and every shape derivation downstream is unchanged;
    what the split form adds is the address table, which travels separately.
    """
    if not isinstance(weight, CuteMoKSplitWeight):
        return weight
    if weight.scale is None:
        return weight.data
    if weight.transposed_data is None:
        return (weight.data, weight.scale)
    return (weight.data, weight.scale, weight.transposed_data, weight.transposed_scale, weight.native_columnwise)


def _routed_storage_tables(gate, up, down):
    """``RoutedStorageTables`` when any routed weight is split, else ``None``.

    Splitting is all-or-nothing: the kernel keeps one descriptor per operand and retargets it per
    expert, so a mix would mean two different addressing schemes inside one GEMM.
    """
    split = [isinstance(weight, CuteMoKSplitWeight) for weight in (gate, up, down)]
    if not any(split):
        return None
    if not all(split):
        raise ValueError(
            "routed Gate, Up and Down must all be CuteMoKSplitWeight or none of them; got " f"{tuple(type(weight).__name__ for weight in (gate, up, down))}"
        )

    def group(attribute):
        tables = tuple(getattr(weight, attribute) for weight in (gate, up, down))
        if any(table is None for table in tables) and not all(table is None for table in tables):
            raise ValueError(f"a split routed weight's {attribute} is all-or-nothing: the kernel addresses " "every expert of an operand the same way")
        return None if all(table is None for table in tables) else tables

    return RoutedStorageTables(
        gate=gate.storage_table,
        up=up.storage_table,
        down=down.storage_table,
        transposed=group("transposed_storage_table"),
        scales=group("scale_storage_table"),
        transposed_scales=group("transposed_scale_storage_table"),
    )


def _combined_fc1(routed_gate_weights, routed_up_weights):
    """The Gate and Up payloads, and where Up starts when they are one combined FC1.

    A caller that owns the routed FC1 as one ``[E, 2I, H]`` parameter -- ``[2I, H]`` per expert
    when the experts are split -- passes that same object as both the gate and the up argument.
    The kernels read the whole tensor under both names and find Up ``I`` rows down (on the
    payload, on the MXFP8 scale and on the backward's transposed payload); the halves are never
    sliced apart.  This returns that ``I`` (0 for separate Gate and Up weights) for the kernels to
    apply as a tile coordinate.

    The payload object is returned once, under both names, so the kernels can check that the two
    really are one tensor.
    """
    gate = _routed_weight_payload(routed_gate_weights)
    if routed_gate_weights is routed_up_weights:
        combined = True
    else:
        # One combined FC1 is the same payload tensor under both names, whatever container carries
        # it: two slices ``fc1[:2]`` of one quantized FC1 are the same weight too.  Its scale must
        # then be shared as well, and separate payloads must have separate scales.
        up = _routed_weight_payload(routed_up_weights)
        gate_data = gate[0] if isinstance(gate, tuple) else gate
        up_data = up[0] if isinstance(up, tuple) else up
        combined = gate_data is up_data
        if isinstance(gate, tuple) and isinstance(up, tuple) and combined != (gate[1] is up[1]):
            raise ValueError("routed gate and up must share both their payload and its scale (one combined " "FC1) or neither")
        if not combined:
            return gate, up, 0
    rows = int((gate[0] if isinstance(gate, tuple) else gate).shape[-2])
    if rows % 2:
        raise ValueError(
            "routed_gate_weights and routed_up_weights are the same object, so it must be a "
            "combined FC1 of 2 * intermediate_size rows: (E, 2I, H), or (2I, H) for one split "
            f"expert; got {rows} rows"
        )
    return gate, gate, rows // 2


def forward(
    config: CuteMoKConfig,
    workspace: CuteMoKWorkspace,
    schedule: CuteMoKSchedule,
    x: torch.Tensor,
    router_weights: torch.Tensor,
    shared_gate_weights: torch.Tensor | None,
    shared_up_weights: torch.Tensor | None,
    shared_down_weights: torch.Tensor | None,
    routed_gate_weights: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    routed_up_weights: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    routed_down_weights: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    swiglu_limit: float | None = None,
    routed_input: torch.Tensor | None = None,
    *,
    shared_output_gate_weight: torch.Tensor | None = None,
    situ_beta: tuple[float, float] | None = None,
) -> tuple[
    torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    CuteMoKForwardContext,
]:
    """Runs the MoE forward pass.

    All three shared weights may be None for routed-only execution: no shared tasks or buffers
    run, the output is the routed sum alone and the context's shared fields are None.

    The routed weights may also be ``CuteMoKSplitWeight`` (per-expert allocations; all three or none).
    Passing one object as both ``routed_gate_weights`` and ``routed_up_weights`` means a combined FC1
    of ``2 * intermediate_size`` rows, Gate first.

    Inputs:
        config:              CuteMoKConfig
        workspace:           CuteMoKWorkspace
        schedule:            CuteMoKSchedule
        x:                   bfloat16 [num_local_tokens, hidden_size]
        router_weights:      float32 [num_local_tokens, topk]
        shared_gate_weights: bfloat16 [intermediate_size, hidden_size]
        shared_up_weights:   bfloat16 [intermediate_size, hidden_size]
        shared_down_weights: bfloat16 [hidden_size, intermediate_size]
        routed_gate_weights: bfloat16 [num_local_experts, intermediate_size, hidden_size] or MXFP8 data/scale tuple
        routed_up_weights:   bfloat16 [num_local_experts, intermediate_size, hidden_size] or MXFP8 data/scale tuple
        routed_down_weights: bfloat16 [num_local_experts, hidden_size, intermediate_size] or MXFP8 data/scale tuple
        swiglu_limit:        float | None
        routed_input:            bfloat16 [num_local_tokens, routed_hidden_size] | None -- routed-expert input
                             (required when workspace.routed_hidden_size != hidden_size; the routed weights are
                             then [.., routed_hidden_size] / [.., routed_hidden_size, ..], BF16 or MXFP8 tuples)
        shared_output_gate_weight: optional bfloat16 [1, hidden_size] -- the shared expert's output gate:
                             the shared output is scaled by G = sigmoid(x @ w^T) [num_local_tokens, 1],
                             merged in FP32 by the epilogue; S and G are kept for backward
        situ_beta:           optional (beta1, beta2): SiTU-GLU instead of SwiGLU for the routed and shared MLPs,
                             two positive finite FP32 numbers, never with swiglu_limit; backward uses the betas
                             saved in the context.  It does not change the shared output gate's sigmoid.

    Outputs:
        output:          bfloat16 [num_local_tokens, hidden_size]  (routed_hidden_size == hidden_size:
                         y_shared + sum_k w_k y_k) or (y_shared [num_local_tokens, hidden_size],
                         y_routed [num_local_tokens, routed_hidden_size]) when they differ: the caller
                         combines the two (e.g. forward_latent_layer)
        forward_context: CuteMoKForwardContext

    MXFP8 routed weights run the MXFP8 forward; ``config.epilogue_combine`` and ``config.comm_tile_shape`` are BF16
    only.
    """
    return _forward(
        config,
        workspace,
        schedule,
        x,
        router_weights,
        shared_gate_weights,
        shared_up_weights,
        shared_down_weights,
        routed_gate_weights,
        routed_up_weights,
        routed_down_weights,
        swiglu_limit,
        routed_input,
        shared_output_gate_weight=shared_output_gate_weight,
        situ_beta=situ_beta,
    )


def _forward(
    config,
    workspace,
    schedule,
    x,
    router_weights,
    shared_gate_weights,
    shared_up_weights,
    shared_down_weights,
    routed_gate_weights,
    routed_up_weights,
    routed_down_weights,
    swiglu_limit,
    routed_input,
    *,
    shared_output_gate_weight=None,
    situ_beta=None,
    buffers=None,
):
    """The forward of ``forward`` and of ``forward_execute``: the same checks, staging copy, barriers and kernels.

    ``forward`` leaves ``buffers`` None: the megakernel's outputs and counters and the epilogue's output are
    allocated per call, and a kernel variant is compiled when it is first used.  ``forward_execute`` passes
    ``buffers = (context_out, out, counters, scratch, gate_out)`` -- the caller's megakernel outputs (nine BF16, or
    the MXFP8 forward's thirteen / eleven) and final output, the plan's counter buffer, for MXFP8 the plan's four
    scratch buffers and, gated, the caller's BF16 [num_local_tokens, 1] buffer of G -- so nothing is allocated.
    """
    _validate_config_type(config)
    validate_inputs(config, workspace, schedule, x, router_weights)
    from .cutedsl.persistent_bf16_fwd import shared_presence

    context_out, out, counters, scratch, gate_out = (None,) * 5 if buffers is None else buffers
    has_shared = shared_presence(shared_gate_weights, shared_up_weights, shared_down_weights)
    if shared_output_gate_weight is not None and not has_shared:
        raise ValueError("shared output gate requires shared expert weights")
    if shared_output_gate_weight is not None:
        _validate_shared_output_gate_weight(shared_output_gate_weight, x)
    from .cutedsl._host_common import normalize_situ_beta

    situ_beta = normalize_situ_beta(situ_beta, swiglu_limit)
    split_hidden = _routed_hidden_differs(workspace)
    if split_hidden and shared_output_gate_weight is not None:
        raise NotImplementedError("the shared output gate needs routed_hidden_size == hidden_size")
    if split_hidden and not has_shared:
        raise NotImplementedError("a routed-only layer needs routed_hidden_size == hidden_size")
    if split_hidden:
        if routed_input is None:
            raise ValueError("routed_input is required when workspace.routed_hidden_size != hidden_size")
        if (
            not routed_input.is_cuda
            or routed_input.device != workspace.device
            or routed_input.dtype != torch.bfloat16
            or not routed_input.is_contiguous()
            or tuple(routed_input.shape) != (workspace.num_local_tokens, workspace.routed_hidden_size)
        ):
            raise ValueError("routed_input must be a contiguous bfloat16 [num_local_tokens, routed_hidden_size] tensor " "on the workspace device")
    elif routed_input is not None and routed_input is not x:
        raise ValueError("routed_input is only accepted when workspace.routed_hidden_size != hidden_size")

    storage_tables = _routed_storage_tables(routed_gate_weights, routed_up_weights, routed_down_weights)
    routed_gate_weights, routed_up_weights, up_row_offset = _combined_fc1(routed_gate_weights, routed_up_weights)
    routed_down_weights = _routed_weight_payload(routed_down_weights)
    ops = _device_ops(config)
    # Peers read x_buffer only at routed rows.  An aligned x stands in for its buffer as the shared
    # expert's operand, so only its routed rows are copied; otherwise, or on the MXFP8 /
    # split-hidden paths, x is copied in full.
    routed_rows_only = not split_hidden and not isinstance(routed_gate_weights, tuple) and _can_stand_in_for_buffer(x)
    if routed_rows_only:
        from .cutedsl.copy_routed_rows import copy_routed_rows

        copy_routed_rows(schedule.top_experts, [x], [workspace.x_buffer], router_weights, workspace.router_weight_buffer)
    else:
        workspace.x_buffer.copy_(routed_input if split_hidden else x)
        workspace.router_weight_buffer.copy_(router_weights)
    ops.barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)

    if isinstance(routed_gate_weights, tuple):
        # The forward reads the rowwise pair; a caller may carry more (the backward's columnwise
        # payload and the native-columnwise flag ride along in the same argument).
        routed_gate_weights = routed_gate_weights[:2]
        routed_up_weights = routed_up_weights[:2]
        routed_down_weights = routed_down_weights[:2]
        if config.epilogue_combine:
            raise NotImplementedError("epilogue_combine is implemented for the BF16 forward only")
        _require_default_comm_tile_shape(workspace, config)
        # Thirteen outputs with gate_up_context="mxfp8"; with "bf16" eleven, the routed Gate / Up
        # context one BF16 tensor each.  The host work (checks, state, arguments) follows the entry
        # barrier, which then waits for the other ranks while this one prepares.  With differing
        # hiddens the shared expert reads x and the routed chain the x buffer (routed_input).
        from .cutedsl.persistent_mxfp8_fwd import forward_mxfp8

        if storage_tables is not None and storage_tables.scales is None:
            raise ValueError("split MXFP8 routed weights need their scales' per-expert addresses too " "(CuteMoKSplitWeight.scale_storage_table)")
        mxfp8_outputs = forward_mxfp8(
            workspace,
            schedule,
            shared_gate_weights,
            routed_gate_weights,
            shared_up_weights,
            routed_up_weights,
            shared_down_weights,
            routed_down_weights,
            macrobatch_size=config.macrobatch_size,
            minibatch_size=_fwd_minibatch_size(config),
            num_comm_sms=config.fwd_num_comm_sms,
            swiglu_limit=swiglu_limit,
            gate_up_context=config.gate_up_context,
            x_shared=x if split_hidden else None,
            up_row_offset=up_row_offset,
            # The three payloads, then their three scales.
            routed_storage_tables=(None if storage_tables is None else (storage_tables.gate, storage_tables.up, storage_tables.down, *storage_tables.scales)),
            situ_beta=situ_beta,
            caller_buffers=None if buffers is None else (context_out, scratch, counters),
        )
        if config.gate_up_context == "bf16":
            # The thirteen-output order with the Gate / Up scale slots empty: the context keeps the BF16 tensors.
            mxfp8_outputs = (*mxfp8_outputs[:4], None, *mxfp8_outputs[4:6], None, *mxfp8_outputs[6:])
        (
            x_fp8_t_routed,
            x_sc_t_routed,
            gate_shared,
            gate_fp8_routed,
            gate_sc_routed,
            up_shared,
            up_fp8_routed,
            up_sc_routed,
            hidden_shared,
            hidden_fp8_t_routed,
            hidden_sc_t_routed,
            y_shared,
            y_routed,
        ) = mxfp8_outputs
        forward_context = CuteMoKForwardContext(
            x_routed=(x_fp8_t_routed, x_sc_t_routed),
            gate_shared=gate_shared,
            # No scales: the BF16 Gate / Up context (gate_up_context="bf16").
            gate_routed=gate_fp8_routed if gate_sc_routed is None else (gate_fp8_routed, gate_sc_routed),
            up_shared=up_shared,
            up_routed=up_fp8_routed if up_sc_routed is None else (up_fp8_routed, up_sc_routed),
            hidden_shared=hidden_shared,
            hidden_routed=(hidden_fp8_t_routed, hidden_sc_t_routed),
        )
    else:
        from .cutedsl.persistent_bf16_fwd import (
            forward_bf16 as cutedsl_forward_bf16,
        )

        x_routed, gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, y_shared, y_routed = cutedsl_forward_bf16(
            workspace,
            schedule,
            shared_gate_weights,
            routed_gate_weights,
            shared_up_weights,
            routed_up_weights,
            shared_down_weights,
            routed_down_weights,
            macrobatch_size=config.macrobatch_size,
            minibatch_size=_fwd_minibatch_size(config),
            swiglu_limit=swiglu_limit,
            num_comm_sms=config.fwd_num_comm_sms,
            x_shared=x if split_hidden or routed_rows_only else None,
            epilogue_combine=config.epilogue_combine,
            comm_tile_shape=config.comm_tile_shape,
            routed_storage_tables=(None if storage_tables is None else (storage_tables.gate, storage_tables.up, storage_tables.down)),
            up_row_offset=up_row_offset,
            situ_beta=situ_beta,
            outputs=context_out,
            counters=counters,
        )
        forward_context = CuteMoKForwardContext(
            x_routed=x_routed,
            gate_shared=gate_shared,
            gate_routed=gate_routed,
            up_shared=up_shared,
            up_routed=up_routed,
            hidden_shared=hidden_shared,
            hidden_routed=hidden_routed,
        )

    if situ_beta is not None:
        forward_context = _replace_fields(forward_context, situ_beta=situ_beta)
    shared_output_gate = None
    if shared_output_gate_weight is not None:
        # This functional API has a manual backward: no autograd graph (nor Z) is kept.
        with torch.no_grad():
            if gate_out is None:
                shared_output_gate = torch.sigmoid(torch.nn.functional.linear(x, shared_output_gate_weight))
            else:  # forward_execute: the same GEMM (linear's x @ w^T) and sigmoid, into the caller's buffer
                shared_output_gate = torch.sigmoid(torch.mm(x, shared_output_gate_weight.t(), out=gate_out), out=gate_out)
        forward_context = _replace_fields(forward_context, shared_output=y_shared, shared_output_gate=shared_output_gate)
    ops.barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    if _routed_hidden_differs(workspace):
        # The shared and routed outputs live in different hiddens: reduce the routed top-k rows
        # here and leave combining the two outputs to the caller.
        return (y_shared, _reduce_topk_routed(ops, workspace, schedule.top_experts, out)), forward_context
    output = ops.fwd_epilogue(
        y_shared, workspace.combine_buffer, workspace.router_weight_buffer, schedule.top_experts, shared_output_gate=shared_output_gate, out=out
    )
    return output, forward_context


def recompute_forward_context(
    config: CuteMoKConfig,
    workspace: CuteMoKWorkspace,
    schedule: CuteMoKSchedule,
    x: torch.Tensor,
    shared_gate_weights: torch.Tensor | None,
    shared_up_weights: torch.Tensor | None,
    routed_gate_weights: torch.Tensor | tuple[torch.Tensor, ...] | CuteMoKSplitWeight,
    routed_up_weights: torch.Tensor | tuple[torch.Tensor, ...] | CuteMoKSplitWeight,
    swiglu_limit: float | None = None,
    *,
    situ_beta: tuple[float, float] | None = None,
) -> CuteMoKForwardContext:
    """Rebuilds the context ``backward`` needs when the one ``forward`` returned was discarded, so that no
    forward context is held between a layer's forward and its backward.

    Only the shared expert's Gate / Up / Hidden are recomputed here (``num_local_tokens x intermediate_size``
    each): one launch of the forward megakernel's shared FC1 tasks alone, in the precision of the routed
    weights as ``forward`` runs them, so the three tensors are the forward context's bit for bit.  The launch
    does no communication and runs no barrier.  The routed activations of the returned context are None:
    ``backward`` replays the first macrobatch's routed X / Gate / Up in its kernel, as it does for every later
    macrobatch, so the macrobatch-sized routed context exists only while the backward runs.  A caller that can
    keep the shared activations needs no recompute at all: ``forward_context.without_routed()``.

    A layer with a shared output gate is not supported (its saved S / G come from the forward only).  For
    SiTU-GLU pass the forward's ``situ_beta``.  The routed weights only select the forward kernel (BF16 or
    MXFP8); they are not read: the replay uses the ones passed to ``backward``.

    Inputs:
        config:              CuteMoKConfig
        workspace:           CuteMoKWorkspace
        schedule:            CuteMoKSchedule
        x:                   bfloat16 [num_local_tokens, hidden_size]
        shared_gate_weights: bfloat16 [intermediate_size, hidden_size], or None for a routed-only layer
        shared_up_weights:   bfloat16 [intermediate_size, hidden_size], or None for a routed-only layer
        routed_gate_weights: the routed Gate weights in any form ``forward`` takes
        routed_up_weights:   the routed Up weights, in the same form
        swiglu_limit:        float | None
        situ_beta:           optional (beta1, beta2), as passed to ``forward``

    Outputs:
        forward_context: CuteMoKForwardContext
    """
    from .cutedsl._host_common import normalize_situ_beta
    from .cutedsl.persistent_bf16_fwd import shared_presence

    validate_inputs(config, workspace, schedule, x)
    situ_beta = normalize_situ_beta(situ_beta, swiglu_limit)
    if isinstance(routed_gate_weights, tuple) != isinstance(routed_up_weights, tuple):
        raise TypeError("routed gate and up weights must use the same precision representation")
    gate_shared = up_shared = hidden_shared = None
    if shared_presence(shared_gate_weights, shared_up_weights):
        # The shared expert's operand: x itself, or a private copy of it.  The symmetric x buffer is not written:
        # no barrier separates this call from the peers' reads of that buffer in the previous launch.
        x_shared = x
        if not _can_stand_in_for_buffer(x):
            x_shared = torch.empty(x.shape, dtype=x.dtype, device=x.device).copy_(x)
        # Each precision's forward kernel computes the shared expert its own way.
        if isinstance(_routed_weight_payload(routed_gate_weights), tuple):
            from .cutedsl.persistent_mxfp8_fwd import shared_fc1_mxfp8 as shared_fc1
        else:
            from .cutedsl.persistent_bf16_fwd import shared_fc1_bf16 as shared_fc1
        gate_shared, up_shared, hidden_shared = shared_fc1(
            workspace, x_shared, shared_gate_weights, shared_up_weights, swiglu_limit=swiglu_limit, situ_beta=situ_beta
        )
    return CuteMoKForwardContext(
        x_routed=None,
        gate_shared=gate_shared,
        gate_routed=None,
        up_shared=up_shared,
        up_routed=None,
        hidden_shared=hidden_shared,
        hidden_routed=None,
        situ_beta=situ_beta,
    )


def _reduce_topk_routed(ops, workspace: CuteMoKWorkspace, top_experts: torch.Tensor, out=None) -> torch.Tensor:
    """sum_k router_weight[t, k] * combine_buffer[t * topk + k, :] (fp32 accumulation, one bf16
    rounding): the routed half of the forward epilogue, its routed-only variant."""
    return ops.fwd_epilogue(None, workspace.combine_buffer, workspace.router_weight_buffer, top_experts, out=out)


def _validate_latent_projections(config, workspace, latent_down_weights, latent_up_weights) -> None:
    _validate_config_type(config)
    if not _routed_hidden_differs(workspace):
        raise ValueError("the latent layer requires a workspace with routed_hidden_size != hidden_size")
    for name, tensor, shape in (
        ("latent_down_weights", latent_down_weights, (workspace.routed_hidden_size, workspace.hidden_size)),
        ("latent_up_weights", latent_up_weights, (workspace.hidden_size, workspace.routed_hidden_size)),
    ):
        if not isinstance(tensor, torch.Tensor) or tensor.device != workspace.device or tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must be bfloat16 {shape} on the workspace device")


def forward_latent_layer(
    config: CuteMoKConfig,
    workspace: CuteMoKWorkspace,
    schedule: CuteMoKSchedule,
    x: torch.Tensor,
    router_weights: torch.Tensor,
    latent_down_weights: torch.Tensor,
    latent_up_weights: torch.Tensor,
    shared_gate_weights: torch.Tensor,
    shared_up_weights: torch.Tensor,
    shared_down_weights: torch.Tensor,
    routed_gate_weights: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    routed_up_weights: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    routed_down_weights: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    swiglu_limit: float | None = None,
) -> tuple[torch.Tensor, CuteMoKLatentForwardContext]:
    """Latent MoE layer forward: the routed experts work in routed_hidden_size, the shared expert in hidden_size.

        routed_input = x @ latent_down_weights^T             (GEMM; kept for the backward)
        y_shared, y_routed = forward(..., routed_input)      (mega kernel + routed top-k reduction)
        output   = y_shared + y_routed @ latent_up_weights^T   (one GEMM with the add)

    Inputs:
        x:                   bfloat16 [num_local_tokens, hidden_size]
        latent_down_weights: bfloat16 [routed_hidden_size, hidden_size]
        latent_up_weights:   bfloat16 [hidden_size, routed_hidden_size]
        routed_*_weights:    in routed_hidden_size ([E, I, routed_hidden_size] / [E, routed_hidden_size, I]): bfloat16,
                             or MXFP8 data/scale tuples as in ``forward`` (the routed experts then run on the
                             MXFP8 forward; the shared expert and the two latent projections stay BF16);
                             the other inputs as in ``forward``.
    Outputs:
        output:          bfloat16 [num_local_tokens, hidden_size]
        forward_context: CuteMoKLatentForwardContext (for backward_latent_layer)
    """
    _validate_latent_projections(config, workspace, latent_down_weights, latent_up_weights)
    validate_inputs(config, workspace, schedule, x, router_weights)
    routed_input = torch.matmul(x, latent_down_weights.t())
    (y_shared, y_routed), forward_context = forward(
        config,
        workspace,
        schedule,
        x,
        router_weights,
        shared_gate_weights,
        shared_up_weights,
        shared_down_weights,
        routed_gate_weights,
        routed_up_weights,
        routed_down_weights,
        swiglu_limit,
        routed_input=routed_input,
    )
    output = torch.addmm(y_shared, y_routed, latent_up_weights.t())
    return output, CuteMoKLatentForwardContext(forward_context, routed_input, y_routed)


def backward_latent_layer(
    config: CuteMoKConfig,
    workspace: CuteMoKWorkspace,
    schedule: CuteMoKSchedule,
    forward_context: CuteMoKLatentForwardContext,
    grad_output: torch.Tensor,
    x: torch.Tensor,
    router_weights: torch.Tensor,
    latent_down_weights: torch.Tensor,
    latent_up_weights: torch.Tensor,
    shared_gate_weights: torch.Tensor,
    shared_up_weights: torch.Tensor,
    shared_down_weights: torch.Tensor,
    routed_gate_weights: torch.Tensor | tuple[torch.Tensor, ...],
    routed_up_weights: torch.Tensor | tuple[torch.Tensor, ...],
    routed_down_weights: torch.Tensor | tuple[torch.Tensor, ...],
    swiglu_limit: float | None = None,
) -> tuple[torch.Tensor, ...]:
    """Backward of forward_latent_layer.

        routed_grad_output = grad_output @ latent_up_weights                  (GEMM)
        d_latent_up_weights = grad_output^T @ y_routed                 (GEMM)
        (d_x_shared, d_routed_input), ... = backward(..., routed_grad_output, routed_input)   (mega kernel + reductions)
        d_x = d_x_shared + d_routed_input @ latent_down_weights            (one GEMM with the add)
        d_latent_down_weights = d_routed_input^T @ x                       (GEMM)

    Routed weights as in ``backward``: bfloat16 in routed_hidden_size, or MXFP8 tuples -- Gate / Up
    ``(e4m3, scale, e4m3_t, scale_t)``, Down ``(e4m3_t, scale_t)`` (the transposed Down only).

    Outputs: d_x [T, hidden_size], d_router_weights [T, topk] (fp32), d_latent_down_weights
    [routed_hidden_size, hidden_size], d_latent_up_weights [hidden_size, routed_hidden_size], then the routed
    gate / up / down and shared gate / up / down weight gradients as in ``backward``.
    """
    _validate_latent_projections(config, workspace, latent_down_weights, latent_up_weights)
    if not isinstance(forward_context, CuteMoKLatentForwardContext):
        raise TypeError("forward_context must come from forward_latent_layer")
    validate_inputs(config, workspace, schedule, x, router_weights, grad_output)
    routed_grad_output = torch.matmul(grad_output, latent_up_weights)
    d_latent_up_weights = torch.matmul(grad_output.t(), forward_context.y_routed)
    (d_x_shared, d_routed_input), d_router_weights, *weight_grads, _ = backward(
        config,
        workspace,
        schedule,
        forward_context.mok,
        grad_output,
        x,
        router_weights,
        shared_gate_weights,
        shared_up_weights,
        shared_down_weights,
        routed_gate_weights,
        routed_up_weights,
        routed_down_weights,
        swiglu_limit,
        routed_grad_output=routed_grad_output,
        routed_input=forward_context.routed_input,
    )
    d_x = torch.addmm(d_x_shared, d_routed_input, latent_down_weights)
    d_latent_down_weights = torch.matmul(d_routed_input.t(), x)
    return (d_x, d_router_weights, d_latent_down_weights, d_latent_up_weights, *weight_grads)


#: ``main_grads`` in call order: the weights' gradients interleaved shared, routed per matrix.
MAIN_GRAD_NAMES = (
    "shared_gate",
    "routed_gate",
    "shared_up",
    "routed_up",
    "shared_down",
    "routed_down",
)


# The exact types of a plain operand; NoneType admits an absent optional one.
_PLAIN_TENSOR_TYPES = frozenset((torch.Tensor, torch.nn.Parameter, type(None)))


def _can_stand_in_for_buffer(tensor: torch.Tensor) -> bool:
    """Whether a validated caller activation can replace its workspace buffer as the shared expert's
    operand: a plain tensor, 256-byte aligned.  The kernels read it through TMA descriptors; peers
    read the symmetric buffer at routed rows only."""
    return type(tensor) in _PLAIN_TENSOR_TYPES and tensor.data_ptr() % 256 == 0


def _fused_gate_kernels(*tensors: torch.Tensor) -> bool:
    """Whether a gated backward launches the fused Triton kernels (_gate_fused) on these operands.

    The kernels read raw data pointers, so outside
    torch.compile, whose Dynamo traces them, they need plain tensors and no dispatch mode,
    pre-dispatch tracing or torch.func transform.  Otherwise the reference torch operations they
    replace run instead, which dispatch through tensor subclasses and which tracers record.
    """
    if torch.compiler.is_dynamo_compiling():
        return True
    from torch.utils import _python_dispatch

    if (
        torch._C._len_torch_dispatch_stack()
        or (_python_dispatch._is_in_torch_dispatch_mode and torch._C._dispatch_tls_is_dispatch_key_included(torch._C.DispatchKey.PreDispatch))
        or torch._C._are_functorch_transforms_active()
    ):
        return False
    return _PLAIN_TENSOR_TYPES.issuperset(map(type, tensors))


def _gate_fused_module():
    """The fused Triton kernels of the gated backward (``_gate_fused``), imported on first use: Triton is the optional
    ``nvidia-cudnn-frontend[triton]`` dependency, and only a backward that launches these kernels needs it."""
    try:
        from . import _gate_fused
    except ModuleNotFoundError as error:
        if (error.name or "").split(".")[0] != "triton":
            raise
        raise RuntimeError(
            "the backward of a layer with a shared output gate runs fused Triton kernels, and Triton is not "
            "installed; install it (pip install 'nvidia-cudnn-frontend[triton]')"
        ) from error
    return _gate_fused


def _gate_bwd_pre_reference(
    grad_output: torch.Tensor,
    shared_output: torch.Tensor,
    gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The torch operations _gate_fused.gate_bwd_pre replaces: dS = dY * G and dZ = sigmoid_backward(dG, G)
    with dG = (dY * S).sum(-1), at the same BF16 tensor boundaries."""
    d_shared = grad_output * gate
    d_gate = (grad_output * shared_output).sum(-1, keepdim=True)
    return d_shared, torch.ops.aten.sigmoid_backward.default(d_gate, gate)


def _validate_shared_output_gate_weight(weight: torch.Tensor, x: torch.Tensor) -> None:
    """Checks the output-gate weight that forward and backward take: contiguous BF16 [1, H] on x's device."""
    if (
        not isinstance(weight, torch.Tensor)
        or weight.dtype != torch.bfloat16
        or weight.device != x.device
        or tuple(weight.shape) != (1, x.shape[1])
        or not weight.is_contiguous()
    ):
        raise ValueError("shared_output_gate_weight must be contiguous BF16 [1, H] on the input device")


def _shared_output_gate_wgrad(
    d_z: torch.Tensor,
    x: torch.Tensor,
    main_grad: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """The gate weight's FP32 gradient from BF16 d_z and x.

    With ``main_grad`` the contribution is added in place and that buffer returned; a BF16 buffer
    rounds after the addition, not before it.  ``out``: an FP32 [1, H] buffer the contribution is stored
    into instead of a new tensor (backward_execute's).
    """
    contribution = torch.mm(d_z.T, x, out_dtype=torch.float32, out=out)
    if main_grad is None:
        return contribution
    main_grad.add_(contribution)
    return main_grad


def backward(
    config: CuteMoKConfig,
    workspace: CuteMoKWorkspace,
    schedule: CuteMoKSchedule,
    forward_context: CuteMoKForwardContext | None,
    grad_output: torch.Tensor,
    x: torch.Tensor,
    router_weights: torch.Tensor,
    shared_gate_weights: torch.Tensor | None,
    shared_up_weights: torch.Tensor | None,
    shared_down_weights: torch.Tensor | None,
    routed_gate_weights: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    routed_up_weights: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    routed_down_weights: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    swiglu_limit: float | None = None,
    main_grads: tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor] | None = None,
    main_grad_storage_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    routed_grad_output: torch.Tensor | None = None,
    routed_input: torch.Tensor | None = None,
    *,
    shared_output_gate_weight: torch.Tensor | None = None,
    shared_output_gate_main_grad: torch.Tensor | None = None,
    zero_unrouted_router_grads: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Runs the MoE backward pass.

    MXFP8 routed weights run the MXFP8 backward (macrobatch-sized forward context, replayed);
    ``config.comm_tile_shape`` is BF16 only.

    The three shared weights are None exactly when the forward ran routed-only; the shared weight
    gradients are then None too.

    Inputs:
        config:              CuteMoKConfig
        workspace:           CuteMoKWorkspace
        schedule:            CuteMoKSchedule
        forward_context:     CuteMoKForwardContext.  Without routed activations (``without_routed``,
                             ``recompute_forward_context``) the kernel replays the first macrobatch's routed
                             X / Gate / Up as well, a compile-time variant of the backward kernels.  None:
                             nothing was kept from the forward -- the same as passing
                             ``recompute_forward_context(...)`` of these arguments (SwiGLU; a SiTU-GLU layer
                             passes that context, which carries its betas; not for a shared output gate)
        grad_output:         bfloat16 [num_local_tokens, hidden_size]
        x:                   bfloat16 [num_local_tokens, hidden_size]
        router_weights:      float32 [num_local_tokens, topk]
        shared_gate_weights: bfloat16 [intermediate_size, hidden_size]
        shared_up_weights:   bfloat16 [intermediate_size, hidden_size]
        shared_down_weights: bfloat16 [hidden_size, intermediate_size]
        routed_gate_weights: bfloat16 [num_local_experts, intermediate_size, hidden_size], or the MXFP8 tuple
                             (e4m3, scale, e4m3_t, scale_t), or CuteMoKSplitWeight
        routed_up_weights:   the same forms as routed_gate_weights (passing the same object means a
                             combined FC1, as in ``forward``)
        routed_down_weights: bfloat16 [num_local_experts, hidden_size, intermediate_size], or the MXFP8 tuple
                             (e4m3_t, scale_t), or CuteMoKSplitWeight.  An MXFP8 tuple may end with one bool:
                             the columnwise payload is the native TransformerEngine layout
        swiglu_limit:        float | None
        main_grads:          six accumulation buffers ordered MAIN_GRAD_NAMES (shared, routed per
                             matrix: gate, up, down), or None.  With them a weight gradient is added
                             into the caller's buffer and the returned entry aliases it (gradient
                             accumulation fused into the kernel).  A shared slot is None exactly when
                             the layer has no shared expert, and a combined FC1 appears under both the
                             gate and the up slot.
        main_grad_storage_tables:
                             the routed Gate, Up and Down main_grads' per-expert address tables
                             (``cudnn.moe_ep._cutemok.ops.make_routed_d_weight_storage_table``) when they are
                             per-expert allocations: required with split routed weights, which also
                             require main_grads; a combined FC1 passes one table under both Gate and Up.
                             (routed_*_weights and their gradients use routed_hidden_size instead of hidden_size
                             when the two differ)
        routed_grad_output:         bfloat16 [num_local_tokens, routed_hidden_size] | None -- gradient of the
                             routed top-k reduced output (grad_output is then the shared output's gradient)
        routed_input:            bfloat16 [num_local_tokens, routed_hidden_size] | None -- the forward's routed
                             input (replayed by the peers); both are required when routed_hidden_size != hidden_size
        shared_output_gate_weight: the same unchanged BF16 [1, H] weight passed to forward, or None
        shared_output_gate_main_grad: optional BF16/FP32 [1, H] buffer the gate weight's gradient is added into
        zero_unrouted_router_grads: zero d_router_weights where top_experts is -1 (default). A caller that
                             reads d_router_weights only at routed slots may pass False to skip that masking;
                             the unrouted entries are then unspecified (zero, or a gradient an earlier call
                             computed for that slot).

    Outputs:
        d_x:                   bfloat16 [num_local_tokens, hidden_size]; differing hiddens:
                               (d_x_shared [T, hidden_size], d_routed_input [T, routed_hidden_size])
                               -- the caller combines the two
        d_router_weights:      float32 [num_local_tokens, topk]
        d_routed_gate_weights: bfloat16 [num_local_experts, intermediate_size, hidden_size]
        d_routed_up_weights:   bfloat16 [num_local_experts, intermediate_size, hidden_size]
        d_routed_down_weights: bfloat16 [num_local_experts, hidden_size, intermediate_size]
        d_shared_gate_weights: bfloat16 [intermediate_size, hidden_size]
        d_shared_up_weights:   bfloat16 [intermediate_size, hidden_size]
        d_shared_down_weights: bfloat16 [hidden_size, intermediate_size]
        d_shared_output_gate_weight: fresh float32 [1, H], or the shared_output_gate_main_grad
                               buffer it was added into; None when ungated
    """
    return _backward(
        config,
        workspace,
        schedule,
        forward_context,
        grad_output,
        x,
        router_weights,
        shared_gate_weights,
        shared_up_weights,
        shared_down_weights,
        routed_gate_weights,
        routed_up_weights,
        routed_down_weights,
        swiglu_limit,
        main_grads,
        main_grad_storage_tables,
        routed_grad_output,
        routed_input,
        shared_output_gate_weight=shared_output_gate_weight,
        shared_output_gate_main_grad=shared_output_gate_main_grad,
        zero_unrouted_router_grads=zero_unrouted_router_grads,
    )


def _backward(
    config,
    workspace,
    schedule,
    forward_context,
    grad_output,
    x,
    router_weights,
    shared_gate_weights,
    shared_up_weights,
    shared_down_weights,
    routed_gate_weights,
    routed_up_weights,
    routed_down_weights,
    swiglu_limit,
    main_grads,
    main_grad_storage_tables,
    routed_grad_output,
    routed_input,
    *,
    shared_output_gate_weight=None,
    shared_output_gate_main_grad=None,
    zero_unrouted_router_grads=True,
    buffers=None,
):
    """The backward of ``backward`` and of ``backward_execute``: the same checks, staging copy, barriers and
    kernels.

    ``backward`` leaves ``buffers`` None: the scratch, the weight gradients (without ``main_grads``), d_x and
    d_router_weights are allocated per call, the workspace keeps one counter / metadata set per configuration,
    and a kernel variant is compiled when it is first used.  ``backward_execute`` passes ``buffers = (scratch,
    d_weights, storage, out, d_router_weights_out, route_mask, gate_buffers)`` -- the plan's scratch, storage and
    route mask and the caller's output buffers -- so nothing is allocated (BF16: seven scratch tensors and the
    (int32, float32) storage; MXFP8: the flat scratch buffers by name and the int32 counter buffer).  Gated,
    ``gate_buffers`` is (G * dY, dZ, the gate weight gradient's FP32 scratch) of the plan and the caller's
    ``d_w_output_gate_out``; otherwise None.
    """
    _validate_config_type(config)
    if buffers is not None and not isinstance(forward_context, CuteMoKForwardContext):
        raise NotImplementedError("backward_execute takes a forward context with routed activations")
    # The shared output gate: a gated forward context needs the weight back, and only a gated one
    # takes it.
    gate = forward_context.shared_output_gate if isinstance(forward_context, CuteMoKForwardContext) else None
    shared_output = forward_context.shared_output if isinstance(forward_context, CuteMoKForwardContext) else None
    if shared_output_gate_weight is None:
        if gate is not None or shared_output is not None:
            raise ValueError("a gated forward context requires shared_output_gate_weight in backward")
        if shared_output_gate_main_grad is not None:
            raise ValueError("shared output gate main-grad requires shared_output_gate_weight")
    else:
        _validate_shared_output_gate_weight(shared_output_gate_weight, x)
        if gate is None or shared_output is None:
            raise ValueError("gated backward requires saved S/G from the original gated forward context")
        if shared_output_gate_main_grad is not None and (
            not isinstance(shared_output_gate_main_grad, torch.Tensor)
            or shared_output_gate_main_grad.dtype not in (torch.bfloat16, torch.float32)
            or shared_output_gate_main_grad.device != x.device
            or tuple(shared_output_gate_main_grad.shape) != (1, x.shape[1])
            or not shared_output_gate_main_grad.is_contiguous()
        ):
            raise ValueError("shared output gate main-grad must be contiguous BF16/FP32 [1, H] on the " "input device")
    split_hidden = _routed_hidden_differs(workspace)
    if split_hidden:
        for name, tensor in (("routed_grad_output", routed_grad_output), ("routed_input", routed_input)):
            if (
                tensor is None
                or not tensor.is_cuda
                or tensor.device != workspace.device
                or tensor.dtype != torch.bfloat16
                or not tensor.is_contiguous()
                or tuple(tensor.shape) != (workspace.num_local_tokens, workspace.routed_hidden_size)
            ):
                raise ValueError(
                    f"{name} must be a contiguous bfloat16 [num_local_tokens, routed_hidden_size] tensor "
                    "on the workspace device when routed_hidden_size != hidden_size"
                )
    elif (routed_grad_output is not None and routed_grad_output is not grad_output) or (routed_input is not None and routed_input is not x):
        raise ValueError("routed_grad_output / routed_input are only accepted when " "workspace.routed_hidden_size != hidden_size")
    validate_inputs(config, workspace, schedule, x, router_weights, grad_output)
    if forward_context is None:
        # Nothing was kept from the forward: the shared activations are recomputed and the kernels replay every
        # macrobatch's routed ones.
        forward_context = recompute_forward_context(
            config, workspace, schedule, x, shared_gate_weights, shared_up_weights, routed_gate_weights, routed_up_weights, swiglu_limit
        )
    if not isinstance(forward_context, CuteMoKForwardContext):
        raise TypeError("forward_context must be a CuteMoKForwardContext or None")
    from .cutedsl.persistent_bf16_fwd import shared_presence

    has_shared = shared_presence(shared_gate_weights, shared_up_weights, shared_down_weights)
    if (forward_context.gate_shared is not None) != has_shared:
        raise ValueError("shared weight presence must match the forward context")
    # No routed activations in the context: the kernels replay the first macrobatch too (their compile-time
    # recompute_forward_context variant).
    routed_context = (forward_context.x_routed, forward_context.gate_routed, forward_context.up_routed, forward_context.hidden_routed)
    recompute = all(item is None for item in routed_context)
    if not recompute and any(item is None for item in routed_context):
        raise ValueError("the forward context's routed activations must be all present or all None")
    if buffers is not None and recompute:
        raise NotImplementedError("backward_execute takes a forward context with routed activations")
    if split_hidden and not has_shared:
        raise NotImplementedError("a routed-only layer needs routed_hidden_size == hidden_size")
    storage_tables = _routed_storage_tables(routed_gate_weights, routed_up_weights, routed_down_weights)
    # Split routed weights are supported only with fused accumulation into per-expert main_grads
    # (and their address tables).
    if storage_tables is not None and main_grads is None:
        raise ValueError("split routed weights require fused main-grad accumulation")
    if (main_grad_storage_tables is not None) != (storage_tables is not None):
        raise ValueError("split routed weights require three split main-grad storage tables")
    routed_gate_weights, routed_up_weights, up_row_offset = _combined_fc1(routed_gate_weights, routed_up_weights)
    routed_down_weights = _routed_weight_payload(routed_down_weights)
    if storage_tables is not None and isinstance(routed_gate_weights, tuple):
        for name in ("transposed", "scales", "transposed_scales"):
            if getattr(storage_tables, name) is None:
                raise ValueError(
                    "split MXFP8 routed weights need per-expert addresses for the columnwise "
                    "payloads and for both sets of scales (CuteMoKSplitWeight's transposed_* and "
                    "scale_* tables)"
                )
        # A CuteMoKSplitWeight carries both payloads, but the backward's Down operand is the
        # columnwise one alone -- the same (e4m3_t, scale_t, native) triple the tuple form passes.
        routed_down_weights = routed_down_weights[2:]

    # Fused gradient accumulation: the weight gradients belong in the caller's main_grad buffers,
    # so the kernel writes there and nothing is accumulated on the host afterwards.  A combined FC1
    # arrives as one buffer under both the Gate and the Up slot, as the weights do, and each half
    # gets its own descriptor.
    fused_main_grads = None
    main_grad_up_row_offset = 0
    if main_grads is not None:
        if len(main_grads) != len(MAIN_GRAD_NAMES):
            raise ValueError(f"main_grads must have {len(MAIN_GRAD_NAMES)} entries, ordered " f"{MAIN_GRAD_NAMES}; got {len(main_grads)}")
        # A combined FC1's main_grad is one buffer under both the Gate and the Up slot, written
        # whole with Up I rows down.  Split buffers must be combined: one table names both halves.
        combined_main_grad = main_grads[1] is main_grads[3]
        if storage_tables is not None and not (combined_main_grad and main_grad_storage_tables[0] is main_grad_storage_tables[1]):
            raise ValueError("split routed main_grad buffers must pass one combined FC1 buffer, " "and one table, under both the Gate and the Up slot")
        if combined_main_grad:
            rows = int(main_grads[1].shape[-2])
            if rows % 2:
                raise ValueError("a combined FC1 main_grad has 2 * intermediate_size rows")
            main_grad_up_row_offset = rows // 2
        fused_main_grads = tuple(main_grads)

    # Capability errors come before the call touches the workspace (the symmetric buffer copies
    # and the barrier below), like the validation errors above.
    caller_buffers = gate_buffers = None
    if buffers is not None:
        scratch, d_weights, storage, out, d_router_weights_out, route_mask, gate_buffers = buffers
        if (
            type(d_router_weights_out) is not torch.Tensor
            or d_router_weights_out.dtype != torch.float32
            or d_router_weights_out.device != workspace.device
            or not d_router_weights_out.is_contiguous()
            or tuple(d_router_weights_out.shape) != (workspace.num_local_tokens, workspace.topk)
        ):
            raise ValueError("d_router_weights_out must be a contiguous float32 [num_local_tokens, topk] tensor " "on the workspace device")
        # With differing hiddens the shared input gradient is an output (no epilogue merges it): the caller's.
        if isinstance(routed_gate_weights, tuple):
            caller_buffers = (scratch, d_weights, storage, out[0] if split_hidden else None)
        else:
            caller_buffers = ((out[0], *scratch[1:]) if split_hidden else scratch, d_weights, storage)
    if isinstance(routed_gate_weights, tuple):
        # The MXFP8 backward takes the macrobatch-sized context with replay, buffers released per
        # macrobatch (differing hiddens: the shared chain reads x / grad_output, the routed chain the
        # symmetric buffers, as the BF16 backward).
        if isinstance(forward_context.x_routed, tuple) and forward_context.x_routed[0].shape[1] != config.macrobatch_size:
            raise ValueError("the forward context's routed x does not match config.macrobatch_size")
        _require_default_comm_tile_shape(workspace, config)
        # The routed Gate / Up context: e4m3 / scale pairs, or one BF16 tensor each (gate_up_context="bf16").
        if isinstance(forward_context.gate_routed, tuple) != isinstance(forward_context.up_routed, tuple):
            raise TypeError("the routed Gate and Up context must have the same representation")

    # routed sources pulled by the peers (routed-hidden rows)
    d_y_shared = grad_output if split_hidden else None
    fused_gate = gate is not None and _fused_gate_kernels(grad_output, shared_output, gate)
    if gate is not None:
        _gate_fused = _gate_fused_module() if fused_gate else None
        with torch.no_grad():
            # BF16 tensor boundaries as in the torch reference: dS = dY * G uses FP32 opmath, dG sums
            # BF16 products with an FP32 accumulator and dZ = sigmoid_backward(dG, G) rounds like aten
            # on BF16.  The fused kernel makes one pass over grad_output and S; it differs from the
            # reference operations only in the FP32 summation order of dG.
            if fused_gate:
                d_y_shared, d_z = _gate_fused._launch_gate_bwd_pre(grad_output, shared_output, gate, out=None if gate_buffers is None else gate_buffers[:2])
            else:
                d_y_shared, d_z = _gate_bwd_pre_reference(grad_output, shared_output, gate)
    # Peers read d_y_buffer and x_buffer only at routed rows.  With grad_output and x aligned, the
    # shared expert reads them (or the gated dS) itself and only their routed rows are copied;
    # otherwise, or on the MXFP8 / split-hidden paths, the copies are full.
    routed_rows_only = not split_hidden and not isinstance(routed_gate_weights, tuple) and _can_stand_in_for_buffer(grad_output) and _can_stand_in_for_buffer(x)
    if routed_rows_only:
        from .cutedsl.copy_routed_rows import copy_routed_rows

        # The same kernel zeroes the unused slots of the router-weight gradient buffer: the peers write
        # only this call's used slots, after the barrier below, so the buffer needs no masking afterwards.
        copy_routed_rows(
            schedule.top_experts,
            [grad_output, x],
            [workspace.d_y_buffer, workspace.x_buffer],
            router_weights,
            workspace.router_weight_buffer,
            zero_unrouted=workspace.d_router_weight_buffer,
        )
        if d_y_shared is None:
            d_y_shared = grad_output
    else:
        workspace.d_y_buffer.copy_(routed_grad_output if split_hidden else grad_output)
        workspace.x_buffer.copy_(routed_input if split_hidden else x)  # the routed replay's source rows
        workspace.router_weight_buffer.copy_(router_weights)
    ops = _device_ops(config)
    ops.barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    if isinstance(routed_gate_weights, tuple):
        routed_gate_weights, routed_up_weights, routed_down_weights, native_columnwise = _normalize_mxfp8_backward_weights(
            routed_gate_weights, routed_up_weights, routed_down_weights
        )
        from .cutedsl.persistent_mxfp8_bwd import backward_mxfp8 as cutedsl_backward_mxfp8

        if recompute:
            # The host allocates the routed context as scratch, in the configured Gate / Up representation.
            mxfp8_context = dict(gate_shared=forward_context.gate_shared, up_shared=forward_context.up_shared, hidden_shared=forward_context.hidden_shared)
            replay_arguments = dict(recompute_forward_context=True, gate_up_context=config.gate_up_context)
        else:
            x_fp8_t_routed, x_sc_t_routed = forward_context.x_routed
            if isinstance(forward_context.gate_routed, tuple):
                gate_fp8_routed, gate_sc_routed = forward_context.gate_routed
                up_fp8_routed, up_sc_routed = forward_context.up_routed
                gate_up_context = dict(gate_fp8_routed=gate_fp8_routed, gate_sc_routed=gate_sc_routed, up_fp8_routed=up_fp8_routed, up_sc_routed=up_sc_routed)
            else:  # the BF16 Gate / Up context (gate_up_context="bf16")
                gate_up_context = dict(gate_routed_bf16=forward_context.gate_routed, up_routed_bf16=forward_context.up_routed)
            hidden_fp8_t_routed, hidden_sc_t_routed = forward_context.hidden_routed
            mxfp8_context = dict(
                x_fp8_t_routed=x_fp8_t_routed,
                x_sc_t_routed=x_sc_t_routed,
                gate_shared=forward_context.gate_shared,
                up_shared=forward_context.up_shared,
                **gate_up_context,
                hidden_shared=forward_context.hidden_shared,
                hidden_fp8_t_routed=hidden_fp8_t_routed,
                hidden_sc_t_routed=hidden_sc_t_routed,
            )
            replay_arguments = {}

        outputs = cutedsl_backward_mxfp8(
            workspace,
            schedule,
            mxfp8_context,
            shared_gate_weights,
            routed_gate_weights,
            shared_up_weights,
            routed_up_weights,
            shared_down_weights,
            routed_down_weights,
            macrobatch_size=config.macrobatch_size,
            minibatch_size=_bwd_minibatch_size(config),
            num_comm_sms=_bwd_num_comm_sms(config, recompute),
            swiglu_limit=swiglu_limit,
            x_shared=x if split_hidden else None,
            native_columnwise=native_columnwise,
            up_row_offset=up_row_offset,
            # The five weights the dgrad reads, then their five scales: Gate, Up, Gate
            # transposed, Up transposed, Down transposed.
            routed_storage_tables=(
                None
                if storage_tables is None
                else (
                    storage_tables.gate,
                    storage_tables.up,
                    *storage_tables.transposed[:2],
                    storage_tables.transposed[2],
                    *storage_tables.scales[:2],
                    *storage_tables.transposed_scales[:2],
                    storage_tables.transposed_scales[2],
                )
            ),
            main_grads=fused_main_grads,
            main_grad_storage_tables=main_grad_storage_tables,
            main_grad_up_row_offset=main_grad_up_row_offset,
            d_y_shared=d_y_shared,
            situ_beta=forward_context.situ_beta,
            caller_buffers=caller_buffers,
            **replay_arguments,
        )
        (
            d_x_shared,
            d_x_routed,
            d_gate_shared,
            d_gate_fp8_routed,
            d_gate_sc_routed,
            d_up_shared,
            d_up_fp8_routed,
            d_up_sc_routed,
            d_hidden_shared,
            d_hidden_routed,
            d_y_fp8_routed,
            d_y_sc_routed,
            d_w_shared_gate,
            d_w_routed_gate,
            d_w_shared_up,
            d_w_routed_up,
            d_w_shared_down,
            d_w_routed_down,
        ) = outputs
    else:
        from .cutedsl.persistent_bf16_bwd import (
            backward_bf16 as cutedsl_backward_bf16,
        )

        raw = cutedsl_backward_bf16(
            workspace,
            schedule,
            forward_context,
            shared_gate_weights,
            routed_gate_weights,
            shared_up_weights,
            routed_up_weights,
            shared_down_weights,
            routed_down_weights,
            macrobatch_size=config.macrobatch_size,
            minibatch_size=_bwd_minibatch_size(config),
            num_comm_sms=_bwd_num_comm_sms(config, recompute),
            swiglu_limit=swiglu_limit,
            x_shared=x if split_hidden or routed_rows_only else None,
            d_y_shared=d_y_shared,
            routed_storage_tables=(None if storage_tables is None else (storage_tables.gate, storage_tables.up, storage_tables.down)),
            up_row_offset=up_row_offset,
            main_grads=fused_main_grads,
            main_grad_storage_tables=main_grad_storage_tables,
            main_grad_up_row_offset=main_grad_up_row_offset,
            situ_beta=forward_context.situ_beta,
            recompute_forward_context=recompute,
            caller_buffers=caller_buffers,
            comm_tile_shape=config.comm_tile_shape,
        )
        d_x_shared = raw[0]
        d_w_shared_gate, d_w_routed_gate = raw[7], raw[8]
        d_w_shared_up, d_w_routed_up = raw[9], raw[10]
        d_w_shared_down, d_w_routed_down = raw[11], raw[12]
    ops.barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    if split_hidden:
        # The shared and routed input gradients live in different hiddens: reduce the routed top-k
        # rows with the epilogue's routed-only variant and leave the projection to the caller.
        # d_x_shared is this call's own allocation.
        d_x = (d_x_shared, ops.bwd_epilogue(None, workspace.d_x_routed_buffer, schedule.top_experts, out=None if buffers is None else out[1]))
    else:
        d_x = ops.bwd_epilogue(d_x_shared, workspace.d_x_routed_buffer, schedule.top_experts, out=None if buffers is None else out)
    # An unused route slot has no gradient: this call never wrote its buffer entry.  copy_routed_rows
    # zeroed those entries already; the full copies of the other paths leave them to the mask.
    mask_unrouted = zero_unrouted_router_grads and not routed_rows_only
    if buffers is None:
        d_router_weights = workspace.d_router_weight_buffer.clone()  # TODO: we can remove this
        if mask_unrouted:
            d_router_weights.masked_fill_(schedule.top_experts < 0, 0.0)
    else:
        d_router_weights = d_router_weights_out.copy_(workspace.d_router_weight_buffer)
        if mask_unrouted:
            d_router_weights.masked_fill_(torch.lt(schedule.top_experts, 0, out=route_mask[: workspace.num_local_tokens]), 0.0)
    d_w_output_gate = None
    if gate is not None:
        with torch.no_grad():
            d_w_output_gate = _shared_output_gate_wgrad(
                d_z, x, shared_output_gate_main_grad, out=None if gate_buffers is None else gate_buffers[2 if shared_output_gate_main_grad is not None else 3]
            )
            # Deliberate BF16 merge of dZ @ Wg after the original MLP epilogue, in place on its fresh d_x.
            if fused_gate and _fused_gate_kernels(d_x, shared_output_gate_weight):
                d_x = _gate_fused._launch_gate_bwd_post_(d_x, d_z, shared_output_gate_weight)
            elif buffers is None:
                d_x = d_x + d_z @ shared_output_gate_weight
            else:  # backward_execute: the caller's d_x buffer
                d_x.add_(d_z @ shared_output_gate_weight)
    if main_grads is None and up_row_offset:
        # Not accumulating, a combined FC1's gradient is this call's one [E, 2I, H] buffer, with
        # Up written I rows down; it is returned as the Gate and Up halves, the routed weights'
        # own shape.
        d_w_routed_gate = d_w_routed_gate[..., :up_row_offset, :]
        d_w_routed_up = d_w_routed_up[..., up_row_offset:, :]
    return (d_x, d_router_weights, d_w_routed_gate, d_w_routed_up, d_w_routed_down, d_w_shared_gate, d_w_shared_up, d_w_shared_down, d_w_output_gate)


# ---------------------------------------------------------------------------------------------
# Plan / execute entries (transitional names, all in this section)
#
# make_plan compiles every kernel a layer configuration launches and allocates what a call would
# otherwise allocate, its workspace included; schedule_execute / forward_execute / backward_execute then
# run build_schedule / forward / backward into the caller's buffers: they compile nothing and allocate no
# device memory.  They are the same code as the layer functions (_build_schedule, _forward, _backward); only
# the owner of the buffers differs.  BF16 or MXFP8 routed weights (one per plan).
# ---------------------------------------------------------------------------------------------
@dataclass(slots=True)
class CuteMoKPlan:
    """What ``make_plan`` prepared for one layer configuration.

    The plan owns the megakernels' counters, the backward's scratch, the scheduler's scratch and the workspace
    of ``max_tokens_per_rank`` (the symmetric buffers, from ``create_workspace``: not in ``get_workspace``'s
    cache, so the plan holds the only reference and dropping it frees them).  The compiled kernels stay in the
    process caches of their modules.  Calls on one plan must not overlap: they share its buffers, in stream
    order.  Drop a plan only after every call and graph replay on it has finished on every rank of its group
    (other ranks write into its symmetric buffers), outside graph capture.
    """

    workspace: CuteMoKWorkspace  # the buffers, sized for max_tokens_per_rank
    num_local_experts: int
    intermediate_size: int
    has_shared: bool
    macrobatch_size: int  # the largest macrobatch of a backward (MXFP8: of a forward too)
    routed_precision: str  # "bf16" or "mxfp8": the routed weights of every call on the plan
    forward_counters: torch.Tensor  # int32: the forward megakernel's readiness counters at their largest
    forward_scratch: tuple | None  # MXFP8: the forward's e4m3 x / Hidden and their scales, flat, at the
    # largest macrobatch (FORWARD_SCRATCH order); BF16: None
    schedule_scratch: torch.Tensor  # int32: the scheduler's intermediate counts
    backward_scratch: tuple | dict | None  # BF16: the backward's seven scratch tensors at their largest; MXFP8:
    # its flat scratch buffers by name (backward_mxfp8_plan_sizes)
    backward_storage: tuple | torch.Tensor | None  # BF16 (int32, float32): its counters, decoder metadata,
    # router weights / partials; MXFP8: its int32 counters
    route_mask: torch.Tensor | None  # bool [max_tokens_per_rank, topk]: the unrouted slots of d_router_weights
    shared_output_gate: bool  # calls pass shared_output_gate_weight (make_plan's shared_output_gate)
    gate_scratch: tuple | None  # gated with the backward: BF16 G * dY [max_tokens_per_rank, H] and dZ
    # [max_tokens_per_rank, 1], and the FP32 [1, H] gate weight gradient
    # added into a shared_output_gate_main_grad
    _views: dict  # num_local_tokens -> that token count's view of the workspace
    _shapes: dict  # (step, ...) -> _schedule_shapes / _forward_shapes, computed once

    def workspace_for(self, num_local_tokens: int) -> CuteMoKWorkspace:
        """The workspace of calls with ``num_local_tokens`` tokens per rank (at most the plan's maximum; the
        same on every rank of a call, which is the caller's to ensure -- checking it would read back from the
        device): leading views of the plan's buffers, made once per token count and allocating nothing."""
        workspace = self.workspace
        if type(num_local_tokens) is not int:
            raise ValueError("num_local_tokens must be an int")
        if num_local_tokens == workspace.num_local_tokens:
            return workspace
        view = self._views.get(num_local_tokens)
        if view is None:
            if num_local_tokens % 256 or not 512 <= num_local_tokens <= workspace.num_local_tokens:
                raise ValueError(
                    "num_local_tokens must be a multiple of 256 between 512 and the plan's "
                    f"max_tokens_per_rank ({workspace.num_local_tokens}); got {num_local_tokens}"
                )
            rows = num_local_tokens * workspace.topk
            view = self._views[num_local_tokens] = _replace_fields(
                workspace,
                num_local_tokens=num_local_tokens,
                schedule_capacity=workspace.schedule_capacity // workspace.num_local_tokens * num_local_tokens,
                x_buffer=workspace.x_buffer[:num_local_tokens],
                combine_buffer=workspace.combine_buffer[:rows],
                d_y_buffer=workspace.d_y_buffer[:num_local_tokens],
                d_x_routed_buffer=workspace.d_x_routed_buffer[:rows],
                router_weight_buffer=workspace.router_weight_buffer[:num_local_tokens],
                d_router_weight_buffer=workspace.d_router_weight_buffer[:num_local_tokens],
                # Every rank's routes, packed for this token count from the buffer's start.
                all_gather_top_experts_buffer=workspace.all_gather_top_experts_buffer.view(-1)[: workspace.ep_size * rows].view(
                    workspace.ep_size, num_local_tokens, workspace.topk
                ),
                _cutedsl_bwd_preparation=None,
                _cutedsl_bwd_preparations=None,
            )
        return view

    def _schedule_shapes(self, num_local_tokens: int) -> tuple[tuple[int], ...]:
        """The shapes of ``schedule_execute``'s three int32 ``out`` buffers: peer_rank, peer_token_idx, counts."""
        from .cutedsl.scheduler import schedule_buffer_lengths

        key = ("schedule", num_local_tokens)
        shapes = self._shapes.get(key)
        if shapes is None:
            workspace = self.workspace_for(num_local_tokens)
            shapes = self._shapes[key] = tuple(
                (length,) for length in schedule_buffer_lengths(self.num_local_experts, workspace.ep_size, workspace.schedule_capacity)[:3]
            )
        return shapes

    def _forward_shapes(self, config: CuteMoKConfig, num_local_tokens: int) -> tuple[tuple, tuple[int, int]]:
        """The shapes of ``forward_execute``'s buffers: ``(context_out shapes, out shape)``; ``out`` is BF16.

        BF16 routed weights: nine BF16 context shapes in the megakernel's output order (x_routed, gate_shared,
        gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, y_shared, y_routed; the routed ones hold
        ``config.macrobatch_size`` rows).  MXFP8: ``(shape, dtype)`` pairs in the MXFP8 forward's output order,
        thirteen with ``config.gate_up_context="mxfp8"`` (x_routed_t, x_routed_sc_t, gate_shared,
        gate_routed_fp8, gate_routed_sc, up_shared, up_routed_fp8, up_routed_sc, hidden_shared, hidden_routed_t,
        hidden_routed_sc_t, y_shared, y_routed) and eleven with ``"bf16"`` (one BF16 gate_routed / up_routed
        each, no scales); e4m3, uint8 scale tiles or BF16.  None where a routed-only layer has no shared
        output.  A plan with the shared output gate appends G's BF16 [num_local_tokens, 1] buffer."""
        key = ("forward", config, num_local_tokens)
        result = self._shapes.get(key)
        if result is not None:
            return result
        from .cutedsl.persistent_bf16_fwd import SHARED_OUTPUT_INDICES, PersistentBf16Plan, persistent_bf16_output_shapes

        workspace = self.workspace
        if self.routed_precision == "mxfp8":
            from .cutedsl.persistent_mxfp8_fwd import forward_mxfp8_buffer_specs

            outputs, _ = forward_mxfp8_buffer_specs(
                num_local_tokens,
                hidden_size=workspace.hidden_size,
                routed_hidden_size=workspace.routed_hidden_size,
                intermediate_size=self.intermediate_size,
                topk=workspace.topk,
                macrobatch_size=config.macrobatch_size,
                gate_up_context=config.gate_up_context,
                has_shared=self.has_shared,
            )
            if self.shared_output_gate:
                outputs += (((num_local_tokens, 1), torch.bfloat16),)
            result = outputs, (num_local_tokens, workspace.routed_hidden_size)
        else:
            shapes = persistent_bf16_output_shapes(
                num_local_tokens,
                plan=PersistentBf16Plan(
                    ep_size=workspace.ep_size,
                    num_local_experts=self.num_local_experts,
                    hidden_size=workspace.hidden_size,
                    routed_hidden_size=workspace.routed_hidden_size,
                    intermediate_size=self.intermediate_size,
                    topk=workspace.topk,
                    macrobatch_size=config.macrobatch_size,
                    minibatch_size=_fwd_minibatch_size(config),
                    num_comm_sms=config.fwd_num_comm_sms,
                    has_shared=self.has_shared,
                ),
            )
            if not self.has_shared:
                shapes = tuple(None if index in SHARED_OUTPUT_INDICES else shape for index, shape in enumerate(shapes))
            if self.shared_output_gate:
                shapes = (*shapes, (num_local_tokens, 1))
            result = shapes, (num_local_tokens, workspace.routed_hidden_size)
        self._shapes[key] = result
        return result

    def _backward_shapes(self, num_local_tokens: int) -> tuple[object, tuple[int, int], tuple]:
        """The shapes of ``backward_execute``'s buffers: ``(out, d_router_weights_out, d_weights_out)``.
        ``out`` is BF16 [num_local_tokens, hidden_size], or the pair (d_x_shared [.., hidden_size],
        d_routed_input [.., routed_hidden_size]) when the hiddens differ; d_router_weights_out is float32;
        the six BF16 d_weights_out shapes are ordered shared / routed per matrix (gate, up, down), for separate
        grouped Gate / Up weights, None where a routed-only layer has no shared weight.  The same for MXFP8 routed
        weights (their gradients are BF16 too)."""
        workspace = self.workspace
        h, r, i, e = workspace.hidden_size, workspace.routed_hidden_size, self.intermediate_size, self.num_local_experts
        out = (num_local_tokens, h) if h == r else ((num_local_tokens, h), (num_local_tokens, r))
        shared = ((i, h), (i, h), (h, i)) if self.has_shared else (None, None, None)
        return out, (num_local_tokens, workspace.topk), (shared[0], (e, i, r), shared[1], (e, i, r), shared[2], (e, r, i))


def _empty_buffers(shapes, dtype: torch.dtype, device: torch.device):
    """``torch.empty`` buffers of a plan's ``_*_shapes``: a shape, a (shape, dtype) pair (the MXFP8 context, of
    that dtype), None, or a tuple of them (the same structure, of tensors)."""
    if shapes is None:
        return None
    if isinstance(shapes[0], int):
        return torch.empty(shapes, dtype=dtype, device=device)
    if isinstance(shapes[1], torch.dtype):
        return torch.empty(shapes[0], dtype=shapes[1], device=device)
    return tuple(_empty_buffers(item, dtype, device) for item in shapes)


#: The earlier name, kept as an alias.
MoKPlan = CuteMoKPlan


def make_plan(
    config: CuteMoKConfig,
    group: dist.ProcessGroup,
    *,
    device: torch.device,
    max_tokens_per_rank: int,
    hidden_size: int,
    intermediate_size: int,
    num_local_experts: int,
    topk: int,
    routed_hidden_size: int | None = None,
    swiglu_clamped: bool = False,
    situ: bool = False,
    has_shared: bool = True,
    split_routed_weights: bool = False,
    shared_output_gate: bool = False,
    backward: bool = True,
    main_grad_dtype: torch.dtype | None = None,
    routed_precision: str = "bf16",
    native_columnwise: bool = False,
) -> CuteMoKPlan:
    """The plan step, once per layer configuration, called by every rank of the group: compiles every kernel
    the execute steps launch for this configuration and allocates what they would otherwise allocate per call,
    except their outputs, which the caller passes.

    Compiled (over stand-in tensors of the kernels' ranks and dtypes; no caller data): the route all-gather and
    the scheduler; ``barrier_all``; the forward megakernel variant selected by the arguments below and
    ``config.epilogue_combine`` (MXFP8: ``config.gate_up_context``), and the forward epilogue variant; with
    ``backward``, the backward megakernel variant and the backward epilogue variant.  BF16 adds the routed-row
    staging copies and the backward's metadata builder; the MXFP8 backward's tensormap prologue is part of its
    megakernel's compiled call.

    Allocated: the plan's own workspace (``create_workspace``, not cached: symmetric buffers for
    ``max_tokens_per_rank`` and the schedule capacity ``config.schedule_capacity_multiplier`` gives, peer address
    tables), the forward's counters (MXFP8: and its scratch, for ``config.macrobatch_size``), the scheduler's
    scratch and, with ``backward``, its scratch (token- and macrobatch-sized, for ``config.macrobatch_size``), its
    counter / metadata / router-weight storage, the route mask and, gated, the shared output gate's scratch (whose
    fused Triton kernels are compiled here too).

    Token counts, routed rows, macrobatch / minibatch sizes and the communication SMs are not part of the plan:
    they are runtime values of every execute call (a backward's macrobatch, and an MXFP8 forward's, at most the
    plan config's).

    Inputs:
        config:               CuteMoKConfig (``schedule_capacity_multiplier``, ``epilogue_combine``,
                              ``comm_tile_shape``, ``macrobatch_size`` as the backward's bound)
        group:                torch.distributed.ProcessGroup
        device:               torch.device
        max_tokens_per_rank:  int  the largest ``num_local_tokens`` of a call
        hidden_size, intermediate_size, num_local_experts, topk, routed_hidden_size: the layer's sizes
        swiglu_clamped:       whether calls pass a ``swiglu_limit``
        situ:                 whether calls pass ``situ_beta``
        has_shared:           False for a routed-only layer
        split_routed_weights: whether the routed weights are ``CuteMoKSplitWeight``
        shared_output_gate:   whether calls pass ``shared_output_gate_weight`` (every call on the plan does)
        backward:             False for a plan that only runs the schedule and the forward
        main_grad_dtype:      the dtype of the ``main_grads`` backward calls accumulate into (float32 or
                              bfloat16), or None when they store into ``d_weights_out``
        routed_precision:     ``"bf16"``, or ``"mxfp8"`` for MXFP8 routed weights (data / scale tuples or
                              ``CuteMoKSplitWeight`` with scales)
        native_columnwise:    MXFP8: the backward's transposed routed payloads are TransformerEngine's
                              columnwise data (``CuteMoKSplitWeight.native_columnwise``, or the tuples' flag)
    """
    from .cutedsl.all_gather_top_experts import compile_all_gather_top_experts
    from .cutedsl.barrier_all import compile_barrier_all
    from .cutedsl.copy_routed_rows import compile_copy_routed_rows
    from .cutedsl.fwd_epilogue import compile_fwd_epilogue
    from .cutedsl.persistent_bf16_fwd import compile_forward_bf16, forward_bf16_counter_words
    from .cutedsl.scheduler import compile_scheduler, schedule_buffer_lengths

    # The plan's own workspace: not in get_workspace's cache, so dropping the plan frees it.
    workspace = create_workspace(
        config, group, device=device, num_local_tokens=max_tokens_per_rank, hidden_size=hidden_size, topk=topk, routed_hidden_size=routed_hidden_size
    )
    device = workspace.device
    split_hidden = _routed_hidden_differs(workspace)
    if shared_output_gate and (split_hidden or not has_shared):
        raise ValueError("the shared output gate needs a shared expert and routed_hidden_size == hidden_size")
    if split_hidden and not has_shared:
        raise NotImplementedError("a routed-only layer needs routed_hidden_size == hidden_size")
    if split_routed_weights and backward and main_grad_dtype is None:
        raise ValueError("split routed weights require fused main-grad accumulation (main_grad_dtype)")
    if routed_precision not in ("bf16", "mxfp8"):
        raise ValueError("routed_precision must be 'bf16' or 'mxfp8'")
    mxfp8 = routed_precision == "mxfp8"
    if mxfp8 and config.epilogue_combine:
        raise NotImplementedError("epilogue_combine is implemented for the BF16 forward only")
    if mxfp8:
        _require_default_comm_tile_shape(workspace, config)
    if native_columnwise and not mxfp8:
        raise ValueError("native_columnwise is a layout of MXFP8 routed weights")

    def words(length):
        return torch.zeros(length, dtype=torch.int32, device=device)

    forward_scratch = None
    if mxfp8:
        from .cutedsl.persistent_mxfp8_fwd import forward_mxfp8_buffer_specs, forward_mxfp8_counter_words

        sizes = dict(hidden_size=hidden_size, routed_hidden_size=workspace.routed_hidden_size, intermediate_size=intermediate_size, topk=topk)
        forward_counters = words(forward_mxfp8_counter_words(max_tokens_per_rank, workspace.schedule_capacity, **sizes))
        _, scratch_specs = forward_mxfp8_buffer_specs(
            max_tokens_per_rank, **sizes, macrobatch_size=config.macrobatch_size, gate_up_context=config.gate_up_context, has_shared=has_shared
        )
        forward_scratch = tuple(torch.empty(math.prod(shape), dtype=dtype, device=device) for shape, dtype in scratch_specs)
    else:
        forward_counters = words(
            forward_bf16_counter_words(
                max_tokens_per_rank,
                workspace.schedule_capacity,
                hidden_size=hidden_size,
                routed_hidden_size=routed_hidden_size,
                intermediate_size=intermediate_size,
                topk=topk,
            )
        )
    schedule_scratch = words(schedule_buffer_lengths(num_local_experts, workspace.ep_size, workspace.schedule_capacity)[3])
    # Stand-ins of the callers' tensors: one rank's slice of the gathered routes for top_experts, a workspace
    # buffer of routed rows for a [T, H] activation.
    routes = workspace.all_gather_top_experts_buffer[0]
    compile_all_gather_top_experts(routes, workspace.all_gather_top_experts_buffer_multicast_ptr)
    compile_scheduler(workspace.all_gather_top_experts_buffer, schedule_scratch)
    compile_barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    if mxfp8:
        from .cutedsl.persistent_mxfp8_fwd import compile_forward_mxfp8

        compile_forward_mxfp8(
            workspace,
            num_local_experts=num_local_experts,
            intermediate_size=intermediate_size,
            is_clamped=swiglu_clamped,
            is_situ=situ,
            has_shared=has_shared,
            split_weights=split_routed_weights,
            gate_up_context=config.gate_up_context,
        )
    else:
        compile_forward_bf16(
            workspace,
            num_local_experts=num_local_experts,
            is_clamped=swiglu_clamped,
            is_situ=situ,
            has_shared=has_shared,
            split_weights=split_routed_weights,
            epilogue_combine=config.epilogue_combine,
            comm_tile_shape=config.comm_tile_shape,
        )
    if not split_hidden and not mxfp8:
        # The staging copy of aligned inputs (x; with the backward, dY and x); any other, and MXFP8's, is copied
        # whole by torch.
        compile_copy_routed_rows(routes, workspace.x_buffer, workspace.router_weight_buffer)
    # With differing hiddens the epilogues reduce the routed rows alone, as for a routed-only layer.
    merged = has_shared and not split_hidden
    compile_fwd_epilogue(workspace.x_buffer if merged else None, workspace.combine_buffer, workspace.router_weight_buffer, routes, gated=shared_output_gate)
    backward_scratch = backward_storage = route_mask = None
    if backward:
        from .cutedsl.bwd_epilogue import compile_bwd_epilogue

        compile_bwd_epilogue(workspace.d_y_buffer if merged else None, workspace.d_x_routed_buffer, routes)
        route_mask = torch.zeros(max_tokens_per_rank, topk, dtype=torch.bool, device=device)
    if backward and mxfp8:
        from .cutedsl.persistent_mxfp8_bwd import backward_mxfp8_plan_sizes, compile_backward_mxfp8

        compile_backward_mxfp8(
            workspace,
            num_local_experts=num_local_experts,
            intermediate_size=intermediate_size,
            is_clamped=swiglu_clamped,
            is_situ=situ,
            has_shared=has_shared,
            split_weights=split_routed_weights,
            main_grad_dtype=main_grad_dtype,
            native_columnwise=native_columnwise,
            gate_up_context=config.gate_up_context,
        )
        scratch_specs, counter_words = backward_mxfp8_plan_sizes(
            num_local_tokens=max_tokens_per_rank,
            hidden_size=hidden_size,
            routed_hidden_size=workspace.routed_hidden_size,
            intermediate_size=intermediate_size,
            num_local_experts=num_local_experts,
            schedule_capacity=workspace.schedule_capacity,
            macrobatch_size=config.macrobatch_size,
            gate_up_context=config.gate_up_context,
            has_shared=has_shared,
        )
        backward_scratch = {name: torch.empty(numel, dtype=dtype, device=device) for name, numel, dtype in scratch_specs}
        backward_storage = words(counter_words)
    elif backward:
        from .cutedsl.persistent_bf16_bwd import backward_bf16_storage_lengths, compile_backward_bf16

        compile_backward_bf16(
            workspace,
            num_local_experts=num_local_experts,
            intermediate_size=intermediate_size,
            is_clamped=swiglu_clamped,
            is_situ=situ,
            has_shared=has_shared,
            split_weights=split_routed_weights,
            main_grad_dtype=main_grad_dtype,
            comm_tile_shape=config.comm_tile_shape,
        )
        if not split_hidden:
            compile_copy_routed_rows(routes, workspace.x_buffer, workspace.router_weight_buffer, num_pairs=2)
        tokens, macro = max_tokens_per_rank, config.macrobatch_size
        h, r, i = hidden_size, workspace.routed_hidden_size, intermediate_size
        # raw13 0-6: d_x_shared, d_x_routed, d_gate_shared, d_gate_routed, d_up_shared, d_up_routed, d_y_routed.
        # d_x_shared is the caller's output when the hiddens differ; a routed-only layer has no shared ones.
        backward_scratch = tuple(
            None if shape is None else torch.empty(shape, dtype=torch.bfloat16, device=device)
            for shape in (
                (tokens, h) if merged else None,
                (macro, r),
                (tokens, i) if has_shared else None,
                (macro, i),
                (tokens, i) if has_shared else None,
                (macro, i),
                (macro, r),
            )
        )
        int_words, float_words = backward_bf16_storage_lengths(
            workspace,
            num_local_experts=num_local_experts,
            intermediate_size=intermediate_size,
            macrobatch_size=macro,
            minibatch_size=_bwd_minibatch_size(config),
        )
        backward_storage = (words(int_words), torch.zeros(float_words, dtype=torch.float32, device=device))
    gate_scratch = None
    if shared_output_gate and backward:
        gate_scratch = (
            torch.empty(max_tokens_per_rank, hidden_size, dtype=torch.bfloat16, device=device),
            torch.empty(max_tokens_per_rank, 1, dtype=torch.bfloat16, device=device),
            torch.empty(1, hidden_size, dtype=torch.float32, device=device),
        )
        # The gated backward's fused kernels, compiled over rows of the scratch (the row count is their grid).
        d_shared, d_z, _ = gate_scratch
        fused = _gate_fused_module()
        fused._launch_gate_bwd_pre(d_shared[:1], d_shared[:1], d_z[:1], out=(d_shared[1:2], d_z[1:2]))
        fused._launch_gate_bwd_post_(d_shared[1:2], d_z[:1], d_shared[:1])
    return CuteMoKPlan(
        workspace=workspace,
        num_local_experts=num_local_experts,
        intermediate_size=intermediate_size,
        has_shared=has_shared,
        macrobatch_size=config.macrobatch_size,
        routed_precision=routed_precision,
        forward_counters=forward_counters,
        forward_scratch=forward_scratch,
        schedule_scratch=schedule_scratch,
        backward_scratch=backward_scratch,
        backward_storage=backward_storage,
        route_mask=route_mask,
        shared_output_gate=shared_output_gate,
        gate_scratch=gate_scratch,
        _views={},
        _shapes={},
    )


def _plan_workspace(plan, tensor, name: str, routed_weight=None) -> CuteMoKWorkspace:
    if not isinstance(plan, CuteMoKPlan):
        raise TypeError("plan must be a CuteMoKPlan")
    if routed_weight is not None:
        mxfp8 = isinstance(routed_weight, tuple) or (isinstance(routed_weight, CuteMoKSplitWeight) and routed_weight.scale is not None)
        if mxfp8 != (plan.routed_precision == "mxfp8"):
            raise ValueError(f"the plan was made for {plan.routed_precision} routed weights " "(make_plan's routed_precision)")
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
        raise ValueError(f"{name} must be a [num_local_tokens, ...] tensor")
    return plan.workspace_for(int(tensor.shape[0]))


def _require_plan_gate(plan, shared_output_gate_weight) -> None:
    """A gated call needs a plan prepared with the shared output gate, and such a plan takes gated calls only."""
    if shared_output_gate_weight is None and plan.shared_output_gate:
        raise ValueError("the plan was prepared with shared_output_gate=True: pass shared_output_gate_weight")
    if shared_output_gate_weight is not None and not plan.shared_output_gate:
        raise ValueError("shared_output_gate_weight on a plan prepared without the shared output gate (make_plan's shared_output_gate)")


def _require_backward_gate(plan, forward_context, shared_output_gate_weight) -> None:
    """``_require_plan_gate`` for a backward; a gated forward context without the weight raises ``_backward``'s
    own error instead."""
    if shared_output_gate_weight is not None or getattr(forward_context, "shared_output_gate", None) is None:
        _require_plan_gate(plan, shared_output_gate_weight)


def _check_buffer(name: str, tensor, dtype: torch.dtype, shape: tuple, device: torch.device) -> None:
    if type(tensor) is not torch.Tensor or tensor.dtype != dtype or tensor.device != device or tuple(tensor.shape) != shape or not tensor.is_contiguous():
        raise ValueError(f"{name} must be a contiguous {dtype} {list(shape)} tensor on the plan's device")


def schedule_execute(
    plan: CuteMoKPlan, config: CuteMoKConfig, top_experts: torch.Tensor, *, out: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> CuteMoKSchedule:
    """The execute step of ``build_schedule``: into the caller's buffers, compiling and allocating nothing.

    Per call it launches the route all-gather, ``barrier_all`` and the three scheduler kernels.

    Inputs:
        top_experts: int32 [num_local_tokens, topk] (any token count up to the plan's maximum, the same on
                     every rank); -1 marks an unused route slot.  The schedule keeps a reference to it.
        out:         three int32 buffers of ``plan._schedule_shapes(num_local_tokens)``, 16-byte aligned:
                     peer_rank, peer_token_idx and the counts (num_tokens and the per-expert rows are views
                     of it).  They are the returned schedule and must stay untouched until the layer's backward.
    """
    from .cutedsl._host_common import compiled_only

    workspace = _plan_workspace(plan, top_experts, "top_experts")
    with compiled_only():
        return _build_schedule(workspace, config, top_experts, plan.num_local_experts, out=(*out, plan.schedule_scratch))


def forward_execute(
    plan: CuteMoKPlan,
    config: CuteMoKConfig,
    schedule: CuteMoKSchedule,
    x: torch.Tensor,
    router_weights: torch.Tensor,
    shared_gate_weights: torch.Tensor | None,
    shared_up_weights: torch.Tensor | None,
    shared_down_weights: torch.Tensor | None,
    routed_gate_weights: torch.Tensor | CuteMoKSplitWeight,
    routed_up_weights: torch.Tensor | CuteMoKSplitWeight,
    routed_down_weights: torch.Tensor | CuteMoKSplitWeight,
    swiglu_limit: float | None = None,
    routed_input: torch.Tensor | None = None,
    *,
    context_out: tuple[torch.Tensor | None, ...],
    out: torch.Tensor,
    shared_output_gate_weight: torch.Tensor | None = None,
    situ_beta: tuple[float, float] | None = None,
) -> tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], CuteMoKForwardContext]:
    """The execute step of ``forward``: into the caller's buffers, compiling nothing (a kernel variant the plan
    did not compile raises) and allocating no device memory.

    Two things happen once rather than per call: the first call over new weight tensors checks them and builds
    the host-side launch arguments; the first call over new ``CuteMoKSplitWeight`` address tables allocates and
    builds their TMA descriptors (one builder launch per routed operand, once per set of tables, as in
    ``forward``).

    Per call it launches, in order: the staging copy of the routed rows of ``x`` and of the router weights into
    the symmetric buffers (BF16: one kernel when ``x`` is 256-byte aligned and the hiddens coincide; otherwise,
    and with MXFP8 weights, two ``copy_``), ``barrier_all``, the zeroing of the plan's counters, the forward
    megakernel, ``barrier_all``, gated the GEMM and sigmoid of G, and the forward epilogue.

    The inputs are ``forward``'s, with ``x`` of any token count up to the plan's maximum (the same on every
    rank) and ``schedule`` from ``schedule_execute`` (or ``build_schedule(plan.workspace_for(tokens), ...)``), and:

        context_out:        the megakernel outputs' buffers (``plan._forward_shapes``): nine BF16, or with
                            MXFP8 weights thirteen / eleven (``config.gate_up_context``) of their dtypes;
                            contiguous, 16-byte aligned; they become the returned context's tensors and must
                            stay untouched until its backward (MXFP8: a macrobatch at most the plan config's);
                            a gated plan's last one holds G = sigmoid(x @ w^T), BF16 [num_local_tokens, 1]
        out:                the output's buffer, BF16 [num_local_tokens, routed_hidden_size]: the layer
                            output, or y_routed when the routed hidden differs (y_shared is context_out[7])
        shared_output_gate_weight: ``forward``'s; given exactly when the plan has the shared output gate

    Outputs: as ``forward``, over the caller's buffers.
    """
    from .cutedsl._host_common import compiled_only

    workspace = _plan_workspace(plan, x, "x", routed_gate_weights)
    if not isinstance(out, torch.Tensor) or context_out is None:
        raise ValueError("forward_execute needs the caller's context_out and out buffers")
    _require_plan_gate(plan, shared_output_gate_weight)
    gate_out = None
    if plan.shared_output_gate:
        *context_out, gate_out = context_out
        _check_buffer("context_out's last buffer (G)", gate_out, torch.bfloat16, (workspace.num_local_tokens, 1), workspace.device)
    with compiled_only():
        return _forward(
            config,
            workspace,
            schedule,
            x,
            router_weights,
            shared_gate_weights,
            shared_up_weights,
            shared_down_weights,
            routed_gate_weights,
            routed_up_weights,
            routed_down_weights,
            swiglu_limit,
            routed_input,
            situ_beta=situ_beta,
            shared_output_gate_weight=shared_output_gate_weight,
            buffers=(tuple(context_out), out, plan.forward_counters, plan.forward_scratch, gate_out),
        )


def backward_execute(
    plan: CuteMoKPlan,
    config: CuteMoKConfig,
    schedule: CuteMoKSchedule,
    forward_context: CuteMoKForwardContext,
    grad_output: torch.Tensor,
    x: torch.Tensor,
    router_weights: torch.Tensor,
    shared_gate_weights: torch.Tensor | None,
    shared_up_weights: torch.Tensor | None,
    shared_down_weights: torch.Tensor | None,
    routed_gate_weights: torch.Tensor | CuteMoKSplitWeight,
    routed_up_weights: torch.Tensor | CuteMoKSplitWeight,
    routed_down_weights: torch.Tensor | CuteMoKSplitWeight,
    swiglu_limit: float | None = None,
    main_grads: tuple | None = None,
    main_grad_storage_tables: tuple | None = None,
    routed_grad_output: torch.Tensor | None = None,
    routed_input: torch.Tensor | None = None,
    *,
    out: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    d_router_weights_out: torch.Tensor,
    d_weights_out: tuple[torch.Tensor | None, ...] | None = None,
    zero_unrouted_router_grads: bool = True,
    shared_output_gate_weight: torch.Tensor | None = None,
    shared_output_gate_main_grad: torch.Tensor | None = None,
    d_w_output_gate_out: torch.Tensor | None = None,
) -> tuple:
    """The execute step of ``backward``: into the caller's buffers, compiling nothing and allocating no device
    memory.

    Once rather than per call, as in ``forward_execute``: the host-side launch arguments of new weight tensors
    or of a new geometry (token count, macrobatch / minibatch size, communication SMs), carved out of the
    plan's storage; the TMA descriptors of new ``CuteMoKSplitWeight`` / main_grad address tables.

    Per call it launches, in order: the staging copy of the routed rows of ``grad_output`` and ``x`` and of the
    router weights (BF16: one kernel when both are 256-byte aligned and the hiddens coincide; otherwise, and
    with MXFP8 weights, three ``copy_``), ``barrier_all``, the zeroing of the plan's counters, the metadata
    builder (BF16) or the tensormap prologue (MXFP8), the backward megakernel (MXFP8 without ``main_grads``: and
    the zeroing of empty experts' weight gradients, in the same compiled call), ``barrier_all``, the backward
    epilogue, the copy of the router-weight gradients and, with ``zero_unrouted_router_grads``, a compare and a
    masked fill of their unrouted slots.  Gated, the fused Triton kernel of G * dY and dZ comes before the staging
    copy, and the gate weight's gradient GEMM (with a main_grad: and its addition) and the fused merge of
    dZ @ w into d_x follow the epilogue; G * dY and dZ are the plan's scratch.  (Under a tensor subclass, dispatch
    mode or torch.func transform the fused kernels' reference torch operations run instead, and allocate.)

    The inputs are ``backward``'s -- a forward context with its routed activations, a macrobatch at most the
    plan config's; ``shared_output_gate_weight`` given exactly when the plan has the shared output gate -- and:

        out:                  the input gradient's buffer(s) (``plan._backward_shapes``): BF16
                              [num_local_tokens, hidden_size], or (d_x_shared, d_routed_input) when the
                              hiddens differ
        d_router_weights_out: float32 [num_local_tokens, topk]
        d_weights_out:        without ``main_grads``: the six weight gradients' buffers, shaped like the
                              weights and ordered shared / routed per matrix (gate, up, down), None where a
                              routed-only layer has no shared weight; a combined FC1 passes its one
                              [E, 2I, H] buffer as both the routed gate and up entries.  The kernel stores
                              into them.  With ``main_grads`` the gradients are accumulated there instead.
        d_w_output_gate_out:  gated without ``shared_output_gate_main_grad``: FP32 [1, hidden_size], the gate
                              weight's gradient; None otherwise

    Outputs: as ``backward``, over the caller's buffers (the last entry, the gate weight's gradient, is
    ``d_w_output_gate_out``, the ``shared_output_gate_main_grad`` it was added into, or None ungated).
    """
    from .cutedsl._host_common import compiled_only

    workspace = _plan_workspace(plan, x, "x", routed_gate_weights)
    if plan.backward_scratch is None:
        raise ValueError("the plan was made without the backward (make_plan(backward=False))")
    if (main_grads is None) == (d_weights_out is None):
        raise ValueError("backward_execute takes main_grads or d_weights_out, not both")
    _require_backward_gate(plan, forward_context, shared_output_gate_weight)
    gate_buffers = None
    if shared_output_gate_weight is not None:
        if (shared_output_gate_main_grad is None) == (d_w_output_gate_out is None):
            raise ValueError("a gated backward_execute takes shared_output_gate_main_grad or d_w_output_gate_out, not both")
        if d_w_output_gate_out is not None:
            _check_buffer("d_w_output_gate_out", d_w_output_gate_out, torch.float32, (1, workspace.hidden_size), workspace.device)
        tokens = workspace.num_local_tokens
        d_shared, d_z, contribution = plan.gate_scratch
        gate_buffers = (d_shared[:tokens], d_z[:tokens], contribution, d_w_output_gate_out)
    elif d_w_output_gate_out is not None:
        raise ValueError("d_w_output_gate_out is the gradient of shared_output_gate_weight, which is None")
    with compiled_only():
        return _backward(
            config,
            workspace,
            schedule,
            forward_context,
            grad_output,
            x,
            router_weights,
            shared_gate_weights,
            shared_up_weights,
            shared_down_weights,
            routed_gate_weights,
            routed_up_weights,
            routed_down_weights,
            swiglu_limit,
            main_grads,
            main_grad_storage_tables,
            routed_grad_output,
            routed_input,
            shared_output_gate_weight=shared_output_gate_weight,
            shared_output_gate_main_grad=shared_output_gate_main_grad,
            zero_unrouted_router_grads=zero_unrouted_router_grads,
            buffers=(plan.backward_scratch, d_weights_out, plan.backward_storage, out, d_router_weights_out, plan.route_mask, gate_buffers),
        )
