# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Python API surface for fused SwiGLU MoE with expert parallelism.

The public API performs contract validation and dispatches through a private,
lazy backend seam. Device-runtime implementation details remain outside this
module.
"""

from __future__ import annotations

import contextlib
import math
import threading
import warnings
from numbers import Real
from typing import Literal, Optional, Tuple, Union

import torch
import torch.distributed as dist

from ._contracts import ForwardConfig, ValidatedBackwardRequest
from ._tuning import MoeEpTuningConfig
from ._types import (
    BlockScaledTensor,
    MoeEpWgradForwardStash,
    MoeEpWgradOperands,
    MoeFormat,
    MoeTensor,
    parse_format as _parse_format,
)
from ._validation import validate_backward, validate_forward

def _resolve_ep_topology(
    ep_group: Optional[dist.ProcessGroup],
) -> tuple[int, int, tuple[int, ...]]:
    """Return dense EP rank/size plus its ordered global-rank membership."""

    if ep_group is None:
        return 1, 0, ()
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "ep_group requires an initialized torch.distributed process group"
        )

    ep_size = dist.get_world_size(ep_group)
    ep_rank = dist.get_rank(ep_group)
    if ep_size <= 0 or ep_rank < 0 or ep_rank >= ep_size:
        raise ValueError("the current process must be a member of ep_group")

    ep_global_ranks = tuple(
        dist.get_global_rank(ep_group, group_rank)
        for group_rank in range(ep_size)
    )
    if len(set(ep_global_ranks)) != ep_size:
        raise RuntimeError("ep_group returned duplicate global ranks")
    if ep_global_ranks[ep_rank] != dist.get_rank():
        raise RuntimeError(
            "ep_group rank mapping is inconsistent with the current global rank"
        )
    return ep_size, ep_rank, ep_global_ranks


class MoeEp:
    """Fused SwiGLU MoE operator with contiguous expert parallel sharding.

    Global expert ``e`` belongs to group-relative EP rank
    ``e // experts_per_rank``.  The constructor captures static configuration;
    calling the instance accepts runtime tensors for this rank.

    The Rubin training-Mega backend accepts plain BF16/FP16/FP32 operands
    (staged to MXFP8 E4M3) or MXFP8 ``BlockScaledTensor`` operands. Final
    output is BF16. ``combine_format`` may be BF16 or MXFP8; forward MXFP8
    combine quantizes each FP32 route accumulator directly before top-k
    reduction. The Rubin training backend requires
    ``apply_topk_in_fc1=True``.
    Native NVFP4 operands and NVFP4 combine/output are not executable.

    With ``generate_c=True`` (training integration), ``__call__`` additionally
    returns ``fc1_c`` and ``route_metadata``.  ``fc1_c`` is the raw pre-SwiGLU
    FC1 accumulator for every route this rank's experts processed, BF16, shape
    ``(local_routes, 2 * intermediate)``.  Rows are grouped by local expert
    (ascending) and ordered within each expert by source rank, then the source
    rank's token-major route order.  The rows are captured before the gate/up
    clamp and carry no router weight.  ``route_metadata`` is Int32
    ``(local_routes, 4)`` with columns
    ``(local_expert, src_rank, src_token, src_slot)``, row-aligned with
    ``fc1_c``, identifying each route for the backward gradient re-dispatch.

    With ``backward_wgrad_mode="operands"``, ``generate_c=True``,
    ``token_padding_size=256``, and ``sf_padding_size=128`` are required.
    Forward additionally returns a caller-owned
    :class:`MoeEpWgradForwardStash`; backward accepts that exact routed-call
    stash by keyword and additionally returns :class:`MoeEpWgradOperands`.
    This opt-in path is available under the Rubin MXFP8 backward capability
    gates documented below.

    The backend is created lazily on the first supported forward call. Valid
    combinations outside the current backend capability matrix fail explicitly
    instead of returning uninitialized storage. Once created, a backend and its
    workspaces are bound to that call's device; use a separate ``MoeEp``
    instance for another device.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int,
        ep_group: Optional[dist.ProcessGroup] = None,
        max_tokens_per_rank: Optional[int] = None,
        output_format: Union[MoeFormat, str] = MoeFormat.BF16,
        combine_format: Union[MoeFormat, str] = MoeFormat.BF16,
        apply_topk_in_fc1: bool = True,
        gate_up_clamp: Optional[float] = None,
        generate_c: bool = False,
        backward_wgrad_mode: Literal["none", "operands"] = "none",
        token_padding_size: int = 128,
        sf_padding_size: int = 128,
        tuning: Optional[MoeEpTuningConfig] = None,
    ) -> None:
        self._lifecycle_lock = threading.RLock()
        for name, value in (
            ("num_experts", num_experts),
            ("hidden_size", hidden_size),
            ("intermediate_size", intermediate_size),
            ("top_k", top_k),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) cannot exceed num_experts ({num_experts})")
        if max_tokens_per_rank is not None and (
            isinstance(max_tokens_per_rank, bool)
            or not isinstance(max_tokens_per_rank, int)
            or max_tokens_per_rank < 0
        ):
            raise ValueError(
                "max_tokens_per_rank must be a non-negative integer or None"
            )
        if not isinstance(apply_topk_in_fc1, bool):
            raise ValueError("apply_topk_in_fc1 must be a bool")
        if not isinstance(generate_c, bool):
            raise ValueError("generate_c must be a bool")
        if backward_wgrad_mode not in ("none", "operands"):
            raise ValueError(
                "backward_wgrad_mode must be 'none' or 'operands', "
                f"got {backward_wgrad_mode!r}"
            )
        if backward_wgrad_mode == "operands" and not generate_c:
            raise ValueError(
                "backward_wgrad_mode='operands' requires generate_c=True"
            )
        for name, value in (
            ("token_padding_size", token_padding_size),
            ("sf_padding_size", sf_padding_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"{name} must be a positive integer, got {value!r}"
                )
        if backward_wgrad_mode == "operands" and token_padding_size != 256:
            raise ValueError(
                "backward_wgrad_mode='operands' requires "
                "token_padding_size=256"
            )
        if backward_wgrad_mode == "operands" and sf_padding_size != 128:
            raise ValueError(
                "backward_wgrad_mode='operands' requires "
                "sf_padding_size=128"
            )
        if sf_padding_size % 128:
            raise ValueError(
                "sf_padding_size must be a positive multiple of 128, "
                f"got {sf_padding_size}"
            )
        if tuning is not None and not isinstance(tuning, MoeEpTuningConfig):
            raise TypeError(
                "tuning must be a MoeEpTuningConfig or None, "
                f"got {type(tuning).__name__}"
            )
        if gate_up_clamp is not None:
            if isinstance(gate_up_clamp, bool) or not isinstance(gate_up_clamp, Real):
                raise ValueError("gate_up_clamp must be a finite real number or None")
            gate_up_clamp = float(gate_up_clamp)
            if not math.isfinite(gate_up_clamp):
                raise ValueError("gate_up_clamp must be a finite real number or None")

        if ep_group is not None and not isinstance(ep_group, dist.ProcessGroup):
            raise ValueError(
                f"ep_group must be a torch.distributed.ProcessGroup or None, "
                f"got {type(ep_group).__name__}"
            )
        ep_size, ep_rank, ep_global_ranks = _resolve_ep_topology(ep_group)
        if num_experts % ep_size != 0:
            raise ValueError(f"num_experts ({num_experts}) must be divisible by EP size ({ep_size})")

        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.ep_group = ep_group
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.ep_global_ranks = ep_global_ranks
        self.experts_per_rank = num_experts // ep_size
        self.max_tokens_per_rank = max_tokens_per_rank
        self.output_format = _parse_format(output_format)
        self.combine_format = _parse_format(combine_format)
        self.apply_topk_in_fc1 = apply_topk_in_fc1
        self.gate_up_clamp = None if gate_up_clamp is None else abs(gate_up_clamp)
        self.generate_c = generate_c
        self.backward_wgrad_mode = backward_wgrad_mode
        self.token_padding_size = token_padding_size
        self.sf_padding_size = sf_padding_size
        self.tuning = MoeEpTuningConfig() if tuning is None else tuning
        if self.tuning.reduce_topk_in_kernel and (
            self.combine_format is not MoeFormat.BF16
            or self.output_format is not MoeFormat.BF16
            or not self.apply_topk_in_fc1
        ):
            raise ValueError(
                "reduce_topk_in_kernel requires BF16 combine/output and "
                "apply_topk_in_fc1=True"
            )

        for name, fmt in (
            ("output_format", self.output_format),
            ("combine_format", self.combine_format),
        ):
            required_multiple = 32 if fmt is MoeFormat.MXFP8 else 16 if fmt is MoeFormat.NVFP4 else 1
            if hidden_size % required_multiple != 0:
                raise ValueError(f"hidden_size ({hidden_size}) must be divisible by " f"{required_multiple} for {name}={fmt.value}")

        self._forward_config = ForwardConfig(
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            top_k=self.top_k,
            experts_per_rank=self.experts_per_rank,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            ep_group=self.ep_group,
            ep_global_ranks=self.ep_global_ranks,
            max_tokens_per_rank=self.max_tokens_per_rank,
            output_format=self.output_format.value,
            combine_format=self.combine_format.value,
            apply_topk_in_fc1=self.apply_topk_in_fc1,
            gate_up_clamp=self.gate_up_clamp,
            generate_c=self.generate_c,
            token_padding_size=self.token_padding_size,
            sf_padding_size=self.sf_padding_size,
            tuning=self.tuning,
            backward_wgrad_mode=self.backward_wgrad_mode,
        )
        self._forward_backend = None
        self._forward_backend_device = None
        self._validated_topk_idx = None
        self._validated_topk_version = None
        self._closed = False

    @staticmethod
    def _tensor_version(tensor: torch.Tensor) -> int | None:
        if not isinstance(tensor, torch.Tensor):
            return None
        try:
            return tensor._version
        except RuntimeError:
            return None

    def _get_backend(self, request, *, backward: bool):
        """Create and cache the private backend on first supported use."""

        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("MoeEp is closed")
            from . import _backend

            if (
                self._forward_backend is not None
                and request.device != self._forward_backend_device
            ):
                raise ValueError(
                    f"MoeEp backend is bound to {self._forward_backend_device}; "
                    f"create a separate MoeEp instance for {request.device}"
                )

            _backend.validate_config(self._forward_config)
            if backward:
                _backend.validate_backward_request(request)
            else:
                _backend.validate_request(request)

            if self._forward_backend is None:
                self._forward_backend = _backend.create_backend(
                    self._forward_config,
                    request.device,
                )
                self._forward_backend_device = request.device
            return self._forward_backend

    def _count_local_routes(self, request: ValidatedBackwardRequest) -> int:
        """Number of valid routes this rank's experts receive.

        The request already passed expert-id validation. Data-dependent:
        single-rank counts locally, while EP exchanges per-rank route counts
        (the same exchange the device dispatch performs).
        """

        flat = request.topk_idx.reshape(-1).to(torch.int64)
        expert = flat[flat != -1]
        if self.ep_size == 1:
            return int(expert.numel())
        destination = torch.div(expert, self.experts_per_rank, rounding_mode="floor")
        send_counts = torch.bincount(destination, minlength=self.ep_size)
        if send_counts.device.type != "cpu" and dist.get_backend(self.ep_group) == "gloo":
            send_counts = send_counts.cpu()
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)
        return int(recv_counts.sum().item())

    def __call__(
        self,
        activation: MoeTensor,
        fc1_weight: MoeTensor,
        fc2_weight: MoeTensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Union[
        MoeTensor,
        Tuple[MoeTensor, torch.Tensor, torch.Tensor],
        Tuple[
            MoeTensor,
            torch.Tensor,
            torch.Tensor,
            MoeEpWgradForwardStash,
        ],
    ]:
        """Validate and dispatch one fused MoE+EP forward call.

        Expected logical shapes are ``activation=(T,H)``,
        ``fc1_weight=(E_local,H,2I)``, ``fc2_weight=(E_local,I,H)``, and
        ``topk_idx=topk_weights=(T,K)``.

        Returns the ``(T, H)`` result, or ``(result, fc1_c, route_metadata)``
        when constructed with ``generate_c=True``. In
        ``backward_wgrad_mode="operands"``, the latter tuple has a fourth
        ``MoeEpWgradForwardStash`` item.
        """

        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("MoeEp is closed")
            topk_version = self._tensor_version(topk_idx)
            validate_expert_ids = not (
                self._validated_topk_idx is topk_idx
                and topk_version is not None
                and topk_version == self._validated_topk_version
            )
            request = validate_forward(
                self._forward_config,
                activation,
                fc1_weight,
                fc2_weight,
                topk_idx,
                topk_weights,
                validate_expert_ids=validate_expert_ids,
            )
            version_after_validation = self._tensor_version(topk_idx)
            if (
                topk_version is not None
                and topk_version == version_after_validation
            ):
                self._validated_topk_idx = topk_idx
                self._validated_topk_version = topk_version
            else:
                self._validated_topk_idx = None
                self._validated_topk_version = None
            return self._get_backend(request, backward=False).forward(request)

    def warmup(
        self,
        activation: MoeTensor,
        fc1_weight: MoeTensor,
        fc2_weight: MoeTensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        """Prepare a forward plan for CUDA Graph capture.

        This runs one complete eager forward and synchronizes its CUDA device,
        forcing runtime bootstrap, symmetric allocation, weight staging, JIT
        compilation, and the first real kernel launch to finish before capture.

        For expert-parallel execution this method is collective by contract:
        every rank in ``ep_group`` must call it concurrently with valid inputs.
        It intentionally does not issue a process-group barrier; callers should
        align all ranks after warmup and replay captured graphs in lockstep.
        """

        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("MoeEp is closed")
            output = self(
                activation,
                fc1_weight,
                fc2_weight,
                topk_idx,
                topk_weights,
            )
            del output
            device = activation.device
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    def backward(
        self,
        grad_output: torch.Tensor,
        fc1_weight: MoeTensor,
        fc2_weight: MoeTensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        fc1_c: torch.Tensor,
        route_metadata: torch.Tensor,
        *,
        wgrad_forward_stash: Optional[MoeEpWgradForwardStash] = None,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, MoeEpWgradOperands],
    ]:
        """Validate and dispatch one device MoE backward call.

        Requires ``generate_c=True``; consumes the forward stash
        (``fc1_c``, ``route_metadata``) plus the re-supplied weights and
        routing inputs. Returns float32
        ``(grad_activation, grad_topk_weights)``. In
        ``backward_wgrad_mode="operands"``, ``wgrad_forward_stash`` is
        required and the return tuple has a third ``MoeEpWgradOperands`` item.
        The Rubin MXFP8 device path supports BF16/MXFP8 combine and BF16 output
        for any positive EP size under its documented capability gates.
        Backward has hardware acceptance coverage at EP1/EP2/EP4. Forward and
        backward both quantize each FP32 route accumulator directly to MXFP8
        before top-k reduction.
        """

        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("MoeEp is closed")
            if not self.generate_c:
                raise RuntimeError(
                    "backward requires the operator to be constructed with "
                    "generate_c=True"
                )

            backward_request = validate_backward(
                self._forward_config,
                grad_output,
                fc1_weight,
                fc2_weight,
                topk_idx,
                topk_weights,
                fc1_c,
                route_metadata,
                wgrad_forward_stash=wgrad_forward_stash,
            )
            local_routes = self._count_local_routes(backward_request)
            if backward_request.local_routes != local_routes:
                raise ValueError(
                    "route_metadata row count must match the routes received "
                    "from the re-supplied topk_idx"
                )
            return self._get_backend(backward_request, backward=True).backward(
                backward_request
            )

    def close(self) -> None:
        """Release compiled-backend instance resources; idempotent."""

        with self._lifecycle_lock:
            if self._closed:
                return
            if self._forward_backend is not None:
                close_backend = getattr(self._forward_backend, "close", None)
                if close_backend is not None:
                    close_backend()
                self._forward_backend = None
                self._forward_backend_device = None
            self._validated_topk_idx = None
            self._validated_topk_version = None
            self._closed = True

    def __enter__(self) -> "MoeEp":
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("MoeEp is closed")
            return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        del exc_type, exc_value, traceback
        self.close()
        return False

    def __del__(self) -> None:
        if not hasattr(self, "_closed"):
            return
        try:
            self.close()
        except Exception as exc:
            # Explicit close propagates cleanup failures. During GC there is no
            # safe global point to retry CUDA/NVSHMEM teardown, so report the
            # failure without retaining the backend indefinitely.
            with contextlib.suppress(Exception):
                warnings.warn(
                    f"MoeEp finalizer could not release backend resources: {exc}",
                    ResourceWarning,
                    stacklevel=2,
                )


__all__ = [
    "BlockScaledTensor",
    "MoeEp",
    "MoeEpWgradForwardStash",
    "MoeEpWgradOperands",
    "MoeFormat",
    "MoeTensor",
]
