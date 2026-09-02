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
from dataclasses import replace
from numbers import Real
from typing import Mapping, Optional, Sequence, Union

import torch
import torch.distributed as dist

from ._contracts import Fc1WeightLayout, ForwardConfig, normalize_fc1_weight_layout
from ._tuning import (
    MoeEpAutotuneCandidateResult,
    MoeEpAutotuneResult,
    MoeEpTuningConfig,
)
from ._types import (
    BlockScaledTensor,
    MoeEpBackwardWeightStaging,
    MoeEpBackwardWeights,
    MoeEpExecutionLane,
    MoeEpForwardWeightStaging,
    MoeEpForwardWeights,
    MoeEpNativeBackwardWeights,
    MoeEpNativeForwardWeights,
    MoeEpTrainingBackwardOutputs,
    MoeEpTrainingForwardOutputs,
    MoeEpTrainingWgradOperands,
    MoeFormat,
    MoeTensor,
    parse_format as _parse_format,
)
from ._validation import (
    validate_backward_source_weights,
    validate_forward,
    validate_forward_source_weights,
    validate_native_backward_weights,
    validate_native_forward_weights,
    validate_training_backward_outputs,
    validate_training_forward_outputs,
    validate_training_forward_state,
    validate_training_input,
    validate_training_non_aliasing,
)


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
        dist.get_global_rank(ep_group, group_rank) for group_rank in range(ep_size)
    )
    if len(set(ep_global_ranks)) != ep_size:
        raise RuntimeError("ep_group returned duplicate global ranks")
    if ep_global_ranks[ep_rank] != dist.get_rank():
        raise RuntimeError(
            "ep_group rank mapping is inconsistent with the current global rank"
        )
    return ep_size, ep_rank, ep_global_ranks


def _validate_training_assert_capability(config: ForwardConfig) -> None:
    """Fail before allocation when graph error-mode primitives are unavailable."""

    if config.drop_on_overflow:
        return
    if not callable(getattr(torch, "_assert_async", None)):
        raise RuntimeError(
            "drop_on_overflow=False training requires callable "
            "torch._assert_async before CUDA Graph capture"
        )
    if config.ep_size <= 1:
        return
    backend = dist.get_backend(config.ep_group)
    if backend != dist.Backend.NCCL and str(backend).lower() != "nccl":
        raise NotImplementedError(
            "drop_on_overflow=False EP2+ training requires an NCCL "
            "process group for the captured scalar global overflow OR"
        )


def _resolve_training_device(
    device: torch.device | str | int | None,
) -> torch.device:
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("prepare_training requires an available CUDA device")
        return torch.device("cuda", torch.cuda.current_device())
    if isinstance(device, bool):
        raise TypeError("device must be a CUDA device, ordinal, or None")
    if isinstance(device, int):
        resolved = torch.device("cuda", device)
    else:
        resolved = torch.device(device)
        if resolved.type == "cuda" and resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
    if resolved.type != "cuda":
        raise ValueError(f"training device must be CUDA, got {resolved}")
    if (
        resolved.index is None
        or resolved.index < 0
        or resolved.index >= torch.cuda.device_count()
    ):
        raise ValueError(f"CUDA device {resolved} is not available")
    return resolved


def _named_moe_tensors(
    name: str,
    value: MoeTensor,
) -> dict[str, torch.Tensor]:
    if isinstance(value, BlockScaledTensor):
        return {
            f"{name}.data": value.data,
            f"{name}.scale": value.scale,
        }
    return {name: value}


def pack_forward_weights(
    weights: MoeEpForwardWeights,
    *,
    out: MoeEpForwardWeightStaging,
) -> MoeEpNativeForwardWeights:
    """Standalone allocation-free forward weight materialization."""

    from ._megamoe_backend.mxfp8._training_weights import materialize_forward

    return materialize_forward(
        weights,
        out=out,
        fc1_weight_layout=Fc1WeightLayout.GATE_UP_INTERLEAVED_32,
    )


def pack_backward_weights(
    weights: MoeEpBackwardWeights,
    *,
    out: MoeEpBackwardWeightStaging,
) -> MoeEpNativeBackwardWeights:
    """Standalone allocation-free backward weight materialization."""

    from ._megamoe_backend.mxfp8._training_weights import materialize_backward

    return materialize_backward(
        weights,
        out=out,
        fc1_weight_layout=Fc1WeightLayout.GATE_UP_INTERLEAVED_32,
    )


class MoeEp:
    """Fused SwiGLU MoE operator with contiguous expert parallel sharding.

    Global expert ``e`` belongs to group-relative EP rank
    ``e // experts_per_rank``.  The constructor captures static configuration;
    calling the instance accepts runtime tensors for this rank.

    The Rubin training-Mega backend accepts plain BF16/FP32 operands
    (staged to MXFP8 E4M3) or MXFP8 ``BlockScaledTensor`` operands. Final
    output is BF16. ``combine_format`` may be BF16 or MXFP8; forward MXFP8
    combine quantizes each FP32 route accumulator directly before top-k
    reduction. The Rubin training backend requires
    ``apply_topk_in_fc1=True``.
    Native NVFP4 operands and NVFP4 combine/output are not executable.

    ``__call__`` is the inference-only forward surface. Training uses
    :meth:`prepare_training`, :meth:`training_forward`, and
    :meth:`training_backward`. Caller-owned output bundles carry all explicit
    cross-phase state; the operator retains only private runtime and lane
    scratch.

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
        max_recv_size_per_rank: Optional[int] = None,
        drop_on_overflow: bool = False,
        output_format: Union[MoeFormat, str] = MoeFormat.BF16,
        combine_format: Union[MoeFormat, str] = MoeFormat.BF16,
        apply_topk_in_fc1: bool = True,
        weight_interleave_size: Optional[int] = None,
        gate_up_clamp: Optional[float] = None,
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
            raise ValueError(
                f"top_k ({top_k}) cannot exceed num_experts ({num_experts})"
            )
        if max_tokens_per_rank is not None and (
            isinstance(max_tokens_per_rank, bool)
            or not isinstance(max_tokens_per_rank, int)
            or max_tokens_per_rank < 0
        ):
            raise ValueError(
                "max_tokens_per_rank must be a non-negative integer or None"
            )
        if max_recv_size_per_rank is not None and (
            isinstance(max_recv_size_per_rank, bool)
            or not isinstance(max_recv_size_per_rank, int)
            or max_recv_size_per_rank <= 0
        ):
            raise ValueError(
                "max_recv_size_per_rank must be a positive integer or None"
            )
        if not isinstance(drop_on_overflow, bool):
            raise ValueError("drop_on_overflow must be a bool")
        if not isinstance(apply_topk_in_fc1, bool):
            raise ValueError("apply_topk_in_fc1 must be a bool")
        fc1_weight_layout = normalize_fc1_weight_layout(weight_interleave_size)
        for name, value in (
            ("token_padding_size", token_padding_size),
            ("sf_padding_size", sf_padding_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
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
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by EP size ({ep_size})"
            )

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
        self.max_recv_size_per_rank = max_recv_size_per_rank
        self.drop_on_overflow = drop_on_overflow
        self.output_format = _parse_format(output_format)
        self.combine_format = _parse_format(combine_format)
        self.apply_topk_in_fc1 = apply_topk_in_fc1
        self.weight_interleave_size = weight_interleave_size
        self._fc1_weight_layout = fc1_weight_layout
        self.gate_up_clamp = None if gate_up_clamp is None else abs(gate_up_clamp)
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
            required_multiple = (
                32 if fmt is MoeFormat.MXFP8 else 16 if fmt is MoeFormat.NVFP4 else 1
            )
            if hidden_size % required_multiple != 0:
                raise ValueError(
                    f"hidden_size ({hidden_size}) must be divisible by "
                    f"{required_multiple} for {name}={fmt.value}"
                )

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
            max_recv_size_per_rank=self.max_recv_size_per_rank,
            drop_on_overflow=self.drop_on_overflow,
            output_format=self.output_format.value,
            combine_format=self.combine_format.value,
            apply_topk_in_fc1=self.apply_topk_in_fc1,
            fc1_weight_layout=self._fc1_weight_layout,
            gate_up_clamp=self.gate_up_clamp,
            generate_c=False,
            token_padding_size=self.token_padding_size,
            sf_padding_size=self.sf_padding_size,
            tuning=self.tuning,
            backward_wgrad_mode="none",
        )
        self._forward_backend = None
        self._forward_backend_device = None
        self._validated_topk_idx = None
        self._validated_topk_version = None
        self._operator_token = object()
        self._training_state = None
        self._training_lanes: tuple[MoeEpExecutionLane, ...] = ()
        self._training_requirements: (
            Mapping[
                str,
                tuple[tuple[int, ...], tuple[int, ...], torch.dtype, int],
            ]
            | None
        ) = None
        self._poisoned = False
        self._closed = False

    @staticmethod
    def _tensor_version(tensor: torch.Tensor) -> int | None:
        if not isinstance(tensor, torch.Tensor):
            return None
        try:
            return tensor._version
        except RuntimeError:
            return None

    def _get_backend(self, request):
        """Create and cache the private backend on first supported use."""

        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("MoeEp is closed")
            if self._poisoned:
                raise RuntimeError(
                    "MoeEp is unusable after an autotune runtime failure"
                )
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
            _backend.validate_request(request)

            if self._forward_backend is None:
                self._forward_backend = _backend.create_backend(
                    self._forward_config,
                    request.device,
                )
                self._forward_backend_device = request.device
            return self._forward_backend

    def __call__(
        self,
        activation: MoeTensor,
        fc1_weight: MoeTensor,
        fc2_weight: MoeTensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> MoeTensor:
        """Validate and dispatch one fused MoE+EP forward call.

        Expected logical shapes are ``activation=(T,H)``,
        ``fc1_weight=(E_local,H,2I)``, ``fc2_weight=(E_local,I,H)``, and
        ``topk_idx=topk_weights=(T,K)``.

        Training callers must use :meth:`prepare_training` followed by the
        stateless training methods.
        """

        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("MoeEp is closed")
            if self._poisoned:
                raise RuntimeError(
                    "MoeEp is unusable after an autotune runtime failure"
                )
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
            if topk_version is not None and topk_version == version_after_validation:
                self._validated_topk_idx = topk_idx
                self._validated_topk_version = topk_version
            else:
                self._validated_topk_idx = None
                self._validated_topk_version = None
            return self._get_backend(request).forward(request)

    def autotune(
        self,
        activation: MoeTensor,
        fc1_weight: MoeTensor,
        fc2_weight: MoeTensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        candidates: Sequence[MoeEpTuningConfig],
        warmup_iters: int = 3,
        timed_iters: int = 10,
        max_candidates: int = 32,
    ) -> MoeEpAutotuneResult:
        """Collectively sweep inference configurations and apply the winner.

        Candidate compilation, allocation, and warmup are excluded from CUDA
        Event timing. The measured region includes input/weight staging, the
        MegaMoE launch, and the output copy performed by a normal forward.
        """

        from . import _backend
        from ._autotune import (
            benchmark_candidate,
            normalize_candidates,
            raise_preflight_errors,
            select_winner,
            synchronize_candidate,
            verify_candidates_across_ranks,
            verify_state_across_ranks,
        )

        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("MoeEp is closed")
            if self._poisoned:
                raise RuntimeError(
                    "MoeEp is unusable after an autotune runtime failure"
                )
            if self._training_state is not None:
                raise RuntimeError("autotune must be called before prepare_training()")

            normalized = normalize_candidates(
                self.tuning,
                candidates,
                warmup_iters=warmup_iters,
                timed_iters=timed_iters,
                max_candidates=max_candidates,
            )
            verify_candidates_across_ranks(normalized, self._forward_config.ep_group)
            verify_state_across_ranks(
                (
                    self._forward_backend is not None,
                    self._training_state is not None,
                    (
                        None
                        if self._forward_backend_device is None
                        else str(self._forward_backend_device)
                    ),
                ),
                self._forward_config.ep_group,
            )

            candidate_requests = []
            preflight_error: BaseException | None = None
            try:
                for index, tuning in enumerate(normalized):
                    try:
                        config = replace(self._forward_config, tuning=tuning)
                        request = validate_forward(
                            config,
                            activation,
                            fc1_weight,
                            fc2_weight,
                            topk_idx,
                            topk_weights,
                        )
                        if request.device.type != "cuda":
                            raise ValueError(
                                f"autotune requires CUDA inputs, got {request.device}"
                            )
                        with torch.cuda.device(request.device):
                            if torch.cuda.is_current_stream_capturing():
                                raise RuntimeError(
                                    "autotune cannot run during CUDA Graph capture"
                                )
                        _backend.validate_config(config)
                        _backend.validate_request(request)
                        candidate_requests.append(request)
                    except BaseException as exc:
                        raise RuntimeError(
                            f"MoeEp autotune candidate {index} {tuning!r} "
                            f"failed during preflight: {exc}"
                        ) from exc
            except BaseException as exc:
                preflight_error = exc
            raise_preflight_errors(
                preflight_error,
                phase="inference preflight",
                group=self._forward_config.ep_group,
            )
            assert candidate_requests
            device = candidate_requests[0].device

            if self._forward_backend is not None:
                try:
                    synchronize_candidate(device, self._forward_config.ep_group)
                    self._forward_backend.close()
                    self._forward_backend = None
                    self._forward_backend_device = None
                    if self._forward_config.ep_group is not None:
                        dist.barrier(group=self._forward_config.ep_group)
                except BaseException as exc:
                    self._poisoned = True
                    raise RuntimeError(
                        f"MoeEp autotune failed during active backend teardown: {exc}"
                    ) from exc

            results: list[MoeEpAutotuneCandidateResult] = []
            for index, (tuning, request) in enumerate(
                zip(normalized, candidate_requests)
            ):
                backend = None
                runtime_entered = False
                phase = "backend creation"
                try:
                    backend = _backend.create_backend(request.config, device)
                    phase = "compile/prime"
                    runtime_entered = True
                    with torch.cuda.device(device):
                        output = backend.forward(request)
                        del output
                        phase = "warmup"
                        for _ in range(warmup_iters):
                            output = backend.forward(request)
                            del output
                        phase = "pre-timing synchronize"
                        synchronize_candidate(device, self._forward_config.ep_group)
                        phase = "timing"
                        latency_ms, samples_ms = benchmark_candidate(
                            lambda: backend.forward(request),
                            device=device,
                            group=self._forward_config.ep_group,
                            timed_iters=timed_iters,
                        )
                        phase = "post-timing synchronize"
                        synchronize_candidate(device, self._forward_config.ep_group)
                    results.append(
                        MoeEpAutotuneCandidateResult(
                            tuning=tuning,
                            latency_ms=latency_ms,
                            samples_ms=samples_ms,
                        )
                    )
                    phase = "teardown"
                    backend.close()
                    backend = None
                    if self._forward_config.ep_group is not None:
                        dist.barrier(group=self._forward_config.ep_group)
                except BaseException as exc:
                    if backend is not None and self._forward_config.ep_size == 1:
                        with contextlib.suppress(Exception):
                            backend.close()
                    if runtime_entered:
                        self._poisoned = True
                    raise RuntimeError(
                        f"MoeEp autotune candidate {index} {tuning!r} failed during {phase}: {exc}"
                    ) from exc

            winner = select_winner(results)
            winner_request = candidate_requests[normalized.index(winner.tuning)]
            winner_backend = None
            try:
                winner_backend = _backend.create_backend(
                    winner_request.config,
                    device,
                )
                with torch.cuda.device(device):
                    output = winner_backend.forward(winner_request)
                    del output
                    synchronize_candidate(device, self._forward_config.ep_group)
            except BaseException as exc:
                if winner_backend is not None and self._forward_config.ep_size == 1:
                    with contextlib.suppress(Exception):
                        winner_backend.close()
                self._poisoned = True
                raise RuntimeError(
                    f"MoeEp autotune winner {winner.tuning!r} failed final validation: {exc}"
                ) from exc

            self.tuning = winner.tuning
            self._forward_config = winner_request.config
            self._forward_backend = winner_backend
            self._forward_backend_device = device
            self._validated_topk_idx = None
            self._validated_topk_version = None
            return MoeEpAutotuneResult(
                mode="inference",
                winner=winner.tuning,
                candidates=tuple(results),
            )

    def autotune_training(
        self,
        activation: MoeTensor,
        grad_output: MoeTensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        forward_weights: MoeEpNativeForwardWeights,
        backward_weights: MoeEpNativeBackwardWeights,
        candidates: Sequence[MoeEpTuningConfig],
        warmup_iters: int = 3,
        timed_iters: int = 10,
        max_candidates: int = 32,
    ) -> MoeEpAutotuneResult:
        """Sweep complete training forward+backward latency and apply the winner.

        This collective API uses private one-lane temporary resources. It must
        run before :meth:`prepare_training` and accepts kernel-native weights
        so packing allocation and source-layout conversion are not timed.
        """

        from . import _backend
        from ._autotune import (
            allocate_training_outputs,
            benchmark_candidate,
            normalize_candidates,
            raise_preflight_errors,
            select_winner,
            synchronize_candidate,
            verify_candidates_across_ranks,
            verify_state_across_ranks,
        )
        from ._megamoe_backend.mxfp8._training_execute import (
            launch_training_backward,
            launch_training_forward,
        )

        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("MoeEp is closed")
            if self._poisoned:
                raise RuntimeError(
                    "MoeEp is unusable after an autotune runtime failure"
                )
            if self._training_state is not None:
                raise RuntimeError(
                    "autotune_training must be called before prepare_training()"
                )

            normalized = normalize_candidates(
                self.tuning,
                candidates,
                warmup_iters=warmup_iters,
                timed_iters=timed_iters,
                max_candidates=max_candidates,
            )
            verify_candidates_across_ranks(normalized, self._forward_config.ep_group)
            verify_state_across_ranks(
                (
                    self._forward_backend is not None,
                    self._training_state is not None,
                    (
                        None
                        if self._forward_backend_device is None
                        else str(self._forward_backend_device)
                    ),
                ),
                self._forward_config.ep_group,
            )

            device: torch.device | None = None
            preflight_error: BaseException | None = None
            candidate_configs: list[ForwardConfig] = []
            token_count = -1
            try:
                device = torch.device(activation.device)
                if device.type != "cuda":
                    raise ValueError(
                        f"autotune_training requires CUDA inputs, got {device}"
                    )
                with torch.cuda.device(device):
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            "autotune_training cannot run during CUDA Graph capture"
                        )
                for index, tuning in enumerate(normalized):
                    try:
                        config = replace(self._forward_config, tuning=tuning)
                        _validate_training_assert_capability(config)
                        _backend.validate_config(config)
                        activation_tokens = validate_training_input(
                            config,
                            "activation",
                            activation,
                            topk_idx,
                            topk_weights,
                            device=device,
                        )
                        grad_tokens = validate_training_input(
                            config,
                            "grad_output",
                            grad_output,
                            topk_idx,
                            topk_weights,
                            device=device,
                        )
                        if activation_tokens != grad_tokens:
                            raise ValueError(
                                "activation and grad_output must have the same token "
                                f"count, got {activation_tokens} and {grad_tokens}"
                            )
                        validate_native_forward_weights(
                            config, forward_weights, device=device
                        )
                        validate_native_backward_weights(
                            config, backward_weights, device=device
                        )
                        token_count = activation_tokens
                        candidate_configs.append(config)
                    except BaseException as exc:
                        raise RuntimeError(
                            f"MoeEp autotune_training candidate {index} {tuning!r} "
                            f"failed during preflight: {exc}"
                        ) from exc
            except BaseException as exc:
                preflight_error = exc
            raise_preflight_errors(
                preflight_error,
                phase="training preflight",
                group=self._forward_config.ep_group,
            )
            assert device is not None and candidate_configs and token_count >= 0

            if self._forward_backend is not None:
                try:
                    synchronize_candidate(device, self._forward_config.ep_group)
                    self._forward_backend.close()
                    self._forward_backend = None
                    self._forward_backend_device = None
                    if self._forward_config.ep_group is not None:
                        dist.barrier(group=self._forward_config.ep_group)
                except BaseException as exc:
                    self._poisoned = True
                    raise RuntimeError(
                        "MoeEp autotune_training failed during active backend "
                        f"teardown: {exc}"
                    ) from exc

            results: list[MoeEpAutotuneCandidateResult] = []
            for index, (tuning, config) in enumerate(
                zip(normalized, candidate_configs)
            ):
                backend = None
                runtime_entered = False
                phase = "backend creation"
                try:
                    backend = _backend.create_backend(config, device)
                    runtime_entered = True
                    phase = "training preparation"
                    with torch.cuda.device(device):
                        state = backend.prepare_training(lane_count=1)
                        requirements = state.public_requirements()
                        forward_out, backward_out = allocate_training_outputs(
                            requirements,
                            device,
                        )
                        forward_names = (
                            "output",
                            "fc1_preact",
                            "fc1_a",
                            "fc1_sfa",
                            "valid_route_counts",
                            "expert_offsets",
                        )
                        backward_names = (
                            "grad_activation",
                            "dprob",
                            "fc1_b",
                            "fc1_sfb",
                            "fc2_a",
                            "fc2_sfa",
                            "fc2_b",
                            "fc2_sfb",
                        )
                        validate_training_forward_outputs(
                            forward_out,
                            {name: requirements[name] for name in forward_names},
                            device=device,
                        )
                        validate_training_backward_outputs(
                            backward_out,
                            {name: requirements[name] for name in backward_names},
                            device=device,
                        )
                        validate_training_forward_state(
                            fc1_preact=forward_out.fc1_preact,
                            fc1_a=forward_out.fc1_a,
                            fc1_sfa=forward_out.fc1_sfa,
                            valid_route_counts=forward_out.valid_route_counts,
                            expert_offsets=forward_out.expert_offsets,
                            requirements={
                                name: requirements[name]
                                for name in (
                                    "fc1_preact",
                                    "fc1_a",
                                    "fc1_sfa",
                                    "valid_route_counts",
                                    "expert_offsets",
                                )
                            },
                            device=device,
                        )
                        execution = state.views(lane=0, token_count=token_count)

                        def run_training_pair():
                            launch_training_forward(
                                state,
                                execution,
                                activation,
                                topk_idx,
                                topk_weights,
                                weights=forward_weights,
                                out=forward_out,
                            )
                            return launch_training_backward(
                                state,
                                execution,
                                grad_output,
                                topk_idx,
                                topk_weights,
                                weights=backward_weights,
                                fc1_preact=forward_out.fc1_preact,
                                fc1_a=forward_out.fc1_a,
                                fc1_sfa=forward_out.fc1_sfa,
                                valid_route_counts=forward_out.valid_route_counts,
                                expert_offsets=forward_out.expert_offsets,
                                out=backward_out,
                            )

                        phase = "compile/prime"
                        run_training_pair()
                        phase = "warmup"
                        for _ in range(warmup_iters):
                            run_training_pair()
                        phase = "pre-timing synchronize"
                        synchronize_candidate(device, self._forward_config.ep_group)
                        phase = "timing"
                        latency_ms, samples_ms = benchmark_candidate(
                            run_training_pair,
                            device=device,
                            group=self._forward_config.ep_group,
                            timed_iters=timed_iters,
                        )
                        phase = "post-timing synchronize"
                        synchronize_candidate(device, self._forward_config.ep_group)
                    results.append(
                        MoeEpAutotuneCandidateResult(
                            tuning=tuning,
                            latency_ms=latency_ms,
                            samples_ms=samples_ms,
                        )
                    )
                    phase = "teardown"
                    backend.close()
                    backend = None
                    if self._forward_config.ep_group is not None:
                        dist.barrier(group=self._forward_config.ep_group)
                except BaseException as exc:
                    if backend is not None and self._forward_config.ep_size == 1:
                        with contextlib.suppress(Exception):
                            backend.close()
                    if runtime_entered:
                        self._poisoned = True
                    raise RuntimeError(
                        f"MoeEp autotune_training candidate {index} {tuning!r} "
                        f"failed during {phase}: {exc}"
                    ) from exc

            winner = select_winner(results)
            winner_config = candidate_configs[normalized.index(winner.tuning)]
            self.tuning = winner.tuning
            self._forward_config = winner_config
            self._forward_backend = None
            self._forward_backend_device = None
            self._validated_topk_idx = None
            self._validated_topk_version = None
            return MoeEpAutotuneResult(
                mode="training",
                winner=winner.tuning,
                candidates=tuple(results),
            )

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
            if self._poisoned:
                raise RuntimeError(
                    "MoeEp is unusable after an autotune runtime failure"
                )
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

    @property
    def training_lanes(self) -> tuple[MoeEpExecutionLane, ...]:
        """Operator-bound lanes created by :meth:`prepare_training`."""

        return self._training_lanes

    def prepare_training(
        self,
        *,
        lane_count: int = 1,
        device: torch.device | str | int | None = None,
    ) -> Mapping[
        str,
        tuple[tuple[int, ...], tuple[int, ...], torch.dtype, int],
    ]:
        """Collectively prepare private training runtime and return contracts.

        ``device`` defaults to the current CUDA device. No weights or caller
        output buffers are retained by the operator.
        """

        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("MoeEp is closed")
            if self._poisoned:
                raise RuntimeError(
                    "MoeEp is unusable after an autotune runtime failure"
                )
            if (
                isinstance(lane_count, bool)
                or not isinstance(lane_count, int)
                or lane_count <= 0
            ):
                raise ValueError(
                    f"lane_count must be a positive integer, got {lane_count!r}"
                )
            if self._training_state is not None:
                raise RuntimeError("MoeEp training is already prepared")
            if self._fc1_weight_layout is not Fc1WeightLayout.GATE_UP_INTERLEAVED_32:
                raise ValueError("prepare_training requires weight_interleave_size=32")
            resolved_device = _resolve_training_device(device)
            _validate_training_assert_capability(self._forward_config)
            from . import _backend

            _backend.validate_config(self._forward_config)
            if (
                self._forward_backend is not None
                and resolved_device != self._forward_backend_device
            ):
                raise ValueError(
                    f"MoeEp backend is bound to {self._forward_backend_device}; "
                    f"got {resolved_device}"
                )
            if self._forward_backend is None:
                self._forward_backend = _backend.create_backend(
                    self._forward_config,
                    resolved_device,
                )
                self._forward_backend_device = resolved_device
            with torch.cuda.device(resolved_device):
                state = self._forward_backend.prepare_training(
                    lane_count=lane_count,
                )
            self._training_state = state
            self._training_lanes = tuple(
                MoeEpExecutionLane(index, self._operator_token)
                for index in range(lane_count)
            )
            self._training_requirements = state.public_requirements()
            return self._training_requirements

    def _require_training_lane(
        self,
        lane: MoeEpExecutionLane,
    ) -> None:
        if self._closed:
            raise RuntimeError("MoeEp is closed")
        if self._poisoned:
            raise RuntimeError("MoeEp is unusable after an autotune runtime failure")
        if self._training_state is None or self._training_requirements is None:
            raise RuntimeError("prepare_training() must be called first")
        if (
            not isinstance(lane, MoeEpExecutionLane)
            or lane._operator_token is not self._operator_token
            or lane not in self._training_lanes
        ):
            raise ValueError("execution lane does not belong to this MoeEp")

    def _training_requirement_subset(
        self,
        names: tuple[str, ...],
    ) -> dict[str, tuple[tuple[int, ...], tuple[int, ...], torch.dtype, int]]:
        assert self._training_requirements is not None
        return {name: self._training_requirements[name] for name in names}

    def pack_forward_weights(
        self,
        weights: MoeEpForwardWeights,
        *,
        out: MoeEpForwardWeightStaging,
    ) -> MoeEpNativeForwardWeights:
        """Materialize source weights into caller-owned native storage."""

        validate_forward_source_weights(self._forward_config, weights)
        from ._megamoe_backend.mxfp8._training_weights import materialize_forward

        return materialize_forward(
            weights,
            out=out,
            fc1_weight_layout=self._forward_config.fc1_weight_layout,
        )

    def pack_backward_weights(
        self,
        weights: MoeEpBackwardWeights,
        *,
        out: MoeEpBackwardWeightStaging,
    ) -> MoeEpNativeBackwardWeights:
        """Materialize source transpose weights into caller-owned storage."""

        validate_backward_source_weights(self._forward_config, weights)
        from ._megamoe_backend.mxfp8._training_weights import materialize_backward

        return materialize_backward(
            weights,
            out=out,
            fc1_weight_layout=self._forward_config.fc1_weight_layout,
        )

    def training_forward(
        self,
        lane: MoeEpExecutionLane,
        activation: MoeTensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        weights: MoeEpNativeForwardWeights,
        out: MoeEpTrainingForwardOutputs,
    ) -> torch.Tensor:
        """Run forward into caller-owned prepared-training outputs."""

        with self._lifecycle_lock:
            self._require_training_lane(lane)
            assert self._training_state is not None
            assert self._training_requirements is not None
            assert self._forward_backend_device is not None
            token_count = validate_training_input(
                self._forward_config,
                "activation",
                activation,
                topk_idx,
                topk_weights,
                device=self._forward_backend_device,
            )
            validate_native_forward_weights(
                self._forward_config,
                weights,
                device=self._forward_backend_device,
            )
            validate_training_forward_outputs(
                out,
                self._training_requirement_subset(
                    (
                        "output",
                        "fc1_preact",
                        "fc1_a",
                        "fc1_sfa",
                        "valid_route_counts",
                        "expert_offsets",
                    )
                ),
                device=self._forward_backend_device,
            )
            validate_training_non_aliasing(
                {
                    **_named_moe_tensors("activation", activation),
                    "topk_idx": topk_idx,
                    "topk_weights": topk_weights,
                    "weights.fc1.payload": weights.fc1.payload,
                    "weights.fc1.scale": weights.fc1.scale,
                    "weights.fc2.payload": weights.fc2.payload,
                    "weights.fc2.scale": weights.fc2.scale,
                    "out.output": out.output,
                    "out.fc1_preact": out.fc1_preact,
                    "out.fc1_a": out.fc1_a,
                    "out.fc1_sfa": out.fc1_sfa,
                    "out.valid_route_counts": out.valid_route_counts,
                    "out.expert_offsets": out.expert_offsets,
                }
            )
            from ._megamoe_backend.mxfp8._training_execute import (
                launch_training_forward,
            )

            with torch.cuda.device(self._forward_backend_device):
                execution = self._training_state.views(
                    lane=lane.index,
                    token_count=token_count,
                )
                return launch_training_forward(
                    self._training_state,
                    execution,
                    activation,
                    topk_idx,
                    topk_weights,
                    weights=weights,
                    out=out,
                )

    def training_backward(
        self,
        lane: MoeEpExecutionLane,
        grad_output: MoeTensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        weights: MoeEpNativeBackwardWeights,
        fc1_preact: torch.Tensor,
        fc1_a: torch.Tensor | None = None,
        fc1_sfa: torch.Tensor | None = None,
        valid_route_counts: torch.Tensor | None = None,
        expert_offsets: torch.Tensor | None = None,
        out: MoeEpTrainingBackwardOutputs | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        MoeEpTrainingWgradOperands,
    ]:
        """Run backward into caller-owned outputs using explicit forward state."""

        with self._lifecycle_lock:
            self._require_training_lane(lane)
            assert self._training_state is not None
            assert self._forward_backend_device is not None
            if out is None:
                raise TypeError("out must be a MoeEpTrainingBackwardOutputs")
            token_count = validate_training_input(
                self._forward_config,
                "grad_output",
                grad_output,
                topk_idx,
                topk_weights,
                device=self._forward_backend_device,
            )
            validate_native_backward_weights(
                self._forward_config,
                weights,
                device=self._forward_backend_device,
            )
            if fc1_preact is None:
                raise ValueError("fc1_preact from the matching forward is required")
            backward_output = out
            validate_training_backward_outputs(
                backward_output,
                self._training_requirement_subset(
                    (
                        "grad_activation",
                        "dprob",
                        "fc1_b",
                        "fc1_sfb",
                        "fc2_a",
                        "fc2_sfa",
                        "fc2_b",
                        "fc2_sfb",
                    )
                ),
                device=self._forward_backend_device,
            )
            validate_training_forward_state(
                fc1_preact=fc1_preact,
                fc1_a=fc1_a,
                fc1_sfa=fc1_sfa,
                valid_route_counts=valid_route_counts,
                expert_offsets=expert_offsets,
                requirements=self._training_requirement_subset(
                    (
                        "fc1_preact",
                        "fc1_a",
                        "fc1_sfa",
                        "valid_route_counts",
                        "expert_offsets",
                    )
                ),
                device=self._forward_backend_device,
            )
            validate_training_non_aliasing(
                {
                    **_named_moe_tensors("grad_output", grad_output),
                    "topk_idx": topk_idx,
                    "topk_weights": topk_weights,
                    "weights.w2_transpose.payload": weights.w2_transpose.payload,
                    "weights.w2_transpose.scale": weights.w2_transpose.scale,
                    "weights.w1_transpose.payload": weights.w1_transpose.payload,
                    "weights.w1_transpose.scale": weights.w1_transpose.scale,
                    "fc1_preact": fc1_preact,
                    "fc1_a": fc1_a,
                    "fc1_sfa": fc1_sfa,
                    "valid_route_counts": valid_route_counts,
                    "expert_offsets": expert_offsets,
                    "out.grad_activation": backward_output.grad_activation,
                    "out.dprob": backward_output.dprob,
                    "out.fc1_b": backward_output.fc1_b,
                    "out.fc1_sfb": backward_output.fc1_sfb,
                    "out.fc2_a": backward_output.fc2_a,
                    "out.fc2_sfa": backward_output.fc2_sfa,
                    "out.fc2_b": backward_output.fc2_b,
                    "out.fc2_sfb": backward_output.fc2_sfb,
                }
            )
            from ._megamoe_backend.mxfp8._training_execute import (
                launch_training_backward,
            )

            with torch.cuda.device(self._forward_backend_device):
                execution = self._training_state.views(
                    lane=lane.index,
                    token_count=token_count,
                )
                return launch_training_backward(
                    self._training_state,
                    execution,
                    grad_output,
                    topk_idx,
                    topk_weights,
                    weights=weights,
                    fc1_preact=fc1_preact,
                    fc1_a=fc1_a,
                    fc1_sfa=fc1_sfa,
                    valid_route_counts=valid_route_counts,
                    expert_offsets=expert_offsets,
                    out=backward_output,
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
            self._training_state = None
            self._training_lanes = ()
            self._training_requirements = None
            self._closed = True

    def __enter__(self) -> "MoeEp":
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("MoeEp is closed")
            if self._poisoned:
                raise RuntimeError(
                    "MoeEp is unusable after an autotune runtime failure"
                )
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
    "MoeEpAutotuneCandidateResult",
    "MoeEpAutotuneResult",
    "MoeEpBackwardWeightStaging",
    "MoeEpBackwardWeights",
    "MoeEpExecutionLane",
    "MoeEpForwardWeightStaging",
    "MoeEpForwardWeights",
    "MoeEpNativeBackwardWeights",
    "MoeEpNativeForwardWeights",
    "MoeEpTrainingBackwardOutputs",
    "MoeEpTrainingForwardOutputs",
    "MoeEpTrainingWgradOperands",
    "MoeFormat",
    "MoeTensor",
    "pack_backward_weights",
    "pack_forward_weights",
]
