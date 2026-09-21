# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python API surface for fused SwiGLU MoE with expert parallelism.

The public API performs contract validation and dispatches through a private,
lazy backend seam. Device-runtime implementation details remain outside this
module.
"""

from __future__ import annotations

import contextlib
import threading
import warnings
import weakref
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, Mapping, Sequence

import torch
import torch.distributed as dist

from ._config import (
    MoeEpConfig,
    MoeEpFc1WeightLayout,
    ResolvedMoeEpConfig,
    resolve_moe_ep_config,
)
from ._tuning import (
    MoeEpAutotuneCandidateResult,
    MoeEpAutotuneResult,
    MoeEpTuningConfig,
)
from ._types import (
    BlockScaledTensor,
    MoeEpBackwardWeightStaging,
    MoeEpBackwardWeights,
    MoeEpForwardWeightStaging,
    MoeEpForwardWeights,
    MoeEpNativeBackwardWeights,
    MoeEpNativeDiscreteBackwardWeights,
    MoeEpNativeDiscreteForwardWeights,
    MoeEpNativeDiscreteWeight,
    MoeEpNativeForwardWeights,
    MoeEpNativeWeightStorageMode,
    MoeEpTrainingBackwardOutputs,
    MoeEpTrainingForwardOutputs,
    MoeEpTrainingWgradOperands,
    MoeFormat,
    MoeTensor,
)
from ._validation import (
    validate_backward_source_weights,
    validate_forward,
    validate_forward_source_weights,
    validate_native_backward_weights,
    validate_native_discrete_backward_weights,
    validate_native_discrete_forward_weights,
    validate_native_forward_weights,
    validate_training_backward_outputs,
    validate_training_forward_outputs,
    validate_training_forward_state,
    validate_training_input,
    validate_training_non_aliasing,
)

if TYPE_CHECKING:
    from ._backend import MoeEpBackend


@dataclass(frozen=True)
class _MoeEpExecutionState:
    resolved_config: ResolvedMoeEpConfig
    backend: "MoeEpBackend | None"

    def __post_init__(self) -> None:
        if self.backend is None:
            return
        if self.backend.resolved_config is not self.resolved_config:
            raise ValueError("backend must own the exact resolved config generation")
        if self.backend.device.type != "cuda" or self.backend.device.index is None:
            raise ValueError(
                "backend must expose a CUDA device with a concrete ordinal"
            )


def _validate_training_assert_capability(
    config: ResolvedMoeEpConfig,
) -> None:
    """Fail before allocation when graph error-mode primitives are unavailable."""

    parallel = config.public_config.parallel
    if parallel.drop_on_overflow:
        return
    if not callable(getattr(torch, "_assert_async", None)):
        raise RuntimeError(
            "drop_on_overflow=False training requires callable "
            "torch._assert_async before CUDA Graph capture"
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
        fc1_weight_layout=MoeEpFc1WeightLayout.GATE_UP_INTERLEAVED_32,
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
        fc1_weight_layout=MoeEpFc1WeightLayout.GATE_UP_INTERLEAVED_32,
    )


class MoeEp:
    """Fused SwiGLU MoE operator with contiguous expert parallel sharding.

    Global expert ``e`` belongs to group-relative EP rank
    ``e // experts_per_rank``.  The constructor captures static configuration;
    calling the instance accepts runtime tensors for this rank.

    Construction is config-only: pass one frozen :class:`MoeEpConfig`.
    Nested model, parallel, data-path, and phase-specific tuning values are
    exposed through read-only properties. There is no scalar-kwargs
    constructor or post-construction general reconfiguration surface.

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
    cross-phase state; the operator retains only private runtime and instance
    scratch.

    The backend is created lazily on the first supported forward call. Valid
    combinations outside the current backend capability matrix fail explicitly
    instead of returning uninitialized storage. Once created, a backend and its
    workspaces are bound to that call's device; use a separate ``MoeEp``
    instance for another device.

    ``validation_mode="strict"`` validates expert IDs before eager execution.
    ``validation_mode="trusted"`` skips only that value-range check; callers
    must guarantee that every routing ID belongs to ``[0, num_experts)``.
    Negative IDs and dropped-route sentinels are not supported. CUDA Graph
    replay does not repeat this value check, so replayed routing contents must
    preserve the same dense-routing invariant. Structural tensor, workspace,
    aliasing, and overflow checks remain enabled in both modes.

    ``physical_recv_pool_rows`` is the exact number of physical padded
    receive-pool rows. An explicit value ``P`` must satisfy ``P % 128 == 0``;
    the backend derives an operation-scoped logical route limit without
    shrinking the requested backing pool. ``None`` selects a static canonical
    pool that covers both inference and training padding policies.
    """

    def __init__(self, config: MoeEpConfig) -> None:
        self._lifecycle_lock = threading.RLock()
        resolved = resolve_moe_ep_config(config)
        self._execution_state = _MoeEpExecutionState(
            resolved_config=resolved,
            backend=None,
        )
        self._validated_topk_idx = None
        self._validated_topk_version = None
        self._training_state = None
        self._training_requirements: (
            Mapping[
                str,
                tuple[tuple[int, ...], tuple[int, ...], torch.dtype, int],
            ]
            | None
        ) = None
        self._validated_discrete_pointer_tables: dict[
            int,
            tuple[weakref.ReferenceType[torch.Tensor], int, int],
        ] = {}
        self._poisoned = False
        self._closed = False

    @property
    def config(self) -> MoeEpConfig:
        return self._execution_state.resolved_config.public_config

    @property
    def num_experts(self) -> int:
        return self.config.model.num_experts

    @property
    def hidden_size(self) -> int:
        return self.config.model.hidden_size

    @property
    def intermediate_size(self) -> int:
        return self.config.model.intermediate_size

    @property
    def top_k(self) -> int:
        return self.config.model.top_k

    @property
    def ep_group(self) -> dist.ProcessGroup | None:
        return self.config.parallel.ep_group

    @property
    def ep_size(self) -> int:
        return self._execution_state.resolved_config.topology.ep_size

    @property
    def ep_rank(self) -> int:
        return self._execution_state.resolved_config.topology.ep_rank

    @property
    def ep_global_ranks(self) -> tuple[int, ...]:
        return self._execution_state.resolved_config.topology.ep_global_ranks

    @property
    def experts_per_rank(self) -> int:
        return self._execution_state.resolved_config.topology.experts_per_rank

    @property
    def max_tokens_per_rank(self) -> int | None:
        return self.config.parallel.max_tokens_per_rank

    @property
    def physical_recv_pool_rows(self) -> int:
        return (
            self._execution_state.resolved_config.receive_capacity.physical_recv_pool_rows
        )

    @property
    def drop_on_overflow(self) -> bool:
        return self.config.parallel.drop_on_overflow

    @property
    def token_padding_size(self) -> int:
        return self.config.parallel.token_padding_size

    @property
    def sf_padding_size(self) -> int:
        return self.config.parallel.sf_padding_size

    @property
    def output_format(self) -> MoeFormat:
        return self.config.data_path.output_format

    @property
    def combine_format(self) -> MoeFormat:
        return self.config.data_path.combine_format

    @property
    def apply_topk_in_fc1(self) -> bool:
        return self.config.data_path.apply_topk_in_fc1

    @property
    def fc1_weight_layout(self) -> MoeEpFc1WeightLayout:
        return self.config.data_path.fc1_weight_layout

    @property
    def gate_up_clamp(self) -> float | None:
        return self.config.data_path.gate_up_clamp

    @property
    def inference_tuning(self) -> MoeEpTuningConfig:
        return self.config.inference_tuning

    @property
    def training_forward_tuning(self) -> MoeEpTuningConfig:
        return self.config.training_forward_tuning

    @property
    def training_backward_tuning(self) -> MoeEpTuningConfig:
        return self.config.training_backward_tuning

    @property
    def training_weight_storage_mode(self) -> MoeEpNativeWeightStorageMode:
        return self.config.training_weight_storage_mode

    @property
    def validation_mode(self) -> Literal["strict", "trusted"]:
        """Return the immutable expert-ID validation policy."""

        return self.config.validation_mode

    @staticmethod
    def _tensor_version(tensor: torch.Tensor) -> int | None:
        if not isinstance(tensor, torch.Tensor):
            return None
        try:
            return tensor._version
        except RuntimeError:
            return None

    @staticmethod
    def _discrete_pointer_tables(
        weights: MoeEpNativeDiscreteForwardWeights | MoeEpNativeDiscreteBackwardWeights,
    ) -> tuple[torch.Tensor, ...]:
        if isinstance(weights, MoeEpNativeDiscreteForwardWeights):
            pair = (weights.fc1, weights.fc2)
        elif isinstance(weights, MoeEpNativeDiscreteBackwardWeights):
            pair = (weights.w2_transpose, weights.w1_transpose)
        else:
            raise TypeError(
                "discrete weights must be a forward or backward discrete bundle"
            )
        return tuple(
            tensor
            for weight in pair
            for tensor in (weight.payload_ptrs, weight.scale_ptrs)
        )

    def _validate_discrete_pointer_lifetime(
        self,
        weights: MoeEpNativeDiscreteForwardWeights | MoeEpNativeDiscreteBackwardWeights,
        *,
        validate,
        device: torch.device,
    ) -> None:
        tables = self._discrete_pointer_tables(weights)
        with torch.cuda.device(device):
            capturing = torch.cuda.is_current_stream_capturing()
        self._validated_discrete_pointer_tables = {
            identity: record
            for identity, record in self._validated_discrete_pointer_tables.items()
            if record[0]() is not None
        }

        def is_validated(table: torch.Tensor) -> bool:
            version = self._tensor_version(table)
            if version is None:
                return False
            record = self._validated_discrete_pointer_tables.get(id(table))
            return (
                record is not None
                and record[0]() is table
                and record[1] == int(table.data_ptr())
                and record[2] == version
            )

        if self.validation_mode == "strict" and capturing:
            versionless = tuple(
                index
                for index, table in enumerate(tables)
                if self._tensor_version(table) is None
            )
            if versionless:
                raise RuntimeError(
                    "strict CUDA Graph capture requires discrete pointer tables "
                    "with PyTorch version counters; inference-mode tables are "
                    f"unsupported (table indices {versionless})"
                )
            missing = tuple(
                index for index, table in enumerate(tables) if not is_validated(table)
            )
            if missing:
                raise RuntimeError(
                    "discrete pointer tables must pass one strict eager validation "
                    "before CUDA Graph capture"
                )
        validate_pointees = (
            self.validation_mode == "strict"
            and not capturing
            and any(not is_validated(table) for table in tables)
        )
        validate(
            self._execution_state.resolved_config,
            weights,
            device=device,
            validate_pointees=validate_pointees,
        )
        if self.validation_mode == "strict" and not capturing:
            for table in tables:
                version = self._tensor_version(table)
                if version is not None:
                    self._validated_discrete_pointer_tables[id(table)] = (
                        weakref.ref(table),
                        int(table.data_ptr()),
                        version,
                    )

    @staticmethod
    def _native_weight_tensors(
        weights: (
            MoeEpNativeForwardWeights
            | MoeEpNativeBackwardWeights
            | MoeEpNativeDiscreteForwardWeights
            | MoeEpNativeDiscreteBackwardWeights
        ),
    ) -> dict[str, torch.Tensor]:
        if isinstance(weights, MoeEpNativeForwardWeights):
            return {
                "weights.fc1.payload": weights.fc1.payload,
                "weights.fc1.scale": weights.fc1.scale,
                "weights.fc2.payload": weights.fc2.payload,
                "weights.fc2.scale": weights.fc2.scale,
            }
        if isinstance(weights, MoeEpNativeBackwardWeights):
            return {
                "weights.w2_transpose.payload": weights.w2_transpose.payload,
                "weights.w2_transpose.scale": weights.w2_transpose.scale,
                "weights.w1_transpose.payload": weights.w1_transpose.payload,
                "weights.w1_transpose.scale": weights.w1_transpose.scale,
            }
        if isinstance(weights, MoeEpNativeDiscreteForwardWeights):
            return {
                "weights.fc1.payload_ptrs": weights.fc1.payload_ptrs,
                "weights.fc1.scale_ptrs": weights.fc1.scale_ptrs,
                "weights.fc2.payload_ptrs": weights.fc2.payload_ptrs,
                "weights.fc2.scale_ptrs": weights.fc2.scale_ptrs,
            }
        if isinstance(weights, MoeEpNativeDiscreteBackwardWeights):
            return {
                "weights.w2_transpose.payload_ptrs": weights.w2_transpose.payload_ptrs,
                "weights.w2_transpose.scale_ptrs": weights.w2_transpose.scale_ptrs,
                "weights.w1_transpose.payload_ptrs": weights.w1_transpose.payload_ptrs,
                "weights.w1_transpose.scale_ptrs": weights.w1_transpose.scale_ptrs,
            }
        raise TypeError(f"unsupported native weight bundle {type(weights).__name__}")

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

            state = self._execution_state
            resolved = state.resolved_config
            backend = state.backend
            if backend is not None and request.device != backend.device:
                raise ValueError(
                    f"MoeEp backend is bound to {backend.device}; "
                    f"create a separate MoeEp instance for {request.device}"
                )

            _backend.validate_config(resolved)
            _backend.validate_request(resolved, request)

            if backend is None:
                backend = _backend.create_backend(
                    resolved,
                    request.device,
                )
                self._execution_state = replace(state, backend=backend)
            return backend

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
            strict_validation = self.validation_mode == "strict"
            topk_version = self._tensor_version(topk_idx) if strict_validation else None
            validate_expert_ids = strict_validation and not (
                self._validated_topk_idx is topk_idx
                and topk_version is not None
                and topk_version == self._validated_topk_version
            )
            resolved = self._execution_state.resolved_config
            request = validate_forward(
                resolved,
                activation,
                fc1_weight,
                fc2_weight,
                topk_idx,
                topk_weights,
                validate_expert_ids=validate_expert_ids,
            )
            version_after_validation = self._tensor_version(topk_idx)
            if (
                strict_validation
                and topk_version is not None
                and topk_version == version_after_validation
            ):
                self._validated_topk_idx = topk_idx
                self._validated_topk_version = topk_version
            else:
                self._validated_topk_idx = None
                self._validated_topk_version = None
            return self._get_backend(request).forward(request)

    def autotune_inference(
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
                raise RuntimeError(
                    "autotune_inference must be called before " "prepare_training()"
                )

            state = self._execution_state
            # Keep the active backend alive until a winner passes final
            # validation. Candidate failure must not invalidate its warmed
            # workspace or any CUDA Graph that still references it.
            group = state.resolved_config.public_config.parallel.ep_group
            normalized = normalize_candidates(
                state.resolved_config.public_config.inference_tuning,
                candidates,
                warmup_iters=warmup_iters,
                timed_iters=timed_iters,
                max_candidates=max_candidates,
            )
            verify_candidates_across_ranks(normalized, group)
            # CUDA ordinals are rank-local; device binding is checked against
            # this rank's request during preflight below.
            verify_state_across_ranks(
                (
                    state.backend is not None,
                    self._training_state is not None,
                ),
                group,
            )

            candidate_requests: list[tuple[ResolvedMoeEpConfig, object]] = []
            preflight_error: BaseException | None = None
            try:
                for index, tuning in enumerate(normalized):
                    try:
                        public_config = replace(
                            state.resolved_config.public_config,
                            inference_tuning=tuning,
                        )
                        config = ResolvedMoeEpConfig(
                            public_config=public_config,
                            topology=state.resolved_config.topology,
                        )
                        request = validate_forward(
                            config,
                            activation,
                            fc1_weight,
                            fc2_weight,
                            topk_idx,
                            topk_weights,
                            validate_expert_ids=self.validation_mode == "strict",
                        )
                        if request.device.type != "cuda":
                            raise ValueError(
                                f"autotune requires CUDA inputs, got {request.device}"
                            )
                        if (
                            state.backend is not None
                            and state.backend.device != request.device
                        ):
                            raise ValueError(
                                "MoeEp backend is bound to "
                                f"{state.backend.device}; cannot autotune on "
                                f"{request.device}"
                            )
                        with torch.cuda.device(request.device):
                            if torch.cuda.is_current_stream_capturing():
                                raise RuntimeError(
                                    "autotune cannot run during CUDA Graph capture"
                                )
                        _backend.validate_config(config)
                        _backend.validate_request(config, request)
                        candidate_requests.append((config, request))
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
                group=group,
            )
            assert candidate_requests
            device = candidate_requests[0][1].device

            results: list[MoeEpAutotuneCandidateResult] = []
            for index, (tuning, candidate) in enumerate(
                zip(normalized, candidate_requests)
            ):
                candidate_config, request = candidate
                backend = None
                phase = "backend creation"
                try:
                    backend = _backend.create_backend(
                        candidate_config,
                        device,
                    )
                    phase = "compile/prime"
                    with torch.cuda.device(device):
                        output = backend.forward(request)
                        del output
                        phase = "warmup"
                        for _ in range(warmup_iters):
                            output = backend.forward(request)
                            del output
                        phase = "pre-timing synchronize"
                        synchronize_candidate(device, group)
                        phase = "timing"
                        latency_ms, samples_ms = benchmark_candidate(
                            lambda: backend.forward(request),
                            device=device,
                            group=group,
                            timed_iters=timed_iters,
                        )
                        phase = "post-timing synchronize"
                        synchronize_candidate(device, group)
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
                    if group is not None:
                        dist.barrier(group=group)
                except BaseException as exc:
                    if backend is not None:
                        with contextlib.suppress(Exception):
                            backend.close()
                    raise RuntimeError(
                        f"MoeEp autotune candidate {index} {tuning!r} failed during {phase}: {exc}"
                    ) from exc

            winner = select_winner(results)
            winner_config, winner_request = candidate_requests[
                normalized.index(winner.tuning)
            ]
            winner_backend = None
            try:
                winner_backend = _backend.create_backend(
                    winner_config,
                    device,
                )
                with torch.cuda.device(device):
                    output = winner_backend.forward(winner_request)
                    del output
                    synchronize_candidate(device, group)
            except BaseException as exc:
                if winner_backend is not None:
                    with contextlib.suppress(Exception):
                        winner_backend.close()
                raise RuntimeError(
                    f"MoeEp autotune winner {winner.tuning!r} failed final validation: {exc}"
                ) from exc

            old_backend = state.backend
            if old_backend is not None:
                try:
                    synchronize_candidate(device, group)
                    old_backend.close()
                    if group is not None:
                        dist.barrier(group=group)
                except BaseException as exc:
                    with contextlib.suppress(Exception):
                        winner_backend.close()
                    self._poisoned = True
                    raise RuntimeError(
                        "MoeEp autotune_inference failed during active "
                        f"backend replacement: {exc}"
                    ) from exc
            self._execution_state = _MoeEpExecutionState(
                resolved_config=winner_config,
                backend=winner_backend,
            )
            self._validated_topk_idx = None
            self._validated_topk_version = None
            return MoeEpAutotuneResult(
                mode="inference",
                winner=winner.tuning,
                candidates=tuple(results),
            )

    def _autotune_training_phase(
        self,
        phase_name: Literal["training_forward", "training_backward"],
        activation: MoeTensor,
        grad_output: MoeTensor | None,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        forward_weights: MoeEpNativeForwardWeights | MoeEpNativeDiscreteForwardWeights,
        backward_weights: (
            MoeEpNativeBackwardWeights | MoeEpNativeDiscreteBackwardWeights | None
        ),
        candidates: Sequence[MoeEpTuningConfig],
        warmup_iters: int = 3,
        timed_iters: int = 10,
        max_candidates: int = 32,
    ) -> MoeEpAutotuneResult:
        """Sweep one training phase and atomically apply its winner."""

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
                    f"autotune_{phase_name} must be called before " "prepare_training()"
                )
            execution_state = self._execution_state
            weight_storage_mode = (
                execution_state.resolved_config.public_config.training_weight_storage_mode
            )
            # Training candidates coexist with the active inference backend;
            # only a successful winner commit retires that backend.
            group = execution_state.resolved_config.public_config.parallel.ep_group
            if phase_name == "training_backward":
                if grad_output is None or backward_weights is None:
                    raise TypeError(
                        "training backward autotune requires grad_output and "
                        "backward_weights"
                    )

            baseline_tuning = (
                execution_state.resolved_config.public_config.training_forward_tuning
                if phase_name == "training_forward"
                else execution_state.resolved_config.public_config.training_backward_tuning
            )
            normalized = normalize_candidates(
                baseline_tuning,
                candidates,
                warmup_iters=warmup_iters,
                timed_iters=timed_iters,
                max_candidates=max_candidates,
            )
            verify_candidates_across_ranks(normalized, group)
            # CUDA ordinals are rank-local; device binding is checked against
            # this rank's activation during preflight below.
            verify_state_across_ranks(
                (
                    execution_state.backend is not None,
                    self._training_state is not None,
                    weight_storage_mode.value,
                ),
                group,
            )

            device: torch.device | None = None
            preflight_error: BaseException | None = None
            candidate_configs: list[ResolvedMoeEpConfig] = []
            token_count = -1
            try:
                device = torch.device(activation.device)
                if device.type != "cuda":
                    raise ValueError(
                        f"autotune_{phase_name} requires CUDA inputs, " f"got {device}"
                    )
                if (
                    execution_state.backend is not None
                    and execution_state.backend.device != device
                ):
                    raise ValueError(
                        "MoeEp backend is bound to "
                        f"{execution_state.backend.device}; cannot autotune "
                        f"on {device}"
                    )
                with torch.cuda.device(device):
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            f"autotune_{phase_name} cannot run during "
                            "CUDA Graph capture"
                        )
                for index, tuning in enumerate(normalized):
                    try:
                        public_config = execution_state.resolved_config.public_config
                        if phase_name == "training_forward":
                            public_config = replace(
                                public_config,
                                training_forward_tuning=tuning,
                            )
                        else:
                            public_config = replace(
                                public_config,
                                training_backward_tuning=tuning,
                            )
                        config = ResolvedMoeEpConfig(
                            public_config=public_config,
                            topology=execution_state.resolved_config.topology,
                        )
                        _validate_training_assert_capability(config)
                        _backend.validate_config(config)
                        activation_tokens = validate_training_input(
                            config,
                            "activation",
                            activation,
                            topk_idx,
                            topk_weights,
                            device=device,
                            validate_expert_ids=self.validation_mode == "strict",
                        )
                        if phase_name == "training_backward":
                            assert grad_output is not None
                            grad_tokens = validate_training_input(
                                config,
                                "grad_output",
                                grad_output,
                                topk_idx,
                                topk_weights,
                                device=device,
                                validate_expert_ids=(self.validation_mode == "strict"),
                            )
                            if activation_tokens != grad_tokens:
                                raise ValueError(
                                    "activation and grad_output must have the "
                                    "same token count, got "
                                    f"{activation_tokens} and {grad_tokens}"
                                )
                        if weight_storage_mode is MoeEpNativeWeightStorageMode.DISCRETE:
                            self._validate_discrete_pointer_lifetime(
                                forward_weights,
                                validate=validate_native_discrete_forward_weights,
                                device=device,
                            )
                            if phase_name == "training_backward":
                                assert backward_weights is not None
                                self._validate_discrete_pointer_lifetime(
                                    backward_weights,
                                    validate=(
                                        validate_native_discrete_backward_weights
                                    ),
                                    device=device,
                                )
                        else:
                            validate_native_forward_weights(
                                config,
                                forward_weights,
                                device=device,
                            )
                            if phase_name == "training_backward":
                                assert backward_weights is not None
                                validate_native_backward_weights(
                                    config,
                                    backward_weights,
                                    device=device,
                                )
                        token_count = activation_tokens
                        candidate_configs.append(config)
                    except BaseException as exc:
                        raise RuntimeError(
                            f"MoeEp autotune_{phase_name} candidate {index} "
                            f"{tuning!r} failed during preflight: {exc}"
                        ) from exc
            except BaseException as exc:
                preflight_error = exc
            raise_preflight_errors(
                preflight_error,
                phase=f"{phase_name} preflight",
                group=group,
            )
            assert device is not None and candidate_configs and token_count >= 0

            results: list[MoeEpAutotuneCandidateResult] = []
            for index, (tuning, config) in enumerate(
                zip(normalized, candidate_configs)
            ):
                backend = None
                phase = "backend creation"
                try:
                    backend = _backend.create_backend(config, device)
                    phase = "training preparation"
                    with torch.cuda.device(device):
                        training_state = backend.prepare_training()
                        requirements = training_state.public_requirements()
                        symmetric_buffers = training_state.public_symmetric_buffers()
                        forward_out, backward_out = allocate_training_outputs(
                            requirements,
                            device,
                            symmetric_buffers,
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
                        execution = training_state.views(token_count=token_count)

                        def run_forward():
                            return launch_training_forward(
                                training_state,
                                execution,
                                activation,
                                topk_idx,
                                topk_weights,
                                weights=forward_weights,
                                out=forward_out,
                            )

                        def run_backward():
                            assert grad_output is not None
                            assert backward_weights is not None
                            return launch_training_backward(
                                training_state,
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

                        if phase_name == "training_forward":
                            timed_call = run_forward
                        else:
                            # Backward treats forward-produced saved tensors as
                            # immutable inputs. Generate them outside timing;
                            # backward overwrites only its explicit outputs.
                            run_forward()
                            timed_call = run_backward
                        phase = "compile/prime"
                        timed_call()
                        phase = "warmup"
                        for _ in range(warmup_iters):
                            timed_call()
                        phase = "pre-timing synchronize"
                        synchronize_candidate(device, group)
                        phase = "timing"
                        latency_ms, samples_ms = benchmark_candidate(
                            timed_call,
                            device=device,
                            group=group,
                            timed_iters=timed_iters,
                        )
                        phase = "post-timing synchronize"
                        synchronize_candidate(device, group)
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
                    if group is not None:
                        dist.barrier(group=group)
                except BaseException as exc:
                    if backend is not None:
                        with contextlib.suppress(Exception):
                            backend.close()
                    raise RuntimeError(
                        f"MoeEp autotune_{phase_name} candidate {index} "
                        f"{tuning!r} failed during {phase}: {exc}"
                    ) from exc

            winner = select_winner(results)
            winner_config = candidate_configs[normalized.index(winner.tuning)]
            old_backend = execution_state.backend
            if old_backend is not None:
                try:
                    synchronize_candidate(device, group)
                    old_backend.close()
                    if group is not None:
                        dist.barrier(group=group)
                except BaseException as exc:
                    self._poisoned = True
                    raise RuntimeError(
                        f"MoeEp autotune_{phase_name} failed during active "
                        f"backend teardown: {exc}"
                    ) from exc
            self._execution_state = _MoeEpExecutionState(
                resolved_config=winner_config,
                backend=None,
            )
            self._validated_topk_idx = None
            self._validated_topk_version = None
            return MoeEpAutotuneResult(
                mode=phase_name,
                winner=winner.tuning,
                candidates=tuple(results),
            )

    def autotune_training_forward(
        self,
        activation: MoeTensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        forward_weights: MoeEpNativeForwardWeights | MoeEpNativeDiscreteForwardWeights,
        candidates: Sequence[MoeEpTuningConfig],
        warmup_iters: int = 3,
        timed_iters: int = 10,
        max_candidates: int = 32,
    ) -> MoeEpAutotuneResult:
        """Tune only the training-forward phase and apply its winner."""

        return self._autotune_training_phase(
            "training_forward",
            activation,
            None,
            topk_idx,
            topk_weights,
            forward_weights=forward_weights,
            backward_weights=None,
            candidates=candidates,
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
            max_candidates=max_candidates,
        )

    def autotune_training_backward(
        self,
        activation: MoeTensor,
        grad_output: MoeTensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        forward_weights: MoeEpNativeForwardWeights | MoeEpNativeDiscreteForwardWeights,
        backward_weights: (
            MoeEpNativeBackwardWeights | MoeEpNativeDiscreteBackwardWeights
        ),
        candidates: Sequence[MoeEpTuningConfig],
        warmup_iters: int = 3,
        timed_iters: int = 10,
        max_candidates: int = 32,
    ) -> MoeEpAutotuneResult:
        """Tune only training backward and apply its winner."""

        return self._autotune_training_phase(
            "training_backward",
            activation,
            grad_output,
            topk_idx,
            topk_weights,
            forward_weights=forward_weights,
            backward_weights=backward_weights,
            candidates=candidates,
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
            max_candidates=max_candidates,
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

    def training_symmetric_buffers(
        self,
    ) -> Mapping[str, torch.Tensor]:
        """Return stable instance-owned symmetric input/output buffers.

        Later training calls on this instance may overwrite their contents.
        """

        with self._lifecycle_lock:
            self._require_training_prepared()
            assert self._training_state is not None
            return self._training_state.public_symmetric_buffers()

    def prepare_training(
        self,
        *,
        device: torch.device | str | int | None = None,
    ) -> Mapping[
        str,
        tuple[tuple[int, ...], tuple[int, ...], torch.dtype, int],
    ]:
        """Collectively prepare private training runtime and return contracts.

        ``device`` defaults to the current CUDA device. No weights are retained.
        Stable symmetric input and final-output buffers are available through
        :meth:`training_symmetric_buffers`.

        All training work for this instance must execute sequentially on one
        CUDA stream. This contract is documented but not checked at runtime.
        """

        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("MoeEp is closed")
            if self._poisoned:
                raise RuntimeError(
                    "MoeEp is unusable after an autotune runtime failure"
                )
            if self._training_state is not None:
                raise RuntimeError("MoeEp training is already prepared")
            if (
                self.fc1_weight_layout
                is not MoeEpFc1WeightLayout.GATE_UP_INTERLEAVED_32
            ):
                raise ValueError(
                    "prepare_training requires "
                    "fc1_weight_layout=GATE_UP_INTERLEAVED_32"
                )
            execution_state = self._execution_state
            weight_storage_mode = (
                execution_state.resolved_config.public_config.training_weight_storage_mode
            )
            resolved_device = _resolve_training_device(device)
            resolved = execution_state.resolved_config
            group = resolved.public_config.parallel.ep_group
            if group is not None:
                modes: list[str | None] = [None] * resolved.topology.ep_size
                with torch.cuda.device(resolved_device):
                    dist.all_gather_object(
                        modes,
                        weight_storage_mode.value,
                        group=group,
                    )
                if any(mode != modes[0] for mode in modes[1:]):
                    raise RuntimeError(
                        "MoeEpConfig.training_weight_storage_mode must match "
                        "on every expert-parallel rank; modes by rank: "
                        f"{modes}"
                    )
            _validate_training_assert_capability(resolved)
            from . import _backend

            _backend.validate_config(resolved)
            backend = execution_state.backend
            if backend is not None and resolved_device != backend.device:
                raise ValueError(
                    f"MoeEp backend is bound to {backend.device}; "
                    f"got {resolved_device}"
                )
            if backend is None:
                backend = _backend.create_backend(
                    resolved,
                    resolved_device,
                )
                self._execution_state = replace(
                    execution_state,
                    backend=backend,
                )
            with torch.cuda.device(resolved_device):
                training_state = backend.prepare_training()
            self._training_state = training_state
            self._training_requirements = training_state.public_requirements()
            return self._training_requirements

    def _require_training_prepared(self) -> None:
        if self._closed:
            raise RuntimeError("MoeEp is closed")
        if self._poisoned:
            raise RuntimeError("MoeEp is unusable after an autotune runtime failure")
        if self._training_state is None or self._training_requirements is None:
            raise RuntimeError("prepare_training() must be called first")

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

        resolved = self._execution_state.resolved_config
        validate_forward_source_weights(resolved, weights)
        from ._megamoe_backend.mxfp8._training_weights import materialize_forward

        return materialize_forward(
            weights,
            out=out,
            fc1_weight_layout=resolved.public_config.data_path.fc1_weight_layout,
        )

    def pack_backward_weights(
        self,
        weights: MoeEpBackwardWeights,
        *,
        out: MoeEpBackwardWeightStaging,
    ) -> MoeEpNativeBackwardWeights:
        """Materialize source transpose weights into caller-owned storage."""

        resolved = self._execution_state.resolved_config
        validate_backward_source_weights(resolved, weights)
        from ._megamoe_backend.mxfp8._training_weights import materialize_backward

        return materialize_backward(
            weights,
            out=out,
            fc1_weight_layout=resolved.public_config.data_path.fc1_weight_layout,
        )

    def training_forward(
        self,
        activation: MoeTensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        weights: MoeEpNativeForwardWeights | MoeEpNativeDiscreteForwardWeights,
        out: MoeEpTrainingForwardOutputs,
    ) -> torch.Tensor:
        """Run forward sequentially on this instance's training CUDA stream."""

        with self._lifecycle_lock:
            self._require_training_prepared()
            assert self._training_state is not None
            assert self._training_requirements is not None
            execution_state = self._execution_state
            assert execution_state.backend is not None
            device = execution_state.backend.device
            token_count = validate_training_input(
                execution_state.resolved_config,
                "activation",
                activation,
                topk_idx,
                topk_weights,
                device=device,
                validate_expert_ids=self.validation_mode == "strict",
            )
            if self._training_state.weight_storage_mode == "discrete":
                self._validate_discrete_pointer_lifetime(
                    weights,
                    validate=validate_native_discrete_forward_weights,
                    device=device,
                )
            else:
                validate_native_forward_weights(
                    execution_state.resolved_config,
                    weights,
                    device=device,
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
                device=device,
            )
            validate_training_non_aliasing(
                {
                    **_named_moe_tensors("activation", activation),
                    "topk_idx": topk_idx,
                    "topk_weights": topk_weights,
                    **self._native_weight_tensors(weights),
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

            with torch.cuda.device(device):
                execution = self._training_state.views(
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
        grad_output: MoeTensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        weights: MoeEpNativeBackwardWeights | MoeEpNativeDiscreteBackwardWeights,
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
        """Run backward sequentially using explicit caller-owned forward state."""

        with self._lifecycle_lock:
            self._require_training_prepared()
            assert self._training_state is not None
            execution_state = self._execution_state
            assert execution_state.backend is not None
            device = execution_state.backend.device
            if out is None:
                raise TypeError("out must be a MoeEpTrainingBackwardOutputs")
            token_count = validate_training_input(
                execution_state.resolved_config,
                "grad_output",
                grad_output,
                topk_idx,
                topk_weights,
                device=device,
                validate_expert_ids=self.validation_mode == "strict",
            )
            if self._training_state.weight_storage_mode == "discrete":
                self._validate_discrete_pointer_lifetime(
                    weights,
                    validate=validate_native_discrete_backward_weights,
                    device=device,
                )
            else:
                validate_native_backward_weights(
                    execution_state.resolved_config,
                    weights,
                    device=device,
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
                device=device,
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
                device=device,
            )
            validate_training_non_aliasing(
                {
                    **_named_moe_tensors("grad_output", grad_output),
                    "topk_idx": topk_idx,
                    "topk_weights": topk_weights,
                    **self._native_weight_tensors(weights),
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

            with torch.cuda.device(device):
                execution = self._training_state.views(
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
            state = self._execution_state
            if state.backend is not None:
                close_backend = getattr(state.backend, "close", None)
                if close_backend is not None:
                    close_backend()
                self._execution_state = replace(state, backend=None)
            self._validated_topk_idx = None
            self._validated_topk_version = None
            self._validated_discrete_pointer_tables.clear()
            self._training_state = None
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
    "MoeEpForwardWeightStaging",
    "MoeEpForwardWeights",
    "MoeEpNativeBackwardWeights",
    "MoeEpNativeDiscreteBackwardWeights",
    "MoeEpNativeDiscreteForwardWeights",
    "MoeEpNativeDiscreteWeight",
    "MoeEpNativeForwardWeights",
    "MoeEpNativeWeightStorageMode",
    "MoeEpTrainingBackwardOutputs",
    "MoeEpTrainingForwardOutputs",
    "MoeEpTrainingWgradOperands",
    "MoeFormat",
    "MoeTensor",
    "pack_backward_weights",
    "pack_forward_weights",
]
