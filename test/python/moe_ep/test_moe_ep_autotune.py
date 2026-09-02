# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Contracts and smoke coverage for the explicit MoeEP sweep autotuner."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

from moe_ep.moe_ep_test_support import (
    _allocate_stateless_training_outputs,
    _allocate_training_weight_staging,
    _assert_backward_matches,
    _assert_matches_reference,
    _fixed_training_reference,
    _fixed_training_weights,
    _forward_config,
    _grad_output,
    _reference_forward,
    _replay_cuda_graph,
    _sm107_device,
    make_forward_inputs,
)


def _validated_request(config, *args, **kwargs):
    del args, kwargs
    return SimpleNamespace(config=config, device=torch.device("cuda", 0))


def _patch_common_inference_dependencies(
    patch,
    *,
    api_module,
    backend_module,
    create_backend,
) -> None:
    patch.setattr(api_module, "validate_forward", _validated_request)
    patch.setattr(backend_module, "validate_config", lambda config: None)
    patch.setattr(backend_module, "validate_request", lambda request: None)
    patch.setattr(backend_module, "create_backend", create_backend)
    patch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    patch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


@pytest.mark.L0
def test_autotune_core_contracts(monkeypatch):
    import cudnn.moe_ep._autotune as autotune_module
    from cudnn import (
        MoeEpAutotuneCandidateResult,
        MoeEpAutotuneResult,
        MoeEpTuningConfig,
    )

    baseline = MoeEpTuningConfig()
    candidate = MoeEpTuningConfig(token_in_flag_batch=2)
    normalize = autotune_module.normalize_candidates
    assert normalize(
        baseline,
        [candidate, baseline, candidate],
        warmup_iters=0,
        timed_iters=1,
        max_candidates=2,
    ) == (baseline, candidate)

    invalid_limits = (
        ({"warmup_iters": -1}, "warmup_iters"),
        ({"timed_iters": 0}, "timed_iters"),
        ({"max_candidates": 33}, "max_candidates"),
    )
    for overrides, message in invalid_limits:
        arguments = {
            "warmup_iters": 0,
            "timed_iters": 1,
            "max_candidates": 32,
            **overrides,
        }
        with pytest.raises(ValueError, match=message):
            normalize(baseline, [baseline], **arguments)

    with pytest.raises(ValueError, match="does not sweep reduce_topk_in_kernel"):
        normalize(
            baseline,
            [MoeEpTuningConfig(reduce_topk_in_kernel=True)],
            warmup_iters=0,
            timed_iters=1,
            max_candidates=32,
        )
    with pytest.raises(ValueError, match="exceeding max_candidates=1"):
        normalize(
            baseline,
            [candidate],
            warmup_iters=0,
            timed_iters=1,
            max_candidates=1,
        )

    first = MoeEpAutotuneCandidateResult(baseline, 1.0, (1.0,))
    second = MoeEpAutotuneCandidateResult(candidate, 1.0, (1.0,))
    assert autotune_module.select_winner((first, second)) is first
    result = MoeEpAutotuneResult("inference", first.tuning, (first, second))
    assert result.evaluated_candidates == 2

    with monkeypatch.context() as patch:
        remote = (candidate,)
        patch.setattr(autotune_module.dist, "get_world_size", lambda group: 2)

        def gather(output, value, *, group):
            del group
            output[:] = [value, remote]

        patch.setattr(autotune_module.dist, "all_gather_object", gather)
        with pytest.raises(RuntimeError, match="must match on every EP rank"):
            autotune_module.verify_candidates_across_ranks((baseline,), object())

    with monkeypatch.context() as patch:
        local_samples = iter((1.0, 7.0, 3.0))

        class Event:
            def record(self, stream):
                del stream

            def synchronize(self):
                pass

            def elapsed_time(self, end):
                del end
                return next(local_samples)

        patch.setattr(torch.cuda, "current_stream", lambda device: object())
        patch.setattr(torch.cuda, "Event", lambda enable_timing: Event())
        patch.setattr(
            autotune_module.dist,
            "all_reduce",
            lambda values, **kwargs: values.copy_(
                torch.tensor([5.0, 8.0, 4.0], dtype=values.dtype)
            ),
        )
        latency, samples = autotune_module.benchmark_candidate(
            lambda: None,
            device=torch.device("cpu"),
            group=object(),
            timed_iters=3,
        )
        assert samples == (5.0, 8.0, 4.0)
        assert latency == 5.0


@pytest.mark.L0
def test_autotune_api_transactions(monkeypatch):
    import cudnn.moe_ep._autotune as autotune_module
    import cudnn.moe_ep._backend as backend_module
    import cudnn.moe_ep._megamoe_backend.mxfp8._training_execute as execute_module
    import cudnn.moe_ep.api as api_module
    from cudnn import MoeEp, MoeEpTuningConfig

    baseline = MoeEpTuningConfig()
    candidate = MoeEpTuningConfig(token_in_flag_batch=2)

    # Validation failures happen before teardown and preserve active state.
    op = MoeEp(**_forward_config())
    active_backend = object()
    op._forward_backend = active_backend
    with pytest.raises(ValueError, match="does not sweep reduce_topk_in_kernel"):
        op.autotune(
            None,
            None,
            None,
            None,
            None,
            candidates=[MoeEpTuningConfig(reduce_topk_in_kernel=True)],
            warmup_iters=0,
            timed_iters=1,
        )
    assert op.tuning == baseline
    assert op._forward_backend is active_backend
    op._forward_backend = None
    op.close()

    # A runtime failure is fail-fast and permanently poisons the instance.
    with monkeypatch.context() as patch:
        calls = []

        class FailingBackend:
            def forward(self, request):
                calls.append(request.config.tuning)
                raise RuntimeError("launch failed")

            def close(self):
                pass

        _patch_common_inference_dependencies(
            patch,
            api_module=api_module,
            backend_module=backend_module,
            create_backend=lambda config, device: FailingBackend(),
        )
        patch.setattr(
            autotune_module,
            "verify_state_across_ranks",
            lambda state, group: None,
        )
        op = MoeEp(**_forward_config())
        with pytest.raises(RuntimeError, match="candidate 0.*compile/prime"):
            op.autotune(
                None,
                None,
                None,
                None,
                None,
                candidates=[candidate],
                warmup_iters=0,
                timed_iters=1,
            )
        assert calls == [baseline]
        with pytest.raises(RuntimeError, match="unusable"):
            op.autotune(
                None,
                None,
                None,
                None,
                None,
                candidates=[candidate],
                warmup_iters=0,
                timed_iters=1,
            )
        op._forward_backend = None
        op.close()

    # Inference commits only the measured winner and retains its rebuilt backend.
    with monkeypatch.context() as patch:
        active_backends = []

        class InferenceBackend:
            def __init__(self, config):
                self.config = config
                self.closed = False

            def forward(self, request):
                assert request.config == self.config
                return object()

            def close(self):
                self.closed = True

        def create_inference_backend(config, device):
            del device
            active_backends.append(InferenceBackend(config))
            return active_backends[-1]

        def benchmark_inference(run, *, device, group, timed_iters):
            del device, group, timed_iters
            run()
            tuning = active_backends[-1].config.tuning
            latency = 1.0 if tuning == candidate else 2.0
            return latency, (latency,)

        _patch_common_inference_dependencies(
            patch,
            api_module=api_module,
            backend_module=backend_module,
            create_backend=create_inference_backend,
        )
        patch.setattr(
            autotune_module,
            "benchmark_candidate",
            benchmark_inference,
        )
        patch.setattr(
            autotune_module,
            "synchronize_candidate",
            lambda device, group: None,
        )
        op = MoeEp(**_forward_config())
        result = op.autotune(
            None,
            None,
            None,
            None,
            None,
            candidates=[candidate],
            warmup_iters=0,
            timed_iters=1,
        )
        assert result.winner == candidate
        assert result.evaluated_candidates == 2
        assert op.tuning == op._forward_config.tuning == candidate
        assert op._forward_backend is active_backends[-1]
        assert not active_backends[-1].closed
        op.close()

    # Training times forward/backward pairs and leaves preparation to the caller.
    with monkeypatch.context() as patch:
        launches = []
        active_backends = []
        requirement_names = (
            "output",
            "fc1_preact",
            "fc1_a",
            "fc1_sfa",
            "valid_route_counts",
            "expert_offsets",
            "grad_activation",
            "dprob",
            "fc1_b",
            "fc1_sfb",
            "fc2_a",
            "fc2_sfa",
            "fc2_b",
            "fc2_sfb",
        )

        class TrainingState:
            def public_requirements(self):
                return {name: None for name in requirement_names}

            def views(self, *, lane, token_count):
                return lane, token_count

        class TrainingBackend:
            def __init__(self, config):
                self.config = config

            def prepare_training(self, *, lane_count):
                assert lane_count == 1
                return TrainingState()

            def close(self):
                pass

        forward_outputs = SimpleNamespace(
            fc1_preact=object(),
            output=object(),
            fc1_a=object(),
            fc1_sfa=object(),
            valid_route_counts=object(),
            expert_offsets=object(),
        )
        backward_outputs = SimpleNamespace()

        patch.setattr(
            api_module,
            "_validate_training_assert_capability",
            lambda config: None,
        )
        for validation_name in (
            "validate_native_forward_weights",
            "validate_native_backward_weights",
            "validate_training_forward_outputs",
            "validate_training_backward_outputs",
            "validate_training_forward_state",
        ):
            patch.setattr(
                api_module,
                validation_name,
                lambda *args, **kwargs: None,
            )
        patch.setattr(
            api_module,
            "validate_training_input",
            lambda *args, **kwargs: 2,
        )
        patch.setattr(backend_module, "validate_config", lambda config: None)

        def create_training_backend(config, device):
            del device
            active_backends.append(TrainingBackend(config))
            return active_backends[-1]

        patch.setattr(backend_module, "create_backend", create_training_backend)
        patch.setattr(
            autotune_module,
            "allocate_training_outputs",
            lambda requirements, device: (forward_outputs, backward_outputs),
        )
        patch.setattr(
            autotune_module,
            "synchronize_candidate",
            lambda device, group: None,
        )

        def benchmark_training(run, *, device, group, timed_iters):
            del device, group, timed_iters
            run()
            tuning = active_backends[-1].config.tuning
            latency = 1.0 if tuning == candidate else 2.0
            return latency, (latency,)

        patch.setattr(
            autotune_module,
            "benchmark_candidate",
            benchmark_training,
        )
        patch.setattr(
            torch.cuda,
            "device",
            lambda device: contextlib.nullcontext(),
        )
        patch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
        patch.setattr(
            execute_module,
            "launch_training_forward",
            lambda *args, **kwargs: launches.append("forward"),
        )
        patch.setattr(
            execute_module,
            "launch_training_backward",
            lambda *args, **kwargs: launches.append("backward"),
        )

        op = MoeEp(**_forward_config(), weight_interleave_size=32)
        value = SimpleNamespace(device=torch.device("cuda", 0))
        result = op.autotune_training(
            value,
            value,
            None,
            None,
            forward_weights=object(),
            backward_weights=object(),
            candidates=[candidate],
            warmup_iters=0,
            timed_iters=1,
        )
        assert result.mode == "training"
        assert result.winner == candidate
        assert launches == ["forward", "backward"] * 4
        assert op._training_state is None
        assert op._forward_backend is None
        op.close()


def _print_candidate_timings(label, result) -> None:
    print(f"\n{label} autotune timings:", flush=True)
    for index, measurement in enumerate(result.candidates):
        samples = ", ".join(f"{sample:.4f}" for sample in measurement.samples_ms)
        print(
            f"  [{index}] median={measurement.latency_ms:.4f} ms "
            f"samples=[{samples}] tuning={measurement.tuning}",
            flush=True,
        )


@pytest.mark.L1
@pytest.mark.gpu_exclusive
def test_autotune_sm107_inference_training_and_graph():
    from cudnn import MoeEp, MoeEpTuningConfig

    device = _sm107_device()
    candidates = [
        MoeEpTuningConfig(),
        MoeEpTuningConfig(token_in_flag_batch=2),
        MoeEpTuningConfig(token_in_flag_batch=4),
        MoeEpTuningConfig(epi_flag_batch=(2, 1)),
        MoeEpTuningConfig(group_hint=64),
    ]

    inference_args = make_forward_inputs(device)
    inference_expected = _reference_forward(inference_args)
    original_topk_idx = inference_args[3].clone()
    with MoeEp(**_forward_config()) as op:
        result = op.autotune(
            *inference_args,
            candidates=candidates,
            warmup_iters=1,
            timed_iters=2,
        )
        _print_candidate_timings("inference", result)
        actual = op(*inference_args)
        torch.cuda.synchronize(device)
        assert result.evaluated_candidates == len(candidates) == 5
        assert result.winner in candidates
        assert op.tuning == result.winner
        assert op._forward_backend is not None
        _assert_matches_reference(actual, inference_expected)
        _replay_cuda_graph(
            op,
            inference_args,
            original_topk_idx,
            inference_expected,
            device,
        )

    base_args = make_forward_inputs(device)
    training_args = (
        base_args[0].dequantize(torch.bfloat16),
        base_args[1],
        base_args[2],
        base_args[3],
        base_args[4].float().contiguous(),
    )
    grad_output = _grad_output(
        device,
        training_args[0].shape[0],
        seed=20260903,
    )
    training_expected = _fixed_training_reference(
        training_args,
        grad_output,
        combine_format="bf16",
        gate_up_clamp=None,
    )
    source_weights = _fixed_training_weights(training_args)
    with MoeEp(
        num_experts=2,
        hidden_size=128,
        intermediate_size=256,
        top_k=2,
        max_tokens_per_rank=training_args[0].shape[0],
        max_recv_size_per_rank=(training_args[0].shape[0] * training_args[3].shape[1]),
        drop_on_overflow=True,
        combine_format="bf16",
        weight_interleave_size=32,
    ) as op:
        forward_staging, backward_staging = _allocate_training_weight_staging(
            source_weights
        )
        native_forward = op.pack_forward_weights(
            source_weights[0],
            out=forward_staging,
        )
        native_backward = op.pack_backward_weights(
            source_weights[1],
            out=backward_staging,
        )
        result = op.autotune_training(
            training_args[0],
            grad_output,
            training_args[3],
            training_args[4],
            forward_weights=native_forward,
            backward_weights=native_backward,
            candidates=candidates,
            warmup_iters=1,
            timed_iters=2,
        )
        _print_candidate_timings("training", result)
        requirements = op.prepare_training(lane_count=1, device=device)
        forward_out, backward_out = _allocate_stateless_training_outputs(
            requirements,
            device,
        )
        lane = op.training_lanes[0]
        actual_y = op.training_forward(
            lane,
            training_args[0],
            training_args[3],
            training_args[4],
            weights=native_forward,
            out=forward_out,
        )
        actual_dx, actual_dprob, _ = op.training_backward(
            lane,
            grad_output,
            training_args[3],
            training_args[4],
            weights=native_backward,
            fc1_preact=forward_out.fc1_preact,
            fc1_a=forward_out.fc1_a,
            fc1_sfa=forward_out.fc1_sfa,
            valid_route_counts=forward_out.valid_route_counts,
            expert_offsets=forward_out.expert_offsets,
            out=backward_out,
        )
        torch.cuda.synchronize(device)
        assert result.evaluated_candidates == len(candidates) == 5
        assert result.winner in candidates
        assert op.tuning == result.winner
        _assert_matches_reference(actual_y, training_expected[0])
        _assert_backward_matches(
            (actual_dx, actual_dprob),
            (training_expected[1], training_expected[2]),
            training_args[3],
        )
