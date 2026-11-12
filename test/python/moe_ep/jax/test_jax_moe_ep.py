# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real EP2 JAX inference and explicit-training checks against the EP oracle.

The compiled graphs contain native FFI calls and DLPack inputs, without XLA
arithmetic kernels. This filename avoids the ``*_jax.py`` gate for XLA's
generated-PTX support; native CuTe compilation selects SM107 explicitly.
"""

from datetime import timedelta
import faulthandler
import os
import traceback

import pytest

jax = pytest.importorskip("jax", minversion="0.9.1")
torch = pytest.importorskip("torch")

import torch.distributed as dist
import torch.multiprocessing as mp

from determinism_utils import assert_bitwise, snapshot
from moe_ep.moe_ep_test_support import (
    _allocate_training_weight_staging,
    _assert_backward_matches,
    _assert_matches_reference,
    _assert_wgrads_match_reference,
    _fixed_training_reference,
    _fixed_training_weights,
    _grad_output,
    _moe_ep_config,
    _require_distributed_sm107,
    _unpack_wgrad_scale_part_bytes,
    make_distributed_forward_inputs,
)

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


def _jax_weights(args):
    from cudnn.moe_ep.api import pack_backward_weights, pack_forward_weights
    from cudnn.moe_ep.jax_api import MoeEpJaxBackwardWeights, MoeEpJaxForwardWeights

    sources = _fixed_training_weights(args)
    forward_out, backward_out = _allocate_training_weight_staging(sources)
    forward = pack_forward_weights(sources[0], out=forward_out)
    backward = pack_backward_weights(sources[1], out=backward_out)

    def carrier(value):
        return jax.dlpack.from_dlpack(value.view(torch.uint8))

    return (
        MoeEpJaxForwardWeights(
            carrier(forward.fc1.payload.transpose(1, 2)),
            carrier(forward.fc1.scale),
            carrier(forward.fc2.payload.transpose(1, 2)),
            carrier(forward.fc2.scale),
        ),
        MoeEpJaxBackwardWeights(
            carrier(backward.w2_transpose.payload),
            carrier(backward.w2_transpose.scale),
            carrier(backward.w1_transpose.payload),
            carrier(backward.w1_transpose.scale),
        ),
    )


def _torch_operands(operands):
    from cudnn.moe_ep import MoeEpTrainingWgradOperands

    values = {name: torch.from_dlpack(value) for name, value in operands._asdict().items()}
    for name in ("fc1_a", "fc1_b", "fc2_a", "fc2_b"):
        values[name] = values[name].view(torch.float8_e4m3fn)
    # JAX exposes the physical bytes; the shared oracle expects token-major
    # logical coordinates, including FC2's column-major gradient operand.
    values["fc2_b"] = values["fc2_b"].transpose(0, 1)
    return MoeEpTrainingWgradOperands(**values)


def _torch_results(result):
    inference, forward, backward = result
    values = {
        "inference": torch.from_dlpack(inference.output),
        "forward": torch.from_dlpack(forward.output),
        "dx": torch.from_dlpack(backward.grad_activation),
        "dprob": torch.from_dlpack(backward.dprob),
        "inference_status": torch.from_dlpack(inference.status),
        "forward_status": torch.from_dlpack(forward.status),
        "backward_status": torch.from_dlpack(backward.status),
    }
    for name in ("fc1_preact", "fc1_a", "fc1_sfa", "valid_route_counts", "expert_offsets"):
        values["saved." + name] = torch.from_dlpack(getattr(forward, name))
    values.update({"wgrad." + name: torch.from_dlpack(value) for name, value in backward.wgrad._asdict().items()})
    return values


def _execute_pair(plan, inputs, weights):
    inference = jax.jit(plan.inference)
    forward = jax.jit(plan.training_forward)
    backward = jax.jit(plan.training_backward)

    def run(values, weights):
        first, second = values
        x, indices, scores, dy = first
        y0 = inference(x, indices, scores, weights[0])
        saved0 = forward(x, indices, scores, weights[0], dependency=y0.status)
        x, indices, scores, dy = second
        y1 = inference(x, indices, scores, weights[0], dependency=saved0.status)
        saved1 = forward(x, indices, scores, weights[0], dependency=y1.status)
        # Consume the first saved state after another forward has reused the
        # plan's symmetric scratch. The saved state itself belongs to XLA.
        dx0 = backward(first[3], first[1], first[2], weights[1], saved0, dependency=saved1.status)
        dx1 = backward(dy, indices, scores, weights[1], saved1, dependency=dx0.status)
        return (y0, saved0, dx0), (y1, saved1, dx1)

    # Side-effecting FFI calls lack JAX's C++ dispatch fast path. Compile an
    # executable explicitly so no_tracing checks execution without retracing.
    compiled = jax.jit(run).lower(inputs, weights).compile()
    first = jax.block_until_ready(compiled(inputs, weights))
    with jax.no_tracing(True):
        second = jax.block_until_ready(compiled(inputs[::-1], weights))
    return first, second, compiled


def _check_result(actual, expected, indices):
    tensors = _torch_results(actual)
    for name in ("inference_status", "forward_status", "backward_status"):
        torch.testing.assert_close(tensors[name], torch.zeros_like(tensors[name]), rtol=0, atol=0)
    y, dx, dprob, expected_operands, expected_dense = expected
    _assert_matches_reference(tensors["inference"], y)
    _assert_matches_reference(tensors["forward"], y)
    _assert_backward_matches((tensors["dx"], tensors["dprob"]), (dx, dprob), indices)
    operands = _torch_operands(actual[2].wgrad)
    _assert_wgrads_match_reference(operands, expected_operands, expected_dense=expected_dense, weight_interleave_size=32)
    counts = expected_operands.valid_route_counts
    expected_offsets = (((counts + 127) // 128) * 128).cumsum(0).to(torch.int32)
    torch.testing.assert_close(operands.expert_offsets, expected_offsets, rtol=0, atol=0)


def _defined_results(values):
    # Native TMA stores include unspecified padding. Compare every defined
    # payload/scale byte, while retained-allocation checks cover full storage.
    result = {name: value for name, value in values.items() if not name.startswith(("saved.", "wgrad."))}
    for prefix in ("saved", "wgrad"):
        for name in ("valid_route_counts", "expert_offsets"):
            result[f"{prefix}.{name}"] = values[f"{prefix}.{name}"]
    counts = values["saved.valid_route_counts"].tolist()
    offsets = values["saved.expert_offsets"].tolist()
    starts = [0, *offsets[:-1]]
    for expert, (start, count) in enumerate(zip(starts, counts)):
        result[f"preact.{expert}"] = values["saved.fc1_preact"][start : start + count]
    for prefix, payload, scales in (
        ("saved", "fc1_a", "fc1_sfa"),
        ("wgrad", "fc1_a", "fc1_sfa"),
        ("wgrad", "fc1_b", "fc1_sfb"),
        ("wgrad", "fc2_a", "fc2_sfa"),
        ("wgrad", "fc2_b", "fc2_sfb"),
    ):
        data = values[f"{prefix}.{payload}"]
        if payload == "fc2_b":
            data = data.transpose(0, 1)
        flat_scales = values[f"{prefix}.{scales}"].reshape(-1)
        non_k = data.shape[1]
        scale_offset = 0
        for expert, (start, end, count) in enumerate(zip(starts, offsets, counts)):
            result[f"{prefix}.{payload}.{expert}"] = data[start : start + count]
            columns = (end - start) // 32
            scale_bytes = ((non_k + 127) // 128 * 128) * ((columns + 3) // 4 * 4)
            packed = flat_scales[scale_offset : scale_offset + scale_bytes]
            used_scales = _unpack_wgrad_scale_part_bytes(packed, non_k, columns) if columns else packed
            result[f"{prefix}.{scales}.{expert}"] = used_scales[:, : (count + 31) // 32] if columns else used_scales
            scale_offset += scale_bytes
    return result


def _ep2_worker(rank, init_file, replay):
    from cudnn import MoeEpFc1WeightLayout
    from cudnn.moe_ep._megamoe_backend._runtime import RuntimeWorld, _DefaultNvshmemRuntimeProvider
    from cudnn.moe_ep.jax_api import prepare_moe_ep_jax

    timeout = float(os.environ.get("CUDNN_TEST_TIMEOUT", "1500"))
    if timeout > 0:
        faulthandler.dump_traceback_later(timeout * 0.8, exit=True)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", init_method=f"file://{init_file}", rank=rank, world_size=2, device_id=device, timeout=timedelta(seconds=180))
    provider = _DefaultNvshmemRuntimeProvider()
    initialized = False
    plan = None
    try:
        target = next(d for d in jax.local_devices() if d.platform == "gpu" and d.local_hardware_id == rank)
        with jax.default_device(target), torch.cuda.device(device):
            args = make_distributed_forward_inputs(rank, 2, device)
            x = args[0].dequantize().to(torch.bfloat16)
            args = (x, args[1], args[2], args[3], args[4].float())
            alternate = (-x, args[1], args[2], (args[3] + 1).remainder(4), args[4].flip(1).contiguous())
            cases = (args, alternate)
            gradients = tuple(_grad_output(device, x.shape[0], seed=811 + rank + variant * 2) for variant in range(2))
            expected = []
            for case, gradient in zip(cases, gradients):
                reference = _fixed_training_reference(
                    case,
                    gradient,
                    combine_format="bf16",
                    gate_up_clamp=None,
                    ep_group=dist.group.WORLD,
                    num_experts=4,
                    max_tokens_per_rank=8,
                    physical_recv_pool_rows=256,
                    drop_on_overflow=True,
                )
                expected.append((*reference, reference[3].dense_wgrads()))
            weights = _jax_weights(args)
            inputs = tuple(tuple(jax.dlpack.from_dlpack(value) for value in (case[0], case[3], case[4], gradient)) for case, gradient in zip(cases, gradients))
            originals = {f"input{case}.{field}": torch.from_dlpack(value) for case, values in enumerate(inputs) for field, value in enumerate(values)}
            originals.update({f"weight{kind}.{field}": torch.from_dlpack(value) for kind, values in enumerate(weights) for field, value in enumerate(values)})
            before = snapshot(originals)
            provider.initialize(device, RuntimeWorld(rank, 2, dist.group.WORLD, (0, 1)))
            initialized = True
            config = _moe_ep_config(
                num_experts=4,
                hidden_size=128,
                intermediate_size=256,
                top_k=2,
                max_tokens_per_rank=8,
                physical_recv_pool_rows=256,
                drop_on_overflow=True,
                fc1_weight_layout=MoeEpFc1WeightLayout.GATE_UP_INTERLEAVED_32,
            )
            plan = prepare_moe_ep_jax(config, device=target, rank=rank, world_size=2)
            first, changed, compiled = _execute_pair(plan, inputs, weights)
            dist.barrier()
            retained = tuple({name: value.clone() for name, value in _torch_results(result).items()} for result in first)
            replays = []
            if replay:
                stream = torch.cuda.Stream(device=device)
                noise = torch.ones(1024 * 1024, device=device)
                for _ in range(3):
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(32):
                            noise.mul_(1.00001).add_(0.00001)
                    result = jax.block_until_ready(compiled(inputs, weights))
                    torch.cuda.current_stream().wait_stream(stream)
                    replays.append(tuple({name: tensor.clone() for name, tensor in _torch_results(value).items()} for value in result))
            dist.barrier()
            assertion_error = None
            try:
                for index in range(2):
                    _check_result(first[index], expected[index], cases[index][3])
                    _check_result(changed[index], expected[1 - index], cases[1 - index][3])
                    assert_bitwise(_torch_results(first[index]), retained[index], "retained output mutation")
                    for repeat, values in enumerate(replays):
                        assert_bitwise(_defined_results(values[index]), _defined_results(retained[index]), f"EP2 output replay {repeat + 1}")
                assert_bitwise(snapshot(originals), before, "input mutation")
            except AssertionError as exc:
                assertion_error = exc
            finally:
                dist.barrier()
            if assertion_error is not None:
                raise assertion_error
    except Exception:
        traceback.print_exc()
        raise
    finally:
        from cudnn._device import ensure_current_context

        restored_device = True
        if plan is not None:
            with torch.cuda.device(1 - rank):
                plan.close()
                restored_device = torch.cuda.current_device() == 1 - rank
        with torch.cuda.device(device):
            if initialized:
                ensure_current_context(device=rank)
                provider.finalize()
            dist.destroy_process_group()
        faulthandler.cancel_dump_traceback_later()
        assert restored_device, "closing the JAX plan did not restore the caller's CUDA device"


def _spawn_ep2(tmp_path, monkeypatch, *, replay):
    _require_distributed_sm107(2)
    monkeypatch.setenv("NVIDIA_IMEX_CHANNELS", "0")
    mp.spawn(_ep2_worker, args=(str(tmp_path / "jax_ep2.init"), replay), nprocs=2, join=True)


@pytest.mark.L1
def test_ep2_inference_and_explicit_backward_match_reference(tmp_path, monkeypatch):
    _spawn_ep2(tmp_path, monkeypatch, replay=False)


@pytest.mark.L0
def test_ep2_forward_and_backward_replay(tmp_path, monkeypatch):
    """Compare every defined output and saved carrier byte, excluding native padding."""
    _spawn_ep2(tmp_path, monkeypatch, replay=True)
