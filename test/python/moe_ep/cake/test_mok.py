# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-rank MoK training accuracy, repeats, and capture."""

import hashlib
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from cudnn._cake import loader
from cudnn.moe_ep.cake import create_mok_workspace, prepare_mok_training, quantize_mok_weights

pytestmark = pytest.mark.L0


@pytest.fixture(scope="module")
def group():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("MoK adapter requires SM100")
    try:
        loader.check_toolkit("sm_100a", "test")
    except RuntimeError as error:
        pytest.skip(str(error))
    device = torch.device("cuda", torch.cuda.current_device())
    owned = not dist.is_initialized()
    if owned:
        dist.init_process_group("nccl", store=dist.HashStore(), rank=0, world_size=1, device_id=device)
    elif dist.get_world_size() != 1 or dist.get_backend() != "nccl":
        pytest.skip("this test needs one NCCL rank")
    yield dist.group.WORLD, device
    if owned:
        dist.destroy_process_group()


def _inputs(device):
    torch.manual_seed(19)
    options = dict(device=device, dtype=torch.bfloat16)
    x, dy = (torch.randn(257, 512, **options) * 0.125 for _ in range(2))
    shared = [torch.randn(256, 512, **options) / 512**0.5 for _ in range(2)]
    shared.append(torch.randn(512, 256, **options) / 256**0.5)
    routed = [torch.randn(4, *w.shape, **options) / w.shape[1] ** 0.5 for w in shared]
    ids = torch.arange(2, device=device, dtype=torch.int64).expand(257, 2).contiguous()
    scores = torch.rand(257, 2, device=device) + 0.1
    return x, dy, ids, scores, (*shared, *routed)


def _reference(x, dy, ids, scores, weights, limit):
    x, scores = x.double().requires_grad_(), scores.double().requires_grad_()
    weights = [w.double().requires_grad_() for w in weights]

    def expert(gate, up, down):
        g, u = x @ gate.T, x @ up.T
        if limit is not None:
            g, u = g.clamp(max=limit), u.clamp(-limit, limit)
        return (torch.nn.functional.silu(g) * u) @ down.T

    out = expert(*weights[:3])
    for e in range(4):
        selected_scores = ((ids == e) * scores).sum(-1, keepdim=True)
        out = out + selected_scores * expert(*(w[e] for w in weights[3:]))
    grads = torch.autograd.grad(out, (x, scores, *weights), dy.double())
    return (out.detach(), grads[0], grads[1], *grads[5:], *grads[2:5])


@pytest.mark.parametrize("mxfp8,fp32_wgrad,limit", [(False, False, None), (False, True, None), (False, False, 0.2), (True, False, None)])
def test_training(group, mxfp8, fp32_wgrad, limit, monkeypatch):
    process_group, device = group
    x, dy, ids, scores, weights = _inputs(device)
    adapter = prepare_mok_training(device=device, clamped_swiglu=limit is not None, fp32_wgrad=fp32_wgrad)
    adapter.prepare(limit, mxfp8)
    adapter.prepare_recompute(limit, mxfp8)
    config, workspace = create_mok_workspace(
        group=process_group,
        device=device,
        num_local_tokens=x.shape[0],
        hidden_size=512,
        minibatch_size=256,
        macrobatch_size=768,
        schedule_capacity_multiplier=1.0,
        fwd_num_comm_sms=24,
        bwd_num_comm_sms=28,
    )
    fw = bw = weights
    if mxfp8:
        forward_weights, backward_weights = quantize_mok_weights(*weights[3:])
        fw, bw = (*weights[:3], *forward_weights), (*weights[:3], *backward_weights)
    accumulators = [torch.zeros(w.shape, dtype=torch.float32, device=device) for w in weights] if fp32_wgrad else None

    def run(recompute=False, reset=True):
        schedule = adapter.build_schedule(workspace, config, ids, num_local_experts=4)
        y, context = adapter.forward(config, workspace, schedule, x, scores, *fw, swiglu_limit=limit)
        if recompute:
            context = adapter.recompute_forward_context(config, workspace, schedule, x, fw[0], fw[1], fw[3], fw[4], swiglu_limit=limit)
        if accumulators is not None and reset:
            for a in accumulators:
                a.zero_()
        grads = adapter.backward(config, workspace, schedule, context, dy, x, scores, *bw, swiglu_limit=limit, weight_grad_accumulators=accumulators)
        return (y, *grads)

    got = tuple(v.clone() for v in run())
    ref = _reference(x, dy, ids, scores, weights, limit)
    for index, (value, expected) in enumerate(zip(got, ref)):
        bound = 0.15 if mxfp8 and index < 6 else (0.07 if limit is not None else 0.025)
        error = (value.double() - expected).norm() / expected.norm().clamp_min(1e-10)
        assert error < bound, (index, float(error), bound)
    if fp32_wgrad:
        accumulated = run(reset=False)
        for value, previous in zip(accumulated[3:], got[3:]):
            torch.testing.assert_close(value, 2 * previous, atol=1e-7, rtol=1e-5)
        schedule = adapter.build_schedule(workspace, config, ids, num_local_experts=4)
        _, context = adapter.forward(config, workspace, schedule, x, scores, *fw)

        def forbidden(*args, **kwargs):
            raise AssertionError("invalid accumulators reached native backward")

        with monkeypatch.context() as patch:
            patch.setattr(adapter, "_inputs", forbidden)
            overlapping = [accumulators[0], accumulators[0], *accumulators[2:]]
            with pytest.raises(ValueError, match="alias"):
                adapter.backward(config, workspace, schedule, context, dy, x, scores, *bw, weight_grad_accumulators=overlapping)
            overlapping = [weights[0].view(torch.float32), *accumulators[1:]]
            with pytest.raises(ValueError, match="alias"):
                adapter.backward(config, workspace, schedule, context, dy, x, scores, *bw, weight_grad_accumulators=overlapping)
    for value, previous in zip(run(recompute=True), got):
        assert torch.equal(value.view(torch.uint8), previous.view(torch.uint8))
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        side = run()
    stream.synchronize()
    for value, previous in zip(side, got):
        assert torch.equal(value.view(torch.uint8), previous.view(torch.uint8))
    graph = torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=stream):
        captured = run()
    graph.replay()
    for value, previous in zip(captured, got):
        assert torch.equal(value.view(torch.uint8), previous.view(torch.uint8))
    if not mxfp8 and not fp32_wgrad and limit is None:
        x.mul_(0.5)
        ids.add_(2)
        graph.replay()
        changed = _reference(x, dy, ids, scores, weights, limit)
        for value, expected in zip(captured, changed):
            assert (value.double() - expected).norm() / expected.norm().clamp_min(1e-10) < 0.025


def test_declines_unqualified_layouts():
    with pytest.raises(NotImplementedError, match="EP1"):
        prepare_mok_training(ep_size=4)
    with pytest.raises(TypeError, match="bool"):
        prepare_mok_training(fp32_wgrad=1)


def test_public_exports():
    import cudnn.ops as ops

    for helper in (prepare_mok_training, create_mok_workspace, quantize_mok_weights):
        assert getattr(ops, helper.__name__) is helper


def test_source_hashes():
    vendor = Path(__file__).resolve().parents[4] / "python/cudnn/_cake/_vendor"
    record = json.loads((vendor / "VENDORED_MOK.json").read_text())
    for name, expected in record["files"].items():
        assert hashlib.sha256((vendor / "cake_mok_bf16" / name).read_bytes()).hexdigest() == expected
