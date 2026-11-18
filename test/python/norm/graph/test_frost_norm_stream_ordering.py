# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``execute`` must run on the stream the handle was set to.

The engines reach the GPU two ways -- CuTe kernel launches and torch ``copy_``
into the graph's output buffers -- and both follow torch's *current* stream.
``ExecutionContext.stream`` carries the stream cuDNN was asked to use via
``cudnn.set_stream(handle, ...)``, which need not be torch's current one, and
the plan used to discard it.

Nothing about that is loud. The graph runs, every output is finite, and the
only evidence is ordering: work the caller queued on the requested stream is
not waited for, so the norm reads whatever was in the input *before* its
producer ran.

The probe below makes that deterministic rather than racy. A ``_sleep`` of ~50 ms
is queued on the requested stream ahead of the write that produces the input, so
an execute that ignores the stream cannot accidentally be late enough to look
correct -- it reads the old input every time.
"""

import pytest
import torch

import cudnn

pytest.importorskip("cudnn.norm", reason="frost norm engines require a built cudnn frontend")

_HEUR = [cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]
_PY_ENGINE_FLOOR = 20_000
_SLEEP_CYCLES = 100_000_000  # ~50 ms at 2 GHz: far longer than the norm itself


def _build(handle, x_gpu, scale, bias):
    D = x_gpu.shape[-1]
    eps_cpu = torch.full((1, 1, 1), 1e-5, dtype=torch.float32)
    g = cudnn.pygraph(
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=handle,
    )
    X = g.tensor(name="X", dim=x_gpu.size(), stride=x_gpu.stride(), data_type=x_gpu.dtype)
    SC = g.tensor(name="scale", dim=scale.size(), stride=scale.stride(), data_type=scale.dtype)
    BI = g.tensor(name="bias", dim=bias.size(), stride=bias.stride(), data_type=bias.dtype)
    EP = g.tensor_like(eps_cpu)
    Y, MEAN, IVAR = g.layernorm(name="LN", norm_forward_phase=cudnn.norm_forward_phase.TRAINING, input=X, scale=SC, bias=BI, epsilon=EP)
    B, S = x_gpu.shape[0], x_gpu.shape[1]
    Y.set_output(True).set_data_type(x_gpu.dtype)
    Y.set_dim((B, S, D))
    Y.set_stride(x_gpu.stride())
    for t in (MEAN, IVAR):
        t.set_output(True).set_data_type(torch.float32)
        t.set_dim((B, S, 1))
        t.set_stride((S, 1, 1))

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans(_HEUR)
    idx = next((i for i, p in enumerate(g.plans) if p.engine_id >= _PY_ENGINE_FLOOR), None)
    if idx is None:
        pytest.skip("frost norm engine proposed no plan for this graph")
    g.select_plan(idx)
    g.check_support()
    g.build_plans()
    return g, (X, SC, BI, EP, Y, MEAN, IVAR)


@pytest.mark.L0
def test_execute_is_ordered_against_the_handle_stream():
    """A producer queued on the handle's stream must be waited for."""
    B, S, D = 4, 8, 256
    torch.manual_seed(0)
    x = torch.randn(B, S, D, device="cuda", dtype=torch.float32)
    fresh = torch.randn(B, S, D, device="cuda", dtype=torch.float32)
    scale = torch.ones(1, 1, D, device="cuda", dtype=torch.float32)
    bias = torch.zeros(1, 1, D, device="cuda", dtype=torch.float32)
    eps_cpu = torch.full((1, 1, 1), 1e-5, dtype=torch.float32)

    side = torch.cuda.Stream()
    handle = cudnn.create_handle()
    cudnn.set_stream(handle=handle, stream=side.cuda_stream)
    g, (X, SC, BI, EP, Y, MEAN, IVAR) = _build(handle, x, scale, bias)

    y = torch.empty_like(x)
    mean = torch.empty(B, S, 1, device="cuda", dtype=torch.float32)
    ivar = torch.empty_like(mean)
    ws = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)
    pack = {X: x, SC: scale, BI: bias, EP: eps_cpu, Y: y, MEAN: mean, IVAR: ivar}

    g.execute(pack, ws, handle=handle)  # warm: first call JIT-compiles
    torch.cuda.synchronize()

    # The producer: a long stall, then the write that makes `x` current. Both are
    # queued on the stream the handle names, and torch's current stream is NOT it.
    with torch.cuda.stream(side):
        torch.cuda._sleep(_SLEEP_CYCLES)
        x.copy_(fresh)
    assert torch.cuda.current_stream() != side, "the probe needs the producer on a DIFFERENT stream"

    g.execute(pack, ws, handle=handle)
    torch.cuda.synchronize()

    ref = torch.nn.functional.layer_norm(fresh, [D], scale.reshape(-1), bias.reshape(-1), 1e-5)
    err = (y - ref).abs().max().item()
    torch.testing.assert_close(
        y,
        ref,
        check_dtype=False,
        msg=lambda detail: f"execute ran ahead of a producer queued on the handle's stream (max err {err:.5f}); the kernels and output copies are not following ExecutionContext.stream"
        + ": "
        + detail,
    )


@pytest.mark.L0
def test_execute_output_copies_follow_the_handle_stream():
    """The output copies specifically -- a consumer on the handle's stream sees them."""
    B, S, D = 4, 8, 256
    torch.manual_seed(1)
    x = torch.randn(B, S, D, device="cuda", dtype=torch.float32)
    scale = torch.ones(1, 1, D, device="cuda", dtype=torch.float32)
    bias = torch.zeros(1, 1, D, device="cuda", dtype=torch.float32)
    eps_cpu = torch.full((1, 1, 1), 1e-5, dtype=torch.float32)

    side = torch.cuda.Stream()
    handle = cudnn.create_handle()
    cudnn.set_stream(handle=handle, stream=side.cuda_stream)
    g, (X, SC, BI, EP, Y, MEAN, IVAR) = _build(handle, x, scale, bias)

    y = torch.empty_like(x)
    mean = torch.empty(B, S, 1, device="cuda", dtype=torch.float32)
    ivar = torch.empty_like(mean)
    ws = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)
    pack = {X: x, SC: scale, BI: bias, EP: eps_cpu, Y: y, MEAN: mean, IVAR: ivar}
    g.execute(pack, ws, handle=handle)
    torch.cuda.synchronize()

    # Queue the execute, then a consumer on the SAME stream that copies Y away.
    # If execute ran elsewhere, the consumer reads Y before the copies land.
    y.zero_()
    torch.cuda.synchronize()
    with torch.cuda.stream(side):
        torch.cuda._sleep(_SLEEP_CYCLES)
    g.execute(pack, ws, handle=handle)
    with torch.cuda.stream(side):
        seen = y.clone()
    torch.cuda.synchronize()

    ref = torch.nn.functional.layer_norm(x, [D], scale.reshape(-1), bias.reshape(-1), 1e-5)
    err = (seen - ref).abs().max().item()
    torch.testing.assert_close(
        seen, ref, check_dtype=False, msg=lambda detail: f"a consumer on the handle's stream did not see the output copies (max err {err:.5f})" + ": " + detail
    )


@pytest.mark.L0
def test_cuda_graph_replay_recomputes_from_changed_inputs():
    """Capture execute() into a CUDA graph; every replay must honour the new input.

    This is the same defect seen from the other side. With the kernels going to
    the default stream while capture ran on another, the capture recorded only
    the output copies -- so a replay re-copied whatever the (uncaptured) kernel
    had last left in the engine's temporary. The first replay looked right
    because that temporary still held the warm-up result; later ones returned
    stale normalization.
    """
    B, S, D = 4, 8, 256
    torch.manual_seed(2)
    x = torch.randn(B, S, D, device="cuda", dtype=torch.float32)
    scale = torch.ones(1, 1, D, device="cuda", dtype=torch.float32)
    bias = torch.zeros(1, 1, D, device="cuda", dtype=torch.float32)
    eps_cpu = torch.full((1, 1, 1), 1e-5, dtype=torch.float32)

    side = torch.cuda.Stream()
    handle = cudnn.create_handle()
    cudnn.set_stream(handle=handle, stream=side.cuda_stream)
    g, (X, SC, BI, EP, Y, MEAN, IVAR) = _build(handle, x, scale, bias)

    y = torch.empty_like(x)
    mean = torch.empty(B, S, 1, device="cuda", dtype=torch.float32)
    ivar = torch.empty_like(mean)
    ws = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)
    pack = {X: x, SC: scale, BI: bias, EP: eps_cpu, Y: y, MEAN: mean, IVAR: ivar}

    with torch.cuda.stream(side):
        g.execute(pack, ws, handle=handle)  # warm up before capturing
    side.synchronize()

    cuda_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cuda_graph, stream=side):
        g.execute(pack, ws, handle=handle)

    # More than one replay: the first can pass on a stale temporary alone.
    for i in range(3):
        x.copy_(torch.randn_like(x))
        torch.cuda.synchronize()
        ref = torch.nn.functional.layer_norm(x, [D], scale.reshape(-1), bias.reshape(-1), 1e-5)
        cuda_graph.replay()
        torch.cuda.synchronize()
        err = (y - ref).abs().max().item()
        torch.testing.assert_close(y, ref, check_dtype=False, msg=lambda detail: f"replay {i} returned stale normalization (max err {err:.5f})" + ": " + detail)


@pytest.mark.L0
def test_explicitly_requested_default_stream_is_entered():
    """Handle stream 0 is a REQUEST, not "unspecified".

    The two look identical at the call site and behave oppositely. A context
    carrying no stream means torch's current one is already right; a context
    carrying handle 0 means the caller asked for the legacy default stream, and
    whatever side stream happens to be current must be stepped out of. Treating
    the second as the first leaves the side stream in force, so a producer
    queued on the requested default stream goes unwaited for.

    The probe therefore inverts the one above: the producer is on the DEFAULT
    stream and a side stream is current when execute is called.
    """
    B, S, D = 4, 8, 256
    torch.manual_seed(3)
    x = torch.randn(B, S, D, device="cuda", dtype=torch.float32)
    fresh = torch.randn(B, S, D, device="cuda", dtype=torch.float32)
    scale = torch.ones(1, 1, D, device="cuda", dtype=torch.float32)
    bias = torch.zeros(1, 1, D, device="cuda", dtype=torch.float32)
    eps_cpu = torch.full((1, 1, 1), 1e-5, dtype=torch.float32)

    default = torch.cuda.default_stream()
    handle = cudnn.create_handle()
    cudnn.set_stream(handle=handle, stream=default.cuda_stream)
    assert default.cuda_stream == 0, "this probe needs the legacy default stream handle"
    g, (X, SC, BI, EP, Y, MEAN, IVAR) = _build(handle, x, scale, bias)

    y = torch.empty_like(x)
    mean = torch.empty(B, S, 1, device="cuda", dtype=torch.float32)
    ivar = torch.empty_like(mean)
    ws = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)
    pack = {X: x, SC: scale, BI: bias, EP: eps_cpu, Y: y, MEAN: mean, IVAR: ivar}

    g.execute(pack, ws, handle=handle)  # warm
    torch.cuda.synchronize()

    # Producer on the stream the handle names (the default one).
    with torch.cuda.stream(default):
        torch.cuda._sleep(_SLEEP_CYCLES)
        x.copy_(fresh)

    # ...and a DIFFERENT stream current when execute is called.
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        assert torch.cuda.current_stream() != default
        g.execute(pack, ws, handle=handle)
    torch.cuda.synchronize()

    ref = torch.nn.functional.layer_norm(fresh, [D], scale.reshape(-1), bias.reshape(-1), 1e-5)
    err = (y - ref).abs().max().item()
    torch.testing.assert_close(
        y,
        ref,
        check_dtype=False,
        msg=lambda detail: f"execute stayed on the ambient side stream instead of the requested default one (max err {err:.5f}); handle 0 is being read as 'no stream requested'"
        + ": "
        + detail,
    )
