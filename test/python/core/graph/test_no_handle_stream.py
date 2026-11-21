# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A graph executed without a handle runs on the caller's current stream, whichever engine serves it.

The graph resolves that stream once (``pygraph._resolve_stream``) and hands it to the python
engines in ``ExecutionContext.stream``; the backend gets the per-thread default handle re-streamed
to it. The detectors here are a producer queued on a side stream behind a long sleep, and a CUDA
graph capture: a launch on stream 0 inside a global-mode capture is a capture error, and outside
one it reads the input before the producer wrote it.
"""

import pytest

torch = pytest.importorskip("torch")

import cudnn
from cudnn._pygraph import pygraph
from cudnn.engines import BaseEngine, CompiledPlan
from cudnn.engines.engine_ids import PYTHON_ENGINE_ID_BASE

pytestmark = pytest.mark.L0

_FAKE = PYTHON_ENGINE_ID_BASE + 9_000
_SLEEP_CYCLES = 100_000_000  # ~50 ms: far longer than the relu


def _require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")


def _offer(monkeypatch, engine):
    from cudnn.engines import manifest

    family = manifest.EngineFamily(_FAKE, "no_handle_stream", __name__, "unused_factory", slots={engine.name: manifest.EngineSlot(engine.engine_id - _FAKE)})
    monkeypatch.setattr(manifest, "MANIFEST", (family,))
    monkeypatch.setattr(manifest, "_ANCHOR_NODE_TO_FAMILY", {"MATMUL": "no_handle_stream"})
    monkeypatch.setattr(manifest, "instantiate", lambda fam, ids: [engine] if engine.name in ids else [])


def _recording_engine(seen):
    class Plan(CompiledPlan):
        def execute(self, graph, tensor_data, ctx):
            seen.append(("execute", ctx.stream))

    class Engine(BaseEngine):
        name = "no_handle_stream_recorder"
        engine_id = _FAKE + 1

        def build_plan(self, graph, plan, ctx=None):
            seen.append(("build", ctx.stream))
            return Plan()

    return Engine()


def test_python_plan_without_handle_gets_the_current_stream(monkeypatch):
    """No handle: build and execute both see torch's current stream as a concrete handle, never None."""
    _require_gpu()
    seen = []
    _offer(monkeypatch, _recording_engine(seen))
    graph = pygraph()
    out = graph.matmul(torch.randn(2, 2), torch.randn(2, 2))
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        graph.execute({out: torch.empty(2, 2)})
    graph.execute({out: torch.empty(2, 2)})
    assert seen == [("build", side.cuda_stream), ("execute", side.cuda_stream), ("execute", torch.cuda.current_stream().cuda_stream)]


def test_python_plan_handle_stream_still_wins(monkeypatch):
    """A supplied handle keeps classic set_stream semantics, whatever torch's current stream is."""
    _require_gpu()
    seen = []
    _offer(monkeypatch, _recording_engine(seen))
    handle = cudnn.create_handle()
    try:
        pinned = torch.cuda.Stream()
        cudnn.set_stream(handle, pinned.cuda_stream)
        graph = pygraph()
        out = graph.matmul(torch.randn(2, 2), torch.randn(2, 2))
        with torch.cuda.stream(torch.cuda.Stream()):
            graph.execute({out: torch.empty(2, 2)}, handle=handle)
        assert seen == [("build", pinned.cuda_stream), ("execute", pinned.cuda_stream)]
    finally:
        cudnn.destroy_handle(handle)


def _relu_graph():
    graph = cudnn.pygraph(io_data_type=cudnn.data_type.FLOAT, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    x = graph.tensor(dim=[1, 32, 32], stride=[1024, 32, 1], data_type=cudnn.data_type.FLOAT)
    y = graph.relu(x).set_output(True).set_data_type(cudnn.data_type.FLOAT)
    graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    assert graph.selected_engine is None
    return graph, x, y


def test_backend_without_handle_is_ordered_after_a_side_stream_producer():
    _require_gpu()
    graph, tx, ty = _relu_graph()
    x = torch.randn(1, 32, 32, device="cuda")
    fresh = torch.randn_like(x)
    y = torch.zeros_like(x)
    ws = torch.empty(max(graph.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    graph.execute({tx: x, ty: y}, ws)
    torch.cuda.synchronize()

    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        torch.cuda._sleep(_SLEEP_CYCLES)
        x.copy_(fresh)
        graph.execute({tx: x, ty: y}, ws)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, fresh.relu(), msg=lambda d: "the backend ran without waiting for the side-stream producer: " + d)


def test_backend_without_handle_is_capturable():
    _require_gpu()
    graph, tx, ty = _relu_graph()
    x = torch.randn(1, 32, 32, device="cuda")
    y = torch.zeros_like(x)
    ws = torch.empty(max(graph.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    graph.execute({tx: x, ty: y}, ws)
    torch.cuda.synchronize()

    cg = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(cg):
            graph.execute({tx: x, ty: y}, ws)
        x.copy_(torch.randn_like(x))
        y.zero_()
        cg.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(y, x.relu())
    finally:
        cg.reset()
    # Outside the capture the default handle follows the current stream again.
    x.copy_(torch.randn_like(x))
    graph.execute({tx: x, ty: y}, ws)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x.relu())


def test_engine_launch_refuses_a_missing_stream():
    """A graph-API engine never falls back to stream 0; a None that bypassed the graph is refused."""
    from cudnn.engines.base import require_stream

    assert require_stream(0, "engine") == 0
    with pytest.raises(ValueError, match="engine: no launch stream"):
        require_stream(None, "engine")
