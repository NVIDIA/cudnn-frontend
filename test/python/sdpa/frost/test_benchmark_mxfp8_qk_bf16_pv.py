# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Regression tests for the hybrid MXFP8 benchmark harness."""

import importlib.util
from pathlib import Path

import pytest


def _load_benchmark_module():
    path = Path(__file__).parents[4] / "benchmark/attention_training/benchmark_mxfp8_qk_bf16_pv.py"
    spec = importlib.util.spec_from_file_location("benchmark_mxfp8_qk_bf16_pv", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.L0
def test_cuda_graph_capture_warms_up_on_a_side_stream(monkeypatch):
    """The first kernel initialization must finish before graph capture."""
    benchmark = _load_benchmark_module()
    events = []

    class FakeStream:
        def __init__(self, name="warmup"):
            self.name = name

        def wait_stream(self, other):
            events.append(f"{self.name}.wait_stream({other.name})")

    class FakeContext:
        def __init__(self, label):
            self.label = label

        def __enter__(self):
            events.append(f"enter:{self.label}")

        def __exit__(self, *_exc):
            events.append(f"exit:{self.label}")

    class FakeGraph:
        def replay(self):
            events.append("replay")

    current = FakeStream("current")
    monkeypatch.setattr(benchmark.torch.cuda, "current_stream", lambda: current)
    monkeypatch.setattr(benchmark.torch.cuda, "Stream", FakeStream)
    monkeypatch.setattr(benchmark.torch.cuda, "stream", lambda stream: FakeContext(stream.name))
    monkeypatch.setattr(benchmark.torch.cuda, "synchronize", lambda: events.append("synchronize"))
    monkeypatch.setattr(benchmark.torch.cuda, "CUDAGraph", FakeGraph)
    monkeypatch.setattr(benchmark.torch.cuda, "graph", lambda _graph: FakeContext("capture"))

    replay = benchmark._capture_cuda_graph(lambda: events.append("launch"))
    replay()

    assert events == [
        "warmup.wait_stream(current)",
        "enter:warmup",
        "launch",
        "exit:warmup",
        "current.wait_stream(warmup)",
        "synchronize",
        "enter:capture",
        "launch",
        "exit:capture",
        "synchronize",
        "replay",
    ]
