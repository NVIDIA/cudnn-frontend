# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planning and python-plan builds follow the graph handle's GPU, not the current device.

The graph scopes ``frost.device.build_device`` to the handle's ordinal around planning and
around every python plan build, so the analyzers' device facts and every engine's
device-derived constants describe the GPU the plan will run on. Each case makes another GPU,
of a different compute capability, current.
"""

import pytest

torch = pytest.importorskip("torch")

import cudnn
from cudnn._pygraph import pygraph
from cudnn.engines import BaseEngine, CompiledPlan
from cudnn.engines.engine_ids import PYTHON_ENGINE_ID_BASE
from cudnn.frost import device as D

pytestmark = pytest.mark.L0

_FAKE = PYTHON_ENGINE_ID_BASE + 9_000


def _props(ordinal):
    p = torch.cuda.get_device_properties(ordinal)
    return (p.major, p.minor), p.multi_processor_count, p.shared_memory_per_multiprocessor


def _foreign_pair(differ=lambda a, b: a[0] != b[0]):
    """(current, target): two GPUs whose properties differ as ``differ`` asks."""
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    for current in range(count):
        for target in range(count):
            if current != target and differ(_props(current), _props(target)):
                return current, target
    pytest.skip("requires two visible GPUs of different compute capability")


def _handle_on(ordinal):
    with torch.cuda.device(ordinal):
        return cudnn.create_handle()


def _offer(monkeypatch, engine):
    from cudnn.engines import manifest

    family = manifest.EngineFamily(_FAKE, "build_device_scope", __name__, "unused_factory", slots={engine.name: manifest.EngineSlot(engine.engine_id - _FAKE)})
    monkeypatch.setattr(manifest, "MANIFEST", (family,))
    monkeypatch.setattr(manifest, "_ANCHOR_NODE_TO_FAMILY", {"MATMUL": "build_device_scope"})
    monkeypatch.setattr(manifest, "instantiate", lambda fam, ids: [engine] if engine.name in ids else [])


def test_python_plan_builds_on_the_graph_handle_device(monkeypatch):
    current, target = _foreign_pair()
    seen = {}

    class Plan(CompiledPlan):
        def execute(self, graph, tensor_data, ctx):
            pass

    class Engine(BaseEngine):
        name = "build_device_recorder"
        engine_id = _FAKE + 2

        def check_support(self, graph):
            seen.setdefault("check_support", D.current_device())

        def build_plan(self, graph, plan, ctx=None):
            seen["build"] = D.current_device()
            seen["arch"] = D.compute_capability(D.current_device())
            return Plan()

    _offer(monkeypatch, Engine())
    handle = _handle_on(target)
    try:
        with torch.cuda.device(current):
            graph = pygraph(handle=handle)
            graph.matmul(torch.randn(2, 2), torch.randn(2, 2))
            graph.create_execution_plans()
            graph.check_support()
            graph.build_plans()
            assert torch.cuda.current_device() == current
        assert seen == {"check_support": target, "build": target, "arch": _props(target)[0]}
    finally:
        cudnn.destroy_handle(handle)


def test_sdpa_and_norm_device_facts_follow_the_build_scope():
    from cudnn.norm import graph_analyzer as norm_ga
    from cudnn.sdpa import graph_analyzer as sdpa_ga

    current, target = _foreign_pair()
    with torch.cuda.device(current):
        for ordinal in (current, target):
            with D.build_device(ordinal):
                cc, sms, _ = _props(ordinal)
                assert sdpa_ga._device_cc() == cc
                assert sdpa_ga._device_sm_count() == sms
                assert norm_ga._device_cc() == cc


def test_norm_shared_memory_and_tmem_follow_the_build_scope():
    pytest.importorskip("cutlass")
    from cudnn.norm import utils

    current, target = _foreign_pair(lambda a, b: a[2] != b[2])
    with torch.cuda.device(current):
        for ordinal in (current, target, current):
            with D.build_device(ordinal):
                cc, _, per_sm = _props(ordinal)
                assert utils.smem_per_sm() == per_sm
                assert utils.has_tmem() is (cc[0] in (10, 11))


def test_sdpa_graph_facts_describe_the_graph_handle_device(monkeypatch):
    if cudnn.backend_version() < 92400:
        pytest.skip("the decode SDPA graph requires cuDNN 9.24+")
    current, target = _foreign_pair(lambda a, b: a[0] != b[0] and b[0] >= (8, 0))
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    handle = _handle_on(target)
    try:
        with torch.cuda.device(current):
            graph = cudnn.pygraph(
                handle=handle,
                io_data_type=cudnn.data_type.BFLOAT16,
                intermediate_data_type=cudnn.data_type.FLOAT,
                compute_data_type=cudnn.data_type.FLOAT,
            )
            q = torch.randn(2, 4, 1, 128, dtype=torch.bfloat16)
            k = torch.randn(2, 4, 64, 128, dtype=torch.bfloat16)
            tq, tk, tv = [graph.tensor_like(x) for x in (q, k, k)]
            to, _ = graph.sdpa(q=tq, k=tk, v=tv, generate_stats=False, attn_scale=128**-0.5)
            to.set_output(True).set_dim(q.shape).set_stride(q.stride())
            graph.validate()
            graph.build_operation_graph()
            graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        facts = [f for f in graph._facts.values() if hasattr(f, "device_cc")]
        if not facts:
            pytest.skip("no python SDPA family analyzed this graph")
        cc, sms, _ = _props(target)
        assert [(f.device_cc, f.device_sm_count) for f in facts] == [(cc, sms)] * len(facts)
    finally:
        cudnn.destroy_handle(handle)
