# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SDPA harness reports the plan that actually served a native graph."""

from types import SimpleNamespace

import pytest

import frost_routing
from sdpa.helpers import note_frost_routing

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("selected", [0, 2])
def test_native_routing_reports_selected_plan(selected, capsys, monkeypatch):
    names = ["eng5_k24=2", "eng7_k24=1", "eng9_k24=3"]
    graph = SimpleNamespace(selected_engine=None, _plan_index=selected, get_plan_name_at_index=names.__getitem__)
    monkeypatch.setattr(frost_routing, "COUNTS", {})
    monkeypatch.setattr(frost_routing, "LAST_PLAN", ("previous", {}))
    monkeypatch.setattr(frost_routing, "LAST_ARMS", "previous")

    note_frost_routing(graph, label="mxfp8-bwd")

    assert f"mxfp8-bwd graph: native plan '{names[selected]}' serves this graph" in capsys.readouterr().out
    assert frost_routing.COUNTS == {"native:mxfp8-bwd": 1}
    assert frost_routing.LAST_PLAN == (None, None)
    assert frost_routing.LAST_ARMS is None


def test_python_routing_keeps_its_engine_and_knobs(capsys, monkeypatch):
    engine = SimpleNamespace(name="sdpa_bwd_sm100_mxfp8")
    graph = SimpleNamespace(selected_engine=engine, _selected_plan_config=SimpleNamespace(knobs={"tile_m": 128}))
    monkeypatch.setattr(frost_routing, "COUNTS", {})
    monkeypatch.setattr(frost_routing, "LAST_PLAN", (None, None))
    monkeypatch.setattr(frost_routing, "LAST_ARMS", None)

    note_frost_routing(graph, label="mxfp8-bwd")

    output = capsys.readouterr().out
    assert "python engine 'sdpa_bwd_sm100_mxfp8'" in output
    assert "tile_m" in output
    assert "native plan" not in output
    assert frost_routing.COUNTS == {"frost:sdpa_bwd_sm100_mxfp8": 1}
    assert frost_routing.LAST_PLAN == (engine.name, {"tile_m": 128})


def test_native_routing_reports_a_real_built_graph(cudnn_handle, capsys):
    import cudnn
    from cudnn.engines.manifest import MANIFEST

    graph = cudnn.pygraph(
        handle=cudnn_handle,
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    a = graph.tensor(dim=[1, 16, 16], stride=[256, 16, 1])
    b = graph.tensor(dim=[1, 16, 16], stride=[256, 16, 1])
    graph.matmul(a, b).set_output(True).set_data_type(cudnn.data_type.HALF)
    graph.deselect_engines([name for family in MANIFEST if family.name == "frost_gemm" for name in family.slots])
    graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    assert graph.selected_engine is None
    selected_name = graph.get_plan_name_at_index(graph._plan_index)

    note_frost_routing(graph, label="native-control")

    assert f"native plan '{selected_name}' serves this graph" in capsys.readouterr().out
