# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public SM107 MXFP8 records name their body: one engine per native head shape."""

import pytest
import torch

from frost_test_utils import requires_dsl, requires_rubin
import test_sdpa_fwd_mxfp8_sm100 as ref

pytestmark = [pytest.mark.L0, requires_dsl, requires_rubin]
OWNERS = {
    (128, 128): ("sdpa_fwd_sm107_d128_mxfp8_multi_q", 20565, "prefill_d128_mxfp8"),
    (192, 128): ("sdpa_fwd_sm107_d192_mxfp8_multi_q", 20566, "prefill_d192_d128_mxfp8"),
    (256, 256): ("sdpa_fwd_sm107_d256_mxfp8_single_q", 20567, "prefill_d256_mxfp8"),
    (512, 512): ("sdpa_fwd_sm107_d512_mxfp8_role_split", 20568, "prefill_d512_mxfp8"),
}
LEGACY_ID = 20516
CAUSAL = dict(use_causal_mask=True)


def _pinned(monkeypatch, engine_id, knobs):
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    graphs = []

    def pin(g, *args, **kwargs):
        g.create_execution_plan(engine_id, SdpaFwdKnobs(**knobs).to_public())
        g.select_plan(len(g.plans) - 1)
        graphs.append(g)

    monkeypatch.setattr(ref, "_select_engine", pin)
    return graphs


def _run(shape, band, **kwargs):
    d_qk, d_v = shape
    return ref._run(1, 4, 4, 512, "e4m3", torch.float16, scale=d_qk**-0.5, sdpa_kwargs=band, d_qk=d_qk, d_v=d_v, **kwargs)


@pytest.mark.parametrize(
    "shape,band,knobs",
    [
        ((128, 128), {}, {}),
        ((128, 128), {}, {"cga": 1}),
        ((128, 128), CAUSAL, {"split_kv": 2}),
        ((192, 128), CAUSAL, {}),
        ((192, 128), CAUSAL, {"cga": 1}),
        ((256, 256), {}, {"cga": 1}),
        ((256, 256), CAUSAL, {"cga": 2}),
        ((512, 512), CAUSAL, {}),
    ],
)
@pytest.mark.parametrize("capture_first", [False, True])
def test_partial_public_record_names_and_runs_its_body(monkeypatch, shape, band, knobs, capture_first):
    name, engine_id, template = OWNERS[shape]
    graphs = _pinned(monkeypatch, engine_id, knobs)
    output, expected, *_ = _run(shape, band, capture_first=capture_first)
    ref._check(output, expected, torch.float16, "e4m3", d_qk=shape[0])
    (g,) = graphs
    assert g.selected_engine.name == name
    compiled = g._compiled_plans[g._plan_index]._compiled
    assert compiled.kernel_template == template
    assert compiled.implementation_config["split_kv"] == knobs.get("split_kv", 1)
    if "cga" in knobs:
        assert compiled.implementation_config["cta_group"] == knobs["cga"]


@pytest.mark.parametrize("record,graph", [((128, 128), (256, 256)), ((256, 256), (128, 128)), ((512, 512), (192, 128))])
def test_record_for_another_shape_is_declined(monkeypatch, record, graph):
    _pinned(monkeypatch, OWNERS[record][1], {})
    with pytest.raises((RuntimeError, ValueError, NotImplementedError), match="serves exact D_QK/D_V"):
        _run(graph, CAUSAL)


@pytest.mark.parametrize("shape", sorted(OWNERS))
def test_retired_legacy_record_is_declined(monkeypatch, shape):
    _pinned(monkeypatch, LEGACY_ID, {})
    with pytest.raises((RuntimeError, ValueError, NotImplementedError), match="owned by"):
        _run(shape, CAUSAL)
