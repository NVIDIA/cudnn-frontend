# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public MXFP8 D512 records name their body: output-slice recompute or retained P."""

import pytest
import torch

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell
import test_sdpa_fwd_mxfp8_sm100 as ref

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]
SLICE, RETAINED = "sdpa_fwd_sm100_d512_mxfp8_slice_recompute", "sdpa_fwd_sm100_d512_mxfp8_retained_p"
IDS = {SLICE: 20552, RETAINED: 20553, "sdpa_fwd_prefill_sm100_mxfp8": 20512}
CAUSAL = dict(use_causal_mask=True)
SWA = dict(use_causal_mask=True, diagonal_band_left_bound=128)


def _pinned(monkeypatch, name, knobs):
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    graphs = []

    def pin(g, *args, **kwargs):
        g.create_execution_plan(IDS[name], SdpaFwdKnobs(**knobs).to_public())
        g.select_plan(len(g.plans) - 1)
        graphs.append(g)

    monkeypatch.setattr(ref, "_select_engine", pin)
    return graphs


@pytest.mark.parametrize(
    "name,band,knobs,template",
    [
        (SLICE, {}, {}, "prefill_d512_mxfp8"),
        (SLICE, CAUSAL, {}, "prefill_d512_mxfp8"),
        (SLICE, CAUSAL, {"split_kv": 3}, "prefill_d512_mxfp8"),
        (SLICE, SWA, {"split_kv": 2}, "prefill_d512_mxfp8"),
        (RETAINED, SWA, {}, "prefill_d512_mxfp8_retained_p"),
    ],
)
@pytest.mark.parametrize("capture_first", [False, True])
def test_partial_public_record_names_and_runs_its_body(monkeypatch, name, band, knobs, template, capture_first):
    graphs = _pinned(monkeypatch, name, knobs)
    output, expected, *_ = ref._run(1, 4, 4, 512, "e4m3", torch.float16, scale=512**-0.5, sdpa_kwargs=band, d_qk=512, d_v=512, capture_first=capture_first)
    ref._check(output, expected, torch.float16, "e4m3", d_qk=512)
    (g,) = graphs
    assert g.selected_engine.name == name
    compiled = g._compiled_plans[g._plan_index]._compiled
    assert compiled.kernel_template == template
    assert compiled.implementation_config["split_kv"] == knobs.get("split_kv", 1)


@pytest.mark.parametrize("name,band", [(RETAINED, CAUSAL), (RETAINED, dict(SWA, diagonal_band_left_bound=127)), (SLICE, SWA)])
def test_record_for_the_other_body_is_declined(monkeypatch, name, band):
    _pinned(monkeypatch, name, {})
    with pytest.raises((RuntimeError, ValueError, NotImplementedError), match="retained-P"):
        ref._run(1, 4, 4, 512, "e4m3", torch.float16, scale=512**-0.5, sdpa_kwargs=band, d_qk=512, d_v=512)


def test_retired_legacy_record_is_declined(monkeypatch):
    _pinned(monkeypatch, "sdpa_fwd_prefill_sm100_mxfp8", {})
    with pytest.raises((RuntimeError, ValueError, NotImplementedError), match="owned by"):
        ref._run(1, 4, 4, 512, "e4m3", torch.float16, scale=512**-0.5, sdpa_kwargs=CAUSAL, d_qk=512, d_v=512)
