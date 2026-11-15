# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-tile SM120 half records name their body by TILE_M and head tile and replay it without selection."""

import cudnn
import pytest
import torch

from frost_test_utils import requires_blackwell_geforce, requires_dsl
from test_sdpa_fwd_sm120_d512_paired_engine import _check, _graph

pytestmark = [pytest.mark.L0, requires_dsl, requires_blackwell_geforce]

_ALL_COMPUTE = (20537, "sdpa_fwd_sm120_d256_all_compute", "prefill_d256_f16")
_GENERAL = (20544, "sdpa_fwd_sm120_f16_general", "prefill_f16")
_KEY_GROUP = (20538, "sdpa_fwd_sm120_decode_key_group", "decode_f16")


def _owner(d_qk, d_v, tile_m):
    if tile_m in (16, 32):
        return _KEY_GROUP
    return _ALL_COMPUTE if (d_qk, d_v) == (256, 256) else _GENERAL


def _record(engine_id, tile_m, split):
    return engine_id, {
        cudnn.knob_type.TILE_M: tile_m,
        cudnn.knob_type.TILE_CGA_M: 1,
        cudnn.knob_type.SCHED_POLICY: 0,
        cudnn.knob_type.PACK_GQA: 0,
        cudnn.knob_type.SPLIT_KV: split,
    }


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "d_qk,d_v,tile_m,split,sq",
    [
        (256, 256, 128, 1, 97),
        (256, 256, 64, 3, 97),
        (256, 256, 16, 1, 3),
        (256, 256, 32, 4, 5),
        (128, 128, 128, 1, 97),
        (128, 128, 64, 2, 97),
        (64, 64, 16, 1, 3),
        (128, 128, 32, 3, 5),
        (96, 160, 64, 1, 97),
    ],
)
def test_public_record_names_and_replays_its_body(monkeypatch, dtype, d_qk, d_v, tile_m, split, sq):
    from cudnn.sdpa.fwd import heuristics

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit public replay consulted candidate selection")

    monkeypatch.setattr(heuristics, "recommend", forbidden)
    engine_id, name, template = _owner(d_qk, d_v, tile_m)
    g, vp, ws, bufs, _ = _graph(d_qk, d_v, dtype, record=_record(engine_id, tile_m, split), sq=sq)
    assert g.selected_engine.name == name
    compiled = g._compiled_plans[g._plan_index]._compiled
    assert compiled.kernel_template == template
    metadata = compiled.implementation_config
    assert (metadata["query_tile"], metadata["split_kv"]) == (tile_m, split)
    g.execute(vp, ws)
    _check(bufs)
    record = g.get_engine_and_knobs_at_index(g._plan_index)
    rebuilt, rvp, rws, rbufs, _ = _graph(d_qk, d_v, dtype, record=record, sq=sq)
    assert rebuilt._compiled_plans[rebuilt._plan_index]._compiled.implementation_config == metadata
    rebuilt.execute(rvp, rws)
    _check(rbufs)


@pytest.mark.parametrize("d,tile_m,sq", [(256, 128, 97), (256, 16, 3), (128, 64, 97), (64, 32, 5)])
def test_retired_legacy_record_is_declined(d, tile_m, sq):
    _, knobs = _record(20505, tile_m, 1)
    with pytest.raises((ValueError, RuntimeError, NotImplementedError), match="owned by"):
        _graph(d, d, torch.bfloat16, record=(20505, knobs), sq=sq)


def test_envelope_keeps_the_legacy_owner():
    g, vp, ws, bufs, _ = _graph(120, 120, torch.bfloat16, record=_record(20505, 64, 1), sq=97)
    assert g.selected_engine.name == "sdpa_fwd_prefill_sm120"
    g.execute(vp, ws)
    _check(bufs)
