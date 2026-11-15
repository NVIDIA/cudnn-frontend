# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-D256 SM120 records name their body by TILE_M and replay it without selection."""

import cudnn
import pytest
import torch

from frost_test_utils import requires_blackwell_geforce, requires_dsl
from test_sdpa_fwd_sm120_d512_paired_engine import _check, _graph

pytestmark = [pytest.mark.L0, requires_dsl, requires_blackwell_geforce]

_OWNERS = {64: (20537, "sdpa_fwd_sm120_d256_all_compute", "prefill_d256_f16"), 128: (20537, "sdpa_fwd_sm120_d256_all_compute", "prefill_d256_f16")}
_OWNERS.update({16: (20538, "sdpa_fwd_sm120_decode_key_group", "decode_f16"), 32: (20538, "sdpa_fwd_sm120_decode_key_group", "decode_f16")})


def _record(tile_m, split):
    engine_id = _OWNERS[tile_m][0]
    return engine_id, {
        cudnn.knob_type.TILE_M: tile_m,
        cudnn.knob_type.TILE_CGA_M: 1,
        cudnn.knob_type.SCHED_POLICY: 0,
        cudnn.knob_type.PACK_GQA: 0,
        cudnn.knob_type.SPLIT_KV: split,
    }


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tile_m,split,sq", [(128, 1, 97), (64, 3, 97), (16, 1, 3), (32, 4, 5)])
def test_public_record_names_and_replays_its_body(monkeypatch, dtype, tile_m, split, sq):
    from cudnn.sdpa.fwd import heuristics

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit public replay consulted candidate selection")

    monkeypatch.setattr(heuristics, "recommend", forbidden)
    engine_id, name, template = _OWNERS[tile_m]
    g, vp, ws, bufs, _ = _graph(256, 256, dtype, record=_record(tile_m, split), sq=sq)
    assert g.selected_engine.name == name
    compiled = g._compiled_plans[g._plan_index]._compiled
    assert compiled.kernel_template == template
    metadata = compiled.implementation_config
    assert (metadata["query_tile"], metadata["split_kv"]) == (tile_m, split)
    g.execute(vp, ws)
    _check(bufs)
    record = g.get_engine_and_knobs_at_index(g._plan_index)
    rebuilt, rvp, rws, rbufs, _ = _graph(256, 256, dtype, record=record, sq=sq)
    assert rebuilt._compiled_plans[rebuilt._plan_index]._compiled.implementation_config == metadata
    rebuilt.execute(rvp, rws)
    _check(rbufs)


@pytest.mark.parametrize("tile_m,sq", [(128, 97), (16, 3)])
def test_retired_legacy_record_is_declined(tile_m, sq):
    _, knobs = _record(tile_m, 1)
    with pytest.raises((ValueError, RuntimeError, NotImplementedError), match="owned by"):
        _graph(256, 256, torch.bfloat16, record=(20505, knobs), sq=sq)
