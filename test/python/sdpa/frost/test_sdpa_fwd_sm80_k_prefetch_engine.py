# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM80 K-prefetch records name their body and replay it without selection; retired 20508 records decline."""

import cudnn
import pytest
import torch

from frost_test_utils import _SM, requires_dsl
from test_sdpa_fwd_sm120_d512_paired_engine import _check, _graph

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM != 80, reason="the SM80 engines serve A100 (cc 8.0) only")]

_OWNER, _OWNER_ID, _LEGACY_ID = "sdpa_fwd_sm80_f16_k_prefetch", 20546, 20508


def _record(engine_id, split=1, sched=0):
    return engine_id, {cudnn.knob_type.SCHED_POLICY: sched, cudnn.knob_type.PACK_GQA: 0, cudnn.knob_type.SPLIT_KV: split}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dims,split,sq", [((64, 64), 1, 97), ((128, 128), 3, 97), ((192, 128), 1, 97), ((72, 40), 1, 5), ((128, 128), 1, 3)])
def test_public_record_names_and_replays_its_body(monkeypatch, dtype, dims, split, sq):
    from cudnn.sdpa.fwd import heuristics

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit public replay consulted candidate selection")

    monkeypatch.setattr(heuristics, "recommend", forbidden)
    g, vp, ws, bufs, _ = _graph(*dims, dtype, record=_record(_OWNER_ID, split), sq=sq)
    assert g.selected_engine.name == _OWNER
    compiled = g._compiled_plans[g._plan_index]._compiled
    assert compiled.kernel_template == "prefill_f16"
    metadata = compiled.implementation_config
    from cudnn.sdpa.fwd import config_sm80

    flavor = config_sm80.FLAVOR_CFGS[config_sm80.pick_flavor(*dims)]  # the body compiles the covering flavor's widths
    assert (metadata["d_qk"], metadata["d_v"], metadata["split_kv"]) == (flavor.D_QK, flavor.D_V, split)
    g.execute(vp, ws)
    _check(bufs)
    record = g.get_engine_and_knobs_at_index(g._plan_index)
    rebuilt, rvp, rws, rbufs, _ = _graph(*dims, dtype, record=record, sq=sq)
    assert rebuilt._compiled_plans[rebuilt._plan_index]._compiled.implementation_config == metadata
    rebuilt.execute(rvp, rws)
    _check(rbufs)


@pytest.mark.parametrize("dims", [(64, 64), (128, 128)])
def test_retired_legacy_record_is_declined(dims):
    with pytest.raises((ValueError, RuntimeError, NotImplementedError), match="owned by"):
        _graph(*dims, torch.bfloat16, record=_record(_LEGACY_ID))


@pytest.mark.parametrize("dims", [(256, 256), (512, 512)])
def test_d256_and_wide_keep_the_legacy_row(dims):
    g, vp, ws, bufs, _ = _graph(*dims, torch.bfloat16, record=_record(_LEGACY_ID))
    assert g.selected_engine.name == "sdpa_fwd_prefill_sm80"
    assert g._compiled_plans[g._plan_index]._compiled.implementation_config is None
    g.execute(vp, ws)
    _check(bufs)
