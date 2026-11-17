# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""SM120 public-S forward conversion and captured binding qualification.

This reuses the independent oracle and byte replay checks. Batch invariance,
goldens, stress and backward public-S qualification remain separate work.
"""

import pytest

from cudnn.engines.manifest import MANIFEST
from cudnn.sdpa.fwd.engines import SdpaFwdKnobs
from frost_test_utils import requires_blackwell_geforce, requires_dsl
import test_sdpa_public_s_scale as shared
from test_sdpa_prepared_fp8 import _case

pytestmark = [pytest.mark.L0, requires_blackwell_geforce, requires_dsl]


def _graph(owner, cga, thd, skv, host=(False, False), source="buffer", pair=(1.0, 1.0)):
    d = 128 if owner == "general" else 512
    name = "sdpa_fwd_sm120_" + ("fp8_general" if owner == "general" else "d512_fp8_paired")
    tile = 128 if owner == "general" else 64
    family = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd")
    knobs = SdpaFwdKnobs(tile_m=tile, tile_n=tile, cga=1, pack_gqa=False, split_kv=1, sched_policy=0)
    result = _case(
        arch="sm120",
        b=1,
        sq=128,
        skv=skv,
        d=d,
        dv=d,
        thd=thd,
        heads=(4, 2),
        record=(family.offered_ids()[name], knobs.to_public()),
        s_scale_host=host,
        s_scale_source=source,
        s_scale_pair=pair,
    )
    g, _, _, bufs, _ = result
    assert g.selected_engine.name == name
    bufs["q"].zero_()
    bufs["k"].zero_()
    bufs["v"].fill_(1)
    for key in ("descale_q", "descale_k", "descale_v", "scale_o"):
        bufs[key].fill_(1)
    return result


@pytest.fixture(autouse=True)
def sm120_graph(monkeypatch):
    monkeypatch.setattr(shared, "_graph", _graph)


@pytest.mark.parametrize("owner", ["general", "paired"])
@pytest.mark.parametrize("thd", [False, True], ids=["dense", "thd"])
@pytest.mark.parametrize("skv", [128, 8192], ids=["K128", "K8192-refill"])
@pytest.mark.parametrize("host", [(False, False), (True, True), (True, False), (False, True)], ids=["device", "host", "host-device", "device-host"])
def test_sm120_public_s_conversion(owner, thd, skv, host):
    shared.test_public_s_conversion_rebind_and_replay(owner, 1, thd, skv, host)


@pytest.mark.parametrize("owner", ["general", "paired"])
@pytest.mark.parametrize("thd", [False, True], ids=["dense", "thd"])
@pytest.mark.parametrize("source,pair", [("omitted", (256.0, 1 / 256)), ("embedded", (1.0, 1.0)), ("embedded", (0.0, 1.0))])
def test_sm120_public_s_default(owner, thd, source, pair):
    shared.test_graph_owned_s_scales(owner, 1, thd, 128, source, pair)


@pytest.mark.parametrize("owner", ["general", "paired"])
@pytest.mark.parametrize("thd", [False, True], ids=["dense", "thd"])
@pytest.mark.parametrize("source", ["omitted", "embedded"])
@pytest.mark.parametrize("ordered", [False, True], ids=["mapping", "ordered"])
def test_sm120_public_s_runtime_override(owner, thd, source, ordered):
    shared.test_graph_owned_runtime_s_override_and_capture(owner, 1, thd, source, ordered)


@pytest.mark.parametrize("owner", ["general", "paired"])
@pytest.mark.parametrize("thd", [False, True], ids=["dense", "thd"])
def test_sm120_public_s_retained_device_values(owner, thd):
    from test_sdpa_prepared_fp8 import _cuda_graph
    import torch

    g, vp, ws, bufs, _ = _graph(owner, 1, thd, 8192)
    g.execute(vp, ws)
    with _cuda_graph() as replay:
        with torch.cuda.graph(replay):
            g.execute(vp, ws)
        for s, ds in ((1, 1), (256, 1 / 256), (2, 1), (1, 2), (0, 1)):
            bufs["scale_s"].fill_(s)
            bufs["descale_s"].fill_(ds)
            shared._poison(bufs)
            replay.replay()
            shared._check(bufs, 8192, 1, s, ds)


@pytest.mark.parametrize("owner", ["general", "paired"])
@pytest.mark.parametrize("thd", [False, True], ids=["dense", "thd"])
@pytest.mark.parametrize("source,pair", [("omitted", (256.0, 1 / 256)), ("embedded", (2.0, 1.0))])
def test_sm120_public_s_graph_and_adapter_defaults(owner, thd, source, pair):
    shared.check_graph_and_adapter_defaults(owner, 1, thd, source, pair)
