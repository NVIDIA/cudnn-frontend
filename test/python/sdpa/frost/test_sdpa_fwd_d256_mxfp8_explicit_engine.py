# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public MXFP8 D256 records fix the body across rebuild and retained capture."""

import json

import pytest
import torch

import cudnn
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell
import test_sdpa_prepared_mxfp8 as shared

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("output", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("thd,split", [(False, 1), (False, 4), (True, 1)])
def test_public_record_rebuild_rebind_and_retained_capture(monkeypatch, dtype, output, thd, split):
    from cudnn.sdpa.fwd import heuristics
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    record = (20545, SdpaFwdKnobs(tile_m=128, tile_n=128, cga=1, sched_policy=0, pack_gqa=False, split_kv=split).to_public())

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit public replay consulted candidate selection")

    monkeypatch.setattr(heuristics, "recommend", forbidden)

    def build(saved):
        result = shared._case(d=256, thd=thd, sq=17, skv=256, dtype=dtype, output_dtype=output, split_kv=split, record=saved)
        graph = result[0]
        assert graph.get_engine_and_knobs_at_index(graph._plan_index) == saved
        compiled = graph._compiled_plans[graph._plan_index]._compiled
        assert compiled.kernel_template == "prefill_d256_mxfp8"
        from cudnn.sdpa.fwd.kernels.sm100.d256_mxfp8_single_q.config import CfgD256

        assert type(compiled.kernel_module.CFG) is CfgD256
        assert compiled.kernel_module.__file__.endswith("sm100/d256_mxfp8_single_q/kernel.py")
        metadata = compiled.implementation_config
        assert graph.selected_engine.name == "sdpa_fwd_sm100_d256_mxfp8_single_q"
        assert (metadata["query_tile"], metadata["cluster_m"], metadata["cta_group"], metadata["split_kv"]) == (128, 1, 1, split)
        return result

    g, vp, ws, bufs, tensors = build(record)
    g.execute(vp, ws)
    shared._check(bufs, thd=thd, sq=17, skv=256)

    with shared._cuda_graph() as capture:
        with torch.cuda.graph(capture):
            previous = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                g.execute(vp, ws)
            finally:
                torch.cuda.set_sync_debug_mode(previous)
        rebuilt, rebound, scratch, other, _ = build(g.get_engine_and_knobs_at_index(g._plan_index))
        rebuilt.execute(rebound, scratch)
        shared._check(other, thd=thd, sq=17, skv=256)
        shared._change_scales(bufs)
        for name in ("o", "lse", "amax_o"):
            bufs[name].fill_(float("nan"))
        capture.replay()
        shared._check(bufs, thd=thd, sq=17, skv=256)

    # Eager execution rebinds caller storage and scales independently of the
    # retained executable's capture-time pointers.
    for name in ("q", "k", "v", "o", "lse", "amax_o", "sf_q", "sf_k", "sf_v"):
        bufs[name] = bufs[name].clone()
        vp[tensors[name]] = bufs[name]
    g.execute(vp, torch.empty_like(ws))
    shared._check(bufs, thd=thd, sq=17, skv=256)


def test_retired_mxfp8_recipe_is_rejected_without_silent_substitution():
    import cudnn._plan_serdes as ps

    graph, _, _, _, _ = shared._case(d=256, sq=17, skv=256)
    blob = graph.serialize(as_bytes=True, include_kernels=False)
    start = len(ps.MAGIC) + ps._LEN.size
    (length,) = ps._LEN.unpack_from(blob, len(ps.MAGIC))
    header = json.loads(blob[start : start + length])
    header["plan"].update(engine_id=20512, engine_name="sdpa_fwd_prefill_sm100_mxfp8", knobs=[])
    encoded = json.dumps(header).encode()
    target = cudnn.pygraph()
    with pytest.raises((ValueError, RuntimeError), match="20512|owned by|not supported|No execution plans"):
        target.deserialize(ps.MAGIC + ps._LEN.pack(len(encoded)) + encoded)
    assert not target._nodes and not target._planning_done
    # The same empty destination remains reusable for an eligible current recipe.
    target.deserialize(blob)
    assert target.get_engine_and_knobs_at_index(target._plan_index)[0] == 20545
