# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public small/mid-width MXFP8 records fix the body across rebuild and retained capture."""

import pytest
import torch

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell
import test_sdpa_prepared_mxfp8 as shared

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("output", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    "dim,cga,thd,split",
    [
        (64, 1, False, 1),
        (128, 1, False, 1),
        (128, 2, False, 1),
        (128, 2, True, 1),
        (128, 2, False, 4),
        (192, 1, False, 1),
        (192, 2, False, 1),
        (192, 1, True, 1),
        (192, 2, True, 1),
        (192, 2, False, 4),
    ],
)
def test_public_record_rebuild_rebind_and_retained_capture(monkeypatch, dtype, output, dim, cga, thd, split):
    from cudnn.sdpa.fwd import heuristics
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    owner_id = 20550 if dim <= 128 else 20551
    implementation = "d128_mxfp8_multi_q" if dim <= 128 else "d192_mxfp8_multi_q"
    record = (owner_id, SdpaFwdKnobs(tile_m=128, tile_n=128, cga=cga, sched_policy=0, pack_gqa=False, split_kv=split).to_public())

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit public replay consulted candidate selection")

    monkeypatch.setattr(heuristics, "recommend", forbidden)

    def build(saved):
        result = shared._case(d=dim, dv=min(dim, 128), thd=thd, sq=17, skv=256, dtype=dtype, output_dtype=output, split_kv=split, record=saved)
        graph = result[0]
        assert graph.get_engine_and_knobs_at_index(graph._plan_index) == saved
        compiled = graph._compiled_plans[graph._plan_index]._compiled
        assert compiled.kernel_template == ("prefill_d128_mxfp8" if dim <= 128 else "prefill_d192_d128_mxfp8")
        from cudnn.sdpa.fwd.kernels.sm100.d128_mxfp8_multi_q import config as small_config
        from cudnn.sdpa.fwd.kernels.sm100.d192_mxfp8_multi_q import config as mid_config

        cfg = compiled.kernel_module.CFG
        assert type(cfg) is getattr(small_config if dim <= 128 else mid_config, f"CfgD{dim}")
        assert (cfg.TILE_K, cfg.TILE_O, cfg.CGA_M, cfg.CTA_MMA) == (dim, min(dim, 128), cga, cga)
        assert compiled.kernel_module.__file__.endswith(f"sm100/{implementation}/kernel.py")
        metadata = compiled.implementation_config
        assert graph.selected_engine.name == f"sdpa_fwd_sm100_{implementation}"
        assert (metadata["query_tile"], metadata["cluster_m"], metadata["cta_group"], metadata["split_kv"]) == (128, cga, cga, split)
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
