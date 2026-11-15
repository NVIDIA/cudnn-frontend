# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public FP8 D192/D256 records fix the body across rebuild and retained capture."""

import json

import pytest
import torch

import cudnn
from frost_test_utils import requires_dsl, requires_rubin
import test_sdpa_prepared_fp8 as shared

pytestmark = [pytest.mark.L0, requires_dsl, requires_rubin]


@pytest.mark.parametrize("d", [192, 256])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("output", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("split", [1, 4])
def test_public_record_rebuild_rebind_and_retained_capture(monkeypatch, d, dtype, output, split):
    from cudnn.sdpa.fwd import heuristics
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    owner = "d192_fp8_multi_q" if d == 192 else "d256_fp8_single_q"
    engine_id = 20561 if d == 192 else 20562
    cga = 2 if d == 192 else 1
    record = (engine_id, SdpaFwdKnobs(tile_m=128, tile_n=128, cga=cga, sched_policy=0, pack_gqa=True, split_kv=split).to_public())

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit public replay consulted candidate selection")

    monkeypatch.setattr(heuristics, "recommend", forbidden)

    def build(saved):
        result = shared._case(arch="sm107", d=d, dv=128 if d == 192 else 256, sq=17, skv=256, dtype=dtype, output_dtype=output, split_kv=split, record=saved)
        graph = result[0]
        assert graph.get_engine_and_knobs_at_index(graph._plan_index) == saved
        compiled = graph._compiled_plans[graph._plan_index]._compiled
        assert compiled.kernel_template == ("prefill_d192_d128_fp8" if d == 192 else "prefill_d256_fp8")
        assert compiled.kernel_module.__file__.endswith(f"sm107/{owner}/kernel.py")
        assert type(compiled.kernel_module.CFG).__module__ == f"cudnn.sdpa.fwd.kernels.sm107.{owner}.config"
        metadata = compiled.implementation_config
        assert graph.selected_engine.name == f"sdpa_fwd_sm107_{owner}"
        assert (metadata["query_tile"], metadata["cluster_m"], metadata["cta_group"], metadata["split_kv"]) == (128, cga, cga, split)
        return result

    g, vp, ws, bufs, tensors = build(record)
    g.execute(vp, ws)
    shared._check(bufs, thd=False, sq=17, skv=256)

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
        shared._check(other, thd=False, sq=17, skv=256)
        bufs["descale_q"].fill_(0.5)
        bufs["scale_o"].fill_(1.7)
        for name in ("o", "lse", "amax_o"):
            bufs[name].fill_(float("nan"))
        capture.replay()
        shared._check(bufs, thd=False, sq=17, skv=256)

    # Eager execution rebinds caller storage and scales independently of the
    # retained executable's capture-time pointers.
    for name in ("q", "k", "v", "o", "lse", "amax_o", "descale_q", "scale_o"):
        bufs[name] = bufs[name].clone()
        vp[tensors[name]] = bufs[name]
    g.execute(vp, torch.empty_like(ws))
    shared._check(bufs, thd=False, sq=17, skv=256)


@pytest.mark.parametrize("d", [192, 256])
def test_retired_fp8_recipe_is_rejected_without_silent_substitution(d):
    import cudnn._plan_serdes as ps

    graph, _, _, _, _ = shared._case(arch="sm107", d=d, dv=128 if d == 192 else 256, sq=17, skv=256)
    blob = graph.serialize(as_bytes=True, include_kernels=False)
    start = len(ps.MAGIC) + ps._LEN.size
    (length,) = ps._LEN.unpack_from(blob, len(ps.MAGIC))
    header = json.loads(blob[start : start + length])
    header["plan"].update(engine_id=20514, engine_name="sdpa_fwd_prefill_sm107_fp8", knobs=[])
    encoded = json.dumps(header).encode()
    target = cudnn.pygraph()
    with pytest.raises((ValueError, RuntimeError), match="20514|owned by|not supported|No execution plans"):
        target.deserialize(ps.MAGIC + ps._LEN.pack(len(encoded)) + encoded)
    assert not target._nodes and not target._planning_done
    # The same empty destination remains reusable for an eligible current recipe.
    target.deserialize(blob)
    assert target.get_engine_and_knobs_at_index(target._plan_index)[0] == (20561 if d == 192 else 20562)


@pytest.mark.parametrize("d", [192, 256])
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group("gpu_exclusive")
def test_mid_fp8_refilled_stage_warm_stress(d):
    from warm_stress import StressCase, assert_warm_stress
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    cga, engine_id, b = (2, 20561, 256) if d == 192 else (1, 20562, 512)
    record = (engine_id, SdpaFwdKnobs(tile_m=128, tile_n=128, cga=cga, sched_policy=0, pack_gqa=True, split_kv=1).to_public())
    graph, vp, ws, bufs, _ = shared._case(arch="sm107", d=d, dv=128 if d == 192 else 256, b=b, sq=17, skv=1024, record=record)
    graph.execute(vp, ws)
    shared._check(bufs, thd=False, b=b, sq=17, skv=1024)
    # 1024 physical CTAs and eight KV tiles exercise successive ring reuse.
    assert_warm_stress(
        StressCase(
            f"SM107 D{d} per-tensor FP8",
            lambda: graph.execute(vp, ws),
            {"o": bufs["o"].transpose(1, 2), "stats": bufs["lse"], "amax_o": bufs["amax_o"]},
            thread_blocks=b * 2 * cga,
            refills_per_launch=4,
        ),
        warm=256,
        poisoned=256,
    )


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_d192_cga1_causal_public_record(dtype):
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs
    from test_sdpa_fwd_dsl_sm107 import _fp8_d192_oracle
    from determinism_utils import assert_replays

    record = (20561, SdpaFwdKnobs(tile_m=128, tile_n=128, cga=1, sched_policy=0, pack_gqa=False, split_kv=1).to_public())
    graph, vp, ws, bufs, _ = shared._case(
        arch="sm107",
        d=192,
        dv=128,
        b=2,
        heads=(2, 2),
        sq=129,
        skv=1024,
        dtype=dtype,
        output_dtype=torch.float8_e4m3fn,
        record=record,
        sdpa_kwargs={"right_bound": 0},
    )
    assert graph.get_engine_and_knobs_at_index(graph._plan_index) == record
    compiled = graph._compiled_plans[graph._plan_index]._compiled
    assert compiled.kernel_module.__file__.endswith("sm107/d192_fp8_multi_q/kernel.py")
    assert compiled.implementation_config["cluster_m"] == 1

    def check():
        q, k, v = (bufs[n].double() * bufs["descale_" + n].double() for n in ("q", "k", "v"))
        ref, stats = _fp8_d192_oracle(q, k, v, attn_scale=192**-0.5, causal=True)
        am = ref.abs().amax()
        ref *= bufs["scale_o"].double()
        floor = (ref - ref.to(bufs["o"].dtype).double()).abs().max().item()
        torch.testing.assert_close(bufs["o"].double(), ref, atol=max(0.03, 3.0 * floor), rtol=0.03)
        torch.testing.assert_close(bufs["lse"].double(), stats, atol=0.01, rtol=0.01)
        torch.testing.assert_close(bufs["amax_o"][0].double(), am, atol=0.03, rtol=0.03)

    def run():
        graph.execute(vp, ws)
        return {n: bufs[n].clone() for n in ("o", "lse", "amax_o")}

    assert_replays(run, repeats=3)
    check()
    with shared._cuda_graph() as capture:
        with torch.cuda.graph(capture):
            graph.execute(vp, ws)
        bufs["descale_v"].fill_(0.3)
        bufs["scale_o"].fill_(1.7)
        for name in ("o", "lse", "amax_o"):
            bufs[name].fill_(float("nan"))
        capture.replay()
        check()
