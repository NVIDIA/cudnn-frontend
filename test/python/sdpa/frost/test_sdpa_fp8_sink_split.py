# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A split FP8 attention folds its sink once, before final scaling and Amax."""

import pytest
import torch

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell
import test_sdpa_fwd_paged_sm100 as paged
import test_sdpa_prepared_fp8 as dense

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]


@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("split", [2, 9])
@pytest.mark.parametrize("in_key", ["e4m3", "e5m2"])
@pytest.mark.parametrize(
    "s_q,H,KH,pack,hnd,P,out_key,stats,causal",
    [
        (1, 8, 2, True, True, 16, "bf16", True, "bottom_right"),
        (9, 8, 2, False, False, 128, "fp16", True, "top_left"),
        (17, 16, 2, True, False, 16, "e4m3", False, "bottom_right"),
        (3, 8, 8, False, True, 256, "e5m2", True, None),
    ],
)
def test_paged_fp8_sink_split(d, split, in_key, s_q, H, KH, pack, hnd, P, out_key, stats, causal):
    paged.test_paged_graph_fp8_sink(d, in_key, s_q, H, KH, pack, hnd, P, out_key, stats, causal, split=split)


@pytest.mark.parametrize("d,cga", [(64, 1), (128, 2), (256, 1)])
@pytest.mark.parametrize("split", [2, 9])
def test_paged_fp8_sink_split_rebind_replay(d, cga, split):
    paged.test_paged_graph_fp8_sink_rebind_replay(d, cga, split=split)


@pytest.mark.parametrize("d,dv", [(64, 64), (128, 128), (192, 128), (256, 256)])
@pytest.mark.parametrize("output", [torch.bfloat16, torch.float8_e4m3fn])
def test_dense_fp8_sink_split_rebind_replay(d, dv, output):
    sink = torch.linspace(-12, 12, 4, device="cuda").view(1, 4, 1, 1)
    g, vp, ws, bufs, ts = dense._case(d=d, dv=dv, sq=16, skv=512, split_kv=4, output_dtype=output, sink=sink)
    prepared = g._compiled_plans[g._plan_index]._prepared
    assert prepared.spec.has_sink and prepared.spec.combine is not None
    for iteration in range(2):
        if iteration:
            for name in ("q", "k", "v", "o", "lse", "amax_o", "sinks", "scale_o"):
                bufs[name] = bufs[name].clone()
                vp[ts[name]] = bufs[name]
            bufs["sinks"].fill_(5)
            bufs["scale_o"].fill_(1.9)
            ws = torch.empty_like(ws)
        g.execute(vp, ws)
        dense._check(bufs, thd=False, sq=16, skv=512)
        with dense._cuda_graph() as graph:
            torch.cuda.set_sync_debug_mode("error")
            try:
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                g.execute(vp, ws)
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            finally:
                torch.cuda.set_sync_debug_mode("default")
            with torch.cuda.graph(graph):
                g.execute(vp, ws)
            # Rebinding the same plan must not redirect an already captured
            # combine's sink/scalar/output pointers to the fresh buffers.
            fresh = {name: value.clone() for name, value in bufs.items()}
            rebound = dict(vp)
            for name, tensor in ts.items():
                if name in fresh:
                    rebound[tensor] = fresh[name]
            fresh["sinks"].fill_(7)
            fresh_ws = torch.empty_like(ws)
            g.execute(rebound, fresh_ws)
            dense._check(fresh, thd=False, sq=16, skv=512)
            bufs["sinks"].fill_(-3)
            for name in ("o", "lse", "amax_o"):
                bufs[name].fill_(float("nan"))
            graph.replay()
            dense._check(bufs, thd=False, sq=16, skv=512)


@pytest.mark.parametrize("q_lens", [False, True])
@pytest.mark.parametrize("output", ["bfloat16", "float8_e4m3fn"])
def test_quantized_sink_native_frame_and_storage(q_lens, output):
    import inspect
    import cudnn
    from cudnn.sdpa.fwd import prepared as prep
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine
    from test_sdpa_native_fp8_binding import _fixture, _native_pack, _INDICES

    spec, facts, frames, combined = _fixture(128, 128, 4, "float8_e4m3fn", output)
    spec.has_sink = True
    spec.seq_q_present = spec.combine_q_lens = q_lens
    facts["sinks"] = prep.BufferFacts(0x47000000, "float32", (2, 0), spec.qh, (spec.qh,), (1,))
    if not q_lens:
        facts.pop("seq_q_lens")
    spec.native = cudnn._pybind_module._SdpaDenseBinder(spec)
    slots = [n for n, p in inspect.signature(split_combine._host_ptr_quantized_sink).parameters.items() if "Constexpr" not in str(p.annotation)]
    for offset in (0, 0x100000000):
        fresh = {role: f._replace(ptr=f.ptr + offset) if f is not None else None for role, f in facts.items()}
        _, tail, identity = spec.native.bind_quantized(_native_pack(fresh), _INDICES, 0x50000000 + offset, 17)
        assert identity == 0 and len(tail) == len(slots)
        bound = dict(zip(slots, tail))
        assert bound["sinks_ptr"] == fresh["sinks"].ptr
        assert bound["amax_o_ptr"] == fresh["amax_o"].ptr
        assert bound["scale_o_ptr"] == fresh["scale_o"].ptr
        assert bound["stream"] == 17
        if q_lens:
            assert bound["q_lens_ptr"] == fresh["seq_q_lens"].ptr
    for role in (["sinks", "seq_q_lens"] if q_lens else ["sinks"]):
        bad = dict(facts, **{role: facts[role]._replace(span=0)})
        with pytest.raises(ValueError):
            spec.native.execute(_native_pack(bad), _INDICES, 17, workspace=0x50000000)
        assert frames == combined == []


def test_fp8_sink_split_requires_current_native_extension(monkeypatch):
    import cudnn

    class OldBinder:
        supports_combine_q_lens = True

    monkeypatch.setattr(cudnn._pybind_module, "_SdpaDenseBinder", OldBinder)
    sink = torch.zeros((1, 4, 1, 1), device="cuda")
    with pytest.raises(NotImplementedError, match="rebuild the extension"):
        dense._case(sq=16, skv=256, split_kv=4, sink=sink)


def test_fp8_sink_keeps_existing_split_domain():
    """A sink does not widen the independent FP8 split head-dimension domain."""
    sink = torch.zeros((1, 4, 1, 1), device="cuda")
    with pytest.raises(NotImplementedError, match="split_kv > 1 is wired only"):
        dense._case(d=512, sq=16, skv=256, split_kv=4, sink=sink)


@pytest.mark.parametrize("d,dv", [(64, 64), (128, 128), (192, 128), (256, 256), (512, 512)])
@pytest.mark.parametrize("output", [torch.bfloat16, torch.float8_e4m3fn])
def test_standalone_fp8_sink_split_default_scales(d, dv, output):
    from test_sdpa_prepared_fp8_split import test_prepared_fp8_split_standalone_default_scales

    test_prepared_fp8_split_standalone_default_scales("sm100", d, dv, output, sink=True)
