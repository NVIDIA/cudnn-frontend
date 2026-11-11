# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native half D256 prefill split: public admission, masks and empty partials."""

import pytest

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("shape", [(256, 256), (248, 224)])
@pytest.mark.parametrize("dtype", ["HALF", "BFLOAT16"])
@pytest.mark.parametrize("splits", [2, 4, 16])
def test_native_d256_prefill_split_admission(shape, dtype, splits):
    import cudnn
    from cudnn.sdpa.graph_analyzer import SdpaGraphFacts
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS, SdpaFwdKnobs, mismatch

    caps = next(s.capabilities for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm107")
    facts = SdpaGraphFacts(b=2, h_q=4, h_kv=2, s_q=385, s_kv=512, d_qk=shape[0], d_v=shape[1], dtype=getattr(cudnn.data_type, dtype), device_cc=(10, 7))
    assert mismatch(caps, facts, SdpaFwdKnobs(split_kv=splits, tile_m=128, tile_n=128, cga=2, pack_gqa=False)) is None


@pytest.mark.parametrize("splits", [1, 2, 16])
def test_native_d256_prefill_partial_workspace_dtype(splits, monkeypatch):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    api = object.__new__(SdpaFwdDslSm100)
    api._device_cc, api._fp8, api.flavor = (10, 7), False, (256, 256)
    api.thd, api.pack_gqa, api.cga, api.split_kv = False, False, 2, splits
    api.paged_page_size = 0
    monkeypatch.setattr(api, "_decode_q_tile", lambda: 0)
    assert api._fp32_partial_split() == (splits > 1)
    if splits > 1:
        import torch

        assert api._partial_torch_dtype() == torch.float32


_CASES = [
    # name, B, H, HK, SQ, SKV, split, causal, bottom-right, left, sink, Stats, scheduler, head pair
    ("unsplit_plain", 2, 4, 2, 256, 512, 1, False, False, None, False, True, 0, (256, 256)),
    ("unsplit_window_sink", 2, 4, 2, 385, 257, 1, True, True, 63, True, False, 0, (256, 256)),
    ("plain", 2, 4, 2, 256, 512, 2, False, False, None, False, True, 0, (256, 256)),
    ("empty_partitions", 2, 4, 4, 257, 128, 16, False, False, None, False, False, 0, (256, 256)),
    ("top_left_tail", 2, 4, 2, 129, 257, 4, True, False, None, False, True, 0, (256, 256)),
    ("bottom_right_dead_rows", 2, 4, 2, 385, 257, 4, True, True, None, False, True, 0, (256, 256)),
    ("window_sink", 2, 4, 2, 385, 257, 4, True, True, 63, True, True, 0, (256, 256)),
    ("sink_without_stats", 2, 4, 2, 257, 128, 16, False, False, None, True, False, 0, (256, 256)),
    ("lpt_persistent", 4, 8, 2, 513, 1024, 4, False, False, None, False, True, 1, (256, 256)),
    ("logical_envelope", 2, 4, 2, 385, 257, 4, True, True, 31, False, False, 0, (248, 224)),
]


@pytest.mark.parametrize("case", _CASES, ids=[c[0] for c in _CASES])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_native_d256_prefill_split_replay(case, dtype, monkeypatch):
    _run_case(case, dtype, monkeypatch)


@pytest.mark.parametrize("stats_kind", ["none", "natural", "log2"])
@pytest.mark.parametrize("ordered", [False, True])
def test_native_d256_prefill_split_graph(stats_kind, ordered, monkeypatch):
    case = ("graph_window_sink", 2, 4, 2, 257, 385, 4, True, True, 63, True, stats_kind != "none", 0, (256, 256))
    _run_case(case, "bfloat16", monkeypatch, graph_mode=True, stats_log2=stats_kind == "log2", ordered=ordered)


def _run_case(case, dtype, monkeypatch, *, graph_mode=False, stats_log2=False, ordered=False):
    import torch
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100
    from sdpa.frost.native_binding_utils import forbid_python_forward_binding

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("requires native SM107")
    from test_sdpa_fwd_dsl_sm100 import _require_dsl

    _require_dsl()
    _, b, h, hk, sq, sk, splits, causal, bottom, left, sink, stats, sched, shape = case
    d, dv = shape
    dtype = getattr(torch, dtype)
    torch.manual_seed(924)

    def tensor(heads, seq, width):
        # Real runtime strides include unread poisoned storage after each batch.
        backing = torch.randn(b, seq + 16, heads, width, device="cuda", dtype=dtype)
        backing[:, seq:] = torch.nan
        return backing[:, :seq].transpose(1, 2)

    q, k, v = tensor(h, sq, d), tensor(hk, sk, d), tensor(hk, sk, dv)
    ob = torch.full((b * sq * h * dv + 32,), 77, device="cuda", dtype=dtype)
    o = ob[16:-16].view(b, sq, h, dv).transpose(1, 2)
    lb = torch.full((b * h * sq + 32,), 77, device="cuda", dtype=torch.float32)
    lse = lb[16:-16].view(b, h, sq)
    sinks = torch.linspace(-0.5, 0.5, h, device="cuda") if sink else None
    scale = 0.073
    api = SdpaFwdDslSm100(
        q,
        k,
        v,
        o,
        lse if stats else None,
        has_sink=sink,
        scale_softmax=scale,
        is_causal=causal,
        causal_bottom_right=bottom,
        window_size_left=left,
        cga=2,
        split_kv=splits,
        pack_gqa=False,
        tile_m=128,
        tile_n=128,
        sched_policy=sched,
    )
    assert api.check_support()
    graph = None
    if graph_mode:
        import cudnn
        import inspect
        from cudnn.engines.manifest import family_for
        from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

        graph = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        inputs = [graph.tensor_like(t).set_uid(i + 1) for i, t in enumerate((q, k, v))]
        sink_node = graph.tensor_like(sinks.view(1, h, 1, 1)).set_uid(6)
        out, st = graph.sdpa(
            q=inputs[0],
            k=inputs[1],
            v=inputs[2],
            generate_stats=stats,
            attn_scale=scale,
            use_causal_mask_bottom_right=bottom,
            sliding_window_length=left + 1,
            sink_token=sink_node,
            stats_use_log2=stats_log2,
        )
        out.set_uid(4).set_output(True).set_dim(o.shape).set_stride(o.stride())
        if stats:
            st.set_uid(5).set_output(True).set_dim((*lse.shape, 1)).set_stride((*lse.stride(), 1)).set_data_type(cudnn.data_type.FLOAT)
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plan(
            family_for(graph).offered_ids()["sdpa_fwd_prefill_sm107"],
            SdpaFwdKnobs(tile_m=128, tile_n=128, cga=2, pack_gqa=False, split_kv=splits, sched_policy=sched).to_public(),
        )
        graph.select_plan(0)
        graph.check_support()
        graph.build_plans()
        api = inspect.getclosurevars(graph._compiled_plans[0]._compiled.default_stream).nonlocals["api"]
    else:
        api.compile()
    assert api._k_mod.__file__.endswith("sm107/prefill_d256_f16.py")
    assert api._k_mod.CFG.SPLIT_KV == splits and api._k_mod._FP32_PARTIALS == (splits > 1)
    assert api._dense_spec.native is not None
    assert api._dense_spec.fp32_partial == (splits > 1)
    if splits > 1:
        assert api._partial_torch_dtype() == torch.float32
    size = graph.get_workspace_size() if graph_mode else api.scratch_workspace_bytes()
    slab = torch.full((max(size, 1) + 512,), 0xB7, device="cuda", dtype=torch.uint8)
    workspace = slab[256 : 256 + max(size, 1)]
    forbid_python_forward_binding(monkeypatch)

    def run():
        torch.cuda.set_sync_debug_mode("error")
        try:
            if graph_mode:
                buffers = {1: q, 2: k, 3: v, 4: o, 6: sinks.view(1, h, 1, 1)}
                if stats:
                    buffers[5] = lse.unsqueeze(-1)
                if ordered:
                    graph.execute(tuple(buffers.values()), workspace, tensor_uids=tuple(buffers))
                else:
                    graph.execute(buffers, workspace)
            else:
                api.execute(q, k, v, o, lse if stats else None, sinks=sinks, workspace=workspace)
        finally:
            torch.cuda.set_sync_debug_mode("default")

    def check():
        # Keep the independent reference off the instrumented GPU.
        qq = q.cpu().double()
        kk, vv = (t.cpu().double().repeat_interleave(h // hk, dim=1) for t in (k, v))
        scores = qq @ kk.transpose(-1, -2) * scale
        if causal:
            row = torch.arange(sq)[:, None]
            col = torch.arange(sk)[None, :]
            anchor = row + (sk - sq if bottom else 0)
            live = col <= anchor
            if left is not None:
                live &= col >= anchor - left
            scores.masked_fill_(~live, -torch.inf)
        ref_lse = scores.logsumexp(-1)
        if sinks is not None:
            ref_lse = torch.logaddexp(ref_lse, sinks.cpu().double()[None, :, None])
        prob = torch.where(torch.isfinite(ref_lse[..., None]), (scores - ref_lse[..., None]).exp(), 0)
        ref = prob @ vv
        torch.testing.assert_close(o.cpu().float(), ref.float(), atol=2e-2, rtol=3e-2)
        if stats:
            expected_lse = ref_lse / 0.6931471805599453 if stats_log2 else ref_lse
            torch.testing.assert_close(lse.cpu().double(), expected_lse, atol=1e-3, rtol=1e-3)
        else:
            assert (lse == 77).all()
        for buf in (ob, lb):
            assert (buf[:16] == 77).all() and (buf[-16:] == 77).all()
        assert (slab[:256] == 0xB7).all() and (slab[256 + max(size, 1) :] == 0xB7).all()

    run()
    check()
    # Fresh pointers bind to the same compiled artifact.
    q, k, v = tensor(h, sq, d), tensor(hk, sk, d), tensor(hk, sk, dv)
    run()
    check()
    captured = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(captured):
            run()
        v.mul_(-0.7)
        o.fill_(torch.nan)
        captured.replay()
        check()
        expected = o.clone()
        captured.replay()
        assert torch.equal(o, expected)
    finally:
        captured.reset()
