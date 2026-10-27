# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Paged K/V tables and physical THD Q/O/Stats origins are independent."""

import math

import pytest

from frost_test_utils import requires_dsl

pytestmark = [requires_dsl]


@pytest.mark.parametrize(
    "dim,dtype_name,pool_layout,stats_layout,cga,splits,pack_gqa",
    [
        pytest.param(128, "bfloat16", "hnd", "token", 2, 1, True, marks=pytest.mark.L0),
        pytest.param((576, 512), "bfloat16", "mixed", "token", 2, 1, True, marks=pytest.mark.L0, id="d576_partial_pack_mixed"),
        pytest.param((576, 512), "bfloat16", "hnd", "head", 2, 1, False, marks=pytest.mark.L0, id="d576_unpacked_hnd"),
        pytest.param((576, 512), "float16", "nhd", "head", 2, 1, True, marks=pytest.mark.L1, id="d576_fp16_partial_pack"),
        pytest.param((576, 512), "bfloat16", "mixed", "token", 2, 5, True, marks=pytest.mark.L0, id="d576_split5_partial_pack_mixed"),
        pytest.param((576, 512), "bfloat16", "hnd", "head", 2, 2, False, marks=pytest.mark.L0, id="d576_split2_unpacked_hnd"),
        pytest.param((576, 512), "float16", "nhd", "head", 2, 9, True, marks=pytest.mark.L1, id="d576_split9_fp16_partial_pack"),
        pytest.param(128, "float16", "nhd", "head", 2, 1, True, marks=pytest.mark.L1),
        pytest.param(128, "bfloat16", "mixed", "token", 2, 1, True, marks=pytest.mark.L1),
        pytest.param(256, "bfloat16", "hnd", "head", 2, 1, True, marks=pytest.mark.L0),
        pytest.param(256, "float16", "nhd", "token", 2, 1, True, marks=pytest.mark.L1),
        pytest.param(256, "bfloat16", "mixed", "head", 2, 1, True, marks=pytest.mark.L1),
        pytest.param(128, "bfloat16", "hnd", "token", 1, 1, True, marks=pytest.mark.L0),
        pytest.param(128, "float16", "nhd", "head", 1, 1, True, marks=pytest.mark.L1),
        pytest.param(128, "bfloat16", "mixed", "token", 1, 1, True, marks=pytest.mark.L1),
        pytest.param(128, "bfloat16", "hnd", "token", None, 1, True, marks=pytest.mark.L0, id="engine_default"),
        pytest.param(128, "bfloat16", "hnd", "token", 1, 2, True, marks=pytest.mark.L0, id="split2_hnd"),
        pytest.param(128, "bfloat16", "hnd", "token", 1, 5, True, marks=pytest.mark.L0, id="integration_split5"),
        pytest.param(128, "float16", "nhd", "head", 1, 3, True, marks=pytest.mark.L1, id="split3_nhd"),
        pytest.param(128, "bfloat16", "mixed", "token", 1, 8, True, marks=pytest.mark.L1, id="split8_mixed"),
        pytest.param(128, "bfloat16", "hnd", "head", 1, 2, False, marks=pytest.mark.L1, id="split2_unpacked"),
    ],
)
def test_paged_thd_origins_rebind_and_capture(dim, dtype_name, pool_layout, stats_layout, cga, splits, pack_gqa):
    import cudnn
    import torch
    from cudnn.engines.manifest import family_for

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("SM100/SM103 physical THD origin path")
    if cga == 1 and splits == 1 and torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("CGA1 paged prefill is currently admitted on SM100")
    torch.manual_seed(261026737)
    d_qk, d_v = dim if isinstance(dim, tuple) else (dim, dim)
    batch, heads, kv_heads, maximum, page, kv_max = 2, (24 if d_qk == 576 else 16), (2 if d_qk == 576 else 4), 320, 16, (1024 if d_qk == 576 else 8192)
    pages_per_sequence, page_count = kv_max // page, batch * kv_max // page
    dtype = getattr(torch, dtype_name)
    io_type = cudnn.data_type.HALF if dtype_name == "float16" else cudnn.data_type.BFLOAT16
    widths = {"q": torch.int64, "o": torch.int32, "stats": torch.int64}
    multipliers = {"q": heads * (16 if d_qk == 576 else d_qk), "o": d_v, "stats": 1}
    graph_ports = ("q", "k", "v", "o", "stats", "ql", "kl", "kt", "vt", "ro_q", "ro_o", "ro_stats")

    def buffers(wide=False):
        q_capacity, o_capacity, stats_capacity = (640 if wide else 512), 640, 672
        pitch = d_qk + (16 if wide else 0)
        result = {
            "q": torch.empty(q_capacity, heads, pitch, device="cuda", dtype=dtype)[..., :d_qk],
            "backing_o": torch.empty(o_capacity, heads, pitch, device="cuda", dtype=dtype),
            "stats": torch.empty((stats_capacity, heads) if stats_layout == "token" else (heads, stats_capacity), device="cuda", dtype=torch.float32),
            "ql": torch.empty(batch, 1, 1, 1, device="cuda", dtype=torch.int32),
            "kl": torch.empty(batch, 1, 1, 1, device="cuda", dtype=torch.int32),
        }
        result["o"] = result["backing_o"][..., :d_v]
        if pool_layout != "mixed":
            shape = (page_count + 1, kv_heads, page, 2 * d_qk) if pool_layout == "hnd" else (page_count + 1, page, kv_heads, 2 * d_qk)
            interleaved = torch.empty(shape, device="cuda", dtype=dtype)
            if pool_layout == "nhd":
                interleaved = interleaved.transpose(1, 2)
            result["k"], value_record = interleaved.split(d_qk, dim=-1)
            result["v"] = value_record[..., :d_v]
        else:
            result["k"] = torch.empty(page_count + 1, kv_heads, page, d_qk, device="cuda", dtype=dtype)
            result["v"] = torch.empty(page_count + 1, page, kv_heads, d_v, device="cuda", dtype=dtype).transpose(1, 2)
        for name in ("kt", "vt"):
            result[name] = torch.empty(batch, 1, pages_per_sequence, 1, device="cuda", dtype=torch.int32)
        if d_qk == 576:
            # Independent table pitches must survive both binding and replay.
            # A shared/contiguous stride reads the poisoned padding pages.
            step = 3 if wide else 2
            table_storage = torch.full((batch, 1, pages_per_sequence * step + 5, 1), page_count, device="cuda", dtype=torch.int32)
            result["vt"] = table_storage[:, :, 1 : 1 + pages_per_sequence * step : step, :]
        for name, width in widths.items():
            result["ro_" + name] = torch.empty(batch + 1, 1, 1, 1, device="cuda", dtype=width)
        return result

    def fill(values, shift, replay=False, keyless=False, empty=False):
        ql = ([257, 49] if replay else [300, 177]) if not empty else [0, 0]
        kl = [kv_max - 31, kv_max // 2 + 7] if replay else [kv_max, kv_max]
        if keyless:
            kl[1] = 0
        # shift=0 is the integration reproducer's shared [0,320,512] Q slots.
        # Subsequent bindings/replays also separate the three physical origins.
        starts = {
            "q": [shift, 320 + shift],
            "o": [shift + (16 if shift else 0), 320 + shift + (32 if shift else 0)],
            "stats": [shift + (32 if shift else 0), 320 + shift + (64 if shift else 0)],
        }
        for name, lengths in (("ql", ql), ("kl", kl)):
            values[name].copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32).view(batch, 1, 1, 1))
        for name in ("q", "o", "stats"):
            tensor = values[name]
            token_stride = tensor.stride(0) if name != "stats" or stats_layout == "token" else 1
            capacity = tensor.shape[0] if name != "stats" or stats_layout == "token" else tensor.shape[1]
            assert all(x * token_stride % multipliers[name] == 0 for x in starts[name] + [capacity])
            offsets = [(x * token_stride) // multipliers[name] for x in starts[name] + [capacity]]
            values["ro_" + name].copy_(torch.tensor(offsets, device="cuda", dtype=widths[name]).view(batch + 1, 1, 1, 1))
        values["q"].fill_(float("nan"))
        for start, length in zip(starts["q"], ql):
            values["q"][start : start + length].copy_(torch.randn(length, heads, d_qk, device="cuda", dtype=dtype))
        for name in ("k", "v"):
            values[name].normal_()
            values[name][-1].fill_(float("nan"))
        for name in ("kt", "vt"):
            # Page order is host test data; avoid an unrelated GPU radix sort.
            values[name].copy_(torch.randperm(page_count, dtype=torch.int32).view(batch, 1, pages_per_sequence, 1))
        return ql, kl, starts

    first = buffers()
    graph = cudnn.pygraph(
        io_data_type=io_type,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        is_override_shape_enabled=True,
    )
    nodes = {name: graph.tensor_like(first[name]) for name in graph_ports if name not in ("q", "o", "stats")}
    nodes["q"] = graph.tensor(dim=(batch, heads, maximum, d_qk), stride=(maximum * heads * d_qk, d_qk, heads * d_qk, 1), data_type=io_type)
    nodes["q"].set_ragged_offset(nodes["ro_q"]).set_ragged_offset_multiplier(multipliers["q"])
    nodes["o"], nodes["stats"] = graph.sdpa(
        q=nodes["q"],
        k=nodes["k"],
        v=nodes["v"],
        is_inference=False,
        attn_scale=d_qk**-0.5,
        use_causal_mask_bottom_right=True,
        use_padding_mask=True,
        seq_len_q=nodes["ql"],
        seq_len_kv=nodes["kl"],
        paged_attention_k_table=nodes["kt"],
        paged_attention_v_table=nodes["vt"],
        paged_attention_max_seq_len_kv=kv_max,
        max_total_seq_len_q=477,
        stats_use_log2=stats_layout == "head",
    )
    nodes["o"].set_output(True).set_dim((batch, heads, maximum, d_v)).set_stride((maximum * heads * d_qk, d_qk, heads * d_qk, 1)).set_ragged_offset(
        nodes["ro_o"]
    ).set_ragged_offset_multiplier(multipliers["o"])
    stats_stride = (maximum * heads, 1, heads, 1) if stats_layout == "token" else (maximum, first["stats"].shape[1], 1, 1)
    nodes["stats"].set_output(True).set_dim((batch, heads, maximum, 1)).set_stride(stats_stride).set_data_type(cudnn.data_type.FLOAT).set_ragged_offset(
        nodes["ro_stats"]
    )
    graph.validate()
    graph.build_operation_graph()
    if cga is None:
        graph.create_execution_plans([cudnn.heur_mode.A])
        graph.select_plan(next(i for i in range(len(graph.plans)) if graph.get_plan_name_at_index(i).startswith("sdpa_fwd_prefill_sm100")))
    else:
        graph.create_execution_plan(
            family_for(graph).offered_ids()["sdpa_fwd_prefill_sm100"],
            {cudnn.knob_type.TILE_CGA_M: cga, cudnn.knob_type.SPLIT_KV: splits, cudnn.knob_type.PACK_GQA: int(pack_gqa)},
        )
        graph.select_plan(0)
    graph.check_support()
    graph.build_plans()
    assert graph.get_plan_name_at_index(graph._plan_index).startswith("sdpa_fwd_prefill_sm100")
    print("ORIGIN_PLAN", graph.get_plan_name_at_index(graph._plan_index), flush=True)
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    def poison(values):
        values["backing_o"].fill_(-17)
        values["stats"].fill_(-29)
        workspace.fill_(255)

    def check(values, state):
        ql, kl, starts = state
        stats = values["stats"] if stats_layout == "token" else values["stats"].T
        live_o = torch.zeros(values["o"].shape[0], device="cuda", dtype=torch.bool)
        live_stats = torch.zeros(stats.shape[0], device="cuda", dtype=torch.bool)
        for b, (nq, nk) in enumerate(zip(ql, kl)):
            if not nq:
                continue
            k = values["k"][values["kt"][b].flatten().long()].transpose(0, 1).reshape(kv_heads, kv_max, d_qk)[:, :nk].double()
            v = values["v"][values["vt"][b].flatten().long()].transpose(0, 1).reshape(kv_heads, kv_max, d_v)[:, :nk].double()
            k, v = k.repeat_interleave(heads // kv_heads, 0), v.repeat_interleave(heads // kv_heads, 0)
            for lo in range(0, nq, 32):
                hi = min(lo + 32, nq)
                q = values["q"][starts["q"][b] + lo : starts["q"][b] + hi].double().transpose(0, 1)
                scores = q @ k.transpose(-1, -2) * d_qk**-0.5
                scores.masked_fill_(torch.arange(nk, device="cuda")[None, :] > torch.arange(nk - nq + lo, nk - nq + hi, device="cuda")[:, None], -torch.inf)
                expected = (scores.softmax(-1) @ v).transpose(0, 1)
                expected_stats = scores.logsumexp(-1).T * (math.log2(math.e) if stats_layout == "head" else 1)
                os, ss = starts["o"][b], starts["stats"][b]
                torch.testing.assert_close(values["o"][os + lo : os + hi].double(), expected, atol=0.004, rtol=0.004)
                torch.testing.assert_close(stats[ss + lo : ss + hi].double(), expected_stats, atol=0.0006, rtol=0)
            live_o[starts["o"][b] : starts["o"][b] + nq] = True
            live_stats[starts["stats"][b] : starts["stats"][b] + nq] = True
        assert torch.all(values["o"][~live_o] == -17)
        assert torch.all(stats[~live_stats] == -29)
        assert torch.all(values["backing_o"][..., d_v:] == -17)
        assert torch.isnan(values["k"][-1]).all() and torch.isnan(values["v"][-1]).all()

    captures = []
    try:
        for values, shift in ((first, 0), (buffers(wide=True), 32)):
            state = fill(values, shift)
            pack = {nodes[name]: values[name] for name in graph_ports}
            overrides = dict(
                override_uids=[nodes[name].get_uid() for name in ("q", "o")],
                override_shapes=[nodes[name].get_dim() for name in ("q", "o")],
                override_strides=[(maximum * values[name].stride(0), values[name].stride(1), values[name].stride(0), 1) for name in ("q", "o")],
            )
            poison(values)
            torch.cuda.set_sync_debug_mode("error")
            try:
                graph.execute(pack, workspace, **overrides)
            finally:
                torch.cuda.set_sync_debug_mode("default")
            torch.cuda.synchronize()
            check(values, state)
            captured = torch.cuda.CUDAGraph()
            captures.append((captured, values, shift))
            with torch.cuda.graph(captured):
                graph.execute(pack, workspace, **overrides)
        for captured, values, shift in reversed(captures):
            state = fill(values, shift + 16, replay=True, keyless=shift == 0)
            poison(values)
            captured.replay()
            torch.cuda.synchronize()
            check(values, state)
        for captured, values, shift in captures:
            state = fill(values, shift, empty=True)
            poison(values)
            captured.replay()
            torch.cuda.synchronize()
            check(values, state)
    finally:
        for captured, _, _ in captures:
            captured.reset()
