# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Physical THD origins remain independent from masks and compact work prefixes."""

import math

import pytest

from frost_test_utils import requires_dsl

pytestmark = [requires_dsl]


@pytest.mark.parametrize(
    "dim,dtype_name,stats_layout,total_hints",
    [
        pytest.param(
            dim,
            dtype,
            stats,
            totals,
            marks=pytest.mark.L0 if dtype == "bfloat16" and stats == "token" and totals else pytest.mark.L1,
        )
        for dim in (128, 256)
        for dtype in ("float16", "bfloat16")
        for stats in ("token", "head")
        for totals in (False, True)
    ],
)
def test_independent_thd_origins_rebind_and_capture(dim, dtype_name, stats_layout, total_hints):
    import cudnn
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("SM100/SM103 physical THD origin path")
    torch.manual_seed(117)
    batch, heads, kv_heads, maximum = 4, 8, 2, 256
    capacity = batch * maximum
    dtype = getattr(torch, dtype_name)
    ports = ("q", "k", "v", "o", "stats")
    widths = (torch.int64, torch.int32, torch.int64, torch.int32, torch.int64)
    multipliers = (heads * dim, dim, 1, dim, 1)
    q_lengths, kv_lengths = [129, 0, 65, 33], [147, 0, 92, 50]
    log2_stats = stats_layout == "head"

    def tensors(head_padding=0):
        result = {
            name: torch.empty(batch, maximum, h, dim + head_padding, device="cuda", dtype=dtype)[..., :dim].transpose(1, 2)
            for name, h in (("q", heads), ("k", kv_heads), ("v", kv_heads), ("o", heads))
        }
        result["stats"] = (
            torch.empty(capacity, heads, device="cuda", dtype=torch.float32)
            if stats_layout == "token"
            else torch.empty(heads, capacity, device="cuda", dtype=torch.float32)
        )
        result["ql"] = torch.empty(batch, 1, 1, 1, device="cuda", dtype=torch.int32)
        result["kl"] = torch.empty_like(result["ql"])
        for name, width in zip(ports, widths):
            result["ro_" + name] = torch.empty(batch + 1, 1, 1, 1, device="cuda", dtype=width)
        return result

    def flat(t):
        return t.as_strided((capacity, t.shape[1], dim), (t.stride(2), t.stride(1), t.stride(3)))

    def fill(values, shift, short=False):
        ql = [97, 0, 49, 17] if short else q_lengths
        # Replay also turns a live query sequence into a keyless row.
        kl = [101, 0, 76, 0] if short else kv_lengths
        origins = {name: [shift + port * 16 + x for x in (0, 176, 224, 384)] for port, name in enumerate(ports)}
        values["ql"].copy_(torch.tensor(ql, dtype=torch.int32, device="cuda").view(batch, 1, 1, 1))
        values["kl"].copy_(torch.tensor(kl, dtype=torch.int32, device="cuda").view(batch, 1, 1, 1))
        for name, multiplier in zip(ports, multipliers):
            stride = values[name].stride(2) if name != "stats" else (heads if stats_layout == "token" else 1)
            rows = origins[name] + [origins[name][-1] + maximum]
            values["ro_" + name].copy_(
                torch.tensor([x * stride // multiplier for x in rows], device="cuda", dtype=values["ro_" + name].dtype).view(batch + 1, 1, 1, 1)
            )
        for name in ("q", "k", "v"):
            view = flat(values[name])
            view.fill_(float("nan"))
            for start, length in zip(origins[name], ql if name == "q" else kl):
                view[start : start + length].copy_(torch.randn(length, view.shape[1], dim, device="cuda", dtype=dtype))
        return ql, kl, origins

    first = tensors()
    dt = cudnn.data_type.HALF if dtype_name == "float16" else cudnn.data_type.BFLOAT16
    graph = cudnn.pygraph(
        io_data_type=dt, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, is_override_shape_enabled=True
    )
    nodes = {name: graph.tensor_like(first[name]) for name in ("q", "k", "v", "ql", "kl")}
    for name, multiplier in zip(ports, multipliers):
        nodes["ro_" + name] = graph.tensor_like(first["ro_" + name])
        if name in ("q", "k", "v"):
            nodes[name].set_ragged_offset(nodes["ro_" + name]).set_ragged_offset_multiplier(multiplier)
    extra = dict(max_total_seq_len_q=sum(q_lengths), max_total_seq_len_kv=sum(kv_lengths)) if total_hints else {}
    nodes["o"], nodes["stats"] = graph.sdpa(
        q=nodes["q"],
        k=nodes["k"],
        v=nodes["v"],
        is_inference=False,
        attn_scale=dim**-0.5,
        diagonal_band_right_bound=0,
        use_padding_mask=True,
        seq_len_q=nodes["ql"],
        seq_len_kv=nodes["kl"],
        stats_use_log2=log2_stats,
        **extra,
    )
    nodes["o"].set_output(True).set_dim(first["o"].shape).set_stride(first["o"].stride()).set_ragged_offset(nodes["ro_o"]).set_ragged_offset_multiplier(
        multipliers[3]
    )
    stats_stride = (maximum * heads, 1, heads, 1) if stats_layout == "token" else (maximum, capacity, 1, 1)
    nodes["stats"].set_output(True).set_dim((batch, heads, maximum, 1)).set_stride(stats_stride).set_data_type(cudnn.data_type.FLOAT).set_ragged_offset(
        nodes["ro_stats"]
    )
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    index = next(i for i in range(len(graph.plans)) if graph.get_plan_name_at_index(i).startswith("sdpa_fwd_prefill_sm100"))
    graph.select_plan(index)
    graph.check_support()
    graph.build_plans()
    assert graph.get_plan_name_at_index(graph._plan_index).startswith("sdpa_fwd_prefill_sm100")
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    def poison(values):
        values["o"].fill_(-17)
        values["stats"].fill_(-29)
        workspace.fill_(255)

    def check(values, state):
        ql, kl, origins = state
        inputs = {name: flat(values[name]).double().cpu() for name in ("q", "k", "v")}
        for name, tensor in inputs.items():
            live = torch.zeros(capacity, dtype=torch.bool)
            for start, length in zip(origins[name], ql if name == "q" else kl):
                live[start : start + length] = True
            assert torch.isnan(tensor[~live]).all(), f"{name} padding was modified"
        output = flat(values["o"]).float().cpu()
        stats = values["stats"].float().cpu()
        if stats_layout == "head":
            stats = stats.T
        written_o, written_stats = torch.zeros(capacity, dtype=torch.bool), torch.zeros(capacity, dtype=torch.bool)
        for b, (nq, nk) in enumerate(zip(ql, kl)):
            if not nq:
                continue
            q = inputs["q"][origins["q"][b] : origins["q"][b] + nq].transpose(0, 1)
            k = inputs["k"][origins["k"][b] : origins["k"][b] + nk].repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)
            v = inputs["v"][origins["v"][b] : origins["v"][b] + nk].repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)
            scores = (q @ k.transpose(-1, -2)) * dim**-0.5
            scores.masked_fill_(torch.arange(nk)[None, :] > torch.arange(nq)[:, None], float("-inf"))
            expected = (scores.softmax(-1) @ v).transpose(0, 1)
            expected_stats = scores.logsumexp(-1).T * (math.log2(math.e) if log2_stats else 1)
            o_start, st_start = origins["o"][b], origins["stats"][b]
            torch.testing.assert_close(output[o_start : o_start + nq].double(), expected, atol=0.025 if dtype_name == "bfloat16" else 0.004, rtol=0.02)
            torch.testing.assert_close(stats[st_start : st_start + nq].double(), expected_stats, atol=0.003, rtol=0.003)
            written_o[o_start : o_start + nq] = True
            written_stats[st_start : st_start + nq] = True
        assert torch.all(output[~written_o] == -17)
        assert torch.all(stats[~written_stats] == -29)

    captures = []
    try:
        for values, shift in ((first, 0), (tensors(head_padding=16), 80)):
            state = fill(values, shift)
            pack = {nodes[name]: tensor for name, tensor in values.items()}
            # A later binding also changes all four token/head pitches. Both
            # captures must retain their own geometry and still read live offsets.
            overrides = dict(
                override_uids=[nodes[name].get_uid() for name in ports[:4]],
                override_shapes=[values[name].shape for name in ports[:4]],
                override_strides=[values[name].stride() for name in ports[:4]],
            )
            poison(values)
            graph.execute(pack, workspace, **overrides)
            torch.cuda.synchronize()
            check(values, state)
            captured = torch.cuda.CUDAGraph()
            captures.append((captured, values, shift))
            with torch.cuda.graph(captured):
                graph.execute(pack, workspace, **overrides)
        # Both captures survive a later pointer bind; mutate all metadata in place.
        for captured, values, shift in reversed(captures):
            state = fill(values, shift + 32, short=True)
            poison(values)
            captured.replay()
            torch.cuda.synchronize()
            check(values, state)
    finally:
        for captured, _, _ in captures:
            captured.reset()
