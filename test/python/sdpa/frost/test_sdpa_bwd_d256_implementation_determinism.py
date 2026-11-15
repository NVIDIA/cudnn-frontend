# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded repeatability and batch isolation for the fixed D256 and D192 backward bodies."""

import pytest

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell, requires_rubin

pytestmark = [pytest.mark.L0, requires_dsl]


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"], ids=["fp16", "bf16"])
@pytest.mark.parametrize("hkv", [2, 4], ids=["gqa", "mha"])
@pytest.mark.parametrize(
    "record_id,d,dv",
    [
        pytest.param(20607, 256, 256, marks=requires_pre_rubin_blackwell, id="sm100-d256"),
        pytest.param(20607, 192, 128, marks=requires_pre_rubin_blackwell, id="sm100-d192"),
        pytest.param(20632, 256, 256, marks=requires_rubin, id="sm107-native"),
        pytest.param(20633, 256, 256, marks=requires_rubin, id="sm107-two-by-two"),
        pytest.param(20610, 192, 128, marks=requires_rubin, id="sm107-d192"),
    ],
)
def test_fixed_record_replay_batch_and_golden(dtype_name, hkv, record_id, d, dv):
    import torch

    import test_sdpa_bwd_dsl_sm107 as helper
    from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot

    dtype = getattr(torch, dtype_name)
    h, sq, skv = 4, 257, 257
    generator = torch.Generator(device="cpu").manual_seed(34608)
    inputs = {}
    for name, length, heads, width in (("q", sq, h, d), ("k", skv, hkv, d), ("v", skv, hkv, dv), ("do", sq, h, dv)):
        inputs[name] = (torch.randn(3, length, heads, width, generator=generator) * 0.5).to(device="cuda", dtype=dtype).permute(0, 2, 1, 3)
    keep = helper._causal_keep(sq, skv)
    o, stats, _, *expected = helper._reference64(*(inputs[name] for name in ("q", "k", "v", "do")), keep, h // hkv)
    inputs["o"] = helper._bshd_empty(3, sq, h, dv, dt=dtype)
    inputs["o"].copy_(o)
    inputs["stats"] = stats.unsqueeze(-1).contiguous()
    compiled = {}

    def run(indices, *, change_neighbors=False):
        b = len(indices)
        selected = {}
        for name, value in inputs.items():
            shape = (b, *value.shape[1:])
            dest = (
                torch.empty(shape, device="cuda", dtype=value.dtype)
                if name == "stats"
                else helper._bshd_empty(b, value.shape[2], value.shape[1], value.shape[3], dt=value.dtype)
            )
            dest.copy_(value[list(indices)])
            selected[name] = dest
        if change_neighbors:
            # The target sequence is the first one; modify only its neighbor's upstream gradient.
            selected["do"][1:].mul_(1.5)
        before = snapshot(selected)
        for dst, src in (("dq", "q"), ("dk", "k"), ("dv", "v")):
            selected[dst] = torch.empty_like(selected[src]).fill_(float("nan"))
        if b not in compiled:
            graph, refs, outputs = helper._half_bwd_graph(b=b, hq=h, hkv=hkv, sq=sq, skv=skv, d=d, d_v=dv, dt=dtype, use_causal_mask=True)
            graph.validate()
            graph.build_operation_graph()
            graph.create_execution_plan(record_id, {})
            graph.select_plan(0)
            graph.check_support()
            graph.build_plans()
            refs.update(zip(("dq", "dk", "dv"), outputs))
            compiled[b] = graph, refs
        graph, refs = compiled[b]
        workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8).fill_(0xBD)
        graph.execute({refs[name]: value for name, value in selected.items()}, workspace)
        assert_bitwise({name: selected[name] for name in inputs}, before, "input mutation")
        actual = snapshot({name: selected[name] for name in ("dq", "dk", "dv")})
        reference = tuple(value[list(indices)].clone() for value in expected)
        if change_neighbors:
            for value in reference:
                value[1:].mul_(1.5)
        for name, want in zip(("dq", "dk", "dv"), reference):
            helper._check(name, selected[name], want, dtype)
        return actual

    alone = assert_replays(lambda: run([0]))
    for indices in ([0, 1, 2], [2, 0, 1], [0, 0, 2]):
        actual = run(indices)
        for position, source in enumerate(indices):
            if source == 0:
                assert_bitwise({name: value[position : position + 1] for name, value in actual.items()}, alone, f"batch {indices} position {position}")
    changed = run([0, 1, 2], change_neighbors=True)
    assert_bitwise({name: value[:1] for name, value in changed.items()}, alone, "neighbor isolation")
    assert_golden(f"sdpa_bwd_fixed_{record_id}_d{d}v{dv}_{dtype_name}_hkv{hkv}", {name: value[:1] for name, value in inputs.items()}, alone)
