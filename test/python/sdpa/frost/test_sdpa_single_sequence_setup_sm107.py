# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-sequence map setup preserves origins, replay and batch invariance."""

import pytest
import torch

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from frost_test_utils import requires_dsl, requires_rubin

pytestmark = [pytest.mark.L0, requires_dsl, requires_rubin]


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("cga,packed", [(1, False), (2, True)])
def test_single_sequence_physical_origins(dim, dtype_name, cga, packed, monkeypatch):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100
    from test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture

    original = SdpaFwdDslSm100.compile
    observed = []

    def compile_native(api):
        original(api)
        assert api._k_mod.__file__.endswith("sm107/prefill_d128_f16.py")
        observed.append(api.batch_size)

    monkeypatch.setattr(SdpaFwdDslSm100, "compile", compile_native)
    test_independent_thd_origins_rebind_and_capture(
        dim,
        dtype_name,
        "head" if packed else "token",
        True,
        packed,
        1,
        engine_name="sdpa_fwd_prefill_sm107",
        cga=cga,
        batch_override=1,
    )
    assert observed and all(b == 1 for b in observed)


@pytest.mark.parametrize(
    "dim,dtype_name,cga,packed,tile_n",
    [
        (64, "float16", 1, False, 128),
        (64, "bfloat16", 2, True, 64),
        (128, "float16", 2, False, 128),
        (128, "bfloat16", 1, True, 64),
    ],
)
def test_single_sequence_batch_bytes(dim, dtype_name, cga, packed, tile_n, engine_name="sdpa_fwd_prefill_sm107", paged=False, splits=1):
    import cudnn
    from cudnn.engines.manifest import family_for

    dtype = getattr(torch, dtype_name)
    dt = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    h, hk, capacity = 8, 2, 320
    rng = torch.Generator().manual_seed(871931)
    source = {
        name: (torch.randn(4, count, heads, dim, generator=rng) * 0.3).to(dtype).cuda()
        for name, count, heads in (("q", 129, h), ("k", 257, hk), ("v", 257, hk))
    }
    page, pages_per_sequence = 16, 17
    if paged:
        for name in ("k", "v"):
            storage = torch.full((4 * pages_per_sequence + 1, page, hk, dim), 17.0, dtype=dtype, device="cuda")
            for sequence in range(4):
                storage[sequence * pages_per_sequence : (sequence + 1) * pages_per_sequence].view(-1, hk, dim)[:257].copy_(source[name][sequence])
            source[name + "_pages"] = storage.transpose(1, 2)
    preserved = snapshot(source)
    qlens, klens = [129, 17, 0, 1], [257, 33, 0, 0]
    plans = {}
    # Split workspace scales with the declared token capacity. Keep an
    # oversized envelope while bounding the test allocation. Both paths
    # execute runtime B1/B3/B5/B129 through one plan.
    declared_batch = 256 if splits > 1 else 4096

    def run(order):
        b = len(order)
        values = {
            name: torch.full((b, capacity, heads, dim), torch.nan if name != "o" else -17, dtype=dtype, device="cuda").transpose(1, 2)
            for name, heads in ((("q", h), ("o", h)) if paged else (("q", h), ("k", hk), ("v", hk), ("o", h)))
        }
        if paged:
            for name in ("k", "v"):
                values[name] = source[name + "_pages"]
                values[name + "t"] = (
                    torch.tensor(order, device="cuda", dtype=torch.int32)[:, None] * pages_per_sequence
                    + torch.arange(pages_per_sequence, device="cuda", dtype=torch.int32)[None, :]
                ).view(b, 1, pages_per_sequence, 1)
        values["stats"] = torch.full((b * capacity, h), -29.0, device="cuda")
        values["ql"] = torch.tensor([qlens[i] for i in order], device="cuda", dtype=torch.int32).view(b, 1, 1, 1)
        values["kl"] = torch.tensor([klens[i] for i in order], device="cuda", dtype=torch.int32).view(b, 1, 1, 1)
        ports = ("q", "o", "stats") if paged else ("q", "k", "v", "o", "stats")
        origins = {name: [j * capacity + port * 8 for j in range(b)] for port, name in enumerate(ports)}
        flat = {name: values[name].transpose(1, 2).view(-1, values[name].shape[1], dim) for name in ports if name != "stats"}
        for name in origins:
            pitch = h if name == "stats" else values[name].stride(2)
            values["off_" + name] = torch.tensor([*origins[name], b * capacity], device="cuda", dtype=torch.int64).view(b + 1, 1, 1, 1) * pitch
        for name in (("q",) if paged else ("q", "k", "v")):
            for sequence, start in zip(order, origins[name]):
                count = qlens[sequence] if name == "q" else klens[sequence]
                flat[name][start : start + count].copy_(source[name][sequence, :count])
        if not plans:
            graph = cudnn.pygraph(
                io_data_type=dt, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, is_override_shape_enabled=True
            )
            nodes = {name: graph.tensor_like(t) for name, t in values.items() if name not in ("o", "stats")}
            for name in (("q",) if paged else ("q", "k", "v")):
                nodes[name].set_ragged_offset(nodes["off_" + name])
            paged_options = (
                dict(paged_attention_k_table=nodes["kt"], paged_attention_v_table=nodes["vt"], paged_attention_max_seq_len_kv=page * pages_per_sequence)
                if paged
                else {}
            )
            nodes["o"], nodes["stats"] = graph.sdpa(
                q=nodes["q"],
                k=nodes["k"],
                v=nodes["v"],
                generate_stats=True,
                attn_scale=dim**-0.5,
                use_padding_mask=True,
                seq_len_q=nodes["ql"],
                seq_len_kv=nodes["kl"],
                use_causal_mask=True,
                **({"max_total_seq_len_q": declared_batch * max(qlens)} if splits > 1 else {}),
                **paged_options,
            )
            nodes["o"].set_output(True).set_dim(values["o"].shape).set_stride(values["o"].stride()).set_ragged_offset(nodes["off_o"])
            nodes["stats"].set_output(True).set_dim((b, h, capacity, 1)).set_stride((capacity * h, 1, h, 1)).set_data_type(
                cudnn.data_type.FLOAT
            ).set_ragged_offset(nodes["off_stats"])
            # FlashInfer declares a large batch capacity and executes smaller
            # batches through the same plan. Do not specialize on this envelope.
            for name, node in nodes.items():
                if paged and name in ("k", "v"):
                    continue
                shape = list(node.get_dim())
                shape[0] = declared_batch + int(name.startswith("off_"))
                node.set_dim(shape)
            graph.validate()
            graph.build_operation_graph()
            graph.create_execution_plan(
                family_for(graph).offered_ids()[engine_name],
                {
                    cudnn.knob_type.TILE_CGA_M: cga,
                    cudnn.knob_type.PACK_GQA: int(packed),
                    cudnn.knob_type.SPLIT_KV: splits,
                    cudnn.knob_type.TILE_N: tile_n,
                },
            )
            graph.select_plan(0)
            graph.check_support()
            graph.build_plans()
            compiled = graph._compiled_plans[0]
            assert compiled._prepared.spec.origin_specs is not None
            mod = compiled._compiled.kernel_module
            body = "decode_d128_f16.py" if splits > 1 else "prefill_d128_f16.py"
            assert mod.__file__.endswith(engine_name.rsplit("_", 1)[1] + "/" + body)
            assert compiled._prepared.spec.b == declared_batch
            plans["shared"] = graph, nodes
        graph, nodes = plans["shared"]
        pack = {nodes[name]: value for name, value in values.items()}
        ws = torch.empty(max(1, graph.get_workspace_size()), device="cuda", dtype=torch.uint8)
        shapes = {name: list(value.shape) for name, value in values.items()}
        strides = {name: list(value.stride()) for name, value in values.items()}
        shapes["stats"] = [b, h, capacity, 1]
        strides["stats"] = [capacity * h, 1, h, 1]
        overrides = dict(
            override_uids=[nodes[name].get_uid() for name in values],
            override_shapes=[shapes[name] for name in values],
            override_strides=[strides[name] for name in values],
        )
        graph.execute(pack, ws, **overrides)
        expected_bytes = snapshot({name: values[name] for name in ("o", "stats")}, allowed_inf={"stats": -torch.inf})
        capture = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(capture):
                graph.execute(pack, ws, **overrides)
            values["o"].fill_(-17)
            values["stats"].fill_(-29)
            ws.fill_(255)
            capture.replay()
            torch.cuda.synchronize()
            assert_bitwise({name: values[name] for name in expected_bytes}, expected_bytes, "poisoned-workspace replay")
        finally:
            capture.reset()
        result = {"o": [], "stats": []}
        written_o = torch.zeros(b * capacity, dtype=torch.bool, device="cuda")
        written_stats = torch.zeros_like(written_o)
        for sequence, os, ls in zip(order, origins["o"], origins["stats"]):
            nq, nk = qlens[sequence], klens[sequence]
            if not nq:
                continue
            q = source["q"][sequence, :nq].cpu().double().transpose(0, 1)
            k = source["k"][sequence, :nk].cpu().double().repeat_interleave(h // hk, 1).transpose(0, 1)
            v = source["v"][sequence, :nk].cpu().double().repeat_interleave(h // hk, 1).transpose(0, 1)
            scores = (q @ k.transpose(-1, -2)) * dim**-0.5
            scores.masked_fill_(torch.arange(nk)[None, :] > torch.arange(nq)[:, None], -torch.inf)
            reference = (scores.softmax(-1).nan_to_num() @ v).transpose(0, 1)
            stats = scores.logsumexp(-1).T
            actual_o, actual_stats = flat["o"][os : os + nq], values["stats"][ls : ls + nq]
            torch.testing.assert_close(actual_o.cpu().double(), reference, atol=0.004, rtol=0.02)
            torch.testing.assert_close(actual_stats.cpu().double(), stats, atol=0.0006, rtol=0)
            written_o[os : os + nq] = True
            written_stats[ls : ls + nq] = True
            result["o"].append(actual_o)
            result["stats"].append(actual_stats)
        assert torch.all(flat["o"][~written_o] == -17)
        assert torch.all(values["stats"][~written_stats] == -29)
        assert_bitwise(source, preserved, "source data unchanged")
        return {name: torch.cat(parts) for name, parts in result.items()}

    alone = assert_replays(lambda: run([0]), repeats=3, allowed_inf={"stats": -torch.inf})
    for order in ([1, 0, 2], [2, 1, 0], [0, 1, 0, 2, 3], [1] * 64 + [0] + [2] * 64):
        actual = snapshot(run(order), allowed_inf={"stats": -torch.inf})
        offset = 0
        for sequence in order:
            nq = qlens[sequence]
            if sequence == 0:
                assert_bitwise({name: actual[name][offset : offset + nq] for name in alone}, alone, f"target at token {offset}")
            offset += nq
    neighbor = source["q"][1].clone()
    source["q"][1].mul_(-0.5)
    preserved = snapshot(source)
    actual = snapshot(run([1, 0, 2]), allowed_inf={"stats": -torch.inf})
    assert_bitwise({name: actual[name][qlens[1] :] for name in alone}, alone, "changed neighbor")
    source["q"][1].copy_(neighbor)
    preserved = snapshot(source)
    assert_bitwise(run([0]), alone, "single sequence after multi-sequence bindings")
    arch = engine_name.rsplit("_", 1)[1]
    suffix = ("-paged" if paged else "") + (f"-split{splits}" if splits > 1 else "")
    assert_golden(
        f"sdpa-{arch}-single-sequence-d{dim}-{dtype_name}-cga{cga}-pack{int(packed)}-n{tile_n}{suffix}", source, alone, allowed_inf={"stats": -torch.inf}
    )


@pytest.mark.parametrize("dim", [64, 128])
def test_single_sequence_runtime_batch_guard(dim, monkeypatch):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100
    from test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture

    compile_plan = SdpaFwdDslSm100.compile
    seen = []

    def compiled(api):
        compile_plan(api)
        assert api.batch_size == 4
        assert api._k_mod.__file__.endswith("sm107/prefill_d128_f16.py")
        seen.append(api.batch_size)

    monkeypatch.setattr(SdpaFwdDslSm100, "compile", compiled)
    test_independent_thd_origins_rebind_and_capture(dim, "bfloat16", "token", True, False, 1, engine_name="sdpa_fwd_prefill_sm107", cga=1)
    assert seen
