# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Half Q stays half when native D128 reads independently typed FP8 K/V."""

import itertools
import math

import pytest
import torch

from frost_test_utils import _SM, requires_dsl
from test_utils import torch_fork_set_rng


@pytest.mark.L0
@pytest.mark.parametrize("qdtype", ["BFLOAT16", "HALF"])
@pytest.mark.parametrize("kvdtype", ["FP8_E4M3", "FP8_E5M2"])
def test_mixed_graph_dtype_facts_and_admission(qdtype, kvdtype, monkeypatch):
    import cudnn
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.fwd import engines

    monkeypatch.setattr(ga, "_device_cc", lambda: (10, 0))
    io, kv = getattr(cudnn.data_type, qdtype), getattr(cudnn.data_type, kvdtype)
    graph = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    dim, stride = (2, 8, 256, 128), (256 * 8 * 128, 128, 8 * 128, 1)
    q, k, v = (graph.tensor(dim=dim, stride=stride, data_type=t) for t in (io, kv, kv))
    o, stats = graph.sdpa(q=q, k=k, v=v, generate_stats=True, attn_scale=128**-0.5, use_causal_mask=True)
    o.set_output(True).set_dim(dim).set_stride(stride).set_data_type(io)
    stats.set_output(True).set_dim((2, 8, 256, 1)).set_stride((2048, 256, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    spec = next(s for s in engines.ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm100")
    facts, reason = engines.analyze_for(spec, graph, engines.SdpaFwdKnobs(cga=1))
    assert reason is None, reason
    assert facts.kv_dtype == kv and facts.uniform_qo_dtype and not facts.uniform_dtype
    assert not facts.is_fp8 and not facts.is_mxfp8
    for changed, expected in ((v, io), (o, cudnn.data_type.FLOAT)):
        original = changed.get_data_type()
        changed.set_data_type(expected)
        _, reason = engines.analyze_for(spec, graph, engines.SdpaFwdKnobs(cga=1))
        assert reason and "dtypes" in reason
        changed.set_data_type(original)
    # Eight half elements align Q, but only sixteen FP8 elements align K.
    k.set_stride((256 * 8 * 136, 136, 8 * 136, 1))
    _, reason = engines.analyze_for(spec, graph, engines.SdpaFwdKnobs(cga=1))
    assert reason and "without a copy" in reason


_CASES = [
    (q, kv, cga, layout, 1, "graph")
    for q, kv, cga, layout in itertools.product((torch.bfloat16, torch.float16), (torch.float8_e4m3fn, torch.float8_e5m2), (1, 2), ("dense", "paged"))
]
_CASES += [(q, torch.float8_e4m3fn, cga, "paged", 3, "graph") for q, cga in itertools.product((torch.bfloat16, torch.float16), (1, 2))]
_CASES += [(q, kv, 1, "dense", 1, "standalone") for q, kv in itertools.product((torch.bfloat16, torch.float16), (torch.float8_e4m3fn, torch.float8_e5m2))]


@pytest.mark.L0
@requires_dsl
@pytest.mark.skipif(_SM not in (100, 103), reason="native SM100/SM103 mixed KV")
@pytest.mark.parametrize("qdtype,kvdtype,cga,layout,splits,entry", _CASES)
@torch_fork_set_rng(seed=49382)
def test_mixed_dense_paged_rebind_capture(qdtype, kvdtype, cga, layout, splits, entry, monkeypatch, cudnn_handle):
    import cudnn
    from cudnn.engines.manifest import family_for
    from cudnn.sdpa.fwd import api_dsl

    loaded = []
    original = api_dsl._load_sm100_kernel_module

    def load(*args, **kwargs):
        module = original(*args, **kwargs)
        loaded.append(module)
        return module

    monkeypatch.setattr(api_dsl, "_load_sm100_kernel_module", load)
    paged = layout == "paged"
    hnd = kvdtype == torch.float8_e5m2
    page = 128 if hnd else 8
    b, sq, sk, hq, hk, d = (9, 17, 384 if hnd else 264, 16, 4, 128) if paged else (2, 193, 257, 8, 2, 128)
    blocks = sk // page if paged else 0
    pages = b * blocks
    initial_lengths = [257, 0, 145, 1, 129, 33, 257, 17, 65] if paged else [sk] * b
    changed_lengths = [33, 129, 0, 257, 1, 65, 17, 145, 257] if paged else initial_lengths

    def buffers():
        q = (torch.randn(b, sq, hq, d, device="cuda") * 0.5).to(qdtype).transpose(1, 2)
        shape = (pages, hk, page, d) if paged and hnd else (pages, page, hk, d) if paged else (b, sk, hk, d)
        kv = [(torch.randn(shape, device="cuda") * 0.5).to(kvdtype) for _ in range(2)]
        if not (paged and hnd):
            kv = [t.transpose(1, 2) for t in kv]
        result = dict(q=q, k=kv[0], v=kv[1], o=torch.empty_like(q), stats=torch.empty(b, hq, sq, 1, device="cuda"))
        if paged:
            result.update(
                ql=torch.full((b, 1, 1, 1), sq, device="cuda", dtype=torch.int32),
                kl=torch.tensor(initial_lengths, device="cuda", dtype=torch.int32).reshape(b, 1, 1, 1),
                kt=torch.randperm(pages, device="cuda", dtype=torch.int32).reshape(b, 1, blocks, 1),
                vt=torch.randperm(pages, device="cuda", dtype=torch.int32).reshape(b, 1, blocks, 1),
                sink=torch.linspace(-1, 1, hq, device="cuda").reshape(1, hq, 1, 1),
            )
        return result

    first = buffers()
    handle, saved_stream = cudnn_handle, cudnn.get_stream(cudnn_handle)
    captures = []
    try:
        if entry == "graph":
            io = cudnn.data_type.BFLOAT16 if qdtype == torch.bfloat16 else cudnn.data_type.HALF
            graph = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, handle=handle)
            nodes = {n: graph.tensor_like(t) for n, t in first.items() if n not in ("o", "stats")}
            extra = (
                dict(
                    use_padding_mask=True,
                    seq_len_q=nodes["ql"],
                    seq_len_kv=nodes["kl"],
                    paged_attention_k_table=nodes["kt"],
                    paged_attention_v_table=nodes["vt"],
                    paged_attention_max_seq_len_kv=sk,
                    sink_token=nodes["sink"],
                )
                if paged
                else {}
            )
            nodes["o"], nodes["stats"] = graph.sdpa(
                q=nodes["q"],
                k=nodes["k"],
                v=nodes["v"],
                generate_stats=True,
                attn_scale=d**-0.5,
                use_causal_mask=not paged,
                use_causal_mask_bottom_right=paged,
                **extra,
            )
            for name in ("o", "stats"):
                nodes[name].set_output(True).set_dim(first[name].shape).set_stride(first[name].stride()).set_data_type(
                    cudnn.data_type.FLOAT if name == "stats" else io
                )
            graph.validate()
            graph.build_operation_graph()
            graph.create_execution_plan(
                family_for(graph).offered_ids()["sdpa_fwd_prefill_sm100"],
                {cudnn.knob_type.TILE_CGA_M: cga, cudnn.knob_type.PACK_GQA: 1, cudnn.knob_type.SPLIT_KV: splits},
            )
            graph.select_plan(0)
            graph.check_support()
            graph.build_plans()
            workspace = torch.empty(max(1, graph.get_workspace_size()), dtype=torch.uint8, device="cuda")

            def execute(t):
                cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)
                graph.execute({nodes[n]: v for n, v in t.items()}, workspace, handle=handle)

        else:
            api = api_dsl.SdpaFwdDslSm100(
                sample_q=first["q"],
                sample_k=first["k"],
                sample_v=first["v"],
                sample_o=first["o"],
                sample_lse=first["stats"],
                is_causal=True,
                cga=cga,
                pack_gqa=True,
            )
            api.check_support()
            api.compile()
            workspace = torch.empty(max(1, api.scratch_workspace_bytes()), dtype=torch.uint8, device="cuda")

            def execute(t):
                api.execute(t["q"], t["k"], t["v"], t["o"], t["stats"], workspace=workspace)

        assert loaded and loaded[-1].MIXED_FP8_KV
        assert loaded[-1].__file__.endswith("sm100/decode_d128_f16.py" if cga == 1 else "sm100/prefill_d128_f16.py")
        assert loaded[-1].CFG.CTA_MMA == cga and loaded[-1].KV_STORAGE_DTYPE.width == 8

        def verify(t, lengths):
            for batch, nk in enumerate(lengths):
                q = t["q"][batch].double()
                if paged:
                    k = t["k"][t["kt"][batch].flatten().long()].transpose(0, 1).reshape(hk, sk, d)[:, :nk].double()
                    v = t["v"][t["vt"][batch].flatten().long()].transpose(0, 1).reshape(hk, sk, d)[:, :nk].double()
                else:
                    k, v = t["k"][batch].double(), t["v"][batch].double()
                k, v = k.repeat_interleave(hq // hk, 0), v.repeat_interleave(hq // hk, 0)
                scores = q @ k.transpose(-1, -2) / math.sqrt(d)
                bound = torch.arange(sq, device="cuda")[:, None] + (nk - sq if paged else 0)
                scores.masked_fill_(torch.arange(nk, device="cuda")[None, :] > bound, -torch.inf)
                if paged:
                    scores = torch.cat([scores, t["sink"].reshape(hq, 1, 1).double().expand(hq, sq, 1)], -1)
                expected = scores.softmax(-1)[..., :nk] @ v
                torch.testing.assert_close(t["o"][batch].double(), expected, atol=0.003, rtol=0.02)
                torch.testing.assert_close(t["stats"][batch].double(), scores.logsumexp(-1, keepdim=True), atol=0.0006, rtol=0)

        def poison(t):
            t["o"].fill_(float("nan"))
            t["stats"].fill_(float("nan"))
            workspace.fill_(255)

        # A half K buffer cannot bind to a plan compiled for FP8 storage.
        poison(first)
        bad = dict(first, k=first["k"].to(qdtype))
        with pytest.raises((ValueError, RuntimeError), match="dtype"):
            execute(bad)
        assert torch.isnan(first["o"]).all() and torch.isnan(first["stats"]).all()

        for t in (first, buffers()):
            poison(t)
            execute(t)
            verify(t, initial_lengths)
            cap = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cap):
                execute(t)
            captures.append((cap, t))
        for cap, t in reversed(captures):
            poison(t)
            cap.replay()
            verify(t, initial_lengths)
            t["q"].mul_(0.5)
            if paged:
                t["kt"].copy_(t["kt"].flip(2))
                t["vt"].copy_(t["vt"].flip(2))
                t["kl"].copy_(torch.tensor(changed_lengths, device="cuda", dtype=torch.int32).reshape_as(t["kl"]))
            poison(t)
            cap.replay()
            verify(t, changed_lengths)
    finally:
        try:
            torch.cuda.synchronize()
            for cap, _ in captures:
                cap.reset()
        finally:
            cudnn.set_stream(handle, saved_stream)


@pytest.mark.L0
@requires_dsl
@pytest.mark.skipif(_SM not in (100, 103), reason="native SM100/SM103 mixed KV")
@pytest.mark.parametrize(
    "qdtype,kvdtype,paged,splits",
    [(q, kv, paged, 1) for q, kv, paged in itertools.product(("bfloat16", "float16"), ("float8_e4m3fn", "float8_e5m2"), (False, True))]
    + [(q, kv, paged, split) for q, kv, paged, split in itertools.product(("bfloat16", "float16"), ("float8_e4m3fn", "float8_e5m2"), (False, True), (2, 5))],
)
def test_mixed_physical_thd_origins(qdtype, kvdtype, paged, splits, monkeypatch):
    from cudnn.sdpa.fwd import api_dsl
    from test_sdpa_fwd_paged_thd_origins import test_paged_thd_origins_rebind_and_capture as paged_case
    from test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture as thd_case

    loaded = []
    original = api_dsl._load_sm100_kernel_module

    def load(*args, **kwargs):
        module = original(*args, **kwargs)
        loaded.append(module)
        return module

    monkeypatch.setattr(api_dsl, "_load_sm100_kernel_module", load)
    stats = "head" if kvdtype == "float8_e5m2" else "token"
    if paged:
        paged_case(
            128,
            qdtype,
            "hnd" if stats == "head" else "nhd",
            stats,
            1 if splits > 1 else 2,
            splits,
            True,
            page_size=8 if stats == "token" else 128,
            # Preserve the half family's existing SM100 THD split domain.
            # The unsplit prefill body supports sinks; the packed split host
            # does not admit them for this architecture and head dimension.
            sink=0.25 if splits == 1 else None,
            kv_dtype_name=kvdtype,
        )
    else:
        thd_case(128, qdtype, stats, True, splits == 1, splits, kv_dtype_name=kvdtype)
    assert loaded and all(m.MIXED_FP8_KV for m in loaded)


@pytest.mark.L0
@pytest.mark.parametrize("q,kv", itertools.product((2, 3), (0, 1)))
def test_mixed_decode_resource_domain(q, kv):
    from dataclasses import replace
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams, make_cfg_d128_decode, _d128_smem_bytes
    from cudnn.sdpa.fwd.config_sm107 import make_cfg_d128 as rubin_cfg

    params = TemplateParams(dtype_qkv=q, cta_mma=1)
    uniform, _ = make_cfg_d128_decode(params)
    mixed, _ = make_cfg_d128_decode(replace(params, dtype_kv=kv))
    assert uniform.STAGES_KV == 3 and _d128_smem_bytes(uniform) == 224 * 1024
    assert mixed.STAGES_KV == 2 and _d128_smem_bytes(mixed) == 176 * 1024
    assert mixed.DTYPE_KV == kv and mixed.DTYPE_QKV == mixed.DTYPE_O == q
    with pytest.raises(ValueError, match="independent FP8 KV"):
        make_cfg_d128_decode(replace(params, dtype_kv=kv, single_q_head_dim=192, thd_varlen=True, seq_kv_lens_present=True))
    with pytest.raises(ValueError, match="independent FP8 KV"):
        rubin_cfg(replace(params, dtype_kv=kv))


@pytest.mark.L0
@requires_dsl
@pytest.mark.skipif(_SM not in (100, 103), reason="native SM100/SM103 mixed KV")
@pytest.mark.parametrize(
    "qdtype,kvdtype",
    list(itertools.product((torch.bfloat16, torch.float16), (torch.float8_e4m3fn, torch.float8_e5m2)))
    + [(torch.bfloat16, torch.bfloat16), (torch.float16, torch.float16)],
)
@pytest.mark.parametrize("prefix_q", [False, True])
@torch_fork_set_rng(seed=49383)
def test_half_q_single_q_ragged_decode(qdtype, kvdtype, prefix_q, cudnn_handle):
    import cudnn
    from cudnn.engines.manifest import family_for
    from cudnn.sdpa.fwd.prepared import PreparedDenseLaunch

    b, h, kh, d, page, blocks, capacity = 4, 16, 4, 128, 16, 32, 9
    pages = b * blocks
    hnd = kvdtype in (torch.float8_e5m2, torch.float16)
    odtype = torch.int64 if hnd else torch.int32
    io = cudnn.data_type.BFLOAT16 if qdtype == torch.bfloat16 else cudnn.data_type.HALF

    def buffers():
        values = dict(
            q=torch.empty(capacity, h, d, device="cuda", dtype=qdtype),
            o=torch.empty(capacity, h, d, device="cuda", dtype=qdtype),
            stats=torch.empty(capacity, h, device="cuda"),
        )
        for name in ("k", "v"):
            shape = (pages, kh, page, d) if hnd else (pages, page, kh, d)
            pool = (torch.randn(shape, device="cuda") * 0.5).to(kvdtype)
            values[name] = pool if hnd else pool.transpose(1, 2)
        for name in ("kt", "vt"):
            values[name] = torch.randperm(pages, dtype=torch.int32).to("cuda").view(b, 1, blocks, 1)
        for name in ("ql", "kl"):
            values[name] = torch.empty(b + int(prefix_q and name == "ql"), 1, 1, 1, dtype=torch.int32, device="cuda")
        for name in ("q", "o", "stats"):
            values["ro_" + name] = torch.empty(b + 1, 1, 1, 1, dtype=odtype, device="cuda")
        return values

    def fill(values, changed=False, empty=False):
        ql = [0, 1, 1, 1] if changed else [1, 0, 1, 1]
        if empty:
            ql = [0] * b
        kl = [17, 257, 65, 0] if changed else [257, 65, 0, 17]
        starts = {"q": [1, 3, 4, 6], "o": [0, 2, 4, 5], "stats": [1, 3, 5, 6]}
        if changed:
            starts = {name: [x + 1 for x in rows] for name, rows in starts.items()}
            values["kt"].copy_(values["kt"].flip(2))
            values["vt"].copy_(values["vt"].flip(2))
        values["q"].fill_(float("nan"))
        for start, length in zip(starts["q"], ql):
            if length:
                values["q"][start].copy_(torch.randn(h, d, dtype=qdtype, device="cuda") * 0.5)
        for name, lens in (("ql", ql), ("kl", kl)):
            data = torch.tensor(lens, device="cuda", dtype=torch.int32)
            if prefix_q and name == "ql":
                data = torch.cat((data.new_zeros(1), data.cumsum(0).to(torch.int32)))
            values[name].copy_(data.view(-1, 1, 1, 1))
        for name in starts:
            scale = h if name == "stats" else h * d
            values["ro_" + name].copy_(torch.tensor([x * scale for x in starts[name] + [capacity]], device="cuda", dtype=odtype).view(b + 1, 1, 1, 1))
        return ql, kl, starts

    first = buffers()
    handle, saved_stream = cudnn_handle, cudnn.get_stream(cudnn_handle)
    captures = []
    try:
        graph = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, handle=handle)
        nodes = {name: graph.tensor_like(t) for name, t in first.items() if name not in ("q", "o", "stats")}
        nodes["q"] = graph.tensor(dim=(b, h, 1, d), stride=(h * d, d, h * d, 1), data_type=io).set_ragged_offset(nodes["ro_q"])
        nodes["o"], nodes["stats"] = graph.sdpa(
            q=nodes["q"],
            k=nodes["k"],
            v=nodes["v"],
            generate_stats=True,
            attn_scale=d**-0.5,
            use_padding_mask=True,
            **({"cu_seq_len_q": nodes["ql"]} if prefix_q else {"seq_len_q": nodes["ql"]}),
            seq_len_kv=nodes["kl"],
            paged_attention_k_table=nodes["kt"],
            paged_attention_v_table=nodes["vt"],
            paged_attention_max_seq_len_kv=page * blocks,
        )
        nodes["o"].set_output(True).set_dim((b, h, 1, d)).set_stride((h * d, d, h * d, 1)).set_data_type(io).set_ragged_offset(nodes["ro_o"])
        nodes["stats"].set_output(True).set_dim((b, h, 1, 1)).set_stride((h, 1, h, 1)).set_data_type(cudnn.data_type.FLOAT).set_ragged_offset(nodes["ro_stats"])
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plan(
            family_for(graph).offered_ids()["sdpa_fwd_prefill_sm100"],
            {
                cudnn.knob_type.TILE_CGA_M: 1,
                cudnn.knob_type.PACK_GQA: 1,
                cudnn.knob_type.SPLIT_KV: 5 if hnd else 2,
            },
        )
        graph.select_plan(0)
        graph.check_support()
        graph.build_plans()
        compiled = graph._compiled_plans[graph._plan_index]
        assert compiled._compiled.kernel_template == "decode_d128_f16"
        assert isinstance(compiled._prepared, PreparedDenseLaunch)
        assert compiled._prepared.spec.ragged_q_lens == (2 if prefix_q else 1)
        workspace = torch.empty(max(1, graph.get_workspace_size()), dtype=torch.uint8, device="cuda")

        def poison(values):
            values["o"].fill_(-17)
            values["stats"].fill_(-29)
            workspace.fill_(255)

        def execute(values):
            cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)
            graph.execute({nodes[name]: t for name, t in values.items()}, workspace, handle=handle)

        def verify(values, state):
            ql, kl, starts = state
            data = {name: values[name].double().cpu() for name in ("q", "k", "v", "o", "stats")}
            tables = {name: values[name].long().cpu().reshape(b, blocks) for name in ("kt", "vt")}
            o_written, st_written = torch.zeros(capacity, dtype=torch.bool), torch.zeros(capacity, dtype=torch.bool)
            for batch, (nq, nk) in enumerate(zip(ql, kl)):
                if not nq:
                    continue
                q = data["q"][starts["q"][batch]][:, None, :]
                kv = [
                    data[name][tables[table][batch]].transpose(0, 1).reshape(kh, page * blocks, d)[:, :nk].repeat_interleave(h // kh, 0)
                    for name, table in (("k", "kt"), ("v", "vt"))
                ]
                scores = (q @ kv[0].transpose(-1, -2)) * d**-0.5
                expected = (scores.softmax(-1) @ kv[1])[:, 0]
                expected_stats = scores.logsumexp(-1)[:, 0]
                torch.testing.assert_close(data["o"][starts["o"][batch]], expected, atol=0.003, rtol=0.02)
                torch.testing.assert_close(data["stats"][starts["stats"][batch]], expected_stats, atol=0.0006, rtol=0)
                o_written[starts["o"][batch]] = st_written[starts["stats"][batch]] = True
            assert torch.all(data["o"][~o_written] == -17)
            assert torch.all(data["stats"][~st_written] == -29)

        poison(first)
        invalid = dict(first, ql=torch.zeros(b if prefix_q else b - 1, 1, 1, 1, device="cuda", dtype=torch.int32))
        with pytest.raises(ValueError, match="seq_q_lens"):
            execute(invalid)
        assert torch.all(first["o"] == -17) and torch.all(first["stats"] == -29)

        for values in (first, buffers()):
            state = fill(values)
            poison(values)
            torch.cuda.set_sync_debug_mode("error")
            try:
                execute(values)
            finally:
                torch.cuda.set_sync_debug_mode("default")
            verify(values, state)
            cap = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cap):
                execute(values)
            captures.append((cap, values))
        for cap, values in reversed(captures):
            for empty in (False, True):
                state = fill(values, changed=True, empty=empty)
                poison(values)
                cap.replay()
                verify(values, state)
    finally:
        try:
            torch.cuda.synchronize()
            for cap, _ in captures:
                cap.reset()
        finally:
            cudnn.set_stream(handle, saved_stream)


@pytest.mark.L0
@requires_dsl
@pytest.mark.skipif(_SM not in (100, 103), reason="native SM100/SM103 mixed KV")
@pytest.mark.parametrize("qdtype,kvdtype", itertools.product((torch.bfloat16, torch.float16), (torch.float8_e4m3fn, torch.float8_e5m2)))
@torch_fork_set_rng(seed=49384)
def test_mixed_compact_thd_split_standalone(qdtype, kvdtype):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    b, h, kh, sq, sk, d = 3, 8, 2, 256, 512, 128

    def buffers():
        result = {
            name: torch.empty(b, length, heads, d, device="cuda", dtype=dtype).transpose(1, 2)
            for name, length, heads, dtype in (("q", sq, h, qdtype), ("k", sk, kh, kvdtype), ("v", sk, kh, kvdtype), ("o", sq, h, qdtype))
        }
        result["stats"] = torch.empty(b, sq, h, device="cuda").transpose(1, 2)
        result["ql"] = torch.empty(b, device="cuda", dtype=torch.int32)
        result["kl"] = torch.empty(b, device="cuda", dtype=torch.int32)
        return result

    def flat(t):
        return t.transpose(1, 2).view(-1, *t.transpose(1, 2).shape[2:])

    def fill(t, changed=False, empty=False):
        ql = [33, 65, 0] if changed else [129, 0, 65]
        kl = [0, 129, 17] if changed else [257, 0, 145]
        if empty:
            ql = [0] * b
        for name, lengths in (("q", ql), ("k", kl), ("v", kl)):
            view = flat(t[name])
            view.fill_(float("nan"))
            view[: sum(lengths)].copy_(torch.randn(sum(lengths), view.shape[1], d, device="cuda", dtype=qdtype) * 0.5)
        for name, lengths in (("ql", ql), ("kl", kl)):
            t[name].copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        return ql, kl

    first = buffers()
    api = SdpaFwdDslSm100(
        sample_q=first["q"],
        sample_k=first["k"],
        sample_v=first["v"],
        sample_o=first["o"],
        sample_lse=first["stats"],
        thd=True,
        cga=1,
        pack_gqa=False,
        split_kv=2 if kvdtype == torch.float8_e4m3fn else 5,
    )
    api.check_support()
    api.compile()
    assert api.packed_thd_split and api._k_mod.MIXED_FP8_KV
    assert getattr(api, "_thd_origin_specs", None) is None
    workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    captures = []

    def execute(t):
        api.execute(t["q"], t["k"], t["v"], t["o"], flat(t["stats"]), seq_q_lens=t["ql"], seq_kv_lens=t["kl"], workspace=workspace)

    def poison(t):
        t["o"].fill_(-17)
        t["stats"].fill_(-29)
        workspace.fill_(255)

    def verify(t, state):
        data = {name: flat(t[name]).double().cpu() for name in ("q", "k", "v", "o", "stats")}
        oq = ok = 0
        for nq, nk in zip(*state):
            q = data["q"][oq : oq + nq].transpose(0, 1)
            k, v = (data[name][ok : ok + nk].repeat_interleave(h // kh, 1).transpose(0, 1) for name in ("k", "v"))
            scores = (q @ k.transpose(-1, -2)) * d**-0.5
            expected = (scores.softmax(-1) @ v).transpose(0, 1)
            torch.testing.assert_close(data["o"][oq : oq + nq], expected, atol=0.003, rtol=0.02)
            torch.testing.assert_close(data["stats"][oq : oq + nq], scores.logsumexp(-1).T, atol=0.0006, rtol=0)
            oq += nq
            ok += nk
        assert torch.all(data["o"][oq:] == -17)
        assert torch.all(data["stats"][oq:] == -29)

    try:
        for t in (first, buffers()):
            state = fill(t)
            poison(t)
            execute(t)
            verify(t, state)
            cap = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cap):
                execute(t)
            captures.append((cap, t))
        for cap, t in reversed(captures):
            for empty in (False, True):
                state = fill(t, changed=True, empty=empty)
                poison(t)
                cap.replay()
                verify(t, state)
    finally:
        torch.cuda.synchronize()
        for cap, _ in captures:
            cap.reset()


@pytest.mark.L0
@requires_dsl
@pytest.mark.skipif(_SM not in (100, 103), reason="native SM100/SM103 mixed KV")
@pytest.mark.parametrize("qdtype,kvdtype", itertools.product((torch.bfloat16, torch.float16), (torch.float8_e4m3fn, torch.float8_e5m2)))
@pytest.mark.parametrize("swizzle", [64, 128])
def test_mixed_conversion_all_encodings(qdtype, kvdtype, swizzle):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cudnn.frost.tile_dsl.handles import SmemTile
    from cudnn.sdpa.fwd.kernels._mixed_kv import convert_fp8_smem_tile

    @cute.kernel
    def convert_kernel(inp: cute.Tensor, out: cute.Tensor, swz: cutlass.Constexpr):
        lane = cute.arch.thread_idx()[0]
        raw = cutlass.Array(inp.element_type, 16384, alignment=1024, space=cutlass.AddressSpace.smem)
        dest = cutlass.Array(out.element_type, 16384, alignment=1024, space=cutlass.AddressSpace.smem)
        for i in cutlass.range(lane, 16384, 32):
            raw[i] = inp[i]
        cute.arch.barrier()
        width = cutlass.const_expr(swz // 2)
        src = SmemTile(raw, 16384, 16, 1024, 0, tma_granu_elems=width, tma_subtile_stride_elems=width * 128)
        dst = SmemTile(dest, 16384, 16, 1024, 0, tma_granu_elems=width, tma_subtile_stride_elems=width * 128)
        convert_fp8_smem_tile(src, dst, swz, out.element_type)
        cute.arch.barrier()
        for i in cutlass.range(lane, 16384, 32):
            out[i] = dest[i]

    @cute.jit
    def convert_host(inp: cute.Tensor, out: cute.Tensor, swz: cutlass.Constexpr):
        convert_kernel(inp, out, swz).launch(grid=(1, 1, 1), block=(32, 1, 1))

    # Enumerate every input code with different row positions. The reference
    # addresses implement the byte-SMEM and half-SMEM layouts independently.
    width = swizzle // 2
    ix = torch.arange(16384)
    raw = ((ix * 37 + (ix // width) * 13) % 256).to(torch.uint8)
    values = raw.view(kvdtype)
    row, col, outer = (ix // width) % 128, ix % width, (ix // (width * 128)) * (width * 128)
    if swizzle == 128:
        src = outer + row * width + ((col // 16) ^ ((row >> 1) & 3)) * 16 + col % 16
        dst = outer + row * width + (col ^ ((row & 7) * 8))
    else:
        src = outer + row * width + ((col // 16) ^ ((row >> 2) & 1)) * 16 + col % 16
        dst = outer + row * width + ((col // 8) ^ ((row >> 1) & 3)) * 8 + col % 8
    x = values.to("cuda")
    y = torch.empty(16384, dtype=qdtype, device="cuda")
    tx, ty = from_dlpack(x), from_dlpack(y)
    fn = cute.compile(convert_host, tx, ty, swizzle)
    fn(tx, ty)
    actual = y.cpu()
    expected = torch.empty(16384, dtype=qdtype)
    expected[dst] = values.float()[src].to(qdtype)
    finite = torch.isfinite(expected.float())
    torch.testing.assert_close(actual.view(torch.int16)[finite], expected.view(torch.int16)[finite], rtol=0, atol=0)
    assert torch.equal(torch.isnan(actual.float()), torch.isnan(expected.float()))
    assert torch.equal(torch.isinf(actual.float()), torch.isinf(expected.float()))
    non_nan = ~torch.isnan(expected.float())
    assert torch.equal(torch.signbit(actual.float())[non_nan], torch.signbit(expected.float())[non_nan])
