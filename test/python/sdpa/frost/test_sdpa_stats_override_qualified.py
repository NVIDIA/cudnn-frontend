# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hardware qualification for the per-plan packed HN stride-override note."""

import math

import pytest
import torch

import cudnn
from cudnn.engines.manifest import family_for
from cudnn.sdpa.fwd.engines import SdpaFwdKnobs
from test_sdpa_behavior_notes import QUALIFIED_OVERRIDE_CASES

pytestmark = pytest.mark.L0
NOTE = cudnn.behavior_note.SUPPORTS_SDPA_STATS_STRIDE_OVERRIDE


def _graph(case, dtype, layout, *, overrides=True, kv_dtype=None):
    _, owner, dq, dv, m, n, cga, split, pack, warp, paged = case
    b, h, hk, qmax, kv, page = 2, 8, 2, 16, 128, 16
    g = cudnn.pygraph(
        io_data_type=dtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, is_override_shape_enabled=overrides
    )
    refs = {}

    def tensor(name, dim, stride, dt=dtype):
        refs[name] = g.tensor(name=name, dim=dim, stride=stride, data_type=dt)
        return refs[name]

    for name in ("cu_q", "off_q", "off_o", "off_stats"):
        tensor(name, [b + 1, 1, 1, 1], [1, 1, 1, 1], cudnn.data_type.INT32)
    q = tensor("q", [b, h, qmax, dq], [qmax * h * dq, dq, h * dq, 1]).set_ragged_offset(refs["off_q"])
    if paged:
        k = tensor("k", [b * kv // page, hk, page, dq], [hk * page * dq, page * dq, dq, 1], kv_dtype or dtype)
        v = tensor("v", [b * kv // page, hk, page, dv], [hk * page * dv, page * dv, dv, 1], kv_dtype or dtype)
        table = tensor("table", [b, 1, kv // page, 1], [kv // page, kv // page, 1, 1], cudnn.data_type.INT32)
        seq = tensor("kv_lengths", [b, 1, 1, 1], [1, 1, 1, 1], cudnn.data_type.INT32)
        extra = dict(seq_len_kv=seq, paged_attention_k_table=table, paged_attention_v_table=table, paged_attention_max_seq_len_kv=kv)
    else:
        for name in ("cu_kv", "off_k", "off_v"):
            tensor(name, [b + 1, 1, 1, 1], [1, 1, 1, 1], cudnn.data_type.INT32)
        k = tensor("k", [b, hk, kv, dq], [kv * hk * dq, dq, hk * dq, 1], kv_dtype or dtype).set_ragged_offset(refs["off_k"])
        v = tensor("v", [b, hk, kv, dv], [kv * hk * dv, dv, hk * dv, 1], kv_dtype or dtype).set_ragged_offset(refs["off_v"])
        extra = dict(cu_seq_len_kv=refs["cu_kv"], max_total_seq_len_kv=b * kv)
    o, stats = g.sdpa(
        q=q,
        k=k,
        v=v,
        generate_stats=True,
        attn_scale=1 / math.sqrt(dq),
        use_causal_mask_bottom_right=True,
        use_padding_mask=True,
        cu_seq_len_q=refs["cu_q"],
        max_total_seq_len_q=b * qmax,
        **extra,
    )
    o.set_output(True).set_dim([b, h, qmax, dv]).set_stride([qmax * h * dv, dv, h * dv, 1]).set_ragged_offset(refs["off_o"])
    stats.set_output(True).set_dim([b, h, qmax, 1]).set_data_type(cudnn.data_type.FLOAT).set_ragged_offset(refs["off_stats"])
    stats.set_stride([b * qmax * h, b * qmax, 1, 1] if layout == "HN" else [qmax * h, 1, h, 1])
    refs.update(o=o, stats=stats)
    g.validate()
    g.build_operation_graph()
    name = "sdpa_fwd_" + owner
    knobs = SdpaFwdKnobs(tile_m=m, tile_n=n, cga=cga, split_kv=split, pack_gqa=bool(pack), warp_spec_cfg=warp)
    g.create_execution_plan(family_for(g).offered_ids()[name], knobs.to_public())
    g.check_support()
    g.build_plans()
    assert g.selected_engine.name == name
    return g, refs


def _buffers(case, dtype, lengths, layout, generator, kv_dtype=None):
    dq, dv, paged = case[2], case[3], case[-1]
    total, h, hk, kv, page = sum(lengths), 8, 2, 128, 16
    q = torch.randn(total, h, dq, device="cuda", dtype=dtype, generator=generator)
    k = torch.randn(2 * kv, hk, dq, device="cuda", dtype=dtype, generator=generator).to(kv_dtype or dtype)
    v = torch.randn(2 * kv, hk, dv, device="cuda", dtype=dtype, generator=generator).to(kv_dtype or dtype)
    o_storage = torch.full((total * h * dv + 32,), -1234.0, device="cuda", dtype=dtype)
    stats_storage = torch.full((total * h + 32,), -12345.0, device="cuda")
    cp = torch.tensor([0, lengths[0], total], device="cuda", dtype=torch.int32)
    bufs = dict(
        q=q,
        o=o_storage[16:-16].view(total, h, dv),
        cu_q=cp,
        off_q=cp * h * dq,
        off_o=cp * h * dv,
        off_stats=cp if layout == "HN" else cp * h,
        stats=stats_storage[16:-16].view((h, total) if layout == "HN" else (total, h)),
    )
    if paged:
        bufs.update(
            k=k.view(-1, page, hk, dq).transpose(1, 2).contiguous(),
            v=v.view(-1, page, hk, dv).transpose(1, 2).contiguous(),
            table=torch.arange(2 * kv // page, device="cuda", dtype=torch.int32).view(2, 1, -1, 1),
            kv_lengths=torch.full((2, 1, 1, 1), kv, device="cuda", dtype=torch.int32),
        )
    else:
        kp = torch.tensor([0, kv, 2 * kv], device="cuda", dtype=torch.int32)
        bufs.update(k=k, v=v, cu_kv=kp, off_k=kp * hk * dq, off_v=kp * hk * dv)
    return bufs, (k, v), (o_storage, stats_storage)


def _check(case, bufs, references, storage, lengths, layout):
    dq, h, hk, kv = case[2], 8, 2, 128
    k, v = references
    offset = 0
    for batch, length in enumerate(lengths):
        q = bufs["q"][offset : offset + length].double().transpose(0, 1)
        kb = k[batch * kv : (batch + 1) * kv].double().transpose(0, 1).repeat_interleave(h // hk, 0)
        vb = v[batch * kv : (batch + 1) * kv].double().transpose(0, 1).repeat_interleave(h // hk, 0)
        scores = q @ kb.transpose(1, 2) / math.sqrt(dq)
        mask = torch.arange(kv, device="cuda")[None, :] > torch.arange(length, device="cuda")[:, None] + kv - length
        scores.masked_fill_(mask, -float("inf"))
        expected_o = (scores.softmax(-1) @ vb).transpose(0, 1).float()
        expected_lse = scores.logsumexp(-1).float()
        torch.testing.assert_close(bufs["o"][offset : offset + length].float(), expected_o, atol=2e-2, rtol=2e-2)
        actual_lse = bufs["stats"][:, offset : offset + length] if layout == "HN" else bufs["stats"][offset : offset + length].T
        torch.testing.assert_close(actual_lse, expected_lse, atol=1e-3, rtol=1e-3)
        offset += length
    for data, sentinel in zip(storage, (-1234.0, -12345.0)):
        torch.testing.assert_close(data[:16], torch.full_like(data[:16], sentinel), atol=0, rtol=0)
        torch.testing.assert_close(data[-16:], torch.full_like(data[-16:], sentinel), atol=0, rtol=0)


@pytest.mark.parametrize("case", QUALIFIED_OVERRIDE_CASES, ids=lambda c: f"sm{c[0]}-{c[1]}-d{c[2]}-split{c[7]}-warp{c[9]}")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("layout", ["HN", "NH"])
def test_note_owner_stride_rebind_and_retained_capture(case, dtype, layout, monkeypatch):
    _check_stride_lifecycle(case, dtype, layout, monkeypatch)


@pytest.mark.parametrize("arch", [100, 103, 107])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_mixed_kv_stride_rebind_and_retained_capture(arch, dtype, kv_dtype, monkeypatch):
    owner = "sm107_d128_multi_q" if arch == 107 else "sm100_d128_multi_q"
    case = (arch, owner, 128, 128, 128, 128, 2, 1, 0, 0, False)
    _check_stride_lifecycle(case, dtype, "HN", monkeypatch, kv_dtype=kv_dtype)


def _check_stride_lifecycle(case, dtype, layout, monkeypatch, kv_dtype=None):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (case[0] // 10, case[0] % 10):
        pytest.skip("requires this implementation's target architecture")
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    dt = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    kv_dt = {torch.float8_e4m3fn: cudnn.data_type.FP8_E4M3, torch.float8_e5m2: cudnn.data_type.FP8_E5M2}.get(kv_dtype)
    graph, refs = _graph(case, dt, layout, kv_dtype=kv_dt)
    expected_notes = layout == "HN"
    assert (NOTE in graph.get_behavior_notes()) == expected_notes
    assert graph.get_behavior_notes() == graph.get_behavior_notes_for_plan_at_index(graph._plan_index)
    prepared = graph._compiled_plans[graph._plan_index]._prepared
    assert prepared.spec.native is not None
    workspace = torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(723)
    retained = []
    try:
        for lengths in ([16, 9], [9, 13], [15, 16]):
            bufs, reference, storage = _buffers(case, dtype, lengths, layout, generator, kv_dtype=kv_dtype)
            total, qmax = sum(lengths), max(lengths)
            kwargs = dict(
                override_uids=[refs["stats"].get_uid()],
                override_shapes=[[2, 8, qmax, 1]],
                override_strides=[[8 * total, total, 1, 1] if layout == "HN" else [8 * qmax, 1, 8, 1]],
            )
            vp = {refs[name]: value for name, value in bufs.items()}
            graph.execute(vp, workspace, **kwargs)
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture):
                old_debug = torch.cuda.get_sync_debug_mode()
                torch.cuda.set_sync_debug_mode("error")
                try:
                    graph.execute(vp, workspace, **kwargs)
                finally:
                    torch.cuda.set_sync_debug_mode(old_debug)
            retained.append((capture, bufs, reference, storage, lengths, vp, kwargs))
            torch.cuda.synchronize()
            _check(case, bufs, reference, storage, lengths, layout)
            assert graph._compiled_plans[graph._plan_index]._prepared is prepared
        other, _ = _graph(case, dt, layout, overrides=False, kv_dtype=kv_dt)
        assert NOTE not in other.get_behavior_notes()
        assert (NOTE in graph.get_behavior_notes()) == expected_notes
        for capture, bufs, reference, storage, lengths, _, _ in reversed(retained):
            bufs["q"].mul_(0.75)
            bufs["o"].fill_(float("nan"))
            bufs["stats"].fill_(float("nan"))
            capture.replay()
            torch.cuda.synchronize()
            _check(case, bufs, reference, storage, lengths, layout)
        if layout == "HN":
            _, bufs, _, _, lengths, vp, kwargs = retained[-1]
            before = bufs["stats"].clone()
            with pytest.raises(ValueError, match="token axis contiguous"):
                graph.execute(vp, workspace, **dict(kwargs, override_strides=[[8 * max(lengths), 1, 8, 1]]))
            with pytest.raises(ValueError, match="storage"):
                graph.execute(vp, workspace, **dict(kwargs, override_strides=[[8 * 4096, 4096, 1, 1]]))
            torch.testing.assert_close(bufs["stats"], before, atol=0, rtol=0)
    finally:
        for capture, *_ in retained:
            capture.reset()
