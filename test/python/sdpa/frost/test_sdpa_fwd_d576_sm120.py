# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native SM120 D576/V512: direct Q, split-head swizzles and drained-KV epilogue."""

from dataclasses import replace

import pytest
import torch

from frost_test_utils import requires_blackwell_geforce, requires_dsl
from sdpa.frost import test_sdpa_fwd_dsl_sm120 as dense
from sdpa.frost import test_sdpa_fwd_paged_sm120 as paged
from sdpa.frost import test_sdpa_prepared_sm120 as prepared

pytestmark = [pytest.mark.L0, requires_dsl, requires_blackwell_geforce]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "dims,mask,packed,splits,page,sq",
    [
        ((576, 512), "none", False, 1, 16, 1),
        ((576, 512), "none", True, 3, 128, 7),
        ((520, 504), "band", False, 1, 8, 67),
        ((568, 264), "br_swa", True, 3, 256, 37),
        ((576, 512), "none", False, 1, 128, 257),
        ((576, 512), "br_swa", False, 3, 16, 129),
    ],
    ids=["plain", "split_pack", "q_envelope", "v_envelope", "reuse", "empty_splits"],
)
def test_d576_paged(dtype, dims, mask, packed, splits, page, sq, monkeypatch):
    paged.test_sm120_frost_paged_d512(dtype, dims, mask, packed, splits, page, sq, True, monkeypatch)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("dims,packed,splits", [((576, 512), True, 3), ((520, 504), False, 1)], ids=["split_pack", "envelope"])
def test_d576_mixed_paged(dtype, kv_dtype, dims, packed, splits, monkeypatch):
    paged.test_sm120_frost_paged_mixed_fp8(dtype, kv_dtype, dims, 64, 32, packed, splits, "br_swa", 129, monkeypatch)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("splits", [1, 3])
def test_d576_paged_determinism(dtype, packed, splits):
    paged.test_sm120_frost_paged_width_determinism(dtype, (576, 512), packed, tile_m=64, tile_n=32, split_kv=splits)


@pytest.mark.parametrize("dtype,kv_dtype", [(torch.float16, torch.float8_e4m3fn), (torch.bfloat16, torch.float8_e5m2)])
@pytest.mark.parametrize("splits", [1, 3])
def test_d576_mixed_paged_determinism(dtype, kv_dtype, splits):
    paged.test_sm120_frost_paged_width_determinism(dtype, (576, 512), True, tile_m=64, tile_n=32, split_kv=splits, kv_dtype=kv_dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dims,mode", [((576, 512), "plain"), ((520, 504), "band"), ((568, 264), "packed")])
def test_d576_dense_features(dtype, dims, mode, monkeypatch):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120

    monkeypatch.setattr(SdpaFwdDslSm120, "_persistent_ctas", lambda self, device: 2)
    opts = {}
    if mode == "band":
        opts.update(window_size_left=29, window_size_right=3, causal_bottom_right=True)
    elif mode == "packed":
        opts.update(is_causal=True, window_size_left=37, pack_gqa=True)
    dense._run_case(
        batch=2,
        h_q=12,
        h_kv=2,
        s_q=129,
        s_kv=193,
        head_dim=dims[0],
        head_dim_v=dims[1],
        dtype=dtype,
        q_tile=64,
        kv_tile=32,
        with_sink=True,
        check_stats=True,
        **opts,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dims", [(576, 512), (520, 264)])
def test_d576_dense_pitched_rebind(dtype, dims):
    prepared.test_sm120_prepared_rebind_strides_and_storage(*dims, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("stats_layout", ["token_major", "head_major"])
def test_d576_thd(dtype, stats_layout):
    dense._run_thd_case(
        seq_q_lens=[65, 0, 1, 129],
        seq_kv_lens=[97, 3, 0, 193],
        h_q=4,
        h_kv=2,
        head_dim=576,
        head_dim_v=512,
        dtype=dtype,
        is_causal=True,
        with_sink=True,
        check_stats=True,
        stats_layout=stats_layout,
        nan_capacity_tail=True,
    )


def test_d576_admission():
    from cudnn.sdpa.fwd.config_sm120 import D576_FLAVOR, pick_flavor, smem_bytes
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs, _sm120_spec, mismatch
    from cudnn.sdpa.graph_analyzer import analyze

    g, _, _, _, _, _ = paged._case(torch.float16, 2, 16, True, False, dims=(576, 512), tile_m=64, tile_n=32, sq=1, explicit_plan=True)
    facts, caps = analyze(g), _sm120_spec().capabilities
    for dims in ((576, 512), (520, 264), (568, 504)):
        assert pick_flavor(*dims, False) == D576_FLAVOR
        assert pick_flavor(*dims, True) is None
        assert smem_bytes(*dims, 64, 32) == 98328
        for splits in (1, 3):
            assert mismatch(caps, replace(facts, d_qk=dims[0], d_v=dims[1]), SdpaFwdKnobs(tile_m=64, tile_n=32, split_kv=splits)) is None
    for dims in ((584, 512), (576, 520), (576, 256), (516, 512)):
        assert mismatch(caps, replace(facts, d_qk=dims[0], d_v=dims[1]), SdpaFwdKnobs(tile_m=64, tile_n=32)) is not None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("thd,splits", [(False, 1), (False, 3), (True, 1)], ids=["dense", "dense_split", "thd"])
def test_d576_dense_thd_determinism(dtype, thd, splits, *, dims=(576, 512), num_q_heads=6, num_kv_heads=1, neighbor_repeats=64):
    import cudnn
    from cudnn.engines.manifest import engines_for
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs, engine_name
    from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot

    h, hk, sq, sk = num_q_heads, num_kv_heads, 17, 67
    dq, dv = dims
    rng = torch.Generator().manual_seed(576512)
    records = [tuple((torch.randn(shape, generator=rng) * 0.25).to(dtype) for shape in ((h, sq, dq), (hk, sk, dq), (hk, sk, dv))) for _ in range(3)]
    sink_cpu = torch.linspace(-1, 1, h)
    sink = sink_cpu.cuda().view(1, h, 1, 1)
    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16

    def prepare(order):
        b = len(order)
        g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        tensors, buffers, offsets, vp = {}, {}, {}, {}
        for name, heads, length, dim in (("q", h, sq, dq), ("k", hk, sk, dq), ("v", hk, sk, dv), ("o", h, sq, dv)):
            raw = torch.full((b, length, heads, dim), float("nan"), dtype=dtype, device="cuda")
            buffers[name] = raw.transpose(1, 2)
            if name != "o":
                slot = ("q", "k", "v").index(name)
                for row, record in enumerate(order):
                    buffers[name][row].copy_(records[record][slot])
                tensors[name] = g.tensor_like(buffers[name], name=name)
                vp[tensors[name]] = buffers[name]
            if thd:
                offsets[name] = (torch.arange(b + 1, device="cuda", dtype=torch.int64) * length * heads * dim).view(-1, 1, 1, 1)
                buffers["off_" + name] = offsets[name]
                off = g.tensor_like(offsets[name], name="off_" + name)
                vp[off] = offsets[name]
                if name != "o":
                    tensors[name].set_ragged_offset(off)
                else:
                    tensors["off_o"] = off
        opts = {}
        if thd:
            for name, length in (("ql", sq), ("kl", sk)):
                buffers[name] = torch.full((b, 1, 1, 1), length, dtype=torch.int32, device="cuda")
                tensors[name] = g.tensor_like(buffers[name], name=name)
                vp[tensors[name]] = buffers[name]
            opts.update(use_padding_mask=True, seq_len_q=tensors["ql"], seq_len_kv=tensors["kl"])
        st = g.tensor_like(sink, name="sink")
        vp[st] = sink
        o, lse = g.sdpa(
            q=tensors["q"],
            k=tensors["k"],
            v=tensors["v"],
            attn_scale=dq**-0.5,
            generate_stats=True,
            sink_token=st,
            use_causal_mask_bottom_right=True,
            sliding_window_length=24,
            **opts,
        )
        o.set_output(True).set_dim(buffers["o"].shape).set_stride(buffers["o"].stride())
        if thd:
            o.set_ragged_offset(tensors["off_o"])
            raw_lse = torch.full((b, sq, h), float("nan"), device="cuda")
            buffers["lse"] = raw_lse.transpose(1, 2)
            offsets["lse"] = (torch.arange(b + 1, device="cuda", dtype=torch.int64) * sq * h).view(-1, 1, 1, 1)
            buffers["off_lse"] = offsets["lse"]
            off = g.tensor_like(offsets["lse"], name="off_lse")
            vp[off] = offsets["lse"]
            lse.set_ragged_offset(off)
        else:
            buffers["lse"] = torch.full((b, h, sq), float("nan"), device="cuda")
        lse.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim([b, h, sq, 1]).set_stride([*buffers["lse"].stride(), 1])
        vp[o], vp[lse] = buffers["o"], buffers["lse"]
        g.validate()
        g.build_operation_graph()
        selected = next(engine for engine in engines_for(g) if engine.name == engine_name(arch="sm120"))
        g.create_execution_plan(selected.engine_id, SdpaFwdKnobs(tile_m=64, tile_n=32, sched_policy=0, split_kv=splits, pack_gqa=not thd))
        g.select_plan(0)
        g.check_support()
        g.build_plans()
        plan = g._compiled_plans[g._plan_index]
        assert plan._prepared is not None and plan._prepared.spec.native is not None
        ws = torch.full((max(1, g.get_workspace_size()),), 239, dtype=torch.uint8, device="cuda")

        def launch():
            g.execute(vp, ws)
            return {name: buffers[name] for name in ("o", "lse")}

        return buffers, launch

    buffers, launch = prepare([0])
    original = snapshot({name: buffers[name] for name in ("q", "k", "v")})
    expected = assert_replays(launch, repeats=4)
    assert_bitwise({name: buffers[name] for name in original}, original, "input mutation")
    qq, kk, vv = records[0]
    scores = qq.double() @ kk.double().repeat_interleave(h // hk, 0).transpose(-1, -2) * dq**-0.5
    diagonal = torch.arange(sq)[:, None] + sk - sq
    keys = torch.arange(sk)[None, :]
    scores.masked_fill_((keys > diagonal) | (keys < diagonal - 23), -torch.inf)
    ref_lse = torch.logaddexp(scores.logsumexp(-1), sink_cpu.double()[:, None])
    ref_o = (scores - ref_lse[..., None]).exp() @ vv.double().repeat_interleave(h // hk, 0)
    torch.testing.assert_close(expected["o"][0].cpu().double(), ref_o, atol=0.003, rtol=0.02)
    torch.testing.assert_close(expected["lse"][0].cpu().double(), ref_lse, atol=0.0006, rtol=0)
    for order in ([1, 0, 0, 2], [1, 2] * neighbor_repeats + [0]):
        buffers, launch = prepare(order)
        replay_inputs = {name: buffers[name].clone() for name in ("q", "k", "v")} if thd else None
        launch()
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            torch.cuda.set_sync_debug_mode("error")
            try:
                launch()
            finally:
                torch.cuda.set_sync_debug_mode("default")
        try:
            for scale in (0.5, -0.75):
                empty_neighbor = thd and scale > 0
                if thd:
                    # SM120 THD uses compact prefixes. When a neighbor becomes
                    # empty, repack subsequent sequences; fixed physical gaps
                    # are a separate contract this row does not implement.
                    for name in ("q", "k", "v"):
                        if empty_neighbor:
                            buffers[name][:-1].copy_(replay_inputs[name][1:])
                        else:
                            buffers[name].copy_(replay_inputs[name])
                    buffers["ql"][0] = 0 if empty_neighbor else sq
                    buffers["kl"][0] = 0 if empty_neighbor else sk
                    prefix = (torch.arange(len(order) + 1, device="cuda", dtype=torch.int64) - int(empty_neighbor)).clamp_min_(0)
                    for name, span in (("q", sq * h * dq), ("k", sk * hk * dq), ("v", sk * hk * dv), ("o", sq * h * dv), ("lse", sq * h)):
                        buffers["off_" + name].copy_((prefix * span).view(-1, 1, 1, 1))
                if not empty_neighbor:
                    buffers["k"][0].mul_(scale)
                    buffers["v"][0].add_(scale)
                buffers["o"].fill_(float("nan"))
                buffers["lse"].fill_(float("nan"))
                captured.replay()
                for row, record in enumerate(order):
                    if record == 0:
                        physical_row = row - int(empty_neighbor)
                        assert_bitwise(snapshot({name: buffers[name][physical_row : physical_row + 1] for name in expected}), expected, f"batch row {row}")
        finally:
            torch.cuda.synchronize()
            captured.reset()
    suffix = f"-h{h}kv{hk}" if (h, hk) != (6, 1) else ""
    assert_golden(f"sdpa-sm120-d{dq}v{dv}-{'thd' if thd else 'dense'}-split{splits}-{dtype}{suffix}", {"q": qq, "k": kk, "v": vv, "sink": sink_cpu}, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dims", [(512, 512), (576, 512)])
@pytest.mark.parametrize("group", [96, 128])
@pytest.mark.parametrize("splits", [1, 3])
def test_sm120_wide_subgroup_dense_determinism(dtype, dims, group, splits):
    test_d576_dense_thd_determinism(dtype, False, splits, dims=dims, num_q_heads=2 * group, num_kv_heads=2, neighbor_repeats=16)
