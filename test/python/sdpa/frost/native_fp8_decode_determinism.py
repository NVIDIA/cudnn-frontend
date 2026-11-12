# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-payload determinism for native FP8 paged decode.

Forward only; this lowering adds no backward path. The standalone API has no
separate deterministic switch. Packing and split count remain fixed across
batch transformations. AMAX is a whole-batch reduction, not a per-row output.
"""

import pytest
import torch

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot

CASES = [
    (torch.float8_e4m3fn, torch.float8_e4m3fn, 1, 8, 128, 128, 1, True, True, False),
    (torch.float8_e5m2, torch.bfloat16, 2, 4, 128, 128, 4, True, True, True),
    (torch.float8_e4m3fn, torch.float16, 1, 8, 80, 112, 4, False, False, False),
    (torch.float8_e5m2, torch.float8_e4m3fn, 2, 4, 128, 128, 1, False, True, True),
]
IDS = ["e4-e4-q1-g8-split1", "e5-bf16-q2-g4-split4-log2", "e4-f16-d80-v112-split4-no-amax-stats", "e5-e4-q2-g4-no-amax-log2"]


def _runner(dtype, out_dtype, sq, h, d, dv, split, amax, stats, log2):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    rng = torch.Generator().manual_seed(83519)
    page, blocks, pool = 16, 33, 47
    # Quantize each source once on CPU; preserve these exact bytes and scalar
    # descales when sequences are inserted, duplicated or moved.
    q = (torch.randn(5, h, sq, d, generator=rng) * 0.3).to(dtype).cuda()
    k = (torch.randn(pool, 1, page, d + 16, generator=rng) * 0.3).to(dtype).cuda()[..., :d]
    v = (torch.randn(pool, page, 1, dv + 16, generator=rng) * 0.3).to(dtype).cuda()[..., :dv].transpose(1, 2)
    kt = torch.stack([torch.randperm(pool, generator=rng)[:blocks] for _ in range(5)]).int().cuda()
    vt = torch.stack([torch.randperm(pool, generator=rng)[:blocks] for _ in range(5)]).int().cuda()
    ql = torch.tensor([sq, max(sq - 1, 0), 0, sq, 1], device="cuda", dtype=torch.int32)
    kl = torch.tensor([513, 129, 0, 1, 257], device="cuda", dtype=torch.int32)
    scales = {name: torch.tensor([value], device="cuda") for name, value in [("descale_q", 0.75), ("descale_k", 1.25), ("descale_v", 0.5), ("scale_o", 1.5)]}
    inputs = dict(q=q, k=k, v=v, kt=kt, vt=vt, ql=ql, kl=kl, **scales)
    before = snapshot(inputs)
    plans = {}

    def buffers(order):
        index = torch.tensor(order, device="cuda")
        b = len(order)
        qb = torch.empty(b, sq, h, d + 16, device="cuda", dtype=dtype)[..., :d].transpose(1, 2)
        qb.copy_(q.index_select(0, index))
        ob = torch.full((b, sq, h, dv + 16), -120, device="cuda", dtype=out_dtype)
        lseb = torch.full((b, sq, h + 4), -123.0, device="cuda")
        return dict(
            q=qb,
            o=ob[..., :dv].transpose(1, 2),
            o_storage=ob,
            lse=lseb[..., :h].transpose(1, 2),
            lse_storage=lseb,
            amax=torch.full((1,), 999.0, device="cuda"),
            kt=kt.index_select(0, index).t().contiguous().t(),
            vt=vt.index_select(0, index).t().contiguous().t(),
            ql=ql.index_select(0, index),
            kl=kl.index_select(0, index),
            **{name: value.clone() for name, value in scales.items()},
        )

    def outputs(t):
        return {name: t[name] for name in ("o",) + (("lse",) if stats else ()) + (("amax",) if amax else ())}

    def reference(t):
        assert torch.all(t["o_storage"][..., dv:] == -120), "output padding overwritten"
        assert torch.all(t["lse_storage"][..., h:] == -123), "Stats padding overwritten"
        if not amax:
            assert t["amax"].item() == 999, "undeclared AMAX written"
        if not stats:
            assert torch.all(t["lse"] == -123), "undeclared Stats written"
        expected = torch.zeros(t["o"].shape, dtype=torch.float64)
        lse = torch.full(t["lse"].shape, -torch.inf, dtype=torch.float64)
        dq, dk, ds_v, so = (t[name].item() for name in scales)
        for b, (nq, nk) in enumerate(zip(t["ql"].tolist(), t["kl"].tolist())):
            if nq == 0 or nk == 0:
                continue
            qq = t["q"][b, :, :nq].cpu().double()
            kk = k[t["kt"][b].long()].cpu().reshape(blocks * page, d)[:nk].double()
            vv = v[t["vt"][b].long()].cpu().reshape(blocks * page, dv)[:nk].double()
            scores = qq @ kk.t() * (d**-0.5 * dq * dk)
            diagonal = torch.arange(nq)[:, None] + nk - nq
            keys = torch.arange(nk)[None, :]
            scores.masked_fill_((keys > diagonal) | (keys < diagonal - 393), -torch.inf)
            weights = torch.nan_to_num(scores.softmax(-1), nan=0.0)
            expected[b, :, :nq] = weights @ vv * ds_v
            lse[b, :, :nq] = scores.logsumexp(-1)
        scaled = expected * so
        floor = (scaled - scaled.to(out_dtype).double()).abs().max().item()
        atol = max(0.04, 3 * floor) if out_dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else 0.04
        torch.testing.assert_close(t["o"].cpu().double(), scaled, atol=atol, rtol=0)
        if stats:
            torch.testing.assert_close(t["lse"].cpu().double(), lse * (1.4426950408889634 if log2 else 1), atol=0.0006, rtol=0)
        if amax:
            assert abs(t["amax"].item() - expected.abs().max().item()) <= 0.03
        return outputs(t)

    def prepare(t):
        b = len(t["q"])
        if b not in plans:
            api = SdpaFwdDslSm100(
                t["q"],
                k,
                v,
                t["o"],
                t["lse"] if stats else None,
                is_causal=True,
                causal_bottom_right=True,
                window_size_left=393,
                seq_q_lens_present=True,
                seq_kv_lens_present=True,
                pertensor_fp8=True,
                dtype_o=out_dtype,
                has_amax_o=amax,
                stats_log2=log2,
                paged_page_size=page,
                paged_max_seq_len_kv=page * blocks,
                paged_table_stride=t["kt"].stride(),
                paged_table_v_stride=t["vt"].stride(),
                cga=1,
                split_kv=split,
                pack_gqa=True,
            )
            api.check_support()
            api.compile()
            assert api._k_mod.__file__.endswith("sm100/decode_d128_fp8.py")
            assert api._k_mod.STAGES == 5 and api._k_mod.COLS == 8
            assert api._amax_folded_out == (not amax)
            plans[b] = api
        api = plans[b]
        ws = torch.empty(max(api.scratch_workspace_bytes(), 1), dtype=torch.uint8, device="cuda")

        def launch():
            api.execute(
                t["q"],
                k,
                v,
                t["o"],
                t["lse"] if stats else None,
                seq_q_lens=t["ql"],
                seq_kv_lens=t["kl"],
                block_table=t["kt"],
                block_table_v=t["vt"],
                workspace=ws,
                amax_o=t["amax"] if amax else None,
                **{name: t[name] for name in scales},
            )

        return launch, ws

    def run(order, capture=False):
        t = buffers(order)
        launch, ws = prepare(t)
        mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            launch()
        finally:
            torch.cuda.set_sync_debug_mode(mode)
        result = reference(t)
        if capture:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch()
            baseline = snapshot(result)
            saved_v = v.clone()
            try:
                v.copy_((-v.float()).to(dtype))
                t["o"].view(torch.uint8).fill_(127)
                ws.fill_(255)
                graph.replay()
                reference(t)
                v.copy_(saved_v)
                graph.replay()
                assert_bitwise(reference(t), baseline, "restored retained capture")
            finally:
                torch.cuda.synchronize()
                graph.reset()
                v.copy_(saved_v)
        assert_bitwise(inputs, before, "input mutation")
        return result

    return run, before, buffers, prepare, outputs


@pytest.mark.parametrize("dtype,out_dtype,sq,h,d,dv,split,amax,stats,log2", CASES, ids=IDS)
def test_replay_batch(dtype, out_dtype, sq, h, d, dv, split, amax, stats, log2):
    run, _, _, _, _ = _runner(dtype, out_dtype, sq, h, d, dv, split, amax, stats, log2)
    target = assert_replays(lambda: run([0]), repeats=4)
    names = [name for name in target if name != "amax"]
    for order in ([1, 0, 2], [2, 1, 0], [0, 1, 0, 2, 3], [4, 0, 3], [1] * 256 + [0] + [2] * 256):
        actual = snapshot(run(order))
        for row, sequence in enumerate(order):
            if sequence == 0:
                assert_bitwise({name: actual[name][row : row + 1] for name in names}, {name: target[name] for name in names}, f"batch row {row}")
    run([0, 1, 2, 3, 4], capture=True)


@pytest.mark.parametrize("dtype,out_dtype,sq,h,d,dv,split,amax,stats,log2", CASES, ids=IDS)
def test_golden(dtype, out_dtype, sq, h, d, dv, split, amax, stats, log2):
    run, inputs, _, _, _ = _runner(dtype, out_dtype, sq, h, d, dv, split, amax, stats, log2)
    actual = assert_replays(lambda: run([0, 1, 2]), repeats=3)
    assert_golden(f"sdpa-fp8-live8-{dtype}-{out_dtype}-q{sq}-g{h}-d{d}-v{dv}-split{split}-amax{int(amax)}-stats{int(stats)}-log2{int(log2)}", inputs, actual)


@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group("gpu_exclusive")
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3", "e5m2"])
def test_warm_refill(dtype):
    from warm_stress import StressCase, assert_warm_stress

    run, _, buffers, prepare, outputs = _runner(dtype, torch.bfloat16, 1, 8, 128, 128, 1, True, True, False)
    reference = snapshot(run([0]))
    b = 4 * torch.cuda.get_device_properties(0).multi_processor_count
    t = buffers([0] * b)
    # The stress checker poisons every output byte and requires actual,
    # contiguous kernel storage. Strided outputs/canaries are checked above.
    t["o"] = torch.empty((b, 8, 1, 128), device="cuda", dtype=torch.bfloat16)
    t["lse"] = torch.empty((b, 8, 1), device="cuda")
    launch, _ = prepare(t)
    launch()
    expected = {name: value if name == "amax" else value.expand((b,) + value.shape[1:]).contiguous() for name, value in reference.items()}
    assert_bitwise(outputs(t), expected, "stress batch invariance")
    # Five visited KV tiles issue ten alternating K/V loads, refilling the
    # five-slot ring within one CTA. Four CTA waves exercise the reuse order.
    assert_warm_stress(
        StressCase("FP8 D128 five-slot decode", launch, outputs(t), thread_blocks=b, refills_per_launch=5, reference=expected), warm=256, poisoned=256
    )
