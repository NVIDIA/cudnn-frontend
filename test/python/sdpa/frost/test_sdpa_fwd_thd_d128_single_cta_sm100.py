# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""SM100 nonpaged D128 half THD graphs with one query head per KV head run the single-CTA prefill body.

The plan put first is cga 1, unsplit; its output matches an fp32 reference for short, mixed and full-length live sequences
under the bottom-right causal, top-left causal and plain masks, with and without Stats, for BF16 and FP16; grouped heads keep
the pair tile.
"""

import math

import pytest
import torch

from test_utils import torch_fork_set_rng

from cudnn.sdpa.fwd.engines import engine_name
from frost_test_utils import _SM, _dsl_installed, requires_blackwell, select_engine

pytestmark = [requires_blackwell, pytest.mark.skipif(_SM not in (100, 103), reason="the single-CTA THD leg is the SM100 / SM103 line's")]


def _ref_thd(q_pk, k_pk, v_pk, cu_q, cu_kv, *, scale, mask, group):
    o = torch.zeros(q_pk.shape, dtype=torch.float32, device=q_pk.device)
    for b in range(len(cu_q) - 1):
        q = q_pk[cu_q[b] : cu_q[b + 1]].float().permute(1, 0, 2)
        k = k_pk[cu_kv[b] : cu_kv[b + 1]].float().permute(1, 0, 2).repeat_interleave(group, dim=0)
        v = v_pk[cu_kv[b] : cu_kv[b + 1]].float().permute(1, 0, 2).repeat_interleave(group, dim=0)
        s = torch.matmul(q, k.transpose(-1, -2)) * scale
        if mask != "none":
            sq, skv = q.shape[1], k.shape[1]
            i = torch.arange(sq, device=q.device).view(-1, 1)
            j = torch.arange(skv, device=q.device).view(1, -1)
            s = s.masked_fill(j > i + ((skv - sq) if mask == "bottom_right" else 0), float("-inf"))
        o[cu_q[b] : cu_q[b + 1]] = torch.matmul(torch.softmax(s, dim=-1), v).permute(1, 0, 2)
    return o


def _run(q_lens, kv_lens, *, h_q, h_kv, s_max_q, s_max_kv, mask="bottom_right", dtype=torch.bfloat16, stats=False, d=128, gap=0):
    import cudnn

    dev = "cuda"
    B = len(q_lens)
    group = h_q // h_kv
    cu_q, cu_kv = [0], [0]
    for lq, lk in zip(q_lens, kv_lens):
        cu_q.append(cu_q[-1] + lq)
        cu_kv.append(cu_kv[-1] + lk)
    scale = 1.0 / math.sqrt(d)
    q_pk = torch.randn(cu_q[-1], h_q, d, device=dev, dtype=dtype)
    k_pk = torch.randn(cu_kv[-1], h_kv, d, device=dev, dtype=dtype)
    v_pk = torch.randn(cu_kv[-1], h_kv, d, device=dev, dtype=dtype)

    # Storage rows: sequence b starts at cu[b] + b * gap; a gap > 0 leaves `gap` rows of NaN canary between sequences.
    row_q = [c + b * gap for b, c in enumerate(cu_q)]
    row_kv = [c + b * gap for b, c in enumerate(cu_kv)]

    def _buf(packed, lens, rows, s_max, h):
        stor = torch.full((B * s_max * h * d + gap * B * h * d,), float("nan"), device=dev, dtype=dtype).view(-1, h, d)
        start = 0
        for b, n in enumerate(lens):
            stor[rows[b] : rows[b] + n] = packed[start : start + n]
            start += n
        flat = stor.view(-1)
        return flat, flat.as_strided((B, h, s_max, d), (s_max * h * d, d, h * d, 1))

    _, q_gpu = _buf(q_pk, q_lens, row_q, s_max_q, h_q)
    _, k_gpu = _buf(k_pk, kv_lens, row_kv, s_max_kv, h_kv)
    _, v_gpu = _buf(v_pk, kv_lens, row_kv, s_max_kv, h_kv)
    o_stor = torch.full((B * s_max_q * h_q * d + gap * B * h_q * d,), float("nan"), device=dev, dtype=dtype)
    o_gpu = o_stor.as_strided((B, h_q, s_max_q, d), (s_max_q * h_q * d, d, h_q * d, 1))
    slq = torch.tensor(q_lens, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    slk = torch.tensor(kv_lens, dtype=torch.int32, device=dev).view(B, 1, 1, 1)
    ro_q = (torch.tensor(row_q, dtype=torch.int64, device=dev) * h_q * d).view(B + 1, 1, 1, 1)
    ro_kv = (torch.tensor(row_kv, dtype=torch.int64, device=dev) * h_kv * d).view(B + 1, 1, 1, 1)
    ro_s = (torch.tensor(row_q, dtype=torch.int64, device=dev) * h_q).view(B + 1, 1, 1, 1)

    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    t_q = g.tensor(dim=[B, h_q, s_max_q, d], stride=[s_max_q * h_q * d, d, h_q * d, 1], data_type=io, name="q")
    t_k = g.tensor(dim=[B, h_kv, s_max_kv, d], stride=[s_max_kv * h_kv * d, d, h_kv * d, 1], data_type=io, name="k")
    t_v = g.tensor(dim=[B, h_kv, s_max_kv, d], stride=[s_max_kv * h_kv * d, d, h_kv * d, 1], data_type=io, name="v")
    t_sq, t_sk = g.tensor_like(slq), g.tensor_like(slk)
    r_q, r_k, r_v, r_o, r_s = (g.tensor_like(ro_q), g.tensor_like(ro_kv), g.tensor_like(ro_kv), g.tensor_like(ro_q), g.tensor_like(ro_s))
    t_q.set_ragged_offset(r_q)
    t_k.set_ragged_offset(r_k)
    t_v.set_ragged_offset(r_v)
    kw = {"use_causal_mask_bottom_right": True} if mask == "bottom_right" else ({"use_causal_mask": True} if mask == "top_left" else {})
    o, lse = g.sdpa(
        name="sdpa",
        q=t_q,
        k=t_k,
        v=t_v,
        generate_stats=stats,
        attn_scale=scale,
        use_padding_mask=True,
        seq_len_q=t_sq,
        seq_len_kv=t_sk,
        max_total_seq_len_q=cu_q[-1],
        max_total_seq_len_kv=cu_kv[-1],
        **kw,
    )
    o.set_output(True).set_dim([B, h_q, s_max_q, d]).set_stride([s_max_q * h_q * d, d, h_q * d, 1])
    o.set_ragged_offset(r_o)
    vp = {t_q: q_gpu, t_k: k_gpu, t_v: v_gpu, o: o_gpu, t_sq: slq, t_sk: slk, r_q: ro_q, r_k: ro_kv, r_v: ro_kv, r_o: ro_q}
    if stats:
        stats_gpu = torch.zeros(B * s_max_q * h_q + gap * B * h_q, device=dev, dtype=torch.float32)
        lse.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim([B, h_q, s_max_q, 1]).set_stride([s_max_q * h_q, 1, h_q, 1])
        lse.set_ragged_offset(r_s)
        vp[lse] = stats_gpu.as_strided((B, h_q, s_max_q, 1), (s_max_q * h_q, 1, h_q, 1))
        vp[r_s] = ro_s
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    plan = select_engine(g, engine_name(arch="sm100"))
    g.check_support()
    g.build_plans()
    ws = torch.zeros(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    g.execute(vp, ws)
    torch.cuda.synchronize()
    o_ref = _ref_thd(q_pk, k_pk, v_pk, cu_q, cu_kv, scale=scale, mask=mask, group=group)
    tol = dict(atol=5e-2, rtol=3e-2) if dtype == torch.bfloat16 else dict(atol=2e-2, rtol=2e-2)
    o_rows = o_stor.view(-1, h_q, d)
    for b, n in enumerate(q_lens):
        torch.testing.assert_close(o_rows[row_q[b] : row_q[b] + n].float(), o_ref[cu_q[b] : cu_q[b + 1]], **tol)
        if gap:
            # The canary rows between sequences stay NaN: no store lands outside the declared origins.
            assert torch.isnan(o_rows[row_q[b] + n : row_q[b] + n + gap]).all(), f"sequence {b}: a store hit the gap after it"
    return plan.knobs


# Live lengths and the declared per-sequence extent (equal for Q and KV, as a ragged cache declaration is).
_CASES = {
    "chunks": ([256] * 4, [4096] * 4, 4096),
    "mixed": ([64, 1024, 256, 8], [4096, 1024, 2048, 4096], 4096),
    "full": ([1024] * 4, [1024] * 4, 1024),
    "one_row": ([1, 300], [2048, 300], 2048),
}


@pytest.mark.L0
@pytest.mark.parametrize("case", sorted(_CASES))
@pytest.mark.parametrize("mask", ["bottom_right", "top_left", "none"])
@torch_fork_set_rng(seed=0)
def test_single_cta_body_leads_and_matches(case, mask):
    if not _dsl_installed():
        pytest.skip("cutlass/dsl not installed")
    q_lens, kv_lens, s_max = _CASES[case]
    knobs = _run(q_lens, kv_lens, h_q=8, h_kv=8, s_max_q=s_max, s_max_kv=s_max, mask=mask)
    assert knobs.cga == 1 and knobs.split_kv in (None, 1) and not knobs.pack_gqa, knobs


@pytest.mark.L0
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@torch_fork_set_rng(seed=0)
def test_single_cta_body_with_stats(dtype):
    if not _dsl_installed():
        pytest.skip("cutlass/dsl not installed")
    q_lens, kv_lens, s_max = _CASES["mixed"]
    knobs = _run(q_lens, kv_lens, h_q=8, h_kv=8, s_max_q=s_max, s_max_kv=s_max, dtype=dtype, stats=True)
    assert knobs.cga == 1 and knobs.split_kv in (None, 1), knobs


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_heads_keep_the_pair_tile():
    if not _dsl_installed():
        pytest.skip("cutlass/dsl not installed")
    q_lens, kv_lens, s_max = _CASES["chunks"]
    knobs = _run(q_lens, kv_lens, h_q=16, h_kv=4, s_max_q=s_max, s_max_kv=s_max)
    assert knobs.cga == 2 and knobs.pack_gqa, knobs


@pytest.mark.L0
@pytest.mark.parametrize("mask", ["bottom_right", "none"])
@torch_fork_set_rng(seed=0)
def test_single_cta_body_reads_the_declared_origins_across_gaps(mask):
    """Ragged offsets with storage gaps between sequences: the single-CTA body reads Q/K/V at the declared origins and
    stores O there, leaving the gap rows untouched."""
    if not _dsl_installed():
        pytest.skip("cutlass/dsl not installed")
    q_lens, kv_lens, s_max = _CASES["mixed"]
    knobs = _run(q_lens, kv_lens, h_q=8, h_kv=8, s_max_q=s_max, s_max_kv=s_max, mask=mask, gap=24)
    assert knobs.cga == 1 and knobs.split_kv in (None, 1), knobs
