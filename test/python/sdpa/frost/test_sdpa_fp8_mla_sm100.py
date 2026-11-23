# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-tensor FP8 absorbed MLA with an aliased 576-column cache record."""

import math

import pytest
import torch

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

pytestmark = [pytest.mark.L0, requires_pre_rubin_blackwell, requires_dsl]


@pytest.mark.parametrize("paged,heads,sq", [(False, 128, 1), (True, 64, 8), (True, 8, 3)])
def test_fp8_mla_shared_record(paged, heads, sq):
    import cudnn
    import cudnn.sdpa  # noqa: F401

    from test_sdpa_fwd_paged_sm100 import _check_fp8_o

    fp8 = torch.float8_e5m2 if heads == 8 else torch.float8_e4m3fn
    output_dtype = torch.bfloat16 if heads == 8 else fp8
    kv, page = 256, 64
    rng = torch.Generator(device="cuda").manual_seed(2588)
    record = torch.randn(1, kv, 1, 576, generator=rng, device="cuda").to(fp8)
    q = torch.randn(1, sq, heads, 576, generator=rng, device="cuda").to(fp8).transpose(1, 2)
    # Scores vanish if the last 64 RoPE columns are dropped.
    q[..., :512] = 0
    input_type = cudnn.data_type.FP8_E5M2 if fp8 == torch.float8_e5m2 else cudnn.data_type.FP8_E4M3
    output_type = cudnn.data_type.BFLOAT16 if output_dtype == torch.bfloat16 else input_type
    graph = cudnn.pygraph(io_data_type=input_type, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    scalars = torch.ones(1, 1, 1, 1, device="cuda")
    scale = graph.tensor_like(scalars)
    pack = {scale: scalars}
    options = {}
    if paged:
        indices = torch.tensor([3, 1, 0, 2], device="cuda", dtype=torch.int32)
        table = indices.view(1, 1, -1, 1)
        pool = torch.empty(4, 1, page, 576, device="cuda", dtype=fp8)
        pool.view(torch.uint8).index_copy_(0, indices.long(), record.view(4, 1, page, 576).view(torch.uint8))
        live_q = torch.full((1, 1, 1, 1), sq, device="cuda", dtype=torch.int32)
        live_k = torch.full_like(live_q, kv)
        qt, kt, bt = (graph.tensor_like(t) for t in (live_q, live_k, table))
        pack.update({qt: live_q, kt: live_k, bt: table})
        options.update(
            use_padding_mask=True, seq_len_q=qt, seq_len_kv=kt, paged_attention_k_table=bt, paged_attention_v_table=bt, paged_attention_max_seq_len_kv=kv
        )
    else:
        pool = record.transpose(1, 2)
    value = pool[..., :512]
    assert pool.data_ptr() == value.data_ptr() and value.stride(2) == 576
    qt, kt, vt = (graph.tensor_like(t) for t in (q, pool, value))
    attn_scale = 1 / math.sqrt(192)
    out, stats, _, amax = graph.sdpa_fp8(
        q=qt,
        k=kt,
        v=vt,
        descale_q=scale,
        descale_k=scale,
        descale_v=scale,
        descale_s=scale,
        scale_s=scale,
        scale_o=scale,
        attn_scale=attn_scale,
        generate_stats=True,
        use_causal_mask_bottom_right=True,
        **options,
    )
    o = torch.empty(1, sq, heads, 512, device="cuda", dtype=output_dtype).transpose(1, 2)
    lse = torch.empty(1, heads, sq, 1, device="cuda")
    peak = torch.full_like(scalars, float("nan"))
    for tensor, buf, dtype in [(out, o, output_type), (stats, lse, cudnn.data_type.FLOAT), (amax, peak, cudnn.data_type.FLOAT)]:
        tensor.set_output(True).set_dim(buf.shape).set_stride(buf.stride()).set_data_type(dtype)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.OPENSOURCE])
    graph.check_support()
    graph.build_plans()
    assert graph.plans[graph._plan_index].engine_id == 20523
    pack.update({qt: q, kt: pool, vt: value, out: o, stats: lse, amax: peak})
    ws = torch.empty(max(graph.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    graph.execute(pack, ws)
    scores = torch.einsum("bhqd,bkd->bhqk", q.float(), record[:, :, 0].float()) * attn_scale
    mask = torch.ones(sq, kv, device="cuda", dtype=torch.bool).triu(kv - sq + 1)
    scores.masked_fill_(mask, -torch.inf)
    ref = torch.einsum("bhqk,bkd->bhqd", scores.softmax(-1), record[:, :, 0, :512].float())
    _check_fp8_o(o.float(), ref, output_dtype, str(fp8))
    torch.testing.assert_close(lse[..., 0], scores.logsumexp(-1), atol=3e-3, rtol=0)
    torch.testing.assert_close(peak.reshape(()), ref.abs().max(), atol=3e-2, rtol=0)
