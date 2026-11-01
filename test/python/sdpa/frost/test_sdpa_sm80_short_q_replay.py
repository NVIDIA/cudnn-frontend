# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native short-Q CTA geometry across masks, tile boundaries and fresh bindings."""

import math

import pytest
import torch

from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm80
from frost_test_utils import _SM, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM != 80, reason="requires native SM80")]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "sq,tile_m,mask,d,dv",
    [
        (8, None, "none", 128, 128),
        (64, None, "causal", 128, 128),
        (8, None, "br", 128, 128),
        (64, None, "swa", 128, 128),
        (8, None, "br_swa", 128, 128),
        (65, None, "none", 128, 128),
        (129, 64, "causal", 128, 128),
        (64, 128, "none", 128, 128),
        (129, 64, "causal", 96, 96),
        (96, None, "br", 128, 128),
        (128, None, "swa", 128, 128),
        (129, None, "none", 128, 128),
        (8, None, "none", 192, 128),
        (64, None, "br_swa", 192, 128),
        (65, None, "causal", 192, 128),
        (64, 128, "none", 192, 128),
        (129, 64, "causal", 192, 128),
    ],
)
def test_short_q_masked_rebind_replay(sq, tile_m, mask, d, dv, dtype):
    torch.manual_seed(721)
    b, h, hk, sk = 2, 8, 2, 257
    causal = mask in ("causal", "br", "br_swa")
    br = mask in ("br", "br_swa")
    window = 16 if mask in ("swa", "br_swa") else None
    log2 = mask == "br_swa"

    def buffers():
        tensors = []
        for heads, seq, width in ((h, sq, d), (hk, sk, d), (hk, sk, dv)):
            tensors.append(torch.randn(b, seq, heads, width + 8, device="cuda", dtype=dtype)[..., :width].transpose(1, 2))
        ob = torch.full((b, sq, h, dv + 8), 77, device="cuda", dtype=dtype)
        lb = torch.full((b, h, sq * 2), 77.0, device="cuda")
        return [*tensors, ob[..., :dv].transpose(1, 2), lb[..., ::2]], ob, lb

    values, ob, lb = buffers()
    api = SdpaFwdDslSm80(
        *values,
        is_causal=causal,
        causal_bottom_right=br,
        window_size_left=window,
        stats_log2=log2,
        tile_m=tile_m,
        sched_policy=2 if causal else 0,
    )
    assert api.check_support()
    short_q = (d == dv == 128 and sq <= 128) or (d == 192 and dv == 128 and sq <= 64)
    expected = tile_m if tile_m is not None else (64 if short_q else 128)
    assert (api.kernel_tile_m, api.kernel_num_warps) == (expected, expected // 16)
    api.compile()
    artifact = api._compiled_kernel
    assert api.scratch_workspace_bytes() == 0

    for iteration in range(2):
        if iteration:
            values, ob, lb = buffers()
        q, k, v, o, lse = values

        def check():
            scores = (q.double() @ k.double().repeat_interleave(h // hk, 1).transpose(-1, -2)) / math.sqrt(d)
            qpos = torch.arange(sq, device="cuda")[:, None] + (sk - sq if br else 0)
            kpos = torch.arange(sk, device="cuda")[None, :]
            if causal:
                scores.masked_fill_(kpos > qpos, float("-inf"))
            if window is not None:
                scores.masked_fill_(kpos < qpos - window, float("-inf"))
            ref_lse = scores.logsumexp(-1)
            ref_o = scores.softmax(-1) @ v.double().repeat_interleave(h // hk, 1)
            torch.testing.assert_close(o.double(), ref_o, atol=4e-3, rtol=1e-2)
            torch.testing.assert_close(lse.double(), ref_lse * (math.log2(math.e) if log2 else 1), atol=2e-3, rtol=1e-3)
            assert torch.all(ob[..., dv:] == 77)
            assert torch.all(lb[..., 1::2] == 77)

        api.execute(*values)
        check()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                api.execute(*values)
            v.mul_(-0.7)
            o.fill_(float("nan"))
            lse.fill_(float("nan"))
            graph.replay()
            check()
            assert api._compiled_kernel is artifact
        finally:
            graph.reset()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("d", [128, 192])
def test_thd_explicit_64_row_tile(dtype, d):
    from cudnn.api_base import TensorDesc
    from test_sdpa_fwd_thd_sm80 import _make_graph

    torch.manual_seed(722)
    _, pack, nodes, (cq, ck) = _make_graph((8, 129), (257, 193), hkv=1, d=d, dv=128, dtype=dtype, causal=True)
    q, k, v, o, stats = nodes
    descs = {
        "sample_"
        + name: TensorDesc(
            dtype=pack[node].dtype,
            shape=tuple(node.get_dim()),
            stride=tuple(node.get_stride()),
            stride_order=(3, 1, 2, 0),
            device=pack[node].device,
            name=name,
        )
        for name, node in zip(("q", "k", "v", "o", "lse"), nodes)
    }
    # TILE_M is a standalone tuning argument; the graph row does not expose
    # a tile domain. Reuse only the graph fixture's packed buffers/metadata.
    api = SdpaFwdDslSm80(
        **descs,
        thd=True,
        tile_m=64,
        is_causal=True,
        seq_q_lens_present=True,
        seq_kv_lens_present=True,
        max_total_seq_len_q=cq[-1],
        max_total_seq_len_kv=ck[-1],
    )
    assert api.check_support()
    api.compile()
    assert (api._params.tile_m, api._params.num_warps) == (64, 4)
    lengths = {node.get_name(): tensor for node, tensor in pack.items() if node.get_name().startswith("seq_len_")}
    workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    captured = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(captured):
            api.execute(*(pack[n] for n in nodes), seq_q_lens=lengths["seq_len_q"], seq_kv_lens=lengths["seq_len_kv"], workspace=workspace)
        pack[v].mul_(0.5)
        captured.replay()
        for i in range(2):
            qs = pack[q][0, cq[i] : cq[i + 1]].transpose(0, 1).double()
            ks = pack[k][0, ck[i] : ck[i + 1]].transpose(0, 1).double().repeat_interleave(2, 0)
            vs = pack[v][0, ck[i] : ck[i + 1]].transpose(0, 1).double().repeat_interleave(2, 0)
            scores = qs @ ks.transpose(-1, -2) / math.sqrt(d)
            scores.masked_fill_(torch.arange(ks.shape[1], device="cuda")[None, :] > torch.arange(qs.shape[1], device="cuda")[:, None], float("-inf"))
            expected = (scores.softmax(-1) @ vs).transpose(0, 1)
            torch.testing.assert_close(pack[o][0, cq[i] : cq[i + 1]].double(), expected, atol=4e-3, rtol=1e-2)
            torch.testing.assert_close(pack[stats][0, :, cq[i] : cq[i + 1]].double(), scores.logsumexp(-1), atol=2e-3, rtol=1e-3)
    finally:
        captured.reset()
