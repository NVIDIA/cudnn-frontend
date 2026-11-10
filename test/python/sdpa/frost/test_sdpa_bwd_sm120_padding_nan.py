# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM120 native backward: runtime padding is unread data, including NaNs in physically valid storage."""

import cudnn
import pytest
import torch

from frost_test_utils import requires_dsl, select_engine
from sdpa.frost.test_sdpa_bwd_dsl_sm120 import _ref_bwd

pytestmark = [
    pytest.mark.L0,
    requires_dsl,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12, reason="requires SM120"),
]


@pytest.fixture(autouse=True)
def _enable_frost(monkeypatch):
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")


@pytest.mark.parametrize("d,dv,hk,causal", [(64, 64, 4, False), (128, 128, 2, True), (192, 128, 1, True), (256, 256, 2, False)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sq,skv", [(128, 128), (96, 192)])
def test_runtime_padding_nan_backward(d, dv, hk, causal, dtype, sq, skv):
    """K/V rows past seq_len_kv and Q/dO/O/Stats rows past seq_len_q hold NaN; the valid rows
    of dQ, dK and dV must match a reference computed on clean data."""
    torch.manual_seed(120 + d + sq)
    b, h = 2, 4
    scale = d**-0.5
    lens_q, lens_kv = [sq - 7, sq // 2], [skv - 3, skv // 2]

    def view(heads, seq, width):
        return (torch.randn(b, seq, heads, width + 8, dtype=dtype, device="cuda") * 0.3)[..., :width].transpose(1, 2)

    bufs = {"q": view(h, sq, d), "k": view(hk, skv, d), "v": view(hk, skv, dv), "do": view(h, sq, dv)}
    for name, lens in (("q", lens_q), ("do", lens_q), ("k", lens_kv), ("v", lens_kv)):
        for bi, n in enumerate(lens):
            bufs[name][bi, :, n:] = 0
    o, stats, dq, dk, dv_ref, _ = _ref_bwd(*(bufs[n] for n in ("q", "k", "v", "do")), scale=scale, is_causal=causal, padding=(lens_q, lens_kv))
    bufs["o"] = view(h, sq, dv).copy_(o)
    bufs["stats"] = torch.empty(b, h, sq * 2, 1, device="cuda")[:, :, ::2].copy_(stats)
    for name, lens in (("q", lens_q), ("do", lens_q), ("o", lens_q), ("k", lens_kv), ("v", lens_kv)):
        for bi, n in enumerate(lens):
            bufs[name][bi, :, n:] = float("nan")
    for bi, n in enumerate(lens_q):
        bufs["stats"][bi, :, n:] = float("nan")
    bufs["seq_q"] = torch.tensor(lens_q, dtype=torch.int32, device="cuda").reshape(b, 1, 1, 1)
    bufs["seq_kv"] = torch.tensor(lens_kv, dtype=torch.int32, device="cuda").reshape(b, 1, 1, 1)
    for name, src in (("dq", "q"), ("dk", "k"), ("dv", "v")):
        bufs[name] = torch.full(bufs[src].shape, float("nan"), dtype=dtype, device="cuda")

    io = cudnn.data_type.BFLOAT16 if dtype == torch.bfloat16 else cudnn.data_type.HALF
    graph = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    refs = {name: graph.tensor_like(buf, name=name) for name, buf in bufs.items() if name not in ("dq", "dk", "dv")}
    grads = graph.sdpa_backward(
        q=refs["q"],
        k=refs["k"],
        v=refs["v"],
        o=refs["o"],
        dO=refs["do"],
        stats=refs["stats"],
        attn_scale=scale,
        use_causal_mask=causal,
        use_padding_mask=True,
        seq_len_q=refs["seq_q"],
        seq_len_kv=refs["seq_kv"],
    )
    for name, ref in zip(("dq", "dk", "dv"), grads):
        ref.set_output(True).set_dim(bufs[name].shape).set_stride(bufs[name].stride()).set_data_type(io)
        refs[name] = ref
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    select_engine(graph, "sdpa_bwd_sm120")
    graph.check_support()
    graph.build_plans()
    workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda")
    graph.execute({refs[name]: buf for name, buf in bufs.items()}, workspace)
    torch.cuda.synchronize()

    for name, expected, lens in (("dq", dq, lens_q), ("dk", dk, lens_kv), ("dv", dv_ref, lens_kv)):
        for bi, n in enumerate(lens):
            got = bufs[name][bi, :, :n].float()
            assert not torch.isnan(got).any(), f"{name}[{bi}]: NaN in valid rows"
            torch.testing.assert_close(got, expected[bi, :, :n].float(), atol=0.03, rtol=0.03)
