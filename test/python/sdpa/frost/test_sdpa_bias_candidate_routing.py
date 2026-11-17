# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A-mode bias candidates execute with fresh bindings and changed captured inputs."""

import cudnn
import pytest
import torch

from cudnn.engines.manifest import MANIFEST
from determinism_utils import assert_bitwise, snapshot
from frost_test_utils import requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]


def _check(bufs, d, causal):
    q, k, v = (bufs[name].double() for name in ("q", "k", "v"))
    scores = (q @ k.repeat_interleave(2, 1).transpose(-1, -2)) * d**-0.5 + bufs["bias"].double()
    if causal:
        scores.masked_fill_(torch.arange(k.shape[-2], device="cuda")[None, :] > torch.arange(q.shape[-2], device="cuda")[:, None], -torch.inf)
    reference = scores.softmax(-1) @ v.repeat_interleave(2, 1)
    torch.testing.assert_close(bufs["o"].double(), reference, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(bufs["lse"].double(), scores.logsumexp(-1), atol=3e-4, rtol=3e-4)
    assert torch.isfinite(bufs["o"]).all() and torch.isfinite(bufs["lse"]).all()


def _exercise(dtype, bias_fp32, causal, sq, d, sk, bias_heads, cudnn_handle):
    b, h, hk = 2, 4, 2
    generator = torch.Generator().manual_seed(2595)
    raw = {
        name: (torch.randn(b, seq, heads, d, generator=generator) * 0.3).to(dtype).cuda() for name, heads, seq in (("q", h, sq), ("k", hk, sk), ("v", hk, sk))
    }
    raw["o"] = torch.empty_like(raw["q"])
    pitch = (sk + 7) // 8 * 8
    raw["bias"] = (torch.randn(1, bias_heads, sq, pitch, generator=generator) * 0.7).to(torch.float32 if bias_fp32 else dtype).cuda()
    raw["lse"] = torch.empty(b, h, sq, dtype=torch.float32, device="cuda")

    def views(storage):
        return {**{name: storage[name].transpose(1, 2) for name in ("q", "k", "v", "o")}, "bias": storage["bias"][..., :sk], "lse": storage["lse"]}

    bufs = views(raw)
    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    graph = cudnn.pygraph(handle=cudnn_handle, io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tensors = {name: graph.tensor_like(bufs[name], name=name) for name in ("q", "k", "v", "bias")}
    o, stats = graph.sdpa(**tensors, generate_stats=True, attn_scale=d**-0.5, use_causal_mask=causal)
    o.set_output(True).set_data_type(io).set_dim(bufs["o"].shape).set_stride(bufs["o"].stride())
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim([b, h, sq, 1]).set_stride([h * sq, sq, 1, 1])
    tensors.update(o=o, lse=stats)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    offered = next(family for family in MANIFEST if family.name == "frost_sdpa_fwd").offered_ids()
    candidates = [index for index, plan in enumerate(graph.plans) if plan.engine_id in offered.values()]
    assert candidates, "A-mode must retain a native bias plan; a backend-only result does not cover this regression"
    graph.select_plan(candidates[0])
    graph.check_support()
    graph.build_plans()
    assert graph.get_engine_and_knobs_at_index(graph._plan_index)[0] in offered.values()
    workspace = torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    original = None
    for fresh_bindings in (False, True):
        if fresh_bindings:
            previous = raw
            raw = {name: value.clone() for name, value in raw.items()}
            assert all(raw[name].data_ptr() != previous[name].data_ptr() for name in raw)
            bufs = views(raw)
        pack = {tensors[name]: buf for name, buf in bufs.items()}
        bufs["o"].fill_(torch.nan)
        bufs["lse"].fill_(torch.nan)
        torch.cuda.synchronize()
        graph.execute(pack, workspace, handle=cudnn_handle)
        torch.cuda.synchronize()
        _check(bufs, d, causal)
        result = snapshot({name: bufs[name] for name in ("o", "lse")})
        if original is None:
            original = result
        else:
            assert_bitwise(original, result, "fresh bindings preserve the selected bias plan")

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    captured = torch.cuda.CUDAGraph()
    previous_stream = cudnn.get_stream(cudnn_handle)
    try:
        cudnn.set_stream(cudnn_handle, stream.cuda_stream)
        with torch.cuda.graph(captured, stream=stream):
            graph.execute(pack, workspace, handle=cudnn_handle)
        torch.cuda.current_stream().wait_stream(stream)
        bufs["bias"].mul_(-0.7)
        bufs["v"].mul_(0.6)
        torch.cuda.synchronize()
        graph.execute(pack, workspace, handle=cudnn_handle)
        torch.cuda.synchronize()
        eager = snapshot({name: bufs[name] for name in ("o", "lse")})
        bufs["o"].fill_(torch.nan)
        bufs["lse"].fill_(torch.nan)
        captured.replay()
        torch.cuda.synchronize()
        _check(bufs, d, causal)
        assert_bitwise(eager, snapshot({name: bufs[name] for name in ("o", "lse")}), "changed-input capture matches eager")
    finally:
        cudnn.set_stream(cudnn_handle, previous_stream)
        captured.reset()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("bias_fp32", [False, True], ids=["io-bias", "fp32-bias"])
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize("sq", [17, 67], ids=["short", "tail"])
def test_sm120_bias_a_mode_routing(dtype, bias_fp32, causal, sq, cudnn_handle):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("requires an SM120-family GPU")
    _exercise(dtype, bias_fp32, causal, sq, 128, 139, 1, cudnn_handle)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
def test_sm100_bias_a_mode_routing(dtype, causal, cudnn_handle):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("requires SM100/SM103")
    _exercise(dtype, False, causal, 128, 64, 256, 4, cudnn_handle)
