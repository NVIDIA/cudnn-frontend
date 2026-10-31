# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native D64 causal interior math with tail rows, sinks and retained replay."""

import math
import warnings

import pytest
import torch

from frost_test_utils import requires_dsl
from test_utils import torch_fork_set_rng


def _reference(values, bottom_right, scale):
    q, k, v = (values[n].double() for n in ("q", "k", "v"))
    k = k.repeat_interleave(q.shape[1] // k.shape[1], 1)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], 1)
    scores = q @ k.transpose(-1, -2) * scale
    i = torch.arange(q.shape[2], device=q.device)[:, None]
    j = torch.arange(k.shape[2], device=q.device)[None, :]
    scores.masked_fill_(j > i + (k.shape[2] - q.shape[2] if bottom_right else 0), -torch.inf)
    if "sink" in values:
        scores = torch.cat([scores, values["sink"].double().expand(*scores.shape[:-1], 1)], -1)
    probs = scores.softmax(-1).nan_to_num(0.0)[..., : k.shape[2]]
    return probs @ v, scores.logsumexp(-1).unsqueeze(-1)


@requires_dsl
@pytest.mark.L0
@pytest.mark.parametrize("cga", [1, 2])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("bottom_right,with_sink", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("k_tile_growth", [1.0, 2.0], ids=["ordinary", "rescale"])
@torch_fork_set_rng(seed=1391)
def test_d64_causal_softmax_rebind_and_replay(cga, packed, bottom_right, with_sink, k_tile_growth, cudnn_handle, monkeypatch):
    import cudnn
    from cudnn.engines.manifest import MANIFEST
    from cudnn.sdpa.fwd import api_dsl

    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("native SM100-family D64 prefill")
    original_stream = cudnn.get_stream(cudnn_handle)
    capture = None
    loaded_modules = []
    original_load = api_dsl.load_template

    def record_template(path, params, *args, **kwargs):
        module = original_load(path, params, *args, **kwargs)
        if path.endswith("/sm100/prefill_d128_f16.py") and params.d_flavor == 64:
            loaded_modules.append(module)
        return module

    monkeypatch.setattr(api_dsl, "load_template", record_template)
    b, hq, hk, sq, sk, d = 2, 8, 2, 769, 641, 64
    if bottom_right:
        sq, sk = sk, sq
    scale = (-1 if bottom_right else 1) / math.sqrt(d)

    def inputs():
        result = {
            n: (torch.randn(b, s, h, d, device="cuda") * (2.0 if n != "v" else 0.5)).bfloat16().transpose(1, 2)
            for n, s, h in (("q", sq, hq), ("k", sk, hk), ("v", sk, hk))
        }
        # Increasing key-tile magnitude forces online-max updates and O rescaling;
        # ordinary random logits often leave that pipeline dependency dormant.
        result["k"].mul_(k_tile_growth ** (torch.arange(sk, device="cuda") // 128)[None, None, :, None])
        if with_sink:
            result["sink"] = torch.tensor([-torch.inf, -100, 0, 9, -torch.inf, -100, 0, 9], device="cuda").view(1, hq, 1, 1)
        return result

    try:
        cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
        values = inputs()
        graph = cudnn.pygraph(
            io_data_type=cudnn.data_type.BFLOAT16,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=cudnn_handle,
        )
        nodes = {n: graph.tensor_like(t) for n, t in values.items()}
        kwargs = dict(q=nodes["q"], k=nodes["k"], v=nodes["v"], attn_scale=scale, generate_stats=True)
        kwargs["use_causal_mask_bottom_right" if bottom_right else "use_causal_mask"] = True
        if with_sink:
            kwargs["sink_token"] = nodes["sink"]
        out, stats = graph.sdpa(**kwargs)
        o = torch.empty_like(values["q"])
        lse = torch.empty((b, sq, hq, 1), device="cuda", dtype=torch.float32).transpose(1, 2)
        out.set_output(True).set_dim(o.shape).set_stride(o.stride())
        stats.set_output(True).set_dim(lse.shape).set_stride(lse.stride()).set_data_type(cudnn.data_type.FLOAT)
        graph.validate()
        graph.build_operation_graph()
        engine = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd").offered_ids()["sdpa_fwd_prefill_sm100"]
        knobs = {
            getattr(cudnn.knob_type, n): v for n, v in dict(TILE_M=128, TILE_N=128, TILE_CGA_M=cga, PACK_GQA=int(packed), SPLIT_KV=1, SCHED_POLICY=1).items()
        }
        graph.create_execution_plan(engine, knobs)
        graph.select_plan(graph.get_execution_plan_count() - 1)
        graph.check_support()
        graph.build_plans()
        assert loaded_modules, "the graph must compile the native D64 prefill module"
        for module in loaded_modules:
            assert module.CFG.CTA_MMA == cga
            assert module._E2E_ENABLED is (torch.cuda.get_device_capability() == (10, 0)), "the graph lost its architecture-specific exp2 path"
            assert module._D64_EARLY_SOFTMAX_HANDOFF is (torch.cuda.get_device_capability() == (10, 0))
            assert module._D64_JOINT_ROW_MAX is (torch.cuda.get_device_capability() == (10, 0) and cga == 1)
        workspace = torch.empty(max(1, graph.get_workspace_size()), device="cuda", dtype=torch.uint8)

        def bindings():
            return {**{nodes[n]: t for n, t in values.items()}, out: o, stats: lse}

        def poison():
            o.fill_(torch.nan)
            lse.fill_(torch.nan)

        def check():
            expected_o, expected_lse = _reference(values, bottom_right, scale)
            torch.testing.assert_close(o.double(), expected_o, atol=0.015, rtol=0.02)
            torch.testing.assert_close(lse.double(), expected_lse, atol=1e-4, rtol=1e-4)

        for _ in range(2):
            values = inputs()  # New storage must bind to the already compiled plan.
            poison()
            graph.execute(bindings(), workspace, handle=cudnn_handle)
            check()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        cudnn.set_stream(cudnn_handle, stream.cuda_stream)
        capture = torch.cuda.CUDAGraph()
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="The CUDA Graph is empty.*")
            with torch.cuda.graph(capture, stream=stream):
                graph.execute(bindings(), workspace, handle=cudnn_handle)
        for _ in range(3):
            for name, tensor in inputs().items():
                values[name].copy_(tensor)
            poison()
            capture.replay()
            check()
    finally:
        try:
            if capture is not None:
                capture.reset()
        finally:
            cudnn.set_stream(cudnn_handle, original_stream)
