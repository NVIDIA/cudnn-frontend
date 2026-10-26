# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Rubin D64 FP8 geometry through graph binding and retained replay."""

import math
import warnings

import pytest
import torch

from frost_test_utils import requires_dsl
from test_sdpa_fwd_fp8_sm100 import _check, _ref
from test_utils import torch_fork_set_rng

_CASES = [
    ("e4m3", torch.float8_e4m3fn, "causal", 64, False, 1, 0),
    ("e5m2", torch.float8_e5m2, "full", 64, False, 1, 1),
    ("e4m3", torch.bfloat16, "swa", 64, False, 1, 2),
    ("e5m2", torch.float16, "br", 64, False, 1, 0),
    ("e4m3", torch.float8_e5m2, "sink", 64, True, 1, 1),
    ("e5m2", torch.float8_e4m3fn, "swa", 64, True, 1, 2),
    ("e4m3", torch.bfloat16, "padded", 64, True, 1, 0),
    ("e5m2", torch.float16, "split", 64, True, 4, 1),
    ("e4m3", torch.float8_e4m3fn, "split", 64, False, 4, 2),
    ("e4m3", torch.bfloat16, "half_softmax", 64, False, 1, 0),
    ("e5m2", torch.bfloat16, "base2", 64, False, 1, 2),
    ("e4m3", torch.float16, "optional", 48, False, 1, 1),
    ("e5m2", torch.bfloat16, "causal", 16, True, 1, 2),
]


@requires_dsl
@pytest.mark.L0
@pytest.mark.parametrize("cga", [1, 2])
@pytest.mark.parametrize("in_key,out_dtype,mode,d,packed,splits,scheduler", _CASES)
@torch_fork_set_rng(seed=1397)
def test_native_d64_fp8_rebind_replay(in_key, out_dtype, mode, d, packed, splits, scheduler, cga, cudnn_handle, monkeypatch):
    import cudnn
    from cudnn.engines.manifest import MANIFEST
    from cudnn.sdpa.fwd import api_dsl

    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("native SM107 FP8 D64")
    original_stream = cudnn.get_stream(cudnn_handle)
    capture = None
    loaded = []
    load = api_dsl.load_template

    def record(path, params, *args, **kwargs):
        module = load(path, params, *args, **kwargs)
        if path.endswith("/sm107/prefill_d128_fp8.py"):
            loaded.append(module)
        return module

    monkeypatch.setattr(api_dsl, "load_template", record)
    b, hq, hk, sq, sk = 2, 8, 2, 385, 641
    in_dtype = torch.float8_e4m3fn if in_key == "e4m3" else torch.float8_e5m2
    sink = mode in ("sink", "padded")
    padded = mode == "padded"
    causal = mode not in ("full", "optional", "padded")
    br = mode == "br"
    scale = -0.125 if br else 0.125
    qlens, klens = ([385, 129], [321, 0]) if padded else (None, None)
    has_stats = has_amax = mode != "optional"
    descales = (1.25, 0.75, 0.5)

    def inputs():
        values = {n: (torch.randn(b, s, h, d, device="cuda") * 0.5).to(in_dtype).transpose(1, 2) for n, s, h in (("q", sq, hq), ("k", sk, hk), ("v", sk, hk))}
        for name, value in zip(("descale_q", "descale_k", "descale_v", "descale_s", "scale_s", "scale_o"), (*descales, 1.0, 1.0, 1.0)):
            values[name] = torch.full((1, 1, 1, 1), value, device="cuda", dtype=torch.float32)
        if sink:
            values["sink"] = torch.linspace(-3, 9, hq, device="cuda").view(1, hq, 1, 1)
        if padded:
            values["qlens"] = torch.tensor(qlens, device="cuda", dtype=torch.int32).view(b, 1, 1, 1)
            values["klens"] = torch.tensor(klens, device="cuda", dtype=torch.int32).view(b, 1, 1, 1)
        return values

    try:
        cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
        values = inputs()
        graph = cudnn.pygraph(
            io_data_type=cudnn.data_type.FP8_E4M3, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, handle=cudnn_handle
        )
        nodes = {name: graph.tensor_like(tensor) for name, tensor in values.items()}
        kwargs = {name: nodes[name] for name in ("q", "k", "v", "descale_q", "descale_k", "descale_v", "descale_s", "scale_s", "scale_o")}
        kwargs.update(attn_scale=scale, generate_stats=has_stats)
        if causal:
            kwargs["use_causal_mask_bottom_right" if br else "use_causal_mask"] = True
        if mode == "swa":
            kwargs["left_bound"] = 128
        if sink:
            kwargs["sink_token"] = nodes["sink"]
        if padded:
            kwargs.update(use_padding_mask=True, seq_len_q=nodes["qlens"], seq_len_kv=nodes["klens"])
        if mode == "base2":
            kwargs["stats_use_log2"] = True
        if mode == "half_softmax":
            kwargs["softmax_precision"] = cudnn.data_type.HALF
        out, stats, _, am = graph.sdpa_fp8(**kwargs)
        o = torch.empty_like(values["q"], dtype=out_dtype)
        lse = torch.empty((b, sq, hq, 1), device="cuda", dtype=torch.float32).transpose(1, 2)
        amax = torch.empty((1, 1, 1, 1), device="cuda", dtype=torch.float32)
        dtype = {
            torch.float16: cudnn.data_type.HALF,
            torch.bfloat16: cudnn.data_type.BFLOAT16,
            torch.float8_e4m3fn: cudnn.data_type.FP8_E4M3,
            torch.float8_e5m2: cudnn.data_type.FP8_E5M2,
        }[out_dtype]
        out.set_output(True).set_dim(o.shape).set_stride(o.stride()).set_data_type(dtype)
        if has_stats:
            stats.set_output(True).set_dim(lse.shape).set_stride(lse.stride()).set_data_type(cudnn.data_type.FLOAT)
        if has_amax:
            am.set_output(True).set_dim(amax.shape).set_stride(amax.stride()).set_data_type(cudnn.data_type.FLOAT)
        graph.validate()
        graph.build_operation_graph()
        engine = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd").offered_ids()["sdpa_fwd_prefill_sm107_fp8"]
        knobs = {
            getattr(cudnn.knob_type, name): value
            for name, value in dict(TILE_M=128, TILE_N=128, TILE_CGA_M=cga, PACK_GQA=int(packed), SPLIT_KV=splits, SCHED_POLICY=scheduler).items()
        }
        graph.create_execution_plan(engine, knobs)
        graph.select_plan(graph.get_execution_plan_count() - 1)
        graph.check_support()
        graph.build_plans()
        assert loaded and all((m.CFG.TILE_K, m.CFG.TILE_O, m.CFG.CTA_MMA) == (64, 64, cga) for m in loaded)
        workspace = torch.empty(max(1, graph.get_workspace_size()), device="cuda", dtype=torch.uint8)

        def bindings():
            pack = {nodes[name]: tensor for name, tensor in values.items()}
            pack[out] = o
            if has_stats:
                pack[stats] = lse
            if has_amax:
                pack[am] = amax
            return pack

        def poison():
            o.fill_(torch.nan)
            lse.fill_(torch.nan)
            amax.zero_()

        def check():
            ref = _ref(
                values["q"],
                values["k"],
                values["v"],
                *descales,
                in_key,
                scale=scale,
                is_causal=causal,
                bottom_right=br,
                swa_window=127 if mode == "swa" else None,
                sinks=values["sink"].flatten() if sink else None,
                seq_lens_q=qlens,
                seq_lens_kv=klens,
                return_stats=True,
            )
            reference_amax = ref.output.abs().max().item()
            _check(o, ref.output, out_dtype, in_key, amax.item() if has_amax else reference_amax, reference_amax)
            if has_stats:
                torch.testing.assert_close(lse.squeeze(-1), ref.stats * (math.log2(math.e) if mode == "base2" else 1.0), atol=0.05, rtol=0.03)

        for _ in range(2):
            values = inputs()
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
