# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native D512 packed subgroups and FP32 split partials through both APIs."""

import inspect
import math

import pytest

from frost_test_utils import requires_dsl

pytestmark = [requires_dsl]

# Group, queries, keys, splits, packed, causal BR, left window, sink, Stats, dtype.
CASES = {
    "g64_unsplit": (64, 1, 1025, 1, True, True, None, False, "none", "float16"),
    "g128_unsplit": (128, 3, 257, 1, True, False, None, True, "natural", "bfloat16"),
    "g64_split": (64, 1, 2048, 8, True, False, None, False, "log2", "float16"),
    "g128_split": (128, 4, 1025, 8, True, True, None, True, "log2", "bfloat16"),
    "partial6": (6, 17, 385, 3, True, True, 31, True, "natural", "bfloat16"),
    "keyless": (128, 129, 65, 8, True, True, 0, False, "log2", "bfloat16"),
    "unpacked": (128, 3, 257, 8, False, True, None, True, "none", "bfloat16"),
    "mha": (1, 129, 257, 3, True, False, None, False, "natural", "float16"),
}


def _cpu_oracle(q, k, v, scale, mask, sinks):
    """Independent FP64 reference using bounded ordinary CPU matrix products."""
    import numpy as np
    import torch

    q, k, v = (tensor.double().numpy() for tensor in (q, k, v))
    mask = None if mask is None else mask.numpy()
    sinks = None if sinks is None else sinks.double().numpy()
    batch, heads, sq, dim = q.shape
    group = heads // k.shape[1]
    out = np.empty_like(q)
    lse = np.empty((batch, heads, sq), dtype=np.float64)
    for b in range(batch):
        for kv_head in range(k.shape[1]):
            for first in range(kv_head * group, (kv_head + 1) * group, 8):
                last = min(first + 8, (kv_head + 1) * group)
                scores = (q[b, first:last].reshape(-1, dim) @ k[b, kv_head].T) * scale
                if mask is not None:
                    scores = np.where(np.broadcast_to(mask, (last - first, *mask.shape)).reshape(scores.shape), -np.inf, scores)
                full = scores if sinks is None else np.concatenate((scores, np.repeat(sinks[first:last], sq)[:, None]), axis=1)
                maximum = full.max(axis=1, keepdims=True)
                safe_maximum = np.where(np.isfinite(maximum), maximum, 0)
                weights = np.exp(full - safe_maximum)
                denominator = weights.sum(axis=1, keepdims=True)
                probabilities = np.divide(weights, denominator, out=np.zeros_like(weights), where=denominator != 0)
                out[b, first:last] = (probabilities[:, : k.shape[2]] @ v[b, kv_head]).reshape(last - first, sq, dim)
                with np.errstate(divide="ignore"):
                    lse[b, first:last] = (safe_maximum + np.log(denominator)).reshape(last - first, sq)
    return torch.from_numpy(out), torch.from_numpy(lse)


@pytest.mark.parametrize("graph_api", [False, True], ids=["standalone", "graph"])
@pytest.mark.parametrize("case", [pytest.param(name, marks=pytest.mark.L0 if name == "g128_split" else pytest.mark.L1) for name in CASES])
def test_d512_packed_split_rebind_replay(case, graph_api, cudnn_handle):
    import cudnn
    import torch
    from cuda.bindings import driver as cuda
    from cudnn.engines.manifest import family_for
    from test_sdpa_fwd_dsl_sm107 import _dense_mask, _half_prefold_api, _rubin_only

    _rubin_only()
    group, sq, sk, splits, packed, causal, window, sink, stats, dtype_name = CASES[case]
    api, buffers = _half_prefold_api(
        512,
        512,
        getattr(torch, dtype_name),
        prefolded=False,
        with_stats=stats != "none",
        split_kv=splits,
        b=2,
        hq=2 * group,
        hkv=2,
        s_q=sq,
        s_kv=sk,
        causal=causal,
        bottom_right=causal,
        window_left=window,
        sink=sink,
        pack_gqa=packed,
    )
    api.sched_policy = 0
    api.stats_log2 = stats == "log2"
    assert api.check_support()
    stream = torch.cuda.Stream()
    saved_stream = cudnn.get_stream(cudnn_handle)
    captures = []
    try:
        if graph_api:
            dtype = cudnn.data_type.HALF if dtype_name == "float16" else cudnn.data_type.BFLOAT16
            graph = cudnn.pygraph(io_data_type=dtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
            nodes = {name: graph.tensor_like(buffers[name]) for name in ("q", "k", "v")}
            if sink:
                nodes["sinks"] = graph.tensor_like(buffers["sinks"]).set_dim([1, 2 * group, 1, 1]).set_stride([2 * group, 1, 1, 1])
            nodes["out"], lse = graph.sdpa(
                q=nodes["q"],
                k=nodes["k"],
                v=nodes["v"],
                generate_stats=stats != "none",
                attn_scale=buffers["attn_scale"],
                use_causal_mask_bottom_right=causal,
                sliding_window_length=None if window is None else window + 1,
                sink_token=nodes.get("sinks"),
                stats_use_log2=stats == "log2",
            )
            nodes["out"].set_output(True).set_dim(buffers["out"].shape).set_stride(buffers["out"].stride())
            if stats != "none":
                buffers["lse"] = buffers["lse"].unsqueeze(-1)
                nodes["lse"] = lse.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim(buffers["lse"].shape).set_stride(buffers["lse"].stride())
            if causal and sq > sk:
                # The common graph contract rejects this geometry before engine
                # selection. The standalone case still exercises keyless rows.
                with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="max_s_q > max_s_kv"):
                    graph.validate()
                return
            graph.validate()
            graph.build_operation_graph()
            graph.create_execution_plan(
                family_for(graph).offered_ids()["sdpa_fwd_prefill_sm107"],
                {cudnn.knob_type.TILE_CGA_M: 2, cudnn.knob_type.SPLIT_KV: splits, cudnn.knob_type.PACK_GQA: int(packed), cudnn.knob_type.SCHED_POLICY: 0},
            )
            graph.select_plan(0)
            graph.check_support()
            graph.build_plans()
            api = inspect.getclosurevars(graph._compiled_plans[0]._compiled.default_stream).nonlocals["api"]
            workspace_bytes = graph.get_workspace_size()
            cudnn.set_stream(cudnn_handle, stream.cuda_stream)
        else:
            api.compile()
            workspace_bytes = api.scratch_workspace_bytes()
        assert api._k_mod.__file__.endswith("sm107/prefill_d512_f16_2x2.py")
        assert api._k_mod.CFG.SPLIT_KV == splits
        assert api._k_mod.CFG.PACK_G == (math.gcd(group, 64) if packed else 1)
        assert api._partial_torch_dtype() == (torch.float32 if splits > 1 else getattr(torch, dtype_name))

        def check(values):
            # Keep the independent FP64 reference off the instrumented GPU:
            # racecheck can spend the test deadline inside cuBLAS rather than
            # the attention kernels. Chunk heads to bound CPU reference storage.
            cpu = {name: tensor.cpu() for name, tensor in values.items()}
            mask = _dense_mask(sq, sk, causal=causal, bottom_right=causal, window_left=window)
            mask = None if mask is None else mask.cpu()
            expected_o, expected_lse = _cpu_oracle(cpu["q"], cpu["k"], cpu["v"], buffers["attn_scale"], mask, cpu.get("sinks"))
            torch.testing.assert_close(cpu["out"].double(), expected_o, atol=4e-3, rtol=4e-3)
            if stats != "none":
                if stats == "log2":
                    expected_lse = expected_lse / math.log(2)
                torch.testing.assert_close(cpu["lse"].reshape_as(expected_lse).double(), expected_lse, atol=6e-4, rtol=0)

        for ordinal in range(2):
            values = {
                name: torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device="cuda") for name, t in buffers.items() if isinstance(t, torch.Tensor)
            }
            for name, tensor in values.items():
                tensor.copy_(buffers[name])
            values["v"].mul_(1 if ordinal == 0 else -1)
            if sink:
                values["sinks"].add_(2 * ordinal)
            workspace = torch.empty(max(workspace_bytes, 1), dtype=torch.uint8, device="cuda")

            def execute():
                if graph_api:
                    graph.execute({nodes[name]: value for name, value in values.items()}, workspace, handle=cudnn_handle)
                else:
                    api.execute(
                        q_tensor=values["q"],
                        k_tensor=values["k"],
                        v_tensor=values["v"],
                        o_tensor=values["out"],
                        lse_tensor=values.get("lse"),
                        sinks=values.get("sinks"),
                        workspace=workspace,
                        current_stream=cuda.CUstream(stream.cuda_stream),
                    )

            values["out"].fill_(torch.nan)
            workspace.fill_(0xA5)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                torch.cuda.set_sync_debug_mode("error")
                try:
                    execute()
                finally:
                    torch.cuda.set_sync_debug_mode("default")
            stream.synchronize()
            check(values)
            captured = torch.cuda.CUDAGraph()
            captures.append((captured, values, workspace))
            with torch.cuda.graph(captured, stream=stream):
                execute()
        # Rebinding the plan must not replace either capture's pointer set.
        for captured, values, workspace in reversed(captures):
            values["v"].neg_()
            if sink:
                values["sinks"].add_(0.75)
            values["out"].fill_(torch.nan)
            if stats != "none":
                values["lse"].fill_(torch.nan)
            workspace.fill_(0x5A)
            torch.cuda.synchronize()
            captured.replay()
            torch.cuda.synchronize()
            check(values)
    finally:
        for captured, _, _ in captures:
            captured.reset()
        cudnn.set_stream(cudnn_handle, saved_stream)
