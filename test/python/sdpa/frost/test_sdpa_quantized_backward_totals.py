# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Declared packed totals preserve native FP8 backward results (#746).

MXFP8's THD harness currently has no native backward numerical path. Its
binding/lowering controls remain in test_sdpa_fp8_bwd_thd_attrs.py; the FROST
SM107 quantized THD rows require totals and cannot serve an omitted-total twin.
"""

import json

import cudnn
import pytest
import torch

from determinism_utils import assert_bitwise, snapshot
from sdpa import fp8
from sdpa.random_config import ExecConfig

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize(
    "input_dtype,output_dtype",
    [(torch.float8_e4m3fn, torch.float8_e4m3fn), (torch.float8_e4m3fn, torch.float16), (torch.float8_e5m2, torch.bfloat16)],
    ids=["e4m3-e4m3", "e4m3-half", "e5m2-bf16"],
)
@pytest.mark.parametrize("causal", [False, True], ids=["unmasked", "causal"])
@pytest.mark.parametrize("hkv", [2, 1], ids=["mha", "gqa"])
def test_native_fp8_backward_declared_totals_preserve_results(monkeypatch, request, cudnn_handle, d, input_dtype, output_dtype, causal, hkv):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("native ragged FP8 backward requires Blackwell or newer")
    if cudnn.backend_version() <= 92100:
        pytest.skip("native ragged FP8 backward requires cuDNN after 9.21.0")
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "0")
    cfg = ExecConfig(
        data_type=input_dtype,
        output_type=output_dtype,
        rng_data_seed=1412,
        rng_geom_seed=746,
        is_infer=False,
        is_padding=True,
        is_ragged=True,
        is_determin=True,
        batches=2,
        h_q=2,
        h_k=hkv,
        h_v=hkv,
        s_q=128,
        s_kv=192,
        d_qk=d,
        d_v=d,
        seq_len_q=[128, 64],
        seq_len_kv=[192, 128],
        diag_align=cudnn.diagonal_alignment.TOP_LEFT,
        right_bound=0 if causal else None,
        with_ragged_token_gap=False,
        with_ragged_head_gap=False,
    )
    cfg.fill_derived_fields()
    totals = dict(max_total_seq_len_q=sum(cfg.seq_len_q), max_total_seq_len_kv=sum(cfg.seq_len_kv))
    original_op = cudnn.pygraph.sdpa_fp8_backward
    original_execute = cudnn.pygraph.execute
    original_stats = fp8.print_tensor_stats
    declared = False
    outputs, records, inputs = {}, [], {}

    def backward(graph, *args, **kwargs):
        if declared:
            kwargs.update(totals)
        return original_op(graph, *args, **kwargs)

    def execute(graph, *args, **kwargs):
        nonlocal inputs
        quantized_backward = len(graph.nodes) == 1 and graph.nodes[0].node_type.name == "SDPA_FP8_BWD"
        if quantized_backward:
            assert graph.selected_engine is None, "the comparison must exercise a native backend plan"
            if not inputs:
                excluded = {int(getattr(fp8.GraphBwdUid, name)) for name in ("dQ", "dK", "dV", "dQ_amax", "dK_amax", "dV_amax", "dP_amax")}
                inputs = {str(uid): tensor.detach().clone() for uid, tensor in args[0].items() if int(uid) not in excluded}
            node = graph.nodes[0]
            for name, value in totals.items():
                assert node.params.get(name) == (value if declared else None)
            native = [n for n in json.loads(repr(graph._lowered_graph))["nodes"] if n.get("tag") == "SDPA_FP8_BWD"]
            assert len(native) == 1
            for name, value in totals.items():
                assert native[0][name] == (value if declared else None)
            records.append((graph.get_engine_and_knobs_at_index(graph._plan_index), graph.get_workspace_size()))
        return original_execute(graph, *args, **kwargs)

    def collect_stats(tensor, tag=None):
        result = original_stats(tensor, tag=tag)
        live_rows = {"dQ_gpu": totals["max_total_seq_len_q"], "dK_gpu": totals["max_total_seq_len_kv"], "dV_gpu": totals["max_total_seq_len_kv"]}
        if tag in live_rows:
            # Packed allocation tails are intentionally NaN-poisoned; only
            # the declared live rows are gradient outputs.
            outputs[tag] = tensor[: live_rows[tag]].detach().clone()
        elif tag in {"dQ_amax_gpu", "dK_amax_gpu", "dV_amax_gpu", "dP_amax_gpu"}:
            outputs[tag] = tensor.detach().clone()
        return result

    monkeypatch.setattr(cudnn.pygraph, "sdpa_fp8_backward", backward)
    monkeypatch.setattr(cudnn.pygraph, "execute", execute)
    monkeypatch.setattr(fp8, "print_tensor_stats", collect_stats)
    runs = []
    for declare_totals in (False, True):
        declared = declare_totals
        outputs, records, inputs = {}, [], {}
        # Both runs retain the harness's independent quantized reference and
        # deterministic replay checks; observations do not replace assertions.
        try:
            fp8.exec_sdpa_fp8(cfg, request, cudnn_handle)
        except pytest.skip.Exception as exc:
            # Every cell is qualified on this device/runtime. Losing its
            # plans must fail, not turn numerical coverage into skips.
            if torch.cuda.get_device_capability() == (10, 0) and cudnn.backend_version() == 92800:
                pytest.fail(f"qualified native FP8 totals case unexpectedly skipped: {exc}")
            raise
        assert records, "no native backward execution was observed"
        assert {"dQ_gpu", "dK_gpu", "dV_gpu", "dQ_amax_gpu", "dK_amax_gpu", "dV_amax_gpu", "dP_amax_gpu"} <= outputs.keys()
        runs.append((snapshot(outputs), records, inputs))
    # Same-seed generation is also checked at the actual bound input buffers,
    # including payloads, scales, offsets, O and Stats; NaN tails are compared
    # as bytes without treating unused storage as a finite output.
    assert_bitwise(runs[1][2], runs[0][2], "paired backward inputs")
    assert runs[0][1] == runs[1][1], "native configuration or workspace changed"
    request.node.user_properties.append(("native_backward_records", str(runs[0][1])))
    assert_bitwise(runs[1][0], runs[0][0], "declaring backward packed totals")
