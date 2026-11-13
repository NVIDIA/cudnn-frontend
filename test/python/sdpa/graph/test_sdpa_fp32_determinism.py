# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit unified FP32 forward, pinned to backend engine 21 on SM120.

Checks O and saved LSE. This engine's FP32 path has no backward implementation;
AUTO/composite routing and other GPU architectures are outside this suite.
"""

import cudnn
import pytest
import torch

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def require_sm120_fp32():
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("This suite qualifies the SM120 backend FP32 path")
    if cudnn.backend_version() < 93000:
        pytest.skip("Unified FP32 I/O requires cuDNN 9.30 or newer")


def _inputs(sq, skv, dq, dv, heads, bias_dtype=None):
    # CPU generation fixes the dataset independently of the GPU's SM count.
    rng = torch.Generator().manual_seed(20261113)
    result = {
        name: torch.randn(shape, generator=rng).mul_(0.1)
        for name, shape in (
            ("q", (4, heads, sq, dq)),
            ("k", (4, 2, skv, dq)),
            ("v", (4, 2, skv, dv)),
        )
    }
    if bias_dtype is not None:
        bias = torch.randn((4, 1, sq, skv), generator=rng).mul_(0.1).to(bias_dtype)
        bias[:, :, 0] = -torch.inf  # Declared empty-row outputs are O=0, LSE=-inf.
        result["bias"] = bias
    return result


def _reference(values):
    q, k, v = (values[name].double() for name in ("q", "k", "v"))
    group = q.shape[1] // k.shape[1]
    scores = (q @ k.repeat_interleave(group, 1).transpose(-1, -2)) * 0.125
    if "bias" in values:
        scores = scores + values["bias"].double()
    probability = torch.nan_to_num(scores.softmax(-1), nan=0.0)
    return {"o": (probability @ v.repeat_interleave(group, 1)).float(), "lse": scores.logsumexp(-1, keepdim=True).float()}


def _plan(values, handle):
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    nodes = {name: graph.tensor_like(value, name=name) for name, value in values.items()}
    o, lse = graph.sdpa(
        q=nodes["q"],
        k=nodes["k"],
        v=nodes["v"],
        bias=nodes.get("bias"),
        generate_stats=True,
        attn_scale=0.125,
        implementation=cudnn.attention_implementation.UNIFIED,
    )
    b, h, sq, _ = values["q"].shape
    dv = values["v"].shape[-1]
    outputs = {
        "o": torch.full((b, h, sq, dv), torch.nan, device="cuda"),
        "lse": torch.full((b, h, sq, 1), torch.nan, device="cuda"),
    }
    for node, value in ((o, outputs["o"]), (lse, outputs["lse"])):
        node.set_output(True).set_dim(list(value.shape)).set_stride(list(value.stride())).set_data_type(cudnn.data_type.FLOAT)
    # No fallback: lack of this plan is a failure on the supported runtime/GPU.
    graph.create_execution_plan(21, {cudnn.knob_type.KERNEL_CFG: -1, cudnn.knob_type.STREAM_K: 1})
    graph.build_plan_at_index(0)
    assert graph.get_engine_and_knobs_at_index(0)[0] == 21
    workspace = torch.empty(graph.get_workspace_size_plan_at_index(0), device="cuda", dtype=torch.uint8)

    def execute(inputs, result):
        cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)
        bindings = {node: inputs[name] for name, node in nodes.items()}
        graph.execute_plan_at_index(bindings | {o: result["o"], lse: result["lse"]}, workspace, 0, handle)

    return execute, outputs


def _check_reference(actual, expected):
    for name in expected:
        torch.testing.assert_close(actual[name].cpu(), expected[name], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize(
    "sq,skv,dq,dv,heads,bias_dtype,golden",
    [
        pytest.param(15, 31, 16, 32, 4, None, True, id="fprop-fp32-q15-k31-d16-v32"),
        pytest.param(16, 32, 32, 16, 4, None, False, id="fprop-fp32-q16-k32-d32-v16"),
        pytest.param(17, 33, 56, 80, 6, torch.float16, True, id="fprop-fp32-q17-k33-d56-v80-bias-fp16"),
        pytest.param(17, 33, 56, 80, 6, torch.bfloat16, True, id="fprop-fp32-q17-k33-d56-v80-bias-bf16"),
        pytest.param(17, 33, 56, 80, 6, torch.float32, True, id="fprop-fp32-q17-k33-d56-v80-bias-fp32"),
        pytest.param(257, 129, 128, 96, 16, None, False, id="fprop-fp32-q257-k129-d128-v96"),
    ],
)
def test_unified_fp32_determinism(sq, skv, dq, dv, heads, bias_dtype, golden, cudnn_handle):
    pool = _inputs(sq, skv, dq, dv, heads, bias_dtype)
    target = {name: value[:1].clone() for name, value in pool.items()}
    reference = _reference(target)
    baseline = None
    # Insert, reorder, duplicate, and change neighbors without changing sequence 0.
    for order in ([0], [0, 1, 2], [2, 0, 1], [0, 0, 2], [0, 3, 2]):
        cpu = {name: value[order].clone() for name, value in pool.items()}
        values = {name: value.cuda() for name, value in cpu.items()}
        execute, template = _plan(values, cudnn_handle)

        def run():
            inputs = {name: value.clone() for name, value in values.items()}
            outputs = {name: torch.full_like(value, torch.nan) for name, value in template.items()}
            execute(inputs, outputs)
            assert_bitwise(inputs, values, "inputs remain unchanged")
            return outputs

        actual = assert_replays(run)
        _check_reference(actual, _reference(cpu))
        for position, sequence in enumerate(order):
            if sequence != 0:
                continue
            selected = snapshot({name: value[position : position + 1] for name, value in actual.items()})
            if baseline is None:
                baseline = selected
            assert_bitwise(selected, baseline, f"batch {order}, position {position}")
            _check_reference(selected, reference)
    if golden:
        name = f"unified_fp32_fprop_q{sq}_k{skv}_d{dq}_v{dv}_bias_{bias_dtype}"
        assert_golden(name, target, baseline)


def test_unified_fp32_warm_refill(cudnn_handle):
    from warm_stress import StressCase, assert_warm_stress

    cpu = {name: value[:2] for name, value in _inputs(257, 129, 64, 64, 16).items()}
    values = {name: value.cuda() for name, value in cpu.items()}
    execute, outputs = _plan(values, cudnn_handle)
    execute(values, outputs)
    _check_reference(outputs, _reference(cpu))
    report = assert_warm_stress(
        StressCase(
            name="unified FP32 forward q257 k129 d64",
            launch=lambda: execute(values, outputs),
            outputs=outputs,
            thread_blocks=2 * 2 * ((257 * 8 + 15) // 16),
            refills_per_launch=4,
        ),
        warm=256,
        poisoned=256,
    )
    assert_bitwise(values, {name: value.cuda() for name, value in cpu.items()}, "warm inputs remain unchanged")
    print(report.text())
