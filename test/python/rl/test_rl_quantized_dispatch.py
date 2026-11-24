# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import cudnn
from cudnn import data_type as dt

pytestmark = pytest.mark.L0


def _graph():
    return cudnn.pygraph(compute_data_type=dt.FLOAT, intermediate_data_type=dt.FLOAT)


def test_tensor_scale_multiplication_cannot_be_reassociated():
    graph = _graph()
    a = graph.tensor([4, 32], data_type=dt.FP8_E4M3)
    b = graph.tensor([32, 16], data_type=dt.FP8_E4M3)
    sa, sb = [graph.tensor([1], data_type=dt.FLOAT) for _ in range(2)]
    # Rounding (acc * sa) * sb can differ from acc * (sa * sb).
    out = graph.mul(a=graph.mul(a=graph.matmul(a, b), b=sa), b=sb)
    out.set_output(True).set_data_type(dt.BFLOAT16)
    graph.select_numeric_notes([cudnn.numerical_note.RL])
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="rl_gemm"):
        graph.create_execution_plans()


@pytest.mark.parametrize("rl", [False, True])
def test_matmul_pdl_requires_rl_mixed_projection(rl):
    graph = _graph()
    a = graph.tensor([4, 32], data_type=dt.BFLOAT16)
    b = graph.tensor([32, 16], data_type=dt.BFLOAT16)
    graph.matmul(a, b, enable_pdl=True).set_output(True).set_data_type(dt.BFLOAT16)
    if rl:
        graph.select_numeric_notes([cudnn.numerical_note.RL])
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="PDL|enable_pdl"):
        graph.create_execution_plans()


def test_swizzled_scale_checks_physical_capacity_dtype_and_layout(rl_device, monkeypatch):
    from cudnn.gemm.rl import quantized

    launched = []

    class Kernel:
        workspace_bytes = 0

        def execute(self, *args):
            launched.append(args)

    monkeypatch.setattr(quantized, "build", lambda op, device: Kernel())
    g = _graph()
    m, n, k = 4, 8, 128
    a = g.tensor([m, k], data_type=dt.FP8_E4M3)
    b = g.tensor([k, n], stride=[1, k], data_type=dt.FP8_E4M3)
    sf = dict(data_type=dt.FP8_E8M0, reordering_type=cudnn.tensor_reordering.F8_128x4)
    sa = g.tensor([128, 4], stride=[4, 1], **sf)
    sb = g.tensor([4, 128], stride=[1, 4], **sf)
    qa = g.block_scale_dequantize(a, sa, block_size=[1, 32])
    qb = g.block_scale_dequantize(b, sb, block_size=[32, 1])
    out = g.matmul(qa, qb).set_output(True).set_data_type(dt.BFLOAT16)
    g.select_numeric_notes([cudnn.numerical_note.RL])
    g.build()
    assert g.selected_engine.name == "rl_gemm"
    storage = torch.empty(1024, device=rl_device, dtype=torch.uint8)
    buffers = {
        a: torch.empty((m, k), device=rl_device, dtype=torch.float8_e4m3fn),
        b: torch.empty((n, k), device=rl_device, dtype=torch.float8_e4m3fn).T,
        sa: storage[:512],
        sb: storage[512:],
        out: torch.empty((m, n), device=rl_device, dtype=torch.bfloat16),
    }
    g.execute(buffers, None)
    assert launched[0][3] == storage.data_ptr()
    for invalid, reason in (
        (storage[:511], "requires 512 bytes"),
        (storage.view(torch.float16), "dtype"),
        (storage[::2], "dense buffer"),
    ):
        with pytest.raises(ValueError, match=reason):
            g.execute({**buffers, sa: invalid}, None)
    assert len(launched) == 1
