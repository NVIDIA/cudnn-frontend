# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch

import cudnn
from determinism_utils import assert_bitwise, assert_golden, assert_replays

from .matmul import build_matmul, execute_native, native_matmul

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("n,k,biased", [(72, 264, False), (128, 8192, True)], ids=["kfold-tail", "four-chunks-bias"])
def test_batch_invariant_rows_and_replays(gemm_device, n, k, biased):
    rng = torch.Generator(device=gemm_device).manual_seed(981)
    x = torch.randn((135, k), device=gemm_device, dtype=torch.bfloat16, generator=rng)
    b = (torch.randn((n, k), device=gemm_device, dtype=torch.bfloat16, generator=rng) * k**-0.5).T
    bias = torch.randn((1, n), device=gemm_device, dtype=torch.bfloat16, generator=rng) if biased else None
    out = assert_replays(build_matmul(x, b, bias))["out"]
    ref = x.double() @ b.double()
    if biased:
        ref += bias.double()
    torch.testing.assert_close(out.double(), ref, atol=0.01, rtol=0.01)

    rows = [129, 0, 129, 134, 17]
    assert_bitwise(build_matmul(x[rows], b.contiguous(), bias)(), {"out": out[rows]}, "reorder and duplicate")
    assert_bitwise(build_matmul(x[129:130], b, bias)(), {"out": out[129:130]}, "rollout row")


@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
def test_batch_invariant_matmul_pipeline_refill(gemm_device):
    from cudnn.gemm.batch_invariant import chunks_for, schedule_for
    from warm_stress import StressCase, assert_warm_stress, stress_validation

    m, n, k = 2048, 2048, 2048
    chunks = chunks_for(n, k)
    schedule = schedule_for(m, n, chunks)
    rng = torch.Generator().manual_seed(83)
    a, b = [torch.randint(-4, 5, shape, generator=rng).to(device=gemm_device, dtype=torch.bfloat16) for shape in ((m, k), (k, n))]
    run = build_matmul(a, b)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=gemm_device)
    expected = (a.double() @ b.double()).bfloat16()
    assert_warm_stress(
        StressCase(
            "batch_invariant_bf16_pipeline",
            lambda: run(out),
            {"out": out},
            thread_blocks=schedule.grid_x(m, n) * schedule.cluster_n * chunks,
            refills_per_launch=k // (64 * chunks * schedule.stages) - 1,
            reference={"out": expected},
        ),
        validation=stress_validation(gemm_device, cluster=schedule.mma_group * schedule.cluster_n),
    )


def test_batch_invariant_forward_with_native_cudnn_linear_backward(gemm_device):
    rng = torch.Generator(device=gemm_device).manual_seed(101)
    x = torch.randn((136, 256), device=gemm_device, dtype=torch.bfloat16, generator=rng) * 0.125
    weight = torch.randn((64, 256), device=gemm_device, dtype=torch.bfloat16, generator=rng) * 0.125
    bias = torch.randn((1, 64), device=gemm_device, dtype=torch.bfloat16, generator=rng) * 0.125
    y = build_matmul(x, weight.T, bias)()["out"]
    dy = y * 0.5  # gradient of sum(y ** 2) / 4
    dy3 = dy.unsqueeze(0)
    dx = native_matmul(dy3, weight.unsqueeze(0))
    dw = native_matmul(dy3.transpose(-1, -2), x.unsqueeze(0))
    torch.testing.assert_close(dx[0].double(), dy.double() @ weight.double(), atol=0.008, rtol=0.008)
    torch.testing.assert_close(dw[0].double(), dy.double().T @ x.double(), atol=0.008, rtol=0.008)

    g = cudnn.pygraph(compute_data_type=cudnn.data_type.FLOAT)
    d = g.tensor_like(dy3)
    db = g.reduction(d, mode=cudnn.reduction_mode.ADD, out_dims=[1, 1, dy.shape[-1]])
    db.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    result = torch.empty((1, 1, dy.shape[-1]), dtype=torch.float32, device=gemm_device)
    execute_native(g, {d: dy3, db: result}, gemm_device)
    torch.testing.assert_close(result.flatten().double(), dy.double().sum(0))


@pytest.mark.parametrize("n,k", [(64, 120), (16, 1024)], ids=["kfold", "kchunk"])
def test_batch_invariant_forward_golden(gemm_device, n, k):
    # Exact dyadic values make the reference independent of the reference GEMM's reduction order.
    rng = torch.Generator().manual_seed(43)
    a, b, bias = [(torch.randint(-8, 9, shape, generator=rng).to(device=gemm_device, dtype=torch.bfloat16) / 8) for shape in ((5, k), (k, n), (1, n))]
    output = assert_replays(build_matmul(a, b, bias), repeats=2)
    expected = (a.double() @ b.double() + bias.double()).bfloat16()
    assert_bitwise(output, {"out": expected})
    assert_golden(f"batch_invariant_bf16_matmul_n{n}_k{k}_bias", {"a": a, "b": b, "bias": bias}, output, baseline_dir=Path(__file__).with_name("goldens"))


def test_batch_invariant_note_rejects_fp16_matmul(gemm_device):
    a = torch.ones((1, 32, 128), device=gemm_device, dtype=torch.float16)
    b = torch.ones((1, 128, 64), device=gemm_device, dtype=torch.float16)
    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        build_matmul(a, b)
