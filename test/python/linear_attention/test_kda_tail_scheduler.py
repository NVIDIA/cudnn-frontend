# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KDA tail work must not depend on which SM executes a persistent CTA (#1032)."""

import os
import time
from unittest.mock import patch

import pytest
import torch

pytestmark = [pytest.mark.L1, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

__global__ void occupy_sms(int* smids, unsigned long long cycles) {
    extern __shared__ volatile unsigned char scratch[];
    scratch[threadIdx.x] = static_cast<unsigned char>(threadIdx.x);
    __syncthreads();
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    if (threadIdx.x == 0) smids[blockIdx.x] = smid;
    unsigned long long start = clock64();
    while (clock64() - start < cycles) {
        asm volatile("nanosleep.u32 1000;");
    }
    scratch[threadIdx.x] = scratch[threadIdx.x];
}

void block_sms(torch::Tensor smids, int64_t cycles) {
    c10::cuda::CUDAGuard guard(smids.device());
    constexpr int shared_bytes = 160 * 1024;
    C10_CUDA_CHECK(cudaFuncSetAttribute(occupy_sms,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    occupy_sms<<<smids.numel(), 32, shared_bytes,
        c10::cuda::getCurrentCUDAStream()>>>(smids.data_ptr<int>(), cycles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


@pytest.fixture(scope="module")
def sm_blocker():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires Blackwell FROST KDA")
    if os.environ.get("CUDA_LAUNCH_BLOCKING") == "1":
        pytest.skip("requires concurrent CUDA streams")
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

    available, version = cutedsl_state()
    if not available or cutedsl_too_old(version):
        pytest.skip("requires a FROST-compatible CuTe DSL")
    from torch.utils.cpp_extension import CUDA_HOME, load_inline

    if CUDA_HOME is None:
        pytest.skip("requires nvcc for the occupancy kernel")
    major, minor = torch.cuda.get_device_capability()
    return load_inline(
        name="kda_tail_scheduler_test",
        cpp_sources="void block_sms(torch::Tensor smids, int64_t cycles);",
        cuda_sources=_CUDA_SOURCE,
        functions=["block_sms"],
        extra_cuda_cflags=["-O2", f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"],
    )


@pytest.mark.parametrize("batch_invariant", [False, True], ids=["default", "bi"])
def test_kda_backward_concurrent_tail(sm_blocker, batch_invariant):
    from cudnn.linear_attention import kimi_delta_attention

    # This work count reserves a partial tail wave on the 148-SM B300.
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    tail_count = (34 * 64 - 1) % num_sms + 1
    if tail_count * 2 < num_sms:
        pytest.skip("this SM count does not exercise the reserved tail wave")
    torch.manual_seed(42)
    shape = (8192, 64, 128)
    xs = {name: (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1).requires_grad_() for name in ("q", "k", "v", "g")}
    xs.update(
        beta=torch.ones(shape[:2], device="cuda", dtype=torch.bfloat16),
        a_log=torch.zeros(64, device="cuda", requires_grad=True),
        dt_bias=torch.full((64, 128), -4.0, device="cuda", requires_grad=True),
        cu_seqlens=torch.tensor([i * 248 for i in range(32)] + [7935, 8182, 8192], device="cuda", dtype=torch.int32),
    )
    names = [name for name, value in xs.items() if value.requires_grad]
    targets = [xs[name] for name in names]
    dy = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.01
    output = kimi_delta_attention(**xs, scale=128**-0.5, batch_invariant=batch_invariant, safe_gate=True, gate_lower_bound=-5.0, plan_name="kda_frost")[0]
    reference = tuple(g.detach().clone() for g in torch.autograd.grad(output, targets, dy, retain_graph=True))
    assert all(bool(g.isfinite().all()) for g in reference)

    stream = torch.cuda.Stream()
    smids = torch.empty(32, device="cuda", dtype=torch.int32)
    with torch.cuda.stream(stream):
        sm_blocker.block_sms(smids, 1000)
    torch.cuda.synchronize()
    original_empty, original_empty_like = torch.empty, torch.empty_like

    def poison(allocate):
        def empty(*args, **kwargs):
            tensor = allocate(*args, **kwargs)
            if tensor.is_cuda and tensor.is_floating_point():
                tensor.fill_(float("nan"))
            return tensor

        return empty

    for _ in range(3):
        with torch.cuda.stream(stream):
            sm_blocker.block_sms(smids, 500_000_000)
        time.sleep(0.02)
        try:
            with patch("torch.empty", poison(original_empty)), patch("torch.empty_like", poison(original_empty_like)):
                gradients = torch.autograd.grad(output, targets, dy, retain_graph=True)
        finally:
            torch.cuda.synchronize()
        for name, actual, expected in zip(names, gradients, reference):
            assert torch.isfinite(actual).all(), f"unwritten d{name}; occupied SMs: {smids.tolist()}"
            assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)), name
