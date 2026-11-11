# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fresh outputs and AMAX initialization belong to the caller's launch stream."""

import pytest
import torch
from cuda.bindings import driver as cuda
from gemm.cutedsl.test_grouped_swiglu_sfd_tiles import _operands, api

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("target", ["side", "default"])
def test_swiglu_wrapper_outputs_and_amax_use_launch_stream(api, monkeypatch, warm, target):
    api._cache_of_GroupedGemmSwigluSm100Objects.clear()
    inputs, counts, _, _ = _operands(4, 512, True)
    launch = torch.cuda.Stream() if target == "side" else torch.cuda.default_stream()
    ambient = torch.cuda.Stream()
    kwargs = dict(
        **inputs,
        sf_vec_size=32,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        mma_tiler_mn=(256, 128),
        cluster_shape_mn=(2, 1),
        m_aligned=256,
        current_stream=cuda.CUstream(launch.cuda_stream),
    )
    torch.cuda.synchronize()
    if warm:
        api.grouped_gemm_swiglu_wrapper_sm100(**kwargs)
        torch.cuda.synchronize()
    # Keep this storage live throughout. A mocked allocator removes cudaMalloc
    # synchronization from the probe but retains the real per-call fill kernel.
    amax_storage = torch.full((len(counts), 1), float("inf"), dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    allocations = []
    original_empty, original_empty_strided, original_full = torch.empty, torch.empty_strided, torch.full

    def record(name):
        stream = torch.cuda.current_stream(inputs["a_tensor"].device)
        allocations.append((name, stream.cuda_stream, stream.device))

    def empty(*args, **kw):
        record("empty")
        return original_empty(*args, **kw)

    def empty_strided(*args, **kw):
        record("empty_strided")
        return original_empty_strided(*args, **kw)

    def full(size, fill_value, **kw):
        record("full")
        if tuple(size) == tuple(amax_storage.shape) and fill_value == -float("inf") and kw.get("dtype") == torch.float32:
            amax_storage.fill_(fill_value)
            return amax_storage
        return original_full(size, fill_value, **kw)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", empty)
        patch.setattr(torch, "empty_strided", empty_strided)
        patch.setattr(torch, "full", full)
        with torch.cuda.stream(ambient):
            if warm:
                # Bounded delay on a separate stream makes a misplaced AMAX
                # reset observable as a real output error, not a timing guess.
                torch.cuda._sleep(100_000_000)
            result = api.grouped_gemm_swiglu_wrapper_sm100(**kwargs)
            assert torch.cuda.current_stream().cuda_stream == ambient.cuda_stream
    torch.cuda.synchronize()
    assert allocations and any(name == "full" for name, _, _ in allocations)
    live = torch.tensor([n > 0 for n in counts], device="cuda")
    assert torch.isfinite(result["amax_tensor"][live]).all()
    assert (result["amax_tensor"][live] > 0).all()
    assert torch.isneginf(result["amax_tensor"][~live]).all()
    assert all(handle == launch.cuda_stream and device == launch.device for _, handle, device in allocations), allocations
