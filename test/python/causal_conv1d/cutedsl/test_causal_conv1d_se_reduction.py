# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parameter-gradient reduction preserves ordered sums and strided outputs."""

import pytest
import torch
from determinism_utils import assert_bitwise, snapshot
from test_causal_conv1d_packed import _dsl

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


@pytest.mark.parametrize("channels,splits,bias", [(8, 1, False), (24, 33, True), (256, 64, True), (256, 81, True), (768, 98, False), (1032, 129, True)])
@pytest.mark.parametrize("strided", [False, True])
def test_parallel_bf16_reduction_preserves_all_output_bytes(channels, splits, bias, strided):
    cute, dlpack, cuda = _dsl()
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("the BF16 parallel reducer is selected on SM100, SM103, and SM107")
    from cudnn.causal_conv1d_se.packed_kernel import packed_parameter_reduce, packed_parameter_reduce_parallel

    gen = torch.Generator().manual_seed(47219)
    partials = (torch.randn((splits, 5, channels), generator=gen) * 0.25).cuda()
    # Alternating large values exercise cancellation without changing sum order.
    if splits > 1:
        partials[0, :, ::2] = 2**18
        partials[1, :, ::2] = -(2**18)
    dw = torch.empty((4, channels) if strided else (channels, 4), device="cuda", dtype=torch.bfloat16)
    if strided:
        dw = dw.T
    db = torch.empty((channels * 2 if strided else channels,), device="cuda", dtype=torch.bfloat16) if bias else None
    if strided and db is not None:
        db = db[::2]
    args = (partials, dw, db)
    views = tuple(None if t is None else dlpack(t, assumed_align=16) for t in args)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    @cute.jit
    def serial(p, w, b, stream: cuda.CUstream):
        packed_parameter_reduce(p, w, b, 4).launch(grid=(cute.ceil_div(w.shape[0], 128), 1, 1), block=(128, 1, 1), stream=stream)

    @cute.jit
    def parallel(p, w, b, stream: cuda.CUstream):
        packed_parameter_reduce_parallel(p, w, b, 4).launch(grid=(cute.ceil_div(w.shape[0], 128) * (5 if bias else 4), 1, 1), block=(128, 1, 1), stream=stream)

    old = cute.compile(serial, *views, stream, options="--enable-tvm-ffi")
    new = cute.compile(parallel, *views, stream, options="--enable-tvm-ffi")
    old(*args, stream)
    outputs = {"dw": dw, **({"db": db} if db is not None else {})}
    expected = snapshot(outputs)
    for _ in range(5):
        for t in outputs.values():
            t.fill_(float("nan"))
        new(*args, stream)
        assert all(torch.isfinite(t).all() for t in outputs.values())
        assert_bitwise(outputs, expected, "parallel versus serial parameter reduction")
