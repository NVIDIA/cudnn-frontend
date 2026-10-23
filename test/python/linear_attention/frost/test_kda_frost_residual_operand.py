# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct FROST GPU checks of H/M residual packing and FP32 beta inputs."""

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error, current_sm

requirement_error = cutedsl_requirement_error("KDA residual operand tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

sm = current_sm()
if sm is None or not (100 <= sm <= 103 or sm == 107):
    pytest.skip("requires SM100 / SM103 / SM107 FROST KDA", allow_module_level=True)

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from cudnn.frost.tile_dsl.pointwise import f16x2_to_f32, fp32_to_fp16
from cudnn.linear_attention.frost.common.kda_residual import residual_operand

pytestmark = [pytest.mark.L1, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


@cute.kernel
def residual_probe(
    v: cute.Tensor,
    p: cute.Tensor,
    beta: cute.Tensor,
    out: cute.Tensor,
    dtype: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
    has_projection: cutlass.Constexpr,
):
    """Pack one residual pair per thread without using the full KDA pipeline."""
    index = cute.arch.thread_idx()[0] * 2
    word = fp32_to_fp16(v[index].to(cutlass.Float32), v[index + 1].to(cutlass.Float32), dtype=dtype)
    p0, p1 = None, None
    if cutlass.const_expr(has_projection):
        p0, p1 = p[index], p[index + 1]
    b0, b1 = None, None
    if cutlass.const_expr(has_beta):
        b0, b1 = beta[index], beta[index + 1]
    packed = residual_operand(word, p0, p1, b0, b1, dtype=dtype)
    o0, o1 = f16x2_to_f32(packed, dtype=dtype)
    out[index], out[index + 1] = o0.to(dtype), o1.to(dtype)


residual_probe.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def probe_host(
    v: cute.Tensor,
    p: cute.Tensor,
    beta: cute.Tensor,
    out: cute.Tensor,
    dtype: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
    has_projection: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    """Launch the direct packing probe on the caller's current CUDA stream."""
    residual_probe(v, p, beta, out, dtype, has_beta, has_projection).launch(grid=(1,), block=(128,), stream=stream)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("has_beta", [False, True], ids=["prep_residual", "scaled_residual"])
@pytest.mark.parametrize("has_projection", [False, True], ids=["zero_seed", "projected"])
@pytest.mark.parametrize("zero_v", [False, True], ids=["H", "M"])
def test_residual_operand_precision(dtype, has_beta, has_projection, zero_v):
    """Match FP32-then-BF16 or legacy FP16 packing for H/M operand variants."""
    generator = torch.Generator(device="cpu").manual_seed(1731)
    v = torch.randn(256, generator=generator).to(dtype)
    if zero_v:
        v.zero_()
    projection = v.float() + torch.randn(256, generator=generator) * 0.002
    beta = torch.rand(256, generator=generator) * 0.9 + 0.01
    if dtype == torch.bfloat16:
        expected = v.float() - projection if has_projection else v.float()
        if has_beta:
            expected = expected * beta
    else:
        expected = (v.float() - projection.to(dtype).float()).to(dtype) if has_projection else v
        if has_beta:
            expected = expected.float() * beta.to(dtype).float()
    expected = expected.to(dtype)
    v_gpu, p_gpu, b_gpu = v.cuda(), projection.cuda(), beta.cuda()
    out = torch.empty_like(v_gpu)
    tensors = [from_dlpack(t, assumed_align=16) for t in (v_gpu, p_gpu, b_gpu, out)]
    native_dtype = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(probe_host, *tensors, native_dtype, has_beta, has_projection, stream, options="--enable-tvm-ffi")
    compiled(v_gpu, p_gpu, b_gpu, out, stream)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
