# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The combine's final outputs honor strides independently of compact partials."""

import math

import pytest
import torch

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]


def _partials(b, h, sq, d, splits=3):
    generator = torch.Generator().manual_seed(20260919)
    o = torch.randn(splits, b, sq, h, d, generator=generator) * 0.4
    lse = torch.randn(splits, b, h, sq, generator=generator) * 2.0 + 0.7
    lse[:, :, :, 0] = -torch.inf  # Every split dead on one row.
    lse[1, :, :, 1] = -torch.inf  # A dead split next to two live splits.
    dead = torch.isneginf(lse).permute(0, 1, 3, 2).unsqueeze(-1)
    o.masked_fill_(dead, torch.nan)  # Dead payloads must never enter arithmetic.
    scores = lse.double()
    maximum = scores.amax(0)
    safe_maximum = torch.where(torch.isfinite(maximum), maximum, 0)
    weights = (scores - safe_maximum).exp()
    denominator = weights.sum(0)
    safe_o = o.double().masked_fill(dead, 0)
    ref_o = (weights.permute(0, 1, 3, 2).unsqueeze(-1) * safe_o).sum(0) / denominator.clamp_min(1).permute(0, 2, 1).unsqueeze(-1)
    ref_lse = safe_maximum + denominator.log()
    return o.reshape(splits * b, sq, h, d).cuda(), lse.reshape(splits * b, h, sq).cuda(), ref_o, ref_lse


def _output(shape, strides, dtype):
    span = 1 + sum((n - 1) * stride for n, stride in zip(shape, strides))
    storage = torch.full((span + 32,), -31.0, device="cuda", dtype=torch.float32).to(dtype)
    view = storage.as_strided(shape, strides, storage_offset=16)
    offsets = torch.zeros(shape, dtype=torch.int64)
    for axis, (n, stride) in enumerate(zip(shape, strides)):
        dims = [1] * len(shape)
        dims[axis] = n
        offsets += torch.arange(n).view(dims) * stride
    occupied = torch.zeros(storage.numel(), dtype=torch.bool)
    occupied[offsets.flatten() + 16] = True
    return view, storage, occupied


def _strides(b, h, sq, d, layout):
    if layout == "compact":
        return (sq * h * d, h * d, d, 1), (h * sq, sq, 1)
    head = d * 2 + 8
    token = h * head + 16
    batch = sq * token + 32
    lse_head = sq * 2 + 3
    lse_batch = h * lse_head + 5
    if layout == "int64_singleton":
        assert b == 1
        batch, lse_batch = 2**33 + 17, 2**34 + 19
    return (batch, token, head, 2), (lse_batch, lse_head, 2)


def _check(o, lse, o_storage, o_used, lse_storage, lse_used, ref_o, ref_lse, stats_log2):
    torch.cuda.synchronize()
    torch.testing.assert_close(o.cpu().float(), ref_o.float(), atol=0.004, rtol=0.004)
    assert torch.count_nonzero(o[:, 0]).item() == 0
    assert torch.all(o_storage.float().cpu()[~o_used] == -31.0)
    if lse is not None:
        if stats_log2:
            ref_lse = ref_lse * math.log2(math.e)
        torch.testing.assert_close(lse.cpu().double(), ref_lse, atol=2e-5, rtol=2e-5)
        assert torch.all(torch.isneginf(lse[:, :, 0]))
        assert torch.all(lse_storage.cpu()[~lse_used] == -31.0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("stats", ["none", "ln", "log2"])
@pytest.mark.parametrize("layout", ["compact", "strided", "int64_singleton"])
def test_pointer_combine_strided_outputs_and_dead_splits(dtype, stats, layout):
    from cudnn.frost.compiled_cache import positional_entry
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d, splits = (1 if layout == "int64_singleton" else 2), 3, 5, 160, 3
    op, lp, ref_o, ref_lse = _partials(b, h, sq, d, splits)
    ostride, lstride = _strides(b, h, sq, d, layout)
    o, ostorage, oused = _output((b, sq, h, d), ostride, dtype)
    lse, lstorage, lused = _output((b, h, sq), lstride, torch.float32) if stats != "none" else (None, None, None)
    owner = comb.compile_ptr(dtype_o="f16" if dtype == torch.float16 else "bf16", has_lse=lse is not None, stats_log2=stats == "log2")
    fn = positional_entry(owner)
    assert fn is not None
    fn(
        op.data_ptr(),
        lp.data_ptr(),
        o.data_ptr(),
        lse.data_ptr() if lse is not None else None,
        (b, h, sq, d),
        splits,
        ostride,
        lstride,
        torch.cuda.current_stream().cuda_stream,
    )
    _check(o, lse, ostorage, oused, lstorage, lused, ref_o, ref_lse, stats == "log2")


def test_pointer_combine_reuses_artifact_for_new_shapes():
    from cudnn.frost.compiled_cache import positional_entry
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    owner = comb.compile_ptr(dtype_o="bf16", has_lse=True)
    fn = positional_entry(owner)
    assert fn is not None
    for b, h, sq, d, splits in [(2, 3, 5, 160, 3), (1, 2, 3, 256, 5)]:
        op, lp, ref_o, ref_lse = _partials(b, h, sq, d, splits)
        ostride, lstride = _strides(b, h, sq, d, "strided")
        o, ostorage, oused = _output((b, sq, h, d), ostride, torch.bfloat16)
        lse, lstorage, lused = _output((b, h, sq), lstride, torch.float32)
        fn(op.data_ptr(), lp.data_ptr(), o.data_ptr(), lse.data_ptr(), (b, h, sq, d), splits, ostride, lstride, torch.cuda.current_stream().cuda_stream)
        _check(o, lse, ostorage, oused, lstorage, lused, ref_o, ref_lse, False)


@pytest.mark.parametrize("layout", ["compact", "strided", "int64_singleton"])
def test_tensor_combine_strided_output(layout):
    import cutlass
    import cuda.bindings.driver as cuda_driver

    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d, splits = (1 if layout == "int64_singleton" else 2), 3, 5, 160, 3
    op, lp, ref_o, ref_lse = _partials(b, h, sq, d, splits)
    ostride, lstride = _strides(b, h, sq, d, layout)
    o, ostorage, oused = _output((b, sq, h, d), ostride, torch.float16)
    lse, lstorage, lused = _output((b, h, sq), lstride, torch.float32)
    fn = comb.compile(b, h, sq, d, splits, has_lse=True, lse_stride=lstride, dtype_partial="f32", o_stride=ostride)
    fn(op, lp, o, lse, None, None, (b, h, sq, d), cutlass.Int32(splits), stream=cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream))
    _check(o, lse, ostorage, oused, lstorage, lused, ref_o, ref_lse, False)


@pytest.mark.parametrize("dtype,tag", [(torch.float8_e4m3fn, "e4m3"), (torch.float8_e5m2, "e5m2")])
def test_quantized_tensor_combine_keeps_scale_and_amax(dtype, tag):
    import cutlass
    import cuda.bindings.driver as cuda_driver

    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d, splits = 2, 3, 5, 160, 3
    op, lp, ref_o, ref_lse = _partials(b, h, sq, d, splits)
    # Existing quantized callers keep the compact ABI (no o_stride argument).
    o = torch.empty((b, sq, h, d), device="cuda", dtype=dtype)
    lse = torch.full((b, h, sq), torch.nan, device="cuda")
    amax = torch.zeros(1, dtype=torch.float32, device="cuda")
    scale = torch.tensor([1.75], dtype=torch.float32, device="cuda")
    fn = comb.compile(b, h, sq, d, splits, dtype_o=tag, has_lse=True, has_amax=True, dtype_partial="f32", has_scale_o=True, stats_log2=True)
    fn(op, lp, o, lse, amax, scale, (b, h, sq, d), cutlass.Int32(splits), stream=cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()
    expected = (ref_o * 1.75).float().to(dtype).float()
    # A last-bit pre-quant reduction difference may cross one quantization midpoint.
    torch.testing.assert_close(o.cpu().float(), expected, atol=0.03125, rtol=0.01)
    torch.testing.assert_close(amax.cpu(), ref_o.abs().amax().float().view(1), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(lse.cpu().double(), ref_lse * math.log2(math.e), atol=2e-5, rtol=2e-5)
    assert torch.count_nonzero(o[:, 0].float()).item() == 0
