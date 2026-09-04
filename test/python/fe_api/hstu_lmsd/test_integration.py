# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forward-save to explicit-backward integration coverage for HSTU LMSD."""

import pytest
import torch

try:
    import cutlass  # noqa: F401
except (ImportError, OSError) as exc:
    pytest.skip(f"CuTe DSL is unavailable: {exc}", allow_module_level=True)

import cudnn.hstu_lmsd.ops as _ops
from cudnn.hstu_lmsd import hstu_lmsd_backward, hstu_lmsd_forward

pytestmark = [
    pytest.mark.gpu_exclusive,
    pytest.mark.xdist_group(name="gpu_exclusive"),
]

_IS_SM10X = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


def _layer_norm_reference(x, eps):
    xf = x.float()
    mean = xf.mean(dim=1)
    variance = (xf.square().mean(dim=1) - mean.square()).clamp_min(0.0)
    return mean, torch.rsqrt(variance + eps)


def _backward_reference(dy, x, u, weight, bias, mask, p, eps):
    d = x.shape[1]
    scale = 1.0 / (1.0 - p)
    xf = x.float()
    uf = u.float()
    wf = weight.float()
    bf = bias.float()
    dy_silu, dy_x, dy_lmsd = (part.float() for part in dy.split(d, dim=1))
    mask_i32 = mask.to(torch.int32)
    zero = torch.zeros((), device=x.device)
    direct_du = torch.where((mask_i32 & 4) != 0, dy_silu * scale, zero)
    direct_dx = torch.where((mask_i32 & 2) != 0, dy_x * scale, zero)
    fused_dy = torch.where((mask_i32 & 1) != 0, dy_lmsd * scale, zero)

    mean, rstd = _layer_norm_reference(x, eps)
    xhat = (xf - mean[:, None]) * rstd[:, None]
    ln = xhat * wf + bf
    sigmoid = torch.sigmoid(uf)
    silu = uf * sigmoid
    dsilu = sigmoid + silu * (1.0 - sigmoid)

    dln = fused_dy * silu
    dweight = torch.sum(dln * xhat, dim=0)
    dbias = torch.sum(dln, dim=0)
    weighted = dln * wf
    mean_weighted = torch.mean(weighted, dim=1, keepdim=True)
    mean_xhat_weighted = torch.mean(xhat * weighted, dim=1, keepdim=True)
    dx = direct_dx + (weighted - mean_weighted - xhat * mean_xhat_weighted) * rstd[:, None]
    du = (fused_dy * ln + direct_du) * dsilu
    return dx, du, dweight, dbias


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="HSTU LMSD requires SM10x")
def test_forward_saved_tensors_feed_explicit_backward():
    torch.manual_seed(2026)
    n, d, p, eps = 513, 512, 0.1, 1e-6
    x = torch.randn((n, d), device="cuda", dtype=torch.bfloat16)
    u_storage = torch.randn((n, 4 * d), device="cuda", dtype=torch.bfloat16)
    u = u_storage[:, :d]
    weight = torch.randn((d,), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((d,), device="cuda", dtype=torch.bfloat16)
    dy = torch.randn((n, 3 * d), device="cuda", dtype=torch.bfloat16)

    forward = hstu_lmsd_forward(x, u, weight, bias, eps=eps, dropout_ratio=p, seed=29)
    y, mean, rstd, mask = forward
    expected_mean, expected_rstd = _layer_norm_reference(x, eps)
    torch.testing.assert_close(mean, expected_mean, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(rstd, expected_rstd, rtol=2e-4, atol=2e-4)
    with pytest.raises(ValueError, match="must match the value used"):
        hstu_lmsd_backward(
            dy,
            x,
            u,
            weight,
            bias,
            mean,
            rstd,
            mask,
            dropout_ratio=0.2,
        )
    copied_mask = mask.clone()
    with pytest.raises(ValueError, match="dropout_ratio is required"):
        hstu_lmsd_backward(
            dy,
            x,
            u,
            weight,
            bias,
            mean,
            rstd,
            copied_mask,
        )
    copied_mask_actual = hstu_lmsd_backward(
        dy,
        x,
        u,
        weight,
        bias,
        mean,
        rstd,
        copied_mask,
        dropout_ratio=p,
    )
    actual = hstu_lmsd_backward(
        dy,
        x,
        u,
        weight,
        bias,
        mean,
        rstd,
        mask,
    )
    expected = _backward_reference(dy, x, u, weight, bias, mask, p, eps)
    tolerances = (
        (2.5e-2, 2.5e-2),
        (2.5e-2, 2.5e-2),
        (3.5e-2, 5.0e-1),
        (3.5e-2, 5.0e-1),
    )
    for got, ref, (rtol, atol) in zip(actual, expected, tolerances):
        torch.testing.assert_close(got.float(), ref, rtol=rtol, atol=atol)
    for got, ref, (rtol, atol) in zip(copied_mask_actual, expected, tolerances):
        torch.testing.assert_close(got.float(), ref, rtol=rtol, atol=atol)


@pytest.mark.L1
@pytest.mark.skipif(not _IS_SM10X, reason="HSTU LMSD requires SM10x")
def test_wrappers_reuse_compiled_binaries_across_dynamic_n(monkeypatch):
    """Different positive N values must rebind one plan-time compile per direction.

    This is intentionally a compile-count and correctness regression, not a timing
    benchmark. Before dynamic-N cache keys, the second shape creates a second API
    object and this test fails with two compile calls and two cache entries.
    """
    original_fwd_compile = _ops.HSTULMSDFwdSm100.compile
    original_bwd_compile = _ops.HSTULMSDBwdSm100.compile
    compile_calls = {"forward": 0, "backward": 0}

    def counted_fwd_compile(self):
        compile_calls["forward"] += 1
        return original_fwd_compile(self)

    def counted_bwd_compile(self):
        compile_calls["backward"] += 1
        return original_bwd_compile(self)

    monkeypatch.setattr(_ops.HSTULMSDFwdSm100, "compile", counted_fwd_compile)
    monkeypatch.setattr(_ops.HSTULMSDBwdSm100, "compile", counted_bwd_compile)
    _ops._FWD_CACHE.clear()
    _ops._BWD_CACHE.clear()

    first_fwd_api = first_bwd_api = None
    first_fwd_binary = first_bwd_binary = None
    try:
        for case, n in enumerate((37, 513)):
            torch.manual_seed(3100 + n)
            d, p, eps = 512, 0.1, 1e-6
            x = torch.randn((n, d), device="cuda", dtype=torch.bfloat16)
            u_storage = torch.randn((n, 4 * d), device="cuda", dtype=torch.bfloat16)
            u = u_storage[:, :d]
            weight = torch.randn((d,), device="cuda", dtype=torch.bfloat16)
            bias = torch.randn((d,), device="cuda", dtype=torch.bfloat16)

            forward = hstu_lmsd_forward(
                x,
                u,
                weight,
                bias,
                eps=eps,
                dropout_ratio=p,
                seed=91 + case,
            )
            y, mean, rstd, mask = forward
            assert y.shape == (n, 3 * d)
            assert mean.shape == rstd.shape == (n,)
            assert mask.shape == (n, d)

            expected_mean, expected_rstd = _layer_norm_reference(x, eps)
            torch.testing.assert_close(mean, expected_mean, rtol=2e-4, atol=2e-4)
            torch.testing.assert_close(rstd, expected_rstd, rtol=2e-4, atol=2e-4)
            mask_i32 = mask.to(torch.int32)
            assert torch.count_nonzero(mask_i32 & ~0x7) == 0
            scale = 1.0 / (1.0 - p)
            zero = torch.zeros((), device=x.device)
            xf = x.float()
            expected_mean = xf.mean(dim=1)
            expected_var = (xf.square().mean(dim=1) - expected_mean.square()).clamp_min(0.0)
            expected_rstd = torch.rsqrt(expected_var + 1e-6)
            torch.testing.assert_close(mean, expected_mean, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(rstd, expected_rstd, rtol=1e-4, atol=1e-4)
            silu = torch.nn.functional.silu(u.float())
            ln = (xf - expected_mean[:, None]) * expected_rstd[:, None]
            ln = ln * weight.float() + bias.float()
            expected_y = torch.cat(
                (
                    torch.where((mask_i32 & 4) != 0, silu * scale, zero),
                    torch.where((mask_i32 & 2) != 0, xf * scale, zero),
                    torch.where((mask_i32 & 1) != 0, ln * silu * scale, zero),
                ),
                dim=1,
            )
            torch.testing.assert_close(y.float(), expected_y, rtol=1.5e-2, atol=1.5e-2)

            dy = torch.randn_like(y)
            actual = hstu_lmsd_backward(
                dy,
                x,
                u,
                weight,
                bias,
                mean,
                rstd,
                mask,
                dropout_ratio=p,
            )
            expected = _backward_reference(dy, x, u, weight, bias, mask, p, eps)
            assert actual["dx_tensor"].shape == (n, d)
            assert actual["du_tensor"].shape == (n, d)
            assert actual["dweight_tensor"].shape == (d,)
            assert actual["dbias_tensor"].shape == (d,)
            tolerances = (
                (2.5e-2, 2.5e-2),
                (2.5e-2, 2.5e-2),
                (3.5e-2, 5.0e-1),
                (3.5e-2, 5.0e-1),
            )
            for got, ref, (rtol, atol) in zip(actual, expected, tolerances):
                torch.testing.assert_close(got.float(), ref, rtol=rtol, atol=atol)

            assert len(_ops._FWD_CACHE) == 1
            assert len(_ops._BWD_CACHE) == 1
            fwd_api = next(iter(_ops._FWD_CACHE.values()))
            bwd_api = next(iter(_ops._BWD_CACHE.values()))
            if case == 0:
                first_fwd_api = fwd_api
                first_bwd_api = bwd_api
                first_fwd_binary = fwd_api._compiled_kernel
                first_bwd_binary = bwd_api._compiled_kernel
            else:
                assert fwd_api is first_fwd_api
                assert bwd_api is first_bwd_api
                assert fwd_api._compiled_kernel is first_fwd_binary
                assert bwd_api._compiled_kernel is first_bwd_binary

        assert compile_calls == {"forward": 1, "backward": 1}
    finally:
        _ops._FWD_CACHE.clear()
        _ops._BWD_CACHE.clear()


@pytest.mark.L1
@pytest.mark.skipif(not _IS_SM10X, reason="HSTU LMSD requires SM10x")
def test_wrappers_use_tensor_device_and_custom_stream():
    """Compile and launch on the tensor device even when another device is current."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible CUDA devices")

    original_device = torch.cuda.current_device()
    target_device = (original_device + 1) % torch.cuda.device_count()
    if torch.cuda.get_device_capability(target_device)[0] != 10:
        pytest.skip("target device is not SM10x")

    _ops._FWD_CACHE.clear()
    _ops._BWD_CACHE.clear()
    try:
        with torch.cuda.device(target_device):
            torch.manual_seed(909)
            n, d = 37, 512
            x = torch.randn((n, d), device=target_device, dtype=torch.bfloat16)
            u_storage = torch.randn((n, 4 * d), device=target_device, dtype=torch.bfloat16)
            u = u_storage[:, :d]
            weight = torch.randn((d,), device=target_device, dtype=torch.bfloat16)
            bias = torch.randn((d,), device=target_device, dtype=torch.bfloat16)
            dy = torch.randn((n, 3 * d), device=target_device, dtype=torch.bfloat16)
            stream = torch.cuda.Stream(device=target_device)
            torch.cuda.synchronize(target_device)

        with torch.cuda.device(original_device):
            forward = hstu_lmsd_forward(
                x,
                u,
                weight,
                bias,
                eps=1e-6,
                dropout_ratio=0.1,
                seed=17,
                stream=stream,
            )
            y, mean, rstd, mask = forward
            backward = hstu_lmsd_backward(
                dy,
                x,
                u,
                weight,
                bias,
                mean,
                rstd,
                mask,
                stream=stream,
            )
            assert torch.cuda.current_device() == original_device

        stream.synchronize()
        for tensor in (*forward, *backward):
            assert tensor.device.index == target_device
            assert torch.isfinite(tensor).all()
    finally:
        _ops._FWD_CACHE.clear()
        _ops._BWD_CACHE.clear()
