# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forward-to-backward integration coverage for HSTU LMSD."""

from dataclasses import replace

import pytest
import torch

try:
    import cutlass  # noqa: F401
except (ImportError, OSError) as exc:
    pytest.skip(f"CuTe DSL is unavailable: {exc}", allow_module_level=True)

import cudnn.hstu.hstu_lmsd.ops as _ops
from cudnn.hstu.hstu_lmsd import hstu_lmsd_backward, hstu_lmsd_forward
from reference import hstu_lmsd_backward_reference, hstu_lmsd_forward_reference, layer_norm_stats

pytestmark = [
    pytest.mark.gpu_exclusive,
    pytest.mark.xdist_group(name="gpu_exclusive"),
]

_IS_SM10X = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


@pytest.mark.L0
def test_backward_requires_explicit_forward_configuration():
    x = torch.empty((1, 8), dtype=torch.bfloat16)
    u = torch.empty_like(x)
    weight = torch.empty((8,), dtype=torch.bfloat16)
    bias = torch.empty_like(weight)
    mean = torch.empty((1,), dtype=torch.float32)
    rstd = torch.empty_like(mean)
    mask = torch.empty_like(x, dtype=torch.int8)
    dy = torch.empty((1, 24), dtype=torch.bfloat16)

    for missing in ("dropout_ratio", "apply_u_silu", "concat_u", "concat_x"):
        config = {
            "dropout_ratio": 0.1,
            "apply_u_silu": True,
            "concat_u": True,
            "concat_x": True,
        }
        del config[missing]
        with pytest.raises(ValueError, match=f"explicit forward configuration:.*{missing}"):
            hstu_lmsd_backward(dy, x, u, weight, bias, mean, rstd, mask, **config)


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="HSTU LMSD requires SM10x")
@pytest.mark.parametrize("d", (8, 24, 128, 264, 512, 1016))
def test_forward_outputs_feed_explicit_backward(d):
    torch.manual_seed(2026)
    n, p, eps = 513, 0.1, 1e-6
    x_storage = torch.randn((n, 3 * d), device="cuda", dtype=torch.bfloat16)
    x = x_storage[:, :d]
    u_storage = torch.randn((n, 4 * d), device="cuda", dtype=torch.bfloat16)
    u = u_storage[:, :d]
    weight = torch.randn((d,), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((d,), device="cuda", dtype=torch.bfloat16)
    dy = torch.randn((n, 3 * d), device="cuda", dtype=torch.bfloat16)

    forward = hstu_lmsd_forward(x, u, weight, bias, eps=eps, dropout_ratio=p, seed=29)
    y, mean, rstd, mask = forward
    expected_mean, expected_rstd = layer_norm_stats(x, eps)
    torch.testing.assert_close(mean, expected_mean, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(rstd, expected_rstd, rtol=2e-4, atol=2e-4)
    actual = hstu_lmsd_backward(
        dy,
        x,
        u,
        weight,
        bias,
        mean.clone(),
        rstd.clone(),
        mask.clone(),
        dropout_ratio=p,
        apply_u_silu=True,
        concat_u=True,
        concat_x=True,
    )
    expected = hstu_lmsd_backward_reference(dy, x, u, weight, bias, mask, p, eps)
    tolerances = (
        (2.5e-2, 2.5e-2),
        (2.5e-2, 2.5e-2),
        (3.5e-2, 5.0e-1),
        (3.5e-2, 5.0e-1),
    )
    for got, ref, (rtol, atol) in zip(actual, expected, tolerances):
        torch.testing.assert_close(got.float(), ref, rtol=rtol, atol=atol)


@pytest.mark.L1
@pytest.mark.skipif(not _IS_SM10X, reason="HSTU LMSD requires SM10x")
@pytest.mark.parametrize(
    "d,p,apply_u_silu,concat_u,concat_x,compute_dweight",
    (
        (8, 0.0, False, False, False, False),
        (256, 0.2, True, True, False, False),
        (1016, 0.2, False, False, True, True),
    ),
)
def test_optional_forward_backward_configurations(d, p, apply_u_silu, concat_u, concat_x, compute_dweight):
    torch.manual_seed(4701)
    n, eps = 129, 1e-6
    x = torch.randn((n, d), device="cuda", dtype=torch.bfloat16)
    u = torch.randn_like(x)
    weight = torch.randn((d,), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((d,), device="cuda", dtype=torch.bfloat16)

    forward = hstu_lmsd_forward(
        x,
        u,
        weight,
        bias,
        eps=eps,
        dropout_ratio=p,
        seed=71,
        apply_u_silu=apply_u_silu,
        concat_u=concat_u,
        concat_x=concat_x,
    )
    y, mean, rstd, mask = forward
    assert y.shape == (n, (1 + int(concat_u) + int(concat_x)) * d)
    assert (mask is not None) == (p > 0.0)

    expected_y, expected_mean, expected_rstd = hstu_lmsd_forward_reference(
        x,
        u,
        weight,
        bias,
        mask,
        p,
        eps,
        apply_u_silu=apply_u_silu,
        concat_u=concat_u,
        concat_x=concat_x,
    )
    torch.testing.assert_close(y.float(), expected_y, rtol=1.5e-2, atol=1.5e-2)
    torch.testing.assert_close(mean, expected_mean, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(rstd, expected_rstd, rtol=2e-4, atol=2e-4)

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
        apply_u_silu=apply_u_silu,
        concat_u=concat_u,
        concat_x=concat_x,
        compute_dweight=compute_dweight,
    )
    expected = hstu_lmsd_backward_reference(
        dy,
        x,
        u,
        weight,
        bias,
        mask,
        p,
        eps,
        apply_u_silu=apply_u_silu,
        concat_u=concat_u,
        concat_x=concat_x,
        compute_dweight=compute_dweight,
    )
    assert (actual["dweight_tensor"] is not None) == compute_dweight
    tolerances = ((2.5e-2, 2.5e-2), (2.5e-2, 2.5e-2), (3.5e-2, 5.0e-1), (3.5e-2, 5.0e-1))
    for got, ref, (rtol, atol) in zip(actual, expected, tolerances):
        if ref is None:
            assert got is None
        else:
            torch.testing.assert_close(got.float(), ref, rtol=rtol, atol=atol)


@pytest.mark.L1
@pytest.mark.skipif(not _IS_SM10X, reason="HSTU LMSD requires SM10x")
def test_wrappers_reuse_compiled_binaries_across_dynamic_n_and_row_strides(monkeypatch):
    """Runtime N and padded row strides must rebind one compile per direction."""
    original_fwd_compile = _ops.HSTULMSDFwd.compile
    original_bwd_compile = _ops.HSTULMSDBwd.compile
    compile_calls = {"forward": 0, "backward": 0}

    def counted_fwd_compile(self):
        compile_calls["forward"] += 1
        return original_fwd_compile(self)

    def counted_bwd_compile(self):
        compile_calls["backward"] += 1
        return original_bwd_compile(self)

    monkeypatch.setattr(_ops.HSTULMSDFwd, "compile", counted_fwd_compile)
    monkeypatch.setattr(_ops.HSTULMSDBwd, "compile", counted_bwd_compile)
    _ops._FWD_CACHE.clear()
    _ops._BWD_CACHE.clear()

    first_fwd_api = first_bwd_api = None
    first_fwd_binary = first_bwd_binary = None
    try:
        cases = (
            (37, 512, 2048, 1536),
            (513, 1536, 2560, 2048),
        )
        for case, (n, x_row_stride, u_row_stride, dy_row_stride) in enumerate(cases):
            torch.manual_seed(3100 + n)
            d, p, eps = 512, 0.1, 1e-6
            x_storage = torch.randn((n, x_row_stride), device="cuda", dtype=torch.bfloat16)
            x = x_storage[:, :d]
            u_storage = torch.randn((n, u_row_stride), device="cuda", dtype=torch.bfloat16)
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

            expected_mean, expected_rstd = layer_norm_stats(x, eps)
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

            dy_storage = torch.randn((n, dy_row_stride), device="cuda", dtype=torch.bfloat16)
            dy = dy_storage[:, : 3 * d]
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
                apply_u_silu=True,
                concat_u=True,
                concat_x=True,
            )
            expected = hstu_lmsd_backward_reference(dy, x, u, weight, bias, mask, p, eps)
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
                assert tuple(fwd_api.x_desc.stride) != tuple(x.stride())
                assert tuple(fwd_api.u_desc.stride) != tuple(u.stride())
                assert tuple(bwd_api.dy_desc.stride) != tuple(dy.stride())

        assert compile_calls == {"forward": 1, "backward": 1}
    finally:
        _ops._FWD_CACHE.clear()
        _ops._BWD_CACHE.clear()


@pytest.mark.L1
@pytest.mark.skipif(not _IS_SM10X, reason="HSTU LMSD requires SM10x")
def test_forward_persistent_loop_matches_default_grid():
    """A one-block grid exercises repeated persistent-loop iterations."""
    _ops._FWD_CACHE.clear()
    try:
        torch.manual_seed(812)
        n, d = 37, 512
        x = torch.randn((n, d), device="cuda", dtype=torch.bfloat16)
        u_storage = torch.randn((n, 4 * d), device="cuda", dtype=torch.bfloat16)
        u = u_storage[:, :d]
        weight = torch.randn((d,), device="cuda", dtype=torch.bfloat16)
        bias = torch.randn((d,), device="cuda", dtype=torch.bfloat16)

        expected = hstu_lmsd_forward(x, u, weight, bias, dropout_ratio=0.1, seed=43)
        api = next(iter(_ops._FWD_CACHE.values()))
        api._multiprocessor_count = 1
        api._kernel_config = replace(api._kernel_config, grid_ctas_per_sm=1)
        actual = hstu_lmsd_forward(x, u, weight, bias, dropout_ratio=0.1, seed=43)
        torch.cuda.synchronize()

        assert len(_ops._FWD_CACHE) == 1
        for expected_tensor, actual_tensor in zip(expected, actual):
            assert torch.equal(expected_tensor, actual_tensor)
    finally:
        _ops._FWD_CACHE.clear()


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
                dropout_ratio=0.1,
                apply_u_silu=True,
                concat_u=True,
                concat_x=True,
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
