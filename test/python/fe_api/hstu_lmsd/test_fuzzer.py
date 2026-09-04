# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shape and dropout fuzz coverage for HSTU LMSD forward."""

import pytest
import torch

try:
    import cutlass  # noqa: F401
except (ImportError, OSError) as exc:
    pytest.skip(f"CuTe DSL is unavailable: {exc}", allow_module_level=True)

from cudnn.hstu_lmsd import hstu_lmsd_backward, hstu_lmsd_forward
from cudnn.hstu_lmsd.cutedsl import cute_dsl_ln_mul_dropout as _standalone_fwd
from cudnn.hstu_lmsd.cutedsl import cute_dsl_ln_mul_dropout_bwd as _standalone_bwd

pytestmark = [
    pytest.mark.gpu_exclusive,
    pytest.mark.xdist_group(name="gpu_exclusive"),
]

_IS_SM10X = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


@pytest.mark.L1
@pytest.mark.skipif(not _IS_SM10X, reason="HSTU LMSD requires SM10x")
@pytest.mark.parametrize(
    "n,dropout_ratio,seed",
    ((1, 0.0, 0), (37, 0.1, 7), (1025, 0.35, 2**40 + 9)),
)
def test_forward_matches_mask_reconstruction(n, dropout_ratio, seed):
    torch.manual_seed(11 + n)
    d = 512
    x = torch.randn((n, d), device="cuda", dtype=torch.bfloat16)
    u_storage = torch.randn((n, 4 * d), device="cuda", dtype=torch.bfloat16)
    u = u_storage[:, :d]
    weight = torch.randn((d,), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((d,), device="cuda", dtype=torch.bfloat16)

    eps = 1e-6
    result = hstu_lmsd_forward(
        x,
        u,
        weight,
        bias,
        eps=eps,
        dropout_ratio=dropout_ratio,
        seed=seed,
    )
    y, mean, rstd, mask = result
    mask_i32 = mask.to(torch.int32)
    assert torch.count_nonzero(mask_i32 & ~0x7) == 0

    xf = x.float()
    uf = u.float()
    expected_mean = xf.mean(dim=1)
    expected_var = (xf.square().mean(dim=1) - expected_mean.square()).clamp_min(0.0)
    expected_rstd = torch.rsqrt(expected_var + eps)
    torch.testing.assert_close(mean, expected_mean, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(rstd, expected_rstd, rtol=2e-4, atol=2e-4)

    normalized = (xf - expected_mean[:, None]) * expected_rstd[:, None]
    ln = normalized * weight.float() + bias.float()
    silu = torch.nn.functional.silu(uf)
    scale = 1.0 / (1.0 - dropout_ratio)
    zero = torch.zeros((), device=x.device)
    expected = torch.cat(
        (
            torch.where((mask_i32 & 4) != 0, silu * scale, zero),
            torch.where((mask_i32 & 2) != 0, xf * scale, zero),
            torch.where((mask_i32 & 1) != 0, ln * silu * scale, zero),
        ),
        dim=1,
    )
    torch.testing.assert_close(y.float(), expected, rtol=1.5e-2, atol=1.5e-2)


@pytest.mark.L1
@pytest.mark.skipif(not _IS_SM10X, reason="HSTU LMSD requires SM10x")
def test_standalone_forward_training_false_disables_dropout():
    """The standalone compatibility path must honor its training flag.

    Before the training=False fix, the supplied nonzero dropout ratio still
    produced dropped elements, so the all-kept mask assertion is RED.
    """
    torch.manual_seed(404)
    n, d = 17, 512
    x = torch.randn((n, d), device="cuda", dtype=torch.bfloat16)
    u = torch.randn((n, d), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((d,), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((d,), device="cuda", dtype=torch.bfloat16)

    eps = 1e-6
    y, mean, rstd, mask = _standalone_fwd.ln_mul_dropout_fwd(
        x,
        u,
        weight,
        bias,
        eps=eps,
        dropout_ratio=0.75,
        training=False,
        seed=99,
    )

    assert torch.all(mask == 0x7)
    xf = x.float()
    expected_mean = xf.mean(dim=1)
    expected_var = (xf.square().mean(dim=1) - expected_mean.square()).clamp_min(0.0)
    expected_rstd = torch.rsqrt(expected_var + eps)
    torch.testing.assert_close(mean, expected_mean, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(rstd, expected_rstd, rtol=2e-4, atol=2e-4)
    silu = torch.nn.functional.silu(u.float())
    ln = (xf - expected_mean[:, None]) * expected_rstd[:, None]
    ln = ln * weight.float() + bias.float()
    expected = torch.cat((silu, xf, ln * silu), dim=1)
    torch.testing.assert_close(y.float(), expected, rtol=1.5e-2, atol=1.5e-2)


@pytest.mark.L1
@pytest.mark.skipif(not _IS_SM10X, reason="HSTU LMSD requires SM10x")
def test_standalone_helpers_reuse_compiled_binaries_across_dynamic_n():
    """Compatibility helpers must not compile a new binary for each runtime N."""
    _standalone_fwd._COMPILED.clear()
    _standalone_bwd._COMPILED.clear()
    first_fwd_binary = first_bwd_binary = None
    try:
        for case, n in enumerate((17, 37)):
            torch.manual_seed(700 + case)
            d = 512
            x = torch.randn((n, d), device="cuda", dtype=torch.bfloat16)
            u_storage = torch.randn((n, 4 * d), device="cuda", dtype=torch.bfloat16)
            u = u_storage[:, :d]
            weight = torch.randn((d,), device="cuda", dtype=torch.bfloat16)
            bias = torch.randn((d,), device="cuda", dtype=torch.bfloat16)
            y, mean, rstd, mask = _standalone_fwd.ln_mul_dropout_fwd(
                x,
                u,
                weight,
                bias,
                eps=1e-6,
                dropout_ratio=0.1,
                training=True,
                seed=41 + case,
            )
            dy = torch.randn_like(y)
            dx, du, recomputed_y, dw_partial, db_partial = _standalone_bwd.ln_mul_dropout_bwd(
                dy,
                x,
                u,
                weight,
                bias,
                mean,
                rstd,
                mask,
                dropout_ratio=0.1,
                compute_y=False,
            )
            assert dx.shape == du.shape == (n, d)
            assert recomputed_y.numel() == 0
            assert dw_partial.shape == db_partial.shape == (_standalone_bwd.TARGET_TILES, d)
            reference_grads = hstu_lmsd_backward(
                dy,
                x,
                u,
                weight,
                bias,
                mean,
                rstd,
                mask,
                dropout_ratio=0.1,
            )
            assert torch.equal(dx, reference_grads["dx_tensor"])
            assert torch.equal(du, reference_grads["du_tensor"])
            assert len(_standalone_fwd._COMPILED) == 1
            assert len(_standalone_bwd._COMPILED) == 1

            fwd_binary = next(iter(_standalone_fwd._COMPILED.values()))
            bwd_binary = next(iter(_standalone_bwd._COMPILED.values()))
            if case == 0:
                first_fwd_binary = fwd_binary
                first_bwd_binary = bwd_binary
            else:
                assert fwd_binary is first_fwd_binary
                assert bwd_binary is first_bwd_binary

        _, _, _, eval_mask = _standalone_fwd.ln_mul_dropout_fwd(
            x,
            u,
            weight,
            bias,
            eps=1e-6,
            dropout_ratio=0.75,
            training=False,
            seed=99,
        )
        assert len(_standalone_fwd._COMPILED) == 1
        assert torch.all(eval_mask == 0x7)

        _, _, full_y, _, _ = _standalone_bwd.ln_mul_dropout_bwd(
            dy,
            x,
            u,
            weight,
            bias,
            mean,
            rstd,
            mask,
            dropout_ratio=0.1,
            compute_y=True,
        )
        assert full_y.shape == (n, 3 * d)
        torch.testing.assert_close(full_y.float(), y.float(), rtol=1.5e-2, atol=1.5e-2)
        assert len(_standalone_bwd._COMPILED) == 2

        _standalone_fwd.ln_mul_dropout_fwd(
            x,
            u.contiguous(),
            weight,
            bias,
            eps=1e-6,
            dropout_ratio=0.1,
            training=True,
            seed=101,
        )
        assert len(_standalone_fwd._COMPILED) == 2
    finally:
        try:
            torch.cuda.synchronize()
        finally:
            _standalone_fwd._COMPILED.clear()
            _standalone_bwd._COMPILED.clear()
