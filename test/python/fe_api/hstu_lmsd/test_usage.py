# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal public-API usage tests for HSTU LMSD."""

import pytest
import torch

try:
    import cutlass  # noqa: F401
except (ImportError, OSError) as exc:
    pytest.skip(f"CuTe DSL is unavailable: {exc}", allow_module_level=True)

from cudnn.api_base import TensorDesc
from cudnn.hstu_lmsd import (
    HSTULMSDBwdSm100,
    HSTULMSDFwdSm100,
    hstu_lmsd_backward,
    hstu_lmsd_forward,
)
from cudnn.hstu_lmsd.cutedsl.cute_dsl_ln_mul_dropout_bwd import TARGET_TILES

pytestmark = [
    pytest.mark.gpu_exclusive,
    pytest.mark.xdist_group(name="gpu_exclusive"),
]

_IS_SM10X = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


def _inputs(n: int = 257):
    torch.manual_seed(123)
    d = 512
    x = torch.randn((n, d), device="cuda", dtype=torch.bfloat16)
    u_storage = torch.randn((n, 4 * d), device="cuda", dtype=torch.bfloat16)
    u = u_storage[:, :d]
    weight = torch.randn((d,), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((d,), device="cuda", dtype=torch.bfloat16)
    return x, u, weight, bias


def _desc(shape, dtype, *, stride=None, name=""):
    """Build a storage-free descriptor with the same metadata as a CUDA tensor."""
    shape = tuple(shape)
    if stride is None:
        running = 1
        reversed_stride = []
        for extent in reversed(shape):
            reversed_stride.append(running)
            running *= extent
        stride = tuple(reversed(reversed_stride))
    else:
        stride = tuple(stride)
    stride_order = tuple(dim for dim, _ in sorted(enumerate(stride), key=lambda item: (item[1], shape[item[0]])))
    return TensorDesc(
        dtype=dtype,
        shape=shape,
        stride=stride,
        stride_order=stride_order,
        device=torch.device("cuda", torch.cuda.current_device()),
        name=name,
    )


def _metadata_only_apis(*, n=37, d=512, dtype=torch.bfloat16, x_stride=None):
    """Construct both APIs exclusively from TensorDesc metadata."""
    x = _desc((n, d), dtype, stride=x_stride, name="x")
    u = _desc((n, d), dtype, stride=(4 * d, 1), name="u")
    weight = _desc((d,), dtype, name="weight")
    bias = _desc((d,), dtype, name="bias")

    fwd = HSTULMSDFwdSm100(
        sample_x=x,
        sample_u=u,
        sample_weight=weight,
        sample_bias=bias,
        sample_y=_desc((n, 3 * d), dtype, name="y"),
        sample_mean=_desc((n,), torch.float32, name="mean"),
        sample_rstd=_desc((n,), torch.float32, name="rstd"),
        sample_mask=_desc((n, d), torch.int8, name="mask"),
        eps=1e-6,
        dropout_ratio=0.1,
    )
    bwd = HSTULMSDBwdSm100(
        sample_dy=_desc((n, 3 * d), dtype, name="dy"),
        sample_x=x,
        sample_u=u,
        sample_weight=weight,
        sample_bias=bias,
        sample_mean=_desc((n,), torch.float32, name="mean"),
        sample_rstd=_desc((n,), torch.float32, name="rstd"),
        sample_mask=_desc((n, d), torch.int8, name="mask"),
        sample_dx=_desc((n, d), dtype, name="dx"),
        sample_du=_desc((n, d), dtype, name="du"),
        sample_dweight=_desc((d,), dtype, name="dweight"),
        sample_dbias=_desc((d,), dtype, name="dbias"),
        sample_dweight_workspace=_desc((TARGET_TILES, d), torch.float32, name="dweight_workspace"),
        sample_dbias_workspace=_desc((TARGET_TILES, d), torch.float32, name="dbias_workspace"),
        dropout_ratio=0.1,
    )
    return fwd, bwd


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="HSTU LMSD requires SM10x")
def test_metadata_only_check_support_accepts_forward_and_backward():
    """Plan-time support checks need tensor metadata, not allocated GPU storage."""
    for api in _metadata_only_apis():
        assert api.check_support() is True


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="HSTU LMSD requires SM10x")
@pytest.mark.parametrize("api_index", (0, 1), ids=("forward", "backward"))
@pytest.mark.parametrize(
    "overrides,match",
    (
        ({"d": 256}, "supports D=512"),
        ({"dtype": torch.float16}, "x must have dtype torch.bfloat16"),
        ({"x_stride": (1024, 2)}, "x must have a unit innermost stride"),
        ({"n": 4_194_305}, "x row count must be in"),
    ),
    ids=("hidden-size", "dtype", "layout", "row-count"),
)
def test_metadata_only_check_support_rejects_invalid_contracts(api_index, overrides, match):
    """These cases were accepted too late unless check_support read TensorDesc."""
    api = _metadata_only_apis(**overrides)[api_index]
    with pytest.raises(ValueError, match=match):
        api.check_support()


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="HSTU LMSD requires SM10x")
def test_explicit_forward_backward_usage():
    x, u, weight, bias = _inputs()
    forward = hstu_lmsd_forward(
        x,
        u,
        weight,
        bias,
        eps=1e-6,
        dropout_ratio=0.1,
        seed=17,
    )
    y, mean, rstd, mask = forward
    assert y.shape == (x.shape[0], 3 * x.shape[1])
    assert mean.shape == rstd.shape == (x.shape[0],)
    assert mask.shape == x.shape
    assert y.dtype == x.dtype
    assert mean.dtype == rstd.dtype == torch.float32
    assert mask.dtype == torch.int8

    dy = torch.randn_like(y)
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
    )
    dx, du, dweight, dbias = backward
    assert dx.shape == du.shape == x.shape
    assert dweight.shape == dbias.shape == weight.shape
    assert dx.dtype == du.dtype == dweight.dtype == dbias.dtype == x.dtype
    assert torch.isfinite(dx).all()
    assert torch.isfinite(du).all()
    assert torch.isfinite(dweight).all()
    assert torch.isfinite(dbias).all()
