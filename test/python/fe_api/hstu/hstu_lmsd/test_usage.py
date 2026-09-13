# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal public-API usage tests for HSTU LMSD."""

import itertools

import cudnn
import pytest
import torch

try:
    import cutlass  # noqa: F401
except (ImportError, OSError) as exc:
    pytest.skip(f"CuTe DSL is unavailable: {exc}", allow_module_level=True)

from cudnn.api_base import TensorDesc
from cudnn.hstu import hstu_lmsd
from cudnn.hstu.hstu_lmsd import hstu_lmsd_backward, hstu_lmsd_forward
from cudnn.hstu.hstu_lmsd.api import HSTULMSDBwd, HSTULMSDFwd
from cudnn.hstu.hstu_lmsd._kernels._config import HSTULMSDBwdConfig, HSTULMSDFwdConfig

pytestmark = [
    pytest.mark.gpu_exclusive,
    pytest.mark.xdist_group(name="gpu_exclusive"),
]

_IS_SM10X = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


@pytest.mark.L0
def test_public_exports_are_function_only():
    assert hstu_lmsd.__all__ == ["hstu_lmsd_forward", "hstu_lmsd_backward"]
    assert not hasattr(hstu_lmsd, "HSTULMSDFwd")
    assert not hasattr(hstu_lmsd, "HSTULMSDBwd")
    assert not hasattr(cudnn, "HSTULMSDFwd")
    assert not hasattr(cudnn, "HSTULMSDBwd")


def _inputs(n: int = 257, d: int = 512):
    torch.manual_seed(123)
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


def _metadata_only_apis(
    *,
    n=37,
    d=512,
    dtype=torch.bfloat16,
    x_stride=None,
    u_stride=None,
    dropout_ratio=0.1,
    apply_u_silu=True,
    concat_u=True,
    concat_x=True,
    compute_dweight=True,
):
    """Construct both APIs exclusively from TensorDesc metadata."""
    x = _desc((n, d), dtype, stride=x_stride, name="x")
    u = _desc((n, d), dtype, stride=(4 * d, 1) if u_stride is None else u_stride, name="u")
    weight = _desc((d,), dtype, name="weight")
    bias = _desc((d,), dtype, name="bias")

    output_width = (1 + int(concat_u) + int(concat_x)) * d
    mask = _desc((n, d), torch.int8, name="mask") if dropout_ratio > 0.0 else None
    dweight = _desc((d,), dtype, name="dweight") if compute_dweight else None
    kernel_config = HSTULMSDBwdConfig.from_hidden_size(d)
    multiprocessor_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    workspace_rows = kernel_config.workspace_rows(multiprocessor_count)
    dweight_workspace = _desc((workspace_rows, d), torch.float32, name="dweight_workspace") if compute_dweight else None

    fwd = HSTULMSDFwd(
        sample_x=x,
        sample_u=u,
        sample_weight=weight,
        sample_bias=bias,
        sample_y=_desc((n, output_width), dtype, name="y"),
        sample_mean=_desc((n,), torch.float32, name="mean"),
        sample_rstd=_desc((n,), torch.float32, name="rstd"),
        sample_mask=mask,
        eps=1e-6,
        dropout_ratio=dropout_ratio,
        apply_u_silu=apply_u_silu,
        concat_u=concat_u,
        concat_x=concat_x,
    )
    bwd = HSTULMSDBwd(
        sample_dy=_desc((n, output_width), dtype, name="dy"),
        sample_x=x,
        sample_u=u,
        sample_weight=weight,
        sample_bias=bias,
        sample_mean=_desc((n,), torch.float32, name="mean"),
        sample_rstd=_desc((n,), torch.float32, name="rstd"),
        sample_mask=mask,
        sample_dx=_desc((n, d), dtype, name="dx"),
        sample_du=_desc((n, d), dtype, name="du"),
        sample_dweight=dweight,
        sample_dbias=_desc((d,), dtype, name="dbias"),
        sample_dweight_workspace=dweight_workspace,
        sample_dbias_workspace=_desc((workspace_rows, d), torch.float32, name="dbias_workspace"),
        dropout_ratio=dropout_ratio,
        apply_u_silu=apply_u_silu,
        concat_u=concat_u,
        concat_x=concat_x,
        compute_dweight=compute_dweight,
    )
    return fwd, bwd


def test_launch_grids_follow_runtime_row_count():
    fwd = HSTULMSDFwdConfig.from_hidden_size(512)
    fwd_wide = HSTULMSDFwdConfig.from_hidden_size(960)
    bwd_small = HSTULMSDBwdConfig.from_hidden_size(128)
    bwd_large = HSTULMSDBwdConfig.from_hidden_size(512)
    bwd_wide = HSTULMSDBwdConfig.from_hidden_size(768)

    assert fwd.launch_grid(1, 148) == 1
    assert fwd.launch_grid(1_000_000, 148) == 148 * fwd.grid_ctas_per_sm
    assert (fwd.vector_size, fwd.rows_per_cta, fwd.min_blocks_per_mp) == (4, 4, 8)
    assert (fwd_wide.vector_size, fwd_wide.rows_per_cta, fwd_wide.min_blocks_per_mp) == (8, 4, 6)
    assert bwd_small.launch_grid(513, 148) == 513
    assert bwd_large.workspace_rows(32) == 32 * bwd_large.grid_ctas_per_sm
    assert bwd_large.workspace_rows(148) == 148 * bwd_large.grid_ctas_per_sm
    assert bwd_large.launch_grid(1_000_000, 148) == 148 * bwd_large.grid_ctas_per_sm
    assert (bwd_small.threads_per_row, bwd_large.threads_per_row, bwd_wide.threads_per_row) == (32, 64, 128)


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="HSTU LMSD requires SM10x")
def test_metadata_only_check_support_accepts_forward_and_backward():
    """Plan-time support checks need tensor metadata, not allocated GPU storage."""
    for d in range(8, 1024, 8):
        for api in _metadata_only_apis(d=d, x_stride=(3 * d, 1)):
            assert api.check_support() is True


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="HSTU LMSD requires SM10x")
def test_metadata_only_check_support_accepts_optional_configurations():
    for dropout_ratio in (0.0, 0.2):
        for apply_u_silu, concat_u, concat_x, compute_dweight in itertools.product((False, True), repeat=4):
            options = {
                "dropout_ratio": dropout_ratio,
                "apply_u_silu": apply_u_silu,
                "concat_u": concat_u,
                "concat_x": concat_x,
                "compute_dweight": compute_dweight,
            }
            for api in _metadata_only_apis(d=128, **options):
                assert api.check_support() is True


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="HSTU LMSD requires SM10x")
@pytest.mark.parametrize("d", (0, 7, 10, 1024))
def test_metadata_only_check_support_rejects_invalid_hidden_dimensions(d):
    for api in _metadata_only_apis(d=d):
        with pytest.raises(ValueError, match="supports D divisible by"):
            api.check_support()


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="HSTU LMSD requires SM10x")
@pytest.mark.parametrize("api_index", (0, 1), ids=("forward", "backward"))
@pytest.mark.parametrize(
    "overrides,match",
    (
        ({"d": 1024}, "supports D divisible by"),
        ({"dtype": torch.float16}, "x must have dtype torch.bfloat16"),
        ({"x_stride": (1024, 2)}, "x must have a unit innermost stride"),
        ({"x_stride": (504, 1)}, "x rows must not overlap"),
        ({"n": 4_194_305}, "x row count must be in"),
    ),
    ids=("hidden-size", "dtype", "inner-stride", "overlapping-rows", "row-count"),
)
def test_metadata_only_check_support_rejects_invalid_contracts(api_index, overrides, match):
    """Unsupported descriptors must fail during check_support()."""
    api = _metadata_only_apis(**overrides)[api_index]
    with pytest.raises(ValueError, match=match):
        api.check_support()


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="HSTU LMSD requires SM10x")
@pytest.mark.parametrize("api_index", (0, 1), ids=("forward", "backward"))
@pytest.mark.parametrize(
    "dropout_ratio,match",
    (
        (-0.1, "finite and in"),
        (1.0, "finite and in"),
        (float("inf"), "finite and in"),
        (float("nan"), "finite and in"),
        (0.99999999, "after float32 conversion"),
    ),
    ids=("negative", "one", "infinity", "nan", "rounds-to-one-f32"),
)
def test_metadata_only_check_support_rejects_invalid_dropout(api_index, dropout_ratio, match):
    api = _metadata_only_apis(dropout_ratio=dropout_ratio)[api_index]
    with pytest.raises(ValueError, match=match):
        api.check_support()


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="HSTU LMSD requires SM10x")
def test_metadata_only_check_support_normalizes_dropout_to_float32():
    dropout_ratio = 0.99999994
    expected = torch.tensor(dropout_ratio, dtype=torch.float32).item()
    fwd, bwd = _metadata_only_apis(dropout_ratio=dropout_ratio)
    for api in (fwd, bwd):
        assert api.check_support() is True
        assert api.dropout_ratio == expected
    assert 0 <= fwd._threshold < (1 << 32)


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="HSTU LMSD requires SM10x")
def test_forward_wrapper_rejects_dropout_that_rounds_to_one():
    x, u, weight, bias = _inputs(n=1)
    with pytest.raises(ValueError, match="after float32 conversion"):
        hstu_lmsd_forward(x, u, weight, bias, dropout_ratio=0.99999999)


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="HSTU LMSD requires SM10x")
@pytest.mark.parametrize("d", (128, 512))
def test_explicit_forward_backward_usage(d):
    x, u, weight, bias = _inputs(d=d)
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
        apply_u_silu=True,
        concat_u=True,
        concat_x=True,
    )
    dx, du, dweight, dbias = backward
    assert dx.shape == du.shape == x.shape
    assert dweight.shape == dbias.shape == weight.shape
    assert dx.dtype == du.dtype == dweight.dtype == dbias.dtype == x.dtype
    assert torch.isfinite(dx).all()
    assert torch.isfinite(du).all()
    assert torch.isfinite(dweight).all()
    assert torch.isfinite(dbias).all()
