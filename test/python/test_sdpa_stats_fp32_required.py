# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test: the SDPA Stats output must be declared FP32.

The SDPA kernels always compute and store Stats (logsumexp) as FP32, 4 bytes
per row. A graph that declared Stats with a narrower dtype -- either explicitly
or implicitly, by leaving it unset with a non-FP32 ``io_data_type`` -- would
previously build fine and then write FP32 values into a buffer the caller sized
for the narrower dtype, running past its end (silent corruption of adjacent
allocations, illegal memory accesses, or kernel launch failures).

Now: an unset Stats dtype is inferred as FP32 at shape inference, and an
explicitly non-FP32 Stats dtype is rejected at ``validate()``.
"""

import cudnn
import pytest
import torch

pytestmark = pytest.mark.L0

_DTYPE = {
    torch.float32: cudnn.data_type.FLOAT,
    torch.float16: cudnn.data_type.HALF,
    torch.bfloat16: cudnn.data_type.BFLOAT16,
}


def _build_sdpa_with_stats(io_dtype, stats_dtype):
    """Build an SDPA graph with generate_stats=True through validate().

    ``stats_dtype`` of None leaves the Stats dtype unset, mimicking callers
    that rely on inference; the returned stats tensor lets the caller assert
    what was inferred.
    """
    b, h, s, d = 2, 4, 128, 64

    graph = cudnn.pygraph(
        io_data_type=_DTYPE[io_dtype],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    dim = [b, h, s, d]
    stride = [h * s * d, s * d, d, 1]
    q = graph.tensor(name="q", dim=dim, stride=stride, data_type=_DTYPE[io_dtype])
    k = graph.tensor(name="k", dim=dim, stride=stride, data_type=_DTYPE[io_dtype])
    v = graph.tensor(name="v", dim=dim, stride=stride, data_type=_DTYPE[io_dtype])
    o, stats = graph.sdpa(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        generate_stats=True,
        attn_scale=1.0 / (d**0.5),
    )
    o.set_output(True).set_dim(dim).set_stride(stride).set_data_type(_DTYPE[io_dtype])
    stats.set_output(True).set_dim([b, h, s, 1]).set_stride([h * s, s, 1, 1])
    if stats_dtype is not None:
        stats.set_data_type(_DTYPE[stats_dtype])

    graph.validate()
    return stats


@pytest.mark.parametrize("io_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("stats_dtype", [torch.float16, torch.bfloat16])
def test_non_fp32_stats_rejected(io_dtype, stats_dtype):
    """An explicitly non-FP32 Stats output is rejected at validate(), before
    any kernel can write FP32 rows past the end of an undersized buffer."""
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="Stats output of sdpa must be an FP32"):
        _build_sdpa_with_stats(io_dtype, stats_dtype)


@pytest.mark.parametrize("io_dtype", [torch.float16, torch.bfloat16])
def test_unset_stats_dtype_inferred_fp32(io_dtype):
    """An unset Stats dtype must be inferred as FP32, not the io dtype: it is
    what tells callers (via the graph/tensor introspection) how large a buffer
    to bind, so inheriting a 2-byte io dtype would halve the expected size."""
    stats = _build_sdpa_with_stats(io_dtype, None)
    assert stats.get_data_type() == cudnn.data_type.FLOAT


def test_fp32_stats_accepted():
    """The required FP32 declaration itself must not be over-rejected."""
    stats = _build_sdpa_with_stats(torch.float16, torch.float32)
    assert stats.get_data_type() == cudnn.data_type.FLOAT
