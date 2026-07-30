# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test for https://github.com/NVIDIA/cudnn-frontend/issues/424.

The unified SDPA node has no FP32 I/O kernel: ``mma_core_mode`` is auto-set to
HALF in ``Graph::sdpa()`` regardless of the I/O dtype, so an FP32 graph would
previously build and then dispatch a half-precision kernel onto 4-byte data,
reading/writing shared memory out of bounds (a CUDA invalid memory reference).

The support surface now rejects unsupported unified Q/K/V/O I/O dtypes, so such a
graph fails cleanly at build time with GRAPH_NOT_SUPPORTED. Under AUTO, FP32 is
routed to the composite implementation, which does support FP32 I/O.
"""

import cudnn
import pytest
import torch

pytestmark = pytest.mark.L0

_DTYPE = {
    torch.float32: cudnn.data_type.FLOAT,
    torch.float64: cudnn.data_type.DOUBLE,
    torch.float16: cudnn.data_type.HALF,
    torch.bfloat16: cudnn.data_type.BFLOAT16,
}


def _build_sdpa(io_dtype, implementation):
    b, h, s, d = 2, 4, 128, 64
    cudnn_dtype = _DTYPE[io_dtype]
    graph = cudnn.pygraph(
        io_data_type=cudnn_dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    dim = [b, h, s, d]
    stride = [h * s * d, s * d, d, 1]
    q = graph.tensor(name="q", dim=dim, stride=stride, data_type=cudnn_dtype)
    k = graph.tensor(name="k", dim=dim, stride=stride, data_type=cudnn_dtype)
    v = graph.tensor(name="v", dim=dim, stride=stride, data_type=cudnn_dtype)
    o, _ = graph.sdpa(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        generate_stats=False,
        attn_scale=1.0 / (d**0.5),
        implementation=implementation,
    )
    o.set_output(True).set_dim(dim).set_stride(stride)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()
    return graph


def test_fp32_unified_rejected():
    """FP32 forced onto the unified node fails cleanly (it has no FP32 kernel)
    instead of crashing -- the original issue #424."""
    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        _build_sdpa(torch.float32, cudnn.attention_implementation.UNIFIED)


@pytest.mark.parametrize(
    "implementation",
    [cudnn.attention_implementation.UNIFIED, cudnn.attention_implementation.COMPOSITE],
)
def test_fp64_rejected(implementation):
    """FP64 I/O is supported by no SDPA engine and must be rejected on both the
    unified and composite paths. The guards are allowlists, so they cover dtypes
    beyond the FP32 case from issue #424 (e.g. avoid 'someone files a bug for
    FP64 next'). Under AUTO, neither path accepts FP64, so auto-select instead
    fails to find any implementation -- covered by the framework, not here."""
    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        _build_sdpa(torch.float64, implementation)


@pytest.mark.parametrize("io_dtype", [torch.float16, torch.bfloat16])
def test_fp16_bf16_unified_supported(io_dtype):
    """The FP32/FP64 guard must not over-reject the supported FP16/BF16 dtypes."""
    _build_sdpa(io_dtype, cudnn.attention_implementation.UNIFIED)


@pytest.mark.parametrize(
    "implementation",
    [cudnn.attention_implementation.COMPOSITE, cudnn.attention_implementation.AUTO],
)
def test_fp32_composite_supported(implementation):
    """FP32 must not be routed to the unified node; it is accepted by the
    composite engines (AUTO auto-selects composite for FP32)."""
    try:
        _build_sdpa(torch.float32, implementation)
    except cudnn.cudnnGraphNotSupportedError:
        # No composite FP32 engine available on this architecture; the important
        # invariant (FP32 is not silently sent to the unified node) still holds.
        pytest.skip("no composite FP32 SDPA engine available on this GPU")
