# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test for https://github.com/NVIDIA/cudnn-frontend/issues/424.

The unified SDPA node has no FP32 I/O kernel: ``mma_core_mode`` is auto-set to
HALF in ``Graph::sdpa()`` regardless of the I/O dtype, so an FP32 graph would
previously build and then dispatch a half-precision kernel onto 4-byte data,
reading/writing shared memory out of bounds (a CUDA invalid memory reference).

``verify_sdpa_support_surface_for_implementation`` now checks each Q/K/V/O I/O
dtype against an allowlist on both paths: the unified path allows FP16/BF16/FP8,
the composite path additionally allows FP32; neither allows FP64 or anything
else. Unsupported dtypes fail cleanly at build time with GRAPH_NOT_SUPPORTED.
Under AUTO, FP32 is routed to the composite implementation.
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

_PORTS = ["q", "k", "v", "o"]


def _is_dtype_allowlist_rejection(exc):
    """Whether ``exc`` is our unified/composite support-surface dtype rejection.

    Both allowlist messages lead with "<impl> SDPA node supports only ... I/O",
    so a positive test that hits this has been wrongly rejected by the very
    contract under test (a regression) -- as opposed to a genuine engine/backend
    unavailability, which raises a different message and is safe to skip.
    """
    return "SDPA node supports only" in str(exc)


def _build_sdpa(implementation, base_dtype=torch.float16, *, overrides=None):
    """Build and finalize an SDPA graph.

    Every Q/K/V/O tensor uses ``base_dtype`` except the ports named in
    ``overrides`` ({port: torch_dtype}). Overriding a single port lets each
    port's dtype check be exercised independently.
    """
    overrides = overrides or {}
    b, h, s, d = 2, 4, 128, 64

    def dt(port):
        return _DTYPE[overrides.get(port, base_dtype)]

    graph = cudnn.pygraph(
        io_data_type=_DTYPE[base_dtype],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    dim = [b, h, s, d]
    stride = [h * s * d, s * d, d, 1]
    q = graph.tensor(name="q", dim=dim, stride=stride, data_type=dt("q"))
    k = graph.tensor(name="k", dim=dim, stride=stride, data_type=dt("k"))
    v = graph.tensor(name="v", dim=dim, stride=stride, data_type=dt("v"))
    o, _ = graph.sdpa(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        generate_stats=False,
        attn_scale=1.0 / (d**0.5),
        implementation=implementation,
    )
    o.set_output(True).set_dim(dim).set_stride(stride).set_data_type(dt("o"))

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()
    return graph


@pytest.mark.parametrize("bad_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("port", _PORTS)
def test_unified_rejects_unsupported_io_dtype(port, bad_dtype):
    """Each Q/K/V/O port is checked independently on the unified path: an
    unsupported dtype on any single port (the rest FP16) is rejected by the
    unified support surface. Covers FP32 (issue #424) and FP64. The ``match``
    asserts the rejection comes from our dtype check, not an incidental one."""
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="Unified SDPA node"):
        _build_sdpa(
            cudnn.attention_implementation.UNIFIED,
            torch.float16,
            overrides={port: bad_dtype},
        )


@pytest.mark.parametrize("port", _PORTS)
def test_composite_rejects_fp64_io_dtype(port):
    """The composite path additionally supports FP32 but not FP64; an FP64 dtype
    on any single port (the rest FP16) is rejected, per-port, by our check."""
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="Composite SDPA node"):
        _build_sdpa(
            cudnn.attention_implementation.COMPOSITE,
            torch.float16,
            overrides={port: torch.float64},
        )


@pytest.mark.parametrize("io_dtype", [torch.float16, torch.bfloat16])
def test_fp16_bf16_unified_supported(io_dtype):
    """The FP32/FP64 guard must not over-reject the supported FP16/BF16 dtypes."""
    try:
        _build_sdpa(cudnn.attention_implementation.UNIFIED, io_dtype)
    except cudnn.cudnnGraphNotSupportedError as e:
        # Our dtype allowlist must never reject FP16/BF16 -- that over-rejection is
        # exactly what this test guards against. Any other rejection means unified
        # SDPA is unsupported on this cuDNN/GPU combo (e.g. too-old backend), so
        # skip rather than fail.
        assert not _is_dtype_allowlist_rejection(e), f"FP16/BF16 wrongly rejected by the guard: {e}"
        pytest.skip(f"unified FP16/BF16 SDPA unsupported on this setup: {e}")


@pytest.mark.parametrize(
    "implementation",
    [cudnn.attention_implementation.COMPOSITE, cudnn.attention_implementation.AUTO],
)
def test_fp32_composite_supported(implementation):
    """FP32 must not be routed to the unified node; it is accepted by the
    composite engines (AUTO auto-selects composite for FP32)."""
    try:
        _build_sdpa(implementation, torch.float32)
    except cudnn.cudnnGraphNotSupportedError as e:
        # Either a unified misroute (FP32 must never reach the unified node) or a
        # composite over-rejection would surface as our allowlist message and is a
        # real regression -- fail. Only skip on a genuine composite-engine
        # unavailability, which raises a different message.
        assert not _is_dtype_allowlist_rejection(e), f"FP32 wrongly rejected by a support surface: {e}"
        pytest.skip("no composite FP32 SDPA engine available on this GPU")
