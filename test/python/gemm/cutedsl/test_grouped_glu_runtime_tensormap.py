# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime TMA descriptors must observe new addresses during graph replay."""

import pytest
import torch

from determinism_utils import assert_bitwise
from test_grouped_gemm_glu_branch_layout import make_problem, pack_scales

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("precision", ["bf16", "nvfp4", "mxfp8"])
@pytest.mark.parametrize("tile_m", [128, 256])
def test_canonical_discrete_graph_reuses_freed_operands(precision, tile_m):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Canonical GLU branches require SM100/103")
    from cudnn import grouped_gemm_glu_wrapper_sm100

    options, b, sb, _, _ = make_problem(f=256, k=512, counts=(256,) * 4, precision=precision)
    sfb = pack_scales(sb)
    options.update(
        glu_layout="gate_then_up",
        use_dynamic_sched=True,
        discrete_col_sfd=precision == "mxfp8",
        mma_tiler_mn=(tile_m, 256),
        cluster_shape_mn=(2, 1) if tile_m == 256 else (1, 1),
        n=b.shape[1],
        b_dtype=b.dtype,
        b_major="k",
    )
    b_offsets = torch.arange(4, device="cuda", dtype=torch.int64) * b[0].numel() * b.element_size()
    sf_offsets = None if sfb is None else torch.arange(4, device="cuda", dtype=torch.int64) * (sfb.numel() // 4)

    def invoke():
        weights = b.view(torch.uint8).clone().view(b.dtype)
        scales = None if sfb is None else sfb.clone()
        result = grouped_gemm_glu_wrapper_sm100(
            **options,
            b_ptrs=b_offsets + weights.data_ptr(),
            sfb_ptrs=None if scales is None else sf_offsets + scales.data_ptr(),
        )
        # These owners intentionally die when this call returns, as in TE's
        # per-invocation quantized discrete weights and swizzled scale buffers.
        return {
            name: tensor for name, tensor in result.items() if tensor is not None and (precision == "mxfp8" or name in ("c_tensor", "d_tensor", "amax_tensor"))
        }

    expected = {name: tensor.clone() for name, tensor in invoke().items()}
    for _ in range(3):
        invoke()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    poisons = []
    with torch.cuda.graph(graph):
        for _ in range(4):
            actual = invoke()
            # Reuse the freed operands and keep the poison alive. The next
            # invocation must acquire descriptors containing its new addresses;
            # a stale descriptor otherwise reads these overwritten buffers.
            poisons.append(torch.full_like(b.view(torch.uint8), 0))
            if sfb is not None:
                poisons.append(torch.full_like(sfb.view(torch.uint8), 130))
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert_bitwise(actual, expected, "runtime TMA descriptor publication")


@pytest.mark.parametrize("precision", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("glu_layout", ["interleaved_32", "gate_then_up"])
def test_discrete_dglu_graph_reacquires_changed_operands(precision, glu_layout):
    """Descriptor reuse within a launch must never survive a graph invocation."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Canonical GLU branches require SM100/103")
    from cudnn import grouped_gemm_dglu_wrapper_sm100

    options, b, sb, _, _ = make_problem(f=512, k=512, counts=(4096, 0, 4096, 512), precision=precision)
    sfb = pack_scales(sb)
    rows, n = options["a_tensor"].shape[0], b.shape[1]
    c = torch.randn(rows, 2 * n, device="cuda", dtype=torch.bfloat16) * 0.125
    dprob = torch.zeros(rows, device="cuda", dtype=torch.float32)
    kwargs = {
        name: options[name]
        for name in ("a_tensor", "sfa_tensor", "padded_offsets", "alpha_tensor", "prob_tensor", "norm_const_tensor", "sf_vec_size", "d_dtype")
    }
    kwargs.update(
        c_tensor=c,
        beta_tensor=torch.ones(len(b), device="cuda"),
        dprob_tensor=dprob,
        glu_layout=glu_layout,
        deterministic=True,
        use_dynamic_sched=True,
        discrete_col_sfd=precision == "mxfp8",
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        n=n,
        b_dtype=b.dtype,
        b_major="k",
    )
    b_offsets = torch.arange(len(b), device="cuda", dtype=torch.int64) * b[0].numel() * b.element_size()
    sf_offsets = torch.arange(len(b), device="cuda", dtype=torch.int64) * (sfb.numel() // len(b))

    def invoke():
        weights = b.view(torch.uint8).clone().view(b.dtype)
        scales = sfb.clone()
        dprob.zero_()
        result = grouped_gemm_dglu_wrapper_sm100(**kwargs, b_ptrs=b_offsets + weights.data_ptr(), sfb_ptrs=sf_offsets + scales.data_ptr())
        return {name: value for name, value in result.items() if value is not None and (precision == "mxfp8" or name in ("d_row_tensor", "dprob_tensor"))}

    expected = {name: value.clone() for name, value in invoke().items()}
    for _ in range(3):
        invoke()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    poisons = []
    with torch.cuda.graph(graph):
        for _ in range(4):
            actual = invoke()
            poisons.append(torch.zeros_like(b.view(torch.uint8)))
            poisons.append(torch.full_like(sfb.view(torch.uint8), 130))
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert_bitwise(actual, expected, "dGLU descriptor acquisition resets each invocation")
