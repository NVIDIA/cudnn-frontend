# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cold compilation and replay of block-scaled grouped wgrad with no tokens."""

import pytest
import torch
import cudnn

from cudnn.gemm.cutedsl.grouped.wgrad import api as wgrad_api
from cudnn.tensor_adapter import has_tcgen05
from determinism_utils import assert_bitwise

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("precision", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("output_mode", ["dense", "discrete"])
@pytest.mark.parametrize("accumulate", [False, True], ids=["overwrite", "accumulate"])
@pytest.mark.parametrize("flat_scales", [False, True], ids=["matrix-sf", "flat-sf"])
@pytest.mark.parametrize("tile_m", [128, 256], ids=["1cta", "2cta"])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32], ids=["bf16-dw", "fp32-dw"])
def test_grouped_wgrad_cold_empty_rebind_and_capture(monkeypatch, precision, output_mode, accumulate, flat_scales, tile_m, output_dtype):
    if not has_tcgen05(torch.cuda.get_device_capability()[0]):
        pytest.skip("Requires SM100+ for grouped GEMM wgrad.")

    # Other tests must not warm a nonempty specialization and hide cold-empty compilation.
    monkeypatch.setattr(wgrad_api, "_cache_of_GroupedGemmWgradSm100Objects", {})
    monkeypatch.setattr(wgrad_api, "_wgrad_wrapper_memo", {})
    fp4 = precision == "nvfp4"
    data_dtype = torch.float4_e2m1fn_x2 if fp4 else torch.float8_e4m3fn
    scale_dtype = torch.float8_e4m3fn if fp4 else torch.float8_e8m0fnu
    sf_vec_size = 16 if fp4 else 32
    groups, m, n = 3, 256, 256

    def problem(group_tokens):
        tokens = sum(group_tokens)
        if tokens == 0:
            # Explicit strides preserve the FP4 packing axis, unlike an ambiguous
            # torch.empty((m, 0)) with stride (1, 1).
            a = torch.empty_strided((m, 0), (16, 1), dtype=data_dtype, device="cuda")
            b = torch.empty_strided((0, n), (1, 16), dtype=data_dtype, device="cuda")
            sfa = torch.empty_strided((m, 0), (16, 1), dtype=scale_dtype, device="cuda")
            sfb = torch.empty_strided((n, 0), (16, 1), dtype=scale_dtype, device="cuda")
        else:
            if fp4:
                # Two exactly representable E2M1 ones per byte.
                a = torch.full((m, tokens // 2), 0x22, dtype=torch.uint8, device="cuda").view(data_dtype)
                b = torch.full((n, tokens // 2), 0x22, dtype=torch.uint8, device="cuda").view(data_dtype).T
            else:
                a = torch.ones((m, tokens), device="cuda").to(data_dtype)
                b = torch.ones((n, tokens), device="cuda").to(data_dtype).T
            sfa = torch.ones((m, tokens // sf_vec_size), device="cuda").to(scale_dtype)
            sfb = torch.ones((n, tokens // sf_vec_size), device="cuda").to(scale_dtype)
        if flat_scales:
            sfa, sfb = sfa.view(-1), sfb.view(-1)
        output = torch.full((groups, m, n), 0.125, dtype=output_dtype, device="cuda")
        output[0, 0, 0] = -0.0
        initial = output.clone()
        offsets = torch.tensor(group_tokens, dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)
        kwargs = dict(
            a_tensor=a,
            b_tensor=b,
            sfa_tensor=sfa,
            sfb_tensor=sfb,
            offsets_tensor=offsets,
            output_mode=output_mode,
            wgrad_dtype=output.dtype,
            acc_dtype=torch.float32,
            mma_tiler_mn=(tile_m, 256),
            cluster_shape_mn=(tile_m // 128, 1),
            sf_vec_size=sf_vec_size,
            accumulate_on_output=accumulate,
            input_order="tensor_ragged",
        )
        if output_mode == "dense":
            kwargs["wgrad_tensor"] = output
        else:
            kwargs["wgrad_ptrs"] = torch.tensor([expert.data_ptr() for expert in output], dtype=torch.int64, device="cuda")
        if fp4:
            kwargs["global_scale_a"] = torch.ones(groups, device="cuda")
            kwargs["global_scale_b"] = torch.ones(groups, device="cuda")
        return kwargs, output, initial

    def check_empty(output, initial):
        expected = initial if accumulate else torch.zeros_like(output)
        # Accumulating an empty batch is a no-op, including the sign of zero.
        assert_bitwise({"wgrad": output}, {"wgrad": expected})

    kwargs, output, initial = problem([0, 0, 0])
    cudnn.grouped_gemm_wgrad_wrapper_sm100(**kwargs)
    check_empty(output, initial)

    def unexpected_compile(*args, **kwargs):
        pytest.fail("Changing the token count must reuse the cold-empty compiled kernel")

    monkeypatch.setattr(wgrad_api.GroupedGemmWgradSm100, "compile", unexpected_compile)
    # A kernel compiled from empty tensors must also compute nonempty experts.
    for group_tokens in ([256, 0, 256], [0, 512, 0]):
        nonempty_kwargs, nonempty_output, nonempty_initial = problem(group_tokens)
        cudnn.grouped_gemm_wgrad_wrapper_sm100(**nonempty_kwargs)
        expected = nonempty_initial.clone() if accumulate else torch.zeros_like(nonempty_output)
        for expert, tokens in enumerate(group_tokens):
            if tokens:
                expected[expert].add_(tokens)
        assert_bitwise({"wgrad": nonempty_output}, {"wgrad": expected})

    # Rebind fresh empty allocations after the nonempty launch, then replay a graph.
    kwargs, output, initial = problem([0, 0, 0])
    cudnn.grouped_gemm_wgrad_wrapper_sm100(**kwargs)
    check_empty(output, initial)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cudnn.grouped_gemm_wgrad_wrapper_sm100(**kwargs)
    for _ in range(2):
        output.copy_(initial)
        graph.replay()
        check_empty(output, initial)
