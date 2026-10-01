# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent WGrad consumers must acquire rebuilt maps on every launch."""

import pytest
import torch

import cudnn
from gemm.cutedsl.test_grouped_gemm_wgrad_utils import _wgrad_assemble_scales_2d2d, _wgrad_create_fp4_tensor
from test_utils import torch_fork_set_rng


@pytest.mark.L0
@pytest.mark.parametrize("fp4", [False, True], ids=["fp8", "fp4"])
@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
@pytest.mark.parametrize("cga", [1, 2])
@torch_fork_set_rng(seed=1313)
def test_wgrad_tensormaps_rebind_and_replay(fp4, discrete, cga):
    major, minor = torch.cuda.get_device_capability()
    if not 100 <= major * 10 + minor <= 107:
        pytest.skip("SM100/SM107 WGrad qualification")

    # More output tiles per expert than resident CTAs, including two-CTA MMA.
    rubin = (major, minor) == (10, 7)
    m, n, total_k, experts = 3072, 2048, 768 if rubin else 640, 2
    vec = 16 if fp4 else 32
    sf_dtype = torch.float8_e4m3fn if fp4 else torch.float8_e8m0fnu
    global_scale = torch.ones(experts, device="cuda") if fp4 else None

    def inputs(lengths):
        if fp4:
            a, logical_a = _wgrad_create_fp4_tensor((m, total_k), packed_dim=-1, return_logical=True)
            b, logical_b = _wgrad_create_fp4_tensor((total_k, n), packed_dim=0, return_logical=True)
        else:
            logical_a = torch.randint(-1, 2, (m, total_k), device="cuda").float()
            logical_b = torch.randint(-1, 2, (total_k, n), device="cuda").float()
            a = logical_a.to(torch.float8_e4m3fn)
            b = logical_b.to(torch.float8_e4m3fn).T.contiguous().T
        factors = [_wgrad_assemble_scales_2d2d([torch.ones(width, k // vec, device="cuda").to(sf_dtype) for k in lengths], width) for width in (m, n)]
        out = torch.empty(experts, m, n, dtype=torch.bfloat16, device="cuda")
        return dict(
            a=a,
            b=b,
            logical_a=logical_a,
            logical_b=logical_b,
            sfa=factors[0],
            sfb=factors[1],
            offsets=torch.tensor(lengths, dtype=torch.int32, device="cuda").cumsum(0).to(torch.int32),
            out=out,
            pointers=torch.tensor([out[i].data_ptr() for i in range(experts)], dtype=torch.int64, device="cuda"),
        )

    first_lengths = [256, 512] if rubin else [256, 384]
    first = inputs(first_lengths)
    output_kwargs = dict(num_experts=experts, wgrad_shape=(m, n), wgrad_dtype=torch.bfloat16) if discrete else dict(sample_wgrad=first["out"])
    api = cudnn.GroupedGemmWgradSm100(
        sample_a=first["a"],
        sample_b=first["b"],
        sample_sfa=first["sfa"],
        sample_sfb=first["sfb"],
        sample_offsets=first["offsets"],
        sample_global_scale_a=global_scale,
        sample_global_scale_b=global_scale,
        acc_dtype=torch.float32,
        mma_tiler_mn=(128, 128) if cga == 1 else (256, 256),
        cluster_shape_mn=(cga, 1),
        sf_vec_size=vec,
        **output_kwargs,
    )
    assert api.check_support()
    api.compile()
    workspace = torch.empty(
        cudnn.get_grouped_gemm_wgrad_workspace_size_sm100(experts, output_mode="discrete" if discrete else "dense"), dtype=torch.uint8, device="cuda"
    )

    def run(x):
        api.execute(
            a_tensor=x["a"],
            b_tensor=x["b"],
            sfa_tensor=x["sfa"],
            sfb_tensor=x["sfb"],
            offsets_tensor=x["offsets"],
            wgrad_tensor=None if discrete else x["out"],
            wgrad_ptrs=x["pointers"] if discrete else None,
            global_scale_a=global_scale,
            global_scale_b=global_scale,
            descriptor_workspace=workspace,
        )

    def check(x, lengths):
        begin = 0
        for expert, length in enumerate(lengths):
            expected = (x["logical_a"][:, begin : begin + length].double() @ x["logical_b"][begin : begin + length].double()).bfloat16()
            torch.testing.assert_close(x["out"][expert], expected, atol=0, rtol=0)
            begin += length

    first["out"].fill_(float("nan"))
    run(first)
    check(first, first_lengths)
    second_lengths = [512, 256] if rubin else [384, 256]
    second = inputs(second_lengths)
    second["out"].fill_(float("nan"))
    run(second)
    check(second, second_lengths)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(second)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            run(second)
        second["out"].fill_(float("nan"))
        graph.replay()
        check(second, second_lengths)

        third_lengths = [256, 512] if rubin else [128, 512]
        second["offsets"].copy_(torch.tensor(third_lengths, dtype=torch.int32).cumsum(0).to(torch.int32))
        if fp4:
            second["a"].view(torch.uint8).bitwise_xor_(0x88)
        else:
            second["a"].copy_((-second["a"].float()).to(second["a"].dtype))
        second["logical_a"].neg_()
        # In discrete mode the setup kernel must also rebuild output addresses.
        old_output = second["out"]
        if discrete:
            second["out"] = torch.empty_like(old_output)
            second["pointers"].copy_(torch.tensor([second["out"][i].data_ptr() for i in range(experts)], dtype=torch.int64))
        second["out"].fill_(float("nan"))
        graph.replay()
        check(second, third_lengths)
    finally:
        graph.reset()
