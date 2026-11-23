# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical gate/up weights: numerical, legacy-layout and replay coverage."""

import pytest
import torch

from determinism_utils import assert_bitwise

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def require_blackwell():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Canonical GLU branches require SM100/103")


def pack_scales(scales):
    if scales is None:
        return None
    e, rows, cols = scales.shape
    return scales.view(e, rows // 128, 4, 32, cols // 4, 4).permute(0, 1, 4, 3, 2, 5).contiguous().view(-1)


def make_problem(f=256, k=512, counts=(256, 0, 512), precision="mxfp8"):
    # Fixed quantized payloads and independently varying per-row/K block scales.
    torch.manual_seed(401)
    e, m, n = len(counts), sum(counts), 2 * f
    if precision == "nvfp4":
        a = torch.randint(0, 256, (m, k // 2), device="cuda", dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        b = torch.randint(0, 256, (e, n, k // 2), device="cuda", dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        sa = torch.exp2(torch.randint(-5, -2, (1, m, k // 16), device="cuda").float()).to(torch.float8_e4m3fn)
        sb = torch.exp2(torch.randint(-5, -2, (e, n, k // 16), device="cuda").float()).to(torch.float8_e4m3fn)
    else:
        dtype = torch.bfloat16 if precision == "bf16" else torch.float8_e4m3fn
        a = (torch.randn(m, k, device="cuda") * 0.1).to(dtype)
        b = (torch.randn(e, n, k, device="cuda") * 0.1).to(dtype)
        sa = sb = None
        if precision == "mxfp8":
            sa = torch.randint(124, 129, (1, m, k // 32), dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu)
            sb = torch.randint(124, 129, (e, n, k // 32), dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu)
    bias = (torch.randn(e, n, device="cuda") * 0.1).to(torch.bfloat16).t()
    options = dict(
        a_tensor=a,
        sfa_tensor=pack_scales(sa),
        padded_offsets=torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32),
        alpha_tensor=torch.linspace(0.5, 1.0, e, device="cuda"),
        prob_tensor=torch.linspace(0.5, 0.9, m, device="cuda"),
        bias_tensor=bias,
        norm_const_tensor=torch.ones(1, device="cuda") if precision == "mxfp8" else None,
        sf_vec_size=32 if precision == "mxfp8" else 16,
        c_dtype=torch.bfloat16,
        d_dtype=torch.float8_e4m3fn if precision == "mxfp8" else torch.bfloat16,
        generate_c=True,
    )
    return options, b, sb, sa, counts


def call_problem(problem, *, layout, discrete=False, tile_m=128):
    from cudnn import grouped_gemm_glu_wrapper_sm100

    options, b, sb, _, _ = problem
    options = dict(options)
    if layout == "interleaved_32":
        n = b.shape[1]
        order = torch.arange(n, device="cuda").view(2, n // 64, 32).permute(1, 0, 2).reshape(-1)
        b = b.view(torch.uint8).index_select(1, order).view(b.dtype)
        if sb is not None:
            sb = sb.view(torch.uint8).index_select(1, order).view(sb.dtype)
        options["bias_tensor"] = options["bias_tensor"].t().index_select(1, order).t()
    scales = pack_scales(sb)
    if discrete:
        # Keep allocations alive until the queued launch has consumed their pointers.
        options.update(
            b_ptrs=torch.tensor([b[i].data_ptr() for i in range(b.shape[0])], device="cuda", dtype=torch.int64),
            sfb_ptrs=(
                None
                if scales is None
                else torch.tensor([scales.data_ptr() + i * scales.numel() // b.shape[0] for i in range(b.shape[0])], device="cuda", dtype=torch.int64)
            ),
            n=b.shape[1],
            b_dtype=b.dtype,
        )
    else:
        options.update(b_tensor=b, sfb_tensor=scales)
    result = grouped_gemm_glu_wrapper_sm100(**options, glu_layout=layout, mma_tiler_mn=(tile_m, 256), cluster_shape_mn=(2, 1) if tile_m == 256 else (1, 1))
    # The wrapper's output dict does not own discrete source buffers.
    torch.cuda.current_stream().synchronize()
    # BF16/NVFP4 have no column quantization. The legacy wrapper allocates a
    # dead D_col slot; it is not a kernel output and its bytes are unspecified.
    return {
        name: value
        for name, value in result.items()
        if options["norm_const_tensor"] is not None or name not in ("d_col_tensor", "sfd_row_tensor", "sfd_col_tensor")
    }


def canonicalize_c(c):
    m, n = c.shape
    return c.view(m, n // 64, 2, 32).permute(0, 2, 1, 3).reshape(m, n)


@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
@pytest.mark.parametrize("tile_m", [128, 256])
@pytest.mark.parametrize("f", [128, 256, 1024])
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
def test_gate_then_up_matches_interleaved(discrete, tile_m, f, dynamic):
    problem = make_problem(f=f)
    problem[0]["use_dynamic_sched"] = dynamic
    original = snapshot_inputs(problem)
    expected = call_problem(problem, layout="interleaved_32", discrete=discrete, tile_m=tile_m)
    expected["c_tensor"] = canonicalize_c(expected["c_tensor"])
    actual = call_problem(problem, layout="gate_then_up", discrete=discrete, tile_m=tile_m)
    for name, tensor in actual.items():
        if tensor is not None:
            assert_bitwise({name: tensor}, {name: expected[name]})

    # Independent reference checks the canonical preactivation, including bias,
    # alpha and nonuniform scales (a scale permutation bug cannot hide here).
    options, b, sb, sa, counts = problem
    a_deq = options["a_tensor"].float() * sa[0].float().repeat_interleave(32, dim=1)
    b_deq = b.float() * sb.float().repeat_interleave(32, dim=2)
    expected_d = torch.empty_like(actual["d_tensor"], dtype=torch.float32)
    begin = 0
    for expert, count in enumerate(counts):
        end = begin + count
        ref_c = (a_deq[begin:end].double() @ b_deq[expert].double().t()) * options["alpha_tensor"][expert].double()
        ref_c += options["bias_tensor"][:, expert].double()
        torch.testing.assert_close(actual["c_tensor"][begin:end].float(), ref_c.float(), rtol=0.01, atol=0.005)
        f = b.shape[1] // 2
        expected_d[begin:end] = (torch.nn.functional.silu(ref_c[:, :f]) * ref_c[:, f:] * options["prob_tensor"][begin:end, None]).float()
        begin = end
    row_scales = unpack_scales(actual["sfd_row_tensor"], expected_d.shape[0], f // 32)[0].float()
    col_scales = unpack_scales(actual["sfd_col_tensor"], f, expected_d.shape[0] // 32)[0].float()
    deq_row = actual["d_tensor"].float() * row_scales.repeat_interleave(32, dim=1)
    deq_col = actual["d_col_tensor"].float() * col_scales.repeat_interleave(32, dim=1).t()
    for dequantized in (deq_row, deq_col):
        assert torch.isfinite(dequantized).all()
        torch.testing.assert_close(dequantized, expected_d, rtol=0.125, atol=0.002)
    assert_bitwise(snapshot_inputs(problem), original, "inputs preserved")


@pytest.mark.parametrize("tile_m", [128, 256])
@pytest.mark.parametrize("precision", ["bf16", "nvfp4", "mxfp8"])
def test_gate_then_up_replay_fresh_allocations(tile_m, precision):
    problem = make_problem(f=256, counts=(256, 512, 768, 256), precision=precision)
    expected = {name: value.clone() if value is not None else None for name, value in call_problem(problem, layout="gate_then_up", tile_m=tile_m).items()}
    noise_stream = torch.cuda.Stream()
    noise = torch.randn(512, 512, device="cuda")
    noise_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(noise_stream):
        _ = noise @ noise
    for _ in range(3):
        actual = call_problem(problem, layout="gate_then_up", tile_m=tile_m)
        for name, tensor in actual.items():
            if tensor is not None:
                assert_bitwise({name: tensor}, {name: expected[name]})
    torch.cuda.current_stream().wait_stream(noise_stream)


def unpack_scales(scales, rows, cols):
    return scales.view(-1, rows // 128, cols // 4, 32, 4, 4).permute(0, 1, 4, 3, 2, 5).reshape(-1, rows, cols)


def snapshot_inputs(problem):
    options, b, sb, sa, _ = problem
    inputs = {name: value.detach().clone() for name, value in options.items() if isinstance(value, torch.Tensor)}
    inputs["b"] = b.view(torch.uint8).clone().view(b.dtype)
    if sb is not None:
        inputs.update(sb=sb.clone(), sa=sa.clone())
    return inputs


def select_experts(problem, selected):
    options, b, sb, sa, counts = problem
    starts = [sum(counts[:i]) for i in range(len(counts))]
    selected_options = dict(options)
    selected_options["a_tensor"] = torch.cat([options["a_tensor"][starts[i] : starts[i] + counts[i]].view(torch.uint8) for i in selected]).view(b.dtype)
    selected_sa = None if sa is None else torch.cat([sa[:, starts[i] : starts[i] + counts[i]].view(torch.uint8) for i in selected], dim=1).view(sa.dtype)
    selected_options["sfa_tensor"] = pack_scales(selected_sa)
    selected_options["prob_tensor"] = torch.cat([options["prob_tensor"][starts[i] : starts[i] + counts[i]] for i in selected])
    selected_options["alpha_tensor"] = options["alpha_tensor"][list(selected)].clone()
    selected_options["bias_tensor"] = options["bias_tensor"].t()[list(selected)].contiguous().t()
    selected_counts = tuple(counts[i] for i in selected)
    selected_options["padded_offsets"] = torch.tensor(selected_counts, device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    return (
        selected_options,
        b.view(torch.uint8)[list(selected)].view(b.dtype),
        None if sb is None else sb.view(torch.uint8)[list(selected)].view(sb.dtype),
        selected_sa,
        selected_counts,
    )


def expert_outputs(outputs, start, count):
    m, f = outputs["d_tensor"].shape
    result = dict(
        c=outputs["c_tensor"][start : start + count],
        d=outputs["d_tensor"][start : start + count],
    )
    if outputs.get("sfd_row_tensor") is not None:
        result.update(
            d_col=outputs["d_col_tensor"][start : start + count],
            sf_row_bits=unpack_scales(outputs["sfd_row_tensor"], m, f // 32)[:, start : start + count].view(torch.uint8),
            sf_col_bits=unpack_scales(outputs["sfd_col_tensor"], f, m // 32)[:, :, start // 32 : (start + count) // 32].view(torch.uint8),
        )
    return result


@pytest.mark.parametrize("tile_m", [128, 256])
@pytest.mark.parametrize("precision", ["bf16", "nvfp4", "mxfp8"])
def test_gate_then_up_batch_invariance(precision, tile_m):
    problem = make_problem(f=1024, counts=(256, 512, 2048), precision=precision)
    target = select_experts(problem, (0,))
    expected = expert_outputs(call_problem(target, layout="gate_then_up", tile_m=tile_m), 0, 256)
    for selected in ((0, 1, 2), (2, 0, 1), (0, 0, 2), (0, 2)):
        packed = select_experts(problem, selected)
        output = call_problem(packed, layout="gate_then_up", tile_m=tile_m)
        start = 0
        for i, count in zip(selected, packed[-1]):
            if i == 0:
                assert_bitwise(expert_outputs(output, start, count), expected, f"packing={selected}")
            start += count
    # Neighbor changes cannot influence the first expert. Preserve all target data.
    changed = select_experts(problem, (0, 1, 2))
    changed[0]["a_tensor"][256:].view(torch.uint8).zero_()
    changed[1][1:].view(torch.uint8).zero_()
    assert_bitwise(expert_outputs(call_problem(changed, layout="gate_then_up", tile_m=tile_m), 0, 256), expected, "changed neighbors")


@pytest.mark.parametrize("tile_m", [128, 256])
@pytest.mark.parametrize("precision", ["bf16", "nvfp4", "mxfp8"])
def test_gate_then_up_graph_replay_changed_weights_and_scales(precision, tile_m):
    from cudnn import grouped_gemm_glu_wrapper_sm100

    problem = make_problem(f=256, counts=(256, 512), precision=precision)
    options, b, sb, _, _ = problem
    kwargs = dict(
        options,
        b_tensor=b,
        sfb_tensor=pack_scales(sb),
        glu_layout="gate_then_up",
        mma_tiler_mn=(tile_m, 256),
        cluster_shape_mn=(2, 1) if tile_m == 256 else (1, 1),
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        grouped_gemm_glu_wrapper_sm100(**kwargs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = grouped_gemm_glu_wrapper_sm100(**kwargs)
    for change in ("weights",) if precision == "bf16" else ("weights", "scales"):
        if change == "weights":
            if precision == "nvfp4":
                kwargs["b_tensor"].view(torch.uint8).bitwise_xor_(1)
            else:
                kwargs["b_tensor"].copy_((kwargs["b_tensor"].float() * 0.5).to(kwargs["b_tensor"].dtype))
        else:
            kwargs["sfb_tensor"].view(torch.uint8).add_(1)
        expected = grouped_gemm_glu_wrapper_sm100(**kwargs)
        graph.replay()
        torch.cuda.synchronize()
        live_names = ("c_tensor", "d_tensor", "amax_tensor") if precision != "mxfp8" else tuple(expected.keys())
        assert_bitwise({k: captured[k] for k in live_names if captured[k] is not None}, {k: expected[k] for k in live_names if expected[k] is not None}, change)


@pytest.mark.parametrize("option", ["layout", "e5m2", "branch_width", "major", "saved_dtype", "output_dtype", "activation"])
def test_gate_then_up_rejects_unsupported_contract(option):
    from cudnn import grouped_gemm_glu_wrapper_sm100

    options, b, sb, _, _ = make_problem(f=256)
    kwargs = dict(options, b_tensor=b, sfb_tensor=pack_scales(sb), glu_layout="gate_then_up")
    error = NotImplementedError
    match = "gate_then_up"
    if option == "layout":
        kwargs["glu_layout"] = "not_a_layout"
        error, match = ValueError, "glu_layout"
    elif option == "e5m2":
        kwargs.update(a_tensor=kwargs["a_tensor"].to(torch.float8_e5m2), b_tensor=b.to(torch.float8_e5m2))
    elif option == "branch_width":
        # Warm the dynamic plan first: a cache hit must still enforce N.
        grouped_gemm_glu_wrapper_sm100(**kwargs)
        # N=128 is valid on the legacy interleave path, but a canonical branch
        # must contain a whole 128-row scale atom in the initial implementation.
        kwargs["b_tensor"] = b[:, :128].contiguous()
        kwargs["sfb_tensor"] = pack_scales(sb[:, :128].contiguous())
        kwargs["bias_tensor"] = options["bias_tensor"][:128].t().contiguous().t()
    elif option == "major":
        # Dense mode infers major order from storage; the b_major hint is for
        # pointer arrays. Build an actual N-major (N,K,E) weight view.
        kwargs["b_tensor"] = b.permute(0, 2, 1).contiguous().permute(2, 1, 0)
        kwargs["b_major"] = "n"
        error, match = ValueError, "k-major"
    elif option == "saved_dtype":
        kwargs["c_dtype"] = torch.float16
    elif option == "output_dtype":
        kwargs["d_dtype"] = torch.bfloat16
    elif option == "activation":
        kwargs["act_func"] = "geglu"
    with pytest.raises(error, match=match):
        grouped_gemm_glu_wrapper_sm100(**kwargs)


def unpack_fp4(tensor):
    # Independent E2M1 decoder; quantized payload is never requantized.
    table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device=tensor.device)
    bits = tensor.view(torch.uint8)
    return torch.stack((table[(bits & 15).long()], table[(bits >> 4).long()]), dim=-1).flatten(-2)


@pytest.mark.parametrize("precision", ["bf16", "nvfp4"])
@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
@pytest.mark.parametrize("tile_m", [128, 256])
@pytest.mark.parametrize("f", [128, 1024])
def test_gate_then_up_bf16_nvfp4_reference_replay(precision, discrete, tile_m, f):
    problem = make_problem(f=f, precision=precision)
    problem[0]["use_dynamic_sched"] = True
    original = snapshot_inputs(problem)
    expected = call_problem(problem, layout="interleaved_32", discrete=discrete, tile_m=tile_m)
    expected["c_tensor"] = canonicalize_c(expected["c_tensor"])
    actual = call_problem(problem, layout="gate_then_up", discrete=discrete, tile_m=tile_m)
    assert_bitwise({k: v for k, v in actual.items() if v is not None}, {k: v for k, v in expected.items() if v is not None}, precision)
    saved = {name: tensor.clone() for name, tensor in actual.items() if tensor is not None}
    options, b, sb, sa, counts = problem
    if precision == "nvfp4":
        a_deq = unpack_fp4(options["a_tensor"]) * sa[0].float().repeat_interleave(16, dim=1)
        b_deq = unpack_fp4(b) * sb.float().repeat_interleave(16, dim=2)
    else:
        a_deq, b_deq = options["a_tensor"].float(), b.float()
    start = 0
    for expert, count in enumerate(counts):
        end = start + count
        pre = (a_deq[start:end].double() @ b_deq[expert].double().t()) * options["alpha_tensor"][expert].double()
        pre += options["bias_tensor"][:, expert].double()
        out = torch.nn.functional.silu(pre[:, :f]) * pre[:, f:] * options["prob_tensor"][start:end, None]
        torch.testing.assert_close(actual["c_tensor"][start:end].float(), pre.float(), rtol=0.01, atol=0.005)
        torch.testing.assert_close(actual["d_tensor"][start:end].float(), out.float(), rtol=0.015, atol=0.005)
        start = end
    for _ in range(3):
        replay = call_problem(problem, layout="gate_then_up", discrete=discrete, tile_m=tile_m)
        assert_bitwise({name: value for name, value in replay.items() if value is not None}, saved, precision)
    assert_bitwise(snapshot_inputs(problem), original, "inputs preserved")


@pytest.mark.parametrize("precision", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
def test_gate_then_up_legacy_swiglu_aliases(precision, discrete):
    from cudnn.gemm.cutedsl.grouped.swiglu.api import GroupedGemmSwigluSm100, grouped_gemm_swiglu_wrapper_sm100
    from cudnn.gemm.cutedsl.discrete_grouped.swiglu.api import DiscreteGroupedGemmSwigluSm100, discrete_grouped_gemm_swiglu_wrapper_sm100

    problem = make_problem(f=256, counts=(256, 512), precision=precision)
    options, b, sb, _, _ = problem
    options["bias_tensor"] = None
    expected = call_problem(problem, layout="gate_then_up", discrete=discrete)
    scales = pack_scales(sb)
    config = dict(glu_layout="gate_then_up", mma_tiler_mn=(128, 256), cluster_shape_mn=(1, 1), sf_vec_size=options["sf_vec_size"])
    call = {name: value for name, value in options.items() if name not in ("generate_c", "bias_tensor", "sf_vec_size")}
    if discrete:
        weights = dict(
            b_ptrs=torch.tensor([x.data_ptr() for x in b], dtype=torch.int64, device="cuda"),
            sfb_ptrs=torch.tensor([scales.data_ptr() + i * scales.numel() // len(b) for i in range(len(b))], dtype=torch.int64, device="cuda"),
        )
        result = discrete_grouped_gemm_swiglu_wrapper_sm100(**call, **weights, n=b.shape[1], b_dtype=b.dtype, **config)
    else:
        weights = dict(b_tensor=b, sfb_tensor=scales)
        result = grouped_gemm_swiglu_wrapper_sm100(**call, **weights, **config)
    live = {name: value for name, value in result.items() if name in expected and value is not None}
    assert_bitwise(live, {name: expected[name] for name in live}, "wrapper alias")

    # The low-level alias must honor caller-owned output buffers too.
    outputs = {name: torch.empty_like(value) if value is not None else None for name, value in result.items()}
    if outputs["amax_tensor"] is not None:
        outputs["amax_tensor"].fill_(-torch.inf)
    samples = dict(
        sample_a=options["a_tensor"],
        sample_c=outputs["c_tensor"],
        sample_d=outputs["d_tensor"],
        sample_d_col=outputs["d_col_tensor"],
        sample_sfa=options["sfa_tensor"],
        sample_padded_offsets=options["padded_offsets"],
        sample_alpha=options["alpha_tensor"],
        sample_sfd_row=outputs["sfd_row_tensor"],
        sample_sfd_col=outputs["sfd_col_tensor"],
        sample_amax=outputs["amax_tensor"],
        sample_norm_const=options["norm_const_tensor"],
        sample_prob=options["prob_tensor"],
    )
    if discrete:
        api = DiscreteGroupedGemmSwigluSm100(**samples, num_experts=len(b), b_shape=(b.shape[1], 512), b_dtype=b.dtype, **config)
    else:
        api = GroupedGemmSwigluSm100(**samples, sample_b=b, sample_sfb=scales, **config)
    assert api.check_support()
    api.compile()
    execute_options = {name: value for name, value in call.items() if name not in ("c_dtype", "d_dtype")}
    api.execute(**execute_options, **weights, **outputs)
    torch.cuda.synchronize()
    assert_bitwise({name: outputs[name] for name in live}, {name: expected[name] for name in live}, "preallocated alias")


@pytest.mark.parametrize("precision", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
def test_gate_then_up_two_cta_scale_pipeline_tail(precision, discrete):
    # K=1664 wraps the five AB stages and leaves an NVFP4 K-tile tail.
    # Nonuniform scales and an empty expert detect wrong stage/phase ownership.
    problem = make_problem(f=128, k=1664, counts=(256, 0, 512), precision=precision)
    original = snapshot_inputs(problem)
    expected = call_problem(problem, layout="interleaved_32", discrete=discrete, tile_m=256)
    expected["c_tensor"] = canonicalize_c(expected["c_tensor"])
    actual = call_problem(problem, layout="gate_then_up", discrete=discrete, tile_m=256)
    saved = {name: tensor.clone() for name, tensor in actual.items() if tensor is not None}
    assert_bitwise(saved, {name: tensor for name, tensor in expected.items() if tensor is not None}, precision)

    options, b, sb, sa, counts = problem
    sf_vec = 16 if precision == "nvfp4" else 32
    a_data = unpack_fp4(options["a_tensor"]) if precision == "nvfp4" else options["a_tensor"].float()
    b_data = unpack_fp4(b) if precision == "nvfp4" else b.float()
    a_deq = a_data * sa[0].float().repeat_interleave(sf_vec, dim=1)
    b_deq = b_data * sb.float().repeat_interleave(sf_vec, dim=2)
    reference_d = torch.empty_like(actual["d_tensor"], dtype=torch.float32)
    start = 0
    for expert, count in enumerate(counts):
        end = start + count
        pre = (a_deq[start:end].double() @ b_deq[expert].double().t()) * options["alpha_tensor"][expert].double()
        pre += options["bias_tensor"][:, expert].double()
        reference_d[start:end] = (torch.nn.functional.silu(pre[:, :128]) * pre[:, 128:] * options["prob_tensor"][start:end, None]).float()
        torch.testing.assert_close(actual["c_tensor"][start:end].float(), pre.float(), rtol=0.01, atol=0.005)
        start = end
    if precision == "nvfp4":
        torch.testing.assert_close(actual["d_tensor"].float(), reference_d, rtol=0.015, atol=0.005)
    else:
        m = reference_d.shape[0]
        row_sf = unpack_scales(actual["sfd_row_tensor"], m, 4)[0].float()
        col_sf = unpack_scales(actual["sfd_col_tensor"], 128, m // 32)[0].float()
        row_deq = actual["d_tensor"].float() * row_sf.repeat_interleave(32, dim=1)
        col_deq = actual["d_col_tensor"].float() * col_sf.repeat_interleave(32, dim=1).t()
        for dequantized in (row_deq, col_deq):
            assert torch.isfinite(dequantized).all()
            torch.testing.assert_close(dequantized, reference_d, rtol=0.125, atol=0.002)
    for _ in range(3):
        replay = call_problem(problem, layout="gate_then_up", discrete=discrete, tile_m=256)
        assert_bitwise({name: tensor for name, tensor in replay.items() if tensor is not None}, saved, "stage wrap replay")
    assert_bitwise(snapshot_inputs(problem), original, "inputs preserved")
