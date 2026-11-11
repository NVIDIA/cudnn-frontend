# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical FC1 storage through the existing fused RHT/NVFP4 epilogue."""

import pytest
import torch

from determinism_utils import assert_bitwise
from gemm.cutedsl.test_grouped_gemm_glu_branch_layout import (
    make_problem,
    pack_scales,
    unpack_fp4,
    canonicalize_c,
    snapshot_inputs,
    select_experts,
)
from gemm.cutedsl.test_grouped_gemm_glu_hadamard_quant import (
    _rht_ref,
    _check_nvfp4_output,
    _check_colwise_rht,
    _swizzled_sf_to_flat,
)

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def require_blackwell():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Canonical RHT quant requires SM100/103")


def sf_view(sf):
    e, rows, cols = sf.shape
    return pack_scales(sf).view(e, rows // 128, cols // 4, 32, 4, 4).permute(3, 4, 1, 5, 2, 0)


def kwargs_for(problem, mode, discrete=False, layout="gate_then_up"):
    options, b, sb, sa, counts = problem
    m, k = options["a_tensor"].shape
    n, e = b.shape[1], b.shape[0]
    bias = options["bias_tensor"]
    if layout == "interleaved_32":
        order = torch.arange(n, device=b.device).view(2, n // 64, 32).permute(1, 0, 2).reshape(-1)
        b = b.view(torch.uint8).index_select(1, order).view(b.dtype)
        sb = sb.view(torch.uint8).index_select(1, order).view(sb.dtype)
        bias = bias.t().index_select(1, order).t()
    sfb = sf_view(sb)
    kwargs = dict(
        a_tensor=options["a_tensor"].as_strided((m, k, 1), (k, 1, m * k)),
        sfa_tensor=sf_view(sa),
        padded_offsets=options["padded_offsets"],
        alpha_tensor=options["alpha_tensor"],
        prob_tensor=options["prob_tensor"].as_strided((m, 1, 1), (1, m, m)),
        bias_tensor=bias,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16 if mode == "row_bf16" else torch.float4_e2m1fn_x2,
        rht_rowwise_dtype=torch.bfloat16 if mode == "row_bf16" else torch.float4_e2m1fn_x2 if mode == "row_fp4" else None,
        rht_colwise_dtype=torch.float4_e2m1fn_x2 if mode == "col_fp4" else None,
        norm_const=16.0,
        rht_norm_const=16.0,
        glu_layout=layout,
        mma_tiler_mn=(128, 256) if layout == "gate_then_up" else (256, 256),
        cluster_shape_mn=(1, 1) if layout == "gate_then_up" else (2, 1),
        use_dynamic_sched=True,
    )
    if discrete:
        kwargs.update(
            b_ptrs=torch.tensor([x.data_ptr() for x in b], dtype=torch.int64, device=b.device),
            sfb_ptrs=torch.tensor([sfb.data_ptr() + i * sfb.numel() // e for i in range(e)], dtype=torch.int64, device=b.device),
            n=n,
            b_dtype=b.dtype,
        )
    else:
        kwargs.update(b_tensor=b.permute(1, 2, 0), sfb_tensor=sfb)
    return kwargs, (b, sfb)


def run(problem, mode, discrete=False, layout="gate_then_up"):
    from cudnn import grouped_gemm_glu_hadamard_quant_wrapper_sm100

    kwargs, owners = kwargs_for(problem, mode, discrete, layout)
    out = grouped_gemm_glu_hadamard_quant_wrapper_sm100(**kwargs)
    torch.cuda.synchronize()
    return out


def live_bits(out):
    return {key: value.view(torch.uint8) if value.dtype == torch.float4_e2m1fn_x2 else value for key, value in out.items() if value is not None}


def check_reference(problem, out, mode):
    options, b, sb, sa, counts = problem
    f = b.shape[1] // 2
    a_deq = unpack_fp4(options["a_tensor"]) * sa[0].float().repeat_interleave(16, dim=1)
    b_deq = unpack_fp4(b) * sb.float().repeat_interleave(16, dim=2)
    ref_d = torch.empty((a_deq.shape[0], f), device=b.device)
    start = 0
    for expert, count in enumerate(counts):
        end = start + count
        pre = a_deq[start:end].double() @ b_deq[expert].double().t()
        pre = pre * options["alpha_tensor"][expert].double() + options["bias_tensor"][:, expert].double()
        torch.testing.assert_close(out["c_tensor"][start:end, :, 0].float(), pre.float(), atol=0.005, rtol=0.01)
        ref_d[start:end] = (torch.nn.functional.silu(pre[:, :f]) * pre[:, f:] * options["prob_tensor"][start:end, None]).float()
        start = end
    rht = _rht_ref(ref_d.unsqueeze(-1), rowwise=mode != "col_fp4")
    if mode == "row_bf16":
        torch.testing.assert_close(out["d_tensor"][:, :, 0].float(), ref_d, atol=0.005, rtol=0.015)
        torch.testing.assert_close(out["rht_rowwise_tensor"][:, :, 0].float(), rht.float(), atol=0.01, rtol=0.03)
    else:
        _check_nvfp4_output(unpack_fp4(out["d_tensor"][:, :, 0]), out["sfd_tensor"], ref_d, 16.0, "D")
        if mode == "row_fp4":
            _check_nvfp4_output(unpack_fp4(out["rht_rowwise_tensor"][:, :, 0]), out["sfrht_rowwise_tensor"], rht.float(), 16.0, "RHT rowwise")
        else:
            _check_colwise_rht({"aligned_group_m_list": counts}, out, rht.cpu(), 16.0, None)


@pytest.mark.parametrize("mode", ["row_bf16", "row_fp4", "col_fp4"])
@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
@pytest.mark.parametrize("f", [128, 1024])
def test_canonical_rht_quant_reference_legacy_replay(mode, discrete, f):
    problem = make_problem(f=f, precision="nvfp4")
    original = snapshot_inputs(problem)
    old = run(problem, mode, discrete, "interleaved_32")
    old["c_tensor"] = canonicalize_c(old["c_tensor"][:, :, 0]).unsqueeze(-1)
    out = run(problem, mode, discrete)
    assert_bitwise(live_bits(out), live_bits(old), "legacy32")
    check_reference(problem, out, mode)
    expected = {name: value.clone() for name, value in live_bits(out).items()}
    noise_stream = torch.cuda.Stream()
    noise = torch.ones(1024 * 1024, device="cuda")
    for _ in range(3):
        with torch.cuda.stream(noise_stream):
            noise.mul_(1.001).add_(0.001)
        fresh = make_problem(f=f, precision="nvfp4")
        assert_bitwise(live_bits(run(fresh, mode, discrete)), expected, "fresh allocation / noise replay")
    torch.cuda.current_stream().wait_stream(noise_stream)
    assert_bitwise(snapshot_inputs(problem), original, "inputs preserved")


@pytest.mark.parametrize("mode", ["row_bf16", "row_fp4", "col_fp4"])
def test_canonical_rht_quant_graph_changed_weights_scales(mode):
    from cudnn import grouped_gemm_glu_hadamard_quant_wrapper_sm100 as call

    problem = make_problem(f=128, precision="nvfp4")
    kwargs, owners = kwargs_for(problem, mode)
    call(**kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call(**kwargs)
    for key in ("b_tensor", "sfb_tensor"):
        kwargs[key].view(torch.uint8).bitwise_xor_(1)
        expected = call(**kwargs)
        graph.replay()
        torch.cuda.synchronize()
        assert_bitwise(live_bits(captured), live_bits(expected), key)


def expert_outputs(out, start, count, f):
    selected = {}
    for key, value in out.items():
        if value is None:
            continue
        if key == "rht_colwise_tensor":
            selected[key] = value.view(torch.uint8)[start * f // 2 : (start + count) * f // 2]
        elif key == "sfrht_colwise_tensor":
            selected[key] = value[start * f // 16 : (start + count) * f // 16]
        elif key in ("sfd_tensor", "sfrht_rowwise_tensor"):
            selected[key] = _swizzled_sf_to_flat(value, out["c_tensor"].shape[0], f)[start : start + count]
        else:
            selected[key] = (value.view(torch.uint8) if value.dtype == torch.float4_e2m1fn_x2 else value)[start : start + count]
    return selected


@pytest.mark.parametrize("mode", ["row_bf16", "row_fp4", "col_fp4"])
def test_canonical_rht_quant_batch_invariance(mode):
    problem = make_problem(f=128, counts=(256, 512, 256), precision="nvfp4")
    target = select_experts(problem, (0,))
    expected = expert_outputs(run(target, mode), 0, 256, 128)
    for selected in ((0, 1, 2), (2, 0, 1), (0, 0, 2), (0, 2)):
        packed = select_experts(problem, selected)
        out = run(packed, mode)
        start = 0
        for index, count in zip(selected, packed[-1]):
            if index == 0:
                assert_bitwise(expert_outputs(out, start, count, 128), expected, str(selected))
            start += count
    changed = select_experts(problem, (0, 1, 2))
    changed[1][1:].view(torch.uint8).zero_()
    changed[0]["a_tensor"][256:].view(torch.uint8).zero_()
    assert_bitwise(expert_outputs(run(changed, mode), 0, 256, 128), expected, "changed neighbors")


@pytest.mark.parametrize("option", ["layout", "tile", "branch", "activation"])
def test_canonical_rht_quant_rejects_unsupported(option):
    from cudnn import grouped_gemm_glu_hadamard_quant_wrapper_sm100 as call

    kwargs, owners = kwargs_for(make_problem(f=128, precision="nvfp4"), "row_fp4")
    if option == "layout":
        kwargs["glu_layout"] = "invalid"
    elif option == "tile":
        kwargs["mma_tiler_mn"], kwargs["cluster_shape_mn"] = (256, 256), (2, 1)
    elif option == "branch":
        kwargs["b_tensor"] = kwargs["b_tensor"][:128]
    else:
        kwargs["act_func"] = "geglu"
    with pytest.raises(ValueError, match="glu_layout|gate_then_up"):
        call(**kwargs)


@pytest.mark.parametrize("layout", ["interleaved_32", "gate_then_up"])
@pytest.mark.parametrize("f,discrete", [(128, False), (1024, True)])
def test_rht_quant_mma_scales_reference_replay_graph(layout, f, discrete):
    from cudnn import grouped_gemm_glu_hadamard_quant_wrapper_sm100 as call

    problem = make_problem(f=f, precision="nvfp4")
    original = snapshot_inputs(problem)
    out = run(problem, "col_fp4", discrete, layout)
    expected = {name: value.clone() for name, value in live_bits(out).items()}
    canonical = dict(out)
    if layout == "interleaved_32":
        canonical["c_tensor"] = canonicalize_c(out["c_tensor"][:, :, 0]).unsqueeze(-1)
    check_reference(problem, canonical, "col_fp4")
    noise_stream = torch.cuda.Stream()
    noise = torch.ones(1024 * 1024, device="cuda")
    for _ in range(3):
        with torch.cuda.stream(noise_stream):
            noise.mul_(1.001).add_(0.001)
        fresh = make_problem(f=f, precision="nvfp4")
        assert_bitwise(live_bits(run(fresh, "col_fp4", discrete, layout)), expected, "MMA fresh/noise replay")
    torch.cuda.current_stream().wait_stream(noise_stream)
    assert_bitwise(snapshot_inputs(problem), original, "MMA inputs preserved")
    kwargs, owners = kwargs_for(problem, "col_fp4", discrete, layout)
    call(**kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call(**kwargs)
    owners[0].view(torch.uint8).bitwise_xor_(1)
    owners[1].view(torch.uint8).bitwise_xor_(1)
    eager = call(**kwargs)
    graph.replay()
    torch.cuda.synchronize()
    assert_bitwise(live_bits(captured), live_bits(eager), "MMA graph changed weights and scales")


@pytest.mark.parametrize("layout", ["interleaved_32", "gate_then_up"])
def test_rht_quant_mma_scales_batch_invariance(layout):
    problem = make_problem(f=128, counts=(256, 512, 256), precision="nvfp4")
    target = select_experts(problem, (0,))
    expected = expert_outputs(run(target, "col_fp4", layout=layout), 0, 256, 128)
    for selected in ((0, 1, 2), (2, 0, 1), (0, 0, 2), (0, 2)):
        packed = select_experts(problem, selected)
        out = run(packed, "col_fp4", layout=layout)
        start = 0
        for index, count in zip(selected, packed[-1]):
            if index == 0:
                assert_bitwise(expert_outputs(out, start, count, 128), expected, str(selected))
            start += count
    changed = select_experts(problem, (0, 1, 2))
    changed[1][1:].view(torch.uint8).zero_()
    changed[0]["a_tensor"][256:].view(torch.uint8).zero_()
    assert_bitwise(expert_outputs(run(changed, "col_fp4", layout=layout), 0, 256, 128), expected, "MMA changed neighbors")
