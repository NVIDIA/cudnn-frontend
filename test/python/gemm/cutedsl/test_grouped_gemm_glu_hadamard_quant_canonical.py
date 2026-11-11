# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Natural-rank RHT+quant is a metadata adapter, independent of GLU branch order.

Every live payload and scale byte is compared against the existing kernel-facing
API. Numerical checks use independently dequantized double GEMM/SwiGLU/Hadamard
references. Replay, changed-input graph replay, and expert batch invariance are
separate from numerical tolerance checks.
"""

import pytest
import torch

from determinism_utils import assert_bitwise
from gemm.cutedsl.test_grouped_gemm_glu_branch_layout import canonicalize_c, make_problem, select_experts, snapshot_inputs
from gemm.cutedsl.test_grouped_gemm_glu_hadamard_quant_gate_then_up import (
    check_reference,
    expert_outputs,
    kwargs_for,
    live_bits,
)

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def require_blackwell():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Natural-rank GLU/RHT/quant qualification requires SM100/103")


def natural_kwargs(problem, mode, discrete=False, layout="gate_then_up"):
    kwargs, owners = kwargs_for(problem, mode, discrete, layout)
    kwargs["a_tensor"] = kwargs["a_tensor"].squeeze(-1)
    kwargs["prob_tensor"] = kwargs["prob_tensor"].reshape(-1)
    # These are views of the same physical MMA-packed bytes, not scale repacks.
    kwargs["sfa_tensor"] = kwargs["sfa_tensor"].permute(5, 2, 4, 0, 1, 3).view(-1)
    if not discrete:
        kwargs["b_tensor"] = kwargs["b_tensor"].permute(2, 0, 1)
        kwargs["sfb_tensor"] = kwargs["sfb_tensor"].permute(5, 2, 4, 0, 1, 3).view(-1)
    return kwargs, owners


def legacy_outputs(out):
    """Expose natural outputs as legacy views for the independent reference decoder."""
    result = dict(out)
    m = out["c_tensor"].shape[0]
    f = out["c_tensor"].shape[1] // 2
    for name in ("c_tensor", "d_tensor", "rht_rowwise_tensor"):
        if result[name] is not None:
            assert result[name].ndim == 2
            result[name] = result[name].unsqueeze(-1)
    for name in ("sfd_tensor", "sfrht_rowwise_tensor"):
        if result[name] is not None:
            assert result[name].ndim == 1 and result[name].is_contiguous()
            result[name] = result[name].view(1, (m + 127) // 128, (f + 63) // 64, 32, 4, 4).permute(3, 4, 1, 5, 2, 0)
    for name in ("rht_colwise_tensor", "sfrht_colwise_tensor"):
        if result[name] is not None:
            assert result[name].ndim == 1
    return result


def run_natural(problem, mode, discrete=False, layout="gate_then_up"):
    from cudnn import grouped_gemm_glu_hadamard_quant_wrapper_sm100 as call

    kwargs, owners = natural_kwargs(problem, mode, discrete, layout)
    result = call(**kwargs)
    torch.cuda.synchronize()
    return result


MODES = ["row_bf16", "row_fp4", "col_fp4"]


@pytest.mark.parametrize("layout", ["gate_then_up", "interleaved_32"])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
@pytest.mark.parametrize("f", [128, 1024])
def test_natural_rank_reference_legacy_replay(layout, mode, discrete, f):
    from cudnn import grouped_gemm_glu_hadamard_quant_wrapper_sm100 as call

    assert call.supports_canonical_layouts
    problem = make_problem(f=f, precision="nvfp4")
    original = snapshot_inputs(problem)
    kwargs, owners = kwargs_for(problem, mode, discrete, layout)
    old = call(**kwargs)
    natural = run_natural(problem, mode, discrete, layout)
    decoded = legacy_outputs(natural)
    assert_bitwise(live_bits(decoded), live_bits(old), "natural / kernel-facing tensor ranks")
    if layout == "interleaved_32":
        decoded["c_tensor"] = canonicalize_c(decoded["c_tensor"][:, :, 0]).unsqueeze(-1)
    check_reference(problem, decoded, mode)
    expected = {name: value.clone() for name, value in live_bits(natural).items()}
    noise = torch.ones(1024 * 1024, device="cuda")
    stream = torch.cuda.Stream()
    for _ in range(3):
        with torch.cuda.stream(stream):
            noise.mul_(1.001).add_(0.001)
        fresh = make_problem(f=f, precision="nvfp4")
        actual = run_natural(fresh, mode, discrete, layout)
        assert_bitwise(live_bits(actual), expected, "fresh allocations / stream noise")
    torch.cuda.current_stream().wait_stream(stream)
    assert_bitwise(snapshot_inputs(problem), original, "inputs preserved")


@pytest.mark.parametrize("layout", ["gate_then_up", "interleaved_32"])
@pytest.mark.parametrize("mode", MODES)
def test_natural_rank_batch_invariance(layout, mode):
    problem = make_problem(f=128, counts=(256, 512, 256), precision="nvfp4")
    expected = expert_outputs(legacy_outputs(run_natural(select_experts(problem, (0,)), mode, layout=layout)), 0, 256, 128)
    for selected in ((0, 1, 2), (2, 0, 1), (0, 0, 2), (0, 2)):
        packed = select_experts(problem, selected)
        out = legacy_outputs(run_natural(packed, mode, layout=layout))
        start = 0
        for index, count in zip(selected, packed[-1]):
            if index == 0:
                assert_bitwise(expert_outputs(out, start, count, 128), expected, str(selected))
            start += count
    changed = select_experts(problem, (0, 1, 2))
    changed[1][1:].view(torch.uint8).zero_()
    changed[0]["a_tensor"][256:].view(torch.uint8).zero_()
    out = legacy_outputs(run_natural(changed, mode, layout=layout))
    assert_bitwise(expert_outputs(out, 0, 256, 128), expected, "changed neighbors")


@pytest.mark.parametrize("layout", ["gate_then_up", "interleaved_32"])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("discrete", [False, True])
def test_natural_rank_graph_changed_inputs(layout, mode, discrete):
    from cudnn import grouped_gemm_glu_hadamard_quant_wrapper_sm100 as call

    kwargs, owners = natural_kwargs(make_problem(f=128, precision="nvfp4"), mode, discrete, layout)
    call(**kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call(**kwargs)
    targets = (kwargs["a_tensor"], kwargs["sfa_tensor"], *owners)
    for target in targets:
        target.view(torch.uint8).bitwise_xor_(1)
        expected = call(**kwargs)
        graph.replay()
        torch.cuda.synchronize()
        assert_bitwise(live_bits(captured), live_bits(expected), "changed graph input / packed scales")


@pytest.mark.parametrize("layout", ["gate_then_up", "interleaved_32"])
def test_natural_rank_dynamic_m_and_cache_guards(layout):
    from cudnn import grouped_gemm_glu_hadamard_quant_wrapper_sm100 as call
    from cudnn.gemm.cutedsl.grouped.glu_hadamard_quant.api import _cache_of_GroupedGemmGluHadamardQuantSm100Objects as cache

    cache.clear()
    for counts in ((256, 0, 512), (512, 256, 512)):
        kwargs, owners = natural_kwargs(make_problem(f=128, counts=counts, precision="nvfp4"), "row_fp4", layout=layout)
        call(**kwargs)
    assert len(cache) == 1, "Natural M and flat scale length must remain dynamic"
    bad = dict(kwargs, sfa_tensor=kwargs["sfa_tensor"][:-1])
    with pytest.raises(ValueError, match="complete MMA-packed"):
        call(**bad)
    bad = dict(kwargs, a_tensor=kwargs["a_tensor"][:, ::2])
    with pytest.raises(ValueError, match="contiguous"):
        call(**bad)
    # An arbitrary-rank contiguous physical scale buffer is also natural.
    kwargs["sfa_tensor"] = kwargs["sfa_tensor"].view(1, 1, 1, 1, 1, -1)
    kwargs["sfb_tensor"] = kwargs["sfb_tensor"].view(1, -1)
    call(**kwargs)
    torch.cuda.synchronize()


@pytest.mark.parametrize("layout", ["gate_then_up", "interleaved_32"])
@pytest.mark.parametrize("mode", ["row_fp4", "col_fp4"])
def test_natural_rank_prepared_outputs_without_host_views(layout, mode, monkeypatch):
    from cudnn import GroupedGemmGluHadamardQuantSm100, grouped_gemm_glu_hadamard_quant_wrapper_sm100 as call

    kwargs, owners = natural_kwargs(make_problem(f=128, precision="nvfp4"), mode, layout=layout)
    expected = call(**kwargs)
    out = {name: torch.empty_like(value) if value is not None else None for name, value in expected.items()}
    # Contiguous rank-six scale storage must remain physical, not be mistaken for
    # the old noncontiguous logical six-dimensional MMA view.
    for name in ("sfd_tensor", "sfrht_rowwise_tensor"):
        if out[name] is not None:
            out[name] = out[name].view(1, 1, 1, 1, 1, -1)
    api = GroupedGemmGluHadamardQuantSm100(
        sample_a=kwargs["a_tensor"],
        sample_b=kwargs["b_tensor"],
        sample_c=out["c_tensor"],
        sample_d=out["d_tensor"],
        sample_sfa=kwargs["sfa_tensor"],
        sample_sfb=kwargs["sfb_tensor"],
        sample_padded_offsets=kwargs["padded_offsets"],
        sample_alpha=kwargs["alpha_tensor"],
        sample_prob=kwargs["prob_tensor"],
        sample_bias=kwargs["bias_tensor"],
        sample_sfd=out["sfd_tensor"],
        sample_rht_rowwise=out["rht_rowwise_tensor"],
        sample_sfrht_rowwise=out["sfrht_rowwise_tensor"],
        sample_rht_colwise=out["rht_colwise_tensor"],
        sample_sfrht_colwise=out["sfrht_colwise_tensor"],
        glu_layout=layout,
        mma_tiler_mn=kwargs["mma_tiler_mn"],
        cluster_shape_mn=kwargs["cluster_shape_mn"],
        use_dynamic_sched=True,
    )
    assert api.supports_canonical_layouts
    api.check_support()
    api.compile()
    execute = {
        name: kwargs[name] for name in ("a_tensor", "b_tensor", "sfa_tensor", "sfb_tensor", "padded_offsets", "alpha_tensor", "prob_tensor", "bias_tensor")
    }
    execute.update(out, norm_const=16.0, rht_norm_const=16.0)

    def forbid_view(*args, **kwargs):
        raise AssertionError("Prepared execute must not construct framework tensor views")

    with monkeypatch.context() as patch:
        for method in ("view", "reshape", "permute", "unsqueeze", "as_strided"):
            patch.setattr(torch.Tensor, method, forbid_view)
        api.execute(**execute)
    torch.cuda.synchronize()
    for name in ("sfd_tensor", "sfrht_rowwise_tensor"):
        if out[name] is not None:
            out[name] = out[name].view(-1)
    assert_bitwise(live_bits(out), live_bits(expected), "caller-owned natural outputs / physical rank-six scales")


@pytest.mark.parametrize("layout", ["gate_then_up", "interleaved_32"])
@pytest.mark.parametrize("empty", [False, True])
def test_natural_rank_no_rht_and_empty(layout, empty):
    from cudnn import grouped_gemm_glu_hadamard_quant_wrapper_sm100 as call

    counts = (0, 0, 0) if empty else (256, 0, 512)
    problem = make_problem(f=128, counts=counts, precision="nvfp4")
    kwargs, owners = natural_kwargs(problem, "no_rht", layout=layout)
    actual = call(**kwargs)
    legacy_kwargs, legacy_owners = kwargs_for(problem, "no_rht", layout=layout)
    old = call(**legacy_kwargs)
    assert_bitwise(live_bits(legacy_outputs(actual)), live_bits(old), "empty or no-RHT natural call")
    assert actual["rht_rowwise_tensor"] is None and actual["rht_colwise_tensor"] is None
    torch.cuda.synchronize()
