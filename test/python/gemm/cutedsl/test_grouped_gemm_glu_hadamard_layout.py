# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical NVFP4 SwiGLU + nonquant Hadamard forward qualification.

The API exposes pre-RHT D and pre/post-RHT amax; it has no backward entry point.
"""

import pytest
import torch

from determinism_utils import assert_bitwise, assert_replays
from gemm.cutedsl.test_grouped_gemm_glu_branch_layout import (
    canonicalize_c,
    make_problem,
    pack_scales,
    select_experts,
    snapshot_inputs,
    unpack_fp4,
)

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def require_blackwell():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Canonical GLU branches require SM100/103")


def prepare(problem, *, layout="gate_then_up", discrete=False, tmem=False):
    options, b, sb, _, _ = problem
    args = {name: value for name, value in options.items() if name not in ("norm_const_tensor", "generate_c")}
    if layout == "interleaved_32":
        n = b.shape[1]
        order = torch.arange(n, device=b.device).view(2, n // 64, 32).permute(1, 0, 2).reshape(-1)
        b = b.view(torch.uint8).index_select(1, order).view(b.dtype)
        sb = sb.view(torch.uint8).index_select(1, order).view(sb.dtype)
        if args["bias_tensor"] is not None:
            args["bias_tensor"] = args["bias_tensor"].t().index_select(1, order).t()
    scales = pack_scales(sb)
    owners = (b, scales)
    if discrete:
        args.update(
            b_ptrs=torch.tensor([b[i].data_ptr() for i in range(b.shape[0])], device="cuda", dtype=torch.int64),
            sfb_ptrs=torch.tensor([scales.data_ptr() + i * scales.numel() // b.shape[0] for i in range(b.shape[0])], device="cuda", dtype=torch.int64),
            n=b.shape[1],
            b_dtype=b.dtype,
        )
    else:
        args.update(b_tensor=b, sfb_tensor=scales)
    args.update(glu_layout=layout, mma_tiler_mn=(256, 256), cluster_shape_mn=(2, 1), use_tmem_post_rht_amax=tmem, use_dynamic_sched=True)
    return args, owners


def invoke(args):
    from cudnn import grouped_gemm_glu_hadamard_wrapper_sm100

    return dict(grouped_gemm_glu_hadamard_wrapper_sm100(**args))


def reference(problem, actual):
    options, b, sb, sa, counts = problem
    a_deq = unpack_fp4(options["a_tensor"]) * sa[0].float().repeat_interleave(16, dim=1)
    b_deq = unpack_fp4(b) * sb.float().repeat_interleave(16, dim=2)
    h = torch.ones((1, 1), dtype=torch.float32, device=b.device)
    for _ in range(4):
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
    f, start = b.shape[1] // 2, 0
    for expert, count in enumerate(counts):
        end = start + count
        if count:
            pre = (a_deq[start:end].double() @ b_deq[expert].double().t()) * options["alpha_tensor"][expert].double()
            if options["bias_tensor"] is not None:
                pre += options["bias_tensor"][:, expert].double()
            out = torch.nn.functional.silu(pre[:, :f]) * pre[:, f:] * options["prob_tensor"][start:end, None]
            torch.testing.assert_close(actual["c_tensor"][start:end].float(), pre.float(), rtol=0.01, atol=0.005)
            torch.testing.assert_close(actual["d_tensor"][start:end].float(), out.float(), rtol=0.015, atol=0.005)
            torch.testing.assert_close(actual["amax_tensor"][expert, 0], out.abs().max().float(), rtol=0.015, atol=0.005)
            # RHT acts on 16-token blocks after BF16 rounding of SwiGLU D.
            stored_d = actual["d_tensor"][start:end].float().t().contiguous().view(f, count // 16, 16)
            rht_amax = ((stored_d @ h) * 0.25).abs().max()
            torch.testing.assert_close(actual["post_rht_amax_tensor"][expert, 0], rht_amax, rtol=1e-4, atol=1e-4)
        else:
            assert actual["amax_tensor"][expert, 0] == 0
            assert actual["post_rht_amax_tensor"][expert, 0] == 0
        start = end


@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
@pytest.mark.parametrize("tmem", [False, True], ids=["smem_rht", "tmem_rht"])
@pytest.mark.parametrize("f,k", [(128, 512), (1024, 4096)])
def test_hadamard_canonical_reference_legacy_replay(discrete, tmem, f, k):
    problem = make_problem(f=f, k=k, precision="nvfp4")
    original = snapshot_inputs(problem)
    args, owners = prepare(problem, discrete=discrete, tmem=tmem)
    old_args, old_owners = prepare(problem, layout="interleaved_32", discrete=discrete, tmem=tmem)
    expected = invoke(old_args)
    expected["c_tensor"] = canonicalize_c(expected["c_tensor"])
    if discrete and tmem:
        # The old option remains available; canonical rejects this compiler
        # combination before launch instead of substituting another RHT path.
        reference(problem, expected)
        with pytest.raises(NotImplementedError, match="discrete weights and use_tmem_post_rht_amax=True"):
            invoke(args)
        assert_bitwise(snapshot_inputs(problem), original, "input bytes")
        return
    actual = invoke(args)
    assert_bitwise(actual, expected)
    reference(problem, actual)
    assert_replays(lambda: invoke(args), repeats=3)
    assert_bitwise(snapshot_inputs(problem), original, "input bytes")


def test_hadamard_canonical_no_bias():
    problem = make_problem(f=128, counts=(256,), precision="nvfp4")
    problem[0]["bias_tensor"] = None
    args, owners = prepare(problem, discrete=True)
    old_args, old_owners = prepare(problem, layout="interleaved_32", discrete=True)
    expected = invoke(old_args)
    expected["c_tensor"] = canonicalize_c(expected["c_tensor"])
    actual = invoke(args)
    assert_bitwise(actual, expected)
    reference(problem, actual)


@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
def test_hadamard_canonical_changed_graph_inputs(discrete):
    problem = make_problem(f=128, counts=(256, 0, 256), precision="nvfp4")
    args, owners = prepare(problem, discrete=discrete)
    invoke(args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = invoke(args)
    for value in (0.75, 0.25, 0.75):
        args["prob_tensor"].fill_(value)
        graph.replay()
        saved = {name: tensor.clone() for name, tensor in captured.items()}
        assert_bitwise(saved, invoke(args), "changed graph inputs")


def test_hadamard_canonical_expert_batch_invariance():
    problem = make_problem(f=128, counts=(256, 512, 768, 256), precision="nvfp4")
    original = snapshot_inputs(problem)
    args, owners = prepare(problem)
    packed = invoke(args)
    for selected in ((2,), (3, 0, 2), (1, 0, 1, 3)):
        selected_problem = select_experts(problem, selected)
        selected_args, selected_owners = prepare(selected_problem)
        result = invoke(selected_args)
        start = 0
        for position, expert in enumerate(selected):
            count = problem[4][expert]
            original_start = sum(problem[4][:expert])
            for name in ("c_tensor", "d_tensor"):
                assert_bitwise({name: result[name][start : start + count]}, {name: packed[name][original_start : original_start + count]})
            for name in ("amax_tensor", "post_rht_amax_tensor"):
                assert_bitwise({name: result[name][position]}, {name: packed[name][expert]})
            start += count
    # A different neighboring expert must not alter the retained expert's bytes.
    changed_problem = select_experts(problem, (0, 1))
    changed_problem[0]["a_tensor"][256:].view(torch.uint8).zero_()
    changed_args, changed_owners = prepare(changed_problem)
    changed = invoke(changed_args)
    for name in ("c_tensor", "d_tensor"):
        assert_bitwise({name: changed[name][:256]}, {name: packed[name][:256]})
    for name in ("amax_tensor", "post_rht_amax_tensor"):
        assert_bitwise({name: changed[name][0]}, {name: packed[name][0]})
    assert_bitwise(snapshot_inputs(problem), original, "input bytes")


def test_hadamard_canonical_reference_and_replay():
    problem = make_problem(f=128, k=512, counts=(256,), precision="nvfp4")
    args, owners = prepare(problem)
    reference(problem, invoke(args))
    assert_replays(lambda: invoke(args), repeats=4)
