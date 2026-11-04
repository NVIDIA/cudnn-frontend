# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KDA execution and output-layout regressions through the public ops and graph API."""

import math

import pytest
import torch

import cudnn
from cudnn.linear_attention.ops import kimi_delta_attention

from linear_attention.test_la import Backend, CUDNN_DTYPE, LEAF_NAMES, assert_bwd_parity, frost_la_unavailable, make_case, thd_tensors

requirement_error = frost_la_unavailable()
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


@pytest.fixture(autouse=True)
def blackwell():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family Frost KDA")


def test_kda_retains_history_under_tiny_channel_gates():
    """KDA keeps state that a per-channel gate product would call decayed.

    With two equal live key channels, choosing alpha so that
    ``alpha * (1 - ||k||^2) == -1`` preserves the state component parallel to
    k even though every individual channel has ``alpha**16 < exp(-10)``: the
    rank-one update routes the state around the decay.  An impulse 17 tokens
    before the checked window must remain visible there.
    """
    total, heads, dim = 3840, 96, 64
    window, impulse = 1280, 1263
    live = 1.2265625  # exactly representable in bfloat16
    alpha = 1.0 / (2.0 * live * live - 1.0)
    log_alpha = float(torch.tensor(math.log(alpha), dtype=torch.float32))

    q = torch.zeros(total, heads, dim, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros_like(q)
    q[..., :2] = live
    k[..., :2] = live
    v = torch.zeros_like(q)
    v[impulse, :, 0] = 1.0
    g = torch.full((total, heads, dim), log_alpha, dtype=torch.float32, device="cuda")
    beta = torch.ones(total, heads, dtype=torch.float32, device="cuda")
    cu = torch.tensor([0, total], dtype=torch.int32, device="cuda")

    actual, _ = kimi_delta_attention(q, k, v, g, beta, cu, plan_name="kda_frost")

    # Independent recurrence over the only two live key channels.  Use the
    # rounded float32 gate value consumed by the kernel, not ideal alpha.
    rounded_alpha = math.exp(log_alpha)
    state = [0.0, 0.0]
    expected = []
    scale = 1.0 / math.sqrt(dim)
    for token in range(total):
        state[0] *= rounded_alpha
        state[1] *= rounded_alpha
        residual = (1.0 if token == impulse else 0.0) - live * (state[0] + state[1])
        state[0] += live * residual
        state[1] += live * residual
        if window <= token < window + 16:
            expected.append(scale * live * (state[0] + state[1]))

    expected = torch.tensor(expected, dtype=torch.float64, device="cuda")[:, None]
    observed = actual[window : window + 16, :, 0].double()
    relative_l2 = ((observed - expected).square().mean() / expected.square().mean()).sqrt().item()
    assert relative_l2 < 0.03, f"history under tiny channel gates was lost: relative L2 {relative_l2:.5f}"
    assert alpha**16 < math.exp(-10), "every fixed channel's gate product over a chunk must be negligible"


@pytest.mark.parametrize("heads", [(1, 1, 2), (2, 2, 1)], ids=["fold_qk", "fold_v"])
@pytest.mark.parametrize("total", [64, 2048], ids=["t64", "t2048"])
def test_kda_backward_respects_output_strides(total, heads):
    """The graph-API backward writes into caller-provided gradient buffers of twice the row pitch and leaves the padding
    alone, at 64 and at 2048 tokens on one or two heads."""
    hq, hk, hv = heads
    case = make_case("kda", torch.bfloat16, T=total, H=hq, HK=hk, HV=hv, K=64, V=64)
    values = dict(zip(LEAF_NAMES["kda"], thd_tensors(case)))
    values["cu_seqlens"] = case.cu
    values["dO"] = torch.randn(total, max(hq, hv), 64, dtype=torch.bfloat16, device="cuda")
    graph = cudnn.pygraph()
    tensors = {name: graph.tensor(list(t.shape), data_type=CUDNN_DTYPE[t.dtype], name=name) for name, t in values.items()}
    sources = ("q", "k", "v", "g", "beta")
    grads_t = [
        out.set_output(True).set_data_type(CUDNN_DTYPE[values[src].dtype]) for out, src in zip(graph.kda_bwd(**tensors, batch_invariant=total == 64), sources)
    ]
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    graph.select_plan([graph.get_plan_name_at_index(i) for i in range(len(graph.plans))].index("kda_frost"))
    graph.check_support()
    graph.build_plans()
    workspace = torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")

    def run(outputs):
        pack = {tensors[name]: t for name, t in values.items()}
        pack.update(zip(grads_t, outputs))
        graph.execute(pack, workspace)
        torch.cuda.synchronize()
        assert all(torch.isfinite(t).all() for t in outputs)

    dense = [torch.empty(values[src].shape, dtype=values[src].dtype, device="cuda") for src in sources]
    run(dense)
    padded = [torch.full((*t.shape[:-1], 2 * t.shape[-1]), float("nan"), dtype=t.dtype, device="cuda") for t in dense]
    run([p[..., : t.shape[-1]] for p, t in zip(padded, dense)])
    for name, p, want in zip(sources, padded, dense):
        torch.testing.assert_close(p[..., : want.shape[-1]], want, rtol=0, atol=0)
        assert torch.isnan(p[..., want.shape[-1] :]).all(), f"d{name} wrote into output padding"


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("inputs", ["ordinary", "small_norm", "cancellation"])
def test_kda_bwd_l2norm_reference(dim, dtype, inputs):
    """Compare output and every input gradient to the suite's FP64 recurrence."""
    case = make_case("kda", dtype, T=65, H=2, K=dim, V=dim, seed=418)
    if inputs == "small_norm":
        case.q[:, ::3] *= 1e-6
        case.k[:, 1::3] *= 1e-6
    elif inputs == "cancellation":
        # Correlated keys and opposite values stress the L2 projection's subtraction.
        case.k.copy_(case.k[:, :1].clone().expand_as(case.k))
        case.v[:, 1::2].copy_(-case.v[:, :-1:2])
    assert_bwd_parity(Backend("frost"), case, l2norm=True)


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("checkpoint", [0, 16])
def test_uncut_bwd_raw_ring_wraps_and_changed_input_replay(dim, dtype, checkpoint):
    """Five chunks cover raw-ring phase wraps, a tail and fresh replay operands."""
    generator = torch.Generator().manual_seed(418)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        shape = (65, 1, dim)
        q, k, v = [torch.randn(shape, dtype=dtype, generator=generator).cuda().requires_grad_(True) for _ in range(3)]
        g = (torch.randn(shape, generator=generator) - 4).cuda().requires_grad_(True)
        beta = torch.randn(shape[:2], generator=generator).cuda().requires_grad_(True)
        cu = torch.tensor([0, shape[0]], device="cuda", dtype=torch.int32)
        do = torch.randn(shape, dtype=dtype, generator=generator).cuda()
        leaves = [q, k, v, g, beta]
        options = dict(use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True, safe_gate=True, batch_invariant=True, plan_name="kda_frost")
        if checkpoint == 0:
            state = torch.randn((1, 1, dim, dim), generator=generator).cuda().requires_grad_(True)
            a_log = torch.zeros(1, device="cuda", requires_grad=True)
            dt_bias = torch.zeros((1, dim), device="cuda", requires_grad=True)
            options.update(initial_state=state, a_log=a_log, dt_bias=dt_bias)
            leaves.extend((state, a_log, dt_bias))

        def call():
            """Use the public autograd path, with and without a checkpoint series."""
            output = kimi_delta_attention(q, k, v, g, beta, cu, checkpoint_every_n_tokens=checkpoint, **options)[0]
            return torch.autograd.grad(output, leaves, do)

        call()
        baseline = call()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream):
                actual = call()
            for _ in range(2):
                graph.replay()
                for got, want in zip(actual, baseline, strict=True):
                    torch.testing.assert_close(got, want, rtol=0, atol=0)
            with torch.no_grad():
                v.add_(0.25)
            baseline = call()
            graph.replay()
            for got, want in zip(actual, baseline, strict=True):
                assert torch.isfinite(got).all()
                torch.testing.assert_close(got, want, rtol=0, atol=0)
        finally:
            stream.synchronize()
            graph.reset()


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("optional", [False, True])
@pytest.mark.parametrize("length", [65, 513])
def test_kda_backward_only_changed_input_replay(dim, dtype, optional, length):
    """The raw backward on caller checkpoints, 65 and 513 tokens of two heads (the long one under the default call, the
    short one under ``batch_invariant``): a captured graph replays bitwise, with changed inputs too."""
    generator = torch.Generator().manual_seed(419)
    shape = (length, 2, dim)
    q, k, v, do = [(torch.randn(shape, generator=generator) * 0.1).to(dtype).cuda() for _ in range(4)]
    g = torch.full(shape, -4.0, device="cuda")
    beta = torch.zeros(shape[:2], device="cuda")
    cu = torch.tensor([0, shape[0]], device="cuda", dtype=torch.int32)
    checkpoints = (torch.randn(shape[0] // 16 + 1, 2, dim, dim, generator=generator) * 0.01).to(dtype).cuda()
    options = dict(
        state_checkpoints=checkpoints,
        checkpoint_every_n_tokens=16,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        batch_invariant=length == 65,
        plan_name="kda_frost",
    )
    if optional:
        state = torch.zeros((1, 2, dim, dim), device="cuda")
        options.update(
            initial_state=state, d_final_state=torch.full_like(state, 0.125), a_log=torch.zeros(2, device="cuda"), dt_bias=torch.zeros(2, dim, device="cuda")
        )

    def call():
        """Execute backward with caller checkpoints and the current live input contents."""
        return torch.ops.cudnn.kimi_delta_attention_bwd(do, q, k, v, g, beta, cu, dim**-0.5, **options)

    call()
    expected = tuple(t.clone() for t in call())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            actual = call()
        graph.replay()
        for got, want in zip(actual, expected, strict=True):
            assert torch.isfinite(got).all()
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        do.add_(0.125)
        v.add_(0.25)
        expected = tuple(t.clone() for t in call())
        graph.replay()
        for got, want in zip(actual, expected, strict=True):
            assert torch.isfinite(got).all()
            torch.testing.assert_close(got, want, rtol=0, atol=0)
    finally:
        torch.cuda.synchronize()
        graph.reset()
