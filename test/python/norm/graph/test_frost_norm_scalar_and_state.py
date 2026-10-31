# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph scalars and BatchNorm running state, through a real cuDNN graph.

``epsilon`` and ``momentum`` are input TENSORS on a cuDNN graph, and the previous
and next running statistics are SEPARATE ports bound to separate buffers. Reading
any of them from the wrong place is silent -- the graph still runs, the output is
still finite, and the numbers are just wrong:

* epsilon read from node params pinned every graph to the 1e-5 default. A graph
  asking for 1e-2 was normalized with 1e-5 and nothing complained.
* momentum did the same against the 0.1 default, so the running statistics
  tracked at whatever rate the default happened to be.
* binding only ``next_running_mean`` made the in-place update read the OUTPUT
  buffer as the previous state. Initialising that output to 0 versus 100 moved
  the next running mean by exactly ``(1 - momentum) * 100`` while the
  previous-state input was ignored entirely.

The last one is why this file binds *distinct* in/out buffers and fills the
outputs with a sentinel: with the input and output aliased to one tensor -- the
common case, and what the existing BatchNorm tests do -- the bug is invisible,
because reading the output happens to give the right answer.

Run under pytest (needs the built extension, like every graph test here):

    pytest test/python/norm/graph/test_frost_norm_scalar_and_state.py -m L0
"""

import pytest
import torch

import cudnn

pytest.importorskip("cudnn.norm", reason="frost norm engines require a built cudnn frontend")

_HEUR = [cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]
_PY_ENGINE_FLOOR = 20_000


def _select_frost(graph):
    """Rank, then pick the frost engine; skip if it did not propose a plan."""
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans(_HEUR)
    idx = next((i for i, p in enumerate(graph.plans) if p.engine_id >= _PY_ENGINE_FLOOR), None)
    if idx is None:
        pytest.skip("frost norm engine proposed no plan for this graph")
    graph.select_plan(idx)
    graph.check_support()
    graph.build_plans()
    return graph


def _bn_once(handle, x_gpu, scale, bias, prev_mean, prev_var, *, eps, momentum, out_fill):
    """One BatchNorm graph execution with in/out running stats in DISTINCT buffers.

    ``out_fill`` seeds the next-state outputs; a correct engine must ignore it.
    """
    n, c = x_gpu.shape[0], x_gpu.shape[1]
    eps_cpu = torch.full((1, 1, 1, 1), eps, dtype=torch.float32)
    mom_cpu = torch.full((1, 1, 1, 1), momentum, dtype=torch.float32)

    g = cudnn.pygraph(
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=handle,
    )
    X = g.tensor_like(x_gpu)
    SC = g.tensor_like(scale)
    BI = g.tensor_like(bias)
    IRM = g.tensor_like(prev_mean)
    IRV = g.tensor_like(prev_var)
    EP = g.tensor_like(eps_cpu)
    MO = g.tensor_like(mom_cpu)
    Y, SM, SIV, ORM, ORV = g.batchnorm(name="BN", input=X, scale=SC, bias=BI, in_running_mean=IRM, in_running_var=IRV, epsilon=EP, momentum=MO)
    Y.set_output(True).set_data_type(x_gpu.dtype)
    for t in (SM, SIV, ORM, ORV):
        t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    _select_frost(g)

    y = torch.empty_like(x_gpu)
    saved_mean = torch.empty(1, c, 1, 1, device="cuda", dtype=torch.float32)
    saved_ivar = torch.empty_like(saved_mean)
    # distinct from the inputs, and deliberately pre-filled
    next_mean = torch.full_like(saved_mean, out_fill)
    next_var = torch.full_like(saved_mean, out_fill)

    ws = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)
    g.execute(
        {
            X: x_gpu,
            SC: scale,
            BI: bias,
            IRM: prev_mean,
            IRV: prev_var,
            EP: eps_cpu,
            MO: mom_cpu,
            Y: y,
            SM: saved_mean,
            SIV: saved_ivar,
            ORM: next_mean,
            ORV: next_var,
        },
        ws,
        handle=handle,
    )
    torch.cuda.synchronize()
    return y, saved_mean, saved_ivar, next_mean, next_var


@pytest.fixture(scope="module")
def _handle():
    h = cudnn.create_handle()
    cudnn.set_stream(handle=h, stream=torch.cuda.current_stream().cuda_stream)
    return h


def _inputs(n=4, c=8, hw=4):
    torch.manual_seed(0)
    x = torch.randn(n, c, hw, hw, device="cuda", dtype=torch.float32)
    scale = torch.ones(1, c, 1, 1, device="cuda", dtype=torch.float32)
    bias = torch.zeros(1, c, 1, 1, device="cuda", dtype=torch.float32)
    prev_mean = torch.full((1, c, 1, 1), 5.0, device="cuda", dtype=torch.float32)
    prev_var = torch.full((1, c, 1, 1), 7.0, device="cuda", dtype=torch.float32)
    return x, scale, bias, prev_mean, prev_var


@pytest.mark.L0
def test_previous_running_stats_come_from_the_input_port(_handle):
    """Seeding the OUTPUT buffer must not change the result."""
    x, scale, bias, pm, pv = _inputs()
    _, _, _, m0, v0 = _bn_once(_handle, x, scale, bias, pm, pv, eps=1e-5, momentum=0.1, out_fill=0.0)
    _, _, _, m1, v1 = _bn_once(_handle, x, scale, bias, pm, pv, eps=1e-5, momentum=0.1, out_fill=100.0)
    assert torch.equal(m0, m1), f"next running mean depends on the output buffer's prior contents: {m0.flatten()[:4]} vs {m1.flatten()[:4]}"
    assert torch.equal(v0, v1), f"next running var depends on the output buffer's prior contents: {v0.flatten()[:4]} vs {v1.flatten()[:4]}"

    # and it must be the documented update against the INPUT state
    xf = x.double().permute(1, 0, 2, 3).reshape(x.shape[1], -1)
    want_m = 0.9 * 5.0 + 0.1 * xf.mean(-1)
    want_v = 0.9 * 7.0 + 0.1 * xf.var(-1, correction=1)
    torch.testing.assert_close(m0.flatten(), want_m, check_dtype=False)
    torch.testing.assert_close(v0.flatten(), want_v, check_dtype=False)

    # the previous-state inputs are the caller's and must come back untouched
    assert torch.equal(pm, torch.full_like(pm, 5.0)), "in_running_mean was overwritten"
    assert torch.equal(pv, torch.full_like(pv, 7.0)), "in_running_var was overwritten"


@pytest.mark.L0
def test_momentum_tensor_is_honored(_handle):
    """A graph asking for momentum 0.5 must not be served the 0.1 default."""
    x, scale, bias, pm, pv = _inputs()
    _, _, _, m_lo, _ = _bn_once(_handle, x, scale, bias, pm, pv, eps=1e-5, momentum=0.1, out_fill=0.0)
    _, _, _, m_hi, _ = _bn_once(_handle, x, scale, bias, pm, pv, eps=1e-5, momentum=0.5, out_fill=0.0)
    assert not torch.allclose(m_lo, m_hi), "momentum 0.1 and 0.5 produced the same running mean -- the tensor is being ignored"
    xf = x.double().permute(1, 0, 2, 3).reshape(x.shape[1], -1)
    want = 0.5 * 5.0 + 0.5 * xf.mean(-1)
    torch.testing.assert_close(m_hi.flatten(), want, check_dtype=False)


@pytest.mark.L0
def test_epsilon_tensor_is_honored(_handle):
    """Same, for epsilon: it reaches rsqrt(var + eps), so it moves saved_inv_var."""
    x, scale, bias, pm, pv = _inputs()
    _, _, iv_small, _, _ = _bn_once(_handle, x, scale, bias, pm, pv, eps=1e-5, momentum=0.1, out_fill=0.0)
    _, _, iv_big, _, _ = _bn_once(_handle, x, scale, bias, pm, pv, eps=1e-1, momentum=0.1, out_fill=0.0)
    assert not torch.allclose(iv_small, iv_big), "epsilon 1e-5 and 1e-1 produced the same inv_variance -- the tensor is being ignored"
    xf = x.double().permute(1, 0, 2, 3).reshape(x.shape[1], -1)
    want = torch.rsqrt(xf.var(-1, correction=0) + 1e-1)
    torch.testing.assert_close(iv_big.flatten(), want, check_dtype=False)
