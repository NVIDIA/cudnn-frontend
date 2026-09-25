# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent FP64 autograd oracle for clamped grouped dGLU gradients."""

import pytest
import torch

from fe_api.grouped_gemm._workspace import ws
import torch.nn.functional as F

from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

pytestmark = [
    pytest.mark.L0,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)) or cutedsl_too_old(cutedsl_state()[1]),
        reason="Grouped BF16 dGLU test requires Blackwell and CuTe DSL4.7+",
    ),
]


def as_mkl(x):
    m, k = x.shape
    return x.as_strided((m, k, 1), (k, 1, m * k))


def close(actual, expected):
    delta = actual.double() - expected.double()
    assert torch.isfinite(actual).all()
    relative = float(delta.norm() / expected.double().norm().clamp_min(1e-20))
    maximum = float(delta.abs().max() / expected.double().abs().max().clamp_min(1e-20))
    assert relative < 1e-4 and maximum < 4e-4, (relative, maximum)


@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("slope,minimum,maximum,offset", [(1.702, -7.0, 7.0, 1.0), (1.0, -10.0, 10.0, 0.0)])
def test_grouped_dgeglu_autograd(vector, slope, minimum, maximum, offset):
    import cudnn

    torch.manual_seed(4102026)
    e, m, n, k = 2, 512, 128, 128
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.125
    b = torch.randn(e, n, k, device="cuda", dtype=torch.bfloat16) * 0.125
    c = torch.randn(m, 2 * n, device="cuda", dtype=torch.bfloat16) * 4
    # Include both sides of, and exactly on, each clamp boundary.
    boundary = torch.tensor([minimum - 1, minimum, minimum + 1, 0.0, maximum - 1, maximum, maximum + 1], device="cuda", dtype=c.dtype)
    c[:, :7] = boundary
    c[:, 32:39] = boundary
    probability = torch.rand(m, 1, 1, device="cuda") * 1.5
    probability[::17] = 0
    offsets = torch.tensor([256, 512], device="cuda", dtype=torch.int32)
    alpha = torch.ones(e, device="cuda")
    dprob = torch.zeros_like(probability)
    cr, pr = c.double().requires_grad_(), probability.double().requires_grad_()
    pair = cr.view(m, n // 32, 2, 32)
    gate = pair[:, :, 0].reshape(m, n).clamp(max=maximum)
    up = pair[:, :, 1].reshape(m, n).clamp(minimum, maximum)
    activation = (gate * torch.sigmoid(slope * gate)) * (up + offset) * pr.view(m, 1)
    dact = torch.cat([F.linear(a[j * 256 : (j + 1) * 256].double(), b[j].double()) for j in range(e)])
    dc, dp = torch.autograd.grad(activation, (cr, pr), dact)
    out = cudnn.grouped_gemm_dglu_wrapper_sm100(
        a_tensor=as_mkl(a),
        c_tensor=as_mkl(c),
        sfa_tensor=None,
        padded_offsets=offsets,
        alpha_tensor=alpha,
        beta_tensor=alpha,
        prob_tensor=probability,
        dprob_tensor=dprob,
        b_tensor=b.permute(1, 2, 0),
        d_dtype=torch.float32,
        act_func="dgeglu",
        vector_f32=vector,
        geglu_alpha=slope,
        glu_clamp_min=minimum,
        glu_clamp_max=maximum,
        linear_offset=offset,
    )
    close(out["d_row_tensor"].squeeze(-1), dc)
    close(out["dprob_tensor"], dp)
    api = cudnn.GroupedGemmDgluSm100(
        as_mkl(a),
        as_mkl(c),
        out["d_row_tensor"],
        None,
        None,
        offsets,
        alpha,
        alpha,
        probability,
        dprob,
        sample_b=b.permute(1, 2, 0),
        act_func="dgeglu",
        vector_f32=vector,
        geglu_alpha=slope,
        glu_clamp_min=minimum,
        glu_clamp_max=maximum,
        linear_offset=offset,
    )
    api.compile()
    workspace = ws(api)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dprob.zero_()
        api.execute(
            as_mkl(a), as_mkl(c), out["d_row_tensor"], None, None, offsets, alpha, alpha, probability, dprob, b_tensor=b.permute(1, 2, 0), workspace=workspace
        )
    probability.mul_(0.375)
    graph.replay()
    torch.cuda.synchronize()
    close(out["d_row_tensor"].squeeze(-1), dc * 0.375)
    close(out["dprob_tensor"], dp)
