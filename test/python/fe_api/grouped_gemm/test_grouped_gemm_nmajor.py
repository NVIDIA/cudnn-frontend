# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read original forward weights directly for BF16 input gradients."""

import pytest
import torch
from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

pytestmark = [
    pytest.mark.L0,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)) or cutedsl_too_old(cutedsl_state()[1]),
        reason="requires Blackwell and CuTe DSL4.7+",
    ),
]


def as_mkl(x):
    m, k = x.shape
    return x.as_strided((m, k, 1), (k, 1, m * k))


@pytest.mark.parametrize("major", ["k", "n"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_discrete_grouped_bf16_layout_and_changed_weight_graph(major, dynamic):
    import cudnn

    torch.manual_seed(4191)
    e, m, n, k = 2, 256, 320, 256
    a = torch.randn(e * m, k, device="cuda", dtype=torch.bfloat16) * 0.125
    # Identical allocation metadata across K/N plans exercises the semantic key.
    storage = torch.randn(e, n * k, device="cuda", dtype=torch.bfloat16) * 0.125
    ptrs = torch.tensor([row.data_ptr() for row in storage], device="cuda", dtype=torch.int64)
    offsets = torch.tensor([m, 2 * m], device="cuda", dtype=torch.int32)
    alpha = torch.tensor([0.75, 1.25], device="cuda")
    prob = torch.rand(e * m, 1, 1, device="cuda")
    prob[::17] = 0
    c = as_mkl(torch.empty(e * m, n, device="cuda"))
    d = torch.empty_like(c)
    plan = cudnn.GroupedGemmSm100(
        as_mkl(a),
        c,
        d,
        offsets,
        alpha,
        sample_prob=prob,
        num_experts=e,
        b_shape=(n, k),
        b_dtype=torch.bfloat16,
        b_major=major,
        generate_c=True,
        use_dynamic_sched=dynamic,
    )
    plan.compile()

    def run():
        plan.execute(as_mkl(a), c, d, offsets, alpha, b_ptrs=ptrs, prob_tensor=prob)

    def check():
        weight = storage.view(e, n, k) if major == "k" else storage.view(e, k, n).transpose(1, 2)
        reference = torch.cat([a[j * m : (j + 1) * m].double() @ weight[j].double().T * alpha[j].double() for j in range(e)])
        torch.testing.assert_close(c[..., 0].double(), reference, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(d[..., 0].double(), reference * prob[:, 0].double(), atol=2e-6, rtol=2e-5)

    run()
    check()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    storage.mul_(0.5)
    prob.mul_(0.375)
    graph.replay()
    check()
    out = cudnn.grouped_gemm_wrapper_sm100(
        a_tensor=as_mkl(a),
        padded_offsets=offsets,
        alpha_tensor=alpha,
        b_ptrs=ptrs,
        n=n,
        b_dtype=torch.bfloat16,
        b_major=major,
        prob_tensor=prob,
        c_dtype=torch.float32,
        d_dtype=torch.float32,
        generate_c=True,
        use_dynamic_sched=dynamic,
    )
    torch.testing.assert_close(out["d_tensor"], d, atol=0, rtol=0)
