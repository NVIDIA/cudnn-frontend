# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
sdpa(stats_use_log2=True) must return Stats in base 2, i.e. log2(e) times the default
natural-log Stats, while leaving O untouched.

LSE consumers that merge partial results across kernels (cascade / split-KV reductions,
speculative decoding) assume one log base for every producer. A backend that returns a
different base produces correct O and wrong Stats, so an output-only comparison never
notices; this test compares Stats directly against both the torch reference and the
default-base graph.
"""

import math

import cudnn
import pytest
import torch

from sdpa.fp16_ref import compute_ref

LOG2_E = math.log2(math.e)


def _cudnn_dtype(torch_dtype):
    return {torch.float16: cudnn.data_type.HALF, torch.bfloat16: cudnn.data_type.BFLOAT16}[torch_dtype]


def _run_sdpa(cudnn_handle, q, k, v, attn_scale, right_bound, implementation, stats_use_log2):
    b, h_q, s_q, d_qk = q.shape
    _, h_kv, s_kv, d_v = v.shape
    dtype = _cudnn_dtype(q.dtype)

    graph = cudnn.pygraph(
        io_data_type=dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )
    q_t = graph.tensor_like(q)
    k_t = graph.tensor_like(k)
    v_t = graph.tensor_like(v)

    o_t, stats_t = graph.sdpa(
        name="sdpa_stats_log2",
        q=q_t,
        k=k_t,
        v=v_t,
        generate_stats=True,
        attn_scale=attn_scale,
        diagonal_band_right_bound=right_bound,
        implementation=implementation,
        stats_use_log2=stats_use_log2,
    )
    o = torch.empty(b, h_q, s_q, d_v, dtype=q.dtype, device="cuda")
    stats = torch.empty(b, h_q, s_q, 1, dtype=torch.float32, device="cuda")
    o_t.set_output(True).set_dim(o.shape).set_stride(o.stride())
    stats_t.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim(stats.shape).set_stride(stats.stride())

    try:
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans()
    except cudnn.cudnnGraphNotSupportedError as e:
        pytest.skip(f"graph not supported ({implementation}): {e}")

    workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda")
    graph.execute({q_t: q, k_t: k, v_t: v, o_t: o, stats_t: stats}, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()
    return o, stats


@pytest.mark.L0
@pytest.mark.parametrize(
    "implementation", [cudnn.attention_implementation.AUTO, cudnn.attention_implementation.COMPOSITE, cudnn.attention_implementation.UNIFIED]
)
@pytest.mark.parametrize("right_bound", [None, 0], ids=["no_mask", "causal"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sdpa_stats_use_log2(cudnn_handle, dtype, right_bound, implementation):
    b, h_q, h_kv, s_q, s_kv, d = 2, 4, 2, 192, 256, 64
    attn_scale = 1.0 / math.sqrt(d)
    rng = torch.Generator(device="cuda").manual_seed(0x10E)
    q = torch.randn(b, h_q, s_q, d, dtype=dtype, device="cuda", generator=rng)
    k = torch.randn(b, h_kv, s_kv, d, dtype=dtype, device="cuda", generator=rng)
    v = torch.randn(b, h_kv, s_kv, d, dtype=dtype, device="cuda", generator=rng)

    o_nat, stats_nat = _run_sdpa(cudnn_handle, q, k, v, attn_scale, right_bound, implementation, stats_use_log2=False)
    o_log2, stats_log2 = _run_sdpa(cudnn_handle, q, k, v, attn_scale, right_bound, implementation, stats_use_log2=True)

    _, stats_ref, _, _ = compute_ref(q.float(), k.float(), v.float(), attn_scale=attn_scale, right_bound=right_bound, torch_type=dtype)

    # The flag only touches Stats. The two graphs may still land on different plans,
    # so O is compared at dtype tolerance rather than bitwise.
    torch.testing.assert_close(o_log2, o_nat, rtol=1e-2, atol=1e-2)
    # Base-2 stats are the natural-log stats scaled by log2(e) (a 1.44x factor), so a
    # 1e-3 bound separates a base mismatch from plan-to-plan fp32 reduction noise.
    torch.testing.assert_close(stats_log2, stats_nat * LOG2_E, rtol=1e-3, atol=1e-3)
    # Both agree with the reference softmax statistics in their own base.
    torch.testing.assert_close(stats_nat, stats_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(stats_log2, stats_ref * LOG2_E, rtol=2e-2, atol=2e-2)
    assert not torch.allclose(stats_log2, stats_nat), "stats_use_log2 had no effect"
