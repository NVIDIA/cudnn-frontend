# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routing and CUDA-graph behaviour of the ``kda_hopper_cuda`` sm90 engine.

The numerics of this engine are covered by the shared ``backend`` fixture in
``test_la.py`` (``hopper_cuda``). What is covered here is everything that fixture
cannot reach:

* that the KDA heuristics hook actually makes it the default on Hopper, since
  that is a default-behaviour change for every sm90 KDA user;
* that ranking it first does NOT change who is ELIGIBLE. ``_build_plan_at()``
  builds a ranked entry without re-affirming ``check_support``, so a graph this
  128-only engine rejects must never reach it. When the hook first landed it
  did, and a head-dim-64 graph launched the kernel on 64-dim tensors and died
  with an illegal memory access. That is what ``test_declined_graph_falls_through``
  pins;
* that the path captures into a CUDA graph. Per-call host dispatch is ~130 us
  against a ~60 us kernel, so capture is how the overhead is actually removed
  (measured 143 us -> 2.8 us of host time), and capture breaks silently the
  moment anything on the path allocates or synchronises per call.
"""

import pytest
import torch

import cudnn  # noqa: F401 -- import-order requirement, see test/python/conftest.py
from cudnn.linear_attention import kimi_delta_attention

pytestmark = [
    pytest.mark.L0,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]

ENGINE = "kda_hopper_cuda"
HOPPER = (9, 0)


def _hopper() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == HOPPER


requires_hopper = pytest.mark.skipif(not _hopper(), reason="the sm90 CUDA KDA engine needs Hopper")


def make_case(T=2048, H=12, N=1, dim=128, lower_bound=-5.0, seed=7):
    """A production-gate KDA case: L2-normalised q/k, gate on (-5, 0), non-zero
    initial state. The gate range matters -- a kernel tuned at -1 can be 100%
    NaN at -5."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(T, H, dim, generator=g, device="cuda", dtype=torch.float32)
    k = torch.randn(T, H, dim, generator=g, device="cuda", dtype=torch.float32)
    q = (q / q.norm(dim=-1, keepdim=True)).to(torch.bfloat16)
    k = (k / k.norm(dim=-1, keepdim=True)).to(torch.bfloat16)
    v = torch.randn(T, H, dim, generator=g, device="cuda", dtype=torch.bfloat16) * 0.5
    raw = torch.randn(T, H, dim, generator=g, device="cuda", dtype=torch.float32)
    gate = (lower_bound * torch.sigmoid(raw)).contiguous()
    beta = torch.sigmoid(torch.randn(T, H, generator=g, device="cuda", dtype=torch.float32)).contiguous()
    state = (torch.randn(N, H, dim, dim, generator=g, device="cuda", dtype=torch.float32) * 0.05).contiguous()
    span = T // N
    bounds = [0]
    for i in range(N):
        bounds.append(bounds[-1] + (T - span * (N - 1) if i == N - 1 else span))
    cu = torch.tensor(bounds, device="cuda", dtype=torch.int32)
    return q, k, v, gate, beta, cu, state


def launched_kernels(fn):
    """Device kernels one call launches, busiest first."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    events = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    return [e.key for e in sorted(events, key=lambda e: -e.self_device_time_total)]


@requires_hopper
def test_is_the_default_on_hopper():
    """An unpinned call picks the fused CUDA engine, not slot-1 cuTile."""
    q, k, v, g, beta, cu, s0 = make_case()
    kernels = launched_kernels(lambda: kimi_delta_attention(q, k, v, g, beta, cu, initial_state=s0, output_final_state=True))
    assert any("kda_fused" in name for name in kernels), f"expected kda_fused, got {kernels[:3]}"


@requires_hopper
def test_plan_name_still_overrides():
    """Ranking is a default, not a lock: naming another engine still runs it."""
    q, k, v, g, beta, cu, s0 = make_case()
    kernels = launched_kernels(lambda: kimi_delta_attention(q, k, v, g, beta, cu, initial_state=s0, output_final_state=True, plan_name="kda_cutile"))
    assert not any("kda_fused" in name for name in kernels), f"plan_name ignored, got {kernels[:3]}"


@requires_hopper
def test_declined_graph_falls_through():
    """A head dim the sm90 kernels do not serve must not reach them.

    RED-then-green history: with ranking but no eligibility filter this launched
    the 128-only kernel on 64-dim tensors and raised an illegal memory access.
    """
    q, k, v, g, beta, cu, s0 = make_case(T=512, H=4, dim=64)
    out, final_state = kimi_delta_attention(q, k, v, g, beta, cu, initial_state=s0, output_final_state=True)[:2]
    torch.cuda.synchronize()
    assert torch.isfinite(out).all() and torch.isfinite(final_state).all()
    kernels = launched_kernels(lambda: kimi_delta_attention(q, k, v, g, beta, cu, initial_state=s0, output_final_state=True))
    assert not any("kda_fused" in name for name in kernels), "the 128-only kernel served a 64-dim graph"


@requires_hopper
@pytest.mark.parametrize("shape", [(2048, 12, 1), (4096, 12, 1)], ids=["2048x12", "4096x12"])
def test_cuda_graph_capture_replays(shape):
    """The engine captures, and replay reproduces eager numerics.

    Capture is the answer to this path's host-dispatch cost, and it only works
    while steady state allocates and synchronises nothing -- so this fails the
    moment a per-call allocation creeps back in.
    """
    T, H, N = shape
    q, k, v, g, beta, cu, s0 = make_case(T=T, H=H, N=N)
    call = lambda: kimi_delta_attention(q, k, v, g, beta, cu, initial_state=s0, output_final_state=True, plan_name=ENGINE)  # noqa: E731

    # Warm everything that is lazy: NVRTC compile, graph build, plan buffers.
    for _ in range(5):
        eager_o, eager_fs = call()[:2]
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    with torch.cuda.graph(graph):
        captured_o, captured_fs = call()[:2]

    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(captured_o.float(), eager_o.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(captured_fs, eager_fs, atol=2e-2, rtol=2e-2)
