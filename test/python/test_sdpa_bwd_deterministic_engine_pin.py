# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The deterministic SDPA backward on Blackwell selects the part's own bprop engine.

``CompositeSDPABackwardNode::override_heuristics_query()`` pins a backend engine for
a deterministic backward on Blackwell -- engine 5 on cc 10.0 / 10.3, engine 17 past
cc 10.3 -- because the heuristics answer that graph with the SM80 bprop engine (7)
plus a separate dQ matmul over a quadratic dS workspace: at s=8192 on cc 10.7 that
is 4.9x slower on d128 dense GQA and 26x on d64 sliding-window attention. Not every
backend serves the pinned engine on every part (cuDNN 9.24.0.43 declines engine 17
on cc 10.7 at the engine's finalize), so ``Graph::create_execution_plans`` falls
back to the heuristics query when the pinned engine cannot be created.

Two contracts, on the two shapes of the regression (d64 sliding-window causal MHA,
d128 dense GQA) at an L0-sized sequence length:

1. the deterministic backward builds and runs on every Blackwell part, whether or
   not the backend serves the pinned engine (the fallback), and
2. when the backend serves the pinned engine -- probed by appending that engine
   explicitly to a second graph -- the default selection IS that engine (the pin).

Before the fix, (2) failed on cc 10.7 with every backend (#2086 declined the pin
for the whole part) and, before #2086, (1) failed on cc 10.7 with a backend that
does not serve engine 17 (a generic backend-API error instead of a plan).
"""

import math

import pytest
import torch

import cudnn
from cudnn._pygraph import pygraph
from cudnn.engines import is_python_engine

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

_CC = torch.cuda.get_device_capability()

pytestmark = [
    pytest.mark.L0,
    pytest.mark.skipif(_CC[0] != 10, reason="the deterministic SDPA backward pin applies to Blackwell parts (cc 10.x)"),
    pytest.mark.skipif(cudnn.backend_version() < 91800, reason="the deterministic SDPA backward on Blackwell needs cuDNN >= 9.18"),
]

DT = cudnn.data_type.BFLOAT16
F32 = cudnn.data_type.FLOAT
B, S = 1, 512

# The two shapes of the regression at a small sequence length: d64 sliding-window causal MHA (the window
# shorter than S, so it masks) and d128 dense GQA.
CASES = {
    "d64_swa128_causal_mha": dict(h_q=4, h_kv=4, d=64, causal=True, window=128),
    "d128_dense_gqa": dict(h_q=8, h_kv=2, d=128, causal=False, window=None),
}


def pinned_engine(cc):
    """The engine override_heuristics_query() pins for a deterministic backward on this Blackwell part."""
    return 17 if cc > (10, 3) else 5


def _tensor(g, name, shape, dt):
    b, h, s, d = shape
    return g.tensor(name=name, dim=[b, h, s, d], stride=[h * s * d, s * d, d, 1], data_type=dt)


def _mask_kwargs(case):
    kwargs = dict(use_causal_mask=case["causal"], diagonal_alignment=cudnn.diagonal_alignment.TOP_LEFT)
    if case["window"] is not None:
        kwargs["diagonal_band_left_bound"] = case["window"]
    return kwargs


def _forward(case, handle, q, k, v):
    """o and stats from the backend's own forward, so the backward's inputs carry its mask semantics."""
    g = pygraph(io_data_type=DT, intermediate_data_type=F32, compute_data_type=F32, handle=handle)
    gq, gk, gv = _tensor(g, "q", q.shape, DT), _tensor(g, "k", k.shape, DT), _tensor(g, "v", v.shape, DT)
    o, stats = g.sdpa(gq, gk, gv, generate_stats=True, attn_scale=1.0 / math.sqrt(case["d"]), **_mask_kwargs(case))
    o.set_output(True).set_data_type(DT)
    stats.set_output(True).set_data_type(F32)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    g.check_support()
    g.build_plans()
    o_t = torch.empty_like(q)
    stats_t = torch.empty(q.shape[0], q.shape[1], q.shape[2], 1, dtype=torch.float32, device="cuda")
    ws = torch.empty(max(int(g.get_workspace_size()), 1), dtype=torch.uint8, device="cuda")
    g.execute({gq: q, gk: k, gv: v, o: o_t, stats: stats_t}, ws, handle=handle)
    torch.cuda.synchronize()
    return o_t, stats_t


class BackwardGraph:
    """One deterministic backward graph and its tensors, built up to the operation graph."""

    def __init__(self, case, handle):
        hq, hkv, d = case["h_q"], case["h_kv"], case["d"]
        self.g = g = pygraph(io_data_type=DT, intermediate_data_type=F32, compute_data_type=F32, handle=handle)
        self.q = _tensor(g, "q", (B, hq, S, d), DT)
        self.k = _tensor(g, "k", (B, hkv, S, d), DT)
        self.v = _tensor(g, "v", (B, hkv, S, d), DT)
        self.o = _tensor(g, "o", (B, hq, S, d), DT)
        self.dO = _tensor(g, "dO", (B, hq, S, d), DT)
        self.stats = _tensor(g, "stats", (B, hq, S, 1), F32)
        self.dQ, self.dK, self.dV = g.sdpa_backward(
            name="sdpa_backward",
            q=self.q,
            k=self.k,
            v=self.v,
            o=self.o,
            dO=self.dO,
            stats=self.stats,
            attn_scale=1.0 / math.sqrt(d),
            use_deterministic_algorithm=True,
            **_mask_kwargs(case),
        )
        for t in (self.dQ, self.dK, self.dV):
            t.set_output(True).set_data_type(DT)
        g.validate()
        g.build_operation_graph()


@pytest.mark.parametrize("case", list(CASES))
def test_deterministic_bwd_selects_the_part_bprop_engine(case, cudnn_handle):
    spec = CASES[case]
    bwd = BackwardGraph(spec, cudnn_handle)
    g = bwd.g
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    backend = [index for index, plan in enumerate(g.plans) if not is_python_engine(plan.engine_id)]
    assert backend, "the backend offers no plan for the deterministic backward"
    g.select_plan(backend[0])
    try:
        g.check_support()
    except cudnn.cudnnGraphNotSupportedError as error:
        pytest.skip(f"the backend declines this deterministic backward: {error}")
    g.build_plans()
    chosen = g.plans[backend[0]].engine_id
    name = g.get_plan_name_at_index(g._plan_index)
    print(f"@@@@ {case}: cc {_CC}, cuDNN {cudnn.backend_version()}: plan {name} (engine {chosen})")

    # Contract 1: the selected plan runs and produces gradients.
    hq, hkv, d = spec["h_q"], spec["h_kv"], spec["d"]
    gen = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(B, hq, S, d, dtype=torch.bfloat16, device="cuda", generator=gen)
    k = torch.randn(B, hkv, S, d, dtype=torch.bfloat16, device="cuda", generator=gen)
    v = torch.randn(B, hkv, S, d, dtype=torch.bfloat16, device="cuda", generator=gen)
    dO = torch.randn(B, hq, S, d, dtype=torch.bfloat16, device="cuda", generator=gen)
    o, stats = _forward(spec, cudnn_handle, q, k, v)
    dq, dk, dv = torch.full_like(q, float("nan")), torch.full_like(k, float("nan")), torch.full_like(v, float("nan"))
    ws = torch.empty(max(int(g.get_workspace_size()), 1), dtype=torch.uint8, device="cuda")
    g.execute({bwd.q: q, bwd.k: k, bwd.v: v, bwd.o: o, bwd.dO: dO, bwd.stats: stats, bwd.dQ: dq, bwd.dK: dk, bwd.dV: dv}, ws, handle=cudnn_handle)
    torch.cuda.synchronize()
    for label, t in (("dQ", dq), ("dK", dk), ("dV", dv)):
        assert torch.isfinite(t).all(), f"{label} has non-finite values under plan {name}"
        assert t.abs().sum() > 0, f"{label} is all zero under plan {name}"

    # Contract 2: the pin, whenever the backend serves the pinned engine on this part.
    pinned = pinned_engine(_CC)
    if chosen == pinned:
        return
    probe = BackwardGraph(spec, cudnn_handle).g
    try:
        probe.create_execution_plan(pinned, {})
        served, detail = True, probe.get_plan_name_at_index(probe.get_execution_plan_count() - 1)
    except Exception as error:  # the backend declines the engine at its finalize; the message names the status
        served, detail = False, str(error).splitlines()[0][:160]
    assert not served, (
        f"cuDNN {cudnn.backend_version()} serves engine {pinned} on cc {_CC} ({detail}) but the deterministic backward "
        f"selected engine {chosen} ({name}): the override_heuristics_query pin is missing"
    )
    print(f"@@@@ {case}: cuDNN {cudnn.backend_version()} does not serve engine {pinned} on cc {_CC} ({detail}); the fallback built {name}")
