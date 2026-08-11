# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host cost of one execute() for the FROST sdpa and gemm engines.

The counterpart of bench_gdn_host.py for the two engines that have NOT migrated
to the normalized variant pack, so the difference between them and GDN is the
cost the migration is expected to move. Every number is a burst from a drained
queue, swept over burst size: one that climbs with n is the device rate, not
host cost.
"""

from __future__ import annotations

import os
import sys
import time

# the FROST manifest rows are opt-in; set before cudnn reads the manifest
os.environ["CUDNN_FRONTEND_ENABLE_FROST_ENGINES"] = "1"

import torch  # noqa: E402

import cudnn  # noqa: E402
from cudnn.engines import is_python_engine  # noqa: E402

BURSTS = (1, 16, 64)
HALF, F32, BF16 = cudnn.data_type.HALF, cudnn.data_type.FLOAT, cudnn.data_type.BFLOAT16


def burst(fn, n, reps=25):
    for _ in range(30):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        for _ in range(n):
            fn()
        out.append((time.perf_counter_ns() - t0) / n / 1000.0)
    torch.cuda.synchronize()
    return min(out)


def _pin_python(g):
    """Select the python engine's plan, or None when none claimed the graph."""
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    python = [i for i, p in enumerate(g.plans) if is_python_engine(p.engine_id)]
    if not python:
        return None
    g.select_plan(python[0])
    g.check_support()
    g.build_plans()
    return g


def sdpa_case(b=2, h=8, s=256, d=256):
    dims, strides = (b, h, s, d), (s * h * d, d, h * d, 1)
    g = cudnn.pygraph(io_data_type=HALF, intermediate_data_type=F32, compute_data_type=F32)
    q = g.tensor(dim=dims, stride=strides, data_type=HALF, name="q")
    k = g.tensor(dim=dims, stride=strides, data_type=HALF, name="k")
    v = g.tensor(dim=dims, stride=strides, data_type=HALF, name="v")
    o, _ = g.sdpa(name="sdpa", q=q, k=k, v=v, attn_scale=1.0 / d**0.5, is_inference=True, use_causal_mask=True)
    o.set_output(True).set_dim(dims).set_stride(strides)
    if _pin_python(g) is None:
        return None
    mk = lambda: torch.randn(b, s, h, d, device="cuda", dtype=torch.float16).transpose(1, 2)  # noqa: E731
    data = {q: mk(), k: mk(), v: mk(), o: torch.empty(b, s, h, d, device="cuda", dtype=torch.float16).transpose(1, 2)}
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    return g, data, ws


def gemm_case(m=256, n=256, k=128):
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    A = g.tensor(name="A", uid=1, dim=[1, m, k], stride=[m * k, k, 1], data_type=BF16)
    B = g.tensor(name="B", uid=2, dim=[1, k, n], stride=[k * n, 1, k], data_type=BF16)
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(BF16).set_uid(3)
    if _pin_python(g) is None:
        return None
    a = torch.randn(1, m, k, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(1, n, k, dtype=torch.bfloat16, device="cuda")
    c = torch.empty(1, m, n, dtype=torch.bfloat16, device="cuda")
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    return g, {1: a, 2: b, 3: c}, ws


def report(label, built):
    if built is None:
        print(f"{label:34s}  no python engine claimed this graph")
        return
    g, data, ws = built
    g.execute(data, ws)
    torch.cuda.synchronize()
    plan = g._compiled_plans[g._plan_index]
    row = [burst(lambda: g.execute(data, ws), n) for n in BURSTS]
    print(f"{label:34s}" + "".join(f"{v:10.2f}" for v in row) + f"   migrated={plan.takes_variant_pack}")
    uid_to_data = g._uid_to_data(data)
    print(f"{'  _uid_to_data':34s}{burst(lambda: g._uid_to_data(data), 64):10.2f}")
    print(f"{'  _normalize (not on this path yet)':34s}{burst(lambda: g._normalize(uid_to_data, ws), 64):10.2f}")


def main():
    print(f"{'':34s}" + "".join(f"{'n=' + str(n):>10s}" for n in BURSTS))
    report("frost sdpa fwd (2,8,256,256)", sdpa_case())
    report("frost gemm (256x256x128)", gemm_case())
    print("\nmin us/call over 25 reps")


if __name__ == "__main__":
    sys.exit(main())
