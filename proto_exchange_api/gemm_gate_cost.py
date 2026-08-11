# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How much of frost_gemm's execute is the per-call gate?

frost_gemm_execute_design.md proposes replacing five intermediate lists and four
walks with one pass over a table built when the plan compiles. That is only
worth doing if the gate is most of what the compiler layer costs, so this sizes
it before any code moves:

    graph.execute        - the whole thing
      plan.execute       - minus graph.execute's entry and _normalize
        run_resolved     - the gate plus the launch
          _call_positional - the launch alone

gate = run_resolved - _call_positional.
"""

import os
import sys
import time

os.environ["CUDNN_FRONTEND_ENABLE_FROST_ENGINES"] = "1"

import torch  # noqa: E402

import cudnn  # noqa: E402
from cudnn.engines import is_python_engine  # noqa: E402

sys.path.insert(0, "/home/scratch.yanxu_gpu/fe_pr1")

BF16, F32 = cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT
M = N = 256
K = 128


def burst(fn, n, reps=25):
    for _ in range(40):
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


def build():
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    a = g.tensor(name="A", uid=1, dim=[1, M, K], stride=[M * K, K, 1], data_type=BF16)
    b = g.tensor(name="B", uid=2, dim=[1, K, N], stride=[K * N, 1, K], data_type=BF16)
    c = g.matmul(A=a, B=b, name="mm")
    c.set_output(True).set_data_type(BF16).set_uid(3)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    index = next(i for i, p in enumerate(g.plans) if is_python_engine(p.engine_id))
    g.select_plan(index)
    g.check_support()
    g.build_plans()
    data = {
        1: torch.randn(1, M, K, dtype=torch.bfloat16, device="cuda"),
        2: torch.randn(1, N, K, dtype=torch.bfloat16, device="cuda"),
        3: torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda"),
    }
    ws = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    return g, data, ws


def main():
    g, data, ws = build()
    g.execute(data, ws)
    torch.cuda.synchronize()

    from cudnn.engines.base import ExecutionContext, bind_ports  # noqa: F401

    plan = g._compiled_plans[g._plan_index]
    compiled = plan._compiled
    handle = g._handle
    ctx = ExecutionContext(handle=handle, stream=g._resolve_stream(handle), workspace=ws)
    uid_to_data = g._uid_to_data(data)
    pack = g._normalize(uid_to_data, ws)

    # capture what run_resolved hands _call_positional, to time the launch alone
    captured = {}
    original = type(compiled)._call_positional

    def spy(self, *args, **kwargs):
        captured["args"] = (args, kwargs)
        return original(self, *args, **kwargs)

    type(compiled)._call_positional = spy
    g.execute(data, ws)
    torch.cuda.synchronize()
    type(compiled)._call_positional = original
    args, kwargs = captured["args"]

    slots = [pack.slot(t.get_uid()) for t in plan._tensors]
    resolved = {id(t): v for t, v in zip(plan._tensors, pack.views(slots))}

    rows = [
        ("graph.execute(data, ws)", lambda: g.execute(data, ws)),
        ("  _uid_to_data", lambda: g._uid_to_data(data)),
        ("  _normalize", lambda: g._normalize(uid_to_data, ws)),
        ("  plan.execute(pack)", lambda: plan.execute(g, pack, ctx)),
        ("    pack.views(slots)", lambda: pack.views(slots)),
        ("    run_resolved (gate + launch)", lambda: compiled.run_resolved(resolved, stream=ctx.stream)),
        ("      _call_positional (launch)", lambda: original(compiled, *args, **kwargs)),
    ]

    print(f"{'':38s}{'n=1':>9s}{'n=16':>9s}{'n=64':>9s}")
    values = {}
    for label, fn in rows:
        row = [burst(fn, n) for n in (1, 16, 64)]
        values[label.strip()] = min(row)
        print(f"{label:38s}" + "".join(f"{v:9.2f}" for v in row))

    gate = values["run_resolved (gate + launch)"] - values["_call_positional (launch)"]
    total = values["graph.execute(data, ws)"]
    print()
    print(f"the gate alone            {gate:7.2f} us   ({100 * gate / total:.0f}% of execute)")
    print(f"everything above it       {total - values['run_resolved (gate + launch)']:7.2f} us")


if __name__ == "__main__":
    main()
