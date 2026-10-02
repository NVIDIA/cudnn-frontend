# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Compile an SDPA forward plan ahead of time and run it without a JIT.

    python samples/aot/export_plan.py OUT_DIR

Builds a causal bf16 attention graph (BSHD storage), lets the FROST engine plan
it, and writes:

    OUT_DIR/plan.bin     graph.serialize(): the kernel and its launch sequence
    OUT_DIR/<uid>.bin    each tensor's bytes (inputs random, outputs zero)
    OUT_DIR/expected/    the outputs the JIT'd plan produced, for comparison

Then it loads plan.bin into a fresh graph and checks the output matches the JIT'd
plan bit for bit. ``run_plan.cpp`` in this directory does the same load from C++.
Needs an SM100-class GPU (the prepared FROST SDPA launch is SM100/SM107 f16/bf16).
"""

import math
import os
import sys

import torch

import cudnn
from cudnn.engines.engine_ids import is_python_engine


def build(b=2, h=8, h_kv=2, s_q=256, s_kv=512, d=128):
    graph = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)

    def bshd(heads, s):  # (B, H, S, D) dims over (B, S, H, D) storage
        return [s * heads * d, d, heads * d, 1]

    q = graph.tensor(dim=[b, h, s_q, d], stride=bshd(h, s_q), data_type=cudnn.data_type.BFLOAT16, name="q")
    k = graph.tensor(dim=[b, h_kv, s_kv, d], stride=bshd(h_kv, s_kv), data_type=cudnn.data_type.BFLOAT16, name="k")
    v = graph.tensor(dim=[b, h_kv, s_kv, d], stride=bshd(h_kv, s_kv), data_type=cudnn.data_type.BFLOAT16, name="v")
    o, stats = graph.sdpa(name="sdpa", q=q, k=k, v=v, generate_stats=True, attn_scale=1.0 / math.sqrt(d), use_causal_mask=True)
    o.set_output(True).set_dim([b, h, s_q, d]).set_stride(bshd(h, s_q))
    stats.set_output(True).set_dim([b, h, s_q, 1]).set_stride([h * s_q, s_q, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    # Heuristics may rank a cuDNN backend plan first; this sample is about the Python (FROST) plan.
    graph.select_plan(next(i for i, plan in enumerate(graph.plans) if is_python_engine(plan.engine_id)))
    graph.check_support()
    graph.build_plans()

    torch.manual_seed(0)
    buffers = {
        q: torch.randn(b, s_q, h, d, device="cuda", dtype=torch.bfloat16).transpose(1, 2),
        k: torch.randn(b, s_kv, h_kv, d, device="cuda", dtype=torch.bfloat16).transpose(1, 2),
        v: torch.randn(b, s_kv, h_kv, d, device="cuda", dtype=torch.bfloat16).transpose(1, 2),
        o: torch.zeros(b, s_q, h, d, device="cuda", dtype=torch.bfloat16).transpose(1, 2),
        stats: torch.zeros(b, h, s_q, 1, device="cuda", dtype=torch.float32),
    }
    return graph, buffers, (o, stats)


def storage(t):
    """The bytes a tensor's view covers, from its first element."""
    span = 1 + sum((n - 1) * s for n, s in zip(t.shape, t.stride()))
    return torch.as_strided(t, (span,), (1,))


def main(out_dir):
    os.makedirs(os.path.join(out_dir, "expected"), exist_ok=True)
    graph, buffers, outputs = build()
    print("selected plan:", graph.selected_engine.name)
    workspace = torch.empty(max(graph.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)

    for t, buf in buffers.items():  # inputs as they are, outputs zero
        storage(buf).contiguous().view(torch.uint8).cpu().numpy().tofile(os.path.join(out_dir, f"{t.get_uid()}.bin"))

    graph.execute(buffers, workspace)
    torch.cuda.synchronize()
    expected = {t: buffers[t].clone() for t in outputs}
    for t in outputs:
        storage(expected[t]).contiguous().view(torch.uint8).cpu().numpy().tofile(os.path.join(out_dir, "expected", f"{t.get_uid()}.bin"))

    blob = bytes(graph.serialize())
    with open(os.path.join(out_dir, "plan.bin"), "wb") as f:
        f.write(blob)
    print(f"wrote {len(blob)} bytes to {os.path.join(out_dir, 'plan.bin')}")

    # The same blob, loaded back: no Python engine and no JIT are involved from here on.
    loaded = cudnn.pygraph()
    loaded.deserialize(blob)
    for t in outputs:
        buffers[t].zero_()
    loaded.execute({t.get_uid(): buf for t, buf in buffers.items()}, workspace)
    torch.cuda.synchronize()
    same = all(torch.equal(buffers[t], expected[t]) for t in outputs)
    print("deserialized plan matches the JIT'd plan bit for bit:", same)
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "aot_plan"))
