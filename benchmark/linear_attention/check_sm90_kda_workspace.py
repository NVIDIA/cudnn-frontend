# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the shipped launch path without autograd-worker context probes."""

import torch
from benchmark_sm90_kda_precision import inputs, NAMES
from cudnn.linear_attention.hopper import cuda_bwd_host as host


def main():
    torch.set_num_threads(1)
    for lengths, heads in (([1024], 8), ([1025], 8), ([1024], 2), ([1025], 2), ([0, 1025, 257, 65], 8)):
        x = inputs(lengths, heads)
        outs = {n: torch.empty_like(x[n]) for n in NAMES}
        size = host.workspace_bytes(sum(lengths), heads, len(lengths))
        storage = torch.full((size + 512,), 0xCD, dtype=torch.uint8, device="cuda")
        ws = storage[256:-256]

        def launch():
            host.launch(
                device=0,
                stream=torch.cuda.current_stream().cuda_stream,
                workspace=ws.data_ptr(),
                **{n: x[n].data_ptr() for n in ("q", "k", "v", "g", "beta", "dO")},
                cu_seqlens=x["cu"].data_ptr(),
                initial_state=x["state"].data_ptr(),
                d_final_state=x["dS"].data_ptr(),
                **{n: outs[key].data_ptr() for n, key in zip(("dq", "dk", "dv", "dg", "dbeta", "d_initial_state"), NAMES)},
                total_tokens=sum(lengths),
                n_heads=heads,
                n_seqs=len(lengths),
                scale=128**-0.5,
            )

        launch()
        expected = {n: out.clone() for n, out in outs.items()}
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        for _ in range(10):
            graph.replay()
            for n, out in outs.items():
                assert torch.isfinite(out).all(), n
                assert torch.equal(out.view(torch.uint8), expected[n].view(torch.uint8)), n
            assert (storage[:256] == 0xCD).all() and (storage[-256:] == 0xCD).all()
        torch.cuda.synchronize()
        graph.reset()
        print("PASS", lengths, heads, flush=True)


if __name__ == "__main__":
    main()
