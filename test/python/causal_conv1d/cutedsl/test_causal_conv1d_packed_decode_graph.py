# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public decode graph replay with changing packed boundaries and history."""

import pytest
import torch
from test_causal_conv1d_packed import reference


@pytest.mark.L0
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("silu", [False, True])
@pytest.mark.parametrize("with_history", [False, True])
def test_decode_graph_replays_packed_boundaries(dtype, silu, with_history):
    from functools import partial
    from cudnn.ops import causal_conv1d as _causal_conv1d

    causal_conv1d = partial(_causal_conv1d, backend="se")

    torch.manual_seed(8441)
    tokens, hidden, width = 33, 136, 4
    x = (torch.randn(1, tokens, hidden, device="cuda", dtype=dtype) * 0.125).transpose(1, 2)
    weight = torch.randn(hidden, width, device="cuda", dtype=dtype) * 0.125
    bias = torch.randn(hidden, device="cuda", dtype=dtype) * 0.125
    initial = (torch.randn(tokens, width - 1, hidden, device="cuda", dtype=dtype) * 0.125).transpose(1, 2) if with_history else None
    final_out = torch.empty(tokens, hidden, width - 1, device="cuda", dtype=dtype)
    cu = torch.arange(tokens + 1, device="cuda", dtype=torch.int32)

    def call():
        return causal_conv1d(
            x,
            weight,
            bias,
            activation="silu" if silu else None,
            cu_seqlens=cu,
            initial_states=initial,
            return_final_states=True,
            final_states_out=final_out,
        )

    for _ in range(2):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y, final = call()
    assert final is final_out
    packs = ([1] * tokens, [0, 0, tokens] + [0] * (tokens - 3), [2, 0] + [1] * (tokens - 2))
    try:
        for lengths in packs:
            for value in (x, weight, bias, initial):
                if value is not None:
                    value.copy_(torch.randn_like(value) * 0.125)
            cu.copy_(torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), device="cuda", dtype=torch.int32))
            graph.replay()
            torch.cuda.synchronize()
            history = None if initial is None else initial.transpose(1, 2).double()
            expected_y, expected_final = reference(x[0].T.double(), weight.double(), bias.double(), history, lengths, silu)
            atol, rtol = (0.03, 0.02) if dtype == torch.bfloat16 else (0.003, 0.002)
            torch.testing.assert_close(y, expected_y.T[None].to(dtype), atol=atol, rtol=rtol)
            torch.testing.assert_close(final, expected_final.transpose(1, 2).to(dtype), atol=0, rtol=0)
    finally:
        graph.reset()
