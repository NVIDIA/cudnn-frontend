# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph replay requires a GPU")
def test_graph_replay_executes_the_captured_calls_on_the_current_stream():
    from benchmark.attention_inference.benchmark_single_attention import time_graph_replay

    with torch.cuda.stream(torch.cuda.Stream()):
        value = torch.zeros(1, device="cuda")
        elapsed = time_graph_replay(lambda: value.add_(1), num_warmup=2, num_iters=3, calls_per_replay=5)
        assert elapsed > 0
        # Eager warmup and one warm replay both precede the timed replays.
        assert value.item() == 2 + (1 + 3) * 5
