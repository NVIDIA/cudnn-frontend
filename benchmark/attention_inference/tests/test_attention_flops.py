# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest

pytest.importorskip("torch")
from benchmark.attention_inference import benchmark_single_attention as bench


@pytest.mark.L0
@pytest.mark.parametrize(
    "phase, sq, skv, causal, window, q_lens, kv_lens, pairs",
    [
        ("context", 4, 4, False, None, [4, 2], [4, 2], 16 + 4),
        ("context", 4, 4, True, None, [4, 2], [4, 2], 10 + 3),
        ("context", 4, 8, True, None, [4, 2], [8, 5], 26 + 9),
        ("context", 4, 4, True, 0, [4, 2], [4, 2], 10 + 3),
        ("context", 4, 4, True, 2, [4, 2], [4, 2], 7 + 3),
        ("generation", 3, 8, False, 2, [3], [8], 6),
        ("generation", 3, 8, False, None, [3], [8], 21),
        ("generation", 1, 8, False, 2, [1], [8], 2),
        ("generation", 1, 8, False, None, [1], [8], 8),
    ],
)
def test_result_counts_attended_pairs(monkeypatch, capsys, phase, sq, skv, causal, window, q_lens, kv_lens, pairs):
    # Fix the elapsed time so the reported TFLOPS directly exposes the numerator.
    args = [
        "bench",
        "--backend",
        "cudnn",
        "--phase",
        phase,
        "--q_tokens",
        str(sq),
        "--kv_len",
        str(skv),
        "--batch_size",
        str(len(q_lens)),
        "--num_q_heads",
        "4",
        "--num_kv_heads",
        "2",
        "--head_dim_qk",
        "64",
        "--head_dim_vo",
        "32",
    ]
    if causal:
        args.append("--causal")
    if window is not None:
        args.extend(["--sliding_window_size", str(window)])
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setitem(bench.SETUPS, "cudnn", lambda *args: (lambda: None, "test"))
    monkeypatch.setattr(bench, "time_fn", lambda *args: 1e-6)
    monkeypatch.setattr(bench, "request_lens", lambda args: (bench.torch.tensor(q_lens), bench.torch.tensor(kv_lens)))
    monkeypatch.setattr(bench.torch.cuda, "get_device_name", lambda: "test")
    bench.main()
    assert float(capsys.readouterr().out.split(",")[2]) == pytest.approx(2 * 4 * (64 + 32) * pairs / 1000)
