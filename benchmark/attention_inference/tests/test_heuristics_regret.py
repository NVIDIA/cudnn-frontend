# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""benchmark.heuristics_regret pairs cudnn_auto with the faster cudnn / cudnn_oss row of the same case."""

import csv
import subprocess
import sys


def _write(path, rows):
    fields = ["config_name", "model_name", "phase", "backend", "data_type", "q_tokens", "kv_len", "time_ms", "success", "backend_detail"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**dict(config_name="c", model_name="m", phase="context", data_type="bfloat16", success="True", backend_detail=""), **r})


def test_regret_flags_misrank_and_coverage_loss(tmp_path):
    _write(
        tmp_path / "r.csv",
        [
            dict(backend="cudnn", q_tokens=1024, kv_len=1024, time_ms="1.0"),
            dict(backend="cudnn_oss", q_tokens=1024, kv_len=1024, time_ms="0.5"),
            dict(backend="cudnn_auto", q_tokens=1024, kv_len=1024, time_ms="0.5"),
            dict(backend="cudnn", q_tokens=2048, kv_len=2048, time_ms="1.0"),
            dict(backend="cudnn_oss", q_tokens=2048, kv_len=2048, time_ms="2.0"),
            dict(backend="cudnn_auto", q_tokens=2048, kv_len=2048, time_ms="2.0"),
            dict(backend="cudnn", q_tokens=4096, kv_len=4096, time_ms="1.0"),
            dict(backend="cudnn_auto", q_tokens=4096, kv_len=4096, time_ms="", success="False"),
            dict(backend="cudnn", q_tokens=1024, kv_len=1024, time_ms="0.4", data_type="float16"),
        ],
    )
    run = lambda *extra: subprocess.run([sys.executable, "-m", "benchmark.heuristics_regret", str(tmp_path), *extra], capture_output=True, text=True)
    out = run()
    assert out.returncode == 0, out.stderr
    assert "compared cases: 2" in out.stdout  # the float16 row has no cudnn_auto partner
    assert "misranks (regret > 1.05): 1" in out.stdout and "| 2.00 |" in out.stdout
    assert "coverage losses (cudnn_auto failed, another leg ran): 1" in out.stdout
    assert run("--fail").returncode == 1


def test_repeated_runs_stay_separate(tmp_path):
    """Two timestamped CSVs of the same case are two cases, not one overwriting the other."""
    _write(
        tmp_path / "a.csv",
        [dict(backend="cudnn", q_tokens=1024, kv_len=1024, time_ms="1.0"), dict(backend="cudnn_auto", q_tokens=1024, kv_len=1024, time_ms="2.0")],
    )
    _write(
        tmp_path / "b.csv",
        [dict(backend="cudnn", q_tokens=1024, kv_len=1024, time_ms="1.0"), dict(backend="cudnn_auto", q_tokens=1024, kv_len=1024, time_ms="1.0")],
    )
    out = subprocess.run([sys.executable, "-m", "benchmark.heuristics_regret", str(tmp_path)], capture_output=True, text=True)
    assert "compared cases: 2" in out.stdout and "misranks (regret > 1.05): 1" in out.stdout, out.stdout


def test_same_plan_is_noise_not_misrank(tmp_path):
    """cudnn_auto running the faster leg's own plan is a correct pick; the timing gap is reported as noise."""
    _write(
        tmp_path / "r.csv",
        [
            dict(backend="cudnn", q_tokens=1024, kv_len=1024, time_ms="1.0", backend_detail="eng10_k24=1"),
            dict(backend="cudnn_auto", q_tokens=1024, kv_len=1024, time_ms="1.2", backend_detail="cudnn_auto plan=eng10_k24=1"),
            dict(backend="cudnn", q_tokens=2048, kv_len=2048, time_ms="1.0", backend_detail="eng10_k24=1"),
            dict(backend="cudnn_oss", q_tokens=2048, kv_len=2048, time_ms="0.5", backend_detail="sdpa_fwd_prefill_sm100[TILE_M=128, TILE_N=128]"),
            dict(backend="cudnn_auto", q_tokens=2048, kv_len=2048, time_ms="1.1", backend_detail="eng10_k24=1"),
        ],
    )
    out = subprocess.run([sys.executable, "-m", "benchmark.heuristics_regret", str(tmp_path)], capture_output=True, text=True)
    assert "misranks (regret > 1.05): 1" in out.stdout and "| 2.20 |" in out.stdout, out.stdout
    assert "same plan as the faster leg (regret 1.0): 1" in out.stdout and "max 1.200" in out.stdout, out.stdout
