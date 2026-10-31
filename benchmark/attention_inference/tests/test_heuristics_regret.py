# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""benchmark.heuristics_regret pairs cudnn_auto with the faster cudnn / cudnn_oss row of the same case."""

import csv
import subprocess
import sys


def _write(path, rows, extra=()):
    fields = ["config_name", "model_name", "phase", "backend", "data_type", "q_tokens", "kv_len", "time_ms", "success", "backend_detail", *extra]
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
            # inference details (benchmark_single_attention) and a bare training detail
            dict(backend="cudnn", q_tokens=1024, kv_len=1024, time_ms="1.0", backend_detail="cudnn 9.29.0 plan=eng10_k24=1"),
            dict(backend="cudnn_auto", q_tokens=1024, kv_len=1024, time_ms="1.2", backend_detail="cudnn_auto plan=eng10_k24=1"),
            dict(backend="cudnn", q_tokens=2048, kv_len=2048, time_ms="1.0", backend_detail="eng10_k24=1"),
            dict(backend="cudnn_oss", q_tokens=2048, kv_len=2048, time_ms="0.5", backend_detail="sdpa_fwd_prefill_sm100[TILE_M=128, TILE_N=128]"),
            dict(backend="cudnn_auto", q_tokens=2048, kv_len=2048, time_ms="1.1", backend_detail="eng10_k24=1"),
            # an unrecorded plan identity never counts as the same plan
            dict(backend="cudnn", q_tokens=4096, kv_len=4096, time_ms="1.0", backend_detail="unknown"),
            dict(backend="cudnn_auto", q_tokens=4096, kv_len=4096, time_ms="20.0", backend_detail="unknown"),
        ],
    )
    out = subprocess.run([sys.executable, "-m", "benchmark.heuristics_regret", str(tmp_path)], capture_output=True, text=True)
    assert "misranks (regret > 1.05): 2" in out.stdout and "| 2.20 |" in out.stdout and "| 20.00 |" in out.stdout, out.stdout
    assert "same plan as the faster leg (regret 1.0): 1" in out.stdout and "max 1.200" in out.stdout, out.stdout


def test_baseline_separates_new_default_plans_from_same_plan_slowdowns(tmp_path):
    """--baseline matches cases across runs on the columns both record and splits slowdowns by whether the default plan moved."""
    (tmp_path / "then").mkdir()
    (tmp_path / "now").mkdir()
    auto = lambda q, ms, plan, **kw: dict(backend="cudnn_auto", q_tokens=q, kv_len=8192, time_ms=ms, backend_detail=f"cudnn_auto plan={plan}", **kw)
    _write(tmp_path / "then" / "r.csv", [auto(1, "1.0", "A"), auto(2, "1.0", "B"), auto(3, "1.0", "C"), auto(4, "1.0", "D")])
    _write(
        tmp_path / "now" / "r.csv",
        [auto(1, "3.0", "A2"), auto(2, "1.5", "B"), auto(3, "", "C", success="False"), auto(4, "0.5", "D2")],
        extra=("attn_mask",),  # a column the baseline predates
    )
    run = lambda *extra: subprocess.run(
        [sys.executable, "-m", "benchmark.heuristics_regret", str(tmp_path / "now"), "--baseline", str(tmp_path / "then"), *extra],
        capture_output=True,
        text=True,
    )
    out = run()
    assert out.returncode == 0, out.stderr
    assert "matched cases: 3" in out.stdout, out.stdout
    assert "slower than 1.20x: 1 with a new default plan, 1 on the same plan" in out.stdout, out.stdout
    assert "| 3.00 |" in out.stdout and "| 1.50 |" in out.stdout and "default stopped running: 1" in out.stdout, out.stdout
    assert run("--fail").returncode == 1


def test_a_file_with_duplicate_legs_is_skipped_not_fatal(tmp_path):
    _write(tmp_path / "dup.csv", [dict(backend="cudnn", q_tokens=1, kv_len=1, time_ms="1.0"), dict(backend="cudnn", q_tokens=1, kv_len=1, time_ms="2.0")])
    _write(tmp_path / "ok.csv", [dict(backend="cudnn", q_tokens=1, kv_len=1, time_ms="1.0"), dict(backend="cudnn_auto", q_tokens=1, kv_len=1, time_ms="1.0")])
    out = subprocess.run([sys.executable, "-m", "benchmark.heuristics_regret", str(tmp_path)], capture_output=True, text=True)
    assert out.returncode == 0 and "compared cases: 1" in out.stdout, out.stdout + out.stderr
    assert "skipping" in out.stderr and "dup.csv" in out.stderr


def test_baseline_pairs_within_compatible_schemas(tmp_path):
    """A training CSV that already records attn_mask must not hide an inference regression whose baseline predates it."""
    for run in ("then", "now"):
        (tmp_path / run).mkdir()
        _write(
            tmp_path / run / "train.csv",
            [dict(backend="cudnn_auto", q_tokens=9, kv_len=9, time_ms="1.0", backend_detail="T", attn_mask="top_left", profile_pass="fwd")],
            extra=("attn_mask", "profile_pass"),
        )
    auto = lambda ms, plan: dict(backend="cudnn_auto", q_tokens=1, kv_len=8192, time_ms=ms, backend_detail=f"cudnn_auto plan={plan}")
    _write(tmp_path / "then" / "inf.csv", [auto("1.0", "A")])
    _write(tmp_path / "now" / "inf.csv", [dict(auto("3.0", "B"), attn_mask="no_mask")], extra=("attn_mask",))
    out = subprocess.run(
        [sys.executable, "-m", "benchmark.heuristics_regret", str(tmp_path / "now"), "--baseline", str(tmp_path / "then"), "--fail"],
        capture_output=True,
        text=True,
    )
    assert "matched cases: 2" in out.stdout and "1 with a new default plan" in out.stdout, out.stdout
    assert out.returncode == 1
