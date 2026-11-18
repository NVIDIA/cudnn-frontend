# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from typing import Optional

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))

from benchmark.attention_training.flops import count_causal_nonmasked_elems  # noqa: E402

pytestmark = pytest.mark.L0


def _reference_causal_nonmasked_elems(q_seqlen: int, kv_seqlen: int, attn_mask: str, sliding_window_size: Optional[int]):
    diagonal_offset = kv_seqlen - q_seqlen if attn_mask == "bottom_right" else 0
    total = 0

    for q_idx in range(q_seqlen):
        for kv_idx in range(kv_seqlen):
            distance_from_diagonal = kv_idx - (q_idx + diagonal_offset)
            if distance_from_diagonal > 0:
                continue
            if sliding_window_size is not None and distance_from_diagonal <= -sliding_window_size:
                continue
            total += 1

    return total


def test_count_causal_nonmasked_elems_square_top_left():
    assert count_causal_nonmasked_elems(4, 4, "top_left") == 10


def test_count_causal_nonmasked_elems_rectangular_top_left():
    assert count_causal_nonmasked_elems(5, 3, "top_left") == 12
    assert count_causal_nonmasked_elems(3, 6, "top_left") == 6


def test_count_causal_nonmasked_elems_rectangular_bottom_right():
    assert count_causal_nonmasked_elems(5, 3, "bottom_right") == 6
    assert count_causal_nonmasked_elems(3, 6, "bottom_right") == 15


def test_count_causal_nonmasked_elems_sliding_window():
    assert count_causal_nonmasked_elems(6, 6, "top_left", sliding_window_size=3) == 15
    assert count_causal_nonmasked_elems(5, 3, "bottom_right", sliding_window_size=2) == 5


def test_count_causal_nonmasked_elems_matches_reference():
    cases = [
        (4, 4, "top_left", None),
        (5, 3, "top_left", None),
        (3, 6, "top_left", 2),
        (4, 4, "bottom_right", None),
        (5, 3, "bottom_right", 2),
        (3, 6, "bottom_right", 3),
    ]

    for q_seqlen, kv_seqlen, attn_mask, sliding_window_size in cases:
        assert count_causal_nonmasked_elems(
            q_seqlen=q_seqlen,
            kv_seqlen=kv_seqlen,
            attn_mask=attn_mask,
            sliding_window_size=sliding_window_size,
        ) == _reference_causal_nonmasked_elems(
            q_seqlen=q_seqlen,
            kv_seqlen=kv_seqlen,
            attn_mask=attn_mask,
            sliding_window_size=sliding_window_size,
        )


def _backward_graph_call_kwargs(src: str):
    """``{method: {keyword, ...}}`` for every ``<graph>.sdpa*backward(...)`` call the harness builds."""
    import ast

    out = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr.startswith("sdpa") and node.func.attr.endswith("backward"):
            out.setdefault(node.func.attr, set()).update(k.arg for k in node.keywords if k.arg)
    return out


def test_backward_graphs_carry_the_sliding_window_the_flop_model_counts():
    """Rate each backward graph using the sliding window that it actually executes.

    Half, FP8 and MXFP8 backward carry the band. FP8 and MXFP8 use
    ``left_bound``; the FP8 binding derives the causal right bound and rejects
    an explicit one. The backward FLOP count must use the same window.
    """
    src = (_REPO_ROOT / "benchmark" / "attention_training" / "benchmark_single_sdpa.py").read_text()
    kw = _backward_graph_call_kwargs(src)
    assert "diagonal_band_left_bound" in kw["sdpa_backward"], "the half backward call dropped the band"
    assert "left_bound" in kw["sdpa_fp8_backward"], "the fp8 backward call dropped the band (a window run measures a plain-causal graph)"
    assert "right_bound" not in kw["sdpa_fp8_backward"], "sdpa_fp8_backward raises when use_causal_mask and right_bound are both set"
    assert "left_bound" in kw["sdpa_mxfp8_backward"], "the mxfp8 backward call dropped the band (a window run measures a plain-causal graph)"
    assert "bwd_sliding_window = args.sliding_window_size" in src, "every backward graph carries the band: rate it on the window"
    assert src.count("bwd_sliding_window,") == 1, "the backward tflops_per_sec call must take bwd_sliding_window, not args.sliding_window_size"


@pytest.mark.parametrize("profile_pass", ["fwd", "bwd"])
def test_benchmark_fits_one_workspace_across_iterations(profile_pass):
    """A fixed graph must run several iterations within one workspace allocation."""
    import subprocess
    import textwrap

    script = _REPO_ROOT / "benchmark" / "attention_training" / "benchmark_single_sdpa.py"
    # For this BF16 case, uint8 CUDA allocations are the graph workspace. Reject
    # another allocation before it can consume memory, reproducing a device
    # that fits one workspace but cannot temporarily hold a second copy.
    probe = textwrap.dedent("""\
        import runpy, sys
        import torch
        script, profile_pass = sys.argv[1:]
        original_empty = torch.empty
        workspace_allocations = []
        def bounded_empty(*args, **kwargs):
            if kwargs.get('dtype') == torch.uint8 and str(kwargs.get('device', '')).startswith('cuda'):
                assert not workspace_allocations, 'benchmark exceeded its single-workspace memory budget'
                workspace_allocations.append(args)
            return original_empty(*args, **kwargs)
        torch.empty = bounded_empty
        sys.path.insert(0, str(__import__('pathlib').Path(script).parent))
        sys.argv = [script, '--sdpa_backend', 'cudnn', '--data_type', 'bfloat16',
                    '--batch_size', '1', '--q_seqlen', '128', '--kv_seqlen', '128',
                    '--num_q_heads', '4', '--num_kv_heads', '2', '--head_dim', '64',
                    '--profile_pass', profile_pass, '--num_iterations', '2', '--skip_ref']
        runpy.run_path(script, run_name='__main__')
        assert len(workspace_allocations) == 1, workspace_allocations
        print('SINGLE_WORKSPACE_BENCHMARK_PASSED')
        """)
    result = subprocess.run([sys.executable, "-c", probe, str(script), profile_pass], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SINGLE_WORKSPACE_BENCHMARK_PASSED" in result.stdout


@pytest.mark.parametrize("skip_ref", [False, True])
def test_hopper_fa3_benchmark_runs_requested_provider(monkeypatch, capsys, skip_ref):
    import math
    import runpy

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires Hopper for FlashAttention 3")
    fa3 = pytest.importorskip("flash_attn_interface")
    calls = []

    def record(provider, fn):
        def observed(q, k, v, **kwargs):
            calls.append((provider, q.shape, k.shape))
            return fn(q, k, v, **kwargs)

        return observed

    monkeypatch.setattr(fa3, "flash_attn_func", record("FA3", fa3.flash_attn_func))
    if skip_ref:
        monkeypatch.setitem(sys.modules, "flash_attn.cute.interface", None)
    else:
        fa4 = pytest.importorskip("flash_attn.cute.interface")
        monkeypatch.setattr(fa4, "flash_attn_func", record("FA4", fa4.flash_attn_func))
    script = _REPO_ROOT / "benchmark/attention_training/benchmark_single_sdpa.py"
    monkeypatch.syspath_prepend(str(script.parent))
    args = (
        "--sdpa_backend flash_attention_3 --data_type bfloat16 --batch_size 1 --q_seqlen 128 --kv_seqlen 128 "
        "--num_q_heads 4 --num_kv_heads 2 --head_dim 64 --profile_pass both --num_iterations 1 --fa4_num_splits 1 --format_output"
    ).split()
    monkeypatch.setattr(sys, "argv", [str(script), *args] + (["--skip_ref"] if skip_ref else []))
    runpy.run_path(str(script), run_name="__main__")
    assert calls[0][0] == "FA3"
    assert sum(provider == "FA4" for provider, _, _ in calls) == int(not skip_ref)
    assert all(q == (1, 128, 4, 64) and k == (1, 128, 2, 64) for _, q, k in calls)
    stdout = capsys.readouterr().out
    assert "Failed reference check" not in stdout
    assert math.isnan(float(stdout.strip().splitlines()[-1].split(",")[12])) == skip_ref


@pytest.mark.parametrize(
    ("profile_pass", "perturbation", "iterations"),
    [("bwd", None, 1), ("bwd", "gradient", 1), ("fwd", "output", 6)],
)
def test_benchmark_reports_reference_check_for_single_pass(monkeypatch, capsys, profile_pass, perturbation, iterations):
    import math
    import runpy

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires Hopper for the FA4 reference")
    pytest.importorskip("flash_attn_interface")
    fa4 = pytest.importorskip("flash_attn.cute.interface")
    original = fa4.flash_attn_func
    calls = 0

    def reference(q, k, v, **kwargs):
        nonlocal calls
        calls += 1
        if perturbation == "gradient":
            q = q.view_as(q)
            q.register_hook(lambda grad: grad + 1)
        output, lse = original(q, k, v, **kwargs)
        if perturbation == "output" and calls == 1:
            output = output + 1
        return output, lse

    monkeypatch.setattr(fa4, "flash_attn_func", reference)
    script = _REPO_ROOT / "benchmark/attention_training/benchmark_single_sdpa.py"
    monkeypatch.syspath_prepend(str(script.parent))
    args = (
        "--sdpa_backend flash_attention_3 --data_type bfloat16 --batch_size 1 --q_seqlen 128 --kv_seqlen 128 "
        "--num_q_heads 4 --num_kv_heads 2 --head_dim 64 --fa4_num_splits 1 --format_output"
    ).split()
    monkeypatch.setattr(sys, "argv", [str(script), *args, "--profile_pass", profile_pass, "--num_iterations", str(iterations)])
    runpy.run_path(str(script), run_name="__main__")
    stdout = capsys.readouterr().out
    fields = stdout.strip().splitlines()[-1].split(",")
    assert float(fields[8 if profile_pass == "fwd" else 9]) > 0
    assert calls == iterations
    assert ("Failed reference check" in stdout) == (perturbation is not None)
    assert math.isnan(float(fields[12])) == (perturbation is not None)


def test_runner_csv_preserves_small_and_missing_reference_errors(tmp_path):
    import csv
    from dataclasses import replace

    pytest.importorskip("pandas")
    from benchmark.attention_training.config_types import BenchmarkConfig, BenchmarkResult, ModelPreset
    from benchmark.attention_training.runner import BenchmarkRunner

    config = BenchmarkConfig(name="reference", models=[ModelPreset("gqa", 4, 2)], seqlens=[(128, 128)])
    result = BenchmarkResult(
        config_name=config.name,
        model_name="gqa",
        backend="cudnn",
        data_type="float16",
        attn_mask="no_mask",
        batch_size=1,
        q_seqlen=128,
        kv_seqlen=128,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim_qk=64,
        head_dim_vo=64,
        profile_pass="fwd",
        deterministic_bwd=False,
        time_ms=0.125,
        tflops=1,
        max_diff=2.0**-22,
        num_iterations=1,
    )
    path = BenchmarkRunner().save_csv([result, replace(result, max_diff=float("nan"))], config, tmp_path / "results.csv")
    with path.open() as file:
        measured, unchecked = csv.DictReader(file)
    assert float(measured["max_diff"]) == result.max_diff
    assert unchecked["max_diff"] == ""
