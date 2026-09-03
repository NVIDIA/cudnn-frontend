# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import pytest
import torch

pytestmark = pytest.mark.L0

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_BENCHMARK_PATH = _REPOSITORY_ROOT / "benchmark/flex_attention/benchmark_flex_attention.py"
_SPEC = importlib.util.spec_from_file_location("cudnn_flex_attention_benchmark", _BENCHMARK_PATH)
assert _SPEC is not None and _SPEC.loader is not None
benchmark = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = benchmark
_SPEC.loader.exec_module(benchmark)


def test_standard_mask_specs_are_valid_interval_unions():
    for name in benchmark.MASK_NAMES:
        spec = benchmark.make_mask_spec(name, 257)
        endpoints = spec.endpoints
        assert endpoints.dtype == torch.int32
        assert endpoints.shape[1] == 257
        assert endpoints.shape[0] % 2 == 1
        assert endpoints.shape[0] < 33
        assert endpoints.is_contiguous()
        assert torch.all((0 <= endpoints) & (endpoints <= 257))
        assert torch.all(endpoints[1:] >= endpoints[:-1])
        assert spec.visible_pairs == benchmark.visible_pair_count(endpoints)
        assert 0 < spec.density <= 1


def test_visible_pair_count_matches_endpoint_predicate():
    seqlen = 257
    for name in benchmark.MASK_NAMES:
        spec = benchmark.make_mask_spec(name, seqlen)
        expected = sum(benchmark.endpoint_visible(spec.endpoints, q_idx, kv_idx) for q_idx in range(seqlen) for kv_idx in range(seqlen))
        assert spec.visible_pairs == expected


def test_active_flop_formulas():
    workload = benchmark.Workload(seqlen=128)
    visible_pairs = 1234
    head_pairs = workload.num_q_heads * visible_pairs
    assert benchmark._phase_flops(workload, visible_pairs, "forward") == 4 * head_pairs * 128
    assert benchmark._phase_flops(workload, visible_pairs, "backward") == 10 * head_pairs * 128
    assert benchmark._phase_flops(workload, visible_pairs, "combined") == 14 * head_pairs * 128
    assert math.isclose(
        benchmark._phase_flops(workload, visible_pairs, "combined") / benchmark._phase_flops(workload, visible_pairs, "forward"),
        3.5,
    )


def test_benchmark_is_flex_only():
    help_text = benchmark._build_parser().format_help()
    assert "--backend" not in help_text
    assert "FA4" not in help_text
    assert "Magi" not in help_text
    assert "PyTorch" not in help_text


def test_environment_has_no_flex_specific_cutlass_version_gate(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 0))
    monkeypatch.setattr(benchmark, "_safe_version", lambda _package: "4.5.0")
    benchmark._validate_environment()


def test_benchmark_dry_run_does_not_query_cuda(monkeypatch, capsys):
    def unexpected_cuda_query():
        raise AssertionError("dry-run must not query the CUDA runtime")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_cuda_query)
    benchmark.main(
        (
            "--dry-run",
            "--seqlen",
            "256",
            "--head-dim",
            "192",
            "--mask",
            "causal,hstu",
        )
    )
    output = capsys.readouterr().out
    assert "masks=2" in output
    assert "Dqk=192 Dv=128" in output
    assert "backend=flex_attention" in output
    assert "causal: nfunc=1" in output
    assert "hstu: nfunc=5" in output
