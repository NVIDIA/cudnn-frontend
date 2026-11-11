# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU smoke of the dense Qwen model sample (``samples/models/qwen_dense/00_decode.py``): the script is loaded by path and its
``main()`` is called as a user's default run -- the reduced seeded model, 1 and 4 sequences, the sample's own checks against its
fp64-accumulated reference (replay, every provider block teacher-forced, the greedy tokens, each sequence alone against the
same sequence in the batch).  The run must print ``[00] PASS`` and the environment line; on a GPU, a library version or a
free-memory budget the sample does not serve it prints ``[00] SKIP <requirement>`` and the cell is reported as skipped with
that sentence -- unless a check failed before it: a skip never covers a failure.  Each sequence count also reports its
traced pass (GPU busy, idle and launches per step from kernel activity records, or the reason there are none) and the
bytes of a step.  The
verdict line, the stack, the versions and the wall time are recorded through ``frost_routing.measured``, so the result -- or
the skip -- reaches the job log of a passing run.  L0 runs the two-layer model (one linear-attention + one attention layer), L1
the four-layer one (three + one)."""

import importlib.util
import sys
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cudnn")

import frost_routing  # noqa: E402

MODELS = Path(__file__).resolve().parents[4] / "samples" / "models"
SCRIPT = MODELS / "qwen_dense" / "00_decode.py"


@pytest.fixture()
def sample():
    """The sample script as a module; its ``sys.path`` entry and every ``model_samples`` module leave with the test."""
    paths = list(sys.path)
    spec = importlib.util.spec_from_file_location("model_sample_qwen_dense_00_decode", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    sys.path[:] = paths
    for name in [n for n in sys.modules if n == "model_samples" or n.startswith("model_samples.")]:
        del sys.modules[name]


def _run(sample, capsys, arguments):
    start = time.perf_counter()
    code = sample.main(arguments)
    seconds = time.perf_counter() - start
    out = capsys.readouterr().out
    tagged = [line for line in out.splitlines() if line.startswith("[00] ")]
    versions = next((line[5:] for line in tagged if line.startswith("[00] cuDNN Frontend ")), "no environment line")
    device = next((line[5:].split(" | ")[0] for line in tagged if " SMs)" in line), "no device line")
    verdict = tagged[-1] if tagged else "no output"
    key = "model sample qwen_dense 00_decode " + (" ".join(arguments) or "(default)")
    frost_routing.measured(key, f"{verdict}; stack fe-cublas; {device}; {versions}; {seconds:.0f} s")
    if verdict.startswith("[00] SKIP"):
        assert " FAIL" not in out and "[00] ERROR" not in out, out[-6000:]  # a skip never covers a failed check
        pytest.skip(verdict)
    assert code == 0 and verdict == "[00] PASS", out[-6000:]
    assert "distribution" in versions and "source hash" in versions, "the environment check ran and its line was printed"
    assert "by provider: cuDNN FE 1 | fallbacks 8 (cuBLAS over copies 4, PyTorch chains 4) | PyTorch 2" in out
    for check in ("replay", "blocks", "tokens"):
        assert out.count(f"  {check} ") == 2 and f"  {check} " in out, f"the {check} check ran at both sequence counts"
    assert "linear_attention.step equal (gated)" in out or "REPORTED" in out, "batch invariance was checked at 4 sequences"
    assert out.count("\ntraced pass ") == 2 and out.count("\nbytes per step ") == 2, "each count reports its traced pass (or why there is none) and its bytes"
    assert " FAIL" not in out
    return out


@pytest.mark.L0
def test_default_run_passes_its_own_checks(sample, capsys):
    out = _run(sample, capsys, [])
    assert "2 of 64 layers (1 linear-attention + 1 attention)" in out and "20 block calls per step" in out


@pytest.mark.L1
def test_four_layer_run_passes_its_own_checks(sample, capsys):
    out = _run(sample, capsys, ["--layers", "4"])
    assert "4 of 64 layers (3 linear-attention + 1 attention)" in out and "34 block calls per step" in out
