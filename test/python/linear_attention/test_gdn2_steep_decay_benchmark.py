# SPDX-FileCopyrightText: Copyright (c) 2026 Alfred Wong
# SPDX-License-Identifier: Apache-2.0

"""CPU subprocess checks for benchmark validation with mocked GPU boundaries."""

import contextlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.L0
SCRIPT = Path(__file__).with_name("benchmark_gdn2_steep_decay.py")


def probe(output_dir):
    """Exercise the real benchmark functions with CPU tensors and fake GPU calls."""
    import torch

    torch.set_num_threads(1)
    modules = {name: ModuleType(name) for name in ("cudnn", "cudnn.linear_attention.ops.gdn2", "cudnn.linear_attention.frost")}
    modules["cudnn"].__file__ = str(SCRIPT.parents[3] / "python/cudnn/__init__.py")
    modules["cudnn"].backend_version = lambda: 0
    modules["cudnn.linear_attention.ops.gdn2"].gated_delta_net_v2 = lambda *args, **kwargs: (sum(args[:6]), None)
    modules["cudnn.linear_attention.frost"].gdn2_engine = SimpleNamespace()
    with patch.dict(sys.modules, modules):
        namespace = runpy.run_path(str(SCRIPT))["main"].__globals__

    def inputs(*args):
        return [torch.ones(2, 1, 1, requires_grad=True) for _ in range(6)], torch.ones(2, 1, 1)

    def reference(*args):
        return [torch.full((2, 1, 1), 6.0), *(torch.ones(2, 1, 1) for _ in range(6))]

    scheduler_flag = 2

    @contextlib.contextmanager
    def observe(records):
        records.append({"plan": "CompiledGdn2Bwd", "schedulers": {"scheduler_bwd": [0, scheduler_flag]}})
        yield

    event = SimpleNamespace(device_type=torch.autograd.DeviceType.CUDA, time_range=SimpleNamespace(start=0.0, end=1000.0), name="mock_kernel")
    profile = SimpleNamespace(events=lambda: [event])
    tensor = torch.tensor

    def expect_error(fn, message):
        try:
            fn()
        except ValueError as error:
            if message not in str(error):
                raise RuntimeError(f"Expected {message!r}, got {error!r}") from error
        else:
            raise RuntimeError(f"Validation did not reject {message!r}")

    with (
        patch.dict(namespace, inputs=inputs, reference=reference, observe_schedulers=observe, timed=lambda fn: {}),
        patch.object(torch.cuda, "synchronize"),
        patch.object(torch, "tensor", side_effect=lambda *args, **kwargs: tensor(*args, **{**kwargs, "device": "cpu"})),
        patch.object(torch.profiler, "profile", side_effect=lambda **kwargs: contextlib.nullcontext(profile)),
    ):
        run_case = lambda steep=1: namespace["run_case"](2, steep, output_dir, "probe")
        valid = run_case()
        if not valid["correct_within_one_percent"]:
            raise RuntimeError("Valid case failed")
        expect_error(lambda: run_case(0), "Unexpected backward scheduler")
        scheduler_flag = 0
        expect_error(run_case, "Unexpected backward scheduler")
        if not run_case(0)["correct_within_one_percent"]:
            raise RuntimeError("Valid mild case failed")
        scheduler_flag = 2
        with patch.dict(namespace, observe_schedulers=lambda records: contextlib.nullcontext()):
            expect_error(run_case, "No backward scheduler")
        with patch.object(profile, "events", return_value=[]):
            expect_error(run_case, "Profiler recorded no GPU activity")
        with patch.dict(namespace, reference=lambda *args: [torch.full((2, 1, 1), float("nan")), *reference()[1:]]):
            expect_error(run_case, "Non-finite output")
        with patch.dict(namespace, gated_delta_net_v2=lambda *args, **kwargs: (sum(args[:6]) * float("nan"), None)):
            expect_error(run_case, "Non-finite output")

    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "get_device_name", return_value="mock"),
        patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)),
        patch("importlib.metadata.version", return_value="mock"),
        patch.object(sys, "argv", [str(SCRIPT), "--label", "probe", "--output", str(output_dir / "run.json")]),
        patch.dict(namespace, run_case=lambda *args: {"correct_within_one_percent": False}),
    ):
        expect_error(namespace["main"], "FP64 tolerance failed")
        failed = json.loads((output_dir / "run.json").read_text())
        if len(failed["results"]) != 6 or any(row["correct_within_one_percent"] for row in failed["results"]):
            raise RuntimeError("Failed-case evidence was not preserved")
        with patch.dict(namespace, run_case=lambda *args: {"correct_within_one_percent": True}):
            namespace["main"]()
        passed = json.loads((output_dir / "run.json").read_text())
        if len(passed["results"]) != 6 or not all(row["correct_within_one_percent"] for row in passed["results"]):
            raise RuntimeError("Valid benchmark did not complete")


@pytest.mark.parametrize("optimized", [False, True])
def test_benchmark_validation_survives_optimized_python(tmp_path, optimized):
    """Accept valid records and reject invalid records in both Python modes."""
    pytest.importorskip("torch")
    env = os.environ.copy()
    env.pop("PYTHONOPTIMIZE", None)
    result = subprocess.run(
        [sys.executable, *(["-O"] if optimized else []), str(Path(__file__).resolve()), str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    probe(Path(sys.argv[1]))
