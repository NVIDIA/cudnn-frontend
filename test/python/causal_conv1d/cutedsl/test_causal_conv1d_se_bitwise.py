# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact SE levels.layer cases, through CuTe DSL and the frontend backend bindings."""

import json
import os
from pathlib import Path

import pytest
import torch
from causal_conv1d.cutedsl.se_bitwise import CASES, MANIFEST, bitwise_difference, parameter_difference, run_case
from causal_conv1d.cutedsl.test_causal_conv1d_se import _dsl

pytestmark = [pytest.mark.L1, pytest.mark.gpu_exclusive, pytest.mark.xdist_group("causal_se_bitwise")]


def _write(name, result):
    """Persist optional per-case diagnostics with the pinned backend manifest metadata."""
    if destination := os.environ.get("CUDNN_SE_BITWISE_RESULTS"):
        folder = Path(destination)
        folder.mkdir(parents=True, exist_ok=True)
        result["manifest"] = {key: value for key, value in MANIFEST.items() if key != "cases"}
        (folder / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")


def test_representation_comparison():
    """Reject signed-zero mismatches and nonfinite values in exact data comparisons."""
    assert not bitwise_difference(torch.tensor([-0.0]), torch.tensor([0.0]))["equal"]
    assert not bitwise_difference(torch.tensor([float("nan")]), torch.tensor([float("nan")]))["equal"]
    assert bitwise_difference(torch.tensor([1.0]), torch.tensor([1.0]))["equal"]


def test_parameter_tolerance_contract():
    """Check both accepted rounding differences and rejected parameter-gradient errors."""
    reference = torch.tensor([0.0, 1.0], dtype=torch.float32)
    assert parameter_difference(reference + 1e-5, reference, 1)["accepted"]
    assert not parameter_difference(reference + 1e-2, reference, 1)["accepted"]
    assert not parameter_difference(torch.full_like(reference, float("nan")), reference, 1)["accepted"]
    double = reference.double()
    assert not parameter_difference(double + 1e-8, double, 1)["accepted"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_levels_bitwise(case):
    """Compare each named backend case with exact data bits and bounded atomic-gradient differences."""
    _dsl()
    cc = torch.cuda.get_device_capability()
    if cc[0] * 100 + cc[1] * 10 < case["min_device_version"]:
        pytest.skip("Below the backend manifest's architecture floor")
    try:
        result = run_case(case)
    except Exception as error:
        unsupported = "SHARED_MEMORY_INSUFFICIENT" in str(error) or "failed with status 3010" in str(error)
        _write(
            case["name"],
            {
                "case": case,
                "status": "backend_unsupported" if unsupported else "error",
                "error": str(error),
                "gpu": torch.cuda.get_device_name(),
                "cc": list(cc),
            },
        )
        if unsupported:
            pytest.skip(str(error))
        raise
    result["status"] = "pass" if result["accepted"] else ("data_bitwise_mismatch" if not result["data_bitwise_match"] else "parameter_tolerance_failure")
    _write(case["name"], result)
    if not result["accepted"]:
        counts = {name: value["mismatched_elements"] for name, value in result["trials"][0]["cross"].items() if not value["equal"]}
        pytest.fail(
            f"{case['name']}: {result['status']}, raw-bit mismatch counts={counts}; backend_repeatable={result['backend_repeatable']}, candidate_repeatable={result['candidate_repeatable']}"
        )
