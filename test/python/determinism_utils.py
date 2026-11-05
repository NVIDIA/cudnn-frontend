# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Byte comparisons, scheduling pressure, and versioned golden checks."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import pytest
import torch

from test_utils import bitwise_bits


def snapshot(outputs, *, allowed_inf=None):
    assert outputs, "no outputs to compare"
    allowed_inf = {"lse": -torch.inf} if allowed_inf is None else allowed_inf
    result = {}
    for name, value in outputs.items():
        assert isinstance(value, torch.Tensor), f"{name}: missing tensor"
        valid = torch.isfinite(value)
        if name in allowed_inf:
            assert allowed_inf[name] in (-torch.inf, torch.inf), "only infinity sentinels may be allowed"
            valid |= value == allowed_inf[name]
        assert valid.all(), f"{name}: non-finite result"
        result[name] = value.detach().clone()
    return result


def assert_bitwise(actual, expected, label=""):
    assert actual.keys() == expected.keys(), f"{label}: output names changed"
    for name, want in expected.items():
        got = actual[name]
        assert got.shape == want.shape and got.dtype == want.dtype, f"{label} {name}: shape or dtype changed"
        a, b = bitwise_bits(got.reshape(-1)), bitwise_bits(want.reshape(-1))
        count = torch.count_nonzero(a != b).item()
        assert count == 0, f"{label} {name}: {count}/{a.numel()} bytes differ"


def assert_replays(run, repeats=8, *, allowed_inf=None):
    assert repeats >= 2, "at least two executions are required"
    expected = snapshot(run(), allowed_inf=allowed_inf)  # Compile before adding scheduling pressure.
    stream = torch.cuda.Stream()
    noise = torch.ones(1024 * 1024, device="cuda")
    for repeat in range(1, repeats):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(32):
                noise.mul_(1.00001).add_(0.00001)
        try:
            actual = snapshot(run(), allowed_inf=allowed_inf)
        finally:
            torch.cuda.current_stream().wait_stream(stream)
        assert_bitwise(actual, expected, f"replay {repeat}")
    return expected


def tensor_record(value):
    data = bitwise_bits(value.detach().cpu().reshape(-1)).numpy().tobytes()
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(data).hexdigest()}


def runtime():
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    import cudnn
    from cuda.bindings import driver

    error, driver_version = driver.cuDriverGetVersion()
    assert error == driver.CUresult.CUDA_SUCCESS, "cannot read CUDA driver version"

    return {
        "gpu": props.name,
        "sm": [props.major, props.minor],
        "sm_count": props.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_driver": driver_version,
        "cudnn": cudnn.backend_version(),
        "cutlass": importlib.metadata.version("nvidia-cutlass-dsl"),
        "tvm_ffi": importlib.metadata.version("apache-tvm-ffi"),
    }


def assert_golden(name, inputs, outputs, *, allowed_inf=None):
    """Compare committed bytes; recording requires an explicit, new destination."""
    environment = runtime()
    key = hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest()[:16]
    record_dir = os.environ.get("CUDNN_DETERMINISM_RECORD")
    check_dir = os.environ.get("CUDNN_DETERMINISM_GOLDENS")
    assert not (record_dir and check_dir), "choose recording or comparison"
    root = Path(record_dir or check_dir or Path(__file__).with_name("determinism_goldens"))
    path = root / key / f"{name}.json"
    record = {
        "schema": 1,
        "runtime": environment,
        "inputs": {name: tensor_record(value) for name, value in inputs.items()},
        "outputs": {name: tensor_record(value) for name, value in snapshot(outputs, allowed_inf=allowed_inf).items()},
    }
    if record_dir:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as file:
            json.dump(record, file, indent=2, sort_keys=True)
            file.write("\n")
        return
    if not path.exists():
        message = f"no golden for {name} on {environment}: {path}"
        if check_dir:
            pytest.fail(message)
        pytest.skip(message)
    expected = json.loads(path.read_text())
    assert record == expected, f"{name}: golden mismatch in {path}; compare input, runtime, and output records"
