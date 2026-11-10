# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cake fused activation contracts and dequantized reference checks."""

import hashlib
import json
from pathlib import Path

import pytest

from cudnn.gemm.cake.gated_act_mxfp8 import GatedActMxfp8Plan, _geometry, _outputs

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("shape", [(0, 256), (128, 128), (127, 256), (128, 257), (128,), (2**24, 256), (True, 256)])
def test_reject_geometry(shape):
    with pytest.raises(ValueError):
        _geometry(shape, True, False)


@pytest.mark.parametrize("orientations", [(False, False), (1, False), (True, None)])
def test_reject_orientations(orientations):
    with pytest.raises(ValueError):
        _geometry((128, 256), *orientations)


def test_vendor_files_match_pin():
    root = Path(__file__).resolve().parents[3] / "python/cudnn/_cake/_vendor"
    record = json.loads((root / "VENDORED_GATED_ACT_MXFP8.json").read_text())
    for name, digest in record["files"].items():
        assert hashlib.sha256((root / "cake_gated_act_mxfp8_backend" / name).read_bytes()).hexdigest() == digest, name


def test_invalid_outputs_fail_before_launch():
    torch = pytest.importorskip("torch")
    pytest.importorskip("tvm_ffi")
    plan = object.__new__(GatedActMxfp8Plan)
    plan.m, plan.k, plan.n = 128, 128, 128
    plan.device = torch.device("cpu")
    plan.backward, plan.rowwise, plan.colwise = False, True, True
    plan._entry = lambda *args: pytest.fail("invalid bindings reached a launch")
    plan._arguments = ()
    x = torch.empty((128, 256), dtype=torch.bfloat16)
    outputs = _outputs(plan)
    bad = list(outputs)
    bad[3] = bad[3][:-1]
    with pytest.raises(ValueError, match="col_scales must have shape"):
        plan.execute(x, bad)
    bad = list(outputs)
    bad[1] = bad[0].t()
    with pytest.raises(ValueError, match="overlap"):
        plan.execute(x, bad)
    with pytest.raises(ValueError, match="forward plan"):
        plan.execute(x, outputs, x)


@pytest.fixture
def torch_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Cake requires SM100a/SM103a/SM107a")
    from cudnn._cake import loader

    try:
        loader.check_toolkit(loader.device_arch(), "gated activation test")
    except RuntimeError as error:
        pytest.skip(str(error))
    return torch


def _dequantize(torch, values, scales):
    rows, cols = values.shape
    r = torch.arange(rows, device=values.device)[:, None]
    b = torch.arange(cols // 32, device=values.device)[None, :]
    offset = ((r // 128) * (cols // 128) + b // 4) * 512 + (r % 32) * 16 + (r % 128 // 32) * 4 + b % 4
    exponent = scales.flatten()[offset].long().repeat_interleave(32, dim=1) - 127
    return values.double() * torch.exp2(exponent.double())


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("rowwise,colwise", [(True, False), (False, True), (True, True)])
def test_outputs_and_capture(torch_cuda, backward, rowwise, colwise):
    torch = torch_cuda
    torch.manual_seed(9)
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16) if backward else None
    plan = GatedActMxfp8Plan(x.shape, backward=backward, rowwise=rowwise, colwise=colwise, device=x.device)
    outputs = _outputs(plan)
    gate, up = x.double().chunk(2, -1)
    sigmoid = torch.sigmoid(gate)
    if backward:
        reference = torch.cat(((sigmoid + gate * sigmoid * (1 - sigmoid)) * dy.double() * up, gate * sigmoid * dy.double()), dim=1)
    else:
        reference = gate * sigmoid * up
    plan.execute(x, outputs, dy)
    if rowwise:
        error = (_dequantize(torch, outputs[0], outputs[2]) - reference).square().sum().sqrt() / reference.square().sum().sqrt()
        assert error < 0.05
    if colwise:
        error = (_dequantize(torch, outputs[1].t(), outputs[3]) - reference.t()).square().sum().sqrt() / reference.square().sum().sqrt()
        assert error < 0.05
    saved = [value.view(torch.uint8).clone() for value in outputs]
    torch.cuda.set_sync_debug_mode("error")
    try:
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        plan.execute(x, outputs, dy)
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    finally:
        torch.cuda.set_sync_debug_mode("default")
    for value, expected in zip(outputs, saved):
        assert torch.equal(value.view(torch.uint8), expected)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        plan.execute(x, outputs, dy)
    x.neg_()
    graph.replay()
    captured = [value.view(torch.uint8).clone() for value in outputs]
    plan.execute(x, outputs, dy)
    for value, expected in zip(outputs, captured):
        assert torch.equal(value.view(torch.uint8), expected)
    bad = list(outputs)
    index = 0 if rowwise else 1
    bad[index] = bad[index][:-1]
    with pytest.raises(ValueError, match="must have shape"):
        plan.execute(x, bad, dy)
    for value, expected in zip(outputs, captured):
        assert torch.equal(value.view(torch.uint8), expected)
