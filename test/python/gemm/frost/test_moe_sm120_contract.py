# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the explicit SM120 grouped projection graph engine."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import cudnn
from cudnn import _pybind_module as native
from cudnn.gemm.frost import moe_sm120 as m
from cudnn.gemm.frost.engine import FrostGemmEngines

pytestmark = pytest.mark.L0


def graph(
    *, r=8, e=128, groups=None, n=2048, k=768, dtype=None, output_dtype=None, wstride=None, xstride=None, ystride=None, offset_dtype=None, epilogue=False
):
    b, f = cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT
    groups = e if groups is None else groups
    g = cudnn.pygraph(io_data_type=b, intermediate_data_type=f, compute_data_type=f)
    x = g.tensor(name="x", dim=[1, r, k], stride=xstride or [r * k, k, 1], data_type=dtype or b)
    w = g.tensor(name="w", dim=[e, k, n], stride=wstride or [n * k, 1, k], data_type=b)
    off = g.tensor(name="offsets", dim=[groups, 1, 1], stride=[1, 1, 1], data_type=offset_dtype or cudnn.data_type.INT32)
    y = g.moe_grouped_matmul(x, w, off, mode=cudnn.moe_grouped_matmul_mode.NONE)
    if epilogue:
        y = g.swish(y)
    y.set_dim([1, r, n]).set_stride(ystride or [r * n, n, 1]).set_data_type(output_dtype or b).set_output(True)
    return g


@pytest.mark.parametrize(
    "options",
    [
        dict(),
        dict(r=1),
        dict(r=64),
        dict(r=512),
        dict(r=65537, e=40, k=128, n=256),
        dict(r=17, e=2, groups=4, k=64, n=64),
        dict(r=17, e=3, groups=5, k=64, n=64),
        dict(r=17, e=7, k=272, n=144),
    ],
)
def test_supported_graphs(options):
    spec = m.analyze(graph(**options))
    assert spec.binding.bound_tensors() == [spec.tokens, spec.weight, spec.output, spec.offsets]
    assert spec.groups == options.get("groups", options.get("e", 128))


@pytest.mark.parametrize(
    "options",
    [
        dict(r=0),
        dict(r=65538),
        dict(e=129),
        dict(groups=129),
        dict(n=8),
        dict(n=24),
        dict(n=8208),
        dict(k=7),
        dict(k=8193),
        dict(dtype=cudnn.data_type.FLOAT),
        dict(output_dtype=cudnn.data_type.FLOAT),
        dict(offset_dtype=cudnn.data_type.INT64),
        dict(epilogue=True),
        dict(wstride=[2048 * 776, 1, 776]),
        dict(xstride=[8 * 776, 776, 1]),
        dict(ystride=[8 * 2056, 2056, 1]),
    ],
)
def test_declines_graphs(options):
    with pytest.raises((NotImplementedError, ValueError, RuntimeError)):
        m.analyze(graph(**options))


def test_dynamic_declines():
    with pytest.raises(NotImplementedError, match="fixed graph"):
        m.analyze(graph(), dynamic_shapes=True)


@pytest.mark.parametrize("policy", [0, 1])
def test_public_knobs_and_plan_identity(policy):
    knob = m.Sm120Knobs(m.geometry(policy))
    (engine,) = FrostGemmEngines({m.ENGINE: 20403})
    assert engine.engine_id == 20403
    record = engine.knobs_to_public(knob)
    assert record[cudnn.knob_type.STAGES] == 2
    assert engine.knobs_from_public(record) == knob
    seen = []

    def load(path, params, tag):
        seen.append(params)
        return SimpleNamespace(compile=lambda: object())

    with patch.object(m, "device_params", return_value=m.KernelParams(0, 188, "identity")):
        with patch("cudnn.frost.template_loader.load_template", side_effect=load):
            plan = m.build(graph(), record)
    assert plan.workspace_bytes == 128 and len(seen) == 1
    assert seen[0].static_sched == bool(policy)
    assert seen[0].ab_stages == 2
    assert replace(seen[0], source_digest="other") != seen[0]


@pytest.mark.parametrize(
    "key,value",
    [
        (cudnn.knob_type.STAGES, True),
        (cudnn.knob_type.STAGES, 2.0),
        (cudnn.knob_type.STAGES, 4),
        (cudnn.knob_type.MMA_TILE_N, 16),
        (cudnn.knob_type.SWAP_AB, 0),
        (cudnn.knob_type.PIPELINE_ARCH, 100),
        (cudnn.knob_type.SCHED_POLICY, 2),
        (cudnn.knob_type.TILE_N, 16),
        (cudnn.knob_type.WARPS_N, 2),
    ],
)
def test_rejected_knobs_before_device(key, value):
    record = {**m.Sm120Knobs().to_public(), key: value}
    with patch.object(m, "device_params", side_effect=AssertionError("must decline before device")):
        with pytest.raises(NotImplementedError):
            m.build(graph(), record)


def test_dsl_gate_before_device_and_kernel_import():
    with patch("cudnn.gemm.frost.compiler.probe_cutedsl", side_effect=NotImplementedError("DSL too old")):
        with patch("cudnn.frost.device.resolve_device", side_effect=AssertionError("device before DSL gate")):
            with pytest.raises(NotImplementedError, match="DSL too old"):
                m.device_params()


def operand(pointer, shape, stride, code=4, bits=16):
    pack = native.VariantPackNative(1)
    pack.set_operand(0, pointer, list(shape), list(stride), code, bits)
    return pack.operand(0, 0)


def test_native_metadata_live_binding_and_invalid_operands():
    spec = m.analyze(graph(r=17, e=3, groups=5, k=64, n=64))
    compiled = m.ProjectionCompiled.__new__(m.ProjectionCompiled)
    compiled.spec = spec
    compiled.params = m.KernelParams(0, 188, "identity")
    calls = []
    compiled.kernel = lambda *a, **kw: calls.append((a, kw))

    class Workspace:
        def view(self, offset, dtype, shape):
            assert (offset, dtype, shape) == (0, "int64", (16,))
            return operand(0x900000, shape, (1,), 0, 64)

    tensors = (spec.tokens, spec.weight, spec.output, spec.offsets)
    pack = {t: operand((i + 1) * 0x100000, t.get_dim(), t.get_stride(), 0 if i == 3 else 4, 32 if i == 3 else 16) for i, t in enumerate(tensors)}
    for pointer in (0x200000, 0x600000):
        pack[spec.weight] = operand(pointer, spec.weight.get_dim(), spec.weight.get_stride())
        compiled(pack, Workspace(), stream=17)
        args, kw = calls.pop()
        assert args[0] == (64, 17, 64, 3, 5, 64, 1, 4096, 64, 1, 1088, 1, 64, 1088)
        assert args[3].data_ptr() == pointer and tuple(args[3].shape) == (64, 64, 3)
        assert args[4].data_ptr() == 0x100000 and args[5].data_ptr() == 0x300000
        assert tuple(args[1].shape) == (5,) and int(kw["stream"]) == 17
    for bad in [
        operand(0x200002, spec.weight.get_dim(), spec.weight.get_stride()),
        operand(0x200000, spec.weight.get_dim(), [4096, 64, 1]),
        operand(0x200000, spec.weight.get_dim(), spec.weight.get_stride(), 2, 32),
    ]:
        pack[spec.weight] = bad
        with pytest.raises(ValueError):
            compiled(pack, Workspace(), stream=17)
        assert not calls
    pack[spec.weight] = operand(0x200000, spec.weight.get_dim(), spec.weight.get_stride())
    compiled.params = replace(compiled.params, device_ordinal=1)
    with pytest.raises(ValueError, match="compiled device"):
        compiled(pack, Workspace(), stream=17)
    assert not calls


def launch_fixture():
    spec = m.analyze(graph(r=17, e=3, groups=5, k=64, n=64))
    compiled = m.ProjectionCompiled.__new__(m.ProjectionCompiled)
    compiled.spec = spec
    compiled.params = m.KernelParams(0, 188, "contract")
    calls = []
    compiled.kernel = lambda *args, **kwargs: calls.append((args, kwargs))
    tensors = (spec.tokens, spec.weight, spec.output, spec.offsets)
    pack = {t: operand((i + 1) * 0x100000, t.get_dim(), t.get_stride(), 0 if i == 3 else 4, 32 if i == 3 else 16) for i, t in enumerate(tensors)}

    class Scratch:
        def __init__(self, pointer=0x900000, device=0):
            self.pointer, self.device = pointer, device

        def view(self, offset, dtype, shape):
            assert (offset, dtype, shape) == (0, "int64", (16,))
            return native.make_operand_buffer(self.pointer, list(shape), 0, 64, self.device)

    return compiled, pack, Scratch, calls


@pytest.mark.parametrize("slot", range(4))
def test_null_operands_decline_before_launch(slot):
    compiled, pack, Scratch, calls = launch_fixture()
    tensor = list(pack)[slot]
    pack[tensor] = operand(0, tensor.get_dim(), tensor.get_stride(), 0 if slot == 3 else 4, 32 if slot == 3 else 16)
    with pytest.raises(ValueError, match="aligned CUDA operands"):
        compiled(pack, Scratch(), stream=17)
    assert not calls


def test_missing_stream_declines_before_launch():
    compiled, pack, Scratch, calls = launch_fixture()
    with pytest.raises(ValueError, match="explicit stream"):
        compiled(pack, Scratch(), stream=None)
    assert not calls


@pytest.mark.parametrize("slot", [0, 1, 3])
def test_output_overlap_declines_before_launch(slot):
    compiled, pack, Scratch, calls = launch_fixture()
    source = list(pack)[slot]
    output = compiled.spec.output
    pack[output] = operand(pack[source].data_ptr(), output.get_dim(), output.get_stride())
    with pytest.raises(ValueError, match="output must not overlap"):
        compiled(pack, Scratch(), stream=17)
    assert not calls


@pytest.mark.parametrize("slot", range(4))
def test_workspace_overlap_declines_before_launch(slot):
    compiled, pack, Scratch, calls = launch_fixture()
    pointer = pack[list(pack)[slot]].data_ptr()
    with pytest.raises(ValueError, match="workspace must not overlap"):
        compiled(pack, Scratch(pointer), stream=17)
    assert not calls


@pytest.mark.parametrize("pointer,device", [(0, 0), (0x900008, 0), (0x900000, 1)])
def test_invalid_workspace_declines_before_launch(pointer, device):
    compiled, pack, Scratch, calls = launch_fixture()
    with pytest.raises(ValueError, match="workspace"):
        compiled(pack, Scratch(pointer, device), stream=17)
    assert not calls


def test_version_floor_precedes_device_and_template_import(monkeypatch):
    from cudnn.frost import buffers, device, template_loader

    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.6.2")))

    with (
        patch.object(device, "resolve_device", side_effect=AssertionError("device used before version check")),
        patch.object(template_loader, "load_template", side_effect=AssertionError("kernel used before version check")),
    ):
        with pytest.raises(NotImplementedError, match="4.7"):
            m.build(graph(), None)


def test_adjacent_output_and_workspace_remain_supported():
    compiled, pack, Scratch, calls = launch_fixture()
    spec = compiled.spec
    pointer = pack[spec.tokens].data_ptr() + spec.rows * spec.reduction * 2
    pack[spec.output] = operand(pointer, spec.output.get_dim(), spec.output.get_stride())
    scratch = Scratch(pack[spec.tokens].data_ptr() - 128)
    compiled(pack, scratch, stream=0)
    assert len(calls) == 1 and int(calls[0][1]["stream"]) == 0
    assert calls[0][0][5].data_ptr() == pointer


def test_missing_workspace_declines_before_launch():
    compiled, pack, Scratch, calls = launch_fixture()
    with pytest.raises(ValueError, match="workspace"):
        compiled(pack, None, stream=17)
    assert not calls
