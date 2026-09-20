# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph and launch contracts for the explicit SM120 SwiGLU engine."""

from dataclasses import replace
from unittest.mock import patch
import pytest
import cudnn
from cudnn import _pybind_module as native
from cudnn.frost import buffers, device, template_loader
from cudnn.gemm.frost import moe_simt_sm120 as impl
from cudnn.gemm.frost.engine import FrostGemmEngines

pytestmark = pytest.mark.L0


def graph(
    *,
    r=8,
    n=768,
    k=2048,
    e=128,
    parent=True,
    reverse=False,
    reverse_halves=False,
    wrong_mul=False,
    narrow=False,
    extra=False,
    offset_count=None,
    padded=False,
    dtype=None,
    compact=False,
):
    bf16, fp32 = cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT
    g = cudnn.pygraph(io_data_type=bf16, intermediate_data_type=fp32, compute_data_type=fp32)
    x = g.tensor(name="tokens", dim=[1, r, k], stride=[r * k, k, 1], data_type=bf16)
    o = g.tensor(name="offsets", dim=[e if offset_count is None else offset_count, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    pitch = [(n if compact else 2 * n) * k, 1, k + 8 if padded else k]
    if parent:
        w = g.tensor(name="parent", dim=[e, k, 2 * n], stride=pitch, data_type=dtype or bf16)
        a = g.slice(w, [slice(None), slice(None), slice(0, n)], name="gate").set_stride(pitch)
        b = g.slice(w, [slice(None), slice(None), slice(n, 2 * n)], name="up").set_stride(pitch)
    else:
        a = g.tensor(name="gate", dim=[e, k, n], stride=pitch, data_type=dtype or bf16)
        b = g.tensor(name="up", dim=[e, k, n], stride=pitch, data_type=dtype or bf16)
    gate, up = (b, a) if reverse_halves else (a, b)
    weights = [up, gate] if reverse else [gate, up]
    mms = [g.moe_grouped_matmul(x, w, o, mode=cudnn.moe_grouped_matmul_mode.NONE, name="mm" + str(i)) for i, w in enumerate(weights)]
    mg, mu = mms[::-1] if reverse else mms
    sg = g.swish(mg, name="silu")
    if narrow:
        sg.set_data_type(bf16)
    y = g.mul(sg, mg if wrong_mul else mu, name="output").set_dim([1, r, n]).set_stride([r * n, n, 1]).set_output(True).set_data_type(bf16)
    if extra:
        mg.set_output(True)
    return g


def operand(pointer, dim, stride, *, dtype="bfloat16", device=0):
    pack = native.VariantPackNative(1)
    code, bits = {"bfloat16": (4, 16), "float32": (2, 32), "int32": (0, 32)}[dtype]
    pack.set_operand(0, pointer, list(dim), list(stride), code, bits)
    return pack.operand(0, device)


def frame(spec, offset=0):
    return {
        t: operand((i + 1) * 0x100000000 + offset, t.get_dim(), t.get_stride(), dtype="int32" if t is spec.offsets else "bfloat16")
        for i, t in enumerate(spec.binding.bound_tensors())
    }


@pytest.mark.parametrize("r", [1, 8, 9, 64, 4096])
@pytest.mark.parametrize("parent,compact", [(True, False), (False, False), (False, True)])
@pytest.mark.parametrize("reverse", [False, True])
def test_supported_graph_and_parent_binding(r, parent, compact, reverse):
    spec = impl.analyze(graph(r=r, parent=parent, compact=compact, reverse=reverse))
    assert spec.rows == r
    assert (spec.gate.source is spec.up.source) is parent
    assert spec.gate.byte_offset == 0
    assert spec.up.byte_offset == (3145728 if parent else 0)
    assert spec.gate.tensor.get_stride()[0] == (1572864 if compact else 3145728)


@pytest.mark.parametrize("reverse", [False, True])
def test_graph_semantics_determine_weight_order(reverse):
    spec = impl.analyze(graph(reverse=reverse, reverse_halves=True))
    assert (spec.gate.byte_offset, spec.up.byte_offset) == (3145728, 0)


@pytest.mark.parametrize(
    "options",
    [
        dict(r=0),
        dict(r=4097),
        dict(n=64),
        dict(k=1024),
        dict(e=64),
        dict(wrong_mul=True),
        dict(narrow=True),
        dict(extra=True),
        dict(offset_count=129),
        dict(padded=True),
        dict(dtype=cudnn.data_type.FLOAT),
    ],
)
def test_unsupported_graph_declines_before_device(options):
    (engine,) = FrostGemmEngines({impl.ENGINE: 20404})
    with patch.object(impl, "device_params", side_effect=AssertionError("unsupported graph reached device")):
        with pytest.raises(NotImplementedError):
            engine.check_support(graph(**options))


def test_dynamic_shape_and_nonempty_knobs_decline():
    with pytest.raises(NotImplementedError, match="fixed graph"):
        impl.analyze(graph(), dynamic_shapes=True)
    for value in [{1: 1}, False, 1, "default"]:
        with pytest.raises(ValueError):
            impl.validate_knobs(value)
    assert impl.validate_knobs(None) is None and impl.validate_knobs({}) is None


def test_engine_record_preserves_explicit_opt_in(monkeypatch):
    from cudnn.engines.manifest import engine_for_id

    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "0")
    assert engine_for_id(20404) is None
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    engine = engine_for_id(20404)
    assert engine.name == impl.ENGINE and engine.engine_id == 20404
    assert engine.knobs_to_public(None) == {}
    assert engine.knobs_from_public({}) is None
    with pytest.raises(ValueError):
        engine.knobs_from_public({cudnn.knob_type.STAGES: 2})


def make_compiled(spec):
    compiled = impl.SimtCompiled.__new__(impl.SimtCompiled)
    compiled.spec, compiled.binding, compiled.params = spec, spec.binding, impl.KernelParams(0, "cpu")
    calls = []
    compiled.kernel = lambda *a: calls.append(a)
    return compiled, calls


@pytest.mark.parametrize("parent,compact", [(True, False), (False, False), (False, True)])
def test_launch_uses_live_native_pointers_and_explicit_stream(parent, compact):
    spec = impl.analyze(graph(parent=parent, compact=compact))
    compiled, calls = make_compiled(spec)
    for offset, stream in [(0, 17), (0x1000000000, 29)]:
        pack = frame(spec, offset)
        compiled(pack, stream=stream)
        args = calls.pop()
        assert all(type(a) is int for a in args[:8])
        assert list(args[:5]) == [
            pack[spec.tokens].data_ptr(),
            pack[spec.gate.source].data_ptr() + spec.gate.byte_offset,
            pack[spec.up.source].data_ptr() + spec.up.byte_offset,
            pack[spec.offsets].data_ptr(),
            pack[spec.output].data_ptr(),
        ]
        assert args[5:8] == (8, 3145728 if compact else 6291456, 3145728 if compact else 6291456)
        assert int(args[8]) == stream
    assert not calls


@pytest.mark.parametrize("parent,compact", [(True, False), (False, False), (False, True)])
@pytest.mark.parametrize("bad", ["null", "unaligned", "device", "dtype", "shape", "stride", "offset_extent", "output_alias", "missing_stream"])
def test_invalid_runtime_bindings_do_not_launch(parent, compact, bad):
    spec = impl.analyze(graph(parent=parent, compact=compact))
    compiled, calls = make_compiled(spec)
    pack = frame(spec)
    if bad == "missing_stream":
        with pytest.raises(ValueError, match="explicit stream"):
            compiled(pack, stream=None)
    else:
        tensor = spec.output if bad == "output_alias" else spec.offsets if bad == "offset_extent" else spec.tokens
        options = dict(
            pointer=pack[tensor].data_ptr(), dim=tensor.get_dim(), stride=tensor.get_stride(), dtype="int32" if tensor is spec.offsets else "bfloat16"
        )
        change = {
            "null": dict(pointer=0),
            "unaligned": dict(pointer=options["pointer"] + 2),
            "device": dict(device=1),
            "dtype": dict(dtype="float32"),
            "shape": dict(dim=[1, 7, 2048]),
            "stride": dict(stride=[16384, 2049, 1]),
            "offset_extent": dict(dim=[129, 1, 1]),
            "output_alias": dict(pointer=pack[spec.tokens].data_ptr()),
        }[bad]
        pack[tensor] = operand(**{**options, **change})
        with pytest.raises(ValueError):
            compiled(pack, stream=17)
    assert not calls


@pytest.mark.parametrize("entry", ["support", "build"])
def test_dsl_version_gate_precedes_device_and_template(entry, monkeypatch):
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.6.2")))
    (engine,) = FrostGemmEngines({impl.ENGINE: 20404})
    with (
        patch.object(device, "resolve_device", side_effect=AssertionError("device before version gate")),
        patch.object(template_loader, "load_template", side_effect=AssertionError("template before version gate")),
    ):
        with pytest.raises(NotImplementedError, match=r"found 4\.6\.2"):
            engine.check_support(graph()) if entry == "support" else impl.build(graph(), {})


@pytest.mark.parametrize("arch,sms", [((10, 0), 148), ((12, 0), 170), ((12, 1), 188)])
def test_other_devices_decline(arch, sms):
    from cudnn.gemm.frost import compiler

    with (
        patch.object(compiler, "probe_cutedsl", return_value=None),
        patch.object(device, "resolve_device", return_value=0),
        patch.object(device, "compute_capability", return_value=arch),
        patch.object(device, "multiprocessor_count", return_value=sms),
    ):
        with pytest.raises(NotImplementedError, match="188-SM SM120"):
            impl.device_params()
