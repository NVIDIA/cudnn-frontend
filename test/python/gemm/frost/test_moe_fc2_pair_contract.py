# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paired FC2 graph eligibility, native metadata, and plan-only specialization."""

from dataclasses import replace
from unittest.mock import patch
from types import SimpleNamespace

import pytest
import cudnn
from cudnn import _pybind_module as native
from cudnn.gemm.frost.moe_fc2_pair import analyze_fc2, build_fc2, Fc2Compiled, Fc2Knobs
from cudnn.gemm.frost.moe_pair import KernelParams, pair_knobs
from cudnn.gemm.frost.engine import FrostGemmEngines
from cudnn.gemm.frost.heuristics import analyze_facts, recommend

pytestmark = pytest.mark.L0


def make_graph(*, r=8, n=256, k=128, pitched=False, weight_stride=None, output_stride=None, epilogue=False, output_fp32=False, input_fp32=False):
    b, f = cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT
    e = 4
    pitch = k + (8 if pitched else 0)
    g = cudnn.pygraph(io_data_type=b, intermediate_data_type=f, compute_data_type=f)
    x = g.tensor(name="x", dim=[1, r, k], stride=[r * k, k, 1], data_type=f if input_fp32 else b)
    w = g.tensor(name="w", dim=[e, k, n], stride=weight_stride or [n * pitch + (8 if pitched else 0), 1, pitch], data_type=b)
    o = g.tensor(name="offsets", dim=[e, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    y = g.moe_grouped_matmul(x, w, o, mode=cudnn.moe_grouped_matmul_mode.NONE, name="fc2")
    if epilogue:
        y = g.swish(y, name="silu")
    y.set_dim([1, r, n]).set_stride(output_stride or [r * n, n, 1]).set_data_type(f if output_fp32 else b).set_output(True)
    return g


@pytest.mark.parametrize("options", [{}, {"r": 1}, {"pitched": True}, {"n": 2048, "k": 768}, {"r": 9}, {"r": 17}, {"r": 64}, {"r": 512}, {"r": 513}])
def test_fc2_supported_graphs(options):
    g = make_graph(**options)
    spec = analyze_fc2(g)
    assert spec.binding.bound_tensors() == [spec.tokens, spec.weight, spec.output, spec.offsets]
    facts = analyze_facts(g)
    assert facts.fc2 == spec and facts.pair is None
    # Device eligibility is independently enforced, without importing the template.
    with patch("cudnn.gemm.frost.moe_pair.device_params", return_value=None):
        proposals = recommend("A", facts, {"frost_moe_fc2_pair": 20402})
    assert [p.engine_id for p in proposals] == [20402, 20402]
    assert [p.knobs.ab_stages for p in proposals] == [12, 6]
    engines = FrostGemmEngines({"frost_gemm": 20400, "frost_moe_swiglu_pair": 20401, "frost_moe_fc2_pair": 20402})
    assert [engine.engine_id for engine in engines] == [20400, 20401, 20402]
    knobs = pair_knobs()
    assert engines[-1].knobs_from_public(engines[-1].knobs_to_public(knobs)).to_public() == knobs.to_public()


@pytest.mark.parametrize(
    "options",
    [
        {"r": 514},
        {"r": 0},
        {"n": 64},
        {"k": 32},
        {"epilogue": True},
        {"output_fp32": True},
        {"input_fp32": True},
        {"weight_stride": [32768, 1, 132]},
        {"weight_stride": [32760, 1, 128]},
        {"weight_stride": [65536, 2, 256]},
        {"output_stride": [4096, 512, 1]},
    ],
)
def test_fc2_declines_unsupported_graphs(options):
    with pytest.raises((NotImplementedError, ValueError)):
        analyze_fc2(make_graph(**options))


def test_fc2_dynamic_and_unimplemented_knobs_decline_before_compilation():
    with pytest.raises(NotImplementedError, match="fixed graph"):
        analyze_fc2(make_graph(), dynamic_shapes=True)
    with patch("cudnn.gemm.frost.moe_fc2_pair.device_params", side_effect=AssertionError("must decline before probing device")):
        with pytest.raises(NotImplementedError, match="exact M128N8"):
            build_fc2(make_graph(), replace(pair_knobs(), moe_sched_policy=0))
    with patch("cudnn.gemm.frost.moe_pair.device_params", side_effect=NotImplementedError("wrong architecture")):
        assert recommend("A", analyze_facts(make_graph()), {"frost_moe_fc2_pair": 20402}) == []


def test_fc2_native_metadata_and_runtime_validation():
    def operand(pointer, shape, stride, code=4, bits=16):
        pack = native.VariantPackNative(1)
        pack.set_operand(0, pointer, list(shape), list(stride), code, bits)
        return pack.operand(0, 0)

    class Workspace:
        def view(self, offset, dtype, shape):
            assert (offset, dtype, shape) == (0, "int64", (4752,))
            return operand(0x500000, shape, (1,), 0, 64)

    spec = analyze_fc2(make_graph(pitched=True))
    compiled = Fc2Compiled.__new__(Fc2Compiled)
    compiled.spec = spec
    compiled.workspace_bytes = 38016
    calls = []
    compiled.kernel = lambda *args, **kwargs: calls.append((args, kwargs))
    tensors = (spec.tokens, spec.weight, spec.output, spec.offsets)
    pack = {t: operand((i + 1) * 0x100000, t.get_dim(), t.get_stride(), 0 if i == 3 else 4, 32 if i == 3 else 16) for i, t in enumerate(tensors)}
    for address in (0x200000, 0x600000):
        pack[spec.weight] = operand(address, spec.weight.get_dim(), spec.weight.get_stride())
        compiled(pack, Workspace(), stream=17)
        args, kw = calls.pop()
        assert args[0] == (8, 128, 128, 4, 4, 128, 1, 1024, 136, 1, 34824, 256, 1, 2048)
        assert args[4].data_ptr() == args[5].data_ptr() == address
        assert tuple(args[4].shape) == (256, 128, 4)
        assert int(kw["stream"]) == 17
    for bad in [
        operand(0x200002, spec.weight.get_dim(), spec.weight.get_stride()),
        operand(0x200000, spec.weight.get_dim(), [32768, 1, 128]),
        operand(0x200000, spec.weight.get_dim(), spec.weight.get_stride(), 2, 32),
    ]:
        pack[spec.weight] = bad
        with pytest.raises(ValueError):
            compiled(pack, Workspace(), stream=17)
        assert not calls


def test_fc2_pipeline_depth_replay_and_plan_identity():
    engines = FrostGemmEngines({"frost_gemm": 20400, "frost_moe_swiglu_pair": 20401, "frost_moe_fc2_pair": 20402})
    legacy = pair_knobs().to_public()
    assert Fc2Knobs().to_public() == legacy
    assert Fc2Knobs.from_public(legacy).ab_stages == 12
    for stages in (6, 12):
        record = {**legacy, cudnn.knob_type.STAGES: stages}
        native_knobs = engines[-1].knobs_from_public(record)
        assert native_knobs.ab_stages == stages
        assert Fc2Knobs.from_public(engines[-1].knobs_to_public(native_knobs)) == native_knobs
        for engine in engines[:2]:
            with pytest.raises(ValueError, match="not a tile axis"):
                engine.knobs_from_public(record)
    seen = []

    def template(path, params, tag):
        seen.append(params)
        return SimpleNamespace(compile=lambda: object())

    params = KernelParams(148, 44214954, "test helper digest")
    with patch("cudnn.gemm.frost.moe_fc2_pair.device_params", return_value=params):
        with patch("cudnn.frost.template_loader.load_template", side_effect=template):
            for knobs in (None, pair_knobs(), Fc2Knobs(ab_stages=6), Fc2Knobs.from_public(legacy)):
                assert build_fc2(make_graph(), knobs).workspace_bytes == 38016
    assert [p.ab_stages for p in seen] == [12, 12, 6, 12]
    # Depth reaches the actual template-loader key and source-digest parameters.
    assert seen[0] == seen[1] == seen[3] and seen[2] != seen[0]
    assert len(set(seen)) == 2 and repr(seen[2]) != repr(seen[0])


@pytest.mark.parametrize("value", [0, 1, 5, 7, 13, 6.0, "6", True, None])
def test_fc2_invalid_pipeline_depth_declines_before_device_probe(value):
    record = {**pair_knobs().to_public(), cudnn.knob_type.STAGES: value}
    with patch("cudnn.gemm.frost.moe_fc2_pair.device_params", side_effect=AssertionError("must decline before probing device")):
        with pytest.raises(NotImplementedError, match="STAGES must be 6 or 12"):
            build_fc2(make_graph(), record)


def test_fc2_depth_does_not_hide_invalid_geometry():
    with pytest.raises(ValueError, match="missing required"):
        Fc2Knobs.from_public({cudnn.knob_type.STAGES: 6})
    record = {**pair_knobs().to_public(), cudnn.knob_type.STAGES: 6, cudnn.knob_type.SCHED_POLICY: 0}
    with pytest.raises(ValueError, match="exact M128N8"):
        Fc2Knobs.from_public(record)
