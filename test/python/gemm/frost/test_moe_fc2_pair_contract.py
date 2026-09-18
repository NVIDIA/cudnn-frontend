# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paired FC2 graph eligibility, native metadata, and plan-only specialization."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import cudnn
from cudnn import _pybind_module as native
from cudnn.gemm.frost.moe_fc2_pair import analyze_fc2, build_fc2, Fc2Compiled
from cudnn.gemm.frost.moe_pair import pair_knobs
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
    assert len(proposals) == 1 and proposals[0].engine_id == 20402
    engines = FrostGemmEngines({"frost_gemm": 20400, "frost_moe_swiglu_pair": 20401, "frost_moe_fc2_pair": 20402})
    assert [engine.engine_id for engine in engines] == [20400, 20401, 20402]
    knobs = pair_knobs()
    assert engines[-1].knobs_from_public(engines[-1].knobs_to_public(knobs)) == knobs


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
