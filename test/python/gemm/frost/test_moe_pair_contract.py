# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit parent binding, paired eligibility, and native metadata contracts."""

from dataclasses import replace

import pytest
import cudnn
from cudnn import _pybind_module as native
from cudnn.gemm.frost.graph_analyzer import analyze_with_binding, resolve_variant_pack, swap_ab_binding
from cudnn.gemm.frost.moe_pair import analyze_pair, pair_knobs
from cudnn.gemm.frost.knobs import GemmKnobs
from cudnn.gemm.frost.engine import FrostGemmEngines

pytestmark = pytest.mark.L0


def test_explicit_parent_binding_contract():
    bf16, fp32 = cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT
    checks = []

    def graph_case(*, slices=None, materialized=False, extra_consumer=False, dtype=bf16, output_stride=None):
        g = cudnn.pygraph(io_data_type=bf16, intermediate_data_type=fp32, compute_data_type=fp32)
        x = g.tensor(name="x", dim=[1, 8, 64], stride=[512, 64, 1], data_type=bf16)
        w = g.tensor(name="parent", dim=[4, 64, 128], stride=[8192, 1, 64], data_type=dtype)
        offsets = g.tensor(name="offsets", dim=[4, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
        gate = g.slice(w, slices or [slice(None), slice(None), slice(0, 64)], name="gate_weight")
        gate.set_stride(output_stride or [8192, 1, 64])
        up = g.slice(w, [slice(None), slice(None), slice(64, 128)], name="up_weight")
        up.set_stride([8192, 1, 64])
        if materialized:
            gate.set_output(True)
        if extra_consumer:
            tap = g.identity(gate, name="illegal_weight_tap")
            tap.set_output(True)
        mm_gate = g.moe_grouped_matmul(x, gate, offsets, mode=cudnn.moe_grouped_matmul_mode.NONE, name="gate")
        mm_up = g.moe_grouped_matmul(x, up, offsets, mode=cudnn.moe_grouped_matmul_mode.NONE, name="up")
        out = g.mul(g.swish(mm_gate, name="silu"), mm_up, name="out")
        out.set_dim([1, 8, 64]).set_stride([512, 64, 1]).set_output(True).set_data_type(bf16)
        return g, (x, w, offsets, out, gate, up)

    def operand(ptr, dim, stride, code=4, bits=16):
        pack = native.VariantPackNative(1)
        pack.set_operand(0, ptr, dim, stride, code, bits)
        return pack.operand(0, 0)

    def reject(label, fn, error=(NotImplementedError, ValueError, KeyError)):
        try:
            fn()
        except error as exc:
            checks.append(dict(name=label, rejected=type(exc).__name__, message=str(exc)))
        else:
            raise AssertionError(f"{label} was accepted")

    g, (x, w, offsets, out, gate, up) = graph_case()
    chain, binding = analyze_with_binding(g)
    assert chain.num_gemms == 2 and chain.num_b_operands == 2
    assert binding.bound_tensors() == [x, w, out, offsets]
    assert len(binding.operand_slices) == 2
    checks.append(dict(name="one_external_weight_two_logical_operands"))
    for address in (0x100000, 0x300000):
        parent = operand(address, [4, 64, 128], [8192, 1, 64])
        resolved = resolve_variant_pack({w: parent}, binding)
        assert resolved[id(gate)].data_ptr() == address
        assert resolved[id(up)].data_ptr() == address + 64 * 64 * 2
        for tensor in (gate, up):
            assert tuple(resolved[id(tensor)].shape) == (4, 64, 64)
            assert tuple(resolved[id(tensor)].stride()) == (8192, 1, 64)
        checks.append(dict(name="live_parent_pointer", address=address))
    swapped = swap_ab_binding(binding)
    assert swapped.operand_slices == binding.operand_slices
    assert {id(t) for t in swapped.bound_tensors()} == {id(t) for t in binding.bound_tensors()}
    checks.append(dict(name="swap_ab_retains_explicit_parent"))
    reject("missing_parent", lambda: resolve_variant_pack({}, binding))
    reject("virtual_weight_not_an_external_input", lambda: resolve_variant_pack({gate: parent}, binding))
    reject("wrong_parent_stride", lambda: resolve_variant_pack({w: operand(0x100000, [4, 64, 128], [8192, 128, 1])}, binding))
    reject("wrong_parent_dtype", lambda: resolve_variant_pack({w: operand(0x100000, [4, 64, 128], [8192, 1, 64], 2, 32)}, binding))
    for label, kw in (
        ("materialized_slice", dict(materialized=True)),
        ("non_gemm_consumer", dict(extra_consumer=True)),
        ("non_bf16_parent", dict(dtype=fp32)),
        ("slice_step_two", dict(slices=[slice(None), slice(None), slice(0, 128, 2)])),
        ("slice_experts", dict(slices=[slice(0, 2), slice(None), slice(0, 64)])),
        ("slice_k", dict(slices=[slice(None), slice(0, 32), slice(0, 64)])),
        ("wrong_slice_strides", dict(output_stride=[4096, 1, 64])),
    ):
        graph, _ = graph_case(**kw)
        reject(label, lambda: analyze_with_binding(graph))


def test_paired_graph_eligibility_and_native_metadata():
    bf16, fp32 = cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT
    checks = []

    def make(*, r=8, n=64, k=128, reverse_nodes=False, reverse_halves=False, narrow_activation=False, wrong_mul=False, extra_output=False, padded=False):
        e = 4
        pitch = k + 8 if padded else k
        stride = [2 * n * pitch + (8 if padded else 0), 1, pitch]
        g = cudnn.pygraph(io_data_type=bf16, intermediate_data_type=fp32, compute_data_type=fp32)
        x = g.tensor(name="x", dim=[1, r, k], stride=[r * k, k, 1], data_type=bf16)
        w = g.tensor(name="parent", dim=[e, k, 2 * n], stride=stride, data_type=bf16)
        o = g.tensor(name="offsets", dim=[e, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
        a = g.slice(w, [slice(None), slice(None), slice(0, n)], name="first").set_stride(stride)
        b = g.slice(w, [slice(None), slice(None), slice(n, 2 * n)], name="second").set_stride(stride)
        gate, up = (b, a) if reverse_halves else (a, b)
        weights = [up, gate] if reverse_nodes else [gate, up]
        mms = [g.moe_grouped_matmul(x, weight, o, mode=cudnn.moe_grouped_matmul_mode.NONE, name=f"mm{i}") for i, weight in enumerate(weights)]
        mg, mu = mms[::-1] if reverse_nodes else mms
        sg = g.swish(mg, name="silu")
        if narrow_activation:
            sg.set_data_type(bf16)
        y = g.mul(sg, mg if wrong_mul else mu, name="out").set_dim([1, r, n]).set_stride([r * n, n, 1]).set_data_type(bf16).set_output(True)
        if extra_output:
            mg.set_output(True)
        return g

    for options in ({}, {"reverse_nodes": True}, {"padded": True}, {"r": 1}, {"n": 768, "k": 2048}):
        spec = analyze_pair(make(**options))
        assert len(spec.binding.operand_slices) == 2
        assert spec.binding.bound_tensors() == [spec.tokens, spec.weight, spec.output, spec.offsets]
        checks.append(dict(accepted=options, rows=spec.rows, features=spec.features))
    for options in ({"r": 9}, {"n": 32}, {"k": 32}, {"reverse_halves": True}, {"narrow_activation": True}, {"wrong_mul": True}, {"extra_output": True}):
        try:
            analyze_pair(make(**options))
        except (NotImplementedError, ValueError) as exc:
            checks.append(dict(rejected=options, reason=str(exc)))
        else:
            raise AssertionError(options)
    try:
        analyze_pair(make(), dynamic_shapes=True)
    except NotImplementedError:
        checks.append(dict(rejected="dynamic_shapes"))
    else:
        raise AssertionError("dynamic accepted")
    knobs = pair_knobs()
    assert GemmKnobs.from_public(knobs.to_public()) == knobs
    assert knobs.swap_ab and knobs.moe_sched_policy == 1 and (knobs.mma_tile_m, knobs.mma_tile_n, knobs.mma_tile_k_bytes) == (128, 8, 32)
    engines = FrostGemmEngines({"frost_gemm": 20400, "frost_moe_swiglu_pair": 20401})
    assert [engine.engine_id for engine in engines] == [20400, 20401]
    assert engines[1].knobs_from_public(engines[1].knobs_to_public(knobs)) == knobs
    checks.append(dict(name="public_knob_roundtrip_and_append_only_engine_slot"))
    from cudnn import _pybind_module as native
    from cudnn.gemm.frost.moe_pair import PairedCompiled, build_pair

    try:
        build_pair(make(), replace(knobs, moe_sched_policy=0))
    except NotImplementedError:
        checks.append(dict(name="unsupported_knob_declines_before_device_or_compile"))
    else:
        raise AssertionError("unimplemented knob silently accepted")

    def operand(pointer, shape, stride, code=4, bits=16):
        pack = native.VariantPackNative(1)
        pack.set_operand(0, pointer, list(shape), list(stride), code, bits)
        return pack.operand(0, 0)

    class Workspace:
        def view(self, offset, dtype, shape):
            assert offset == 0 and dtype == "int64"
            return operand(0x500000, shape, (1,), 0, 64)

    spec = analyze_pair(make(padded=True))
    compiled = PairedCompiled.__new__(PairedCompiled)
    compiled.spec = spec
    compiled.workspace_bytes = 38016
    calls = []
    compiled.kernel = lambda *args, **kwargs: calls.append((args, kwargs))
    tensors = (spec.tokens, spec.weight, spec.output, spec.offsets)
    pack = {
        tensor: operand((index + 1) * 0x100000, tensor.get_dim(), tensor.get_stride(), 0 if index == 3 else 4, 32 if index == 3 else 16)
        for index, tensor in enumerate(tensors)
    }
    compiled(pack, Workspace(), stream=17)
    args_, kwargs = calls.pop()
    assert args_[0] == (8, 64, 128, 4, 4, 128, 1, 1024, 136, 1, 17416, 64, 1, 512)
    assert args_[4].data_ptr() == args_[5].data_ptr() == 0x200000
    assert tuple(args_[4].shape) == (128, 128, 4)
    assert tuple(args_[1].shape) == (4,) and int(kwargs["stream"]) == 17
    checks.append(dict(name="native_execute_metadata_uses_live_parent_and_caller_workspace"))
