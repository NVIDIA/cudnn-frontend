# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operand coordinate-map composition, actual transforms and full-GEMM contracts."""

import ast
import importlib.util
import textwrap
import warnings
from dataclasses import replace
from itertools import product
from math import prod
from pathlib import Path

import cudnn
import pytest
import torch

from cudnn.engines import ExecutionContext
from cudnn.gemm.frost.engine import _FrostGemmPlan
from cudnn.gemm.frost.fusion_ir import Axis, swap_ab
from cudnn.gemm.frost.graph_analyzer import analyze, analyze_with_binding, swap_ab_binding
from cudnn.gemm.frost.sm100 import compiler, epilogue_codegen
from cudnn.gemm.frost.tile_config import by_name
from gemm_test_utils import requires_sm100

BF16, FP16, FP32 = cudnn.data_type.BFLOAT16, cudnn.data_type.HALF, cudnn.data_type.FLOAT


def _config(swapped=False, cta_group=1, split=1):
    return replace(by_name(f"CONFIG_sm100_128x128x64_128x128x32_cluster{cta_group}x1_{cta_group}ctamma"), swap_ab=swapped, split_k_slices=split)


def _graph(side="a", mode="reshape", dtype=BF16, axis=1, combine=True, spatial=128):
    """Build view compositions on the same carrier, with explicit node rounding."""
    g = cudnn.pygraph(io_data_type=dtype, intermediate_data_type=FP32, compute_data_type=FP32)
    tensors, views, operands = {}, {}, {}
    for key, source, target in (("a", (2, 2, spatial // 2, 136), (2, spatial, 136)), ("b", (2, 17, 8, spatial), (2, 136, spatial))):
        steps = [("reshape", target)]
        if mode == "transpose_reshape":
            source = (*source[:-2], source[-1], source[-2])
            steps.insert(0, ("transpose", (*range(len(source) - 2), len(source) - 1, len(source) - 2)))
        elif mode == "reshape_transpose":
            source = (2, 2, 136, spatial // 2) if key == "a" else (2, 2, spatial, 68)
            steps = [("reshape", (*target[:-2], target[-1], target[-2])), ("transpose", (0, 2, 1))]
        elif mode == "consecutive":
            steps = [("reshape", (2, prod(source[1:]))), ("reshape", target), ("reshape", target)]
        strides = [prod(source[i + 1 :]) for i in range(len(source))]
        if mode == "transpose_reshape":
            strides[-2], strides[-1] = 1, source[-2]
        value = tensors[key] = g.tensor(name=key, dim=list(source), stride=strides, data_type=dtype)
        if key in side:
            indexed = g.gen_index(input=value, axis=axis, name=f"{key}_index").set_data_type(dtype)
            value = g.add(a=value, b=indexed).set_data_type(dtype) if combine else indexed
            value = g.leaky_relu(input=value, negative_slope=0.125).set_data_type(dtype)
        for kind, argument in steps:
            if kind == "reshape":
                value = g.reshape(input=value).set_dim(list(argument))
            else:
                value = g.transpose(input=value, permutation=list(argument))
        views[key] = steps
        operands[key] = value
    tensors["y"] = g.matmul(A=operands["a"], B=operands["b"]).set_output(True)
    g.validate()
    return g, tensors, views


def _buffers(tensors, device="cpu", fill=None):
    """Honor original source layouts; view projection must retain these pointers."""
    result = {}
    for key, tensor in tensors.items():
        dtype = torch.float16 if tensor.get_data_type() == FP16 else torch.bfloat16
        value = torch.empty_strided(tuple(tensor.get_dim()), tuple(tensor.get_stride()), dtype=dtype, device=device)
        result[key] = value.normal_() if fill is None and key != "y" else value.fill_(float("nan") if key == "y" else fill)
    return result


def _view(value, steps):
    for kind, argument in steps:
        value = value.reshape(argument) if kind == "reshape" else value.permute(argument)
    return value


def _reference(value, steps, axis, combine=True):
    """Independent Cartesian indices and framework views, never IndexExpr evaluation."""
    shape = [1] * value.ndim
    shape[axis] = value.shape[axis]
    index = torch.arange(value.shape[axis], device=value.device, dtype=torch.float32).reshape(shape).expand(value.shape).to(value.dtype)
    result = (value.float() + index.float()).to(value.dtype) if combine else index
    result = torch.nn.functional.leaky_relu(result.float(), 0.125).to(value.dtype)
    return _view(result, steps)


def _pack(g, tensors, buffers):
    return g._normalize(
        {t.get_uid(): buffers[key] for key, t in tensors.items()},
        None,
        [t.get_uid() for t in tensors.values()],
        [list(buffers[key].shape) for key in tensors],
        [list(buffers[key].stride()) for key in tensors],
    )


@pytest.mark.L0
@pytest.mark.parametrize(
    "side,mode,swapped,axis", list(product(("a", "b", "ab"), ("reshape", "transpose_reshape", "reshape_transpose", "consecutive"), (False, True), (0, 1, 3)))
)
def test_frost_operand_index_maps_and_binding(monkeypatch, side, mode, swapped, axis):
    """All compositions share map lowering and transport only the original storage."""
    g, tensors, views = _graph(side, mode, axis=axis)
    before = repr([(dict(n.params), [(t.get_dim(), t.get_stride()) for t in n.outputs.values()]) for n in g.nodes])
    chain, binding = analyze_with_binding(g)
    if swapped:
        chain, binding = swap_ab(chain), swap_ab_binding(binding)
    buffers = _buffers(tensors)
    for original in side:
        operand = ("b" if original == "a" else "a") if swapped else original
        ops = chain.mainloop_a_ops if operand == "a" else chain.mainloop_b_ops
        assert [op.op for op in ops] == ["gen_index", "add", "leaky_relu"]
        assert ops[1].parent_idx == -1 and ops[1].parent_idx_b == 0
        logical = torch.arange(tensors[original].get_dim()[axis]).reshape([tensors[original].get_dim()[axis] if i == axis else 1 for i in range(4)])
        expected = _view(logical.expand(tuple(tensors[original].get_dim())), views[original])
        if original == "b":
            expected = expected.transpose(-1, -2)
        row_axis = Axis.M if operand == "a" else Axis.N
        for b, row, k in ((0, 0, 0), (1, 77, 135), (1, 127, 17)):
            assert ops[0].index_expr.evaluate({Axis.BATCH: b, row_axis: row, Axis.K: k}) == expected[b, row, k].item()
    monkeypatch.setattr(compiler, "_check_plan_device", lambda device: None)
    frames = []
    plan = compiler.CompiledFusedGemm(
        chain, _config(swapped), [], Path("unused"), lambda *args, **kwargs: frames.append(args), binding=binding, vec_bytes_epi=16
    )
    engine = _FrostGemmPlan(plan)
    engine.execute(g, _pack(g, tensors, buffers), ExecutionContext(stream=0))
    for argument, original in zip(frames[-1][1:3], ("b", "a") if swapped else ("a", "b")):
        assert argument.data_ptr() == buffers[original].data_ptr()
    invalid = dict(buffers, a=buffers["a"].flatten(1))
    buffers["y"].fill_(123)
    with pytest.raises(ValueError, match="prepared source shape"):
        engine.execute(g, _pack(g, tensors, invalid), ExecutionContext(stream=0))
    assert len(frames) == 1 and torch.all(buffers["y"] == 123)
    engine.execute(g, _pack(g, tensors, buffers), ExecutionContext(stream=0))
    assert len(frames) == 2 and not chain.aux_tensors
    snippets = epilogue_codegen.generate(chain, vec_bytes_epi=16, output_elem_bytes=2, tma_slots=compiler._tma_slots_for(chain, _config(swapped)))
    source = compiler._render_template(chain, snippets, _config(swapped))
    ast.parse(source)
    assert "mainloop_has_index = True" in source and "mainloop_koob_fix = True" in source
    assert "ml_raw_m = _m_idx" in source and "ml_batch = tile_l // split_k_slices" in source
    assert before == repr([(dict(n.params), [(t.get_dim(), t.get_stride()) for t in n.outputs.values()]) for n in g.nodes])


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for generated operand transform")
@pytest.mark.parametrize(
    "side,mode,swapped,dtype", list(product(("a", "b"), ("reshape", "transpose_reshape", "reshape_transpose", "consecutive"), (False, True), (BF16, FP16)))
)
def test_frost_operand_index_generated_transform(tmp_path, side, mode, swapped, dtype):
    """Execute actual transform fragments on SM90+, not full Blackwell scheduling."""
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("generated fragments require SM90+")
    axis = 1
    g, tensors, views = _graph(side, mode, dtype=dtype, axis=axis)
    chain = analyze(g)
    if swapped:
        chain = swap_ab(chain)
    operand = ("b" if side == "a" else "a") if swapped else side
    snippet = epilogue_codegen.generate_mainloop(chain, operand)
    major = chain.matmul.a_major == "m" if operand == "a" else chain.matmul.b_major == "n"
    dt = "cutlass.BFloat16" if dtype == BF16 else "cutlass.Float16"
    source = f"""import cutlass
import cutlass.cute as cute

cta_tile_mnk = (128, 128, 64)
a_tma_group_elems = 64
b_tma_group_elems = 64

@cute.kernel
def kernel(X: cute.Tensor, C: cute.Tensor, batch: cutlass.Int64, ktile: cutlass.Int64):
    tid = cute.arch.thread_idx()[0]
    ml_vec_elems = 8
    ml_coord_m = cutlass.Int64(0)
    ml_coord_n = cutlass.Int64(0)
    ml_batch = batch
    k_tile_idx = ktile
    k = cutlass.Int64(136)
    m, n = cutlass.Int64(128), cutlass.Int64(128)
    for chunk in cutlass.range_constexpr(8):
        ml_L = (tid + chunk * 128) * 8
        ml_vec_{operand} = (X.iterator.raw_ptr() + ml_L).load(count=8, alignment=16)
{textwrap.indent(snippet, '        ')}
        (C.iterator.raw_ptr() + ml_L).store(ml_out_{operand}, alignment=16)

@cute.jit
def entry(X: cute.Tensor, C: cute.Tensor, batch: cutlass.Int64, ktile: cutlass.Int64):
    kernel(X, C, batch, ktile).launch(grid=(1, 1, 1), block=(128, 1, 1))

fake = cute.runtime.make_fake_tensor({dt}, (8192,), stride=(1,), assumed_align=16)
compiled = cute.compile(entry, fake, fake, cutlass.Int64(0), cutlass.Int64(0), options="--enable-tvm-ffi")
"""
    path = tmp_path / "operand_index_fragment.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("operand_index_fragment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    buffers = _buffers(tensors, "cuda")
    raw = _view(buffers[side], views[side])
    reference = _reference(buffers[side], views[side], axis)
    if side == "b":
        raw, reference = raw.transpose(-1, -2), reference.transpose(-1, -2)

    def pack_tile(value, batch, ktile):
        tile = torch.zeros(128, 64, device="cuda", dtype=value.dtype)
        start = ktile * 64
        count = min(64, 136 - start)
        tile[:, :count] = value[batch, :, start : start + count]
        return tile.reshape(2, 64, 64).permute(0, 2, 1).contiguous().reshape(-1) if major else tile.reshape(-1)

    for batch, ktile in ((0, 0), (1, 1), (1, 2)):
        data, expected = pack_tile(raw, batch, ktile), pack_tile(reference, batch, ktile)
        output = torch.full_like(data, float("nan"))
        module.compiled(data, output, batch, ktile)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)


@requires_sm100
@pytest.mark.parametrize(
    "side,mode,swapped,cta_group,split,axis",
    [
        pytest.param(s, mode, swap, 1, 1, 1, marks=pytest.mark.L0 if mode == "reshape" and not swap else pytest.mark.L1)
        for s, mode, swap in product(("a", "b", "ab"), ("reshape", "transpose_reshape", "reshape_transpose", "consecutive"), (False, True))
    ]
    + [
        pytest.param("ab", "reshape", swap, cta, split, axis, marks=pytest.mark.L1)
        for swap, cta, split, axis in [(False, 2, 1, 1), (True, 2, 1, 1), (False, 1, 2, 1), (False, 1, 1, 0)]
    ],
)
def test_frost_operand_index_gpu(side, mode, swapped, cta_group, split, axis):
    """Full kernels: changed-input capture, fresh bindings, tails and repeatability."""
    dtype = FP16 if swapped else BF16
    g, tensors, views = _graph(side, mode, dtype=dtype, spatial=136, axis=axis)
    plan = compiler.jit_from_cudnn_graph(g, _config(swapped, cta_group, split))
    engine = _FrostGemmPlan(plan)
    stream = torch.cuda.Stream()
    for fill in (1, 3):
        buffers = _buffers(tensors, "cuda", fill)
        pack = _pack(g, tensors, buffers)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            engine.execute(g, pack, ExecutionContext(stream=stream.cuda_stream))
        stream.synchronize()
        captured = torch.cuda.CUDAGraph()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message="The CUDA Graph is empty.*")
                with torch.cuda.graph(captured, stream=stream):
                    engine.execute(g, pack, ExecutionContext(stream=torch.cuda.current_stream().cuda_stream))
            for shift in (2, 4):
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    buffers["a"].fill_(fill + shift)
                    buffers["y"].fill_(float("nan"))
                    captured.replay()
                stream.synchronize()
                operands = {key: _reference(buffers[key], views[key], axis) if key in side else _view(buffers[key], views[key]) for key in "ab"}
                expected = (operands["a"].float() @ operands["b"].float()).to(buffers["y"].dtype)
                torch.testing.assert_close(buffers["y"], expected, atol=0, rtol=0)
                previous = buffers["y"].clone()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    buffers["y"].fill_(float("nan"))
                    captured.replay()
                stream.synchronize()
                assert torch.equal(buffers["y"], previous)
        finally:
            stream.synchronize()
            captured.reset()


@pytest.mark.L0
@pytest.mark.parametrize("cta_group,split,swapped", [(1, 2, False), (2, 1, False), (2, 1, True)])
def test_frost_operand_index_scheduler_source(cta_group, split, swapped):
    """Coordinate-bearing plans retain pair and split-K scheduler projections."""
    from cudnn.gemm.frost.kernel_registry import TEMPLATES

    g, _, _ = _graph("ab", spatial=136)
    chain = swap_ab(analyze(g)) if swapped else analyze(g)
    config = _config(swapped, cta_group, split)
    template = next(t for t in TEMPLATES if t.file == "sm100_matmul_mainloop.py")
    assert template._extra_reject(chain, config) is None
    snippets = epilogue_codegen.generate(chain, vec_bytes_epi=16, output_elem_bytes=2, tma_slots=compiler._tma_slots_for(chain, config))
    source = compiler._render_template(chain, snippets, config)
    ast.parse(source)
    assert f"cta_group = {cta_group}" in source and f"split_k_slices = {split}" in source
    bad = replace(chain, mainloop_a_load_dtype="fp32")
    assert "no operand load cast" in template._extra_reject(bad, config)


@pytest.mark.L0
@pytest.mark.parametrize("problem", ["other_carrier", "observable", "stride_gap"])
def test_frost_operand_index_rejects_unsupported_graphs(problem):
    """Logical coordinates never override storage proofs or observable outputs."""
    g, tensors, _ = _graph()
    if problem == "observable":
        next(n for n in g.nodes if n.name == "a_index").outputs["OUT_0"].set_output(True)
    elif problem == "stride_gap":
        tensors["a"].set_stride([18000, 9000, 140, 1])
    else:
        index_node = next(n for n in g.nodes if n.name == "a_index")
        other = g.tensor(name="other", dim=tensors["a"].get_dim(), stride=tensors["a"].get_stride(), data_type=BF16)
        index_node.inputs[next(iter(index_node.inputs))] = other
    reason = {"other_carrier": "same storage carrier", "observable": "materialized intermediate", "stride_gap": "stride gap"}[problem]
    with pytest.raises(ValueError, match=reason):
        analyze(g)


@pytest.mark.L0
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_frost_operand_index_unit_axes_and_batch_broadcast(axis):
    """Prepared unit matrix axes remain live; broadcast batch coordinates are zero."""
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=FP32, compute_data_type=FP32)
    a = g.tensor(name="A", dim=[1, 1, 136], stride=[136, 136, 1])
    b = g.tensor(name="B", dim=[2, 136, 128], stride=[17408, 128, 1])
    value = g.gen_index(input=a, axis=axis).set_data_type(BF16)
    g.matmul(A=value, B=b).set_output(True)
    expr = analyze(g).mainloop_a_ops[0].index_expr
    assert expr.evaluate({Axis.BATCH: 1, Axis.M: 17, Axis.K: 135}) == (0, 17, 135)[axis]
