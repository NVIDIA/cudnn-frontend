# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logical reshape view geometry, graph projection and unchanged-storage binding."""

import ast
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
from cudnn.gemm.frost.layouts import OperandView, ViewStep, reshape_layout
from cudnn.gemm.frost.sm100 import compiler, epilogue_codegen
from cudnn.gemm.frost.tile_config import by_name
from gemm_test_utils import requires_sm100

BF16, FP32 = cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT


def _config(swapped=False):
    return replace(by_name("CONFIG_sm100_128x128x64_128x128x32_cluster1x1_1ctamma"), swap_ab=swapped)


def _graph(side="a", position="before", mode=None, transpose_b=False, dtype=BF16):
    """Join two row axes on A or two contraction axes on B without moving storage."""
    graph = cudnn.pygraph(io_data_type=dtype, intermediate_data_type=FP32, compute_data_type=FP32)
    tensors, operands = {}, {}
    for key, dim in (("a", (2, 128, 64)), ("b", (2, 64, 128))):
        reshape = key in side
        source = (2, 2, 64, 64) if key == "a" else (2, 4, 16, 128)
        if transpose_b and key == "b":
            source, dim = (2, 2, 64, 64), (2, 128, 64)
            reshape = True
        shape = source if reshape else dim
        stride = [prod(shape[i + 1 :]) for i in range(len(shape))]
        value = tensors[key] = graph.tensor(name=key, dim=list(shape), stride=stride, data_type=dtype)
        if position == "before":
            value = graph.leaky_relu(input=value, negative_slope=0.125).set_data_type(dtype)
        if reshape:
            attrs = {} if mode is None else {"reshape_mode": mode}
            value = graph.reshape(input=value, **attrs).set_dim(list(dim)).set_stride([dim[1] * dim[2], dim[2], 1])
        if transpose_b and key == "b":
            value = graph.transpose(input=value, permutation=[0, 2, 1])
        if position == "after":
            value = graph.leaky_relu(input=value, negative_slope=0.125).set_data_type(dtype)
        operands[key] = value
    tensors["y"] = graph.matmul(A=operands["a"], B=operands["b"]).set_output(True)
    graph.validate()
    return graph, tensors


def _mock(graph, monkeypatch, swapped=False):
    """Exercise the real recipe and lowered launch frame without a full MMA kernel."""
    chain, binding = analyze_with_binding(graph)
    if swapped:
        chain, binding = swap_ab(chain), swap_ab_binding(binding)
    frames = []
    monkeypatch.setattr(compiler, "_check_plan_device", lambda device: None)
    plan = compiler.CompiledFusedGemm(
        chain, _config(swapped), [], Path("unused"), lambda *args, **kwargs: frames.append(args), binding=binding, vec_bytes_epi=16
    )
    return plan, frames


def _buffers(tensors, device="cpu"):
    return {
        key: (
            torch.randn(tuple(t.get_dim()), device=device, dtype=torch.float16 if t.get_data_type() == cudnn.data_type.HALF else torch.bfloat16)
            if key != "y"
            else torch.full(
                tuple(t.get_dim()), float("nan"), device=device, dtype=torch.float16 if t.get_data_type() == cudnn.data_type.HALF else torch.bfloat16
            )
        )
        for key, t in tensors.items()
    }


def _pack(graph, tensors, buffers):
    """Normalize explicit live graph coordinates, not ambiguous direct-plan axes."""
    return graph._normalize(
        {t.get_uid(): buffers[key] for key, t in tensors.items()},
        None,
        [t.get_uid() for t in tensors.values()],
        [list(buffers[key].shape) for key in tensors],
        [list(buffers[key].stride()) for key in tensors],
    )


@pytest.mark.L0
@pytest.mark.parametrize("position", ["plain", "before", "after"])
def test_frost_gen_index_operand_view_coordinates(position):
    """Indices before/after fixed reshape retain their original logical axes."""
    graph = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=FP32, compute_data_type=FP32)
    shape = (2, 128, 64) if position == "plain" else (2, 2, 64, 64)
    a = graph.tensor(name="A", dim=list(shape), stride=[prod(shape[i + 1 :]) for i in range(len(shape))])
    value = graph.gen_index(input=a, axis=1).set_data_type(BF16) if position == "before" else a
    if position != "plain":
        value = graph.reshape(input=value).set_dim([2, 128, 64]).set_stride([8192, 64, 1])
    if position != "before":
        value = graph.gen_index(input=value, axis=1).set_data_type(BF16)
    b = graph.tensor(name="B", dim=[2, 64, 128], stride=[8192, 128, 1])
    graph.matmul(A=value, B=b).set_output(True)
    graph.validate()
    chain = analyze(graph)
    expr = chain.mainloop_a_ops[0].index_expr
    assert expr.evaluate({Axis.BATCH: 1, Axis.M: 77, Axis.K: 17}) == (77 if position in ("plain", "after") else 77 // 64)


@pytest.mark.L0
@pytest.mark.parametrize(
    "shape,stride,target",
    [
        ((2, 3, 4), (12, 4, 1), (2, 12)),
        ((2, 3, 4), (15, 5, 1), (2, 3, 2, 2)),
        ((2, 1, 3), (3, 900, 1), (6,)),
        ((2, 3, 4), (1, 8, 2), (2, 12)),
        ((1,), (900,), (1, 1, 1)),
    ],
)
def test_frost_reshape_address_map(shape, stride, target):
    """Check exact element addresses against the independent Torch zero-copy view."""
    span = 1 + sum((d - 1) * s for d, s in zip(shape, stride))
    storage = torch.arange(span)
    value = storage.as_strided(shape, stride)
    view = value.view(target)
    dim, st = reshape_layout(shape, stride, target)
    got = storage.as_strided(dim, st)
    assert torch.equal(got, view) and view.data_ptr() == value.data_ptr()


@pytest.mark.L0
@pytest.mark.parametrize(
    "shape,stride,target", [((2, 3, 4), (15, 5, 1), (2, 12)), ((2, 3), (1, 2), (6,)), ((2, 3), (3, 1), (7,)), ((2**32, 2**32), (2**32, 1), (2**64,))]
)
def test_frost_reshape_invalid_address_map(shape, stride, target):
    with pytest.raises(ValueError, match="copying|element count|64-bit"):
        reshape_layout(shape, stride, target)


@pytest.mark.L0
@pytest.mark.parametrize("side,position,swapped", list(product(("a", "b", "ab"), ("before", "after"), (False, True))))
def test_frost_reshape_projection(side, position, swapped, monkeypatch):
    graph, tensors = _graph(side, position)
    before = [
        (dict(n.params), [(tuple(t.get_dim()), tuple(t.get_stride()), t.get_data_type()) for t in (*n.inputs.values(), *n.outputs.values())])
        for n in graph._nodes
    ]
    plan, _ = _mock(graph, monkeypatch, swapped)
    assert (plan.chain.matmul.M, plan.chain.matmul.N, plan.chain.matmul.K) == (128, 128, 64)
    assert len(plan.recipe.operand_reshapes) == len(side)
    assert [op.op for op in plan.chain.mainloop_a_ops] == ["leaky_relu"]
    assert [op.op for op in plan.chain.mainloop_b_ops] == ["leaky_relu"]
    snippets = epilogue_codegen.generate(plan.chain, vec_bytes_epi=16, output_elem_bytes=2, tma_slots=compiler._tma_slots_for(plan.chain, _config(swapped)))
    ast.parse(compiler._render_template(plan.chain, snippets, _config(swapped)))
    after = [
        (dict(n.params), [(tuple(t.get_dim()), tuple(t.get_stride()), t.get_data_type()) for t in (*n.inputs.values(), *n.outputs.values())])
        for n in graph._nodes
    ]
    assert before == after


@pytest.mark.L0
@pytest.mark.parametrize("swapped,transpose_b", list(product((False, True), (False, True))))
def test_frost_reshape_original_storage(monkeypatch, swapped, transpose_b):
    graph, tensors = _graph("ab", transpose_b=transpose_b)
    plan, frames = _mock(graph, monkeypatch, swapped)
    buffers = _buffers(tensors)
    plan({t: buffers[key] for key, t in tensors.items()})
    frame = frames[-1]
    for index, key in enumerate(("b", "a") if swapped else ("a", "b"), 1):
        assert frame[index].data_ptr() == buffers[key].data_ptr()
        assert tuple(frame[index].shape) == (128, 64, 2)
    for key in "ab":
        bad = buffers[key].flatten(1)
        with pytest.raises(ValueError, match="prepared source shape"):
            plan({t: (bad if name == key else buffers[name]) for name, t in tensors.items()})
    assert len(frames) == 1
    plan({t: buffers[key] for key, t in tensors.items()})
    assert len(frames) == 2


@pytest.mark.L0
@pytest.mark.parametrize("problem", ["count", "dtype", "stride", "output", "epilogue"])
def test_frost_reshape_rejection(problem):
    graph, tensors = _graph()
    node = next(n for n in graph._nodes if n.node_type.name == "RESHAPE")
    if problem == "count":
        node.outputs["OUT_0"].set_dim([2, 127, 64])
    elif problem == "dtype":
        node.outputs["OUT_0"].set_data_type(FP32)
    elif problem == "stride":
        node.outputs["OUT_0"].set_stride([8192, 1, 128])
    elif problem == "output":
        node.outputs["OUT_0"].set_output(True)
    else:
        graph.reshape(input=tensors["y"]).set_dim([2, 64, 256])
    with pytest.raises(ValueError):
        analyze(graph)


@requires_sm100
@pytest.mark.parametrize(
    "side,swapped,transpose_b",
    [
        pytest.param("a", False, False, marks=pytest.mark.L0),
        pytest.param("b", True, False, marks=pytest.mark.L0),
        pytest.param("ab", False, True, marks=pytest.mark.L0),
        pytest.param("ab", True, True, marks=pytest.mark.L1),
    ],
)
def test_frost_reshape_gpu(side, swapped, transpose_b):
    """Check actual kernels, fresh source storage and poisoned changed-input replay."""
    dtype = cudnn.data_type.HALF if swapped and transpose_b else BF16
    tdtype = torch.float16 if dtype == cudnn.data_type.HALF else torch.bfloat16
    graph, tensors = _graph(side, transpose_b=transpose_b, dtype=dtype)
    plan = compiler.jit_from_cudnn_graph(graph, _config(swapped))
    engine = _FrostGemmPlan(plan)
    stream = torch.cuda.Stream()
    for _ in range(2):
        buffers = _buffers(tensors, "cuda")
        pack = _pack(graph, tensors, buffers)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            engine.execute(graph, pack, ExecutionContext(stream=stream.cuda_stream))
        stream.synchronize()
        captured = torch.cuda.CUDAGraph()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message="The CUDA Graph is empty.*")
                with torch.cuda.graph(captured, stream=stream):
                    engine.execute(graph, pack, ExecutionContext(stream=torch.cuda.current_stream().cuda_stream))
            for _ in range(2):
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    buffers["a"].neg_()
                    buffers["y"].fill_(float("nan"))
                    captured.replay()
                stream.synchronize()
                a = torch.nn.functional.leaky_relu(buffers["a"].float(), 0.125).to(tdtype).view(2, 128, 64)
                b = torch.nn.functional.leaky_relu(buffers["b"].float(), 0.125).to(tdtype)
                b = b.view(2, 128, 64).transpose(1, 2) if transpose_b else b.view(2, 64, 128)
                expected = (a.float() @ b.float()).to(tdtype)
                torch.testing.assert_close(buffers["y"], expected, atol=0.25, rtol=0.02)
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
@pytest.mark.parametrize("swapped", [False, True])
def test_frost_reshape_native_transport(monkeypatch, swapped):
    """Preserve original storage through the graph-coordinate native engine route."""
    graph, tensors = _graph("ab", transpose_b=True)
    plan, frames = _mock(graph, monkeypatch, swapped)
    engine = _FrostGemmPlan(plan)
    for _ in range(2):
        buffers = _buffers(tensors)
        engine.execute(graph, _pack(graph, tensors, buffers), ExecutionContext(stream=0))
        for index, key in enumerate(("b", "a") if swapped else ("a", "b"), 1):
            assert frames[-1][index].data_ptr() == buffers[key].data_ptr()
            assert tuple(frames[-1][index].shape) == (128, 64, 2)
    assert len(frames) == 2


@pytest.mark.L0
@pytest.mark.parametrize("target,stride", [((2, 3, 2, 2), (15, 5, 2, 1)), ((2, 12), (24, 1)), ((2**32, 2**32), (2**32, 1)), ((24,), (-1,))])
def test_frost_reshape_native_span(target, stride):
    """Checked native metadata views cannot exceed observed or effective storage."""
    native = cudnn._pybind_module.VariantPackNative(1)
    native.set_operand(0, 4096, (2, 3, 4), (15, 5, 1), 4, 16, 1, 58, 2, 0)
    source = native.operand(0, 0)
    if target == (2, 3, 2, 2):
        view = source.metadata_view(target, stride)
        assert view.data_ptr() == source.data_ptr() == 4096
        assert tuple(view.shape) == target and tuple(view.stride()) == stride
        assert view.dtype == source.dtype
    else:
        with pytest.raises(ValueError, match="geometry|storage|span"):
            source.metadata_view(target, stride)


@pytest.mark.L0
@pytest.mark.parametrize("mode", [cudnn.reshape_mode.LOGICAL, cudnn.reshape_mode.VIEW_ONLY])
def test_frost_reshape_modes(mode):
    """Both modes are served only when their declared layout is the logical view."""
    graph, _ = _graph(mode=mode)
    assert analyze(graph).matmul.M == 128


@pytest.mark.L0
@pytest.mark.parametrize(
    "source,target", [((128, 64), (1, 128, 64)), ((8192,), (1, 128, 64)), ((1, 2, 1, 64, 64), (1, 128, 64)), ((1, 128, 64), (1, 2, 64, 64))]
)
def test_frost_reshape_rank_independent(source, target, monkeypatch):
    """Use the same projection algorithm for vector/matrix and rank-5 sources."""
    graph = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=FP32, compute_data_type=FP32)
    a = graph.tensor(name="A", dim=list(source), stride=[prod(source[i + 1 :]) for i in range(len(source))])
    value = graph.relu(input=a).set_data_type(BF16)
    value = graph.reshape(input=value).set_dim(list(target)).set_stride([prod(target[i + 1 :]) for i in range(len(target))])
    if len(target) == 4:
        value = graph.reshape(input=value).set_dim([1, 128, 64]).set_stride([8192, 64, 1])
    b = graph.tensor(name="B", dim=[1, 64, 128], stride=[8192, 128, 1])
    graph.matmul(A=value, B=b).set_output(True)
    graph.validate()
    plan, _ = _mock(graph, monkeypatch)
    assert (plan.chain.matmul.M, plan.chain.matmul.N, plan.chain.matmul.K) == (128, 128, 64)
    assert plan.recipe.operand_reshapes[0][1].source_shape == source


@pytest.mark.L0
def test_frost_reshape_dynamic_unreshaped_operand(monkeypatch):
    """Pin reshape sources but retain dynamic N binding and workspace inference."""
    graph, tensors = _graph("a")
    plan, frames = _mock(graph, monkeypatch)
    buffers = _buffers(tensors)
    buffers["b"] = torch.randn(2, 64, 192, dtype=torch.bfloat16)
    buffers["y"] = torch.empty(2, 128, 192, dtype=torch.bfloat16)
    _FrostGemmPlan(plan).execute(graph, _pack(graph, tensors, buffers), ExecutionContext(stream=0))
    assert frames[-1][0][:3] == (128, 192, 64)
    assert plan.recipe.operand_shape(0, tensors["a"].get_dim()) == (2, 128, 64)


@pytest.mark.L0
@pytest.mark.parametrize("compatible", [False, True])
def test_frost_reshape_shared_storage(compatible):
    """Reuse identical prepared views and reject incompatible observable coordinates."""
    graph = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=FP32, compute_data_type=FP32)
    a = graph.tensor(name="A", dim=[2, 2, 64, 128], stride=[16384, 8192, 128, 1])
    value = graph.relu(input=a).set_data_type(BF16)
    value = graph.reshape(input=value).set_dim([2, 128, 128]).set_stride([16384, 128, 1])
    b = value if compatible else graph.transpose(input=value, permutation=[0, 2, 1])
    graph.matmul(A=value, B=b).set_output(True)
    graph.validate()
    if compatible:
        _, binding = analyze_with_binding(graph)
        assert len(binding.operand_views) == 1 and binding.a_operands == binding.b_operands == [a]
    else:
        with pytest.raises(ValueError, match="incompatible operand views"):
            analyze(graph)


@pytest.mark.L0
def test_frost_reshape_scalar_operand_order():
    """Reshape projection does not swap a noncommutative scalar pointwise input."""
    graph = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=FP32, compute_data_type=FP32)
    a = graph.tensor(name="A", dim=[1, 2, 64, 64], stride=[8192, 4096, 64, 1])
    scalar = graph.tensor(name="scalar", dim=[1], stride=[1])
    value = graph.sub(a=scalar, b=a).set_data_type(BF16)
    value = graph.reshape(input=value).set_dim([1, 128, 64]).set_stride([8192, 64, 1])
    b = graph.tensor(name="B", dim=[1, 64, 128], stride=[8192, 128, 1])
    graph.matmul(A=value, B=b).set_output(True)
    graph.validate()
    chain = analyze(graph)
    assert chain.mainloop_a_ops[0].op == "sub" and not chain.mainloop_a_ops[0].aux_on_rhs


@pytest.mark.L0
def test_frost_reshape_chunk_oracle():
    """Compare accepted/rejected views and exact addresses across stride chunks."""
    for shape, stride in [((2, 3, 4), (12, 4, 1)), ((2, 3, 4), (15, 5, 1)), ((2, 3, 4), (1, 8, 2)), ((2, 1, 3, 4), (12, 999, 4, 1))]:
        span = 1 + sum((d - 1) * s for d, s in zip(shape, stride))
        storage = torch.arange(span)
        value = storage.as_strided(shape, stride)
        for target in [(24,), (1, 24), (2, 12), (3, 8), (4, 6), (6, 4), (2, 3, 4), (2, 3, 2, 2), (2, 1, 12)]:
            try:
                expected = value.view(target)
            except RuntimeError:
                with pytest.raises(ValueError, match="copying"):
                    reshape_layout(shape, stride, target)
            else:
                dim, st = reshape_layout(shape, stride, target)
                assert torch.equal(storage.as_strided(dim, st), expected)


@pytest.mark.L0
def test_frost_reshape_fp16_rounding_contract():
    """Projection preserves FP16 node rounding and scalar activation attributes."""
    graph, _ = _graph("ab", dtype=cudnn.data_type.HALF)
    chain = analyze(graph)
    for op in (*chain.mainloop_a_ops, *chain.mainloop_b_ops):
        assert op.out_dtype == "fp16" and op.compute_dtype == "fp32"
        assert dict(op.attrs) == {"negative_slope": 0.125}


@pytest.mark.L0
def test_frost_reshape_after_transpose(monkeypatch):
    """Compose transpose then two rank-changing views in their original order."""
    graph = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=FP32, compute_data_type=FP32)
    a = graph.tensor(name="A", dim=[2, 64, 128], stride=[8192, 1, 64])
    value = graph.transpose(input=a, permutation=[0, 2, 1])
    value = graph.reshape(input=value).set_dim([2, 2, 64, 64]).set_stride([8192, 4096, 64, 1])
    value = graph.leaky_relu(input=value, negative_slope=0.125).set_data_type(BF16)
    value = graph.reshape(input=value).set_dim([2, 128, 64]).set_stride([8192, 64, 1])
    b = graph.tensor(name="B", dim=[2, 64, 128], stride=[8192, 128, 1])
    y = graph.matmul(A=value, B=b).set_output(True)
    graph.validate()
    plan, frames = _mock(graph, monkeypatch)
    av = torch.randn(2, 128, 64, dtype=torch.bfloat16).transpose(1, 2)
    bv = torch.randn(2, 64, 128, dtype=torch.bfloat16)
    yv = torch.empty(2, 128, 128, dtype=torch.bfloat16)
    tensors, buffers = {"a": a, "b": b, "y": y}, {"a": av, "b": bv, "y": yv}
    _FrostGemmPlan(plan).execute(graph, _pack(graph, tensors, buffers), ExecutionContext(stream=0))
    argument = frames[-1][1]
    assert argument.data_ptr() == av.data_ptr()
    got = av.as_strided(tuple(argument.shape), tuple(argument.stride())).permute(2, 0, 1)
    assert torch.equal(got, av.transpose(1, 2))
    assert [op.op for op in plan.chain.mainloop_a_ops] == ["leaky_relu"]


@pytest.mark.L0
def test_frost_reshape_native_byte_width():
    """Element geometry fitting Int64 must also fit when multiplied by dtype width."""
    native = cudnn._pybind_module.VariantPackNative(1)
    native.set_operand(0, 4096, (1, 1 << 62), (1 << 62, 1), 4, 16, 1, -1, 2, 0)
    source = native.operand(0, 0)
    with pytest.raises(ValueError, match="64-bit byte span"):
        source.metadata_view((1, 1 << 62), (1 << 62, 1))


def _view_chain_graph(count):
    """Build equal-shaped math with identity, one view or a long view composition."""
    graph = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=FP32, compute_data_type=FP32)
    source = (1, 2, 64, 128) if count else (1, 128, 128)
    a = graph.tensor(name="A", dim=list(source), stride=[prod(source[i + 1 :]) for i in range(len(source))])
    value = graph.leaky_relu(input=a, negative_slope=0.125).set_data_type(BF16)
    for i in range(count):
        target = (1, 128, 128) if i % 2 == 0 else source
        value = graph.reshape(input=value).set_dim(list(target)).set_stride([prod(target[i + 1 :]) for i in range(len(target))])
    b = graph.tensor(name="B", dim=[1, 128, 128], stride=[16384, 128, 1])
    y = graph.matmul(A=value, B=b).set_output(True)
    graph.validate()
    return graph, {"a": a, "b": b, "y": y}


def _native_operands(plan):
    """Synthetic addresses for host-only metadata tests; no consumer dereferences."""
    native = cudnn._pybind_module.VariantPackNative(len(plan.binding.bound_tensors()))
    for i, tensor in enumerate(plan.binding.bound_tensors()):
        native.set_operand(i, 4096 * (i + 1), tensor.get_dim(), tensor.get_stride(), 4, 16, 1, -1, 2, 0)
    return [native.operand(i, 0) for i in range(len(plan.binding.bound_tensors()))]


@pytest.mark.L0
@pytest.mark.parametrize(
    "steps",
    [
        (ViewStep("reshape", (2, 12)),),
        (ViewStep("reshape", (2, 3, 2, 2)), ViewStep("reshape", (2, 12))),
        (ViewStep("transpose", (0, 2, 1)), ViewStep("transpose", (0, 2, 1)), ViewStep("reshape", (2, 12))),
        (ViewStep("transpose", (2, 0, 1)), ViewStep("transpose", (1, 2, 0)), ViewStep("reshape", (2, 12))),
    ],
)
def test_frost_view_composition_equivalence(steps):
    """The geometry layer supports arbitrary permutations, unlike the GEMM gate."""
    value = torch.arange(24).view(2, 3, 4)
    expected = value
    for step in steps:
        expected = expected.view(step.argument) if step.kind == "reshape" else expected.permute(step.argument)
    view = OperandView.prepare(value.shape, value.stride(), steps)
    assert torch.equal(value.as_strided(view.target_shape, view.target_stride), expected)
    assert view.target_shape == (2, 12) and view.target_stride == (12, 1)


@pytest.mark.L0
@pytest.mark.parametrize(
    "kind,argument,declared",
    [
        ("unknown", (2, 3), ()),
        ("transpose", (0, 0, 2), ()),
        ("transpose", (0, 2), ()),
        ("transpose", (0, 1, 3), ()),
        ("transpose", (0, 1, 2), (12, 4, 1)),
        ("reshape", (2, 12), (12,)),
        ("reshape", (2, 12), (12, 1, 1)),
        ("reshape", (2, 12), (-12, 1)),
        ("reshape", (2, 12), (12, 2)),
        ("reshape", (0, 12), ()),
        ("reshape", (-1, 12), ()),
    ],
)
def test_frost_view_invalid_geometry(kind, argument, declared):
    with pytest.raises(ValueError, match="unsupported|permutation|strides|extents"):
        OperandView.prepare((2, 3, 4), (12, 4, 1), (ViewStep(kind, argument, declared),))


@pytest.mark.L0
def test_frost_view_composition_proves_intermediate_steps():
    """A valid final shape must not conceal an illegal intermediate copy."""
    steps = (ViewStep("transpose", (1, 0)), ViewStep("reshape", (6,)), ViewStep("reshape", (2, 3)))
    with pytest.raises(ValueError, match="copying"):
        OperandView.prepare((2, 3), (3, 1), steps)


@pytest.mark.L0
def test_frost_view_runtime_stride_rebinding():
    """Re-prove changed runtime strides; never reuse cached layouts by shape alone."""
    shape = (1, 2, 1, 64, 64)
    stride = (8192, 4096, 4096, 64, 1)
    view = OperandView.prepare(shape, stride, (ViewStep("reshape", (1, 128, 64), (8192, 64, 1)),), fold_batch=True)
    assert view.layout(shape, stride) == (view.target_shape, view.target_stride)
    value = torch.arange(8192).view(shape).as_strided(shape, (99, 4096, 777, 64, 1))
    target, st = view.layout(value.shape, value.stride())
    assert torch.equal(value.as_strided(target, st), value.view(target))
    with pytest.raises(ValueError, match="copying|declared strides"):
        view.layout(shape, (8192, 4097, 4096, 64, 1))
    with pytest.raises(ValueError, match="prepared source shape"):
        view.layout((1, 128, 64), (8192, 64, 1))


@pytest.mark.L0
@pytest.mark.parametrize("count", [0, 1, 17])
def test_frost_view_runtime_call_count(monkeypatch, count):
    """Identity is allocation-free; long native chains create one final view."""
    graph, _ = _view_chain_graph(count)
    plan, _ = _mock(graph, monkeypatch)
    ordered = _native_operands(plan)
    calls = []

    class CountedOperand:
        def __init__(self, operand):
            self.operand = operand
            self.shape = operand.shape

        def stride(self):
            return self.operand.stride()

        def metadata_view(self, shape, stride):
            calls.append((shape, stride))
            return self.operand.metadata_view(shape, stride)

    ordered[0] = CountedOperand(ordered[0])
    result = plan.recipe.operand_views(ordered)
    assert len(calls) == bool(count)
    if count:
        assert tuple(result[0].shape) == (1, 128, 128)
        assert result[0].data_ptr() == 4096
        assert all(slot != 0 for slot, _ in plan.recipe.matrix_views)
        view = plan.recipe.operand_reshapes[0][1]
        # Warm launches must not replay geometry when source shape/stride match.
        monkeypatch.setattr(ViewStep, "layout", lambda *args: pytest.fail("recomputed a prepared view"))
        assert view.layout(ordered[0].shape, ordered[0].stride()) == (view.target_shape, view.target_stride)
    else:
        assert result is ordered


@pytest.mark.L0
def test_frost_view_tensor_storage_offset(monkeypatch):
    graph, tensors = _view_chain_graph(17)
    plan, frames = _mock(graph, monkeypatch)
    buffers = _buffers(tensors)
    storage = torch.arange(prod(tensors["a"].get_dim()) + 8, dtype=torch.float32).bfloat16()
    buffers["a"] = storage[8:].view(tuple(tensors["a"].get_dim()))
    plan({t: buffers[key] for key, t in tensors.items()})
    argument = frames[-1][1]
    assert argument.data_ptr() == buffers["a"].data_ptr()
    assert argument.storage_offset() == 8
    assert torch.equal(argument.permute(2, 0, 1), buffers["a"].view(1, 128, 128))


@pytest.mark.L1
def test_frost_view_host_cost_report(monkeypatch):
    """Report bounded CPU view cost; timing rank is not a flaky CI assertion."""
    from statistics import median
    from time import perf_counter_ns

    cases = {}
    for count in (0, 1, 17):
        graph, _ = _view_chain_graph(count)
        plan, _ = _mock(graph, monkeypatch)
        ordered = _native_operands(plan)
        for _ in range(100):
            plan.recipe.operand_views(ordered)
        cases[count] = plan.recipe, ordered
    samples = {count: [] for count in cases}
    for sample in range(7):
        for count in ((0, 1, 17) if sample % 2 == 0 else (17, 1, 0)):
            recipe, ordered = cases[count]
            start = perf_counter_ns()
            for _ in range(200):
                result = recipe.operand_views(ordered)
            samples[count].append((perf_counter_ns() - start) / 200 / 1000)
            assert result[0].data_ptr() == ordered[0].data_ptr()
            assert tuple(result[0].shape) == (1, 128, 128)
    print("CPU metadata view medians (us), no kernel/device timing:", {count: median(cost) for count, cost in samples.items()})


@pytest.mark.L0
def test_frost_view_shared_equivalent_paths(monkeypatch):
    """Different view spellings share storage without discarding path constraints."""
    graph = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=FP32, compute_data_type=FP32)
    a = graph.tensor(name="A", dim=[1, 2, 64, 128], stride=[16384, 8192, 128, 1])
    value = graph.relu(input=a).set_data_type(BF16)
    left = graph.reshape(input=value).set_dim([1, 128, 128]).set_stride([16384, 128, 1])
    right = graph.reshape(input=value).set_dim([1, 1, 128, 128]).set_stride([16384, 16384, 128, 1])
    right = graph.reshape(input=right).set_dim([1, 128, 128]).set_stride([16384, 128, 1])
    graph.matmul(A=left, B=right).set_output(True)
    graph.validate()
    plan, _ = _mock(graph, monkeypatch)
    assert plan.binding.a_operands == plan.binding.b_operands == [a]
    assert len(plan.recipe.operand_reshapes) == 1
    assert len(plan.recipe.operand_reshapes[0][1].alternate_steps) == 1

    shape, stride = (2, 3, 4), (12, 4, 1)
    direct = OperandView.prepare(shape, stride, (ViewStep("reshape", shape),))
    flattened = OperandView.prepare(shape, stride, (ViewStep("reshape", (24,)), ViewStep("reshape", shape)))
    merged = direct.merged(flattened)
    assert direct.layout(shape, (15, 5, 1)) == (shape, (15, 5, 1))
    with pytest.raises(ValueError, match="copying"):
        merged.layout(shape, (15, 5, 1))


@pytest.mark.L0
@pytest.mark.parametrize("position", ["before", "after"])
def test_frost_view_padded_pointwise_layout(monkeypatch, position):
    """Virtual pointwise output inherits the projected input's padded row map."""
    graph, tensors = _graph("a", position=position)
    tensors["a"].set_stride([10240, 5120, 80, 1])
    reshape = next(node for node in graph._nodes if node.node_type.name == "RESHAPE")
    reshape.outputs["OUT_0"].set_stride([10240, 80, 1])
    graph.validate()
    plan, frames = _mock(graph, monkeypatch)
    buffers = _buffers(tensors)
    buffers["a"] = torch.empty_strided((2, 2, 64, 64), (10240, 5120, 80, 1), dtype=torch.bfloat16).normal_()
    plan({t: buffers[key] for key, t in tensors.items()})
    argument = frames[-1][1]
    assert argument.data_ptr() == buffers["a"].data_ptr()
    assert tuple(argument.stride()) == (80, 1, 10240)
    assert torch.equal(argument.permute(2, 0, 1), buffers["a"].view(2, 128, 64))


@pytest.mark.L0
@pytest.mark.parametrize("swapped", [False, True])
def test_frost_view_rank_n_folded_transport(monkeypatch, swapped):
    """Join rows and fold rank-N batches in the same final native operand view."""
    graph = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=FP32, compute_data_type=FP32)
    source = (2, 3, 2, 64, 64)
    a = graph.tensor(name="A", dim=list(source), stride=[prod(source[i + 1 :]) for i in range(len(source))])
    value = graph.reshape(input=graph.relu(input=a).set_data_type(BF16)).set_dim([2, 3, 128, 64]).set_stride([24576, 8192, 64, 1])
    b = graph.tensor(name="B", dim=[2, 3, 64, 128], stride=[24576, 8192, 128, 1])
    y = graph.matmul(A=value, B=b).set_output(True)
    graph.validate()
    plan, frames = _mock(graph, monkeypatch, swapped)
    tensors = {"a": a, "b": b, "y": y}
    for n in (128, 192):
        buffers = {
            "a": torch.randn(source, dtype=torch.bfloat16),
            "b": torch.randn(2, 3, 64, n, dtype=torch.bfloat16),
            "y": torch.empty(2, 3, 128, n, dtype=torch.bfloat16),
        }
        _FrostGemmPlan(plan).execute(graph, _pack(graph, tensors, buffers), ExecutionContext(stream=0))
        slot, view = plan.recipe.operand_reshapes[0]
        assert view.target_shape == (6, 128, 64)
        assert all(index != slot for index, _ in plan.recipe.matrix_views)
        argument = frames[-1][2 if swapped else 1]
        assert argument.data_ptr() == buffers["a"].data_ptr()
        got = buffers["a"].as_strided(tuple(argument.shape), tuple(argument.stride())).permute(2, 0, 1)
        assert torch.equal(got, buffers["a"].view(6, 128, 64))


@pytest.mark.L0
def test_frost_view_identity_composition_preserves_operands(monkeypatch):
    """A prepared identity reshape validates its contract without creating views."""
    graph, tensors = _view_chain_graph(0)
    mm = next(node for node in graph._nodes if node.node_type.name == "MATMUL")
    value = graph.reshape(input=mm.inputs["A"]).set_dim([1, 128, 128]).set_stride([16384, 128, 1])
    mm.inputs["A"] = value
    graph.validate()
    plan, _ = _mock(graph, monkeypatch)
    buffers = _buffers(tensors)
    ordered = [buffers[next(key for key, t in tensors.items() if t is tensor)] for tensor in plan.binding.bound_tensors()]
    assert plan.recipe.operand_views(ordered) is ordered
