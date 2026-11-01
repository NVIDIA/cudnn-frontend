# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata-only operand transpose normalization, binding and kernel contracts."""

import ast
import importlib.util
import textwrap
import warnings
from dataclasses import replace
from pathlib import Path
from math import prod
from itertools import product

import cudnn
import pytest
import torch

from cudnn.engines import ExecutionContext
from cudnn.gemm.frost.engine import _FrostGemmPlan
from cudnn.gemm.frost.layouts import fold_batch_layout
from cudnn.frost.buffers import DeviceView, dtype_name
from cudnn.gemm.frost.fusion_ir import Axis, swap_ab
from cudnn.gemm.frost.graph_analyzer import matrix_axis_index, analyze, analyze_with_binding, resolve_variant_pack, swap_ab_binding
from cudnn.gemm.frost.sm100 import compiler, epilogue_codegen
from cudnn.gemm.frost.tile_config import by_name
from gemm_test_utils import requires_sm100, requires_sm120

BF16, F32 = cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT


def _config(swapped=False):
    """Select the existing one-CTA configuration with optional operand swapping."""
    return replace(by_name("CONFIG_sm100_128x128x64_128x128x32_cluster1x1_1ctamma"), swap_ab=swapped)


def _strides(prefix, matrix_shape, matrix_strides, gap=0):
    """Construct affine batch pitches while preserving matrix strides and gaps."""
    pitch = max(d * s for d, s in zip(matrix_shape, matrix_strides)) + gap
    return [pitch * prod(prefix[i + 1 :]) for i in range(len(prefix))] + list(matrix_strides)


def _graph(
    M=128,
    N=192,
    K=136,
    *,
    side="ab",
    position="after",
    scaled=False,
    batch=1,
    dtype=BF16,
    prefix=None,
    broadcast=False,
    epilogue=False,
    gap=0,
    a_prefix=None,
    bias_prefix=(),
    scale_prefix=None,
    index_axis=None,
    index_source="matmul",
):
    """Build a validated logical graph with configurable operand views and epilogue."""
    g = cudnn.pygraph(io_data_type=dtype, intermediate_data_type=F32, compute_data_type=F32)
    prefix = (batch,) if prefix is None else tuple(prefix)
    permutation = [*range(len(prefix)), len(prefix) + 1, len(prefix)]
    tensors, operands = {}, {}
    for s, extent in (("a", M), ("b", N)):
        odd = s in side and position not in ("double", "consecutive")
        batches = (a_prefix if a_prefix is not None else ((1,) * len(prefix) if broadcast else prefix)) if s == "a" else prefix
        matrix = (K, extent) if (s == "a") == odd else (extent, K)
        dim = [*batches, *matrix]
        stride = _strides(batches, matrix, (1, K) if (s == "a") == odd else (K, 1), gap)
        value = tensors[s] = g.tensor(name=s.upper(), dim=dim, stride=stride)
        if s not in side:
            operands[s] = value
            continue
        if position in ("before", "double", "consecutive"):
            value = g.transpose(input=value, permutation=permutation)
        if position == "consecutive":
            value = g.transpose(input=value, permutation=permutation)
        if position == "surrounded":
            value = g.cos(input=value).set_data_type(dtype)
        if scaled:
            scale_batch = (1,) * len(prefix) if scale_prefix is None else tuple(scale_prefix)
            dim = [*scale_batch, 1, K] if (s == "a") == (position in ("before", "consecutive")) else [*scale_batch, K, 1]
            stride = _strides(scale_batch, dim[-2:], (dim[-1], 1))
            scale = tensors[f"scale_{s}"] = g.tensor(name=f"scale_{s}", dim=dim, stride=stride, data_type=F32)
            value = g.mul(a=scale, b=value).set_data_type(dtype)
        value = g.leaky_relu(input=value, negative_slope=0.125).set_data_type(dtype)
        if position not in ("before", "consecutive"):
            value = g.transpose(input=value, permutation=permutation)
        if position == "surrounded":
            value = g.relu(input=value).set_data_type(dtype)
        operands[s] = value
    y = g.matmul(A=operands["a"], B=operands["b"])
    if epilogue:
        bias = tensors["bias"] = g.tensor(name="bias", dim=[*bias_prefix, 1, N], stride=_strides(bias_prefix, (1, N), (N, 1)), data_type=dtype)
        y = g.add(a=y, b=bias).set_data_type(dtype)
    if index_axis is not None:
        source = y
        if index_source != "matmul":
            dim = [*prefix, M, N]
            dim[index_axis] = 1
            stride = [prod(dim[i + 1 :]) for i in range(len(dim))]
            source = tensors["index_input"] = g.tensor(name="index_input", dim=dim, stride=stride, data_type=dtype)
            if index_source == "aux_unary":
                source = g.relu(input=source).set_data_type(dtype)
        y = g.add(a=y, b=g.gen_index(input=source, axis=index_axis, name="index"))
    tensors["y"] = y.set_output(True)
    g.validate()
    return g, tensors


def _buffers(tensors, m=128, n=192, k=136, *, device="cpu", dtype=torch.bfloat16, gap=0):
    """Allocate caller-coordinate buffers with dynamic matrix extents and fixed batches."""
    prefix = tuple(tensors["y"].get_dim()[:-2])
    result = {"y": torch.zeros(*prefix, m, n, dtype=dtype, device=device)}
    for side, extent in (("a", m), ("b", n)):
        batches = tuple(tensors[side].get_dim()[:-2])
        transposed = tensors[side].get_stride()[-2] == 1
        matrix = (k, extent) if transposed else (extent, k)
        pitch = k + (8 if gap else 0)
        stride = _strides(batches, matrix, (1, pitch) if transposed else (pitch, 1), gap)
        result[side] = torch.empty_strided((*batches, *matrix), stride, dtype=dtype, device=device).normal_()
        key = f"scale_{side}"
        if key in tensors:
            scale = torch.linspace(-1.25, 1.25, k, dtype=torch.float32, device=device)
            batches = tuple(tensors[key].get_dim()[:-2])
            dim = (*batches, k, 1) if tensors[key].get_dim()[-2] != 1 else (*batches, 1, k)
            result[key] = scale.reshape(dim)
    if "bias" in tensors:
        result["bias"] = torch.randn(*tensors["bias"].get_dim()[:-2], 1, n, dtype=dtype, device=device)
    if "index_input" in tensors:
        dim = tensors["index_input"].get_dim()
        shape = (*dim[:-2], 1 if dim[-2] == 1 else m, 1 if dim[-1] == 1 else n)
        result["index_input"] = torch.ones(shape, device=device, dtype=dtype)
    return result


def _pack(tensors, buffers):
    """Bind graph tensors to their current runtime buffers."""
    return {tensor: buffers[key] for key, tensor in tensors.items()}


def _mock_plan(monkeypatch, g, swapped=False):
    """Record real binding frames without compiling or launching the full kernel."""
    chain, binding = analyze_with_binding(g)
    if swapped:
        chain, binding = swap_ab(chain), swap_ab_binding(binding)
    launches = []
    monkeypatch.setattr(compiler, "_check_plan_device", lambda device: None)
    plan = compiler.CompiledFusedGemm(
        chain,
        _config(swapped),
        [r.name for r in chain.aux_tensors],
        Path("unused"),
        lambda *args, **kw: launches.append(args),
        binding=binding,
        vec_bytes_epi=16,
    )
    return plan, launches


def _evaluate_frame(frame, chain):
    """Host algebra oracle over the actual views handed to the kernel (not kernel validation)."""
    aux = {ref.name: arg for ref, arg in zip(chain.aux_tensors, frame[4:])}
    values = []
    for s, arg in zip("ab", frame[1:3]):
        value = arg.permute(2, 0, 1)
        for op in getattr(chain, f"mainloop_{s}_ops"):
            value = value.float()
            if op.op == "cos":
                value = value.cos()
            elif op.op == "leaky_relu":
                value = torch.where(value >= 0, value, value * dict(op.attrs)["negative_slope"])
            elif op.op == "relu":
                value = value.relu()
            elif op.op in ("mul", "sub"):
                other = aux[op.aux].float()
                if s == "b" and other.ndim == 3:
                    other = other.transpose(1, 2)
                lhs, rhs = (value, other) if op.aux_on_rhs else (other, value)
                value = lhs * rhs if op.op == "mul" else lhs - rhs
            else:
                raise AssertionError(f"no oracle for {op.op}")
            value = value.to({"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[op.out_dtype])
        values.append(value.float())
    return values[0] @ values[1].transpose(1, 2)


def _transpose_cases(*, rebinding=False):
    """Keep representative cells in L0 and the unchanged cross-product in L1."""
    sides = ("a", "b", "ab")
    positions = ("before", "after", "surrounded", "double", "consecutive")
    prefixes = ((), (1,), (2, 3), (2, 1, 3))
    dtypes = (BF16, cudnn.data_type.HALF)
    representatives = {(sides[(p + j) % 3], positions[j], prefixes[p], dtypes[(p + j) % 2]) for p in range(4) for j in (p % 5, (p + 1) % 5, (p + 3) % 5)}
    for side, position, prefix, dtype in product(sides, positions, prefixes, dtypes):
        representative = (side, position, prefix, dtype) in representatives
        if not rebinding:
            yield pytest.param(side, position, prefix, dtype, marks=pytest.mark.L0 if representative else pytest.mark.L1)
        else:
            for swapped, scaled in product((False, True), (False, True) if dtype == BF16 else (True,)):
                selected = representative and scaled == (dtype != BF16 or positions.index(position) % 2 == 0)
                yield pytest.param(side, position, swapped, prefix, dtype, scaled, marks=pytest.mark.L0 if selected else pytest.mark.L1)


@pytest.mark.parametrize("side,position,prefix,dtype", list(_transpose_cases()))
def test_frost_transpose_analysis_and_render(side, position, prefix, dtype):
    """Check immutable analysis and both operand orders against generated Python."""
    g, tensors = _graph(side=side, position=position, prefix=prefix, dtype=dtype)

    def snapshot():
        """Record logical graph metadata to detect accidental analysis mutation."""
        return [
            (repr(n.params), [(id(t), list(t.get_dim()), list(t.get_stride()), t.get_data_type()) for t in (*n.inputs.values(), *n.outputs.values())])
            for n in g._nodes
        ]

    before = snapshot()
    chain, binding = analyze_with_binding(g)
    assert (chain.matmul.M, chain.matmul.N, chain.matmul.K) == (128, 192, 136)
    assert binding.a_operands == [tensors["a"]] and binding.b_operands == [tensors["b"]]
    assert binding.a_transposed == ("a" in side and position not in ("double", "consecutive"),)
    assert binding.b_transposed == ("b" in side and position not in ("double", "consecutive"),)
    assert chain.ops == []
    for s in "ab":
        ops = getattr(chain, f"mainloop_{s}_ops")
        expected = (["cos", "leaky_relu", "relu"] if position == "surrounded" else ["leaky_relu"]) if s in side else []
        assert [op.op for op in ops] == expected
        assert all(op.out_dtype == ("bf16" if dtype == BF16 else "fp16") and op.compute_dtype == "fp32" for op in ops)
        assert all(op.attrs == (("negative_slope", 0.125),) for op in ops if op.op == "leaky_relu")
    assert before == snapshot()
    assert analyze(g) == chain
    for swapped in (False, True):
        normalized = swap_ab(chain) if swapped else chain
        cfg = _config(swapped)
        snippets = epilogue_codegen.generate(normalized, vec_bytes_epi=16, output_elem_bytes=2, tma_slots=compiler._tma_slots_for(normalized, cfg))
        src = compiler._render_template(normalized, snippets, cfg)
        ast.parse(src)
        assert "@@INJECT_" not in src
        assert (".to(cutlass.BFloat16)" if dtype == BF16 else ".to(cutlass.Float16)") in snippets.mainloop_transform_a + snippets.mainloop_transform_b
        assert "mainloop_koob_fix = True" in src


@pytest.mark.parametrize("side,position,swapped,prefix,dtype,scaled", list(_transpose_cases(rebinding=True)))
def test_frost_transpose_original_axis_rebinding(monkeypatch, side, position, swapped, prefix, dtype, scaled):
    """Verify dynamic caller coordinates and rounding through recorded launch frames."""
    g, tensors = _graph(side=side, position=position, prefix=prefix, dtype=dtype, scaled=scaled)
    tdtype = torch.bfloat16 if dtype == BF16 else torch.float16
    plan, launches = _mock_plan(monkeypatch, g, swapped)
    for m, n, k in ((128, 192, 136), (160, 224, 264)):
        buffers = _buffers(tensors, m, n, k, dtype=tdtype)
        plan(_pack(tensors, buffers))
        frame = launches[-1]
        assert frame[0][:3] == ((n, m, k) if swapped else (m, n, k))
        ordered = (("b", n), ("a", m)) if swapped else (("a", m), ("b", n))
        for i, (s, extent) in enumerate(ordered, 1):
            assert frame[i].shape == (extent, k, prod(prefix))
            assert frame[i].stride()[:2] == (k, 1)
            if prod(prefix) > 1:
                assert frame[i].stride()[2] == extent * k
            assert frame[i].data_ptr() == buffers[s].data_ptr()
        assert frame[3].data_ptr() == buffers["y"].data_ptr()
        got = _evaluate_frame(frame, plan.chain)
        got = got.transpose(1, 2) if swapped else got
        expected = _reference(buffers, side, position, scaled, tdtype).view(prod(prefix), m, n)
        torch.testing.assert_close(got.to(tdtype), expected, atol=0, rtol=0)
    assert len(launches) == 2


@pytest.mark.parametrize("prefix", [(), (1,), (2, 3), (2, 1, 3)])
@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.L0
def test_frost_matrix_view_without_transpose(monkeypatch, prefix, swapped):
    """Normalize non-rank-3 GEMMs without requiring a transpose node."""
    g, tensors = _graph(side="", prefix=prefix)
    plan, launches = _mock_plan(monkeypatch, g, swapped)
    buffers = _buffers(tensors)
    plan(_pack(tensors, buffers))
    frame = launches[-1]
    assert frame[0][:4] == ((192, 128, 136, prod(prefix)) if swapped else (128, 192, 136, prod(prefix)))
    got = _evaluate_frame(frame, plan.chain)
    got = got.transpose(1, 2) if swapped else got
    expected = _reference(buffers, "", "after", False, torch.bfloat16).view(prod(prefix), 128, 192)
    torch.testing.assert_close(got.bfloat16(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("position", ["before", "after", "surrounded", "double", "consecutive"])
@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.L0
def test_frost_transpose_per_k_binding(monkeypatch, position, swapped):
    """Reject invalid scale layouts and dtypes before launch, then recover."""
    g, tensors = _graph(position=position, scaled=True)
    plan, launches = _mock_plan(monkeypatch, g, swapped)
    for m, n, k in ((128, 192, 136), (160, 224, 264)):
        buffers = _buffers(tensors, m, n, k)
        for key in ("scale_a", "scale_b"):
            scale = buffers[key]
            for bad in (scale.transpose(1, 2), scale.to(torch.bfloat16)):
                with pytest.raises(ValueError, match="per-K mainloop scale"):
                    plan(_pack(tensors, dict(buffers, **{key: bad})))
                assert len(launches) == (0 if k == 136 else 2)
                assert torch.count_nonzero(buffers["y"]) == 0
        plan(_pack(tensors, buffers))
        got = _evaluate_frame(launches[-1], plan.chain)
        got = got.transpose(1, 2) if swapped else got
        torch.testing.assert_close(got.bfloat16(), _reference(buffers, "ab", position, True, torch.bfloat16), atol=0, rtol=0)
        views = dict(buffers)
        for key in ("scale_a", "scale_b"):
            views[key] = DeviceView(buffers[key].data_ptr(), buffers[key].shape, dtype_name(buffers[key]), 0)
        plan(_pack(tensors, views))
        for arg, ref in zip(launches[-1][4:], plan.chain.aux_tensors):
            assert arg.shape[ref.k_axis] == k
            assert any(arg.data_ptr() == buffers[key].data_ptr() for key in ("scale_a", "scale_b"))


@pytest.mark.L0
def test_frost_transpose_shared_input_and_binding_keys(monkeypatch):
    """Deduplicate shared storage across tensor, name and UID bindings."""
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    x = g.tensor(name="shared", dim=[1, 128, 136], stride=[128 * 136, 136, 1])
    a = g.leaky_relu(input=x, negative_slope=0.125).set_data_type(BF16)
    b = g.transpose(input=g.relu(input=x).set_data_type(BF16), permutation=[0, 2, 1])
    y = g.matmul(A=a, B=b).set_output(True)
    _, binding = analyze_with_binding(g)
    assert binding.bound_tensors() == [x, y]
    assert binding.a_transposed == (False,) and binding.b_transposed == (True,)
    plan, launches = _mock_plan(monkeypatch, g)
    xv, yv = torch.randn(1, 160, 264, dtype=torch.bfloat16), torch.empty(1, 160, 160, dtype=torch.bfloat16)
    for pack in ({x: xv, y: yv}, {x.get_name(): xv, y.get_name(): yv}, {x.get_uid(): xv, y.get_uid(): yv}):
        assert len(resolve_variant_pack(pack, binding)) == 2
        plan(pack)
        assert launches[-1][1].data_ptr() == launches[-1][2].data_ptr() == xv.data_ptr()
        assert launches[-1][0][:3] == (160, 160, 264)


@pytest.mark.L0
def test_frost_transpose_dtype_inheritance():
    """Keep transpose outputs at the original pointwise dtype."""
    g, _ = _graph()
    g.validate()
    assert analyze(g).mainloop_a_ops[0].out_dtype == "bf16"
    nodes = [n for n in g._nodes if n.node_type.name == "TRANSPOSE"]
    assert all(n.outputs["OUT_0"].get_data_type() == BF16 for n in nodes)


@pytest.mark.parametrize(
    "problem,reason", [("permutation", "rank-3|last-two"), ("dtype", "dtype changes"), ("shape", "expected output shape"), ("output", "materialized")]
)
@pytest.mark.L0
def test_frost_transpose_rejection(problem, reason):
    """Reject unsupported permutations, conversions, shapes and observable intermediates."""
    g, _ = _graph(side="a")
    node = next(n for n in g._nodes if n.node_type.name == "TRANSPOSE")
    if problem == "permutation":
        node.params["permutation"] = [1, 0, 2]
    elif problem == "dtype":
        node.outputs["OUT_0"].set_data_type(F32)
    elif problem == "shape":
        node.outputs["OUT_0"].set_dim([1, 128, 144])
    else:
        node.inputs["input"].set_output(True)
    with pytest.raises(ValueError, match=reason):
        analyze(g)


@pytest.mark.L0
def test_frost_transpose_epilogue_rejected():
    """Decline epilogue transposes rather than silently dropping their semantics."""
    g, tensors = _graph(side="a")
    tensors["y"].set_output(False)
    g.transpose(input=tensors["y"], permutation=[0, 2, 1]).set_output(True)
    with pytest.raises(ValueError, match="operand-only"):
        analyze(g)


def _reference(buffers, side, position, scaled, dtype, epilogue=False):
    """Evaluate logical operations with graph-prescribed per-node rounding."""
    values = {}
    for s in "ab":
        value = buffers[s]
        if s in side:
            if position in ("before", "double"):
                value = value.transpose(-1, -2)
            if position == "surrounded":
                value = torch.cos(value.float()).to(dtype)
            if scaled:
                value = (buffers[f"scale_{s}"].float() * value.float()).to(dtype)
            value = torch.where(value.float() >= 0, value.float(), value.float() * 0.125).to(dtype)
            if position not in ("before", "consecutive"):
                value = value.transpose(-1, -2)
            if position == "surrounded":
                value = torch.relu(value)
        values[s] = value
    value = values["a"].float() @ values["b"].float()
    if epilogue:
        value = value + buffers["bias"].float()  # virtual GEMM stays FP32 until the add
    return value.to(dtype)


@pytest.mark.L0
def test_frost_transpose_epilogue_reference_rounding():
    """Keep virtual GEMM precision until the epilogue add rounds its output."""
    g, _ = _graph(prefix=(), epilogue=True)
    assert analyze(g).matmul.out_dtype == "fp32"
    buffers = {
        "a": torch.ones(2, 1, dtype=torch.bfloat16),
        "b": torch.tensor([[1, 1 / 256]], dtype=torch.bfloat16),
        "bias": torch.tensor([[-1]], dtype=torch.bfloat16),
    }
    # GEMM's 1 + 1/256 is a BF16 midpoint. Narrowing it before the FP32 add
    # would lose the residual and produce zero instead of the graph's 1/256.
    expected = torch.tensor([[1 / 256]], dtype=torch.bfloat16)
    torch.testing.assert_close(_reference(buffers, "ab", "after", False, torch.bfloat16, True), expected, atol=0, rtol=0)


@requires_sm100
@pytest.mark.parametrize(
    "side,position,scaled,swapped,dtype,prefix,broadcast,epilogue",
    [
        ("a", "after", False, False, BF16, (1,), False, False),
        ("b", "before", False, True, BF16, (1,), False, False),
        ("ab", "surrounded", True, False, BF16, (2,), False, False),
        ("ab", "after", True, True, cudnn.data_type.HALF, (1,), False, False),
        ("ab", "double", False, True, BF16, (1,), False, False),
        ("ab", "consecutive", True, False, cudnn.data_type.HALF, (), False, False),
        ("ab", "double", True, True, cudnn.data_type.HALF, (2, 3), False, False),
        ("ab", "surrounded", True, False, BF16, (2, 1, 3), False, False),
        ("ab", "after", True, False, BF16, (), False, False),
        ("ab", "after", False, True, BF16, (), False, True),
        ("ab", "after", True, False, BF16, (2, 3), True, False),
        ("ab", "after", False, True, BF16, (2, 1, 3), False, True),
    ],
)
@pytest.mark.L0
def test_frost_transpose_gpu(side, position, scaled, swapped, dtype, prefix, broadcast, epilogue):
    """Validate full kernels with dynamic shapes and changed-input capture replay."""
    g, tensors = _graph(side=side, position=position, scaled=scaled, dtype=dtype, prefix=prefix, broadcast=broadcast, epilogue=epilogue)
    plan = compiler.jit_from_cudnn_graph(g, _config(swapped))
    tdtype = torch.bfloat16 if dtype == BF16 else torch.float16
    torch.manual_seed(42)
    stream = torch.cuda.Stream()
    for m, n, k in ((128, 192, 136), (160, 224, 264)):
        buffers = _buffers(tensors, m, n, k, device="cuda", dtype=tdtype)
        pack = _pack(tensors, buffers)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            plan(pack, stream=stream.cuda_stream)
        stream.synchronize()
        captured = torch.cuda.CUDAGraph()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message="The CUDA Graph is empty.*")
                with torch.cuda.graph(captured, stream=stream):
                    plan(pack, stream=torch.cuda.current_stream().cuda_stream)
            for _ in range(2):
                buffers["a"].neg_()
                if scaled:
                    for s in side:
                        buffers[f"scale_{s}"].neg_()
                buffers["y"].fill_(float("nan"))
                captured.replay()
                torch.testing.assert_close(buffers["y"], _reference(buffers, side, position, scaled, tdtype, epilogue), atol=0.25, rtol=0.02)
                previous = buffers["y"].clone()
                buffers["y"].fill_(float("nan"))
                captured.replay()
                assert torch.equal(buffers["y"], previous)
            # Exact K indexing after transforms, and exact cos(0)=1 for the K-tail case.
            if position == "surrounded":
                buffers["a"].zero_()
                buffers["b"].zero_()
            else:
                av = buffers["a"].transpose(-1, -2) if tensors["a"].get_stride()[-2] == 1 else buffers["a"]
                bv = buffers["b"].transpose(-1, -2) if tensors["b"].get_stride()[-2] == 1 else buffers["b"]
                av.copy_((torch.arange(k, device="cuda") % 17 - 8).to(tdtype).reshape(*(1,) * len(prefix), 1, k).expand_as(av))
                bv.zero_()
                bv.view(prod(prefix), n, k)[:, torch.arange(n, device="cuda"), torch.arange(n, device="cuda") * 5 % k] = 1
            if scaled:
                for s in side:
                    buffers[f"scale_{s}"].copy_((torch.arange(k, device="cuda") % 5 - 2).reshape(buffers[f"scale_{s}"].shape))
            buffers["y"].fill_(float("nan"))
            captured.replay()
            torch.testing.assert_close(buffers["y"], _reference(buffers, side, position, scaled, tdtype, epilogue), atol=0, rtol=0)
        finally:
            captured.reset()


@pytest.mark.L0
def test_frost_transpose_identity_pure_and_mixed_rounding():
    """Preserve mixed pointwise rounding through identity and matrix transposes."""
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    a = g.tensor(name="A", dim=[1, 136, 128], stride=[128 * 136, 1, 136])
    b = g.tensor(name="B", dim=[1, 136, 192], stride=[136 * 192, 1, 136])
    # The FLOAT intermediate and BF16 result must remain two distinct rounding boundaries.
    value = g.cos(input=a).set_data_type(F32)
    value = g.transpose(input=value, permutation=[0, 1, 2])
    value = g.leaky_relu(input=value, negative_slope=0.125).set_data_type(BF16)
    value = g.transpose(input=value, permutation=[0, 2, 1])
    g.matmul(A=value, B=b).set_output(True)
    chain = analyze(g)
    assert [op.out_dtype for op in chain.mainloop_a_ops] == ["fp32", "bf16"]
    assert chain.mainloop_a_load_dtype is None
    snippet = epilogue_codegen.generate_mainloop(chain, "a")
    assert "_c_ml_a_1 = (_op_900).to(cutlass.Float32)" in snippet
    assert "_r_ml_a_1 = (_op_901).to(cutlass.BFloat16)" in snippet
    # A pure transpose needs no transform or extra device launch.
    pure = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    pa = pure.tensor(name="A", dim=[1, 136, 128], stride=[128 * 136, 1, 136])
    pb = pure.tensor(name="B", dim=[1, 136, 192], stride=[136 * 192, 1, 136])
    pure.matmul(A=pure.transpose(input=pa, permutation=[0, 2, 1]), B=pb).set_output(True)
    chain, binding = analyze_with_binding(pure)
    assert chain.mainloop_a_ops == [] and chain.mainloop_b_ops == []
    assert binding.a_transposed == (True,)


@pytest.mark.parametrize("scalar_first", [False, True])
@pytest.mark.L0
def test_frost_transpose_scalar_and_shared_scale(monkeypatch, scalar_first):
    """Retain scalar operand order and deduplicate shared scale storage."""
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    a = g.tensor(name="A", dim=[1, 136, 128], stride=[128 * 136, 1, 136])
    b = g.tensor(name="B", dim=[1, 136, 192], stride=[136 * 192, 1, 136])
    scale = g.tensor(name="scale", dim=[1, 136, 1], stride=[136, 1, 136], data_type=F32)
    scalar = g.tensor(name="A_mn_view", dim=[1], stride=[1])
    value = g.mul(a=scale, b=a).set_data_type(BF16)
    value = g.sub(a=scalar if scalar_first else value, b=value if scalar_first else scalar).set_data_type(BF16)
    value = g.transpose(input=value, permutation=[0, 2, 1])
    bvalue = g.mul(a=b, b=scale).set_data_type(BF16)
    g.matmul(A=value, B=bvalue).set_output(True)
    chain, binding = analyze_with_binding(g)
    assert len(binding.bound_tensors()) == 5  # A, B, output, scalar and one shared scale
    assert len(chain.aux_tensors) == 3  # scalar and two scale orientations
    assert len(set(ref.name for ref in chain.aux_tensors)) == 3
    assert chain.mainloop_a_ops[-1].aux_on_rhs is not scalar_first
    plan, launches = _mock_plan(monkeypatch, g)
    av = torch.randn(1, 160, 264, dtype=torch.bfloat16).transpose(1, 2)
    bv = torch.randn(1, 224, 264, dtype=torch.bfloat16).transpose(1, 2)
    sv = torch.ones(1, 264, 1, dtype=torch.float32)
    y = binding.outputs[0]
    plan({a: av, b: bv, scale: sv, scalar: torch.ones(1, dtype=torch.bfloat16), y: torch.zeros(1, 160, 224, dtype=torch.bfloat16)})
    assert launches[-1][0][:3] == (160, 224, 264)


@pytest.mark.parametrize("problem", ["rank", "unrelated", "multi", "transpose-output"])
@pytest.mark.L0
def test_frost_transpose_scope_rejection(problem):
    """Reject paths outside the single unobservable operand-view contract."""
    g, tensors = _graph(side="a")
    if problem == "rank":
        node = next(n for n in g._nodes if n.node_type.name == "TRANSPOSE")
        node.inputs["input"].set_dim([136, 128])
    elif problem == "unrelated":
        g.transpose(input=tensors["b"], permutation=[0, 2, 1])
    elif problem == "multi":
        g.matmul(A=tensors["a"], B=tensors["b"])
    else:
        node = next(n for n in g._nodes if n.node_type.name == "TRANSPOSE")
        node.outputs["OUT_0"].set_output(True)
    with pytest.raises(ValueError):
        analyze(g)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for the transform fragment")
@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize("dtype", [BF16, cudnn.data_type.HALF])
@pytest.mark.L0
def test_frost_transpose_generated_fragment(tmp_path, side, dtype):
    """Run normalized transforms on SM90+; not a full mainloop/MMA qualification."""
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 90:
        pytest.skip("generated transform fragment requires SM90+")
    chain = analyze(_graph(side=side, position="surrounded", scaled=True, dtype=dtype)[0])
    ref = chain.aux_tensors[0]
    snippet = epilogue_codegen.generate_mainloop(chain, side)
    fake = compiler._aux_fake_block(chain.aux_tensors, dynamic_strides=True)
    src = f"""import cutlass
import cutlass.cute as cute

mainloop_koob_fix = True
a_is_m_major = False
b_is_n_major = False
cta_tile_mnk = (128, 128, 64)

@cute.kernel
def kernel(A: cute.Tensor, {ref.name}: cute.Tensor, C: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    ml_vec_elems = 8
    k_tile_idx = (tid * 8) // 64
    ml_L = (tid * 8) % 64
    k = cutlass.Int64({ref.name}.shape[{ref.k_axis}])
    ml_vec_{side} = (A.iterator.raw_ptr() + tid * 8).load(count=8, alignment=16)
{textwrap.indent(snippet, '    ')}
    (C.iterator.raw_ptr() + tid * 8).store(ml_out_{side}, alignment=16)

@cute.jit
def entry(A: cute.Tensor, {ref.name}: cute.Tensor, C: cute.Tensor):
    kernel(A, {ref.name}, C).launch(grid=(1,1,1), block=(64,1,1))

sym_k = cute.sym_int64()
{fake}
fake_data = cute.runtime.make_fake_tensor({compiler.DTYPE_TO_CUTLASS[chain.matmul.a_dtype]}, (512,), stride=(1,), assumed_align=16)
compiled = cute.compile(entry, fake_data, fake_{ref.name}, fake_data, options="--enable-tvm-ffi")
"""
    path = tmp_path / f"transpose_fragment_{side}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(f"transpose_fragment_{side}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tdtype = torch.bfloat16 if dtype == BF16 else torch.float16
    av = torch.linspace(-3, 3, 512, device="cuda").to(tdtype)
    out = torch.empty_like(av)
    for k in (128, 136, 264):
        scale = ((torch.arange(k, device="cuda") % 17 - 8).float() / 8).reshape((1, 1, k) if side == "a" else (1, k, 1))
        out.fill_(float("nan"))
        module.compiled(av, scale, out)
        value = torch.cos(av[:k].float()).to(tdtype)
        value = (value.float() * scale.reshape(-1)).to(tdtype)
        value = torch.where(value >= 0, value, value * 0.125).relu()
        # Padded cos(0) would be one: verify every padded lane is cleared exactly.
        torch.testing.assert_close(out[:k], value, atol=0.002 if dtype == BF16 else 0.0003, rtol=0.002)
        assert torch.count_nonzero(out[k:]) == 0


@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.L0
def test_frost_transpose_invalid_input_preserves_output(monkeypatch, swapped):
    """Validate bad operand views before output writes and allow subsequent recovery."""
    g, tensors = _graph()
    plan, launches = _mock_plan(monkeypatch, g, swapped)
    buffers = _buffers(tensors, 160, 224, 264)
    for key in "ab":
        for bad in (buffers[key].transpose(1, 2), buffers[key][..., ::2]):
            with pytest.raises(ValueError):
                plan(_pack(tensors, dict(buffers, **{key: bad})))
            assert launches == []
            assert torch.count_nonzero(buffers["y"]) == 0
    plan(_pack(tensors, buffers))
    assert len(launches) == 1


@pytest.mark.parametrize("m_major", [False, True])
@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.L0
def test_frost_pure_transpose_layout_binding(monkeypatch, m_major, swapped):
    """Bind pure transposes with either matrix-major layout without copying."""
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    x = g.tensor(name="X", dim=[1, 136, 128], stride=[136 * 128, 128, 1] if m_major else [136 * 128, 1, 136])
    b = g.tensor(name="B", dim=[1, 136, 192], stride=[136 * 192, 1, 136])
    y = g.matmul(A=g.transpose(input=x, permutation=[0, 2, 1]), B=b).set_output(True)
    plan, launches = _mock_plan(monkeypatch, g, swapped)
    assert plan.chain.mainloop_a_ops == [] and plan.chain.mainloop_b_ops == []
    for m, n, k in ((128, 192, 136), (160, 224, 264)):
        xv = torch.randn(1, k, m, dtype=torch.bfloat16) if m_major else torch.randn(1, m, k, dtype=torch.bfloat16).transpose(1, 2)
        bv = torch.randn(1, n, k, dtype=torch.bfloat16).transpose(1, 2)
        yv = torch.empty(1, m, n, dtype=torch.bfloat16)
        plan({x: xv, b: bv, y: yv})
        frame = launches[-1]
        assert frame[0][:3] == ((n, m, k) if swapped else (m, n, k))
        got = _evaluate_frame(frame, plan.chain)
        got = got.transpose(1, 2) if swapped else got
        torch.testing.assert_close(got.bfloat16(), (xv.float().transpose(1, 2) @ bv.float()).bfloat16(), atol=0, rtol=0)
        assert frame[2 if swapped else 1].data_ptr() == xv.data_ptr()


@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.parametrize("prefix", [(), (1,), (2, 3), (2, 1, 3)])
@pytest.mark.parametrize("dtype", [BF16, cudnn.data_type.HALF])
@pytest.mark.L0
def test_frost_transpose_keeps_untransposed_epilogue_branch(monkeypatch, swapped, prefix, dtype):
    """A shared pre-transpose pointwise node remains in its original coordinates."""
    g = cudnn.pygraph(io_data_type=dtype, intermediate_data_type=F32, compute_data_type=F32)
    dim = [*prefix, 128, 128]
    stride = [16384 * prod(prefix[i + 1 :]) for i in range(len(prefix))] + [1, 128]
    permutation = [*range(len(prefix)), len(prefix) + 1, len(prefix)]
    a = g.tensor(name="A", dim=dim, stride=stride)
    b = g.tensor(name="B", dim=dim, stride=stride)
    pointwise = g.leaky_relu(input=a, negative_slope=0.125).set_data_type(dtype)
    transposed = g.transpose(input=pointwise, permutation=permutation)
    c = g.matmul(A=transposed, B=b)
    y = g.add(a=c, b=pointwise).set_output(True)
    chain, binding = analyze_with_binding(g)
    assert [op.op for op in chain.ops] == ["aux_load", "leaky_relu", "add"]
    assert chain.ops[1].attrs == chain.mainloop_a_ops[0].attrs
    assert binding.aux == [a] and binding.aux_transposed == (False,)
    assert binding.bound_tensors() == [a, b, y]
    plan, launches = _mock_plan(monkeypatch, g, swapped)
    tdtype = torch.bfloat16 if dtype == BF16 else torch.float16
    av = torch.randn(*prefix, 128, 128, dtype=tdtype).transpose(-1, -2)
    bv = torch.randn(*prefix, 128, 128, dtype=tdtype).transpose(-1, -2)
    plan({a: av, b: bv, y: torch.empty_like(av)})
    aux = launches[-1][4]
    assert aux.data_ptr() == av.data_ptr()
    matrix_shape = (prod(prefix), 128, 128)
    expected_aux = av if tuple(av.shape) == matrix_shape else av.view(matrix_shape)
    if swapped:
        expected_aux = expected_aux.transpose(1, 2)
    assert tuple(aux.stride()) == tuple(expected_aux.stride())
    torch.testing.assert_close(aux, expected_aux, atol=0, rtol=0)
    got = _evaluate_frame(launches[-1], plan.chain)
    got = got.transpose(1, 2) if swapped else got
    activated = torch.nn.functional.leaky_relu(av.float(), 0.125).to(tdtype)
    expected = activated.float().transpose(-1, -2) @ bv.float()
    torch.testing.assert_close(got, expected.view(prod(prefix), 128, 128), atol=0, rtol=0)


@pytest.mark.L0
def test_frost_transpose_shared_epilogue_view_rejected():
    """Reject a shared epilogue branch that still requires an observable transpose."""
    g, tensors = _graph(side="a")
    transpose = next(n for n in g._nodes if n.node_type.name == "TRANSPOSE").outputs["OUT_0"]
    tensors["y"].set_output(False)
    g.add(a=tensors["y"], b=transpose).set_output(True)
    with pytest.raises(ValueError, match="not a graph input|broadcast"):
        analyze(g)


@pytest.mark.parametrize("prefix", [(), (1,), (2, 3)])
@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.L0
def test_frost_transpose_native_transport_and_overrides(monkeypatch, prefix, swapped, raw, ordered):
    """Preserve original pointers and padded pitches through both native pack transports."""
    g, tensors = _graph(prefix=prefix, scaled=True, gap=16, scale_prefix=())
    g.validate()
    compiled, launches = _mock_plan(monkeypatch, g, swapped)
    engine = _FrostGemmPlan(compiled)
    buffers = _buffers(tensors, gap=16)
    uids = [t.get_uid() for t in tensors.values()]
    values = [buffers[key].data_ptr() if raw else buffers[key] for key in tensors]
    # Live shape is smaller than allocation; retain each original physical pitch.
    shapes = []
    for key in tensors:
        shape = list(buffers[key].shape)
        if key == "a":
            shape[-2:] = [64, 128]
        elif key == "b":
            shape[-2:] = [128, 64]
        elif key == "y":
            shape[-2:] = [128, 128]
        else:
            shape = [64, 1] if key == "scale_a" else [1, 64]
        shapes.append(shape)
    strides = [list(buffers[key].stride()) for key in tensors]
    if ordered:
        pack = g._normalize_ordered(values, uids, None, uids, shapes, strides)
    else:
        pack = g._normalize(dict(zip(uids, values)), None, uids, shapes, strides)
    engine.execute(g, pack, ExecutionContext(stream=0))
    frame = launches[-1]
    assert frame[0][:4] == (128, 128, 64, prod(prefix))
    for i, key in enumerate(("b", "a") if swapped else ("a", "b"), 1):
        assert frame[i].data_ptr() == buffers[key].data_ptr()
        assert tuple(frame[i].shape) == (128, 64, prod(prefix))
        assert frame[i].stride()[0] == 144  # the original allocation's K+padding, not live K


@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.L0
def test_frost_transpose_rank_broadcast_and_epilogue(monkeypatch, swapped):
    """Fold full batch broadcast while retaining auxiliary and output storage."""
    g, tensors = _graph(prefix=(2, 3), broadcast=True, epilogue=True)
    chain, binding = analyze_with_binding(g)
    assert (chain.matmul.a_batch, chain.matmul.b_batch, chain.matmul.batch) == (1, 6, 6)
    assert chain.aux_tensors[0].dim == (1, 1, 192)
    plan, launches = _mock_plan(monkeypatch, g, swapped)
    buffers = _buffers(tensors)
    plan(_pack(tensors, buffers))
    frame = launches[-1]
    assert frame[1 if not swapped else 2].shape[-1] == 1
    assert frame[4].data_ptr() == buffers["bias"].data_ptr()
    assert frame[3].data_ptr() == buffers["y"].data_ptr()


@pytest.mark.L0
def test_frost_transpose_rank_rejects_non_affine_batch(monkeypatch):
    """Reject non-affine runtime and declared batch layouts without launching."""
    g, tensors = _graph(prefix=(2, 3))
    plan, launches = _mock_plan(monkeypatch, g)
    buffers = _buffers(tensors)
    # Safe backing allocation, but inter-batch axes no longer share one pitch.
    bad = torch.empty_strided((2, 3, 136, 128), (4 * 136 * 128, 136 * 128, 1, 136), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="not affine"):
        plan(_pack(tensors, dict(buffers, a=bad)))
    assert launches == [] and torch.count_nonzero(buffers["y"]) == 0
    plan(_pack(tensors, buffers))
    assert len(launches) == 1
    tensors["a"].set_stride(list(bad.stride()))
    with pytest.raises(ValueError, match="not affine"):
        analyze(g)


@pytest.mark.parametrize("prefix", [(2, 3), (1, 3)])
@pytest.mark.L0
def test_frost_transpose_rank_rejects_partial_batch_broadcast(prefix):
    """Decline crossed or partial batches that need more than one linear pitch."""
    g, _ = _graph(prefix=prefix, a_prefix=(2, 1))
    with pytest.raises(ValueError, match="partial/crossed"):
        analyze(g)


@pytest.mark.L0
def test_frost_transpose_rank_rejects_batch_axis_permutation():
    """Reject batch permutations even when equal extents hide the shape change."""
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    a = g.tensor(name="A", dim=[2, 2, 136, 128], stride=[2 * 136 * 128, 136 * 128, 1, 136])
    b = g.tensor(name="B", dim=[2, 2, 136, 192], stride=[2 * 136 * 192, 136 * 192, 1, 136])
    a = g.transpose(input=a, permutation=[1, 0, 3, 2])
    g.matmul(A=a, B=b).set_output(True)
    g.validate()  # Equal batch extents mask the permutation in the output shape, not in its semantics.
    with pytest.raises(ValueError, match="batch axes must stay fixed"):
        analyze(g)


@pytest.mark.L0
def test_frost_transpose_rank_pins_batch_axes_and_rank(monkeypatch):
    """Require every prepared batch axis and rank to remain in caller coordinates."""
    g, tensors = _graph(prefix=(2, 3))
    plan, launches = _mock_plan(monkeypatch, g)
    buffers = _buffers(tensors)
    for bad in (buffers["a"].transpose(0, 1), buffers["a"].view(6, 136, 128)):
        with pytest.raises(ValueError, match="original batch axes"):
            plan(_pack(tensors, dict(buffers, a=bad)))
        assert launches == []
    plan(_pack(tensors, buffers))
    assert len(launches) == 1


@pytest.mark.parametrize("prefix", [(), (1,), (2, 3)])
@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.L0
def test_frost_transpose_workspace_uses_original_axes(monkeypatch, prefix, swapped):
    """Derive dynamic split workspace from caller buffers or original-rank overrides."""
    g, tensors = _graph(prefix=prefix)
    plan, _ = _mock_plan(monkeypatch, g, swapped)
    plan = replace(plan, config=replace(plan.config, split_k_slices=2))
    buffers = _buffers(tensors, 160, 224, 264)
    ordered = [buffers[next(key for key, tensor in tensors.items() if tensor is t)] for t in plan.bound]
    expected = prod(prefix) * 2 * 160 * 224 * 4
    assert plan.workspace_bytes_for_operands(ordered) == expected
    uids = [t.get_uid() for t in tensors.values()]
    assert plan.workspace_bytes_for_overrides(uids, [list(buffers[key].shape) for key in tensors]) == expected
    # An output-only override also carries the dynamic M/N in original coordinates.
    assert plan.workspace_bytes_for_overrides([tensors["y"].get_uid()], [list(buffers["y"].shape)]) == expected


@pytest.mark.L0
def test_frost_transpose_rank_rejects_overlapping_output(monkeypatch):
    """Reject overlapping writes before launch and preserve recovery on the same plan."""
    g, tensors = _graph(prefix=(2, 3))
    plan, launches = _mock_plan(monkeypatch, g)
    buffers = _buffers(tensors)
    bad = torch.empty_strided((2, 3, 128, 192), (3 * 192, 192, 192, 1), dtype=torch.bfloat16).fill_(7)
    with pytest.raises(ValueError, match="non-overlapping"):
        plan(_pack(tensors, dict(buffers, y=bad)))
    assert launches == [] and torch.all(bad == 7)
    plan(_pack(tensors, buffers))
    assert len(launches) == 1
    tensors["y"].set_stride(list(bad.stride()))
    with pytest.raises(ValueError, match="non-overlapping"):
        analyze(g)


@pytest.mark.parametrize("prefix,bias_prefix", [((), (1,)), ((2, 3), (1, 1, 1))])
@pytest.mark.L0
def test_frost_transpose_rank_unit_aux_axes_and_recovery(monkeypatch, prefix, bias_prefix):
    """Pin otherwise unobservable unit auxiliary axes to the prepared declaration."""
    g, tensors = _graph(prefix=prefix, epilogue=True, bias_prefix=bias_prefix)
    plan, launches = _mock_plan(monkeypatch, g)
    buffers = _buffers(tensors)
    bad = buffers["bias"].view(1, 192)
    with pytest.raises(ValueError, match="original batch axes"):
        plan(_pack(tensors, dict(buffers, bias=bad)))
    assert launches == [] and torch.count_nonzero(buffers["y"]) == 0
    plan(_pack(tensors, buffers))
    assert len(launches) == 1


@pytest.mark.parametrize("shape,stride", [((2, 3, 4, 8), (96, 32, 8, 1)), ((2, 3, 4, 8), (120, 40, 1, 4)), ((2, 1, 3, 4, 8), (120, 999, 40, 1, 4))])
@pytest.mark.L0
def test_frost_matrix_batch_layout_contract(shape, stride):
    """Preserve matrix strides when folding contiguous, gapped and unit batch axes."""
    dim, st = fold_batch_layout(shape, stride)
    assert dim == (6, 4, 8)
    assert st[-2:] == stride[-2:]


@pytest.mark.L0
def test_frost_matrix_batch_int64_boundaries():
    # Validate host widths without attempting an enormous allocation.
    """Accept wide pitches and reject signed-64-bit geometry overflow without allocation."""
    dim, stride = fold_batch_layout((2, 2, 1, 1), (1 << 33, 1 << 32, 1, 1))
    assert dim == (4, 1, 1) and stride[0] == 1 << 32
    with pytest.raises(ValueError, match="64-bit"):
        fold_batch_layout((1 << 32, 1 << 32, 1, 1), (1 << 32, 1, 1, 1))


@pytest.mark.parametrize(
    "shape,stride",
    [
        ((4, 4, 8), (0, 8, 1)),
        ((2, 3, 4, 8), (0, 0, 8, 1)),
        ((2, 2, 1, 1), (1 << 33, 1 << 32, 1, 1)),
        ((2, 1, 3, 4, 8), (120, 999, 40, 1, 4)),
        ((1, 1, 4, 8), (500, 97, 8, 1)),
        ((2, 3, 4, 8), (160, 40, 1, 4)),
        ((1 << 32, 1 << 32, 1, 1), (1 << 32, 1, 1, 1)),
    ],
)
@pytest.mark.L0
def test_frost_matrix_batch_native_geometry(shape, stride):
    """Compare native metadata-only folding with the independent Python geometry checks."""
    from cudnn import _pybind_module

    # A synthetic address is inspected only as metadata; no kernel or DLPack consumer reads it.
    native = _pybind_module.VariantPackNative(1)
    native.set_operand(0, 4096, shape, stride, 4, 16, 1, -1, 2, 0)
    value = native.operand(0, 0)
    try:
        expected_shape, expected_stride = fold_batch_layout(shape, stride)
    except ValueError as error:
        with pytest.raises(ValueError, match="not affine|64-bit"):
            value.flatten_batch()
        assert "not affine" in str(error) or "64-bit" in str(error)
    else:
        folded = value.flatten_batch()
        assert tuple(folded.shape) == expected_shape and tuple(folded.stride()) == expected_stride
        assert folded.data_ptr() == value.data_ptr() == 4096
        assert folded.dtype == value.dtype and folded.element_size() == value.element_size()


@pytest.mark.L0
@pytest.mark.parametrize("prefix", [(), (1,), (2, 3), (2, 1, 3)])
def test_frost_logical_index_public_graph_composition(prefix):
    """Keep the removed sample's all-axis public graph composition as a regression."""
    g, tensors = _graph(prefix=prefix, side="")
    value = tensors["y"].set_output(False)
    for axis in range(len(prefix) + 2):
        value = g.add(a=value, b=g.gen_index(input=value, axis=axis))
    value.set_output(True).set_data_type(F32)
    g.validate()
    chain = analyze(g)
    indices = [op for op in chain.ops if op.op == "gen_index"]
    assert len(indices) == len(prefix) + 2
    assert [op.index_expr for op in indices] == [matrix_axis_index((*prefix, 128, 192), axis) for axis in range(len(prefix) + 2)]
    snippets = epilogue_codegen.generate(chain, vec_bytes_epi=16, output_elem_bytes=4, tma_slots=compiler._tma_slots_for(chain, _config()))
    ast.parse(compiler._render_template(chain, snippets, _config()))
    # Verify the DAG transports a caller's fragment map, not only _emit_op.
    fragment = epilogue_codegen._FragmentCoordinates(
        ((Axis.BATCH, ("plane",) * 4), (Axis.M, tuple(f"lane_row + {i}" for i in range(4))), (Axis.N, ("lane_column",) * 4))
    )
    lines, _ = epilogue_codegen._emit_pointwise_dag(
        chain, vsize=4, bounded=False, aux_load_expr=lambda *args, **kwargs: "unused", fragment_coordinates=fragment
    )
    source = "\n".join(lines)
    for i, op in enumerate(chain.ops):
        if op.op == "gen_index" and op.index_expr.axes == {Axis.M}:
            assert f"_gi{i} = cute.make_rmem_tensor" in source and "lane_row" in source
        elif op.op == "gen_index" and op.index_expr.axes == {Axis.N}:
            assert f"_op_{i} = cutlass.full_like" in source and "lane_column" in source


@pytest.mark.L0
@pytest.mark.parametrize("prefix", [(), (1,), (2, 3)])
@pytest.mark.parametrize("matrix_axis", [0, 1])
@pytest.mark.parametrize("index_source", ["aux", "aux_unary"])
def test_frost_gen_index_broadcast_source_coordinates(monkeypatch, prefix, matrix_axis, index_source):
    """Singleton producer coordinates are zero, not the consumer GEMM coordinates."""
    g, tensors = _graph(prefix=prefix, side="", index_axis=len(prefix) + matrix_axis, index_source=index_source)
    chain = analyze(g)
    op = next(op for op in chain.ops if op.op == "gen_index")
    assert op.index_expr.evaluate({Axis.BATCH: 0, Axis.M: 17, Axis.N: 23}) == 0
    assert not op.index_expr.axes
    plan, frames = _mock_plan(monkeypatch, g)
    engine = _FrostGemmPlan(plan)
    for m, n, k in ((128, 192, 136), (160, 224, 264)):
        buffers = _buffers(tensors, m, n, k)
        engine.execute(g, _graph_coordinate_pack(g, tensors, buffers), ExecutionContext(stream=0))
        assert any(arg.data_ptr() == buffers["index_input"].data_ptr() for arg in frames[-1][4:])
    # Fixed singleton axes are part of the real FFI entry contract. The mock
    # records pointers only; exercise its recipe-level rejection on the output.
    ref = next(ref for ref in plan.chain.aux_tensors if ref.name == "index_input")
    assert ref.symbolic_shape[-2 + matrix_axis] == 1
    invalid = dict(buffers)
    invalid["y"] = buffers["y"][..., :-1]
    buffers["y"].fill_(123)
    before = len(frames)
    with pytest.raises(ValueError):
        engine.execute(g, _graph_coordinate_pack(g, tensors, invalid), ExecutionContext(stream=0))
    assert len(frames) == before and torch.all(buffers["y"] == 123)
    engine.execute(g, _graph_coordinate_pack(g, tensors, buffers), ExecutionContext(stream=0))
    assert len(frames) == before + 1


@pytest.mark.L0
@pytest.mark.parametrize("matrix_axis", [0, 1])
@pytest.mark.parametrize("epilogue", [False, True])
def test_frost_gen_index_unit_anchor_axes_remain_dynamic(matrix_axis, epilogue):
    """A prepared unit GEMM axis can grow; broadcasting auxiliaries cannot."""
    g, _ = _graph(M=1 if matrix_axis == 0 else 128, N=1 if matrix_axis == 1 else 192, prefix=(1,), side="", epilogue=epilogue, index_axis=matrix_axis + 1)
    op = next(op for op in analyze(g).ops if op.op == "gen_index")
    assert op.index_expr.axes == {Axis.M if matrix_axis == 0 else Axis.N}
    assert op.index_expr.evaluate({Axis.M: 17, Axis.N: 23}) == (17 if matrix_axis == 0 else 23)


@pytest.mark.parametrize("prefix,axis", [(p, a) for p in [(), (1,), (2, 3), (2, 1, 3)] for a in range(len(p) + 2)])
@pytest.mark.parametrize("side", ["", "ab"])
@pytest.mark.L0
def test_frost_matrix_gen_index_axes(prefix, axis, side):
    """Every logical coordinate survives rank folding and swap-AB without graph mutation."""
    g, _ = _graph(prefix=prefix, side=side, index_axis=axis)
    original = [dict(node.params) for node in g._nodes]
    chain = analyze(g)
    index = next(op for op in chain.ops if op.op == "gen_index")
    assert dict(index.attrs)["axis"] == axis
    assert index.index_expr == matrix_axis_index((*prefix, 128, 192), axis)
    swapped = next(op for op in swap_ab(chain).ops if op.op == "gen_index")
    assert dict(swapped.attrs)["axis"] == axis
    assert swapped.index_expr.evaluate({Axis.BATCH: 0, Axis.M: 19, Axis.N: 13}) == index.index_expr.evaluate({Axis.BATCH: 0, Axis.M: 13, Axis.N: 19})
    assert next(op for op in swap_ab(swap_ab(chain)).ops if op.op == "gen_index") == index
    assert original == [dict(node.params) for node in g._nodes]
    from cudnn.gemm.frost.sm120 import compiler as sm120_compiler, epilogue_codegen as sm120_epilogue

    for config_swapped in (False, True):
        current = swap_ab(chain) if config_swapped else chain
        for slices in (1, 2):
            config = replace(_config(config_swapped), split_k_slices=slices)
            for force_stg in (False, True):
                with compiler.force_stg_epi(force_stg):
                    slots = frozenset({0}) if config.workspace_split_k and not force_stg else compiler._tma_slots_for(current, config)
                    snippets = epilogue_codegen.generate(
                        current, vec_bytes_epi=16, output_elem_bytes=2, tma_slots=slots, workspace_split_k=config.workspace_split_k
                    )
                    ast.parse(compiler._render_template(current, snippets, config))
            # The SM120 STG and split-K routes consume the shared coordinate IR.
            config = replace(by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"), swap_ab=config_swapped, split_k_slices=slices)
            snippets = sm120_epilogue.generate(current, vec_bytes_epi=16, output_elem_bytes=2, workspace_split_k=config.workspace_split_k)
            if current.has_mainloop_fusion:
                ast.parse(snippets.epilogue)  # SM120 has no operand pointwise template.
            else:
                ast.parse(sm120_compiler._render_template(current, snippets, config))


@pytest.mark.parametrize("prefix", [(1,), (2,)])
@pytest.mark.L0
def test_frost_matrix_identity_recipe_fast_path(monkeypatch, prefix):
    """Plain rank-3 calls retain the original operands and runtime validation."""
    from cudnn.gemm.frost import recipe

    g, tensors = _graph(side="", prefix=prefix)
    plan, launches = _mock_plan(monkeypatch, g)
    assert plan.recipe.matrix_views == ()
    buffers = _buffers(tensors)
    ordered = [buffers[next(key for key, tensor in tensors.items() if tensor is t)] for t in plan.bound]

    def unexpected_fold(*args, **kwargs):
        """Fail if the prepared identity path invokes general batch geometry."""
        raise AssertionError("identity-rank launch must not fold layouts")

    monkeypatch.setattr(recipe, "fold_batch_layout", unexpected_fold)
    assert plan.recipe.operand_views(ordered) is ordered
    plan(_pack(tensors, buffers))
    assert len(launches) == 1
    for key, bad in (("a", buffers["a"].view(-1, 136)), ("a", buffers["a"].transpose(1, 2)), ("y", buffers["y"][..., :128])):
        with pytest.raises(ValueError):
            plan(_pack(tensors, dict(buffers, **{key: bad})))
        assert len(launches) == 1
    plan(_pack(tensors, buffers))
    assert len(launches) == 2


def _graph_coordinate_pack(graph, tensors, buffers, workspace=None):
    """Declare live caller shapes/strides explicitly for native graph binding."""
    return graph._normalize(
        {t.get_uid(): buffers[key] for key, t in tensors.items()},
        workspace,
        [t.get_uid() for t in tensors.values()],
        [list(buffers[key].shape) for key in tensors],
        [list(buffers[key].stride()) for key in tensors],
    )


# Preserve the original wrong-axis controls and add logical batch-axis probes.
_GEN_INDEX_GPU_L0 = {
    ((), 1, False),
    ((1,), 0, False),
    ((1,), 1, False),
    ((1,), 2, False),
    ((2, 3), 2, False),
    ((2, 3), 3, True),
    ((2, 1, 3), 3, True),
    ((2, 3), 0, False),
    ((2, 3), 1, True),
    ((2, 1, 3), 1, False),
    ((2, 1, 3), 2, True),
}


@pytest.mark.parametrize(
    "prefix,axis,swapped,route,index_source",
    [
        pytest.param(p, a, s, "sm100", "matmul", marks=[requires_sm100, pytest.mark.L0 if (p, a, s) in _GEN_INDEX_GPU_L0 else pytest.mark.L1])
        for p in [(), (1,), (2, 3), (2, 1, 3)]
        for a in range(len(p) + 2)
        for s in (False, True)
    ]
    + [
        pytest.param(
            p,
            a,
            s,
            route,
            "matmul",
            marks=[requires_sm120 if route == "sm120" else requires_sm100, pytest.mark.L0 if route == "sm120" and a < 2 else pytest.mark.L1],
        )
        for route in ("sm100_stg", "sm100_split", "sm120")
        for p, a, s in [((2, 3), 0, False), ((2, 3), 1, True), ((2, 1, 3), 4, True)]
    ]
    + [
        pytest.param(
            p,
            len(p) + a,
            bool(a),
            route,
            "aux_unary",
            marks=[requires_sm120 if route == "sm120" else requires_sm100, pytest.mark.L0 if p == (1,) else pytest.mark.L1],
        )
        for route in ("sm100", "sm120")
        for p in [(1,), (2, 3)]
        for a in (0, 1)
    ],
)
def test_frost_matrix_gen_index_gpu(prefix, axis, swapped, route, index_source):
    """Check logical indices, dynamic shapes and changed-input capture on both families."""
    dtype = BF16 if axis % 2 == 0 else cudnn.data_type.HALF
    tdtype = torch.bfloat16 if dtype == BF16 else torch.float16
    side = "ab" if swapped and route != "sm120" else ""
    g, tensors = _graph(prefix=prefix, side=side, dtype=dtype, index_axis=axis, index_source=index_source)
    if route == "sm120":
        from cudnn.gemm.frost.sm120 import compiler as backend

        config = replace(by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"), swap_ab=swapped)
        plan = backend.jit_from_cudnn_graph(g, config)
    else:
        config = replace(_config(swapped), split_k_slices=2 if route == "sm100_split" else 1)
        with compiler.force_stg_epi(route == "sm100_stg"):
            plan = compiler.jit_from_cudnn_graph(g, config)
    engine = _FrostGemmPlan(plan)
    stream = torch.cuda.Stream()
    for m, n, k in ((128, 192, 136), (160, 224, 264)):
        buffers = _buffers(tensors, m, n, k, device="cuda", dtype=tdtype)
        buffers["a"].zero_()
        buffers["b"].zero_()
        size = engine.get_workspace_size_for_shapes([t.get_uid() for t in tensors.values()], [list(buffers[key].shape) for key in tensors])
        workspace = torch.empty(size, device="cuda", dtype=torch.uint8) if size else None
        pack = _graph_coordinate_pack(g, tensors, buffers, workspace)
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
            for fill in (0.0, 0.0625):
                # Changes and poisoning are ordered on the replay stream too.
                with torch.cuda.stream(stream):
                    buffers["a"].fill_(fill)
                    buffers["b"].fill_(fill)
                    buffers["y"].fill_(float("nan"))
                    captured.replay()
                stream.synchronize()
                shape = (*prefix, m, n)
                extent = shape[axis] if index_source == "matmul" else 1
                index = torch.arange(extent, device="cuda", dtype=torch.float32)
                view_shape = [1] * len(shape)
                view_shape[axis] = extent
                index = index.reshape(view_shape)
                expected = (index + k * fill * fill).expand(*prefix, m, n).to(tdtype)
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
@pytest.mark.parametrize("side", ["a", "b", "ab"])
def test_frost_identity_zero_batch_stride(monkeypatch, side):
    """Retain read-only identity batch broadcasts without runtime folding."""
    g, tensors = _graph(side="", prefix=(4,))
    for key in side:
        stride = list(tensors[key].get_stride())
        stride[0] = 0
        tensors[key].set_stride(stride)
    plan, launches = _mock_plan(monkeypatch, g)
    assert plan.recipe.matrix_views == ()
    buffers = _buffers(tensors)
    for key in side:
        buffers[key] = buffers[key][:1].expand_as(buffers[key])
    engine = _FrostGemmPlan(plan)
    pack = _graph_coordinate_pack(g, tensors, buffers)
    engine.execute(g, pack, ExecutionContext(stream=0))
    assert len(launches) == 1
    for i, key in enumerate("ab", 1):
        assert launches[-1][i].data_ptr() == buffers[key].data_ptr()
        if key in side:
            assert launches[-1][i].stride()[-1] == 0
    # Reconstruct the recorded native metadata over the known CPU storage.
    a, b = [buffers[key].as_strided(tuple(arg.shape), tuple(arg.stride())).permute(2, 0, 1) for key, arg in zip("ab", launches[-1][1:3])]
    got = a.float() @ b.float().transpose(1, 2)
    expected = buffers["a"].float() @ buffers["b"].float()
    torch.testing.assert_close(got, expected, atol=0, rtol=0)


@pytest.mark.L0
def test_frost_identity_zero_batch_output_rejected():
    """Accept read-only broadcasting without permitting overlapping output writes."""
    g, tensors = _graph(side="", prefix=(4,))
    stride = list(tensors["y"].get_stride())
    stride[0] = 0
    tensors["y"].set_stride(stride)
    with pytest.raises(ValueError, match="non-overlapping"):
        analyze(g)


@pytest.mark.L0
@pytest.mark.parametrize("prefix,side", [((4,), "ab"), ((2, 3), "")])
def test_frost_matrix_view_zero_batch_stride_remains_rejected(prefix, side):
    """Do not extend zero-pitch admission to unqualified runtime-view paths."""
    g, tensors = _graph(prefix=prefix, side=side)
    stride = list(tensors["a"].get_stride())
    stride[:-2] = [0] * len(prefix)
    tensors["a"].set_stride(stride)
    with pytest.raises(ValueError, match="not affine"):
        analyze(g)


@pytest.mark.L0
@pytest.mark.parametrize("prefix,axis", [(p, a) for p in [(), (1,), (2, 3), (2, 1, 3)] for a in range(len(p) + 2)])
@pytest.mark.parametrize("swapped", [False, True])
def test_frost_matrix_gen_index_graph_binding(monkeypatch, prefix, axis, swapped):
    """Bind both dynamic shapes in explicit graph coordinates before kernel entry."""
    g, tensors = _graph(prefix=prefix, side="ab" if swapped else "", index_axis=axis)
    plan, launches = _mock_plan(monkeypatch, g, swapped)
    engine = _FrostGemmPlan(plan)
    for m, n, k in ((128, 192, 136), (160, 224, 264)):
        buffers = _buffers(tensors, m, n, k)
        pack = _graph_coordinate_pack(g, tensors, buffers)
        engine.execute(g, pack, ExecutionContext(stream=0))
        frame = launches[-1]
        assert frame[0][:4] == ((n, m, k, prod(prefix)) if swapped else (m, n, k, prod(prefix)))
        for index, (key, extent) in enumerate((("b", n), ("a", m)) if swapped else (("a", m), ("b", n)), 1):
            assert tuple(frame[index].shape) == (extent, k, prod(prefix))
            assert frame[index].data_ptr() == buffers[key].data_ptr()
    if len(prefix) > 1 and prod(prefix) > 1:
        # Same flattened count is insufficient: logical batch digits are fixed.
        bad = dict(buffers)
        bad["a"] = buffers["a"].reshape(*reversed(prefix), *buffers["a"].shape[-2:])
        before = len(launches)
        buffers["y"].fill_(123)
        with pytest.raises(ValueError, match="original batch axes"):
            engine.execute(g, _graph_coordinate_pack(g, tensors, bad), ExecutionContext(stream=0))
        assert len(launches) == before and torch.all(buffers["y"] == 123)


@pytest.mark.L0
def test_frost_identity_broadcast_layout_scope():
    """Zero-pitch opt-in cannot relax nonidentity folds or writable overlap checks."""
    shape, stride = (4, 8, 16), (0, 16, 1)
    assert fold_batch_layout(shape, stride, allow_identity_broadcast=True) == (shape, stride)
    with pytest.raises(ValueError, match="not affine"):
        fold_batch_layout(shape, stride)
    with pytest.raises(ValueError, match="non-overlapping"):
        fold_batch_layout(shape, stride, writable=True, allow_identity_broadcast=True)
    with pytest.raises(ValueError, match="not affine"):
        fold_batch_layout((2, 2, 8, 16), (0, 0, 16, 1), allow_identity_broadcast=True)
