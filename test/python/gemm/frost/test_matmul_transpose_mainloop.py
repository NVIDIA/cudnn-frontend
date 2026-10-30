# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata-only operand transpose normalization, binding and kernel contracts."""

import ast
import importlib.util
import textwrap
import warnings
from dataclasses import replace
from pathlib import Path

import cudnn
import pytest
import torch

from cudnn.frost.buffers import DeviceView, dtype_name
from cudnn.gemm.frost.fusion_ir import swap_ab
from cudnn.gemm.frost.graph_analyzer import analyze, analyze_with_binding, resolve_variant_pack, swap_ab_binding
from cudnn.gemm.frost.sm100 import compiler, epilogue_codegen
from cudnn.gemm.frost.tile_config import by_name
from gemm_test_utils import requires_sm100

pytestmark = pytest.mark.L0
BF16, F32 = cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT


def _config(swapped=False):
    return replace(by_name("CONFIG_sm100_128x128x64_128x128x32_cluster1x1_1ctamma"), swap_ab=swapped)


def _graph(M=128, N=192, K=136, *, side="ab", position="after", scaled=False, batch=1, dtype=BF16):
    g = cudnn.pygraph(io_data_type=dtype, intermediate_data_type=F32, compute_data_type=F32)
    tensors, operands = {}, {}
    for s, extent in (("a", M), ("b", N)):
        odd = s in side and position != "double"
        dim = [batch, K, extent] if (s == "a") == odd else [batch, extent, K]
        stride = [K * extent, 1, K] if (s == "a") == odd else [K * extent, K, 1]
        value = tensors[s] = g.tensor(name=s.upper(), dim=dim, stride=stride)
        if s not in side:
            operands[s] = value
            continue
        if position in ("before", "double"):
            value = g.transpose(input=value, permutation=[0, 2, 1])
        if position == "surrounded":
            value = g.cos(input=value).set_data_type(dtype)
        if scaled:
            dim = [1, 1, K] if (s == "a") == (position == "before") else [1, K, 1]
            stride = [K, K, 1] if dim[2] == K else [K, 1, K]
            scale = tensors[f"scale_{s}"] = g.tensor(name=f"scale_{s}", dim=dim, stride=stride, data_type=F32)
            value = g.mul(a=scale, b=value).set_data_type(dtype)
        value = g.leaky_relu(input=value, negative_slope=0.125).set_data_type(dtype)
        if position != "before":
            value = g.transpose(input=value, permutation=[0, 2, 1])
        if position == "surrounded":
            value = g.relu(input=value).set_data_type(dtype)
        operands[s] = value
    tensors["y"] = g.matmul(A=operands["a"], B=operands["b"]).set_output(True)
    return g, tensors


def _buffers(tensors, m=128, n=192, k=136, *, batch=1, device="cpu", dtype=torch.bfloat16):
    result = {"y": torch.zeros(batch, m, n, dtype=dtype, device=device)}
    for s, extent in (("a", m), ("b", n)):
        value = torch.randn(batch, extent, k, dtype=dtype, device=device)
        result[s] = value.transpose(1, 2) if tensors[s].get_stride()[1] == 1 else value
        key = f"scale_{s}"
        if key in tensors:
            scale = torch.linspace(-1.25, 1.25, k, dtype=torch.float32, device=device)
            result[key] = scale.reshape(1, k, 1) if tensors[key].get_dim()[1] != 1 else scale.reshape(1, 1, k)
    return result


def _pack(tensors, buffers):
    return {tensor: buffers[key] for key, tensor in tensors.items()}


def _mock_plan(monkeypatch, g, swapped=False):
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


@pytest.mark.parametrize("side", ["a", "b", "ab"])
@pytest.mark.parametrize("position", ["before", "after", "surrounded", "double"])
def test_frost_transpose_analysis_and_render(side, position):
    g, tensors = _graph(side=side, position=position)

    def snapshot():
        return [
            (repr(n.params), [(id(t), list(t.get_dim()), list(t.get_stride()), t.get_data_type()) for t in (*n.inputs.values(), *n.outputs.values())])
            for n in g._nodes
        ]

    before = snapshot()
    chain, binding = analyze_with_binding(g)
    assert (chain.matmul.M, chain.matmul.N, chain.matmul.K) == (128, 192, 136)
    assert binding.a_operands == [tensors["a"]] and binding.b_operands == [tensors["b"]]
    assert binding.a_transposed == ("a" in side and position != "double",)
    assert binding.b_transposed == ("b" in side and position != "double",)
    assert chain.ops == []
    for s in "ab":
        ops = getattr(chain, f"mainloop_{s}_ops")
        expected = (["cos", "leaky_relu", "relu"] if position == "surrounded" else ["leaky_relu"]) if s in side else []
        assert [op.op for op in ops] == expected
        assert all(op.out_dtype == "bf16" and op.compute_dtype == "fp32" for op in ops)
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
        assert ".to(cutlass.BFloat16)" in snippets.mainloop_transform_a + snippets.mainloop_transform_b
        assert "mainloop_koob_fix = True" in src


@pytest.mark.parametrize("side", ["a", "b", "ab"])
@pytest.mark.parametrize("position", ["before", "after", "surrounded", "double"])
@pytest.mark.parametrize("swapped", [False, True])
def test_frost_transpose_original_axis_rebinding(monkeypatch, side, position, swapped):
    g, tensors = _graph(side=side, position=position)
    plan, launches = _mock_plan(monkeypatch, g, swapped)
    for m, n, k in ((128, 192, 136), (160, 224, 264)):
        buffers = _buffers(tensors, m, n, k)
        plan(_pack(tensors, buffers))
        frame = launches[-1]
        assert frame[0][:3] == ((n, m, k) if swapped else (m, n, k))
        ordered = (("b", n), ("a", m)) if swapped else (("a", m), ("b", n))
        for i, (s, extent) in enumerate(ordered, 1):
            assert frame[i].shape == (extent, k, 1)
            assert frame[i].stride() == (k, 1, extent * k)
            assert frame[i].data_ptr() == buffers[s].data_ptr()
        assert frame[3].data_ptr() == buffers["y"].data_ptr()
        got = _evaluate_frame(frame, plan.chain)
        got = got.transpose(1, 2) if swapped else got
        torch.testing.assert_close(got.bfloat16(), _reference(buffers, side, position, False, torch.bfloat16), atol=0, rtol=0)
    assert len(launches) == 2


@pytest.mark.parametrize("position", ["before", "after", "surrounded", "double"])
@pytest.mark.parametrize("swapped", [False, True])
def test_frost_transpose_per_k_binding(monkeypatch, position, swapped):
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


def test_frost_transpose_shared_input_and_binding_keys(monkeypatch):
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


def test_frost_transpose_dtype_inheritance():
    g, _ = _graph()
    g.validate()
    assert analyze(g).mainloop_a_ops[0].out_dtype == "bf16"
    nodes = [n for n in g._nodes if n.node_type.name == "TRANSPOSE"]
    assert all(n.outputs["OUT_0"].get_data_type() == BF16 for n in nodes)


@pytest.mark.parametrize(
    "problem,reason", [("permutation", "rank-3|last-two"), ("dtype", "dtype changes"), ("shape", "expected output shape"), ("output", "materialized")]
)
def test_frost_transpose_rejection(problem, reason):
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


def test_frost_transpose_epilogue_rejected():
    g, tensors = _graph(side="a")
    tensors["y"].set_output(False)
    g.transpose(input=tensors["y"], permutation=[0, 2, 1]).set_output(True)
    with pytest.raises(ValueError, match="operand-only"):
        analyze(g)


def _reference(buffers, side, position, scaled, dtype):
    values = {}
    for s in "ab":
        value = buffers[s]
        if s in side:
            if position in ("before", "double"):
                value = value.transpose(1, 2)
            if position == "surrounded":
                value = torch.cos(value.float()).to(dtype)
            if scaled:
                value = (buffers[f"scale_{s}"].float() * value.float()).to(dtype)
            value = torch.where(value.float() >= 0, value.float(), value.float() * 0.125).to(dtype)
            if position != "before":
                value = value.transpose(1, 2)
            if position == "surrounded":
                value = torch.relu(value)
        values[s] = value
    return (values["a"].float() @ values["b"].float()).to(dtype)


@requires_sm100
@pytest.mark.parametrize(
    "side,position,scaled,swapped,dtype,batch",
    [
        ("a", "after", False, False, BF16, 1),
        ("b", "before", False, True, BF16, 1),
        ("ab", "surrounded", True, False, BF16, 2),
        ("ab", "after", True, True, cudnn.data_type.HALF, 1),
        ("ab", "double", False, True, BF16, 1),
    ],
)
def test_frost_transpose_gpu(side, position, scaled, swapped, dtype, batch):
    g, tensors = _graph(side=side, position=position, scaled=scaled, dtype=dtype, batch=batch)
    plan = compiler.jit_from_cudnn_graph(g, _config(swapped))
    tdtype = torch.bfloat16 if dtype == BF16 else torch.float16
    torch.manual_seed(42)
    stream = torch.cuda.Stream()
    for m, n, k in ((128, 192, 136), (160, 224, 264)):
        buffers = _buffers(tensors, m, n, k, batch=batch, device="cuda", dtype=tdtype)
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
                torch.testing.assert_close(buffers["y"], _reference(buffers, side, position, scaled, tdtype), atol=0.25, rtol=0.02)
                previous = buffers["y"].clone()
                buffers["y"].fill_(float("nan"))
                captured.replay()
                assert torch.equal(buffers["y"], previous)
            # Exact K indexing after transforms, and exact cos(0)=1 for the K-tail case.
            if position == "surrounded":
                buffers["a"].zero_()
                buffers["b"].zero_()
            else:
                av = buffers["a"].transpose(1, 2) if tensors["a"].get_stride()[1] == 1 else buffers["a"]
                bv = buffers["b"].transpose(1, 2) if tensors["b"].get_stride()[1] == 1 else buffers["b"]
                av.copy_((torch.arange(k, device="cuda") % 17 - 8).to(tdtype).reshape(1, 1, k).expand_as(av))
                bv.zero_()
                bv[:, torch.arange(n, device="cuda"), torch.arange(n, device="cuda") * 5 % k] = 1
            if scaled:
                for s in side:
                    buffers[f"scale_{s}"].copy_((torch.arange(k, device="cuda") % 5 - 2).reshape(buffers[f"scale_{s}"].shape))
            buffers["y"].fill_(float("nan"))
            captured.replay()
            torch.testing.assert_close(buffers["y"], _reference(buffers, side, position, scaled, tdtype), atol=0, rtol=0)
        finally:
            captured.reset()


@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("raw", [False, True])
def test_frost_transpose_native_transport_and_overrides(monkeypatch, swapped, ordered, raw):
    """The actual public binder carries original graph coordinates and live strides."""
    from cudnn.engines import ExecutionContext
    from cudnn.gemm.frost.engine import _FrostGemmPlan

    g, tensors = _graph()
    compiled, launches = _mock_plan(monkeypatch, g, swapped)
    engine_plan = _FrostGemmPlan(compiled)
    # No kernel reads these CPU buffers; native views are inspected by the real lowered host.
    buffers = _buffers(tensors)
    uids = [t.get_uid() for t in tensors.values()]
    values = [buffers[key].data_ptr() if raw else buffers[key] for key in tensors]
    # Smaller logical M/N/K inside the original allocation; retain physical row pitches.
    shapes = [[1, 64, 128], [1, 128, 64], [1, 128, 128]]
    strides = [list(buffers[key].stride()) for key in tensors]
    normalize = g._normalize_ordered if ordered else g._normalize
    if ordered:
        pack = normalize(values, uids, None, uids, shapes, strides)
    else:
        pack = normalize(dict(zip(uids, values)), None, uids, shapes, strides)
    engine_plan.execute(g, pack, ExecutionContext(stream=0))
    frame = launches[-1]
    assert frame[0][:3] == (128, 128, 64)
    for i, key in enumerate(("b", "a") if swapped else ("a", "b"), 1):
        assert tuple(frame[i].shape) == (128, 64, 1)
        assert frame[i].data_ptr() == buffers[key].data_ptr()
        assert frame[i].stride()[1] == 1


def test_frost_transpose_identity_pure_and_mixed_rounding():
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
def test_frost_transpose_scalar_and_shared_scale(monkeypatch, scalar_first):
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
def test_frost_transpose_scope_rejection(problem):
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
def test_frost_transpose_workspace_uses_original_axes(monkeypatch, swapped):
    g, tensors = _graph()
    plan, _ = _mock_plan(monkeypatch, g, swapped)
    plan = replace(plan, config=replace(plan.config, split_k_slices=2))
    buffers = _buffers(tensors, 160, 224, 264)
    ordered = [buffers[next(key for key, tensor in tensors.items() if tensor is t)] for t in plan.bound]
    expected = 2 * 160 * 224 * 4
    assert plan.workspace_bytes_for_operands(ordered) == expected
    assert plan.workspace_bytes_for_overrides([t.get_uid() for t in tensors.values()], [list(buffers[key].shape) for key in tensors]) == expected


@pytest.mark.parametrize("swapped", [False, True])
def test_frost_transpose_invalid_input_preserves_output(monkeypatch, swapped):
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
def test_frost_pure_transpose_layout_binding(monkeypatch, m_major, swapped):
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
def test_frost_transpose_keeps_untransposed_epilogue_branch(monkeypatch, swapped):
    """A shared pre-transpose pointwise node remains in its original coordinates."""
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    a = g.tensor(name="A", dim=[1, 128, 128], stride=[16384, 1, 128])
    b = g.tensor(name="B", dim=[1, 128, 128], stride=[16384, 1, 128])
    pointwise = g.leaky_relu(input=a, negative_slope=0.125).set_data_type(BF16)
    transposed = g.transpose(input=pointwise, permutation=[0, 2, 1])
    c = g.matmul(A=transposed, B=b)
    y = g.add(a=c, b=pointwise).set_output(True)
    chain, binding = analyze_with_binding(g)
    assert [op.op for op in chain.ops] == ["aux_load", "leaky_relu", "add"]
    assert chain.ops[1].attrs == chain.mainloop_a_ops[0].attrs
    assert binding.aux == [a] and binding.aux_transposed == (False,)
    assert binding.bound_tensors() == [a, b, y]
    plan, launches = _mock_plan(monkeypatch, g, swapped)
    av = torch.randn(1, 128, 128, dtype=torch.bfloat16).transpose(1, 2)
    bv = torch.randn(1, 128, 128, dtype=torch.bfloat16).transpose(1, 2)
    plan({a: av, b: bv, y: torch.empty_like(av)})
    aux = launches[-1][4]
    assert aux.data_ptr() == av.data_ptr()
    assert tuple(aux.stride()) == tuple(av.transpose(1, 2).stride() if swapped else av.stride())


def test_frost_transpose_shared_epilogue_view_rejected():
    g, tensors = _graph(side="a")
    transpose = next(n for n in g._nodes if n.node_type.name == "TRANSPOSE").outputs["OUT_0"]
    tensors["y"].set_output(False)
    g.add(a=tensors["y"], b=transpose).set_output(True)
    with pytest.raises(ValueError, match="not a graph input|do not broadcast"):
        analyze(g)
