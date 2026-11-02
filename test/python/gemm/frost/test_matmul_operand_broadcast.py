# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank-independent operand broadcasts: logical maps, bindings and actual fragments."""

import ast
import importlib.util
import textwrap
from itertools import product
from math import prod
from pathlib import Path
import warnings

from gemm_test_utils import requires_sm100

import cudnn
import pytest
import torch

from cudnn.engines import ExecutionContext
from cudnn.gemm.frost.engine import _FrostGemmPlan
from cudnn.gemm.frost.fusion_ir import Axis, swap_ab
from cudnn.gemm.frost.graph_analyzer import analyze_with_binding, swap_ab_binding
from cudnn.gemm.frost.sm100 import compiler, epilogue_codegen
from cudnn.gemm.frost.tile_config import by_name

BF16, FP16, FP32 = cudnn.data_type.BFLOAT16, cudnn.data_type.HALF, cudnn.data_type.FLOAT
_DTYPE = {BF16: torch.bfloat16, FP16: torch.float16, FP32: torch.float32}


def _view(value, mode, side):
    if mode == "transpose":
        return value.transpose(-1, -2)
    if mode == "reshape":
        return value.reshape(value.shape[0], prod(value.shape[1:-1]), value.shape[-1])
    return value


def _graph(side, prefix, aux_shape, mode="plain", op="add", dtype=BF16, mn=8, k=16, aux_first=False, aux_dtype=FP32):
    g = cudnn.pygraph(io_data_type=dtype, intermediate_data_type=FP32, compute_data_type=FP32)
    shape = (*prefix, mn, k) if side == "a" else (*prefix, k, mn)
    if mode == "transpose":
        shape = (*shape[:-2], shape[-1], shape[-2])
    stride = [prod(shape[i + 1 :]) for i in range(len(shape))]
    if side == "b" and mode != "transpose":
        stride[-2:] = [1, shape[-2]]
    x = g.tensor(name="x", dim=list(shape), stride=stride, data_type=dtype)
    u = g.tensor(name="u", dim=list(aux_shape), stride=[2 * prod(aux_shape[i + 1 :]) for i in range(len(aux_shape))], data_type=aux_dtype)
    value = getattr(g, op)(a=u if aux_first else x, b=x if aux_first else u).set_data_type(dtype)
    if mode == "transpose":
        value = g.transpose(input=value, permutation=[*range(len(shape) - 2), len(shape) - 1, len(shape) - 2])
        target = (*shape[:-2], shape[-1], shape[-2])
    elif mode == "reshape":
        assert side == "a"
        target = (shape[0], prod(shape[1:-1]), shape[-1])
        value = g.reshape(input=value).set_dim(list(target))
    else:
        target = shape
    partner = (*target[:-2], target[-1], 24) if side == "a" else (*target[:-2], 24, target[-2])
    ps = [prod(partner[i + 1 :]) for i in range(len(partner))]
    if side == "a":
        ps[-2:] = [1, partner[-2]]
    other = g.tensor(name="other", dim=list(partner), stride=ps, data_type=dtype)
    y = g.matmul(A=value if side == "a" else other, B=other if side == "a" else value).set_output(True)
    g.validate()
    return g, {"x": x, "u": u, "other": other, "y": y}


def _config(swapped=False, cta_group=1, split=1):
    from dataclasses import replace

    return replace(by_name(f"CONFIG_sm100_128x128x64_128x128x32_cluster{cta_group}x1_{cta_group}ctamma"), swap_ab=swapped, split_k_slices=split)


@pytest.mark.L0
@pytest.mark.parametrize("side,prefix,pattern,swapped", list(product(("a", "b"), ((), (2,), (2, 3, 2)), ("mn", "k", "batch", "full"), (False, True))))
def test_frost_operand_broadcast_maps_and_rebinding(monkeypatch, side, prefix, pattern, swapped):
    shape = (*prefix, 8, 16) if side == "a" else (*prefix, 16, 8)
    aux = {"mn": (8, 1) if side == "a" else (1, 8), "k": (16,) if side == "a" else (16, 1), "batch": (*prefix, 1, 1), "full": shape[-2:]}[pattern]
    if pattern == "batch" and not prefix:
        pytest.skip("unit scalar already uses its existing constant-address route")
    g, tensors = _graph(side, prefix, aux, aux_first=True, op="sub")
    chain, binding = analyze_with_binding(g)
    if swapped:
        chain, binding = swap_ab(chain), swap_ab_binding(binding)
    operand = ("b" if side == "a" else "a") if swapped else side
    op = getattr(chain, f"mainloop_{operand}_ops")[0]
    assert op.operand_access is not None and not op.aux_on_rhs
    ref = chain.aux_by_name(op.aux)
    assert ref.dim == aux and ref.coordinate_mapped
    assert swap_ab(swap_ab(chain)).mainloop_a_ops == chain.mainloop_a_ops
    values = torch.arange(prod(aux), dtype=torch.int64).reshape(aux)
    expanded = values.expand(shape)
    expected = expanded if side == "a" else expanded.transpose(-1, -2)
    expected = expected.reshape(prod(prefix), 8, 16)
    row_axis = Axis.M if operand == "a" else Axis.N
    for b, row, k in product(range(prod(prefix)), (0, 7), (0, 15)):
        coords = tuple(expr.evaluate({Axis.BATCH: b, row_axis: row, Axis.K: k}) for expr in op.operand_access.coordinates)
        assert values[coords] == expected[b, row, k]
    buffers = {key: torch.empty_strided(t.get_dim(), t.get_stride(), dtype=_DTYPE[t.get_data_type()]) for key, t in tensors.items()}
    monkeypatch.setattr(compiler, "_check_plan_device", lambda _: None)
    frames = []
    compiled = compiler.CompiledFusedGemm(
        chain, _config(swapped), [ref.name], Path("unused"), lambda *args, **kwargs: frames.append(args), binding=binding, vec_bytes_epi=16
    )
    engine = _FrostGemmPlan(compiled)

    def pack(bufs):
        return g._normalize(
            {t.get_uid(): bufs[key] for key, t in tensors.items()},
            None,
            [t.get_uid() for t in tensors.values()],
            [list(bufs[key].shape) for key in tensors],
            [list(bufs[key].stride()) for key in tensors],
        )

    engine.execute(g, pack(buffers), ExecutionContext(stream=0))
    assert frames[-1][-1].data_ptr() == buffers["u"].data_ptr()
    broadcast_buffer = torch.randn((), dtype=torch.float32).expand(aux)
    engine.execute(g, pack(dict(buffers, u=broadcast_buffer)), ExecutionContext(stream=0))
    assert frames[-1][-1].data_ptr() == broadcast_buffer.data_ptr()
    with pytest.raises(ValueError, match="operand aux"):
        engine.execute(g, pack(dict(buffers, u=buffers["u"].to(torch.float16))), ExecutionContext(stream=0))
    assert len(frames) == 2
    source = epilogue_codegen.generate_mainloop(chain, operand)
    ast.parse(source)
    assert ".stride[" in source
    for slots in (frozenset(), frozenset({0})):
        snippets = epilogue_codegen.generate(chain, tma_slots=slots)
        assert f"_auxt_{ref.name}" not in snippets.epilogue


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for generated broadcast fragments")
@pytest.mark.parametrize(
    "side,prefix,aux,mode,op,dtype,aux_dtype",
    [
        ("a", (2, 3), (136, 1), "plain", "add", BF16, BF16),
        ("b", (2, 3), (1, 136), "plain", "sub", FP16, FP16),
        ("a", (2, 3), (1, 3, 1, 136), "reshape", "mul", BF16, FP32),
        ("a", (2, 3), (1, 3, 1, 1), "reshape", "add", FP16, FP32),
        ("a", (2, 3), (136,), "plain", "sub", BF16, FP32),
        ("a", (), (136, 136), "plain", "add", BF16, FP32),
        ("b", (1,), (136, 1), "plain", "mul", FP16, FP32),
        ("a", (2, 3), (136,), "transpose", "add", BF16, FP32),
    ],
)
def test_frost_operand_broadcast_generated_fragment(tmp_path, side, prefix, aux, mode, op, dtype, aux_dtype):
    """Execute mapped loads with source strides, view composition, rounding and K tails."""
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("generated fragments require SM90+")
    g, tensors = _graph(side, prefix, aux, mode, op, dtype, mn=136, k=136, aux_dtype=aux_dtype)
    chain, _ = analyze_with_binding(g)
    ref = chain.aux_tensors[0]
    snippet = epilogue_codegen.generate_mainloop(chain, side)
    mn = chain.matmul.M if side == "a" else chain.matmul.N
    dt = compiler.DTYPE_TO_CUTLASS[getattr(chain.matmul, f"{side}_dtype")]
    major = chain.matmul.a_major == "m" if side == "a" else chain.matmul.b_major == "n"
    fake = compiler._aux_fake_block([ref], dynamic_strides=True, align_reqs={ref.name: 4})
    source = f"""import cutlass
import cutlass.cute as cute
cta_tile_mnk = (128, 128, 64)
a_tma_group_elems = 64
b_tma_group_elems = 64
mainloop_koob_fix = True
a_is_m_major = {chain.matmul.a_major == "m"}
b_is_n_major = {chain.matmul.b_major == "n"}
@cute.kernel
def kernel(X: cute.Tensor, u: cute.Tensor, C: cute.Tensor, batch: cutlass.Int64, origin: cutlass.Int64, ktile: cutlass.Int64):
    tid = cute.arch.thread_idx()[0]
    ml_vec_elems = 8
    ml_coord_m, ml_coord_n = origin, origin
    ml_batch = batch
    k_tile_idx = ktile
    k = cutlass.Int64(136)
    m, n = cutlass.Int64({mn}), cutlass.Int64({mn})
    for chunk in cutlass.range_constexpr(8):
        ml_L = (tid + chunk * 128) * 8
        ml_vec_{side} = (X.iterator.raw_ptr() + ml_L).load(count=8, alignment=16)
{textwrap.indent(snippet, '        ')}
        (C.iterator.raw_ptr() + ml_L).store(ml_out_{side}, alignment=16)
@cute.jit
def entry(X: cute.Tensor, u: cute.Tensor, C: cute.Tensor, batch: cutlass.Int64, origin: cutlass.Int64, ktile: cutlass.Int64):
    kernel(X, u, C, batch, origin, ktile).launch(grid=(1,1,1), block=(128,1,1))
sym_k = 136
{fake}
fake_data = cute.runtime.make_fake_tensor({dt}, (8192,), stride=(1,), assumed_align=16)
compiled = cute.compile(entry, fake_data, fake_u, fake_data, cutlass.Int64(0), cutlass.Int64(0), cutlass.Int64(0), options="--enable-tvm-ffi")
"""
    path = tmp_path / "broadcast_fragment.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("broadcast_fragment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    x = torch.empty_strided(tensors["x"].get_dim(), tensors["x"].get_stride(), device="cuda", dtype=_DTYPE[dtype]).normal_()
    u = torch.empty_strided(aux, tensors["u"].get_stride(), device="cuda", dtype=_DTYPE[aux_dtype]).normal_()
    result = {"add": torch.add, "sub": torch.sub, "mul": torch.mul}[op](x.float(), u).to(x.dtype)
    raw, result = _view(x, mode, side), _view(result, mode, side)
    if side == "b":
        raw, result = raw.transpose(-1, -2), result.transpose(-1, -2)
    raw, result = raw.reshape(-1, mn, 136), result.reshape(-1, mn, 136)

    def tile(value, b, origin, ktile):
        packed = torch.zeros((128, 64), device="cuda", dtype=x.dtype)
        start, count = ktile * 64, min(64, 136 - ktile * 64)
        rows = min(128, value.shape[1] - origin)
        packed[:rows, :count] = value[b, origin : origin + rows, start : start + count]
        return packed.reshape(2, 64, 64).permute(0, 2, 1).contiguous().flatten() if major else packed.flatten()

    for b, origin, ktile in ((0, 0, 0), (raw.shape[0] - 1, mn // 128 * 128, 2)):
        data, expected = tile(raw, b, origin, ktile), tile(result, b, origin, ktile)
        output = torch.full_like(data, float("nan"))
        module.compiled(data, compiler._reshape_aux_to_fake(u, ref), output, b, origin, ktile)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)


@pytest.mark.L0
@pytest.mark.parametrize("steps", ("transpose_reshape", "reshape_transpose", "consecutive"))
def test_frost_operand_broadcast_composed_views(steps):
    """Different view chains use the same consumer-to-aux composition."""
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=FP32, compute_data_type=FP32)
    shape, aux_shape = (2, 3, 8, 16), (1, 3, 1, 16)
    x = g.tensor(name="x", dim=list(shape), stride=[384, 128, 16, 1], data_type=BF16)
    u = g.tensor(name="u", dim=list(aux_shape), stride=[48, 16, 16, 1], data_type=FP32)
    value = g.add(a=x, b=u).set_data_type(BF16)
    # Reuse the same input at two nodes, retaining each node's rounding.
    value = g.mul(a=value, b=u).set_data_type(BF16)
    expected = torch.arange(prod(aux_shape)).reshape(aux_shape).expand(shape)
    if steps == "transpose_reshape":
        value = g.transpose(input=value, permutation=[0, 1, 3, 2])
        value = g.reshape(input=value).set_dim([6, 16, 8]).set_stride([128, 1, 16])
        expected = expected.transpose(-1, -2).reshape(6, 16, 8)
    else:
        if steps == "consecutive":
            value = g.reshape(input=value).set_dim([2, 24, 16])
        value = g.reshape(input=value).set_dim([6, 8, 16])
        value = g.transpose(input=value, permutation=[0, 2, 1])
        expected = expected.reshape(6, 8, 16).transpose(-1, -2)
    b = g.tensor(name="b", dim=[6, 8, 16], stride=[128, 1, 8], data_type=BF16)
    y = g.matmul(A=value, B=b)
    g.relu(input=y).set_data_type(BF16).set_output(True)
    g.validate()
    chain, _ = analyze_with_binding(g)
    assert len(chain.aux_tensors) == 1
    assert chain.ops[0].op == "relu"
    epilogue_codegen.generate(chain)
    source = torch.arange(prod(aux_shape)).reshape(aux_shape)
    for op in chain.mainloop_a_ops:
        assert op.operand_access is not None
        for batch, row, k in product(range(6), range(16), range(8)):
            coords = tuple(e.evaluate({Axis.BATCH: batch, Axis.M: row, Axis.K: k}) for e in op.operand_access.coordinates)
            assert source[coords] == expected[batch, row, k]


@pytest.mark.L0
@pytest.mark.parametrize(
    "stride,shape,m,k,error",
    [
        ((32, 2), (8, 1), 8, 16, None),
        ((0, 0), (8, 1), 8, 16, None),  # read-only overlapping storage is legal
        ((-1, 2), (8, 1), 8, 16, "nonnegative"),
        (((1 << 63) - 1, 2), (8, 1), 8, 16, "64-bit"),
        ((32, 2), (4, 1), 8, 16, "expected"),
        ((32, 2), (8, 1), 16, 16, "requires rebuilding"),
        ((32, 2), (8, 1), 8, 24, "requires rebuilding"),
    ],
)
def test_frost_operand_broadcast_storage_contract(stride, shape, m, k, error):
    from types import SimpleNamespace
    from cudnn.gemm.frost.recipe import OperandAuxRule

    rule = OperandAuxRule((8, 1), "float32", ((False, (2, 8, 16)),), "bias")
    buffer = SimpleNamespace(shape=shape, stride=lambda: stride, dtype=torch.float32)
    if error is None:
        rule.check(buffer, k, m, 8)
    else:
        with pytest.raises(ValueError, match=error):
            rule.check(buffer, k, m, 8)


@requires_sm100
@pytest.mark.parametrize(
    "side,prefix,aux,mode,op,swapped,cta,split",
    [
        pytest.param("a", (2, 3), (128, 1), "plain", "add", False, 1, 1, marks=pytest.mark.L0),
        pytest.param("b", (2, 3), (1, 128), "plain", "sub", False, 1, 1, marks=pytest.mark.L0),
        pytest.param("a", (2, 3), (1, 3, 1, 136), "reshape", "mul", False, 1, 1, marks=pytest.mark.L0),
        pytest.param("a", (2, 3), (128,), "transpose", "add", True, 1, 1, marks=pytest.mark.L1),
        pytest.param("b", (1,), (136, 1), "plain", "mul", True, 2, 1, marks=pytest.mark.L1),
        pytest.param("a", (2, 3), (1, 3, 1, 1), "reshape", "add", False, 1, 2, marks=pytest.mark.L1),
    ],
)
def test_frost_operand_broadcast_gpu(side, prefix, aux, mode, op, swapped, cta, split):
    """Graph-coordinate bindings, mutated auxiliary replay and bitwise repeats."""
    dtype = FP16 if swapped else BF16
    g, tensors = _graph(side, prefix, aux, mode, op, dtype, mn=128, k=136)
    plan = compiler.jit_from_cudnn_graph(g, _config(swapped, cta, split))
    engine = _FrostGemmPlan(plan)
    workspace_size = engine.get_workspace_size()
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device="cuda") if workspace_size else None
    buffers = {key: torch.empty_strided(t.get_dim(), t.get_stride(), dtype=_DTYPE[t.get_data_type()], device="cuda") for key, t in tensors.items()}
    buffers["x"].fill_(0.25)
    buffers["other"].fill_(0.125)
    buffers["u"].fill_(0.5)
    # Normalize in graph coordinates, not the direct plan's canonical ABI.
    pack = g._normalize(
        {t.get_uid(): buffers[key] for key, t in tensors.items()},
        workspace,
        [t.get_uid() for t in tensors.values()],
        [list(buffers[key].shape) for key in tensors],
        [list(buffers[key].stride()) for key in tensors],
    )
    stream = torch.cuda.Stream()
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
        for fill in (0.75, 1.0):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                buffers["u"].fill_(fill)
                buffers["y"].fill_(float("nan"))
                captured.replay()
            stream.synchronize()
            transformed = {"add": torch.add, "sub": torch.sub, "mul": torch.mul}[op](buffers["x"].float(), buffers["u"]).to(_DTYPE[dtype])
            transformed = _view(transformed, mode, side)
            a, b = (transformed, buffers["other"]) if side == "a" else (buffers["other"], transformed)
            expected = (a.float() @ b.float()).to(_DTYPE[dtype])
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
