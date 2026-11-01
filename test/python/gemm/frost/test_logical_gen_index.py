# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logical-coordinate IR and executable shared epilogue fragments, not MMA validation."""

import ast
import importlib.util
import textwrap
from itertools import product
from math import prod

import pytest
import torch

from cudnn.gemm.frost.epilogue_codegen import _FragmentCoordinates, _emit_op, _render_index_expr
from cudnn.gemm.frost.fusion_ir import Axis, FusionOp, IndexExpr
from cudnn.gemm.frost.graph_analyzer import matrix_axis_index


@pytest.mark.L0
@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4), (2, 3, 3, 4), (2, 1, 3, 3, 4), (1, 2, 1, 3, 3, 4)])
def test_frost_logical_index_coordinate_grid(shape):
    """Compare every axis to an independent Cartesian grid, including unit digits."""
    expressions = [matrix_axis_index(shape, axis) for axis in range(len(shape))]
    for logical in product(*(range(d) for d in shape)):
        batch = 0
        for digit, extent in zip(logical[:-2], shape[:-2]):
            batch = batch * extent + digit
        coordinates = {Axis.BATCH: batch, Axis.M: logical[-2], Axis.N: logical[-1]}
        assert tuple(expr.evaluate(coordinates) for expr in expressions) == logical
    # Only the batch prefix is a prepared constant: matrix coordinates remain live.
    for axis, coordinate in ((len(shape) - 2, Axis.M), (len(shape) - 1, Axis.N)):
        assert expressions[axis].evaluate({coordinate: 1001}) == 1001


@pytest.mark.L0
@pytest.mark.parametrize("shape,axis", [((), 0), ((2,), 0), ((2, 0), 0), ((2, 3), -1), ((2, 3), 2), ((1 << 62, 4, 2, 3), 0)])
def test_frost_logical_index_invalid_geometry(shape, axis):
    with pytest.raises(ValueError, match="gen_index"):
        matrix_axis_index(shape, axis)


@pytest.mark.L0
@pytest.mark.parametrize(
    "op,value,operand",
    [
        ("unknown", 1, None),
        ("axis", 1, None),
        ("constant", -1, None),
        ("constant", 1 << 63, None),
        ("mod", 0, IndexExpr.coordinate(Axis.M)),
        ("floor_div", 2, None),
    ],
)
def test_frost_logical_index_invalid_expression(op, value, operand):
    with pytest.raises(ValueError, match="index"):
        IndexExpr(op, value, operand)


@pytest.mark.L0
def test_frost_logical_index_composition_and_validation():
    m, n = IndexExpr.coordinate(Axis.M), IndexExpr.coordinate(Axis.N)
    expr = m.floor_div(3).modulo(5)
    substituted = expr.substitute({Axis.M: n})
    assert substituted.axes == {Axis.N}
    assert substituted.evaluate({Axis.N: 44}) == 4
    assert substituted.substitute({Axis.N: m}) == expr
    assert m.floor_div(1) is m
    assert m.modulo(1) == IndexExpr.constant(0)
    assert IndexExpr.constant(17).floor_div(3).modulo(4) == IndexExpr.constant(1)
    for value in (0, -1, 1 << 63):
        for method in (m.floor_div, m.modulo, IndexExpr.constant(0).floor_div, IndexExpr.constant(0).modulo):
            with pytest.raises(ValueError, match="Int64"):
                method(value)
    for value in (-1, 1 << 63):
        with pytest.raises(ValueError, match="Int64"):
            m.evaluate({Axis.M: value})
    with pytest.raises(ValueError, match="expression values"):
        m.substitute({Axis.M: 3})
    with pytest.raises(ValueError, match="only gen_index"):
        FusionOp("relu", index_expr=m)
    with pytest.raises(ValueError, match="only batch"):
        FusionOp("gen_index", index_expr=IndexExpr.coordinate(Axis.K))


@pytest.mark.L0
@pytest.mark.parametrize("axis", [1, 2])
def test_frost_logical_index_legacy_ir(axis):
    """Manually constructed rank-3 IR keeps the established axis attribute ABI."""
    legacy, _ = _emit_op(FusionOp("gen_index", attrs=(("axis", axis),)), "previous", 0, {}, vsize=4)
    mapped, _ = _emit_op(FusionOp("gen_index", index_expr=matrix_axis_index((2, 3, 4), axis)), "previous", 0, {}, vsize=4)
    assert ("row" if axis == 1 else "col_j") in "\n".join(legacy)
    ast.parse("\n".join(mapped))
    assert "Int64" in "\n".join(mapped)
    assert legacy == mapped


@pytest.mark.L0
@pytest.mark.parametrize("axis", [Axis.M, Axis.N, Axis.BATCH])
def test_frost_index_fragment_uniformity_is_layout_owned(axis):
    """A column vector reverses M/N uniformity; batch can vary too."""
    fragment = _FragmentCoordinates(((Axis.M, ("r", "r + 1", "r + 2", "r + 3")), (Axis.N, ("c",) * 4), (Axis.BATCH, ("l", "l + 1", "l + 2", "l + 3"))))
    expr = IndexExpr.coordinate(axis)
    lines, _ = _emit_op(FusionOp("gen_index", index_expr=expr), "previous", 0, {}, vsize=4, fragment_coordinates=fragment)
    assert fragment.is_uniform(expr) == (axis == Axis.N)
    assert ("full_like" in "\n".join(lines)) == (axis == Axis.N)
    assert ("make_rmem_tensor" in "\n".join(lines)) == (axis != Axis.N)
    assert fragment.is_uniform(IndexExpr.constant(0))
    # Symbol names are supplied by lowering, never embedded in the IR.
    assert _render_index_expr(expr.floor_div(3).modulo(5), {axis: "coordinate"}) == "((cutlass.Int64(coordinate) // 3) % 5)"


@pytest.mark.L0
@pytest.mark.parametrize(
    "axes", [(), ((Axis.M, ()),), ((Axis.M, ("r",)), (Axis.M, ("r",))), ((Axis.M, ("r",)), (Axis.N, ("c", "c + 1"))), (("M", ("r",)),), ((Axis.M, ("",)),)]
)
def test_frost_index_fragment_invalid_coordinates(axes):
    with pytest.raises(ValueError, match="fragment coordinates"):
        _FragmentCoordinates(axes)


@pytest.mark.L0
def test_frost_index_fragment_rejects_missing_axis_or_vector_size():
    fragment = _FragmentCoordinates(((Axis.M, ("r",) * 4),))
    with pytest.raises(ValueError, match="absent"):
        fragment.is_uniform(IndexExpr.coordinate(Axis.N))
    with pytest.raises(ValueError, match="size"):
        _emit_op(FusionOp("gen_index", index_expr=IndexExpr.coordinate(Axis.M)), "previous", 0, {}, vsize=8, fragment_coordinates=fragment)


def _test_fragment(kind, batch, size=4):
    """Explicit alternative register-vector coordinates, not new kernel layouts."""
    rows = tuple(f"row + {i if kind == 'column' else i // 2 if kind == 'mixed' else 0}" for i in range(size))
    cols = tuple(f"col_j + {0 if kind == 'column' else i % 2 if kind == 'mixed' else i}" for i in range(size))
    batches = tuple(f"(tile_l + {i}) % {batch}" if kind == "batch" else "tile_l" for i in range(size))
    return _FragmentCoordinates(((Axis.M, rows), (Axis.N, cols), (Axis.BATCH, batches)))


# Small arrays exercise large logical coordinates without enormous allocations.
_FRAGMENT_CASES = [
    ((2, 3, 7, 8), 0, 0),
    ((2, 3, 7, 8), 1, 0),
    ((2, 1, 3, 7, 8), 1, 0),
    ((2, 1, 3, 7, 8), 2, 0),
    ((7, 8), 0, 0),
    ((7, 8), 1, 0),
    (((1 << 25) + 64, 3, 7, 8), 0, 3 * ((1 << 24) + 1)),
    ((3, (1 << 40) + 3, 7, 8), 0, (1 << 40) + 4),
    ((3, (1 << 40) + 3, 7, 8), 1, (1 << 39) + 1),
]
_FRAGMENT_CASES = [(*case, "row") for case in _FRAGMENT_CASES] + [
    ((7, 8), 0, 0, "column"),
    ((7, 8), 1, 0, "column"),
    ((7, 8), 0, 0, "mixed"),
    ((7, 8), 1, 0, "mixed"),
    ((2, 3, 7, 8), 1, 0, "batch"),
]


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for emitted epilogue fragments")
@pytest.mark.parametrize("shape,axis,offset,fragment_kind", _FRAGMENT_CASES)
@pytest.mark.parametrize("compute_dtype", ["int32", "fp32"])
def test_frost_logical_index_generated_fragment(tmp_path, shape, axis, offset, fragment_kind, compute_dtype):
    """Execute the actual shared Int64 div/mod emitter on SM90+, then cast.

    This tests coordinate arithmetic and compute rounding, not the full Blackwell
    GEMM's scheduler, output transport or tensor-core instructions.
    """
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("generated epilogue fragment requires SM90+")
    op = FusionOp("gen_index", compute_dtype=compute_dtype, out_dtype=compute_dtype, index_expr=matrix_axis_index(shape, axis))
    fragment = None if fragment_kind == "row" else _test_fragment(fragment_kind, prod(shape[:-2]))
    lines, result = _emit_op(op, "previous", 0, {}, vsize=4, fragment_coordinates=fragment)
    dtype = "cutlass.Int32" if compute_dtype == "int32" else "cutlass.Float32"
    source = f"""import cutlass
import cutlass.cute as cute

@cute.kernel
def kernel(C: cute.Tensor, offset: cutlass.Int64):
    tid = cute.arch.thread_idx()[0]
    tile_l = (offset + cutlass.Int64(tid)) % {prod(shape[:-2])}
    row = offset + cutlass.Int64(tid)
    col_j = offset + cutlass.Int64(tid) * 4
    previous = cutlass.full((4,), 0, {dtype})
{textwrap.indent(chr(10).join(lines), '    ')}
    (C.iterator.raw_ptr() + tid * 4).store({result}, alignment=16)

@cute.jit
def entry(C: cute.Tensor, offset: cutlass.Int64):
    kernel(C, offset).launch(grid=(1, 1, 1), block=(32, 1, 1))

fake = cute.runtime.make_fake_tensor({dtype}, (128,), stride=(1,), assumed_align=16)
compiled = cute.compile(entry, fake, cutlass.Int64(0), options="--enable-tvm-ffi")
"""
    path = tmp_path / "index_fragment.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("logical_index_fragment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = torch.empty(128, device="cuda", dtype=torch.int32 if compute_dtype == "int32" else torch.float32)
    # Wide remainder remains within Int32 for this case, testing widening BEFORE mod.
    if shape[axis] > (1 << 32) and axis < len(shape) - 2:
        offset = prod(shape[axis + 1 : -2]) * shape[axis] + 17
    for shift in (0, 7):
        live_offset = offset + shift
        output.fill_(-1)
        module.compiled(output, live_offset)
        lanes = torch.arange(32, device="cuda", dtype=torch.int64) + live_offset
        element = torch.arange(128, device="cuda", dtype=torch.int64) % 4
        if axis == len(shape) - 1:
            expected = (lanes - live_offset).repeat_interleave(4) * 4 + live_offset
            expected += 0 if fragment_kind == "column" else element % 2 if fragment_kind == "mixed" else element
        elif axis == len(shape) - 2:
            expected = lanes.repeat_interleave(4)
            expected += element if fragment_kind == "column" else element // 2 if fragment_kind == "mixed" else 0
        else:
            batch = (lanes % prod(shape[:-2])).repeat_interleave(4)
            if fragment_kind == "batch":
                batch = (batch + element) % prod(shape[:-2])
            expected = (batch // prod(shape[axis + 1 : -2])) % shape[axis]
        if axis == 0 and shape[0] > (1 << 24):
            assert torch.all(expected > (1 << 24))
        torch.testing.assert_close(output, expected.to(output.dtype), atol=0, rtol=0)


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for the prepared broadcast entry")
@pytest.mark.parametrize("unit_axis", [1, 2])
def test_frost_index_broadcast_unit_extent_ffi_contract(tmp_path, unit_axis):
    """Exercise actual FFI unit-axis checks, which a launch-recorder mock omits."""
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("generated epilogue fragment requires SM90+")
    lines, result = _emit_op(FusionOp("gen_index", index_expr=IndexExpr.constant(0)), "previous", 0, {}, vsize=4)
    shape = "(1, 1, cute.sym_int64())" if unit_axis == 1 else "(1, cute.sym_int64(), 1)"
    stride = "(cute.sym_int64(), cute.sym_int64(), 1)" if unit_axis == 1 else "(cute.sym_int64(), 1, cute.sym_int64())"
    source = f"""import cutlass
import cutlass.cute as cute

@cute.kernel
def kernel(aux: cute.Tensor, C: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    previous = cutlass.full((4,), 0.0, cutlass.Float32)
{textwrap.indent(chr(10).join(lines), '    ')}
    (C.iterator.raw_ptr() + tid * 4).store({result}, alignment=16)

@cute.jit
def entry(aux: cute.Tensor, C: cute.Tensor):
    kernel(aux, C).launch(grid=(1, 1, 1), block=(32, 1, 1))

fake_aux = cute.runtime.make_fake_tensor(cutlass.BFloat16, {shape}, stride={stride}, assumed_align=16)
fake_output = cute.runtime.make_fake_tensor(cutlass.Float32, (128,), stride=(1,), assumed_align=16)
compiled = cute.compile(entry, fake_aux, fake_output, options="--enable-tvm-ffi")
"""
    path = tmp_path / "broadcast_index_entry.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("broadcast_index_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dims = (1, 1, 128) if unit_axis == 1 else (1, 128, 1)
    aux = torch.ones(dims, device="cuda", dtype=torch.bfloat16)
    output = torch.full((128,), float("nan"), device="cuda")
    module.compiled(aux, output)
    torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)
    output.fill_(123)
    with pytest.raises((ValueError, RuntimeError)):
        module.compiled(aux.expand(1, 128, 128), output)
    assert torch.all(output == 123)
    output.fill_(float("nan"))
    module.compiled(aux, output)
    torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)
