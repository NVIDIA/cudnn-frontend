# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank-2/rank-N matrix views: affine batch folding without storage conversion."""

import warnings
from dataclasses import replace
from math import prod

import cudnn
import pytest
import torch

from cudnn.engines import ExecutionContext
from cudnn.gemm.frost.engine import _FrostGemmPlan
from cudnn.gemm.frost.graph_analyzer import analyze, analyze_with_binding
from cudnn.gemm.frost.matrix_views import matrix_layout
from cudnn.gemm.frost.sm100 import compiler
from gemm_test_utils import requires_sm100
from test_matmul_transpose_mainloop import BF16, F32, _config, _mock_plan

pytestmark = pytest.mark.L0


def _strides(prefix, matrix_shape, matrix_strides, gap=0):
    pitch = max(d * s for d, s in zip(matrix_shape, matrix_strides)) + gap
    return [pitch * prod(prefix[i + 1 :]) for i in range(len(prefix))] + list(matrix_strides)


def _graph(prefix=(), *, transposed=True, scaled=False, broadcast=False, epilogue=False, gap=0, a_prefix=None, bias_prefix=()):
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    tensors, operands = {}, {}
    for side, extent in (("a", 128), ("b", 192)):
        batches = (a_prefix if a_prefix is not None else ((1,) * len(prefix) if broadcast else prefix)) if side == "a" else prefix
        # The originals remain in caller coordinates; both normalized inputs are K-major.
        dim = [136, extent] if (side == "a") == transposed else [extent, 136]
        stride = [1, 136] if (side == "a") == transposed else [136, 1]
        value = tensors[side] = g.tensor(name=side.upper(), dim=[*batches, *dim], stride=_strides(batches, dim, stride, gap))
        if scaled:
            scale_dim = [136, 1] if (side == "a") == transposed else [1, 136]
            scale = tensors[f"scale_{side}"] = g.tensor(name=f"scale_{side}", dim=scale_dim, stride=[scale_dim[1], 1], data_type=F32)
            value = g.mul(a=value, b=scale).set_data_type(BF16)
        value = g.leaky_relu(input=value, negative_slope=0.125).set_data_type(BF16)
        if transposed:
            permutation = [*range(len(prefix)), len(prefix) + 1, len(prefix)]
            value = g.transpose(input=value, permutation=permutation)
        operands[side] = value
    y = g.matmul(A=operands["a"], B=operands["b"])
    if epilogue:
        bias = tensors["bias"] = g.tensor(name="bias", dim=[*bias_prefix, 1, 192], stride=_strides(bias_prefix, (1, 192), (192, 1)))
        y = g.add(a=y, b=bias).set_data_type(BF16)
    tensors["y"] = y.set_output(True)
    g.validate()
    return g, tensors


def _buffers(tensors, m=128, n=192, k=136, *, device="cpu", gap=0):
    buffers = {}
    for side, extent in (("a", m), ("b", n)):
        t = tensors[side]
        prefix = tuple(t.get_dim()[:-2])
        matrix = (k, extent) if t.get_stride()[-2] == 1 else (extent, k)
        row_pitch = k + (8 if gap else 0)
        strides = (1, row_pitch) if matrix == (k, extent) else (row_pitch, 1)
        value = torch.empty_strided((*prefix, *matrix), _strides(prefix, matrix, strides, gap), dtype=torch.bfloat16, device=device)
        value.normal_()
        buffers[side] = value
        key = f"scale_{side}"
        if key in tensors:
            dim = (k, 1) if tensors[key].get_dim()[0] != 1 else (1, k)
            buffers[key] = torch.linspace(-1, 1, k, device=device).reshape(dim)
    prefix = tuple(tensors["y"].get_dim()[:-2])
    buffers["y"] = torch.zeros(*prefix, m, n, dtype=torch.bfloat16, device=device)
    if "bias" in tensors:
        buffers["bias"] = torch.randn(*tensors["bias"].get_dim()[:-2], 1, n, dtype=torch.bfloat16, device=device)
    return buffers


def _pack(tensors, buffers):
    return {t: buffers[key] for key, t in tensors.items()}


def _reference(buffers, transposed, scaled, epilogue=False):
    operands = {}
    for side in "ab":
        value = buffers[side]
        if scaled:
            value = (value.float() * buffers[f"scale_{side}"].float()).bfloat16()
        value = torch.where(value.float() >= 0, value.float(), value.float() * 0.125).bfloat16()
        operands[side] = value.transpose(-1, -2) if transposed else value
    value = (operands["a"].float() @ operands["b"].float()).bfloat16()
    if epilogue:
        value = (value.float() + buffers["bias"].float()).bfloat16()
    return value


@pytest.mark.parametrize("prefix", [(), (2, 3)])
@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("ordered", [False, True])
def test_frost_transpose_rank_native_overrides(monkeypatch, prefix, swapped, raw, ordered):
    g, tensors = _graph(prefix, scaled=True, gap=16)
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
def test_frost_transpose_rank_broadcast_and_epilogue(monkeypatch, swapped):
    g, tensors = _graph((2, 3), broadcast=True, epilogue=True)
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


def test_frost_transpose_rank_rejects_non_affine_batch(monkeypatch):
    g, tensors = _graph((2, 3))
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
def test_frost_transpose_rank_rejects_partial_batch_broadcast(prefix):
    g, _ = _graph(prefix, a_prefix=(2, 1))
    with pytest.raises(ValueError, match="partial/crossed"):
        analyze(g)


def test_frost_transpose_rank_rejects_batch_axis_permutation():
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    a = g.tensor(name="A", dim=[2, 2, 136, 128], stride=[2 * 136 * 128, 136 * 128, 1, 136])
    b = g.tensor(name="B", dim=[2, 2, 136, 192], stride=[2 * 136 * 192, 136 * 192, 1, 136])
    a = g.transpose(input=a, permutation=[1, 0, 3, 2])
    g.matmul(A=a, B=b).set_output(True)
    g.validate()  # Equal batch extents mask the permutation in the output shape, not in its semantics.
    with pytest.raises(ValueError, match="batch axes must stay fixed"):
        analyze(g)


def test_frost_transpose_rank_pins_batch_axes_and_rank(monkeypatch):
    g, tensors = _graph((2, 3))
    plan, launches = _mock_plan(monkeypatch, g)
    buffers = _buffers(tensors)
    for bad in (buffers["a"].transpose(0, 1), buffers["a"].view(6, 136, 128)):
        with pytest.raises(ValueError, match="original batch axes"):
            plan(_pack(tensors, dict(buffers, a=bad)))
        assert launches == []
    plan(_pack(tensors, buffers))
    assert len(launches) == 1


@pytest.mark.parametrize("prefix", [(), (2, 3)])
@pytest.mark.parametrize("swapped", [False, True])
def test_frost_transpose_rank_workspace_uses_original_axes(monkeypatch, prefix, swapped):
    g, tensors = _graph(prefix)
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


def test_frost_transpose_rank_rejects_overlapping_output(monkeypatch):
    g, tensors = _graph((2, 3))
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
def test_frost_transpose_rank_unit_aux_axes_and_recovery(monkeypatch, prefix, bias_prefix):
    g, tensors = _graph(prefix, epilogue=True, bias_prefix=bias_prefix)
    plan, launches = _mock_plan(monkeypatch, g)
    buffers = _buffers(tensors)
    bad = buffers["bias"].view(1, 192)
    with pytest.raises(ValueError, match="original batch axes"):
        plan(_pack(tensors, dict(buffers, bias=bad)))
    assert launches == [] and torch.count_nonzero(buffers["y"]) == 0
    plan(_pack(tensors, buffers))
    assert len(launches) == 1


@pytest.mark.parametrize("shape,stride", [((2, 3, 4, 8), (96, 32, 8, 1)), ((2, 3, 4, 8), (120, 40, 1, 4)), ((2, 1, 3, 4, 8), (120, 999, 40, 1, 4))])
def test_frost_matrix_batch_layout_contract(shape, stride):
    dim, st = matrix_layout(shape, stride)
    assert dim == (6, 4, 8)
    assert st[-2:] == stride[-2:]


def test_frost_matrix_batch_int64_boundaries():
    # Validate host widths without attempting an enormous allocation.
    dim, stride = matrix_layout((2, 2, 1, 1), (1 << 33, 1 << 32, 1, 1))
    assert dim == (4, 1, 1) and stride[0] == 1 << 32
    with pytest.raises(ValueError, match="64-bit"):
        matrix_layout((1 << 32, 1 << 32, 1, 1), (1 << 32, 1, 1, 1))


@pytest.mark.parametrize(
    "shape,stride",
    [
        ((2, 2, 1, 1), (1 << 33, 1 << 32, 1, 1)),
        ((2, 1, 3, 4, 8), (120, 999, 40, 1, 4)),
        ((1, 1, 4, 8), (500, 97, 8, 1)),
        ((2, 3, 4, 8), (160, 40, 1, 4)),
        ((1 << 32, 1 << 32, 1, 1), (1 << 32, 1, 1, 1)),
    ],
)
def test_frost_matrix_batch_native_geometry(shape, stride):
    from cudnn import _pybind_module

    # A synthetic address is inspected only as metadata; no kernel or DLPack consumer reads it.
    native = _pybind_module.VariantPackNative(1)
    native.set_operand(0, 4096, shape, stride, 4, 16, 1, -1, 2, 0)
    value = native.operand(0, 0)
    try:
        expected_shape, expected_stride = matrix_layout(shape, stride)
    except ValueError as error:
        with pytest.raises(ValueError, match="not affine|64-bit"):
            value.flatten_batch()
        assert "not affine" in str(error) or "64-bit" in str(error)
    else:
        folded = value.flatten_batch()
        assert tuple(folded.shape) == expected_shape and tuple(folded.stride()) == expected_stride
        assert folded.data_ptr() == value.data_ptr() == 4096
        assert folded.dtype == value.dtype and folded.element_size() == value.element_size()


@requires_sm100
@pytest.mark.parametrize(
    "prefix,swapped,scaled,broadcast,epilogue",
    [((), False, True, False, False), ((), True, False, False, True), ((2, 3), False, True, True, False), ((2, 1, 3), True, False, False, True)],
)
def test_frost_transpose_rank_gpu(prefix, swapped, scaled, broadcast, epilogue):
    g, tensors = _graph(prefix, scaled=scaled, broadcast=broadcast, epilogue=epilogue)
    plan = compiler.jit_from_cudnn_graph(g, _config(swapped))
    stream = torch.cuda.Stream()
    for m, n, k in ((128, 192, 136), (160, 224, 264)):
        buffers = _buffers(tensors, m, n, k, device="cuda")
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
                    buffers["scale_b"].neg_()
                buffers["y"].fill_(float("nan"))
                captured.replay()
                torch.testing.assert_close(buffers["y"], _reference(buffers, True, scaled, epilogue), atol=0.25, rtol=0.02)
                previous = buffers["y"].clone()
                buffers["y"].fill_(float("nan"))
                captured.replay()
                assert torch.equal(buffers["y"], previous)
        finally:
            captured.reset()
