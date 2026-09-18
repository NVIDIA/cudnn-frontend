# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared K64 layout is an explicit graph contract and live native operand."""

from dataclasses import replace
from types import SimpleNamespace
import pytest
import cudnn
from cudnn import _pybind_module as native
from cudnn.gemm.frost.graph_analyzer import analyze_with_binding, resolve_variant_pack
from cudnn.gemm.frost.moe_pair import analyze_pair, PairedCompiled, KernelParams

pytestmark = pytest.mark.L0
LAYOUT = "k_blocked_64_v1"


def graph_case(*, r=9, n=64, k=128, layout=LAYOUT, reverse=False, mixed=False, bad_axis=False, bad_stride=False, extra=False):
    e = 4
    bf16, fp32 = cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT
    g = cudnn.pygraph(io_data_type=bf16, intermediate_data_type=fp32, compute_data_type=fp32)
    x = g.tensor(name="x", dim=[1, r, k], stride=[r * k, k, 1], data_type=bf16)
    strides = [2 * n * k, 2 * n * 64, 64, 1]
    if bad_stride:
        strides[0] += 8
    w = g.tensor(name="w", dim=[e, k // 64, 2 * n, 64], stride=strides, data_type=bf16)
    off = g.tensor(name="offsets", dim=[e, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    a = g.slice(w, [slice(None), slice(0, 1) if bad_axis else slice(None), slice(0, n), slice(None)]).set_stride(strides)
    b = g.slice(w, [slice(None), slice(None), slice(n, 2 * n), slice(None)]).set_stride(strides)
    gate, up = (b, a) if reverse else (a, b)
    mg = g.moe_grouped_matmul(x, gate, off, mode=cudnn.moe_grouped_matmul_mode.NONE, weight_layout=layout)
    mu = g.moe_grouped_matmul(x, up, off, mode=cudnn.moe_grouped_matmul_mode.NONE, weight_layout=None if mixed else layout)
    y = g.mul(g.swish(mg), mu).set_dim([1, r, n]).set_stride([r * n, n, 1]).set_data_type(bf16).set_output(True)
    if extra:
        gate.set_output(True)
    return g, (x, w, off, y, gate, up, mg)


def operand(pointer, shape, stride, code=4, bits=16):
    p = native.VariantPackNative(1)
    p.set_operand(0, pointer, list(shape), list(stride), code, bits)
    return p.operand(0, 0)


@pytest.mark.parametrize("r,n,k", [(1, 64, 64), (8, 192, 256), (9, 64, 128), (64, 768, 2048), (512, 768, 2048), (513, 128, 768)])
def test_k64_graph_native_contract(r, n, k):
    g, (x, w, off, y, gate, up, mg) = graph_case(r=r, n=n, k=k)
    spec = analyze_pair(g)
    chain, binding = analyze_with_binding(g)
    assert spec.weight_layout == chain.moe.weight_layout == LAYOUT
    assert (spec.rows, spec.features, spec.reduction) == (r, n, k)
    assert binding.bound_tensors() == [x, w, y, off]
    g.validate()
    assert list(mg.get_dim()) == [1, r, n]
    for pointer in (0x200000, 0x900000):
        parent = operand(pointer, w.get_dim(), w.get_stride())
        resolved = resolve_variant_pack({w: parent}, binding)
        assert resolved[id(gate)].data_ptr() == pointer
        assert resolved[id(up)].data_ptr() == pointer + n * 64 * 2
        calls = []
        compiled = PairedCompiled.__new__(PairedCompiled)
        compiled.spec = spec
        compiled.workspace_bytes = 38016
        compiled.kernel = lambda *args, **kwargs: calls.append((args, kwargs))
        pack = {
            x: operand(0x100000, x.get_dim(), x.get_stride()),
            w: parent,
            y: operand(0x500000, y.get_dim(), y.get_stride()),
            off: operand(0x800000, off.get_dim(), off.get_stride(), 0, 32),
        }
        workspace = SimpleNamespace(view=lambda *a: operand(0xA00000, (4752,), (1,), 0, 64))
        compiled(pack, workspace, stream=17)
        args, kwargs = calls.pop()
        assert args[4].data_ptr() == args[5].data_ptr() == pointer
        assert tuple(args[4].shape) == (2 * n, k, 4) and tuple(args[4].stride()) == (k, 1, 2 * n * k)
        assert int(kwargs["stream"]) == 17
        wrong = operand(pointer, [4, k, 2 * n], [2 * n * k, 1, k])
        with pytest.raises(ValueError, match="declared"):
            compiled({**pack, w: wrong}, workspace, stream=17)


@pytest.mark.parametrize(
    "kw",
    [
        dict(layout=None),
        dict(layout="unknown"),
        dict(reverse=True),
        dict(mixed=True),
        dict(bad_axis=True),
        dict(bad_stride=True),
        dict(extra=True),
        dict(r=514),
        dict(n=32),
    ],
)
def test_k64_declines_incomplete_contract(kw):
    with pytest.raises((NotImplementedError, ValueError)):
        analyze_pair(graph_case(**kw)[0])


def test_k64_ordinary_engines_decline_before_launch():
    from cudnn.gemm.frost.sm100.compiler import _check_executable as check100
    from cudnn.gemm.frost.sm120.compiler import _check_executable as check120

    chain, _ = analyze_with_binding(graph_case()[0])
    for check in (check100, check120):
        with pytest.raises(NotImplementedError, match="paired SwiGLU"):
            check(chain)


def test_k64_plan_cache_separates_layout(monkeypatch, tmp_path):
    from cudnn.frost import template_loader

    template = tmp_path / "layout.py"
    template.write_text("layout=FROST_TEMPLATE_PARAMS.weight_layout\ndef compile(): return object()\n")
    loader = template_loader.load_template
    seen = []

    def load(path, params, *, tag):
        seen.append(params)
        return loader(str(template), params, tag=tag)

    monkeypatch.setattr(template_loader, "load_template", load)
    params = KernelParams(148, 44214954, "helpers")
    plans = [PairedCompiled(SimpleNamespace(rows=9, binding=None, weight_layout=v), params) for v in (None, LAYOUT, None, LAYOUT)]
    assert plans[0].module is plans[2].module and plans[1].module is plans[3].module
    assert plans[0].module is not plans[1].module
    assert plans[0].module.FROST_SOURCE_DIGEST != plans[1].module.FROST_SOURCE_DIGEST
    assert params.weight_layout is None and [p.weight_layout for p in seen] == [None, LAYOUT, None, LAYOUT]
