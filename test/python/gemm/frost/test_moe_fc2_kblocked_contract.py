# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared FC2 K64 layout: explicit declaration, live pointer, cache identity."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import cudnn
from cudnn import _pybind_module as native
from cudnn.gemm.frost.moe_fc2_pair import analyze_fc2, build_fc2, Fc2Compiled, Fc2Knobs
from cudnn.gemm.frost.moe_pair import KernelParams

pytestmark = pytest.mark.L0
LAYOUT = "k_blocked_64_v1"


def graph_case(*, e=4, r=9, n=256, k=128, layout=LAYOUT, pitched=False, wrong_rank=False):
    bf16, fp32 = cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT
    g = cudnn.pygraph(io_data_type=bf16, intermediate_data_type=fp32, compute_data_type=fp32)
    x = g.tensor(name="x", dim=[1, r, k], stride=[r * k, k, 1], data_type=bf16)
    dim, stride = [e, k // 64, n, 64], [n * k + (8 if pitched else 0), n * 64, 64, 1]
    if wrong_rank:
        dim, stride = [e, k, n], [n * k, 1, k]
    w = g.tensor(name="w", dim=dim, stride=stride, data_type=bf16)
    off = g.tensor(name="offsets", dim=[e, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    y = g.moe_grouped_matmul(x, w, off, mode=cudnn.moe_grouped_matmul_mode.NONE, weight_layout=layout)
    y.set_dim([1, r, n]).set_stride([r * n, n, 1]).set_data_type(bf16).set_output(True)
    return g


def operand(pointer, shape, stride, code=4, bits=16):
    pack = native.VariantPackNative(1)
    pack.set_operand(0, pointer, list(shape), list(stride), code, bits)
    return pack.operand(0, 0)


@pytest.mark.parametrize("e,r,n,k", [(1, 1, 128, 64), (4, 8, 256, 128), (4, 9, 384, 256), (128, 64, 2048, 768), (128, 512, 2048, 128), (257, 513, 256, 128)])
def test_fc2_k64_native_pointer_and_declaration(e, r, n, k):
    g = graph_case(e=e, r=r, n=n, k=k)
    spec = analyze_fc2(g)
    assert (spec.rows, spec.features, spec.reduction, spec.experts, spec.weight_layout) == (r, n, k, e, LAYOUT)
    g.validate()
    tensors = (spec.tokens, spec.weight, spec.output, spec.offsets)
    assert spec.binding.bound_tensors() == list(tensors)
    calls = []
    compiled = Fc2Compiled.__new__(Fc2Compiled)
    compiled.spec, compiled.workspace_bytes = spec, 38016
    compiled.kernel = lambda *args, **kwargs: calls.append((args, kwargs))
    pack = {t: operand((i + 1) * 0x100000, t.get_dim(), t.get_stride(), 0 if i == 3 else 4, 32 if i == 3 else 16) for i, t in enumerate(tensors)}
    workspace = SimpleNamespace(view=lambda *args: operand(0xA00000, (4752,), (1,), 0, 64))
    for address in (0x200000, 0x800000):
        pack[spec.weight] = operand(address, spec.weight.get_dim(), spec.weight.get_stride())
        compiled(pack, workspace, stream=17)
        args, kwargs = calls.pop()
        assert args[4].data_ptr() == args[5].data_ptr() == address
        assert tuple(args[4].shape) == (n, k, e) and tuple(args[4].stride()) == (k, 1, n * k)
        assert args[0][:5] == (r, n // 2, k, e, e)
        assert int(kwargs["stream"]) == 17
    # Canonical storage cannot be silently accepted as prepared K64 storage.
    pack[spec.weight] = operand(0x200000, [e, k, n], [n * k, 1, k])
    with pytest.raises(ValueError, match="declared"):
        compiled(pack, workspace, stream=17)
    assert not calls


@pytest.mark.parametrize(
    "options", [dict(layout=None), dict(layout="unknown"), dict(pitched=True), dict(wrong_rank=True), dict(r=0), dict(r=514), dict(n=64), dict(k=32)]
)
def test_fc2_k64_declines_incomplete_layout(options):
    with pytest.raises((ValueError, NotImplementedError)):
        analyze_fc2(graph_case(**options))


def test_fc2_k64_plan_cache_separates_layout_and_stages(monkeypatch, tmp_path):
    from cudnn.frost import template_loader

    template = tmp_path / "layout.py"
    template.write_text("layout=FROST_TEMPLATE_PARAMS.weight_layout\nstages=FROST_TEMPLATE_PARAMS.ab_stages\ndef compile(): return object()\n")
    loader = template_loader.load_template
    seen = []

    def load(path, params, *, tag):
        seen.append(params)
        return loader(str(template), params, tag=tag)

    monkeypatch.setattr(template_loader, "load_template", load)
    params = KernelParams(148, 44214954, "helpers")
    plans = []
    with patch("cudnn.gemm.frost.moe_fc2_pair.device_params", return_value=params):
        for _ in range(2):
            for layout in (None, LAYOUT):
                for stages in (12, 6):
                    plans.append(build_fc2(graph_case(layout=layout, wrong_rank=layout is None), Fc2Knobs(ab_stages=stages)))
    assert all(a.module is b.module for a, b in zip(plans[:4], plans[4:]))
    assert len({p.module.FROST_SOURCE_DIGEST for p in plans}) == 4
    assert [(p.weight_layout, p.ab_stages) for p in seen[:4]] == [(None, 12), (None, 6), (LAYOUT, 12), (LAYOUT, 6)]
    assert params.weight_layout is None and all(p.workspace_bytes == 38016 for p in plans)
