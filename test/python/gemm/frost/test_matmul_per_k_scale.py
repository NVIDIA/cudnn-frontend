# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A-side per-K multiply: analyzer, logical-SMEM codegen, bindings and GPU parity."""

from dataclasses import replace
from pathlib import Path
import ast

import cudnn
import pytest
import torch

from cudnn.gemm.frost.graph_analyzer import analyze, analyze_with_binding
from cudnn.gemm.frost.fusion_ir import swap_ab
from cudnn.gemm.frost.sm100 import compiler, epilogue_codegen
from cudnn.gemm.frost.tile_config import by_name
from gemm_test_utils import requires_sm100

pytestmark = pytest.mark.L0
_CONFIG = "CONFIG_sm100_128x128x64_128x128x32_cluster1x1_1ctamma"


def _graph(
    M=128, N=128, K=136, *, aux_first=False, side="a", op="mul", batch=1, a_major="k", scale_stride=1, dtype=cudnn.data_type.BFLOAT16, surrounding=False
):
    g = cudnn.pygraph(io_data_type=dtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    a = g.tensor(name="A", dim=[batch, M, K], stride=[M * K, K, 1] if a_major == "k" else [M * K, 1, M])
    b = g.tensor(name="B", dim=[batch, K, N], stride=[K * N, 1, K])
    dim = [1, 1, K] if side == "a" else [1, K, 1]
    s = g.tensor(name="scale", dim=dim, stride=[K * scale_stride, K * scale_stride, scale_stride] if side == "a" else [K, 1, K])
    source = a if side == "a" else b
    if surrounding:
        source = g.cos(input=source).set_data_type(dtype)
    scaled = getattr(g, op)(a=s if aux_first else source, b=source if aux_first else s).set_data_type(dtype)
    if surrounding:
        scaled = g.relu(input=scaled).set_data_type(dtype)
    y = g.matmul(A=scaled if side == "a" else a, B=b if side == "a" else scaled).set_output(True)
    return g, (a, b, s, y)


@pytest.mark.parametrize("aux_first", [False, True])
@pytest.mark.parametrize("surrounding", [False, True])
def test_frost_per_k_analyzer(aux_first, surrounding):
    g, (a, b, s, y) = _graph(aux_first=aux_first, surrounding=surrounding)
    chain, binding = analyze_with_binding(g)
    assert tuple(binding.a_operands) == (a,)
    assert tuple(binding.b_operands) == (b,)
    assert tuple(binding.aux) == (s,)
    assert tuple(binding.outputs) == (y,)
    assert chain.ops == []
    assert [op.op for op in chain.mainloop_a_ops] == (["cos", "mul", "relu"] if surrounding else ["mul"])
    mul = next(op for op in chain.mainloop_a_ops if op.op == "mul")
    assert mul.aux_on_rhs is not aux_first
    assert (mul.compute_dtype, mul.out_dtype) == ("fp32", "bf16")
    assert chain.aux_by_name("scale").bcast_mode == "per_k"


@pytest.mark.parametrize(
    "kwargs", [{"side": "b"}, {"op": "add"}, {"batch": 2}, {"a_major": "m"}, {"scale_stride": 2}, {"dtype": cudnn.data_type.HALF}, {"K": 135}]
)
def test_frost_per_k_reject_unsupported(kwargs):
    with pytest.raises(ValueError, match="SCALAR|per-K mainloop scale"):
        analyze(_graph(**kwargs)[0])


def test_frost_per_k_ir_gates():
    chain = analyze(_graph()[0])
    with pytest.raises(ValueError, match="swap_ab"):
        swap_ab(chain)
    for change in ({"compute_dtype": "int32"}, {"out_dtype": "fp32"}, {"op": "add"}):
        with pytest.raises(ValueError, match="per-K mainloop scale"):
            replace(chain, mainloop_a_ops=[replace(chain.mainloop_a_ops[0], **change)])
    with pytest.raises(ValueError, match="mainloop-only"):
        replace(chain, ops=[chain.mainloop_a_ops[0]])


@pytest.mark.parametrize("cta_group", [1, 2])
def test_frost_per_k_render(cta_group):
    chain = analyze(_graph(surrounding=True)[0])
    cfg = by_name(_CONFIG if cta_group == 1 else "CONFIG_sm100_128x128x64_128x128x32_cluster2x1_2ctamma")
    tma_slots = compiler._tma_slots_for(chain, cfg)
    snippets = epilogue_codegen.generate(chain, vec_bytes_epi=compiler._epi_vec_bytes(chain, cfg), output_elem_bytes=2, tma_slots=tma_slots)
    src = compiler._render_template(chain, snippets, cfg)
    ast.parse(src)
    assert "@@INJECT_" not in src
    assert "mainloop_koob_fix = True" in src
    assert "fake_scale = cute.runtime.make_fake_tensor(cutlass.BFloat16, (1, 1, 136)" in src
    assert "k_tile_idx * cta_tile_mnk[2] + ml_L % cta_tile_mnk[2]" in src
    assert "if ml_a_1_gk < k:" in src
    assert "ml_ptr_a.load_swizzled" in src
    assert "_r_ml_a_1 = (_op_901).to(cutlass.BFloat16)" in src


@pytest.mark.parametrize("invalid", ["short", "long", "strided", "shape", "problem", "dtype"])
def test_frost_per_k_runtime_reject_before_launch(monkeypatch, invalid):
    g, (a, b, s, y) = _graph()
    chain, binding = analyze_with_binding(g)
    launches = []
    monkeypatch.setattr(compiler, "_check_plan_device", lambda device: None)
    plan = compiler.CompiledFusedGemm(
        chain, by_name(_CONFIG), ["scale"], Path("unused"), lambda *args, **kw: launches.append(args), binding=binding, vec_bytes_epi=16
    )
    av, bv, sv, out = (
        torch.ones(1, 128, 136, dtype=torch.bfloat16),
        torch.ones(1, 128, 136, dtype=torch.bfloat16),
        torch.ones(1, 1, 136, dtype=torch.bfloat16),
        torch.zeros(1, 128, 128, dtype=torch.bfloat16),
    )
    bad = sv
    if invalid in ("short", "long"):
        bad = torch.ones(1, 1, 128 if invalid == "short" else 144, dtype=torch.bfloat16)
    elif invalid == "strided":
        bad = torch.ones(1, 1, 272, dtype=torch.bfloat16)[..., ::2]
    elif invalid == "shape":
        bad = sv.reshape(1, 136, 1)
    elif invalid == "problem":
        av = torch.ones(1, 128, 144, dtype=torch.bfloat16)
        bv = torch.ones(1, 128, 144, dtype=torch.bfloat16)
    elif invalid == "dtype":
        bad = sv.to(torch.float16)
    with pytest.raises(ValueError, match="per-K mainloop scale"):
        plan({a: av, b: bv, s: bad, y: out})
    assert launches == []
    assert torch.count_nonzero(out) == 0
    plan({a: torch.ones(1, 128, 136, dtype=torch.bfloat16), b: torch.ones(1, 128, 136, dtype=torch.bfloat16), s: sv, y: out})
    assert len(launches) == 1


@requires_sm100
@pytest.mark.parametrize("cta_group", [1, 2])
@pytest.mark.parametrize("K", [128, 136, 264])
@pytest.mark.parametrize("surrounding", [False, True])
@pytest.mark.parametrize("aux_first", [False, True])
def test_frost_per_k_gpu(K, cta_group, surrounding, aux_first):
    g, (a, b, s, y) = _graph(K=K, surrounding=surrounding, aux_first=aux_first)
    cfg = by_name(_CONFIG if cta_group == 1 else "CONFIG_sm100_128x128x64_128x128x32_cluster2x1_2ctamma")
    plan = compiler.jit_from_cudnn_graph(g, cfg)
    torch.manual_seed(42)
    av = torch.randn(1, 128, K, device="cuda", dtype=torch.bfloat16)
    sv = ((torch.arange(K, device="cuda") % 17 - 8).float() / 7).reshape(1, 1, K).to(torch.bfloat16)
    bv = torch.randn(1, 128, K, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(1, 128, 128, device="cuda", dtype=torch.bfloat16)
    for scale in (sv, -sv):
        out.fill_(float("nan"))
        plan({a: av, b: bv, s: scale, y: out})
        transformed = torch.cos(av.float()).to(torch.bfloat16) if surrounding else av
        transformed = (transformed.float() * scale.float()).to(torch.bfloat16)
        if surrounding:
            transformed = torch.relu(transformed)
        ref = (transformed.float() @ bv.float().transpose(-1, -2)).to(torch.bfloat16)
        torch.testing.assert_close(out, ref, atol=0.25, rtol=0.02)
        assert torch.isfinite(out).all()


@requires_sm100
@pytest.mark.parametrize("cta_group", [1, 2])
@pytest.mark.parametrize("K", [136, 264])
def test_frost_per_k_exact_index_and_capture(K, cta_group):
    """One-hot B detects wrong K lanes/tile offsets without a tolerance budget."""
    g, (a, b, s, y) = _graph(K=K)
    cfg = by_name(_CONFIG if cta_group == 1 else "CONFIG_sm100_128x128x64_128x128x32_cluster2x1_2ctamma")
    plan = compiler.jit_from_cudnn_graph(g, cfg)
    av = torch.ones(1, 128, K, device="cuda", dtype=torch.bfloat16)
    selected_k = torch.arange(128, device="cuda") * 5 % K
    bv = torch.zeros(1, 128, K, device="cuda", dtype=torch.bfloat16)
    bv[0, torch.arange(128, device="cuda"), selected_k] = 1
    sv = (torch.arange(K, device="cuda") % 17 - 8).reshape(1, 1, K).to(torch.bfloat16)
    out = torch.empty(1, 128, 128, device="cuda", dtype=torch.bfloat16)
    pack = {a: av, b: bv, s: sv, y: out}
    plan(pack)
    torch.cuda.synchronize()
    captured = torch.cuda.CUDAGraph()
    with torch.cuda.graph(captured):
        plan(pack)
    for sign in (1, -1):
        sv.copy_((sign * (torch.arange(K, device="cuda") % 17 - 8)).reshape(1, 1, K))
        out.fill_(float("nan"))
        captured.replay()
        ref = sv[..., selected_k].expand(1, 128, 128)
        torch.testing.assert_close(out, ref, atol=0, rtol=0)
