# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-K multiply on both operands: analyzer, codegen, bindings and GPU parity."""

from dataclasses import replace
from pathlib import Path
import importlib.util
import textwrap
import ast

import cudnn
import pytest
import torch

from cudnn.gemm.frost.graph_analyzer import analyze, analyze_with_binding, swap_ab_binding
from cudnn.gemm.frost.fusion_ir import swap_ab
from cudnn.gemm.frost.kernel_registry import select_template
from cudnn.gemm.frost.sm100 import compiler, epilogue_codegen
from cudnn.gemm.frost.tile_config import by_name
from gemm_test_utils import requires_sm100

pytestmark = pytest.mark.L0
_CONFIG = "CONFIG_sm100_128x128x64_128x128x32_cluster1x1_1ctamma"


def _graph(
    M=128,
    N=128,
    K=136,
    *,
    aux_first=False,
    side="a",
    op="mul",
    batch=1,
    a_major="k",
    b_major="k",
    scale_stride=1,
    dtype=cudnn.data_type.BFLOAT16,
    surrounding=False,
):
    g = cudnn.pygraph(io_data_type=dtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    a = g.tensor(name="A", dim=[batch, M, K], stride=[M * K, K, 1] if a_major == "k" else [M * K, 1, M])
    b = g.tensor(name="B", dim=[batch, K, N], stride=[K * N, 1, K] if b_major == "k" else [K * N, N, 1])
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


@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize("aux_first", [False, True])
@pytest.mark.parametrize("surrounding", [False, True])
def test_frost_per_k_analyzer(side, aux_first, surrounding):
    g, (a, b, s, y) = _graph(side=side, aux_first=aux_first, surrounding=surrounding)
    chain, binding = analyze_with_binding(g)
    assert tuple(binding.a_operands) == (a,)
    assert tuple(binding.b_operands) == (b,)
    assert tuple(binding.aux) == (s,)
    assert tuple(binding.outputs) == (y,)
    assert chain.ops == []
    ops = getattr(chain, f"mainloop_{side}_ops")
    assert [op.op for op in ops] == (["cos", "mul", "relu"] if surrounding else ["mul"])
    mul = next(op for op in ops if op.op == "mul")
    assert mul.aux_on_rhs is not aux_first
    assert (mul.compute_dtype, mul.out_dtype) == ("fp32", "bf16")
    assert chain.aux_by_name("scale").bcast_mode == "per_k"


@pytest.mark.parametrize(
    "kwargs,reason",
    [({"a_major": "m"}, "K-major"), ({"side": "b", "b_major": "n"}, "K-major"), ({"scale_stride": 2}, "unit stride"), ({"K": 135}, "divisible by 8")],
)
def test_frost_per_k_template_rejects_not_ir(kwargs, reason):
    # Valid graph semantics are retained; the selected implementation declines them.
    chain = analyze(_graph(**kwargs)[0])
    assert reason in select_template(chain, by_name(_CONFIG))._extra_reject(chain, by_name(_CONFIG))


@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize("dtype", [cudnn.data_type.BFLOAT16, cudnn.data_type.HALF])
def test_frost_per_k_supported_dtypes_and_batch(side, dtype):
    chain = analyze(_graph(side=side, dtype=dtype, batch=2)[0])
    assert select_template(chain, by_name(_CONFIG))._extra_reject(chain, by_name(_CONFIG)) is None


def test_frost_per_k_ir_gates():
    chain = analyze(_graph()[0])
    for change, reason in (({"compute_dtype": "int32"}, "FP32"), ({"out_dtype": "fp32"}, "result"), ({"op": "add"}, "only mul")):
        other = replace(chain, mainloop_a_ops=[replace(chain.mainloop_a_ops[0], **change)])
        assert reason in select_template(other, by_name(_CONFIG))._extra_reject(other, by_name(_CONFIG))
    with pytest.raises(ValueError, match="mainloop-only"):
        replace(chain, ops=[chain.mainloop_a_ops[0]])
    with pytest.raises(ValueError, match="expected"):
        replace(chain, aux_tensors=[replace(chain.aux_tensors[0], dim=(1, 1, 144))])


@pytest.mark.parametrize("side", ["a", "b"])
def test_frost_per_k_swap_roundtrip(side):
    chain = analyze(_graph(side=side)[0])
    other = swap_ab(chain)
    assert getattr(other, f"mainloop_{'b' if side == 'a' else 'a'}_ops") == getattr(chain, f"mainloop_{side}_ops")
    assert other.aux_tensors[0].k_axis == 3 - chain.aux_tensors[0].k_axis
    assert select_template(other, by_name(_CONFIG))._extra_reject(other, by_name(_CONFIG)) is None
    restored = swap_ab(other)
    assert restored.matmul == chain.matmul
    assert restored.aux_tensors == chain.aux_tensors
    assert restored.mainloop_a_ops == chain.mainloop_a_ops
    assert restored.mainloop_b_ops == chain.mainloop_b_ops


@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize("cta_group", [1, 2])
def test_frost_per_k_render(cta_group, side):
    chain = analyze(_graph(side=side, surrounding=True)[0])
    cfg = by_name(_CONFIG if cta_group == 1 else "CONFIG_sm100_128x128x64_128x128x32_cluster2x1_2ctamma")
    tma_slots = compiler._tma_slots_for(chain, cfg)
    snippets = epilogue_codegen.generate(chain, vec_bytes_epi=compiler._epi_vec_bytes(chain, cfg), output_elem_bytes=2, tma_slots=tma_slots)
    src = compiler._render_template(chain, snippets, cfg)
    ast.parse(src)
    assert "@@INJECT_" not in src
    assert "mainloop_koob_fix = True" in src
    shape = "(1, 1, sym_k)" if side == "a" else "(1, sym_k, 1)"
    assert f"fake_scale = cute.runtime.make_fake_tensor(cutlass.BFloat16, {shape}" in src
    assert "k_tile_idx * cta_tile_mnk[2] + ml_L % cta_tile_mnk[2]" in src
    assert f"if ml_{side}_1_gk < k:" in src
    assert f"ml_ptr_{side}.load_swizzled" in src
    assert f"_r_ml_{side}_1" in src and ".to(cutlass.BFloat16)" in src


@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.parametrize("invalid", ["short", "long", "strided", "shape", "dtype"])
def test_frost_per_k_runtime_reject_before_launch(monkeypatch, invalid, side, swapped):
    g, (a, b, s, y) = _graph(side=side)
    chain, binding = analyze_with_binding(g)
    if swapped:
        chain, binding = swap_ab(chain), swap_ab_binding(binding)
    launches = []
    monkeypatch.setattr(compiler, "_check_plan_device", lambda device: None)
    plan = compiler.CompiledFusedGemm(
        chain,
        replace(by_name(_CONFIG), swap_ab=swapped),
        ["scale"],
        Path("unused"),
        lambda *args, **kw: launches.append(args),
        binding=binding,
        vec_bytes_epi=16,
    )
    av, bv, sv, out = (
        torch.ones(1, 128, 136, dtype=torch.bfloat16),
        torch.ones(1, 128, 136, dtype=torch.bfloat16),
        torch.ones(1, 1, 136, dtype=torch.bfloat16),
        torch.zeros(1, 128, 128, dtype=torch.bfloat16),
    )
    if side == "b":
        sv = sv.transpose(1, 2)
    bad = sv
    if invalid in ("short", "long"):
        bad = torch.ones(1, 1, 128 if invalid == "short" else 144, dtype=torch.bfloat16)
    elif invalid == "strided":
        bad = torch.ones(1, 1, 272, dtype=torch.bfloat16)[..., ::2]
    elif invalid == "shape":
        bad = sv.transpose(1, 2)
    elif invalid == "dtype":
        bad = sv.to(torch.float16)
    if side == "b" and invalid in ("short", "long", "strided"):
        bad = bad.transpose(1, 2)
    with pytest.raises(ValueError, match="per-K mainloop scale"):
        plan({a: av, b: bv, s: bad, y: out})
    assert launches == []
    assert torch.count_nonzero(out) == 0
    plan({a: torch.ones(1, 128, 136, dtype=torch.bfloat16), b: torch.ones(1, 128, 136, dtype=torch.bfloat16), s: sv, y: out})
    assert len(launches) == 1


@requires_sm100
@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.parametrize("cta_group", [1, 2])
@pytest.mark.parametrize("K", [128, 136, 264])
@pytest.mark.parametrize("surrounding", [False, True])
@pytest.mark.parametrize("aux_first", [False, True])
def test_frost_per_k_gpu(K, cta_group, surrounding, aux_first, side, swapped):
    g, (a, b, s, y) = _graph(K=K, side=side, surrounding=surrounding, aux_first=aux_first)
    cfg = by_name(_CONFIG if cta_group == 1 else "CONFIG_sm100_128x128x64_128x128x32_cluster2x1_2ctamma")
    plan = compiler.jit_from_cudnn_graph(g, replace(cfg, swap_ab=swapped))
    torch.manual_seed(42)
    av = torch.randn(1, 128, K, device="cuda", dtype=torch.bfloat16)
    sv = ((torch.arange(K, device="cuda") % 17 - 8).float() / 7).reshape(1, 1, K).to(torch.bfloat16)
    bv = torch.randn(1, 128, K, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(1, 128, 128, device="cuda", dtype=torch.bfloat16)
    for scale in (sv, -sv):
        out.fill_(float("nan"))
        plan({a: av, b: bv, s: scale if side == "a" else scale.transpose(1, 2), y: out})
        operand = av if side == "a" else bv
        transformed = torch.cos(operand.float()).to(torch.bfloat16) if surrounding else operand
        transformed = (transformed.float() * scale.float()).to(torch.bfloat16)
        if surrounding:
            transformed = torch.relu(transformed)
        aa, bb = (transformed, bv) if side == "a" else (av, transformed)
        ref = (aa.float() @ bb.float().transpose(-1, -2)).to(torch.bfloat16)
        torch.testing.assert_close(out, ref, atol=0.25, rtol=0.02)
        assert torch.isfinite(out).all()


@requires_sm100
@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.parametrize("cta_group", [1, 2])
@pytest.mark.parametrize("K", [136, 264])
def test_frost_per_k_exact_index_and_capture(K, cta_group, side, swapped):
    """One-hot opposite operand detects wrong K lanes/tile offsets exactly."""
    g, (a, b, s, y) = _graph(K=K, side=side)
    cfg = by_name(_CONFIG if cta_group == 1 else "CONFIG_sm100_128x128x64_128x128x32_cluster2x1_2ctamma")
    plan = compiler.jit_from_cudnn_graph(g, replace(cfg, swap_ab=swapped))
    av = torch.ones(1, 128, K, device="cuda", dtype=torch.bfloat16)
    selected_k = torch.arange(128, device="cuda") * 5 % K
    bv = torch.zeros(1, 128, K, device="cuda", dtype=torch.bfloat16)
    bv[0, torch.arange(128, device="cuda"), selected_k] = 1
    sv = (torch.arange(K, device="cuda") % 17 - 8).reshape(1, 1, K).to(torch.bfloat16)
    out = torch.empty(1, 128, 128, device="cuda", dtype=torch.bfloat16)
    if side == "b":
        av, bv = bv, av
    pack = {a: av, b: bv, s: sv if side == "a" else sv.transpose(1, 2), y: out}
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
        if side == "b":
            ref = ref.transpose(1, 2)
        torch.testing.assert_close(out, ref, atol=0, rtol=0)


@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize("swapped", [False, True])
def test_frost_per_k_runtime_dynamic_mnk(monkeypatch, side, swapped):
    g, (a, b, s, y) = _graph(side=side)
    chain, binding = analyze_with_binding(g)
    if swapped:
        chain, binding = swap_ab(chain), swap_ab_binding(binding)
    launches = []
    monkeypatch.setattr(compiler, "_check_plan_device", lambda device: None)
    plan = compiler.CompiledFusedGemm(
        chain,
        replace(by_name(_CONFIG), swap_ab=swapped),
        ["scale"],
        Path("unused"),
        lambda *args, **kw: launches.append(args),
        binding=binding,
        vec_bytes_epi=16,
    )
    for m, n, k in ((128, 128, 136), (160, 192, 264), (128, 128, 144)):
        sv = torch.ones((1, 1, k) if side == "a" else (1, k, 1), dtype=torch.bfloat16)
        plan({a: torch.ones(1, m, k, dtype=torch.bfloat16), b: torch.ones(1, n, k, dtype=torch.bfloat16), s: sv, y: torch.zeros(1, m, n, dtype=torch.bfloat16)})
        assert launches[-1][0][:3] == ((n, m, k) if swapped else (m, n, k))
        axis = 2 if side == "a" else 1
        if swapped:
            axis = 3 - axis
        assert launches[-1][4].shape[axis] == k
    assert len(launches) == 3


def _dual_graph(K=136, dtype=cudnn.data_type.BFLOAT16, repeated=False):
    """Different scale dtypes on both sides; optional second A-side multiply."""
    g = cudnn.pygraph(io_data_type=dtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    a = g.tensor(name="A", dim=[1, 128, K], stride=[128 * K, K, 1])
    b = g.tensor(name="B", dim=[1, K, 128], stride=[128 * K, 1, K])
    sa = g.tensor(name="scale_a", dim=[1, 1, K], stride=[K, K, 1], data_type=cudnn.data_type.FLOAT)
    sb = g.tensor(name="scale_b", dim=[1, K, 1], stride=[K, 1, K])
    aa = g.mul(a=a, b=sa).set_data_type(dtype)
    if repeated:
        aa = g.mul(a=aa, b=sa).set_data_type(dtype)
    bb = g.mul(a=sb, b=b).set_data_type(dtype)
    y = g.matmul(A=aa, B=bb).set_output(True)
    return g, (a, b, sa, sb, y)


@pytest.mark.parametrize("repeated", [False, True])
def test_frost_per_k_dual_scale_bindings(monkeypatch, repeated):
    g, (a, b, sa, sb, y) = _dual_graph(repeated=repeated)
    chain, binding = analyze_with_binding(g)
    assert [op.op for op in chain.mainloop_a_ops] == (["mul", "mul"] if repeated else ["mul"])
    assert [op.op for op in chain.mainloop_b_ops] == ["mul"]
    assert tuple(binding.aux) == (sa, sb)  # Repeated references do not duplicate bindings.
    assert select_template(chain, by_name(_CONFIG))._extra_reject(chain, by_name(_CONFIG)) is None
    launches = []
    monkeypatch.setattr(compiler, "_check_plan_device", lambda device: None)
    plan = compiler.CompiledFusedGemm(
        chain, by_name(_CONFIG), ["scale_a", "scale_b"], Path("unused"), lambda *args, **kw: launches.append(args), binding=binding, vec_bytes_epi=16
    )
    pack = {
        a: torch.ones(1, 128, 136, dtype=torch.bfloat16),
        b: torch.ones(1, 128, 136, dtype=torch.bfloat16),
        sa: torch.ones(1, 1, 136, dtype=torch.float32),
        sb: torch.ones(1, 136, 1, dtype=torch.bfloat16),
        y: torch.zeros(1, 128, 128, dtype=torch.bfloat16),
    }
    for scale in (sa, sb):
        invalid = dict(pack)
        invalid[scale] = pack[scale].to(torch.int32)
        with pytest.raises(ValueError, match="expected dtype"):
            plan(invalid)
        assert launches == []
    plan(pack)
    assert len(launches) == 1


@requires_sm100
@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_frost_per_k_both_operands_dynamic_capture(swapped, dtype):
    g, (a, b, sa, sb, y) = _dual_graph(dtype=cudnn.data_type.BFLOAT16 if dtype == torch.bfloat16 else cudnn.data_type.HALF, repeated=True)
    plan = compiler.jit_from_cudnn_graph(g, replace(by_name(_CONFIG), swap_ab=swapped))
    torch.manual_seed(4)
    for m, n, k in ((128, 128, 136), (160, 192, 264)):
        av, bv = torch.randn(1, m, k, device="cuda", dtype=dtype), torch.randn(1, n, k, device="cuda", dtype=dtype)
        sav = torch.linspace(-1.5, 1.5, k, device="cuda").reshape(1, 1, k)
        sbv = torch.linspace(0.7, 1.3, k, device="cuda").reshape(1, k, 1).to(dtype)
        out = torch.empty(1, m, n, device="cuda", dtype=dtype)
        pack = {a: av, b: bv, sa: sav, sb: sbv, y: out}
        plan(pack)
        torch.cuda.synchronize()
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            plan(pack)
        for sign in (1, -1):
            sbv.mul_(sign)
            out.fill_(float("nan"))
            captured.replay()
            aa = (av.float() * sav).to(dtype)
            aa = (aa.float() * sav).to(dtype)
            bb = (bv.float() * sbv.transpose(1, 2).float()).to(dtype)
            ref = (aa.float() @ bb.float().transpose(-1, -2)).to(dtype)
            torch.testing.assert_close(out, ref, atol=0.25, rtol=0.02)
            assert torch.isfinite(out).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU for the transform fragment")
@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize("operand_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scale_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_frost_per_k_generated_fragment_dynamic_k(tmp_path, side, operand_dtype, scale_dtype):
    """Run the emitted transform and dynamic fake ABI on Hopper or Blackwell.

    This isolates dtype, per-lane loads and K padding; it is NOT validation of
    the mainloop template's swizzle, synchronization or tensor-core MMA.
    """
    chain = analyze(_graph(side=side, dtype=cudnn.data_type.BFLOAT16 if operand_dtype == torch.bfloat16 else cudnn.data_type.HALF)[0])
    scale_type = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}[scale_dtype]
    chain = replace(chain, aux_tensors=[replace(chain.aux_tensors[0], dtype=scale_type)])
    snippet = epilogue_codegen.generate_mainloop(chain, side)
    fake = compiler._aux_fake_block(chain.aux_tensors, dynamic_strides=True)
    src = f"""import cutlass
import cutlass.cute as cute

mainloop_koob_fix = True
a_is_m_major = False
b_is_n_major = False
cta_tile_mnk = (128, 128, 64)

@cute.kernel
def kernel(A: cute.Tensor, scale: cute.Tensor, C: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    ml_vec_elems = 8
    k_tile_idx = (tid * 8) // 64
    ml_L = (tid * 8) % 64
    k = cutlass.Int64(scale.shape[{chain.aux_tensors[0].k_axis}])
    ml_vec_{side} = (A.iterator.raw_ptr() + tid * 8).load(count=8, alignment=16)
{textwrap.indent(snippet, '    ')}
    (C.iterator.raw_ptr() + tid * 8).store(ml_out_{side}, alignment=16)

@cute.jit
def entry(A: cute.Tensor, scale: cute.Tensor, C: cute.Tensor):
    kernel(A, scale, C).launch(grid=(1,1,1), block=(64,1,1))

sym_k = cute.sym_int64()
{fake}
fake_data = cute.runtime.make_fake_tensor({compiler.DTYPE_TO_CUTLASS[chain.matmul.a_dtype]}, (512,), stride=(1,), assumed_align=16)
compiled = cute.compile(entry, fake_data, fake_scale, fake_data, options="--enable-tvm-ffi")
"""
    path = tmp_path / f"fragment_{side}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(f"per_k_fragment_{side}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    av = torch.linspace(-3, 3, 512, device="cuda").to(operand_dtype)
    out = torch.empty_like(av)
    for k in (128, 136, 264):
        scale = ((torch.arange(k, device="cuda") % 13 - 6).float() / 7).to(scale_dtype)
        scale = scale.reshape((1, 1, k) if side == "a" else (1, k, 1))
        out.fill_(float("nan"))
        module.compiled(av, scale, out)
        ref = torch.zeros_like(out)
        ref[:k] = (av[:k].float() * scale.reshape(-1).float()).to(operand_dtype)
        torch.testing.assert_close(out, ref, atol=0, rtol=0)


@requires_sm100
@pytest.mark.parametrize("side", ["a", "b"])
def test_frost_per_k_batched_shared_scale(side):
    g, (a, b, s, y) = _graph(K=144, side=side, batch=2)
    plan = compiler.jit_from_cudnn_graph(g, by_name(_CONFIG))
    av = torch.ones(2, 128, 144, device="cuda", dtype=torch.bfloat16)
    av[1].mul_(2)
    bv = torch.ones_like(av)
    scale = (torch.arange(144, device="cuda") % 17 - 8).to(torch.bfloat16)
    scale = scale.reshape((1, 1, 144) if side == "a" else (1, 144, 1))
    out = torch.empty(2, 128, 128, device="cuda", dtype=torch.bfloat16)
    plan({a: av, b: bv, s: scale, y: out})
    ref = torch.empty_like(out)
    ref[0].fill_(scale.float().sum())
    ref[1].fill_(scale.float().sum() * 2)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)
