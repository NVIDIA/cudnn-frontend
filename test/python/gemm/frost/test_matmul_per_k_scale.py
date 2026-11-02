# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-K scaling: structural contracts, binding lifecycle and representative GPU paths."""

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
from cudnn.gemm.frost.fusion_ir import Axis, TensorRef, swap_ab
from cudnn.gemm.frost.graph_analyzer import analyze, analyze_with_binding, swap_ab_binding
from cudnn.gemm.frost.kernel_registry import select_template
from cudnn.gemm.frost.sm100 import compiler, epilogue_codegen
from cudnn.gemm.frost.tile_config import by_name
from gemm_test_utils import requires_sm100

pytestmark = pytest.mark.L0
_TORCH_DTYPES = {cudnn.data_type.BFLOAT16: torch.bfloat16, cudnn.data_type.HALF: torch.float16, cudnn.data_type.FLOAT: torch.float32}
# Cover distinct risks without compiling the full side × swap × CTA × K × ops × order product.
_CASES = [
    pytest.param(dict(side="a", K=128), 1, False, id="a-plain"),
    pytest.param(dict(side="b", K=128, aux_first=True), 1, False, id="b-plain-reversed"),
    pytest.param(dict(side="a", surrounding=True, aux_first=True), 1, True, id="a-unary-swap-reversed"),
    pytest.param(dict(side="b", surrounding=True), 1, True, id="b-unary-swap"),
    pytest.param(dict(side="ab"), 2, False, id="both-pair"),
    pytest.param(dict(side="ab", repeated=True), 2, True, id="both-repeated-pair-swap"),
    pytest.param(dict(side="ab", repeated=True, dtype=cudnn.data_type.HALF), 1, False, id="fp16-repeated"),
    pytest.param(dict(side="ab", dtype=cudnn.data_type.HALF), 1, True, id="fp16-swap"),
    pytest.param(dict(side="a", batch=2, K=144), 1, False, id="a-batch"),
    pytest.param(dict(side="b", batch=2, K=144), 2, True, id="b-batch-pair-swap"),
]


def _config(cta_group=1, swapped=False):
    return replace(by_name(f"CONFIG_sm100_128x128x64_128x128x32_cluster{cta_group}x1_{cta_group}ctamma"), swap_ab=swapped)


def _graph(
    M=128,
    N=128,
    K=136,
    *,
    side="a",
    dtype=cudnn.data_type.BFLOAT16,
    batch=1,
    surrounding=False,
    repeated=False,
    aux_first=False,
    a_major="k",
    b_major="k",
    scale_stride=1,
):
    g = cudnn.pygraph(io_data_type=dtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tensors = {
        "a": g.tensor(name="A", dim=[batch, M, K], stride=[M * K, K, 1] if a_major == "k" else [M * K, 1, M]),
        "b": g.tensor(name="B", dim=[batch, K, N], stride=[K * N, 1, K] if b_major == "k" else [K * N, N, 1]),
    }
    operands = dict(tensors)
    for operand in side:
        dim = [1, 1, K] if operand == "a" else [1, K, 1]
        stride = [K * scale_stride, K * scale_stride, scale_stride] if operand == "a" else [K * scale_stride, scale_stride, K * scale_stride]
        # Simultaneous scaling deliberately uses heterogeneous scale dtypes.
        scale_dtype = cudnn.data_type.FLOAT if side == "ab" and operand == "a" else dtype
        scale = g.tensor(name=f"scale_{operand}" if side == "ab" else "scale", dim=dim, stride=stride, data_type=scale_dtype)
        tensors[f"scale_{operand}"] = scale
        value = operands[operand]
        if surrounding:
            value = g.cos(input=value).set_data_type(dtype)
        for _ in range(2 if repeated and operand == "a" else 1):
            value = g.mul(a=scale if aux_first else value, b=value if aux_first else scale).set_data_type(dtype)
        operands[operand] = g.relu(input=value).set_data_type(dtype) if surrounding else value
    tensors["y"] = g.matmul(A=operands["a"], B=operands["b"]).set_output(True)
    return g, tensors


def _buffers(tensors, m=128, n=128, k=136, *, batch=1, dtype=torch.bfloat16, device="cpu"):
    buffers = {
        "a": torch.ones(batch, m, k, dtype=dtype, device=device),
        "b": torch.ones(batch, n, k, dtype=dtype, device=device),
        "y": torch.zeros(batch, m, n, dtype=dtype, device=device),
    }
    for side in "ab":
        key = f"scale_{side}"
        if key in tensors:
            buffers[key] = torch.ones((1, 1, k) if side == "a" else (1, k, 1), dtype=_TORCH_DTYPES[tensors[key].get_data_type()], device=device)
    return buffers


def _pack(tensors, buffers):
    return {tensor: buffers[key] for key, tensor in tensors.items()}


def _mock_plan(monkeypatch, g, swapped):
    chain, binding = analyze_with_binding(g)
    if swapped:
        chain, binding = swap_ab(chain), swap_ab_binding(binding)
    launches = []
    monkeypatch.setattr(compiler, "_check_plan_device", lambda device: None)
    plan = compiler.CompiledFusedGemm(
        chain,
        _config(swapped=swapped),
        [ref.name for ref in chain.aux_tensors],
        Path("unused"),
        lambda *args, **kw: launches.append(args),
        binding=binding,
        vec_bytes_epi=16,
    )
    return plan, launches


@pytest.mark.parametrize("options,cta_group,swapped", _CASES)
def test_frost_per_k_graph_and_render(options, cta_group, swapped):
    g, tensors = _graph(**options)
    chain, binding = analyze_with_binding(g)
    assert tuple(binding.a_operands) == (tensors["a"],)
    assert tuple(binding.b_operands) == (tensors["b"],)
    assert tuple(binding.outputs) == (tensors["y"],)
    assert tuple(binding.aux) == tuple(t for key, t in tensors.items() if key.startswith("scale_"))
    assert chain.ops == []
    for side in "ab":
        ops = getattr(chain, f"mainloop_{side}_ops")
        expected = []
        if f"scale_{side}" in tensors:
            expected = ["mul"] * (2 if options.get("repeated") and side == "a" else 1)
            if options.get("surrounding"):
                expected = ["cos", *expected, "relu"]
        assert [op.op for op in ops] == expected
        for op in (op for op in ops if op.op == "mul"):
            assert op.aux_on_rhs is not options.get("aux_first", False)
            assert (op.compute_dtype, op.out_dtype) == ("fp32", getattr(chain.matmul, f"{side}_dtype"))
    transformed = swap_ab(chain) if swapped else chain
    restored = swap_ab(swap_ab(chain))
    assert restored.matmul == chain.matmul and restored.aux_tensors == chain.aux_tensors
    assert restored.mainloop_a_ops == chain.mainloop_a_ops and restored.mainloop_b_ops == chain.mainloop_b_ops
    cfg = _config(cta_group, swapped)
    assert select_template(transformed, cfg)._other_reject(transformed, cfg) is None
    snippets = epilogue_codegen.generate(
        transformed, vec_bytes_epi=compiler._epi_vec_bytes(transformed, cfg), output_elem_bytes=2, tma_slots=compiler._tma_slots_for(transformed, cfg)
    )
    src = compiler._render_template(transformed, snippets, cfg)
    ast.parse(src)
    assert "@@INJECT_" not in src and "mainloop_koob_fix = True" in src
    for ref in transformed.aux_tensors:
        shape = "(1, 1, sym_k)" if ref.k_axis == 2 else "(1, sym_k, 1)"
        assert f"{shape}, stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64())" in src
    assert "k_tile_idx * cta_tile_mnk[2] + ml_L % cta_tile_mnk[2]" in src
    for side in "ab":
        for i, op in enumerate(getattr(transformed, f"mainloop_{side}_ops")):
            if op.op == "mul":
                assert f"if ml_{side}_{i}_gk < k:" in src
                assert f"ml_ptr_{side}.load_swizzled" in src
                snippet = getattr(snippets, f"mainloop_transform_{side}")
                assert any(
                    line.startswith(f"_r_ml_{side}_{i} = ") and f".to({compiler.DTYPE_TO_CUTLASS[op.out_dtype]})" in line for line in snippet.splitlines()
                )


@pytest.mark.parametrize(
    "options,change,reason",
    [
        ({"a_major": "m"}, {}, "K-major"),
        ({"side": "b", "b_major": "n"}, {}, "K-major"),
        ({"scale_stride": 2}, {}, "unit stride"),
        ({"K": 135}, {}, "K % 8"),
        ({}, {"compute_dtype": "int32"}, "FP32"),
        ({}, {"out_dtype": "fp32"}, "result"),
        ({}, {"op": "add"}, "only mul"),
    ],
)
def test_frost_per_k_capability_not_ir(options, change, reason):
    chain = analyze(_graph(**options)[0])
    if change:
        chain = replace(chain, mainloop_a_ops=[replace(chain.mainloop_a_ops[0], **change)])
    assert reason in select_template(chain, _config())._other_reject(chain, _config())


def test_frost_per_k_ir_rejects_invalid_broadcast():
    chain = analyze(_graph()[0])
    with pytest.raises(ValueError, match="mainloop-only"):
        replace(chain, ops=[chain.mainloop_a_ops[0]])
    with pytest.raises(ValueError, match="expected"):
        replace(chain, aux_tensors=[replace(chain.aux_tensors[0], dim=(1, 1, 144))])
    with pytest.raises(ValueError, match="explicit reduction axis"):
        replace(chain.aux_tensors[0], reduction_axis=None)


def test_frost_aux_symbolic_shape_contract():
    # Equal and unit extents cannot identify a logical GEMM axis by value.
    cases = [
        ("scalar", (), False, (), "(1, 1)"),
        ("scalar", (1,), False, (1,), "(1, 1)"),
        ("per_col", (128,), False, (Axis.N,), "(1, sym_n)"),
        ("per_row", (128, 1), False, (Axis.M, 1), "(sym_m, 1)"),
        ("per_elem", (128, 128), False, (Axis.M, Axis.N), "(sym_m, sym_n)"),
        ("scalar", (2, 1, 1), False, (Axis.BATCH, 1, 1), "(sym_l, 1, 1)"),
        ("per_row", (1, 1, 1), False, (1, Axis.M, 1), "(1, sym_m, 1)"),
        ("per_col", (2, 1, 128), False, (Axis.BATCH, 1, Axis.N), "(sym_l, 1, sym_n)"),
        ("per_elem", (2, 128, 128), False, (Axis.BATCH, Axis.M, Axis.N), "(sym_l, sym_m, sym_n)"),
        ("scalar", (4, 1, 1), True, (Axis.GROUP, 1, 1), "(sym_g, 1, 1)"),
        ("per_row", (4, 128, 1), True, (Axis.GROUP, Axis.M, 1), "(sym_g, sym_m, 1)"),
        ("per_col", (4, 1, 128), True, (Axis.GROUP, 1, Axis.N), "(sym_g, 1, sym_n)"),
    ]
    for mode, dim, grouped, shape, code in cases:
        ref = TensorRef("aux", dim, (1,) * len(dim), "bf16", mode, grouped)
        assert ref.symbolic_shape == shape
        assert len(ref.symbolic_shape) == len(ref.dim)
        assert compiler._aux_fake_shape_code(ref) == code
        # Fake shape and fake stride must use the lowered ABI rank, not IR rank.
        src = compiler._aux_fake_block([ref])
        assert f", {code}, " in src
        stride = "(" + ", ".join("cute.sym_int64()" for _ in range(max(2, len(dim)))) + ")"
        assert f"stride={stride}" in src
    chain = analyze(_graph(side="ab")[0])
    chain = replace(chain, matmul=replace(chain.matmul, K=1), aux_tensors=[replace(ref, dim=(1, 1, 1)) for ref in chain.aux_tensors])
    assert [ref.symbolic_shape for ref in chain.aux_tensors] == [(1, 1, Axis.K), (1, Axis.K, 1)]
    swapped = swap_ab(chain)
    assert [ref.symbolic_shape for ref in swapped.aux_tensors] == [(1, Axis.K, 1), (1, 1, Axis.K)]
    assert swap_ab(swapped).aux_tensors == chain.aux_tensors
    with pytest.raises(ValueError, match="expected"):
        replace(chain, aux_tensors=[replace(ref, reduction_axis=3 - ref.k_axis) for ref in chain.aux_tensors])


@pytest.mark.parametrize("side", ["a", "b", "ab"])
@pytest.mark.parametrize("swapped", [False, True])
def test_frost_per_k_binding_lifecycle(monkeypatch, side, swapped):
    g, tensors = _graph(side=side, repeated=True)
    plan, launches = _mock_plan(monkeypatch, g, swapped)
    for m, n, k in ((128, 128, 136), (160, 192, 264)):
        buffers = _buffers(tensors, m, n, k)
        for key in (key for key in tensors if key.startswith("scale_")):
            scale = buffers[key]
            axis = 2 if key == "scale_a" else 1
            invalid = [torch.ones(1, 1, extent, dtype=scale.dtype) for extent in (k - 8, k + 8)]
            invalid.append(torch.ones(1, 1, 2 * k, dtype=scale.dtype)[..., ::2])
            if axis == 1:
                invalid = [v.transpose(1, 2) for v in invalid]
            invalid += [scale.transpose(1, 2), scale.to(torch.float16 if scale.dtype == torch.bfloat16 else torch.bfloat16)]
            for bad in invalid:
                with pytest.raises(ValueError, match="per-K mainloop scale") as error:
                    plan(_pack(tensors, dict(buffers, **{key: bad})))
                assert repr(tensors[key].get_name()) in str(error.value)
                assert len(launches) == (0 if k == 136 else 1)
                assert torch.count_nonzero(buffers["y"]) == 0
        plan(_pack(tensors, buffers))
        assert launches[-1][0][:3] == ((n, m, k) if swapped else (m, n, k))
        for arg, ref in zip(launches[-1][4:], plan.chain.aux_tensors):
            assert arg.shape[ref.k_axis] == k
    assert len(launches) == 2


@pytest.mark.parametrize("side", ["a", "b", "ab"])
@pytest.mark.parametrize("swapped", [False, True])
def test_frost_per_k_device_view_binding(monkeypatch, side, swapped):
    """Per-K binding accepts zero-copy views with a tuple-only stride API."""
    g, tensors = _graph(side=side)
    plan, launches = _mock_plan(monkeypatch, g, swapped)
    for m, n, k in ((128, 128, 136), (160, 192, 264)):
        buffers = _buffers(tensors, m, n, k)
        views = dict(buffers)
        for key in (key for key in tensors if key.startswith("scale_")):
            scale = buffers[key]
            # Keep the backing tensor alive: DeviceView owns no allocation.
            views[key] = DeviceView(scale.data_ptr(), scale.shape, dtype_name(scale), 0)
        plan(_pack(tensors, views))
        assert launches[-1][0][:3] == ((n, m, k) if swapped else (m, n, k))
        for arg, ref in zip(launches[-1][4:], plan.chain.aux_tensors):
            assert arg.shape[ref.k_axis] == k
            assert any(arg.data_ptr() == views[key].data_ptr() for key in tensors if key.startswith("scale_"))
    assert len(launches) == 2


def _reference(buffers, dtype, options):
    operands = {side: buffers[side] for side in "ab"}
    for side in "ab":
        key = f"scale_{side}"
        if key not in buffers:
            continue
        value = operands[side]
        if options.get("surrounding"):
            value = torch.cos(value.float()).to(dtype)
        scale = buffers[key] if side == "a" else buffers[key].transpose(1, 2)
        for _ in range(2 if options.get("repeated") and side == "a" else 1):
            value = (value.float() * scale.float()).to(dtype)
        operands[side] = torch.relu(value) if options.get("surrounding") else value
    return (operands["a"].float() @ operands["b"].float().transpose(-1, -2)).to(dtype)


@requires_sm100
@pytest.mark.parametrize("options,cta_group,swapped", _CASES)
def test_frost_per_k_gpu(options, cta_group, swapped):
    """Each plan covers random numerics, exact indexing, capture replay and rebinding."""
    g, tensors = _graph(**options)
    plan = compiler.jit_from_cudnn_graph(g, _config(cta_group, swapped))
    dtype, batch = _TORCH_DTYPES[options.get("dtype", cudnn.data_type.BFLOAT16)], options.get("batch", 1)
    torch.manual_seed(42)
    stream = torch.cuda.Stream()
    for m, n, k in ((128, 128, options.get("K", 136)), (160, 192, 264)):
        buffers = _buffers(tensors, m, n, k, batch=batch, dtype=dtype, device="cuda")
        buffers["a"].normal_()
        buffers["b"].normal_()
        scales = [key for key in tensors if key.startswith("scale_")]
        for key in scales:
            buffers[key].copy_(torch.linspace(-1.5, 1.5, k, device="cuda").reshape(buffers[key].shape))
        pack = _pack(tensors, buffers)
        # The compiled plan defaults to stream 0, not Torch's current stream.
        # Order input production before side-stream warmup and capture explicitly.
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
                buffers[scales[-1]].neg_()
                buffers["y"].fill_(float("nan"))
                captured.replay()
                torch.testing.assert_close(buffers["y"], _reference(buffers, dtype, options), atol=0.25, rtol=0.02)
            # One-hot opposite operand makes wrong K lanes detectable with zero tolerance.
            av, bv = buffers["a"], buffers["b"]
            av.fill_(1)
            bv.fill_(1)
            if options["side"] == "b":
                av.zero_()
                av[:, torch.arange(m, device="cuda"), torch.arange(m, device="cuda") * 5 % k] = 1
            else:
                bv.zero_()
                bv[:, torch.arange(n, device="cuda"), torch.arange(n, device="cuda") * 5 % k] = 1
            if batch > 1:
                av[1].mul_(2)
            for key in scales:
                buffers[key].copy_((torch.arange(k, device="cuda") % 17 - 8).reshape(buffers[key].shape))
            buffers["y"].fill_(float("nan"))
            captured.replay()
            torch.testing.assert_close(buffers["y"], _reference(buffers, dtype, options), atol=0, rtol=0)
        finally:
            captured.reset()


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
