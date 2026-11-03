# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Row gather -> matmul uses the existing grouped gather operand loader."""

from math import prod
from types import SimpleNamespace

import pytest
import torch
import cudnn

from cudnn._gemm_validate import validate_graph
from cudnn.gemm.frost.graph_analyzer import analyze_with_binding
from cudnn.graph_types import NodeType
from cudnn.nodes import gather_shape

pytestmark = pytest.mark.L0


def _stride(shape):
    return [prod(shape[i + 1 :]) for i in range(len(shape))]


def _graph(prefix=(), *, broadcast=False, dtype=cudnn.data_type.BFLOAT16, rows=37, k=136, n=128, source_rows=71):
    g = cudnn.pygraph(io_data_type=dtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)

    def tensor(shape, dt, name):
        return g.tensor(dim=list(shape), stride=_stride(shape), data_type=dt, name=name)

    source = tensor((source_rows, k), dtype, "cache")
    indices = tensor((*prefix, rows), cudnn.data_type.INT32, "indices")
    weight = tensor((*((1,) * len(prefix) if broadcast else prefix), k, n), dtype, "weight")
    gathered = g.gather(source, indices, axis=0, name="selected")
    out = g.matmul(gathered, weight, name="scores").set_output(True).set_data_type(dtype)
    for node in g.nodes:
        node.infer_properties(g.context)
    return g, source, indices, weight, gathered, out


@pytest.mark.parametrize("prefix", [(), (2,), (2, 3)])
@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize("dtype", [cudnn.data_type.HALF, cudnn.data_type.BFLOAT16])
def test_frost_row_gather_lowering(prefix, broadcast, dtype):
    g, source, indices, weight, gathered, out = _graph(prefix, broadcast=broadcast, dtype=dtype)
    valid = validate_graph(g)
    assert valid
    chain, binding = analyze_with_binding(g)
    assert chain.moe.mode == "gather"
    assert chain.moe.uniform_group_rows == 37
    assert chain.moe.num_groups == prod(prefix)
    assert chain.moe.num_experts == (1 if broadcast else prod(prefix))
    assert (chain.matmul.M, chain.matmul.K, chain.matmul.N) == (prod(prefix) * 37, 136, 128)
    assert binding.first_token_offset is None
    assert binding.token_index is indices
    assert binding.a_operands == [source]
    assert binding.b_operands == [weight]
    assert binding.outputs == [out]
    views = {id(t): v for t, v in binding.operand_views}
    assert views[id(indices)].target_shape == (1, prod(prefix) * 37, 1)
    assert source.get_dim() == [71, 136]
    assert gathered.get_dim() == [*prefix, 37, 136]
    assert [node.node_type for node in g.nodes] == [NodeType.GATHER, NodeType.MATMUL]
    assert not g._backend_lowerable()


@pytest.mark.parametrize("axis", [0, 1, -1, -2])
def test_frost_gather_logical_shape(axis):
    result = gather_shape((5, 7), (2, 3), axis)
    expected = [2, 3, 7] if axis in (0, -2) else [5, 2, 3]
    assert result == expected


@pytest.mark.parametrize(
    "case,reason",
    [
        ("axis", "axis=0"),
        ("dtype", "matching FP16/BF16"),
        ("int64", "INT32"),
        ("materialized", "virtual"),
        ("indices_stride", "contiguous"),
        ("output_overlap", "non-overlapping"),
        ("source_stride", "K-major"),
        ("partial_broadcast", "batch"),
    ],
)
def test_frost_row_gather_declines_unsupported(case, reason):
    g, source, indices, weight, gathered, out = _graph((2, 3))
    if case == "axis":
        g.nodes[0].params["axis"] = 1
    elif case == "dtype":
        weight.set_data_type(cudnn.data_type.FLOAT)
    elif case == "int64":
        indices.set_data_type(cudnn.data_type.INT64)
    elif case == "materialized":
        gathered.set_output(True)
    elif case == "indices_stride":
        indices.set_stride([222, 74, 2])
    elif case == "output_overlap":
        out.set_stride([0, 0, 0, 1])
    elif case == "source_stride":
        source.set_stride([1, 71])
    else:
        weight.set_dim([2, 1, 136, 128]).set_stride([136 * 128, 136 * 128, 128, 1])
    with pytest.raises((ValueError, NotImplementedError), match=reason):
        analyze_with_binding(g)


@pytest.mark.parametrize("pipeline,swap_ab", [("sm100", True), ("sm120", False)])
def test_frost_row_gather_declines_other_loaders(pipeline, swap_ab):
    from cudnn.gemm.frost.kernel_registry import select_template

    g, *_ = _graph()
    chain, _ = analyze_with_binding(g)
    with pytest.raises(NotImplementedError, match="non-swapped SM100"):
        select_template(chain, SimpleNamespace(pipeline=pipeline, swap_ab=swap_ab))


def test_frost_gather_validation():
    g, source, indices, _, gathered, _ = _graph()
    gathered.set_data_type(cudnn.data_type.FLOAT)
    with pytest.raises(ValueError, match="preserves"):
        validate_graph(g)
    gathered.set_data_type(source.get_data_type())
    indices.set_data_type(cudnn.data_type.FLOAT)
    with pytest.raises(ValueError, match="INT32 or INT64"):
        validate_graph(g)
    with pytest.raises(ValueError, match="axis"):
        gather_shape((5, 7), (2,), 2)


def test_frost_row_gather_runtime_projection_and_rejection():
    # Actual CUDA carriers, but no SM100 launch. Exercise the prepared transport
    # used by compiled plans independently of kernel availability.
    if not torch.cuda.is_available():
        pytest.skip("runtime carrier test requires CUDA")
    from cudnn.gemm.frost.compiler import _resolve_moe_variant_pack
    from cudnn.gemm.frost import compiler

    g, source, indices, weight, _, out = _graph((2, 3))
    chain, binding = analyze_with_binding(g)
    compiled = SimpleNamespace(chain=chain, binding=binding, _compiler=compiler, device=torch.cuda.current_device())
    source_buf = torch.randn(source.get_dim(), device="cuda", dtype=torch.bfloat16)
    index_buf = torch.zeros(indices.get_dim(), device="cuda", dtype=torch.int32)
    weight_buf = torch.randn(weight.get_dim(), device="cuda", dtype=torch.bfloat16)
    output_buf = torch.full(out.get_dim(), float("nan"), device="cuda", dtype=torch.bfloat16)
    pack = {source: source_buf, indices: index_buf, weight: weight_buf, out: output_buf}
    resolved = _resolve_moe_variant_pack(compiled, pack)
    assert resolved[0][0].shape == (1, 71, 136)
    assert resolved[1][0].shape == (6, 128, 136)
    assert resolved[2][0].shape == (1, 222, 128)
    assert resolved[4] is None
    assert resolved[7].shape == (222,)
    assert resolved[0][0].data_ptr() == source_buf.data_ptr()
    # Native slots are the public graph execution transport; declarations stay
    # logical rank-N, while the prepared ABI view keeps the same addresses.
    from cudnn.frost.buffers import DTYPES, dtype_name

    native_pack = {}
    for tensor, value in pack.items():
        code, bits = DTYPES[dtype_name(value)]
        native_pack[tensor] = cudnn._pybind_module.make_operand_buffer(value.data_ptr(), list(value.shape), code, bits, compiled.device)
    native = _resolve_moe_variant_pack(compiled, native_pack)
    assert tuple(native[1][0].shape) == (6, 128, 136)
    assert native[1][0].data_ptr() == weight_buf.data_ptr()
    assert native[2][0].data_ptr() == output_buf.data_ptr()
    bad = dict(pack)
    bad[indices] = torch.zeros((2, 4, 37), device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="source shape"):
        _resolve_moe_variant_pack(compiled, bad)
    bad[indices] = index_buf.float()
    with pytest.raises(ValueError, match="dtype"):
        _resolve_moe_variant_pack(compiled, bad)
    bad = dict(pack)
    bad[source] = source_buf.cpu()
    with pytest.raises(ValueError, match="device"):
        _resolve_moe_variant_pack(compiled, bad)
    bad = dict(pack)
    bad[source] = output_buf.reshape(-1)[: source_buf.numel()].view(source_buf.shape)
    with pytest.raises(ValueError, match="overlap"):
        _resolve_moe_variant_pack(compiled, bad)
    # Same-plan bindings stay valid after rejected calls.
    resolved = _resolve_moe_variant_pack(compiled, pack)
    assert resolved[10] == (222, 128, 136)


@pytest.mark.no_workspace_shim
@pytest.mark.parametrize("prefix,broadcast", [((), False), ((2, 3), True), pytest.param((2, 3), False, marks=pytest.mark.L1)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("public_api", [False, pytest.param(True, marks=pytest.mark.L1)])
def test_frost_row_gather_native_capture(prefix, broadcast, dtype, public_api):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("full GEMM qualification requires SM100/SM103")
    from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
    from cudnn.gemm.frost.tile_config import by_name

    dt = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g, source, indices, weight, _, out = _graph(prefix, broadcast=broadcast, dtype=dt)
    cache = torch.randn(source.get_dim(), device="cuda", dtype=dtype)
    ids = torch.arange(prod(indices.get_dim()), device="cuda", dtype=torch.int32).reshape(indices.get_dim()) % 71
    ids.reshape(-1)[::7] = -1
    ids.reshape(-1)[1::11] = 71
    w = torch.randn(weight.get_dim(), device="cuda", dtype=dtype)
    y = torch.full(out.get_dim(), float("nan"), device="cuda", dtype=dtype)
    cfg = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    handle = None
    if public_api:
        from cudnn.engines.manifest import MANIFEST
        from cudnn.gemm.frost.knobs import GemmKnobs

        g.validate()
        g.build_operation_graph()
        engine_id = next(row.engine_id for row in MANIFEST if row.name == "frost_gemm")
        g.create_execution_plan(engine_id, GemmKnobs.from_config(cfg).to_public())
        g.check_support()
        g.build_plans()
        workspace_bytes = g.get_workspace_size()
        handle = cudnn.create_handle()
        cudnn.set_stream(handle, stream.cuda_stream)

        def launch(pack, workspace):
            g.execute(pack, workspace, handle=handle)

    else:
        compiled = jit_from_cudnn_graph(g, config=cfg)
        workspace_bytes = compiled.workspace_bytes

        def launch(pack, workspace):
            compiled(pack, workspace, stream=stream.cuda_stream)

    workspace = torch.empty(workspace_bytes, device="cuda", dtype=torch.uint8)
    pack = {source: cache, indices: ids, weight: w, out: y}
    captured = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            launch(pack, workspace)
        with torch.cuda.graph(captured, stream=stream):
            launch(pack, workspace)
        for delta in (0, 1):
            cache.add_(delta * 0.25)
            ids.copy_((ids + delta) % 73 - 1)
            y.fill_(float("nan"))
            captured.replay()
            valid = (ids >= 0) & (ids < cache.shape[0])
            selected = cache[ids.clamp(0, cache.shape[0] - 1).long()]
            selected = torch.where(valid[..., None], selected, torch.zeros_like(selected))
            ref = (selected.float() @ w.float()).to(dtype)
            torch.testing.assert_close(y, ref, rtol=0.03, atol=0.03)
            same = y.clone()
            captured.replay()
            torch.testing.assert_close(y, same, rtol=0, atol=0)
    finally:
        captured.reset()
        if handle is not None:
            cudnn.destroy_handle(handle)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_frost_row_gather_index_helper_on_hopper(dtype):
    """Qualifies index/padding semantics, not the SM100 MMA pipeline."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("index helper probe requires Hopper or newer")
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_tensor
    from cudnn.gemm.frost.tile_helpers import moe_gather_row
    from cutlass.cute.runtime import make_fake_stream
    from cuda.bindings import driver

    @cute.kernel
    def probe(src, ids, dst):
        tid, _, _ = cute.arch.thread_idx()
        for pos in range(tid, 12 * 13, 32):
            row, col = pos // 13, pos % 13
            # Same indirect row lookup and unsigned range test as the loader.
            token = moe_gather_row(ids, row, cutlass.Int32(12), cutlass.Int32(-1))
            value = cutlass.Float32(0)
            if cutlass.Uint32(token) < cutlass.Uint32(8):
                value = src[token, col].to(cutlass.Float32)
            dst[row, col] = value.to(dst.element_type)

    probe.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    @cute.jit
    def host(src, ids, dst, stream):
        probe(src, ids, dst).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    dt = cutlass.Float16 if dtype == torch.float16 else cutlass.BFloat16
    fake = (
        make_fake_tensor(dt, (8, 13), (13, 1), assumed_align=16),
        make_fake_tensor(cutlass.Int32, (12,), (1,), assumed_align=16),
        make_fake_tensor(dt, (12, 13), (13, 1), assumed_align=16),
    )
    fn = cute.compile(host, *fake, make_fake_stream(use_tvm_ffi_env_stream=False), options="--enable-tvm-ffi")
    src = torch.arange(8 * 13, device="cuda", dtype=torch.float32).reshape(8, 13).to(dtype)
    ids = torch.tensor([7, 2, -1, 0, 8, 7, 3, 3, 100, -100, 6, 1], device="cuda", dtype=torch.int32)
    out = torch.full((12, 13), float("nan"), device="cuda", dtype=dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    captured = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            fn(src, ids, out, driver.CUstream(stream.cuda_stream))
        with torch.cuda.graph(captured, stream=stream):
            fn(src, ids, out, driver.CUstream(stream.cuda_stream))
        for delta in (0, 1):
            src.add_(delta)
            ids.copy_(torch.roll(ids, shifts=delta))
            out.fill_(float("nan"))
            captured.replay()
            expected = src[ids.clamp(0, 7).long()]
            expected = torch.where(((ids >= 0) & (ids < 8))[:, None], expected, torch.zeros_like(expected))
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
    finally:
        captured.reset()
