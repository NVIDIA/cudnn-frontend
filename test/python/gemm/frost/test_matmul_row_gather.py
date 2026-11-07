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
    assert chain.grouped_gemm.partition.rows_per_group == 37
    assert chain.grouped_gemm.num_weight_groups == (1 if broadcast else prod(prefix))
    assert chain.grouped_gemm.partition.offset_dtype is None
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


@pytest.mark.parametrize("pipeline", ["sm100", "sm120"])
def test_frost_row_gather_loader_selection(pipeline):
    from dataclasses import replace
    from cudnn.gemm.frost.kernel_registry import select_template
    from cudnn.gemm.frost.tile_config import by_name

    cfg = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1" if pipeline == "sm100" else "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2")
    g, *_ = _graph()
    chain, _ = analyze_with_binding(g)
    template = select_template(chain, cfg)
    assert template.async_gather == (pipeline == "sm100")
    with pytest.raises(NotImplementedError, match="non-swapped"):
        select_template(chain, replace(cfg, swap_ab=True))


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


@pytest.mark.parametrize("source_shape,axis", [((3, 4), 0), ((3, 4), -1), ((2, 3, 4), 1), ((2, 3, 4), 2)])
@pytest.mark.parametrize("view", ["identity", "reshape", "transpose"])
def test_frost_indirect_index_map(source_shape, axis, view):
    from itertools import product
    from cudnn.gemm.frost.fusion_ir import AccessMap, IndexMap

    indices_shape = (2, 2)
    access = AccessMap.take("indices", source_shape, indices_shape, axis)
    normalized_axis = axis % len(source_shape)
    logical = access.consumer_shape
    mapping = (
        IndexMap.identity(logical)
        if view == "identity"
        else IndexMap.reshape(logical, (prod(logical),)) if view == "reshape" else IndexMap.permutation(logical, tuple(reversed(range(len(logical)))))
    )
    projected = access.compose(mapping)
    assert projected.indirections[0][1].indices_ref == "indices"
    assert projected.indirections[0][0] == normalized_axis
    for coordinates in product(*(range(d) for d in mapping.consumer_shape)):
        if view == "reshape":
            linear = coordinates[0]
            original = tuple((linear // prod(logical[i + 1 :])) % d for i, d in enumerate(logical))
        elif view == "transpose":
            original = tuple(reversed(coordinates))
        else:
            original = coordinates
        actual = tuple(expr.evaluate(coordinates) for expr in projected.indirections[0][1].indices_map.coordinates)
        assert actual == original[normalized_axis : normalized_axis + len(indices_shape)]


def _composition_graph(*, source_prefix=(), repartition=False, epilogue="relu", dtype=cudnn.data_type.BFLOAT16):
    graph = cudnn.pygraph(io_data_type=dtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)

    def tensor(shape, dt, name):
        return graph.tensor(dim=list(shape), stride=_stride(shape), data_type=dt, name=name)

    source = tensor((*source_prefix, 71, 128), dtype, "source")
    indices = tensor((2, 3, 16), cudnn.data_type.INT32, "indices")
    gathered = graph.gather(source, indices, axis=len(source_prefix))
    shape = (*source_prefix, 2, 3, 16, 128)
    if repartition:
        shape = (3, 32, 128)
        gathered = graph.reshape(gathered).set_dim(list(shape)).set_stride(_stride(shape))
    weight = tensor((*shape[:-2], 128, 128), dtype, "weight")
    raw = graph.matmul(gathered, weight).set_data_type(dtype)
    aux = None
    if epilogue == "bias":
        aux = tensor((128,), cudnn.data_type.FLOAT, "bias")
        out = graph.add(raw, aux)
    elif epilogue == "shared":
        raw.set_output(True)
        out = graph.add(graph.relu(raw), graph.neg(raw))
    elif epilogue == "index":
        out = raw
        for axis in range(len(shape)):
            out = graph.add(out, graph.gen_index(raw, axis=axis))
    else:
        out = graph.relu(raw)
    out.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    for node in graph.nodes:
        node.infer_properties(graph.context)
    return graph, source, indices, weight, raw, out, aux


@pytest.mark.parametrize("source_prefix,repartition", [((), False), ((1,), False), ((1, 1), False), ((), True)])
@pytest.mark.parametrize("epilogue", ["relu", "shared", "bias", "index"])
def test_frost_gather_composes_with_epilogue_and_views(source_prefix, repartition, epilogue):
    graph, source, indices, weight, raw, out, aux = _composition_graph(source_prefix=source_prefix, repartition=repartition, epilogue=epilogue)
    assert validate_graph(graph)
    chain, binding = analyze_with_binding(graph)
    assert binding.a_operands == [source]
    assert binding.b_operands == [weight]
    assert out in binding.outputs
    assert (raw in binding.outputs) == (epilogue == "shared")
    assert chain.source_access.indirections[0][1].indices_ref == "indices"
    assert chain.source_access.indirections[0][0] == len(source_prefix)
    assert chain.grouped_gemm.partition.rows_per_group == (32 if repartition else 16)
    assert chain.matmul.M == 96
    if aux is not None:
        assert chain.aux_tensors[0].dtype == "fp32"
        assert binding.aux == [aux]
    if epilogue == "index":
        from cudnn.gemm.frost.fusion_ir import Axis

        shape = tuple(raw.get_dim())
        index_ops = [op for op in chain.ops if op.op == "gen_index"]
        for row in range(96):
            coords = {Axis.BATCH: 0, Axis.M: row, Axis.N: 17}
            expected = tuple((row // prod(shape[i + 1 : -1])) % d for i, d in enumerate(shape[:-1])) + (17,)
            assert tuple(op.index_expr.evaluate(coords) for op in index_ops) == expected


@pytest.mark.parametrize("dtype", [cudnn.data_type.INT8, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2])
def test_frost_gather_declines_unsupported_output_before_binding(dtype):
    graph, *_, out, _aux = _composition_graph()
    out.set_data_type(dtype)
    with pytest.raises(NotImplementedError, match="outputs only"):
        analyze_with_binding(graph)


def test_frost_gather_epilogue_runtime_binding():
    if not torch.cuda.is_available():
        pytest.skip("runtime binding requires CUDA")
    from cudnn.gemm.frost import compiler

    graph, source, indices, weight, raw, out, aux = _composition_graph(epilogue="bias", repartition=True)
    chain, binding = analyze_with_binding(graph)
    compiled = SimpleNamespace(chain=chain, binding=binding, _compiler=compiler, device=torch.cuda.current_device())
    pack = {
        source: torch.empty(source.get_dim(), device="cuda", dtype=torch.bfloat16),
        indices: torch.zeros(indices.get_dim(), device="cuda", dtype=torch.int32),
        weight: torch.empty(weight.get_dim(), device="cuda", dtype=torch.bfloat16),
        out: torch.empty(out.get_dim(), device="cuda", dtype=torch.float32),
        aux: torch.empty(aux.get_dim(), device="cuda", dtype=torch.float32),
    }
    resolved = compiler._resolve_moe_variant_pack(compiled, pack)
    assert tuple(resolved[2][0].shape) == (1, 96, 128)
    assert tuple(resolved[3][0].shape) == (1, 1, 128)
    assert resolved[3][0].data_ptr() == pack[aux].data_ptr()
    bad = {**pack, aux: pack[aux].bfloat16()}
    with pytest.raises(ValueError, match="dtype"):
        compiler._resolve_moe_variant_pack(compiled, bad)


@pytest.mark.parametrize("epilogue", ["bias", "shared", "index"])
def test_frost_gather_composed_launch_uses_fixed_groups_without_allocating(epilogue, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("runtime launch-frame test requires CUDA")
    from pathlib import Path
    import cutlass.cute as cute
    from cudnn.gemm.frost.sm100 import compiler
    from cudnn.gemm.frost.tile_config import by_name

    graph, source, indices, weight, raw, out, aux = _composition_graph(epilogue=epilogue, repartition=True)
    chain, binding = analyze_with_binding(graph)
    frames = []
    compiled = compiler.CompiledMoeGemm(
        chain,
        by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1"),
        Path("unused"),
        lambda *args, **kwargs: frames.append((args, kwargs)),
        binding=binding,
        device=torch.cuda.current_device(),
        vec_bytes_epi=16,
    )
    pack = {
        source: torch.empty(source.get_dim(), device="cuda", dtype=torch.bfloat16),
        indices: torch.zeros(indices.get_dim(), device="cuda", dtype=torch.int32),
        weight: torch.empty(weight.get_dim(), device="cuda", dtype=torch.bfloat16),
        out: torch.empty(out.get_dim(), device="cuda", dtype=torch.float32),
    }
    if aux is not None:
        pack[aux] = torch.empty(aux.get_dim(), device="cuda", dtype=torch.float32)
    if raw in binding.outputs:
        pack[raw] = torch.empty(raw.get_dim(), device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(compiled.workspace_bytes, device="cuda", dtype=torch.uint8)
    stream = torch.cuda.Stream()

    def forbidden(*args, **kwargs):
        raise AssertionError("prepared gather launch must not allocate or compile")

    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(cute, "compile", forbidden)
    torch.cuda.set_sync_debug_mode("error")
    try:
        compiled(pack, workspace, stream=stream.cuda_stream)
        compiled(pack, workspace, stream=stream.cuda_stream)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert len(frames) == 2
    for args, kwargs in frames:
        assert args[0][:5] == (96, 128, 128, 3, 3)
        assert args[1] is None  # No dummy offsets or host-read routing metadata.
        assert int(kwargs["stream"]) == stream.cuda_stream


@pytest.mark.no_workspace_shim
@pytest.mark.parametrize("pipeline", ["sm100", "sm120"])
@pytest.mark.parametrize("epilogue", ["bias", pytest.param("index", marks=pytest.mark.L1), pytest.param("shared", marks=pytest.mark.L1)])
def test_frost_gather_composed_native_capture(pipeline, epilogue):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("full grouped GEMM requires SM100/SM103; sm120 template is tested in SM100 compatibility mode")
    from cudnn.gemm.frost.sm100 import compiler as sm100
    from cudnn.gemm.frost.sm120 import compiler as sm120
    from cudnn.gemm.frost.tile_config import by_name

    graph, source, indices, weight, raw, out, aux = _composition_graph(epilogue=epilogue, repartition=True)
    cfg = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1" if pipeline == "sm100" else "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2")
    compiled = (sm100 if pipeline == "sm100" else sm120).jit_from_cudnn_graph(graph, config=cfg)
    x = torch.randn(source.get_dim(), device="cuda", dtype=torch.bfloat16)
    ids = torch.arange(96, device="cuda", dtype=torch.int32).reshape(indices.get_dim()) % 71
    ids.reshape(-1)[::7] = -1
    ids.reshape(-1)[1::11] = 71
    w = torch.randn(weight.get_dim(), device="cuda", dtype=torch.bfloat16)
    y = torch.full(out.get_dim(), float("nan"), device="cuda", dtype=torch.float32)
    pack = {source: x, indices: ids, weight: w, out: y}
    if aux is not None:
        pack[aux] = torch.randn(aux.get_dim(), device="cuda", dtype=torch.float32)
    if raw in compiled.binding.outputs:
        pack[raw] = torch.full(raw.get_dim(), float("nan"), device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(compiled.workspace_bytes, device="cuda", dtype=torch.uint8)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    captured = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            compiled(pack, workspace, stream=stream.cuda_stream)
        with torch.cuda.graph(captured, stream=stream):
            compiled(pack, workspace, stream=stream.cuda_stream)
        for iteration in range(2):
            with torch.cuda.stream(stream):
                x.mul_(-0.5)
                ids.add_(3).remainder_(73).sub_(1)
                y.fill_(float("nan"))
                captured.replay()
            stream.synchronize()
            valid = (ids >= 0) & (ids < 71)
            materialized = torch.where(valid[..., None], x[ids.clamp(0, 70).long()], 0).reshape(3, 32, 128)
            reference_raw = (materialized.float() @ w.float()).bfloat16()
            if epilogue == "bias":
                reference = reference_raw.float() + pack[aux]
            elif epilogue == "shared":
                reference = reference_raw.float().relu() - reference_raw.float()
                torch.testing.assert_close(pack[raw], reference_raw, rtol=0.02, atol=0.25)
            else:
                reference = reference_raw.float()
                for axis, extent in enumerate(raw.get_dim()):
                    shape = [1] * len(raw.get_dim())
                    shape[axis] = extent
                    reference = reference + torch.arange(extent, device="cuda").reshape(shape)
            assert torch.isfinite(y).all()
            torch.testing.assert_close(y, reference, rtol=0.02, atol=0.25)
            previous = y.clone()
            with torch.cuda.stream(stream):
                captured.replay()
            stream.synchronize()
            assert torch.equal(y, previous)
    finally:
        captured.reset()


def test_frost_moe_gather_uses_the_same_indirect_description():
    graph = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)

    def tensor(shape, dt, name):
        return graph.tensor(dim=list(shape), stride=_stride(shape), data_type=dt, name=name)

    source = tensor((1, 71, 128), cudnn.data_type.BFLOAT16, "source")
    weight = tensor((3, 128, 128), cudnn.data_type.BFLOAT16, "weight")
    indices = tensor((1, 96, 1), cudnn.data_type.INT32, "indices")
    offsets = tensor((4, 1, 1), cudnn.data_type.INT32, "offsets")
    raw = graph.moe_grouped_matmul(source, weight, offsets, token_index=indices, mode=cudnn.moe_grouped_matmul_mode.GATHER)
    graph.relu(raw).set_output(True)
    for node in graph.nodes:
        node.infer_properties(graph.context)
    chain, binding = analyze_with_binding(graph)
    access = chain.source_access
    assert access.indirections[0][1].indices_ref == "indices"
    assert access.indirections[0][0] == 1
    assert tuple(expr.evaluate((0, 29, 37)) for expr in access.indirections[0][1].indices_map.coordinates) == (0, 29, 0)
    assert binding.first_token_offset is offsets
    assert not chain.grouped_gemm.partition.is_uniform
    assert chain.grouped_gemm.partition.num_groups == 3
    assert chain.grouped_gemm.partition.offset_dtype == "int32"
    assert chain.grouped_gemm.num_weight_groups == 3
    assert [op.op for op in chain.ops] == ["relu"]


@pytest.mark.parametrize("uniform", [False, True], ids=["offsets", "uniform"])
def test_frost_grouped_spec_preserves_moe_adapter_and_swap(uniform):
    from dataclasses import replace
    from cudnn.gemm.frost.fusion_ir import GroupPartition, GroupedGemmSpec, MoeSpec, swap_ab

    partition = GroupPartition.uniform(3, 37) if uniform else GroupPartition.offsets(3, "int64", 8)
    grouped = GroupedGemmSpec(3, partition)
    # Existing positional arguments keep their order; the generic descriptor
    # is keyword-only and survives swap without becoming a nested dict.
    moe = MoeSpec(3, "gather", "int64", 3, 8, 1, grouped_gemm=grouped)
    chain, _ = analyze_with_binding(_graph(prefix=(3,))[0])
    chain = replace(chain, moe=moe)
    assert chain.grouped_gemm is grouped
    assert swap_ab(chain).grouped_gemm is grouped
    assert swap_ab(swap_ab(chain)).moe == moe
    if not uniform:
        assert MoeSpec(3, "gather", "int64", 3, 8, 1).grouped_gemm == grouped
    from cudnn.gemm.frost.compiler import _CompiledMoeBase

    executor = SimpleNamespace(chain=chain, _compiler=SimpleNamespace(_offset_vector=lambda value: value))
    offsets = object()
    if uniform:
        assert _CompiledMoeBase._group_offsets(executor, None) is None
        with pytest.raises(ValueError, match="does not accept"):
            _CompiledMoeBase._group_offsets(executor, offsets)
    else:
        assert _CompiledMoeBase._group_offsets(executor, offsets) is offsets
        with pytest.raises(ValueError, match="requires first_token_offset"):
            _CompiledMoeBase._group_offsets(executor, None)


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"num_groups": 0, "rows_per_group": 3}, "group count"),
        ({"num_groups": 2, "rows_per_group": 0}, "positive Int64"),
        ({"num_groups": 2, "rows_per_group": 1 << 63}, "Int64"),
        ({"num_groups": 2, "rows_per_group": 3, "offset_dtype": "int32"}, "do not have"),
        ({"num_groups": 2, "offset_dtype": "fp32"}, "int32 or int64"),
        ({"num_groups": 2, "offset_dtype": "int32", "offset_alignment": 0}, "alignment"),
    ],
)
def test_frost_group_partition_rejects_invalid_declarations(kwargs, reason):
    from cudnn.gemm.frost.fusion_ir import GroupPartition

    with pytest.raises(ValueError, match=reason):
        GroupPartition(**kwargs)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
@pytest.mark.parametrize("rows_per_group", [0, 5], ids=["offsets", "uniform"])
def test_frost_group_partition_device_bounds_and_replay(dtype, rows_per_group):
    """Same partition decoder serves fixed groups and runtime ragged offsets."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("partition helper probe requires Hopper or newer")
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_tensor, make_fake_stream
    from cuda.bindings import driver
    from cudnn.gemm.frost.kernel_templates.group_partition import group_row_bounds, group_used_rows

    @cute.kernel
    def probe(offsets, output):
        tid, _, _ = cute.arch.thread_idx()
        if tid < 4:
            begin, end = group_row_bounds(tid, cutlass.Int64(17), offsets, rows_per_group)
            output[tid, 0] = begin
            output[tid, 1] = end
            output[tid, 2] = cutlass.Int32(group_used_rows(cutlass.Int64(17), cutlass.Int32(4), offsets, rows_per_group))

    probe.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    @cute.jit
    def host(offsets, output, stream):
        probe(offsets, output).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    dt = cutlass.Int32 if dtype == torch.int32 else cutlass.Int64
    fake_offsets = None if rows_per_group else make_fake_tensor(dt, (5,), (1,), assumed_align=16)
    fn = cute.compile(
        host,
        fake_offsets,
        make_fake_tensor(cutlass.Int32, (4, 3), (3, 1), assumed_align=16),
        make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
    )
    offsets = None if rows_per_group else torch.tensor([0, 0, 7, 7, 17], device="cuda", dtype=dtype)
    output = torch.full((4, 3), -99, device="cuda", dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    captured = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            fn(offsets, output, driver.CUstream(stream.cuda_stream))
        with torch.cuda.graph(captured, stream=stream):
            fn(offsets, output, driver.CUstream(stream.cuda_stream))
        # Empty groups, malformed offsets, empty routing, and changed device
        # offsets use the same compiled plan. Int64 is checked before narrowing.
        scenarios = [[0, 0, 7, 7, 17], [0, 3, 8, 11, 17], [9, -3, 23, 5, 17], [0, 0, 0, 0, 0]]
        if dtype == torch.int64:
            scenarios.append([0, 3, 1 << 40, -(1 << 40), 17])
        for values in scenarios:
            if offsets is not None:
                offsets.copy_(torch.tensor(values, device="cuda", dtype=dtype))
            output.fill_(-99)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                captured.replay()
            torch.cuda.current_stream().wait_stream(stream)
            bounds = []
            for g in range(4):
                lo, hi = (g * rows_per_group, (g + 1) * rows_per_group) if rows_per_group else (0 if g == 0 else values[g], values[g + 1])
                hi = min(max(hi, 0), 17)
                bounds.append([min(max(lo, 0), hi), hi, 17 if rows_per_group else max(min(max(values[-1], 0), 17), 1)])
            torch.testing.assert_close(output, torch.tensor(bounds, device="cuda", dtype=torch.int32), rtol=0, atol=0)
            previous = output.clone()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                captured.replay()
            torch.cuda.current_stream().wait_stream(stream)
            assert torch.equal(output.view(torch.uint8), previous.view(torch.uint8))
    finally:
        captured.reset()


def test_frost_indirect_map_matches_sparse_attention_key_and_value_views():
    """One query/slot index relation serves both QK and PV operand layouts."""
    from cudnn.gemm.frost.fusion_ir import AccessMap, IndexMap

    keys = AccessMap.take("topk_indices", (71, 576), (3, 5), 0)
    qk_keys = keys.compose(IndexMap.permutation((3, 5, 576), (0, 2, 1)))
    values = AccessMap.take("topk_indices", (71, 512), (3, 5), 0)
    for query in range(3):
        for slot in range(5):
            assert tuple(expr.evaluate((query, 575, slot)) for expr in qk_keys.indirections[0][1].indices_map.coordinates) == (query, slot)
            assert tuple(expr.evaluate((query, slot, 511)) for expr in values.indirections[0][1].indices_map.coordinates) == (query, slot)


@pytest.mark.parametrize("indirect", [False, True], ids=["direct", "indirect"])
def test_frost_access_map_composition_and_lowering(indirect):
    from itertools import product
    from cudnn.gemm.frost.fusion_ir import AccessMap, IndexExpr, IndexMap

    access = AccessMap.take("indices", (5, 7), (2, 3), 0) if indirect else AccessMap.direct(IndexMap.reshape((2, 3, 7), (2, 3, 7)))
    view = IndexMap.permutation(access.consumer_shape, (0, 2, 1))
    composed = access.compose(view).compose(IndexMap.reshape(view.consumer_shape, (42,)))
    assert composed.tensor_dependencies == (frozenset({"indices"}) if indirect else frozenset())
    assert composed.axes == frozenset({0})
    table = [[4, 0, -1], [2, 5, 4]]
    reader = lambda ref, coords: table[coords[0]][coords[1]]
    for coords in product(*(range(d) for d in composed.consumer_shape)):
        old = (coords[0] // 21, coords[0] % 3, (coords[0] // 3) % 7)
        assert composed.evaluate(coords, reader) == access.evaluate(old, reader)
    if indirect:
        with pytest.raises(ValueError, match="index reader"):
            composed.apply((IndexExpr.coordinate(0),))
    else:
        assert composed.apply((IndexExpr.coordinate(0),)) == composed.lower((IndexExpr.coordinate(0),), lambda expr: expr)
    with pytest.raises(ValueError, match="consumer domain"):
        composed.evaluate((42,), reader)
    with pytest.raises(ValueError, match="producer extent"):
        AccessMap.direct(IndexMap((2,), (1,), (IndexExpr.coordinate(0),)))


def _materialization_graph(axis=0, *, side="a", observed=False, index_dtype=cudnn.data_type.INT64, rows=37):
    graph = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    shape = (71, 128) if axis == 0 else (128, 71)
    source = graph.tensor(dim=list(shape), stride=_stride(shape), data_type=cudnn.data_type.BFLOAT16, name="source")
    indices = graph.tensor(dim=[rows], stride=[1], data_type=index_dtype, name="indices")
    selected = graph.gather(source, indices, axis=axis)
    if observed:
        selected.set_output(True)
    if side == "a":
        weight_shape = (selected.dim[-1], 128)
        weight = graph.tensor(dim=list(weight_shape), stride=_stride(weight_shape), data_type=cudnn.data_type.BFLOAT16, name="weight")
        result = graph.matmul(selected, weight)
    else:
        weight_shape = (128, selected.dim[0])
        weight = graph.tensor(dim=list(weight_shape), stride=_stride(weight_shape), data_type=cudnn.data_type.BFLOAT16, name="weight")
        result = graph.matmul(weight, selected)
    result.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    for node in graph.nodes:
        node.infer_properties(graph.context)
    return graph, source, indices, weight, selected, result


@pytest.mark.parametrize("axis,side,observed", [(0, "a", False), (1, "a", False), (0, "b", False), (0, "a", True)])
def test_frost_materialization_is_an_independent_route(axis, side, observed):
    from cudnn.gemm.frost.materialization import available_routes

    graph, source, indices, weight, selected, result = _materialization_graph(axis, side=side, observed=observed)
    before = [(tuple(t.dim), tuple(t.stride), t.is_virtual) for node in graph.nodes for t in node.outputs.values()]
    routes = available_routes(graph)
    assert len(routes) == 1  # INT64 indices deliberately decline the fused loader
    route = routes[0]
    assert len(route.gathers) == 1
    assert route.gathers[0].observable == observed
    assert route.gathers[0].access.producer_shape == tuple(source.dim)
    assert route.binding.a_operands == ([selected] if side == "a" else [weight])
    assert route.binding.b_operands == ([weight] if side == "a" else [selected])
    assert route.binding.outputs == [result]
    assert before == [(tuple(t.dim), tuple(t.stride), t.is_virtual) for node in graph.nodes for t in node.outputs.values()]
    assert [node.node_type for node in graph.nodes] == [NodeType.GATHER, NodeType.MATMUL]


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
def test_frost_materialized_gather_native_capture(axis, dtype, indices_dtype):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("materialized gather requires CUDA SM80 or newer")
    from cudnn.gemm.frost.materialization import MaterializedGatherPlan, available_routes

    graph_dt = {torch.float16: cudnn.data_type.HALF, torch.bfloat16: cudnn.data_type.BFLOAT16, torch.float32: cudnn.data_type.FLOAT}[dtype]
    index_dt = cudnn.data_type.INT32 if indices_dtype == torch.int32 else cudnn.data_type.INT64
    graph = cudnn.pygraph(io_data_type=graph_dt, compute_data_type=cudnn.data_type.FLOAT)
    # Strided storage, duplicate indices, negative/OOB IDs and a K tail.
    source_value = torch.arange(3 * 5 * 14, device="cuda", dtype=torch.float32).to(dtype).reshape(3, 5, 14)[..., ::2]
    extreme = 1 << (40 if indices_dtype == torch.int64 else 30)
    index_storage = torch.tensor([0, 99, -extreme, 99, 1, 99, extreme, 99, 0, 99, 2, 99], device="cuda", dtype=indices_dtype)
    indices_value = index_storage[::2].reshape(2, 3)
    source = graph.tensor(dim=list(source_value.shape), stride=list(source_value.stride()), data_type=graph_dt, name="source")
    indices = graph.tensor(dim=[2, 3], stride=list(indices_value.stride()), data_type=index_dt, name="indices")
    output = graph.gather(source, indices, axis=axis).set_output(True)
    for node in graph.nodes:
        node.infer_properties(graph.context)
    plan = MaterializedGatherPlan(available_routes(graph)[0])
    assert plan.workspace_bytes == 0
    output_value = torch.full(output.dim, float("nan"), device="cuda", dtype=dtype)
    pack = {source: source_value, indices: indices_value, output: output_value}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    captured = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            plan(pack, stream=stream.cuda_stream)
        with torch.cuda.graph(captured, stream=stream):
            plan(pack, stream=stream.cuda_stream)
        for changed in (False, True):
            if changed:
                source_value.add_(1)
                indices_value.copy_(torch.roll(indices_value.flatten(), 1).reshape(2, 3))
            output_value.fill_(float("nan"))
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                captured.replay()
            torch.cuda.current_stream().wait_stream(stream)
            flat_ids = indices_value.flatten().long()
            actual_axis = axis % 3
            valid = (flat_ids >= 0) & (flat_ids < source_value.shape[actual_axis])
            expected = torch.index_select(source_value, actual_axis, flat_ids.clamp(0, source_value.shape[actual_axis] - 1))
            mask_shape = [1] * 3
            mask_shape[actual_axis] = 6
            expected = torch.where(valid.reshape(mask_shape), expected, torch.zeros_like(expected)).reshape(output.dim)
            torch.testing.assert_close(output_value, expected, rtol=0, atol=0)
            previous = output_value.clone()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                captured.replay()
            torch.cuda.current_stream().wait_stream(stream)
            assert torch.equal(output_value.view(torch.uint8), previous.view(torch.uint8))
        bad = dict(pack)
        bad[source] = source_value.to(torch.float64)
        with pytest.raises(ValueError, match="prepared shape/stride/dtype"):
            plan(bad, stream=stream.cuda_stream)
    finally:
        captured.reset()


def test_frost_access_map_multiple_index_dependencies():
    from cudnn.gemm.frost.fusion_ir import AccessMap, IndexExpr, IndexLookup, IndexMap

    index_map = IndexMap.identity((3,))
    access = AccessMap((3,), (5, 7), (IndexLookup("rows", index_map), IndexLookup("columns", index_map)))
    assert access.tensor_dependencies == frozenset({"rows", "columns"})
    assert access.axes == frozenset({0})
    tables = {"rows": [4, -1, 2], "columns": [6, 1, 7]}
    reader = lambda ref, coords: tables[ref][coords[0]]
    assert access.evaluate((0,), reader) == (4, 6)
    assert access.evaluate((1,), reader) is None
    assert access.evaluate((2,), reader) is None
    assert access.compose(IndexMap.reshape((3,), (1, 3))).evaluate((0, 0), reader) == (4, 6)
    with pytest.raises(ValueError, match="domain"):
        AccessMap((2,), (5,), (IndexLookup("rows", index_map),))
    with pytest.raises(ValueError, match="matching intermediate shapes"):
        access.compose(IndexMap.identity((2,)))
    with pytest.raises(ValueError, match="every consumer coordinate"):
        access.lower((), lambda expr: expr)
    bound = access.lower((IndexExpr.coordinate(0),), lambda expr: expr, lambda ref, coords, extent: (ref, coords, extent))
    assert bound == (("rows", (IndexExpr.coordinate(0),), 5), ("columns", (IndexExpr.coordinate(0),), 7))


def test_frost_materialized_gather_public_graph_capture():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("materialized gather requires CUDA SM80 or newer")
    from cudnn.gemm.frost.heuristics import analyze_facts

    graph = cudnn.pygraph(io_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    source = graph.tensor(dim=[5, 7], stride=[7, 1], data_type=cudnn.data_type.FLOAT, name="source")
    indices = graph.tensor(dim=[3], stride=[1], data_type=cudnn.data_type.INT64, name="indices")
    output = graph.gather(source, indices, axis=1).set_output(True)
    graph.validate()
    facts = analyze_facts(graph)
    assert facts.chain is None
    graph.create_execution_plans([cudnn.heur_mode.A])
    graph.check_support()
    graph.build_plans()
    assert graph.get_workspace_size() == 0
    src = torch.randn(5, 7, device="cuda")
    ids = torch.tensor([6, -1, 2], device="cuda", dtype=torch.int64)
    out = torch.full((5, 3), float("nan"), device="cuda")
    pack = {source: src, indices: ids, output: out}
    handle = cudnn.create_handle()
    stream = torch.cuda.Stream()
    cudnn.set_stream(handle, stream.cuda_stream)
    captured = torch.cuda.CUDAGraph()
    try:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph.execute(pack, None, handle=handle)
        with torch.cuda.graph(captured, stream=stream):
            graph.execute(pack, None, handle=handle)
        src.add_(1)
        ids.copy_(torch.tensor([1, 7, 4], device="cuda", dtype=torch.int64))
        out.fill_(float("nan"))
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            captured.replay()
        torch.cuda.current_stream().wait_stream(stream)
        expected = src[:, [1, 0, 4]].clone()
        expected[:, 1] = 0
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    finally:
        captured.reset()
        cudnn.destroy_handle(handle)


@pytest.mark.parametrize("observed", [False, True], ids=["workspace", "caller-output"])
def test_frost_materialized_gather_gemm_transport_on_hopper(observed, monkeypatch):
    """Runs the native gather plus a GEMM spy, not a Blackwell MMA kernel."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("materialized gather requires CUDA SM80 or newer")
    from cudnn.gemm.frost.materialization import MaterializedGatherPlan, available_routes
    from cudnn.frost.workspace import Workspace
    from cudnn._torch_stream import stream_context

    graph, source, indices, weight, selected, output = _materialization_graph(observed=observed)
    route = available_routes(graph)[0]
    calls = []
    gemm_workspace_bytes = 128

    def gemm_spy(values, *, stream, workspace):
        # Inspect the exact source pointer, graph-order shape and nested scratch
        # transport without pretending to qualify the Blackwell GEMM kernel.
        gathered = values[selected]
        assert tuple(gathered.shape) == tuple(selected.dim)
        assert workspace.numel() == gemm_workspace_bytes
        assert buffers.data_ptr(workspace) % 128 == 0
        calls.append((gathered.data_ptr(), int(stream)))
        gathered_torch = gathered if isinstance(gathered, torch.Tensor) else torch.utils.dlpack.from_dlpack(gathered)
        with stream_context(stream, torch.cuda.current_device()):
            torch.matmul(gathered_torch, values[weight], out=values[output])

    from cudnn.frost import buffers

    spy = SimpleNamespace(binding=route.binding, workspace_bytes=gemm_workspace_bytes)

    # SimpleNamespace is not callable; use a tiny compiled-plan stand-in.
    class GemmSpy:
        binding = spy.binding
        workspace_bytes = spy.workspace_bytes

        def __call__(self, *args, **kwargs):
            gemm_spy(*args, **kwargs)

    plan = MaterializedGatherPlan(route, GemmSpy())
    src = torch.randn(source.dim, device="cuda", dtype=torch.bfloat16)
    ids = torch.arange(37, device="cuda", dtype=torch.int64)
    ids[0], ids[-1] = -1, 1 << 40
    weights = torch.randn(weight.dim, device="cuda", dtype=torch.bfloat16)
    out = torch.full(output.dim, float("nan"), device="cuda", dtype=torch.bfloat16)
    ws = torch.empty(plan.workspace_bytes, device="cuda", dtype=torch.uint8)
    pack = {source: src, indices: ids, weight: weights, output: out}
    if observed:
        pack[selected] = torch.full(selected.dim, float("nan"), device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    captured = torch.cuda.CUDAGraph()
    try:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            plan(pack, workspace=ws, stream=stream.cuda_stream)
        first_ptr = calls[-1][0]
        with torch.cuda.graph(captured, stream=stream):
            plan(pack, workspace=ws, stream=stream.cuda_stream)
        assert calls[-1][0] == first_ptr
        src.add_(1)
        ids.copy_(torch.roll(ids, 3))
        out.fill_(float("nan"))
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            captured.replay()
        torch.cuda.current_stream().wait_stream(stream)
        expected = src[ids.clamp(0, 70).long()].clone()
        expected[(ids < 0) | (ids >= 71)] = 0
        torch.testing.assert_close(out, expected @ weights, rtol=0, atol=0)
        if observed:
            torch.testing.assert_close(pack[selected], expected, rtol=0, atol=0)
        else:
            assert first_ptr == ws.data_ptr()
        before = out.clone()
        for bad_workspace in (None, ws[:0], ws[: plan.workspace_bytes - 1]):
            with pytest.raises(ValueError, match="workspace"):
                plan(pack, workspace=bad_workspace, stream=stream.cuda_stream)
        bad_pack = dict(pack)
        bad_pack[indices] = ids.to(torch.int32)
        with pytest.raises(ValueError, match="prepared shape/stride/dtype"):
            plan(bad_pack, workspace=ws, stream=stream.cuda_stream)
        assert torch.equal(out.view(torch.uint8), before.view(torch.uint8))
    finally:
        captured.reset()


def test_frost_materialized_gather_rejects_alias_and_output_overlap():
    if not torch.cuda.is_available():
        pytest.skip("buffer binding checks require CUDA")
    from cudnn.gemm.frost.materialization import MaterializedGatherPlan, available_routes

    graph = cudnn.pygraph(io_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    source = graph.tensor(dim=[5, 7], stride=[7, 1], data_type=cudnn.data_type.FLOAT, name="source")
    indices = graph.tensor(dim=[5], stride=[1], data_type=cudnn.data_type.INT32, name="indices")
    output = graph.gather(source, indices, axis=0).set_output(True)
    for node in graph.nodes:
        node.infer_properties(graph.context)
    plan = MaterializedGatherPlan(available_routes(graph)[0])
    src = torch.randn(5, 7, device="cuda")
    ids = torch.arange(5, device="cuda", dtype=torch.int32)
    previous = src.clone()
    with pytest.raises(ValueError, match="overlap"):
        plan({source: src, indices: ids, output: src})
    assert torch.equal(src, previous)
    output.set_stride([0, 1])
    with pytest.raises(ValueError, match="non-overlapping"):
        available_routes(graph)


@pytest.mark.parametrize("side", ["a", "b"])
def test_frost_materialized_gather_full_gemm_native_capture(side):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("full FROST GEMM fallback requires Blackwell")
    from cudnn.gemm.frost.engine import FrostGemmEngine
    from cudnn.gemm.frost.materialization import MaterializedGatherPlan
    from cudnn.engines.base import ExecutionContext, PlanConfig

    rows = 37 if side == "a" else 32
    graph, source, indices, weight, selected, output = _materialization_graph(side=side, rows=rows)
    engine = FrostGemmEngine()
    engine.check_support(graph)
    compiled = engine.build_plan(graph, PlanConfig(0), ExecutionContext())
    plan = compiled._compiled
    assert isinstance(plan, MaterializedGatherPlan)
    src = torch.randn(source.dim, device="cuda", dtype=torch.bfloat16)
    ids = torch.arange(rows, device="cuda", dtype=torch.int64)
    ids[0], ids[-1] = -1, 1 << 40
    weights = torch.empty_strided(weight.dim, weight.stride, device="cuda", dtype=torch.bfloat16).normal_()
    out = torch.full(output.dim, float("nan"), device="cuda", dtype=torch.bfloat16)
    ws = torch.empty(compiled.get_workspace_size(), device="cuda", dtype=torch.uint8)
    stream = torch.cuda.Stream()
    pack = {source: src, indices: ids, weight: weights, output: out}
    captured = torch.cuda.CUDAGraph()
    try:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            plan(pack, workspace=ws, stream=stream.cuda_stream)
        with torch.cuda.graph(captured, stream=stream):
            plan(pack, workspace=ws, stream=stream.cuda_stream)
        for changed in (False, True):
            if changed:
                src.add_(1)
                ids.copy_(torch.roll(ids, 5))
            out.fill_(float("nan"))
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                captured.replay()
            torch.cuda.current_stream().wait_stream(stream)
            gathered = src[ids.clamp(0, 70)].clone()
            gathered[(ids < 0) | (ids >= 71)] = 0
            expected = gathered.float() @ weights.float() if side == "a" else weights.float() @ gathered.float()
            torch.testing.assert_close(out, expected.to(torch.bfloat16), rtol=2e-2, atol=2e-2)
            previous = out.clone()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                captured.replay()
            torch.cuda.current_stream().wait_stream(stream)
            assert torch.equal(out.view(torch.uint8), previous.view(torch.uint8))
    finally:
        captured.reset()


def test_frost_materialization_survives_fused_loader_decline(monkeypatch, compile_allocates_nothing):
    if not torch.cuda.is_available():
        pytest.skip("plan compilation detector requires CUDA")
    from cudnn.gemm.frost import compiler
    from cudnn.gemm.frost.engine import FrostGemmEngine
    from cudnn.gemm.frost.heuristics import analyze_facts, recommend
    from cudnn.gemm.frost.materialization import MaterializedGatherPlan
    from cudnn.gemm.frost.tile_config import by_name
    from cudnn.engines.base import ExecutionContext, PlanConfig

    graph, *_ = _graph((2,))  # INT32 row gather: both execution routes exist.
    facts = analyze_facts(graph)
    assert facts.chain.has_moe and len(facts.alternatives) == 1
    assert not facts.alternatives[0].has_moe
    config = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1")

    def probe(chain, config):
        if chain.has_moe:
            raise NotImplementedError("selected indirect loader cannot serve this layout")

    built = []

    def jit(chain, binding, config):
        assert not chain.has_moe
        built.append(chain)
        return SimpleNamespace(binding=binding, workspace_bytes=0, lowered=lambda *args, **kwargs: None)

    monkeypatch.setattr(compiler, "probe_chain", probe)
    monkeypatch.setattr(compiler, "plan_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(compiler, "plan_configs", lambda *args, **kwargs: [config])
    monkeypatch.setattr(compiler, "jit_from_chain", jit)
    proposed = recommend("A", facts, {"frost_gemm": 42})
    assert len(proposed) == 1
    engine = FrostGemmEngine()
    engine.check_support(graph)
    captured = []
    compile_allocates_nothing(SimpleNamespace(compile=lambda: captured.append(engine.build_plan(graph, PlanConfig(42), ExecutionContext()))))
    assert len(built) == 1
    assert isinstance(captured[0]._compiled, MaterializedGatherPlan)
    assert captured[0].get_workspace_size() > 0
    # The low-level compiler entry must not stop at the rejected fused analysis.
    direct = compiler.jit_from_cudnn_graph(graph, config=config)
    assert isinstance(direct, MaterializedGatherPlan)
    assert direct.workspace_bytes == captured[0].get_workspace_size()
    assert [node.node_type for node in graph.nodes] == [NodeType.GATHER, NodeType.MATMUL]


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_frost_materialized_integer_gather_strided_output(dtype):
    if not torch.cuda.is_available():
        pytest.skip("materialized gather requires CUDA")
    from cudnn.gemm.frost.materialization import MaterializedGatherPlan, available_routes

    dt = cudnn.data_type.INT32 if dtype == torch.int32 else cudnn.data_type.INT64
    graph = cudnn.pygraph(io_data_type=dt, compute_data_type=cudnn.data_type.FLOAT)
    source = graph.tensor(dim=[7], stride=[0], data_type=dt, name="source")
    indices = graph.tensor(dim=[2, 3], stride=[3, 1], data_type=cudnn.data_type.INT64, name="indices")
    output = graph.gather(source, indices).set_output(True)
    for node in graph.nodes:
        node.infer_properties(graph.context)
    output.set_stride([1, 2])
    plan = MaterializedGatherPlan(available_routes(graph)[0])
    src = torch.full((1,), (1 << 40) + 7 if dtype == torch.int64 else 7, device="cuda", dtype=dtype).expand(7)
    ids = torch.tensor([[0, -1, 6], [7, 3, -(1 << 40)]], device="cuda", dtype=torch.int64)
    dst = torch.full((3, 2), -99, device="cuda", dtype=dtype).T
    stream = torch.cuda.current_stream().cuda_stream
    plan({source: src, indices: ids, output: dst}, stream=stream)
    expected = torch.where((ids >= 0) & (ids < 7), src[0], 0)
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)
    previous = dst.clone()
    plan({source: src, indices: ids, output: dst}, stream=stream)
    assert torch.equal(dst, previous)
