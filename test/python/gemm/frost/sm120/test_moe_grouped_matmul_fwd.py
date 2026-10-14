# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoE grouped matmul forward (NONE, GATHER and SCATTER): analyzer detection + end-to-end
correctness vs a torch group-loop reference (uneven + empty groups)."""

from __future__ import annotations

import pathlib

import cudnn
import cutlass
from dataclasses import replace

import pytest
import torch

from gemm_test_utils import (
    requires_sm120,
    requires_matmul_gpu,
    Plan as _plan,
    ceil_div as _ceil_div,
    to_blocked as _to_blocked,
    block_quant_ref as _block_quant_ref,
    reduction_ref as _reduction_ref,
    reduction_dims as _reduction_dims,
    FULL_EXPERT_REDUCE_OFFSETS as _FULL_EXPERT_REDUCE_OFFSETS,
)

from cudnn.gemm.frost.compiler import force_stg_epi, jit_from_cudnn_graph
from cudnn.gemm.frost.graph_analyzer import analyze, analyze_with_binding
from cudnn.gemm.frost.tile_config import by_name

pytestmark = pytest.mark.L0


def _vp_moe(compiled, token, weight, fto, output):
    """MoE single-GEMM variant-pack dict from the binding."""
    from gemm_test_utils import graph_binding

    bd = graph_binding(compiled)
    outs = list(output) if isinstance(output, (list, tuple)) else [output]
    vp = {
        bd.a_operands[0]: token,
        bd.b_operands[0]: weight,
        bd.first_token_offset: fto,
    }
    vp.update({t: buf for t, buf in zip(bd.outputs, outs)})
    return vp


_CFG = "CONFIG_sm120_128x256x128_16x16x32_cluster1x1_warps4x2"
# (config name, cta_group). SM120 runs a fixed 16x16x32 warp MMA with no cluster
# and no CTA pair, so cta_group is always 1. Each entry stands in for the parent's
# sm100 geometry at the same CTA tile, with M capped at the 256-row TMA box; where
# the parent varied the cluster, this list varies the compute-warp grid instead.
_GEOMETRIES = [
    ("CONFIG_sm120_128x256x128_16x16x32_cluster1x1_warps4x2", 1),
    ("CONFIG_sm120_128x256x128_16x16x32_cluster1x1_warps2x4", 1),
    ("CONFIG_sm120_256x256x64_16x16x32_cluster1x1_warps4x2", 1),  # 256x256 exists only with a 32/64-byte K row
    ("CONFIG_sm120_256x128x128_16x16x32_cluster1x1_warps4x2", 1),
    ("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2", 1),
    ("CONFIG_sm120_256x128x128_16x16x32_cluster1x1_warps2x4", 1),  # parent: 512-row tile
    ("CONFIG_sm120_256x128x128_16x16x32_cluster1x1_warps8x1", 1),  # parent: 512-row tile
    ("CONFIG_sm120_128x256x128_16x16x32_cluster1x1_warps8x1", 1),
    ("CONFIG_sm120_128x256x128_16x16x32_cluster1x1_warps1x8", 1),
]

# The plain-e2e test additionally covers N-tiles that are not a multiple of 32
# (pow2 epilogue subtile spans). Kept OUT of _GEOMETRIES: the N-major-weight
# and block-quant tests legitimately reject these tiles (swizzle-group /
# vsize-divisibility gates).
_GEOMETRIES_STEP8 = _GEOMETRIES + [
    ("CONFIG_sm120_128x48x128_16x16x32_cluster1x1_warps8x1", 1),  # parent: 128x40; SM120 warp tiles are multiples of 16
    ("CONFIG_sm120_128x144x128_16x16x32_cluster1x1_warps8x1", 1),
]

_QUANT_CASES = [
    (
        "e4m3_out_e8m0_scale",
        cudnn.data_type.FP8_E4M3,
        torch.float8_e4m3fn,
        cudnn.data_type.FP8_E8M0,
        torch.float8_e8m0fnu,
        False,
        [64, 0, 128, 64],
        256,
    ),
    (
        "e5m2_out_e8m0_scale",
        cudnn.data_type.FP8_E5M2,
        torch.float8_e5m2,
        cudnn.data_type.FP8_E8M0,
        torch.float8_e8m0fnu,
        False,
        [64, 0, 128, 64],
        256,
    ),
    (
        "e4m3_out_e4m3_scale",
        cudnn.data_type.FP8_E4M3,
        torch.float8_e4m3fn,
        cudnn.data_type.FP8_E4M3,
        torch.float8_e4m3fn,
        False,
        [64, 0, 128, 64],
        256,
    ),
    (
        "e4m3_out_e8m0_scale_f8_128x4",
        cudnn.data_type.FP8_E4M3,
        torch.float8_e4m3fn,
        cudnn.data_type.FP8_E8M0,
        torch.float8_e8m0fnu,
        True,
        [100, 0, 120, 80],
        160,
    ),
]


def _quant_scale_shape(S: int, N: int, reorder: bool) -> tuple[int, int, int]:
    if reorder:
        return (1, _ceil_div(S, 128) * 128, _ceil_div(N // 32, 4) * 4)
    return (1, S, N // 32)


def _build_graph(
    E: int,
    S: int,
    N: int,
    K: int,
    mode=cudnn.moe_grouped_matmul_mode.NONE,
    token_index=None,
    offset_dt=cudnn.data_type.INT32,
    num_groups: int | None = None,
    output_dt=cudnn.data_type.BFLOAT16,
    reduction_mode=None,
    reduction_dims: tuple[int, int, int] | None = None,
    reduction_stride: tuple[int, int, int] | None = None,
    reduction_dt=cudnn.data_type.FLOAT,
    reduction_compute_dt=None,
    reduction_group_offset: bool = False,
    quant: bool = False,
    quant_out_dt=cudnn.data_type.FP8_E4M3,
    quant_scale_dt=cudnn.data_type.FP8_E8M0,
    quant_scale_reorder: bool = False,
    quant_scale_dim: tuple[int, int, int] | None = None,
    weight_major: str = "k",
):
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    tok = g.tensor(
        name="token",
        dim=[1, S, K],
        stride=[S * K, K, 1],
        data_type=cudnn.data_type.BFLOAT16,
    )
    w = g.tensor(
        name="weight",
        dim=[E, K, N],
        stride=[K * N, 1, K] if weight_major == "k" else [K * N, N, 1],
        data_type=cudnn.data_type.BFLOAT16,
    )
    fto_groups = E if num_groups is None else num_groups
    fto = g.tensor(
        name="first_token_offset",
        dim=[fto_groups + 1, 1, 1],
        stride=[1, 1, 1],
        data_type=offset_dt,
    )
    kwargs = {} if token_index is None else {"token_index": token_index}
    out = g.moe_grouped_matmul(
        tok,
        w,
        fto,
        mode=mode,
        compute_data_type=cudnn.data_type.FLOAT,
        name="moe",
        **kwargs,
    )
    if reduction_mode is not None:
        red_kwargs = {}
        if reduction_compute_dt is not None:
            red_kwargs["compute_data_type"] = reduction_compute_dt
        if reduction_group_offset:
            red_kwargs["group_offset"] = fto
        R = g.reduction(input=out, mode=reduction_mode, name="red", **red_kwargs)
        assert reduction_dims is not None
        stride = reduction_stride
        if stride is None:
            stride = (
                reduction_dims[1] * reduction_dims[2],
                reduction_dims[2],
                1,
            )
        R.set_dim(list(reduction_dims)).set_stride(list(stride))
        R.set_output(True).set_data_type(reduction_dt)
    if quant:
        q, q_scale = g.block_scale_quantize(input=out, block_size=32, name="q")
        q.set_data_type(quant_out_dt).set_output(True)
        if quant_scale_dim is not None:
            q_scale.set_dim(list(quant_scale_dim)).set_stride([quant_scale_dim[1] * quant_scale_dim[2], quant_scale_dim[2], 1])
        q_scale.set_data_type(quant_scale_dt).set_output(True)
        if quant_scale_reorder:
            q_scale.set_reordering_type(cudnn.tensor_reordering.F8_128x4)
        return g
    out.set_data_type(output_dt).set_output(True)
    return g


# --------------------------------------------------------------------------- #
# Analyzer (no GPU needed)
# --------------------------------------------------------------------------- #


def test_analyzer_detects_moe_grouped_matmul_fwd() -> None:
    E, S, N, K = 8, 768, 256, 128
    chain = analyze(_build_graph(E, S, N, K))
    assert chain.has_moe
    assert chain.moe.num_experts == E
    assert chain.moe.mode == "none"
    assert chain.moe.offset_dtype == "int32"
    assert (chain.matmul.M, chain.matmul.N, chain.matmul.K) == (S, N, K)
    assert chain.matmul.a_major == "k" and chain.matmul.b_major == "k"
    assert chain.output_dtype == "bf16"


def test_analyzer_offset_dtype_int64() -> None:
    chain = analyze(_build_graph(8, 768, 256, 128, offset_dt=cudnn.data_type.INT64))
    assert chain.moe.offset_dtype == "int64"


def test_analyzer_detects_moe_grouped_matmul_fwd_reduction() -> None:
    chain = analyze(
        _build_graph(
            8,
            768,
            256,
            128,
            reduction_mode=cudnn.reduction_mode.AMAX,
            reduction_dims=(1, 1, 256),
        )
    )
    assert chain.has_moe
    assert len(chain.reductions) == 1
    assert chain.reductions[0].mode == "amax"
    assert chain.reductions[0].source_ref < 0
    assert not chain.reductions[0].grouped_by_moe
    assert [o.source for o in chain.outputs] == ["matmul", "reduction_0"]


def test_analyzer_detects_moe_grouped_matmul_fwd_group_reduction() -> None:
    chain = analyze(
        _build_graph(
            8,
            768,
            256,
            128,
            reduction_mode=cudnn.reduction_mode.AMAX,
            reduction_dims=(8, 1, 1),
            reduction_group_offset=True,
        )
    )
    assert chain.has_moe
    assert len(chain.reductions) == 1
    assert chain.reductions[0].mode == "amax"
    assert chain.reductions[0].dim == (8, 1, 1)
    assert chain.reductions[0].grouped_by_moe


def test_analyzer_rejects_moe_group_reduction_without_group_offset() -> None:
    with pytest.raises(ValueError, match="axis 0"):
        analyze(
            _build_graph(
                8,
                768,
                256,
                128,
                reduction_mode=cudnn.reduction_mode.AMAX,
                reduction_dims=(8, 1, 1),
            )
        )


def test_analyzer_rejects_moe_group_reduction_wrong_offset_dim() -> None:
    with pytest.raises(ValueError, match="groupOffset.*num_groups"):
        analyze(
            _build_graph(
                8,
                768,
                256,
                128,
                reduction_mode=cudnn.reduction_mode.AMAX,
                reduction_dims=(1, 1, 1),
                reduction_group_offset=True,
            )
        )


def test_analyzer_rejects_gather_index_with_wrong_axes() -> None:
    E, S, N, K = 8, 768, 256, 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    tok = g.tensor(
        name="token",
        dim=[1, S, K],
        stride=[S * K, K, 1],
        data_type=cudnn.data_type.BFLOAT16,
    )
    w = g.tensor(
        name="weight",
        dim=[E, K, N],
        stride=[K * N, 1, K],
        data_type=cudnn.data_type.BFLOAT16,
    )
    fto = g.tensor(
        name="first_token_offset",
        dim=[E + 1, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.INT32,
    )
    idx = g.tensor(name="idx", dim=[S, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    out = g.moe_grouped_matmul(
        tok,
        w,
        fto,
        token_index=idx,
        mode=cudnn.moe_grouped_matmul_mode.GATHER,
        name="moe",
    )
    out.set_output(True)
    with pytest.raises(ValueError, match="token_index must have shape"):
        analyze(g)


# --------------------------------------------------------------------------- #
# GATHER: indexed token loads, with independent source and routed extents
# --------------------------------------------------------------------------- #


def _build_gather_graph(
    *,
    source_rows=137,
    rows=257,
    k=96,
    n=96,
    swiglu=False,
    m_major=False,
    index_stride=1,
    activation=False,
    input_dt=cudnn.data_type.BFLOAT16,
    compute_dt=cudnn.data_type.FLOAT,
):
    g = cudnn.pygraph(
        io_data_type=input_dt,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=compute_dt,
    )
    x = g.tensor(name="token", dim=[1, source_rows, k], stride=[source_rows * k, k, 1])
    weights = [g.tensor(name=f"w{i}", dim=[3, k, n], stride=[k * n, 1, k]) for i in range(2 if swiglu else 1)]
    offsets = g.tensor(name="offsets", dim=[7, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    indices = g.tensor(name="indices", dim=[1, rows, 1], stride=[rows * index_stride, index_stride, 1], data_type=cudnn.data_type.INT32)
    mm = [g.moe_grouped_matmul(x, w, offsets, token_index=indices, mode=cudnn.moe_grouped_matmul_mode.GATHER) for w in weights]
    y = g.mul(a=g.swish(input=mm[0]), b=mm[1]) if swiglu else mm[0]
    if activation:
        y = g.swish(input=y)
    y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    if m_major:
        y.set_stride([n * (rows + 7), 1, rows + 7])
    return g, x, weights, offsets, indices, y


def _build_scatter_graph(*, rows=387, n=96, k=96, top_k=3, m_major=False, activation=False):
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    x = g.tensor(name="tokens", dim=[1, rows, k], stride=[rows * k, k, 1])
    w = g.tensor(name="weights", dim=[3, k, n], stride=[k * n, 1, k])
    offsets = g.tensor(name="offsets", dim=[7, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    indices = g.tensor(name="indices", dim=[1, rows, 1], stride=[rows, 1, 1], data_type=cudnn.data_type.INT32)
    ks = g.tensor(name="ks", dim=[1, rows, 1], stride=[rows, 1, 1], data_type=cudnn.data_type.INT32)
    y = g.moe_grouped_matmul(x, w, offsets, token_index=indices, token_ks=ks, top_k=top_k, mode=cudnn.moe_grouped_matmul_mode.SCATTER)
    if activation:
        y = g.swish(input=y)
    y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    if m_major:
        y.set_stride([n * (rows + 5), 1, rows + 5])
    return g, x, w, offsets, indices, ks, y


def _build_combine_graph(*, m_major=False, output_dt=cudnn.data_type.BFLOAT16, rows=387, tokens=129, n=96):
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    x = g.tensor(name="tokens", dim=[1, rows, 96], stride=[rows * 96, 96, 1])
    w = g.tensor(name="weights", dim=[3, 96, n], stride=[96 * n, 1, 96])
    offsets = g.tensor(name="offsets", dim=[7, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    indices = g.tensor(name="indices", dim=[1, rows, 1], stride=[rows, 1, 1], data_type=cudnn.data_type.INT32)
    ks = g.tensor(name="ks", dim=[1, rows, 1], stride=[rows, 1, 1], data_type=cudnn.data_type.INT32)
    scores = g.tensor(name="scores", dim=[1, tokens, 3], stride=[tokens * 3, 3, 1], data_type=cudnn.data_type.FLOAT)
    y = g.moe_grouped_matmul(x, w, offsets, token_index=indices, token_ks=ks, top_k_scores=scores, top_k=3, mode=cudnn.moe_grouped_matmul_mode.COMBINE)
    y.set_output(True).set_data_type(output_dt)
    y.set_stride([n * (tokens + 5), 1, tokens + 5] if m_major else [tokens * (n + 16), n + 16, 1])
    return g, x, w, offsets, indices, ks, scores, y


def _jit_combine(compiler, graph, cfg):
    """A positive COMBINE build: any decline is a failure here (the frost conftest
    would otherwise report an arch-tree decline message as a skip)."""
    try:
        return compiler.jit_from_cudnn_graph(graph, config=cfg)
    except NotImplementedError as exc:
        pytest.fail(f"the sm120 tree declined a COMBINE plan it serves: {exc}")


def test_combine_shape_binding_and_support():
    from cudnn.gemm.frost.sm120 import compiler as C

    g, x, w, offsets, indices, ks, scores, y = _build_combine_graph()
    # Build through the public keyword API as well as analyze the graph.
    z = g.moe_grouped_matmul(x, w, offsets, token_index=indices, token_ks=ks, top_k_scores=scores, top_k=3, mode=cudnn.moe_grouped_matmul_mode.COMBINE)
    assert z.get_dim() == [1, 129, 96]
    with pytest.raises(NotImplementedError, match="COMBINE"):
        analyze(g)  # Two GEMMs cannot share the combined output yet.
    g, *_, scores, y = _build_combine_graph()
    g.validate()
    chain, binding = analyze_with_binding(g)
    assert chain.moe.mode == "combine" and chain.moe.top_k == 3
    assert chain.matmul.M == 387 and chain.matmul.out_dtype == "fp32"
    assert binding.top_k_scores is scores and scores in binding.bound_tensors()
    assert y.get_dim() == [1, 129, 96]
    # The parent asserts both arch trees' executable gates; this copy checks the
    # sm120 one, which admits COMBINE (one GEMM, one fp32/fp16/bf16 output, fp32 acc).
    C._check_executable(chain)
    # The frontend can represent the enum; native operation construction declines.
    cpp = _build_combine_graph()[0]._lower_to_cpp()
    cpp.validate()
    assert '"COMBINE"' in repr(cpp)
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="COMBINE"):
        cpp.build_operation_graph()
    from cudnn._gemm_validate import validate_graph

    scores.set_dim([1, 129, 2])
    with pytest.raises(ValueError, match="top_k_scores"):
        validate_graph(g)
    scores.set_dim([1, 129, 3])
    y.set_output(False)
    g.swish(input=y).set_output(True)
    with pytest.raises(NotImplementedError, match="COMBINE"):
        analyze(g)


@pytest.mark.parametrize("fault", ["missing", "dtype", "shape", "stride", "output", "other_mode"])
def test_combine_metadata_rejected(fault):
    g, *_, scores, y = _build_combine_graph()
    if fault == "missing":
        g.nodes[0].inputs.pop("top_k_scores")
    elif fault == "dtype":
        scores.set_data_type(cudnn.data_type.BFLOAT16)
    elif fault == "shape":
        scores.set_dim([1, 129, 2])
    elif fault == "stride":
        scores.set_stride([774, 6, 2])
    elif fault == "output":
        y.set_dim([1, 387, 96])
    else:
        g.nodes[0].params["mode"] = cudnn.moe_grouped_matmul_mode.SCATTER
    with pytest.raises((ValueError, NotImplementedError), match="top_k_scores|COMBINE"):
        analyze(g)


@requires_sm120
@pytest.mark.no_workspace_shim
@pytest.mark.parametrize("swap_ab", [False, True])
@pytest.mark.parametrize("m_major,output_dt", [(False, torch.bfloat16), (False, torch.float16), (True, torch.float16), (False, torch.float32)])
def test_combine_numerics(swap_ab, m_major, output_dt, monkeypatch):
    from cuda.bindings import runtime
    from cudnn.gemm.frost import compiler as C
    from cudnn.engines.manifest import MANIFEST
    from cudnn.gemm.frost.knobs import GemmKnobs

    dtype = {torch.bfloat16: cudnn.data_type.BFLOAT16, torch.float16: cudnn.data_type.HALF, torch.float32: cudnn.data_type.FLOAT}[output_dt]
    g, x, w, offsets, indices, ks, scores, y = _build_combine_graph(m_major=m_major, output_dt=dtype)
    cfg = replace(by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"), swap_ab=swap_ab)
    compiled = _jit_combine(C, g, cfg)
    assert compiled.store_modes == ("stg",)
    # SM120 stores COMBINE through the STG drain with global reductions: packed
    # vector adds only where one lane's chunk is one token row's adjacent features.
    src = compiled.generated_path.read_text()
    assert "moe_combine_add(" in src or "moe_combine_add_vector(" in src
    assert ("moe_combine_add_vector(_combine_ptr" in src) == (not swap_ab and not m_major)
    torch.manual_seed(734)
    tokens = torch.randn(1, 387, 96, dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(3, 96, 96, dtype=torch.bfloat16, device="cuda") * 0.1
    starts = [0, 3, 3, 134, 257, 258, 383]  # Unused capacity at the end.
    fto = torch.tensor(starts, device="cuda", dtype=torch.int32).view(7, 1, 1)
    destination = torch.randperm(387, device="cuda", dtype=torch.int32)
    index = (destination // 3).view(1, 387, 1)
    slots = (destination % 3).view(1, 387, 1)
    index[0, 10, 0], index[0, 11, 0] = -1, 129
    slots[0, 12, 0], slots[0, 13, 0] = -1, 3
    routing = torch.rand(1, 129, 3, device="cuda")
    routing[0, 0] = 0
    backing = torch.full((96, 134) if m_major else (134, 112), 123, device="cuda", dtype=output_dt)
    out = (backing[:, :129].T if m_major else backing[:129, :96]).unsqueeze(0)
    vp = {x: tokens, w: weights, offsets: fto, indices: index, ks: slots, scores: routing, y: out}
    workspace = torch.empty(compiled.workspace_bytes, dtype=torch.uint8, device="cuda")
    memset_calls = []
    memset, memset2d = runtime.cudaMemsetAsync, runtime.cudaMemset2DAsync

    def record_memset(ptr, value, nbytes, stream):
        memset_calls.append(("1d", int(ptr), int(nbytes), int(stream)))
        assert value == 0
        return memset(ptr, value, nbytes, stream)

    def record_memset2d(ptr, pitch, value, width, height, stream):
        memset_calls.append(("2d", int(ptr), int(pitch), int(width), int(height), int(stream)))
        assert value == 0
        return memset2d(ptr, pitch, value, width, height, stream)

    monkeypatch.setattr(runtime, "cudaMemsetAsync", record_memset)
    monkeypatch.setattr(runtime, "cudaMemset2DAsync", record_memset2d)

    def check_memset(stream=0):
        elem = out.element_size()
        if not m_major and out.stride(1) == 96:
            expected = ("1d", out.data_ptr(), out.numel() * elem, stream)
        else:
            pitch, width, height = (out.stride(2), 129, 96) if m_major else (out.stride(1), 96, 129)
            expected = ("2d", out.data_ptr(), pitch * elem, width * elem, height, stream)
        assert memset_calls == [expected]
        memset_calls.clear()

    def check():
        ref = torch.zeros(129, 96, device="cuda")
        for group, (begin, end) in enumerate(zip(starts, starts[1:])):
            result = tokens[0, begin:end].float() @ weights[group % 3].float().T
            ids, k = index.flatten()[begin:end].long(), slots.flatten()[begin:end].long()
            valid = (ids >= 0) & (ids < 129) & (k >= 0) & (k < 3)
            weighted = result[valid] * routing[0, ids[valid], k[valid], None]
            ref.index_add_(0, ids[valid], weighted.to(output_dt).float())
        tolerance = {torch.bfloat16: 0.025, torch.float16: 0.003, torch.float32: 2e-5}[output_dt]
        torch.testing.assert_close(out[0].float(), ref, atol=tolerance, rtol=tolerance)
        if m_major:
            assert torch.all(backing[:, 129:] == 123)
        else:
            assert torch.all(backing[129:] == 123) and torch.all(backing[:, 96:] == 123)

    compiled(vp, workspace=workspace)
    check_memset()
    check()
    if not swap_ab and not m_major:
        # Reuse the same plan with 16/8/4/2-byte destination alignment. Routing
        # changes row alignment for odd pitches; padding must never be added to.
        for shift, pitch in ((0, 96), (0, 100), (0, 98), (0, 97), (1, 112), (2, 112), (4, 112)):
            storage = torch.full((shift + 134 * pitch + 8,), 123, device="cuda", dtype=output_dt)
            backing = storage[shift:-8].view(134, pitch)
            out = backing[:129, :96].unsqueeze(0)
            vp[y] = out
            compiled(vp, workspace=workspace)
            check_memset()
            check()
            assert torch.all(storage[:shift] == 123) and torch.all(storage[-8:] == 123)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            compiled(vp, workspace=workspace, stream=torch.cuda.current_stream().cuda_stream)
        index.copy_(128 - index)
        slots.copy_(2 - slots)
        routing.mul_(0.5)
        starts[:] = [0, 0, 17, 129, 129, 380, 381]
        fto.copy_(torch.tensor(starts, device="cuda", dtype=torch.int32).view(7, 1, 1))
        out.fill_(99)
        graph.replay()
        check()
        index.fill_(-1)
        graph.replay()
        assert torch.count_nonzero(out) == 0
    finally:
        graph.reset()
    memset_calls.clear()
    out.fill_(77)
    for bad_scores in (routing.to(torch.float16), routing[:, :, :2], torch.empty(1, 129, 6, device="cuda")[:, :, ::2]):
        with pytest.raises(ValueError, match="top_k_scores"):
            compiled({**vp, scores: bad_scores}, workspace=workspace)
    with pytest.raises(ValueError, match="output"):
        compiled({**vp, y: out[:, :-1]}, workspace=workspace)
    with pytest.raises(ValueError, match="workspace"):
        compiled(vp)
    assert not memset_calls
    assert torch.all(out == 77)
    # Public engine lifecycle and exact knob replay must keep COMBINE in FROST.
    g.validate()
    g.build_operation_graph()
    engine_id = next(row.engine_id for row in MANIFEST if row.name == "frost_gemm")
    g.create_execution_plan(engine_id, GemmKnobs.from_config(cfg).to_public())
    g.check_support()
    g.build_plans()
    assert g.selected_engine.name == "frost_gemm"
    ws = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)
    index.copy_((destination // 3).view_as(index))
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    handle = cudnn.create_handle()
    try:
        cudnn.set_stream(handle, side.cuda_stream)
        memset_calls.clear()
        g.execute(vp, ws, handle=handle)
        check_memset(side.cuda_stream)
        torch.cuda.current_stream().wait_stream(side)
        check()
    finally:
        cudnn.destroy_handle(handle)


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True])
@pytest.mark.parametrize(
    "count,n,geometry",
    [
        (1, 96, "64x128x128_16x16x32_cluster1x1_warps4x2"),
        (5, 96, "128x128x128_16x16x32_cluster1x1_warps2x4"),
        (3, 96, "256x128x128_16x16x32_cluster1x1_warps8x1"),  # parent: 512-row tile
        (2, 40, "128x128x128_16x16x32_cluster1x1_warps4x2"),
    ],
)
def test_combine_small_tokens_and_runtime_shape(swap_ab, count, n, geometry):
    from cudnn.gemm.frost import compiler as C

    rows = count * 3 + 2
    g, x, w, offsets, indices, ks, scores, y = _build_combine_graph(rows=rows, tokens=count, n=n, output_dt=cudnn.data_type.FLOAT)
    cfg = replace(by_name("CONFIG_sm120_" + geometry), swap_ab=swap_ab)
    compiled = _jit_combine(C, g, cfg)
    if count == 1 and not swap_ab:
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A])
        g.check_support()
        g.build_plans()
        assert g.selected_engine.name == "frost_gemm"
    token = torch.ones(1, rows, 96, device="cuda", dtype=torch.bfloat16)
    weight = torch.arange(1, 4, device="cuda", dtype=torch.bfloat16).view(3, 1, 1).expand(3, n, 96).contiguous()
    starts = [0, 1, 1, 2, count * 3 - 1, count * 3, count * 3]
    fto = torch.tensor(starts, device="cuda", dtype=torch.int64).to(torch.int32)
    destination = torch.arange(rows, device="cuda", dtype=torch.int32)
    index = (destination // 3).view(1, rows, 1)
    slots = (destination % 3).view(1, rows, 1)
    workspace = torch.empty(compiled.workspace_bytes, device="cuda", dtype=torch.uint8)
    # T is runtime geometry, independent of routed capacity S. No recompilation.
    for t in (count, count + 2):
        routing = torch.rand(1, t, 3, device="cuda")
        out = torch.full((1, t, n), 99.0, device="cuda")
        vp = {x: token, w: weight, offsets: fto, indices: index, ks: slots, scores: routing, y: out}
        compiled(vp, workspace=workspace)
        if count == 1 and not swap_ab:
            public_workspace = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)
            g.execute(vp, public_workspace)
        ref = torch.zeros(t, n, device="cuda")
        for group, (begin, end) in enumerate(zip(starts, starts[1:])):
            for r in range(begin, end):
                ref[r // 3] += 96 * (group % 3 + 1) * routing[0, r // 3, r % 3]
        torch.testing.assert_close(out[0], ref, atol=1e-4, rtol=1e-5)


# COMBINE coverage beyond the parent's: every output dtype x major x orientation
# with exact arithmetic (bit-exact against an fp64 reference), guard bytes around
# the output, CUDA-graph replay into a dirty output, input/layout variants, and
# heavy atomic contention.

_COMBINE_TORCH_DT = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
_COMBINE_CUDNN_DT = {"bf16": cudnn.data_type.BFLOAT16, "fp16": cudnn.data_type.HALF, "fp32": cudnn.data_type.FLOAT}


def _build_combine_graph_ex(*, rows, tokens, n, k, experts, groups, top_k, output_dt, m_major, input_dt=cudnn.data_type.BFLOAT16, weight_n_major=False, pad=8):
    """`_build_combine_graph` with every extent free, plus input dtype / weight major."""
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    x = g.tensor(name="tokens", dim=[1, rows, k], stride=[rows * k, k, 1], data_type=input_dt)
    w = g.tensor(name="weights", dim=[experts, k, n], stride=[k * n, n, 1] if weight_n_major else [k * n, 1, k], data_type=input_dt)
    offsets = g.tensor(name="offsets", dim=[groups + 1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    indices = g.tensor(name="indices", dim=[1, rows, 1], stride=[rows, 1, 1], data_type=cudnn.data_type.INT32)
    ks = g.tensor(name="ks", dim=[1, rows, 1], stride=[rows, 1, 1], data_type=cudnn.data_type.INT32)
    scores = g.tensor(name="scores", dim=[1, tokens, top_k], stride=[tokens * top_k, top_k, 1], data_type=cudnn.data_type.FLOAT)
    y = g.moe_grouped_matmul(x, w, offsets, token_index=indices, token_ks=ks, top_k_scores=scores, top_k=top_k, mode=cudnn.moe_grouped_matmul_mode.COMBINE)
    y.set_output(True).set_data_type(output_dt)
    y.set_stride([n * (tokens + pad), 1, tokens + pad] if m_major else [tokens * (n + pad), n + pad, 1])
    return g, x, w, offsets, indices, ks, scores, y


def _combine_guarded_output(tokens, n, m_major, dtype, *, pad=8, sentinel=-7.0, shift=0):
    """The output view inside a sentinel-filled allocation: `pad` columns (or rows,
    M-major) past the logical extent, two spare rows, and a head of `shift` plus a
    32-element tail. Returns (storage, view, inside) -- `inside` marks the view."""
    rows_, pitch = (n + 2, tokens + pad) if m_major else (tokens + 2, n + pad)
    storage = torch.full((shift + rows_ * pitch + 32,), sentinel, device="cuda", dtype=dtype)
    backing = storage[shift : shift + rows_ * pitch].view(rows_, pitch)
    view = (backing[:n, :tokens].T if m_major else backing[:tokens, :n]).unsqueeze(0)
    inside = torch.zeros_like(storage, dtype=torch.bool)
    inside_backing = inside[shift : shift + rows_ * pitch].view(rows_, pitch)
    (inside_backing[:n, :tokens] if m_major else inside_backing[:tokens, :n]).fill_(True)
    return storage, view, inside


def _combine_exact_data(rows, tokens, n, k, experts, top_k, starts, *, seed, input_dtype=torch.bfloat16, invalid=True):
    """Integer-valued data whose every contribution and partial sum is exactly
    representable in bf16 (so in every output dtype, in any atomic order):
    tokens carry two +-1 entries per row, weights are integers in [-2, 2], so an
    FC2 value is an integer in [-4, 4]; scores are 1/4, 1/2 or 1; each token
    receives ceil(rows / tokens) contributions. The bf16 partial sums are then
    multiples of 1/4 below 4 * ceil(rows / tokens) in magnitude -- exact while that
    stays <= 64 (8 significant bits)."""
    assert -(-rows // tokens) * 4 <= 64
    gen = torch.Generator(device="cuda").manual_seed(seed)
    token = torch.zeros(1, rows, k, device="cuda")
    for _ in range(2):
        pos = torch.randint(0, k, (rows,), device="cuda", generator=gen)
        sign = torch.randint(0, 2, (rows,), device="cuda", generator=gen).float() * 2 - 1
        token[0, torch.arange(rows, device="cuda"), pos] = sign
    weight = torch.randint(-2, 3, (experts, n, k), device="cuda", generator=gen).float()
    perm = torch.randperm(rows, device="cuda", generator=gen)
    index = (perm % tokens).to(torch.int32).view(1, rows, 1)
    slots = torch.randint(0, top_k, (1, rows, 1), device="cuda", generator=gen, dtype=torch.int32)
    if invalid:
        index[0, 1, 0], index[0, 2, 0] = -1, tokens
        slots[0, 3, 0], slots[0, 4, 0] = -1, top_k
    scores = torch.tensor([0.25, 0.5, 1.0], device="cuda")[torch.randint(0, 3, (1, tokens, top_k), device="cuda", generator=gen)]
    return token.to(input_dtype), weight.to(input_dtype), index, slots, scores.contiguous()


def _combine_reference(token, weight, index, slots, scores, starts, n):
    """fp64 COMBINE: every in-group routed row with an in-range token and slot adds
    score * (token @ weight[group % E].T) into its token row."""
    tokens, top_k = scores.shape[1], scores.shape[2]
    ref = torch.zeros(tokens, n, device="cuda", dtype=torch.float64)
    experts = weight.shape[0]
    for group, (begin, end) in enumerate(zip(starts, starts[1:])):
        if end <= begin:
            continue
        result = token[0, begin:end].double() @ weight[group % experts].double().T
        ids, k = index.flatten()[begin:end].long(), slots.flatten()[begin:end].long()
        valid = (ids >= 0) & (ids < tokens) & (k >= 0) & (k < top_k)
        ref.index_add_(0, ids[valid], result[valid] * scores[0, ids[valid], k[valid], None].double())
    return ref


def _run_combine_exact(
    cfg, *, swap_ab, m_major, out, input_dt="bf16", weight_n_major=False, offset_alignment=None, int64_offsets=False, top_k=3, honor_alignment=True, pad=8
):
    from cudnn.gemm.frost import compiler as C
    from cudnn.gemm.frost.kernel_registry import select_template

    rows, tokens, n, k, experts = 304, 101, 200, 160, 4
    # Empty groups, a one-row group, groups spanning several M tiles, a group per
    # expert twice over (G = 2E), and unused routed capacity past the last group.
    # A promised alignment is honored unless the case breaks it on purpose: the
    # drain still re-checks every element against its routed group's end.
    sizes = [0, 40, 136, 8, 0, 64, 40, 8] if offset_alignment and honor_alignment else [0, 37, 129, 1, 0, 64, 40, 21]
    starts = [sum(sizes[:i]) for i in range(len(sizes) + 1)]
    in_dt = {
        "bf16": (cudnn.data_type.BFLOAT16, torch.bfloat16),
        "fp16": (cudnn.data_type.HALF, torch.float16),
        "fp8": (cudnn.data_type.FP8_E4M3, torch.float8_e4m3fn),
    }
    g, x, w, offsets, indices, ks, scores, y = _build_combine_graph_ex(
        rows=rows,
        tokens=tokens,
        n=n,
        k=k,
        experts=experts,
        groups=len(sizes),
        top_k=top_k,
        output_dt=_COMBINE_CUDNN_DT[out],
        m_major=m_major,
        input_dt=in_dt[input_dt][0],
        weight_n_major=weight_n_major,
        pad=pad,
    )
    if offset_alignment:
        offsets.set_alignment_value(offset_alignment)
    if int64_offsets:
        offsets.set_data_type(cudnn.data_type.INT64)
    compiled = _jit_combine(C, g, replace(cfg, swap_ab=swap_ab))
    assert compiled.store_modes == ("stg",)
    # The registry funnel (candidate listing) admits what the build accepted.
    assert select_template(compiled.chain, compiled.config).accepts(compiled.chain, compiled.config) is None
    vsize = compiled.vec_bytes_epi // torch.empty((), dtype=_COMBINE_TORCH_DT[out]).element_size()
    if swap_ab and offset_alignment:
        # The promised group alignment lets the swap-AB drain hand out multi-element
        # chunks; every element still re-checks its routed group's end.
        assert vsize > 1
    token, weight, index, slots, routing = _combine_exact_data(rows, tokens, n, k, experts, top_k, starts, seed=4242, input_dtype=in_dt[input_dt][1])
    if weight_n_major:
        weight = weight.transpose(1, 2).contiguous().transpose(1, 2)
    fto = torch.tensor(starts, device="cuda", dtype=torch.int64 if int64_offsets else torch.int32).view(-1, 1, 1)
    storage, out_view, inside = _combine_guarded_output(tokens, n, m_major, _COMBINE_TORCH_DT[out], pad=pad)
    vp = {x: token, w: weight.transpose(1, 2), offsets: fto, indices: index, ks: slots, scores: routing, y: out_view}
    workspace = torch.empty(compiled.workspace_bytes, dtype=torch.uint8, device="cuda")

    def check():
        ref = _combine_reference(token, weight, index, slots, routing, starts, n)
        torch.testing.assert_close(out_view[0].double(), ref, atol=0, rtol=0)
        assert torch.all(storage[~inside] == -7.0), "COMBINE wrote outside the output view"

    out_view.fill_(55.0)  # the plan zero-fills the view itself
    compiled(vp, workspace=workspace)
    check()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            compiled(vp, workspace=workspace, stream=torch.cuda.current_stream().cuda_stream)
        new = _combine_exact_data(rows, tokens, n, k, experts, top_k, starts, seed=99, input_dtype=in_dt[input_dt][1])
        token.copy_(new[0])
        index.copy_(new[2])
        slots.copy_(new[3])
        routing.copy_(new[4])
        out_view.fill_(-3.0)
        graph.replay()
        check()
        graph.replay()  # replays from the dirty result of the previous one
        check()
    finally:
        graph.reset()


@requires_sm120
@pytest.mark.parametrize("out", ["bf16", "fp16", "fp32"])
@pytest.mark.parametrize("m_major", [False, True], ids=["n_major", "m_major"])
@pytest.mark.parametrize("swap_ab", [False, True], ids=["token_by_weight", "weight_by_token"])
def test_combine_exact_every_layout(swap_ab, m_major, out):
    _run_combine_exact(by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"), swap_ab=swap_ab, m_major=m_major, out=out)


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True], ids=["token_by_weight", "weight_by_token"])
@pytest.mark.parametrize(
    "case",
    [
        "fp8_inputs",
        "fp16_inputs",
        "n_major_weight",
        "aligned_int64_offsets",
        "broken_alignment_promise",
        "odd_output_pitch",
        "top_k_1",
        "geometry_64x256",
        "geometry_256x64",
    ],
)
def test_combine_exact_variants(case, swap_ab):
    geometry = {
        "geometry_64x256": "CONFIG_sm120_64x256x128_16x16x32_cluster1x1_warps2x4",
        "geometry_256x64": "CONFIG_sm120_256x64x128_16x16x32_cluster1x1_warps8x1",
    }.get(case, "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2")
    _run_combine_exact(
        by_name(geometry),
        swap_ab=swap_ab,
        m_major=case == "n_major_weight",
        out="bf16",
        input_dt={"fp8_inputs": "fp8", "fp16_inputs": "fp16"}.get(case, "bf16"),
        weight_n_major=case == "n_major_weight",
        offset_alignment=8 if case in ("aligned_int64_offsets", "broken_alignment_promise") else None,
        int64_offsets=case == "aligned_int64_offsets",
        top_k=1 if case == "top_k_1" else 3,
        honor_alignment=case != "broken_alignment_promise",
        # A declared pitch of N + 1 admits one-element chunks only: the reductions
        # take their width from the address at run time, so COMBINE still serves it.
        pad=1 if case == "odd_output_pitch" else 8,
    )


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True], ids=["token_by_weight", "weight_by_token"])
@pytest.mark.parametrize("out,tokens", [("fp32", 1), ("fp16", 4), ("bf16", 16)])
def test_combine_heavy_contention(swap_ab, out, tokens):
    """4096 routed rows onto 1 / 4 / 16 tokens: every output element takes 4096 /
    1024 / 256 concurrent reductions (vector adds on the token-by-weight kernel,
    per-element adds after swap-AB). Each FC2 value is exactly 1 or 2 (the token is
    1/64 over K=64, the expert weight row constant), and the scores are 1, so every
    partial sum is an integer the output dtype holds exactly -- the result is
    order-independent and compared bit for bit."""
    from cudnn.gemm.frost import compiler as C

    rows, n, k, experts = 4096, 256, 64, 4
    starts = [0, 1000, 1001, 3000, 4096]
    g, x, w, offsets, indices, ks, scores, y = _build_combine_graph_ex(
        rows=rows, tokens=tokens, n=n, k=k, experts=experts, groups=4, top_k=experts, output_dt=_COMBINE_CUDNN_DT[out], m_major=False
    )
    compiled = _jit_combine(C, g, replace(by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"), swap_ab=swap_ab))
    token = torch.full((1, rows, k), 1.0 / 64, device="cuda", dtype=torch.bfloat16)
    # bf16 sums stay <= 256 with unit contributions; fp16 <= 2048 and fp32 hold 1s and 2s.
    per_feature = torch.ones(experts, n, device="cuda") if out == "bf16" else (torch.arange(n, device="cuda") % 2 + 1).float().expand(experts, n)
    weight = per_feature[:, :, None].expand(experts, n, k).to(torch.bfloat16).contiguous()
    perm = torch.randperm(rows, device="cuda")
    index = (perm % tokens).to(torch.int32).view(1, rows, 1)
    slots = (perm % experts).to(torch.int32).view(1, rows, 1)
    routing = torch.ones(1, tokens, experts, device="cuda")
    storage, out_view, inside = _combine_guarded_output(tokens, n, False, _COMBINE_TORCH_DT[out])
    vp = {
        x: token,
        w: weight.transpose(1, 2),
        offsets: torch.tensor(starts, device="cuda", dtype=torch.int32),
        indices: index,
        ks: slots,
        scores: routing,
        y: out_view,
    }
    workspace = torch.empty(compiled.workspace_bytes, dtype=torch.uint8, device="cuda")
    ref = _combine_reference(token, weight, index, slots, routing, starts, n)
    assert ref.abs().max() <= {"fp32": 2**24, "fp16": 2048, "bf16": 256}[out]
    for _ in range(3):
        compiled(vp, workspace=workspace)
        torch.testing.assert_close(out_view[0].double(), ref, atol=0, rtol=0)
        assert torch.all(storage[~inside] == -7.0)


@requires_sm120
@pytest.mark.no_workspace_shim
def test_combine_runtime_contract_precedes_the_memset(monkeypatch):
    """Every runtime rejection lands before the zero-fill: an N the compiled chunk
    does not divide, a wrong output dtype, an overlapping output and an output of
    the other major leave the caller's buffer untouched and issue no memset."""
    from cuda.bindings import runtime
    from cudnn.gemm.frost import compiler as C

    g, x, w, offsets, indices, ks, scores, y = _build_combine_graph()
    compiled = _jit_combine(C, g, by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"))
    vsize = compiled.vec_bytes_epi // 2
    assert vsize > 1
    calls = []
    monkeypatch.setattr(runtime, "cudaMemsetAsync", lambda *a: calls.append(a))
    monkeypatch.setattr(runtime, "cudaMemset2DAsync", lambda *a: calls.append(a))
    rows, tokens = 387, 129
    token = torch.randn(1, rows, 96, device="cuda", dtype=torch.bfloat16)
    fto = torch.tensor([0, 3, 3, 134, 257, 258, 383], device="cuda", dtype=torch.int32).view(7, 1, 1)
    index = torch.randint(0, tokens, (1, rows, 1), device="cuda", dtype=torch.int32)
    slots = torch.randint(0, 3, (1, rows, 1), device="cuda", dtype=torch.int32)
    routing = torch.rand(1, tokens, 3, device="cuda")
    workspace = torch.empty(compiled.workspace_bytes, dtype=torch.uint8, device="cuda")
    n_bad = 96 + vsize // 2
    bad_n = {
        x: token,
        # (E, N, K): a buffer that differs from the declaration is taken in kernel order.
        w: torch.randn(3, n_bad, 96, device="cuda", dtype=torch.bfloat16),
        offsets: fto,
        indices: index,
        ks: slots,
        scores: routing,
        y: torch.full((1, tokens, n_bad), 77, device="cuda", dtype=torch.bfloat16),
    }
    with pytest.raises(ValueError, match="divisible"):
        compiled(bad_n, workspace=workspace)
    assert torch.all(bad_n[y] == 77)
    good_w = torch.randn(3, 96, 96, device="cuda", dtype=torch.bfloat16)
    for bad_out, match in (
        (torch.full((1, tokens, 96), 77, device="cuda", dtype=torch.float16), "bf16"),
        (torch.full((1, 1, 96), 77, device="cuda", dtype=torch.bfloat16).expand(1, tokens, 96), "non-overlapping"),
        (torch.full((1, 96, tokens), 77, device="cuda", dtype=torch.bfloat16).transpose(1, 2), "N-major"),
    ):
        with pytest.raises(ValueError, match=match):
            compiled({**bad_n, w: good_w, y: bad_out}, workspace=workspace)
        assert torch.all(bad_out == 77)
    assert not calls


@requires_sm120
@pytest.mark.no_workspace_shim
@pytest.mark.parametrize("swap_ab", [False, True], ids=["token_by_weight", "weight_by_token"])
def test_combine_operand_checks_precede_the_memset(swap_ab, monkeypatch):
    """COMBINE zero-fills the output before the launch, and the launch binding is
    what would check some operand facts: a token, weight or offsets buffer of
    another dtype, a weight whose K is not the token's, a token row pitch that is
    not a 16-byte multiple, rank-2 offsets, and a routing / output buffer off the
    GPU (or, the workspace included, on another GPU than the rest) each raise
    before the zero-fill (no memset, output untouched), by the binding's own rules
    -- offsets of the declared width but unsigned, and a token whose row pitch is
    padded to another 16-byte multiple, still run exactly."""
    from cuda.bindings import runtime
    from cudnn.gemm.frost import compiler as C

    rows, tokens, n, k, experts = 304, 101, 200, 160, 4
    sizes = [0, 37, 129, 1, 0, 64, 40, 21]
    starts = [sum(sizes[:i]) for i in range(len(sizes) + 1)]
    g, x, w, offsets, indices, ks, scores, y = _build_combine_graph_ex(
        rows=rows, tokens=tokens, n=n, k=k, experts=experts, groups=len(sizes), top_k=3, output_dt=cudnn.data_type.BFLOAT16, m_major=False
    )
    compiled = _jit_combine(C, g, replace(by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"), swap_ab=swap_ab))
    token, weight, index, slots, routing = _combine_exact_data(rows, tokens, n, k, experts, 3, starts, seed=31)
    fto = torch.tensor(starts, device="cuda", dtype=torch.int32).view(-1, 1, 1)
    storage, out_view, inside = _combine_guarded_output(tokens, n, False, torch.bfloat16)
    vp = {x: token, w: weight.transpose(1, 2), offsets: fto, indices: index, ks: slots, scores: routing, y: out_view}
    workspace = torch.empty(compiled.workspace_bytes, dtype=torch.uint8, device="cuda")
    calls = []
    memset, memset2d = runtime.cudaMemsetAsync, runtime.cudaMemset2DAsync
    monkeypatch.setattr(runtime, "cudaMemsetAsync", lambda *a: calls.append(a) or memset(*a))
    monkeypatch.setattr(runtime, "cudaMemset2DAsync", lambda *a: calls.append(a) or memset2d(*a))
    out_view.fill_(77)

    def padded_token(pad):
        rows_pitched = torch.zeros(1, rows, k + pad, device="cuda", dtype=token.dtype)
        rows_pitched[..., :k] = token
        return rows_pitched[..., :k]

    for override, match in (
        ({x: token.view(torch.float16)}, "token must be bfloat16"),
        ({w: weight.transpose(1, 2).view(torch.float16)}, "weight must be bfloat16"),
        ({offsets: fto.to(torch.int64)}, "first_token_offset must be int32"),
        # (E, N, K'): a buffer that differs from the declaration is taken in kernel order.
        ({w: torch.zeros(experts, n, k - 32, device="cuda", dtype=torch.bfloat16)}, "must share"),
        ({x: padded_token(4)}, "token strides must be multiples of 16 bytes"),
        ({offsets: fto.view(-1, 1)}, "first_token_offset must be rank-1"),
        ({scores: routing.cpu()}, "top_k_scores must be a CUDA buffer"),
        ({indices: index.cpu()}, "token_index must be a CUDA buffer"),
        ({y: out_view.cpu()}, "output must be a CUDA buffer"),
    ):
        with pytest.raises(ValueError, match=match):
            compiled({**vp, **override}, workspace=workspace)
    assert not calls
    assert torch.all(out_view == 77)

    # The binding gives every tensor argument -- the carved workspace included -- one
    # device index. This process sees one GPU, so a stand-in reports the other one.
    class OnAnotherGpu:
        def __dlpack_device__(self):
            return (2, 1)  # (kDLCUDA, 1)

    operands = ([weight], [token]) if swap_ab else ([token], [weight])
    good_routing = (index.view(-1), slots.view(-1), routing.view(tokens, 3))
    for routing_bufs, ws, match in (
        ((*good_routing[:2], OnAnotherGpu()), workspace, r"top_k_scores must be a CUDA buffer on the same device .*; got cuda:1 \(the \w+ is on cuda:0\)"),
        (good_routing, OnAnotherGpu(), r"workspace must be a CUDA buffer on the same device .*; got cuda:1"),
    ):
        with pytest.raises(ValueError, match=match):
            C._check_moe_combine_operands(compiled.chain, *operands, fto.view(-1), [], [], routing_bufs, [out_view], k, ws)
    C._check_moe_combine_operands(compiled.chain, *operands, fto.view(-1), [], [], good_routing, [out_view], k, workspace)
    ref = _combine_reference(token, weight, index, slots, routing, starts, n)
    for override in ({offsets: fto.view(torch.uint32)}, {x: padded_token(8)}):
        out_view.fill_(77)
        compiled({**vp, **override}, workspace=workspace)
        assert len(calls) == 1  # the zero-fill
        calls.clear()
        torch.testing.assert_close(out_view[0].double(), ref, atol=0, rtol=0)
        assert torch.all(storage[~inside] == -7.0)


def test_combine_has_no_gather_form():
    """COMBINE's token input stays grouped (as SCATTER's); the routing mode is one
    enum, so a GATHER node carrying routing scores is a malformed graph, not a
    GATHER + COMBINE fusion an engine could admit."""
    g, x, w, offsets, indices, y = _build_gather_graph()
    node = g.nodes[0]
    node.inputs["top_k_scores"] = g.tensor(name="scores", dim=[1, 129, 3], stride=[387, 3, 1], data_type=cudnn.data_type.FLOAT)
    with pytest.raises(ValueError, match="top_k_scores"):
        analyze(g)


@requires_sm120
@pytest.mark.parametrize(
    "case,in_dt,acc_dt,out_dt",
    [
        ("int32_accumulation", "INT8", "INT32", "FLOAT"),
        ("fp8_output", "FP8_E4M3", "FLOAT", "FP8_E4M3"),
        ("admitted", "BFLOAT16", "FLOAT", "BFLOAT16"),
    ],
)
def test_combine_admitted_set_in_the_build_and_the_funnel(case, in_dt, acc_dt, out_dt):
    """The admitted set is the parent's: int8 inputs accumulate in int32 and an fp8
    output takes no reduction, so COMBINE (an fp32 score multiply into an
    fp32/fp16/bf16 output) declines both -- in the build AND in the registry funnel,
    which lists no sm120 candidate for a graph the build declines, and lists the
    admitted control."""
    from cudnn.gemm.frost import kernel_registry as R
    from cudnn.gemm.frost.sm120 import compiler as C

    dt = cudnn.data_type
    g = cudnn.pygraph(io_data_type=getattr(dt, in_dt), intermediate_data_type=dt.FLOAT, compute_data_type=getattr(dt, acc_dt))
    x = g.tensor(name="tokens", dim=[1, 387, 96], stride=[387 * 96, 96, 1], data_type=getattr(dt, in_dt))
    w = g.tensor(name="weights", dim=[3, 96, 96], stride=[96 * 96, 1, 96], data_type=getattr(dt, in_dt))
    offsets = g.tensor(name="offsets", dim=[7, 1, 1], stride=[1, 1, 1], data_type=dt.INT32)
    indices = g.tensor(name="indices", dim=[1, 387, 1], stride=[387, 1, 1], data_type=dt.INT32)
    ks = g.tensor(name="ks", dim=[1, 387, 1], stride=[387, 1, 1], data_type=dt.INT32)
    scores = g.tensor(name="scores", dim=[1, 129, 3], stride=[387, 3, 1], data_type=dt.FLOAT)
    mode = cudnn.moe_grouped_matmul_mode.COMBINE
    y = g.moe_grouped_matmul(x, w, offsets, token_index=indices, token_ks=ks, top_k_scores=scores, top_k=3, mode=mode, compute_data_type=getattr(dt, acc_dt))
    y.set_output(True).set_data_type(getattr(dt, out_dt))
    chain = analyze(g)
    listed = [cfg.name for tmpl, cfg in R.candidates(chain, sweep_swap_ab=True) if tmpl.pipeline == "sm120"]
    if case == "admitted":
        C._check_executable(chain)
        assert listed, "the funnel lists no sm120 candidate for an admitted COMBINE graph"
        return
    assert (chain.matmul.accum_dtype, chain.output_dtype) == {"int32_accumulation": ("int32", "fp32"), "fp8_output": ("fp32", "fp8_e4m3")}[case]
    with pytest.raises(NotImplementedError, match="FP32 MMA accumulation"):
        C._check_executable(chain)
    assert listed == []
    assert "FP32 MMA accumulation" in R.Sm120KernelTemplate.moe_combine_reject(chain)


def test_scatter_shape_and_binding():
    g, _, _, _, indices, ks, y = _build_scatter_graph()
    chain, binding = analyze_with_binding(g)
    assert chain.moe.mode == "scatter"
    assert chain.moe.top_k == 3
    assert chain.matmul.M == 387
    assert binding.token_index is indices
    assert binding.token_ks is ks
    assert ks in binding.bound_tensors()
    assert y.get_dim() == [1, 387, 96]


@pytest.mark.parametrize("fault", ["missing_ks", "ks_dtype", "ks_stride", "ks_shape", "index_rows", "top_k_zero", "top_k_large", "top_k_indivisible"])
def test_scatter_metadata_rejected(fault):
    g, _, _, _, indices, ks, _ = _build_scatter_graph()
    if fault == "missing_ks":
        g.nodes[0].inputs.pop("token_ks")
        with pytest.raises(ValueError, match="token_ks"):
            g.validate()
    elif fault == "ks_dtype":
        ks.set_data_type(cudnn.data_type.INT64)
    elif fault == "ks_stride":
        ks.set_stride([774, 2, 1])
    elif fault == "ks_shape":
        ks.set_dim([1, 386, 1])
    elif fault == "index_rows":
        indices.set_dim([1, 384, 1])
        ks.set_dim([1, 384, 1])
    else:
        g.nodes[0].params["top_k"] = {"top_k_zero": 0, "top_k_large": 4, "top_k_indivisible": 2}[fault]
    with pytest.raises((ValueError, NotImplementedError), match="token_ks|token_index|top_k"):
        analyze_with_binding(g)


@pytest.mark.parametrize("family", ["sm120", "sm120"])
@pytest.mark.parametrize("feature", ["quant", "reduction", "packed_output"])
def test_scatter_epilogue_support_surface(family, feature):
    from importlib import import_module

    compiler = import_module(f"cudnn.gemm.frost.{family}.compiler")
    g, *_, y = _build_scatter_graph()
    if feature == "quant":
        q, sf = g.block_scale_quantize(input=y, block_size=32)
        q.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
        sf.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)
    elif feature == "reduction":
        r = g.reduction(input=y, mode=cudnn.reduction_mode.ADD)
        r.set_dim([1, 1, 96]).set_stride([96, 96, 1]).set_output(True).set_data_type(cudnn.data_type.FLOAT)
    else:
        y.set_data_type(cudnn.data_type.FP4_E2M1)
    with pytest.raises(NotImplementedError, match="SCATTER"):
        compiler._check_executable(analyze(g))


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True])
@pytest.mark.parametrize(
    "geometry",
    [
        "128x128x128_16x16x32_cluster1x1_warps4x2",
        "64x128x128_16x16x32_cluster1x1_warps4x2",
        "64x128x128_16x16x32_cluster1x1_warps2x4",
        "256x128x128_16x16x32_cluster1x1_warps4x2",
    ],
)
def test_scatter_tma_geometry(geometry, swap_ab):
    from cudnn.gemm.frost import compiler as C

    _run_scatter_numerics(C, replace(by_name("CONFIG_sm120_" + geometry), swap_ab=swap_ab))


def _run_scatter_numerics(compiler, cfg, *, m_major=False, stg=False, activation=False):
    g, x, w, offsets, indices, ks, y = _build_scatter_graph(m_major=m_major, activation=activation)
    if stg:
        with compiler.force_stg_epi():
            compiled = compiler.jit_from_cudnn_graph(g, config=cfg)
    else:
        compiled = compiler.jit_from_cudnn_graph(g, config=cfg)
    if cfg.pipeline == "sm120":
        # The parent pins sm100's TMA scatter4 store here (`cfg.pipeline == "sm100"`).
        # SM120 has no TMA-store epilogue (sm120/compiler.py _TMA_STORE_EPI_PIPELINES
        # excludes it): SCATTER always stores through an indexed STG, whose row comes
        # from moe_scatter_row(token_index, token_ks, ...), for every layout/arm.
        assert compiled.tma_slots == frozenset()
        assert compiled.store_modes == ("stg",)
        src = compiled.generated_path.read_text()
        assert "tma_scatter4(" not in src and "moe_scatter_row(" in src
    torch.manual_seed(136)
    tokens = torch.randn(1, 387, 96, dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(3, 96, 96, dtype=torch.bfloat16, device="cuda") * 0.1
    starts = [0, 3, 3, 134, 257, 258, 387]
    fto = torch.tensor(starts, device="cuda", dtype=torch.int32).view(7, 1, 1)
    destination = torch.randperm(387, device="cuda", dtype=torch.int32)
    index = (destination // 3).view(1, 387, 1)
    slots = (destination % 3).view(1, 387, 1)
    backing = torch.full((96, 392) if m_major else (392, 112), 123, device="cuda", dtype=torch.bfloat16)
    out = (backing[:, :387].T if m_major else backing[:387, :96]).unsqueeze(0)
    vp = {x: tokens, w: weights, offsets: fto, indices: index, ks: slots, y: out}
    workspace = torch.empty(compiled.workspace_bytes, dtype=torch.uint8, device="cuda")

    def check():
        grouped = torch.empty(387, 96, device="cuda", dtype=torch.float32)
        for group, begin in enumerate(starts[:-1]):
            end = starts[group + 1]
            grouped[begin:end] = tokens[0, begin:end].float() @ weights[group % 3].float().T
        if activation:
            grouped = torch.nn.functional.silu(grouped)
        ref = torch.empty_like(grouped)
        ref[(index.flatten() * 3 + slots.flatten()).long()] = grouped
        torch.testing.assert_close(out[0].float(), ref.to(torch.bfloat16).float(), atol=0.03, rtol=0.02)
        if m_major:
            assert torch.all(backing[:, 387:] == 123)
        else:
            assert torch.all(backing[387:] == 123)
            assert torch.all(backing[:, 96:] == 123)

    compiled(vp, workspace=workspace)
    check()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            compiled(vp, workspace=workspace, stream=torch.cuda.current_stream().cuda_stream)
        index.copy_(128 - index)
        slots.copy_(2 - slots)
        starts[:] = [0, 0, 17, 129, 129, 386, 387]
        fto.copy_(torch.tensor(starts, device="cuda", dtype=torch.int32).view(7, 1, 1))
        graph.replay()
        check()
    finally:
        graph.reset()


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True])
@pytest.mark.parametrize("m_major,stg,activation", [(False, False, False), (False, True, True), (True, False, True)])
def test_scatter_numerics(swap_ab, m_major, stg, activation):
    from cudnn.gemm.frost import compiler as C

    cfg = replace(by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"), swap_ab=swap_ab)
    _run_scatter_numerics(C, cfg, m_major=m_major, stg=stg, activation=activation)


@requires_matmul_gpu
@pytest.mark.parametrize("m_major,activation", [(False, False), (False, True), (True, True)])
def test_scatter_numerics_sm120(m_major, activation):
    from cudnn.gemm.frost.sm120 import compiler as C

    _run_scatter_numerics(C, by_name(_SM120_CFG), m_major=m_major, activation=activation)


def _run_scatter_coordinates(compiler, cfg):
    from cudnn.engines.manifest import MANIFEST
    from cudnn.gemm.frost.knobs import GemmKnobs

    g, x, w, offsets, indices, ks, base = _build_scatter_graph()
    base.set_output(False).set_data_type(cudnn.data_type.FLOAT)
    base = g.identity(input=base)
    base.set_data_type(cudnn.data_type.FLOAT)
    value = base
    aux_buffers = {}
    for name, shape in (("rows", (1, 387, 1)), ("columns", (1, 1, 96)), ("elements", (1, 387, 96))):
        buf = torch.arange(shape[1] * shape[2], device="cuda", dtype=torch.float32).view(shape) % 7
        aux = g.tensor(name=name, dim=list(shape), stride=list(buf.stride()), data_type=cudnn.data_type.FLOAT)
        aux_buffers[aux] = buf
        value = g.add(a=value, b=aux)
    for axis in (1, 2):
        value = g.add(a=value, b=g.gen_index(input=base, axis=axis))
    base.set_output(True)
    value.set_output(True).set_data_type(cudnn.data_type.BFLOAT16).set_stride([96 * 392, 1, 392])
    compiled = compiler.jit_from_cudnn_graph(g, config=cfg)
    # The parent expects ("tma", "stg") only for `cfg.pipeline == "sm100"` (N-major
    # base on TMA scatter4). SM120 has no TMA-store epilogue: both outputs use STG.
    assert compiled.store_modes == ("stg", "stg")
    tokens = (torch.arange(387, device="cuda") % 7).to(torch.bfloat16).view(1, 387, 1).expand(1, 387, 96).contiguous()
    weights = torch.arange(1, 4, device="cuda", dtype=torch.bfloat16).view(3, 1, 1).expand(3, 96, 96).contiguous()
    starts = [0, 3, 3, 134, 257, 258, 387]
    fto = torch.tensor(starts, device="cuda", dtype=torch.int32)
    destination = torch.randperm(387, device="cuda", dtype=torch.int32)
    index, slots = (destination // 3).view(1, 387, 1), (destination % 3).view(1, 387, 1)
    out_base = torch.empty(1, 387, 96, device="cuda", dtype=torch.float32)
    backing = torch.full((96, 392), 123, device="cuda", dtype=torch.bfloat16)
    out_value = backing[:, :387].T.unsqueeze(0)
    vp = {x: tokens, w: weights, offsets: fto, indices: index, ks: slots, base: out_base, value: out_value, **aux_buffers}
    ref = torch.empty_like(out_base)
    for group, (begin, end) in enumerate(zip(starts, starts[1:])):
        ref[0, destination[begin:end].long()] = tokens[0, begin:end].float() * (96 * (group % 3 + 1))
    expected = ref + sum(aux_buffers.values()) + torch.arange(387, device="cuda").view(1, 387, 1) + torch.arange(96, device="cuda")

    g.validate()
    g.build_operation_graph()
    engine_id = next(row.engine_id for row in MANIFEST if row.name == "frost_gemm")
    g.create_execution_plan(engine_id, GemmKnobs.from_config(cfg).to_public())
    g.check_support()
    g.build_plans()
    assert g.selected_engine.name == "frost_gemm"
    workspace = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)
    g.execute(vp, workspace)
    torch.testing.assert_close(out_base, ref, atol=0, rtol=0)
    torch.testing.assert_close(out_value, expected.to(torch.bfloat16), atol=0, rtol=0)
    assert torch.all(backing[:, 387:] == 123)
    for bad_ks in (slots.to(torch.int64), slots[:, :-1], torch.empty(1, 387, 2, device="cuda", dtype=torch.int32)[:, :, :1]):
        with pytest.raises(ValueError, match="token_ks"):
            compiled({**vp, ks: bad_ks})


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True])
def test_scatter_coordinates_and_public_replay(swap_ab):
    from cudnn.gemm.frost import compiler as C

    _run_scatter_coordinates(C, replace(by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"), swap_ab=swap_ab))


@requires_matmul_gpu
def test_scatter_coordinates_and_public_replay_sm120():
    from cudnn.gemm.frost.arch_family import active_family
    from cudnn.gemm.frost.sm120 import compiler as C

    if active_family() != "sm120":
        pytest.skip("set CUDNN_FRONTEND_GEMM_ARCH_FAMILY=sm120 before import to replay the public SM120 plan on SM10.x")
    _run_scatter_coordinates(C, by_name(_SM120_CFG))


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True])
def test_scatter_parallel_gemms(swap_ab):
    from cudnn.gemm.frost import compiler as C

    g, x, w, offsets, indices, ks, first = _build_scatter_graph()
    first.set_output(False).set_data_type(cudnn.data_type.FLOAT)
    w2 = g.tensor(name="weights2", dim=[3, 96, 96], stride=[96 * 96, 1, 96])
    second = g.moe_grouped_matmul(x, w2, offsets, token_index=indices, token_ks=ks, top_k=3, mode=cudnn.moe_grouped_matmul_mode.SCATTER)
    y = g.mul(a=g.swish(input=first), b=second)
    y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    cfg = replace(by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"), swap_ab=swap_ab)
    compiled = C.jit_from_cudnn_graph(g, config=cfg)
    token_values = (torch.arange(387, device="cuda") % 7).float()
    tokens = token_values.to(torch.bfloat16).view(1, 387, 1).expand(1, 387, 96).contiguous()
    expert_values = torch.arange(1, 4, device="cuda", dtype=torch.bfloat16)
    weights = (expert_values / 128).view(3, 1, 1).expand(3, 96, 96).contiguous()
    weights2 = ((expert_values + 1) / 128).view(3, 1, 1).expand(3, 96, 96).contiguous()
    starts = [0, 3, 3, 134, 257, 258, 387]
    fto = torch.tensor(starts, device="cuda", dtype=torch.int32)
    destination = torch.randperm(387, device="cuda", dtype=torch.int32)
    out = torch.empty_like(tokens)
    compiled({x: tokens, w: weights, w2: weights2, offsets: fto, indices: (destination // 3).view(1, 387, 1), ks: (destination % 3).view(1, 387, 1), y: out})
    ref = torch.empty(387, device="cuda")
    for group, (begin, end) in enumerate(zip(starts, starts[1:])):
        a = token_values[begin:end] * (96 * (group % 3 + 1) / 128)
        b = token_values[begin:end] * (96 * (group % 3 + 2) / 128)
        ref[destination[begin:end].long()] = torch.nn.functional.silu(a) * b
    torch.testing.assert_close(out, ref.to(torch.bfloat16).view(1, 387, 1).expand_as(out), atol=0.01, rtol=0.01)


def test_gather_shape_and_binding():
    g, x, weights, offsets, indices, y = _build_gather_graph(swiglu=True)
    assert g.nodes[0].outputs["OUT_0"].get_dim() == [1, 257, 96]
    chain, binding = analyze_with_binding(g)
    assert chain.matmul.M == 257
    assert chain.moe.mode == "gather"
    assert chain.num_a_operands == 1
    assert chain.num_gemms == 2
    assert binding.token_index is indices
    assert indices in binding.bound_tensors()


@pytest.mark.parametrize("fault", ["missing", "dtype", "stride", "different_indices", "different_modes"])
def test_gather_metadata_rejected(fault):
    g, _, _, _, indices, _ = _build_gather_graph(swiglu=True)
    if fault == "missing":
        for node in g.nodes[:2]:
            node.inputs.pop("token_index")
    elif fault == "dtype":
        indices.set_data_type(cudnn.data_type.INT64)
    elif fault == "stride":
        indices.set_stride([514, 2, 1])
    elif fault == "different_indices":
        g.nodes[1].inputs["token_index"] = g.tensor(name="other_index", dim=[1, 257, 1], stride=[257, 1, 1], data_type=cudnn.data_type.INT32)
    else:
        g.nodes[1].params["mode"] = cudnn.moe_grouped_matmul_mode.NONE
    with pytest.raises((ValueError, NotImplementedError), match="token_index|same mode"):
        analyze_with_binding(g)


def test_gather_requires_index_in_frontend_validation():
    g, *_ = _build_gather_graph()
    g.nodes[0].inputs.pop("token_index")
    with pytest.raises(ValueError, match="token_index not set"):
        g.validate()


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True], ids=["token_by_weight", "weight_by_token"])
@pytest.mark.parametrize(
    "geometry,swiglu,m_major,stg",
    [
        ("128x128x128_16x16x32_cluster1x1_warps4x2", False, False, False),
        ("128x128x128_16x16x32_cluster1x1_warps2x4", False, False, False),
        ("128x128x128_16x16x32_cluster1x1_warps8x1", False, True, True),
        ("128x128x128_16x16x32_cluster1x1_warps1x8", False, False, False),
        ("128x128x128_16x16x32_cluster1x1_warps4x2", True, False, False),
        ("128x128x128_16x16x32_cluster1x1_warps2x4", True, True, True),
        ("256x128x128_16x16x32_cluster1x1_warps4x2", False, False, False),
        ("128x128x64_16x16x32_cluster1x1_warps4x2", False, False, False),
        ("128x32x128_16x16x32_cluster1x1_warps4x2", False, False, False),
        ("64x64x128_16x16x32_cluster1x1_warps4x2", False, True, True),
        ("256x96x128_16x16x32_cluster1x1_warps4x2", False, False, False),
        ("128x256x128_16x16x32_cluster1x1_warps4x2", False, False, False),
    ],
)
def test_gather_numerics(geometry, swiglu, m_major, stg, swap_ab):
    from cudnn.gemm.frost import compiler as C

    cfg = replace(by_name("CONFIG_sm120_" + geometry), swap_ab=swap_ab)
    _run_gather_numerics(C, cfg, swiglu=swiglu, m_major=m_major, stg=stg)


def _run_gather_numerics(compiler, cfg, *, swiglu=False, m_major=False, stg=False, activation=False):
    g, x, weights, offsets, indices, y = _build_gather_graph(swiglu=swiglu, m_major=m_major, activation=activation)
    if stg:
        with compiler.force_stg_epi():
            compiled = compiler.jit_from_cudnn_graph(g, config=cfg)
    else:
        compiled = compiler.jit_from_cudnn_graph(g, config=cfg)
    torch.manual_seed(123)
    token = torch.randn(1, 137, 96, device="cuda", dtype=torch.bfloat16)
    wbufs = [torch.randn(3, 96, 96, device="cuda", dtype=torch.bfloat16) * 0.1 for _ in weights]
    starts = [0, 65, 65, 68, 197, 198, 257]
    fto = torch.tensor(starts, device="cuda", dtype=torch.int32).view(7, 1, 1)
    index = torch.randint(137, (1, 257, 1), device="cuda", dtype=torch.int32)
    # Guard every unused output row/column, including the group tail.
    backing = torch.full((96, 264) if m_major else (264, 112), 123, device="cuda", dtype=torch.bfloat16)
    out = (backing[:, :257].T if m_major else backing[:257, :96]).unsqueeze(0)
    vp = {x: token, offsets: fto, indices: index, y: out, **dict(zip(weights, wbufs))}
    workspace = torch.empty(compiled.workspace_bytes, dtype=torch.uint8, device="cuda")

    def check():
        ref = torch.empty(257, 96, device="cuda", dtype=torch.float32)
        for group, begin in enumerate(starts[:-1]):
            end = starts[group + 1]
            a = token[0, index.flatten()[begin:end].long()].float()
            vals = [a @ w[group % 3].float().T for w in wbufs]
            ref[begin:end] = torch.nn.functional.silu(vals[0]) * vals[1] if swiglu else vals[0]
        if activation:
            ref = torch.nn.functional.silu(ref)
        torch.testing.assert_close(out[0].float(), ref.to(torch.bfloat16).float(), atol=0.03, rtol=0.02)
        if m_major:
            assert torch.all(backing[:, 257:] == 123)
        else:
            assert torch.all(backing[257:] == 123)
            assert torch.all(backing[:, 96:] == 123)

    compiled(vp, workspace=workspace)
    check()
    if (swiglu or activation) and not m_major:
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                compiled(vp, workspace=workspace, stream=torch.cuda.current_stream().cuda_stream)
            index.copy_(136 - index)
            starts[:] = [0, 0, 17, 129, 129, 256, 257]
            fto.copy_(torch.tensor(starts, device="cuda", dtype=torch.int32).view(7, 1, 1))
            graph.replay()
            check()
        finally:
            graph.reset()


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True], ids=["token_by_weight", "weight_by_token"])
@pytest.mark.parametrize("case", ["tiny_fp16", "aligned_fp8", "n_major_weight", "frontend"])
def test_gather_layouts_and_frontend(case, swap_ab):
    from cudnn.gemm.frost import compiler as C

    cfg = replace(by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"), swap_ab=swap_ab)
    _run_gather_layouts_and_frontend(C, cfg, case)


def _run_gather_layouts_and_frontend(compiler, cfg, case):
    tiny = case == "tiny_fp16"
    aligned = case == "aligned_fp8"
    source_rows, rows, k, n = (137, 3, 80, 40) if tiny else (17, 384, 96, 80)
    starts = [0, 0, 1, 1, 2, 3, rows] if tiny else [0, 128, 128, 256, 256, 384, rows]
    g, x, weights, offsets, indices, y = _build_gather_graph(
        source_rows=source_rows,
        rows=rows,
        k=k,
        n=n,
        input_dt=cudnn.data_type.INT8 if case == "int8" else cudnn.data_type.BFLOAT16,
        compute_dt=cudnn.data_type.INT32 if case == "int8" else cudnn.data_type.FLOAT,
    )
    if case == "n_major_weight":
        weights[0].set_stride([k * n, n, 1])
    a_dtype, b_dtype = torch.bfloat16, torch.bfloat16
    if case == "int8":
        a_dtype = b_dtype = torch.int8
    if tiny:
        x.set_data_type(cudnn.data_type.HALF)
        weights[0].set_data_type(cudnn.data_type.HALF)
        a_dtype = b_dtype = torch.float16
    elif aligned:
        x.set_data_type(cudnn.data_type.FP8_E4M3)
        weights[0].set_data_type(cudnn.data_type.FP8_E5M2)
        a_dtype, b_dtype = torch.float8_e4m3fn, torch.float8_e5m2
        offsets.set_alignment_value(128)
        y.set_stride([rows * n, 1, rows])
    offsets.set_data_type(cudnn.data_type.INT64)
    if case == "frontend":
        g.validate()
        g.build_operation_graph()
        if cfg.swap_ab or cfg.pipeline == "sm120":
            from cudnn.engines.manifest import MANIFEST
            from cudnn.gemm.frost.knobs import GemmKnobs

            engine_id = next(row.engine_id for row in MANIFEST if row.name == "frost_gemm")
            public = GemmKnobs.from_config(cfg).to_public()
            assert public[cudnn.knob_type.SWAP_AB] == int(cfg.swap_ab)
            g.create_execution_plan(engine_id, public)
        else:
            g.create_execution_plans([cudnn.heur_mode.A])
        plan = next(i for i in range(g.get_execution_plan_count()) if g.get_plan_name_at_index(i).startswith("frost_gemm"))
        g.select_plan(plan)
        g.check_support()
        g.build_plans()
        workspace_bytes = g.get_workspace_size()
    else:
        compiled = compiler.jit_from_cudnn_graph(g, config=cfg)
        workspace_bytes = compiled.workspace_bytes
        if aligned:
            assert compiled.store_modes == (("stg",) if cfg.pipeline == "sm120" else ("tma",))
    torch.manual_seed(321)
    token = (torch.randn(1, source_rows, k, device="cuda") * 0.3).to(a_dtype)
    weight = (torch.randn(3, n, k, device="cuda") * 0.3).to(b_dtype)
    if case == "int8":
        token = torch.randint(-3, 4, (1, source_rows, k), device="cuda", dtype=a_dtype)
        weight = torch.randint(-3, 4, (3, n, k), device="cuda", dtype=b_dtype)
    if case == "n_major_weight":
        weight = weight.transpose(1, 2).contiguous().transpose(1, 2)
    fto = torch.tensor(starts, device="cuda", dtype=torch.int64).view(7, 1, 1)
    index = torch.randint(source_rows, (1, rows, 1), device="cuda", dtype=torch.int32)
    out = torch.empty((n, rows) if aligned else (rows, n), device="cuda", dtype=torch.bfloat16)
    out = (out.T if aligned else out).unsqueeze(0)
    vp = {x: token, weights[0]: weight.transpose(1, 2), offsets: fto, indices: index, y: out}
    workspace = torch.empty(workspace_bytes, device="cuda", dtype=torch.uint8)
    if case == "frontend":
        # Bare addresses inherit graph shapes, including source vs routed M.
        g.execute({t: buf.data_ptr() for t, buf in vp.items()}, workspace)
    else:
        compiled(vp, workspace=workspace)
    ref = torch.empty(rows, n, device="cuda", dtype=torch.float32)
    for group, begin in enumerate(starts[:-1]):
        end = starts[group + 1]
        ref[begin:end] = token[0, index.flatten()[begin:end].long()].float() @ weight[group % 3].float().T
    torch.testing.assert_close(out[0].float(), ref.bfloat16().float(), atol=0.01, rtol=0.01)
    if tiny:
        # Metadata failures must precede the reset/launch and any output write.
        out.fill_(123)
        for bad in (index.to(torch.int64), torch.empty(1, rows * 2, 1, dtype=torch.int32, device="cuda")[:, ::2]):
            with pytest.raises(ValueError, match="token_index"):
                compiled({**vp, indices: bad}, workspace=workspace)
        with pytest.raises(KeyError, match="token_index"):
            compiled({t: v for t, v in vp.items() if t is not indices}, workspace=workspace)
        assert torch.all(out == 123)


# --------------------------------------------------------------------------- #
# End-to-end correctness (GPU)
# --------------------------------------------------------------------------- #


def _offsets(group_sizes, S, dtype=torch.int32):
    starts, cur = [], 0
    for gs in group_sizes:
        starts.append(cur)
        cur += gs
    assert cur == S
    starts.append(cur)
    return torch.tensor(starts, dtype=dtype, device="cuda")


def _ref_f32(token, weight, offsets, S, N, E):
    out = torch.zeros((S, N), dtype=torch.float32, device="cuda")
    starts = offsets.tolist()
    for g in range(len(starts) - 1):
        b = starts[g]
        e = starts[g + 1]
        if b == e:
            continue
        out[b:e] = token[0, b:e].float() @ weight[g % E].float().T
    return out


def _block_quant_q_atol(scale_dtype) -> float:
    # Non-pow2 E4M3 scales use the kernel's approximate reciprocal → up to one
    # smallest E4M3 output step off the torch reference.
    return 1.0 / 512.0 if scale_dtype is torch.float8_e4m3fn else 0.0


def _group_reduction_ref(
    x: torch.Tensor,
    offsets: torch.Tensor,
    mode,
    out_dims: tuple[int, int, int],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    group_count, _, n = out_dims
    starts = offsets.tolist()
    out = torch.empty(out_dims, dtype=out_dtype, device=x.device)
    if mode in (cudnn.reduction_mode.ADD, cudnn.reduction_mode.AMAX, cudnn.reduction_mode.NORM1, cudnn.reduction_mode.AVG):
        out.fill_(0)
    elif mode in (cudnn.reduction_mode.MUL, cudnn.reduction_mode.MUL_NO_ZEROS):
        out.fill_(1)
    elif mode == cudnn.reduction_mode.MAX:
        out.fill_(-(2**31) if out_dtype == torch.int32 else -float("inf"))
    elif mode == cudnn.reduction_mode.MIN:
        out.fill_(2**31 - 1 if out_dtype == torch.int32 else float("inf"))
    else:
        raise AssertionError(f"unsupported reduction mode {mode!r}")
    for g in range(group_count):
        begin = starts[g]
        end = starts[g + 1]
        if begin == end:
            continue
        src = x[begin:end].to(out_dtype) if out_dtype == torch.int32 else x[begin:end]
        if out_dims[1:] == (1, 1):
            reduce_dims = (0, 1)
            out[g, 0, 0] = _reduction_ref(src, mode, reduce_dims)
        elif out_dims[1:] == (1, x.shape[1]):
            out[g, 0, :n] = _reduction_ref(src, mode, (0,)).view(-1)
        elif out_dims[1:] == (x.shape[0], 1):
            out[g, begin:end, 0] = _reduction_ref(src, mode, (1,)).view(-1)
        else:
            raise AssertionError(f"unsupported group reduction dims {out_dims}")
    return out


def _group_sizes_from_offsets(offsets: list[int]) -> list[int]:
    return [end - begin for begin, end in zip(offsets, offsets[1:])]


def _mk_nonpacked_data(S, N, K, E, mode):
    torch.manual_seed(0)
    if mode == "zero_stride":
        token_base = torch.randn(K, dtype=torch.bfloat16, device="cuda")
        weight_base = torch.randn(K, dtype=torch.bfloat16, device="cuda")
        token = torch.as_strided(token_base, (1, S, K), (0, 0, 1))
        weight = torch.as_strided(weight_base, (E, N, K), (0, 0, 1))
    else:
        pad = 16
        token_storage = torch.randn(1, S, K + pad, dtype=torch.bfloat16, device="cuda")
        weight_storage = torch.randn(E, N, K + pad, dtype=torch.bfloat16, device="cuda")
        token = token_storage[:, :, :K]
        weight = weight_storage[:, :, :K]
    output_storage = torch.zeros(1, S, N + 16, dtype=torch.bfloat16, device="cuda")
    return token, weight, output_storage[:, :, :N]


# first_token_offset accepts INT32 or INT64; the kernel bakes the dtype at JIT
# and casts reads to Int32 internally.
_OFFSET_DTYPES = [
    (cudnn.data_type.INT32, torch.int32),
    (cudnn.data_type.INT64, torch.int64),
]


@requires_sm120
@pytest.mark.parametrize("cfg_name,cta_group", _GEOMETRIES_STEP8)
@pytest.mark.parametrize("offset_cudnn_dt,offset_torch_dt", _OFFSET_DTYPES)
@pytest.mark.parametrize(
    "group_sizes",
    [
        [64, 0, 200, 128, 100, 12, 196, 68],  # uneven + one empty group
        [96, 96, 96, 96, 96, 96, 96, 96],  # balanced
        [768, 0, 0, 0, 0, 0, 0, 0],  # all tokens in group 0
    ],
)
def test_moe_grouped_matmul_fwd_e2e(group_sizes, offset_cudnn_dt, offset_torch_dt, cfg_name, cta_group) -> None:
    E, N, K = 8, 256, 128
    S = sum(group_sizes)
    cfg = by_name(cfg_name)
    compiled = _plan(
        _build_graph(E, S, N, K, offset_dt=offset_cudnn_dt),
        config=cfg,
        cta_group=cta_group,
    )

    torch.manual_seed(0)
    token = torch.randn(1, S, K, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    output = torch.zeros(1, S, N, dtype=torch.bfloat16, device="cuda")
    offsets = _offsets(group_sizes, S, dtype=offset_torch_dt)

    compiled(_vp_moe(compiled, token, weight, offsets, output))
    torch.cuda.synchronize()
    torch.testing.assert_close(output[0], _ref_f32(token, weight, offsets, S, N, E).to(torch.bfloat16), atol=1e-1, rtol=1e-2)


@requires_sm120
@pytest.mark.no_workspace_shim
def test_moe_grouped_matmul_fwd_direct_call_requires_workspace() -> None:
    """Rule 8: the plan owns no workspace. A direct call without one is a contract
    error naming the byte count; the test harness supplies ``workspace_bytes``."""
    E, N, K = 8, 256, 128
    group_sizes = [96] * E
    S = sum(group_sizes)
    compiled = _plan(_build_graph(E, S, N, K), config=by_name(_CFG))
    assert compiled.workspace_bytes > 0
    torch.manual_seed(0)
    token = torch.randn(1, S, K, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    output = torch.zeros(1, S, N, dtype=torch.bfloat16, device="cuda")
    vp = _vp_moe(compiled, token, weight, _offsets(group_sizes, S, dtype=torch.int32), output)
    with pytest.raises(ValueError, match="workspace"):
        compiled._compiled(vp)
    compiled(vp)
    torch.cuda.synchronize()
    torch.testing.assert_close(output[0], _ref_f32(token, weight, _offsets(group_sizes, S), S, N, E).to(torch.bfloat16), atol=1e-1, rtol=1e-2)


def test_analyzer_detects_n_major_weight() -> None:
    chain = analyze(_build_graph(8, 768, 256, 128, weight_major="n"))
    assert chain.matmul.a_major == "k" and chain.matmul.b_major == "n"


@requires_sm120
@pytest.mark.parametrize("cfg_name,cta_group", _GEOMETRIES)
@pytest.mark.parametrize(
    "group_sizes",
    [
        [64, 0, 200, 128, 100, 12, 196, 68],  # uneven + one empty group
        [96, 96, 96, 96, 96, 96, 96, 96],  # balanced
        [768, 0, 0, 0, 0, 0, 0, 0],  # all tokens in group 0
    ],
)
def test_moe_grouped_matmul_fwd_e2e_weight_n_major(group_sizes, cfg_name, cta_group) -> None:
    E, N, K = 8, 256, 128
    S = sum(group_sizes)
    cfg = by_name(cfg_name)

    torch.manual_seed(0)
    token = torch.randn(1, S, K, dtype=torch.bfloat16, device="cuda")
    weight_k = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    weight_n = weight_k.transpose(1, 2).contiguous().transpose(1, 2)
    offsets = _offsets(group_sizes, S, dtype=torch.int32)
    ref = _ref_f32(token, weight_k, offsets, S, N, E).to(torch.bfloat16)

    out_n = torch.zeros(1, S, N, dtype=torch.bfloat16, device="cuda")
    compiled_n = _plan(_build_graph(E, S, N, K, weight_major="n"), config=cfg, cta_group=cta_group)
    assert compiled_n.chain.matmul.b_major == "n"
    compiled_n(_vp_moe(compiled_n, token, weight_n, offsets, out_n))

    out_k = torch.zeros(1, S, N, dtype=torch.bfloat16, device="cuda")
    compiled_k = _plan(_build_graph(E, S, N, K), config=cfg, cta_group=cta_group)
    compiled_k(_vp_moe(compiled_k, token, weight_k, offsets, out_k))
    torch.cuda.synchronize()

    torch.testing.assert_close(out_n[0], ref, atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(out_n, out_k, atol=0, rtol=0)


@requires_sm120
def test_moe_grouped_matmul_fwd_n_major_weight_more_groups_than_experts() -> None:
    E, N, K = 4, 256, 128
    group_sizes = [64, 0, 128, 64, 96, 32, 160, 96, 32, 32, 32, 32]
    S = sum(group_sizes)
    cfg = by_name(_CFG)
    compiled = _plan(
        _build_graph(E, S, N, K, num_groups=len(group_sizes), weight_major="n"),
        config=cfg,
        cta_group=1,
    )

    torch.manual_seed(0)
    token = torch.randn(1, S, K, dtype=torch.bfloat16, device="cuda")
    weight_k = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    weight_n = weight_k.transpose(1, 2).contiguous().transpose(1, 2)
    output = torch.zeros(1, S, N, dtype=torch.bfloat16, device="cuda")
    offsets = _offsets(group_sizes, S, dtype=torch.int32)

    compiled(_vp_moe(compiled, token, weight_n, offsets, output))
    torch.cuda.synchronize()
    torch.testing.assert_close(output[0], _ref_f32(token, weight_k, offsets, S, N, E).to(torch.bfloat16), atol=1e-1, rtol=1e-2)


def test_select_config_lifts_the_n_tile_for_n_major_b() -> None:
    """N-major B loads whole swizzle groups of columns, so the auto heuristic
    must not hand back a per-CTA N tile smaller than one group."""
    from cudnn.gemm.frost.tile_config import select_config

    cfg_k = select_config(64, 32, 1)
    assert (cfg_k.cta_tile_n, cfg_k.cta_group) == (32, 1)

    cfg_n = select_config(64, 32, 1, b_n_major=True)
    assert (cfg_n.cta_tile_n, cfg_n.cta_group) == (64, 1)

    cfg_2 = select_config(256, 32, 1, b_n_major=True)
    assert (cfg_2.cta_tile_n, cfg_2.cta_group) == (128, 2)


@requires_sm120
def test_moe_grouped_matmul_fwd_auto_config_n_major_small_n() -> None:
    """The auto path on the N tile the K-major heuristic would have picked too
    small (32) — it must lift to a legal geometry, not fail to build."""
    from cudnn.gemm.frost.graph_analyzer import build_gemm_plan

    E, N, K = 4, 32, 128
    group_sizes = [64, 64, 64, 64]
    S = sum(group_sizes)
    compiled = build_gemm_plan(_build_graph(E, S, N, K, weight_major="n"))
    assert compiled.config.cta_tile_n >= 64

    torch.manual_seed(0)
    token = torch.randn(1, S, K, dtype=torch.bfloat16, device="cuda")
    weight_k = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    weight_n = weight_k.transpose(1, 2).contiguous().transpose(1, 2)
    output = torch.zeros(1, S, N, dtype=torch.bfloat16, device="cuda")
    offsets = _offsets(group_sizes, S, dtype=torch.int32)

    compiled(_vp_moe(compiled, token, weight_n, offsets, output))
    torch.cuda.synchronize()
    torch.testing.assert_close(output[0], _ref_f32(token, weight_k, offsets, S, N, E).to(torch.bfloat16), atol=1e-1, rtol=1e-2)


@pytest.mark.L0
@pytest.mark.skip(
    reason="N/A on sm120: checks the sm100 MoE TMA-store descriptor (UTMASTG.2D); sm120 has no TMA-store epilogue, "
    "so _store_modes never puts an sm120 output on a TMA-C surface (test_moe_m_major_output covers the STG store)"
)
@pytest.mark.parametrize("major", ("n", "m"))
def test_moe_tma_store_uses_rank2_output_descriptor(major: str) -> None:
    """MoE's output is one flat (S, N) surface, so its TMA store must not carry
    the fixed-one batch dimension paid by ordinary batched GEMM. Besides being
    redundant in the descriptor, that extra coordinate selects UTMASTG.3D
    instead of UTMASTG.2D in SASS."""
    from cudnn.gemm.frost.compiler import _epi_n, _host_tma_c_descs, _tma_c_plumbing, _tma_store_sequence

    g = _build_graph(E=4, S=512, N=256, K=128)
    if major == "m":
        g.tensors["moe::OUT_0"].set_stride([512 * 256, 1, 512])
        g.tensors["first_token_offset"].set_alignment_value(8)
    chain = analyze(g)
    cfg = by_name(_CFG)
    epi_n = _epi_n(cfg, chain.output_dtype)
    host = _host_tma_c_descs(chain, cfg, frozenset({0}), epi_n)
    sequence = _tma_store_sequence(chain, cfg, frozenset({0}), epi_n)

    assert ("global_dims=[m, n]" if major == "m" else "global_dims=[n, m]") in host
    assert (f"box_dims=[64, {epi_n}]" if major == "m" else f"box_dims=[{epi_n}, epi_tile_mn[0]]") in host
    assert "out_stride_l_0" not in host
    assert ("(coord_m + 0, col)" if major == "m" else "(col, coord_m)") in sequence
    assert "tile_l" not in sequence
    # N/A on SM120: `tma_c_m_major` is the sm100 MoE TMA-C list entry (sm100
    # _tma_c_plumbing, consumed by the sm100 MoE templates). No sm120 template has
    # a TMA-C marker and _store_modes never puts an sm120 output on the TMA-C
    # surface, so the helpers above are sm100-inherited and unused by sm120. The
    # SM120 fact is that this MoE output, either major, is stored with STG.
    from cudnn.gemm.frost.sm120.compiler import _store_modes, _use_tma_store_epi

    assert _store_modes(chain, cfg) == ("stg",)
    assert not _use_tma_store_epi(chain, cfg)


def test_moe_dim0_descriptor_patch_ir() -> None:
    """Keep the dim-0 update below the public NVVM ordinal verifier, with
    memory side effects so the following descriptor copy cannot pass it."""
    import cutlass.cute as cute
    from cudnn.gemm.frost.tile_helpers import replace_tensormap_global_dim_0

    @cute.kernel
    def patch_dim(new_dim: cutlass.Int32):
        desc = cutlass.Array(cutlass.Int64, 16, space=cutlass.AddressSpace.smem, alignment=128)
        replace_tensormap_global_dim_0(desc, new_dim)

    @cute.jit
    def host(new_dim: cutlass.Int32):
        patch_dim(new_dim).launch(grid=(1, 1, 1), block=(32, 1, 1))

    modules = []

    def capture_ir(dsl, module, function_name):
        modules.append(str(module))

    # Only trace IR: this also runs with public DSL builds that cannot target
    # the local GPU's architecture, and never launches a synthetic descriptor.
    cute.compile.to_precompiled_mlir(host, 104, options="--gpu-arch=sm_100a", trace_finalize_hooks=capture_ir)
    src = "\n".join(modules)
    assert "llvm.inline_asm has_side_effects" in src
    assert "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [$0], 0, $1;" in src
    assert '"r,r,~{memory}"' in src
    assert "nvvm.tensormap.replace" not in src


@requires_sm120
@pytest.mark.parametrize("cfg_name", ("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2", "CONFIG_sm120_128x256x128_16x16x32_cluster1x1_warps4x2"))
@pytest.mark.parametrize(
    "offset_multiple,bounds,S,store_mode,global_descriptors",
    [
        (1, [0, 100, 300, 512], 512, "stg", False),
        (4, [0, 104, 304, 512], 512, "stg", False),
        (8, [0, 104, 304, 512], 512, "tma", False),
        (256, [0, 256, 256, 512], 512, "tma", True),
        (2, [0, 104, 304, 510], 510, "stg", False),
        (2, [0, 256, 256, 510], 510, "stg", False),
        (256, [0, 256, 256, 512], 520, "tma", True),
    ],
)
@pytest.mark.parametrize("tap", (False, True))
def test_moe_m_major_output(cfg_name: str, offset_multiple: int, bounds: list[int], S: int, store_mode: str, global_descriptors: bool, tap: bool) -> None:
    """TMA needs a promise of 16-byte group boundaries, independently for each
    output layout. Partial groups must clip on M even beside an N-major tap."""
    from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
    from cudnn.gemm.frost.sm120.compiler import _MOE_DESC_SLOT_BYTES, _MOE_SCHED_COUNTER_SLOTS, _store_modes
    from cudnn.gemm.frost.graph_analyzer import analyze
    from cudnn.gemm.frost.tile_config import by_name

    N, K, E = 256, 128, 3
    ldm = _ceil_div(S, 8) * 8
    BF = cudnn.data_type.BFLOAT16
    cfg = by_name(cfg_name)

    def build():
        g = cudnn.pygraph(io_data_type=BF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=BF)
        w = g.tensor(name="weight", dim=[E, K, N], stride=[K * N, 1, K], data_type=BF)
        fto = g.tensor(name="first_token_offset", dim=[E + 1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
        fto.set_alignment_value(offset_multiple)
        out = g.moe_grouped_matmul(tok, w, fto, mode=cudnn.moe_grouped_matmul_mode.NONE, compute_data_type=cudnn.data_type.FLOAT, name="moe")
        if tap:
            tap_tensor = g.identity(input=out, name="tap")
            out = g.relu(input=tap_tensor, name="relu")
            tap_tensor.set_data_type(cudnn.data_type.FLOAT).set_output(True).set_stride([S * N, N, 1])
        out.set_data_type(BF).set_output(True)
        out.set_stride([ldm * N, 1, ldm])
        return g, tok, w, fto, out

    g, _, _, _, _ = build()
    chain = analyze(g)
    assert tuple(out.major for out in chain.output_specs) == (("n", "m") if tap else ("m",))
    assert chain.moe.offset_multiple == offset_multiple
    # sm120: no TMA-store epilogue (cfg.pipeline is not in _TMA_STORE_EPI_PIPELINES), so
    # every output -- the N-major tap included -- stores STG and the ragged group tail
    # is clipped by `row < group_end`, whatever the offsets promise. `store_mode` and
    # `global_descriptors` are the sm100 TMA-store / descriptor-rewrite choices.
    assert _store_modes(chain, cfg) == (("stg", "stg") if tap else ("stg",))

    torch.manual_seed(0)
    tk = torch.randn(1, S, K, device="cuda", dtype=torch.bfloat16)
    wt = torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16)
    ft = torch.tensor(bounds, device="cuda", dtype=torch.int32)
    slack = 4096
    raw = torch.full((2 * ldm * N + slack,), 0xAB, device="cuda", dtype=torch.uint8)
    storage = raw[: 2 * ldm * N].view(1, N, 2 * ldm)
    out = storage.view(torch.bfloat16).transpose(1, 2)[:, :S, :]
    out.fill_(-123)
    tail = raw[2 * ldm * N :].clone()
    compiled = jit_from_cudnn_graph(g, config=cfg)
    # sm120 addresses A by coordinate on ONE global descriptor for every promise: the
    # kernel carries no moe_aligned_offsets switch, patches no tensormap (its workspace
    # is the scheduler counter alone) and masks the ragged tail at the STG store.
    src = pathlib.Path(compiled.generated_path).read_text()
    assert "moe_desc_slots = 0\n" in src and "moe_aligned_offsets" not in src
    assert "if row < group_end:" in src
    assert compiled.store_modes == _store_modes(chain, cfg)
    assert compiled.workspace_bytes == _MOE_SCHED_COUNTER_SLOTS * _MOE_DESC_SLOT_BYTES
    tap_out = torch.full((1, S, N), -123, device="cuda", dtype=torch.float32) if tap else None
    compiled(_vp_moe(compiled, tk, wt, ft, [tap_out, out] if tap else out))
    torch.cuda.synchronize()

    ref = torch.full((1, S, N), -123, device="cuda", dtype=torch.float32)
    b = bounds
    for gi in range(E):
        ref[0, b[gi] : b[gi + 1]] = tk[0, b[gi] : b[gi + 1]].float() @ wt[gi].float().T
    if tap:
        torch.testing.assert_close(tap_out, ref, atol=1e-4, rtol=1e-4)
        ref[:, : bounds[-1]].relu_()
    assert torch.equal(raw[2 * ldm * N :], tail), "the store ran past the output"
    assert (storage[:, :, 2 * S :] == 0xAB).all(), "the store overwrote column padding"
    assert (out.float() - ref).abs().max().item() < 0.5


@requires_sm120
@pytest.mark.parametrize(
    "offset_multiple,S,store_modes",
    # The parent's tuples are sm100's per-dtype TMA eligibility (the promise must
    # cover 16 bytes of the output dtype). SM120 has no TMA-store epilogue, so every
    # output is STG whatever the promise; the bit-exact numerics below still run.
    [
        (1, 768, ("stg", "stg", "stg")),
        (4, 768, ("stg", "stg", "stg")),
        (8, 768, ("stg", "stg", "stg")),
        (12, 768, ("stg", "stg", "stg")),
        (16, 768, ("stg", "stg", "stg")),
        (24, 768, ("stg", "stg", "stg")),
        (4, 772, ("stg", "stg", "stg")),
        (8, 776, ("stg", "stg", "stg")),
        (1, 767, ("stg", "stg", "stg")),
    ],
)
def test_moe_m_major_alignment_is_per_output_dtype(offset_multiple: int, S: int, store_modes: tuple[str, ...]) -> None:
    from cudnn.gemm.frost.compiler import _store_modes, jit_from_cudnn_graph

    ldm = (S + 15) // 16 * 16
    g = _build_graph(E=4, S=S, N=256, K=128)
    fto = next(t for t in g.tensors.values() if t.get_name() == "first_token_offset")
    fto.set_alignment_value(offset_multiple)
    # One promise can suffice for fp32/bf16 while fp8 still needs STG.
    mm = g.tensors["moe::OUT_0"].set_data_type(cudnn.data_type.FLOAT)
    c = g.identity(input=mm, name="float")
    h = g.relu(input=c, name="half")
    z = g.relu(input=h, name="byte")
    for t, dt in ((c, cudnn.data_type.FLOAT), (h, cudnn.data_type.BFLOAT16), (z, cudnn.data_type.FP8_E4M3)):
        t.set_data_type(dt).set_output(True).set_stride([ldm * 256, 1, ldm])
    chain = analyze(g)
    assert chain.moe.offset_multiple == offset_multiple
    assert _store_modes(chain, by_name(_CFG)) == store_modes
    torch.manual_seed(0)
    token = torch.randint(-2, 3, (1, S, 128), device="cuda").to(torch.bfloat16)
    weight = torch.randint(-2, 3, (4, 256, 128), device="cuda").to(torch.bfloat16)
    offsets = torch.tensor([0, 96, 96, 288, S], dtype=torch.int32, device="cuda")
    outputs = [torch.empty(1, 256, ldm, dtype=dt, device="cuda").transpose(1, 2)[:, :S, :] for dt in (torch.float32, torch.bfloat16, torch.float8_e4m3fn)]
    compiled = jit_from_cudnn_graph(g, config=by_name(_CFG))
    compiled(_vp_moe(compiled, token, weight, offsets, outputs))
    ref = _ref_f32(token, weight, offsets, S, 256, 4).unsqueeze(0)
    for actual, expected in zip(outputs, (ref, ref.relu(), ref.relu())):
        torch.testing.assert_close(actual.float(), expected.to(actual.dtype).float(), atol=0, rtol=0)


def test_moe_grouped_matmul_fwd_rejects_m_major_token() -> None:
    from cudnn.gemm.frost.compiler import jit_from_cudnn_graph

    E, S, N, K = 4, 256, 256, 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, 1, S], data_type=cudnn.data_type.BFLOAT16)
    w = g.tensor(name="weight", dim=[E, K, N], stride=[K * N, 1, K], data_type=cudnn.data_type.BFLOAT16)
    fto = g.tensor(name="first_token_offset", dim=[E + 1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    out = g.moe_grouped_matmul(tok, w, fto, mode=cudnn.moe_grouped_matmul_mode.NONE)
    out.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

    with pytest.raises(NotImplementedError, match="K-major token"):
        jit_from_cudnn_graph(g, config=by_name(_CFG))


@requires_sm120
@pytest.mark.parametrize("cfg_name,cta_group", _GEOMETRIES)
@pytest.mark.parametrize(
    "case_name,out_dt,out_torch_dt,scale_dt,scale_torch_dt,scale_reorder,group_sizes,N",
    _QUANT_CASES,
    ids=[case[0] for case in _QUANT_CASES],
)
def test_moe_grouped_matmul_fwd_block_quant_epilogue(
    cfg_name,
    cta_group,
    case_name,
    out_dt,
    out_torch_dt,
    scale_dt,
    scale_torch_dt,
    scale_reorder,
    group_sizes,
    N,
) -> None:
    E, K = 4, 128
    N = int(N)
    S = sum(group_sizes)
    scale_shape = _quant_scale_shape(S, N, scale_reorder)
    cfg = by_name(cfg_name)
    compiled = _plan(
        _build_graph(
            E,
            S,
            N,
            K,
            quant=True,
            quant_out_dt=out_dt,
            quant_scale_dt=scale_dt,
            quant_scale_reorder=scale_reorder,
            quant_scale_dim=scale_shape if scale_reorder else None,
        ),
        config=cfg,
        cta_group=cta_group,
    )

    torch.manual_seed(0)
    token = torch.randn(1, S, K, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    q = torch.empty(1, S, N, dtype=out_torch_dt, device="cuda")
    if scale_reorder:
        q_scale = torch.zeros(*scale_shape, dtype=scale_torch_dt, device="cuda")
    else:
        q_scale = torch.empty(*scale_shape, dtype=scale_torch_dt, device="cuda")
    offsets = _offsets(group_sizes, S)

    compiled(_vp_moe(compiled, token, weight, offsets, [q, q_scale]))
    torch.cuda.synchronize()

    # mma.sync sums K in a different order than torch's FFMA SGEMM, and the last
    # fp32 bit decides an fp8 round-to-nearest tie: at seed 0 element (144, 214) is
    # 4.749999 on the tensor cores and 4.75 in the SGEMM, e4m3 code 72 vs 80. So check
    # the GEMM against the SGEMM at fp32 tolerance, and the quant epilogue bit-exactly
    # against a quant of the kernel's OWN accumulators (same config, quant removed).
    acc = torch.empty(1, S, N, dtype=torch.float32, device="cuda")
    plain = _plan(_build_graph(E, S, N, K, output_dt=cudnn.data_type.FLOAT), config=cfg, cta_group=cta_group)
    plain(_vp_moe(plain, token, weight, offsets, acc))
    torch.cuda.synchronize()
    torch.testing.assert_close(acc[0], _ref_f32(token, weight, offsets, S, N, E), atol=1e-4, rtol=1e-4)
    q_ref, scale_ref = _block_quant_ref(acc[0], 32, out_torch_dt, scale_torch_dt)
    if scale_reorder:
        scale_ref = _to_blocked(scale_ref[0]).view_as(q_scale)
    torch.testing.assert_close(q_scale.float(), scale_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(
        q.float(),
        q_ref.float(),
        atol=_block_quant_q_atol(scale_torch_dt),
        rtol=0,
    )


def _run_moe_reduction(
    cfg_name,
    cta_group,
    mode,
    red_dims,
    *,
    E=4,
    N=128,
    K=128,
    red_stride=None,
    red_dt=cudnn.data_type.FLOAT,
    red_torch_dt=torch.float32,
    red_compute_dt=None,
    integer_inputs=False,
    group_sizes=None,
    group_reduction=False,
) -> None:
    if group_sizes is None:
        group_sizes = [64, 0, 120, 72]
    S = sum(group_sizes)
    cfg = by_name(cfg_name)
    compiled = _plan(
        _build_graph(
            E,
            S,
            N,
            K,
            reduction_mode=mode,
            reduction_dims=tuple(red_dims),
            reduction_stride=red_stride,
            reduction_dt=red_dt,
            reduction_compute_dt=red_compute_dt,
            reduction_group_offset=group_reduction,
            num_groups=len(group_sizes),
        ),
        config=cfg,
        cta_group=cta_group,
    )

    torch.manual_seed(0)
    if integer_inputs:
        token = torch.randint(-2, 3, (1, S, K), device="cuda").to(torch.bfloat16)
        weight = torch.randint(-2, 3, (E, N, K), device="cuda").to(torch.bfloat16)
    else:
        token = torch.randn(1, S, K, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    output = torch.empty(1, S, N, dtype=torch.bfloat16, device="cuda")
    if red_stride is None:
        red = torch.empty(*red_dims, dtype=red_torch_dt, device="cuda")
    else:
        red = torch.empty_strided(red_dims, red_stride, dtype=red_torch_dt, device="cuda")
    offsets = _offsets(group_sizes, S)

    compiled(_vp_moe(compiled, token, weight, offsets, [output, red]))
    torch.cuda.synchronize()

    ref = _ref_f32(token, weight, offsets, S, N, E)
    torch.testing.assert_close(output[0], ref.to(torch.bfloat16), atol=1e-1, rtol=1e-2)
    red_src = ref.to(red_torch_dt) if red_torch_dt == torch.int32 else ref
    if group_reduction:
        red_ref = _group_reduction_ref(red_src, offsets, mode, tuple(red_dims), red_torch_dt)
    else:
        ref_dims = _reduction_dims(tuple(red_dims), (1, S, N))
        red_ref = _reduction_ref(red_src.view(1, S, N), mode, ref_dims).to(red_torch_dt)
    torch.testing.assert_close(
        red,
        red_ref,
        atol=1e-4 if mode == cudnn.reduction_mode.AVG else 1e-1 if red_torch_dt == torch.float32 else 0,
        rtol=1e-5 if mode == cudnn.reduction_mode.AVG else 1e-2 if red_torch_dt == torch.float32 else 0,
    )


@requires_sm120
@pytest.mark.parametrize("cfg_name,cta_group", _GEOMETRIES)
@pytest.mark.parametrize(
    "mode",
    [
        cudnn.reduction_mode.ADD,
        cudnn.reduction_mode.AMAX,
        cudnn.reduction_mode.MAX,
        cudnn.reduction_mode.MIN,
    ],
)
def test_moe_grouped_matmul_fwd_reduction_scalar_fp32(mode, cfg_name, cta_group) -> None:
    _run_moe_reduction(cfg_name, cta_group, mode, [1, 1, 1])


@requires_sm120
@pytest.mark.parametrize(
    "mode,red_dims,red_stride",
    [
        (cudnn.reduction_mode.ADD, [1, 256, 1], [0, 2, 1]),
        (cudnn.reduction_mode.AMAX, [1, 1, 128], [0, 0, 2]),
    ],
)
def test_moe_grouped_matmul_fwd_reduction_partial_strided_fp32(mode, red_dims, red_stride) -> None:
    _run_moe_reduction(
        _CFG,
        1,
        mode,
        red_dims,
        red_stride=red_stride,
        integer_inputs=True,
    )


@requires_sm120
@pytest.mark.parametrize(
    "mode",
    [
        cudnn.reduction_mode.ADD,
        cudnn.reduction_mode.AMAX,
        cudnn.reduction_mode.MAX,
        cudnn.reduction_mode.MIN,
    ],
)
def test_moe_grouped_matmul_fwd_reduction_scalar_int32(mode) -> None:
    _run_moe_reduction(
        _GEOMETRIES[1][0],
        _GEOMETRIES[1][1],
        mode,
        [1, 1, 1],
        red_dt=cudnn.data_type.INT32,
        red_torch_dt=torch.int32,
        red_compute_dt=cudnn.data_type.INT32,
        integer_inputs=True,
    )


@requires_sm120
@pytest.mark.parametrize("cfg_name,cta_group", _GEOMETRIES)
def test_moe_grouped_matmul_fwd_group_reduction_amax_scalar_fp32(cfg_name, cta_group) -> None:
    _run_moe_reduction(
        cfg_name,
        cta_group,
        cudnn.reduction_mode.AMAX,
        [4, 1, 1],
        group_sizes=[64, 0, 120, 72],
        group_reduction=True,
    )


@requires_sm120
def test_moe_grouped_matmul_fwd_group_reduction_full_expert_amax_fp32() -> None:
    group_sizes = _group_sizes_from_offsets(_FULL_EXPERT_REDUCE_OFFSETS)
    _run_moe_reduction(
        _CFG,
        1,
        cudnn.reduction_mode.AMAX,
        [36, 1, 1],
        E=9,
        N=248,
        K=520,
        group_sizes=group_sizes,
        group_reduction=True,
    )


@requires_sm120
@pytest.mark.parametrize(
    "mode",
    [
        cudnn.reduction_mode.ADD,
        cudnn.reduction_mode.MAX,
        cudnn.reduction_mode.MIN,
    ],
)
def test_moe_grouped_matmul_fwd_group_reduction_per_col_fp32(mode) -> None:
    _run_moe_reduction(
        _CFG,
        1,
        mode,
        [4, 1, 128],
        group_sizes=[32, 96, 0, 128],
        group_reduction=True,
        integer_inputs=True,
    )


@requires_sm120
@pytest.mark.parametrize(
    "cfg_name,cta_group,mode",
    [
        ("CONFIG_sm120_128x256x128_16x16x32_cluster1x1_warps4x2", 1, "padded"),
        ("CONFIG_sm120_128x256x128_16x16x32_cluster1x1_warps2x4", 1, "padded"),
        ("CONFIG_sm120_128x256x128_16x16x32_cluster1x1_warps2x4", 1, "zero_stride"),
    ],
)
def test_moe_grouped_matmul_fwd_nonpacked_tensors(cfg_name, cta_group, mode) -> None:
    group_sizes = [64, 0, 200, 128, 100, 12, 196, 68]
    E, N, K = 8, 256, 128
    S = sum(group_sizes)
    cfg = by_name(cfg_name)
    compiled = _plan(_build_graph(E, S, N, K), config=cfg, cta_group=cta_group)

    token, weight, output = _mk_nonpacked_data(S, N, K, E, mode)
    offsets = _offsets(group_sizes, S)
    assert not token.is_contiguous() or not weight.is_contiguous()
    assert not output.is_contiguous()

    compiled(_vp_moe(compiled, token, weight, offsets, output))
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output[0],
        _ref_f32(token, weight, offsets, S, N, E).to(torch.bfloat16),
        atol=1e-1,
        rtol=1e-2,
    )


@requires_sm120
@pytest.mark.parametrize("cfg_name,cta_group", _GEOMETRIES)
def test_moe_grouped_matmul_fwd_bxe_gt_e(cfg_name, cta_group) -> None:
    """num_groups (BxE) > num_experts (E): expert = group % E."""
    S, N, K, E = 2000, 248, 520, 9
    offset_values = _FULL_EXPERT_REDUCE_OFFSETS
    cfg = by_name(cfg_name)
    compiled = _plan(_build_graph(E, S, N, K), config=cfg, cta_group=cta_group)

    torch.manual_seed(0)
    token = torch.randn(1, S, K, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    output = torch.zeros(1, S, N, dtype=torch.bfloat16, device="cuda")
    offsets = torch.tensor(offset_values, dtype=torch.int32, device="cuda")

    compiled(_vp_moe(compiled, token, weight, offsets, output))
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output[0],
        _ref_f32(token, weight, offsets, S, N, E).to(torch.bfloat16),
        atol=2e-1,
        rtol=5e-2,
    )


# INT8 × INT8 → INT32 MoE (integer warp MMA: mma.sync m16n8k32 .s32.s8.s8.s32)
# — the MoE pipeline follows the plain-matmul combos exactly. Unlike the sm100
# tcgen05 int8 kind (SM 100 / SM 110 only), the sm120 family has NO
# MMA_GPU_ARCH_SPECIAL_CASES row for int8: a missing row means "no narrowing
# beyond the family gate" (the same fallback kernel_registry.mma_arch_reject
# uses), i.e. PIPELINE_ARCH_RANGES["sm120"].


@requires_sm120
@pytest.mark.parametrize("cfg_name", ["CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2", "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps2x4"])
def test_moe_int8(cfg_name):
    from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
    from cudnn.gemm.frost.kernel_registry import MMA_GPU_ARCH_SPECIAL_CASES, PIPELINE_ARCH_RANGES
    from gemm_test_utils import _active_sm

    sm = _active_sm()
    ranges = MMA_GPU_ARCH_SPECIAL_CASES.get(("sm120", ("int8", "int8", "int32")), PIPELINE_ARCH_RANGES["sm120"])
    if sm is not None and not any(lo <= sm < hi for lo, hi in ranges):
        spans = " or ".join(f"{lo} <= SM < {hi}" for lo, hi in ranges)
        pytest.skip(f"sm120 int8 MMA unsupported on sm_{sm} (needs {spans})")

    E, S, N, K = 4, 512, 256, 512
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.INT8,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.INT32,
    )
    tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=cudnn.data_type.INT8)
    w = g.tensor(name="weight", dim=[E, K, N], stride=[K * N, 1, K], data_type=cudnn.data_type.INT8)
    fto = g.tensor(name="first_token_offset", dim=[E + 1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    out = g.moe_grouped_matmul(tok, w, fto, mode=cudnn.moe_grouped_matmul_mode.NONE)
    out.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    compiled = jit_from_cudnn_graph(g, config=by_name(cfg_name))
    assert compiled.chain.matmul.accum_dtype == "int32"

    torch.manual_seed(0)
    a = torch.randint(-8, 8, (1, S, K), dtype=torch.int8, device="cuda")
    b = torch.randint(-8, 8, (E, N, K), dtype=torch.int8, device="cuda")
    fto_t = torch.tensor([0, 128, 200, 384, S], dtype=torch.int32, device="cuda")
    outb = torch.zeros(1, S, N, dtype=torch.bfloat16, device="cuda")
    from gemm_test_utils import graph_binding

    bd = graph_binding(compiled)
    compiled({bd.a_operands[0]: a, bd.b_operands[0]: b, bd.first_token_offset: fto_t, bd.outputs[0]: outb})
    torch.cuda.synchronize()

    ref = torch.zeros(1, S, N, dtype=torch.float32, device="cuda")
    bounds = fto_t.tolist()
    for gi in range(E):
        lo, hi = bounds[gi], bounds[gi + 1]
        if hi > lo:
            ref[0, lo:hi] = a[0, lo:hi].float() @ b[gi].float().t()
    # Small-magnitude integer products are exact in bf16's range → bit-exact.
    torch.testing.assert_close(outb, ref.to(torch.bfloat16), atol=0.0, rtol=0.0)


# --- launch ABI -------------------------------------------------------------

# MoE has no recipe (`_check_executable` returns early for it), so its four
# launch sites are hand-written positional calls and nothing else pins their
# order against the rendered `_host` signature.


def test_moe_launch_tail_puts_the_tma_slot_last():
    """The host signature is TAP, AUX, then the trailing TMA-C parameter. Under
    STG every dense output rides a tap slot; an output on the TMA surface binds a
    trailing TMA-only parameter and moves to the END. Passing `*cs, *aux` in both modes
    keeps the ARITY but shifts the mapping by one, which binds the output buffer
    to an aux parameter -- both are `cute.Tensor`, so nothing raises."""
    from cudnn.gemm.frost.compiler import _moe_launch_tail

    NONE, S0, S1, BOTH = frozenset(), frozenset({0}), frozenset({1}), frozenset({0, 1})

    assert _moe_launch_tail(["c0"], (), tma_slots=NONE) == ("c0",)
    assert _moe_launch_tail(["c0"], (), tma_slots=S0) == ("c0",)
    assert _moe_launch_tail(["c0"], ["x"], tma_slots=NONE) == ("c0", "x")
    assert _moe_launch_tail(["c0"], ["x"], tma_slots=S0) == ("x", "c0")
    assert _moe_launch_tail(["c0", "c1"], ["x"], tma_slots=NONE) == ("c0", "c1", "x")
    assert _moe_launch_tail(["c0", "c1"], ["x"], tma_slots=S0) == ("c1", "x", "c0")
    assert _moe_launch_tail([], ["x"], tma_slots=S0) == ("x",)
    # Which slot takes the surface is a CHOICE, so the order follows the SET, not
    # the position: slot 1 on the surface leaves slot 0 as the tap.
    assert _moe_launch_tail(["c0", "c1"], ["x"], tma_slots=S1) == ("c0", "x", "c1")
    assert _moe_launch_tail(["c0", "c1"], ["x"], tma_slots=BOTH) == ("x", "c0", "c1")


@requires_sm120
@pytest.mark.parametrize("force_stg", [False, True], ids=["tmastg", "stg"])
def test_moe_host_signature_matches_the_launch_order(force_stg: bool) -> None:
    """Pin the rendered `_host` parameter list the four launchers feed: every
    dense output is a `c_tap_<i>`, except the slot-0 output under TMA-store,
    which becomes the trailing `c_<i>` the TMA-C marker injects."""
    import re

    from cudnn.gemm.frost.compiler import _moe_launch_tail

    g = _build_graph(E=8, S=2048, N=256, K=256)
    plan = _plan(g, config=by_name("CONFIG_sm120_128x256x128_16x16x32_cluster1x1_warps4x2"), cta_group=1, force_stg_epi=force_stg)
    tma = plan._compiled.use_tma_store
    # The parent's `tma is not force_stg` holds for the sm100 cluster2x1 plan only.
    # SM120 has no TMA-store epilogue: the default plan is STG too, so every output
    # must be a `c_tap_<i>` and no trailing TMA-C `c_<i>` may appear in `_host`.
    assert tma is False

    src = pathlib.Path(plan.generated_path).read_text()
    m = re.search(r"^def _host\(\n(.*?)^\) -> None:", src, re.S | re.M)
    assert m, f"no _host signature in {plan.generated_path}"
    body = m.group(1)
    params = [ln.strip().split(":")[0] for ln in body.splitlines() if ln.strip()]
    taps = [p for p in params if p.startswith("c_tap_")]
    tma_c = [p for p in params if re.fullmatch(r"c_\d+", p)]
    n_out = len(plan.chain.outputs)

    assert len(taps) == n_out - (1 if tma else 0)
    assert len(tma_c) == (1 if tma else 0)
    if tma:
        assert params[-2] == tma_c[0], params
    assert len(_moe_launch_tail(range(n_out), plan.aux_names, tma_slots=plan._compiled.tma_slots)) == len(taps) + len(plan.aux_names) + len(tma_c)


# SM120 GATHER uses the shared numerical checks through its own compiler.

_SM120_GEOMETRIES = [
    "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2",  # the flagship grid
    "CONFIG_sm120_64x256x128_16x16x32_cluster1x1_warps2x4",  # short-M tiles: every group is a ragged tail
    "CONFIG_sm120_256x128x64_16x16x32_cluster1x1_warps8x1",  # tall tile, narrow K row (s64b swizzle)
]
_SM120_CFG = _SM120_GEOMETRIES[0]


@requires_matmul_gpu
@pytest.mark.parametrize("cfg_name", _SM120_GEOMETRIES)
@pytest.mark.parametrize("m_major,activation,swiglu", [(False, False, False), (True, False, False), (False, True, False), (False, False, True)])
def test_gather_numerics_sm120(cfg_name, m_major, activation, swiglu):
    # The ordinary SM120 instruction path also runs on SM10.x. Select its
    # compiler explicitly so this test covers both supported hardware families.
    from cudnn.gemm.frost.sm120 import compiler as C120

    _run_gather_numerics(C120, by_name(cfg_name), m_major=m_major, activation=activation, swiglu=swiglu)


@requires_matmul_gpu
@pytest.mark.parametrize("case", ["tiny_fp16", "aligned_fp8", "n_major_weight", "int8", "frontend"])
def test_gather_layouts_and_frontend_sm120(case):
    from cudnn.gemm.frost.arch_family import active_family
    from cudnn.gemm.frost.sm120 import compiler as C120

    if case == "frontend" and active_family() != "sm120":
        pytest.skip("set CUDNN_FRONTEND_GEMM_ARCH_FAMILY=sm120 before import to replay the public SM120 plan on SM10.x")
    _run_gather_layouts_and_frontend(C120, by_name(_SM120_CFG), case)


@pytest.mark.parametrize("weight_major", ["k", "n"])
@pytest.mark.parametrize("gather", [False, True])
def test_sm120_moe_render_smoke(weight_major: str, gather: bool) -> None:
    """Render the sm120 MoE template end-to-end (tile constants + epilogue
    snippets, no cute.compile) through the sm120 tree by name, so this covers
    every lane. Marker-free, parseable, and carrying the grouped-scheduler
    constants the dense sm120 kernel does not have."""
    import ast
    import re

    from cudnn.gemm.frost.dtypes import DTYPE_BYTES
    from cudnn.gemm.frost.sm120 import compiler as C120
    from cudnn.gemm.frost.sm120.epilogue_codegen import generate

    if gather:
        g, _, weights, _, _, _ = _build_gather_graph(k=128, n=256)
        if weight_major == "n":
            weights[0].set_stride([128 * 256, 256, 1])
    else:
        g = _build_graph(8, 768, 256, 128, weight_major=weight_major)
    chain = analyze(g)
    cfg = by_name(_SM120_CFG)
    snippets = generate(
        chain,
        vec_bytes_epi=C120._epi_chunk_bytes(chain, cfg, False),
        output_elem_bytes=DTYPE_BYTES[chain.output_dtype],
        tma_slots=frozenset(),
        packed_lanes=C120._epi_packed_lanes(cfg),
    )
    src = C120._render_template(chain, snippets, cfg)
    assert "@@" not in "\n".join(line for line in src.splitlines() if not line.lstrip().startswith(("#", '"""')) and "marker" not in line)
    ast.parse(src)
    assert "frost_sm120_moe_grouped_matmul_fwd_" in src
    assert re.search(r"^grid_num_clusters = \d+$", src, re.M) and re.search(r"^offset_cutlass_dtype = cutlass\.Int32$", src, re.M)
    assert "moe_desc_slots = 0" in src  # the workspace is the scheduler counter alone
    assert "tensormap_replace" not in src and "fallback_cluster_shape_mnk" not in src
    assert "if row < group_end:" in src  # the ragged tail is masked at the store, not clipped by a descriptor
    # the host takes the MoE launch ABI the compiler feeds: problem_size, offsets, workspace, A, B, taps
    m = re.search(r"^def _host\(\n(.*?)^\) -> None:", src, re.S | re.M)
    params = [ln.strip().split(":")[0] for ln in m.group(1).splitlines() if ln.strip()]
    assert params == ["problem_size", "first_token_offset", "a_tma_workspace", *(["token_index"] if gather else []), "a_0", "b_0", "c_tap_0", "stream"], params


@requires_sm120
@pytest.mark.parametrize(
    "cfg_name",
    [
        "CONFIG_sm120_128x32x128_16x16x32_cluster1x1_warps4x2",
        "CONFIG_sm120_128x32x128_16x16x32_cluster1x1_warps8x1",
        "CONFIG_sm120_64x64x128_16x16x32_cluster1x1_warps4x2",
        "CONFIG_sm120_64x64x128_16x16x32_cluster1x1_warps2x4",
        "CONFIG_sm120_128x48x128_16x16x32_cluster1x1_warps8x1",
        "CONFIG_sm120_256x96x128_16x16x32_cluster1x1_warps4x2",
    ],
)
@pytest.mark.parametrize("major,aligned,force_stg", [("n", False, False), ("m", False, False), ("m", True, False), ("n", False, True)])
def test_moe_swap_ab_plain(cfg_name, major, aligned, force_stg):
    from dataclasses import replace
    from contextlib import nullcontext
    from cudnn.gemm.frost import fusion_ir
    from cudnn.gemm.frost.sm120 import compiler as C

    torch.manual_seed(17)
    E, S, N, K = 2, (264 if aligned else 263), 272, 136
    starts = [0, 8, 8, 32, 128, S, S] if aligned else [0, 3, 3, 35, 129, S, S]
    g = _build_graph(E, S, N, K, num_groups=len(starts) - 1, weight_major="n", offset_dt=cudnn.data_type.INT64)
    from cudnn.gemm.frost.graph_analyzer import analyze_with_binding

    _, binding = analyze_with_binding(g)
    out = binding.outputs[0]
    fto = binding.first_token_offset
    fto.set_alignment_value(8 if aligned else 1)
    ld = ((S if major == "m" else N) + 7) // 8 * 8
    out.set_dim([1, S, N]).set_stride([N * ld, 1, ld] if major == "m" else [S * ld, ld, 1])
    before = analyze(g)
    cfg = by_name(cfg_name)

    with C.force_stg_epi() if force_stg else nullcontext():
        compiled = C.jit_from_cudnn_graph(g, config=replace(cfg, swap_ab=True))
    assert compiled.chain == fusion_ir.swap_ab(before)
    assert analyze(g) == before
    assert "sm120_moe_grouped_matmul_fwd_swap_ab" in compiled.generated_path.read_text()
    # sm120: _output_store_mode returns "stg" outside _TMA_STORE_EPI_PIPELINES; no TMA-store epilogue.
    assert compiled.store_modes == ("stg",)
    token = torch.randn(1, S, K, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(E, K, N, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    offsets = torch.tensor(starts, device="cuda", dtype=torch.int64)
    size = ld * (N if major == "m" else S)
    raw = torch.full((size + 2048,), -123, device="cuda", dtype=torch.bfloat16)
    output = raw[:size].view(1, N, ld).transpose(1, 2)[:, :S] if major == "m" else raw[:size].view(1, S, ld)[:, :, :N]
    for _ in range(2):
        compiled(_vp_moe(compiled, token, weight, offsets, output))
    torch.cuda.synchronize()
    ref = _ref_f32(token, weight, offsets, S, N, E)
    torch.testing.assert_close(output[0], ref.to(torch.bfloat16), atol=0.125, rtol=0.01)
    assert (raw[size:] == -123).all()
    if major == "m":
        assert (raw[:size].view(N, ld)[:, S:] == -123).all()


@requires_sm120
@pytest.mark.parametrize(
    "a_dt,b_dt,cfg_name,S,N,starts,alignment,major,global_desc",
    [
        ("HALF", "HALF", "CONFIG_sm120_128x32x128_16x16x32_cluster1x1_warps4x2", 1, 1, [0, 0, 1], 1, "n", False),
        ("BFLOAT16", "BFLOAT16", "CONFIG_sm120_128x64x128_16x16x32_cluster1x1_warps4x2", 256, 256, [0, 64, 64, 128, 256], 64, "m", True),
        ("BFLOAT16", "BFLOAT16", "CONFIG_sm120_128x64x128_16x16x32_cluster1x1_warps4x2", 264, 256, [0, 64, 64, 128, 256], 64, "m", True),
        ("FP8_E4M3", "FP8_E5M2", "CONFIG_sm120_128x48x128_16x16x32_cluster1x1_warps8x1", 173, 19, [0, 3, 3, 99, 173], 1, "n", False),
        ("FP8_E5M2", "FP8_E4M3", "CONFIG_sm120_128x32x128_16x16x32_cluster1x1_warps8x1", 173, 256, [0, 3, 3, 99, 173], 1, "m", False),
        ("FP8_E4M3", "FP8_E5M2", "CONFIG_sm120_128x64x128_16x16x32_cluster1x1_warps2x4", 256, 256, [0, 64, 64, 128, 256], 64, "m", True),
    ],
)
def test_moe_swap_ab_plain_dtypes(a_dt, b_dt, cfg_name, S, N, starts, alignment, major, global_desc):
    from cudnn.gemm.frost.graph_analyzer import analyze_with_binding
    from cudnn.gemm.frost.sm120 import compiler as C

    E, K = 2, 144
    g = _build_graph(E, S, N, K, num_groups=len(starts) - 1, output_dt=cudnn.data_type.FLOAT)
    _, binding = analyze_with_binding(g)
    binding.a_operands[0].set_data_type(getattr(cudnn.data_type, a_dt))
    binding.b_operands[0].set_data_type(getattr(cudnn.data_type, b_dt))
    binding.first_token_offset.set_alignment_value(alignment)
    ld = ((S if major == "m" else N) + 3) // 4 * 4
    binding.outputs[0].set_stride([N * ld, 1, ld] if major == "m" else [S * ld, ld, 1])
    cfg = by_name(cfg_name)
    compiled = C.jit_from_cudnn_graph(g, config=replace(cfg, swap_ab=True))
    # sm120 has no _moe_aligned_offsets: the swap-AB kernel addresses the routed token (B)
    # by coordinate on one global descriptor for every promise (`global_desc` is the sm100
    # choice), so it patches no tensormap, its workspace is the scheduler counter alone,
    # and it stores STG, masking token columns past the group end at the store.
    src = compiled.generated_path.read_text()
    assert "moe_desc_slots = 0\n" in src and "moe_aligned_offsets" not in src
    assert "if col_j < group_end:" in src
    assert compiled.workspace_bytes == C._MOE_SCHED_COUNTER_SLOTS * C._MOE_DESC_SLOT_BYTES
    assert compiled.store_modes == ("stg",)
    types = {"HALF": torch.float16, "BFLOAT16": torch.bfloat16, "FP8_E4M3": torch.float8_e4m3fn, "FP8_E5M2": torch.float8_e5m2}
    token = torch.randint(-3, 4, (1, S, K), device="cuda").to(types[a_dt])
    weight = torch.randint(-3, 4, (E, N, K), device="cuda").to(types[b_dt])
    output = torch.empty_strided((1, S, N), (N * ld, 1, ld) if major == "m" else (S * ld, ld, 1), device="cuda", dtype=torch.float32)
    output.fill_(-123)
    end = starts[-1]
    offsets = torch.tensor(starts, device="cuda", dtype=torch.int32)
    compiled(_vp_moe(compiled, token, weight, offsets, output))
    torch.cuda.synchronize()
    ref = _ref_f32(token, weight, offsets, S, N, E)
    ref[end:] = -123
    torch.testing.assert_close(output[0], ref, atol=0, rtol=0)


@requires_sm120
@pytest.mark.parametrize("cfg_name", ["CONFIG_sm120_128x32x128_16x16x32_cluster1x1_warps4x2", "CONFIG_sm120_128x32x128_16x16x32_cluster1x1_warps8x1"])
def test_moe_swap_ab_plain_swiglu(cfg_name, monkeypatch):
    from functools import partial
    import test_moe_grouped_matmul_fwd_swiglu as cases

    monkeypatch.setattr(cases, "_plan", partial(_plan, swap_ab=True))
    from test_moe_grouped_matmul_fwd_swiglu import test_dual_moe_grouped_matmul_fwd_swiglu_exact_case

    test_dual_moe_grouped_matmul_fwd_swiglu_exact_case(cfg_name, 1)


@requires_sm120
@pytest.mark.parametrize("aligned", (False, True))
def test_moe_swap_ab_plain_graph_coordinates_and_public_replay(aligned):
    from cudnn.gemm.frost.graph_analyzer import analyze_with_binding
    from cudnn.engines.manifest import MANIFEST
    from cudnn.gemm.frost.knobs import GemmKnobs

    S, N, K, G, E = (256 if aligned else 173), (256 if aligned else 136), 128, 4, 2
    bounds = [0, 8, 8, 80, S]
    graph = _build_graph(E, S, N, K, num_groups=G, output_dt=cudnn.data_type.FLOAT)
    _, binding = analyze_with_binding(graph)
    binding.first_token_offset.set_alignment_value(8 if aligned else 1)
    binding.outputs[0].set_output(False)
    base = graph.identity(input=binding.outputs[0], name="materialized")
    base.set_data_type(cudnn.data_type.FLOAT)
    bias = graph.tensor(name="group_bias", dim=[G, 1, N], stride=[N, N, 1], data_type=cudnn.data_type.FLOAT)
    value = graph.add(a=base, b=bias, name="biased")
    for axis in (1, 2):
        value = graph.add(a=value, b=graph.gen_index(input=base, axis=axis), name=f"index_{axis}")
    base.set_output(True)
    ldm = _ceil_div(S, 8) * 8
    value.set_output(True).set_data_type(cudnn.data_type.BFLOAT16).set_stride([ldm * N, 1, ldm])
    cfg = by_name("CONFIG_sm120_128x32x128_16x16x32_cluster1x1_warps4x2")
    plan = _plan(graph, config=cfg, swap_ab=True)
    assert plan._compiled.store_modes == ("stg", "stg")  # sm120: every output is STG-stored (no TMA-store epilogue), aligned or not
    tok = torch.ones((1, S, K), dtype=torch.bfloat16, device="cuda")
    weight = torch.ones((E, N, K), dtype=torch.bfloat16, device="cuda")
    offsets = torch.tensor(bounds, dtype=torch.int32, device="cuda")
    bias_buf = torch.arange(G * N, dtype=torch.float32, device="cuda").view(G, 1, N)
    raw = torch.full((4 * S * N + 4096,), 0xAB, dtype=torch.uint8, device="cuda")
    base_buf = raw[: 4 * S * N].view(torch.float32).view(1, S, N)
    raw_m = torch.full((2 * ldm * N + 4096,), 0xAB, dtype=torch.uint8, device="cuda")
    value_buf = raw_m[: 2 * ldm * N].view(torch.bfloat16).view(1, N, ldm).transpose(1, 2)[:, :S, :]
    vp = _vp_moe(plan, tok, weight, offsets, [base_buf, value_buf])
    vp[bias] = bias_buf
    if aligned:
        engine_id = next(row.engine_id for row in MANIFEST if row.name == "frost_gemm")
        graph.validate()
        graph.build_operation_graph()
        public = GemmKnobs.from_config(replace(cfg, swap_ab=True)).to_public()
        assert public[cudnn.knob_type.SWAP_AB] == 1
        graph.create_execution_plan(engine_id, public)
        graph.check_support()
        graph.build_plans()
        assert graph.get_engine_and_knobs_at_index(0) == (engine_id, public)
        assert graph.selected_engine.name == "frost_gemm"
        workspace = torch.empty(max(graph.get_workspace_size(), plan.workspace_bytes), dtype=torch.uint8, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        handle = cudnn.create_handle()
        cudnn.set_stream(handle, stream.cuda_stream)
        try:
            with torch.cuda.stream(stream):
                graph.execute(vp, workspace, handle=handle)
                base_expected = torch.full_like(base_buf, K)
                torch.testing.assert_close(base_buf, base_expected, atol=0, rtol=0)
                base_buf.fill_(float("nan"))
                value_buf.fill_(float("nan"))
                plan._compiled(vp, workspace=workspace, stream=stream.cuda_stream)
            stream.synchronize()
        finally:
            cudnn.destroy_handle(handle)
    else:
        plan(vp)
    torch.cuda.synchronize()
    torch.testing.assert_close(base_buf, torch.full_like(base_buf, K), atol=0, rtol=0)
    ref = torch.empty(S, N, device="cuda")
    for group, (begin, end) in enumerate(zip(bounds, bounds[1:])):
        ref[begin:end] = K + bias_buf[group] + torch.arange(begin, end, device="cuda")[:, None] + torch.arange(N, device="cuda")[None, :]
    torch.testing.assert_close(value_buf[0], ref.to(torch.bfloat16), atol=0, rtol=0)
    assert (raw[4 * S * N :] == 0xAB).all()
    assert (raw_m[2 * ldm * N :] == 0xAB).all()
    assert (raw_m[: 2 * ldm * N].view(N, 2 * ldm)[:, 2 * S :] == 0xAB).all()


@requires_sm120
@pytest.mark.parametrize("block_scale", [False, True])
def test_moe_templates_follow_internal_operation(block_scale, monkeypatch):
    from cudnn.gemm.frost import fusion_ir, kernel_registry as R
    from cudnn.gemm.frost.sm120 import compiler as C
    from test_moe_grouped_block_scale_matmul_fwd import _build_graph as build_bs

    graph = (build_bs if block_scale else _build_graph)(2, 176, 256, 256, num_groups=4)
    chain = analyze(graph)
    swapped = fusion_ir.swap_ab(chain)
    cfg = by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2")
    monkeypatch.setattr(R, "CATALOG", (cfg,))
    matches = R.candidates(chain, sweep_swap_ab=True)
    assert len(matches) == 2
    assert {candidate.swap_ab for _, candidate in matches} == {False, True}
    for template, candidate in matches:
        lowered = swapped if candidate.swap_ab else chain
        other = chain if candidate.swap_ab else swapped
        assert R.select_template(lowered, candidate) is template
        assert template.accepts(lowered, candidate) is None
        assert "graph_type" in template.accepts(other, candidate)
        C.probe_chain(chain, candidate)
    assert analyze(graph) == chain
    assert fusion_ir.swap_ab(swapped) == chain


@requires_sm120
def test_moe_swap_ab_config_executes_the_same_public_graph():
    from cudnn.gemm.frost.fusion_ir import swap_ab
    from cudnn.gemm.frost.graph_analyzer import analyze_with_binding
    from cudnn.gemm.frost.sm120 import compiler as C

    E, S, N, K = 2, 173, 256, 128
    graph = _build_graph(E, S, N, K, num_groups=4)
    chain, binding = analyze_with_binding(graph)
    cfg = by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2")
    offsets = torch.tensor([0, 7, 7, 80, S], dtype=torch.int32, device="cuda")
    token = torch.randint(-2, 3, (1, S, K), device="cuda").to(torch.bfloat16)
    weight = torch.randint(-2, 3, (E, N, K), device="cuda").to(torch.bfloat16)
    output = torch.empty((1, S, N), dtype=torch.bfloat16, device="cuda")
    vp = {binding.a_operands[0]: token, binding.b_operands[0]: weight, binding.first_token_offset: offsets, binding.outputs[0]: output}
    reference = _ref_f32(token, weight, offsets, S, N, E).to(torch.bfloat16)
    for flag in (False, True):
        config = replace(cfg, swap_ab=flag)
        compiled = C.jit_from_cudnn_graph(graph, config=config)
        assert compiled.chain == (swap_ab(chain) if flag else chain)
        assert compiled.config == config
        output.fill_(float("nan"))
        compiled(vp)
        torch.cuda.synchronize()
        torch.testing.assert_close(output[0], reference, atol=0, rtol=0)


@requires_sm120
@pytest.mark.parametrize("stg", [False, True])
@pytest.mark.parametrize("aux_major", ["n", "m"])
def test_moe_swap_ab_transposes_strided_aux(stg, aux_major):
    from cudnn.gemm.frost.graph_analyzer import analyze_with_binding
    from cudnn.gemm.frost.sm120 import compiler as C

    E, S, N, K = 2, 173, 136, 128
    graph = _build_graph(E, S, N, K, num_groups=4, output_dt=cudnn.data_type.FLOAT)
    _, binding = analyze_with_binding(graph)
    base = binding.outputs[0].set_output(False)
    stride = (S * N, N, 1) if aux_major == "n" else (S * N, 1, S)
    elem = graph.tensor(name="elem", dim=[1, S, N], stride=list(stride), data_type=cudnn.data_type.FLOAT)
    row = graph.tensor(name="token_bias", dim=[1, S, 1], stride=[2 * S, 2, 1], data_type=cudnn.data_type.FLOAT)
    value = graph.add(a=graph.add(a=base, b=elem), b=row).set_output(True).set_data_type(cudnn.data_type.FLOAT)
    config = replace(by_name("CONFIG_sm120_128x32x128_16x16x32_cluster1x1_warps4x2"), swap_ab=True)
    with C.force_stg_epi(stg):
        compiled = C.jit_from_cudnn_graph(graph, config=config)
    assert compiled.store_modes == ("stg",)  # sm120: STG with or without force_stg_epi (no TMA-store epilogue)
    token = torch.ones(1, S, K, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(E, N, K, device="cuda", dtype=torch.bfloat16)
    offsets = torch.tensor([0, 3, 3, 99, S], device="cuda", dtype=torch.int32)
    elem_buf = torch.empty_strided((1, S, N), stride, device="cuda", dtype=torch.float32)
    elem_buf.copy_(torch.arange(S * N, device="cuda").view(1, S, N))
    row_buf = torch.arange(2 * S, device="cuda", dtype=torch.float32)[::2].view(1, S, 1)
    output = torch.full((1, S, N), float("nan"), device="cuda", dtype=torch.float32)
    compiled({binding.a_operands[0]: token, binding.b_operands[0]: weight, binding.first_token_offset: offsets, elem: elem_buf, row: row_buf, value: output})
    torch.cuda.synchronize()
    torch.testing.assert_close(output, K + elem_buf + row_buf, atol=0, rtol=0)


@requires_sm120
def test_moe_swap_ab_sweep_includes_geometry_rejected_by_original_graph(monkeypatch):
    from cudnn.gemm.frost import kernel_registry as R
    from cudnn.gemm.frost.graph_analyzer import analyze_with_binding
    from cudnn.gemm.frost.sm120 import compiler as C

    graph = _build_graph(2, 173, 256, 256, num_groups=4, weight_major="n")
    _, binding = analyze_with_binding(graph)
    for tensor in binding.a_operands + binding.b_operands:
        tensor.set_data_type(cudnn.data_type.FP8_E4M3)
    chain = analyze(graph)
    cfg = by_name("CONFIG_sm120_128x48x128_16x16x32_cluster1x1_warps8x1")
    monkeypatch.setattr(R, "CATALOG", (cfg,))
    assert R.candidates(chain) == []
    [(template, swapped)] = R.candidates(chain, sweep_swap_ab=True)
    assert swapped == replace(cfg, swap_ab=True)
    assert template.graph_type is R.GraphType.MOE_SWAP_AB
    C.probe_chain(chain, swapped)


@pytest.mark.parametrize("block_scale", [False, True])
@pytest.mark.parametrize("swap", [False, True])
def test_moe_norm2_is_refused_at_probe_and_jit(block_scale, swap):
    from cudnn.gemm.frost.sm120 import compiler
    from test_moe_grouped_block_scale_matmul_fwd import _build_graph as build_bs

    graph = (build_bs if block_scale else _build_graph)(
        2,
        256,
        256,
        256,
        num_groups=4,
        reduction_mode=cudnn.reduction_mode.NORM2,
        reduction_dims=(1, 1, 256),
    )
    config = replace(by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"), swap_ab=swap)
    chain = analyze(graph)
    with pytest.raises(NotImplementedError, match="norm2.*square root"):
        compiler.probe_chain(chain, config)
    with pytest.raises(NotImplementedError, match="norm2.*square root"):
        compiler.jit_from_cudnn_graph(graph, config=config)


def _run_moe_swap_ab_quant(case, *, block_scale=False):
    from contextlib import nullcontext
    from cudnn.gemm.frost.fusion_ir import segmented_row_scale_capacity_rows
    from cudnn.gemm.frost.graph_analyzer import analyze_with_binding
    from cudnn.gemm.frost.sm120 import compiler as C
    from test_moe_grouped_block_scale_matmul_fwd import _build_graph as build_bs

    col = case in ("col", "col_grouped", "multi")
    grouped = case in ("row_grouped", "col_grouped")
    reorder = case not in ("row", "row_stg", "multi")
    bs = 16 if case == "packed" else 32
    S = (512 if grouped else 320) if col else 173
    starts = ([0, 128, 128, 256, 384, S, S] if grouped else [0, 32, 32, 192, 256, S, S]) if col else [0, 3, 3, 99, 151, S, S]
    E, N, K, G = 2, 160, 256, len(starts) - 1
    graph = (build_bs if block_scale else _build_graph)(E, S, N, K, num_groups=G, output_dt=cudnn.data_type.FLOAT)
    _, binding = analyze_with_binding(graph)
    base = binding.outputs[0]
    base.set_output(False)
    binding.first_token_offset.set_alignment_value((128 if grouped else 32) if col else 1)
    bias = graph.tensor(name="quant_bias", dim=[1, S, 1], stride=[S, 1, 1], data_type=cudnn.data_type.FLOAT)
    source = graph.add(a=base, b=bias, name="quant_source")
    quant_outputs = []
    for axis in ([1, 2] if case == "multi" else [1 if col else 2]):
        kwargs = {"group_offset": binding.first_token_offset} if grouped else {}
        q, sf = graph.block_scale_quantize(input=source, block_size=bs, axis=axis, name=f"quant_{axis}", **kwargs)
        q_dtype = torch.float8_e5m2 if case == "col" else torch.float8_e4m3fn
        q.set_data_type(cudnn.data_type.FP8_E5M2 if case == "col" else cudnn.data_type.FP8_E4M3).set_output(True)
        q.set_dim([1, S, N]).set_stride([S * N, 1, S] if col or case == "row_mmajor" else [S * N, N, 1])
        sf_dtype = torch.float8_e4m3fn if case == "row_reorder" else torch.float8_e8m0fnu
        sf.set_data_type(cudnn.data_type.FP8_E4M3 if case == "row_reorder" else cudnn.data_type.FP8_E8M0).set_output(True)
        if reorder:
            rows = N if axis == 1 else (segmented_row_scale_capacity_rows(S, G) if grouped else S)
            cols = (S if axis == 1 else N) // bs
            dims = (1, _ceil_div(rows, 128) * 128, _ceil_div(cols, 4) * 4)
            sf.set_dim(dims).set_stride([dims[1] * dims[2], dims[2], 1]).set_reordering_type(cudnn.tensor_reordering.F8_128x4)
        else:
            dims = (1, S // bs, N) if axis == 1 else (1, S, N // bs)
            sf.set_dim(dims).set_stride([dims[1] * dims[2] * 2, dims[2] * 2, 2])
        quant_outputs.append((axis, q, sf, q_dtype, sf_dtype, dims))
    red = graph.reduction(input=source, mode=cudnn.reduction_mode.ADD, name="sum")
    red.set_dim([1, 1, 1]).set_stride([1, 1, 1]).set_data_type(cudnn.data_type.FLOAT).set_output(True)
    cta_m = 64 if case == "packed" and not block_scale else 128
    cta_n = 128 if block_scale else 64
    cfg = replace(by_name(f"CONFIG_sm120_{cta_m}x{cta_n}x128_16x16x32_cluster1x1_warps4x2"), swap_ab=True)
    with C.force_stg_epi() if case in ("row_stg", "packed") else nullcontext():
        compiled = C.jit_from_cudnn_graph(graph, config=cfg)
    assert compiled.chain.quants[0].axis == (2 if col else 1)
    # sm120 renders only the transposed-STG epilogue (_TMA_STORE_EPI_PIPELINES is
    # ("sm100", "sm103")), so every output is "stg"; the parent's TMA/STG split
    # is the sm100 store-mode gate.
    assert compiled.store_modes[0] == "stg"
    tok_values = torch.arange(S, device="cuda") % 3 + 1
    w_values = (torch.arange(N, device="cuda") % 3 + 1)[None, :].expand(E, N)
    signs = torch.where((torch.arange(N, device="cuda")[None, :] + torch.arange(E, device="cuda")[:, None]) % 2 == 0, 1, -1)
    token = torch.zeros((1, S, K), device="cuda", dtype=torch.bfloat16)
    weight = torch.zeros((E, N, K), device="cuda", dtype=torch.bfloat16)
    token[0, :, 0] = tok_values
    weight[:, :, 0] = w_values * signs
    if block_scale:
        codes = torch.tensor([2, 4, 5], device="cuda", dtype=torch.uint8)
        token = torch.zeros((1, S, K // 2), device="cuda", dtype=torch.uint8)
        weight = torch.zeros((E, N, K // 2), device="cuda", dtype=torch.uint8)
        token[0, :, 0] = codes[tok_values - 1]
        weight[:, :, 0] = codes[w_values - 1] + (signs < 0).to(torch.uint8) * 8
        token = token.view(torch.float4_e2m1fn_x2)
        weight = weight.view(torch.float4_e2m1fn_x2)
    offsets = torch.tensor(starts, device="cuda", dtype=torch.int32)
    bias_buf = (torch.arange(S, device="cuda") % 5 - 2).float().view(1, S, 1)
    vp = {binding.a_operands[0]: token, binding.b_operands[0]: weight, binding.first_token_offset: offsets, bias: bias_buf}
    if block_scale:
        sf_rows = segmented_row_scale_capacity_rows(S, G)
        vp[binding.sfa_operands[0]] = torch.ones((1, sf_rows, K // 16), device="cuda", dtype=torch.float8_e4m3fn)
        vp[binding.sfb_operands[0]] = torch.ones((E, 256, K // 16), device="cuda", dtype=torch.float8_e4m3fn)
    ref = torch.empty((S, N), device="cuda")
    for group, (begin, end) in enumerate(zip(starts, starts[1:])):
        ref[begin:end] = tok_values[begin:end, None] * (w_values * signs)[group % E] + bias_buf[0, begin:end]
    guards = []
    for axis, q, sf, q_dtype, sf_dtype, dims in quant_outputs:
        for tensor, shape, dtype in ((q, (1, S, N), q_dtype), (sf, dims, sf_dtype)):
            stride = tuple(tensor.get_stride())
            size = 1 + sum((d - 1) * st for d, st in zip(shape, stride))
            raw = torch.full((size + 32,), 0xA5, device="cuda", dtype=torch.uint8)
            vp[tensor] = raw[:size].view(dtype).as_strided(shape, stride)
            untouched = torch.ones_like(raw, dtype=torch.bool)
            untouched.as_strided(shape, stride).fill_(False)
            guards.append((raw, untouched))
    vp[red] = torch.empty((1, 1, 1), device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    workspace = torch.empty(compiled.workspace_bytes, device="cuda", dtype=torch.uint8)
    if case == "row":
        bad = dict(vp)
        bad[binding.a_operands[0]] = torch.empty((1, S + 1, token.shape[2]), device="cuda", dtype=token.dtype)
        with pytest.raises(ValueError, match="scale_dim"):
            compiled(bad, workspace=workspace, stream=stream.cuda_stream)
    if reorder:
        scale = quant_outputs[0][2]
        valid = vp[scale]
        vp[scale] = torch.empty_strided(valid.shape, (valid.numel() * 2, valid.shape[2] * 2, 2), device="cuda", dtype=valid.dtype)
        with pytest.raises(ValueError, match="dense byte run"):
            compiled(vp, workspace=workspace, stream=stream.cuda_stream)
        vp[scale] = valid
    for _ in range(2):
        compiled(vp, workspace=workspace, stream=stream.cuda_stream)
        stream.synchronize()
        torch.testing.assert_close(vp[red], ref.sum().reshape(1, 1, 1), atol=0, rtol=0)
        for axis, q, sf, q_dtype, sf_dtype, dims in quant_outputs:
            data = ref.T.contiguous() if axis == 1 else ref
            q_ref, scale_ref = _block_quant_ref(data, bs, q_dtype, sf_dtype)
            if axis == 1:
                q_ref = q_ref.transpose(1, 2)
            torch.testing.assert_close(vp[q].float(), q_ref.float(), atol=_block_quant_q_atol(sf_dtype), rtol=0)
            if reorder:
                expected = torch.full(dims, 0xA5, device="cuda", dtype=torch.uint8).flatten()
                cursor = 0
                for begin, end in (list(zip(starts, starts[1:])) if grouped else [(0, S)]):
                    part = scale_ref[0, :, begin // bs : end // bs] if axis == 1 else scale_ref[0, begin:end]
                    if part.numel() == 0:
                        continue
                    padded = torch.full((_ceil_div(part.shape[0], 128) * 128, _ceil_div(part.shape[1], 4) * 4), 0xA5, device="cuda", dtype=torch.uint8)
                    padded[: part.shape[0], : part.shape[1]] = part.view(torch.uint8)
                    packed = _to_blocked(padded).flatten()
                    expected[cursor : cursor + packed.numel()] = packed
                    cursor += packed.numel()
                torch.testing.assert_close(vp[sf].view(torch.uint8).flatten(), expected, atol=0, rtol=0)
            else:
                expected = scale_ref.transpose(1, 2) if axis == 1 else scale_ref
                torch.testing.assert_close(vp[sf].float(), expected.float(), atol=0, rtol=0)
        for raw, untouched in guards:
            assert (raw[untouched] == 0xA5).all()


@requires_sm120
@pytest.mark.parametrize("case", ["row", "row_stg", "row_mmajor", "row_reorder", "col", "col_grouped", "multi", "packed"])
def test_moe_swap_ab_quant(case):
    _run_moe_swap_ab_quant(case)


@pytest.mark.parametrize("block_scale", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
def test_moe_swap_ab_token_quant_requires_aligned_groups(block_scale, grouped):
    from cudnn.gemm.frost.graph_analyzer import analyze_with_binding
    from cudnn.gemm.frost.sm120 import compiler as C
    from test_moe_grouped_block_scale_matmul_fwd import _build_graph as build_bs

    graph = (build_bs if block_scale else _build_graph)(2, 256, 128, 256, num_groups=4)
    _, binding = analyze_with_binding(graph)
    base = binding.outputs[0].set_output(False)
    kwargs = {"group_offset": binding.first_token_offset} if grouped else {}
    q, sf = graph.block_scale_quantize(input=base, block_size=32, axis=1, **kwargs)
    q.set_data_type(cudnn.data_type.FP8_E4M3).set_output(True)
    sf.set_data_type(cudnn.data_type.FP8_E8M0).set_output(True)
    if grouped:
        sf.set_dim([1, 128, 8]).set_stride([1024, 8, 1]).set_reordering_type(cudnn.tensor_reordering.F8_128x4)
    cfg = replace(by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"), swap_ab=True)
    with pytest.raises(NotImplementedError, match="first_token_offset value alignment"):
        C.probe_chain(analyze(graph), cfg)
    binding.first_token_offset.set_alignment_value(128 if grouped else 32)
    C.probe_chain(analyze(graph), cfg)


def test_moe_swap_ab_segmented_feature_quant_requires_scale_prefix():
    from cudnn.gemm.frost.graph_analyzer import analyze_with_binding
    from cudnn.gemm.frost.sm120 import compiler as C

    graph = _build_graph(2, 173, 128, 256, num_groups=4)
    _, binding = analyze_with_binding(graph)
    base = binding.outputs[0].set_output(False)
    q, sf = graph.block_scale_quantize(input=base, block_size=32, group_offset=binding.first_token_offset)
    q.set_data_type(cudnn.data_type.FP8_E4M3).set_output(True)
    sf.set_data_type(cudnn.data_type.FP8_E8M0).set_dim([1, 640, 4]).set_stride([2560, 4, 1])
    sf.set_reordering_type(cudnn.tensor_reordering.F8_128x4).set_output(True)
    cfg = replace(by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"), swap_ab=True)
    with pytest.raises(NotImplementedError, match="requires a block-scaled MoE"):
        C.probe_chain(analyze(graph), cfg)


def _run_moe_swap_ab_reductions(axis, *, block_scale=False, reduction_kind="basic", swap=True):
    from contextlib import nullcontext
    from cudnn.gemm.frost.graph_analyzer import analyze_with_binding
    from cudnn.gemm.frost.sm120 import compiler as C
    from test_moe_grouped_block_scale_matmul_fwd import _build_graph as build_bs

    avg = reduction_kind == "avg"
    products = reduction_kind == "products"
    red_dt = cudnn.data_type.INT32 if reduction_kind == "int32" else cudnn.data_type.FLOAT
    red_torch_dt = torch.int32 if reduction_kind == "int32" else torch.float32
    E, N, K = 2, (136 if swap else 128), 256
    aligned = axis == "group_feature"
    S = 176 if aligned else 173
    starts = [0, 8, 8, 80, 152, S, S] if aligned else [0, 3, 3, 99, 151, S, S]
    G = len(starts) - 1
    graph = (build_bs if block_scale else _build_graph)(E, S, N, K, num_groups=G, output_dt=cudnn.data_type.FLOAT)
    _, binding = analyze_with_binding(graph)
    base = binding.outputs[0]
    if axis == "feature":
        base.set_output(False)
        base = graph.identity(input=base, name="materialized").set_data_type(cudnn.data_type.FLOAT)
    base.set_dim([1, S, N]).set_stride([S * N, N, 1])
    binding.first_token_offset.set_alignment_value(8 if aligned else 1)
    if aligned:
        base.set_stride([S * N, 1, S])
    source = graph.add(a=base, b=base, name="twice") if axis == "feature" and not products else base
    reduction_only = axis == "group"
    base.set_output(not reduction_only)
    dims = {
        "scalar": (1, 1, 1),
        "feature": (1, 1, N),
        "token": (1, S, 1),
        "group": (G, 1, 1),
        "group_feature": (G, 1, N),
        "group_token": (G, S, 1),
    }[axis.removesuffix("_tma")]
    strides = (dims[1] * dims[2] * 2 + 8, dims[2] * 2, 2)
    modes = [cudnn.reduction_mode.AMAX, cudnn.reduction_mode.ADD, cudnn.reduction_mode.MAX, cudnn.reduction_mode.MIN, cudnn.reduction_mode.NORM1]
    if avg:
        modes = [cudnn.reduction_mode.AVG]
    if products:
        modes = [cudnn.reduction_mode.MUL, cudnn.reduction_mode.MUL_NO_ZEROS]
    reductions = []
    for i, mode in enumerate(modes):
        kwargs = {"group_offset": binding.first_token_offset} if axis.startswith("group") else {}
        red = graph.reduction(input=source, mode=mode, name=f"red_{i}", compute_data_type=red_dt, **kwargs)
        red.set_dim(list(dims)).set_stride(list(strides)).set_data_type(red_dt).set_output(True)
        reductions.append(red)
    cta_m = 64 if axis == "scalar" and not block_scale else 128
    cta_n = 128 if block_scale else 64
    cfg = by_name(f"CONFIG_sm120_{cta_m}x{cta_n}x128_16x16x32_cluster1x1_warps4x2")
    with C.force_stg_epi() if axis == "token" else nullcontext():
        compiled = C.jit_from_cudnn_graph(graph, config=replace(cfg, swap_ab=swap))
    # sm120: the dense output is STG-stored for every axis ("token_tma" included).
    dense_modes = () if reduction_only else ("stg",)
    assert compiled.store_modes == dense_modes + ("stg",) * len(modes)

    # Exact, nonzero integer products: negative feature columns catch accidental
    # zero contributions to MAX; positive columns do the same for MIN.
    token_ref = (torch.arange(S, device="cuda") % 3 + 1).view(1, S, 1).expand(1, S, K).to(torch.bfloat16).contiguous()
    signs = torch.where(torch.arange(N, device="cuda") % (3 if avg else 2) == 0, 1, -1)
    weight_ref = (torch.arange(1, E + 1, device="cuda")[:, None, None] * signs[None, :, None]).expand(E, N, K).to(torch.bfloat16).contiguous()
    if products:
        token_ref.zero_()
        token_ref[0, :, 0] = torch.where(torch.arange(S, device="cuda") % 3 == 0, -1, 1)
        weight_ref.zero_()
        weight_ref[:, :, 0] = signs
    token, weight = token_ref, weight_ref
    if block_scale:
        codes = torch.tensor([2, 4, 5], device="cuda", dtype=torch.uint8)
        tok_codes = codes[torch.arange(S, device="cuda") % 3]
        token = (tok_codes * 17).view(1, S, 1).expand(1, S, K // 2).contiguous().view(torch.float4_e2m1fn_x2)
        w_codes = 2 * torch.arange(1, E + 1, device="cuda")[:, None] + (signs < 0)[None, :] * 8
        weight = (w_codes * 17).to(torch.uint8)[:, :, None].expand(E, N, K // 2).contiguous().view(torch.float4_e2m1fn_x2)
        if products:
            token.view(torch.uint8).zero_()
            weight.view(torch.uint8).zero_()
            token.view(torch.uint8)[0, :, 0] = (token_ref[0, :, 0] < 0).to(torch.uint8) * 8 + 2
            weight.view(torch.uint8)[:, :, 0] = (weight_ref[:, :, 0] < 0).to(torch.uint8) * 8 + 2
    offsets = torch.tensor(starts, device="cuda", dtype=torch.int32)
    vp = {binding.a_operands[0]: token, binding.b_operands[0]: weight, binding.first_token_offset: offsets}
    if block_scale:
        sf_rows = (min(S, G) + (S - min(S, G)) // 128) * 128
        vp[binding.sfa_operands[0]] = torch.ones((1, sf_rows, K // 16), device="cuda", dtype=torch.float8_e4m3fn)
        vp[binding.sfb_operands[0]] = torch.ones((E, 256, K // 16), device="cuda", dtype=torch.float8_e4m3fn)
    if not reduction_only:
        vp[base] = torch.empty_strided((1, S, N), tuple(base.get_stride()), device="cuda", dtype=torch.float32)
    guards = []
    for red in reductions:
        size = sum((d - 1) * s for d, s in zip(dims, strides)) + 1
        raw = torch.full((size + 32,), -12345, dtype=red_torch_dt, device="cuda")
        vp[red] = raw.as_strided(dims, strides)
        untouched = torch.ones_like(raw, dtype=torch.bool)
        untouched.as_strided(dims, strides).fill_(False)
        guards.append((raw, untouched))
    ref = _ref_f32(token_ref, weight_ref, offsets, S, N, E)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    workspace = torch.empty(compiled.workspace_bytes, device="cuda", dtype=torch.uint8)
    if axis == "feature":
        valid = vp[reductions[-1]]
        for bad, message in (
            (valid.as_strided(dims, (0, 0, 0)), "cannot write an element twice"),
            (torch.empty_strided(dims, strides, device="cuda", dtype=torch.float16), "4-byte elements"),
        ):
            vp[reductions[-1]] = bad
            with pytest.raises(ValueError, match=message):
                compiled(vp, stream=stream.cuda_stream, workspace=workspace)
            stream.synchronize()
            assert all((raw == -12345).all() for raw, _ in guards)
        vp[reductions[-1]] = valid
    # Repeat with explicit stream/workspace: every call must reseed each output,
    # including the empty group's identity, without touching stride padding.
    for iteration in range(2):
        if products and iteration == 1:
            token_ref[0, [1, 100], 0] = 0
            if block_scale:
                token.view(torch.uint8)[0, [1, 100], 0] = 0
            ref = _ref_f32(token_ref, weight_ref, offsets, S, N, E)
            stream.wait_stream(torch.cuda.current_stream())
        compiled(vp, stream=stream.cuda_stream, workspace=workspace)
        stream.synchronize()
        if not reduction_only:
            torch.testing.assert_close(vp[base][0], ref, atol=0, rtol=0)
        for red, mode in zip(reductions, modes):
            red_ref = (ref * 2 if axis == "feature" and not products else ref).to(red_torch_dt)
            if axis.startswith("group"):
                expected = _group_reduction_ref(red_ref, offsets, mode, dims, red_torch_dt)
            else:
                expected = _reduction_ref(red_ref[None], mode, _reduction_dims(dims, (1, S, N)))
            approximate = avg
            torch.testing.assert_close(
                vp[red], expected.to(red_torch_dt), atol=1e-4 if approximate else 0, rtol=1e-4 if approximate else 0, msg=lambda message: f"{mode}: {message}"
            )
        for raw, untouched in guards:
            assert (raw[untouched] == -12345).all()


@requires_sm120
@pytest.mark.parametrize("axis", ["scalar", "feature", "token", "token_tma", "group", "group_feature"])
def test_moe_swap_ab_reductions_fp32(axis):
    _run_moe_swap_ab_reductions(axis)


@requires_sm120
@pytest.mark.parametrize("axis", ["scalar", "feature", "token", "token_tma", "group", "group_feature", "group_token"])
def test_moe_swap_ab_avg_fp32(axis):
    _run_moe_swap_ab_reductions(axis, reduction_kind="avg")


@requires_sm120
@pytest.mark.parametrize("axis", ["scalar", "feature", "token", "token_tma", "group", "group_feature", "group_token"])
@pytest.mark.parametrize("kind", ["products", "int32"])
def test_moe_swap_ab_remaining_reductions(axis, kind):
    _run_moe_swap_ab_reductions(axis, reduction_kind=kind)


@requires_sm120
@pytest.mark.parametrize("block_scale", [False, True])
def test_moe_product_reductions_fp32(block_scale):
    _run_moe_swap_ab_reductions("feature", block_scale=block_scale, reduction_kind="products", swap=False)


@requires_sm120
@pytest.mark.parametrize("mode", [cudnn.reduction_mode.ADD, cudnn.reduction_mode.AVG], ids=["add", "avg"])
def test_moe_swap_ab_reduction_runtime_token_tail(mode):
    from cudnn.gemm.frost.graph_analyzer import analyze_with_binding
    from cudnn.gemm.frost.sm120 import compiler as C

    E, S, N, K = 2, 192, 136, 128
    graph = _build_graph(E, S, N, K, num_groups=4, output_dt=cudnn.data_type.FLOAT, reduction_mode=mode, reduction_dims=(1, 1, 1))
    _, binding = analyze_with_binding(graph)
    binding.outputs[0].set_stride([S * N, 1, S])
    binding.first_token_offset.set_alignment_value(8)
    cfg = replace(by_name("CONFIG_sm120_128x64x128_16x16x32_cluster1x1_warps4x2"), swap_ab=True)
    compiled = C.jit_from_cudnn_graph(graph, config=cfg)
    assert compiled.store_modes == ("stg", "stg")  # sm120: dense output STG-stored too (no TMA-store epilogue)
    weight = torch.ones((E, N, K), device="cuda", dtype=torch.bfloat16)
    red = torch.empty((1, 1, 1), device="cuda")
    # The declared S divides the 32-column TMA chunk; a legal runtime S need
    # only divide the compiled 8-column STG/alignment contract.
    for runtime_s in (192, 200):
        offsets = torch.tensor([0, 32, 32, 96, runtime_s], device="cuda", dtype=torch.int32)
        token = torch.ones((1, runtime_s, K), device="cuda", dtype=torch.bfloat16)
        out = torch.empty((1, N, runtime_s), device="cuda").transpose(1, 2)
        compiled(_vp_moe(compiled, token, weight, offsets, [out, red]))
        torch.cuda.synchronize()
        torch.testing.assert_close(out, torch.full_like(out, K), atol=0, rtol=0)
        expected = K if mode == cudnn.reduction_mode.AVG else runtime_s * N * K
        torch.testing.assert_close(red, torch.full_like(red, expected), atol=1e-4, rtol=1e-5)


@requires_sm120
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("axis", ["scalar", "feature", "token"])
def test_moe_avg_fp32(grouped, axis):
    _run_moe_reduction(
        "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2",
        1,
        cudnn.reduction_mode.AVG,
        [6 if grouped else 1, 173 if axis == "token" else 1, 136 if axis == "feature" else 1],
        E=2,
        N=136,
        group_sizes=[3, 0, 96, 52, 22, 0],
        group_reduction=grouped,
        integer_inputs=True,
    )


def test_moe_reduction_initialization_validates_all_outputs_before_writing():
    from cudnn.gemm.frost.graph_analyzer import analyze_with_binding
    from cudnn.gemm.frost.sm120 import compiler

    graph = _build_graph(2, 128, 128, 128, reduction_mode=cudnn.reduction_mode.ADD, reduction_dims=(1, 1, 128))
    _, binding = analyze_with_binding(graph)
    source = binding.outputs[0].set_output(False)
    last = graph.reduction(input=source, mode=cudnn.reduction_mode.MAX)
    last.set_dim([1, 1, 128]).set_stride([256, 256, 2]).set_data_type(cudnn.data_type.FLOAT).set_output(True)
    chain = analyze(graph)
    first = torch.full((1, 1, 128), -12345.0, device="cuda")
    last_raw = torch.full((256,), -12345.0, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    for bad in (last_raw.as_strided((1, 1, 128), (0, 0, 0)), torch.empty((1, 1, 128), device="cuda", dtype=torch.float16)):
        with pytest.raises(ValueError, match="cannot write an element twice|4-byte elements"):
            compiler._initialize_reduction_outputs(chain, [first, bad], stream.cuda_stream)
        stream.synchronize()
        assert (first == -12345).all()
        assert (last_raw == -12345).all()
    compiler._initialize_reduction_outputs(chain, [first, last_raw.as_strided((1, 1, 128), (256, 256, 2))], stream.cuda_stream)
    stream.synchronize()
    assert (first == 0).all()
    assert torch.isneginf(last_raw[::2]).all()
    assert (last_raw[1::2] == -12345).all()


# --------------------------------------------------------------------------- #
# sm120 regressions: the row-per-lane quant drain, the token-axis alignment
# gate, the quant-aware auto config, per-row aux / reduction-only chains on the
# plain MoE kernels, and probe / render agreement.
# --------------------------------------------------------------------------- #


@requires_sm120
@pytest.mark.parametrize("block_scale", [False, True])
@pytest.mark.parametrize("axis", ["scalar", "group"])
def test_moe_reductions_plain_kernel(axis, block_scale):
    # "group" is reduction-only (no dense output), which once failed to trace on
    # the plain MoE kernels (NameError: out_stride_l_0).
    _run_moe_swap_ab_reductions(axis, block_scale=block_scale, swap=False)


@requires_sm120
@pytest.mark.parametrize("quant", [False, True], ids=["fp32_out", "quant_out"])
def test_moe_per_row_aux_plain_kernel(quant):
    """A (1, S, 1) per-row aux on the plain MoE kernel. Its prefetch reads `row`,
    which this template binds only inside the drain -- the lane-run drain (fp32
    output) and the row-per-lane one (quant) both host it under their row guard;
    it once failed to trace with NameError: row. Integer data: exact."""
    E, N, K = 4, 256, 128
    starts = [0, 64, 64, 195, 272]
    S = starts[-1]
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=cudnn.data_type.BFLOAT16)
    w = g.tensor(name="weight", dim=[E, K, N], stride=[K * N, 1, K], data_type=cudnn.data_type.BFLOAT16)
    fto = g.tensor(name="first_token_offset", dim=[E + 1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    out = g.moe_grouped_matmul(tok, w, fto, mode=cudnn.moe_grouped_matmul_mode.NONE)
    bias = g.tensor(name="row_bias", dim=[1, S, 1], stride=[S, 1, 1], data_type=cudnn.data_type.FLOAT)
    y = g.bias(input=out, bias=bias, name="biased")
    if quant:
        q, sf = g.block_scale_quantize(input=y, block_size=32, name="q")
        q.set_data_type(cudnn.data_type.FP8_E4M3).set_output(True)
        sf.set_data_type(cudnn.data_type.FP8_E8M0).set_output(True)
    else:
        y.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    compiled = _plan(g, config=by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"))

    gen = torch.Generator(device="cuda").manual_seed(7)
    token = torch.randint(-3, 4, (1, S, K), device="cuda", generator=gen).to(torch.bfloat16)
    weight = torch.randint(-3, 4, (E, N, K), device="cuda", generator=gen).to(torch.bfloat16)
    row_bias = torch.randint(-50, 50, (1, S, 1), device="cuda", generator=gen).float()
    offsets = torch.tensor(starts, dtype=torch.int32, device="cuda")
    ref = _ref_f32(token, weight, offsets, S, N, E) + row_bias[0]
    if quant:
        outs = [torch.empty(1, S, N, dtype=torch.float8_e4m3fn, device="cuda"), torch.empty(1, S, N // 32, dtype=torch.float8_e8m0fnu, device="cuda")]
    else:
        outs = [torch.empty(1, S, N, dtype=torch.float32, device="cuda")]
    vp = _vp_moe(compiled, token, weight, offsets, outs)
    vp[bias] = row_bias
    compiled(vp)
    torch.cuda.synchronize()
    if quant:
        q_ref, scale_ref = _block_quant_ref(ref, 32, torch.float8_e4m3fn, torch.float8_e8m0fnu)
        torch.testing.assert_close(outs[1].float(), scale_ref.float(), atol=0, rtol=0)
        torch.testing.assert_close(outs[0].float(), q_ref.float(), atol=0, rtol=0)
    else:
        torch.testing.assert_close(outs[0][0], ref, atol=0, rtol=0)


def _token_axis_col_quant_graph(E, S, N, K, bs, alignment):
    g = _build_graph(E, S, N, K, num_groups=E, output_dt=cudnn.data_type.FLOAT)
    _, binding = analyze_with_binding(g)
    binding.first_token_offset.set_alignment_value(alignment)
    q, sf = g.block_scale_quantize(input=binding.outputs[0], block_size=bs, axis=1, name="colq")
    q.set_data_type(cudnn.data_type.FP8_E4M3).set_output(True)
    sf.set_dim([1, S // bs, N]).set_stride([S // bs * N, N, 1]).set_data_type(cudnn.data_type.FP8_E8M0).set_output(True)
    return g


@requires_sm120
@pytest.mark.parametrize("cfg_name", ["CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2", "CONFIG_sm120_64x128x128_16x16x32_cluster1x1_warps2x4"])
@pytest.mark.parametrize("bs", [16, 32])
def test_moe_token_axis_col_quant_plain_kernel(bs, cfg_name):
    """Token-axis (col) quant on the plain MoE kernel. A block may not straddle a
    routed group, so the DECLARED first_token_offset alignment has to cover it:
    without it the graph is declined (it once hung the GPU at block 16 and wrote
    wrong bytes at 32 under ragged routing); with it the kernel is bit-exact
    against a quant of its own accumulators."""
    from cudnn.gemm.frost.sm120 import compiler as C

    E, N, K = 4, 256, 128
    starts = [0, 64, 64, 192, 320]
    S = starts[-1]
    cfg = by_name(cfg_name)
    with pytest.raises(NotImplementedError, match="first_token_offset value alignment"):
        C.probe_chain(analyze(_token_axis_col_quant_graph(E, S, N, K, bs, 1)), cfg)
    compiled = _plan(_token_axis_col_quant_graph(E, S, N, K, bs, bs), config=cfg)
    torch.manual_seed(0)
    token = torch.randn(1, S, K, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    offsets = torch.tensor(starts, dtype=torch.int32, device="cuda")
    q = torch.empty(1, S, N, dtype=torch.float8_e4m3fn, device="cuda")
    sf = torch.empty(1, S // bs, N, dtype=torch.float8_e8m0fnu, device="cuda")
    compiled(_vp_moe(compiled, token, weight, offsets, [q, sf]))
    acc = torch.empty(1, S, N, dtype=torch.float32, device="cuda")
    plain = _plan(_build_graph(E, S, N, K, num_groups=E, output_dt=cudnn.data_type.FLOAT), config=cfg)
    plain(_vp_moe(plain, token, weight, offsets, acc))
    torch.cuda.synchronize()
    q_ref, scale_ref = _block_quant_ref(acc[0].T.contiguous(), bs, torch.float8_e4m3fn, torch.float8_e8m0fnu)
    torch.testing.assert_close(sf.float(), scale_ref.transpose(1, 2).float(), atol=0, rtol=0)
    torch.testing.assert_close(q.float(), q_ref.transpose(1, 2).float(), atol=0, rtol=0)


@requires_sm120
@pytest.mark.parametrize("side_output", [False, True], ids=["quant_only", "bf16_side"])
@pytest.mark.parametrize("bs,scale", [(16, "e4m3"), (32, "e8m0")], ids=["nvfp4", "mxfp4"])
def test_moe_fp4_quant_output_is_declined(bs, scale, side_output):
    """A packed fp4 output on the sm120 MoE kernels is declined at probe and build
    alike: their STG tap is an Int8 carrier at the logical (S, N), while the call
    contract is (1, S, N / 2) and graph.execute binds a float4_e2m1fn_x2 slot, so
    no buffer could pass both (block-16/32 quant once let the probe list such a
    plan, and every execute failed). A bf16 side output in slot 0 must not hide
    the fp4 one."""
    from cudnn.gemm.frost.sm120 import compiler as C

    E, S, N, K = 4, 256, 256, 128
    g = _build_graph(E, S, N, K, num_groups=E, output_dt=cudnn.data_type.FLOAT)
    _, binding = analyze_with_binding(g)
    base = binding.outputs[0]
    base.set_output(False)
    source = base
    if side_output:
        side = g.identity(input=base, name="side")
        side.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
        source = g.identity(input=base, name="quant_source").set_data_type(cudnn.data_type.FLOAT)
    q, sf = g.block_scale_quantize(input=source, block_size=bs, name="q4")
    q.set_data_type(cudnn.data_type.FP4_E2M1).set_dim([1, S, N]).set_stride([S * N, N, 1]).set_output(True)
    sf.set_dim([1, S, N // bs]).set_stride([S * (N // bs), N // bs, 1]).set_output(True)
    sf.set_data_type(cudnn.data_type.FP8_E8M0 if scale == "e8m0" else cudnn.data_type.FP8_E4M3)
    cfg = by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2")
    with pytest.raises(NotImplementedError, match="packed fp4 output"):
        C.probe_chain(analyze(g), cfg)
    with pytest.raises(NotImplementedError, match="packed fp4 output"):
        C.jit_from_cudnn_graph(g, config=cfg)


@requires_sm120
@pytest.mark.parametrize("shape", [(8, 256, 256, 128), (8, 1024, 2048, 1024), (32, 8192, 7168, 4096), (16, 2048, 1536, 2048)])
@pytest.mark.parametrize("axis", [-1, 1], ids=["feature", "token"])
def test_moe_quant_auto_config_is_buildable(shape, axis):
    """The automatic config of a block-quantized MoE graph satisfies the
    row-per-lane drain's gates and the quant accumulator budget, so the engine
    lists a plan (the warp-grid order alone picks e.g. 64x32 warps4x2, a 16x16
    warp tile the drain cannot walk, and the MoE planner has no other pick)."""
    from cudnn.gemm.frost.sm120 import compiler as C

    E, S, N, K = shape
    g = _build_graph(E, S, N, K, num_groups=E, output_dt=cudnn.data_type.FLOAT)
    _, binding = analyze_with_binding(g)
    binding.first_token_offset.set_alignment_value(32)
    q, sf = g.block_scale_quantize(input=binding.outputs[0], block_size=32, axis=axis, name="q")
    q.set_data_type(cudnn.data_type.FP8_E4M3).set_output(True)
    sf.set_data_type(cudnn.data_type.FP8_E8M0).set_output(True)
    chain = analyze(g)
    cfg = C.plan_config(chain)
    C.probe_chain(chain, cfg)
    assert chain.num_gemms * cfg.warp_tile_m * cfg.warp_tile_n <= 32 * C._ROW_PER_LANE_QUANT_ACC_REGS


@requires_sm120
@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize(
    "cfg_name",
    ["CONFIG_sm120_128x96x128_16x16x32_cluster1x1_warps4x2", "CONFIG_sm120_128x128x64_16x16x32_cluster1x1_warps4x2"],
)
def test_moe_block_scale_probe_runs_the_render_geometry_gates(cfg_name, swap):
    """probe_chain declines what the block-scale renderer declines (a CTA N that
    is neither a multiple nor a divisor of 128; a K tile other than 128 bytes),
    so the heuristics never list a plan the build drops."""
    from cudnn.gemm.frost.sm120 import compiler as C
    from test_moe_grouped_block_scale_matmul_fwd import _build_graph as build_bs

    chain = analyze(build_bs(4, 512, 256, 256, 4))
    with pytest.raises(NotImplementedError, match="block-scaled matmul on sm120 requires"):
        C.probe_chain(chain, replace(by_name(cfg_name), swap_ab=swap))


@requires_sm120
@pytest.mark.parametrize("case", ["col_compact", "row_f8_128x4"])
def test_moe_quant_refuses_runtime_tokens_beyond_the_scale_geometry(case):
    """A quant scale output keeps the shape the build declared, while the kernel
    writes the scales of every runtime token: a call whose token count needs
    another scale geometry is refused (it once wrote past the caller's scale
    buffer -- 2 KiB here, 256 KiB at S 2048 -> 4096 through graph.execute)."""
    E, N, K, bs = 4, 256, 128, 32
    if case == "col_compact":
        S0, S1 = 256, 512
        graph = _token_axis_col_quant_graph(E, S0, N, K, bs, bs)
        sf_shape = (1, S0 // bs, N)
    else:
        S0, S1 = 192, 512
        sf_shape = _quant_scale_shape(S0, N, True)
        graph = _build_graph(E, S0, N, K, num_groups=E, quant=True, quant_scale_reorder=True, quant_scale_dim=sf_shape)
    compiled = _plan(graph, config=by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"))
    token = torch.randn(1, S1, K, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    offsets = torch.tensor([0, S1 // 4, S1 // 2, 3 * S1 // 4, S1], dtype=torch.int32, device="cuda")
    q = torch.empty(1, S1, N, dtype=torch.float8_e4m3fn, device="cuda")
    sf = torch.empty(sf_shape, dtype=torch.float8_e8m0fnu, device="cuda")
    with pytest.raises(ValueError, match="scale_dim"):
        compiled(_vp_moe(compiled, token, weight, offsets, [q, sf]))


@requires_sm120
@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("combo", ["nvfp4", "mxfp8"])
def test_moe_block_scale_quant_epilogue(combo, swap):
    """A block-scale quantize on the block-scaled MoE kernels: the row-per-lane
    drain of sm120_moe_grouped_block_scale_matmul_fwd{,_swap_ab}.py (row quant
    of the features; a kernel-frame col quant after the swap)."""
    from test_moe_grouped_block_scale_matmul_fwd import _run_e2e

    _run_e2e(
        2,
        1024,
        256,
        512,
        [0, 256, 384, 512, 1024],
        combo=combo,
        config_name="CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2",
        cta_group=1,
        quant=True,
        swap_ab=swap,
    )


# --------------------------------------------------------------------------- #
# sm120 regressions: the MoE prechecks run the registry's whole funnel (C102),
# and N-major outputs whose layout admits only one-element epilogue chunks (odd
# N, odd row stride; an M-major token-contiguous output without an alignment
# promise under swap-AB) are served by both dense MoE templates (G38).
# --------------------------------------------------------------------------- #

_MOE_ODD_GUARD = 256
_MOE_ODD_DT = {
    "bf16": (cudnn.data_type.BFLOAT16, torch.bfloat16),
    "fp32": (cudnn.data_type.FLOAT, torch.float32),
    "e4m3": (cudnn.data_type.FP8_E4M3, torch.float8_e4m3fn),
}
_MOE_ODD_BOUNDS = [0, 3, 3, 35, 129, 263, 263]  # uneven + empty groups, S = 263


def _moe_odd_graph(N, out_dt, *, ld=None, major="n", align=None, E=2, K=136, bounds=_MOE_ODD_BOUNDS):
    S = bounds[-1]
    g = _build_graph(E, S, N, K, num_groups=len(bounds) - 1, output_dt=_MOE_ODD_DT[out_dt][0])
    _, binding = analyze_with_binding(g)
    if align is not None:
        binding.first_token_offset.set_alignment_value(align)
    if major == "m":
        binding.outputs[0].set_dim([1, S, N]).set_stride([N * (ld or S), 1, ld or S])
    elif ld is not None:
        binding.outputs[0].set_dim([1, S, N]).set_stride([S * ld, ld, 1])
    return g


def _moe_odd_buffer(S, N, dtype, ld=None, major="n"):
    """(raw, body, view): rows ``ld`` apart (columns for M-major) inside a 0xA5 canary."""
    inner, outer = (N, S) if major == "n" else (S, N)
    ld = ld or inner
    nbytes = outer * ld * torch.empty((), dtype=dtype).element_size()
    raw = torch.full((nbytes + 2 * _MOE_ODD_GUARD,), 0xA5, dtype=torch.uint8, device="cuda")
    body = raw[_MOE_ODD_GUARD : _MOE_ODD_GUARD + nbytes].view(dtype).view(1, outer, ld)
    view = body[:, :, :inner]
    return raw, body, (view if major == "n" else view.transpose(1, 2)), inner


_MOE_ODD_CASES = {
    # id -> (N, output dtype, ld, major, align, swap-AB)
    "N127": (127, "bf16", None, "n", None, False),
    "N255_fp32": (255, "fp32", None, "n", None, False),
    "N256_ld257": (256, "bf16", 257, "n", None, False),
    "N129_ld131_e4m3": (129, "e4m3", 131, "n", None, False),
    "swapab_N255": (255, "bf16", None, "n", 1, True),
    "swapab_Mmajor_align1": (272, "bf16", 264, "m", 1, True),
    "swapab_Mmajor_align1_ld263_fp32": (272, "fp32", 263, "m", 1, True),
}


@requires_sm120
@pytest.mark.parametrize("cfg_name", ["CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2", "CONFIG_sm120_128x48x128_16x16x32_cluster1x1_warps8x1"])
@pytest.mark.parametrize("case", list(_MOE_ODD_CASES))
def test_sm120_moe_one_element_epilogue_chunks(case, cfg_name):
    """Both dense MoE templates drain one-element N-major chunks bit-exactly with
    every canary (row padding, guard bytes) intact; the plan passes the registry
    funnel too, which the MoE prechecks now run."""
    from cudnn.gemm.frost import kernel_registry as R
    from cudnn.gemm.frost.dtypes import DTYPE_BYTES
    from cudnn.gemm.frost.sm120 import compiler as C

    N, out_dt, ld, major, align, swap = _MOE_ODD_CASES[case]
    S, E, K = _MOE_ODD_BOUNDS[-1], 2, 136
    cfg = replace(by_name(cfg_name), swap_ab=swap)
    g = _moe_odd_graph(N, out_dt, ld=ld, major=major, align=align)
    compiled = C.jit_from_cudnn_graph(g, config=cfg)
    assert compiled.vec_bytes_epi == DTYPE_BYTES[compiled.chain.output_dtype], compiled.vec_bytes_epi  # one element
    # The kernel frame: a token-contiguous (M-major) output is N-major after the
    # swap -- the layout the pair rule declined; an N-major one scatters there.
    assert compiled.chain.out_major == ("m" if swap and major == "n" else "n")
    assert R.select_template(compiled.chain, cfg).accepts(compiled.chain, cfg) is None
    torch.manual_seed(31)
    token = torch.randint(-2, 3, (1, S, K), device="cuda").to(torch.bfloat16)
    weight = torch.randint(-2, 3, (E, N, K), device="cuda").to(torch.bfloat16)
    offsets = torch.tensor(_MOE_ODD_BOUNDS, dtype=torch.int32, device="cuda")
    tdt = _MOE_ODD_DT[out_dt][1]
    raw, body, out, inner = _moe_odd_buffer(S, N, tdt, ld=ld, major=major)
    compiled(_vp_moe(compiled, token, weight, offsets, out), workspace=torch.empty(compiled.workspace_bytes, dtype=torch.uint8, device="cuda"))
    torch.cuda.synchronize()
    ref = _ref_f32(token, weight, offsets, S, N, E).to(tdt)
    torch.testing.assert_close(out[0].float(), ref.float(), atol=0, rtol=0)
    assert bool((raw[:_MOE_ODD_GUARD] == 0xA5).all()) and bool((raw[-_MOE_ODD_GUARD:] == 0xA5).all()), "a store ran past the output"
    if body.shape[-1] > inner:
        assert bool((body[:, :, inner:].contiguous().view(torch.uint8) == 0xA5).all()), "a store wrote into the row padding"


@requires_sm120
@pytest.mark.parametrize("case", ["N127", "N129_ld131_e4m3", "swapab_Mmajor_align1"])
def test_sm120_moe_probe_and_funnel_agree(case):
    """probe_chain (check_support / heuristics / build) and kernel_registry.accepts()
    (candidates) agree on every sm120 catalog point of both orientations: the
    MoE prechecks end in the registry's whole funnel, as the parent's MoE
    prechecks run accepts() (they used to skip the template's own scope, so the
    funnel declined odd-N plans the probe listed)."""
    from cudnn.gemm.frost import kernel_registry as R
    from cudnn.gemm.frost.fusion_ir import swap_ab
    from cudnn.gemm.frost.sm120 import compiler as C
    from cudnn.gemm.frost.tile_config import CATALOG

    N, out_dt, ld, major, align, _swap = _MOE_ODD_CASES[case]
    chain = analyze(_moe_odd_graph(N, out_dt, ld=ld, major=major, align=align))
    sm120 = [c for c in CATALOG if c.pipeline == "sm120"]
    listed = R.candidates(chain, sweep_swap_ab=True)
    assert {c.swap_ab for _t, c in listed if c.pipeline == "sm120"} == {False, True}
    disagree = []
    for cfg in sm120 + [replace(c, swap_ab=True) for c in sm120]:
        lowered = swap_ab(chain) if cfg.swap_ab else chain
        funnel = R.select_template(lowered, cfg).accepts(lowered, cfg)
        try:
            C.probe_chain(chain, cfg)
            probe = None
        except (NotImplementedError, ValueError) as e:
            probe = str(e)
        if (funnel is None) != (probe is None):
            disagree.append((cfg.name, cfg.swap_ab, funnel, probe))
    assert not disagree, disagree[:3]


@requires_sm120
@pytest.mark.parametrize("swap", [False, True])
def test_sm120_moe_precheck_consults_the_template_scope(swap, monkeypatch):
    """A rule in Sm120KernelTemplate._extra_reject (the template's own scope) gates
    the MoE probe and build, not only the registry funnel: both run it."""
    from cudnn.gemm.frost import kernel_registry as R
    from cudnn.gemm.frost.sm120 import compiler as C

    def scoped(self, chain, config):
        return "sentinel template-scope rule" if chain.has_moe else None

    monkeypatch.setattr(R.Sm120KernelTemplate, "_extra_reject", scoped)
    g = _moe_odd_graph(256, "bf16")
    cfg = replace(by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"), swap_ab=swap)
    with pytest.raises(NotImplementedError, match="sentinel template-scope rule"):
        C.probe_chain(analyze(g), cfg)
    with pytest.raises(NotImplementedError, match="sentinel template-scope rule"):
        C.jit_from_cudnn_graph(g, config=cfg)


def _moe_quant_parity_graph(case):
    """A block-quantized MoE graph: fp8 row quant (``fp8_row32``) or an fp4 quant
    behind a bf16 side output, the fp4 data in slot 1 (``fp4_behind_bf16``)."""
    E, S, N, K = 4, 256, 256, 128
    if case == "fp8_row32":
        return _build_graph(E, S, N, K, num_groups=E, quant=True, quant_scale_dim=(1, S, N // 32))
    g = _build_graph(E, S, N, K, num_groups=E, output_dt=cudnn.data_type.FLOAT)
    _, binding = analyze_with_binding(g)
    base = binding.outputs[0]
    base.set_output(False)
    side = g.identity(input=base, name="side")
    side.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    source = g.identity(input=base, name="quant_source").set_data_type(cudnn.data_type.FLOAT)
    q, sf = g.block_scale_quantize(input=source, block_size=16, name="q4")
    q.set_data_type(cudnn.data_type.FP4_E2M1).set_dim([1, S, N]).set_stride([S * N, N, 1]).set_output(True)
    sf.set_dim([1, S, N // 16]).set_stride([S * (N // 16), N // 16, 1]).set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    return g


@requires_sm120
@pytest.mark.parametrize("case", ["fp8_row32", "fp4_behind_bf16"])
def test_sm120_moe_quant_probe_and_funnel_agree(case):
    """probe_chain and kernel_registry.accepts() / candidates() agree on every sm120
    catalog point (both orientations) for block-quantized MoE chains: the funnel
    asks the drain gates the MoE precheck runs, and declines a packed fp4 output
    in ANY slot, as _check_moe_output_dtypes does (it used to look at slot 0
    only, so a bf16 side output in slot 0 got 225 sm120 plans listed that the
    probe declined)."""
    from cudnn.gemm.frost import kernel_registry as R
    from cudnn.gemm.frost.fusion_ir import swap_ab
    from cudnn.gemm.frost.sm120 import compiler as C
    from cudnn.gemm.frost.tile_config import CATALOG

    chain = analyze(_moe_quant_parity_graph(case))
    sm120 = [c for c in CATALOG if c.pipeline == "sm120"]
    disagree, probed = [], 0
    for cfg in sm120 + [replace(c, swap_ab=True) for c in sm120]:
        try:
            lowered = swap_ab(chain) if cfg.swap_ab else chain
            funnel = R.select_template(lowered, cfg).accepts(lowered, cfg)
        except (NotImplementedError, ValueError) as e:  # no swap-AB lowering (candidates() skips it too)
            funnel = f"swap-AB lowering: {e}"
        try:
            C.probe_chain(chain, cfg)
            probe = None
            probed += 1
        except (NotImplementedError, ValueError) as e:
            probe = str(e)
        if (funnel is None) != (probe is None):
            disagree.append((cfg.name, cfg.swap_ab, funnel, probe))
    assert not disagree, disagree[:3]
    listed = [c for _t, c in R.candidates(chain, sweep_swap_ab=True) if c.pipeline == "sm120"]
    if case == "fp4_behind_bf16":
        assert probed == 0 and not listed
    else:
        assert probed > 0 and listed
