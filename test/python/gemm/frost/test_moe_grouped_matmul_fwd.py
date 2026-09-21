# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoE grouped matmul forward (mode=NONE): analyzer detection + end-to-end
correctness vs a torch group-loop reference (uneven + empty groups)."""

from __future__ import annotations

import pathlib

import cudnn
import cudnn.gemm.frost  # noqa: F401  (installs hook)
import cutlass
from dataclasses import replace

import pytest
import torch

from gemm_test_utils import (
    requires_sm100,
    requires_sm120,
    Plan as _plan,
    ceil_div as _ceil_div,
    to_blocked as _to_blocked,
    block_quant_ref as _block_quant_ref,
    reduction_ref as _reduction_ref,
    reduction_dims as _reduction_dims,
    FULL_EXPERT_REDUCE_OFFSETS as _FULL_EXPERT_REDUCE_OFFSETS,
)

from cudnn.gemm.frost.graph_analyzer import analyze
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


_CFG = "CONFIG_sm100_128x256x128_128x256x32_cluster2x1"
# (config name, cta_group): 2-CTA cluster2x1 (reference) + 1-CTA cluster1x1.
_GEOMETRIES = [
    ("CONFIG_sm100_128x256x128_128x256x32_cluster2x1", 2),
    ("CONFIG_sm100_128x256x128_128x256x32_cluster1x1", 1),
    # CTA tiles split across several MMA instructions along M (mma_size_m).
    ("CONFIG_sm100_256x256x128_128x256x32_cluster2x1", 2),  # mma_size_m=2 on the pair
    ("CONFIG_sm100_256x128x128_128x128x32_cluster1x1", 1),  # mma_size_m=2
    ("CONFIG_sm100_128x128x128_64x128x32_cluster1x1", 1),  # mma_size_m=2 at mma_inst_m=64
    ("CONFIG_sm100_512x128x128_128x128x32_cluster1x1", 1),  # mma_size_m=4, two A TMA boxes
    ("CONFIG_sm100_512x128x128_128x128x32_cluster2x1", 2),
    ("CONFIG_sm100_128x256x128_128x256x32_cluster2x2", 1),
    ("CONFIG_sm100_128x256x128_128x256x32_cluster4x2", 2),
]

# The plain-e2e test additionally covers N-tiles that are not a multiple of 32
# (pow2 epilogue subtile spans). Kept OUT of _GEOMETRIES: the N-major-weight
# and block-quant tests legitimately reject these tiles (swizzle-group /
# vsize-divisibility gates).
_GEOMETRIES_STEP8 = _GEOMETRIES + [
    ("CONFIG_sm100_128x40x128_128x40x32_cluster1x1", 1),
    ("CONFIG_sm100_128x144x128_128x144x32_cluster2x1", 2),
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
        dim=[fto_groups, 1, 1],
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


def test_analyzer_rejects_gather() -> None:
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
        dim=[E, 1, 1],
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
    with pytest.raises(NotImplementedError, match="mode=NONE"):
        analyze(g)


# --------------------------------------------------------------------------- #
# End-to-end correctness (GPU)
# --------------------------------------------------------------------------- #


def _offsets(group_sizes, S, dtype=torch.int32):
    starts, cur = [], 0
    for gs in group_sizes:
        starts.append(cur)
        cur += gs
    assert cur == S
    return torch.tensor(starts, dtype=dtype, device="cuda")


def _ref_f32(token, weight, offsets, S, N, E):
    out = torch.zeros((S, N), dtype=torch.float32, device="cuda")
    starts = offsets.tolist()
    for g in range(len(starts)):
        b = starts[g]
        e = starts[g + 1] if g + 1 < len(starts) else S
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
        end = starts[g + 1] if g + 1 < group_count else x.shape[0]
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


def _group_sizes_from_offsets(offsets: list[int], total: int) -> list[int]:
    return [(offsets[i + 1] if i + 1 < len(offsets) else total) - offsets[i] for i in range(len(offsets))]


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


@requires_sm100
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


@requires_sm100
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


@requires_sm100
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


@requires_sm100
def test_moe_grouped_matmul_fwd_n_major_weight_more_groups_than_experts() -> None:
    E, N, K = 4, 256, 128
    group_sizes = [64, 0, 128, 64, 96, 32, 160, 96, 128]
    S = sum(group_sizes)
    cfg = by_name(_CFG)
    compiled = _plan(
        _build_graph(E, S, N, K, num_groups=len(group_sizes), weight_major="n"),
        config=cfg,
        cta_group=2,
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


@requires_sm100
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
    assert f"tma_c_m_major = ({major == 'm'},)" in _tma_c_plumbing(chain)["INJECT_TMA_C_LISTS"]


def test_moe_dim0_descriptor_patch_ir() -> None:
    """Keep the dim-0 update below the public NVVM ordinal verifier, with
    memory side effects so the following descriptor copy cannot pass it."""
    import cutlass.cute as cute
    from cudnn.gemm.frost.sm100.kernel_templates._tile_helpers import replace_tensormap_global_dim_0

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


@requires_sm100
@pytest.mark.parametrize("cta_group", (1, 2))
@pytest.mark.parametrize(
    "offset_multiple,bounds,S,store_mode,global_descriptors",
    [
        (1, [0, 100, 300], 512, "stg", False),
        (4, [0, 104, 304], 512, "stg", False),
        (8, [0, 104, 304], 512, "tma", False),
        (256, [0, 256, 256], 512, "tma", True),
        (8, [0, 104, 304], 510, "stg", False),
        (256, [0, 256, 256], 510, "stg", False),
        (256, [0, 256, 256], 504, "tma", False),
    ],
)
@pytest.mark.parametrize("tap", (False, True))
def test_moe_m_major_output(cta_group: int, offset_multiple: int, bounds: list[int], S: int, store_mode: str, global_descriptors: bool, tap: bool) -> None:
    """TMA needs a promise of 16-byte group boundaries, independently for each
    output layout. Partial groups must clip on M even beside an N-major tap."""
    from cudnn.gemm.frost.compiler import _moe_aligned_offsets, _store_modes, jit_from_cudnn_graph
    from cudnn.gemm.frost.graph_analyzer import analyze
    from cudnn.gemm.frost.tile_config import by_name

    N, K, E = 256, 128, 3
    ldm = 512  # Keep the column stride aligned even when the endpoint S is not.
    BF = cudnn.data_type.BFLOAT16
    cfg = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma" if cta_group == 1 else "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma")

    def build():
        g = cudnn.pygraph(io_data_type=BF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=BF)
        w = g.tensor(name="weight", dim=[E, K, N], stride=[K * N, 1, K], data_type=BF)
        fto = g.tensor(name="first_token_offset", dim=[E, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
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
    assert _store_modes(chain, cfg) == (("tma", store_mode) if tap else (store_mode,))
    assert _moe_aligned_offsets(chain, cfg) is global_descriptors

    torch.manual_seed(0)
    tk = torch.randn(1, S, K, device="cuda", dtype=torch.bfloat16)
    wt = torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16)
    ft = torch.tensor(bounds, device="cuda", dtype=torch.int32)
    slack = 4096
    raw = torch.full((2 * ldm * N + slack,), 0xAB, device="cuda", dtype=torch.uint8)
    storage = raw[: 2 * ldm * N].view(1, N, 2 * ldm)
    out = storage.view(torch.bfloat16).transpose(1, 2)[:, :S, :]
    out.zero_()
    tail = raw[2 * ldm * N :].clone()
    compiled = jit_from_cudnn_graph(g, config=cfg)
    assert f"moe_aligned_offsets = {global_descriptors}\n" in pathlib.Path(compiled.generated_path).read_text()
    tap_out = torch.full((1, S, N), float("nan"), device="cuda", dtype=torch.float32) if tap else None
    compiled(_vp_moe(compiled, tk, wt, ft, [tap_out, out] if tap else out))
    torch.cuda.synchronize()

    ref = torch.zeros(1, S, N, device="cuda", dtype=torch.float32)
    b = bounds + [S]
    for gi in range(E):
        ref[0, b[gi] : b[gi + 1]] = tk[0, b[gi] : b[gi + 1]].float() @ wt[gi].float().T
    if tap:
        torch.testing.assert_close(tap_out, ref, atol=1e-4, rtol=1e-4)
        ref = ref.relu()
    assert torch.equal(raw[2 * ldm * N :], tail), "the store ran past the output"
    assert (storage[:, :, 2 * S :] == 0xAB).all(), "the store overwrote column padding"
    assert (out.float() - ref).abs().max().item() < 0.5


@requires_sm100
@pytest.mark.parametrize(
    "offset_multiple,S,store_modes",
    [
        (1, 768, ("stg", "stg", "stg")),
        (4, 768, ("tma", "stg", "stg")),
        (8, 768, ("tma", "tma", "stg")),
        (12, 768, ("tma", "stg", "stg")),
        (16, 768, ("tma", "tma", "tma")),
        (24, 768, ("tma", "tma", "stg")),
        (16, 772, ("tma", "stg", "stg")),
        (16, 776, ("tma", "tma", "stg")),
        (16, 767, ("stg", "stg", "stg")),
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
    offsets = torch.tensor([0, 96, 96, 288], dtype=torch.int32, device="cuda")
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
    fto = g.tensor(name="first_token_offset", dim=[E, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    out = g.moe_grouped_matmul(tok, w, fto, mode=cudnn.moe_grouped_matmul_mode.NONE)
    out.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

    with pytest.raises(NotImplementedError, match="K-major token"):
        jit_from_cudnn_graph(g, config=by_name(_CFG))


@requires_sm100
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

    ref = _ref_f32(token, weight, offsets, S, N, E)
    q_ref, scale_ref = _block_quant_ref(ref, 32, out_torch_dt, scale_torch_dt)
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


@requires_sm100
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


@requires_sm100
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
        2,
        mode,
        red_dims,
        red_stride=red_stride,
        integer_inputs=True,
    )


@requires_sm100
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


@requires_sm100
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


@requires_sm100
def test_moe_grouped_matmul_fwd_group_reduction_full_expert_amax_fp32() -> None:
    group_sizes = _group_sizes_from_offsets(_FULL_EXPERT_REDUCE_OFFSETS, 2000)
    _run_moe_reduction(
        _CFG,
        2,
        cudnn.reduction_mode.AMAX,
        [36, 1, 1],
        E=9,
        N=248,
        K=520,
        group_sizes=group_sizes,
        group_reduction=True,
    )


@requires_sm100
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
        2,
        mode,
        [4, 1, 128],
        group_sizes=[32, 96, 0, 128],
        group_reduction=True,
        integer_inputs=True,
    )


@requires_sm100
@pytest.mark.parametrize(
    "cfg_name,cta_group,mode",
    [
        ("CONFIG_sm100_128x256x128_128x256x32_cluster2x1", 2, "padded"),
        ("CONFIG_sm100_128x256x128_128x256x32_cluster1x1", 1, "padded"),
        ("CONFIG_sm100_128x256x128_128x256x32_cluster1x1", 1, "zero_stride"),
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


@requires_sm100
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

    # num_experts/num_groups are derived from weight.shape[0] /
    # first_token_offset.shape[0] inside the call.
    compiled(_vp_moe(compiled, token, weight, offsets, output))
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output[0],
        _ref_f32(token, weight, offsets, S, N, E).to(torch.bfloat16),
        atol=2e-1,
        rtol=5e-2,
    )


# INT8 × INT8 → INT32 MoE (integer tensor-core MMA) — the MoE pipeline follows
# the plain-matmul combos exactly; int8's GPU support comes from the shared
# MMA_GPU_ARCH_SPECIAL_CASES entry (SM 100 / SM 110 only).


@requires_sm100
@pytest.mark.parametrize("cta_group", [1, 2])
def test_moe_int8(cta_group):
    from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
    from cudnn.gemm.frost.kernel_registry import MMA_GPU_ARCH_SPECIAL_CASES
    from gemm_test_utils import _active_sm

    sm = _active_sm()
    ranges = MMA_GPU_ARCH_SPECIAL_CASES[("sm100", ("int8", "int8", "int32"))]
    if sm is not None and not any(lo <= sm < hi for lo, hi in ranges):
        pytest.skip(f"int8 MMA unsupported on sm_{sm} (SM 100/110 only)")

    E, S, N, K = 4, 512, 256, 512
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.INT8,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.INT32,
    )
    tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=cudnn.data_type.INT8)
    w = g.tensor(name="weight", dim=[E, K, N], stride=[K * N, 1, K], data_type=cudnn.data_type.INT8)
    fto = g.tensor(name="first_token_offset", dim=[E, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    out = g.moe_grouped_matmul(tok, w, fto, mode=cudnn.moe_grouped_matmul_mode.NONE)
    out.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    cfg_name = "CONFIG_sm100_128x128x128_128x128x32_cluster1x1" if cta_group == 1 else "CONFIG_sm100_128x128x128_128x128x32_cluster2x1"
    compiled = jit_from_cudnn_graph(g, config=by_name(cfg_name))
    assert compiled.chain.matmul.accum_dtype == "int32"

    torch.manual_seed(0)
    a = torch.randint(-8, 8, (1, S, K), dtype=torch.int8, device="cuda")
    b = torch.randint(-8, 8, (E, N, K), dtype=torch.int8, device="cuda")
    fto_t = torch.tensor([0, 128, 200, 384], dtype=torch.int32, device="cuda")
    outb = torch.zeros(1, S, N, dtype=torch.bfloat16, device="cuda")
    from gemm_test_utils import graph_binding

    bd = graph_binding(compiled)
    compiled({bd.a_operands[0]: a, bd.b_operands[0]: b, bd.first_token_offset: fto_t, bd.outputs[0]: outb})
    torch.cuda.synchronize()

    ref = torch.zeros(1, S, N, dtype=torch.float32, device="cuda")
    bounds = fto_t.tolist() + [S]
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


@requires_sm100
@pytest.mark.parametrize("force_stg", [False, True], ids=["tmastg", "stg"])
def test_moe_host_signature_matches_the_launch_order(force_stg: bool) -> None:
    """Pin the rendered `_host` parameter list the four launchers feed: every
    dense output is a `c_tap_<i>`, except the slot-0 output under TMA-store,
    which becomes the trailing `c_<i>` the TMA-C marker injects."""
    import re

    from cudnn.gemm.frost.compiler import _moe_launch_tail

    g = _build_graph(E=8, S=2048, N=256, K=256)
    plan = _plan(g, config=by_name("CONFIG_sm100_128x256x128_128x256x32_cluster2x1"), cta_group=2, force_stg_epi=force_stg)
    tma = plan._compiled.use_tma_store
    assert tma is not force_stg

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


# --- sm120 (consumer Blackwell, warp-scoped MMA) ------------------------------
#
# The sm120 MoE template is the dense sm120 kernel's mainloop + STG epilogue
# under the sm100 MoE kernel's grouped persistent scheduler (atomic tile
# counter + warp prefix scan over first_token_offset). A is addressed by
# coordinate on ONE global descriptor -- no per-group tensormap replacement --
# so the ragged tail rows a tile loads past group_end are masked at the store.
# These tests mirror the sm100 e2e coverage above on the sm120 geometries.

_SM120_GEOMETRIES = [
    "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2",  # the flagship grid
    "CONFIG_sm120_64x256x128_16x16x32_cluster1x1_warps2x4",  # short-M tiles: every group is a ragged tail
    "CONFIG_sm120_256x128x64_16x16x32_cluster1x1_warps8x1",  # tall tile, narrow K row (s64b swizzle)
]
_SM120_CFG = _SM120_GEOMETRIES[0]


def test_sm120_moe_template_is_registered_in_the_sm120_tree() -> None:
    """Registry wiring: one MoE template of the sm120 family, rendered by the
    sm120 tree only, and the auto path targets it on an SM 12.x part."""
    import cudnn.gemm.frost.compiler as C
    from cudnn.gemm.frost.kernel_registry import GraphType, TEMPLATES, Sm120KernelTemplate, preferred_pipeline, select_template
    from cudnn.gemm.frost.sm100 import compiler as C100

    (tmpl,) = [t for t in TEMPLATES if t.pipeline == "sm120" and t.graph_type is GraphType.MOE]
    assert tmpl.file == "sm120_moe_grouped_matmul_fwd.py" and tmpl.family == "sm120"
    assert isinstance(tmpl, Sm120KernelTemplate) and not tmpl.supports_multi_gemm
    assert tmpl.path.is_file()

    chain = analyze(_build_graph(8, 768, 256, 128))
    cfg = by_name(_SM120_CFG)
    assert select_template(chain, cfg) is tmpl
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("CUDNN_FRONTEND_GEMM_ARCH_FAMILY", raising=False)
        mp.setattr(C, "_current_arch", lambda: 120)
        assert preferred_pipeline(chain) == "sm120"
        mp.setattr(C, "_current_arch", lambda: 100)
        assert preferred_pipeline(chain) == "sm100"  # the tcgen05 MoE kernel keeps SM 10.x
    # The sm100 tree has no renderer for it.
    with pytest.raises(NotImplementedError, match="served by the sm120 arch tree"):
        C100._render_tile_constants(cfg, chain, tmpl)


@pytest.mark.parametrize("weight_major", ["k", "n"])
def test_sm120_moe_render_smoke(weight_major: str) -> None:
    """Render the sm120 MoE template end-to-end (tile constants + epilogue
    snippets, no cute.compile) through the sm120 tree by name, so this covers
    every lane. Marker-free, parseable, and carrying the grouped-scheduler
    constants the dense sm120 kernel does not have."""
    import ast
    import re

    from cudnn.gemm.frost.dtypes import DTYPE_BYTES
    from cudnn.gemm.frost.sm120 import compiler as C120
    from cudnn.gemm.frost.sm120.epilogue_codegen import generate

    chain = analyze(_build_graph(8, 768, 256, 128, weight_major=weight_major))
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
    assert params == ["problem_size", "first_token_offset", "a_tma_workspace", "a_0", "b_0", "c_tap_0", "stream"], params


@requires_sm120
@pytest.mark.parametrize("cfg_name", _SM120_GEOMETRIES, ids=lambda n: n.removeprefix("CONFIG_sm120_"))
@pytest.mark.parametrize("offset_cudnn_dt,offset_torch_dt", _OFFSET_DTYPES)
@pytest.mark.parametrize(
    "group_sizes",
    [
        [64, 0, 200, 128, 100, 12, 196, 68],  # uneven + one empty group
        [96, 96, 96, 96, 96, 96, 96, 96],  # balanced
        [768, 0, 0, 0, 0, 0, 0, 0],  # all tokens in group 0
    ],
)
def test_moe_grouped_matmul_fwd_e2e_sm120(group_sizes, offset_cudnn_dt, offset_torch_dt, cfg_name) -> None:
    E, N, K = 8, 256, 128
    S = sum(group_sizes)
    compiled = _plan(_build_graph(E, S, N, K, offset_dt=offset_cudnn_dt), config=by_name(cfg_name))
    assert compiled.workspace_bytes == 128  # the scheduler counter slot, no descriptor scratch

    torch.manual_seed(0)
    token = torch.randn(1, S, K, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    # NaN canary: a row the kernel never stores can only belong to an EMPTY group.
    output = torch.full((1, S, N), float("nan"), dtype=torch.bfloat16, device="cuda")
    offsets = _offsets(group_sizes, S, dtype=offset_torch_dt)

    compiled(_vp_moe(compiled, token, weight, offsets, output))
    torch.cuda.synchronize()
    assert not torch.isnan(output).any()
    torch.testing.assert_close(output[0], _ref_f32(token, weight, offsets, S, N, E).to(torch.bfloat16), atol=1e-1, rtol=1e-2)


@requires_sm120
@pytest.mark.parametrize("cfg_name", _SM120_GEOMETRIES, ids=lambda n: n.removeprefix("CONFIG_sm120_"))
@pytest.mark.parametrize("group_sizes", [[64, 0, 200, 128, 100, 12, 196, 68], [768, 0, 0, 0, 0, 0, 0, 0]])
def test_moe_grouped_matmul_fwd_e2e_weight_n_major_sm120(group_sizes, cfg_name) -> None:
    """An N-major weight rides the transposing ldmatrix; bit-identical to the K-major run."""
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
    compiled_n = _plan(_build_graph(E, S, N, K, weight_major="n"), config=cfg)
    assert compiled_n.chain.matmul.b_major == "n"
    compiled_n(_vp_moe(compiled_n, token, weight_n, offsets, out_n))

    out_k = torch.zeros(1, S, N, dtype=torch.bfloat16, device="cuda")
    compiled_k = _plan(_build_graph(E, S, N, K), config=cfg)
    compiled_k(_vp_moe(compiled_k, token, weight_k, offsets, out_k))
    torch.cuda.synchronize()

    torch.testing.assert_close(out_n[0], ref, atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(out_n, out_k, atol=0, rtol=0)


@requires_sm120
@pytest.mark.parametrize("cfg_name", _SM120_GEOMETRIES, ids=lambda n: n.removeprefix("CONFIG_sm120_"))
def test_moe_grouped_matmul_fwd_bxe_gt_e_sm120(cfg_name) -> None:
    """num_groups (BxE) > num_experts (E): expert = group % E, visited expert-major."""
    S, N, K, E = 2000, 248, 520, 9
    compiled = _plan(_build_graph(E, S, N, K, num_groups=len(_FULL_EXPERT_REDUCE_OFFSETS)), config=by_name(cfg_name))

    torch.manual_seed(0)
    token = torch.randn(1, S, K, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    output = torch.zeros(1, S, N, dtype=torch.bfloat16, device="cuda")
    offsets = torch.tensor(_FULL_EXPERT_REDUCE_OFFSETS, dtype=torch.int32, device="cuda")

    compiled(_vp_moe(compiled, token, weight, offsets, output))
    torch.cuda.synchronize()
    torch.testing.assert_close(output[0], _ref_f32(token, weight, offsets, S, N, E).to(torch.bfloat16), atol=2e-1, rtol=5e-2)


@requires_sm120
@pytest.mark.parametrize("mode", ["padded", "zero_stride"])
def test_moe_grouped_matmul_fwd_nonpacked_tensors_sm120(mode) -> None:
    group_sizes = [64, 0, 200, 128, 100, 12, 196, 68]
    E, N, K = 8, 256, 128
    S = sum(group_sizes)
    compiled = _plan(_build_graph(E, S, N, K), config=by_name(_SM120_CFG))

    token, weight, output = _mk_nonpacked_data(S, N, K, E, mode)
    offsets = _offsets(group_sizes, S)
    assert not token.is_contiguous() or not weight.is_contiguous()
    assert not output.is_contiguous()

    compiled(_vp_moe(compiled, token, weight, offsets, output))
    torch.cuda.synchronize()
    torch.testing.assert_close(output[0], _ref_f32(token, weight, offsets, S, N, E).to(torch.bfloat16), atol=1e-1, rtol=1e-2)


@requires_sm120
@pytest.mark.parametrize("mode", [cudnn.reduction_mode.ADD, cudnn.reduction_mode.AMAX, cudnn.reduction_mode.AVG], ids=["add", "amax", "avg"])
def test_moe_grouped_matmul_fwd_reduction_scalar_fp32_sm120(mode) -> None:
    """The fused reduction epilogue is the shared codegen; on sm120 it rides the STG drain."""
    _run_moe_reduction(_SM120_CFG, None, mode, [1, 1, 1])


@requires_sm120
@pytest.mark.parametrize("cfg_name", _SM120_GEOMETRIES, ids=lambda n: n.removeprefix("CONFIG_sm120_"))
def test_moe_grouped_matmul_fwd_group_reduction_amax_scalar_fp32_sm120(cfg_name) -> None:
    """Per-group AMAX indexes the reduction output by the routed group and must
    see only the group's own rows: the ragged-tail rows a tile loads past
    group_end (the next group's tokens) are masked before the reduction."""
    _run_moe_reduction(cfg_name, None, cudnn.reduction_mode.AMAX, [4, 1, 1], group_sizes=[64, 0, 120, 72], group_reduction=True)


@requires_sm120
def test_moe_grouped_matmul_fwd_auto_config_sm120() -> None:
    """The engine's auto path on an SM 12.x part: preferred_pipeline lands on
    the sm120 MoE template, select_config hands it a legal sm120 geometry, and
    the plan runs."""
    from cudnn.gemm.frost.graph_analyzer import build_gemm_plan

    E, N, K = 4, 256, 128
    group_sizes = [64, 0, 200, 248]
    S = sum(group_sizes)
    compiled = build_gemm_plan(_build_graph(E, S, N, K))
    assert compiled.config.pipeline == "sm120", compiled.config.name

    torch.manual_seed(0)
    token = torch.randn(1, S, K, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    output = torch.zeros(1, S, N, dtype=torch.bfloat16, device="cuda")
    offsets = _offsets(group_sizes, S)

    compiled(_vp_moe(compiled, token, weight, offsets, output))
    torch.cuda.synchronize()
    torch.testing.assert_close(output[0], _ref_f32(token, weight, offsets, S, N, E).to(torch.bfloat16), atol=1e-1, rtol=1e-2)


@requires_sm120
def test_moe_int8_sm120():
    """INT8 x INT8 -> INT32 on the warp MMA (m16n8k32.s8): small-magnitude
    integer products are exact in bf16, so the check is bit-exact."""
    from cudnn.gemm.frost.compiler import jit_from_cudnn_graph

    E, S, N, K = 4, 512, 256, 512
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.INT8,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.INT32,
    )
    tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=cudnn.data_type.INT8)
    w = g.tensor(name="weight", dim=[E, K, N], stride=[K * N, 1, K], data_type=cudnn.data_type.INT8)
    fto = g.tensor(name="first_token_offset", dim=[E, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    out = g.moe_grouped_matmul(tok, w, fto, mode=cudnn.moe_grouped_matmul_mode.NONE)
    out.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    compiled = jit_from_cudnn_graph(g, config=by_name(_SM120_CFG))
    assert compiled.chain.matmul.accum_dtype == "int32"

    torch.manual_seed(0)
    a = torch.randint(-8, 8, (1, S, K), dtype=torch.int8, device="cuda")
    b = torch.randint(-8, 8, (E, N, K), dtype=torch.int8, device="cuda")
    fto_t = torch.tensor([0, 128, 200, 384], dtype=torch.int32, device="cuda")
    outb = torch.zeros(1, S, N, dtype=torch.bfloat16, device="cuda")
    from gemm_test_utils import graph_binding

    bd = graph_binding(compiled)
    compiled({bd.a_operands[0]: a, bd.b_operands[0]: b, bd.first_token_offset: fto_t, bd.outputs[0]: outb})
    torch.cuda.synchronize()

    ref = torch.zeros(1, S, N, dtype=torch.float32, device="cuda")
    bounds = fto_t.tolist() + [S]
    for gi in range(E):
        lo, hi = bounds[gi], bounds[gi + 1]
        if hi > lo:
            ref[0, lo:hi] = a[0, lo:hi].float() @ b[gi].float().t()
    torch.testing.assert_close(outb, ref.to(torch.bfloat16), atol=0.0, rtol=0.0)


@requires_sm100
@pytest.mark.parametrize(
    "cta_group,cta_n,mma_m,cta_m,cluster",
    [
        (1, 32, 128, 128, "1x1"),
        (2, 32, 128, 128, "2x1"),
        (1, 64, 64, 64, "1x1"),
        (2, 64, 64, 64, "2x1"),
        (1, 40, 128, 128, "2x2"),
        (2, 96, 128, 256, "4x2"),
    ],
)
@pytest.mark.parametrize("major,aligned,force_stg", [("n", False, False), ("m", False, False), ("m", True, False), ("n", False, True)])
def test_moe_swap_ab_plain(cta_group, cta_n, mma_m, cta_m, cluster, major, aligned, force_stg):
    from dataclasses import replace
    from contextlib import nullcontext
    from cudnn.gemm.frost import fusion_ir
    from cudnn.gemm.frost.sm100 import compiler as C

    torch.manual_seed(17)
    E, S, N, K = 2, (264 if aligned else 263), 272, 136
    starts = [0, 8, 8, 32, 128] if aligned else [0, 3, 3, 35, 129]
    g = _build_graph(E, S, N, K, num_groups=len(starts), weight_major="n", offset_dt=cudnn.data_type.INT64)
    from cudnn.gemm.frost.graph_analyzer import analyze_with_binding

    _, binding = analyze_with_binding(g)
    out = binding.outputs[0]
    fto = binding.first_token_offset
    fto.set_alignment_value(8 if aligned else 1)
    ld = ((S if major == "m" else N) + 7) // 8 * 8
    out.set_dim([1, S, N]).set_stride([N * ld, 1, ld] if major == "m" else [S * ld, ld, 1])
    before = analyze(g)
    cfg = replace(by_name(f"CONFIG_sm100_{cta_m}x{cta_n}x128_{mma_m}x{cta_n}x32_cluster{cluster}"), cta_group=cta_group)

    with C.force_stg_epi() if force_stg else nullcontext():
        compiled = C.jit_from_cudnn_graph(g, config=replace(cfg, swap_ab=True))
    assert compiled.chain == fusion_ir.swap_ab(before)
    assert analyze(g) == before
    assert "sm100_moe_grouped_matmul_fwd_swap_ab" in compiled.generated_path.read_text()
    assert compiled.store_modes == (("stg",) if force_stg or (major == "m" and not aligned) else ("tma",))
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


@requires_sm100
@pytest.mark.parametrize(
    "a_dt,b_dt,cta_group,cta_n,mma_k,S,N,starts,alignment,major,global_desc",
    [
        ("HALF", "HALF", 1, 32, 32, 1, 1, [0, 0], 1, "n", False),
        ("BFLOAT16", "BFLOAT16", 2, 64, 32, 256, 256, [0, 64, 64, 128], 64, "m", True),
        ("BFLOAT16", "BFLOAT16", 2, 64, 32, 248, 256, [0, 64, 64, 128], 64, "m", False),
        ("FP8_E4M3", "FP8_E5M2", 1, 40, 32, 173, 19, [0, 3, 3, 99], 1, "n", False),
        ("FP8_E5M2", "FP8_E4M3", 2, 32, 32, 173, 256, [0, 3, 3, 99], 1, "m", False),
        ("FP8_E4M3", "FP8_E5M2", 2, 64, 64, 256, 256, [0, 64, 64, 128], 64, "m", True),
    ],
)
def test_moe_swap_ab_plain_dtypes(a_dt, b_dt, cta_group, cta_n, mma_k, S, N, starts, alignment, major, global_desc):
    from cudnn.gemm.frost.graph_analyzer import analyze_with_binding
    from cudnn.gemm.frost.sm100 import compiler as C

    if mma_k == 64 and not 107 <= C._current_arch() < 110:
        pytest.skip("plain FP8 K64 requires SM10.7–10.9")
    E, K = 2, 144
    g = _build_graph(E, S, N, K, num_groups=len(starts), output_dt=cudnn.data_type.FLOAT)
    _, binding = analyze_with_binding(g)
    binding.a_operands[0].set_data_type(getattr(cudnn.data_type, a_dt))
    binding.b_operands[0].set_data_type(getattr(cudnn.data_type, b_dt))
    binding.first_token_offset.set_alignment_value(alignment)
    ld = ((S if major == "m" else N) + 3) // 4 * 4
    binding.outputs[0].set_stride([N * ld, 1, ld] if major == "m" else [S * ld, ld, 1])
    cfg = by_name(f"CONFIG_sm100_128x{cta_n}x128_128x{cta_n}x{mma_k}_cluster{cta_group}x1_{cta_group}ctamma")
    compiled = C.jit_from_cudnn_graph(g, config=replace(cfg, swap_ab=True))
    from cudnn.gemm.frost.sm100.compiler import _moe_aligned_offsets as aligned_offsets

    assert aligned_offsets(compiled.chain, cfg) is global_desc
    types = {"HALF": torch.float16, "BFLOAT16": torch.bfloat16, "FP8_E4M3": torch.float8_e4m3fn, "FP8_E5M2": torch.float8_e5m2}
    token = torch.randint(-3, 4, (1, S, K), device="cuda").to(types[a_dt])
    weight = torch.randint(-3, 4, (E, N, K), device="cuda").to(types[b_dt])
    output = torch.empty_strided((1, S, N), (N * ld, 1, ld) if major == "m" else (S * ld, ld, 1), device="cuda", dtype=torch.float32)
    offsets = torch.tensor(starts, device="cuda", dtype=torch.int32)
    compiled(_vp_moe(compiled, token, weight, offsets, output))
    torch.cuda.synchronize()
    torch.testing.assert_close(output[0], _ref_f32(token, weight, offsets, S, N, E), atol=0, rtol=0)


@requires_sm100
@pytest.mark.parametrize("cta_group", [1, 2])
def test_moe_swap_ab_plain_swiglu(cta_group, monkeypatch):
    from functools import partial
    import test_moe_grouped_matmul_fwd_swiglu as cases

    monkeypatch.setattr(cases, "_plan", partial(_plan, swap_ab=True))
    from test_moe_grouped_matmul_fwd_swiglu import test_dual_moe_grouped_matmul_fwd_swiglu_exact_case

    cfg = f"CONFIG_sm100_128x32x128_128x32x32_cluster{cta_group}x1_{cta_group}ctamma"
    test_dual_moe_grouped_matmul_fwd_swiglu_exact_case(cfg, cta_group)


@requires_sm100
@pytest.mark.parametrize("aligned", (False, True))
def test_moe_swap_ab_plain_graph_coordinates_and_public_replay(aligned):
    from cudnn.gemm.frost.graph_analyzer import analyze_with_binding
    from cudnn.engines.manifest import MANIFEST
    from cudnn.gemm.frost.knobs import GemmKnobs

    S, N, K, G, E = (256 if aligned else 173), (256 if aligned else 136), 128, 4, 2
    bounds = [0, 8, 8, 80]
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
    cfg = by_name("CONFIG_sm100_128x32x128_128x32x32_cluster2x2_2ctamma")
    plan = _plan(graph, config=cfg, swap_ab=True)
    assert plan._compiled.store_modes == ("tma", "tma" if aligned else "stg")
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
    for group, begin in enumerate(bounds):
        end = bounds[group + 1] if group + 1 < G else S
        ref[begin:end] = K + bias_buf[group] + torch.arange(begin, end, device="cuda")[:, None] + torch.arange(N, device="cuda")[None, :]
    torch.testing.assert_close(value_buf[0], ref.to(torch.bfloat16), atol=0, rtol=0)
    assert (raw[4 * S * N :] == 0xAB).all()
    assert (raw_m[2 * ldm * N :] == 0xAB).all()
    assert (raw_m[: 2 * ldm * N].view(N, 2 * ldm)[:, 2 * S :] == 0xAB).all()


@requires_sm100
@pytest.mark.parametrize("block_scale", [False, True])
def test_moe_templates_follow_internal_operation(block_scale, monkeypatch):
    from cudnn.gemm.frost import fusion_ir, kernel_registry as R
    from cudnn.gemm.frost.sm100 import compiler as C
    from test_moe_grouped_block_scale_matmul_fwd import _build_graph as build_bs

    graph = (build_bs if block_scale else _build_graph)(2, 176, 256, 256, num_groups=4)
    chain = analyze(graph)
    swapped = fusion_ir.swap_ab(chain)
    cfg = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster2x1_2ctamma")
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


@requires_sm100
def test_moe_swap_ab_config_executes_the_same_public_graph():
    from cudnn.gemm.frost.fusion_ir import swap_ab
    from cudnn.gemm.frost.graph_analyzer import analyze_with_binding
    from cudnn.gemm.frost.sm100 import compiler as C

    E, S, N, K = 2, 173, 256, 128
    graph = _build_graph(E, S, N, K, num_groups=4)
    chain, binding = analyze_with_binding(graph)
    cfg = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster2x1_2ctamma")
    offsets = torch.tensor([0, 7, 7, 80], dtype=torch.int32, device="cuda")
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


@requires_sm100
@pytest.mark.parametrize("stg", [False, True])
@pytest.mark.parametrize("aux_major", ["n", "m"])
def test_moe_swap_ab_transposes_strided_aux(stg, aux_major):
    from cudnn.gemm.frost.graph_analyzer import analyze_with_binding
    from cudnn.gemm.frost.sm100 import compiler as C

    E, S, N, K = 2, 173, 136, 128
    graph = _build_graph(E, S, N, K, num_groups=4, output_dt=cudnn.data_type.FLOAT)
    _, binding = analyze_with_binding(graph)
    base = binding.outputs[0].set_output(False)
    stride = (S * N, N, 1) if aux_major == "n" else (S * N, 1, S)
    elem = graph.tensor(name="elem", dim=[1, S, N], stride=list(stride), data_type=cudnn.data_type.FLOAT)
    row = graph.tensor(name="token_bias", dim=[1, S, 1], stride=[2 * S, 2, 1], data_type=cudnn.data_type.FLOAT)
    value = graph.add(a=graph.add(a=base, b=elem), b=row).set_output(True).set_data_type(cudnn.data_type.FLOAT)
    config = replace(by_name("CONFIG_sm100_128x32x128_128x32x32_cluster2x1_2ctamma"), swap_ab=True)
    with C.force_stg_epi(stg):
        compiled = C.jit_from_cudnn_graph(graph, config=config)
    assert compiled.store_modes == ("stg" if stg else "tma",)
    token = torch.ones(1, S, K, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(E, N, K, device="cuda", dtype=torch.bfloat16)
    offsets = torch.tensor([0, 3, 3, 99], device="cuda", dtype=torch.int32)
    elem_buf = torch.empty_strided((1, S, N), stride, device="cuda", dtype=torch.float32)
    elem_buf.copy_(torch.arange(S * N, device="cuda").view(1, S, N))
    row_buf = torch.arange(2 * S, device="cuda", dtype=torch.float32)[::2].view(1, S, 1)
    output = torch.full((1, S, N), float("nan"), device="cuda", dtype=torch.float32)
    compiled({binding.a_operands[0]: token, binding.b_operands[0]: weight, binding.first_token_offset: offsets, elem: elem_buf, row: row_buf, value: output})
    torch.cuda.synchronize()
    torch.testing.assert_close(output, K + elem_buf + row_buf, atol=0, rtol=0)


@pytest.mark.parametrize("block_scale", [False, True])
def test_sm120_declines_moe_swap_ab_operation(block_scale, monkeypatch):
    from cudnn.gemm.frost import compiler as active_compiler
    from cudnn.gemm.frost.sm120 import compiler as C
    from test_moe_grouped_block_scale_matmul_fwd import _build_graph as build_bs

    chain = analyze((build_bs if block_scale else _build_graph)(2, 256, 256, 256, num_groups=2))
    config = replace(by_name(_SM120_CFG), swap_ab=True)
    monkeypatch.setattr(active_compiler, "_current_arch", lambda: 120)
    monkeypatch.setattr(C, "_current_arch", lambda: 120)
    with pytest.raises(ValueError, match="no kernel template.*moe_grouped.*swap_ab"):
        C.probe_chain(chain, config)


@requires_sm100
def test_moe_swap_ab_sweep_includes_geometry_rejected_by_original_graph(monkeypatch):
    from cudnn.gemm.frost import kernel_registry as R
    from cudnn.gemm.frost.graph_analyzer import analyze_with_binding
    from cudnn.gemm.frost.sm100 import compiler as C

    graph = _build_graph(2, 173, 256, 256, num_groups=4, weight_major="n")
    _, binding = analyze_with_binding(graph)
    for tensor in binding.a_operands + binding.b_operands:
        tensor.set_data_type(cudnn.data_type.FP8_E4M3)
    chain = analyze(graph)
    cfg = by_name("CONFIG_sm100_128x40x128_128x40x32_cluster1x1_1ctamma")
    monkeypatch.setattr(R, "CATALOG", (cfg,))
    assert R.candidates(chain) == []
    [(template, swapped)] = R.candidates(chain, sweep_swap_ab=True)
    assert swapped == replace(cfg, swap_ab=True)
    assert template.graph_type is R.GraphType.MOE_SWAP_AB
    C.probe_chain(chain, swapped)


@pytest.mark.parametrize("family", ["sm100", "sm120"])
@pytest.mark.parametrize("block_scale", [False, True])
@pytest.mark.parametrize("swap", [False, True])
def test_moe_norm2_is_refused_at_probe_and_jit(family, block_scale, swap):
    from importlib import import_module
    from test_moe_grouped_block_scale_matmul_fwd import _build_graph as build_bs

    compiler = import_module(f"cudnn.gemm.frost.{family}.compiler")
    graph = (build_bs if block_scale else _build_graph)(
        2,
        256,
        256,
        256,
        num_groups=4,
        reduction_mode=cudnn.reduction_mode.NORM2,
        reduction_dims=(1, 1, 256),
    )
    config = replace(by_name(_SM120_CFG if family == "sm120" else "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma"), swap_ab=swap)
    chain = analyze(graph)
    with pytest.raises(NotImplementedError, match="norm2.*square root"):
        compiler.probe_chain(chain, config)
    with pytest.raises(NotImplementedError, match="norm2.*square root"):
        compiler.jit_from_cudnn_graph(graph, config=config)


def _run_moe_swap_ab_quant(case, *, block_scale=False):
    from contextlib import nullcontext
    from cudnn.gemm.frost.fusion_ir import segmented_row_scale_capacity_rows
    from cudnn.gemm.frost.graph_analyzer import analyze_with_binding
    from cudnn.gemm.frost.sm100 import compiler as C
    from test_moe_grouped_block_scale_matmul_fwd import _build_graph as build_bs

    col = case in ("col", "col_grouped", "multi")
    grouped = case in ("row_grouped", "col_grouped")
    reorder = case not in ("row", "row_stg", "multi")
    bs = 16 if case == "packed" else 32
    S = (512 if grouped else 320) if col else 173
    starts = ([0, 128, 128, 256, 384] if grouped else [0, 32, 32, 192, 256]) if col else [0, 3, 3, 99, 151]
    E, N, K, G = 2, 160, 256, len(starts)
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
    pair = 2 if case in ("row_reorder", "col_grouped", "multi") else 1
    mma_m = 64 if case == "packed" and not block_scale else 128
    cta_n = 128 if block_scale else 64
    cfg = replace(by_name(f"CONFIG_sm100_{mma_m}x{cta_n}x128_{mma_m}x{cta_n}x32_cluster{pair}x1_{pair}ctamma"), swap_ab=True)
    with C.force_stg_epi() if case in ("row_stg", "packed") else nullcontext():
        compiled = C.jit_from_cudnn_graph(graph, config=cfg)
    assert compiled.chain.quants[0].axis == (2 if col else 1)
    assert compiled.store_modes[0] == ("stg" if case in ("row_stg", "row_mmajor", "packed") else "tma")
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
    for group, (begin, end) in enumerate(zip(starts, starts[1:] + [S])):
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
                for begin, end in (list(zip(starts, starts[1:] + [S])) if grouped else [(0, S)]):
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


@requires_sm100
@pytest.mark.parametrize("case", ["row", "row_stg", "row_mmajor", "row_reorder", "col", "col_grouped", "multi", "packed"])
def test_moe_swap_ab_quant(case):
    _run_moe_swap_ab_quant(case)


@pytest.mark.parametrize("block_scale", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
def test_moe_swap_ab_token_quant_requires_aligned_groups(block_scale, grouped):
    from cudnn.gemm.frost.graph_analyzer import analyze_with_binding
    from cudnn.gemm.frost.sm100 import compiler as C
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
    cfg = replace(by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma"), swap_ab=True)
    with pytest.raises(NotImplementedError, match="first_token_offset value alignment"):
        C.probe_chain(analyze(graph), cfg)
    binding.first_token_offset.set_alignment_value(128 if grouped else 32)
    C.probe_chain(analyze(graph), cfg)


def test_moe_swap_ab_segmented_feature_quant_requires_scale_prefix():
    from cudnn.gemm.frost.graph_analyzer import analyze_with_binding
    from cudnn.gemm.frost.sm100 import compiler as C

    graph = _build_graph(2, 173, 128, 256, num_groups=4)
    _, binding = analyze_with_binding(graph)
    base = binding.outputs[0].set_output(False)
    q, sf = graph.block_scale_quantize(input=base, block_size=32, group_offset=binding.first_token_offset)
    q.set_data_type(cudnn.data_type.FP8_E4M3).set_output(True)
    sf.set_data_type(cudnn.data_type.FP8_E8M0).set_dim([1, 640, 4]).set_stride([2560, 4, 1])
    sf.set_reordering_type(cudnn.tensor_reordering.F8_128x4).set_output(True)
    cfg = replace(by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma"), swap_ab=True)
    with pytest.raises(NotImplementedError, match="requires a block-scaled MoE"):
        C.probe_chain(analyze(graph), cfg)


def _run_moe_swap_ab_reductions(axis, *, block_scale=False, reduction_kind="basic", swap=True):
    from contextlib import nullcontext
    from cudnn.gemm.frost.graph_analyzer import analyze_with_binding
    from cudnn.gemm.frost.sm100 import compiler as C
    from test_moe_grouped_block_scale_matmul_fwd import _build_graph as build_bs

    avg = reduction_kind == "avg"
    products = reduction_kind == "products"
    red_dt = cudnn.data_type.INT32 if reduction_kind == "int32" else cudnn.data_type.FLOAT
    red_torch_dt = torch.int32 if reduction_kind == "int32" else torch.float32
    E, N, K = 2, (136 if swap else 128), 256
    aligned = axis == "group_feature"
    S = 176 if aligned else 173
    starts = [0, 8, 8, 80, 152] if aligned else [0, 3, 3, 99, 151]
    G = len(starts)
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
    pair = 2 if axis in ("token", "token_tma", "group", "group_feature") else 1
    mma_m = 64 if axis == "scalar" and not block_scale else 128
    cta_n = 128 if block_scale else 64
    cfg = by_name(f"CONFIG_sm100_{mma_m}x{cta_n}x128_{mma_m}x{cta_n}x32_cluster{pair}x1_{pair}ctamma")
    with C.force_stg_epi() if axis == "token" else nullcontext():
        compiled = C.jit_from_cudnn_graph(graph, config=replace(cfg, swap_ab=swap))
    dense_modes = () if reduction_only else ("stg" if axis == "token" else "tma",)
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


@requires_sm100
@pytest.mark.parametrize("axis", ["scalar", "feature", "token", "token_tma", "group", "group_feature"])
def test_moe_swap_ab_reductions_fp32(axis):
    _run_moe_swap_ab_reductions(axis)


@requires_sm100
@pytest.mark.parametrize("axis", ["scalar", "feature", "token", "token_tma", "group", "group_feature", "group_token"])
def test_moe_swap_ab_avg_fp32(axis):
    _run_moe_swap_ab_reductions(axis, reduction_kind="avg")


@requires_sm100
@pytest.mark.parametrize("axis", ["scalar", "feature", "token", "token_tma", "group", "group_feature", "group_token"])
@pytest.mark.parametrize("kind", ["products", "int32"])
def test_moe_swap_ab_remaining_reductions(axis, kind):
    _run_moe_swap_ab_reductions(axis, reduction_kind=kind)


@requires_sm100
@pytest.mark.parametrize("block_scale", [False, True])
def test_moe_product_reductions_fp32(block_scale):
    _run_moe_swap_ab_reductions("feature", block_scale=block_scale, reduction_kind="products", swap=False)


@requires_sm100
@pytest.mark.parametrize("mode", [cudnn.reduction_mode.ADD, cudnn.reduction_mode.AVG], ids=["add", "avg"])
def test_moe_swap_ab_reduction_runtime_token_tail(mode):
    from cudnn.gemm.frost.graph_analyzer import analyze_with_binding
    from cudnn.gemm.frost.sm100 import compiler as C

    E, S, N, K = 2, 192, 136, 128
    graph = _build_graph(E, S, N, K, num_groups=4, output_dt=cudnn.data_type.FLOAT, reduction_mode=mode, reduction_dims=(1, 1, 1))
    _, binding = analyze_with_binding(graph)
    binding.outputs[0].set_stride([S * N, 1, S])
    binding.first_token_offset.set_alignment_value(32)
    cfg = replace(by_name("CONFIG_sm100_128x64x128_128x64x32_cluster1x1_1ctamma"), swap_ab=True)
    compiled = C.jit_from_cudnn_graph(graph, config=cfg)
    assert compiled.store_modes == ("tma", "stg")
    weight = torch.ones((E, N, K), device="cuda", dtype=torch.bfloat16)
    offsets = torch.tensor([0, 32, 32, 96], device="cuda", dtype=torch.int32)
    red = torch.empty((1, 1, 1), device="cuda")
    # The declared S divides the 32-column TMA chunk; a legal runtime S need
    # only divide the compiled 8-column STG/alignment contract.
    for runtime_s in (192, 200):
        token = torch.ones((1, runtime_s, K), device="cuda", dtype=torch.bfloat16)
        out = torch.empty((1, N, runtime_s), device="cuda").transpose(1, 2)
        compiled(_vp_moe(compiled, token, weight, offsets, [out, red]))
        torch.cuda.synchronize()
        torch.testing.assert_close(out, torch.full_like(out, K), atol=0, rtol=0)
        expected = K if mode == cudnn.reduction_mode.AVG else runtime_s * N * K
        torch.testing.assert_close(red, torch.full_like(red, expected), atol=1e-4, rtol=1e-5)


@requires_sm100
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("axis", ["scalar", "feature", "token"])
def test_moe_avg_fp32(grouped, axis):
    _run_moe_reduction(
        "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
        1,
        cudnn.reduction_mode.AVG,
        [5 if grouped else 1, 173 if axis == "token" else 1, 136 if axis == "feature" else 1],
        E=2,
        N=136,
        group_sizes=[3, 0, 96, 52, 22],
        group_reduction=grouped,
        integer_inputs=True,
    )


@pytest.mark.parametrize("family", ["sm100", "sm120"])
def test_moe_reduction_initialization_validates_all_outputs_before_writing(family):
    import importlib
    from cudnn.gemm.frost.graph_analyzer import analyze_with_binding

    compiler = importlib.import_module(f"cudnn.gemm.frost.{family}.compiler")
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
