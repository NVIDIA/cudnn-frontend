# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoE grouped matmul forward (mode=NONE): analyzer detection + end-to-end
correctness vs a torch group-loop reference (uneven + empty groups)."""

from __future__ import annotations

import pathlib

import cudnn
import cudnn.gemm.frost  # noqa: F401  (installs hook)
import pytest
import torch

from gemm_test_utils import (
    requires_sm100,
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
    bd = compiled.binding
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
    # CTA tiles split across several MMA instructions along M (num_mma_m).
    ("CONFIG_sm100_256x256x128_128x256x32_cluster2x1", 2),  # num_mma_m=2 on the pair
    ("CONFIG_sm100_256x128x128_128x128x32_cluster1x1", 1),  # num_mma_m=2
    ("CONFIG_sm100_128x128x128_64x128x32_cluster1x1", 1),  # num_mma_m=2 at mma_inst_m=64
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
    if mode in (cudnn.reduction_mode.ADD, cudnn.reduction_mode.AMAX):
        out.fill_(0)
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

    cfg_k, cta_group_k = select_config(64, 32, 1)
    assert (cfg_k.cta_tile_n, cta_group_k) == (32, 1)

    cfg_n, cta_group_n = select_config(64, 32, 1, b_n_major=True)
    assert (cfg_n.cta_tile_n, cta_group_n) == (64, 1)

    cfg_2, cta_group_2 = select_config(256, 32, 1, b_n_major=True)
    assert (cfg_2.cta_tile_n, cta_group_2) == (128, 2)


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


@pytest.mark.parametrize("cta_group", (1, 2))
def test_moe_m_major_output(cta_group: int) -> None:
    """A routed MoE output may be M-major. It takes STG: TMA would clip D to
    `group_end` on the INNERMOST dim of a transposed descriptor, whose extent TMA
    requires to be 16-byte aligned, and routed group bounds are runtime data --
    a group ending at row 100 (200 B) faults, 104 (208 B) does not."""
    from cudnn.gemm.frost.compiler import _epi_vec_bytes, _store_modes, jit_from_cudnn_graph
    from cudnn.gemm.frost.graph_analyzer import analyze
    from cudnn.gemm.frost.tile_config import by_name

    S, N, K, E = 512, 256, 128, 3
    bounds = [0, 100, 300]  # neither tile- nor 16-byte-aligned
    BF = cudnn.data_type.BFLOAT16
    cfg = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1" if cta_group == 1 else "CONFIG_sm100_128x256x128_128x256x32_cluster2x1")

    def build():
        g = cudnn.pygraph(io_data_type=BF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=BF)
        w = g.tensor(name="weight", dim=[E, K, N], stride=[K * N, 1, K], data_type=BF)
        fto = g.tensor(name="first_token_offset", dim=[E, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
        out = g.moe_grouped_matmul(tok, w, fto, mode=cudnn.moe_grouped_matmul_mode.NONE, compute_data_type=cudnn.data_type.FLOAT, name="moe")
        out.set_data_type(BF).set_output(True)
        out.set_stride([S * N, 1, S])
        return g, tok, w, fto, out

    g, _, _, _, _ = build()
    chain = analyze(g)
    assert chain.out_major == "m"
    assert _store_modes(chain, cfg, cta_group) == ("stg",)

    torch.manual_seed(0)
    tk = torch.randn(1, S, K, device="cuda", dtype=torch.bfloat16)
    wt = torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16)
    ft = torch.tensor(bounds, device="cuda", dtype=torch.int32)
    slack = 4096
    raw = torch.full((2 * S * N + slack,), 0xAB, device="cuda", dtype=torch.uint8)
    out = raw.view(torch.bfloat16)[: S * N].view(1, N, S).transpose(1, 2)
    out.zero_()
    tail = raw[2 * S * N :].clone()
    gg, tt, ww, ff, oo = build()
    jit_from_cudnn_graph(gg, config=cfg, cta_group=cta_group)({tt: tk, ww: wt, ff: ft, oo: out})
    torch.cuda.synchronize()

    ref = torch.zeros(1, S, N, device="cuda", dtype=torch.float32)
    b = bounds + [S]
    for gi in range(E):
        ref[0, b[gi] : b[gi + 1]] = tk[0, b[gi] : b[gi + 1]].float() @ wt[gi].float().T
    assert torch.equal(raw[2 * S * N :], tail), "the store ran past the output"
    assert (out.float() - ref).abs().max().item() < 0.5


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
        jit_from_cudnn_graph(g, config=by_name(_CFG), cta_group=2)


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
        atol=1e-1 if red_torch_dt == torch.float32 else 0,
        rtol=1e-2 if red_torch_dt == torch.float32 else 0,
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
    compiled = jit_from_cudnn_graph(g, config=by_name(cfg_name), cta_group=cta_group)
    assert compiled.chain.matmul.accum_dtype == "int32"

    torch.manual_seed(0)
    a = torch.randint(-8, 8, (1, S, K), dtype=torch.int8, device="cuda")
    b = torch.randint(-8, 8, (E, N, K), dtype=torch.int8, device="cuda")
    fto_t = torch.tensor([0, 128, 200, 384], dtype=torch.int32, device="cuda")
    outb = torch.zeros(1, S, N, dtype=torch.bfloat16, device="cuda")
    bd = compiled.binding
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
