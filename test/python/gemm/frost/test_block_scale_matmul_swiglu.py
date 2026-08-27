# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Block-scale multi-GEMM: two block-scaled GEMMs sharing dequantized A feed
one epilogue, ``silu(dq(A) @ dq(B0)) * (dq(A) @ dq(B1))``. Dual N capped at 128
(2 × cta_n + SF must fit 512 TMEM cols)."""

import pytest
import torch

from gemm_test_utils import (
    requires_sm100,
    requires_sm107,
    Plan as _plan,
    kw as _kw,
    E2M1 as _E2M1,
    to_blocked as _to_blocked,
    unpack_fp4 as _unpack_fp4,
    rand_e8m0 as _rand_e8m0,
    reduction_ref as _reduction_ref,
    assert_block_scale_reduction_close as _assert_block_scale_reduction_close,
    block_quant_ref as _block_quant_ref,
)
import cudnn
import cudnn.gemm.frost  # noqa: F401  installs the recorder

from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
from cudnn.gemm.frost.graph_analyzer import analyze

pytestmark = pytest.mark.L0


def _vp_bs_mg(compiled, gemm_pairs, outs, *aux):
    """Block-scale multi-GEMM variant-pack dict from the binding. Each pair is
    ``((a, sfa), (b, sfb))``; dedup by packed-data identity into distinct A/B slots."""
    bd = compiled.binding
    a_seen, b_seen, sfa_seen, sfb_seen = [], [], [], []
    for (ag, sfag), (bg, sfbg) in gemm_pairs:
        if not any(ag is x for x in a_seen):
            a_seen.append(ag)
            sfa_seen.append(sfag)
        if not any(bg is x for x in b_seen):
            b_seen.append(bg)
            sfb_seen.append(sfbg)
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    vp = {}
    vp.update({t: buf for t, buf in zip(bd.a_operands, a_seen)})
    vp.update({t: buf for t, buf in zip(bd.b_operands, b_seen)})
    vp.update({t: buf for t, buf in zip(bd.sfa_operands, sfa_seen)})
    vp.update({t: buf for t, buf in zip(bd.sfb_operands, sfb_seen)})
    vp.update({o: buf for o, buf in zip(bd.outputs, outs)})
    vp.update({x: buf for x, buf in zip(bd.aux, aux)})
    return vp


# Graph builder: shared dequant(A) → two GEMMs → silu(C0) * C1
def _build_dual_bs_graph(
    M,
    N,
    K,
    *,
    combo="nvfp4",
    out_dt=cudnn.data_type.FLOAT,
    reduction_mode=None,
    red_dims=None,
    red_compute=cudnn.data_type.FLOAT,
    red_dtype=cudnn.data_type.FLOAT,
    quant_block=None,
):
    is_fp4 = combo in ("nvfp4", "mxfp4")
    bs = 16 if combo == "nvfp4" else 32
    sf_k = K // bs
    a_dt = cudnn.data_type.FP4_E2M1 if is_fp4 else cudnn.data_type.FP8_E4M3
    sf_dt = cudnn.data_type.FP8_E4M3 if combo == "nvfp4" else cudnn.data_type.FP8_E8M0
    rk = dict(reordering_type=cudnn.tensor_reordering.F8_128x4)

    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1], data_type=a_dt)
    SFA = g.tensor(name="SFA", dim=[1, M, sf_k], stride=[M * sf_k, sf_k, 1], data_type=sf_dt, **rk)
    B0 = g.tensor(name="B0", dim=[1, K, N], stride=[K * N, 1, K], data_type=a_dt)
    SFB0 = g.tensor(name="SFB0", dim=[1, sf_k, N], stride=[sf_k * N, 1, sf_k], data_type=sf_dt, **rk)
    B1 = g.tensor(name="B1", dim=[1, K, N], stride=[K * N, 1, K], data_type=a_dt)
    SFB1 = g.tensor(name="SFB1", dim=[1, sf_k, N], stride=[sf_k * N, 1, sf_k], data_type=sf_dt, **rk)

    Ad = g.block_scale_dequantize(input=A, descale=SFA, block_size=[1, bs])  # shared
    B0d = g.block_scale_dequantize(input=B0, descale=SFB0, block_size=[bs, 1])
    B1d = g.block_scale_dequantize(input=B1, descale=SFB1, block_size=[bs, 1])
    C0 = g.matmul(A=Ad, B=B0d, name="mm0")
    C1 = g.matmul(A=Ad, B=B1d, name="mm1")
    Y = g.mul(a=g.swish(input=C0), b=C1)
    if quant_block is not None:
        Q, QS = g.block_scale_quantize(input=Y, block_size=quant_block, name="q")
        Q.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
        QS.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)
    else:
        Y.set_output(True).set_data_type(out_dt)
    if reduction_mode is not None:
        if red_dims is None:
            red_dims = [1, 1, 1]
        R = g.reduction(
            input=Y,
            mode=reduction_mode,
            name="red",
            compute_data_type=red_compute,
        )
        R.set_dim(red_dims).set_stride([red_dims[1] * red_dims[2], red_dims[2], 1])
        R.set_output(True).set_data_type(red_dtype)
    return g


# Analyzer: shared-dequant dedup
def test_shared_dequant_dedup():
    chain = analyze(_build_dual_bs_graph(256, 128, 512))
    assert chain.is_multi_gemm and chain.num_gemms == 2
    # Shared dequant(A) → ONE distinct A operand.
    assert chain.num_a_operands == 1
    assert chain.num_b_operands == 2
    assert chain.gemm_operands == [(0, 0), (0, 1)]
    assert chain.has_block_scale
    assert (chain.block_scale.sf_dtype, chain.block_scale.block_size) == ("fp8_e4m3", 16)


def test_shared_dequant_reduction_detected():
    chain = analyze(
        _build_dual_bs_graph(
            256,
            128,
            512,
            reduction_mode=cudnn.reduction_mode.AMAX,
            red_dims=[1, 1, 1],
        )
    )
    assert chain.is_multi_gemm and chain.has_block_scale
    assert len(chain.reductions) == 1
    assert chain.reductions[0].mode == "amax"
    assert chain.outputs[1].source == "reduction_0"


def test_shared_dequant_quant_epilogue_detected():
    chain = analyze(_build_dual_bs_graph(256, 128, 512, quant_block=32))
    assert chain.is_multi_gemm and chain.has_block_scale
    assert len(chain.quants) == 1
    assert chain.quants[0].block_size == 32
    assert chain.quants[0].scale_dtype == "fp8_e8m0"
    assert chain.quants[0].source_ref == 1  # the mul feeding the quant
    assert chain.output_dtype == "fp8_e4m3"


# End-to-end numerics
_GPU = requires_sm100


def _dual_bs_runtime(combo, M, N, K):
    dev = "cuda"
    torch.manual_seed(0)
    is_fp4 = combo in ("nvfp4", "mxfp4")
    bs = 16 if combo == "nvfp4" else 32
    sf_k = K // bs

    if is_fp4:
        lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)

        def _mk(rows):
            u8 = torch.randint(0, 256, (1, rows, K // 2), dtype=torch.uint8, device=dev)
            return u8.view(torch.float4_e2m1fn_x2), _unpack_fp4(u8, lut).view(rows, K)

        a_rt, a_deq = _mk(M)
        b0_rt, b0_deq = _mk(N)
        b1_rt, b1_deq = _mk(N)
    else:

        def _mk(rows):
            t = (torch.randn(1, rows, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
            return t, t.float().view(rows, K)

        a_rt, a_deq = _mk(M)
        b0_rt, b0_deq = _mk(N)
        b1_rt, b1_deq = _mk(N)

    if combo == "nvfp4":
        sfa = torch.randint(1, 4, (M, sf_k), device=dev).to(torch.float8_e4m3fn)
        sfb0 = torch.randint(1, 4, (N, sf_k), device=dev).to(torch.float8_e4m3fn)
        sfb1 = torch.randint(1, 4, (N, sf_k), device=dev).to(torch.float8_e4m3fn)
    else:
        sfa, sfb0, sfb1 = (
            _rand_e8m0((M, sf_k), dev),
            _rand_e8m0((N, sf_k), dev),
            _rand_e8m0((N, sf_k), dev),
        )

    a_s = a_deq * sfa.float().repeat_interleave(bs, 1)
    b0_s = b0_deq * sfb0.float().repeat_interleave(bs, 1)
    b1_s = b1_deq * sfb1.float().repeat_interleave(bs, 1)
    C0 = a_s @ b0_s.t()
    C1 = a_s @ b1_s.t()
    ref = torch.nn.functional.silu(C0) * C1
    sfa_b = _to_blocked(sfa).view(1, M, sf_k)
    pairs = [
        ((a_rt, sfa_b), (b0_rt, _to_blocked(sfb0).view(1, N, sf_k))),
        ((a_rt, sfa_b), (b1_rt, _to_blocked(sfb1).view(1, N, sf_k))),
    ]
    return pairs, ref


def _run(combo, config_name, M, N, K):
    dev = "cuda"
    pairs, ref = _dual_bs_runtime(combo, M, N, K)

    g = _build_dual_bs_graph(M, N, K, combo=combo)
    compiled = _plan(g, **_kw(config_name))
    assert compiled.chain.is_multi_gemm and compiled.block_scale
    _bs = compiled.chain.block_scale
    assert (_bs.sf_dtype, _bs.block_size) == (("fp8_e4m3", 16) if combo == "nvfp4" else ("fp8_e8m0", 32))

    c = torch.zeros(1, M, N, dtype=torch.float32, device=dev)
    compiled(_vp_bs_mg(compiled, pairs, c))
    torch.cuda.synchronize()

    # nvfp4 matmul is exact, but swish uses fast approximate intrinsics
    # (__expf/__fdividef, ~1e-3 rel error vs torch silu); mx also has fp8/fp16 rounding.
    torch.testing.assert_close(c[0], ref, rtol=2e-2, atol=2e-1)


def _run_reduction(
    combo,
    config_name,
    M,
    N,
    K,
    mode,
    red_dims,
    ref_dims,
    *,
    red_stride=None,
):
    dev = "cuda"
    pairs, ref = _dual_bs_runtime(combo, M, N, K)
    g = _build_dual_bs_graph(
        M,
        N,
        K,
        combo=combo,
        reduction_mode=mode,
        red_dims=red_dims,
    )
    compiled = _plan(g, **_kw(config_name))
    assert compiled.chain.is_multi_gemm and compiled.block_scale
    assert compiled.chain.reductions

    c_term = torch.zeros(1, M, N, dtype=torch.float32, device=dev)
    if red_stride is None:
        c_red = torch.empty(tuple(red_dims), dtype=torch.float32, device=dev)
    else:
        c_red = torch.empty_strided(tuple(red_dims), tuple(red_stride), dtype=torch.float32, device=dev)
        assert not c_red.is_contiguous()
    compiled(_vp_bs_mg(compiled, pairs, [c_term, c_red]))
    torch.cuda.synchronize()

    torch.testing.assert_close(c_term[0], ref, rtol=2e-2, atol=2e-1)
    _assert_block_scale_reduction_close(c_red, _reduction_ref(c_term, mode, ref_dims), mode)


@_GPU
@pytest.mark.parametrize(
    "combo,config_name,M,N,K",
    [
        # 1ctamma CLC — dual nvfp4 + mxfp8. cta_n=128 → 2×128+SF fits 512.
        (
            "nvfp4",
            "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
            256,
            128,
            512,
        ),
        (
            "mxfp8",
            "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
            256,
            128,
            512,
        ),
        # A-multicast cluster (SFA mirrors A).
        (
            "nvfp4",
            "CONFIG_sm100_128x128x128_128x128x32_cluster1x4_1ctamma",
            512,
            128,
            512,
        ),
        # 2ctamma CLC (2-CTA MMA pair, cluster2x1; cta_n=128 → 2×128+SF fits 512).
        (
            "nvfp4",
            "CONFIG_sm100_128x128x128_128x128x32_cluster2x1_2ctamma",
            256,
            128,
            512,
        ),
        (
            "mxfp8",
            "CONFIG_sm100_128x128x128_128x128x32_cluster2x1_2ctamma",
            256,
            128,
            512,
        ),
        # 2ctamma B-multicast pair (cluster4x1).
        (
            "nvfp4",
            "CONFIG_sm100_128x128x128_128x128x32_cluster4x1_2ctamma",
            512,
            128,
            512,
        ),
    ],
)
def test_dual_block_scale_matmul_numerics(combo, config_name, M, N, K):
    _run(combo, config_name, M, N, K)


@_GPU
@pytest.mark.parametrize(
    "config_name",
    [
        "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
        "CONFIG_sm100_128x128x128_128x128x32_cluster2x1_2ctamma",
    ],
)
@pytest.mark.parametrize("combo", ["nvfp4", "mxfp8"])
def test_dual_block_scale_matmul_swiglu_quant_epilogue(combo, config_name):
    """Terminal block_scale_quantize on the dual block-scale SwiGLU chain: the
    fused result is re-quantized to FP8 E4M3 + per-32-block E8M0 scale."""
    dev = "cuda"
    M, N, K = 256, 128, 512
    qblock = 32
    pairs, ref = _dual_bs_runtime(combo, M, N, K)

    g = _build_dual_bs_graph(M, N, K, combo=combo, quant_block=qblock)
    compiled = _plan(g, **_kw(config_name))
    assert compiled.chain.quants

    q = torch.zeros(1, M, N, dtype=torch.float8_e4m3fn, device=dev)
    q_scale = torch.zeros(1, M, N // qblock, dtype=torch.float8_e8m0fnu, device=dev)
    compiled(_vp_bs_mg(compiled, pairs, [q, q_scale]))
    torch.cuda.synchronize()

    q_ref, scale_ref = _block_quant_ref(ref, qblock, torch.float8_e4m3fn, torch.float8_e8m0fnu)
    # The kernel's swish uses fast __expf → pre-quant values sit within ~1e-3
    # rel of torch; allow one E4M3 mantissa step where that crosses a boundary.
    torch.testing.assert_close(q_scale.float(), scale_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(q.float(), q_ref.float(), atol=2**-8, rtol=1 / 8)


@_GPU
@pytest.mark.parametrize(
    "mode",
    (
        cudnn.reduction_mode.ADD,
        cudnn.reduction_mode.AMAX,
        cudnn.reduction_mode.MAX,
        cudnn.reduction_mode.MIN,
    ),
    ids=("add", "amax", "max", "min"),
)
def test_dual_block_scale_matmul_reduction_scalar_fp32(mode):
    _run_reduction(
        "nvfp4",
        "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
        256,
        128,
        512,
        mode,
        [1, 1, 1],
        (0, 1, 2),
    )


@_GPU
@pytest.mark.parametrize(
    "combo,config_name,M,N,K",
    [
        (
            "mxfp8",
            "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
            256,
            128,
            512,
        ),
        (
            "nvfp4",
            "CONFIG_sm100_128x128x128_128x128x32_cluster1x4_1ctamma",
            512,
            128,
            512,
        ),
        (
            "nvfp4",
            "CONFIG_sm100_128x128x128_128x128x32_cluster2x1_2ctamma",
            256,
            128,
            512,
        ),
        (
            "nvfp4",
            "CONFIG_sm100_128x128x128_128x128x32_cluster4x1_2ctamma",
            512,
            128,
            512,
        ),
    ],
)
def test_dual_block_scale_matmul_reduction_configs(combo, config_name, M, N, K):
    _run_reduction(
        combo,
        config_name,
        M,
        N,
        K,
        cudnn.reduction_mode.ADD,
        [1, 1, 1],
        (0, 1, 2),
    )


@_GPU
@pytest.mark.parametrize(
    "mode,red_dims,red_stride,ref_dims",
    (
        (cudnn.reduction_mode.ADD, [1, 1, 128], [0, 0, 2], (0, 1)),
        (cudnn.reduction_mode.AMAX, [1, 256, 1], [0, 2, 1], (0, 2)),
    ),
    ids=("add_per_col", "amax_per_row"),
)
def test_dual_block_scale_matmul_reduction_strided_output(mode, red_dims, red_stride, ref_dims):
    _run_reduction(
        "nvfp4",
        "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
        256,
        128,
        512,
        mode,
        red_dims,
        ref_dims,
        red_stride=red_stride,
    )


def test_dual_block_scale_matmul_reduction_rejects_int32():
    g = _build_dual_bs_graph(
        256,
        128,
        512,
        reduction_mode=cudnn.reduction_mode.ADD,
        red_compute=cudnn.data_type.INT32,
        red_dtype=cudnn.data_type.INT32,
    )
    with pytest.raises(NotImplementedError, match="fp32 compute/output"):
        jit_from_cudnn_graph(g, **_kw("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma"))


# ---------------------------------------------------------------------------
# sm107 (64-byte-K MMA): the same dual block-scale SwiGLU chain
# ---------------------------------------------------------------------------


@requires_sm107
@pytest.mark.parametrize("combo", ["nvfp4", "mxfp4", "mxfp8"])
@pytest.mark.parametrize(
    "config_name",
    [
        "CONFIG_sm107_128x128x128_128x128x64_cluster1x1_1ctamma",
        "CONFIG_sm107_128x128x128_128x128x64_cluster2x1_2ctamma",
    ],
)
def test_sm107_dual_block_scale_matmul_numerics(combo, config_name):
    """Two parallel block-scale GEMMs sharing A + one epilogue. Both GEMMs land
    in TMEM alongside the per-operand SF words — the tighter budget the 576
    columns buy back."""
    _run(combo, config_name, 256, 128, 512)


@requires_sm107
@pytest.mark.parametrize("combo", ["nvfp4", "mxfp8"])
def test_sm107_dual_block_scale_matmul_swiglu_quant_epilogue(combo):
    test_dual_block_scale_matmul_swiglu_quant_epilogue(combo, "CONFIG_sm107_128x128x128_128x128x64_cluster1x1_1ctamma")
