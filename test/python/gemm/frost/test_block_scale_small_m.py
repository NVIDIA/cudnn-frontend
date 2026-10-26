# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Block-scale N tiles below 128 (swap-AB at decode row counts): a tile reads its
rows of the shared 128-row SFB block, by a TMEM column shift (64-wide) or a utccp
SMEM source shift (8/16/32, kernel N <= 32). Numerics, geometry gates, the
execute-time N guard and the small-M planning rule."""

from __future__ import annotations

import dataclasses

import cudnn
import cudnn.gemm.frost  # noqa: F401  (installs recorder)
import pytest
import torch

from gemm_test_utils import E2M1 as _E2M1, rand_e8m0 as _rand_e8m0, requires_sm100, to_blocked as _to_blocked, unpack_fp4 as _unpack_fp4
from test_block_scale_matmul import _build_nvfp4_graph, _splitk_workspace

from cudnn.gemm.frost.compiler import jit_from_cudnn_graph

from cudnn.gemm.frost.graph_analyzer import analyze
from cudnn.gemm.frost.planning import DeviceProperties, small_m_block_scale_config
from cudnn.gemm.frost.tile_config import BLOCK_SCALE_NARROW_MAX_N, by_name, validate_block_scale_config_sm100

pytestmark = pytest.mark.L0


def _narrow(width: int) -> str:
    return f"CONFIG_sm100_128x{width}x128_128x{width}x32_cluster1x1_1ctamma_swapAB"


def _graph(M, N, K, combo):
    """(graph, {role: tensor}) for C[M, N] = deq(A[M, K]) @ deq(B[K, N])."""
    bs = 16 if combo == "nvfp4" else 32
    dt = cudnn.data_type.FP4_E2M1 if combo == "nvfp4" else cudnn.data_type.FP8_E4M3
    sf_dt = cudnn.data_type.FP8_E4M3 if combo == "nvfp4" else cudnn.data_type.FP8_E8M0
    sf_m, sf_n, sf_k = -(-M // 128) * 128, -(-N // 128) * 128, -(-(K // bs) // 4) * 4
    g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    t = {
        "A": g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1], data_type=dt),
        "B": g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K], data_type=dt),
        "SFA": g.tensor(name="SFA", dim=[1, sf_m, sf_k], stride=[sf_m * sf_k, sf_k, 1], data_type=sf_dt, reordering_type=cudnn.tensor_reordering.F8_128x4),
        "SFB": g.tensor(name="SFB", dim=[1, sf_k, sf_n], stride=[sf_k * sf_n, 1, sf_k], data_type=sf_dt, reordering_type=cudnn.tensor_reordering.F8_128x4),
    }
    a = g.block_scale_dequantize(input=t["A"], descale=t["SFA"], block_size=[1, bs])
    b = g.block_scale_dequantize(input=t["B"], descale=t["SFB"], block_size=[bs, 1])
    t["C"] = g.matmul(A=a, B=b, name="mm")
    t["C"].set_output(True).set_data_type(cudnn.data_type.HALF)
    return g, t, bs


def _data(M, N, K, combo, bs):
    if combo == "nvfp4":
        lut = torch.tensor(_E2M1, dtype=torch.float32, device="cuda")
        a8 = torch.randint(0, 256, (1, M, K // 2), dtype=torch.uint8, device="cuda")
        b8 = torch.randint(0, 256, (1, N, K // 2), dtype=torch.uint8, device="cuda")
        a, b = a8.view(torch.float4_e2m1fn_x2), b8.view(torch.float4_e2m1fn_x2)
        a_deq, b_deq = _unpack_fp4(a8, lut).view(M, K), _unpack_fp4(b8, lut).view(N, K)
        sfa = torch.randint(1, 4, (M, K // bs), device="cuda").to(torch.float8_e4m3fn)
        sfb = torch.randint(1, 4, (N, K // bs), device="cuda").to(torch.float8_e4m3fn)
    else:
        a = (torch.randn(1, M, K, device="cuda") * 0.5).to(torch.float8_e4m3fn)
        b = (torch.randn(1, N, K, device="cuda") * 0.5).to(torch.float8_e4m3fn)
        a_deq, b_deq = a.float().view(M, K), b.float().view(N, K)
        sfa, sfb = _rand_e8m0((M, K // bs), "cuda"), _rand_e8m0((N, K // bs), "cuda")
    ref = (a_deq * sfa.float().repeat_interleave(bs, 1)) @ (b_deq * sfb.float().repeat_interleave(bs, 1)).t()
    return a, b, _to_blocked(sfa).view(1, -(-M // 128) * 128, -1), _to_blocked(sfb).view(1, -(-N // 128) * 128, -1), ref


def _run(combo, config_name, M, N, K, split_k=1, run_m=None):
    torch.manual_seed(0)
    g, t, bs = _graph(M, N, K, combo)
    compiled = jit_from_cudnn_graph(g, config=dataclasses.replace(by_name(config_name), split_k_slices=split_k))
    run_m = M if run_m is None else run_m
    a, b, sfa, sfb, ref = _data(run_m, N, K, combo, bs)
    c = torch.zeros(1, run_m, N, dtype=torch.float16, device="cuda")
    ws, _keep = _splitk_workspace(compiled)
    compiled({t["A"]: a, t["B"]: b, t["SFA"]: sfa, t["SFB"]: sfb, t["C"]: c}, workspace=ws)
    torch.cuda.synchronize()
    torch.testing.assert_close(c[0], ref.to(torch.float16), atol=2e-1, rtol=2e-2)


# (M tokens, N weight rows, K, tile width, split-K). M=24 at width 8 walks the
# SMEM row shift (offsets 8, 16); M=100 / 300 at width 64 walk the 0/2 TMEM
# column shift across several 128-row SFB blocks.
_CASES = [
    (8, 512, 2048, 8, 1),
    (8, 512, 2048, 8, 4),
    (24, 384, 1024, 8, 1),
    (24, 384, 1024, 16, 2),
    (32, 256, 2048, 32, 1),
    (100, 384, 1024, 64, 1),
    (300, 256, 1024, 64, 2),
]


@requires_sm100
@pytest.mark.parametrize("combo", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("M,N,K,width,split_k", _CASES)
def test_narrow_n_numerics(combo, M, N, K, width, split_k):
    _run(combo, _narrow(width), M, N, K, split_k=split_k)


@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_narrow_n_geometry_accepted_on_one_cta(width):
    validate_block_scale_config_sm100(by_name(_narrow(width)), 16, 256)


@pytest.mark.parametrize(
    "name",
    [
        "CONFIG_sm100_128x64x128_128x64x32_cluster1x2_1ctamma_swapAB",  # cluster N > 1
        "CONFIG_sm100_256x64x128_128x64x32_cluster2x1_2ctamma",  # 2-CTA MMA
        "CONFIG_sm100_128x48x128_128x48x32_cluster1x1_1ctamma",  # not a narrow width
    ],
)
def test_narrow_n_geometry_rejected(name):
    try:
        cfg = by_name(name)
    except (NotImplementedError, ValueError):
        return
    with pytest.raises(NotImplementedError, match="mma_tile_n"):
        validate_block_scale_config_sm100(cfg, 16, 256)


@requires_sm100
def test_narrow_n_compile_rejects_kernel_n_past_32():
    g, _, _ = _graph(48, 256, 1024, "nvfp4")
    with pytest.raises(NotImplementedError, match="kernel N extent"):
        jit_from_cudnn_graph(g, config=by_name(_narrow(16)))


@requires_sm100
def test_narrow_n_execute_refuses_a_wider_call():
    """A plan built for N <= 32 must refuse, not fault, on a larger runtime N."""
    with pytest.raises(ValueError, match=f"at most {BLOCK_SCALE_NARROW_MAX_N}"):
        _run("nvfp4", _narrow(16), 16, 256, 1024, run_m=48)


_B200 = DeviceProperties(100, "NVIDIA B200", 148, 126 * 1024 * 1024)


def _small_m(M, N, K, narrow_n_ok):
    chain = analyze(_build_nvfp4_graph(M, N, K))
    baseline = by_name("CONFIG_sm100_256x128x128_128x128x32_cluster2x1_2ctamma")
    return small_m_block_scale_config(chain, baseline, _B200, narrow_n_ok)


def test_small_m_rule_fixed_shapes_take_a_narrow_tile_and_one_wave():
    cfg = _small_m(8, 2560, 8192, narrow_n_ok=True)
    assert (cfg.cta_tile_n, cfg.swap_ab, cfg.split_k_slices) == (8, True, 4)  # 20 tiles x 4 <= 148
    assert _small_m(24, 4096, 8192, narrow_n_ok=True).cta_tile_n == 32


def test_small_m_rule_overridable_shapes_take_the_64_wide_tile():
    cfg = _small_m(8, 2560, 8192, narrow_n_ok=False)
    assert (cfg.cta_tile_n, cfg.swap_ab) == (64, True)
    # Past 48 weight tiles the 64-wide tile loses to the default plan.
    assert _small_m(8, 8192, 8192, narrow_n_ok=False) is None


def test_small_m_rule_leaves_larger_m_to_the_cost_model():
    assert _small_m(64, 2560, 8192, narrow_n_ok=True) is None


def test_small_m_rule_split_k_respects_k_tiles():
    cfg = _small_m(8, 256, 512, narrow_n_ok=True)  # K=512 nvfp4 is 2 K-tiles
    assert cfg.split_k_slices <= 2


def test_override_shape_graph_declines_narrow_knobs():
    from cudnn.gemm.frost.compiler import plan_config
    from cudnn.gemm.frost.knobs import GemmKnobs

    chain = analyze(_build_nvfp4_graph(8, 256, 1024))
    knobs = GemmKnobs.from_config(dataclasses.replace(by_name(_narrow(8))))
    with pytest.raises(NotImplementedError, match="fixed graph shapes"):
        plan_config(chain, knobs=knobs, narrow_n_ok=False)
    assert plan_config(chain, knobs=knobs, narrow_n_ok=True).cta_tile_n == 8
